"""CrossRegionManager — multi-region fallback with pre-flight checks.

Handles cross-region resilience for SageMaker Batch Transform jobs:
pre-flight checks (S3 mirror bucket, ECR image warning, IAM validation),
on-demand S3 data replication with SDK tags, cross-region job submission
via a strategy engine, and output retrieval back to the home region.

Interacts with jobs exclusively through the ``JobAdapter`` interface.
Creates a new ``JobAdapter`` (via ``adapter_factory``) for each region.

Requirements covered: 8.1–8.4, 9.1–9.8, 10.1–10.5, 11.1–11.4,
12.1–12.3.
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

import boto3

from sm_resilient_launcher._exceptions import (
    AllRegionsExhaustedError,
    AllTypesExhaustedError,
)

if TYPE_CHECKING:
    from sm_resilient_launcher._error_classifier import ClassifiedError
    from sm_resilient_launcher._job_adapter import JobAdapter
    from sm_resilient_launcher._launch_result import LaunchResult

logger = logging.getLogger(__name__)

# Published SageMaker algorithm account IDs per region.  Used by
# ``_auto_create_model`` to resolve the target-region container image
# when the source image is AWS-managed (account != caller account).
SAGEMAKER_ALGO_ACCOUNTS: dict[str, str] = {
    "us-east-1": "811284229777",
    "us-east-2": "825641698319",
    "us-west-1": "632365934929",
    "us-west-2": "433757028032",
    "eu-west-1": "685385470294",
    "eu-west-2": "644912444149",
    "eu-central-1": "813361260812",
    "ap-northeast-1": "501404015308",
    "ap-northeast-2": "306986355934",
    "ap-southeast-1": "475088953585",
    "ap-southeast-2": "544295431143",
    "ap-south-1": "991648021394",
    "ca-central-1": "469771592824",
    "eu-north-1": "669576153137",
    "sa-east-1": "855470959533",
}


class CrossRegionManager:
    """Manages cross-region fallback for capacity resilience.

    Iterates regions in the exact ``AllowedRegions`` order.  For each
    non-home region it runs pre-flight checks (S3 mirror bucket, ECR
    image, IAM role), replicates data, delegates to a strategy engine,
    and retrieves output back to the home region on success.

    All created resources are tagged with the SDK tag prefix.
    """

    TAG_PREFIX = "sm-resilient-launcher"

    def __init__(
        self,
        adapter_factory: Callable[[str], JobAdapter],
        strategy_engine_factory: Callable[[JobAdapter], object],
        home_region: str,
        allowed_regions: list[str],
        base_job_name: str,
        region_config: dict | None = None,
        auto_create_model: bool = False,
    ) -> None:
        self._adapter_factory = adapter_factory
        self._strategy_engine_factory = strategy_engine_factory
        self._home_region = home_region
        self._allowed_regions = allowed_regions
        self._base_job_name = base_job_name
        self._region_config = region_config or {}
        self._auto_create_model = auto_create_model
        self._warnings: list[str] = []
        self._created_models: dict[str, str] = {}  # region → model name

    # ── Public entry point ─────────────────────────────────────

    def execute(
        self,
        instance_types: list[str],
        boto3_params: dict,
        base_job_name: str,
    ) -> LaunchResult:
        """Try each region in order until a launch succeeds.

        For non-home regions: runs pre-flight checks, replicates
        data, then delegates to the strategy engine.

        Args:
            instance_types: Ordered list — never reordered.
            boto3_params: The boto3 params (SDK params already stripped).
            base_job_name: Base name; each attempt appends a suffix.

        Returns:
            LaunchResult (with ``data_replicated=True`` and ``warnings``
            if cross-region was triggered).

        Raises:
            AllRegionsExhaustedError: When all regions fail.
        """
        all_errors: list[ClassifiedError] = []
        globally_skipped_types: set[str] = set()

        for region in self._allowed_regions:
            is_cross_region = region != self._home_region

            if is_cross_region:
                preflight_ok = self._preflight_check(
                    region, boto3_params
                )
                if not preflight_ok:
                    logger.info(
                        "Pre-flight failed for %s; skipping region",
                        region,
                    )
                    continue

                # Req 9.4, 9.5: If S3 copy fails, skip this region.
                try:
                    self._replicate_data(region, boto3_params)
                except Exception as exc:
                    logger.warning(
                        "Data replication to %s failed: %s; "
                        "skipping region",
                        region,
                        exc,
                    )
                    continue

            adapter = self._adapter_factory(region)
            engine = self._strategy_engine_factory(adapter)

            try:
                # Determine the boto3 params for this region.
                region_cfg = self._region_config.get(region, {})
                if is_cross_region:
                    # Always rewrite S3 URIs for cross-region (input/output need mirror bucket)
                    region_params = self._rewrite_s3_uris(
                        boto3_params, region
                    )
                    # Additionally substitute ModelName if RegionConfig provides one
                    if region_cfg.get("ModelName"):
                        region_params["ModelName"] = region_cfg[
                            "ModelName"
                        ]
                    # Or use auto-created model name
                    elif region in self._created_models:
                        region_params["ModelName"] = (
                            self._created_models[region]
                        )
                else:
                    region_params = boto3_params

                result: LaunchResult = engine.execute(
                    instance_types=instance_types,
                    boto3_params=region_params,
                    base_job_name=base_job_name,
                    region=region,
                    globally_skipped_types=globally_skipped_types,
                )
                result.data_replicated = is_cross_region
                result.warnings = list(self._warnings)

                if is_cross_region:
                    self._retrieve_output(region, boto3_params)

                return result

            except AllTypesExhaustedError as exc:
                all_errors.extend(exc.errors)
                logger.info(
                    "All types exhausted in %s; trying next region",
                    region,
                )

        raise AllRegionsExhaustedError(all_errors)

    # ── Pre-flight checks ──────────────────────────────────────

    def _preflight_check(
        self, region: str, boto3_params: dict
    ) -> bool:
        """Run S3, ECR, and IAM pre-flight checks for a target region.

        Returns True if the region passes.

        Model resolution priority:
        1. RegionConfig provides a ModelName → trust it, skip ECR check.
        2. auto_create_model=True → attempt to create model in target
           region.  On success the created model name is cached in
           ``_created_models``; on failure the region is skipped.
        3. Neither → legacy ECR check; missing image → skip region.

        IAM warnings are always non-blocking.
        """
        logger.info("Pre-flight checks for %s", region)

        # 1. S3 mirror bucket
        self._ensure_mirror_bucket(region, boto3_params)

        # 2. Model resolution
        region_cfg = self._region_config.get(region, {})
        if region_cfg.get("ModelName"):
            # Explicit RegionConfig — trust the customer's setup.
            logger.info(
                "Using RegionConfig model %s for %s; skipping ECR check",
                region_cfg["ModelName"],
                region,
            )
        elif self._auto_create_model:
            # Attempt auto-creation of the model in the target region.
            created_name = self._auto_create_model_in_region(
                region, boto3_params
            )
            if created_name is None:
                return False
            # Cache for use during execution and to avoid re-creation.
            self._created_models[region] = created_name
        else:
            # Legacy ECR image check — blocks the region if missing.
            ecr_ok = self._check_ecr_image(region, boto3_params)
            if not ecr_ok:
                return False

        # 3. IAM validation — non-blocking (Req 11.1)
        self._validate_iam(region, boto3_params)

        return True

    def _auto_create_model_in_region(
        self, region: str, boto3_params: dict
    ) -> str | None:
        """Auto-create a SageMaker Model in the target region.

        Steps:
        1. Describe the home-region model to extract image URI,
           artifact path, execution role, and environment vars.
        2. Resolve the target-region container image URI:
           - Customer-owned ECR (image account == caller account):
             replace the region in the URI, verify image exists.
           - AWS-managed image (image account != caller account):
             use ``SAGEMAKER_ALGO_ACCOUNTS`` to map the account,
             or fall back to a region-string replacement.
        3. Create the model in the target region with SDK tags.

        Returns the created model name, or ``None`` if creation
        failed (caller should skip the region).
        """
        # Return cached model if already created for this region.
        if region in self._created_models:
            return self._created_models[region]

        model_name = boto3_params.get("ModelName", "")
        if not model_name:
            logger.warning(
                "AutoCreateModel: no ModelName in boto3_params; "
                "skipping region %s",
                region,
            )
            return None

        # 1. Describe the home-region model.
        try:
            sm_home = boto3.client(
                "sagemaker", region_name=self._home_region
            )
            model_desc = sm_home.describe_model(ModelName=model_name)
        except Exception as exc:
            warning_msg = (
                f"AutoCreateModel: could not describe model "
                f"{model_name} in {self._home_region}: {exc}"
            )
            logger.warning(warning_msg)
            self._warnings.append(warning_msg)
            return None

        container = model_desc.get("PrimaryContainer", {})
        image_uri = container.get("Image", "")
        model_data_url = container.get("ModelDataUrl", "")
        execution_role = model_desc.get("ExecutionRoleArn", "")
        environment = container.get("Environment", {})

        if not image_uri:
            warning_msg = (
                f"AutoCreateModel: model {model_name} has no "
                f"container image; skipping region {region}"
            )
            logger.warning(warning_msg)
            self._warnings.append(warning_msg)
            return None

        # 2. Resolve target-region image URI.
        target_image_uri = self._resolve_target_image_uri(
            image_uri, region
        )
        if target_image_uri is None:
            return None

        # 3. Rewrite model artifact S3 URI to mirror bucket.
        target_model_data_url = model_data_url
        if model_data_url:
            bucket, prefix = self._parse_s3_uri(model_data_url)
            if bucket:
                mirror_bucket = f"{bucket}-{region}"
                target_model_data_url = (
                    f"s3://{mirror_bucket}/{prefix}"
                )

        # 4. Create model in target region.
        target_model_name = f"{model_name}-{region}"
        try:
            sm_target = boto3.client(
                "sagemaker", region_name=region
            )
            tags = self._build_sdk_tags()
            sm_tags = [
                {"Key": t["Key"], "Value": t["Value"]}
                for t in tags
            ]

            create_params: dict = {
                "ModelName": target_model_name,
                "PrimaryContainer": {
                    "Image": target_image_uri,
                    "ModelDataUrl": target_model_data_url,
                },
                "ExecutionRoleArn": execution_role,
                "Tags": sm_tags,
            }

            # Preserve environment variables if any.
            if environment:
                create_params["PrimaryContainer"][
                    "Environment"
                ] = environment

            sm_target.create_model(**create_params)
            logger.info(
                "AutoCreateModel: created model %s in %s "
                "(image: %s)",
                target_model_name,
                region,
                target_image_uri,
            )
            return target_model_name

        except Exception as exc:
            warning_msg = (
                f"AutoCreateModel: failed to create model "
                f"{target_model_name} in {region}: {exc}"
            )
            logger.warning(warning_msg)
            self._warnings.append(warning_msg)
            return None

    def _resolve_target_image_uri(
        self, image_uri: str, target_region: str
    ) -> str | None:
        """Resolve the container image URI for the target region.

        For customer-owned ECR images (image account == caller
        account), replaces the region in the URI and verifies
        the image exists.

        For AWS-managed images (image account != caller account),
        uses ``SAGEMAKER_ALGO_ACCOUNTS`` to map to the correct
        regional account, or falls back to region-string
        replacement.

        Returns the resolved URI, or ``None`` if the image
        cannot be resolved (with a warning appended).
        """
        if ".ecr." not in image_uri:
            # Not an ECR image (e.g., Docker Hub).  Not supported
            # for cross-region auto-creation.
            warning_msg = (
                f"AutoCreateModel: image {image_uri} is not an "
                f"ECR image; cannot auto-resolve for {target_region}"
            )
            logger.warning(warning_msg)
            self._warnings.append(warning_msg)
            return None

        source_account = self._parse_ecr_account(image_uri)
        repo_name = self._parse_ecr_repo(image_uri)
        tag = self._parse_ecr_tag(image_uri)

        # Determine caller account via STS.
        try:
            sts = boto3.client("sts")
            caller_account = sts.get_caller_identity()["Account"]
        except Exception:
            caller_account = ""

        is_customer_owned = (
            caller_account and source_account == caller_account
        )

        if is_customer_owned:
            # Replace region in the URI.
            target_uri = image_uri.replace(
                f".ecr.{self._home_region}.",
                f".ecr.{target_region}.",
            )
            # Verify image exists in target ECR.
            try:
                ecr = boto3.client("ecr", region_name=target_region)
                ecr.describe_images(
                    repositoryName=repo_name,
                    imageIds=[{"imageTag": tag}],
                )
                return target_uri
            except Exception:
                # Image doesn't exist — warn with CLI commands.
                commands = (
                    f"# Replicate ECR image to {target_region}:\n"
                    f"aws ecr get-login-password --region "
                    f"{self._home_region} | docker login "
                    f"--username AWS --password-stdin "
                    f"{source_account}.dkr.ecr."
                    f"{self._home_region}.amazonaws.com\n"
                    f"docker pull {image_uri}\n"
                    f"docker tag {image_uri} "
                    f"{source_account}.dkr.ecr."
                    f"{target_region}.amazonaws.com/"
                    f"{repo_name}:{tag}\n"
                    f"aws ecr get-login-password --region "
                    f"{target_region} | docker login "
                    f"--username AWS --password-stdin "
                    f"{source_account}.dkr.ecr."
                    f"{target_region}.amazonaws.com\n"
                    f"aws ecr create-repository "
                    f"--repository-name {repo_name} "
                    f"--region {target_region} || true\n"
                    f"docker push {source_account}.dkr.ecr."
                    f"{target_region}.amazonaws.com/"
                    f"{repo_name}:{tag}"
                )
                warning_msg = (
                    f"AutoCreateModel: customer ECR image not "
                    f"found in {target_region}. Run:\n{commands}"
                )
                logger.warning(warning_msg)
                self._warnings.append(warning_msg)
                return None
        else:
            # AWS-managed image — resolve target account ID.
            target_account = SAGEMAKER_ALGO_ACCOUNTS.get(
                target_region
            )
            if target_account:
                target_uri = (
                    f"{target_account}.dkr.ecr.{target_region}"
                    f".amazonaws.com/{repo_name}:{tag}"
                )
            else:
                # Fallback: replace region string in URI.
                target_uri = image_uri.replace(
                    f".ecr.{self._home_region}.",
                    f".ecr.{target_region}.",
                )
            return target_uri

    def _ensure_mirror_bucket(
        self, region: str, boto3_params: dict
    ) -> None:
        """Check/create mirror S3 bucket with SDK tags.

        Bucket naming: ``{original-bucket}-{target-region}``.

        Req 9.1: Check if mirror bucket exists.
        Req 9.2: Create if it doesn't.
        Req 9.3: Apply all four SDK tags at creation time.
        Req 9.8: Only log actual operations, not no-ops.
        """
        source_bucket = self._extract_bucket_name(
            self._resolve_input_s3_uri(boto3_params)
        )
        if not source_bucket:
            return

        mirror_bucket = f"{source_bucket}-{region}"
        s3 = boto3.client("s3", region_name=region)

        try:
            s3.head_bucket(Bucket=mirror_bucket)
            # Bucket exists — no log per Req 9.8 (no-op).
        except Exception:
            logger.info("Creating mirror bucket %s", mirror_bucket)
            create_kwargs: dict = {"Bucket": mirror_bucket}
            # us-east-1 does not accept LocationConstraint.
            if region != "us-east-1":
                create_kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": region
                }
            s3.create_bucket(**create_kwargs)

            # Req 9.3 / 12.2: Apply SDK tags at creation time.
            tags = self._build_sdk_tags()
            s3.put_bucket_tagging(
                Bucket=mirror_bucket,
                Tagging={"TagSet": tags},
            )
            logger.info(
                "Created and tagged mirror bucket %s", mirror_bucket
            )

    def _check_ecr_image(
        self, region: str, boto3_params: dict
    ) -> bool:
        """Check if container image exists in target region's ECR.

        If missing: emit WARNING log with CLI commands, add to
        ``warnings``, and return ``False`` (skip region).

        If ``RegionConfig`` provides a ``ModelName`` for this region,
        the check is skipped entirely — the customer configured the
        model and we trust their setup.

        Req 10.1–10.5: Warn, never replicate ECR automatically.
        """
        # If RegionConfig provides a model for this region, skip ECR check.
        region_cfg = self._region_config.get(region, {})
        if region_cfg.get("ModelName"):
            model_name = region_cfg["ModelName"]
            logger.info(
                "Using RegionConfig model %s for %s; skipping ECR check",
                model_name,
                region,
            )
            return True
        model_name = boto3_params.get("ModelName", "")
        # Resolve the image URI from the model description.
        try:
            sm_home = boto3.client(
                "sagemaker", region_name=self._home_region
            )
            model_desc = sm_home.describe_model(ModelName=model_name)
            image_uri = (
                model_desc.get("PrimaryContainer", {}).get("Image", "")
            )
        except Exception:
            logger.warning(
                "Could not resolve model image URI for %s", model_name
            )
            return True  # Proceed if we can't check.

        if not image_uri or ".ecr." not in image_uri:
            return True  # Not an ECR image, skip check.

        # Check if image exists in target region's ECR.
        repo_name = self._parse_ecr_repo(image_uri)
        tag = self._parse_ecr_tag(image_uri)
        try:
            ecr_target = boto3.client("ecr", region_name=region)
            ecr_target.describe_images(
                repositoryName=repo_name,
                imageIds=[{"imageTag": tag}],
            )
            return True
        except Exception:
            # Image not found — emit warning with exact CLI commands.
            source_account = self._parse_ecr_account(image_uri)
            cli_commands = (
                f"# Replicate ECR image to {region}:\n"
                f"aws ecr get-login-password --region "
                f"{self._home_region} | docker login --username AWS "
                f"--password-stdin {source_account}.dkr.ecr."
                f"{self._home_region}.amazonaws.com\n"
                f"docker pull {image_uri}\n"
                f"docker tag {image_uri} {source_account}.dkr.ecr."
                f"{region}.amazonaws.com/{repo_name}:{tag}\n"
                f"aws ecr get-login-password --region {region} | "
                f"docker login --username AWS --password-stdin "
                f"{source_account}.dkr.ecr.{region}.amazonaws.com\n"
                f"aws ecr create-repository --repository-name "
                f"{repo_name} --region {region} || true\n"
                f"docker push {source_account}.dkr.ecr.{region}"
                f".amazonaws.com/{repo_name}:{tag}"
            )
            warning_msg = (
                f"ECR image {image_uri} not found in {region}. "
                f"Run the following to replicate:\n{cli_commands}"
            )
            logger.warning(warning_msg)
            self._warnings.append(warning_msg)
            return False

    def _validate_iam(
        self, region: str, boto3_params: dict
    ) -> None:
        """Validate execution role exists and warn about policy gaps.

        IAM roles are global, but policies may not cover target
        region mirror buckets.

        Req 11.1–11.4: Validate, warn, never modify IAM.
        Non-blocking — job submission proceeds regardless.
        """
        role_arn = boto3_params.get("TransformResources", {}).get(
            "RoleArn"
        )
        if not role_arn:
            return

        iam = boto3.client("iam")
        role_name = role_arn.split("/")[-1]

        try:
            iam.get_role(RoleName=role_name)
        except Exception as exc:
            warning_msg = (
                f"Execution role {role_arn} could not be validated: "
                f"{exc}"
            )
            logger.warning(warning_msg)
            self._warnings.append(warning_msg)
            return

        # Warn about potential policy gap for the mirror bucket.
        source_bucket = self._extract_bucket_name(
            self._resolve_input_s3_uri(boto3_params)
        )
        if not source_bucket:
            return

        mirror_bucket = f"{source_bucket}-{region}"

        try:
            # List attached and inline policies for heuristic check.
            policies = iam.list_attached_role_policies(
                RoleName=role_name
            )
            policy_names = [
                p["PolicyName"]
                for p in policies.get("AttachedPolicies", [])
            ]
            inline = iam.list_role_policies(RoleName=role_name)
            policy_names.extend(inline.get("PolicyNames", []))

            # Heuristic: warn unconditionally — full evaluation would
            # require the IAM policy simulator, which is out of scope.
            warning_msg = (
                f"Execution role {role_arn} may not have policies "
                f"covering mirror bucket s3://{mirror_bucket} in "
                f"{region}. Verify that the role's S3 policies "
                f"include arn:aws:s3:::{mirror_bucket}/* or use "
                f"a wildcard bucket policy."
            )
            logger.warning(warning_msg)
            self._warnings.append(warning_msg)

        except Exception as exc:
            logger.warning(
                "IAM policy check failed for %s: %s", role_name, exc
            )

    # ── Data replication ───────────────────────────────────────

    def _replicate_data(
        self, target_region: str, boto3_params: dict
    ) -> None:
        """Copy model artifact and input data to the target region.

        All copied objects are tagged with SDK tags.

        When ``RegionConfig`` provides a ``ModelName`` for the target
        region, model artifact replication is skipped — the customer's
        model already points to the right artifact.  Input data is
        always replicated.

        Req 9.4: Copy model artifact.
        Req 9.5: Copy input data prefix.
        Req 9.6: Apply SDK tags on each copied object.
        Req 9.8: Log every copy operation actually performed.
        """
        logger.info("Replicating data to %s", target_region)

        # Skip model artifact replication if RegionConfig provides
        # a model for this region.
        region_cfg = self._region_config.get(target_region, {})
        if not region_cfg.get("ModelName"):
            model_uri = self._resolve_model_s3_uri(boto3_params)
            if model_uri:
                self._copy_s3_prefix(
                    self._home_region, target_region, model_uri
                )

        input_uri = self._resolve_input_s3_uri(boto3_params)
        if input_uri:
            self._copy_s3_prefix(
                self._home_region, target_region, input_uri
            )

    def _retrieve_output(
        self, source_region: str, boto3_params: dict
    ) -> None:
        """Copy job output back to the home region.

        Req 9.7: Retrieve output after cross-region completion.
        """
        output_uri = self._resolve_output_s3_uri(boto3_params)
        if output_uri:
            logger.info(
                "Retrieving output from %s to %s",
                source_region,
                self._home_region,
            )
            self._copy_s3_prefix(
                source_region, self._home_region, output_uri
            )

    def _copy_s3_prefix(
        self, src_region: str, dst_region: str, s3_uri: str
    ) -> None:
        """Copy S3 objects between regions with SDK tags.

        Tags are applied at copy time, not retroactively (Req 12.2).
        Logs every object copy for audit (Req 9.8).
        """
        src_bucket, prefix = self._parse_s3_uri(s3_uri)
        dst_bucket = f"{src_bucket}-{dst_region}"

        s3_src = boto3.client("s3", region_name=src_region)
        s3_dst = boto3.client("s3", region_name=dst_region)

        paginator = s3_src.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=src_bucket, Prefix=prefix
        ):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                tag_string = self._build_sdk_tag_string()
                s3_dst.copy_object(
                    CopySource={"Bucket": src_bucket, "Key": key},
                    Bucket=dst_bucket,
                    Key=key,
                    Tagging=tag_string,
                    TaggingDirective="REPLACE",
                )
                logger.info(
                    "Copied s3://%s/%s → s3://%s/%s",
                    src_bucket,
                    key,
                    dst_bucket,
                    key,
                )

    # ── S3 URI rewriting ───────────────────────────────────────

    def _rewrite_s3_uris(
        self, boto3_params: dict, target_region: str
    ) -> dict:
        """Rewrite S3 URIs in boto3_params to point to mirror buckets.

        Returns a deep copy with input and output URIs rewritten.
        The original dict is never mutated.
        """
        params = copy.deepcopy(boto3_params)

        # Rewrite input S3 URI.
        input_uri = self._resolve_input_s3_uri(params)
        if input_uri:
            bucket, prefix = self._parse_s3_uri(input_uri)
            mirror_bucket = f"{bucket}-{target_region}"
            new_uri = f"s3://{mirror_bucket}/{prefix}"
            params["TransformInput"]["DataSource"]["S3DataSource"][
                "S3Uri"
            ] = new_uri

        # Rewrite output S3 URI.
        output_uri = self._resolve_output_s3_uri(params)
        if output_uri:
            bucket, prefix = self._parse_s3_uri(output_uri)
            mirror_bucket = f"{bucket}-{target_region}"
            params["TransformOutput"][
                "S3OutputPath"
            ] = f"s3://{mirror_bucket}/{prefix}"

        return params

    # ── SDK tag helpers ────────────────────────────────────────

    def _build_sdk_tags(self) -> list[dict[str, str]]:
        """Build SDK tag list for S3 bucket tagging API.

        Four tags per Req 12.1:
          - sm-resilient-launcher:managed = true
          - sm-resilient-launcher:source-region = <home_region>
          - sm-resilient-launcher:job-name = <base_job_name>
          - sm-resilient-launcher:created-at = <ISO 8601>
        """
        now = datetime.now(timezone.utc).isoformat()
        return [
            {"Key": f"{self.TAG_PREFIX}:managed", "Value": "true"},
            {
                "Key": f"{self.TAG_PREFIX}:source-region",
                "Value": self._home_region,
            },
            {
                "Key": f"{self.TAG_PREFIX}:job-name",
                "Value": self._base_job_name,
            },
            {
                "Key": f"{self.TAG_PREFIX}:created-at",
                "Value": now,
            },
        ]

    def _build_sdk_tag_string(self) -> str:
        """Build URL-encoded tag string for S3 ``copy_object`` Tagging.

        Same four tags as ``_build_sdk_tags()``, serialised as
        ``key=value&key=value`` for the ``Tagging`` parameter.
        """
        now = datetime.now(timezone.utc).isoformat()
        pairs = [
            f"{self.TAG_PREFIX}:managed=true",
            f"{self.TAG_PREFIX}:source-region={self._home_region}",
            f"{self.TAG_PREFIX}:job-name={self._base_job_name}",
            f"{self.TAG_PREFIX}:created-at={now}",
        ]
        return "&".join(pairs)

    # ── URI resolution helpers ─────────────────────────────────

    @staticmethod
    def _resolve_model_s3_uri(boto3_params: dict) -> str:
        """Extract model artifact S3 URI.

        The model S3 URI is resolved from ``ModelName`` — the actual
        artifact location would come from ``DescribeModel``, but here
        we rely on the param as a placeholder reference.
        """
        return boto3_params.get("ModelName", "")

    @staticmethod
    def _resolve_input_s3_uri(boto3_params: dict) -> str:
        """Extract the input data S3 URI from ``TransformInput``."""
        return (
            boto3_params.get("TransformInput", {})
            .get("DataSource", {})
            .get("S3DataSource", {})
            .get("S3Uri", "")
        )

    @staticmethod
    def _resolve_output_s3_uri(boto3_params: dict) -> str:
        """Extract the output S3 path from ``TransformOutput``."""
        return boto3_params.get("TransformOutput", {}).get(
            "S3OutputPath", ""
        )

    # ── S3 / ECR URI parsing helpers ───────────────────────────

    @staticmethod
    def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
        """Parse ``s3://bucket/prefix`` into ``(bucket, prefix)``.

        Returns:
            A (bucket, prefix) tuple. ``prefix`` is empty string when
            only a bucket is provided.
        """
        path = s3_uri.replace("s3://", "")
        parts = path.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return bucket, prefix

    @staticmethod
    def _extract_bucket_name(s3_uri: str) -> str:
        """Extract just the bucket name from an S3 URI."""
        bucket, _ = CrossRegionManager._parse_s3_uri(s3_uri)
        return bucket

    @staticmethod
    def _parse_ecr_repo(image_uri: str) -> str:
        """Extract repo name from an ECR image URI.

        Format: ``{account}.dkr.ecr.{region}.amazonaws.com/{repo}:{tag}``
        """
        path = image_uri.split("/", 1)[-1]
        return path.split(":")[0]

    @staticmethod
    def _parse_ecr_tag(image_uri: str) -> str:
        """Extract tag from an ECR image URI; defaults to ``latest``."""
        if ":" in image_uri.split("/")[-1]:
            return image_uri.split(":")[-1]
        return "latest"

    @staticmethod
    def _parse_ecr_account(image_uri: str) -> str:
        """Extract the AWS account ID from an ECR image URI."""
        return image_uri.split(".")[0]
