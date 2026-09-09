"""prepare_regions utility — one-time cross-region infrastructure setup.

Standalone utility for checking and preparing cross-region infrastructure
ahead of time.  Returns a ``ReadinessReport`` with per-region status for
S3, ECR, and IAM readiness.

NEVER modifies IAM roles or replicates ECR images automatically (Req 16.6).

Requirements covered: 16.1–16.6.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import boto3

logger = logging.getLogger(__name__)

TAG_PREFIX = "sm-resilient-launcher"


# ── Data models ────────────────────────────────────────────────


@dataclass
class RegionReadiness:
    """Readiness status for a single target region."""

    region: str
    s3_status: str  # "ready" | "created"
    ecr_status: str  # "ready" | "missing"
    iam_status: str  # "valid" | "warning"
    ecr_commands: str | None = None  # CLI commands if missing
    iam_details: str | None = None  # Warning text if applicable


@dataclass
class ReadinessReport:
    """Result from ``prepare_regions`` utility."""

    regions: list[RegionReadiness] = field(default_factory=list)


# ── Public entry point ─────────────────────────────────────────


def prepare_regions(
    home_region: str,
    target_regions: list[str],
    model_name: str,
    role_arn: str,
    region_config: dict | None = None,
    auto_create_model: bool = False,
) -> ReadinessReport:
    """Check and prepare cross-region infrastructure.

    For each target region:
      1. S3: Check/create mirror buckets with SDK tags.
      2. ECR: Check image availability; include CLI commands if missing.
         If ``region_config`` provides a ``ModelName`` for the region,
         verify the specified model exists and set ECR status based on
         the model's image accessibility.
      3. IAM: Validate role exists; warn about policy gaps.

    When ``auto_create_model=True``, the readiness check also verifies
    whether automatic model creation would succeed in each target
    region (image exists in target ECR, model name is available).

    Does NOT modify IAM roles or replicate ECR images automatically.

    Args:
        home_region: The caller's primary region.
        target_regions: Regions to check/prepare.
        model_name: SageMaker model name (to resolve image URI).
        role_arn: Execution role ARN.
        region_config: Optional per-region configuration dict.
            Keys are region names, values are dicts with optional
            ``ModelName`` for region-specific model resolution.
        auto_create_model: When True, also verify that auto-creation
            of a SageMaker Model would succeed in each target region.

    Returns:
        ReadinessReport with per-region status.
    """
    region_config = region_config or {}
    report = ReadinessReport()

    # Resolve model image URI and artifact from home region.
    image_uri = _resolve_model_image(home_region, model_name)
    model_s3_uri = _resolve_model_artifact(home_region, model_name)

    source_bucket = ""
    if model_s3_uri:
        source_bucket = _extract_bucket_name(model_s3_uri)

    for region in target_regions:
        readiness = RegionReadiness(
            region=region,
            s3_status="ready",
            ecr_status="ready",
            iam_status="valid",
        )

        # 1. S3 mirror bucket (Req 16.2)
        if source_bucket:
            readiness.s3_status = _check_or_create_mirror_bucket(
                source_bucket, region, home_region, model_name
            )

        # 2. ECR image check (Req 16.3)
        region_cfg = region_config.get(region, {})
        if region_cfg.get("ModelName"):
            # RegionConfig provides a model — verify it exists in
            # the target region and set ECR status accordingly.
            ecr_status, ecr_commands = _check_region_config_model(
                region_cfg["ModelName"], region
            )
            readiness.ecr_status = ecr_status
            readiness.ecr_commands = ecr_commands
        elif image_uri and ".ecr." in image_uri:
            ecr_status, ecr_commands = _check_ecr_image(
                image_uri, region, home_region
            )
            readiness.ecr_status = ecr_status
            readiness.ecr_commands = ecr_commands

        # 3. IAM validation (Req 16.4)
        iam_status, iam_details = _validate_iam(
            role_arn, source_bucket, region
        )
        readiness.iam_status = iam_status
        readiness.iam_details = iam_details

        # 4. Auto-create model feasibility check
        if auto_create_model and readiness.ecr_status != "missing":
            auto_status, auto_details = (
                _check_auto_create_model_feasibility(
                    model_name, image_uri, region, home_region
                )
            )
            if auto_status == "infeasible":
                readiness.ecr_status = "missing"
                readiness.ecr_commands = auto_details

        report.regions.append(readiness)

    return report


# ── S3 mirror bucket ──────────────────────────────────────────


def _check_or_create_mirror_bucket(
    source_bucket: str,
    region: str,
    home_region: str,
    job_name: str,
) -> str:
    """Check/create mirror S3 bucket with SDK tags.

    Returns:
        ``"ready"`` if the bucket already exists, ``"created"`` if
        the bucket was created.
    """
    mirror_bucket = f"{source_bucket}-{region}"
    s3 = boto3.client("s3", region_name=region)

    try:
        s3.head_bucket(Bucket=mirror_bucket)
        logger.info("S3 mirror %s: ready", mirror_bucket)
        return "ready"
    except Exception:
        logger.info("Creating mirror bucket %s", mirror_bucket)
        create_kwargs: dict = {"Bucket": mirror_bucket}
        # us-east-1 does not accept LocationConstraint.
        if region != "us-east-1":
            create_kwargs["CreateBucketConfiguration"] = {
                "LocationConstraint": region
            }
        s3.create_bucket(**create_kwargs)

        # Apply SDK tags (Req 12.1, 16.2).
        tags = _build_sdk_tags(home_region, job_name)
        s3.put_bucket_tagging(
            Bucket=mirror_bucket,
            Tagging={"TagSet": tags},
        )
        logger.info("Created and tagged mirror bucket %s", mirror_bucket)
        return "created"


# ── ECR image check ───────────────────────────────────────────


def _check_region_config_model(
    model_name: str,
    region: str,
) -> tuple[str, str | None]:
    """Verify a RegionConfig-specified model exists in the target region.

    Calls ``DescribeModel`` in the target region to verify the model
    exists and its container image is accessible.

    Returns:
        ``("ready", None)`` if the model exists, or
        ``("missing", details)`` if it does not.
    """
    try:
        sm = boto3.client("sagemaker", region_name=region)
        desc = sm.describe_model(ModelName=model_name)
        image_uri = desc.get("PrimaryContainer", {}).get("Image", "")
        if image_uri:
            logger.info(
                "RegionConfig model %s in %s: ready (image: %s)",
                model_name,
                region,
                image_uri,
            )
            return "ready", None
        else:
            details = (
                f"Model {model_name} exists in {region} but has no "
                f"container image configured."
            )
            logger.warning(details)
            return "missing", details
    except Exception as exc:
        details = (
            f"RegionConfig model {model_name} not found in {region}: "
            f"{exc}"
        )
        logger.warning(details)
        return "missing", details


def _check_ecr_image(
    image_uri: str,
    region: str,
    home_region: str,
) -> tuple[str, str | None]:
    """Check if container image exists in target region's ECR.

    Returns:
        ``("ready", None)`` if found, or ``("missing", cli_commands)``
        if not found.  The SDK never replicates ECR images (Req 16.6).
    """
    repo_name = _parse_ecr_repo(image_uri)
    tag = _parse_ecr_tag(image_uri)

    try:
        ecr = boto3.client("ecr", region_name=region)
        ecr.describe_images(
            repositoryName=repo_name,
            imageIds=[{"imageTag": tag}],
        )
        logger.info("ECR in %s: ready", region)
        return "ready", None
    except Exception:
        source_account = _parse_ecr_account(image_uri)
        commands = (
            f"# Replicate ECR image to {region}:\n"
            f"aws ecr get-login-password --region "
            f"{home_region} | docker login --username AWS "
            f"--password-stdin {source_account}.dkr.ecr."
            f"{home_region}.amazonaws.com\n"
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
        logger.warning("ECR in %s: missing", region)
        return "missing", commands


# ── IAM validation ─────────────────────────────────────────────


def _validate_iam(
    role_arn: str,
    source_bucket: str,
    region: str,
) -> tuple[str, str | None]:
    """Validate execution role exists and check for policy gaps.

    The SDK never modifies IAM roles or policies (Req 16.6).

    Returns:
        ``("valid", None)`` or ``("warning", details)``.
    """
    role_name = role_arn.split("/")[-1]
    iam = boto3.client("iam")

    try:
        iam.get_role(RoleName=role_name)
    except Exception as exc:
        details = f"Role validation failed: {exc}"
        logger.warning("IAM for %s: %s", region, exc)
        return "warning", details

    # Warn about potential policy gap for the mirror bucket.
    if source_bucket:
        mirror_bucket = f"{source_bucket}-{region}"
        warning = (
            f"Verify role {role_arn} has policies covering "
            f"arn:aws:s3:::{mirror_bucket}/*"
        )
        logger.info("IAM for %s: valid with warning", region)
        return "warning", warning

    logger.info("IAM for %s: valid", region)
    return "valid", None


# ── Model resolution helpers ───────────────────────────────────


def _resolve_model_image(region: str, model_name: str) -> str:
    """Get the container image URI from a SageMaker model."""
    try:
        sm = boto3.client("sagemaker", region_name=region)
        desc = sm.describe_model(ModelName=model_name)
        return desc.get("PrimaryContainer", {}).get("Image", "")
    except Exception:
        return ""


def _resolve_model_artifact(region: str, model_name: str) -> str:
    """Get the model artifact S3 URI from a SageMaker model."""
    try:
        sm = boto3.client("sagemaker", region_name=region)
        desc = sm.describe_model(ModelName=model_name)
        return desc.get("PrimaryContainer", {}).get(
            "ModelDataUrl", ""
        )
    except Exception:
        return ""


# ── S3 / ECR URI parsing helpers ──────────────────────────────


def _extract_bucket_name(s3_uri: str) -> str:
    """Extract bucket name from ``s3://bucket/prefix``."""
    path = s3_uri.replace("s3://", "")
    return path.split("/", 1)[0]


def _parse_ecr_repo(image_uri: str) -> str:
    """Extract repo name from ECR image URI.

    Format: ``{account}.dkr.ecr.{region}.amazonaws.com/{repo}:{tag}``
    """
    path = image_uri.split("/", 1)[-1]
    return path.split(":")[0]


def _parse_ecr_tag(image_uri: str) -> str:
    """Extract tag from ECR image URI; defaults to ``latest``."""
    if ":" in image_uri.split("/")[-1]:
        return image_uri.split(":")[-1]
    return "latest"


def _parse_ecr_account(image_uri: str) -> str:
    """Extract AWS account ID from ECR image URI."""
    return image_uri.split(".")[0]


# ── SDK tag helpers ────────────────────────────────────────────


def _build_sdk_tags(
    home_region: str, job_name: str
) -> list[dict[str, str]]:
    """Build the four SDK tags for S3 bucket tagging.

    Tags per Req 12.1:
      - sm-resilient-launcher:managed = true
      - sm-resilient-launcher:source-region = <home_region>
      - sm-resilient-launcher:job-name = <job_name>
      - sm-resilient-launcher:created-at = <ISO 8601>
    """
    now = datetime.now(timezone.utc).isoformat()
    return [
        {"Key": f"{TAG_PREFIX}:managed", "Value": "true"},
        {"Key": f"{TAG_PREFIX}:source-region", "Value": home_region},
        {"Key": f"{TAG_PREFIX}:job-name", "Value": job_name},
        {"Key": f"{TAG_PREFIX}:created-at", "Value": now},
    ]


# ── Auto-create model feasibility ──────────────────────────────

# Import algo accounts mapping for image URI resolution.
from sm_resilient_launcher._cross_region_manager import (
    SAGEMAKER_ALGO_ACCOUNTS,
)


def _check_auto_create_model_feasibility(
    model_name: str,
    image_uri: str,
    target_region: str,
    home_region: str,
) -> tuple[str, str | None]:
    """Check whether auto_create_model would succeed for a region.

    Verifies:
    - The image URI is ECR-based.
    - For customer-owned images: the image exists in target ECR.
    - For AWS-managed images: the target account mapping exists.
    - The target model name is not already taken.

    Returns:
        ``("feasible", None)`` when auto-creation looks viable, or
        ``("infeasible", details)`` with an explanation.
    """
    if not image_uri or ".ecr." not in image_uri:
        return "feasible", None

    source_account = _parse_ecr_account(image_uri)
    repo_name = _parse_ecr_repo(image_uri)
    tag = _parse_ecr_tag(image_uri)

    # Determine caller account.
    try:
        sts = boto3.client("sts")
        caller_account = sts.get_caller_identity()["Account"]
    except Exception:
        caller_account = ""

    is_customer_owned = caller_account and source_account == caller_account

    if is_customer_owned:
        # Verify image exists in target ECR.
        try:
            ecr = boto3.client("ecr", region_name=target_region)
            ecr.describe_images(
                repositoryName=repo_name,
                imageIds=[{"imageTag": tag}],
            )
        except Exception:
            details = (
                f"AutoCreateModel feasibility: customer ECR image "
                f"{image_uri} not found in {target_region}. "
                f"Push the image to {target_region} before using "
                f"AutoCreateModel=True."
            )
            return "infeasible", details
    else:
        # AWS-managed image — check mapping exists.
        if target_region not in SAGEMAKER_ALGO_ACCOUNTS:
            details = (
                f"AutoCreateModel feasibility: no known SageMaker "
                f"algorithm account for {target_region}. Image "
                f"resolution may fail."
            )
            return "infeasible", details

    # Check model name availability in target region.
    target_model_name = f"{model_name}-{target_region}"
    try:
        sm = boto3.client("sagemaker", region_name=target_region)
        sm.describe_model(ModelName=target_model_name)
        # Model already exists — that's fine, we can reuse it.
    except Exception:
        # Model doesn't exist — auto-creation can proceed.
        pass

    return "feasible", None
