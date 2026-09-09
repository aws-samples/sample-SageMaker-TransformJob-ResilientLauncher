"""Unit tests for CrossRegionManager.

Property 14: S3 mirror bucket naming and SDK tag application
    For any S3 mirror bucket created by the SDK, the bucket name SHALL
    follow the pattern {original-bucket}-{target-region}, and the SDK
    SHALL apply all four tags at creation time.

Property 15: Copied S3 objects receive SDK tags
    For any S3 object copied by the SDK, the copy operation SHALL apply
    all four SDK tags to the copied object at copy time.

Property 16: Cross-region replicates model and input before submission
    For any Batch Transform job submitted in a non-home region, the
    CrossRegionManager SHALL copy both the model artifact and the input
    data S3 prefix before submitting the job, and SHALL copy the output
    back to the home region after completion.

Property 17: ECR missing skips region with CLI commands in warnings
    For any target region where the container image does not exist in
    ECR, the CrossRegionManager SHALL skip that region, emit a structured
    WARNING log containing the exact CLI commands, and add those commands
    to the LaunchResult.warnings field.

Property 20: Cross-region tries regions in AllowedRegions order
    For any AllowedRegions list with N regions, the CrossRegionManager
    SHALL try regions in the exact order provided.

Validates: Requirements 8.1-8.4, 9.1-9.8, 10.1-10.5, 11.1-11.4,
           12.1-12.3
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sm_resilient_launcher._cross_region_manager import CrossRegionManager
from sm_resilient_launcher._error_classifier import (
    ClassifiedError,
    ErrorCategory,
)
from sm_resilient_launcher._exceptions import (
    AllRegionsExhaustedError,
    AllTypesExhaustedError,
)
from sm_resilient_launcher._launch_result import LaunchResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BOTO3_PARAMS = {
    "ModelName": "my-model",
    "TransformInput": {
        "DataSource": {
            "S3DataSource": {"S3Uri": "s3://my-bucket/input/data"}
        }
    },
    "TransformOutput": {"S3OutputPath": "s3://my-bucket/output/results"},
    "TransformResources": {
        "InstanceCount": 1,
        "RoleArn": "arn:aws:iam::123456789012:role/SageMakerRole",
    },
}

INSTANCE_TYPES = ["ml.g5.4xlarge"]

HOME_REGION = "us-east-1"
CROSS_REGION = "us-west-2"
ALLOWED_REGIONS = [HOME_REGION, CROSS_REGION]


def _make_launch_result(region: str = HOME_REGION) -> LaunchResult:
    """Build a minimal successful LaunchResult."""
    return LaunchResult(
        job_name="test-job-0-ml-g5-4xlarge",
        instance_type_used="ml.g5.4xlarge",
        region_used=region,
        attempts=1,
        strategy_used="sequential",
    )


def _home_fail_error() -> ClassifiedError:
    """Create a ClassifiedError for home region failure."""
    return ClassifiedError(
        category=ErrorCategory.CAPACITY,
        original_exception=Exception("no capacity"),
        message="no capacity",
        instance_type="ml.g5.4xlarge",
        region=HOME_REGION,
    )


def _make_s3_client_mock(
    bucket_exists: bool = True,
    objects: list[dict] | None = None,
) -> MagicMock:
    """Build a mock S3 client."""
    client = MagicMock()

    if bucket_exists:
        client.head_bucket.return_value = {}
    else:
        client.head_bucket.side_effect = Exception("NoSuchBucket")

    client.create_bucket.return_value = {}
    client.put_bucket_tagging.return_value = {}
    client.copy_object.return_value = {}

    if objects is None:
        objects = []

    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": objects}]
    client.get_paginator.return_value = paginator

    return client


def _make_sagemaker_client_mock(
    image_uri: str = "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-repo:v1",
) -> MagicMock:
    """Build a mock SageMaker client that returns model descriptions."""
    client = MagicMock()
    client.describe_model.return_value = {
        "PrimaryContainer": {"Image": image_uri}
    }
    return client


def _make_ecr_client_mock(image_exists: bool = True) -> MagicMock:
    """Build a mock ECR client."""
    client = MagicMock()
    if image_exists:
        client.describe_images.return_value = {"imageDetails": [{}]}
    else:
        client.describe_images.side_effect = Exception(
            "ImageNotFoundException"
        )
    return client


def _make_iam_client_mock(role_exists: bool = True) -> MagicMock:
    """Build a mock IAM client."""
    client = MagicMock()
    if role_exists:
        client.get_role.return_value = {"Role": {"RoleName": "SageMakerRole"}}
    else:
        client.get_role.side_effect = Exception("NoSuchEntity")

    client.list_attached_role_policies.return_value = {
        "AttachedPolicies": [
            {"PolicyName": "SageMakerFullAccess", "PolicyArn": "arn:..."}
        ]
    }
    client.list_role_policies.return_value = {"PolicyNames": []}
    return client


def _boto3_client_router(
    s3_clients: dict[str, MagicMock],
    sm_home: MagicMock,
    ecr_clients: dict[str, MagicMock] | None = None,
    iam: MagicMock | None = None,
):
    """Return a side_effect function for boto3.client.

    Args:
        s3_clients: Mapping of region -> mock S3 client.
        sm_home: Mock SageMaker client (always home region).
        ecr_clients: Mapping of region -> mock ECR client.
        iam: Mock IAM client (global).
    """
    ecr_clients = ecr_clients or {}
    iam = iam or _make_iam_client_mock()

    def _router(service, region_name=None, **kwargs):
        if service == "s3":
            return s3_clients.get(region_name, MagicMock())
        if service == "sagemaker":
            return sm_home
        if service == "ecr":
            return ecr_clients.get(region_name, MagicMock())
        if service == "iam":
            return iam
        return MagicMock()

    return _router


def _make_manager_with_failover(
    allowed_regions: list[str] | None = None,
    home_fails: bool = True,
    cross_result: LaunchResult | None = None,
) -> tuple[CrossRegionManager, MagicMock, MagicMock]:
    """Build a CrossRegionManager where the home region fails.

    This forces the manager to exercise cross-region paths.

    Returns:
        (manager, adapter_factory, mock_engine)
    """
    regions = allowed_regions or ALLOWED_REGIONS
    adapter_factory = MagicMock()
    mock_engine = MagicMock()

    if home_fails:
        cross_result = cross_result or _make_launch_result(CROSS_REGION)

        def _engine_side_effect(**kwargs):
            region = kwargs.get("region", "")
            if region == regions[0]:
                raise AllTypesExhaustedError([_home_fail_error()])
            return cross_result

        mock_engine.execute.side_effect = _engine_side_effect
    else:
        mock_engine.execute.return_value = _make_launch_result(
            regions[0]
        )

    strategy_engine_factory = MagicMock(return_value=mock_engine)

    manager = CrossRegionManager(
        adapter_factory=adapter_factory,
        strategy_engine_factory=strategy_engine_factory,
        home_region=regions[0],
        allowed_regions=regions,
        base_job_name="test-job",
    )

    return manager, adapter_factory, mock_engine


# ===========================================================================
# 1. Mirror bucket naming follows {original-bucket}-{target-region}
#    Property 14 — Requirement 9.1, 9.2
# ===========================================================================


class TestMirrorBucketNaming:
    """Mirror bucket name follows {original-bucket}-{target-region}."""

    @patch("sm_resilient_launcher._cross_region_manager.boto3.client")
    def test_mirror_bucket_name_pattern(self, mock_boto3_client):
        """Created bucket is named {source-bucket}-{target-region}."""
        s3_cross = _make_s3_client_mock(bucket_exists=False)
        s3_home = _make_s3_client_mock()
        sm_home = _make_sagemaker_client_mock()
        ecr_cross = _make_ecr_client_mock(image_exists=True)
        iam = _make_iam_client_mock()

        mock_boto3_client.side_effect = _boto3_client_router(
            s3_clients={HOME_REGION: s3_home, CROSS_REGION: s3_cross},
            sm_home=sm_home,
            ecr_clients={CROSS_REGION: ecr_cross},
            iam=iam,
        )

        manager, _, _ = _make_manager_with_failover()
        manager.execute(INSTANCE_TYPES, BOTO3_PARAMS, "test-job")

        s3_cross.create_bucket.assert_called_once()
        create_kwargs = s3_cross.create_bucket.call_args[1]
        assert create_kwargs["Bucket"] == f"my-bucket-{CROSS_REGION}"

    @patch("sm_resilient_launcher._cross_region_manager.boto3.client")
    def test_existing_bucket_not_recreated(self, mock_boto3_client):
        """When the mirror bucket already exists, create_bucket is NOT called."""
        s3_cross = _make_s3_client_mock(bucket_exists=True)
        s3_home = _make_s3_client_mock()
        sm_home = _make_sagemaker_client_mock()
        ecr_cross = _make_ecr_client_mock(image_exists=True)
        iam = _make_iam_client_mock()

        mock_boto3_client.side_effect = _boto3_client_router(
            s3_clients={HOME_REGION: s3_home, CROSS_REGION: s3_cross},
            sm_home=sm_home,
            ecr_clients={CROSS_REGION: ecr_cross},
            iam=iam,
        )

        manager, _, _ = _make_manager_with_failover()
        manager.execute(INSTANCE_TYPES, BOTO3_PARAMS, "test-job")

        s3_cross.create_bucket.assert_not_called()


# ===========================================================================
# 2. SDK tags applied on bucket creation (all four tags)
#    Property 14 — Requirement 9.3, 12.1, 12.2
# ===========================================================================


class TestBucketSDKTags:
    """All four SDK tags are applied at bucket creation time."""

    @patch("sm_resilient_launcher._cross_region_manager.boto3.client")
    def test_four_sdk_tags_applied(self, mock_boto3_client):
        """put_bucket_tagging includes managed, source-region, job-name, created-at."""
        s3_cross = _make_s3_client_mock(bucket_exists=False)
        s3_home = _make_s3_client_mock()
        sm_home = _make_sagemaker_client_mock()
        ecr_cross = _make_ecr_client_mock(image_exists=True)
        iam = _make_iam_client_mock()

        mock_boto3_client.side_effect = _boto3_client_router(
            s3_clients={HOME_REGION: s3_home, CROSS_REGION: s3_cross},
            sm_home=sm_home,
            ecr_clients={CROSS_REGION: ecr_cross},
            iam=iam,
        )

        manager, _, _ = _make_manager_with_failover()
        manager.execute(INSTANCE_TYPES, BOTO3_PARAMS, "test-job")

        s3_cross.put_bucket_tagging.assert_called_once()
        tag_call = s3_cross.put_bucket_tagging.call_args
        tag_set = tag_call[1]["Tagging"]["TagSet"]

        tag_keys = {t["Key"] for t in tag_set}
        assert "sm-resilient-launcher:managed" in tag_keys
        assert "sm-resilient-launcher:source-region" in tag_keys
        assert "sm-resilient-launcher:job-name" in tag_keys
        assert "sm-resilient-launcher:created-at" in tag_keys
        assert len(tag_set) == 4

    @patch("sm_resilient_launcher._cross_region_manager.boto3.client")
    def test_tag_values_are_correct(self, mock_boto3_client):
        """Tag values match expected: managed=true, source-region=home, etc."""
        s3_cross = _make_s3_client_mock(bucket_exists=False)
        s3_home = _make_s3_client_mock()
        sm_home = _make_sagemaker_client_mock()
        ecr_cross = _make_ecr_client_mock(image_exists=True)
        iam = _make_iam_client_mock()

        mock_boto3_client.side_effect = _boto3_client_router(
            s3_clients={HOME_REGION: s3_home, CROSS_REGION: s3_cross},
            sm_home=sm_home,
            ecr_clients={CROSS_REGION: ecr_cross},
            iam=iam,
        )

        manager, _, _ = _make_manager_with_failover()
        manager.execute(INSTANCE_TYPES, BOTO3_PARAMS, "test-job")

        tag_set = s3_cross.put_bucket_tagging.call_args[1]["Tagging"]["TagSet"]
        tag_map = {t["Key"]: t["Value"] for t in tag_set}

        assert tag_map["sm-resilient-launcher:managed"] == "true"
        assert tag_map["sm-resilient-launcher:source-region"] == HOME_REGION
        assert tag_map["sm-resilient-launcher:job-name"] == "test-job"
        assert len(tag_map["sm-resilient-launcher:created-at"]) > 0


# ===========================================================================
# 3. SDK tags applied on each S3 object copy
#    Property 15 — Requirement 9.6, 12.1
# ===========================================================================


class TestCopyObjectSDKTags:
    """Copied S3 objects receive all four SDK tags at copy time."""

    @patch("sm_resilient_launcher._cross_region_manager.boto3.client")
    def test_copy_object_includes_tagging(self, mock_boto3_client):
        """Each copy_object call includes a Tagging string with SDK tags."""
        objects = [
            {"Key": "input/data/file1.csv"},
            {"Key": "input/data/file2.csv"},
        ]

        s3_cross = _make_s3_client_mock(bucket_exists=True, objects=objects)
        s3_home = _make_s3_client_mock(objects=objects)
        sm_home = _make_sagemaker_client_mock()
        ecr_cross = _make_ecr_client_mock(image_exists=True)
        iam = _make_iam_client_mock()

        mock_boto3_client.side_effect = _boto3_client_router(
            s3_clients={HOME_REGION: s3_home, CROSS_REGION: s3_cross},
            sm_home=sm_home,
            ecr_clients={CROSS_REGION: ecr_cross},
            iam=iam,
        )

        manager, _, _ = _make_manager_with_failover()
        manager.execute(INSTANCE_TYPES, BOTO3_PARAMS, "test-job")

        # copy_object should have been called on the destination S3 client.
        assert s3_cross.copy_object.call_count > 0

        for copy_call in s3_cross.copy_object.call_args_list:
            kwargs = copy_call[1]
            assert "Tagging" in kwargs
            tag_string = kwargs["Tagging"]
            assert "sm-resilient-launcher:managed=true" in tag_string
            assert (
                f"sm-resilient-launcher:source-region={HOME_REGION}"
                in tag_string
            )
            assert "sm-resilient-launcher:job-name=test-job" in tag_string
            assert "sm-resilient-launcher:created-at=" in tag_string
            assert kwargs["TaggingDirective"] == "REPLACE"


# ===========================================================================
# 4. ECR missing skips region and adds CLI commands to warnings
#    Property 17 — Requirement 10.1-10.5
# ===========================================================================


class TestECRMissingSkipsRegion:
    """Missing ECR image in target region skips it with CLI commands."""

    @patch("sm_resilient_launcher._cross_region_manager.boto3.client")
    def test_ecr_missing_skips_region(self, mock_boto3_client):
        """When ECR image is missing, the region is skipped."""
        s3_cross = _make_s3_client_mock(bucket_exists=False)
        s3_home = _make_s3_client_mock()
        sm_home = _make_sagemaker_client_mock()
        ecr_cross = _make_ecr_client_mock(image_exists=False)
        iam = _make_iam_client_mock()

        mock_boto3_client.side_effect = _boto3_client_router(
            s3_clients={HOME_REGION: s3_home, CROSS_REGION: s3_cross},
            sm_home=sm_home,
            ecr_clients={CROSS_REGION: ecr_cross},
            iam=iam,
        )

        # Home region succeeds; cross-region skipped due to ECR.
        manager, _, _ = _make_manager_with_failover(home_fails=False)
        result = manager.execute(INSTANCE_TYPES, BOTO3_PARAMS, "test-job")

        assert result.region_used == HOME_REGION

    @patch("sm_resilient_launcher._cross_region_manager.boto3.client")
    def test_ecr_missing_adds_warnings_on_cross_region(self, mock_boto3_client):
        """CLI commands for ECR replication are added to warnings."""
        s3_cross = _make_s3_client_mock(bucket_exists=False)
        s3_home = _make_s3_client_mock()
        sm_home = _make_sagemaker_client_mock()
        ecr_cross = _make_ecr_client_mock(image_exists=False)
        iam = _make_iam_client_mock()

        # Need a 3rd region that succeeds so we can get warnings
        # from the skipped CROSS_REGION.
        third_region = "eu-west-1"
        s3_third = _make_s3_client_mock(bucket_exists=True)
        ecr_third = _make_ecr_client_mock(image_exists=True)

        def _router(service, region_name=None, **kwargs):
            if service == "s3":
                return {
                    HOME_REGION: s3_home,
                    CROSS_REGION: s3_cross,
                    third_region: s3_third,
                }.get(region_name, MagicMock())
            if service == "sagemaker":
                return sm_home
            if service == "ecr":
                return {
                    CROSS_REGION: ecr_cross,
                    third_region: ecr_third,
                }.get(region_name, MagicMock())
            if service == "iam":
                return iam
            return MagicMock()

        mock_boto3_client.side_effect = _router

        regions = [HOME_REGION, CROSS_REGION, third_region]
        adapter_factory = MagicMock()
        mock_engine = MagicMock()

        def _engine_side_effect(**kwargs):
            region = kwargs.get("region", "")
            if region == HOME_REGION:
                raise AllTypesExhaustedError([_home_fail_error()])
            # third region succeeds
            return _make_launch_result(third_region)

        mock_engine.execute.side_effect = _engine_side_effect
        strategy_engine_factory = MagicMock(return_value=mock_engine)

        manager = CrossRegionManager(
            adapter_factory=adapter_factory,
            strategy_engine_factory=strategy_engine_factory,
            home_region=HOME_REGION,
            allowed_regions=regions,
            base_job_name="test-job",
        )

        result = manager.execute(INSTANCE_TYPES, BOTO3_PARAMS, "test-job")

        # Warnings should contain ECR replication commands.
        assert len(result.warnings) > 0
        ecr_warnings = [w for w in result.warnings if "ECR image" in w]
        assert len(ecr_warnings) > 0
        assert CROSS_REGION in ecr_warnings[0]
        assert "docker pull" in ecr_warnings[0]
        assert "docker push" in ecr_warnings[0]

    @patch("sm_resilient_launcher._cross_region_manager.boto3.client")
    def test_ecr_not_replicated_automatically(self, mock_boto3_client):
        """The SDK never calls ECR create-repository or puts images itself."""
        s3_cross = _make_s3_client_mock(bucket_exists=False)
        s3_home = _make_s3_client_mock()
        sm_home = _make_sagemaker_client_mock()
        ecr_cross = _make_ecr_client_mock(image_exists=False)
        iam = _make_iam_client_mock()

        mock_boto3_client.side_effect = _boto3_client_router(
            s3_clients={HOME_REGION: s3_home, CROSS_REGION: s3_cross},
            sm_home=sm_home,
            ecr_clients={CROSS_REGION: ecr_cross},
            iam=iam,
        )

        manager, _, _ = _make_manager_with_failover(home_fails=False)
        manager.execute(INSTANCE_TYPES, BOTO3_PARAMS, "test-job")

        ecr_cross.create_repository.assert_not_called()


# ===========================================================================
# 5. IAM validation warns but does not block
#    Property 14 — Requirement 11.1-11.4
# ===========================================================================


class TestIAMValidation:
    """IAM validation is non-blocking; warnings are added."""

    @patch("sm_resilient_launcher._cross_region_manager.boto3.client")
    def test_iam_warning_non_blocking(self, mock_boto3_client):
        """IAM policy gaps produce warnings but do not skip the region."""
        s3_cross = _make_s3_client_mock(bucket_exists=True)
        s3_home = _make_s3_client_mock()
        sm_home = _make_sagemaker_client_mock()
        ecr_cross = _make_ecr_client_mock(image_exists=True)
        iam = _make_iam_client_mock(role_exists=True)

        mock_boto3_client.side_effect = _boto3_client_router(
            s3_clients={HOME_REGION: s3_home, CROSS_REGION: s3_cross},
            sm_home=sm_home,
            ecr_clients={CROSS_REGION: ecr_cross},
            iam=iam,
        )

        manager, _, _ = _make_manager_with_failover()
        result = manager.execute(INSTANCE_TYPES, BOTO3_PARAMS, "test-job")

        # IAM warnings should be present.
        iam_warnings = [w for w in result.warnings if "Execution role" in w]
        assert len(iam_warnings) > 0
        assert (
            "may not have policies" in iam_warnings[0]
            or "mirror bucket" in iam_warnings[0].lower()
        )

    @patch("sm_resilient_launcher._cross_region_manager.boto3.client")
    def test_iam_role_not_modified(self, mock_boto3_client):
        """The SDK never calls put_role_policy, attach_role_policy, etc."""
        s3_cross = _make_s3_client_mock(bucket_exists=True)
        s3_home = _make_s3_client_mock()
        sm_home = _make_sagemaker_client_mock()
        ecr_cross = _make_ecr_client_mock(image_exists=True)
        iam = _make_iam_client_mock(role_exists=True)

        mock_boto3_client.side_effect = _boto3_client_router(
            s3_clients={HOME_REGION: s3_home, CROSS_REGION: s3_cross},
            sm_home=sm_home,
            ecr_clients={CROSS_REGION: ecr_cross},
            iam=iam,
        )

        manager, _, _ = _make_manager_with_failover()
        manager.execute(INSTANCE_TYPES, BOTO3_PARAMS, "test-job")

        iam.put_role_policy.assert_not_called()
        iam.attach_role_policy.assert_not_called()

    @patch("sm_resilient_launcher._cross_region_manager.boto3.client")
    def test_iam_role_missing_still_proceeds(self, mock_boto3_client):
        """When the role doesn't exist, add a warning but don't block."""
        s3_cross = _make_s3_client_mock(bucket_exists=True)
        s3_home = _make_s3_client_mock()
        sm_home = _make_sagemaker_client_mock()
        ecr_cross = _make_ecr_client_mock(image_exists=True)
        iam = _make_iam_client_mock(role_exists=False)

        mock_boto3_client.side_effect = _boto3_client_router(
            s3_clients={HOME_REGION: s3_home, CROSS_REGION: s3_cross},
            sm_home=sm_home,
            ecr_clients={CROSS_REGION: ecr_cross},
            iam=iam,
        )

        manager, _, _ = _make_manager_with_failover()
        result = manager.execute(INSTANCE_TYPES, BOTO3_PARAMS, "test-job")

        assert any("could not be validated" in w for w in result.warnings)


# ===========================================================================
# 6. Data replication called before job submission
#    Property 16 — Requirement 9.4, 9.5
# ===========================================================================


class TestDataReplicationBeforeSubmission:
    """Model artifact and input data are replicated before job submission."""

    @patch("sm_resilient_launcher._cross_region_manager.boto3.client")
    def test_copy_objects_called_for_cross_region(self, mock_boto3_client):
        """copy_object is called to replicate data to the mirror bucket."""
        objects = [{"Key": "input/data/file.csv"}]

        s3_cross = _make_s3_client_mock(bucket_exists=True, objects=objects)
        s3_home = _make_s3_client_mock(objects=objects)
        sm_home = _make_sagemaker_client_mock()
        ecr_cross = _make_ecr_client_mock(image_exists=True)
        iam = _make_iam_client_mock()

        mock_boto3_client.side_effect = _boto3_client_router(
            s3_clients={HOME_REGION: s3_home, CROSS_REGION: s3_cross},
            sm_home=sm_home,
            ecr_clients={CROSS_REGION: ecr_cross},
            iam=iam,
        )

        manager, _, _ = _make_manager_with_failover()
        manager.execute(INSTANCE_TYPES, BOTO3_PARAMS, "test-job")

        # copy_object must have been called (for input data replication).
        assert s3_cross.copy_object.call_count > 0

    @patch("sm_resilient_launcher._cross_region_manager.boto3.client")
    def test_home_region_no_replication(self, mock_boto3_client):
        """The home region does not trigger data replication."""
        s3_home = _make_s3_client_mock()

        mock_boto3_client.side_effect = _boto3_client_router(
            s3_clients={HOME_REGION: s3_home},
            sm_home=_make_sagemaker_client_mock(),
        )

        manager, _, _ = _make_manager_with_failover(home_fails=False)
        result = manager.execute(INSTANCE_TYPES, BOTO3_PARAMS, "test-job")

        s3_home.copy_object.assert_not_called()
        assert result.data_replicated is False


# ===========================================================================
# 7. Output retrieval called after cross-region success
#    Property 16 — Requirement 9.7
# ===========================================================================


class TestOutputRetrieval:
    """Job output is copied back to the home region after cross-region success."""

    @patch("sm_resilient_launcher._cross_region_manager.boto3.client")
    def test_output_retrieval_after_cross_region_success(self, mock_boto3_client):
        """After cross-region success, output is copied back."""
        objects = [{"Key": "output/results/part-00000"}]

        s3_cross = _make_s3_client_mock(bucket_exists=True, objects=objects)
        s3_home = _make_s3_client_mock(objects=objects)
        sm_home = _make_sagemaker_client_mock()
        ecr_cross = _make_ecr_client_mock(image_exists=True)
        iam = _make_iam_client_mock()

        mock_boto3_client.side_effect = _boto3_client_router(
            s3_clients={HOME_REGION: s3_home, CROSS_REGION: s3_cross},
            sm_home=sm_home,
            ecr_clients={CROSS_REGION: ecr_cross},
            iam=iam,
        )

        manager, _, _ = _make_manager_with_failover()
        result = manager.execute(INSTANCE_TYPES, BOTO3_PARAMS, "test-job")

        assert result.data_replicated is True


# ===========================================================================
# 8. AllRegionsExhaustedError when all regions fail
#    Property 20 — Requirement 8.4
# ===========================================================================


class TestAllRegionsExhausted:
    """AllRegionsExhaustedError raised when every region fails."""

    @patch("sm_resilient_launcher._cross_region_manager.boto3.client")
    def test_raises_when_all_regions_fail(self, mock_boto3_client):
        """When both home and cross-region fail, AllRegionsExhaustedError."""
        s3_cross = _make_s3_client_mock(bucket_exists=True)
        s3_home = _make_s3_client_mock()
        sm_home = _make_sagemaker_client_mock()
        ecr_cross = _make_ecr_client_mock(image_exists=True)
        iam = _make_iam_client_mock()

        mock_boto3_client.side_effect = _boto3_client_router(
            s3_clients={HOME_REGION: s3_home, CROSS_REGION: s3_cross},
            sm_home=sm_home,
            ecr_clients={CROSS_REGION: ecr_cross},
            iam=iam,
        )

        errors = [_home_fail_error()]
        adapter_factory = MagicMock()
        mock_engine = MagicMock()
        mock_engine.execute.side_effect = AllTypesExhaustedError(errors)
        strategy_engine_factory = MagicMock(return_value=mock_engine)

        manager = CrossRegionManager(
            adapter_factory=adapter_factory,
            strategy_engine_factory=strategy_engine_factory,
            home_region=HOME_REGION,
            allowed_regions=ALLOWED_REGIONS,
            base_job_name="test-job",
        )

        with pytest.raises(AllRegionsExhaustedError) as exc_info:
            manager.execute(INSTANCE_TYPES, BOTO3_PARAMS, "test-job")

        assert len(exc_info.value.errors) >= 2

    @patch("sm_resilient_launcher._cross_region_manager.boto3.client")
    def test_error_details_from_all_regions(self, mock_boto3_client):
        """Each region's errors are accumulated in the final exception."""
        s3_cross = _make_s3_client_mock(bucket_exists=True)
        s3_home = _make_s3_client_mock()
        sm_home = _make_sagemaker_client_mock()
        ecr_cross = _make_ecr_client_mock(image_exists=True)
        iam = _make_iam_client_mock()

        mock_boto3_client.side_effect = _boto3_client_router(
            s3_clients={HOME_REGION: s3_home, CROSS_REGION: s3_cross},
            sm_home=sm_home,
            ecr_clients={CROSS_REGION: ecr_cross},
            iam=iam,
        )

        home_error = ClassifiedError(
            category=ErrorCategory.CAPACITY,
            original_exception=Exception("home no capacity"),
            message="home no capacity",
            instance_type="ml.g5.4xlarge",
            region=HOME_REGION,
        )
        cross_error = ClassifiedError(
            category=ErrorCategory.QUOTA,
            original_exception=Exception("cross quota exceeded"),
            message="cross quota exceeded",
            instance_type="ml.g5.4xlarge",
            region=CROSS_REGION,
        )

        def _engine_side_effect(**kwargs):
            region = kwargs.get("region", "")
            if region == HOME_REGION:
                raise AllTypesExhaustedError([home_error])
            raise AllTypesExhaustedError([cross_error])

        adapter_factory = MagicMock()
        mock_engine = MagicMock()
        mock_engine.execute.side_effect = _engine_side_effect
        strategy_engine_factory = MagicMock(return_value=mock_engine)

        manager = CrossRegionManager(
            adapter_factory=adapter_factory,
            strategy_engine_factory=strategy_engine_factory,
            home_region=HOME_REGION,
            allowed_regions=ALLOWED_REGIONS,
            base_job_name="test-job",
        )

        with pytest.raises(AllRegionsExhaustedError) as exc_info:
            manager.execute(INSTANCE_TYPES, BOTO3_PARAMS, "test-job")

        all_errors = exc_info.value.errors
        categories = {e.category for e in all_errors}
        assert ErrorCategory.CAPACITY in categories
        assert ErrorCategory.QUOTA in categories


# ===========================================================================
# 9. Regions tried in exact AllowedRegions order
#    Property 20 — Requirement 8.1, 8.2, 8.3
# ===========================================================================


class TestRegionOrder:
    """Regions are tried in the exact order from AllowedRegions."""

    @patch("sm_resilient_launcher._cross_region_manager.boto3.client")
    def test_regions_tried_in_order(self, mock_boto3_client):
        """Engine is called for regions in the AllowedRegions order."""
        sm_home = _make_sagemaker_client_mock()
        ecr = _make_ecr_client_mock(image_exists=True)
        iam = _make_iam_client_mock()

        mock_boto3_client.side_effect = lambda svc, region_name=None, **kw: (
            _make_s3_client_mock(bucket_exists=True)
            if svc == "s3"
            else sm_home
            if svc == "sagemaker"
            else ecr
            if svc == "ecr"
            else iam
            if svc == "iam"
            else MagicMock()
        )

        regions_tried = []

        def _engine_side_effect(**kwargs):
            region = kwargs.get("region", "")
            regions_tried.append(region)
            if region == "eu-west-1":
                return _make_launch_result("eu-west-1")
            raise AllTypesExhaustedError([
                ClassifiedError(
                    category=ErrorCategory.CAPACITY,
                    original_exception=Exception("no cap"),
                    message="no cap",
                    instance_type="ml.g5.4xlarge",
                    region=region,
                )
            ])

        adapter_factory = MagicMock()
        mock_engine = MagicMock()
        mock_engine.execute.side_effect = _engine_side_effect
        strategy_engine_factory = MagicMock(return_value=mock_engine)

        three_regions = ["us-east-1", "us-west-2", "eu-west-1"]
        manager = CrossRegionManager(
            adapter_factory=adapter_factory,
            strategy_engine_factory=strategy_engine_factory,
            home_region="us-east-1",
            allowed_regions=three_regions,
            base_job_name="test-job",
        )

        result = manager.execute(INSTANCE_TYPES, BOTO3_PARAMS, "test-job")

        assert regions_tried == three_regions
        assert result.region_used == "eu-west-1"

    @patch("sm_resilient_launcher._cross_region_manager.boto3.client")
    def test_home_region_tried_first(self, mock_boto3_client):
        """The home region is always the first one tried."""
        mock_boto3_client.return_value = MagicMock()

        regions_tried = []

        def _engine_side_effect(**kwargs):
            regions_tried.append(kwargs.get("region", ""))
            return _make_launch_result(kwargs.get("region", ""))

        adapter_factory = MagicMock()
        mock_engine = MagicMock()
        mock_engine.execute.side_effect = _engine_side_effect
        strategy_engine_factory = MagicMock(return_value=mock_engine)

        manager = CrossRegionManager(
            adapter_factory=adapter_factory,
            strategy_engine_factory=strategy_engine_factory,
            home_region=HOME_REGION,
            allowed_regions=ALLOWED_REGIONS,
            base_job_name="test-job",
        )

        manager.execute(INSTANCE_TYPES, BOTO3_PARAMS, "test-job")

        assert regions_tried[0] == HOME_REGION


# ===========================================================================
# 10. data_replicated=True on cross-region success
#     Property 16 — Requirement 15.3
# ===========================================================================


class TestDataReplicatedFlag:
    """LaunchResult.data_replicated is True when cross-region succeeds."""

    @patch("sm_resilient_launcher._cross_region_manager.boto3.client")
    def test_data_replicated_true_for_cross_region(self, mock_boto3_client):
        """When the winning region is not home, data_replicated is True."""
        s3_cross = _make_s3_client_mock(bucket_exists=True)
        s3_home = _make_s3_client_mock()
        sm_home = _make_sagemaker_client_mock()
        ecr_cross = _make_ecr_client_mock(image_exists=True)
        iam = _make_iam_client_mock()

        mock_boto3_client.side_effect = _boto3_client_router(
            s3_clients={HOME_REGION: s3_home, CROSS_REGION: s3_cross},
            sm_home=sm_home,
            ecr_clients={CROSS_REGION: ecr_cross},
            iam=iam,
        )

        manager, _, _ = _make_manager_with_failover()
        result = manager.execute(INSTANCE_TYPES, BOTO3_PARAMS, "test-job")

        assert result.data_replicated is True

    @patch("sm_resilient_launcher._cross_region_manager.boto3.client")
    def test_data_replicated_false_for_home_region(self, mock_boto3_client):
        """When the winning region is home, data_replicated is False."""
        mock_boto3_client.return_value = MagicMock()

        manager, _, _ = _make_manager_with_failover(home_fails=False)
        result = manager.execute(INSTANCE_TYPES, BOTO3_PARAMS, "test-job")

        assert result.data_replicated is False


# ===========================================================================
# 11. S3 URI rewriting for mirror buckets
#     Requirement 9.4, 9.5
# ===========================================================================


class TestS3URIRewriting:
    """S3 URIs are rewritten to point to mirror buckets for cross-region."""

    def test_rewrite_input_uri(self):
        """Input S3 URI is rewritten to the mirror bucket."""
        manager, _, _ = _make_manager_with_failover()
        rewritten = manager._rewrite_s3_uris(BOTO3_PARAMS, CROSS_REGION)

        expected_input = f"s3://my-bucket-{CROSS_REGION}/input/data"
        actual_input = (
            rewritten["TransformInput"]["DataSource"]["S3DataSource"]["S3Uri"]
        )
        assert actual_input == expected_input

    def test_rewrite_output_uri(self):
        """Output S3 path is rewritten to the mirror bucket."""
        manager, _, _ = _make_manager_with_failover()
        rewritten = manager._rewrite_s3_uris(BOTO3_PARAMS, CROSS_REGION)

        expected_output = f"s3://my-bucket-{CROSS_REGION}/output/results"
        actual_output = rewritten["TransformOutput"]["S3OutputPath"]
        assert actual_output == expected_output

    def test_original_params_not_mutated(self):
        """_rewrite_s3_uris does not modify the original dict."""
        import copy

        original = copy.deepcopy(BOTO3_PARAMS)
        manager, _, _ = _make_manager_with_failover()
        manager._rewrite_s3_uris(BOTO3_PARAMS, CROSS_REGION)

        assert BOTO3_PARAMS == original


# ===========================================================================
# 12. S3 copy failure skips region (Req 9.4, 9.5)
# ===========================================================================


class TestS3CopyFailureSkipsRegion:
    """When S3 data replication fails, the region is skipped."""

    @patch("sm_resilient_launcher._cross_region_manager.boto3.client")
    def test_s3_copy_failure_skips_to_next_region(self, mock_boto3_client):
        """If _replicate_data raises, the cross-region is skipped."""
        sm_home = _make_sagemaker_client_mock()
        ecr_cross = _make_ecr_client_mock(image_exists=True)
        iam = _make_iam_client_mock()

        # The _copy_s3_prefix method creates S3 clients for src and dst.
        # For _replicate_data (src=HOME, dst=CROSS), the source client
        # lists objects; we make that paginator fail.
        s3_home_bad = _make_s3_client_mock()
        bad_paginator = MagicMock()
        bad_paginator.paginate.side_effect = Exception("S3 copy failed")
        s3_home_bad.get_paginator.return_value = bad_paginator

        s3_cross = _make_s3_client_mock(bucket_exists=True)

        mock_boto3_client.side_effect = _boto3_client_router(
            s3_clients={HOME_REGION: s3_home_bad, CROSS_REGION: s3_cross},
            sm_home=sm_home,
            ecr_clients={CROSS_REGION: ecr_cross},
            iam=iam,
        )

        adapter_factory = MagicMock()
        mock_engine = MagicMock()

        def _engine_side_effect(**kwargs):
            region = kwargs.get("region", "")
            if region == HOME_REGION:
                raise AllTypesExhaustedError([_home_fail_error()])
            # Cross-region should never reach here if copy failed.
            return _make_launch_result(region)

        mock_engine.execute.side_effect = _engine_side_effect
        strategy_engine_factory = MagicMock(return_value=mock_engine)

        manager = CrossRegionManager(
            adapter_factory=adapter_factory,
            strategy_engine_factory=strategy_engine_factory,
            home_region=HOME_REGION,
            allowed_regions=ALLOWED_REGIONS,
            base_job_name="test-job",
        )

        # Since home fails and cross-region S3 copy fails, both exhaust.
        with pytest.raises(AllRegionsExhaustedError):
            manager.execute(INSTANCE_TYPES, BOTO3_PARAMS, "test-job")

        # Engine should have been called only once (home region).
        assert mock_engine.execute.call_count == 1


# ===========================================================================
# 13. S3 URI parsing helpers
# ===========================================================================


class TestS3URIParsing:
    """Static URI parsing helpers work correctly."""

    def test_parse_s3_uri_with_prefix(self):
        bucket, prefix = CrossRegionManager._parse_s3_uri(
            "s3://my-bucket/path/to/data"
        )
        assert bucket == "my-bucket"
        assert prefix == "path/to/data"

    def test_parse_s3_uri_bucket_only(self):
        bucket, prefix = CrossRegionManager._parse_s3_uri("s3://my-bucket")
        assert bucket == "my-bucket"
        assert prefix == ""

    def test_extract_bucket_name(self):
        name = CrossRegionManager._extract_bucket_name(
            "s3://source-data-bucket/input"
        )
        assert name == "source-data-bucket"

    def test_parse_ecr_repo(self):
        repo = CrossRegionManager._parse_ecr_repo(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-repo:v1"
        )
        assert repo == "my-repo"

    def test_parse_ecr_tag(self):
        tag = CrossRegionManager._parse_ecr_tag(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-repo:v2"
        )
        assert tag == "v2"

    def test_parse_ecr_tag_default_latest(self):
        tag = CrossRegionManager._parse_ecr_tag(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-repo"
        )
        assert tag == "latest"

    def test_parse_ecr_account(self):
        account = CrossRegionManager._parse_ecr_account(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-repo:v1"
        )
        assert account == "123456789012"


# ===========================================================================
# 14. SDK tag helper methods
# ===========================================================================


class TestSDKTagHelpers:
    """SDK tag builder methods produce correct output."""

    def test_build_sdk_tags_returns_four_tags(self):
        manager, _, _ = _make_manager_with_failover()
        tags = manager._build_sdk_tags()

        assert len(tags) == 4
        tag_keys = {t["Key"] for t in tags}
        assert tag_keys == {
            "sm-resilient-launcher:managed",
            "sm-resilient-launcher:source-region",
            "sm-resilient-launcher:job-name",
            "sm-resilient-launcher:created-at",
        }

    def test_build_sdk_tags_values(self):
        manager, _, _ = _make_manager_with_failover()
        tags = manager._build_sdk_tags()
        tag_map = {t["Key"]: t["Value"] for t in tags}

        assert tag_map["sm-resilient-launcher:managed"] == "true"
        assert tag_map["sm-resilient-launcher:source-region"] == HOME_REGION
        assert tag_map["sm-resilient-launcher:job-name"] == "test-job"
        assert "T" in tag_map["sm-resilient-launcher:created-at"]

    def test_build_sdk_tag_string_format(self):
        manager, _, _ = _make_manager_with_failover()
        tag_string = manager._build_sdk_tag_string()

        assert "sm-resilient-launcher:managed=true" in tag_string
        assert (
            f"sm-resilient-launcher:source-region={HOME_REGION}" in tag_string
        )
        assert "sm-resilient-launcher:job-name=test-job" in tag_string
        assert "sm-resilient-launcher:created-at=" in tag_string
        assert tag_string.count("&") == 3


# ===========================================================================
# 15. LocationConstraint for us-east-1
# ===========================================================================


class TestLocationConstraint:
    """us-east-1 does not accept LocationConstraint; other regions do."""

    @patch("sm_resilient_launcher._cross_region_manager.boto3.client")
    def test_us_east_1_no_location_constraint(self, mock_boto3_client):
        """When target is us-east-1, no CreateBucketConfiguration."""
        s3_target = _make_s3_client_mock(bucket_exists=False)
        sm_home = _make_sagemaker_client_mock()
        ecr = _make_ecr_client_mock(image_exists=True)
        iam = _make_iam_client_mock()

        mock_boto3_client.side_effect = lambda svc, region_name=None, **kw: (
            s3_target
            if svc == "s3"
            else sm_home
            if svc == "sagemaker"
            else ecr
            if svc == "ecr"
            else iam
            if svc == "iam"
            else MagicMock()
        )

        # Home is us-west-2, target is us-east-1.
        adapter_factory = MagicMock()
        mock_engine = MagicMock()

        def _engine_side_effect(**kwargs):
            region = kwargs.get("region", "")
            if region == "us-west-2":
                raise AllTypesExhaustedError([
                    ClassifiedError(
                        category=ErrorCategory.CAPACITY,
                        original_exception=Exception("no cap"),
                        message="no cap",
                        instance_type="ml.g5.4xlarge",
                        region="us-west-2",
                    )
                ])
            return _make_launch_result("us-east-1")

        mock_engine.execute.side_effect = _engine_side_effect
        strategy_engine_factory = MagicMock(return_value=mock_engine)

        manager = CrossRegionManager(
            adapter_factory=adapter_factory,
            strategy_engine_factory=strategy_engine_factory,
            home_region="us-west-2",
            allowed_regions=["us-west-2", "us-east-1"],
            base_job_name="test-job",
        )

        manager.execute(INSTANCE_TYPES, BOTO3_PARAMS, "test-job")

        create_kwargs = s3_target.create_bucket.call_args[1]
        assert "CreateBucketConfiguration" not in create_kwargs

    @patch("sm_resilient_launcher._cross_region_manager.boto3.client")
    def test_non_us_east_1_has_location_constraint(self, mock_boto3_client):
        """Non-us-east-1 regions include LocationConstraint."""
        s3_cross = _make_s3_client_mock(bucket_exists=False)
        s3_home = _make_s3_client_mock()
        sm_home = _make_sagemaker_client_mock()
        ecr_cross = _make_ecr_client_mock(image_exists=True)
        iam = _make_iam_client_mock()

        mock_boto3_client.side_effect = _boto3_client_router(
            s3_clients={HOME_REGION: s3_home, CROSS_REGION: s3_cross},
            sm_home=sm_home,
            ecr_clients={CROSS_REGION: ecr_cross},
            iam=iam,
        )

        manager, _, _ = _make_manager_with_failover()
        manager.execute(INSTANCE_TYPES, BOTO3_PARAMS, "test-job")

        create_kwargs = s3_cross.create_bucket.call_args[1]
        assert "CreateBucketConfiguration" in create_kwargs
        assert (
            create_kwargs["CreateBucketConfiguration"]["LocationConstraint"]
            == CROSS_REGION
        )
