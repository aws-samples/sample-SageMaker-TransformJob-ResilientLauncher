"""Unit tests for prepare_regions utility.

Property 19: Readiness Report covers all target regions
    For any list of N target regions passed to prepare_regions(),
    the returned ReadinessReport SHALL contain exactly N
    RegionReadiness entries, each with s3_status, ecr_status,
    and iam_status.  The function SHALL NOT modify IAM roles
    or replicate ECR images.

Validates: Requirements 16.1, 16.2, 16.3, 16.4, 16.5, 16.6
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from sm_resilient_launcher._prepare_regions import (
    ReadinessReport,
    RegionReadiness,
    prepare_regions,
    _build_sdk_tags,
    _check_ecr_image,
    _check_or_create_mirror_bucket,
    _extract_bucket_name,
    _parse_ecr_account,
    _parse_ecr_repo,
    _parse_ecr_tag,
    _resolve_model_artifact,
    _resolve_model_image,
    _validate_iam,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODEL_IMAGE_URI = (
    "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-repo:v1"
)
MODEL_ARTIFACT_URI = "s3://my-model-bucket/models/resnet/model.tar.gz"
HOME_REGION = "us-east-1"
ROLE_ARN = "arn:aws:iam::123456789012:role/SageMakerRole"


def _mock_describe_model(model_name: str = "my-model") -> dict:
    """Standard DescribeModel response fixture."""
    return {
        "PrimaryContainer": {
            "Image": MODEL_IMAGE_URI,
            "ModelDataUrl": MODEL_ARTIFACT_URI,
        }
    }


# ---------------------------------------------------------------------------
# 1. RegionReadiness dataclass
# ---------------------------------------------------------------------------


class TestRegionReadiness:
    """RegionReadiness carries required and optional fields."""

    def test_required_fields(self):
        r = RegionReadiness(
            region="us-west-2",
            s3_status="ready",
            ecr_status="ready",
            iam_status="valid",
        )
        assert r.region == "us-west-2"
        assert r.s3_status == "ready"
        assert r.ecr_status == "ready"
        assert r.iam_status == "valid"
        assert r.ecr_commands is None
        assert r.iam_details is None

    def test_optional_fields(self):
        r = RegionReadiness(
            region="eu-west-1",
            s3_status="created",
            ecr_status="missing",
            iam_status="warning",
            ecr_commands="docker pull ...",
            iam_details="Policy gap found",
        )
        assert r.ecr_commands == "docker pull ..."
        assert r.iam_details == "Policy gap found"


# ---------------------------------------------------------------------------
# 2. ReadinessReport dataclass
# ---------------------------------------------------------------------------


class TestReadinessReport:
    """ReadinessReport holds a list of RegionReadiness entries."""

    def test_default_empty(self):
        report = ReadinessReport()
        assert report.regions == []

    def test_contains_entries(self):
        r1 = RegionReadiness(
            region="us-west-2",
            s3_status="ready",
            ecr_status="ready",
            iam_status="valid",
        )
        report = ReadinessReport(regions=[r1])
        assert len(report.regions) == 1
        assert report.regions[0].region == "us-west-2"


# ---------------------------------------------------------------------------
# 3. _resolve_model_image
# ---------------------------------------------------------------------------


class TestResolveModelImage:
    """_resolve_model_image calls DescribeModel and extracts Image URI."""

    def test_returns_image_uri(self):
        mock_sm = MagicMock()
        mock_sm.describe_model.return_value = _mock_describe_model()
        with patch("boto3.client", return_value=mock_sm):
            result = _resolve_model_image("us-east-1", "my-model")
        assert result == MODEL_IMAGE_URI

    def test_returns_empty_on_failure(self):
        mock_sm = MagicMock()
        mock_sm.describe_model.side_effect = Exception("not found")
        with patch("boto3.client", return_value=mock_sm):
            result = _resolve_model_image("us-east-1", "bad-model")
        assert result == ""

    def test_returns_empty_when_no_primary_container(self):
        mock_sm = MagicMock()
        mock_sm.describe_model.return_value = {}
        with patch("boto3.client", return_value=mock_sm):
            result = _resolve_model_image("us-east-1", "my-model")
        assert result == ""


# ---------------------------------------------------------------------------
# 4. _resolve_model_artifact
# ---------------------------------------------------------------------------


class TestResolveModelArtifact:
    """_resolve_model_artifact calls DescribeModel and extracts ModelDataUrl."""

    def test_returns_artifact_uri(self):
        mock_sm = MagicMock()
        mock_sm.describe_model.return_value = _mock_describe_model()
        with patch("boto3.client", return_value=mock_sm):
            result = _resolve_model_artifact("us-east-1", "my-model")
        assert result == MODEL_ARTIFACT_URI

    def test_returns_empty_on_failure(self):
        mock_sm = MagicMock()
        mock_sm.describe_model.side_effect = Exception("error")
        with patch("boto3.client", return_value=mock_sm):
            result = _resolve_model_artifact("us-east-1", "bad-model")
        assert result == ""


# ---------------------------------------------------------------------------
# 5. _extract_bucket_name
# ---------------------------------------------------------------------------


class TestExtractBucketName:
    """_extract_bucket_name parses s3://bucket/prefix URIs."""

    def test_with_prefix(self):
        assert _extract_bucket_name("s3://my-bucket/path/to/data") == "my-bucket"

    def test_without_prefix(self):
        assert _extract_bucket_name("s3://my-bucket") == "my-bucket"

    def test_trailing_slash(self):
        assert _extract_bucket_name("s3://my-bucket/") == "my-bucket"


# ---------------------------------------------------------------------------
# 6. ECR URI parsing helpers
# ---------------------------------------------------------------------------


class TestEcrParsing:
    """ECR URI parsing helpers extract repo, tag, and account."""

    def test_parse_ecr_repo(self):
        assert _parse_ecr_repo(MODEL_IMAGE_URI) == "my-repo"

    def test_parse_ecr_tag(self):
        assert _parse_ecr_tag(MODEL_IMAGE_URI) == "v1"

    def test_parse_ecr_tag_defaults_to_latest(self):
        uri = "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-repo"
        assert _parse_ecr_tag(uri) == "latest"

    def test_parse_ecr_account(self):
        assert _parse_ecr_account(MODEL_IMAGE_URI) == "123456789012"


# ---------------------------------------------------------------------------
# 7. _build_sdk_tags
# ---------------------------------------------------------------------------


class TestBuildSdkTags:
    """_build_sdk_tags produces the four required SDK tags."""

    def test_contains_four_tags(self):
        tags = _build_sdk_tags("us-east-1", "my-job")
        assert len(tags) == 4

    def test_tag_keys_and_values(self):
        tags = _build_sdk_tags("us-east-1", "my-job")
        tag_map = {t["Key"]: t["Value"] for t in tags}

        assert tag_map["sm-resilient-launcher:managed"] == "true"
        assert tag_map["sm-resilient-launcher:source-region"] == "us-east-1"
        assert tag_map["sm-resilient-launcher:job-name"] == "my-job"
        assert "sm-resilient-launcher:created-at" in tag_map
        # created-at should be an ISO 8601 timestamp.
        assert "T" in tag_map["sm-resilient-launcher:created-at"]


# ---------------------------------------------------------------------------
# 8. _check_or_create_mirror_bucket — bucket exists
# ---------------------------------------------------------------------------


class TestCheckOrCreateMirrorBucketExists:
    """When the mirror bucket exists, return 'ready' without creating."""

    def test_returns_ready(self):
        mock_s3 = MagicMock()
        mock_s3.head_bucket.return_value = {}  # No exception = exists
        with patch("boto3.client", return_value=mock_s3):
            status = _check_or_create_mirror_bucket(
                "my-bucket", "us-west-2", "us-east-1", "my-job"
            )
        assert status == "ready"
        mock_s3.create_bucket.assert_not_called()
        mock_s3.put_bucket_tagging.assert_not_called()


# ---------------------------------------------------------------------------
# 9. _check_or_create_mirror_bucket — bucket does not exist
# ---------------------------------------------------------------------------


class TestCheckOrCreateMirrorBucketCreated:
    """When the mirror bucket does not exist, create it with SDK tags."""

    def test_creates_bucket_with_location_constraint(self):
        mock_s3 = MagicMock()
        mock_s3.head_bucket.side_effect = Exception("404")
        with patch("boto3.client", return_value=mock_s3):
            status = _check_or_create_mirror_bucket(
                "my-bucket", "us-west-2", "us-east-1", "my-job"
            )
        assert status == "created"
        mock_s3.create_bucket.assert_called_once()
        create_kwargs = mock_s3.create_bucket.call_args[1]
        assert create_kwargs["Bucket"] == "my-bucket-us-west-2"
        assert create_kwargs["CreateBucketConfiguration"] == {
            "LocationConstraint": "us-west-2"
        }
        mock_s3.put_bucket_tagging.assert_called_once()

    def test_us_east_1_no_location_constraint(self):
        mock_s3 = MagicMock()
        mock_s3.head_bucket.side_effect = Exception("404")
        with patch("boto3.client", return_value=mock_s3):
            status = _check_or_create_mirror_bucket(
                "my-bucket", "us-east-1", "us-west-2", "my-job"
            )
        assert status == "created"
        create_kwargs = mock_s3.create_bucket.call_args[1]
        assert "CreateBucketConfiguration" not in create_kwargs

    def test_sdk_tags_applied_at_creation(self):
        mock_s3 = MagicMock()
        mock_s3.head_bucket.side_effect = Exception("404")
        with patch("boto3.client", return_value=mock_s3):
            _check_or_create_mirror_bucket(
                "my-bucket", "eu-west-1", "us-east-1", "my-job"
            )
        tagging_kwargs = mock_s3.put_bucket_tagging.call_args[1]
        assert tagging_kwargs["Bucket"] == "my-bucket-eu-west-1"
        tag_keys = {
            t["Key"] for t in tagging_kwargs["Tagging"]["TagSet"]
        }
        assert "sm-resilient-launcher:managed" in tag_keys
        assert "sm-resilient-launcher:source-region" in tag_keys
        assert "sm-resilient-launcher:job-name" in tag_keys
        assert "sm-resilient-launcher:created-at" in tag_keys


# ---------------------------------------------------------------------------
# 10. _check_ecr_image — image exists
# ---------------------------------------------------------------------------


class TestCheckEcrImageReady:
    """When the ECR image exists in the target region, return 'ready'."""

    def test_returns_ready(self):
        mock_ecr = MagicMock()
        mock_ecr.describe_images.return_value = {"imageDetails": [{}]}
        with patch("boto3.client", return_value=mock_ecr):
            status, commands = _check_ecr_image(
                MODEL_IMAGE_URI, "us-west-2", "us-east-1"
            )
        assert status == "ready"
        assert commands is None


# ---------------------------------------------------------------------------
# 11. _check_ecr_image — image missing
# ---------------------------------------------------------------------------


class TestCheckEcrImageMissing:
    """When the ECR image is missing, return 'missing' with CLI commands."""

    def test_returns_missing_with_commands(self):
        mock_ecr = MagicMock()
        mock_ecr.describe_images.side_effect = Exception("not found")
        with patch("boto3.client", return_value=mock_ecr):
            status, commands = _check_ecr_image(
                MODEL_IMAGE_URI, "us-west-2", "us-east-1"
            )
        assert status == "missing"
        assert commands is not None
        assert "us-west-2" in commands
        assert "my-repo" in commands
        assert "docker pull" in commands
        assert "docker push" in commands

    def test_cli_commands_contain_source_account(self):
        mock_ecr = MagicMock()
        mock_ecr.describe_images.side_effect = Exception("not found")
        with patch("boto3.client", return_value=mock_ecr):
            _, commands = _check_ecr_image(
                MODEL_IMAGE_URI, "eu-west-1", "us-east-1"
            )
        assert "123456789012" in commands


# ---------------------------------------------------------------------------
# 12. _validate_iam — role exists with source bucket
# ---------------------------------------------------------------------------


class TestValidateIamWithBucket:
    """When the role exists and there is a source bucket, warn about
    potential policy gap for the mirror bucket."""

    def test_returns_warning_with_mirror_bucket(self):
        mock_iam = MagicMock()
        mock_iam.get_role.return_value = {"Role": {"RoleName": "SageMakerRole"}}
        with patch("boto3.client", return_value=mock_iam):
            status, details = _validate_iam(
                ROLE_ARN, "my-bucket", "us-west-2"
            )
        assert status == "warning"
        assert details is not None
        assert "my-bucket-us-west-2" in details


# ---------------------------------------------------------------------------
# 13. _validate_iam — role does not exist
# ---------------------------------------------------------------------------


class TestValidateIamRoleNotFound:
    """When the role cannot be validated, return warning."""

    def test_returns_warning_on_failure(self):
        mock_iam = MagicMock()
        mock_iam.get_role.side_effect = Exception("NoSuchEntity")
        with patch("boto3.client", return_value=mock_iam):
            status, details = _validate_iam(
                ROLE_ARN, "my-bucket", "us-west-2"
            )
        assert status == "warning"
        assert "Role validation failed" in details


# ---------------------------------------------------------------------------
# 14. _validate_iam — no source bucket
# ---------------------------------------------------------------------------


class TestValidateIamNoBucket:
    """When there is no source bucket, return valid (no mirror to check)."""

    def test_returns_valid_when_no_bucket(self):
        mock_iam = MagicMock()
        mock_iam.get_role.return_value = {"Role": {"RoleName": "SageMakerRole"}}
        with patch("boto3.client", return_value=mock_iam):
            status, details = _validate_iam(ROLE_ARN, "", "us-west-2")
        assert status == "valid"
        assert details is None


# ---------------------------------------------------------------------------
# 15. prepare_regions — full integration (happy path)
# ---------------------------------------------------------------------------


class TestPrepareRegionsHappyPath:
    """prepare_regions returns N entries for N regions (Req 16.1)."""

    def _mock_boto3_client(self, service, region_name=None):
        """Route boto3.client calls to the correct mock."""
        if service == "sagemaker":
            mock = MagicMock()
            mock.describe_model.return_value = _mock_describe_model()
            return mock
        if service == "s3":
            mock = MagicMock()
            mock.head_bucket.return_value = {}  # bucket exists
            return mock
        if service == "ecr":
            mock = MagicMock()
            mock.describe_images.return_value = {"imageDetails": [{}]}
            return mock
        if service == "iam":
            mock = MagicMock()
            mock.get_role.return_value = {"Role": {"RoleName": "SageMakerRole"}}
            return mock
        return MagicMock()

    def test_two_regions_two_entries(self):
        with patch("boto3.client", side_effect=self._mock_boto3_client):
            report = prepare_regions(
                home_region="us-east-1",
                target_regions=["us-west-2", "eu-west-1"],
                model_name="my-model",
                role_arn=ROLE_ARN,
            )
        assert isinstance(report, ReadinessReport)
        assert len(report.regions) == 2
        assert report.regions[0].region == "us-west-2"
        assert report.regions[1].region == "eu-west-1"

    def test_s3_ready_when_buckets_exist(self):
        with patch("boto3.client", side_effect=self._mock_boto3_client):
            report = prepare_regions(
                home_region="us-east-1",
                target_regions=["us-west-2"],
                model_name="my-model",
                role_arn=ROLE_ARN,
            )
        assert report.regions[0].s3_status == "ready"

    def test_ecr_ready_when_image_exists(self):
        with patch("boto3.client", side_effect=self._mock_boto3_client):
            report = prepare_regions(
                home_region="us-east-1",
                target_regions=["us-west-2"],
                model_name="my-model",
                role_arn=ROLE_ARN,
            )
        assert report.regions[0].ecr_status == "ready"
        assert report.regions[0].ecr_commands is None

    def test_iam_warning_with_bucket_present(self):
        """When source bucket exists, IAM check warns about mirror bucket."""
        with patch("boto3.client", side_effect=self._mock_boto3_client):
            report = prepare_regions(
                home_region="us-east-1",
                target_regions=["us-west-2"],
                model_name="my-model",
                role_arn=ROLE_ARN,
            )
        assert report.regions[0].iam_status == "warning"
        assert "my-model-bucket-us-west-2" in report.regions[0].iam_details


# ---------------------------------------------------------------------------
# 16. prepare_regions — S3 bucket creation path
# ---------------------------------------------------------------------------


class TestPrepareRegionsBucketCreation:
    """When the mirror bucket does not exist, prepare_regions creates it."""

    def _mock_boto3_client(self, service, region_name=None):
        if service == "sagemaker":
            mock = MagicMock()
            mock.describe_model.return_value = _mock_describe_model()
            return mock
        if service == "s3":
            mock = MagicMock()
            mock.head_bucket.side_effect = Exception("404")
            return mock
        if service == "ecr":
            mock = MagicMock()
            mock.describe_images.return_value = {"imageDetails": [{}]}
            return mock
        if service == "iam":
            mock = MagicMock()
            mock.get_role.return_value = {"Role": {"RoleName": "SageMakerRole"}}
            return mock
        return MagicMock()

    def test_s3_status_created(self):
        with patch("boto3.client", side_effect=self._mock_boto3_client):
            report = prepare_regions(
                home_region="us-east-1",
                target_regions=["us-west-2"],
                model_name="my-model",
                role_arn=ROLE_ARN,
            )
        assert report.regions[0].s3_status == "created"


# ---------------------------------------------------------------------------
# 17. prepare_regions — ECR missing path
# ---------------------------------------------------------------------------


class TestPrepareRegionsEcrMissing:
    """When the ECR image is missing, report includes CLI commands."""

    def _mock_boto3_client(self, service, region_name=None):
        if service == "sagemaker":
            mock = MagicMock()
            mock.describe_model.return_value = _mock_describe_model()
            return mock
        if service == "s3":
            mock = MagicMock()
            mock.head_bucket.return_value = {}
            return mock
        if service == "ecr":
            mock = MagicMock()
            mock.describe_images.side_effect = Exception("not found")
            return mock
        if service == "iam":
            mock = MagicMock()
            mock.get_role.return_value = {"Role": {"RoleName": "SageMakerRole"}}
            return mock
        return MagicMock()

    def test_ecr_missing_with_commands(self):
        with patch("boto3.client", side_effect=self._mock_boto3_client):
            report = prepare_regions(
                home_region="us-east-1",
                target_regions=["eu-west-1"],
                model_name="my-model",
                role_arn=ROLE_ARN,
            )
        assert report.regions[0].ecr_status == "missing"
        assert report.regions[0].ecr_commands is not None
        assert "docker pull" in report.regions[0].ecr_commands


# ---------------------------------------------------------------------------
# 18. prepare_regions — empty target regions
# ---------------------------------------------------------------------------


class TestPrepareRegionsEmpty:
    """When no target regions are provided, report is empty."""

    def test_empty_report(self):
        with patch("boto3.client"):
            report = prepare_regions(
                home_region="us-east-1",
                target_regions=[],
                model_name="my-model",
                role_arn=ROLE_ARN,
            )
        assert len(report.regions) == 0


# ---------------------------------------------------------------------------
# 19. prepare_regions — no model artifact (no source bucket)
# ---------------------------------------------------------------------------


class TestPrepareRegionsNoModelArtifact:
    """When model artifact resolution fails, S3 check is skipped."""

    def _mock_boto3_client(self, service, region_name=None):
        if service == "sagemaker":
            mock = MagicMock()
            mock.describe_model.side_effect = Exception("model not found")
            return mock
        if service == "iam":
            mock = MagicMock()
            mock.get_role.return_value = {"Role": {"RoleName": "SageMakerRole"}}
            return mock
        return MagicMock()

    def test_s3_status_remains_ready(self):
        """Without a source bucket, S3 mirror check is skipped."""
        with patch("boto3.client", side_effect=self._mock_boto3_client):
            report = prepare_regions(
                home_region="us-east-1",
                target_regions=["us-west-2"],
                model_name="bad-model",
                role_arn=ROLE_ARN,
            )
        # S3 check skipped → default "ready"
        assert report.regions[0].s3_status == "ready"
        # ECR check skipped → default "ready" (no image URI)
        assert report.regions[0].ecr_status == "ready"


# ---------------------------------------------------------------------------
# 20. SDK never modifies IAM (Req 16.6 — verify no IAM write calls)
# ---------------------------------------------------------------------------


class TestPrepareRegionsNeverModifiesIam:
    """prepare_regions MUST NOT call any IAM write operations."""

    def _mock_boto3_client(self, service, region_name=None):
        if service == "sagemaker":
            mock = MagicMock()
            mock.describe_model.return_value = _mock_describe_model()
            return mock
        if service == "s3":
            mock = MagicMock()
            mock.head_bucket.return_value = {}
            return mock
        if service == "ecr":
            mock = MagicMock()
            mock.describe_images.return_value = {"imageDetails": [{}]}
            return mock
        if service == "iam":
            mock = MagicMock()
            mock.get_role.return_value = {"Role": {"RoleName": "SageMakerRole"}}
            return mock
        return MagicMock()

    def test_no_iam_writes(self):
        iam_mock = MagicMock()
        iam_mock.get_role.return_value = {"Role": {"RoleName": "SageMakerRole"}}

        clients = {}

        def client_factory(service, region_name=None):
            if service == "iam":
                return iam_mock
            return self._mock_boto3_client(service, region_name)

        with patch("boto3.client", side_effect=client_factory):
            prepare_regions(
                home_region="us-east-1",
                target_regions=["us-west-2"],
                model_name="my-model",
                role_arn=ROLE_ARN,
            )

        # IAM write operations must never be called.
        iam_mock.create_role.assert_not_called()
        iam_mock.put_role_policy.assert_not_called()
        iam_mock.attach_role_policy.assert_not_called()
        iam_mock.update_role.assert_not_called()
        iam_mock.delete_role.assert_not_called()


# ---------------------------------------------------------------------------
# 21. SDK never replicates ECR images (Req 16.6)
# ---------------------------------------------------------------------------


class TestPrepareRegionsNeverReplicatesEcr:
    """prepare_regions MUST NOT call ECR write operations to replicate."""

    def test_no_ecr_writes(self):
        ecr_mock = MagicMock()
        ecr_mock.describe_images.side_effect = Exception("not found")

        def client_factory(service, region_name=None):
            if service == "ecr":
                return ecr_mock
            if service == "sagemaker":
                mock = MagicMock()
                mock.describe_model.return_value = _mock_describe_model()
                return mock
            if service == "s3":
                mock = MagicMock()
                mock.head_bucket.return_value = {}
                return mock
            if service == "iam":
                mock = MagicMock()
                mock.get_role.return_value = {"Role": {"RoleName": "SageMakerRole"}}
                return mock
            return MagicMock()

        with patch("boto3.client", side_effect=client_factory):
            report = prepare_regions(
                home_region="us-east-1",
                target_regions=["us-west-2"],
                model_name="my-model",
                role_arn=ROLE_ARN,
            )

        # ECR write operations must never be called.
        ecr_mock.create_repository.assert_not_called()
        ecr_mock.put_image.assert_not_called()
        ecr_mock.batch_delete_image.assert_not_called()
        # But the report should indicate the image is missing.
        assert report.regions[0].ecr_status == "missing"
