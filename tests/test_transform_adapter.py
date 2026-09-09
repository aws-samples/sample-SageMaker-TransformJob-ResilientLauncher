"""Unit tests for TransformJobAdapter.

Property 13: Instance type scalar substitution
    For any instance type selected from the priority list,
    TransformJobAdapter.create_job() SHALL set
    TransformResources.InstanceType to that type as a scalar string
    in the boto3 call -- never a list.  All other caller parameters
    are preserved unchanged.

Validates: Requirements 13.1, 13.2, 14.2
"""

from unittest.mock import MagicMock, patch

import pytest

from sm_resilient_launcher._transform_adapter import TransformJobAdapter


@pytest.fixture()
def mock_client():
    """Return a mocked SageMaker client."""
    return MagicMock()


@pytest.fixture()
def adapter(mock_client):
    """Return a TransformJobAdapter wired to the mocked client."""
    with patch("boto3.client", return_value=mock_client):
        return TransformJobAdapter(region="us-east-1")


# ---------------------------------------------------------------------------
# 1. create_job substitutes InstanceType as a SCALAR string
# ---------------------------------------------------------------------------

class TestCreateJobScalarSubstitution:
    """create_job must always pass InstanceType as a scalar string."""

    def test_instance_type_set_as_scalar(self, adapter, mock_client):
        boto3_params = {
            "ModelName": "my-model",
            "TransformInput": {"DataSource": {"S3DataSource": {"S3Uri": "s3://bucket/input"}}},
            "TransformOutput": {"S3OutputPath": "s3://bucket/output"},
            "TransformResources": {"InstanceCount": 1},
        }

        adapter.create_job("job-1", "ml.g5.4xlarge", boto3_params)

        call_kwargs = mock_client.create_transform_job.call_args[1]
        assert call_kwargs["TransformResources"]["InstanceType"] == "ml.g5.4xlarge"
        assert isinstance(call_kwargs["TransformResources"]["InstanceType"], str)

    def test_instance_type_overrides_existing_value(self, adapter, mock_client):
        """If boto3_params already contains an InstanceType, the
        adapter substitutes it with the explicit argument."""
        boto3_params = {
            "TransformResources": {
                "InstanceType": "ml.m5.large",
                "InstanceCount": 1,
            },
        }

        adapter.create_job("job-1", "ml.g5.4xlarge", boto3_params)

        call_kwargs = mock_client.create_transform_job.call_args[1]
        assert call_kwargs["TransformResources"]["InstanceType"] == "ml.g5.4xlarge"


# ---------------------------------------------------------------------------
# 2. create_job sets TransformJobName from the job_name argument
# ---------------------------------------------------------------------------

class TestCreateJobSetsJobName:
    """TransformJobName in the API call comes from the job_name arg."""

    def test_job_name_set(self, adapter, mock_client):
        boto3_params = {
            "ModelName": "my-model",
            "TransformResources": {"InstanceCount": 1},
        }

        adapter.create_job("my-unique-job-42", "ml.g5.xlarge", boto3_params)

        call_kwargs = mock_client.create_transform_job.call_args[1]
        assert call_kwargs["TransformJobName"] == "my-unique-job-42"

    def test_job_name_overrides_existing(self, adapter, mock_client):
        """If boto3_params already has TransformJobName, the adapter
        overwrites it with the explicit job_name argument."""
        boto3_params = {
            "TransformJobName": "old-name",
            "TransformResources": {"InstanceCount": 1},
        }

        adapter.create_job("new-name", "ml.g5.xlarge", boto3_params)

        call_kwargs = mock_client.create_transform_job.call_args[1]
        assert call_kwargs["TransformJobName"] == "new-name"


# ---------------------------------------------------------------------------
# 3. Original boto3_params dict is NOT mutated (deep copy verification)
# ---------------------------------------------------------------------------

class TestCreateJobDeepCopy:
    """create_job must deep-copy boto3_params so the caller's dict
    is never modified."""

    def test_original_params_unchanged(self, adapter, mock_client):
        boto3_params = {
            "ModelName": "my-model",
            "TransformResources": {
                "InstanceType": "ml.m5.large",
                "InstanceCount": 1,
            },
        }
        original_instance_type = boto3_params["TransformResources"]["InstanceType"]
        original_keys = set(boto3_params.keys())

        adapter.create_job("job-1", "ml.g5.4xlarge", boto3_params)

        # Instance type in original dict must be unchanged
        assert boto3_params["TransformResources"]["InstanceType"] == original_instance_type
        # No new top-level keys added to the original
        assert set(boto3_params.keys()) == original_keys
        # TransformJobName should NOT be injected into the original
        assert "TransformJobName" not in boto3_params

    def test_nested_dict_not_mutated(self, adapter, mock_client):
        """Deep-nested dicts inside boto3_params must remain untouched."""
        transform_input = {
            "DataSource": {
                "S3DataSource": {
                    "S3Uri": "s3://bucket/input",
                    "S3DataType": "S3Prefix",
                }
            },
            "ContentType": "text/csv",
        }
        boto3_params = {
            "TransformInput": transform_input,
            "TransformResources": {"InstanceCount": 1},
        }

        adapter.create_job("job-1", "ml.g5.xlarge", boto3_params)

        assert boto3_params["TransformInput"]["DataSource"]["S3DataSource"]["S3Uri"] == "s3://bucket/input"
        assert boto3_params["TransformInput"]["ContentType"] == "text/csv"


# ---------------------------------------------------------------------------
# 4. All other caller parameters are preserved unchanged in the API call
# ---------------------------------------------------------------------------

class TestCreateJobPreservesCallerParams:
    """All caller-provided params (besides InstanceType and TransformJobName)
    must appear unchanged in the API call."""

    def test_standard_params_preserved(self, adapter, mock_client):
        boto3_params = {
            "ModelName": "resnet-50",
            "TransformInput": {
                "DataSource": {"S3DataSource": {"S3Uri": "s3://bucket/input"}},
                "ContentType": "image/jpeg",
            },
            "TransformOutput": {"S3OutputPath": "s3://bucket/output"},
            "TransformResources": {"InstanceCount": 2},
            "MaxConcurrentTransforms": 4,
            "BatchStrategy": "MultiRecord",
        }

        adapter.create_job("job-1", "ml.g5.4xlarge", boto3_params)

        call_kwargs = mock_client.create_transform_job.call_args[1]
        assert call_kwargs["ModelName"] == "resnet-50"
        assert call_kwargs["TransformInput"]["ContentType"] == "image/jpeg"
        assert call_kwargs["TransformOutput"]["S3OutputPath"] == "s3://bucket/output"
        assert call_kwargs["TransformResources"]["InstanceCount"] == 2
        assert call_kwargs["MaxConcurrentTransforms"] == 4
        assert call_kwargs["BatchStrategy"] == "MultiRecord"


# ---------------------------------------------------------------------------
# 5. Unrecognized future params are passed through to the boto3 call
# ---------------------------------------------------------------------------

class TestCreateJobUnrecognizedParams:
    """Future/unknown params must pass through to the API call unchanged."""

    def test_unknown_params_forwarded(self, adapter, mock_client):
        boto3_params = {
            "TransformResources": {"InstanceCount": 1},
            "FutureParam": {"SomeKey": "SomeValue"},
            "AnotherNewParam": 42,
        }

        adapter.create_job("job-1", "ml.g5.xlarge", boto3_params)

        call_kwargs = mock_client.create_transform_job.call_args[1]
        assert call_kwargs["FutureParam"] == {"SomeKey": "SomeValue"}
        assert call_kwargs["AnotherNewParam"] == 42


# ---------------------------------------------------------------------------
# 6. is_completed returns True only for "Completed" status
# ---------------------------------------------------------------------------

class TestIsCompleted:
    """is_completed returns True only when TransformJobStatus is 'Completed'."""

    def test_completed_returns_true(self, adapter):
        assert adapter.is_completed({"TransformJobStatus": "Completed"}) is True

    @pytest.mark.parametrize("status", ["InProgress", "Failed", "Stopped"])
    def test_non_completed_returns_false(self, adapter, status):
        assert adapter.is_completed({"TransformJobStatus": status}) is False

    def test_missing_status_returns_false(self, adapter):
        assert adapter.is_completed({}) is False


# ---------------------------------------------------------------------------
# 7. is_failed returns True for "Failed" and "Stopped"
# ---------------------------------------------------------------------------

class TestIsFailed:
    """is_failed returns True for both 'Failed' and 'Stopped' terminal states."""

    @pytest.mark.parametrize("status", ["Failed", "Stopped"])
    def test_failed_statuses_return_true(self, adapter, status):
        assert adapter.is_failed({"TransformJobStatus": status}) is True

    @pytest.mark.parametrize("status", ["InProgress", "Completed"])
    def test_non_failed_returns_false(self, adapter, status):
        assert adapter.is_failed({"TransformJobStatus": status}) is False

    def test_missing_status_returns_false(self, adapter):
        assert adapter.is_failed({}) is False


# ---------------------------------------------------------------------------
# 8. extract_error returns FailureReason or "Unknown error"
# ---------------------------------------------------------------------------

class TestExtractError:
    """extract_error returns the FailureReason field, or a default."""

    def test_returns_failure_reason(self, adapter):
        desc = {"FailureReason": "CapacityError: no ml.g5.4xlarge available"}
        assert adapter.extract_error(desc) == "CapacityError: no ml.g5.4xlarge available"

    def test_returns_unknown_when_absent(self, adapter):
        assert adapter.extract_error({}) == "Unknown error"

    def test_returns_unknown_for_empty_dict(self, adapter):
        assert adapter.extract_error({"TransformJobStatus": "Failed"}) == "Unknown error"


# ---------------------------------------------------------------------------
# 9. describe_job calls describe_transform_job with correct name
# ---------------------------------------------------------------------------

class TestDescribeJob:
    """describe_job delegates to the SageMaker describe_transform_job API."""

    def test_calls_with_correct_name(self, adapter, mock_client):
        mock_client.describe_transform_job.return_value = {
            "TransformJobStatus": "InProgress",
        }

        result = adapter.describe_job("my-job-123")

        mock_client.describe_transform_job.assert_called_once_with(
            TransformJobName="my-job-123"
        )
        assert result == {"TransformJobStatus": "InProgress"}


# ---------------------------------------------------------------------------
# 10. stop_job calls stop_transform_job with correct name
# ---------------------------------------------------------------------------

class TestStopJob:
    """stop_job delegates to the SageMaker stop_transform_job API."""

    def test_calls_with_correct_name(self, adapter, mock_client):
        adapter.stop_job("my-job-123")

        mock_client.stop_transform_job.assert_called_once_with(
            TransformJobName="my-job-123"
        )


# ---------------------------------------------------------------------------
# 11. region property returns the configured region
# ---------------------------------------------------------------------------

class TestRegionProperty:
    """The region property reflects the region passed at construction."""

    def test_region_value(self, adapter):
        assert adapter.region == "us-east-1"


# ---------------------------------------------------------------------------
# 12. create_job creates TransformResources if absent
# ---------------------------------------------------------------------------

class TestCreateJobMissingTransformResources:
    """create_job uses setdefault to create TransformResources when
    the caller's params omit it entirely."""

    def test_creates_transform_resources_when_missing(self, adapter, mock_client):
        boto3_params = {"ModelName": "my-model"}

        adapter.create_job("job-1", "ml.g5.xlarge", boto3_params)

        call_kwargs = mock_client.create_transform_job.call_args[1]
        assert call_kwargs["TransformResources"]["InstanceType"] == "ml.g5.xlarge"
        # Original params must NOT have TransformResources injected
        assert "TransformResources" not in boto3_params


# ---------------------------------------------------------------------------
# 13. create_job returns the raw API response
# ---------------------------------------------------------------------------

class TestCreateJobReturnsResponse:
    """create_job returns the raw dict from create_transform_job."""

    def test_returns_api_response(self, adapter, mock_client):
        expected = {"TransformJobArn": "arn:aws:sagemaker:us-east-1:123456789012:transform-job/job-1"}
        mock_client.create_transform_job.return_value = expected

        result = adapter.create_job("job-1", "ml.g5.xlarge", {"TransformResources": {"InstanceCount": 1}})

        assert result == expected
