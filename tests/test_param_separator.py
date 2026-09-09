"""Unit tests for ParamSeparator.separate().

Property 1: Parameter separation round trip
    For any call to launch_create_transform_job with an arbitrary parameter
    dictionary, ParamSeparator.separate() SHALL extract exactly the SDK-specific
    parameters (Strategy, AllowedRegions, WorkloadDescription) into sdk_params,
    pass all remaining parameters -- including unrecognized keys -- into
    boto3_params unchanged, and never raise an error for unrecognized parameter
    names.

Validates: Requirements 1.6, 1.7, 4.1, 4.2, 4.3
"""

import pytest

from sm_resilient_launcher._param_separator import ParamSeparator


# ---------------------------------------------------------------------------
# 1. SDK params are extracted into sdk_params
# ---------------------------------------------------------------------------

class TestSdkParamExtraction:
    """SDK-specific params (Strategy, AllowedRegions, WorkloadDescription)
    are extracted into sdk_params."""

    def test_strategy_extracted(self):
        kwargs = {
            "TransformJobName": "job-1",
            "Strategy": "sequential",
        }
        boto3_params, sdk_params, _ = ParamSeparator.separate(kwargs)

        assert sdk_params == {"Strategy": "sequential"}
        assert "Strategy" not in boto3_params

    def test_allowed_regions_extracted(self):
        kwargs = {
            "TransformJobName": "job-1",
            "AllowedRegions": ["us-east-1", "us-west-2"],
        }
        boto3_params, sdk_params, _ = ParamSeparator.separate(kwargs)

        assert sdk_params == {"AllowedRegions": ["us-east-1", "us-west-2"]}
        assert "AllowedRegions" not in boto3_params

    def test_workload_description_extracted(self):
        kwargs = {
            "TransformJobName": "job-1",
            "WorkloadDescription": "ResNet-50 inference",
        }
        boto3_params, sdk_params, _ = ParamSeparator.separate(kwargs)

        assert sdk_params == {"WorkloadDescription": "ResNet-50 inference"}
        assert "WorkloadDescription" not in boto3_params

    def test_all_sdk_params_extracted_together(self):
        kwargs = {
            "TransformJobName": "job-1",
            "ModelName": "my-model",
            "Strategy": "race",
            "AllowedRegions": ["us-east-1"],
            "WorkloadDescription": "LLM inference batch",
        }
        boto3_params, sdk_params, _ = ParamSeparator.separate(kwargs)

        assert sdk_params == {
            "Strategy": "race",
            "AllowedRegions": ["us-east-1"],
            "WorkloadDescription": "LLM inference batch",
        }
        assert "Strategy" not in boto3_params
        assert "AllowedRegions" not in boto3_params
        assert "WorkloadDescription" not in boto3_params


# ---------------------------------------------------------------------------
# 2. boto3 params include everything that is NOT an SDK param
# ---------------------------------------------------------------------------

class TestBoto3ParamForwarding:
    """All non-SDK params are forwarded into boto3_params unchanged."""

    def test_standard_boto3_params_forwarded(self):
        kwargs = {
            "TransformJobName": "job-1",
            "ModelName": "my-model",
            "TransformInput": {"DataSource": {"S3DataSource": {"S3Uri": "s3://bucket/input"}}},
            "TransformOutput": {"S3OutputPath": "s3://bucket/output"},
            "TransformResources": {"InstanceType": "ml.g5.4xlarge", "InstanceCount": 1},
        }
        boto3_params, sdk_params, _ = ParamSeparator.separate(kwargs)

        assert boto3_params == kwargs
        assert sdk_params == {}

    def test_boto3_param_values_unchanged(self):
        """Param values must be forwarded exactly as provided -- no
        mutation, type conversion, or key renaming."""
        transform_input = {
            "DataSource": {
                "S3DataSource": {
                    "S3Uri": "s3://my-bucket/data",
                    "S3DataType": "S3Prefix",
                }
            },
            "ContentType": "text/csv",
        }
        kwargs = {
            "TransformJobName": "exact-values",
            "TransformInput": transform_input,
            "TransformResources": {"InstanceType": "ml.g5.xlarge", "InstanceCount": 2},
            "Strategy": "sequential",
        }
        boto3_params, _, _ = ParamSeparator.separate(kwargs)

        assert boto3_params["TransformInput"] is transform_input
        assert boto3_params["TransformResources"]["InstanceCount"] == 2


# ---------------------------------------------------------------------------
# 3. Unrecognized params are NOT rejected — they appear in boto3_params
# ---------------------------------------------------------------------------

class TestUnrecognizedParamPassthrough:
    """Unrecognized / future params must pass through into boto3_params
    without error (Requirement 4.1, 4.3)."""

    def test_unknown_param_forwarded(self):
        kwargs = {
            "TransformJobName": "job-1",
            "TransformResources": {"InstanceType": "ml.g5.4xlarge", "InstanceCount": 1},
            "FutureApiParam": {"SomeKey": "SomeValue"},
        }
        boto3_params, sdk_params, _ = ParamSeparator.separate(kwargs)

        assert "FutureApiParam" in boto3_params
        assert boto3_params["FutureApiParam"] == {"SomeKey": "SomeValue"}
        assert sdk_params == {}

    def test_multiple_unknown_params_forwarded(self):
        kwargs = {
            "TransformJobName": "job-1",
            "NewParam1": 42,
            "NewParam2": [1, 2, 3],
            "NewParam3": True,
        }
        boto3_params, sdk_params, _ = ParamSeparator.separate(kwargs)

        assert boto3_params["NewParam1"] == 42
        assert boto3_params["NewParam2"] == [1, 2, 3]
        assert boto3_params["NewParam3"] is True
        assert sdk_params == {}

    def test_unrecognized_params_alongside_sdk_params(self):
        kwargs = {
            "TransformJobName": "job-1",
            "Strategy": "passthrough",
            "ExperimentalFlag": "enabled",
        }
        boto3_params, sdk_params, _ = ParamSeparator.separate(kwargs)

        assert "ExperimentalFlag" in boto3_params
        assert boto3_params["ExperimentalFlag"] == "enabled"
        assert sdk_params == {"Strategy": "passthrough"}


# ---------------------------------------------------------------------------
# 4. is_list_mode=True when TransformResources.InstanceType is a list
# ---------------------------------------------------------------------------

class TestListModeDetection:
    """is_list_mode flag correctly detects list vs scalar InstanceType."""

    def test_list_instance_type(self):
        kwargs = {
            "TransformResources": {
                "InstanceType": ["ml.g5.4xlarge", "ml.g4dn.2xlarge"],
                "InstanceCount": 1,
            },
        }
        _, _, is_list_mode = ParamSeparator.separate(kwargs)
        assert is_list_mode is True

    def test_single_element_list(self):
        kwargs = {
            "TransformResources": {
                "InstanceType": ["ml.g5.4xlarge"],
                "InstanceCount": 1,
            },
        }
        _, _, is_list_mode = ParamSeparator.separate(kwargs)
        assert is_list_mode is True


# ---------------------------------------------------------------------------
# 5. is_list_mode=False when TransformResources.InstanceType is a scalar
# ---------------------------------------------------------------------------

class TestScalarModeDetection:
    """is_list_mode=False for scalar string InstanceType."""

    def test_scalar_instance_type(self):
        kwargs = {
            "TransformResources": {
                "InstanceType": "ml.g5.4xlarge",
                "InstanceCount": 1,
            },
        }
        _, _, is_list_mode = ParamSeparator.separate(kwargs)
        assert is_list_mode is False


# ---------------------------------------------------------------------------
# 6. is_list_mode=False when TransformResources is missing entirely
# ---------------------------------------------------------------------------

class TestMissingTransformResources:
    """is_list_mode=False when TransformResources or InstanceType is absent."""

    def test_no_transform_resources(self):
        kwargs = {"TransformJobName": "job-1"}
        _, _, is_list_mode = ParamSeparator.separate(kwargs)
        assert is_list_mode is False

    def test_transform_resources_without_instance_type(self):
        kwargs = {
            "TransformResources": {"InstanceCount": 1},
        }
        _, _, is_list_mode = ParamSeparator.separate(kwargs)
        assert is_list_mode is False


# ---------------------------------------------------------------------------
# 7. Empty SDK params when none provided
# ---------------------------------------------------------------------------

class TestEmptySdkParams:
    """When no SDK-specific params are in the input, sdk_params is empty."""

    def test_no_sdk_params(self):
        kwargs = {
            "TransformJobName": "job-1",
            "ModelName": "my-model",
            "TransformResources": {"InstanceType": "ml.g5.4xlarge", "InstanceCount": 1},
        }
        _, sdk_params, _ = ParamSeparator.separate(kwargs)
        assert sdk_params == {}

    def test_empty_input(self):
        boto3_params, sdk_params, is_list_mode = ParamSeparator.separate({})
        assert boto3_params == {}
        assert sdk_params == {}
        assert is_list_mode is False


# ---------------------------------------------------------------------------
# 8. SDK params never appear in boto3_params
# ---------------------------------------------------------------------------

class TestSdkParamsNeverInBoto3:
    """SDK params must never leak into boto3_params, regardless of
    how many SDK params are provided."""

    def test_sdk_params_excluded_from_boto3(self):
        kwargs = {
            "TransformJobName": "job-1",
            "ModelName": "model-1",
            "TransformResources": {"InstanceType": "ml.g5.4xlarge", "InstanceCount": 1},
            "Strategy": "race",
            "AllowedRegions": ["us-east-1", "eu-west-1"],
            "WorkloadDescription": "batch inference",
        }
        boto3_params, _, _ = ParamSeparator.separate(kwargs)

        for sdk_key in ParamSeparator.SDK_PARAMS:
            assert sdk_key not in boto3_params, (
                f"SDK param '{sdk_key}' must not appear in boto3_params"
            )

    def test_partial_sdk_params_excluded(self):
        kwargs = {
            "TransformJobName": "job-1",
            "WorkloadDescription": "ResNet-50 fp32",
        }
        boto3_params, _, _ = ParamSeparator.separate(kwargs)

        for sdk_key in ParamSeparator.SDK_PARAMS:
            assert sdk_key not in boto3_params
