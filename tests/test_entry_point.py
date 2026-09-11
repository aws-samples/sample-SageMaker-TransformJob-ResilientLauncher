"""Unit tests for launch_create_transform_job entry point.

Property 2: Passthrough transparency
    For any valid create_transform_job parameter set where Strategy is
    "passthrough", calling launch_create_transform_job SHALL produce an
    identical boto3 create_transform_job call with no retry, no error
    classification, and the return value SHALL be the unmodified boto3
    response dict.

Property 18: LaunchResult contains all required fields
    For any successful launch, the returned LaunchResult SHALL contain
    non-null values for job_name, instance_type_used, region_used,
    attempts, and strategy_used.

Validates: Requirements 1.4, 1.6, 1.7, 2.1-2.8, 4.1-4.3, 15.1-15.5
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

from sm_resilient_launcher._entry_point import launch_create_transform_job
from sm_resilient_launcher._launch_result import LaunchResult


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

MINIMAL_BOTO3_PARAMS = {
    "TransformJobName": "my-job",
    "ModelName": "my-model",
    "TransformInput": {
        "DataSource": {"S3DataSource": {"S3Uri": "s3://bucket/input"}}
    },
    "TransformOutput": {"S3OutputPath": "s3://bucket/output"},
    "TransformResources": {"InstanceType": "ml.g5.4xlarge", "InstanceCount": 1},
}


def _mock_session(region: str = "us-east-1") -> MagicMock:
    """Build a mock boto3.session.Session with a fixed region."""
    session = MagicMock()
    session.region_name = region
    return session


def _success_result(**overrides) -> LaunchResult:
    defaults = dict(
        job_name="test-job-0-ml-g5-4xlarge",
        instance_type_used="ml.g5.4xlarge",
        region_used="us-east-1",
        attempts=1,
        strategy_used="sequential",
    )
    defaults.update(overrides)
    return LaunchResult(**defaults)


# ---------------------------------------------------------------------------
# 1. Passthrough — Strategy="passthrough" calls boto3 directly, returns dict
#    Property 2: Passthrough transparency
# ---------------------------------------------------------------------------


class TestPassthroughPath:
    """Strategy='passthrough' bypasses all SDK logic and returns raw dict.

    Req 2.4: The SDK SHALL forward the request directly to boto3's
    create_transform_job with zero SDK logic, zero retry, zero error
    classification, and return the native boto3 API response unchanged.
    """

    @patch("sm_resilient_launcher._entry_point.boto3")
    def test_passthrough_calls_boto3_directly(self, mock_boto3):
        """Passthrough calls boto3.client('sagemaker').create_transform_job."""
        mock_session = _mock_session("us-east-1")
        mock_boto3.session.Session.return_value = mock_session

        mock_client = MagicMock()
        mock_client.create_transform_job.return_value = {
            "TransformJobArn": "arn:aws:sagemaker:us-east-1:123:transform-job/my-job"
        }
        mock_boto3.client.return_value = mock_client

        result = launch_create_transform_job(
            TransformJobName="my-job",
            ModelName="my-model",
            TransformInput={
                "DataSource": {"S3DataSource": {"S3Uri": "s3://bucket/input"}}
            },
            TransformOutput={"S3OutputPath": "s3://bucket/output"},
            TransformResources={
                "InstanceType": "ml.g5.4xlarge",
                "InstanceCount": 1,
            },
            Strategy="passthrough",
        )

        # Verify boto3.client was called with sagemaker.
        mock_boto3.client.assert_called_once_with(
            "sagemaker", region_name="us-east-1"
        )

        # Verify create_transform_job was called with only boto3 params
        # (Strategy must NOT appear).
        call_kwargs = mock_client.create_transform_job.call_args.kwargs
        assert "Strategy" not in call_kwargs
        assert call_kwargs["TransformJobName"] == "my-job"
        assert call_kwargs["ModelName"] == "my-model"

    @patch("sm_resilient_launcher._entry_point.boto3")
    def test_passthrough_returns_raw_dict(self, mock_boto3):
        """Passthrough returns the raw boto3 response dict, not LaunchResult."""
        mock_session = _mock_session("us-east-1")
        mock_boto3.session.Session.return_value = mock_session

        raw_response = {
            "TransformJobArn": "arn:aws:sagemaker:us-east-1:123:transform-job/my-job",
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }
        mock_client = MagicMock()
        mock_client.create_transform_job.return_value = raw_response
        mock_boto3.client.return_value = mock_client

        result = launch_create_transform_job(
            TransformJobName="my-job",
            ModelName="my-model",
            TransformResources={
                "InstanceType": "ml.g5.4xlarge",
                "InstanceCount": 1,
            },
            Strategy="passthrough",
        )

        assert isinstance(result, dict)
        assert not isinstance(result, LaunchResult)
        assert result == raw_response

    @patch("sm_resilient_launcher._entry_point.boto3")
    def test_passthrough_ignores_workload_description(self, mock_boto3):
        """Passthrough ignores WorkloadDescription — no Bedrock call is made."""
        mock_session = _mock_session("us-east-1")
        mock_boto3.session.Session.return_value = mock_session

        mock_client = MagicMock()
        mock_client.create_transform_job.return_value = {"TransformJobArn": "arn:test"}
        mock_boto3.client.return_value = mock_client

        result = launch_create_transform_job(
            TransformJobName="my-job",
            ModelName="my-model",
            TransformResources={
                "InstanceType": "ml.g5.4xlarge",
                "InstanceCount": 1,
            },
            Strategy="passthrough",
            WorkloadDescription="ResNet-50 inference, fp32",
        )

        # WorkloadDescription must not appear in boto3 call.
        call_kwargs = mock_client.create_transform_job.call_args.kwargs
        assert "WorkloadDescription" not in call_kwargs
        assert isinstance(result, dict)

    @patch("sm_resilient_launcher._entry_point.boto3")
    def test_passthrough_ignores_allowed_regions(self, mock_boto3):
        """Passthrough ignores AllowedRegions — no cross-region logic."""
        mock_session = _mock_session("us-east-1")
        mock_boto3.session.Session.return_value = mock_session

        mock_client = MagicMock()
        mock_client.create_transform_job.return_value = {"TransformJobArn": "arn:test"}
        mock_boto3.client.return_value = mock_client

        result = launch_create_transform_job(
            TransformJobName="my-job",
            ModelName="my-model",
            TransformResources={
                "InstanceType": "ml.g5.4xlarge",
                "InstanceCount": 1,
            },
            Strategy="passthrough",
            AllowedRegions=["us-east-1", "us-west-2"],
        )

        # AllowedRegions must not appear in the boto3 call.
        call_kwargs = mock_client.create_transform_job.call_args.kwargs
        assert "AllowedRegions" not in call_kwargs
        assert isinstance(result, dict)

    @patch("sm_resilient_launcher._entry_point.boto3")
    def test_passthrough_uses_session_region(self, mock_boto3):
        """Passthrough resolves region from boto3.session.Session."""
        mock_session = _mock_session("eu-west-1")
        mock_boto3.session.Session.return_value = mock_session

        mock_client = MagicMock()
        mock_client.create_transform_job.return_value = {"TransformJobArn": "arn:test"}
        mock_boto3.client.return_value = mock_client

        launch_create_transform_job(
            TransformJobName="my-job",
            ModelName="my-model",
            TransformResources={
                "InstanceType": "ml.g5.4xlarge",
                "InstanceCount": 1,
            },
            Strategy="passthrough",
        )

        mock_boto3.client.assert_called_once_with(
            "sagemaker", region_name="eu-west-1"
        )

    @patch("sm_resilient_launcher._entry_point.boto3")
    def test_passthrough_defaults_to_us_east_1_when_no_session_region(
        self, mock_boto3
    ):
        """Passthrough defaults to us-east-1 when session region is None."""
        mock_session = MagicMock()
        mock_session.region_name = None
        mock_boto3.session.Session.return_value = mock_session

        mock_client = MagicMock()
        mock_client.create_transform_job.return_value = {"TransformJobArn": "arn:test"}
        mock_boto3.client.return_value = mock_client

        launch_create_transform_job(
            TransformJobName="my-job",
            ModelName="my-model",
            TransformResources={
                "InstanceType": "ml.g5.4xlarge",
                "InstanceCount": 1,
            },
            Strategy="passthrough",
        )

        mock_boto3.client.assert_called_once_with(
            "sagemaker", region_name="us-east-1"
        )


# ---------------------------------------------------------------------------
# 2. SDK params never appear in boto3 call (passthrough path)
#    Req 4.2: SDK-specific parameters SHALL NOT conflict with boto3 params.
# ---------------------------------------------------------------------------


class TestSDKParamsSeparation:
    """SDK params (Strategy, AllowedRegions, WorkloadDescription) never
    appear in the boto3 create_transform_job call.

    Req 4.2: The SDK SHALL separate SDK-specific parameters from boto3
    parameters before forwarding to the SageMaker API.
    """

    @patch("sm_resilient_launcher._entry_point.boto3")
    def test_all_three_sdk_params_stripped_in_passthrough(self, mock_boto3):
        """All three SDK params are stripped before the boto3 call."""
        mock_session = _mock_session("us-east-1")
        mock_boto3.session.Session.return_value = mock_session

        mock_client = MagicMock()
        mock_client.create_transform_job.return_value = {"TransformJobArn": "arn:test"}
        mock_boto3.client.return_value = mock_client

        launch_create_transform_job(
            TransformJobName="my-job",
            ModelName="my-model",
            TransformResources={
                "InstanceType": "ml.g5.4xlarge",
                "InstanceCount": 1,
            },
            Strategy="passthrough",
            AllowedRegions=["us-east-1"],
            WorkloadDescription="test workload",
        )

        call_kwargs = mock_client.create_transform_job.call_args.kwargs
        assert "Strategy" not in call_kwargs
        assert "AllowedRegions" not in call_kwargs
        assert "WorkloadDescription" not in call_kwargs


# ---------------------------------------------------------------------------
# 3. Unrecognized params pass through unchanged
#    Req 4.1, 4.3: Forward compatibility — never reject unknown params.
# ---------------------------------------------------------------------------


class TestUnrecognizedParamPassthrough:
    """Unrecognized parameters pass through to boto3 unchanged.

    Req 4.1: Unrecognized params SHALL be passed through to boto3.
    Req 4.3: The SDK SHALL NOT reject unrecognized parameters.
    """

    @patch("sm_resilient_launcher._entry_point.boto3")
    def test_unrecognized_params_forwarded_in_passthrough(self, mock_boto3):
        """Unknown params appear in the boto3 call unchanged."""
        mock_session = _mock_session("us-east-1")
        mock_boto3.session.Session.return_value = mock_session

        mock_client = MagicMock()
        mock_client.create_transform_job.return_value = {"TransformJobArn": "arn:test"}
        mock_boto3.client.return_value = mock_client

        launch_create_transform_job(
            TransformJobName="my-job",
            ModelName="my-model",
            TransformResources={
                "InstanceType": "ml.g5.4xlarge",
                "InstanceCount": 1,
            },
            Strategy="passthrough",
            FutureNewParam="some-value",
            AnotherUnknown=42,
        )

        call_kwargs = mock_client.create_transform_job.call_args.kwargs
        assert call_kwargs["FutureNewParam"] == "some-value"
        assert call_kwargs["AnotherUnknown"] == 42


# ---------------------------------------------------------------------------
# 4. List InstanceType activates Launcher-based execution
#    (returns LaunchResult, not dict)
# ---------------------------------------------------------------------------


class TestListModeActivatesLauncher:
    """List InstanceType activates the Launcher path, returning LaunchResult.

    Req 2.5: WHEN InstanceType is a list and no Strategy, the Launcher
    SHALL execute sequential with the caller's order.
    """

    @patch("sm_resilient_launcher._entry_point.Launcher")
    @patch("sm_resilient_launcher._entry_point.TransformJobAdapter")
    @patch("sm_resilient_launcher._entry_point.IntelligentRouter")
    @patch("sm_resilient_launcher._entry_point.TypeExpander")
    @patch("sm_resilient_launcher._entry_point.PricingFetcher")
    @patch("sm_resilient_launcher._entry_point.QuotaFetcher")
    @patch("sm_resilient_launcher._entry_point.boto3")
    def test_list_mode_returns_launch_result(
        self,
        mock_boto3,
        MockQuota,
        MockPricing,
        MockTypeExpander,
        MockRouter,
        MockAdapter,
        MockLauncher,
    ):
        """List InstanceType returns LaunchResult from Launcher.launch()."""
        mock_session = _mock_session("us-east-1")
        mock_boto3.session.Session.return_value = mock_session

        expected = _success_result(
            job_name="list-job-0-ml-g5-4xlarge",
            instance_type_used="ml.g5.4xlarge",
            strategy_used="sequential",
        )
        mock_launcher_instance = MagicMock()
        mock_launcher_instance.launch.return_value = expected
        MockLauncher.return_value = mock_launcher_instance

        result = launch_create_transform_job(
            TransformJobName="list-job",
            ModelName="my-model",
            TransformInput={
                "DataSource": {"S3DataSource": {"S3Uri": "s3://bucket/input"}}
            },
            TransformOutput={"S3OutputPath": "s3://bucket/output"},
            TransformResources={
                "InstanceType": ["ml.g5.4xlarge", "ml.g4dn.xlarge"],
                "InstanceCount": 1,
            },
        )

        assert isinstance(result, LaunchResult)
        assert result.job_name == "list-job-0-ml-g5-4xlarge"

        # Verify Launcher.launch() was called with the correct instance_types list.
        launch_call = mock_launcher_instance.launch.call_args
        assert launch_call.kwargs["instance_types"] == [
            "ml.g5.4xlarge",
            "ml.g4dn.xlarge",
        ]

    @patch("sm_resilient_launcher._entry_point.Launcher")
    @patch("sm_resilient_launcher._entry_point.TransformJobAdapter")
    @patch("sm_resilient_launcher._entry_point.IntelligentRouter")
    @patch("sm_resilient_launcher._entry_point.TypeExpander")
    @patch("sm_resilient_launcher._entry_point.PricingFetcher")
    @patch("sm_resilient_launcher._entry_point.QuotaFetcher")
    @patch("sm_resilient_launcher._entry_point.boto3")
    def test_list_mode_preserves_caller_order(
        self,
        mock_boto3,
        MockQuota,
        MockPricing,
        MockTypeExpander,
        MockRouter,
        MockAdapter,
        MockLauncher,
    ):
        """List order is preserved exactly as the caller provided."""
        mock_session = _mock_session("us-east-1")
        mock_boto3.session.Session.return_value = mock_session

        mock_launcher_instance = MagicMock()
        mock_launcher_instance.launch.return_value = _success_result()
        MockLauncher.return_value = mock_launcher_instance

        caller_order = ["ml.p5.48xlarge", "ml.g5.xlarge", "ml.g4dn.xlarge"]

        launch_create_transform_job(
            TransformJobName="order-test",
            ModelName="my-model",
            TransformResources={
                "InstanceType": caller_order,
                "InstanceCount": 1,
            },
        )

        launch_call = mock_launcher_instance.launch.call_args
        assert launch_call.kwargs["instance_types"] == caller_order


# ---------------------------------------------------------------------------
# 5. Scalar InstanceType without Strategy activates AI expansion path
# ---------------------------------------------------------------------------


class TestScalarAIExpansionPath:
    """Scalar InstanceType without Strategy triggers the Launcher path
    (which internally uses AI expansion or TypeExpander).

    Req 2.1: Scalar + no Strategy → AI Expansion → sequential.
    """

    @patch("sm_resilient_launcher._entry_point.Launcher")
    @patch("sm_resilient_launcher._entry_point.TransformJobAdapter")
    @patch("sm_resilient_launcher._entry_point.IntelligentRouter")
    @patch("sm_resilient_launcher._entry_point.TypeExpander")
    @patch("sm_resilient_launcher._entry_point.PricingFetcher")
    @patch("sm_resilient_launcher._entry_point.QuotaFetcher")
    @patch("sm_resilient_launcher._entry_point.boto3")
    def test_scalar_no_strategy_goes_to_launcher(
        self,
        mock_boto3,
        MockQuota,
        MockPricing,
        MockTypeExpander,
        MockRouter,
        MockAdapter,
        MockLauncher,
    ):
        """Scalar InstanceType without Strategy delegates to Launcher.launch."""
        mock_session = _mock_session("us-east-1")
        mock_boto3.session.Session.return_value = mock_session

        expected = _success_result()
        mock_launcher_instance = MagicMock()
        mock_launcher_instance.launch.return_value = expected
        MockLauncher.return_value = mock_launcher_instance

        result = launch_create_transform_job(
            TransformJobName="scalar-test",
            ModelName="my-model",
            TransformResources={
                "InstanceType": "ml.g5.4xlarge",
                "InstanceCount": 1,
            },
        Strategy="sequential",
        )

        # Launcher must be constructed and .launch() called.
        MockLauncher.assert_called_once()
        mock_launcher_instance.launch.assert_called_once()

        # Instance types should be a single-element list.
        launch_call = mock_launcher_instance.launch.call_args
        assert launch_call.kwargs["instance_types"] == ["ml.g5.4xlarge"]
        assert isinstance(result, LaunchResult)

    @patch("sm_resilient_launcher._entry_point.Launcher")
    @patch("sm_resilient_launcher._entry_point.TransformJobAdapter")
    @patch("sm_resilient_launcher._entry_point.IntelligentRouter")
    @patch("sm_resilient_launcher._entry_point.TypeExpander")
    @patch("sm_resilient_launcher._entry_point.PricingFetcher")
    @patch("sm_resilient_launcher._entry_point.QuotaFetcher")
    @patch("sm_resilient_launcher._entry_point.boto3")
    def test_scalar_explicit_strategy_goes_to_launcher(
        self,
        mock_boto3,
        MockQuota,
        MockPricing,
        MockTypeExpander,
        MockRouter,
        MockAdapter,
        MockLauncher,
    ):
        """Scalar with explicit Strategy='race' goes through Launcher."""
        mock_session = _mock_session("us-east-1")
        mock_boto3.session.Session.return_value = mock_session

        mock_launcher_instance = MagicMock()
        mock_launcher_instance.launch.return_value = _success_result(
            strategy_used="race"
        )
        MockLauncher.return_value = mock_launcher_instance

        result = launch_create_transform_job(
            TransformJobName="race-test",
            ModelName="my-model",
            TransformResources={
                "InstanceType": "ml.g5.4xlarge",
                "InstanceCount": 1,
            },
            Strategy="race",
        )

        launch_call = mock_launcher_instance.launch.call_args
        # Strategy should be in sdk_params.
        assert launch_call.kwargs["sdk_params"]["Strategy"] == "race"
        assert isinstance(result, LaunchResult)


# ---------------------------------------------------------------------------
# 6. Component graph instantiation — correct wiring
# ---------------------------------------------------------------------------


class TestComponentGraphWiring:
    """Verify the entry point instantiates the correct component graph
    and passes components to Launcher.
    """

    @patch("sm_resilient_launcher._entry_point.Launcher")
    @patch("sm_resilient_launcher._entry_point.TransformJobAdapter")
    @patch("sm_resilient_launcher._entry_point.IntelligentRouter")
    @patch("sm_resilient_launcher._entry_point.TypeExpander")
    @patch("sm_resilient_launcher._entry_point.PricingFetcher")
    @patch("sm_resilient_launcher._entry_point.QuotaFetcher")
    @patch("sm_resilient_launcher._entry_point.boto3")
    def test_launcher_receives_all_components(
        self,
        mock_boto3,
        MockQuota,
        MockPricing,
        MockTypeExpander,
        MockRouter,
        MockAdapter,
        MockLauncher,
    ):
        """Launcher is constructed with adapter, classifier, router, expander."""
        mock_session = _mock_session("us-east-1")
        mock_boto3.session.Session.return_value = mock_session

        mock_launcher_instance = MagicMock()
        mock_launcher_instance.launch.return_value = _success_result()
        MockLauncher.return_value = mock_launcher_instance

        launch_create_transform_job(
            TransformJobName="wiring-test",
            ModelName="my-model",
            TransformResources={
                "InstanceType": "ml.g5.4xlarge",
                "InstanceCount": 1,
            },
        Strategy="sequential",
        )

        # Verify Launcher was constructed with the expected component types.
        launcher_kwargs = MockLauncher.call_args.kwargs
        assert "adapter" in launcher_kwargs
        assert "error_classifier" in launcher_kwargs
        assert "intelligent_router" in launcher_kwargs
        assert "type_expander" in launcher_kwargs
        assert "adapter_factory" in launcher_kwargs

    @patch("sm_resilient_launcher._entry_point.Launcher")
    @patch("sm_resilient_launcher._entry_point.TransformJobAdapter")
    @patch("sm_resilient_launcher._entry_point.IntelligentRouter")
    @patch("sm_resilient_launcher._entry_point.TypeExpander")
    @patch("sm_resilient_launcher._entry_point.PricingFetcher")
    @patch("sm_resilient_launcher._entry_point.QuotaFetcher")
    @patch("sm_resilient_launcher._entry_point.boto3")
    def test_adapter_created_with_correct_region(
        self,
        mock_boto3,
        MockQuota,
        MockPricing,
        MockTypeExpander,
        MockRouter,
        MockAdapter,
        MockLauncher,
    ):
        """TransformJobAdapter is created with the resolved region."""
        mock_session = _mock_session("eu-west-1")
        mock_boto3.session.Session.return_value = mock_session

        mock_launcher_instance = MagicMock()
        mock_launcher_instance.launch.return_value = _success_result()
        MockLauncher.return_value = mock_launcher_instance

        launch_create_transform_job(
            TransformJobName="region-test",
            ModelName="my-model",
            TransformResources={
                "InstanceType": "ml.g5.4xlarge",
                "InstanceCount": 1,
            },
        Strategy="sequential",
        )

        MockAdapter.assert_called_once_with("eu-west-1")

    @patch("sm_resilient_launcher._entry_point.Launcher")
    @patch("sm_resilient_launcher._entry_point.TransformJobAdapter")
    @patch("sm_resilient_launcher._entry_point.IntelligentRouter")
    @patch("sm_resilient_launcher._entry_point.TypeExpander")
    @patch("sm_resilient_launcher._entry_point.PricingFetcher")
    @patch("sm_resilient_launcher._entry_point.QuotaFetcher")
    @patch("sm_resilient_launcher._entry_point.boto3")
    def test_allowed_regions_first_used_as_home_region(
        self,
        mock_boto3,
        MockQuota,
        MockPricing,
        MockTypeExpander,
        MockRouter,
        MockAdapter,
        MockLauncher,
    ):
        """When AllowedRegions is provided, the first region is used as home."""
        mock_session = _mock_session("us-east-1")
        mock_boto3.session.Session.return_value = mock_session

        mock_launcher_instance = MagicMock()
        mock_launcher_instance.launch.return_value = _success_result()
        MockLauncher.return_value = mock_launcher_instance

        launch_create_transform_job(
            TransformJobName="region-override",
            ModelName="my-model",
            TransformResources={
                "InstanceType": "ml.g5.4xlarge",
                "InstanceCount": 1,
            },
            AllowedRegions=["ap-southeast-1", "us-west-2"],
        )

        # Adapter should be created with the first AllowedRegion.
        MockAdapter.assert_called_once_with("ap-southeast-1")

        # Launcher.launch region arg should be first AllowedRegion.
        launch_call = mock_launcher_instance.launch.call_args
        assert launch_call.kwargs["region"] == "ap-southeast-1"


# ---------------------------------------------------------------------------
# 7. Return type: LaunchResult for non-passthrough, dict for passthrough
#    Property 18: LaunchResult contains all required fields
# ---------------------------------------------------------------------------


class TestReturnTypes:
    """Non-passthrough returns LaunchResult; passthrough returns dict.

    Property 18: LaunchResult has non-null job_name, instance_type_used,
    region_used, attempts, and strategy_used.
    """

    @patch("sm_resilient_launcher._entry_point.boto3")
    def test_passthrough_returns_dict(self, mock_boto3):
        """Passthrough always returns a plain dict."""
        mock_session = _mock_session("us-east-1")
        mock_boto3.session.Session.return_value = mock_session

        mock_client = MagicMock()
        mock_client.create_transform_job.return_value = {"TransformJobArn": "arn:test"}
        mock_boto3.client.return_value = mock_client

        result = launch_create_transform_job(
            TransformJobName="my-job",
            TransformResources={
                "InstanceType": "ml.g5.4xlarge",
                "InstanceCount": 1,
            },
            Strategy="passthrough",
        )

        assert isinstance(result, dict)
        assert not isinstance(result, LaunchResult)

    @patch("sm_resilient_launcher._entry_point.Launcher")
    @patch("sm_resilient_launcher._entry_point.TransformJobAdapter")
    @patch("sm_resilient_launcher._entry_point.IntelligentRouter")
    @patch("sm_resilient_launcher._entry_point.TypeExpander")
    @patch("sm_resilient_launcher._entry_point.PricingFetcher")
    @patch("sm_resilient_launcher._entry_point.QuotaFetcher")
    @patch("sm_resilient_launcher._entry_point.boto3")
    def test_non_passthrough_returns_launch_result(
        self,
        mock_boto3,
        MockQuota,
        MockPricing,
        MockTypeExpander,
        MockRouter,
        MockAdapter,
        MockLauncher,
    ):
        """Non-passthrough paths return LaunchResult."""
        mock_session = _mock_session("us-east-1")
        mock_boto3.session.Session.return_value = mock_session

        expected = _success_result()
        mock_launcher_instance = MagicMock()
        mock_launcher_instance.launch.return_value = expected
        MockLauncher.return_value = mock_launcher_instance

        result = launch_create_transform_job(
            TransformJobName="my-job",
            ModelName="my-model",
            TransformResources={
                "InstanceType": "ml.g5.4xlarge",
                "InstanceCount": 1,
            },
            Strategy="sequential",
        )

        assert isinstance(result, LaunchResult)

    @patch("sm_resilient_launcher._entry_point.Launcher")
    @patch("sm_resilient_launcher._entry_point.TransformJobAdapter")
    @patch("sm_resilient_launcher._entry_point.IntelligentRouter")
    @patch("sm_resilient_launcher._entry_point.TypeExpander")
    @patch("sm_resilient_launcher._entry_point.PricingFetcher")
    @patch("sm_resilient_launcher._entry_point.QuotaFetcher")
    @patch("sm_resilient_launcher._entry_point.boto3")
    def test_launch_result_has_all_required_fields(
        self,
        mock_boto3,
        MockQuota,
        MockPricing,
        MockTypeExpander,
        MockRouter,
        MockAdapter,
        MockLauncher,
    ):
        """LaunchResult returned has all non-null required fields.

        Property 18: job_name, instance_type_used, region_used, attempts,
        strategy_used SHALL be non-null.
        """
        mock_session = _mock_session("us-east-1")
        mock_boto3.session.Session.return_value = mock_session

        expected = _success_result(
            job_name="result-fields-job",
            instance_type_used="ml.g5.4xlarge",
            region_used="us-east-1",
            attempts=3,
            strategy_used="sequential",
        )
        mock_launcher_instance = MagicMock()
        mock_launcher_instance.launch.return_value = expected
        MockLauncher.return_value = mock_launcher_instance

        result = launch_create_transform_job(
            TransformJobName="result-fields-test",
            ModelName="my-model",
            TransformResources={
                "InstanceType": ["ml.g5.4xlarge", "ml.g4dn.xlarge"],
                "InstanceCount": 1,
            },
        )

        assert result.job_name is not None
        assert result.instance_type_used is not None
        assert result.region_used is not None
        assert result.attempts is not None
        assert result.strategy_used is not None
        assert result.job_name == "result-fields-job"
        assert result.attempts == 3


# ---------------------------------------------------------------------------
# 8. Boto3 params forwarded correctly to Launcher (non-passthrough)
# ---------------------------------------------------------------------------


class TestBoto3ParamsForwarding:
    """Verify that boto3 params are correctly forwarded to Launcher.launch."""

    @patch("sm_resilient_launcher._entry_point.Launcher")
    @patch("sm_resilient_launcher._entry_point.TransformJobAdapter")
    @patch("sm_resilient_launcher._entry_point.IntelligentRouter")
    @patch("sm_resilient_launcher._entry_point.TypeExpander")
    @patch("sm_resilient_launcher._entry_point.PricingFetcher")
    @patch("sm_resilient_launcher._entry_point.QuotaFetcher")
    @patch("sm_resilient_launcher._entry_point.boto3")
    def test_boto3_params_exclude_sdk_params(
        self,
        mock_boto3,
        MockQuota,
        MockPricing,
        MockTypeExpander,
        MockRouter,
        MockAdapter,
        MockLauncher,
    ):
        """boto3_params passed to Launcher must not contain SDK params."""
        mock_session = _mock_session("us-east-1")
        mock_boto3.session.Session.return_value = mock_session

        mock_launcher_instance = MagicMock()
        mock_launcher_instance.launch.return_value = _success_result()
        MockLauncher.return_value = mock_launcher_instance

        launch_create_transform_job(
            TransformJobName="my-job",
            ModelName="my-model",
            TransformResources={
                "InstanceType": "ml.g5.4xlarge",
                "InstanceCount": 1,
            },
            Strategy="sequential",
            AllowedRegions=["us-east-1"],
            WorkloadDescription="test workload",
        )

        launch_call = mock_launcher_instance.launch.call_args
        boto3_params = launch_call.kwargs["boto3_params"]
        sdk_params = launch_call.kwargs["sdk_params"]

        # SDK params must be in sdk_params, not boto3_params.
        assert "Strategy" not in boto3_params
        assert "AllowedRegions" not in boto3_params
        assert "WorkloadDescription" not in boto3_params
        assert sdk_params["Strategy"] == "sequential"
        assert sdk_params["AllowedRegions"] == ["us-east-1"]
        assert sdk_params["WorkloadDescription"] == "test workload"

        # boto3 params must be present.
        assert boto3_params["TransformJobName"] == "my-job"
        assert boto3_params["ModelName"] == "my-model"

    @patch("sm_resilient_launcher._entry_point.Launcher")
    @patch("sm_resilient_launcher._entry_point.TransformJobAdapter")
    @patch("sm_resilient_launcher._entry_point.IntelligentRouter")
    @patch("sm_resilient_launcher._entry_point.TypeExpander")
    @patch("sm_resilient_launcher._entry_point.PricingFetcher")
    @patch("sm_resilient_launcher._entry_point.QuotaFetcher")
    @patch("sm_resilient_launcher._entry_point.boto3")
    def test_unrecognized_params_forwarded_to_launcher(
        self,
        mock_boto3,
        MockQuota,
        MockPricing,
        MockTypeExpander,
        MockRouter,
        MockAdapter,
        MockLauncher,
    ):
        """Unrecognized params appear in boto3_params for Launcher."""
        mock_session = _mock_session("us-east-1")
        mock_boto3.session.Session.return_value = mock_session

        mock_launcher_instance = MagicMock()
        mock_launcher_instance.launch.return_value = _success_result()
        MockLauncher.return_value = mock_launcher_instance

        launch_create_transform_job(
            TransformJobName="my-job",
            ModelName="my-model",
            TransformResources={
                "InstanceType": "ml.g5.4xlarge",
                "InstanceCount": 1,
            },
            FutureParam="hello",
            NewApiField={"nested": "value"},
                    Strategy="sequential",
        )

        launch_call = mock_launcher_instance.launch.call_args
        boto3_params = launch_call.kwargs["boto3_params"]

        assert boto3_params["FutureParam"] == "hello"
        assert boto3_params["NewApiField"] == {"nested": "value"}


# ---------------------------------------------------------------------------
# 9. Edge case: InstanceType missing or empty
# ---------------------------------------------------------------------------


class TestInstanceTypeEdgeCases:
    """Edge cases around TransformResources.InstanceType."""

    @patch("sm_resilient_launcher._entry_point.Launcher")
    @patch("sm_resilient_launcher._entry_point.TransformJobAdapter")
    @patch("sm_resilient_launcher._entry_point.IntelligentRouter")
    @patch("sm_resilient_launcher._entry_point.TypeExpander")
    @patch("sm_resilient_launcher._entry_point.PricingFetcher")
    @patch("sm_resilient_launcher._entry_point.QuotaFetcher")
    @patch("sm_resilient_launcher._entry_point.boto3")
    def test_missing_instance_type_sends_empty_list(
        self,
        mock_boto3,
        MockQuota,
        MockPricing,
        MockTypeExpander,
        MockRouter,
        MockAdapter,
        MockLauncher,
    ):
        """When InstanceType is absent, instance_types is an empty list."""
        mock_session = _mock_session("us-east-1")
        mock_boto3.session.Session.return_value = mock_session

        mock_launcher_instance = MagicMock()
        mock_launcher_instance.launch.return_value = _success_result()
        MockLauncher.return_value = mock_launcher_instance

        launch_create_transform_job(
            TransformJobName="no-type-test",
            ModelName="my-model",
            TransformResources={"InstanceCount": 1},
            Strategy="sequential",
        )

        launch_call = mock_launcher_instance.launch.call_args
        assert launch_call.kwargs["instance_types"] == []
