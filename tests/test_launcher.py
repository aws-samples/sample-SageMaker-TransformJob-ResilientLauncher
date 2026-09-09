"""Unit tests for Launcher — behavior matrix orchestrator.

Property 3: Caller list order invariant
    For any list of instance types provided by the caller, the list passed
    to the strategy engine SHALL be identical in order and content.

Validates: Requirements 2.1, 2.2, 2.3, 2.5, 2.6, 2.7, 15.1, 15.2, 15.3
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

from sm_resilient_launcher._error_classifier import (
    ErrorCategory,
    ErrorClassifier,
)
from sm_resilient_launcher._exceptions import (
    AllTypesExhaustedError,
    AllRegionsExhaustedError,
)
from sm_resilient_launcher._launch_result import LaunchResult
from sm_resilient_launcher._launcher import Launcher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


BOTO3_PARAMS = {
    "TransformJobName": "my-job",
    "ModelName": "my-model",
    "TransformInput": {
        "DataSource": {"S3DataSource": {"S3Uri": "s3://bucket/input"}}
    },
    "TransformOutput": {"S3OutputPath": "s3://bucket/output"},
    "TransformResources": {"InstanceCount": 1},
}


def _make_adapter() -> MagicMock:
    """Build a minimal mock JobAdapter."""
    return MagicMock()


def _success_result(**overrides) -> LaunchResult:
    """Build a default successful LaunchResult."""
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
# 1. Scalar + no Strategy + no/single region → AI Expansion
#    with WorkloadDescription → IntelligentRouter
# ---------------------------------------------------------------------------


class TestScalarAIExpansionWithWorkload:
    """Scalar + no Strategy + WorkloadDescription → IntelligentRouter.

    Req 2.1: The Launcher SHALL invoke AI Expansion (IntelligentRouter)
    when scalar type, no explicit strategy, and WorkloadDescription present.
    """

    @patch(
        "sm_resilient_launcher._launcher.SequentialEngine.execute",
    )
    def test_uses_intelligent_router(self, mock_execute):
        """IntelligentRouter.generate_plan is called and its results are used."""
        mock_execute.return_value = _success_result()

        intelligent_router = MagicMock()
        intelligent_router.generate_plan.return_value = (
            ["ml.g5.4xlarge", "ml.g5.xlarge"],
            "sequential",
            "GPU memory analysis reasoning",
        )

        launcher = Launcher(
            adapter=_make_adapter(),
            error_classifier=ErrorClassifier,
            intelligent_router=intelligent_router,
            type_expander=MagicMock(),
        )

        result = launcher.launch(
            instance_types=["ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            sdk_params={"WorkloadDescription": "ResNet-50 inference"},
            region="us-east-1",
        )

        intelligent_router.generate_plan.assert_called_once_with(
            "ml.g5.4xlarge", "ResNet-50 inference"
        )
        # The expanded types should be passed to the engine.
        engine_call = mock_execute.call_args
        assert engine_call.kwargs["instance_types"] == [
            "ml.g5.4xlarge",
            "ml.g5.xlarge",
        ]

    @patch(
        "sm_resilient_launcher._launcher.SequentialEngine.execute",
    )
    def test_agent_reasoning_attached(self, mock_execute):
        """agent_reasoning is set on LaunchResult when Bedrock was used."""
        mock_execute.return_value = _success_result()

        intelligent_router = MagicMock()
        intelligent_router.generate_plan.return_value = (
            ["ml.g5.4xlarge"],
            "sequential",
            "Bedrock reasoning chain",
        )

        launcher = Launcher(
            adapter=_make_adapter(),
            error_classifier=ErrorClassifier,
            intelligent_router=intelligent_router,
        )

        result = launcher.launch(
            instance_types=["ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            sdk_params={"WorkloadDescription": "LLM inference"},
            region="us-east-1",
        )

        # Req 15.2: agent_reasoning non-null when Bedrock was used.
        assert result.agent_reasoning == "Bedrock reasoning chain"

    @patch(
        "sm_resilient_launcher._launcher.SequentialEngine.execute",
    )
    def test_uses_strategy_from_router(self, mock_execute):
        """The strategy recommended by IntelligentRouter is honored."""
        mock_execute.return_value = _success_result(strategy_used="race")

        intelligent_router = MagicMock()
        intelligent_router.generate_plan.return_value = (
            ["ml.g5.4xlarge", "ml.g5.xlarge"],
            "race",
            "recommend race for fast provisioning",
        )

        launcher = Launcher(
            adapter=_make_adapter(),
            error_classifier=ErrorClassifier,
            intelligent_router=intelligent_router,
        )

        # Should use RaceEngine since router recommended "race"
        with patch(
            "sm_resilient_launcher._launcher.RaceEngine.execute",
        ) as mock_race:
            mock_race.return_value = _success_result(strategy_used="race")
            result = launcher.launch(
                instance_types=["ml.g5.4xlarge"],
                boto3_params=BOTO3_PARAMS,
                sdk_params={"WorkloadDescription": "quick inference"},
                region="us-east-1",
            )
            mock_race.assert_called_once()


# ---------------------------------------------------------------------------
# 2. Scalar + no Strategy + no WorkloadDescription → TypeExpander
# ---------------------------------------------------------------------------


class TestScalarAIExpansionWithoutWorkload:
    """Scalar + no Strategy + no WorkloadDescription → TypeExpander.

    Req 2.1: The Launcher SHALL invoke TypeExpander for deterministic
    expansion when no WorkloadDescription is provided.
    """

    @patch(
        "sm_resilient_launcher._launcher.SequentialEngine.execute",
    )
    def test_uses_type_expander(self, mock_execute):
        """TypeExpander.expand is called when no WorkloadDescription."""
        mock_execute.return_value = _success_result()

        type_expander = MagicMock()
        type_expander.expand.return_value = [
            "ml.g5.4xlarge",
            "ml.g5.xlarge",
            "ml.g5.12xlarge",
        ]

        launcher = Launcher(
            adapter=_make_adapter(),
            error_classifier=ErrorClassifier,
            type_expander=type_expander,
        )

        result = launcher.launch(
            instance_types=["ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            sdk_params={},
            region="us-east-1",
        )

        type_expander.expand.assert_called_once_with(
            "ml.g5.4xlarge", "us-east-1"
        )
        engine_call = mock_execute.call_args
        assert engine_call.kwargs["instance_types"] == [
            "ml.g5.4xlarge",
            "ml.g5.xlarge",
            "ml.g5.12xlarge",
        ]

    @patch(
        "sm_resilient_launcher._launcher.SequentialEngine.execute",
    )
    def test_agent_reasoning_none_for_deterministic(self, mock_execute):
        """agent_reasoning is None when TypeExpander (deterministic) was used."""
        mock_execute.return_value = _success_result()

        type_expander = MagicMock()
        type_expander.expand.return_value = ["ml.g5.4xlarge"]

        launcher = Launcher(
            adapter=_make_adapter(),
            error_classifier=ErrorClassifier,
            type_expander=type_expander,
        )

        result = launcher.launch(
            instance_types=["ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            sdk_params={},
            region="us-east-1",
        )

        # Req 15.2: agent_reasoning is None for deterministic path.
        assert result.agent_reasoning is None

    @patch(
        "sm_resilient_launcher._launcher.SequentialEngine.execute",
    )
    def test_defaults_to_sequential_strategy(self, mock_execute):
        """TypeExpander expansion defaults to sequential strategy."""
        mock_execute.return_value = _success_result()

        type_expander = MagicMock()
        type_expander.expand.return_value = ["ml.g5.4xlarge"]

        launcher = Launcher(
            adapter=_make_adapter(),
            error_classifier=ErrorClassifier,
            type_expander=type_expander,
        )

        result = launcher.launch(
            instance_types=["ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            sdk_params={},
            region="us-east-1",
        )

        # SequentialEngine.execute was called (not RaceEngine).
        mock_execute.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Scalar + no Strategy + no expanders → single-type sequential
# ---------------------------------------------------------------------------


class TestScalarNoExpanders:
    """Scalar + no Strategy + no expanders available → sequential with one type."""

    @patch(
        "sm_resilient_launcher._launcher.SequentialEngine.execute",
    )
    def test_single_type_sequential(self, mock_execute):
        mock_execute.return_value = _success_result()

        launcher = Launcher(
            adapter=_make_adapter(),
            error_classifier=ErrorClassifier,
        )

        result = launcher.launch(
            instance_types=["ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            sdk_params={},
            region="us-east-1",
        )

        engine_call = mock_execute.call_args
        assert engine_call.kwargs["instance_types"] == ["ml.g5.4xlarge"]


# ---------------------------------------------------------------------------
# 4. Scalar + no Strategy + multi-region → CrossRegionManager
# ---------------------------------------------------------------------------


class TestScalarMultiRegion:
    """Scalar + no Strategy + AllowedRegions with 2+ regions → cross-region.

    Req 2.2: The Launcher SHALL attempt the single instance type across
    allowed regions using sequential fallback per region.
    """

    @patch(
        "sm_resilient_launcher._launcher.CrossRegionManager",
    )
    def test_triggers_cross_region(self, MockCRM):
        """CrossRegionManager is instantiated with correct arguments."""
        mock_crm_instance = MagicMock()
        mock_crm_instance.execute.return_value = _success_result(
            region_used="us-west-2",
            data_replicated=True,
        )
        MockCRM.return_value = mock_crm_instance

        adapter_factory = MagicMock()
        launcher = Launcher(
            adapter=_make_adapter(),
            error_classifier=ErrorClassifier,
            adapter_factory=adapter_factory,
        )

        result = launcher.launch(
            instance_types=["ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            sdk_params={
                "AllowedRegions": ["us-east-1", "us-west-2", "eu-west-1"],
            },
            region="us-east-1",
        )

        # CrossRegionManager should have been created.
        MockCRM.assert_called_once()
        crm_kwargs = MockCRM.call_args.kwargs
        assert crm_kwargs["adapter_factory"] == adapter_factory
        assert crm_kwargs["home_region"] == "us-east-1"
        assert crm_kwargs["allowed_regions"] == [
            "us-east-1",
            "us-west-2",
            "eu-west-1",
        ]
        assert crm_kwargs["base_job_name"] == "my-job"

        # execute() should be called with the single type.
        mock_crm_instance.execute.assert_called_once_with(
            ["ml.g5.4xlarge"], BOTO3_PARAMS, "my-job"
        )

    @patch(
        "sm_resilient_launcher._launcher.CrossRegionManager",
    )
    def test_uses_sequential_strategy_for_cross_region(self, MockCRM):
        """Cross-region defaults to sequential per region."""
        mock_crm_instance = MagicMock()
        mock_crm_instance.execute.return_value = _success_result()
        MockCRM.return_value = mock_crm_instance

        launcher = Launcher(
            adapter=_make_adapter(),
            error_classifier=ErrorClassifier,
            adapter_factory=MagicMock(),
        )

        result = launcher.launch(
            instance_types=["ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            sdk_params={
                "AllowedRegions": ["us-east-1", "eu-west-1"],
            },
            region="us-east-1",
        )

        # Verify the strategy_engine_factory builds a SequentialEngine.
        crm_kwargs = MockCRM.call_args.kwargs
        factory = crm_kwargs["strategy_engine_factory"]
        engine = factory(_make_adapter())
        from sm_resilient_launcher._sequential_engine import SequentialEngine

        assert isinstance(engine, SequentialEngine)


# ---------------------------------------------------------------------------
# 5. Scalar + explicit Strategy → execute specified strategy
# ---------------------------------------------------------------------------


class TestScalarExplicitStrategy:
    """Scalar + explicit Strategy → execute that strategy.

    Req 2.3: The Launcher SHALL execute the specified strategy using
    the single instance type.
    """

    @patch(
        "sm_resilient_launcher._launcher.SequentialEngine.execute",
    )
    def test_explicit_sequential(self, mock_execute):
        mock_execute.return_value = _success_result()

        launcher = Launcher(
            adapter=_make_adapter(),
            error_classifier=ErrorClassifier,
            intelligent_router=MagicMock(),
            type_expander=MagicMock(),
        )

        result = launcher.launch(
            instance_types=["ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            sdk_params={"Strategy": "sequential"},
            region="us-east-1",
        )

        mock_execute.assert_called_once()
        # IntelligentRouter and TypeExpander should NOT have been called.
        launcher._intelligent_router.generate_plan.assert_not_called()
        launcher._type_expander.expand.assert_not_called()

    @patch(
        "sm_resilient_launcher._launcher.RaceEngine.execute",
    )
    def test_explicit_race(self, mock_execute):
        mock_execute.return_value = _success_result(strategy_used="race")

        launcher = Launcher(
            adapter=_make_adapter(),
            error_classifier=ErrorClassifier,
            intelligent_router=MagicMock(),
            type_expander=MagicMock(),
        )

        result = launcher.launch(
            instance_types=["ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            sdk_params={"Strategy": "race"},
            region="us-east-1",
        )

        mock_execute.assert_called_once()
        # No AI expansion when strategy is explicit.
        launcher._intelligent_router.generate_plan.assert_not_called()
        launcher._type_expander.expand.assert_not_called()


# ---------------------------------------------------------------------------
# 6. List + no Strategy → sequential with caller's exact order
# ---------------------------------------------------------------------------


class TestListDefaultSequential:
    """List + no Strategy → sequential with caller's exact order.

    Req 2.5: The Launcher SHALL execute sequential using the caller's
    list in the exact order provided, with no AI reordering.

    Property 3: Caller list order invariant.
    """

    @patch(
        "sm_resilient_launcher._launcher.SequentialEngine.execute",
    )
    def test_caller_order_preserved(self, mock_execute):
        mock_execute.return_value = _success_result()

        caller_types = [
            "ml.g5.4xlarge",
            "ml.g4dn.2xlarge",
            "ml.g4dn.xlarge",
        ]

        launcher = Launcher(
            adapter=_make_adapter(),
            error_classifier=ErrorClassifier,
            intelligent_router=MagicMock(),
            type_expander=MagicMock(),
        )

        result = launcher.launch(
            instance_types=caller_types,
            boto3_params=BOTO3_PARAMS,
            sdk_params={},
            region="us-east-1",
        )

        engine_call = mock_execute.call_args
        assert engine_call.kwargs["instance_types"] == caller_types
        # No AI expansion for list mode.
        launcher._intelligent_router.generate_plan.assert_not_called()
        launcher._type_expander.expand.assert_not_called()

    @patch(
        "sm_resilient_launcher._launcher.SequentialEngine.execute",
    )
    def test_list_never_reordered(self, mock_execute):
        """The SDK never reorders a caller-provided list (Req 2.7)."""
        mock_execute.return_value = _success_result()

        # Descending cost order — SDK must not "optimize" this.
        caller_types = [
            "ml.p5.48xlarge",
            "ml.p4d.24xlarge",
            "ml.g5.xlarge",
        ]

        launcher = Launcher(
            adapter=_make_adapter(),
            error_classifier=ErrorClassifier,
        )

        result = launcher.launch(
            instance_types=caller_types,
            boto3_params=BOTO3_PARAMS,
            sdk_params={},
            region="us-east-1",
        )

        engine_call = mock_execute.call_args
        assert engine_call.kwargs["instance_types"] is caller_types


# ---------------------------------------------------------------------------
# 7. List + explicit Strategy → execute that strategy with caller's order
# ---------------------------------------------------------------------------


class TestListExplicitStrategy:
    """List + explicit Strategy → execute specified strategy.

    Req 2.6: The Launcher SHALL execute the specified strategy using
    the caller's list in the exact order provided.
    """

    @patch(
        "sm_resilient_launcher._launcher.RaceEngine.execute",
    )
    def test_list_with_race(self, mock_execute):
        mock_execute.return_value = _success_result(strategy_used="race")

        caller_types = ["ml.g5.4xlarge", "ml.g4dn.xlarge"]

        launcher = Launcher(
            adapter=_make_adapter(),
            error_classifier=ErrorClassifier,
        )

        result = launcher.launch(
            instance_types=caller_types,
            boto3_params=BOTO3_PARAMS,
            sdk_params={"Strategy": "race"},
            region="us-east-1",
        )

        engine_call = mock_execute.call_args
        assert engine_call.kwargs["instance_types"] == caller_types

    @patch(
        "sm_resilient_launcher._launcher.SequentialEngine.execute",
    )
    def test_list_with_sequential(self, mock_execute):
        mock_execute.return_value = _success_result()

        caller_types = ["ml.g5.4xlarge", "ml.g4dn.xlarge"]

        launcher = Launcher(
            adapter=_make_adapter(),
            error_classifier=ErrorClassifier,
        )

        result = launcher.launch(
            instance_types=caller_types,
            boto3_params=BOTO3_PARAMS,
            sdk_params={"Strategy": "sequential"},
            region="us-east-1",
        )

        engine_call = mock_execute.call_args
        assert engine_call.kwargs["instance_types"] == caller_types


# ---------------------------------------------------------------------------
# 8. List + multi-region + explicit Strategy → cross-region
# ---------------------------------------------------------------------------


class TestListMultiRegion:
    """List + multi-region triggers CrossRegionManager."""

    @patch(
        "sm_resilient_launcher._launcher.CrossRegionManager",
    )
    def test_list_with_cross_region(self, MockCRM):
        mock_crm = MagicMock()
        mock_crm.execute.return_value = _success_result(
            data_replicated=True,
        )
        MockCRM.return_value = mock_crm

        caller_types = ["ml.g5.4xlarge", "ml.g4dn.xlarge"]

        launcher = Launcher(
            adapter=_make_adapter(),
            error_classifier=ErrorClassifier,
            adapter_factory=MagicMock(),
        )

        result = launcher.launch(
            instance_types=caller_types,
            boto3_params=BOTO3_PARAMS,
            sdk_params={
                "Strategy": "race",
                "AllowedRegions": ["us-east-1", "eu-west-1"],
            },
            region="us-east-1",
        )

        MockCRM.assert_called_once()
        mock_crm.execute.assert_called_once_with(
            caller_types, BOTO3_PARAMS, "my-job"
        )

        # Verify strategy_engine_factory builds RaceEngine.
        crm_kwargs = MockCRM.call_args.kwargs
        factory = crm_kwargs["strategy_engine_factory"]
        engine = factory(_make_adapter())
        from sm_resilient_launcher._race_engine import RaceEngine

        assert isinstance(engine, RaceEngine)


# ---------------------------------------------------------------------------
# 9. _build_engine factory
# ---------------------------------------------------------------------------


class TestBuildEngine:
    """_build_engine returns the correct engine type."""

    def test_sequential(self):
        launcher = Launcher(
            adapter=_make_adapter(),
            error_classifier=ErrorClassifier,
        )
        from sm_resilient_launcher._sequential_engine import SequentialEngine

        engine = launcher._build_engine("sequential", _make_adapter())
        assert isinstance(engine, SequentialEngine)

    def test_race(self):
        launcher = Launcher(
            adapter=_make_adapter(),
            error_classifier=ErrorClassifier,
        )
        from sm_resilient_launcher._race_engine import RaceEngine

        engine = launcher._build_engine("race", _make_adapter())
        assert isinstance(engine, RaceEngine)

    def test_unknown_strategy_raises(self):
        launcher = Launcher(
            adapter=_make_adapter(),
            error_classifier=ErrorClassifier,
        )
        with pytest.raises(ValueError, match="Unknown strategy"):
            launcher._build_engine("invalid", _make_adapter())


# ---------------------------------------------------------------------------
# 10. Single AllowedRegion does NOT trigger cross-region
# ---------------------------------------------------------------------------


class TestSingleAllowedRegion:
    """AllowedRegions with 1 region does not trigger cross-region."""

    @patch(
        "sm_resilient_launcher._launcher.SequentialEngine.execute",
    )
    @patch(
        "sm_resilient_launcher._launcher.CrossRegionManager",
    )
    def test_single_region_no_cross_region(self, MockCRM, mock_execute):
        mock_execute.return_value = _success_result()

        type_expander = MagicMock()
        type_expander.expand.return_value = ["ml.g5.4xlarge"]

        launcher = Launcher(
            adapter=_make_adapter(),
            error_classifier=ErrorClassifier,
            type_expander=type_expander,
        )

        result = launcher.launch(
            instance_types=["ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            sdk_params={"AllowedRegions": ["us-east-1"]},
            region="us-east-1",
        )

        # CrossRegionManager should NOT have been instantiated.
        MockCRM.assert_not_called()
        # TypeExpander should be called (scalar, single region, no strategy).
        type_expander.expand.assert_called_once()


# ---------------------------------------------------------------------------
# 11. base_job_name extraction from boto3_params
# ---------------------------------------------------------------------------


class TestBaseJobName:
    """base_job_name is extracted from TransformJobName in boto3_params."""

    @patch(
        "sm_resilient_launcher._launcher.SequentialEngine.execute",
    )
    def test_uses_transform_job_name(self, mock_execute):
        mock_execute.return_value = _success_result()

        launcher = Launcher(
            adapter=_make_adapter(),
            error_classifier=ErrorClassifier,
        )

        result = launcher.launch(
            instance_types=["ml.g5.4xlarge", "ml.g4dn.xlarge"],
            boto3_params={**BOTO3_PARAMS, "TransformJobName": "custom-base"},
            sdk_params={},
            region="us-east-1",
        )

        engine_call = mock_execute.call_args
        assert engine_call.kwargs["base_job_name"] == "custom-base"

    @patch(
        "sm_resilient_launcher._launcher.SequentialEngine.execute",
    )
    def test_default_base_job_name(self, mock_execute):
        mock_execute.return_value = _success_result()

        params_no_name = {
            "ModelName": "my-model",
            "TransformResources": {"InstanceCount": 1},
        }

        launcher = Launcher(
            adapter=_make_adapter(),
            error_classifier=ErrorClassifier,
        )

        result = launcher.launch(
            instance_types=["ml.g5.4xlarge"],
            boto3_params=params_no_name,
            sdk_params={"Strategy": "sequential"},
            region="us-east-1",
        )

        engine_call = mock_execute.call_args
        assert engine_call.kwargs["base_job_name"] == "sm-resilient"


# ---------------------------------------------------------------------------
# 12. LaunchResult fields — region and strategy are set
# ---------------------------------------------------------------------------


class TestLaunchResultFields:
    """LaunchResult has all required fields (Req 15.1)."""

    @patch(
        "sm_resilient_launcher._launcher.SequentialEngine.execute",
    )
    def test_result_passthrough(self, mock_execute):
        """The LaunchResult from the engine is returned as-is (plus reasoning)."""
        expected = _success_result(
            job_name="j1",
            instance_type_used="ml.g5.4xlarge",
            region_used="us-east-1",
            attempts=2,
            strategy_used="sequential",
        )
        mock_execute.return_value = expected

        launcher = Launcher(
            adapter=_make_adapter(),
            error_classifier=ErrorClassifier,
        )

        result = launcher.launch(
            instance_types=["ml.g5.4xlarge", "ml.g4dn.xlarge"],
            boto3_params=BOTO3_PARAMS,
            sdk_params={"Strategy": "sequential"},
            region="us-east-1",
        )

        assert result.job_name == "j1"
        assert result.instance_type_used == "ml.g5.4xlarge"
        assert result.region_used == "us-east-1"
        assert result.attempts == 2
        assert result.strategy_used == "sequential"


# ---------------------------------------------------------------------------
# 13. WorkloadDescription ignored when Strategy is explicit
# ---------------------------------------------------------------------------


class TestWorkloadIgnoredWithExplicitStrategy:
    """WorkloadDescription is ignored when an explicit Strategy is set."""

    @patch(
        "sm_resilient_launcher._launcher.SequentialEngine.execute",
    )
    def test_no_ai_expansion_with_explicit_strategy(self, mock_execute):
        mock_execute.return_value = _success_result()

        intelligent_router = MagicMock()
        type_expander = MagicMock()

        launcher = Launcher(
            adapter=_make_adapter(),
            error_classifier=ErrorClassifier,
            intelligent_router=intelligent_router,
            type_expander=type_expander,
        )

        result = launcher.launch(
            instance_types=["ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            sdk_params={
                "Strategy": "sequential",
                "WorkloadDescription": "This should be ignored",
            },
            region="us-east-1",
        )

        intelligent_router.generate_plan.assert_not_called()
        type_expander.expand.assert_not_called()
        assert result.agent_reasoning is None
