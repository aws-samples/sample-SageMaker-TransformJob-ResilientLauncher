"""Unit tests for SequentialEngine.

Property 8: Sequential engine preserves attempt order and reports result
    For any ordered list of instance types in sequential mode, the
    SequentialEngine SHALL attempt types in the exact order provided
    (skipping globally-excluded types), and on success the
    LaunchResult.attempts SHALL equal the count of types actually
    attempted, instance_type_used SHALL be the type that succeeded,
    and errors SHALL contain one entry per failed attempt.

Property 9: Family support errors skip type globally
    For any instance type that fails with a family_support error
    classification, the engine SHALL add that type to
    globally_skipped_types, ensuring it is skipped in all subsequent
    regions.

Property 10: Exhaustion raises with full error details
    For any launch where all instance types have been exhausted, the
    raised AllTypesExhaustedError SHALL contain an errors list with
    one ClassifiedError per failed attempt.

Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from sm_resilient_launcher._error_classifier import (
    ClassifiedError,
    ErrorCategory,
    ErrorClassifier,
)
from sm_resilient_launcher._exceptions import AllTypesExhaustedError
from sm_resilient_launcher._launch_result import LaunchResult
from sm_resilient_launcher._sequential_engine import SequentialEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(
    create_side_effects: list | None = None,
    describe_side_effects: list | None = None,
    completed_types: set[str] | None = None,
    failed_types: set[str] | None = None,
) -> MagicMock:
    """Build a mock JobAdapter with configurable behaviour.

    Args:
        create_side_effects: Per-call side effects for ``create_job``.
            Use ``Exception(...)`` entries to simulate create-time errors.
        describe_side_effects: Per-call side effects for ``describe_job``.
            Each entry is a dict returned by describe_job.
        completed_types: Instance types whose describe returns a
            "completed" status.
        failed_types: Instance types whose describe returns a
            "failed" status (with an error message as the dict value).
    """
    adapter = MagicMock()

    if create_side_effects is not None:
        adapter.create_job.side_effect = create_side_effects
    else:
        adapter.create_job.return_value = {}

    # Track which job names map to which instance types.
    _job_to_type: dict[str, str] = {}
    _original_create = adapter.create_job.side_effect

    completed_types = completed_types or set()
    failed_types = failed_types or set()

    def _track_create(job_name: str, instance_type: str, boto3_params: dict):
        _job_to_type[job_name] = instance_type
        if _original_create:
            # Call the original side_effect (may raise).
            if callable(_original_create):
                return _original_create(job_name, instance_type, boto3_params)
        return {}

    if create_side_effects is None:
        adapter.create_job.side_effect = _track_create

    if describe_side_effects is not None:
        adapter.describe_job.side_effect = describe_side_effects
    else:
        def _describe(job_name: str):
            itype = _job_to_type.get(job_name, "")
            if itype in completed_types:
                return {"TransformJobStatus": "Completed"}
            if itype in failed_types:
                return {
                    "TransformJobStatus": "Failed",
                    "FailureReason": failed_types[itype]
                    if isinstance(failed_types, dict)
                    else "Unknown error",
                }
            return {"TransformJobStatus": "InProgress"}

        adapter.describe_job.side_effect = _describe

    def _is_completed(desc: dict) -> bool:
        return desc.get("TransformJobStatus") == "Completed"

    def _is_failed(desc: dict) -> bool:
        return desc.get("TransformJobStatus") in ("Failed", "Stopped")

    def _extract_error(desc: dict) -> str:
        return desc.get("FailureReason", "Unknown error")

    adapter.is_completed.side_effect = _is_completed
    adapter.is_failed.side_effect = _is_failed
    adapter.extract_error.side_effect = _extract_error

    return adapter


def _make_failing_adapter(failure_messages: dict[str, str]) -> MagicMock:
    """Build a mock adapter where every type fails with a specific message.

    Args:
        failure_messages: Mapping of instance_type -> error message.
    """
    adapter = MagicMock()
    _job_to_type: dict[str, str] = {}

    def _create(job_name, instance_type, boto3_params):
        _job_to_type[job_name] = instance_type
        return {}

    adapter.create_job.side_effect = _create

    def _describe(job_name):
        itype = _job_to_type.get(job_name, "")
        msg = failure_messages.get(itype, "Unknown error")
        return {"TransformJobStatus": "Failed", "FailureReason": msg}

    adapter.describe_job.side_effect = _describe

    adapter.is_completed.side_effect = (
        lambda d: d.get("TransformJobStatus") == "Completed"
    )
    adapter.is_failed.side_effect = (
        lambda d: d.get("TransformJobStatus") in ("Failed", "Stopped")
    )
    adapter.extract_error.side_effect = (
        lambda d: d.get("FailureReason", "Unknown error")
    )

    return adapter


BOTO3_PARAMS = {
    "ModelName": "my-model",
    "TransformInput": {
        "DataSource": {"S3DataSource": {"S3Uri": "s3://bucket/input"}}
    },
    "TransformOutput": {"S3OutputPath": "s3://bucket/output"},
    "TransformResources": {"InstanceCount": 1},
}


# ---------------------------------------------------------------------------
# 1. Ordered traversal — types are attempted in exact caller order
# ---------------------------------------------------------------------------


class TestOrderedTraversal:
    """The engine attempts instance types in the exact order provided."""

    @patch("sm_resilient_launcher._sequential_engine.time.sleep")
    def test_tries_types_in_order_until_success(self, mock_sleep):
        """Second type succeeds; first type must have been tried first."""
        adapter = _make_failing_adapter(
            {"ml.g4dn.xlarge": "CapacityError: no capacity"}
        )
        # Override: second type succeeds.
        call_count = {"n": 0}

        def _describe(job_name):
            call_count["n"] += 1
            if "ml-g5-4xlarge" in job_name:
                return {"TransformJobStatus": "Completed"}
            return {
                "TransformJobStatus": "Failed",
                "FailureReason": "CapacityError: no capacity",
            }

        adapter.describe_job.side_effect = _describe

        engine = SequentialEngine(adapter)
        result = engine.execute(
            instance_types=["ml.g4dn.xlarge", "ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="us-east-1",
        )

        assert result.instance_type_used == "ml.g5.4xlarge"
        # create_job should have been called with g4dn first, g5 second.
        create_calls = adapter.create_job.call_args_list
        assert "ml.g4dn.xlarge" in str(create_calls[0])
        assert "ml.g5.4xlarge" in str(create_calls[1])

    @patch("sm_resilient_launcher._sequential_engine.time.sleep")
    def test_first_type_succeeds_immediately(self, mock_sleep):
        """When the first type succeeds, only one attempt is made."""
        adapter = MagicMock()
        adapter.create_job.return_value = {}
        adapter.describe_job.return_value = {"TransformJobStatus": "Completed"}
        adapter.is_completed.side_effect = (
            lambda d: d.get("TransformJobStatus") == "Completed"
        )
        adapter.is_failed.side_effect = (
            lambda d: d.get("TransformJobStatus") in ("Failed", "Stopped")
        )

        engine = SequentialEngine(adapter)
        result = engine.execute(
            instance_types=["ml.g5.4xlarge", "ml.g4dn.xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="us-east-1",
        )

        assert result.instance_type_used == "ml.g5.4xlarge"
        assert result.attempts == 1
        assert adapter.create_job.call_count == 1


# ---------------------------------------------------------------------------
# 2. LaunchResult fields are populated correctly on success
# ---------------------------------------------------------------------------


class TestLaunchResultOnSuccess:
    """A successful launch returns a correctly populated LaunchResult."""

    @patch("sm_resilient_launcher._sequential_engine.time.sleep")
    def test_result_fields(self, mock_sleep):
        adapter = _make_failing_adapter(
            {"ml.g4dn.xlarge": "CapacityError: no capacity"}
        )

        def _describe(job_name):
            if "ml-g5-4xlarge" in job_name:
                return {"TransformJobStatus": "Completed"}
            return {
                "TransformJobStatus": "Failed",
                "FailureReason": "CapacityError: no capacity",
            }

        adapter.describe_job.side_effect = _describe

        engine = SequentialEngine(adapter)
        result = engine.execute(
            instance_types=["ml.g4dn.xlarge", "ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="test-job",
            region="us-west-2",
        )

        assert isinstance(result, LaunchResult)
        assert result.instance_type_used == "ml.g5.4xlarge"
        assert result.region_used == "us-west-2"
        assert result.attempts == 2
        assert result.strategy_used == "sequential"
        assert len(result.errors) == 1
        assert result.errors[0].category == ErrorCategory.CAPACITY


# ---------------------------------------------------------------------------
# 3. Capacity error skips type in current region (try next type)
# ---------------------------------------------------------------------------


class TestCapacityErrorSkips:
    """A capacity error skips the type in the current region only."""

    @patch("sm_resilient_launcher._sequential_engine.time.sleep")
    def test_capacity_error_moves_to_next_type(self, mock_sleep):
        adapter = _make_failing_adapter(
            {
                "ml.g4dn.xlarge": "CapacityError: no capacity",
                "ml.g5.xlarge": "CapacityError: no capacity",
            }
        )

        def _describe(job_name):
            if "ml-g5-4xlarge" in job_name:
                return {"TransformJobStatus": "Completed"}
            return {
                "TransformJobStatus": "Failed",
                "FailureReason": "CapacityError: no capacity",
            }

        adapter.describe_job.side_effect = _describe

        globally_skipped = set()
        engine = SequentialEngine(adapter)
        result = engine.execute(
            instance_types=[
                "ml.g4dn.xlarge",
                "ml.g5.xlarge",
                "ml.g5.4xlarge",
            ],
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="us-east-1",
            globally_skipped_types=globally_skipped,
        )

        assert result.instance_type_used == "ml.g5.4xlarge"
        # Capacity errors do NOT add to global skip set.
        assert "ml.g4dn.xlarge" not in globally_skipped
        assert "ml.g5.xlarge" not in globally_skipped


# ---------------------------------------------------------------------------
# 4. Quota error skips type in current region
# ---------------------------------------------------------------------------


class TestQuotaErrorSkips:
    """A quota error skips the type in the current region only."""

    @patch("sm_resilient_launcher._sequential_engine.time.sleep")
    def test_quota_error_moves_to_next_type(self, mock_sleep):
        adapter = _make_failing_adapter(
            {"ml.g4dn.xlarge": "ResourceLimitExceeded: limit reached"}
        )

        def _describe(job_name):
            if "ml-g5-4xlarge" in job_name:
                return {"TransformJobStatus": "Completed"}
            return {
                "TransformJobStatus": "Failed",
                "FailureReason": "ResourceLimitExceeded: limit reached",
            }

        adapter.describe_job.side_effect = _describe

        globally_skipped = set()
        engine = SequentialEngine(adapter)
        result = engine.execute(
            instance_types=["ml.g4dn.xlarge", "ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="us-east-1",
            globally_skipped_types=globally_skipped,
        )

        assert result.instance_type_used == "ml.g5.4xlarge"
        assert "ml.g4dn.xlarge" not in globally_skipped


# ---------------------------------------------------------------------------
# 5. Family support error adds type to globally_skipped_types
# ---------------------------------------------------------------------------


class TestFamilySupportGlobalSkip:
    """A family_support error adds the type to the global skip set."""

    @patch("sm_resilient_launcher._sequential_engine.time.sleep")
    def test_family_support_adds_to_global_skip(self, mock_sleep):
        adapter = _make_failing_adapter(
            {"ml.p4d.24xlarge": "ml.p4d.24xlarge is not supported for Batch Transform"}
        )

        def _describe(job_name):
            if "ml-g5-4xlarge" in job_name:
                return {"TransformJobStatus": "Completed"}
            return {
                "TransformJobStatus": "Failed",
                "FailureReason": "ml.p4d.24xlarge is not supported for Batch Transform",
            }

        adapter.describe_job.side_effect = _describe

        globally_skipped: set[str] = set()
        engine = SequentialEngine(adapter)
        result = engine.execute(
            instance_types=["ml.p4d.24xlarge", "ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="us-east-1",
            globally_skipped_types=globally_skipped,
        )

        assert result.instance_type_used == "ml.g5.4xlarge"
        # p4d must now be in the global skip set.
        assert "ml.p4d.24xlarge" in globally_skipped

    @patch("sm_resilient_launcher._sequential_engine.time.sleep")
    def test_globally_skipped_types_are_skipped(self, mock_sleep):
        """Types already in the global skip set are never attempted."""
        adapter = MagicMock()
        adapter.create_job.return_value = {}
        adapter.describe_job.return_value = {"TransformJobStatus": "Completed"}
        adapter.is_completed.side_effect = (
            lambda d: d.get("TransformJobStatus") == "Completed"
        )
        adapter.is_failed.side_effect = (
            lambda d: d.get("TransformJobStatus") in ("Failed", "Stopped")
        )

        engine = SequentialEngine(adapter)
        result = engine.execute(
            instance_types=["ml.p4d.24xlarge", "ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="eu-west-1",
            globally_skipped_types={"ml.p4d.24xlarge"},
        )

        # Only g5 should have been attempted (p4d is globally skipped).
        assert result.instance_type_used == "ml.g5.4xlarge"
        assert result.attempts == 1
        assert adapter.create_job.call_count == 1
        create_args = adapter.create_job.call_args
        assert create_args[0][1] == "ml.g5.4xlarge"


# ---------------------------------------------------------------------------
# 6. Unknown error skips type, tries next
# ---------------------------------------------------------------------------


class TestUnknownErrorSkips:
    """An unknown error classification skips the type and moves on."""

    @patch("sm_resilient_launcher._sequential_engine.time.sleep")
    def test_unknown_error_moves_to_next(self, mock_sleep):
        adapter = _make_failing_adapter(
            {"ml.g4dn.xlarge": "Some unexpected internal error"}
        )

        def _describe(job_name):
            if "ml-g5-4xlarge" in job_name:
                return {"TransformJobStatus": "Completed"}
            return {
                "TransformJobStatus": "Failed",
                "FailureReason": "Some unexpected internal error",
            }

        adapter.describe_job.side_effect = _describe

        globally_skipped: set[str] = set()
        engine = SequentialEngine(adapter)
        result = engine.execute(
            instance_types=["ml.g4dn.xlarge", "ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="us-east-1",
            globally_skipped_types=globally_skipped,
        )

        assert result.instance_type_used == "ml.g5.4xlarge"
        assert result.errors[0].category == ErrorCategory.UNKNOWN
        # Unknown errors do NOT add to global skip.
        assert "ml.g4dn.xlarge" not in globally_skipped


# ---------------------------------------------------------------------------
# 7. AllTypesExhaustedError raised when all types fail
# ---------------------------------------------------------------------------


class TestAllTypesExhausted:
    """The engine raises AllTypesExhaustedError with all classified errors."""

    @patch("sm_resilient_launcher._sequential_engine.time.sleep")
    def test_raises_with_all_errors(self, mock_sleep):
        adapter = _make_failing_adapter(
            {
                "ml.g4dn.xlarge": "CapacityError: no capacity",
                "ml.g5.4xlarge": "ResourceLimitExceeded: limit reached",
            }
        )

        engine = SequentialEngine(adapter)

        with pytest.raises(AllTypesExhaustedError) as exc_info:
            engine.execute(
                instance_types=["ml.g4dn.xlarge", "ml.g5.4xlarge"],
                boto3_params=BOTO3_PARAMS,
                base_job_name="job",
                region="us-east-1",
            )

        err = exc_info.value
        assert len(err.errors) == 2
        categories = {e.category for e in err.errors}
        assert ErrorCategory.CAPACITY in categories
        assert ErrorCategory.QUOTA in categories

    @patch("sm_resilient_launcher._sequential_engine.time.sleep")
    def test_error_entries_annotated(self, mock_sleep):
        """Each ClassifiedError entry has instance_type, region, and category."""
        adapter = _make_failing_adapter(
            {"ml.g5.xlarge": "unable to provision requested capacity"}
        )

        engine = SequentialEngine(adapter)

        with pytest.raises(AllTypesExhaustedError) as exc_info:
            engine.execute(
                instance_types=["ml.g5.xlarge"],
                boto3_params=BOTO3_PARAMS,
                base_job_name="job",
                region="us-west-2",
            )

        error = exc_info.value.errors[0]
        assert error.instance_type == "ml.g5.xlarge"
        assert error.region == "us-west-2"
        assert error.category == ErrorCategory.CAPACITY


# ---------------------------------------------------------------------------
# 8. create_job exception is caught and classified
# ---------------------------------------------------------------------------


class TestCreateJobExceptionHandling:
    """Exceptions thrown by create_job are caught and classified."""

    @patch("sm_resilient_launcher._sequential_engine.time.sleep")
    def test_create_job_exception_classified(self, mock_sleep):
        adapter = MagicMock()
        adapter.create_job.side_effect = [
            Exception("CapacityError: ml.g4dn.xlarge no available"),
            {},  # second call succeeds
        ]
        adapter.describe_job.return_value = {"TransformJobStatus": "Completed"}
        adapter.is_completed.side_effect = (
            lambda d: d.get("TransformJobStatus") == "Completed"
        )
        adapter.is_failed.side_effect = (
            lambda d: d.get("TransformJobStatus") in ("Failed", "Stopped")
        )

        engine = SequentialEngine(adapter)
        result = engine.execute(
            instance_types=["ml.g4dn.xlarge", "ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="us-east-1",
        )

        assert result.instance_type_used == "ml.g5.4xlarge"
        assert len(result.errors) == 1
        assert result.errors[0].category == ErrorCategory.CAPACITY


# ---------------------------------------------------------------------------
# 9. Job name generation uses expected pattern
# ---------------------------------------------------------------------------


class TestJobNameGeneration:
    """Job names follow the pattern {base}-{idx}-{sanitized_type}."""

    @patch("sm_resilient_launcher._sequential_engine.time.sleep")
    def test_job_name_format(self, mock_sleep):
        adapter = MagicMock()
        adapter.create_job.return_value = {}
        adapter.describe_job.return_value = {"TransformJobStatus": "Completed"}
        adapter.is_completed.side_effect = (
            lambda d: d.get("TransformJobStatus") == "Completed"
        )
        adapter.is_failed.side_effect = (
            lambda d: d.get("TransformJobStatus") in ("Failed", "Stopped")
        )

        engine = SequentialEngine(adapter)
        engine.execute(
            instance_types=["ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="my-job",
            region="us-east-1",
        )

        job_name_used = adapter.create_job.call_args[0][0]
        assert job_name_used == "my-job-0-ml-g5-4xlarge"


# ---------------------------------------------------------------------------
# 10. Empty instance_types list raises immediately
# ---------------------------------------------------------------------------


class TestEmptyInstanceTypes:
    """An empty instance_types list raises AllTypesExhaustedError immediately."""

    def test_raises_immediately(self):
        adapter = MagicMock()
        engine = SequentialEngine(adapter)

        with pytest.raises(AllTypesExhaustedError) as exc_info:
            engine.execute(
                instance_types=[],
                boto3_params=BOTO3_PARAMS,
                base_job_name="job",
                region="us-east-1",
            )

        assert len(exc_info.value.errors) == 0


# ---------------------------------------------------------------------------
# 11. Polling loops until terminal state
# ---------------------------------------------------------------------------


class TestPollingBehaviour:
    """_poll_until_terminal loops until is_completed or is_failed is True."""

    @patch("sm_resilient_launcher._sequential_engine.time.sleep")
    def test_polls_until_completed(self, mock_sleep):
        adapter = MagicMock()
        adapter.create_job.return_value = {}
        adapter.describe_job.side_effect = [
            {"TransformJobStatus": "InProgress"},
            {"TransformJobStatus": "InProgress"},
            {"TransformJobStatus": "Completed"},
        ]
        adapter.is_completed.side_effect = (
            lambda d: d.get("TransformJobStatus") == "Completed"
        )
        adapter.is_failed.side_effect = (
            lambda d: d.get("TransformJobStatus") in ("Failed", "Stopped")
        )

        engine = SequentialEngine(adapter)
        result = engine.execute(
            instance_types=["ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="us-east-1",
        )

        assert result.instance_type_used == "ml.g5.4xlarge"
        # describe_job called 3 times (2 InProgress + 1 Completed).
        assert adapter.describe_job.call_count == 3
        # sleep called 2 times (once per InProgress poll).
        assert mock_sleep.call_count == 2
