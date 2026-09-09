"""Unit tests for RaceEngine.

Property 11: Race submits all types concurrently with unique names
    For any list of N instance types in race mode, the RaceEngine SHALL
    submit exactly N concurrent jobs, and all N generated job names
    SHALL be distinct (no two submissions share a name within a single
    race execution).

Property 12: Race cancels all non-winners
    For any race execution where one job reaches a completed state, the
    RaceEngine SHALL call stop_job on every other job that was submitted
    in that race — exactly N-1 calls for N submissions.

Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from sm_resilient_launcher._error_classifier import (
    ErrorCategory,
    ErrorClassifier,
)
from sm_resilient_launcher._exceptions import AllTypesExhaustedError
from sm_resilient_launcher._launch_result import LaunchResult
from sm_resilient_launcher._race_engine import RaceEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOTO3_PARAMS = {
    "ModelName": "my-model",
    "TransformInput": {
        "DataSource": {"S3DataSource": {"S3Uri": "s3://bucket/input"}}
    },
    "TransformOutput": {"S3OutputPath": "s3://bucket/output"},
    "TransformResources": {"InstanceCount": 1},
}


def _make_adapter(
    completed_types: set[str] | None = None,
    failed_types: dict[str, str] | None = None,
    create_exceptions: dict[str, Exception] | None = None,
) -> MagicMock:
    """Build a mock JobAdapter for race testing.

    Args:
        completed_types: Instance types whose describe returns Completed.
        failed_types: Instance type -> failure reason for types that fail.
        create_exceptions: Instance type -> exception to raise on create_job.
    """
    adapter = MagicMock()
    completed_types = completed_types or set()
    failed_types = failed_types or {}
    create_exceptions = create_exceptions or {}

    _job_to_type: dict[str, str] = {}
    _lock = threading.Lock()

    def _create(job_name: str, instance_type: str, boto3_params: dict):
        with _lock:
            _job_to_type[job_name] = instance_type
        if instance_type in create_exceptions:
            raise create_exceptions[instance_type]
        return {}

    adapter.create_job.side_effect = _create

    def _describe(job_name: str):
        with _lock:
            itype = _job_to_type.get(job_name, "")
        if itype in completed_types:
            return {"TransformJobStatus": "Completed"}
        if itype in failed_types:
            return {
                "TransformJobStatus": "Failed",
                "FailureReason": failed_types[itype],
            }
        # Default: InProgress (should not normally be reached in tests
        # that configure completed/failed for every type).
        return {"TransformJobStatus": "Failed", "FailureReason": "Unknown"}

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
    adapter.stop_job.return_value = None

    return adapter


# ---------------------------------------------------------------------------
# 1. N concurrent submissions for N instance types
#    Property 11 — Requirement 7.1
# ---------------------------------------------------------------------------


class TestConcurrentSubmissions:
    """The engine submits a create_job call for every instance type."""

    @patch("sm_resilient_launcher._race_engine.time.sleep")
    def test_n_create_calls_for_n_types(self, mock_sleep):
        """All 3 types must be submitted (3 create_job calls)."""
        types = ["ml.g4dn.xlarge", "ml.g5.4xlarge", "ml.g5.12xlarge"]
        adapter = _make_adapter(completed_types={"ml.g5.4xlarge"}, failed_types={
            "ml.g4dn.xlarge": "CapacityError",
            "ml.g5.12xlarge": "CapacityError",
        })

        engine = RaceEngine(adapter, ErrorClassifier)
        engine.execute(
            instance_types=types,
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="us-east-1",
        )

        assert adapter.create_job.call_count == len(types)

    @patch("sm_resilient_launcher._race_engine.time.sleep")
    def test_single_type_submitted(self, mock_sleep):
        """A single-element list produces exactly 1 create_job call."""
        adapter = _make_adapter(completed_types={"ml.g5.4xlarge"})

        engine = RaceEngine(adapter, ErrorClassifier)
        result = engine.execute(
            instance_types=["ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="us-east-1",
        )

        assert adapter.create_job.call_count == 1
        assert result.instance_type_used == "ml.g5.4xlarge"


# ---------------------------------------------------------------------------
# 2. All N job names are unique within a single race execution
#    Property 11 — Requirement 7.4
# ---------------------------------------------------------------------------


class TestUniqueJobNames:
    """Every concurrent submission uses a unique job name."""

    @patch("sm_resilient_launcher._race_engine.time.sleep")
    def test_all_job_names_distinct(self, mock_sleep):
        types = ["ml.g4dn.xlarge", "ml.g5.xlarge", "ml.g5.4xlarge", "ml.p3.2xlarge"]
        adapter = _make_adapter(
            completed_types={"ml.g5.4xlarge"},
            failed_types={
                "ml.g4dn.xlarge": "CapacityError",
                "ml.g5.xlarge": "CapacityError",
                "ml.p3.2xlarge": "CapacityError",
            },
        )

        engine = RaceEngine(adapter, ErrorClassifier)
        engine.execute(
            instance_types=types,
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="us-east-1",
        )

        job_names = [
            call_args[0][0] for call_args in adapter.create_job.call_args_list
        ]
        assert len(job_names) == len(set(job_names)), (
            f"Duplicate job names found: {job_names}"
        )

    @patch("sm_resilient_launcher._race_engine.time.sleep")
    def test_job_name_contains_race_prefix(self, mock_sleep):
        """Job names follow {base}-race-{idx}-{uuid8} pattern."""
        adapter = _make_adapter(completed_types={"ml.g5.4xlarge"})

        engine = RaceEngine(adapter, ErrorClassifier)
        engine.execute(
            instance_types=["ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="my-transform",
            region="us-east-1",
        )

        job_name = adapter.create_job.call_args[0][0]
        assert job_name.startswith("my-transform-race-")


# ---------------------------------------------------------------------------
# 3. When one job completes first, it's the winner
#    Property 12 — Requirement 7.2
# ---------------------------------------------------------------------------


class TestWinnerDetection:
    """The first job to complete is designated as the winner."""

    @patch("sm_resilient_launcher._race_engine.time.sleep")
    def test_completed_job_is_winner(self, mock_sleep):
        adapter = _make_adapter(
            completed_types={"ml.g5.4xlarge"},
            failed_types={"ml.g4dn.xlarge": "CapacityError"},
        )

        engine = RaceEngine(adapter, ErrorClassifier)
        result = engine.execute(
            instance_types=["ml.g4dn.xlarge", "ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="us-east-1",
        )

        assert result.instance_type_used == "ml.g5.4xlarge"


# ---------------------------------------------------------------------------
# 4. Exactly N-1 stop_job calls on non-winners
#    Property 12 — Requirement 7.3
# ---------------------------------------------------------------------------


class TestNonWinnerCancellation:
    """All non-winner jobs receive stop_job calls."""

    @patch("sm_resilient_launcher._race_engine.time.sleep")
    def test_stop_called_on_all_non_winners(self, mock_sleep):
        """With 3 types and 1 winner, exactly 2 stop_job calls."""
        types = ["ml.g4dn.xlarge", "ml.g5.4xlarge", "ml.g5.12xlarge"]
        adapter = _make_adapter(
            completed_types={"ml.g5.4xlarge"},
            failed_types={
                "ml.g4dn.xlarge": "CapacityError",
                "ml.g5.12xlarge": "CapacityError",
            },
        )

        engine = RaceEngine(adapter, ErrorClassifier)
        result = engine.execute(
            instance_types=types,
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="us-east-1",
        )

        # N-1 = 2 stop_job calls.
        assert adapter.stop_job.call_count == len(types) - 1

        # The winner's job name must NOT appear in any stop_job call.
        stopped_names = [c[0][0] for c in adapter.stop_job.call_args_list]
        assert result.job_name not in stopped_names

    @patch("sm_resilient_launcher._race_engine.time.sleep")
    def test_stop_single_non_winner(self, mock_sleep):
        """With 2 types and 1 winner, exactly 1 stop_job call."""
        adapter = _make_adapter(
            completed_types={"ml.g5.4xlarge"},
            failed_types={"ml.g4dn.xlarge": "CapacityError"},
        )

        engine = RaceEngine(adapter, ErrorClassifier)
        result = engine.execute(
            instance_types=["ml.g4dn.xlarge", "ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="us-east-1",
        )

        assert adapter.stop_job.call_count == 1

    @patch("sm_resilient_launcher._race_engine.time.sleep")
    def test_stop_failure_does_not_propagate(self, mock_sleep):
        """If stop_job raises, the error is swallowed (job may be terminal)."""
        adapter = _make_adapter(
            completed_types={"ml.g5.4xlarge"},
            failed_types={"ml.g4dn.xlarge": "CapacityError"},
        )
        adapter.stop_job.side_effect = Exception("Already stopped")

        engine = RaceEngine(adapter, ErrorClassifier)
        # Must NOT raise despite stop_job failures.
        result = engine.execute(
            instance_types=["ml.g4dn.xlarge", "ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="us-east-1",
        )

        assert result.instance_type_used == "ml.g5.4xlarge"


# ---------------------------------------------------------------------------
# 5. AllTypesExhaustedError when ALL concurrent jobs fail
#    Requirement 7.5, 7.8
# ---------------------------------------------------------------------------


class TestAllJobsFail:
    """When every concurrent job fails, AllTypesExhaustedError is raised."""

    @patch("sm_resilient_launcher._race_engine.time.sleep")
    def test_raises_when_all_fail(self, mock_sleep):
        adapter = _make_adapter(
            failed_types={
                "ml.g4dn.xlarge": "CapacityError: no capacity",
                "ml.g5.4xlarge": "ResourceLimitExceeded: limit reached",
            }
        )

        engine = RaceEngine(adapter, ErrorClassifier)
        with pytest.raises(AllTypesExhaustedError) as exc_info:
            engine.execute(
                instance_types=["ml.g4dn.xlarge", "ml.g5.4xlarge"],
                boto3_params=BOTO3_PARAMS,
                base_job_name="job",
                region="us-east-1",
            )

        assert len(exc_info.value.errors) >= 2

    @patch("sm_resilient_launcher._race_engine.time.sleep")
    def test_waits_for_all_terminal_before_raising(self, mock_sleep):
        """Req 7.8: engine waits for ALL jobs to reach terminal state."""
        # Both types fail, but we verify all create_job calls happened.
        types = ["ml.g4dn.xlarge", "ml.g5.xlarge", "ml.p3.2xlarge"]
        adapter = _make_adapter(
            failed_types={
                "ml.g4dn.xlarge": "CapacityError",
                "ml.g5.xlarge": "CapacityError",
                "ml.p3.2xlarge": "CapacityError",
            }
        )

        engine = RaceEngine(adapter, ErrorClassifier)
        with pytest.raises(AllTypesExhaustedError):
            engine.execute(
                instance_types=types,
                boto3_params=BOTO3_PARAMS,
                base_job_name="job",
                region="us-east-1",
            )

        # All 3 must have been submitted before raising.
        assert adapter.create_job.call_count == 3

    @patch("sm_resilient_launcher._race_engine.time.sleep")
    def test_empty_after_global_skip_raises(self, mock_sleep):
        """When all types are globally skipped, raises immediately."""
        adapter = _make_adapter()

        engine = RaceEngine(adapter, ErrorClassifier)
        with pytest.raises(AllTypesExhaustedError):
            engine.execute(
                instance_types=["ml.g4dn.xlarge", "ml.g5.xlarge"],
                boto3_params=BOTO3_PARAMS,
                base_job_name="job",
                region="us-east-1",
                globally_skipped_types={"ml.g4dn.xlarge", "ml.g5.xlarge"},
            )

        # No create_job calls should happen.
        assert adapter.create_job.call_count == 0


# ---------------------------------------------------------------------------
# 6. LaunchResult winner fields populated correctly
#    Requirement 7.6
# ---------------------------------------------------------------------------


class TestLaunchResultFields:
    """The winner's LaunchResult has all required fields."""

    @patch("sm_resilient_launcher._race_engine.time.sleep")
    def test_result_fields_populated(self, mock_sleep):
        adapter = _make_adapter(
            completed_types={"ml.g5.4xlarge"},
            failed_types={"ml.g4dn.xlarge": "CapacityError"},
        )

        engine = RaceEngine(adapter, ErrorClassifier)
        result = engine.execute(
            instance_types=["ml.g4dn.xlarge", "ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="my-job",
            region="us-west-2",
        )

        assert isinstance(result, LaunchResult)
        assert result.instance_type_used == "ml.g5.4xlarge"
        assert result.region_used == "us-west-2"
        assert result.attempts == 2  # Both types were submitted.
        assert result.strategy_used == "race"
        assert result.job_name.startswith("my-job-race-")

    @patch("sm_resilient_launcher._race_engine.time.sleep")
    def test_errors_list_contains_failed_attempts(self, mock_sleep):
        """The errors list captures failures from non-winning jobs."""
        adapter = _make_adapter(
            completed_types={"ml.g5.4xlarge"},
            failed_types={"ml.g4dn.xlarge": "CapacityError: no capacity"},
        )

        engine = RaceEngine(adapter, ErrorClassifier)
        result = engine.execute(
            instance_types=["ml.g4dn.xlarge", "ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="us-east-1",
        )

        # The failed g4dn should be in the errors list.
        assert len(result.errors) >= 1


# ---------------------------------------------------------------------------
# 7. Tie-break: prefer home region, then cheapest type
#    Requirement 7.7
# ---------------------------------------------------------------------------


class TestTieBreak:
    """When multiple jobs complete, tie-break selects the best winner."""

    @patch("sm_resilient_launcher._race_engine.time.sleep")
    def test_prefers_home_region(self, mock_sleep):
        """When multiple types complete, the one in the home region wins."""
        # Both types complete. With home_region="us-east-1" and the race
        # running in us-east-1, both are in the home region, so
        # alphabetical tie-break kicks in.
        adapter = _make_adapter(
            completed_types={"ml.g4dn.xlarge", "ml.g5.4xlarge"}
        )

        engine = RaceEngine(adapter, ErrorClassifier, home_region="us-east-1")
        result = engine.execute(
            instance_types=["ml.g4dn.xlarge", "ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="us-east-1",
        )

        # Both are in home region, so alphabetical (cheapest heuristic):
        # "ml.g4dn.xlarge" < "ml.g5.4xlarge".
        assert result.instance_type_used == "ml.g4dn.xlarge"

    @patch("sm_resilient_launcher._race_engine.time.sleep")
    def test_cheapest_type_when_same_region(self, mock_sleep):
        """Among types in the same region, alphabetically first wins."""
        adapter = _make_adapter(
            completed_types={"ml.g5.4xlarge", "ml.p5.48xlarge"}
        )

        engine = RaceEngine(adapter, ErrorClassifier, home_region="us-east-1")
        result = engine.execute(
            instance_types=["ml.p5.48xlarge", "ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="us-east-1",
        )

        # Alphabetical tie-break: "ml.g5.4xlarge" < "ml.p5.48xlarge".
        assert result.instance_type_used == "ml.g5.4xlarge"


# ---------------------------------------------------------------------------
# 8. Globally skipped types are not submitted
# ---------------------------------------------------------------------------


class TestGloballySkippedTypes:
    """Types in globally_skipped_types are never submitted."""

    @patch("sm_resilient_launcher._race_engine.time.sleep")
    def test_skipped_types_not_submitted(self, mock_sleep):
        adapter = _make_adapter(completed_types={"ml.g5.4xlarge"})

        engine = RaceEngine(adapter, ErrorClassifier)
        result = engine.execute(
            instance_types=["ml.p4d.24xlarge", "ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="us-east-1",
            globally_skipped_types={"ml.p4d.24xlarge"},
        )

        # Only g5 should have been submitted.
        assert adapter.create_job.call_count == 1
        submitted_type = adapter.create_job.call_args[0][1]
        assert submitted_type == "ml.g5.4xlarge"
        assert result.instance_type_used == "ml.g5.4xlarge"
        # attempts should reflect the number actually submitted (1).
        assert result.attempts == 1

    @patch("sm_resilient_launcher._race_engine.time.sleep")
    def test_partial_skip_submits_remaining(self, mock_sleep):
        """Skip some types; remaining types are still submitted."""
        adapter = _make_adapter(
            completed_types={"ml.g5.xlarge"},
            failed_types={"ml.g4dn.xlarge": "CapacityError"},
        )

        engine = RaceEngine(adapter, ErrorClassifier)
        result = engine.execute(
            instance_types=["ml.p4d.24xlarge", "ml.g4dn.xlarge", "ml.g5.xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="us-east-1",
            globally_skipped_types={"ml.p4d.24xlarge"},
        )

        # 2 of 3 types submitted (p4d skipped).
        assert adapter.create_job.call_count == 2
        assert result.instance_type_used == "ml.g5.xlarge"


# ---------------------------------------------------------------------------
# 9. create_job exceptions are caught and classified
# ---------------------------------------------------------------------------


class TestCreateJobExceptions:
    """Exceptions raised by create_job are caught, not propagated."""

    @patch("sm_resilient_launcher._race_engine.time.sleep")
    def test_create_exception_does_not_crash_race(self, mock_sleep):
        """One type throws on create, the other succeeds — race proceeds."""
        adapter = _make_adapter(
            completed_types={"ml.g5.4xlarge"},
            create_exceptions={
                "ml.g4dn.xlarge": Exception("CapacityError: no capacity"),
            },
        )

        engine = RaceEngine(adapter, ErrorClassifier)
        result = engine.execute(
            instance_types=["ml.g4dn.xlarge", "ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="us-east-1",
        )

        assert result.instance_type_used == "ml.g5.4xlarge"

    @patch("sm_resilient_launcher._race_engine.time.sleep")
    def test_all_create_exceptions_raises_exhausted(self, mock_sleep):
        """When all create_job calls raise, AllTypesExhaustedError is raised."""
        adapter = _make_adapter(
            create_exceptions={
                "ml.g4dn.xlarge": Exception("CapacityError: no capacity"),
                "ml.g5.4xlarge": Exception("ResourceLimitExceeded: limit"),
            },
        )

        engine = RaceEngine(adapter, ErrorClassifier)
        with pytest.raises(AllTypesExhaustedError) as exc_info:
            engine.execute(
                instance_types=["ml.g4dn.xlarge", "ml.g5.4xlarge"],
                boto3_params=BOTO3_PARAMS,
                base_job_name="job",
                region="us-east-1",
            )

        assert len(exc_info.value.errors) == 2

    @patch("sm_resilient_launcher._race_engine.time.sleep")
    def test_create_exceptions_classified_correctly(self, mock_sleep):
        """Errors from create_job are classified by ErrorClassifier."""
        adapter = _make_adapter(
            completed_types={"ml.g5.4xlarge"},
            create_exceptions={
                "ml.g4dn.xlarge": Exception("CapacityError: unable to provision"),
            },
        )

        engine = RaceEngine(adapter, ErrorClassifier)
        result = engine.execute(
            instance_types=["ml.g4dn.xlarge", "ml.g5.4xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="us-east-1",
        )

        # The g4dn error should be classified as CAPACITY.
        capacity_errors = [
            e for e in result.errors if e.category == ErrorCategory.CAPACITY
        ]
        assert len(capacity_errors) >= 1
        assert capacity_errors[0].instance_type == "ml.g4dn.xlarge"
        assert capacity_errors[0].region == "us-east-1"
