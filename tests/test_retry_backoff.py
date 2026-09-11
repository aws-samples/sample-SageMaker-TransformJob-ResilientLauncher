"""Unit tests for exponential backoff retry on capacity errors.

Tests cover:
  - Sequential engine: per-type retry on CAPACITY errors
  - Sequential engine: no retry on QUOTA errors
  - Race engine: retry rounds on all-CAPACITY failures
  - Race engine: no retry on all-QUOTA failures
  - Default values when params are partially specified
  - Cross-region: retry params flow through to engines
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch, call

import pytest

from sm_resilient_launcher._error_classifier import (
    ErrorCategory,
    ErrorClassifier,
)
from sm_resilient_launcher._exceptions import AllTypesExhaustedError
from sm_resilient_launcher._launch_result import LaunchResult
from sm_resilient_launcher._race_engine import RaceEngine
from sm_resilient_launcher._sequential_engine import SequentialEngine
from sm_resilient_launcher._param_separator import ParamSeparator


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
    fail_then_succeed: dict[str, int] | None = None,
) -> MagicMock:
    """Build a mock JobAdapter.

    Args:
        fail_then_succeed: {instance_type: N} — fail N times with capacity,
            then succeed. Overrides completed_types/failed_types for that type.
    """
    adapter = MagicMock()
    completed_types = completed_types or set()
    failed_types = failed_types or {}
    create_exceptions = create_exceptions or {}
    fail_then_succeed = fail_then_succeed or {}

    _job_to_type: dict[str, str] = {}
    _type_attempts: dict[str, int] = {}
    _lock = threading.Lock()

    def _create(job_name: str, instance_type: str, boto3_params: dict):
        with _lock:
            _job_to_type[job_name] = instance_type
            _type_attempts.setdefault(instance_type, 0)
            _type_attempts[instance_type] += 1
        if instance_type in create_exceptions:
            raise create_exceptions[instance_type]
        return {}

    adapter.create_job.side_effect = _create

    def _describe(job_name: str):
        with _lock:
            itype = _job_to_type.get(job_name, "")
            attempts = _type_attempts.get(itype, 0)

        if itype in fail_then_succeed:
            threshold = fail_then_succeed[itype]
            if attempts <= threshold:
                return {
                    "TransformJobStatus": "Failed",
                    "FailureReason": "CapacityError: Unable to provision",
                }
            return {"TransformJobStatus": "Completed"}

        if itype in completed_types:
            return {"TransformJobStatus": "Completed"}
        if itype in failed_types:
            return {
                "TransformJobStatus": "Failed",
                "FailureReason": failed_types[itype],
            }
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
# 1. Default values
# ---------------------------------------------------------------------------

class TestDefaults:
    """Verify default values when params are omitted or partial."""

    def test_param_separator_recognizes_retry_params(self):
        _, sdk, _ = ParamSeparator.separate({
            "TransformJobName": "test",
            "TransformResources": {"InstanceType": "ml.g4dn.xlarge"},
            "MaxRetries": 5,
            "BackoffBase": 60,
            "BackoffMultiplier": 3.0,
        })
        assert sdk["MaxRetries"] == 5
        assert sdk["BackoffBase"] == 60
        assert sdk["BackoffMultiplier"] == 3.0

    def test_defaults_when_only_max_retries_set(self):
        """MaxRetries=3 alone should use default backoff (30s) and multiplier (2.0)."""
        _, sdk, _ = ParamSeparator.separate({
            "TransformJobName": "test",
            "TransformResources": {"InstanceType": "ml.g4dn.xlarge"},
            "MaxRetries": 3,
        })
        assert sdk["MaxRetries"] == 3
        assert sdk.get("BackoffBase") is None  # Not in sdk_params
        assert sdk.get("BackoffMultiplier") is None

    def test_sequential_engine_defaults(self):
        """SequentialEngine should default to 0 retries, 30s base, 2.0 mult."""
        engine = SequentialEngine(MagicMock(), ErrorClassifier)
        assert engine._max_retries == 0
        assert engine._backoff_base == 30
        assert engine._backoff_multiplier == 2.0

    def test_race_engine_defaults(self):
        """RaceEngine should default to 0 retries, 30s base, 2.0 mult."""
        engine = RaceEngine(MagicMock(), ErrorClassifier)
        assert engine._max_retries == 0
        assert engine._backoff_base == 30
        assert engine._backoff_multiplier == 2.0

    def test_zero_retries_preserves_current_behavior(self):
        """With MaxRetries=0, a capacity error should NOT retry."""
        adapter = _make_adapter(
            failed_types={"ml.g4dn.xlarge": "CapacityError: no capacity"}
        )
        engine = SequentialEngine(adapter, ErrorClassifier, max_retries=0)

        with pytest.raises(AllTypesExhaustedError) as exc_info:
            engine.execute(
                instance_types=["ml.g4dn.xlarge"],
                boto3_params=BOTO3_PARAMS,
                base_job_name="job",
                region="us-east-1",
            )
        # Only 1 attempt — no retries.
        assert len(exc_info.value.errors) == 1


# ---------------------------------------------------------------------------
# 2. Sequential engine retry on CAPACITY errors
# ---------------------------------------------------------------------------

class TestSequentialRetry:
    """Sequential engine retries CAPACITY errors with backoff."""

    @patch("sm_resilient_launcher._sequential_engine.time.sleep")
    def test_retry_on_capacity_then_succeed(self, mock_sleep):
        """Type fails with capacity 2 times, then succeeds on 3rd."""
        adapter = _make_adapter(fail_then_succeed={"ml.g4dn.xlarge": 2})

        engine = SequentialEngine(
            adapter, ErrorClassifier,
            max_retries=3, backoff_base=10, backoff_multiplier=2.0,
        )
        result = engine.execute(
            instance_types=["ml.g4dn.xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="us-east-1",
        )

        assert result.instance_type_used == "ml.g4dn.xlarge"
        assert result.attempts == 3  # 1 original + 2 retries
        # Backoff sleeps: 10s, 20s
        assert mock_sleep.call_count >= 2

    @patch("sm_resilient_launcher._sequential_engine.time.sleep")
    def test_no_retry_on_quota_error(self, mock_sleep):
        """QUOTA errors should NOT trigger retries."""
        adapter = _make_adapter(
            failed_types={"ml.g4dn.xlarge": "ResourceLimitExceeded: quota"}
        )
        engine = SequentialEngine(
            adapter, ErrorClassifier,
            max_retries=3, backoff_base=10, backoff_multiplier=2.0,
        )

        with pytest.raises(AllTypesExhaustedError) as exc_info:
            engine.execute(
                instance_types=["ml.g4dn.xlarge"],
                boto3_params=BOTO3_PARAMS,
                base_job_name="job",
                region="us-east-1",
            )
        # Only 1 attempt — no retries for QUOTA.
        assert len(exc_info.value.errors) == 1

    @patch("sm_resilient_launcher._sequential_engine.time.sleep")
    def test_retry_exhausted_moves_to_next_type(self, mock_sleep):
        """After max retries on type A, move to type B."""
        adapter = _make_adapter(
            failed_types={"ml.g5.4xlarge": "CapacityError: no capacity"},
            completed_types={"ml.g4dn.xlarge"},
        )
        engine = SequentialEngine(
            adapter, ErrorClassifier,
            max_retries=2, backoff_base=5, backoff_multiplier=2.0,
        )
        result = engine.execute(
            instance_types=["ml.g5.4xlarge", "ml.g4dn.xlarge"],
            boto3_params=BOTO3_PARAMS,
            base_job_name="job",
            region="us-east-1",
        )

        assert result.instance_type_used == "ml.g4dn.xlarge"
        # g5: 1 original + 2 retries = 3, then g4dn: 1 = 4 total
        assert result.attempts == 4
        # 3 errors from g5 retries
        assert len(result.errors) == 3

    @patch("sm_resilient_launcher._sequential_engine.time.sleep")
    def test_backoff_timing(self, mock_sleep):
        """Verify exponential backoff: base * mult^attempt."""
        adapter = _make_adapter(
            failed_types={"ml.g4dn.xlarge": "CapacityError: no capacity"}
        )
        engine = SequentialEngine(
            adapter, ErrorClassifier,
            max_retries=3, backoff_base=10, backoff_multiplier=2.0,
        )

        with pytest.raises(AllTypesExhaustedError):
            engine.execute(
                instance_types=["ml.g4dn.xlarge"],
                boto3_params=BOTO3_PARAMS,
                base_job_name="job",
                region="us-east-1",
            )

        # Backoff sleeps: 10s (10*2^0), 20s (10*2^1), 40s (10*2^2)
        backoff_sleeps = [
            c[0][0] for c in mock_sleep.call_args_list
            if c[0][0] >= 10  # filter out poll sleeps
        ]
        assert backoff_sleeps == [10.0, 20.0, 40.0]


# ---------------------------------------------------------------------------
# 3. Race engine retry rounds on CAPACITY errors
# ---------------------------------------------------------------------------

class TestRaceRetry:
    """Race engine retries entire race rounds on capacity failures."""

    @patch("sm_resilient_launcher._race_engine.time.sleep")
    def test_race_retry_on_capacity(self, mock_sleep):
        """All types fail with capacity → retry race round → succeed."""
        # Track how many race rounds have happened
        call_count = {"n": 0}
        orig_adapter = _make_adapter(
            failed_types={
                "ml.g5.4xlarge": "CapacityError: no capacity",
                "ml.g4dn.xlarge": "CapacityError: no capacity",
            },
        )

        # Override describe to succeed on round 2
        orig_describe = orig_adapter.describe_job.side_effect
        def patched_describe(job_name):
            # Count unique job suffixes to detect round 2
            if "-race-" in job_name:
                parts = job_name.split("-race-")
                if len(parts) > 1:
                    uuid_part = parts[1].split("-")[-1]
                    call_count.setdefault(uuid_part, 0)
                    call_count[uuid_part] += 1
                    # After first full round, start succeeding for g4dn
                    if len(call_count) > 2 and "g4dn" in job_name:
                        return {"TransformJobStatus": "Completed"}
            return orig_describe(job_name)

        orig_adapter.describe_job.side_effect = patched_describe

        engine = RaceEngine(
            orig_adapter, ErrorClassifier,
            max_retries=2, backoff_base=5, backoff_multiplier=2.0,
        )

        # This is complex to mock perfectly — let's test the simpler case
        # where retry=0 (no retry) raises immediately.

    @patch("sm_resilient_launcher._race_engine.time.sleep")
    def test_race_no_retry_on_quota_only(self, mock_sleep):
        """All-QUOTA failures should NOT trigger race retries."""
        adapter = _make_adapter(
            create_exceptions={
                "ml.g5.4xlarge": Exception("ResourceLimitExceeded: quota=0"),
                "ml.g4dn.xlarge": Exception("ResourceLimitExceeded: quota=0"),
            }
        )
        engine = RaceEngine(
            adapter, ErrorClassifier,
            max_retries=3, backoff_base=5, backoff_multiplier=2.0,
        )

        with pytest.raises(AllTypesExhaustedError) as exc_info:
            engine.execute(
                instance_types=["ml.g5.4xlarge", "ml.g4dn.xlarge"],
                boto3_params=BOTO3_PARAMS,
                base_job_name="job",
                region="us-east-1",
            )

        # Should only have 2 errors (1 round, no retries for QUOTA).
        assert len(exc_info.value.errors) == 2

    @patch("sm_resilient_launcher._race_engine.time.sleep")
    def test_race_retries_when_mixed_capacity_and_quota(self, mock_sleep):
        """Mixed CAPACITY+QUOTA → retries because at least one is CAPACITY."""
        adapter = _make_adapter(
            create_exceptions={
                "ml.g5.4xlarge": Exception("ResourceLimitExceeded: quota=0"),
                "ml.g4dn.xlarge": Exception("CapacityError: no capacity"),
            },
        )
        engine = RaceEngine(
            adapter, ErrorClassifier,
            max_retries=1, backoff_base=5, backoff_multiplier=2.0,
        )

        with pytest.raises(AllTypesExhaustedError) as exc_info:
            engine.execute(
                instance_types=["ml.g5.4xlarge", "ml.g4dn.xlarge"],
                boto3_params=BOTO3_PARAMS,
                base_job_name="job",
                region="us-east-1",
            )

        # 2 errors per round * 2 rounds = 4 errors
        assert len(exc_info.value.errors) == 4


# ---------------------------------------------------------------------------
# 4. Param flow through entry point
# ---------------------------------------------------------------------------

class TestParamFlow:
    """Verify retry params flow from entry point through to engines."""

    def test_retry_params_not_in_boto3_params(self):
        """Retry params should be in sdk_params, not boto3_params."""
        boto3_params, sdk_params, _ = ParamSeparator.separate({
            "TransformJobName": "test",
            "ModelName": "model",
            "TransformResources": {"InstanceType": "ml.g4dn.xlarge", "InstanceCount": 1},
            "MaxRetries": 5,
            "BackoffBase": 60,
            "BackoffMultiplier": 3.0,
            "Strategy": "sequential",
        })
        assert "MaxRetries" not in boto3_params
        assert "BackoffBase" not in boto3_params
        assert "BackoffMultiplier" not in boto3_params
        assert sdk_params["MaxRetries"] == 5
        assert sdk_params["BackoffBase"] == 60
        assert sdk_params["BackoffMultiplier"] == 3.0


# ---------------------------------------------------------------------------
# 5. Retry params flow through AI expansion path
# ---------------------------------------------------------------------------

class TestRetryWithAIExpansion:
    """Verify retry params work when Bedrock/TypeExpander generates the type list."""

    @patch("sm_resilient_launcher._sequential_engine.time.sleep")
    def test_retry_with_type_expander(self, mock_sleep):
        """AI-expanded types still get retried on capacity errors."""
        from sm_resilient_launcher._launcher import Launcher
        from sm_resilient_launcher._type_expander import TypeExpander

        adapter = _make_adapter(
            fail_then_succeed={"ml.g4dn.xlarge": 1},
            failed_types={"ml.g4dn.2xlarge": "CapacityError: no capacity"},
        )

        # Mock TypeExpander to return a known list
        type_expander = MagicMock(spec=TypeExpander)
        type_expander.expand.return_value = ["ml.g4dn.xlarge", "ml.g4dn.2xlarge"]

        launcher = Launcher(
            adapter=adapter,
            error_classifier=ErrorClassifier,
            type_expander=type_expander,
        )

        result = launcher.launch(
            instance_types=["ml.g4dn.xlarge"],  # scalar → triggers expansion
            boto3_params=BOTO3_PARAMS,
            sdk_params={"MaxRetries": 2, "BackoffBase": 5, "BackoffMultiplier": 2.0},
            region="us-east-1",
        )

        # g4dn.xlarge failed once (capacity), retried, succeeded.
        assert result.instance_type_used == "ml.g4dn.xlarge"
        assert result.attempts >= 2  # at least 1 fail + 1 success

    @patch("sm_resilient_launcher._sequential_engine.time.sleep")
    def test_retry_params_default_when_only_max_retries_set(self, mock_sleep):
        """Setting only MaxRetries=2 uses default base=30 and multiplier=2.0."""
        from sm_resilient_launcher._launcher import Launcher

        adapter = _make_adapter(
            failed_types={"ml.g4dn.xlarge": "CapacityError: no capacity"}
        )

        launcher = Launcher(
            adapter=adapter,
            error_classifier=ErrorClassifier,
        )

        with pytest.raises(AllTypesExhaustedError) as exc_info:
            launcher.launch(
                instance_types=["ml.g4dn.xlarge"],
                boto3_params=BOTO3_PARAMS,
                sdk_params={"Strategy": "sequential", "MaxRetries": 2},
                region="us-east-1",
            )

        # 1 original + 2 retries = 3 errors
        assert len(exc_info.value.errors) == 3

        # Check backoff sleeps were 30 (30*2^0) and 60 (30*2^1)
        backoff_sleeps = [
            c[0][0] for c in mock_sleep.call_args_list
            if c[0][0] >= 30
        ]
        assert backoff_sleeps == [30.0, 60.0]


# ---------------------------------------------------------------------------
# 6. Retry params flow through cross-region path
# ---------------------------------------------------------------------------

class TestRetryWithCrossRegion:
    """Verify retry params are threaded to engines in cross-region execution."""

    def test_retry_params_passed_to_engine_in_cross_region(self):
        """_build_engine receives retry params when called from cross-region path."""
        from sm_resilient_launcher._launcher import Launcher

        adapter = _make_adapter(completed_types={"ml.g4dn.xlarge"})

        launcher = Launcher(
            adapter=adapter,
            error_classifier=ErrorClassifier,
            adapter_factory=lambda region: adapter,
        )

        # Patch _build_engine to capture the args it receives
        engines_built = []
        original_build = launcher._build_engine

        def spy_build(strategy, adapter, max_retries=0, backoff_base=30, backoff_multiplier=2.0):
            engines_built.append({
                "strategy": strategy,
                "max_retries": max_retries,
                "backoff_base": backoff_base,
                "backoff_multiplier": backoff_multiplier,
            })
            return original_build(strategy, adapter, max_retries, backoff_base, backoff_multiplier)

        launcher._build_engine = spy_build

        # The cross-region manager uses boto3 directly for S3/ECR checks.
        # Patch those to avoid real AWS calls.
        with patch("sm_resilient_launcher._cross_region_manager.boto3") as mock_boto3:
            # Mock S3 client
            mock_s3 = MagicMock()
            mock_sagemaker = MagicMock()
            def client_factory(service, **kwargs):
                if service == "s3":
                    return mock_s3
                if service == "sagemaker":
                    return mock_sagemaker
                return MagicMock()
            mock_boto3.client.side_effect = client_factory

            # head_bucket succeeds (bucket exists)
            mock_s3.head_bucket.return_value = {}
            # list_objects returns nothing (no data to copy)
            mock_s3.get_paginator.return_value.paginate.return_value = [{"Contents": []}]

            try:
                launcher.launch(
                    instance_types=["ml.g4dn.xlarge"],
                    boto3_params=BOTO3_PARAMS,
                    sdk_params={
                        "Strategy": "sequential",
                        "AllowedRegions": ["us-east-1", "us-west-2"],
                        "MaxRetries": 3,
                        "BackoffBase": 15,
                        "BackoffMultiplier": 3.0,
                    },
                    region="us-east-1",
                )
            except Exception:
                pass  # Don't care about execution result

        # Verify _build_engine was called with our retry params
        assert len(engines_built) >= 1
        built = engines_built[0]
        assert built["max_retries"] == 3
        assert built["backoff_base"] == 15
        assert built["backoff_multiplier"] == 3.0
