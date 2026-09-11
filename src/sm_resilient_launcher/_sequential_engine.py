"""SequentialEngine — ordered instance-type fallback with retry/backoff.

Tries instance types from the priority list in order.  Error classification
drives behavior:
  - CAPACITY errors trigger exponential-backoff retries (hardware may free up)
  - QUOTA errors skip immediately (needs quota increase, not time)
  - FAMILY_SUPPORT errors skip the type globally across all regions
  - UNKNOWN errors skip the type
"""

from __future__ import annotations

import logging
import time

from sm_resilient_launcher._error_classifier import (
    ClassifiedError,
    ErrorCategory,
    ErrorClassifier,
)
from sm_resilient_launcher._exceptions import AllTypesExhaustedError
from sm_resilient_launcher._job_adapter import JobAdapter
from sm_resilient_launcher._launch_result import LaunchResult

logger = logging.getLogger(__name__)

# Default polling interval in seconds between describe_job calls.
POLL_INTERVAL_SECONDS: int = 10

# Default retry/backoff settings (matching SageMaker Pipelines naming).
DEFAULT_MAX_RETRIES: int = 0       # No retries by default (current behavior)
DEFAULT_BACKOFF_BASE: int = 30     # First retry wait: 30 seconds
DEFAULT_BACKOFF_MULTIPLIER: float = 2.0  # Double each retry


class SequentialEngine:
    """Ordered instance-type fallback for sequential strategy execution.

    Supports exponential-backoff retries for CAPACITY errors. When a type
    fails with a capacity error, the engine waits and retries the same type
    before moving to the next. Only CAPACITY errors are retried — QUOTA and
    FAMILY_SUPPORT errors skip immediately.

    Interacts with jobs exclusively through the JobAdapter interface.
    """

    def __init__(
        self,
        adapter: JobAdapter,
        error_classifier: type[ErrorClassifier] | ErrorClassifier | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: int = DEFAULT_BACKOFF_BASE,
        backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
    ) -> None:
        self._adapter = adapter
        self._classifier = error_classifier or ErrorClassifier
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_multiplier = backoff_multiplier

    def execute(
        self,
        instance_types: list[str],
        boto3_params: dict,
        base_job_name: str,
        region: str,
        globally_skipped_types: set[str] | None = None,
    ) -> LaunchResult:
        """Try each instance type in order until one succeeds.

        For each type, if a CAPACITY error occurs and retries are
        configured, the engine waits with exponential backoff and
        retries the same type up to ``max_retries`` times before
        moving to the next type.

        Args:
            instance_types: Ordered list — never reordered by the engine.
            boto3_params: The boto3 params (SDK params already stripped).
            base_job_name: Base name; each attempt appends a suffix.
            region: The region for this execution.
            globally_skipped_types: Mutable set shared across regions.

        Returns:
            LaunchResult on success.

        Raises:
            AllTypesExhaustedError: When all types fail in this region.
        """
        if globally_skipped_types is None:
            globally_skipped_types = set()

        errors: list[ClassifiedError] = []
        attempt_count = 0

        for idx, instance_type in enumerate(instance_types):
            if instance_type in globally_skipped_types:
                logger.info(
                    "Skipping %s (globally excluded due to family_support)",
                    instance_type,
                )
                continue

            # Try this type with retries for CAPACITY errors.
            retry_attempt = 0
            while True:
                attempt_count += 1
                suffix = f"-{idx}-{instance_type.replace('.', '-')}"
                if retry_attempt > 0:
                    suffix += f"-r{retry_attempt}"
                job_name = f"{base_job_name}{suffix}"

                classified = self._try_once(
                    job_name, instance_type, boto3_params, region
                )

                if classified is None:
                    # Success!
                    return LaunchResult(
                        job_name=job_name,
                        instance_type_used=instance_type,
                        region_used=region,
                        attempts=attempt_count,
                        strategy_used="sequential",
                        errors=errors,
                    )

                errors.append(classified)

                # Only retry CAPACITY errors — everything else skips.
                if (
                    classified.category == ErrorCategory.CAPACITY
                    and retry_attempt < self._max_retries
                ):
                    wait = self._backoff_base * (
                        self._backoff_multiplier ** retry_attempt
                    )
                    logger.info(
                        "%s capacity error (attempt %d/%d); "
                        "retrying in %.0fs...",
                        instance_type,
                        retry_attempt + 1,
                        self._max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    retry_attempt += 1
                    continue

                # Not retryable or retries exhausted — move to next type.
                self._handle_error(
                    classified, instance_type, globally_skipped_types
                )
                break

        raise AllTypesExhaustedError(errors)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _try_once(
        self,
        job_name: str,
        instance_type: str,
        boto3_params: dict,
        region: str,
    ) -> ClassifiedError | None:
        """Submit and poll one job. Return None on success, ClassifiedError on failure."""
        try:
            self._adapter.create_job(job_name, instance_type, boto3_params)

            result = self._poll_until_terminal(job_name)

            if self._adapter.is_completed(result):
                return None

            error_msg = self._adapter.extract_error(result)
            return ErrorClassifier.classify(
                Exception(error_msg), instance_type, region
            )

        except Exception as exc:
            return ErrorClassifier.classify(exc, instance_type, region)

    def _poll_until_terminal(self, job_name: str) -> dict:
        """Poll describe_job until a terminal state is reached."""
        while True:
            desc = self._adapter.describe_job(job_name)
            if self._adapter.is_completed(desc) or self._adapter.is_failed(
                desc
            ):
                return desc
            time.sleep(POLL_INTERVAL_SECONDS)

    @staticmethod
    def _handle_error(
        classified: ClassifiedError,
        instance_type: str,
        globally_skipped_types: set[str],
    ) -> None:
        """Apply skip behaviour based on the error classification."""
        if classified.category == ErrorCategory.FAMILY_SUPPORT:
            globally_skipped_types.add(instance_type)
            logger.warning(
                "%s not supported for Batch Transform; "
                "skipping in all regions",
                instance_type,
            )
        else:
            logger.info(
                "%s failed with %s in %s; moving to next type",
                instance_type,
                classified.category.value,
                classified.region,
            )
