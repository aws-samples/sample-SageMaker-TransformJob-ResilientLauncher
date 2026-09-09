"""SequentialEngine — ordered instance-type fallback strategy.

Tries instance types from the priority list in order. Error classification
drives skip behavior: capacity/quota/unknown skip the type in the current
region; family_support skips the type globally across all regions.
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


class SequentialEngine:
    """Ordered instance-type fallback for sequential strategy execution.

    Interacts with jobs exclusively through the JobAdapter interface.
    Contains no Batch Transform-specific logic — any concrete adapter
    can be plugged in.
    """

    def __init__(
        self,
        adapter: JobAdapter,
        error_classifier: type[ErrorClassifier] | ErrorClassifier | None = None,
    ) -> None:
        self._adapter = adapter
        # Accept the ErrorClassifier class itself (static method) or
        # an instance — the classify method is static so either works.
        self._classifier = error_classifier or ErrorClassifier

    def execute(
        self,
        instance_types: list[str],
        boto3_params: dict,
        base_job_name: str,
        region: str,
        globally_skipped_types: set[str] | None = None,
    ) -> LaunchResult:
        """Try each instance type in order until one succeeds.

        Skip behaviour by error classification:
          - capacity:       skip type in this region, try next type
          - quota:          skip type in this region, try next type
          - family_support: add to *globally_skipped_types* (all regions)
          - unknown:        skip type, try next type

        Args:
            instance_types: Ordered list — never reordered by the engine.
            boto3_params: The boto3 params (SDK params already stripped).
            base_job_name: Base name; each attempt appends a suffix.
            region: The region for this execution.
            globally_skipped_types: Mutable set shared across regions;
                types added here are skipped in all subsequent regions.

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

            attempt_count += 1
            job_name = (
                f"{base_job_name}-{idx}-"
                f"{instance_type.replace('.', '-')}"
            )

            try:
                self._adapter.create_job(
                    job_name, instance_type, boto3_params
                )

                result = self._poll_until_terminal(job_name)

                if self._adapter.is_completed(result):
                    return LaunchResult(
                        job_name=job_name,
                        instance_type_used=instance_type,
                        region_used=region,
                        attempts=attempt_count,
                        strategy_used="sequential",
                        errors=errors,
                    )

                # Job reached a failed terminal state — classify error.
                error_msg = self._adapter.extract_error(result)
                classified = ErrorClassifier.classify(
                    Exception(error_msg), instance_type, region
                )
                errors.append(classified)
                self._handle_error(
                    classified, instance_type, globally_skipped_types
                )

            except Exception as exc:
                classified = ErrorClassifier.classify(
                    exc, instance_type, region
                )
                errors.append(classified)
                self._handle_error(
                    classified, instance_type, globally_skipped_types
                )

        raise AllTypesExhaustedError(errors)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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
