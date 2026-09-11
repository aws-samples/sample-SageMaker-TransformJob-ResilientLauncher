"""RaceEngine — parallel job submission with winner detection and retry.

Submits all candidate instance types concurrently using a thread pool.
The first job to reach a completed state wins; all non-winners receive
stop_job calls to limit billing. If ALL types fail with capacity errors
and retries are configured, the engine waits with exponential backoff
and races again.

Interacts with jobs exclusively through the JobAdapter interface.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
import uuid
from typing import TYPE_CHECKING

from sm_resilient_launcher._error_classifier import (
    ErrorCategory,
    ErrorClassifier,
)
from sm_resilient_launcher._exceptions import AllTypesExhaustedError
from sm_resilient_launcher._launch_result import LaunchResult

if TYPE_CHECKING:
    from sm_resilient_launcher._error_classifier import ClassifiedError
    from sm_resilient_launcher._job_adapter import JobAdapter

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 10

# Default retry/backoff settings.
DEFAULT_MAX_RETRIES: int = 0
DEFAULT_BACKOFF_BASE: int = 30
DEFAULT_BACKOFF_MULTIPLIER: float = 2.0


class RaceEngine:
    """Parallel job submission with winner detection, cancellation, and retry.

    Supports exponential-backoff retry rounds. If all types fail in a
    race round AND at least one failure was a CAPACITY error (hardware
    may free up), the engine waits and races again. QUOTA-only failures
    do not trigger retries (waiting won't change your quota).

    Requirement 7.1: Submit all types concurrently.
    Requirement 7.2: First completed job is the winner.
    Requirement 7.3: All non-winners receive stop_job calls.
    Requirement 7.4: Unique job names per submission.
    Requirement 7.5: Raise AllTypesExhaustedError when all fail.
    Requirement 7.6: Return LaunchResult with winner details.
    Requirement 7.7: Tie-break: first to complete wins.
    """

    def __init__(
        self,
        adapter: JobAdapter,
        error_classifier: ErrorClassifier,
        home_region: str = "",
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: int = DEFAULT_BACKOFF_BASE,
        backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
    ) -> None:
        self._adapter = adapter
        self._classifier = error_classifier
        self._home_region = home_region
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
        """Submit all types concurrently; first to complete wins.

        If all types fail and retries are configured, waits with
        exponential backoff and races again (only if at least one
        failure was CAPACITY — QUOTA-only rounds do not retry).

        Args:
            instance_types: Ordered list — never reordered by the engine.
            boto3_params: The boto3 params (SDK params already stripped).
            base_job_name: Base name; each submission appends a suffix.
            region: The region for this execution.
            globally_skipped_types: Mutable set shared across regions.

        Returns:
            LaunchResult with winner details.

        Raises:
            AllTypesExhaustedError: When all concurrent jobs fail.
        """
        if globally_skipped_types is None:
            globally_skipped_types = set()

        all_errors: list[ClassifiedError] = []

        for retry_round in range(self._max_retries + 1):
            try:
                result = self._race_once(
                    instance_types,
                    boto3_params,
                    base_job_name,
                    region,
                    globally_skipped_types,
                )
                # Include errors from previous rounds.
                result.errors = all_errors + (result.errors or [])
                return result

            except AllTypesExhaustedError as e:
                all_errors.extend(e.errors)

                # Only retry if at least one error was CAPACITY.
                has_capacity_error = any(
                    err.category == ErrorCategory.CAPACITY
                    for err in e.errors
                )

                if (
                    has_capacity_error
                    and retry_round < self._max_retries
                ):
                    wait = self._backoff_base * (
                        self._backoff_multiplier ** retry_round
                    )
                    logger.info(
                        "All types failed in race round %d/%d "
                        "(capacity errors present); retrying in %.0fs...",
                        retry_round + 1,
                        self._max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    continue

                # No capacity errors or retries exhausted.
                raise AllTypesExhaustedError(all_errors)

        # Should not reach here, but safety net.
        raise AllTypesExhaustedError(all_errors)

    # ── Single race round ──────────────────────────────────────

    def _race_once(
        self,
        instance_types: list[str],
        boto3_params: dict,
        base_job_name: str,
        region: str,
        globally_skipped_types: set[str],
    ) -> LaunchResult:
        """Run one race round — submit all types concurrently."""
        suffix = uuid.uuid4().hex[:8]
        job_entries: list[tuple[str, str]] = []
        for idx, itype in enumerate(instance_types):
            if itype in globally_skipped_types:
                logger.info(
                    "Skipping %s in race (globally excluded)", itype
                )
                continue
            job_name = f"{base_job_name}-race-{idx}-{suffix}"
            job_entries.append((job_name, itype))

        if not job_entries:
            raise AllTypesExhaustedError([])

        lock = threading.Lock()
        winner_event = threading.Event()
        completed_results: list[LaunchResult] = []
        errors: list[ClassifiedError] = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(job_entries),
        ) as pool:
            future_to_entry: dict[
                concurrent.futures.Future, tuple[str, str]
            ] = {}
            for job_name, itype in job_entries:
                future = pool.submit(
                    self._submit_and_poll,
                    job_name,
                    itype,
                    boto3_params,
                    region,
                    winner_event,
                )
                future_to_entry[future] = (job_name, itype)

            for future in concurrent.futures.as_completed(future_to_entry):
                job_name, itype = future_to_entry[future]
                try:
                    result = future.result()
                    if result is not None:
                        with lock:
                            completed_results.append(result)
                            winner_event.set()
                    else:
                        classified = self._classifier.classify(
                            Exception(f"Job {job_name} failed"),
                            itype,
                            region,
                        )
                        with lock:
                            errors.append(classified)
                except Exception as exc:
                    classified = self._classifier.classify(
                        exc, itype, region
                    )
                    with lock:
                        errors.append(classified)

        if not completed_results:
            raise AllTypesExhaustedError(errors)

        # First to complete wins (order from as_completed).
        winner = completed_results[0]
        winner.attempts = len(job_entries)
        winner.strategy_used = "race"
        winner.errors = list(errors)

        # Cancel non-winners.
        for job_name, _itype in job_entries:
            if job_name != winner.job_name:
                try:
                    self._adapter.stop_job(job_name)
                    logger.info(
                        "Cancelled non-winner %s (%s)",
                        job_name,
                        _itype,
                    )
                except Exception:
                    logger.debug(
                        "Stop failed for %s (may already be terminal)",
                        job_name,
                    )

        return winner

    # ── Internal helpers ───────────────────────────────────────

    def _submit_and_poll(
        self,
        job_name: str,
        instance_type: str,
        boto3_params: dict,
        region: str,
        winner_event: threading.Event,
    ) -> LaunchResult | None:
        """Submit one job and poll until terminal or another thread wins."""
        logger.info("Submitting %s...", instance_type)
        try:
            self._adapter.create_job(job_name, instance_type, boto3_params)
        except Exception as exc:
            logger.info("%s failed to submit: %s", instance_type, str(exc)[:60])
            raise
        logger.info("%s submitted, polling...", instance_type)

        while True:
            if winner_event.is_set():
                logger.info(
                    "%s: another type won, stopping poll", instance_type
                )
                return None

            desc = self._adapter.describe_job(job_name)
            if self._adapter.is_completed(desc):
                logger.info("Winner: %s completed", instance_type)
                return LaunchResult(
                    job_name=job_name,
                    instance_type_used=instance_type,
                    region_used=region,
                    attempts=1,
                    strategy_used="race",
                )
            if self._adapter.is_failed(desc):
                error_msg = self._adapter.extract_error(desc)
                logger.info("%s failed: %s", instance_type, error_msg[:50])
                return None

            status = desc.get("TransformJobStatus", "unknown")
            logger.info("%s: %s", instance_type, status)
            time.sleep(_POLL_INTERVAL_SECONDS)

    def _pick_winner(
        self,
        candidates: list[LaunchResult],
        current_region: str,
    ) -> LaunchResult:
        """Select the winner — first to complete wins."""
        return candidates[0]
