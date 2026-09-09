"""RaceEngine — parallel job submission with winner detection.

Submits all candidate instance types concurrently using a thread pool.
The first job to reach a completed state wins; all non-winners receive
stop_job calls to limit billing. When multiple jobs complete, tie-breaks
prefer the home region and then cheapest instance type.

Interacts with jobs exclusively through the JobAdapter interface —
contains no Batch Transform-specific logic.
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

# Default poll interval in seconds — matches TransformJobAdapter.
_POLL_INTERVAL_SECONDS = 10


class RaceEngine:
    """Parallel job submission with winner detection and loser cancellation.

    Interacts with jobs exclusively through the JobAdapter interface.

    Requirement 7.1: Submit all types concurrently.
    Requirement 7.2: First completed job is the winner.
    Requirement 7.3: All non-winners receive stop_job calls.
    Requirement 7.4: Unique job names per submission.
    Requirement 7.5: Raise AllTypesExhaustedError when all fail.
    Requirement 7.6: Return LaunchResult with winner details.
    Requirement 7.7: Tie-break: prefer home region, then cheapest type.
    Requirement 7.8: Wait for ALL jobs to reach terminal before raising.
    """

    def __init__(
        self,
        adapter: JobAdapter,
        error_classifier: ErrorClassifier,
        home_region: str = "",
    ) -> None:
        self._adapter = adapter
        self._classifier = error_classifier
        self._home_region = home_region

    def execute(
        self,
        instance_types: list[str],
        boto3_params: dict,
        base_job_name: str,
        region: str,
        globally_skipped_types: set[str] | None = None,
    ) -> LaunchResult:
        """Submit all types concurrently; first to complete wins.

        Each submission uses a unique job name:
        ``{base_job_name}-race-{index}-{uuid8}``

        Non-winners receive ``stop_job`` calls regardless of billing
        state. Batch Transform billing starts at ``TransformStartTime``
        (per-second, 1-minute minimum); jobs that never reach
        ``TransformStartTime`` cost $0.

        Args:
            instance_types: Ordered list — never reordered by the engine.
            boto3_params: The boto3 params (SDK params already stripped).
            base_job_name: Base name; each submission appends a suffix.
            region: The region for this execution.
            globally_skipped_types: Mutable set shared across regions;
                types here are skipped (not submitted).

        Returns:
            LaunchResult with winner details. ``attempts`` equals the
            number of types actually submitted.

        Raises:
            AllTypesExhaustedError: When all concurrent jobs fail.
        """
        if globally_skipped_types is None:
            globally_skipped_types = set()

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

        # Shared state protected by a lock.
        lock = threading.Lock()
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
                )
                future_to_entry[future] = (job_name, itype)

            # Req 7.8: Wait for ALL submitted jobs to reach a terminal
            # state before deciding winner or raising an error.
            for future in concurrent.futures.as_completed(future_to_entry):
                job_name, itype = future_to_entry[future]
                try:
                    result = future.result()
                    if result is not None:
                        with lock:
                            completed_results.append(result)
                    else:
                        # Job failed but didn't raise — classify as unknown.
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

        # Req 7.5 / 7.8: If no jobs completed, raise with all errors.
        if not completed_results:
            raise AllTypesExhaustedError(errors)

        # Req 7.7: When multiple jobs completed, tie-break by:
        # 1. Prefer home region.
        # 2. If tied on region, prefer cheapest (alphabetical as heuristic).
        winner = self._pick_winner(completed_results, region)
        winner.attempts = len(job_entries)
        winner.strategy_used = "race"
        winner.errors = list(errors)

        # Req 7.3: Cancel ALL non-winners, regardless of billing state.
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
    ) -> LaunchResult | None:
        """Submit one job and poll until a terminal state is reached.

        Returns:
            A LaunchResult if the job completed, or None if it failed.
        """
        self._adapter.create_job(job_name, instance_type, boto3_params)

        while True:
            desc = self._adapter.describe_job(job_name)
            if self._adapter.is_completed(desc):
                return LaunchResult(
                    job_name=job_name,
                    instance_type_used=instance_type,
                    region_used=region,
                    attempts=1,  # Overwritten by execute().
                    strategy_used="race",
                )
            if self._adapter.is_failed(desc):
                return None
            time.sleep(_POLL_INTERVAL_SECONDS)

    def _pick_winner(
        self,
        candidates: list[LaunchResult],
        current_region: str,
    ) -> LaunchResult:
        """Select the best winner when multiple jobs completed.

        Tie-breaking order (Req 7.7):
        1. Prefer the job in the home region (if known).
        2. If tied on region, prefer cheapest instance type — uses
           alphabetical ordering of the instance type string as a
           simple cost heuristic (e.g. g4dn < g5 < p4d < p5).

        When ``home_region`` is empty or not relevant, falls back to
        the current execution region.
        """
        if len(candidates) == 1:
            return candidates[0]

        effective_home = self._home_region or current_region

        def sort_key(result: LaunchResult) -> tuple[int, str]:
            is_home = 0 if result.region_used == effective_home else 1
            return (is_home, result.instance_type_used)

        candidates.sort(key=sort_key)
        return candidates[0]
