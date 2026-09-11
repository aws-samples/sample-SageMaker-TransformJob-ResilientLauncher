"""Launcher — job-type-agnostic orchestrator applying the behavior matrix.

Strategies:
  - passthrough:  handled upstream in the entry point (not here)
  - sequential:   try types in order, one at a time
  - race:         submit all types in parallel, first to complete wins
  - intelligent:  AI-powered type expansion + recommended strategy

Contains NO Batch Transform-specific logic — works through the
``JobAdapter`` interface exclusively.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from sm_resilient_launcher._cross_region_manager import CrossRegionManager
from sm_resilient_launcher._error_classifier import ErrorClassifier
from sm_resilient_launcher._launch_result import LaunchResult
from sm_resilient_launcher._race_engine import RaceEngine
from sm_resilient_launcher._sequential_engine import SequentialEngine

if TYPE_CHECKING:
    from sm_resilient_launcher._intelligent_router import IntelligentRouter
    from sm_resilient_launcher._job_adapter import JobAdapter
    from sm_resilient_launcher._type_expander import TypeExpander

logger = logging.getLogger(__name__)

# Params that only control retry behavior — they don't trigger
# type expansion or AI routing.
_RETRY_ONLY_PARAMS = {"MaxRetries", "BackoffBase", "BackoffMultiplier"}


class Launcher:
    """Job-type-agnostic orchestrator for capacity-resilient launches.

    Strategies:
      passthrough   — handled upstream (entry point), not here
      sequential    — try types in caller's order, one at a time
      race          — all types in parallel, first wins, rest cancelled
      intelligent   — Bedrock AI (if WorkloadDescription) or TypeExpander
                      generates a prioritized fallback list, then executes
                      with the AI-recommended sub-strategy (sequential/race)
    """

    def __init__(
        self,
        adapter: JobAdapter,
        error_classifier: type[ErrorClassifier] | ErrorClassifier,
        intelligent_router: IntelligentRouter | None = None,
        type_expander: TypeExpander | None = None,
        adapter_factory: Callable[[str], JobAdapter] | None = None,
    ) -> None:
        self._adapter = adapter
        self._classifier = error_classifier
        self._intelligent_router = intelligent_router
        self._type_expander = type_expander
        self._adapter_factory = adapter_factory

    def launch(
        self,
        instance_types: list[str],
        boto3_params: dict,
        sdk_params: dict,
        region: str,
    ) -> LaunchResult:
        """Execute the requested strategy per the behavior matrix."""
        strategy = sdk_params.get("Strategy")
        allowed_regions = sdk_params.get("AllowedRegions")
        workload_desc = sdk_params.get("WorkloadDescription")
        region_config = sdk_params.get("RegionConfig")
        auto_create_model = sdk_params.get("AutoCreateModel", False)
        base_job_name = boto3_params.get(
            "TransformJobName", "sm-resilient"
        )

        # Retry / backoff params
        max_retries = sdk_params.get("MaxRetries", 0)
        backoff_base = sdk_params.get("BackoffBase", 30)
        backoff_multiplier = sdk_params.get("BackoffMultiplier", 2.0)

        agent_reasoning: str | None = None
        is_single = len(instance_types) == 1

        # Check if the only SDK params present are retry-related.
        non_retry_params = set(sdk_params.keys()) - _RETRY_ONLY_PARAMS
        has_only_retry_params = len(non_retry_params) == 0

        # ── Handle Strategy="intelligent" explicitly ──────────
        if strategy == "intelligent":
            instance_types, strategy, agent_reasoning = (
                self._run_intelligent_expansion(
                    instance_types[0] if is_single else instance_types[0],
                    workload_desc,
                    region,
                )
            )

        # ── Behavior matrix for single type with no explicit strategy ──
        elif is_single and strategy is None:
            has_multi_region = (
                allowed_regions is not None and len(allowed_regions) > 1
            )

            if has_multi_region:
                strategy = "sequential"
            elif has_only_retry_params:
                # Only retry params → sequential with this one type.
                strategy = "sequential"
            elif workload_desc and self._intelligent_router:
                # WorkloadDescription triggers intelligent expansion.
                instance_types, strategy, agent_reasoning = (
                    self._run_intelligent_expansion(
                        instance_types[0], workload_desc, region
                    )
                )
            elif self._type_expander:
                instance_types = self._type_expander.expand(
                    instance_types[0], region
                )
                strategy = "sequential"
            else:
                strategy = "sequential"

        # Default strategy for list mode.
        if strategy is None:
            strategy = "sequential"

        # ── Execute: cross-region or single-region ──
        if allowed_regions is not None and len(allowed_regions) > 1:
            result = self._execute_cross_region(
                instance_types,
                boto3_params,
                base_job_name,
                strategy,
                allowed_regions,
                region_config,
                auto_create_model,
                max_retries,
                backoff_base,
                backoff_multiplier,
            )
        else:
            result = self._execute_single_region(
                instance_types,
                boto3_params,
                base_job_name,
                strategy,
                region,
                max_retries,
                backoff_base,
                backoff_multiplier,
            )

        if agent_reasoning is not None:
            result.agent_reasoning = agent_reasoning

        return result

    # ── Private helpers ────────────────────────────────────────

    def _run_intelligent_expansion(
        self,
        original_type: str,
        workload_desc: str | None,
        region: str,
    ) -> tuple[list[str], str, str | None]:
        """Run AI or deterministic type expansion.

        Returns:
            (expanded_types, recommended_strategy, reasoning_or_none)
        """
        if workload_desc and self._intelligent_router:
            ordered_types, rec_strategy, reasoning = (
                self._intelligent_router.generate_plan(
                    original_type, workload_desc
                )
            )
            return ordered_types, rec_strategy, reasoning
        elif self._type_expander:
            expanded = self._type_expander.expand(original_type, region)
            return expanded, "sequential", None
        else:
            return [original_type], "sequential", None

    def _execute_single_region(
        self,
        instance_types: list[str],
        boto3_params: dict,
        base_job_name: str,
        strategy: str,
        region: str,
        max_retries: int = 0,
        backoff_base: int = 30,
        backoff_multiplier: float = 2.0,
    ) -> LaunchResult:
        """Delegate to the strategy engine for a single region."""
        engine = self._build_engine(
            strategy, self._adapter, max_retries, backoff_base, backoff_multiplier
        )
        return engine.execute(
            instance_types=instance_types,
            boto3_params=boto3_params,
            base_job_name=base_job_name,
            region=region,
        )

    def _execute_cross_region(
        self,
        instance_types: list[str],
        boto3_params: dict,
        base_job_name: str,
        strategy: str,
        allowed_regions: list[str],
        region_config: dict | None = None,
        auto_create_model: bool = False,
        max_retries: int = 0,
        backoff_base: int = 30,
        backoff_multiplier: float = 2.0,
    ) -> LaunchResult:
        """Delegate to CrossRegionManager for multi-region fallback."""
        home_region = allowed_regions[0]
        manager = CrossRegionManager(
            adapter_factory=self._adapter_factory,
            strategy_engine_factory=lambda adapter: self._build_engine(
                strategy, adapter, max_retries, backoff_base, backoff_multiplier
            ),
            home_region=home_region,
            allowed_regions=allowed_regions,
            base_job_name=base_job_name,
            region_config=region_config,
            auto_create_model=auto_create_model,
        )
        return manager.execute(
            instance_types, boto3_params, base_job_name
        )

    def _build_engine(
        self,
        strategy: str,
        adapter: JobAdapter,
        max_retries: int = 0,
        backoff_base: int = 30,
        backoff_multiplier: float = 2.0,
    ) -> SequentialEngine | RaceEngine:
        """Instantiate the strategy engine with retry params."""
        if strategy == "sequential":
            return SequentialEngine(
                adapter,
                self._classifier,
                max_retries=max_retries,
                backoff_base=backoff_base,
                backoff_multiplier=backoff_multiplier,
            )
        elif strategy == "race":
            return RaceEngine(
                adapter,
                self._classifier,
                max_retries=max_retries,
                backoff_base=backoff_base,
                backoff_multiplier=backoff_multiplier,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
