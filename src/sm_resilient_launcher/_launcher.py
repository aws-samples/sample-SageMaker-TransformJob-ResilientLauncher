"""Launcher — job-type-agnostic orchestrator applying the behavior matrix.

Applies the strategy selection logic from the caller's input shape
to determine the execution path. Threads retry/backoff params to engines.

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


class Launcher:
    """Job-type-agnostic orchestrator for capacity-resilient launches.

    Applies the behavior matrix:

    - Scalar + no Strategy + no/single region -> AI Expansion -> sequential
    - Scalar + no Strategy + multi-region -> cross-region with single type
    - Scalar + explicit Strategy -> execute strategy with single type
    - List + no Strategy -> sequential with caller's order
    - List + explicit Strategy -> execute strategy with caller's order

    Passthrough is handled upstream in the entry point, not here.
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
        is_scalar = len(instance_types) == 1

        # ── Behavior matrix for scalar input with no explicit strategy ──
        if is_scalar and strategy is None:
            has_multi_region = (
                allowed_regions is not None and len(allowed_regions) > 1
            )

            if has_multi_region:
                strategy = "sequential"
            elif workload_desc and self._intelligent_router:
                ordered_types, rec_strategy, reasoning = (
                    self._intelligent_router.generate_plan(
                        instance_types[0], workload_desc
                    )
                )
                instance_types = ordered_types
                strategy = rec_strategy
                agent_reasoning = reasoning
            elif self._type_expander:
                instance_types = self._type_expander.expand(
                    instance_types[0], region
                )
                strategy = "sequential"
            else:
                strategy = "sequential"

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
