"""IntelligentRouter — Bedrock-powered type expansion with deterministic facts.

Interfaces with Amazon Bedrock Converse API for AI-powered instance-type
expansion.  Quota and pricing data are fetched deterministically by
``QuotaFetcher`` and ``PricingFetcher`` and injected as facts in the
prompt — Bedrock only reasons about GPU memory fit, priority ordering,
and strategy recommendation.  Falls back to ``TypeExpander`` if Bedrock
fails.
"""

from __future__ import annotations

import json
import logging

import boto3

from sm_resilient_launcher._exceptions import IntelligentRoutingError
from sm_resilient_launcher._pricing_fetcher import PricingFetcher
from sm_resilient_launcher._quota_fetcher import QuotaFetcher
from sm_resilient_launcher._type_expander import TypeExpander

logger = logging.getLogger(__name__)


class IntelligentRouter:
    """Generates an instance-type priority list using Bedrock Converse.

    Bedrock's role is limited to reasoning about GPU memory fit,
    optimal priority order, and strategy recommendation.  Quotas and
    pricing are fetched deterministically and injected as facts.

    Falls back to ``TypeExpander`` if Bedrock fails.
    """

    MODEL_ID: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    def __init__(
        self,
        quota_fetcher: QuotaFetcher,
        pricing_fetcher: PricingFetcher,
        type_expander: TypeExpander,
        region: str,
    ) -> None:
        self._quota_fetcher = quota_fetcher
        self._pricing_fetcher = pricing_fetcher
        self._type_expander = type_expander
        self._region = region
        self._bedrock = boto3.client(
            "bedrock-runtime", region_name=region
        )

    def generate_plan(
        self,
        original_instance_type: str,
        workload_description: str,
        excluded_types: set[str] | None = None,
    ) -> tuple[list[str], str, str | None]:
        """Generate an instance-type priority list and strategy.

        1. Get candidates from ``TypeExpander.expand()``.
        2. Fetch deterministic facts via ``QuotaFetcher`` and
           ``PricingFetcher``.
        3. Build prompt and call Bedrock Converse API.
        4. Parse response to extract priority list, strategy, and
           reasoning.
        5. Ensure original type is always first in the returned list.

        On Bedrock failure: fall back to ``TypeExpander.expand()``,
        return ``"sequential"`` for strategy and ``None`` for reasoning.

        Args:
            original_instance_type: The caller's starting instance type.
            workload_description: Natural-language workload description.
            excluded_types: Types excluded by prior ``family_support``
                errors.

        Returns:
            ``(ordered_types, strategy, reasoning)`` where strategy is
            ``"sequential"`` or ``"race"``, and reasoning is the
            Bedrock reasoning chain or ``None`` if the deterministic
            fallback was used.
        """
        if excluded_types is None:
            excluded_types = set()

        # Step 1: Discover ALL GPU types dynamically from the quota API
        all_types = self._quota_fetcher.get_all_transform_types()
        gpu_prefixes = ("ml.g", "ml.p", "ml.trn", "ml.inf")
        gpu_types = [t for t in all_types if any(t.startswith(p) for p in gpu_prefixes)]

        # Exclude types in the excluded set
        if excluded_types:
            gpu_types = [t for t in gpu_types if t not in excluded_types]

        # Ensure original type is in the list (might not be if not a GPU type)
        if original_instance_type not in gpu_types:
            gpu_types.insert(0, original_instance_type)

        candidates = gpu_types

        # Step 2: Fetch deterministic facts for all candidates
        quotas = self._quota_fetcher.fetch_quotas(candidates)

        # For the Bedrock prompt, exclude types with quota=0 (definitely unavailable).
        # Types with quota=None (lookup failed) are included — might still work.
        # This is intelligent pre-selection for AI — different from runtime where
        # the customer's list is always fully respected.
        prompt_candidates = [
            t for t in candidates
            if quotas.get(t) is None or quotas.get(t, 0) > 0
        ]
        # Always keep the original type in the prompt
        if original_instance_type not in prompt_candidates:
            prompt_candidates.insert(0, original_instance_type)
        prompt_quotas = {t: quotas.get(t) for t in prompt_candidates}

        # Step 3: Build prompt and call Bedrock
        prompt = self._build_prompt(
            original_instance_type,
            prompt_candidates,
            workload_description,
            prompt_quotas,
        )

        try:
            response = self._bedrock.converse(
                modelId=self.MODEL_ID,
                messages=[
                    {"role": "user", "content": [{"text": prompt}]}
                ],
            )
            ordered_types, strategy, reasoning = self._parse_response(
                response
            )

            # Step 5: Ensure original type is always first
            if (
                ordered_types
                and ordered_types[0] != original_instance_type
            ):
                if original_instance_type in ordered_types:
                    ordered_types.remove(original_instance_type)
                ordered_types.insert(0, original_instance_type)

            return ordered_types, strategy, reasoning

        except Exception as exc:
            logger.warning(
                "Bedrock call failed, falling back to TypeExpander: %s",
                exc,
            )
            # Fallback to deterministic expansion
            fallback_list = self._type_expander.expand(
                original_instance_type, self._region, excluded_types
            )
            return fallback_list, "sequential", None

    def _build_prompt(
        self,
        original_type: str,
        candidates: list[str],
        workload_description: str,
        quotas: dict[str, int | None],
    ) -> str:
        """Construct the Bedrock prompt with deterministic facts.

        The prompt includes the workload description, original type,
        GPU memory reference data, and quota values — all fetched
        deterministically.  Bedrock is never used to look up quotas.
        Pricing is omitted; Bedrock knows approximate pricing from
        its training data.
        """
        gpu_catalog = TypeExpander.GPU_CATALOG
        facts_lines = []
        for itype in candidates:
            quota_val = quotas.get(itype, "unknown")
            meta = gpu_catalog.get(itype)
            gpu_str = (
                f"{meta.gpu_name} {meta.gpu_memory_gb}GB x{meta.gpu_count}"
                if meta
                else ""  # Bedrock knows the GPU specs from its training data
            )
            facts_lines.append(
                f"  - {itype}: {gpu_str}, quota={quota_val}"
            )
        facts_block = "\n".join(facts_lines)

        return (
            f"You are an expert at selecting SageMaker instance types for "
            f"Batch Transform jobs.\n\n"
            f"Workload description: {workload_description}\n\n"
            f"Original requested type: {original_type}\n\n"
            f"You know the GPU specifications and approximate pricing "
            f"for each instance type from your training data.\n\n"
            f"Instance types available for Batch Transform "
            f"(quota > 0 from Service Quotas API):\n"
            f"{facts_block}\n\n"
            f"Instructions:\n"
            f"1. Reason about GPU memory fit for the workload.\n"
            f"2. Place the original type first.\n"
            f"3. Order remaining types by fit and cost-effectiveness.\n"
            f"4. Quota values are advisory — the SDK handles quota "
            f"failures at runtime. Prefer types with higher quota "
            f"but include all types.\n"
            f"5. Include ALL candidate types in the priority list — "
            f"the SDK handles failures at runtime.\n"
            f"6. Recommend 'sequential' or 'race' strategy.\n\n"
            f"Respond in JSON:\n"
            f'{{"priority_list": ["type1", "type2"], '
            f'"strategy": "sequential"|"race", '
            f'"reasoning": "..."}}'
        )

    def _parse_response(
        self, response: dict
    ) -> tuple[list[str], str, str]:
        """Extract priority list, strategy, and reasoning from Bedrock.

        Parses the JSON object from the Bedrock Converse response text.
        Defaults an invalid strategy to ``"sequential"``.

        Args:
            response: The raw Bedrock Converse API response dict.

        Returns:
            ``(priority_list, strategy, reasoning)``
        """
        text = response["output"]["message"]["content"][0]["text"]
        start = text.index("{")
        end = text.rindex("}") + 1
        parsed = json.loads(text[start:end])

        priority_list = parsed["priority_list"]
        strategy = parsed["strategy"]
        reasoning = parsed.get("reasoning", text)

        if strategy not in ("sequential", "race"):
            strategy = "sequential"

        return priority_list, strategy, reasoning
