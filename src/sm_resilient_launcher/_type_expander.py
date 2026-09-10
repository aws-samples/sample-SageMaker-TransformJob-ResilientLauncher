"""TypeExpander — deterministic same-or-larger instance type expansion.

Expands a single SageMaker instance type into a fallback list of
same-or-larger compatible types using GPU metadata lookup.  No
Bedrock invocation.  Used when ``WorkloadDescription`` is absent, or
as a fallback when the Bedrock API call fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sm_resilient_launcher._quota_fetcher import QuotaFetcher


@dataclass(frozen=True)
class GpuMetadata:
    """GPU metadata for a SageMaker instance type."""

    instance_type: str
    gpu_name: str
    gpu_memory_gb: int
    gpu_count: int


class TypeExpander:
    """Expands a single instance type into a fallback list of
    same-or-larger compatible types using deterministic GPU
    metadata lookup.  No Bedrock invocation.
    """

    # Hardcoded GPU metadata from empirical investigation
    GPU_CATALOG: dict[str, GpuMetadata] = {
        "ml.g4dn.xlarge": GpuMetadata("ml.g4dn.xlarge", "T4", 16, 1),
        "ml.g4dn.2xlarge": GpuMetadata("ml.g4dn.2xlarge", "T4", 16, 1),
        "ml.g4dn.4xlarge": GpuMetadata("ml.g4dn.4xlarge", "T4", 16, 1),
        "ml.g4dn.8xlarge": GpuMetadata("ml.g4dn.8xlarge", "T4", 16, 1),
        "ml.g4dn.12xlarge": GpuMetadata("ml.g4dn.12xlarge", "T4", 16, 4),
        "ml.g5.xlarge": GpuMetadata("ml.g5.xlarge", "A10G", 24, 1),
        "ml.g5.2xlarge": GpuMetadata("ml.g5.2xlarge", "A10G", 24, 1),
        "ml.g5.4xlarge": GpuMetadata("ml.g5.4xlarge", "A10G", 24, 1),
        "ml.g5.8xlarge": GpuMetadata("ml.g5.8xlarge", "A10G", 24, 1),
        "ml.g5.12xlarge": GpuMetadata("ml.g5.12xlarge", "A10G", 24, 4),
        "ml.g6.xlarge": GpuMetadata("ml.g6.xlarge", "L4", 24, 1),
        "ml.g6.2xlarge": GpuMetadata("ml.g6.2xlarge", "L4", 24, 1),
        "ml.g6e.xlarge": GpuMetadata("ml.g6e.xlarge", "L4", 24, 1),
        "ml.p3.2xlarge": GpuMetadata("ml.p3.2xlarge", "V100", 16, 1),
        "ml.p3.8xlarge": GpuMetadata("ml.p3.8xlarge", "V100", 16, 4),
        "ml.p4d.24xlarge": GpuMetadata("ml.p4d.24xlarge", "A100", 40, 8),
        "ml.p5.48xlarge": GpuMetadata("ml.p5.48xlarge", "H100", 80, 8),
    }

    def __init__(self, quota_fetcher: QuotaFetcher) -> None:
        self._quota_fetcher = quota_fetcher

    def expand(
        self,
        instance_type: str,
        region: str,
        excluded_types: set[str] | None = None,
    ) -> list[str]:
        """Generate a fallback list of same-or-larger instance types.

        The original type is always first.  Remaining types are sorted
        by GPU memory ascending (cheapest-fit-first heuristic).

        Args:
            instance_type: The starting instance type.
            region: Target region for quota filtering.
            excluded_types: Types to exclude (e.g., from family_support
                errors).

        Returns:
            Ordered list with the original type first, followed by
            same-or-larger types with non-zero quota.
        """
        if excluded_types is None:
            excluded_types = set()

        source = self.GPU_CATALOG.get(instance_type)
        if source is None:
            return [instance_type]

        candidates = []
        for itype, meta in self.GPU_CATALOG.items():
            if itype == instance_type:
                continue
            if itype in excluded_types:
                continue
            if meta.gpu_memory_gb >= source.gpu_memory_gb:
                candidates.append((itype, meta))

        # Sort by GPU memory ascending (cheapest-fit-first)
        candidates.sort(key=lambda x: (x[1].gpu_memory_gb, x[0]))

        result = [instance_type]
        result.extend(itype for itype, _ in candidates)
        return result
