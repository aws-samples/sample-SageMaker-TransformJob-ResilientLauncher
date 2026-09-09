"""SageMaker Capacity-Resilient Job Launcher.

Provides automatic capacity fallback across instance types and AWS
regions for SageMaker Batch Transform jobs.

Public API:
    launch_create_transform_job  — main launcher entry point
    prepare_regions              — one-time cross-region setup utility
    LaunchResult                 — structured result from a successful launch
    AllTypesExhaustedError       — all instance types exhausted
    AllRegionsExhaustedError     — all regions and types exhausted
    ReadinessReport              — result from prepare_regions
    RegionReadiness              — per-region readiness status
"""

from sm_resilient_launcher._entry_point import launch_create_transform_job
from sm_resilient_launcher._exceptions import (
    AllRegionsExhaustedError,
    AllTypesExhaustedError,
)
from sm_resilient_launcher._launch_result import LaunchResult
from sm_resilient_launcher._prepare_regions import (
    ReadinessReport,
    RegionReadiness,
    prepare_regions,
)

__all__ = [
    "launch_create_transform_job",
    "prepare_regions",
    "LaunchResult",
    "AllTypesExhaustedError",
    "AllRegionsExhaustedError",
    "ReadinessReport",
    "RegionReadiness",
]
