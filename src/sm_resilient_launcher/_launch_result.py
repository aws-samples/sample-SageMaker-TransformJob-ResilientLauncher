"""LaunchResult dataclass — structured result from a successful launch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sm_resilient_launcher._error_classifier import ClassifiedError


@dataclass
class LaunchResult:
    """Structured result from a successful Batch Transform launch.

    Returned by the Launcher after a job succeeds. Contains the winning
    job details, attempt metrics, optional AI reasoning, cross-region
    replication status, accumulated errors from failed attempts, and
    pre-flight warnings (ECR commands, IAM advisories).
    """

    job_name: str
    instance_type_used: str
    region_used: str
    attempts: int
    strategy_used: str
    agent_reasoning: str | None = None
    data_replicated: bool = False
    errors: list[ClassifiedError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
