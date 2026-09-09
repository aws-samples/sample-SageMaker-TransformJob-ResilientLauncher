"""Custom exceptions for the sm_resilient_launcher SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sm_resilient_launcher._error_classifier import ClassifiedError


class AllTypesExhaustedError(Exception):
    """Raised when all instance types have been tried and failed."""

    def __init__(self, errors: list[ClassifiedError]) -> None:
        self.errors = errors
        types = sorted({e.instance_type for e in errors})
        super().__init__(
            f"All instance types exhausted: {types}. "
            f"Errors: {[e.message for e in errors]}"
        )


class AllRegionsExhaustedError(Exception):
    """Raised when all regions and types have been tried and failed."""

    def __init__(self, errors: list[ClassifiedError]) -> None:
        self.errors = errors
        regions = sorted({e.region for e in errors})
        types = sorted({e.instance_type for e in errors})
        super().__init__(
            f"All regions exhausted: {regions}. "
            f"All types exhausted: {types}. "
            f"Total errors: {len(errors)}"
        )


class IntelligentRoutingError(Exception):
    """Raised when the Bedrock Converse API call or parsing fails."""

    pass
