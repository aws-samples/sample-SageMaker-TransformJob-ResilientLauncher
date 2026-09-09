"""ErrorClassifier, ErrorCategory enum, and ClassifiedError dataclass.

Categorizes SageMaker API exceptions into exactly one of four classes
based on priority-ordered, case-insensitive marker matching.
"""

from dataclasses import dataclass
from enum import Enum


class ErrorCategory(Enum):
    """Classification categories for SageMaker API errors."""

    CAPACITY = "capacity"
    QUOTA = "quota"
    FAMILY_SUPPORT = "family_support"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClassifiedError:
    """Immutable record of a classified SageMaker error.

    Attributes:
        category: One of the four error classifications.
        original_exception: The original exception object.
        message: The string representation of the exception.
        instance_type: The instance type that caused the error.
        region: The region where the error occurred.
    """

    category: ErrorCategory
    original_exception: Exception
    message: str
    instance_type: str
    region: str


class ErrorClassifier:
    """Categorizes SageMaker exceptions by case-insensitive marker matching.

    Matching order: capacity -> quota -> family_support -> unknown.
    First match wins. Case-insensitive against the full error message.
    """

    CAPACITY_MARKERS = [
        "capacityerror",
        "insufficientinstancecapacity",
        "unable to provision",
    ]
    QUOTA_MARKERS = [
        "resourcelimitexceeded",
        "account-level service limit",
    ]
    FAMILY_SUPPORT_MARKERS = [
        "not supported",
        "not one of the supported values",
    ]

    @staticmethod
    def classify(
        exception: Exception, instance_type: str, region: str
    ) -> ClassifiedError:
        """Classify a SageMaker exception into exactly one category.

        Priority-ordered matching: capacity markers are checked first,
        then quota, then family_support. If no markers match, returns
        unknown. The same message always produces the same classification.

        Args:
            exception: The SageMaker API exception.
            instance_type: The instance type that triggered the error.
            region: The AWS region where the error occurred.

        Returns:
            A ClassifiedError with exactly one category assigned.
        """
        message = str(exception).lower()

        for marker in ErrorClassifier.CAPACITY_MARKERS:
            if marker in message:
                return ClassifiedError(
                    category=ErrorCategory.CAPACITY,
                    original_exception=exception,
                    message=str(exception),
                    instance_type=instance_type,
                    region=region,
                )

        for marker in ErrorClassifier.QUOTA_MARKERS:
            if marker in message:
                return ClassifiedError(
                    category=ErrorCategory.QUOTA,
                    original_exception=exception,
                    message=str(exception),
                    instance_type=instance_type,
                    region=region,
                )

        for marker in ErrorClassifier.FAMILY_SUPPORT_MARKERS:
            if marker in message:
                return ClassifiedError(
                    category=ErrorCategory.FAMILY_SUPPORT,
                    original_exception=exception,
                    message=str(exception),
                    instance_type=instance_type,
                    region=region,
                )

        return ClassifiedError(
            category=ErrorCategory.UNKNOWN,
            original_exception=exception,
            message=str(exception),
            instance_type=instance_type,
            region=region,
        )
