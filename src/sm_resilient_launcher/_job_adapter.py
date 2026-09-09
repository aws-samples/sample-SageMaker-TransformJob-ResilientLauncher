"""JobAdapter — abstract base class for SageMaker job lifecycle operations.

Defines the interface that strategy engines (SequentialEngine, RaceEngine),
the Launcher orchestrator, and the CrossRegionManager interact with.
Contains no Batch Transform-specific logic — a coworker can later add
TrainingJobAdapter by implementing this interface without modifying any
upstream component.
"""

from abc import ABC, abstractmethod


class JobAdapter(ABC):
    """Abstract interface for SageMaker job lifecycle operations.

    All job-type-specific behavior lives in concrete subclasses
    (e.g. TransformJobAdapter). The Launcher, SequentialEngine,
    RaceEngine, and CrossRegionManager call only these methods,
    so adding a new job type requires only a new subclass.
    """

    @abstractmethod
    def create_job(
        self, job_name: str, instance_type: str, boto3_params: dict
    ) -> dict:
        """Submit the job and return the raw API response.

        Args:
            job_name: Unique name for the SageMaker job.
            instance_type: The instance type to use (scalar string).
            boto3_params: Parameters for the underlying create API call.
                SDK-specific params have already been stripped.

        Returns:
            The raw API response dict from SageMaker.
        """
        ...

    @abstractmethod
    def describe_job(self, job_name: str) -> dict:
        """Return the current job description from SageMaker.

        Args:
            job_name: The name of the SageMaker job to describe.

        Returns:
            The raw DescribeJob response dict.
        """
        ...

    @abstractmethod
    def stop_job(self, job_name: str) -> None:
        """Request cancellation of a running or pending job.

        Args:
            job_name: The name of the SageMaker job to stop.
        """
        ...

    @abstractmethod
    def is_completed(self, description: dict) -> bool:
        """Return True if the job has reached a successful terminal state.

        Args:
            description: The dict returned by describe_job.

        Returns:
            True if the job completed successfully.
        """
        ...

    @abstractmethod
    def is_failed(self, description: dict) -> bool:
        """Return True if the job has reached a failed terminal state.

        Args:
            description: The dict returned by describe_job.

        Returns:
            True if the job failed or was stopped.
        """
        ...

    @abstractmethod
    def extract_error(self, description: dict) -> str:
        """Extract the error message from a failed job description.

        Args:
            description: The dict returned by describe_job for a
                failed job.

        Returns:
            The error/failure reason string.
        """
        ...
