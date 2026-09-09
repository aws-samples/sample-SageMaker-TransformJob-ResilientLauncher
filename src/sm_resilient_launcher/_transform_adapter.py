"""TransformJobAdapter — concrete JobAdapter for Batch Transform.

Implements the JobAdapter interface for SageMaker Batch Transform jobs.
Polls DescribeTransformJob to detect completion, failure, or capacity
error states. Batch Transform hard-fails in ~40 seconds, so with a
10-second poll interval, failures are detected within ~60 seconds.
"""

import copy

import boto3

from sm_resilient_launcher._job_adapter import JobAdapter


class TransformJobAdapter(JobAdapter):
    """JobAdapter for SageMaker Batch Transform jobs.

    Wraps the SageMaker CreateTransformJob, DescribeTransformJob, and
    StopTransformJob APIs behind the JobAdapter abstract interface.
    Strategy engines, the Launcher, and the CrossRegionManager interact
    exclusively through this interface.
    """

    POLL_INTERVAL_SECONDS: int = 10
    COMPLETED_STATUS: str = "Completed"
    FAILED_STATUS: str = "Failed"
    STOPPED_STATUS: str = "Stopped"

    def __init__(self, region: str) -> None:
        self._client = boto3.client("sagemaker", region_name=region)
        self._region = region

    @property
    def region(self) -> str:
        """The AWS region this adapter targets."""
        return self._region

    def create_job(
        self, job_name: str, instance_type: str, boto3_params: dict
    ) -> dict:
        """Call CreateTransformJob with the selected instance type.

        Deep-copies ``boto3_params`` to avoid mutating the caller's
        dict, sets ``TransformJobName``, and substitutes
        ``TransformResources.InstanceType`` with the scalar
        ``instance_type`` argument.  All other caller params are
        preserved unchanged — including unrecognised future API
        parameters.

        Args:
            job_name: Unique name for the SageMaker transform job.
            instance_type: The instance type to use (scalar string).
            boto3_params: Parameters for create_transform_job.
                SDK-specific params have already been stripped.

        Returns:
            The raw CreateTransformJob API response dict.
        """
        params = copy.deepcopy(boto3_params)
        params["TransformJobName"] = job_name
        params.setdefault("TransformResources", {})
        params["TransformResources"]["InstanceType"] = instance_type
        return self._client.create_transform_job(**params)

    def describe_job(self, job_name: str) -> dict:
        """Return the current job description from SageMaker.

        Args:
            job_name: The name of the transform job to describe.

        Returns:
            The raw DescribeTransformJob response dict.
        """
        return self._client.describe_transform_job(
            TransformJobName=job_name
        )

    def stop_job(self, job_name: str) -> None:
        """Request cancellation of a running or pending transform job.

        Args:
            job_name: The name of the transform job to stop.
        """
        self._client.stop_transform_job(TransformJobName=job_name)

    def is_completed(self, description: dict) -> bool:
        """Return True if the transform job completed successfully.

        Args:
            description: The dict returned by describe_job.

        Returns:
            True when ``TransformJobStatus`` is ``"Completed"``.
        """
        return (
            description.get("TransformJobStatus") == self.COMPLETED_STATUS
        )

    def is_failed(self, description: dict) -> bool:
        """Return True if the transform job has reached a failed state.

        Both ``"Failed"`` and ``"Stopped"`` are treated as failure
        terminal states.

        Args:
            description: The dict returned by describe_job.

        Returns:
            True when ``TransformJobStatus`` is ``"Failed"`` or
            ``"Stopped"``.
        """
        return description.get("TransformJobStatus") in (
            self.FAILED_STATUS,
            self.STOPPED_STATUS,
        )

    def extract_error(self, description: dict) -> str:
        """Extract the error message from a failed job description.

        Args:
            description: The dict returned by describe_job for a
                failed job.

        Returns:
            The ``FailureReason`` string, or ``"Unknown error"`` if
            the field is absent.
        """
        return description.get("FailureReason", "Unknown error")
