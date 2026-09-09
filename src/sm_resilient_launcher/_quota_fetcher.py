"""QuotaFetcher — retrieves SageMaker Batch Transform service quotas.

Uses the ``service_quotas`` client's ``get_service_quota()`` API to
fetch per-instance-type quota values deterministically.  Results are
never fetched via Bedrock.
"""

import logging

import boto3

logger = logging.getLogger(__name__)


class QuotaFetcher:
    """Fetches SageMaker Batch Transform service quotas deterministically.

    Uses the ``service_quotas`` client's ``get_service_quota()`` API.
    Quota codes are resolved once per instance type via
    ``list_service_quotas`` and cached for subsequent lookups.
    """

    SERVICE_CODE = "sagemaker"

    def __init__(self, region: str) -> None:
        self._client = boto3.client("service-quotas", region_name=region)
        self._region = region
        self._quota_code_cache: dict[str, str | None] = {}

    def fetch_quotas(
        self, instance_types: list[str]
    ) -> dict[str, int | None]:
        """Return ``{instance_type: quota_value}`` for each type.

        Returns ``None`` for a type if the quota cannot be retrieved
        (quota does not exist, or API error).

        Args:
            instance_types: List of SageMaker instance type strings
                (e.g. ``["ml.g5.4xlarge", "ml.g4dn.xlarge"]``).

        Returns:
            A dict mapping each instance type to its integer quota
            value, or ``None`` when the quota is unavailable.
        """
        results: dict[str, int | None] = {}
        for itype in instance_types:
            quota_code = self._resolve_quota_code(itype)
            if quota_code is None:
                results[itype] = None
                continue
            try:
                resp = self._client.get_service_quota(
                    ServiceCode=self.SERVICE_CODE,
                    QuotaCode=quota_code,
                )
                results[itype] = int(resp["Quota"]["Value"])
            except Exception as exc:
                logger.debug(
                    "Quota fetch failed for %s in %s: %s",
                    itype,
                    self._region,
                    exc,
                )
                results[itype] = None
        return results

    def _resolve_quota_code(self, instance_type: str) -> str | None:
        """Look up the quota code for a Batch Transform instance type.

        Uses ``list_service_quotas`` with keyword filter.  Results are
        cached so each instance type is resolved at most once.

        Args:
            instance_type: A SageMaker instance type string
                (e.g. ``"ml.g5.4xlarge"``).

        Returns:
            The quota code string, or ``None`` if it cannot be found.
        """
        if instance_type in self._quota_code_cache:
            return self._quota_code_cache[instance_type]

        try:
            # Extract family.size (e.g., "g5.4xlarge" from "ml.g5.4xlarge")
            short_name = instance_type.replace("ml.", "")
            paginator = self._client.get_paginator("list_service_quotas")
            for page in paginator.paginate(ServiceCode=self.SERVICE_CODE):
                for quota in page["Quotas"]:
                    name_lower = quota["QuotaName"].lower()
                    if (
                        short_name.lower() in name_lower
                        and "transform" in name_lower
                    ):
                        code = quota["QuotaCode"]
                        self._quota_code_cache[instance_type] = code
                        return code
        except Exception as exc:
            logger.debug(
                "Quota code lookup failed for %s: %s",
                instance_type,
                exc,
            )

        self._quota_code_cache[instance_type] = None
        return None
