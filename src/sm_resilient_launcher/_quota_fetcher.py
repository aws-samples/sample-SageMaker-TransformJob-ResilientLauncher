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

    Uses a single-pass approach: on first call, paginates ALL SageMaker
    quotas via ``list_service_quotas``, filters for transform-related
    entries, and caches every quota code and value.  Subsequent lookups
    are instant from the cache.

    This avoids the bug where per-type pagination missed types whose
    quota entries appeared on different pages.
    """

    SERVICE_CODE = "sagemaker"

    def __init__(self, region: str) -> None:
        self._client = boto3.client("service-quotas", region_name=region)
        self._region = region
        self._quota_code_cache: dict[str, str | None] = {}
        self._quota_value_cache: dict[str, int] = {}
        self._cache_loaded: bool = False

    def get_all_transform_types(self) -> list[str]:
        """Return all Batch Transform instance types found in the quota API.

        Loads the cache first if needed, then returns all instance type
        names that have a transform quota entry.
        """
        self._ensure_cache_loaded()
        return sorted(self._quota_value_cache.keys())

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
        self._ensure_cache_loaded()
        results: dict[str, int | None] = {}
        for itype in instance_types:
            if itype in self._quota_value_cache:
                results[itype] = self._quota_value_cache[itype]
            elif itype in self._quota_code_cache:
                # Have code but no cached value — fetch it
                try:
                    resp = self._client.get_service_quota(
                        ServiceCode=self.SERVICE_CODE,
                        QuotaCode=self._quota_code_cache[itype],
                    )
                    results[itype] = int(resp["Quota"]["Value"])
                except Exception:
                    results[itype] = None
            else:
                results[itype] = None
        return results

    def _ensure_cache_loaded(self) -> None:
        """Load all SageMaker transform quotas into cache in one pass.

        Paginates ``list_service_quotas`` once, filters for quotas
        containing ``'transform'`` in the name, and builds a mapping
        from instance type to quota code + value.
        """
        if self._cache_loaded:
            return

        try:
            paginator = self._client.get_paginator("list_service_quotas")
            for page in paginator.paginate(ServiceCode=self.SERVICE_CODE):
                for quota in page["Quotas"]:
                    name_lower = quota["QuotaName"].lower()
                    if "transform" not in name_lower:
                        continue
                    # Extract the instance type pattern from the quota name
                    # Pattern: "ml.g4dn.xlarge for transform job usage"
                    # We match against the part before " for "
                    parts = name_lower.split(" for ")
                    if parts:
                        instance_pattern = parts[0].strip()
                        # Remove "ml." prefix for matching
                        short = instance_pattern.replace("ml.", "")
                        self._quota_code_cache[f"ml.{short}"] = quota[
                            "QuotaCode"
                        ]
                        self._quota_value_cache[f"ml.{short}"] = int(
                            quota["Value"]
                        )
        except Exception as exc:
            logger.debug("Failed to load quota cache: %s", exc)

        self._cache_loaded = True
