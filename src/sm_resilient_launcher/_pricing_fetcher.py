"""PricingFetcher — retrieves SageMaker instance pricing.

Uses the AWS Pricing API's ``get_products()`` to fetch current
on-demand pricing for SageMaker Batch Transform instance types.
The Pricing API is only available in ``us-east-1`` and ``ap-south-1``;
the client is pinned to ``us-east-1``.

Data is fetched deterministically — never by Bedrock.
"""

import json
import logging

import boto3

logger = logging.getLogger(__name__)


class PricingFetcher:
    """Fetches SageMaker instance pricing deterministically.

    Uses the ``pricing`` client's ``get_products()`` API with
    ``TERM_MATCH`` filters for ``instanceType`` and
    ``component=Transform``.  Parses on-demand pricing from the
    returned price-list JSON documents.

    The Pricing API is only available in ``us-east-1`` and
    ``ap-south-1``; the client is always created in ``us-east-1``.
    """

    SERVICE_CODE: str = "AmazonSageMaker"

    def __init__(self) -> None:
        self._client = boto3.client("pricing", region_name="us-east-1")

    def fetch_prices(
        self, instance_types: list[str]
    ) -> dict[str, float | None]:
        """Return ``{instance_type: hourly_usd}`` for each type.

        Queries the Pricing API for each instance type individually,
        filtering on ``instanceType`` and ``component=Transform``
        to retrieve Batch Transform on-demand pricing.

        Args:
            instance_types: List of SageMaker instance type names
                (e.g. ``["ml.g5.4xlarge", "ml.p3.2xlarge"]``).

        Returns:
            A dict mapping each instance type to its hourly USD
            price, or ``None`` if pricing cannot be retrieved.
        """
        results: dict[str, float | None] = {}
        for itype in instance_types:
            try:
                resp = self._client.get_products(
                    ServiceCode=self.SERVICE_CODE,
                    Filters=[
                        {
                            "Type": "TERM_MATCH",
                            "Field": "instanceType",
                            "Value": itype,
                        },
                        {
                            "Type": "TERM_MATCH",
                            "Field": "component",
                            "Value": "Transform",
                        },
                    ],
                    MaxResults=1,
                )
                if resp["PriceList"]:
                    price_doc = json.loads(resp["PriceList"][0])
                    on_demand = price_doc.get("terms", {}).get(
                        "OnDemand", {}
                    )
                    price_found = False
                    for term in on_demand.values():
                        for dim in term.get(
                            "priceDimensions", {}
                        ).values():
                            usd = dim["pricePerUnit"].get("USD")
                            if usd:
                                results[itype] = float(usd)
                                price_found = True
                                break
                        if price_found:
                            break
                    if not price_found:
                        results[itype] = None
                else:
                    results[itype] = None
            except Exception as exc:
                logger.debug(
                    "Pricing fetch failed for %s: %s", itype, exc
                )
                results[itype] = None
        return results
