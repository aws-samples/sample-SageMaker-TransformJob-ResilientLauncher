"""Unit tests for QuotaFetcher.

Validates that QuotaFetcher retrieves SageMaker Batch Transform
service quotas via a single-pass cache approach.  On first call,
it paginates all SageMaker quotas, filters for transform-related
entries, and caches quota codes and values.  Subsequent lookups
are instant from the cache.

Validates: Requirements 3.8
"""

from unittest.mock import MagicMock, patch

import pytest

from sm_resilient_launcher._quota_fetcher import QuotaFetcher


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_client():
    """Return a mocked service-quotas client."""
    client = MagicMock()
    # Default: paginator returns no pages (empty)
    paginator = MagicMock()
    paginator.paginate.return_value = []
    client.get_paginator.return_value = paginator
    return client


@pytest.fixture()
def fetcher(mock_client):
    """Return a QuotaFetcher wired to the mocked client."""
    with patch("boto3.client", return_value=mock_client):
        return QuotaFetcher(region="us-east-1")


def _make_quota_page(quotas: list[dict]) -> dict:
    """Build a single list_service_quotas response page."""
    return {"Quotas": quotas}


# ---------------------------------------------------------------------------
# 1. fetch_quotas returns correct quota values for known types
# ---------------------------------------------------------------------------

class TestFetchQuotasSuccess:
    """fetch_quotas returns integer quota values for resolvable types."""

    def test_single_type_returns_quota(self, fetcher, mock_client):
        # Set up single-pass cache loading
        paginator = MagicMock()
        paginator.paginate.return_value = [
            _make_quota_page([
                {
                    "QuotaName": "ml.g5.4xlarge for transform job usage",
                    "QuotaCode": "L-ABC12345",
                    "Value": 4.0,
                },
            ])
        ]
        mock_client.get_paginator.return_value = paginator

        result = fetcher.fetch_quotas(["ml.g5.4xlarge"])

        assert result == {"ml.g5.4xlarge": 4}

    def test_multiple_types_returns_all(self, fetcher, mock_client):
        """fetch_quotas handles multiple instance types in one call."""
        paginator = MagicMock()
        paginator.paginate.return_value = [
            _make_quota_page([
                {
                    "QuotaName": "ml.g5.4xlarge for transform job usage",
                    "QuotaCode": "L-ABC12345",
                    "Value": 4.0,
                },
                {
                    "QuotaName": "ml.g4dn.xlarge for transform job usage",
                    "QuotaCode": "L-DEF67890",
                    "Value": 2.0,
                },
            ])
        ]
        mock_client.get_paginator.return_value = paginator

        result = fetcher.fetch_quotas(["ml.g5.4xlarge", "ml.g4dn.xlarge"])

        assert result == {"ml.g5.4xlarge": 4, "ml.g4dn.xlarge": 2}

    def test_quota_value_truncated_to_int(self, fetcher, mock_client):
        """Floating-point quota values are converted to int."""
        paginator = MagicMock()
        paginator.paginate.return_value = [
            _make_quota_page([
                {
                    "QuotaName": "ml.g5.xlarge for transform job usage",
                    "QuotaCode": "L-111",
                    "Value": 8.0,
                },
            ])
        ]
        mock_client.get_paginator.return_value = paginator

        result = fetcher.fetch_quotas(["ml.g5.xlarge"])
        assert result["ml.g5.xlarge"] == 8
        assert isinstance(result["ml.g5.xlarge"], int)


# ---------------------------------------------------------------------------
# 2. fetch_quotas returns None for unresolvable types
# ---------------------------------------------------------------------------

class TestFetchQuotasNone:
    """fetch_quotas returns None when quota code cannot be resolved."""

    def test_no_matching_quota_name(self, fetcher, mock_client):
        """Returns None when no quota name matches the instance type."""
        paginator = MagicMock()
        paginator.paginate.return_value = [
            _make_quota_page([
                {
                    "QuotaName": "ml.p3.2xlarge for training job usage",
                    "QuotaCode": "L-TRAIN",
                    "Value": 4.0,
                },
            ])
        ]
        mock_client.get_paginator.return_value = paginator

        result = fetcher.fetch_quotas(["ml.g5.4xlarge"])
        assert result == {"ml.g5.4xlarge": None}

    def test_empty_quota_list(self, fetcher, mock_client):
        """Returns None when list_service_quotas returns no quotas."""
        paginator = MagicMock()
        paginator.paginate.return_value = [_make_quota_page([])]
        mock_client.get_paginator.return_value = paginator

        result = fetcher.fetch_quotas(["ml.g5.4xlarge"])
        assert result == {"ml.g5.4xlarge": None}

    def test_empty_instance_types_list(self, fetcher):
        """Returns empty dict for an empty instance_types list."""
        result = fetcher.fetch_quotas([])
        assert result == {}


# ---------------------------------------------------------------------------
# 3. fetch_quotas handles API errors gracefully
# ---------------------------------------------------------------------------

class TestFetchQuotasErrorHandling:
    """fetch_quotas returns None and logs at debug level on failures."""

    def test_list_service_quotas_raises(self, fetcher, mock_client):
        """Returns None when list_service_quotas pagination raises."""
        paginator = MagicMock()
        paginator.paginate.side_effect = Exception("Access denied")
        mock_client.get_paginator.return_value = paginator

        result = fetcher.fetch_quotas(["ml.g5.4xlarge"])
        assert result == {"ml.g5.4xlarge": None}

    def test_get_service_quota_fallback_on_code_only(
        self, fetcher, mock_client
    ):
        """When cache has code but no value, get_service_quota failure
        returns None."""
        # Manually populate cache with code only (no value)
        fetcher._quota_code_cache["ml.g5.4xlarge"] = "L-ABC12345"
        fetcher._cache_loaded = True

        mock_client.get_service_quota.side_effect = Exception(
            "Throttling: Rate exceeded"
        )

        result = fetcher.fetch_quotas(["ml.g5.4xlarge"])
        assert result == {"ml.g5.4xlarge": None}


# ---------------------------------------------------------------------------
# 4. Single-pass cache loads all transform quotas across pages
# ---------------------------------------------------------------------------

class TestSinglePassCache:
    """_ensure_cache_loaded paginates once and caches all transform quotas."""

    def test_multi_page_all_types_cached(self, fetcher, mock_client):
        """Types on different pages are all cached in a single pass."""
        paginator = MagicMock()
        paginator.paginate.return_value = [
            _make_quota_page([
                {
                    "QuotaName": "ml.p3.2xlarge for transform job usage",
                    "QuotaCode": "L-P3",
                    "Value": 2.0,
                },
            ]),
            _make_quota_page([
                {
                    "QuotaName": "ml.g5.4xlarge for transform job usage",
                    "QuotaCode": "L-G5",
                    "Value": 4.0,
                },
            ]),
        ]
        mock_client.get_paginator.return_value = paginator

        result = fetcher.fetch_quotas(
            ["ml.g5.4xlarge", "ml.p3.2xlarge"]
        )
        assert result == {"ml.g5.4xlarge": 4, "ml.p3.2xlarge": 2}

    def test_cache_only_loaded_once(self, fetcher, mock_client):
        """Second fetch_quotas call reuses cached data, no new pagination."""
        paginator = MagicMock()
        paginator.paginate.return_value = [
            _make_quota_page([
                {
                    "QuotaName": "ml.g5.xlarge for transform job usage",
                    "QuotaCode": "L-G5X",
                    "Value": 8.0,
                },
            ])
        ]
        mock_client.get_paginator.return_value = paginator

        # First call — loads cache
        result1 = fetcher.fetch_quotas(["ml.g5.xlarge"])
        assert result1 == {"ml.g5.xlarge": 8}

        # Reset paginator mock
        mock_client.get_paginator.reset_mock()

        # Second call — should use cached values, no paginator call
        result2 = fetcher.fetch_quotas(["ml.g5.xlarge"])
        assert result2 == {"ml.g5.xlarge": 8}
        mock_client.get_paginator.assert_not_called()

    def test_training_quotas_not_cached(self, fetcher, mock_client):
        """Quotas without 'transform' in the name are ignored."""
        paginator = MagicMock()
        paginator.paginate.return_value = [
            _make_quota_page([
                {
                    "QuotaName": "ml.g5.4xlarge for training job usage",
                    "QuotaCode": "L-TRAIN",
                    "Value": 10.0,
                },
                {
                    "QuotaName": "ml.g5.4xlarge for transform job usage",
                    "QuotaCode": "L-TRANSFORM",
                    "Value": 4.0,
                },
            ])
        ]
        mock_client.get_paginator.return_value = paginator

        result = fetcher.fetch_quotas(["ml.g5.4xlarge"])
        # Should use the transform quota, not training
        assert result == {"ml.g5.4xlarge": 4}
        assert fetcher._quota_code_cache["ml.g5.4xlarge"] == "L-TRANSFORM"

    def test_case_insensitive_matching(self, fetcher, mock_client):
        """Matching is case-insensitive for 'transform' in quota names."""
        paginator = MagicMock()
        paginator.paginate.return_value = [
            _make_quota_page([
                {
                    "QuotaName": "ML.G5.4XLARGE for TRANSFORM JOB usage",
                    "QuotaCode": "L-UPPER",
                    "Value": 6.0,
                },
            ])
        ]
        mock_client.get_paginator.return_value = paginator

        result = fetcher.fetch_quotas(["ml.g5.4xlarge"])
        assert result == {"ml.g5.4xlarge": 6}


# ---------------------------------------------------------------------------
# 5. Constructor creates service-quotas client for the given region
# ---------------------------------------------------------------------------

class TestConstructor:
    """QuotaFetcher.__init__ creates a service-quotas client."""

    def test_client_created_with_region(self):
        with patch("boto3.client") as mock_boto:
            QuotaFetcher(region="eu-west-1")
            mock_boto.assert_called_once_with(
                "service-quotas", region_name="eu-west-1"
            )

    def test_different_region(self):
        with patch("boto3.client") as mock_boto:
            QuotaFetcher(region="ap-southeast-1")
            mock_boto.assert_called_once_with(
                "service-quotas", region_name="ap-southeast-1"
            )

    def test_initial_state(self):
        """New instance has empty caches and cache_loaded=False."""
        with patch("boto3.client"):
            qf = QuotaFetcher(region="us-east-1")
            assert qf._cache_loaded is False
            assert qf._quota_code_cache == {}
            assert qf._quota_value_cache == {}
