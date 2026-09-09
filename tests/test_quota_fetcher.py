"""Unit tests for QuotaFetcher.

Validates that QuotaFetcher retrieves SageMaker Batch Transform
service quotas via the service_quotas client's get_service_quota()
API, resolves quota codes via list_service_quotas with keyword
filter, caches resolved codes, and gracefully returns None for
types whose quota cannot be retrieved.

Validates: Requirements 3.8
"""

from unittest.mock import MagicMock, patch, call

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
        # Set up quota code resolution
        paginator = MagicMock()
        paginator.paginate.return_value = [
            _make_quota_page([
                {
                    "QuotaName": "ml.g5.4xlarge for transform job usage",
                    "QuotaCode": "L-ABC12345",
                },
            ])
        ]
        mock_client.get_paginator.return_value = paginator

        # Set up get_service_quota response
        mock_client.get_service_quota.return_value = {
            "Quota": {"Value": 4.0}
        }

        result = fetcher.fetch_quotas(["ml.g5.4xlarge"])

        assert result == {"ml.g5.4xlarge": 4}
        mock_client.get_service_quota.assert_called_once_with(
            ServiceCode="sagemaker",
            QuotaCode="L-ABC12345",
        )

    def test_multiple_types_returns_all(self, fetcher, mock_client):
        """fetch_quotas handles multiple instance types in one call."""
        paginator = MagicMock()
        paginator.paginate.return_value = [
            _make_quota_page([
                {
                    "QuotaName": "ml.g5.4xlarge for transform job usage",
                    "QuotaCode": "L-ABC12345",
                },
                {
                    "QuotaName": "ml.g4dn.xlarge for transform job usage",
                    "QuotaCode": "L-DEF67890",
                },
            ])
        ]
        mock_client.get_paginator.return_value = paginator

        mock_client.get_service_quota.side_effect = [
            {"Quota": {"Value": 4.0}},
            {"Quota": {"Value": 2.0}},
        ]

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
                },
            ])
        ]
        mock_client.get_paginator.return_value = paginator

        mock_client.get_service_quota.return_value = {
            "Quota": {"Value": 8.0}
        }

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

    def test_get_service_quota_raises(self, fetcher, mock_client):
        """Returns None when get_service_quota raises an exception."""
        paginator = MagicMock()
        paginator.paginate.return_value = [
            _make_quota_page([
                {
                    "QuotaName": "ml.g5.4xlarge for transform job usage",
                    "QuotaCode": "L-ABC12345",
                },
            ])
        ]
        mock_client.get_paginator.return_value = paginator

        mock_client.get_service_quota.side_effect = Exception(
            "Throttling: Rate exceeded"
        )

        result = fetcher.fetch_quotas(["ml.g5.4xlarge"])
        assert result == {"ml.g5.4xlarge": None}

    def test_list_service_quotas_raises(self, fetcher, mock_client):
        """Returns None when list_service_quotas pagination raises."""
        paginator = MagicMock()
        paginator.paginate.side_effect = Exception("Access denied")
        mock_client.get_paginator.return_value = paginator

        result = fetcher.fetch_quotas(["ml.g5.4xlarge"])
        assert result == {"ml.g5.4xlarge": None}

    def test_partial_failure(self, fetcher, mock_client):
        """Some types succeed and others fail; each gets its own result."""
        paginator = MagicMock()
        paginator.paginate.return_value = [
            _make_quota_page([
                {
                    "QuotaName": "ml.g5.4xlarge for transform job usage",
                    "QuotaCode": "L-ABC",
                },
                {
                    "QuotaName": "ml.g4dn.xlarge for transform job usage",
                    "QuotaCode": "L-DEF",
                },
            ])
        ]
        mock_client.get_paginator.return_value = paginator

        # First call succeeds, second raises
        mock_client.get_service_quota.side_effect = [
            {"Quota": {"Value": 4.0}},
            Exception("Service error"),
        ]

        result = fetcher.fetch_quotas(["ml.g5.4xlarge", "ml.g4dn.xlarge"])
        assert result == {"ml.g5.4xlarge": 4, "ml.g4dn.xlarge": None}


# ---------------------------------------------------------------------------
# 4. Quota code resolution uses keyword matching correctly
# ---------------------------------------------------------------------------

class TestQuotaCodeResolution:
    """_resolve_quota_code uses case-insensitive keyword matching."""

    def test_matches_short_name_and_transform(self, fetcher, mock_client):
        """Quota code resolved when name contains short_name and 'transform'."""
        paginator = MagicMock()
        paginator.paginate.return_value = [
            _make_quota_page([
                {
                    "QuotaName": "ml.g5.4xlarge for Transform Job usage",
                    "QuotaCode": "L-MATCH",
                },
            ])
        ]
        mock_client.get_paginator.return_value = paginator

        code = fetcher._resolve_quota_code("ml.g5.4xlarge")
        assert code == "L-MATCH"

    def test_ignores_training_quota(self, fetcher, mock_client):
        """Quota entries for training (no 'transform') are ignored."""
        paginator = MagicMock()
        paginator.paginate.return_value = [
            _make_quota_page([
                {
                    "QuotaName": "ml.g5.4xlarge for training job usage",
                    "QuotaCode": "L-TRAIN",
                },
            ])
        ]
        mock_client.get_paginator.return_value = paginator

        code = fetcher._resolve_quota_code("ml.g5.4xlarge")
        assert code is None

    def test_case_insensitive_matching(self, fetcher, mock_client):
        """Matching is case-insensitive for both short_name and 'transform'."""
        paginator = MagicMock()
        paginator.paginate.return_value = [
            _make_quota_page([
                {
                    "QuotaName": "ML.G5.4XLARGE for TRANSFORM JOB usage",
                    "QuotaCode": "L-UPPER",
                },
            ])
        ]
        mock_client.get_paginator.return_value = paginator

        code = fetcher._resolve_quota_code("ml.g5.4xlarge")
        assert code == "L-UPPER"

    def test_multi_page_resolution(self, fetcher, mock_client):
        """Quota code can be found across multiple paginator pages."""
        paginator = MagicMock()
        paginator.paginate.return_value = [
            _make_quota_page([
                {
                    "QuotaName": "ml.p3.2xlarge for transform job usage",
                    "QuotaCode": "L-P3",
                },
            ]),
            _make_quota_page([
                {
                    "QuotaName": "ml.g5.4xlarge for transform job usage",
                    "QuotaCode": "L-G5",
                },
            ]),
        ]
        mock_client.get_paginator.return_value = paginator

        code = fetcher._resolve_quota_code("ml.g5.4xlarge")
        assert code == "L-G5"


# ---------------------------------------------------------------------------
# 5. Quota code caching
# ---------------------------------------------------------------------------

class TestQuotaCodeCaching:
    """_resolve_quota_code caches results so each type is looked up once."""

    def test_cache_hit_avoids_paginator(self, fetcher, mock_client):
        """Second call for the same type uses the cache, no API call."""
        paginator = MagicMock()
        paginator.paginate.return_value = [
            _make_quota_page([
                {
                    "QuotaName": "ml.g5.4xlarge for transform job usage",
                    "QuotaCode": "L-CACHED",
                },
            ])
        ]
        mock_client.get_paginator.return_value = paginator

        # First call — resolves via API
        code1 = fetcher._resolve_quota_code("ml.g5.4xlarge")
        assert code1 == "L-CACHED"

        # Reset the paginator to verify it's not called again
        mock_client.get_paginator.reset_mock()

        # Second call — should use cache
        code2 = fetcher._resolve_quota_code("ml.g5.4xlarge")
        assert code2 == "L-CACHED"
        mock_client.get_paginator.assert_not_called()

    def test_cache_stores_none_for_unresolvable(self, fetcher, mock_client):
        """None is cached for types that cannot be resolved, avoiding
        repeated paginator lookups."""
        paginator = MagicMock()
        paginator.paginate.return_value = [_make_quota_page([])]
        mock_client.get_paginator.return_value = paginator

        # First call — resolves to None
        code1 = fetcher._resolve_quota_code("ml.unknown.xlarge")
        assert code1 is None

        mock_client.get_paginator.reset_mock()

        # Second call — should use cached None
        code2 = fetcher._resolve_quota_code("ml.unknown.xlarge")
        assert code2 is None
        mock_client.get_paginator.assert_not_called()

    def test_fetch_quotas_uses_cached_codes(self, fetcher, mock_client):
        """fetch_quotas reuses cached quota codes across multiple calls."""
        paginator = MagicMock()
        paginator.paginate.return_value = [
            _make_quota_page([
                {
                    "QuotaName": "ml.g5.xlarge for transform job usage",
                    "QuotaCode": "L-G5X",
                },
            ])
        ]
        mock_client.get_paginator.return_value = paginator

        mock_client.get_service_quota.return_value = {
            "Quota": {"Value": 8.0}
        }

        # First call — resolves quota code via paginator
        result1 = fetcher.fetch_quotas(["ml.g5.xlarge"])
        assert result1 == {"ml.g5.xlarge": 8}

        # Reset paginator mock
        mock_client.get_paginator.reset_mock()

        # Second call — should use cached code, no paginator call
        mock_client.get_service_quota.return_value = {
            "Quota": {"Value": 16.0}
        }
        result2 = fetcher.fetch_quotas(["ml.g5.xlarge"])
        assert result2 == {"ml.g5.xlarge": 16}
        mock_client.get_paginator.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Constructor creates service-quotas client for the given region
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
