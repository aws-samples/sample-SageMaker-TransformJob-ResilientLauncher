"""Unit tests for IntelligentRouter.

Property 6: IntelligentRouter prompt contains all deterministic facts
    For any workload description and set of candidate instance types, the
    Bedrock prompt constructed by _build_prompt() SHALL contain: the workload
    description text, the original instance type, GPU memory reference data
    for each candidate, the deterministically-fetched quota value for each
    candidate, and the deterministically-fetched price for each candidate.
    Bedrock is never used to look up quotas or pricing.

Property 7: Bedrock failure falls back to TypeExpander
    For any Bedrock Converse API call failure during AI Expansion, the
    IntelligentRouter.generate_plan() method SHALL fall back to the
    deterministic TypeExpander and return a valid fallback list with
    strategy "sequential" and reasoning None.

Validates: Requirements 3.1, 3.9, 3.10, 3.11, 3.12
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from sm_resilient_launcher._intelligent_router import IntelligentRouter
from sm_resilient_launcher._type_expander import TypeExpander


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_bedrock_response(
    priority_list: list[str],
    strategy: str = "sequential",
    reasoning: str = "GPU fit looks good",
) -> dict:
    """Build a minimal Bedrock Converse API response dict."""
    body = json.dumps(
        {
            "priority_list": priority_list,
            "strategy": strategy,
            "reasoning": reasoning,
        }
    )
    return {
        "output": {
            "message": {
                "content": [{"text": body}]
            }
        }
    }


def _make_bedrock_response_with_preamble(
    priority_list: list[str],
    strategy: str = "sequential",
    reasoning: str = "GPU fit looks good",
) -> dict:
    """Build a Bedrock response where JSON is wrapped in prose text."""
    body = json.dumps(
        {
            "priority_list": priority_list,
            "strategy": strategy,
            "reasoning": reasoning,
        }
    )
    text = f"Here is my analysis:\n{body}\nLet me know if you need more."
    return {
        "output": {
            "message": {
                "content": [{"text": text}]
            }
        }
    }


@pytest.fixture()
def mock_quota_fetcher():
    fetcher = MagicMock()
    fetcher.fetch_quotas.return_value = {
        "ml.g5.4xlarge": 4,
        "ml.g5.12xlarge": 2,
        "ml.p4d.24xlarge": 1,
    }
    return fetcher


@pytest.fixture()
def mock_pricing_fetcher():
    fetcher = MagicMock()
    fetcher.fetch_prices.return_value = {
        "ml.g5.4xlarge": 2.03,
        "ml.g5.12xlarge": 7.09,
        "ml.p4d.24xlarge": 37.688,
    }
    return fetcher


@pytest.fixture()
def mock_type_expander():
    expander = MagicMock(spec=TypeExpander)
    expander.expand.return_value = [
        "ml.g5.4xlarge",
        "ml.g5.12xlarge",
        "ml.p4d.24xlarge",
    ]
    return expander


@pytest.fixture()
def router(mock_quota_fetcher, mock_pricing_fetcher, mock_type_expander):
    """Build an IntelligentRouter with all mocked dependencies."""
    with patch("sm_resilient_launcher._intelligent_router.boto3") as mock_boto3:
        mock_bedrock = MagicMock()
        mock_boto3.client.return_value = mock_bedrock

        r = IntelligentRouter(
            quota_fetcher=mock_quota_fetcher,
            pricing_fetcher=mock_pricing_fetcher,
            type_expander=mock_type_expander,
            region="us-east-1",
        )
        # Expose the mock bedrock client for test assertions
        r._bedrock = mock_bedrock
        yield r


# ---------------------------------------------------------------------------
# 1. generate_plan — happy path: Bedrock returns valid JSON
# ---------------------------------------------------------------------------

class TestGeneratePlanHappyPath:
    """generate_plan calls TypeExpander, fetches facts, calls Bedrock,
    and returns the parsed priority list, strategy, and reasoning."""

    def test_returns_bedrock_ordered_list(self, router, mock_type_expander):
        router._bedrock.converse.return_value = _make_bedrock_response(
            priority_list=["ml.g5.4xlarge", "ml.p4d.24xlarge"],
            strategy="sequential",
            reasoning="A10G fits, A100 as fallback",
        )

        types, strategy, reasoning = router.generate_plan(
            "ml.g5.4xlarge", "ResNet-50 inference"
        )

        assert types == ["ml.g5.4xlarge", "ml.p4d.24xlarge"]
        assert strategy == "sequential"
        assert reasoning == "A10G fits, A100 as fallback"

    def test_calls_type_expander_for_candidates(
        self, router, mock_type_expander
    ):
        router._bedrock.converse.return_value = _make_bedrock_response(
            ["ml.g5.4xlarge"]
        )

        router.generate_plan("ml.g5.4xlarge", "batch job")

        mock_type_expander.expand.assert_called_once_with(
            "ml.g5.4xlarge", "us-east-1", set()
        )

    def test_fetches_quotas_for_candidates(
        self, router, mock_quota_fetcher, mock_type_expander
    ):
        router._bedrock.converse.return_value = _make_bedrock_response(
            ["ml.g5.4xlarge"]
        )

        router.generate_plan("ml.g5.4xlarge", "batch job")

        mock_quota_fetcher.fetch_quotas.assert_called_once_with(
            ["ml.g5.4xlarge", "ml.g5.12xlarge", "ml.p4d.24xlarge"]
        )

    def test_fetches_prices_for_candidates(
        self, router, mock_pricing_fetcher
    ):
        router._bedrock.converse.return_value = _make_bedrock_response(
            ["ml.g5.4xlarge"]
        )

        router.generate_plan("ml.g5.4xlarge", "batch job")

        mock_pricing_fetcher.fetch_prices.assert_called_once_with(
            ["ml.g5.4xlarge", "ml.g5.12xlarge", "ml.p4d.24xlarge"]
        )

    def test_race_strategy_returned(self, router):
        router._bedrock.converse.return_value = _make_bedrock_response(
            ["ml.g5.4xlarge", "ml.g5.12xlarge"],
            strategy="race",
            reasoning="plenty of quota, race for speed",
        )

        _, strategy, _ = router.generate_plan(
            "ml.g5.4xlarge", "urgent batch"
        )

        assert strategy == "race"

    def test_excluded_types_forwarded_to_expander(
        self, router, mock_type_expander
    ):
        router._bedrock.converse.return_value = _make_bedrock_response(
            ["ml.g5.4xlarge"]
        )

        router.generate_plan(
            "ml.g5.4xlarge",
            "batch job",
            excluded_types={"ml.p3.2xlarge"},
        )

        mock_type_expander.expand.assert_called_once_with(
            "ml.g5.4xlarge", "us-east-1", {"ml.p3.2xlarge"}
        )


# ---------------------------------------------------------------------------
# 2. generate_plan — original type is always first
# ---------------------------------------------------------------------------

class TestOriginalTypeFirst:
    """The original instance type is always the first element in
    the returned list, even if Bedrock returns it elsewhere."""

    def test_original_type_moved_to_front(self, router):
        router._bedrock.converse.return_value = _make_bedrock_response(
            ["ml.p4d.24xlarge", "ml.g5.4xlarge"],
            strategy="sequential",
        )

        types, _, _ = router.generate_plan(
            "ml.g5.4xlarge", "batch inference"
        )

        assert types[0] == "ml.g5.4xlarge"
        assert "ml.p4d.24xlarge" in types

    def test_original_type_already_first(self, router):
        router._bedrock.converse.return_value = _make_bedrock_response(
            ["ml.g5.4xlarge", "ml.g5.12xlarge"],
        )

        types, _, _ = router.generate_plan(
            "ml.g5.4xlarge", "batch inference"
        )

        assert types[0] == "ml.g5.4xlarge"
        assert types == ["ml.g5.4xlarge", "ml.g5.12xlarge"]

    def test_original_type_not_in_bedrock_list(self, router):
        """If Bedrock omits the original type entirely, it gets inserted."""
        router._bedrock.converse.return_value = _make_bedrock_response(
            ["ml.p4d.24xlarge", "ml.g5.12xlarge"],
        )

        types, _, _ = router.generate_plan(
            "ml.g5.4xlarge", "batch inference"
        )

        assert types[0] == "ml.g5.4xlarge"


# ---------------------------------------------------------------------------
# 3. generate_plan — Bedrock failure falls back to TypeExpander
# ---------------------------------------------------------------------------

class TestBedrockFallback:
    """When Bedrock fails, generate_plan falls back to TypeExpander
    and returns strategy='sequential', reasoning=None."""

    def test_bedrock_exception_triggers_fallback(
        self, router, mock_type_expander
    ):
        router._bedrock.converse.side_effect = RuntimeError(
            "Bedrock unavailable"
        )

        types, strategy, reasoning = router.generate_plan(
            "ml.g5.4xlarge", "batch job"
        )

        assert types == [
            "ml.g5.4xlarge",
            "ml.g5.12xlarge",
            "ml.p4d.24xlarge",
        ]
        assert strategy == "sequential"
        assert reasoning is None

    def test_fallback_calls_type_expander_again(
        self, router, mock_type_expander
    ):
        router._bedrock.converse.side_effect = RuntimeError("fail")

        router.generate_plan("ml.g5.4xlarge", "batch job")

        # TypeExpander.expand called twice: once for candidates, once
        # for fallback
        assert mock_type_expander.expand.call_count == 2

    def test_fallback_passes_excluded_types(
        self, router, mock_type_expander
    ):
        router._bedrock.converse.side_effect = RuntimeError("fail")

        router.generate_plan(
            "ml.g5.4xlarge",
            "batch job",
            excluded_types={"ml.p3.2xlarge"},
        )

        # The fallback call should include excluded_types
        fallback_call = mock_type_expander.expand.call_args_list[1]
        assert fallback_call.args == (
            "ml.g5.4xlarge",
            "us-east-1",
            {"ml.p3.2xlarge"},
        )


# ---------------------------------------------------------------------------
# 4. _build_prompt — contains all deterministic facts
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    """_build_prompt includes workload description, original type,
    GPU metadata, quota values, and pricing for every candidate."""

    def test_prompt_contains_workload_description(self, router):
        prompt = router._build_prompt(
            "ml.g5.4xlarge",
            ["ml.g5.4xlarge"],
            "ResNet-50 inference, fp32, ~200MB GPU memory",
            {"ml.g5.4xlarge": 4},
            {"ml.g5.4xlarge": 2.03},
        )

        assert "ResNet-50 inference, fp32, ~200MB GPU memory" in prompt

    def test_prompt_contains_original_type(self, router):
        prompt = router._build_prompt(
            "ml.g5.4xlarge",
            ["ml.g5.4xlarge"],
            "batch inference",
            {"ml.g5.4xlarge": 4},
            {"ml.g5.4xlarge": 2.03},
        )

        assert "ml.g5.4xlarge" in prompt
        assert "Original requested type: ml.g5.4xlarge" in prompt

    def test_prompt_contains_gpu_metadata(self, router):
        prompt = router._build_prompt(
            "ml.g5.4xlarge",
            ["ml.g5.4xlarge", "ml.p4d.24xlarge"],
            "batch job",
            {"ml.g5.4xlarge": 4, "ml.p4d.24xlarge": 1},
            {"ml.g5.4xlarge": 2.03, "ml.p4d.24xlarge": 37.688},
        )

        assert "A10G 24GB x1" in prompt
        assert "A100 40GB x8" in prompt

    def test_prompt_contains_quota_values(self, router):
        prompt = router._build_prompt(
            "ml.g5.4xlarge",
            ["ml.g5.4xlarge", "ml.p4d.24xlarge"],
            "batch job",
            {"ml.g5.4xlarge": 4, "ml.p4d.24xlarge": 1},
            {"ml.g5.4xlarge": 2.03, "ml.p4d.24xlarge": 37.688},
        )

        assert "quota=4" in prompt
        assert "quota=1" in prompt

    def test_prompt_contains_pricing(self, router):
        prompt = router._build_prompt(
            "ml.g5.4xlarge",
            ["ml.g5.4xlarge", "ml.p4d.24xlarge"],
            "batch job",
            {"ml.g5.4xlarge": 4, "ml.p4d.24xlarge": 1},
            {"ml.g5.4xlarge": 2.03, "ml.p4d.24xlarge": 37.688},
        )

        assert "$2.030/hr" in prompt
        assert "$37.688/hr" in prompt

    def test_prompt_handles_none_quota(self, router):
        prompt = router._build_prompt(
            "ml.g5.4xlarge",
            ["ml.g5.4xlarge"],
            "batch job",
            {"ml.g5.4xlarge": None},
            {"ml.g5.4xlarge": 2.03},
        )

        assert "quota=None" in prompt

    def test_prompt_handles_none_pricing(self, router):
        prompt = router._build_prompt(
            "ml.g5.4xlarge",
            ["ml.g5.4xlarge"],
            "batch job",
            {"ml.g5.4xlarge": 4},
            {"ml.g5.4xlarge": None},
        )

        assert "price=unknown" in prompt

    def test_prompt_handles_unknown_instance_type(self, router):
        """Instance type not in GPU_CATALOG gets 'unknown GPU'."""
        prompt = router._build_prompt(
            "ml.x99.mega",
            ["ml.x99.mega"],
            "batch job",
            {"ml.x99.mega": 2},
            {"ml.x99.mega": 10.0},
        )

        assert "unknown GPU" in prompt


# ---------------------------------------------------------------------------
# 5. _parse_response — extracts JSON from Bedrock response
# ---------------------------------------------------------------------------

class TestParseResponse:
    """_parse_response extracts priority_list, strategy, and reasoning
    from the Bedrock Converse API response."""

    def test_parses_clean_json(self, router):
        response = _make_bedrock_response(
            ["ml.g5.4xlarge", "ml.p4d.24xlarge"],
            strategy="race",
            reasoning="race for speed",
        )

        types, strategy, reasoning = router._parse_response(response)

        assert types == ["ml.g5.4xlarge", "ml.p4d.24xlarge"]
        assert strategy == "race"
        assert reasoning == "race for speed"

    def test_parses_json_wrapped_in_text(self, router):
        response = _make_bedrock_response_with_preamble(
            ["ml.g5.4xlarge"],
            strategy="sequential",
            reasoning="single type is enough",
        )

        types, strategy, reasoning = router._parse_response(response)

        assert types == ["ml.g5.4xlarge"]
        assert strategy == "sequential"

    def test_invalid_strategy_defaults_to_sequential(self, router):
        body = json.dumps(
            {
                "priority_list": ["ml.g5.4xlarge"],
                "strategy": "turbo",
                "reasoning": "invented strategy",
            }
        )
        response = {
            "output": {
                "message": {"content": [{"text": body}]}
            }
        }

        _, strategy, _ = router._parse_response(response)

        assert strategy == "sequential"

    def test_missing_reasoning_uses_raw_text(self, router):
        body = json.dumps(
            {
                "priority_list": ["ml.g5.4xlarge"],
                "strategy": "sequential",
            }
        )
        response = {
            "output": {
                "message": {"content": [{"text": body}]}
            }
        }

        _, _, reasoning = router._parse_response(response)

        # Falls back to the full text when "reasoning" key is absent
        assert reasoning == body


# ---------------------------------------------------------------------------
# 6. Bedrock Converse call is constructed correctly
# ---------------------------------------------------------------------------

class TestBedrockCallConstruction:
    """The Bedrock Converse API call uses the correct model ID and
    message structure."""

    def test_bedrock_called_with_model_id(self, router):
        router._bedrock.converse.return_value = _make_bedrock_response(
            ["ml.g5.4xlarge"]
        )

        router.generate_plan("ml.g5.4xlarge", "batch job")

        call_kwargs = router._bedrock.converse.call_args.kwargs
        assert call_kwargs["modelId"] == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    def test_bedrock_receives_user_message(self, router):
        router._bedrock.converse.return_value = _make_bedrock_response(
            ["ml.g5.4xlarge"]
        )

        router.generate_plan("ml.g5.4xlarge", "batch job")

        call_kwargs = router._bedrock.converse.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert len(messages[0]["content"]) == 1
        assert "text" in messages[0]["content"][0]


# ---------------------------------------------------------------------------
# 7. Constructor creates bedrock-runtime client in the specified region
# ---------------------------------------------------------------------------

class TestConstructor:
    """IntelligentRouter.__init__ wires dependencies and creates
    a bedrock-runtime client in the specified region."""

    def test_creates_bedrock_client_in_region(self):
        with patch(
            "sm_resilient_launcher._intelligent_router.boto3"
        ) as mock_boto3:
            mock_qf = MagicMock()
            mock_pf = MagicMock()
            mock_te = MagicMock()

            IntelligentRouter(
                quota_fetcher=mock_qf,
                pricing_fetcher=mock_pf,
                type_expander=mock_te,
                region="eu-west-1",
            )

            mock_boto3.client.assert_called_once_with(
                "bedrock-runtime", region_name="eu-west-1"
            )

    def test_stores_dependencies(self):
        with patch(
            "sm_resilient_launcher._intelligent_router.boto3"
        ):
            mock_qf = MagicMock()
            mock_pf = MagicMock()
            mock_te = MagicMock()

            r = IntelligentRouter(
                quota_fetcher=mock_qf,
                pricing_fetcher=mock_pf,
                type_expander=mock_te,
                region="us-west-2",
            )

            assert r._quota_fetcher is mock_qf
            assert r._pricing_fetcher is mock_pf
            assert r._type_expander is mock_te
            assert r._region == "us-west-2"
