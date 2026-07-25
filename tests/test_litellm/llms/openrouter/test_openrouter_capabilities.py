"""Tests for the live OpenRouter capability lookup.

The behaviour under test is a gate that decides whether a caller's reasoning
knob survives into the request body. Getting it wrong in one direction drops a
parameter the user asked for; getting it wrong in the other sends a parameter
the model does not accept. Both directions are covered here, as is the
fallback, which is the part that has to hold when the network does not.
"""

from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.llms.openrouter import capabilities
from litellm.llms.openrouter.chat.transformation import OpenrouterConfig

CAPABILITY_PAYLOAD = {
    "data": [
        {
            "id": "moonshotai/kimi-k3",
            "supported_parameters": [
                "reasoning",
                "reasoning_effort",
                "include_reasoning",
                "max_tokens",
            ],
        },
        {
            # Advertises the reasoning object but NOT reasoning_effort. This is
            # a real shape, not a contrived one, and it is the case the bundled
            # model-cost map cannot express.
            "id": "poolside/laguna-s-2.1",
            "supported_parameters": ["reasoning", "include_reasoning", "max_tokens"],
        },
        {
            "id": "openai/gpt-4o-mini",
            "supported_parameters": ["max_tokens", "temperature"],
        },
    ]
}


def _mock_response(payload, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


@pytest.fixture(autouse=True)
def _enable_lookup(monkeypatch):
    # conftest.py disables the lookup for every other test in this directory;
    # these tests are about the lookup itself, so turn it back on. The fetch is
    # still patched in each test, so nothing here touches the network.
    monkeypatch.setenv("LITELLM_OPENROUTER_CAPABILITY_FETCH", "1")
    capabilities.reset_cache()
    yield
    capabilities.reset_cache()


@pytest.fixture
def live_payload():
    with patch.object(
        capabilities,
        "_fetch",
        return_value={
            entry["id"]: set(entry["supported_parameters"])
            for entry in CAPABILITY_PAYLOAD["data"]
        },
    ) as mock_fetch:
        yield mock_fetch


class TestCapabilityLookup:
    def test_returns_advertised_parameters(self, live_payload):
        assert "reasoning_effort" in capabilities.get_supported_parameters(
            "moonshotai/kimi-k3"
        )

    def test_unknown_slug_returns_none(self, live_payload):
        # None means "no opinion", which is what makes the caller fall back
        # rather than treat an unlisted model as supporting nothing.
        assert capabilities.get_supported_parameters("vendor/never-published") is None

    def test_provider_prefixed_slug_resolves(self, live_payload):
        # litellm_params.model is commonly spelled with the provider prefix.
        assert capabilities.get_supported_parameters(
            "openrouter/moonshotai/kimi-k3"
        ) == capabilities.get_supported_parameters("moonshotai/kimi-k3")

    def test_variant_suffix_falls_back_to_base_slug(self, live_payload):
        assert "reasoning_effort" in capabilities.get_supported_parameters(
            "moonshotai/kimi-k3:nitro"
        )

    def test_fetches_once_across_repeated_lookups(self, live_payload):
        for _ in range(5):
            capabilities.get_supported_parameters("moonshotai/kimi-k3")
        assert live_payload.call_count == 1

    def test_opt_out_skips_the_lookup_entirely(self, live_payload, monkeypatch):
        monkeypatch.setenv("LITELLM_OPENROUTER_CAPABILITY_FETCH", "0")
        assert capabilities.get_supported_parameters("moonshotai/kimi-k3") is None
        assert live_payload.call_count == 0


class TestFetchFailsSoft:
    """A capability lookup must never be able to break a request."""

    def test_non_200_yields_no_opinion(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.get",
            return_value=_mock_response({}, status_code=503),
        ):
            assert capabilities.get_supported_parameters("moonshotai/kimi-k3") is None

    def test_transport_error_yields_no_opinion(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.get",
            side_effect=Exception("connection refused"),
        ):
            assert capabilities.get_supported_parameters("moonshotai/kimi-k3") is None

    def test_malformed_payload_yields_no_opinion(self):
        with patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.get",
            return_value=_mock_response({"unexpected": "shape"}),
        ):
            assert capabilities.get_supported_parameters("moonshotai/kimi-k3") is None

    def test_a_failed_fetch_is_not_retried_every_call(self):
        # An offline deployment must not pay a network timeout per request.
        with patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.get",
            side_effect=Exception("connection refused"),
        ) as mock_get:
            for _ in range(5):
                capabilities.get_supported_parameters("moonshotai/kimi-k3")
        assert mock_get.call_count == 1

    def test_ttl_expiry_allows_a_later_refresh(self, monkeypatch):
        monkeypatch.setenv("LITELLM_OPENROUTER_CAPABILITY_TTL", "0.0001")
        with patch(
            "litellm.llms.custom_httpx.http_handler.HTTPHandler.get",
            return_value=_mock_response(CAPABILITY_PAYLOAD),
        ) as mock_get:
            capabilities.get_supported_parameters("moonshotai/kimi-k3")
            import time

            time.sleep(0.001)
            capabilities.get_supported_parameters("moonshotai/kimi-k3")
        assert mock_get.call_count == 2


class TestSupportedParamsGate:
    """The behaviour operators actually see."""

    def test_live_data_admits_a_model_the_cost_map_does_not_carry(self, live_payload):
        # The whole point: kimi-k3 is absent from the bundled map, so the old
        # gate answered False and the knob was dropped.
        assert not litellm.supports_reasoning(
            model="moonshotai/kimi-k3", custom_llm_provider="openrouter"
        )
        params = OpenrouterConfig().get_supported_openai_params("moonshotai/kimi-k3")
        assert "reasoning_effort" in params
        assert "thinking" in params

    def test_a_model_advertising_only_the_reasoning_object_gets_thinking(
        self, live_payload
    ):
        # laguna advertises `reasoning` but not `reasoning_effort`, and is
        # absent from the bundled map, so neither source admits the effort
        # knob. `thinking` still comes through on the strength of `reasoning`.
        params = OpenrouterConfig().get_supported_openai_params(
            "poolside/laguna-s-2.1"
        )
        assert "reasoning_effort" not in params
        assert "thinking" in params

    def test_live_data_never_withholds_what_the_cost_map_allows(self, live_payload):
        # deepseek-r1 is flagged supports_reasoning in the bundled map but
        # OpenRouter advertises only `reasoning` for it, because
        # `reasoning_effort` is an alias rather than a listed parameter.
        # Reading that absence as a denial would drop a working knob, so the
        # two sources are unioned rather than the live one winning outright.
        with patch.object(
            capabilities,
            "_fetch",
            return_value={"deepseek/deepseek-r1": {"reasoning", "include_reasoning"}},
        ):
            params = OpenrouterConfig().get_supported_openai_params(
                "deepseek/deepseek-r1"
            )
        assert "reasoning_effort" in params
        assert "thinking" in params

    def test_non_reasoning_model_gets_neither_knob(self, live_payload):
        params = OpenrouterConfig().get_supported_openai_params("openai/gpt-4o-mini")
        assert "reasoning_effort" not in params
        assert "thinking" not in params

    def test_base_openai_params_are_untouched(self, live_payload):
        params = OpenrouterConfig().get_supported_openai_params("moonshotai/kimi-k3")
        for expected in ("temperature", "top_p", "max_tokens", "tools"):
            assert expected in params

    def test_falls_back_to_the_cost_map_when_live_data_is_unavailable(self):
        # deepseek-r1 is flagged in the bundled map, so the previous behaviour
        # must still admit it when the network is not there.
        with patch.object(capabilities, "_fetch", return_value={}):
            params = OpenrouterConfig().get_supported_openai_params(
                "deepseek/deepseek-r1"
            )
        assert "reasoning_effort" in params

    def test_fallback_matches_previous_behaviour_for_an_unknown_slug(self):
        with patch.object(capabilities, "_fetch", return_value={}):
            params = OpenrouterConfig().get_supported_openai_params(
                "vendor/never-published"
            )
        assert "reasoning_effort" not in params
