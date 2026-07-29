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
        {
            # Prices by prompt length. OpenRouter really does publish qwen3-max
            # this way; the boundaries are arbitrary and the model-cost map
            # cannot express them, which is why this entry stays unpriced.
            "id": "qwen/qwen3-max",
            "supported_parameters": ["max_tokens"],
            "pricing": {
                "prompt": "0.00000078",
                "completion": "0.0000039",
                "overrides": [
                    {"min_prompt_tokens": 32000, "prompt": "0.00000156"},
                    {"min_prompt_tokens": 128000, "prompt": "0.00000195"},
                ],
            },
        },
    ]
}

# Rates are published as decimal STRINGS in USD per token, and a model may omit
# any of the cache rates. Attached to the two slugs above that the pricing tests
# read; the rest stay price-free on purpose, so "advertises params but not
# pricing" is covered by the same payload.
CAPABILITY_PAYLOAD["data"][0]["pricing"] = {
    "prompt": "0.0000006",
    "completion": "0.0000025",
    "input_cache_read": "0.00000015",
}
CAPABILITY_PAYLOAD["data"][1]["pricing"] = {
    "prompt": "0.0000001",
    "completion": "0.0000002",
    "input_cache_read": "0.00000001",
    "input_cache_write": "0.0000005",
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
    """Serve CAPABILITY_PAYLOAD over a stubbed HTTP handler.

    Patched at the transport rather than at ``_fetch`` so the parsing — which
    is where the pricing translation lives — is under test rather than mocked
    past. The returned mock records one call per constructed handler, so a test
    can still assert how many fetches actually happened.
    """
    handler = MagicMock()
    handler.return_value.get.return_value = _mock_response(CAPABILITY_PAYLOAD)
    with patch(
        "litellm.llms.custom_httpx.http_handler.HTTPHandler",
        handler,
    ):
        yield handler


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
            return_value={
                "deepseek/deepseek-r1": {
                    "id": "deepseek/deepseek-r1",
                    "parameters": {"reasoning", "include_reasoning"},
                    "cost_entry": None,
                    "declined_thresholds": None,
                }
            },
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


class TestPricingLookup:
    """The rate-card half of the same fetch.

    The bundled model-cost map does not carry current OpenRouter slugs, so
    ``get_model_info`` raises for them and no cost is ever computed. These
    assert the live card fills that gap without ever becoming a source of
    anything other than cost.
    """

    def test_translates_the_published_rate_card(self, live_payload):
        entry = capabilities.get_model_cost_entry("openrouter/poolside/laguna-s-2.1")
        assert entry == {
            "key": "poolside/laguna-s-2.1",
            "litellm_provider": "openrouter",
            "mode": "chat",
            "input_cost_per_token": 1e-07,
            "output_cost_per_token": 2e-07,
            "cache_read_input_token_cost": 1e-08,
            # OpenRouter's `input_cache_write` is this map's `cache_creation_*`.
            # Swapping the pair prices cache writes at the read rate — a ~5x
            # understatement of the most expensive turn in a session.
            "cache_creation_input_token_cost": 5e-07,
        }

    def test_carries_cost_fields_only(self, live_payload):
        # A supports_* flag arriving through this door would change parameter
        # admission, which is get_supported_parameters' job and is gated far
        # more carefully (see the `thinking` note in the module docstring).
        entry = capabilities.get_model_cost_entry("moonshotai/kimi-k3")
        assert not [k for k in entry if k.startswith("supports_")]
        assert not [k for k in entry if "tokens" in k], "no context lengths either"

    def test_omits_a_rate_the_provider_did_not_publish(self, live_payload):
        entry = capabilities.get_model_cost_entry("moonshotai/kimi-k3")
        assert "cache_read_input_token_cost" in entry
        assert "cache_creation_input_token_cost" not in entry

    def test_a_tiered_rate_card_is_declined_rather_than_under_reported(
        self, live_payload
    ):
        # qwen3-max charges 2x above 32000 prompt tokens and 2.5x above 128000.
        # The map has slots for three fixed thresholds and drops the rest, so
        # registering the base tier would under-report long prompts silently,
        # under a field an operator would use to compare models.
        assert capabilities.get_model_cost_entry("qwen/qwen3-max") is None
        # ...and the parameter half of the same entry is unaffected.
        assert capabilities.get_supported_parameters("qwen/qwen3-max") == {"max_tokens"}

    def test_answers_for_openrouter_only(self, live_payload):
        # The call site is a generic model-info lookup every provider reaches.
        assert capabilities.get_model_cost_entry("poolside/laguna-s-2.1", "bedrock") is None
        assert (
            capabilities.get_model_cost_entry("poolside/laguna-s-2.1", "openrouter")
            is not None
        )
        # None means the caller could not attribute it, which a bare slug is.
        assert capabilities.get_model_cost_entry("poolside/laguna-s-2.1", None) is not None

    def test_unknown_slug_has_no_price_opinion(self, live_payload):
        assert capabilities.get_model_cost_entry("vendor/never-published") is None

    def test_an_entry_without_pricing_is_still_a_parameter_answer(self, live_payload):
        # The two halves are independent; one missing is not the other's problem.
        assert capabilities.get_supported_parameters("openai/gpt-4o-mini") is not None
        assert capabilities.get_model_cost_entry("openai/gpt-4o-mini") is None

    def test_returned_entry_is_a_copy(self, live_payload):
        # The cache is process-wide and get_model_info mutates what it is handed.
        first = capabilities.get_model_cost_entry("poolside/laguna-s-2.1")
        first["input_cost_per_token"] = 999.0
        assert (
            capabilities.get_model_cost_entry("poolside/laguna-s-2.1")[
                "input_cost_per_token"
            ]
            == 1e-07
        )

    def test_pricing_can_be_disabled_without_disabling_capabilities(
        self, live_payload, monkeypatch
    ):
        monkeypatch.setenv("LITELLM_OPENROUTER_PRICING", "0")
        assert capabilities.get_model_cost_entry("poolside/laguna-s-2.1") is None
        assert capabilities.get_supported_parameters("poolside/laguna-s-2.1") is not None

    def test_the_master_switch_disables_pricing_too(self, live_payload, monkeypatch):
        monkeypatch.setenv("LITELLM_OPENROUTER_CAPABILITY_FETCH", "0")
        assert capabilities.get_model_cost_entry("poolside/laguna-s-2.1") is None
        assert live_payload.call_count == 0, "no network call when the lookup is off"

    @pytest.mark.parametrize(
        "pricing",
        [
            {"completion": "0.000002"},
            {"prompt": "0.000001"},
            {"prompt": "not-a-number", "completion": "0.000002"},
            {"prompt": "-0.000001", "completion": "0.000002"},
            {"prompt": "Infinity", "completion": "0.000002"},
            "not-a-dict",
        ],
    )
    def test_an_unusable_rate_card_is_no_opinion(self, pricing):
        # Without both rates the entry cannot price a chat turn; a negative or
        # non-finite rate is not a rate, and an `inf` would serialize as a
        # non-standard JSON token downstream.
        handler = MagicMock()
        handler.return_value.get.return_value = _mock_response(
            {"data": [{"id": "some/model", "pricing": pricing}]}
        )
        with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler", handler):
            assert capabilities.get_model_cost_entry("some/model") is None

    def test_zero_is_a_real_rate(self):
        # A `:free` variant really is priced at zero. Deciding what to do with
        # a zero estimate belongs to whoever reads it.
        handler = MagicMock()
        handler.return_value.get.return_value = _mock_response(
            {"data": [{"id": "a/b:free", "pricing": {"prompt": "0", "completion": "0"}}]}
        )
        with patch("litellm.llms.custom_httpx.http_handler.HTTPHandler", handler):
            entry = capabilities.get_model_cost_entry("a/b:free")
        assert entry["input_cost_per_token"] == 0.0
        assert entry["output_cost_per_token"] == 0.0


class TestModelInfoIntegration:
    """The hook in ``_get_model_info_helper``, which is the point of all this."""

    @pytest.fixture(autouse=True)
    def _clear_model_info_cache(self):
        # get_model_info memoizes, so without this each test here would be
        # asserting against whatever a previous one happened to warm — and the
        # order would decide which of them actually exercised the hook.
        from litellm.utils import _cached_get_model_info, _cached_get_model_info_helper

        for cache in (_cached_get_model_info, _cached_get_model_info_helper):
            cache.cache_clear()
        yield
        for cache in (_cached_get_model_info, _cached_get_model_info_helper):
            cache.cache_clear()

    def test_an_unmapped_slug_is_priced_from_the_live_card(self, live_payload):
        info = litellm.get_model_info(model="openrouter/poolside/laguna-s-2.1")
        assert info["input_cost_per_token"] == 1e-07
        assert info["output_cost_per_token"] == 2e-07
        assert info["cache_read_input_token_cost"] == 1e-08
        assert info["litellm_provider"] == "openrouter"

    def test_cost_is_computed_end_to_end(self, live_payload):
        # 10k uncached input + 90k cached + 2k output, against the rates above.
        prompt, cached, completion = 100_000, 90_000, 2_000
        expected = (prompt - cached) * 1e-07 + cached * 1e-08 + completion * 2e-07
        usage = litellm.Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            prompt_tokens_details={"cached_tokens": cached},
        )
        prompt_cost, completion_cost = litellm.cost_per_token(
            model="openrouter/poolside/laguna-s-2.1",
            custom_llm_provider="openrouter",
            usage_object=usage,
        )
        assert prompt_cost + completion_cost == pytest.approx(expected)

    def test_a_declined_tiered_model_still_raises(self, live_payload):
        # Better an explicit "not mapped" than a number that is 2.5x low on the
        # prompt lengths this model is actually used at.
        with pytest.raises(Exception, match="isn't mapped yet"):
            litellm.get_model_info(model="openrouter/qwen/qwen3-max")

    def test_an_unknown_slug_still_raises(self, live_payload):
        with pytest.raises(Exception, match="isn't mapped yet"):
            litellm.get_model_info(model="openrouter/vendor/never-published")

    def test_a_mapped_slug_keeps_the_bundled_rate(self, live_payload):
        # The hook sits after every map lookup, so the live card can add a
        # model but never reprice one.
        info = litellm.get_model_info(model="gpt-4o-mini")
        assert info["litellm_provider"] == "openai"
