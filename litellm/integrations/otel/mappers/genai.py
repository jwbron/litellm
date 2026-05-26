"""Canonical OpenTelemetry GenAI semantic-convention mapper (always active)."""

from litellm.integrations.otel.mappers.base import AttributeMap, drop_none
from litellm.integrations.otel.payloads import LLMCallSpanData
from litellm.integrations.otel.semconv import Error, GenAI, LiteLLM, Server


class GenAIMapper:
    """Emits ``gen_ai.*`` (and a few ``litellm.*`` vendor) attributes."""

    def map_llm_call(self, data: LLMCallSpanData) -> AttributeMap:
        rp, u, s, idn = data.request_params, data.usage, data.server, data.identity
        # Empty collections are pre-converted to ``None`` so ``drop_none`` skips
        # them — strict ``is not None`` keeps legitimate zero values intact.
        stop = list(rp.stop_sequences) if rp.stop_sequences else None
        finishes = list(data.finish_reasons) if data.finish_reasons else None
        return drop_none(
            {
                GenAI.OPERATION_NAME: data.operation.value,
                GenAI.PROVIDER_NAME: data.provider or None,
                GenAI.REQUEST_MODEL: data.request_model or None,
                GenAI.REQUEST_TEMPERATURE: rp.temperature,
                GenAI.REQUEST_TOP_P: rp.top_p,
                GenAI.REQUEST_TOP_K: rp.top_k,
                GenAI.REQUEST_MAX_TOKENS: rp.max_tokens,
                GenAI.REQUEST_FREQUENCY_PENALTY: rp.frequency_penalty,
                GenAI.REQUEST_PRESENCE_PENALTY: rp.presence_penalty,
                GenAI.REQUEST_STOP_SEQUENCES: stop,
                GenAI.REQUEST_SEED: rp.seed,
                GenAI.RESPONSE_MODEL: data.response_model,
                GenAI.RESPONSE_ID: data.response_id,
                GenAI.RESPONSE_FINISH_REASONS: finishes,
                GenAI.USAGE_INPUT_TOKENS: u.input_tokens,
                GenAI.USAGE_OUTPUT_TOKENS: u.output_tokens,
                Error.TYPE: data.error.error_type if data.error else None,
                Server.ADDRESS: s.address if s else None,
                Server.PORT: s.port if s else None,
                LiteLLM.CALL_ID: idn.call_id or None,
                f"{LiteLLM.COST_PREFIX}total": data.response_cost,
                LiteLLM.REQUEST_STREAMING: data.is_streaming,
            }
        )
