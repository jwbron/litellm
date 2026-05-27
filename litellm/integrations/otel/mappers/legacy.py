"""Dual-emit mapper: deprecated attribute keys kept for backward compatibility."""

from typing import Final

from litellm.integrations.otel.mappers.base import (
    AttributeMap,
    SpanData,
    drop_none,
)
from litellm.integrations.otel.payloads import LLMCallSpanData, ServiceSpanData
from litellm.integrations.otel.spans import SpanRole

# Deprecated keys (semconv-ai / Traceloop era).
_LEGACY_SYSTEM: Final = "gen_ai.system"
_LEGACY_PROMPT_TOKENS: Final = "gen_ai.usage.prompt_tokens"
_LEGACY_COMPLETION_TOKENS: Final = "gen_ai.usage.completion_tokens"
_LEGACY_TOTAL_TOKENS: Final = "gen_ai.usage.total_tokens"
_LEGACY_IS_STREAMING: Final = "llm.is_streaming"
_LEGACY_TOP_K: Final = "llm.top_k"
_LEGACY_FREQUENCY_PENALTY: Final = "llm.frequency_penalty"
_LEGACY_PRESENCE_PENALTY: Final = "llm.presence_penalty"
_LEGACY_STOP_SEQUENCES: Final = "llm.chat.stop_sequences"
# Service-span bare keys used by V1 (no namespace). Dual-emitted for any
# dashboard that filters on them today.
_LEGACY_SERVICE: Final = "service"
_LEGACY_CALL_TYPE: Final = "call_type"
_LEGACY_ERROR: Final = "error"


class LegacyMapper:
    """Re-emits values under their deprecated key names (LLM-call + service)."""

    def map(self, role: SpanRole, data: SpanData) -> AttributeMap:
        if isinstance(data, LLMCallSpanData):
            return self._llm_call(data)
        if isinstance(data, ServiceSpanData):
            return self._service(data)
        return {}

    @staticmethod
    def _llm_call(data: LLMCallSpanData) -> AttributeMap:
        rp, u = data.request_params, data.usage
        stop = list(rp.stop_sequences) if rp.stop_sequences else None
        return drop_none(
            {
                _LEGACY_SYSTEM: data.provider or None,
                _LEGACY_PROMPT_TOKENS: u.input_tokens,
                _LEGACY_COMPLETION_TOKENS: u.output_tokens,
                _LEGACY_TOTAL_TOKENS: u.total_tokens,
                _LEGACY_IS_STREAMING: data.is_streaming,
                _LEGACY_TOP_K: rp.top_k,
                _LEGACY_FREQUENCY_PENALTY: rp.frequency_penalty,
                _LEGACY_PRESENCE_PENALTY: rp.presence_penalty,
                _LEGACY_STOP_SEQUENCES: stop,
            }
        )

    @staticmethod
    def _service(data: ServiceSpanData) -> AttributeMap:
        # V1 stamps these as bare keys (no namespace) on the service span.
        attrs: AttributeMap = {_LEGACY_SERVICE: data.service_name}
        if data.call_type is not None:
            attrs[_LEGACY_CALL_TYPE] = data.call_type
        if data.error is not None and data.error.message:
            attrs[_LEGACY_ERROR] = data.error.message
        # V1 stamps event_metadata sub-keys bare (under the user-supplied names).
        for key, value in data.event_metadata.items():
            attrs[key] = value
        return attrs
