"""Dual-emit mapper: deprecated attribute keys kept for backward compatibility.

These keys are intentionally NOT in ``semconv.py`` — that module holds only the
canonical keys. This is the single place legacy/deprecated strings live, so they
can be deleted wholesale once the deprecation window closes.
"""

from typing import Final

from litellm.integrations.otel.mappers.base import AttributeMap, drop_none
from litellm.integrations.otel.payloads import LLMCallSpanData

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


class LegacyMapper:
    """Re-emits canonical values under their deprecated key names."""

    def map_llm_call(self, data: LLMCallSpanData) -> AttributeMap:
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
