from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from opentelemetry.context import Context
from opentelemetry.trace import Span, Tracer
from opentelemetry.trace.status import Status, StatusCode

from litellm.integrations.otel.config import OpenTelemetryV2Config
from litellm.integrations.otel.mappers.base import AttributeMapper, SpanData
from litellm.integrations.otel.mappers.genai import GenAIMapper
from litellm.integrations.otel.mappers.legacy import LegacyMapper
from litellm.integrations.otel.payloads import LLMCallSpanData, ServiceSpanData
from litellm.integrations.otel.providers import to_otel_span_kind
from litellm.integrations.otel.semconv import Error
from litellm.integrations.otel.spans import (
    SPAN_REGISTRY,
    SpanRole,
    guardrail_span_name,
    llm_call_span_name,
    service_span_name,
)


def default_mappers(config: OpenTelemetryV2Config) -> List[AttributeMapper]:
    """The canonical mapper chain: GenAI always; Legacy during the dual-emit window."""
    mappers: List[AttributeMapper] = [GenAIMapper()]
    if config.legacy_compat:
        mappers.append(LegacyMapper())
    return mappers


# Roles emit() knows how to emit. PROXY_REQUEST / MANAGEMENT are caller-owned roots.
_NAME_BUILDERS: Dict[SpanRole, Callable[..., str]] = {
    SpanRole.LLM_CALL: llm_call_span_name,
    SpanRole.GUARDRAIL: guardrail_span_name,
    SpanRole.SERVICE: service_span_name,
}


class SpanEmitter:
    def __init__(
        self,
        tracer: Tracer,
        config: OpenTelemetryV2Config,
        mappers: Optional[Sequence[AttributeMapper]] = None,
    ) -> None:
        self._tracer = tracer
        self._config = config
        self._mappers: List[AttributeMapper] = (
            list(mappers) if mappers is not None else default_mappers(config)
        )
        self._emitted: Set[Tuple[str, SpanRole]] = set()

    # -- low-level helpers --------------------------------------------------- #

    def _start(
        self,
        role: SpanRole,
        name: str,
        parent_context: Optional[Context] = None,
        start_time_ns: Optional[int] = None,
    ) -> Span:
        return self._tracer.start_span(
            name,
            context=parent_context,
            kind=to_otel_span_kind(SPAN_REGISTRY[role].kind),
            start_time=start_time_ns,
        )

    def _seen(self, dedup_key: Optional[str], role: SpanRole) -> bool:
        """Idempotency guard for the streaming sync+async dual-fire."""
        if not dedup_key:
            return False
        marker = (dedup_key, role)
        if marker in self._emitted:
            return True
        self._emitted.add(marker)
        return False

    # -- the engine ---------------------------------------------------------- #

    def emit(
        self,
        role: SpanRole,
        data: SpanData,
        parent_context: Optional[Context] = None,
        *,
        start_time_ns: Optional[int] = None,
        end_time_ns: Optional[int] = None,
    ) -> Optional[Span]:
        """Dedup → start → mapper chain → status → end. ``None`` only when deduped."""
        # Only LLM-call spans carry a dedup key; only LLM-call and service spans
        # carry an ``error`` field. ``isinstance`` gives mypy real narrowing and
        # keeps the engine free of duck-typed attribute reads.
        dedup_key = data.identity.call_id if isinstance(data, LLMCallSpanData) else None
        if self._seen(dedup_key, role):
            return None
        span = self._start(
            role,
            _NAME_BUILDERS[role](data),
            parent_context=parent_context,
            start_time_ns=start_time_ns,
        )
        for mapper in self._mappers:
            for key, value in mapper.map(role, data).items():
                span.set_attribute(key, value)
        error = (
            data.error if isinstance(data, (LLMCallSpanData, ServiceSpanData)) else None
        )
        if error and (error.error_type or error.message):
            span.set_attribute(Error.TYPE, error.error_type or "error")
            span.set_status(
                Status(StatusCode.ERROR, error.message or error.error_type or "error")
            )
        else:
            span.set_status(Status(StatusCode.OK))
        span.end(end_time=end_time_ns)
        return span
