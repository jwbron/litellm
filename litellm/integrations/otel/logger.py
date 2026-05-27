"""Phase 2 / Phase 3 ``CustomLogger`` adapter on top of the new OTel package.

This is the V2 of ``litellm.integrations.opentelemetry.OpenTelemetry``. Behind
the off-by-default ``LITELLM_OTEL_V2`` flag, the factory in
``litellm_core_utils/litellm_logging.py`` constructs ``OpenTelemetryV2`` instead
of the legacy 3,227-line god-class. It implements only the callback surfaces
the existing call sites use:

- LLM-call sync/async success + failure -> ``emit(LLM_CALL, ...)``
- Service success/failure hooks         -> ``emit(SERVICE, ...)``
- Proxy SERVER span lifecycle           -> ``SpanEmitter._start(PROXY_REQUEST, ...)``
  plus route / status_code / preprocessing_duration setters
- First-registered-wins claim of ``proxy_server.open_telemetry_logger``

Out of scope (RFC Phase 2 closure / Phase 4 work, separate PRs):
- ``raw_gen_ai_request`` and "Failed Proxy Server Request" legacy child spans
- OTel logs emission (``_emit_semantic_logs``)
- Tool-call serialization (``set_tools_attributes``)
- Per-request dynamic tracer headers (Langfuse multi-tenancy)
- ``async_post_call_*`` / ``async_management_endpoint_*`` hooks
- The six subclass override surfaces (Arize, Phoenix, Langfuse-OTEL, ...)
"""

from datetime import datetime
from typing import Any, Mapping, Optional, Union, cast

from opentelemetry import trace
from opentelemetry.context import Context, get_current
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Span, Tracer

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.otel.config import OpenTelemetryV2Config
from litellm.integrations.otel.context import (
    context_from_span,
    extract_traceparent,
    set_request_baggage,
)
from litellm.integrations.otel.emitter import SpanEmitter
from litellm.integrations.otel.payloads import (
    LLMCallSpanData,
    ServiceSpanData,
    SpanError,
    promoted_baggage,
)
from litellm.integrations.otel.providers import build_tracer_provider, get_tracer
from litellm.integrations.otel.semconv import HTTP, LiteLLM
from litellm.integrations.otel.spans import SpanRole

# Legacy span name kept for parity. RFC §4.2 leaves the HTTP-semconv rename
# (``"{method} {route}"``) for Phase 5/6 — the proxy span is started before the
# route is known, so renaming requires either a placeholder + ``update_name`` or
# a deferred-start design. Out of scope for Phase 2.
LITELLM_PROXY_REQUEST_SPAN_NAME = "Received Proxy Server Request"
LITELLM_TRACER_NAME = "litellm"


# --------------------------------------------------------------------------- #
#  Small typed helpers
# --------------------------------------------------------------------------- #


def _to_ns(value: Optional[Union[datetime, float, int]]) -> Optional[int]:
    """Convert ``datetime`` / ``float`` (epoch seconds) to nanoseconds; None passthrough."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return int(value.timestamp() * 1e9)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(float(value) * 1e9)
    return None


def _to_seconds(value: Optional[Union[datetime, float, int, str]]) -> Optional[float]:
    """Convert datetime / float / common timestamp strings to epoch seconds."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(value, fmt).timestamp()
            except ValueError:
                continue
    return None


def _is_recordable_span(obj: object) -> bool:
    """Whether ``obj`` looks like a live OTel Span we can derive context from."""
    if obj is None:
        return False
    if not isinstance(obj, trace.Span):
        return False
    try:
        ctx = obj.get_span_context()
    except Exception:
        return False
    return ctx is not None and ctx.is_valid


# --------------------------------------------------------------------------- #
#  The V2 logger
# --------------------------------------------------------------------------- #


class OpenTelemetryV2(CustomLogger):
    """V2 of ``OpenTelemetry`` — the same callback surface, the new engine.

    Constructor mirrors V1's positional shape so the factory and tests don't
    care which version they're holding.
    """

    def __init__(
        self,
        config: Optional[OpenTelemetryV2Config] = None,
        callback_name: Optional[str] = None,
        tracer_provider: Optional[TracerProvider] = None,
        logger_provider: Optional[Any] = None,  # reserved (OTel logs) — Phase 2 closure
        meter_provider: Optional[Any] = None,  # reserved (metrics) — Phase 2 closure
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.config: OpenTelemetryV2Config = config or OpenTelemetryV2Config()
        self.callback_name = callback_name
        self._tracer_provider: TracerProvider = (
            tracer_provider
            if tracer_provider is not None
            else build_tracer_provider(self.config)
        )
        self.tracer: Tracer = get_tracer(self._tracer_provider, LITELLM_TRACER_NAME)
        self._emitter = SpanEmitter(self.tracer, self.config)
        self._init_otel_logger_on_litellm_proxy()

    # -- proxy global registration ----------------------------------------- #

    def _init_otel_logger_on_litellm_proxy(self) -> None:
        """Claim ``proxy_server.open_telemetry_logger`` if no one else has.

        Mirrors V1's first-registered-wins guard so the V1 and V2 classes can
        coexist during the dual-emit / phased-migration window without fighting
        over the global.
        """
        try:
            from litellm.proxy import proxy_server
        except Exception:
            return
        # Add to litellm.service_callback unless an OTel instance is already there.
        try:
            existing = getattr(litellm, "service_callback", None) or []
            already_otel = any(
                cb.__class__.__module__.startswith("litellm.integrations.otel")
                or cb.__class__.__module__.startswith(
                    "litellm.integrations.opentelemetry"
                )
                for cb in existing
                if hasattr(cb, "__class__")
            )
            if not already_otel:
                existing.append(self)
        except Exception:
            pass
        if getattr(proxy_server, "open_telemetry_logger", None) is None:
            setattr(proxy_server, "open_telemetry_logger", self)

    # -- LLM-call callbacks ------------------------------------------------- #

    def log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ):  # noqa: D401
        self._emit_llm_call(kwargs, start_time, end_time)

    def log_failure_event(
        self, kwargs, response_obj, start_time, end_time
    ):  # noqa: D401
        self._emit_llm_call(kwargs, start_time, end_time)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._emit_llm_call(kwargs, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._emit_llm_call(kwargs, start_time, end_time)

    def _emit_llm_call(
        self,
        kwargs: Mapping[str, Any],
        start_time: Optional[Union[datetime, float]],
        end_time: Optional[Union[datetime, float]],
    ) -> Optional[Span]:
        payload = kwargs.get("standard_logging_object")
        if not payload:
            return None
        data = LLMCallSpanData.from_standard_logging_payload(
            cast("Any", payload)  # StandardLoggingPayload TypedDict
        )
        parent_ctx = self._resolve_parent_context(kwargs)
        # Promote identity to Baggage so child spans (guardrails, services
        # emitted in the same request scope) inherit the team/key/model triple.
        bag = promoted_baggage(
            data.identity,
            data.request_model,
            promoted_keys=tuple(self.config.baggage_promoted_keys),
            metadata_keys=tuple(self.config.baggage_metadata_keys),
        )
        if bag:
            parent_ctx = set_request_baggage(bag, context=parent_ctx)
        return self._emitter.emit(
            SpanRole.LLM_CALL,
            data,
            parent_context=parent_ctx,
            start_time_ns=_to_ns(start_time),
            end_time_ns=_to_ns(end_time),
        )

    # -- parent context resolution ---------------------------------------- #

    def _resolve_parent_context(self, kwargs: Mapping[str, Any]) -> Optional[Context]:
        """Replicates V1's 4-priority ``_get_span_context`` resolution.

        Priority order matches ``opentelemetry.py:2600``: explicit parent span
        in metadata, traceparent header on the proxy request, current global
        span context, then no parent (root).
        """
        litellm_params = kwargs.get("litellm_params") or {}
        if isinstance(litellm_params, dict):
            metadata = litellm_params.get("metadata") or {}
            parent_span = (
                metadata.get("litellm_parent_otel_span")
                if isinstance(metadata, dict)
                else None
            )
            if _is_recordable_span(parent_span):
                return context_from_span(cast(Span, parent_span))
            proxy_request = litellm_params.get("proxy_server_request") or {}
            if isinstance(proxy_request, dict):
                headers = proxy_request.get("headers")
                if (
                    isinstance(headers, Mapping)
                    and not self.config.ignore_context_propagation
                ):
                    ctx = extract_traceparent(headers)
                    if ctx is not None:
                        return ctx
        # Fall back to the current global context (may itself be the root span
        # set by ``create_litellm_proxy_request_started_span`` in the proxy).
        return get_current()

    # -- service hooks (Phase 3) ------------------------------------------- #

    async def async_service_success_hook(
        self,
        payload: Any,  # ServiceLoggerPayload
        parent_otel_span: Optional[Span] = None,
        start_time: Optional[Union[datetime, float]] = None,
        end_time: Optional[Union[datetime, float]] = None,
        event_metadata: Optional[dict] = None,
    ) -> None:
        self._emit_service(
            payload,
            parent_otel_span=parent_otel_span,
            start_time=start_time,
            end_time=end_time,
            event_metadata=event_metadata,
            error_override=None,
        )

    async def async_service_failure_hook(
        self,
        payload: Any,  # ServiceLoggerPayload
        error: Optional[str] = "",
        parent_otel_span: Optional[Span] = None,
        start_time: Optional[Union[datetime, float]] = None,
        end_time: Optional[Union[datetime, float]] = None,
        event_metadata: Optional[dict] = None,
    ) -> None:
        self._emit_service(
            payload,
            parent_otel_span=parent_otel_span,
            start_time=start_time,
            end_time=end_time,
            event_metadata=event_metadata,
            error_override=error or "error",
        )

    def _emit_service(
        self,
        payload: Any,
        *,
        parent_otel_span: Optional[Span],
        start_time: Optional[Union[datetime, float]],
        end_time: Optional[Union[datetime, float]],
        event_metadata: Optional[dict],
        error_override: Optional[str],
    ) -> Optional[Span]:
        # V1 only emits a service span when a parent OTel span exists — there
        # are no free-standing service roots. Preserve that contract.
        if not _is_recordable_span(parent_otel_span):
            return None
        data = ServiceSpanData.from_payload(payload, event_metadata=event_metadata)
        if error_override is not None and data.error is None:
            data = ServiceSpanData(
                service_name=data.service_name,
                call_type=data.call_type,
                error=SpanError(message=error_override),
                event_metadata=data.event_metadata,
            )
        return self._emitter.emit(
            SpanRole.SERVICE,
            data,
            parent_context=context_from_span(cast(Span, parent_otel_span)),
            start_time_ns=_to_ns(start_time),
            end_time_ns=_to_ns(end_time),
        )

    # -- proxy SERVER-span lifecycle --------------------------------------- #

    def create_litellm_proxy_request_started_span(
        self, start_time: datetime, headers: Optional[Mapping[str, str]]
    ) -> Optional[Span]:
        """Start (but don't end) the proxy SERVER root span.

        The span is held by the caller (auth/middleware) and ended when the
        response is sent. We only resolve the parent context (a remote
        traceparent if propagation is enabled) and hand back the live span.
        """
        parent_ctx: Optional[Context] = None
        if (
            headers is not None
            and isinstance(headers, Mapping)
            and not self.config.ignore_context_propagation
        ):
            parent_ctx = extract_traceparent(headers)
        return self._emitter._start(
            SpanRole.PROXY_REQUEST,
            LITELLM_PROXY_REQUEST_SPAN_NAME,
            parent_context=parent_ctx,
            start_time_ns=_to_ns(start_time),
        )

    @staticmethod
    def set_proxy_request_route_attributes(
        span: Optional[Span],
        *,
        url_path: Optional[str] = None,
        http_route: Optional[str] = None,
    ) -> None:
        if span is None:
            return
        if url_path is not None:
            span.set_attribute(HTTP.URL_PATH, url_path)
        if http_route is not None:
            span.set_attribute(HTTP.ROUTE, http_route)

    @staticmethod
    def set_response_status_code_attribute(
        span: Optional[Span], status_code: Optional[int]
    ) -> None:
        if span is None or status_code is None:
            return
        span.set_attribute(HTTP.RESPONSE_STATUS_CODE, int(status_code))

    @staticmethod
    def set_preprocessing_duration_attribute(
        span: Optional[Span], container: Any
    ) -> None:
        """Stamp ``litellm.preprocessing.duration_ms`` on the proxy span.

        ``litellm_received_at`` rides request metadata; ``first_api_call_start_time``
        is the set-once first-handoff instant. No-op if either anchor or the
        span is missing. Ported from V1 ``opentelemetry.py:3186``.
        """
        if span is None or not isinstance(container, dict):
            return
        first_handoff = container.get("first_api_call_start_time")
        received_at = None
        litellm_params = container.get("litellm_params")
        candidates = (
            (
                litellm_params.get("metadata")
                if isinstance(litellm_params, dict)
                else None
            ),
            container.get("metadata"),
            container.get("litellm_metadata"),
        )
        for source in candidates:
            if isinstance(source, dict):
                received_at = received_at or source.get("litellm_received_at")
        if received_at is None or first_handoff is None:
            return
        start_ts = _to_seconds(received_at)
        end_ts = _to_seconds(first_handoff)
        if start_ts is None or end_ts is None:
            return
        duration_ms = (end_ts - start_ts) * 1000.0
        if duration_ms < 0:
            return  # clock skew — emit nothing
        span.set_attribute(LiteLLM.PREPROCESSING_MS, duration_ms)
