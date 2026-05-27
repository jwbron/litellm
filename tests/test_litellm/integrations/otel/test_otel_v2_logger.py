"""Phase 2 / Phase 3 tests for the V2 ``OpenTelemetryV2`` CustomLogger adapter.

Exercises the callback surface the existing call sites use: LLM-call sync/async
success + failure, service hooks, proxy SERVER span lifecycle (start + setters),
parent-context resolution (explicit span, traceparent header), and Baggage
promotion onto child spans.
"""

import asyncio
from datetime import datetime, timezone

import pytest

pytest.importorskip("opentelemetry")

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind  # noqa: E402
from opentelemetry.trace.status import StatusCode  # noqa: E402

from litellm.integrations.otel import (  # noqa: E402
    GenAI,
    HTTP,
    LiteLLM,
    OpenTelemetryV2Config,
)
from litellm.integrations.otel import providers  # noqa: E402
from litellm.integrations.otel.logger import (  # noqa: E402
    LITELLM_PROXY_REQUEST_SPAN_NAME,
    OpenTelemetryV2,
    _to_ns,
    _to_seconds,
)
from litellm.integrations.otel.spans import SpanRole  # noqa: E402

# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #


def _payload(**overrides):
    payload = {
        "call_type": "acompletion",
        "custom_llm_provider": "openai",
        "model": "gpt-4o",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "stream": False,
        "model_parameters": {"temperature": 0.7, "max_tokens": 256},
        "response": {
            "id": "resp_1",
            "model": "gpt-4o-2024",
            "choices": [{"finish_reason": "stop"}],
        },
        "metadata": {
            "team_id": "t1",
            "team_alias": "team one",
            "user_api_key_hash": "hsh",
        },
        "api_base": "https://api.openai.com:443/v1",
        "status": "success",
        "litellm_call_id": "call_1",
        "response_cost": 0.002,
        "hidden_params": {},
    }
    payload.update(overrides)
    return payload


def _kwargs(payload=None, parent_span=None, traceparent=None):
    payload = payload if payload is not None else _payload()
    metadata = {}
    if parent_span is not None:
        metadata["litellm_parent_otel_span"] = parent_span
    proxy_request = {}
    if traceparent is not None:
        proxy_request["headers"] = {"traceparent": traceparent}
    return {
        "standard_logging_object": payload,
        "litellm_params": {"metadata": metadata, "proxy_server_request": proxy_request},
    }


def _logger(legacy_compat=True):
    cfg = OpenTelemetryV2Config(exporter="in_memory", legacy_compat=legacy_compat)
    exporter = InMemorySpanExporter()
    tracer_provider = providers.build_tracer_provider(cfg, exporter=exporter)
    return OpenTelemetryV2(config=cfg, tracer_provider=tracer_provider), exporter


# --------------------------------------------------------------------------- #
#  Time helpers
# --------------------------------------------------------------------------- #


def test_to_ns_handles_datetime_and_float():
    dt = datetime(2026, 5, 26, 12, 0, 0, tzinfo=timezone.utc)
    assert _to_ns(dt) == int(dt.timestamp() * 1e9)
    assert _to_ns(1.5) == 1_500_000_000
    assert _to_ns(None) is None
    assert _to_ns(True) is None  # bool is rejected — not a real epoch value


def test_to_seconds_parses_string_formats():
    assert _to_seconds("2026-05-26 12:00:00.123") is not None
    assert _to_seconds("2026-05-26 12:00:00") is not None
    assert _to_seconds("nonsense") is None
    assert _to_seconds(None) is None
    assert _to_seconds(1.5) == 1.5


# --------------------------------------------------------------------------- #
#  LLM-call callbacks
# --------------------------------------------------------------------------- #


def test_log_success_event_emits_llm_call_span():
    logger, exporter = _logger()
    logger.log_success_event(_kwargs(), None, None, None)
    (span,) = exporter.get_finished_spans()
    assert span.name == "chat gpt-4o"
    assert span.kind is SpanKind.CLIENT
    assert span.attributes[GenAI.OPERATION_NAME] == "chat"
    assert span.attributes[GenAI.REQUEST_MODEL] == "gpt-4o"
    assert span.attributes[LiteLLM.CALL_ID] == "call_1"
    assert span.status.status_code is StatusCode.OK


def test_log_failure_event_marks_error_status():
    logger, exporter = _logger()
    payload = _payload(
        status="failure",
        error_information={"error_class": "RateLimitError", "error_message": "429"},
    )
    logger.log_failure_event(_kwargs(payload=payload), None, None, None)
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == "RateLimitError"


def test_async_log_success_event_dispatches():
    logger, exporter = _logger()
    asyncio.run(logger.async_log_success_event(_kwargs(), None, None, None))
    spans = exporter.get_finished_spans()
    assert len(spans) == 1


def test_missing_standard_logging_object_is_noop():
    logger, exporter = _logger()
    logger.log_success_event({"litellm_params": {}}, None, None, None)
    assert exporter.get_finished_spans() == ()


def test_idempotent_on_repeat_call_id():
    """Same StandardLoggingPayload (same id) emits once across sync+async dual-fire."""
    logger, exporter = _logger()
    kwargs = _kwargs()
    logger.log_success_event(kwargs, None, None, None)
    asyncio.run(logger.async_log_success_event(kwargs, None, None, None))
    spans = exporter.get_finished_spans()
    assert len(spans) == 1


# --------------------------------------------------------------------------- #
#  Parent context resolution
# --------------------------------------------------------------------------- #


def test_explicit_parent_span_in_metadata_is_used():
    logger, exporter = _logger()
    parent = logger._emitter._start(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    try:
        logger.log_success_event(_kwargs(parent_span=parent), None, None, None)
    finally:
        parent.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    llm_span = by_name["chat gpt-4o"]
    parent_span = by_name[LITELLM_PROXY_REQUEST_SPAN_NAME]
    assert llm_span.parent is not None
    assert llm_span.parent.span_id == parent_span.get_span_context().span_id


def test_traceparent_header_resolves_remote_parent():
    """When no explicit parent span exists, a W3C traceparent header is honored."""
    logger, exporter = _logger()
    # Valid traceparent: version 00, trace_id, span_id, flags
    tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    logger.log_success_event(_kwargs(traceparent=tp), None, None, None)
    (span,) = exporter.get_finished_spans()
    # The remote parent's trace_id (hex) should equal the span's trace_id
    assert format(span.context.trace_id, "032x") == "0af7651916cd43dd8448eb211c80319c"


def test_ignore_context_propagation_skips_traceparent():
    cfg = OpenTelemetryV2Config(exporter="in_memory", ignore_context_propagation=True)
    exporter = InMemorySpanExporter()
    tracer_provider = providers.build_tracer_provider(cfg, exporter=exporter)
    logger = OpenTelemetryV2(config=cfg, tracer_provider=tracer_provider)
    tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    logger.log_success_event(_kwargs(traceparent=tp), None, None, None)
    (span,) = exporter.get_finished_spans()
    # Trace id must NOT match the remote one — we ignored the header.
    assert format(span.context.trace_id, "032x") != "0af7651916cd43dd8448eb211c80319c"


# --------------------------------------------------------------------------- #
#  Baggage promotion (LLM call writes identity into baggage so child spans
#  inherit team/key/model attrs).
# --------------------------------------------------------------------------- #


def test_baggage_identity_promoted_onto_llm_call():
    logger, exporter = _logger()
    logger.log_success_event(_kwargs(), None, None, None)
    (span,) = exporter.get_finished_spans()
    assert span.attributes[LiteLLM.TEAM_ID] == "t1"
    assert span.attributes[LiteLLM.TEAM_ALIAS] == "team one"
    assert span.attributes[GenAI.REQUEST_MODEL] == "gpt-4o"


# --------------------------------------------------------------------------- #
#  Service hooks (Phase 3)
# --------------------------------------------------------------------------- #


class _Service:
    """Stub matching ``ServiceTypes(str, Enum)``."""

    def __init__(self, value):
        self.value = value


class _ServicePayload:
    def __init__(self, service="redis", call_type="set", error=None):
        self.service = _Service(service)
        self.call_type = call_type
        self.error = error


def _service_parent(logger):
    """Helper: a live PROXY_REQUEST span to parent service spans under."""
    return logger._emitter._start(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )


def test_async_service_success_hook_emits_service_span():
    logger, exporter = _logger()
    parent = _service_parent(logger)
    try:
        asyncio.run(
            logger.async_service_success_hook(
                payload=_ServicePayload("redis", "set"),
                parent_otel_span=parent,
                event_metadata={"key1": "val1"},
            )
        )
    finally:
        parent.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    span = by_name["redis"]
    assert span.kind is SpanKind.INTERNAL
    assert span.attributes[LiteLLM.SERVICE_NAME] == "redis"
    assert span.attributes[LiteLLM.SERVICE_CALL_TYPE] == "set"
    # Canonical (V2) namespaced metadata key
    assert span.attributes[f"{LiteLLM.METADATA_PREFIX}key1"] == "val1"
    # V1 bare key (legacy dual-emit)
    assert span.attributes["key1"] == "val1"
    assert span.attributes["service"] == "redis"  # V1 bare key
    assert span.attributes["call_type"] == "set"  # V1 bare key
    assert span.status.status_code is StatusCode.OK


def test_async_service_failure_hook_marks_error_status():
    logger, exporter = _logger()
    parent = _service_parent(logger)
    try:
        asyncio.run(
            logger.async_service_failure_hook(
                payload=_ServicePayload("postgres", "query"),
                error="boom",
                parent_otel_span=parent,
            )
        )
    finally:
        parent.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    span = by_name["postgres"]
    assert span.status.status_code is StatusCode.ERROR
    # Without an explicit error_type from the payload, V2 stamps the fallback.
    assert span.attributes["error.type"] == "error"
    assert span.attributes[LiteLLM.SERVICE_NAME] == "postgres"


def test_async_service_failure_hook_preserves_payload_error_over_override():
    """When the payload itself carries an error, that takes precedence over the override."""
    logger, exporter = _logger()
    parent = _service_parent(logger)
    try:
        asyncio.run(
            logger.async_service_failure_hook(
                payload=_ServicePayload("postgres", "query", error="db-down"),
                error="override-only-used-when-payload-clean",
                parent_otel_span=parent,
            )
        )
    finally:
        parent.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    span = by_name["postgres"]
    assert span.status.status_code is StatusCode.ERROR
    assert "db-down" in (span.status.description or "")


def test_service_hook_without_parent_is_noop():
    """Mirrors V1: no parent OTel span → no service span (no free-standing roots)."""
    logger, exporter = _logger()
    asyncio.run(
        logger.async_service_success_hook(
            payload=_ServicePayload(), parent_otel_span=None
        )
    )
    assert exporter.get_finished_spans() == ()


def test_service_span_inherits_parent_when_provided():
    logger, exporter = _logger()
    parent = logger._emitter._start(
        SpanRole.PROXY_REQUEST, LITELLM_PROXY_REQUEST_SPAN_NAME
    )
    try:
        asyncio.run(
            logger.async_service_success_hook(
                payload=_ServicePayload(), parent_otel_span=parent
            )
        )
    finally:
        parent.end()
    by_name = {s.name: s for s in exporter.get_finished_spans()}
    assert (
        by_name["redis"].parent.span_id
        == by_name[LITELLM_PROXY_REQUEST_SPAN_NAME].get_span_context().span_id
    )


# --------------------------------------------------------------------------- #
#  Proxy SERVER span lifecycle
# --------------------------------------------------------------------------- #


def test_create_proxy_request_started_span_returns_server_span():
    logger, exporter = _logger()
    span = logger.create_litellm_proxy_request_started_span(
        start_time=datetime(2026, 5, 26, 12, 0, 0, tzinfo=timezone.utc), headers={}
    )
    assert span is not None
    span.end()
    (finished,) = exporter.get_finished_spans()
    assert finished.name == LITELLM_PROXY_REQUEST_SPAN_NAME
    assert finished.kind is SpanKind.SERVER


def test_create_proxy_request_started_span_extracts_traceparent():
    logger, exporter = _logger()
    tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    span = logger.create_litellm_proxy_request_started_span(
        start_time=datetime(2026, 5, 26, tzinfo=timezone.utc),
        headers={"traceparent": tp},
    )
    assert span is not None
    span.end()
    (finished,) = exporter.get_finished_spans()
    assert format(finished.context.trace_id, "032x") == (
        "0af7651916cd43dd8448eb211c80319c"
    )


def test_route_status_code_setters_stamp_http_attrs():
    logger, exporter = _logger()
    span = logger.create_litellm_proxy_request_started_span(
        start_time=datetime.now(timezone.utc), headers={}
    )
    OpenTelemetryV2.set_proxy_request_route_attributes(
        span, url_path="/chat/completions", http_route="/chat/completions"
    )
    OpenTelemetryV2.set_response_status_code_attribute(span, 200)
    span.end()
    (finished,) = exporter.get_finished_spans()
    assert finished.attributes[HTTP.URL_PATH] == "/chat/completions"
    assert finished.attributes[HTTP.ROUTE] == "/chat/completions"
    assert finished.attributes[HTTP.RESPONSE_STATUS_CODE] == 200


def test_route_setter_is_noop_on_none_span():
    """All setters must accept ``None`` span gracefully — callers don't pre-check."""
    OpenTelemetryV2.set_proxy_request_route_attributes(None, http_route="/x")
    OpenTelemetryV2.set_response_status_code_attribute(None, 200)
    OpenTelemetryV2.set_preprocessing_duration_attribute(None, {})


def test_preprocessing_duration_attribute_uses_received_and_first_handoff():
    logger, exporter = _logger()
    span = logger.create_litellm_proxy_request_started_span(
        start_time=datetime.now(timezone.utc), headers={}
    )
    container = {
        "first_api_call_start_time": 1000.5,  # epoch seconds
        "litellm_params": {"metadata": {"litellm_received_at": 1000.0}},
    }
    OpenTelemetryV2.set_preprocessing_duration_attribute(span, container)
    span.end()
    (finished,) = exporter.get_finished_spans()
    assert finished.attributes[LiteLLM.PREPROCESSING_MS] == pytest.approx(500.0)


def test_preprocessing_duration_negative_skew_is_skipped():
    logger, exporter = _logger()
    span = logger.create_litellm_proxy_request_started_span(
        start_time=datetime.now(timezone.utc), headers={}
    )
    container = {
        "first_api_call_start_time": 1000.0,
        "litellm_params": {
            "metadata": {"litellm_received_at": 1001.0}
        },  # later than handoff
    }
    OpenTelemetryV2.set_preprocessing_duration_attribute(span, container)
    span.end()
    (finished,) = exporter.get_finished_spans()
    assert LiteLLM.PREPROCESSING_MS not in finished.attributes


def test_preprocessing_duration_missing_anchor_is_noop():
    logger, exporter = _logger()
    span = logger.create_litellm_proxy_request_started_span(
        start_time=datetime.now(timezone.utc), headers={}
    )
    OpenTelemetryV2.set_preprocessing_duration_attribute(
        span, {"first_api_call_start_time": 1000.0}  # no received_at
    )
    span.end()
    (finished,) = exporter.get_finished_spans()
    assert LiteLLM.PREPROCESSING_MS not in finished.attributes


# --------------------------------------------------------------------------- #
#  Constructor / proxy global guard
# --------------------------------------------------------------------------- #


def test_constructor_accepts_v1_compatible_kwargs():
    """Mirrors V1's positional shape — config / callback_name / providers / **kwargs."""
    cfg = OpenTelemetryV2Config(exporter="in_memory")
    tp = providers.build_tracer_provider(cfg)
    logger = OpenTelemetryV2(
        config=cfg,
        callback_name="otel",
        tracer_provider=tp,
        logger_provider=None,
        meter_provider=None,
        turn_off_message_logging=True,
    )
    assert logger.callback_name == "otel"
    assert logger.turn_off_message_logging is True
    assert logger.tracer is not None


def test_default_config_reads_env(monkeypatch):
    """No explicit config → reads env (exporter=console by default)."""
    monkeypatch.delenv("OTEL_EXPORTER", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_PROTOCOL", raising=False)
    logger = OpenTelemetryV2(
        tracer_provider=providers.build_tracer_provider(
            OpenTelemetryV2Config(exporter="in_memory")
        )
    )
    assert logger.config.exporter == "console"


def test_proxy_global_first_registered_wins(monkeypatch):
    """``_init_otel_logger_on_litellm_proxy`` claims the global only when empty."""
    proxy_server = pytest.importorskip("litellm.proxy.proxy_server")
    monkeypatch.setattr(proxy_server, "open_telemetry_logger", None, raising=False)
    cfg = OpenTelemetryV2Config(exporter="in_memory")
    tp = providers.build_tracer_provider(cfg)

    first = OpenTelemetryV2(config=cfg, tracer_provider=tp)
    assert proxy_server.open_telemetry_logger is first

    second = OpenTelemetryV2(config=cfg, tracer_provider=tp)
    # Global still points at the first registration.
    assert proxy_server.open_telemetry_logger is first
    assert second is not first
