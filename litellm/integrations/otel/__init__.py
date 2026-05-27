"""Typed, semconv-aligned OpenTelemetry instrumentation for LiteLLM.

The three sources of truth — attribute keys (:mod:`semconv`), the span+hierarchy
registry (:mod:`spans`), and the typed span-data inputs (:mod:`payloads`) — plus
:mod:`config` are exported here and are free of any ``opentelemetry`` import.
The engine layer (``emitter``, ``providers``, ``context``, ``metrics``) and the
``CustomLogger`` adapter (``logger``) are reached via their submodule paths so
that importing this package never requires the OTel SDK.

The ``LITELLM_OTEL_V2`` env var gates whether the factory in
``litellm_core_utils.litellm_logging`` constructs the V2 ``OpenTelemetryV2``
class (from :mod:`logger`) instead of the legacy ``OpenTelemetry``. The two
classes can coexist while Phases 2-4 land.
"""

from litellm.integrations.otel.config import (
    OTEL_V2_ENV,
    OpenTelemetryV2Config,
    is_otel_v2_enabled,
)
from litellm.integrations.otel.payloads import (
    GuardrailSpanData,
    LLMCallSpanData,
    LLMRequestParams,
    LLMUsage,
    ManagementSpanData,
    ProxyRequestSpanData,
    RequestIdentity,
    ServerInfo,
    ServiceSpanData,
    SpanError,
    promoted_baggage,
)
from litellm.integrations.otel.semconv import (
    BAGGAGE_PROMOTED_KEYS,
    DEFAULT_BAGGAGE_METADATA_KEYS,
    Error,
    GenAI,
    GenAIOperation,
    GenAIProvider,
    HTTP,
    LiteLLM,
    Metric,
    Server,
    resolve_operation,
    resolve_provider,
)
from litellm.integrations.otel.spans import (
    SPAN_REGISTRY,
    LiteLLMSpanKind,
    SpanRole,
    SpanSpec,
    validate_registry,
)

__all__ = [
    # config
    "OTEL_V2_ENV",
    "OpenTelemetryV2Config",
    "is_otel_v2_enabled",
    # semconv
    "BAGGAGE_PROMOTED_KEYS",
    "DEFAULT_BAGGAGE_METADATA_KEYS",
    "Error",
    "GenAI",
    "GenAIOperation",
    "GenAIProvider",
    "HTTP",
    "LiteLLM",
    "Metric",
    "Server",
    "resolve_operation",
    "resolve_provider",
    # spans
    "SPAN_REGISTRY",
    "LiteLLMSpanKind",
    "SpanRole",
    "SpanSpec",
    "validate_registry",
    # payloads
    "GuardrailSpanData",
    "LLMCallSpanData",
    "LLMRequestParams",
    "LLMUsage",
    "ManagementSpanData",
    "ProxyRequestSpanData",
    "RequestIdentity",
    "ServerInfo",
    "ServiceSpanData",
    "SpanError",
    "promoted_baggage",
]
