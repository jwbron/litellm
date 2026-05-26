"""Mapper protocol and attribute value types."""

from typing import Dict, Mapping, Optional, Sequence, Union

from typing_extensions import Protocol, runtime_checkable

from litellm.integrations.otel.payloads import (
    GuardrailSpanData,
    LLMCallSpanData,
    ServiceSpanData,
)
from litellm.integrations.otel.spans import SpanRole

AttrScalar = Union[str, bool, int, float]
# Mirrors ``opentelemetry.util.types.AttributeValue`` (homogeneous sequences)
# without importing the SDK, so mappers stay OTel-free.
AttrValue = Union[
    AttrScalar,
    Sequence[str],
    Sequence[bool],
    Sequence[int],
    Sequence[float],
]
AttributeMap = Dict[str, AttrValue]

# The closed set of span-data types the engine routes through the mapper chain.
# Wider span roles (PROXY_REQUEST, MANAGEMENT) are root spans owned by callers
# and don't flow through ``emit``.
SpanData = Union[LLMCallSpanData, GuardrailSpanData, ServiceSpanData]


def drop_none(values: Mapping[str, Optional[AttrValue]]) -> AttributeMap:
    """Return ``values`` with ``None``-valued entries removed."""
    return {k: v for k, v in values.items() if v is not None}


@runtime_checkable
class AttributeMapper(Protocol):
    """Maps a typed span input to a flat dict of OTel span attributes.

    One method per mapper, role-dispatched internally. The engine calls this
    uniformly for every span kind — mappers that don't speak a given role
    return ``{}``. This is why the engine itself contains no attribute keys.
    """

    def map(self, role: SpanRole, data: SpanData) -> AttributeMap: ...
