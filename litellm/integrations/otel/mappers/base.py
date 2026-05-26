"""Mapper protocol and attribute value types."""

from typing import Dict, Mapping, Optional, Sequence, Union

from typing_extensions import Protocol, runtime_checkable

from litellm.integrations.otel.payloads import LLMCallSpanData

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


def drop_none(values: Mapping[str, Optional[AttrValue]]) -> AttributeMap:
    """Return ``values`` with ``None``-valued entries removed.

    Mappers declare ``{attribute_key: source_value}`` as a single dict literal and
    pipe it through this helper to skip absent fields — instead of guarding every
    assignment with ``if x is not None``. Uses strict ``is not None`` so legitimate
    zero / empty-string / ``False`` values survive; callers convert empty
    collections to ``None`` themselves when they want those skipped.
    """
    return {k: v for k, v in values.items() if v is not None}


@runtime_checkable
class AttributeMapper(Protocol):
    """Maps a typed span input to a flat dict of OTel span attributes."""

    def map_llm_call(self, data: LLMCallSpanData) -> AttributeMap: ...
