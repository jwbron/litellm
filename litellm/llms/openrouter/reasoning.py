"""Carry prior-turn assistant reasoning to OpenRouter on the request path.

The Anthropic adapter (``/v1/messages``, used by Anthropic-format clients such
as Claude Code) converts each assistant ``thinking`` content block into an
entry on ``assistant_message["thinking_blocks"]``; see
``_translate_anthropic_messages_to_openai`` in
``llms/anthropic/experimental_pass_through/adapters/transformation.py``.

Nothing on the OpenRouter *request* path consumes that field.
``llms/openrouter/chat/transformation.py`` names reasoning only on the
response side (``reasoning`` -> ``reasoning_content`` on streaming deltas), and
``OpenAIGPTConfig.transform_request`` puts ``messages`` straight into the
request body. So ``thinking_blocks`` is transmitted as a field no provider
reads, and every historical assistant turn reaches the model with its reasoning
missing.

For a model whose chat template re-renders prior thinking that is a malformed
history rather than a lost optimisation. Poolside Laguna renders
``'<think>' + message.reasoning|message.reasoning_content + '</think>'`` for
each previous assistant turn, so every one of them arrives as a literal empty
``<think></think>``; Poolside's model card warns that this degrades follow-up
behaviour. Any OpenRouter reasoning model with a template of that shape is
affected.

OpenRouter accepts ``reasoning``, ``reasoning_content`` and
``reasoning_details`` interchangeably on an assistant message and documents
this for exactly this multi-turn tool-calling case. We send ``reasoning_content``
as a plain string: ``reasoning_details`` exists to carry encrypted or summarised
blocks, and the Anthropic adapter produces neither. Anthropic ``signature``
values are re-verification tokens meaningful only to Anthropic and are not
forwarded.
"""

from typing import Any

# ``redacted_thinking`` blocks carry an opaque ``data`` payload rather than
# text (see ``ChatCompletionRedactedThinkingBlock``), so there is no plaintext
# to hand a provider expecting a string; they contribute nothing.
_THINKING_BLOCK_TYPE = "thinking"

# The field the Anthropic adapter writes, and the field OpenRouter reads.
_SOURCE_FIELD = "thinking_blocks"
_TARGET_FIELD = "reasoning_content"


def _extract_reasoning_text(blocks: Any) -> str:
    """Concatenate the plaintext of ``blocks``, in order.

    Non-dict entries, entries of a type other than ``thinking``, and entries
    whose ``thinking`` is not a string are skipped individually: one malformed
    block should not cost the reasoning of its well-formed siblings. A
    ``blocks`` that is not a list at all is a shape we do not understand, and
    the caller leaves the message untouched.
    """
    if not isinstance(blocks, list):
        raise TypeError(f"{_SOURCE_FIELD} is {type(blocks).__name__}, expected list")
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") != _THINKING_BLOCK_TYPE:
            continue
        text = block.get(_THINKING_BLOCK_TYPE)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def map_thinking_blocks_to_reasoning_content(messages: Any) -> Any:
    """Return ``messages`` with assistant reasoning moved to the wire field.

    For each assistant message carrying ``thinking_blocks``: drop that field so
    no unknown key is transmitted, and set ``reasoning_content`` to the
    concatenated plaintext when there is any. A message whose blocks yield only
    whitespace (or nothing at all, e.g. ``redacted_thinking`` only) loses the
    field and gains nothing.

    User, system and tool messages are returned untouched, as is any assistant
    message with no ``thinking_blocks`` key. The input is not mutated: touched
    messages are shallow-copied.

    Never raises. A message this cannot make sense of is passed through exactly
    as it arrived, so the worst case is the previous behaviour.
    """
    if not isinstance(messages, list):
        return messages

    out: list[Any] = []
    for message in messages:
        try:
            if not isinstance(message, dict):
                out.append(message)
                continue
            if message.get("role") != "assistant" or _SOURCE_FIELD not in message:
                out.append(message)
                continue

            text = _extract_reasoning_text(message.get(_SOURCE_FIELD))

            updated: dict[str, Any] = dict(message)
            updated.pop(_SOURCE_FIELD, None)
            # Whitespace-only reasoning is not reasoning; emitting it would
            # render as an empty <think></think> on templates that re-render
            # prior thinking, which is the failure this exists to remove.
            if text.strip():
                updated[_TARGET_FIELD] = text
            out.append(updated)
        except Exception:
            # A diagnostic-grade transform must never break a request.
            out.append(message)
    return out


__all__ = ["map_thinking_blocks_to_reasoning_content"]
