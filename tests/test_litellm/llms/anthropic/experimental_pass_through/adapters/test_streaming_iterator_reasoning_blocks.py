"""
Test that AnthropicStreamWrapper translates OpenAI-style reasoning_content
streams into proper Anthropic thinking blocks, without losing the first
delta of each block on a block-type transition.

Providers routed through OpenAI-compatible endpoints (e.g. OpenRouter
reasoning models like Kimi K2.7 / DeepSeek) stream reasoning via
delta.reasoning_content rather than Anthropic-native thinking_blocks.
Without the fix:
  - the reasoning was emitted as thinking_delta events inside a block
    whose content_block_start said type="text" (malformed stream; clients
    like Claude Code then render the reasoning as the visible answer), and
  - the chunk that triggers a block-type transition had its delta dropped
    unless it was an input_json_delta, eating the first token of every
    thinking block and of the answer text that follows it.
"""

import os
import sys
from typing import List
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import Delta, StreamingChoices


def _make_chunk(
    delta: Delta,
    finish_reason: str = None,
) -> MagicMock:
    """Create a minimal streaming chunk with the given delta and finish_reason."""
    chunk = MagicMock()
    chunk.choices = [
        StreamingChoices(
            finish_reason=finish_reason,
            index=0,
            delta=delta,
            logprobs=None,
        )
    ]
    chunk.usage = None
    chunk._hidden_params = {}
    return chunk


def _reasoning_then_text_chunks() -> List[MagicMock]:
    """A reasoning-model stream: reasoning deltas, then answer text, then finish."""
    return [
        _make_chunk(Delta(content=None, role="assistant", reasoning_content="We")),
        _make_chunk(Delta(content=None, role="assistant", reasoning_content=" think.")),
        _make_chunk(Delta(content="The", role="assistant")),
        _make_chunk(Delta(content=" answer.", role="assistant")),
        _make_chunk(
            Delta(content=None, role="assistant"),
            finish_reason="stop",
        ),
    ]


def _collect_events_sync(wrapper: AnthropicStreamWrapper) -> List[dict]:
    """Drain all events from a sync AnthropicStreamWrapper."""
    events = []
    for event in wrapper:
        events.append(event)
    return events


async def _collect_events_async(wrapper: AnthropicStreamWrapper) -> List[dict]:
    """Drain all events from an async AnthropicStreamWrapper."""
    events = []
    async for event in wrapper:
        events.append(event)
    return events


def _reassemble_blocks(events: List[dict]) -> List[dict]:
    """Reassemble content blocks (type + accumulated content) from a stream."""
    blocks = {}
    order = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("type") == "content_block_start":
            index = event["index"]
            content_block = event["content_block"]
            blocks[index] = {
                "type": content_block.get("type"),
                "content": content_block.get("thinking")
                or content_block.get("text")
                or "",
            }
            order.append(index)
        elif event.get("type") == "content_block_delta":
            delta = event["delta"]
            blocks[event["index"]]["content"] += (
                delta.get("thinking") or delta.get("text") or ""
            )
    return [blocks[i] for i in order]


def _assert_reasoning_then_text_blocks(events: List[dict]) -> None:
    blocks = _reassemble_blocks(events)

    thinking_blocks = [b for b in blocks if b["type"] == "thinking"]
    assert len(thinking_blocks) == 1, (
        f"Expected exactly one thinking content block; blocks: {blocks}"
    )
    # Includes the first reasoning token ("We") from the transition trigger chunk.
    assert thinking_blocks[0]["content"] == "We think."

    text_blocks = [b for b in blocks if b["type"] == "text" and b["content"]]
    assert len(text_blocks) == 1, (
        f"Expected exactly one non-empty text content block; blocks: {blocks}"
    )
    # Includes the first answer token ("The") from the transition trigger chunk.
    assert text_blocks[0]["content"] == "The answer."


@pytest.mark.asyncio
async def test_async_reasoning_content_streams_as_thinking_block():
    """
    reasoning_content deltas must open a thinking content block (not pour
    thinking_delta events into a text block), and the answer text must land
    in a separate text block — with the first token of each block intact.
    """

    async def mock_stream():
        for c in _reasoning_then_text_chunks():
            yield c

    wrapper = AnthropicStreamWrapper(
        completion_stream=mock_stream(),
        model="test-model",
    )

    events = await _collect_events_async(wrapper)
    _assert_reasoning_then_text_blocks(events)


def test_sync_reasoning_content_streams_as_thinking_block():
    """Sync counterpart of the reasoning_content thinking-block test."""
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(_reasoning_then_text_chunks()),
        model="test-model",
    )

    events = _collect_events_sync(wrapper)
    _assert_reasoning_then_text_blocks(events)


@pytest.mark.asyncio
async def test_async_native_thinking_blocks_not_duplicated_on_transition():
    """
    Providers that send Anthropic-native thinking_blocks embed the thinking
    content in the content_block_start itself. The transition trigger chunk's
    thinking_delta must NOT also be re-queued, or the first chunk's thinking
    would appear twice.
    """
    chunks = [
        _make_chunk(
            Delta(
                content=None,
                role="assistant",
                thinking_blocks=[
                    {"type": "thinking", "thinking": "abc", "signature": None}
                ],
                reasoning_content="abc",
            )
        ),
        _make_chunk(
            Delta(
                content=None,
                role="assistant",
                thinking_blocks=[
                    {"type": "thinking", "thinking": "def", "signature": None}
                ],
                reasoning_content="def",
            )
        ),
        _make_chunk(Delta(content="Hi", role="assistant")),
        _make_chunk(
            Delta(content=None, role="assistant"),
            finish_reason="stop",
        ),
    ]

    async def mock_stream():
        for c in chunks:
            yield c

    wrapper = AnthropicStreamWrapper(
        completion_stream=mock_stream(),
        model="test-model",
    )

    events = await _collect_events_async(wrapper)
    blocks = _reassemble_blocks(events)

    thinking_blocks = [b for b in blocks if b["type"] == "thinking"]
    assert len(thinking_blocks) == 1, (
        f"Expected exactly one thinking content block; blocks: {blocks}"
    )
    assert thinking_blocks[0]["content"] == "abcdef", (
        "Trigger chunk's thinking must not be duplicated when the "
        "content_block_start already embeds it"
    )

    text_blocks = [b for b in blocks if b["type"] == "text" and b["content"]]
    assert len(text_blocks) == 1
    assert text_blocks[0]["content"] == "Hi"
