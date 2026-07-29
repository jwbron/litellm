import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.openrouter.chat.transformation import OpenrouterConfig
from litellm.llms.openrouter.reasoning import map_thinking_blocks_to_reasoning_content


def _thinking(text, signature="sig"):
    return {"type": "thinking", "thinking": text, "signature": signature}


class TestMapThinkingBlocksToReasoningContent:
    def test_assistant_thinking_becomes_reasoning_content(self):
        """The field the Anthropic adapter writes must reach the field
        OpenRouter reads, and the one it does not read must stop being sent."""
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "answer",
                "thinking_blocks": [_thinking("because X")],
            },
        ]

        out = map_thinking_blocks_to_reasoning_content(messages)

        assert out[1]["reasoning_content"] == "because X"
        assert "thinking_blocks" not in out[1]
        assert out[1]["content"] == "answer"

    def test_anthropic_signature_is_not_forwarded(self):
        """`signature` is an Anthropic re-verification token; it is meaningless
        to OpenRouter and must not ride along."""
        out = map_thinking_blocks_to_reasoning_content(
            [
                {
                    "role": "assistant",
                    "thinking_blocks": [_thinking("t", signature="deadbeef")],
                }
            ]
        )

        assert out[0]["reasoning_content"] == "t"
        assert "deadbeef" not in repr(out[0])

    def test_multiple_blocks_concatenate_in_order(self):
        out = map_thinking_blocks_to_reasoning_content(
            [
                {
                    "role": "assistant",
                    "thinking_blocks": [
                        _thinking("first "),
                        _thinking("second "),
                        _thinking("third"),
                    ],
                }
            ]
        )

        assert out[0]["reasoning_content"] == "first second third"

    def test_redacted_thinking_is_skipped(self):
        """A redacted block carries opaque `data`, not text. There is nothing to
        send, and an empty string would render as the very <think></think> this
        change exists to remove."""
        out = map_thinking_blocks_to_reasoning_content(
            [
                {
                    "role": "assistant",
                    "thinking_blocks": [
                        {"type": "redacted_thinking", "data": "opaque"},
                        _thinking("visible"),
                    ],
                }
            ]
        )

        assert out[0]["reasoning_content"] == "visible"
        assert "opaque" not in repr(out[0])

    def test_only_redacted_blocks_emit_no_field(self):
        out = map_thinking_blocks_to_reasoning_content(
            [
                {
                    "role": "assistant",
                    "thinking_blocks": [{"type": "redacted_thinking", "data": "d"}],
                }
            ]
        )

        assert "reasoning_content" not in out[0]
        assert "thinking_blocks" not in out[0]

    @pytest.mark.parametrize("text", ["", "   ", "\n\t "])
    def test_whitespace_only_reasoning_emits_nothing(self, text):
        out = map_thinking_blocks_to_reasoning_content(
            [{"role": "assistant", "thinking_blocks": [_thinking(text)]}]
        )

        assert "reasoning_content" not in out[0]
        assert "thinking_blocks" not in out[0]

    @pytest.mark.parametrize("role", ["user", "system", "tool"])
    def test_non_assistant_messages_are_untouched(self, role):
        original = {"role": role, "content": "c", "thinking_blocks": [_thinking("t")]}

        out = map_thinking_blocks_to_reasoning_content([dict(original)])

        assert out[0] == original
        assert "reasoning_content" not in out[0]

    def test_assistant_without_thinking_blocks_is_returned_as_is(self):
        original = {"role": "assistant", "content": "plain"}

        out = map_thinking_blocks_to_reasoning_content([original])

        assert out[0] is original

    @pytest.mark.parametrize(
        "blocks",
        [
            "not-a-list",
            {"type": "thinking"},
            17,
            [None],
            ["bare string"],
            [{"type": "thinking", "thinking": None}],
            [{"type": "thinking"}],
            [{}],
        ],
    )
    def test_malformed_blocks_never_raise(self, blocks):
        """Fail soft: an unanticipated shape costs the reasoning, not the
        request."""
        out = map_thinking_blocks_to_reasoning_content(
            [{"role": "assistant", "content": "c", "thinking_blocks": blocks}]
        )

        assert len(out) == 1
        assert out[0]["content"] == "c"
        assert "reasoning_content" not in out[0]

    def test_a_malformed_sibling_does_not_cost_a_good_block(self):
        out = map_thinking_blocks_to_reasoning_content(
            [
                {
                    "role": "assistant",
                    "thinking_blocks": [
                        _thinking("kept "),
                        None,
                        {"type": "thinking"},
                        _thinking("too"),
                    ],
                }
            ]
        )

        assert out[0]["reasoning_content"] == "kept too"

    def test_unparseable_blocks_field_leaves_the_message_untouched(self):
        original = {"role": "assistant", "content": "c", "thinking_blocks": "not-a-list"}

        out = map_thinking_blocks_to_reasoning_content([dict(original)])

        assert out[0] == original

    def test_input_messages_are_not_mutated(self):
        block = _thinking("t")
        messages = [{"role": "assistant", "content": "a", "thinking_blocks": [block]}]
        before = [
            {"role": "assistant", "content": "a", "thinking_blocks": [dict(block)]}
        ]

        map_thinking_blocks_to_reasoning_content(messages)

        assert messages == before

    def test_mapping_is_idempotent(self):
        messages = [
            {"role": "assistant", "content": "a", "thinking_blocks": [_thinking("why")]}
        ]

        once = map_thinking_blocks_to_reasoning_content(messages)
        twice = map_thinking_blocks_to_reasoning_content(once)

        assert twice == once
        assert twice[0]["reasoning_content"] == "why"

    def test_non_list_messages_pass_through(self):
        assert map_thinking_blocks_to_reasoning_content(None) is None
        assert map_thinking_blocks_to_reasoning_content("nope") == "nope"


class TestTransformRequestCarriesReasoning:
    def test_transform_request_maps_reasoning_onto_the_body(self):
        """End to end through the config: the request body an OpenRouter model
        receives carries `reasoning_content` and no `thinking_blocks`."""
        config = OpenrouterConfig()

        body = config.transform_request(
            model="poolside/laguna-s-2.1",
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "answer",
                    "thinking_blocks": [_thinking("prior reasoning")],
                },
                {"role": "user", "content": "follow up"},
            ],
            optional_params={},
            litellm_params={},
            headers={},
        )

        assistant = [m for m in body["messages"] if m.get("role") == "assistant"][0]
        assert assistant["reasoning_content"] == "prior reasoning"
        assert "thinking_blocks" not in assistant

    def test_transform_request_leaves_a_plain_history_alone(self):
        config = OpenrouterConfig()

        body = config.transform_request(
            model="poolside/laguna-s-2.1",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "answer"},
            ],
            optional_params={},
            litellm_params={},
            headers={},
        )

        assistant = [m for m in body["messages"] if m.get("role") == "assistant"][0]
        assert "reasoning_content" not in assistant
        assert assistant["content"] == "answer"
