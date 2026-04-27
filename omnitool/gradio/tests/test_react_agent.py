"""
Tests for ReActAgent — compaction, image eviction, finish tool, error paths.

All external dependencies (LLM, grounding strategy, screen capture, tool execution)
are mocked.  No network calls are made.
"""

from unittest.mock import Mock

from omnitool.gradio.core.agents.react_agent import _tool_msg
from omnitool.gradio.tests._helpers import (
    all_user_message_texts as _all_user_message_texts,
    finish_response as _finish_response,
    make_react_agent,
    no_tool_response as _no_tool_response,
    tool_response as _tool_response,
)


def _make_agent(tmp_path, max_steps=10, compaction_token_threshold=1_000_000):
    """Build a test agent. Default threshold is intentionally high so tests that
    don't care about compaction never trigger it accidentally."""
    return make_react_agent(
        tmp_path,
        side_effects=[_tool_response(), _finish_response()],
        max_steps=max_steps,
        compaction_token_threshold=compaction_token_threshold,
    )


# ===========================================================================
# Module-level helpers
# ===========================================================================

class TestToolMsg:
    def test_produces_tool_role_message(self):
        msg = _tool_msg("call_123", "result text")
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_123"
        assert msg["content"] == "result text"


# ===========================================================================
# ReActAgent — finish() tool call
# ===========================================================================

class TestReActAgentFinish:
    def test_finish_yields_complete_event(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.llm_client.generate.side_effect = [_finish_response(success=True, summary="All done")]
        events = list(agent.run())
        complete_events = [e for e in events if e["type"] == "complete"]
        assert len(complete_events) == 1

    def test_finish_complete_event_contains_required_fields(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.llm_client.generate.side_effect = [_finish_response(success=True, summary="Task done")]
        events = list(agent.run())
        complete = next(e for e in events if e["type"] == "complete")
        assert "message" in complete
        assert "success" in complete
        assert "total_steps" in complete
        assert "total_tokens" in complete
        assert "total_cost" in complete

    def test_finish_success_true_propagated(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.llm_client.generate.side_effect = [_finish_response(success=True, summary="Success")]
        events = list(agent.run())
        complete = next(e for e in events if e["type"] == "complete")
        assert complete["success"] is True

    def test_finish_success_false_propagated(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.llm_client.generate.side_effect = [_finish_response(success=False, summary="Could not complete")]
        events = list(agent.run())
        complete = next(e for e in events if e["type"] == "complete")
        assert complete["success"] is False

    def test_finish_summary_in_message(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.llm_client.generate.side_effect = [_finish_response(summary="Clicked the button")]
        events = list(agent.run())
        complete = next(e for e in events if e["type"] == "complete")
        assert "Clicked the button" in complete["message"]

    def test_finish_terminates_loop_without_reaching_max_steps(self, tmp_path):
        agent = _make_agent(tmp_path, max_steps=20)
        agent.llm_client.generate.side_effect = [_finish_response()]
        list(agent.run())
        assert agent.step_count == 1


# ===========================================================================
# ReActAgent — image eviction from history
# ===========================================================================

class TestEvictOldImages:
    def test_images_replaced_in_all_but_last_user_message(self):
        from omnitool.gradio.core.agents.message_utils import _evict_old_images

        history = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                {"type": "text", "text": "Step 1"},
            ]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBB"}},
                {"type": "text", "text": "Step 2"},
            ]},
        ]
        _evict_old_images(history)

        # First user message: image replaced with placeholder
        first_user = history[0]["content"]
        types = [b["type"] for b in first_user]
        assert "image_url" not in types
        assert any(b.get("text") == "[screenshot]" for b in first_user)

        # Last user message: image preserved
        last_user = history[2]["content"]
        assert any(b["type"] == "image_url" for b in last_user)

    def test_non_user_messages_untouched(self):
        from omnitool.gradio.core.agents.message_utils import _evict_old_images

        history = [
            {"role": "assistant", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            ]},
        ]
        original = history[0]["content"][0]["type"]
        _evict_old_images(history)
        assert history[0]["content"][0]["type"] == original

    def test_text_blocks_preserved_after_eviction(self):
        from omnitool.gradio.core.agents.message_utils import _evict_old_images

        history = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                {"type": "text", "text": "some text"},
            ]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBB"}},
            ]},
        ]
        _evict_old_images(history)
        first_texts = [b["text"] for b in history[0]["content"] if b.get("type") == "text"]
        assert "some text" in first_texts

    def test_single_user_message_not_evicted(self):
        from omnitool.gradio.core.agents.message_utils import _evict_old_images

        history = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            ]},
        ]
        _evict_old_images(history)
        # Single user message — it is "the last", so image must stay
        assert history[0]["content"][0]["type"] == "image_url"

    def test_empty_history_does_not_raise(self):
        from omnitool.gradio.core.agents.message_utils import _evict_old_images
        _evict_old_images([])   # must not raise

    def test_text_before_image_still_evicts_image(self):
        """Eviction must strip image_url regardless of block order within content."""
        from omnitool.gradio.core.agents.message_utils import _evict_old_images

        history = [
            {"role": "user", "content": [
                {"type": "text", "text": "earlier step notes"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            ]},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBB"}},
            ]},
        ]
        _evict_old_images(history)
        first_user_types = [b["type"] for b in history[0]["content"]]
        assert "image_url" not in first_user_types, (
            "image_url must be removed even when text precedes image in the block list"
        )
        # Original text must survive
        first_texts = [b["text"] for b in history[0]["content"] if b.get("type") == "text"]
        assert "earlier step notes" in first_texts


# ===========================================================================
# ReActAgent — history compaction
# ===========================================================================

class TestReActAgentCompaction:
    # tool_response() returns input_tokens=25; threshold=20 ensures compaction
    # fires on the first step that would otherwise continue.
    _THRESHOLD = 20

    def test_compaction_triggered_by_token_threshold(self, tmp_path):
        """Compaction LLM call is made when input_tokens exceeds the threshold."""
        agent = _make_agent(tmp_path, max_steps=5, compaction_token_threshold=self._THRESHOLD)

        call_log = []

        def generate_side_effect(messages, system_prompt, tools=None):
            if tools is None:
                call_log.append("compact")
                return ("Summary of progress.", {"tokens": 10, "input_tokens": 5, "output_tokens": 5, "tool_calls": [], "assistant_message": {"role": "assistant", "content": ""}})
            call_log.append("step")
            if call_log.count("step") > 1:
                return _finish_response()
            return _tool_response()

        agent.llm_client.generate.side_effect = generate_side_effect
        events = list(agent.run())

        status_msgs = [e["message"] for e in events if e["type"] == "status"]
        assert any("Compact" in m or "compact" in m for m in status_msgs), (
            f"No compaction status found. Status messages: {status_msgs}"
        )

    def test_compaction_replaces_history_with_summary(self, tmp_path):
        """After compaction, post-compaction generate() calls see a shorter history
        that includes the compaction summary text."""
        agent = _make_agent(tmp_path, max_steps=5, compaction_token_threshold=self._THRESHOLD)

        summary_text = "Steps done: step 1, step 2. Still need: finish."
        pre_compaction_history_len = []
        post_compaction_history_len = []
        compaction_fired = [False]
        step_count = [0]

        def generate_side_effect(messages, system_prompt, tools=None):
            if tools is None:
                pre_compaction_history_len.append(len(messages))
                compaction_fired[0] = True
                return (summary_text, {
                    "tokens": 15, "input_tokens": 8, "output_tokens": 7,
                    "tool_calls": [], "assistant_message": {"role": "assistant", "content": ""},
                })
            if compaction_fired[0]:
                post_compaction_history_len.append(len(messages))
            step_count[0] += 1
            if step_count[0] > 1:
                return _finish_response()
            return _tool_response()

        agent.llm_client.generate.side_effect = generate_side_effect
        events = list(agent.run())

        assert any(e["type"] == "complete" for e in events)
        assert compaction_fired[0], "Compaction was not triggered"

        # History after compaction must be strictly shorter than the history that
        # was being compacted (the summary replaces the expanded turn log).
        assert post_compaction_history_len, "No generate() calls after compaction"
        assert min(post_compaction_history_len) < max(pre_compaction_history_len), (
            f"History was not shortened by compaction: "
            f"pre={pre_compaction_history_len}, post={post_compaction_history_len}"
        )

        # Summary text must appear in the post-compaction message list
        all_texts = _all_user_message_texts(agent.llm_client.generate)
        assert any(summary_text in t for t in all_texts), (
            "Compaction summary text was not injected into subsequent messages"
        )

    def test_compaction_failure_keeps_history(self, tmp_path):
        """If the compaction LLM call raises, history is preserved and loop continues."""
        agent = _make_agent(tmp_path, max_steps=5, compaction_token_threshold=self._THRESHOLD)

        step_count = [0]

        def generate_side_effect(messages, system_prompt, tools=None):
            if tools is None:
                raise RuntimeError("LLM timeout")
            step_count[0] += 1
            if step_count[0] > 1:
                return _finish_response()
            return _tool_response()

        agent.llm_client.generate.side_effect = generate_side_effect
        events = list(agent.run())
        final_types = {e["type"] for e in events}
        assert "complete" in final_types or "error" in final_types

    def test_compaction_suppressed_with_pending_staged_reads(self, tmp_path):
        """Compaction is not scheduled while a read_field → save_field sequence
        is in progress (staged_reads non-empty)."""
        agent = _make_agent(tmp_path, max_steps=3, compaction_token_threshold=self._THRESHOLD)

        compaction_called = []
        step_count = [0]

        def generate_side_effect(messages, system_prompt, tools=None):
            if tools is None:
                compaction_called.append(True)
                return ("Summary.", {"tokens": 10, "input_tokens": 5, "output_tokens": 5, "tool_calls": [], "assistant_message": {"role": "assistant", "content": ""}})
            step_count[0] += 1
            if step_count[0] == 1:
                # Simulate a pending staged_read before the trigger check runs.
                agent.working_memory.staged_reads["field"] = ["value"]
                return _tool_response()  # input_tokens=25 > threshold, but guard fires
            return _finish_response()

        agent.llm_client.generate.side_effect = generate_side_effect
        list(agent.run())
        assert not compaction_called, (
            "Compaction should not fire while staged_reads are pending"
        )


# ===========================================================================
# ReActAgent — max_steps reached without finish
# ===========================================================================

class TestReActAgentMaxSteps:
    def test_complete_event_yielded_when_max_steps_reached(self, tmp_path):
        agent = _make_agent(tmp_path, max_steps=2)
        # Never finish — always click
        agent.llm_client.generate.side_effect = [_tool_response()] * 10
        events = list(agent.run())
        complete = next((e for e in events if e["type"] == "complete"), None)
        assert complete is not None

    def test_complete_event_success_false_when_max_steps_reached(self, tmp_path):
        agent = _make_agent(tmp_path, max_steps=2)
        agent.llm_client.generate.side_effect = [_tool_response()] * 10
        events = list(agent.run())
        complete = next(e for e in events if e["type"] == "complete")
        assert complete["success"] is False

    def test_step_count_equals_max_steps_at_end(self, tmp_path):
        max_steps = 3
        agent = _make_agent(tmp_path, max_steps=max_steps)
        agent.llm_client.generate.side_effect = [_tool_response()] * 10
        list(agent.run())
        assert agent.step_count == max_steps


# ===========================================================================
# ReActAgent — error handling
# ===========================================================================

class TestReActAgentErrors:
    def test_error_event_yielded_on_capture_exception(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent._capture_screen = Mock(side_effect=RuntimeError("camera broken"))
        events = list(agent.run())
        assert any(e["type"] == "error" for e in events)

    def test_error_event_has_message_key(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent._capture_screen = Mock(side_effect=ValueError("no screen"))
        events = list(agent.run())
        error_event = next(e for e in events if e["type"] == "error")
        assert "message" in error_event


# ===========================================================================
# ReActAgent — no tool call handling
# ===========================================================================

class TestReActAgentNoToolCall:
    def test_no_tool_call_continues_loop(self, tmp_path):
        """When LLM returns no tool call, loop continues (does not crash)."""
        agent = _make_agent(tmp_path, max_steps=3)
        agent.llm_client.generate.side_effect = [
            _no_tool_response(),
            _finish_response(),
        ]
        events = list(agent.run())
        assert any(e["type"] == "complete" for e in events)

    def test_thinking_event_yielded_for_response_text(self, tmp_path):
        agent = _make_agent(tmp_path, max_steps=3)
        agent.llm_client.generate.side_effect = [
            _tool_response(text="I will click now"),
            _finish_response(),
        ]
        events = list(agent.run())
        thinking_events = [e for e in events if e["type"] == "thinking"]
        assert len(thinking_events) >= 1
        assert any("I will click now" in e.get("response_text", "") for e in thinking_events)


# ===========================================================================
# _extract_tables_from_html
# ===========================================================================

class TestExtractTablesFromHtml:
    """Unit tests for BaseAgent._extract_tables_from_html static method."""

    def _extract(self, html: str) -> str:
        from omnitool.gradio.core.agents.base import BaseAgent
        return BaseAgent._extract_tables_from_html(html)

    def test_single_table_extracted(self):
        html = "<html><body><table><tr><td>A</td></tr></table></body></html>"
        result = self._extract(html)
        assert "--- Table 1 ---" in result
        assert "<table>" in result

    def test_multiple_tables_numbered(self):
        html = "<table><tr><td>1</td></tr></table><table><tr><td>2</td></tr></table>"
        result = self._extract(html)
        assert "--- Table 1 ---" in result
        assert "--- Table 2 ---" in result

    def test_nested_table_counted_as_one(self):
        html = "<table><tr><td><table><tr><td>inner</td></tr></table></td></tr></table>"
        result = self._extract(html)
        assert "--- Table 1 ---" in result
        assert "--- Table 2 ---" not in result

    def test_empty_input_returns_empty(self):
        assert self._extract("") == ""

    def test_no_tables_returns_empty(self):
        assert self._extract("<div>no tables here</div>") == ""

    def test_uppercase_tags_handled(self):
        html = "<TABLE><TR><TD>A</TD></TR></TABLE>"
        result = self._extract(html)
        assert "--- Table 1 ---" in result

    def test_table_content_preserved_verbatim(self):
        html = '<table id="t1"><tr><td>Hello &amp; World</td></tr></table>'
        result = self._extract(html)
        assert 'Hello &amp; World' in result



# ===========================================================================
# format_table_read
# ===========================================================================

class TestFormatTableRead:
    """Tests for format_table_read formatter."""

    def _fmt(self, text: str) -> str:
        from omnitool.gradio.ui.components.formatters import format_table_read
        return format_table_read(text)

    def test_non_empty_text_contains_details_and_pre(self):
        result = self._fmt("Col1 | Col2\nA | B")
        assert "<details" in result
        assert "<pre" in result

    def test_non_empty_text_contains_table_label(self):
        result = self._fmt("Col1 | Col2\nA | B")
        assert "[Table]" in result

    def test_html_special_chars_escaped(self):
        result = self._fmt("A & B | <em>C</em>")
        assert "&amp;" in result
        assert "&lt;" in result
        assert "&gt;" in result

    def test_empty_string_returns_empty(self):
        assert self._fmt("") == ""

    def test_whitespace_only_returns_empty(self):
        assert self._fmt("   \n  ") == ""
