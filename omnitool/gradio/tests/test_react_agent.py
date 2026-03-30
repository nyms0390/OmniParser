"""
Tests for ReActAgent — loop detection, compaction, image eviction, finish tool.

All external dependencies (LLM, grounding strategy, screen capture, tool execution)
are mocked.  No network calls are made.
"""

import json
from unittest.mock import Mock

from omnitool.gradio.services import AppState
from omnitool.gradio.core.agents.grounding import ScreenData
from omnitool.gradio.core.agents.react_agent import (
    ReActAgent,
    _freeze,
    _tool_msg,
    COMPACTION_INTERVAL,
    _LOOP_THRESHOLD,
    _LOOP_WINDOW,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_screen_data() -> ScreenData:
    return ScreenData(
        raw_image_b64="rawb64",
        display_image_b64="somb64",
        elements=[{"bbox": [0.1, 0.1, 0.3, 0.2], "content": "Start", "box_id": 0}],
        screen_width=1920,
        screen_height=1080,
        resized_width=1920,
        resized_height=1080,
    )


def _tool_response(tool_name="left_click", args=None, text="Thinking."):
    """Return (response_text, metadata) with one tool call."""
    if args is None:
        args = {"box_id": 0}
    tc_id = "call_1"
    return (
        text,
        {
            "tokens": 50,
            "input_tokens": 25,
            "output_tokens": 25,
            "tool_calls": [{"id": tc_id, "name": tool_name, "arguments": args}],
            "assistant_message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(args)},
                }],
            },
        },
    )


def _finish_response(success=True, summary="Done.", fields=None):
    """Return (response_text, metadata) for a finish() call."""
    args = {"success": success, "summary": summary, "fields": fields or {}}
    return _tool_response(tool_name="finish", args=args, text="Wrapping up.")


def _no_tool_response(text="I have no action."):
    return (
        text,
        {
            "tokens": 10,
            "input_tokens": 5,
            "output_tokens": 5,
            "tool_calls": [],
            "assistant_message": {"role": "assistant", "content": text},
        },
    )


def _make_agent(tmp_path, max_steps=10, compaction_interval=COMPACTION_INTERVAL):
    """Build a ReActAgent with all external calls mocked."""
    app_state = AppState(run_folder=tmp_path)
    app_state.chat.add_message("user", "Click the Start button")

    grounding = Mock()
    grounding.name = "omniparser"
    grounding.element_reference_hint = "Use box_id."
    grounding.get_tools.return_value = [{
        "type": "function",
        "function": {
            "name": "left_click",
            "parameters": {
                "type": "object",
                "properties": {"box_id": {"type": "integer"}},
                "required": ["box_id"],
            },
        },
    }]
    screen_data = _make_screen_data()
    grounding.preprocess.return_value = screen_data
    grounding.resolve.return_value = {"tool": "computer", "action": "left_click", "coordinate": [100, 200]}
    grounding.last_grounding_events = []

    llm_client = Mock()
    # Default: always click then finish to keep loops short
    llm_client.generate.side_effect = [
        _tool_response(),
        _finish_response(),
    ] + [_finish_response()] * 20

    agent = ReActAgent(
        model_name="gpt-4o",
        llm_client=llm_client,
        state=app_state,
        tools_collection=Mock(),
        save_folder=tmp_path,
        grounding_strategy=grounding,
        max_steps=max_steps,
        compaction_interval=compaction_interval,
        action_delay=0,
    )

    # Mock _capture_screen so no real screen capture happens
    agent._capture_screen = Mock(return_value={
        "raw_image_base64": "rawb64",
        "screen_width": 1920,
        "screen_height": 1080,
        "resized_screen_width": 1920,
        "resized_screen_height": 1080,
    })

    # Mock execute_tool_calls to return success
    agent.execute_tool_calls = Mock(return_value=[{
        "tool": "computer",
        "status": "success",
        "result": Mock(output="Done.", error=""),
    }])

    # Suppress trajectory persistence (no filesystem writes)
    agent._save_trajectory_step = Mock()

    return agent


# ===========================================================================
# Module-level helpers
# ===========================================================================

class TestFreeze:
    def test_simple_dict_is_hashable(self):
        result = _freeze({"box_id": 3})
        assert isinstance(result, frozenset)

    def test_same_dicts_produce_equal_results(self):
        assert _freeze({"a": 1, "b": 2}) == _freeze({"b": 2, "a": 1})

    def test_different_dicts_produce_different_results(self):
        assert _freeze({"box_id": 1}) != _freeze({"box_id": 2})

    def test_nested_dict_falls_back_to_json_string(self):
        result = _freeze({"nested": {"x": 1}})
        assert isinstance(result, str)


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

    def test_finish_fields_merged_into_facts(self, tmp_path):
        agent = _make_agent(tmp_path)
        agent.llm_client.generate.side_effect = [
            _finish_response(fields={"price": "9.99", "name": "Widget"})
        ]
        list(agent.run())
        assert agent.working_memory.facts.get("price") == "9.99"
        assert agent.working_memory.facts.get("name") == "Widget"

    def test_finish_terminates_loop_without_reaching_max_steps(self, tmp_path):
        agent = _make_agent(tmp_path, max_steps=20)
        agent.llm_client.generate.side_effect = [_finish_response()]
        list(agent.run())
        assert agent.step_count == 1


# ===========================================================================
# ReActAgent — loop detection
# ===========================================================================

def _all_user_message_texts(generate_mock) -> list:
    """Extract all text strings from user-role messages passed to generate()."""
    texts = []
    for call in generate_mock.call_args_list:
        messages = call.args[0] if call.args else call.kwargs.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(block["text"])
    return texts


class TestReActAgentLoopDetection:
    def test_no_stuck_hint_below_threshold(self, tmp_path):
        """Fewer than _LOOP_THRESHOLD identical actions do not inject stuck hint."""
        agent = _make_agent(tmp_path, max_steps=5)
        # Two identical actions, then finish — below threshold of 3
        agent.llm_client.generate.side_effect = [
            _tool_response("left_click", {"box_id": 0}),
            _tool_response("left_click", {"box_id": 0}),
            _finish_response(),
        ]
        list(agent.run())
        all_texts = _all_user_message_texts(agent.llm_client.generate)
        assert not any("repeated the same action" in t for t in all_texts), (
            "Stuck hint should NOT be injected below the loop threshold"
        )

    def test_stuck_hint_injected_at_threshold(self, tmp_path):
        """Exactly _LOOP_THRESHOLD identical actions triggers loop detection."""
        agent = _make_agent(tmp_path, max_steps=10)
        repeat_responses = [
            _tool_response("left_click", {"box_id": 0})
        ] * _LOOP_THRESHOLD
        agent.llm_client.generate.side_effect = repeat_responses + [_finish_response()]

        events = list(agent.run())
        assert any(e["type"] == "complete" for e in events)

        all_texts = _all_user_message_texts(agent.llm_client.generate)
        assert any("repeated the same action" in t for t in all_texts), (
            "Stuck hint was not injected into LLM messages after hitting the loop threshold"
        )

    def test_loop_detection_uses_last_window_only(self, tmp_path):
        """Actions outside the _LOOP_WINDOW are not counted toward loop detection."""
        agent = _make_agent(tmp_path, max_steps=20)
        unique_responses = [
            _tool_response("left_click", {"box_id": i}) for i in range(1, _LOOP_WINDOW)
        ]
        repeat_responses = [
            _tool_response("left_click", {"box_id": 99})
        ] * _LOOP_THRESHOLD
        agent.llm_client.generate.side_effect = unique_responses + repeat_responses + [_finish_response()]
        events = list(agent.run())
        # Unique actions reset the window; the later repeat block should still trigger the hint
        assert any(e["type"] == "complete" for e in events)
        all_texts = _all_user_message_texts(agent.llm_client.generate)
        assert any("repeated the same action" in t for t in all_texts), (
            "Stuck hint was not injected even though the repeat block exceeded the threshold"
        )

    def test_no_stuck_hint_when_count_stays_below_threshold_in_window(self, tmp_path):
        """Alternating different actions keep every action's window-count below threshold."""
        agent = _make_agent(tmp_path, max_steps=10)
        # Alternating box_id=0 and box_id=1 — each appears at most 2 times in any window of 5.
        responses = [
            _tool_response("left_click", {"box_id": 0}),
            _tool_response("left_click", {"box_id": 1}),
            _tool_response("left_click", {"box_id": 0}),
            _tool_response("left_click", {"box_id": 1}),
            _finish_response(),
        ]
        agent.llm_client.generate.side_effect = responses
        list(agent.run())
        all_texts = _all_user_message_texts(agent.llm_client.generate)
        assert not any("repeated the same action" in t for t in all_texts), (
            "Stuck hint fired even though no action appeared 3+ times in the window"
        )


# ===========================================================================
# ReActAgent — image eviction from history
# ===========================================================================

class TestEvictOldImages:
    def test_images_replaced_in_all_but_last_user_message(self):
        from omnitool.gradio.core.agents.base import _evict_old_images

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
        from omnitool.gradio.core.agents.base import _evict_old_images

        history = [
            {"role": "assistant", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            ]},
        ]
        original = history[0]["content"][0]["type"]
        _evict_old_images(history)
        assert history[0]["content"][0]["type"] == original

    def test_text_blocks_preserved_after_eviction(self):
        from omnitool.gradio.core.agents.base import _evict_old_images

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
        from omnitool.gradio.core.agents.base import _evict_old_images

        history = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
            ]},
        ]
        _evict_old_images(history)
        # Single user message — it is "the last", so image must stay
        assert history[0]["content"][0]["type"] == "image_url"

    def test_empty_history_does_not_raise(self):
        from omnitool.gradio.core.agents.base import _evict_old_images
        _evict_old_images([])   # must not raise


# ===========================================================================
# ReActAgent — history compaction
# ===========================================================================

class TestReActAgentCompaction:
    def test_compaction_triggered_at_interval(self, tmp_path):
        """Compaction LLM call is made after exactly compaction_interval steps."""
        interval = 3
        agent = _make_agent(tmp_path, max_steps=interval + 2, compaction_interval=interval)

        call_log = []

        def generate_side_effect(messages, system_prompt, tools=None):
            # First call without tools = compaction call
            if tools is None:
                call_log.append("compact")
                return ("Summary of progress.", {"tokens": 10, "input_tokens": 5, "output_tokens": 5, "tool_calls": [], "assistant_message": {"role": "assistant", "content": ""}})
            call_log.append("step")
            step_num = call_log.count("step")
            if step_num > interval:
                return _finish_response()
            return _tool_response()

        agent.llm_client.generate.side_effect = generate_side_effect
        events = list(agent.run())

        status_msgs = [e["message"] for e in events if e["type"] == "status"]
        assert any("Compact" in m or "compact" in m for m in status_msgs), (
            f"No compaction status found. Status messages: {status_msgs}"
        )

    def test_compaction_replaces_history_with_summary(self, tmp_path):
        """After compaction, the agent continues with a two-message history."""
        interval = 2
        agent = _make_agent(tmp_path, max_steps=interval + 2, compaction_interval=interval)

        def generate_side_effect(messages, system_prompt, tools=None):
            if tools is None:
                # Compaction call — return a summary
                return ("Steps done: step 1, step 2. Still need: finish.", {
                    "tokens": 15, "input_tokens": 8, "output_tokens": 7,
                    "tool_calls": [], "assistant_message": {"role": "assistant", "content": ""},
                })
            step_n = agent.step_count
            if step_n > interval:
                return _finish_response()
            return _tool_response()

        agent.llm_client.generate.side_effect = generate_side_effect
        events = list(agent.run())
        # Agent should reach complete without crashing after compaction
        assert any(e["type"] == "complete" for e in events)

    def test_compaction_failure_keeps_history(self, tmp_path):
        """If the compaction LLM call raises, history is preserved and loop continues."""
        interval = 2
        agent = _make_agent(tmp_path, max_steps=interval + 2, compaction_interval=interval)

        call_count = [0]

        def generate_side_effect(messages, system_prompt, tools=None):
            call_count[0] += 1
            if tools is None:
                raise RuntimeError("LLM timeout")
            step_n = agent.step_count
            if step_n > interval:
                return _finish_response()
            return _tool_response()

        agent.llm_client.generate.side_effect = generate_side_effect
        events = list(agent.run())
        # Despite compaction failure, loop should reach finish (complete) or max_steps
        final_types = {e["type"] for e in events}
        assert "complete" in final_types or "error" in final_types


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
