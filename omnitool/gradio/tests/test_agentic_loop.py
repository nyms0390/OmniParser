"""
Tests for the agentic loop refactor — runs without any real model or OCR access.

All external dependencies (LLM, OmniParser, screen capture) are mocked.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from omnitool.gradio.config import AgentMode
from omnitool.gradio.config.prompts import PLAN_PROMPT, REFLECT_PROMPT, TASK_PARSE_PROMPT
from omnitool.gradio.core.checklist import Checklist, ChecklistItem
from omnitool.gradio.services import AppState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app_state(tmp_path):
    state = AppState(run_folder=tmp_path)
    state.chat.add_message("user", "Open Notepad and type hello world")
    return state


@pytest.fixture
def mock_agent(tmp_path):
    """Fully mocked BaseAgent."""
    agent = Mock()
    agent.save_folder = tmp_path / "execution"
    agent.save_folder.mkdir(exist_ok=True)
    agent.total_tokens = 0
    agent.total_cost = 0.0

    # Default plan response — one tool call
    agent.plan.return_value = {
        "response_text": '{"Reasoning": "click start", "Next Action": "left_click", "Box ID": 0}',
        "tool_calls": [{"tool": "computer", "action": "left_click", "coordinate": [100, 200]}],
        "metadata": {"tokens": 50},
        "cost": 0.001,
    }
    agent.execute_tool_calls.return_value = [
        {"tool": "computer", "status": "success", "result": Mock(output="ok", base64_image="", error="")}
    ]
    agent.update_token_usage = Mock()
    agent.update_step_count = Mock()
    agent.update_cost = Mock()
    agent._extract_data = Mock(side_effect=lambda text, fmt: text)

    # LLM client used by _generate_plan / _reflect
    agent.llm_client = Mock()
    return agent


@pytest.fixture
def mock_screen():
    return {
        "som_image_base64": "fakeb64",
        "parsed_content_list": [{"bbox": [0.1, 0.1, 0.3, 0.2], "content": "Start"}],
        "screen_width": 1920,
        "screen_height": 1080,
    }


def _make_orchestrator(app_state, mock_agent, mock_screen, mode=AgentMode.INTERACTIVE, max_steps=3):
    """Build a SamplingOrchestrator with all external calls mocked out."""
    from omnitool.gradio.core.orchestrator import SamplingOrchestrator

    with patch("omnitool.gradio.core.orchestrator.create_agent", return_value=mock_agent):
        orch = SamplingOrchestrator(
            model_name="omniparser + gpt-4o",
            state=app_state,
            tools_collection=Mock(),
            omniparser_client=Mock(),
            max_steps=max_steps,
            mode=mode,
        )

    orch._capture_screen = Mock(return_value=mock_screen)
    return orch


# ===========================================================================
# Checklist unit tests
# ===========================================================================

class TestChecklistItem:
    def test_defaults(self):
        item = ChecklistItem(id=1, step="Do something")
        assert item.status == "pending"
        assert item.verification_hint == ""

    def test_to_dict_round_trip(self):
        item = ChecklistItem(id=2, step="Click OK", status="done", verification_hint="Dialog closed")
        d = item.to_dict()
        assert d == {"id": 2, "step": "Click OK", "status": "done", "verification_hint": "Dialog closed"}


class TestChecklistFromUserText:
    def test_numbered_list(self):
        c = Checklist.from_user_text("1. Open browser\n2. Navigate to site\n3. Login")
        assert len(c.items) == 3
        assert c.items[0].step == "Open browser"
        assert c.items[2].id == 3

    def test_numbered_list_parenthesis(self):
        c = Checklist.from_user_text("1) First\n2) Second")
        assert [i.step for i in c.items] == ["First", "Second"]

    def test_bullet_dash(self):
        c = Checklist.from_user_text("- step one\n- step two\n- step three")
        assert len(c.items) == 3
        assert c.items[1].step == "step two"

    def test_bullet_star(self):
        c = Checklist.from_user_text("* alpha\n* beta")
        assert [i.step for i in c.items] == ["alpha", "beta"]

    def test_single_item_fallback(self):
        c = Checklist.from_user_text("just do the thing")
        assert len(c.items) == 1
        assert c.items[0].step == "just do the thing"

    def test_ignores_blank_lines(self):
        c = Checklist.from_user_text("1. First\n\n2. Second\n\n3. Third")
        assert len(c.items) == 3


class TestChecklistFromLLMJson:
    def test_valid_array(self):
        raw = json.dumps([
            {"id": 1, "step": "Open browser", "verification_hint": "Browser visible"},
            {"id": 2, "step": "Login", "verification_hint": "Dashboard shown"},
        ])
        c = Checklist.from_llm_json(raw)
        assert len(c.items) == 2
        assert c.items[0].verification_hint == "Browser visible"

    def test_strips_code_fence(self):
        raw = "```json\n[{\"id\":1,\"step\":\"A\",\"verification_hint\":\"\"}]\n```"
        c = Checklist.from_llm_json(raw)
        assert c.items[0].step == "A"

    def test_fallback_on_invalid_json(self):
        c = Checklist.from_llm_json("not json at all")
        assert len(c.items) == 1
        assert c.items[0].step == "not json at all"

    def test_missing_step_field_skipped(self):
        raw = json.dumps([{"id": 1}, {"id": 2, "step": "Valid"}])
        c = Checklist.from_llm_json(raw)
        assert len(c.items) == 1
        assert c.items[0].step == "Valid"


class TestChecklistMutations:
    def test_apply_updates_status(self):
        c = Checklist.from_user_text("1. A\n2. B\n3. C")
        c.apply_updates([{"id": 1, "status": "done"}, {"id": 3, "status": "in_progress"}])
        assert c.items[0].status == "done"
        assert c.items[1].status == "pending"
        assert c.items[2].status == "in_progress"

    def test_apply_updates_unknown_id_ignored(self):
        c = Checklist.from_user_text("1. A")
        c.apply_updates([{"id": 99, "status": "done"}])
        assert c.items[0].status == "pending"  # unchanged

    def test_all_done_false_when_pending(self):
        c = Checklist.from_user_text("1. A\n2. B")
        c.apply_updates([{"id": 1, "status": "done"}])
        assert not c.all_done()

    def test_all_done_true(self):
        c = Checklist.from_user_text("1. A\n2. B")
        c.apply_updates([{"id": 1, "status": "done"}, {"id": 2, "status": "skipped"}])
        assert c.all_done()

    def test_all_done_empty(self):
        assert not Checklist().all_done()

    def test_get_active_returns_first_pending(self):
        c = Checklist.from_user_text("1. A\n2. B\n3. C")
        c.apply_updates([{"id": 1, "status": "done"}])
        active = c.get_active()
        assert active is not None
        assert active.id == 2

    def test_get_active_none_when_all_done(self):
        c = Checklist.from_user_text("1. A")
        c.apply_updates([{"id": 1, "status": "done"}])
        assert c.get_active() is None


class TestChecklistToPromptText:
    def test_includes_symbols(self):
        c = Checklist.from_user_text("1. A\n2. B")
        c.apply_updates([{"id": 1, "status": "done"}])
        text = c.to_prompt_text()
        assert "✅" in text
        assert "⬜" in text
        assert "[1]" in text
        assert "A" in text

    def test_includes_hint(self):
        c = Checklist.from_llm_json('[{"id":1,"step":"X","verification_hint":"check Y"}]')
        text = c.to_prompt_text()
        assert "check Y" in text


# ===========================================================================
# Prompt template tests
# ===========================================================================

class TestPromptTemplates:
    def test_plan_prompt_format(self):
        out = PLAN_PROMPT.format(task="open notepad")
        assert "open notepad" in out
        assert "verification_hint" in out
        assert "id" in out

    def test_reflect_prompt_format(self):
        out = REFLECT_PROMPT.format(
            task="test task",
            checklist_section="Checklist:\n  ⬜ [1] Step one",
            recent_actions="  Step 1: left_click at [100,200]",
        )
        assert "test task" in out
        assert "checklist_updates" in out
        assert "is_request_satisfied" in out
        assert "screen_description" in out

    def test_task_parse_prompt_format(self):
        out = TASK_PARSE_PROMPT.format(user_text="do X then Y")
        assert "do X then Y" in out
        assert "verification_hint" in out


# ===========================================================================
# Orchestrator — INTERACTIVE mode
# ===========================================================================

class TestOrchestratorInteractive:
    def test_yields_step_events(self, app_state, mock_agent, mock_screen):
        orch = _make_orchestrator(app_state, mock_agent, mock_screen, AgentMode.INTERACTIVE, max_steps=2)
        events = list(orch.sampling_loop())
        types = [e["type"] for e in events]
        assert "status" in types
        assert "step" in types
        assert "action_result" in types

    def test_stops_when_no_tool_calls(self, app_state, mock_agent, mock_screen):
        mock_agent.plan.return_value = {
            "response_text": "Task is done.",
            "tool_calls": [],
            "metadata": {"tokens": 10},
            "cost": 0.0,
        }
        orch = _make_orchestrator(app_state, mock_agent, mock_screen, AgentMode.INTERACTIVE, max_steps=5)
        events = list(orch.sampling_loop())
        types = [e["type"] for e in events]
        assert "assistant_reply" in types
        # Should stop after step 1
        step_events = [e for e in events if e["type"] == "step"]
        assert len(step_events) == 1

    def test_no_reflect_in_interactive_mode(self, app_state, mock_agent, mock_screen):
        orch = _make_orchestrator(app_state, mock_agent, mock_screen, AgentMode.INTERACTIVE, max_steps=2)
        list(orch.sampling_loop())
        # LLM client on agent should not be called for reflect in interactive mode
        assert mock_agent.llm_client.generate.call_count == 0

    def test_repeated_action_detection_stops_loop(self, app_state, mock_agent, mock_screen):
        orch = _make_orchestrator(app_state, mock_agent, mock_screen, AgentMode.INTERACTIVE, max_steps=20)
        # Pre-populate trajectory with 4 identical actions so 5th triggers detection
        for i in range(4):
            orch.trajectory.append({
                "step": i + 1,
                "tool_calls": [{"action": "left_click", "coordinate": [100, 200]}],
            })
        orch.step_count = 4
        events = list(orch.sampling_loop())
        types = [e["type"] for e in events]
        assert "assistant_reply" in types
        # Stopped early — not 20 steps
        step_events = [e for e in events if e["type"] == "step"]
        assert len(step_events) <= 2

    def test_complete_event_always_yielded(self, app_state, mock_agent, mock_screen):
        mock_agent.plan.return_value = {"response_text": "done", "tool_calls": [], "metadata": {"tokens": 5}, "cost": 0.0}
        orch = _make_orchestrator(app_state, mock_agent, mock_screen)
        events = list(orch.sampling_loop())
        assert events[-1]["type"] == "complete"

    def test_error_event_on_exception(self, app_state, mock_agent, mock_screen):
        orch = _make_orchestrator(app_state, mock_agent, mock_screen)
        orch._capture_screen = Mock(side_effect=RuntimeError("camera broken"))
        events = list(orch.sampling_loop())
        assert any(e["type"] == "error" for e in events)


# ===========================================================================
# Orchestrator — ORCHESTRATED mode
# ===========================================================================

PLAN_JSON = json.dumps([
    {"id": 1, "step": "Open Notepad", "verification_hint": "Notepad window visible"},
    {"id": 2, "step": "Type hello world", "verification_hint": "Text appears in editor"},
])

REFLECT_JSON = json.dumps({
    "screen_description": "Desktop with Notepad open",
    "is_request_satisfied": {"reason": "not done yet", "answer": False},
    "is_in_loop": {"reason": "no loop", "answer": False},
    "is_progress_being_made": {"reason": "opened notepad", "answer": True},
    "instruction_or_question": {"reason": "", "answer": "Type hello world"},
    "checklist_updates": [{"id": 1, "status": "done"}],
})


class TestOrchestratorOrchestrated:
    def test_plan_event_yielded(self, app_state, mock_agent, mock_screen):
        mock_agent.llm_client.generate.return_value = (PLAN_JSON, {"tokens": 80})
        mock_agent._extract_data.side_effect = lambda text, fmt: text
        orch = _make_orchestrator(app_state, mock_agent, mock_screen, AgentMode.ORCHESTRATED, max_steps=1)
        events = list(orch.sampling_loop())
        plan_events = [e for e in events if e["type"] == "plan"]
        assert len(plan_events) == 1
        assert "checklist" in plan_events[0]

    def test_checklist_built_from_plan(self, app_state, mock_agent, mock_screen):
        mock_agent.llm_client.generate.return_value = (PLAN_JSON, {"tokens": 80})
        mock_agent._extract_data.side_effect = lambda text, fmt: text
        orch = _make_orchestrator(app_state, mock_agent, mock_screen, AgentMode.ORCHESTRATED, max_steps=1)
        list(orch.sampling_loop())
        assert orch.checklist is not None
        assert len(orch.checklist.items) == 2

    def test_reflect_skipped_on_step_1(self, app_state, mock_agent, mock_screen):
        # Only one generate call expected: the plan call. Reflect should not fire on step 1.
        mock_agent.llm_client.generate.return_value = (PLAN_JSON, {"tokens": 80})
        mock_agent._extract_data.side_effect = lambda text, fmt: text
        orch = _make_orchestrator(app_state, mock_agent, mock_screen, AgentMode.ORCHESTRATED, max_steps=1)
        list(orch.sampling_loop())
        # Only the plan LLM call, no reflect call on step 1
        assert mock_agent.llm_client.generate.call_count == 1

    def test_reflect_fires_on_step_2(self, app_state, mock_agent, mock_screen):
        # Step 1: plan call. Step 2: plan + reflect.
        call_count = [0]
        def side_effect(messages, system_prompt):
            call_count[0] += 1
            if call_count[0] == 1:  # plan
                return (PLAN_JSON, {"tokens": 80})
            return (REFLECT_JSON, {"tokens": 60})

        mock_agent.llm_client.generate.side_effect = side_effect
        mock_agent._extract_data.side_effect = lambda text, fmt: text
        orch = _make_orchestrator(app_state, mock_agent, mock_screen, AgentMode.ORCHESTRATED, max_steps=2)
        events = list(orch.sampling_loop())
        ledger_events = [e for e in events if e["type"] == "ledger"]
        assert len(ledger_events) == 1  # Only on step 2

    def test_reflect_updates_checklist_statuses(self, app_state, mock_agent, mock_screen):
        call_count = [0]
        def side_effect(messages, system_prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return (PLAN_JSON, {"tokens": 80})
            return (REFLECT_JSON, {"tokens": 60})

        mock_agent.llm_client.generate.side_effect = side_effect
        mock_agent._extract_data.side_effect = lambda text, fmt: text
        orch = _make_orchestrator(app_state, mock_agent, mock_screen, AgentMode.ORCHESTRATED, max_steps=2)
        list(orch.sampling_loop())
        assert orch.checklist is not None
        assert orch.checklist.items[0].status == "done"  # updated by REFLECT_JSON

    def test_initial_screen_reused_on_step_1(self, app_state, mock_agent, mock_screen):
        mock_agent.llm_client.generate.return_value = (PLAN_JSON, {"tokens": 80})
        mock_agent._extract_data.side_effect = lambda text, fmt: text
        orch = _make_orchestrator(app_state, mock_agent, mock_screen, AgentMode.ORCHESTRATED, max_steps=1)
        list(orch.sampling_loop())
        # _capture_screen called once (for init), not again on step 1
        assert orch._capture_screen.call_count == 1

    def test_task_complete_when_all_checklist_done(self, app_state, mock_agent, mock_screen):
        all_done_reflect = json.dumps({
            "screen_description": "Done",
            "is_request_satisfied": {"reason": "complete", "answer": False},
            "is_in_loop": {"reason": "", "answer": False},
            "is_progress_being_made": {"reason": "", "answer": True},
            "instruction_or_question": {"reason": "", "answer": ""},
            "checklist_updates": [
                {"id": 1, "status": "done"},
                {"id": 2, "status": "done"},
            ],
        })
        call_count = [0]
        def side_effect(messages, system_prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return (PLAN_JSON, {"tokens": 80})
            return (all_done_reflect, {"tokens": 60})

        mock_agent.llm_client.generate.side_effect = side_effect
        mock_agent._extract_data.side_effect = lambda text, fmt: text
        orch = _make_orchestrator(app_state, mock_agent, mock_screen, AgentMode.ORCHESTRATED, max_steps=5)
        events = list(orch.sampling_loop())
        reply_events = [e for e in events if e["type"] == "assistant_reply"]
        assert len(reply_events) == 1
        # Only reached step 2 (plan on 1, reflect+stop on 2)
        step_events = [e for e in events if e["type"] == "step"]
        assert len(step_events) == 2


# ===========================================================================
# Orchestrator — TASK mode
# ===========================================================================

class TestOrchestratorTask:
    def test_task_mode_no_longer_errors(self, app_state, mock_agent, mock_screen):
        mock_agent.llm_client.generate.return_value = (REFLECT_JSON, {"tokens": 60})
        mock_agent._extract_data.side_effect = lambda text, fmt: text
        orch = _make_orchestrator(app_state, mock_agent, mock_screen, AgentMode.TASK, max_steps=1)
        events = list(orch.sampling_loop())
        assert not any(e.get("message", "").startswith("Task mode is not yet") for e in events if e["type"] == "error")

    def test_task_checklist_from_numbered_message(self, app_state, mock_agent, mock_screen):
        app_state.chat.messages[0]["content"] = "1. Open Notepad\n2. Type hello\n3. Save file"
        orch = _make_orchestrator(app_state, mock_agent, mock_screen, AgentMode.TASK, max_steps=1)
        events = list(orch.sampling_loop())
        plan_events = [e for e in events if e["type"] == "plan"]
        assert len(plan_events) == 1
        assert len(plan_events[0]["checklist"]) == 3

    def test_task_plan_event_before_first_step(self, app_state, mock_agent, mock_screen):
        app_state.chat.messages[0]["content"] = "1. Step A\n2. Step B"
        orch = _make_orchestrator(app_state, mock_agent, mock_screen, AgentMode.TASK, max_steps=1)
        events = list(orch.sampling_loop())
        indices = {e["type"]: i for i, e in enumerate(events)}
        # plan event must come before first step event
        assert indices["plan"] < indices["step"]

    def test_task_llm_fallback_for_prose(self, app_state, mock_agent, mock_screen):
        """Multi-line prose with no list structure triggers LLM fallback parse."""
        parsed_checklist = json.dumps([
            {"id": 1, "step": "Open browser", "verification_hint": "Browser open"},
            {"id": 2, "step": "Search for cats", "verification_hint": "Results shown"},
        ])
        app_state.chat.messages[0]["content"] = (
            "I need you to do a few things for me.\n"
            "Please open the browser first,\n"
            "then search for cats on Google."
        )
        mock_agent.llm_client.generate.return_value = (parsed_checklist, {"tokens": 40})
        mock_agent._extract_data.side_effect = lambda text, fmt: text
        orch = _make_orchestrator(app_state, mock_agent, mock_screen, AgentMode.TASK, max_steps=1)
        events = list(orch.sampling_loop())
        plan_events = [e for e in events if e["type"] == "plan"]
        # LLM fallback should have produced 2 items
        assert len(plan_events[0]["checklist"]) == 2


# ===========================================================================
# VLMAgent — message preparation and tool-call parsing
# ===========================================================================

class TestVLMAgentPrepareMessages:
    def _make_agent(self, tmp_path):
        from omnitool.gradio.core.agents.vlm import VLMAgent
        llm_client = Mock()
        state = AppState(run_folder=tmp_path)
        tools = Mock()
        with patch("omnitool.gradio.core.agents.vlm.get_model_config", return_value={}):
            agent = VLMAgent("omniparser + gpt-4o", llm_client, state, tools, tmp_path)
        return agent

    def test_som_image_attached_when_available(self, tmp_path):
        agent = self._make_agent(tmp_path)
        messages = [{"role": "user", "content": "do task"}]
        parsed_screen = {
            "som_image_base64": "sombase64data",
            "parsed_content_list": [],
            "screen_description": "",
        }
        prepared = agent._prepare_messages(messages, parsed_screen)
        last = prepared[-1]
        # Content should be a list (multipart) with an image_url entry
        assert isinstance(last["content"], list)
        types = [part.get("type") for part in last["content"]]
        assert "image_url" in types

    def test_text_fallback_when_no_som(self, tmp_path):
        agent = self._make_agent(tmp_path)
        messages = [{"role": "user", "content": "do task"}]
        parsed_screen = {
            "som_image_base64": "",
            "parsed_content_list": [{"content": "button"}],
            "screen_description": "",
        }
        prepared = agent._prepare_messages(messages, parsed_screen)
        last = prepared[-1]
        assert isinstance(last["content"], str)
        assert "screen_elements" in last["content"]

    def test_screen_description_injected(self, tmp_path):
        agent = self._make_agent(tmp_path)
        messages = [{"role": "user", "content": "do task"}]
        parsed_screen = {
            "som_image_base64": "",
            "parsed_content_list": [],
            "screen_description": "The desktop is visible",
        }
        prepared = agent._prepare_messages(messages, parsed_screen)
        last_content = prepared[-1]["content"]
        assert "The desktop is visible" in last_content


class TestVLMAgentParseToolCalls:
    def _make_agent(self, tmp_path):
        from omnitool.gradio.core.agents.vlm import VLMAgent
        with patch("omnitool.gradio.core.agents.vlm.get_model_config", return_value={}):
            return VLMAgent("test", Mock(), AppState(run_folder=tmp_path), Mock(), tmp_path)

    def test_left_click_with_box_id(self, tmp_path):
        agent = self._make_agent(tmp_path)
        response = '```json\n{"Reasoning": "click", "Next Action": "left_click", "Box ID": 0}\n```'
        parsed_screen = {
            "parsed_content_list": [{"bbox": [0.0, 0.0, 0.2, 0.1]}],
            "screen_width": 1000,
            "screen_height": 1000,
        }
        calls = agent._parse_tool_calls(response, parsed_screen)
        actions = [c["action"] for c in calls]
        assert "mouse_move" in actions
        assert "left_click" in actions

    def test_type_action(self, tmp_path):
        agent = self._make_agent(tmp_path)
        response = '{"Next Action": "type", "value": "hello world"}'
        calls = agent._parse_tool_calls(response, {"parsed_content_list": [], "screen_width": 1920, "screen_height": 1080})
        assert any(c["action"] == "type" and c.get("text") == "hello world" for c in calls)

    def test_none_action_returns_empty(self, tmp_path):
        agent = self._make_agent(tmp_path)
        response = '{"Next Action": "None", "Reasoning": "task complete"}'
        calls = agent._parse_tool_calls(response, {"parsed_content_list": [], "screen_width": 1920, "screen_height": 1080})
        assert calls == []

    def test_invalid_json_returns_empty(self, tmp_path):
        agent = self._make_agent(tmp_path)
        calls = agent._parse_tool_calls("not json", {"parsed_content_list": [], "screen_width": 1920, "screen_height": 1080})
        assert calls == []

    def test_invalid_box_id_skips_coordinate(self, tmp_path):
        agent = self._make_agent(tmp_path)
        response = '{"Next Action": "left_click", "Box ID": 999}'
        calls = agent._parse_tool_calls(response, {"parsed_content_list": [], "screen_width": 1920, "screen_height": 1080})
        # Should still produce a left_click, just no mouse_move (no coordinate)
        actions = [c["action"] for c in calls]
        assert "mouse_move" not in actions
        assert "left_click" in actions


# ===========================================================================
# Orchestrator — _verify_step (no LLM)
# ===========================================================================

class TestVerifyStep:
    def test_no_error_no_repeat(self, app_state, mock_agent, mock_screen):
        orch = _make_orchestrator(app_state, mock_agent, mock_screen)
        tool_results = [
            {"status": "success", "result": Mock(output="ok", error="")}
        ]
        result = orch._verify_step(tool_results)
        assert not result["has_error"]
        assert not result["is_repeated"]

    def test_detects_tool_error(self, app_state, mock_agent, mock_screen):
        orch = _make_orchestrator(app_state, mock_agent, mock_screen)
        tool_results = [{"status": "error", "error": "tool crashed"}]
        result = orch._verify_step(tool_results)
        assert result["has_error"]
        assert "tool crashed" in result["error_detail"]

    def test_detects_repeated_action(self, app_state, mock_agent, mock_screen):
        orch = _make_orchestrator(app_state, mock_agent, mock_screen)
        for i in range(5):
            orch.trajectory.append({
                "step": i + 1,
                "tool_calls": [{"action": "left_click", "coordinate": [100, 200]}],
            })
        result = orch._verify_step([])
        assert result["is_repeated"]
