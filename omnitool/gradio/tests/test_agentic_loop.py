"""
Tests for the agentic loop — VLMAgent + Plan→Reflect, runs without any real model or OCR.

All external dependencies (LLM, OmniParser, screen capture) are mocked.
"""

import json
from unittest.mock import Mock

import pytest

from omnitool.gradio.config import AgentMode
from omnitool.gradio.config.prompts import CHECKLIST_GEN_PROMPT, REFLECT_PROMPT
from omnitool.gradio.core.agents.checklist import Checklist, ChecklistItem
from omnitool.gradio.services import AppState


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

PLAN_JSON = json.dumps([
    {"id": 1, "step": "Open Notepad", "verification_hint": "Notepad window visible"},
    {"id": 2, "step": "Type hello world", "verification_hint": "Text appears in editor"},
])

REFLECT_JSON = json.dumps({
    "screen_observation": "Notepad is open.",
    "is_request_satisfied": {"reason": "not done yet", "answer": False},
    "is_in_loop": {"reason": "no loop", "answer": False},
    "checklist_updates": [{"id": 1, "status": "done"}],
})

REFLECT_DONE_JSON = json.dumps({
    "screen_observation": "Task complete.",
    "is_request_satisfied": {"reason": "all done", "answer": True},
    "is_in_loop": {"reason": "", "answer": False},
    "checklist_updates": [
        {"id": 1, "status": "done"},
        {"id": 2, "status": "done"},
    ],
})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app_state(tmp_path):
    state = AppState(run_folder=tmp_path)
    state.chat.add_message("user", "Open Notepad and type hello world")
    return state


def _make_screen_data():
    """Return a minimal ScreenData object."""
    from omnitool.gradio.core.agents.grounding import ScreenData
    return ScreenData(
        raw_image_b64="fakeraw",
        display_image_b64="fakesom",
        elements=[{"bbox": [0.1, 0.1, 0.3, 0.2], "content": "Start", "box_id": 0}],
        screen_width=1920,
        screen_height=1080,
        resized_width=1920,
        resized_height=1080,
    )


def _tool_response(tool_name="left_click", args=None, text="Planning action."):
    """Build a metadata dict with one tool call, as returned by llm_client.generate()."""
    if args is None:
        args = {"box_id": 0}
    tc_id = "call_1"
    return (
        text,
        {
            "tokens": 50,
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


def _empty_response(text="Task is done."):
    """Build a metadata dict with no tool calls (LLM signals stop)."""
    return (
        text,
        {
            "tokens": 10,
            "tool_calls": [],
            "assistant_message": {"role": "assistant", "content": text},
        },
    )


def _make_orchestrator(app_state, mode=AgentMode.INTERACTIVE, max_steps=3):
    """Build a VLMAgent with all external calls mocked out."""
    from omnitool.gradio.core.agents import VLMAgent

    llm_client = Mock()
    llm_client.generate.side_effect = [
        _tool_response(),
        _empty_response(),
        _empty_response(),
        _empty_response(),
        _empty_response(),
    ]

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
    grounding.resolve.return_value = {"action": "left_click", "coordinate": [100, 200]}
    grounding.last_grounding_events = []

    screen_data = _make_screen_data()

    orch = VLMAgent(
        model_name="gpt-4o",
        llm_client=llm_client,
        state=app_state,
        tools_collection=Mock(),
        save_folder=app_state.session.run_folder,
        grounding_strategy=grounding,
        max_steps=max_steps,
        mode=mode,
    )

    # _do_capture must update working_memory (VLMAgent.run() reads from it)
    def do_capture():
        orch.working_memory.screen_data = screen_data
        orch.working_memory.parsed_screen = {
            "resized_image_base64": screen_data.raw_image_b64,
            "som_image_base64": screen_data.display_image_b64,
            "screen_width": screen_data.screen_width,
            "screen_height": screen_data.screen_height,
            "resized_screen_width": screen_data.resized_width,
            "resized_screen_height": screen_data.resized_height,
            "parsed_content_list": screen_data.elements,
        }
        return screen_data

    orch._do_capture = Mock(side_effect=do_capture)

    # Bypass checklist generation by default — tests that need it override this
    orch._generate_checklist = Mock(return_value=Checklist.from_llm_json(PLAN_JSON))

    # Mock execute_tool_calls to return success
    orch.execute_tool_calls = Mock(return_value=[{
        "tool": "computer",
        "status": "success",
        "result": Mock(output="ok", error=""),
    }])

    # Mock _reflect to set a default ledger; orchestrated tests may override
    def fake_reflect(**kwargs):
        orch.working_memory.ledger = REFLECT_JSON

    orch._reflect = Mock(side_effect=fake_reflect)

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

    def test_unwraps_steps_object(self):
        raw = json.dumps({
            "steps": [
                {"id": 1, "step": "Open browser", "verification_hint": "Browser visible"},
                {"id": 2, "step": "Login", "verification_hint": "Dashboard shown"},
            ]
        })
        c = Checklist.from_llm_json(raw)
        assert len(c.items) == 2
        assert c.items[0].step == "Open browser"
        assert c.items[1].verification_hint == "Dashboard shown"

    def test_dict_without_steps_key_falls_back(self):
        raw = json.dumps({"something_else": "value"})
        c = Checklist.from_llm_json(raw)
        assert len(c.items) == 1
        assert c.items[0].step == raw


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
        assert c.items[0].status == "pending"

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
    def test_checklist_gen_prompt_format(self):
        out = CHECKLIST_GEN_PROMPT.format(task="open notepad")
        assert "open notepad" in out
        assert "verification_hint" in out
        assert "id" in out

    def test_reflect_prompt_format(self):
        out = REFLECT_PROMPT.format(
            working_memory_section="Checklist:\n  ⬜ [1] Step one\n",
            active_step_section="step [1]: click Save (verify when done: Save dialog closes)\n\n",
        )
        assert "step [1]" in out
        assert "SCREEN OBSERVATION" in out
        assert "CHECKLIST UPDATE" in out
        assert "checklist_updates" in out
        assert "is_request_satisfied" in out
        assert "is_progress_being_made" not in out
        assert "instruction_or_question" not in out
        assert "screen_description" not in out


# ===========================================================================
# VLMAgent — INTERACTIVE mode
# ===========================================================================

class TestOrchestratorInteractive:
    def test_yields_step_events(self, app_state):
        orch = _make_orchestrator(app_state, AgentMode.INTERACTIVE, max_steps=2)
        events = list(orch.run())
        types = [e["type"] for e in events]
        assert "status" in types
        assert "step" in types
        assert "action_result" in types

    def test_stops_when_no_tool_calls(self, app_state):
        orch = _make_orchestrator(app_state, AgentMode.INTERACTIVE, max_steps=5)
        orch.llm_client.generate.side_effect = [_empty_response()]
        events = list(orch.run())
        types = [e["type"] for e in events]
        assert "assistant_reply" in types
        assert len([e for e in events if e["type"] == "step"]) == 1

    def test_no_reflect_in_interactive_mode(self, app_state):
        orch = _make_orchestrator(app_state, AgentMode.INTERACTIVE, max_steps=2)
        list(orch.run())
        orch._reflect.assert_not_called()

    def test_repeated_action_detection_stops_loop(self, app_state):
        orch = _make_orchestrator(app_state, AgentMode.INTERACTIVE, max_steps=20)
        # Pre-populate trajectory with 4 identical actions so 5th triggers detection
        for i in range(4):
            orch.working_memory.trajectory.append({
                "step": i + 1,
                "tool_calls": [{"action": "left_click", "coordinate": [100, 200]}],
            })
        orch.step_count = 4
        orch.llm_client.generate.side_effect = [_tool_response()] * 20
        events = list(orch.run())
        types = [e["type"] for e in events]
        assert "assistant_reply" in types
        assert len([e for e in events if e["type"] == "step"]) <= 2

    def test_complete_event_always_yielded(self, app_state):
        orch = _make_orchestrator(app_state, AgentMode.INTERACTIVE)
        orch.llm_client.generate.side_effect = [_empty_response()]
        events = list(orch.run())
        assert events[-1]["type"] == "complete"

    def test_complete_event_has_required_fields(self, app_state):
        orch = _make_orchestrator(app_state, AgentMode.INTERACTIVE)
        orch.llm_client.generate.side_effect = [_empty_response()]
        complete = list(orch.run())[-1]
        assert complete["type"] == "complete"
        assert "message" in complete
        assert "success" in complete
        assert "total_steps" in complete
        assert "total_tokens" in complete

    def test_error_event_on_exception(self, app_state):
        orch = _make_orchestrator(app_state, AgentMode.INTERACTIVE)
        orch._do_capture = Mock(side_effect=RuntimeError("camera broken"))
        events = list(orch.run())
        assert any(e["type"] == "error" for e in events)


# ===========================================================================
# VLMAgent — ORCHESTRATED mode
# ===========================================================================

class TestOrchestratorOrchestrated:
    def test_plan_event_yielded(self, app_state):
        orch = _make_orchestrator(app_state, AgentMode.ORCHESTRATED, max_steps=1)
        events = list(orch.run())
        plan_events = [e for e in events if e["type"] == "plan"]
        assert len(plan_events) == 1
        assert "checklist" in plan_events[0]

    def test_checklist_built_from_plan(self, app_state):
        orch = _make_orchestrator(app_state, AgentMode.ORCHESTRATED, max_steps=1)
        list(orch.run())
        assert orch.working_memory.checklist is not None
        assert len(orch.working_memory.checklist.items) == 2

    def test_reflect_fires_on_step_1(self, app_state):
        orch = _make_orchestrator(app_state, AgentMode.ORCHESTRATED, max_steps=1)
        list(orch.run())
        orch._reflect.assert_called_once()

    def test_reflect_fires_every_step(self, app_state):
        orch = _make_orchestrator(app_state, AgentMode.ORCHESTRATED, max_steps=2)
        orch.llm_client.generate.side_effect = [_tool_response(), _tool_response()]
        list(orch.run())
        assert orch._reflect.call_count == 2

    def test_reflect_updates_checklist_statuses(self, app_state):
        orch = _make_orchestrator(app_state, AgentMode.ORCHESTRATED, max_steps=1)

        def fake_reflect_with_update(**kwargs):
            orch.working_memory.ledger = REFLECT_JSON
            orch.working_memory.checklist.apply_updates([{"id": 1, "status": "done"}])

        orch._reflect = Mock(side_effect=fake_reflect_with_update)
        list(orch.run())
        assert orch.working_memory.checklist.items[0].status == "done"

    def test_capture_called_each_step(self, app_state):
        orch = _make_orchestrator(app_state, AgentMode.ORCHESTRATED, max_steps=2)
        list(orch.run())
        # 1 pre-loop + 1 observe/step + 1 post-action/step = at least 3 for max_steps=2
        assert orch._do_capture.call_count >= 3

    def test_task_complete_when_satisfied_and_all_done(self, app_state):
        orch = _make_orchestrator(app_state, AgentMode.ORCHESTRATED, max_steps=5)

        def fake_reflect_done(**kwargs):
            orch.working_memory.ledger = REFLECT_DONE_JSON
            orch.working_memory.checklist.apply_updates([
                {"id": 1, "status": "done"},
                {"id": 2, "status": "done"},
            ])

        orch._reflect = Mock(side_effect=fake_reflect_done)
        events = list(orch.run())
        assert any(e["type"] == "assistant_reply" for e in events)
        # Stopped after step 1 — didn't run to max_steps
        assert len([e for e in events if e["type"] == "step"]) == 1


# ===========================================================================
# VLMAgent — TASK mode
# ===========================================================================

class TestOrchestratorTask:
    def test_task_mode_runs_without_error(self, app_state):
        orch = _make_orchestrator(app_state, AgentMode.TASK, max_steps=1)
        events = list(orch.run())
        assert not any(
            e.get("message", "").startswith("Task mode is not yet")
            for e in events if e["type"] == "error"
        )

    def test_task_checklist_from_numbered_message(self, app_state):
        app_state.chat.messages[0]["content"] = "1. Open Notepad\n2. Type hello\n3. Save file"
        orch = _make_orchestrator(app_state, AgentMode.TASK, max_steps=1)
        orch._generate_checklist = Mock(
            return_value=Checklist.from_user_text("1. Open Notepad\n2. Type hello\n3. Save file")
        )
        events = list(orch.run())
        plan_events = [e for e in events if e["type"] == "plan"]
        assert len(plan_events) == 1
        assert len(plan_events[0]["checklist"]) == 3

    def test_task_plan_event_before_first_step(self, app_state):
        app_state.chat.messages[0]["content"] = "1. Step A\n2. Step B"
        orch = _make_orchestrator(app_state, AgentMode.TASK, max_steps=1)
        events = list(orch.run())
        event_types = [e["type"] for e in events]
        assert "plan" in event_types and "step" in event_types
        assert event_types.index("plan") < event_types.index("step")

    def test_task_llm_fallback_for_prose(self, app_state):
        """Prose task with no list structure — LLM fallback produces structured checklist."""
        parsed_checklist = json.dumps([
            {"id": 1, "step": "Open browser", "verification_hint": "Browser open"},
            {"id": 2, "step": "Search for cats", "verification_hint": "Results shown"},
        ])
        app_state.chat.messages[0]["content"] = (
            "I need you to do a few things for me.\n"
            "Please open the browser first,\n"
            "then search for cats on Google."
        )
        orch = _make_orchestrator(app_state, AgentMode.TASK, max_steps=1)
        orch._generate_checklist = Mock(
            return_value=Checklist.from_llm_json(parsed_checklist)
        )
        events = list(orch.run())
        plan_events = [e for e in events if e["type"] == "plan"]
        assert len(plan_events[0]["checklist"]) == 2


# ===========================================================================
# VLMAgent — _build_observation (message construction)
# ===========================================================================

class TestVLMAgentBuildObservation:
    def _make_agent(self, tmp_path):
        from omnitool.gradio.core.agents import VLMAgent
        grounding = Mock()
        grounding.name = "omniparser"
        grounding.element_reference_hint = ""
        grounding.get_tools.return_value = []
        return VLMAgent(
            model_name="gpt-4o",
            llm_client=Mock(),
            state=AppState(run_folder=tmp_path),
            tools_collection=Mock(),
            save_folder=tmp_path,
            grounding_strategy=grounding,
        )

    def test_som_image_included_when_available(self, tmp_path):
        from omnitool.gradio.core.agents.grounding import ScreenData
        agent = self._make_agent(tmp_path)
        screen = ScreenData("rawb64", "somb64", [], 1920, 1080, 1920, 1080)
        obs = agent._build_observation(screen, "click start")
        assert any(b.get("type") == "image_url" for b in obs["content"])

    def test_no_image_block_when_display_image_empty(self, tmp_path):
        from omnitool.gradio.core.agents.grounding import ScreenData
        agent = self._make_agent(tmp_path)
        screen = ScreenData("rawb64", "", [], 1920, 1080, 1920, 1080)
        obs = agent._build_observation(screen, "")
        assert not any(b.get("type") == "image_url" for b in obs["content"])

    def test_elements_text_included_when_present(self, tmp_path):
        from omnitool.gradio.core.agents.grounding import ScreenData
        agent = self._make_agent(tmp_path)
        screen = ScreenData("rawb64", "", [{"content": "Start btn", "bbox": [0, 0, 1, 1]}], 1920, 1080, 1920, 1080)
        obs = agent._build_observation(screen, "")
        text = " ".join(b["text"] for b in obs["content"] if b.get("type") == "text")
        assert "screen_elements" in text

    def test_subtask_included_in_content(self, tmp_path):
        from omnitool.gradio.core.agents.grounding import ScreenData
        agent = self._make_agent(tmp_path)
        screen = ScreenData("rawb64", "somb64", [], 1920, 1080, 1920, 1080)
        obs = agent._build_observation(screen, "click the save button")
        text = " ".join(b["text"] for b in obs["content"] if b.get("type") == "text")
        assert "click the save button" in text


# ===========================================================================
# VLMAgent — _handle_read_field and _get_tools
# ===========================================================================

class TestVLMAgentReadField:
    def _make_agent(self, tmp_path):
        from omnitool.gradio.core.agents import VLMAgent
        grounding = Mock()
        grounding.name = "omniparser"
        grounding.element_reference_hint = ""
        grounding.get_tools.return_value = []
        return VLMAgent(
            model_name="gpt-4o",
            llm_client=Mock(),
            state=AppState(run_folder=tmp_path),
            tools_collection=Mock(),
            save_folder=tmp_path,
            grounding_strategy=grounding,
        )

    def test_new_field_captured(self, tmp_path):
        agent = self._make_agent(tmp_path)
        result, captured = agent._handle_read_field({"fields": [{"field_name": "price", "value": "12.99"}]})
        assert agent.working_memory.facts["price"] == "12.99"
        assert captured == {"price": "12.99"}
        assert "12.99" in result

    def test_empty_field_name_returns_error(self, tmp_path):
        agent = self._make_agent(tmp_path)
        result, captured = agent._handle_read_field({"fields": [{"field_name": "", "value": "x"}]})
        assert "error" in result.lower()
        assert not captured
        assert not agent.working_memory.facts

    def test_empty_fields_list_returns_error(self, tmp_path):
        agent = self._make_agent(tmp_path)
        result, captured = agent._handle_read_field({"fields": []})
        assert "error" in result.lower()
        assert not captured

    def test_same_value_returns_existing_value(self, tmp_path):
        agent = self._make_agent(tmp_path)
        agent.working_memory.facts["price"] = "12.99"
        result, captured = agent._handle_read_field({"fields": [{"field_name": "price", "value": "12.99"}]})
        assert "12.99" in result
        assert agent.working_memory.facts["price"] == "12.99"  # unchanged
        assert not captured  # already-captured fields are not re-emitted

    def test_different_value_overwrites(self, tmp_path):
        agent = self._make_agent(tmp_path)
        agent.working_memory.facts["price"] = "12.99"
        result, captured = agent._handle_read_field({"fields": [{"field_name": "price", "value": "9.99"}]})
        assert agent.working_memory.facts["price"] == "9.99"
        assert captured == {"price": "9.99"}
        assert "9.99" in result

    def test_multiple_fields_captured_in_one_call(self, tmp_path):
        agent = self._make_agent(tmp_path)
        result, captured = agent._handle_read_field({"fields": [
            {"field_name": "order_id", "value": "ORD-123"},
            {"field_name": "total", "value": "$42.00"},
        ]})
        assert agent.working_memory.facts["order_id"] == "ORD-123"
        assert agent.working_memory.facts["total"] == "$42.00"
        assert captured == {"order_id": "ORD-123", "total": "$42.00"}
        assert "ORD-123" in result
        assert "$42.00" in result

    def test_get_tools_includes_read_field(self, tmp_path):
        agent = self._make_agent(tmp_path)
        tools = agent._get_tools()
        names = [t["function"]["name"] for t in tools]
        assert "read_field" in names

    def test_aggregate_sum_stored_as_fact(self, tmp_path):
        agent = self._make_agent(tmp_path)
        agent._handle_read_field({
            "fields": [
                {"field_name": "item_1", "value": "12.50"},
                {"field_name": "item_2", "value": "7.00"},
                {"field_name": "item_3", "value": "4.25"},
            ],
            "aggregate": {"operation": "sum", "store_as": "total"},
        })
        assert agent.working_memory.facts["total"] == "23.75"

    def test_aggregate_includes_pre_existing_facts_for_deduped_fields(self, tmp_path):
        # Fields already in facts with the same value are deduped (not re-written),
        # but aggregate still sums them — semantics are "sum the fields named in this
        # call", regardless of whether they were freshly captured.
        agent = self._make_agent(tmp_path)
        agent.working_memory.facts["item_a"] = "10.00"
        agent.working_memory.facts["item_b"] = "5.00"
        agent._handle_read_field({
            "fields": [
                {"field_name": "item_a", "value": "10.00"},
                {"field_name": "item_b", "value": "5.00"},
            ],
            "aggregate": {"operation": "sum", "store_as": "subtotal"},
        })
        assert agent.working_memory.facts["subtotal"] == "15"

    def test_aggregate_ignores_non_numeric_fields(self, tmp_path):
        agent = self._make_agent(tmp_path)
        agent._handle_read_field({
            "fields": [
                {"field_name": "order_id", "value": "ORD-999"},
                {"field_name": "amount_1", "value": "20.00"},
                {"field_name": "amount_2", "value": "5.50"},
            ],
            "aggregate": {"operation": "sum", "store_as": "total"},
        })
        assert agent.working_memory.facts["total"] == "25.5"

    def test_aggregate_omitted_when_no_param(self, tmp_path):
        agent = self._make_agent(tmp_path)
        agent._handle_read_field({
            "fields": [
                {"field_name": "val_1", "value": "10.00"},
                {"field_name": "val_2", "value": "5.00"},
            ],
        })
        assert "total" not in agent.working_memory.facts
        assert set(agent.working_memory.facts.keys()) == {"val_1", "val_2"}

    def test_aggregate_skipped_when_all_fields_non_numeric(self, tmp_path):
        agent = self._make_agent(tmp_path)
        result, _ = agent._handle_read_field({
            "fields": [
                {"field_name": "label_1", "value": "N/A"},
                {"field_name": "label_2", "value": "pending"},
            ],
            "aggregate": {"operation": "sum", "store_as": "total"},
        })
        assert "total" not in agent.working_memory.facts
        assert "skipped" in result.lower()

    def test_aggregate_unsupported_operation_returns_error(self, tmp_path):
        agent = self._make_agent(tmp_path)
        result, _ = agent._handle_read_field({
            "fields": [{"field_name": "x", "value": "5.00"}],
            "aggregate": {"operation": "avg", "store_as": "mean"},
        })
        assert "mean" not in agent.working_memory.facts
        assert "unsupported" in result.lower()

    def test_aggregate_overwrites_existing_store_as_key(self, tmp_path):
        agent = self._make_agent(tmp_path)
        agent.working_memory.facts["total"] = "999.00"
        agent._handle_read_field({
            "fields": [
                {"field_name": "a", "value": "1.00"},
                {"field_name": "b", "value": "2.00"},
            ],
            "aggregate": {"operation": "sum", "store_as": "total"},
        })
        assert agent.working_memory.facts["total"] == "3"


# ===========================================================================
# _verify_step and _compare_screens
# ===========================================================================

class TestVerifyStep:
    def test_no_error_no_repeat(self, app_state):
        orch = _make_orchestrator(app_state)
        result = orch._verify_step([{"status": "success", "result": Mock(output="ok", error="")}])
        assert not result["has_error"]
        assert not result["is_repeated"]
        assert not result["screen_unchanged"]

    def test_detects_tool_error(self, app_state):
        orch = _make_orchestrator(app_state)
        result = orch._verify_step([{"status": "error", "error": "tool crashed"}])
        assert result["has_error"]
        assert "tool crashed" in result["error_detail"]

    def test_detects_repeated_action(self, app_state):
        orch = _make_orchestrator(app_state)
        for i in range(5):
            orch.working_memory.trajectory.append({
                "step": i + 1,
                "tool_calls": [{"action": "left_click", "coordinate": [100, 200]}],
            })
        assert orch._verify_step([])["is_repeated"]

    def test_screen_unchanged_when_same_image(self):
        from omnitool.gradio.core.agents import BaseAgent
        screen_a = {"som_image_base64": "abc123", "parsed_content_list": []}
        screen_b = {"som_image_base64": "abc123", "parsed_content_list": []}
        assert BaseAgent._compare_screens(screen_a, screen_b) is True

    def test_screen_changed_when_different_image(self):
        from omnitool.gradio.core.agents import BaseAgent
        screen_a = {"som_image_base64": "abc123", "parsed_content_list": []}
        screen_b = {"som_image_base64": "xyz789", "parsed_content_list": []}
        assert BaseAgent._compare_screens(screen_a, screen_b) is False

    def test_screen_fallback_to_content_list(self):
        from omnitool.gradio.core.agents import BaseAgent
        screen_a = {"som_image_base64": "", "parsed_content_list": [{"content": "A"}]}
        screen_b = {"som_image_base64": "", "parsed_content_list": [{"content": "A"}]}
        assert BaseAgent._compare_screens(screen_a, screen_b) is True
        screen_c = {"som_image_base64": "", "parsed_content_list": [{"content": "B"}]}
        assert BaseAgent._compare_screens(screen_a, screen_c) is False

    def test_screen_compare_returns_false_on_missing(self):
        from omnitool.gradio.core.agents import BaseAgent
        assert BaseAgent._compare_screens(None, None) is False
        assert BaseAgent._compare_screens({"som_image_base64": "x"}, None) is False

    def test_verify_step_screen_unchanged_flag(self, app_state):
        orch = _make_orchestrator(app_state)
        same = {"som_image_base64": "identical", "parsed_content_list": []}
        orch.working_memory.parsed_screen = same   # "before"
        assert orch._verify_step([], same)["screen_unchanged"] is True

    def test_verify_step_screen_changed_flag(self, app_state):
        orch = _make_orchestrator(app_state)
        before = {"som_image_base64": "before_data", "parsed_content_list": []}
        after  = {"som_image_base64": "after_data",  "parsed_content_list": []}
        orch.working_memory.parsed_screen = before  # "before"
        assert orch._verify_step([], after)["screen_unchanged"] is False
