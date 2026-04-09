"""
Tests for screenshot/focus-crop auto-saving, facts.json persistence, and
the mark_screenshot tool — all added in the trajectory-saving refactor.

All filesystem I/O is exercised against tmp_path (no real screen capture).
"""

import base64
import json
import pytest
from io import BytesIO
from unittest.mock import Mock

from PIL import Image

from omnitool.gradio.services import AppState
from omnitool.gradio.core.agents.grounding import ScreenData
from omnitool.gradio.config.enums import AggregateOperation
from omnitool.gradio.config.task_template import TaskOutput, TaskOutputAggregate, TaskProcedure


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_1px_png_b64() -> str:
    """Return a valid base64-encoded 1×1 white PNG."""
    buf = BytesIO()
    Image.new("RGB", (1, 1), color=(255, 255, 255)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_screen_data(b64: str) -> ScreenData:
    return ScreenData(
        raw_image_b64=b64,
        display_image_b64=b64,
        elements=[],
        screen_width=100,
        screen_height=100,
        resized_width=100,
        resized_height=100,
    )


def _tool_response(tool_name="left_click", args=None, text="Acting."):
    if args is None:
        args = {"box_id": 0}
    tc_id = "call_1"
    return (
        text,
        {
            "tokens": 10,
            "input_tokens": 5,
            "output_tokens": 5,
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


def _finish_response():
    return _tool_response("finish", {"success": True, "summary": "done"})


def _make_react_agent(tmp_path, side_effects):
    from omnitool.gradio.core.agents.react_agent import ReActAgent

    app_state = AppState(run_folder=tmp_path)
    app_state.chat.add_message("user", "Do something")

    b64 = _make_1px_png_b64()
    screen_data = _make_screen_data(b64)

    grounding = Mock()
    grounding.name = "omniparser"
    grounding.element_reference_hint = "Use box_id."
    grounding.get_tools.return_value = [{
        "type": "function",
        "function": {
            "name": "left_click",
            "parameters": {"type": "object", "properties": {"box_id": {"type": "integer"}}, "required": ["box_id"]},
        },
    }]
    grounding.preprocess.return_value = screen_data
    grounding.resolve.return_value = {"tool": "computer", "action": "left_click", "coordinate": [10, 10]}
    grounding.last_grounding_events = []

    llm_client = Mock()
    llm_client.generate.side_effect = side_effects + [_finish_response()] * 10

    agent = ReActAgent(
        model_name="gpt-4o",
        llm_client=llm_client,
        state=app_state,
        tools_collection=Mock(),
        save_folder=tmp_path,
        grounding_strategy=grounding,
        max_steps=5,
        action_delay=0,
    )
    agent._capture_screen = Mock(return_value={
        "raw_image_base64":          b64,
        "resized_image_base64":      b64,
        "preprocessed_image_base64": b64,
        "screen_width": 100,
        "screen_height": 100,
        "resized_screen_width": 100,
        "resized_screen_height": 100,
    })
    agent.execute_tool_calls = Mock(return_value=[{
        "tool": "computer", "status": "success",
        "result": Mock(output="ok", error=""),
    }])
    return agent




# ===========================================================================
# _save_trajectory_step — screenshot auto-save
# ===========================================================================

class TestTrajectoryScreenshotSave:
    def test_screenshot_file_written_to_save_folder(self, tmp_path):
        # finish() exits before _save_trajectory_step; need a real computer action first
        agent = _make_react_agent(tmp_path, [_tool_response(), _finish_response()])
        list(agent.run())
        png_files = list(tmp_path.glob("step_*.png"))
        assert len(png_files) >= 1, "Expected at least one step screenshot to be saved"

    def test_screenshot_filename_matches_step_number(self, tmp_path):
        agent = _make_react_agent(tmp_path, [_tool_response(), _finish_response()])
        list(agent.run())
        # step_001.png must exist (step after the first click)
        assert (tmp_path / "step_001.png").exists(), "step_001.png not found"

    def test_saved_screenshot_is_valid_png(self, tmp_path):
        agent = _make_react_agent(tmp_path, [_tool_response(), _finish_response()])
        list(agent.run())
        png_files = sorted(tmp_path.glob("step_*.png"))
        img = Image.open(png_files[0])
        assert img.format == "PNG"

    def test_trajectory_json_contains_screenshot_file_key(self, tmp_path):
        agent = _make_react_agent(tmp_path, [_tool_response(), _finish_response()])
        list(agent.run())
        traj_file = tmp_path / "trajectory.json"
        assert traj_file.exists()
        with open(traj_file) as f:
            records = [json.loads(line) for line in f if line.strip() and not json.loads(line).get("type")]
        assert all("screenshot_file" in r for r in records), "Not all step records have screenshot_file"

    def test_no_screenshot_saved_when_parsed_screen_missing(self, tmp_path):
        """If working_memory.parsed_screen is None, no crash and no PNG written."""
        agent = _make_react_agent(tmp_path, [])
        # Inject a step with no parsed_screen
        agent.working_memory.parsed_screen = None
        plan_response = {"response_text": "", "tool_calls": [], "metadata": {}, "cost": 0}
        agent._save_trajectory_step(plan_response)
        # Should not raise and no PNG for step_0
        assert not (tmp_path / "step_000.png").exists()


# ===========================================================================
# _handle_focus_region — crop auto-save
# ===========================================================================

class TestFocusCropSave:
    def _setup_agent_with_screen(self, tmp_path):
        """Return a bare agent with parsed_screen set to a real PNG."""
        agent = _make_react_agent(tmp_path, [])
        b64 = _make_1px_png_b64()
        agent.working_memory.parsed_screen = {"resized_image_base64": b64}
        return agent

    def test_focus_crop_file_written(self, tmp_path):
        agent = self._setup_agent_with_screen(tmp_path)
        result = agent._handle_focus_region({"bbox": [0, 0, 1, 1]})
        assert result is not None
        crop_files = list(tmp_path.glob("step_*_focus_*.png"))
        assert len(crop_files) == 1, "Expected one focus crop file"

    def test_focus_crop_filename_includes_step_and_index(self, tmp_path):
        agent = self._setup_agent_with_screen(tmp_path)
        agent._handle_focus_region({"bbox": [0, 0, 1, 1]})
        assert (tmp_path / "step_000_focus_00.png").exists()

    def test_second_focus_crop_gets_incremented_index(self, tmp_path):
        agent = self._setup_agent_with_screen(tmp_path)
        agent._handle_focus_region({"bbox": [0, 0, 1, 1]})
        agent._handle_focus_region({"bbox": [0, 0, 1, 1]})
        assert (tmp_path / "step_000_focus_00.png").exists()
        assert (tmp_path / "step_000_focus_01.png").exists()

    def test_focus_crop_counter_starts_at_zero(self, tmp_path):
        agent = _make_react_agent(tmp_path, [])
        assert agent._focus_crop_count == 0

    def test_focus_crop_returns_base64_unchanged(self, tmp_path):
        agent = self._setup_agent_with_screen(tmp_path)
        result = agent._handle_focus_region({"bbox": [0, 0, 1, 1]})
        assert result is not None
        # Must be decodeable base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_invalid_bbox_returns_none_no_file(self, tmp_path):
        agent = self._setup_agent_with_screen(tmp_path)
        result = agent._handle_focus_region({"bbox": [0, 0]})  # too short
        assert result is None
        assert len(list(tmp_path.glob("step_*_focus_*.png"))) == 0


# ===========================================================================
# _handle_read_field — facts.json auto-save
# ===========================================================================

class TestFactsSave:
    def _make_minimal_agent(self, tmp_path):
        agent = _make_react_agent(tmp_path, [])
        agent.gta1_client = None  # disable clipboard correction
        return agent

    def test_facts_stored_in_working_memory_after_read_field(self, tmp_path):
        agent = self._make_minimal_agent(tmp_path)
        agent._handle_read_field({"fields": [{"field_name": "order_id", "value": "12345"}]})
        assert agent.working_memory.facts["order_id"] == ["12345"]
        assert not (tmp_path / "facts.json").exists()

    def test_facts_captured_value_in_working_memory(self, tmp_path):
        agent = self._make_minimal_agent(tmp_path)
        agent._handle_read_field({"fields": [{"field_name": "total", "value": "$99.00"}]})
        assert agent.working_memory.facts["total"] == ["$99.00"]
        assert not (tmp_path / "facts.json").exists()

    def test_facts_accumulate_across_calls_in_working_memory(self, tmp_path):
        agent = self._make_minimal_agent(tmp_path)
        agent._handle_read_field({"fields": [{"field_name": "a", "value": "1"}]})
        agent._handle_read_field({"fields": [{"field_name": "b", "value": "2"}]})
        assert agent.working_memory.facts == {"a": ["1"], "b": ["2"]}
        assert not (tmp_path / "facts.json").exists()

    def test_duplicate_field_is_not_overwritten(self, tmp_path):
        agent = self._make_minimal_agent(tmp_path)
        agent.working_memory.facts["x"] = ["v"]
        agent._handle_read_field({"fields": [{"field_name": "x", "value": "v"}]})
        assert agent.working_memory.facts == {"x": ["v"]}

    def test_overwriting_scalar_fact_with_different_value_stores_new_value(self, tmp_path):
        agent = self._make_minimal_agent(tmp_path)
        agent._handle_read_field({"fields": [{"field_name": "x", "value": "old"}]})
        assert agent.working_memory.facts["x"] == ["old"]
        agent._handle_read_field({"fields": [{"field_name": "x", "value": "new"}]})
        assert agent.working_memory.facts["x"] == ["new"]


# ===========================================================================
# _handle_mark_screenshot
# ===========================================================================

class TestHandleMarkScreenshot:
    def test_appends_flag_record_to_trajectory_json(self, tmp_path):
        agent = _make_react_agent(tmp_path, [])
        agent._handle_mark_screenshot("order confirmed")
        traj = tmp_path / "trajectory.json"
        assert traj.exists()
        records = [json.loads(line) for line in traj.read_text().splitlines() if line.strip()]
        flags = [r for r in records if r.get("type") == "flag"]
        assert len(flags) == 1

    def test_flag_record_contains_reason(self, tmp_path):
        agent = _make_react_agent(tmp_path, [])
        agent._handle_mark_screenshot("error state detected")
        records = [json.loads(line) for line in (tmp_path / "trajectory.json").read_text().splitlines()]
        flag = records[0]
        assert flag["reason"] == "error state detected"

    def test_flag_record_screenshot_file_matches_step(self, tmp_path):
        agent = _make_react_agent(tmp_path, [])
        agent.step_count = 3
        agent._handle_mark_screenshot("done")
        flag = json.loads((tmp_path / "trajectory.json").read_text().strip())
        assert flag["screenshot_file"] == "step_003.png"

    def test_flag_record_step_matches_current_step_count(self, tmp_path):
        agent = _make_react_agent(tmp_path, [])
        agent.step_count = 7
        agent._handle_mark_screenshot("check")
        flag = json.loads((tmp_path / "trajectory.json").read_text().strip())
        assert flag["step"] == 7

    def test_returns_confirmation_string(self, tmp_path):
        agent = _make_react_agent(tmp_path, [])
        result = agent._handle_mark_screenshot("task verified")
        assert "step_000.png" in result
        assert "task verified" in result

    def test_multiple_flags_all_appended(self, tmp_path):
        agent = _make_react_agent(tmp_path, [])
        agent._handle_mark_screenshot("first")
        agent._handle_mark_screenshot("second")
        records = [json.loads(line) for line in (tmp_path / "trajectory.json").read_text().splitlines() if line.strip()]
        assert len(records) == 2
        assert records[0]["reason"] == "first"
        assert records[1]["reason"] == "second"


# ===========================================================================
# mark_screenshot tool dispatch — ReActAgent
# ===========================================================================

class TestReActMarkScreenshotDispatch:
    def test_mark_screenshot_tool_in_react_tool_list(self, tmp_path):
        agent = _make_react_agent(tmp_path, [])
        tool_names = [t["function"]["name"] for t in agent._get_tools()]
        assert "mark_screenshot" in tool_names
        # Verify all expected tools are present (guards against accidental removal)
        assert "read_field" in tool_names
        assert "focus_region" in tool_names
        assert "finish" in tool_names

    def test_react_agent_dispatches_mark_screenshot_tool(self, tmp_path):
        agent = _make_react_agent(tmp_path, [
            _tool_response("mark_screenshot", {"reason": "step confirmed"}),
        ])
        list(agent.run())
        traj = tmp_path / "trajectory.json"
        records = [json.loads(line) for line in traj.read_text().splitlines() if line.strip()]
        flags = [r for r in records if r.get("type") == "flag"]
        assert len(flags) == 1
        assert flags[0]["reason"] == "step confirmed"


# ===========================================================================
# MARK_SCREENSHOT_TOOL schema
# ===========================================================================

class TestMarkScreenshotSchema:
    def test_schema_has_correct_tool_name(self):
        from omnitool.gradio.core.tools.schemas import MARK_SCREENSHOT_TOOL
        assert MARK_SCREENSHOT_TOOL["function"]["name"] == "mark_screenshot"

    def test_schema_requires_reason_parameter(self):
        from omnitool.gradio.core.tools.schemas import MARK_SCREENSHOT_TOOL
        params = MARK_SCREENSHOT_TOOL["function"]["parameters"]
        assert "reason" in params["properties"]
        assert "reason" in params["required"]

    def test_schema_exported_in_all(self):
        from omnitool.gradio.core.tools import schemas
        assert "MARK_SCREENSHOT_TOOL" in schemas.__all__


# ===========================================================================
# _write_run_summary
# ===========================================================================

class TestWriteRunSummary:
    def test_summary_json_written_after_run(self, tmp_path):
        agent = _make_react_agent(tmp_path, [])
        list(agent.run())
        assert (tmp_path / "summary.json").exists()

    def test_summary_contains_expected_keys(self, tmp_path):
        agent = _make_react_agent(tmp_path, [])
        list(agent.run())
        summary = json.loads((tmp_path / "summary.json").read_text())
        for key in ("task", "start_time", "end_time", "duration",
                    "success", "message", "total_steps", "total_tokens",
                    "total_cost_usd", "flags", "facts"):
            assert key in summary, f"missing key: {key}"

    def test_summary_success_true_on_finish(self, tmp_path):
        agent = _make_react_agent(tmp_path, [])
        list(agent.run())
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["success"] is True

    def test_summary_flags_populated_from_mark_screenshot(self, tmp_path):
        agent = _make_react_agent(tmp_path, [
            _tool_response("mark_screenshot", {"reason": "confirmed"}),
        ])
        list(agent.run())
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert any(f["reason"] == "confirmed" for f in summary["flags"])

    def test_summary_facts_populated_from_read_field(self, tmp_path):
        agent = _make_react_agent(tmp_path, [
            _tool_response("read_field", {"fields": [{"field_name": "order_id", "value": "99"}]}),
        ])
        list(agent.run())
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["facts"].get("order_id") == ["99"]

    def test_summary_written_on_crash(self, tmp_path):
        agent = _make_react_agent(tmp_path, [])
        agent._capture_screen = Mock(side_effect=RuntimeError("boom"))
        list(agent.run())
        assert (tmp_path / "summary.json").exists()
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["success"] is False
        assert "boom" in summary["message"]

    def test_summary_duration_is_hh_mm_ss_format(self, tmp_path):
        agent = _make_react_agent(tmp_path, [])
        list(agent.run())
        summary = json.loads((tmp_path / "summary.json").read_text())
        import re
        assert re.fullmatch(r"\d{2}:\d{2}:\d{2}", summary["duration"])

    def test_summary_total_cost_usd_is_float(self, tmp_path):
        agent = _make_react_agent(tmp_path, [])
        list(agent.run())
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert isinstance(summary["total_cost_usd"], float)


# ---------------------------------------------------------------------------
# Template aggregate tests
# ---------------------------------------------------------------------------

class TestAggregateTransience:
    """Template-declared aggregates fire at finish via _apply_template_aggregates."""

    def _make_agent_with_outputs(self, tmp_path, outputs):
        agent = _make_react_agent(tmp_path, [])
        agent.gta1_client = None
        proc = TaskProcedure(id=1, description="test", outputs=outputs)
        agent.task_procedure = proc
        agent._output_def_map = {o.key: o for o in outputs}
        return agent

    def test_dynamic_field_accumulates_across_calls(self, tmp_path):
        outputs = [TaskOutput(key="line_amount", dynamic=True)]
        agent = self._make_agent_with_outputs(tmp_path, outputs)
        agent._handle_read_field({"fields": [{"field_name": "line_amount", "value": "10.00"}]})
        agent._handle_read_field({"fields": [{"field_name": "line_amount", "value": "5.00"}]})
        assert agent.working_memory.facts["line_amount"] == ["10.00", "5.00"]

    def test_aggregate_result_written_to_facts_at_finish(self, tmp_path):
        agg = TaskOutputAggregate(operation=AggregateOperation.SUM, source="line_amount")
        outputs = [
            TaskOutput(key="line_amount", dynamic=True),
            TaskOutput(key="grand_total", aggregate=agg),
        ]
        agent = self._make_agent_with_outputs(tmp_path, outputs)
        agent.working_memory.facts["line_amount"] = ["10.00", "5.00"]
        agent._apply_template_aggregates()
        result = agent.working_memory.facts["grand_total"]
        assert len(result) == 1
        assert float(result[0]) == pytest.approx(15.0)

    def test_all_non_numeric_source_skips_aggregate(self, tmp_path):
        agg = TaskOutputAggregate(operation=AggregateOperation.SUM, source="labels")
        outputs = [
            TaskOutput(key="labels", dynamic=True),
            TaskOutput(key="total", aggregate=agg),
        ]
        agent = self._make_agent_with_outputs(tmp_path, outputs)
        agent.working_memory.facts["labels"] = ["N/A", "—"]
        agent._apply_template_aggregates()
        assert "total" not in agent.working_memory.facts

    def test_unknown_operation_raises_at_load_time(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid aggregate operation"):
            TaskOutputAggregate.from_dict({"operation": "median", "source": "values"})

    def test_no_task_procedure_is_noop(self, tmp_path):
        agent = _make_react_agent(tmp_path, [])
        agent.task_procedure = None
        agent.working_memory.facts["x"] = ["1.00"]
        agent._apply_template_aggregates()  # must not raise
        assert "x" in agent.working_memory.facts  # unchanged

    def test_multiple_aggregate_outputs_all_computed(self, tmp_path):
        agg1 = TaskOutputAggregate(operation=AggregateOperation.SUM, source="prices")
        agg2 = TaskOutputAggregate(operation=AggregateOperation.SUM, source="fees")
        outputs = [
            TaskOutput(key="prices", dynamic=True),
            TaskOutput(key="fees", dynamic=True),
            TaskOutput(key="price_total", aggregate=agg1),
            TaskOutput(key="fee_total", aggregate=agg2),
        ]
        agent = self._make_agent_with_outputs(tmp_path, outputs)
        agent.working_memory.facts["prices"] = ["10.00", "20.00"]
        agent.working_memory.facts["fees"] = ["1.00", "2.00"]
        agent._apply_template_aggregates()
        assert agent.working_memory.facts["price_total"] == ["30"]
        assert agent.working_memory.facts["fee_total"] == ["3"]

    def test_empty_source_field_skips_aggregate(self, tmp_path):
        agg = TaskOutputAggregate(operation=AggregateOperation.SUM, source="line_amount")
        outputs = [
            TaskOutput(key="line_amount", dynamic=True),
            TaskOutput(key="grand_total", aggregate=agg),
        ]
        agent = self._make_agent_with_outputs(tmp_path, outputs)
        # line_amount never captured
        agent._apply_template_aggregates()
        assert "grand_total" not in agent.working_memory.facts

    def test_aggregate_overwrites_existing_value(self, tmp_path):
        agg = TaskOutputAggregate(operation=AggregateOperation.SUM, source="line_amount")
        outputs = [
            TaskOutput(key="line_amount", dynamic=True),
            TaskOutput(key="grand_total", aggregate=agg),
        ]
        agent = self._make_agent_with_outputs(tmp_path, outputs)
        agent.working_memory.facts["grand_total"] = ["OLD"]
        agent.working_memory.facts["line_amount"] = ["7.00"]
        agent._apply_template_aggregates()
        assert agent.working_memory.facts["grand_total"] == ["7"]
