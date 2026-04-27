"""
Tests for kind-aware _handle_read_field and removal of read_table tool.
"""

from unittest.mock import Mock

from omnitool.gradio.config.enums import FieldKind
from omnitool.gradio.config.systems import SystemConfig
from omnitool.gradio.config.task_template import TaskExecution, TaskOutput, TaskProcedure
from omnitool.gradio.core.tools.schemas import AUXILIARY_TOOLS
from omnitool.gradio.tests._helpers import make_react_agent


def _proc_with_output(output: TaskOutput) -> TaskProcedure:
    return TaskProcedure(
        id=1,
        description="test",
        outputs=[output],
        executions=[TaskExecution(type="cua", system="")],
    )


def _agent_with(tmp_path, *, kind: FieldKind, clipboard_correction: bool = True,
                is_browser: bool = False):
    """Build a minimal agent with a single-output procedure and configured system_config."""
    proc = _proc_with_output(
        TaskOutput(key="field1", kind=kind, clipboard_correction=clipboard_correction)
    )
    agent = make_react_agent(tmp_path, task_procedure=proc)
    agent.system_config = SystemConfig(name="T", is_browser=is_browser, prompt_fragment="")
    return agent


# ---------------------------------------------------------------------------
# SCALAR
# ---------------------------------------------------------------------------

class TestScalarKind:
    def test_no_correction_value_used_as_is(self, tmp_path):
        agent = _agent_with(tmp_path, kind=FieldKind.SCALAR, clipboard_correction=False)
        msg, read_values, events = agent._handle_read_field({
            "fields": [{"field_name": "field1", "value": "42.00", "target": "some element"}]
        })
        assert read_values == {"field1": ["42.00"]}
        assert agent.working_memory.staged_reads["field1"] == ["42.00"]
        assert events == []

    def test_no_gta1_client_skips_correction(self, tmp_path):
        agent = _agent_with(tmp_path, kind=FieldKind.SCALAR, clipboard_correction=True)
        agent.gta1_client = None
        msg, read_values, events = agent._handle_read_field({
            "fields": [{"field_name": "field1", "value": "99.00", "target": "price field"}]
        })
        assert read_values == {"field1": ["99.00"]}
        assert events == []

    def test_with_correction_calls_read_field_via_clipboard(self, tmp_path):
        agent = _agent_with(tmp_path, kind=FieldKind.SCALAR, clipboard_correction=True)
        agent.gta1_client = Mock()
        agent._read_field_via_clipboard = Mock(return_value="$42.00")
        msg, read_values, events = agent._handle_read_field({
            "fields": [{"field_name": "field1", "value": "42.00", "target": "price cell"}]
        })
        agent._read_field_via_clipboard.assert_called_once()
        assert read_values == {"field1": ["$42.00"]}
        assert events == []


# ---------------------------------------------------------------------------
# ROW — browser (DevTools path)
# ---------------------------------------------------------------------------

class TestRowKindBrowser:
    def test_routes_to_devtools_and_stages_rows(self, tmp_path):
        agent = _agent_with(tmp_path, kind=FieldKind.ROW, is_browser=True)
        agent._extract_table_via_devtools = Mock(return_value=["Header | Col", "row1 | v1", "row2 | v2"])
        msg, read_values, events = agent._handle_read_field({
            "fields": [{"field_name": "field1", "value": "", "hint": "line items"}]
        })
        agent._extract_table_via_devtools.assert_called_once_with("line items")
        assert agent.working_memory.staged_reads["field1"] == ["Header | Col", "row1 | v1", "row2 | v2"]
        assert read_values == {"field1": ["Header | Col", "row1 | v1", "row2 | v2"]}
        assert len(events) == 1
        assert events[0]["type"] == "table_read"
        assert "row1 | v1" in events[0]["text"]

    def test_deduplicates_extraction_for_same_field_in_one_call(self, tmp_path):
        agent = _agent_with(tmp_path, kind=FieldKind.ROW, is_browser=True)
        agent._extract_table_via_devtools = Mock(return_value=["row1 | v1", "row2 | v2"])
        # Two items for the same field — should extract only once
        agent._handle_read_field({
            "fields": [
                {"field_name": "field1", "value": ""},
                {"field_name": "field1", "value": ""},
            ]
        })
        agent._extract_table_via_devtools.assert_called_once()
        assert agent.working_memory.staged_reads["field1"] == ["row1 | v1", "row2 | v2"]

    def test_devtools_failure_returns_error_string(self, tmp_path):
        agent = _agent_with(tmp_path, kind=FieldKind.ROW, is_browser=True)
        agent._extract_table_via_devtools = Mock(side_effect=RuntimeError("no tables found"))
        msg, read_values, events = agent._handle_read_field({
            "fields": [{"field_name": "field1", "value": ""}]
        })
        assert "field1" in msg
        assert "no tables found" in msg
        assert read_values == {}
        assert "field1" not in agent.working_memory.staged_reads
        assert events == []


# ---------------------------------------------------------------------------
# ROW — non-browser (per-item fallback)
# ---------------------------------------------------------------------------

class TestRowKindNonBrowser:
    def test_accumulates_per_item(self, tmp_path):
        agent = _agent_with(tmp_path, kind=FieldKind.ROW, is_browser=False)
        agent._handle_read_field({
            "fields": [{"field_name": "field1", "value": "first"}]
        })
        agent._handle_read_field({
            "fields": [{"field_name": "field1", "value": "second"}]
        })
        assert agent.working_memory.staged_reads["field1"] == ["first", "second"]

    def test_devtools_not_called(self, tmp_path):
        agent = _agent_with(tmp_path, kind=FieldKind.ROW, is_browser=False)
        agent._extract_table_via_devtools = Mock()
        agent._handle_read_field({
            "fields": [{"field_name": "field1", "value": "v1"}]
        })
        agent._extract_table_via_devtools.assert_not_called()

    def test_no_table_events(self, tmp_path):
        agent = _agent_with(tmp_path, kind=FieldKind.ROW, is_browser=False)
        _, _, events = agent._handle_read_field({
            "fields": [{"field_name": "field1", "value": "v1"}]
        })
        assert events == []


# ---------------------------------------------------------------------------
# TABLE — browser (DevTools path)
# ---------------------------------------------------------------------------

class TestTableKindBrowser:
    def test_stages_rows_and_emits_event(self, tmp_path):
        agent = _agent_with(tmp_path, kind=FieldKind.TABLE, is_browser=True)
        agent._extract_table_via_devtools = Mock(return_value=["a | b", "c | d"])
        msg, read_values, events = agent._handle_read_field({
            "fields": [{"field_name": "field1", "value": "", "hint": "orders"}]
        })
        assert agent.working_memory.staged_reads["field1"] == ["a | b", "c | d"]
        assert read_values == {"field1": ["a | b", "c | d"]}
        assert len(events) == 1
        assert events[0]["type"] == "table_read"


# ---------------------------------------------------------------------------
# TABLE — non-browser (OCR not supported)
# ---------------------------------------------------------------------------

class TestTableKindNonBrowser:
    def test_returns_ocr_error_string(self, tmp_path):
        agent = _agent_with(tmp_path, kind=FieldKind.TABLE, is_browser=False)
        msg, read_values, events = agent._handle_read_field({
            "fields": [{"field_name": "field1", "value": ""}]
        })
        assert "OCR not yet supported" in msg
        assert "field1" in msg
        assert read_values == {}
        assert "field1" not in agent.working_memory.staged_reads
        assert events == []

    def test_no_exception_escapes(self, tmp_path):
        agent = _agent_with(tmp_path, kind=FieldKind.TABLE, is_browser=False)
        # Should not raise even though OCR is unimplemented
        agent._handle_read_field({"fields": [{"field_name": "field1", "value": ""}]})


# ---------------------------------------------------------------------------
# Schema — read_table removed
# ---------------------------------------------------------------------------

class TestReadTableRemovedFromSchema:
    def test_not_in_auxiliary_tools(self):
        names = {t["function"]["name"] for t in AUXILIARY_TOOLS}
        assert "read_table" not in names

    def test_read_table_tool_not_importable(self):
        import omnitool.gradio.core.tools.schemas as schemas
        assert not hasattr(schemas, "READ_TABLE_TOOL")
