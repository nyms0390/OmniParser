"""
Tests for kind-aware _handle_read_field and removal of read_table tool.
"""

from unittest.mock import Mock

from omnitool.gradio.config.enums import ColumnKind
from omnitool.gradio.config.systems import SystemConfig
from omnitool.gradio.config.task_template import TaskExecution, TaskOutput, TaskProcedure
from omnitool.gradio.core.tools.schemas import AUXILIARY_TOOLS
from omnitool.gradio.tests._helpers import make_1px_png_b64, make_react_agent


def _proc_with_output(output: TaskOutput) -> TaskProcedure:
    return TaskProcedure(
        id=1,
        description="test",
        outputs=[output],
        executions=[TaskExecution(type="cua", system="")],
    )


def _agent_with(tmp_path, *, kind: ColumnKind, clipboard_correction: bool = True,
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
        agent = _agent_with(tmp_path, kind=ColumnKind.SCALAR, clipboard_correction=False)
        msg, read_values, events = agent._handle_read_field({
            "fields": [{"field_name": "field1", "value": "42.00", "target": "some element"}]
        })
        assert read_values == {"field1": ["42.00"]}
        assert agent.working_memory.staged_reads["field1"] == ["42.00"]
        assert events == []

    def test_no_gta1_client_skips_correction(self, tmp_path):
        agent = _agent_with(tmp_path, kind=ColumnKind.SCALAR, clipboard_correction=True)
        agent.gta1_client = None
        msg, read_values, events = agent._handle_read_field({
            "fields": [{"field_name": "field1", "value": "99.00", "target": "price field"}]
        })
        assert read_values == {"field1": ["99.00"]}
        assert events == []

    def test_with_correction_calls_read_field_via_clipboard(self, tmp_path):
        agent = _agent_with(tmp_path, kind=ColumnKind.SCALAR, clipboard_correction=True)
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
    def test_routes_to_column_extractor_and_stages_values(self, tmp_path):
        agent = _agent_with(tmp_path, kind=ColumnKind.ROW, is_browser=True)
        agent._extract_column = Mock(return_value=["10.00", "20.00", "30.00"])
        msg, read_values, events = agent._handle_read_field({
            "fields": [{"field_name": "field1", "value": "", "hint": "line items"}]
        })
        agent._extract_column.assert_called_once_with("field1", "", "line items")
        assert agent.working_memory.staged_reads["field1"] == ["10.00", "20.00", "30.00"]
        assert read_values == {"field1": ["10.00", "20.00", "30.00"]}
        assert len(events) == 1
        assert events[0]["type"] == "table_read"
        assert "20.00" in events[0]["text"]

    def test_devtools_failure_returns_error_string(self, tmp_path):
        agent = _agent_with(tmp_path, kind=ColumnKind.ROW, is_browser=True)
        agent._extract_column = Mock(side_effect=RuntimeError("no tables found"))
        msg, read_values, events = agent._handle_read_field({
            "fields": [{"field_name": "field1", "value": ""}]
        })
        assert "field1" in msg
        assert "no tables found" in msg
        assert read_values == {}
        assert "field1" not in agent.working_memory.staged_reads
        assert events == []


# ---------------------------------------------------------------------------
# Non-browser OCR — exercised end-to-end through _capture_tables_html
# ---------------------------------------------------------------------------

def _ocr_agent(tmp_path, *, kind: ColumnKind, paddleocr_client):
    """Build an agent whose non-browser OCR path is wired to the given client."""
    agent = _agent_with(tmp_path, kind=kind, is_browser=False)
    agent.paddleocr_client = paddleocr_client
    agent.working_memory.parsed_screen = {"resized_image_base64": make_1px_png_b64()}
    agent.working_memory.focus_image_b64 = make_1px_png_b64()
    return agent


def _llm_returning(text: str) -> Mock:
    return Mock(return_value=(text, {
        "tokens": 10, "input_tokens": 5, "output_tokens": 5,
    }))


_TABLE_HTML = (
    "<html><body><table>"
    "<tr><th>name</th></tr><tr><td>Alice</td></tr><tr><td>Bob</td></tr>"
    "</table></body></html>"
)


class TestNonBrowserOCR:
    def test_scalar_kind_preserves_focus_crop(self, tmp_path):
        # Scalar reads do not consume the crop — it stays available for follow-up
        # reads in the same focused area until a screen-changing action invalidates it.
        agent = _ocr_agent(tmp_path, kind=ColumnKind.SCALAR, paddleocr_client=Mock())
        # Override the procedure to a SCALAR field (no extraction, no consume).
        agent.task_procedure = _proc_with_output(
            TaskOutput(key="field1", kind=ColumnKind.SCALAR, clipboard_correction=False)
        )
        crop_before = agent.working_memory.focus_image_b64
        agent._handle_read_field({
            "fields": [{"field_name": "field1", "value": "42"}]
        })
        assert agent.working_memory.focus_image_b64 == crop_before

    def test_row_kind_uses_column_extraction_prompt(self, tmp_path):
        client = Mock()
        client.recognize_vl.return_value = _TABLE_HTML
        agent = _ocr_agent(tmp_path, kind=ColumnKind.ROW, paddleocr_client=client)
        agent.llm_client.generate = _llm_returning("Alice\nBob")

        agent._handle_read_field({
            "fields": [{"field_name": "field1", "value": "", "hint": "name col"}]
        })

        # Verify routing went through OCR + column-extraction prompt
        client.recognize_vl.assert_called_once()
        prompt_text = agent.llm_client.generate.call_args.kwargs["messages"][0]["content"]
        assert "Field key: field1" in prompt_text
        assert agent.working_memory.staged_reads["field1"] == ["Alice", "Bob"]

    def test_missing_focus_returns_extraction_error(self, tmp_path):
        client = Mock()
        agent = _agent_with(tmp_path, kind=ColumnKind.ROW, is_browser=False)
        agent.paddleocr_client = client
        agent.working_memory.parsed_screen = {"resized_image_base64": make_1px_png_b64()}
        # Intentionally do not set focus_image_b64

        msg, read_values, events = agent._handle_read_field({
            "fields": [{"field_name": "field1", "value": ""}]
        })

        assert "Extraction failed" in msg
        assert "focus_region" in msg
        assert read_values == {}
        assert "field1" not in agent.working_memory.staged_reads
        assert events == []
        client.recognize_vl.assert_not_called()

    def test_unconfigured_client_returns_extraction_error(self, tmp_path):
        agent = _ocr_agent(tmp_path, kind=ColumnKind.ROW, paddleocr_client=None)
        msg, read_values, events = agent._handle_read_field({
            "fields": [{"field_name": "field1", "value": ""}]
        })
        assert "Extraction failed" in msg
        assert "paddleocr_client not configured" in msg
        assert read_values == {}
        assert "field1" not in agent.working_memory.staged_reads
        assert events == []


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
