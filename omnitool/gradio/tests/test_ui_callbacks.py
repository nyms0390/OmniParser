"""
Tests for GradioCallbacks (omnitool/gradio/ui/callbacks.py).

Covers the event-handler methods extracted from app.py in the
ui/gradio/ → ui/ flattening refactor.  All Gradio UI machinery and
external services are mocked; no real HTTP calls or file-system writes
to the upload folder are made.
"""

from unittest.mock import Mock, patch

from omnitool.gradio.config import AgentMode
from omnitool.gradio.config.enums import ColumnKind
from omnitool.gradio.config.task_template import (
    TaskExecution,
    TaskTemplate,
    TemplateField,
)
from omnitool.gradio.services import AppState
from omnitool.gradio.ui.callbacks import GradioCallbacks, MAX_TASK_USER_FIELDS
from omnitool.gradio.ui.components import get_provider_options_for_model


# ---------------------------------------------------------------------------
# Helpers — build a minimal GradioCallbacks-based object without Gradio UI
# ---------------------------------------------------------------------------

class _StubApp(GradioCallbacks):
    """Minimal concrete class that mixes in GradioCallbacks without any real
    Gradio widgets or HTTP clients."""

    def __init__(self, settings=None, tools=None, tmp_path=None):
        super().__init__()
        self.settings = settings or _make_settings(tmp_path)
        self.tools = tools or Mock()
        self.omniparser_client = Mock()
        self.gta1_client = Mock()
        self.paddleocr_client = Mock()
        self.orchestrator = None


def _make_settings(tmp_path=None):
    s = Mock()
    s.run_folder = str(tmp_path or "/tmp/omni_test")
    s.azure_endpoint = ""
    s.task_user_values_path = ""
    return s


def _make_template(description="Do something"):
    return TaskTemplate(
        name="Test Template",
        description=description,
        fields={
            "input1": TemplateField(label="Input 1", source="user", kind=ColumnKind.SCALAR),
            "result": TemplateField(label="Result", source="generated", kind=ColumnKind.SCALAR),
        },
        export=["result"],
        executions=[],
    )


# ---------------------------------------------------------------------------
# on_agent_change
# ---------------------------------------------------------------------------

class TestOnAgentChange:
    def test_vlmagent_makes_grounding_visible(self, tmp_path):
        app = _StubApp(tmp_path=tmp_path)
        with patch("omnitool.gradio.ui.callbacks.gr") as mock_gr:
            mock_gr.update.return_value = {"visible": True}
            app.on_agent_change("VLMAgent")
        mock_gr.update.assert_called_once_with(visible=True)

    def test_react_agent_makes_grounding_visible(self, tmp_path):
        app = _StubApp(tmp_path=tmp_path)
        with patch("omnitool.gradio.ui.callbacks.gr") as mock_gr:
            mock_gr.update.return_value = {"visible": True}
            app.on_agent_change("ReActAgent")
        mock_gr.update.assert_called_once_with(visible=True)

    def test_anthropic_agent_hides_grounding(self, tmp_path):
        app = _StubApp(tmp_path=tmp_path)
        with patch("omnitool.gradio.ui.callbacks.gr") as mock_gr:
            mock_gr.update.return_value = {"visible": False}
            app.on_agent_change("AnthropicAgent")
        mock_gr.update.assert_called_once_with(visible=False)

    def test_unknown_agent_type_hides_grounding(self, tmp_path):
        app = _StubApp(tmp_path=tmp_path)
        with patch("omnitool.gradio.ui.callbacks.gr") as mock_gr:
            mock_gr.update.return_value = {"visible": False}
            app.on_agent_change("SomeFutureAgent")
        mock_gr.update.assert_called_once_with(visible=False)


# ---------------------------------------------------------------------------
# on_model_change
# ---------------------------------------------------------------------------

class TestOnModelChange:
    def test_returns_provider_choices_for_known_model(self, tmp_path):
        app = _StubApp(tmp_path=tmp_path)
        with patch("omnitool.gradio.ui.callbacks.gr") as mock_gr:
            mock_gr.update.return_value = object()
            app.on_model_change("gpt-4o")
        # get_provider_options_for_model is called internally
        providers = get_provider_options_for_model("gpt-4o")
        assert len(providers) > 0
        mock_gr.update.assert_called_once_with(
            choices=providers,
            value=providers[0],
        )

    def test_empty_provider_list_uses_empty_string_as_value(self, tmp_path):
        app = _StubApp(tmp_path=tmp_path)
        with patch(
            "omnitool.gradio.ui.callbacks.get_provider_options_for_model",
            return_value=[],
        ), patch("omnitool.gradio.ui.callbacks.gr") as mock_gr:
            mock_gr.update.return_value = None
            app.on_model_change("no-providers-model")
        mock_gr.update.assert_called_once_with(choices=[], value="")


# ---------------------------------------------------------------------------
# on_mode_change
# ---------------------------------------------------------------------------

class TestOnModeChange:
    def test_task_mode_shows_template_dropdown(self, tmp_path):
        app = _StubApp(tmp_path=tmp_path)
        with patch("omnitool.gradio.ui.callbacks.gr") as mock_gr:
            mock_gr.update.return_value = {"visible": True}
            result = app.on_mode_change(AgentMode.TASK.value)
        assert len(result) == 2 + MAX_TASK_USER_FIELDS
        mock_gr.update.assert_any_call(visible=True)

    def test_interactive_mode_hides_template_dropdown(self, tmp_path):
        app = _StubApp(tmp_path=tmp_path)
        with patch("omnitool.gradio.ui.callbacks.gr") as mock_gr:
            mock_gr.update.return_value = {"visible": False}
            result = app.on_mode_change(AgentMode.INTERACTIVE.value)
        assert len(result) == 2 + MAX_TASK_USER_FIELDS
        mock_gr.update.assert_any_call(visible=False)

    def test_orchestrated_mode_hides_template_dropdown(self, tmp_path):
        app = _StubApp(tmp_path=tmp_path)
        with patch("omnitool.gradio.ui.callbacks.gr") as mock_gr:
            mock_gr.update.return_value = {"visible": False}
            result = app.on_mode_change(AgentMode.ORCHESTRATED.value)
        assert len(result) == 2 + MAX_TASK_USER_FIELDS
        mock_gr.update.assert_any_call(visible=False)


# ---------------------------------------------------------------------------
# on_template_select
# ---------------------------------------------------------------------------

class TestOnTemplateSelect:
    def test_none_filepath_returns_none_and_hidden_dropdown(self, tmp_path):
        app = _StubApp(tmp_path=tmp_path)
        with patch("omnitool.gradio.ui.callbacks.gr") as mock_gr:
            mock_gr.update.return_value = {"visible": False}
            template, _exec, *_inputs = app.on_template_select(None)
        assert template is None
        assert mock_gr.update.call_count == 1 + MAX_TASK_USER_FIELDS

    def test_valid_yaml_returns_template_and_execution_dropdown(self, tmp_path):
        yaml_content = """
name: Test Template
description: "Test task"
fields:
  input1:
    label: Input 1
    source: user
    kind: scalar
  out1:
    label: Output One
    source: generated
    kind: scalar
export:
  - out1
executions:
  - id: 1
    title: Get output
    tool: cua
    system: iWeb
    foreach: input1
    uses: [input1]
    writes: [out1]
    steps: "Step 1: do <input1>（Input 1）"
"""
        yaml_file = tmp_path / "template.yaml"
        yaml_file.write_text(yaml_content)

        app = _StubApp(tmp_path=tmp_path)
        with patch("omnitool.gradio.ui.callbacks.gr") as mock_gr:
            mock_gr.update.return_value = {"visible": True}
            template, _exec, *_inputs = app.on_template_select(str(yaml_file))

        assert template is not None
        assert template.description == "Test task"
        assert mock_gr.update.call_count == 1 + MAX_TASK_USER_FIELDS
        exec_call_kwargs = mock_gr.update.call_args_list[0].kwargs
        assert exec_call_kwargs["value"] is None
        assert ("Whole task", None) in exec_call_kwargs["choices"]
        assert ("Execution 1", 1) in exec_call_kwargs["choices"]
        input_call_kwargs = mock_gr.update.call_args_list[1].kwargs
        assert input_call_kwargs["label"] == "Input 1"
        assert input_call_kwargs["visible"] is True

    def test_template_select_prefills_user_inputs_from_provider(self, tmp_path):
        yaml_file = tmp_path / "template.yaml"
        yaml_file.write_text("""
name: Test Template
description: "Test task"
fields:
  input1:
    label: Input 1
    source: user
    kind: scalar
  out1:
    label: Output One
    source: generated
    kind: scalar
export: [out1]
executions: []
""")
        provider = Mock()
        provider.values_for_template.return_value = {"input1": "from-file"}
        app = _StubApp(tmp_path=tmp_path)
        app.task_user_value_provider = provider

        with patch("omnitool.gradio.ui.callbacks.gr") as mock_gr:
            mock_gr.update.return_value = {"visible": True}
            template, _exec, *_inputs = app.on_template_select(str(yaml_file))

        assert template is not None
        provider.values_for_template.assert_called_once_with(template, str(yaml_file))
        input_call_kwargs = mock_gr.update.call_args_list[1].kwargs
        assert input_call_kwargs["value"] == "from-file"

    def test_invalid_yaml_returns_none(self, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("not: valid: yaml: [[[")

        app = _StubApp(tmp_path=tmp_path)
        with patch("omnitool.gradio.ui.callbacks.gr") as mock_gr:
            mock_gr.update.return_value = {"visible": False}
            template, _exec, *_inputs = app.on_template_select(str(bad_file))

        assert template is None

    def test_nonexistent_filepath_returns_none(self, tmp_path):
        app = _StubApp(tmp_path=tmp_path)
        with patch("omnitool.gradio.ui.callbacks.gr") as mock_gr:
            mock_gr.update.return_value = {"visible": False}
            template, _exec, *_inputs = app.on_template_select(str(tmp_path / "ghost.yaml"))

        assert template is None

    def test_yaml_missing_fields_key_returns_none(self, tmp_path):
        yaml_file = tmp_path / "nofields.yaml"
        yaml_file.write_text("name: Test\nexecutions: []\n")

        app = _StubApp(tmp_path=tmp_path)
        with patch("omnitool.gradio.ui.callbacks.gr") as mock_gr:
            mock_gr.update.return_value = {"visible": False}
            template, _exec, *_inputs = app.on_template_select(str(yaml_file))

        assert template is None


# ---------------------------------------------------------------------------
# scan_templates
# ---------------------------------------------------------------------------

class TestScanTemplates:
    def test_empty_directory_returns_empty_list(self, tmp_path):
        from omnitool.gradio.config.task_template import scan_templates
        assert scan_templates(tmp_path) == []

    def test_valid_yaml_returns_label_and_filepath_tuple(self, tmp_path):
        from omnitool.gradio.config.task_template import scan_templates
        yaml_file = tmp_path / "mytemplate.yaml"
        yaml_file.write_text(
            "name: My Template\ndescription: My Task\nfields: {}\nexport: []\nexecutions: []\n"
        )
        choices = scan_templates(tmp_path)
        assert len(choices) == 1
        label, filepath = choices[0]
        assert label == "My Template"
        assert filepath == str(yaml_file)

    def test_invalid_yaml_is_skipped(self, tmp_path):
        from omnitool.gradio.config.task_template import scan_templates
        (tmp_path / "good.yaml").write_text(
            "name: Good\ndescription: Good\nfields: {}\nexport: []\nexecutions: []\n"
        )
        (tmp_path / "bad.yaml").write_text("not: valid: yaml: [[[")
        choices = scan_templates(tmp_path)
        assert len(choices) == 1
        assert choices[0][0] == "Good"

    def test_yml_extension_is_included(self, tmp_path):
        from omnitool.gradio.config.task_template import scan_templates
        (tmp_path / "tmpl.yml").write_text(
            "name: YML Template\ndescription: \"\"\nfields: {}\nexport: []\nexecutions: []\n"
        )
        choices = scan_templates(tmp_path)
        assert len(choices) == 1
        assert choices[0][0] == "YML Template"

    def test_missing_directory_returns_empty_list(self, tmp_path):
        from omnitool.gradio.config.task_template import scan_templates
        assert scan_templates(tmp_path / "nonexistent") == []

    def test_empty_name_falls_back_to_stem(self, tmp_path):
        from omnitool.gradio.config.task_template import scan_templates
        (tmp_path / "myfile.yaml").write_text(
            "description: \"\"\nfields: {}\nexport: []\nexecutions: []\n"
        )
        choices = scan_templates(tmp_path)
        assert len(choices) == 1
        assert choices[0][0] == "myfile"


# ---------------------------------------------------------------------------
# on_app_load
# ---------------------------------------------------------------------------

class TestOnAppLoad:
    def test_no_computer_tool_returns_fallback_message(self, tmp_path):
        tools = Mock()
        tools.get_tool.return_value = None  # ComputerTool not in collection
        app = _StubApp(tmp_path=tmp_path, tools=tools)

        result = app.on_app_load()

        assert isinstance(result, list)
        assert len(result) == 1
        assert "ComputerTool not available" in result[0]["content"]

    def test_screenshot_with_no_image_returns_fallback_message(self, tmp_path):
        screenshot_result = Mock()
        screenshot_result.base64_image = None
        computer_tool = Mock()
        computer_tool.run.return_value = screenshot_result

        tools = Mock()
        tools.get_tool.return_value = computer_tool

        app = _StubApp(tmp_path=tmp_path, tools=tools)
        result = app.on_app_load()

        assert len(result) == 1
        assert "No image data available" in result[0]["content"]

    def test_screenshot_success_returns_img_html(self, tmp_path):
        screenshot_result = Mock()
        screenshot_result.base64_image = "fakebase64data"
        computer_tool = Mock()
        computer_tool.run.return_value = screenshot_result

        tools = Mock()
        tools.get_tool.return_value = computer_tool

        app = _StubApp(tmp_path=tmp_path, tools=tools)
        result = app.on_app_load()

        assert len(result) == 1
        assert "fakebase64data" in result[0]["content"]
        assert result[0]["role"] == "assistant"

    def test_exception_in_screenshot_returns_error_message(self, tmp_path):
        computer_tool = Mock()
        computer_tool.run.side_effect = RuntimeError("connection refused")

        tools = Mock()
        tools.get_tool.return_value = computer_tool

        app = _StubApp(tmp_path=tmp_path, tools=tools)
        result = app.on_app_load()

        assert len(result) == 1
        assert "Failed to capture" in result[0]["content"]
        assert "connection refused" in result[0]["content"]


# ---------------------------------------------------------------------------
# _render_file_list (static method)
# ---------------------------------------------------------------------------

class TestRenderFileList:
    def test_empty_list_returns_no_files_paragraph(self):
        html = GradioCallbacks._render_file_list([])
        assert "No files uploaded" in html

    def test_single_file_name_appears_in_html(self, tmp_path):
        test_file = tmp_path / "report.txt"
        test_file.write_text("hello")
        html = GradioCallbacks._render_file_list([test_file])
        assert "report.txt" in html

    def test_multiple_files_all_names_appear(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.png"
        f1.write_text("x")
        f2.write_bytes(b"\x89PNG")
        html = GradioCallbacks._render_file_list([f1, f2])
        assert "a.txt" in html
        assert "b.png" in html

    def test_file_list_html_is_structured_list(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("content")
        html = GradioCallbacks._render_file_list([f])
        assert "<ul>" in html
        assert "<li>" in html


# ---------------------------------------------------------------------------
# _render_task_summary — HTML escaping for untrusted agent-extracted text
# ---------------------------------------------------------------------------


class TestRenderTaskSummary:
    def test_empty_rows_renders_placeholder(self):
        html = GradioCallbacks._render_task_summary([], None)
        assert "(no rows)" in html

    def test_csv_path_appears_when_provided(self):
        html = GradioCallbacks._render_task_summary(
            [{"k": "v"}], "/tmp/proc_1.csv",
        )
        assert "/tmp/proc_1.csv" in html

    def test_row_values_are_html_escaped(self):
        rows = [{"name": "<script>alert(1)</script>", "tag": "a & b"}]
        html = GradioCallbacks._render_task_summary(rows, None)
        assert "<script>" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
        assert "a &amp; b" in html

    def test_column_names_are_html_escaped(self):
        rows = [{"<bad>": "v"}]
        html = GradioCallbacks._render_task_summary(rows, None)
        assert "<th><bad></th>" not in html
        assert "&lt;bad&gt;" in html

    def test_csv_path_is_html_escaped(self):
        html = GradioCallbacks._render_task_summary(
            [{"k": "v"}], "/tmp/<evil>.csv",
        )
        assert "<evil>" not in html
        assert "&lt;evil&gt;" in html


# ---------------------------------------------------------------------------
# on_file_upload
# ---------------------------------------------------------------------------

class TestOnFileUpload:
    def test_no_files_returns_empty_html(self, tmp_path):
        state = AppState(run_folder=tmp_path)
        app = _StubApp(tmp_path=tmp_path)

        html, returned_state = app.on_file_upload(state, [])
        assert html == ""
        assert returned_state is state

    def test_none_files_returns_empty_html(self, tmp_path):
        state = AppState(run_folder=tmp_path)
        app = _StubApp(tmp_path=tmp_path)

        html, returned_state = app.on_file_upload(state, None)
        assert html == ""

    def test_uploaded_file_name_appears_in_html(self, tmp_path):
        # Create a real file to upload
        src = tmp_path / "upload.txt"
        src.write_text("test content")
        state = AppState(run_folder=tmp_path)
        app = _StubApp(tmp_path=tmp_path)

        html, returned_state = app.on_file_upload(state, [str(src)])

        assert "upload.txt" in html

    def test_none_state_creates_new_app_state(self, tmp_path):
        settings = _make_settings(tmp_path)
        app = _StubApp(tmp_path=tmp_path, settings=settings)

        html, returned_state = app.on_file_upload(None, [])
        assert isinstance(returned_state, AppState)


# ---------------------------------------------------------------------------
# on_submit — event routing logic
# ---------------------------------------------------------------------------

def _make_submit_app(tmp_path, orchestrator_events):
    """Return a (_StubApp, mock_orchestrator) pair.

    Callers must patch 'omnitool.gradio.ui.callbacks.create_agent' with
    mock_orchestrator as the return value — see _run_submit.
    """
    mock_orchestrator = Mock()
    mock_orchestrator.run.return_value = iter(orchestrator_events)
    mock_orchestrator.step_count = len(orchestrator_events)
    app = _StubApp(tmp_path=tmp_path)
    return app, mock_orchestrator


def _run_submit(app, tmp_path, message="test task", extra_kwargs=None, mock_orchestrator=None):
    """Drive on_submit to completion and collect all yielded tuples."""
    state = AppState(run_folder=tmp_path)
    kwargs = dict(
        state=state,
        message=message,
        agent_type="ReActAgent",
        grounding="omniparser",
        preprocessing_mode="raw",
        model_name="gpt-4o",
        provider="openai",
        chatbot_history=[],
        mode=AgentMode.TASK.value,
        platform="windows",
        max_steps=50,
        yaml_template=None,
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    with patch(
        "omnitool.gradio.ui.callbacks.validate_api_key",
        return_value=(True, ""),
    ), patch(
        "omnitool.gradio.ui.callbacks.create_agent",
        return_value=mock_orchestrator,
    ):
        gen = app.on_submit(**kwargs)
        return list(gen)


class TestOnSubmitEventRouting:
    """Each test checks that a single event type is correctly routed into
    chat history by on_submit."""

    def test_user_message_appended_to_history_immediately(self, tmp_path):
        app, mock_orch = _make_submit_app(tmp_path, [{"type": "complete", "total_steps": 0, "total_tokens": 0, "total_cost": 0}])
        updates = _run_submit(app, tmp_path, message="hello world", mock_orchestrator=mock_orch)
        # First yield contains the user message before validation
        first_history = updates[0][0]
        assert any(m.get("content") == "hello world" for m in first_history)

    def test_api_key_invalid_appends_error_and_stops(self, tmp_path):
        app = _StubApp(tmp_path=tmp_path)
        state = AppState(run_folder=tmp_path)

        with patch(
            "omnitool.gradio.ui.callbacks.validate_api_key",
            return_value=(False, "Bad API key"),
        ), patch(
            "omnitool.gradio.ui.callbacks.create_agent",
        ) as mock_factory:
            updates = list(app.on_submit(
                state=state,
                message="task",
                agent_type="ReActAgent",
                grounding="omniparser",
                preprocessing_mode="raw",
                model_name="gpt-4o",
                provider="openai",
                chatbot_history=[],
                mode=AgentMode.TASK.value,
                platform="windows",
                max_steps=50,
                yaml_template=None,
            ))

        # create_agent should never be called when key is invalid
        mock_factory.assert_not_called()
        last_status = updates[-1][2]
        assert "Bad API key" in last_status

    def test_thinking_event_appended_to_history(self, tmp_path):
        events = [
            {"type": "thinking", "response_text": "I am reasoning"},
            {"type": "complete", "total_steps": 1, "total_tokens": 10, "total_cost": 0},
        ]
        app, mock_orch = _make_submit_app(tmp_path, events)
        updates = _run_submit(app, tmp_path, mock_orchestrator=mock_orch)

        all_contents = [m.get("content", "") for update in updates for m in update[0]]
        assert any("I am reasoning" in c for c in all_contents)

    def test_empty_thinking_event_not_appended(self, tmp_path):
        events = [
            {"type": "thinking", "response_text": ""},
            {"type": "complete", "total_steps": 1, "total_tokens": 10, "total_cost": 0},
        ]
        app, mock_orch = _make_submit_app(tmp_path, events)
        updates = _run_submit(app, tmp_path, mock_orchestrator=mock_orch)

        # The thinking message should not appear since it is empty
        all_contents = [m.get("content", "") for update in updates for m in update[0]]
        # Only the user message and the complete status should be present
        non_empty = [c for c in all_contents if c.strip()]
        assert not any("[Think]" in c for c in non_empty)

    def test_action_result_success_appended_to_history(self, tmp_path):
        events = [
            {"type": "action_result", "tool": "click", "output": "clicked", "error": ""},
            {"type": "complete", "total_steps": 1, "total_tokens": 10, "total_cost": 0},
        ]
        app, mock_orch = _make_submit_app(tmp_path, events)
        updates = _run_submit(app, tmp_path, mock_orchestrator=mock_orch)

        all_contents = [m.get("content", "") for update in updates for m in update[0]]
        assert any("click" in c for c in all_contents)
        assert any("[ACT]" in c for c in all_contents)

    def test_action_result_error_uses_fail_prefix(self, tmp_path):
        events = [
            {"type": "action_result", "tool": "click", "output": "", "error": "element not found"},
            {"type": "complete", "total_steps": 1, "total_tokens": 10, "total_cost": 0},
        ]
        app, mock_orch = _make_submit_app(tmp_path, events)
        updates = _run_submit(app, tmp_path, mock_orchestrator=mock_orch)

        all_contents = [m.get("content", "") for update in updates for m in update[0]]
        assert any("[FAIL]" in c for c in all_contents)

    def test_error_event_stops_generator(self, tmp_path):
        events = [
            {"type": "error", "message": "agent crashed"},
        ]
        app, mock_orch = _make_submit_app(tmp_path, events)
        updates = _run_submit(app, tmp_path, mock_orchestrator=mock_orch)

        last_status = updates[-1][2]
        assert "agent crashed" in last_status

    def test_complete_event_status_contains_step_and_token_info(self, tmp_path):
        events = [
            {"type": "complete", "total_steps": 5, "total_tokens": 300, "total_cost": 0.02},
        ]
        app, mock_orch = _make_submit_app(tmp_path, events)
        updates = _run_submit(app, tmp_path, mock_orchestrator=mock_orch)

        last_status = updates[-1][2]
        assert "5" in last_status
        assert "300" in last_status

    def test_plan_event_appended_to_history(self, tmp_path):
        events = [
            {"type": "plan", "plan_text": "Step 1: open browser"},
            {"type": "complete", "total_steps": 1, "total_tokens": 10, "total_cost": 0},
        ]
        app, mock_orch = _make_submit_app(tmp_path, events)
        updates = _run_submit(app, tmp_path, mock_orchestrator=mock_orch)

        all_contents = [m.get("content", "") for update in updates for m in update[0]]
        assert any("[Plan]" in c for c in all_contents)

    def test_status_event_updates_status_text(self, tmp_path):
        events = [
            {"type": "status", "message": "Processing step"},
            {"type": "complete", "total_steps": 1, "total_tokens": 10, "total_cost": 0},
        ]
        app, mock_orch = _make_submit_app(tmp_path, events)
        updates = _run_submit(app, tmp_path, mock_orchestrator=mock_orch)

        all_statuses = [u[2] for u in updates]
        assert "Processing step" in all_statuses

    def test_step_event_formats_step_number_in_status(self, tmp_path):
        events = [
            {"type": "step", "step_num": 3},
            {"type": "complete", "total_steps": 3, "total_tokens": 10, "total_cost": 0},
        ]
        app, mock_orch = _make_submit_app(tmp_path, events)
        updates = _run_submit(app, tmp_path, mock_orchestrator=mock_orch)

        all_statuses = [u[2] for u in updates]
        assert any("3" in s for s in all_statuses)

    def test_max_steps_reached_without_complete_emits_warn_message(self, tmp_path):
        # Generator ends without yielding "complete"
        events = [
            {"type": "step", "step_num": 1},
        ]
        app, mock_orch = _make_submit_app(tmp_path, events)
        mock_orch.step_count = 1
        updates = _run_submit(app, tmp_path, mock_orchestrator=mock_orch)

        last_content_msgs = updates[-1][0]
        last_msg_content = last_content_msgs[-1]["content"] if last_content_msgs else ""
        assert "Stopped" in last_msg_content or "WARN" in last_msg_content

    def test_exception_during_agent_run_yields_error_message(self, tmp_path):
        app = _StubApp(tmp_path=tmp_path)
        state = AppState(run_folder=tmp_path)

        with patch(
            "omnitool.gradio.ui.callbacks.validate_api_key",
            return_value=(True, ""),
        ), patch(
            "omnitool.gradio.ui.callbacks.create_agent",
            side_effect=ValueError("bad config"),
        ):
            updates = list(app.on_submit(
                state=state,
                message="task",
                agent_type="ReActAgent",
                grounding="omniparser",
                preprocessing_mode="raw",
                model_name="gpt-4o",
                provider="openai",
                chatbot_history=[],
                mode=AgentMode.TASK.value,
                platform="windows",
                max_steps=50,
                yaml_template=None,
            ))

        last_status = updates[-1][2]
        assert "bad config" in last_status or "Execution failed" in last_status

    def test_extraction_result_event_appended_to_history(self, tmp_path):
        events = [
            {"type": "complete", "facts": {"price": "9.99"}, "total_steps": 1, "total_tokens": 10, "total_cost": 0},
        ]
        app, mock_orch = _make_submit_app(tmp_path, events)
        updates = _run_submit(app, tmp_path, mock_orchestrator=mock_orch)

        all_contents = [m.get("content", "") for update in updates for m in update[0]]
        assert any("price" in c for c in all_contents)

    def test_assistant_reply_event_appended_to_history(self, tmp_path):
        events = [
            {"type": "assistant_reply", "message": "Task complete!"},
        ]
        app, mock_orch = _make_submit_app(tmp_path, events)
        updates = _run_submit(app, tmp_path, mock_orchestrator=mock_orch)

        all_contents = [m.get("content", "") for update in updates for m in update[0]]
        assert any("Task complete!" in c for c in all_contents)

    def test_table_read_event_appended_to_history(self, tmp_path):
        events = [
            {"type": "table_read", "text": "Name | Price\nWidget | 9.99"},
            {"type": "complete", "total_steps": 1, "total_tokens": 10, "total_cost": 0},
        ]
        app, mock_orch = _make_submit_app(tmp_path, events)
        updates = _run_submit(app, tmp_path, mock_orchestrator=mock_orch)

        all_contents = [m.get("content", "") for update in updates for m in update[0]]
        assert any("[Table]" in c for c in all_contents)

    def test_table_read_empty_text_not_appended(self, tmp_path):
        events = [
            {"type": "table_read", "text": ""},
            {"type": "complete", "total_steps": 1, "total_tokens": 10, "total_cost": 0},
        ]
        app, mock_orch = _make_submit_app(tmp_path, events)
        updates = _run_submit(app, tmp_path, mock_orchestrator=mock_orch)

        all_contents = [m.get("content", "") for update in updates for m in update[0]]
        assert not any("[Table]" in c for c in all_contents)


class TestOnSubmitYamlTemplateIntegration:
    """Verify YAML template overrides message in TASK mode."""

    def test_yaml_template_overrides_message_in_task_mode(self, tmp_path):
        template = _make_template(description="Automated task")

        events = [{"type": "complete", "total_steps": 0, "total_tokens": 0, "total_cost": 0}]
        app, mock_orch = _make_submit_app(tmp_path, events)
        state = AppState(run_folder=tmp_path)

        with patch(
            "omnitool.gradio.ui.callbacks.validate_api_key",
            return_value=(True, ""),
        ), patch(
            "omnitool.gradio.ui.callbacks.create_agent",
            return_value=mock_orch,
        ):
            updates = list(app.on_submit(
                state=state,
                message="original user message",
                agent_type="ReActAgent",
                grounding="omniparser",
                preprocessing_mode="raw",
                model_name="gpt-4o",
                provider="openai",
                chatbot_history=[],
                mode=AgentMode.TASK.value,
                platform="windows",
                max_steps=50,
                yaml_template=template,
            ))

        # Chat history should contain the template description, not "original user message"
        first_history = updates[0][0]
        assert not any(m.get("content") == "original user message" for m in first_history)

    def test_yaml_template_not_applied_in_interactive_mode(self, tmp_path):
        template = _make_template(description="Automated task")

        events = [{"type": "complete", "total_steps": 0, "total_tokens": 0, "total_cost": 0}]
        app, mock_orch = _make_submit_app(tmp_path, events)
        state = AppState(run_folder=tmp_path)

        with patch(
            "omnitool.gradio.ui.callbacks.validate_api_key",
            return_value=(True, ""),
        ), patch(
            "omnitool.gradio.ui.callbacks.create_agent",
            return_value=mock_orch,
        ):
            updates = list(app.on_submit(
                state=state,
                message="original user message",
                agent_type="ReActAgent",
                grounding="omniparser",
                preprocessing_mode="raw",
                model_name="gpt-4o",
                provider="openai",
                chatbot_history=[],
                mode=AgentMode.INTERACTIVE.value,  # NOT task mode
                platform="windows",
                max_steps=50,
                yaml_template=template,
            ))

        first_history = updates[0][0]
        assert any(m.get("content") == "original user message" for m in first_history)


# ---------------------------------------------------------------------------
# _eligible_executions — UI dropdown filter
# ---------------------------------------------------------------------------


def _two_execution_template() -> TaskTemplate:
    """Worked example: execution 1 iterates user_id; execution 2 iterates account_id rows."""
    fields = {
        "user_id": TemplateField(label="User ID", source="user", kind=ColumnKind.SCALAR),
        "account_id": TemplateField(label="Account ID", source="generated", kind=ColumnKind.ROW, expand=True),
        "balance": TemplateField(label="Balance", source="generated", kind=ColumnKind.SCALAR),
    }
    return TaskTemplate(
        name="Two Execution Template",
        description="Pull accounts and enrich each.",
        fields=fields,
        export=["account_id", "balance"],
        executions=[
            TaskExecution(
                id=1, title="Get accounts", tool="cua", system="iWeb",
                foreach="user_id", uses=["user_id"], writes=["account_id"],
                steps="Open <user_id>（User ID） account list.",
                resolved_writes={"account_id": fields["account_id"]},
            ),
            TaskExecution(
                id=2, title="Get balance", tool="cua", system="iWeb",
                foreach="account_id", uses=["account_id"], writes=["balance"],
                steps="Open profile for <account_id>（Account ID）.",
                resolved_writes={"balance": fields["balance"]},
            ),
        ],
    )


# ---------------------------------------------------------------------------
# on_submit — TASK routes through TaskRunner
# ---------------------------------------------------------------------------


class _RunnerCalls:
    def __init__(self):
        self.run_task_called = False
        self.run_once_calls = []
        self.user_values = None


def _patch_runner(runner_calls: _RunnerCalls, events):
    """Patch TaskRunner so its `run_task`/`run_once` yield scripted events."""

    class _StubRunner:
        def __init__(self, *args, **kwargs):
            runner_calls.user_values = kwargs.get("user_values")

        def run_task(self):
            runner_calls.run_task_called = True
            for evt in events:
                yield evt

        def run_once(self, execution):
            runner_calls.run_once_calls.append(execution.id)
            for evt in events:
                yield evt

    return patch("omnitool.gradio.ui.callbacks.TaskRunner", _StubRunner)


def _run_task_submit(
    app,
    tmp_path,
    template,
    execution_selection,
    events,
    task_user_inputs=(),
):
    """Drive on_submit in TASK mode with the runner patched."""
    state = AppState(run_folder=tmp_path)
    runner_calls = _RunnerCalls()
    with patch(
        "omnitool.gradio.ui.callbacks.validate_api_key", return_value=(True, ""),
    ), patch(
        "omnitool.gradio.ui.callbacks.create_agent",
        return_value=Mock(step_count=0),
    ), _patch_runner(runner_calls, events):
        gen = app.on_submit(
            state,
            "ignored; overridden by task description",
            "ReActAgent",
            "omniparser",
            "raw",
            "gpt-4o",
            "openai",
            [],
            AgentMode.TASK.value,
            "windows",
            50,
            template,
            execution_selection,
            *task_user_inputs,
        )
        updates = list(gen)
    return updates, runner_calls


class TestOnSubmitTaskRunnerRouting:
    def test_whole_task_calls_run_task_and_emits_task_complete(self, tmp_path):
        template = _two_execution_template()
        events = [
            {
                "type": "task_execution_start",
                "execution_id": 1,
                "row_idx": 0,
                "row": {"user_id": "U123"},
                "task_string": "Open account list for U123",
            },
            {"type": "execution_complete", "execution_id": 1},
            {"type": "execution_complete", "execution_id": 2},
            {
                "type": "task_complete",
                "success": True,
                "csv_path": str(tmp_path / "task_result.csv"),
                "rows": [{"account_id": "A1", "balance": "10"}],
            },
        ]
        app = _StubApp(tmp_path=tmp_path)
        updates, runner_calls = _run_task_submit(
            app, tmp_path, template, None, events,
        )

        assert runner_calls.run_task_called is True
        assert runner_calls.run_once_calls == []
        all_statuses = [u[2] for u in updates]
        all_content = "\n".join(
            message["content"]
            for update in updates
            for message in update[0]
        )
        assert "Open account list for U123" in all_content
        assert any("Execution 1" in s for s in all_statuses)
        assert any("Task complete" in s for s in all_statuses)

    def test_execution_complete_renders_rows(self, tmp_path):
        template = _two_execution_template()
        events = [
            {
                "type": "execution_complete",
                "execution_id": 1,
                "rows": [{"user_id": "U123", "account_id": "A1"}],
            },
            {
                "type": "task_complete",
                "success": True,
                "csv_path": None,
                "rows": [{"user_id": "U123", "account_id": "A1"}],
            },
        ]
        app = _StubApp(tmp_path=tmp_path)
        updates, _ = _run_task_submit(
            app, tmp_path, template, None, events,
        )

        all_content = "\n".join(
            message["content"]
            for update in updates
            for message in update[0]
        )
        assert "Rows after execution 1" in all_content
        assert "A1" in all_content

    def test_task_user_inputs_are_passed_to_runner(self, tmp_path):
        template = _two_execution_template()
        events = [
            {
                "type": "task_complete",
                "success": True,
                "csv_path": None,
                "rows": [{"account_id": "A1", "balance": "10"}],
            },
        ]
        app = _StubApp(tmp_path=tmp_path)
        _, runner_calls = _run_task_submit(
            app,
            tmp_path,
            template,
            None,
            events,
            task_user_inputs=("U123",),
        )

        assert runner_calls.user_values == {"user_id": "U123"}

    def test_single_execution_selection_calls_run_once(self, tmp_path):
        template = _two_execution_template()
        events = [
            {"type": "complete", "facts": {}, "total_steps": 1, "total_tokens": 10, "total_cost": 0},
        ]
        app = _StubApp(tmp_path=tmp_path)
        updates, runner_calls = _run_task_submit(
            app, tmp_path, template, 1, events,
        )

        assert runner_calls.run_task_called is False
        assert runner_calls.run_once_calls == [1]
        last_status = updates[-1][2]
        assert "[OK] Complete" in last_status

    def test_task_complete_failure_status_is_error(self, tmp_path):
        template = _two_execution_template()
        events = [
            {
                "type": "task_complete",
                "success": False,
                "csv_path": None,
                "rows": [],
            },
        ]
        app = _StubApp(tmp_path=tmp_path)
        updates, _ = _run_task_submit(
            app, tmp_path, template, None, events,
        )
        last_status = updates[-1][2]
        assert "Task aborted" in last_status

    def test_unknown_execution_id_falls_back_to_run_task(self, tmp_path):
        """An execution id that no execution claims falls back to the whole task."""
        template = _two_execution_template()
        events = [
            {
                "type": "task_complete",
                "success": True,
                "csv_path": None,
                "rows": [],
            },
        ]
        app = _StubApp(tmp_path=tmp_path)
        _, runner_calls = _run_task_submit(
            app, tmp_path, template, 999, events,
        )
        assert runner_calls.run_task_called is True
        assert runner_calls.run_once_calls == []


# ---------------------------------------------------------------------------
# Import path correctness — new module paths after refactor
# ---------------------------------------------------------------------------

class TestRefactoredImportPaths:
    def test_services_app_state_importable(self):
        from omnitool.gradio.services import AppState
        assert AppState is not None

    def test_services_auth_validator_importable(self):
        from omnitool.gradio.services import AuthValidator
        assert AuthValidator is not None

    def test_services_file_handler_importable(self):
        from omnitool.gradio.services import FileHandler
        assert FileHandler is not None

    def test_ui_callbacks_importable(self):
        from omnitool.gradio.ui.callbacks import GradioCallbacks
        assert GradioCallbacks is not None

    def test_ui_components_formatters_importable(self):
        from omnitool.gradio.ui.components import format_action_result
        assert format_action_result is not None

    def test_ui_app_importable(self):
        from omnitool.gradio.ui.app import GradioApp
        assert GradioApp is not None

    def test_no_old_ui_gradio_path(self):
        """The old omnitool.gradio.ui.gradio path must not exist."""
        import importlib
        import importlib.util
        spec = importlib.util.find_spec("omnitool.gradio.ui.gradio")
        assert spec is None, "Old ui.gradio subpackage still exists — remove it"
