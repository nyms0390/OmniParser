"""
Gradio event handlers for OmniParser UI.

Extracted from app.py so that UI layout (build_interface) and event dispatch
(on_submit, on_app_load, etc.) can be read and modified independently.
"""

import logging
from html import escape
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import gradio as gr

from omnitool.gradio.config import AgentMode, TaskTemplate, load_task_template
from omnitool.gradio.config.task_template import TaskExecution, TaskProcedure
from omnitool.gradio.core import create_agent
from omnitool.gradio.core.procedure_runner import ProcedureRunner
from omnitool.gradio.services import AppState, FileHandler, validate_api_key
from omnitool.gradio.ui.components import (
    format_action_result,
    format_extraction_result,
    format_field_saved,
    format_focus_region,
    format_grounding,
    format_ledger,
    format_parsed_screen,
    format_plan,
    format_raw_screen,
    format_compaction,
    format_table_read,
    format_thinking,
    get_provider_options_for_model,
    render_image,
)

logger = logging.getLogger(__name__)


def _eligible_executions(
    procedure: TaskProcedure, template: TaskTemplate,
) -> List[TaskExecution]:
    """Executions runnable standalone — every input key resolves to a template scalar.

    Executions whose ``inputs`` reference a row-kind procedure output need a
    populated dataframe, which only the whole-procedure run produces.
    """
    template_keys = {inp.key for inp in template.inputs}
    return [
        e for e in procedure.executions
        if all(k in template_keys for k in e.inputs)
    ]


def _execution_choices(
    procedure: TaskProcedure, template: TaskTemplate,
) -> List[Tuple[str, Optional[int]]]:
    """Gradio (label, value) choices. ``None`` value = whole procedure;
    ``int`` value = a specific execution id."""
    return [("Whole procedure", None)] + [
        (f"Execution {e.id}", e.id)
        for e in _eligible_executions(procedure, template)
    ]


def _hidden_execution_dropdown():
    """Reset the execution dropdown to its hidden default (whole-procedure only)."""
    return gr.update(
        choices=[("Whole procedure", None)], value=None, visible=False,
    )


class GradioCallbacks:
    """Mixin providing all Gradio event handler methods for GradioApp."""

    def on_app_load(self) -> list:
        """Capture initial screenshot on app startup.

        Returns:
            Initial chatbot history with screenshot
        """
        try:
            # Capture screenshot via ComputerTool
            computer_tool = self.tools.get_tool("computer")
            if not computer_tool:
                logger.warning("ComputerTool not available in tools collection")
                return [{
                    "role": "assistant",
                    "content": "Initial screenshot: ComputerTool not available",
                }]

            screenshot_result = computer_tool.run("screenshot")
            screenshot_base64 = screenshot_result.base64_image

            if not screenshot_base64:
                logger.warning("Screenshot returned but no image data")
                return [{
                    "role": "assistant",
                    "content": "Initial screenshot: No image data available",
                }]

            img_html = render_image(screenshot_base64, hint=True)
            return [{"role": "assistant", "content": f"Initial desktop state:\n\n{img_html}"}]

        except Exception as e:
            error_msg = f"Failed to capture initial screenshot: {str(e)}"
            logger.error(error_msg)
            return [{"role": "assistant", "content": error_msg}]

    def on_agent_change(self, agent_type: str):
        """Show the grounding dropdown for VLMAgent and ReActAgent."""
        return gr.update(visible=(agent_type in ("VLMAgent", "ReActAgent")))

    def on_model_change(self, model_name: str) -> Tuple:
        """Handle model selection change.

        Args:
            model_name: Selected model name

        Returns:
            Updated provider choices
        """
        providers = get_provider_options_for_model(model_name)
        return gr.update(
            choices=providers,
            value=providers[0] if providers else "",
        )

    def on_submit(
        self,
        state,
        message: str,
        agent_type: str,
        grounding: str,
        preprocessing_mode: str,
        model_name: str,
        provider: str,
        chatbot_history,
        mode: str,
        platform: str,
        max_steps: int,
        yaml_template,
        selected_procedure_id,
        execution_selection: Optional[int] = None,
    ) -> Generator:
        """Handle submit button click.

        This is a sync generator - Gradio 4+ auto-detects generators and
        streams each ``yield`` as an incremental UI update.

        Args:
            state: App state
            message: User message
            model_name: Selected model
            provider: Selected provider
            chatbot_history: Chat history

        Yields:
            Tuple of (chatbot_history, message_input, status_text, state)
        """
        # Initialize state if needed
        if state is None or not isinstance(state, AppState):
            state = AppState(run_folder=Path(self.settings.run_folder))

        history = list(chatbot_history) if chatbot_history else []

        # Resolve mode enum once — used throughout this method.
        try:
            agent_mode = AgentMode(mode)
        except ValueError:
            agent_mode = AgentMode.INTERACTIVE
        in_task_mode = agent_mode == AgentMode.TASK

        # Resolve the selected procedure and override message only in TASK mode.
        yaml_procedure = None
        if in_task_mode and isinstance(yaml_template, TaskTemplate):
            try:
                yaml_procedure = yaml_template.get_procedure(selected_procedure_id)
            except (ValueError, TypeError):
                yaml_procedure = yaml_template.procedures[0]
            if yaml_procedure is not None:
                message = yaml_procedure.description
        # Add user message
        state.chat.add_message("user", message)
        history.append({"role": "user", "content": message})

        # Yield immediately so the user message appears right away
        yield history, "", "Validating...", state

        # Validate API key
        is_valid, error_msg = validate_api_key(
            provider,
            azure_endpoint=self.settings.azure_endpoint if provider == "azure" else None
        )

        if not is_valid:
            history.append({"role": "assistant", "content": f"Error: {error_msg}"})
            yield history, "", error_msg, state
            return

        # Create orchestrator
        try:
            # Prepare orchestrator kwargs
            orchestrator_kwargs = {
                "model_name": model_name,
                "state": state,
                "tools_collection": self.tools,
                "omniparser_client": self.omniparser_client,
                "save_folder": state.session.run_folder,
                "max_steps": max_steps,
                "provider": provider,
                "mode": agent_mode,
                "platform": platform,
                "azure_endpoint": self.settings.azure_endpoint,
                "gta1_client": self.gta1_client,
                "paddleocr_client": self.paddleocr_client,
                "grounding": grounding,
                "preprocessing_mode": preprocessing_mode,
                "task_procedure": yaml_procedure,
            }

            if in_task_mode and yaml_procedure is not None:
                # TASK + procedure: drive multiple agents via ProcedureRunner.
                base_kwargs = {
                    k: v for k, v in orchestrator_kwargs.items()
                    if k != "task_procedure"
                }

                def agent_factory(execution: TaskExecution, task_string: str):
                    agent = create_agent(
                        **base_kwargs,
                        task_procedure=yaml_procedure,
                        task_execution=execution,
                    )
                    agent.working_memory.task = task_string
                    return agent

                runner = ProcedureRunner(
                    yaml_procedure, yaml_template, agent_factory,
                    Path(state.session.run_folder),
                )
                if execution_selection is None:
                    event_gen = runner.run()
                else:
                    try:
                        execution = next(
                            e for e in yaml_procedure.executions
                            if e.id == execution_selection
                        )
                        event_gen = runner.run_execution(execution)
                    except StopIteration:
                        event_gen = runner.run()
                self.orchestrator = None
            else:
                self.orchestrator = create_agent(**orchestrator_kwargs)
                event_gen = self.orchestrator.run()

            # Stream sampling loop updates to the chatbot
            status = "Running..."
            is_first_screen = True  # Auto-expand the initial screen capture
            loop_complete = False
            for update in event_gen:
                update_type = update.get("type", "")

                if update_type == "parsed_screen":
                    som_b64 = update.get("som_image_base64", "")
                    raw_b64 = update.get("raw_image_base64", "")
                    if som_b64:
                        screen_html = format_parsed_screen(
                            som_image_base64=som_b64,
                            screen_info=update.get("screen_info", ""),
                            auto_expand=is_first_screen,
                        )
                    elif raw_b64:
                        screen_html = format_raw_screen(
                            raw_image_base64=raw_b64,
                            auto_expand=is_first_screen,
                        )
                    else:
                        screen_html = None  # nothing to show
                    is_first_screen = False
                    if screen_html:
                        history.append({"role": "assistant", "content": screen_html})
                        status = "Screen captured"
                        yield history, "", status, state

                elif update_type == "grounding":
                    grounding_html = format_grounding(update.get("events", []))
                    if grounding_html:
                        history.append({"role": "assistant", "content": grounding_html})
                        yield history, "", "Grounding resolved", state

                elif update_type == "thinking":
                    # Render LLM reasoning - only shown when non-empty
                    thinking_html = format_thinking(update.get("response_text", ""))
                    if thinking_html is not None:
                        history.append({"role": "assistant", "content": thinking_html})
                        yield history, "", "Agent is thinking...", state

                elif update_type == "compaction":
                    compaction_html = format_compaction(update.get("summary", ""))
                    if compaction_html is not None:
                        history.append({"role": "assistant", "content": compaction_html})
                        yield history, "", "History compacted", state

                elif update_type == "plan":
                    plan_html = format_plan(update.get("plan_text", ""))
                    history.append({"role": "assistant", "content": plan_html})
                    yield history, "", "Plan generated", state

                elif update_type == "ledger":
                    ledger_html = format_ledger(update.get("ledger_text", ""))
                    history.append({"role": "assistant", "content": ledger_html})
                    yield history, "", "Ledger updated", state

                elif update_type == "action_result":
                    # Render the executed action and optional post-action screenshot
                    action_html = format_action_result(
                        tool_name=update.get("tool", "unknown"),
                        output=update.get("output", ""),
                        error=update.get("error", ""),
                        base64_image=update.get("base64_image", ""),
                    )
                    history.append({"role": "assistant", "content": action_html})
                    status = f"Executed: {update.get('tool', 'unknown')}"
                    yield history, "", status, state

                elif update_type == "status":
                    status = update.get("message", "")
                    yield history, "", status, state

                elif update_type == "step":
                    status = f"Step {update.get('step_num', '?')}..."
                    yield history, "", status, state

                elif update_type == "progress":
                    status = (
                        f"Step {update.get('step', '?')} - "
                        f"Tokens: {update.get('tokens_total', 0)}, "
                        f"Cost: {update.get('cost_total', '$0')}"
                    )
                    yield history, "", status, state

                elif update_type == "assistant_reply":
                    # LLM gave a conversational response (no tool calls = done)
                    reply_msg = update.get("message", "")
                    if reply_msg:
                        history.append({"role": "assistant", "content": reply_msg})
                    yield history, "", "Agent finished", state

                elif update_type == "focus_region":
                    img_b64 = update.get("image_base64", "")
                    if img_b64:
                        history.append({
                            "role": "assistant",
                            "content": format_focus_region(img_b64),
                        })
                    yield history, "", status, state

                elif update_type == "screen_reading":
                    fields = update.get("fields", {})
                    if fields:
                        history.append({
                            "role": "assistant",
                            "content": format_extraction_result(fields),
                        })
                    yield history, "", status, state

                elif update_type == "field_saved":
                    saved_html = format_field_saved(
                        text=update.get("text", ""),
                        fields=update.get("fields", {}),
                    )
                    history.append({"role": "assistant", "content": saved_html})
                    yield history, "", status, state

                elif update_type == "table_read":
                    table_html = format_table_read(update.get("text", ""))
                    if table_html:
                        history.append({"role": "assistant", "content": table_html})
                    yield history, "", status, state

                elif update_type == "complete":
                    facts = update.get("facts", {})
                    if facts:
                        history.append({
                            "role": "assistant",
                            "content": format_extraction_result(facts),
                        })
                    status = (
                        f"[OK] Complete - "
                        f"Steps: {update.get('total_steps')}, "
                        f"Tokens: {update.get('total_tokens')}, "
                        f"Cost: {update.get('total_cost')}"
                    )
                    history.append({"role": "assistant", "content": status})
                    loop_complete = True
                    yield history, "", status, state
                    return  # End of execution

                elif update_type == "error":
                    status = f"[ERROR]: {update.get('message')}"
                    history.append({"role": "assistant", "content": status})
                    yield history, "", status, state
                    return

                elif update_type == "execution_complete":
                    status = f"Execution {update.get('execution_id', '?')} complete"
                    history.append({"role": "assistant", "content": status})
                    loop_complete = True
                    yield history, "", status, state

                elif update_type == "procedure_complete":
                    summary = self._render_procedure_summary(
                        update.get("rows", []),
                        update.get("csv_path"),
                    )
                    if summary:
                        history.append({"role": "assistant", "content": summary})
                    if update.get("success", True):
                        status = "[OK] Procedure complete"
                    else:
                        status = "[ERROR] Procedure aborted"
                    history.append({"role": "assistant", "content": status})
                    loop_complete = True
                    yield history, "", status, state
                    return

            # Loop ended without explicit complete/error (hit max_steps)
            if not loop_complete:
                step_count = getattr(self.orchestrator, "step_count", "?")
                status = f"[WARN] Stopped after {step_count} steps (max reached)"
                history.append({"role": "assistant", "content": status})
                yield history, "", status, state

        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            history.append({"role": "assistant", "content": f"[ERROR] {error_msg}"})
            yield history, "", error_msg, state

    def on_mode_change(self, mode: str):
        """Show/hide the YAML upload widget based on selected mode.

        Args:
            mode: Selected agent mode string.

        Returns:
            Gradio update for the yaml_upload component visibility.
        """
        return gr.update(visible=(mode == AgentMode.TASK.value))

    def on_yaml_upload(self, file):
        """Parse an uploaded YAML task template file.

        Args:
            file: Gradio file object (has a ``.name`` filepath attribute),
                or None if cleared.

        Returns:
            Tuple of (template, procedure_dropdown_update, execution_dropdown_update).
        """
        if file is None:
            return (
                None,
                gr.update(choices=[], value=None, visible=False),
                _hidden_execution_dropdown(),
            )
        try:
            template = load_task_template(file.name)
            proc_choices = [(f"[{p.id}] {p.description}", p.id) for p in template.procedures]
            first = template.procedures[0]
            exec_choices = _execution_choices(first, template)
            return (
                template,
                gr.update(choices=proc_choices, value=first.id, visible=True),
                gr.update(choices=exec_choices, value=None, visible=True),
            )
        except Exception as exc:
            logger.warning("Failed to load task template: %s", exc)
            return (
                None,
                gr.update(choices=[], value=None, visible=False),
                _hidden_execution_dropdown(),
            )

    def on_procedure_change(self, procedure_id, template):
        """Repopulate the execution dropdown when the procedure selection changes."""
        if not isinstance(template, TaskTemplate) or procedure_id is None:
            return _hidden_execution_dropdown()
        try:
            procedure = template.get_procedure(procedure_id)
        except (ValueError, TypeError):
            return _hidden_execution_dropdown()
        return gr.update(
            choices=_execution_choices(procedure, template),
            value=None,
            visible=True,
        )

    @staticmethod
    def _render_procedure_summary(rows, csv_path) -> str:
        """Render the dataframe rows as an HTML table plus an optional CSV link.

        Cell values originate from agent screen reads — untrusted text — so
        every interpolated value is HTML-escaped to prevent layout breakage
        and stored-XSS in the chatbot pane.
        """
        if not rows:
            body = "<p>(no rows)</p>"
        else:
            columns = list(rows[0].keys())
            header = "".join(f"<th>{escape(str(c))}</th>" for c in columns)
            body_rows = "".join(
                "<tr>" + "".join(
                    f"<td>{escape(str(r.get(c, '')))}</td>" for c in columns
                ) + "</tr>"
                for r in rows
            )
            body = (
                f"<table border='1'><thead><tr>{header}</tr></thead>"
                f"<tbody>{body_rows}</tbody></table>"
            )
        if csv_path:
            body += f"<p>CSV: <code>{escape(str(csv_path))}</code></p>"
        return body

    def on_file_upload(self, state, files) -> Tuple:
        """Handle file upload.

        Args:
            state: App state
            files: Uploaded files

        Returns:
            File viewer HTML, updated state
        """
        if state is None or not isinstance(state, AppState):
            state = AppState(run_folder=Path(self.settings.run_folder))

        if not files:
            return "", state

        # Save files to session folder
        output_folder = state.session.run_folder / "uploads"
        for file_path in files:
            FileHandler.upload_file(
                file_path,
                output_folder,
                Path(file_path).name,
            )
            state.files.add_file(Path(file_path))

        # Render file viewer
        file_list_html = self._render_file_list(state.files.get_files())

        return file_list_html, state

    @staticmethod
    def _render_file_list(files: list) -> str:
        """Render file list HTML.

        Args:
            files: List of files

        Returns:
            HTML string
        """
        if not files:
            return "<p>No files uploaded</p>"

        html_parts = ["<div><h4>Uploaded Files:</h4><ul>"]
        for f in files:
            info = FileHandler.get_file_info(f)
            html_parts.append(f"<li>{info['name']} ({info['size_mb']:.2f} MB)</li>")
        html_parts.append("</ul></div>")

        return "\n".join(html_parts)
