"""
Main Gradio application entry point.

This is the monolithic app.py that consolidates all Gradio functionality.
All state management, callbacks, and UI logic centered here, with helper
components imported from components/ subdirectory.

To run:
    python omnitool/gradio_refactored/ui/gradio/app.py

Environment variables (or set via --config-file):
    OPENAI_API_KEY: OpenAI API key
    ANTHROPIC_API_KEY: Anthropic API key
    GROQ_API_KEY: Groq API key
    DASHSCOPE_API_KEY: DashScope API key
    AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL
    OMNIPARSER_URL: OmniParser server URL (default: http://localhost:8000)
    WINDOWS_HOST_URL: Windows host URL (default: http://localhost:8006)
"""

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple

import gradio as gr
from PIL import Image

from omnitool.gradio.clients.external import (
    OmniParserClient, PaddleOCRClient, WindowsHostClient, ServiceValidator,
)
from omnitool.gradio.config import (
    AgentMode,
    create_argument_parser,
    get_settings,
    setup_logging,
    TaskProcedure,
    load_task_template,
)
from omnitool.gradio.core import (
    create_agent,
    ToolCollection,
)
from omnitool.gradio.app import AppState, FileHandler, validate_api_key
from omnitool.gradio.ui.gradio.components import (
    render_image,
    format_action_result,
    format_extraction_result,
    format_grounding,
    format_ledger,
    format_parsed_screen,
    format_plan,
    format_raw_screen,
    format_thinking,
    get_agent_choices,
    get_model_choices,
    get_provider_options_for_model,
    DEFAULT_AGENT,
    DEFAULT_MODEL,
    GROUNDING_CHOICES,
    DEFAULT_GROUNDING,
)

logger = None


class GradioApp:
    """Main Gradio application."""

    def __init__(self, settings):
        """Initialize Gradio application.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.state = None  # Will be created in Gradio
        self.omniparser_client = OmniParserClient(settings.omniparser_url)
        self.windows_host_client = WindowsHostClient(settings.windows_host_url)
        self.paddleocr_client = PaddleOCRClient(settings.paddleocr_url)
        self.tools = ToolCollection(windows_host_client=self.windows_host_client)
        self.orchestrator = None

        # Validate all services on startup
        validator = ServiceValidator()
        validator.register("OmniParser", self.omniparser_client)
        validator.register("Windows Host", self.windows_host_client)
        validator.register("PaddleOCR", self.paddleocr_client)
        validator.validate_all()

    def build_interface(self):
        """Build Gradio interface."""
        with gr.Blocks(title="OmniParser") as interface:
            # Header
            gr.Markdown("# OmniParser")
            gr.Markdown("Vision-Language Model for Computer Interaction")

            # Initialize state
            state_var = gr.State()

            # Settings panel
            with gr.Accordion(label="Settings"):
                with gr.Row():
                    agent_dropdown = gr.Dropdown(
                        choices=get_agent_choices(),
                        value=DEFAULT_AGENT,
                        label="Agent",
                    )
                    grounding_dropdown = gr.Dropdown(
                        choices=GROUNDING_CHOICES,
                        value=DEFAULT_GROUNDING,
                        label="Grounding",
                        visible=True,  # shown for VLMAgent and ReActAgent
                    )
                    model_dropdown = gr.Dropdown(
                        choices=get_model_choices(),
                        value=DEFAULT_MODEL,
                        label="Model",
                    )
                    provider_dropdown = gr.Dropdown(
                        choices=get_provider_options_for_model(DEFAULT_MODEL),
                        value=(
                            get_provider_options_for_model(DEFAULT_MODEL)[0]
                            if get_provider_options_for_model(DEFAULT_MODEL)
                            else ""
                        ),
                        label="Provider",
                    )

                with gr.Row():
                    context_n_slider = gr.Slider(
                        minimum=0,
                        maximum=50,
                        step=1,
                        value=10,
                        label="Context: last N messages (0 = all)",
                    )
                    max_steps_slider = gr.Slider(
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=20,
                        label="Max steps",
                    )

                with gr.Row():
                    mode_dropdown = gr.Dropdown(
                        choices=[
                            ("Interactive", AgentMode.INTERACTIVE.value),
                            ("Orchestrated", AgentMode.ORCHESTRATED.value),
                            ("Task", AgentMode.TASK.value),
                        ],
                        value=AgentMode.ORCHESTRATED.value,
                        label="Mode",
                    )
                    platform_dropdown = gr.Dropdown(
                        choices=["windows", "macos", "linux", "generic"],
                        value="windows",
                        label="Platform",
                    )

                # Show grounding dropdown only for ReActAgent
                agent_dropdown.change(
                    fn=self.on_agent_change,
                    inputs=[agent_dropdown],
                    outputs=[grounding_dropdown],
                )

                # Update provider options when model changes
                model_dropdown.change(
                    fn=self.on_model_change,
                    inputs=[model_dropdown],
                    outputs=[provider_dropdown],
                )

            # Status and progress
            with gr.Accordion(label="Execution"):
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3,
                )
                gr.Progress()  # reserved for future progress tracking

            # Chat interface
            with gr.Accordion(label="Chat"):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    sanitize_html=False,
                )

                with gr.Row():
                    with gr.Column(scale=4):
                        message_input = gr.Textbox(
                            placeholder="Enter your request...",
                            show_label=False,
                        )
                    with gr.Column(scale=1):
                        submit_button = gr.Button("Send")

                with gr.Row():
                    extract_fields_input = gr.Textbox(
                        label="Extract fields after task (optional)",
                        placeholder=(
                            "e.g. price: 2 decimal places\n"
                            "status\nunified_number: 4 digits"
                        ),
                        lines=3,
                        show_label=True,
                    )

                # YAML task template upload — visible only in TASK mode
                yaml_template_state = gr.State(None)
                with gr.Row():
                    yaml_upload = gr.File(
                        label="Task Template (YAML) — TASK mode only",
                        file_count="single",
                        file_types=[".yaml", ".yml"],
                        visible=False,
                    )

            # Capture initial screenshot on app load (non-blocking)
            interface.load(
                fn=self.on_app_load,
                outputs=[chatbot],
            )

            # File upload and viewer
            with gr.Accordion(label="Files"):
                file_upload = gr.File(
                    label="Upload Files",
                    file_count="multiple",
                    type="filepath",
                )
                file_viewer = gr.HTML(label="File Viewer")

            # Wire up interactions
            submit_button.click(
                fn=self.on_submit,
                inputs=[
                    state_var,
                    message_input,
                    agent_dropdown,
                    grounding_dropdown,
                    model_dropdown,
                    provider_dropdown,
                    chatbot,
                    mode_dropdown,
                    platform_dropdown,
                    context_n_slider,
                    max_steps_slider,
                    extract_fields_input,
                    yaml_template_state,
                ],
                outputs=[
                    chatbot,
                    message_input,
                    status_text,
                    state_var,
                ],
            )

            mode_dropdown.change(
                fn=self.on_mode_change,
                inputs=[mode_dropdown],
                outputs=[yaml_upload],
            )

            yaml_upload.change(
                fn=self.on_yaml_upload,
                inputs=[yaml_upload],
                outputs=[yaml_template_state],
            )

            file_upload.change(
                fn=self.on_file_upload,
                inputs=[state_var, file_upload],
                outputs=[file_viewer, state_var],
            )

        return interface

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
        model_name: str,
        provider: str,
        chatbot_history,
        mode: str,
        platform: str,
        context_n: int,
        max_steps: int,
        extract_fields_raw: str,
        yaml_template,
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

        # When a YAML template is loaded in TASK mode, override message and extract_fields.
        if yaml_template is not None and mode == AgentMode.TASK.value:
            message = yaml_template.to_task_string()
            extract_fields_raw = "\n".join(
                f"{k}: {v}" for k, v in yaml_template.to_extract_fields().items()
            )

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
            # Resolve mode enum
            try:
                agent_mode = AgentMode(mode)
            except ValueError:
                agent_mode = AgentMode.INTERACTIVE

            # Parse extract_fields — supports "field: constraint" per line or plain names.
            extract_fields: Dict[str, str] = {}
            for line in extract_fields_raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                if ":" in line:
                    name, _, constraint = line.partition(":")
                    extract_fields[name.strip()] = constraint.strip()
                else:
                    extract_fields[line] = ""

            # Prepare orchestrator kwargs
            orchestrator_kwargs = {
                "agent_type": agent_type,
                "model_name": model_name,
                "state": state,
                "tools_collection": self.tools,
                "omniparser_client": self.omniparser_client,
                "save_folder": state.session.run_folder,
                "max_steps": max_steps,
                "provider": provider,
                "mode": agent_mode,
                "platform": platform,
                "context_n": context_n,
                "extract_fields": extract_fields,
                "azure_endpoint": self.settings.azure_endpoint,
                "gta1_url": self.settings.gta1_url,
                "grounding": grounding,
            }

            self.orchestrator = create_agent(**orchestrator_kwargs)

            if yaml_template is not None and mode == AgentMode.TASK.value:
                self.orchestrator.task_template = yaml_template

            # Stream sampling loop updates to the chatbot
            status = "Running..."
            is_first_screen = True  # Auto-expand the initial screen capture
            loop_complete = False
            for update in self.orchestrator.run():
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

                elif update_type == "screen_reading":
                    fields = update.get("fields", {})
                    if fields:
                        history.append({
                            "role": "assistant",
                            "content": format_extraction_result(fields),
                        })
                    yield history, "", status, state

                elif update_type == "extraction_result":
                    fields = update.get("fields", {})
                    collected_facts = update.get("collected_facts", {})
                    if fields:
                        history.append({
                            "role": "assistant",
                            "content": format_extraction_result(fields),
                        })
                    if collected_facts:
                        history.append({
                            "role": "assistant",
                            "content": (
                                "**Collected facts (mid-loop readings)**\n"
                                + format_extraction_result(collected_facts)
                            ),
                        })
                    yield history, "", "Extraction complete", state

                elif update_type == "complete":
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

            # Loop ended without explicit complete/error (hit max_steps)
            if not loop_complete:
                status = f"[WARN] Stopped after {self.orchestrator.step_count} steps (max reached)"
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

    def on_yaml_upload(self, file) -> Optional[TaskProcedure]:
        """Parse an uploaded YAML task template file.

        Args:
            file: Gradio file object (has a ``.name`` filepath attribute),
                or None if cleared.

        Returns:
            Parsed :class:`TaskProcedure` (first procedure), or None.
        """
        if file is None:
            return None
        try:
            template = load_task_template(file.name)
            return template
        except Exception as exc:
            logger.warning("Failed to load task template: %s", exc)
            return None

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
    def _render_file_list(files: list[Path]) -> str:
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


def main():
    """Main entry point."""
    global logger

    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup logging (BEFORE loading settings)
    log_file = args.log_file or "omniparser_app.log"
    logger = setup_logging("omniparser_app", level=args.log_level, log_file=log_file)

    # Load settings
    settings = get_settings(args)

    # Create and launch app
    app = GradioApp(settings)
    interface = app.build_interface()
    interface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
    )


if __name__ == "__main__":
    main()
