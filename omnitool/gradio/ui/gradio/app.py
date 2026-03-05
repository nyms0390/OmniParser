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
from typing import Generator, Tuple

import gradio as gr
from PIL import Image

from omnitool.gradio.clients.external import OmniParserClient, PaddleOCRClient, WindowsHostClient, ServiceValidator
from omnitool.gradio.config import (
    AgentMode,
    create_argument_parser,
    get_settings,
    setup_logging,
)
from omnitool.gradio.core import (
    create_agent,
    ToolCollection,
)
from omnitool.gradio.app import AppState, FileHandler, validate_api_key
from omnitool.gradio.ui.gradio.components import (
    render_image,
    format_action_result,
    format_grounding,
    format_ledger,
    format_parsed_screen,
    format_plan,
    format_raw_screen,
    format_thinking,
    get_model_choices,
    get_provider_options_for_model,
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
                    model_dropdown = gr.Dropdown(
                        choices=get_model_choices(),
                        value="gta1 + gpt-4o",
                        label="Model",
                    )
                    provider_dropdown = gr.Dropdown(
                        choices=get_provider_options_for_model("gta1 + gpt-4o"),
                        value="azure",
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
                    model_dropdown,
                    provider_dropdown,
                    chatbot,
                    mode_dropdown,
                    platform_dropdown,
                    context_n_slider,
                    max_steps_slider,
                ],
                outputs=[
                    chatbot,
                    message_input,
                    status_text,
                    state_var,
                ],
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
                return [{"role": "system", "content": "Initial screenshot: ComputerTool not available"}]
            
            screenshot_result = computer_tool.run("screenshot")
            screenshot_base64 = screenshot_result.base64_image
            
            if not screenshot_base64:
                logger.warning("Screenshot returned but no image data")
                return [{"role": "system", "content": "Initial screenshot: No image data available"}]
            
            img_html = render_image(screenshot_base64, hint=True)
            return [{"role": "system", "content": f"Initial desktop state:\n\n{img_html}"}]
        
        except Exception as e:
            error_msg = f"Failed to capture initial screenshot: {str(e)}"
            logger.error(error_msg)
            return [{"role": "system", "content": error_msg}]
    
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
        model_name: str,
        provider: str,
        chatbot_history,
        mode: str,
        platform: str,
        context_n: int,
        max_steps: int,
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
            
            # Prepare orchestrator kwargs
            orchestrator_kwargs = {
                "model_name": model_name,
                "state": state,
                "tools_collection": self.tools,
                "omniparser_client": self.omniparser_client,
                "save_folder": Path(self.settings.run_folder),
                "max_steps": max_steps,
                "provider": provider,
                "mode": agent_mode,
                "platform": platform,
                "context_n": context_n,
            }
            
            # Add Azure endpoint if using Azure provider
            if provider == "azure":
                orchestrator_kwargs["azure_endpoint"] = self.settings.azure_endpoint
            
            self.orchestrator = create_agent(**orchestrator_kwargs)
            
            # Stream sampling loop updates to the chatbot
            status = "Running..."
            is_first_screen = True  # Auto-expand the initial screen capture
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
                
                elif update_type == "complete":
                    status = (
                        f"[OK] Complete - "
                        f"Steps: {update.get('total_steps')}, "
                        f"Tokens: {update.get('total_tokens')}, "
                        f"Cost: {update.get('total_cost')}"
                    )
                    history.append({"role": "assistant", "content": status})
                    yield history, "", status, state
                    return
                
                elif update_type == "error":
                    status = f"[ERROR]: {update.get('message')}"
                    history.append({"role": "assistant", "content": status})
                    yield history, "", status, state
                    return
            
            # Loop ended without explicit complete/error (hit max_steps)
            status = f"[WARN] Stopped after {self.orchestrator.step_count} steps (max reached)"
            history.append({"role": "assistant", "content": status})
            yield history, "", status, state
        
        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            history.append({"role": "assistant", "content": f"[ERROR] {error_msg}"})
            yield history, "", error_msg, state
    
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
