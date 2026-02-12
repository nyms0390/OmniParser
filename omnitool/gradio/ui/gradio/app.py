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

import asyncio
import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
from PIL import Image

from omnitool.gradio.clients import OmniParserClient, PaddleOCRClient, WindowsHostClient
from omnitool.gradio.clients.services import ServiceValidator
from omnitool.gradio.config import (
    APIProvider,
    create_argument_parser,
    get_all_model_names,
    get_model_config,
    get_settings,
    setup_logging,
)
from omnitool.gradio.core import (
    SamplingOrchestrator,
    ToolCollection,
    get_available_agents,
)
from omnitool.gradio.services import AppState, FileHandler, validate_api_key
from omnitool.gradio.ui.gradio.components import (
    create_settings_panel,
    format_message_for_display,
    get_model_choices,
    get_provider_options_for_model,
    render_file_viewer,
)

logger = logging.getLogger(__name__)


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
        with gr.Blocks(title="OmniParser Refactored") as interface:
            # Header
            gr.Markdown("# OmniParser - Refactored")
            gr.Markdown("Vision-Language Model for Computer Interaction")
            
            # Initialize state
            state_var = gr.State()
            
            # Settings panel
            with gr.Accordion(label="Settings"):
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=get_model_choices(),
                        value="omniparser + gpt-4o",
                        label="Model",
                    )
                    provider_dropdown = gr.Dropdown(
                        choices=["openai"],
                        value="openai",
                        label="Provider",
                    )
                
                # Update provider options when model changes
                model_dropdown.change(
                    fn=self.on_model_change,
                    inputs=[model_dropdown],
                    outputs=[provider_dropdown],
                )
            
            # Chat interface
            with gr.Accordion(label="Chat"):
                chatbot = gr.Chatbot(
                    label="Conversation",
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
            
            # Status and progress
            with gr.Accordion(label="Execution"):
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3,
                )
                progress_bar = gr.Progress()
            
            # Wire up interactions
            submit_button.click(
                fn=self.on_submit,
                inputs=[
                    state_var,
                    message_input,
                    model_dropdown,
                    provider_dropdown,
                    chatbot,
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
    
    async def on_app_load(self) -> list:
        """Capture initial screenshot on app startup (non-blocking).
        
        Returns:
            Initial chatbot history with screenshot
        """
        try:
            # Capture screenshot
            screenshot_data = await asyncio.to_thread(
                self.windows_host_client.get_screenshot
            )
            screenshot_base64 = screenshot_data.get("screenshot_base64")
            
            if not screenshot_base64:
                logger.warning("Screenshot returned but no image data")
                return [{"role": "system", "content": "Initial screenshot: No image data available"}]
            
            # Optionally resize to reasonable max width (1024px)
            try:
                # Decode base64 to PIL Image
                img_data = base64.b64decode(screenshot_base64)
                img = Image.open(BytesIO(img_data))
                
                # Resize if width exceeds 1024px
                max_width = 1024
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_height = int(img.height * ratio)
                    img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Re-encode to base64
                    buffer = BytesIO()
                    img.save(buffer, format="PNG")
                    screenshot_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    logger.debug(f"Resized screenshot from {screenshot_data.get('width')}x{screenshot_data.get('height')} to {img.width}x{img.height}")
            except Exception as resize_error:
                logger.debug(f"Screenshot resize failed, using original: {resize_error}")
            
            # Format as HTML img tag with base64 data URI
            img_html = f'<img src="data:image/png;base64,{screenshot_base64}" style="max-width: 100%; border-radius: 8px; margin: 10px 0;">'
            
            # Return initial message history with context text
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
    ) -> Tuple[list, str, str, AppState]:
        """Handle submit button click.
        
        Args:
            state: App state
            message: User message
            model_name: Selected model
            provider: Selected provider
            chatbot_history: Chat history
            
        Returns:
            Updated chatbot, input, status, state
        """
        # Initialize state if needed
        if state is None or not isinstance(state, AppState):
            state = AppState(run_folder=Path(self.settings.run_folder))
        
        history = list(chatbot_history) if chatbot_history else []

        # Add user message
        state.chat.add_message("user", message)
        history.append({"role": "user", "content": message})
        
        # Validate API key
        is_valid, error_msg = validate_api_key(
            provider,
            azure_endpoint=self.settings.azure_endpoint if provider == "azure" else None
        )
        
        if not is_valid:
            history.append({"role": "assistant", "content": f"Error: {error_msg}"})
            return history, "", error_msg, state
        
        # Create orchestrator
        try:
            # Prepare orchestrator kwargs
            orchestrator_kwargs = {
                "model_name": model_name,
                "state": state,
                "tools_collection": self.tools,
                "omniparser_client": self.omniparser_client,
                "windows_host_client": self.windows_host_client,
                "max_steps": 20,
                "provider": provider,
            }
            
            # Add Azure endpoint if using Azure provider
            if provider == "azure":
                orchestrator_kwargs["azure_endpoint"] = self.settings.azure_endpoint
            
            self.orchestrator = SamplingOrchestrator(**orchestrator_kwargs)
            
            # Run sampling loop
            status = "Running..."
            for update in self.orchestrator.sampling_loop():
                if update.get('type') == 'complete':
                    status = f"Complete. Steps: {update.get('total_steps')}, Tokens: {update.get('total_tokens')}, Cost: {update.get('total_cost')}"
                    break
                elif update.get('type') == 'error':
                    status = f"Error: {update.get('message')}"
                    break
            
            # Update history with assistant response
            history.append({"role": "assistant", "content": status})
            return history, "", status, state
        
        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            history.append({"role": "assistant", "content": error_msg})
            return history, "", error_msg, state
    
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
