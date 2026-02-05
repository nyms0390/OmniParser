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
    OMNIPARSER_URL: OmniParser server URL (default: http://localhost:8000)
    WINDOWS_HOST_URL: Windows host URL (default: http://localhost:8006)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

from omnitool.gradio.clients import OmniParserClient
from omnitool.gradio.config import (
    APIProvider,
    create_argument_parser,
    get_all_model_names,
    get_model_config,
    get_settings,
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
        self.tools = ToolCollection()  # Initialize with available tools
        self.orchestrator = None
    
    def build_interface(self):
        """Build Gradio interface."""
        with gr.Blocks(title="OmniParser Refactored") as interface:
            # Header
            gr.Markdown("# OmniParser - Refactored")
            gr.Markdown("Vision-Language Model for Computer Interaction")
            
            # Initialize state
            state_var = gr.State()
            
            # Settings panel
            with gr.Group(label="Settings"):
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
            with gr.Group(label="Chat"):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=400,
                )
                
                with gr.Row():
                    message_input = gr.Textbox(
                        placeholder="Enter your request...",
                        show_label=False,
                        scale=4,
                    )
                    submit_button = gr.Button("Send", scale=1)
            
            # File upload and viewer
            with gr.Group(label="Files"):
                file_upload = gr.File(
                    label="Upload Files",
                    file_count="multiple",
                    type="filepath",
                )
                file_viewer = gr.HTML(label="File Viewer")
            
            # Status and progress
            with gr.Group(label="Execution"):
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3,
                )
                progress_bar = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    label="Progress",
                    interactive=False,
                )
            
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
                    progress_bar,
                    state_var,
                ],
            )
            
            file_upload.change(
                fn=self.on_file_upload,
                inputs=[state_var, file_upload],
                outputs=[file_viewer, state_var],
            )
        
        return interface
    
    def on_model_change(self, model_name: str) -> Tuple:
        """Handle model selection change.
        
        Args:
            model_name: Selected model name
            
        Returns:
            Updated provider choices
        """
        providers = get_provider_options_for_model(model_name)
        return gr.Dropdown.update(
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
    ) -> Tuple:
        """Handle submit button click.
        
        Args:
            state: App state
            message: User message
            model_name: Selected model
            provider: Selected provider
            chatbot_history: Chat history
            
        Returns:
            Updated chatbot, input, status, progress, state
        """
        # Initialize state if needed
        if state is None or not isinstance(state, AppState):
            state = AppState(run_folder=Path(self.settings.run_folder))
        
        # Add user message
        state.chat.add_message("user", message)
        updated_history = chatbot_history + [(message, None)]
        
        # Validate API key
        config = get_model_config(model_name)
        api_provider = config.get('provider')
        is_valid, error_msg = validate_api_key(api_provider)
        
        if not is_valid:
            updated_history.append((None, f"Error: {error_msg}"))
            return updated_history, "", error_msg, 0, state
        
        # Create orchestrator
        try:
            save_folder = state.session.run_folder / "execution"
            self.orchestrator = SamplingOrchestrator(
                model_name=model_name,
                state=state,
                tools_collection=self.tools,
                omniparser_client=self.omniparser_client,
                max_steps=20,
            )
            
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
            updated_history.append((None, "Execution complete"))
            
            return updated_history, "", status, 100, state
        
        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            updated_history.append((None, error_msg))
            return updated_history, "", error_msg, 0, state
    
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
    
    # Load settings
    settings = get_settings(args)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
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
