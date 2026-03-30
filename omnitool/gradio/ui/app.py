"""
Main Gradio application entry point.

UI layout and wiring live here; event handlers are in callbacks.py.

To run:
    python -m omnitool.gradio.ui.app

Environment variables (or set via --config-file):
    OPENAI_API_KEY: OpenAI API key
    ANTHROPIC_API_KEY: Anthropic API key
    GROQ_API_KEY: Groq API key
    DASHSCOPE_API_KEY: DashScope API key
    AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL
    OMNIPARSER_URL: OmniParser server URL (default: http://localhost:8000)
    WINDOWS_HOST_URL: Windows host URL (default: http://localhost:8006)
"""

import logging

import gradio as gr

from omnitool.gradio.clients.external import (
    GTA1Client, OmniParserClient, PaddleOCRClient, WindowsHostClient, ServiceValidator,
)
from omnitool.gradio.config import (
    AgentMode,
    create_argument_parser,
    get_settings,
    setup_logging,
)
from omnitool.gradio.core import ToolCollection
from omnitool.gradio.ui.callbacks import GradioCallbacks
from omnitool.gradio.ui.components import (
    get_agent_choices,
    get_model_choices,
    get_provider_options_for_model,
    DEFAULT_AGENT,
    DEFAULT_MODEL,
    GROUNDING_CHOICES,
    DEFAULT_GROUNDING,
    PREPROCESSING_CHOICES,
    DEFAULT_PREPROCESSING,
)

logger = logging.getLogger(__name__)


class GradioApp(GradioCallbacks):
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
        self.gta1_client = GTA1Client(settings.gta1_url)
        self.tools = ToolCollection(windows_host_client=self.windows_host_client)
        self.orchestrator = None

        # Validate all services on startup
        validator = ServiceValidator()
        validator.register("OmniParser", self.omniparser_client)
        validator.register("Windows Host", self.windows_host_client)
        validator.register("PaddleOCR", self.paddleocr_client)
        validator.register("GTA1", self.gta1_client)
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
                        visible=(DEFAULT_AGENT in ("VLMAgent", "ReActAgent")),
                    )
                    preprocessing_dropdown = gr.Dropdown(
                        choices=PREPROCESSING_CHOICES,
                        value=DEFAULT_PREPROCESSING,
                        label="Image Preprocessing",
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
                        value=50,
                        label="Max steps",
                    )

                with gr.Row():
                    mode_dropdown = gr.Dropdown(
                        choices=[
                            ("Interactive", AgentMode.INTERACTIVE.value),
                            ("Orchestrated", AgentMode.ORCHESTRATED.value),
                            ("Task", AgentMode.TASK.value),
                        ],
                        value=AgentMode.TASK.value,
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
                        visible=True,
                    )
                    procedure_dropdown = gr.Dropdown(
                        label="Select Procedure",
                        choices=[],
                        value=None,
                        visible=False,
                        interactive=True,
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
                    preprocessing_dropdown,
                    model_dropdown,
                    provider_dropdown,
                    chatbot,
                    mode_dropdown,
                    platform_dropdown,
                    context_n_slider,
                    max_steps_slider,
                    extract_fields_input,
                    yaml_template_state,
                    procedure_dropdown,
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
                outputs=[yaml_template_state, procedure_dropdown],
            )

            file_upload.change(
                fn=self.on_file_upload,
                inputs=[state_var, file_upload],
                outputs=[file_viewer, state_var],
            )

        return interface


def main():
    """Main entry point."""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup logging (BEFORE loading settings)
    log_file = args.log_file or "omniparser_app.log"
    setup_logging("omniparser_app", level=args.log_level, log_file=log_file)

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
