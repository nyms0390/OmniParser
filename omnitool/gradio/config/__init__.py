"""
Configuration module for OmniParser Gradio refactored version.
"""

from omnitool.gradio.config.constants import (
    ALLOWED_FILE_EXTENSIONS,
    CONFIG_DIR_NAME,
    LLM_TIMEOUT_SECONDS,
    MAX_UPLOAD_SIZE_MB,
    OMNIPARSER_TIMEOUT_SECONDS,
    OUTPUT_DIR,
    SCREENSHOT_RETENTION_COUNT,
    TYPING_DELAY_MS,
    WINDOWS_HOST_TIMEOUT_SECONDS,
)
from omnitool.gradio.config.enums import AgentMode, APIProvider
from omnitool.gradio.config.models import (
    MODEL_CONFIG,
    OCR_CONFIG,
    get_all_model_names,
    get_model_config,
    get_ocr_config,
    get_all_ocr_backends,
)
from omnitool.gradio.config.logging_config import setup_logging
from omnitool.gradio.config.prompts import (
    PLATFORM_PROMPTS,
    PlatformPrompt,
    build_anthropic_system_prompt,
    build_gta1_system_prompt,
    build_vlm_system_prompt,
    GTA1_SYSTEM_PROMPT,
    PLAN_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    REFLECT_PROMPT,
    TASK_PARSE_PROMPT,
)
from omnitool.gradio.config.settings import (
    Settings,
    create_argument_parser,
    get_settings,
    load_settings,
    load_yaml_config,
)

__all__ = [
    # Constants
    "TYPING_DELAY_MS",
    "SCREENSHOT_RETENTION_COUNT",
    "OUTPUT_DIR",
    "LLM_TIMEOUT_SECONDS",
    "OMNIPARSER_TIMEOUT_SECONDS",
    "WINDOWS_HOST_TIMEOUT_SECONDS",
    "MAX_UPLOAD_SIZE_MB",
    "ALLOWED_FILE_EXTENSIONS",
    "CONFIG_DIR_NAME",
    # Enums
    "AgentMode",
    "APIProvider",
    # Models
    "MODEL_CONFIG",
    "get_model_config",
    "get_all_model_names",
    # Prompts
    "PLATFORM_PROMPTS",
    "PlatformPrompt",
    "build_vlm_system_prompt",
    "build_anthropic_system_prompt",
    "build_gta1_system_prompt",
    "GTA1_SYSTEM_PROMPT",
    "PLAN_PROMPT",
    "PLANNER_SYSTEM_PROMPT",
    "REFLECT_PROMPT",
    "TASK_PARSE_PROMPT",
    # OCR
    "OCR_CONFIG",
    "get_ocr_config",
    "get_all_ocr_backends",
    # Settings
    "Settings",
    "create_argument_parser",
    "load_yaml_config",
    "load_settings",
    "get_settings",
    # Logging
    "setup_logging",
]
