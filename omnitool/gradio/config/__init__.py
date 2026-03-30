"""
Configuration module for OmniParser Gradio refactored version.
"""

from omnitool.gradio.config.constants import (
    ALLOWED_FILE_EXTENSIONS,
    CONFIG_DIR_NAME,
    SCREENSHOT_MAX_WIDTH,
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
    PROVIDER_CONFIG,
    LLM_MODELS,
    get_llm_config,
    get_provider_config,
    get_all_model_names,
    get_supported_providers,
    get_pricing,
    OCR_CONFIG,
    get_ocr_config,
    get_all_ocr_backends,
)
from omnitool.gradio.config.logging_config import setup_logging
from omnitool.gradio.config.prompts import (
    PLATFORM_PROMPTS,
    PlatformPrompt,
    build_anthropic_system_prompt,
    build_react_system_prompt,
    build_vlm_tool_system_prompt,
    VLM_TOOL_SYSTEM_PROMPT,
    CHECKLIST_GEN_PROMPT,
    CHECKLIST_GEN_SYSTEM_PROMPT,
    COMPACTION_PROMPT,
    REACT_SYSTEM_PROMPT,
    REFLECT_PROMPT,
    REFLECT_SYSTEM_PROMPT,
)
from omnitool.gradio.config.settings import (
    Settings,
    create_argument_parser,
    get_settings,
    load_settings,
    load_yaml_config,
)
from omnitool.gradio.config.task_template import TaskInput, TaskProcedure, TaskTemplate, load_task_template

__all__ = [
    # Constants
    "TYPING_DELAY_MS",
    "SCREENSHOT_RETENTION_COUNT",
    "SCREENSHOT_MAX_WIDTH",
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
    "PROVIDER_CONFIG",
    "LLM_MODELS",
    "get_llm_config",
    "get_provider_config",
    "get_all_model_names",
    "get_supported_providers",
    "get_pricing",
    # Prompts
    "PLATFORM_PROMPTS",
    "PlatformPrompt",
    "build_anthropic_system_prompt",
    "build_react_system_prompt",
    "build_vlm_tool_system_prompt",
    "VLM_TOOL_SYSTEM_PROMPT",
    "REACT_SYSTEM_PROMPT",
    "COMPACTION_PROMPT",
    "CHECKLIST_GEN_PROMPT",
    "CHECKLIST_GEN_SYSTEM_PROMPT",
    "REFLECT_PROMPT",
    "REFLECT_SYSTEM_PROMPT",
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
    # Task template
    "TaskInput",
    "TaskProcedure",
    "TaskTemplate",
    "load_task_template",
]
