"""
Configuration module for OmniParser Gradio refactored version.
"""

from omnitool.gradio.config.constants import (
    ALLOWED_FILE_EXTENSIONS,
    CONFIG_DIR_NAME,
    LLM_TIMEOUT_SECONDS,
    MAX_TOKENS_DEFAULT,
    MAX_UPLOAD_SIZE_MB,
    OMNIPARSER_TIMEOUT_SECONDS,
    OUTPUT_DIR,
    SCREENSHOT_RETENTION_COUNT,
    TEMPERATURE_DEFAULT,
    TYPING_DELAY_MS,
    WINDOWS_HOST_TIMEOUT_SECONDS,
)
from omnitool.gradio.config.models import (
    APIProvider,
    MODEL_CONFIG,
    get_all_model_names,
    get_model_config,
    is_orchestrated_model,
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
    "MAX_TOKENS_DEFAULT",
    "TEMPERATURE_DEFAULT",
    "LLM_TIMEOUT_SECONDS",
    "OMNIPARSER_TIMEOUT_SECONDS",
    "WINDOWS_HOST_TIMEOUT_SECONDS",
    "MAX_UPLOAD_SIZE_MB",
    "ALLOWED_FILE_EXTENSIONS",
    "CONFIG_DIR_NAME",
    # Models
    "APIProvider",
    "MODEL_CONFIG",
    "get_model_config",
    "get_all_model_names",
    "is_orchestrated_model",
    # Settings
    "Settings",
    "create_argument_parser",
    "load_yaml_config",
    "load_settings",
    "get_settings",
]
