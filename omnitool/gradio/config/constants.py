"""
Application-wide constants for OmniParser Gradio refactored version.
"""

# UI Constants
TYPING_DELAY_MS = 50  # Delay for typing animation in UI

# Screenshot Management
SCREENSHOT_RETENTION_COUNT = 2  # Number of screenshots to keep in memory
OUTPUT_DIR = "./tmp/outputs"  # Default output directory

# Timeout Constants
LLM_TIMEOUT_SECONDS = 300
OMNIPARSER_TIMEOUT_SECONDS = 60
WINDOWS_HOST_TIMEOUT_SECONDS = 30

# File Upload Constants
MAX_UPLOAD_SIZE_MB = 500
ALLOWED_FILE_EXTENSIONS = ['.txt', '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']

# Config Directory
CONFIG_DIR_NAME = ".omniparser_config"
