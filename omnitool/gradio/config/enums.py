"""
Shared enumerations used across the OmniParser application.
"""

from enum import StrEnum


class APIProvider(StrEnum):
    """Supported API providers for LLM integration."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    GROQ = "groq"
    QWEN = "qwen"
    DEEPSEEK = "deepseek"


class Sender(StrEnum):
    """Message sender types in chat."""
    USER = "user"
    BOT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
