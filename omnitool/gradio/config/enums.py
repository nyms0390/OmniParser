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
    DASHSCOPE = "dashscope"
    AZURE = "azure"


class AgentMode(StrEnum):
    """Agent execution modes."""
    INTERACTIVE = "interactive"
    ORCHESTRATED = "orchestrated"
    TASK = "task"


class AggregateOperation(StrEnum):
    """Supported aggregation operations for the read_field tool."""
    SUM = "sum"

    def apply(self, values: list) -> float:
        """Apply this operation to a list of floats."""
        if self == AggregateOperation.SUM:
            return sum(values)
        raise NotImplementedError(self)


class Sender(StrEnum):
    """Message sender types in chat."""
    USER = "user"
    BOT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
