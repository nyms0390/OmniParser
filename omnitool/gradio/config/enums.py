"""
Shared enumerations used across the OmniParser application.
"""

import re
from enum import StrEnum

# Currency symbols, thousands separators, whitespace, and percent are stripped
# before float conversion. Intentionally conservative: "ORD-999" survives and
# raises ValueError rather than silently becoming -999.
_NUMERIC_STRIP_RE = re.compile(r"[$€£¥₹,\s%]")


def _strip_numeric(value: str) -> float:
    """Strip formatting characters from *value* and parse as float."""
    cleaned = _NUMERIC_STRIP_RE.sub("", value.strip())
    if not cleaned:
        raise ValueError(f"No numeric content in {value!r}")
    return float(cleaned)


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
    CONCAT = "concat"

    def apply(self, values: list[str]) -> str:
        """Apply this operation to a list of raw string values and return a string result.

        Raises:
            ValueError: If *values* is empty, or if SUM encounters a non-numeric string.
        """
        if not values:
            raise ValueError(f"{self!r}.apply() called with empty list")
        if self == AggregateOperation.SUM:
            numeric = [_strip_numeric(v) for v in values]
            return f"{sum(numeric):.10g}"
        if self == AggregateOperation.CONCAT:
            return "".join(values)
        raise NotImplementedError(self)

