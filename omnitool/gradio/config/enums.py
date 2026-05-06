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


class FieldKind(StrEnum):
    """Shape of a TaskOutput value as captured from the screen.

    SCALAR — single value, optionally clipboard-corrected.
    ROW    — accumulating list, one entry per read_field item.
    """
    SCALAR = "scalar"
    ROW = "row"


class AggregateOperation(StrEnum):
    """Supported aggregation operations for the read_field tool."""
    SUM = "sum"
    CONCAT = "concat"
    DEDUP = "dedup"

    def apply(self, values: list[str]) -> str | list[str]:
        """Apply this operation to a list of raw string values.

        Returns a ``str`` for reducing operations (SUM, CONCAT) or a
        ``list[str]`` for list-returning operations (DEDUP).

        Raises:
            ValueError: If *values* is empty for reducing operations (SUM,
                CONCAT), or if SUM encounters a non-numeric string. DEDUP
                tolerates empty input.
        """
        if self == AggregateOperation.DEDUP:
            # dict.fromkeys preserves insertion order (Python 3.7+) while dropping duplicate keys
            return list(dict.fromkeys(values))
        if not values:
            raise ValueError(f"{self!r}.apply() called with empty list")
        if self == AggregateOperation.SUM:
            numeric = [_strip_numeric(v) for v in values]
            return f"{sum(numeric):.10g}"
        if self == AggregateOperation.CONCAT:
            return "".join(values)
        raise NotImplementedError(self)

