"""
Settings UI components.
"""

from typing import List

from omnitool.gradio.config import get_all_model_names, get_supported_providers
from omnitool.gradio.core.agents.preprocessing import PreprocessingMode


AGENT_CHOICES = ["VLMAgent", "AnthropicAgent", "ReActAgent"]
DEFAULT_AGENT = "VLMAgent"
DEFAULT_MODEL = "gpt-4o"

GROUNDING_CHOICES = ["omniparser", "gta1"]
DEFAULT_GROUNDING = "omniparser"

PREPROCESSING_CHOICES = [mode.value for mode in PreprocessingMode]
DEFAULT_PREPROCESSING = PreprocessingMode.RAW.value


def get_agent_choices() -> List[str]:
    """Return available agent types for the UI dropdown."""
    return AGENT_CHOICES


def get_model_choices() -> List[str]:
    """Return all registered model IDs for the UI dropdown."""
    return get_all_model_names()


def get_provider_options_for_model(model_name: str) -> List[str]:
    """Return provider strings for *model_name* (for the provider dropdown).

    Args:
        model_name: Model ID (key in LLM_MODELS)

    Returns:
        List of provider name strings, or empty list if model not found.
    """
    try:
        return get_supported_providers(model_name)
    except ValueError:
        return []


__all__ = [
    "get_agent_choices",
    "get_model_choices",
    "get_provider_options_for_model",
    "AGENT_CHOICES",
    "DEFAULT_AGENT",
    "DEFAULT_MODEL",
    "GROUNDING_CHOICES",
    "DEFAULT_GROUNDING",
    "PREPROCESSING_CHOICES",
    "DEFAULT_PREPROCESSING",
]
