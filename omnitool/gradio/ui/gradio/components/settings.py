"""
Settings UI components.
"""

from typing import List

from omnitool.gradio.config import get_all_model_names, get_supported_providers


AGENT_CHOICES = ["OmniAgent", "GTAAgent", "AnthropicAgent", "ReActAgent"]
DEFAULT_AGENT = "OmniAgent"
DEFAULT_MODEL = "gpt-4o"


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


def create_settings_panel() -> dict:
    """Return default settings panel configuration."""
    return {
        "agent_choices": get_agent_choices(),
        "default_agent": DEFAULT_AGENT,
        "model_choices": get_model_choices(),
        "default_model": DEFAULT_MODEL,
    }


__all__ = [
    "get_agent_choices",
    "get_model_choices",
    "get_provider_options_for_model",
    "create_settings_panel",
    "AGENT_CHOICES",
    "DEFAULT_AGENT",
    "DEFAULT_MODEL",
]
