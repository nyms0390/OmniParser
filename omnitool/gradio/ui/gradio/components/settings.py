"""
Settings UI components.
"""

from typing import List

from omnitool.gradio.config import MODEL_CONFIG, get_all_model_names


def get_model_choices() -> List[str]:
    """Get list of available model choices.
    
    Returns:
        List of model display names
    """
    return get_all_model_names()


def get_provider_options_for_model(model_name: str) -> List[str]:
    """Get available provider options for a model.
    
    Args:
        model_name: Model display name
        
    Returns:
        List of provider options (provider names as strings)
    """
    config = MODEL_CONFIG.get(model_name, {})
    providers = config.get('provider')
    
    # Provider field is now a list of APIProvider enums
    if isinstance(providers, list):
        # Convert APIProvider enums to strings
        return [str(p) for p in providers]
    
    # Fallback for single provider (shouldn't happen with new schema)
    return [str(providers)] if providers else []


def create_settings_panel() -> dict:
    """Create settings panel configuration.
    
    Returns:
        Dictionary with settings panel configuration
    """
    return {
        "model_choices": get_model_choices(),
        "default_model": "omniparser + gpt-4o",
    }


__all__ = [
    "get_model_choices",
    "get_provider_options_for_model",
    "create_settings_panel",
]
