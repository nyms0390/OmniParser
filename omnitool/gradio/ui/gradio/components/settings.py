"""
Settings UI components.
"""

from typing import Callable, List

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
        List of provider options
    """
    config = MODEL_CONFIG.get(model_name, {})
    provider = config.get('provider')
    
    # For Anthropic models, list all possible providers
    if config.get('anthropic_providers'):
        return config['anthropic_providers']
    
    # For other models, return single provider
    return [provider] if provider else []


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
