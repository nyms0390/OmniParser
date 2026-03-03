"""
Agent factory for creating agents from model configuration.
"""

from pathlib import Path
from typing import Optional

from omnitool.gradio.clients import BaseLLMClient, get_llm_client
from omnitool.gradio.config import MODEL_CONFIG, get_model_config
from omnitool.gradio.app import AppState, get_api_key, AuthProvider

from .anthropic import AnthropicAgent
from .base import BaseAgent
from .vlm import VLMAgent


def create_agent(
    model_name: str,
    state: AppState,
    tools_collection,  # ToolCollection type
    save_folder: Path,
    provider_override: Optional[str] = None,
    provider: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
) -> BaseAgent:
    """Factory function to create appropriate agent for given model.
    
    Args:
        model_name: Display name from MODEL_CONFIG
        state: Application state
        tools_collection: Available tools
        save_folder: Output folder for agent
        provider_override: Override provider (for Anthropic variants like bedrock/vertex)
        provider: LLM provider to use (e.g., 'openai', 'azure', 'anthropic')
        azure_endpoint: Azure OpenAI endpoint (required if provider is 'azure')
        
    Returns:
        Initialized agent
        
    Raises:
        ValueError: If model not found or initialization fails
    """
    # Get model config
    config = get_model_config(model_name)
    agent_type = config.get('agent_type')
    llm_client_name = config.get('llm_client')
    
    # Use provider parameter if provided (from UI), otherwise fall back to override or config
    if provider is None:
        provider = provider_override or config.get('provider')
    
    # Handle provider as list (from config) - take first element or the selected one
    if isinstance(provider, list):
        provider = provider[0]  # Default to first available provider
    
    # Get API key from environment
    # Convert provider string to AuthProvider enum
    try:
        auth_provider = AuthProvider(provider)
        api_key = get_api_key(auth_provider)
    except ValueError:
        # Unknown provider, try as-is
        api_key = ""
    
    if not api_key and provider != 'azure':
        # Azure uses DefaultAzureCredential, not API key
        raise ValueError(
            f"API key not found in environment for provider: {provider}. "
            f"Set environment variable for: {model_name}"
        )
    
    # Create LLM client
    llm_client = _create_llm_client(
        llm_client_name,
        provider,
        model_name,
        config,
        api_key,
        azure_endpoint=azure_endpoint,
    )
    
    # Create and return appropriate agent
    if agent_type == "VLMAgent":
        return VLMAgent(
            model_name=model_name,
            llm_client=llm_client,
            state=state,
            tools_collection=tools_collection,
            save_folder=save_folder,
        )
    
    elif agent_type == "AnthropicAgent":
        return AnthropicAgent(
            model_name=model_name,
            llm_client=llm_client,
            state=state,
            tools_collection=tools_collection,
            save_folder=save_folder,
        )
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def _create_llm_client(
    llm_client_name: str,
    provider: str,
    model_name: str,
    config: dict,
    api_key: str,
    azure_endpoint: Optional[str] = None,
) -> BaseLLMClient:
    """Create LLM client for model.
    
    Args:
        llm_client_name: Client name ('openai', 'groq', 'anthropic')
        provider: Provider name
        model_name: Display model name
        config: Model configuration
        api_key: API key from environment
        azure_endpoint: Azure OpenAI endpoint (required if provider is 'azure')
        
    Returns:
        Initialized LLM client
    """
    internal_name = config.get('internal_name')
    base_url = config.get('provider_base_url')

    # Pull per-model generation defaults from the model config so they flow
    # into the client's self.kwargs and act as the second-priority fallback
    # (after per-call overrides, before hard-coded constants).
    generation_kwargs = {}
    if 'temperature' in config:
        generation_kwargs['temperature'] = config['temperature']
    if 'max_tokens' in config:
        generation_kwargs['max_tokens'] = config['max_tokens']

    # Create client based on provider/type
    if llm_client_name == 'openai' or provider in ['openai', 'dashscope', 'azure']:
        # For Azure, pass the endpoint
        kwargs = {
            'provider': provider,
            'model': internal_name,
            'api_key': api_key,
            **generation_kwargs,
        }
        if base_url:
            kwargs['base_url'] = base_url
        if provider == 'azure' and azure_endpoint:
            kwargs['azure_endpoint'] = azure_endpoint

        return get_llm_client(**kwargs)

    elif llm_client_name == 'groq':
        return get_llm_client(
            provider='groq',
            model=internal_name,
            api_key=api_key,
            **generation_kwargs,
        )

    elif llm_client_name == 'anthropic':
        # Check if we should use bedrock/vertex instead
        if provider == 'bedrock':
            return get_llm_client(
                provider='bedrock',
                model=internal_name,
                api_key=api_key,
                **generation_kwargs,
            )
        elif provider == 'vertex':
            return get_llm_client(
                provider='vertex',
                model=internal_name,
                api_key=api_key,
                **generation_kwargs,
            )
        else:
            return get_llm_client(
                provider='anthropic',
                model=internal_name,
                api_key=api_key,
                **generation_kwargs,
            )

    else:
        raise ValueError(f"Unknown LLM client: {llm_client_name}")


def get_available_agents() -> list[str]:
    """Get list of all available agent configurations from MODEL_CONFIG.
    
    Returns:
        List of available model display names
    """
    return list(MODEL_CONFIG.keys())


def get_agent_info(model_name: str) -> dict:
    """Get detailed information about an agent configuration.
    
    Args:
        model_name: Display name of model
        
    Returns:
        Configuration dictionary
    """
    return get_model_config(model_name)
