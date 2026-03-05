"""
LLM clients module initialization and factory.
"""

from typing import Optional

from omnitool.gradio.clients.llm.base import BaseLLMClient
from omnitool.gradio.config import APIProvider, get_model_config

from .anthropic import AnthropicClient
from .azure import AzureOpenAIClient
from .groq import GroqClient
from .openai import OpenAIClient


def get_llm_client(
    provider: str,
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    **kwargs
) -> BaseLLMClient:
    """Factory function to create LLM client for given provider.
    
    Args:
        provider: Provider name ('openai', 'groq', 'anthropic', 'bedrock', 'vertex')
        model: Model name
        api_key: API key for authentication
        base_url: Optional base URL (for OpenAI-compatible APIs)
        **kwargs: Additional provider-specific arguments
        
    Returns:
        Initialized LLM client
        
    Raises:
        ValueError: If provider is unknown
    """
    provider_lower = provider.lower()
    
    if provider_lower in [APIProvider.OPENAI, "openai"]:
        return OpenAIClient(
            api_key=api_key,
            model=model,
            base_url=base_url or "https://api.openai.com/v1",
            **kwargs
        )
    
    elif provider_lower in [APIProvider.DASHSCOPE, "dashscope"]:
        # DashScope uses OpenAI-compatible API
        return OpenAIClient(
            api_key=api_key,
            model=model,
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            **kwargs
        )
    
    elif provider_lower in [APIProvider.GROQ, "groq"]:
        return GroqClient(
            api_key=api_key,
            model=model,
            **kwargs
        )
    
    elif provider_lower in [APIProvider.ANTHROPIC, "anthropic"]:
        return AnthropicClient(
            api_key=api_key,
            model=model,
            provider="anthropic",
            **kwargs
        )
    
    elif provider_lower in [APIProvider.BEDROCK, "bedrock"]:
        return AnthropicClient(
            api_key=api_key,
            model=model,
            provider="bedrock",
            region=kwargs.get("region", "us-east-1"),
            **kwargs
        )
    
    elif provider_lower in [APIProvider.VERTEX, "vertex"]:
        return AnthropicClient(
            api_key=api_key,
            model=model,
            provider="vertex",
            region=kwargs.get("region", "us-central1"),
            **kwargs
        )
    
    elif provider_lower in [APIProvider.AZURE, "azure"]:
        azure_endpoint = kwargs.pop("azure_endpoint", None)
        return AzureOpenAIClient(
            api_key=api_key,
            model=model,
            azure_endpoint=azure_endpoint,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def get_llm_client_for_model(
    model_name: str,
    api_key: str,
    **kwargs
) -> BaseLLMClient:
    """Create LLM client for a model from MODEL_CONFIG.
    
    Args:
        model_name: Display name of model from MODEL_CONFIG
        api_key: API key for the provider
        **kwargs: Additional arguments
        
    Returns:
        Initialized LLM client
        
    Raises:
        ValueError: If model not found in MODEL_CONFIG
    """
    config = get_model_config(model_name)
    
    return get_llm_client(
        provider=config["provider"],
        model=config["internal_name"],
        api_key=api_key,
        base_url=config.get("provider_base_url"),
        **kwargs
    )


__all__ = [
    "BaseLLMClient",
    "OpenAIClient",
    "GroqClient",
    "AnthropicClient",
    "AzureOpenAIClient",
    "get_llm_client",
    "get_llm_client_for_model",
]
