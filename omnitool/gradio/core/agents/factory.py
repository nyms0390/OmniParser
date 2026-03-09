"""
Agent factory — creates the appropriate BaseAgent subclass from model config.
"""

import os
from pathlib import Path
from typing import Dict, Optional

from omnitool.gradio.clients import BaseLLMClient, get_llm_client
from omnitool.gradio.clients.external.gta1 import GTA1Client
from omnitool.gradio.clients.external.omniparser import OmniParserClient
from omnitool.gradio.config import MODEL_CONFIG, AgentMode, get_model_config
from omnitool.gradio.app import AppState, get_api_key, AuthProvider

from .anthropic import AnthropicAgent
from .base import BaseAgent
from .gta import GTAAgent
from .omniagent import OmniAgent


def create_agent(
    model_name: str,
    state: AppState,
    tools_collection,
    save_folder: Path,
    omniparser_client: Optional[OmniParserClient] = None,
    mode: AgentMode = AgentMode.INTERACTIVE,
    platform: str = "windows",
    max_steps: int = 20,
    context_n: int = 15,
    action_delay: float = 1.5,
    provider: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    gta1_url: Optional[str] = None,
    extract_fields: Optional[Dict[str, str]] = None,
    **kwargs,
) -> BaseAgent:
    """Factory function — returns the right BaseAgent subclass for *model_name*.

    Args:
        model_name: Display name from MODEL_CONFIG.
        state: Application state.
        tools_collection: Available tools.
        save_folder: Output folder for the agent.
        omniparser_client: Required for OmniAgent / AnthropicAgent.
        mode: Agent operating mode.
        platform: Target OS (affects system prompt).
        max_steps: Maximum loop iterations.
        context_n: Chat history window size.
        provider: LLM provider override.
        azure_endpoint: Azure OpenAI endpoint (provider="azure" only).
        gta1_url: GTA1 server URL override (GTAAgent only); falls back to
            ``GTA1_URL`` env var then ``http://localhost:8002``.

    Returns:
        Initialised BaseAgent subclass.

    Raises:
        ValueError: If model not found or required clients are missing.
    """
    config = get_model_config(model_name)
    agent_type = config.get("agent_type")

    # Resolve provider
    if provider is None:
        provider = config.get("provider")
    if isinstance(provider, list):
        provider = provider[0]

    # Get API key
    try:
        auth_provider = AuthProvider(provider)
        api_key = get_api_key(auth_provider)
    except ValueError:
        api_key = ""

    if not api_key and provider != "azure":
        raise ValueError(
            f"API key not found for provider '{provider}' (model: {model_name}). "
            "Set the appropriate environment variable."
        )

    llm_client = _create_llm_client(
        config.get("llm_client"),
        provider,
        config,
        api_key,
        azure_endpoint=azure_endpoint,
    )

    common = dict(
        model_name=model_name,
        llm_client=llm_client,
        state=state,
        tools_collection=tools_collection,
        save_folder=save_folder,
        mode=mode,
        platform=platform,
        max_steps=max_steps,
        context_n=context_n,
        action_delay=action_delay,
        extract_fields=extract_fields or None,
    )

    if agent_type == "OmniAgent":
        if omniparser_client is None:
            raise ValueError("OmniAgent requires omniparser_client")
        return OmniAgent(omniparser_client=omniparser_client, **common)

    elif agent_type == "AnthropicAgent":
        if omniparser_client is None:
            raise ValueError("AnthropicAgent requires omniparser_client")
        return AnthropicAgent(omniparser_client=omniparser_client, **common)

    elif agent_type == "GTAAgent":
        resolved_gta1_url = (
            gta1_url
            or config.get("gta1_url")
            or os.environ.get("GTA1_URL", "http://localhost:8002")
        )
        return GTAAgent(
            gta1_client=GTA1Client(base_url=resolved_gta1_url),
            omniparser_client=omniparser_client,
            **common,
        )

    else:
        raise ValueError(f"Unknown agent_type '{agent_type}' for model '{model_name}'")


def _create_llm_client(
    llm_client_name: str,
    provider: str,
    config: dict,
    api_key: str,
    azure_endpoint: Optional[str] = None,
) -> BaseLLMClient:
    """Create LLM client from config."""
    internal_name = config.get("internal_name")
    base_url = config.get("provider_base_url")

    generation_kwargs = {}
    if "temperature" in config:
        generation_kwargs["temperature"] = config["temperature"]
    if "max_tokens" in config:
        generation_kwargs["max_tokens"] = config["max_tokens"]

    if llm_client_name == "openai" or provider in ("openai", "dashscope", "azure"):
        kwargs = {
            "provider": provider,
            "model": internal_name,
            "api_key": api_key,
            **generation_kwargs,
        }
        if base_url:
            kwargs["base_url"] = base_url
        if provider == "azure" and azure_endpoint:
            kwargs["azure_endpoint"] = azure_endpoint
        return get_llm_client(**kwargs)

    elif llm_client_name == "groq":
        return get_llm_client(
            provider="groq",
            model=internal_name,
            api_key=api_key,
            **generation_kwargs,
        )

    elif llm_client_name == "anthropic":
        actual_provider = provider if provider in ("bedrock", "vertex") else "anthropic"
        return get_llm_client(
            provider=actual_provider,
            model=internal_name,
            api_key=api_key,
            **generation_kwargs,
        )

    else:
        raise ValueError(f"Unknown LLM client: {llm_client_name}")


def get_available_agents() -> list[str]:
    return list(MODEL_CONFIG.keys())


def get_agent_info(model_name: str) -> dict:
    return get_model_config(model_name)
