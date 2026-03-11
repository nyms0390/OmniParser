"""
Agent factory — creates the appropriate BaseAgent subclass from model and provider config.
"""

from pathlib import Path
from typing import Dict, Optional

from omnitool.gradio.clients import BaseLLMClient, get_llm_client
from omnitool.gradio.clients.external.gta1 import GTA1Client
from omnitool.gradio.clients.external.omniparser import OmniParserClient
from omnitool.gradio.config import AgentMode, get_llm_config, get_provider_config
from omnitool.gradio.app import AppState, get_api_key, AuthProvider

from .anthropic import AnthropicAgent
from .base import BaseAgent
from .gta import GTAAgent
from .omniagent import OmniAgent


def create_agent(
    agent_type: str,
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
    extract_fields: Optional[Dict[str, str]] = None,
    azure_endpoint: Optional[str] = None,
    gta1_url: str = "http://localhost:8002",
) -> BaseAgent:
    """Factory function — returns the right BaseAgent subclass.

    Args:
        agent_type: Agent class name ("OmniAgent", "GTAAgent", "AnthropicAgent").
        model_name: LLM model ID from LLM_MODELS (e.g. "gpt-4o").
        state: Application runtime state.
        tools_collection: Available tools.
        save_folder: Output folder for the agent.
        omniparser_client: Required for OmniAgent / AnthropicAgent.
        mode: Agent operating mode.
        platform: Target OS (affects system prompt).
        max_steps: Maximum loop iterations.
        context_n: Chat history window size.
        provider: LLM provider (e.g. "openai", "azure"). Defaults to first
            supported provider of the model.
        azure_endpoint: Azure OpenAI endpoint URL (required when provider="azure").
        gta1_url: GTA1 server URL (required for GTAAgent).

    Returns:
        Initialised BaseAgent subclass.

    Raises:
        ValueError: If model/agent not found or required clients are missing.
    """
    llm_cfg = get_llm_config(model_name)

    # Resolve provider — default to first supported provider for this model
    if provider is None:
        provider = str(llm_cfg["supported_providers"][0])

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
        provider,
        llm_cfg,
        api_key,
        azure_endpoint=azure_endpoint if provider == "azure" else None,
    )

    common = dict(
        model_name=model_name,
        provider=provider,
        llm_client=llm_client,
        state=state,
        tools_collection=tools_collection,
        save_folder=save_folder,
        mode=mode,
        platform=platform,
        max_steps=max_steps,
        context_n=context_n,
        action_delay=action_delay,
        extract_fields=extract_fields,
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
        return GTAAgent(
            gta1_client=GTA1Client(base_url=gta1_url),
            omniparser_client=omniparser_client,
            **common,
        )

    else:
        raise ValueError(f"Unknown agent_type: {agent_type!r}")


def _create_llm_client(
    provider: str,
    llm_cfg: dict,
    api_key: str,
    azure_endpoint: Optional[str] = None,
) -> BaseLLMClient:
    """Instantiate the correct LLM client from provider + model config.

    Provider → SDK mapping (deterministic, not stored in config):
        openai, dashscope          → OpenAI-compatible SDK
        azure                      → Azure OpenAI SDK (endpoint from settings)
        anthropic, bedrock, vertex → Anthropic SDK
        groq                       → Groq SDK
    """
    internal_name = llm_cfg["internal_name"]
    api_mode = llm_cfg.get("api_mode", "chat")
    generation_kwargs = {
        k: llm_cfg[k] for k in ("temperature", "max_tokens") if k in llm_cfg
    }

    kwargs = dict(
        provider=provider,
        model=internal_name,
        api_key=api_key,
        api_mode=api_mode,
        **generation_kwargs,
    )

    if provider in ("openai", "dashscope"):
        kwargs["base_url"] = get_provider_config(provider)
    elif provider == "azure" and azure_endpoint:
        kwargs["azure_endpoint"] = azure_endpoint

    return get_llm_client(**kwargs)
