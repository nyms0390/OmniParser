"""
Agent factory — creates the appropriate BaseAgent subclass from model and provider config.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

from omnitool.gradio.clients import BaseLLMClient, get_llm_client
from omnitool.gradio.clients.external.gta1 import GTA1Client
from omnitool.gradio.clients.external.omniparser import OmniParserClient
from omnitool.gradio.config import AgentMode, get_llm_config, get_provider_config
from omnitool.gradio.services import AppState, get_api_key, AuthProvider

from .anthropic_agent import AnthropicAgent
from .base import BaseAgent
from .grounding import GTA1Grounding, GroundingStrategy, OmniParserGrounding
from .react_agent import ReActAgent
from .vlm_agent import VLMAgent


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
    gta1_client: Optional[GTA1Client] = None,
    grounding: str = "omniparser",
) -> BaseAgent:
    """Factory function — returns the right BaseAgent subclass.

    Args:
        agent_type: Agent class name ("VLMAgent", "AnthropicAgent", "ReActAgent").
        model_name: LLM model ID from LLM_MODELS (e.g. "gpt-4o").
        state: Application runtime state.
        tools_collection: Available tools.
        save_folder: Output folder for the agent.
        omniparser_client: Required for VLMAgent / AnthropicAgent / ReActAgent
            with omniparser grounding.
        mode: Agent operating mode.
        platform: Target OS (affects system prompt).
        max_steps: Maximum loop iterations.
        context_n: Chat history window size (unused by ReActAgent).
        provider: LLM provider (e.g. "openai", "azure"). Defaults to first
            supported provider of the model.
        azure_endpoint: Azure OpenAI endpoint URL (required when provider="azure").
        gta1_client: Pre-constructed GTA1Client (required for VLMAgent / ReActAgent
            with gta1 grounding; also enables clipboard correction in all agents).
        grounding: Grounding strategy for VLMAgent and ReActAgent — "omniparser"
            or "gta1".

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

    # Only construct GTA1Client when the agent/grounding actually needs it.
    # AnthropicAgent doesn't use GTA1 grounding and doesn't need the client.
    if gta1_client is None and agent_type != "AnthropicAgent":
        gta1_client = GTA1Client()
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
        gta1_client=gta1_client,
    )

    if agent_type == "VLMAgent":
        strategy, grounding_kwargs = _resolve_grounding(
            grounding, omniparser_client, gta1_client, agent_type
        )
        return VLMAgent(grounding_strategy=strategy, **grounding_kwargs, **common)

    elif agent_type == "AnthropicAgent":
        if omniparser_client is None:
            raise ValueError("AnthropicAgent requires omniparser_client")
        return AnthropicAgent(omniparser_client=omniparser_client, **common)

    elif agent_type == "ReActAgent":
        strategy, grounding_kwargs = _resolve_grounding(
            grounding, omniparser_client, gta1_client, agent_type
        )
        return ReActAgent(grounding_strategy=strategy, **grounding_kwargs, **common)

    else:
        raise ValueError(f"Unknown agent_type: {agent_type!r}")


def _resolve_grounding(
    grounding: str,
    omniparser_client: Optional[OmniParserClient],
    gta1_client: GTA1Client,
    agent_type: str,
) -> Tuple[GroundingStrategy, Dict]:
    """Return (GroundingStrategy, extra_kwargs) for the requested grounding mode."""
    if grounding == "gta1":
        return GTA1Grounding(gta1_client), {}
    else:
        if omniparser_client is None:
            raise ValueError(
                f"{agent_type} with omniparser grounding requires omniparser_client"
            )
        return OmniParserGrounding(omniparser_client), {"omniparser_client": omniparser_client}


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
