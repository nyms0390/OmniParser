"""
Agent factory — constructs ReActAgent from model and provider config.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from omnitool.gradio.clients import BaseLLMClient, get_llm_client
from omnitool.gradio.clients.external.gta1 import GTA1Client
from omnitool.gradio.clients.external.omniparser import OmniParserClient
from omnitool.gradio.clients.external.paddleocr import PaddleOCRClient
from omnitool.gradio.config import AgentMode, TaskExecution, get_llm_config, get_provider_config
from omnitool.gradio.services import AppState, get_api_key, AuthProvider

from omnitool.gradio.core.agents.preprocessing import PreprocessingMode
from omnitool.gradio.core.agents.base import BaseAgent
from omnitool.gradio.core.agents.grounding import GTA1Grounding, GroundingStrategy, OmniParserGrounding
from omnitool.gradio.core.agents.react_agent import ReActAgent


def create_agent(
    model_name: str,
    state: AppState,
    tools_collection,
    save_folder: Path,
    omniparser_client: Optional[OmniParserClient] = None,
    mode: AgentMode = AgentMode.INTERACTIVE,
    platform: str = "windows",
    max_steps: int = 20,
    action_delay: float = 1.5,
    provider: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    gta1_client: Optional[GTA1Client] = None,
    paddleocr_client: Optional[PaddleOCRClient] = None,
    grounding: str = "gta1",
    preprocessing_mode: str = "raw",
    task_execution: Optional[TaskExecution] = None,
) -> BaseAgent:
    """Factory function — returns a ReActAgent configured for the requested grounding.

    Args:
        model_name: LLM model ID from LLM_MODELS (e.g. "gpt-4o").
        state: Application runtime state.
        tools_collection: Available tools.
        save_folder: Output folder for the agent run.
        omniparser_client: Required when grounding="omniparser".
        mode: Agent operating mode.
        platform: Target OS (affects system prompt).
        max_steps: Maximum loop iterations.
        action_delay: Seconds to wait after each action before the next screenshot.
        provider: LLM provider (e.g. "openai", "azure"). Defaults to first
            supported provider of the model.
        azure_endpoint: Azure OpenAI endpoint URL (required when provider="azure").
        gta1_client: Pre-constructed GTA1Client. Auto-constructed when None;
            used for both GTA1 grounding and read_field clipboard correction.
        grounding: Grounding strategy — "gta1" (default) or "omniparser".
        preprocessing_mode: Image preprocessing applied to screenshots before
            grounding/LLM. One of "raw", "clahe", "adaptive_thresh",
            "edge_overlay", "clahe+edges". Defaults to "raw".
        task_execution: The specific execution this agent run is driving.
            Determines ``system_config`` and per-execution aggregate behaviour.
            Set by ``ProcedureRunner`` for each execution; carries resolved_outputs.

    Returns:
        Initialised ReActAgent.

    Raises:
        ValueError: If the model is not found, required clients are missing,
            or the API key is unavailable.
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

    # Always try to construct GTA1Client — needed for read_field clipboard correction
    # even when grounding="omniparser". Construction failure degrades gracefully:
    # clipboard correction is disabled but the agent still runs.
    if gta1_client is None:
        try:
            gta1_client = GTA1Client()
        except Exception as exc:
            logger.warning(
                "GTA1Client construction failed — clipboard correction disabled: %s", exc
            )
            gta1_client = None

    strategy = _resolve_grounding(grounding, omniparser_client, gta1_client)

    return ReActAgent(
        model_name=model_name,
        provider=provider,
        llm_client=llm_client,
        state=state,
        tools_collection=tools_collection,
        save_folder=save_folder,
        mode=mode,
        platform=platform,
        max_steps=max_steps,
        action_delay=action_delay,
        gta1_client=gta1_client,
        paddleocr_client=paddleocr_client,
        preprocessing_mode=_resolve_preprocessing(preprocessing_mode),
        grounding_strategy=strategy,
        task_execution=task_execution,
    )


def _resolve_preprocessing(preprocessing_mode: str) -> PreprocessingMode:
    """Parse and validate a preprocessing mode string."""
    try:
        return PreprocessingMode(preprocessing_mode)
    except ValueError:
        valid = [m.value for m in PreprocessingMode]
        raise ValueError(
            f"Unknown preprocessing_mode {preprocessing_mode!r}. Valid options: {valid}"
        )


_GROUNDING_MODES = frozenset({"gta1", "omniparser"})


def _resolve_grounding(
    grounding: str,
    omniparser_client: Optional[OmniParserClient],
    gta1_client: GTA1Client,
) -> GroundingStrategy:
    """Return the GroundingStrategy for the requested grounding mode."""
    if grounding not in _GROUNDING_MODES:
        raise ValueError(
            f"Unknown grounding mode {grounding!r}. Valid options: {sorted(_GROUNDING_MODES)}"
        )
    if grounding == "gta1":
        return GTA1Grounding(gta1_client)
    if omniparser_client is None:
        raise ValueError("omniparser grounding requires omniparser_client")
    return OmniParserGrounding(omniparser_client)


def _create_llm_client(
    provider: str,
    llm_cfg: dict,
    api_key: str,
    azure_endpoint: Optional[str] = None,
) -> BaseLLMClient:
    """Instantiate the correct LLM client from provider + model config."""
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
