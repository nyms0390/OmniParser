"""
Model and provider configuration registries.

Three concerns are now separated:
- PROVIDER_CONFIG : API endpoint overrides (only providers that need a non-SDK-managed base URL)
- LLM_MODELS      : per-model capabilities, generation defaults, and pricing
- OCR_CONFIG      : OCR backend configuration (unchanged)

Agent type is selected in the UI and passed directly to the factory — it is NOT stored here.

Pricing structure (Option C — per-provider overrides):
    "pricing": {
        "_default": {"input": <$/1M>, "output": <$/1M>},
        "<provider>": {"input": ..., "output": ...},   # only when it differs from _default
    }
"""

from typing import Any, Dict, List, Optional

from omnitool.gradio.config.enums import APIProvider


# ---------------------------------------------------------------------------
# Provider configuration
#
# Only providers that require an explicit base_url override are listed.
# Providers not listed here have their endpoints managed by their SDK:
#   azure     → endpoint comes from settings.azure_endpoint
#   anthropic → SDK-managed
#   bedrock   → SDK-managed
#   vertex    → SDK-managed
#   groq      → SDK-managed
# ---------------------------------------------------------------------------

PROVIDER_CONFIG: Dict[str, str] = {
    APIProvider.OPENAI:    "https://api.openai.com/v1",
    APIProvider.DASHSCOPE: "https://dashscope.aliyuncs.com/compatible-mode/v1",
}


# ---------------------------------------------------------------------------
# LLM model registry
#
# Keys are short model IDs used in the UI and factory (e.g. "gpt-4o").
# Fields:
#   internal_name       : exact model string sent to the API
#   supported_providers : APIProvider values the model is available through
#   pricing             : per-provider cost metadata (input/output USD per 1M tokens)
#                         "_default" is the fallback; add a provider key only when
#                         that provider's pricing differs from the default.
#   max_tokens          : upper bound on generated tokens
#   temperature         : default sampling temperature
# ---------------------------------------------------------------------------

LLM_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-4o": {
        "internal_name": "gpt-4o-2024-11-20",
        "supported_providers": [APIProvider.OPENAI],
        "pricing": {
            "_default": {"input": 2.50, "output": 10.00},
        },
        "max_tokens": 4096,
        "temperature": 0.0,
    },
    "gpt-4.1": {
        "internal_name": "gpt-4.1",
        "supported_providers": [APIProvider.AZURE, APIProvider.OPENAI],
        "pricing": {
            "_default": {"input": 2.00, "output": 8.00},
        },
        "max_tokens": 4096,
        "temperature": 0.0,
    },
    "gpt-5.1-codex": {
        "internal_name": "gpt-5.1-codex",
        "api_mode": "responses",
        "supported_providers": [APIProvider.AZURE, APIProvider.OPENAI],
        "pricing": {
            "_default": {"input": 1.25, "output": 10.00},
        },
        "max_tokens": 4096,
        "temperature": 0.0,
    },
    "gpt-5.2": {
        "internal_name": "gpt-5.2",
        "supported_providers": [APIProvider.AZURE, APIProvider.OPENAI],
        "pricing": {
            "_default": {"input": 1.75, "output": 14.00},
        },
        "max_tokens": 4096,
        "temperature": 0.0,
    },
    "gpt-5.4": {
        "internal_name": "gpt-5.4",
        "supported_providers": [APIProvider.AZURE, APIProvider.OPENAI],
        "pricing": {
            "_default": {"input": 2.00, "output": 16.00},
        },
        "max_tokens": 4096,
        "temperature": 0.0,
    },
    "o1": {
        "internal_name": "o1",
        "supported_providers": [APIProvider.OPENAI],
        "pricing": {
            "_default": {"input": 15.00, "output": 60.00},
        },
        "max_tokens": 4096,
        "temperature": 0.0,
    },
    "o3-mini": {
        "internal_name": "o3-mini",
        "supported_providers": [APIProvider.OPENAI],
        "pricing": {
            "_default": {"input": 1.10, "output": 4.40},
        },
        "max_tokens": 4096,
        "temperature": 0.0,
    },
    "qwen2.5vl": {
        "internal_name": "qwen2.5-vl-72b-instruct",
        "supported_providers": [APIProvider.DASHSCOPE],
        "pricing": {
            "_default": {"input": 2.20, "output": 2.20},
        },
        "max_tokens": 2048,
        "temperature": 0.0,
    },
    "deepseek-r1": {
        "internal_name": "deepseek-r1-distill-llama-70b",
        "supported_providers": [APIProvider.GROQ],
        "pricing": {
            "_default": {"input": 0.75, "output": 0.99},
        },
        "max_tokens": 4096,
        "temperature": 0.6,
    },
    "deepseek-V3.2": {
        "internal_name": "deepseek-V3.2",
        "supported_providers": [APIProvider.AZURE],
        "pricing": {
            "_default": {"input": 0.58, "output": 1.68},
        },
        "max_tokens": 4096,
        "temperature": 0.0,
    },
    "kimi-k2.5": {
        "internal_name": "Kimi-K2.5",
        "supported_providers": [APIProvider.AZURE],
        "pricing": {
            "_default": {"input": 0.60, "output": 2.50},
        },
        "max_tokens": 4096,
        "temperature": 0.0,
    },
    "claude-3-5-sonnet": {
        "internal_name": "claude-3-5-sonnet-20241022",
        "supported_providers": [
            APIProvider.ANTHROPIC, APIProvider.BEDROCK, APIProvider.VERTEX,
        ],
        "pricing": {
            "_default": {"input": 3.00, "output": 15.00},
            "bedrock":  {"input": 3.37, "output": 16.88},
        },
        "max_tokens": 4096,
        "temperature": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_llm_config(model_name: str) -> Dict[str, Any]:
    """Return the LLM_MODELS entry for *model_name*.

    Raises:
        ValueError: If *model_name* is not registered.
    """
    if model_name not in LLM_MODELS:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            f"Available models: {list(LLM_MODELS)}"
        )
    return LLM_MODELS[model_name]


def get_provider_config(provider: str) -> Optional[str]:
    """Return the base_url for *provider*, or None if SDK-managed.

    Args:
        provider: APIProvider value (or plain string).
    """
    return PROVIDER_CONFIG.get(provider)


def get_all_model_names() -> List[str]:
    """Return all registered model IDs (for UI dropdowns)."""
    return list(LLM_MODELS)


def get_supported_providers(model_name: str) -> List[str]:
    """Return provider strings for *model_name* (for UI provider dropdown)."""
    cfg = get_llm_config(model_name)
    return [str(p) for p in cfg["supported_providers"]]


def get_pricing(model_name: str, provider: str) -> Dict[str, float]:
    """Return ``{"input": <$/1M>, "output": <$/1M>}`` for *model_name* + *provider*.

    Falls back to the ``"_default"`` entry when no provider-specific override exists.
    Returns zeros if the model has no pricing entry at all.

    Args:
        model_name: Model ID from LLM_MODELS (e.g. "gpt-4o").
        provider:   Active provider string (e.g. "openai", "azure").
    """
    try:
        pricing = get_llm_config(model_name).get("pricing", {})
    except ValueError:
        return {"input": 0.0, "output": 0.0}
    return pricing.get(provider) or pricing.get("_default") or {"input": 0.0, "output": 0.0}


# ---------------------------------------------------------------------------
# OCR backend configuration (unchanged)
# ---------------------------------------------------------------------------

OCR_CONFIG: Dict[str, Dict[str, Any]] = {
    "easyocr": {
        "backend_class": "EasyOCRBackend",
        "language": "ch",
        "use_gpu": None,
        "backend_config": {},
    },
    "paddleocr": {
        "backend_class": "PaddleOCRBackend",
        "language": "ch",
        "use_gpu": None,
        "backend_config": {
            "text_threshold": 0.5,
            "api_url": "http://localhost:8080/",
        },
    },
}


def get_ocr_config(backend: str) -> Dict[str, Any]:
    """Return OCR backend configuration.

    Raises:
        ValueError: If *backend* is not registered.
    """
    key = backend.lower()
    if key not in OCR_CONFIG:
        raise ValueError(
            f"Unknown OCR backend: {backend!r}. "
            f"Available backends: {list(OCR_CONFIG)}"
        )
    return OCR_CONFIG[key]


def get_all_ocr_backends() -> List[str]:
    """Return all registered OCR backend names."""
    return list(OCR_CONFIG)
