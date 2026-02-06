"""
Model configuration registry mapping model names to agent types, LLM clients, and pricing.
"""

from enum import StrEnum
from typing import Any, Dict


class APIProvider(StrEnum):
    """Extended APIProvider enum including all supported providers."""
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    OPENAI = "openai"
    GROQ = "groq"
    DASHSCOPE = "dashscope"


# Pricing format: {token_type: "total"|"separate", cost_per_1m: float or dict}
# For separate token types (Anthropic), cost_per_1m is dict: {input: float, output: float}
MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    # GPT-4o Standard (OpenAI)
    "omniparser + gpt-4o": {
        "internal_name": "gpt-4o-2024-11-20",
        "agent_type": "VLMAgent",
        "llm_client": "openai",
        "provider": APIProvider.OPENAI,
        "provider_base_url": "https://api.openai.com/v1",
        "pricing": {
            "token_type": "total",
            "cost_per_1m": 2.5,  # USD per 1M tokens
        },
        "max_tokens": 4096,
        "temperature": 0.0,
        "supports_images": True,
        "orchestrated_variant": "omniparser + gpt-4o-orchestrated",
    },
    # O1 Standard (OpenAI)
    "omniparser + o1": {
        "internal_name": "o1",
        "agent_type": "VLMAgent",
        "llm_client": "openai",
        "provider": APIProvider.OPENAI,
        "provider_base_url": "https://api.openai.com/v1",
        "pricing": {
            "token_type": "total",
            "cost_per_1m": 15.0,  # USD per 1M tokens
        },
        "max_tokens": 4096,
        "temperature": 0.0,
        "supports_images": True,
        "orchestrated_variant": "omniparser + o1-orchestrated",
    },
    # O3-Mini (OpenAI)
    "omniparser + o3-mini": {
        "internal_name": "o3-mini",
        "agent_type": "VLMAgent",
        "llm_client": "openai",
        "provider": APIProvider.OPENAI,
        "provider_base_url": "https://api.openai.com/v1",
        "pricing": {
            "token_type": "total",
            "cost_per_1m": 1.1,  # USD per 1M tokens
        },
        "max_tokens": 4096,
        "temperature": 0.0,
        "supports_images": True,
        "orchestrated_variant": "omniparser + o3-mini-orchestrated",
    },
    # Qwen 2.5 VL (DashScope/Aliyun)
    "omniparser + qwen2.5vl": {
        "internal_name": "qwen2.5-vl-72b-instruct",
        "agent_type": "VLMAgent",
        "llm_client": "openai",  # Uses OpenAI-compatible API
        "provider": APIProvider.DASHSCOPE,
        "provider_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "pricing": {
            "token_type": "total",
            "cost_per_1m": 2.2,  # USD per 1M tokens
        },
        "max_tokens": 2048,  # Qwen has lower max tokens
        "temperature": 0.0,
        "supports_images": True,
        "orchestrated_variant": "omniparser + qwen2.5vl-orchestrated",
    },
    # DeepSeek R1 (Groq)
    "omniparser + R1": {
        "internal_name": "deepseek-r1-distill-llama-70b",
        "agent_type": "VLMAgent",
        "llm_client": "groq",
        "provider": APIProvider.GROQ,
        "provider_base_url": None,  # Groq manages its own base URL
        "pricing": {
            "token_type": "total",
            "cost_per_1m": 0.99,  # USD per 1M tokens
        },
        "max_tokens": 4096,
        "temperature": 0.6,  # Groq R1 uses different temperature
        "supports_images": False,  # R1 doesn't support images
        "orchestrated_variant": "omniparser + R1-orchestrated",
    },
    # GPT-4o Orchestrated (OpenAI)
    "omniparser + gpt-4o-orchestrated": {
        "internal_name": "gpt-4o-2024-11-20",
        "agent_type": "VLMAgent",  # Same agent, orchestration in core.orchestrator
        "llm_client": "openai",
        "provider": APIProvider.OPENAI,
        "provider_base_url": "https://api.openai.com/v1",
        "pricing": {
            "token_type": "total",
            "cost_per_1m": 2.5,
        },
        "max_tokens": 4096,
        "temperature": 0.0,
        "supports_images": True,
        "orchestrated_variant": None,
    },
    # O1 Orchestrated (OpenAI)
    "omniparser + o1-orchestrated": {
        "internal_name": "o1",
        "agent_type": "VLMAgent",
        "llm_client": "openai",
        "provider": APIProvider.OPENAI,
        "provider_base_url": "https://api.openai.com/v1",
        "pricing": {
            "token_type": "total",
            "cost_per_1m": 15.0,
        },
        "max_tokens": 4096,
        "temperature": 0.0,
        "supports_images": True,
        "orchestrated_variant": None,
    },
    # O3-Mini Orchestrated (OpenAI)
    "omniparser + o3-mini-orchestrated": {
        "internal_name": "o3-mini",
        "agent_type": "VLMAgent",
        "llm_client": "openai",
        "provider": APIProvider.OPENAI,
        "provider_base_url": "https://api.openai.com/v1",
        "pricing": {
            "token_type": "total",
            "cost_per_1m": 1.1,
        },
        "max_tokens": 4096,
        "temperature": 0.0,
        "supports_images": True,
        "orchestrated_variant": None,
    },
    # Qwen Orchestrated (DashScope/Aliyun)
    "omniparser + qwen2.5vl-orchestrated": {
        "internal_name": "qwen2.5-vl-72b-instruct",
        "agent_type": "VLMAgent",
        "llm_client": "openai",
        "provider": APIProvider.DASHSCOPE,
        "provider_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "pricing": {
            "token_type": "total",
            "cost_per_1m": 2.2,
        },
        "max_tokens": 2048,
        "temperature": 0.0,
        "supports_images": True,
        "orchestrated_variant": None,
    },
    # R1 Orchestrated (Groq)
    "omniparser + R1-orchestrated": {
        "internal_name": "deepseek-r1-distill-llama-70b",
        "agent_type": "VLMAgent",
        "llm_client": "groq",
        "provider": APIProvider.GROQ,
        "provider_base_url": None,
        "pricing": {
            "token_type": "total",
            "cost_per_1m": 0.99,
        },
        "max_tokens": 4096,
        "temperature": 0.6,
        "supports_images": False,
        "orchestrated_variant": None,
    },
    # Claude 3.5 Sonnet (Anthropic)
    "claude-3-5-sonnet-20241022": {
        "internal_name": "claude-3-5-sonnet-20241022",
        "agent_type": "AnthropicAgent",
        "llm_client": "anthropic",
        "provider": APIProvider.ANTHROPIC,  # Default; can be overridden to bedrock or vertex
        "provider_base_url": None,  # Anthropic SDK manages base URL
        "pricing": {
            "token_type": "separate",
            "cost_per_1m": {
                "input": 3.0,      # USD per 1M input tokens
                "output": 15.0,    # USD per 1M output tokens
            },
        },
        "max_tokens": 4096,
        "temperature": 0.0,
        "supports_images": True,
        "orchestrated_variant": None,
        "anthropic_providers": [
            APIProvider.ANTHROPIC,
            APIProvider.BEDROCK,
            APIProvider.VERTEX,
        ],
    },
}


# OCR Backend Configuration Registry
# Extensible configuration for different OCR backends
OCR_CONFIG: Dict[str, Dict[str, Any]] = {
    "easyocr": {
        "backend_class": "EasyOCRBackend",
        "language": "en",
        "use_gpu": None,  # None = auto-detect, True/False = explicit
        "backend_config": {
            # EasyOCR-specific parameters passed to reader.readtext()
            # Common options: detail (0=simple, 1=detailed), paragraph (bool)
        },
    },
    "paddleocr": {
        "backend_class": "PaddleOCRBackend",
        "language": "en",
        "use_gpu": None,  # None = auto-detect, True/False = explicit
        "backend_config": {
            # PaddleOCR-specific initialization parameters
            "text_threshold": 0.5,        # Confidence threshold for text detection
            "max_batch_size": 1024,       # Batch size for detection
            "use_dilation": True,         # Improves accuracy
            "det_db_score_mode": "slow",  # Improves accuracy (slow/fast)
            "rec_batch_num": 1024,        # Recognition batch number
            # For GPU version via API: set use_gpu=True and provide api_url in backend_config
            # "api_url": "http://localhost:8000/ocr"  # GPU API server endpoint
        },
    },
}


def get_ocr_config(backend: str) -> Dict[str, Any]:
    """Get configuration for a specific OCR backend.
    
    Args:
        backend: Backend name (e.g., 'easyocr', 'paddleocr')
        
    Returns:
        OCR backend configuration dictionary
        
    Raises:
        ValueError: If backend not found in OCR_CONFIG
    """
    backend_lower = backend.lower()
    if backend_lower not in OCR_CONFIG:
        raise ValueError(
            f"Unknown OCR backend: {backend}. "
            f"Available backends: {list(OCR_CONFIG.keys())}"
        )
    return OCR_CONFIG[backend_lower]


def get_all_ocr_backends() -> list[str]:
    """Get list of all available OCR backends."""
    return list(OCR_CONFIG.keys())


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model.
    
    Args:
        model_name: Display name of the model (key in MODEL_CONFIG)
        
    Returns:
        Model configuration dictionary
        
    Raises:
        ValueError: If model_name not found in MODEL_CONFIG
    """
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_CONFIG.keys())}")
    return MODEL_CONFIG[model_name]


def get_all_model_names() -> list[str]:
    """Get list of all available model display names."""
    return list(MODEL_CONFIG.keys())


def is_orchestrated_model(model_name: str) -> bool:
    """Check if a model is orchestrated variant.
    
    Args:
        model_name: Display name of the model
        
    Returns:
        True if model has 'orchestrated' in its name
    """
    return "orchestrated" in model_name.lower()
