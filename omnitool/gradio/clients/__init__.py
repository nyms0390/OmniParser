"""
Clients module initialization.
"""

from omnitool.gradio.clients.llm.base import BaseLLMClient
from omnitool.gradio.clients.llm import (
    AnthropicClient,
    GroqClient,
    OpenAIClient,
    get_llm_client,
    get_llm_client_for_model,
)
from omnitool.gradio.clients.external.omniparser import OmniParserClient
from omnitool.gradio.clients.external.paddleocr import PaddleOCRClient
from omnitool.gradio.clients.external.windows_host import WindowsHostClient

__all__ = [
    "BaseLLMClient",
    "OpenAIClient",
    "GroqClient",
    "AnthropicClient",
    "OmniParserClient",
    "PaddleOCRClient",
    "WindowsHostClient",
    "get_llm_client",
    "get_llm_client_for_model",
]
