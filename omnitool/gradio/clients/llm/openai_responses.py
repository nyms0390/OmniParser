"""
OpenAI Responses API client for models that use /v1/responses (e.g. gpt-5.1-codex).

Supports both OpenAI (api_key + base_url) and Azure OpenAI
(azure_endpoint + DefaultAzureCredential).
"""

from typing import Any, Dict, List, Optional, Tuple

from omnitool.gradio.clients.llm.base import BaseLLMClient


class OpenAIResponsesClient(BaseLLMClient):
    """LLM client for the OpenAI Responses API.

    Used for models with api_mode == "responses" (e.g. gpt-5.1-codex).
    Works for both direct OpenAI access and Azure OpenAI.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        **kwargs,
    ):
        """Initialize Responses API client.

        For OpenAI: provide api_key and optionally base_url.
        For Azure:  provide azure_endpoint (uses DefaultAzureCredential).

        Args:
            model: Model deployment name.
            api_key: OpenAI API key (not used for Azure).
            base_url: Override base URL (e.g. for OpenAI direct).
            azure_endpoint: Azure OpenAI endpoint URL (triggers Azure auth).
            **kwargs: Generation defaults (temperature, max_tokens, etc).
        """
        super().__init__(api_key, model, **kwargs)

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        if azure_endpoint:
            try:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            except ImportError:
                raise ImportError(
                    "azure-identity package required. "
                    "Install with: pip install azure-identity"
                )
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default",
            )
            azure_base = azure_endpoint.rstrip("/") + "/openai/v1/"
            self.client = OpenAI(
                api_key="placeholder",  # ignored — Azure uses token auth
                base_url=azure_base,
                default_headers={
                    "Authorization": f"Bearer {token_provider()}",
                },
            )
            self._provider = "azure"
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url or "https://api.openai.com/v1",
            )
            self._provider = "openai"

    @staticmethod
    def _to_responses_input(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert chat-style messages to Responses API input format.

        Args:
            messages: Standard messages list with role/content.

        Returns:
            Responses API input array.
        """
        result = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                content_type = "output_text" if role == "assistant" else "input_text"
                converted = [{"type": content_type, "text": content}]
            elif isinstance(content, list):
                converted = []
                for item in content:
                    if item.get("type") == "text":
                        content_type = (
                            "output_text" if role == "assistant" else "input_text"
                        )
                        converted.append({"type": content_type, "text": item["text"]})
                    elif item.get("type") == "image_url":
                        converted.append({
                            "type": "input_image",
                            "image_url": item["image_url"]["url"],
                        })
            else:
                converted = [{"type": "input_text", "text": str(content)}]

            result.append({"role": role, "content": converted})

        return result

    def generate(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str = "",
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate response using the Responses API.

        Args:
            messages: Conversation messages.
            system_prompt: System instructions (mapped to `instructions` param).
            **kwargs: Generation parameters (temperature, max_tokens, etc).

        Returns:
            (response_text, metadata)
        """
        input_messages = self._to_responses_input(
            self._process_messages(messages)
        )

        generation_params: Dict[str, Any] = {
            "model": self.model,
            "input": input_messages,
        }

        if system_prompt:
            generation_params["instructions"] = system_prompt

        max_tokens = kwargs.get("max_tokens", self.kwargs.get("max_tokens"))
        if max_tokens is not None:
            generation_params["max_output_tokens"] = max_tokens

        temperature = kwargs.get("temperature", self.kwargs.get("temperature"))
        if temperature is not None:
            generation_params["temperature"] = temperature

        try:
            response = self.client.responses.create(**generation_params)

            # Extract text from first output message
            response_text = ""
            for output_item in response.output:
                for content_item in output_item.content:
                    if content_item.type in ("output_text", "text"):
                        response_text = content_item.text
                        break
                if response_text:
                    break

            metadata = {
                "tokens": response.usage.total_tokens,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "model": self.model,
                "provider": self._provider,
            }

            return response_text, metadata

        except Exception as e:
            raise Exception(f"Responses API call failed: {str(e)}")
