"""
Azure OpenAI LLM client implementation supporting GPT-4o via Azure OpenAI Service.
"""

from typing import Any, Dict, List, Tuple

from omnitool.gradio.clients.llm.base import BaseLLMClient


class AzureOpenAIClient(BaseLLMClient):
    """Azure OpenAI LLM client.

    Uses Azure OpenAI Service with DefaultAzureCredential for authentication.
    Supports GPT-4o and other models deployed in Azure OpenAI.
    """

    def __init__(
        self,
        model: str,
        azure_endpoint: str,
        **kwargs
    ):
        """Initialize Azure OpenAI client.

        Args:
            model: Model name (e.g., 'gpt-4o-2024-11-20')
            azure_endpoint: Azure OpenAI endpoint URL (e.g., 'https://xxx.openai.azure.com/')
            **kwargs: Additional arguments (temperature, max_tokens, etc)

        Raises:
            ImportError: If azure-identity or azure-openai packages are not installed
        """
        super().__init__(None, model, **kwargs)
        self.azure_endpoint = (azure_endpoint or "").rstrip("/")

        # Import here to avoid hard dependency
        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            from openai import AzureOpenAI

            # Create token provider using DefaultAzureCredential
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default"
            )

            # Initialize Azure OpenAI client
            self.client = AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                azure_ad_token_provider=token_provider,
                api_version="2025-01-01-preview",
            )
        except ImportError:
            raise ImportError(
                "azure-identity and azure-openai packages required. "
                "Install with: pip install azure-identity azure-openai"
            )

    def generate(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str = "",
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate response using Azure OpenAI API.

        Args:
            messages: Conversation messages
            system_prompt: System prompt
            **kwargs: Generation parameters (temperature, max_tokens, etc)

        Returns:
            (response_text, metadata)
        """
        # Prepare messages
        prepared_messages = self._process_messages(messages)

        # Add system prompt if provided
        if system_prompt:
            prepared_messages.insert(0, {
                "role": "system",
                "content": system_prompt
            })

        # Prepare generation parameters
        generation_params = {
            "model": self.model,
            "messages": prepared_messages,
            "temperature": kwargs.get("temperature", self.kwargs.get("temperature")),
            "max_completion_tokens": kwargs.get(
                "max_tokens", self.kwargs.get("max_tokens")
            ),
        }

        # Remove None values
        generation_params = {k: v for k, v in generation_params.items() if v is not None}

        tools = kwargs.get("tools")
        if tools:
            generation_params["tools"] = tools
            generation_params["tool_choice"] = "auto"
            generation_params["parallel_tool_calls"] = False

        try:
            # Call API
            response = self.client.chat.completions.create(**generation_params)

            choice = response.choices[0]
            msg = choice.message

            # Extract tool calls when present.
            tool_calls_out = []
            if msg.tool_calls:
                import json as _json
                for tc in msg.tool_calls:
                    try:
                        arguments = _json.loads(tc.function.arguments or "{}")
                    except (_json.JSONDecodeError, TypeError):
                        arguments = {}
                    tool_calls_out.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": arguments,
                    })

            # Build assistant message dict for history (preserves tool_calls).
            assistant_message: dict = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                import json as _json
                assistant_message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in msg.tool_calls
                ]

            response_text = msg.content or ""

            # Prepare metadata
            metadata = {
                "tokens": response.usage.total_tokens,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "model": self.model,
                "provider": "azure",
                "tool_calls": tool_calls_out,
                "assistant_message": assistant_message,
            }

            return response_text, metadata

        except Exception as e:
            raise Exception(f"Azure API call failed: {str(e)}")
