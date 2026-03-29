"""
OpenAI provider.

Requires: pip install openai

Usage:
    from agentprobe.llm import OpenAI

    llm = OpenAI()                              # defaults to gpt-4o
    llm = OpenAI(model="gpt-4o-mini")           # cheaper model
    llm = OpenAI(model="o1-preview")            # reasoning model
    llm = OpenAI(api_key="sk-...", base_url="https://my-proxy.com/v1")  # custom endpoint

    # Also works with any OpenAI-compatible API (Azure, local, etc.):
    llm = OpenAI(base_url="http://localhost:11434/v1", model="llama3")

    response = llm.complete([
        SystemMessage("You are a test plan generator."),
        UserMessage("Here are my policy documents..."),
    ])
"""

from __future__ import annotations

from typing import List, Optional

from agentprobe.llm.base import LLMProvider, LLMResponse, Message, Usage


class OpenAI(LLMProvider):
    """OpenAI provider. Also works with any OpenAI-compatible API."""

    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **client_kwargs,
    ):
        """
        Args:
            model: Model name (e.g. "gpt-4o", "gpt-4o-mini", "o1-preview").
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            base_url: Custom API endpoint. Use for Azure, local models, proxies.
            max_tokens: Default max tokens for completions.
            temperature: Default temperature for completions.
            **client_kwargs: Additional kwargs passed to openai.OpenAI().
        """
        super().__init__(
            provider_name="openai",
            model=model,
            default_max_tokens=max_tokens,
            default_temperature=temperature,
        )
        self._api_key = api_key
        self._base_url = base_url
        self._client_kwargs = client_kwargs
        self._client = None

    def _get_client(self):
        """Lazy-initialize the OpenAI client."""
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "The 'openai' package is required for the OpenAI provider. "
                    "Install it with: pip install openai"
                )
            kwargs = dict(self._client_kwargs)
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def complete(
        self,
        messages: List[Message],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> LLMResponse:
        client = self._get_client()

        # OpenAI accepts system messages in the messages list directly
        api_messages = [msg.to_dict() for msg in messages]

        # Build request
        request = {
            "model": self.model,
            "messages": api_messages,
            **kwargs,
        }

        # Some models (o1, o3) don't support max_tokens or temperature
        # Let users pass them explicitly via kwargs to override
        max_tok = max_tokens or self.default_max_tokens
        temp = temperature if temperature is not None else self.default_temperature

        if max_tok and "max_tokens" not in kwargs:
            request["max_tokens"] = max_tok
        if temp is not None and "temperature" not in kwargs:
            request["temperature"] = temp

        # Call API
        response = client.chat.completions.create(**request)

        choice = response.choices[0]
        usage_data = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            usage=Usage(
                prompt_tokens=usage_data.prompt_tokens if usage_data else 0,
                completion_tokens=usage_data.completion_tokens if usage_data else 0,
            ),
            model=response.model,
            stop_reason=choice.finish_reason,
            raw=response,
        )
