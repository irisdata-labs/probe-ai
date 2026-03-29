"""
Anthropic Claude provider.

Requires: pip install anthropic

Usage:
    from agentprobe.llm import Claude

    llm = Claude()                                          # defaults to claude-sonnet-4-20250514
    llm = Claude(model="claude-sonnet-4-20250514")   # specific model
    llm = Claude(api_key="sk-...")                          # explicit key (otherwise uses ANTHROPIC_API_KEY)

    response = llm.complete([
        SystemMessage("You are a test plan generator."),
        UserMessage("Here are my policy documents..."),
    ])
"""

from __future__ import annotations

from typing import List, Optional

from agentprobe.llm.base import LLMProvider, LLMResponse, Message, Usage


class Claude(LLMProvider):
    """Anthropic Claude provider."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **client_kwargs,
    ):
        """
        Args:
            model: Claude model name (e.g. "claude-sonnet-4-20250514").
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            max_tokens: Default max tokens for completions.
            temperature: Default temperature for completions.
            **client_kwargs: Additional kwargs passed to anthropic.Anthropic().
        """
        super().__init__(
            provider_name="anthropic",
            model=model,
            default_max_tokens=max_tokens,
            default_temperature=temperature,
        )
        self._api_key = api_key
        self._client_kwargs = client_kwargs
        self._client = None

    def _get_client(self):
        """Lazy-initialize the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "The 'anthropic' package is required for the Claude provider. "
                    "Install it with: pip install anthropic"
                )
            kwargs = dict(self._client_kwargs)
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = anthropic.Anthropic(**kwargs)
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

        # Anthropic API takes system as a separate parameter
        system_text = None
        api_messages = []
        for msg in messages:
            if msg.role == "system":
                system_text = msg.content
            else:
                api_messages.append(msg.to_dict())

        # Build request
        request = {
            "model": self.model,
            "max_tokens": max_tokens or self.default_max_tokens,
            "messages": api_messages,
            **kwargs,
        }

        if system_text:
            request["system"] = system_text

        temp = temperature if temperature is not None else self.default_temperature
        if temp is not None:
            request["temperature"] = temp

        # Call API
        response = client.messages.create(**request)

        # Extract text content
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return LLMResponse(
            content=content,
            usage=Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            ),
            model=response.model,
            stop_reason=response.stop_reason,
            raw=response,
        )
