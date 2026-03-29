"""
LiteLLM provider — supports 100+ models through a single interface.

Requires: pip install litellm

Usage:
    from agentprobe.llm import LiteLLMProvider

    # Any model LiteLLM supports:
    llm = LiteLLMProvider(model="claude-sonnet-4-20250514")
    llm = LiteLLMProvider(model="gpt-4o")
    llm = LiteLLMProvider(model="bedrock/claude-3-sonnet")
    llm = LiteLLMProvider(model="azure/gpt-4")
    llm = LiteLLMProvider(model="ollama/llama3")
    llm = LiteLLMProvider(model="together_ai/mistralai/Mixtral-8x7B")

    response = llm.complete([
        SystemMessage("You are a test plan generator."),
        UserMessage("Here are my policy documents..."),
    ])
"""

from __future__ import annotations

from typing import List, Optional

from agentprobe.llm.base import LLMProvider, LLMResponse, Message, Usage


class LiteLLMProvider(LLMProvider):
    """
    LiteLLM provider — universal interface to 100+ LLM providers.

    This is a thin wrapper around litellm.completion(). Any model string
    that LiteLLM supports works here. See: https://docs.litellm.ai/docs/providers
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **litellm_kwargs,
    ):
        """
        Args:
            model: Any LiteLLM-compatible model string.
            max_tokens: Default max tokens for completions.
            temperature: Default temperature for completions.
            **litellm_kwargs: Additional kwargs passed to litellm.completion().
                Can include api_key, api_base, custom_llm_provider, etc.
        """
        super().__init__(
            provider_name="litellm",
            model=model,
            default_max_tokens=max_tokens,
            default_temperature=temperature,
        )
        self._litellm_kwargs = litellm_kwargs

    def complete(
        self,
        messages: List[Message],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> LLMResponse:
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "The 'litellm' package is required for the LiteLLM provider. "
                "Install it with: pip install litellm"
            )

        api_messages = [msg.to_dict() for msg in messages]

        # Merge default kwargs with per-call kwargs
        request = {
            **self._litellm_kwargs,
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_tokens or self.default_max_tokens,
            "temperature": temperature if temperature is not None else self.default_temperature,
            **kwargs,
        }

        response = litellm.completion(**request)

        choice = response.choices[0]
        usage_data = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            usage=Usage(
                prompt_tokens=getattr(usage_data, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage_data, "completion_tokens", 0) or 0,
            ),
            model=getattr(response, "model", self.model) or self.model,
            stop_reason=getattr(choice, "finish_reason", None),
            raw=response,
        )
