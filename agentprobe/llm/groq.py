"""
Groq LLM provider — uses Groq's OpenAI-compatible API.

Groq is free (no credit card), fast, and runs Llama 3.3 70B.
This adapter wraps the OpenAI SDK pointed at Groq's endpoint.

Usage:
    from agentprobe.llm.groq import Groq

    llm = Groq(api_key="your-key", model="llama-3.3-70b-versatile")
    response = llm.complete([UserMessage("Hello")])
"""

from __future__ import annotations

from typing import List, Optional

from agentprobe.llm.base import LLMProvider, LLMResponse, Message, Usage


class Groq(LLMProvider):
    """
    Groq LLM provider using the OpenAI-compatible SDK.

    Requires: pip install openai
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        default_max_tokens: int = 8192,
        default_temperature: float = 0.0,
    ):
        super().__init__(
            provider_name="groq",
            model=model,
            default_max_tokens=default_max_tokens,
            default_temperature=default_temperature,
        )
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("Groq provider requires the openai package: pip install openai")

            import os
            key = self._api_key or os.environ.get("GROQ_API_KEY")
            if not key:
                raise ValueError("Groq API key required. Pass api_key= or set GROQ_API_KEY env var.")

            self._client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=key,
            )
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

        response = client.chat.completions.create(
            model=self.model,
            messages=[m.to_dict() for m in messages],
            max_tokens=max_tokens or self.default_max_tokens,
            temperature=temperature if temperature is not None else self.default_temperature,
            **kwargs,
        )

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            usage=Usage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
            model=response.model or self.model,
            stop_reason=choice.finish_reason,
            raw=response,
        )
