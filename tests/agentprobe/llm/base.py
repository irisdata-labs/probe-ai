"""
Base classes for the LLM provider abstraction.

LLMProvider is the protocol. Message types are the common input format.
LLMResponse is the common output format. Providers convert between these
and their SDK-specific formats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


# ============================================================================
# Message types — common input format for all providers
# ============================================================================

@dataclass
class Message:
    """A single message in a conversation."""
    role: Literal["system", "user", "assistant"]
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


def SystemMessage(content: str) -> Message:
    """Create a system message."""
    return Message(role="system", content=content)


def UserMessage(content: str) -> Message:
    """Create a user message."""
    return Message(role="user", content=content)


def AssistantMessage(content: str) -> Message:
    """Create an assistant message."""
    return Message(role="assistant", content=content)


# ============================================================================
# Response types — common output format from all providers
# ============================================================================

@dataclass
class Usage:
    """Token usage from an LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self):
        if self.total_tokens == 0 and (self.prompt_tokens or self.completion_tokens):
            self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    stop_reason: Optional[str] = None
    raw: Optional[Any] = None           # original SDK response for debugging

    @property
    def total_tokens(self) -> int:
        return self.usage.total_tokens


# ============================================================================
# Provider protocol — what every provider must implement
# ============================================================================

class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Subclass this and implement complete() to support any LLM.

    Minimal implementation:
        class MyLLM(LLMProvider):
            def __init__(self):
                super().__init__(provider_name="my-llm", model="my-model-v1")

            def complete(self, messages, **kwargs):
                # call your API
                text = my_api_call(messages)
                return LLMResponse(content=text, model="my-model-v1")
    """

    def __init__(
        self,
        provider_name: str = "unknown",
        model: str = "",
        default_max_tokens: int = 4096,
        default_temperature: float = 0.0,
    ):
        self.provider_name = provider_name
        self.model = model
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature

    @abstractmethod
    def complete(
        self,
        messages: List[Message],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send messages to the LLM and return a response.

        Args:
            messages: List of Message objects (system, user, assistant).
            max_tokens: Maximum tokens to generate. Uses provider default if None.
            temperature: Sampling temperature. Uses provider default if None.
            **kwargs: Provider-specific parameters passed through to the API.

        Returns:
            LLMResponse with content, usage stats, and model info.

        Raises:
            ImportError: If the provider's SDK is not installed.
            Exception: Provider-specific API errors are passed through.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
