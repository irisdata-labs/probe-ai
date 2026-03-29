"""
agentprobe.llm — Model-agnostic LLM interface.

Provides a simple protocol for calling LLMs, with built-in support for
Anthropic Claude and OpenAI. Anyone can plug in their own provider by
subclassing LLMProvider and implementing complete().

Usage:
    from agentprobe.llm import Claude, OpenAI, LiteLLMProvider

    # Claude (default)
    llm = Claude(model="claude-sonnet-4-20250514")

    # OpenAI
    llm = OpenAI(model="gpt-4o")

    # Any LiteLLM-supported model
    llm = LiteLLMProvider(model="bedrock/claude-3-sonnet")

    # Custom provider
    class MyLLM(LLMProvider):
        def complete(self, messages, **kwargs):
            return LLMResponse(content="...", usage=Usage(...))

    # All providers work the same way:
    response = llm.complete([
        SystemMessage("You are a helpful assistant."),
        UserMessage("What is 2+2?"),
    ])
    print(response.content)
    print(response.usage.total_tokens)
"""

from agentprobe.llm.base import (
    LLMProvider,
    LLMResponse,
    Usage,
    Message,
    SystemMessage,
    UserMessage,
    AssistantMessage,
)
from agentprobe.llm.claude import Claude
from agentprobe.llm.openai import OpenAI
from agentprobe.llm.litellm import LiteLLMProvider

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "Usage",
    "Message",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "Claude",
    "OpenAI",
    "LiteLLMProvider",
]
