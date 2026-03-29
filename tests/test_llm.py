"""
Tests for agentprobe.llm — the model-agnostic LLM interface.

Tests cover:
- Message construction and serialization
- Usage calculation
- LLMResponse properties
- LLMProvider contract (subclassing, repr)
- Claude provider initialization and message formatting
- OpenAI provider initialization and message formatting
- Custom provider implementation
"""

import pytest
from unittest.mock import MagicMock, patch
from agentprobe.llm.base import (
    LLMProvider,
    LLMResponse,
    Usage,
    Message,
    SystemMessage,
    UserMessage,
    AssistantMessage,
)


# ============================================================================
# Message tests
# ============================================================================

class TestMessages:
    def test_system_message(self):
        msg = SystemMessage("You are helpful.")
        assert msg.role == "system"
        assert msg.content == "You are helpful."

    def test_user_message(self):
        msg = UserMessage("Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_assistant_message(self):
        msg = AssistantMessage("Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_to_dict(self):
        msg = UserMessage("test")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "test"}

    def test_message_direct(self):
        msg = Message(role="user", content="direct")
        assert msg.role == "user"
        assert msg.content == "direct"


# ============================================================================
# Usage tests
# ============================================================================

class TestUsage:
    def test_auto_total(self):
        u = Usage(prompt_tokens=100, completion_tokens=50)
        assert u.total_tokens == 150

    def test_explicit_total(self):
        u = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=200)
        assert u.total_tokens == 200  # explicit overrides

    def test_zero_default(self):
        u = Usage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0

    def test_partial(self):
        u = Usage(prompt_tokens=100)
        assert u.total_tokens == 100


# ============================================================================
# LLMResponse tests
# ============================================================================

class TestLLMResponse:
    def test_basic(self):
        r = LLMResponse(content="Hello", model="test-model")
        assert r.content == "Hello"
        assert r.model == "test-model"
        assert r.total_tokens == 0

    def test_with_usage(self):
        r = LLMResponse(
            content="Hi",
            usage=Usage(prompt_tokens=10, completion_tokens=5),
            model="test",
        )
        assert r.total_tokens == 15

    def test_defaults(self):
        r = LLMResponse(content="x")
        assert r.model == ""
        assert r.stop_reason is None
        assert r.raw is None


# ============================================================================
# LLMProvider contract tests
# ============================================================================

class TestLLMProvider:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            LLMProvider()

    def test_custom_provider(self):
        class MockLLM(LLMProvider):
            def __init__(self):
                super().__init__(provider_name="mock", model="mock-v1")

            def complete(self, messages, **kwargs):
                text = f"Got {len(messages)} messages"
                return LLMResponse(
                    content=text,
                    usage=Usage(prompt_tokens=10, completion_tokens=5),
                    model=self.model,
                )

        llm = MockLLM()
        assert llm.provider_name == "mock"
        assert llm.model == "mock-v1"

        response = llm.complete([UserMessage("test")])
        assert response.content == "Got 1 messages"
        assert response.total_tokens == 15
        assert response.model == "mock-v1"

    def test_repr(self):
        class MockLLM(LLMProvider):
            def __init__(self):
                super().__init__(provider_name="mock", model="mock-v1")
            def complete(self, messages, **kwargs):
                return LLMResponse(content="")

        llm = MockLLM()
        assert repr(llm) == "MockLLM(model='mock-v1')"

    def test_defaults(self):
        class MockLLM(LLMProvider):
            def __init__(self):
                super().__init__(provider_name="test", model="m1",
                                 default_max_tokens=2048, default_temperature=0.5)
            def complete(self, messages, **kwargs):
                return LLMResponse(content="")

        llm = MockLLM()
        assert llm.default_max_tokens == 2048
        assert llm.default_temperature == 0.5


# ============================================================================
# Claude provider tests (without real API calls)
# ============================================================================

class TestClaudeProvider:
    def test_init_defaults(self):
        from agentprobe.llm.claude import Claude
        llm = Claude()
        assert llm.model == "claude-sonnet-4-20250514"
        assert llm.provider_name == "anthropic"
        assert llm.default_max_tokens == 4096
        assert llm.default_temperature == 0.0

    def test_init_custom(self):
        from agentprobe.llm.claude import Claude
        llm = Claude(model="claude-opus-4-20250514", max_tokens=8192, temperature=0.7)
        assert llm.model == "claude-opus-4-20250514"
        assert llm.default_max_tokens == 8192
        assert llm.default_temperature == 0.7

    def test_repr(self):
        from agentprobe.llm.claude import Claude
        llm = Claude(model="claude-haiku-4-20250414")
        assert "claude-haiku-4-20250414" in repr(llm)

    def test_system_message_extraction(self):
        """Verify Claude provider extracts system messages correctly."""
        from agentprobe.llm.claude import Claude

        llm = Claude(api_key="test-key")

        # Mock the anthropic client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response text")]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "end_turn"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        llm._client = mock_client

        messages = [
            SystemMessage("Be helpful"),
            UserMessage("Hello"),
            AssistantMessage("Hi"),
            UserMessage("How are you?"),
        ]

        response = llm.complete(messages)

        # Check system was extracted
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "Be helpful"
        assert len(call_kwargs["messages"]) == 3  # user, assistant, user (no system)
        assert call_kwargs["messages"][0] == {"role": "user", "content": "Hello"}

        # Check response
        assert response.content == "response text"
        assert response.usage.prompt_tokens == 100
        assert response.usage.completion_tokens == 50
        assert response.total_tokens == 150
        assert response.model == "claude-sonnet-4-20250514"

    def test_no_system_message(self):
        """Claude provider works without a system message."""
        from agentprobe.llm.claude import Claude

        llm = Claude(api_key="test-key")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="ok")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "end_turn"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        llm._client = mock_client

        llm.complete([UserMessage("test")])

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "system" not in call_kwargs

    def test_import_error(self):
        """Claude provider gives clear error if anthropic not installed."""
        from agentprobe.llm.claude import Claude
        llm = Claude(api_key="test")
        llm._client = None  # force re-init

        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="anthropic"):
                llm._get_client()


# ============================================================================
# OpenAI provider tests (without real API calls)
# ============================================================================

class TestOpenAIProvider:
    def test_init_defaults(self):
        from agentprobe.llm.openai import OpenAI
        llm = OpenAI()
        assert llm.model == "gpt-4o"
        assert llm.provider_name == "openai"

    def test_init_custom(self):
        from agentprobe.llm.openai import OpenAI
        llm = OpenAI(model="gpt-4o-mini", base_url="http://localhost:11434/v1")
        assert llm.model == "gpt-4o-mini"
        assert llm._base_url == "http://localhost:11434/v1"

    def test_messages_passed_directly(self):
        """OpenAI provider passes system messages in the messages list."""
        from agentprobe.llm.openai import OpenAI

        llm = OpenAI(api_key="test-key")

        mock_choice = MagicMock()
        mock_choice.message.content = "response"
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 80
        mock_usage.completion_tokens = 40

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = "gpt-4o"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        llm._client = mock_client

        messages = [
            SystemMessage("Be helpful"),
            UserMessage("Hello"),
        ]

        response = llm.complete(messages)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        # OpenAI keeps system in the messages list
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0] == {"role": "system", "content": "Be helpful"}

        assert response.content == "response"
        assert response.total_tokens == 120

    def test_import_error(self):
        from agentprobe.llm.openai import OpenAI
        llm = OpenAI(api_key="test")
        llm._client = None

        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai"):
                llm._get_client()


# ============================================================================
# Integration: provider interchangeability
# ============================================================================

class TestProviderInterchangeability:
    """Verify any provider can be used in the same code path."""

    def _make_mock_provider(self, name, response_text="mock response"):
        class MockProvider(LLMProvider):
            def __init__(self):
                super().__init__(provider_name=name, model=f"{name}-v1")
            def complete(self, messages, **kwargs):
                return LLMResponse(
                    content=response_text,
                    usage=Usage(prompt_tokens=50, completion_tokens=25),
                    model=self.model,
                )
        return MockProvider()

    def test_same_interface(self):
        """All providers work with the same calling code."""
        providers = [
            self._make_mock_provider("claude"),
            self._make_mock_provider("openai"),
            self._make_mock_provider("litellm"),
            self._make_mock_provider("custom"),
        ]

        messages = [
            SystemMessage("You are a test generator."),
            UserMessage("Generate a test plan."),
        ]

        for provider in providers:
            response = provider.complete(messages)
            assert isinstance(response, LLMResponse)
            assert isinstance(response.content, str)
            assert isinstance(response.usage, Usage)
            assert response.total_tokens == 75

    def test_function_accepts_any_provider(self):
        """Simulate how plan_generator will use the provider."""

        def generate_plan(docs: str, llm: LLMProvider) -> str:
            response = llm.complete([
                SystemMessage("You generate test plans."),
                UserMessage(f"Documents: {docs}"),
            ])
            return response.content

        provider_a = self._make_mock_provider("a", "Plan A")
        provider_b = self._make_mock_provider("b", "Plan B")

        assert generate_plan("policy docs", provider_a) == "Plan A"
        assert generate_plan("policy docs", provider_b) == "Plan B"
