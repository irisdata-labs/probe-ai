"""Tests for async support in decorators."""

import pytest
from agentprobe.core import (
    AgentProbe,
    trace_agent,
    trace_step,
    trace_tool,
    step_span,
    record_state_change,
    record_decision,
)


@pytest.fixture(autouse=True)
def reset_probe():
    AgentProbe._instance = None
    yield
    AgentProbe._instance = None


class TestAsyncDecorators:
    async def test_async_trace_agent(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("async-agent")
        async def agent(x):
            return x * 2

        result = await agent(5)
        assert result == 10
        assert len(probe.traces) == 1
        assert probe.traces[0].status == "ok"
        assert probe.traces[0].output_data == 10

    async def test_async_with_steps(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("async-agent")
        async def agent(x):
            return await process(x)

        @trace_step("process")
        async def process(x):
            return x * 2

        await agent(5)
        trace = probe.traces[0]
        assert trace.step_count == 1
        assert trace.spans[0].name == "process"

    async def test_async_error_capture(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("async-agent")
        async def agent(x):
            raise ValueError("async error")

        with pytest.raises(ValueError):
            await agent(1)

        trace = probe.traces[0]
        assert trace.status == "error"
        assert trace.exception_info["type"] == "ValueError"

    async def test_async_step_error(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("async-agent")
        async def agent(x):
            return await bad_step(x)

        @trace_step("bad")
        async def bad_step(x):
            raise RuntimeError("step failed")

        with pytest.raises(RuntimeError):
            await agent(1)

        span = probe.traces[0].spans[0]
        assert span.status == "error"

    async def test_mixed_sync_async_steps(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("mixed")
        async def agent(x):
            a = sync_step(x)
            b = await async_step(a)
            return b

        @trace_step("sync")
        def sync_step(x):
            return x + 1

        @trace_step("async")
        async def async_step(x):
            return x * 2

        result = await agent(5)
        assert result == 12
        trace = probe.traces[0]
        assert trace.step_count == 2

    async def test_async_state_and_decision(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("async-agent")
        async def agent(x):
            record_state_change("counter", before=0, after=x)
            record_decision("process", alternatives=["skip"])
            return x

        await agent(42)
        trace = probe.traces[0]
        assert len(trace.state_changes) == 1
        assert len(trace.decisions) == 1

    async def test_async_tool(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("async-agent")
        async def agent(x):
            return await lookup(x)

        @trace_tool("async-lookup")
        async def lookup(x):
            return {"found": True}

        await agent("test")
        span = probe.traces[0].spans[0]
        assert span.span_type == "tool_call"
        assert span.name == "async-lookup"
