"""Tests for agentprobe.core — decorators, context management, state/decision tracking."""

import pytest
from agentprobe.core import (
    AgentProbe,
    get_current_trace,
    get_current_span,
    record_state_change,
    record_decision,
    trace_agent,
    trace_step,
    trace_tool,
    trace_llm_call,
    step_span,
    _serialize_value,
)
from agentprobe.models import TraceRecord


@pytest.fixture(autouse=True)
def reset_probe():
    """Reset the global AgentProbe singleton between tests."""
    AgentProbe._instance = None
    yield
    AgentProbe._instance = None


class TestSerializeValue:
    def test_primitives(self):
        assert _serialize_value(None) is None
        assert _serialize_value(42) == 42
        assert _serialize_value(3.14) == 3.14
        assert _serialize_value("hello") == "hello"
        assert _serialize_value(True) is True

    def test_collections(self):
        assert _serialize_value([1, 2, 3]) == [1, 2, 3]
        assert _serialize_value({"a": 1}) == {"a": 1}
        assert _serialize_value((1, "two")) == [1, "two"]

    def test_nested(self):
        val = {"items": [{"name": "x", "count": 3}]}
        assert _serialize_value(val) == val

    def test_non_serializable(self):
        class Custom:
            pass
        result = _serialize_value(Custom())
        assert isinstance(result, str)

    def test_dict_with_non_string_keys(self):
        result = _serialize_value({1: "a", 2: "b"})
        assert result == {"1": "a", "2": "b"}


class TestAgentProbe:
    def test_singleton(self):
        probe = AgentProbe()
        probe.init()
        assert AgentProbe.get_instance() is probe

    def test_record_trace(self):
        probe = AgentProbe()
        probe.init()
        trace = TraceRecord(agent_name="test")
        probe.record_trace(trace)
        assert len(probe.traces) == 1

    def test_traces_returns_copy(self):
        probe = AgentProbe()
        probe.init()
        traces = probe.traces
        traces.append(TraceRecord(agent_name="fake"))
        assert len(probe.traces) == 0  # internal list unaffected

    def test_clear(self):
        probe = AgentProbe()
        probe.init()
        probe.record_trace(TraceRecord(agent_name="test"))
        probe.clear()
        assert len(probe.traces) == 0

    def test_exporter_called(self):
        exported = []

        class MockExporter:
            def export(self, trace):
                exported.append(trace)

        probe = AgentProbe(exporters=[MockExporter()])
        probe.init()

        @trace_agent("test")
        def my_agent(x):
            return x * 2

        my_agent(5)
        assert len(exported) == 1
        assert exported[0].agent_name == "test"

    def test_exporter_failure_doesnt_crash(self):
        class BadExporter:
            def export(self, trace):
                raise RuntimeError("export failed")

        probe = AgentProbe(exporters=[BadExporter()])
        probe.init()

        @trace_agent("test")
        def my_agent(x):
            return x

        result = my_agent(42)
        assert result == 42  # agent still works


class TestTraceAgent:
    def test_captures_trace(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("my-agent")
        def my_agent(msg):
            return {"response": msg}

        result = my_agent("hello")
        assert result == {"response": "hello"}
        assert len(probe.traces) == 1
        trace = probe.traces[0]
        assert trace.agent_name == "my-agent"
        assert trace.status == "ok"
        assert trace.duration_ms is not None
        assert trace.duration_ms > 0

    def test_captures_input_output(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test")
        def agent(data):
            return {"result": data["x"] * 2}

        agent({"x": 5})
        trace = probe.traces[0]
        assert trace.input_data == {"x": 5}
        assert trace.output_data == {"result": 10}

    def test_captures_error(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test")
        def failing_agent(x):
            raise ValueError("bad input")

        with pytest.raises(ValueError, match="bad input"):
            failing_agent(1)

        trace = probe.traces[0]
        assert trace.status == "error"
        assert trace.exception_info["type"] == "ValueError"
        assert trace.exception_info["message"] == "bad input"

    def test_with_tags(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test", tags=["prod", "v2"])
        def agent(x):
            return x

        agent(1)
        assert probe.traces[0].tags == ["prod", "v2"]

    def test_with_thread_id(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test", thread_id="thread-123")
        def agent(x):
            return x

        agent(1)
        assert probe.traces[0].thread_id == "thread-123"

    def test_with_metadata(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test", metadata={"env": "staging"})
        def agent(x):
            return x

        agent(1)
        assert probe.traces[0].metadata == {"env": "staging"}

    def test_no_probe_still_works(self):
        """Agent runs fine even without AgentProbe initialized."""
        @trace_agent("test")
        def agent(x):
            return x * 2

        assert agent(5) == 10


class TestTraceStep:
    def test_captures_span(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test")
        def agent(x):
            return process(x)

        @trace_step("process", step_type="reasoning")
        def process(x):
            return x * 2

        agent(5)
        trace = probe.traces[0]
        assert trace.step_count == 1
        assert trace.spans[0].name == "process"
        assert trace.spans[0].span_type == "reasoning"
        assert trace.spans[0].status == "ok"

    def test_nested_steps(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test")
        def agent(x):
            return outer(x)

        @trace_step("outer")
        def outer(x):
            return inner(x)

        @trace_step("inner")
        def inner(x):
            return x * 2

        agent(5)
        trace = probe.traces[0]
        assert trace.step_count == 2

        outer_span = next(s for s in trace.spans if s.name == "outer")
        inner_span = next(s for s in trace.spans if s.name == "inner")
        assert inner_span.parent_span_id == outer_span.span_id

    def test_step_outside_trace_runs_normally(self):
        @trace_step("standalone")
        def process(x):
            return x * 2

        assert process(5) == 10

    def test_step_captures_error(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test")
        def agent(x):
            return bad_step(x)

        @trace_step("bad")
        def bad_step(x):
            raise RuntimeError("oops")

        with pytest.raises(RuntimeError):
            agent(1)

        trace = probe.traces[0]
        assert trace.spans[0].status == "error"
        assert trace.spans[0].exception_info["type"] == "RuntimeError"

    def test_step_with_tags(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test")
        def agent(x):
            return step(x)

        @trace_step("step", tags=["important"])
        def step(x):
            return x

        agent(1)
        assert probe.traces[0].spans[0].tags == ["important"]


class TestTraceTool:
    def test_tool_span_type(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test")
        def agent(x):
            return lookup(x)

        @trace_tool("order-lookup")
        def lookup(x):
            return {"order_id": x}

        agent("ORD-123")
        span = probe.traces[0].spans[0]
        assert span.span_type == "tool_call"
        assert span.name == "order-lookup"


class TestTraceLLMCall:
    def test_llm_span_type(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test")
        def agent(x):
            return generate(x)

        @trace_llm_call("generate-response")
        def generate(prompt):
            from agentprobe.core import get_current_span
            span = get_current_span()
            if span:
                span.set_llm_metadata(model="claude-sonnet-4-20250514", total_tokens=150)
            return "response"

        agent("hello")
        span = probe.traces[0].spans[0]
        assert span.span_type == "llm_call"
        assert span.model == "claude-sonnet-4-20250514"
        assert span.total_tokens == 150


class TestStepSpan:
    def test_context_manager(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test")
        def agent(x):
            with step_span("classify", step_type="reasoning", input_data=x) as span:
                result = "positive" if x > 0 else "negative"
                span.set_output(result)
            return result

        agent(5)
        trace = probe.traces[0]
        assert trace.step_count == 1
        span = trace.spans[0]
        assert span.name == "classify"
        assert span.span_type == "reasoning"
        assert span.output_data == "positive"
        assert span.status == "ok"

    def test_context_manager_error(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test")
        def agent(x):
            with step_span("bad") as span:
                raise ValueError("oops")

        with pytest.raises(ValueError):
            agent(1)

        span = probe.traces[0].spans[0]
        assert span.status == "error"

    def test_context_manager_outside_trace(self):
        """step_span works without a trace — just yields a dummy span."""
        with step_span("standalone") as span:
            span.set_output("test")
        assert span.output_data == "test"

    def test_nested_with_decorator(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test")
        def agent(x):
            return outer(x)

        @trace_step("outer")
        def outer(x):
            with step_span("inner") as span:
                span.set_output(x * 2)
            return x * 2

        agent(5)
        trace = probe.traces[0]
        assert trace.step_count == 2
        inner = next(s for s in trace.spans if s.name == "inner")
        outer_s = next(s for s in trace.spans if s.name == "outer")
        assert inner.parent_span_id == outer_s.span_id


class TestStateChangeTracking:
    def test_record_state_change(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test")
        def agent(x):
            record_state_change("balance", before=500, after=800)
            return x

        agent(1)
        trace = probe.traces[0]
        assert len(trace.state_changes) == 1
        sc = trace.state_changes[0]
        assert sc.key == "balance"
        assert sc.before == 500
        assert sc.after == 800

    def test_state_change_captures_step_name(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test")
        def agent(x):
            return process(x)

        @trace_step("process-refund")
        def process(x):
            record_state_change("inventory", before=10, after=11)
            return x

        agent(1)
        sc = probe.traces[0].state_changes[0]
        assert sc.step_name == "process-refund"

    def test_state_change_outside_trace_is_noop(self):
        # Should not raise
        record_state_change("anything", before=1, after=2)

    def test_multiple_state_changes(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test")
        def agent(x):
            record_state_change("a", before=1, after=2)
            record_state_change("b", before="x", after="y")
            return x

        agent(1)
        assert len(probe.traces[0].state_changes) == 2

    def test_state_changes_in_to_dict(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test")
        def agent(x):
            record_state_change("key", before=1, after=2)
            return x

        agent(1)
        d = probe.traces[0].to_dict()
        assert "state_changes" in d
        assert d["state_changes"][0]["key"] == "key"


class TestDecisionTracking:
    def test_record_decision(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test")
        def agent(x):
            record_decision("escalate", alternatives=["resolve", "transfer"], reason="high value")
            return x

        agent(1)
        trace = probe.traces[0]
        assert len(trace.decisions) == 1
        dec = trace.decisions[0]
        assert dec.chosen == "escalate"
        assert dec.alternatives == ["resolve", "transfer"]
        assert dec.reason == "high value"

    def test_decision_captures_step_name(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test")
        def agent(x):
            return evaluate(x)

        @trace_step("evaluate-return")
        def evaluate(x):
            record_decision("approve", alternatives=["deny"])
            return x

        agent(1)
        dec = probe.traces[0].decisions[0]
        assert dec.step_name == "evaluate-return"

    def test_decision_outside_trace_is_noop(self):
        record_decision("anything")

    def test_decisions_in_to_dict(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test")
        def agent(x):
            record_decision("proceed")
            return x

        agent(1)
        d = probe.traces[0].to_dict()
        assert "decisions" in d
        assert d["decisions"][0]["chosen"] == "proceed"


class TestGetCurrentTrace:
    def test_inside_agent(self):
        probe = AgentProbe()
        probe.init()

        captured = {}

        @trace_agent("test")
        def agent(x):
            captured["trace"] = get_current_trace()
            return x

        agent(1)
        assert captured["trace"] is not None
        assert captured["trace"].agent_name == "test"

    def test_outside_agent(self):
        assert get_current_trace() is None


class TestFullPipeline:
    def test_end_to_end(self):
        """Full pipeline: agent with tools, LLM, steps, state, decisions."""
        probe = AgentProbe()
        probe.init()

        @trace_agent("support-agent", tags=["test"])
        def handle_customer(message):
            order = lookup_order(message)
            with step_span("evaluate", step_type="reasoning", input_data=order) as span:
                if order["days"] > 30:
                    record_decision("deny", alternatives=["approve"], reason="outside window")
                    result = "denied"
                else:
                    record_decision("approve", alternatives=["deny"])
                    record_state_change("refund_status", before="none", after="processing")
                    result = "approved"
                span.set_output(result)
            return {"action": result, "order": order["order_id"]}

        @trace_tool("order-lookup")
        def lookup_order(msg):
            return {"order_id": "ORD-123", "days": 15, "amount": 79.99}

        result = handle_customer("I want to return my headphones")
        assert result["action"] == "approved"

        trace = probe.traces[0]
        assert trace.agent_name == "support-agent"
        assert trace.status == "ok"
        assert trace.tags == ["test"]
        assert trace.step_count == 2  # tool + evaluate
        assert len(trace.tool_calls) == 1
        assert len(trace.state_changes) == 1
        assert trace.state_changes[0].key == "refund_status"
        assert len(trace.decisions) == 1
        assert trace.decisions[0].chosen == "approve"

        # Verify serialization
        d = trace.to_dict()
        assert d["summary"]["step_count"] == 2
        assert d["summary"]["tool_calls"] == 1
        assert len(d["state_changes"]) == 1
        assert len(d["decisions"]) == 1
