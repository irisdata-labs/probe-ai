"""Tests for agentprobe.models — data structures for traces."""

import time
import pytest
from agentprobe.models import SpanRecord, TraceRecord, StateChange, Decision


class TestSpanRecord:
    def test_basic_creation(self):
        span = SpanRecord(name="test-step")
        assert span.name == "test-step"
        assert span.span_type == "generic"
        assert len(span.span_id) == 16

    def test_timing(self):
        span = SpanRecord(name="timed")
        span.start()
        time.sleep(0.01)
        span.end()
        assert span.duration_ms is not None
        assert span.duration_ms >= 10

    def test_duration_none_when_incomplete(self):
        span = SpanRecord(name="incomplete")
        assert span.duration_ms is None
        span.start()
        assert span.duration_ms is None

    def test_set_output(self):
        span = SpanRecord(name="test")
        span.set_output({"result": 42})
        assert span.output_data == {"result": 42}

    def test_set_status(self):
        span = SpanRecord(name="test")
        span.set_status("ok", "all good")
        assert span.status == "ok"
        assert span.status_message == "all good"

    def test_record_exception(self):
        span = SpanRecord(name="test")
        try:
            raise ValueError("test error")
        except ValueError as e:
            span.record_exception(e)
        assert span.status == "error"
        assert span.exception_info["type"] == "ValueError"
        assert span.exception_info["message"] == "test error"
        assert "traceback" in span.exception_info

    def test_llm_metadata(self):
        span = SpanRecord(name="llm")
        span.set_llm_metadata(
            model="claude-sonnet-4-20250514",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            temperature=0.7,
        )
        assert span.model == "claude-sonnet-4-20250514"
        assert span.prompt_tokens == 100
        assert span.total_tokens == 150
        assert span.temperature == 0.7

    def test_llm_metadata_partial(self):
        span = SpanRecord(name="llm")
        span.set_llm_metadata(model="gpt-4o")
        assert span.model == "gpt-4o"
        assert span.prompt_tokens is None

    def test_tool_metadata(self):
        span = SpanRecord(name="tool")
        span.set_tool_metadata(
            tool_name="order-lookup",
            tool_args={"order_id": "123"},
            tool_result={"status": "delivered"},
            tool_success=True,
        )
        assert span.tool_name == "order-lookup"
        assert span.tool_args == {"order_id": "123"}
        assert span.tool_success is True

    def test_to_dict_minimal(self):
        span = SpanRecord(name="test", span_type="generic")
        span.set_status("ok")
        d = span.to_dict()
        assert d["name"] == "test"
        assert d["span_type"] == "generic"
        assert d["status"] == "ok"
        assert "llm" not in d  # not populated
        assert "tool" not in d

    def test_to_dict_with_llm(self):
        span = SpanRecord(name="llm")
        span.set_llm_metadata(model="claude-sonnet-4-20250514", total_tokens=100)
        d = span.to_dict()
        assert "llm" in d
        assert d["llm"]["model"] == "claude-sonnet-4-20250514"

    def test_to_dict_with_tool(self):
        span = SpanRecord(name="tool")
        span.set_tool_metadata(tool_name="search", tool_success=True)
        d = span.to_dict()
        assert "tool" in d
        assert d["tool"]["tool_name"] == "search"

    def test_to_dict_with_tags(self):
        span = SpanRecord(name="tagged", tags=["critical", "v2"])
        d = span.to_dict()
        assert d["tags"] == ["critical", "v2"]

    def test_to_dict_without_tags(self):
        span = SpanRecord(name="untagged")
        d = span.to_dict()
        assert "tags" not in d


class TestTraceRecord:
    def test_basic_creation(self):
        trace = TraceRecord(agent_name="test-agent")
        assert trace.agent_name == "test-agent"
        assert len(trace.trace_id) == 32

    def test_timing(self):
        trace = TraceRecord(agent_name="timed")
        trace.start()
        time.sleep(0.01)
        trace.end()
        assert trace.duration_ms >= 10

    def test_add_span(self):
        trace = TraceRecord(agent_name="test")
        span = SpanRecord(name="step1", span_type="tool_call")
        trace.add_span(span)
        assert len(trace.spans) == 1
        assert trace.spans[0].name == "step1"

    def test_computed_properties(self):
        trace = TraceRecord(agent_name="test")

        llm = SpanRecord(name="llm", span_type="llm_call")
        llm.set_llm_metadata(total_tokens=100)
        llm.start_time = 1.0
        llm.end_time = 1.5

        tool = SpanRecord(name="tool", span_type="tool_call")
        generic = SpanRecord(name="step", span_type="generic")

        trace.add_span(llm)
        trace.add_span(tool)
        trace.add_span(generic)

        assert len(trace.llm_calls) == 1
        assert len(trace.tool_calls) == 1
        assert trace.total_tokens == 100
        assert trace.total_llm_latency_ms == 500.0
        assert trace.step_count == 3

    def test_set_input_output(self):
        trace = TraceRecord(agent_name="test")
        trace.set_input({"query": "hello"})
        trace.set_output({"response": "hi"})
        assert trace.input_data == {"query": "hello"}
        assert trace.output_data == {"response": "hi"}

    def test_record_exception(self):
        trace = TraceRecord(agent_name="test")
        try:
            raise RuntimeError("boom")
        except RuntimeError as e:
            trace.record_exception(e)
        assert trace.status == "error"
        assert trace.exception_info["type"] == "RuntimeError"

    def test_tags_and_thread_id(self):
        trace = TraceRecord(
            agent_name="test",
            tags=["prod", "v2"],
            thread_id="thread-abc",
        )
        d = trace.to_dict()
        assert d["tags"] == ["prod", "v2"]
        assert d["thread_id"] == "thread-abc"

    def test_to_dict_summary(self):
        trace = TraceRecord(agent_name="test")
        llm = SpanRecord(name="llm", span_type="llm_call")
        llm.set_llm_metadata(total_tokens=200)
        trace.add_span(llm)
        trace.add_span(SpanRecord(name="tool", span_type="tool_call"))

        d = trace.to_dict()
        assert d["summary"]["step_count"] == 2
        assert d["summary"]["llm_calls"] == 1
        assert d["summary"]["tool_calls"] == 1
        assert d["summary"]["total_tokens"] == 200

    def test_to_dict_without_optional_fields(self):
        trace = TraceRecord(agent_name="test")
        d = trace.to_dict()
        assert "tags" not in d
        assert "thread_id" not in d
        assert "state_changes" not in d
        assert "decisions" not in d


class TestStateChange:
    def test_creation(self):
        sc = StateChange(key="balance", before=500, after=800, step_name="process-refund")
        assert sc.key == "balance"
        assert sc.before == 500
        assert sc.after == 800
        assert sc.timestamp is not None

    def test_to_dict(self):
        sc = StateChange(key="status", before="pending", after="completed")
        d = sc.to_dict()
        assert d["key"] == "status"
        assert d["before"] == "pending"
        assert d["after"] == "completed"

    def test_add_to_trace(self):
        trace = TraceRecord(agent_name="test")
        trace.add_state_change("balance", before=500, after=800, step_name="refund")
        assert len(trace.state_changes) == 1
        assert trace.state_changes[0].key == "balance"

    def test_in_to_dict(self):
        trace = TraceRecord(agent_name="test")
        trace.add_state_change("x", before=1, after=2)
        d = trace.to_dict()
        assert "state_changes" in d
        assert len(d["state_changes"]) == 1


class TestDecision:
    def test_creation(self):
        dec = Decision(
            chosen="escalate",
            alternatives=["resolve", "transfer"],
            reason="high value order",
            step_name="evaluate",
        )
        assert dec.chosen == "escalate"
        assert len(dec.alternatives) == 2

    def test_to_dict(self):
        dec = Decision(chosen="approve", alternatives=["deny"])
        d = dec.to_dict()
        assert d["chosen"] == "approve"
        assert d["alternatives"] == ["deny"]

    def test_add_to_trace(self):
        trace = TraceRecord(agent_name="test")
        trace.add_decision("escalate", alternatives=["resolve"], reason="legal threat")
        assert len(trace.decisions) == 1
        assert trace.decisions[0].chosen == "escalate"

    def test_in_to_dict(self):
        trace = TraceRecord(agent_name="test")
        trace.add_decision("proceed")
        d = trace.to_dict()
        assert "decisions" in d
        assert len(d["decisions"]) == 1
