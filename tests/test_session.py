"""Tests for agentprobe.session — batch trace management."""

import json
import os
import tempfile
import pytest
from agentprobe.models import SpanRecord, TraceRecord
from agentprobe.session import TraceSession


def _make_trace(name="test-agent", status="ok", tokens=100, steps=3):
    trace = TraceRecord(agent_name=name)
    trace.start_time = 1000.0
    trace.end_time = 1001.0
    trace.set_status(status)
    for i in range(steps):
        span = SpanRecord(name=f"step-{i}", span_type="generic")
        span.start_time = 1000.0 + i * 0.1
        span.end_time = 1000.0 + (i + 1) * 0.1
        span.set_status("ok")
        trace.add_span(span)
    # Add an LLM span with tokens
    llm = SpanRecord(name="llm", span_type="llm_call")
    llm.start_time = 1000.5
    llm.end_time = 1000.9
    llm.set_llm_metadata(model="test-model", total_tokens=tokens)
    llm.set_status("ok")
    trace.add_span(llm)
    return trace


class TestTraceSession:
    def test_add_trace(self):
        session = TraceSession(name="test-session")
        session.add_trace(_make_trace())
        assert len(session.traces) == 1

    def test_traces_returns_copy(self):
        session = TraceSession()
        traces = session.traces
        traces.append(_make_trace())
        assert len(session.traces) == 0

    def test_summary_empty(self):
        session = TraceSession(name="empty")
        s = session.summary()
        assert s["total_traces"] == 0

    def test_summary(self):
        session = TraceSession(name="test")
        session.add_trace(_make_trace(status="ok", tokens=100))
        session.add_trace(_make_trace(status="ok", tokens=200))
        session.add_trace(_make_trace(status="error", tokens=50))

        s = session.summary()
        assert s["total_traces"] == 3
        assert s["status_counts"]["ok"] == 2
        assert s["status_counts"]["error"] == 1
        assert s["pass_rate"] == pytest.approx(2 / 3)
        assert s["total_tokens"] == 350
        assert s["total_llm_calls"] == 3  # one per trace

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = TraceSession(name="test-session")
            session.metadata = {"version": "1.0"}

            t1 = _make_trace(name="agent-a", status="ok", tokens=100)
            t2 = _make_trace(name="agent-a", status="error", tokens=50)
            session.add_trace(t1)
            session.add_trace(t2)

            session_dir = session.save(tmpdir)

            # Verify files exist
            assert os.path.exists(os.path.join(session_dir, "_session.json"))
            files = [f for f in os.listdir(session_dir) if f != "_session.json"]
            assert len(files) == 2

            # Load and verify
            loaded = TraceSession.load(session_dir)
            assert loaded.session_id == session.session_id
            assert loaded.name == "test-session"
            assert loaded.metadata == {"version": "1.0"}
            assert len(loaded.traces) == 2
            assert loaded.traces[0].agent_name == "agent-a"

    def test_save_load_preserves_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = TraceSession(name="round-trip")
            trace = _make_trace(tokens=200)
            trace.tags = ["prod"]
            trace.thread_id = "thread-1"
            trace.add_state_change("x", before=1, after=2, step_name="step")
            trace.add_decision("go", alternatives=["stop"])
            session.add_trace(trace)

            session_dir = session.save(tmpdir)
            loaded = TraceSession.load(session_dir)

            lt = loaded.traces[0]
            assert lt.tags == ["prod"]
            assert lt.thread_id == "thread-1"
            assert lt.total_tokens == 200
            assert lt.step_count == trace.step_count

    def test_save_load_preserves_llm_tool_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = TraceSession()
            trace = TraceRecord(agent_name="test")
            trace.start_time = 1.0
            trace.end_time = 2.0
            trace.set_status("ok")

            llm = SpanRecord(name="gen", span_type="llm_call")
            llm.set_llm_metadata(model="claude-sonnet-4-20250514", prompt_tokens=50,
                                 completion_tokens=100, total_tokens=150, temperature=0.5)
            llm.set_status("ok")
            trace.add_span(llm)

            tool = SpanRecord(name="search", span_type="tool_call")
            tool.set_tool_metadata(tool_name="order-lookup",
                                   tool_args={"id": "123"},
                                   tool_result={"found": True},
                                   tool_success=True)
            tool.set_status("ok")
            trace.add_span(tool)

            session.add_trace(trace)
            session_dir = session.save(tmpdir)
            loaded = TraceSession.load(session_dir)

            lt = loaded.traces[0]
            llm_span = lt.llm_calls[0]
            assert llm_span.model == "claude-sonnet-4-20250514"
            assert llm_span.prompt_tokens == 50
            assert llm_span.completion_tokens == 100
            assert llm_span.temperature == 0.5

            tool_span = lt.tool_calls[0]
            assert tool_span.tool_name == "order-lookup"
            assert tool_span.tool_args == {"id": "123"}
            assert tool_span.tool_success is True

    def test_session_index_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = TraceSession(name="index-test")
            session.add_trace(_make_trace())
            session_dir = session.save(tmpdir)

            with open(os.path.join(session_dir, "_session.json")) as f:
                index = json.load(f)

            assert "session_id" in index
            assert "traces" in index
            assert "summary" in index
            assert len(index["traces"]) == 1
            assert "filename" in index["traces"][0]
            assert "trace_id" in index["traces"][0]
