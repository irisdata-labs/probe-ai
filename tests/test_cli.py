"""Tests for agentprobe CLI."""

import json
import os
import tempfile
import pytest
from agentprobe.cli import cmd_view, cmd_list, cmd_summary


def _make_trace_file(directory, agent_name="test-agent", status="ok"):
    """Create a trace JSON file for testing."""
    data = {
        "trace_id": "abc123def456",
        "agent_name": agent_name,
        "start_time": 1000.0,
        "end_time": 1001.5,
        "duration_ms": 1500.0,
        "input_data": {"query": "test"},
        "output_data": {"response": "ok"},
        "status": status,
        "spans": [
            {
                "span_id": "span001",
                "name": "lookup",
                "span_type": "tool_call",
                "duration_ms": 100.0,
                "status": "ok",
                "tool": {"tool_name": "order-lookup", "tool_success": True},
            },
            {
                "span_id": "span002",
                "name": "generate",
                "span_type": "llm_call",
                "duration_ms": 800.0,
                "status": "ok",
                "llm": {"model": "claude-sonnet-4-20250514", "total_tokens": 150},
            },
        ],
        "summary": {
            "step_count": 2,
            "llm_calls": 1,
            "tool_calls": 1,
            "total_tokens": 150,
            "total_llm_latency_ms": 800.0,
        },
        "state_changes": [
            {"key": "balance", "before": 500, "after": 800, "step_name": "refund"},
        ],
        "decisions": [
            {"chosen": "approve", "alternatives": ["deny"], "step_name": "evaluate"},
        ],
    }
    if status == "error":
        data["exception_info"] = {"type": "ValueError", "message": "bad input", "traceback": "..."}

    filename = f"{agent_name}_{data['trace_id'][:8]}.json"
    filepath = os.path.join(directory, filename)
    with open(filepath, "w") as f:
        json.dump(data, f)
    return filepath


def _make_session(directory):
    """Create a session directory for testing."""
    session_dir = os.path.join(directory, "session_test123")
    os.makedirs(session_dir)

    index = {
        "session_id": "test123",
        "name": "test-session",
        "traces": [
            {"filename": "trace1.json", "trace_id": "t1", "agent_name": "agent",
             "status": "ok", "duration_ms": 1000},
            {"filename": "trace2.json", "trace_id": "t2", "agent_name": "agent",
             "status": "error", "duration_ms": 500},
        ],
        "summary": {
            "total_traces": 2,
            "status_counts": {"ok": 1, "error": 1},
            "pass_rate": 0.5,
            "avg_duration_ms": 750,
            "avg_steps": 3,
            "total_tokens": 300,
            "total_llm_calls": 2,
            "total_tool_calls": 2,
        },
    }
    with open(os.path.join(session_dir, "_session.json"), "w") as f:
        json.dump(index, f)
    return session_dir


class _Args:
    """Minimal args namespace for CLI testing."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestCLIView:
    def test_view_trace(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = _make_trace_file(tmpdir)
            cmd_view(_Args(file=filepath, verbose=False))
            output = capsys.readouterr().out
            assert "test-agent" in output
            assert "ok" in output
            assert "lookup" in output
            assert "generate" in output
            assert "State" in output
            assert "Decision" in output

    def test_view_verbose(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = _make_trace_file(tmpdir)
            cmd_view(_Args(file=filepath, verbose=True))
            output = capsys.readouterr().out
            assert "trace_id" in output  # raw JSON shown

    def test_view_error_trace(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = _make_trace_file(tmpdir, status="error")
            cmd_view(_Args(file=filepath, verbose=False))
            output = capsys.readouterr().out
            assert "ERROR" in output
            assert "ValueError" in output


class TestCLIList:
    def test_list_traces(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_trace_file(tmpdir, agent_name="agent-a")
            _make_trace_file(tmpdir, agent_name="agent-b")
            cmd_list(_Args(directory=tmpdir))
            output = capsys.readouterr().out
            assert "agent-a" in output
            assert "agent-b" in output

    def test_list_empty(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd_list(_Args(directory=tmpdir))
            output = capsys.readouterr().out
            assert "No trace files" in output


class TestCLISummary:
    def test_summary(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = _make_session(tmpdir)
            cmd_summary(_Args(session_dir=session_dir))
            output = capsys.readouterr().out
            assert "test-session" in output
            assert "50%" in output
            assert "300" in output  # total tokens
