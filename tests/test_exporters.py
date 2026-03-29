"""Tests for exporters — Console, JSON file, OTel."""

import json
import os
import tempfile
import pytest
from agentprobe.models import SpanRecord, TraceRecord
from agentprobe.exporters.console import ConsoleExporter
from agentprobe.exporters.json_file import JSONFileExporter
from agentprobe.exporters.otel import OTelExporter


def _make_trace():
    """Create a sample trace for testing."""
    trace = TraceRecord(agent_name="test-agent")
    trace.start_time = 1000.0
    trace.end_time = 1001.5
    trace.set_input({"query": "hello"})
    trace.set_output({"response": "hi"})
    trace.set_status("ok")

    llm = SpanRecord(name="generate", span_type="llm_call")
    llm.start_time = 1000.1
    llm.end_time = 1000.9
    llm.set_llm_metadata(model="claude-sonnet-4-20250514", total_tokens=150)
    llm.set_status("ok")

    tool = SpanRecord(name="lookup", span_type="tool_call")
    tool.start_time = 1000.0
    tool.end_time = 1000.1
    tool.set_tool_metadata(tool_name="order-lookup", tool_success=True)
    tool.set_status("ok")

    trace.add_span(tool)
    trace.add_span(llm)
    trace.add_state_change("balance", before=500, after=800, step_name="refund")
    trace.add_decision("approve", alternatives=["deny"], step_name="evaluate")

    return trace


class TestConsoleExporter:
    def test_export_ok(self, capsys):
        exporter = ConsoleExporter()
        trace = _make_trace()
        exporter.export(trace)
        output = capsys.readouterr().out
        assert "test-agent" in output
        assert "\u2713" in output  # checkmark

    def test_export_error(self, capsys):
        exporter = ConsoleExporter()
        trace = TraceRecord(agent_name="bad")
        trace.set_status("error")
        trace.exception_info = {"type": "ValueError", "message": "oops"}
        trace.start_time = 1.0
        trace.end_time = 2.0
        exporter.export(trace)
        output = capsys.readouterr().out
        assert "ERROR" in output
        assert "ValueError" in output

    def test_verbose_mode(self, capsys):
        exporter = ConsoleExporter(verbose=True)
        trace = _make_trace()
        exporter.export(trace)
        output = capsys.readouterr().out
        assert "llm_call" in output
        assert "tool_call" in output
        assert "claude" in output
        assert "State changes" in output
        assert "Decisions" in output


class TestJSONFileExporter:
    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = JSONFileExporter(output_dir=tmpdir)
            trace = _make_trace()
            filepath = exporter.export(trace)

            assert os.path.exists(filepath)
            assert filepath.endswith(".json")

    def test_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = JSONFileExporter(output_dir=tmpdir)
            trace = _make_trace()
            filepath = exporter.export(trace)

            with open(filepath) as f:
                data = json.load(f)

            assert data["agent_name"] == "test-agent"
            assert data["status"] == "ok"
            assert len(data["spans"]) == 2
            assert data["summary"]["total_tokens"] == 150
            assert len(data["state_changes"]) == 1
            assert len(data["decisions"]) == 1

    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = os.path.join(tmpdir, "new_subdir")
            exporter = JSONFileExporter(output_dir=outdir)
            trace = _make_trace()
            exporter.export(trace)
            assert os.path.isdir(outdir)

    def test_filename_pattern(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = JSONFileExporter(output_dir=tmpdir)
            trace = _make_trace()
            filepath = exporter.export(trace)
            filename = os.path.basename(filepath)
            assert filename.startswith("test-agent_")
            assert filename.endswith(".json")


class TestOTelExporter:
    def test_export_runs(self):
        """OTel export doesn't crash."""
        exporter = OTelExporter(service_name="test")
        trace = _make_trace()
        exporter.export(trace)  # Should not raise
        exporter.shutdown()

    def test_export_with_error_trace(self):
        exporter = OTelExporter(service_name="test")
        trace = TraceRecord(agent_name="bad")
        trace.set_status("error", "something failed")
        trace.start_time = 1.0
        trace.end_time = 2.0
        exporter.export(trace)
        exporter.shutdown()
