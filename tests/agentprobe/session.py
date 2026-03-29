"""
TraceSession — groups multiple traces for batch analysis.

Create a session, run your agent many times, collect the traces,
and get aggregate statistics. Sessions can be saved to disk and
loaded later for comparison.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agentprobe.models import SpanRecord, TraceRecord


@dataclass
class TraceSession:
    """Groups multiple traces for batch analysis."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    _traces: List[TraceRecord] = field(default_factory=list)

    def add_trace(self, trace: TraceRecord) -> None:
        """Add a completed trace to the session."""
        self._traces.append(trace)

    @property
    def traces(self) -> List[TraceRecord]:
        return list(self._traces)

    def summary(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all traces."""
        total = len(self._traces)
        if total == 0:
            return {
                "session_id": self.session_id,
                "name": self.name,
                "total_traces": 0,
            }

        ok = sum(1 for t in self._traces if t.status == "ok")
        error = sum(1 for t in self._traces if t.status == "error")
        durations = [t.duration_ms for t in self._traces if t.duration_ms]
        steps = [t.step_count for t in self._traces]
        tokens = [t.total_tokens for t in self._traces]
        llm_calls = sum(len(t.llm_calls) for t in self._traces)
        tool_calls = sum(len(t.tool_calls) for t in self._traces)

        return {
            "session_id": self.session_id,
            "name": self.name,
            "total_traces": total,
            "status_counts": {"ok": ok, "error": error},
            "pass_rate": ok / total if total else 0,
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "avg_steps": sum(steps) / len(steps) if steps else 0,
            "total_tokens": sum(tokens),
            "avg_tokens": sum(tokens) / total if total else 0,
            "total_llm_calls": llm_calls,
            "total_tool_calls": tool_calls,
        }

    def save(self, output_dir: str) -> str:
        """
        Save the session to a directory.

        Creates a directory with one JSON file per trace plus
        a _session.json index file with metadata and summary.
        """
        session_dir = os.path.join(output_dir, f"session_{self.session_id}")
        os.makedirs(session_dir, exist_ok=True)

        # Write each trace
        trace_files = []
        for trace in self._traces:
            filename = f"{trace.agent_name}_{trace.trace_id[:8]}.json"
            filepath = os.path.join(session_dir, filename)
            with open(filepath, "w") as f:
                json.dump(trace.to_dict(), f, indent=2, default=str)
            trace_files.append({
                "filename": filename,
                "trace_id": trace.trace_id,
                "agent_name": trace.agent_name,
                "status": trace.status,
                "duration_ms": trace.duration_ms,
            })

        # Write session index
        index = {
            "session_id": self.session_id,
            "name": self.name,
            "metadata": self.metadata,
            "traces": trace_files,
            "summary": self.summary(),
        }
        index_path = os.path.join(session_dir, "_session.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2, default=str)

        return session_dir

    @classmethod
    def load(cls, session_dir: str) -> TraceSession:
        """Reconstruct a session from a saved directory."""
        index_path = os.path.join(session_dir, "_session.json")
        with open(index_path) as f:
            index = json.load(f)

        session = cls(
            session_id=index["session_id"],
            name=index.get("name", ""),
            metadata=index.get("metadata", {}),
        )

        for trace_info in index.get("traces", []):
            trace_path = os.path.join(session_dir, trace_info["filename"])
            if os.path.exists(trace_path):
                with open(trace_path) as f:
                    data = json.load(f)
                trace = _dict_to_trace(data)
                session.add_trace(trace)

        return session


def _dict_to_trace(data: Dict[str, Any]) -> TraceRecord:
    """Reconstruct a TraceRecord from a JSON dict."""
    trace = TraceRecord(
        trace_id=data.get("trace_id", ""),
        agent_name=data.get("agent_name", ""),
        start_time=data.get("start_time"),
        end_time=data.get("end_time"),
        input_data=data.get("input_data"),
        output_data=data.get("output_data"),
        metadata=data.get("metadata", {}),
        tags=data.get("tags", []),
        thread_id=data.get("thread_id"),
        status=data.get("status"),
        status_message=data.get("status_message"),
        exception_info=data.get("exception_info"),
    )

    for span_data in data.get("spans", []):
        span = SpanRecord(
            span_id=span_data.get("span_id", ""),
            name=span_data.get("name", ""),
            span_type=span_data.get("span_type", "generic"),
            parent_span_id=span_data.get("parent_span_id"),
            start_time=span_data.get("start_time"),
            end_time=span_data.get("end_time"),
            input_data=span_data.get("input_data"),
            output_data=span_data.get("output_data"),
            metadata=span_data.get("metadata", {}),
            tags=span_data.get("tags", []),
            status=span_data.get("status"),
            status_message=span_data.get("status_message"),
            exception_info=span_data.get("exception_info"),
        )
        # Restore LLM metadata
        llm = span_data.get("llm", {})
        if llm:
            span.model = llm.get("model")
            span.prompt_tokens = llm.get("prompt_tokens")
            span.completion_tokens = llm.get("completion_tokens")
            span.total_tokens = llm.get("total_tokens")
            span.temperature = llm.get("temperature")
        # Restore tool metadata
        tool = span_data.get("tool", {})
        if tool:
            span.tool_name = tool.get("tool_name")
            span.tool_args = tool.get("tool_args")
            span.tool_result = tool.get("tool_result")
            span.tool_success = tool.get("tool_success")

        trace.add_span(span)

    return trace
