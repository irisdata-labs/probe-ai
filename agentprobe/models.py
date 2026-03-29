"""
Data models for agent execution traces.

SpanRecord — one step in an agent's execution (tool call, LLM call, reasoning step).
TraceRecord — a complete execution trace containing multiple spans.
StateChange — records a mutation to the agent's working memory.
Decision — records a branching decision the agent made.
"""

from __future__ import annotations

import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StateChange:
    """Records a mutation to the agent's working memory or external state."""
    key: str
    before: Any = None
    after: Any = None
    step_name: str = ""
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "before": self.before,
            "after": self.after,
            "step_name": self.step_name,
            "timestamp": self.timestamp,
        }


@dataclass
class Decision:
    """Records a branching decision the agent made."""
    chosen: str
    alternatives: List[str] = field(default_factory=list)
    reason: str = ""
    step_name: str = ""
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chosen": self.chosen,
            "alternatives": self.alternatives,
            "reason": self.reason,
            "step_name": self.step_name,
            "timestamp": self.timestamp,
        }


@dataclass
class SpanRecord:
    """
    One step in an agent's execution pipeline.

    Every decorated function call becomes one SpanRecord. Spans form a tree
    via parent_span_id — if step A calls step B, B's parent is A.
    """

    # Identity
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    name: str = ""
    span_type: str = "generic"  # generic, llm_call, tool_call, reasoning, retrieval
    parent_span_id: Optional[str] = None

    # Timing
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # Data
    input_data: Any = None
    output_data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Status
    status: Optional[str] = None  # ok, error, warning
    status_message: Optional[str] = None
    exception_info: Optional[Dict[str, str]] = None

    # LLM-specific (populated via set_llm_metadata)
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    temperature: Optional[float] = None

    # Tool-specific (populated via set_tool_metadata)
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Any = None
    tool_success: Optional[bool] = None

    def start(self) -> None:
        self.start_time = time.time()

    def end(self) -> None:
        self.end_time = time.time()

    @property
    def duration_ms(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None

    def set_output(self, output: Any) -> None:
        self.output_data = output

    def set_status(self, status: str, message: Optional[str] = None) -> None:
        self.status = status
        self.status_message = message

    def record_exception(self, exc: Exception) -> None:
        self.status = "error"
        self.exception_info = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    def set_llm_metadata(
        self,
        model: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        if model is not None:
            self.model = model
        if prompt_tokens is not None:
            self.prompt_tokens = prompt_tokens
        if completion_tokens is not None:
            self.completion_tokens = completion_tokens
        if total_tokens is not None:
            self.total_tokens = total_tokens
        if temperature is not None:
            self.temperature = temperature

    def set_tool_metadata(
        self,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        tool_result: Any = None,
        tool_success: Optional[bool] = None,
    ) -> None:
        if tool_name is not None:
            self.tool_name = tool_name
        if tool_args is not None:
            self.tool_args = tool_args
        if tool_result is not None:
            self.tool_result = tool_result
        if tool_success is not None:
            self.tool_success = tool_success

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "span_id": self.span_id,
            "name": self.name,
            "span_type": self.span_type,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "metadata": self.metadata,
            "status": self.status,
            "status_message": self.status_message,
            "exception_info": self.exception_info,
        }
        if self.tags:
            d["tags"] = self.tags
        # LLM fields — only include if populated
        if self.model is not None:
            d["llm"] = {
                "model": self.model,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "temperature": self.temperature,
            }
        # Tool fields — only include if populated
        if self.tool_name is not None:
            d["tool"] = {
                "tool_name": self.tool_name,
                "tool_args": self.tool_args,
                "tool_result": self.tool_result,
                "tool_success": self.tool_success,
            }
        return d


@dataclass
class TraceRecord:
    """
    A complete execution trace for one agent run.

    Contains all spans (steps), plus optional state changes and decision
    records for richer analysis.
    """

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    agent_name: str = ""

    # Timing
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # Data
    input_data: Any = None
    output_data: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    thread_id: Optional[str] = None

    # Status
    status: Optional[str] = None
    status_message: Optional[str] = None
    exception_info: Optional[Dict[str, str]] = None

    # Spans (execution steps)
    spans: List[SpanRecord] = field(default_factory=list)

    # State tracking (opt-in enrichments)
    state_changes: List[StateChange] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)

    def start(self) -> None:
        self.start_time = time.time()

    def end(self) -> None:
        self.end_time = time.time()

    @property
    def duration_ms(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None

    def set_input(self, input_data: Any) -> None:
        self.input_data = input_data

    def set_output(self, output_data: Any) -> None:
        self.output_data = output_data

    def set_status(self, status: str, message: Optional[str] = None) -> None:
        self.status = status
        self.status_message = message

    def record_exception(self, exc: Exception) -> None:
        self.status = "error"
        self.exception_info = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }

    def add_span(self, span: SpanRecord) -> None:
        self.spans.append(span)

    def add_state_change(self, key: str, before: Any = None, after: Any = None,
                         step_name: str = "") -> None:
        self.state_changes.append(StateChange(
            key=key, before=before, after=after, step_name=step_name,
        ))

    def add_decision(self, chosen: str, alternatives: Optional[List[str]] = None,
                     reason: str = "", step_name: str = "") -> None:
        self.decisions.append(Decision(
            chosen=chosen, alternatives=alternatives or [], reason=reason,
            step_name=step_name,
        ))

    # Computed properties

    @property
    def llm_calls(self) -> List[SpanRecord]:
        return [s for s in self.spans if s.span_type == "llm_call"]

    @property
    def tool_calls(self) -> List[SpanRecord]:
        return [s for s in self.spans if s.span_type == "tool_call"]

    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens or 0 for s in self.spans)

    @property
    def total_llm_latency_ms(self) -> float:
        return sum(s.duration_ms or 0 for s in self.llm_calls)

    @property
    def step_count(self) -> int:
        return len(self.spans)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "trace_id": self.trace_id,
            "agent_name": self.agent_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "metadata": self.metadata,
            "status": self.status,
            "status_message": self.status_message,
            "exception_info": self.exception_info,
            "spans": [s.to_dict() for s in self.spans],
            "summary": {
                "step_count": self.step_count,
                "llm_calls": len(self.llm_calls),
                "tool_calls": len(self.tool_calls),
                "total_tokens": self.total_tokens,
                "total_llm_latency_ms": self.total_llm_latency_ms,
            },
        }
        if self.tags:
            d["tags"] = self.tags
        if self.thread_id:
            d["thread_id"] = self.thread_id
        if self.state_changes:
            d["state_changes"] = [sc.to_dict() for sc in self.state_changes]
        if self.decisions:
            d["decisions"] = [dec.to_dict() for dec in self.decisions]
        return d
