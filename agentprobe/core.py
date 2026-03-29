"""
Core instrumentation for AI agent tracing.

Provides decorators and context managers to capture full execution traces
from any AI agent — every LLM call, tool invocation, reasoning step,
state change, and decision point.

Usage:
    from agentprobe import AgentProbe, trace_agent, trace_step, trace_tool

    probe = AgentProbe(exporters=[JSONFileExporter("traces/")])
    probe.init()

    @trace_agent("my-agent")
    def run_agent(query: str) -> str:
        data = lookup(query)
        return generate_response(data)
"""

from __future__ import annotations

import asyncio
import functools
import json
from contextvars import ContextVar
from typing import Any, Callable, Dict, List, Optional, TypeVar

from agentprobe.models import SpanRecord, TraceRecord

F = TypeVar("F", bound=Callable[..., Any])

# Context variables for async-safe trace/span nesting
_current_trace: ContextVar[Optional[TraceRecord]] = ContextVar("_current_trace", default=None)
_current_span: ContextVar[Optional[SpanRecord]] = ContextVar("_current_span", default=None)


def get_current_trace() -> Optional[TraceRecord]:
    """Get the currently active trace, if any."""
    return _current_trace.get()


def get_current_span() -> Optional[SpanRecord]:
    """Get the currently active span, if any."""
    return _current_span.get()


def _serialize_value(v: Any) -> Any:
    """Safely serialize arbitrary Python objects for JSON storage."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_serialize_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _serialize_value(val) for k, val in v.items()}
    # Fallback: stringify anything we don't recognize
    try:
        json.dumps(v)
        return v
    except (TypeError, ValueError):
        return str(v)


# ============================================================================
# AgentProbe — the configuration hub
# ============================================================================

class AgentProbe:
    """
    Central configuration for trace capture.

    Create once at startup, register exporters, call init(). Decorators
    find this instance automatically via the singleton pattern.
    """

    _instance: Optional[AgentProbe] = None

    def __init__(self, exporters: Optional[List[Any]] = None):
        self._exporters = exporters or []
        self._traces: List[TraceRecord] = []

    def init(self) -> AgentProbe:
        """Register this instance as the global singleton."""
        AgentProbe._instance = self
        return self

    @classmethod
    def get_instance(cls) -> Optional[AgentProbe]:
        return cls._instance

    def record_trace(self, trace: TraceRecord) -> None:
        """Store a completed trace and send to all exporters."""
        self._traces.append(trace)
        for exporter in self._exporters:
            try:
                exporter.export(trace)
            except Exception as e:
                print(f"[agentprobe] Exporter {type(exporter).__name__} failed: {e}")

    @property
    def traces(self) -> List[TraceRecord]:
        """Return a copy of all recorded traces."""
        return list(self._traces)

    def clear(self) -> None:
        """Clear all recorded traces."""
        self._traces.clear()


# ============================================================================
# State and decision tracking — opt-in enrichments
# ============================================================================

def record_state_change(key: str, before: Any = None, after: Any = None) -> None:
    """
    Record a state mutation on the current trace.

    Call this inside a traced function to capture what changed in the
    agent's working memory or external state.

    Args:
        key: Name of the state being changed (e.g., "customer_balance").
        before: Value before the change.
        after: Value after the change.
    """
    trace = _current_trace.get()
    if trace is None:
        return
    span = _current_span.get()
    step_name = span.name if span else ""
    trace.add_state_change(
        key=key,
        before=_serialize_value(before),
        after=_serialize_value(after),
        step_name=step_name,
    )


def record_decision(chosen: str, alternatives: Optional[List[str]] = None,
                    reason: str = "") -> None:
    """
    Record a branching decision on the current trace.

    Call this inside a traced function to capture what the agent chose
    and what alternatives it considered.

    Args:
        chosen: The option that was selected.
        alternatives: Other options that were available.
        reason: Why this option was chosen.
    """
    trace = _current_trace.get()
    if trace is None:
        return
    span = _current_span.get()
    step_name = span.name if span else ""
    trace.add_decision(
        chosen=chosen,
        alternatives=alternatives,
        reason=reason,
        step_name=step_name,
    )


# ============================================================================
# Decorators
# ============================================================================

def trace_agent(
    name: str,
    tags: Optional[List[str]] = None,
    thread_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """
    Decorator that creates a new trace for each invocation of the agent.

    This is the top-level decorator — use it on your agent's entry point.
    All @trace_step, @trace_tool, and @trace_llm_call calls inside
    will be captured as spans within this trace.

    Example:
        @trace_agent("my-support-agent")
        def handle_customer(message: str) -> dict:
            ...
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            trace = TraceRecord(agent_name=name)
            if tags:
                trace.tags = list(tags)
            if thread_id:
                trace.thread_id = thread_id
            if metadata:
                trace.metadata = dict(metadata)

            trace.set_input(_serialize_value(args[0] if len(args) == 1 else args))
            trace.start()

            token = _current_trace.set(trace)
            try:
                result = fn(*args, **kwargs)
                trace.set_output(_serialize_value(result))
                trace.set_status("ok")
                return result
            except Exception as e:
                trace.record_exception(e)
                raise
            finally:
                trace.end()
                _current_trace.reset(token)
                probe = AgentProbe.get_instance()
                if probe:
                    probe.record_trace(trace)

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            trace = TraceRecord(agent_name=name)
            if tags:
                trace.tags = list(tags)
            if thread_id:
                trace.thread_id = thread_id
            if metadata:
                trace.metadata = dict(metadata)

            trace.set_input(_serialize_value(args[0] if len(args) == 1 else args))
            trace.start()

            token = _current_trace.set(trace)
            try:
                result = await fn(*args, **kwargs)
                trace.set_output(_serialize_value(result))
                trace.set_status("ok")
                return result
            except Exception as e:
                trace.record_exception(e)
                raise
            finally:
                trace.end()
                _current_trace.reset(token)
                probe = AgentProbe.get_instance()
                if probe:
                    probe.record_trace(trace)

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def trace_step(
    name: str,
    step_type: str = "generic",
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """
    Decorator that captures one step within an agent trace.

    Supports nesting — if step A calls step B, B becomes a child of A.

    Example:
        @trace_step("evaluate-return", step_type="reasoning")
        def evaluate_return(order: dict) -> dict:
            ...
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            trace = _current_trace.get()
            if trace is None:
                # No active trace — run untraced
                return fn(*args, **kwargs)

            span = SpanRecord(name=name, span_type=step_type)
            if tags:
                span.tags = list(tags)
            if metadata:
                span.metadata = dict(metadata)

            parent = _current_span.get()
            if parent:
                span.parent_span_id = parent.span_id

            span.input_data = _serialize_value(args[0] if len(args) == 1 else args)
            span.start()

            span_token = _current_span.set(span)
            try:
                result = fn(*args, **kwargs)
                span.set_output(_serialize_value(result))
                span.set_status("ok")
                return result
            except Exception as e:
                span.record_exception(e)
                raise
            finally:
                span.end()
                _current_span.reset(span_token)
                trace.add_span(span)

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            trace = _current_trace.get()
            if trace is None:
                return await fn(*args, **kwargs)

            span = SpanRecord(name=name, span_type=step_type)
            if tags:
                span.tags = list(tags)
            if metadata:
                span.metadata = dict(metadata)

            parent = _current_span.get()
            if parent:
                span.parent_span_id = parent.span_id

            span.input_data = _serialize_value(args[0] if len(args) == 1 else args)
            span.start()

            span_token = _current_span.set(span)
            try:
                result = await fn(*args, **kwargs)
                span.set_output(_serialize_value(result))
                span.set_status("ok")
                return result
            except Exception as e:
                span.record_exception(e)
                raise
            finally:
                span.end()
                _current_span.reset(span_token)
                trace.add_span(span)

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def trace_tool(
    name: str,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """
    Convenience decorator for tool calls. Sets span_type='tool_call'.

    Example:
        @trace_tool("order-lookup")
        def lookup_order(order_id: str) -> dict:
            ...
    """
    return trace_step(name, step_type="tool_call", tags=tags, metadata=metadata)


def trace_llm_call(
    name: str = "llm_call",
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Callable[[F], F]:
    """
    Convenience decorator for LLM calls. Sets span_type='llm_call'.

    Example:
        @trace_llm_call("generate-response")
        def generate_response(prompt: str) -> str:
            ...
    """
    return trace_step(name, step_type="llm_call", tags=tags, metadata=metadata)


# ============================================================================
# Context manager for inline tracing
# ============================================================================

from contextlib import contextmanager


@contextmanager
def step_span(
    name: str,
    step_type: str = "generic",
    input_data: Any = None,
    tags: Optional[List[str]] = None,
):
    """
    Context manager for tracing inline code without a separate function.

    Example:
        with step_span("classify-anomaly", step_type="reasoning",
                        input_data=telemetry) as span:
            anomaly_type = "battery_low" if voltage < 27.5 else "unknown"
            span.set_output({"anomaly_type": anomaly_type})
    """
    trace = _current_trace.get()
    if trace is None:
        # No active trace — yield a dummy span
        yield SpanRecord(name=name, span_type=step_type)
        return

    span = SpanRecord(name=name, span_type=step_type)
    if tags:
        span.tags = list(tags)
    if input_data is not None:
        span.input_data = _serialize_value(input_data)

    parent = _current_span.get()
    if parent:
        span.parent_span_id = parent.span_id

    span.start()
    span_token = _current_span.set(span)
    try:
        yield span
        if span.status is None:
            span.set_status("ok")
    except Exception as e:
        span.record_exception(e)
        raise
    finally:
        span.end()
        _current_span.reset(span_token)
        trace.add_span(span)
