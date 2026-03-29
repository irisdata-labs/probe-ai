"""
LangGraph automatic instrumentation.

Wraps a compiled LangGraph app so all nodes are traced automatically.
Also captures state changes between node executions.

Usage:
    from agentprobe.integrations.langgraph import instrument_langgraph

    app = instrument_langgraph(existing_app)
    result = app.invoke({"messages": [...]})
"""

from __future__ import annotations

import functools
import time
from contextvars import ContextVar
from typing import Any, Dict, Optional

from agentprobe.core import (
    AgentProbe,
    _current_span,
    _current_trace,
    _serialize_value,
)
from agentprobe.models import SpanRecord, TraceRecord

# Track original node functions for unpatch
_original_funcs: ContextVar[Dict[str, Any]] = ContextVar("_original_funcs", default={})


def instrument_langgraph(
    compiled_graph: Any,
    agent_name: Optional[str] = None,
    capture_state: bool = True,
) -> Any:
    """
    Instrument a compiled LangGraph app for automatic tracing.

    All graph nodes will be traced as spans. State changes between
    nodes are captured automatically if capture_state=True.

    Args:
        compiled_graph: A compiled LangGraph StateGraph.
        agent_name: Name for the trace. Defaults to the graph's name.
        capture_state: Whether to capture LangGraph state diffs between nodes.

    Returns:
        The same graph, with invoke/ainvoke patched for tracing.
    """
    name = agent_name or getattr(compiled_graph, "name", "langgraph-agent")

    # Save originals
    original_invoke = compiled_graph.invoke
    original_ainvoke = getattr(compiled_graph, "ainvoke", None)

    @functools.wraps(original_invoke)
    def instrumented_invoke(input_data: Any, config: Any = None, **kwargs: Any) -> Any:
        trace = TraceRecord(agent_name=name)
        trace.set_input(_serialize_value(input_data))
        trace.start()

        token = _current_trace.set(trace)
        try:
            _patch_nodes(compiled_graph, trace, capture_state)
            result = original_invoke(input_data, config=config, **kwargs)
            trace.set_output(_serialize_value(result))
            trace.set_status("ok")
            return result
        except Exception as e:
            trace.record_exception(e)
            raise
        finally:
            trace.end()
            _unpatch_nodes(compiled_graph)
            _current_trace.reset(token)
            probe = AgentProbe.get_instance()
            if probe:
                probe.record_trace(trace)

    compiled_graph.invoke = instrumented_invoke

    if original_ainvoke:
        @functools.wraps(original_ainvoke)
        async def instrumented_ainvoke(input_data: Any, config: Any = None, **kwargs: Any) -> Any:
            trace = TraceRecord(agent_name=name)
            trace.set_input(_serialize_value(input_data))
            trace.start()

            token = _current_trace.set(trace)
            try:
                _patch_nodes(compiled_graph, trace, capture_state)
                result = await original_ainvoke(input_data, config=config, **kwargs)
                trace.set_output(_serialize_value(result))
                trace.set_status("ok")
                return result
            except Exception as e:
                trace.record_exception(e)
                raise
            finally:
                trace.end()
                _unpatch_nodes(compiled_graph)
                _current_trace.reset(token)
                probe = AgentProbe.get_instance()
                if probe:
                    probe.record_trace(trace)

        compiled_graph.ainvoke = instrumented_ainvoke

    return compiled_graph


def _patch_nodes(compiled_graph: Any, trace: TraceRecord,
                 capture_state: bool = True) -> None:
    """Replace each node's function with an instrumented wrapper."""
    originals = {}

    # LangGraph stores nodes as {name: PregelNode}
    nodes = getattr(compiled_graph, "nodes", None)
    if nodes is None:
        return

    for node_name, node in nodes.items():
        bound = getattr(node, "bound", None)
        if bound is None:
            continue

        func = getattr(bound, "func", None)
        if func is None:
            continue

        originals[node_name] = func

        def make_wrapper(name: str, orig: Any) -> Any:
            @functools.wraps(orig)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                span = SpanRecord(
                    name=name,
                    span_type=_infer_span_type(name),
                )

                parent = _current_span.get()
                if parent:
                    span.parent_span_id = parent.span_id

                span.input_data = _serialize_value(
                    args[0] if len(args) == 1 else args
                )
                span.start()

                span_token = _current_span.set(span)
                try:
                    result = orig(*args, **kwargs)
                    span.set_output(_serialize_value(result))
                    span.set_status("ok")

                    # Capture state changes if result is a dict (LangGraph state)
                    if capture_state and isinstance(result, dict):
                        input_state = args[0] if args and isinstance(args[0], dict) else {}
                        for key in result:
                            old_val = input_state.get(key)
                            new_val = result[key]
                            if old_val != new_val:
                                trace.add_state_change(
                                    key=key,
                                    before=_serialize_value(old_val),
                                    after=_serialize_value(new_val),
                                    step_name=name,
                                )

                    return result
                except Exception as e:
                    span.record_exception(e)
                    raise
                finally:
                    span.end()
                    _current_span.reset(span_token)
                    trace.add_span(span)

            return wrapper

        bound.func = make_wrapper(node_name, func)

    _original_funcs.set(originals)


def _unpatch_nodes(compiled_graph: Any) -> None:
    """Restore original node functions."""
    originals = _original_funcs.get()
    if not originals:
        return

    nodes = getattr(compiled_graph, "nodes", None)
    if nodes is None:
        return

    for node_name, original_func in originals.items():
        node = nodes.get(node_name)
        if node:
            bound = getattr(node, "bound", None)
            if bound:
                bound.func = original_func

    _original_funcs.set({})


def _infer_span_type(node_name: str) -> str:
    """Heuristic: guess span type from node name."""
    name_lower = node_name.lower()

    llm_keywords = ["llm", "generate", "chat", "complete", "respond", "gpt", "claude", "model"]
    tool_keywords = ["tool", "search", "lookup", "fetch", "query", "api", "call", "invoke"]
    retrieval_keywords = ["retrieve", "rag", "embed", "vector", "index", "document"]
    reasoning_keywords = ["reason", "think", "plan", "decide", "classify", "evaluate", "assess"]

    for kw in llm_keywords:
        if kw in name_lower:
            return "llm_call"
    for kw in tool_keywords:
        if kw in name_lower:
            return "tool_call"
    for kw in retrieval_keywords:
        if kw in name_lower:
            return "retrieval"
    for kw in reasoning_keywords:
        if kw in name_lower:
            return "reasoning"

    return "generic"
