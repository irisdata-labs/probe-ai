"""
agentprobe — Open-source trace capture and evaluation SDK for AI agents.

Instrument any AI agent with two lines of code:

    from agentprobe import trace_agent

    @trace_agent("my-agent")
    def run_agent(query: str) -> str:
        ...

Or wrap a LangGraph app:

    from agentprobe.integrations.langgraph import instrument_langgraph

    app = instrument_langgraph(existing_app)

Capture state changes and decisions:

    from agentprobe import record_state_change, record_decision

    record_state_change("balance", before=500, after=800)
    record_decision("escalate", alternatives=["resolve", "transfer"], reason="high value")
"""

__version__ = "0.2.0"

from agentprobe.core import (
    AgentProbe,
    get_current_span,
    get_current_trace,
    record_decision,
    record_state_change,
    step_span,
    trace_agent,
    trace_llm_call,
    trace_step,
    trace_tool,
)
from agentprobe.models import Decision, SpanRecord, StateChange, TraceRecord
from agentprobe.session import TraceSession

__all__ = [
    # Core
    "AgentProbe",
    "trace_agent",
    "trace_step",
    "trace_tool",
    "trace_llm_call",
    "step_span",
    "get_current_trace",
    "get_current_span",
    # State/decision tracking
    "record_state_change",
    "record_decision",
    # Models
    "TraceRecord",
    "SpanRecord",
    "StateChange",
    "Decision",
    # Session
    "TraceSession",
]
