# agentprobe

Open-source trace capture SDK for AI agents. Instrument any agent with two lines of code and capture full execution traces — every LLM call, tool invocation, state change, and decision point.

## Quick Start

```bash
pip install agentprobe
```

```python
from agentprobe import AgentProbe, trace_agent, trace_tool
from agentprobe.exporters import ConsoleExporter, JSONFileExporter

# Setup (once)
probe = AgentProbe(exporters=[ConsoleExporter(), JSONFileExporter("traces/")])
probe.init()

# Instrument your agent
@trace_agent("my-agent")
def handle_request(message: str) -> dict:
    order = lookup_order(message)
    return {"status": "done", "order": order}

@trace_tool("order-lookup")
def lookup_order(msg: str) -> dict:
    return {"order_id": "ORD-123", "status": "delivered"}

# Run — traces are captured automatically
handle_request("Check my order status")
```

Output:
```
[✓] my-agent (a3f8c1d2) | 15ms | 1 steps | 0 llm | 1 tool | 0 tokens
```

## Features

**Decorators** — Zero-change instrumentation for any Python function:
- `@trace_agent` — Top-level agent entry point (creates a new trace)
- `@trace_step` — Any step within the agent
- `@trace_tool` — Tool/API calls
- `@trace_llm_call` — LLM invocations
- `step_span` — Context manager for inline tracing

**State & Decision Tracking** — Capture what changed and why:
- `record_state_change(key, before, after)` — Track mutations
- `record_decision(chosen, alternatives, reason)` — Track branching decisions

**Exporters** — Send traces where you need them:
- `ConsoleExporter` — Human-readable terminal output
- `JSONFileExporter` — One JSON file per trace
- `OTelExporter` — OpenTelemetry spans (Datadog, Grafana, Jaeger)

**Sessions** — Batch multiple traces for comparison:
- `TraceSession` — Group traces, compute aggregate stats, save/load

**CLI** — Inspect traces from the terminal:
- `agentprobe list <dir>` — List trace files
- `agentprobe view <file>` — View trace detail
- `agentprobe summary <session>` — View session statistics

**LangGraph** — Automatic instrumentation:
```python
from agentprobe.integrations.langgraph import instrument_langgraph
app = instrument_langgraph(existing_app)  # All nodes traced automatically
```

## State Change & Decision Tracking

Track what your agent modifies and what choices it makes:

```python
from agentprobe import trace_agent, trace_step, record_state_change, record_decision

@trace_agent("support-agent")
def handle_return(order):
    return evaluate(order)

@trace_step("evaluate-return")
def evaluate(order):
    if order["days"] > 30:
        record_decision("deny", alternatives=["approve", "escalate"],
                        reason="Outside 30-day return window")
        return {"action": "deny"}

    record_decision("approve", alternatives=["deny", "escalate"],
                    reason="Within window, standard value")
    record_state_change("refund_status", before="none", after="processing")
    return {"action": "approve"}
```

The trace JSON includes:
```json
{
  "state_changes": [
    {"key": "refund_status", "before": "none", "after": "processing",
     "step_name": "evaluate-return"}
  ],
  "decisions": [
    {"chosen": "approve", "alternatives": ["deny", "escalate"],
     "reason": "Within window, standard value", "step_name": "evaluate-return"}
  ]
}
```

## Async Support

All decorators work with both sync and async functions:

```python
@trace_agent("async-agent")
async def handle(msg):
    result = await async_tool(msg)
    return result

@trace_tool("async-lookup")
async def async_tool(msg):
    return await some_api_call(msg)
```

## CLI

```bash
# List all traces in a directory
agentprobe list traces/

# View a specific trace
agentprobe view traces/my-agent_a3f8c1d2.json

# View with full JSON dump
agentprobe view traces/my-agent_a3f8c1d2.json -v

# View session summary
agentprobe summary traces/session_abc123/
```

## Examples

See the `examples/` directory:

- `novamart_support_agent.py` — Customer support agent with returns, escalation, all SDK features
- `mission_ops_copilot.py` — Satellite anomaly diagnosis agent (space domain)

```bash
python examples/novamart_support_agent.py
python examples/mission_ops_copilot.py
```

## Development

```bash
git clone https://github.com/irisdatalabs/agentprobe.git
cd agentprobe
pip install -e ".[dev]"
pytest tests/ -v
```

## Architecture

```
agentprobe/
├── __init__.py          # Public API
├── core.py              # Decorators, context management, state/decision tracking
├── models.py            # SpanRecord, TraceRecord, StateChange, Decision
├── session.py           # TraceSession batch management
├── cli.py               # Terminal commands
├── exporters/
│   ├── console.py       # Terminal output
│   ├── json_file.py     # JSON file storage
│   └── otel.py          # OpenTelemetry bridge
└── integrations/
    └── langgraph.py     # Automatic LangGraph instrumentation
```

## License

MIT — see [LICENSE](LICENSE).
