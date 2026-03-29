# probe-ai

**Chaos engineering for AI agents.**

Your agent works when everything goes right. What happens when tools time out, APIs return garbage, and services go down?

probe-ai generates failure scenarios, injects them into your agent's tool calls, and produces a report showing exactly what breaks and why.

```
┌─────────────────────────────────────────────────────────┐
│  probe-ai Evaluation Report                           │
│                                                         │
│  Scenarios: 200                                         │
│  Passed:     162  — correct, complete output             │
│  Degraded:    31  — tools failed, agent survived         │
│  Failed:       7  — crash, policy violation, bad output  │
│                                                         │
│  Resilience: Grade A (96/100)                           │
│                                                         │
│  Failure Clusters:                                      │
│    #1: 4 crashes — get-weather timeout + search timeout  │
│    #2: 3 policy violations — stale data not disclosed    │
└─────────────────────────────────────────────────────────┘
```

---

## Install

```bash
# From PyPI
pip install probe-ai

# From source
pip install -e /path/to/probe-ai
```

> **Note:** The package installs as `probe-ai` but you import it as `agentprobe` in Python. This is because the name `agentprobe` was already taken on PyPI.

## How it works — 5 steps

### Step 1: Describe your agent

Tell probe-ai what your agent does and what tools it uses. Everything else is optional — add whatever context you have.

```python
from agentprobe.config import generate_config, save_config, load_config

config = generate_config(
    agent_description="...",   # required — what your agent does
    tools=["tool-a", "tool-b"],# required — names of tools the agent calls
    policy_docs=[...],         # optional — policies, rules, SLAs, compliance docs
    chaos_level="moderate",    # optional — gentle, moderate, hostile, adversarial
)
```

The more context you give, the better the test scenarios. A bare description and tool list works. Adding policy docs, workflow descriptions, or compliance rules makes the generated tests much richer.

```python
# Minimal — works fine
config = generate_config(
    agent_description="Travel agent that fetches weather and country info",
    tools=["get-weather", "get-country-info"],
)

# Richer — probe-ai generates more targeted scenarios
config = generate_config(
    agent_description="Travel agent that fetches weather and country info",
    tools=["get-weather", "get-country-info", "search-facts"],
    policy_docs=[
        "Briefs must include weather data when available.",
        "Agent must not fabricate data when a tool fails.",
        open("travel_policy.md").read(),  # load from file
    ],
)
```

### Step 2: Define your tool responses

probe-ai needs to know what your tools normally return so it can create realistic mock data. This is the one thing it can't infer — only you know your tool's response format.

```python
config["chaos"]["tools"]["get-weather"]["example_response"] = {
    "city": "Paris", "temperature_c": 22.5, "wind_speed_kmh": 15.0
}
config["chaos"]["tools"]["get-country-info"]["example_response"] = {
    "country": "France", "capital": "Paris", "currency": "Euro"
}

save_config(config, "eval_config.yaml")
```

### Step 3: Add one line to each tool function

Your existing agent code stays unchanged. Just add `mock_tool_response()` at the top of each tool — it returns chaos when probe-ai injects a failure, or `None` when your real code should run.

```python
from agentprobe import trace_agent, trace_tool
from agentprobe.engine import mock_tool_response

@trace_tool("get-weather")
def get_weather(city):
    mock = mock_tool_response("get-weather")  # ← this is the only new line
    if mock is not None: return mock           # chaos injected → return failure
    return call_real_weather_api(city)         # no chaos → your existing code

@trace_agent("my-agent")
def my_agent(query):
    weather = get_weather(query)
    # ... your agent logic, unchanged ...
    return {"action": "complete", "brief": "..."}
```

### Step 4: Run the evaluation

probe-ai loads your config, generates test scenarios (a mix of normal requests and injected failures), runs your agent against each one, and checks the results.

```python
from agentprobe import AgentProbe
from agentprobe.engine import VariationEngine, Runner, ContentEvaluator

# Load the config you saved in Step 2
plan, world = load_config("eval_config.yaml")

# Generate 200 test scenarios — happy paths, edge cases, and chaos
scenarios = VariationEngine(plan, world).generate(n=200)

# Set up evaluation — what does a good response look like?
probe = AgentProbe(exporters=[])
probe.init()

evaluator = ContentEvaluator(
    required_fields=["brief"],                           # these fields must be present
    tool_output_fields={"get-weather": ["temperature_c"]}, # tool data should appear in output
    min_text_length={"brief": 50},                        # brief should be at least 50 chars
)

# Run your agent against all scenarios
runner = Runner(agent_fn=my_agent, evaluator=evaluator, probe=probe)
results = [runner.run_one(s) for s in scenarios]
```

### Step 5: Get your report

```python
from agentprobe.engine import EvaluationReport, export_html
from agentprobe.analysis import analyze_failures

report = EvaluationReport.build(results, scenarios, analyze_failures(results))

export_html(report, "report.html")  # open in browser — full dashboard
report.to_json("report.json")       # structured data for CI/CD
print(report.render())              # console summary
```

---

## LLM configuration

probe-ai does **not** require an LLM for evaluation — scenario generation, chaos injection, and reporting are all deterministic. But there are two places where an LLM is useful:

### Your agent's own LLM

If your agent calls an LLM (most do), you configure that yourself as you normally would. probe-ai doesn't interfere — it just traces the calls.

```python
import os
from openai import OpenAI

# Your agent's LLM — use any provider
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Or Groq (free tier available)
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

# Or Anthropic, Azure, local models — whatever your agent uses
```

### LLM-powered config generation (optional)

The heuristic config generator needs no LLM. But if you want richer, policy-aware test plans, you can use `generate_config_with_llm()` which sends your policy docs to an LLM to extract domain-specific rules:

```python
from agentprobe.config import generate_config_with_llm
from agentprobe.llm.groq import Groq

# Supported providers: Groq, OpenAI, Claude, LiteLLM
llm = Groq(model="llama-3.3-70b-versatile")  # free tier at console.groq.com

config = generate_config_with_llm(
    agent_description="...",
    tools=["..."],
    policy_docs=["..."],
    llm=llm,
)
```

Available LLM providers:
```python
from agentprobe.llm.groq import Groq           # Groq (free tier available)
from agentprobe.llm.openai import OpenAILLM     # OpenAI
from agentprobe.llm.claude import ClaudeLLM     # Anthropic Claude
from agentprobe.llm.litellm import LiteLLMLLM   # Any model via LiteLLM
```

---

## What probe-ai does

### 1. Generates failure scenarios from your agent description

You describe your agent and its tools. probe-ai generates hundreds of test scenarios — happy paths, edge cases, and chaos conditions — without you writing a single test case.

```yaml
# Auto-generated config (you edit the example_response, everything else is inferred)
chaos:
  tools:
    get-weather:
      failure_modes: [timeout, empty_response, stale_data, schema_drift]
      example_response:
        city: Paris
        temperature_c: 22.5
```

Two config modes:
- **Heuristic** — no LLM needed. Infers rules from tool names and domain. Fast.
- **LLM-powered** — sends your policy docs to an LLM, extracts domain-specific rules, dimensions, and edge cases. Richer.

`policy_docs` is optional but powerful. It accepts any text that describes how your agent should behave — pass whatever you have:

```python
config = generate_config(
    agent_description="Insurance underwriting agent",
    tools=["assess-risk", "lookup-claims", "generate-quote"],
    policy_docs=[
        open("underwriting_policy.txt").read(),     # policy document
        open("compliance_rules.md").read(),          # compliance rules
        open("workflow.md").read(),                   # workflow description
        "Quotes over $1M require manager approval",  # plain English rule
    ],
)
```

Policy docs, SLAs, workflow descriptions, compliance rules, runbooks, onboarding guides — if it describes what the agent should or shouldn't do, pass it in. In heuristic mode, these are stored in the config for reference. In LLM mode, they're parsed to extract testable rules automatically.

### 2. Injects chaos into tool calls

9 failure modes, injected at the tool level during execution:

| Failure | What it does |
|---|---|
| `timeout` | Tool doesn't respond |
| `empty_response` | Tool returns nothing |
| `partial_data` | Some fields missing |
| `stale_data` | Data is outdated |
| `rate_limited` | Too many requests |
| `schema_drift` | Response schema changed |
| `malformed_response` | Invalid data format |
| `intermittent` | Fails randomly |
| `contradiction` | Conflicts with other tools |

Your agent's tool functions use `mock_tool_response()` — when chaos is injected, it returns the failure. When not, your real tool runs normally.

### 3. Evaluates with three verdicts

Not just pass/fail. Three outcomes that map to production reality:

- **Passed** ✓ — Agent produced complete, correct output.
- **Degraded** ⚠ — Tools had outages (timeout, empty). Agent didn't crash, but output is incomplete.
- **Failed** ✗ — Agent crashed, violated policy, or produced bad output.

The `ContentEvaluator` checks response quality deterministically — no LLM needed for evaluation:
- Required fields present?
- Tool data included when tool succeeded?
- Placeholder text in the response?
- Response long enough?

### 4. Produces structured reports

**HTML** — self-contained file, open in any browser. Summary cards, difficulty distribution, scenarios grouped by category.

**JSON** — structured data for CI/CD pipelines. Parse it, set thresholds, fail the build.

**Console** — text output for notebooks and terminals.

Reports include difficulty scoring (easy/medium/hard/adversarial), failure clustering (which failures share a root cause), and resilience grading (A through F).

---

## Real tools or mocks — you choose

```python
# In your tool function:
@trace_tool("get-weather")
def get_weather(city):
    mock = mock_tool_response("get-weather")
    if mock is not None: return mock    # chaos injected → return mock failure
    return call_real_api(city)          # no chaos → call real API
```

- **Normal scenarios**: `mock_tool_response()` returns `None`. Your real API runs.
- **Chaos scenarios**: Returns the injected failure (timeout error, empty response, etc.)
- **Full mock mode**: Add a fallback to `example_response` for fast CI/CD runs without hitting real APIs.

---

## Works with any agent

probe-ai is framework-agnostic. It works with:

- **LangChain / LangGraph** — first-class integration via `instrument_langgraph()`
- **OpenAI Agents SDK** — decorator-based instrumentation
- **Custom agents** — any Python function with `@trace_agent`

The only requirement: your tool functions check `mock_tool_response()` at the top. That's one line per tool.

---

## Key concepts

### example_response
The developer's contract for mock data. probe-ai doesn't guess what your tools return — you tell it. This keeps mocks realistic and the SDK generalizable to any domain.

### Three-verdict model
Traditional testing: pass or fail. But "the tool timed out and the agent returned a partial response" isn't a pass OR a fail — it's a degraded outcome. The three-verdict model captures this, which matters for resilience scoring.

### Chaos ratio
`min_chaos_pct` in the config controls what percentage of scenarios get tool failures. Default 20% — enough to stress-test resilience without drowning in failures.

---

## API Reference

### Decorators
```python
@trace_agent("agent-name")      # wraps your agent's main function
@trace_tool("tool-name")        # wraps each tool function
@trace_llm_call("call-name")    # wraps LLM API calls
```

### Config
```python
generate_config(description, tools, policy_docs, chaos_level)          # heuristic
generate_config_with_llm(description, tools, policy_docs, llm)         # LLM-powered
save_config(config, "path.yaml")
plan, world = load_config("path.yaml")
```

### Scenarios & execution
```python
engine = VariationEngine(plan, world, seed=42)
scenarios = engine.generate(n=200, chaos_ratio=0.2)

runner = Runner(agent_fn, evaluator, probe, input_builder)
results = [runner.run_one(s) for s in scenarios]
```

### Evaluation
```python
evaluator = ContentEvaluator(
    required_fields=["brief"],
    tool_output_fields={"get-weather": ["temperature_c"]},
    min_text_length={"brief": 50},
)

analysis = analyze_failures(results)
```

### Reporting
```python
report = EvaluationReport.build(results, scenarios, analysis)
print(report.render())                # console
export_html(report, "report.html")    # browser
report.to_json("report.json")         # CI/CD
```

---

## Why not [existing tool]?

### Evaluation tools (test output quality)

| Tool | What it does | What's missing |
|---|---|---|
| **DeepEval** | LLM-as-judge metrics (hallucination, relevance) | No chaos injection. Tests answers, not resilience. Requires LLM for every evaluation. |
| **cane-eval** | YAML test suites scored by Claude | You write every test case manually. No scenario generation. No tool failure injection. |
| **LangChain agentevals** | Trajectory evaluation (were steps correct?) | Evaluates steps taken. Doesn't inject failures to see what happens when steps break. |
| **Promptfoo** | Prompt regression testing | Tests prompts, not agent behavior. No tool-level chaos. |

### Chaos engineering tools (test resilience)

| Tool | What it does | How probe-ai differs |
|---|---|---|
| **agent-chaos** | Injects LLM rate limits, tool errors, stream failures | Tied to pydantic-ai. Requires LLM-as-judge (DeepEval) for evaluation. You write test scenarios manually. |
| **balagan-agent** | Wraps agents with fault injection | Tied to CrewAI. No scenario generation — you define every test. No structured reporting. |

### Observability platforms (monitor production)

| Tool | What it does | What's missing |
|---|---|---|
| **LangSmith / Arize** | Trace and monitor agents in production | Observes what happened. Doesn't generate failure scenarios or test before deployment. |
| **AWS agent-evaluation** | Multi-turn conversation simulation | AWS-specific. No chaos injection. |

### What makes probe-ai different

1. **Scenario generation from description + policy docs.** You don't write test cases. Describe your agent, pass your policy docs, and probe-ai generates hundreds of scenarios automatically — including edge cases and adversarial inputs you wouldn't think to test.

2. **Framework-agnostic.** Works with LangChain, LangGraph, OpenAI Agents SDK, pydantic-ai, CrewAI, or plain Python functions. One line per tool (`mock_tool_response()`), no framework lock-in.

3. **Deterministic evaluation — no LLM needed.** The `ContentEvaluator` checks response quality with reproducible, deterministic checks. No LLM-as-judge means faster, cheaper, and consistent results across runs. You get the same result every time.

4. **Three-verdict model.** Not just pass/fail. Passed (complete output), degraded (tools failed, agent survived), and failed (crash/violation/bad output). Maps to how agents actually behave in production.

5. **Real tools or mocks — you choose.** Normal scenarios can hit your real APIs while chaos scenarios inject failures. Or run everything mocked for fast CI/CD. One setting, same code.

**Other tools test what the agent says. probe-ai tests what happens when the world the agent depends on breaks.**

---

## License

MIT

---

Built by [IrisDataLabs](https://irisdatalabs.com)
