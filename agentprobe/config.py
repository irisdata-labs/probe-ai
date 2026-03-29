"""
agentprobe.config — Generate and load YAML test configurations.

Two workflows:

1. Heuristic (no LLM):
    config = generate_config(
        agent_description="Travel research agent",
        tools=["get-weather", "get-country-info", "search-facts"],
    )
    save_config(config, "test_config.yaml")

2. LLM-powered (richer output):
    config = generate_config_with_llm(
        agent_description="Travel research agent",
        tools=["get-weather", "get-country-info", "search-facts"],
        policy_docs=["Must return weather data..."],
        llm=my_llm_provider,
    )
    save_config(config, "test_config.yaml")

Then load and run:
    plan, world = load_config("test_config.yaml")
    engine = VariationEngine(plan=plan, world=world)
    scenarios = engine.generate(n=200)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

try:
    import yaml

    class _NoAliasDumper(yaml.SafeDumper):
        """YAML dumper that never uses anchors/aliases (item #13)."""
        def ignore_aliases(self, data):
            return True

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from agentprobe.scenarios.plan import (
    TestPlan, ScenarioCategory, WorldDimension, DimensionValue,
    PolicyRule, EdgeCase, RubricDimension,
)
from agentprobe.chaos import (
    WorldSimulator, ToolBehavior, ChaosProfile,
    Timeout, PartialData, StaleData, EmptyResponse,
    RateLimited, MalformedResponse, SchemaDrift, IntermittentFailure,
)


# ============================================================================
# Tool name heuristics (item #11: broader patterns)
# ============================================================================

TOOL_PATTERNS = {
    r"(order|purchase|transaction).*?(lookup|get|fetch|find|search|read|pull|load|retrieve|history)": {
        "category": "data_retrieval", "likely_params": ["order_id"],
        "description": "Retrieves order/transaction data",
        "failures": ["timeout", "partial_data", "empty_response", "stale_data"],
    },
    r"(weather|forecast|climate|temperature)": {
        "category": "external_api", "likely_params": ["location"],
        "description": "Fetches weather/forecast data from an external API",
        "failures": ["timeout", "empty_response", "stale_data", "schema_drift"],
    },
    r"(country|region|geo|location|city)": {
        "category": "external_api", "likely_params": ["name"],
        "description": "Fetches geographic or country data",
        "failures": ["timeout", "partial_data", "empty_response"],
    },
    r"(search|query|browse|discover).*?(fact|info|data|result|web|answer)": {
        "category": "search", "likely_params": ["query"],
        "description": "Searches for information",
        "failures": ["timeout", "empty_response", "rate_limited"],
    },
    r"(fetch|get|pull|load|retrieve|read|lookup|find|search|query).*?(order|user|customer|account|product|item|balance|record|transaction|history|data|info)": {
        "category": "data_retrieval", "likely_params": ["id"],
        "description": "Retrieves data from a store or database",
        "failures": ["timeout", "partial_data", "empty_response", "stale_data"],
    },
    r"(process|execute|create|submit|initiate|make|place).*?(return|refund|payment|order|transfer|transaction|booking)": {
        "category": "mutation", "likely_params": ["id"],
        "description": "Performs a write/mutation operation",
        "failures": ["timeout", "empty_response", "malformed_response"],
    },
    r"(escalat|transfer|handoff|route|assign|forward)": {
        "category": "routing", "likely_params": ["id", "reason"],
        "description": "Routes or escalates to another handler",
        "failures": ["timeout", "empty_response"],
    },
    r"(send|email|notify|alert|sms|push)": {
        "category": "notification", "likely_params": ["recipient", "message"],
        "description": "Sends a notification or message",
        "failures": ["timeout", "rate_limited"],
    },
    r"(generate|draft|compose|write|summarize|synthesize)": {
        "category": "llm_action", "likely_params": ["context"],
        "description": "Generates content (likely LLM-powered)",
        "failures": ["timeout", "empty_response", "malformed_response"],
    },
    r"(check|verify|validate|confirm|assess|authenticate).*?(identity|auth|user|access|permission|credential)": {
        "category": "validation_gate", "likely_params": ["id"],
        "description": "Validates identity or access — gating function",
        "failures": ["timeout", "empty_response", "malformed_response"],
    },
    r"(check|verify|validate|confirm|assess)": {
        "category": "validation", "likely_params": ["data"],
        "description": "Validates or checks data",
        "failures": ["timeout", "empty_response", "malformed_response"],
    },
}

DEFAULT_TOOL_PROFILE = {
    "category": "generic", "likely_params": ["input"],
    "description": "Tool", "failures": ["timeout", "empty_response"],
}


def _classify_tool(tool_name: str) -> Dict[str, Any]:
    name_lower = tool_name.lower().replace("-", " ").replace("_", " ")
    for pattern, profile in TOOL_PATTERNS.items():
        if re.search(pattern, name_lower):
            return {**profile, "matched_pattern": pattern}
    return {**DEFAULT_TOOL_PROFILE, "matched_pattern": None}


def _infer_domain(agent_description: str) -> str:
    desc = agent_description.lower()
    domain_keywords = {
        "customer_support": ["support", "customer", "help desk", "service", "ticket", "return", "refund"],
        "travel": ["travel", "flight", "hotel", "booking", "destination", "trip", "weather", "tourism"],
        "finance": ["finance", "banking", "payment", "transaction", "account", "trading", "balance", "transfer"],
        "healthcare": ["health", "medical", "patient", "diagnosis", "clinical", "prescription"],
        "ecommerce": ["shop", "store", "product", "cart", "checkout", "order"],
        "space_operations": ["satellite", "spacecraft", "mission", "telemetry", "orbit", "copilot"],
        "insurance": ["insurance", "policy", "claim", "underwriting", "premium"],
    }
    for domain, keywords in domain_keywords.items():
        if any(kw in desc for kw in keywords):
            return domain
    return "general"


# ============================================================================
# Item #10: Extract rules from policy_docs using regex
# ============================================================================

def _extract_rules_from_policy(policy_docs: List[str]) -> List[Dict]:
    rules = []
    if not policy_docs:
        return rules
    full_text = " ".join(policy_docs)

    # "X over/above $AMOUNT require Y"
    for m in re.finditer(
        r"(\w[\w\s]*?)\s+(?:over|above|exceeding|greater than|more than)\s+\$?([\d,]+(?:\.\d+)?)\s+(?:require|need|must|should)\s+([\w\s]+?)(?:\.|$)",
        full_text, re.IGNORECASE
    ):
        subject, amount, action = m.groups()
        amount_clean = amount.replace(",", "")
        rules.append({
            "name": f"threshold_{amount_clean}",
            "description": m.group(0).strip().rstrip("."),
            "condition": f"amount > {amount_clean}",
            "expected_outcome": action.strip(),
            "severity": "critical", "source": "policy_doc",
        })

    # "within X days"
    for m in re.finditer(r"(?:within|under|less than)\s+(\d+)\s+(day|hour|minute|week)s?", full_text, re.IGNORECASE):
        amount, unit = m.groups()
        rules.append({
            "name": f"time_window_{amount}_{unit}s",
            "description": f"Must be within {amount} {unit}s",
            "condition": f"time_elapsed <= {amount} {unit}s",
            "expected_outcome": f"Action taken within {amount} {unit}s",
            "severity": "critical", "source": "policy_doc",
        })

    # "X must be Y before Z"
    for m in re.finditer(r"(\w[\w\s]*?)\s+must\s+be\s+(\w[\w\s]*?)\s+before\s+(\w[\w\s]*?)(?:\.|$)", full_text, re.IGNORECASE):
        subject, state, action = m.groups()
        rules.append({
            "name": f"{subject.strip().lower().replace(' ', '_')}_required",
            "description": m.group(0).strip().rstrip("."),
            "condition": f"{subject.strip()} is {state.strip()}",
            "expected_outcome": f"{subject.strip()} {state.strip()} before {action.strip()}",
            "severity": "critical", "source": "policy_doc",
        })

    return rules


def _extract_dimensions_from_policy(policy_docs: List[str]) -> List[Dict]:
    dimensions = []
    if not policy_docs:
        return dimensions
    full_text = " ".join(policy_docs)

    raw_amounts = re.findall(r"\$(\d[\d,]*(?:\.\d+)?)", full_text)
    amounts = []
    for a in raw_amounts:
        try:
            val = float(a.replace(",", ""))
            if val > 0:
                amounts.append(val)
        except ValueError:
            continue
    if amounts:
        dimensions.append({
            "name": "amount", "category": "input",
            "description": f"Dollar amount (policy mentions: ${', $'.join(str(int(a)) for a in amounts)})",
            "type": "numeric_range", "range": [1, int(max(amounts) * 2)],
        })

    days = [int(d) for d in re.findall(r"(\d+)\s*days?", full_text, re.IGNORECASE) if int(d) > 0]
    if days:
        dimensions.append({
            "name": "days_elapsed", "category": "input",
            "description": f"Days (policy mentions: {', '.join(str(d) for d in days)} days)",
            "type": "numeric_range", "range": [1, int(max(days) * 2)],
        })

    return dimensions


# ============================================================================
# Item #5: Domain-aware templates
# ============================================================================

DOMAIN_RULES: Dict[str, List[Dict]] = {
    "customer_support": [
        {"name": "professional_tone", "description": "Agent must maintain professional and empathetic tone",
         "condition": "all interactions", "expected_outcome": "Professional response", "severity": "major"},
    ],
    "travel": [
        {"name": "data_completeness", "description": "Travel brief must include weather and country data when available",
         "condition": "tools return data", "expected_outcome": "Brief includes all available data", "severity": "major"},
        {"name": "stale_data_disclosure", "description": "Agent must disclose when data may be outdated",
         "condition": "data staleness > 24 hours", "expected_outcome": "Agent warns about data freshness", "severity": "major"},
    ],
    "finance": [
        {"name": "audit_trail", "description": "All transactions must be logged with full details",
         "condition": "any mutation", "expected_outcome": "Transaction details recorded", "severity": "critical"},
        {"name": "double_spend_prevention", "description": "Agent must not process duplicate transactions",
         "condition": "duplicate request detected", "expected_outcome": "Second request rejected", "severity": "critical"},
    ],
    "healthcare": [
        {"name": "patient_privacy", "description": "Agent must not expose patient data inappropriately",
         "condition": "patient data accessed", "expected_outcome": "Data shown only to authorized users", "severity": "critical"},
    ],
    "space_operations": [
        {"name": "safety_conservatism", "description": "Agent must err on the side of safety in ambiguous situations",
         "condition": "ambiguous telemetry", "expected_outcome": "Conservative recommendation", "severity": "critical"},
    ],
}

DOMAIN_EDGE_CASES: Dict[str, List[Dict]] = {
    "customer_support": [
        {"name": "legal_threat", "description": "Customer threatens legal action",
         "inject_description": "I've been waiting 3 weeks for my refund. If this isn't resolved by tomorrow I'm contacting my lawyer and filing a complaint with the BBB."},
        {"name": "multi_request", "description": "Customer sends multiple requests in one message",
         "inject_description": "I need to return order 5847, check the status of order 5901, and also I was charged twice for shipping on order 5847. Can you help with all three?"},
    ],
    "travel": [
        {"name": "unknown_destination", "description": "City not in any database",
         "inject_description": "I'm planning a trip to Xanadu. What's the weather like and what currency do they use?"},
        {"name": "ambiguous_city", "description": "City name exists in multiple countries",
         "inject_description": "I want to visit Portland. What's the weather and what should I know about traveling there?"},
        {"name": "travel_advisory", "description": "Destination with potential safety concerns",
         "inject_description": "I'm thinking about visiting a remote area in the Sahara desert. What's the weather and travel situation?"},
    ],
    "finance": [
        {"name": "negative_amount", "description": "Transaction with negative or zero amount",
         "inject_description": "Transfer -$500 from my checking to savings account"},
        {"name": "self_transfer", "description": "Transfer to the same account",
         "inject_description": "Send $100 from my account to my account"},
        {"name": "concurrent_request", "description": "Two transactions referencing the same account",
         "inject_description": "Process a payment of $5000 and also refund $2000, both on account 12345"},
    ],
    "healthcare": [
        {"name": "emergency_symptoms", "description": "Patient describes emergency symptoms",
         "inject_description": "I'm having severe chest pain and difficulty breathing. What should I do?"},
    ],
    "space_operations": [
        {"name": "conflicting_telemetry", "description": "Telemetry from redundant sensors disagrees",
         "inject_description": "Sensor A shows battery at 28V but sensor B shows 23V. Eclipse entry in 12 minutes."},
    ],
}

DOMAIN_DIMENSION_VALUES: Dict[str, Dict[str, List]] = {
    "customer_support": {
        "query": ["I want to return my headphones", "Where is my order?", "I need a refund",
                  "Can I speak to a manager?", "My order arrived damaged"],
        "id": ["ORD-1001", "ORD-1002", "ORD-1003", "ORD-1004", "ORD-1005"],
    },
    "travel": {
        "location": ["Paris", "Tokyo", "New York", "London", "Sydney", "Cairo", "Mumbai", "Toronto"],
        "name": ["France", "Japan", "United States", "United Kingdom", "Australia", "Egypt", "India", "Canada"],
        "query": ["Tokyo travel tips", "visa requirements Japan", "best time to visit Paris",
                  "London weather forecast", "Sydney travel advisory"],
    },
    "finance": {
        "id": ["ACCT-10001", "ACCT-10002", "ACCT-10003", "ACCT-10004", "ACCT-10005"],
        "data": ["checking_account", "savings_account", "business_account", "credit_card", "investment"],
        "recipient": ["vendor@company.com", "payroll@company.com", "john.doe@email.com"],
    },
}

DOMAIN_SPECIAL_CHARS: Dict[str, str] = {
    "customer_support": "I need help with order #123 — it's \"broken\" & I want a refund <ASAP> 🔥",
    "travel": "I want to visit São Paulo — what's the weather like? Also interested in Zürich & Naïrobi 🌤️",
    "finance": "Transfer $1,500.00 to acct #98-7654 — it's for invoice \"INV-2024/Q3\" & needs to be ASAP 💰",
    "healthcare": "My temp is 38.5°C & I have \"flu-like\" symptoms — been 3 days. Need Rx refill for naproxen 500mg 💊",
    "space_operations": "SAT-ε47 shows ΔV of −2.3m/s & thermal reading 85°C — possible β-angle issue? 🛰️",
}


# ============================================================================
# Generation functions
# ============================================================================

def _generate_categories(
    agent_description: str, tools: List[str], tool_profiles: Dict[str, Dict],
) -> List[Dict]:
    categories = [
        {"name": "happy-path", "description": "Standard requests where all tools work correctly",
         "count": 30, "rules_tested": [], "dimensions_varied": []},
    ]

    # Item #12: categories for mutation, routing, AND validation_gate tools
    for tool_name, profile in tool_profiles.items():
        if profile["category"] in ("mutation", "routing", "validation_gate"):
            cat_name = tool_name.replace("-", "_").replace(" ", "_")
            categories.append({
                "name": f"{cat_name}-scenarios",
                "description": f"Scenarios that exercise the {tool_name} tool",
                "count": 20, "rules_tested": [], "dimensions_varied": [],
            })

    categories.append({"name": "error-handling",
        "description": "Scenarios where tools fail — agent must handle gracefully",
        "count": 20, "rules_tested": ["graceful_error_handling"], "dimensions_varied": []})
    categories.append({"name": "adversarial",
        "description": "Adversarial scenarios — multiple tool failures, conflicting data, boundary inputs designed to break the agent",
        "count": 10, "rules_tested": [], "dimensions_varied": []})
    categories.append({"name": "edge-cases",
        "description": "Boundary conditions and unusual inputs",
        "count": 10, "rules_tested": [], "dimensions_varied": []})

    return categories


def _generate_rules(
    agent_description: str, tools: List[str], tool_profiles: Dict[str, Dict],
    domain: str, policy_docs: Optional[List[str]] = None,
) -> List[Dict]:
    rules = [
        {"name": "graceful_error_handling",
         "description": "Agent must not crash when tools fail — should return a clear error or fallback",
         "condition": "tool returns error or times out",
         "expected_outcome": "Agent returns graceful error message", "severity": "critical"},
        {"name": "no_hallucinated_data",
         "description": "Agent must not fabricate data when a tool returns empty or partial results",
         "condition": "tool returns empty or partial data",
         "expected_outcome": "Agent acknowledges missing data or asks for clarification", "severity": "critical"},
    ]

    mutation_tools = [t for t, p in tool_profiles.items() if p["category"] == "mutation"]
    if mutation_tools:
        rules.append({"name": "verify_before_mutation",
            "description": f"Agent must verify data before calling {', '.join(mutation_tools)}",
            "condition": "mutation tool is called",
            "expected_outcome": "Agent retrieves and validates data first", "severity": "major"})

    routing_tools = [t for t, p in tool_profiles.items() if p["category"] == "routing"]
    if routing_tools:
        rules.append({"name": "appropriate_escalation",
            "description": "Agent must escalate when the situation requires human judgment",
            "condition": "complex or sensitive request",
            "expected_outcome": "Agent escalates appropriately", "severity": "major"})

    gate_tools = [t for t, p in tool_profiles.items() if p["category"] == "validation_gate"]
    if gate_tools:
        rules.append({"name": "gate_before_action",
            "description": f"Agent must call {', '.join(gate_tools)} before performing sensitive operations",
            "condition": "sensitive operation requested",
            "expected_outcome": f"Agent calls {gate_tools[0]} first", "severity": "critical"})

    # Item #5: domain rules
    for rule in DOMAIN_RULES.get(domain, []):
        rules.append(dict(rule))

    return rules


def _generate_dimensions(
    tool_profiles: Dict[str, Dict], domain: str,
    policy_docs: Optional[List[str]] = None,
) -> List[Dict]:
    dimensions = []
    seen = set()

    # Item #4: deduplication aliases
    aliases = {
        "id": {"id", "order_id", "user_id", "customer_id", "account_id"},
        "location": {"location", "city", "destination"},
        "name": {"name", "country", "region"},
        "query": {"query", "search", "term"},
        "amount": {"amount", "price", "total", "value"},
    }

    def canonical(p):
        for c, a in aliases.items():
            if p in a:
                return c
        return p

    domain_vals = DOMAIN_DIMENSION_VALUES.get(domain, {})

    for tool_name, profile in tool_profiles.items():
        for param in profile.get("likely_params", []):
            c = canonical(param)
            if c in seen:
                continue
            seen.add(c)

            dim: Dict[str, Any] = {"name": c, "category": "input",
                                    "description": f"Input parameter for {tool_name}"}

            # Item #7: domain-aware values
            if c in domain_vals:
                dim["values"] = domain_vals[c]
                dim["type"] = "enum"
            elif c == "id":
                dim["values"] = [f"ID-{i:04d}" for i in range(1, 6)]
                dim["type"] = "enum"
            elif c in ("location",):
                dim["values"] = ["New York", "London", "Tokyo", "Paris", "Sydney"]
                dim["type"] = "enum"
            elif c in ("name",):
                dim["values"] = ["United States", "United Kingdom", "Japan", "France", "Australia"]
                dim["type"] = "enum"
            elif c == "amount":
                dim["type"] = "numeric_range"
                dim["range"] = [1, 1000]
            elif c == "query":
                dim["values"] = domain_vals.get("query", [
                    "typical user request", "complex multi-part question",
                    "very short query", "request with typos"])
                dim["type"] = "enum"
            else:
                dim["values"] = domain_vals.get(c, [f"sample_{c}_1", f"sample_{c}_2", f"sample_{c}_3"])
                dim["type"] = "enum"

            dimensions.append(dim)

    # Item #6: link location + name into single destination dimension
    loc = next((d for d in dimensions if d["name"] == "location"), None)
    nam = next((d for d in dimensions if d["name"] == "name"), None)
    if loc and nam:
        dimensions = [d for d in dimensions if d["name"] not in ("location", "name")]
        dimensions.insert(0, {
            "name": "destination", "category": "input",
            "description": "Destination city (linked to country)",
            "type": "enum", "values": loc.get("values", []),
        })

    return dimensions


def _generate_edge_cases(
    agent_description: str, tools: List[str], tool_profiles: Dict[str, Dict], domain: str,
) -> List[Dict]:
    # Build long input from the agent's actual context
    tool_list = ", ".join(tools[:3])
    long_input = (
        f"I have several questions. First, can you help me with something using {tools[0]}? "
        f"I tried it earlier and it didn't work. Also I need you to use {tools[-1]} for a "
        f"completely different thing. My friend told me you could also do something with "
        f"{tools[1] if len(tools) > 1 else tools[0]}, but I'm not sure how that works. "
        f"On top of that, I'm in a rush and need all of this done quickly. Oh and one more "
        f"thing — I had an issue last week where the response was wrong and I want to make "
        f"sure that doesn't happen again. Can you handle all of this? Thanks."
    )

    edges = [
        {"name": "empty_input", "description": "User sends an empty or whitespace-only message",
         "category": "edge-cases", "inject_description": ""},
        {"name": "very_long_input", "description": "User sends a rambling, multi-topic message touching multiple tools",
         "category": "edge-cases", "inject_description": long_input},
        {"name": "special_characters", "description": "Input contains special characters and unicode",
         "category": "edge-cases",
         "inject_description": DOMAIN_SPECIAL_CHARS.get(domain,
             "Input with spëcial chars: #123 — \"quotes\" & <brackets> plus émojis 🚀")},
    ]

    # Domain-specific edge cases
    for ec in DOMAIN_EDGE_CASES.get(domain, []):
        edges.append({**ec, "category": "edge-cases"})

    # Tool-specific: not-found for retrieval/external tools
    for tool_name, profile in tool_profiles.items():
        if profile["category"] in ("data_retrieval", "external_api"):
            edges.append({
                "name": f"{tool_name.replace('-','_')}_not_found",
                "description": f"{tool_name} returns no results for a valid-looking query",
                "category": "edge-cases",
                "inject_description": f"Look up something that doesn't exist in {tool_name}",
            })

    return edges


# ============================================================================
# Main generate function
# ============================================================================

def _agent_slug(agent_description: str, max_words: int = 4) -> str:
    """Generate a short slug from agent description for file/plan naming."""
    # Remove common filler words
    stop = {"a", "an", "the", "that", "and", "or", "for", "to", "in", "of", "with",
            "is", "it", "its", "this", "by", "from", "on", "at", "as", "be", "do",
            "fetches", "handles", "manages", "processes", "performs", "creates",
            "current", "various", "multiple", "specific", "based", "using"}
    words = re.sub(r"[^a-z0-9\s]", "", agent_description.lower()).split()
    slug_words = [w for w in words if w not in stop and len(w) > 2][:max_words]
    return "-".join(slug_words) if slug_words else "agent"


def suggested_filename(config: Dict[str, Any]) -> str:
    """
    Return a suggested filename for this config based on the agent description.

    Usage:
        config = generate_config(description, tools)
        path = suggested_filename(config)  # e.g. "travel-research-agent_config.yaml"
        save_config(config, path)
    """
    desc = config.get("agent", {}).get("description", "agent")
    slug = _agent_slug(desc)
    return f"{slug}_config.yaml"


def generate_config(
    agent_description: str,
    tools: List[str],
    policy_docs: Optional[List[str]] = None,
    chaos_level: str = "moderate",
) -> Dict[str, Any]:
    """Generate a complete test configuration. No LLM needed."""
    tool_profiles = {t: _classify_tool(t) for t in tools}
    domain = _infer_domain(agent_description)

    slug = _agent_slug(agent_description)

    config: Dict[str, Any] = {
        "_comment": "Generated by agentprobe. Edit this file to customize your test plan.",
        "agent": {
            "description": agent_description,
            "domain": domain,
            "tools": tools,
        },
        "test_plan": {
            "name": f"{slug}-evaluation",
            "categories": _generate_categories(agent_description, tools, tool_profiles),
            "rules": _generate_rules(agent_description, tools, tool_profiles, domain, policy_docs),
            "dimensions": _generate_dimensions(tool_profiles, domain, policy_docs),
            "edge_cases": _generate_edge_cases(agent_description, tools, tool_profiles, domain),
            "rubric": [
                {"dimension": "policy_accuracy", "description": "Does the agent follow its rules?", "weight": 0.4},
                {"dimension": "resilience", "description": "Does the agent handle tool failures?", "weight": 0.3},
                {"dimension": "completeness", "description": "Does the agent use all available data?", "weight": 0.3},
            ],
        },
        "chaos": {
            "level": chaos_level,
            "min_chaos_pct": 20,  # Item #3: minimum chaos guarantee
            "tools": {},
        },
    }

    for tool_name, profile in tool_profiles.items():
        config["chaos"]["tools"][tool_name] = {
            "description": profile["description"],
            "failure_modes": list(profile["failures"]),
            "example_response": {
                "_comment": f"Replace with a real example response from {tool_name}",
                "status": "ok",
            },
        }

    if policy_docs:
        config["agent"]["policy_docs"] = policy_docs
        config["_policy_note"] = (
            "Policy docs provided but not parsed in heuristic mode. "
            "Add domain-specific rules manually below, or use generate_config_with_llm() "
            "for automatic rule extraction from policy text."
        )

    return config


def generate_config_with_llm(
    agent_description: str, tools: List[str],
    policy_docs: Optional[List[str]] = None,
    llm: Optional[Any] = None, chaos_level: str = "moderate",
) -> Dict[str, Any]:
    """
    Generate config using an LLM for richer output. Falls back to heuristics.

    The LLM generates the test plan (rules, dimensions, categories, edge cases)
    from the policy docs. The chaos config and general fixes (items 1-7) always
    come from the heuristic path — the LLM doesn't configure tool failures.

    The final config merges both:
    - Test plan: LLM output (richer rules, dimensions, categories)
    - General rules: always included (graceful_error_handling, no_hallucination)
    - Edge cases: LLM edge cases + heuristic edge cases (realistic long input, domain-specific)
    - Chaos config: always heuristic (tool name → failure modes)
    - min_chaos_pct, no YAML anchors: always applied
    """
    # Start with full heuristic config as baseline
    config = generate_config(agent_description, tools, policy_docs, chaos_level)

    if llm and policy_docs:
        try:
            from agentprobe.scenarios import PlanGenerator
            generator = PlanGenerator(llm)
            plan = generator.generate(agent_description=agent_description,
                                       policy_docs=policy_docs, target_scenarios=100,
                                       tool_names=tools)

            # Save the heuristic baseline before overwriting
            heuristic_rules = config["test_plan"]["rules"]
            heuristic_edges = config["test_plan"]["edge_cases"]

            # Replace test plan with LLM output
            config["test_plan"]["name"] = plan.name

            config["test_plan"]["categories"] = [
                {"name": c.name, "description": c.description, "count": c.count,
                 "rules_tested": c.rules_tested, "dimensions_varied": c.dimensions_varied,
                 "example_scenario": c.example_scenario}
                for c in plan.categories]

            # Merge rules: LLM rules + ONLY universal heuristic rules
            # The LLM extracts policy-specific rules (better than heuristic domain rules),
            # so we only add the universal ones the LLM might have missed
            llm_rules = [
                {"name": r.name, "description": r.description, "condition": r.condition,
                 "expected_outcome": r.expected_outcome, "severity": r.severity,
                 "source": r.source}
                for r in plan.rules]
            llm_rule_names = {r["name"] for r in llm_rules}

            # Only merge universal rules (not domain-specific heuristic rules)
            UNIVERSAL_RULES = {"graceful_error_handling", "no_hallucinated_data",
                               "verify_before_mutation", "appropriate_escalation",
                               "gate_before_action"}
            for hr in heuristic_rules:
                if hr["name"] in UNIVERSAL_RULES and hr["name"] not in llm_rule_names:
                    llm_rules.append(hr)
                    llm_rule_names.add(hr["name"])
            config["test_plan"]["rules"] = llm_rules

            config["test_plan"]["dimensions"] = [
                {"name": d.name, "category": d.category, "description": d.description,
                 "type": d.value_spec.type, "values": d.value_spec.values, "range": d.value_spec.range}
                for d in plan.dimensions]

            # Merge edge cases: LLM + heuristic (deduplicated)
            llm_edges = [
                {"name": ec.name, "description": ec.description, "category": ec.category,
                 "inject_description": ec.inject_description}
                for ec in plan.edge_cases]
            llm_edge_names = {e["name"] for e in llm_edges}
            for he in heuristic_edges:
                if he["name"] not in llm_edge_names:
                    llm_edges.append(he)
                    llm_edge_names.add(he["name"])
            config["test_plan"]["edge_cases"] = llm_edges

            config["_comment"] = "Generated by agentprobe with LLM assistance. Edit to customize."
            # Remove the heuristic-mode policy note since LLM parsed the docs
            config.pop("_policy_note", None)

        except Exception as e:
            config["_llm_error"] = f"LLM generation failed: {e}. Falling back to heuristics."

    return config


# ============================================================================
# Save / Load
# ============================================================================

def save_config(config: Dict[str, Any], path: str) -> str:
    if not HAS_YAML:
        raise ImportError("PyYAML is required: pip install pyyaml")
    with open(path, "w") as f:
        yaml.dump(config, f, Dumper=_NoAliasDumper, default_flow_style=False,
                  sort_keys=False, allow_unicode=True, width=100)
    return path


def load_config(path: str) -> tuple:
    if not HAS_YAML:
        raise ImportError("PyYAML is required: pip install pyyaml")
    with open(path) as f:
        config = yaml.safe_load(f)
    return _config_to_plan(config), _config_to_world(config)


def _config_to_plan(config: Dict) -> TestPlan:
    tp = config.get("test_plan", {})
    agent = config.get("agent", {})
    return TestPlan(
        name=tp.get("name", "evaluation"),
        agent_description=agent.get("description", ""),
        domain=agent.get("domain", "general"),
        categories=[ScenarioCategory(
            name=c["name"], description=c.get("description", ""), count=c.get("count", 20),
            rules_tested=c.get("rules_tested", []), dimensions_varied=c.get("dimensions_varied", []),
            example_scenario=c.get("example_scenario"),
        ) for c in tp.get("categories", [])],
        rules=[PolicyRule(
            name=r["name"], description=r.get("description", ""), condition=r.get("condition", ""),
            expected_outcome=r.get("expected_outcome", ""), severity=r.get("severity", "major"),
            source=r.get("source"),
        ) for r in tp.get("rules", [])],
        dimensions=[WorldDimension(
            name=d["name"], category=d.get("category", "input"), description=d.get("description", ""),
            value_spec=DimensionValue(type=d.get("type", "enum"), values=d.get("values"), range=d.get("range")),
        ) for d in tp.get("dimensions", [])],
        edge_cases=[EdgeCase(
            name=ec.get("name", "edge_case"), description=ec.get("description", ""),
            category=ec.get("category", "edge-cases"), inject_description=ec.get("inject_description", ""),
        ) for ec in tp.get("edge_cases", [])],
        rubric=[RubricDimension(
            dimension=rb.get("dimension", ""), description=rb.get("description", ""), weight=rb.get("weight", 1.0),
        ) for rb in tp.get("rubric", [])],
    )


_FAILURE_MODE_MAP = {
    "timeout": lambda: Timeout(), "partial_data": lambda: PartialData(),
    "stale_data": lambda: StaleData(), "empty_response": lambda: EmptyResponse(),
    "empty": lambda: EmptyResponse(), "rate_limited": lambda: RateLimited(),
    "malformed_response": lambda: MalformedResponse(), "malformed": lambda: MalformedResponse(),
    "schema_drift": lambda: SchemaDrift(), "intermittent": lambda: IntermittentFailure(),
}


def _config_to_world(config: Dict) -> WorldSimulator:
    chaos = config.get("chaos", {})
    profile_map = {"gentle": ChaosProfile.GENTLE, "moderate": ChaosProfile.MODERATE,
                   "hostile": ChaosProfile.HOSTILE, "adversarial": ChaosProfile.ADVERSARIAL}
    profile = profile_map.get(chaos.get("level", "moderate"), ChaosProfile.MODERATE)

    tools = {}
    for tool_name, tool_cfg in chaos.get("tools", {}).items():
        fms = [_FAILURE_MODE_MAP[fm.lower().replace(" ", "_")]()
               for fm in tool_cfg.get("failure_modes", [])
               if fm.lower().replace(" ", "_") in _FAILURE_MODE_MAP]
        example = tool_cfg.get("example_response")
        # Strip placeholder comments
        if example and "_comment" in example:
            example = {k: v for k, v in example.items() if k != "_comment"}
            if not example or example == {"status": "ok"}:
                example = None
        tools[tool_name] = ToolBehavior(
            description=tool_cfg.get("description", ""),
            failure_modes=fms,
            example_response=example,
        )

    # normal_ratio=0.0 means: when the VariationEngine asks the WorldSimulator
    # to generate a chaos config, it ALWAYS injects at least one failure.
    # The VariationEngine's chaos_ratio controls HOW MANY scenarios get chaos.
    return WorldSimulator(tools=tools, profile=profile, seed=42, normal_ratio=0.0)
