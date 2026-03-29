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
        policy_docs=["Must return weather data...", "Handle failures gracefully..."],
        llm=my_llm_provider,
    )
    save_config(config, "test_config.yaml")

Then load and run:
    plan, world = load_config("test_config.yaml")
    engine = VariationEngine(plan=plan, world=world)
    scenarios = engine.generate(n=200)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
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
# Tool name heuristics — infer tool behavior from naming patterns
# ============================================================================

# Pattern → (likely parameters, likely response fields, likely failure modes)
TOOL_PATTERNS = {
    r"(order|purchase|transaction).*?(lookup|get|fetch|find|search)": {
        "category": "data_retrieval",
        "likely_params": ["order_id"],
        "likely_response": ["order_id", "status", "amount", "customer", "date"],
        "failures": ["timeout", "partial_data", "empty_response", "stale_data"],
        "description": "Retrieves order/transaction data",
    },
    r"(lookup|get|fetch|find|search|query|read).*?(order|user|customer|account|product|item)": {
        "category": "data_retrieval",
        "likely_params": ["id"],
        "likely_response": ["id", "name", "status", "data"],
        "failures": ["timeout", "partial_data", "empty_response", "stale_data"],
        "description": "Retrieves data from a store or database",
    },
    r"(weather|forecast|climate|temperature)": {
        "category": "external_api",
        "likely_params": ["location"],
        "likely_response": ["temperature", "conditions", "humidity", "wind"],
        "failures": ["timeout", "empty_response", "stale_data", "schema_drift"],
        "description": "Fetches weather/forecast data from an external API",
    },
    r"(country|region|geo|location|city)": {
        "category": "external_api",
        "likely_params": ["name"],
        "likely_response": ["name", "capital", "population", "currency", "language"],
        "failures": ["timeout", "partial_data", "empty_response"],
        "description": "Fetches geographic or country data",
    },
    r"(search|query|find|lookup|browse|discover)": {
        "category": "search",
        "likely_params": ["query"],
        "likely_response": ["results", "total", "source"],
        "failures": ["timeout", "empty_response", "rate_limited"],
        "description": "Searches for information",
    },
    r"(process|execute|create|submit|initiate).*?(return|refund|payment|order|transfer)": {
        "category": "mutation",
        "likely_params": ["id", "reason"],
        "likely_response": ["confirmation_id", "status"],
        "failures": ["timeout", "empty_response", "malformed_response"],
        "description": "Performs a write/mutation operation",
    },
    r"(escalat|transfer|handoff|route|assign)": {
        "category": "routing",
        "likely_params": ["id", "reason"],
        "likely_response": ["ticket_id", "queue", "assigned_to"],
        "failures": ["timeout", "empty_response"],
        "description": "Routes or escalates to another handler",
    },
    r"(send|email|notify|alert|message|sms)": {
        "category": "notification",
        "likely_params": ["recipient", "message"],
        "likely_response": ["sent", "message_id"],
        "failures": ["timeout", "rate_limited"],
        "description": "Sends a notification or message",
    },
    r"(generate|create|draft|compose|write|summarize)": {
        "category": "llm_action",
        "likely_params": ["prompt", "context"],
        "likely_response": ["text", "content"],
        "failures": ["timeout", "empty_response", "malformed_response"],
        "description": "Generates content (likely LLM-powered)",
    },
    r"(check|verify|validate|confirm|assess)": {
        "category": "validation",
        "likely_params": ["data"],
        "likely_response": ["valid", "result", "errors"],
        "failures": ["timeout", "empty_response", "malformed_response"],
        "description": "Validates or checks data",
    },
}

# Default for unrecognized tools
DEFAULT_TOOL_PROFILE = {
    "category": "generic",
    "likely_params": ["input"],
    "likely_response": ["result", "status"],
    "failures": ["timeout", "empty_response"],
    "description": "Generic tool",
}


def _classify_tool(tool_name: str) -> Dict[str, Any]:
    """Classify a tool based on its name using pattern matching."""
    name_lower = tool_name.lower().replace("-", " ").replace("_", " ")
    for pattern, profile in TOOL_PATTERNS.items():
        if re.search(pattern, name_lower):
            return {**profile, "matched_pattern": pattern}
    return {**DEFAULT_TOOL_PROFILE, "matched_pattern": None}


def _infer_domain(agent_description: str) -> str:
    """Infer the domain from the agent description."""
    desc = agent_description.lower()
    domain_keywords = {
        "customer_support": ["support", "customer", "help desk", "service", "ticket"],
        "travel": ["travel", "flight", "hotel", "booking", "destination", "trip"],
        "finance": ["finance", "banking", "payment", "transaction", "account", "trading"],
        "healthcare": ["health", "medical", "patient", "diagnosis", "clinical"],
        "ecommerce": ["shop", "store", "product", "cart", "checkout", "order", "return"],
        "space_operations": ["satellite", "spacecraft", "mission", "telemetry", "orbit"],
        "insurance": ["insurance", "policy", "claim", "underwriting", "premium"],
    }
    for domain, keywords in domain_keywords.items():
        if any(kw in desc for kw in keywords):
            return domain
    return "general"


def _generate_categories(
    agent_description: str, tools: List[str], tool_profiles: Dict[str, Dict],
) -> List[Dict]:
    """Generate scenario categories from agent description and tools."""
    categories = []
    desc = agent_description.lower()

    # Main happy path category
    categories.append({
        "name": "happy-path",
        "description": "Standard requests where all tools work and the agent should succeed",
        "count": 30,
        "rules_tested": [],
        "dimensions_varied": [],
    })

    # Category for each mutation/action tool
    for tool_name, profile in tool_profiles.items():
        if profile["category"] in ("mutation", "routing"):
            cat_name = tool_name.replace("-", "_").replace(" ", "_")
            categories.append({
                "name": f"{cat_name}-scenarios",
                "description": f"Scenarios that exercise the {tool_name} tool",
                "count": 20,
                "rules_tested": [],
                "dimensions_varied": [],
            })

    # Error handling category
    categories.append({
        "name": "error-handling",
        "description": "Scenarios where tools fail and the agent must handle errors gracefully",
        "count": 20,
        "rules_tested": ["graceful_error_handling"],
        "dimensions_varied": [],
    })

    # Edge cases category
    categories.append({
        "name": "edge-cases",
        "description": "Boundary conditions, unusual inputs, and adversarial scenarios",
        "count": 10,
        "rules_tested": [],
        "dimensions_varied": [],
    })

    return categories


def _generate_rules(
    agent_description: str, tools: List[str], tool_profiles: Dict[str, Dict],
) -> List[Dict]:
    """Generate policy rules from agent description and tools."""
    rules = []

    # Always include: handle errors gracefully
    rules.append({
        "name": "graceful_error_handling",
        "description": "Agent must not crash when tools fail — should return a clear error or fallback",
        "condition": "tool returns error or times out",
        "expected_outcome": "Agent returns graceful error message",
        "severity": "critical",
    })

    # Always include: don't hallucinate data
    rules.append({
        "name": "no_hallucinated_data",
        "description": "Agent must not fabricate data when a tool returns empty or partial results",
        "condition": "tool returns empty or partial data",
        "expected_outcome": "Agent acknowledges missing data or asks for clarification",
        "severity": "critical",
    })

    # If there are mutation tools: verify before mutating
    mutation_tools = [t for t, p in tool_profiles.items() if p["category"] == "mutation"]
    if mutation_tools:
        rules.append({
            "name": "verify_before_mutation",
            "description": f"Agent must verify data before calling {', '.join(mutation_tools)}",
            "condition": "mutation tool is called",
            "expected_outcome": "Agent retrieves and validates data before performing the mutation",
            "severity": "major",
        })

    # If there are routing/escalation tools: escalate when appropriate
    routing_tools = [t for t, p in tool_profiles.items() if p["category"] == "routing"]
    if routing_tools:
        rules.append({
            "name": "appropriate_escalation",
            "description": "Agent must escalate when the situation requires human judgment",
            "condition": "complex or sensitive request",
            "expected_outcome": "Agent escalates rather than attempting to handle independently",
            "severity": "major",
        })

    return rules


def _generate_edge_cases(
    agent_description: str, tools: List[str], tool_profiles: Dict[str, Dict],
) -> List[Dict]:
    """Generate edge cases."""
    edges = []

    edges.append({
        "name": "empty_input",
        "description": "User sends an empty or whitespace-only message",
        "category": "edge-cases",
        "inject_description": "",
    })

    edges.append({
        "name": "very_long_input",
        "description": "User sends an extremely long message (>1000 chars)",
        "category": "edge-cases",
        "inject_description": "Please help me with my issue. " * 50,
    })

    edges.append({
        "name": "special_characters",
        "description": "User input contains special characters and unicode",
        "category": "edge-cases",
        "inject_description": "I need help with order #123 — it's \"broken\" & I want a refund <ASAP> 🔥",
    })

    # Tool-specific edge cases
    for tool_name, profile in tool_profiles.items():
        if profile["category"] == "data_retrieval":
            edges.append({
                "name": f"{tool_name.replace('-','_')}_not_found",
                "description": f"{tool_name} returns no results for a valid-looking query",
                "category": "edge-cases",
                "inject_description": f"Look up something that doesn't exist",
            })

    return edges


# ============================================================================
# Generate config — heuristic (no LLM)
# ============================================================================

def generate_config(
    agent_description: str,
    tools: List[str],
    policy_docs: Optional[List[str]] = None,
    chaos_level: str = "moderate",
) -> Dict[str, Any]:
    """
    Generate a complete test configuration from agent description and tool names.

    No LLM needed — uses heuristics to infer tool behavior, generate categories,
    rules, and chaos profiles from naming patterns.

    Args:
        agent_description: What the agent does (1-3 sentences).
        tools: List of tool names the agent uses.
        policy_docs: Optional policy text (used for comments in the YAML).
        chaos_level: One of: gentle, moderate, hostile, adversarial.

    Returns:
        Dict that can be saved to YAML via save_config().
    """
    # Classify each tool
    tool_profiles = {t: _classify_tool(t) for t in tools}
    domain = _infer_domain(agent_description)

    # Generate the config
    config: Dict[str, Any] = {
        "_comment": "Generated by agentprobe. Edit this file to customize your test plan.",
        "agent": {
            "description": agent_description,
            "domain": domain,
            "tools": tools,
        },
        "test_plan": {
            "name": f"{domain}-evaluation",
            "categories": _generate_categories(agent_description, tools, tool_profiles),
            "rules": _generate_rules(agent_description, tools, tool_profiles),
            "dimensions": _generate_dimensions_from_tools(tool_profiles),
            "edge_cases": _generate_edge_cases(agent_description, tools, tool_profiles),
            "rubric": [
                {"dimension": "policy_accuracy", "description": "Does the agent follow its rules?", "weight": 0.4},
                {"dimension": "resilience", "description": "Does the agent handle tool failures?", "weight": 0.3},
                {"dimension": "completeness", "description": "Does the agent use all available data?", "weight": 0.3},
            ],
        },
        "chaos": {
            "level": chaos_level,
            "tools": {},
        },
    }

    # Generate chaos config per tool
    for tool_name, profile in tool_profiles.items():
        config["chaos"]["tools"][tool_name] = {
            "description": profile["description"],
            "failure_modes": profile["failures"],
        }

    # Add policy docs as comments if provided
    if policy_docs:
        config["agent"]["policy_docs"] = policy_docs

    return config


def _generate_dimensions_from_tools(tool_profiles: Dict[str, Dict]) -> List[Dict]:
    """Generate test dimensions from tool profiles."""
    dimensions = []
    seen_params = set()

    for tool_name, profile in tool_profiles.items():
        for param in profile.get("likely_params", []):
            if param not in seen_params:
                seen_params.add(param)
                dim = {
                    "name": param,
                    "category": "input",
                    "description": f"Input parameter for {tool_name}",
                }
                # Infer type from param name
                if param in ("id", "order_id", "user_id", "customer_id", "account_id"):
                    dim["values"] = [f"ID-{i:04d}" for i in range(1, 6)]
                    dim["type"] = "enum"
                elif param in ("location", "city", "destination"):
                    dim["values"] = ["New York", "London", "Tokyo", "Paris", "Sydney"]
                    dim["type"] = "enum"
                elif param in ("name", "country", "region"):
                    dim["values"] = ["United States", "United Kingdom", "Japan", "France", "Australia"]
                    dim["type"] = "enum"
                elif param in ("query", "search", "term"):
                    dim["values"] = ["common query", "rare query", "empty query", "very long query"]
                    dim["type"] = "enum"
                elif param in ("amount", "price", "total"):
                    dim["type"] = "numeric_range"
                    dim["range"] = [1, 1000]
                else:
                    dim["values"] = [f"sample_{param}_1", f"sample_{param}_2", f"sample_{param}_3"]
                    dim["type"] = "enum"
                dimensions.append(dim)

    return dimensions


# ============================================================================
# Generate config — LLM-powered (richer output)
# ============================================================================

def generate_config_with_llm(
    agent_description: str,
    tools: List[str],
    policy_docs: Optional[List[str]] = None,
    llm: Optional[Any] = None,
    chaos_level: str = "moderate",
) -> Dict[str, Any]:
    """
    Generate a test configuration using an LLM for richer, smarter output.

    Uses the PlanGenerator to extract rules and dimensions from policy docs,
    then merges with heuristic tool analysis for chaos config.

    Args:
        agent_description: What the agent does.
        tools: List of tool names.
        policy_docs: Policy documents for the LLM to analyze.
        llm: An LLM provider instance (Claude, OpenAI, etc.)
        chaos_level: Chaos profile level.

    Returns:
        Dict that can be saved to YAML via save_config().
    """
    # Start with heuristic config
    config = generate_config(agent_description, tools, policy_docs, chaos_level)

    if llm and policy_docs:
        try:
            from agentprobe.scenarios import PlanGenerator

            generator = PlanGenerator(llm)
            plan = generator.generate(
                agent_description=agent_description,
                policy_docs=policy_docs,
                target_scenarios=100,
            )

            # Override the heuristic plan with the LLM-generated one
            config["test_plan"]["name"] = plan.name
            config["test_plan"]["categories"] = [
                {
                    "name": cat.name,
                    "description": cat.description,
                    "count": cat.count,
                    "rules_tested": cat.rules_tested,
                    "dimensions_varied": cat.dimensions_varied,
                    "example_scenario": cat.example_scenario,
                }
                for cat in plan.categories
            ]
            config["test_plan"]["rules"] = [
                {
                    "name": r.name,
                    "description": r.description,
                    "condition": r.condition,
                    "expected_outcome": r.expected_outcome,
                    "severity": r.severity,
                }
                for r in plan.rules
            ]
            config["test_plan"]["dimensions"] = [
                {
                    "name": d.name,
                    "category": d.category,
                    "description": d.description,
                    "type": d.value_spec.type,
                    "values": d.value_spec.values,
                    "range": d.value_spec.range,
                }
                for d in plan.dimensions
            ]
            config["test_plan"]["edge_cases"] = [
                {
                    "name": ec.name,
                    "description": ec.description,
                    "category": ec.category,
                    "inject_description": ec.inject_description,
                }
                for ec in plan.edge_cases
            ]

            config["_comment"] = "Generated by agentprobe with LLM assistance. Edit to customize."

        except Exception as e:
            config["_llm_error"] = f"LLM generation failed: {e}. Falling back to heuristics."

    return config


# ============================================================================
# Save / Load
# ============================================================================

def save_config(config: Dict[str, Any], path: str) -> str:
    """
    Save a config dict to YAML.

    Args:
        config: The configuration dict from generate_config().
        path: File path (should end in .yaml or .yml).

    Returns:
        The path written to.
    """
    if not HAS_YAML:
        raise ImportError("PyYAML is required: pip install pyyaml")

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=100)

    return path


def load_config(path: str) -> tuple:
    """
    Load a YAML config and return (TestPlan, WorldSimulator).

    Args:
        path: Path to the YAML config file.

    Returns:
        Tuple of (TestPlan, WorldSimulator).
    """
    if not HAS_YAML:
        raise ImportError("PyYAML is required: pip install pyyaml")

    with open(path) as f:
        config = yaml.safe_load(f)

    plan = _config_to_plan(config)
    world = _config_to_world(config)

    return plan, world


def _config_to_plan(config: Dict) -> TestPlan:
    """Convert config dict to a TestPlan."""
    tp = config.get("test_plan", {})
    agent = config.get("agent", {})

    categories = []
    for cat in tp.get("categories", []):
        categories.append(ScenarioCategory(
            name=cat["name"],
            description=cat.get("description", ""),
            count=cat.get("count", 20),
            rules_tested=cat.get("rules_tested", []),
            dimensions_varied=cat.get("dimensions_varied", []),
            example_scenario=cat.get("example_scenario"),
        ))

    rules = []
    for r in tp.get("rules", []):
        rules.append(PolicyRule(
            name=r["name"],
            description=r.get("description", ""),
            condition=r.get("condition", ""),
            expected_outcome=r.get("expected_outcome", ""),
            severity=r.get("severity", "major"),
            source=r.get("source"),
        ))

    dimensions = []
    for d in tp.get("dimensions", []):
        spec = DimensionValue(
            type=d.get("type", "enum"),
            values=d.get("values"),
            range=d.get("range"),
        )
        dimensions.append(WorldDimension(
            name=d["name"],
            category=d.get("category", "input"),
            description=d.get("description", ""),
            value_spec=spec,
        ))

    edge_cases = []
    for ec in tp.get("edge_cases", []):
        edge_cases.append(EdgeCase(
            name=ec.get("name", ec.get("description", "edge_case")),
            description=ec.get("description", ""),
            category=ec.get("category", "edge-cases"),
            inject_description=ec.get("inject_description", ""),
        ))

    rubric = []
    for rb in tp.get("rubric", []):
        rubric.append(RubricDimension(
            dimension=rb.get("dimension", rb.get("name", "")),
            description=rb.get("description", ""),
            weight=rb.get("weight", 1.0),
        ))

    return TestPlan(
        name=tp.get("name", "evaluation"),
        agent_description=agent.get("description", ""),
        domain=agent.get("domain", "general"),
        categories=categories,
        rules=rules,
        dimensions=dimensions,
        edge_cases=edge_cases,
        rubric=rubric,
    )


# Failure mode name → class mapping
_FAILURE_MODE_MAP = {
    "timeout": lambda: Timeout(),
    "partial_data": lambda: PartialData(),
    "stale_data": lambda: StaleData(),
    "empty_response": lambda: EmptyResponse(),
    "empty": lambda: EmptyResponse(),
    "rate_limited": lambda: RateLimited(),
    "malformed_response": lambda: MalformedResponse(),
    "malformed": lambda: MalformedResponse(),
    "schema_drift": lambda: SchemaDrift(),
    "intermittent": lambda: IntermittentFailure(),
}


def _config_to_world(config: Dict) -> WorldSimulator:
    """Convert config dict to a WorldSimulator."""
    chaos = config.get("chaos", {})
    level = chaos.get("level", "moderate")

    profile_map = {
        "gentle": ChaosProfile.GENTLE,
        "moderate": ChaosProfile.MODERATE,
        "hostile": ChaosProfile.HOSTILE,
        "adversarial": ChaosProfile.ADVERSARIAL,
    }
    profile = profile_map.get(level, ChaosProfile.MODERATE)

    tools = {}
    for tool_name, tool_cfg in chaos.get("tools", {}).items():
        failure_modes = []
        for fm_name in tool_cfg.get("failure_modes", []):
            fm_name_lower = fm_name.lower().replace(" ", "_")
            factory = _FAILURE_MODE_MAP.get(fm_name_lower)
            if factory:
                failure_modes.append(factory())

        tools[tool_name] = ToolBehavior(
            description=tool_cfg.get("description", ""),
            failure_modes=failure_modes,
        )

    return WorldSimulator(tools=tools, profile=profile, seed=42)
