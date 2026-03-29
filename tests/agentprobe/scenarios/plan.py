"""
TestPlan — the structured representation of a test plan.

This is the bridge between the plan generator (which creates it from docs)
and the variation engine (which generates scenarios from it). It's also
what the user reviews and edits.

Design principle: the TestPlan is a readable, editable document — not an
opaque data structure. Every field has a human-readable description.
The render() method produces text the user can review with their domain expert.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class DimensionValue:
    """A possible value or range for a world state dimension."""
    type: str                                   # string, int, float, bool, enum
    values: Optional[List[str]] = None          # for enum type
    range: Optional[List[float]] = None         # [min, max] for numeric types
    description: Optional[str] = None           # human-readable explanation

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": self.type}
        if self.values:
            d["values"] = self.values
        if self.range:
            d["range"] = self.range
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class WorldDimension:
    """
    A variable in the world state that affects agent behavior.

    Example: "days_since_delivery" with type=int, range=[0, 90],
    description="Days since the order was delivered to the customer"
    """
    name: str
    category: str                               # grouping (e.g., "order", "customer", "product")
    description: str
    value_spec: DimensionValue
    affects: Optional[List[str]] = None         # which rules this dimension triggers
    depends_on: Optional[str] = None            # conditional dependency description

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "value_spec": self.value_spec.to_dict(),
        }
        if self.affects:
            d["affects"] = self.affects
        if self.depends_on:
            d["depends_on"] = self.depends_on
        return d


@dataclass
class PolicyRule:
    """
    A rule the agent must follow, extracted from policy documents.

    Example: "Deny returns outside 30-day window"
    - condition: "days_since_delivery > 30"
    - expected_outcome: "agent denies the return"
    - severity: "critical"
    - source: "Return Policy, Section 3.1"
    """
    name: str
    description: str
    condition: str                              # human-readable condition
    expected_outcome: str                       # what agent should do
    severity: str                               # critical, major, warning, info
    source: Optional[str] = None                # where in the docs this came from

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "description": self.description,
            "condition": self.condition,
            "expected_outcome": self.expected_outcome,
            "severity": self.severity,
        }
        if self.source:
            d["source"] = self.source
        return d


@dataclass
class ScenarioCategory:
    """
    A category of test scenarios.

    Example: "Return Window Enforcement" — tests whether the agent correctly
    enforces the 30-day return window across different order ages.
    """
    name: str
    description: str
    count: int                                  # suggested number of scenarios
    rules_tested: List[str]                     # names of PolicyRules this category exercises
    dimensions_varied: List[str]                # names of WorldDimensions this category varies
    difficulty_mix: Optional[Dict[str, float]] = None  # e.g., {"easy": 0.4, "medium": 0.4, "hard": 0.2}
    example_scenario: Optional[str] = None      # one concrete example for the user to review

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "description": self.description,
            "count": self.count,
            "rules_tested": self.rules_tested,
            "dimensions_varied": self.dimensions_varied,
        }
        if self.difficulty_mix:
            d["difficulty_mix"] = self.difficulty_mix
        if self.example_scenario:
            d["example_scenario"] = self.example_scenario
        return d


@dataclass
class EdgeCase:
    """
    A specific edge case or adversarial scenario to inject.

    These are hand-authored (by the LLM during plan generation, or by the user)
    to ensure important edge cases are always tested.
    """
    name: str
    description: str
    category: str                               # which ScenarioCategory this belongs to
    inject_description: str                     # what to modify in the scenario

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "inject_description": self.inject_description,
        }


@dataclass
class RubricDimension:
    """
    A dimension for LLM-as-judge evaluation.

    Example: "policy_compliance" — "Did the agent follow all applicable policies?"
    with weight 0.3 (most important dimension).
    """
    dimension: str
    description: str
    weight: float                               # 0.0 to 1.0, weights should sum to ~1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "description": self.description,
            "weight": self.weight,
        }


@dataclass
class TestPlan:
    """
    A complete test plan for an AI agent.

    Generated by PlanGenerator from policy documents and agent description.
    Reviewed and edited by the user. Consumed by the variation engine to
    generate concrete scenarios.
    """
    __test__ = False  # prevent pytest collection
    # Metadata
    name: str
    agent_description: str
    domain: str                                 # e.g., "customer-support", "insurance-underwriting"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0"

    # The plan content
    categories: List[ScenarioCategory] = field(default_factory=list)
    rules: List[PolicyRule] = field(default_factory=list)
    dimensions: List[WorldDimension] = field(default_factory=list)
    edge_cases: List[EdgeCase] = field(default_factory=list)
    rubric: List[RubricDimension] = field(default_factory=list)

    # ----------------------------------------------------------------
    # Summary stats
    # ----------------------------------------------------------------

    @property
    def total_scenarios(self) -> int:
        return sum(c.count for c in self.categories)

    @property
    def rule_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for rule in self.rules:
            counts[rule.severity] = counts.get(rule.severity, 0) + 1
        return counts

    # ----------------------------------------------------------------
    # Human-readable output
    # ----------------------------------------------------------------

    def summary(self) -> str:
        """One-line summary of the plan."""
        rc = self.rule_counts
        return (
            f"TestPlan '{self.name}': {len(self.categories)} categories, "
            f"{self.total_scenarios} scenarios, "
            f"{len(self.rules)} rules ({rc.get('critical', 0)} critical), "
            f"{len(self.dimensions)} dimensions, "
            f"{len(self.edge_cases)} edge cases"
        )

    def render(self) -> str:
        """
        Render the full test plan as human-readable text.

        This is what the user reviews with their domain expert. It should be
        clear enough that a non-technical person can understand what's being
        tested and flag anything wrong or missing.
        """
        lines = []

        # Header
        lines.append("=" * 60)
        lines.append(f"  TEST PLAN: {self.name}")
        lines.append(f"  Domain: {self.domain}")
        lines.append(f"  Created: {self.created_at[:10]}")
        lines.append("=" * 60)
        lines.append("")

        # Agent description
        lines.append("AGENT DESCRIPTION:")
        lines.append(f"  {self.agent_description}")
        lines.append("")

        # Scenario categories
        lines.append(f"SCENARIO CATEGORIES ({len(self.categories)} categories, "
                      f"{self.total_scenarios} total scenarios):")
        lines.append("")
        for i, cat in enumerate(self.categories, 1):
            lines.append(f"  {i}. {cat.name} ({cat.count} scenarios)")
            lines.append(f"     {cat.description}")
            if cat.rules_tested:
                lines.append(f"     Rules tested: {', '.join(cat.rules_tested)}")
            if cat.example_scenario:
                lines.append(f"     Example: \"{cat.example_scenario}\"")
            lines.append("")

        # Policy rules
        lines.append(f"POLICY RULES ({len(self.rules)} rules):")
        lines.append("")

        for severity in ["critical", "major", "warning", "info"]:
            severity_rules = [r for r in self.rules if r.severity == severity]
            if severity_rules:
                icon = {"critical": "🔴", "major": "🟠", "warning": "🟡", "info": "ℹ️"}.get(severity, "•")
                lines.append(f"  {icon} {severity.upper()} ({len(severity_rules)}):")
                for rule in severity_rules:
                    lines.append(f"     • {rule.description}")
                    lines.append(f"       When: {rule.condition}")
                    lines.append(f"       Expect: {rule.expected_outcome}")
                    if rule.source:
                        lines.append(f"       Source: {rule.source}")
                lines.append("")

        # World dimensions
        lines.append(f"SCENARIO DIMENSIONS ({len(self.dimensions)} variables):")
        lines.append("")
        categories_seen: Dict[str, List[WorldDimension]] = {}
        for dim in self.dimensions:
            categories_seen.setdefault(dim.category, []).append(dim)
        for cat_name, dims in categories_seen.items():
            lines.append(f"  {cat_name.upper()}:")
            for dim in dims:
                spec = dim.value_spec
                if spec.values:
                    range_str = f"[{', '.join(spec.values)}]"
                elif spec.range:
                    range_str = f"{spec.range[0]} – {spec.range[1]}"
                else:
                    range_str = spec.type
                lines.append(f"    • {dim.name} ({range_str})")
                lines.append(f"      {dim.description}")
            lines.append("")

        # Edge cases
        if self.edge_cases:
            lines.append(f"EDGE CASES ({len(self.edge_cases)}):")
            lines.append("")
            for ec in self.edge_cases:
                lines.append(f"  ⚡ {ec.name}")
                lines.append(f"     {ec.description}")
            lines.append("")

        # Evaluation rubric
        if self.rubric:
            lines.append("EVALUATION RUBRIC:")
            lines.append("")
            for dim in self.rubric:
                pct = int(dim.weight * 100)
                lines.append(f"  • {dim.dimension} ({pct}%): {dim.description}")
            lines.append("")

        # Footer
        lines.append("─" * 60)
        lines.append(f"  Total: {self.total_scenarios} scenarios from "
                      f"{len(self.categories)} categories")
        lines.append(f"  Evaluating against {len(self.rules)} policy rules")
        lines.append("─" * 60)

        return "\n".join(lines)

    # ----------------------------------------------------------------
    # Serialization
    # ----------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "agent_description": self.agent_description,
            "domain": self.domain,
            "created_at": self.created_at,
            "version": self.version,
            "categories": [c.to_dict() for c in self.categories],
            "rules": [r.to_dict() for r in self.rules],
            "dimensions": [d.to_dict() for d in self.dimensions],
            "edge_cases": [e.to_dict() for e in self.edge_cases],
            "rubric": [r.to_dict() for r in self.rubric],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TestPlan:
        plan = cls(
            name=data["name"],
            agent_description=data["agent_description"],
            domain=data["domain"],
            created_at=data.get("created_at", ""),
            version=data.get("version", "1.0"),
        )

        for c in data.get("categories", []):
            plan.categories.append(ScenarioCategory(
                name=c["name"],
                description=c["description"],
                count=c["count"],
                rules_tested=c.get("rules_tested", []),
                dimensions_varied=c.get("dimensions_varied", []),
                difficulty_mix=c.get("difficulty_mix"),
                example_scenario=c.get("example_scenario"),
            ))

        for r in data.get("rules", []):
            plan.rules.append(PolicyRule(
                name=r["name"],
                description=r["description"],
                condition=r["condition"],
                expected_outcome=r["expected_outcome"],
                severity=r["severity"],
                source=r.get("source"),
            ))

        for d in data.get("dimensions", []):
            vs = d["value_spec"]
            plan.dimensions.append(WorldDimension(
                name=d["name"],
                category=d["category"],
                description=d["description"],
                value_spec=DimensionValue(
                    type=vs["type"],
                    values=vs.get("values"),
                    range=vs.get("range"),
                    description=vs.get("description"),
                ),
                affects=d.get("affects"),
                depends_on=d.get("depends_on"),
            ))

        for e in data.get("edge_cases", []):
            plan.edge_cases.append(EdgeCase(
                name=e["name"],
                description=e["description"],
                category=e["category"],
                inject_description=e["inject_description"],
            ))

        for rb in data.get("rubric", []):
            plan.rubric.append(RubricDimension(
                dimension=rb["dimension"],
                description=rb["description"],
                weight=rb["weight"],
            ))

        return plan

    @classmethod
    def from_json(cls, json_str: str) -> TestPlan:
        return cls.from_dict(json.loads(json_str))
