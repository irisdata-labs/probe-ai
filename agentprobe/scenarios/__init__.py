"""
agentprobe.scenarios — Scenario generation and test plan management.

Usage:
    from agentprobe.scenarios import TestPlan, PlanGenerator

    # Generate a test plan from documents
    generator = PlanGenerator(llm=Claude())
    plan = generator.generate(
        agent_description="A customer support agent that handles returns...",
        policy_docs=["Return policy: 30 day window...", "Escalation rules: ..."],
    )

    # Review the plan
    print(plan.summary())
    print(plan.render())

    # Edit conversationally
    plan = generator.refine(plan, "The escalation threshold is $3M not $5M")
"""

from agentprobe.scenarios.plan import (
    TestPlan,
    ScenarioCategory,
    PolicyRule,
    WorldDimension,
    DimensionValue,
    EdgeCase,
    RubricDimension,
)
from agentprobe.scenarios.plan_generator import PlanGenerator

__all__ = [
    "TestPlan",
    "ScenarioCategory",
    "PolicyRule",
    "WorldDimension",
    "DimensionValue",
    "EdgeCase",
    "RubricDimension",
    "PlanGenerator",
]
