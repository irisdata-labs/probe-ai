"""
Scenario — a concrete, runnable test case.

A Scenario is what the VariationEngine produces from a TestPlan + WorldConfiguration.
It contains everything the Runner needs to execute one test:
- The customer message / input
- The world variables (days_since_delivery, price, etc.)
- The tool behaviors (normal, timeout, stale, etc.)
- The rules that should be checked
- The expected behavior

The Runner takes a Scenario, runs the agent, captures a trace, and
produces a ScenarioResult for the analysis pipeline.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolMock:
    """Defines how a tool should respond in this scenario."""
    tool_name: str
    behavior: str = "normal"          # normal, timeout, partial_data, stale_data, etc.
    response: Optional[Dict[str, Any]] = None   # what the tool returns (if normal or modified)
    failure_details: Optional[Dict[str, Any]] = None  # details about the failure mode

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "tool_name": self.tool_name,
            "behavior": self.behavior,
        }
        if self.response is not None:
            d["response"] = self.response
        if self.failure_details:
            d["failure_details"] = self.failure_details
        return d


@dataclass
class Scenario:
    """
    A concrete, runnable test case.

    Produced by the VariationEngine, consumed by the Runner.
    """
    scenario_id: str = field(default_factory=lambda: f"scn_{uuid.uuid4().hex[:8]}")
    category: str = ""                    # which ScenarioCategory this belongs to
    description: str = ""                 # human-readable description

    # The input to the agent
    customer_message: str = ""
    variables: Dict[str, Any] = field(default_factory=dict)

    # Tool behaviors for this scenario
    tool_mocks: List[ToolMock] = field(default_factory=list)

    # Which rules should be checked
    rules_to_check: List[str] = field(default_factory=list)

    # Expected outcome (for evaluation)
    expected_action: Optional[str] = None     # e.g. "approve", "deny", "escalate"
    expected_rules_violated: List[str] = field(default_factory=list)

    # Difficulty
    difficulty: str = "normal"            # easy, normal, hard, adversarial

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_failures(self) -> bool:
        return any(tm.behavior != "normal" for tm in self.tool_mocks)

    @property
    def injected_failures(self) -> List[str]:
        return [f"{tm.tool_name}:{tm.behavior}" for tm in self.tool_mocks
                if tm.behavior != "normal"]

    def get_tool_mock(self, tool_name: str) -> Optional[ToolMock]:
        """Get the mock for a specific tool."""
        for tm in self.tool_mocks:
            if tm.tool_name == tool_name:
                return tm
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "category": self.category,
            "description": self.description,
            "customer_message": self.customer_message,
            "variables": self.variables,
            "tool_mocks": [tm.to_dict() for tm in self.tool_mocks],
            "rules_to_check": self.rules_to_check,
            "expected_action": self.expected_action,
            "difficulty": self.difficulty,
            "has_tool_failures": self.has_tool_failures,
            "injected_failures": self.injected_failures,
            "metadata": self.metadata,
        }
