"""
agentprobe.engine — The evaluation execution pipeline.

Connects scenario generation (TestPlan) with chaos engineering (WorldSimulator)
and agent execution to produce ScenarioResults for analysis.

Components:
    Scenario          — A concrete, runnable test case
    VariationEngine   — Produces Scenarios from TestPlan + WorldSimulator
    Runner            — Executes agent against Scenarios, produces ScenarioResults
    Evaluator         — Checks agent output against rules
    ToolCallValidator — Validates agent's outbound tool call schemas
"""

from agentprobe.engine.scenario import Scenario, ToolMock
from agentprobe.engine.variation import VariationEngine
from agentprobe.engine.runner import (
    Runner,
    Evaluator,
    RuleBasedEvaluator,
    RuleCheck,
    ToolMockRegistry,
    mock_tool_response,
    get_mock_registry,
)
from agentprobe.engine.tool_validation import (
    ToolCallValidator,
    ToolCallViolation,
    ToolSchema,
)

__all__ = [
    "Scenario",
    "ToolMock",
    "VariationEngine",
    "Runner",
    "Evaluator",
    "RuleBasedEvaluator",
    "RuleCheck",
    "ToolMockRegistry",
    "mock_tool_response",
    "get_mock_registry",
    "ToolCallValidator",
    "ToolCallViolation",
    "ToolSchema",
]
