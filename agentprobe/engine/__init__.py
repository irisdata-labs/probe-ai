"""
agentprobe.engine — The evaluation execution pipeline.

Components:
    Scenario          — A concrete, runnable test case
    VariationEngine   — Produces Scenarios from TestPlan + WorldSimulator
    Runner            — Executes agent against Scenarios, produces ScenarioResults
    Evaluator         — Checks agent output against rules
    ContentEvaluator  — Checks response content quality
    ToolCallValidator — Validates agent's outbound tool call schemas
    DifficultyScorer  — Rates scenario difficulty
    EvaluationReport  — Structured report with verdicts and analysis
    export_html       — Renders report as HTML file
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
from agentprobe.engine.content_checks import ContentEvaluator, ContentCheckConfig
from agentprobe.engine.difficulty import DifficultyScorer, DifficultyScore
from agentprobe.engine.report import EvaluationReport
from agentprobe.engine.export import export_html

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
    "ContentEvaluator",
    "ContentCheckConfig",
    "DifficultyScorer",
    "DifficultyScore",
    "EvaluationReport",
    "export_html",
]
