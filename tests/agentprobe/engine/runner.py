"""
Runner — executes an agent against a batch of Scenarios and produces ScenarioResults.

The Runner is the execution core of agentprobe's evaluation pipeline:
1. Takes a list of Scenarios (from VariationEngine)
2. For each scenario, calls the agent function with mocked tool responses
3. Captures the trace (via agentprobe's SDK decorators)
4. Evaluates the result against the scenario's rules
5. Produces a ScenarioResult for the analysis pipeline

Usage:
    from agentprobe.engine import Runner

    runner = Runner(
        agent_fn=my_agent,
        evaluator=my_evaluator,  # optional, for rule checking
    )
    results = runner.run(scenarios)
    # → List[ScenarioResult] ready for analyze_failures()
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agentprobe.analysis import RuleViolation, ScenarioResult
from agentprobe.engine.scenario import Scenario, ToolMock


# ============================================================================
# Evaluator protocol — checks whether the agent's output is correct
# ============================================================================

@dataclass
class RuleCheck:
    """One rule check result."""
    rule_name: str
    passed: bool
    severity: str = "major"
    expected: str = ""
    actual: str = ""
    details: str = ""


class Evaluator:
    """
    Base evaluator — checks agent output against scenario rules.

    Subclass this to implement domain-specific evaluation logic.
    The default implementation does basic checks on expected_action.
    """

    def evaluate(
        self,
        scenario: Scenario,
        agent_output: Any,
        agent_error: Optional[Exception] = None,
        trace: Optional[Any] = None,
    ) -> List[RuleCheck]:
        """
        Evaluate the agent's output for a given scenario.

        Args:
            scenario: The scenario that was run.
            agent_output: What the agent returned (or None if it errored).
            agent_error: The exception if the agent raised one.
            trace: The captured trace (TraceRecord), if available.

        Returns:
            List of RuleCheck results.
        """
        checks = []

        # Check: did the agent crash when it shouldn't have?
        if agent_error and scenario.expected_action != "error":
            checks.append(RuleCheck(
                rule_name="agent_no_crash",
                passed=False,
                severity="critical",
                expected="Agent should not crash",
                actual=f"Agent raised {type(agent_error).__name__}: {agent_error}",
            ))

        # Check: expected action match
        if scenario.expected_action and agent_output and not agent_error:
            actual_action = self._extract_action(agent_output)
            if actual_action and actual_action != scenario.expected_action:
                checks.append(RuleCheck(
                    rule_name="expected_action",
                    passed=False,
                    severity="major",
                    expected=scenario.expected_action,
                    actual=actual_action,
                    details=f"Expected '{scenario.expected_action}' but got '{actual_action}'",
                ))
            elif actual_action:
                checks.append(RuleCheck(
                    rule_name="expected_action",
                    passed=True,
                    expected=scenario.expected_action,
                    actual=actual_action,
                ))

        return checks

    def _extract_action(self, output: Any) -> Optional[str]:
        """Extract the action from agent output."""
        if isinstance(output, dict):
            return output.get("action")
        if isinstance(output, str):
            return output
        return None


class RuleBasedEvaluator(Evaluator):
    """
    Evaluator that runs custom rule functions.

    Register rules as callables that take (scenario, output, trace)
    and return a RuleCheck.
    """

    def __init__(self):
        self._rules: Dict[str, Callable] = {}

    def add_rule(self, name: str, check_fn: Callable[..., RuleCheck]) -> None:
        """Register a rule check function."""
        self._rules[name] = check_fn

    def evaluate(
        self,
        scenario: Scenario,
        agent_output: Any,
        agent_error: Optional[Exception] = None,
        trace: Optional[Any] = None,
    ) -> List[RuleCheck]:
        # Run base checks first
        checks = super().evaluate(scenario, agent_output, agent_error, trace)

        # Run custom rules
        for name, check_fn in self._rules.items():
            if name in scenario.rules_to_check or not scenario.rules_to_check:
                try:
                    result = check_fn(scenario, agent_output, trace)
                    if result:
                        checks.append(result)
                except Exception as e:
                    checks.append(RuleCheck(
                        rule_name=name,
                        passed=False,
                        severity="warning",
                        expected="Rule check should not crash",
                        actual=f"Rule check raised: {e}",
                    ))

        return checks


# ============================================================================
# ToolMockRegistry — intercepts tool calls with scenario-defined responses
# ============================================================================

class ToolMockRegistry:
    """
    Provides mock tool responses for a specific scenario.

    The Runner sets this before each agent invocation. Tool functions
    can check the registry to get their mock response instead of
    calling the real service.
    """

    def __init__(self, mocks: Optional[List[ToolMock]] = None):
        self._mocks: Dict[str, ToolMock] = {}
        if mocks:
            for m in mocks:
                self._mocks[m.tool_name] = m

    def get_response(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get the mock response for a tool, or None if not mocked."""
        mock = self._mocks.get(tool_name)
        if mock:
            return mock.response
        return None

    def get_behavior(self, tool_name: str) -> str:
        """Get the behavior for a tool (normal, timeout, etc.)."""
        mock = self._mocks.get(tool_name)
        return mock.behavior if mock else "normal"

    def has_failure(self, tool_name: str) -> bool:
        mock = self._mocks.get(tool_name)
        return mock is not None and mock.behavior != "normal"


# Global registry — set by Runner before each agent call
_current_mock_registry: Optional[ToolMockRegistry] = None


def get_mock_registry() -> Optional[ToolMockRegistry]:
    """Get the current mock registry (used by tool functions)."""
    return _current_mock_registry


def mock_tool_response(tool_name: str) -> Optional[Dict[str, Any]]:
    """
    Get the mock response for a tool in the current scenario.

    Call this at the top of your tool function to use mocked data:

        @trace_tool("order-lookup")
        def lookup_order(order_id):
            mock = mock_tool_response("order-lookup")
            if mock is not None:
                return mock
            # ... real implementation ...
    """
    if _current_mock_registry:
        return _current_mock_registry.get_response(tool_name)
    return None


# ============================================================================
# Runner
# ============================================================================

@dataclass
class RunResult:
    """Result of running one scenario."""
    scenario: Scenario
    output: Any = None
    error: Optional[Exception] = None
    trace: Optional[Any] = None
    duration_ms: float = 0.0
    checks: List[RuleCheck] = field(default_factory=list)


class Runner:
    """
    Executes an agent against a batch of Scenarios.

    For each scenario:
    1. Sets up tool mocks from the scenario
    2. Calls the agent function
    3. Captures the output (or error)
    4. Runs the evaluator
    5. Produces a ScenarioResult

    The agent function should be decorated with @trace_agent so traces
    are captured automatically.
    """

    def __init__(
        self,
        agent_fn: Callable,
        evaluator: Optional[Evaluator] = None,
        probe: Optional[Any] = None,  # AgentProbe instance
        input_builder: Optional[Callable] = None,
    ):
        self.agent_fn = agent_fn
        self.evaluator = evaluator or Evaluator()
        self.probe = probe
        self._input_builder = input_builder

    def run(
        self,
        scenarios: List[Scenario],
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[ScenarioResult]:
        """
        Run the agent against all scenarios.

        Args:
            scenarios: List of Scenarios to run.
            on_progress: Optional callback(completed, total) for progress tracking.

        Returns:
            List of ScenarioResults ready for analyze_failures().
        """
        global _current_mock_registry
        results = []

        for i, scenario in enumerate(scenarios):
            # Set up tool mocks for this scenario
            _current_mock_registry = ToolMockRegistry(scenario.tool_mocks)

            run_result = self._run_one(scenario)

            # Evaluate
            checks = self.evaluator.evaluate(
                scenario=scenario,
                agent_output=run_result.output,
                agent_error=run_result.error,
                trace=run_result.trace,
            )
            run_result.checks = checks

            # Convert to ScenarioResult for the analysis pipeline
            violations = [
                RuleViolation(
                    rule_name=c.rule_name,
                    severity=c.severity,
                    expected=c.expected,
                    actual=c.actual,
                    details=c.details,
                )
                for c in checks if not c.passed
            ]

            passed = len(violations) == 0 and run_result.error is None

            scenario_result = ScenarioResult(
                scenario_id=scenario.scenario_id,
                passed=passed,
                scenario=scenario.variables,
                world_config={
                    "has_failures": scenario.has_tool_failures,
                    "injected_failures": scenario.injected_failures,
                },
                violations=violations,
                category=scenario.category,
                metadata={
                    "duration_ms": run_result.duration_ms,
                    "customer_message": scenario.customer_message,
                    "difficulty": scenario.difficulty,
                    "expected_action": scenario.expected_action,
                },
            )
            results.append(scenario_result)

            if on_progress:
                on_progress(i + 1, len(scenarios))

        # Clean up
        _current_mock_registry = None

        return results

    def _run_one(self, scenario: Scenario) -> RunResult:
        """Run the agent against a single scenario."""
        result = RunResult(scenario=scenario)

        # Build the agent input from the scenario
        agent_input = self._build_agent_input(scenario)

        start = time.time()
        try:
            if isinstance(agent_input, tuple):
                output = self.agent_fn(*agent_input)
            elif isinstance(agent_input, dict):
                output = self.agent_fn(**agent_input)
            else:
                output = self.agent_fn(agent_input)
            result.output = output
        except Exception as e:
            result.error = e
        finally:
            result.duration_ms = (time.time() - start) * 1000

        # Capture the trace if probe is available
        if self.probe and self.probe.traces:
            result.trace = self.probe.traces[-1]

        return result

    def _build_agent_input(self, scenario: Scenario) -> Any:
        """Build the input arguments for the agent function."""
        if self._input_builder:
            return self._input_builder(scenario)

        # Default: pass message and order_id as positional args
        variables = scenario.variables
        order_id = variables.get("order_id", f"ORD-{hash(scenario.scenario_id) % 10000:04d}")

        return (scenario.customer_message, order_id)

    def run_one(self, scenario: Scenario) -> ScenarioResult:
        """Run a single scenario and return the result."""
        return self.run([scenario])[0]
