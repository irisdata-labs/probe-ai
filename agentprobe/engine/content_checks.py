"""
Content quality checks for agent responses.

These are deterministic checks that verify the *content* of the agent's
response, not just the action. They catch issues the base Evaluator misses:

- Agent returns "complete" but the brief is empty
- Agent includes data that no tool returned (hallucination)
- Agent omits data from tools that succeeded
- Agent returns a placeholder/error message as the brief

Usage:
    evaluator = ContentEvaluator(
        required_fields=["brief", "temperature_c", "currency"],
        tool_output_fields={
            "get-weather": ["temperature_c"],
            "get-country-info": ["currency"],
        },
    )
    checks = evaluator.evaluate(scenario, agent_output, trace=trace)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from agentprobe.engine.runner import Evaluator, RuleCheck
from agentprobe.engine.scenario import Scenario


@dataclass
class ContentCheckConfig:
    """Configuration for content quality checks."""
    # Fields that must be present and non-empty in the output when action="complete"
    required_fields: List[str] = field(default_factory=list)

    # Mapping: tool_name → fields in the output that should come from that tool
    # If the tool succeeded, these fields should be present
    tool_output_fields: Dict[str, List[str]] = field(default_factory=dict)

    # Strings that indicate a placeholder/error response (case-insensitive)
    placeholder_patterns: List[str] = field(default_factory=lambda: [
        "[brief unavailable",
        "n/a",
        "no data available",
        "error",
        "failed to",
        "could not",
    ])

    # Minimum length for text fields (e.g., "brief" should be > 20 chars)
    min_text_length: Dict[str, int] = field(default_factory=dict)


class ContentEvaluator(Evaluator):
    """
    Evaluator that checks response content quality.

    Extends the base Evaluator with content-specific checks:
    1. Required fields present when action="complete"
    2. Tool output data included when tools succeeded
    3. No placeholder/error text in completed responses
    4. Minimum text length for key fields
    5. No hallucinated data (fields present that no tool returned)
    """

    def __init__(self, config: Optional[ContentCheckConfig] = None, **kwargs):
        self.config = config or ContentCheckConfig()
        # Also accept direct kwargs for convenience
        if "required_fields" in kwargs:
            self.config.required_fields = kwargs["required_fields"]
        if "tool_output_fields" in kwargs:
            self.config.tool_output_fields = kwargs["tool_output_fields"]
        if "min_text_length" in kwargs:
            self.config.min_text_length = kwargs["min_text_length"]

    def evaluate(
        self,
        scenario: Scenario,
        agent_output: Any,
        agent_error: Optional[Exception] = None,
        trace: Optional[Any] = None,
    ) -> List[RuleCheck]:
        # Run base checks first
        checks = super().evaluate(scenario, agent_output, agent_error, trace)

        # Only run content checks if agent completed successfully
        if agent_error or not isinstance(agent_output, dict):
            return checks

        action = agent_output.get("action", "")
        if action != "complete":
            return checks

        # Check 1: required fields (skip fields from failed tools)
        checks.extend(self._check_required_fields(agent_output))

        # Check 2: tool output data present
        checks.extend(self._check_tool_outputs(scenario, agent_output))

        # Checks 3 & 4 only apply when tools returned usable data.
        # If tools had real outages (timeout, empty), placeholder text is expected.
        # But stale_data/partial_data still returned data — quality checks apply.
        if not scenario.has_tool_outages:
            # Check 3: no placeholder text
            checks.extend(self._check_no_placeholders(agent_output))

            # Check 4: minimum text length
            checks.extend(self._check_text_length(agent_output))

        return checks

    def _check_required_fields(self, output: Dict) -> List[RuleCheck]:
        checks = []
        for field_name in self.config.required_fields:
            value = output.get(field_name)
            if value is None or value == "" or value == []:
                checks.append(RuleCheck(
                    rule_name="required_field_missing",
                    passed=False,
                    severity="major",
                    expected=f"Field '{field_name}' should be present and non-empty",
                    actual=f"Field '{field_name}' is {repr(value)}",
                ))
        return checks

    def _check_tool_outputs(self, scenario: Scenario, output: Dict) -> List[RuleCheck]:
        """Check that data from successful tools appears in the output."""
        checks = []
        for tool_name, expected_fields in self.config.tool_output_fields.items():
            # Was this tool mocked as working (not failed)?
            tool_mock = scenario.get_tool_mock(tool_name)
            tool_failed = (tool_mock and tool_mock.behavior != "normal") if tool_mock else False

            if not tool_failed:
                # Tool succeeded — its data should be in the output
                for field_name in expected_fields:
                    value = output.get(field_name)
                    if value is None:
                        checks.append(RuleCheck(
                            rule_name="tool_data_missing",
                            passed=False,
                            severity="major",
                            expected=f"'{field_name}' from {tool_name} should be in output (tool succeeded)",
                            actual=f"Field '{field_name}' is missing",
                        ))
        return checks

    def _check_no_placeholders(self, output: Dict) -> List[RuleCheck]:
        """Check that text fields don't contain placeholder/error text."""
        checks = []
        for key, value in output.items():
            if isinstance(value, str) and len(value) > 0:
                value_lower = value.lower()
                for pattern in self.config.placeholder_patterns:
                    if pattern.lower() in value_lower:
                        checks.append(RuleCheck(
                            rule_name="placeholder_response",
                            passed=False,
                            severity="major",
                            expected=f"Field '{key}' should contain real data",
                            actual=f"Field '{key}' contains placeholder: '{value[:80]}'",
                        ))
                        break  # one violation per field
        return checks

    def _check_text_length(self, output: Dict) -> List[RuleCheck]:
        """Check minimum text length for key fields."""
        checks = []
        for field_name, min_len in self.config.min_text_length.items():
            value = output.get(field_name, "")
            if isinstance(value, str) and len(value) < min_len:
                checks.append(RuleCheck(
                    rule_name="response_too_short",
                    passed=False,
                    severity="warning",
                    expected=f"Field '{field_name}' should be at least {min_len} chars",
                    actual=f"Field '{field_name}' is {len(value)} chars",
                ))
        return checks
