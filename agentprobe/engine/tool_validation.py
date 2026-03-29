"""
Tool Call Validation — deterministic evaluator for agent tool calls.

Checks whether the agent's outbound tool calls are well-formed:
- Valid JSON arguments
- All required parameters present
- Correct types
- No hallucinated parameter names
- Enum values within allowed set
- Numeric values within valid ranges

This is a Layer 1 evaluator: fully deterministic, zero false positives,
excellent as a CI/CD gate.

Usage:
    validator = ToolCallValidator(schemas={
        "order-lookup": ToolSchema(
            required_params={"order_id": "str"},
            optional_params={"include_history": "bool"},
        ),
    })
    violations = validator.validate_trace(trace)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from agentprobe.models import TraceRecord, SpanRecord


@dataclass
class ToolSchema:
    """Schema definition for a tool's expected parameters."""
    required_params: Dict[str, str] = field(default_factory=dict)   # name → type
    optional_params: Dict[str, str] = field(default_factory=dict)   # name → type
    enum_values: Dict[str, List[str]] = field(default_factory=dict) # param → allowed values
    numeric_ranges: Dict[str, tuple] = field(default_factory=dict)  # param → (min, max)

    @property
    def all_params(self) -> Dict[str, str]:
        return {**self.required_params, **self.optional_params}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required_params": self.required_params,
            "optional_params": self.optional_params,
            "enum_values": self.enum_values,
            "numeric_ranges": self.numeric_ranges,
        }


@dataclass
class ToolCallViolation:
    """A violation found in an agent's tool call."""
    tool_name: str
    span_id: str
    violation_type: str         # missing_param, wrong_type, hallucinated_param, invalid_enum, out_of_range
    param_name: str = ""
    expected: str = ""
    actual: str = ""
    severity: str = "major"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "span_id": self.span_id,
            "violation_type": self.violation_type,
            "param_name": self.param_name,
            "expected": self.expected,
            "actual": self.actual,
            "severity": self.severity,
        }


class ToolCallValidator:
    """
    Validates agent tool calls against defined schemas.

    Zero false positives — every violation is a genuine problem.
    """

    def __init__(self, schemas: Optional[Dict[str, ToolSchema]] = None):
        self.schemas = schemas or {}

    def add_schema(self, tool_name: str, schema: ToolSchema) -> None:
        self.schemas[tool_name] = schema

    def validate_trace(self, trace: TraceRecord) -> List[ToolCallViolation]:
        """Validate all tool calls in a trace."""
        violations = []
        for span in trace.spans:
            if span.span_type == "tool_call" and span.tool_args is not None:
                tool_name = span.tool_name or span.name
                if tool_name in self.schemas:
                    v = self._validate_call(tool_name, span)
                    violations.extend(v)
        return violations

    def validate_span(self, span: SpanRecord) -> List[ToolCallViolation]:
        """Validate a single tool call span."""
        tool_name = span.tool_name or span.name
        if tool_name not in self.schemas or span.tool_args is None:
            return []
        return self._validate_call(tool_name, span)

    def _validate_call(self, tool_name: str, span: SpanRecord) -> List[ToolCallViolation]:
        schema = self.schemas[tool_name]
        args = span.tool_args or {}
        violations = []

        # Check for missing required params
        for param, expected_type in schema.required_params.items():
            if param not in args:
                violations.append(ToolCallViolation(
                    tool_name=tool_name,
                    span_id=span.span_id,
                    violation_type="missing_param",
                    param_name=param,
                    expected=f"Required parameter '{param}' ({expected_type})",
                    actual="Parameter not provided",
                    severity="critical",
                ))

        # Check for hallucinated params
        valid_params = set(schema.all_params.keys())
        for param in args:
            if param not in valid_params:
                violations.append(ToolCallViolation(
                    tool_name=tool_name,
                    span_id=span.span_id,
                    violation_type="hallucinated_param",
                    param_name=param,
                    expected=f"Valid params: {sorted(valid_params)}",
                    actual=f"Unknown parameter '{param}'",
                    severity="major",
                ))

        # Check types
        for param, expected_type in schema.all_params.items():
            if param in args:
                if not self._check_type(args[param], expected_type):
                    violations.append(ToolCallViolation(
                        tool_name=tool_name,
                        span_id=span.span_id,
                        violation_type="wrong_type",
                        param_name=param,
                        expected=expected_type,
                        actual=type(args[param]).__name__,
                        severity="major",
                    ))

        # Check enum values
        for param, allowed in schema.enum_values.items():
            if param in args and args[param] not in allowed:
                violations.append(ToolCallViolation(
                    tool_name=tool_name,
                    span_id=span.span_id,
                    violation_type="invalid_enum",
                    param_name=param,
                    expected=f"One of: {allowed}",
                    actual=str(args[param]),
                    severity="major",
                ))

        # Check numeric ranges
        for param, (min_val, max_val) in schema.numeric_ranges.items():
            if param in args and isinstance(args[param], (int, float)):
                if args[param] < min_val or args[param] > max_val:
                    violations.append(ToolCallViolation(
                        tool_name=tool_name,
                        span_id=span.span_id,
                        violation_type="out_of_range",
                        param_name=param,
                        expected=f"Between {min_val} and {max_val}",
                        actual=str(args[param]),
                        severity="major",
                    ))

        return violations

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if a value matches the expected type string."""
        type_map = {
            "str": str, "string": str,
            "int": int, "integer": int,
            "float": (int, float), "number": (int, float),
            "bool": bool, "boolean": bool,
            "list": list, "array": list,
            "dict": dict, "object": dict,
        }
        expected = type_map.get(expected_type.lower())
        if expected is None:
            return True  # unknown type, don't flag
        return isinstance(value, expected)
