"""
VariationEngine — produces runnable Scenarios from a TestPlan + WorldSimulator.

This is the connective tissue between:
- TestPlan (what to test: rules, dimensions, categories)
- WorldSimulator (how tools break: timeouts, stale data, contradictions)
- Scenario (concrete runnable test case)

The engine generates scenarios using three strategies:
1. Parametric: sweep dimension values across their ranges
2. LLM-powered: generate realistic combinations (requires LLM provider)
3. Chaos: overlay WorldSimulator failures onto scenarios

Usage:
    from agentprobe.engine import VariationEngine
    from agentprobe.scenarios import TestPlan
    from agentprobe.chaos import WorldSimulator, ChaosProfile

    engine = VariationEngine(
        plan=test_plan,
        world=WorldSimulator.from_tool_names(["order-lookup", "process-return"],
                                              profile=ChaosProfile.MODERATE),
    )
    scenarios = engine.generate(n=200)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from agentprobe.engine.scenario import Scenario, ToolMock
from agentprobe.scenarios.plan import (
    DimensionValue,
    EdgeCase,
    PolicyRule,
    ScenarioCategory,
    TestPlan,
    WorldDimension,
)


@dataclass
class VariationEngine:
    """
    Generates concrete Scenarios from a TestPlan and optional WorldSimulator.

    The engine doesn't need an LLM — it uses the TestPlan's dimensions
    and categories to produce parametric variations. An LLM can optionally
    be used to generate more realistic customer messages.
    """
    plan: TestPlan
    world: Optional[Any] = None   # WorldSimulator, optional
    seed: Optional[int] = None
    message_generator: Optional[Callable] = None  # optional LLM-based message gen

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def generate(
        self,
        n: int = 200,
        chaos_ratio: float = 0.3,
    ) -> List[Scenario]:
        """
        Generate n scenarios.

        Args:
            n: Total number of scenarios to generate.
            chaos_ratio: Fraction of scenarios with tool failures (0.0 to 1.0).
                         Only applies if a WorldSimulator is configured.

        Returns:
            List of concrete, runnable Scenarios.
        """
        scenarios = []

        # Reserve slots for edge cases within the budget
        edge_scenarios = self._generate_edge_cases()
        n_edges = min(len(edge_scenarios), max(1, n // 4))  # at most 25% of budget
        n_categories = n - n_edges

        # Distribute remaining scenarios across categories proportionally
        category_counts = self._distribute_across_categories(n_categories)

        for category, count in category_counts:
            for i in range(count):
                # Generate variables from dimensions
                variables = self._sample_dimensions(category)

                # Generate the customer message
                message = self._generate_message(category, variables)

                # Determine which rules apply
                rules = self._rules_for_category(category)

                # Determine expected action based on variables and rules
                expected = self._infer_expected_action(category, variables, rules)

                # Determine difficulty
                difficulty = self._sample_difficulty(category)

                # Build tool mocks
                tool_mocks = self._build_tool_mocks(variables)

                # Apply chaos based on category
                cat_name_lower = category.name.lower()
                if self.world and "adversarial" in cat_name_lower:
                    # Adversarial: ALWAYS inject chaos, multiple tools fail
                    tool_mocks = self._apply_multi_chaos(tool_mocks)
                    difficulty = "adversarial"
                elif self.world and self._rng.random() < chaos_ratio:
                    tool_mocks = self._apply_chaos(tool_mocks)

                scenario = Scenario(
                    category=category.name,
                    description=f"{category.name}: {message[:60]}...",
                    customer_message=message,
                    variables=variables,
                    tool_mocks=tool_mocks,
                    rules_to_check=[r.name for r in rules],
                    expected_action=expected,
                    difficulty=difficulty,
                )
                scenarios.append(scenario)

        # Add edge cases (already budgeted)
        self._rng.shuffle(edge_scenarios)
        scenarios.extend(edge_scenarios[:n_edges])

        # Shuffle everything
        self._rng.shuffle(scenarios)

        return scenarios[:n]

    # ----------------------------------------------------------------
    # Internal methods
    # ----------------------------------------------------------------

    def _distribute_across_categories(
        self, n: int
    ) -> List[Tuple[ScenarioCategory, int]]:
        """Distribute n scenarios across categories proportionally."""
        categories = self.plan.categories
        if not categories:
            return []

        total_weight = sum(c.count for c in categories)
        if total_weight == 0:
            per_cat = n // len(categories)
            return [(c, per_cat) for c in categories]

        result = []
        allocated = 0
        for i, cat in enumerate(categories):
            if i == len(categories) - 1:
                count = n - allocated
            else:
                count = round(n * cat.count / total_weight)
            result.append((cat, count))
            allocated += count

        return result

    def _sample_dimensions(self, category: ScenarioCategory) -> Dict[str, Any]:
        """Sample variable values from the plan's dimensions."""
        variables = {}

        for dim in self.plan.dimensions:
            # Agent input dimensions (category="agent_input" or common input names)
            # should ALWAYS be sampled — they're the primary input to the agent
            is_agent_input = (
                dim.category == "agent_input"
                or dim.name in ("destination", "location", "city", "query", "name",
                                "order_id", "id", "product", "item", "account")
            )

            if is_agent_input:
                # Always sample agent inputs
                variables[dim.name] = self._sample_dimension(dim)
            elif (category.dimensions_varied
                    and dim.name not in category.dimensions_varied):
                # Test condition dimensions: use default when not varied
                variables[dim.name] = self._default_value(dim)
            else:
                variables[dim.name] = self._sample_dimension(dim)

        return variables

    def _sample_dimension(self, dim: WorldDimension) -> Any:
        """Sample a single dimension value."""
        spec = dim.value_spec

        if spec.type == "enum" and spec.values:
            return self._rng.choice(spec.values)
        elif spec.type == "numeric_range" and spec.range and len(spec.range) == 2:
            min_val, max_val = spec.range[0], spec.range[1]
            if isinstance(min_val, int) and isinstance(max_val, int):
                return self._rng.randint(int(min_val), int(max_val))
            return round(self._rng.uniform(float(min_val), float(max_val)), 2)
        elif spec.type in ("boolean", "bool"):
            return self._rng.choice([True, False])
        elif spec.values:
            return self._rng.choice(spec.values)

        # Fallback: try to find example values from category example_scenarios
        example_values = self._find_example_values(dim.name)
        if example_values:
            return self._rng.choice(example_values)

        return None

    def _find_example_values(self, dim_name: str) -> List[Any]:
        """Extract example values for a dimension from category example_scenarios."""
        values = set()
        for cat in self.plan.categories:
            if cat.example_scenario and isinstance(cat.example_scenario, dict):
                val = cat.example_scenario.get(dim_name)
                if val is not None:
                    values.add(val)
        return list(values) if values else []

    def _default_value(self, dim: WorldDimension) -> Any:
        """Get a default/middle value for a dimension."""
        spec = dim.value_spec
        if spec.type == "numeric_range" and spec.range and len(spec.range) == 2:
            return int((spec.range[0] + spec.range[1]) // 2)
        if spec.type == "enum" and spec.values:
            return spec.values[0]
        if spec.type in ("boolean", "bool"):
            return False

        # Fallback: use first example value from categories
        examples = self._find_example_values(dim.name)
        if examples:
            return examples[0]

        return None

    def _generate_message(self, category: ScenarioCategory, variables: Dict) -> str:
        """Generate a customer message for this scenario."""
        if self.message_generator:
            try:
                return self.message_generator(category, variables)
            except Exception:
                pass  # fall through to template

        # Template-based generation
        return self._template_message(category, variables)

    def _template_message(self, category: ScenarioCategory, variables: Dict) -> str:
        """Generate a message from templates based on category and variables."""
        # Priority 1: use the example_scenario from the category
        if category.example_scenario and isinstance(category.example_scenario, str):
            return self._vary_template(category.example_scenario, variables)
        elif category.example_scenario and isinstance(category.example_scenario, dict):
            # LLM returns dicts like {"destination": "Beijing", "tool_status": "failed"}
            # Use sampled variables first, fall back to example values
            ex = category.example_scenario
            subject = None
            for key in ("destination", "location", "city", "query", "name",
                         "order_id", "id", "product", "item", "input", "message", "text"):
                # Prefer the sampled variable over the example
                if key in variables and variables[key]:
                    subject = str(variables[key])
                    break
                elif key in ex and ex[key]:
                    subject = str(ex[key])
                    break
            if subject:
                return f"Help me with {subject}"
            return f"I need help with: {category.description}"

        # Priority 2: build a message from the most prominent variable
        subject = None
        for key in ("destination", "location", "city", "query", "name",
                     "order_id", "id", "product", "item", "account"):
            if key in variables and variables[key]:
                subject = str(variables[key])
                break

        if subject:
            cat_name = category.name.lower()
            if "happy" in cat_name:
                return self._rng.choice([
                    f"Help me with {subject}",
                    f"I need information about {subject}",
                    f"Can you look into {subject} for me?",
                ])
            elif "error" in cat_name or "failure" in cat_name:
                return self._rng.choice([
                    f"Help me with {subject}",
                    f"I need to know about {subject}",
                ])
            else:
                return f"I have a request about {subject}"

        # Priority 3: use the category description
        return f"I need help with: {category.description}"

    def _vary_template(self, template: str, variables: Dict) -> str:
        """Apply light variation to an example scenario template."""
        result = template
        for key, val in variables.items():
            result = result.replace(f"{{{key}}}", str(val))
        return result

    def _rules_for_category(self, category: ScenarioCategory) -> List[PolicyRule]:
        """Get the rules that apply to this category."""
        if category.rules_tested:
            return [r for r in self.plan.rules if r.name in category.rules_tested]
        return list(self.plan.rules)

    def _infer_expected_action(
        self,
        category: ScenarioCategory,
        variables: Dict,
        rules: List[PolicyRule],
    ) -> Optional[str]:
        """
        Infer what the agent should do based on category and variables.

        Only infers from universal patterns. Domain-specific logic should
        be configured via expected_action in the scenario or custom evaluator rules.
        """
        cat_name = category.name.lower()

        # Universal: happy-path categories should succeed
        if "happy" in cat_name:
            return "complete"

        # Universal: error/failure categories — don't prescribe, could go either way
        if "error" in cat_name or "failure" in cat_name:
            return None

        # Universal: edge cases — don't prescribe
        if "edge" in cat_name or "adversarial" in cat_name or "boundary" in cat_name:
            return None

        # If the category has a tool_status=failed dimension, expect error
        if variables.get("tool_status") in ("failed", "failure"):
            return "error"

        return None

    def _sample_difficulty(self, category: ScenarioCategory) -> str:
        """Sample a difficulty level based on the category's difficulty mix."""
        mix = category.difficulty_mix or {"easy": 0.2, "normal": 0.5, "hard": 0.25, "adversarial": 0.05}

        roll = self._rng.random()
        cumulative = 0.0
        for level, prob in mix.items():
            cumulative += prob
            if roll <= cumulative:
                return level
        return "normal"

    def _build_tool_mocks(self, variables: Dict) -> List[ToolMock]:
        """Build default (normal) tool mocks from the WorldSimulator's tool list."""
        if not self.world:
            return []

        mocks = []
        for tool_name, behavior in self.world.tools.items():
            response = self._build_tool_response(tool_name, behavior, variables)
            mock = ToolMock(
                tool_name=tool_name,
                behavior="normal",
                response=response,
            )
            mocks.append(mock)
        return mocks

    def _build_tool_response(
        self, tool_name: str, behavior: Any, variables: Dict,
    ) -> Dict[str, Any]:
        """
        Build a tool response for a scenario.

        Uses the tool's example_response from the config if available.
        Falls back to a generic response with scenario variables.
        """
        # Priority 1: developer-provided example_response from config
        if behavior and behavior.example_response:
            # Clone the example and inject variable values where field names match
            response = dict(behavior.example_response)
            for key in response:
                if key in variables:
                    response[key] = variables[key]
            return response

        # Priority 2: generic response that passes through scenario variables
        response = {"status": "ok", "tool": tool_name}
        for key, val in variables.items():
            response[key] = val
        return response

    def _apply_chaos(self, tool_mocks: List[ToolMock]) -> List[ToolMock]:
        """Overlay a WorldSimulator failure onto the tool mocks."""
        if not self.world:
            return tool_mocks

        world_config = self.world.generate()

        result = []
        for mock in tool_mocks:
            tool_state = world_config.tool_states.get(mock.tool_name)
            if tool_state and tool_state.behavior != "normal":
                # Apply the failure
                failed_mock = ToolMock(
                    tool_name=mock.tool_name,
                    behavior=tool_state.behavior,
                    response=self._apply_failure_to_response(
                        mock.response or {}, tool_state
                    ),
                    failure_details=tool_state.details if tool_state.details else {
                        "failure_type": tool_state.behavior,
                    },
                )
                result.append(failed_mock)
            else:
                result.append(mock)
        return result

    def _apply_failure_to_response(
        self, normal_response: Dict, tool_state: Any
    ) -> Optional[Dict]:
        """Apply a failure mode to transform the normal tool response."""
        if tool_state.failure_mode:
            try:
                return tool_state.failure_mode.apply(dict(normal_response), self._rng)
            except Exception:
                pass

        # Fallback based on behavior type
        behavior = tool_state.behavior
        if behavior == "timeout":
            return {"error": "timeout", "message": "Tool timed out"}
        elif behavior == "empty_response":
            return {}
        elif behavior == "rate_limited":
            return {"error": "rate_limited", "message": "Rate limit exceeded"}
        return normal_response

    def _apply_multi_chaos(self, tool_mocks: List[ToolMock]) -> List[ToolMock]:
        """Apply chaos to multiple tools for adversarial scenarios.

        Unlike _apply_chaos which lets the WorldSimulator decide (may fail 0-1 tools),
        this forces at least 2 tools to fail, or all tools if there are fewer than 2.
        """
        if not self.world or not tool_mocks:
            return tool_mocks

        result = []
        # Pick which tools to break — at least 2, or all if fewer
        n_to_break = max(2, len(tool_mocks) // 2 + 1)
        n_to_break = min(n_to_break, len(tool_mocks))
        indices_to_break = set(self._rng.sample(range(len(tool_mocks)), n_to_break))

        for idx, mock in enumerate(tool_mocks):
            if idx in indices_to_break:
                behavior = self.world.tools.get(mock.tool_name)
                if behavior and behavior.failure_modes:
                    fm = self._rng.choice(behavior.failure_modes)
                    from agentprobe.chaos import ToolState
                    tool_state = ToolState(
                        tool_name=mock.tool_name,
                        behavior=fm.failure_type(),
                        failure_mode=fm,
                    )
                    failed_mock = ToolMock(
                        tool_name=mock.tool_name,
                        behavior=tool_state.behavior,
                        response=self._apply_failure_to_response(
                            mock.response or {}, tool_state
                        ),
                        failure_details={"failure_type": tool_state.behavior},
                    )
                    result.append(failed_mock)
                else:
                    result.append(mock)
            else:
                result.append(mock)
        return result

    def _generate_edge_cases(self) -> List[Scenario]:
        """Generate scenarios from the plan's edge cases."""
        scenarios = []
        for ec in self.plan.edge_cases:
            scenario = Scenario(
                category=ec.category or "edge_case",
                description=f"Edge case: {ec.description}",
                customer_message=ec.inject_description if ec.inject_description else ec.description,
                variables={},
                difficulty="hard",
                metadata={"edge_case": True, "edge_case_name": ec.name},
            )
            scenarios.append(scenario)
        return scenarios

    def summary(self) -> Dict[str, Any]:
        """Summarize the engine configuration."""
        return {
            "plan": self.plan.name,
            "categories": len(self.plan.categories),
            "dimensions": len(self.plan.dimensions),
            "rules": len(self.plan.rules),
            "edge_cases": len(self.plan.edge_cases),
            "world_simulator": self.world is not None,
            "seed": self.seed,
        }
