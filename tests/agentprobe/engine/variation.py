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

        # Distribute scenarios across categories proportionally
        category_counts = self._distribute_across_categories(n)

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

                # Optionally overlay chaos
                if self.world and self._rng.random() < chaos_ratio:
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

        # Add edge cases
        edge_scenarios = self._generate_edge_cases()
        scenarios.extend(edge_scenarios)

        # Shuffle for randomness
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
            # If category specifies which dimensions to vary, respect that
            if (category.dimensions_varied
                    and dim.name not in category.dimensions_varied):
                # Use a default/middle value
                variables[dim.name] = self._default_value(dim)
                continue

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
        elif spec.type == "boolean":
            return self._rng.choice([True, False])
        elif spec.values:
            return self._rng.choice(spec.values)

        return None

    def _default_value(self, dim: WorldDimension) -> Any:
        """Get a default/middle value for a dimension."""
        spec = dim.value_spec
        if spec.type == "numeric_range" and spec.range and len(spec.range) == 2:
            return int((spec.range[0] + spec.range[1]) // 2)
        if spec.type == "enum" and spec.values:
            return spec.values[0]
        if spec.type == "boolean":
            return False
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
        # Use the example_scenario if available
        if category.example_scenario:
            return self._vary_template(category.example_scenario, variables)

        # Generic templates based on category name
        name = category.name.lower()
        item = variables.get("product_category", variables.get("item", "my order"))
        order_id = variables.get("order_id", f"ORD-{self._rng.randint(1000,9999)}")

        templates = {
            "return": [
                f"I'd like to return {item}, order {order_id}",
                f"I want a refund for {item}",
                f"Can I send back {item}? Order number {order_id}",
                f"I need to return {item}, it's not what I expected",
            ],
            "escalat": [
                f"I want to speak to a manager about order {order_id}",
                f"This is unacceptable, I need to escalate my complaint about {item}",
                f"I've been waiting too long, I want a supervisor",
            ],
            "status": [
                f"What's the status of order {order_id}?",
                f"Where is my {item}?",
                f"Can you check on order {order_id} for me?",
            ],
            "complaint": [
                f"I'm very unhappy with {item} from order {order_id}",
                f"This product is defective, order {order_id}",
                f"I'm filing a complaint about my experience with {item}",
            ],
        }

        # Match category name to template group
        for key, msgs in templates.items():
            if key in name:
                return self._rng.choice(msgs)

        # Default
        return f"I have a question about order {order_id} for {item}"

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
        Infer what the agent should do based on variables and rules.

        This is heuristic — it looks for common patterns like return windows,
        value thresholds, etc. For complex logic, the user should set
        expected_action explicitly or use an LLM evaluator.
        """
        days = variables.get("days_since_delivery")
        price = variables.get("item_price", variables.get("price", variables.get("order_total")))

        cat_name = category.name.lower()

        if "return" in cat_name:
            if days is not None and days > 30:
                return "deny"
            if price is not None and price > 500:
                return "escalate"
            if days is not None and days <= 30:
                return "approve"

        if "escalat" in cat_name:
            return "escalate"

        if "status" in cat_name or "inquiry" in cat_name:
            return "respond"

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
        for tool_name in self.world.tools:
            mock = ToolMock(
                tool_name=tool_name,
                behavior="normal",
                response=self._build_tool_response(tool_name, variables),
            )
            mocks.append(mock)
        return mocks

    def _build_tool_response(self, tool_name: str, variables: Dict) -> Dict[str, Any]:
        """Build a plausible tool response from scenario variables."""
        name_lower = tool_name.lower().replace("-", "_").replace(" ", "_")

        if "order" in name_lower or "lookup" in name_lower:
            return {
                "order_id": variables.get("order_id", f"ORD-{self._rng.randint(1000,9999)}"),
                "status": variables.get("order_status", "delivered"),
                "customer": variables.get("customer_name", "Test Customer"),
                "item": variables.get("product_category", "Widget"),
                "price": variables.get("item_price", variables.get("price", 49.99)),
                "days_since_delivery": variables.get("days_since_delivery", 15),
                "payment": variables.get("payment_method", "credit_card"),
            }
        elif "return" in name_lower or "refund" in name_lower:
            return {
                "return_id": f"RET-{self._rng.randint(1000,9999)}",
                "status": "initiated",
                "refund_amount": variables.get("item_price", variables.get("price", 49.99)),
            }
        elif "escalat" in name_lower:
            return {
                "ticket_id": f"TKT-{self._rng.randint(1000,9999)}",
                "queue": "tier2-support",
            }

        # Generic response
        return {"status": "ok", "data": variables}

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
