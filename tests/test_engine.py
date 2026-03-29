"""Tests for agentprobe.engine — VariationEngine, Runner, ToolCallValidator, full pipeline."""

import pytest
from agentprobe.engine import (
    Scenario, ToolMock, VariationEngine, Runner, Evaluator,
    RuleBasedEvaluator, RuleCheck, ToolCallValidator, ToolCallViolation,
    ToolSchema, mock_tool_response,
)
from agentprobe.engine.runner import _current_mock_registry, ToolMockRegistry
from agentprobe.scenarios.plan import (
    TestPlan, ScenarioCategory, WorldDimension, DimensionValue,
    PolicyRule, EdgeCase, RubricDimension,
)
from agentprobe.chaos import (
    WorldSimulator, ToolBehavior, ChaosProfile, Timeout, PartialData, SchemaDrift,
)
from agentprobe.analysis import ScenarioResult, analyze_failures
from agentprobe.models import SpanRecord, TraceRecord
from agentprobe.core import AgentProbe, trace_agent, trace_tool


# ============================================================================
# Fixtures
# ============================================================================

def _make_plan():
    """Create a sample TestPlan for testing."""
    return TestPlan(
        name="novamart-test",
        agent_description="Customer support agent for NovaMart",
        domain="customer_support",
        categories=[
            ScenarioCategory(
                name="returns",
                description="Return request scenarios",
                count=60,
                rules_tested=["30_day_window", "high_value_escalation"],
                dimensions_varied=["days_since_delivery", "item_price"],
                example_scenario="I want to return my order",
            ),
            ScenarioCategory(
                name="status-inquiry",
                description="Order status check",
                count=30,
                rules_tested=[],
                dimensions_varied=["order_status"],
                example_scenario="What's the status of my order?",
            ),
            ScenarioCategory(
                name="escalation",
                description="Escalation scenarios",
                count=10,
                rules_tested=["high_value_escalation"],
                dimensions_varied=["item_price"],
            ),
        ],
        rules=[
            PolicyRule(name="30_day_window", description="Returns within 30 days only",
                       condition="days_since_delivery <= 30", expected_outcome="approve return",
                       severity="critical", source="return_policy"),
            PolicyRule(name="high_value_escalation", description="Orders over $500 escalate",
                       condition="item_price > 500", expected_outcome="escalate to manager",
                       severity="critical", source="escalation_policy"),
        ],
        dimensions=[
            WorldDimension(name="days_since_delivery", category="order",
                           description="Days since the order was delivered",
                           value_spec=DimensionValue(type="numeric_range", range=[1, 60])),
            WorldDimension(name="item_price", category="order",
                           description="Price of the item",
                           value_spec=DimensionValue(type="numeric_range", range=[10, 800])),
            WorldDimension(name="product_category", category="product",
                           description="Product type",
                           value_spec=DimensionValue(type="enum", values=["electronics", "clothing", "grocery"])),
            WorldDimension(name="order_status", category="order",
                           description="Current order status",
                           value_spec=DimensionValue(type="enum", values=["delivered", "shipped", "processing"])),
        ],
        edge_cases=[
            EdgeCase(name="lawyer_threat", description="Customer mentions a lawyer",
                     category="escalation",
                     inject_description="I want to return this or I'll contact my lawyer"),
            EdgeCase(name="empty_message", description="Empty message",
                     category="returns", inject_description=""),
        ],
        rubric=[
            RubricDimension(dimension="accuracy", description="Did the agent follow policy?",
                            weight=0.5),
        ],
    )


def _make_world():
    """Create a WorldSimulator for testing."""
    return WorldSimulator(
        tools={
            "order-lookup": ToolBehavior(
                description="Look up order",
                failure_modes=[Timeout(probability=0.3), PartialData(probability=0.2)],
            ),
            "process-return": ToolBehavior(
                description="Process a return",
                failure_modes=[Timeout(probability=0.2)],
            ),
        },
        seed=42,
        normal_ratio=0.5,
    )


@pytest.fixture(autouse=True)
def reset_probe():
    AgentProbe._instance = None
    yield
    AgentProbe._instance = None


# ============================================================================
# Scenario Tests
# ============================================================================

class TestScenario:
    def test_basic(self):
        s = Scenario(category="returns", customer_message="I want to return this")
        assert s.category == "returns"
        assert not s.has_tool_failures
        assert s.injected_failures == []

    def test_with_tool_mocks(self):
        s = Scenario(
            tool_mocks=[
                ToolMock(tool_name="order-lookup", behavior="normal"),
                ToolMock(tool_name="process-return", behavior="timeout"),
            ]
        )
        assert s.has_tool_failures
        assert s.injected_failures == ["process-return:timeout"]

    def test_get_tool_mock(self):
        s = Scenario(
            tool_mocks=[ToolMock(tool_name="order-lookup", behavior="normal",
                                 response={"order_id": "123"})]
        )
        mock = s.get_tool_mock("order-lookup")
        assert mock is not None
        assert mock.response == {"order_id": "123"}
        assert s.get_tool_mock("nonexistent") is None

    def test_to_dict(self):
        s = Scenario(category="returns", customer_message="test",
                     variables={"days": 15}, difficulty="hard")
        d = s.to_dict()
        assert d["category"] == "returns"
        assert d["difficulty"] == "hard"
        assert d["variables"] == {"days": 15}


# ============================================================================
# VariationEngine Tests
# ============================================================================

class TestVariationEngine:
    def test_generate_basic(self):
        plan = _make_plan()
        engine = VariationEngine(plan=plan, seed=42)
        scenarios = engine.generate(n=50)
        assert len(scenarios) == 50
        assert all(isinstance(s, Scenario) for s in scenarios)

    def test_categories_distributed(self):
        plan = _make_plan()
        engine = VariationEngine(plan=plan, seed=42)
        scenarios = engine.generate(n=100)

        cats = {}
        for s in scenarios:
            cats[s.category] = cats.get(s.category, 0) + 1

        # Should have all categories represented
        assert "returns" in cats
        assert "status-inquiry" in cats

    def test_dimensions_sampled(self):
        plan = _make_plan()
        engine = VariationEngine(plan=plan, seed=42)
        scenarios = engine.generate(n=50)

        # Check that variables are populated (exclude edge cases which have empty vars)
        for s in scenarios:
            if not s.metadata.get("edge_case"):
                assert "days_since_delivery" in s.variables or "order_status" in s.variables

    def test_with_world_simulator(self):
        plan = _make_plan()
        world = _make_world()
        engine = VariationEngine(plan=plan, world=world, seed=42)
        scenarios = engine.generate(n=100, chaos_ratio=0.5)

        with_failures = sum(1 for s in scenarios if s.has_tool_failures)
        without_failures = sum(1 for s in scenarios if not s.has_tool_failures)
        assert with_failures > 0
        assert without_failures > 0

    def test_edge_cases_included(self):
        plan = _make_plan()
        engine = VariationEngine(plan=plan, seed=42)
        scenarios = engine.generate(n=100)

        edge = [s for s in scenarios if s.metadata.get("edge_case")]
        assert len(edge) > 0

    def test_messages_generated(self):
        plan = _make_plan()
        engine = VariationEngine(plan=plan, seed=42)
        scenarios = engine.generate(n=20)

        for s in scenarios:
            assert isinstance(s.customer_message, str)

    def test_expected_action_inferred(self):
        """Verify _infer_expected_action sets values for happy-path categories."""
        from agentprobe.scenarios.plan import ScenarioCategory
        # Create a plan with a happy-path category
        plan = _make_plan()
        plan.categories.append(ScenarioCategory(
            name="happy-path", description="Standard requests",
            count=20, rules_tested=[], dimensions_varied=[],
        ))
        engine = VariationEngine(plan=plan, seed=42)
        scenarios = engine.generate(n=100)

        with_expected = [s for s in scenarios if s.expected_action is not None]
        assert len(with_expected) > 0
        # Happy-path scenarios get "complete"
        happy = [s for s in scenarios if s.category == "happy-path"]
        assert all(s.expected_action == "complete" for s in happy)

    def test_summary(self):
        plan = _make_plan()
        engine = VariationEngine(plan=plan, seed=42)
        s = engine.summary()
        assert s["plan"] == "novamart-test"
        assert s["categories"] == 3
        assert s["dimensions"] == 4

    def test_reproducible(self):
        plan = _make_plan()
        e1 = VariationEngine(plan=plan, seed=99)
        e2 = VariationEngine(plan=plan, seed=99)
        s1 = e1.generate(n=20)
        s2 = e2.generate(n=20)
        for a, b in zip(s1, s2):
            assert a.variables == b.variables
            assert a.category == b.category


# ============================================================================
# Runner Tests
# ============================================================================

class TestRunner:
    def test_run_basic(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test-agent")
        def agent(message, order_id):
            return {"action": "respond", "message": f"Got {message}"}

        scenarios = [
            Scenario(category="test", customer_message="hello",
                     variables={"order_id": "ORD-1"}, expected_action="respond"),
        ]
        runner = Runner(agent_fn=agent, probe=probe)
        results = runner.run(scenarios)

        assert len(results) == 1
        assert results[0].passed
        assert results[0].scenario_id == scenarios[0].scenario_id

    def test_run_failure_detected(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test-agent")
        def agent(message, order_id):
            return {"action": "approve"}

        scenarios = [
            Scenario(category="test", customer_message="test",
                     variables={"order_id": "ORD-1"}, expected_action="deny"),
        ]
        runner = Runner(agent_fn=agent, probe=probe)
        results = runner.run(scenarios)

        assert len(results) == 1
        assert not results[0].passed
        assert any(v.rule_name == "expected_action" for v in results[0].violations)

    def test_run_agent_error(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("bad-agent")
        def agent(message, order_id):
            raise ValueError("boom")

        scenarios = [
            Scenario(category="test", customer_message="test",
                     variables={"order_id": "ORD-1"}),
        ]
        runner = Runner(agent_fn=agent, probe=probe)
        results = runner.run(scenarios)

        assert len(results) == 1
        assert not results[0].passed
        assert any(v.rule_name == "agent_no_crash" for v in results[0].violations)

    def test_run_with_mock_tools(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("mock-agent")
        def agent(message, order_id):
            mock = mock_tool_response("order-lookup")
            if mock:
                return {"action": "respond", "data": mock}
            return {"action": "respond", "data": "no mock"}

        scenarios = [
            Scenario(
                category="test", customer_message="test",
                variables={"order_id": "ORD-1"},
                tool_mocks=[
                    ToolMock(tool_name="order-lookup", behavior="normal",
                             response={"order_id": "ORD-1", "status": "delivered"}),
                ],
            ),
        ]
        runner = Runner(agent_fn=agent, probe=probe)
        results = runner.run(scenarios)

        assert results[0].passed

    def test_run_batch(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("batch-agent")
        def agent(message, order_id):
            return {"action": "respond"}

        scenarios = [
            Scenario(category="test", customer_message=f"msg {i}",
                     variables={"order_id": f"ORD-{i}"}, expected_action="respond")
            for i in range(10)
        ]
        runner = Runner(agent_fn=agent, probe=probe)
        results = runner.run(scenarios)

        assert len(results) == 10
        assert all(r.passed for r in results)

    def test_run_progress_callback(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test-agent")
        def agent(message, order_id):
            return {"action": "respond"}

        progress = []
        runner = Runner(agent_fn=agent, probe=probe)
        runner.run(
            [Scenario(category="t", customer_message="m",
                      variables={"order_id": "O"}) for _ in range(5)],
            on_progress=lambda done, total: progress.append((done, total)),
        )
        assert progress == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]

    def test_run_one(self):
        probe = AgentProbe()
        probe.init()

        @trace_agent("test-agent")
        def agent(message, order_id):
            return {"action": "respond"}

        runner = Runner(agent_fn=agent, probe=probe)
        result = runner.run_one(
            Scenario(category="t", customer_message="hi",
                     variables={"order_id": "O"}, expected_action="respond")
        )
        assert result.passed


# ============================================================================
# RuleBasedEvaluator Tests
# ============================================================================

class TestRuleBasedEvaluator:
    def test_custom_rule(self):
        evaluator = RuleBasedEvaluator()
        evaluator.add_rule("check_greeting", lambda s, out, t: RuleCheck(
            rule_name="check_greeting",
            passed="hello" in str(out).lower() if out else False,
            expected="Response should contain greeting",
            actual=str(out),
        ))

        scenario = Scenario(category="test", customer_message="hi",
                            rules_to_check=["check_greeting"])

        # Passing case
        checks = evaluator.evaluate(scenario, {"message": "Hello! How can I help?"})
        greeting_check = next(c for c in checks if c.rule_name == "check_greeting")
        assert greeting_check.passed

        # Failing case
        checks = evaluator.evaluate(scenario, {"message": "Your order is shipped."})
        greeting_check = next(c for c in checks if c.rule_name == "check_greeting")
        assert not greeting_check.passed


# ============================================================================
# ToolCallValidator Tests
# ============================================================================

class TestToolCallValidator:
    def _make_span(self, tool_name, args):
        span = SpanRecord(name=tool_name, span_type="tool_call")
        span.tool_name = tool_name
        span.tool_args = args
        return span

    def test_valid_call(self):
        validator = ToolCallValidator(schemas={
            "order-lookup": ToolSchema(
                required_params={"order_id": "str"},
                optional_params={"include_history": "bool"},
            ),
        })
        span = self._make_span("order-lookup", {"order_id": "ORD-123"})
        violations = validator.validate_span(span)
        assert violations == []

    def test_missing_required_param(self):
        validator = ToolCallValidator(schemas={
            "order-lookup": ToolSchema(required_params={"order_id": "str"}),
        })
        span = self._make_span("order-lookup", {})
        violations = validator.validate_span(span)
        assert len(violations) == 1
        assert violations[0].violation_type == "missing_param"
        assert violations[0].severity == "critical"

    def test_hallucinated_param(self):
        validator = ToolCallValidator(schemas={
            "order-lookup": ToolSchema(required_params={"order_id": "str"}),
        })
        span = self._make_span("order-lookup", {"order_id": "123", "fake_param": "xyz"})
        violations = validator.validate_span(span)
        assert len(violations) == 1
        assert violations[0].violation_type == "hallucinated_param"

    def test_wrong_type(self):
        validator = ToolCallValidator(schemas={
            "order-lookup": ToolSchema(required_params={"order_id": "str", "count": "int"}),
        })
        span = self._make_span("order-lookup", {"order_id": "123", "count": "five"})
        violations = validator.validate_span(span)
        assert any(v.violation_type == "wrong_type" and v.param_name == "count" for v in violations)

    def test_invalid_enum(self):
        validator = ToolCallValidator(schemas={
            "search": ToolSchema(
                required_params={"query": "str", "sort": "str"},
                enum_values={"sort": ["price_asc", "price_desc", "rating"]},
            ),
        })
        span = self._make_span("search", {"query": "shoes", "sort": "random_sort"})
        violations = validator.validate_span(span)
        assert any(v.violation_type == "invalid_enum" for v in violations)

    def test_out_of_range(self):
        validator = ToolCallValidator(schemas={
            "search": ToolSchema(
                required_params={"limit": "int"},
                numeric_ranges={"limit": (1, 100)},
            ),
        })
        span = self._make_span("search", {"limit": 500})
        violations = validator.validate_span(span)
        assert any(v.violation_type == "out_of_range" for v in violations)

    def test_validate_trace(self):
        validator = ToolCallValidator(schemas={
            "order-lookup": ToolSchema(required_params={"order_id": "str"}),
        })
        trace = TraceRecord(agent_name="test")
        span = self._make_span("order-lookup", {"wrong_param": "123"})
        trace.add_span(span)
        violations = validator.validate_trace(trace)
        assert len(violations) == 2  # missing + hallucinated

    def test_unknown_tool_skipped(self):
        validator = ToolCallValidator(schemas={
            "order-lookup": ToolSchema(required_params={"order_id": "str"}),
        })
        span = self._make_span("unknown-tool", {"anything": "goes"})
        violations = validator.validate_span(span)
        assert violations == []


# ============================================================================
# SchemaDrift Tests
# ============================================================================

class TestSchemaDrift:
    def test_field_renamed(self):
        import random
        sd = SchemaDrift(drift_type="field_renamed", target_field="delivery_date",
                         new_field_name="delivered_at")
        result = sd.apply({"delivery_date": "2025-01-15", "status": "ok"}, random.Random(42))
        assert "delivered_at" in result
        assert "delivery_date" not in result

    def test_field_removed(self):
        import random
        sd = SchemaDrift(drift_type="field_removed", target_field="tracking")
        result = sd.apply({"order_id": "123", "tracking": "TRK-456"}, random.Random(42))
        assert "tracking" not in result
        assert "order_id" in result

    def test_type_changed(self):
        import random
        sd = SchemaDrift(drift_type="type_changed", target_field="price")
        result = sd.apply({"price": 49.99, "name": "Widget"}, random.Random(42))
        assert isinstance(result["price"], str)  # was float, now string

    def test_nested_restructured(self):
        import random
        sd = SchemaDrift(drift_type="nested_restructured", target_field="address")
        result = sd.apply({"address": "123 Main St", "city": "NYC"}, random.Random(42))
        assert isinstance(result["address"], dict)
        assert result["address"]["value"] == "123 Main St"

    def test_failure_type(self):
        sd = SchemaDrift()
        assert sd.failure_type() == "schema_drift"


# ============================================================================
# Full Pipeline Test
# ============================================================================

class TestFullPipeline:
    def test_end_to_end(self):
        """VariationEngine → Runner → analyze_failures — the complete loop."""
        probe = AgentProbe()
        probe.init()

        # A simple agent
        @trace_agent("pipeline-agent")
        def agent(message, order_id):
            mock = mock_tool_response("order-lookup")
            if mock and mock.get("error"):
                return {"action": "error", "reason": mock["error"]}
            days = mock.get("days_since_delivery", 15) if mock else 15
            price = mock.get("price", 50) if mock else 50
            if days > 30:
                return {"action": "deny"}
            if price > 500:
                return {"action": "escalate"}
            return {"action": "approve"}

        # Generate scenarios
        plan = _make_plan()
        world = _make_world()
        engine = VariationEngine(plan=plan, world=world, seed=42)
        scenarios = engine.generate(n=50, chaos_ratio=0.3)

        assert len(scenarios) == 50

        # Run them
        runner = Runner(agent_fn=agent, probe=probe)
        results = runner.run(scenarios)

        assert len(results) == 50
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        assert passed > 0

        # Analyze
        analysis = analyze_failures(results)
        assert analysis.total_scenarios == 50
        assert analysis.resilience is not None
        assert 0 <= analysis.resilience.overall <= 100

    def test_results_feed_analysis(self):
        """Verify ScenarioResults from Runner have the right shape for analysis."""
        probe = AgentProbe()
        probe.init()

        @trace_agent("test")
        def agent(message, order_id):
            return {"action": "approve"}

        scenarios = [
            Scenario(category="returns", customer_message="return this",
                     variables={"days_since_delivery": 15, "item_price": 50,
                                "order_id": "ORD-1"},
                     expected_action="approve"),
            Scenario(category="returns", customer_message="return this",
                     variables={"days_since_delivery": 45, "item_price": 50,
                                "order_id": "ORD-2"},
                     expected_action="deny"),
        ]

        runner = Runner(agent_fn=agent, probe=probe)
        results = runner.run(scenarios)

        # First should pass, second should fail
        assert results[0].passed
        assert not results[1].passed

        # Feed into analysis
        analysis = analyze_failures(results)
        assert analysis.total_failed == 1
        assert len(analysis.clusters) >= 1
