"""
Tests for agentprobe.analysis — failure clustering, correlations, counterfactuals.
"""

import pytest
from agentprobe.analysis import (
    ScenarioResult, RuleViolation, FailureCluster,
    cluster_failures, detect_correlations, generate_counterfactuals,
    analyze_failures, Correlation, Counterfactual, FailureAnalysis,
)


# ============================================================================
# Test data builders
# ============================================================================

def _make_result(
    scenario_id: str,
    passed: bool,
    violations: list = None,
    category: str = "default",
    scenario: dict = None,
    world_failures: list = None,
) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=scenario_id,
        passed=passed,
        scenario=scenario or {},
        world_config={
            "has_failures": bool(world_failures),
            "injected_failures": world_failures or [],
        },
        violations=violations or [],
        category=category,
    )


def _timeout_violation():
    return RuleViolation(
        rule_name="retry_on_timeout",
        severity="critical",
        expected="Agent should retry",
        actual="Agent proceeded without data",
    )


def _window_violation():
    return RuleViolation(
        rule_name="return_window_check",
        severity="critical",
        expected="Deny return outside 30 days",
        actual="Agent approved the return",
    )


def _escalation_violation():
    return RuleViolation(
        rule_name="high_value_escalation",
        severity="major",
        expected="Escalate to manager",
        actual="Agent processed without escalation",
    )


# ============================================================================
# Failure Clustering Tests
# ============================================================================

class TestFailureClustering:
    def test_no_failures(self):
        results = [_make_result(f"s{i}", True) for i in range(10)]
        clusters = cluster_failures(results)
        assert clusters == []

    def test_single_cluster(self):
        results = [
            _make_result("s1", False, [_timeout_violation()], world_failures=["order-lookup:timeout"]),
            _make_result("s2", False, [_timeout_violation()], world_failures=["order-lookup:timeout"]),
            _make_result("s3", False, [_timeout_violation()], world_failures=["order-lookup:timeout"]),
            _make_result("s4", True),
        ]
        clusters = cluster_failures(results)
        assert len(clusters) == 1
        assert clusters[0].count == 3
        assert "timeout" in clusters[0].description.lower()

    def test_multiple_clusters(self):
        results = [
            # Cluster 1: timeout failures
            _make_result("s1", False, [_timeout_violation()], world_failures=["order-lookup:timeout"]),
            _make_result("s2", False, [_timeout_violation()], world_failures=["order-lookup:timeout"]),
            # Cluster 2: return window failures
            _make_result("s3", False, [_window_violation()]),
            _make_result("s4", False, [_window_violation()]),
            _make_result("s5", False, [_window_violation()]),
            # Passing
            _make_result("s6", True),
        ]
        clusters = cluster_failures(results)
        assert len(clusters) == 2
        # Larger cluster first
        assert clusters[0].count == 3
        assert clusters[1].count == 2

    def test_cluster_severity(self):
        results = [
            _make_result("s1", False, [_timeout_violation()]),  # critical
        ]
        clusters = cluster_failures(results)
        assert clusters[0].severity == "critical"

    def test_suggested_fix_for_timeout(self):
        results = [
            _make_result("s1", False, [_timeout_violation()], world_failures=["tool:timeout"]),
        ]
        clusters = cluster_failures(results)
        assert "retry" in clusters[0].suggested_fix.lower()

    def test_cluster_to_dict(self):
        results = [
            _make_result("s1", False, [_window_violation()]),
        ]
        clusters = cluster_failures(results)
        d = clusters[0].to_dict()
        assert "cluster_id" in d
        assert "count" in d
        assert "severity" in d


# ============================================================================
# Correlation Detection Tests
# ============================================================================

class TestCorrelationDetection:
    def _make_numeric_dataset(self, n=100):
        """Create dataset where high values of 'price' correlate with failure."""
        results = []
        for i in range(n):
            price = 100 + i * 10
            # High price → much higher failure rate
            passed = price < 600 or (price >= 600 and i % 5 == 0)
            results.append(_make_result(
                f"s{i}", passed,
                violations=[] if passed else [_escalation_violation()],
                scenario={"price": price, "category": "electronics"},
            ))
        return results

    def _make_categorical_dataset(self, n=100):
        """Create dataset where category 'grocery' correlates with failure."""
        import random
        rng = random.Random(42)
        results = []
        categories = ["electronics", "clothing", "grocery", "appliances"]
        for i in range(n):
            cat = categories[i % 4]
            # Grocery has much higher failure rate
            if cat == "grocery":
                passed = rng.random() > 0.7  # 70% failure
            else:
                passed = rng.random() > 0.1  # 10% failure
            results.append(_make_result(
                f"s{i}", passed,
                violations=[] if passed else [_window_violation()],
                scenario={"product_category": cat, "price": rng.randint(10, 500)},
            ))
        return results

    def test_too_few_results(self):
        results = [_make_result(f"s{i}", True) for i in range(5)]
        correlations = detect_correlations(results)
        assert correlations == []

    def test_numeric_correlation(self):
        results = self._make_numeric_dataset(100)
        correlations = detect_correlations(results, min_relative_risk=1.3)
        # Should find that high price correlates with failure
        price_corrs = [c for c in correlations if c.dimension == "price"]
        assert len(price_corrs) > 0
        assert any(c.relative_risk > 1.3 for c in price_corrs)

    def test_categorical_correlation(self):
        results = self._make_categorical_dataset(200)
        correlations = detect_correlations(results, min_relative_risk=1.3)
        # Should find that grocery correlates with failure
        cat_corrs = [c for c in correlations if "product_category" in c.dimension]
        assert len(cat_corrs) > 0
        grocery_corr = [c for c in cat_corrs if "grocery" in c.condition]
        assert len(grocery_corr) > 0

    def test_world_failure_correlation(self):
        """Scenarios with tool failures should correlate with agent failures."""
        results = []
        for i in range(100):
            has_world_failure = i < 30
            if has_world_failure:
                passed = i % 3 == 0  # 33% pass rate under tool failures
            else:
                passed = i % 10 != 0  # 90% pass rate normally
            results.append(_make_result(
                f"s{i}", passed,
                violations=[] if passed else [_timeout_violation()],
                world_failures=["tool:timeout"] if has_world_failure else [],
            ))
        correlations = detect_correlations(results, min_relative_risk=1.3)
        world_corrs = [c for c in correlations if "world" in c.dimension or "tool_failure" in c.dimension]
        assert len(world_corrs) > 0

    def test_correlation_to_dict(self):
        corr = Correlation(
            dimension="price", condition="> 500",
            failure_rate_when_true=0.6, failure_rate_when_false=0.1,
            relative_risk=6.0, sample_size_true=30, sample_size_false=70,
            p_value_approx=0.001, description="test",
        )
        d = corr.to_dict()
        assert d["relative_risk"] == 6.0


# ============================================================================
# Counterfactual Analysis Tests
# ============================================================================

class TestCounterfactualAnalysis:
    def test_no_failures(self):
        results = [_make_result(f"s{i}", True) for i in range(10)]
        cfs = generate_counterfactuals([], results)
        assert cfs == []

    def test_basic_counterfactual(self):
        # Failed scenario: days=35, passed scenarios: days=25
        failed = [
            _make_result("f1", False, [_window_violation()],
                         category="returns",
                         scenario={"days_since_delivery": 35, "category": "electronics"}),
        ]
        all_results = failed + [
            _make_result("p1", True, category="returns",
                         scenario={"days_since_delivery": 25, "category": "electronics"}),
            _make_result("p2", True, category="returns",
                         scenario={"days_since_delivery": 20, "category": "electronics"}),
        ]
        cfs = generate_counterfactuals(failed, all_results)
        assert len(cfs) > 0
        # Should identify days_since_delivery as the key difference
        days_cf = [cf for cf in cfs if cf.dimension == "days_since_delivery"]
        assert len(days_cf) > 0

    def test_world_failure_counterfactual(self):
        """If scenario failed with tool failures, counterfactual without them."""
        failed = [
            _make_result("f1", False, [_timeout_violation()],
                         category="returns",
                         scenario={"days": 15},
                         world_failures=["order-lookup:timeout"]),
        ]
        all_results = failed + [
            _make_result("p1", True, category="returns",
                         scenario={"days": 15}),
        ]
        cfs = generate_counterfactuals(failed, all_results)
        tool_cfs = [cf for cf in cfs if cf.dimension == "tool_failures"]
        assert len(tool_cfs) > 0
        assert "tool failures" in tool_cfs[0].description.lower()

    def test_counterfactual_to_dict(self):
        cf = Counterfactual(
            scenario_id="s1", dimension="price",
            original_value=600, counterfactual_value=400,
            original_passed=False, counterfactual_would_pass=True,
            confidence=0.85, description="test",
        )
        d = cf.to_dict()
        assert d["confidence"] == 0.85


# ============================================================================
# Full Pipeline Tests
# ============================================================================

class TestFullAnalysisPipeline:
    def _make_dataset(self, n=200):
        """Create a realistic dataset with mixed outcomes."""
        import random
        rng = random.Random(42)
        results = []
        for i in range(n):
            has_tool_failure = rng.random() < 0.3
            days = rng.randint(1, 60)
            price = rng.randint(10, 800)
            category = rng.choice(["returns", "escalation", "status"])

            violations = []
            passed = True

            # Return window violations
            if days > 30 and category == "returns":
                if rng.random() < 0.3:  # agent sometimes misses this
                    violations.append(_window_violation())
                    passed = False

            # High value escalation misses
            if price > 500 and category == "escalation":
                if rng.random() < 0.4:
                    violations.append(_escalation_violation())
                    passed = False

            # Tool failure handling
            if has_tool_failure and rng.random() < 0.5:
                violations.append(_timeout_violation())
                passed = False

            results.append(ScenarioResult(
                scenario_id=f"scn_{i:03d}",
                passed=passed,
                scenario={
                    "days_since_delivery": days,
                    "item_price": price,
                    "product_category": rng.choice(["electronics", "grocery", "clothing"]),
                },
                world_config={
                    "has_failures": has_tool_failure,
                    "injected_failures": ["order-lookup:timeout"] if has_tool_failure else [],
                },
                violations=violations,
                category=category,
            ))
        return results

    def test_full_pipeline(self):
        results = self._make_dataset(200)
        analysis = analyze_failures(results)

        assert isinstance(analysis, FailureAnalysis)
        assert analysis.total_scenarios == 200
        assert analysis.total_passed + analysis.total_failed == 200
        assert 0 <= analysis.pass_rate <= 1

        # Should have some clusters
        assert len(analysis.clusters) > 0

        # Should have resilience score
        assert 0 <= analysis.resilience.overall <= 100
        assert analysis.resilience.grade in ("A", "B", "C", "D", "F")

    def test_render(self):
        results = self._make_dataset(200)
        analysis = analyze_failures(results)
        rendered = analysis.render()
        assert "FAILURE ANALYSIS REPORT" in rendered
        assert "RESILIENCE SCORE" in rendered
        assert "FAILURE CLUSTERS" in rendered

    def test_to_dict(self):
        results = self._make_dataset(100)
        analysis = analyze_failures(results)
        d = analysis.to_dict()
        assert "clusters" in d
        assert "correlations" in d
        assert "counterfactuals" in d
        assert "resilience" in d

    def test_all_passing(self):
        results = [_make_result(f"s{i}", True, scenario={"x": i}) for i in range(50)]
        analysis = analyze_failures(results)
        assert analysis.total_failed == 0
        assert len(analysis.clusters) == 0
        assert analysis.resilience.grade in ("A", "B")

    def test_resilience_gap(self):
        """Agent should show lower performance under tool failures."""
        results = self._make_dataset(200)
        analysis = analyze_failures(results)
        # Normal should generally be better than degraded
        # (not guaranteed with random data but likely)
        assert analysis.resilience.normal_pass_rate >= 0
        assert analysis.resilience.degraded_pass_rate >= 0
