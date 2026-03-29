"""Tests for Week 7 — content checks, difficulty scoring, report, HTML export."""

import os
import tempfile
import pytest
from agentprobe.engine import (
    Scenario, ToolMock, ContentEvaluator, ContentCheckConfig,
    DifficultyScorer, DifficultyScore, EvaluationReport, export_html,
)
from agentprobe.analysis import ScenarioResult, RuleViolation, analyze_failures


class TestContentEvaluator:
    def test_passes_complete_response(self):
        evaluator = ContentEvaluator(required_fields=["brief", "temperature_c"])
        scenario = Scenario(category="test", customer_message="test")
        output = {"action": "complete", "brief": "Nice weather in Paris", "temperature_c": 22.5}
        checks = evaluator.evaluate(scenario, output)
        assert all(c.passed for c in checks)

    def test_fails_missing_required_field(self):
        evaluator = ContentEvaluator(required_fields=["brief", "currency"])
        scenario = Scenario(category="test", customer_message="test")
        output = {"action": "complete", "brief": "Nice weather"}
        checks = evaluator.evaluate(scenario, output)
        failed = [c for c in checks if not c.passed]
        assert any(c.rule_name == "required_field_missing" for c in failed)

    def test_skips_non_complete_action(self):
        evaluator = ContentEvaluator(required_fields=["brief"])
        scenario = Scenario(category="test", customer_message="test")
        output = {"action": "error", "reason": "tool failed"}
        checks = evaluator.evaluate(scenario, output)
        # Should not fail — content checks only apply to action="complete"
        assert not any(c.rule_name == "required_field_missing" for c in checks)

    def test_detects_placeholder_text(self):
        evaluator = ContentEvaluator()
        scenario = Scenario(category="test", customer_message="test")
        output = {"action": "complete", "brief": "[Brief unavailable: RateLimitError]"}
        checks = evaluator.evaluate(scenario, output)
        failed = [c for c in checks if not c.passed]
        assert any(c.rule_name == "placeholder_response" for c in failed)

    def test_checks_text_length(self):
        evaluator = ContentEvaluator(min_text_length={"brief": 50})
        scenario = Scenario(category="test", customer_message="test")
        output = {"action": "complete", "brief": "Short."}
        checks = evaluator.evaluate(scenario, output)
        assert any(c.rule_name == "response_too_short" for c in checks)

    def test_tool_data_check(self):
        evaluator = ContentEvaluator(
            tool_output_fields={"get-weather": ["temperature_c"]}
        )
        scenario = Scenario(
            category="test", customer_message="test",
            tool_mocks=[ToolMock(tool_name="get-weather", behavior="normal")],
        )
        output = {"action": "complete", "brief": "Nice weather"}  # missing temperature_c
        checks = evaluator.evaluate(scenario, output)
        assert any(c.rule_name == "tool_data_missing" for c in checks)

    def test_tool_data_ok_when_tool_failed(self):
        evaluator = ContentEvaluator(
            tool_output_fields={"get-weather": ["temperature_c"]}
        )
        scenario = Scenario(
            category="test", customer_message="test",
            tool_mocks=[ToolMock(tool_name="get-weather", behavior="timeout")],
        )
        output = {"action": "complete", "brief": "Weather unavailable"}
        checks = evaluator.evaluate(scenario, output)
        # Should NOT flag missing temperature_c since tool failed
        assert not any(c.rule_name == "tool_data_missing" for c in checks)


class TestDifficultyScorer:
    def test_easy_scenario(self):
        s = Scenario(category="happy-path", customer_message="Research Paris")
        scorer = DifficultyScorer()
        score = scorer.score(s)
        assert score.level == "easy"
        assert score.score < 25

    def test_hard_scenario_with_chaos(self):
        s = Scenario(
            category="test", customer_message="test",
            tool_mocks=[
                ToolMock(tool_name="t1", behavior="timeout"),
                ToolMock(tool_name="t2", behavior="empty_response"),
            ],
        )
        scorer = DifficultyScorer()
        score = scorer.score(s)
        assert score.score >= 40

    def test_edge_case_adds_difficulty(self):
        s = Scenario(category="test", customer_message="test",
                     metadata={"edge_case": True})
        scorer = DifficultyScorer()
        score = scorer.score(s)
        assert score.factors.get("edge_case", 0) == 25

    def test_empty_input(self):
        s = Scenario(category="test", customer_message="")
        scorer = DifficultyScorer()
        score = scorer.score(s)
        assert score.factors.get("empty_input", 0) == 15

    def test_summary(self):
        scenarios = [
            Scenario(category="happy", customer_message="easy"),
            Scenario(category="hard", customer_message="hard",
                     tool_mocks=[ToolMock(tool_name="t", behavior="timeout")],
                     metadata={"edge_case": True}),
        ]
        scorer = DifficultyScorer()
        summary = scorer.summary(scenarios)
        assert summary["total"] == 2
        assert summary["average_difficulty"] > 0


class TestEvaluationReport:
    def _make_results(self):
        results = [
            ScenarioResult(scenario_id="s1", passed=True, scenario={"destination": "Paris"},
                world_config={"has_failures": False, "injected_failures": []},
                violations=[], category="happy-path"),
            ScenarioResult(scenario_id="s2", passed=False, scenario={"destination": "Tokyo"},
                world_config={"has_failures": True, "injected_failures": ["get-weather:timeout"]},
                violations=[RuleViolation(rule_name="expected_action", severity="major",
                    expected="complete", actual="error")],
                category="error-handling"),
        ]
        scenarios = [
            Scenario(category="happy-path", customer_message="Research Paris",
                     variables={"destination": "Paris"}, expected_action="complete"),
            Scenario(category="error-handling", customer_message="Research Tokyo",
                     variables={"destination": "Tokyo"}, expected_action="complete",
                     tool_mocks=[ToolMock(tool_name="get-weather", behavior="timeout")]),
        ]
        return results, scenarios

    def test_build(self):
        results, scenarios = self._make_results()
        analysis = analyze_failures(results)
        report = EvaluationReport.build(results, scenarios, analysis, agent_name="test-agent")
        assert report.total_scenarios == 2
        assert report.total_passed == 1
        assert report.total_failed == 1
        assert report.pass_rate == 50.0
        assert len(report.verdicts) == 2

    def test_render(self):
        results, scenarios = self._make_results()
        analysis = analyze_failures(results)
        report = EvaluationReport.build(results, scenarios, analysis)
        text = report.render()
        assert "EVALUATION REPORT" in text
        assert "2" in text  # total scenarios

    def test_to_json(self):
        results, scenarios = self._make_results()
        analysis = analyze_failures(results)
        report = EvaluationReport.build(results, scenarios, analysis)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            report.to_json(path)
            assert os.path.getsize(path) > 100
        finally:
            os.unlink(path)

    def test_difficulty_breakdown(self):
        results, scenarios = self._make_results()
        analysis = analyze_failures(results)
        report = EvaluationReport.build(results, scenarios, analysis)
        assert report.difficulty_summary.get("total") == 2
        assert len(report.pass_rate_by_difficulty) > 0

    def test_failure_tags(self):
        results, scenarios = self._make_results()
        analysis = analyze_failures(results)
        report = EvaluationReport.build(results, scenarios, analysis)
        assert "policy_violation" in report.failures_by_tag


class TestHTMLExport:
    def test_export(self):
        results = [
            ScenarioResult(scenario_id="s1", passed=True, scenario={"destination": "Paris"},
                world_config={}, violations=[], category="happy"),
        ]
        scenarios = [
            Scenario(category="happy", customer_message="Research Paris",
                     variables={"destination": "Paris"}),
        ]
        analysis = analyze_failures(results)
        report = EvaluationReport.build(results, scenarios, analysis, agent_name="test")

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            export_html(report, path)
            content = open(path).read()
            assert "<!DOCTYPE html>" in content
            assert "probe-ai" in content
            assert "Paris" in content
            assert os.path.getsize(path) > 1000
        finally:
            os.unlink(path)
