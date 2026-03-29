"""
Structured Evaluation Report.

Takes ScenarioResults + FailureAnalysis + DifficultyScores and produces
a structured report object with:
- Summary stats
- Per-scenario verdicts with category tags (tool_failure, policy_violation, etc.)
- Root cause identification (which step went wrong first)
- Difficulty-contextualized pass rates
- Ranked failure list

This is the object that feeds both the HTML export and the console output.

Usage:
    report = EvaluationReport.build(
        results=results,
        scenarios=scenarios,
        analysis=analysis,
    )
    print(report.render())
    report.to_json("report.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from agentprobe.analysis import FailureAnalysis, ScenarioResult
from agentprobe.engine.difficulty import DifficultyScorer, DifficultyScore
from agentprobe.engine.scenario import Scenario


@dataclass
class ScenarioVerdict:
    """Detailed verdict for a single scenario."""
    scenario_id: str
    category: str
    passed: bool
    verdict: str = ""                   # "passed", "degraded", "failed"
    action: str = ""                    # what the agent did
    expected_action: Optional[str] = None
    difficulty: DifficultyScore = None

    # Category tag for the failure (if failed)
    failure_tag: str = ""               # tool_failure, policy_violation, crash, incomplete_response, quality_issue
    failure_reason: str = ""
    violations: List[Dict] = field(default_factory=list)

    # Context
    customer_message: str = ""
    destination: str = ""               # primary input variable
    tool_failures: List[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "scenario_id": self.scenario_id,
            "category": self.category,
            "passed": self.passed,
            "verdict": self.verdict,
            "action": self.action,
            "expected_action": self.expected_action,
            "difficulty": self.difficulty.to_dict() if self.difficulty else None,
            "failure_tag": self.failure_tag,
            "failure_reason": self.failure_reason,
            "violations": self.violations,
            "customer_message": self.customer_message[:100],
            "destination": self.destination,
            "tool_failures": self.tool_failures,
            "duration_ms": self.duration_ms,
        }


def _classify_failure(result: ScenarioResult) -> tuple:
    """Classify a failure into a category tag and reason."""
    if not result.violations:
        return "", ""

    violation_names = [v.rule_name for v in result.violations]

    if "agent_no_crash" in violation_names:
        crash_v = next(v for v in result.violations if v.rule_name == "agent_no_crash")
        return "crash", crash_v.actual[:100]

    if "expected_action" in violation_names:
        v = next(v for v in result.violations if v.rule_name == "expected_action")
        return "policy_violation", f"Expected {v.expected}, got {v.actual}"

    if "unnecessary_error" in violation_names:
        return "unnecessary_error", "Agent returned error when tools were working"

    if any(n in ("required_field_missing", "tool_data_missing") for n in violation_names):
        return "incomplete_response", "Response missing required data"

    if "placeholder_response" in violation_names:
        return "quality_issue", "Response contains placeholder text"

    if "response_too_short" in violation_names:
        return "quality_issue", "Response too short"

    if "agent_produced_output" in violation_names:
        return "crash", "Agent returned empty output"

    # Generic
    return "evaluation_failure", result.violations[0].details or result.violations[0].actual[:100]


@dataclass
class EvaluationReport:
    """
    Complete evaluation report for an agent test run.
    """
    # Metadata
    agent_name: str = ""
    plan_name: str = ""
    generated_at: str = ""
    total_duration_ms: float = 0.0
    total_tokens: int = 0

    # Summary
    total_scenarios: int = 0
    total_passed: int = 0
    total_degraded: int = 0
    total_failed: int = 0
    pass_rate: float = 0.0

    # Difficulty breakdown
    difficulty_summary: Dict = field(default_factory=dict)
    pass_rate_by_difficulty: Dict[str, float] = field(default_factory=dict)

    # Per-scenario verdicts
    verdicts: List[ScenarioVerdict] = field(default_factory=list)

    # Failure summary by tag
    failures_by_tag: Dict[str, int] = field(default_factory=dict)

    # From FailureAnalysis
    resilience: Optional[Dict] = None
    clusters: List[Dict] = field(default_factory=list)
    correlations: List[Dict] = field(default_factory=list)
    counterfactuals: List[Dict] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        results: List[ScenarioResult],
        scenarios: List[Scenario],
        analysis: FailureAnalysis,
        agent_name: str = "",
        plan_name: str = "",
        traces: Optional[List] = None,
    ) -> "EvaluationReport":
        """Build a report from run results."""
        scorer = DifficultyScorer()

        verdicts = []
        failures_by_tag = {}
        difficulty_pass = {}  # level → (passed, total)

        for scenario, result in zip(scenarios, results):
            diff = scorer.score(scenario)

            # Classify failure
            tag, reason = ("", "") if result.passed else _classify_failure(result)

            # Determine verdict: passed / degraded / failed
            result_verdict = getattr(result, "verdict", "")
            if not result_verdict:
                # Backward compat: derive from passed + injected failures
                OUTAGE_BEHAVIORS = {"timeout", "empty_response", "rate_limited",
                                    "schema_drift", "malformed_response", "intermittent"}
                injected = result.world_config.get("injected_failures", [])
                has_outage = any(
                    f.split(":")[-1] in OUTAGE_BEHAVIORS for f in injected if ":" in f
                )
                if not result.passed:
                    result_verdict = "failed"
                elif has_outage:
                    result_verdict = "degraded"
                else:
                    result_verdict = "passed"

            # Find primary input variable
            dest = ""
            for key in ("destination", "location", "city", "query", "name", "order_id", "id"):
                if key in result.scenario and result.scenario[key]:
                    dest = str(result.scenario[key])
                    break

            verdict = ScenarioVerdict(
                scenario_id=result.scenario_id,
                category=result.category,
                passed=result.passed,
                verdict=result_verdict,
                expected_action=scenario.expected_action,
                difficulty=diff,
                failure_tag=tag,
                failure_reason=reason,
                violations=[{"rule": v.rule_name, "severity": v.severity,
                             "expected": v.expected, "actual": v.actual}
                            for v in result.violations],
                customer_message=scenario.customer_message,
                destination=dest,
                tool_failures=result.world_config.get("injected_failures", []),
                duration_ms=result.metadata.get("duration_ms", 0),
            )
            verdicts.append(verdict)

            # Aggregate by failure tag
            if tag:
                failures_by_tag[tag] = failures_by_tag.get(tag, 0) + 1

            # Aggregate pass rate by difficulty (only full passes, not degraded)
            level = diff.level
            if level not in difficulty_pass:
                difficulty_pass[level] = [0, 0, 0]  # [passed, degraded, total]
            difficulty_pass[level][2] += 1
            if result_verdict == "passed":
                difficulty_pass[level][0] += 1
            elif result_verdict == "degraded":
                difficulty_pass[level][1] += 1

        pass_rate_by_diff = {
            level: round(counts[0] / counts[2] * 100, 1) if counts[2] else 0
            for level, counts in difficulty_pass.items()
        }

        # Total tokens from traces
        total_tokens = 0
        total_duration = 0.0
        if traces:
            total_tokens = sum(t.total_tokens for t in traces)
            total_duration = sum(t.duration_ms or 0 for t in traces)

        passed = sum(1 for v in verdicts if v.verdict == "passed")
        degraded = sum(1 for v in verdicts if v.verdict == "degraded")
        failed = sum(1 for v in verdicts if v.verdict == "failed")

        return cls(
            agent_name=agent_name,
            plan_name=plan_name,
            generated_at=datetime.now().isoformat(),
            total_duration_ms=total_duration,
            total_tokens=total_tokens,
            total_scenarios=len(results),
            total_passed=passed,
            total_degraded=degraded,
            total_failed=failed,
            pass_rate=round(passed / len(results) * 100, 1) if results else 0,
            difficulty_summary=scorer.summary(scenarios),
            pass_rate_by_difficulty=pass_rate_by_diff,
            verdicts=verdicts,
            failures_by_tag=failures_by_tag,
            resilience=analysis.resilience.to_dict() if analysis.resilience else None,
            clusters=[{
                "id": c.cluster_id, "count": c.count,
                "rules": c.rule_violations, "tools": c.world_failures,
                "severity": c.severity, "fix": c.suggested_fix,
            } for c in analysis.clusters],
            correlations=[{
                "description": c.description,
                "risk": c.relative_risk,
                "rate_true": c.failure_rate_when_true,
                "rate_false": c.failure_rate_when_false,
            } for c in analysis.correlations],
            counterfactuals=[{
                "scenario_id": cf.scenario_id,
                "description": cf.description,
                "confidence": cf.confidence,
            } for cf in analysis.counterfactuals[:20]],
        )

    def render(self) -> str:
        """Render as formatted text."""
        lines = []
        w = 60
        lines.append("=" * w)
        lines.append("  PROBE-AI EVALUATION REPORT")
        lines.append("=" * w)
        lines.append(f"  Agent: {self.agent_name or '(unnamed)'}")
        lines.append(f"  Plan:  {self.plan_name or '(unnamed)'}")
        lines.append(f"  Date:  {self.generated_at[:19]}")
        lines.append("")
        lines.append(f"  Scenarios: {self.total_scenarios}")
        lines.append(f"  Passed:    {self.total_passed} ({self.pass_rate}%)")
        lines.append(f"  Degraded:  {self.total_degraded} (tools failed, agent handled gracefully)")
        lines.append(f"  Failed:    {self.total_failed}")
        lines.append(f"  Tokens:    {self.total_tokens:,}")
        lines.append(f"  Duration:  {self.total_duration_ms/1000:.0f}s")

        # Resilience
        if self.resilience:
            lines.append("")
            lines.append(f"  Resilience: {self.resilience.get('overall', 0):.0f}/100 "
                        f"(Grade {self.resilience.get('grade', '?')})")
            lines.append(f"    Normal:   {self.resilience.get('normal_pass_rate', 0):.1%}")
            lines.append(f"    Degraded: {self.resilience.get('degraded_pass_rate', 0):.1%}")

        # Difficulty breakdown
        if self.pass_rate_by_difficulty:
            lines.append("")
            lines.append("  Pass Rate by Difficulty:")
            for level in ["easy", "medium", "hard", "adversarial"]:
                if level in self.pass_rate_by_difficulty:
                    dist = self.difficulty_summary.get("distribution", {})
                    count = dist.get(level, 0)
                    if count == 0:
                        continue
                    rate = self.pass_rate_by_difficulty[level]
                    lines.append(f"    {level:14s} {rate:5.1f}% ({count} scenarios)")

        # Failure summary
        if self.failures_by_tag:
            lines.append("")
            lines.append("  Failures by Type:")
            for tag, count in sorted(self.failures_by_tag.items(), key=lambda x: -x[1]):
                lines.append(f"    {tag:25s} {count}")

        # Clusters
        if self.clusters:
            lines.append("")
            lines.append(f"  Failure Clusters ({len(self.clusters)}):")
            for c in self.clusters:
                lines.append(f"    #{c['id']}: {c['count']} failures — {c['rules']}")
                if c.get("fix"):
                    lines.append(f"      Fix: {c['fix']}")

        # Failed scenarios
        failed = [v for v in self.verdicts if not v.passed]
        if failed:
            lines.append("")
            lines.append(f"  Failed Scenarios ({len(failed)}):")
            for v in failed[:15]:
                dest = v.destination or v.customer_message[:30]
                lines.append(f"    [{v.failure_tag:20s}] {v.category:25s} {dest}")
                if v.failure_reason:
                    lines.append(f"      → {v.failure_reason[:80]}")

        lines.append("")
        lines.append("=" * w)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "agent_name": self.agent_name,
            "plan_name": self.plan_name,
            "generated_at": self.generated_at,
            "summary": {
                "total": self.total_scenarios,
                "passed": self.total_passed,
                "degraded": self.total_degraded,
                "failed": self.total_failed,
                "pass_rate": self.pass_rate,
                "tokens": self.total_tokens,
                "duration_ms": self.total_duration_ms,
            },
            "resilience": self.resilience,
            "difficulty": self.difficulty_summary,
            "pass_rate_by_difficulty": self.pass_rate_by_difficulty,
            "failures_by_tag": self.failures_by_tag,
            "clusters": self.clusters,
            "correlations": self.correlations,
            "counterfactuals": self.counterfactuals,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }

    def to_json(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path
