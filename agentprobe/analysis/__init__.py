"""
agentprobe.analysis — Failure clustering, root cause attribution, and counterfactual analysis.

Three capabilities that work together:

1. FailureClustering: Groups failures by root cause pattern across hundreds
   of scenarios. Instead of "47 failed," you get "47 failed in 5 root cause
   groups: 12 from tool timeouts, 8 from contradictory data, ..."

2. CorrelationDetector: Finds statistical correlations between scenario
   dimensions and failure rates. "Failure rate is 3x higher when customer
   message exceeds 200 words" or "12% worse performance on southern states."

3. CounterfactualAnalyzer: For each failure, systematically mutates one
   variable at a time to find the minimum change that would have produced
   a pass. "This failed because days_since_delivery was 35. At 30, it passes."

Architecture:
    ScenarioResult (one scenario's outcome)
    ├── passed: bool
    ├── scenario: Dict (the input)
    ├── world_config: WorldConfiguration (tool behaviors)
    ├── trace: List (agent's execution trace)
    ├── violations: List[RuleViolation] (which rules were violated)
    └── metadata: Dict (timing, tokens, etc.)

    FailureAnalysis (output of full analysis)
    ├── clusters: List[FailureCluster]
    ├── correlations: List[Correlation]
    ├── counterfactuals: List[Counterfactual]
    └── resilience_score: float
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# ============================================================================
# Input: Scenario Results
# ============================================================================

@dataclass
class RuleViolation:
    """A single rule that was violated in a scenario."""
    rule_name: str
    severity: str                    # critical, major, warning
    expected: str                    # what should have happened
    actual: str                      # what actually happened
    step: Optional[int] = None       # which step in the trace
    details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "rule_name": self.rule_name,
            "severity": self.severity,
            "expected": self.expected,
            "actual": self.actual,
        }
        if self.step is not None:
            d["step"] = self.step
        if self.details:
            d["details"] = self.details
        return d


@dataclass
class ScenarioResult:
    """The outcome of running one scenario against the agent."""
    scenario_id: str
    passed: bool
    scenario: Dict[str, Any]         # the input (customer message, variables)
    world_config: Dict[str, Any]     # tool behaviors (from WorldConfiguration.to_dict())
    violations: List[RuleViolation] = field(default_factory=list)
    trace: Optional[List[Dict]] = None
    category: str = ""               # which scenario category
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Three-verdict model: "passed" (complete, correct), "degraded" (no violations
    # but output is incomplete due to tool failures), "failed" (violations or crash)
    verdict: str = ""                # set by Runner or post-hoc

    @property
    def failure_signatures(self) -> List[str]:
        """Unique failure signature for clustering."""
        if self.passed:
            return []
        sigs = []
        for v in self.violations:
            sigs.append(f"{v.rule_name}:{v.severity}")
        # Include world failures in the signature
        injected = self.world_config.get("injected_failures", [])
        for f in injected:
            sigs.append(f"world:{f}")
        return sorted(sigs)

    @property
    def signature_key(self) -> str:
        """Single string signature for grouping."""
        return "|".join(self.failure_signatures)


# ============================================================================
# Failure Clustering
# ============================================================================

@dataclass
class FailureCluster:
    """A group of failures that share the same root cause pattern."""
    cluster_id: int
    signature: str                   # the shared failure signature
    count: int                       # number of failures in this cluster
    scenario_ids: List[str]          # which scenarios are in this cluster
    rule_violations: List[str]       # which rules were violated
    world_failures: List[str]        # which tool failures were injected
    severity: str                    # worst severity in the cluster
    description: str = ""            # LLM-generated or template description
    suggested_fix: str = ""          # what might fix this class of failures

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "signature": self.signature,
            "count": self.count,
            "scenario_ids": self.scenario_ids[:10],  # cap for readability
            "rule_violations": self.rule_violations,
            "world_failures": self.world_failures,
            "severity": self.severity,
            "description": self.description,
            "suggested_fix": self.suggested_fix,
        }


def cluster_failures(results: List[ScenarioResult]) -> List[FailureCluster]:
    """
    Group failed scenarios by their failure signature.

    Failures with the same combination of rule violations and world failures
    are likely to have the same root cause. This groups them together so the
    report can say "12 failures from tool timeouts" instead of listing 12
    individual failures.
    """
    failures = [r for r in results if not r.passed]
    if not failures:
        return []

    # Group by signature
    groups: Dict[str, List[ScenarioResult]] = defaultdict(list)
    for r in failures:
        groups[r.signature_key].append(r)

    # Build clusters
    clusters = []
    severity_order = {"critical": 0, "major": 1, "warning": 2, "info": 3}

    for i, (sig, members) in enumerate(
        sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    ):
        # Extract rule violations and world failures from signature
        rule_violations = []
        world_failures = []
        for part in sig.split("|"):
            if part.startswith("world:"):
                world_failures.append(part[6:])
            else:
                rule_violations.append(part)

        # Determine worst severity
        worst = "info"
        for member in members:
            for v in member.violations:
                if severity_order.get(v.severity, 3) < severity_order.get(worst, 3):
                    worst = v.severity

        # Generate description
        desc_parts = []
        if world_failures:
            desc_parts.append(f"Tool failures: {', '.join(world_failures)}")
        if rule_violations:
            rules = [rv.split(":")[0] for rv in rule_violations]
            desc_parts.append(f"Violated: {', '.join(rules)}")
        description = ". ".join(desc_parts)

        # Generate suggested fix based on pattern
        suggested_fix = _suggest_fix(rule_violations, world_failures)

        clusters.append(FailureCluster(
            cluster_id=i,
            signature=sig,
            count=len(members),
            scenario_ids=[m.scenario_id for m in members],
            rule_violations=rule_violations,
            world_failures=world_failures,
            severity=worst,
            description=description,
            suggested_fix=suggested_fix,
        ))

    return clusters


def _suggest_fix(rule_violations: List[str], world_failures: List[str]) -> str:
    """Generate a fix suggestion based on failure pattern."""
    suggestions = []

    for wf in world_failures:
        if "timeout" in wf:
            suggestions.append("Add retry logic with exponential backoff for tool timeouts")
        elif "partial_data" in wf:
            suggestions.append("Add null checks and handle missing fields gracefully")
        elif "rate_limited" in wf:
            suggestions.append("Implement rate limit handling with queue/backoff")
        elif "contradiction" in wf:
            suggestions.append("Add data consistency checks across tool responses")
        elif "empty_response" in wf:
            suggestions.append("Handle empty/null tool responses explicitly")
        elif "stale_data" in wf:
            suggestions.append("Check data freshness and warn user about stale results")
        elif "cascade" in wf:
            suggestions.append("Implement circuit breaker pattern for dependent tool calls")

    for rv in rule_violations:
        rule = rv.split(":")[0]
        if "escalat" in rule.lower():
            suggestions.append(f"Review escalation logic for rule '{rule}'")
        elif "window" in rule.lower() or "threshold" in rule.lower():
            suggestions.append(f"Check boundary condition handling for rule '{rule}'")

    return "; ".join(suggestions) if suggestions else "Review agent logic for this failure pattern"


# ============================================================================
# Correlation Detection
# ============================================================================

@dataclass
class Correlation:
    """A statistical correlation between a scenario dimension and failure rate."""
    dimension: str                   # which variable (e.g., "days_since_delivery")
    condition: str                   # human-readable condition (e.g., "> 28")
    failure_rate_when_true: float    # failure rate when condition is true
    failure_rate_when_false: float   # failure rate when condition is false
    relative_risk: float             # ratio of the two rates
    sample_size_true: int            # how many scenarios match this condition
    sample_size_false: int
    p_value_approx: float            # approximate significance
    description: str = ""            # human-readable description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "condition": self.condition,
            "failure_rate_when_true": round(self.failure_rate_when_true, 3),
            "failure_rate_when_false": round(self.failure_rate_when_false, 3),
            "relative_risk": round(self.relative_risk, 2),
            "sample_size_true": self.sample_size_true,
            "sample_size_false": self.sample_size_false,
            "p_value_approx": round(self.p_value_approx, 4),
            "description": self.description,
        }


def detect_correlations(
    results: List[ScenarioResult],
    dimension_extractors: Optional[Dict[str, Callable]] = None,
    min_sample_size: int = 10,
    min_relative_risk: float = 1.5,
) -> List[Correlation]:
    """
    Find dimensions that correlate with failure rate.

    For each dimension in the scenario data, tests whether scenarios
    with certain values fail more often than others.

    Args:
        results: All scenario results (passed and failed).
        dimension_extractors: Optional mapping of dimension names to functions
            that extract the dimension value from a scenario dict.
            If None, auto-discovers numeric and categorical fields.
        min_sample_size: Minimum scenarios in each group to report.
        min_relative_risk: Minimum relative risk to report (1.5 = 50% more failures).

    Returns:
        List of significant correlations, sorted by relative risk.
    """
    if len(results) < 20:
        return []  # not enough data for meaningful correlations

    correlations = []

    # Auto-discover dimensions from scenario data
    if dimension_extractors is None:
        dimension_extractors = _auto_discover_dimensions(results)

    overall_failure_rate = sum(1 for r in results if not r.passed) / len(results)

    for dim_name, extractor in dimension_extractors.items():
        # Extract values
        values = []
        for r in results:
            try:
                val = extractor(r.scenario)
                if val is not None:
                    values.append((val, not r.passed))  # (value, is_failure)
            except (KeyError, TypeError):
                continue

        if len(values) < min_sample_size * 2:
            continue

        # For numeric dimensions: test above/below median
        if all(isinstance(v[0], (int, float)) for v in values):
            corrs = _test_numeric_dimension(dim_name, values, min_sample_size, min_relative_risk)
            correlations.extend(corrs)

        # For categorical dimensions: test each value
        elif all(isinstance(v[0], str) for v in values):
            corrs = _test_categorical_dimension(dim_name, values, min_sample_size, min_relative_risk)
            correlations.extend(corrs)

    # Also check world configuration correlations
    world_corrs = _check_world_correlations(results, min_sample_size, min_relative_risk)
    correlations.extend(world_corrs)

    # Sort by relative risk (highest first)
    correlations.sort(key=lambda c: c.relative_risk, reverse=True)
    return correlations


def _auto_discover_dimensions(results: List[ScenarioResult]) -> Dict[str, Callable]:
    """Auto-discover extractable dimensions from scenario data."""
    extractors = {}
    sample = results[0].scenario if results else {}

    def _make_extractor(key):
        return lambda s: s.get(key)

    def _walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                path = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (int, float, str, bool)):
                    extractors[path] = _make_extractor_path(path)
                elif isinstance(v, dict):
                    _walk(v, path)

    def _make_extractor_path(path):
        parts = path.split(".")
        def extract(s):
            current = s
            for p in parts:
                if isinstance(current, dict):
                    current = current.get(p)
                else:
                    return None
            return current
        return extract

    _walk(sample)
    return extractors


def _test_numeric_dimension(
    dim_name: str,
    values: List[Tuple[float, bool]],
    min_sample: int,
    min_rr: float,
) -> List[Correlation]:
    """Test numeric dimension at multiple split points."""
    correlations = []
    nums = sorted(set(v[0] for v in values))

    # Test at quartile boundaries
    n = len(nums)
    split_points = set()
    for q in [0.25, 0.5, 0.75]:
        idx = int(n * q)
        if 0 < idx < n:
            split_points.add(nums[idx])

    for split in split_points:
        above = [(v, f) for v, f in values if v > split]
        below = [(v, f) for v, f in values if v <= split]

        if len(above) < min_sample or len(below) < min_sample:
            continue

        rate_above = sum(1 for _, f in above if f) / len(above)
        rate_below = sum(1 for _, f in below if f) / len(below)

        if rate_below == 0:
            continue

        rr = rate_above / rate_below if rate_below > 0 else float('inf')

        if rr >= min_rr or (1 / rr if rr > 0 else float('inf')) >= min_rr:
            if rr >= min_rr:
                condition = f"> {split}"
                fr_true, fr_false = rate_above, rate_below
                n_true, n_false = len(above), len(below)
            else:
                condition = f"<= {split}"
                fr_true, fr_false = rate_below, rate_above
                n_true, n_false = len(below), len(above)
                rr = 1 / rr

            p_val = _approximate_p_value(
                n_true, int(fr_true * n_true),
                n_false, int(fr_false * n_false)
            )

            correlations.append(Correlation(
                dimension=dim_name,
                condition=condition,
                failure_rate_when_true=fr_true,
                failure_rate_when_false=fr_false,
                relative_risk=rr,
                sample_size_true=n_true,
                sample_size_false=n_false,
                p_value_approx=p_val,
                description=(
                    f"Failure rate is {rr:.1f}x higher when {dim_name} {condition} "
                    f"({fr_true:.0%} vs {fr_false:.0%})"
                ),
            ))

    return correlations


def _test_categorical_dimension(
    dim_name: str,
    values: List[Tuple[str, bool]],
    min_sample: int,
    min_rr: float,
) -> List[Correlation]:
    """Test each category value against the rest."""
    correlations = []
    overall_failures = sum(1 for _, f in values if f)
    overall_rate = overall_failures / len(values)

    # Group by category
    by_cat: Dict[str, List[bool]] = defaultdict(list)
    for val, failed in values:
        by_cat[val].append(failed)

    for cat_val, failures_list in by_cat.items():
        n_cat = len(failures_list)
        if n_cat < min_sample:
            continue

        n_rest = len(values) - n_cat
        if n_rest < min_sample:
            continue

        rate_cat = sum(failures_list) / n_cat
        rate_rest = (overall_failures - sum(failures_list)) / n_rest

        if rate_rest == 0:
            continue

        rr = rate_cat / rate_rest
        if rr >= min_rr:
            p_val = _approximate_p_value(
                n_cat, sum(failures_list),
                n_rest, overall_failures - sum(failures_list)
            )

            correlations.append(Correlation(
                dimension=dim_name,
                condition=f"= '{cat_val}'",
                failure_rate_when_true=rate_cat,
                failure_rate_when_false=rate_rest,
                relative_risk=rr,
                sample_size_true=n_cat,
                sample_size_false=n_rest,
                p_value_approx=p_val,
                description=(
                    f"Failure rate is {rr:.1f}x higher when {dim_name} = '{cat_val}' "
                    f"({rate_cat:.0%} vs {rate_rest:.0%})"
                ),
            ))

    return correlations


def _check_world_correlations(
    results: List[ScenarioResult],
    min_sample: int,
    min_rr: float,
) -> List[Correlation]:
    """Check if world/tool failures correlate with agent failures."""
    correlations = []

    # Compare: scenarios with tool failures vs without
    with_failures = [r for r in results if r.world_config.get("has_failures", False)]
    without_failures = [r for r in results if not r.world_config.get("has_failures", False)]

    if len(with_failures) >= min_sample and len(without_failures) >= min_sample:
        rate_with = sum(1 for r in with_failures if not r.passed) / len(with_failures)
        rate_without = sum(1 for r in without_failures if not r.passed) / len(without_failures)

        if rate_without > 0:
            rr = rate_with / rate_without
            if rr >= min_rr:
                correlations.append(Correlation(
                    dimension="world_has_tool_failures",
                    condition="= True",
                    failure_rate_when_true=rate_with,
                    failure_rate_when_false=rate_without,
                    relative_risk=rr,
                    sample_size_true=len(with_failures),
                    sample_size_false=len(without_failures),
                    p_value_approx=_approximate_p_value(
                        len(with_failures), int(rate_with * len(with_failures)),
                        len(without_failures), int(rate_without * len(without_failures)),
                    ),
                    description=(
                        f"Failure rate is {rr:.1f}x higher when tools have injected failures "
                        f"({rate_with:.0%} vs {rate_without:.0%})"
                    ),
                ))

    # Check specific failure types
    failure_type_counts: Dict[str, List[bool]] = defaultdict(list)
    for r in results:
        injected = r.world_config.get("injected_failures", [])
        for f in injected:
            failure_type = f.split(":")[1].split("[")[0] if ":" in f else f
            failure_type_counts[failure_type].append(not r.passed)

    for ftype, agent_failures in failure_type_counts.items():
        if len(agent_failures) < min_sample:
            continue
        n_rest = len(results) - len(agent_failures)
        if n_rest < min_sample:
            continue

        rate_ftype = sum(agent_failures) / len(agent_failures)
        total_failures = sum(1 for r in results if not r.passed)
        rate_rest = (total_failures - sum(agent_failures)) / n_rest

        if rate_rest > 0:
            rr = rate_ftype / rate_rest
            if rr >= min_rr:
                correlations.append(Correlation(
                    dimension=f"tool_failure_type",
                    condition=f"= '{ftype}'",
                    failure_rate_when_true=rate_ftype,
                    failure_rate_when_false=rate_rest,
                    relative_risk=rr,
                    sample_size_true=len(agent_failures),
                    sample_size_false=n_rest,
                    p_value_approx=0.01,  # simplified
                    description=(
                        f"Agent failure rate is {rr:.1f}x higher when tools experience "
                        f"'{ftype}' failures ({rate_ftype:.0%} vs {rate_rest:.0%})"
                    ),
                ))

    return correlations


def _approximate_p_value(n1: int, f1: int, n2: int, f2: int) -> float:
    """
    Approximate p-value using a two-proportion z-test.

    This is a rough approximation — good enough for surfacing patterns,
    not for publishing papers.
    """
    if n1 == 0 or n2 == 0:
        return 1.0

    p1 = f1 / n1
    p2 = f2 / n2
    p_pool = (f1 + f2) / (n1 + n2)

    if p_pool == 0 or p_pool == 1:
        return 1.0

    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0

    z = abs(p1 - p2) / se
    # Rough p-value approximation from z-score
    p_val = math.exp(-0.5 * z * z) / (z * math.sqrt(2 * math.pi)) if z > 0 else 1.0
    return min(1.0, p_val * 2)  # two-tailed


# ============================================================================
# Counterfactual Analysis
# ============================================================================

@dataclass
class Counterfactual:
    """
    A counterfactual finding: "if X had been different, the outcome would change."

    Example:
        scenario_id: "scn_047"
        dimension: "days_since_delivery"
        original_value: 35
        counterfactual_value: 30
        original_passed: False
        counterfactual_would_pass: True
        description: "Would have passed if days_since_delivery were 30 instead of 35"
    """
    scenario_id: str
    dimension: str
    original_value: Any
    counterfactual_value: Any
    original_passed: bool
    counterfactual_would_pass: bool
    confidence: float = 0.0          # how confident are we in this counterfactual
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "dimension": self.dimension,
            "original_value": self.original_value,
            "counterfactual_value": self.counterfactual_value,
            "original_passed": self.original_passed,
            "counterfactual_would_pass": self.counterfactual_would_pass,
            "confidence": round(self.confidence, 2),
            "description": self.description,
        }


def generate_counterfactuals(
    failed_results: List[ScenarioResult],
    all_results: List[ScenarioResult],
    dimensions_to_test: Optional[List[str]] = None,
    max_per_failure: int = 3,
) -> List[Counterfactual]:
    """
    For each failed scenario, find the minimum change that would produce a pass.

    Strategy:
    1. For each failed scenario, look at passing scenarios with similar profiles
    2. Identify which dimensions differ between the failed and passing cases
    3. Report the single-dimension changes most likely to flip the result

    This is a heuristic approach — not causal inference. For true counterfactuals,
    you'd need to re-run the agent with mutated inputs. But this gives strong
    directional signals about what's causing failures.

    Args:
        failed_results: Scenarios that failed.
        all_results: All scenarios (passed and failed).
        dimensions_to_test: Optional list of dimension names to check.
        max_per_failure: Maximum counterfactuals to generate per failed scenario.

    Returns:
        List of counterfactual findings.
    """
    if not failed_results:
        return []

    passing = [r for r in all_results if r.passed]
    if not passing:
        return []

    counterfactuals = []

    # Build index of passing scenarios by category
    passing_by_category: Dict[str, List[ScenarioResult]] = defaultdict(list)
    for r in passing:
        passing_by_category[r.category].append(r)

    for failed in failed_results:
        # Find passing scenarios in the same category
        candidates = passing_by_category.get(failed.category, passing)
        if not candidates:
            continue

        # Find the "nearest pass" — the passing scenario most similar to this failure
        diffs = _find_dimension_diffs(failed, candidates, dimensions_to_test)

        # Take top-N most likely explanatory diffs
        for dim, orig, counterfactual_val, confidence in diffs[:max_per_failure]:
            counterfactuals.append(Counterfactual(
                scenario_id=failed.scenario_id,
                dimension=dim,
                original_value=orig,
                counterfactual_value=counterfactual_val,
                original_passed=False,
                counterfactual_would_pass=True,
                confidence=confidence,
                description=(
                    f"Would likely pass if {dim} were {counterfactual_val!r} "
                    f"instead of {orig!r}"
                ),
            ))

    # Also generate world-level counterfactuals
    for failed in failed_results:
        injected = failed.world_config.get("injected_failures", [])
        if injected:
            # Check: do scenarios with same input but no tool failures pass?
            similar_passing = [
                r for r in passing
                if r.category == failed.category
                and not r.world_config.get("has_failures", False)
            ]
            if similar_passing:
                counterfactuals.append(Counterfactual(
                    scenario_id=failed.scenario_id,
                    dimension="tool_failures",
                    original_value=injected,
                    counterfactual_value="none",
                    original_passed=False,
                    counterfactual_would_pass=True,
                    confidence=0.7,
                    description=(
                        f"Would likely pass without tool failures: {', '.join(injected)}"
                    ),
                ))

    return counterfactuals


def _find_dimension_diffs(
    failed: ScenarioResult,
    candidates: List[ScenarioResult],
    dimensions: Optional[List[str]],
) -> List[Tuple[str, Any, Any, float]]:
    """
    Find dimensions where the failed scenario differs from passing ones.

    Returns: List of (dimension, original_value, counterfactual_value, confidence)
    sorted by confidence (how likely changing this dimension would flip the result).
    """
    diffs = []

    # Extract dimensions from the failed scenario
    failed_dims = _extract_flat_dims(failed.scenario)

    # Score each passing candidate by similarity
    best_candidates = []
    for candidate in candidates:
        cand_dims = _extract_flat_dims(candidate.scenario)
        shared_keys = set(failed_dims.keys()) & set(cand_dims.keys())
        if not shared_keys:
            continue

        # Count matching dimensions
        matches = sum(1 for k in shared_keys if failed_dims[k] == cand_dims[k])
        similarity = matches / len(shared_keys) if shared_keys else 0
        differing = [(k, failed_dims[k], cand_dims[k])
                     for k in shared_keys if failed_dims[k] != cand_dims[k]]
        best_candidates.append((similarity, differing, candidate))

    # Sort by similarity (most similar first) — the nearest passing neighbor
    best_candidates.sort(key=lambda x: x[0], reverse=True)

    # Count how often each dimension appears as a diff across nearest neighbors
    dim_diff_counts: Dict[str, List[Tuple[Any, Any]]] = defaultdict(list)
    for sim, differing, _ in best_candidates[:20]:  # top 20 nearest
        for dim, orig, other in differing:
            if dimensions and dim not in dimensions:
                continue
            dim_diff_counts[dim].append((orig, other))

    # Confidence = how often this dimension differs in the nearest passing neighbors
    for dim, values in dim_diff_counts.items():
        # Pick the most common counterfactual value
        counter = Counter(v[1] for v in values)
        most_common_val, count = counter.most_common(1)[0]
        confidence = count / min(20, len(best_candidates))
        orig_val = values[0][0]

        diffs.append((dim, orig_val, most_common_val, confidence))

    # Sort by confidence
    diffs.sort(key=lambda x: x[3], reverse=True)
    return diffs


def _extract_flat_dims(scenario: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dict into dot-notation keys."""
    flat = {}
    for k, v in scenario.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_extract_flat_dims(v, key))
        elif isinstance(v, (str, int, float, bool)):
            flat[key] = v
    return flat


# ============================================================================
# Full Analysis Pipeline
# ============================================================================

@dataclass
class ResilienceScore:
    """Overall resilience assessment."""
    overall: float                   # 0-100
    normal_pass_rate: float          # pass rate under normal conditions
    degraded_pass_rate: float        # pass rate under tool failures
    resilience_gap: float            # difference between normal and degraded
    grade: str                       # A/B/C/D/F

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": round(self.overall, 1),
            "normal_pass_rate": round(self.normal_pass_rate, 3),
            "degraded_pass_rate": round(self.degraded_pass_rate, 3),
            "resilience_gap": round(self.resilience_gap, 3),
            "grade": self.grade,
        }


@dataclass
class FailureAnalysis:
    """Complete failure analysis output."""
    total_scenarios: int
    total_passed: int
    total_failed: int
    pass_rate: float
    clusters: List[FailureCluster]
    correlations: List[Correlation]
    counterfactuals: List[Counterfactual]
    resilience: ResilienceScore

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_scenarios": self.total_scenarios,
            "total_passed": self.total_passed,
            "total_failed": self.total_failed,
            "pass_rate": round(self.pass_rate, 3),
            "clusters": [c.to_dict() for c in self.clusters],
            "correlations": [c.to_dict() for c in self.correlations],
            "counterfactuals": [c.to_dict() for c in self.counterfactuals[:50]],
            "resilience": self.resilience.to_dict(),
        }

    def render(self) -> str:
        """Human-readable analysis report."""
        lines = []
        lines.append("=" * 60)
        lines.append("  FAILURE ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Summary
        lines.append(f"  Total: {self.total_scenarios} scenarios | "
                      f"{self.total_passed} passed ({self.pass_rate:.0%}) | "
                      f"{self.total_failed} failed")
        lines.append("")

        # Resilience Score
        r = self.resilience
        lines.append(f"  RESILIENCE SCORE: {r.overall:.0f}/100 (Grade: {r.grade})")
        lines.append(f"    Normal conditions:    {r.normal_pass_rate:.0%} pass rate")
        lines.append(f"    Under tool failures:  {r.degraded_pass_rate:.0%} pass rate")
        lines.append(f"    Resilience gap:       {r.resilience_gap:.0%}")
        lines.append("")

        # Failure Clusters
        if self.clusters:
            lines.append(f"  FAILURE CLUSTERS ({len(self.clusters)} root causes):")
            lines.append("")
            severity_icons = {"critical": "🔴", "major": "🟠", "warning": "🟡", "info": "ℹ️"}
            for cluster in self.clusters[:10]:
                icon = severity_icons.get(cluster.severity, "•")
                lines.append(f"    {icon} {cluster.count} failures: {cluster.description}")
                if cluster.suggested_fix:
                    lines.append(f"      Fix: {cluster.suggested_fix}")
                lines.append(f"      Scenarios: {', '.join(cluster.scenario_ids[:5])}"
                             f"{'...' if len(cluster.scenario_ids) > 5 else ''}")
                lines.append("")

        # Correlations
        if self.correlations:
            lines.append(f"  POPULATION-LEVEL PATTERNS ({len(self.correlations)} found):")
            lines.append("")
            for corr in self.correlations[:10]:
                lines.append(f"    📊 {corr.description}")
                lines.append(f"       (n={corr.sample_size_true}, p≈{corr.p_value_approx:.3f})")
            lines.append("")

        # Counterfactuals
        if self.counterfactuals:
            lines.append(f"  COUNTERFACTUAL ANALYSIS ({len(self.counterfactuals)} findings):")
            lines.append("")
            # Group by dimension
            by_dim: Dict[str, List[Counterfactual]] = defaultdict(list)
            for cf in self.counterfactuals:
                by_dim[cf.dimension].append(cf)

            for dim, cfs in sorted(by_dim.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
                lines.append(f"    🔄 {dim}: {len(cfs)} failures affected")
                # Show most common counterfactual
                example = max(cfs, key=lambda c: c.confidence)
                lines.append(f"       Example: {example.description}")
                lines.append(f"       Confidence: {example.confidence:.0%}")
                lines.append("")

        lines.append("─" * 60)
        return "\n".join(lines)


def analyze_failures(
    results: List[ScenarioResult],
    dimension_extractors: Optional[Dict[str, Callable]] = None,
) -> FailureAnalysis:
    """
    Run the complete failure analysis pipeline.

    Args:
        results: All scenario results.
        dimension_extractors: Optional dimension extractors for correlation detection.

    Returns:
        Complete FailureAnalysis with clusters, correlations, counterfactuals,
        and resilience score.
    """
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    pass_rate = passed / total if total > 0 else 0

    # Cluster failures
    clusters = cluster_failures(results)

    # Detect correlations
    correlations = detect_correlations(results, dimension_extractors)

    # Generate counterfactuals
    failed_results = [r for r in results if not r.passed]
    counterfactuals = generate_counterfactuals(failed_results, results)

    # Calculate resilience score
    normal_results = [r for r in results
                      if not r.world_config.get("has_failures", False)]
    degraded_results = [r for r in results
                        if r.world_config.get("has_failures", False)]

    normal_pass = (sum(1 for r in normal_results if r.passed) / len(normal_results)
                   if normal_results else pass_rate)
    degraded_pass = (sum(1 for r in degraded_results if r.passed) / len(degraded_results)
                     if degraded_results else pass_rate)
    gap = normal_pass - degraded_pass

    # Score: weighted combination of overall pass rate and resilience
    overall_score = (normal_pass * 50) + (degraded_pass * 30) + ((1 - gap) * 20)
    overall_score = max(0, min(100, overall_score * 100))

    # Grade
    if overall_score >= 90:
        grade = "A"
    elif overall_score >= 75:
        grade = "B"
    elif overall_score >= 60:
        grade = "C"
    elif overall_score >= 40:
        grade = "D"
    else:
        grade = "F"

    resilience = ResilienceScore(
        overall=overall_score,
        normal_pass_rate=normal_pass,
        degraded_pass_rate=degraded_pass,
        resilience_gap=gap,
        grade=grade,
    )

    return FailureAnalysis(
        total_scenarios=total,
        total_passed=passed,
        total_failed=failed,
        pass_rate=pass_rate,
        clusters=clusters,
        correlations=correlations,
        counterfactuals=counterfactuals,
        resilience=resilience,
    )
