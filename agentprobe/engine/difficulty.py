"""
Scenario difficulty scoring.

Rates each scenario on a 0-100 difficulty scale based on:
- Number of tool failures injected
- Whether it's an edge case
- Number of rules being tested
- Boundary proximity (values near policy thresholds)

This contextualizes results: "95% pass on easy, 60% on hard"
is more useful than just "80% overall pass rate."

Usage:
    scorer = DifficultyScorer()
    for scenario in scenarios:
        score = scorer.score(scenario)
        print(f"{scenario.scenario_id}: difficulty={score.score}, level={score.level}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from agentprobe.engine.scenario import Scenario


@dataclass
class DifficultyScore:
    """Difficulty assessment for a single scenario."""
    score: int = 0              # 0-100
    level: str = "easy"         # easy, medium, hard, adversarial
    factors: Dict[str, int] = None  # breakdown of what contributed

    def __post_init__(self):
        if self.factors is None:
            self.factors = {}
        # Derive level from score
        if self.score >= 75:
            self.level = "adversarial"
        elif self.score >= 50:
            self.level = "hard"
        elif self.score >= 25:
            self.level = "medium"
        else:
            self.level = "easy"

    def to_dict(self) -> Dict:
        return {"score": self.score, "level": self.level, "factors": self.factors}


class DifficultyScorer:
    """
    Scores scenario difficulty on a 0-100 scale.

    Factors:
    - tool_failures: +20 per failed tool (max 60)
    - edge_case: +25 if scenario is an edge case
    - rules_count: +5 per rule being tested (max 20)
    - chaos_variety: +10 if multiple different failure types
    - empty_input: +15 if input is empty/whitespace
    """

    def score(self, scenario: Scenario) -> DifficultyScore:
        factors = {}
        total = 0

        # Tool failures: each failed tool adds difficulty
        num_failures = len(scenario.injected_failures)
        tool_score = min(60, num_failures * 20)
        if tool_score > 0:
            factors["tool_failures"] = tool_score
            total += tool_score

        # Edge case flag
        if scenario.metadata.get("edge_case") or scenario.difficulty == "hard":
            factors["edge_case"] = 25
            total += 25

        # Number of rules being tested
        num_rules = len(scenario.rules_to_check)
        rules_score = min(20, num_rules * 5)
        if rules_score > 0:
            factors["rules_tested"] = rules_score
            total += rules_score

        # Multiple different failure types (e.g., timeout + partial_data)
        if num_failures > 1:
            failure_types = set(f.split(":")[1] for f in scenario.injected_failures if ":" in f)
            if len(failure_types) > 1:
                factors["chaos_variety"] = 10
                total += 10

        # Empty or very short input
        msg = scenario.customer_message.strip()
        if not msg:
            factors["empty_input"] = 15
            total += 15
        elif len(msg) > 500:
            factors["long_input"] = 10
            total += 10

        # Explicit difficulty setting from the scenario
        if scenario.difficulty == "adversarial":
            factors["adversarial_flag"] = 15
            total += 15

        return DifficultyScore(score=min(100, total), factors=factors)

    def score_batch(self, scenarios: List[Scenario]) -> List[DifficultyScore]:
        return [self.score(s) for s in scenarios]

    def summary(self, scenarios: List[Scenario]) -> Dict:
        """Summarize difficulty distribution."""
        scores = self.score_batch(scenarios)
        levels = {"easy": 0, "medium": 0, "hard": 0, "adversarial": 0}
        for s in scores:
            levels[s.level] = levels.get(s.level, 0) + 1

        total = len(scores)
        avg = sum(s.score for s in scores) / total if total else 0

        return {
            "total": total,
            "average_difficulty": round(avg, 1),
            "distribution": levels,
            "pct_easy": round(levels["easy"] / total * 100) if total else 0,
            "pct_medium": round(levels["medium"] / total * 100) if total else 0,
            "pct_hard": round(levels["hard"] / total * 100) if total else 0,
            "pct_adversarial": round(levels["adversarial"] / total * 100) if total else 0,
        }
