"""
Tests for agentprobe.chaos — WorldSimulator and failure modes.
"""

import random
import pytest
from agentprobe.chaos import (
    WorldSimulator, ToolBehavior, ChaosProfile, WorldConfiguration, ToolState,
    Timeout, PartialData, StaleData, Contradiction, RateLimited,
    MalformedResponse, EmptyResponse, IntermittentFailure,
)


class TestFailureModes:
    def test_timeout_returns_error(self):
        fm = Timeout(delay_ms=3000)
        result = fm.apply({"order_id": "123", "status": "delivered"}, random.Random(42))
        assert result["error"] == "timeout"
        assert "3000" in result["message"]

    def test_timeout_partial(self):
        fm = Timeout(delay_ms=3000, returns_partial=True)
        normal = {"a": 1, "b": 2, "c": 3, "d": 4}
        result = fm.apply(normal, random.Random(42))
        assert len(result) < len(normal)
        assert len(result) >= 1

    def test_partial_data_specific_fields(self):
        fm = PartialData(fields_missing=["delivery_date", "tracking"])
        normal = {"order_id": "123", "status": "shipped", "delivery_date": "2025-01-01", "tracking": "TRK001"}
        result = fm.apply(normal, random.Random(42))
        assert "delivery_date" not in result
        assert "tracking" not in result
        assert "order_id" in result

    def test_partial_data_random_fraction(self):
        fm = PartialData(drop_fraction=0.5)
        normal = {"a": 1, "b": 2, "c": 3, "d": 4}
        result = fm.apply(normal, random.Random(42))
        assert len(result) < len(normal)

    def test_stale_data_adds_metadata(self):
        fm = StaleData(staleness_hours=72)
        normal = {"score": 750}
        result = fm.apply(normal, random.Random(42))
        assert result["score"] == 750  # original data preserved
        assert result["_stale"]["hours"] == 72
        assert "score" in result["_stale"]["fields"]

    def test_rate_limited_returns_error(self):
        fm = RateLimited(retry_after_seconds=60)
        result = fm.apply({"data": "value"}, random.Random(42))
        assert result["error"] == "rate_limited"
        assert result["retry_after"] == 60

    def test_rate_limited_cached(self):
        fm = RateLimited(returns_cached=True)
        normal = {"score": 750}
        result = fm.apply(normal, random.Random(42))
        assert result["score"] == 750
        assert result["_metadata"]["cached"] is True

    def test_empty_response_variants(self):
        for variant, expected in [("empty_dict", {}), ("null", None), ("empty_string", ""), ("empty_list", [])]:
            fm = EmptyResponse(response_type=variant)
            result = fm.apply({"data": "value"}, random.Random(42))
            assert result == expected

    def test_malformed_wrong_type(self):
        fm = MalformedResponse(malformation="wrong_type")
        normal = {"count": 42, "name": "test"}
        result = fm.apply(normal, random.Random(42))
        # One numeric field should now be a string
        assert isinstance(result["count"], str) or isinstance(result["name"], int)

    def test_failure_mode_to_dict(self):
        fm = Timeout(probability=0.15, delay_ms=3000, severity="critical")
        d = fm.to_dict()
        assert d["type"] == "timeout"
        assert d["probability"] == 0.15
        assert d["severity"] == "critical"

    def test_contradiction_marks_response(self):
        fm = Contradiction(field="risk_level", contradicts_tool="order-lookup", contradicts_field="tier")
        result = fm.apply({"risk_level": "low"}, random.Random(42))
        assert "_contradiction" in result
        assert result["_contradiction"]["field"] == "risk_level"


class TestToolBehavior:
    def test_basic(self):
        tb = ToolBehavior(
            description="Order lookup",
            failure_modes=[Timeout(), PartialData()],
        )
        assert len(tb.failure_modes) == 2

    def test_to_dict(self):
        tb = ToolBehavior(
            description="test",
            failure_modes=[Timeout(probability=0.1)],
            is_authoritative=True,
        )
        d = tb.to_dict()
        assert d["is_authoritative"] is True
        assert len(d["failure_modes"]) == 1


class TestWorldConfiguration:
    def test_normal_config(self):
        config = WorldConfiguration(
            tool_states={"tool1": ToolState(tool_name="tool1", behavior="normal")},
            chaos_level="normal",
        )
        assert not config.has_failures
        assert config.failure_count == 0
        assert config.get_tool_behavior("tool1") == "normal"
        assert "normal" in config.summary().lower()

    def test_failed_config(self):
        config = WorldConfiguration(
            tool_states={
                "tool1": ToolState(tool_name="tool1", behavior="timeout", failure_mode=Timeout()),
            },
            injected_failures=["tool1:timeout"],
        )
        assert config.has_failures
        assert config.failure_count == 1
        assert "timeout" in config.summary()

    def test_to_dict(self):
        config = WorldConfiguration(
            tool_states={"t": ToolState(tool_name="t", behavior="normal")},
            chaos_level="normal",
            seed=42,
        )
        d = config.to_dict()
        assert d["seed"] == 42
        assert not d["has_failures"]


class TestChaosProfile:
    def test_gentle_rates(self):
        rates = ChaosProfile.GENTLE.get_failure_rates()
        assert all(v <= 0.1 for v in rates.values())

    def test_hostile_rates(self):
        rates = ChaosProfile.HOSTILE.get_failure_rates()
        assert any(v >= 0.1 for v in rates.values())

    def test_adversarial_rates(self):
        rates = ChaosProfile.ADVERSARIAL.get_failure_rates()
        assert any(v >= 0.2 for v in rates.values())


class TestWorldSimulator:
    def _make_simulator(self, **kwargs):
        tools = {
            "order-lookup": ToolBehavior(
                description="Looks up orders",
                failure_modes=[Timeout(probability=0.2), PartialData(probability=0.15)],
            ),
            "product-search": ToolBehavior(
                description="Searches products",
                failure_modes=[RateLimited(probability=0.1), EmptyResponse(probability=0.1)],
            ),
        }
        return WorldSimulator(tools=tools, seed=42, **kwargs)

    def test_generate_single(self):
        sim = self._make_simulator()
        config = sim.generate()
        assert isinstance(config, WorldConfiguration)
        assert "order-lookup" in config.tool_states
        assert "product-search" in config.tool_states

    def test_generate_batch(self):
        sim = self._make_simulator()
        configs = sim.generate_batch(n=100)
        assert len(configs) == 100

        # Should have a mix of normal and failed
        normal_count = sum(1 for c in configs if not c.has_failures)
        failed_count = sum(1 for c in configs if c.has_failures)
        assert normal_count > 0
        assert failed_count > 0

    def test_normal_ratio(self):
        sim = self._make_simulator(normal_ratio=0.8)
        configs = sim.generate_batch(n=200)
        normal_count = sum(1 for c in configs if c.chaos_level == "normal")
        # Should be roughly 80% normal (allow variance)
        assert normal_count > 100

    def test_reproducible_with_seed(self):
        sim1 = self._make_simulator()
        sim2 = self._make_simulator()
        configs1 = sim1.generate_batch(n=50)
        configs2 = sim2.generate_batch(n=50)
        for c1, c2 in zip(configs1, configs2):
            assert c1.chaos_level == c2.chaos_level
            assert c1.injected_failures == c2.injected_failures

    def test_from_profile(self):
        tools = {
            "tool1": ToolBehavior(failure_modes=[Timeout(), PartialData()]),
        }
        sim = WorldSimulator.from_profile(tools=tools, profile=ChaosProfile.HOSTILE)
        assert sim.profile == ChaosProfile.HOSTILE
        # Hostile profile should set higher probabilities
        for fm in sim.tools["tool1"].failure_modes:
            assert fm.probability >= 0.1

    def test_from_tool_names(self):
        sim = WorldSimulator.from_tool_names(
            ["order-lookup", "credit-check", "escalate"],
            profile=ChaosProfile.MODERATE,
        )
        assert len(sim.tools) == 3
        assert "order-lookup" in sim.tools
        # Each tool should have auto-generated failure modes
        assert len(sim.tools["order-lookup"].failure_modes) >= 4

    def test_batch_has_all_chaos_types(self):
        sim = self._make_simulator(normal_ratio=0.3)
        configs = sim.generate_batch(n=200)
        chaos_types = {c.chaos_level for c in configs}
        assert "normal" in chaos_types
        assert "single_failure" in chaos_types
        # multi_failure and cascade may or may not appear in 200 samples

    def test_describe(self):
        sim = self._make_simulator()
        desc = sim.describe()
        assert "order-lookup" in desc
        assert "timeout" in desc.lower()

    def test_to_dict(self):
        sim = self._make_simulator()
        d = sim.to_dict()
        assert "tools" in d
        assert "order-lookup" in d["tools"]
