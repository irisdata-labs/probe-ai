"""
agentprobe.chaos — Chaos engineering for AI agents.

The WorldSimulator creates a controllable, breakable environment for agent
testing. Instead of just varying inputs, it varies what the agent's tools
return and how they behave — timeouts, stale data, contradictions, partial
responses, rate limits, and silent failures.

This is the "break your agent's world and see if it survives" layer.

Architecture:
    WorldSimulator
    ├── ToolBehavior (per-tool config: normal + failure modes)
    ├── FailureMode (abstract: what can go wrong)
    │   ├── Timeout
    │   ├── PartialData
    │   ├── StaleData
    │   ├── Contradiction
    │   ├── RateLimited
    │   ├── MalformedResponse
    │   ├── EmptyResponse
    │   └── IntermittentFailure
    ├── WorldState (tracks entity state across tools for consistency)
    └── ChaosProfile (predefined failure mixes: gentle, moderate, hostile)

Usage:
    from agentprobe.chaos import WorldSimulator, ToolBehavior, ChaosProfile
    from agentprobe.chaos import Timeout, PartialData, StaleData, Contradiction

    # Define tool behaviors with failure modes
    world = WorldSimulator(
        tools={
            "order-lookup": ToolBehavior(
                description="Looks up order by ID",
                response_schema={"order_id": "str", "status": "str", "items": "list"},
                failure_modes=[
                    Timeout(probability=0.1, delay_ms=5000),
                    PartialData(probability=0.05, fields_missing=["delivery_date"]),
                    StaleData(probability=0.08, staleness_hours=48),
                ],
            ),
            "credit-check": ToolBehavior(
                description="Checks customer credit score",
                response_schema={"score": "int", "risk_level": "str"},
                failure_modes=[
                    RateLimited(probability=0.03, retry_after_seconds=30),
                    Contradiction(
                        probability=0.02,
                        field="risk_level",
                        contradicts_tool="order-lookup",
                        contradicts_field="customer_tier",
                    ),
                ],
            ),
        },
        seed=42,  # reproducible chaos
    )

    # Or use a preset chaos profile
    world = WorldSimulator.from_profile(
        tools={...},
        profile=ChaosProfile.MODERATE,  # applies reasonable failure rates
    )

    # Generate a world configuration for one scenario
    scenario_world = world.generate()
    # → {
    #     "order-lookup": {"behavior": "partial_data", "missing_fields": ["delivery_date"]},
    #     "credit-check": {"behavior": "normal"},
    #     "injected_failures": ["order-lookup:partial_data"],
    #     "chaos_level": "moderate",
    # }

    # Generate N world configurations for batch testing
    worlds = world.generate_batch(n=200)
    # Mix of normal + increasingly chaotic configurations
"""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# Failure Modes — what can go wrong with a tool
# ============================================================================

@dataclass
class FailureMode(ABC):
    """Base class for tool failure modes."""
    probability: float = 0.1        # chance this failure occurs (0.0 to 1.0)
    severity: str = "major"         # critical, major, minor — for reporting
    description: str = ""           # human-readable description

    @abstractmethod
    def apply(self, normal_response: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        """
        Apply this failure to a normal tool response.

        Args:
            normal_response: What the tool would normally return.
            rng: Seeded random instance for reproducibility.

        Returns:
            Modified response reflecting this failure mode.
        """
        ...

    @abstractmethod
    def failure_type(self) -> str:
        """Short identifier for this failure type."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.failure_type(),
            "probability": self.probability,
            "severity": self.severity,
            "description": self.description or self._default_description(),
        }

    def _default_description(self) -> str:
        return f"{self.failure_type()} failure"


@dataclass
class Timeout(FailureMode):
    """Tool takes too long to respond or doesn't respond at all."""
    delay_ms: int = 5000            # how long before timeout
    returns_partial: bool = False   # does it return partial data before dying?

    def failure_type(self) -> str:
        return "timeout"

    def apply(self, normal_response: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        if self.returns_partial:
            # Return a subset of fields
            keys = list(normal_response.keys())
            n_keep = max(1, len(keys) // 2)
            kept = rng.sample(keys, n_keep)
            return {k: normal_response[k] for k in kept}
        return {"error": "timeout", "message": f"Tool timed out after {self.delay_ms}ms"}

    def _default_description(self) -> str:
        return f"Tool times out after {self.delay_ms}ms"


@dataclass
class PartialData(FailureMode):
    """Tool returns a response but with missing fields."""
    fields_missing: Optional[List[str]] = None   # specific fields to drop
    drop_fraction: float = 0.3                    # if no specific fields, drop this fraction

    def failure_type(self) -> str:
        return "partial_data"

    def apply(self, normal_response: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        result = dict(normal_response)
        if self.fields_missing:
            for f in self.fields_missing:
                result.pop(f, None)
        else:
            keys = list(result.keys())
            n_drop = max(1, int(len(keys) * self.drop_fraction))
            to_drop = rng.sample(keys, min(n_drop, len(keys)))
            for k in to_drop:
                del result[k]
        return result

    def _default_description(self) -> str:
        if self.fields_missing:
            return f"Response missing fields: {', '.join(self.fields_missing)}"
        return f"Response missing ~{int(self.drop_fraction * 100)}% of fields"


@dataclass
class StaleData(FailureMode):
    """Tool returns outdated information."""
    staleness_hours: int = 48       # how old the data is
    stale_fields: Optional[List[str]] = None  # which fields are stale (None = all)

    def failure_type(self) -> str:
        return "stale_data"

    def apply(self, normal_response: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        result = dict(normal_response)
        result["_metadata"] = {
            "data_staleness_hours": self.staleness_hours,
            "stale_fields": self.stale_fields or list(result.keys()),
            "warning": f"Data may be up to {self.staleness_hours} hours old",
        }
        return result

    def _default_description(self) -> str:
        return f"Data is {self.staleness_hours} hours stale"


@dataclass
class Contradiction(FailureMode):
    """Tool returns data that contradicts another tool's response."""
    field: str = ""                          # which field in this tool's response
    contradicts_tool: str = ""               # which other tool
    contradicts_field: str = ""              # which field in the other tool
    contradiction_type: str = "value"        # value (different values) or logic (impossible combo)

    def failure_type(self) -> str:
        return "contradiction"

    def apply(self, normal_response: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        result = dict(normal_response)
        # The actual contradiction is injected at the WorldSimulator level
        # since it requires coordination between tools. Here we just mark it.
        result["_contradiction"] = {
            "field": self.field,
            "contradicts": f"{self.contradicts_tool}.{self.contradicts_field}",
            "type": self.contradiction_type,
        }
        return result

    def _default_description(self) -> str:
        return (f"Field '{self.field}' contradicts "
                f"{self.contradicts_tool}.{self.contradicts_field}")


@dataclass
class RateLimited(FailureMode):
    """Tool rejects the request due to rate limiting."""
    retry_after_seconds: int = 30
    returns_cached: bool = False    # does it return a cached (possibly stale) response?

    def failure_type(self) -> str:
        return "rate_limited"

    def apply(self, normal_response: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        if self.returns_cached:
            result = dict(normal_response)
            result["_metadata"] = {"cached": True, "cache_age_seconds": rng.randint(60, 3600)}
            return result
        return {
            "error": "rate_limited",
            "message": f"Rate limit exceeded. Retry after {self.retry_after_seconds}s",
            "retry_after": self.retry_after_seconds,
        }

    def _default_description(self) -> str:
        return f"Rate limited, retry after {self.retry_after_seconds}s"


@dataclass
class MalformedResponse(FailureMode):
    """Tool returns a response with wrong types or unexpected structure."""
    malformation: str = "wrong_type"  # wrong_type, extra_fields, nested_error, truncated

    def failure_type(self) -> str:
        return "malformed_response"

    def apply(self, normal_response: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        result = dict(normal_response)
        if self.malformation == "wrong_type":
            # Convert a numeric field to string or vice versa
            for k, v in result.items():
                if isinstance(v, (int, float)):
                    result[k] = str(v)
                    break
                elif isinstance(v, str) and v.isdigit():
                    result[k] = int(v)
                    break
        elif self.malformation == "extra_fields":
            result["_unexpected_field"] = "unexpected_value"
            result["_internal_debug"] = {"trace_id": "abc123", "server": "prod-3"}
        elif self.malformation == "nested_error":
            result["error"] = {"code": 500, "message": "Internal error", "data": result}
        elif self.malformation == "truncated":
            keys = list(result.keys())
            if keys:
                last_key = keys[-1]
                if isinstance(result[last_key], str):
                    result[last_key] = result[last_key][:len(result[last_key]) // 2]
        return result

    def _default_description(self) -> str:
        return f"Response has {self.malformation} malformation"


@dataclass
class EmptyResponse(FailureMode):
    """Tool returns an empty or null response."""
    response_type: str = "empty_dict"  # empty_dict, null, empty_string, empty_list

    def failure_type(self) -> str:
        return "empty_response"

    def apply(self, normal_response: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        variants = {
            "empty_dict": {},
            "null": None,
            "empty_string": "",
            "empty_list": [],
        }
        return variants.get(self.response_type, {})

    def _default_description(self) -> str:
        return f"Tool returns {self.response_type}"


@dataclass
class IntermittentFailure(FailureMode):
    """Tool works on first call but fails on retry, or vice versa."""
    fails_on: str = "first"        # first, second, random
    underlying_failure: str = "timeout"  # what kind of failure when it fails

    def failure_type(self) -> str:
        return "intermittent"

    def apply(self, normal_response: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        # Intermittent behavior is handled at the WorldSimulator level
        # which tracks call count per tool. When it does fail:
        return {"error": self.underlying_failure,
                "message": f"Intermittent {self.underlying_failure}"}

    def _default_description(self) -> str:
        return f"Fails on {self.fails_on} call with {self.underlying_failure}"


@dataclass
class SchemaDrift(FailureMode):
    """
    Tool's response schema changes without warning.

    Unlike MalformedResponse (tool is broken), SchemaDrift simulates the tool
    being *updated* — the response is valid according to the new schema but
    breaks the agent's expectations. Common in production when upstream APIs
    ship breaking changes.

    Drift types:
    - field_renamed: A field gets a new name (e.g., "delivery_date" → "delivered_at")
    - field_removed: A field is removed from the response
    - field_added: A new required field appears
    - type_changed: A field's type changes (e.g., string → int, or string → list)
    - nested_restructured: A flat field becomes nested (e.g., "address" → {"street": ..., "city": ...})
    """
    drift_type: str = "field_renamed"  # field_renamed, field_removed, field_added, type_changed, nested_restructured
    target_field: Optional[str] = None    # which field is affected (None = random)
    new_field_name: Optional[str] = None  # for field_renamed

    def failure_type(self) -> str:
        return "schema_drift"

    def apply(self, normal_response: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        result = dict(normal_response)
        fields = [k for k in result.keys() if not k.startswith("_")]
        if not fields:
            return result

        target = self.target_field if self.target_field and self.target_field in result else rng.choice(fields)

        if self.drift_type == "field_renamed":
            new_name = self.new_field_name or f"{target}_v2"
            result[new_name] = result.pop(target)

        elif self.drift_type == "field_removed":
            result.pop(target, None)

        elif self.drift_type == "field_added":
            result["_schema_version"] = "2.0"
            result["_deprecated_warning"] = f"Field '{target}' will be removed in v3"

        elif self.drift_type == "type_changed":
            val = result.get(target)
            if isinstance(val, str):
                result[target] = [val]  # string → list
            elif isinstance(val, (int, float)):
                result[target] = str(val)  # number → string
            elif isinstance(val, list):
                result[target] = val[0] if val else None  # list → single value
            elif isinstance(val, bool):
                result[target] = 1 if val else 0  # bool → int

        elif self.drift_type == "nested_restructured":
            val = result.pop(target)
            result[target] = {"value": val, "metadata": {"source": "v2", "migrated": True}}

        return result

    def _default_description(self) -> str:
        field = self.target_field or "(random)"
        return f"Schema drift: {self.drift_type} on field {field}"

@dataclass
class ToolBehavior:
    """
    Configuration for a single tool in the world simulator.

    Defines what the tool does, what a normal response looks like,
    and what failure modes can be injected.
    """
    description: str = ""
    response_schema: Optional[Dict[str, str]] = None   # field → type mapping
    failure_modes: List[FailureMode] = field(default_factory=list)
    # Whether this tool's data can be used as ground truth for contradiction detection
    is_authoritative: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "response_schema": self.response_schema,
            "failure_modes": [fm.to_dict() for fm in self.failure_modes],
            "is_authoritative": self.is_authoritative,
        }


# ============================================================================
# ChaosProfile — predefined failure mixes
# ============================================================================

class ChaosProfile(Enum):
    """Predefined chaos intensity levels."""
    GENTLE = "gentle"           # ~5% failure rate, only minor issues
    MODERATE = "moderate"       # ~15% failure rate, mix of minor and major
    HOSTILE = "hostile"         # ~30% failure rate, major and critical failures
    ADVERSARIAL = "adversarial" # ~50% failure rate, everything breaks

    def get_failure_rates(self) -> Dict[str, float]:
        """Returns base probability multipliers for each failure type."""
        rates = {
            ChaosProfile.GENTLE: {
                "timeout": 0.03, "partial_data": 0.05, "stale_data": 0.05,
                "rate_limited": 0.02, "malformed_response": 0.01,
                "empty_response": 0.01, "contradiction": 0.01, "intermittent": 0.02,
            },
            ChaosProfile.MODERATE: {
                "timeout": 0.08, "partial_data": 0.10, "stale_data": 0.10,
                "rate_limited": 0.05, "malformed_response": 0.03,
                "empty_response": 0.03, "contradiction": 0.03, "intermittent": 0.05,
            },
            ChaosProfile.HOSTILE: {
                "timeout": 0.15, "partial_data": 0.15, "stale_data": 0.15,
                "rate_limited": 0.10, "malformed_response": 0.08,
                "empty_response": 0.05, "contradiction": 0.08, "intermittent": 0.10,
            },
            ChaosProfile.ADVERSARIAL: {
                "timeout": 0.25, "partial_data": 0.20, "stale_data": 0.20,
                "rate_limited": 0.15, "malformed_response": 0.15,
                "empty_response": 0.10, "contradiction": 0.15, "intermittent": 0.15,
            },
        }
        return rates[self]


# ============================================================================
# WorldConfiguration — one scenario's world state
# ============================================================================

@dataclass
class ToolState:
    """The state of a single tool for one scenario."""
    tool_name: str
    behavior: str                    # "normal" or failure type name
    failure_mode: Optional[FailureMode] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "tool_name": self.tool_name,
            "behavior": self.behavior,
        }
        if self.failure_mode:
            d["failure_details"] = self.failure_mode.to_dict()
        if self.details:
            d["details"] = self.details
        return d


@dataclass
class WorldConfiguration:
    """
    A complete world configuration for one test scenario.

    This describes how every tool behaves in this particular scenario.
    The agent harness uses this to mock tool responses during the test run.
    """
    tool_states: Dict[str, ToolState] = field(default_factory=dict)
    chaos_level: str = "normal"
    injected_failures: List[str] = field(default_factory=list)
    seed: Optional[int] = None

    @property
    def has_failures(self) -> bool:
        return len(self.injected_failures) > 0

    @property
    def failure_count(self) -> int:
        return len(self.injected_failures)

    def get_tool_behavior(self, tool_name: str) -> str:
        """Get the behavior for a specific tool."""
        if tool_name in self.tool_states:
            return self.tool_states[tool_name].behavior
        return "normal"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_states": {k: v.to_dict() for k, v in self.tool_states.items()},
            "chaos_level": self.chaos_level,
            "injected_failures": self.injected_failures,
            "seed": self.seed,
            "has_failures": self.has_failures,
            "failure_count": self.failure_count,
        }

    def summary(self) -> str:
        if not self.has_failures:
            return "All tools normal"
        parts = []
        for f in self.injected_failures:
            parts.append(f)
        return f"{len(parts)} failure(s): {', '.join(parts)}"


# ============================================================================
# WorldSimulator — the main chaos engine
# ============================================================================

class WorldSimulator:
    """
    Creates controllable, breakable environments for agent testing.

    The simulator generates WorldConfigurations that describe how each
    tool behaves in a given scenario. Some scenarios have all tools
    working normally. Others have specific failures injected. The mix
    is controlled by the chaos profile.
    """

    def __init__(
        self,
        tools: Dict[str, ToolBehavior],
        seed: Optional[int] = None,
        profile: Optional[ChaosProfile] = None,
        normal_ratio: float = 0.5,   # fraction of scenarios with zero failures
    ):
        """
        Args:
            tools: Mapping of tool names to their behaviors and failure modes.
            seed: Random seed for reproducible chaos.
            profile: Optional ChaosProfile to auto-configure failure rates.
            normal_ratio: Fraction of generated scenarios with no failures.
                          Ensures baseline comparison data. Default 0.5 (50%).
        """
        self.tools = tools
        self.seed = seed
        self.profile = profile
        self.normal_ratio = normal_ratio
        self._rng = random.Random(seed)

        # If a profile is set, apply its rates to any failure mode
        # that doesn't have an explicit probability
        if profile:
            self._apply_profile(profile)

    def _apply_profile(self, profile: ChaosProfile):
        """Apply chaos profile rates to failure modes."""
        rates = profile.get_failure_rates()
        for tool_name, tool_behavior in self.tools.items():
            for fm in tool_behavior.failure_modes:
                ft = fm.failure_type()
                if ft in rates:
                    fm.probability = rates[ft]

    @classmethod
    def from_profile(
        cls,
        tools: Dict[str, ToolBehavior],
        profile: ChaosProfile,
        seed: Optional[int] = None,
    ) -> WorldSimulator:
        """Create a simulator with a predefined chaos profile."""
        return cls(tools=tools, profile=profile, seed=seed)

    @classmethod
    def from_tool_names(
        cls,
        tool_names: List[str],
        profile: ChaosProfile = ChaosProfile.MODERATE,
        seed: Optional[int] = None,
    ) -> WorldSimulator:
        """
        Create a simulator from just tool names — auto-generates failure modes.

        Convenience method for users who don't want to configure each failure
        mode manually. Applies a standard set of failure modes to every tool.
        """
        tools = {}
        for name in tool_names:
            tools[name] = ToolBehavior(
                description=f"Tool: {name}",
                failure_modes=[
                    Timeout(),
                    PartialData(),
                    StaleData(),
                    RateLimited(),
                    MalformedResponse(),
                    EmptyResponse(),
                ],
            )
        return cls(tools=tools, profile=profile, seed=seed)

    def generate(self) -> WorldConfiguration:
        """Generate a single world configuration."""
        # Decide if this is a normal or chaotic scenario
        if self._rng.random() < self.normal_ratio:
            return self._generate_normal()
        return self._generate_chaotic()

    def generate_batch(
        self,
        n: int = 200,
        chaos_distribution: Optional[Dict[str, float]] = None,
    ) -> List[WorldConfiguration]:
        """
        Generate N world configurations with controlled chaos distribution.

        Args:
            n: Number of configurations to generate.
            chaos_distribution: Optional override for the mix.
                Default: {normal_ratio}% normal, rest distributed across chaos levels.
                Example: {"normal": 0.5, "single_failure": 0.25, "multi_failure": 0.15, "cascade": 0.10}

        Returns:
            List of WorldConfiguration objects.
        """
        if chaos_distribution is None:
            chaos_distribution = {
                "normal": self.normal_ratio,
                "single_failure": (1 - self.normal_ratio) * 0.5,
                "multi_failure": (1 - self.normal_ratio) * 0.3,
                "cascade": (1 - self.normal_ratio) * 0.2,
            }

        configs = []
        for i in range(n):
            scenario_seed = (self.seed or 0) + i
            rng = random.Random(scenario_seed)

            # Pick chaos type
            roll = rng.random()
            cumulative = 0.0
            chaos_type = "normal"
            for ct, prob in chaos_distribution.items():
                cumulative += prob
                if roll <= cumulative:
                    chaos_type = ct
                    break

            if chaos_type == "normal":
                config = self._generate_normal(rng=rng)
            elif chaos_type == "single_failure":
                config = self._generate_single_failure(rng=rng)
            elif chaos_type == "multi_failure":
                config = self._generate_multi_failure(rng=rng)
            elif chaos_type == "cascade":
                config = self._generate_cascade(rng=rng)
            else:
                config = self._generate_chaotic(rng=rng)

            config.seed = scenario_seed
            config.chaos_level = chaos_type
            configs.append(config)

        return configs

    # ----------------------------------------------------------------
    # Generation strategies
    # ----------------------------------------------------------------

    def _generate_normal(self, rng: Optional[random.Random] = None) -> WorldConfiguration:
        """All tools work normally."""
        tool_states = {}
        for name in self.tools:
            tool_states[name] = ToolState(tool_name=name, behavior="normal")
        return WorldConfiguration(
            tool_states=tool_states,
            chaos_level="normal",
            injected_failures=[],
        )

    def _generate_chaotic(self, rng: Optional[random.Random] = None) -> WorldConfiguration:
        """Randomly apply failures based on each failure mode's probability."""
        rng = rng or self._rng
        tool_states = {}
        injected = []

        for name, behavior in self.tools.items():
            # Check each failure mode
            triggered = None
            for fm in behavior.failure_modes:
                if rng.random() < fm.probability:
                    triggered = fm
                    break  # one failure per tool per scenario

            if triggered:
                tool_states[name] = ToolState(
                    tool_name=name,
                    behavior=triggered.failure_type(),
                    failure_mode=triggered,
                )
                injected.append(f"{name}:{triggered.failure_type()}")
            else:
                tool_states[name] = ToolState(tool_name=name, behavior="normal")

        return WorldConfiguration(
            tool_states=tool_states,
            chaos_level="chaotic",
            injected_failures=injected,
        )

    def _generate_single_failure(self, rng: Optional[random.Random] = None) -> WorldConfiguration:
        """Exactly one tool fails, the rest work normally."""
        rng = rng or self._rng
        tool_states = {}
        injected = []

        # Pick one tool to break
        tool_names = list(self.tools.keys())
        target_tool = rng.choice(tool_names)

        for name, behavior in self.tools.items():
            if name == target_tool and behavior.failure_modes:
                fm = rng.choice(behavior.failure_modes)
                tool_states[name] = ToolState(
                    tool_name=name,
                    behavior=fm.failure_type(),
                    failure_mode=fm,
                )
                injected.append(f"{name}:{fm.failure_type()}")
            else:
                tool_states[name] = ToolState(tool_name=name, behavior="normal")

        return WorldConfiguration(
            tool_states=tool_states,
            chaos_level="single_failure",
            injected_failures=injected,
        )

    def _generate_multi_failure(self, rng: Optional[random.Random] = None) -> WorldConfiguration:
        """Multiple tools fail simultaneously."""
        rng = rng or self._rng
        tool_states = {}
        injected = []

        tool_names = list(self.tools.keys())
        # Fail 2 to N-1 tools (at least 2, leave at least 1 working)
        n_fail = rng.randint(2, max(2, len(tool_names) - 1))
        failing_tools = set(rng.sample(tool_names, min(n_fail, len(tool_names))))

        for name, behavior in self.tools.items():
            if name in failing_tools and behavior.failure_modes:
                fm = rng.choice(behavior.failure_modes)
                tool_states[name] = ToolState(
                    tool_name=name,
                    behavior=fm.failure_type(),
                    failure_mode=fm,
                )
                injected.append(f"{name}:{fm.failure_type()}")
            else:
                tool_states[name] = ToolState(tool_name=name, behavior="normal")

        return WorldConfiguration(
            tool_states=tool_states,
            chaos_level="multi_failure",
            injected_failures=injected,
        )

    def _generate_cascade(self, rng: Optional[random.Random] = None) -> WorldConfiguration:
        """
        Cascade failure: one tool fails, causing dependent tools to also fail.

        This simulates real-world cascade scenarios like:
        - Database timeout → lookup fails → agent has no data → makes bad decision
        - Auth service down → all authenticated tools fail
        """
        rng = rng or self._rng
        tool_states = {}
        injected = []

        tool_names = list(self.tools.keys())
        # Pick the root cause tool
        root_tool = rng.choice(tool_names)
        root_behavior = self.tools[root_tool]

        # Root tool has a hard failure
        if root_behavior.failure_modes:
            root_fm = rng.choice([fm for fm in root_behavior.failure_modes
                                  if fm.failure_type() in ("timeout", "empty_response", "rate_limited")]
                                 or root_behavior.failure_modes)
        else:
            root_fm = Timeout(probability=1.0)

        tool_states[root_tool] = ToolState(
            tool_name=root_tool,
            behavior=root_fm.failure_type(),
            failure_mode=root_fm,
            details={"cascade_role": "root_cause"},
        )
        injected.append(f"{root_tool}:{root_fm.failure_type()}[root]")

        # Cascade: other tools get secondary failures
        for name in tool_names:
            if name == root_tool:
                continue
            if rng.random() < 0.6:  # 60% chance of cascade to each other tool
                cascade_fm = PartialData(
                    probability=1.0,
                    fields_missing=None,
                    drop_fraction=0.5,
                )
                tool_states[name] = ToolState(
                    tool_name=name,
                    behavior="partial_data",
                    failure_mode=cascade_fm,
                    details={"cascade_role": "affected", "caused_by": root_tool},
                )
                injected.append(f"{name}:partial_data[cascade from {root_tool}]")
            else:
                tool_states[name] = ToolState(tool_name=name, behavior="normal")

        return WorldConfiguration(
            tool_states=tool_states,
            chaos_level="cascade",
            injected_failures=injected,
        )

    # ----------------------------------------------------------------
    # Reporting
    # ----------------------------------------------------------------

    def describe(self) -> str:
        """Human-readable description of the simulator configuration."""
        lines = [
            "WorldSimulator Configuration",
            "=" * 40,
            f"Tools: {len(self.tools)}",
            f"Profile: {self.profile.value if self.profile else 'custom'}",
            f"Normal ratio: {self.normal_ratio:.0%}",
            f"Seed: {self.seed}",
            "",
        ]
        for name, behavior in self.tools.items():
            lines.append(f"  {name}:")
            if behavior.description:
                lines.append(f"    {behavior.description}")
            lines.append(f"    Failure modes: {len(behavior.failure_modes)}")
            for fm in behavior.failure_modes:
                lines.append(f"      • {fm.failure_type()} (p={fm.probability:.0%}): "
                             f"{fm._default_description()}")
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tools": {k: v.to_dict() for k, v in self.tools.items()},
            "profile": self.profile.value if self.profile else None,
            "normal_ratio": self.normal_ratio,
            "seed": self.seed,
        }
