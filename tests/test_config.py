"""Tests for agentprobe.config — generate and load YAML configs."""

import os
import tempfile
import pytest
from agentprobe.config import (
    generate_config, generate_config_with_llm,
    save_config, load_config,
    _classify_tool, _infer_domain,
)


class TestToolClassification:
    def test_order_lookup(self):
        p = _classify_tool("order-lookup")
        assert p["category"] == "data_retrieval"
        assert "timeout" in p["failures"]

    def test_get_weather(self):
        p = _classify_tool("get-weather")
        assert p["category"] == "external_api"
        assert "schema_drift" in p["failures"]

    def test_process_return(self):
        p = _classify_tool("process-return")
        assert p["category"] == "mutation"

    def test_escalate(self):
        p = _classify_tool("escalate-to-human")
        assert p["category"] == "routing"

    def test_search_facts(self):
        p = _classify_tool("search-facts")
        assert p["category"] == "search"

    def test_send_email(self):
        p = _classify_tool("send-email")
        assert p["category"] == "notification"

    def test_unknown_tool(self):
        p = _classify_tool("foobar-xyz")
        assert p["category"] == "generic"

    def test_generate_response(self):
        p = _classify_tool("generate-response")
        assert p["category"] == "llm_action"

    def test_check_inventory(self):
        p = _classify_tool("check-inventory")
        assert p["category"] == "validation"

    def test_get_country_info(self):
        p = _classify_tool("get-country-info")
        assert p["category"] == "external_api"


class TestDomainInference:
    def test_customer_support(self):
        assert _infer_domain("Customer support agent for NovaMart") == "customer_support"

    def test_travel(self):
        assert _infer_domain("Travel research agent for trip planning") == "travel"

    def test_finance(self):
        assert _infer_domain("Banking transaction processing agent") == "finance"

    def test_space(self):
        assert _infer_domain("Satellite mission operations copilot") == "space_operations"

    def test_general(self):
        assert _infer_domain("A generic automation tool") == "general"


class TestGenerateConfig:
    def test_basic(self):
        config = generate_config(
            agent_description="Customer support agent",
            tools=["order-lookup", "process-return", "escalate"],
        )
        assert "agent" in config
        assert "test_plan" in config
        assert "chaos" in config
        assert config["agent"]["description"] == "Customer support agent"
        assert config["agent"]["tools"] == ["order-lookup", "process-return", "escalate"]

    def test_categories_generated(self):
        config = generate_config(
            agent_description="Support agent",
            tools=["order-lookup", "process-return"],
        )
        cats = config["test_plan"]["categories"]
        assert len(cats) >= 3  # happy path + process-return + error handling + edge cases
        cat_names = [c["name"] for c in cats]
        assert "happy-path" in cat_names
        assert "error-handling" in cat_names

    def test_rules_generated(self):
        config = generate_config(
            agent_description="Support agent",
            tools=["order-lookup", "process-return", "escalate"],
        )
        rules = config["test_plan"]["rules"]
        rule_names = [r["name"] for r in rules]
        assert "graceful_error_handling" in rule_names
        assert "no_hallucinated_data" in rule_names
        assert "verify_before_mutation" in rule_names  # because process-return is a mutation tool
        assert "appropriate_escalation" in rule_names  # because escalate is a routing tool

    def test_chaos_generated(self):
        config = generate_config(
            agent_description="Agent",
            tools=["get-weather", "search-facts"],
        )
        chaos = config["chaos"]
        assert chaos["level"] == "moderate"
        assert "get-weather" in chaos["tools"]
        assert "search-facts" in chaos["tools"]
        assert "timeout" in chaos["tools"]["get-weather"]["failure_modes"]

    def test_with_policy_docs(self):
        config = generate_config(
            agent_description="Agent",
            tools=["order-lookup"],
            policy_docs=["Returns within 30 days only"],
        )
        assert config["agent"]["policy_docs"] == ["Returns within 30 days only"]

    def test_chaos_level(self):
        config = generate_config("Agent", ["tool-a"], chaos_level="hostile")
        assert config["chaos"]["level"] == "hostile"

    def test_dimensions_generated(self):
        config = generate_config(
            agent_description="Agent",
            tools=["order-lookup", "get-weather"],
        )
        dims = config["test_plan"]["dimensions"]
        assert len(dims) > 0
        dim_names = [d["name"] for d in dims]
        # order-lookup has likely_param "order_id", get-weather has "location"
        assert any("id" in n or "location" in n for n in dim_names)

    def test_edge_cases_generated(self):
        config = generate_config("Agent", ["order-lookup"])
        edges = config["test_plan"]["edge_cases"]
        edge_names = [e["name"] for e in edges]
        assert "empty_input" in edge_names
        assert "very_long_input" in edge_names


class TestSaveAndLoad:
    def test_roundtrip(self):
        config = generate_config(
            agent_description="Travel research agent with weather and country tools",
            tools=["get-weather", "get-country-info", "search-facts"],
        )

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            path = f.name

        try:
            save_config(config, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 100

            plan, world = load_config(path)

            # Verify plan
            assert plan.name == config["test_plan"]["name"]
            assert len(plan.categories) == len(config["test_plan"]["categories"])
            assert len(plan.rules) == len(config["test_plan"]["rules"])
            assert len(plan.dimensions) > 0

            # Verify world
            assert "get-weather" in world.tools
            assert "get-country-info" in world.tools
            assert len(world.tools["get-weather"].failure_modes) > 0

        finally:
            os.unlink(path)

    def test_load_preserves_failure_modes(self):
        config = generate_config("Agent", ["order-lookup"])
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            save_config(config, path)
            _, world = load_config(path)

            fm_types = [fm.failure_type() for fm in world.tools["order-lookup"].failure_modes]
            assert "timeout" in fm_types
        finally:
            os.unlink(path)

    def test_load_preserves_rules(self):
        config = generate_config("Agent", ["order-lookup", "process-return"])
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            save_config(config, path)
            plan, _ = load_config(path)

            rule_names = [r.name for r in plan.rules]
            assert "graceful_error_handling" in rule_names
            assert "verify_before_mutation" in rule_names
        finally:
            os.unlink(path)

    def test_yaml_is_readable(self):
        """The YAML output should be human-readable."""
        config = generate_config(
            agent_description="NovaMart customer support agent handling returns and inquiries",
            tools=["order-lookup", "process-return", "escalate-to-human"],
        )
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            path = f.name
        try:
            save_config(config, path)
            with open(path) as f:
                content = f.read()

            # Should be readable YAML, not a JSON dump
            assert "agent:" in content
            assert "test_plan:" in content
            assert "chaos:" in content
            assert "failure_modes:" in content

        finally:
            os.unlink(path)


class TestGenerateConfigWithLLM:
    def test_falls_back_without_llm(self):
        """Without an LLM, should produce the same as generate_config."""
        config = generate_config_with_llm(
            agent_description="Agent",
            tools=["order-lookup"],
        )
        assert "test_plan" in config
        assert len(config["test_plan"]["rules"]) > 0

    def test_falls_back_without_policy_docs(self):
        """Without policy docs, LLM has nothing to analyze — should fall back."""
        config = generate_config_with_llm(
            agent_description="Agent",
            tools=["order-lookup"],
            llm="fake_llm",  # not a real LLM, will fail
        )
        assert "test_plan" in config
