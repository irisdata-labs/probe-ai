"""
Mission Operations Copilot — Space Domain Example

Demonstrates agentprobe tracing a mission operations copilot
that diagnoses satellite anomalies.

Run:
    python examples/mission_ops_copilot.py
"""

import random
import time

from agentprobe import (
    AgentProbe,
    TraceSession,
    record_decision,
    record_state_change,
    step_span,
    trace_agent,
    trace_llm_call,
    trace_tool,
)
from agentprobe.core import get_current_span
from agentprobe.exporters import ConsoleExporter, JSONFileExporter

probe = AgentProbe(exporters=[
    ConsoleExporter(verbose=True),
    JSONFileExporter("traces/"),
])
probe.init()


# ============================================================================
# Simulated Space Tools
# ============================================================================

@trace_tool("fetch-telemetry")
def fetch_telemetry(satellite_id: str) -> dict:
    time.sleep(0.02)
    voltage = round(random.uniform(24.0, 29.5), 2)
    return {
        "satellite_id": satellite_id,
        "battery_voltage": voltage,
        "temperature_c": round(random.uniform(-15, 45), 1),
        "solar_panel_output_w": round(random.uniform(100, 300), 1),
        "orbit_altitude_km": round(random.uniform(400, 600), 1),
        "in_eclipse": random.choice([True, False]),
        "data_age_seconds": random.randint(5, 120),
    }


@trace_tool("lookup-procedure")
def lookup_procedure(anomaly_type: str) -> dict:
    time.sleep(0.015)
    procedures = {
        "battery_low": {
            "procedure_id": "PROC-BAT-001",
            "title": "Battery Voltage Anomaly Response",
            "steps": ["Verify telemetry accuracy", "Check eclipse schedule",
                      "Reduce non-essential loads", "Monitor for 2 orbits"],
        },
        "thermal_warning": {
            "procedure_id": "PROC-THM-002",
            "title": "Thermal Limit Approach",
            "steps": ["Identify heat source", "Adjust attitude if safe",
                      "Reduce duty cycle"],
        },
        "unknown": {
            "procedure_id": "PROC-GEN-999",
            "title": "General Anomaly Investigation",
            "steps": ["Collect full telemetry dump", "Compare with constellation",
                      "Escalate to anomaly review board"],
        },
    }
    return procedures.get(anomaly_type, procedures["unknown"])


@trace_tool("check-constellation")
def check_constellation(satellite_id: str) -> dict:
    time.sleep(0.01)
    return {
        "constellation_size": 6,
        "satellites_reporting": 5,
        "similar_anomalies": random.randint(0, 2),
        "fleet_health": "nominal" if random.random() > 0.3 else "degraded",
    }


# ============================================================================
# LLM Diagnosis
# ============================================================================

@trace_llm_call("generate-diagnosis")
def generate_diagnosis(context: dict) -> dict:
    time.sleep(0.05)

    span = get_current_span()
    if span:
        span.set_llm_metadata(
            model="claude-sonnet-4-20250514",
            prompt_tokens=random.randint(400, 800),
            completion_tokens=random.randint(100, 300),
            total_tokens=random.randint(500, 1100),
        )

    telemetry = context["telemetry"]
    anomaly = context["anomaly_type"]

    if anomaly == "battery_low":
        diagnosis = (f"Battery voltage at {telemetry['battery_voltage']}V is below "
                     f"nominal range. {'Eclipse conditions may be contributing. ' if telemetry['in_eclipse'] else ''}"
                     f"Recommend reducing non-essential loads and monitoring for 2 orbits.")
        severity = "warning"
    elif anomaly == "thermal_warning":
        diagnosis = (f"Temperature at {telemetry['temperature_c']}°C approaching limits. "
                     f"Recommend attitude adjustment to reduce solar exposure.")
        severity = "caution"
    else:
        diagnosis = "Anomaly type unknown. Recommend full telemetry dump and escalation."
        severity = "advisory"

    return {
        "diagnosis": diagnosis,
        "severity": severity,
        "recommended_action": context["procedure"]["steps"][0] if context.get("procedure") else "investigate",
    }


# ============================================================================
# Agent Pipeline
# ============================================================================

@trace_agent("mission-ops-copilot", tags=["space", "demo"])
def run_copilot(query: dict) -> dict:
    satellite_id = query["satellite_id"]

    # Fetch telemetry
    telemetry = fetch_telemetry(satellite_id)

    # Classify the anomaly
    with step_span("classify-anomaly", step_type="reasoning",
                    input_data=telemetry) as span:
        voltage = telemetry["battery_voltage"]
        temp = telemetry["temperature_c"]

        if voltage < 26.0:
            anomaly_type = "battery_low"
        elif temp > 40.0:
            anomaly_type = "thermal_warning"
        else:
            anomaly_type = "nominal"

        span.set_output({"anomaly_type": anomaly_type, "voltage": voltage, "temp": temp})

    # If nominal, respond quickly
    if anomaly_type == "nominal":
        record_decision("no_action", alternatives=["investigate", "escalate"],
                        reason="All parameters within nominal range")
        return {
            "satellite_id": satellite_id,
            "status": "nominal",
            "message": f"All systems nominal for {satellite_id}.",
        }

    # Look up procedure and check constellation
    procedure = lookup_procedure(anomaly_type)
    constellation = check_constellation(satellite_id)

    # Decide: handle or escalate?
    with step_span("escalation-decision", step_type="reasoning") as span:
        needs_escalation = (
            constellation["similar_anomalies"] >= 2
            or constellation["fleet_health"] == "degraded"
        )

        if needs_escalation:
            record_decision("escalate", alternatives=["handle_autonomously"],
                            reason=f"Fleet-wide concern: {constellation['similar_anomalies']} similar anomalies")
            record_state_change("alert_level", before="normal", after="elevated")
            span.set_output({"escalate": True})
        else:
            record_decision("handle_autonomously", alternatives=["escalate"],
                            reason="Isolated anomaly, standard procedure applies")
            span.set_output({"escalate": False})

    # Generate diagnosis
    diagnosis = generate_diagnosis({
        "telemetry": telemetry,
        "anomaly_type": anomaly_type,
        "procedure": procedure,
        "constellation": constellation,
    })

    record_state_change("copilot_status", before="analyzing", after="recommendation_ready")

    return {
        "satellite_id": satellite_id,
        "anomaly_type": anomaly_type,
        "severity": diagnosis["severity"],
        "diagnosis": diagnosis["diagnosis"],
        "procedure": procedure["procedure_id"],
        "escalated": needs_escalation,
    }


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Mission Operations Copilot — Demo")
    print("=" * 60)
    print()

    # Single run
    result = run_copilot({"satellite_id": "SAT-001", "concern": "battery voltage dropping"})
    print(f"\nResult: {result}")

    # Batch run
    print()
    print("=" * 60)
    print("  Batch Analysis — 5 Satellites")
    print("=" * 60)
    print()

    session = TraceSession(name="constellation-check")
    for i in range(5):
        result = run_copilot({"satellite_id": f"SAT-{i+1:03d}"})
        session.add_trace(probe.traces[-1])

    session_dir = session.save("traces")
    summary = session.summary()
    print(f"\nSession: {session_dir}")
    print(f"  Runs: {summary['total_traces']}")
    print(f"  Pass rate: {summary['pass_rate']:.0%}")
    print(f"  Avg tokens: {summary['avg_tokens']:.0f}")
