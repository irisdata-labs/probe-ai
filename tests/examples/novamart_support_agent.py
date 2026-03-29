"""
NovaMart Customer Support Agent — agentprobe SDK Demo

Demonstrates every feature of the SDK:
- @trace_agent, @trace_step, @trace_tool, @trace_llm_call
- step_span context manager
- record_state_change() and record_decision()
- ConsoleExporter and JSONFileExporter
- TraceSession for batch runs

Run:
    python examples/novamart_support_agent.py
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

# ============================================================================
# Setup
# ============================================================================

probe = AgentProbe(exporters=[
    ConsoleExporter(verbose=True),
    JSONFileExporter("traces/"),
])
probe.init()


# ============================================================================
# Simulated Tools
# ============================================================================

@trace_tool("order-lookup")
def lookup_order(order_id: str) -> dict:
    """Simulate looking up an order from the database."""
    time.sleep(0.02)  # simulate latency
    return {
        "order_id": order_id,
        "customer_name": "Alex Johnson",
        "status": "delivered",
        "delivery_date": "2025-12-20",
        "days_since_delivery": random.randint(5, 45),
        "items": [{"name": "Wireless Headphones", "price": 79.99}],
        "total": 79.99,
        "payment_method": "credit_card",
    }


@trace_tool("product-search")
def search_products(query: str) -> dict:
    time.sleep(0.01)
    return {
        "results": [
            {"name": "Wireless Headphones Pro", "price": 129.99, "in_stock": True},
            {"name": "Wireless Earbuds", "price": 49.99, "in_stock": True},
        ],
        "total_results": 2,
    }


@trace_tool("process-return")
def process_return(order_id: str, reason: str) -> dict:
    time.sleep(0.03)
    return {
        "return_id": f"RET-{random.randint(1000, 9999)}",
        "status": "initiated",
        "refund_amount": 79.99,
        "refund_method": "credit_card",
    }


@trace_tool("escalate-to-human")
def escalate(order_id: str, reason: str) -> dict:
    time.sleep(0.01)
    return {"ticket_id": f"TKT-{random.randint(1000, 9999)}", "queue": "tier2"}


# ============================================================================
# Simulated LLM
# ============================================================================

@trace_llm_call("generate-response")
def generate_response(context: dict) -> str:
    """Simulate an LLM generating a customer response."""
    time.sleep(0.05)

    # Attach LLM metadata to the current span
    span = get_current_span()
    if span:
        span.set_llm_metadata(
            model="claude-sonnet-4-20250514",
            prompt_tokens=random.randint(200, 500),
            completion_tokens=random.randint(50, 150),
            total_tokens=random.randint(250, 650),
            temperature=0.3,
        )

    if context.get("action") == "approve_return":
        return (f"Hi {context['customer_name']}, I've initiated your return "
                f"(#{context['return_id']}). You'll receive a refund of "
                f"${context['refund_amount']:.2f} to your card within 5-7 business days.")
    elif context.get("action") == "deny_return":
        return (f"Hi {context['customer_name']}, I'm sorry but your order is outside "
                f"our 30-day return window ({context['days']} days since delivery). "
                f"Is there anything else I can help with?")
    elif context.get("action") == "escalate":
        return (f"Hi {context['customer_name']}, I'm connecting you with a specialist "
                f"who can better assist you. Your ticket number is {context['ticket_id']}.")
    return "How can I help you today?"


# ============================================================================
# Agent Pipeline
# ============================================================================

@trace_agent("novamart-support", tags=["demo", "v1"])
def handle_customer(message: str, order_id: str = "ORD-9821") -> dict:
    """Main agent entry point."""

    # Step 1: Look up the order
    order = lookup_order(order_id)

    # Step 2: Classify the intent
    with step_span("classify-intent", step_type="reasoning",
                    input_data={"message": message}) as span:
        intent = "return" if "return" in message.lower() else "inquiry"
        span.set_output({"intent": intent})

    # Step 3: Route based on intent
    if intent == "return":
        return handle_return(order, message)
    else:
        context = {"customer_name": order["customer_name"], "action": "inquiry"}
        response = generate_response(context)
        return {"action": "respond", "response": response}


def handle_return(order: dict, message: str) -> dict:
    """Handle return requests with policy checks."""

    # Check return window
    days = order["days_since_delivery"]

    with step_span("evaluate-return-policy", step_type="reasoning",
                    input_data={"days": days, "amount": order["total"]}) as span:

        # Decision: within return window?
        if days > 30:
            record_decision(
                "deny_return",
                alternatives=["approve_return", "escalate"],
                reason=f"Outside 30-day window ({days} days)",
            )
            span.set_output({"eligible": False, "reason": "outside_window"})

            context = {
                "customer_name": order["customer_name"],
                "action": "deny_return",
                "days": days,
            }
            response = generate_response(context)
            return {"action": "deny_return", "response": response}

        # Decision: high value needs escalation?
        if order["total"] > 500:
            record_decision(
                "escalate",
                alternatives=["approve_return"],
                reason=f"High value order (${order['total']:.2f})",
            )
            span.set_output({"eligible": True, "escalated": True})

            result = escalate(order["order_id"], "high_value_return")
            context = {
                "customer_name": order["customer_name"],
                "action": "escalate",
                "ticket_id": result["ticket_id"],
            }
            response = generate_response(context)
            return {"action": "escalate", "response": response}

        # Approve the return
        record_decision(
            "approve_return",
            alternatives=["deny_return", "escalate"],
            reason=f"Within window ({days} days), normal value",
        )
        span.set_output({"eligible": True, "escalated": False})

    # Process the return
    result = process_return(order["order_id"], "customer_request")

    # Record state change
    record_state_change(
        "refund_status",
        before="none",
        after="processing",
    )
    record_state_change(
        "order_status",
        before=order["status"],
        after="return_initiated",
    )

    context = {
        "customer_name": order["customer_name"],
        "action": "approve_return",
        "return_id": result["return_id"],
        "refund_amount": result["refund_amount"],
    }
    response = generate_response(context)
    return {"action": "approve_return", "response": response}


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  NovaMart Support Agent — Single Run")
    print("=" * 60)
    print()

    result = handle_customer("I want to return my headphones", "ORD-9821")
    print(f"\nAgent response: {result['response']}")

    print()
    print("=" * 60)
    print("  Batch Run with Session")
    print("=" * 60)
    print()

    session = TraceSession(name="novamart-demo")
    messages = [
        ("I want to return my headphones", "ORD-1001"),
        ("What's the status of my order?", "ORD-1002"),
        ("I need to return this, it's broken", "ORD-1003"),
        ("Please return order 1004", "ORD-1004"),
        ("Can I get a refund?", "ORD-1005"),
    ]

    for msg, oid in messages:
        result = handle_customer(msg, oid)
        trace = probe.traces[-1]
        session.add_trace(trace)

    # Save session and print summary
    session_dir = session.save("traces")
    print(f"\nSession saved to: {session_dir}")

    summary = session.summary()
    print(f"\nSession Summary:")
    print(f"  Total runs:    {summary['total_traces']}")
    print(f"  Pass rate:     {summary['pass_rate']:.0%}")
    print(f"  Avg duration:  {summary['avg_duration_ms']:.0f}ms")
    print(f"  Total tokens:  {summary['total_tokens']}")
