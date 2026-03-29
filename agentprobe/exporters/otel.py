"""
OpenTelemetry exporter — converts agentprobe traces to OTel spans.

Allows traces to flow into Datadog, Grafana, Jaeger, or any OTel-compatible
backend alongside existing application telemetry.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from agentprobe.models import SpanRecord, TraceRecord

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SpanExporter,
    )
    from opentelemetry.trace.status import Status, StatusCode
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False


class OTelExporter:
    """Converts agentprobe traces to OpenTelemetry spans."""

    def __init__(
        self,
        service_name: str = "agentprobe",
        span_exporter: Optional[Any] = None,
    ):
        if not HAS_OTEL:
            raise ImportError(
                "OpenTelemetry SDK not installed. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk"
            )

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        exporter = span_exporter or ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter))

        self._tracer = provider.get_tracer("agentprobe")
        self._provider = provider

    def export(self, trace_record: TraceRecord) -> None:
        with self._tracer.start_as_current_span(trace_record.agent_name) as root:
            # Set root span attributes
            for k, v in _trace_attributes(trace_record).items():
                root.set_attribute(k, v)

            if trace_record.status == "ok":
                root.set_status(Status(StatusCode.OK))
            elif trace_record.status == "error":
                root.set_status(Status(StatusCode.ERROR,
                                       trace_record.status_message or ""))

            # Create child spans for each step
            for span_record in trace_record.spans:
                self._export_span(span_record, root)

    def _export_span(self, span_record: SpanRecord, parent: Any) -> None:
        with self._tracer.start_as_current_span(
            span_record.name,
            context=trace.set_span_in_context(parent),
        ) as span:
            for k, v in _span_attributes(span_record).items():
                span.set_attribute(k, v)

            if span_record.status == "ok":
                span.set_status(Status(StatusCode.OK))
            elif span_record.status == "error":
                span.set_status(Status(StatusCode.ERROR,
                                       span_record.status_message or ""))

    def shutdown(self) -> None:
        self._provider.shutdown()


def _trace_attributes(t: TraceRecord) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {
        "agentprobe.trace_id": t.trace_id,
        "agentprobe.agent_name": t.agent_name,
        "agentprobe.step_count": t.step_count,
        "agentprobe.total_tokens": t.total_tokens,
        "agentprobe.llm_calls": len(t.llm_calls),
        "agentprobe.tool_calls": len(t.tool_calls),
    }
    if t.duration_ms:
        attrs["agentprobe.duration_ms"] = t.duration_ms
    if t.tags:
        attrs["agentprobe.tags"] = ",".join(t.tags)
    if t.thread_id:
        attrs["agentprobe.thread_id"] = t.thread_id
    if t.state_changes:
        attrs["agentprobe.state_changes"] = len(t.state_changes)
    if t.decisions:
        attrs["agentprobe.decisions"] = len(t.decisions)
    return attrs


def _span_attributes(s: SpanRecord) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {
        "agentprobe.span_type": s.span_type,
    }
    if s.duration_ms:
        attrs["agentprobe.duration_ms"] = s.duration_ms
    if s.model:
        attrs["llm.model"] = s.model
    if s.prompt_tokens is not None:
        attrs["llm.prompt_tokens"] = s.prompt_tokens
    if s.completion_tokens is not None:
        attrs["llm.completion_tokens"] = s.completion_tokens
    if s.total_tokens is not None:
        attrs["llm.total_tokens"] = s.total_tokens
    if s.tool_name:
        attrs["tool.name"] = s.tool_name
    if s.tool_success is not None:
        attrs["tool.success"] = s.tool_success
    if s.tags:
        attrs["agentprobe.tags"] = ",".join(s.tags)
    return attrs
