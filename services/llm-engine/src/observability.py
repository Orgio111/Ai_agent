"""OpenTelemetry + Prometheus instrumentation for the LLM engine."""
from __future__ import annotations

import os

from prometheus_client import Counter, Histogram

TOKENS_USED = Counter(
    "jarvis_llm_tokens_total",
    "Total tokens used",
    ["model", "type"],
)

LLM_LATENCY = Histogram(
    "jarvis_llm_latency_seconds",
    "LLM request latency in seconds",
    ["model", "tier"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

REQUESTS_TOTAL = Counter(
    "jarvis_llm_requests_total",
    "Total LLM requests",
    ["endpoint", "model", "status"],
)


def setup_telemetry() -> None:
    otel_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if not otel_endpoint:
        return

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=otel_endpoint)))
    trace.set_tracer_provider(provider)


def track_request(endpoint: str, model: str, latency: float, success: bool) -> None:
    status = "success" if success else "error"
    REQUESTS_TOTAL.labels(endpoint=endpoint, model=model, status=status).inc()
