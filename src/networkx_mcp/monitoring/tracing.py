"""Distributed tracing service with OpenTelemetry integration."""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from ..core.base import Component

logger = logging.getLogger(__name__)


@dataclass
class SpanContext:
    """Span context for distributed tracing."""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    baggage: dict[str, str] = field(default_factory=dict)


@dataclass
class Span:
    """A single span in a distributed trace."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    operation_name: str
    start_time: float
    end_time: float | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"  # ok, error, timeout

    @property
    def duration(self) -> float | None:
        """Get span duration in seconds."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None

    def add_tag(self, key: str, value: Any) -> None:
        """Add a tag to the span."""
        self.tags[key] = value

    def add_log(self, message: str, level: str = "info", **kwargs) -> None:
        """Add a log entry to the span."""
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            **kwargs,
        }
        self.logs.append(log_entry)

    def set_error(self, error: Exception) -> None:
        """Mark span as error and add error details."""
        self.status = "error"
        self.add_tag("error", True)
        self.add_tag("error.type", type(error).__name__)
        self.add_tag("error.message", str(error))
        self.add_log(f"Error: {error}", level="error")

    def finish(self) -> None:
        """Finish the span."""
        if self.end_time is None:
            self.end_time = time.time()


class TraceContext:
    """Context for managing current trace and span."""

    def __init__(self):
        self._local = asyncio.local()

    def get_current_span(self) -> Span | None:
        """Get the current active span."""
        return getattr(self._local, "current_span", None)

    def set_current_span(self, span: Span | None) -> None:
        """Set the current active span."""
        self._local.current_span = span

    def get_trace_id(self) -> str | None:
        """Get the current trace ID."""
        span = self.get_current_span()
        return span.trace_id if span else None

    def get_span_id(self) -> str | None:
        """Get the current span ID."""
        span = self.get_current_span()
        return span.span_id if span else None


class TracingService(Component):
    """Service for distributed tracing with OpenTelemetry integration."""

    def __init__(self, service_name: str = "networkx-mcp-server"):
        super().__init__("tracing_service")
        self.service_name = service_name
        self.context = TraceContext()
        self.spans: dict[str, Span] = {}
        self.finished_spans: list[Span] = []
        self._max_finished_spans = 10000

    def start_trace(self, operation_name: str, **tags) -> Span:
        """Start a new trace with a root span."""
        trace_id = self._generate_trace_id()
        span_id = self._generate_span_id()

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            operation_name=operation_name,
            start_time=time.time(),
        )

        # Add service tags
        span.add_tag("service.name", self.service_name)
        span.add_tag("span.kind", "server")

        # Add custom tags
        for key, value in tags.items():
            span.add_tag(key, value)

        self.spans[span_id] = span
        self.context.set_current_span(span)

        logger.debug(f"Started trace {trace_id} with root span {span_id}")
        return span

    def start_span(
        self, operation_name: str, parent_span: Span | None = None, **tags
    ) -> Span:
        """Start a new span."""
        if parent_span is None:
            parent_span = self.context.get_current_span()

        if parent_span is None:
            # Start a new trace if no parent
            return self.start_trace(operation_name, **tags)

        span_id = self._generate_span_id()

        span = Span(
            trace_id=parent_span.trace_id,
            span_id=span_id,
            parent_span_id=parent_span.span_id,
            operation_name=operation_name,
            start_time=time.time(),
        )

        # Add service tags
        span.add_tag("service.name", self.service_name)
        span.add_tag("span.kind", "internal")

        # Add custom tags
        for key, value in tags.items():
            span.add_tag(key, value)

        self.spans[span_id] = span

        logger.debug(f"Started span {span_id} in trace {span.trace_id}")
        return span

    def finish_span(self, span: Span) -> None:
        """Finish a span and move it to finished spans."""
        span.finish()

        # Remove from active spans
        if span.span_id in self.spans:
            del self.spans[span.span_id]

        # Add to finished spans
        self.finished_spans.append(span)

        # Limit finished spans to prevent memory issues
        if len(self.finished_spans) > self._max_finished_spans:
            self.finished_spans = self.finished_spans[-self._max_finished_spans :]

        logger.debug(f"Finished span {span.span_id} (duration: {span.duration:.4f}s)")

    @asynccontextmanager
    async def trace(self, operation_name: str, **tags) -> AsyncGenerator[Span, None]:
        """Async context manager for tracing operations."""
        span = self.start_span(operation_name, **tags)
        old_span = self.context.get_current_span()

        try:
            self.context.set_current_span(span)
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            self.finish_span(span)
            self.context.set_current_span(old_span)

    def add_event(self, name: str, **attributes) -> None:
        """Add an event to the current span."""
        span = self.context.get_current_span()
        if span:
            span.add_log(name, **attributes)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the current span."""
        span = self.context.get_current_span()
        if span:
            span.add_tag(key, value)

    def get_trace_info(self) -> dict[str, str | None]:
        """Get current trace information."""
        return {
            "trace_id": self.context.get_trace_id(),
            "span_id": self.context.get_span_id(),
        }

    def get_active_spans(self) -> list[Span]:
        """Get all currently active spans."""
        return list(self.spans.values())

    def get_finished_spans(self, trace_id: str | None = None) -> list[Span]:
        """Get finished spans, optionally filtered by trace ID."""
        if trace_id:
            return [span for span in self.finished_spans if span.trace_id == trace_id]
        return self.finished_spans.copy()

    def export_spans(self, format_type: str = "jaeger") -> list[dict[str, Any]]:
        """Export spans in specified format."""
        spans_data = []

        for span in self.finished_spans:
            span_data = {
                "traceID": span.trace_id,
                "spanID": span.span_id,
                "parentSpanID": span.parent_span_id,
                "operationName": span.operation_name,
                "startTime": int(span.start_time * 1_000_000),  # microseconds
                "duration": int((span.duration or 0) * 1_000_000),  # microseconds
                "tags": [
                    {"key": k, "value": v, "type": self._get_tag_type(v)}
                    for k, v in span.tags.items()
                ],
                "logs": [
                    {
                        "timestamp": int(log["timestamp"] * 1_000_000),
                        "fields": [
                            {"key": k, "value": v}
                            for k, v in log.items()
                            if k != "timestamp"
                        ],
                    }
                    for log in span.logs
                ],
                "process": {
                    "serviceName": self.service_name,
                    "tags": [
                        {
                            "key": "service.name",
                            "value": self.service_name,
                            "type": "string",
                        }
                    ],
                },
            }
            spans_data.append(span_data)

        return spans_data

    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        return str(uuid.uuid4()).replace("-", "")

    def _generate_span_id(self) -> str:
        """Generate a unique span ID."""
        return str(uuid.uuid4()).replace("-", "")[:16]

    def _get_tag_type(self, value: Any) -> str:
        """Get the tag type for tracing export."""
        if isinstance(value, bool):
            return "bool"
        elif isinstance(value, int):
            return "number"
        elif isinstance(value, float):
            return "number"
        else:
            return "string"

    async def initialize(self) -> None:
        """Initialize the tracing service."""
        await super().initialize()
        logger.info(f"Tracing service initialized for {self.service_name}")

    async def cleanup(self) -> None:
        """Cleanup the tracing service."""
        # Finish any remaining active spans
        for span in list(self.spans.values()):
            span.add_log("Service shutdown", level="info")
            self.finish_span(span)

        await super().cleanup()
        logger.info("Tracing service cleaned up")


# Global tracing service instance
_tracing_service: TracingService | None = None


def get_tracer() -> TracingService:
    """Get the global tracing service instance."""
    global _tracing_service
    if _tracing_service is None:
        _tracing_service = TracingService()
    return _tracing_service


def trace(operation_name: str, **tags):
    """Decorator for tracing functions."""

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                tracer = get_tracer()
                async with tracer.trace(operation_name, **tags) as span:
                    # Add function info
                    span.add_tag("function.name", func.__name__)
                    span.add_tag("function.module", func.__module__)
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                tracer = get_tracer()
                span = tracer.start_span(operation_name, **tags)
                span.add_tag("function.name", func.__name__)
                span.add_tag("function.module", func.__module__)

                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_error(e)
                    raise
                finally:
                    tracer.finish_span(span)

            return sync_wrapper

    return decorator
