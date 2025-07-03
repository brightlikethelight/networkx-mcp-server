"""Graph event system for NetworkX MCP Server.

This module provides an event-driven architecture for graph operations,
enabling loose coupling between components and extensible functionality.
"""

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ..core.base import Component, ComponentStatus

logger = logging.getLogger(__name__)


class GraphEventType(Enum):
    """Types of graph events."""

    # Graph lifecycle
    GRAPH_CREATED = "graph_created"
    GRAPH_DELETED = "graph_deleted"
    GRAPH_LOADED = "graph_loaded"
    GRAPH_SAVED = "graph_saved"

    # Graph modifications
    NODES_ADDED = "nodes_added"
    NODES_REMOVED = "nodes_removed"
    NODES_UPDATED = "nodes_updated"
    EDGES_ADDED = "edges_added"
    EDGES_REMOVED = "edges_removed"
    EDGES_UPDATED = "edges_updated"

    # Algorithm operations
    ALGORITHM_STARTED = "algorithm_started"
    ALGORITHM_COMPLETED = "algorithm_completed"
    ALGORITHM_FAILED = "algorithm_failed"

    # Performance events
    MEMORY_WARNING = "memory_warning"
    PERFORMANCE_ALERT = "performance_alert"
    TIMEOUT_WARNING = "timeout_warning"

    # Security events
    VALIDATION_FAILED = "validation_failed"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


@dataclass
class GraphEvent:
    """A graph event."""

    event_type: GraphEventType
    graph_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_type": self.event_type.value,
            "graph_id": self.graph_id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphEvent":
        """Create event from dictionary."""
        return cls(
            event_type=GraphEventType(data["event_type"]),
            graph_id=data["graph_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data.get("source", "unknown"),
            metadata=data.get("metadata", {}),
        )


class EventSubscription:
    """Subscription to events."""

    def __init__(
        self,
        handler: Callable[[GraphEvent], None],
        event_types: list[GraphEventType] | None = None,
        graph_filter: str | None = None,
        source_filter: str | None = None,
    ):
        self.handler = handler
        self.event_types = event_types or list(GraphEventType)
        self.graph_filter = graph_filter
        self.source_filter = source_filter
        self.active = True

    def matches(self, event: GraphEvent) -> bool:
        """Check if subscription matches event."""
        if not self.active:
            return False

        if event.event_type not in self.event_types:
            return False

        if self.graph_filter and event.graph_id != self.graph_filter:
            return False

        if self.source_filter and event.source != self.source_filter:
            return False

        return True


class GraphEventPublisher(Component):
    """Publisher for graph events."""

    def __init__(self, name: str = "GraphEventPublisher"):
        super().__init__(name)
        self._subscriptions: list[EventSubscription] = []
        self._event_history: list[GraphEvent] = []
        self._max_history = 10000
        self._metrics = {
            "events_published": 0,
            "events_by_type": {},
            "failed_deliveries": 0,
        }

    async def initialize(self) -> None:
        """Initialize the event publisher."""
        await self._set_status(ComponentStatus.READY)
        logger.info("Graph event publisher initialized")

    async def shutdown(self) -> None:
        """Shutdown the event publisher."""
        await self._set_status(ComponentStatus.SHUTTING_DOWN)
        self._subscriptions.clear()
        self._event_history.clear()
        await self._set_status(ComponentStatus.SHUTDOWN)
        logger.info("Graph event publisher shutdown")

    def subscribe(
        self,
        handler: Callable[[GraphEvent], None],
        event_types: list[GraphEventType] | None = None,
        graph_filter: str | None = None,
        source_filter: str | None = None,
    ) -> EventSubscription:
        """Subscribe to events."""
        subscription = EventSubscription(
            handler=handler,
            event_types=event_types,
            graph_filter=graph_filter,
            source_filter=source_filter,
        )
        self._subscriptions.append(subscription)

        logger.debug(
            f"Added event subscription for {len(event_types or [])} event types"
        )
        return subscription

    def unsubscribe(self, subscription: EventSubscription) -> None:
        """Unsubscribe from events."""
        subscription.active = False
        if subscription in self._subscriptions:
            self._subscriptions.remove(subscription)
        logger.debug("Removed event subscription")

    async def publish(self, event: GraphEvent) -> None:
        """Publish an event to all subscribers."""
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)

        # Update metrics
        self._metrics["events_published"] += 1
        event_type_key = event.event_type.value
        self._metrics["events_by_type"][event_type_key] = (
            self._metrics["events_by_type"].get(event_type_key, 0) + 1
        )

        # Publish to subscribers
        for subscription in self._subscriptions[
            :
        ]:  # Copy to avoid modification during iteration
            if subscription.matches(event):
                try:
                    if asyncio.iscoroutinefunction(subscription.handler):
                        await subscription.handler(event)
                    else:
                        subscription.handler(event)
                except Exception as e:
                    logger.error(f"Event handler failed: {e}")
                    self._metrics["failed_deliveries"] += 1

        logger.debug(
            f"Published event: {event.event_type.value} for graph {event.graph_id}"
        )

    # Convenience methods for common events
    async def publish_graph_created(
        self, graph_id: str, graph_type: str, source: str = "graph_service"
    ) -> None:
        """Publish graph created event."""
        event = GraphEvent(
            event_type=GraphEventType.GRAPH_CREATED,
            graph_id=graph_id,
            source=source,
            metadata={"graph_type": graph_type},
        )
        await self.publish(event)

    async def publish_graph_deleted(
        self, graph_id: str, source: str = "graph_service"
    ) -> None:
        """Publish graph deleted event."""
        event = GraphEvent(
            event_type=GraphEventType.GRAPH_DELETED, graph_id=graph_id, source=source
        )
        await self.publish(event)

    async def publish_graph_loaded(
        self, graph_id: str, source: str = "graph_service"
    ) -> None:
        """Publish graph loaded event."""
        event = GraphEvent(
            event_type=GraphEventType.GRAPH_LOADED, graph_id=graph_id, source=source
        )
        await self.publish(event)

    async def publish_graph_saved(
        self, graph_id: str, source: str = "graph_service"
    ) -> None:
        """Publish graph saved event."""
        event = GraphEvent(
            event_type=GraphEventType.GRAPH_SAVED, graph_id=graph_id, source=source
        )
        await self.publish(event)

    async def publish_nodes_added(
        self, graph_id: str, count: int, source: str = "graph_service"
    ) -> None:
        """Publish nodes added event."""
        event = GraphEvent(
            event_type=GraphEventType.NODES_ADDED,
            graph_id=graph_id,
            source=source,
            metadata={"count": count},
        )
        await self.publish(event)

    async def publish_nodes_removed(
        self, graph_id: str, count: int, source: str = "graph_service"
    ) -> None:
        """Publish nodes removed event."""
        event = GraphEvent(
            event_type=GraphEventType.NODES_REMOVED,
            graph_id=graph_id,
            source=source,
            metadata={"count": count},
        )
        await self.publish(event)

    async def publish_edges_added(
        self, graph_id: str, count: int, source: str = "graph_service"
    ) -> None:
        """Publish edges added event."""
        event = GraphEvent(
            event_type=GraphEventType.EDGES_ADDED,
            graph_id=graph_id,
            source=source,
            metadata={"count": count},
        )
        await self.publish(event)

    async def publish_edges_removed(
        self, graph_id: str, count: int, source: str = "graph_service"
    ) -> None:
        """Publish edges removed event."""
        event = GraphEvent(
            event_type=GraphEventType.EDGES_REMOVED,
            graph_id=graph_id,
            source=source,
            metadata={"count": count},
        )
        await self.publish(event)

    async def publish_algorithm_started(
        self,
        graph_id: str,
        algorithm: str,
        parameters: dict[str, Any],
        source: str = "algorithm_service",
    ) -> None:
        """Publish algorithm started event."""
        event = GraphEvent(
            event_type=GraphEventType.ALGORITHM_STARTED,
            graph_id=graph_id,
            source=source,
            metadata={"algorithm": algorithm, "parameters": parameters},
        )
        await self.publish(event)

    async def publish_algorithm_completed(
        self,
        graph_id: str,
        algorithm: str,
        execution_time: float,
        source: str = "algorithm_service",
    ) -> None:
        """Publish algorithm completed event."""
        event = GraphEvent(
            event_type=GraphEventType.ALGORITHM_COMPLETED,
            graph_id=graph_id,
            source=source,
            metadata={"algorithm": algorithm, "execution_time": execution_time},
        )
        await self.publish(event)

    async def publish_algorithm_failed(
        self,
        graph_id: str,
        algorithm: str,
        error: str,
        source: str = "algorithm_service",
    ) -> None:
        """Publish algorithm failed event."""
        event = GraphEvent(
            event_type=GraphEventType.ALGORITHM_FAILED,
            graph_id=graph_id,
            source=source,
            metadata={"algorithm": algorithm, "error": error},
        )
        await self.publish(event)

    async def publish_memory_warning(
        self,
        graph_id: str,
        memory_usage_mb: float,
        threshold_mb: float,
        source: str = "performance_monitor",
    ) -> None:
        """Publish memory warning event."""
        event = GraphEvent(
            event_type=GraphEventType.MEMORY_WARNING,
            graph_id=graph_id,
            source=source,
            metadata={"memory_usage_mb": memory_usage_mb, "threshold_mb": threshold_mb},
        )
        await self.publish(event)

    async def publish_validation_failed(
        self,
        graph_id: str,
        operation: str,
        errors: list[str],
        source: str = "validator",
    ) -> None:
        """Publish validation failed event."""
        event = GraphEvent(
            event_type=GraphEventType.VALIDATION_FAILED,
            graph_id=graph_id,
            source=source,
            metadata={"operation": operation, "errors": errors},
        )
        await self.publish(event)

    def get_event_history(
        self,
        graph_id: str | None = None,
        event_type: GraphEventType | None = None,
        limit: int = 100,
    ) -> list[GraphEvent]:
        """Get event history with optional filtering."""
        filtered_events = []

        for event in reversed(self._event_history):
            if graph_id and event.graph_id != graph_id:
                continue

            if event_type and event.event_type != event_type:
                continue

            filtered_events.append(event)

            if len(filtered_events) >= limit:
                break

        return filtered_events

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        return {
            "healthy": self.status == ComponentStatus.READY,
            "subscriptions": len([s for s in self._subscriptions if s.active]),
            "events_in_history": len(self._event_history),
            "metrics": self._metrics.copy(),
        }

    def get_metrics(self) -> dict[str, Any]:
        """Get event metrics."""
        return self._metrics.copy()


class EventListener:
    """Base class for event listeners."""

    def __init__(self, name: str):
        self.name = name
        self._subscriptions: list[EventSubscription] = []

    async def initialize(self, publisher: GraphEventPublisher) -> None:
        """Initialize the listener with event publisher."""
        self.publisher = publisher
        await self._setup_subscriptions()

    async def _setup_subscriptions(self) -> None:
        """Setup event subscriptions."""
        # Override in subclasses

    def subscribe_to_events(
        self,
        event_types: list[GraphEventType],
        handler: Callable[[GraphEvent], None],
        graph_filter: str | None = None,
    ) -> EventSubscription:
        """Subscribe to specific events."""
        subscription = self.publisher.subscribe(
            handler=handler,
            event_types=event_types,
            graph_filter=graph_filter,
            source_filter=None,
        )
        self._subscriptions.append(subscription)
        return subscription

    async def shutdown(self) -> None:
        """Shutdown the listener."""
        for subscription in self._subscriptions:
            self.publisher.unsubscribe(subscription)
        self._subscriptions.clear()


class LoggingEventListener(EventListener):
    """Event listener that logs important events."""

    def __init__(self):
        super().__init__("LoggingEventListener")
        self.logger = logging.getLogger(f"{__name__}.LoggingEventListener")

    async def _setup_subscriptions(self) -> None:
        """Setup logging subscriptions."""
        # Log all graph lifecycle events
        lifecycle_events = [
            GraphEventType.GRAPH_CREATED,
            GraphEventType.GRAPH_DELETED,
            GraphEventType.GRAPH_LOADED,
            GraphEventType.GRAPH_SAVED,
        ]
        self.subscribe_to_events(lifecycle_events, self._log_lifecycle_event)

        # Log algorithm events
        algorithm_events = [
            GraphEventType.ALGORITHM_STARTED,
            GraphEventType.ALGORITHM_COMPLETED,
            GraphEventType.ALGORITHM_FAILED,
        ]
        self.subscribe_to_events(algorithm_events, self._log_algorithm_event)

        # Log warnings and errors
        warning_events = [
            GraphEventType.MEMORY_WARNING,
            GraphEventType.PERFORMANCE_ALERT,
            GraphEventType.VALIDATION_FAILED,
        ]
        self.subscribe_to_events(warning_events, self._log_warning_event)

    def _log_lifecycle_event(self, event: GraphEvent) -> None:
        """Log lifecycle events."""
        self.logger.info(f"Graph {event.graph_id}: {event.event_type.value}")

    def _log_algorithm_event(self, event: GraphEvent) -> None:
        """Log algorithm events."""
        algorithm = event.metadata.get("algorithm", "unknown")
        if event.event_type == GraphEventType.ALGORITHM_COMPLETED:
            exec_time = event.metadata.get("execution_time", 0)
            self.logger.info(
                f"Algorithm {algorithm} completed on {event.graph_id} in {exec_time:.3f}s"
            )
        elif event.event_type == GraphEventType.ALGORITHM_FAILED:
            error = event.metadata.get("error", "unknown error")
            self.logger.error(
                f"Algorithm {algorithm} failed on {event.graph_id}: {error}"
            )
        else:
            self.logger.info(f"Algorithm {algorithm} started on {event.graph_id}")

    def _log_warning_event(self, event: GraphEvent) -> None:
        """Log warning events."""
        self.logger.warning(
            f"Graph {event.graph_id}: {event.event_type.value} - {event.metadata}"
        )


class MetricsEventListener(EventListener):
    """Event listener that collects metrics."""

    def __init__(self):
        super().__init__("MetricsEventListener")
        self.metrics = {
            "graphs_created": 0,
            "graphs_deleted": 0,
            "algorithms_run": 0,
            "algorithm_failures": 0,
            "nodes_added_total": 0,
            "edges_added_total": 0,
            "validation_failures": 0,
        }

    async def _setup_subscriptions(self) -> None:
        """Setup metrics subscriptions."""
        # Subscribe to all events for metrics collection
        self.subscribe_to_events(list(GraphEventType), self._collect_metrics)

    def _collect_metrics(self, event: GraphEvent) -> None:
        """Collect metrics from events."""
        if event.event_type == GraphEventType.GRAPH_CREATED:
            self.metrics["graphs_created"] += 1
        elif event.event_type == GraphEventType.GRAPH_DELETED:
            self.metrics["graphs_deleted"] += 1
        elif event.event_type == GraphEventType.ALGORITHM_COMPLETED:
            self.metrics["algorithms_run"] += 1
        elif event.event_type == GraphEventType.ALGORITHM_FAILED:
            self.metrics["algorithm_failures"] += 1
        elif event.event_type == GraphEventType.NODES_ADDED:
            count = event.metadata.get("count", 0)
            self.metrics["nodes_added_total"] += count
        elif event.event_type == GraphEventType.EDGES_ADDED:
            count = event.metadata.get("count", 0)
            self.metrics["edges_added_total"] += count
        elif event.event_type == GraphEventType.VALIDATION_FAILED:
            self.metrics["validation_failures"] += 1

    def get_metrics(self) -> dict[str, Any]:
        """Get collected metrics."""
        return self.metrics.copy()
