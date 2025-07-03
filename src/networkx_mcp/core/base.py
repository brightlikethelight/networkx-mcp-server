"""Base classes and interfaces for NetworkX MCP Server.

This module defines the core abstractions and base classes that form
the foundation of the modular architecture.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar

# Type variables
T = TypeVar("T")
GraphType = TypeVar("GraphType")

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Status of a server component."""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


@dataclass
class ComponentMetrics:
    """Metrics for a server component."""

    requests_total: int = 0
    requests_failed: int = 0
    processing_time_total: float = 0.0
    last_request_time: datetime | None = None
    error_count: int = 0

    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time per request."""
        if self.requests_total == 0:
            return 0.0
        return self.processing_time_total / self.requests_total

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.requests_total == 0:
            return 100.0
        return (
            (self.requests_total - self.requests_failed) / self.requests_total
        ) * 100


class Component(ABC):
    """Base class for all server components."""

    def __init__(self, name: str):
        self.name = name
        self.status = ComponentStatus.UNINITIALIZED
        self.metrics = ComponentMetrics()
        self._lock = asyncio.Lock()
        logger.info(f"Component {name} created")

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the component gracefully."""

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Perform health check and return status."""

    async def _set_status(self, status: ComponentStatus) -> None:
        """Update component status thread-safely."""
        async with self._lock:
            self.status = status
            logger.debug(f"Component {self.name} status changed to {status.value}")

    def get_metrics(self) -> dict[str, Any]:
        """Get component metrics as dictionary."""
        return {
            "component": self.name,
            "status": self.status.value,
            "requests_total": self.metrics.requests_total,
            "requests_failed": self.metrics.requests_failed,
            "success_rate": self.metrics.success_rate,
            "average_processing_time": self.metrics.average_processing_time,
            "error_count": self.metrics.error_count,
            "last_request_time": (
                self.metrics.last_request_time.isoformat()
                if self.metrics.last_request_time
                else None
            ),
        }


class Handler(Protocol):
    """Protocol for request handlers."""

    async def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle a request and return response."""
        ...


class Middleware(ABC):
    """Base class for middleware components."""

    @abstractmethod
    async def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process request before handler."""

    @abstractmethod
    async def process_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Process response after handler."""


class Repository(ABC, Generic[T]):
    """Base repository interface for data access."""

    @abstractmethod
    async def get(self, id: str) -> T | None:
        """Get entity by ID."""

    @abstractmethod
    async def list(self, **filters) -> list[T]:
        """List entities with optional filters."""

    @abstractmethod
    async def create(self, _entity: T) -> T:
        """Create new entity."""

    @abstractmethod
    async def update(self, id: str, _entity: T) -> T | None:
        """Update existing entity."""

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete entity by ID."""


class Service(Component):
    """Base class for business logic services."""

    def __init__(self, name: str):
        super().__init__(name)
        self.dependencies: dict[str, Component] = {}

    def add_dependency(self, name: str, component: Component) -> None:
        """Add a dependency to this service."""
        self.dependencies[name] = component
        logger.debug(f"Added dependency {name} to service {self.name}")

    async def initialize(self) -> None:
        """Initialize the service and verify dependencies."""
        await self._set_status(ComponentStatus.INITIALIZING)

        # Verify all dependencies are ready
        for dep_name, dep in self.dependencies.items():
            if dep.status != ComponentStatus.READY:
                raise RuntimeError(
                    f"Dependency {dep_name} not ready for service {self.name}"
                )

        await self._set_status(ComponentStatus.READY)
        logger.info(f"Service {self.name} initialized successfully")

    @asynccontextmanager
    async def track_request(self):
        """Context manager to track request metrics."""
        start_time = asyncio.get_event_loop().time()
        try:
            self.metrics.requests_total += 1
            self.metrics.last_request_time = datetime.now()
            yield
        except Exception:
            self.metrics.requests_failed += 1
            self.metrics.error_count += 1
            raise
        finally:
            processing_time = asyncio.get_event_loop().time() - start_time
            self.metrics.processing_time_total += processing_time


class EventBus(Component):
    """Simple event bus for inter-component communication."""

    def __init__(self):
        super().__init__("EventBus")
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    async def initialize(self) -> None:
        """Initialize the event bus."""
        await self._set_status(ComponentStatus.READY)

    async def shutdown(self) -> None:
        """Shutdown the event bus."""
        await self._set_status(ComponentStatus.SHUTTING_DOWN)
        # Clear all subscribers
        self._subscribers.clear()
        await self._set_status(ComponentStatus.SHUTDOWN)

    async def health_check(self) -> dict[str, Any]:
        """Check event bus health."""
        return {
            "healthy": self.status == ComponentStatus.READY,
            "subscriber_count": sum(len(subs) for subs in self._subscribers.values()),
            "event_types": len(self._subscribers),
        }

    async def publish(self, event_type: str, data: Any) -> None:
        """Publish an event to all subscribers."""
        if event_type in self._subscribers:
            for queue in self._subscribers[event_type]:
                await queue.put(data)

    async def subscribe(self, event_type: str) -> asyncio.Queue:
        """Subscribe to an event type and return queue for receiving events."""
        queue = asyncio.Queue()
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(queue)
        return queue


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)


class Validator(ABC):
    """Base class for validators."""

    @abstractmethod
    async def validate(self, data: Any) -> ValidationResult:
        """Validate data and return result."""


class Cache(ABC, Generic[T]):
    """Base cache interface."""

    @abstractmethod
    async def get(self, key: str) -> T | None:
        """Get value from cache."""

    @abstractmethod
    async def set(self, key: str, value: T, ttl: int | None = None) -> None:
        """Set value in cache with optional TTL."""

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""


class Registry(Generic[T]):
    """Generic registry for managing named components."""

    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, T] = {}
        self._lock = asyncio.Lock()

    async def register(self, name: str, item: T) -> None:
        """Register an item."""
        async with self._lock:
            if name in self._items:
                raise ValueError(f"Item {name} already registered in {self.name}")
            self._items[name] = item
            logger.debug(f"Registered {name} in {self.name}")

    async def unregister(self, name: str) -> T | None:
        """Unregister an item."""
        async with self._lock:
            return self._items.pop(name, None)

    async def get(self, name: str) -> T | None:
        """Get a registered item."""
        async with self._lock:
            return self._items.get(name)

    async def list(self) -> list[str]:
        """List all registered item names."""
        async with self._lock:
            return list(self._items.keys())

    async def clear(self) -> None:
        """Clear all registered items."""
        async with self._lock:
            self._items.clear()


class Pipeline(Generic[T]):
    """Generic pipeline for processing data through stages."""

    def __init__(self, name: str):
        self.name = name
        self._stages: list[Any] = []

    def add_stage(self, stage: Any) -> "Pipeline[T]":
        """Add a processing stage."""
        self._stages.append(stage)
        return self

    async def process(self, data: T) -> T:
        """Process data through all stages."""
        result = data
        for stage in self._stages:
            if asyncio.iscoroutinefunction(stage):
                result = await stage(result)
            else:
                result = stage(result)
        return result


# Exception hierarchy
class NetworkXMCPError(Exception):
    """Base exception for NetworkX MCP Server."""



class ComponentError(NetworkXMCPError):
    """Error in component operation."""



class ValidationError(NetworkXMCPError):
    """Validation error."""



class ConfigurationError(NetworkXMCPError):
    """Configuration error."""

