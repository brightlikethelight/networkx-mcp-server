"""Component lifecycle management for NetworkX MCP Server.

This module provides enhanced lifecycle management with:
- Async context manager support
- Graceful shutdown handling
- State tracking and validation
- Resource cleanup coordination
"""

import asyncio
import logging
import signal
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="ManagedComponent")


class LifecycleState(Enum):
    """Lifecycle states for managed components."""

    CREATED = "created"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    RESUMING = "resuming"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class LifecycleMetrics:
    """Metrics for component lifecycle."""

    created_at: datetime = field(default_factory=datetime.now)
    initialized_at: Optional[datetime] = None
    shutdown_at: Optional[datetime] = None
    restart_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    last_error_at: Optional[datetime] = None

    @property
    def uptime_seconds(self) -> Optional[float]:
        """Calculate uptime in seconds."""
        if self.initialized_at is None:
            return None
        end_time = self.shutdown_at or datetime.now()
        return (end_time - self.initialized_at).total_seconds()


class ManagedComponent(ABC):
    """Base class for components with managed lifecycle.

    Provides:
    - Lifecycle state management
    - Async context manager support
    - Graceful shutdown handling
    - Health check interface
    - Metrics tracking

    Usage:
        async with managed_component(MyComponent()) as component:
            await component.do_work()

    Or manually:
        component = MyComponent()
        await component.initialize()
        try:
            await component.do_work()
        finally:
            await component.shutdown()
    """

    def __init__(self, name: str) -> None:
        """Initialize the managed component.

        Args:
            name: Human-readable name for the component
        """
        self.name = name
        self._state = LifecycleState.CREATED
        self._metrics = LifecycleMetrics()
        self._shutdown_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._dependencies: List["ManagedComponent"] = []
        self._shutdown_hooks: List[Callable[[], Any]] = []
        logger.debug(f"Component {name} created")

    @property
    def state(self) -> LifecycleState:
        """Get current lifecycle state."""
        return self._state

    @property
    def metrics(self) -> LifecycleMetrics:
        """Get lifecycle metrics."""
        return self._metrics

    @property
    def is_running(self) -> bool:
        """Check if component is in running state."""
        return self._state == LifecycleState.RUNNING

    @property
    def is_initialized(self) -> bool:
        """Check if component has been initialized."""
        return self._state in (
            LifecycleState.RUNNING,
            LifecycleState.PAUSING,
            LifecycleState.PAUSED,
            LifecycleState.RESUMING,
        )

    def add_dependency(self, component: "ManagedComponent") -> None:
        """Add a dependency that must be initialized first.

        Args:
            component: Component that this component depends on
        """
        self._dependencies.append(component)

    def add_shutdown_hook(self, hook: Callable[[], Any]) -> None:
        """Add a hook to be called during shutdown.

        Args:
            hook: Callable to execute during shutdown (sync or async)
        """
        self._shutdown_hooks.append(hook)

    async def initialize(self) -> None:
        """Initialize the component.

        This method handles state transitions and calls _do_initialize().
        Override _do_initialize() in subclasses, not this method.
        """
        async with self._lock:
            if self._state not in (LifecycleState.CREATED, LifecycleState.STOPPED):
                logger.warning(
                    f"Component {self.name} initialize called in state {self._state}"
                )
                return

            self._state = LifecycleState.INITIALIZING
            logger.info(f"Initializing component {self.name}")

            try:
                # Initialize dependencies first
                for dep in self._dependencies:
                    if not dep.is_initialized:
                        await dep.initialize()

                # Component-specific initialization
                await self._do_initialize()

                self._state = LifecycleState.RUNNING
                self._metrics.initialized_at = datetime.now()
                self._shutdown_event.clear()
                logger.info(f"Component {self.name} initialized successfully")

            except Exception as e:
                self._state = LifecycleState.ERROR
                self._metrics.error_count += 1
                self._metrics.last_error = str(e)
                self._metrics.last_error_at = datetime.now()
                logger.error(f"Failed to initialize component {self.name}: {e}")
                raise

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Shutdown the component gracefully.

        This method handles state transitions and calls _do_shutdown().
        Override _do_shutdown() in subclasses, not this method.

        Args:
            timeout: Maximum time to wait for shutdown in seconds
        """
        async with self._lock:
            if self._state in (LifecycleState.STOPPED, LifecycleState.CREATED):
                return

            if self._state == LifecycleState.STOPPING:
                # Already shutting down, wait for completion
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout waiting for {self.name} shutdown")
                return

            self._state = LifecycleState.STOPPING
            logger.info(f"Shutting down component {self.name}")

            try:
                # Run shutdown hooks
                for hook in reversed(self._shutdown_hooks):
                    try:
                        if asyncio.iscoroutinefunction(hook):
                            await asyncio.wait_for(hook(), timeout=5.0)
                        else:
                            hook()
                    except Exception as e:
                        logger.error(f"Shutdown hook error in {self.name}: {e}")

                # Component-specific shutdown
                await asyncio.wait_for(self._do_shutdown(), timeout=timeout)

                self._state = LifecycleState.STOPPED
                self._metrics.shutdown_at = datetime.now()
                self._shutdown_event.set()
                logger.info(f"Component {self.name} shutdown complete")

            except asyncio.TimeoutError:
                self._state = LifecycleState.ERROR
                logger.error(f"Component {self.name} shutdown timed out")
                self._shutdown_event.set()
                raise

            except Exception as e:
                self._state = LifecycleState.ERROR
                self._metrics.error_count += 1
                self._metrics.last_error = str(e)
                self._metrics.last_error_at = datetime.now()
                logger.error(f"Error during {self.name} shutdown: {e}")
                self._shutdown_event.set()
                raise

    async def restart(self) -> None:
        """Restart the component."""
        logger.info(f"Restarting component {self.name}")
        await self.shutdown()
        await self.initialize()
        self._metrics.restart_count += 1

    async def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """Wait for component to shutdown.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if shutdown completed, False if timed out
        """
        try:
            await asyncio.wait_for(self._shutdown_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check.

        Override in subclasses to add component-specific checks.

        Returns:
            Health status dictionary
        """
        return {
            "name": self.name,
            "state": self._state.value,
            "healthy": self._state == LifecycleState.RUNNING,
            "uptime_seconds": self._metrics.uptime_seconds,
            "error_count": self._metrics.error_count,
            "last_error": self._metrics.last_error,
        }

    async def __aenter__(self: T) -> T:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Async context manager exit."""
        await self.shutdown()

    @abstractmethod
    async def _do_initialize(self) -> None:
        """Component-specific initialization logic.

        Override this method in subclasses to implement initialization.
        This is called after dependencies are initialized.
        """

    @abstractmethod
    async def _do_shutdown(self) -> None:
        """Component-specific shutdown logic.

        Override this method in subclasses to implement cleanup.
        This is called before state is set to STOPPED.
        """


@asynccontextmanager
async def managed_component(component: T):
    """Async context manager for component lifecycle.

    Ensures proper initialization and cleanup of a component.

    Args:
        component: Component to manage

    Yields:
        Initialized component

    Example:
        async with managed_component(MyService()) as service:
            result = await service.process()
    """
    await component.initialize()
    try:
        yield component
    finally:
        await component.shutdown()


@asynccontextmanager
async def managed_components(*components: ManagedComponent):
    """Async context manager for multiple components.

    Initializes components in order and shuts them down in reverse order.

    Args:
        *components: Components to manage

    Yields:
        Tuple of initialized components
    """
    initialized: List[ManagedComponent] = []
    try:
        for component in components:
            await component.initialize()
            initialized.append(component)
        yield components
    finally:
        for component in reversed(initialized):
            try:
                await component.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down {component.name}: {e}")


class ComponentRegistry:
    """Registry for managing multiple components.

    Provides centralized lifecycle management for all registered components.
    """

    def __init__(self, name: str = "default") -> None:
        """Initialize the registry.

        Args:
            name: Registry name for logging
        """
        self.name = name
        self._components: Dict[str, ManagedComponent] = {}
        self._lock = asyncio.Lock()
        self._shutdown_order: List[str] = []

    async def register(
        self,
        name: str,
        component: ManagedComponent,
        auto_initialize: bool = False,
    ) -> None:
        """Register a component.

        Args:
            name: Unique name for the component
            component: Component to register
            auto_initialize: Whether to initialize immediately
        """
        async with self._lock:
            if name in self._components:
                raise ValueError(f"Component {name} already registered")

            self._components[name] = component
            self._shutdown_order.append(name)
            logger.debug(f"Registered component {name} in registry {self.name}")

            if auto_initialize:
                await component.initialize()

    async def unregister(self, name: str, shutdown: bool = True) -> None:
        """Unregister a component.

        Args:
            name: Component name
            shutdown: Whether to shutdown before unregistering
        """
        async with self._lock:
            if name not in self._components:
                return

            component = self._components[name]
            if shutdown and component.is_initialized:
                await component.shutdown()

            del self._components[name]
            if name in self._shutdown_order:
                self._shutdown_order.remove(name)

    async def get(self, name: str) -> Optional[ManagedComponent]:
        """Get a registered component."""
        async with self._lock:
            return self._components.get(name)

    async def initialize_all(self) -> None:
        """Initialize all registered components."""
        async with self._lock:
            for name in self._shutdown_order:
                component = self._components[name]
                if not component.is_initialized:
                    await component.initialize()

    async def shutdown_all(self) -> None:
        """Shutdown all registered components in reverse order."""
        async with self._lock:
            for name in reversed(self._shutdown_order):
                component = self._components.get(name)
                if component and component.is_initialized:
                    try:
                        await component.shutdown()
                    except Exception as e:
                        logger.error(f"Error shutting down {name}: {e}")

    async def health_check_all(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        results = {}
        async with self._lock:
            for name, component in self._components.items():
                results[name] = await component.health_check()
        return {
            "registry": self.name,
            "components": results,
            "healthy": all(r.get("healthy", False) for r in results.values()),
        }


class ShutdownCoordinator:
    """Coordinates graceful shutdown across multiple components.

    Handles OS signals and ensures proper cleanup order.
    """

    def __init__(self) -> None:
        """Initialize the shutdown coordinator."""
        self._registries: List[ComponentRegistry] = []
        self._shutdown_requested = asyncio.Event()
        self._shutdown_complete = asyncio.Event()
        self._lock = asyncio.Lock()
        self._signal_handlers_installed = False

    def add_registry(self, registry: ComponentRegistry) -> None:
        """Add a registry to coordinate."""
        self._registries.append(registry)

    def install_signal_handlers(self) -> None:
        """Install OS signal handlers for graceful shutdown."""
        if self._signal_handlers_installed:
            return

        loop = asyncio.get_event_loop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._handle_signal, sig)
            except (NotImplementedError, RuntimeError):
                # Signal handlers not supported (e.g., Windows)
                pass

        self._signal_handlers_installed = True
        logger.debug("Signal handlers installed for graceful shutdown")

    def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle shutdown signal."""
        logger.info(f"Received signal {sig.name}, initiating shutdown")
        self._shutdown_requested.set()

    async def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """Wait for shutdown signal.

        Args:
            timeout: Maximum time to wait

        Returns:
            True if shutdown was requested, False if timed out
        """
        try:
            await asyncio.wait_for(self._shutdown_requested.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Perform coordinated shutdown of all registries."""
        async with self._lock:
            logger.info("Starting coordinated shutdown")

            for registry in reversed(self._registries):
                try:
                    await asyncio.wait_for(
                        registry.shutdown_all(), timeout=timeout / len(self._registries)
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Registry {registry.name} shutdown timed out")
                except Exception as e:
                    logger.error(f"Error shutting down registry {registry.name}: {e}")

            self._shutdown_complete.set()
            logger.info("Coordinated shutdown complete")

    @property
    def shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested.is_set()

    @property
    def shutdown_complete(self) -> bool:
        """Check if shutdown is complete."""
        return self._shutdown_complete.is_set()


# Global shutdown coordinator (can be replaced via DI)
_shutdown_coordinator: Optional[ShutdownCoordinator] = None


def get_shutdown_coordinator() -> ShutdownCoordinator:
    """Get the global shutdown coordinator."""
    global _shutdown_coordinator
    if _shutdown_coordinator is None:
        _shutdown_coordinator = ShutdownCoordinator()
    return _shutdown_coordinator


def set_shutdown_coordinator(coordinator: ShutdownCoordinator) -> None:
    """Set the global shutdown coordinator (for testing)."""
    global _shutdown_coordinator
    _shutdown_coordinator = coordinator
