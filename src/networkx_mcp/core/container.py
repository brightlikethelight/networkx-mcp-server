"""Dependency injection container for NetworkX MCP Server.

This module provides a sophisticated dependency injection system that enables
loose coupling between components and easy testing/mocking.
"""

import asyncio
import inspect
import logging
from collections.abc import Callable
from enum import Enum
from typing import Any, Optional, TypeVar

from .base import Component, ComponentStatus

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Scope(Enum):
    """Dependency scope."""

    SINGLETON = "singleton"  # One instance for entire application
    REQUEST = "request"  # New instance per request
    TRANSIENT = "transient"  # New instance every time


class DependencyDescriptor:
    """Descriptor for a dependency."""

    def __init__(self, interface: type[T], implementation: type[T] | Callable[..., T] | None = None, factory: Callable[..., T] | None = None, scope: Scope = Scope.SINGLETON, name: str | None = None, tags: list[str] | None = None):
        self.interface = interface
        self.implementation = implementation or interface
        self.factory = factory
        self.scope = scope
        self.name = name or interface.__name__
        self.tags = tags or []
        self._instance: T | None = None
        self._lock = asyncio.Lock()

    async def resolve(self, container: "Container") -> T:
        """Resolve the dependency."""
        if self.scope == Scope.SINGLETON:
            async with self._lock:
                if self._instance is None:
                    self._instance = await self._create_instance(container)
                return self._instance
        else:
            return await self._create_instance(container)

    async def _create_instance(self, container: "Container") -> T:
        """Create a new instance."""
        if self.factory:
            # Use factory function
            if asyncio.iscoroutinefunction(self.factory):
                return await self.factory(container)
            else:
                return self.factory(container)
        else:
            # Use constructor injection
            return await container.create_instance(self.implementation)


class Container:
    """Dependency injection container."""

    def __init__(self, name: str = "default") -> None:
        self.name = name
        self._dependencies: dict[type, DependencyDescriptor] = {}
        self._named_dependencies: dict[str, DependencyDescriptor] = {}
        self._request_scope: dict[type, Any] = {}
        self._lock = asyncio.Lock()
        logger.info(f"Container {name} created")

    def register(self, interface: type[T], implementation: type[T] | Callable[..., T] | None = None, factory: Callable[..., T] | None = None, scope: Scope = Scope.SINGLETON, name: str | None = None, tags: list[str] | None = None) -> None:
        """Register a dependency."""
        descriptor = DependencyDescriptor(
            interface=interface,
            implementation=implementation,
            factory=factory,
            scope=scope,
            name=name,
            tags=tags,
        )

        self._dependencies[interface] = descriptor
        if name:
            self._named_dependencies[name] = descriptor

        logger.debug(f"Registered {interface.__name__} with scope {scope.value}")

    def register_singleton(self, interface: type[T], implementation: type[T] | None = None) -> None:
        """Register a singleton dependency."""
        self.register(interface, implementation, scope=Scope.SINGLETON)

    def register_transient(self, interface: type[T], implementation: type[T] | None = None) -> None:
        """Register a transient dependency."""
        self.register(interface, implementation, scope=Scope.TRANSIENT)

    def register_factory(self, interface: type[T], factory: Callable[["Container"], T], scope: Scope = Scope.SINGLETON) -> None:
        """Register a factory function."""
        self.register(interface, factory=factory, scope=scope)

    async def resolve(self, interface: type[T], name: str | None = None) -> T:
        """Resolve a dependency."""
        if name and name in self._named_dependencies:
            descriptor = self._named_dependencies[name]
        elif interface in self._dependencies:
            descriptor = self._dependencies[interface]
        else:
            raise ValueError(f"No registration found for {interface.__name__}")

        # Handle request scope
        if descriptor.scope == Scope.REQUEST:
            if interface in self._request_scope:
                return self._request_scope[interface]
            instance = await descriptor.resolve(self)
            self._request_scope[interface] = instance
            return instance

        return await descriptor.resolve(self)

    async def resolve_all(self, interface: type[T]) -> list[T]:
        """Resolve all implementations of an interface."""
        instances = []
        for descriptor in self._dependencies.values():
            if issubclass(descriptor.implementation, interface):
                instances.append(await descriptor.resolve(self))
        return instances

    async def resolve_by_tag(self, tag: str) -> list[Any]:
        """Resolve all dependencies with a specific tag."""
        instances = []
        for descriptor in self._dependencies.values():
            if tag in descriptor.tags:
                instances.append(await descriptor.resolve(self))
        return instances

    async def create_instance(self, cls: type[T]) -> T:
        """Create an instance with dependency injection."""
        # Get constructor parameters
        sig = inspect.signature(cls.__init__)
        params = {}

        # Skip 'self' parameter
        for param_name, param in list[Any](sig.parameters.items())[1:]:
            if param.annotation != param.empty:
                # Try to resolve the dependency
                try:
                    params[param_name] = await self.resolve(param.annotation)
                except ValueError:
                    # If no dependency registered, skip if has default
                    if param.default == param.empty:
                        raise

        # Create instance
        if asyncio.iscoroutinefunction(cls.__init__):
            instance = cls(**params)
            await instance.__init__(**params)
        else:
            instance = cls(**params)

        # Handle post-construction injection
        await self._inject_properties(instance)

        # Initialize if it's a Component
        if (
            isinstance(instance, Component)
            and instance.status == ComponentStatus.UNINITIALIZED
        ):
            await instance.initialize()

        return instance

    async def _inject_properties(self, instance: Any) -> None:
        """Inject properties marked with @inject."""
        for attr_name in dir(instance):
            attr = getattr(instance, attr_name)
            if hasattr(attr, "_inject_type"):
                injected = await self.resolve(attr._inject_type)
                setattr(instance, attr_name, injected)

    def clear_request_scope(self) -> None:
        """Clear request-scoped instances."""
        self._request_scope.clear()

    async def dispose(self) -> None:
        """Dispose of all singleton instances."""
        for descriptor in self._dependencies.values():
            if descriptor._instance and isinstance(descriptor._instance, Component):
                await descriptor._instance.shutdown()
        logger.info(f"Container {self.name} disposed")


# Decorators for dependency injection
def inject(dependency_type: type[T]) -> property:
    """Property injection decorator."""

    def getter(self) -> T:
        if not hasattr(self, f"_{dependency_type.__name__}"):
            raise RuntimeError(f"Dependency {dependency_type.__name__} not injected")
        return getattr(self, f"_{dependency_type.__name__}")

    def setter(self, value: T) -> None:
        setattr(self, f"_{dependency_type.__name__}", value)

    prop = property(getter, setter)
    prop._inject_type = dependency_type
    return prop


def injectable(cls: type[T]) -> type[T]:
    """Mark a class as injectable."""
    cls._injectable = True
    return cls


def factory(scope: Scope = Scope.SINGLETON) -> Any:
    """Factory method decorator."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        func._factory_scope = scope
        return func

    return decorator


class ServiceLocator:
    """Global service locator for the application."""

    _instance: Optional["ServiceLocator"] = None
    _lock = asyncio.Lock()

    def __init__(self) -> None:
        self.containers: dict[str, Container] = {}
        self.default_container = Container("default")
        self.containers["default"] = self.default_container

    @classmethod
    async def get_instance(cls: Any) -> "ServiceLocator":
        """Get the singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def add_container(self, name: str, container: Container) -> None:
        """Add a named container."""
        self.containers[name] = container

    def get_container(self, name: str = "default") -> Container:
        """Get a container by name."""
        if name not in self.containers:
            raise ValueError(f"Container {name} not found")
        return self.containers[name]

    async def resolve(self, interface: type[T], container_name: str = "default") -> T:
        """Resolve a dependency from a container."""
        container = self.get_container(container_name)
        return await container.resolve(interface)


# Context manager for request scope
class RequestScope:
    """Context manager for request-scoped dependencies."""

    def __init__(self, container: Container) -> None:
        self.container = container

    async def __aenter__(self) -> Any:
        return self

    async def __aexit__(self, exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        self.container.clear_request_scope()


# Bootstrap helper
class Bootstrap:
    """Helper class for bootstrapping the application."""

    def __init__(self, container: Container) -> None:
        self.container = container
        self._modules: list[Callable[[Container], None]] = []

    def add_module(self, module: Callable[[Container], None]) -> "Bootstrap":
        """Add a configuration module."""
        self._modules.append(module)
        return self

    async def bootstrap(self) -> Container:
        """Bootstrap the application."""
        # Run all configuration modules
        for module in self._modules:
            if asyncio.iscoroutinefunction(module):
                await module(self.container)
            else:
                module(self.container)

        logger.info("Application bootstrapped successfully")
        return self.container


# Example configuration module
def configure_core_services(container: Container) -> None:
    """Configure core services."""
    from ..repositories.graph_repository import GraphRepository
    from ..services.algorithm_service import AlgorithmService
    from ..services.graph_service import GraphService

    # Register repositories
    container.register_singleton(GraphRepository)

    # Register services
    container.register_singleton(GraphService)
    container.register_singleton(AlgorithmService)

    logger.debug("Core services configured")
