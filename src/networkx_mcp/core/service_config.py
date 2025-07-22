"""Service configuration module for NetworkX MCP Server.

This module provides dependency injection configuration for the modular
service architecture, setting up all services, repositories, validators,
and other components.
"""

import logging
from typing import Any

from ..caching.cache_service import CacheService, MemoryCache, RedisCache
from ..events.graph_events import (
    GraphEventPublisher,
    LoggingEventListener,
    MetricsEventListener,
)
from ..repositories.graph_repository import (
    FileStorageBackend,
    GraphRepository,
    RedisStorageBackend,
)
from ..services.algorithm_service import AlgorithmService

# Import all services and components
from ..services.graph_service import GraphService
from ..validators.algorithm_validator import AlgorithmValidator
from ..validators.graph_validator import GraphValidator
from .config import AppConfig, get_config
from .container import Bootstrap, Container

logger = logging.getLogger(__name__)


def configure_core_services(container: Container) -> None:
    """Configure core services in the DI container."""

    # Configuration
    def config_factory(c: Container) -> AppConfig:
        return get_config()

    container.register_factory(AppConfig, config_factory)

    logger.debug("Core services configuration completed")


def configure_storage_backends(container: Container) -> None:
    """Configure storage backends."""

    def storage_backend_factory(c: Container) -> object | None:
        config = c.resolve(AppConfig)
        if hasattr(config, "redis") and config.redis.url:
            return RedisStorageBackend(config.redis.url)
        else:
            return FileStorageBackend()

    # Register storage backend factory
    container.register_factory(object, storage_backend_factory)

    logger.debug("Storage backends configuration completed")


def configure_caching(container: Container) -> None:
    """Configure caching services."""

    def cache_backend_factory(c: Container) -> object | None:
        config = c.resolve(AppConfig)
        if (
            hasattr(config, "redis")
            and config.redis.url
            and config.performance.enable_caching
        ):
            return RedisCache(config.redis.url)
        elif config.performance.enable_caching:
            return MemoryCache(
                max_size=config.performance.cache_size_mb * 100,  # Rough conversion
                default_ttl=config.performance.cache_ttl,
            )
        else:
            return None

    # Register cache service
    def cache_service_factory(c: Container) -> CacheService | None:
        backend = cache_backend_factory(c)
        if backend:
            return CacheService(backend)
        return None

    container.register_factory(CacheService, cache_service_factory)

    logger.debug("Caching configuration completed")


def configure_repositories(container: Container) -> None:
    """Configure repository services."""

    def graph_repository_factory(c: Container) -> GraphRepository:
        storage_backend = c.resolve(object)  # Get storage backend
        return GraphRepository(storage_backend)

    container.register_factory(GraphRepository, graph_repository_factory)

    logger.debug("Repositories configuration completed")


def configure_validators(container: Container) -> None:
    """Configure validation services."""

    def graph_validator_factory(c: Container) -> GraphValidator:
        config = c.resolve(AppConfig)
        validator_config = {
            "max_nodes": config.performance.max_nodes,
            "max_edges": config.performance.max_edges,
        }
        return GraphValidator(validator_config)

    def algorithm_validator_factory(c: Container) -> AlgorithmValidator:
        config = c.resolve(AppConfig)
        validator_config = {
            "max_computation_nodes": config.performance.max_nodes,
            "max_computation_edges": config.performance.max_edges,
            "timeout_threshold_nodes": config.performance.timeout_seconds,
        }
        return AlgorithmValidator(validator_config)

    container.register_factory(GraphValidator, graph_validator_factory)
    container.register_factory(AlgorithmValidator, algorithm_validator_factory)

    logger.debug("Validators configuration completed")


def configure_events(container: Container) -> None:
    """Configure event system."""

    # Register event publisher as singleton
    container.register_singleton(GraphEventPublisher)

    # Register event listeners
    def logging_listener_factory(c: Container) -> LoggingEventListener:
        return LoggingEventListener()

    def metrics_listener_factory(c: Container) -> MetricsEventListener:
        return MetricsEventListener()

    container.register_factory(LoggingEventListener, logging_listener_factory)
    container.register_factory(MetricsEventListener, metrics_listener_factory)

    logger.debug("Events configuration completed")


def configure_business_services(container: Container) -> None:
    """Configure business services."""

    def graph_service_factory(c: Container) -> GraphService:
        repository = c.resolve(GraphRepository)
        validator = c.resolve(GraphValidator)
        event_publisher = c.resolve(GraphEventPublisher)
        cache_service = (
            c.resolve(CacheService) if CacheService in c._dependencies else None
        )

        return GraphService(
            repository=repository,
            validator=validator,
            event_publisher=event_publisher,
            cache_service=cache_service,
        )

    def algorithm_service_factory(c: Container) -> AlgorithmService:
        graph_service = c.resolve(GraphService)
        validator = c.resolve(AlgorithmValidator)
        cache_service = (
            c.resolve(CacheService) if CacheService in c._dependencies else None
        )

        return AlgorithmService(
            graph_service=graph_service,
            validator=validator,
            cache_service=cache_service,
        )

    container.register_factory(GraphService, graph_service_factory)
    container.register_factory(AlgorithmService, algorithm_service_factory)

    logger.debug("Business services configuration completed")


async def setup_event_listeners(container: Container) -> None:
    """Setup event listeners after services are initialized."""
    try:
        publisher = await container.resolve(GraphEventPublisher)

        # Initialize and setup listeners
        logging_listener = await container.resolve(LoggingEventListener)
        await logging_listener.initialize(publisher)

        metrics_listener = await container.resolve(MetricsEventListener)
        await metrics_listener.initialize(publisher)

        logger.info("Event listeners configured successfully")
    except Exception as e:
        logger.error(f"Failed to setup event listeners: {e}")
        raise


async def create_service_container() -> Container:
    """Create and configure the service container."""
    container = Container("NetworkXMCPServices")

    # Create bootstrap configuration
    bootstrap = Bootstrap(container)
    bootstrap.add_module(configure_core_services)
    bootstrap.add_module(configure_storage_backends)
    bootstrap.add_module(configure_caching)
    bootstrap.add_module(configure_repositories)
    bootstrap.add_module(configure_validators)
    bootstrap.add_module(configure_events)
    bootstrap.add_module(configure_business_services)

    # Bootstrap the container
    await bootstrap.bootstrap()

    # Setup event listeners
    await setup_event_listeners(container)

    logger.info("Service container created and configured successfully")
    return container


async def initialize_services(container: Container) -> None:
    """Initialize all services in dependency order."""
    try:
        # Initialize in dependency order
        logger.info("Initializing services...")

        # Core components first
        await container.resolve(GraphEventPublisher)

        # Storage and caching
        await container.resolve(GraphRepository)
        cache_service = await container.resolve(CacheService)
        if cache_service:
            await cache_service.initialize()

        # Validators
        await container.resolve(GraphValidator)
        await container.resolve(AlgorithmValidator)

        # Business services
        await container.resolve(GraphService)
        await container.resolve(AlgorithmService)

        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


async def shutdown_services(container: Container) -> None:
    """Shutdown all services gracefully."""
    try:
        logger.info("Shutting down services...")

        # Shutdown in reverse dependency order
        services_to_shutdown = [
            AlgorithmService,
            GraphService,
            GraphRepository,
            CacheService,
            GraphEventPublisher,
        ]

        for service_type in services_to_shutdown:
            try:
                if service_type in container._dependencies:
                    service = await container.resolve(service_type)
                    if hasattr(service, "shutdown"):
                        await service.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down {service_type.__name__}: {e}")

        # Dispose container
        await container.dispose()

        logger.info("All services shutdown successfully")
    except Exception as e:
        logger.error(f"Failed to shutdown services: {e}")
        raise


class ServiceManager:
    """Manager for the service lifecycle."""

    def __init__(self) -> None:
        self.container: Container | None = None
        self._initialized = False

    async def start(self) -> Container:
        """Start all services."""
        if self._initialized:
            raise RuntimeError("Services already initialized")

        try:
            self.container = await create_service_container()
            await initialize_services(self.container)
            self._initialized = True

            logger.info("Service manager started successfully")
            return self.container
        except Exception as e:
            logger.error(f"Failed to start service manager: {e}")
            if self.container:
                await self.container.dispose()
            raise

    async def stop(self) -> None:
        """Stop all services."""
        if not self._initialized or not self.container:
            return

        try:
            await shutdown_services(self.container)
            self.container = None
            self._initialized = False

            logger.info("Service manager stopped successfully")
        except Exception as e:
            logger.error(f"Failed to stop service manager: {e}")
            raise

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on all services."""
        if not self._initialized or not self.container:
            return {"healthy": False, "error": "Services not initialized"}

        try:
            # Check core services
            graph_service = await self.container.resolve(GraphService)
            algorithm_service = await self.container.resolve(AlgorithmService)

            graph_health = await graph_service.health_check()
            algorithm_health = await algorithm_service.health_check()

            overall_healthy = graph_health.get(
                "healthy", False
            ) and algorithm_health.get("healthy", False)

            return {
                "healthy": overall_healthy,
                "services": {
                    "graph_service": graph_health,
                    "algorithm_service": algorithm_health,
                },
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}
