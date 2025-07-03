"""Health check service for monitoring system status and readiness."""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import psutil

from ..core.base import Component

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status enumeration."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    details: dict[str, Any]
    duration_ms: float
    timestamp: float
    error: str | None = None


@dataclass
class SystemHealth:
    """Overall system health information."""

    status: HealthStatus
    checks: list[HealthCheckResult]
    timestamp: float
    version: str
    uptime_seconds: float


class HealthCheck:
    """Individual health check definition."""

    def __init__(
        self,
        name: str,
        check_func: Callable[[], bool],
        timeout: float = 5.0,
        interval: float = 30.0,
        critical: bool = True,
    ):
        self.name = name
        self.check_func = check_func
        self.timeout = timeout
        self.interval = interval
        self.critical = critical
        self.last_check: HealthCheckResult | None = None

    async def execute(self) -> HealthCheckResult:
        """Execute the health check."""
        start_time = time.time()
        timestamp = start_time

        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                asyncio.create_task(self._run_check()), timeout=self.timeout
            )

            duration_ms = (time.time() - start_time) * 1000

            if result:
                status = HealthStatus.HEALTHY
                error = None
            else:
                status = HealthStatus.UNHEALTHY
                error = "Check returned False"

            check_result = HealthCheckResult(
                name=self.name,
                status=status,
                details={"result": result},
                duration_ms=duration_ms,
                timestamp=timestamp,
                error=error,
            )

        except TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            check_result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                details={"timeout": self.timeout},
                duration_ms=duration_ms,
                timestamp=timestamp,
                error=f"Health check timed out after {self.timeout}s",
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            check_result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                details={"exception": str(e)},
                duration_ms=duration_ms,
                timestamp=timestamp,
                error=str(e),
            )

        self.last_check = check_result
        return check_result

    async def _run_check(self) -> bool:
        """Run the actual check function."""
        if asyncio.iscoroutinefunction(self.check_func):
            return await self.check_func()
        else:
            # Run in thread pool for sync functions
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.check_func)


class HealthCheckService(Component):
    """Service for managing and executing health checks."""

    def __init__(self, version: str = "1.0.0"):
        super().__init__("health_check_service")
        self.version = version
        self.start_time = time.time()
        self.checks: dict[str, HealthCheck] = {}
        self.background_task: asyncio.Task | None = None

        # Register default system health checks
        self._register_default_checks()

    def _register_default_checks(self):
        """Register default system health checks."""

        def memory_check() -> bool:
            """Check if memory usage is acceptable."""
            memory = psutil.virtual_memory()
            return memory.percent < 90.0

        def disk_check() -> bool:
            """Check if disk usage is acceptable."""
            disk = psutil.disk_usage("/")
            return (disk.used / disk.total) < 0.95

        def cpu_check() -> bool:
            """Check if CPU usage is acceptable."""
            cpu_percent = psutil.cpu_percent(interval=1)
            return cpu_percent < 95.0

        # Register checks
        self.register_check("memory", memory_check, critical=True)
        self.register_check("disk", disk_check, critical=True)
        self.register_check("cpu", cpu_check, critical=False)

    def register_check(
        self,
        name: str,
        check_func: Callable[[], bool],
        timeout: float = 5.0,
        interval: float = 30.0,
        critical: bool = True,
    ) -> None:
        """Register a new health check."""
        health_check = HealthCheck(
            name=name,
            check_func=check_func,
            timeout=timeout,
            interval=interval,
            critical=critical,
        )
        self.checks[name] = health_check
        logger.info(f"Registered health check: {name}")

    def unregister_check(self, name: str) -> None:
        """Unregister a health check."""
        if name in self.checks:
            del self.checks[name]
            logger.info(f"Unregistered health check: {name}")

    async def run_check(self, name: str) -> HealthCheckResult | None:
        """Run a specific health check."""
        if name not in self.checks:
            logger.warning(f"Health check not found: {name}")
            return None

        return await self.checks[name].execute()

    async def run_all_checks(self) -> list[HealthCheckResult]:
        """Run all registered health checks."""
        results = []

        # Run checks concurrently
        tasks = [check.execute() for check in self.checks.values()]

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle any exceptions
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    check_name = list(self.checks.keys())[i]
                    valid_results.append(
                        HealthCheckResult(
                            name=check_name,
                            status=HealthStatus.UNHEALTHY,
                            details={"exception": str(result)},
                            duration_ms=0.0,
                            timestamp=time.time(),
                            error=str(result),
                        )
                    )
                else:
                    valid_results.append(result)

            results = valid_results

        return results

    async def get_system_health(self) -> SystemHealth:
        """Get overall system health status."""
        check_results = await self.run_all_checks()

        # Determine overall status
        overall_status = HealthStatus.HEALTHY

        for result in check_results:
            check = self.checks.get(result.name)
            if not check:
                continue

            if result.status == HealthStatus.UNHEALTHY:
                if check.critical:
                    overall_status = HealthStatus.UNHEALTHY
                    break
                else:
                    if overall_status == HealthStatus.HEALTHY:
                        overall_status = HealthStatus.DEGRADED

        uptime_seconds = time.time() - self.start_time

        return SystemHealth(
            status=overall_status,
            checks=check_results,
            timestamp=time.time(),
            version=self.version,
            uptime_seconds=uptime_seconds,
        )

    async def start_background_monitoring(self) -> None:
        """Start background health monitoring."""
        if self.background_task:
            logger.warning("Background monitoring already running")
            return

        self.background_task = asyncio.create_task(self._background_monitor())
        logger.info("Started background health monitoring")

    async def stop_background_monitoring(self) -> None:
        """Stop background health monitoring."""
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
            self.background_task = None
            logger.info("Stopped background health monitoring")

    async def _background_monitor(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                system_health = await self.get_system_health()

                # Log health status
                if system_health.status == HealthStatus.UNHEALTHY:
                    logger.error(f"System health: {system_health.status.value}")
                    for check in system_health.checks:
                        if check.status == HealthStatus.UNHEALTHY:
                            logger.error(
                                f"Health check failed: {check.name} - {check.error}"
                            )
                elif system_health.status == HealthStatus.DEGRADED:
                    logger.warning(f"System health: {system_health.status.value}")

                # Wait before next check (use minimum interval)
                min_interval = (
                    min(check.interval for check in self.checks.values())
                    if self.checks
                    else 30.0
                )
                await asyncio.sleep(min_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background health monitoring: {e}")
                await asyncio.sleep(10)  # Wait before retry

    async def initialize(self) -> None:
        """Initialize the health check service."""
        await super().initialize()
        await self.start_background_monitoring()

    async def cleanup(self) -> None:
        """Cleanup the health check service."""
        await self.stop_background_monitoring()
        await super().cleanup()
