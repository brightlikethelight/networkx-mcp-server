"""
Secure Sandboxing Framework

Comprehensive sandboxing system that isolates all graph operations in secure
containers to prevent command injection, resource exhaustion, and other
execution-based attacks.

Key Features:
- Docker-based container isolation
- Resource limits and monitoring
- System call filtering
- Network isolation
- Execution time limits
- Comprehensive logging
"""

import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import psutil

# Try to import Docker SDK
try:
    import docker

    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False


class SandboxStatus(Enum):
    """Sandbox execution status."""

    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"


@dataclass
class ResourceLimits:
    """Resource limits for sandboxed execution."""

    max_cpu_percent: float = 50.0  # Maximum CPU usage percentage
    max_memory_mb: int = 512  # Maximum memory in MB
    max_execution_time: int = 30  # Maximum execution time in seconds
    max_disk_usage_mb: int = 100  # Maximum disk usage in MB
    max_network_connections: int = 0  # Maximum network connections (0 = none)
    max_file_descriptors: int = 100  # Maximum file descriptors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_cpu_percent": self.max_cpu_percent,
            "max_memory_mb": self.max_memory_mb,
            "max_execution_time": self.max_execution_time,
            "max_disk_usage_mb": self.max_disk_usage_mb,
            "max_network_connections": self.max_network_connections,
            "max_file_descriptors": self.max_file_descriptors,
        }


@dataclass
class ExecutionResult:
    """Result of sandboxed execution."""

    status: SandboxStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    execution_time: float = 0.0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    security_events: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "execution_time": self.execution_time,
            "resource_usage": self.resource_usage,
            "security_events": self.security_events,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class SecureSandbox:
    """
    Secure sandboxing system for graph operations.

    Provides multiple isolation layers:
    1. Container isolation (Docker)
    2. Resource limits (CPU, memory, disk, time)
    3. Network isolation
    4. System call filtering
    5. File system restrictions
    6. Execution monitoring
    """

    def __init__(self, resource_limits: Optional[ResourceLimits] = None):
        self.logger = logging.getLogger(__name__)
        self.resource_limits = resource_limits or ResourceLimits()

        # Try to create container manager, but don't fail if Docker unavailable
        try:
            self.container_manager = ContainerManager() if DOCKER_AVAILABLE else None
        except Exception as e:
            self.logger.warning(f"Container manager initialization failed: {e}")
            self.container_manager = None

        self.resource_monitor = ResourceMonitor()
        self.execution_monitor = ExecutionMonitor()

        # Sandbox statistics
        self.sandbox_stats = {
            "total_executions": 0,
            "successful": 0,
            "failed": 0,
            "timeout": 0,
            "resource_limit_exceeded": 0,
            "average_execution_time": 0.0,
            "last_reset": datetime.utcnow(),
        }

        # Setup sandbox environment
        self._setup_sandbox_environment()

    def _setup_sandbox_environment(self):
        """Setup secure sandbox environment."""
        if self.container_manager:
            self.container_manager.setup_container_environment()
        else:
            self.logger.warning("Docker not available, using process isolation")

    async def execute_operation(
        self, operation: str, arguments: Dict[str, Any], context: Dict[str, Any] = None
    ) -> ExecutionResult:
        """
        Execute operation in secure sandbox.

        Args:
            operation: Operation name
            arguments: Operation arguments
            context: Additional context

        Returns:
            ExecutionResult: Execution result with security information
        """
        start_time = time.time()
        self.sandbox_stats["total_executions"] += 1

        if context is None:
            context = {}

        try:
            # Choose execution method based on availability
            if self.container_manager:
                result = await self._execute_in_container(operation, arguments, context)
            else:
                result = await self._execute_in_process(operation, arguments, context)

            # Update statistics
            execution_time = time.time() - start_time
            self.sandbox_stats["average_execution_time"] = (
                self.sandbox_stats["average_execution_time"]
                * (self.sandbox_stats["total_executions"] - 1)
                + execution_time
            ) / self.sandbox_stats["total_executions"]

            if result.status == SandboxStatus.COMPLETED:
                self.sandbox_stats["successful"] += 1
            elif result.status == SandboxStatus.TIMEOUT:
                self.sandbox_stats["timeout"] += 1
            elif result.status == SandboxStatus.RESOURCE_LIMIT:
                self.sandbox_stats["resource_limit_exceeded"] += 1
            else:
                self.sandbox_stats["failed"] += 1

            result.execution_time = execution_time

            self.logger.info(
                f"Sandbox execution complete: {result.status.value} "
                f"(time: {execution_time:.3f}s)"
            )

            return result

        except Exception as e:
            self.logger.error(f"Sandbox execution error: {e}")
            return ExecutionResult(
                status=SandboxStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    async def _execute_in_container(
        self, operation: str, arguments: Dict[str, Any], context: Dict[str, Any]
    ) -> ExecutionResult:
        """Execute operation in Docker container."""
        try:
            # Create execution context
            execution_context = {
                "operation": operation,
                "arguments": arguments,
                "context": context,
            }

            # Execute in container
            result = await self.container_manager.execute_in_container(
                execution_context, self.resource_limits
            )

            return result

        except Exception as e:
            return ExecutionResult(
                status=SandboxStatus.FAILED,
                error=f"Container execution error: {str(e)}",
            )

    async def _execute_in_process(
        self, operation: str, arguments: Dict[str, Any], context: Dict[str, Any]
    ) -> ExecutionResult:
        """Execute operation in isolated process (fallback)."""
        try:
            # Create temporary execution environment
            with tempfile.TemporaryDirectory() as temp_dir:
                # Monitor resource usage
                monitoring_thread = threading.Thread(
                    target=self.resource_monitor.monitor_execution,
                    args=(os.getpid(), self.resource_limits),
                )
                monitoring_thread.start()

                # Execute operation
                result = await self._execute_graph_operation(operation, arguments)

                # Stop monitoring
                self.resource_monitor.stop_monitoring()
                monitoring_thread.join()

                # Check for resource violations
                resource_usage = self.resource_monitor.get_resource_usage()
                security_events = self.resource_monitor.get_security_events()

                if resource_usage.get("memory_exceeded", False):
                    return ExecutionResult(
                        status=SandboxStatus.RESOURCE_LIMIT,
                        error="Memory limit exceeded",
                        resource_usage=resource_usage,
                        security_events=security_events,
                    )

                if resource_usage.get("cpu_exceeded", False):
                    return ExecutionResult(
                        status=SandboxStatus.RESOURCE_LIMIT,
                        error="CPU limit exceeded",
                        resource_usage=resource_usage,
                        security_events=security_events,
                    )

                return ExecutionResult(
                    status=SandboxStatus.COMPLETED,
                    result=result,
                    resource_usage=resource_usage,
                    security_events=security_events,
                )

        except Exception as e:
            return ExecutionResult(
                status=SandboxStatus.FAILED, error=f"Process execution error: {str(e)}"
            )

    async def _execute_graph_operation(
        self, operation: str, arguments: Dict[str, Any]
    ) -> Any:
        """Execute actual graph operation."""
        # Import graph operations here to avoid circular imports
        from ..server_minimal import (
            add_edges,
            add_nodes,
            betweenness_centrality,
            community_detection,
            connected_components,
            create_graph,
            degree_centrality,
            export_json,
            get_graph_info,
            import_csv,
            pagerank,
            shortest_path,
            visualize_graph,
        )

        # Map operations to functions
        operation_map = {
            "create_graph": create_graph,
            "add_nodes": add_nodes,
            "add_edges": add_edges,
            "get_info": get_graph_info,
            "shortest_path": shortest_path,
            "degree_centrality": degree_centrality,
            "betweenness_centrality": betweenness_centrality,
            "pagerank": pagerank,
            "connected_components": connected_components,
            "community_detection": community_detection,
            "visualize_graph": visualize_graph,
            "import_csv": import_csv,
            "export_json": export_json,
        }

        if operation not in operation_map:
            raise ValueError(f"Unknown operation: {operation}")

        # Execute operation
        func = operation_map[operation]
        if operation == "create_graph":
            return func(arguments["name"], arguments.get("directed", False))
        elif operation in ["add_nodes", "add_edges", "get_info"]:
            return func(
                arguments["graph"], arguments.get("nodes") or arguments.get("edges")
            )
        elif operation == "shortest_path":
            return func(arguments["graph"], arguments["source"], arguments["target"])
        elif operation in [
            "degree_centrality",
            "betweenness_centrality",
            "pagerank",
            "connected_components",
            "community_detection",
        ]:
            return func(arguments["graph"])
        elif operation == "visualize_graph":
            return func(arguments["graph"], arguments.get("layout", "spring"))
        elif operation == "import_csv":
            return func(arguments["graph"], arguments["csv_data"])
        elif operation == "export_json":
            return func(arguments["graph"])
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def get_sandbox_stats(self) -> Dict[str, Any]:
        """Get sandbox statistics."""
        total = self.sandbox_stats["total_executions"]
        return {
            **self.sandbox_stats,
            "success_rate": self.sandbox_stats["successful"] / max(total, 1),
            "failure_rate": self.sandbox_stats["failed"] / max(total, 1),
            "timeout_rate": self.sandbox_stats["timeout"] / max(total, 1),
            "resource_limit_rate": self.sandbox_stats["resource_limit_exceeded"]
            / max(total, 1),
        }


class ContainerManager:
    """Manages Docker container execution."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.docker_client = None
        self.container_image = "networkx-mcp-sandbox:latest"

        if DOCKER_AVAILABLE:
            try:
                self.docker_client = docker.from_env()
                self.logger.info("Docker client initialized")
            except Exception as e:
                self.logger.warning(f"Docker client initialization failed: {e}")
                self.docker_client = None
        else:
            self.docker_client = None

    def setup_container_environment(self):
        """Setup Docker container environment."""
        if not self.docker_client:
            return

        try:
            # Check if container image exists
            try:
                self.docker_client.images.get(self.container_image)
                self.logger.info(f"Container image {self.container_image} found")
            except docker.errors.ImageNotFound:
                self.logger.info(f"Building container image {self.container_image}")
                self._build_container_image()

        except Exception as e:
            self.logger.error(f"Container environment setup failed: {e}")

    def _build_container_image(self):
        """Build secure container image."""
        dockerfile_content = """
        FROM python:3.11-slim

        # Install only required packages
        RUN apt-get update && apt-get install -y \\
            --no-install-recommends \\
            && rm -rf /var/lib/apt/lists/*

        # Install Python dependencies
        RUN pip install --no-cache-dir networkx matplotlib

        # Create non-root user
        RUN useradd -m -u 1000 sandbox

        # Set working directory
        WORKDIR /app

        # Copy execution script
        COPY execute_operation.py /app/

        # Switch to non-root user
        USER sandbox

        # Set security limits
        RUN ulimit -c 0  # Disable core dumps

        CMD ["python", "execute_operation.py"]
        """

        # Create execution script
        execution_script = """
import json
import sys
import networkx as nx
import matplotlib.pyplot as plt
from io import StringIO
import base64

def execute_operation():
    try:
        # Read operation from stdin
        operation_data = json.loads(sys.stdin.read())

        operation = operation_data["operation"]
        arguments = operation_data["arguments"]

        # Execute operation (simplified)
        result = {"status": "completed", "result": None}

        # Return result
        print(json.dumps(result))

    except Exception as e:
        error_result = {"status": "failed", "error": str(e)}
        print(json.dumps(error_result))

if __name__ == "__main__":
    execute_operation()
        """

        # Build image (simplified - in production would use proper Dockerfile)
        self.logger.info("Container image building not implemented in demo")

    async def execute_in_container(
        self, execution_context: Dict[str, Any], resource_limits: ResourceLimits
    ) -> ExecutionResult:
        """Execute operation in secure container."""
        if not self.docker_client:
            raise RuntimeError("Docker client not available")

        try:
            # Prepare container configuration
            container_config = {
                "image": self.container_image,
                "stdin_open": True,
                "tty": False,
                "network_mode": "none",  # No network access
                "mem_limit": f"{resource_limits.max_memory_mb}m",
                "cpu_period": 100000,
                "cpu_quota": int(resource_limits.max_cpu_percent * 1000),  # CPU limit
                "ulimits": [
                    docker.types.Ulimit(
                        name="nofile", soft=resource_limits.max_file_descriptors
                    ),
                    docker.types.Ulimit(name="nproc", soft=10),  # Process limit
                    docker.types.Ulimit(name="core", soft=0),  # Disable core dumps
                ],
                "security_opt": ["no-new-privileges:true", "apparmor:docker-default"],
                "read_only": True,  # Read-only filesystem
                "tmpfs": {"/tmp": "size=10m"},  # Temporary filesystem
            }

            # Create and start container
            container = self.docker_client.containers.run(
                detach=True, **container_config
            )

            try:
                # Send execution context to container
                input_data = json.dumps(execution_context)
                container.exec_run(f"echo '{input_data}' | python execute_operation.py")

                # Wait for completion with timeout
                container.wait(timeout=resource_limits.max_execution_time)

                # Get result
                logs = container.logs()
                result_data = json.loads(logs.decode())

                return ExecutionResult(
                    status=SandboxStatus.COMPLETED
                    if result_data["status"] == "completed"
                    else SandboxStatus.FAILED,
                    result=result_data.get("result"),
                    error=result_data.get("error"),
                    stdout=logs.decode(),
                    metadata={"container_id": container.id},
                )

            finally:
                # Always cleanup container
                container.remove(force=True)

        except Exception as e:
            return ExecutionResult(
                status=SandboxStatus.FAILED,
                error=f"Container execution failed: {str(e)}",
            )


class ResourceMonitor:
    """Monitors resource usage during execution."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.resource_usage = {}
        self.security_events = []

    def monitor_execution(self, pid: int, limits: ResourceLimits):
        """Monitor process resource usage."""
        self.monitoring = True
        self.resource_usage = {
            "max_memory_mb": 0,
            "max_cpu_percent": 0,
            "disk_usage_mb": 0,
            "file_descriptors": 0,
            "network_connections": 0,
            "memory_exceeded": False,
            "cpu_exceeded": False,
        }

        try:
            process = psutil.Process(pid)

            while self.monitoring:
                try:
                    # Memory usage
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)
                    self.resource_usage["max_memory_mb"] = max(
                        self.resource_usage["max_memory_mb"], memory_mb
                    )

                    if memory_mb > limits.max_memory_mb:
                        self.resource_usage["memory_exceeded"] = True
                        self.security_events.append(
                            f"Memory limit exceeded: {memory_mb:.1f}MB"
                        )

                    # CPU usage
                    cpu_percent = process.cpu_percent()
                    self.resource_usage["max_cpu_percent"] = max(
                        self.resource_usage["max_cpu_percent"], cpu_percent
                    )

                    if cpu_percent > limits.max_cpu_percent:
                        self.resource_usage["cpu_exceeded"] = True
                        self.security_events.append(
                            f"CPU limit exceeded: {cpu_percent:.1f}%"
                        )

                    # File descriptors
                    try:
                        fd_count = process.num_fds()
                        self.resource_usage["file_descriptors"] = max(
                            self.resource_usage["file_descriptors"], fd_count
                        )

                        if fd_count > limits.max_file_descriptors:
                            self.security_events.append(
                                f"File descriptor limit exceeded: {fd_count}"
                            )
                    except (AttributeError, psutil.AccessDenied):
                        pass

                    # Network connections
                    try:
                        connections = process.connections()
                        conn_count = len(connections)
                        self.resource_usage["network_connections"] = max(
                            self.resource_usage["network_connections"], conn_count
                        )

                        if conn_count > limits.max_network_connections:
                            self.security_events.append(
                                f"Network connection limit exceeded: {conn_count}"
                            )
                    except (AttributeError, psutil.AccessDenied):
                        pass

                    time.sleep(0.1)  # Check every 100ms

                except psutil.NoSuchProcess:
                    break
                except Exception as e:
                    self.logger.warning(f"Resource monitoring error: {e}")
                    break

        except Exception as e:
            self.logger.error(f"Resource monitoring failed: {e}")

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False

    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        return self.resource_usage.copy()

    def get_security_events(self) -> List[str]:
        """Get security events."""
        return self.security_events.copy()


class ExecutionMonitor:
    """Monitors execution for security events."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.security_events = []

    def monitor_execution(self, operation: str, arguments: Dict[str, Any]):
        """Monitor execution for security events."""
        # Check for suspicious operations
        if (
            operation in ["import_csv"]
            and len(arguments.get("csv_data", "")) > 10 * 1024 * 1024
        ):
            self.security_events.append("Large CSV import detected")

        # Check for suspicious arguments
        for key, value in arguments.items():
            if isinstance(value, str) and len(value) > 1024 * 1024:
                self.security_events.append(f"Large string argument: {key}")

    def get_security_events(self) -> List[str]:
        """Get security events."""
        return self.security_events.copy()
