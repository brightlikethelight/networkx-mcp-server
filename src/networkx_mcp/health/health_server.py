#!/usr/bin/env python3
"""Health check server for Kubernetes deployments.

Provides HTTP endpoints for liveness and readiness probes.
Based on production testing limits and real performance data.
"""

import asyncio
import json
import os
import psutil
import time
from typing import Dict, Any, Optional
from aiohttp import web, web_runner
import aiohttp.web

from ..config.production import production_config
from ..logging import get_logger

logger = get_logger(__name__)


class HealthServer:
    """HTTP server for Kubernetes health checks."""
    
    def __init__(self, mcp_server=None, port: int = None):
        self.mcp_server = mcp_server
        self.port = port or production_config.HEALTH_CHECK_PORT
        self.app = web.Application()
        self.runner: Optional[web_runner.AppRunner] = None
        self.site: Optional[web_runner.TCPSite] = None
        self.start_time = time.time()
        
        # Setup routes
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/ready', self.readiness_check)
        self.app.router.add_get('/startup', self.startup_check)
        self.app.router.add_get('/metrics', self.basic_metrics)
        
        logger.info(f"Health server configured on port {self.port}")
    
    async def start(self):
        """Start the health check HTTP server."""
        try:
            self.runner = web_runner.AppRunner(self.app)
            await self.runner.setup()
            
            self.site = web_runner.TCPSite(
                self.runner,
                host='0.0.0.0',  # Listen on all interfaces for K8s
                port=self.port
            )
            await self.site.start()
            
            logger.info(f"Health server started on 0.0.0.0:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start health server: {e}")
            raise
    
    async def stop(self):
        """Stop the health check HTTP server."""
        try:
            if self.runner:
                await self.runner.cleanup()
                logger.info("Health server stopped")
        except Exception as e:
            logger.error(f"Error stopping health server: {e}")
    
    async def health_check(self, request) -> web.Response:
        """
        Liveness probe - is the server process running and responding?
        
        Returns 200 if the process is alive, regardless of load or readiness.
        Kubernetes will restart the pod if this fails.
        """
        try:
            uptime = time.time() - self.start_time
            
            health_data = {
                "status": "healthy",
                "timestamp": time.time(),
                "uptime_seconds": round(uptime, 2),
                "version": production_config.SERVER_VERSION,
                "server_name": production_config.SERVER_NAME
            }
            
            # Add detailed checks if requested
            include_details = request.query.get('details', '').lower() == 'true'
            if include_details:
                health_data.update({
                    "memory_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 1),
                    "cpu_percent": psutil.Process().cpu_percent(),
                    "process_id": os.getpid(),
                    "environment": os.getenv("ENVIRONMENT", "unknown")
                })
            
            logger.debug("Health check passed")
            return web.json_response(health_data)
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return web.json_response(
                {
                    "status": "unhealthy",
                    "timestamp": time.time(),
                    "error": str(e)
                },
                status=500
            )
    
    async def readiness_check(self, request) -> web.Response:
        """
        Readiness probe - is the server ready to handle requests?
        
        Returns 200 only if the server can safely receive traffic.
        Kubernetes will remove the pod from load balancing if this fails.
        """
        try:
            checks = await self._perform_readiness_checks()
            
            all_ready = all(checks.values())
            status_code = 200 if all_ready else 503
            
            response_data = {
                "ready": all_ready,
                "timestamp": time.time(),
                "checks": checks
            }
            
            if all_ready:
                logger.debug("Readiness check passed")
            else:
                logger.warning(f"Readiness check failed: {checks}")
            
            return web.json_response(response_data, status=status_code)
            
        except Exception as e:
            logger.error(f"Readiness check error: {e}")
            return web.json_response(
                {
                    "ready": False,
                    "timestamp": time.time(),
                    "error": str(e)
                },
                status=503
            )
    
    async def startup_check(self, request) -> web.Response:
        """
        Startup probe - has the server finished initializing?
        
        Used for slow-starting containers. Kubernetes won't send traffic
        until this passes, and won't run liveness/readiness until then.
        """
        try:
            uptime = time.time() - self.start_time
            
            # Consider started if running for more than 10 seconds
            # and basic components are available
            is_started = (
                uptime > 10 and
                self.mcp_server is not None
            )
            
            status_code = 200 if is_started else 503
            
            response_data = {
                "started": is_started,
                "timestamp": time.time(),
                "uptime_seconds": round(uptime, 2),
                "message": "Server fully initialized" if is_started else "Server still starting"
            }
            
            return web.json_response(response_data, status=status_code)
            
        except Exception as e:
            logger.error(f"Startup check error: {e}")
            return web.json_response(
                {
                    "started": False,
                    "timestamp": time.time(),
                    "error": str(e)
                },
                status=503
            )
    
    async def basic_metrics(self, request) -> web.Response:
        """
        Basic metrics endpoint for monitoring.
        
        Provides essential metrics without requiring full Prometheus setup.
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            metrics = {
                "timestamp": time.time(),
                "uptime_seconds": round(time.time() - self.start_time, 2),
                "memory": {
                    "rss_mb": round(memory_info.rss / 1024 / 1024, 1),
                    "vms_mb": round(memory_info.vms / 1024 / 1024, 1),
                    "percent": round(process.memory_percent(), 2)
                },
                "cpu": {
                    "percent": round(process.cpu_percent(), 2)
                },
                "connections": {
                    "active": self._get_active_connections(),
                    "max_allowed": production_config.MAX_CONCURRENT_CONNECTIONS
                },
                "limits": {
                    "max_memory_mb": production_config.MAX_MEMORY_MB,
                    "max_graph_nodes": production_config.MAX_GRAPH_SIZE_NODES,
                    "request_timeout": production_config.REQUEST_TIMEOUT
                }
            }
            
            return web.json_response(metrics)
            
        except Exception as e:
            logger.error(f"Metrics endpoint error: {e}")
            return web.json_response(
                {"error": str(e), "timestamp": time.time()},
                status=500
            )
    
    async def _perform_readiness_checks(self) -> Dict[str, bool]:
        """Perform all readiness checks and return results."""
        checks = {}
        
        # 1. Memory usage check (based on testing: 450MB for 50K nodes)
        checks["memory"] = await self._check_memory_usage()
        
        # 2. Storage backend check
        checks["storage"] = await self._check_storage_backend()
        
        # 3. Connection capacity check (based on testing: 50 users max)
        checks["connections"] = self._check_connection_capacity()
        
        # 4. MCP server availability
        checks["mcp_server"] = self._check_mcp_server()
        
        # 5. System resources
        checks["system_resources"] = self._check_system_resources()
        
        return checks
    
    async def _check_memory_usage(self) -> bool:
        """Check if memory usage is within safe limits."""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            # Use production limits from testing
            memory_limit = production_config.MAX_MEMORY_MB
            warning_threshold = memory_limit * production_config.memory_limits["warning_threshold"]
            
            return memory_mb < warning_threshold
            
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return False
    
    async def _check_storage_backend(self) -> bool:
        """Check if storage backend is responding."""
        try:
            if not self.mcp_server:
                return True  # No storage dependency
                
            # Simple check - try to access storage
            if hasattr(self.mcp_server, 'storage_backend'):
                storage = self.mcp_server.storage_backend
                if storage and hasattr(storage, 'ping'):
                    return await storage.ping()
            
            return True  # Assume OK if no storage backend
            
        except Exception as e:
            logger.warning(f"Storage check failed: {e}")
            return False
    
    def _check_connection_capacity(self) -> bool:
        """Check if connection count is within limits."""
        try:
            active_connections = self._get_active_connections()
            max_connections = production_config.MAX_CONCURRENT_CONNECTIONS
            
            # Use 90% of max as readiness threshold
            threshold = max_connections * 0.9
            
            return active_connections < threshold
            
        except Exception:
            return True  # Assume OK if can't determine
    
    def _check_mcp_server(self) -> bool:
        """Check if MCP server is available and responding."""
        try:
            if not self.mcp_server:
                return False
                
            # Basic availability check
            return hasattr(self.mcp_server, 'graphs') or hasattr(self.mcp_server, 'graph_manager')
            
        except Exception:
            return False
    
    def _check_system_resources(self) -> bool:
        """Check basic system resource availability."""
        try:
            # Check CPU usage isn't too high
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > 90:
                return False
                
            # Check disk space if needed
            # (Add disk checks here if using persistent storage)
            
            return True
            
        except Exception:
            return True  # Assume OK if can't check
    
    def _get_active_connections(self) -> int:
        """Get current number of active connections."""
        try:
            if self.mcp_server and hasattr(self.mcp_server, 'active_connections'):
                return getattr(self.mcp_server, 'active_connections', 0)
            
            # Fallback: count network connections to our process
            process = psutil.Process()
            connections = process.connections()
            return len([c for c in connections if c.status == 'ESTABLISHED'])
            
        except Exception:
            return 0


class ProductionHealthServer(HealthServer):
    """Production-specific health server with enhanced monitoring."""
    
    async def _perform_readiness_checks(self) -> Dict[str, bool]:
        """Enhanced readiness checks for production."""
        checks = await super()._perform_readiness_checks()
        
        # Additional production checks
        checks["performance"] = await self._check_performance_metrics()
        checks["external_dependencies"] = await self._check_external_dependencies()
        
        return checks
    
    async def _check_performance_metrics(self) -> bool:
        """Check if performance metrics are within acceptable bounds."""
        try:
            # Check response times if performance tracker is available
            if (self.mcp_server and 
                hasattr(self.mcp_server, 'performance_tracker')):
                
                tracker = self.mcp_server.performance_tracker
                avg_response_time = tracker.get_average_response_time()
                
                # Based on testing: P95 should be under 650ms for good performance
                return avg_response_time < 1.0  # 1 second threshold
            
            return True
            
        except Exception:
            return True  # Assume OK if can't check
    
    async def _check_external_dependencies(self) -> bool:
        """Check external dependencies like Redis, databases, etc."""
        try:
            # Redis check
            if production_config.STORAGE_BACKEND == "redis":
                # Add Redis ping check here
                pass
            
            # Add other external dependency checks here
            
            return True
            
        except Exception:
            return False


def create_health_server(mcp_server=None, port=None, production=None) -> HealthServer:
    """Factory function to create appropriate health server."""
    is_production = production if production is not None else production_config.is_production
    
    if is_production:
        return ProductionHealthServer(mcp_server, port)
    else:
        return HealthServer(mcp_server, port)