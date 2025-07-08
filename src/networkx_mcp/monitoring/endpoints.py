"""HTTP endpoints for health monitoring and metrics."""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

import psutil

from .health_checks import get_health_checker, HealthStatus
from .metrics import MetricsCollector
from ..features import is_feature_enabled

logger = logging.getLogger(__name__)


class HealthEndpoint:
    """Comprehensive health check endpoint for load balancers and monitoring."""
    
    def __init__(self):
        self.health_checker = get_health_checker()
        self.start_time = time.time()
    
    def check_health(self, include_details: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive health check.
        
        Parameters:
        -----------
        include_details : bool
            Whether to include detailed component information
            
        Returns:
        --------
        Dict containing health status and details
        """
        start_time = time.time()
        
        # Core health checks
        checks = {
            "database": self._check_storage_health(),
            "memory": self._check_memory_health(),
            "disk": self._check_disk_health(),
            "network": self._check_network_health(),
            "features": self._check_feature_health(),
            "dependencies": self._check_dependency_health(),
        }
        
        # Determine overall health
        failed_checks = [name for name, result in checks.items() 
                        if result["status"] not in ["healthy", "degraded"]]
        
        degraded_checks = [name for name, result in checks.items() 
                          if result["status"] == "degraded"]
        
        if failed_checks:
            overall_status = "unhealthy"
        elif degraded_checks:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        # Calculate uptime
        uptime_seconds = time.time() - self.start_time
        
        response = {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "version": "1.0.0",
            "uptime_seconds": round(uptime_seconds, 2),
            "checks": checks if include_details else {
                name: {"status": result["status"]} 
                for name, result in checks.items()
            },
            "summary": {
                "total_checks": len(checks),
                "healthy": len([c for c in checks.values() if c["status"] == "healthy"]),
                "degraded": len(degraded_checks),
                "unhealthy": len(failed_checks),
            }
        }
        
        # Add performance metrics
        if include_details:
            response["performance"] = {
                "check_duration_ms": round((time.time() - start_time) * 1000, 2),
                "memory_usage_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 2),
                "cpu_percent": psutil.cpu_percent(interval=0.1),
            }
        
        return response
    
    def _check_storage_health(self) -> Dict[str, Any]:
        """Check storage/database connectivity."""
        try:
            # Import storage components
            from ..storage.factory import get_storage_backend
            
            backend = get_storage_backend()
            
            # Simple health check using the backend's health check method
            if hasattr(backend, 'check_health'):
                # For now, just check if backend is available and initialized
                if hasattr(backend, '_initialized') and backend._initialized:
                    return {
                        "status": "healthy",
                        "message": "Storage backend operational",
                        "details": {
                            "backend_type": backend.__class__.__name__,
                            "initialized": True
                        }
                    }
                else:
                    return {
                        "status": "degraded",
                        "message": "Storage backend not initialized",
                        "details": {
                            "backend_type": backend.__class__.__name__,
                            "initialized": False
                        }
                    }
            else:
                return {
                    "status": "degraded",
                    "message": "Storage backend health check not available",
                    "details": {
                        "backend_type": backend.__class__.__name__
                    }
                }
            
        except Exception as e:
            logger.error(f"Storage health check failed: {e}")
            return {
                "status": "unhealthy",
                "message": "Storage backend unavailable",
                "error": str(e)
            }
    
    def _check_memory_health(self) -> Dict[str, Any]:
        """Check memory usage and availability."""
        try:
            # System memory
            system_memory = psutil.virtual_memory()
            
            # Process memory
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # Memory thresholds
            system_warning_threshold = 80  # %
            system_critical_threshold = 95  # %
            process_warning_threshold = 1024  # MB
            process_critical_threshold = 2048  # MB
            
            system_usage_percent = system_memory.percent
            process_usage_mb = process_memory.rss / 1024 / 1024
            
            # Determine status
            if (system_usage_percent > system_critical_threshold or 
                process_usage_mb > process_critical_threshold):
                status = "unhealthy"
                message = "Critical memory usage"
            elif (system_usage_percent > system_warning_threshold or 
                  process_usage_mb > process_warning_threshold):
                status = "degraded"
                message = "High memory usage"
            else:
                status = "healthy"
                message = "Memory usage normal"
            
            return {
                "status": status,
                "message": message,
                "details": {
                    "system_memory_percent": round(system_usage_percent, 1),
                    "system_memory_available_gb": round(system_memory.available / 1024**3, 2),
                    "process_memory_mb": round(process_usage_mb, 2),
                    "process_memory_percent": round(process_memory.percent, 1) if hasattr(process_memory, 'percent') else None
                }
            }
            
        except Exception as e:
            logger.error(f"Memory health check failed: {e}")
            return {
                "status": "unhealthy",
                "message": "Memory check failed",
                "error": str(e)
            }
    
    def _check_disk_health(self) -> Dict[str, Any]:
        """Check disk space and I/O."""
        try:
            # Check disk usage
            disk_usage = psutil.disk_usage('/')
            
            # Disk thresholds
            warning_threshold = 80  # %
            critical_threshold = 95  # %
            
            usage_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Determine status
            if usage_percent > critical_threshold:
                status = "unhealthy"
                message = "Critical disk usage"
            elif usage_percent > warning_threshold:
                status = "degraded"
                message = "High disk usage"
            else:
                status = "healthy"
                message = "Disk usage normal"
            
            return {
                "status": status,
                "message": message,
                "details": {
                    "usage_percent": round(usage_percent, 1),
                    "free_gb": round(disk_usage.free / 1024**3, 2),
                    "total_gb": round(disk_usage.total / 1024**3, 2)
                }
            }
            
        except Exception as e:
            logger.error(f"Disk health check failed: {e}")
            return {
                "status": "unhealthy",
                "message": "Disk check failed",
                "error": str(e)
            }
    
    def _check_network_health(self) -> Dict[str, Any]:
        """Check network connectivity and performance."""
        try:
            # Check network interfaces
            net_io = psutil.net_io_counters()
            
            # Simple connectivity test (check if we can resolve localhost)
            import socket
            try:
                socket.getaddrinfo("localhost", 80)
                connectivity = "ok"
            except socket.gaierror:
                connectivity = "failed"
            
            status = "healthy" if connectivity == "ok" else "degraded"
            
            return {
                "status": status,
                "message": f"Network connectivity {connectivity}",
                "details": {
                    "connectivity": connectivity,
                    "bytes_sent": net_io.bytes_sent if net_io else 0,
                    "bytes_recv": net_io.bytes_recv if net_io else 0,
                    "packets_sent": net_io.packets_sent if net_io else 0,
                    "packets_recv": net_io.packets_recv if net_io else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Network health check failed: {e}")
            return {
                "status": "degraded",
                "message": "Network check limited",
                "error": str(e)
            }
    
    def _check_feature_health(self) -> Dict[str, Any]:
        """Check feature flag system health."""
        try:
            from ..features import get_flag_manager, get_feature_flags
            
            manager = get_flag_manager()
            all_flags = get_feature_flags()
            
            # Validate dependencies
            validation_errors = manager.validate_dependencies()
            
            if validation_errors:
                status = "degraded"
                message = f"Feature validation issues: {len(validation_errors)}"
            else:
                status = "healthy"
                message = "All features configured correctly"
            
            enabled_count = sum(1 for f in all_flags.values() if f['enabled'])
            
            return {
                "status": status,
                "message": message,
                "details": {
                    "total_features": len(all_flags),
                    "enabled_features": enabled_count,
                    "validation_errors": validation_errors
                }
            }
            
        except Exception as e:
            logger.error(f"Feature health check failed: {e}")
            return {
                "status": "degraded",
                "message": "Feature system check failed",
                "error": str(e)
            }
    
    def _check_dependency_health(self) -> Dict[str, Any]:
        """Check external dependencies."""
        try:
            dependencies = {
                "networkx": self._check_networkx(),
                "numpy": self._check_numpy(),
                "psutil": self._check_psutil(),
            }
            
            # Check optional dependencies
            optional_deps = {
                "redis": self._check_redis(),
                "sklearn": self._check_sklearn(),
            }
            
            # Count failures
            failed_required = [name for name, status in dependencies.items() if not status]
            failed_optional = [name for name, status in optional_deps.items() if not status]
            
            if failed_required:
                status = "unhealthy"
                message = f"Required dependencies failed: {', '.join(failed_required)}"
            elif failed_optional:
                status = "degraded"
                message = f"Optional dependencies missing: {', '.join(failed_optional)}"
            else:
                status = "healthy"
                message = "All dependencies available"
            
            return {
                "status": status,
                "message": message,
                "details": {
                    "required": dependencies,
                    "optional": optional_deps
                }
            }
            
        except Exception as e:
            logger.error(f"Dependency health check failed: {e}")
            return {
                "status": "unhealthy",
                "message": "Dependency check failed",
                "error": str(e)
            }
    
    def _check_networkx(self) -> bool:
        """Check NetworkX availability."""
        try:
            import networkx as nx
            # Quick test
            G = nx.Graph()
            G.add_edge(1, 2)
            return len(G) == 2
        except Exception:
            return False
    
    def _check_numpy(self) -> bool:
        """Check NumPy availability."""
        try:
            import numpy as np
            # Quick test
            arr = np.array([1, 2, 3])
            return len(arr) == 3
        except Exception:
            return False
    
    def _check_psutil(self) -> bool:
        """Check psutil availability."""
        try:
            return psutil.cpu_percent(interval=None) >= 0
        except Exception:
            return False
    
    def _check_redis(self) -> bool:
        """Check Redis availability (optional)."""
        try:
            import redis
            return True
        except ImportError:
            return False
    
    def _check_sklearn(self) -> bool:
        """Check scikit-learn availability (optional)."""
        try:
            import sklearn
            return True
        except ImportError:
            return False


class ReadinessEndpoint:
    """Kubernetes readiness probe endpoint."""
    
    def __init__(self):
        self.health_endpoint = HealthEndpoint()
    
    def check_readiness(self) -> Dict[str, Any]:
        """
        Check if service is ready to accept traffic.
        
        This is stricter than health check - service must be fully operational.
        """
        start_time = time.time()
        
        # Critical readiness checks
        checks = {
            "storage": self._check_storage_ready(),
            "core_features": self._check_core_features_ready(),
            "memory": self._check_memory_ready(),
        }
        
        # All checks must pass for readiness
        failed_checks = [name for name, result in checks.items() 
                        if result["status"] != "ready"]
        
        is_ready = len(failed_checks) == 0
        
        response = {
            "ready": is_ready,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "checks": checks,
            "failed_checks": failed_checks,
            "check_duration_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        return response
    
    def _check_storage_ready(self) -> Dict[str, Any]:
        """Check if storage is ready for operations."""
        try:
            from ..storage.factory import get_storage_backend
            
            backend = get_storage_backend()
            
            # Simple readiness check - storage must be initialized and healthy
            if hasattr(backend, '_initialized') and backend._initialized:
                return {"status": "ready", "message": "Storage operational"}
            else:
                return {"status": "not_ready", "message": "Storage not initialized"}
                
        except Exception as e:
            return {"status": "not_ready", "message": f"Storage error: {str(e)}"}
    
    def _check_core_features_ready(self) -> Dict[str, Any]:
        """Check if core features are ready."""
        try:
            # Test graph operations
            from ..core.graph_operations import GraphManager
            
            manager = GraphManager()
            
            # Quick functionality test
            test_graph_name = f"readiness_test_{int(time.time())}"
            manager.create_graph(test_graph_name, "Graph")
            manager.add_nodes_from(test_graph_name, ["A", "B"])
            manager.add_edges_from(test_graph_name, [("A", "B")])
            
            # Verify
            graph = manager.get_graph(test_graph_name)
            ready = graph.number_of_nodes() == 2 and graph.number_of_edges() == 1
            
            # Cleanup
            if test_graph_name in manager.graphs:
                del manager.graphs[test_graph_name]
            
            if ready:
                return {"status": "ready", "message": "Core features operational"}
            else:
                return {"status": "not_ready", "message": "Core features test failed"}
                
        except Exception as e:
            return {"status": "not_ready", "message": f"Core features error: {str(e)}"}
    
    def _check_memory_ready(self) -> Dict[str, Any]:
        """Check if memory usage is within ready thresholds."""
        try:
            memory = psutil.virtual_memory()
            
            # Stricter thresholds for readiness
            ready_threshold = 90  # %
            
            if memory.percent < ready_threshold:
                return {"status": "ready", "message": "Memory usage acceptable"}
            else:
                return {
                    "status": "not_ready", 
                    "message": f"Memory usage too high: {memory.percent:.1f}%"
                }
                
        except Exception as e:
            return {"status": "not_ready", "message": f"Memory check error: {str(e)}"}


class MetricsEndpoint:
    """Prometheus-compatible metrics endpoint."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.start_time = time.time()
    
    def get_metrics(self, format: str = "prometheus") -> str:
        """
        Export metrics in specified format.
        
        Parameters:
        -----------
        format : str
            Export format: 'prometheus', 'json'
            
        Returns:
        --------
        Formatted metrics string
        """
        if format == "prometheus":
            return self._export_prometheus_format()
        elif format == "json":
            return self._export_json_format()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        metrics = []
        timestamp = int(time.time() * 1000)  # Prometheus expects milliseconds
        
        # System metrics
        metrics.extend(self._get_system_metrics())
        
        # Application metrics
        metrics.extend(self._get_application_metrics())
        
        # Feature metrics
        metrics.extend(self._get_feature_metrics())
        
        # Performance metrics
        metrics.extend(self._get_performance_metrics())
        
        return "\n".join(metrics) + "\n"
    
    def _get_system_metrics(self) -> list[str]:
        """Get system-level metrics."""
        metrics = []
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.append(f'# HELP system_memory_usage_bytes System memory usage in bytes')
        metrics.append(f'# TYPE system_memory_usage_bytes gauge')
        metrics.append(f'system_memory_usage_bytes{{type="used"}} {memory.used}')
        metrics.append(f'system_memory_usage_bytes{{type="available"}} {memory.available}')
        metrics.append(f'system_memory_usage_bytes{{type="total"}} {memory.total}')
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        metrics.append(f'# HELP system_cpu_usage_percent System CPU usage percentage')
        metrics.append(f'# TYPE system_cpu_usage_percent gauge')
        metrics.append(f'system_cpu_usage_percent {cpu_percent}')
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics.append(f'# HELP system_disk_usage_bytes System disk usage in bytes')
        metrics.append(f'# TYPE system_disk_usage_bytes gauge')
        metrics.append(f'system_disk_usage_bytes{{type="used"}} {disk.used}')
        metrics.append(f'system_disk_usage_bytes{{type="free"}} {disk.free}')
        metrics.append(f'system_disk_usage_bytes{{type="total"}} {disk.total}')
        
        return metrics
    
    def _get_application_metrics(self) -> list[str]:
        """Get application-specific metrics."""
        metrics = []
        
        # Uptime
        uptime = time.time() - self.start_time
        metrics.append(f'# HELP networkx_mcp_uptime_seconds Application uptime in seconds')
        metrics.append(f'# TYPE networkx_mcp_uptime_seconds counter')
        metrics.append(f'networkx_mcp_uptime_seconds {uptime:.2f}')
        
        # Graph count
        try:
            from ..core.graph_operations import GraphManager
            manager = GraphManager()
            graph_count = len(manager.graphs)
            
            metrics.append(f'# HELP networkx_mcp_graphs_total Total number of graphs')
            metrics.append(f'# TYPE networkx_mcp_graphs_total gauge')
            metrics.append(f'networkx_mcp_graphs_total {graph_count}')
            
            # Graph size metrics
            total_nodes = 0
            total_edges = 0
            for graph in manager.graphs.values():
                total_nodes += graph.number_of_nodes()
                total_edges += graph.number_of_edges()
            
            metrics.append(f'# HELP networkx_mcp_nodes_total Total number of nodes across all graphs')
            metrics.append(f'# TYPE networkx_mcp_nodes_total gauge')
            metrics.append(f'networkx_mcp_nodes_total {total_nodes}')
            
            metrics.append(f'# HELP networkx_mcp_edges_total Total number of edges across all graphs')
            metrics.append(f'# TYPE networkx_mcp_edges_total gauge')
            metrics.append(f'networkx_mcp_edges_total {total_edges}')
            
        except Exception as e:
            logger.error(f"Failed to collect graph metrics: {e}")
        
        return metrics
    
    def _get_feature_metrics(self) -> list[str]:
        """Get feature flag metrics."""
        metrics = []
        
        try:
            from ..features import get_feature_flags
            
            flags = get_feature_flags()
            enabled_count = sum(1 for f in flags.values() if f['enabled'])
            total_count = len(flags)
            
            metrics.append(f'# HELP networkx_mcp_features_enabled Number of enabled features')
            metrics.append(f'# TYPE networkx_mcp_features_enabled gauge')
            metrics.append(f'networkx_mcp_features_enabled {enabled_count}')
            
            metrics.append(f'# HELP networkx_mcp_features_total Total number of features')
            metrics.append(f'# TYPE networkx_mcp_features_total gauge')
            metrics.append(f'networkx_mcp_features_total {total_count}')
            
            # Feature breakdown by category
            by_category = {}
            for name, info in flags.items():
                category = info.get('category', 'general')
                if category not in by_category:
                    by_category[category] = {'enabled': 0, 'total': 0}
                by_category[category]['total'] += 1
                if info['enabled']:
                    by_category[category]['enabled'] += 1
            
            metrics.append(f'# HELP networkx_mcp_features_by_category Features by category')
            metrics.append(f'# TYPE networkx_mcp_features_by_category gauge')
            for category, counts in by_category.items():
                metrics.append(f'networkx_mcp_features_by_category{{category="{category}",status="enabled"}} {counts["enabled"]}')
                metrics.append(f'networkx_mcp_features_by_category{{category="{category}",status="total"}} {counts["total"]}')
            
        except Exception as e:
            logger.error(f"Failed to collect feature metrics: {e}")
        
        return metrics
    
    def _get_performance_metrics(self) -> list[str]:
        """Get performance metrics."""
        metrics = []
        
        # Process memory
        process = psutil.Process()
        memory_info = process.memory_info()
        
        metrics.append(f'# HELP networkx_mcp_process_memory_bytes Process memory usage in bytes')
        metrics.append(f'# TYPE networkx_mcp_process_memory_bytes gauge')
        metrics.append(f'networkx_mcp_process_memory_bytes{{type="rss"}} {memory_info.rss}')
        metrics.append(f'networkx_mcp_process_memory_bytes{{type="vms"}} {memory_info.vms}')
        
        # Open files
        try:
            open_files = len(process.open_files())
            metrics.append(f'# HELP networkx_mcp_open_files Number of open file descriptors')
            metrics.append(f'# TYPE networkx_mcp_open_files gauge')
            metrics.append(f'networkx_mcp_open_files {open_files}')
        except Exception:
            pass  # Not always available on all systems
        
        # Threads
        try:
            num_threads = process.num_threads()
            metrics.append(f'# HELP networkx_mcp_threads Number of threads')
            metrics.append(f'# TYPE networkx_mcp_threads gauge')
            metrics.append(f'networkx_mcp_threads {num_threads}')
        except Exception:
            pass
        
        return metrics
    
    def _export_json_format(self) -> str:
        """Export metrics in JSON format."""
        metrics_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "system": self._get_system_metrics_json(),
            "application": self._get_application_metrics_json(),
            "features": self._get_feature_metrics_json(),
            "performance": self._get_performance_metrics_json()
        }
        
        return json.dumps(metrics_data, indent=2)
    
    def _get_system_metrics_json(self) -> Dict[str, Any]:
        """Get system metrics in JSON format."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "memory": {
                "used_bytes": memory.used,
                "available_bytes": memory.available,
                "total_bytes": memory.total,
                "percent": memory.percent
            },
            "cpu": {
                "usage_percent": psutil.cpu_percent(interval=0.1)
            },
            "disk": {
                "used_bytes": disk.used,
                "free_bytes": disk.free,
                "total_bytes": disk.total,
                "usage_percent": (disk.used / disk.total) * 100
            }
        }
    
    def _get_application_metrics_json(self) -> Dict[str, Any]:
        """Get application metrics in JSON format."""
        try:
            from ..core.graph_operations import GraphManager
            manager = GraphManager()
            
            total_nodes = sum(g.number_of_nodes() for g in manager.graphs.values())
            total_edges = sum(g.number_of_edges() for g in manager.graphs.values())
            
            return {
                "uptime_seconds": time.time() - self.start_time,
                "graphs": {
                    "total": len(manager.graphs),
                    "total_nodes": total_nodes,
                    "total_edges": total_edges
                }
            }
        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")
            return {
                "uptime_seconds": time.time() - self.start_time,
                "graphs": {"error": str(e)}
            }
    
    def _get_feature_metrics_json(self) -> Dict[str, Any]:
        """Get feature metrics in JSON format."""
        try:
            from ..features import get_feature_flags
            
            flags = get_feature_flags()
            enabled_count = sum(1 for f in flags.values() if f['enabled'])
            
            return {
                "total": len(flags),
                "enabled": enabled_count,
                "disabled": len(flags) - enabled_count
            }
        except Exception as e:
            logger.error(f"Failed to collect feature metrics: {e}")
            return {"error": str(e)}
    
    def _get_performance_metrics_json(self) -> Dict[str, Any]:
        """Get performance metrics in JSON format."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "memory": {
                "rss_bytes": memory_info.rss,
                "vms_bytes": memory_info.vms
            },
            "threads": getattr(process, 'num_threads', lambda: None)() or 0
        }


def create_monitoring_endpoints():
    """Create and return monitoring endpoint instances."""
    return {
        "health": HealthEndpoint(),
        "ready": ReadinessEndpoint(),
        "metrics": MetricsEndpoint()
    }