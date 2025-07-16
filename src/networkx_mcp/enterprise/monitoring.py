"""
Enterprise Monitoring and Observability

Comprehensive monitoring system with:
- Prometheus metrics collection
- Structured audit logging
- Performance tracking
- Health checks
- Request tracing
"""

import time
import uuid
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from collections import defaultdict, deque

try:
    import structlog
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    )
except ImportError:
    raise ImportError(
        "Enterprise monitoring requires structlog and prometheus-client. "
        "Install with: pip install networkx-mcp-server[enterprise]"
    )

from .config import get_config


@dataclass
class AuditEvent:
    """Audit event for security and compliance logging."""
    event_id: str
    timestamp: datetime
    user_id: str
    username: Optional[str]
    operation: str
    resource: Optional[str]
    success: bool
    duration_ms: float
    ip_address: Optional[str]
    user_agent: Optional[str]
    request_id: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class MetricsCollector:
    """Prometheus metrics collector for performance monitoring."""
    
    def __init__(self):
        self.config = get_config()
        self.registry = CollectorRegistry()
        self._setup_metrics()
        self._start_time = time.time()
    
    def _setup_metrics(self):
        """Initialize all Prometheus metrics."""
        
        # Request metrics
        self.request_count = Counter(
            'networkx_mcp_requests_total',
            'Total number of requests',
            ['method', 'operation', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'networkx_mcp_request_duration_seconds',
            'Request duration in seconds',
            ['operation'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        self.request_size = Histogram(
            'networkx_mcp_request_size_bytes',
            'Request size in bytes',
            ['operation'],
            registry=self.registry
        )
        
        self.response_size = Histogram(
            'networkx_mcp_response_size_bytes',
            'Response size in bytes',
            ['operation'],
            registry=self.registry
        )
        
        # Graph metrics
        self.graph_count = Gauge(
            'networkx_mcp_graphs_total',
            'Total number of graphs in memory',
            registry=self.registry
        )
        
        self.graph_nodes = Histogram(
            'networkx_mcp_graph_nodes',
            'Number of nodes per graph',
            buckets=[1, 10, 100, 1000, 10000, 100000, 1000000],
            registry=self.registry
        )
        
        self.graph_edges = Histogram(
            'networkx_mcp_graph_edges',
            'Number of edges per graph',
            buckets=[1, 10, 100, 1000, 10000, 100000, 1000000],
            registry=self.registry
        )
        
        # System metrics
        self.memory_usage = Gauge(
            'networkx_mcp_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'networkx_mcp_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.uptime = Gauge(
            'networkx_mcp_uptime_seconds',
            'Server uptime in seconds',
            registry=self.registry
        )
        
        # Authentication metrics
        self.auth_attempts = Counter(
            'networkx_mcp_auth_attempts_total',
            'Authentication attempts',
            ['method', 'result'],
            registry=self.registry
        )
        
        self.active_users = Gauge(
            'networkx_mcp_active_users',
            'Number of active users',
            registry=self.registry
        )
        
        # Rate limiting metrics
        self.rate_limit_hits = Counter(
            'networkx_mcp_rate_limit_hits_total',
            'Rate limit violations',
            ['user_id', 'limit_type'],
            registry=self.registry
        )
        
        # Error metrics
        self.error_count = Counter(
            'networkx_mcp_errors_total',
            'Total number of errors',
            ['operation', 'error_type'],
            registry=self.registry
        )
        
        # Server info
        self.server_info = Info(
            'networkx_mcp_server',
            'Server information',
            registry=self.registry
        )
        
        # Set server info
        self.server_info.info({
            'version': '1.1.0',
            'enterprise_mode': str(self.config.enterprise_mode),
            'transport': self.config.server.transport,
            'auth_methods': self._get_auth_methods()
        })
    
    def _get_auth_methods(self) -> str:
        """Get enabled authentication methods."""
        methods = []
        if self.config.security.api_key_enabled:
            methods.append('api_key')
        if self.config.security.oauth_enabled:
            methods.append('oauth')
        return ','.join(methods) if methods else 'none'
    
    def record_request(self, operation: str, duration: float, status: str, 
                      request_size: int = 0, response_size: int = 0):
        """Record request metrics."""
        self.request_count.labels(
            method='mcp',
            operation=operation,
            status=status
        ).inc()
        
        self.request_duration.labels(operation=operation).observe(duration)
        
        if request_size > 0:
            self.request_size.labels(operation=operation).observe(request_size)
        
        if response_size > 0:
            self.response_size.labels(operation=operation).observe(response_size)
    
    def record_graph_metrics(self, graph_count: int, nodes: int = 0, edges: int = 0):
        """Record graph-related metrics."""
        self.graph_count.set(graph_count)
        
        if nodes > 0:
            self.graph_nodes.observe(nodes)
        
        if edges > 0:
            self.graph_edges.observe(edges)
    
    def record_auth_attempt(self, method: str, success: bool):
        """Record authentication attempt."""
        result = 'success' if success else 'failure'
        self.auth_attempts.labels(method=method, result=result).inc()
    
    def record_rate_limit_hit(self, user_id: str, limit_type: str):
        """Record rate limit violation."""
        self.rate_limit_hits.labels(user_id=user_id, limit_type=limit_type).inc()
    
    def record_error(self, operation: str, error_type: str):
        """Record error occurrence."""
        self.error_count.labels(operation=operation, error_type=error_type).inc()
    
    def update_system_metrics(self):
        """Update system resource metrics."""
        try:
            import psutil
            
            # Memory usage
            memory = psutil.Process().memory_info()
            self.memory_usage.set(memory.rss)
            
            # CPU usage
            cpu_percent = psutil.Process().cpu_percent()
            self.cpu_usage.set(cpu_percent)
            
            # Uptime
            uptime = time.time() - self._start_time
            self.uptime.set(uptime)
            
        except ImportError:
            # psutil not available, skip system metrics
            pass
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        self.update_system_metrics()
        return generate_latest(self.registry).decode('utf-8')
    
    def get_content_type(self) -> str:
        """Get metrics content type."""
        return CONTENT_TYPE_LATEST


class AuditLogger:
    """Structured audit logging for security and compliance."""
    
    def __init__(self):
        self.config = get_config()
        self._setup_logging()
        self._request_context = threading.local()
    
    def _setup_logging(self):
        """Setup structured logging configuration."""
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]
        
        if self.config.monitoring.log_format == "json":
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger("audit")
        self.performance_logger = structlog.get_logger("performance")
        self.security_logger = structlog.get_logger("security")
    
    def set_request_context(self, request_id: str, user_id: str, ip_address: str, user_agent: str):
        """Set request context for current thread."""
        self._request_context.request_id = request_id
        self._request_context.user_id = user_id
        self._request_context.ip_address = ip_address
        self._request_context.user_agent = user_agent
    
    def get_request_context(self) -> Dict[str, str]:
        """Get current request context."""
        return {
            'request_id': getattr(self._request_context, 'request_id', 'unknown'),
            'user_id': getattr(self._request_context, 'user_id', 'unknown'),
            'ip_address': getattr(self._request_context, 'ip_address', 'unknown'),
            'user_agent': getattr(self._request_context, 'user_agent', 'unknown'),
        }
    
    def log_authentication(self, user_id: str, method: str, success: bool, 
                          ip_address: str, details: Dict[str, Any] = None):
        """Log authentication event."""
        event_data = {
            'event_type': 'authentication',
            'user_id': user_id,
            'method': method,
            'success': success,
            'ip_address': ip_address,
            'timestamp': datetime.utcnow().isoformat(),
            **(details or {})
        }
        
        if success:
            self.security_logger.info("Authentication successful", **event_data)
        else:
            self.security_logger.warning("Authentication failed", **event_data)
    
    def log_authorization(self, user_id: str, operation: str, resource: str, 
                         allowed: bool, reason: str = None):
        """Log authorization decision."""
        event_data = {
            'event_type': 'authorization',
            'user_id': user_id,
            'operation': operation,
            'resource': resource,
            'allowed': allowed,
            'reason': reason,
            'timestamp': datetime.utcnow().isoformat(),
            **self.get_request_context()
        }
        
        if allowed:
            self.security_logger.info("Authorization granted", **event_data)
        else:
            self.security_logger.warning("Authorization denied", **event_data)
    
    def log_operation(self, operation: str, resource: str, success: bool, 
                     duration_ms: float, metadata: Dict[str, Any] = None):
        """Log operation execution."""
        context = self.get_request_context()
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            user_id=context['user_id'],
            username=None,  # Could be enriched from user context
            operation=operation,
            resource=resource,
            success=success,
            duration_ms=duration_ms,
            ip_address=context['ip_address'],
            user_agent=context['user_agent'],
            request_id=context['request_id'],
            metadata=metadata or {}
        )
        
        if success:
            self.logger.info("Operation completed", **event.to_dict())
        else:
            self.logger.error("Operation failed", **event.to_dict())
    
    def log_performance(self, operation: str, duration_ms: float, 
                       resource_usage: Dict[str, Any] = None):
        """Log performance metrics."""
        event_data = {
            'operation': operation,
            'duration_ms': duration_ms,
            'timestamp': datetime.utcnow().isoformat(),
            **self.get_request_context(),
            **(resource_usage or {})
        }
        
        self.performance_logger.info("Performance metric", **event_data)
    
    def log_security_event(self, event_type: str, severity: str, 
                          details: Dict[str, Any]):
        """Log security-related events."""
        event_data = {
            'event_type': event_type,
            'severity': severity,
            'timestamp': datetime.utcnow().isoformat(),
            **self.get_request_context(),
            **details
        }
        
        if severity in ['high', 'critical']:
            self.security_logger.error("Security event", **event_data)
        elif severity == 'medium':
            self.security_logger.warning("Security event", **event_data)
        else:
            self.security_logger.info("Security event", **event_data)
    
    @contextmanager
    def operation_context(self, operation: str, resource: str = None):
        """Context manager for operation logging."""
        start_time = time.time()
        success = False
        error = None
        
        try:
            yield
            success = True
        except Exception as e:
            error = str(e)
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            metadata = {'error': error} if error else {}
            self.log_operation(operation, resource or 'unknown', success, duration_ms, metadata)


# Global instances
_metrics_collector: Optional[MetricsCollector] = None
_audit_logger: Optional[AuditLogger] = None

def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector

def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger