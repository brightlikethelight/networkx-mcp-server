# 🚀 NetworkX MCP Server: Production Transformation Plan

## 🎯 Executive Summary

The NetworkX MCP Server is currently a **functional prototype** with **critical production blockers**. This plan transforms it into a **production-grade system** through systematic fixes prioritized by risk.

**Current State**: Working but UNSAFE for production
**Target State**: Secure, scalable, observable, and maintainable

---

## 🔴 PHASE 1: CRITICAL SECURITY & PERSISTENCE (Week 1-2)

### 1.1 Persistence Layer with Transaction Support

```python
# src/networkx_mcp/storage/base.py
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
import asyncio
import networkx as nx

class Transaction(ABC):
    """Ensures atomicity for multi-step operations."""
    
    @abstractmethod
    async def commit(self) -> None:
        pass
    
    @abstractmethod
    async def rollback(self) -> None:
        pass

class StorageBackend(ABC):
    """Abstract storage with transaction support."""
    
    @abstractmethod
    @asynccontextmanager
    async def transaction(self) -> Transaction:
        """Context manager for atomic operations."""
        pass
    
    @abstractmethod
    async def save_graph(self, user_id: str, graph_id: str, graph: nx.Graph, 
                        tx: Optional[Transaction] = None) -> bool:
        pass
    
    @abstractmethod
    async def load_graph(self, user_id: str, graph_id: str,
                        tx: Optional[Transaction] = None) -> Optional[nx.Graph]:
        pass
    
    @abstractmethod
    async def list_graphs(self, user_id: str,
                         tx: Optional[Transaction] = None) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def get_storage_stats(self, user_id: str) -> Dict[str, int]:
        """Get storage usage for rate limiting."""
        pass

# src/networkx_mcp/storage/redis_backend.py
import redis.asyncio as redis
import pickle
import zlib
import json
from datetime import datetime

class RedisBackend(StorageBackend):
    """Production Redis backend with compression and metadata."""
    
    def __init__(self, redis_url: str, max_graph_size_mb: int = 100):
        self.redis_url = redis_url
        self.max_size = max_graph_size_mb * 1024 * 1024
        self.pool = None
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Create connection pool."""
        self.pool = redis.ConnectionPool.from_url(
            self.redis_url,
            max_connections=50,
            decode_responses=False
        )
        
    @asynccontextmanager
    async def transaction(self):
        """Redis transaction support."""
        async with self.pool.get_connection() as conn:
            async with conn.pipeline(transaction=True) as pipe:
                try:
                    yield RedisTransaction(pipe)
                    await pipe.execute()
                except Exception:
                    await pipe.reset()
                    raise
    
    async def save_graph(self, user_id: str, graph_id: str, graph: nx.Graph, 
                        tx: Optional[Transaction] = None):
        """Save with compression and metadata."""
        # Serialize and compress
        data = pickle.dumps(graph, protocol=5)
        compressed = zlib.compress(data, level=6)
        
        # Check size limit
        if len(compressed) > self.max_size:
            raise ValueError(f"Graph exceeds size limit: {len(compressed)/1024/1024:.1f}MB")
        
        # Prepare metadata
        metadata = {
            "user_id": user_id,
            "graph_id": graph_id,
            "created_at": datetime.utcnow().isoformat(),
            "size_bytes": len(compressed),
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "graph_type": type(graph).__name__
        }
        
        # Use transaction if provided
        client = tx.pipe if tx else await self._get_client()
        
        # Atomic save
        key = f"graph:{user_id}:{graph_id}"
        meta_key = f"graph_meta:{user_id}:{graph_id}"
        user_graphs_key = f"user_graphs:{user_id}"
        
        if not tx:
            async with client.pipeline(transaction=True) as pipe:
                await pipe.set(key, compressed)
                await pipe.set(meta_key, json.dumps(metadata))
                await pipe.sadd(user_graphs_key, graph_id)
                await pipe.execute()
        else:
            await client.set(key, compressed)
            await client.set(meta_key, json.dumps(metadata))
            await client.sadd(user_graphs_key, graph_id)
        
        return True
```

### 1.2 Security Layer with Defense in Depth

```python
# src/networkx_mcp/security/validator.py
import re
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import hmac

class SecurityValidator:
    """Comprehensive input validation and sanitization."""
    
    # Allowed patterns
    GRAPH_ID_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}$')
    NODE_ID_PATTERN = re.compile(r'^[^<>&"\']{1,1000}$')
    
    # Size limits
    MAX_STRING_LENGTH = 10_000
    MAX_DICT_SIZE = 1_000
    MAX_LIST_LENGTH = 10_000
    
    @classmethod
    def validate_graph_id(cls, graph_id: str) -> str:
        """Prevent injection attacks."""
        if not isinstance(graph_id, str):
            raise TypeError("Graph ID must be string")
        
        if not cls.GRAPH_ID_PATTERN.match(graph_id):
            raise ValueError(
                "Invalid graph ID. Must be 1-100 chars, "
                "alphanumeric with underscores/hyphens"
            )
        
        return graph_id
    
    @classmethod
    def validate_node_id(cls, node_id: Any) -> Any:
        """Validate node identifiers."""
        if isinstance(node_id, str):
            if len(node_id) > 1000:
                raise ValueError("Node ID too long (max 1000 chars)")
            if not cls.NODE_ID_PATTERN.match(node_id):
                raise ValueError("Node ID contains invalid characters")
        elif isinstance(node_id, (int, float)):
            if abs(node_id) > 1e15:
                raise ValueError("Numeric node ID out of range")
        else:
            raise TypeError(f"Invalid node ID type: {type(node_id)}")
        
        return node_id
    
    @classmethod
    def sanitize_attributes(cls, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Remove dangerous attributes and limit sizes."""
        if not isinstance(attrs, dict):
            return {}
        
        if len(attrs) > cls.MAX_DICT_SIZE:
            raise ValueError(f"Too many attributes (max {cls.MAX_DICT_SIZE})")
        
        # Dangerous patterns
        dangerous_keys = [
            '__', 'eval', 'exec', 'compile', 'globals', 'locals',
            'import', 'open', 'file', 'input', 'raw_input'
        ]
        
        sanitized = {}
        for key, value in attrs.items():
            # Check key
            if not isinstance(key, str):
                continue
            if any(danger in key.lower() for danger in dangerous_keys):
                continue
            if len(key) > 200:
                continue
            
            # Sanitize value
            sanitized[key] = cls._sanitize_value(value)
        
        return sanitized
    
    @classmethod
    def _sanitize_value(cls, value: Any, depth: int = 0) -> Any:
        """Recursively sanitize values."""
        if depth > 10:  # Prevent deep recursion
            return None
        
        if isinstance(value, str):
            return value[:cls.MAX_STRING_LENGTH]
        elif isinstance(value, (int, float, bool, type(None))):
            return value
        elif isinstance(value, list):
            if len(value) > cls.MAX_LIST_LENGTH:
                value = value[:cls.MAX_LIST_LENGTH]
            return [cls._sanitize_value(v, depth+1) for v in value]
        elif isinstance(value, dict):
            if len(value) > cls.MAX_DICT_SIZE:
                return {"error": "dict too large"}
            return {
                k: cls._sanitize_value(v, depth+1) 
                for k, v in value.items() 
                if isinstance(k, str) and len(k) < 200
            }
        else:
            return str(value)[:cls.MAX_STRING_LENGTH]

# src/networkx_mcp/security/file_security.py
class SecureFileHandler:
    """Secure file operations with strict validation."""
    
    def __init__(self, allowed_dirs: Optional[List[str]] = None):
        if allowed_dirs is None:
            # Default to temp directory only
            self.allowed_dirs = [Path(tempfile.gettempdir()).resolve()]
        else:
            self.allowed_dirs = [Path(d).resolve() for d in allowed_dirs]
        
        # Create secure temp directory
        self.secure_temp = Path(tempfile.mkdtemp(prefix="networkx_mcp_"))
        self.allowed_dirs.append(self.secure_temp)
    
    def validate_path(self, filepath: str) -> Path:
        """Prevent directory traversal attacks."""
        # Normalize and resolve path
        try:
            requested = Path(filepath).resolve(strict=False)
        except Exception:
            raise ValueError(f"Invalid path: {filepath}")
        
        # Check against allowed directories
        for allowed in self.allowed_dirs:
            try:
                requested.relative_to(allowed)
                # Additional checks
                if requested.is_symlink():
                    raise ValueError("Symlinks not allowed")
                return requested
            except ValueError:
                continue
        
        raise PermissionError(f"Access denied: {filepath}")
    
    def validate_format(self, format: str) -> str:
        """Only allow safe formats."""
        safe_formats = {
            'graphml', 'gml', 'pajek', 'edgelist', 
            'adjlist', 'json', 'yaml'
        }
        
        if format.lower() not in safe_formats:
            raise ValueError(
                f"Unsafe format '{format}'. "
                f"Allowed: {', '.join(safe_formats)}"
            )
        
        return format.lower()
```

### 1.3 Concurrency-Safe State Management

```python
# src/networkx_mcp/core/graph_manager_v2.py
import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
import time

class ConcurrentGraphManager:
    """Thread-safe graph manager with proper locking."""
    
    def __init__(self, storage: StorageBackend, max_graphs_per_user: int = 100):
        self.storage = storage
        self.max_graphs_per_user = max_graphs_per_user
        
        # Concurrency control
        self._locks = defaultdict(asyncio.Lock)  # Per graph locks
        self._user_locks = defaultdict(asyncio.Lock)  # Per user locks
        self._global_lock = asyncio.RLock()  # Global operations
        
        # Caching with TTL
        self._cache = {}  # graph_id -> (graph, timestamp)
        self._cache_ttl = 300  # 5 minutes
        
    @asynccontextmanager
    async def _graph_lock(self, user_id: str, graph_id: str):
        """Acquire lock for specific graph."""
        lock_key = f"{user_id}:{graph_id}"
        async with self._locks[lock_key]:
            yield
    
    async def create_graph(self, user_id: str, graph_id: str, 
                          graph_type: str, **kwargs) -> Dict[str, Any]:
        """Create graph with concurrency control."""
        # Validate inputs
        user_id = SecurityValidator.validate_graph_id(user_id)
        graph_id = SecurityValidator.validate_graph_id(graph_id)
        
        # Check user limits
        async with self._user_locks[user_id]:
            user_graphs = await self.storage.list_graphs(user_id)
            if len(user_graphs) >= self.max_graphs_per_user:
                raise LimitExceeded(
                    f"User {user_id} has reached the limit of "
                    f"{self.max_graphs_per_user} graphs"
                )
        
        # Create graph with proper locking
        async with self._graph_lock(user_id, graph_id):
            # Check if already exists
            existing = await self.storage.load_graph(user_id, graph_id)
            if existing is not None:
                raise ValueError(f"Graph {graph_id} already exists")
            
            # Create new graph
            graph = self._create_graph_instance(graph_type, **kwargs)
            
            # Save atomically
            async with self.storage.transaction() as tx:
                await self.storage.save_graph(user_id, graph_id, graph, tx)
                # Log creation
                await self._audit_log(
                    user_id, "create_graph", 
                    {"graph_id": graph_id, "type": graph_type}, tx
                )
            
            return {
                "status": "success",
                "graph_id": graph_id,
                "type": graph_type,
                "created_at": datetime.utcnow().isoformat()
            }
```

### 1.4 Resource Management & Monitoring

```python
# src/networkx_mcp/monitoring/resource_manager.py
import psutil
import resource
import gc
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ResourceLimits:
    max_memory_mb: int = 1000
    max_cpu_percent: float = 80.0
    max_graphs: int = 1000
    max_graph_nodes: int = 1_000_000
    max_graph_edges: int = 10_000_000
    max_operation_time_s: int = 60

class ResourceManager:
    """Enforce resource limits and track usage."""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self.metrics = defaultdict(float)
        self._start_monitoring()
    
    def check_memory(self) -> bool:
        """Check if we're within memory limits."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > self.limits.max_memory_mb:
            # Try garbage collection
            gc.collect()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.limits.max_memory_mb:
                raise MemoryLimitExceeded(
                    f"Memory usage {memory_mb:.1f}MB exceeds "
                    f"limit {self.limits.max_memory_mb}MB"
                )
        
        self.metrics['memory_mb'] = memory_mb
        return True
    
    def check_graph_limits(self, graph: nx.Graph) -> bool:
        """Validate graph size."""
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
        if num_nodes > self.limits.max_graph_nodes:
            raise GraphTooLarge(
                f"Graph has {num_nodes:,} nodes, "
                f"limit is {self.limits.max_graph_nodes:,}"
            )
        
        if num_edges > self.limits.max_graph_edges:
            raise GraphTooLarge(
                f"Graph has {num_edges:,} edges, "
                f"limit is {self.limits.max_graph_edges:,}"
            )
        
        return True
    
    async def with_timeout(self, coro, operation: str):
        """Execute with timeout and resource tracking."""
        start_time = time.time()
        
        try:
            self.check_memory()
            result = await asyncio.wait_for(
                coro, 
                timeout=self.limits.max_operation_time_s
            )
            
            # Track success
            elapsed = time.time() - start_time
            self.metrics[f'{operation}_duration_s'] = elapsed
            self.metrics[f'{operation}_success_count'] += 1
            
            return result
            
        except asyncio.TimeoutError:
            self.metrics[f'{operation}_timeout_count'] += 1
            raise OperationTimeout(
                f"Operation '{operation}' timed out after "
                f"{self.limits.max_operation_time_s}s"
            )
        except Exception as e:
            self.metrics[f'{operation}_error_count'] += 1
            raise
```

### 1.5 Audit Logging & Compliance

```python
# src/networkx_mcp/audit/audit_logger.py
import json
from datetime import datetime
from typing import Any, Dict, Optional

class AuditLogger:
    """Comprehensive audit logging for compliance."""
    
    def __init__(self, storage: StorageBackend):
        self.storage = storage
        
    async def log(self, user_id: str, action: str, 
                  details: Dict[str, Any], 
                  tx: Optional[Transaction] = None):
        """Log user action with full context."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,
            "details": details,
            "ip_address": self._get_client_ip(),
            "session_id": self._get_session_id(),
            "request_id": self._get_request_id(),
            "server_version": self._get_version()
        }
        
        # Add to audit log
        key = f"audit:{datetime.utcnow().strftime('%Y%m%d')}:{user_id}"
        if tx:
            await tx.append(key, json.dumps(entry))
        else:
            await self.storage.append_to_list(key, json.dumps(entry))
        
        # Alert on suspicious activity
        if self._is_suspicious(action, details):
            await self._alert_security_team(entry)
    
    def _is_suspicious(self, action: str, details: Dict[str, Any]) -> bool:
        """Detect suspicious patterns."""
        suspicious_patterns = [
            # Rapid deletion
            (action == "delete_graph" and 
             details.get("graphs_deleted_today", 0) > 10),
            # Large data export
            (action == "export_graph" and 
             details.get("size_mb", 0) > 100),
            # Unusual file paths
            (action in ["import_graph", "export_graph"] and 
             "../" in str(details.get("filepath", ""))),
            # Failed auth attempts
            (action == "auth_failed" and 
             details.get("attempts", 0) > 5)
        ]
        
        return any(suspicious_patterns)
```

## 🟡 PHASE 2: ARCHITECTURE & TESTING (Week 3-4)

### 2.1 Modular Architecture

```python
# src/networkx_mcp/server_v2.py
"""Refactored modular server."""

from fastmcp import FastMCP
from .handlers import (
    GraphHandlers, AlgorithmHandlers, VisualizationHandlers,
    AnalysisHandlers, IntegrationHandlers, HealthHandlers
)
from .middleware import (
    AuthMiddleware, RateLimitMiddleware, MonitoringMiddleware,
    ErrorHandlingMiddleware, RequestIDMiddleware
)
from .services import GraphService, StorageService
from .config import Settings

class NetworkXMCPServer:
    """Clean, modular server implementation."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.mcp = FastMCP(
            "NetworkX Graph Analysis Server",
            version=settings.version
        )
        
        # Initialize services
        self.storage = StorageService(settings)
        self.graph_service = GraphService(self.storage, settings)
        
        # Initialize handlers
        self.graph_handlers = GraphHandlers(self.graph_service)
        self.algorithm_handlers = AlgorithmHandlers(self.graph_service)
        self.health_handlers = HealthHandlers(self.storage, self.graph_service)
        
        # Setup middleware pipeline
        self._setup_middleware()
        
        # Register all handlers
        self._register_handlers()
    
    def _setup_middleware(self):
        """Configure middleware in correct order."""
        # Order matters! 
        self.mcp.add_middleware(RequestIDMiddleware())  # First - add request ID
        self.mcp.add_middleware(MonitoringMiddleware())  # Track all requests
        self.mcp.add_middleware(ErrorHandlingMiddleware())  # Catch all errors
        self.mcp.add_middleware(AuthMiddleware(self.settings))  # Authenticate
        self.mcp.add_middleware(RateLimitMiddleware(self.settings))  # Rate limit
    
    def _register_handlers(self):
        """Register all MCP tools with proper decorators."""
        # Graph operations
        self.mcp.tool()(self.graph_handlers.create_graph)
        self.mcp.tool()(self.graph_handlers.delete_graph)
        self.mcp.tool()(self.graph_handlers.add_nodes)
        self.mcp.tool()(self.graph_handlers.add_edges)
        
        # Health endpoints (no auth required)
        self.mcp.custom_route("/health", ["GET"])(self.health_handlers.health)
        self.mcp.custom_route("/ready", ["GET"])(self.health_handlers.ready)
        self.mcp.custom_route("/metrics", ["GET"])(self.health_handlers.metrics)
    
    def run(self, transport: str = "stdio", **kwargs):
        """Run server with specified transport."""
        self.mcp.run(transport=transport, **kwargs)
```

### 2.2 Comprehensive Testing

```python
# tests/integration/test_production_scenarios.py
import pytest
import asyncio
from concurrent.futures import ThreadPoolExecutor

class TestProductionScenarios:
    """Test real-world production scenarios."""
    
    @pytest.mark.asyncio
    async def test_concurrent_graph_creation(self, server):
        """Test race conditions in graph creation."""
        graph_id = "concurrent_test"
        user_id = "test_user"
        
        # Try to create same graph from multiple tasks
        async def create_graph():
            try:
                return await server.create_graph(user_id, graph_id, "Graph")
            except Exception as e:
                return e
        
        # Launch 10 concurrent attempts
        tasks = [create_graph() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Exactly one should succeed
        successes = [r for r in results if isinstance(r, dict)]
        failures = [r for r in results if isinstance(r, Exception)]
        
        assert len(successes) == 1
        assert len(failures) == 9
        assert all("already exists" in str(e) for e in failures)
    
    @pytest.mark.asyncio
    async def test_memory_exhaustion_protection(self, server):
        """Test protection against memory exhaustion."""
        # Try to create a massive graph
        with pytest.raises(GraphTooLarge):
            await server.create_graph("user1", "huge_graph", "Graph")
            # Add 2 million nodes
            await server.add_nodes("user1", "huge_graph", 
                                 list(range(2_000_000)))
    
    @pytest.mark.asyncio
    async def test_operation_timeout(self, server):
        """Test long-running operations are terminated."""
        # Create graph with pathological structure for algorithms
        await server.create_graph("user1", "complex", "Graph")
        
        # Create complete graph with 1000 nodes (499,500 edges)
        nodes = list(range(1000))
        edges = [(i, j) for i in nodes for j in nodes if i < j]
        
        await server.add_nodes("user1", "complex", nodes)
        await server.add_edges("user1", "complex", edges)
        
        # Try expensive operation - should timeout
        with pytest.raises(OperationTimeout):
            # All-pairs shortest path on complete graph
            await server.all_pairs_shortest_path("user1", "complex")
```

## 🟢 PHASE 3: PRODUCTION INFRASTRUCTURE (Week 5-6)

### 3.1 Docker & Kubernetes

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: networkx-mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: networkx-mcp
  template:
    metadata:
      labels:
        app: networkx-mcp
    spec:
      containers:
      - name: server
        image: networkx-mcp:latest
        ports:
        - containerPort: 8765
        env:
        - name: STORAGE_BACKEND
          value: "redis"
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8765
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8765
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 3.2 Monitoring & Alerting

```python
# src/networkx_mcp/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info
import time

# Define metrics
graph_operations = Counter(
    'networkx_graph_operations_total',
    'Total number of graph operations',
    ['operation', 'status']
)

operation_duration = Histogram(
    'networkx_operation_duration_seconds',
    'Operation duration in seconds',
    ['operation'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

active_graphs = Gauge(
    'networkx_active_graphs',
    'Number of active graphs',
    ['user_id']
)

memory_usage = Gauge(
    'networkx_memory_usage_bytes',
    'Memory usage in bytes'
)

class MetricsCollector:
    """Collect and expose metrics."""
    
    def track_operation(self, operation: str):
        """Decorator to track operations."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = await func(*args, **kwargs)
                    graph_operations.labels(
                        operation=operation, 
                        status='success'
                    ).inc()
                    return result
                except Exception as e:
                    graph_operations.labels(
                        operation=operation,
                        status='error'
                    ).inc()
                    raise
                finally:
                    duration = time.time() - start
                    operation_duration.labels(
                        operation=operation
                    ).observe(duration)
            return wrapper
        return decorator
```

## 📊 **Implementation Priority Matrix**

| Priority | Component | Risk if Not Fixed | Effort | Impact |
|----------|-----------|-------------------|---------|---------|
| 🔴 P0 | State Persistence | Data loss | High | Critical |
| 🔴 P0 | Security Validation | System compromise | Medium | Critical |
| 🔴 P0 | Resource Limits | Service crash | Medium | Critical |
| 🟡 P1 | Monitoring | Blind operations | Medium | High |
| 🟡 P1 | Modular Architecture | Tech debt | High | High |
| 🟡 P1 | Integration Tests | Undetected bugs | Medium | High |
| 🟢 P2 | Performance Optimization | Slow operations | Medium | Medium |
| 🟢 P2 | Client Libraries | Poor adoption | Low | Medium |

## 🚀 **Migration Path**

1. **Week 1-2**: Implement persistence and security (P0 items)
2. **Week 3-4**: Refactor architecture and add tests
3. **Week 5-6**: Production infrastructure
4. **Week 7-8**: Performance optimization and documentation

## ⚠️ **Backwards Compatibility Strategy**

```python
# Support both old and new APIs during migration
@mcp.tool()
async def create_graph(graph_id: str, graph_type: str = "Graph", 
                      params: Optional[Dict[str, Any]] = None,
                      # NEW: Added for compatibility
                      user_id: Optional[str] = None):
    """Maintain compatibility while adding security."""
    # Default to 'anonymous' for old clients
    user_id = user_id or "anonymous"
    
    # Delegate to new secure implementation
    return await graph_manager_v2.create_graph(
        user_id=user_id,
        graph_id=graph_id,
        graph_type=graph_type,
        **params or {}
    )
```

## 🎯 **Success Metrics**

- **Security**: 0 critical vulnerabilities in security scan
- **Reliability**: 99.9% uptime over 30 days
- **Performance**: 95th percentile latency < 100ms
- **Scalability**: Support 1000 concurrent users
- **Maintainability**: 90%+ test coverage

The key is to **fix the foundations first**. A beautiful house on a rotten foundation will collapse.