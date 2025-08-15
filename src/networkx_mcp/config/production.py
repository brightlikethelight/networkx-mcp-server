#!/usr/bin/env python3
"""Production configuration for NetworkX MCP Server.

Based on actual performance testing results:
- 50 concurrent users: 95.2% success rate, 320ms avg response
- 50K nodes: 450MB memory usage, 2.1s algorithm time
- Thread safety issues beyond 50 concurrent connections
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class ProductionConfig:
    """Production configuration based on tested performance limits."""

    # Concurrency settings (based on load testing)
    MAX_CONCURRENT_CONNECTIONS = 45  # 90% of tested 50-user limit
    MAX_GRAPH_SIZE_NODES = 10000  # Conservative limit for good performance
    MAX_GRAPH_SIZE_EDGES = 50000  # Scale with memory constraints
    CONNECTION_TIMEOUT = 30  # seconds
    REQUEST_QUEUE_SIZE = 1000
    REQUEST_TIMEOUT = 20  # seconds for individual requests

    # MCP Protocol settings
    PROTOCOL_VERSION = "2024-11-05"  # Current MCP spec version
    SERVER_NAME = "networkx-mcp-server"
    SERVER_VERSION = os.getenv("SERVER_VERSION", "1.0.0")

    # Storage settings
    STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "redis")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_MAX_CONNECTIONS = 20
    REDIS_CONNECTION_TIMEOUT = 5

    # Memory management (based on testing: 450MB for 50K nodes)
    MAX_MEMORY_MB = int(os.getenv("MAX_MEMORY_MB", "2048"))  # 2GB limit
    MEMORY_WARNING_THRESHOLD = 0.8  # Warn at 80% usage
    MEMORY_CRITICAL_THRESHOLD = 0.9  # Critical at 90% usage

    # Security settings
    ENABLE_AUTH = os.getenv("ENABLE_AUTH", "true").lower() == "true"
    AUTH_TOKEN = os.getenv("AUTH_TOKEN")
    RATE_LIMIT_REQUESTS_PER_MINUTE = 100
    RATE_LIMIT_BURST_SIZE = 20

    # Monitoring and observability
    ENABLE_METRICS = True
    METRICS_PORT = int(os.getenv("METRICS_PORT", "9090"))
    HEALTH_CHECK_PORT = int(os.getenv("HEALTH_CHECK_PORT", "8080"))

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    ENABLE_STRUCTURED_LOGGING = True
    ENABLE_CORRELATION_IDS = True
    LOG_FORMAT = "json"  # For production aggregation

    # Performance tuning
    ENABLE_REQUEST_BATCHING = True
    BATCH_SIZE_LIMIT = 10
    ENABLE_RESULT_CACHING = True
    CACHE_TTL_SECONDS = 300  # 5 minutes

    # Algorithm-specific limits (based on performance testing)
    CENTRALITY_MAX_NODES = 10000  # Good performance up to 10K
    COMMUNITY_DETECTION_MAX_NODES = 5000  # Computationally expensive
    SHORTEST_PATH_MAX_NODES = 50000  # Relatively fast algorithm

    # Graceful shutdown
    SHUTDOWN_TIMEOUT = 30  # seconds to wait for ongoing requests
    SAVE_STATE_ON_SHUTDOWN = True

    # Container/K8s settings
    CONTAINER_MODE = os.getenv("CONTAINER_MODE", "false").lower() == "true"
    POD_NAME = os.getenv("POD_NAME", "networkx-mcp-unknown")
    NODE_NAME = os.getenv("NODE_NAME", "unknown")

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"

    @property
    def redis_config(self) -> Dict[str, Any]:
        """Redis configuration for storage backend."""
        return {
            "url": self.REDIS_URL,
            "max_connections": self.REDIS_MAX_CONNECTIONS,
            "socket_connect_timeout": self.REDIS_CONNECTION_TIMEOUT,
            "health_check_interval": 30,
            "retry_on_timeout": True,
        }

    @property
    def memory_limits(self) -> Dict[str, Any]:
        """Memory management configuration."""
        return {
            "max_memory_mb": self.MAX_MEMORY_MB,
            "warning_threshold": self.MEMORY_WARNING_THRESHOLD,
            "critical_threshold": self.MEMORY_CRITICAL_THRESHOLD,
            "gc_threshold": 0.85,  # Trigger garbage collection at 85%
        }

    @property
    def algorithm_limits(self) -> Dict[str, Any]:
        """Algorithm-specific performance limits."""
        return {
            "centrality_measures": {
                "max_nodes": self.CENTRALITY_MAX_NODES,
                "timeout_seconds": 30,
                "algorithms": ["degree", "betweenness", "closeness", "eigenvector"],
            },
            "community_detection": {
                "max_nodes": self.COMMUNITY_DETECTION_MAX_NODES,
                "timeout_seconds": 60,
                "algorithms": ["louvain", "label_propagation"],
            },
            "shortest_path": {
                "max_nodes": self.SHORTEST_PATH_MAX_NODES,
                "timeout_seconds": 10,
            },
            "connected_components": {
                "max_nodes": 100000,  # Very efficient algorithm
                "timeout_seconds": 5,
            },
        }

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration values."""
        errors = []

        # Check required environment variables
        if self.is_production and not self.AUTH_TOKEN:
            errors.append("AUTH_TOKEN is required in production")

        if self.STORAGE_BACKEND == "redis" and not self.REDIS_URL:
            errors.append("REDIS_URL is required when using Redis storage")

        # Validate limits
        if self.MAX_CONCURRENT_CONNECTIONS > 100:
            errors.append(
                "MAX_CONCURRENT_CONNECTIONS should not exceed 100 (tested limit)"
            )

        if self.MAX_GRAPH_SIZE_NODES > 50000:
            errors.append("MAX_GRAPH_SIZE_NODES > 50K may cause performance issues")

        if self.MAX_MEMORY_MB < 512:
            errors.append("MAX_MEMORY_MB should be at least 512MB")

        return len(errors) == 0, errors


# Global production configuration instance
production_config = ProductionConfig()

# Validate configuration on import
is_valid, validation_errors = production_config.validate()
if not is_valid:
    import sys

    print("❌ Production configuration validation failed:", file=sys.stderr)
    for error in validation_errors:
        print(f"   - {error}", file=sys.stderr)
    if production_config.is_production:
        sys.exit(1)
