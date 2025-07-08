#!/usr/bin/env python3
"""Production feature flag configurations for gradual rollout.

Based on performance testing and production requirements.
"""

import functools
import time
from datetime import datetime, timedelta
from .feature_flags import FeatureFlag, FeatureStatus, RolloutStrategy

# Production feature flags with performance-based gating
PRODUCTION_FEATURE_FLAGS = {
    
    # Performance optimizations based on testing
    "use_approximate_algorithms": FeatureFlag(
        name="use_approximate_algorithms",
        description="Use approximate algorithms for large graphs (>10K nodes) to improve performance",
        status=FeatureStatus.ROLLOUT,
        rollout_strategy=RolloutStrategy.PERFORMANCE_GATED,
        rollout_percentage=25.0,  # Start with 25% rollout
        category="performance",
        tags=["algorithms", "optimization", "large-graphs"],
        
        # Performance gates based on testing (50K nodes = 2.1s, want <1s)
        max_p95_latency_ms=1000,  # Target 1s P95 for large graphs
        max_error_rate=0.02,  # Stricter 2% error rate for new algorithms
        min_success_rate=0.98,  # Higher success rate requirement
        
        # Gradual rollout over 48 hours
        rollout_start_time=datetime.now(),
        rollout_duration_hours=48,
        
        enabled_environments={"staging", "production"},
        beta_users={"admin", "performance-tester", "algorithm-researcher"}
    ),
    
    # Graph result caching to reduce load
    "enable_graph_caching": FeatureFlag(
        name="enable_graph_caching",
        description="Cache algorithm results to improve performance for repeated operations",
        status=FeatureStatus.ROLLOUT,
        rollout_strategy=RolloutStrategy.PERCENTAGE,
        rollout_percentage=50.0,  # 50% rollout
        category="performance",
        tags=["caching", "performance", "memory"],
        
        # Conservative memory impact monitoring
        max_error_rate=0.03,
        max_p95_latency_ms=500,  # Should reduce latency
        
        enabled_environments={"staging", "production"},
        beta_users={"admin", "performance-tester"}
    ),
    
    # Connection management optimization
    "enhanced_connection_pooling": FeatureFlag(
        name="enhanced_connection_pooling",
        description="Advanced connection pooling to handle 50+ concurrent users more efficiently",
        status=FeatureStatus.BETA,
        rollout_strategy=RolloutStrategy.USER_BASED,
        category="infrastructure",
        tags=["connections", "scalability", "performance"],
        
        # Based on testing: 50 users = 95.2% success rate
        max_error_rate=0.048,  # Slightly better than baseline
        min_success_rate=0.96,  # Improve on 95.2% baseline
        
        enabled_environments={"staging", "production"},
        beta_users={"admin", "load-tester", "scale-tester"}
    ),
    
    # Remote HTTP mode (new feature)
    "enable_remote_mode": FeatureFlag(
        name="enable_remote_mode",
        description="Enable HTTP transport mode for remote MCP access",
        status=FeatureStatus.CANARY,  # Start with canary only
        rollout_strategy=RolloutStrategy.RING_BASED,
        rollout_percentage=10.0,  # Very conservative
        category="transport",
        tags=["http", "remote", "authentication"],
        requires_restart=True,
        
        # Strict performance requirements for new transport
        max_error_rate=0.02,
        max_p95_latency_ms=800,  # HTTP should be competitive with stdio
        min_success_rate=0.98,
        
        enabled_environments={"staging"},  # Not in prod yet
        beta_users={"admin", "remote-tester", "integration-tester"}
    ),
    
    # Memory management improvements
    "aggressive_memory_management": FeatureFlag(
        name="aggressive_memory_management",
        description="Proactive memory cleanup for large graph operations (>5K nodes)",
        status=FeatureStatus.EXPERIMENTAL,
        rollout_strategy=RolloutStrategy.TIME_BASED,
        rollout_percentage=20.0,
        category="memory",
        tags=["memory", "cleanup", "large-graphs"],
        
        # Memory-focused performance gates
        max_error_rate=0.05,
        max_p95_latency_ms=1500,  # May add overhead
        
        rollout_start_time=datetime.now() + timedelta(hours=24),  # Start tomorrow
        rollout_duration_hours=72,  # 3-day rollout
        
        enabled_environments={"staging", "production"},
        beta_users={"admin", "memory-tester"}
    ),
    
    # Algorithm sampling for very large graphs
    "graph_sampling_mode": FeatureFlag(
        name="graph_sampling_mode", 
        description="Use statistical sampling for graphs >25K nodes to improve performance",
        status=FeatureStatus.DISABLED,  # Not ready yet
        rollout_strategy=RolloutStrategy.PERFORMANCE_GATED,
        rollout_percentage=0.0,
        category="algorithms",
        tags=["sampling", "large-graphs", "approximation"],
        
        # Requires careful validation
        max_error_rate=0.10,  # Higher tolerance for approximation
        max_p95_latency_ms=500,  # Should be much faster
        
        enabled_environments={"staging"},
        beta_users={"admin", "algorithm-researcher"}
    ),
    
    # Enhanced monitoring and tracing
    "detailed_performance_tracing": FeatureFlag(
        name="detailed_performance_tracing",
        description="Enhanced OpenTelemetry tracing with detailed algorithm performance data",
        status=FeatureStatus.ROLLOUT,
        rollout_strategy=RolloutStrategy.PERCENTAGE,
        rollout_percentage=75.0,  # High rollout for monitoring
        category="observability",
        tags=["tracing", "monitoring", "performance"],
        
        # Should not impact performance significantly
        max_error_rate=0.05,
        max_p95_latency_ms=1000,  # Allow some overhead for tracing
        
        enabled_environments={"staging", "production"},
        beta_users={"admin", "ops-team", "monitoring-team"}
    ),
    
    # Security enhancements
    "strict_input_validation": FeatureFlag(
        name="strict_input_validation",
        description="Enhanced input validation with detailed error reporting",
        status=FeatureStatus.ENABLED,  # Always on for security
        category="security",
        tags=["validation", "security", "input"],
        
        # Security features should not degrade performance
        max_error_rate=0.05,
        max_p95_latency_ms=1200,
        
        enabled_environments={"staging", "production"}
    ),
    
    # Concurrent user limit management
    "dynamic_connection_limits": FeatureFlag(
        name="dynamic_connection_limits",
        description="Dynamically adjust connection limits based on system performance",
        status=FeatureStatus.BETA,
        rollout_strategy=RolloutStrategy.PERFORMANCE_GATED,
        rollout_percentage=30.0,
        category="scalability",
        tags=["connections", "limits", "adaptive"],
        
        # Should improve system stability
        max_error_rate=0.03,
        min_success_rate=0.97,
        
        enabled_environments={"staging", "production"},
        beta_users={"admin", "ops-team", "scale-tester"}
    ),
    
    # Graph persistence optimization
    "optimized_graph_serialization": FeatureFlag(
        name="optimized_graph_serialization",
        description="Improved graph serialization for Redis backend storage",
        status=FeatureStatus.EXPERIMENTAL,
        rollout_strategy=RolloutStrategy.USER_BASED,
        category="storage",
        tags=["serialization", "redis", "performance"],
        
        # Storage operations performance
        max_error_rate=0.02,
        max_p95_latency_ms=800,
        
        enabled_environments={"staging"},
        beta_users={"admin", "storage-tester"},
        
        dependencies=["enable_graph_caching"]  # Requires caching
    )
}


class ProductionFeatureManager:
    """Production feature flag manager with performance monitoring."""
    
    def __init__(self):
        self.flags = PRODUCTION_FEATURE_FLAGS.copy()
        self._performance_metrics = {}
        
    def is_enabled(self, flag_name: str, user_id: str = None, environment: str = None) -> bool:
        """Check if a production feature is enabled."""
        if flag_name not in self.flags:
            return False
            
        flag = self.flags[flag_name]
        
        # Check dependencies
        for dep in flag.dependencies:
            if not self.is_enabled(dep, user_id, environment):
                return False
                
        # Check conflicts
        for conflict in flag.conflicts:
            if self.is_enabled(conflict, user_id, environment):
                return False
        
        return flag.is_enabled(user_id, environment)
    
    def record_feature_usage(self, flag_name: str, success: bool = True, 
                           latency_ms: float = None, error_type: str = None):
        """Record feature usage with performance metrics."""
        if flag_name not in self.flags:
            return
            
        flag = self.flags[flag_name]
        flag.record_usage(success)
        
        # Track performance metrics for gating decisions
        if flag_name not in self._performance_metrics:
            self._performance_metrics[flag_name] = {
                'total_calls': 0,
                'errors': 0,
                'latencies': [],
                'error_types': {}
            }
            
        metrics = self._performance_metrics[flag_name]
        metrics['total_calls'] += 1
        
        if not success:
            metrics['errors'] += 1
            if error_type:
                metrics['error_types'][error_type] = metrics['error_types'].get(error_type, 0) + 1
                
        if latency_ms is not None:
            metrics['latencies'].append(latency_ms)
            # Keep only last 1000 measurements
            if len(metrics['latencies']) > 1000:
                metrics['latencies'] = metrics['latencies'][-1000:]
    
    def get_feature_metrics(self, flag_name: str) -> dict:
        """Get performance metrics for a feature."""
        if flag_name not in self._performance_metrics:
            return {}
            
        metrics = self._performance_metrics[flag_name]
        
        # Calculate statistics
        error_rate = metrics['errors'] / max(metrics['total_calls'], 1)
        
        latencies = metrics['latencies']
        if latencies:
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
            avg_latency = sum(latencies) / len(latencies)
        else:
            p95_latency = 0
            avg_latency = 0
            
        return {
            'total_calls': metrics['total_calls'],
            'error_rate': error_rate,
            'success_rate': 1 - error_rate,
            'p95_latency_ms': p95_latency,
            'avg_latency_ms': avg_latency,
            'error_types': metrics['error_types'].copy()
        }
    
    def update_rollout_percentage(self, flag_name: str, percentage: float):
        """Update rollout percentage for gradual deployment."""
        if flag_name in self.flags:
            self.flags[flag_name].rollout_percentage = min(100.0, max(0.0, percentage))
            self.flags[flag_name].updated_at = datetime.now()
    
    def get_rollout_status(self) -> dict:
        """Get status of all rollouts for monitoring."""
        rollout_status = {}
        
        for name, flag in self.flags.items():
            if flag.status in [FeatureStatus.ROLLOUT, FeatureStatus.CANARY, FeatureStatus.BETA]:
                rollout_status[name] = {
                    'status': flag.status.value,
                    'rollout_percentage': flag.rollout_percentage,
                    'strategy': flag.rollout_strategy.value,
                    'usage_count': flag.usage_count,
                    'error_count': flag.error_count,
                    'metrics': self.get_feature_metrics(name)
                }
                
        return rollout_status


# Global production feature manager
_production_manager = None


def get_production_feature_manager() -> ProductionFeatureManager:
    """Get the global production feature manager."""
    global _production_manager
    if _production_manager is None:
        _production_manager = ProductionFeatureManager()
    return _production_manager


def feature_flag(flag_name: str):
    """Decorator for feature-flagged functionality."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_production_feature_manager()
            
            # Extract user_id from args/kwargs if available
            user_id = kwargs.get('user_id') or getattr(args[0] if args else None, 'user_id', None)
            
            if not manager.is_enabled(flag_name, user_id):
                # Feature disabled - return None or default behavior
                return None
                
            # Feature enabled - execute with monitoring
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                
                # Record success
                latency_ms = (time.time() - start_time) * 1000
                manager.record_feature_usage(flag_name, True, latency_ms)
                
                return result
                
            except Exception as e:
                # Record failure
                latency_ms = (time.time() - start_time) * 1000
                manager.record_feature_usage(flag_name, False, latency_ms, type(e).__name__)
                
                raise
                
        return wrapper
    return decorator