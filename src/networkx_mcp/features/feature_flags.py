"""Production-ready feature flag implementation with gradual rollout capabilities.

Supports canary deployments, user-based rollouts, and performance monitoring.
"""

import functools
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from ..config.production import production_config

logger = logging.getLogger(__name__)


class FeatureStatus(Enum):
    """Feature flag status for production deployments."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    EXPERIMENTAL = "experimental"  # Enabled but logs usage
    BETA = "beta"  # Enabled with warnings
    CANARY = "canary"  # Enabled for canary deployments only
    ROLLOUT = "rollout"  # Gradual percentage-based rollout
    DEPRECATED = "deprecated"  # Enabled but warns about removal


class RolloutStrategy(Enum):
    """Rollout strategy for gradual feature releases."""
    PERCENTAGE = "percentage"  # Simple percentage rollout
    USER_BASED = "user_based"  # Specific user list
    RING_BASED = "ring_based"  # Ring deployment (dev -> staging -> prod)
    TIME_BASED = "time_based"  # Time-window based rollout
    PERFORMANCE_GATED = "performance_gated"  # Performance metrics gated


@dataclass
class FeatureFlag:
    """Production feature flag configuration."""
    
    name: str
    description: str
    status: FeatureStatus = FeatureStatus.DISABLED
    default_enabled: bool = False
    
    # Production rollout settings
    rollout_strategy: RolloutStrategy = RolloutStrategy.PERCENTAGE
    rollout_percentage: float = 0.0  # 0-100 for gradual rollout
    rollout_start_time: Optional[datetime] = None
    rollout_duration_hours: int = 24  # Default 24-hour rollout
    
    # User targeting
    enabled_for_users: Set[str] = field(default_factory=set)
    disabled_for_users: Set[str] = field(default_factory=set)
    beta_users: Set[str] = field(default_factory=set)
    
    # Environment targeting
    enabled_environments: Set[str] = field(default_factory=set)
    
    # Performance gating (based on our testing limits)
    max_error_rate: float = 0.05  # 5% error rate threshold
    max_p95_latency_ms: float = 2000  # 2s P95 latency from testing
    min_success_rate: float = 0.95  # 95% success rate from testing
    
    # Metadata
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Safety and dependencies
    requires_restart: bool = False
    dependencies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    
    # Monitoring and metrics
    usage_count: int = 0
    error_count: int = 0
    last_used: Optional[datetime] = None
    
    def is_enabled(self, user_id: Optional[str] = None, environment: Optional[str] = None) -> bool:
        """Check if feature is enabled with production rollout logic."""
        # Always disabled
        if self.status == FeatureStatus.DISABLED:
            return False
            
        # Always enabled
        if self.status == FeatureStatus.ENABLED:
            return True
            
        # Environment check
        current_env = environment or os.getenv('ENVIRONMENT', 'development')
        if self.enabled_environments and current_env not in self.enabled_environments:
            return False
            
        # User-specific overrides
        if user_id:
            if user_id in self.disabled_for_users:
                return False
            if user_id in self.enabled_for_users:
                return True
            if user_id in self.beta_users and self.status in [FeatureStatus.BETA, FeatureStatus.EXPERIMENTAL]:
                return True
        
        # Canary deployment check
        if self.status == FeatureStatus.CANARY:
            return os.getenv('CANARY_DEPLOYMENT', 'false').lower() == 'true'
            
        # Rollout logic
        if self.status == FeatureStatus.ROLLOUT:
            return self._check_rollout_eligibility(user_id, current_env)
            
        # Experimental and Beta (for beta users)
        if self.status in [FeatureStatus.EXPERIMENTAL, FeatureStatus.BETA]:
            if user_id and user_id in self.beta_users:
                return True
            return self.rollout_percentage > 0 and self._check_percentage_rollout(user_id)
            
        # Deprecated features are enabled but logged
        if self.status == FeatureStatus.DEPRECATED:
            logger.warning(f"Feature {self.name} is deprecated and will be removed")
            return True
            
        return self.default_enabled
    
    def _check_rollout_eligibility(self, user_id: Optional[str], environment: str) -> bool:
        """Check rollout eligibility based on strategy."""
        if self.rollout_strategy == RolloutStrategy.PERCENTAGE:
            return self._check_percentage_rollout(user_id)
            
        elif self.rollout_strategy == RolloutStrategy.USER_BASED:
            return user_id in self.enabled_for_users if user_id else False
            
        elif self.rollout_strategy == RolloutStrategy.RING_BASED:
            return self._check_ring_rollout(environment)
            
        elif self.rollout_strategy == RolloutStrategy.TIME_BASED:
            return self._check_time_rollout()
            
        elif self.rollout_strategy == RolloutStrategy.PERFORMANCE_GATED:
            return self._check_performance_gate()
            
        return False
    
    def _check_percentage_rollout(self, user_id: Optional[str]) -> bool:
        """Check if user is in percentage rollout bucket."""
        if self.rollout_percentage <= 0:
            return False
        if self.rollout_percentage >= 100:
            return True
            
        # Consistent hashing for user bucketing
        if user_id:
            hash_input = f"{self.name}:{user_id}"
        else:
            # Use instance identifier for anonymous users
            hash_input = f"{self.name}:{os.getenv('POD_NAME', 'default')}"
            
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
        bucket = (hash_value % 100) + 1
        
        return bucket <= self.rollout_percentage
    
    def _check_ring_rollout(self, environment: str) -> bool:
        """Check ring-based rollout (dev -> staging -> prod)."""
        ring_order = ['development', 'staging', 'production']
        
        if environment not in ring_order:
            return False
            
        # Enable based on rollout percentage and environment order
        env_index = ring_order.index(environment)
        required_percentage = (env_index + 1) * 33.3  # 33%, 66%, 100%
        
        return self.rollout_percentage >= required_percentage
    
    def _check_time_rollout(self) -> bool:
        """Check time-based rollout."""
        if not self.rollout_start_time:
            return False
            
        elapsed_hours = (datetime.now() - self.rollout_start_time).total_seconds() / 3600
        
        if elapsed_hours >= self.rollout_duration_hours:
            return True
            
        # Linear rollout over time
        progress = elapsed_hours / self.rollout_duration_hours
        return progress * 100 >= self.rollout_percentage
    
    def _check_performance_gate(self) -> bool:
        """Check if performance metrics allow feature activation."""
        try:
            # This would integrate with metrics system
            # For now, return conservative default
            from ..metrics import get_metrics
            metrics = get_metrics()
            
            # Get current metrics snapshot
            snapshot = metrics.get_snapshot()
            
            # Check error rate
            if snapshot.error_rate > self.max_error_rate:
                logger.warning(f"Feature {self.name} gated due to high error rate: {snapshot.error_rate}")
                return False
                
            # Additional performance checks would go here
            return True
            
        except Exception as e:
            logger.error(f"Error checking performance gate for {self.name}: {e}")
            return False
    
    def record_usage(self, success: bool = True):
        """Record feature usage for monitoring."""
        self.usage_count += 1
        self.last_used = datetime.now()
        
        if not success:
            self.error_count += 1
            
        # Log for monitoring
        logger.info(f"Feature {self.name} used", extra={
            'feature_name': self.name,
            'success': success,
            'usage_count': self.usage_count,
            'error_count': self.error_count
        })
        if self.status == FeatureStatus.DISABLED:
            return False
        
        # Check user-specific overrides
        if user_id:
            if user_id in self.disabled_for_users:
                return False
            if user_id in self.enabled_for_users:
                return True
        
        # Check environment
        if environment and self.enabled_environments:
            if environment not in self.enabled_environments:
                return False
        
        # Check rollout percentage
        if 0 < self.rollout_percentage < 100:
            # Simple hash-based rollout
            if user_id:
                user_hash = hash(f"{self.name}:{user_id}") % 100
                return user_hash < self.rollout_percentage
            else:
                # Without user context, use default
                return self.default_enabled
        
        return self.status in [FeatureStatus.ENABLED, FeatureStatus.EXPERIMENTAL, 
                              FeatureStatus.BETA, FeatureStatus.DEPRECATED]


class FeatureFlagManager:
    """Manages feature flags with persistence and hot reload."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.getenv("FEATURE_FLAGS_PATH", "feature_flags.json")
        self.flags: Dict[str, FeatureFlag] = {}
        self._lock = threading.RLock()
        self._change_callbacks: List[Callable[[str, bool], None]] = []
        self._usage_callbacks: List[Callable[[str, Any], None]] = []
        
        # Initialize default flags
        self._initialize_default_flags()
        
        # Load from configuration
        self.load_flags()
        
        # Apply environment variable overrides
        self._apply_env_overrides()
    
    def _initialize_default_flags(self):
        """Initialize default feature flags."""
        default_flags = [
            # ML Features
            FeatureFlag(
                name="ml_graph_embeddings",
                description="Enable graph embedding generation",
                category="machine_learning",
                status=FeatureStatus.DISABLED,
                tags=["ml", "experimental"],
                dependencies=["ml_base_features"]
            ),
            FeatureFlag(
                name="ml_base_features",
                description="Enable basic ML features",
                category="machine_learning",
                status=FeatureStatus.DISABLED,
                tags=["ml"]
            ),
            FeatureFlag(
                name="ml_link_prediction",
                description="Enable link prediction algorithms",
                category="machine_learning",
                status=FeatureStatus.DISABLED,
                tags=["ml", "experimental"],
                dependencies=["ml_base_features"]
            ),
            FeatureFlag(
                name="ml_node_classification",
                description="Enable node classification",
                category="machine_learning",
                status=FeatureStatus.DISABLED,
                tags=["ml", "experimental"],
                dependencies=["ml_base_features"]
            ),
            FeatureFlag(
                name="ml_anomaly_detection",
                description="Enable graph anomaly detection",
                category="machine_learning",
                status=FeatureStatus.DISABLED,
                tags=["ml", "experimental"],
                dependencies=["ml_base_features"]
            ),
            
            # Advanced Algorithms
            FeatureFlag(
                name="gpu_acceleration",
                description="Enable GPU acceleration for algorithms",
                category="performance",
                status=FeatureStatus.DISABLED,
                requires_restart=True,
                tags=["performance", "experimental"]
            ),
            FeatureFlag(
                name="parallel_algorithms",
                description="Enable parallel algorithm execution",
                category="performance",
                status=FeatureStatus.ENABLED,
                tags=["performance"]
            ),
            FeatureFlag(
                name="advanced_community_detection",
                description="Enable advanced community detection algorithms",
                category="algorithms",
                status=FeatureStatus.BETA,
                tags=["algorithms", "beta"]
            ),
            FeatureFlag(
                name="quantum_algorithms",
                description="Enable quantum-inspired graph algorithms",
                category="algorithms",
                status=FeatureStatus.EXPERIMENTAL,
                tags=["algorithms", "experimental", "research"]
            ),
            
            # Visualization
            FeatureFlag(
                name="3d_visualization",
                description="Enable 3D graph visualization",
                category="visualization",
                status=FeatureStatus.ENABLED,
                tags=["visualization"]
            ),
            FeatureFlag(
                name="ar_visualization",
                description="Enable AR/VR visualization support",
                category="visualization",
                status=FeatureStatus.EXPERIMENTAL,
                tags=["visualization", "experimental"]
            ),
            
            # Storage & Performance
            FeatureFlag(
                name="distributed_storage",
                description="Enable distributed graph storage",
                category="storage",
                status=FeatureStatus.DISABLED,
                requires_restart=True,
                tags=["storage", "enterprise"]
            ),
            FeatureFlag(
                name="graph_sharding",
                description="Enable automatic graph sharding",
                category="storage",
                status=FeatureStatus.EXPERIMENTAL,
                tags=["storage", "performance", "experimental"],
                dependencies=["distributed_storage"]
            ),
            FeatureFlag(
                name="incremental_computation",
                description="Enable incremental algorithm computation",
                category="performance",
                status=FeatureStatus.BETA,
                tags=["performance", "beta"]
            ),
            
            # API Features
            FeatureFlag(
                name="graphql_api",
                description="Enable GraphQL API endpoint",
                category="api",
                status=FeatureStatus.EXPERIMENTAL,
                tags=["api", "experimental"]
            ),
            FeatureFlag(
                name="websocket_streaming",
                description="Enable WebSocket streaming for real-time updates",
                category="api",
                status=FeatureStatus.BETA,
                tags=["api", "beta", "realtime"]
            ),
            FeatureFlag(
                name="batch_operations",
                description="Enable batch graph operations",
                category="api",
                status=FeatureStatus.ENABLED,
                tags=["api", "performance"]
            ),
            
            # Security & Monitoring
            FeatureFlag(
                name="advanced_auth",
                description="Enable advanced authentication features",
                category="security",
                status=FeatureStatus.DISABLED,
                tags=["security", "enterprise"]
            ),
            FeatureFlag(
                name="audit_logging",
                description="Enable detailed audit logging",
                category="security",
                status=FeatureStatus.ENABLED,
                tags=["security", "compliance"]
            ),
            FeatureFlag(
                name="performance_profiling",
                description="Enable performance profiling",
                category="monitoring",
                status=FeatureStatus.BETA,
                tags=["monitoring", "performance"]
            ),
            
            # Experimental Features
            FeatureFlag(
                name="graph_diffing",
                description="Enable graph diffing and versioning",
                category="experimental",
                status=FeatureStatus.EXPERIMENTAL,
                tags=["experimental", "versioning"]
            ),
            FeatureFlag(
                name="natural_language_queries",
                description="Enable natural language graph queries",
                category="experimental",
                status=FeatureStatus.EXPERIMENTAL,
                tags=["experimental", "ai", "nlp"],
                dependencies=["ml_base_features"]
            ),
        ]
        
        for flag in default_flags:
            self.flags[flag.name] = flag
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides for feature flags."""
        for flag_name in self.flags:
            env_var = f"FEATURE_{flag_name.upper()}"
            env_value = os.getenv(env_var)
            
            if env_value is not None:
                try:
                    # Support various boolean representations
                    if env_value.lower() in ["true", "1", "yes", "on", "enabled"]:
                        self.flags[flag_name].status = FeatureStatus.ENABLED
                    elif env_value.lower() in ["false", "0", "no", "off", "disabled"]:
                        self.flags[flag_name].status = FeatureStatus.DISABLED
                    elif env_value.lower() in ["experimental", "beta", "deprecated"]:
                        self.flags[flag_name].status = FeatureStatus[env_value.upper()]
                    
                    logger.info(f"Feature flag '{flag_name}' overridden by {env_var}={env_value}")
                except Exception as e:
                    logger.warning(f"Invalid value for {env_var}: {env_value}")
    
    def load_flags(self):
        """Load feature flags from configuration file."""
        if not os.path.exists(self.config_path):
            return
        
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            
            with self._lock:
                for flag_name, flag_data in data.items():
                    if flag_name in self.flags:
                        # Update existing flag
                        flag = self.flags[flag_name]
                        if "status" in flag_data:
                            flag.status = FeatureStatus(flag_data["status"])
                        if "rollout_percentage" in flag_data:
                            flag.rollout_percentage = float(flag_data["rollout_percentage"])
                        if "enabled_for_users" in flag_data:
                            flag.enabled_for_users = set(flag_data["enabled_for_users"])
                        if "disabled_for_users" in flag_data:
                            flag.disabled_for_users = set(flag_data["disabled_for_users"])
                        if "enabled_environments" in flag_data:
                            flag.enabled_environments = set(flag_data["enabled_environments"])
                        flag.updated_at = datetime.now()
                    else:
                        # Create new flag
                        self.flags[flag_name] = FeatureFlag(
                            name=flag_name,
                            description=flag_data.get("description", ""),
                            status=FeatureStatus(flag_data.get("status", "disabled")),
                            **{k: v for k, v in flag_data.items() 
                               if k not in ["name", "description", "status"]}
                        )
            
            logger.info(f"Loaded {len(data)} feature flags from {self.config_path}")
        
        except Exception as e:
            logger.error(f"Failed to load feature flags: {e}")
    
    def save_flags(self):
        """Save feature flags to configuration file."""
        try:
            with self._lock:
                data = {}
                for flag_name, flag in self.flags.items():
                    data[flag_name] = {
                        "description": flag.description,
                        "status": flag.status.value,
                        "category": flag.category,
                        "tags": flag.tags,
                        "rollout_percentage": flag.rollout_percentage,
                        "enabled_for_users": list(flag.enabled_for_users),
                        "disabled_for_users": list(flag.disabled_for_users),
                        "enabled_environments": list(flag.enabled_environments),
                        "requires_restart": flag.requires_restart,
                        "dependencies": flag.dependencies,
                        "conflicts": flag.conflicts,
                    }
            
            # Pretty print for readability
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2, sort_keys=True)
            
            logger.info(f"Saved {len(data)} feature flags to {self.config_path}")
        
        except Exception as e:
            logger.error(f"Failed to save feature flags: {e}")
    
    def is_enabled(self, flag_name: str, user_id: Optional[str] = None, 
                   environment: Optional[str] = None) -> bool:
        """Check if a feature flag is enabled."""
        with self._lock:
            if flag_name not in self.flags:
                logger.warning(f"Unknown feature flag: {flag_name}")
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
            
            enabled = flag.is_enabled(user_id, environment)
            
            # Log experimental/beta usage
            if enabled and flag.status in [FeatureStatus.EXPERIMENTAL, FeatureStatus.BETA]:
                logger.info(f"{flag.status.value.upper()} feature '{flag_name}' is being used")
            
            # Log deprecated usage
            if enabled and flag.status == FeatureStatus.DEPRECATED:
                logger.warning(f"DEPRECATED feature '{flag_name}' is being used and will be removed")
            
            # Notify usage callbacks
            for callback in self._usage_callbacks:
                try:
                    callback(flag_name, {"enabled": enabled, "user_id": user_id})
                except Exception as e:
                    logger.error(f"Usage callback error: {e}")
            
            return enabled
    
    def set_enabled(self, flag_name: str, enabled: bool, 
                   save: bool = True, notify: bool = True) -> bool:
        """Enable or disable a feature flag."""
        with self._lock:
            if flag_name not in self.flags:
                logger.error(f"Unknown feature flag: {flag_name}")
                return False
            
            flag = self.flags[flag_name]
            old_status = flag.status
            
            # Check if change requires restart
            if flag.requires_restart:
                logger.warning(f"Feature '{flag_name}' requires restart to take effect")
            
            # Update status
            if enabled:
                if flag.status == FeatureStatus.DISABLED:
                    flag.status = FeatureStatus.ENABLED
            else:
                flag.status = FeatureStatus.DISABLED
            
            flag.updated_at = datetime.now()
            
            # Save if requested
            if save:
                self.save_flags()
            
            # Notify callbacks if requested
            if notify and old_status != flag.status:
                for callback in self._change_callbacks:
                    try:
                        callback(flag_name, enabled)
                    except Exception as e:
                        logger.error(f"Change callback error: {e}")
            
            logger.info(f"Feature '{flag_name}' {'enabled' if enabled else 'disabled'}")
            return True
    
    def set_rollout_percentage(self, flag_name: str, percentage: float) -> bool:
        """Set gradual rollout percentage for a feature."""
        with self._lock:
            if flag_name not in self.flags:
                return False
            
            self.flags[flag_name].rollout_percentage = max(0, min(100, percentage))
            self.flags[flag_name].updated_at = datetime.now()
            self.save_flags()
            
            logger.info(f"Feature '{flag_name}' rollout set to {percentage}%")
            return True
    
    def add_user_override(self, flag_name: str, user_id: str, enabled: bool) -> bool:
        """Add user-specific override for a feature."""
        with self._lock:
            if flag_name not in self.flags:
                return False
            
            flag = self.flags[flag_name]
            if enabled:
                flag.enabled_for_users.add(user_id)
                flag.disabled_for_users.discard(user_id)
            else:
                flag.disabled_for_users.add(user_id)
                flag.enabled_for_users.discard(user_id)
            
            flag.updated_at = datetime.now()
            self.save_flags()
            return True
    
    def get_all_flags(self) -> Dict[str, Dict[str, Any]]:
        """Get all feature flags with their current status."""
        with self._lock:
            result = {}
            for name, flag in self.flags.items():
                result[name] = {
                    "description": flag.description,
                    "status": flag.status.value,
                    "category": flag.category,
                    "tags": flag.tags,
                    "enabled": flag.status != FeatureStatus.DISABLED,
                    "requires_restart": flag.requires_restart,
                    "rollout_percentage": flag.rollout_percentage,
                    "dependencies": flag.dependencies,
                    "conflicts": flag.conflicts,
                    "updated_at": flag.updated_at.isoformat()
                }
            return result
    
    def get_flags_by_category(self, category: str) -> Dict[str, FeatureFlag]:
        """Get all flags in a specific category."""
        with self._lock:
            return {
                name: flag for name, flag in self.flags.items()
                if flag.category == category
            }
    
    def get_flags_by_tag(self, tag: str) -> Dict[str, FeatureFlag]:
        """Get all flags with a specific tag."""
        with self._lock:
            return {
                name: flag for name, flag in self.flags.items()
                if tag in flag.tags
            }
    
    def add_change_callback(self, callback: Callable[[str, bool], None]):
        """Add callback for feature flag changes."""
        self._change_callbacks.append(callback)
    
    def add_usage_callback(self, callback: Callable[[str, Any], None]):
        """Add callback for feature flag usage tracking."""
        self._usage_callbacks.append(callback)
    
    def validate_dependencies(self) -> List[str]:
        """Validate all feature flag dependencies."""
        errors = []
        with self._lock:
            for name, flag in self.flags.items():
                # Check dependencies exist
                for dep in flag.dependencies:
                    if dep not in self.flags:
                        errors.append(f"Flag '{name}' depends on unknown flag '{dep}'")
                
                # Check conflicts exist
                for conflict in flag.conflicts:
                    if conflict not in self.flags:
                        errors.append(f"Flag '{name}' conflicts with unknown flag '{conflict}'")
                
                # Check circular dependencies
                visited = set()
                def check_circular(flag_name, path):
                    if flag_name in path:
                        errors.append(f"Circular dependency detected: {' -> '.join(path + [flag_name])}")
                        return
                    if flag_name in visited:
                        return
                    visited.add(flag_name)
                    if flag_name in self.flags:
                        for dep in self.flags[flag_name].dependencies:
                            check_circular(dep, path + [flag_name])
                
                check_circular(name, [])
        
        return errors


# Global feature flag manager
_manager: Optional[FeatureFlagManager] = None


def get_flag_manager() -> FeatureFlagManager:
    """Get the global feature flag manager."""
    global _manager
    if _manager is None:
        _manager = FeatureFlagManager()
    return _manager


def feature_flag(flag_name: str, fallback: Any = None):
    """Decorator to wrap functions with feature flags."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_flag_manager()
            
            # Extract user context if available
            user_id = kwargs.get("user_id") or kwargs.get("user")
            environment = os.getenv("MCP_ENVIRONMENT", "development")
            
            if manager.is_enabled(flag_name, user_id, environment):
                return func(*args, **kwargs)
            else:
                if fallback is not None:
                    if callable(fallback):
                        return fallback(*args, **kwargs)
                    return fallback
                else:
                    raise FeatureNotEnabledError(f"Feature '{flag_name}' is not enabled")
        
        # Add metadata for introspection
        wrapper._feature_flag = flag_name
        wrapper._is_feature_flagged = True
        
        return wrapper
    return decorator


def is_feature_enabled(flag_name: str, user_id: Optional[str] = None) -> bool:
    """Check if a feature is enabled."""
    manager = get_flag_manager()
    environment = os.getenv("MCP_ENVIRONMENT", "development")
    return manager.is_enabled(flag_name, user_id, environment)


def set_feature_enabled(flag_name: str, enabled: bool) -> bool:
    """Enable or disable a feature."""
    manager = get_flag_manager()
    return manager.set_enabled(flag_name, enabled)


def get_feature_flags() -> Dict[str, Dict[str, Any]]:
    """Get all feature flags."""
    manager = get_flag_manager()
    return manager.get_all_flags()


class FeatureNotEnabledError(Exception):
    """Raised when attempting to use a disabled feature."""
    pass