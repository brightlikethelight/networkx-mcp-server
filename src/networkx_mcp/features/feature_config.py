"""Feature flag configuration and integration with main config system."""

import logging
import os
from typing import Dict, Any, Optional

# Import main configuration if available
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from config import get_settings, get_config_manager
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

from .feature_flags import FeatureFlagManager, FeatureStatus

logger = logging.getLogger(__name__)


class FeatureFlagConfig:
    """Integration between feature flags and main configuration system."""
    
    def __init__(self):
        self.manager = FeatureFlagManager()
        
        # Apply configuration if available
        if CONFIG_AVAILABLE:
            self._apply_config_settings()
            self._register_config_callbacks()
    
    def _apply_config_settings(self):
        """Apply settings from main configuration to feature flags."""
        try:
            settings = get_settings()
            
            # Map configuration settings to feature flags
            feature_mappings = {
                # ML Features
                "ml_base_features": settings.features.machine_learning,
                "ml_graph_embeddings": settings.features.machine_learning,
                "ml_link_prediction": settings.features.machine_learning,
                "ml_node_classification": settings.features.machine_learning,
                "ml_anomaly_detection": settings.features.machine_learning,
                
                # Performance Features
                "gpu_acceleration": settings.features.gpu_acceleration,
                "parallel_algorithms": settings.performance.parallel_processing,
                
                # Visualization
                "3d_visualization": settings.features.visualization,
                
                # API Features
                "batch_operations": settings.features.api_v2_enabled,
                
                # Monitoring
                "audit_logging": settings.security.audit_enabled,
                "performance_profiling": settings.features.monitoring,
            }
            
            for flag_name, config_enabled in feature_mappings.items():
                if flag_name in self.manager.flags:
                    # Only enable if config says so, don't disable if already enabled
                    if config_enabled and self.manager.flags[flag_name].status == FeatureStatus.DISABLED:
                        self.manager.set_enabled(flag_name, True, save=False, notify=False)
                        logger.info(f"Enabled feature '{flag_name}' from configuration")
            
            # Apply environment-specific settings
            environment = settings.environment
            if environment == "production":
                # Disable experimental features in production by default
                for name, flag in self.manager.flags.items():
                    if flag.status == FeatureStatus.EXPERIMENTAL:
                        flag.status = FeatureStatus.DISABLED
                        logger.info(f"Disabled experimental feature '{name}' in production")
            
            elif environment == "development":
                # Enable beta features in development
                for name, flag in self.manager.flags.items():
                    if flag.status == FeatureStatus.BETA:
                        flag.status = FeatureStatus.ENABLED
                        logger.info(f"Enabled beta feature '{name}' in development")
            
            # Save final state
            self.manager.save_flags()
            
        except Exception as e:
            logger.error(f"Failed to apply config settings to feature flags: {e}")
    
    def _register_config_callbacks(self):
        """Register callbacks for configuration changes."""
        try:
            config_manager = get_config_manager()
            
            def on_config_change(old_settings, new_settings):
                """Handle configuration changes."""
                # Check if ML features changed
                if old_settings.features.machine_learning != new_settings.features.machine_learning:
                    ml_flags = ["ml_base_features", "ml_graph_embeddings", 
                               "ml_link_prediction", "ml_node_classification"]
                    for flag in ml_flags:
                        self.manager.set_enabled(flag, new_settings.features.machine_learning)
                
                # Check if GPU acceleration changed
                if old_settings.features.gpu_acceleration != new_settings.features.gpu_acceleration:
                    self.manager.set_enabled("gpu_acceleration", new_settings.features.gpu_acceleration)
                
                # Check if monitoring changed
                if old_settings.features.monitoring != new_settings.features.monitoring:
                    self.manager.set_enabled("performance_profiling", new_settings.features.monitoring)
                
                logger.info("Updated feature flags from configuration changes")
            
            config_manager.add_change_callback(on_config_change)
            
        except Exception as e:
            logger.error(f"Failed to register config callbacks: {e}")


# Initialize feature flag configuration
_feature_config: Optional[FeatureFlagConfig] = None


def get_feature_config() -> FeatureFlagConfig:
    """Get the feature flag configuration."""
    global _feature_config
    if _feature_config is None:
        _feature_config = FeatureFlagConfig()
    return _feature_config


# Convenience functions that integrate with config
def is_ml_enabled() -> bool:
    """Check if any ML features are enabled."""
    config = get_feature_config()
    return config.manager.is_enabled("ml_base_features")


def is_gpu_enabled() -> bool:
    """Check if GPU acceleration is enabled."""
    config = get_feature_config()
    return config.manager.is_enabled("gpu_acceleration")


def is_experimental_enabled() -> bool:
    """Check if experimental features are allowed."""
    config = get_feature_config()
    environment = os.getenv("MCP_ENVIRONMENT", "development")
    
    # Never allow experimental in production unless explicitly overridden
    if environment == "production":
        return os.getenv("ALLOW_EXPERIMENTAL", "false").lower() == "true"
    
    return True


def get_enabled_features() -> Dict[str, bool]:
    """Get all enabled features organized by category."""
    config = get_feature_config()
    manager = config.manager
    
    result = {}
    for category in ["machine_learning", "performance", "visualization", 
                    "api", "security", "monitoring", "experimental"]:
        category_flags = manager.get_flags_by_category(category)
        result[category] = {
            name: flag.is_enabled() 
            for name, flag in category_flags.items()
        }
    
    return result