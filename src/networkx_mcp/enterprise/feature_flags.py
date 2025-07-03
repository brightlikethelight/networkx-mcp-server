"""Feature flag service for runtime feature toggles."""

import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..core.base import Component
from ..monitoring.logging import with_logging_context

logger = logging.getLogger(__name__)


class FeatureFlagType(Enum):
    """Types of feature flags."""

    BOOLEAN = "boolean"
    STRING = "string"
    NUMBER = "number"
    JSON = "json"


@dataclass
class FeatureFlag:
    """Feature flag definition."""

    name: str
    flag_type: FeatureFlagType
    default_value: Any
    description: str
    enabled: bool = True
    created_at: float = None
    updated_at: float = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.updated_at is None:
            self.updated_at = time.time()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FeatureFlagRule:
    """Rule for conditional feature flag evaluation."""

    flag_name: str
    condition: str  # e.g., "user_id in ['123', '456']"
    value: Any
    priority: int = 0  # Higher priority rules are evaluated first
    enabled: bool = True


class FeatureFlagService(Component):
    """Service for managing feature flags and runtime toggles."""

    def __init__(self, config_file: str | None = None):
        super().__init__("feature_flag_service")
        self.config_file = config_file
        self.flags: dict[str, FeatureFlag] = {}
        self.rules: dict[str, list[FeatureFlagRule]] = {}
        self.evaluation_cache: dict[str, Any] = {}
        self.cache_ttl = 60  # Cache TTL in seconds

        # Built-in flags
        self._register_builtin_flags()

    def _register_builtin_flags(self):
        """Register built-in feature flags."""
        builtin_flags = [
            FeatureFlag(
                name="api_v2_enabled",
                flag_type=FeatureFlagType.BOOLEAN,
                default_value=True,
                description="Enable API v2 endpoints",
            ),
            FeatureFlag(
                name="advanced_algorithms",
                flag_type=FeatureFlagType.BOOLEAN,
                default_value=True,
                description="Enable advanced graph algorithms",
            ),
            FeatureFlag(
                name="caching_enabled",
                flag_type=FeatureFlagType.BOOLEAN,
                default_value=True,
                description="Enable result caching",
            ),
            FeatureFlag(
                name="max_graph_nodes",
                flag_type=FeatureFlagType.NUMBER,
                default_value=10000,
                description="Maximum number of nodes in a graph",
            ),
            FeatureFlag(
                name="visualization_engine",
                flag_type=FeatureFlagType.STRING,
                default_value="plotly",
                description="Default visualization engine (plotly, matplotlib, pyvis)",
            ),
            FeatureFlag(
                name="rate_limit_config",
                flag_type=FeatureFlagType.JSON,
                default_value={"requests": 100, "window": 60},
                description="Rate limiting configuration",
            ),
        ]

        for flag in builtin_flags:
            self.flags[flag.name] = flag

    @with_logging_context(component="feature_flags")
    async def register_flag(self, flag: FeatureFlag) -> None:
        """Register a new feature flag."""
        self.flags[flag.name] = flag
        self._clear_cache(flag.name)
        logger.info(f"Registered feature flag: {flag.name}")

    @with_logging_context(component="feature_flags")
    async def update_flag(
        self,
        name: str,
        enabled: bool | None = None,
        default_value: Any | None = None,
        **metadata,
    ) -> bool:
        """Update an existing feature flag."""
        if name not in self.flags:
            logger.warning(f"Feature flag not found: {name}")
            return False

        flag = self.flags[name]

        if enabled is not None:
            flag.enabled = enabled

        if default_value is not None:
            flag.default_value = default_value

        if metadata:
            flag.metadata.update(metadata)

        flag.updated_at = time.time()
        self._clear_cache(name)

        logger.info(f"Updated feature flag: {name}")
        return True

    @with_logging_context(component="feature_flags")
    async def add_rule(self, rule: FeatureFlagRule) -> None:
        """Add a conditional rule for feature flag evaluation."""
        if rule.flag_name not in self.flags:
            logger.warning(f"Feature flag not found for rule: {rule.flag_name}")
            return

        if rule.flag_name not in self.rules:
            self.rules[rule.flag_name] = []

        self.rules[rule.flag_name].append(rule)

        # Sort rules by priority (highest first)
        self.rules[rule.flag_name].sort(key=lambda r: r.priority, reverse=True)

        self._clear_cache(rule.flag_name)
        logger.info(f"Added rule for feature flag: {rule.flag_name}")

    @with_logging_context(component="feature_flags")
    async def evaluate_flag(
        self,
        name: str,
        context: dict[str, Any] | None = None,
        default: Any | None = None,
    ) -> Any:
        """Evaluate a feature flag with optional context."""
        context = context or {}

        # Check cache first
        cache_key = self._get_cache_key(name, context)
        if cache_key in self.evaluation_cache:
            cached_result, cached_time = self.evaluation_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_result

        # Get flag
        flag = self.flags.get(name)
        if not flag:
            logger.warning(f"Feature flag not found: {name}")
            return default

        # If flag is disabled, return default
        if not flag.enabled:
            return flag.default_value if default is None else default

        # Evaluate rules if any exist
        flag_rules = self.rules.get(name, [])
        for rule in flag_rules:
            if not rule.enabled:
                continue

            try:
                # Simple condition evaluation (in production, use a proper expression evaluator)
                if self._evaluate_condition(rule.condition, context):
                    result = rule.value
                    self._cache_result(cache_key, result)
                    return result
            except Exception as e:
                logger.error(f"Error evaluating rule condition: {rule.condition} - {e}")
                continue

        # Return default value
        result = flag.default_value if default is None else default
        self._cache_result(cache_key, result)
        return result

    def _evaluate_condition(self, condition: str, context: dict[str, Any]) -> bool:
        """Evaluate a condition string with context variables."""
        # Simple condition evaluation - in production, use a safe expression evaluator
        try:
            # Create a safe evaluation environment
            safe_globals = {
                "__builtins__": {},
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "len": len,
                "isinstance": isinstance,
                "in": lambda x, y: x in y,
                "not": lambda x: not x,
                "and": lambda x, y: x and y,
                "or": lambda x, y: x or y,
            }

            # Add context variables
            safe_locals = context.copy()

            # Evaluate condition
            return eval(condition, safe_globals, safe_locals)

        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False

    def _get_cache_key(self, name: str, context: dict[str, Any]) -> str:
        """Generate cache key for flag evaluation."""
        context_str = json.dumps(context, sort_keys=True, default=str)
        return f"{name}:{hash(context_str)}"

    def _cache_result(self, cache_key: str, result: Any) -> None:
        """Cache evaluation result."""
        self.evaluation_cache[cache_key] = (result, time.time())

    def _clear_cache(self, flag_name: str | None = None) -> None:
        """Clear evaluation cache."""
        if flag_name:
            # Clear cache entries for specific flag
            keys_to_remove = [
                k for k in self.evaluation_cache.keys() if k.startswith(f"{flag_name}:")
            ]
            for key in keys_to_remove:
                del self.evaluation_cache[key]
        else:
            # Clear all cache
            self.evaluation_cache.clear()

    async def is_enabled(
        self, name: str, context: dict[str, Any] | None = None
    ) -> bool:
        """Check if a boolean feature flag is enabled."""
        result = await self.evaluate_flag(name, context, False)
        return bool(result)

    async def get_string_value(
        self, name: str, context: dict[str, Any] | None = None, default: str = ""
    ) -> str:
        """Get string value from feature flag."""
        result = await self.evaluate_flag(name, context, default)
        return str(result)

    async def get_number_value(
        self, name: str, context: dict[str, Any] | None = None, default: float = 0.0
    ) -> float:
        """Get numeric value from feature flag."""
        result = await self.evaluate_flag(name, context, default)
        try:
            return float(result)
        except (ValueError, TypeError):
            return default

    async def get_json_value(
        self,
        name: str,
        context: dict[str, Any] | None = None,
        default: dict | None = None,
    ) -> dict:
        """Get JSON value from feature flag."""
        result = await self.evaluate_flag(name, context, default or {})
        if isinstance(result, dict):
            return result
        elif isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return default or {}
        else:
            return default or {}

    async def get_all_flags(self) -> dict[str, Any]:
        """Get all feature flags and their current states."""
        result = {}
        for name, flag in self.flags.items():
            result[name] = {
                "enabled": flag.enabled,
                "type": flag.flag_type.value,
                "default_value": flag.default_value,
                "description": flag.description,
                "has_rules": name in self.rules and len(self.rules[name]) > 0,
                "created_at": flag.created_at,
                "updated_at": flag.updated_at,
                "metadata": flag.metadata,
            }
        return result

    async def load_from_file(self, file_path: str) -> bool:
        """Load feature flags from configuration file."""
        try:
            with open(file_path) as f:
                config = json.load(f)

            # Load flags
            for flag_data in config.get("flags", []):
                flag = FeatureFlag(
                    name=flag_data["name"],
                    flag_type=FeatureFlagType(flag_data["type"]),
                    default_value=flag_data["default_value"],
                    description=flag_data["description"],
                    enabled=flag_data.get("enabled", True),
                    metadata=flag_data.get("metadata", {}),
                )
                await self.register_flag(flag)

            # Load rules
            for rule_data in config.get("rules", []):
                rule = FeatureFlagRule(
                    flag_name=rule_data["flag_name"],
                    condition=rule_data["condition"],
                    value=rule_data["value"],
                    priority=rule_data.get("priority", 0),
                    enabled=rule_data.get("enabled", True),
                )
                await self.add_rule(rule)

            logger.info(f"Loaded feature flags from: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load feature flags from {file_path}: {e}")
            return False

    async def save_to_file(self, file_path: str) -> bool:
        """Save feature flags to configuration file."""
        try:
            config = {
                "flags": [
                    {
                        "name": flag.name,
                        "type": flag.flag_type.value,
                        "default_value": flag.default_value,
                        "description": flag.description,
                        "enabled": flag.enabled,
                        "metadata": flag.metadata,
                    }
                    for flag in self.flags.values()
                ],
                "rules": [
                    {
                        "flag_name": rule.flag_name,
                        "condition": rule.condition,
                        "value": rule.value,
                        "priority": rule.priority,
                        "enabled": rule.enabled,
                    }
                    for rules_list in self.rules.values()
                    for rule in rules_list
                ],
            }

            with open(file_path, "w") as f:
                json.dump(config, f, indent=2, default=str)

            logger.info(f"Saved feature flags to: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save feature flags to {file_path}: {e}")
            return False

    async def initialize(self) -> None:
        """Initialize the feature flag service."""
        await super().initialize()

        # Load from config file if specified
        if self.config_file:
            await self.load_from_file(self.config_file)

        logger.info(f"Feature flag service initialized with {len(self.flags)} flags")

    async def cleanup(self) -> None:
        """Cleanup the feature flag service."""
        # Save to config file if specified
        if self.config_file:
            await self.save_to_file(self.config_file)

        await super().cleanup()


# Decorator for feature flag gating
def feature_flag(flag_name: str, default_value: Any = False):
    """Decorator to gate functions behind feature flags."""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Get feature flag service (would be injected in real app)
            from ..core.container import get_container

            container = get_container()
            flag_service = container.get(FeatureFlagService)

            # Extract context from request if available
            context = {}
            if args and hasattr(args[0], "user"):
                user = args[0].user
                if user:
                    context["user_id"] = user.user_id
                    context["roles"] = list(user.roles)

            # Evaluate flag
            enabled = await flag_service.evaluate_flag(
                flag_name, context, default_value
            )

            if not enabled:
                raise ValueError(f"Feature '{flag_name}' is not enabled")

            return await func(*args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            # For sync functions, we'd need to handle differently
            # This is a simplified version
            return func(*args, **kwargs)

        if hasattr(func, "__code__") and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
