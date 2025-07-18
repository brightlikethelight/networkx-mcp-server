"""
Enterprise Configuration Management

Comprehensive configuration system for production deployments with:
- Environment variable support
- Type validation
- Security defaults
- Hot-reloadable settings
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_comma_separated_env(env_var: str, default: List[str] = None) -> List[str]:
    """Parse comma-separated environment variable into list."""
    if default is None:
        default = []

    value = os.environ.get(env_var, "")
    if not value:
        return default

    return [item.strip() for item in value.split(",") if item.strip()]


try:
    from pydantic import BaseModel, Field, field_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    raise ImportError(
        "Enterprise features require pydantic and pydantic-settings. "
        "Install with: pip install networkx-mcp-server[enterprise]"
    )


class SecurityConfig(BaseModel):
    """Security-related configuration."""

    # API Key Authentication
    api_key_enabled: bool = Field(
        default=True, description="Enable API key authentication"
    )
    api_keys: List[str] = Field(default_factory=list, description="Valid API keys")
    api_key_header: str = Field(default="X-API-Key", description="API key header name")

    # OAuth 2.1 Configuration
    oauth_enabled: bool = Field(
        default=False, description="Enable OAuth 2.1 authentication"
    )
    oauth_client_id: Optional[str] = Field(default=None, description="OAuth client ID")
    oauth_client_secret: Optional[str] = Field(
        default=None, description="OAuth client secret"
    )
    oauth_auth_url: Optional[str] = Field(
        default=None, description="OAuth authorization URL"
    )
    oauth_token_url: Optional[str] = Field(default=None, description="OAuth token URL")
    oauth_scopes: List[str] = Field(
        default_factory=lambda: ["graph:read", "graph:write"],
        description="Required OAuth scopes",
    )

    # JWT Configuration
    jwt_secret: str = Field(default="", description="JWT signing secret")
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    jwt_expiry_hours: int = Field(default=24, description="JWT token expiry in hours")

    # RBAC Configuration
    rbac_enabled: bool = Field(
        default=True, description="Enable role-based access control"
    )
    admin_users: List[str] = Field(
        default_factory=list, description="Admin user identifiers"
    )

    @field_validator("jwt_secret")
    @classmethod
    def validate_jwt_secret(cls, v):
        if v == "":
            # Generate a secure random secret if not provided
            import secrets

            return secrets.token_urlsafe(32)
        return v

    @classmethod
    def from_env(cls) -> "SecurityConfig":
        """Create config from environment variables."""
        return cls(
            api_key_enabled=os.environ.get(
                "NETWORKX_MCP_SECURITY_API_KEY_ENABLED", "true"
            ).lower()
            == "true",
            api_keys=parse_comma_separated_env("NETWORKX_MCP_SECURITY_API_KEYS"),
            api_key_header=os.environ.get(
                "NETWORKX_MCP_SECURITY_API_KEY_HEADER", "X-API-Key"
            ),
            oauth_enabled=os.environ.get(
                "NETWORKX_MCP_SECURITY_OAUTH_ENABLED", "false"
            ).lower()
            == "true",
            oauth_client_id=os.environ.get("NETWORKX_MCP_SECURITY_OAUTH_CLIENT_ID"),
            oauth_client_secret=os.environ.get(
                "NETWORKX_MCP_SECURITY_OAUTH_CLIENT_SECRET"
            ),
            oauth_auth_url=os.environ.get("NETWORKX_MCP_SECURITY_OAUTH_AUTH_URL"),
            oauth_token_url=os.environ.get("NETWORKX_MCP_SECURITY_OAUTH_TOKEN_URL"),
            jwt_secret=os.environ.get("NETWORKX_MCP_SECURITY_JWT_SECRET", ""),
            jwt_algorithm=os.environ.get(
                "NETWORKX_MCP_SECURITY_JWT_ALGORITHM", "HS256"
            ),
            jwt_expiry_hours=int(
                os.environ.get("NETWORKX_MCP_SECURITY_JWT_EXPIRY_HOURS", "24")
            ),
            rbac_enabled=os.environ.get(
                "NETWORKX_MCP_SECURITY_RBAC_ENABLED", "true"
            ).lower()
            == "true",
            admin_users=parse_comma_separated_env("NETWORKX_MCP_SECURITY_ADMIN_USERS"),
        )


class RateLimitConfig(BaseSettings):
    """Rate limiting configuration."""

    enabled: bool = Field(default=True, description="Enable rate limiting")
    requests_per_minute: int = Field(
        default=60, description="Requests per minute per client"
    )
    requests_per_hour: int = Field(
        default=1000, description="Requests per hour per client"
    )
    burst_size: int = Field(default=10, description="Burst request allowance")

    # Resource limits
    max_graph_size: int = Field(default=100000, description="Maximum nodes per graph")
    max_memory_mb: int = Field(default=512, description="Maximum memory usage in MB")
    max_execution_time: int = Field(
        default=30, description="Maximum execution time in seconds"
    )

    model_config = SettingsConfigDict(env_prefix="NETWORKX_MCP_RATE_LIMIT_")


class MonitoringConfig(BaseSettings):
    """Monitoring and observability configuration."""

    # Metrics
    metrics_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=8090, description="Metrics server port")
    metrics_path: str = Field(default="/metrics", description="Metrics endpoint path")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json/text)")
    audit_enabled: bool = Field(default=True, description="Enable audit logging")
    audit_log_file: Optional[str] = Field(
        default=None, description="Audit log file path"
    )

    # Health checks
    health_check_enabled: bool = Field(
        default=True, description="Enable health check endpoints"
    )
    health_check_port: int = Field(default=8091, description="Health check server port")

    model_config = SettingsConfigDict(env_prefix="NETWORKX_MCP_MONITORING_")


class ServerConfig(BaseSettings):
    """Server configuration."""

    # Server settings
    host: str = Field(default="localhost", description="Server host")
    port: int = Field(default=8080, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")

    # MCP Protocol
    transport: str = Field(
        default="stdio", description="MCP transport (stdio/http/sse)"
    )
    cors_enabled: bool = Field(default=True, description="Enable CORS")
    cors_origins: List[str] = Field(
        default_factory=lambda: ["*"], description="Allowed CORS origins"
    )

    # Performance
    async_enabled: bool = Field(default=True, description="Enable async processing")
    connection_pool_size: int = Field(default=100, description="Connection pool size")
    request_timeout: int = Field(default=60, description="Request timeout in seconds")

    model_config = SettingsConfigDict(env_prefix="NETWORKX_MCP_SERVER_")


class EnterpriseConfig(BaseSettings):
    """Main enterprise configuration container."""

    # Feature flags
    enterprise_mode: bool = Field(
        default=True, description="Enable enterprise features"
    )
    development_mode: bool = Field(default=False, description="Enable development mode")

    # Sub-configurations
    security: SecurityConfig = Field(default_factory=lambda: SecurityConfig.from_env())
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    # Configuration management
    config_file: Optional[str] = Field(
        default=None, description="Configuration file path"
    )
    hot_reload: bool = Field(
        default=False, description="Enable hot configuration reload"
    )

    model_config = SettingsConfigDict(
        env_prefix="NETWORKX_MCP_", env_file=".env", env_file_encoding="utf-8"
    )

    @classmethod
    def from_file(cls, config_path: str) -> "EnterpriseConfig":
        """Load configuration from file."""
        import json

        import yaml

        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if path.suffix.lower() in [".yml", ".yaml"]:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        elif path.suffix.lower() == ".json":
            with open(path, "r") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {path.suffix}")

        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    def validate_enterprise_requirements(self) -> List[str]:
        """Validate enterprise configuration and return any issues."""
        issues = []

        # Security validation
        if self.security.api_key_enabled and not self.security.api_keys:
            issues.append("API key authentication enabled but no API keys configured")

        if self.security.oauth_enabled:
            if not self.security.oauth_client_id:
                issues.append("OAuth enabled but client ID not configured")
            if not self.security.oauth_client_secret:
                issues.append("OAuth enabled but client secret not configured")

        # Rate limiting validation
        if self.rate_limit.enabled:
            if self.rate_limit.requests_per_minute <= 0:
                issues.append(
                    "Invalid rate limit: requests per minute must be positive"
                )

        # Monitoring validation
        if self.monitoring.metrics_enabled:
            if not (1024 <= self.monitoring.metrics_port <= 65535):
                issues.append("Invalid metrics port: must be between 1024 and 65535")

        return issues


# Global configuration instance
_config: Optional[EnterpriseConfig] = None


def get_config() -> EnterpriseConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = EnterpriseConfig()
    return _config


def set_config(config: EnterpriseConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def reload_config() -> EnterpriseConfig:
    """Reload configuration from environment."""
    global _config
    _config = EnterpriseConfig()
    return _config
