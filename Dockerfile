# Multi-stage Dockerfile for NetworkX MCP Server
# Production-ready container with security and optimization

# ================================
# STAGE 1: Builder
# ================================
FROM python:3.11-slim-bullseye AS builder

# Set build arguments
ARG BUILD_ENV=production
ARG VERSION=2.0.0

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml README.md ./

# Install dependencies
RUN pip install --no-cache-dir build wheel && \
    pip install --no-cache-dir -e .[prod]

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/

# Build wheel
RUN python -m build --wheel

# ================================
# STAGE 2: Runtime
# ================================
FROM python:3.11-slim-bullseye AS runtime

# Set build arguments
ARG BUILD_ENV=production
ARG VERSION=2.0.0
ARG BUILD_DATE
ARG GIT_COMMIT

# Add metadata labels
LABEL org.opencontainers.image.title="NetworkX MCP Server" \
      org.opencontainers.image.description="Enterprise-grade MCP server for NetworkX graph operations" \
      org.opencontainers.image.version="$VERSION" \
      org.opencontainers.image.created="$BUILD_DATE" \
      org.opencontainers.image.revision="$GIT_COMMIT" \
      org.opencontainers.image.source="https://github.com/your-org/networkx-mcp-server" \
      org.opencontainers.image.authors="Your Organization" \
      org.opencontainers.image.licenses="MIT"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_ENV=production \
    APP_VERSION=$VERSION

# Install runtime dependencies and security updates
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    ca-certificates \
    curl \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create application user and directories
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /bin/bash appuser && \
    mkdir -p /app/data /app/logs /app/config && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy wheel from builder stage
COPY --from=builder /app/dist/*.whl /tmp/

# Install application
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm -rf /tmp/*.whl

# Copy configuration files
COPY docker/config/ ./config/
COPY docker/entrypoint.sh ./entrypoint.sh

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Create non-root user directories
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]
CMD ["server"]

# ================================
# STAGE 3: Development (optional)
# ================================
FROM runtime AS development

USER root

# Install development tools
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    git \
    vim \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    ruff \
    mypy \
    pre-commit

# Copy source code for development
COPY --chown=appuser:appuser . .

# Install in development mode
RUN pip install -e .[dev,test]

USER appuser

# Override for development
ENV APP_ENV=development
CMD ["uvicorn", "src.networkx_mcp.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]