# Multi-stage build for smaller final image
FROM python:3.12-slim AS builder

# Build arguments
ARG VERSION=dev

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Build wheel
RUN pip install --no-cache-dir build && \
    python -m build --wheel

# Final stage
FROM python:3.12-slim

# Build arguments
ARG VERSION=dev

# Labels
LABEL org.opencontainers.image.title="NetworkX MCP Server"
LABEL org.opencontainers.image.description="Academic-focused graph analysis in your AI conversations"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.authors="Bright Liu <brightliu@college.harvard.edu>"
LABEL org.opencontainers.image.source="https://github.com/Bright-L01/networkx-mcp-server"
LABEL org.opencontainers.image.licenses="MIT"

# Security: Run as non-root user
RUN useradd -m -s /bin/bash mcp && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends --fix-missing \
        # Required for matplotlib backend
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        # Useful for debugging
        curl \
        jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy wheel from builder
COPY --from=builder /build/dist/*.whl /tmp/

# Install the wheel and dependencies
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm -rf /tmp/*.whl

# Create necessary directories with correct permissions
RUN mkdir -p /app/data /app/logs && \
    chown -R mcp:mcp /app

# Switch to non-root user
USER mcp

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MCP_MODE=stdio

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | \
        python -m networkx_mcp.server | \
        jq -e '.result.protocolVersion' || exit 1

# Default to stdio mode
ENTRYPOINT ["python", "-m", "networkx_mcp.server"]

# Allow passing additional arguments
CMD []
