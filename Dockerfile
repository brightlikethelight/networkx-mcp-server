FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional production dependencies
RUN pip install --no-cache-dir \
    redis \
    psutil \
    prometheus-client

# Copy application code
COPY src/ ./src/
COPY security_patches.py .
COPY add_persistence.py .
COPY run_secure_server.py .
COPY validate_production.py .

# Create logs directory
RUN mkdir -p /app/logs

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Environment variables
ENV PYTHONPATH=/app
ENV REDIS_URL=redis://redis:6379/0
ENV STORAGE_BACKEND=redis
ENV MAX_MEMORY_MB=2000
ENV LOG_LEVEL=INFO
ENV MCP_TRANSPORT=sse

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from src.networkx_mcp.server import graph_manager; print('healthy')" || exit 1

# Expose port
EXPOSE 8765

# Run the secure server
CMD ["python", "run_secure_server.py"]