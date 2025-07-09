# Dockerfile
FROM python:3.11-slim

# Security: Run as non-root user
RUN useradd -m -s /bin/bash mcp

WORKDIR /app

# Install only what we need
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only working code
COPY src/ ./src/

# Set Python path to find our module
ENV PYTHONPATH=/app/src

# Switch to non-root user
USER mcp

# Use unbuffered output for better container logging
ENV PYTHONUNBUFFERED=1

# Default to stdio mode (the only mode that works)
ENTRYPOINT ["python", "-m", "networkx_mcp.server"]