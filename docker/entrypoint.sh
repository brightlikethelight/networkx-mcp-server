#!/bin/bash
set -euo pipefail

# NetworkX MCP Server Docker Entrypoint
# Handles initialization, health checks, and graceful shutdown

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1" >&2
    fi
}

# Default environment variables
export APP_ENV="${APP_ENV:-production}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export WORKERS="${WORKERS:-4}"

# Health check function
health_check() {
    local max_attempts=30
    local attempt=1
    
    log_info "Performing health check..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s "http://localhost:${PORT}/health" >/dev/null 2>&1; then
            log_info "Health check passed"
            return 0
        fi
        
        log_debug "Health check attempt $attempt/$max_attempts failed, retrying..."
        sleep 2
        ((attempt++))
    done
    
    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# Wait for dependencies
wait_for_service() {
    local service_name="$1"
    local host="$2"
    local port="$3"
    local max_attempts=30
    local attempt=1
    
    log_info "Waiting for $service_name at $host:$port..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if nc -z "$host" "$port" >/dev/null 2>&1; then
            log_info "$service_name is ready"
            return 0
        fi
        
        log_debug "Waiting for $service_name... (attempt $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    log_error "$service_name not available after $max_attempts attempts"
    return 1
}

# Database migration
run_migrations() {
    log_info "Running database migrations..."
    
    if command -v python >/dev/null 2>&1; then
        if python -c "import sys; sys.path.append('/app/src'); from networkx_mcp.enterprise.migrations import MigrationRunner; MigrationRunner().run_all()" 2>/dev/null; then
            log_info "Database migrations completed successfully"
        else
            log_warn "Database migrations failed or not available"
        fi
    else
        log_warn "Python not available for migrations"
    fi
}

# Initialize application
initialize_app() {
    log_info "Initializing NetworkX MCP Server..."
    
    # Create necessary directories
    mkdir -p /app/data /app/logs /app/config
    
    # Set permissions
    chmod 755 /app/data /app/logs
    
    # Parse Redis URL if provided
    if [[ -n "${REDIS_URL:-}" ]]; then
        log_debug "Redis URL configured: ${REDIS_URL}"
    fi
    
    # Parse PostgreSQL URL if provided
    if [[ -n "${POSTGRES_URL:-}" ]]; then
        log_debug "PostgreSQL URL configured"
    fi
    
    log_info "Application initialization complete"
}

# Graceful shutdown handler
shutdown_handler() {
    log_info "Received shutdown signal, gracefully shutting down..."
    
    # Send SIGTERM to main process
    if [[ -n "${MAIN_PID:-}" ]]; then
        kill -TERM "$MAIN_PID" 2>/dev/null || true
        
        # Wait for graceful shutdown
        local timeout=30
        local count=0
        
        while kill -0 "$MAIN_PID" 2>/dev/null && [[ $count -lt $timeout ]]; do
            sleep 1
            ((count++))
        done
        
        # Force kill if still running
        if kill -0 "$MAIN_PID" 2>/dev/null; then
            log_warn "Force killing main process"
            kill -KILL "$MAIN_PID" 2>/dev/null || true
        fi
    fi
    
    log_info "Shutdown complete"
    exit 0
}

# Set up signal handlers
trap shutdown_handler SIGTERM SIGINT

# Main execution based on command
case "${1:-server}" in
    "server")
        log_info "Starting NetworkX MCP Server in ${APP_ENV} mode"
        
        # Wait for dependencies in production
        if [[ "$APP_ENV" == "production" ]]; then
            # Extract host and port from URLs
            if [[ -n "${REDIS_URL:-}" ]]; then
                redis_host=$(echo "$REDIS_URL" | sed -n 's#redis://\([^:]*\):\([0-9]*\)/.*#\1#p')
                redis_port=$(echo "$REDIS_URL" | sed -n 's#redis://\([^:]*\):\([0-9]*\)/.*#\2#p')
                
                if [[ -n "$redis_host" && -n "$redis_port" ]]; then
                    wait_for_service "Redis" "$redis_host" "$redis_port"
                fi
            fi
            
            if [[ -n "${POSTGRES_URL:-}" ]]; then
                postgres_host=$(echo "$POSTGRES_URL" | sed -n 's#postgresql://[^@]*@\([^:]*\):\([0-9]*\)/.*#\1#p')
                postgres_port=$(echo "$POSTGRES_URL" | sed -n 's#postgresql://[^@]*@\([^:]*\):\([0-9]*\)/.*#\2#p')
                
                if [[ -n "$postgres_host" && -n "$postgres_port" ]]; then
                    wait_for_service "PostgreSQL" "$postgres_host" "$postgres_port"
                fi
            fi
        fi
        
        # Initialize application
        initialize_app
        
        # Run migrations if enabled
        if [[ "${RUN_MIGRATIONS:-true}" == "true" ]]; then
            run_migrations
        fi
        
        # Start the server
        log_info "Starting server on ${HOST}:${PORT} with ${WORKERS} workers"
        
        if [[ "$APP_ENV" == "development" ]]; then
            # Development mode with auto-reload
            exec python -m uvicorn src.networkx_mcp.server:app \
                --host "$HOST" \
                --port "$PORT" \
                --reload \
                --log-level debug &
        else
            # Production mode with gunicorn
            exec python -m gunicorn src.networkx_mcp.server:app \
                --bind "${HOST}:${PORT}" \
                --workers "$WORKERS" \
                --worker-class uvicorn.workers.UvicornWorker \
                --access-logfile - \
                --error-logfile - \
                --log-level info \
                --preload \
                --max-requests 1000 \
                --max-requests-jitter 100 \
                --timeout 30 \
                --keep-alive 2 &
        fi
        
        MAIN_PID=$!
        log_info "Server started with PID $MAIN_PID"
        
        # Wait for process to finish
        wait $MAIN_PID
        ;;
        
    "migrate")
        log_info "Running database migrations only"
        initialize_app
        run_migrations
        ;;
        
    "health")
        log_info "Running health check"
        health_check
        ;;
        
    "shell")
        log_info "Starting interactive shell"
        initialize_app
        exec /bin/bash
        ;;
        
    "test")
        log_info "Running tests"
        exec python -m pytest tests/ -v
        ;;
        
    "worker")
        log_info "Starting background worker"
        initialize_app
        exec python -m src.networkx_mcp.worker
        ;;
        
    *)
        log_info "Running custom command: $*"
        exec "$@"
        ;;
esac