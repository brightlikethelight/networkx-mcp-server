#!/bin/bash
# Script to create Kubernetes secrets for NetworkX MCP Server Production Deployment
# Based on actual performance testing and production configuration

set -e

echo "Creating Kubernetes secrets for NetworkX MCP Server (Production)..."
echo "=================================================================="
echo

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl not found. Please install kubectl first."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "k8s/deployment-production.yaml" ]; then
    echo "Error: Please run this script from the project root directory."
    exit 1
fi

# Function to generate secure password
generate_password() {
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-32
}

# Function to generate secure token
generate_token() {
    openssl rand -hex 32
}

# Check for existing .env file
if [ -f ".env.production" ]; then
    echo "Loading secrets from .env.production file..."
    source .env.production
else
    echo "No .env.production file found. Generating new secrets..."
    
    # Generate new secrets if not provided
    AUTH_TOKEN=${AUTH_TOKEN:-$(generate_token)}
    REDIS_URL=${REDIS_URL:-"redis://redis-service:6379/0"}
    ADMIN_TOKEN=${ADMIN_TOKEN:-$(generate_token)}
    
    # Save to .env.production for reference
    cat > .env.production << EOF
# Auto-generated production secrets - DO NOT COMMIT
# Based on NetworkX MCP Server production configuration

# MCP Authentication
AUTH_TOKEN=${AUTH_TOKEN}

# Redis Configuration
REDIS_URL=${REDIS_URL}

# Admin Features
ADMIN_TOKEN=${ADMIN_TOKEN}

# Production Settings
ENVIRONMENT=production
LOG_LEVEL=INFO
LOG_FORMAT=json
MAX_CONCURRENT_CONNECTIONS=45
MAX_GRAPH_SIZE_NODES=10000
MAX_MEMORY_MB=2048
EOF
    
    echo "Secrets saved to .env.production file (DO NOT COMMIT THIS FILE)"
fi

# Create namespace if it doesn't exist (using default for simplicity)
echo
echo "Creating secrets in default namespace..."

# Create Redis credentials secret
echo
echo "Creating Redis credentials secret..."
kubectl create secret generic redis-credentials \
    --from-literal=url="${REDIS_URL}" \
    --dry-run=client -o yaml | kubectl apply -f -

# Create MCP authentication secret
echo
echo "Creating MCP authentication secret..."
kubectl create secret generic mcp-auth \
    --from-literal=token="${AUTH_TOKEN}" \
    --from-literal=admin-token="${ADMIN_TOKEN}" \
    --dry-run=client -o yaml | kubectl apply -f -

# Verify secrets were created
echo
echo "Verifying secret creation..."
kubectl get secret redis-credentials
kubectl get secret mcp-auth

echo
echo "✅ Production secrets created successfully!"
echo
echo "Configuration Summary:"
echo "- Max Concurrent Connections: 45 (based on 50-user testing limit)"
echo "- Max Graph Size: 10,000 nodes (for good performance)"
echo "- Memory Limit: 2GB (production config)"
echo "- Storage Backend: Redis"
echo
echo "Next steps:"
echo "1. Apply the production deployment: kubectl apply -f k8s/deployment-production.yaml"
echo "2. Check deployment status: kubectl get pods"
echo "3. Check service status: kubectl get svc"
echo "4. Monitor with: kubectl logs -f deployment/networkx-mcp-server"
echo
echo "Health checks available at:"
echo "- Liveness: http://<pod-ip>:8080/health"
echo "- Readiness: http://<pod-ip>:8080/ready"
echo "- Metrics: http://<pod-ip>:9090/metrics"
echo
echo "⚠️  IMPORTANT: Keep your .env.production file secure and never commit it to git!"
echo "⚠️  These limits are based on actual performance testing results."