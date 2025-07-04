#!/bin/bash
# Script to create Kubernetes secrets for NetworkX MCP Server
# This script should be run once before deploying to Kubernetes

set -e

echo "Creating Kubernetes secrets for NetworkX MCP Server..."
echo "=================================================="
echo

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl not found. Please install kubectl first."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "k8s/deployment.yaml" ]; then
    echo "Error: Please run this script from the project root directory."
    exit 1
fi

# Function to generate secure password
generate_password() {
    openssl rand -base64 32 | tr -d "=+/" | cut -c1-32
}

# Check for existing .env file
if [ -f ".env" ]; then
    echo "Loading secrets from .env file..."
    source .env
else
    echo "No .env file found. Generating new secrets..."
    
    # Generate new secrets if not provided
    POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-$(generate_password)}
    JWT_SECRET=${JWT_SECRET:-$(generate_password)}
    REDIS_PASSWORD=${REDIS_PASSWORD:-$(generate_password)}
    
    # Save to .env for reference
    cat > .env << EOF
# Auto-generated secrets - DO NOT COMMIT
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
JWT_SECRET=${JWT_SECRET}
REDIS_PASSWORD=${REDIS_PASSWORD}
EOF
    
    echo "Secrets saved to .env file (DO NOT COMMIT THIS FILE)"
fi

# Create namespace if it doesn't exist
echo
echo "Creating namespace..."
kubectl create namespace networkx-mcp --dry-run=client -o yaml | kubectl apply -f -

# Create the secret
echo
echo "Creating Kubernetes secret..."
kubectl create secret generic networkx-mcp-secrets \
    --namespace=networkx-mcp \
    --from-literal=postgres-password="${POSTGRES_PASSWORD}" \
    --from-literal=jwt-secret="${JWT_SECRET}" \
    --from-literal=redis-password="${REDIS_PASSWORD}" \
    --dry-run=client -o yaml | kubectl apply -f -

# Verify secret was created
echo
echo "Verifying secret creation..."
kubectl get secret networkx-mcp-secrets -n networkx-mcp

echo
echo "✅ Secrets created successfully!"
echo
echo "Next steps:"
echo "1. Apply the deployment: kubectl apply -f k8s/deployment.yaml"
echo "2. Check deployment status: kubectl get pods -n networkx-mcp"
echo
echo "⚠️  IMPORTANT: Keep your .env file secure and never commit it to git!"