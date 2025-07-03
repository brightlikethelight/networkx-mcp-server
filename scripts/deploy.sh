#!/bin/bash
set -euo pipefail

# NetworkX MCP Server Deployment Script
# Supports Docker Compose, Kubernetes, and cloud deployments

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default configuration
DEPLOYMENT_TYPE="docker-compose"
ENVIRONMENT="production"
NAMESPACE="networkx-mcp"
HELM_RELEASE_NAME="networkx-mcp"
BUILD_IMAGE="true"
DRY_RUN="false"
FORCE_RECREATE="false"
SKIP_TESTS="false"
CONFIG_FILE=""
VALUES_FILE=""
TIMEOUT="600"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $1" >&2
}

# Help function
show_help() {
    cat << EOF
NetworkX MCP Server Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -t, --type TYPE              Deployment type: docker-compose, kubernetes, helm (default: docker-compose)
    -e, --environment ENV        Environment: development, staging, production (default: production)
    -n, --namespace NAMESPACE    Kubernetes namespace (default: networkx-mcp)
    -r, --release RELEASE        Helm release name (default: networkx-mcp)
    -b, --build                  Build Docker image before deployment (default: true)
    -c, --config FILE            Configuration file path
    -v, --values FILE            Helm values file path
    --dry-run                    Show what would be deployed without executing
    --force-recreate             Force recreation of existing resources
    --skip-tests                 Skip running tests before deployment
    --timeout SECONDS            Deployment timeout in seconds (default: 600)
    --debug                      Enable debug logging
    -h, --help                   Show this help message

EXAMPLES:
    # Deploy with Docker Compose in development
    $0 -t docker-compose -e development

    # Deploy to Kubernetes with custom namespace
    $0 -t kubernetes -n my-namespace -e production

    # Deploy with Helm using custom values file
    $0 -t helm -v values-prod.yaml -e production

    # Dry run deployment
    $0 -t kubernetes --dry-run

    # Force recreate all resources
    $0 -t docker-compose --force-recreate

ENVIRONMENT VARIABLES:
    DOCKER_REGISTRY          Docker registry for images
    KUBECONFIG               Kubernetes config file path
    HELM_REPO_URL           Helm repository URL
    BUILD_ARGS              Additional Docker build arguments
    DEBUG                   Enable debug mode

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                DEPLOYMENT_TYPE="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -r|--release)
                HELM_RELEASE_NAME="$2"
                shift 2
                ;;
            -b|--build)
                BUILD_IMAGE="true"
                shift
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -v|--values)
                VALUES_FILE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --force-recreate)
                FORCE_RECREATE="true"
                shift
                ;;
            --skip-tests)
                SKIP_TESTS="true"
                shift
                ;;
            --timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --debug)
                DEBUG="true"
                export DEBUG="true"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate dependencies
validate_dependencies() {
    log_step "Validating dependencies..."
    
    case $DEPLOYMENT_TYPE in
        docker-compose)
            if ! command -v docker-compose >/dev/null 2>&1; then
                log_error "docker-compose is required but not installed"
                exit 1
            fi
            ;;
        kubernetes)
            if ! command -v kubectl >/dev/null 2>&1; then
                log_error "kubectl is required but not installed"
                exit 1
            fi
            ;;
        helm)
            if ! command -v helm >/dev/null 2>&1; then
                log_error "helm is required but not installed"
                exit 1
            fi
            if ! command -v kubectl >/dev/null 2>&1; then
                log_error "kubectl is required but not installed"
                exit 1
            fi
            ;;
        *)
            log_error "Unsupported deployment type: $DEPLOYMENT_TYPE"
            exit 1
            ;;
    esac
    
    if [[ "$BUILD_IMAGE" == "true" ]] && ! command -v docker >/dev/null 2>&1; then
        log_error "docker is required for building images"
        exit 1
    fi
}

# Build Docker image
build_image() {
    if [[ "$BUILD_IMAGE" != "true" ]]; then
        log_info "Skipping image build"
        return
    fi
    
    log_step "Building Docker image..."
    
    cd "$PROJECT_ROOT"
    
    local image_tag="networkx-mcp:${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S)"
    local latest_tag="networkx-mcp:${ENVIRONMENT}-latest"
    
    local build_args=""
    if [[ -n "${BUILD_ARGS:-}" ]]; then
        build_args="$BUILD_ARGS"
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would build image with tag: $image_tag"
        return
    fi
    
    docker build \
        --build-arg BUILD_ENV="$ENVIRONMENT" \
        --build-arg VERSION="2.0.0" \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')" \
        $build_args \
        -t "$image_tag" \
        -t "$latest_tag" \
        .
    
    # Tag for registry if specified
    if [[ -n "${DOCKER_REGISTRY:-}" ]]; then
        docker tag "$latest_tag" "${DOCKER_REGISTRY}/networkx-mcp:${ENVIRONMENT}-latest"
        docker push "${DOCKER_REGISTRY}/networkx-mcp:${ENVIRONMENT}-latest"
    fi
    
    log_info "Image built successfully: $image_tag"
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_info "Skipping tests"
        return
    fi
    
    log_step "Running tests..."
    
    cd "$PROJECT_ROOT"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would run test suite"
        return
    fi
    
    # Run tests in Docker container
    docker run --rm \
        -v "$PROJECT_ROOT:/app" \
        -w /app \
        python:3.11-slim \
        bash -c "
            pip install -e .[test] && \
            python -m pytest tests/ -v --tb=short
        "
}

# Deploy with Docker Compose
deploy_docker_compose() {
    log_step "Deploying with Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    local compose_file="docker-compose.yml"
    if [[ "$ENVIRONMENT" == "development" ]]; then
        compose_file="docker-compose.dev.yml"
    fi
    
    local compose_cmd="docker-compose -f $compose_file"
    
    if [[ -n "$CONFIG_FILE" ]]; then
        compose_cmd="$compose_cmd -f $CONFIG_FILE"
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would run: $compose_cmd up -d"
        return
    fi
    
    # Set environment variables
    export APP_ENV="$ENVIRONMENT"
    export BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    export GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
    
    if [[ "$FORCE_RECREATE" == "true" ]]; then
        $compose_cmd down --volumes --remove-orphans
        $compose_cmd up -d --force-recreate
    else
        $compose_cmd up -d
    fi
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    timeout "$TIMEOUT" bash -c "
        while ! $compose_cmd ps | grep -q 'healthy'; do
            sleep 5
        done
    "
    
    log_info "Docker Compose deployment completed successfully"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_step "Deploying to Kubernetes..."
    
    cd "$PROJECT_ROOT"
    
    # Create namespace if it doesn't exist
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "DRY RUN: Would create namespace: $NAMESPACE"
        else
            kubectl create namespace "$NAMESPACE"
        fi
    fi
    
    local kubectl_cmd="kubectl apply -n $NAMESPACE"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        kubectl_cmd="kubectl apply --dry-run=client -o yaml -n $NAMESPACE"
    fi
    
    # Apply Kubernetes manifests
    $kubectl_cmd -f k8s/
    
    if [[ "$DRY_RUN" == "false" ]]; then
        # Wait for deployment to be ready
        log_info "Waiting for deployment to be ready..."
        kubectl wait --for=condition=available --timeout="${TIMEOUT}s" \
            deployment/networkx-mcp -n "$NAMESPACE"
        
        # Check rollout status
        kubectl rollout status deployment/networkx-mcp -n "$NAMESPACE" --timeout="${TIMEOUT}s"
    fi
    
    log_info "Kubernetes deployment completed successfully"
}

# Deploy with Helm
deploy_helm() {
    log_step "Deploying with Helm..."
    
    cd "$PROJECT_ROOT"
    
    local helm_cmd="helm"
    local values_args=""
    
    # Add values file if specified
    if [[ -n "$VALUES_FILE" ]]; then
        values_args="-f $VALUES_FILE"
    fi
    
    # Set environment-specific values
    local env_values="--set app.environment=$ENVIRONMENT"
    env_values="$env_values --set image.tag=${ENVIRONMENT}-latest"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        helm_cmd="helm template"
        log_info "DRY RUN: Generating Helm templates..."
        $helm_cmd "$HELM_RELEASE_NAME" helm/networkx-mcp/ \
            $values_args $env_values \
            --namespace "$NAMESPACE"
        return
    fi
    
    # Create namespace if it doesn't exist
    if ! kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
        kubectl create namespace "$NAMESPACE"
    fi
    
    # Add/update Helm dependencies
    helm dependency update helm/networkx-mcp/
    
    # Deploy or upgrade
    if helm list -n "$NAMESPACE" | grep -q "$HELM_RELEASE_NAME"; then
        log_info "Upgrading existing Helm release..."
        helm upgrade "$HELM_RELEASE_NAME" helm/networkx-mcp/ \
            $values_args $env_values \
            --namespace "$NAMESPACE" \
            --timeout "${TIMEOUT}s" \
            --wait
    else
        log_info "Installing new Helm release..."
        helm install "$HELM_RELEASE_NAME" helm/networkx-mcp/ \
            $values_args $env_values \
            --namespace "$NAMESPACE" \
            --timeout "${TIMEOUT}s" \
            --wait \
            --create-namespace
    fi
    
    # Check deployment status
    kubectl rollout status deployment/networkx-mcp -n "$NAMESPACE" --timeout="${TIMEOUT}s"
    
    log_info "Helm deployment completed successfully"
}

# Post-deployment verification
verify_deployment() {
    log_step "Verifying deployment..."
    
    case $DEPLOYMENT_TYPE in
        docker-compose)
            if docker-compose ps | grep -q "Up"; then
                log_info "✅ Docker Compose services are running"
            else
                log_error "❌ Some Docker Compose services are not running"
                return 1
            fi
            ;;
        kubernetes|helm)
            if kubectl get pods -n "$NAMESPACE" | grep -q "Running"; then
                log_info "✅ Kubernetes pods are running"
                
                # Test service endpoint
                local service_url=""
                if kubectl get ingress -n "$NAMESPACE" >/dev/null 2>&1; then
                    service_url="http://$(kubectl get ingress -n "$NAMESPACE" -o jsonpath='{.items[0].spec.rules[0].host}')"
                else
                    service_url="http://localhost:8000"
                    kubectl port-forward -n "$NAMESPACE" service/networkx-mcp-service 8000:8000 &
                    sleep 5
                fi
                
                if curl -f -s "$service_url/health" >/dev/null; then
                    log_info "✅ Health check passed"
                else
                    log_warn "⚠️ Health check failed"
                fi
            else
                log_error "❌ Some Kubernetes pods are not running"
                return 1
            fi
            ;;
    esac
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
}

# Signal handlers
trap cleanup EXIT INT TERM

# Main function
main() {
    log_info "🚀 Starting NetworkX MCP Server deployment"
    log_info "Deployment type: $DEPLOYMENT_TYPE"
    log_info "Environment: $ENVIRONMENT"
    log_info "Dry run: $DRY_RUN"
    
    # Validate dependencies
    validate_dependencies
    
    # Build image if required
    build_image
    
    # Run tests
    run_tests
    
    # Deploy based on type
    case $DEPLOYMENT_TYPE in
        docker-compose)
            deploy_docker_compose
            ;;
        kubernetes)
            deploy_kubernetes
            ;;
        helm)
            deploy_helm
            ;;
        *)
            log_error "Unsupported deployment type: $DEPLOYMENT_TYPE"
            exit 1
            ;;
    esac
    
    # Verify deployment
    if [[ "$DRY_RUN" == "false" ]]; then
        verify_deployment
    fi
    
    log_info "🎉 Deployment completed successfully!"
    
    # Show access information
    case $DEPLOYMENT_TYPE in
        docker-compose)
            log_info "Access the application at: http://localhost:8000"
            log_info "Grafana dashboard at: http://localhost:3000 (admin/admin)"
            log_info "Prometheus at: http://localhost:9090"
            ;;
        kubernetes|helm)
            log_info "Access the application through the ingress or port-forward:"
            log_info "kubectl port-forward -n $NAMESPACE service/networkx-mcp-service 8000:8000"
            ;;
    esac
}

# Parse arguments and run main function
parse_args "$@"
main