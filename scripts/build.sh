#!/bin/bash
set -euo pipefail

# NetworkX MCP Server Build Script
# Comprehensive build automation with multi-stage Docker builds

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default configuration
BUILD_TYPE="production"
DOCKER_REGISTRY=""
TAG_LATEST="true"
PUSH_IMAGE="false"
BUILD_CONTEXT="."
DOCKERFILE="Dockerfile"
CACHE_FROM=""
BUILD_ARGS=""
PLATFORMS="linux/amd64"
MULTI_ARCH="false"
DRY_RUN="false"
VERSION="2.0.0"
NO_CACHE="false"

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
NetworkX MCP Server Build Script

Usage: $0 [OPTIONS]

OPTIONS:
    -t, --type TYPE              Build type: development, production (default: production)
    -r, --registry REGISTRY      Docker registry URL
    -v, --version VERSION        Version tag (default: 2.0.0)
    --tag-latest                 Tag as latest (default: true)
    --push                       Push image to registry
    --no-cache                   Build without using cache
    --dockerfile PATH            Dockerfile path (default: Dockerfile)
    --context PATH               Build context path (default: .)
    --cache-from IMAGE           Cache source image
    --build-arg ARG=VALUE        Build argument (can be used multiple times)
    --platforms PLATFORMS        Target platforms for multi-arch build (default: linux/amd64)
    --multi-arch                 Enable multi-architecture build
    --dry-run                    Show what would be built without executing
    --debug                      Enable debug logging
    -h, --help                   Show this help message

EXAMPLES:
    # Basic production build
    $0 -t production

    # Development build with push to registry
    $0 -t development -r my-registry.com --push

    # Multi-architecture build
    $0 --multi-arch --platforms linux/amd64,linux/arm64

    # Build with custom version and build args
    $0 -v 2.1.0 --build-arg BUILD_ENV=staging

    # Dry run to see what would be built
    $0 --dry-run

ENVIRONMENT VARIABLES:
    DOCKER_REGISTRY              Default Docker registry
    DOCKER_BUILDKIT              Enable BuildKit (default: 1)
    BUILD_ENV                    Build environment override
    DEBUG                        Enable debug mode

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                BUILD_TYPE="$2"
                shift 2
                ;;
            -r|--registry)
                DOCKER_REGISTRY="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            --tag-latest)
                TAG_LATEST="true"
                shift
                ;;
            --push)
                PUSH_IMAGE="true"
                shift
                ;;
            --no-cache)
                NO_CACHE="true"
                shift
                ;;
            --dockerfile)
                DOCKERFILE="$2"
                shift 2
                ;;
            --context)
                BUILD_CONTEXT="$2"
                shift 2
                ;;
            --cache-from)
                CACHE_FROM="$2"
                shift 2
                ;;
            --build-arg)
                if [[ -n "$BUILD_ARGS" ]]; then
                    BUILD_ARGS="$BUILD_ARGS --build-arg $2"
                else
                    BUILD_ARGS="--build-arg $2"
                fi
                shift 2
                ;;
            --platforms)
                PLATFORMS="$2"
                shift 2
                ;;
            --multi-arch)
                MULTI_ARCH="true"
                shift
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
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
    log_step "Validating build dependencies..."
    
    if ! command -v docker >/dev/null 2>&1; then
        log_error "Docker is required but not installed"
        exit 1
    fi
    
    # Check Docker version
    local docker_version
    docker_version=$(docker version --format '{{.Client.Version}}' 2>/dev/null || echo "0.0.0")
    log_debug "Docker version: $docker_version"
    
    # Enable BuildKit
    export DOCKER_BUILDKIT=1
    
    if [[ "$MULTI_ARCH" == "true" ]]; then
        # Check if buildx is available
        if ! docker buildx version >/dev/null 2>&1; then
            log_error "Docker buildx is required for multi-architecture builds"
            exit 1
        fi
        
        # Create or use buildx builder
        if ! docker buildx inspect multiarch-builder >/dev/null 2>&1; then
            log_info "Creating multi-architecture builder..."
            docker buildx create --name multiarch-builder --use --bootstrap
        else
            docker buildx use multiarch-builder
        fi
    fi
    
    # Validate build context
    if [[ ! -d "$BUILD_CONTEXT" ]]; then
        log_error "Build context directory not found: $BUILD_CONTEXT"
        exit 1
    fi
    
    # Validate Dockerfile
    if [[ ! -f "$BUILD_CONTEXT/$DOCKERFILE" ]]; then
        log_error "Dockerfile not found: $BUILD_CONTEXT/$DOCKERFILE"
        exit 1
    fi
}

# Get Git information
get_git_info() {
    local git_commit="unknown"
    local git_branch="unknown"
    local git_tag="unknown"
    
    if git rev-parse --git-dir >/dev/null 2>&1; then
        git_commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
        git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
        git_tag=$(git describe --tags --exact-match 2>/dev/null || echo "unknown")
    fi
    
    export GIT_COMMIT="$git_commit"
    export GIT_BRANCH="$git_branch"
    export GIT_TAG="$git_tag"
    
    log_debug "Git commit: $git_commit"
    log_debug "Git branch: $git_branch"
    log_debug "Git tag: $git_tag"
}

# Generate build metadata
generate_build_metadata() {
    export BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    export BUILD_USER=$(whoami)
    export BUILD_HOST=$(hostname)
    
    log_debug "Build date: $BUILD_DATE"
    log_debug "Build user: $BUILD_USER"
    log_debug "Build host: $BUILD_HOST"
}

# Construct image tags
construct_image_tags() {
    local base_name="networkx-mcp"
    
    if [[ -n "$DOCKER_REGISTRY" ]]; then
        base_name="${DOCKER_REGISTRY}/${base_name}"
    fi
    
    # Main version tag
    IMAGE_TAGS=("${base_name}:${VERSION}")
    
    # Environment-specific tag
    if [[ "$BUILD_TYPE" != "production" ]]; then
        IMAGE_TAGS+=("${base_name}:${VERSION}-${BUILD_TYPE}")
    fi
    
    # Latest tag
    if [[ "$TAG_LATEST" == "true" ]]; then
        if [[ "$BUILD_TYPE" == "production" ]]; then
            IMAGE_TAGS+=("${base_name}:latest")
        else
            IMAGE_TAGS+=("${base_name}:${BUILD_TYPE}-latest")
        fi
    fi
    
    # Git-based tags
    if [[ "$GIT_TAG" != "unknown" ]]; then
        IMAGE_TAGS+=("${base_name}:${GIT_TAG}")
    fi
    
    if [[ "$GIT_COMMIT" != "unknown" ]]; then
        IMAGE_TAGS+=("${base_name}:${GIT_COMMIT:0:8}")
    fi
    
    log_debug "Image tags: ${IMAGE_TAGS[*]}"
}

# Prepare build arguments
prepare_build_args() {
    local args=""
    
    # Standard build arguments
    args="$args --build-arg BUILD_ENV=${BUILD_TYPE}"
    args="$args --build-arg VERSION=${VERSION}"
    args="$args --build-arg BUILD_DATE=${BUILD_DATE}"
    args="$args --build-arg GIT_COMMIT=${GIT_COMMIT}"
    args="$args --build-arg GIT_BRANCH=${GIT_BRANCH}"
    args="$args --build-arg GIT_TAG=${GIT_TAG}"
    args="$args --build-arg BUILD_USER=${BUILD_USER}"
    args="$args --build-arg BUILD_HOST=${BUILD_HOST}"
    
    # Target stage based on build type
    if [[ "$BUILD_TYPE" == "development" ]]; then
        args="$args --target development"
    else
        args="$args --target runtime"
    fi
    
    # Cache arguments
    if [[ -n "$CACHE_FROM" ]]; then
        args="$args --cache-from ${CACHE_FROM}"
    fi
    
    # No cache option
    if [[ "$NO_CACHE" == "true" ]]; then
        args="$args --no-cache"
    fi
    
    # Custom build arguments
    if [[ -n "$BUILD_ARGS" ]]; then
        args="$args $BUILD_ARGS"
    fi
    
    # Tag arguments
    for tag in "${IMAGE_TAGS[@]}"; do
        args="$args --tag $tag"
    done
    
    echo "$args"
}

# Run pre-build checks
run_pre_build_checks() {
    log_step "Running pre-build checks..."
    
    # Check if pyproject.toml exists
    if [[ ! -f "$BUILD_CONTEXT/pyproject.toml" ]]; then
        log_error "pyproject.toml not found in build context"
        exit 1
    fi
    
    # Validate Python version requirement
    if ! grep -q "requires-python.*>=.*3\.11" "$BUILD_CONTEXT/pyproject.toml"; then
        log_warn "Python version requirement should be >= 3.11"
    fi
    
    # Check for security vulnerabilities (if tools available)
    if command -v trivy >/dev/null 2>&1; then
        log_info "Running Trivy filesystem scan..."
        trivy fs --exit-code 0 --severity HIGH,CRITICAL "$BUILD_CONTEXT"
    fi
    
    # Run linting if in development mode
    if [[ "$BUILD_TYPE" == "development" ]] && command -v python >/dev/null 2>&1; then
        log_info "Running code quality checks..."
        cd "$BUILD_CONTEXT"
        
        # Install and run ruff if available
        if python -m pip list | grep -q ruff; then
            python -m ruff check src/ --exit-zero
        fi
    fi
}

# Build image
build_image() {
    log_step "Building Docker image..."
    
    cd "$BUILD_CONTEXT"
    
    local build_args
    build_args=$(prepare_build_args)
    
    log_info "Build type: $BUILD_TYPE"
    log_info "Version: $VERSION"
    log_info "Target platforms: $PLATFORMS"
    log_info "Image tags: ${IMAGE_TAGS[*]}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would execute the following build command:"
        if [[ "$MULTI_ARCH" == "true" ]]; then
            echo "docker buildx build --platform $PLATFORMS $build_args --file $DOCKERFILE ."
        else
            echo "docker build $build_args --file $DOCKERFILE ."
        fi
        return
    fi
    
    # Execute build
    local start_time
    start_time=$(date +%s)
    
    if [[ "$MULTI_ARCH" == "true" ]]; then
        # Multi-architecture build with buildx
        local buildx_args="$build_args --platform $PLATFORMS"
        
        if [[ "$PUSH_IMAGE" == "true" ]]; then
            buildx_args="$buildx_args --push"
        else
            buildx_args="$buildx_args --load"
        fi
        
        docker buildx build $buildx_args --file "$DOCKERFILE" .
    else
        # Single architecture build
        docker build $build_args --file "$DOCKERFILE" .
    fi
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_info "Build completed in ${duration} seconds"
}

# Test built image
test_image() {
    if [[ "$MULTI_ARCH" == "true" ]] && [[ "$PUSH_IMAGE" == "true" ]]; then
        log_info "Skipping image testing for multi-arch push build"
        return
    fi
    
    log_step "Testing built image..."
    
    local test_image="${IMAGE_TAGS[0]}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would test image: $test_image"
        return
    fi
    
    # Basic health check
    log_info "Running health check..."
    if docker run --rm "$test_image" health; then
        log_info "✅ Health check passed"
    else
        log_error "❌ Health check failed"
        return 1
    fi
    
    # Test that the application starts
    log_info "Testing application startup..."
    local container_id
    container_id=$(docker run -d -p 8080:8000 "$test_image")
    
    # Wait for startup
    sleep 10
    
    # Test health endpoint
    if curl -f -s http://localhost:8080/health >/dev/null; then
        log_info "✅ Application started successfully"
    else
        log_error "❌ Application failed to start"
        docker logs "$container_id"
        docker kill "$container_id" 2>/dev/null || true
        return 1
    fi
    
    # Cleanup
    docker kill "$container_id" 2>/dev/null || true
}

# Push image to registry
push_image() {
    if [[ "$PUSH_IMAGE" != "true" ]]; then
        log_info "Skipping image push"
        return
    fi
    
    if [[ -z "$DOCKER_REGISTRY" ]]; then
        log_error "Docker registry not specified for push"
        exit 1
    fi
    
    if [[ "$MULTI_ARCH" == "true" ]]; then
        log_info "Multi-arch images already pushed during build"
        return
    fi
    
    log_step "Pushing images to registry..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would push images: ${IMAGE_TAGS[*]}"
        return
    fi
    
    for tag in "${IMAGE_TAGS[@]}"; do
        log_info "Pushing $tag..."
        docker push "$tag"
    done
    
    log_info "All images pushed successfully"
}

# Generate build report
generate_build_report() {
    log_step "Generating build report..."
    
    local report_file="build-report-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "$report_file" << EOF
{
  "build_info": {
    "type": "$BUILD_TYPE",
    "version": "$VERSION",
    "date": "$BUILD_DATE",
    "user": "$BUILD_USER",
    "host": "$BUILD_HOST"
  },
  "git_info": {
    "commit": "$GIT_COMMIT",
    "branch": "$GIT_BRANCH",
    "tag": "$GIT_TAG"
  },
  "image_info": {
    "tags": $(printf '%s\n' "${IMAGE_TAGS[@]}" | jq -R . | jq -s .),
    "platforms": "$PLATFORMS",
    "multi_arch": $MULTI_ARCH,
    "pushed": $PUSH_IMAGE
  },
  "build_args": $(echo "$BUILD_ARGS" | jq -Rs 'split(" ")')
}
EOF
    
    log_info "Build report saved to: $report_file"
}

# Cleanup function
cleanup() {
    log_debug "Cleaning up..."
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
}

# Signal handlers
trap cleanup EXIT INT TERM

# Main function
main() {
    log_info "🏗️ Starting NetworkX MCP Server build"
    log_info "Build type: $BUILD_TYPE"
    log_info "Version: $VERSION"
    log_info "Dry run: $DRY_RUN"
    
    # Validate dependencies
    validate_dependencies
    
    # Get Git information
    get_git_info
    
    # Generate build metadata
    generate_build_metadata
    
    # Construct image tags
    construct_image_tags
    
    # Run pre-build checks
    run_pre_build_checks
    
    # Build image
    build_image
    
    # Test built image
    test_image
    
    # Push image to registry
    push_image
    
    # Generate build report
    if [[ "$DRY_RUN" == "false" ]]; then
        generate_build_report
    fi
    
    log_info "🎉 Build completed successfully!"
    
    # Show image information
    log_info "Built images:"
    for tag in "${IMAGE_TAGS[@]}"; do
        log_info "  - $tag"
    done
}

# Parse arguments and run main function
parse_args "$@"
main