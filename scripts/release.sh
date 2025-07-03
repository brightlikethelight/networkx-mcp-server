#!/bin/bash
set -euo pipefail

# NetworkX MCP Server Release Management Script
# Handles semantic versioning, changelog generation, and automated releases

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default configuration
RELEASE_TYPE="patch"
PRE_RELEASE=""
BUILD_METADATA=""
DRY_RUN="false"
SKIP_TESTS="false"
SKIP_BUILD="false"
SKIP_CHANGELOG="false"
SKIP_TAG="false"
SKIP_PUSH="false"
AUTO_CONFIRM="false"
BRANCH="main"
REMOTE="origin"

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

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

# Help function
show_help() {
    cat << EOF
NetworkX MCP Server Release Management Script

Usage: $0 [OPTIONS]

OPTIONS:
    -t, --type TYPE              Release type: major, minor, patch (default: patch)
    -p, --pre-release SUFFIX     Pre-release suffix (alpha, beta, rc)
    -b, --build BUILD            Build metadata
    -v, --version VERSION        Explicit version (overrides type-based calculation)
    --branch BRANCH              Release branch (default: main)
    --remote REMOTE              Git remote (default: origin)
    --dry-run                    Show what would be done without executing
    --skip-tests                 Skip running tests
    --skip-build                 Skip building artifacts
    --skip-changelog             Skip changelog generation
    --skip-tag                   Skip creating Git tag
    --skip-push                  Skip pushing to remote
    --auto-confirm               Skip confirmation prompts
    --debug                      Enable debug logging
    -h, --help                   Show this help message

EXAMPLES:
    # Patch release
    $0 -t patch

    # Minor release with changelog
    $0 -t minor

    # Pre-release
    $0 -t minor -p beta

    # Release with build metadata
    $0 -t patch -b "build.123"

    # Dry run to see what would happen
    $0 -t major --dry-run

    # Explicit version
    $0 -v 2.1.0

ENVIRONMENT VARIABLES:
    GITHUB_TOKEN                 GitHub token for creating releases
    SLACK_WEBHOOK_URL           Slack webhook for notifications
    DOCKER_REGISTRY             Docker registry for image publishing
    DEBUG                       Enable debug mode

SEMANTIC VERSIONING:
    The script follows semantic versioning (SemVer) specification:
    - MAJOR: Incompatible API changes
    - MINOR: Backwards-compatible functionality additions
    - PATCH: Backwards-compatible bug fixes
    - PRE-RELEASE: Optional pre-release identifier
    - BUILD: Optional build metadata

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                RELEASE_TYPE="$2"
                shift 2
                ;;
            -p|--pre-release)
                PRE_RELEASE="$2"
                shift 2
                ;;
            -b|--build)
                BUILD_METADATA="$2"
                shift 2
                ;;
            -v|--version)
                EXPLICIT_VERSION="$2"
                shift 2
                ;;
            --branch)
                BRANCH="$2"
                shift 2
                ;;
            --remote)
                REMOTE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN="true"
                shift
                ;;
            --skip-tests)
                SKIP_TESTS="true"
                shift
                ;;
            --skip-build)
                SKIP_BUILD="true"
                shift
                ;;
            --skip-changelog)
                SKIP_CHANGELOG="true"
                shift
                ;;
            --skip-tag)
                SKIP_TAG="true"
                shift
                ;;
            --skip-push)
                SKIP_PUSH="true"
                shift
                ;;
            --auto-confirm)
                AUTO_CONFIRM="true"
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

# Validate release type
validate_release_type() {
    case $RELEASE_TYPE in
        major|minor|patch)
            ;;
        *)
            log_error "Invalid release type: $RELEASE_TYPE. Must be major, minor, or patch"
            exit 1
            ;;
    esac
}

# Get current version from pyproject.toml
get_current_version() {
    if [[ ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        log_error "pyproject.toml not found"
        exit 1
    fi
    
    local version
    version=$(grep -E '^version = ' "$PROJECT_ROOT/pyproject.toml" | sed 's/version = "\(.*\)"/\1/')
    
    if [[ -z "$version" ]]; then
        log_error "Could not find version in pyproject.toml"
        exit 1
    fi
    
    echo "$version"
}

# Parse semantic version
parse_version() {
    local version="$1"
    
    # Remove 'v' prefix if present
    version="${version#v}"
    
    # Parse semantic version using regex
    if [[ $version =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)(-([a-zA-Z0-9\.-]+))?(\+([a-zA-Z0-9\.-]+))?$ ]]; then
        MAJOR="${BASH_REMATCH[1]}"
        MINOR="${BASH_REMATCH[2]}"
        PATCH="${BASH_REMATCH[3]}"
        PRERELEASE="${BASH_REMATCH[5]}"
        BUILD="${BASH_REMATCH[7]}"
    else
        log_error "Invalid semantic version format: $version"
        exit 1
    fi
}

# Calculate next version
calculate_next_version() {
    local current_version="$1"
    
    parse_version "$current_version"
    
    local new_major="$MAJOR"
    local new_minor="$MINOR"
    local new_patch="$PATCH"
    
    case $RELEASE_TYPE in
        major)
            ((new_major++))
            new_minor=0
            new_patch=0
            ;;
        minor)
            ((new_minor++))
            new_patch=0
            ;;
        patch)
            ((new_patch++))
            ;;
    esac
    
    local new_version="${new_major}.${new_minor}.${new_patch}"
    
    # Add pre-release suffix if specified
    if [[ -n "$PRE_RELEASE" ]]; then
        new_version="${new_version}-${PRE_RELEASE}"
    fi
    
    # Add build metadata if specified
    if [[ -n "$BUILD_METADATA" ]]; then
        new_version="${new_version}+${BUILD_METADATA}"
    fi
    
    echo "$new_version"
}

# Update version in files
update_version_files() {
    local new_version="$1"
    
    log_step "Updating version in project files..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would update version to $new_version in:"
        log_info "  - pyproject.toml"
        log_info "  - src/networkx_mcp/__version__.py"
        log_info "  - helm/networkx-mcp/Chart.yaml"
        return
    fi
    
    # Update pyproject.toml
    sed -i.bak "s/^version = .*/version = \"$new_version\"/" "$PROJECT_ROOT/pyproject.toml"
    
    # Update __version__.py
    if [[ -f "$PROJECT_ROOT/src/networkx_mcp/__version__.py" ]]; then
        sed -i.bak "s/__version__ = .*/__version__ = \"$new_version\"/" "$PROJECT_ROOT/src/networkx_mcp/__version__.py"
    fi
    
    # Update Helm Chart.yaml
    if [[ -f "$PROJECT_ROOT/helm/networkx-mcp/Chart.yaml" ]]; then
        sed -i.bak "s/^version: .*/version: $new_version/" "$PROJECT_ROOT/helm/networkx-mcp/Chart.yaml"
        sed -i.bak "s/^appVersion: .*/appVersion: \"$new_version\"/" "$PROJECT_ROOT/helm/networkx-mcp/Chart.yaml"
    fi
    
    # Remove backup files
    find "$PROJECT_ROOT" -name "*.bak" -delete 2>/dev/null || true
    
    log_info "Version updated to $new_version"
}

# Generate changelog
generate_changelog() {
    local new_version="$1"
    local current_version="$2"
    
    if [[ "$SKIP_CHANGELOG" == "true" ]]; then
        log_info "Skipping changelog generation"
        return
    fi
    
    log_step "Generating changelog..."
    
    local changelog_file="$PROJECT_ROOT/CHANGELOG.md"
    local temp_changelog=$(mktemp)
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would generate changelog for $current_version -> $new_version"
        return
    fi
    
    # Get commits since last tag
    local last_tag=""
    if git tag --list | grep -q "v${current_version}"; then
        last_tag="v${current_version}"
    else
        last_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
    fi
    
    local commit_range=""
    if [[ -n "$last_tag" ]]; then
        commit_range="${last_tag}..HEAD"
    else
        commit_range="HEAD"
    fi
    
    # Generate changelog entry
    cat > "$temp_changelog" << EOF
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [${new_version}] - $(date +%Y-%m-%d)

EOF
    
    # Categorize commits
    local features=""
    local fixes=""
    local breaking=""
    local other=""
    
    while IFS= read -r commit; do
        if [[ $commit =~ ^feat(\(.*\))?!?: ]]; then
            breaking="$breaking\n- ${commit#*: }"
        elif [[ $commit =~ ^feat(\(.*\))?: ]]; then
            features="$features\n- ${commit#*: }"
        elif [[ $commit =~ ^fix(\(.*\))?: ]]; then
            fixes="$fixes\n- ${commit#*: }"
        else
            other="$other\n- $commit"
        fi
    done < <(git log --pretty=format:"%s" "$commit_range" 2>/dev/null)
    
    # Add sections to changelog
    if [[ -n "$breaking" ]]; then
        echo "### ⚠️ BREAKING CHANGES" >> "$temp_changelog"
        echo -e "$breaking" >> "$temp_changelog"
        echo "" >> "$temp_changelog"
    fi
    
    if [[ -n "$features" ]]; then
        echo "### ✨ Features" >> "$temp_changelog"
        echo -e "$features" >> "$temp_changelog"
        echo "" >> "$temp_changelog"
    fi
    
    if [[ -n "$fixes" ]]; then
        echo "### 🐛 Bug Fixes" >> "$temp_changelog"
        echo -e "$fixes" >> "$temp_changelog"
        echo "" >> "$temp_changelog"
    fi
    
    if [[ -n "$other" ]]; then
        echo "### 🔧 Other Changes" >> "$temp_changelog"
        echo -e "$other" >> "$temp_changelog"
        echo "" >> "$temp_changelog"
    fi
    
    # Merge with existing changelog
    if [[ -f "$changelog_file" ]]; then
        # Insert new entry after the header
        local header_lines=5
        {
            head -n $header_lines "$changelog_file"
            tail -n +6 "$temp_changelog"
            tail -n +$((header_lines + 1)) "$changelog_file"
        } > "${changelog_file}.new"
        mv "${changelog_file}.new" "$changelog_file"
    else
        cp "$temp_changelog" "$changelog_file"
    fi
    
    rm "$temp_changelog"
    
    log_info "Changelog generated: $changelog_file"
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_info "Skipping tests"
        return
    fi
    
    log_step "Running tests..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would run test suite"
        return
    fi
    
    cd "$PROJECT_ROOT"
    
    # Run test automation script
    if [[ -f "scripts/test_automation.py" ]]; then
        python scripts/test_automation.py --quality-gate
    else
        # Fallback to pytest
        python -m pytest tests/ -v --tb=short
    fi
}

# Build artifacts
build_artifacts() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        log_info "Skipping build"
        return
    fi
    
    log_step "Building release artifacts..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would build release artifacts"
        return
    fi
    
    cd "$PROJECT_ROOT"
    
    # Build Python package
    python -m build
    
    # Build Docker image
    if [[ -f "scripts/build.sh" ]]; then
        ./scripts/build.sh -t production -v "$NEW_VERSION" --tag-latest
    fi
    
    # Build Helm package
    if [[ -d "helm/networkx-mcp" ]]; then
        helm package helm/networkx-mcp/ --destination dist/
    fi
}

# Create Git tag
create_git_tag() {
    local new_version="$1"
    
    if [[ "$SKIP_TAG" == "true" ]]; then
        log_info "Skipping Git tag creation"
        return
    fi
    
    log_step "Creating Git tag..."
    
    local tag_name="v${new_version}"
    local tag_message="Release ${new_version}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would create tag $tag_name with message: $tag_message"
        return
    fi
    
    # Create annotated tag
    git tag -a "$tag_name" -m "$tag_message"
    
    log_info "Created Git tag: $tag_name"
}

# Push changes
push_changes() {
    if [[ "$SKIP_PUSH" == "true" ]]; then
        log_info "Skipping push to remote"
        return
    fi
    
    log_step "Pushing changes to remote..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would push commits and tags to $REMOTE"
        return
    fi
    
    # Push commits
    git push "$REMOTE" "$BRANCH"
    
    # Push tags
    git push "$REMOTE" --tags
    
    log_info "Changes pushed to $REMOTE"
}

# Create GitHub release
create_github_release() {
    local new_version="$1"
    
    if [[ -z "${GITHUB_TOKEN:-}" ]]; then
        log_warn "GITHUB_TOKEN not set, skipping GitHub release creation"
        return
    fi
    
    log_step "Creating GitHub release..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: Would create GitHub release for v$new_version"
        return
    fi
    
    local tag_name="v${new_version}"
    local release_name="Release ${new_version}"
    local release_notes=""
    
    # Extract release notes from changelog
    if [[ -f "$PROJECT_ROOT/CHANGELOG.md" ]]; then
        release_notes=$(awk "/^## \[${new_version}\]/,/^## \[/{if(/^## \[${new_version}\]/) next; if(/^## \[/) exit; print}" "$PROJECT_ROOT/CHANGELOG.md")
    fi
    
    # Create release using GitHub CLI or API
    if command -v gh >/dev/null 2>&1; then
        gh release create "$tag_name" \
            --title "$release_name" \
            --notes "$release_notes" \
            dist/*
    else
        log_warn "GitHub CLI not available, skipping GitHub release"
    fi
}

# Send notifications
send_notifications() {
    local new_version="$1"
    
    # Slack notification
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        log_step "Sending Slack notification..."
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "DRY RUN: Would send Slack notification"
            return
        fi
        
        local message="🚀 NetworkX MCP Server v${new_version} has been released!"
        
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"$message\"}" \
            "$SLACK_WEBHOOK_URL"
    fi
}

# Confirm release
confirm_release() {
    local current_version="$1"
    local new_version="$2"
    
    if [[ "$AUTO_CONFIRM" == "true" ]]; then
        return
    fi
    
    echo ""
    log_info "📋 Release Summary:"
    log_info "  Current version: $current_version"
    log_info "  New version: $new_version"
    log_info "  Release type: $RELEASE_TYPE"
    log_info "  Branch: $BRANCH"
    log_info "  Remote: $REMOTE"
    echo ""
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "DRY RUN: No confirmation needed"
        return
    fi
    
    read -p "Do you want to proceed with this release? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Release cancelled"
        exit 0
    fi
}

# Validate repository state
validate_repository_state() {
    log_step "Validating repository state..."
    
    cd "$PROJECT_ROOT"
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir >/dev/null 2>&1; then
        log_error "Not in a Git repository"
        exit 1
    fi
    
    # Check if on correct branch
    local current_branch
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    if [[ "$current_branch" != "$BRANCH" ]]; then
        log_error "Not on release branch '$BRANCH' (currently on '$current_branch')"
        exit 1
    fi
    
    # Check for uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        log_error "Repository has uncommitted changes"
        exit 1
    fi
    
    # Check if remote exists
    if ! git remote | grep -q "^${REMOTE}$"; then
        log_error "Remote '$REMOTE' not found"
        exit 1
    fi
    
    # Fetch latest changes
    git fetch "$REMOTE"
    
    # Check if branch is up to date
    local local_commit
    local remote_commit
    local_commit=$(git rev-parse HEAD)
    remote_commit=$(git rev-parse "${REMOTE}/${BRANCH}")
    
    if [[ "$local_commit" != "$remote_commit" ]]; then
        log_error "Local branch is not up to date with remote"
        exit 1
    fi
    
    log_info "Repository state is valid"
}

# Main function
main() {
    log_info "🚀 Starting NetworkX MCP Server release process"
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Validate repository state
    validate_repository_state
    
    # Get current version
    local current_version
    current_version=$(get_current_version)
    log_info "Current version: $current_version"
    
    # Calculate new version
    local new_version
    if [[ -n "${EXPLICIT_VERSION:-}" ]]; then
        new_version="$EXPLICIT_VERSION"
    else
        new_version=$(calculate_next_version "$current_version")
    fi
    
    export NEW_VERSION="$new_version"
    log_info "New version: $new_version"
    
    # Confirm release
    confirm_release "$current_version" "$new_version"
    
    # Run tests
    run_tests
    
    # Update version files
    update_version_files "$new_version"
    
    # Generate changelog
    generate_changelog "$new_version" "$current_version"
    
    # Commit changes
    if [[ "$DRY_RUN" == "false" ]]; then
        git add .
        git commit -m "chore(release): bump version to $new_version"
    fi
    
    # Build artifacts
    build_artifacts
    
    # Create Git tag
    create_git_tag "$new_version"
    
    # Push changes
    push_changes
    
    # Create GitHub release
    create_github_release "$new_version"
    
    # Send notifications
    send_notifications "$new_version"
    
    log_success "🎉 Release $new_version completed successfully!"
    
    # Show next steps
    echo ""
    log_info "📋 Next steps:"
    log_info "  1. Verify the release at: https://github.com/your-org/networkx-mcp-server/releases"
    log_info "  2. Monitor deployment pipeline"
    log_info "  3. Update documentation if needed"
    log_info "  4. Announce the release to stakeholders"
}

# Parse arguments and validate
parse_args "$@"
validate_release_type

# Run main function
main