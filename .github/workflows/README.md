# GitHub Actions Workflows

This directory contains the CI/CD pipelines for the NetworkX MCP Server project.

## Workflows Overview

### 🔄 Continuous Integration (`ci.yml`)

- **Trigger**: Push to main, pull requests
- **Purpose**: Validate code quality and functionality
- **Jobs**:
  - **Test**: Runs on multiple OS (Ubuntu, macOS, Windows) and Python versions (3.11, 3.12)
  - **Pre-commit**: Validates code formatting and style
  - **Docs**: Checks documentation and broken links
  - **Build**: Verifies package builds correctly

### 🚀 Release (`release.yml`)

- **Trigger**: Git tags matching `v*` pattern
- **Purpose**: Automated release to PyPI and Docker Hub
- **Jobs**:
  - **Validate**: Comprehensive testing and version validation
  - **Build Python**: Creates wheel and source distributions
  - **Build Docker**: Multi-platform Docker images (amd64, arm64)
  - **Publish PyPI**: Uses trusted publishing (no tokens in secrets)
  - **Create Release**: Generates GitHub release with changelog
  - **Notify**: Summary of release status

### 🔒 Security (`security.yml`)

- **Trigger**: Push to main, PRs, weekly schedule
- **Purpose**: Security vulnerability scanning
- **Tools**:
  - **Bandit**: Python code security analysis
  - **Safety**: Dependency vulnerability scanning
  - **Semgrep**: Advanced static analysis
- **Features**: Uploads SARIF results to GitHub Security tab

### 🐳 Docker Build (`docker-build.yml`)

- **Trigger**: Push to main, pull requests
- **Purpose**: Continuous Docker image building
- **Features**:
  - Multi-platform builds (amd64, arm64)
  - MCP protocol validation
  - Automatic push to GitHub Container Registry

### 📊 Benchmarks (`benchmarks.yml`)

- **Trigger**: Manual dispatch
- **Purpose**: Performance benchmarking
- **Features**: Compares performance across versions

### 🔍 CodeQL (`codeql.yml`)

- **Trigger**: Push, PRs, weekly schedule
- **Purpose**: Advanced code analysis for security vulnerabilities

### 📦 Dependency Update (`dependency-update.yml`)

- **Trigger**: Monthly schedule
- **Purpose**: Automated dependency updates via Dependabot

### 📚 Documentation (`docs.yml`)

- **Trigger**: Push to main, PRs affecting docs
- **Purpose**: Build and deploy documentation

## Best Practices Implemented

### 1. **Concurrency Control**

All workflows use concurrency groups to prevent duplicate runs:

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

### 2. **Environment Protection**

PyPI publishing uses environment protection:

```yaml
environment: pypi
permissions:
  id-token: write  # For trusted publishing
```

### 3. **Caching**

- Python dependencies cached via `setup-python` action
- Docker layers cached via GitHub Actions cache
- Pre-commit environments cached

### 4. **Error Handling**

- Tests fail fast but continue on other platforms
- Security scans continue even if issues found
- Release notifications sent regardless of status

### 5. **Multi-Platform Support**

- CI tests on Ubuntu, macOS, and Windows
- Docker images built for amd64 and arm64
- Python 3.11 and 3.12 supported

## Secrets Required

### Repository Secrets

- `CODECOV_TOKEN`: For coverage reporting (optional)
- `GITHUB_TOKEN`: Automatically provided by GitHub Actions

### Environment Secrets (pypi environment)

- None required - uses trusted publishing

## Status Badges

Add these to your README:

```markdown
[![CI](https://github.com/YOUR_ORG/networkx-mcp-server/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_ORG/networkx-mcp-server/actions/workflows/ci.yml)
[![Release](https://github.com/YOUR_ORG/networkx-mcp-server/actions/workflows/release.yml/badge.svg)](https://github.com/YOUR_ORG/networkx-mcp-server/actions/workflows/release.yml)
[![Security](https://github.com/YOUR_ORG/networkx-mcp-server/actions/workflows/security.yml/badge.svg)](https://github.com/YOUR_ORG/networkx-mcp-server/actions/workflows/security.yml)
[![Docker](https://github.com/YOUR_ORG/networkx-mcp-server/actions/workflows/docker-build.yml/badge.svg)](https://github.com/YOUR_ORG/networkx-mcp-server/actions/workflows/docker-build.yml)
```

## Maintenance

### Updating Action Versions

1. Check for updates: `gh extension install actions/gh-actions-cache`
2. Update versions in all workflow files
3. Test changes in a pull request

### Adding New Workflows

1. Create workflow file in `.github/workflows/`
2. Add concurrency control
3. Use consistent environment variables
4. Document in this README

### Debugging Failures

1. Check workflow run logs in Actions tab
2. Use `workflow_dispatch` for manual testing
3. Add `ACTIONS_STEP_DEBUG: true` secret for verbose logs
