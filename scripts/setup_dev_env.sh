#!/bin/bash
# Development environment setup for NetworkX MCP Server
# This script sets up a complete development environment with all tools

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${BLUE}▶ $1${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check requirements
check_requirements() {
    print_header "Checking Requirements"

    # Check Python version
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    REQUIRED_VERSION="3.11"

    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)"; then
        print_error "Python ${REQUIRED_VERSION}+ is required, but you have ${PYTHON_VERSION}"
        exit 1
    fi

    print_success "Python ${PYTHON_VERSION} detected"

    # Check pip
    if ! command -v pip &> /dev/null; then
        print_error "pip is required but not installed"
        exit 1
    fi

    print_success "pip is available"

    # Check git
    if ! command -v git &> /dev/null; then
        print_error "git is required but not installed"
        exit 1
    fi

    print_success "git is available"
}

# Create virtual environment
setup_venv() {
    print_header "Setting Up Virtual Environment"

    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists, removing..."
        rm -rf venv
    fi

    python3 -m venv venv
    print_success "Virtual environment created"

    # Activate virtual environment
    source venv/bin/activate
    print_success "Virtual environment activated"

    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    print_success "pip upgraded"
}

# Install dependencies
install_dependencies() {
    print_header "Installing Dependencies"

    # Install development dependencies
    echo "Installing development dependencies..."
    pip install -e ".[dev]"
    print_success "Development dependencies installed"

    # Install additional testing tools
    echo "Installing additional tools..."
    pip install pre-commit tox
    print_success "Additional tools installed"
}

# Setup pre-commit hooks
setup_precommit() {
    print_header "Setting Up Pre-commit Hooks"

    pre-commit install
    pre-commit install --hook-type commit-msg
    print_success "Pre-commit hooks installed"

    # Run pre-commit on all files to ensure everything works
    echo "Running pre-commit on all files..."
    pre-commit run --all-files || {
        print_warning "Pre-commit found issues, but that's normal for initial setup"
    }
    print_success "Pre-commit setup complete"
}

# Setup testing environment
setup_testing() {
    print_header "Setting Up Testing Environment"

    # Create necessary directories
    mkdir -p .hypothesis
    mkdir -p htmlcov
    mkdir -p .benchmarks

    # Run a quick test to ensure everything works
    echo "Running quick test..."
    python -m pytest tests/unit/test_basic.py -v || {
        print_warning "Some tests failed, but environment is set up"
    }

    print_success "Testing environment ready"
}

# Setup IDE configuration
setup_ide() {
    print_header "Setting Up IDE Configuration"

    # Create .vscode directory and settings
    mkdir -p .vscode

    cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": false,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests/"
    ],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".mypy_cache": true,
        ".pytest_cache": true,
        ".ruff_cache": true,
        "htmlcov": true
    },
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": "explicit"
    }
}
EOF

    print_success "VS Code configuration created"
}

# Final verification
verify_setup() {
    print_header "Verifying Setup"

    # Check imports
    python -c "
import networkx
import pytest
import ruff
import mypy
import bandit
print('✓ All required packages can be imported')
"

    # Check CLI tools
    python -m pytest --version > /dev/null
    python -m ruff --version > /dev/null
    python -m mypy --version > /dev/null

    print_success "All tools are working correctly"
}

# Main execution
main() {
    echo "🚀 NetworkX MCP Server - Development Environment Setup"
    echo "======================================================"

    check_requirements
    setup_venv
    install_dependencies
    setup_precommit
    setup_testing
    setup_ide
    verify_setup

    print_header "Setup Complete!"
    echo -e "${GREEN}Your development environment is ready!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Run tests: python -m pytest"
    echo "3. Run linting: python -m ruff check src/"
    echo "4. Start developing! 🎉"
    echo ""
    echo "Useful commands:"
    echo "• Run all tests: python -m pytest"
    echo "• Run with coverage: python -m pytest --cov=src/networkx_mcp"
    echo "• Run security tests: python -m pytest tests/security/"
    echo "• Run performance tests: python -m pytest tests/performance/ --benchmark-only"
    echo "• Format code: python -m black src/ tests/"
    echo "• Check types: python -m mypy src/"
    echo "• Run mutation tests: python -m mutmut run"
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
