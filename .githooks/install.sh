#!/bin/bash
# Git hooks installation script for NetworkX MCP Server
# Sets up all git hooks with proper permissions and configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}     NetworkX MCP Server - Git Hooks Installation${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo -e "${RED}❌ Error: Not in a git repository root directory${NC}"
    echo "Please run this script from the repository root."
    exit 1
fi

# Detect OS for compatibility
OS="$(uname -s)"
case "${OS}" in
    Linux*)     OS_TYPE=Linux;;
    Darwin*)    OS_TYPE=Mac;;
    CYGWIN*)    OS_TYPE=Windows;;
    MINGW*)     OS_TYPE=Windows;;
    *)          OS_TYPE="UNKNOWN:${OS}"
esac

echo -e "${BLUE}🖥️  Detected OS: ${OS_TYPE}${NC}"
echo ""

# Function to install a hook
install_hook() {
    local hook_name=$1
    local source_file=".githooks/$hook_name"
    local target_file=".git/hooks/$hook_name"
    
    if [ ! -f "$source_file" ]; then
        echo -e "${YELLOW}⚠️  Warning: $source_file not found${NC}"
        return 1
    fi
    
    echo -n "Installing $hook_name hook... "
    
    # Create hooks directory if it doesn't exist
    mkdir -p .git/hooks
    
    # Copy hook file
    cp "$source_file" "$target_file"
    
    # Make executable
    chmod +x "$target_file"
    
    echo -e "${GREEN}✅${NC}"
    return 0
}

# Function to check dependencies
check_dependency() {
    local cmd=$1
    local name=$2
    local install_cmd=$3
    
    if command -v $cmd &> /dev/null; then
        echo -e "  ${GREEN}✅ $name installed${NC}"
        return 0
    else
        echo -e "  ${YELLOW}⚠️  $name not installed${NC}"
        echo -e "     Install with: ${CYAN}$install_cmd${NC}"
        return 1
    fi
}

# Check Python dependencies
echo -e "${BLUE}📦 Checking dependencies...${NC}"
MISSING_DEPS=0

check_dependency "python" "Python" "brew install python3" || MISSING_DEPS=$((MISSING_DEPS + 1))
check_dependency "ruff" "Ruff" "pip install ruff" || MISSING_DEPS=$((MISSING_DEPS + 1))
check_dependency "mypy" "MyPy" "pip install mypy" || MISSING_DEPS=$((MISSING_DEPS + 1))
check_dependency "bandit" "Bandit" "pip install bandit" || MISSING_DEPS=$((MISSING_DEPS + 1))
check_dependency "pytest" "Pytest" "pip install pytest pytest-asyncio pytest-cov" || MISSING_DEPS=$((MISSING_DEPS + 1))

if [ $MISSING_DEPS -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Some dependencies are missing.${NC}"
    echo "The hooks will still be installed but some checks may be skipped."
    read -p "Continue with installation? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Installation cancelled${NC}"
        exit 1
    fi
fi

echo ""

# Install hooks
echo -e "${BLUE}🔧 Installing git hooks...${NC}"
HOOKS_INSTALLED=0

install_hook "pre-commit" && HOOKS_INSTALLED=$((HOOKS_INSTALLED + 1))
install_hook "commit-msg" && HOOKS_INSTALLED=$((HOOKS_INSTALLED + 1))
install_hook "post-commit" && HOOKS_INSTALLED=$((HOOKS_INSTALLED + 1))

echo ""

# Configure git to use hooks directory (alternative method)
echo -e "${BLUE}🔧 Configuring git hooks path...${NC}"
git config core.hooksPath .githooks
echo -e "${GREEN}✅ Git configured to use .githooks directory${NC}"

# Create metrics directory
echo ""
echo -e "${BLUE}📊 Setting up metrics tracking...${NC}"
mkdir -p .git/metrics
echo -e "${GREEN}✅ Metrics directory created${NC}"

# Configure hook bypass (for emergencies)
echo ""
echo -e "${BLUE}🚨 Configuring emergency bypass...${NC}"
git config --local hooks.skip false
echo -e "${GREEN}✅ Hook bypass configured (use 'git config hooks.skip true' to bypass)${NC}"

# Optional: Install pre-commit framework
echo ""
echo -e "${BLUE}🎯 Optional: Install pre-commit framework${NC}"
if command -v pre-commit &> /dev/null; then
    echo -e "${GREEN}✅ pre-commit framework detected${NC}"
    read -p "Install pre-commit hooks from .pre-commit-config.yaml? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pre-commit install
        echo -e "${GREEN}✅ pre-commit framework hooks installed${NC}"
    fi
else
    echo -e "${YELLOW}ℹ️  pre-commit framework not installed${NC}"
    echo -e "   Install with: ${CYAN}pip install pre-commit${NC}"
fi

# Create uninstall script
echo ""
echo -e "${BLUE}📝 Creating uninstall script...${NC}"
cat > .githooks/uninstall.sh << 'EOF'
#!/bin/bash
# Uninstall git hooks

echo "Removing git hooks..."
rm -f .git/hooks/pre-commit
rm -f .git/hooks/commit-msg
rm -f .git/hooks/post-commit
git config --unset core.hooksPath
git config --unset hooks.skip
echo "✅ Git hooks removed"
EOF
chmod +x .githooks/uninstall.sh
echo -e "${GREEN}✅ Uninstall script created${NC}"

# Summary
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Installation Complete!${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}📋 Summary:${NC}"
echo -e "  • Hooks installed: $HOOKS_INSTALLED"
echo -e "  • Hooks directory: .githooks/"
echo -e "  • Metrics directory: .git/metrics/"
echo ""
echo -e "${BLUE}💡 Usage:${NC}"
echo -e "  • Hooks will run automatically on git operations"
echo -e "  • To bypass hooks temporarily: ${CYAN}git commit --no-verify${NC}"
echo -e "  • To disable all hooks: ${CYAN}git config hooks.skip true${NC}"
echo -e "  • To re-enable hooks: ${CYAN}git config hooks.skip false${NC}"
echo -e "  • To uninstall: ${CYAN}.githooks/uninstall.sh${NC}"
echo ""
echo -e "${BLUE}🧪 Test the hooks:${NC}"
echo -e "  ${CYAN}echo 'test' > test.txt && git add test.txt && git commit -m 'test'${NC}"
echo ""
echo -e "${GREEN}Happy coding! 🚀${NC}"