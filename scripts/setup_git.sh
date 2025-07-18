#!/bin/bash
# Git Configuration Setup for NetworkX MCP Server
# This script configures Git for optimal development workflow with conventional commits

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Setting up Git configuration for NetworkX MCP Server${NC}"
echo "=================================================="

# Get the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Set up commit message template
echo -e "${YELLOW}📝 Setting up commit message template...${NC}"
git config commit.template .gitmessage
echo -e "${GREEN}✅ Commit message template configured${NC}"

# Configure Git hooks path
echo -e "${YELLOW}🪝 Setting up Git hooks...${NC}"
git config core.hooksPath .githooks

# Create .githooks directory if it doesn't exist
mkdir -p .githooks

# Create commit-msg hook for conventional commits
cat > .githooks/commit-msg << 'EOF'
#!/bin/bash
# Conventional Commits validation hook
# This hook validates commit messages against conventional commit format

commit_regex='^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\(.+\))?: .{1,50}'

error_msg="Aborting commit. Your commit message is invalid. See .gitmessage for the expected format.

Conventional commit format:
<type>(<scope>): <description>

Types:
- feat: A new feature
- fix: A bug fix
- docs: Documentation only changes
- style: Changes that do not affect the meaning of the code
- refactor: A code change that neither fixes a bug nor adds a feature
- perf: A code change that improves performance
- test: Adding missing tests or correcting existing tests
- build: Changes that affect the build system or external dependencies
- ci: Changes to CI configuration files and scripts
- chore: Other changes that don't modify src or test files
- revert: Reverts a previous commit

Examples:
feat(core): add new graph validation system
fix(handlers): resolve memory leak in algorithm service
docs(readme): update installation instructions
test(integration): add comprehensive API tests"

if ! grep -qE "$commit_regex" "$1"; then
    echo "$error_msg" >&2
    exit 1
fi
EOF

# Make hooks executable
chmod +x .githooks/commit-msg

# Create prepare-commit-msg hook
cat > .githooks/prepare-commit-msg << 'EOF'
#!/bin/bash
# Prepare commit message hook
# Adds branch name and issue number to commit messages

COMMIT_MSG_FILE=$1
COMMIT_SOURCE=$2

# Get current branch name
BRANCH_NAME=$(git symbolic-ref --short HEAD 2>/dev/null || echo "detached")

# Skip if not on a feature branch or if it's a merge/squash commit
if [[ "$COMMIT_SOURCE" == "merge" ]] || [[ "$COMMIT_SOURCE" == "squash" ]] || [[ "$BRANCH_NAME" == "main" ]] || [[ "$BRANCH_NAME" == "develop" ]]; then
    exit 0
fi

# Extract issue number from branch name (e.g., feature/123-description -> #123)
ISSUE_NUMBER=""
if [[ $BRANCH_NAME =~ ^(feature|fix|hotfix)/([0-9]+) ]]; then
    ISSUE_NUMBER="#${BASH_REMATCH[2]}"
fi

# If commit message is empty or just comments, don't modify
if ! grep -qv '^#' "$COMMIT_MSG_FILE" 2>/dev/null; then
    exit 0
fi

# Add branch and issue info to commit message
if [[ -n "$ISSUE_NUMBER" ]]; then
    echo "" >> "$COMMIT_MSG_FILE"
    echo "# Branch: $BRANCH_NAME" >> "$COMMIT_MSG_FILE"
    echo "# Related: $ISSUE_NUMBER" >> "$COMMIT_MSG_FILE"
fi
EOF

chmod +x .githooks/prepare-commit-msg

# Create pre-commit hook
cat > .githooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for code quality checks

# Check if pre-commit is installed and configured
if command -v pre-commit >/dev/null 2>&1; then
    echo "🔍 Running pre-commit hooks..."
    pre-commit run --config .pre-commit-config.yaml
else
    echo "⚠️  pre-commit not found. Installing..."
    pip install pre-commit
    pre-commit install
    pre-commit run --config .pre-commit-config.yaml
fi
EOF

chmod +x .githooks/pre-commit

echo -e "${GREEN}✅ Git hooks configured${NC}"

# Configure Git settings for better workflow
echo -e "${YELLOW}⚙️  Configuring Git settings...${NC}"

# Enable automatic setup of upstream branches
git config push.autoSetupRemote true

# Use more descriptive push default
git config push.default simple

# Enable rebase on pull by default
git config pull.rebase true

# Configure merge strategy
git config merge.ff only

# Set up better diff and merge tools
git config diff.algorithm histogram
git config merge.conflictstyle diff3

# Enable helpful Git aliases
git config alias.co checkout
git config alias.br branch
git config alias.ci commit
git config alias.st status
git config alias.lg "log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
git config alias.unstage "reset HEAD --"
git config alias.last "log -1 HEAD"
git config alias.visual "!gitk"
git config alias.amend "commit --amend --no-edit"
git config alias.recommit "commit --amend"
git config alias.squash "merge --squash"
git config alias.conflicts "diff --name-only --diff-filter=U"

# Conventional commit aliases
git config alias.feat "!f() { git commit -m \"feat\$1: \$2\"; }; f"
git config alias.fix "!f() { git commit -m \"fix\$1: \$2\"; }; f"
git config alias.docs "!f() { git commit -m \"docs\$1: \$2\"; }; f"
git config alias.style "!f() { git commit -m \"style\$1: \$2\"; }; f"
git config alias.refactor "!f() { git commit -m \"refactor\$1: \$2\"; }; f"
git config alias.test "!f() { git commit -m \"test\$1: \$2\"; }; f"
git config alias.chore "!f() { git commit -m \"chore\$1: \$2\"; }; f"

echo -e "${GREEN}✅ Git settings configured${NC}"

# Set up commitizen for better commit messages (if Python is available)
if command -v python >/dev/null 2>&1; then
    echo -e "${YELLOW}📝 Setting up Commitizen...${NC}"
    pip install commitizen cz-conventional-commits || {
        echo -e "${YELLOW}⚠️  Could not install commitizen, skipping...${NC}"
    }

    # Configure commitizen
    cat > .cz.yaml << 'EOF'
commitizen:
  name: cz_conventional_commits
  version: 2.0.0
  tag_format: v$version
  update_changelog_on_bump: true
  gpg_sign: false
  major_version_zero: true
  version_files:
    - src/networkx_mcp/__version__.py:version
    - pyproject.toml:version
EOF
    echo -e "${GREEN}✅ Commitizen configured${NC}"
fi

# Install conventional-pre-commit
echo -e "${YELLOW}🔗 Installing conventional-pre-commit...${NC}"
pip install conventional-pre-commit || {
    echo -e "${YELLOW}⚠️  Could not install conventional-pre-commit, skipping...${NC}"
}

# Configure GitHub CLI if available
if command -v gh >/dev/null 2>&1; then
    echo -e "${YELLOW}🐙 Configuring GitHub CLI...${NC}"

    # Set default PR template
    mkdir -p .github/pull_request_template
    cat > .github/pull_request_template/default.md << 'EOF'
## 📋 Summary

<!-- Provide a brief description of your changes -->

## 🔄 Type of Change

- [ ] 🆕 feat: New feature
- [ ] 🐛 fix: Bug fix
- [ ] 📚 docs: Documentation update
- [ ] 🎨 style: Code style changes
- [ ] ♻️ refactor: Code refactoring
- [ ] ⚡ perf: Performance improvement
- [ ] 🧪 test: Testing changes
- [ ] 🔧 build: Build system changes
- [ ] 👷 ci: CI/CD changes
- [ ] 🧹 chore: Other changes

## 🧪 Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] New tests added (if applicable)

## 📝 Checklist

- [ ] Code follows project conventions
- [ ] Self-review completed
- [ ] Documentation updated (if needed)
- [ ] Breaking changes documented
- [ ] Changelog updated (if needed)

## 🔗 Related Issues

<!-- Link to related issues, e.g., Fixes #123, Closes #456 -->

## 📷 Screenshots (if applicable)

<!-- Add screenshots for UI changes -->

## 🚀 Deployment Notes

<!-- Any special deployment considerations -->
EOF

    echo -e "${GREEN}✅ GitHub CLI configured${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Git configuration completed successfully!${NC}"
echo ""
echo -e "${BLUE}💡 Helpful commands:${NC}"
echo "  git feat '(scope): description'  - Create a feature commit"
echo "  git fix '(scope): description'   - Create a bug fix commit"
echo "  git docs ': description'        - Create a docs commit"
echo "  git lg                          - Show pretty log"
echo "  git conflicts                   - Show conflicted files"
echo "  cz commit                       - Interactive commit (if commitizen installed)"
echo ""
echo -e "${YELLOW}📚 Resources:${NC}"
echo "  - Conventional Commits: https://www.conventionalcommits.org/"
echo "  - Git Flow: https://nvie.com/posts/a-successful-git-branching-model/"
echo "  - Commit message template: .gitmessage"
echo ""
echo -e "${BLUE}🔄 Next steps:${NC}"
echo "  1. Run 'pre-commit install' to activate pre-commit hooks"
echo "  2. Use 'git commit' to see the new commit message template"
echo "  3. Try 'git feat \"(core): amazing new feature\"' for quick commits"
