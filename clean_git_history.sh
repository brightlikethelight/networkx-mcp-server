#!/bin/bash

# This script cleans up the git history to remove Claude references and squash related commits

# Create a backup branch first
git branch backup-main

# Start interactive rebase from the beginning
echo "Starting interactive rebase to clean commit history..."

# Create a file with rebase commands
cat > /tmp/rebase-todo.txt << 'EOF'
# Squash early development commits
pick e934bd3 feat(core): implement Phase 1 core graph operations
squash 3cd17b7 feat(advanced): implement Phase 2 advanced analytics
squash 9de0b2d feat(viz): implement Phase 3 visualization and integration
squash f1ec3ea feat(server): implement MCP server with 39 graph analysis tools
reword 32c4f0f test: add comprehensive pytest test suite
pick 344b2f2 chore: add project configuration and documentation
pick 73ad400 fix: resolve deprecation warnings and test failures
pick c6c8067 feat: add production readiness tools and final testing

# Squash linting fixes
pick 5dd06d8 fix: resolve critical ruff linting errors
squash e52379e fix: resolve all 10 critical CI ruff errors
squash f8c7aa1 fix: comprehensive ruff linting fixes across entire codebase
squash b21f96b fix: partial ruff linting fixes - work in progress

# Keep refactoring
pick a32c5bd refactor: major code quality improvements and production readiness

# Squash dependency fixes
pick 34e8e62 fix: upgrade to Pydantic v2 and resolve all dependency conflicts
squash f6f19c3 fix: resolve CI/CD issues and missing dependencies
squash 15d7c52 fix: add missing type stub for PyMySQL

# Keep docs and cleanup
pick 7fc5e3b chore: comprehensive code cleanup and documentation update
pick bd6cf11 docs: add release notes and cleanup summary

# Squash CI/CD fixes
pick 08982b6 fix: resolve all GitHub Actions CI/CD failures
squash 9064908 fix: make scikit-learn optional dependency
squash 51b38a3 fix: make visualization libraries optional dependencies
squash 283766e fix: make enterprise dependencies optional
squash dbf1a7d fix: make all optional dependencies truly optional
squash 53165d7 fix: resolve all CI/CD pipeline failures
EOF

echo "Rebase plan created. You'll need to:"
echo "1. Run: git rebase -i --root"
echo "2. Replace the content with the plan above"
echo "3. Save and close the editor"
echo "4. Edit commit messages to remove Claude references"
echo ""
echo "To remove Claude references from all commits:"
echo "git filter-branch --msg-filter 'sed -e \"s/🤖 Generated with \[Claude Code\].*//g\" -e \"s/Co-Authored-By: Claude.*//g\"' -- --all"
