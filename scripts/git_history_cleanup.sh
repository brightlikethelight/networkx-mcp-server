#!/bin/bash

# Git History Cleanup Script
# Removes Claude references from commit messages while preserving commit structure

echo "🧹 Git History Cleanup Script"
echo "=============================="
echo "This script will remove Claude references from all commit messages."
echo "⚠️  WARNING: This will rewrite git history!"
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "❌ Error: You have uncommitted changes. Please commit or stash them first."
    exit 1
fi

# Backup current branch
current_branch=$(git rev-parse --abbrev-ref HEAD)
echo "📋 Current branch: $current_branch"

read -p "Do you want to continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Aborted by user"
    exit 1
fi

echo ""
echo "🔄 Creating backup branch..."
git branch backup-before-cleanup-$(date +%Y%m%d-%H%M%S)

echo "🧹 Cleaning commit messages..."

# Use git filter-branch to clean up commit messages
git filter-branch --force --msg-filter '
    # Remove Claude-specific lines
    sed -E \
        -e "/^🤖 Generated with \[Claude Code\]/d" \
        -e "/^Co-Authored-By: Claude/d" \
        -e "s/🤖 Generated with \[Claude Code\].*$//g" \
        -e "s/Co-Authored-By: Claude.*$//g" \
        -e "/^[[:space:]]*$/d" | \
    # Remove trailing empty lines
    awk "/^$/ {nlstack=nlstack \"\n\"; next;} {printf \"%s\", nlstack; nlstack=\"\"; print;}"
' --tag-name-filter cat -- --all

echo ""
echo "✅ Commit messages cleaned!"
echo ""
echo "📊 Statistics:"
echo "  - Commits processed: $(git rev-list --count HEAD)"
echo "  - Current branch: $current_branch"
echo ""
echo "⚠️  Important next steps:"
echo "  1. Review the changes: git log --oneline"
echo "  2. If satisfied, force push: git push --force-with-lease origin $current_branch"
echo "  3. If not satisfied, restore from backup: git reset --hard backup-before-cleanup-*"
echo ""
echo "🎯 Optional: Clean up backup refs"
echo "  - Remove backup: git branch -D backup-before-cleanup-*"
echo "  - Remove filter-branch backup: rm -rf .git/refs/original/"
echo ""

# Show recent commits for verification
echo "📝 Recent commits (first 10):"
git log --oneline -10

echo ""
echo "✨ Done! Please review the changes before pushing."