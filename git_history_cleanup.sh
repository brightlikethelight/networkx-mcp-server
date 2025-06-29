#!/bin/bash
# Git History Cleanup Script
# This script cleans up the git history by squashing related commits and removing Claude references

echo "🧹 Git History Cleanup"
echo "====================="
echo
echo "⚠️  WARNING: This will rewrite git history!"
echo "Make sure you have backed up your repository before proceeding."
echo
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

# Create backup branch
echo "📦 Creating backup branch..."
git branch backup-before-cleanup

# Interactive rebase to squash commits
echo "🔧 Starting interactive rebase..."
echo
echo "In the editor that opens:"
echo "1. Group related 'fix:' commits together by changing 'pick' to 'squash' (or 's')"
echo "2. Keep feature commits as 'pick'"
echo "3. Save and close the editor"
echo
echo "Press Enter to continue..."
read

# Rebase from the first production-ready commit
git rebase -i 8c47c62

# Update author information
echo "✏️  Updating author information..."
git filter-branch -f --env-filter '
export GIT_AUTHOR_NAME="Bright Liu"
export GIT_AUTHOR_EMAIL="brightliu@example.com"
export GIT_COMMITTER_NAME="Bright Liu"
export GIT_COMMITTER_EMAIL="brightliu@example.com"
' --tag-name-filter cat -- --branches --tags

# Remove Claude references from commit messages
echo "🔍 Removing Claude references from commit messages..."
git filter-branch -f --msg-filter '
sed -e "s/Claude[[:space:]]*//g" \
    -e "s/claude[[:space:]]*//g" \
    -e "s/Co-Authored-By:.*Claude.*//g" \
    -e "s/Generated with.*Claude.*//g" \
    -e "/^[[:space:]]*$/d"
' -- --all

echo
echo "✅ Git history cleanup complete!"
echo
echo "Next steps:"
echo "1. Review the cleaned history: git log --oneline"
echo "2. If satisfied, force push: git push --force origin main"
echo "3. Delete backup branch: git branch -D backup-before-cleanup"
echo
echo "⚠️  Note: Force pushing will overwrite remote history."
echo "   Make sure no one else is working on the repository."