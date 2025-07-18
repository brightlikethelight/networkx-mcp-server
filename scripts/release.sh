#!/bin/bash
# NetworkX MCP Server Release Script

set -e

echo "🚀 NetworkX MCP Server Release Process"
echo "====================================="

# Check if we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "❌ Error: Releases must be created from main branch"
    echo "Current branch: $CURRENT_BRANCH"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "❌ Error: Uncommitted changes detected"
    echo "Please commit or stash your changes before releasing"
    exit 1
fi

# Get current version
CURRENT_VERSION=$(python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])")
echo "Current version: $CURRENT_VERSION"

# Prompt for new version
echo -n "Enter new version (or press enter to auto-increment): "
read NEW_VERSION

if [ -z "$NEW_VERSION" ]; then
    # Auto-increment patch version
    IFS='.' read -ra VERSION_PARTS <<< "$CURRENT_VERSION"
    MAJOR=${VERSION_PARTS[0]}
    MINOR=${VERSION_PARTS[1]}
    PATCH=${VERSION_PARTS[2]}
    NEW_PATCH=$((PATCH + 1))
    NEW_VERSION="$MAJOR.$MINOR.$NEW_PATCH"
fi

echo "New version will be: $NEW_VERSION"
echo -n "Continue? (y/n): "
read CONFIRM

if [ "$CONFIRM" != "y" ]; then
    echo "Release cancelled"
    exit 0
fi

# Update version in pyproject.toml
echo "📝 Updating version in pyproject.toml..."
sed -i.bak "s/version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" pyproject.toml
rm pyproject.toml.bak

# Update version in __version__.py
echo "📝 Updating version in __version__.py..."
echo "__version__ = \"$NEW_VERSION\"" > src/networkx_mcp/__version__.py

# Run tests
echo "🧪 Running tests..."
pytest tests/ -v

# Run quality checks
echo "🔍 Running quality checks..."
ruff check .
black --check .
mypy src/networkx_mcp/

# Commit version bump
echo "💾 Committing version bump..."
git add pyproject.toml src/networkx_mcp/__version__.py
git commit -m "chore: bump version to $NEW_VERSION"

# Create and push tag
echo "🏷️ Creating tag v$NEW_VERSION..."
git tag -a "v$NEW_VERSION" -m "Release version $NEW_VERSION"

# Push changes
echo "📤 Pushing changes and tag..."
git push origin main
git push origin "v$NEW_VERSION"

echo ""
echo "✅ Release v$NEW_VERSION created successfully!"
echo ""
echo "GitHub Actions will now:"
echo "1. Build the package"
echo "2. Create a GitHub release"
echo "3. Publish to PyPI (if configured)"
echo ""
echo "Monitor the release at: https://github.com/your-org/networkx-mcp-server/actions"
