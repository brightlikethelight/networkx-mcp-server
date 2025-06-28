# 🔄 Git Workflow Guide

This guide explains how to work with the NetworkX MCP Server repository using Git and GitHub. Whether you're contributing code, reporting bugs, or just exploring the project, this guide will help you navigate the development workflow.

## 📋 Table of Contents

- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Branching Strategy](#branching-strategy)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)
- [Common Git Commands](#common-git-commands)
- [Troubleshooting](#troubleshooting)

## 🏗️ Repository Structure

```
networkx-mcp-server/
├── .github/                    # GitHub configuration and templates
│   ├── workflows/             # CI/CD pipelines
│   ├── ISSUE_TEMPLATE/        # Issue templates
│   ├── SECURITY.md           # Security policy
│   └── FUNDING.yml           # Sponsorship information
├── docs/                      # Documentation
│   ├── api/                  # Auto-generated API docs
│   ├── GIT_WORKFLOW.md       # This file
│   └── ...                   # Other guides
├── src/networkx_mcp/          # Main source code
├── tests/                     # Test suite
├── scripts/                   # Utility scripts
├── examples/                  # Usage examples
├── CHANGELOG.md              # Version history
├── CONTRIBUTING.md           # Contribution guidelines
├── README.md                 # Project overview
└── pyproject.toml           # Project configuration
```

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub first, then:
git clone https://github.com/YOUR_USERNAME/networkx-mcp-server.git
cd networkx-mcp-server

# Add upstream remote to keep your fork updated
git remote add upstream https://github.com/brightliu/networkx-mcp-server.git
git remote -v  # Verify remotes
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify setup
pytest
```

### 3. Configure Git (First Time)

```bash
# Set up your identity
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Optional: Set up GPG signing for commits
git config --global user.signingkey YOUR_GPG_KEY_ID
git config --global commit.gpgsign true
```

## 🌿 Branching Strategy

We use **GitFlow-inspired** branching with semantic naming:

### Main Branches

- **`main`**: Production-ready code, always stable
- **`develop`**: Integration branch for features (if needed for major releases)

### Feature Branches

Create descriptive branch names:

```bash
# Feature development
git checkout -b feature/add-graph-clustering-algorithm
git checkout -b feature/improve-visualization-performance

# Bug fixes
git checkout -b fix/memory-leak-in-pathfinding
git checkout -b fix/redis-connection-timeout

# Documentation
git checkout -b docs/update-api-reference
git checkout -b docs/add-tutorial-examples

# Refactoring
git checkout -b refactor/simplify-graph-operations
git checkout -b refactor/extract-validation-utils

# Testing
git checkout -b test/add-integration-tests
git checkout -b test/improve-coverage-algorithms
```

### Branch Lifecycle

```bash
# 1. Start from main
git checkout main
git pull upstream main

# 2. Create feature branch
git checkout -b feature/your-feature-name

# 3. Work on your feature
# ... make changes ...
git add .
git commit -m "feat: add amazing feature"

# 4. Keep branch updated
git fetch upstream
git rebase upstream/main

# 5. Push to your fork
git push origin feature/your-feature-name

# 6. Create Pull Request on GitHub
# 7. After approval and merge, clean up
git checkout main
git pull upstream main
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name
```

## 📝 Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/) for clear, semantic commit messages.

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature for users
- **fix**: Bug fix for users
- **docs**: Documentation changes
- **style**: Code style changes (formatting, no logic changes)
- **refactor**: Code refactoring (no feature changes)
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **build**: Build system or dependency changes
- **ci**: CI/CD configuration changes
- **chore**: Other changes (version bumps, etc.)

### Scopes (Optional)

- **core**: Core graph operations
- **algorithms**: Graph algorithms
- **visualization**: Visualization components
- **security**: Security-related changes
- **docs**: Documentation
- **tests**: Test-related changes

### Examples

```bash
# ✅ Good commits
git commit -m "feat(algorithms): add A* pathfinding algorithm

Implements A* algorithm with customizable heuristic functions.
Supports weighted and unweighted graphs with early termination.

Closes #123"

git commit -m "fix(core): handle empty graphs in persistence layer

Empty NetworkX graphs are falsy, causing save operations to fail.
Changed condition from 'if graph:' to 'if graph is not None:'.

Fixes #456"

git commit -m "docs: add examples for community detection

- Add Louvain algorithm usage example
- Include visualization of detected communities
- Link to theoretical background and parameters"

git commit -m "perf(algorithms): optimize shortest path for large graphs

- Use bidirectional search for better performance
- Add early termination for impossible paths
- 40% speed improvement on graphs with >10k nodes"

# ❌ Bad commits
git commit -m "fixed stuff"
git commit -m "WIP"
git commit -m "update"
git commit -m "more changes"
```

### Commit Best Practices

1. **Keep commits atomic**: One logical change per commit
2. **Write descriptive subjects**: 50 characters or less
3. **Use imperative mood**: "Add feature" not "Added feature"
4. **Include context in body**: Explain why, not what
5. **Reference issues**: Use "Closes #123" or "Fixes #456"

## 🔄 Pull Request Process

### Before Creating a PR

```bash
# 1. Ensure your branch is up to date
git checkout main
git pull upstream main
git checkout your-branch
git rebase main

# 2. Run all quality checks
black src/ tests/               # Format code
ruff check src/ tests/          # Lint
mypy src/                       # Type check
pytest                          # Run tests
pytest --cov=src/networkx_mcp   # Check coverage

# 3. Update documentation if needed
python scripts/generate_api_docs.py  # Regenerate API docs

# 4. Push your changes
git push origin your-branch
```

### Creating the PR

1. **Go to GitHub** and create a pull request
2. **Use the PR template** (auto-filled when you create PR)
3. **Fill out all sections** completely
4. **Link related issues** using "Closes #123"
5. **Request reviewers** if you know who should review

### PR Requirements

- [ ] All CI checks pass
- [ ] Code coverage doesn't decrease
- [ ] Documentation is updated
- [ ] Tests are included for new features
- [ ] CHANGELOG.md is updated (for user-facing changes)

### Review Process

1. **Automated checks** run first (CI, tests, linting)
2. **Code review** by maintainers
3. **Address feedback** promptly and professionally
4. **Update PR** based on review comments
5. **Approval and merge** by maintainers

## 🏷️ Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes (2.0.0)
- **MINOR**: New features, backwards compatible (1.1.0)
- **PATCH**: Bug fixes, backwards compatible (1.0.1)

### Release Steps (Maintainers)

```bash
# 1. Prepare release
git checkout main
git pull origin main

# 2. Update version and changelog
# Edit pyproject.toml version
# Update CHANGELOG.md with new version

# 3. Create release commit
git add pyproject.toml CHANGELOG.md
git commit -m "chore: release v1.1.0"

# 4. Create and push tag
git tag -a v1.1.0 -m "Release v1.1.0

- Feature 1: Description
- Feature 2: Description
- Bug fix: Description

See CHANGELOG.md for complete details."

git push origin main --tags

# 5. GitHub Actions will:
# - Run all tests
# - Build package
# - Publish to PyPI
# - Create GitHub release
# - Update documentation
```

## 🛠️ Common Git Commands

### Daily Workflow

```bash
# Check status
git status
git log --oneline -10

# Update your fork
git fetch upstream
git checkout main
git merge upstream/main
git push origin main

# Create feature branch
git checkout -b feature/new-feature

# Stage and commit changes
git add .                           # Stage all changes
git add -p                          # Stage interactively
git commit -m "feat: add feature"   # Commit with message
git commit --amend                  # Amend last commit

# Push changes
git push origin feature/new-feature
```

### Advanced Operations

```bash
# Interactive rebase (clean up commits)
git rebase -i HEAD~3

# Cherry-pick specific commit
git cherry-pick abc1234

# Squash commits
git reset --soft HEAD~3
git commit -m "feat: combined feature"

# Temporary stash
git stash
git stash pop

# Undo changes
git checkout -- file.py            # Discard file changes
git reset HEAD~1                    # Undo last commit (keep changes)
git reset --hard HEAD~1             # Undo last commit (lose changes)
```

### Working with Remotes

```bash
# View remotes
git remote -v

# Fetch all remotes
git fetch --all

# Sync fork with upstream
git checkout main
git fetch upstream
git reset --hard upstream/main
git push origin main --force
```

## 🐛 Troubleshooting

### Common Issues

**1. Merge Conflicts**
```bash
# During rebase or merge
git status                    # See conflicted files
# Edit files to resolve conflicts
git add .                     # Mark as resolved
git rebase --continue         # Continue rebase
```

**2. Diverged Branches**
```bash
# Your branch has diverged from upstream
git fetch upstream
git rebase upstream/main      # Rebase onto latest
# Or if you prefer merge:
git merge upstream/main
```

**3. Accidentally Committed to Main**
```bash
# Move commits to new branch
git branch feature/new-branch
git reset --hard HEAD~3       # Remove 3 commits from main
git checkout feature/new-branch
```

**4. Wrong Commit Message**
```bash
# Last commit only
git commit --amend -m "New message"

# Older commits
git rebase -i HEAD~3          # Interactive rebase
# Change "pick" to "reword" for commits to edit
```

**5. Large Files Accidentally Committed**
```bash
# Remove from history (careful!)
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch large-file.dat' \
  --prune-empty --tag-name-filter cat -- --all
```

### Getting Help

- **Git documentation**: `git help <command>`
- **GitHub docs**: https://docs.github.com
- **Ask in discussions**: [GitHub Discussions](https://github.com/brightliu/networkx-mcp-server/discussions)
- **Community**: Check our [CONTRIBUTING.md](../CONTRIBUTING.md)

## 📚 Additional Resources

### Git Learning
- [Pro Git Book](https://git-scm.com/book) (Free)
- [GitHub Skills](https://skills.github.com/)
- [Conventional Commits](https://www.conventionalcommits.org/)

### Tools
- **GUI Clients**: GitKraken, SourceTree, GitHub Desktop
- **VS Code**: GitLens extension
- **CLI Tools**: `tig`, `lazygit`, `gh` (GitHub CLI)

### Project-Specific
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [README.md](../README.md) - Project overview
- [docs/api/](../docs/api/) - API documentation

---

## 🤝 Need Help?

- 💬 [GitHub Discussions](https://github.com/brightliu/networkx-mcp-server/discussions)
- 🐛 [Report Issues](https://github.com/brightliu/networkx-mcp-server/issues)
- 📧 Email: support@networkx-mcp.org

**Happy contributing!** 🚀