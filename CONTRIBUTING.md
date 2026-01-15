# Contributing to NetworkX MCP Server

First off, thank you for considering contributing to NetworkX MCP Server! It's people like you that make this project such a great tool. We welcome contributions from everyone, regardless of their experience level.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Process](#development-process)
- [Style Guidelines](#style-guidelines)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [brightliu@college.harvard.edu](mailto:brightliu@college.harvard.edu).

### Our Standards

- **Be respectful and inclusive**: Value each other's ideas, styles, and viewpoints
- **Be constructive**: Provide helpful feedback and accept criticism gracefully
- **Be collaborative**: Work together towards common goals
- **Be patient**: Remember that everyone was new once

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Basic knowledge of graph theory concepts

### Setting Up Your Development Environment

1. **Fork the repository**

   ```bash
   # Click "Fork" button on GitHub
   ```

2. **Clone your fork**

   ```bash
   git clone https://github.com/YOUR_USERNAME/networkx-mcp-server.git
   cd networkx-mcp-server
   ```

3. **Add upstream remote**

   ```bash
   git remote add upstream https://github.com/Bright-L01/networkx-mcp-server.git
   ```

4. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install development dependencies**

   ```bash
   pip install -e ".[dev]"
   ```

6. **Install pre-commit hooks**

   ```bash
   pre-commit install
   ```

7. **Run tests to verify setup**

   ```bash
   pytest
   ```

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce**
- **Expected behavior**
- **Actual behavior**
- **System information** (OS, Python version, etc.)
- **Relevant logs or error messages**

**Example:**

```markdown
### Bug: Shortest path fails with weighted edges

**Steps to reproduce:**
1. Create a graph with weighted edges
2. Call shortest_path with weight parameter
3. Observe error

**Expected:** Returns weighted shortest path
**Actual:** KeyError: 'weight'

**System:** macOS 13.5, Python 3.11.5
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Use case**: Why is this enhancement needed?
- **Proposed solution**: How should it work?
- **Alternatives considered**: What other solutions did you consider?
- **Additional context**: Mockups, examples, etc.

### Submitting Pull Requests

1. **Check existing PRs and issues** first
2. **Discuss major changes** in an issue before starting
3. **Keep PRs focused** - one feature/fix per PR
4. **Include tests** for new functionality
5. **Update documentation** as needed
6. **Follow the style guide**

## Development Process

### 1. Branch Naming

Use descriptive branch names:

- `feature/add-graph-embedding` - New features
- `fix/memory-leak-in-flow` - Bug fixes
- `docs/update-api-reference` - Documentation
- `refactor/simplify-algorithms` - Code refactoring
- `test/add-clustering-tests` - Test additions

### 2. Development Workflow

```bash
# 1. Sync with upstream
git checkout main
git pull upstream main

# 2. Create feature branch
git checkout -b feature/your-feature-name

# 3. Make changes
# ... edit files ...

# 4. Run tests frequently
pytest tests/test_relevant_module.py

# 5. Check code quality
black src/ tests/
ruff check src/ tests/
mypy src/

# 6. Commit changes
git add -p  # Stage changes interactively
git commit -m "feat: add amazing feature"

# 7. Push to your fork
git push origin feature/your-feature-name
```

### 3. Code Organization

Follow the existing project structure:

```
src/networkx_mcp/
├── core/              # Core functionality
├── advanced/          # Advanced analytics
│   ├── community/     # Community detection
│   └── ml/           # Machine learning
├── visualization/     # Visualization backends
├── io/               # I/O operations
├── interfaces/       # Public APIs
└── utils/            # Utilities
```

When adding new features:

- Put algorithms in appropriate category
- Create new subdirectories for major features
- Keep files focused and under 500 lines
- Use clear, descriptive names

## Style Guidelines

### Python Style

We use [Black](https://github.com/psf/black) for formatting and [Ruff](https://github.com/astral-sh/ruff) for linting.

#### Code Style Rules

```python
# ✅ Good: Clear, typed, documented
from typing import Dict, List, Optional
import networkx as nx


def calculate_modularity(
    graph: nx.Graph,
    communities: List[List[str]],
    weight: Optional[str] = None
) -> float:
    """Calculate modularity of community partition.

    Args:
        graph: Input graph
        communities: List of node communities
        weight: Edge weight attribute name

    Returns:
        Modularity score between -1 and 1

    Raises:
        ValueError: If communities overlap
    """
    # Implementation here
    pass


# ❌ Bad: Unclear, untyped, undocumented
def calc_mod(g, comms, w=None):
    # calculate modularity
    pass
```

#### Best Practices

1. **Type hints**: Always use type hints
2. **Docstrings**: Use Google-style docstrings
3. **Variable names**: Be descriptive (`node_count` not `n`)
4. **Functions**: Keep them small and focused
5. **Error handling**: Raise specific exceptions with clear messages
6. **Constants**: Use UPPER_CASE for module-level constants

### Testing Style

```python
# ✅ Good: Descriptive, isolated, comprehensive
import pytest
import networkx as nx
from networkx_mcp.algorithms import find_shortest_path


class TestShortestPath:
    """Test shortest path algorithms."""

    def test_simple_path(self):
        """Test shortest path in simple graph."""
        # Arrange
        graph = nx.Graph()
        graph.add_edges_from([("A", "B"), ("B", "C")])

        # Act
        path = find_shortest_path(graph, "A", "C")

        # Assert
        assert path == ["A", "B", "C"]

    def test_no_path_exists(self):
        """Test when no path exists between nodes."""
        graph = nx.Graph()
        graph.add_nodes_from(["A", "B"])

        with pytest.raises(nx.NetworkXNoPath):
            find_shortest_path(graph, "A", "B")

    @pytest.mark.parametrize("weight,expected", [
        ("weight", ["A", "C"]),
        (None, ["A", "B", "C"]),
    ])
    def test_weighted_paths(self, weight, expected):
        """Test paths with different weight configurations."""
        # Test implementation
```

## Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification.

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation only
- **style**: Code style changes (formatting, missing semicolons, etc)
- **refactor**: Code change that neither fixes a bug nor adds a feature
- **perf**: Performance improvement
- **test**: Adding missing tests
- **build**: Changes to build system or dependencies
- **ci**: Changes to CI configuration
- **chore**: Other changes that don't modify src or test files

### Examples

```bash
# ✅ Good commits
git commit -m "feat(algorithms): add A* pathfinding algorithm

- Implement A* with customizable heuristic
- Add tests for grid and network graphs
- Update documentation with examples

Closes #123"

git commit -m "fix(storage): handle empty graphs in save operations

Empty NetworkX graphs are falsy, causing save failures.
Changed condition from 'if graph:' to 'if graph is not None:'

Fixes #456"

git commit -m "docs: add examples for community detection

- Add Louvain algorithm example
- Include visualization of results
- Link to theoretical background"

# ❌ Bad commits
git commit -m "fixed stuff"
git commit -m "WIP"
git commit -m "add feature"
```

## Pull Request Process

### Before Submitting

1. **Update from upstream main**

   ```bash
   git checkout main
   git pull upstream main
   git checkout your-branch
   git rebase main
   ```

2. **Run all checks**

   ```bash
   # Format code
   black src/ tests/

   # Lint
   ruff check src/ tests/

   # Type check
   mypy src/

   # Run all tests
   pytest

   # Check coverage
   pytest --cov=src/networkx_mcp --cov-report=term-missing
   ```

3. **Update documentation**
   - Add/update docstrings
   - Update README if needed
   - Add to CHANGELOG.md (unreleased section)

### PR Description Template

```markdown
## Description
Brief description of what this PR does.

## Motivation and Context
Why is this change required? What problem does it solve?
If it fixes an open issue, link it here.

## How Has This Been Tested?
Describe the tests you ran. Include OS and Python version.

## Types of changes
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)

## Checklist:
- [ ] My code follows the project style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have added necessary documentation (if appropriate)
- [ ] Any dependent changes have been merged and published
```

### Review Process

1. **Automated checks** must pass (CI, tests, linting)
2. **Code review** by at least one maintainer
3. **Address feedback** promptly and politely
4. **Squash commits** if requested
5. **Celebrate** when merged!

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_algorithms.py

# Run specific test
pytest tests/test_algorithms.py::test_shortest_path

# Run with coverage
pytest --cov=src/networkx_mcp --cov-report=html

# Run only fast tests
pytest -m "not slow"

# Run with verbose output
pytest -v
```

### Writing Tests

1. **Test file naming**: `test_<module_name>.py`
2. **Test class naming**: `Test<FeatureName>`
3. **Test method naming**: `test_<specific_scenario>`
4. **Use fixtures** for common setup
5. **Test edge cases** and error conditions
6. **Mock external dependencies** when needed

### Test Categories

Mark tests appropriately:

```python
@pytest.mark.slow  # Tests taking >1 second
@pytest.mark.integration  # Tests requiring Redis
@pytest.mark.benchmark  # Performance tests
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def complex_function(
    graph: nx.Graph,
    source: str,
    target: str,
    weight: str = "weight",
    cutoff: Optional[int] = None
) -> List[str]:
    """Find shortest path between source and target.

    This function implements Dijkstra's algorithm with optimizations
    for sparse graphs. It handles both weighted and unweighted graphs.

    Args:
        graph: NetworkX graph instance
        source: Starting node for path
        target: Ending node for path
        weight: Name of edge attribute to use as weight.
            If None, all edges have weight 1.
        cutoff: Depth to stop the search. Only paths of
            length <= cutoff are returned.

    Returns:
        List of nodes representing the shortest path.
        Empty list if no path exists.

    Raises:
        NodeNotFound: If source or target not in graph
        ValueError: If weight attribute doesn't exist

    Examples:
        >>> G = nx.Graph()
        >>> G.add_edges_from([('A', 'B'), ('B', 'C')])
        >>> complex_function(G, 'A', 'C')
        ['A', 'B', 'C']

    Note:
        For graphs with negative weights, use Bellman-Ford
        algorithm instead.
    """
```

### Updating Documentation

1. **API changes**: Update docstrings immediately
2. **New features**: Add to relevant docs/guides
3. **Examples**: Provide working code examples
4. **Changelog**: Add entry to unreleased section

## Community

### Getting Help

- [GitHub Discussions](https://github.com/Bright-L01/networkx-mcp-server/discussions) - Ask questions
- [Issue Tracker](https://github.com/Bright-L01/networkx-mcp-server/issues) - Report bugs
- Email: brightliu@college.harvard.edu

### Communication Channels

- **Discussions**: General questions and ideas
- **Issues**: Bug reports and feature requests
- **Pull Requests**: Code contributions
- **Security**: See [SECURITY.md](SECURITY.md) for vulnerability reporting

### Recognition

Contributors are recognized in release notes and project documentation.

## Thank You

Your contributions make this project better for everyone. Whether it's fixing a typo, adding a test, or implementing a new feature, every contribution is valued.
