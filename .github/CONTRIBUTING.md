# Contributing to NetworkX MCP Server

Thank you for your interest in contributing to the NetworkX MCP Server! This document provides guidelines and information for contributors.

## 🎯 How to Contribute

### Reporting Bugs

1. **Check existing issues** to avoid duplicates
2. **Use the bug report template** when creating new issues
3. **Provide detailed information**:
   - Python version and OS
   - NetworkX MCP Server version
   - Minimal reproduction steps
   - Expected vs actual behavior
   - Error messages and stack traces

### Suggesting Features

1. **Check existing issues and discussions** for similar ideas
2. **Use the feature request template**
3. **Explain the use case** and why the feature would be valuable
4. **Consider the scope** - does it fit the project's goals?

### Contributing Code

#### Development Setup

1. **Fork the repository** and clone your fork
2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**:

   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**:

   ```bash
   pre-commit install
   ```

5. **Verify setup**:

   ```bash
   pytest tests/unit/test_server_minimal.py -v
   ```

#### Code Standards

We maintain high code quality standards:

**Formatting and Linting**:

- Code formatted with `black` and `ruff`
- Imports sorted with `isort`
- Type hints required for all functions
- Docstrings required (Google style)

**Testing**:

- All new code must include tests
- Maintain 80%+ test coverage
- Tests should be fast (unit tests < 1s each)
- Follow existing test patterns

**Architecture**:

- Follow modular architecture principles
- Single responsibility per module
- Clear interfaces between components
- Backward compatibility maintained

#### Pull Request Process

1. **Create a feature branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write code following our standards
   - Add comprehensive tests
   - Update documentation if needed
   - Ensure all tests pass

3. **Commit with semantic messages**:

   ```bash
   git commit -m "feat: add new graph algorithm for community detection"
   ```

4. **Push and create pull request**:
   - Use the PR template
   - Provide clear description
   - Reference related issues
   - Request review

5. **Address feedback**:
   - Respond to code review comments
   - Make requested changes
   - Update tests and documentation

#### Commit Message Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Scopes**:

- `core`: Core functionality
- `algorithms`: Graph algorithms
- `handlers`: MCP handlers
- `monitoring`: Monitoring and observability
- `security`: Security features
- `docs`: Documentation
- `ci`: CI/CD pipeline

**Examples**:

```bash
feat(algorithms): add betweenness centrality calculation
fix(core): resolve memory leak in graph storage
docs(api): update algorithm documentation
test(integration): add end-to-end workflow tests
```

## 📝 Development Guidelines

### Code Organization

Follow our modular architecture:

```
src/networkx_mcp/
├── core/               # Core functionality
├── handlers/           # MCP function handlers
├── advanced/           # Advanced algorithms
│   ├── directed/       # Directed graph algorithms
│   ├── ml/            # Machine learning integration
│   └── ...
├── monitoring/         # Observability features
└── security/          # Security features
```

### Adding New Features

#### 1. New Algorithm

```python
# Choose appropriate module (e.g., advanced/community/)
class CommunityDetection:
    """Community detection algorithms."""

    @staticmethod
    def new_algorithm(graph: nx.Graph, **params) -> Dict[str, Any]:
        """
        New algorithm description.

        Parameters:
        -----------
        graph : nx.Graph
            Input graph
        **params : dict
            Algorithm parameters

        Returns:
        --------
        Dict[str, Any]
            Results dictionary
        """
        # Implementation
        return {"status": "success", "result": result}
```

#### 2. New Test

```python
# tests/unit/test_community.py
class TestCommunityDetection:
    """Test community detection algorithms."""

    def test_new_algorithm(self):
        """Test new algorithm."""
        # Create test graph
        G = nx.karate_club_graph()

        # Run algorithm
        result = CommunityDetection.new_algorithm(G)

        # Assertions
        assert result["status"] == "success"
        assert "result" in result
```

### Documentation

Update documentation when:

- Adding new features
- Changing existing APIs
- Improving architecture
- Adding examples

Documentation files:

- `ARCHITECTURE.md` - System architecture
- `docs/MODULE_STRUCTURE.md` - Module organization
- `docs/DEVELOPMENT_GUIDE.md` - Development guide
- `docs/api/` - API documentation

### Performance Considerations

- **Algorithm Complexity**: Document time/space complexity
- **Memory Usage**: Consider memory for large graphs
- **Scalability**: Test with graphs of various sizes
- **Benchmarking**: Add benchmarks for significant changes

### Security Considerations

- **Input Validation**: Validate all user inputs
- **Sanitization**: Sanitize data before processing
- **Error Handling**: Don't expose internal details
- **Dependencies**: Keep dependencies updated

## 🧪 Testing

### Test Categories

1. **Unit Tests** (`tests/unit/`):
   - Fast, isolated tests
   - Test individual functions/classes
   - Mock external dependencies

2. **Integration Tests** (`tests/integration/`):
   - Test component interactions
   - End-to-end workflows
   - MCP protocol compliance

3. **Performance Tests** (`tests/performance/`):
   - Benchmark algorithms
   - Memory usage tests
   - Scalability validation

4. **Security Tests** (`tests/security/`):
   - Input validation
   - Security boundary tests
   - Vulnerability checks

### Running Tests

```bash
# All tests
pytest

# Specific category
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# With coverage
pytest --cov=src/networkx_mcp --cov-report=html

# Specific test
pytest tests/unit/test_server_minimal.py::TestServerBasics::test_import
```

### Test Requirements

- **Coverage**: Maintain 80%+ coverage
- **Speed**: Unit tests should run quickly (< 1s each)
- **Isolation**: Tests should not depend on each other
- **Clarity**: Test names should clearly describe what's being tested
- **Data**: Use fixtures for test data

## 🚀 Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):

- `MAJOR.MINOR.PATCH`
- `MAJOR`: Breaking changes
- `MINOR`: New features (backward compatible)
- `PATCH`: Bug fixes

### Release Checklist

1. **Update version** in `src/networkx_mcp/__version__.py`
2. **Update changelog** with new features and fixes
3. **Run full test suite** with coverage
4. **Update documentation** if needed
5. **Create release PR** with all changes
6. **Tag release** after merge
7. **Create GitHub release** with release notes

## 💬 Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Pull Requests**: Code contributions and reviews

### Code of Conduct

We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/0/code_of_conduct/). Please be respectful and inclusive in all interactions.

### Getting Help

- **Documentation**: Check our comprehensive docs first
- **Discussions**: Ask questions in GitHub Discussions
- **Issues**: Report bugs with detailed information
- **Examples**: Look at usage examples in the `examples/` directory

## 🏆 Recognition

Contributors are recognized in:

- Release notes for their contributions
- The project's contributors page
- Special recognition for significant contributions

Thank you for contributing to NetworkX MCP Server! 🎉
