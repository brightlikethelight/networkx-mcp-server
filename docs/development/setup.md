# Development Setup

Get your development environment ready for contributing to NetworkX MCP Server in minutes!

## Quick Start

### Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/brightliu/networkx-mcp-server.git
cd networkx-mcp-server

# Create venv and install
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e ".[dev,all]"

# Set up pre-commit hooks
pre-commit install
```

### Manual Setup

If you prefer manual setup or need to customize the process:

=== "1. Prerequisites"

    **Required:**
    - Python 3.11+ (3.12 recommended)
    - Git 2.20+
    - pip or uv

    **Optional but recommended:**
    - Docker & Docker Compose (for Redis)
    - VSCode or PyCharm
    - Redis server (for caching)

=== "2. Virtual Environment"

    ```bash
    # Create virtual environment
    python -m venv venv

    # Activate (Linux/Mac)
    source venv/bin/activate

    # Activate (Windows)
    venv\Scripts\activate
    ```

=== "3. Dependencies"

    ```bash
    # Install development dependencies
    pip install -e ".[dev,all]"

    # Verify installation
    python -c "import networkx_mcp; print('✓ Success')"
    ```

=== "4. Development Tools"

    ```bash
    # Install pre-commit hooks
    pre-commit install
    pre-commit install --hook-type pre-push

    # Run hooks on all files
    pre-commit run --all-files
    ```

## Development Environment

### Directory Structure

```
networkx-mcp-server/
├── src/networkx_mcp/          # Main package
├── tests/                     # Test suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   ├── property/             # Property-based tests
│   ├── security/             # Security tests
│   └── performance/          # Performance tests
├── benchmarks/               # ASV benchmarks
├── docs/                     # Documentation
├── scripts/                  # Development scripts
├── .github/                  # GitHub workflows
└── configs/                  # Configuration examples
```

### Development Commands

| Task | Command |
|------|---------|
| Install dev deps | `pip install -e ".[dev]"` |
| Run tests | `pytest tests/working/` |
| Run tests with coverage | `pytest tests/working/ --cov` |
| Start server | `python -m networkx_mcp` |
| Lint + type check | `ruff check . && mypy src/` |

### Environment Configuration

Create `.env.development` for local settings:

```bash
# Development server
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=8765
MCP_LOG_LEVEL=DEBUG
MCP_LOG_FORMAT=text

# Redis (optional)
REDIS_URL=redis://localhost:6379/0
REDIS_PREFIX=networkx_dev

# Development settings
MAX_GRAPH_SIZE=100000
ENABLE_CACHING=true
ENABLE_AUTH=false
RATE_LIMIT_ENABLED=false
```

## IDE Configuration

### VSCode Setup

The automated setup creates VSCode configuration:

**.vscode/settings.json**

```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.ruffEnabled": true,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "editor.formatOnSave": true
}
```

**Recommended Extensions:**

- Python
- Black Formatter
- Ruff
- MyPy Type Checker
- Pytest
- YAML Support
- Markdown All in One

### PyCharm Setup

1. **File → Settings → Project → Python Interpreter**
2. **Add Interpreter → Existing Environment**
3. **Select:** `./venv/bin/python`
4. **Configure pytest as test runner**
5. **Set Black as code formatter**

### Debugging Configuration

**VSCode Launch Configurations:**

```json
{
  "name": "Run MCP Server",
  "type": "python",
  "request": "launch",
  "module": "networkx_mcp.server",
  "env": {"MCP_LOG_LEVEL": "DEBUG"}
},
{
  "name": "Debug Test",
  "type": "python",
  "request": "launch",
  "module": "pytest",
  "args": ["${file}", "-v", "-s"]
}
```

## Development Services

### Redis (Optional)

Start Redis for caching during development:

=== "Docker Compose"

    ```bash
    # Start Redis with UI
    docker-compose -f docker-compose.dev.yml up -d

    # Access Redis Commander at http://localhost:8081
    ```

=== "Local Installation"

    ```bash
    # macOS
    brew install redis
    brew services start redis

    # Ubuntu
    sudo apt-get install redis-server
    sudo systemctl start redis

    # Verify
    redis-cli ping  # Should return PONG
    ```

### Development URLs

| Service | URL | Purpose |
|---------|-----|---------|
| MCP Server | <http://localhost:8765> | Main server |
| Redis UI | <http://localhost:8081> | Redis management |
| Documentation | <http://localhost:8000> | Docs (when running `mkdocs serve`) |

## Development Workflow

### Daily Development

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Pull latest changes
git pull origin main

# 3. Install/update dependencies
pip install -e ".[dev,all]"

# 4. Start development server
python -m networkx_mcp

# 5. Run tests (in another terminal)
pytest tests/working/ --cov
```

### Code Quality Checks

```bash
# Format code
black src tests
ruff format src tests

# Lint code
ruff check src tests

# Type checking
mypy src/networkx_mcp

# Run all quality checks
nox -s lint typecheck

# Pre-commit (runs automatically on commit)
pre-commit run --all-files
```

### Testing

```bash
# Run all tests with coverage
pytest tests/working/ --cov

# Stop on first failure
pytest tests/working/ -x

# Run a specific test file
pytest tests/working/test_handlers.py -v

# Debug a specific test
pytest tests/working/test_handlers.py::TestSyncHandlers::test_create_graph -v -s
```

### Performance Testing

```bash
# Run benchmarks
python -m asv run

# Compare performance
python -m asv compare HEAD~1 HEAD

# Generate benchmark report
python -m asv publish
python -m asv preview
```

### Documentation

```bash
# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build

# Test documentation links
mkdocs build --strict
```

## Git Workflow

### Branch Management

```bash
# Create feature branch
git checkout -b feature/amazing-feature

# Keep branch updated
git fetch origin
git rebase origin/main

# Push feature branch
git push origin feature/amazing-feature
```

### Commit Guidelines

We use [Conventional Commits](https://conventionalcommits.org/):

```bash
# Feature
git commit -m "feat: add graph visualization tool"

# Bug fix
git commit -m "fix: resolve memory leak in graph manager"

# Documentation
git commit -m "docs: update API reference for centrality tools"

# Test
git commit -m "test: add property-based tests for algorithms"

# Refactor
git commit -m "refactor: simplify graph operations interface"
```

**Commit Types:**

- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `ci`: CI/CD changes
- `chore`: Maintenance tasks

### Pre-commit Hooks

Hooks run automatically on commit:

- **Code formatting** (Black, Ruff)
- **Linting** (Ruff, MyPy)
- **Test execution** (Fast unit tests)
- **Security scanning** (Bandit)
- **Conventional commits** validation

## Troubleshooting

### Common Issues

!!! question "Import errors after setup?"

    **Solution:**
    ```bash
    # Ensure virtual environment is activated
    source venv/bin/activate

    # Reinstall in development mode
    pip install -e ".[dev,all]"
    ```

!!! question "Pre-commit hooks failing?"

    **Solution:**
    ```bash
    # Update pre-commit
    pre-commit autoupdate

    # Clear cache and retry
    pre-commit clean
    pre-commit run --all-files
    ```

!!! question "Tests failing with import errors?"

    **Solution:**
    ```bash
    # Add src to Python path
    export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

    # Or use pytest with src
    python -m pytest tests/ --rootdir=.
    ```

!!! question "Redis connection errors?"

    **Solution:**
    ```bash
    # Start Redis
    docker-compose -f docker-compose.dev.yml up -d redis

    # Or disable Redis
    unset REDIS_URL
    ```

### Debugging Tips

**Print debugging:**

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("Debug message")
```

**Interactive debugging:**

```python
import ipdb; ipdb.set_trace()  # Breakpoint
```

**Memory profiling:**

```bash
mprof run python script.py
mprof plot
```

**CPU profiling:**

```bash
py-spy record -o profile.svg -- python script.py
```

### Performance Monitoring

```bash
# Monitor memory usage
python -c "
import psutil
import networkx as nx
from networkx_mcp.graph_cache import graphs

process = psutil.Process()
print(f'Memory before: {process.memory_info().rss / 1024 / 1024:.1f} MB')

G = nx.erdos_renyi_graph(10000, 0.1)
graphs['test'] = G

print(f'Memory after: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

## Advanced Development

### Custom Development Scripts

Create your own development scripts in `scripts/`:

```python
#!/usr/bin/env python3
"""Custom development script."""
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Your development code here
from networkx_mcp.graph_cache import graphs

def main():
    # Development tasks...

if __name__ == "__main__":
    main()
```

### Environment Variables Reference

| Variable | Development Default | Description |
|----------|-------------------|-------------|
| `MCP_LOG_LEVEL` | `DEBUG` | Logging verbosity |
| `MCP_LOG_FORMAT` | `text` | Log format |
| `PYTHONPATH` | `./src` | Python module path |
| `PYTEST_CURRENT_TEST` | - | Current test (set by pytest) |
| `COVERAGE_PROCESS_START` | `.coveragerc` | Coverage config |

### Continuous Integration

Test your changes against CI locally:

```bash
# Run full test suite
pytest tests/working/ -x -q --tb=short

# Lint + format check
ruff check . && ruff format --check .

# Type checking
mypy src/networkx_mcp/
```

## Getting Help

### Resources

- **📖 Documentation**: [Full docs](../index.md)
- **💬 Discussions**: [GitHub Discussions](https://github.com/brightliu/networkx-mcp-server/discussions)
- **🐛 Issues**: [Report bugs](https://github.com/brightliu/networkx-mcp-server/issues)
- **📧 Email**: [dev-support@networkx-mcp-server.com](mailto:dev-support@networkx-mcp-server.com)

### Development Community

- **Code reviews**: All PRs get thorough reviews
- **Weekly sync**: Join our development calls
- **Mentoring**: New contributors get guidance
- **Documentation**: Help improve our docs

## Next Steps

Ready to contribute? Check out:

<div class="grid cards" markdown>

- [:material-bug: **Good First Issues**](https://github.com/brightliu/networkx-mcp-server/labels/good%20first%20issue)

    Perfect for new contributors

- [:material-code-braces: **Architecture Guide**](architecture.md)

    Understand the codebase structure

- [:material-test-tube: **Testing Guide**](testing.md)

    Learn our testing practices

- [:material-source-pull: **Contributing Guide**](contributing.md)

    Full contribution guidelines

</div>

!!! success "You're Ready!"

    Your development environment is now configured! Start by exploring the codebase and running some tests. Happy coding! 🚀
