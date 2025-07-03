# Development Guide

## Getting Started with the Modular Architecture

This guide helps developers understand and work with the NetworkX MCP Server's clean, modular architecture.

## Quick Start

### Setting Up Development Environment

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd networkx-mcp-server
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   ```

2. **Install Development Dependencies**
   ```bash
   pip install pytest pytest-cov pytest-asyncio
   pip install ruff black isort mypy
   pip install autoflake vulture
   ```

3. **Verify Installation**
   ```bash
   python -m pytest tests/unit/test_server_minimal.py -v
   ```

### Understanding the Architecture

The codebase follows a clean, layered architecture:

```
Application Layer    → server.py (MCP endpoints)
Handler Layer        → handlers/ (function organization)
Core Layer           → core/ (fundamental operations)
Advanced Layer       → advanced/ (specialized algorithms)
Supporting Layer     → monitoring/, security/, etc.
```

## Working with Modules

### Adding New Functionality

#### 1. Adding a New Graph Algorithm

**Step 1: Choose the Right Location**
- Core algorithms → `core/algorithms.py`
- Specialized algorithms → `advanced/specialized/`
- Directed graph algorithms → `advanced/directed/`
- ML algorithms → `advanced/ml/`

**Step 2: Implement the Algorithm**
```python
# Example: Adding to advanced/specialized/matching_algorithms.py
class MatchingAlgorithms:
    """Graph matching algorithms."""
    
    @staticmethod
    def maximum_matching(graph, **params):
        """Find maximum matching in graph."""
        import networkx as nx
        
        # Validate inputs
        if not isinstance(graph, (nx.Graph, nx.DiGraph)):
            raise ValueError("Input must be a NetworkX graph")
        
        # Implementation
        matching = nx.max_weight_matching(graph)
        
        return {
            "matching": list(matching),
            "matching_size": len(matching),
            "is_perfect": len(matching) * 2 == len(graph.nodes())
        }
```

**Step 3: Add to Module Exports**
```python
# In advanced/specialized/__init__.py
from .matching_algorithms import MatchingAlgorithms

__all__ = [
    'MatchingAlgorithms',
    # ... other exports
]
```

**Step 4: Write Tests**
```python
# tests/unit/test_matching_algorithms.py
import pytest
import networkx as nx
from networkx_mcp.advanced.specialized import MatchingAlgorithms

class TestMatchingAlgorithms:
    def test_maximum_matching(self):
        """Test maximum matching algorithm."""
        # Create test graph
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4)])
        
        # Test algorithm
        result = MatchingAlgorithms.maximum_matching(G)
        
        # Assertions
        assert "matching" in result
        assert "matching_size" in result
        assert isinstance(result["matching"], list)
        assert result["matching_size"] >= 0
```

#### 2. Adding a New I/O Format

**Step 1: Create Format Handler**
```python
# core/io/pajek_handler.py
import logging
from pathlib import Path
from typing import Any, Dict

import networkx as nx

logger = logging.getLogger(__name__)

class PajekHandler:
    """Pajek format I/O handler."""
    
    @staticmethod
    def import_from_file(file_path: str) -> nx.Graph:
        """Import graph from Pajek file."""
        try:
            return nx.read_pajek(file_path)
        except Exception as e:
            logger.error(f"Failed to import Pajek file {file_path}: {e}")
            raise
    
    @staticmethod
    def export_to_file(graph: nx.Graph, file_path: str) -> bool:
        """Export graph to Pajek file."""
        try:
            nx.write_pajek(graph, file_path)
            return True
        except Exception as e:
            logger.error(f"Failed to export to Pajek file {file_path}: {e}")
            return False
```

**Step 2: Add to I/O Package**
```python
# In core/io/__init__.py
from .pajek_handler import PajekHandler

__all__ = [
    'GraphIOHandler',
    'PajekHandler',
    # ... other handlers
]
```

### Code Style and Standards

#### 1. Function Design
```python
def algorithm_name(graph: nx.Graph, **params) -> Dict[str, Any]:
    """
    Brief description of what the algorithm does.
    
    Parameters:
    -----------
    graph : nx.Graph
        Input graph for analysis
    **params : dict
        Algorithm-specific parameters
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing results with standardized keys
    
    Raises:
    -------
    ValueError
        If graph is invalid or parameters are incorrect
    """
    # Input validation
    if not isinstance(graph, (nx.Graph, nx.DiGraph)):
        raise ValueError("Input must be a NetworkX graph")
    
    # Algorithm implementation
    result = perform_algorithm(graph, **params)
    
    # Return standardized format
    return {
        "status": "success",
        "algorithm": "algorithm_name",
        "result": result,
        "parameters": params
    }
```

#### 2. Error Handling
```python
try:
    # Main algorithm logic
    result = complex_computation(graph)
    return {"status": "success", "result": result}
    
except NetworkXError as e:
    logger.warning(f"NetworkX error in algorithm: {e}")
    return {"status": "error", "message": str(e)}
    
except Exception as e:
    logger.error(f"Unexpected error in algorithm: {e}")
    return {"status": "error", "message": "Internal algorithm error"}
```

#### 3. Module Organization
```python
# Good: Focused module
class BipartiteAlgorithms:
    """All bipartite-specific algorithms."""
    
    @staticmethod
    def is_bipartite(graph):
        """Check if graph is bipartite."""
        pass
    
    @staticmethod  
    def bipartite_projection(graph, nodes):
        """Project bipartite graph."""
        pass

# Avoid: Mixed responsibilities
class MixedAlgorithms:
    """DON'T DO THIS - mixed algorithm types."""
    
    def shortest_path(self):
        pass
    
    def community_detection(self):
        pass
    
    def bipartite_projection(self):
        pass
```

## Testing Strategy

### Test Organization

```
tests/
├── unit/               # Fast, isolated tests (< 1s each)
│   ├── test_core_*.py
│   ├── test_handlers_*.py
│   └── test_advanced_*.py
├── integration/        # Component interaction tests
│   ├── test_mcp_*.py
│   └── test_end_to_end.py
├── performance/        # Performance benchmarks
├── security/          # Security validation tests
└── conftest.py        # Test configuration and fixtures
```

### Writing Good Tests

#### 1. Unit Test Example
```python
import pytest
import networkx as nx
from networkx_mcp.advanced.directed import DirectedAnalysis

class TestDirectedAnalysis:
    """Test directed graph analysis."""
    
    @pytest.fixture
    def sample_dag(self):
        """Create sample DAG for testing."""
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])
        return G
    
    def test_dag_analysis_valid_input(self, sample_dag):
        """Test DAG analysis with valid input."""
        result = DirectedAnalysis.dag_analysis(sample_dag)
        
        assert result["status"] == "success"
        assert "is_dag" in result
        assert result["is_dag"] is True
    
    def test_dag_analysis_invalid_input(self):
        """Test DAG analysis with invalid input."""
        with pytest.raises(ValueError):
            DirectedAnalysis.dag_analysis("not a graph")
    
    def test_dag_analysis_cyclic_graph(self):
        """Test DAG analysis with cyclic graph."""
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (2, 3), (3, 1)])  # Cycle
        
        result = DirectedAnalysis.dag_analysis(G)
        assert result["is_dag"] is False
```

#### 2. Integration Test Example
```python
@pytest.mark.asyncio
async def test_complete_workflow():
    """Test complete graph analysis workflow."""
    from networkx_mcp.handlers.graph_ops import create_graph, add_nodes, add_edges
    from networkx_mcp.handlers.algorithms import shortest_path
    
    # Create graph
    result = await create_graph("workflow_test", "Graph")
    assert result["success"] is True
    
    # Add nodes and edges
    await add_nodes("workflow_test", ["A", "B", "C", "D"])
    await add_edges("workflow_test", [("A", "B"), ("B", "C"), ("C", "D")])
    
    # Run algorithm
    path_result = await shortest_path("workflow_test", "A", "D")
    assert path_result["success"] is True
    assert path_result["path"] == ["A", "B", "C", "D"]
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_server_minimal.py

# Run with coverage
pytest --cov=src/networkx_mcp --cov-report=html

# Run performance tests
pytest tests/performance/ -v

# Run tests for specific module
pytest -k "test_graph_operations"
```

## Code Quality Tools

### Automated Formatting

```bash
# Format code
ruff check . --fix         # Fix linting issues
black .                    # Format code style
isort .                    # Sort imports

# Remove unused code
autoflake --remove-all-unused-imports --recursive --in-place .
vulture src/ --min-confidence 90
```

### Type Checking

```bash
# Run type checking
mypy src/networkx_mcp/ --ignore-missing-imports

# Example type hints
from typing import Dict, List, Optional, Union
import networkx as nx

def analyze_graph(
    graph: nx.Graph,
    algorithms: List[str],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Union[str, int, float, List]]:
    """Type-annotated function example."""
    pass
```

### Pre-commit Hooks

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: local
    hooks:
      - id: tests
        name: Run Tests
        entry: python -m pytest tests/unit/
        language: system
        pass_filenames: false
      
      - id: lint
        name: Lint with Ruff
        entry: ruff check
        language: system
        
      - id: format
        name: Format with Black
        entry: black
        language: system
```

## Debugging and Troubleshooting

### Common Issues

#### 1. Import Errors
```python
# Problem: ModuleNotFoundError
from networkx_mcp.nonexistent import SomeClass

# Solution: Check module structure
from networkx_mcp.advanced.directed import DirectedAnalysis
```

#### 2. Handler Re-export Issues
```python
# Problem: Function not found in handler
from networkx_mcp.handlers.graph_ops import nonexistent_function

# Solution: Check what's actually exported
from networkx_mcp.handlers.graph_ops import create_graph, add_nodes
```

#### 3. Test Import Issues
```python
# Problem: Tests can't import modules
import sys
sys.path.insert(0, 'src')  # Add to test files if needed

# Better: Use pytest configuration
# In pytest.ini:
# [tool:pytest]
# python_paths = src
```

### Debugging Tools

#### 1. Logging
```python
import logging

logger = logging.getLogger(__name__)

def algorithm_with_logging(graph):
    """Algorithm with proper logging."""
    logger.info(f"Starting algorithm on graph with {len(graph.nodes())} nodes")
    
    try:
        result = complex_computation(graph)
        logger.debug(f"Algorithm result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Algorithm failed: {e}", exc_info=True)
        raise
```

#### 2. Performance Profiling
```python
import time
from functools import wraps

def profile_performance(func):
    """Decorator to profile function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@profile_performance
def expensive_algorithm(graph):
    """Algorithm with performance profiling."""
    pass
```

## Contributing Guidelines

### Pull Request Process

1. **Branch Naming**: `feature/description`, `fix/issue-number`, `refactor/component`
2. **Commit Messages**: Use conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`)
3. **Code Review**: All changes require review
4. **Testing**: New code must include tests
5. **Documentation**: Update docs for API changes

### Code Standards Checklist

- [ ] Function has clear purpose and single responsibility
- [ ] Comprehensive input validation
- [ ] Proper error handling and logging
- [ ] Type hints for all parameters and return values
- [ ] Docstring with parameters, returns, and examples
- [ ] Unit tests with edge cases
- [ ] Integration tests if touching multiple modules
- [ ] Performance considerations documented
- [ ] Security implications reviewed

### Release Process

1. **Version Bump**: Update `__version__.py`
2. **Changelog**: Update `CHANGELOG.md`
3. **Testing**: Full test suite passes
4. **Documentation**: Update API docs
5. **Tag Release**: Create git tag
6. **Deploy**: Follow deployment checklist

This development guide provides the foundation for working effectively with the NetworkX MCP Server's modular architecture. The clean structure makes it easy to understand, modify, and extend the codebase while maintaining high code quality standards.