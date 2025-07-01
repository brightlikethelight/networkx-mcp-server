# Server Modularization Plan

## Overview

This document outlines the plan to refactor the monolithic `server.py` (3763 lines) into a modular architecture while maintaining backward compatibility.

## Current Structure Analysis

### server.py Statistics:
- **Total Lines**: 3763
- **Tools**: 39+ MCP tools
- **Imports**: 50+ modules
- **Constants**: 30+ configuration values
- **Single File Concerns**: Graph operations, algorithms, visualization, ML, storage, monitoring

## Proposed Modular Structure

```
src/networkx_mcp/server/
├── __init__.py
├── main.py                    # Entry point and server initialization (< 200 lines)
├── config.py                  # Configuration and constants (< 100 lines)
├── handlers/
│   ├── __init__.py
│   ├── graph_ops.py          # Graph creation, modification tools (< 300 lines)
│   ├── algorithms.py         # Path finding, centrality, etc. (< 300 lines)
│   ├── analysis.py           # Community detection, clustering (< 300 lines)
│   ├── visualization.py      # Visualization tools (< 200 lines)
│   ├── ml_tools.py          # Machine learning tools (< 300 lines)
│   ├── io_tools.py          # Import/export tools (< 200 lines)
│   └── enterprise.py        # Enterprise features (< 300 lines)
├── resources/
│   ├── __init__.py
│   ├── graph_data.py        # Graph data resources
│   ├── analysis_results.py  # Algorithm result resources
│   └── viz_data.py          # Visualization resources
├── prompts/
│   ├── __init__.py
│   ├── workflows.py         # Analysis workflow prompts
│   ├── tutorials.py         # Learning prompts
│   └── templates.py         # Custom prompt templates
└── middleware/
    ├── __init__.py
    ├── auth.py              # Authentication middleware
    ├── logging.py           # Request/response logging
    └── validation.py        # Input validation middleware
```

## Migration Strategy

### Phase 1: Extract Tool Handlers (Week 1)

1. **Create handler modules** matching tool categories
2. **Move tool implementations** to appropriate handlers
3. **Maintain thin wrappers** in server.py for compatibility
4. **Add unit tests** for each handler module

Example migration:
```python
# Old: server.py
@mcp.tool()
async def create_graph(graph_id: str, graph_type: str = "Graph"):
    # 50 lines of implementation
    ...

# New: handlers/graph_ops.py
class GraphOperationHandler:
    def __init__(self, graph_manager):
        self.graph_manager = graph_manager

    async def create_graph(self, graph_id: str, graph_type: str = "Graph"):
        # Same implementation
        ...

# New: server.py (compatibility wrapper)
@mcp.tool()
async def create_graph(graph_id: str, graph_type: str = "Graph"):
    return await graph_ops_handler.create_graph(graph_id, graph_type)
```

### Phase 2: Implement Resources & Prompts (Week 1-2)

1. **Add resource endpoints** as documented
2. **Create prompt templates** for common workflows
3. **Integrate with existing tools**
4. **Add caching layer** for resources

### Phase 3: Create Plugin Architecture (Week 2)

1. **Define plugin interface**:
```python
class ToolPlugin:
    """Base class for tool plugins."""

    def get_tools(self) -> Dict[str, Callable]:
        """Return tool name -> handler mapping."""
        raise NotImplementedError

    def get_resources(self) -> Dict[str, Callable]:
        """Return resource pattern -> handler mapping."""
        return {}

    def get_prompts(self) -> Dict[str, Callable]:
        """Return prompt name -> handler mapping."""
        return {}
```

2. **Convert handlers to plugins**
3. **Add dynamic plugin loading**
4. **Enable/disable features via config**

### Phase 4: Gradual Migration (Week 2-3)

1. **Start with new endpoints** using modular structure
2. **Migrate existing tools** one category at a time
3. **Update tests** to use new structure
4. **Maintain backward compatibility**

## Implementation Details

### 1. Configuration Management

```python
# config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class ServerConfig:
    """Centralized server configuration."""

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8765

    # Graph limits
    max_nodes: int = 1_000_000
    max_edges: int = 10_000_000
    max_graph_memory_mb: int = 1024

    # Performance
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    max_concurrent_operations: int = 100

    # Features
    enable_ml: bool = True
    enable_visualization: bool = True
    enable_enterprise: bool = False

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load configuration from environment variables."""
        # Implementation
        ...
```

### 2. Handler Base Class

```python
# handlers/base.py
from abc import ABC, abstractmethod
from typing import Dict, Callable

class BaseHandler(ABC):
    """Base class for all tool handlers."""

    def __init__(self, graph_manager, config: ServerConfig):
        self.graph_manager = graph_manager
        self.config = config
        self._tools: Dict[str, Callable] = {}
        self._register_tools()

    @abstractmethod
    def _register_tools(self):
        """Register tools provided by this handler."""
        pass

    def get_tools(self) -> Dict[str, Callable]:
        """Get all tools provided by this handler."""
        return self._tools
```

### 3. Dependency Injection

```python
# main.py
class NetworkXMCPServer:
    """Main server orchestrator."""

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig.from_env()
        self.mcp = FastMCP("NetworkX MCP Server")

        # Core dependencies
        self.graph_manager = GraphManager()
        self.validator = GraphValidator()
        self.monitor = PerformanceMonitor()

        # Initialize handlers
        self._init_handlers()

        # Initialize resources and prompts
        self.resources = GraphResources(self.mcp, self.graph_manager)
        self.prompts = GraphPrompts(self.mcp)

    def _init_handlers(self):
        """Initialize and register all handlers."""
        # Create handler instances
        handlers = [
            GraphOperationHandler(self.graph_manager, self.config),
            AlgorithmHandler(self.graph_manager, self.config),
            AnalysisHandler(self.graph_manager, self.config),
            VisualizationHandler(self.graph_manager, self.config),
        ]

        # Register optional handlers
        if self.config.enable_ml:
            handlers.append(MLHandler(self.graph_manager, self.config))

        if self.config.enable_enterprise:
            handlers.append(EnterpriseHandler(self.graph_manager, self.config))

        # Register all tools
        for handler in handlers:
            for tool_name, tool_func in handler.get_tools().items():
                self._register_tool(tool_name, tool_func)
```

### 4. Testing Strategy

```python
# tests/unit/handlers/test_graph_ops.py
import pytest
from networkx_mcp.server.handlers.graph_ops import GraphOperationHandler

class TestGraphOperationHandler:
    @pytest.fixture
    def handler(self, mock_graph_manager, test_config):
        return GraphOperationHandler(mock_graph_manager, test_config)

    async def test_create_graph(self, handler):
        result = await handler.create_graph("test_id", "DiGraph")
        assert result["success"] is True
        assert result["graph_type"] == "DiGraph"

    async def test_add_nodes(self, handler):
        # Test implementation
        ...
```

## Benefits

### 1. **Maintainability**
- Smaller, focused modules
- Clear separation of concerns
- Easier to understand and modify

### 2. **Testability**
- Unit test individual handlers
- Mock dependencies easily
- Better test coverage

### 3. **Extensibility**
- Plugin architecture for new features
- Easy to add/remove functionality
- Third-party extensions possible

### 4. **Performance**
- Lazy loading of optional features
- Better memory management
- Parallel handler initialization

### 5. **Team Collaboration**
- Multiple developers can work on different handlers
- Clear module boundaries
- Reduced merge conflicts

## Migration Checklist

- [ ] Create new directory structure
- [ ] Extract configuration to config.py
- [ ] Create base handler classes
- [ ] Migrate graph operation tools
- [ ] Migrate algorithm tools
- [ ] Migrate analysis tools
- [ ] Migrate visualization tools
- [ ] Migrate ML tools
- [ ] Migrate enterprise tools
- [ ] Add resource implementations
- [ ] Add prompt implementations
- [ ] Create compatibility layer
- [ ] Update all tests
- [ ] Update documentation
- [ ] Performance testing
- [ ] Deprecation notices for old structure

## Backward Compatibility

During migration:
1. Keep `server.py` functional with thin wrappers
2. Add deprecation warnings for direct imports
3. Provide migration guide for users
4. Support both old and new structures for 2 versions
5. Remove old structure in v3.0.0

## Success Metrics

- [ ] All modules < 500 lines
- [ ] Test coverage > 90%
- [ ] No performance regression
- [ ] Zero breaking changes
- [ ] Improved startup time
- [ ] Reduced memory footprint

## Timeline

- **Week 1**: Core modularization
- **Week 2**: Resources & Prompts
- **Week 3**: Testing & Documentation
- **Week 4**: Performance optimization
- **Week 5**: Migration tools
- **Week 6**: Release preparation
