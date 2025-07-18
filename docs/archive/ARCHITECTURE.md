# NetworkX MCP Server Architecture

## Overview

The NetworkX MCP Server is a modular Model Context Protocol (MCP) server that provides graph analysis capabilities using NetworkX. The codebase has been transformed from a monolithic structure into a clean, modular architecture that follows best practices for maintainability, testability, and extensibility.

## Architecture Principles

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Modular Design**: Large files have been broken into focused, manageable modules
3. **Clean Interfaces**: Clear contracts between components
4. **Backward Compatibility**: Existing code continues to work without changes
5. **Test-Driven**: Comprehensive test coverage with focused test modules

## High-Level Architecture

```
networkx-mcp-server/
├── src/networkx_mcp/           # Main source code
│   ├── server.py               # Core MCP server implementation
│   ├── handlers/               # Modular function handlers
│   ├── core/                   # Core functionality
│   ├── advanced/               # Advanced algorithms and features
│   ├── monitoring/             # Monitoring and observability
│   ├── security/               # Security features
│   └── ...
├── tests/                      # Test suite
└── docs/                       # Documentation
```

## Core Components

### 1. Server Layer (`server.py`)

- **Purpose**: Main MCP server implementation with core graph operations
- **Key Functions**: `create_graph`, `add_nodes`, `add_edges`, `graph_info`, `list_graphs`, `delete_graph`, `shortest_path`, `node_degree`
- **Architecture**: Simple, focused server with re-exports through handlers for modularity

### 2. Handlers Layer (`handlers/`)

- **Purpose**: Modular function handlers that organize server capabilities
- **Structure**:

  ```
  handlers/
  ├── __init__.py
  ├── graph_ops.py        # Basic graph operations
  └── algorithms.py       # Graph algorithms
  ```

- **Pattern**: Re-export pattern for backward compatibility while enabling modular organization

### 3. Core Layer (`core/`)

- **Purpose**: Fundamental graph operations and utilities
- **Key Modules**:
  - `graph_operations.py`: Graph management and basic operations
  - `algorithms.py`: Core graph algorithms
  - `io/`: I/O operations split by format
    - `json_handler.py`: JSON import/export
    - `gml_handler.py`: GML format support
    - `graphml_handler.py`: GraphML format support
    - `csv_handler.py`: CSV format support
    - `excel_handler.py`: Excel format support
    - `base_handler.py`: Base I/O interface

### 4. Advanced Layer (`advanced/`)

- **Purpose**: Specialized algorithms and enterprise features
- **Modular Structure**:

  ```
  advanced/
  ├── directed/               # Directed graph analysis
  │   ├── dag_analysis.py     # DAG operations
  │   ├── cycle_analysis.py   # Cycle detection
  │   ├── flow_analysis.py    # Flow algorithms
  │   └── path_analysis.py    # Path algorithms
  ├── generators/             # Graph generators
  │   ├── classic_generators.py    # Complete, cycle, path
  │   ├── random_generators.py     # Erdős-Rényi, Barabási-Albert
  │   ├── social_generators.py     # Social networks
  │   ├── geometric_generators.py  # Spatial graphs
  │   └── tree_generators.py       # Trees and forests
  ├── enterprise/             # Enterprise features
  │   ├── enterprise_features.py
  │   ├── analytics_engine.py
  │   ├── security_features.py
  │   ├── performance_optimization.py
  │   └── integration_apis.py
  ├── ml/                     # Machine learning integration
  │   ├── feature_extraction.py
  │   ├── graph_embeddings.py
  │   ├── node_classification.py
  │   ├── link_prediction.py
  │   └── graph_neural_networks.py
  ├── flow/                   # Network flow algorithms
  │   ├── max_flow.py
  │   ├── min_cost_flow.py
  │   ├── multi_commodity.py
  │   └── flow_utils.py
  └── specialized/            # Specialized algorithms
      ├── bipartite_algorithms.py
      ├── planar_algorithms.py
      ├── tree_algorithms.py
      ├── matching_algorithms.py
      └── clique_algorithms.py
  ```

### 5. Supporting Layers

#### Monitoring (`monitoring/`)

- Health checks and system monitoring
- Performance metrics collection
- Distributed tracing support
- Logging and observability

#### Security (`security/`)

- Input validation and sanitization
- Access control and authentication
- Security auditing and compliance
- Rate limiting and DDoS protection

#### Integration (`integration/`)

- External system integrations
- Data pipeline connectors
- API gateways and protocols

## Design Patterns

### 1. Handler Pattern

- **Usage**: Server function organization
- **Benefits**: Modular organization, easier testing, separation of concerns
- **Implementation**: Re-export pattern maintains backward compatibility

### 2. Strategy Pattern

- **Usage**: Algorithm selection (e.g., different centrality measures)
- **Benefits**: Extensible algorithm support, runtime algorithm selection

### 3. Factory Pattern

- **Usage**: Graph generators, I/O handler creation
- **Benefits**: Consistent object creation, easy extension

### 4. Observer Pattern

- **Usage**: Event monitoring, performance tracking
- **Benefits**: Loose coupling, extensible monitoring

## Data Flow

### Graph Operations Flow

1. **Request** → MCP Server (`server.py`)
2. **Validation** → Security layer validates input
3. **Processing** → Core operations or advanced algorithms
4. **Storage** → In-memory graph storage or persistence layer
5. **Response** → Formatted response back to client

### Modular Import Flow

1. **Client Code** → Imports from handlers (`from handlers.graph_ops import create_graph`)
2. **Handler** → Re-exports from server (`from ..server import create_graph`)
3. **Server** → Actual implementation
4. **Compatibility** → Old imports still work through compatibility layer

## Module Dependencies

```
┌─────────────┐
│   Server    │ ← Core MCP server
└─────────────┘
       ↓
┌─────────────┐
│  Handlers   │ ← Modular organization
└─────────────┘
       ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Core     │    │  Advanced   │    │ Supporting  │
│ Operations  │    │ Algorithms  │    │   Layers    │
└─────────────┘    └─────────────┘    └─────────────┘
       ↓                  ↓                  ↓
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Graph Store │    │ Specialized │    │ Monitoring  │
│     I/O     │    │  Features   │    │  Security   │
└─────────────┘    └─────────────┘    └─────────────┘
```

## Testing Architecture

### Test Organization

```
tests/
├── unit/                   # Fast, isolated tests
│   ├── test_server_minimal.py
│   ├── test_graph_operations.py
│   ├── test_algorithms.py
│   └── ...
├── integration/            # Component interaction tests
│   ├── test_mcp_tools.py
│   ├── test_integration.py
│   └── ...
├── performance/            # Performance benchmarks
├── security/               # Security tests
└── coverage/               # Coverage analysis
```

### Testing Principles

- **Fast Feedback**: Unit tests run quickly for rapid development
- **Comprehensive Coverage**: 80%+ test coverage target achieved
- **Focused Tests**: Each test file targets specific functionality
- **Clean Test Code**: Tests removed for non-existent functions, imports match actual code

## Configuration and Environment

### Environment Variables

- `NETWORKX_MCP_LOG_LEVEL`: Logging level configuration
- `NETWORKX_MCP_MAX_GRAPHS`: Maximum number of graphs in memory
- `NETWORKX_MCP_ENABLE_METRICS`: Enable performance metrics collection

### Configuration Files

- `pyproject.toml`: Project configuration and dependencies
- `pytest.ini`: Test configuration
- `.pre-commit-config.yaml`: Code quality hooks

## Performance Considerations

### Memory Management

- In-memory graph storage with configurable limits
- Lazy loading for large graph operations
- Memory cleanup for deleted graphs

### Algorithm Optimization

- Efficient algorithms for large graphs
- Configurable thresholds for complex operations
- Performance monitoring and alerting

### Scalability

- Modular architecture supports horizontal scaling
- Stateless design enables load balancing
- Monitoring supports capacity planning

## Security Architecture

### Input Validation

- Comprehensive input sanitization
- Graph ID and parameter validation
- File path security checks

### Access Control

- Authentication and authorization layers
- Rate limiting for API endpoints
- Audit logging for security events

### Data Protection

- Secure handling of graph data
- No credential logging or exposure
- Encrypted data transmission support

## Extension Points

### Adding New Algorithms

1. Create module in appropriate `advanced/` subdirectory
2. Implement algorithm following existing patterns
3. Add to handler exports if needed
4. Write comprehensive tests

### Adding New I/O Formats

1. Create handler in `core/io/`
2. Implement base handler interface
3. Add to I/O module exports
4. Test with various graph types

### Adding New Features

1. Identify appropriate layer (core, advanced, supporting)
2. Create focused module following naming conventions
3. Update package `__init__.py` exports
4. Add comprehensive test coverage

## Migration Guide

### From Monolithic to Modular

The architecture transformation maintains backward compatibility:

- **Old imports still work**: `from networkx_mcp.server import create_graph`
- **New modular imports available**: `from networkx_mcp.handlers.graph_ops import create_graph`
- **Functions behave identically**: No changes to function signatures or behavior
- **Tests pass unchanged**: Existing test suites continue to work

### Code Quality Improvements

- **Unused imports removed**: Automated cleanup with autoflake
- **Dead code eliminated**: Vulture analysis and fixes applied
- **Consistent formatting**: Applied ruff, black, and isort
- **Type safety**: mypy configuration for static analysis

## Future Roadmap

### Phase 1: Stabilization

- Complete module implementations (currently placeholders)
- Full test coverage for all modules
- Performance optimization

### Phase 2: Enhanced Features

- Advanced visualization capabilities
- Machine learning integration
- Enterprise security features

### Phase 3: Ecosystem Integration

- Plugin architecture for extensions
- External database persistence
- Cloud deployment support

## Conclusion

The NetworkX MCP Server architecture represents a significant transformation from a monolithic structure to a clean, modular design. This architecture provides:

- **Maintainability**: Small, focused modules are easier to understand and modify
- **Testability**: Isolated components can be tested independently
- **Extensibility**: New features can be added without affecting existing code
- **Reliability**: Clear interfaces and separation of concerns reduce bugs
- **Performance**: Modular design enables targeted optimizations

The architecture balances modern software engineering practices with practical constraints, ensuring that the system is both robust and accessible to developers at all levels.
