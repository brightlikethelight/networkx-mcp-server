# Changelog

All notable changes to NetworkX MCP Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.1.0] - 2026-03-16

### Added

- `topological_sort` tool for DAG workflow ordering
- `subgraph` tool to extract induced subgraphs by node set
- `merge_graphs` tool to compose two graphs into a new graph
- 12 integration tests covering lifecycle, subprocess, and protocol compliance

### Changed

- CI/CD hardening: blocking secret scan, 85% coverage threshold, pinned ruff
- Narrowed bare `except Exception` in `graph_cache.py` and `cicd_control.py`
- Added `DeprecationWarning` to 14 server.py wrapper functions (removal in v4.0)

### Fixed

- Session leak in `citations.py` (requests.Session never closed)

### Removed

- `GraphManager` class (345 lines, 0 production imports)
- `handle_error()` function (duplicated `_call_tool()` exception handling)
- Unused `validate_node_id()`, `validate_edge()`, `validate_required_params()`

## [3.0.0] - 2025-01-14

### Added

- Protocol-based dependency injection system (`core/protocols.py`)
- Async component lifecycle management (`core/lifecycle.py`)
- Structured JSON logging for production (`logging_config.py`)
- Custom CodeQL security queries for MCP-specific vulnerabilities
- Consolidated CI/CD pipelines (reduced from 19 to 8 workflows)

### Changed

- Improved exception handling with specific error types
- Enabled additional mypy type checking rules
- Reduced CI scheduled runs for better resource usage

### Fixed

- Broad exception handlers replaced with specific NetworkX error handling
- Version consistency across all configuration files

## [2.0.0] - 2025-08-01

### Added

- Advanced graph algorithms and analysis tools
- Security hardening with rate limiting and input validation
- Multi-format graph I/O (GraphML, CSV, JSON, edge lists)
- Performance monitoring and benchmarking infrastructure
- Enterprise deployment documentation

### Changed

- Modular architecture with handler separation
- Enhanced error codes following JSON-RPC 2.0 specification

## [1.0.0] - 2025-07-15

### Added

- First production release of NetworkX MCP Server
- 13 essential graph operations:
  - Core: create_graph, add_nodes, add_edges, get_info, shortest_path
  - Analysis: degree_centrality, betweenness_centrality, pagerank, connected_components, community_detection
  - Visualization: visualize_graph with multiple layouts
  - I/O: import_csv, export_json
- Comprehensive test suite with 100% coverage
- Three demo scripts showcasing real-world use cases
- Full documentation and examples

### Changed

- Upgraded from alpha minimal implementation to production-ready server
- Enhanced from 5 to 13 operations based on user needs
- Improved error handling and user feedback

### Technical Details

- Memory footprint: ~70MB (includes visualization)
- Supports graphs up to 10,000 nodes efficiently
- Compatible with Python 3.11+

## [0.1.0-alpha.2] - 2025-07-01

### Fixed

- Reduced memory footprint from 118MB to 54MB
- Removed forced pandas/scipy imports
- Implemented lazy loading for optional dependencies

### Changed

- Architectural improvements for better modularity

## [0.1.0-alpha.1] - 2025-06-27

### Added

- Initial alpha release
- Basic graph operations (5 tools)
- MCP protocol implementation
- Claude Desktop integration

---

[3.1.0]: https://github.com/Bright-L01/networkx-mcp-server/releases/tag/v3.1.0
[3.0.0]: https://github.com/Bright-L01/networkx-mcp-server/releases/tag/v3.0.0
[2.0.0]: https://github.com/Bright-L01/networkx-mcp-server/releases/tag/v2.0.0
[1.0.0]: https://github.com/Bright-L01/networkx-mcp-server/releases/tag/v1.0.0
[0.1.0-alpha.2]: https://github.com/Bright-L01/networkx-mcp-server/releases/tag/v0.1.0-alpha.2
[0.1.0-alpha.1]: https://github.com/Bright-L01/networkx-mcp-server/releases/tag/v0.1.0-alpha.1
