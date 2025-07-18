# Changelog

All notable changes to NetworkX MCP Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.0.0]: https://github.com/Bright-L01/networkx-mcp-server/releases/tag/v1.0.0
[0.1.0-alpha.2]: https://github.com/Bright-L01/networkx-mcp-server/releases/tag/v0.1.0-alpha.2
[0.1.0-alpha.1]: https://github.com/Bright-L01/networkx-mcp-server/releases/tag/v0.1.0-alpha.1
