# Real Performance Report - NetworkX MCP Server v0.1.0

**Date**: 2025-07-08 16:36:21
**Testing Environment**: {'python_version': '3.12.9 | packaged by conda-forge | (main, Feb 14 2025, 07:56:32) [Clang 18.1.8 ]', 'platform': 'darwin', 'cpu_count': 8, 'memory_gb': 16.0, 'networkx_version': '3.4.2'}
**Testing Method**: Real subprocess communication with memory profiling

## Node Scaling Results

| Nodes | Time (s) | Memory (MB) | Memory/Node (KB) | Status |
|-------|----------|-------------|------------------|---------|
| 100 | 0.01 | 59.4 | 0.0 | ✓ |
| 500 | 0.06 | 59.5 | 0.2 | ✓ |
| 1,000 | 0.13 | 59.8 | 0.2 | ✓ |
| 2,500 | 0.29 | 60.3 | 0.2 | ✓ |
| 5,000 | 0.59 | 47.6 | -2.6 | ✓ |
| 10,000 | 1.19 | 49.6 | 0.2 | ✓ |

## Edge Scaling Results

| Edges | Time (s) | Memory (MB) | Memory/Edge (bytes) | Status |
|-------|----------|-------------|---------------------|---------|
| 100 | 0.01 | 34.0 | 492 | ✓ |
| 500 | 0.06 | 34.1 | 328 | ✓ |
| 1,000 | 0.11 | 34.2 | 66 | ✓ |
| 2,500 | 0.30 | 34.5 | 131 | ✓ |
| 5,000 | 0.62 | 27.0 | -1494 | ✓ |
| 10,000 | 1.19 | 29.3 | 234 | ✓ |

## Basic Operation Performance

| Operation | Time (ms) | Status |
|-----------|-----------|---------|
| create_graph | 935.3 | ✓ |
| add_nodes | 11.1 | ✓ |
| add_edges | 24.5 | ✓ |
| get_graph_info | 11.1 | ✓ |

## Algorithm Performance

| Algorithm | Time (ms) | Status |
|-----------|-----------|---------|
| shortest_path | 10.5 | ✓ |
| centrality | 11.1 | ✓ |

## Conclusions

- **Maximum tested nodes**: 10,000
- **Maximum tested edges**: 10,000
- **Memory per node**: ~0.2 KB
- **Memory per edge**: ~234 bytes

*This report contains actual measured performance data, not estimates.*