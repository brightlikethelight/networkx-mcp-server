# Resource Limits and DoS Prevention

## Overview
This document describes the resource limits implemented in the NetworkX MCP Server to prevent Denial of Service (DoS) attacks and ensure server stability.

## 1. Memory Limits

### Configuration
- **Max Memory**: 1GB (default, configurable via `MAX_MEMORY_MB` env var)
- **Max Graph Size**: 100MB per graph (configurable via `MAX_GRAPH_SIZE_MB`)
- **Memory Check Threshold**: Warnings at 80% usage

### Features
- Real-time memory monitoring every 30 seconds
- Automatic garbage collection when memory > 90%
- Memory usage estimation for graphs before operations
- Process memory tracking with psutil

### Example
```python
# Memory is checked before and after operations
create_graph("test", "undirected", {
    "nodes": range(1000000)  # Would exceed memory limit
})
# Result: {"error": "Graph size (95.4MB) exceeds limit (100MB)"}
```

## 2. Operation Timeouts

### Configuration
- **Default Timeout**: 30 seconds (configurable via `OPERATION_TIMEOUT`)
- Applied to all graph operations automatically
- Prevents long-running operations from blocking the server

### Protected Operations
- Graph creation and modification
- Shortest path algorithms
- Graph analysis operations
- All MCP tool functions

### Example
```python
# Operations that take too long are terminated
@timeout(seconds=30)
def shortest_path(...):
    # If this takes > 30s, raises TimeoutError
```

## 3. Concurrent Request Limits

### Configuration
- **Max Concurrent Requests**: 10 (configurable via `MAX_CONCURRENT_REQUESTS`)
- Thread-safe request counting
- Automatic slot release after operation

### Features
- Prevents resource exhaustion from parallel requests
- Returns "Server busy" error when limit reached
- Fair queuing with automatic cleanup

### Example
```python
# When 10 requests are active, new ones are rejected
for i in range(20):
    graph_info("test")  # First 10 succeed, rest get "Server busy"
```

## 4. Graph Size Validation

### Limits
- **Max Nodes**: 100,000 per graph
- **Max Edges**: 1,000,000 per graph
- **Max Graph Memory**: 100MB estimated size

### Operation Feasibility
Different operations have different complexity limits:
- `shortest_path`: Max 10,000 nodes (O(V + E))
- `all_pairs_shortest_path`: Max 1,000 nodes (O(V³))
- `betweenness_centrality`: Max 5,000 nodes (O(VE))
- `diameter`: Max 1,000 nodes (O(V³))

### Example
```python
# Large graphs are rejected
create_graph("huge", "undirected", {
    "nodes": range(200000),  # Exceeds 100,000 limit
    "edges": []
})
# Result: {"error": "Too many nodes. Maximum allowed: 1000"}
```

## 5. Rate Limiting

### Configuration
- **Requests per minute**: 60 (configurable via `REQUESTS_PER_MINUTE`)
- Sliding window rate limiting
- Per-process (not per-user)

### Features
- Prevents request flooding
- Automatic cleanup of old request timestamps
- Returns rate limit error when exceeded

## 6. Environment Variables

Configure limits via environment variables:

```bash
# Memory limits
export MAX_MEMORY_MB=2048           # 2GB max memory
export MAX_GRAPH_SIZE_MB=200        # 200MB per graph

# Time limits
export OPERATION_TIMEOUT=60         # 60 second timeout

# Concurrency
export MAX_CONCURRENT_REQUESTS=20   # 20 concurrent requests

# Graph size
export MAX_NODES_PER_GRAPH=200000   # 200k nodes
export MAX_EDGES_PER_GRAPH=2000000  # 2M edges

# Rate limiting
export REQUESTS_PER_MINUTE=120      # 120 requests/minute
```

## 7. Resource Monitoring

### Status Endpoint
Use the `resource_status()` tool to check current usage:

```json
{
  "memory": {
    "current_mb": 144.3,
    "limit_mb": 1024,
    "available_mb": 2653.3
  },
  "requests": {
    "active": 2,
    "max_concurrent": 10,
    "recent_per_minute": 34,
    "limit_per_minute": 60
  },
  "limits": {
    "max_graph_size_mb": 100,
    "max_nodes_per_graph": 100000,
    "max_edges_per_graph": 1000000,
    "operation_timeout_seconds": 30
  }
}
```

### Background Monitoring
- Automatic memory monitoring every 30 seconds
- Warning logs when approaching limits
- Automatic garbage collection when needed

## 8. Error Messages

Safe error messages that don't expose internals:
- `"Graph would exceed memory limits"`
- `"Operation timed out"`
- `"Server busy. Please try again later."`
- `"Rate limit exceeded. Please try again later."`
- `"Graph too large for operation"`

## 9. Testing Resource Limits

### Unit Tests
```bash
python -m pytest tests/security/test_resource_limits.py -v
```

### Demo Script
```bash
python tests/security/test_dos_prevention_demo.py
```

### Attack Scenarios Tested
1. **Large Graph Creation**: Attempting to create graphs with millions of nodes
2. **Memory Exhaustion**: Creating many graphs to exhaust memory
3. **Concurrent Flooding**: Sending many parallel requests
4. **Long Operations**: Operations that would run indefinitely
5. **Rate Limit Bypass**: Rapid request flooding

## 10. Best Practices

### For Server Operators
1. Monitor resource usage regularly via `resource_status()`
2. Adjust limits based on available hardware
3. Set up alerts for high memory usage
4. Review logs for repeated limit violations (potential attacks)

### For Developers
1. Always use `@with_resource_limits` decorator on new operations
2. Check operation feasibility before expensive algorithms
3. Estimate memory usage for large data structures
4. Handle ResourceLimitError appropriately

### For Users
1. Keep graphs under 100,000 nodes for best performance
2. Batch operations when possible
3. Use pagination for large result sets
4. Respect rate limits

## Conclusion

The NetworkX MCP Server implements comprehensive resource limits to prevent DoS attacks while maintaining good performance for legitimate use cases. All limits are configurable and monitored in real-time.