# MCP Implementation Plan

## Executive Summary

The NetworkX MCP Server currently implements only the **Tools** portion of the MCP protocol. While it has 25+ working tools, the **Resources** and **Prompts** features are already coded but not connected to the main server. This represents a significant missed opportunity - we have ~60% of the code written but only expose ~40% of MCP capabilities.

## Quick Wins (1-2 days)

### 1. Connect Existing Resources (2-4 hours)
**Location:** `src/networkx_mcp/mcp/resources/__init__.py`

```python
# In server.py or server_with_storage.py, add:
from .mcp.resources import GraphResources

# After creating mcp instance:
resources = GraphResources(mcp, graph_manager)
```

**Benefits:**
- Exposes 5 resource endpoints immediately
- Enables read-only graph access
- Supports graph data export
- Provides statistics endpoint

### 2. Connect Existing Prompts (2-4 hours)
**Location:** `src/networkx_mcp/mcp/prompts/__init__.py`

```python
# In server.py or server_with_storage.py, add:
from .mcp.prompts import GraphPrompts

# After creating mcp instance:
prompts = GraphPrompts(mcp)
```

**Benefits:**
- Provides 6 workflow templates
- Guides users through complex operations
- Improves discoverability
- Zero new code needed

### 3. Update FastMCPCompat (1-2 hours)
Enhance the compatibility layer to properly support resources and prompts:

```python
class FastMCPCompat:
    def resource(self, uri_pattern: str):
        """Decorator for resources."""
        # Implementation
    
    def list_resources(self):
        """List all registered resources."""
        # Implementation
```

## Medium-Term Improvements (1 week)

### 1. Add Schema Validation (1-2 days)
Create JSON schemas for all tools:

```python
@mcp.tool(
    description="Create a new graph",
    schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[a-zA-Z0-9_-]+$"},
            "graph_type": {"enum": ["undirected", "directed", "multi", "multi_directed"]},
            "data": {"type": "object"}
        },
        "required": ["name"]
    }
)
def create_graph(name: str, graph_type: str = "undirected", data: dict = None):
    # Existing implementation
```

### 2. Async Tool Support (2-3 days)
Convert I/O-bound tools to async:

```python
@mcp.tool(description="Load graph from file")
async def load_graph_async(file_path: str, format: str = "graphml"):
    # Async file operations
    async with aiofiles.open(file_path) as f:
        data = await f.read()
    # Process...
```

### 3. Progress Reporting (1-2 days)
Add progress callbacks for long operations:

```python
@mcp.tool(description="Compute all pairs shortest paths")
async def all_pairs_shortest_path_with_progress(
    graph_name: str, 
    progress_callback = None
):
    graph = get_graph(graph_name)
    total = graph.number_of_nodes()
    
    for i, source in enumerate(graph.nodes()):
        if progress_callback:
            await progress_callback(i / total, f"Processing node {source}")
        # Compute paths...
```

## Long-Term Enhancements (1 month)

### 1. Full MCP Protocol Support
- Implement proper JSON-RPC handling
- Add request/response correlation
- Support multiple transports (HTTP, WebSocket)
- Protocol version negotiation

### 2. Advanced Features
- **Streaming:** For large graph data
- **Subscriptions:** Real-time graph updates
- **Transactions:** Multi-step atomic operations
- **Batch Operations:** Process multiple requests

### 3. Developer Experience
- Auto-generate API documentation
- Create TypeScript client types
- Build testing framework
- Add debugging tools

## Implementation Strategy

### Phase 1: Connect What Exists (Week 1)
1. **Day 1-2:** Wire up resources and prompts
2. **Day 3-4:** Test integration, fix issues
3. **Day 5:** Update documentation

**Deliverable:** Full MCP server with tools, resources, and prompts

### Phase 2: Enhance Tools (Week 2)
1. **Day 1-2:** Add JSON schemas
2. **Day 3-4:** Implement async tools
3. **Day 5:** Add progress reporting

**Deliverable:** Production-ready tool implementation

### Phase 3: Advanced Protocol (Week 3-4)
1. **Week 3:** Streaming and subscriptions
2. **Week 4:** Full protocol compliance

**Deliverable:** Enterprise-grade MCP server

## Testing Strategy

### Unit Tests
```python
def test_resource_registration():
    """Test that resources are properly registered."""
    assert "graph://catalog" in mcp.list_resources()
    
def test_prompt_discovery():
    """Test that prompts are discoverable."""
    prompts = mcp.list_prompts()
    assert "analyze_social_network" in prompts
```

### Integration Tests
```python
async def test_full_workflow():
    """Test tools, resources, and prompts together."""
    # Create graph via tool
    await create_graph("test")
    
    # Access via resource
    data = await mcp.get_resource("graph://data/test")
    
    # Use prompt
    workflow = await mcp.get_prompt("analyze_social_network", graph_id="test")
```

## Success Metrics

### Immediate (After Phase 1)
- ✅ 25+ tools exposed
- ✅ 5+ resources available
- ✅ 6+ prompts accessible
- ✅ All existing code utilized

### Short-term (After Phase 2)
- ✅ 100% tool schema coverage
- ✅ 50%+ async tool conversion
- ✅ Progress reporting for long operations
- ✅ Improved error messages

### Long-term (After Phase 3)
- ✅ Full MCP protocol compliance
- ✅ Streaming support
- ✅ Real-time subscriptions
- ✅ Enterprise features

## Risk Mitigation

### Technical Risks
- **FastMCP limitations:** Have fallback implementation
- **Async complexity:** Gradual migration
- **Breaking changes:** Version the API

### Resource Risks
- **Time constraints:** Focus on quick wins first
- **Complexity:** Reuse existing code
- **Testing:** Automated test suite

## Conclusion

The NetworkX MCP Server is well-positioned for full MCP compliance. With 60% of the code already written but not connected, we can achieve dramatic improvements with minimal effort. The phased approach ensures quick wins while building toward full compliance.

**Recommendation:** Start immediately with Phase 1 (connecting existing resources and prompts). This requires only 1-2 days of work but doubles the server's MCP capabilities.