# 🔄 Migration Guide: From Prototype to Production

## Overview

This guide shows how to migrate the existing NetworkX MCP Server from a monolithic prototype to a production-ready system with proper security, persistence, and architecture.

## 🚨 Critical Path (Do These First!)

### Step 1: Add Persistence (Day 1-2)

```python
# 1. Install Redis
brew install redis  # macOS
sudo apt-get install redis-server  # Ubuntu

# 2. Start Redis
redis-server

# 3. Test Redis backend
python -c "
from src.networkx_mcp.storage.redis_backend import RedisBackend
import asyncio

async def test():
    backend = RedisBackend()
    await backend.initialize()
    health = await backend.check_health()
    print('Redis health:', health)

asyncio.run(test())
"

# 4. Update pyproject.toml dependencies
redis = ">=5.0.0"  # or aioredis for older Python
```

### Step 2: Implement Security Layer (Day 3-4)

```python
# Example: Securing the create_graph endpoint

# OLD (server.py - UNSAFE):
@mcp.tool()
async def create_graph(graph_id: str, graph_type: str = "Graph", 
                      params: Optional[Dict[str, Any]] = None):
    # No validation!
    result = graph_manager.create_graph(graph_id, graph_type, params)
    return result

# NEW (handlers/graph_handlers.py - SECURE):
from ..security.validator import SecurityValidator
from ..monitoring.resource_manager import ResourceManager
from ..audit.audit_logger import audit_log

class GraphHandlers:
    @audit_log("create_graph")
    @ResourceManager.resource_limited("create_graph")
    async def create_graph(self, user_id: str, graph_id: str, 
                          graph_type: str = "Graph",
                          params: Optional[Dict[str, Any]] = None):
        # 1. Validate ALL inputs
        user_id = SecurityValidator.validate_user_id(user_id)
        graph_id = SecurityValidator.validate_graph_id(graph_id)
        graph_type = SecurityValidator.validate_graph_type(graph_type)
        
        # 2. Check resource limits
        user_stats = await self.storage.get_storage_stats(user_id)
        if user_stats['graph_count'] >= self.limits.max_graphs_per_user:
            raise LimitExceeded(f"User graph limit reached")
        
        # 3. Sanitize parameters
        safe_params = SecurityValidator.sanitize_attributes(params or {})
        
        # 4. Create with transaction
        async with self.storage.transaction() as tx:
            graph = self._create_graph_instance(graph_type, **safe_params)
            await self.storage.save_graph(user_id, graph_id, graph, tx)
            
        return {"status": "success", "graph_id": graph_id}
```

### Step 3: Break Up Monolithic server.py (Day 5-6)

```bash
# Current structure (BAD):
src/networkx_mcp/
└── server.py  # 3500+ lines! 😱

# New structure (GOOD):
src/networkx_mcp/
├── server.py          # ~200 lines - just setup
├── handlers/
│   ├── __init__.py
│   ├── graph_handlers.py      # ~300 lines
│   ├── algorithm_handlers.py  # ~400 lines
│   ├── visualization_handlers.py  # ~300 lines
│   └── health_handlers.py     # ~100 lines
├── services/
│   ├── __init__.py
│   ├── graph_service.py       # Business logic
│   └── algorithm_service.py   # Algorithm implementations
└── middleware/
    ├── __init__.py
    ├── auth.py               # Authentication
    └── monitoring.py         # Metrics
```

### Step 4: Migration Script

```python
#!/usr/bin/env python3
"""Migrate existing graphs to new secure storage."""

import asyncio
import json
from pathlib import Path
from src.networkx_mcp.storage.redis_backend import RedisBackend
from src.networkx_mcp.core.graph_operations import GraphManager

async def migrate_graphs():
    """Migrate from in-memory to Redis."""
    print("🔄 Starting migration...")
    
    # Old system
    old_manager = GraphManager()
    
    # New system
    new_storage = RedisBackend()
    await new_storage.initialize()
    
    # Migrate each graph
    migrated = 0
    for graph_id, graph in old_manager.graphs.items():
        try:
            # Assign to default user for now
            await new_storage.save_graph(
                user_id="migrated_user",
                graph_id=graph_id,
                graph=graph,
                metadata={"migrated_at": datetime.now().isoformat()}
            )
            migrated += 1
            print(f"✅ Migrated: {graph_id}")
        except Exception as e:
            print(f"❌ Failed to migrate {graph_id}: {e}")
    
    print(f"\n✅ Migration complete: {migrated} graphs")
    
    # Verify
    graphs = await new_storage.list_graphs("migrated_user")
    print(f"📊 Verified: {len(graphs)} graphs in new storage")

if __name__ == "__main__":
    asyncio.run(migrate_graphs())
```

## 🔧 Gradual Migration Strategy

### Phase 1: Add Security Wrapper (1 week)

```python
# Create a security wrapper around existing server
class SecureServer:
    def __init__(self, legacy_server):
        self.legacy = legacy_server
        self.validator = SecurityValidator()
        self.rate_limiter = RateLimiter()
    
    async def create_graph(self, user_id: str, graph_id: str, **kwargs):
        # Add security layer
        user_id = self.validator.validate_user_id(user_id)
        graph_id = self.validator.validate_graph_id(graph_id)
        self.rate_limiter.check_rate_limit(user_id)
        
        # Call legacy
        return await self.legacy.create_graph(graph_id, **kwargs)

# Use wrapper
secure_server = SecureServer(legacy_server)
```

### Phase 2: Add Persistence (1 week)

```python
# Add persistence to critical operations
class PersistentGraphManager(GraphManager):
    def __init__(self, storage: StorageBackend):
        super().__init__()
        self.storage = storage
    
    async def create_graph(self, user_id: str, graph_id: str, graph_type: str):
        # Create in memory (legacy)
        result = super().create_graph(graph_id, graph_type)
        
        # Also persist
        graph = self.graphs[graph_id]
        await self.storage.save_graph(user_id, graph_id, graph)
        
        return result
```

### Phase 3: Refactor Architecture (2 weeks)

```python
# Gradually move tools to new handlers
# OLD:
@mcp.tool()
async def shortest_path(graph_id: str, source: str, target: str):
    # 100 lines of code here...

# NEW:
# In handlers/algorithm_handlers.py
class AlgorithmHandlers:
    async def shortest_path(self, user_id: str, graph_id: str, 
                          source: str, target: str):
        # Moved and improved code...

# In server.py (minimal adapter)
@mcp.tool()
async def shortest_path(graph_id: str, source: str, target: str):
    # Just delegate to new handler
    return await algorithm_handlers.shortest_path(
        user_id="legacy",  # Temporary
        graph_id=graph_id,
        source=source,
        target=target
    )
```

## 🧪 Testing Migration

### 1. Parallel Testing

```python
# Run old and new in parallel, compare results
async def test_parallel():
    # Old system
    old_result = await old_server.create_graph("test1", "Graph")
    
    # New system
    new_result = await new_server.create_graph("user1", "test1", "Graph")
    
    # Compare
    assert old_result['status'] == new_result['status']
    print("✅ Results match!")
```

### 2. Load Testing

```bash
# Test new system under load
locust -f load_tests.py --host=http://localhost:8765
```

### 3. Security Testing

```bash
# Run security scanner
python security_tests.py

# Test injection attempts
curl -X POST http://localhost:8765/create_graph \
  -d '{"graph_id": "../../etc/passwd", "graph_type": "Graph"}'
# Should be rejected!
```

## 📊 Monitoring Migration

```python
# Track migration progress
async def migration_status():
    old_graphs = len(old_manager.graphs)
    new_graphs = await new_storage.get_storage_stats("all")
    
    print(f"""
    Migration Status:
    - Old system: {old_graphs} graphs
    - New system: {new_graphs['graph_count']} graphs  
    - Progress: {new_graphs['graph_count']/old_graphs*100:.1f}%
    """)
```

## ⚠️ Rollback Plan

```python
# If things go wrong, rollback procedure:

# 1. Stop new server
systemctl stop networkx-mcp-new

# 2. Export data from new system
python export_new_data.py --output backup.json

# 3. Restart old server
systemctl start networkx-mcp-old

# 4. Import critical data
python import_to_old.py --input backup.json
```

## 🎯 Success Criteria

Before declaring migration complete:

- [ ] All tests pass (unit, integration, e2e)
- [ ] No data loss (verify graph counts match)
- [ ] Performance acceptable (< 10% degradation)
- [ ] Security scan clean
- [ ] Monitoring in place
- [ ] Rollback tested
- [ ] Documentation updated
- [ ] Team trained on new system

## 📅 Timeline

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1 | Security & Persistence | Secure storage layer |
| 2 | Architecture | Modular structure |
| 3 | Migration | Data moved to new system |
| 4 | Testing | All tests passing |
| 5 | Deployment | Running in staging |
| 6 | Cutover | Production switch |

## 🚀 Next Steps

1. **Start with security** - It's easier to add security first than retrofit later
2. **Set up Redis** - Get persistence working ASAP
3. **Create migration script** - Automate the data migration
4. **Test thoroughly** - Especially edge cases and error scenarios
5. **Monitor everything** - You can't fix what you can't see

Remember: **It's better to migrate slowly and safely than quickly and break things!**