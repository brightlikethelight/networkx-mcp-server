#!/usr/bin/env python3
"""Simple synchronous Redis backend for immediate testing."""

import pickle
import json
import redis
import networkx as nx
from typing import Dict, Any, Optional
from datetime import datetime

class RedisGraphStorage:
    """Simple Redis storage backend compatible with current add_persistence.py"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", key_prefix: str = "networkx", compression: bool = True):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.compression = compression
        
        # Create Redis connection
        self.client = redis.from_url(redis_url, decode_responses=False)
        
        # Test connection
        try:
            self.client.ping()
            print(f"✅ Connected to Redis at {redis_url}")
        except Exception as e:
            raise Exception(f"Failed to connect to Redis: {e}")
    
    def _make_key(self, *parts):
        """Create a Redis key with namespace."""
        return f"{self.key_prefix}:" + ":".join(str(p) for p in parts)
    
    def save_graph(self, user_id: str, graph_id: str, graph: nx.Graph, metadata: Optional[Dict[str, Any]] = None):
        """Save graph to Redis."""
        try:
            # Serialize the graph
            graph_data = pickle.dumps(graph)
            
            # Create metadata
            graph_metadata = {
                "user_id": user_id,
                "graph_id": graph_id,
                "created_at": datetime.now().isoformat(),
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "graph_type": type(graph).__name__,
                "size_bytes": len(graph_data)
            }
            
            if metadata:
                graph_metadata.update(metadata)
            
            # Redis keys
            data_key = self._make_key("graph", user_id, graph_id)
            meta_key = self._make_key("metadata", user_id, graph_id)
            user_graphs_key = self._make_key("user_graphs", user_id)
            
            # Save to Redis atomically
            pipe = self.client.pipeline()
            pipe.set(data_key, graph_data)
            pipe.set(meta_key, json.dumps(graph_metadata))
            pipe.sadd(user_graphs_key, graph_id)
            pipe.execute()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save graph {graph_id}: {e}")
            return False
    
    def load_graph(self, user_id: str, graph_id: str) -> Optional[nx.Graph]:
        """Load graph from Redis."""
        try:
            data_key = self._make_key("graph", user_id, graph_id)
            graph_data = self.client.get(data_key)
            
            if graph_data is None:
                return None
            
            # Deserialize
            graph = pickle.loads(graph_data)
            return graph
            
        except Exception as e:
            print(f"❌ Failed to load graph {graph_id}: {e}")
            return None
    
    def delete_graph(self, user_id: str, graph_id: str):
        """Delete graph from Redis."""
        try:
            data_key = self._make_key("graph", user_id, graph_id)
            meta_key = self._make_key("metadata", user_id, graph_id)
            user_graphs_key = self._make_key("user_graphs", user_id)
            
            # Delete atomically
            pipe = self.client.pipeline()
            pipe.delete(data_key, meta_key)
            pipe.srem(user_graphs_key, graph_id)
            result = pipe.execute()
            
            return result[0] > 0  # True if something was deleted
            
        except Exception as e:
            print(f"❌ Failed to delete graph {graph_id}: {e}")
            return False
    
    def list_graphs(self, user_id: str):
        """List user's graphs."""
        try:
            user_graphs_key = self._make_key("user_graphs", user_id)
            graph_ids = self.client.smembers(user_graphs_key)
            
            graphs = []
            for graph_id_bytes in graph_ids:
                graph_id = graph_id_bytes.decode('utf-8') if isinstance(graph_id_bytes, bytes) else graph_id_bytes
                
                # Get metadata
                meta_key = self._make_key("metadata", user_id, graph_id)
                metadata = self.client.get(meta_key)
                
                if metadata:
                    meta_dict = json.loads(metadata)
                    graphs.append(meta_dict)
                else:
                    # Fallback - just graph ID
                    graphs.append({
                        "user_id": user_id,
                        "graph_id": graph_id,
                        "created_at": "unknown"
                    })
            
            return graphs
            
        except Exception as e:
            print(f"❌ Failed to list graphs for {user_id}: {e}")
            return []
    
    def get_stats(self, user_id: str):
        """Get storage statistics."""
        try:
            user_graphs_key = self._make_key("user_graphs", user_id)
            graph_count = self.client.scard(user_graphs_key)
            
            return {
                "user_id": user_id,
                "graph_count": graph_count,
                "backend": "redis"
            }
            
        except Exception as e:
            print(f"❌ Failed to get stats for {user_id}: {e}")
            return {
                "user_id": user_id,
                "graph_count": 0,
                "backend": "redis_error"
            }
    
    def check_health(self):
        """Check Redis health."""
        try:
            self.client.ping()
            info = self.client.info()
            
            return {
                "status": "healthy",
                "backend": "redis",
                "redis_version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown")
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "backend": "redis",
                "error": str(e)
            }

def test_redis_backend():
    """Test the Redis backend directly."""
    print("🧪 Testing Redis backend directly...")
    
    try:
        # Create storage
        storage = RedisGraphStorage()
        
        # Test health
        health = storage.check_health()
        print(f"📊 Redis health: {health}")
        
        if health["status"] != "healthy":
            return False
        
        # Create test graph
        test_graph = nx.DiGraph()
        test_graph.add_nodes_from(["A", "B", "C"])
        test_graph.add_edges_from([("A", "B"), ("B", "C")])
        
        # Save graph
        success = storage.save_graph("test_user", "test_graph", test_graph, {"test": True})
        if not success:
            print("❌ Failed to save test graph")
            return False
        
        print("✅ Saved test graph to Redis")
        
        # Load graph
        loaded_graph = storage.load_graph("test_user", "test_graph")
        if loaded_graph is None:
            print("❌ Failed to load test graph")
            return False
        
        # Verify data integrity
        if (loaded_graph.number_of_nodes() == 3 and 
            loaded_graph.number_of_edges() == 2 and
            list(loaded_graph.nodes()) == ["A", "B", "C"]):
            print("✅ Graph data integrity verified")
        else:
            print("❌ Graph data corruption detected")
            return False
        
        # List graphs
        graphs = storage.list_graphs("test_user")
        if len(graphs) > 0:
            print(f"✅ Listed {len(graphs)} graphs")
        else:
            print("❌ Failed to list graphs")
            return False
        
        # Get stats
        stats = storage.get_stats("test_user")
        print(f"📊 Storage stats: {stats}")
        
        # Clean up
        storage.delete_graph("test_user", "test_graph")
        print("✅ Cleaned up test data")
        
        return True
        
    except Exception as e:
        print(f"❌ Redis backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 REDIS BACKEND TEST")
    print("=" * 40)
    
    success = test_redis_backend()
    
    if success:
        print("\n✅ REDIS BACKEND IS WORKING!")
        print("🎉 Ready for integration with NetworkX MCP server")
    else:
        print("\n❌ REDIS BACKEND FAILED!")
        print("🚫 Not ready for production")
    
    exit(0 if success else 1)