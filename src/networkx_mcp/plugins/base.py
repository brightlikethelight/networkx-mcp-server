"""
Base plugin architecture for NetworkX MCP Server.

This module defines the plugin interface and provides base classes for
creating modular graph analysis plugins.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict, Set, Callable
from datetime import datetime
import logging

from mcp.types import (
    Tool, Resource, Prompt, TextContent,
    GetPromptResult, PromptMessage, PromptArgument
)


logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Metadata for a plugin"""
    name: str
    version: str
    description: str
    author: Optional[str] = None
    dependencies: Set[str] = field(default_factory=set)
    required_gpu: bool = False
    min_memory_mb: int = 0
    tags: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class PluginContext:
    """Context passed to plugins for accessing core services"""
    graph_store: Any  # ThreadSafeGraphStore
    cache_manager: Any  # CacheManager
    error_handler: Any  # ErrorHandler
    config: Any  # ServerConfig
    
    def __post_init__(self):
        """Validate context"""
        if not all([self.graph_store, self.cache_manager, self.error_handler, self.config]):
            raise ValueError("Plugin context is incomplete")


class GraphPlugin(ABC):
    """
    Abstract base class for all graph plugins.
    
    Plugins must implement:
    1. Tool provision (get_tools, execute_tool)
    2. Initialization and shutdown
    3. Health checking
    4. Optional: Resources and prompts
    """
    
    def __init__(self):
        self.metadata = self._get_metadata()
        self.context: Optional[PluginContext] = None
        self.is_initialized = False
        self.tools: List[Tool] = []
        self.resources: List[Resource] = []
        self.prompts: List[Prompt] = []
        
    @abstractmethod
    def _get_metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        pass
    
    @abstractmethod
    async def initialize(self, context: PluginContext) -> bool:
        """
        Initialize the plugin with context.
        
        Args:
            context: Plugin context with core services
            
        Returns:
            True if initialization successful
        """
        self.context = context
        self.is_initialized = True
        
        # Register tools
        self.tools = await self._create_tools()
        
        # Register resources if any
        self.resources = await self._create_resources()
        
        # Register prompts if any
        self.prompts = await self._create_prompts()
        
        logger.info(f"Initialized plugin: {self.metadata.name} v{self.metadata.version}")
        return True
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Shutdown the plugin cleanly.
        
        Returns:
            True if shutdown successful
        """
        self.is_initialized = False
        logger.info(f"Shut down plugin: {self.metadata.name}")
        return True
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check plugin health.
        
        Returns:
            True if healthy
        """
        return self.is_initialized
    
    @abstractmethod
    async def _create_tools(self) -> List[Tool]:
        """Create plugin tools"""
        pass
    
    async def _create_resources(self) -> List[Resource]:
        """Create plugin resources (optional)"""
        return []
    
    async def _create_prompts(self) -> List[Prompt]:
        """Create plugin prompts (optional)"""
        return []
    
    async def get_tools(self) -> List[Tool]:
        """Get list of tools provided by plugin"""
        if not self.is_initialized:
            raise RuntimeError(f"Plugin {self.metadata.name} not initialized")
        return self.tools
    
    async def get_resources(self) -> List[Resource]:
        """Get list of resources provided by plugin"""
        if not self.is_initialized:
            raise RuntimeError(f"Plugin {self.metadata.name} not initialized")
        return self.resources
    
    async def get_prompts(self) -> List[Prompt]:
        """Get list of prompts provided by plugin"""
        if not self.is_initialized:
            raise RuntimeError(f"Plugin {self.metadata.name} not initialized")
        return self.prompts
    
    @abstractmethod
    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: PluginContext
    ) -> Any:
        """
        Execute a tool provided by this plugin.
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            context: Plugin context
            
        Returns:
            Tool execution result
        """
        pass
    
    async def read_resource(
        self,
        uri: str,
        context: PluginContext
    ) -> TextContent:
        """
        Read a resource provided by this plugin.
        
        Args:
            uri: Resource URI
            context: Plugin context
            
        Returns:
            Resource content
        """
        return TextContent(type="text", text=f"Resource {uri} not implemented")
    
    async def get_prompt(
        self,
        name: str,
        arguments: Dict[str, Any],
        context: PluginContext
    ) -> GetPromptResult:
        """
        Get a prompt provided by this plugin.
        
        Args:
            name: Prompt name
            arguments: Prompt arguments
            context: Plugin context
            
        Returns:
            Prompt result
        """
        return GetPromptResult(
            description=f"Prompt {name} not implemented",
            messages=[]
        )


class ComputePlugin(GraphPlugin):
    """
    Base class for plugins that perform graph computations.
    
    Provides common functionality for computation-based plugins.
    """
    
    def __init__(self):
        super().__init__()
        self.computation_registry: Dict[str, Callable] = {}
    
    def register_computation(
        self,
        name: str,
        func: Callable,
        cacheable: bool = True,
        ttl: int = 3600
    ):
        """Register a computation function"""
        self.computation_registry[name] = {
            "func": func,
            "cacheable": cacheable,
            "ttl": ttl
        }
    
    async def execute_computation(
        self,
        name: str,
        graph_id: str,
        **kwargs
    ) -> Any:
        """Execute a registered computation with caching"""
        if name not in self.computation_registry:
            raise ValueError(f"Unknown computation: {name}")
        
        comp = self.computation_registry[name]
        
        # Get graph from store
        graph = await self.context.graph_store.get_graph(graph_id)
        if graph is None:
            raise ValueError(f"Graph {graph_id} not found")
        
        # Check cache if cacheable
        if comp["cacheable"]:
            cache_key = f"{graph_id}:{name}:{str(kwargs)}"
            cached_result = await self.context.cache_manager.get(cache_key)
            
            if cached_result is not None:
                return cached_result
        
        # Execute computation
        result = await self.context.graph_store.compute_on_graph(
            graph_id,
            lambda g: comp["func"](g, **kwargs)
        )
        
        # Cache result if cacheable
        if comp["cacheable"]:
            await self.context.cache_manager.set(
                cache_key,
                result,
                ttl=comp["ttl"],
                tags={self.metadata.name, name}
            )
        
        return result


class AlgorithmPlugin(ComputePlugin):
    """
    Base class for plugins that implement graph algorithms.
    """
    
    def __init__(self):
        super().__init__()
        self.algorithms: Dict[str, Dict[str, Any]] = {}
    
    def register_algorithm(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: List[Dict[str, Any]],
        cacheable: bool = True,
        gpu_accelerated: bool = False
    ):
        """Register a graph algorithm"""
        self.algorithms[name] = {
            "func": func,
            "description": description,
            "parameters": parameters,
            "cacheable": cacheable,
            "gpu_accelerated": gpu_accelerated
        }
        
        # Also register as computation
        self.register_computation(name, func, cacheable)
    
    async def _create_tools(self) -> List[Tool]:
        """Create tools from registered algorithms"""
        tools = []
        
        for name, algo in self.algorithms.items():
            tool = Tool(
                name=f"{self.metadata.name}_{name}",
                description=algo["description"],
                inputSchema={
                    "type": "object",
                    "properties": {
                        "graph_id": {
                            "type": "string",
                            "description": "ID of the graph to analyze"
                        },
                        **{
                            param["name"]: param
                            for param in algo["parameters"]
                        }
                    },
                    "required": ["graph_id"]
                }
            )
            tools.append(tool)
        
        return tools
    
    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: PluginContext
    ) -> Any:
        """Execute an algorithm tool"""
        # Extract algorithm name from tool name
        prefix = f"{self.metadata.name}_"
        if not tool_name.startswith(prefix):
            raise ValueError(f"Unknown tool: {tool_name}")
        
        algo_name = tool_name[len(prefix):]
        
        if algo_name not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algo_name}")
        
        # Execute computation
        graph_id = arguments.pop("graph_id")
        return await self.execute_computation(algo_name, graph_id, **arguments)


class VisualizationPlugin(GraphPlugin):
    """
    Base class for plugins that provide graph visualization.
    """
    
    def __init__(self):
        super().__init__()
        self.layouts: Dict[str, Callable] = {}
        self.renderers: Dict[str, Callable] = {}
    
    def register_layout(self, name: str, func: Callable):
        """Register a layout algorithm"""
        self.layouts[name] = func
    
    def register_renderer(self, name: str, func: Callable):
        """Register a rendering method"""
        self.renderers[name] = func
    
    async def render_graph(
        self,
        graph_id: str,
        layout: str = "spring",
        renderer: str = "html",
        **kwargs
    ) -> str:
        """Render a graph visualization"""
        # Get graph
        graph = await self.context.graph_store.get_graph(graph_id)
        if graph is None:
            raise ValueError(f"Graph {graph_id} not found")
        
        # Apply layout
        if layout not in self.layouts:
            raise ValueError(f"Unknown layout: {layout}")
        
        positions = self.layouts[layout](graph, **kwargs)
        
        # Render
        if renderer not in self.renderers:
            raise ValueError(f"Unknown renderer: {renderer}")
        
        return self.renderers[renderer](graph, positions, **kwargs)


class DataSourcePlugin(GraphPlugin):
    """
    Base class for plugins that provide data sources.
    """
    
    def __init__(self):
        super().__init__()
        self.data_sources: Dict[str, Callable] = {}
    
    def register_data_source(
        self,
        name: str,
        loader_func: Callable,
        description: str
    ):
        """Register a data source"""
        self.data_sources[name] = {
            "loader": loader_func,
            "description": description
        }
    
    async def load_graph(
        self,
        source_name: str,
        graph_id: str,
        **kwargs
    ) -> bool:
        """Load a graph from a data source"""
        if source_name not in self.data_sources:
            raise ValueError(f"Unknown data source: {source_name}")
        
        # Load graph using loader function
        graph = await self.data_sources[source_name]["loader"](**kwargs)
        
        # Store in graph store
        return await self.context.graph_store.create_graph(
            graph_id,
            graph,
            tags={self.metadata.name, source_name}
        )


# Example implementation of a simple plugin
class CoreGraphPlugin(AlgorithmPlugin):
    """Core graph operations plugin"""
    
    def _get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="core_graph",
            version="1.0.0",
            description="Core graph operations and algorithms",
            author="NetworkX MCP",
            tags={"core", "algorithms"}
        )
    
    async def initialize(self, context: PluginContext) -> bool:
        """Initialize core plugin"""
        await super().initialize(context)
        
        # Register core algorithms
        import networkx as nx
        
        self.register_algorithm(
            name="pagerank",
            func=nx.pagerank,
            description="Calculate PageRank centrality",
            parameters=[
                {
                    "name": "alpha",
                    "type": "number",
                    "description": "Damping factor",
                    "default": 0.85
                }
            ]
        )
        
        self.register_algorithm(
            name="betweenness_centrality",
            func=nx.betweenness_centrality,
            description="Calculate betweenness centrality",
            parameters=[
                {
                    "name": "normalized",
                    "type": "boolean",
                    "description": "Normalize values",
                    "default": True
                }
            ]
        )
        
        self.register_algorithm(
            name="shortest_path",
            func=lambda g, source, target: nx.shortest_path(g, source, target),
            description="Find shortest path between nodes",
            parameters=[
                {
                    "name": "source",
                    "type": "string",
                    "description": "Source node"
                },
                {
                    "name": "target",
                    "type": "string",
                    "description": "Target node"
                }
            ],
            cacheable=True
        )
        
        return True


# Test the plugin system
async def test_plugin_system():
    """Test the plugin system"""
    # Would need actual context objects to test
    pass


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_plugin_system())