"""
Modular MCP server core with clean architecture and error handling.

This module provides the core server implementation with < 500 lines,
delegating functionality to plugins and maintaining separation of concerns.
"""

import asyncio
import json
import logging
import sys
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Type
from abc import ABC, abstractmethod

from mcp.server import Server
from mcp.types import (
    Tool, TextContent, Resource, Prompt,
    CallToolResult, ListResourcesResult,
    ListPromptsResult, ListToolsResult,
    GetPromptResult, ReadResourceResult
)

from .thread_safe_store import ThreadSafeGraphStore, GraphOperation
from .error_handling import ErrorHandler, GraphError, GraphErrorType, ErrorRecoveryStrategy
from .cache_manager import CacheManager
from ..plugins.base import GraphPlugin, PluginMetadata, PluginContext


logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Server configuration"""
    name: str = "NetworkX MCP Server"
    version: str = "2.0.0"
    max_memory_mb: int = 1024
    enable_gpu: bool = True
    enable_caching: bool = True
    cache_size_mb: int = 256
    plugin_directory: str = "./plugins"
    enable_persistence: bool = False
    persistence_path: str = "./data"
    log_level: str = "INFO"
    metrics_enabled: bool = True
    health_check_interval: int = 60


class MCPGraphServer:
    """
    Modular MCP server core with plugin architecture.
    
    This is the minimal core server that:
    1. Handles MCP protocol communication
    2. Manages plugins and their lifecycle
    3. Routes requests to appropriate plugins
    4. Handles errors and recovery
    5. Maintains metrics and health
    
    All actual graph operations are delegated to plugins.
    """
    
    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig()
        self.server = Server(self.config.name)
        
        # Core components
        self.graph_store = ThreadSafeGraphStore(
            max_memory_mb=self.config.max_memory_mb,
            enable_persistence=self.config.enable_persistence
        )
        self.error_handler = ErrorHandler()
        self.cache_manager = CacheManager(max_size_mb=self.config.cache_size_mb)
        
        # Plugin management
        self.plugins: Dict[str, GraphPlugin] = {}
        self.plugin_tools: Dict[str, Tuple[GraphPlugin, Tool]] = {}
        self.plugin_resources: Dict[str, Tuple[GraphPlugin, Resource]] = {}
        self.plugin_prompts: Dict[str, Tuple[GraphPlugin, Prompt]] = {}
        
        # Metrics
        self.request_count = 0
        self.error_count = 0
        self.plugin_execution_times: Dict[str, List[float]] = {}
        
        # Health monitoring
        self.is_healthy = True
        self.health_task: Optional[asyncio.Task] = None
        
        self._setup_handlers()
        logger.info(f"Initialized {self.config.name} v{self.config.version}")
    
    def _setup_handlers(self):
        """Set up MCP protocol handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> ListToolsResult:
            """List all available tools from plugins"""
            tools = []
            for plugin_name, plugin in self.plugins.items():
                try:
                    plugin_tools = await plugin.get_tools()
                    for tool in plugin_tools:
                        # Register tool with plugin mapping
                        self.plugin_tools[tool.name] = (plugin, tool)
                        tools.append(tool)
                except Exception as e:
                    logger.error(f"Error getting tools from {plugin_name}: {e}")
                    self.error_count += 1
            
            return ListToolsResult(tools=tools)
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> CallToolResult:
            """Route tool calls to appropriate plugin"""
            self.request_count += 1
            
            if name not in self.plugin_tools:
                error_msg = f"Unknown tool: {name}"
                logger.error(error_msg)
                self.error_count += 1
                return CallToolResult(
                    content=[TextContent(type="text", text=error_msg)],
                    isError=True
                )
            
            plugin, tool = self.plugin_tools[name]
            
            try:
                # Create plugin context
                context = PluginContext(
                    graph_store=self.graph_store,
                    cache_manager=self.cache_manager,
                    error_handler=self.error_handler,
                    config=self.config
                )
                
                # Execute plugin tool with error handling
                result = await self.error_handler.handle_with_recovery(
                    plugin.execute_tool,
                    name,
                    arguments,
                    context
                )
                
                # Track execution time
                if plugin.metadata.name not in self.plugin_execution_times:
                    self.plugin_execution_times[plugin.metadata.name] = []
                # Note: Would need timing logic here
                
                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(result))]
                )
                
            except GraphError as e:
                logger.error(f"Graph error in tool {name}: {e}")
                self.error_count += 1
                
                # Attempt recovery
                recovery_result = await self.error_handler.recover(e, {
                    "tool": name,
                    "arguments": arguments,
                    "plugin": plugin.metadata.name
                })
                
                if recovery_result:
                    return CallToolResult(
                        content=[TextContent(type="text", text=json.dumps(recovery_result))]
                    )
                
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Error: {e.message}. Suggested action: {e.suggested_action}"
                    )],
                    isError=True
                )
                
            except Exception as e:
                logger.error(f"Unexpected error in tool {name}: {e}")
                logger.error(traceback.format_exc())
                self.error_count += 1
                
                return CallToolResult(
                    content=[TextContent(
                        type="text",
                        text=f"Internal error: {str(e)}"
                    )],
                    isError=True
                )
        
        @self.server.list_resources()
        async def list_resources() -> ListResourcesResult:
            """List all available resources from plugins"""
            resources = []
            for plugin_name, plugin in self.plugins.items():
                try:
                    plugin_resources = await plugin.get_resources()
                    for resource in plugin_resources:
                        self.plugin_resources[resource.uri] = (plugin, resource)
                        resources.append(resource)
                except Exception as e:
                    logger.error(f"Error getting resources from {plugin_name}: {e}")
            
            return ListResourcesResult(resources=resources)
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> ReadResourceResult:
            """Read resource from appropriate plugin"""
            if uri not in self.plugin_resources:
                return ReadResourceResult(
                    contents=[TextContent(
                        type="text",
                        text=f"Resource not found: {uri}"
                    )]
                )
            
            plugin, resource = self.plugin_resources[uri]
            
            try:
                context = PluginContext(
                    graph_store=self.graph_store,
                    cache_manager=self.cache_manager,
                    error_handler=self.error_handler,
                    config=self.config
                )
                
                content = await plugin.read_resource(uri, context)
                return ReadResourceResult(contents=[content])
                
            except Exception as e:
                logger.error(f"Error reading resource {uri}: {e}")
                return ReadResourceResult(
                    contents=[TextContent(
                        type="text",
                        text=f"Error reading resource: {str(e)}"
                    )]
                )
        
        @self.server.list_prompts()
        async def list_prompts() -> ListPromptsResult:
            """List all available prompts from plugins"""
            prompts = []
            for plugin_name, plugin in self.plugins.items():
                try:
                    plugin_prompts = await plugin.get_prompts()
                    for prompt in plugin_prompts:
                        self.plugin_prompts[prompt.name] = (plugin, prompt)
                        prompts.append(prompt)
                except Exception as e:
                    logger.error(f"Error getting prompts from {plugin_name}: {e}")
            
            return ListPromptsResult(prompts=prompts)
        
        @self.server.get_prompt()
        async def get_prompt(name: str, arguments: Any) -> GetPromptResult:
            """Get prompt from appropriate plugin"""
            if name not in self.plugin_prompts:
                return GetPromptResult(
                    description=f"Unknown prompt: {name}",
                    messages=[]
                )
            
            plugin, prompt = self.plugin_prompts[name]
            
            try:
                context = PluginContext(
                    graph_store=self.graph_store,
                    cache_manager=self.cache_manager,
                    error_handler=self.error_handler,
                    config=self.config
                )
                
                result = await plugin.get_prompt(name, arguments, context)
                return result
                
            except Exception as e:
                logger.error(f"Error getting prompt {name}: {e}")
                return GetPromptResult(
                    description=f"Error: {str(e)}",
                    messages=[]
                )
    
    async def register_plugin(self, plugin_class: Type[GraphPlugin]) -> bool:
        """
        Register a plugin with the server.
        
        Args:
            plugin_class: Plugin class to instantiate and register
            
        Returns:
            True if registration successful
        """
        try:
            # Instantiate plugin
            plugin = plugin_class()
            
            # Check dependencies
            for dep in plugin.metadata.dependencies:
                if dep not in self.plugins:
                    logger.error(f"Plugin {plugin.metadata.name} requires {dep}")
                    return False
            
            # Initialize plugin
            context = PluginContext(
                graph_store=self.graph_store,
                cache_manager=self.cache_manager,
                error_handler=self.error_handler,
                config=self.config
            )
            
            success = await plugin.initialize(context)
            if not success:
                logger.error(f"Plugin {plugin.metadata.name} failed to initialize")
                return False
            
            # Register plugin
            self.plugins[plugin.metadata.name] = plugin
            logger.info(f"Registered plugin: {plugin.metadata.name} v{plugin.metadata.version}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error registering plugin {plugin_class.__name__}: {e}")
            return False
    
    async def unregister_plugin(self, plugin_name: str) -> bool:
        """Unregister a plugin"""
        if plugin_name not in self.plugins:
            return False
        
        try:
            plugin = self.plugins[plugin_name]
            await plugin.shutdown()
            del self.plugins[plugin_name]
            
            # Clean up tool/resource/prompt mappings
            self.plugin_tools = {
                k: v for k, v in self.plugin_tools.items()
                if v[0].metadata.name != plugin_name
            }
            self.plugin_resources = {
                k: v for k, v in self.plugin_resources.items()
                if v[0].metadata.name != plugin_name
            }
            self.plugin_prompts = {
                k: v for k, v in self.plugin_prompts.items()
                if v[0].metadata.name != plugin_name
            }
            
            logger.info(f"Unregistered plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unregistering plugin {plugin_name}: {e}")
            return False
    
    async def start_health_monitoring(self):
        """Start background health monitoring"""
        async def monitor_health():
            while True:
                try:
                    # Check graph store health
                    stats = self.graph_store.get_stats()
                    memory_usage = stats["memory_usage_mb"]
                    memory_limit = stats["memory_limit_mb"]
                    
                    # Check memory pressure
                    if memory_usage / memory_limit > 0.9:
                        logger.warning(f"High memory usage: {memory_usage:.1f}/{memory_limit:.1f}MB")
                        self.is_healthy = False
                    else:
                        self.is_healthy = True
                    
                    # Check error rate
                    if self.request_count > 0:
                        error_rate = self.error_count / self.request_count
                        if error_rate > 0.1:  # > 10% error rate
                            logger.warning(f"High error rate: {error_rate:.1%}")
                            self.is_healthy = False
                    
                    # Check plugin health
                    for plugin_name, plugin in self.plugins.items():
                        if not await plugin.health_check():
                            logger.warning(f"Plugin {plugin_name} is unhealthy")
                            self.is_healthy = False
                    
                    await asyncio.sleep(self.config.health_check_interval)
                    
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    await asyncio.sleep(self.config.health_check_interval)
        
        self.health_task = asyncio.create_task(monitor_health())
        logger.info("Started health monitoring")
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Starting graceful shutdown...")
        
        # Stop health monitoring
        if self.health_task:
            self.health_task.cancel()
        
        # Shutdown plugins
        for plugin_name, plugin in self.plugins.items():
            try:
                await plugin.shutdown()
                logger.info(f"Shut down plugin: {plugin_name}")
            except Exception as e:
                logger.error(f"Error shutting down plugin {plugin_name}: {e}")
        
        # Clear caches
        self.cache_manager.clear()
        
        logger.info("Shutdown complete")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics"""
        return {
            "server": {
                "name": self.config.name,
                "version": self.config.version,
                "is_healthy": self.is_healthy,
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / max(self.request_count, 1)
            },
            "plugins": {
                "count": len(self.plugins),
                "names": list(self.plugins.keys()),
                "execution_times": self.plugin_execution_times
            },
            "graph_store": self.graph_store.get_stats(),
            "cache": self.cache_manager.get_stats(),
            "error_handler": self.error_handler.get_stats()
        }
    
    async def run(self):
        """Run the MCP server"""
        logger.info(f"Starting {self.config.name}...")
        
        # Start health monitoring
        await self.start_health_monitoring()
        
        # Run server
        async with asyncio.TaskGroup() as group:
            group.create_task(self.server.run())
            
            # Log metrics periodically
            async def log_metrics():
                while True:
                    await asyncio.sleep(300)  # Every 5 minutes
                    metrics = self.get_metrics()
                    logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
            
            group.create_task(log_metrics())


def create_server(config: Optional[ServerConfig] = None) -> MCPGraphServer:
    """Factory function to create configured server"""
    return MCPGraphServer(config)


# Example minimal main entry point
async def main():
    """Main entry point"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create server with configuration
    config = ServerConfig(
        name="NetworkX MCP Server",
        version="2.0.0",
        max_memory_mb=1024,
        enable_gpu=True,
        enable_caching=True
    )
    
    server = create_server(config)
    
    # Register default plugins (would be loaded dynamically)
    # from ..plugins.core import CoreGraphPlugin
    # await server.register_plugin(CoreGraphPlugin)
    
    try:
        await server.run()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        await server.shutdown()
    except Exception as e:
        logger.error(f"Server error: {e}")
        await server.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())