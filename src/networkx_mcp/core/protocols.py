"""Protocol definitions for NetworkX MCP Server.

This module defines protocols (structural subtyping) for core interfaces,
enabling loose coupling and easier testing through dependency injection.
"""

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

import networkx as nx

# Type aliases for graph operations
NodeId = Union[str, int]
EdgeTuple = Tuple[NodeId, NodeId]
EdgeWithAttrs = Tuple[NodeId, NodeId, Dict[str, Any]]
GraphType = Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
AttributeValue = Union[str, int, float, bool, None, List[Any], Dict[str, Any]]

T = TypeVar("T")


@runtime_checkable
class GraphStoreProtocol(Protocol):
    """Protocol for graph storage implementations.

    This protocol defines the interface for storing and retrieving graphs,
    allowing different backends (memory, file, Redis) to be used interchangeably.
    """

    def get(self, key: str) -> Optional[nx.Graph]:
        """Get a graph by key.

        Args:
            key: Graph identifier

        Returns:
            Graph if found, None otherwise
        """
        ...

    def put(self, key: str, graph: nx.Graph) -> None:
        """Store a graph.

        Args:
            key: Graph identifier
            graph: NetworkX graph to store
        """
        ...

    def delete(self, key: str) -> bool:
        """Delete a graph.

        Args:
            key: Graph identifier

        Returns:
            True if deleted, False if not found
        """
        ...

    def list_graphs(self) -> List[str]:
        """List all graph keys.

        Returns:
            List of graph identifiers
        """
        ...

    def clear(self) -> None:
        """Clear all stored graphs."""
        ...

    def __contains__(self, key: object) -> bool:
        """Check if graph exists."""
        ...

    def __getitem__(self, key: str) -> nx.Graph:
        """Get graph by key (raises KeyError if not found)."""
        ...

    def __setitem__(self, key: str, graph: nx.Graph) -> None:
        """Store graph."""
        ...


@runtime_checkable
class CacheProtocol(Protocol[T]):
    """Protocol for cache implementations.

    Generic cache interface that can be used for various data types.
    """

    def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        ...

    def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        ...

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        ...

    def clear(self) -> None:
        """Clear all cache entries."""
        ...


@runtime_checkable
class GraphCacheProtocol(Protocol):
    """Protocol specifically for graph caching with statistics.

    Extends basic cache with graph-specific metadata and stats.
    """

    def get(self, key: str) -> Optional[nx.Graph]:
        """Get graph from cache."""
        ...

    def put(self, key: str, graph: nx.Graph) -> None:
        """Put graph in cache."""
        ...

    def delete(self, key: str) -> bool:
        """Delete graph from cache."""
        ...

    def list_graphs(self) -> List[str]:
        """List all cached graph keys."""
        ...

    def clear(self) -> None:
        """Clear cache."""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        ...

    def shutdown(self) -> None:
        """Shutdown cache and cleanup resources."""
        ...


@runtime_checkable
class GraphOperationsProtocol(Protocol):
    """Protocol for graph operations.

    Defines the interface for performing operations on graphs.
    """

    def create_graph(
        self,
        graph_id: str,
        directed: bool = False,
        multigraph: bool = False,
    ) -> nx.Graph:
        """Create a new graph."""
        ...

    def delete_graph(self, graph_id: str) -> bool:
        """Delete a graph."""
        ...

    def get_graph(self, graph_id: str) -> Optional[nx.Graph]:
        """Get a graph by ID."""
        ...

    def list_graphs(self) -> List[str]:
        """List all graph IDs."""
        ...

    def add_node(
        self,
        graph_id: str,
        node_id: NodeId,
        attributes: Optional[Dict[str, AttributeValue]] = None,
    ) -> None:
        """Add a node to a graph."""
        ...

    def add_edge(
        self,
        graph_id: str,
        source: NodeId,
        target: NodeId,
        attributes: Optional[Dict[str, AttributeValue]] = None,
    ) -> None:
        """Add an edge to a graph."""
        ...

    def get_neighbors(self, graph_id: str, node_id: NodeId) -> List[NodeId]:
        """Get neighbors of a node."""
        ...


@runtime_checkable
class AlgorithmServiceProtocol(Protocol):
    """Protocol for graph algorithm services."""

    def shortest_path(
        self,
        graph_id: str,
        source: NodeId,
        target: NodeId,
        weight: Optional[str] = None,
    ) -> List[NodeId]:
        """Find shortest path between nodes."""
        ...

    def centrality(
        self,
        graph_id: str,
        measure: str = "degree",
    ) -> Dict[NodeId, float]:
        """Calculate centrality measures."""
        ...

    def connected_components(self, graph_id: str) -> List[List[NodeId]]:
        """Find connected components."""
        ...

    def clustering_coefficient(
        self,
        graph_id: str,
        node_id: Optional[NodeId] = None,
    ) -> Union[float, Dict[NodeId, float]]:
        """Calculate clustering coefficient."""
        ...


@runtime_checkable
class RequestHandlerProtocol(Protocol):
    """Protocol for MCP request handlers."""

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an incoming request.

        Args:
            request: JSON-RPC request dictionary

        Returns:
            JSON-RPC response dictionary
        """
        ...


@runtime_checkable
class ToolExecutorProtocol(Protocol):
    """Protocol for tool execution."""

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a tool with given arguments.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        ...

    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools with their schemas."""
        ...


@runtime_checkable
class AuthenticationProtocol(Protocol):
    """Protocol for authentication services."""

    def validate_api_key(self, api_key: str) -> bool:
        """Validate an API key."""
        ...

    def generate_api_key(self) -> str:
        """Generate a new API key."""
        ...

    def is_authentication_required(self) -> bool:
        """Check if authentication is required."""
        ...


@runtime_checkable
class RateLimiterProtocol(Protocol):
    """Protocol for rate limiting."""

    def check_rate_limit(self, client_id: str) -> bool:
        """Check if request is within rate limit.

        Args:
            client_id: Client identifier

        Returns:
            True if within limit, False if exceeded
        """
        ...

    def record_request(self, client_id: str) -> None:
        """Record a request for rate limiting."""
        ...

    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client."""
        ...


@runtime_checkable
class IOHandlerProtocol(Protocol):
    """Protocol for I/O format handlers."""

    format_name: str
    file_extensions: List[str]

    def export_graph(
        self,
        graph: nx.Graph,
        path: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Export graph to this format."""
        ...

    def import_graph(
        self,
        data: Optional[Any] = None,
        path: Optional[str] = None,
        **kwargs: Any,
    ) -> nx.Graph:
        """Import graph from this format."""
        ...


@runtime_checkable
class VisualizationProtocol(Protocol):
    """Protocol for graph visualization."""

    def render(
        self,
        graph: nx.Graph,
        output_path: Optional[str] = None,
        layout: str = "spring",
        **kwargs: Any,
    ) -> Any:
        """Render graph visualization.

        Args:
            graph: Graph to visualize
            output_path: Optional path to save output
            layout: Layout algorithm to use
            **kwargs: Additional rendering options

        Returns:
            Rendered visualization (format depends on implementation)
        """
        ...


@runtime_checkable
class MetricsCollectorProtocol(Protocol):
    """Protocol for metrics collection."""

    def record_metric(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a metric value."""
        ...

    def increment_counter(
        self,
        name: str,
        amount: int = 1,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter metric."""
        ...

    def record_timing(
        self,
        name: str,
        duration_ms: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a timing metric."""
        ...


@runtime_checkable
class HealthCheckProtocol(Protocol):
    """Protocol for health check implementations."""

    async def check_health(self) -> Dict[str, Any]:
        """Perform health check.

        Returns:
            Health status dictionary with 'healthy' boolean and details
        """
        ...


# Type guards for runtime checking
def is_graph_store(obj: Any) -> bool:
    """Check if object implements GraphStoreProtocol."""
    return isinstance(obj, GraphStoreProtocol)


def is_cache(obj: Any) -> bool:
    """Check if object implements CacheProtocol."""
    return isinstance(obj, CacheProtocol)


def is_request_handler(obj: Any) -> bool:
    """Check if object implements RequestHandlerProtocol."""
    return isinstance(obj, RequestHandlerProtocol)
