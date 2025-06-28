"""Public interfaces for NetworkX MCP Server."""

from typing import Protocol, Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
import networkx as nx

class GraphAnalyzer(Protocol):
    """Interface for graph analysis tools."""
    
    async def analyze(self, graph_id: str, **params) -> Dict[str, Any]:
        """Analyze a graph with specific parameters."""
        ...

class Visualizer(Protocol):
    """Interface for visualization backends."""
    
    async def render(self, graph_id: str, layout: str, **options) -> str:
        """Render a graph visualization."""
        ...

class Storage(Protocol):
    """Interface for graph storage backends."""
    
    async def save_graph(self, user_id: str, graph_id: str, graph: nx.Graph, metadata: Optional[Dict] = None) -> bool:
        """Save a graph to storage."""
        ...
    
    async def load_graph(self, user_id: str, graph_id: str) -> Optional[nx.Graph]:
        """Load a graph from storage."""
        ...

class SecurityValidator(Protocol):
    """Interface for security validation."""
    
    def validate_input(self, data: Any) -> bool:
        """Validate user input for security."""
        ...
    
    def sanitize_output(self, data: Any) -> Any:
        """Sanitize output data."""
        ...

# Abstract base classes for implementation

class BaseGraphTool(ABC):
    """Base class for all graph tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    async def execute(self, graph: nx.Graph, **params) -> Dict[str, Any]:
        """Execute the tool on a graph."""
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Get parameter schema for this tool."""
        return {
            "type": "object",
            "properties": {},
            "required": []
        }

class BaseAnalyzer(BaseGraphTool):
    """Base class for graph analysis tools."""
    
    def __init__(self, name: str, description: str, analysis_type: str):
        super().__init__(name, description)
        self.analysis_type = analysis_type

class BaseVisualizer(BaseGraphTool):
    """Base class for visualization tools."""
    
    def __init__(self, name: str, description: str, output_format: str):
        super().__init__(name, description)
        self.output_format = output_format

# Tool registry interface
class ToolRegistry(Protocol):
    """Interface for tool registration and discovery."""
    
    def register_tool(self, tool: BaseGraphTool) -> None:
        """Register a new tool."""
        ...
    
    def get_tool(self, name: str) -> Optional[BaseGraphTool]:
        """Get a tool by name."""
        ...
    
    def list_tools(self) -> List[str]:
        """List all registered tools."""
        ...
