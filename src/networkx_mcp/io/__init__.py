"""Graph I/O operations.

This package provides readers and writers for various graph formats:
- GraphML (.graphml, .xml)

Example usage:
    from networkx_mcp.io import read_graphml, write_graphml
    
    graph = await read_graphml("data.graphml")
    await write_graphml(graph, "output.graphml")
"""

from .base import GraphReader, GraphWriter, validate_file_path, detect_format
from .graphml import GraphMLReader, GraphMLWriter, read_graphml, write_graphml

__all__ = [
    "GraphReader",
    "GraphWriter",
    "GraphMLReader", 
    "GraphMLWriter",
    "read_graphml",
    "write_graphml", 
    "validate_file_path",
    "detect_format"
]

# Factory functions
def get_reader(format_name: str):
    """Get reader for specified format."""
    readers = {
        "graphml": GraphMLReader
    }
    
    if format_name not in readers:
        raise ValueError(f"Unsupported format: {format_name}")
    
    return readers[format_name]()

def get_writer(format_name: str):
    """Get writer for specified format."""
    writers = {
        "graphml": GraphMLWriter
    }
    
    if format_name not in writers:
        raise ValueError(f"Unsupported format: {format_name}")
    
    return writers[format_name]()
