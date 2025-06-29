"""Graph I/O operations.

This package provides readers and writers for various graph formats:
- GraphML (.graphml, .xml)

Example usage:
    from networkx_mcp.io import read_graphml, write_graphml

    graph = await read_graphml("data.graphml")
    await write_graphml(graph, "output.graphml")
"""

from networkx_mcp.io.base import (
    GraphReader,
    GraphWriter,
    detect_format,
    validate_file_path,
)
from networkx_mcp.io.graphml import (
    GraphMLReader,
    GraphMLWriter,
    read_graphml,
    write_graphml,
)

__all__ = [
    "GraphMLReader",
    "GraphMLWriter",
    "GraphReader",
    "GraphWriter",
    "detect_format",
    "read_graphml",
    "validate_file_path",
    "write_graphml"
]

# Factory functions
def get_reader(format_name: str):
    """Get reader for specified format."""
    readers = {
        "graphml": GraphMLReader
    }

    if format_name not in readers:
        msg = f"Unsupported format: {format_name}"
        raise ValueError(msg)

    return readers[format_name]()

def get_writer(format_name: str):
    """Get writer for specified format."""
    writers = {
        "graphml": GraphMLWriter
    }

    if format_name not in writers:
        msg = f"Unsupported format: {format_name}"
        raise ValueError(msg)

    return writers[format_name]()
