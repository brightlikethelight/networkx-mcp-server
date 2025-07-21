"""GraphML format I/O operations."""

from pathlib import Path
from typing import Any

import networkx as nx

from networkx_mcp.io.base import GraphReader, GraphWriter, validate_file_path


class GraphMLReader(GraphReader):
    """Read graphs from GraphML format."""

    def __init__(self) -> None:
        super().__init__("graphml", [".graphml", ".xml"])

    async def read(self, filepath: str | Path, **options: Any) -> nx.Graph:
        """Read GraphML file."""
        path = validate_file_path(filepath, must_exist=True)

        try:
            # Use NetworkX's GraphML reader
            graph = nx.read_graphml(str(path))

            # Add metadata about file
            graph.graph["source_file"] = str(path)
            graph.graph["format"] = "graphml"

            return graph

        except Exception as e:
            msg = f"Failed to read GraphML file {path}: {e}"
            raise RuntimeError(msg) from e


class GraphMLWriter(GraphWriter):
    """Write graphs to GraphML format."""

    def __init__(self) -> None:
        super().__init__("graphml", ".graphml")

    async def write(self, graph: nx.Graph, filepath: str | Path, **options: Any) -> bool:
        """Write graph to GraphML file."""
        path = validate_file_path(filepath, must_exist=False)

        # Ensure .graphml extension
        if not path.suffix:
            path = path.with_suffix(".graphml")

        try:
            # Use NetworkX's GraphML writer
            nx.write_graphml(graph, str(path))
            return True

        except Exception as e:
            msg = f"Failed to write GraphML file {path}: {e}"
            raise RuntimeError(msg) from e


async def read_graphml(filepath: str | Path) -> nx.Graph:
    """Simple function interface for reading GraphML."""
    reader = GraphMLReader()
    return await reader.read(filepath)


async def write_graphml(graph: nx.Graph, filepath: str | Path) -> bool:
    """Simple function interface for writing GraphML."""
    writer = GraphMLWriter()
    return await writer.write(graph, filepath)
