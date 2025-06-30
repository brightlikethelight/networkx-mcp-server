"""Graph I/O handlers for various formats with comprehensive format support."""

import csv
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional, TextIO, Union

import networkx as nx
import numpy as np
import pandas as pd
import yaml

# Optional imports - not all may be available
try:
    from scipy.sparse import coo_matrix

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    coo_matrix = None

try:
    from networkx.drawing.nx_agraph import to_agraph

    HAS_AGRAPH = True
except ImportError:
    HAS_AGRAPH = False
    to_agraph = None


logger = logging.getLogger(__name__)


class GraphIOHandler:
    """Handles graph import/export in various formats with auto-detection."""

    # Supported formats with file extensions
    FORMAT_EXTENSIONS = {
        "graphml": [".graphml", ".xml"],
        "gexf": [".gexf"],
        "json": [".json"],
        "yaml": [".yaml", ".yml"],
        "csv": [".csv"],
        "edgelist": [".edges", ".edgelist", ".txt"],
        "adjacency": [".adj", ".mat"],
        "pickle": [".pickle", ".pkl", ".p"],
        "pajek": [".net", ".pajek"],
        "dot": [".dot", ".gv"],
    }

    @staticmethod
    def detect_format(filepath: Union[str, Path]) -> str:
        """Auto-detect file format from extension."""
        path = Path(filepath)
        ext = path.suffix.lower()

        for fmt_name, extensions in GraphIOHandler.FORMAT_EXTENSIONS.items():
            if ext in extensions:
                logger.info(f"Auto-detected format '{fmt_name}' from extension '{ext}'")
                return fmt_name

        # Default fallback based on common patterns
        if "edge" in path.stem.lower():
            return "edgelist"
        elif "adj" in path.stem.lower():
            return "adjacency"

        msg = f"Cannot auto-detect format for file: {filepath}"
        raise ValueError(msg)

    @staticmethod
    def export_graph(
        graph: nx.Graph,
        output_format: str,
        path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Union[str, bytes, dict[str, Any]]:
        """Export graph to various formats with validation."""
        logger.info(f"Exporting graph to {output_format} format")

        # Validate format
        output_format = output_format.lower()
        valid_formats = [
            "json",
            "graphml",
            "gexf",
            "edgelist",
            "adjacency",
            "pickle",
            "dot",
            "pajek",
            "yaml",
            "csv",
        ]

        if output_format not in valid_formats:
            msg = f"Unsupported export format: {output_format}. Valid formats: {valid_formats}"
            raise ValueError(msg)

        # Handle pretty printing option
        pretty_print = kwargs.pop("pretty_print", True)

        if output_format == "json":
            return GraphIOHandler._export_json(
                graph, pretty_print=pretty_print, **kwargs
            )
        elif output_format == "yaml":
            return GraphIOHandler._export_yaml(graph, path, **kwargs)
        elif output_format == "csv":
            return GraphIOHandler._export_csv(graph, path, **kwargs)
        elif output_format == "graphml":
            return GraphIOHandler._export_graphml(graph, path, **kwargs)
        elif output_format == "gexf":
            return GraphIOHandler._export_gexf(graph, path, **kwargs)
        elif output_format == "edgelist":
            return GraphIOHandler._export_edgelist(graph, path, **kwargs)
        elif output_format == "adjacency":
            return GraphIOHandler._export_adjacency(graph, **kwargs)
        elif output_format == "pickle":
            return GraphIOHandler._export_pickle(graph, path)
        elif output_format == "dot":
            return GraphIOHandler._export_dot(graph, **kwargs)
        elif output_format == "pajek":
            return GraphIOHandler._export_pajek(graph, path)
        else:
            msg = f"Unsupported export format: {output_format}"
            raise ValueError(msg)

    @staticmethod
    def import_graph(
        input_format: Optional[str] = None,
        data: Optional[Union[str, bytes, dict[str, Any]]] = None,
        path: Optional[Union[str, Path]] = None,
        auto_detect: bool = True,
        **kwargs,
    ) -> nx.Graph:
        """Import graph from various formats with auto-detection."""
        # Auto-detect format if not provided
        if input_format is None and path and auto_detect:
            input_format = GraphIOHandler.detect_format(path)

        if input_format is None:
            msg = "Format must be specified or auto-detected from filepath"
            raise ValueError(msg)

        logger.info(f"Importing graph from {input_format} format")

        # Validate format
        input_format = input_format.lower()
        valid_formats = [
            "json",
            "graphml",
            "gexf",
            "edgelist",
            "adjacency",
            "pickle",
            "pajek",
            "yaml",
            "csv",
        ]

        if input_format not in valid_formats:
            msg = f"Unsupported import format: {input_format}. Valid formats: {valid_formats}"
            raise ValueError(msg)

        # Validate file existence
        if path and not data:
            path = Path(path)
            if not path.exists():
                msg = f"File not found: {path}"
                raise FileNotFoundError(msg)
            if not path.is_file():
                msg = f"Not a file: {path}"
                raise ValueError(msg)
            if not path.stat().st_size:
                msg = f"File is empty: {path}"
                raise ValueError(msg)

        if input_format == "json":
            return GraphIOHandler._import_json(data, path, **kwargs)
        elif input_format == "yaml":
            return GraphIOHandler._import_yaml(data, path, **kwargs)
        elif input_format == "csv":
            return GraphIOHandler._import_csv(path, **kwargs)
        elif input_format == "graphml":
            return GraphIOHandler._import_graphml(path)
        elif input_format == "gexf":
            return GraphIOHandler._import_gexf(path)
        elif input_format == "edgelist":
            return GraphIOHandler._import_edgelist(path, **kwargs)
        elif input_format == "adjacency":
            return GraphIOHandler._import_adjacency(data, path, **kwargs)
        elif input_format == "pickle":
            return GraphIOHandler._import_pickle(path)
        elif input_format == "pajek":
            return GraphIOHandler._import_pajek(path)
        else:
            msg = f"Unsupported import format: {input_format}"
            raise ValueError(msg)

    @staticmethod
    def _export_json(
        graph: nx.Graph, pretty_print: bool = True, **kwargs
    ) -> dict[str, Any]:
        """Export graph to JSON format with metadata preservation."""
        data = nx.node_link_data(graph)

        # Add comprehensive metadata
        data["graph"]["directed"] = graph.is_directed()
        data["graph"]["multigraph"] = graph.is_multigraph()
        data["graph"]["graph_type"] = type(graph).__name__
        data["graph"]["num_nodes"] = graph.number_of_nodes()
        data["graph"]["num_edges"] = graph.number_of_edges()

        # Include graph attributes
        if graph.graph:
            data["graph"]["attributes"] = graph.graph

        if pretty_print:
            # Return as formatted string for pretty printing
            return json.dumps(data, indent=2, sort_keys=True, default=str)

        return data

    @staticmethod
    def _import_json(
        data: Optional[dict[str, Any]], path: Optional[Union[str, Path]], **kwargs
    ) -> nx.Graph:
        """Import graph from JSON format."""
        if data is None and path:
            with open(path) as f:
                data = json.load(f)
        elif isinstance(data, str):
            data = json.loads(data)

        if not isinstance(data, dict):
            msg = "JSON data must be a dictionary"
            raise ValueError(msg)

        return nx.node_link_graph(data)

    @staticmethod
    def _export_yaml(graph: nx.Graph, path: Union[str, Path], **kwargs) -> str:
        """Export graph to YAML format."""
        data = GraphIOHandler._export_json(graph, pretty_print=False)

        if path:
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=True)
            return f"Graph exported to YAML: {path}"
        else:
            return yaml.dump(data, default_flow_style=False, sort_keys=True)

    @staticmethod
    def _import_yaml(
        data: Optional[str], path: Optional[Union[str, Path]], **kwargs
    ) -> nx.Graph:
        """Import graph from YAML format."""
        if data is None and path:
            with open(path) as f:
                data = yaml.safe_load(f)
        elif isinstance(data, str):
            data = yaml.safe_load(data)

        return nx.node_link_graph(data)

    @staticmethod
    def _export_csv(graph: nx.Graph, path: Union[str, Path], **kwargs) -> str:
        """Export graph to CSV edge list format."""
        if path is None:
            msg = "Path required for CSV export"
            raise ValueError(msg)

        # Get column names from kwargs
        source_col = kwargs.get("source_col", "source")
        target_col = kwargs.get("target_col", "target")
        weight_col = kwargs.get("weight_col", "weight")
        include_attrs = kwargs.get("include_attrs", True)

        # Prepare data for CSV
        rows = []
        headers = [source_col, target_col]

        # Check if graph has weights
        has_weights = any("weight" in data for _, _, data in graph.edges(data=True))
        if has_weights:
            headers.append(weight_col)

        # Collect all edge attributes
        if include_attrs:
            all_attrs = set()
            for _, _, data in graph.edges(data=True):
                all_attrs.update(data.keys())
            all_attrs.discard("weight")  # Already handled
            headers.extend(sorted(all_attrs))

        # Build rows
        for u, v, data in graph.edges(data=True):
            row = [u, v]
            if has_weights:
                row.append(data.get("weight", 1.0))
            if include_attrs:
                for attr in sorted(all_attrs):
                    row.append(data.get(attr, ""))
            rows.append(row)

        # Write CSV
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

        return f"Graph exported to CSV: {path} ({len(rows)} edges)"

    @staticmethod
    def _import_csv(path: Union[str, Path], **kwargs) -> nx.Graph:
        """Import graph from CSV edge list."""
        source_col = kwargs.get("source_col", "source")
        target_col = kwargs.get("target_col", "target")
        weight_col = kwargs.get("weight_col", "weight")
        create_using = kwargs.get("create_using", nx.Graph())

        df = pd.read_csv(path)

        # Validate required columns
        if source_col not in df.columns or target_col not in df.columns:
            msg = f"CSV must contain '{source_col}' and '{target_col}' columns"
            raise ValueError(msg)

        # Create graph
        if weight_col and weight_col in df.columns:
            # Weighted graph
            G = nx.from_pandas_edgelist(
                df,
                source=source_col,
                target=target_col,
                edge_attr=weight_col,
                create_using=create_using,
            )
        else:
            # Unweighted graph
            G = nx.from_pandas_edgelist(
                df, source=source_col, target=target_col, create_using=create_using
            )

        # Add any additional edge attributes
        edge_attrs = [
            col for col in df.columns if col not in [source_col, target_col, weight_col]
        ]
        if edge_attrs:
            for _, row in df.iterrows():
                attrs = {attr: row[attr] for attr in edge_attrs if pd.notna(row[attr])}
                if attrs:
                    G[row[source_col]][row[target_col]].update(attrs)

        return G

    @staticmethod
    def csv_to_edge_list(
        filepath: Union[str, Path],
        source_col: str,
        target_col: str,
        weight_col: Optional[str] = None,
        delimiter: str = ",",
        **kwargs,
    ) -> list[Union[tuple[Any, Any], tuple[Any, Any, float]]]:
        """Convert CSV file to edge list format.

        Args:
            filepath: Path to CSV file
            source_col: Column name for source nodes
            target_col: Column name for target nodes
            weight_col: Optional column name for edge weights
            delimiter: CSV delimiter
            **kwargs: Additional pandas read_csv arguments

        Returns:
            List of edges as tuples (source, target) or (source, target, weight)
        """
        logger.info(f"Converting CSV to edge list: {filepath}")

        # Read CSV with pandas
        df = pd.read_csv(filepath, delimiter=delimiter, **kwargs)

        # Validate columns
        if source_col not in df.columns:
            msg = f"Source column '{source_col}' not found in CSV"
            raise ValueError(msg)
        if target_col not in df.columns:
            msg = f"Target column '{target_col}' not found in CSV"
            raise ValueError(msg)
        if weight_col and weight_col not in df.columns:
            msg = f"Weight column '{weight_col}' not found in CSV"
            raise ValueError(msg)

        # Build edge list
        edges = []
        for _, row in df.iterrows():
            source = row[source_col]
            target = row[target_col]

            if pd.isna(source) or pd.isna(target):
                logger.warning(f"Skipping edge with NaN values: {source} -> {target}")
                continue

            if weight_col:
                weight = row[weight_col]
                if pd.isna(weight):
                    weight = 1.0
                edges.append((source, target, float(weight)))
            else:
                edges.append((source, target))

        logger.info(f"Converted {len(edges)} edges from CSV")
        return edges

    @staticmethod
    def dataframe_to_graph(
        df: pd.DataFrame,
        source_col: str,
        target_col: str,
        edge_attr: Optional[Union[str, list[str]]] = None,
        create_using: Optional[nx.Graph] = None,
        node_attr_df: Optional[pd.DataFrame] = None,
        node_key: Optional[str] = None,
    ) -> nx.Graph:
        """Convert pandas DataFrame to NetworkX graph.

        Args:
            df: DataFrame containing edge data
            source_col: Column name for source nodes
            target_col: Column name for target nodes
            edge_attr: Column name(s) for edge attributes
            create_using: Graph type to create
            node_attr_df: Optional DataFrame with node attributes
            node_key: Column in node_attr_df to use as node ID

        Returns:
            NetworkX graph
        """
        logger.info(f"Converting DataFrame to graph ({len(df)} rows)")

        # Create graph from edge list
        if isinstance(edge_attr, str):
            edge_attr = [edge_attr]

        G = nx.from_pandas_edgelist(
            df,
            source=source_col,
            target=target_col,
            edge_attr=edge_attr,
            create_using=create_using,
        )

        # Add node attributes if provided
        if node_attr_df is not None and node_key is not None:
            if node_key not in node_attr_df.columns:
                msg = f"Node key column '{node_key}' not found in node attributes DataFrame"
                raise ValueError(msg)

            for _, row in node_attr_df.iterrows():
                node_id = row[node_key]
                if node_id in G:
                    attrs = {
                        k: v for k, v in row.items() if k != node_key and pd.notna(v)
                    }
                    G.nodes[node_id].update(attrs)
                else:
                    logger.warning(
                        f"Node '{node_id}' from attributes not found in graph"
                    )

        logger.info(
            f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
        )
        return G

    @staticmethod
    def adjacency_to_edge_list(
        matrix: Union[list[list[float]], np.ndarray],
        node_labels: Optional[list[Any]] = None,
        threshold: float = 0,
        directed: bool = False,
    ) -> list[tuple[Any, Any, float]]:
        """Convert adjacency matrix to edge list.

        Args:
            matrix: 2D adjacency matrix
            node_labels: Optional node labels (defaults to indices)
            threshold: Minimum edge weight to include
            directed: Whether to treat as directed graph

        Returns:
            List of edges as (source, target, weight) tuples
        """
        # Convert to numpy array
        if isinstance(matrix, list):
            matrix = np.array(matrix)

        n = matrix.shape[0]
        if matrix.shape != (n, n):
            msg = f"Matrix must be square, got shape {matrix.shape}"
            raise ValueError(msg)

        # Default node labels
        if node_labels is None:
            node_labels = list(range(n))
        elif len(node_labels) != n:
            msg = (
                f"Node labels length ({len(node_labels)}) must match matrix size ({n})"
            )
            raise ValueError(msg)

        edges = []

        # Extract edges
        for i in range(n):
            start_j = 0 if directed else i + 1  # Skip lower triangle for undirected
            for j in range(start_j, n):
                if i != j or directed:  # Skip self-loops unless directed
                    weight = matrix[i, j]
                    if abs(weight) > threshold:
                        edges.append((node_labels[i], node_labels[j], float(weight)))

        logger.info(f"Converted {n}x{n} adjacency matrix to {len(edges)} edges")
        return edges

    @staticmethod
    def export_for_streaming(
        graph: nx.Graph,
        output_format: str,
        output_stream: TextIO,
        chunk_size: int = 1000,
        **kwargs,
    ) -> int:
        """Export large graphs using streaming to handle memory constraints.

        Args:
            graph: Graph to export
            output_format: Export format (currently supports 'edgelist', 'csv')
            output_stream: Output stream to write to
            chunk_size: Number of edges to process at a time
            **kwargs: Format-specific options

        Returns:
            Number of edges exported
        """
        logger.info(f"Starting streaming export in {output_format} format")

        if output_format == "edgelist":
            # Write header if needed
            if kwargs.get("header", True):
                output_stream.write("# NetworkX edge list\n")
                output_stream.write(f"# Nodes: {graph.number_of_nodes()}\n")
                output_stream.write(f"# Edges: {graph.number_of_edges()}\n")

            # Stream edges in chunks
            edge_count = 0
            edge_iter = graph.edges(data=True)

            while True:
                chunk = []
                for _ in range(chunk_size):
                    try:
                        edge = next(edge_iter)
                        chunk.append(edge)
                    except StopIteration:
                        break

                if not chunk:
                    break

                # Write chunk
                for u, v, data in chunk:
                    if data:
                        attrs_str = " ".join(
                            f"{k}={v}" for k, v in sorted(data.items())
                        )
                        output_stream.write(f"{u} {v} {attrs_str}\n")
                    else:
                        output_stream.write(f"{u} {v}\n")

                edge_count += len(chunk)

                if edge_count % 10000 == 0:
                    logger.info(f"Exported {edge_count} edges...")

            logger.info(f"Streaming export complete: {edge_count} edges")
            return edge_count

        elif output_format == "csv":
            # CSV writer
            writer = csv.writer(output_stream)

            # Write header
            headers = ["source", "target"]
            # Collect all edge attributes
            all_attrs = set()
            sample_size = min(100, graph.number_of_edges())
            for i, (_, _, data) in enumerate(graph.edges(data=True)):
                if i >= sample_size:
                    break
                all_attrs.update(data.keys())

            headers.extend(sorted(all_attrs))
            writer.writerow(headers)

            # Stream edges
            edge_count = 0
            edge_iter = graph.edges(data=True)

            while True:
                chunk = []
                for _ in range(chunk_size):
                    try:
                        edge = next(edge_iter)
                        chunk.append(edge)
                    except StopIteration:
                        break

                if not chunk:
                    break

                # Write chunk
                for u, v, data in chunk:
                    row = [u, v]
                    for attr in sorted(all_attrs):
                        row.append(data.get(attr, ""))
                    writer.writerow(row)

                edge_count += len(chunk)

            return edge_count

        else:
            msg = f"Streaming not supported for format: {output_format}"
            raise ValueError(msg)

    @staticmethod
    def _export_graphml(graph: nx.Graph, path: Union[str, Path], **kwargs) -> str:
        """Export graph to GraphML format."""
        if path is None:
            msg = "Path required for GraphML export"
            raise ValueError(msg)

        nx.write_graphml(graph, path, **kwargs)
        return f"Graph exported to {path}"

    @staticmethod
    def _import_graphml(path: Union[str, Path]) -> nx.Graph:
        """Import graph from GraphML format."""
        if path is None:
            msg = "Path required for GraphML import"
            raise ValueError(msg)

        return nx.read_graphml(path)

    @staticmethod
    def _export_gexf(graph: nx.Graph, path: Union[str, Path], **kwargs) -> str:
        """Export graph to GEXF format."""
        if path is None:
            msg = "Path required for GEXF export"
            raise ValueError(msg)

        nx.write_gexf(graph, path, **kwargs)
        return f"Graph exported to {path}"

    @staticmethod
    def _import_gexf(path: Union[str, Path]) -> nx.Graph:
        """Import graph from GEXF format."""
        if path is None:
            msg = "Path required for GEXF import"
            raise ValueError(msg)

        return nx.read_gexf(path)

    @staticmethod
    def _export_edgelist(
        graph: nx.Graph, path: Optional[Union[str, Path]], **kwargs
    ) -> Union[str, list[dict]]:
        """Export graph to edge list format."""
        if path:
            # Handle large graphs with streaming
            if graph.number_of_edges() > 100000:  # Large graph threshold
                logger.info(
                    f"Large graph detected ({graph.number_of_edges()} edges), using streaming"
                )
                with open(path, "w") as f:
                    edge_count = GraphIOHandler.export_for_streaming(
                        graph, "edgelist", f, **kwargs
                    )
                return f"Graph exported to {path} ({edge_count} edges using streaming)"
            else:
                nx.write_edgelist(graph, path, **kwargs)
                return f"Graph exported to {path}"
        else:
            # Return as list
            edges = []
            for u, v, data in graph.edges(data=True):
                edge = {"source": str(u), "target": str(v)}
                edge.update(data)
                edges.append(edge)
            return edges

    @staticmethod
    def _import_edgelist(path: Union[str, Path], **kwargs) -> nx.Graph:
        """Import graph from edge list format."""
        if path is None:
            msg = "Path required for edge list import"
            raise ValueError(msg)

        # Determine graph type from kwargs
        create_using = kwargs.get("create_using", nx.Graph())

        return nx.read_edgelist(path, create_using=create_using, **kwargs)

    @staticmethod
    def _export_adjacency(graph: nx.Graph, **kwargs) -> dict[str, Any]:
        """Export graph as adjacency matrix with metadata."""
        nodes = list(graph.nodes())

        # Handle node ordering
        if "node_order" in kwargs:
            node_order = kwargs["node_order"]
            if set(node_order) != set(nodes):
                msg = "node_order must contain all graph nodes"
                raise ValueError(msg)
            nodes = node_order

        # Export with specific dtype for memory efficiency
        dtype = kwargs.get("dtype", np.float64)
        matrix = nx.to_numpy_array(graph, nodelist=nodes, dtype=dtype)

        # Handle sparse matrices for large graphs
        is_sparse = graph.number_of_edges() < 0.1 * (graph.number_of_nodes() ** 2)

        result = {
            "nodes": [str(n) for n in nodes],  # Ensure serializable
            "shape": matrix.shape,
            "directed": graph.is_directed(),
            "weighted": any("weight" in data for _, _, data in graph.edges(data=True)),
            "density": nx.density(graph),
            "is_sparse": is_sparse,
        }

        if is_sparse and kwargs.get("sparse_format", False):
            # Return sparse representation
            if not HAS_SCIPY:
                logger.warning("scipy not available, returning dense matrix")
                result["format"] = "dense"
                result["matrix"] = matrix.tolist()
            else:
                sparse = coo_matrix(matrix)
            result["format"] = "sparse_coo"
            result["row"] = sparse.row.tolist()
            result["col"] = sparse.col.tolist()
            result["data"] = sparse.data.tolist()
        else:
            # Return dense matrix
            result["format"] = "dense"
            result["matrix"] = matrix.tolist()

        return result

    @staticmethod
    def _import_adjacency(
        data: Optional[dict[str, Any]], path: Optional[Union[str, Path]], **kwargs
    ) -> nx.Graph:
        """Import graph from adjacency matrix."""
        if data is None and path:
            # Load from file
            if path.suffix.lower() == ".npy":
                matrix = np.load(path)
                nodes = kwargs.get("nodes", list(range(matrix.shape[0])))
                directed = kwargs.get("directed", False)
            else:
                with open(path) as f:
                    data = json.load(f)

        if data is None:
            msg = "Either data or path must be provided"
            raise ValueError(msg)

        # Handle sparse format
        if data.get("format") == "sparse_coo":
            if not HAS_SCIPY:
                msg = "scipy required for sparse matrix import"
                raise ImportError(msg)
            n = data["shape"][0]
            matrix = coo_matrix(
                (data["data"], (data["row"], data["col"])), shape=(n, n)
            ).todense()
        else:
            matrix = np.array(data["matrix"])

        nodes = data.get("nodes", list(range(matrix.shape[0])))
        directed = data.get("directed", False)

        # Create appropriate graph type
        create_using = kwargs.get("create_using")
        if create_using is None:
            create_using = nx.DiGraph() if directed else nx.Graph()

        graph = nx.from_numpy_array(matrix, create_using=create_using)

        # Relabel nodes if custom node labels provided
        if nodes and len(nodes) == graph.number_of_nodes():
            mapping = {i: nodes[i] for i in range(len(nodes))}
            graph = nx.relabel_nodes(graph, mapping)

        return graph

    @staticmethod
    def _export_pickle(graph: nx.Graph, path: Union[str, Path]) -> str:
        """Export graph to pickle format."""
        if path is None:
            msg = "Path required for pickle export"
            raise ValueError(msg)

        with open(path, "wb") as f:
            pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)

        return f"Graph exported to {path}"

    @staticmethod
    def _import_pickle(path: Union[str, Path]) -> nx.Graph:
        """Import graph from pickle format."""
        if path is None:
            msg = "Path required for pickle import"
            raise ValueError(msg)

        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _export_dot(graph: nx.Graph, **kwargs) -> str:
        """Export graph to DOT format."""
        if not HAS_AGRAPH:
            msg = "pygraphviz required for DOT format export"
            raise ImportError(msg)

        A = to_agraph(graph)
        return A.to_string()

    @staticmethod
    def _export_pajek(graph: nx.Graph, path: Union[str, Path]) -> str:
        """Export graph to Pajek format."""
        if path is None:
            msg = "Path required for Pajek export"
            raise ValueError(msg)

        nx.write_pajek(graph, path)
        return f"Graph exported to {path}"

    @staticmethod
    def _import_pajek(path: Union[str, Path]) -> nx.Graph:
        """Import graph from Pajek format."""
        if path is None:
            msg = "Path required for Pajek import"
            raise ValueError(msg)

        return nx.read_pajek(path)

    @staticmethod
    def export_to_dataframe(graph: nx.Graph) -> dict[str, pd.DataFrame]:
        """Export graph as pandas DataFrames."""
        # Node DataFrame
        node_data = []
        for node, attrs in graph.nodes(data=True):
            node_dict = {"node_id": node}
            node_dict.update(attrs)
            node_data.append(node_dict)

        nodes_df = pd.DataFrame(node_data)

        # Edge DataFrame
        edge_data = []
        for u, v, attrs in graph.edges(data=True):
            edge_dict = {"source": u, "target": v}
            edge_dict.update(attrs)
            edge_data.append(edge_dict)

        edges_df = pd.DataFrame(edge_data)

        return {
            "nodes": nodes_df.to_dict("records"),
            "edges": edges_df.to_dict("records"),
            "node_columns": list(nodes_df.columns),
            "edge_columns": list(edges_df.columns),
        }

    @staticmethod
    def import_from_dataframe(
        nodes_df: Optional[pd.DataFrame] = None,
        edges_df: Optional[pd.DataFrame] = None,
        source_col: str = "source",
        target_col: str = "target",
        node_id_col: str = "node_id",
        create_using: Optional[nx.Graph] = None,
    ) -> nx.Graph:
        """Import graph from pandas DataFrames."""
        if create_using is None:
            graph = nx.Graph()
        else:
            graph = create_using

        # Add nodes
        if nodes_df is not None:
            for _, row in nodes_df.iterrows():
                node_id = row[node_id_col]
                attrs = row.drop(node_id_col).to_dict()
                graph.add_node(node_id, **attrs)

        # Add edges
        if edges_df is not None:
            for _, row in edges_df.iterrows():
                source = row[source_col]
                target = row[target_col]
                attrs = row.drop([source_col, target_col]).to_dict()
                graph.add_edge(source, target, **attrs)

        return graph
