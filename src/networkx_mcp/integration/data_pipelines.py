"""Data integration pipelines for various data sources."""

import asyncio
import json
import logging
import sqlite3
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataPipelines:
    """Intelligent data pipelines for graph construction from various sources."""

    @staticmethod
    def csv_pipeline(
        filepath: str,
        edge_columns: Optional[Tuple[str, str]] = None,
        node_attributes: Optional[List[str]] = None,
        edge_attributes: Optional[List[str]] = None,
        delimiter: str = ",",
        encoding: str = "utf-8",
        type_inference: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Intelligent CSV parsing with type inference.

        Parameters:
        -----------
        filepath : str
            Path to CSV file
        edge_columns : tuple, optional
            Column names for (source, target). Auto-detected if None
        node_attributes : list, optional
            Columns to use as node attributes
        edge_attributes : list, optional
            Columns to use as edge attributes
        type_inference : bool
            Automatically infer data types

        Returns:
        --------
        Dict containing graph and metadata
        """
        start_time = time.time()

        # Read CSV with pandas for intelligent parsing
        df = pd.read_csv(filepath, delimiter=delimiter, encoding=encoding)

        # Auto-detect edge columns if not provided
        if edge_columns is None:
            # Look for common patterns
            possible_source = ["source", "from", "sender", "origin", "start"]
            possible_target = ["target", "to", "receiver", "destination", "end"]

            source_col = None
            target_col = None

            for col in df.columns:
                if col.lower() in possible_source:
                    source_col = col
                elif col.lower() in possible_target:
                    target_col = col

            if source_col and target_col:
                edge_columns = (source_col, target_col)
            else:
                # Use first two columns
                edge_columns = (df.columns[0], df.columns[1])
                logger.warning(f"Auto-detected edge columns: {edge_columns}")

        # Create graph
        graph = nx.DiGraph() if kwargs.get("directed", True) else nx.Graph()

        # Type inference
        if type_inference:
            for col in df.columns:
                # Try to convert to numeric
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    # Try to convert to datetime
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except (ValueError, TypeError):
                        pass  # Keep as string

        # Add edges with attributes
        edge_attr_cols = edge_attributes or [
            col for col in df.columns if col not in edge_columns
        ]

        for _, row in df.iterrows():
            source = row[edge_columns[0]]
            target = row[edge_columns[1]]

            # Edge attributes
            attrs = {}
            for col in edge_attr_cols:
                value = row[col]
                # Handle NaN values
                if pd.notna(value):
                    # Convert numpy types to Python types
                    if isinstance(value, np.integer):
                        value = int(value)
                    elif isinstance(value, np.floating):
                        value = float(value)
                    elif isinstance(value, pd.Timestamp):
                        value = value.isoformat()
                    attrs[col] = value

            graph.add_edge(source, target, **attrs)

        # Add node attributes if specified
        if node_attributes:
            # Read node attributes from another sheet or section
            pass  # Implement based on specific format

        processing_time = time.time() - start_time

        return {
            "graph": graph,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "edge_columns_used": edge_columns,
            "attributes_detected": edge_attr_cols,
            "data_types": {col: str(df[col].dtype) for col in df.columns},
            "processing_time": processing_time,
            "source_file": filepath,
        }

    @staticmethod
    def json_pipeline(
        filepath: str,
        format_type: str = "auto",
        node_path: Optional[str] = None,
        edge_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Convert nested JSON to graph.

        Parameters:
        -----------
        filepath : str
            Path to JSON file
        format_type : str
            Format type: 'auto', 'node_link', 'adjacency', 'tree', 'custom'
        node_path : str, optional
            JSONPath to nodes (for custom format)
        edge_path : str, optional
            JSONPath to edges (for custom format)

        Returns:
        --------
        Dict containing graph and metadata
        """
        start_time = time.time()

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        # Auto-detect format
        if format_type == "auto":
            if isinstance(data, dict):
                if "nodes" in data and "links" in data:
                    format_type = "node_link"
                elif "adjacency" in data:
                    format_type = "adjacency"
                elif all(isinstance(v, (dict, list)) for v in data.values()):
                    format_type = "tree"
            elif isinstance(data, list) and len(data) > 0:
                if all("source" in item and "target" in item for item in data[:5]):
                    format_type = "edge_list"

        # Create graph based on format
        if format_type == "node_link":
            graph = nx.node_link_graph(data)

        elif format_type == "adjacency":
            graph = nx.adjacency_graph(data)

        elif format_type == "tree":
            # Convert hierarchical JSON to tree
            graph = nx.DiGraph()
            DataPipelines._json_tree_to_graph(data, graph)

        elif format_type == "edge_list":
            graph = nx.DiGraph() if kwargs.get("directed", True) else nx.Graph()
            for item in data:
                graph.add_edge(
                    item["source"],
                    item["target"],
                    **{k: v for k, v in item.items() if k not in ["source", "target"]},
                )

        elif format_type == "custom":
            # Use JSONPath to extract nodes and edges
            graph = DataPipelines._parse_custom_json(
                data, node_path, edge_path, **kwargs
            )

        else:
            msg = f"Unknown format type: {format_type}"
            raise ValueError(msg)

        processing_time = time.time() - start_time

        return {
            "graph": graph,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "format_detected": format_type,
            "processing_time": processing_time,
            "source_file": filepath,
        }

    @staticmethod
    def database_pipeline(
        connection_string: str,
        query: str,
        db_type: str = "sqlite",
        edge_columns: Optional[Tuple[str, str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        SQL to graph conversion.

        Parameters:
        -----------
        connection_string : str
            Database connection string
        query : str
            SQL query to fetch data
        db_type : str
            Database type: 'sqlite', 'postgresql', 'mysql'
        edge_columns : tuple, optional
            Column names for (source, target)

        Returns:
        --------
        Dict containing graph and metadata
        """
        start_time = time.time()

        # Connect to database
        if db_type == "sqlite":
            conn = sqlite3.connect(connection_string)
            df = pd.read_sql_query(query, conn)
            conn.close()

        elif db_type == "postgresql":
            # Requires psycopg2
            import psycopg2

            conn = psycopg2.connect(connection_string)
            df = pd.read_sql_query(query, conn)
            conn.close()

        elif db_type == "mysql":
            # Requires pymysql
            import pymysql

            conn = pymysql.connect(connection_string)
            df = pd.read_sql_query(query, conn)
            conn.close()

        else:
            msg = f"Unsupported database type: {db_type}"
            raise ValueError(msg)

        # Convert DataFrame to graph
        result = DataPipelines._dataframe_to_graph(df, edge_columns, **kwargs)

        result["source_database"] = db_type
        result["query"] = query
        result["processing_time"] = time.time() - start_time

        return result

    @staticmethod
    async def api_pipeline(
        base_url: str,
        endpoints: List[Dict[str, Any]],
        rate_limit: float = 1.0,
        max_retries: int = 3,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        REST API pagination and rate limiting.

        Parameters:
        -----------
        base_url : str
            Base API URL
        endpoints : list
            List of endpoint configurations
        rate_limit : float
            Seconds between requests
        max_retries : int
            Maximum retry attempts

        Returns:
        --------
        Dict containing graph and metadata
        """
        start_time = time.time()
        graph = nx.DiGraph() if kwargs.get("directed", True) else nx.Graph()

        async with aiohttp.ClientSession() as session:
            total_requests = 0
            total_items = 0

            for endpoint in endpoints:
                url = base_url + endpoint["path"]
                params = endpoint.get("params", {})

                # Handle pagination
                page = 1
                while True:
                    if endpoint.get("pagination_type") == "page":
                        params[endpoint.get("page_param", "page")] = page

                    # Rate limiting
                    await asyncio.sleep(rate_limit)

                    # Make request with retries
                    for attempt in range(max_retries):
                        try:
                            async with session.get(url, params=params) as response:
                                HTTP_OK = 200  # noqa: PLR2004
                                if response.status == HTTP_OK:
                                    data = await response.json()
                                    total_requests += 1
                                    break
                                HTTP_TOO_MANY_REQUESTS = 429  # noqa: PLR2004
                                if (
                                    response.status == HTTP_TOO_MANY_REQUESTS
                                ):  # Rate limited
                                    await asyncio.sleep(2**attempt)
                        except Exception as e:
                            logger.error(f"API request failed: {e}")
                            if attempt == max_retries - 1:
                                raise

                    # Extract graph data
                    items = data
                    if endpoint.get("data_path"):
                        for path_part in endpoint["data_path"].split("."):
                            items = items.get(path_part, [])

                    if not items:
                        break

                    # Add to graph
                    for item in items:
                        if endpoint["type"] == "nodes":
                            graph.add_node(
                                item[endpoint["id_field"]],
                                **{
                                    k: v
                                    for k, v in item.items()
                                    if k != endpoint["id_field"]
                                },
                            )
                        elif endpoint["type"] == "edges":
                            graph.add_edge(
                                item[endpoint["source_field"]],
                                item[endpoint["target_field"]],
                                **{
                                    k: v
                                    for k, v in item.items()
                                    if k
                                    not in [
                                        endpoint["source_field"],
                                        endpoint["target_field"],
                                    ]
                                },
                            )

                    total_items += len(items)

                    # Check if more pages
                    if endpoint.get("pagination_type") == "page":
                        if len(items) < endpoint.get("page_size", 100):
                            break
                        page += 1
                    else:
                        break

        return {
            "graph": graph,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "total_api_requests": total_requests,
            "total_items_fetched": total_items,
            "processing_time": time.time() - start_time,
        }

    @staticmethod
    def streaming_pipeline(
        stream_generator: Iterator[Dict[str, Any]],
        window_size: Optional[int] = None,
        update_interval: float = 1.0,
        **kwargs,
    ) -> Iterator[Dict[str, Any]]:
        """
        Real-time data ingestion from streaming sources.

        Parameters:
        -----------
        stream_generator : Iterator
            Generator yielding graph updates
        window_size : int, optional
            Sliding window size (None for cumulative)
        update_interval : float
            Minimum seconds between updates

        Yields:
        -------
        Dict containing current graph state
        """
        graph = nx.DiGraph() if kwargs.get("directed", True) else nx.Graph()

        window = []
        last_update = time.time()
        total_items = 0

        for item in stream_generator:
            total_items += 1

            # Add to graph
            if item["type"] == "node":
                graph.add_node(item["id"], **item.get("attributes", {}))
            elif item["type"] == "edge":
                graph.add_edge(
                    item["source"], item["target"], **item.get("attributes", {})
                )

            # Handle windowing
            if window_size:
                window.append(item)
                if len(window) > window_size:
                    old_item = window.pop(0)
                    # Remove old item from graph
                    if old_item["type"] == "node":
                        if old_item["id"] in graph:
                            graph.remove_node(old_item["id"])
                    elif old_item["type"] == "edge":
                        if graph.has_edge(old_item["source"], old_item["target"]):
                            graph.remove_edge(old_item["source"], old_item["target"])

            # Yield update if interval passed
            current_time = time.time()
            if current_time - last_update >= update_interval:
                yield {
                    "graph": graph.copy(),
                    "num_nodes": graph.number_of_nodes(),
                    "num_edges": graph.number_of_edges(),
                    "total_items_processed": total_items,
                    "window_size": len(window) if window_size else None,
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                }
                last_update = current_time

    @staticmethod
    def excel_pipeline(
        filepath: str, sheet_mapping: Optional[Dict[str, str]] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Multi-sheet Excel processing.

        Parameters:
        -----------
        filepath : str
            Path to Excel file
        sheet_mapping : dict, optional
            Mapping of sheet names to data types
            e.g., {"Nodes": "nodes", "Edges": "edges"}

        Returns:
        --------
        Dict containing graph and metadata
        """
        start_time = time.time()

        # Read Excel file
        excel_file = pd.ExcelFile(filepath)

        # Auto-detect sheet mapping if not provided
        if sheet_mapping is None:
            sheet_mapping = {}
            for sheet in excel_file.sheet_names:
                sheet_lower = sheet.lower()
                if "node" in sheet_lower:
                    sheet_mapping[sheet] = "nodes"
                elif "edge" in sheet_lower or "link" in sheet_lower:
                    sheet_mapping[sheet] = "edges"
                elif "attribute" in sheet_lower:
                    sheet_mapping[sheet] = "attributes"

        # Create graph
        graph = nx.DiGraph() if kwargs.get("directed", True) else nx.Graph()

        # Process sheets
        for sheet_name, data_type in sheet_mapping.items():
            df = excel_file.parse(sheet_name)

            if data_type == "nodes":
                # Add nodes
                for _, row in df.iterrows():
                    node_id = row.iloc[0]  # First column as ID
                    attrs = row.iloc[1:].to_dict()
                    graph.add_node(node_id, **attrs)

            elif data_type == "edges":
                # Add edges
                for _, row in df.iterrows():
                    source = row.iloc[0]
                    target = row.iloc[1]
                    MIN_ROW_LENGTH_FOR_ATTRS = 2  # noqa: PLR2004
                    attrs = (
                        row.iloc[2:].to_dict()
                        if len(row) > MIN_ROW_LENGTH_FOR_ATTRS
                        else {}
                    )
                    graph.add_edge(source, target, **attrs)

            elif data_type == "attributes":
                # Update node/edge attributes
                for _, row in df.iterrows():
                    if "node" in df.columns:
                        node = row["node"]
                        attrs = {k: v for k, v in row.items() if k != "node"}
                        graph.nodes[node].update(attrs)
                    elif "source" in df.columns and "target" in df.columns:
                        source = row["source"]
                        target = row["target"]
                        attrs = {
                            k: v
                            for k, v in row.items()
                            if k not in ["source", "target"]
                        }
                        if graph.has_edge(source, target):
                            graph.edges[source, target].update(attrs)

        return {
            "graph": graph,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "sheets_processed": list(sheet_mapping.keys()),
            "sheet_mapping": sheet_mapping,
            "processing_time": time.time() - start_time,
            "source_file": filepath,
        }

    @staticmethod
    def _json_tree_to_graph(
        data: Union[Dict, List],
        graph: nx.DiGraph,
        parent: Optional[Any] = None,
        parent_key: Optional[str] = None,
    ):
        """Convert hierarchical JSON to directed graph."""
        if isinstance(data, dict):
            # Create node for this dict
            node_id = f"{parent_key}_{id(data)}" if parent_key else "root"
            graph.add_node(node_id, type="object", keys=list(data.keys()))

            if parent is not None:
                graph.add_edge(parent, node_id, relation=parent_key)

            # Process children
            for key, value in data.items():
                DataPipelines._json_tree_to_graph(value, graph, node_id, key)

        elif isinstance(data, list):
            # Create node for this list
            node_id = f"{parent_key}_{id(data)}" if parent_key else "root"
            graph.add_node(node_id, type="array", length=len(data))

            if parent is not None:
                graph.add_edge(parent, node_id, relation=parent_key)

            # Process items
            for i, item in enumerate(data):
                DataPipelines._json_tree_to_graph(item, graph, node_id, f"[{i}]")

        else:
            # Leaf node
            node_id = f"{parent_key}_{id(data)}"
            graph.add_node(node_id, type="value", value=data)

            if parent is not None:
                graph.add_edge(parent, node_id, relation=parent_key)

    @staticmethod
    def _dataframe_to_graph(
        df: pd.DataFrame, edge_columns: Optional[Tuple[str, str]] = None, **kwargs
    ) -> Dict[str, Any]:
        """Convert pandas DataFrame to graph."""
        if edge_columns is None:
            edge_columns = (df.columns[0], df.columns[1])

        graph = nx.DiGraph() if kwargs.get("directed", True) else nx.Graph()

        # Add edges with attributes
        edge_attrs = [col for col in df.columns if col not in edge_columns]

        for _, row in df.iterrows():
            attrs = {col: row[col] for col in edge_attrs if pd.notna(row[col])}
            graph.add_edge(row[edge_columns[0]], row[edge_columns[1]], **attrs)

        return {
            "graph": graph,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "edge_columns": edge_columns,
            "attributes": edge_attrs,
        }

    @staticmethod
    def _parse_custom_json(
        data: Any, node_path: Optional[str], edge_path: Optional[str], **kwargs
    ) -> nx.Graph:
        """Parse custom JSON format using JSONPath-like expressions."""
        # Simplified JSONPath implementation
        # In production, use jsonpath-ng or similar
        graph = nx.DiGraph() if kwargs.get("directed", True) else nx.Graph()

        # Extract nodes
        if node_path:
            nodes = DataPipelines._extract_path(data, node_path)
            for node in nodes:
                if isinstance(node, dict) and "id" in node:
                    graph.add_node(
                        node["id"], **{k: v for k, v in node.items() if k != "id"}
                    )

        # Extract edges
        if edge_path:
            edges = DataPipelines._extract_path(data, edge_path)
            for edge in edges:
                if isinstance(edge, dict) and "source" in edge and "target" in edge:
                    graph.add_edge(
                        edge["source"],
                        edge["target"],
                        **{
                            k: v
                            for k, v in edge.items()
                            if k not in ["source", "target"]
                        },
                    )

        return graph

    @staticmethod
    def _extract_path(data: Any, path: str) -> List[Any]:
        """Simple path extraction (replace with proper JSONPath in production)."""
        parts = path.split(".")
        current = data

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part, [])
            elif isinstance(current, list) and part.isdigit():
                current = current[int(part)] if int(part) < len(current) else []
            else:
                return []

        return current if isinstance(current, list) else [current]
