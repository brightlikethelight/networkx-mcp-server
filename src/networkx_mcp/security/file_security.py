"""Secure file operations with strict validation."""

import hashlib
import json
import os
import tempfile

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import networkx as nx
import yaml

from .validator import SecurityError
from .validator import SecurityValidator


class FileSecurityError(SecurityError):
    """File operation security error."""
    pass


class SecureFileHandler:
    """Secure file operations with strict validation."""

    def __init__(self, allowed_dirs: Optional[List[str]] = None):
        """Initialize with allowed directories."""
        if allowed_dirs is None:
            # Default to temp directory only for safety
            self.temp_dir = Path(tempfile.mkdtemp(prefix="networkx_mcp_secure_"))
            self.allowed_dirs = [self.temp_dir]
        else:
            self.allowed_dirs = []
            for dir_path in allowed_dirs:
                try:
                    resolved = Path(dir_path).resolve(strict=True)
                    if not resolved.is_dir():
                        msg = f"Not a directory: {dir_path}"
                        raise ValueError(msg)
                    self.allowed_dirs.append(resolved)
                except Exception as e:
                    msg = f"Invalid allowed directory '{dir_path}': {e}"
                    raise FileSecurityError(msg) from e

            # Always include secure temp
            self.temp_dir = Path(tempfile.mkdtemp(prefix="networkx_mcp_secure_"))
            self.allowed_dirs.append(self.temp_dir)

        # Track created files for cleanup
        self._created_files = set()

    def validate_path(self, filepath: Union[str, Path]) -> Path:
        """Validate and resolve file path securely."""
        try:
            # Convert to Path and resolve
            requested = Path(filepath).resolve(strict=False)
        except Exception as e:
            msg = f"Invalid path '{filepath}': {e}"
            raise FileSecurityError(msg) from e

        # Check for null bytes (path injection)
        if "\x00" in str(filepath):
            msg = "Path contains null bytes"
            raise FileSecurityError(msg)

        # Check if it's a symlink
        if requested.exists() and requested.is_symlink():
            # Resolve symlink target
            try:
                target = requested.resolve(strict=True)
            except Exception as e:
                msg = "Cannot resolve symlink target"
                raise FileSecurityError(msg) from e

            # Symlink target must also be in allowed dirs
            requested = target

        # Check if path is within allowed directories
        is_allowed = False
        for allowed_dir in self.allowed_dirs:
            try:
                # This will raise ValueError if not relative
                requested.relative_to(allowed_dir)
                is_allowed = True
                break
            except ValueError:
                continue

        if not is_allowed:
            msg = f"Access denied: Path '{filepath}' is outside allowed directories"
            raise FileSecurityError(
                msg
            )

        # Additional security checks
        path_str = str(requested)

        # Check for directory traversal attempts
        if ".." in path_str or path_str.count("/") > 20:
            msg = "Suspicious path pattern detected"
            raise FileSecurityError(msg)

        # Check for hidden files (unless explicitly allowed)
        if requested.name.startswith(".") and requested.name != ".":
            msg = "Access to hidden files not allowed"
            raise FileSecurityError(msg)

        return requested

    def validate_format(self, file_format: str) -> str:
        """Validate file format is safe."""
        return SecurityValidator.validate_file_format(file_format)

    def create_secure_temp_file(self, suffix: str = "", prefix: str = "graph_") -> Path:
        """Create a secure temporary file."""
        # Validate suffix
        if suffix and not suffix.startswith("."):
            suffix = "." + suffix

        # Ensure suffix is safe
        if suffix:
            safe_extensions = {
                ".graphml", ".gml", ".json", ".yaml", ".yml",
                ".gexf", ".edgelist", ".adjlist", ".txt"
            }
            if suffix not in safe_extensions:
                msg = f"Unsafe file extension: {suffix}"
                raise FileSecurityError(msg)

        # Create temp file in secure directory
        fd, filepath = tempfile.mkstemp(
            suffix=suffix,
            prefix=prefix,
            dir=str(self.temp_dir)
        )
        os.close(fd)  # Close file descriptor

        secure_path = Path(filepath)
        self._created_files.add(secure_path)

        return secure_path

    def safe_read_graph(self, filepath: Union[str, Path], file_format: str) -> nx.Graph:
        """Safely read graph from file."""
        # Validate inputs
        secure_path = self.validate_path(filepath)
        safe_format = self.validate_format(file_format)

        # Check file exists and is readable
        if not secure_path.exists():
            msg = f"File not found: {filepath}"
            raise FileNotFoundError(msg)

        if not secure_path.is_file():
            msg = "Path is not a file"
            raise FileSecurityError(msg)

        # Check file size (prevent DoS)
        max_size = 100 * 1024 * 1024  # 100MB
        file_size = secure_path.stat().st_size
        if file_size > max_size:
            msg = f"File too large ({file_size} bytes), max {max_size} bytes"
            raise FileSecurityError(
                msg
            )

        # Read based on format
        try:
            if safe_format == "graphml":
                return nx.read_graphml(secure_path)
            elif safe_format == "gml":
                return nx.read_gml(secure_path)
            elif safe_format == "json":
                return self._read_json_graph(secure_path)
            elif safe_format == "yaml":
                return self._read_yaml_graph(secure_path)
            elif safe_format == "edgelist":
                return nx.read_edgelist(secure_path)
            elif safe_format == "adjlist":
                return nx.read_adjlist(secure_path)
            elif safe_format == "gexf":
                return nx.read_gexf(secure_path)
            else:
                msg = f"Unsupported format: {safe_format}"
                raise FileSecurityError(msg)
        except Exception as e:
            # Sanitize error message
            safe_error = SecurityValidator.sanitize_error_message(e)
            msg = f"Failed to read graph: {safe_error}"
            raise FileSecurityError(msg) from e

    def safe_write_graph(self, graph: nx.Graph, filepath: Union[str, Path],
                         file_format: str) -> Path:
        """Safely write graph to file."""
        # Validate inputs
        secure_path = self.validate_path(filepath)
        safe_format = self.validate_format(file_format)

        # Ensure parent directory exists
        secure_path.parent.mkdir(parents=True, exist_ok=True)

        # Write based on format
        try:
            if safe_format == "graphml":
                nx.write_graphml(graph, secure_path)
            elif safe_format == "gml":
                nx.write_gml(graph, secure_path)
            elif safe_format == "json":
                self._write_json_graph(graph, secure_path)
            elif safe_format == "yaml":
                self._write_yaml_graph(graph, secure_path)
            elif safe_format == "edgelist":
                nx.write_edgelist(graph, secure_path)
            elif safe_format == "adjlist":
                nx.write_adjlist(graph, secure_path)
            elif safe_format == "gexf":
                nx.write_gexf(graph, secure_path)
            else:
                msg = f"Unsupported format: {safe_format}"
                raise FileSecurityError(msg)

            # Set secure permissions (read/write for owner only)
            os.chmod(secure_path, 0o600)

            return secure_path

        except Exception as e:
            # Clean up on failure
            if secure_path.exists():
                secure_path.unlink()

            # Sanitize error message
            safe_error = SecurityValidator.sanitize_error_message(e)
            msg = f"Failed to write graph: {safe_error}"
            raise FileSecurityError(msg) from e

    def _read_json_graph(self, filepath: Path) -> nx.Graph:
        """Read graph from JSON with validation."""
        with open(filepath, encoding="utf-8") as f:
            # Read with size limit
            content = f.read(10 * 1024 * 1024)  # 10MB limit

        # Parse JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON: {e}"
            raise FileSecurityError(msg) from e

        # Validate structure
        if not isinstance(data, dict):
            msg = "JSON must be object at root"
            raise FileSecurityError(msg)

        # Convert to graph with validation
        if "nodes" in data and "edges" in data:
            # Node-link format
            return self._json_node_link_to_graph(data)
        else:
            # Try networkx JSON format
            return nx.node_link_graph(data)

    def _write_json_graph(self, graph: nx.Graph, filepath: Path) -> None:
        """Write graph to JSON safely."""
        # Convert to node-link format
        data = nx.node_link_data(graph)

        # Sanitize all attributes
        if "nodes" in data:
            for node in data["nodes"]:
                if isinstance(node, dict) and "attributes" in node:
                    node["attributes"] = SecurityValidator.sanitize_attributes(
                        node["attributes"]
                    )

        if "edges" in data:
            for edge in data["edges"]:
                if isinstance(edge, dict) and "attributes" in edge:
                    edge["attributes"] = SecurityValidator.sanitize_attributes(
                        edge["attributes"]
                    )

        # Write with pretty printing
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _read_yaml_graph(self, filepath: Path) -> nx.Graph:
        """Read graph from YAML with validation."""
        with open(filepath, encoding="utf-8") as f:
            # Use safe loader to prevent code execution
            data = yaml.safe_load(f)

        # Validate and convert
        if not isinstance(data, dict):
            msg = "YAML must be mapping at root"
            raise FileSecurityError(msg)

        # Convert to graph
        if "nodes" in data and "edges" in data:
            return self._json_node_link_to_graph(data)
        else:
            msg = "Invalid YAML graph format"
            raise FileSecurityError(msg)

    def _write_yaml_graph(self, graph: nx.Graph, filepath: Path) -> None:
        """Write graph to YAML safely."""
        # Convert to safe format
        data = {
            "nodes": [
                {
                    "id": SecurityValidator.validate_node_id(node),
                    "attributes": SecurityValidator.sanitize_attributes(attrs)
                }
                for node, attrs in graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": SecurityValidator.validate_node_id(u),
                    "target": SecurityValidator.validate_node_id(v),
                    "attributes": SecurityValidator.sanitize_attributes(attrs)
                }
                for u, v, attrs in graph.edges(data=True)
            ]
        }

        with open(filepath, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False)

    def _json_node_link_to_graph(self, data: Dict[str, Any]) -> nx.Graph:
        """Convert JSON node-link format to graph with validation."""
        # Determine graph type
        directed = data.get("directed", False)
        multigraph = data.get("multigraph", False)

        if directed and multigraph:
            graph = nx.MultiDiGraph()
        elif directed:
            graph = nx.DiGraph()
        elif multigraph:
            graph = nx.MultiGraph()
        else:
            graph = nx.Graph()

        # Add nodes with validation
        if "nodes" in data:
            for node_data in data["nodes"]:
                if isinstance(node_data, dict) and "id" in node_data:
                    node_id = SecurityValidator.validate_node_id(node_data["id"])
                    attrs = SecurityValidator.sanitize_attributes(
                        node_data.get("attributes", {})
                    )
                    graph.add_node(node_id, **attrs)

        # Add edges with validation
        if "edges" in data:
            for edge_data in data["edges"]:
                if isinstance(edge_data, dict):
                    if "source" in edge_data and "target" in edge_data:
                        source = SecurityValidator.validate_node_id(edge_data["source"])
                        target = SecurityValidator.validate_node_id(edge_data["target"])
                        attrs = SecurityValidator.sanitize_attributes(
                            edge_data.get("attributes", {})
                        )
                        graph.add_edge(source, target, **attrs)

        return graph

    def get_file_hash(self, filepath: Union[str, Path]) -> str:
        """Get secure hash of file contents."""
        secure_path = self.validate_path(filepath)

        if not secure_path.exists():
            msg = f"File not found: {filepath}"
            raise FileNotFoundError(msg)

        # Calculate SHA-256 hash
        hash_sha256 = hashlib.sha256()

        with open(secure_path, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()

    def cleanup(self) -> None:
        """Clean up temporary files."""
        for filepath in self._created_files:
            try:
                if filepath.exists():
                    filepath.unlink()
            except Exception:
                pass  # Best effort cleanup

        # Remove temp directory
        try:
            if self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir)
        except Exception:
            pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False
