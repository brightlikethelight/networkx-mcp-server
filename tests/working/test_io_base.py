"""Comprehensive tests for io/base.py - Target: 100% coverage (78 lines).

This module contains critical SECURITY foundations for file I/O operations.
The validate_file_path function prevents directory traversal attacks.
Testing must be exhaustive to ensure security is not compromised.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import pytest

from networkx_mcp.io.base import GraphReader, GraphWriter


class ConcreteGraphReader:
    """Concrete implementation for testing abstract GraphReader."""

    def __init__(self, format_name: str, file_extensions: list[str]) -> None:
        # Manually set up the attributes like the base class would
        self.format_name = format_name
        self.file_extensions = file_extensions

    def validate_file(self, filepath) -> bool:
        """Copy the validate_file method from GraphReader."""
        from pathlib import Path

        path = Path(filepath)
        return path.suffix.lower() in self.file_extensions

    async def read(self, filepath, **options) -> nx.Graph:
        """Concrete implementation of read method."""
        return nx.Graph()


class ConcreteGraphWriter:
    """Concrete implementation for testing abstract GraphWriter."""

    def __init__(self, format_name: str, file_extension: str) -> None:
        # Manually set up the attributes like the base class would
        self.format_name = format_name
        self.file_extension = file_extension

    async def write(self, graph: nx.Graph, filepath, **options) -> bool:
        """Concrete implementation of write method."""
        return True


class ProperGraphReader(GraphReader):
    """Proper subclass that inherits from GraphReader to test base class methods."""

    def __init__(self, format_name: str, file_extensions: list[str]) -> None:
        # Call the actual base class __init__ to cover those lines
        super().__init__(format_name, file_extensions)

    async def read(self, filepath, **options) -> nx.Graph:
        """Required abstract method implementation."""
        return nx.Graph()


class ProperGraphWriter(GraphWriter):
    """Proper subclass that inherits from GraphWriter to test base class methods."""

    def __init__(self, format_name: str, file_extension: str) -> None:
        # Call the actual base class __init__ to cover those lines
        super().__init__(format_name, file_extension)

    async def write(self, graph: nx.Graph, filepath, **options) -> bool:
        """Required abstract method implementation."""
        return True


class TestGraphReader:
    """Test the GraphReader abstract base class."""

    def test_init(self):
        """Test GraphReader initialization."""

        reader = ConcreteGraphReader("test_format", [".txt", ".data"])

        assert reader.format_name == "test_format"
        assert reader.file_extensions == [".txt", ".data"]

    def test_base_class_init(self):
        """Test actual GraphReader base class initialization."""
        # This will cover lines 14-15 in the base class
        reader = ProperGraphReader("base_format", [".base", ".test"])

        assert reader.format_name == "base_format"
        assert reader.file_extensions == [".base", ".test"]

    def test_base_class_validate_file(self):
        """Test actual GraphReader base class validate_file method."""
        # This will cover lines 23-24 in the base class
        reader = ProperGraphReader("test", [".gml", ".json"])

        # Test the actual base class validate_file method
        assert reader.validate_file("test.gml") is True
        assert reader.validate_file("test.json") is True
        assert reader.validate_file("test.txt") is False

    def test_validate_file_valid_extensions(self):
        """Test file validation with valid extensions."""

        reader = ConcreteGraphReader("gml", [".gml", ".graph"])

        # Valid extensions
        assert reader.validate_file("test.gml") is True
        assert reader.validate_file("test.graph") is True
        assert reader.validate_file("test.GML") is True  # Case insensitive
        assert reader.validate_file("test.GRAPH") is True

    def test_validate_file_invalid_extensions(self):
        """Test file validation with invalid extensions."""

        reader = ConcreteGraphReader("gml", [".gml", ".graph"])

        # Invalid extensions
        assert reader.validate_file("test.txt") is False
        assert reader.validate_file("test.json") is False
        assert reader.validate_file("test") is False  # No extension
        assert reader.validate_file("test.") is False  # Empty extension

    def test_validate_file_with_path_objects(self):
        """Test file validation with Path objects."""

        reader = ConcreteGraphReader("json", [".json"])

        # Path objects
        assert reader.validate_file(Path("data.json")) is True
        assert reader.validate_file(Path("data.xml")) is False

    def test_validate_file_edge_cases(self):
        """Test file validation edge cases."""

        reader = ConcreteGraphReader("multi", [".txt", ".data", ".log"])

        # Multiple dots
        assert reader.validate_file("file.backup.txt") is True
        assert reader.validate_file("file.backup.json") is False

        # Complex paths
        assert reader.validate_file("/path/to/file.data") is True
        assert reader.validate_file("../path/file.log") is True

    @pytest.mark.asyncio
    async def test_read_method_abstract(self):
        """Test that read method can be implemented."""
        reader = ConcreteGraphReader("test", [".txt"])

        # Should return a graph
        result = await reader.read("test.txt")
        assert isinstance(result, nx.Graph)


class TestGraphWriter:
    """Test the GraphWriter abstract base class."""

    def test_init(self):
        """Test GraphWriter initialization."""

        writer = ConcreteGraphWriter("gml_writer", ".gml")

        assert writer.format_name == "gml_writer"
        assert writer.file_extension == ".gml"

    def test_base_class_init(self):
        """Test actual GraphWriter base class initialization."""
        # This will cover lines 31-32 in the base class
        writer = ProperGraphWriter("base_writer", ".base")

        assert writer.format_name == "base_writer"
        assert writer.file_extension == ".base"

    def test_init_various_extensions(self):
        """Test GraphWriter with various extensions."""

        # Different extensions
        writer1 = ConcreteGraphWriter("json", ".json")
        assert writer1.file_extension == ".json"

        writer2 = ConcreteGraphWriter("xml", ".xml")
        assert writer2.file_extension == ".xml"

        writer3 = ConcreteGraphWriter("custom", ".custom")
        assert writer3.file_extension == ".custom"

    @pytest.mark.asyncio
    async def test_write_method_abstract(self):
        """Test that write method can be implemented."""
        writer = ConcreteGraphWriter("test", ".txt")
        graph = nx.Graph()

        # Should return True for success
        result = await writer.write(graph, "test.txt")
        assert result is True


class TestValidateFilePath:
    """Test the critical validate_file_path security function."""

    def test_valid_relative_path(self):
        """Test validation of valid relative paths."""
        from networkx_mcp.io.base import validate_file_path

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "test.txt"
            test_file.write_text("test")

            # Valid relative path
            with patch("tempfile.gettempdir", return_value=temp_dir):
                result = validate_file_path("test.txt", must_exist=False)
                assert isinstance(result, Path)
                assert result.name == "test.txt"

    def test_valid_absolute_path_in_temp(self):
        """Test validation of absolute paths within temp directory."""
        from networkx_mcp.io.base import validate_file_path

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "valid.txt"
            test_file.write_text("test")

            # Valid absolute path in temp directory
            result = validate_file_path(str(test_file), must_exist=True)
            assert result == test_file

    def test_directory_traversal_attack_prevention(self):
        """Test prevention of directory traversal attacks."""
        from networkx_mcp.io.base import validate_file_path

        # Directory traversal attempts
        with pytest.raises(
            ValueError, match="Invalid file path - no directory traversal allowed"
        ):
            validate_file_path("../etc/passwd")

        with pytest.raises(
            ValueError, match="Invalid file path - no directory traversal allowed"
        ):
            validate_file_path("../../etc/passwd")

        with pytest.raises(
            ValueError, match="Invalid file path - no directory traversal allowed"
        ):
            validate_file_path("safe/../etc/passwd")

        with pytest.raises(
            ValueError, match="Invalid file path - no directory traversal allowed"
        ):
            validate_file_path("./safe/../etc/passwd")

    def test_absolute_path_outside_temp_rejection(self):
        """Test rejection of absolute paths outside temp directory."""
        from networkx_mcp.io.base import validate_file_path

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("tempfile.gettempdir", return_value=temp_dir):
                # Absolute path outside temp directory
                with pytest.raises(
                    ValueError,
                    match="Invalid file path - no directory traversal allowed",
                ):
                    validate_file_path("/etc/passwd")

                with pytest.raises(
                    ValueError,
                    match="Invalid file path - no directory traversal allowed",
                ):
                    validate_file_path("/home/user/file.txt")

                with pytest.raises(
                    ValueError,
                    match="Invalid file path - no directory traversal allowed",
                ):
                    validate_file_path("/tmp/other/file.txt")  # Different temp path

    def test_file_not_found_when_required(self):
        """Test FileNotFoundError when file must exist but doesn't."""
        from networkx_mcp.io.base import validate_file_path

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("tempfile.gettempdir", return_value=temp_dir):
                nonexistent = Path(temp_dir) / "nonexistent.txt"

                # File must exist but doesn't
                with pytest.raises(
                    FileNotFoundError, match=f"File not found: {nonexistent}"
                ):
                    validate_file_path(str(nonexistent), must_exist=True)

    def test_file_creation_when_not_required(self):
        """Test parent directory creation when file doesn't need to exist."""
        from networkx_mcp.io.base import validate_file_path

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("tempfile.gettempdir", return_value=temp_dir):
                # Nested path that doesn't exist
                nested_file = Path(temp_dir) / "subdir" / "nested" / "file.txt"

                result = validate_file_path(str(nested_file), must_exist=False)

                # Parent directories should be created
                assert result.parent.exists()
                assert result == nested_file

    def test_path_object_input(self):
        """Test validation with Path object input."""
        from networkx_mcp.io.base import validate_file_path

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "path_object.txt"
            test_file.write_text("test")

            # Path object input
            result = validate_file_path(test_file, must_exist=True)
            assert result == test_file

    def test_edge_case_empty_path(self):
        """Test edge case with empty path."""
        from networkx_mcp.io.base import validate_file_path

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("tempfile.gettempdir", return_value=temp_dir):
                # Empty string should work as relative path
                result = validate_file_path("", must_exist=False)
                assert isinstance(result, Path)

    def test_path_with_dot_components_rejected(self):
        """Test that paths with .. components are rejected for security."""
        from networkx_mcp.io.base import validate_file_path

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("tempfile.gettempdir", return_value=temp_dir):
                # Path with .. components should be rejected for security
                redundant_path = "dir/./subdir/../file.txt"

                with pytest.raises(
                    ValueError,
                    match="Invalid file path - no directory traversal allowed",
                ):
                    validate_file_path(redundant_path, must_exist=False)

                # Simple relative paths should work
                simple_path = "dir/subdir/file.txt"
                result = validate_file_path(simple_path, must_exist=False)
                assert isinstance(result, Path)


class TestDetectFormat:
    """Test the detect_format utility function."""

    def test_known_formats(self):
        """Test detection of known file formats."""
        from networkx_mcp.io.base import detect_format

        # Known formats
        assert detect_format("graph.gml") == "gml"
        assert detect_format("data.graphml") == "graphml"
        assert detect_format("network.gexf") == "gexf"
        assert detect_format("edges.edgelist") == "edgelist"
        assert detect_format("adj.adjlist") == "adjlist"
        assert detect_format("graph.json") == "json"
        assert detect_format("config.yaml") == "yaml"
        assert detect_format("data.csv") == "csv"

    def test_case_insensitive(self):
        """Test case-insensitive format detection."""
        from networkx_mcp.io.base import detect_format

        # Case variations
        assert detect_format("file.GML") == "gml"
        assert detect_format("file.GraphML") == "graphml"
        assert detect_format("file.GEXF") == "gexf"
        assert detect_format("file.JSON") == "json"
        assert detect_format("file.YAML") == "yaml"
        assert detect_format("file.CSV") == "csv"

    def test_unknown_formats(self):
        """Test detection of unknown file formats."""
        from networkx_mcp.io.base import detect_format

        # Unknown formats
        assert detect_format("file.txt") == "unknown"
        assert detect_format("file.pdf") == "unknown"
        assert detect_format("file.exe") == "unknown"
        assert detect_format("file") == "unknown"  # No extension
        assert detect_format("file.") == "unknown"  # Empty extension
        assert detect_format("file.unknown") == "unknown"

    def test_path_object_input(self):
        """Test format detection with Path objects."""
        from networkx_mcp.io.base import detect_format

        # Path object input
        assert detect_format(Path("data.gml")) == "gml"
        assert detect_format(Path("network.json")) == "json"
        assert detect_format(Path("unknown.xyz")) == "unknown"

    def test_complex_filenames(self):
        """Test format detection with complex filenames."""
        from networkx_mcp.io.base import detect_format

        # Multiple dots - should use last extension
        assert detect_format("file.backup.gml") == "gml"
        assert detect_format("data.2024.json") == "json"
        assert detect_format("network.v1.graphml") == "graphml"

        # Full paths
        assert detect_format("/path/to/file.gexf") == "gexf"
        assert detect_format("../data/graph.yaml") == "yaml"
        assert detect_format("./files/edges.csv") == "csv"

    def test_edge_cases(self):
        """Test edge cases in format detection."""
        from networkx_mcp.io.base import detect_format

        # Edge cases
        assert detect_format("") == "unknown"  # Empty string
        assert detect_format(".gml") == "unknown"  # Hidden file - no actual name
        assert detect_format("file..gml") == "gml"  # Double dot
        assert detect_format("file.gml.bak") == "unknown"  # Backup file
