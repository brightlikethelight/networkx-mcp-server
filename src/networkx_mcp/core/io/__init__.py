"""
I/O handler modules.

Provides compatibility exports for the old io_handlers module.
"""

# For backward compatibility, re-export GraphIOHandler from core.io_handlers
try:
    from ..io_handlers import GraphIOHandler
except ImportError:
    # If the original file doesn't exist, create a placeholder
    class GraphIOHandler:
        """Placeholder GraphIOHandler for compatibility."""

        @staticmethod
        def import_from_file(file_path, format_type):
            """Placeholder import method."""
            raise NotImplementedError("GraphIOHandler.import_from_file not implemented")

        @staticmethod
        def export_to_file(graph, file_path, format_type):
            """Placeholder export method."""
            raise NotImplementedError("GraphIOHandler.export_to_file not implemented")


# Import all handlers
from .base_handler import *
from .csv_handler import *
from .excel_handler import *
from .gml_handler import *
from .graphml_handler import *
from .json_handler import *

# For backward compatibility
__all__ = [
    "GraphIOHandler",
    "JsonHandler",
    "GmlHandler",
    "GraphmlHandler",
    "CsvHandler",
    "ExcelHandler",
]
