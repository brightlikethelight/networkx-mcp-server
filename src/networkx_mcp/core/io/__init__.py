"""
I/O handler modules.

Provides compatibility exports for the old io_handlers module.
"""

from typing import Any

# For backward compatibility, re-export GraphIOHandler from core.io_handlers
try:
    from ..io_handlers import GraphIOHandler
except ImportError:
    # If the original file doesn't exist, create a placeholder
    class GraphIOHandler:
        """Placeholder GraphIOHandler for compatibility."""

        @staticmethod
        def import_from_file(file_path: Any, format_type: Any) -> None:
            """Placeholder import method."""
            raise NotImplementedError("GraphIOHandler.import_from_file not implemented")

        @staticmethod
        def export_to_file(graph: Any, file_path: Any, format_type: Any) -> None:
            """Placeholder export method."""
            raise NotImplementedError("GraphIOHandler.export_to_file not implemented")


# Import all handlers explicitly
# Note: These have duplicate "Handler" in their names (e.g., GraphmlHandlerHandler)
# This should be fixed in a future refactoring
from .base_handler import BaseHandlerHandler
from .csv_handler import CsvHandlerHandler
from .excel_handler import ExcelHandlerHandler
from .gml_handler import GmlHandlerHandler
from .graphml_handler import GraphmlHandlerHandler
from .json_handler import JsonHandlerHandler

# For backward compatibility, alias them to expected names
JsonHandler = JsonHandlerHandler
GmlHandler = GmlHandlerHandler
GraphmlHandler = GraphmlHandlerHandler
CsvHandler = CsvHandlerHandler
ExcelHandler = ExcelHandlerHandler

# Export all
__all__ = [
    "GraphIOHandler",
    "BaseHandlerHandler",
    "JsonHandlerHandler",
    "JsonHandler",
    "GmlHandlerHandler",
    "GmlHandler",
    "GraphmlHandlerHandler",
    "GraphmlHandler",
    "CsvHandlerHandler",
    "CsvHandler",
    "ExcelHandlerHandler",
    "ExcelHandler",
]
