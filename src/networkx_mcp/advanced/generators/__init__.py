"""
Graph generator modules.

Provides compatibility exports for the old generators module.
"""

# For backward compatibility
try:
    from ..generators import GraphGenerators
except ImportError:
    class GraphGenerators:
        """Placeholder GraphGenerators for compatibility."""


__all__ = [
    'GraphGenerators'
]
