"""
Directed graph analysis modules.

Provides compatibility exports for the old directed_analysis module.
"""

# For backward compatibility, try to import from the original module
try:
    from ..directed_analysis import DirectedAnalysis
except ImportError:
    # If the original file doesn't exist, create a placeholder
    class DirectedAnalysis:
        """Placeholder DirectedAnalysis for compatibility."""
        
        @staticmethod
        def dag_analysis(graph, **params):
            """Placeholder DAG analysis method."""
            raise NotImplementedError("DirectedAnalysis.dag_analysis not implemented")


__all__ = [
    'DirectedAnalysis',
]
