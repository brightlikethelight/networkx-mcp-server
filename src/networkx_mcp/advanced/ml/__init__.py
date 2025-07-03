"""
Machine Learning integration modules.

Provides compatibility exports for the old ml_integration module.
"""

# For backward compatibility
try:
    from ..ml_integration import MLIntegration, HAS_SKLEARN
except ImportError:
    class MLIntegration:
        """Placeholder MLIntegration for compatibility."""
    
    HAS_SKLEARN = False


__all__ = [
    'MLIntegration',
    'HAS_SKLEARN'
]
