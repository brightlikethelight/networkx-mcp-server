"""
Network flow modules.

Provides compatibility exports for the old network_flow module.
"""

# For backward compatibility
try:
    from ..network_flow import NetworkFlow
except ImportError:
    class NetworkFlow:
        """Placeholder NetworkFlow for compatibility."""


__all__ = [
    'NetworkFlow'
]
