"""Resource monitoring utilities."""

import logging
from typing import Any, Dict

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

logger = logging.getLogger(__name__)


def get_system_metrics() -> Dict[str, Any]:
    """Get current system resource metrics."""
    if not HAS_PSUTIL:
        return {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "memory_available_mb": 0.0,
            "error": "psutil not available",
        }
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_available_mb": psutil.virtual_memory().available / (1024 * 1024),
    }


def check_memory_available(required_mb: int = 100) -> bool:
    """Check if enough memory is available."""
    if not HAS_PSUTIL:
        return True  # Assume enough memory if we can't check
    available = psutil.virtual_memory().available / (1024 * 1024)
    return available >= required_mb
