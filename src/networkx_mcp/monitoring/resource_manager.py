"""Resource monitoring utilities."""

import logging
from typing import Any, Dict

import psutil

logger = logging.getLogger(__name__)

def get_system_metrics() -> Dict[str, Any]:
    """Get current system resource metrics."""
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_available_mb": psutil.virtual_memory().available / (1024 * 1024)
    }

def check_memory_available(required_mb: int = 100) -> bool:
    """Check if enough memory is available."""
    available = psutil.virtual_memory().available / (1024 * 1024)
    return available >= required_mb
