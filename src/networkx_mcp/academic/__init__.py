"""
Academic plugin for NetworkX MCP Server.

This module provides specialized functions for academic research including:
- DOI resolution and citation network analysis
- Author impact metrics and collaboration patterns
- Research trend analysis and paper recommendations
- BibTeX export capabilities
"""

from .analytics import (
    analyze_author_impact,
    calculate_h_index,
    detect_research_trends,
    find_collaboration_patterns,
)
from .citations import (
    build_citation_network,
    export_bibtex,
    recommend_papers,
    resolve_doi,
)

__all__ = [
    # Citations
    "resolve_doi",
    "build_citation_network",
    "export_bibtex",
    "recommend_papers",
    # Analytics
    "calculate_h_index",
    "analyze_author_impact",
    "find_collaboration_patterns",
    "detect_research_trends",
]
