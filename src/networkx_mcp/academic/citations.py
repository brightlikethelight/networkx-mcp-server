"""
Citation analysis and DOI resolution functions for academic networks.
"""

import logging
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import bibtexparser
import networkx as nx
import requests
from bibtexparser.bwriter import BibTexWriter
from requests.exceptions import HTTPError, RequestException, Timeout

logger = logging.getLogger(__name__)


def resolve_doi(
    doi: str, retry_count: int = 3, retry_delay: float = 1.0
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Resolve DOI to publication metadata using CrossRef API.

    Args:
        doi: The DOI to resolve
        retry_count: Number of retry attempts for network failures
        retry_delay: Delay between retries in seconds

    Returns:
        Tuple of (metadata_dict, error_message)
        - metadata_dict: Publication metadata if successful, None if failed
        - error_message: Error description if failed, None if successful
    """
    if not doi:
        return None, "Empty DOI provided"

    # Clean DOI format
    doi = doi.strip()
    if not doi.startswith("10."):
        if doi.startswith("doi:"):
            doi = doi[4:]
        elif doi.startswith("https://doi.org/"):
            doi = doi[16:]
        elif doi.startswith("http://doi.org/"):
            doi = doi[15:]

    # Validate DOI format
    if not doi.startswith("10.") or "/" not in doi:
        return None, f"Invalid DOI format: {doi}"

    url = f"https://api.crossref.org/works/{doi}"
    headers = {
        "Accept": "application/json",
        "User-Agent": "NetworkX-MCP-Server/3.0.0 (mailto:support@networkx-mcp.org)",
    }

    last_error = None

    for attempt in range(retry_count):
        try:
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 404:
                return None, f"DOI not found: {doi}"
            elif response.status_code == 429:
                # Rate limited - wait and retry
                wait_time = retry_delay * (2**attempt)  # Exponential backoff
                last_error = (
                    f"Rate limited on DOI {doi} (attempt {attempt + 1}/{retry_count})"
                )
                logger.warning(f"Rate limited on DOI {doi}, waiting {wait_time}s")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()
            work = data.get("message", {})

            # Extract key metadata with safe access
            metadata = {
                "doi": work.get("DOI", doi),
                "title": work.get("title", [""])[0] if work.get("title") else "",
                "authors": [
                    f"{author.get('given', '')} {author.get('family', '')}".strip()
                    for author in work.get("author", [])
                ],
                "journal": work.get("container-title", [""])[0]
                if work.get("container-title")
                else "",
                "year": work.get("published-print", {}).get("date-parts", [[None]])[0][
                    0
                ]
                or work.get("published-online", {}).get("date-parts", [[None]])[0][0],
                "citations": work.get("is-referenced-by-count", 0),
                "references": work.get("reference", []),
            }

            return metadata, None

        except Timeout:
            last_error = (
                f"Timeout resolving DOI {doi} (attempt {attempt + 1}/{retry_count})"
            )
            logger.warning(last_error)
        except HTTPError as e:
            last_error = f"HTTP error {e.response.status_code} for DOI {doi}: {str(e)}"
            logger.error(last_error)
            if e.response.status_code != 429:  # Don't retry non-rate-limit errors
                break
        except RequestException as e:
            last_error = f"Network error resolving DOI {doi}: {str(e)}"
            logger.error(last_error)
        except (KeyError, ValueError, TypeError) as e:
            last_error = f"Invalid response format for DOI {doi}: {str(e)}"
            logger.error(last_error)
            break  # Don't retry parsing errors
        except Exception as e:
            last_error = f"Unexpected error resolving DOI {doi}: {str(e)}"
            logger.error(last_error)
            break

        if attempt < retry_count - 1:
            time.sleep(retry_delay)

    return None, last_error


def build_citation_network(
    graph_name: str,
    seed_dois: List[str],
    max_depth: int = 2,
    graphs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build citation network from seed DOIs using CrossRef API.

    Args:
        graph_name: Name for the new graph
        seed_dois: List of DOIs to start building from
        max_depth: Maximum depth to traverse citations
        graphs: Dictionary to store the graph in

    Returns:
        Dictionary with build statistics and any errors encountered
    """
    if graphs is None:
        graphs = {}

    if graph_name in graphs:
        raise ValueError(f"Graph '{graph_name}' already exists")

    # Create directed graph for citations
    citation_graph: nx.DiGraph = nx.DiGraph()
    processed = set()
    to_process = deque([(doi, 0) for doi in seed_dois])

    nodes_added = 0
    edges_added = 0
    errors = []
    resolution_failures = 0

    while to_process and nodes_added < 1000:  # Limit to prevent overload
        current_doi, depth = to_process.popleft()

        if current_doi in processed or depth > max_depth:
            continue

        processed.add(current_doi)

        # Resolve current DOI with error handling
        paper, error = resolve_doi(current_doi)
        if error:
            resolution_failures += 1
            if resolution_failures <= 10:  # Limit error reporting
                errors.append(f"Failed to resolve {current_doi}: {error}")
            if not paper:  # Complete failure
                continue

        # Add node with metadata
        if paper:
            citation_graph.add_node(current_doi, **paper)
            nodes_added += 1

            # Add citation edges (this paper cites others)
            for ref in paper.get("references", []):
                ref_doi = ref.get("DOI")
                if ref_doi and ref_doi not in processed:
                    citation_graph.add_edge(current_doi, ref_doi)
                    edges_added += 1

                    if depth < max_depth:
                        to_process.append((ref_doi, depth + 1))

    graphs[graph_name] = citation_graph

    result = {
        "created": graph_name,
        "type": "citation_network",
        "nodes": nodes_added,
        "edges": edges_added,
        "seed_dois": seed_dois,
        "max_depth": max_depth,
        "resolution_failures": resolution_failures,
    }

    if errors:
        result["errors"] = errors[:10]  # Limit to first 10 errors
        result["total_errors"] = len(errors)

    return result


def export_bibtex(
    graph_name: str, graphs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Export citation network as BibTeX format."""
    if graphs is None:
        graphs = {}

    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")

    graph = graphs[graph_name]

    # Create BibTeX database
    bib_db = bibtexparser.bibdatabase.BibDatabase()
    bib_db.entries = []

    for node in graph.nodes(data=True):
        node_id, data = node

        # Create BibTeX entry
        entry = {
            "ENTRYTYPE": "article",
            "ID": node_id.replace("/", "_").replace(".", "_"),
            "title": data.get("title", ""),
            "author": " and ".join(data.get("authors", [])),
            "journal": data.get("journal", ""),
            "year": str(data.get("year", "")),
            "doi": data.get("doi", ""),
            "note": f"Citations: {data.get('citations', 0)}",
        }

        bib_db.entries.append(entry)

    # Generate BibTeX string
    writer = BibTexWriter()
    bibtex_str = writer.write(bib_db)

    return {
        "format": "bibtex",
        "entries": len(bib_db.entries),
        "bibtex_data": bibtex_str,
    }


def recommend_papers(
    graph_name: str,
    seed_doi: str,
    max_recommendations: int = 10,
    graphs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Recommend papers based on citation network analysis."""
    if graphs is None:
        graphs = {}

    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")

    graph = graphs[graph_name]

    # Handle alternative parameter names for compatibility
    # Check if seed_doi is actually present, if not return empty recommendations
    if seed_doi not in graph:
        # Return valid structure even when seed not found
        return {
            "seed_paper": seed_doi,
            "recommendations": [],
            "total_found": 0,
            "based_on": {
                "cited_papers": 0,
                "citing_papers": 0,
            },
            "note": f"Seed paper '{seed_doi}' not found in graph",
        }

    # Find papers cited by seed paper
    cited_papers = list(graph.successors(seed_doi))

    # Find papers that cite the seed paper
    citing_papers = list(graph.predecessors(seed_doi))

    # Calculate recommendation scores based on citation patterns
    recommendations = []

    # Score papers that are co-cited with seed paper
    for cited in cited_papers:
        # Find other papers that also cite this paper
        co_citing = list(graph.predecessors(cited))

        for paper in co_citing:
            if paper != seed_doi and paper not in cited_papers:
                score = 1.0  # Base score for co-citation

                # Boost score based on citation count
                paper_data = graph.nodes.get(paper, {})
                citation_count = (
                    paper_data.get("citations", 0)
                    if isinstance(paper_data, dict)
                    else 0
                )
                score += min(citation_count / 100, 2.0)  # Max boost of 2.0

                # Boost score based on recency
                year = (
                    paper_data.get("year")
                    if isinstance(paper_data, dict)
                    else None
                )
                if year:
                    current_year = datetime.now().year
                    recency_score = max(0, (year - (current_year - 10)) / 10)
                    score += recency_score

                recommendations.append(
                    {
                        "paper": paper,  # Use 'paper' for compatibility
                        "doi": paper,
                        "title": paper_data.get("title", paper)
                        if isinstance(paper_data, dict)
                        else paper,
                        "authors": paper_data.get("authors", [])
                        if isinstance(paper_data, dict)
                        else [],
                        "year": year,
                        "citations": citation_count,
                        "score": score,
                        "reason": "co-citation",
                    }
                )

    # Sort by score and return top recommendations
    recommendations.sort(key=lambda x: x["score"], reverse=True)

    return {
        "seed_paper": seed_doi,
        "recommendations": recommendations[:max_recommendations],
        "total_found": len(recommendations),
        "based_on": {
            "cited_papers": len(cited_papers),
            "citing_papers": len(citing_papers),
        },
    }
