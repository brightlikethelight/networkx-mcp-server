"""
Citation analysis and DOI resolution functions for academic networks.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import bibtexparser
import networkx as nx
import requests
from bibtexparser.bwriter import BibTexWriter


def resolve_doi(doi: str) -> Optional[Dict[str, Any]]:
    """Resolve DOI to publication metadata using CrossRef API."""
    if not doi:
        return None

    # Clean DOI format
    doi = doi.strip()
    if not doi.startswith("10."):
        if doi.startswith("doi:"):
            doi = doi[4:]
        elif doi.startswith("https://doi.org/"):
            doi = doi[16:]

    try:
        url = f"https://api.crossref.org/works/{doi}"
        headers = {
            "Accept": "application/json",
            "User-Agent": "NetworkX-MCP-Server/3.0.0 (mailto:support@networkx-mcp.org)",
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        work = data.get("message", {})

        # Extract key metadata
        return {
            "doi": work.get("DOI", doi),
            "title": work.get("title", [""])[0] if work.get("title") else "",
            "authors": [
                f"{author.get('given', '')} {author.get('family', '')}"
                for author in work.get("author", [])
            ],
            "journal": work.get("container-title", [""])[0]
            if work.get("container-title")
            else "",
            "year": work.get("published-print", {}).get("date-parts", [[None]])[0][0]
            or work.get("published-online", {}).get("date-parts", [[None]])[0][0],
            "citations": work.get("is-referenced-by-count", 0),
            "references": work.get("reference", []),
        }
    except Exception as e:
        print(f"Error resolving DOI {doi}: {e}")
        return None


def build_citation_network(
    graph_name: str,
    seed_dois: List[str],
    max_depth: int = 2,
    graphs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build citation network from seed DOIs using CrossRef API."""
    if graphs is None:
        graphs = {}

    if graph_name in graphs:
        raise ValueError(f"Graph '{graph_name}' already exists")

    # Create directed graph for citations
    citation_graph: nx.DiGraph[Any] = nx.DiGraph()
    processed = set[Any]()
    to_process = [(doi, 0) for doi in seed_dois]

    nodes_added = 0
    edges_added = 0

    while to_process and nodes_added < 1000:  # Limit to prevent overload
        current_doi, depth = to_process.pop(0)

        if current_doi in processed or depth > max_depth:
            continue

        processed.add(current_doi)

        # Resolve current DOI
        paper = resolve_doi(current_doi)
        if not paper:
            continue

        # Add node with metadata
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

    return {
        "created": graph_name,
        "type": "citation_network",
        "nodes": nodes_added,
        "edges": edges_added,
        "seed_dois": seed_dois,
        "max_depth": max_depth,
    }


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
    cited_papers = list[Any](graph.successors(seed_doi))

    # Find papers that cite the seed paper
    citing_papers = list[Any](graph.predecessors(seed_doi))

    # Calculate recommendation scores based on citation patterns
    recommendations = []

    # Score papers that are co-cited with seed paper
    for cited in cited_papers:
        # Find other papers that also cite this paper
        co_citing = list[Any](graph.predecessors(cited))

        for paper in co_citing:
            if paper != seed_doi and paper not in cited_papers:
                score = 1.0  # Base score for co-citation

                # Boost score based on citation count
                paper_data = graph.nodes.get(paper, {})
                citation_count = (
                    paper_data.get("citations", 0)
                    if isinstance(paper_data, dict[str, Any])
                    else 0
                )
                score += min(citation_count / 100, 2.0)  # Max boost of 2.0

                # Boost score based on recency
                year = (
                    paper_data.get("year")
                    if isinstance(paper_data, dict[str, Any])
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
                        if isinstance(paper_data, dict[str, Any])
                        else paper,
                        "authors": paper_data.get("authors", [])
                        if isinstance(paper_data, dict[str, Any])
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
