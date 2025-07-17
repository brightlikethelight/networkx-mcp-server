#!/usr/bin/env python3
"""
Actually Minimal NetworkX MCP Server
Only 150 lines. No BS. Just works.
"""

import asyncio
import json
import sys
from typing import Any, Dict, List, Optional, Tuple
import re
from datetime import datetime
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import requests
from dateutil.parser import parse as parse_date
import bibtexparser
from bibtexparser.bwriter import BibTexWriter

matplotlib.use('Agg')  # Use non-interactive backend
import base64
import csv
import io

import networkx.algorithms.community as nx_comm

# Global state - simple and effective
graphs: Dict[str, nx.Graph] = {}

# Compatibility exports for tests
def create_graph(name: str, directed: bool = False):
    """Create a graph - compatibility function."""
    graphs[name] = nx.DiGraph() if directed else nx.Graph()
    return {"created": name, "type": "directed" if directed else "undirected"}

def add_nodes(graph_name: str, nodes: List):
    """Add nodes - compatibility function."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    graph.add_nodes_from(nodes)
    return {"added": len(nodes), "total": graph.number_of_nodes()}

def add_edges(graph_name: str, edges: List):
    """Add edges - compatibility function."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    edge_tuples = [tuple(e) for e in edges]
    graph.add_edges_from(edge_tuples)
    return {"added": len(edge_tuples), "total": graph.number_of_edges()}

def get_graph_info(graph_name: str):
    """Get graph info - compatibility function."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "directed": graph.is_directed()
    }

def shortest_path(graph_name: str, source, target):
    """Find shortest path - compatibility function."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    path = nx.shortest_path(graph, source, target)
    return {"path": path, "length": len(path) - 1}

def degree_centrality(graph_name: str):
    """Calculate degree centrality - compatibility function."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    centrality = nx.degree_centrality(graph)
    # Convert to serializable format and sort by centrality
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return {
        "centrality": dict(sorted_nodes[:10]),  # Top 10 nodes
        "most_central": sorted_nodes[0] if sorted_nodes else None
    }

def betweenness_centrality(graph_name: str):
    """Calculate betweenness centrality - compatibility function."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    centrality = nx.betweenness_centrality(graph)
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return {
        "centrality": dict(sorted_nodes[:10]),  # Top 10 nodes
        "most_central": sorted_nodes[0] if sorted_nodes else None
    }

def connected_components(graph_name: str):
    """Find connected components - compatibility function."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    if graph.is_directed():
        components = list(nx.weakly_connected_components(graph))
    else:
        components = list(nx.connected_components(graph))

    # Convert sets to lists for JSON serialization
    components_list = [list(comp) for comp in components]
    components_list.sort(key=len, reverse=True)  # Largest first

    return {
        "num_components": len(components_list),
        "component_sizes": [len(comp) for comp in components_list],
        "largest_component": components_list[0] if components_list else []
    }

def pagerank(graph_name: str):
    """Calculate PageRank - compatibility function."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]
    pr = nx.pagerank(graph)
    sorted_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    return {
        "pagerank": dict(sorted_nodes[:10]),  # Top 10 nodes
        "highest_rank": sorted_nodes[0] if sorted_nodes else None
    }

def visualize_graph(graph_name: str, layout: str = "spring"):
    """Visualize graph and return as base64 image - compatibility function."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]

    plt.figure(figsize=(10, 8))

    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(graph)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph)
    else:
        pos = nx.spring_layout(graph)

    # Draw the graph
    nx.draw(graph, pos, with_labels=True, node_color='lightblue',
            node_size=500, font_size=10, font_weight='bold',
            edge_color='gray', arrows=True)

    # Save to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    plt.close()

    # Convert to base64
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return {
        "image": f"data:image/png;base64,{image_base64}",
        "format": "png",
        "layout": layout
    }

def import_csv(graph_name: str, csv_data: str, directed: bool = False):
    """Import graph from CSV edge list - compatibility function."""
    # Parse CSV data
    reader = csv.reader(io.StringIO(csv_data))
    edges = []

    for row in reader:
        if len(row) >= 2:
            # Handle both numeric and string nodes
            try:
                source = int(row[0])
            except:
                source = row[0].strip()
            try:
                target = int(row[1])
            except:
                target = row[1].strip()

            edges.append((source, target))

    # Create graph
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_edges_from(edges)
    graphs[graph_name] = graph

    return {
        "imported": graph_name,
        "type": "directed" if directed else "undirected",
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges()
    }

def export_json(graph_name: str):
    """Export graph as JSON - compatibility function."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]

    # Convert to node-link format
    data = nx.node_link_data(graph)

    return {
        "graph_data": data,
        "format": "node-link",
        "nodes": len(data["nodes"]),
        "edges": len(data["links"])
    }

def community_detection(graph_name: str):
    """Detect communities in the graph - compatibility function."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    graph = graphs[graph_name]

    # Use Louvain method for community detection
    communities = nx_comm.louvain_communities(graph)

    # Convert to list format
    communities_list = [list(comm) for comm in communities]
    communities_list.sort(key=len, reverse=True)  # Largest first

    # Create node to community mapping
    node_community = {}
    for i, comm in enumerate(communities_list):
        for node in comm:
            node_community[node] = i

    return {
        "num_communities": len(communities_list),
        "community_sizes": [len(comm) for comm in communities_list],
        "largest_community": communities_list[0] if communities_list else [],
        "node_community_map": dict(list(node_community.items())[:20])  # First 20 nodes
    }

# Academic-focused functions for citation analysis
def resolve_doi(doi: str) -> Optional[Dict]:
    """Resolve DOI to publication metadata using CrossRef API."""
    if not doi:
        return None
    
    # Clean DOI format
    doi = doi.strip()
    if not doi.startswith('10.'):
        if doi.startswith('doi:'):
            doi = doi[4:]
        elif doi.startswith('https://doi.org/'):
            doi = doi[16:]
    
    try:
        url = f"https://api.crossref.org/works/{doi}"
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'NetworkX-MCP-Server/2.2.0 (mailto:support@networkx-mcp.org)'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        work = data.get('message', {})
        
        # Extract key metadata
        return {
            'doi': work.get('DOI', doi),
            'title': work.get('title', [''])[0] if work.get('title') else '',
            'authors': [f"{author.get('given', '')} {author.get('family', '')}" 
                       for author in work.get('author', [])],
            'journal': work.get('container-title', [''])[0] if work.get('container-title') else '',
            'year': work.get('published-print', {}).get('date-parts', [[None]])[0][0] or 
                   work.get('published-online', {}).get('date-parts', [[None]])[0][0],
            'citations': work.get('is-referenced-by-count', 0),
            'references': work.get('reference', [])
        }
    except Exception as e:
        print(f"Error resolving DOI {doi}: {e}")
        return None

def build_citation_network(graph_name: str, seed_dois: List[str], max_depth: int = 2) -> dict:
    """Build citation network from seed DOIs using CrossRef API."""
    if graph_name in graphs:
        raise ValueError(f"Graph '{graph_name}' already exists")
    
    # Create directed graph for citations
    citation_graph = nx.DiGraph()
    processed = set()
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
        for ref in paper.get('references', []):
            ref_doi = ref.get('DOI')
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
        "max_depth": max_depth
    }

def calculate_h_index(author_citations: List[int]) -> int:
    """Calculate h-index from list of citation counts."""
    citations = sorted(author_citations, reverse=True)
    h_index = 0
    
    for i, citations_count in enumerate(citations):
        if citations_count >= i + 1:
            h_index = i + 1
        else:
            break
    
    return h_index

def analyze_author_impact(graph_name: str, author_name: str) -> dict:
    """Analyze author impact in citation network."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    
    graph = graphs[graph_name]
    
    # Find papers by author
    author_papers = []
    citation_counts = []
    
    for node in graph.nodes(data=True):
        node_id, data = node
        authors = data.get('authors', [])
        
        # Simple name matching (could be improved with author disambiguation)
        if any(author_name.lower() in author.lower() for author in authors):
            author_papers.append(node_id)
            citation_counts.append(data.get('citations', 0))
    
    if not author_papers:
        return {
            "author": author_name,
            "papers_found": 0,
            "h_index": 0,
            "total_citations": 0,
            "average_citations": 0
        }
    
    h_index = calculate_h_index(citation_counts)
    total_citations = sum(citation_counts)
    
    return {
        "author": author_name,
        "papers_found": len(author_papers),
        "h_index": h_index,
        "total_citations": total_citations,
        "average_citations": total_citations / len(author_papers) if author_papers else 0,
        "papers": author_papers[:10]  # Show first 10 papers
    }

def find_collaboration_patterns(graph_name: str) -> dict:
    """Find collaboration patterns in citation network."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    
    graph = graphs[graph_name]
    
    # Build co-authorship network
    coauthor_graph = nx.Graph()
    collaboration_counts = defaultdict(int)
    
    for node in graph.nodes(data=True):
        node_id, data = node
        authors = data.get('authors', [])
        
        # Add co-authorship edges
        for i, author1 in enumerate(authors):
            for author2 in authors[i+1:]:
                if author1 and author2:
                    coauthor_graph.add_edge(author1, author2)
                    collaboration_counts[(author1, author2)] += 1
    
    # Find most frequent collaborators
    top_collaborations = sorted(
        collaboration_counts.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:10]
    
    # Calculate network metrics
    if coauthor_graph.number_of_nodes() > 0:
        centrality = nx.degree_centrality(coauthor_graph)
        top_authors = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    else:
        top_authors = []
    
    return {
        "coauthorship_network": {
            "nodes": coauthor_graph.number_of_nodes(),
            "edges": coauthor_graph.number_of_edges()
        },
        "top_collaborations": [
            {"authors": list(authors), "collaborations": count}
            for authors, count in top_collaborations
        ],
        "most_central_authors": [
            {"author": author, "centrality": centrality}
            for author, centrality in top_authors
        ]
    }

def detect_research_trends(graph_name: str, time_window: int = 5) -> dict:
    """Detect research trends in citation network over time."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    
    graph = graphs[graph_name]
    
    # Group papers by year
    year_counts = defaultdict(int)
    yearly_citations = defaultdict(list)
    
    for node in graph.nodes(data=True):
        node_id, data = node
        year = data.get('year')
        citations = data.get('citations', 0)
        
        if year:
            year_counts[year] += 1
            yearly_citations[year].append(citations)
    
    # Calculate trends
    years = sorted(year_counts.keys())
    if len(years) < 2:
        return {
            "trend": "insufficient_data",
            "years_analyzed": len(years),
            "publication_trend": [],
            "citation_trend": []
        }
    
    # Publication trend
    pub_trend = [{"year": year, "publications": year_counts[year]} for year in years]
    
    # Citation trend
    citation_trend = [
        {
            "year": year, 
            "total_citations": sum(yearly_citations[year]),
            "average_citations": sum(yearly_citations[year]) / len(yearly_citations[year]) if yearly_citations[year] else 0
        }
        for year in years
    ]
    
    # Determine overall trend
    recent_years = years[-time_window:]
    early_years = years[:time_window]
    
    if len(recent_years) >= 2 and len(early_years) >= 2:
        recent_avg = sum(year_counts[y] for y in recent_years) / len(recent_years)
        early_avg = sum(year_counts[y] for y in early_years) / len(early_years)
        
        if recent_avg > early_avg * 1.2:
            trend = "increasing"
        elif recent_avg < early_avg * 0.8:
            trend = "decreasing"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"
    
    return {
        "trend": trend,
        "years_analyzed": len(years),
        "publication_trend": pub_trend,
        "citation_trend": citation_trend,
        "time_window": time_window
    }

def export_bibtex(graph_name: str) -> dict:
    """Export citation network as BibTeX format."""
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
            'ENTRYTYPE': 'article',
            'ID': node_id.replace('/', '_').replace('.', '_'),
            'title': data.get('title', ''),
            'author': ' and '.join(data.get('authors', [])),
            'journal': data.get('journal', ''),
            'year': str(data.get('year', '')),
            'doi': data.get('doi', ''),
            'note': f"Citations: {data.get('citations', 0)}"
        }
        
        bib_db.entries.append(entry)
    
    # Generate BibTeX string
    writer = BibTexWriter()
    bibtex_str = writer.write(bib_db)
    
    return {
        "format": "bibtex",
        "entries": len(bib_db.entries),
        "bibtex_data": bibtex_str
    }

def recommend_papers(graph_name: str, seed_doi: str, max_recommendations: int = 10) -> dict:
    """Recommend papers based on citation network analysis."""
    if graph_name not in graphs:
        raise ValueError(f"Graph '{graph_name}' not found")
    
    graph = graphs[graph_name]
    
    if seed_doi not in graph:
        raise ValueError(f"DOI '{seed_doi}' not found in graph")
    
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
                paper_data = graph.nodes[paper]
                citation_count = paper_data.get('citations', 0)
                score += min(citation_count / 100, 2.0)  # Max boost of 2.0
                
                # Boost score based on recency
                year = paper_data.get('year')
                if year:
                    current_year = datetime.now().year
                    recency_score = max(0, (year - (current_year - 10)) / 10)
                    score += recency_score
                
                recommendations.append({
                    'doi': paper,
                    'title': paper_data.get('title', ''),
                    'authors': paper_data.get('authors', []),
                    'year': year,
                    'citations': citation_count,
                    'score': score,
                    'reason': 'co-citation'
                })
    
    # Sort by score and return top recommendations
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    return {
        "seed_paper": seed_doi,
        "recommendations": recommendations[:max_recommendations],
        "total_found": len(recommendations),
        "based_on": {
            "cited_papers": len(cited_papers),
            "citing_papers": len(citing_papers)
        }
    }

class NetworkXMCPServer:
    """Minimal MCP server - no unnecessary abstraction."""

    def __init__(self):
        self.running = True
        self.mcp = self  # For test compatibility
        self.graphs = graphs  # Reference to global graphs
        
    def tool(self, func):
        """Mock tool decorator for test compatibility."""
        return func

    async def handle_request(self, request: dict) -> dict:
        """Route requests to handlers."""
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id")

        # Route to appropriate handler
        if method == "initialize":
            result = {"protocolVersion": "2024-11-05", "serverInfo": {"name": "networkx-minimal"}}
        elif method == "initialized":
            result = {}  # Just acknowledge
        elif method == "tools/list":
            result = {"tools": self._get_tools()}
        elif method == "tools/call":
            result = await self._call_tool(params)
        else:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown method: {method}"}}

        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def _get_tools(self) -> List[Dict[str, Any]]:
        """List available tools."""
        return [
            {
                "name": "create_graph",
                "description": "Create a new graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "directed": {"type": "boolean", "default": False}
                    },
                    "required": ["name"]
                }
            },
            {
                "name": "add_nodes",
                "description": "Add nodes to a graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph": {"type": "string"},
                        "nodes": {"type": "array", "items": {"type": ["string", "number"]}}
                    },
                    "required": ["graph", "nodes"]
                }
            },
            {
                "name": "add_edges",
                "description": "Add edges to a graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph": {"type": "string"},
                        "edges": {"type": "array", "items": {"type": "array", "items": {"type": ["string", "number"]}}}
                    },
                    "required": ["graph", "edges"]
                }
            },
            {
                "name": "shortest_path",
                "description": "Find shortest path between nodes",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph": {"type": "string"},
                        "source": {"type": ["string", "number"]},
                        "target": {"type": ["string", "number"]}
                    },
                    "required": ["graph", "source", "target"]
                }
            },
            {
                "name": "get_info",
                "description": "Get graph information",
                "inputSchema": {
                    "type": "object",
                    "properties": {"graph": {"type": "string"}},
                    "required": ["graph"]
                }
            },
            {
                "name": "degree_centrality",
                "description": "Calculate degree centrality for all nodes",
                "inputSchema": {
                    "type": "object",
                    "properties": {"graph": {"type": "string"}},
                    "required": ["graph"]
                }
            },
            {
                "name": "betweenness_centrality",
                "description": "Calculate betweenness centrality for all nodes",
                "inputSchema": {
                    "type": "object",
                    "properties": {"graph": {"type": "string"}},
                    "required": ["graph"]
                }
            },
            {
                "name": "connected_components",
                "description": "Find connected components in the graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {"graph": {"type": "string"}},
                    "required": ["graph"]
                }
            },
            {
                "name": "pagerank",
                "description": "Calculate PageRank for all nodes",
                "inputSchema": {
                    "type": "object",
                    "properties": {"graph": {"type": "string"}},
                    "required": ["graph"]
                }
            },
            {
                "name": "community_detection",
                "description": "Detect communities in the graph using Louvain method",
                "inputSchema": {
                    "type": "object",
                    "properties": {"graph": {"type": "string"}},
                    "required": ["graph"]
                }
            },
            {
                "name": "visualize_graph",
                "description": "Create a visualization of the graph",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph": {"type": "string"},
                        "layout": {
                            "type": "string",
                            "enum": ["spring", "circular", "kamada_kawai"],
                            "default": "spring"
                        }
                    },
                    "required": ["graph"]
                }
            },
            {
                "name": "import_csv",
                "description": "Import graph from CSV edge list (format: source,target per line)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph": {"type": "string"},
                        "csv_data": {"type": "string"},
                        "directed": {"type": "boolean", "default": False}
                    },
                    "required": ["graph", "csv_data"]
                }
            },
            {
                "name": "export_json",
                "description": "Export graph as JSON in node-link format",
                "inputSchema": {
                    "type": "object",
                    "properties": {"graph": {"type": "string"}},
                    "required": ["graph"]
                }
            },
            {
                "name": "build_citation_network",
                "description": "Build citation network from DOIs using CrossRef API",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph": {"type": "string"},
                        "seed_dois": {"type": "array", "items": {"type": "string"}},
                        "max_depth": {"type": "integer", "default": 2}
                    },
                    "required": ["graph", "seed_dois"]
                }
            },
            {
                "name": "analyze_author_impact",
                "description": "Analyze author impact metrics including h-index",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph": {"type": "string"},
                        "author_name": {"type": "string"}
                    },
                    "required": ["graph", "author_name"]
                }
            },
            {
                "name": "find_collaboration_patterns",
                "description": "Find collaboration patterns in citation network",
                "inputSchema": {
                    "type": "object",
                    "properties": {"graph": {"type": "string"}},
                    "required": ["graph"]
                }
            },
            {
                "name": "detect_research_trends",
                "description": "Detect research trends over time",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph": {"type": "string"},
                        "time_window": {"type": "integer", "default": 5}
                    },
                    "required": ["graph"]
                }
            },
            {
                "name": "export_bibtex",
                "description": "Export citation network as BibTeX format",
                "inputSchema": {
                    "type": "object",
                    "properties": {"graph": {"type": "string"}},
                    "required": ["graph"]
                }
            },
            {
                "name": "recommend_papers",
                "description": "Recommend papers based on citation network analysis",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph": {"type": "string"},
                        "seed_doi": {"type": "string"},
                        "max_recommendations": {"type": "integer", "default": 10}
                    },
                    "required": ["graph", "seed_doi"]
                }
            },
            {
                "name": "resolve_doi",
                "description": "Resolve DOI to publication metadata using CrossRef API",
                "inputSchema": {
                    "type": "object",
                    "properties": {"doi": {"type": "string"}},
                    "required": ["doi"]
                }
            }
        ]

    async def _call_tool(self, params: dict) -> dict:
        """Execute a tool."""
        tool_name = params.get("name")
        args = params.get("arguments", {})

        try:
            if tool_name == "create_graph":
                name = args["name"]
                directed = args.get("directed", False)
                graphs[name] = nx.DiGraph() if directed else nx.Graph()
                result = {"created": name, "type": "directed" if directed else "undirected"}

            elif tool_name == "add_nodes":
                graph_name = args["graph"]
                if graph_name not in graphs:
                    raise ValueError(f"Graph '{graph_name}' not found. Available graphs: {list(graphs.keys())}")
                graph = graphs[graph_name]
                graph.add_nodes_from(args["nodes"])
                result = {"added": len(args["nodes"]), "total": graph.number_of_nodes()}

            elif tool_name == "add_edges":
                graph_name = args["graph"]
                if graph_name not in graphs:
                    raise ValueError(f"Graph '{graph_name}' not found. Available graphs: {list(graphs.keys())}")
                graph = graphs[graph_name]
                edges = [tuple(e) for e in args["edges"]]
                graph.add_edges_from(edges)
                result = {"added": len(edges), "total": graph.number_of_edges()}

            elif tool_name == "shortest_path":
                graph_name = args["graph"]
                if graph_name not in graphs:
                    raise ValueError(f"Graph '{graph_name}' not found. Available graphs: {list(graphs.keys())}")
                graph = graphs[graph_name]
                path = nx.shortest_path(graph, args["source"], args["target"])
                result = {"path": path, "length": len(path) - 1}

            elif tool_name == "get_info":
                graph_name = args["graph"]
                if graph_name not in graphs:
                    raise ValueError(f"Graph '{graph_name}' not found. Available graphs: {list(graphs.keys())}")
                graph = graphs[graph_name]
                result = {
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "directed": graph.is_directed()
                }

            elif tool_name == "degree_centrality":
                result = degree_centrality(args["graph"])

            elif tool_name == "betweenness_centrality":
                result = betweenness_centrality(args["graph"])

            elif tool_name == "connected_components":
                result = connected_components(args["graph"])

            elif tool_name == "pagerank":
                result = pagerank(args["graph"])

            elif tool_name == "community_detection":
                result = community_detection(args["graph"])

            elif tool_name == "visualize_graph":
                layout = args.get("layout", "spring")
                result = visualize_graph(args["graph"], layout)

            elif tool_name == "import_csv":
                result = import_csv(args["graph"], args["csv_data"], args.get("directed", False))

            elif tool_name == "export_json":
                result = export_json(args["graph"])
            
            elif tool_name == "build_citation_network":
                result = build_citation_network(
                    args["graph"], 
                    args["seed_dois"], 
                    args.get("max_depth", 2)
                )
            
            elif tool_name == "analyze_author_impact":
                result = analyze_author_impact(args["graph"], args["author_name"])
            
            elif tool_name == "find_collaboration_patterns":
                result = find_collaboration_patterns(args["graph"])
            
            elif tool_name == "detect_research_trends":
                result = detect_research_trends(
                    args["graph"], 
                    args.get("time_window", 5)
                )
            
            elif tool_name == "export_bibtex":
                result = export_bibtex(args["graph"])
            
            elif tool_name == "recommend_papers":
                result = recommend_papers(
                    args["graph"], 
                    args["seed_doi"], 
                    args.get("max_recommendations", 10)
                )
            
            elif tool_name == "resolve_doi":
                result = resolve_doi(args["doi"])
                if result is None:
                    raise ValueError(f"Could not resolve DOI: {args['doi']}")

            else:
                raise ValueError(f"Unknown tool: {tool_name}")

            return {"content": [{"type": "text", "text": json.dumps(result)}]}

        except Exception as e:
            return {"content": [{"type": "text", "text": f"Error: {str(e)}"}], "isError": True}

    async def run(self):
        """Main server loop - read stdin, write stdout."""
        while self.running:
            try:
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break

                request = json.loads(line.strip())
                response = await self.handle_request(request)
                print(json.dumps(response), flush=True)

            except Exception as e:
                print(json.dumps({"error": str(e)}), file=sys.stderr, flush=True)

# Create module-level mcp instance for test compatibility
mcp = NetworkXMCPServer()

def main():
    """Main entry point for the NetworkX MCP Server."""
    server = NetworkXMCPServer()
    asyncio.run(server.run())

# Run the server
if __name__ == "__main__":
    main()
