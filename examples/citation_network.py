"""Example: Academic Citation Network Analysis using NetworkX MCP Server.

This example demonstrates analyzing a citation network of research papers,
including identifying influential papers, research trends, and collaboration patterns.
"""

import asyncio
import json
from typing import Any, Dict


# Simulated MCP client calls
async def call_tool(tool_name: str, **params) -> Dict[str, Any]:
    """Simulate calling an MCP tool."""
    print(f"\n📚 Calling tool: {tool_name}")
    print(f"   Parameters: {json.dumps(params, indent=2)}")
    return {"success": True, "result": "simulated"}


async def create_citation_network():
    """Create an academic citation network."""
    print("=" * 60)
    print("Academic Citation Network Analysis")
    print("=" * 60)

    # 1. Create directed graph for citations
    print("\n1. Creating citation network...")
    await call_tool(
        "create_graph",
        graph_id="citations",
        graph_type="directed",  # Citations are directed
        attributes={
            "name": "Machine Learning Research Citations",
            "domain": "Computer Science",
            "time_period": "2018-2024",
        },
    )

    # 2. Add papers (nodes)
    print("\n2. Adding research papers...")
    papers = [
        # Foundational papers
        {
            "id": "P001",
            "title": "Attention Is All You Need",
            "year": 2017,
            "venue": "NeurIPS",
            "authors": 8,
            "field": "NLP",
            "citations_count": 50000,
        },
        {
            "id": "P002",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "year": 2018,
            "venue": "NAACL",
            "authors": 12,
            "field": "NLP",
            "citations_count": 35000,
        },
        # Computer Vision papers
        {
            "id": "P003",
            "title": "Vision Transformer",
            "year": 2020,
            "venue": "ICLR",
            "authors": 10,
            "field": "CV",
            "citations_count": 15000,
        },
        {
            "id": "P004",
            "title": "DETR: End-to-End Object Detection",
            "year": 2020,
            "venue": "ECCV",
            "authors": 6,
            "field": "CV",
            "citations_count": 8000,
        },
        # Recent influential papers
        {
            "id": "P005",
            "title": "GPT-3: Language Models are Few-Shot Learners",
            "year": 2020,
            "venue": "NeurIPS",
            "authors": 31,
            "field": "NLP",
            "citations_count": 20000,
        },
        {
            "id": "P006",
            "title": "Scaling Laws for Neural Language Models",
            "year": 2020,
            "venue": "arXiv",
            "authors": 8,
            "field": "NLP",
            "citations_count": 5000,
        },
        # Reinforcement Learning
        {
            "id": "P007",
            "title": "Proximal Policy Optimization",
            "year": 2017,
            "venue": "arXiv",
            "authors": 4,
            "field": "RL",
            "citations_count": 12000,
        },
        {
            "id": "P008",
            "title": "Soft Actor-Critic",
            "year": 2018,
            "venue": "ICML",
            "authors": 5,
            "field": "RL",
            "citations_count": 7000,
        },
        # Recent papers building on foundations
        {
            "id": "P009",
            "title": "RoBERTa: A Robustly Optimized BERT",
            "year": 2019,
            "venue": "arXiv",
            "authors": 9,
            "field": "NLP",
            "citations_count": 10000,
        },
        {
            "id": "P010",
            "title": "DALL-E: Creating Images from Text",
            "year": 2021,
            "venue": "ICML",
            "authors": 15,
            "field": "CV",
            "citations_count": 3000,
        },
        {
            "id": "P011",
            "title": "CLIP: Learning Transferable Visual Models",
            "year": 2021,
            "venue": "ICML",
            "authors": 12,
            "field": "CV",
            "citations_count": 5000,
        },
        {
            "id": "P012",
            "title": "Codex: Evaluating Large Language Models",
            "year": 2021,
            "venue": "arXiv",
            "authors": 20,
            "field": "NLP",
            "citations_count": 2000,
        },
        # Domain-specific applications
        {
            "id": "P013",
            "title": "AlphaFold: Protein Structure Prediction",
            "year": 2021,
            "venue": "Nature",
            "authors": 18,
            "field": "Bio",
            "citations_count": 8000,
        },
        {
            "id": "P014",
            "title": "Graph Neural Networks Survey",
            "year": 2020,
            "venue": "IEEE",
            "authors": 4,
            "field": "ML",
            "citations_count": 4000,
        },
        {
            "id": "P015",
            "title": "Self-Supervised Learning Survey",
            "year": 2021,
            "venue": "IEEE",
            "authors": 6,
            "field": "ML",
            "citations_count": 2500,
        },
    ]

    await call_tool("add_nodes", graph_id="citations", nodes=papers)

    # 3. Add citations (edges)
    print("\n3. Adding citation relationships...")
    citations = [
        # BERT builds on Transformer
        {"source": "P002", "target": "P001", "year": 2018, "context": "architecture"},
        # Vision Transformer inspired by text transformers
        {"source": "P003", "target": "P001", "year": 2020, "context": "architecture"},
        {"source": "P003", "target": "P002", "year": 2020, "context": "methodology"},
        # DETR uses transformers for detection
        {"source": "P004", "target": "P001", "year": 2020, "context": "architecture"},
        {"source": "P004", "target": "P003", "year": 2020, "context": "application"},
        # GPT-3 builds on transformer
        {"source": "P005", "target": "P001", "year": 2020, "context": "architecture"},
        {"source": "P005", "target": "P006", "year": 2020, "context": "scaling"},
        # RoBERTa improves BERT
        {"source": "P009", "target": "P002", "year": 2019, "context": "improvement"},
        {"source": "P009", "target": "P001", "year": 2019, "context": "architecture"},
        # DALL-E uses multiple innovations
        {"source": "P010", "target": "P003", "year": 2021, "context": "vision"},
        {"source": "P010", "target": "P005", "year": 2021, "context": "generation"},
        # CLIP combines vision and language
        {"source": "P011", "target": "P003", "year": 2021, "context": "vision"},
        {"source": "P011", "target": "P002", "year": 2021, "context": "language"},
        # Codex builds on GPT-3
        {"source": "P012", "target": "P005", "year": 2021, "context": "extension"},
        # AlphaFold uses attention mechanisms
        {"source": "P013", "target": "P001", "year": 2021, "context": "architecture"},
        # Surveys cite multiple papers
        {"source": "P014", "target": "P001", "year": 2020, "context": "survey"},
        {"source": "P014", "target": "P003", "year": 2020, "context": "survey"},
        {"source": "P015", "target": "P002", "year": 2021, "context": "survey"},
        {"source": "P015", "target": "P003", "year": 2021, "context": "survey"},
        {"source": "P015", "target": "P009", "year": 2021, "context": "survey"},
        # RL papers cite each other
        {"source": "P008", "target": "P007", "year": 2018, "context": "comparison"},
    ]

    await call_tool("add_edges", graph_id="citations", edges=citations)


async def analyze_paper_influence():
    """Analyze the influence of papers in the network."""
    print("\n" + "=" * 60)
    print("Paper Influence Analysis")
    print("=" * 60)

    # 1. Find most cited papers (high in-degree)
    print("\n1. Finding most cited papers...")
    await call_tool(
        "calculate_centrality", graph_id="citations", centrality_type="degree", top_n=5
    )

    # 2. Find papers that bridge research areas (high betweenness)
    print("\n2. Finding papers that connect different research areas...")
    await call_tool(
        "calculate_centrality",
        graph_id="citations",
        centrality_type="betweenness",
        top_n=5,
    )

    # 3. PageRank for overall importance
    print("\n3. Calculating PageRank (considering citation quality)...")
    await call_tool(
        "calculate_centrality",
        graph_id="citations",
        centrality_type="pagerank",
        top_n=5,
    )


async def analyze_research_trends():
    """Analyze research trends and evolution."""
    print("\n" + "=" * 60)
    print("Research Trends Analysis")
    print("=" * 60)

    # 1. Find citation paths (research evolution)
    print("\n1. Tracing research evolution from Transformer to DALL-E...")
    await call_tool(
        "find_all_paths",
        graph_id="citations",
        source="P010",  # DALL-E
        target="P001",  # Transformer
        max_paths=10,
    )

    # 2. Analyze cycles (mutual citations)
    print("\n2. Finding citation cycles (mutual citations/debates)...")
    await call_tool("cycle_detection", graph_id="citations", max_cycle_length=4)

    # 3. Find papers with no citations (leaves)
    print("\n3. Finding recent papers (no outgoing citations in our network)...")
    # This would identify papers that don't cite others in our network


async def analyze_research_communities():
    """Identify research communities and collaborations."""
    print("\n" + "=" * 60)
    print("Research Community Analysis")
    print("=" * 60)

    # 1. Detect research communities
    print("\n1. Detecting research communities...")
    await call_tool("community_detection", graph_id="citations", method="louvain")

    # 2. Analyze field-specific subnetworks
    print("\n2. Extracting NLP research subnetwork...")
    await call_tool(
        "subgraph_extraction",
        graph_id="citations",
        method="condition",
        condition="field = NLP",
        create_new=True,
        new_graph_id="nlp_citations",
    )

    # 3. Analyze NLP subnetwork
    print("\n3. Analyzing NLP research network...")
    await call_tool("graph_metrics", graph_id="nlp_citations")


async def find_seminal_papers():
    """Identify seminal papers and their impact."""
    print("\n" + "=" * 60)
    print("Seminal Papers Analysis")
    print("=" * 60)

    # 1. Find papers with longest citation chains
    print("\n1. Finding papers with deepest influence (longest paths)...")
    await call_tool("path_analysis", graph_id="citations")

    # 2. Find papers cited by multiple fields
    print("\n2. Finding interdisciplinary papers...")
    # Extract 2-hop citation network for key papers
    await call_tool(
        "subgraph_extraction",
        graph_id="citations",
        method="k_hop",
        center_node="P001",  # Transformer paper
        k_hop=2,
        create_new=True,
        new_graph_id="transformer_impact",
    )

    await call_tool("get_graph_info", graph_id="transformer_impact")


async def analyze_collaboration_patterns():
    """Analyze collaboration patterns through co-citations."""
    print("\n" + "=" * 60)
    print("Collaboration Pattern Analysis")
    print("=" * 60)

    # 1. Find papers often cited together
    print("\n1. Creating co-citation network...")
    # Papers cited by the same paper are related

    # 2. Analyze author collaboration (based on paper attributes)
    print("\n2. Analyzing team sizes by field...")
    # This would analyze the 'authors' attribute by field

    # 3. Venue analysis
    print("\n3. Analyzing publication venues...")
    # Extract papers by venue


async def predict_future_impact():
    """Predict future impact based on early citation patterns."""
    print("\n" + "=" * 60)
    print("Impact Prediction Analysis")
    print("=" * 60)

    # 1. Find rising papers (recent with growing citations)
    print("\n1. Identifying rising papers...")
    await call_tool(
        "subgraph_extraction",
        graph_id="citations",
        method="condition",
        condition="year > 2020",
        create_new=True,
        new_graph_id="recent_papers",
    )

    # 2. Analyze recent paper patterns
    print("\n2. Analyzing recent paper citation patterns...")
    await call_tool(
        "calculate_centrality",
        graph_id="recent_papers",
        centrality_type=["degree", "pagerank"],
        top_n=5,
    )

    # 3. Find papers building on multiple innovations
    print("\n3. Finding papers that synthesize multiple research streams...")
    await call_tool("clustering_analysis", graph_id="citations", include_triangles=True)


async def export_analysis():
    """Export citation network analysis."""
    print("\n" + "=" * 60)
    print("Exporting Analysis Results")
    print("=" * 60)

    # 1. Generate visualization
    print("\n1. Generating citation network visualization...")
    await call_tool(
        "visualize_graph",
        graph_id="citations",
        layout="kamada_kawai",
        output_format="pyvis",
    )

    # 2. Export for bibliometric tools
    print("\n2. Exporting to GraphML format...")
    await call_tool(
        "export_graph",
        graph_id="citations",
        format="graphml",
        path="citation_network.graphml",
    )

    # 3. Export citation data
    print("\n3. Exporting citation edges to CSV...")
    await call_tool(
        "export_graph", graph_id="citations", format="csv", path="citations.csv"
    )


async def main():
    """Run complete citation network analysis."""
    print("\n📖 Academic Citation Network Analysis with NetworkX MCP Server")
    print("=" * 60)

    # Build and analyze the citation network
    await create_citation_network()
    await analyze_paper_influence()
    await analyze_research_trends()
    await analyze_research_communities()
    await find_seminal_papers()
    await analyze_collaboration_patterns()
    await predict_future_impact()
    await export_analysis()

    print("\n" + "=" * 60)
    print("✅ Citation network analysis complete!")
    print("\nKey insights derived:")
    print("- Most influential papers (by various metrics)")
    print("- Research evolution and trends")
    print("- Research communities and fields")
    print("- Interdisciplinary connections")
    print("- Rising papers and future trends")
    print("\nApplications:")
    print("- Literature review automation")
    print("- Research trend identification")
    print("- Collaboration opportunity discovery")
    print("- Impact prediction for new research")
    print("- Funding allocation insights")


if __name__ == "__main__":
    asyncio.run(main())
