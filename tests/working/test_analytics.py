"""Tests for academic analytics functions."""

import networkx as nx
import pytest

from networkx_mcp.errors import GraphNotFoundError

from networkx_mcp.academic.analytics import (
    analyze_author_impact,
    calculate_h_index,
    detect_research_trends,
    find_collaboration_patterns,
)


# ---------------------------------------------------------------------------
# calculate_h_index
# ---------------------------------------------------------------------------


class TestCalculateHIndex:
    def test_known_values(self):
        assert calculate_h_index([10, 8, 5, 4, 3]) == 4

    def test_all_high(self):
        # Every paper has >= 5 citations, 5 papers total => h=5
        assert calculate_h_index([100, 90, 80, 70, 60]) == 5

    def test_empty_list(self):
        assert calculate_h_index([]) == 0

    def test_single_element_nonzero(self):
        assert calculate_h_index([7]) == 1

    def test_single_element_zero(self):
        assert calculate_h_index([0]) == 0

    def test_all_zeros(self):
        assert calculate_h_index([0, 0, 0, 0]) == 0

    def test_all_ones(self):
        assert calculate_h_index([1, 1, 1]) == 1

    def test_descending_natural(self):
        # [6,5,4,3,2,1] => h=3 (3 papers with >=3 citations)
        assert calculate_h_index([6, 5, 4, 3, 2, 1]) == 3

    def test_unsorted_input(self):
        # Function sorts internally, so input order shouldn't matter.
        assert calculate_h_index([3, 10, 4, 8, 5]) == 4

    def test_single_large_value(self):
        assert calculate_h_index([1000]) == 1


# ---------------------------------------------------------------------------
# analyze_author_impact
# ---------------------------------------------------------------------------


def _make_citation_graph() -> nx.DiGraph:
    """Build a small citation graph with author metadata."""
    g = nx.DiGraph()
    g.add_node("p1", authors=["Alice", "Bob"], citations=20)
    g.add_node("p2", authors=["Alice", "Charlie"], citations=15)
    g.add_node("p3", authors=["Alice"], citations=5)
    g.add_node("p4", authors=["Bob"], citations=3)
    g.add_node("p5", authors=["Charlie", "Dave"], citations=0)
    # Edges are irrelevant to author impact; presence is fine.
    g.add_edge("p1", "p2")
    return g


class TestAnalyzeAuthorImpact:
    def test_matching_author(self):
        graphs = {"cit": _make_citation_graph()}
        result = analyze_author_impact("cit", "Alice", graphs=graphs)

        assert result["author"] == "Alice"
        assert result["papers_found"] == 3
        # Citations: [20, 15, 5] => h=3 (all >=1,2,3 respectively? 20>=1,15>=2,5>=3 => h=3)
        assert result["h_index"] == 3
        assert result["total_citations"] == 40
        assert result["average_citations"] == pytest.approx(40 / 3)
        assert set(result["papers"]) == {"p1", "p2", "p3"}

    def test_case_insensitive_match(self):
        graphs = {"cit": _make_citation_graph()}
        result = analyze_author_impact("cit", "alice", graphs=graphs)
        assert result["papers_found"] == 3

    def test_no_matching_author(self):
        graphs = {"cit": _make_citation_graph()}
        result = analyze_author_impact("cit", "Zara", graphs=graphs)

        assert result["papers_found"] == 0
        assert result["h_index"] == 0
        assert result["total_citations"] == 0
        assert result["average_citations"] == 0

    def test_empty_graph(self):
        graphs = {"empty": nx.DiGraph()}
        result = analyze_author_impact("empty", "Alice", graphs=graphs)

        assert result["papers_found"] == 0
        assert result["h_index"] == 0

    def test_graph_not_found_raises(self):
        with pytest.raises(GraphNotFoundError):
            analyze_author_impact("nonexistent", "Alice", graphs={})

    def test_none_graphs_raises(self):
        with pytest.raises(GraphNotFoundError):
            analyze_author_impact("any", "Alice", graphs=None)


# ---------------------------------------------------------------------------
# find_collaboration_patterns
# ---------------------------------------------------------------------------


def _make_coauthor_graph() -> nx.Graph:
    """Graph where nodes have author lists — co-authorship edges emerge."""
    g = nx.Graph()
    # Two papers with overlapping author sets.
    g.add_node("p1", authors=["Alice", "Bob", "Charlie"])
    g.add_node("p2", authors=["Alice", "Bob"])
    g.add_node("p3", authors=["Dave", "Eve"])
    return g


class TestFindCollaborationPatterns:
    def test_with_author_data(self):
        graphs = {"co": _make_coauthor_graph()}
        result = find_collaboration_patterns("co", graphs=graphs)

        net = result["coauthorship_network"]
        # Authors: Alice, Bob, Charlie, Dave, Eve
        assert net["nodes"] == 5
        # Edges: (A-B) from p1+p2, (A-C) from p1, (B-C) from p1, (D-E) from p3
        assert net["edges"] == 4

        # Top collaboration is Alice-Bob with 2 joint papers.
        top = result["top_collaborations"]
        assert len(top) >= 1
        assert top[0]["collaborations"] == 2
        assert set(top[0]["authors"]) == {"Alice", "Bob"}

        # Central authors list should exist.
        assert len(result["most_central_authors"]) > 0

    def test_no_author_data_falls_back(self):
        """When nodes lack 'authors', function reports graph structure."""
        g = nx.Graph()
        g.add_node(1)
        g.add_node(2)
        g.add_edge(1, 2)
        graphs = {"plain": g}
        result = find_collaboration_patterns("plain", graphs=graphs)

        assert result["coauthorship_network"]["nodes"] == 0
        assert result["coauthorship_network"]["edges"] == 0
        assert "note" in result
        # Graph has one connected component of size 2.
        assert result["patterns"]["num_clusters"] == 1

    def test_empty_graph(self):
        graphs = {"empty": nx.Graph()}
        result = find_collaboration_patterns("empty", graphs=graphs)

        assert result["coauthorship_network"]["nodes"] == 0
        assert result["coauthorship_network"]["edges"] == 0
        assert result["top_collaborations"] == []
        assert result["most_central_authors"] == []

    def test_single_node_no_authors(self):
        g = nx.Graph()
        g.add_node("solo")
        graphs = {"solo": g}
        result = find_collaboration_patterns("solo", graphs=graphs)

        assert result["coauthorship_network"]["nodes"] == 0
        assert result["patterns"]["num_clusters"] == 1

    def test_directed_graph_without_authors(self):
        """Directed graphs are converted to undirected for component analysis."""
        g = nx.DiGraph()
        g.add_edge("a", "b")
        g.add_edge("c", "d")
        graphs = {"dg": g}
        result = find_collaboration_patterns("dg", graphs=graphs)

        assert result["patterns"]["num_clusters"] == 2

    def test_graph_not_found_raises(self):
        with pytest.raises(GraphNotFoundError):
            find_collaboration_patterns("nope", graphs={})


# ---------------------------------------------------------------------------
# detect_research_trends
# ---------------------------------------------------------------------------


def _make_trending_up_graph() -> nx.DiGraph:
    """More publications in recent years than earlier years."""
    g = nx.DiGraph()
    nid = 0
    for year in range(2010, 2015):
        # 1 paper per early year
        g.add_node(f"p{nid}", year=year, citations=5)
        nid += 1
    for year in range(2015, 2020):
        # 5 papers per recent year
        for _ in range(5):
            g.add_node(f"p{nid}", year=year, citations=10)
            nid += 1
    return g


def _make_stable_graph() -> nx.DiGraph:
    """Same number of publications every year."""
    g = nx.DiGraph()
    nid = 0
    for year in range(2010, 2020):
        for _ in range(3):
            g.add_node(f"p{nid}", year=year, citations=4)
            nid += 1
    return g


def _make_declining_graph() -> nx.DiGraph:
    """More publications early, fewer recently."""
    g = nx.DiGraph()
    nid = 0
    for year in range(2010, 2015):
        for _ in range(10):
            g.add_node(f"p{nid}", year=year, citations=8)
            nid += 1
    for year in range(2015, 2020):
        g.add_node(f"p{nid}", year=year, citations=2)
        nid += 1
    return g


class TestDetectResearchTrends:
    def test_trending_up(self):
        graphs = {"up": _make_trending_up_graph()}
        result = detect_research_trends("up", graphs=graphs)

        assert result["trend"] == "increasing"
        assert result["years_analyzed"] == 10
        assert len(result["publication_trend"]) == 10
        assert len(result["citation_trend"]) == 10

    def test_stable(self):
        graphs = {"flat": _make_stable_graph()}
        result = detect_research_trends("flat", graphs=graphs)

        assert result["trend"] == "stable"
        assert result["years_analyzed"] == 10
        # Every year should have 3 publications.
        for entry in result["publication_trend"]:
            assert entry["publications"] == 3

    def test_declining(self):
        graphs = {"down": _make_declining_graph()}
        result = detect_research_trends("down", graphs=graphs)

        assert result["trend"] == "decreasing"

    def test_no_temporal_data(self):
        """Graph with nodes but no 'year' attribute."""
        g = nx.Graph()
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        graphs = {"notime": g}
        result = detect_research_trends("notime", graphs=graphs)

        assert result["trend"] == "no_temporal_data"
        assert result["years_analyzed"] == 0
        assert result["publication_trend"] == []
        assert result["citation_trend"] == []
        assert result["node_count"] == 3
        assert result["edge_count"] == 2

    def test_empty_graph(self):
        graphs = {"empty": nx.DiGraph()}
        result = detect_research_trends("empty", graphs=graphs)

        assert result["trend"] == "no_temporal_data"
        assert result["years_analyzed"] == 0
        assert result["node_count"] == 0
        assert result["edge_count"] == 0

    def test_single_year_insufficient(self):
        """Only one distinct year — not enough for trend analysis."""
        g = nx.DiGraph()
        g.add_node("p1", year=2020, citations=10)
        g.add_node("p2", year=2020, citations=5)
        graphs = {"one": g}
        result = detect_research_trends("one", graphs=graphs)

        assert result["trend"] == "no_temporal_data"

    def test_custom_time_window(self):
        graphs = {"up": _make_trending_up_graph()}
        result = detect_research_trends("up", time_window=3, graphs=graphs)

        assert result["time_window"] == 3
        assert result["trend"] == "increasing"

    def test_citation_trend_values(self):
        """Verify actual citation numbers in the trend output."""
        graphs = {"flat": _make_stable_graph()}
        result = detect_research_trends("flat", graphs=graphs)

        # Each year has 3 papers with 4 citations each.
        for entry in result["citation_trend"]:
            assert entry["total_citations"] == 12
            assert entry["average_citations"] == pytest.approx(4.0)

    def test_trends_by_year_present(self):
        graphs = {"flat": _make_stable_graph()}
        result = detect_research_trends("flat", graphs=graphs)

        assert 2010 in result["trends_by_year"]
        assert result["trends_by_year"][2010]["publications"] == 3

    def test_graph_not_found_raises(self):
        with pytest.raises(GraphNotFoundError):
            detect_research_trends("missing", graphs={})
