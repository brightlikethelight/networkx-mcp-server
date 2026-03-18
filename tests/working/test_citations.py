"""Tests for academic/citations.py — DOI resolution, citation network, BibTeX export."""

from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from networkx_mcp.errors import GraphAlreadyExistsError, GraphNotFoundError

from networkx_mcp.academic.citations import (
    _safe_extract_year,
    build_citation_network,
    export_bibtex,
    recommend_papers,
    resolve_doi,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _mock_crossref_response(
    doi="10.1234/test",
    title="Test Paper",
    authors=None,
    year=2023,
    citations=10,
    references=None,
):
    """Build a mock CrossRef API JSON response."""
    if authors is None:
        authors = [{"given": "Alice", "family": "Smith"}]
    if references is None:
        references = []
    return {
        "status": "ok",
        "message": {
            "DOI": doi,
            "title": [title],
            "author": authors,
            "container-title": ["Test Journal"],
            "published-print": {"date-parts": [[year]]},
            "is-referenced-by-count": citations,
            "reference": references,
        },
    }


def _mock_response(status_code=200, json_data=None):
    """Create a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        from requests.exceptions import HTTPError

        resp.raise_for_status.side_effect = HTTPError(response=resp)
    return resp


# ===========================================================================
# resolve_doi
# ===========================================================================


class TestResolveDOI:
    def test_empty_doi(self):
        result, error = resolve_doi("")
        assert result is None
        assert "Empty DOI" in error

    def test_invalid_doi_format(self):
        result, error = resolve_doi("not-a-doi")
        assert result is None
        assert "Invalid DOI format" in error

    def test_doi_prefix_stripping(self):
        """Various DOI URL formats are cleaned to bare DOI."""
        with patch("networkx_mcp.academic.citations._session.get") as mock_get:
            mock_get.return_value = _mock_response(
                200, _mock_crossref_response("10.1234/test")
            )
            result, error = resolve_doi("doi:10.1234/test", retry_count=1)
            assert error is None
            assert result["doi"] == "10.1234/test"

    def test_https_doi_url_stripping(self):
        with patch("networkx_mcp.academic.citations._session.get") as mock_get:
            mock_get.return_value = _mock_response(
                200, _mock_crossref_response("10.1234/test")
            )
            result, error = resolve_doi("https://doi.org/10.1234/test", retry_count=1)
            assert error is None

    def test_http_doi_url_stripping(self):
        with patch("networkx_mcp.academic.citations._session.get") as mock_get:
            mock_get.return_value = _mock_response(
                200, _mock_crossref_response("10.1234/test")
            )
            result, error = resolve_doi("http://doi.org/10.1234/test", retry_count=1)
            assert error is None

    @patch("networkx_mcp.academic.citations._session.get")
    def test_successful_resolution(self, mock_get):
        mock_get.return_value = _mock_response(
            200,
            _mock_crossref_response(
                doi="10.1234/test",
                title="My Paper",
                authors=[{"given": "Bob", "family": "Jones"}],
                year=2022,
                citations=42,
            ),
        )
        result, error = resolve_doi("10.1234/test", retry_count=1)
        assert error is None
        assert result["doi"] == "10.1234/test"
        assert result["title"] == "My Paper"
        assert result["authors"] == ["Bob Jones"]
        assert result["year"] == 2022
        assert result["citations"] == 42

    @patch("networkx_mcp.academic.citations._session.get")
    def test_404_not_found(self, mock_get):
        mock_get.return_value = _mock_response(404)
        result, error = resolve_doi("10.1234/missing", retry_count=1)
        assert result is None
        assert "not found" in error.lower()

    @patch("networkx_mcp.academic.citations._session.get")
    @patch("networkx_mcp.academic.citations.time.sleep")
    def test_429_rate_limit_retries(self, mock_sleep, mock_get):
        """Rate limiting triggers exponential backoff retry."""
        rate_resp = _mock_response(429)
        rate_resp.raise_for_status = MagicMock()  # 429 doesn't raise
        ok_resp = _mock_response(200, _mock_crossref_response())
        mock_get.side_effect = [rate_resp, ok_resp]

        result, error = resolve_doi("10.1234/test", retry_count=2)
        assert result is not None
        assert error is None

    @patch("networkx_mcp.academic.citations._session.get")
    def test_timeout_error(self, mock_get):
        from requests.exceptions import Timeout

        mock_get.side_effect = Timeout("timed out")
        result, error = resolve_doi("10.1234/test", retry_count=1, retry_delay=0)
        assert result is None
        assert "Timeout" in error

    @patch("networkx_mcp.academic.citations._session.get")
    def test_network_error(self, mock_get):
        from requests.exceptions import ConnectionError

        mock_get.side_effect = ConnectionError("no network")
        result, error = resolve_doi("10.1234/test", retry_count=1, retry_delay=0)
        assert result is None
        assert "Network error" in error

    @patch("networkx_mcp.academic.citations._session.get")
    def test_malformed_response(self, mock_get):
        mock_get.return_value = _mock_response(200, {"status": "ok", "message": {}})
        result, error = resolve_doi("10.1234/test", retry_count=1)
        # Should still succeed with empty fields
        assert result is not None
        assert result["title"] == ""

    @patch("networkx_mcp.academic.citations._session.get")
    def test_unexpected_exception(self, mock_get):
        mock_get.side_effect = RuntimeError("boom")
        result, error = resolve_doi("10.1234/test", retry_count=1)
        assert result is None
        assert "Unexpected error" in error


# ===========================================================================
# build_citation_network
# ===========================================================================


class TestBuildCitationNetwork:
    @patch("networkx_mcp.academic.citations.resolve_doi")
    def test_single_doi_no_refs(self, mock_resolve):
        mock_resolve.return_value = (
            {
                "doi": "10.1/a",
                "title": "A",
                "authors": [],
                "year": 2023,
                "citations": 5,
                "references": [],
                "journal": "J",
            },
            None,
        )
        graphs = {}
        result = build_citation_network("cnet", ["10.1/a"], max_depth=1, graphs=graphs)
        assert result["nodes"] == 1
        assert result["edges"] == 0
        assert "cnet" in graphs
        assert isinstance(graphs["cnet"], nx.DiGraph)

    @patch("networkx_mcp.academic.citations.resolve_doi")
    def test_with_references(self, mock_resolve):
        """DOI with references creates edges and queues child DOIs."""

        def side_effect(doi, **kwargs):
            if doi == "10.1/a":
                return (
                    {
                        "doi": "10.1/a",
                        "title": "A",
                        "authors": [],
                        "year": 2023,
                        "citations": 5,
                        "references": [{"DOI": "10.1/b"}],
                        "journal": "J",
                    },
                    None,
                )
            elif doi == "10.1/b":
                return (
                    {
                        "doi": "10.1/b",
                        "title": "B",
                        "authors": [],
                        "year": 2022,
                        "citations": 3,
                        "references": [],
                        "journal": "J",
                    },
                    None,
                )
            return None, "not found"

        mock_resolve.side_effect = side_effect
        graphs = {}
        result = build_citation_network("cnet", ["10.1/a"], max_depth=2, graphs=graphs)
        assert result["nodes"] == 2
        assert result["edges"] >= 1

    @patch("networkx_mcp.academic.citations.resolve_doi")
    def test_resolution_failure_counted(self, mock_resolve):
        mock_resolve.return_value = (None, "failed")
        graphs = {}
        result = build_citation_network(
            "cnet", ["10.1/bad"], max_depth=1, graphs=graphs
        )
        assert result["resolution_failures"] == 1
        assert result["nodes"] == 0

    def test_duplicate_graph_name_raises(self):
        graphs = {"existing": nx.Graph()}
        with pytest.raises(GraphAlreadyExistsError):
            build_citation_network("existing", ["10.1/a"], graphs=graphs)

    @patch("networkx_mcp.academic.citations.resolve_doi")
    def test_errors_limited_to_ten(self, mock_resolve):
        mock_resolve.return_value = (None, "fail")
        graphs = {}
        dois = [f"10.1/bad{i}" for i in range(15)]
        result = build_citation_network("cnet", dois, max_depth=0, graphs=graphs)
        assert result["resolution_failures"] == 15
        # Only first 10 errors reported in detail
        assert len(result.get("errors", [])) <= 10


# ===========================================================================
# export_bibtex
# ===========================================================================


class TestExportBibtex:
    def test_export_with_metadata(self):
        g = nx.DiGraph()
        g.add_node(
            "10.1/a",
            title="Paper A",
            authors=["Alice Smith"],
            journal="Nature",
            year=2023,
            doi="10.1/a",
            citations=50,
        )
        g.add_node(
            "10.1/b",
            title="Paper B",
            authors=["Bob Jones"],
            journal="Science",
            year=2022,
            doi="10.1/b",
            citations=30,
        )
        graphs = {"bib": g}
        result = export_bibtex("bib", graphs)
        assert result["format"] == "bibtex"
        assert result["entries"] == 2
        assert "Alice Smith" in result["bibtex_data"]

    def test_export_empty_graph(self):
        graphs = {"empty": nx.DiGraph()}
        result = export_bibtex("empty", graphs)
        assert result["entries"] == 0

    def test_export_missing_graph(self):
        with pytest.raises(GraphNotFoundError):
            export_bibtex("nope", {})


# ===========================================================================
# recommend_papers
# ===========================================================================


class TestRecommendPapers:
    def _build_recommendation_graph(self):
        """Build a graph where co-citation recommendations are possible."""
        g = nx.DiGraph()
        # Seed paper cites paper B
        g.add_node("seed", title="Seed", authors=[], year=2023, citations=10)
        g.add_node("B", title="B", authors=[], year=2022, citations=20)
        g.add_node("C", title="C", authors=[], year=2021, citations=50)
        g.add_edge("seed", "B")  # seed cites B
        g.add_edge("C", "B")  # C also cites B (co-citation)
        return g

    def test_seed_not_in_graph(self):
        graphs = {"g": nx.DiGraph()}
        result = recommend_papers("g", "missing_doi", graphs=graphs)
        assert result["total_found"] == 0
        assert "not found" in result["note"]

    def test_recommendations_from_co_citation(self):
        g = self._build_recommendation_graph()
        graphs = {"g": g}
        result = recommend_papers("g", "seed", max_recommendations=5, graphs=graphs)
        assert result["seed_paper"] == "seed"
        # C should be recommended (co-cites B with seed)
        rec_dois = [r["doi"] for r in result["recommendations"]]
        assert "C" in rec_dois

    def test_max_recommendations_limit(self):
        g = self._build_recommendation_graph()
        graphs = {"g": g}
        result = recommend_papers("g", "seed", max_recommendations=0, graphs=graphs)
        assert len(result["recommendations"]) == 0

    def test_missing_graph_raises(self):
        with pytest.raises(GraphNotFoundError):
            recommend_papers("nope", "doi", graphs={})


# ===========================================================================
# _safe_extract_year
# ===========================================================================


class TestSafeExtractYear:
    def test_normal_date_parts(self):
        work = {"published-print": {"date-parts": [[2023]]}}
        assert _safe_extract_year(work) == 2023

    def test_empty_date_parts_does_not_crash(self):
        """date-parts is [] (empty list) — should return None, not IndexError."""
        work = {"published-print": {"date-parts": []}}
        assert _safe_extract_year(work) is None

    def test_missing_date_parts(self):
        work = {"published-print": {}}
        assert _safe_extract_year(work) is None

    def test_falls_back_to_online(self):
        work = {"published-online": {"date-parts": [[2021]]}}
        assert _safe_extract_year(work) == 2021

    def test_both_empty(self):
        work = {}
        assert _safe_extract_year(work) is None

    def test_none_in_date_parts(self):
        work = {"published-print": {"date-parts": [[None]]}}
        assert _safe_extract_year(work) is None
