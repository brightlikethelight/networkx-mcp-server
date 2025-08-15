"""Test DOI resolution error handling improvements."""

import unittest
from unittest.mock import Mock, patch

from requests.exceptions import ConnectionError, Timeout

from networkx_mcp.academic.citations import build_citation_network, resolve_doi


class TestDOIErrorHandling(unittest.TestCase):
    """Test improved DOI resolution error handling."""

    def test_empty_doi(self):
        """Test handling of empty DOI."""
        result, error = resolve_doi("")
        self.assertIsNone(result)
        self.assertEqual(error, "Empty DOI provided")

    def test_invalid_doi_format(self):
        """Test handling of invalid DOI format."""
        # Missing prefix
        result, error = resolve_doi("invalid_doi")
        self.assertIsNone(result)
        self.assertIn("Invalid DOI format", error)

        # Missing slash
        result, error = resolve_doi("10.1234")
        self.assertIsNone(result)
        self.assertIn("Invalid DOI format", error)

    def test_doi_cleaning(self):
        """Test DOI format cleaning."""
        test_cases = [
            ("doi:10.1038/nature12373", "10.1038/nature12373"),
            ("https://doi.org/10.1038/nature12373", "10.1038/nature12373"),
            ("http://doi.org/10.1038/nature12373", "10.1038/nature12373"),
        ]

        for input_doi, expected_clean in test_cases:
            with patch("requests.get") as mock_get:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "message": {
                        "DOI": expected_clean,
                        "title": ["Test Paper"],
                    }
                }
                mock_get.return_value = mock_response

                result, error = resolve_doi(input_doi)
                self.assertIsNotNone(result)
                self.assertIsNone(error)
                self.assertEqual(result["doi"], expected_clean)

    @patch("requests.get")
    def test_404_not_found(self, mock_get):
        """Test handling of DOI not found (404)."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result, error = resolve_doi("10.1234/nonexistent")
        self.assertIsNone(result)
        self.assertIn("DOI not found", error)

    @patch("requests.get")
    def test_rate_limiting_429(self, mock_get):
        """Test handling of rate limiting (429)."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response

        result, error = resolve_doi(
            "10.1038/nature12373", retry_count=1, retry_delay=0.1
        )
        self.assertIsNone(result)
        self.assertIn("Rate limited", error)

    @patch("requests.get")
    def test_timeout_handling(self, mock_get):
        """Test handling of request timeout."""
        mock_get.side_effect = Timeout("Request timed out")

        result, error = resolve_doi(
            "10.1038/nature12373", retry_count=2, retry_delay=0.1
        )
        self.assertIsNone(result)
        self.assertIn("Timeout", error)

    @patch("requests.get")
    def test_network_error(self, mock_get):
        """Test handling of network errors."""
        mock_get.side_effect = ConnectionError("Network error")

        result, error = resolve_doi(
            "10.1038/nature12373", retry_count=2, retry_delay=0.1
        )
        self.assertIsNone(result)
        self.assertIn("Network error", error)

    @patch("requests.get")
    def test_invalid_json_response(self, mock_get):
        """Test handling of invalid JSON response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response

        result, error = resolve_doi("10.1038/nature12373")
        self.assertIsNone(result)
        self.assertIn("Invalid response format", error)

    @patch("requests.get")
    def test_retry_success(self, mock_get):
        """Test successful retry after initial failure."""
        # First call fails with timeout, second succeeds
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "message": {
                "DOI": "10.1038/nature12373",
                "title": ["Test Paper"],
            }
        }

        mock_get.side_effect = [Timeout("Timeout"), mock_response_success]

        result, error = resolve_doi(
            "10.1038/nature12373", retry_count=2, retry_delay=0.1
        )
        self.assertIsNotNone(result)
        self.assertIsNone(error)
        self.assertEqual(result["doi"], "10.1038/nature12373")

    def test_build_citation_network_with_errors(self):
        """Test citation network building with DOI resolution errors."""
        graphs = {}

        with patch("networkx_mcp.academic.citations.resolve_doi") as mock_resolve:
            # First DOI succeeds, second fails, third succeeds
            mock_resolve.side_effect = [
                ({"doi": "10.1/001", "title": "Paper 1", "references": []}, None),
                (None, "DOI not found: 10.1/002"),
                ({"doi": "10.1/003", "title": "Paper 3", "references": []}, None),
            ]

            result = build_citation_network(
                "test_network",
                ["10.1/001", "10.1/002", "10.1/003"],
                max_depth=1,
                graphs=graphs,
            )

            self.assertEqual(result["created"], "test_network")
            self.assertEqual(result["nodes"], 2)  # Only 2 successful
            self.assertEqual(result["resolution_failures"], 1)
            self.assertIn("errors", result)
            self.assertEqual(len(result["errors"]), 1)
            self.assertIn("10.1/002", result["errors"][0])
