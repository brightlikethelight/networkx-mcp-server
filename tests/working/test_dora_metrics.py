"""Tests for monitoring/dora_metrics.py — DORA metrics with mocked subprocess."""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from networkx_mcp.monitoring.dora_metrics import DORAMetricsCollector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def collector(tmp_path):
    return DORAMetricsCollector(repo_path=tmp_path)


def _mock_subprocess_result(stdout="", returncode=0):
    result = MagicMock()
    result.stdout = stdout
    result.returncode = returncode
    return result


# ===========================================================================
# Deployment Frequency
# ===========================================================================


class TestDeploymentFrequency:
    @patch("networkx_mcp.monitoring.dora_metrics.subprocess.run")
    def test_basic_frequency(self, mock_run, collector):
        # 7 commits in 7 days = 1.0 per day
        mock_run.return_value = _mock_subprocess_result(
            "abc1234 commit 1\ndef5678 commit 2\n111 c3\n222 c4\n333 c5\n444 c6\n555 c7"
        )
        freq = collector.collect_deployment_frequency(days=7)
        assert freq == 1.0

    @patch("networkx_mcp.monitoring.dora_metrics.subprocess.run")
    def test_empty_repo(self, mock_run, collector):
        mock_run.return_value = _mock_subprocess_result("")
        freq = collector.collect_deployment_frequency(days=7)
        assert freq == 0.0

    @patch("networkx_mcp.monitoring.dora_metrics.subprocess.run")
    def test_subprocess_failure(self, mock_run, collector):
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")
        freq = collector.collect_deployment_frequency(days=7)
        assert freq == 0.0


# ===========================================================================
# Lead Time
# ===========================================================================


class TestLeadTime:
    @patch("networkx_mcp.monitoring.dora_metrics.subprocess.run")
    def test_basic_lead_time(self, mock_run, collector):
        prs = [
            {
                "number": 1,
                "createdAt": "2024-01-01T00:00:00Z",
                "mergedAt": "2024-01-01T12:00:00Z",
            },
            {
                "number": 2,
                "createdAt": "2024-01-02T00:00:00Z",
                "mergedAt": "2024-01-02T06:00:00Z",
            },
        ]
        mock_run.return_value = _mock_subprocess_result(json.dumps(prs))
        lt = collector.collect_lead_time()
        assert lt == pytest.approx(9.0)  # (12 + 6) / 2

    @patch("networkx_mcp.monitoring.dora_metrics.subprocess.run")
    def test_no_prs(self, mock_run, collector):
        mock_run.return_value = _mock_subprocess_result("")
        lt = collector.collect_lead_time()
        assert lt == 0.0

    @patch("networkx_mcp.monitoring.dora_metrics.subprocess.run")
    def test_prs_missing_dates(self, mock_run, collector):
        prs = [{"number": 1, "createdAt": None, "mergedAt": None}]
        mock_run.return_value = _mock_subprocess_result(json.dumps(prs))
        lt = collector.collect_lead_time()
        assert lt == 0.0

    @patch("networkx_mcp.monitoring.dora_metrics.subprocess.run")
    def test_subprocess_failure(self, mock_run, collector):
        mock_run.side_effect = subprocess.CalledProcessError(1, "gh")
        lt = collector.collect_lead_time()
        assert lt == 0.0


# ===========================================================================
# Change Failure Rate
# ===========================================================================


class TestChangeFailureRate:
    @patch("networkx_mcp.monitoring.dora_metrics.subprocess.run")
    def test_some_failures(self, mock_run, collector):
        runs = [
            {"conclusion": "success", "status": "completed"},
            {"conclusion": "failure", "status": "completed"},
            {"conclusion": "success", "status": "completed"},
            {"conclusion": "cancelled", "status": "completed"},
        ]
        mock_run.return_value = _mock_subprocess_result(json.dumps(runs))
        rate = collector.collect_change_failure_rate(runs=4)
        assert rate == pytest.approx(50.0)  # 2/4

    @patch("networkx_mcp.monitoring.dora_metrics.subprocess.run")
    def test_all_success(self, mock_run, collector):
        runs = [{"conclusion": "success", "status": "completed"}] * 5
        mock_run.return_value = _mock_subprocess_result(json.dumps(runs))
        rate = collector.collect_change_failure_rate(runs=5)
        assert rate == 0.0

    @patch("networkx_mcp.monitoring.dora_metrics.subprocess.run")
    def test_empty_runs(self, mock_run, collector):
        mock_run.return_value = _mock_subprocess_result("")
        rate = collector.collect_change_failure_rate()
        assert rate == 0.0

    @patch("networkx_mcp.monitoring.dora_metrics.subprocess.run")
    def test_subprocess_failure(self, mock_run, collector):
        mock_run.side_effect = subprocess.CalledProcessError(1, "gh")
        rate = collector.collect_change_failure_rate()
        assert rate == 0.0


# ===========================================================================
# MTTR
# ===========================================================================


class TestMTTR:
    @patch("networkx_mcp.monitoring.dora_metrics.subprocess.run")
    def test_basic_mttr(self, mock_run, collector):
        runs = [
            {
                "conclusion": "failure",
                "createdAt": "2024-01-01T00:00:00Z",
                "updatedAt": "2024-01-01T00:10:00Z",
            },
            {
                "conclusion": "success",
                "createdAt": "2024-01-01T00:30:00Z",
                "updatedAt": "2024-01-01T00:35:00Z",
            },
        ]
        mock_run.return_value = _mock_subprocess_result(json.dumps(runs))
        mttr = collector.collect_mttr()
        # Recovery: from failure updatedAt (00:10) to success createdAt (00:30) = 20 min
        assert mttr == pytest.approx(20.0)

    @patch("networkx_mcp.monitoring.dora_metrics.subprocess.run")
    def test_no_failures(self, mock_run, collector):
        runs = [
            {
                "conclusion": "success",
                "createdAt": "2024-01-01T00:00:00Z",
                "updatedAt": "2024-01-01T00:05:00Z",
            },
        ]
        mock_run.return_value = _mock_subprocess_result(json.dumps(runs))
        mttr = collector.collect_mttr()
        assert mttr == 30.0  # Default

    @patch("networkx_mcp.monitoring.dora_metrics.subprocess.run")
    def test_subprocess_failure(self, mock_run, collector):
        mock_run.side_effect = subprocess.CalledProcessError(1, "gh")
        mttr = collector.collect_mttr()
        assert mttr == 30.0  # Default


# ===========================================================================
# Performance Level
# ===========================================================================


class TestPerformanceLevel:
    def test_elite(self, collector):
        metrics = {
            "deployment_frequency": 2.0,
            "lead_time_hours": 12.0,
            "change_failure_rate": 5.0,
            "mttr_minutes": 30.0,
        }
        assert collector._calculate_performance_level(metrics) == "Elite"

    def test_high(self, collector):
        metrics = {
            "deployment_frequency": 0.5,
            "lead_time_hours": 100.0,
            "change_failure_rate": 20.0,
            "mttr_minutes": 600.0,
        }
        assert collector._calculate_performance_level(metrics) == "High"

    def test_medium(self, collector):
        metrics = {
            "deployment_frequency": 0.05,
            "lead_time_hours": 500.0,
            "change_failure_rate": 40.0,
            "mttr_minutes": 5000.0,
        }
        assert collector._calculate_performance_level(metrics) == "Medium"

    def test_low(self, collector):
        metrics = {
            "deployment_frequency": 0.01,
            "lead_time_hours": 1000.0,
            "change_failure_rate": 60.0,
            "mttr_minutes": 20000.0,
        }
        assert collector._calculate_performance_level(metrics) == "Low"


# ===========================================================================
# Collect All & Report
# ===========================================================================


class TestCollectAllAndReport:
    @patch("networkx_mcp.monitoring.dora_metrics.subprocess.run")
    def test_collect_all(self, mock_run, collector):
        # Mock all subprocess calls to return empty
        mock_run.return_value = _mock_subprocess_result("")
        metrics = collector.collect_all_metrics()
        assert "timestamp" in metrics
        assert "deployment_frequency" in metrics
        assert "performance_level" in metrics
        assert len(collector.metrics_history) == 1

    @patch("networkx_mcp.monitoring.dora_metrics.subprocess.run")
    def test_generate_report(self, mock_run, collector):
        mock_run.return_value = _mock_subprocess_result("")
        report = collector.generate_report()
        assert "DORA Metrics Report" in report
        assert "Performance Level" in report

    @patch("networkx_mcp.monitoring.dora_metrics.subprocess.run")
    def test_export_metrics(self, mock_run, collector, tmp_path):
        mock_run.return_value = _mock_subprocess_result("")
        collector.collect_all_metrics()
        filepath = tmp_path / "metrics.json"
        collector.export_metrics(filepath)
        assert filepath.exists()
        data = json.loads(filepath.read_text())
        assert "current" in data
        assert "history" in data
