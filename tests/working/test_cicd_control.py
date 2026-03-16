"""Tests for tools/cicd_control.py — CI/CD controller with mocked subprocess."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from networkx_mcp.tools.cicd_control import (
    CICDController,
    mcp_cancel_workflow,
    mcp_get_workflow_status,
    mcp_rerun_failed_jobs,
    mcp_trigger_workflow,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def controller():
    return CICDController()


def _mock_process(stdout=b"", stderr=b"", returncode=0):
    """Create a mock asyncio subprocess result."""
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    return proc


# ===========================================================================
# trigger_workflow
# ===========================================================================


class TestTriggerWorkflow:
    @pytest.mark.asyncio
    @patch("networkx_mcp.tools.cicd_control.asyncio.create_subprocess_exec")
    @patch("networkx_mcp.tools.cicd_control.asyncio.sleep", new_callable=AsyncMock)
    async def test_trigger_success(self, mock_sleep, mock_exec, controller):
        run_info = [
            {
                "databaseId": 123,
                "status": "in_progress",
                "htmlUrl": "https://example.com/run/123",
            }
        ]
        # First call: trigger, second call: list runs
        mock_exec.side_effect = [
            _mock_process(returncode=0),
            _mock_process(stdout=json.dumps(run_info).encode()),
        ]
        result = await controller.trigger_workflow("ci.yml", "main")
        assert result["success"] is True
        assert result["run_id"] == 123

    @pytest.mark.asyncio
    @patch("networkx_mcp.tools.cicd_control.asyncio.create_subprocess_exec")
    async def test_trigger_failure(self, mock_exec, controller):
        mock_exec.return_value = _mock_process(
            stderr=b"workflow not found", returncode=1
        )
        result = await controller.trigger_workflow("bad.yml")
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    @patch("networkx_mcp.tools.cicd_control.asyncio.create_subprocess_exec")
    @patch("networkx_mcp.tools.cicd_control.asyncio.sleep", new_callable=AsyncMock)
    async def test_trigger_with_inputs(self, mock_sleep, mock_exec, controller):
        mock_exec.side_effect = [
            _mock_process(returncode=0),
            _mock_process(stdout=b"[]"),
        ]
        result = await controller.trigger_workflow(
            "ci.yml", "main", inputs={"env": "staging"}
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_trigger_exception(self, controller):
        with patch(
            "networkx_mcp.tools.cicd_control.asyncio.create_subprocess_exec",
            side_effect=OSError("no gh"),
        ):
            result = await controller.trigger_workflow("ci.yml")
            assert result["success"] is False


# ===========================================================================
# get_workflow_status
# ===========================================================================


class TestGetWorkflowStatus:
    @pytest.mark.asyncio
    @patch("networkx_mcp.tools.cicd_control.asyncio.create_subprocess_exec")
    async def test_specific_run(self, mock_exec, controller):
        data = {
            "status": "completed",
            "conclusion": "success",
            "jobs": [{"name": "test", "conclusion": "success"}],
            "createdAt": "2024-01-01T00:00:00Z",
            "updatedAt": "2024-01-01T00:10:00Z",
        }
        mock_exec.return_value = _mock_process(stdout=json.dumps(data).encode())
        result = await controller.get_workflow_status("123")
        assert result["success"] is True
        assert result["run_id"] == "123"
        assert result["status"] == "completed"
        assert result["total_jobs"] == 1

    @pytest.mark.asyncio
    @patch("networkx_mcp.tools.cicd_control.asyncio.create_subprocess_exec")
    async def test_recent_runs(self, mock_exec, controller):
        data = [
            {
                "databaseId": 1,
                "name": "CI",
                "status": "completed",
                "conclusion": "success",
                "createdAt": "2024-01-01T00:00:00Z",
            },
            {
                "databaseId": 2,
                "name": "CI",
                "status": "completed",
                "conclusion": "failure",
                "createdAt": "2024-01-02T00:00:00Z",
            },
        ]
        mock_exec.return_value = _mock_process(stdout=json.dumps(data).encode())
        result = await controller.get_workflow_status(None)
        assert result["success"] is True
        assert result["total_runs"] == 2

    @pytest.mark.asyncio
    @patch("networkx_mcp.tools.cicd_control.asyncio.create_subprocess_exec")
    async def test_status_failure(self, mock_exec, controller):
        mock_exec.return_value = _mock_process(stderr=b"not found", returncode=1)
        result = await controller.get_workflow_status("999")
        assert result["success"] is False


# ===========================================================================
# cancel_workflow
# ===========================================================================


class TestCancelWorkflow:
    @pytest.mark.asyncio
    @patch("networkx_mcp.tools.cicd_control.asyncio.create_subprocess_exec")
    async def test_cancel_success(self, mock_exec, controller):
        controller.active_workflows["42"] = {"status": "running"}
        mock_exec.return_value = _mock_process(returncode=0)
        result = await controller.cancel_workflow("42")
        assert result["success"] is True
        assert "42" not in controller.active_workflows

    @pytest.mark.asyncio
    @patch("networkx_mcp.tools.cicd_control.asyncio.create_subprocess_exec")
    async def test_cancel_failure(self, mock_exec, controller):
        mock_exec.return_value = _mock_process(
            stderr=b"already completed", returncode=1
        )
        result = await controller.cancel_workflow("42")
        assert result["success"] is False


# ===========================================================================
# rerun_failed_jobs
# ===========================================================================


class TestRerunFailedJobs:
    @pytest.mark.asyncio
    @patch("networkx_mcp.tools.cicd_control.asyncio.create_subprocess_exec")
    async def test_rerun_success(self, mock_exec, controller):
        mock_exec.return_value = _mock_process(returncode=0)
        result = await controller.rerun_failed_jobs("42")
        assert result["success"] is True

    @pytest.mark.asyncio
    @patch("networkx_mcp.tools.cicd_control.asyncio.create_subprocess_exec")
    async def test_rerun_failure(self, mock_exec, controller):
        mock_exec.return_value = _mock_process(stderr=b"run not found", returncode=1)
        result = await controller.rerun_failed_jobs("999")
        assert result["success"] is False


# ===========================================================================
# analyze_test_failures
# ===========================================================================


class TestAnalyzeTestFailures:
    @pytest.mark.asyncio
    @patch("networkx_mcp.tools.cicd_control.asyncio.create_subprocess_exec")
    async def test_analyze_with_failures(self, mock_exec, controller):
        data = {
            "jobs": [
                {"name": "test-unit", "conclusion": "failure"},
                {"name": "build-package", "conclusion": "failure"},
                {"name": "lint", "conclusion": "success"},
            ]
        }
        mock_exec.return_value = _mock_process(stdout=json.dumps(data).encode())
        result = await controller.analyze_test_failures("42")
        assert result["success"] is True
        assert result["total_failures"] == 2
        assert len(result["insights"]) >= 1

    @pytest.mark.asyncio
    @patch("networkx_mcp.tools.cicd_control.asyncio.create_subprocess_exec")
    async def test_analyze_no_failures(self, mock_exec, controller):
        data = {"jobs": [{"name": "test", "conclusion": "success"}]}
        mock_exec.return_value = _mock_process(stdout=json.dumps(data).encode())
        result = await controller.analyze_test_failures("42")
        assert result["success"] is True
        assert result["total_failures"] == 0

    @pytest.mark.asyncio
    @patch("networkx_mcp.tools.cicd_control.asyncio.create_subprocess_exec")
    async def test_analyze_failure(self, mock_exec, controller):
        mock_exec.return_value = _mock_process(stderr=b"not found", returncode=1)
        result = await controller.analyze_test_failures("999")
        assert result["success"] is False


# ===========================================================================
# DORA metrics via MCP
# ===========================================================================


class TestDORAMetricsMCP:
    @pytest.mark.asyncio
    @patch("networkx_mcp.tools.cicd_control.get_dora_metrics")
    @patch("networkx_mcp.tools.cicd_control.generate_dora_report")
    async def test_get_dora_metrics_mcp(self, mock_report, mock_metrics, controller):
        mock_metrics.return_value = {
            "deployment_frequency": 1.0,
            "lead_time_hours": 12.0,
            "change_failure_rate": 5.0,
            "mttr_minutes": 30.0,
            "performance_level": "Elite",
        }
        mock_report.return_value = "Test Report"
        result = await controller.get_dora_metrics_mcp()
        assert result["success"] is True
        assert result["performance_level"] == "Elite"
        assert len(result["recommendations"]) == 0  # Elite = no recommendations


# ===========================================================================
# MCP wrapper functions
# ===========================================================================


class TestMCPWrappers:
    @pytest.mark.asyncio
    @patch("networkx_mcp.tools.cicd_control.cicd_controller")
    async def test_mcp_trigger_workflow(self, mock_ctrl):
        mock_ctrl.trigger_workflow = AsyncMock(return_value={"success": True})
        result = await mcp_trigger_workflow("ci.yml", "main", '{"env": "test"}')
        assert result["success"] is True

    @pytest.mark.asyncio
    @patch("networkx_mcp.tools.cicd_control.cicd_controller")
    async def test_mcp_get_workflow_status(self, mock_ctrl):
        mock_ctrl.get_workflow_status = AsyncMock(return_value={"success": True})
        result = await mcp_get_workflow_status("123")
        assert result["success"] is True

    @pytest.mark.asyncio
    @patch("networkx_mcp.tools.cicd_control.cicd_controller")
    async def test_mcp_cancel_workflow(self, mock_ctrl):
        mock_ctrl.cancel_workflow = AsyncMock(return_value={"success": True})
        result = await mcp_cancel_workflow("123")
        assert result["success"] is True

    @pytest.mark.asyncio
    @patch("networkx_mcp.tools.cicd_control.cicd_controller")
    async def test_mcp_rerun_failed_jobs(self, mock_ctrl):
        mock_ctrl.rerun_failed_jobs = AsyncMock(return_value={"success": True})
        result = await mcp_rerun_failed_jobs("123")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_mcp_trigger_workflow_invalid_json(self):
        """Malformed JSON inputs should return error, not crash."""
        result = await mcp_trigger_workflow("ci.yml", "main", "{not valid json}")
        assert result["success"] is False
        assert "Invalid JSON" in result["error"]


# ===========================================================================
# Input sanitization
# ===========================================================================


class TestInputSanitization:
    @pytest.mark.asyncio
    @patch("networkx_mcp.tools.cicd_control.asyncio.create_subprocess_exec")
    @patch("networkx_mcp.tools.cicd_control.asyncio.sleep", new_callable=AsyncMock)
    async def test_trigger_rejects_flag_injection_value(
        self, mock_sleep, mock_exec, controller
    ):
        """Values starting with '-' are rejected to prevent flag injection."""
        result = await controller.trigger_workflow(
            "ci.yml", "main", inputs={"env": "--delete-branch"}
        )
        assert result["success"] is False
        assert "cannot start with" in result["error"]

    @pytest.mark.asyncio
    @patch("networkx_mcp.tools.cicd_control.asyncio.create_subprocess_exec")
    @patch("networkx_mcp.tools.cicd_control.asyncio.sleep", new_callable=AsyncMock)
    async def test_trigger_rejects_invalid_key(self, mock_sleep, mock_exec, controller):
        """Keys with special chars are rejected."""
        result = await controller.trigger_workflow(
            "ci.yml", "main", inputs={"env;rm -rf /": "value"}
        )
        assert result["success"] is False
        assert "Invalid input key" in result["error"]

    @pytest.mark.asyncio
    @patch("networkx_mcp.tools.cicd_control.asyncio.create_subprocess_exec")
    @patch("networkx_mcp.tools.cicd_control.asyncio.sleep", new_callable=AsyncMock)
    async def test_trigger_rejects_oversized_value(
        self, mock_sleep, mock_exec, controller
    ):
        """Values exceeding max length are rejected."""
        result = await controller.trigger_workflow(
            "ci.yml", "main", inputs={"env": "x" * 1001}
        )
        assert result["success"] is False
        assert "exceeds maximum length" in result["error"]
