"""MCP tools for CI/CD pipeline control and monitoring.

Provides MCP-accessible tools for managing CI/CD pipelines, monitoring builds,
and implementing intelligent automation directly through the MCP protocol.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..errors import ValidationError
from ..monitoring.dora_metrics import generate_dora_report, get_dora_metrics

logger = logging.getLogger(__name__)

_RE_WORKFLOW = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
_RE_BRANCH = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9/_.-]*$")
_RE_RUN_ID = re.compile(r"^[0-9]+$")


def _validate_cicd_param(value: str, pattern: re.Pattern, name: str) -> str:
    if not pattern.match(value):
        raise ValidationError(name, value, "does not match expected pattern")
    return value


class CICDController:
    """MCP-based CI/CD pipeline controller."""

    def __init__(self):
        """Initialize CI/CD controller."""
        self.active_workflows: Dict[str, Any] = {}
        self.metrics_cache: Dict[str, Any] = {}

    async def trigger_workflow(
        self,
        workflow_name: str,
        branch: str = "main",
        inputs: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Trigger a GitHub Actions workflow via MCP.

        Args:
            workflow_name: Name of the workflow file
            branch: Branch to run workflow on
            inputs: Optional workflow dispatch inputs

        Returns:
            Workflow run information
        """
        try:
            _validate_cicd_param(workflow_name, _RE_WORKFLOW, "workflow_name")
            _validate_cicd_param(branch, _RE_BRANCH, "branch")

            cmd = ["gh", "workflow", "run", workflow_name, "--ref", branch]

            if inputs:
                _RE_INPUT_VALUE = re.compile(r"^[a-zA-Z0-9_\-./@ :,=]+$")
                for key, value in inputs.items():
                    if not re.match(r"^[a-zA-Z0-9_-]+$", str(key)):
                        return {
                            "success": False,
                            "error": "Invalid input key: keys must be alphanumeric/underscore/hyphen only",
                        }
                    value_str = str(value)
                    # Reject values starting with '-' (flag injection)
                    if value_str.startswith("-"):
                        return {
                            "success": False,
                            "error": f"Invalid input value for '{key}': values cannot start with '-'",
                        }
                    # Strict character whitelist (alphanumeric + safe punctuation)
                    if not _RE_INPUT_VALUE.match(value_str):
                        return {
                            "success": False,
                            "error": f"Invalid input value for '{key}': only alphanumeric, _-./@ :,= allowed",
                        }
                    # Enforce max length
                    if len(value_str) > 1000:
                        return {
                            "success": False,
                            "error": f"Input value for '{key}' exceeds maximum length of 1000 characters",
                        }
                    cmd.extend(["-f", f"{key}={value_str}"])

            # Trigger workflow
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                logger.error(f"Failed to trigger workflow: {stderr.decode()}")
                return {
                    "success": False,
                    "error": "Failed to trigger workflow",
                }

            # Get the run ID
            await asyncio.sleep(2)  # Wait for workflow to start

            list_cmd = [
                "gh",
                "run",
                "list",
                "--workflow",
                workflow_name,
                "--branch",
                branch,
                "--limit",
                "1",
                "--json",
                "databaseId,status,htmlUrl",
            ]

            list_result = await asyncio.create_subprocess_exec(
                *list_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            list_stdout, _ = await list_result.communicate()

            runs = json.loads(list_stdout.decode())
            if runs:
                run_info = runs[0]
                self.active_workflows[str(run_info["databaseId"])] = run_info

                return {
                    "success": True,
                    "run_id": run_info["databaseId"],
                    "status": run_info["status"],
                    "url": run_info["htmlUrl"],
                    "workflow": workflow_name,
                    "branch": branch,
                }

            return {
                "success": True,
                "message": "Workflow triggered but run ID not available yet",
            }

        except (OSError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            logger.error(f"Error triggering workflow: {e}")
            return {
                "success": False,
                "error": "Workflow trigger failed",
            }

    async def get_workflow_status(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of a workflow run or all recent runs.

        Args:
            run_id: Optional specific run ID to check

        Returns:
            Workflow status information
        """
        try:
            if run_id:
                _validate_cicd_param(run_id, _RE_RUN_ID, "run_id")
                # Get specific run
                cmd = [
                    "gh",
                    "run",
                    "view",
                    run_id,
                    "--json",
                    "status,conclusion,jobs,createdAt,updatedAt",
                ]
            else:
                # Get recent runs
                cmd = [
                    "gh",
                    "run",
                    "list",
                    "--limit",
                    "10",
                    "--json",
                    "databaseId,name,status,conclusion,createdAt",
                ]

            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                logger.error(f"Failed to get workflow status: {stderr.decode()}")
                return {
                    "success": False,
                    "error": "Failed to retrieve workflow status",
                }

            data = json.loads(stdout.decode())

            if run_id:
                # Analyze single run
                failed_jobs = [
                    job
                    for job in data.get("jobs", [])
                    if job.get("conclusion") == "failure"
                ]

                return {
                    "success": True,
                    "run_id": run_id,
                    "status": data.get("status"),
                    "conclusion": data.get("conclusion"),
                    "failed_jobs": failed_jobs,
                    "total_jobs": len(data.get("jobs", [])),
                    "created_at": data.get("createdAt"),
                    "updated_at": data.get("updatedAt"),
                }
            else:
                # Analyze recent runs
                failure_rate = sum(
                    1 for run in data if run.get("conclusion") == "failure"
                ) / max(len(data), 1)

                return {
                    "success": True,
                    "total_runs": len(data),
                    "failure_rate": f"{failure_rate:.1%}",
                    "recent_runs": data[:5],  # Top 5 most recent
                }

        except (OSError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            logger.error(f"Error getting workflow status: {e}")
            return {
                "success": False,
                "error": "Failed to retrieve workflow status",
            }

    async def cancel_workflow(self, run_id: str) -> Dict[str, Any]:
        """Cancel a running workflow.

        Args:
            run_id: Workflow run ID to cancel

        Returns:
            Cancellation result
        """
        try:
            _validate_cicd_param(run_id, _RE_RUN_ID, "run_id")
            cmd = ["gh", "run", "cancel", run_id]

            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                logger.error(f"Failed to cancel workflow: {stderr.decode()}")
                return {
                    "success": False,
                    "error": "Failed to cancel workflow",
                }

            # Remove from active workflows
            self.active_workflows.pop(run_id, None)

            return {
                "success": True,
                "run_id": run_id,
                "message": "Workflow cancelled successfully",
            }

        except (OSError, asyncio.TimeoutError) as e:
            logger.error(f"Error cancelling workflow: {e}")
            return {
                "success": False,
                "error": "Workflow cancellation failed",
            }

    async def rerun_failed_jobs(self, run_id: str) -> Dict[str, Any]:
        """Rerun failed jobs in a workflow.

        Args:
            run_id: Workflow run ID

        Returns:
            Rerun result
        """
        try:
            _validate_cicd_param(run_id, _RE_RUN_ID, "run_id")
            cmd = ["gh", "run", "rerun", run_id, "--failed"]

            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                logger.error(f"Failed to rerun jobs: {stderr.decode()}")
                return {
                    "success": False,
                    "error": "Failed to rerun failed jobs",
                }

            return {
                "success": True,
                "run_id": run_id,
                "message": "Failed jobs rerun initiated",
            }

        except (OSError, asyncio.TimeoutError) as e:
            logger.error(f"Error rerunning failed jobs: {e}")
            return {
                "success": False,
                "error": "Failed to rerun failed jobs",
            }

    async def get_dora_metrics_mcp(self) -> Dict[str, Any]:
        """Get DORA metrics via MCP.

        Returns:
            DORA metrics and analysis
        """
        try:
            metrics = get_dora_metrics()
            report = generate_dora_report()

            return {
                "success": True,
                "metrics": metrics,
                "report": report,
                "performance_level": metrics.get("performance_level", "Unknown"),
                "recommendations": self._generate_recommendations(metrics),
            }

        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Error getting DORA metrics: {e}")
            return {
                "success": False,
                "error": "Failed to compute DORA metrics",
            }

    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on metrics.

        Args:
            metrics: DORA metrics dictionary

        Returns:
            List of recommendations
        """
        recommendations = []

        if metrics.get("deployment_frequency", 0) < 1:
            recommendations.append(
                "Increase deployment frequency by implementing continuous deployment"
            )

        if metrics.get("lead_time_hours", 0) > 24:
            recommendations.append(
                "Reduce lead time by optimizing CI pipeline and parallelizing tests"
            )

        if metrics.get("change_failure_rate", 0) > 15:
            recommendations.append(
                "Improve test coverage and implement feature flags for safer deployments"
            )

        if metrics.get("mttr_minutes", 0) > 60:
            recommendations.append(
                "Implement better monitoring and automated rollback mechanisms"
            )

        return recommendations

    async def analyze_test_failures(self, run_id: str) -> Dict[str, Any]:
        """Analyze test failures using AI-powered insights.

        Args:
            run_id: Workflow run ID to analyze

        Returns:
            Analysis results with recommendations
        """
        try:
            _validate_cicd_param(run_id, _RE_RUN_ID, "run_id")
            # Get workflow logs
            cmd = [
                "gh",
                "run",
                "view",
                run_id,
                "--json",
                "jobs",
            ]

            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                logger.error(f"Failed to analyze failures: {stderr.decode()}")
                return {
                    "success": False,
                    "error": "Failed to analyze test failures",
                }

            data = json.loads(stdout.decode())

            # Analyze failures
            failure_patterns: Dict[str, list[str]] = {
                "import_errors": [],
                "timeout_errors": [],
                "assertion_errors": [],
                "network_errors": [],
                "unknown_errors": [],
            }

            for job in data.get("jobs", []):
                if job.get("conclusion") == "failure":
                    # Get job logs (simplified analysis)
                    job_name = job.get("name", "unknown")

                    # Categorize based on job name patterns
                    if "test" in job_name.lower():
                        failure_patterns["assertion_errors"].append(job_name)
                    elif "build" in job_name.lower():
                        failure_patterns["import_errors"].append(job_name)
                    else:
                        failure_patterns["unknown_errors"].append(job_name)

            # Generate insights
            insights = []
            if failure_patterns["import_errors"]:
                insights.append(
                    "Dependency or import issues detected - check requirements"
                )
            if failure_patterns["timeout_errors"]:
                insights.append("Timeout issues detected - consider increasing limits")
            if failure_patterns["assertion_errors"]:
                insights.append("Test assertion failures - review recent code changes")
            if failure_patterns["network_errors"]:
                insights.append(
                    "Network-related failures - check external dependencies"
                )

            return {
                "success": True,
                "run_id": run_id,
                "failure_patterns": failure_patterns,
                "insights": insights,
                "total_failures": sum(len(v) for v in failure_patterns.values()),
            }

        except (OSError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            logger.error(f"Error analyzing test failures: {e}")
            return {
                "success": False,
                "error": "Test failure analysis failed",
            }


# Create singleton controller
cicd_controller = CICDController()


# MCP Tool Functions
async def mcp_trigger_workflow(
    workflow: str,
    branch: str = "main",
    inputs: Optional[str] = None,
) -> Dict[str, Any]:
    """Trigger a CI/CD workflow via MCP.

    Args:
        workflow: Workflow file name (e.g., 'ci.yml')
        branch: Branch to run on
        inputs: JSON string of workflow inputs

    Returns:
        Workflow trigger result
    """
    if inputs:
        try:
            workflow_inputs = json.loads(inputs)
        except json.JSONDecodeError:
            return {"success": False, "error": "Invalid JSON in workflow inputs"}
    else:
        workflow_inputs = None
    return await cicd_controller.trigger_workflow(workflow, branch, workflow_inputs)


async def mcp_get_workflow_status(run_id: Optional[str] = None) -> Dict[str, Any]:
    """Get CI/CD workflow status via MCP.

    Args:
        run_id: Optional specific run ID

    Returns:
        Workflow status information
    """
    return await cicd_controller.get_workflow_status(run_id)


async def mcp_cancel_workflow(run_id: str) -> Dict[str, Any]:
    """Cancel a running workflow via MCP.

    Args:
        run_id: Workflow run ID

    Returns:
        Cancellation result
    """
    return await cicd_controller.cancel_workflow(run_id)


async def mcp_rerun_failed_jobs(run_id: str) -> Dict[str, Any]:
    """Rerun failed jobs via MCP.

    Args:
        run_id: Workflow run ID

    Returns:
        Rerun result
    """
    return await cicd_controller.rerun_failed_jobs(run_id)


async def mcp_get_dora_metrics() -> Dict[str, Any]:
    """Get DORA metrics via MCP.

    Returns:
        DORA metrics and analysis
    """
    return await cicd_controller.get_dora_metrics_mcp()


async def mcp_analyze_failures(run_id: str) -> Dict[str, Any]:
    """Analyze workflow failures via MCP.

    Args:
        run_id: Workflow run ID

    Returns:
        Failure analysis with insights
    """
    return await cicd_controller.analyze_test_failures(run_id)
