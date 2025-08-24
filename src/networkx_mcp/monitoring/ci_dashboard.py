"""CI/CD Monitoring Dashboard for NetworkX MCP Server.

Provides real-time monitoring of GitHub Actions workflows with
comprehensive metrics and alerting capabilities.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Check if GitHub API library is available
try:
    import aiohttp

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


class CIDashboard:
    """Dashboard for monitoring CI/CD pipeline health."""

    def __init__(self, github_token: Optional[str] = None):
        """Initialize CI/CD dashboard.

        Args:
            github_token: GitHub personal access token for API access
        """
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.repo_owner = "brightlikethelight"
        self.repo_name = "networkx-mcp-server"
        self.workflows = ["CI", "Security", "Docker Build", "CodeQL", "Monitoring"]
        self.api_base = "https://api.github.com"

        if not HAS_AIOHTTP:
            logger.warning("aiohttp not installed - dashboard features limited")
            self.enabled = False
        else:
            self.enabled = True

    async def get_workflow_status(self) -> Dict[str, Any]:
        """Get current status of all monitored workflows.

        Returns:
            Dict containing workflow statuses and metrics
        """
        if not self.enabled:
            return {"error": "Dashboard not available (aiohttp not installed)"}

        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "workflows": {},
            "overall_health": "unknown",
            "metrics": {
                "total_runs": 0,
                "success_rate": 0,
                "avg_duration": 0,
                "failed_workflows": [],
            },
        }

        if not self.github_token:
            status["error"] = "GitHub token not configured"
            return status

        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
        }

        async with aiohttp.ClientSession() as session:
            for workflow in self.workflows:
                try:
                    # Get workflow runs
                    url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/actions/workflows/{workflow}.yml/runs"
                    params = {"per_page": 10, "branch": "main"}

                    async with session.get(
                        url, headers=headers, params=params
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            workflow_info = self._analyze_workflow_runs(data, workflow)
                            status["workflows"][workflow] = workflow_info

                            # Update metrics
                            status["metrics"]["total_runs"] += workflow_info.get(
                                "total_runs", 0
                            )
                            if workflow_info.get("status") == "failure":
                                status["metrics"]["failed_workflows"].append(workflow)
                        else:
                            status["workflows"][workflow] = {
                                "status": "unknown",
                                "error": f"API returned {response.status}",
                            }

                except Exception as e:
                    logger.error(f"Error fetching {workflow} status: {e}")
                    status["workflows"][workflow] = {"status": "error", "error": str(e)}

        # Calculate overall health
        status["overall_health"] = self._calculate_overall_health(status["workflows"])

        # Calculate success rate
        total = sum(w.get("total_runs", 0) for w in status["workflows"].values())
        successful = sum(
            w.get("successful_runs", 0) for w in status["workflows"].values()
        )
        if total > 0:
            status["metrics"]["success_rate"] = round((successful / total) * 100, 2)

        return status

    def _analyze_workflow_runs(
        self, data: Dict[str, Any], workflow_name: str
    ) -> Dict[str, Any]:
        """Analyze workflow runs data.

        Args:
            data: GitHub API response data
            workflow_name: Name of the workflow

        Returns:
            Analyzed workflow information
        """
        runs = data.get("workflow_runs", [])
        if not runs:
            return {"status": "no_data", "last_run": None, "total_runs": 0}

        latest_run = runs[0]

        # Count successes and failures
        successful_runs = sum(1 for run in runs if run["conclusion"] == "success")
        failed_runs = sum(1 for run in runs if run["conclusion"] == "failure")

        # Calculate average duration
        durations = []
        for run in runs:
            if run.get("created_at") and run.get("updated_at"):
                created = datetime.fromisoformat(
                    run["created_at"].replace("Z", "+00:00")
                )
                updated = datetime.fromisoformat(
                    run["updated_at"].replace("Z", "+00:00")
                )
                duration = (updated - created).total_seconds()
                durations.append(duration)

        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            "status": latest_run.get("conclusion", "in_progress"),
            "last_run": {
                "id": latest_run["id"],
                "status": latest_run["status"],
                "conclusion": latest_run["conclusion"],
                "created_at": latest_run["created_at"],
                "updated_at": latest_run["updated_at"],
                "html_url": latest_run["html_url"],
                "actor": latest_run["actor"]["login"]
                if latest_run.get("actor")
                else None,
                "head_branch": latest_run.get("head_branch"),
                "head_sha": latest_run.get("head_sha"),
            },
            "total_runs": len(runs),
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "success_rate": round((successful_runs / len(runs)) * 100, 2)
            if runs
            else 0,
            "avg_duration_seconds": round(avg_duration, 2),
        }

    def _calculate_overall_health(self, workflows: Dict[str, Any]) -> str:
        """Calculate overall CI/CD health status.

        Args:
            workflows: Dict of workflow statuses

        Returns:
            Overall health status string
        """
        statuses = [w.get("status") for w in workflows.values()]

        if all(s == "success" for s in statuses):
            return "healthy"
        elif any(s == "failure" for s in statuses):
            failed_count = sum(1 for s in statuses if s == "failure")
            if failed_count >= 3:
                return "critical"
            elif failed_count >= 1:
                return "degraded"
        elif any(s == "in_progress" for s in statuses):
            return "running"
        elif all(s in ["no_data", "unknown", "error"] for s in statuses):
            return "unknown"

        return "warning"

    async def get_recent_failures(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent workflow failures.

        Args:
            limit: Maximum number of failures to return

        Returns:
            List of recent failure details
        """
        if not self.enabled or not self.github_token:
            return []

        failures = []
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
        }

        async with aiohttp.ClientSession() as session:
            # Get all workflow runs
            url = (
                f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/actions/runs"
            )
            params = {"per_page": 50, "branch": "main", "status": "completed"}

            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    runs = data.get("workflow_runs", [])

                    for run in runs:
                        if run.get("conclusion") == "failure":
                            failures.append(
                                {
                                    "workflow": run["name"],
                                    "run_id": run["id"],
                                    "created_at": run["created_at"],
                                    "actor": run["actor"]["login"]
                                    if run.get("actor")
                                    else None,
                                    "branch": run.get("head_branch"),
                                    "commit": run.get("head_sha"),
                                    "message": run.get("head_commit", {}).get(
                                        "message", ""
                                    ),
                                    "url": run["html_url"],
                                }
                            )

                            if len(failures) >= limit:
                                break

        return failures

    async def get_performance_metrics(self, days: int = 7) -> Dict[str, Any]:
        """Get performance metrics for the past N days.

        Args:
            days: Number of days to analyze

        Returns:
            Performance metrics dict
        """
        if not self.enabled or not self.github_token:
            return {"error": "Metrics not available"}

        metrics = {
            "period_days": days,
            "workflows": {},
            "trends": {"improving": [], "degrading": [], "stable": []},
        }

        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
        }

        since = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"

        async with aiohttp.ClientSession() as session:
            for workflow in self.workflows:
                try:
                    url = f"{self.api_base}/repos/{self.repo_owner}/{self.repo_name}/actions/workflows/{workflow}.yml/runs"
                    params = {"per_page": 100, "branch": "main", "created": f">{since}"}

                    async with session.get(
                        url, headers=headers, params=params
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            workflow_metrics = self._calculate_workflow_metrics(
                                data, workflow
                            )
                            metrics["workflows"][workflow] = workflow_metrics

                            # Determine trend
                            if workflow_metrics.get("trend") == "improving":
                                metrics["trends"]["improving"].append(workflow)
                            elif workflow_metrics.get("trend") == "degrading":
                                metrics["trends"]["degrading"].append(workflow)
                            else:
                                metrics["trends"]["stable"].append(workflow)

                except Exception as e:
                    logger.error(f"Error calculating metrics for {workflow}: {e}")

        return metrics

    def _calculate_workflow_metrics(
        self, data: Dict[str, Any], workflow_name: str
    ) -> Dict[str, Any]:
        """Calculate detailed metrics for a workflow.

        Args:
            data: GitHub API response data
            workflow_name: Name of the workflow

        Returns:
            Workflow metrics
        """
        runs = data.get("workflow_runs", [])
        if not runs:
            return {"status": "no_data"}

        # Split runs into time periods
        now = datetime.utcnow()
        recent_runs = []
        older_runs = []

        for run in runs:
            created = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
            days_ago = (now - created.replace(tzinfo=None)).days

            if days_ago <= 1:
                recent_runs.append(run)
            else:
                older_runs.append(run)

        # Calculate success rates
        recent_success_rate = 0
        if recent_runs:
            recent_successes = sum(
                1 for r in recent_runs if r["conclusion"] == "success"
            )
            recent_success_rate = (recent_successes / len(recent_runs)) * 100

        older_success_rate = 0
        if older_runs:
            older_successes = sum(1 for r in older_runs if r["conclusion"] == "success")
            older_success_rate = (older_successes / len(older_runs)) * 100

        # Determine trend
        trend = "stable"
        if recent_success_rate > older_success_rate + 10:
            trend = "improving"
        elif recent_success_rate < older_success_rate - 10:
            trend = "degrading"

        return {
            "total_runs": len(runs),
            "recent_success_rate": round(recent_success_rate, 2),
            "older_success_rate": round(older_success_rate, 2),
            "trend": trend,
            "last_24h_runs": len(recent_runs),
            "failures_last_24h": sum(
                1 for r in recent_runs if r["conclusion"] == "failure"
            ),
        }

    def format_dashboard_html(self, status: Dict[str, Any]) -> str:
        """Format dashboard data as HTML.

        Args:
            status: Dashboard status data

        Returns:
            HTML formatted dashboard
        """
        health_emoji = {
            "healthy": "🟢",
            "degraded": "🟡",
            "critical": "🔴",
            "running": "🔄",
            "warning": "⚠️",
            "unknown": "❓",
        }

        health = status.get("overall_health", "unknown")
        emoji = health_emoji.get(health, "❓")

        html = f"""
        <html>
        <head>
            <title>NetworkX MCP Server - CI/CD Dashboard</title>
            <meta http-equiv="refresh" content="30">
            <style>
                body {{ font-family: sans-serif; padding: 20px; background: #f5f5f5; }}
                h1 {{ color: #333; }}
                .status-card {{
                    background: white;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .healthy {{ border-left: 4px solid #4caf50; }}
                .degraded {{ border-left: 4px solid #ff9800; }}
                .critical {{ border-left: 4px solid #f44336; }}
                .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
                .metric-value {{ font-size: 24px; font-weight: bold; }}
                .metric-label {{ color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <h1>{emoji} NetworkX MCP Server - CI/CD Dashboard</h1>
            <div class="status-card {health}">
                <h2>Overall Health: {health.upper()}</h2>
                <div class="metric">
                    <div class="metric-value">{status["metrics"]["success_rate"]}%</div>
                    <div class="metric-label">Success Rate</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{status["metrics"]["total_runs"]}</div>
                    <div class="metric-label">Total Runs</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(status["metrics"]["failed_workflows"])}</div>
                    <div class="metric-label">Failed Workflows</div>
                </div>
            </div>
        """

        for workflow, info in status.get("workflows", {}).items():
            status_class = "healthy" if info.get("status") == "success" else "degraded"
            if info.get("status") == "failure":
                status_class = "critical"

            html += f"""
            <div class="status-card {status_class}">
                <h3>{workflow}</h3>
                <p>Status: {info.get("status", "unknown")}</p>
                <p>Success Rate: {info.get("success_rate", 0)}%</p>
                <p>Avg Duration: {info.get("avg_duration_seconds", 0)}s</p>
            </div>
            """

        html += """
        </body>
        </html>
        """

        return html


# Global dashboard instance
ci_dashboard = CIDashboard()
