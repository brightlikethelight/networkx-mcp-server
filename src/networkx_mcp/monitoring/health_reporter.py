"""Automated health reporting for CI/CD pipelines."""

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


class HealthReporter:
    """Generate comprehensive health reports for CI/CD pipelines."""

    def __init__(self, github_token: Optional[str] = None):
        """Initialize the health reporter.

        Args:
            github_token: GitHub token for API access
        """
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.repo = os.getenv(
            "GITHUB_REPOSITORY", "brightlikethelight/networkx-mcp-server"
        )
        self.report_dir = Path("reports/ci-health")
        self.report_dir.mkdir(parents=True, exist_ok=True)

    async def generate_daily_report(self) -> Dict[str, Any]:
        """Generate daily CI/CD health report.

        Returns:
            Comprehensive health report dictionary
        """
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "repository": self.repo,
            "period": "24_hours",
            "metrics": await self._collect_metrics(),
            "failures": await self._analyze_failures(),
            "trends": await self._calculate_trends(),
            "recommendations": await self._generate_recommendations(),
        }

        # Save report to file
        report_file = (
            self.report_dir
            / f"health_report_{datetime.utcnow().strftime('%Y%m%d')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        return report

    async def _collect_metrics(self) -> Dict[str, Any]:
        """Collect CI/CD metrics from GitHub API.

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "workflow_runs": {
                "total": 0,
                "success": 0,
                "failure": 0,
                "cancelled": 0,
                "in_progress": 0,
            },
            "average_duration": {},
            "success_rate": {},
            "test_coverage": 0,
            "flaky_tests": [],
        }

        # Fetch workflow runs from GitHub API
        if self.github_token:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"token {self.github_token}"}

                # Get workflow runs from last 24 hours
                since = (datetime.utcnow() - timedelta(hours=24)).isoformat()
                url = f"https://api.github.com/repos/{self.repo}/actions/runs"
                params = {"created": f">={since}", "per_page": 100}

                async with session.get(url, headers=headers, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        runs = data.get("workflow_runs", [])

                        for run in runs:
                            metrics["workflow_runs"]["total"] += 1
                            conclusion = run.get("conclusion")

                            if conclusion == "success":
                                metrics["workflow_runs"]["success"] += 1
                            elif conclusion == "failure":
                                metrics["workflow_runs"]["failure"] += 1
                            elif conclusion == "cancelled":
                                metrics["workflow_runs"]["cancelled"] += 1
                            elif run.get("status") == "in_progress":
                                metrics["workflow_runs"]["in_progress"] += 1

                            # Track per-workflow metrics
                            workflow_name = run.get("name", "unknown")
                            if workflow_name not in metrics["average_duration"]:
                                metrics["average_duration"][workflow_name] = []

                            if run.get("run_started_at") and run.get("updated_at"):
                                start = datetime.fromisoformat(
                                    run["run_started_at"].replace("Z", "+00:00")
                                )
                                end = datetime.fromisoformat(
                                    run["updated_at"].replace("Z", "+00:00")
                                )
                                duration = (end - start).total_seconds()
                                metrics["average_duration"][workflow_name].append(
                                    duration
                                )

        # Calculate success rates
        for workflow, durations in metrics["average_duration"].items():
            if durations:
                metrics["average_duration"][workflow] = sum(durations) / len(durations)

        if metrics["workflow_runs"]["total"] > 0:
            metrics["overall_success_rate"] = (
                metrics["workflow_runs"]["success"]
                / metrics["workflow_runs"]["total"]
                * 100
            )
        else:
            metrics["overall_success_rate"] = 0

        return metrics

    async def _analyze_failures(self) -> List[Dict[str, Any]]:
        """Analyze recent failures for patterns.

        Returns:
            List of failure analysis results
        """
        failures = []

        if self.github_token:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"token {self.github_token}"}

                # Get failed workflow runs
                url = f"https://api.github.com/repos/{self.repo}/actions/runs"
                params = {"status": "failure", "per_page": 20}

                async with session.get(url, headers=headers, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        runs = data.get("workflow_runs", [])

                        for run in runs[:10]:  # Analyze last 10 failures
                            failure = {
                                "workflow": run.get("name"),
                                "run_id": run.get("id"),
                                "created_at": run.get("created_at"),
                                "url": run.get("html_url"),
                                "error_type": "unknown",
                                "priority": "medium",
                            }

                            # Categorize failure type
                            if "test" in run.get("name", "").lower():
                                failure["error_type"] = "test_failure"
                                failure["priority"] = "high"
                            elif "lint" in run.get("name", "").lower():
                                failure["error_type"] = "linting_issue"
                                failure["priority"] = "low"
                            elif "security" in run.get("name", "").lower():
                                failure["error_type"] = "security_issue"
                                failure["priority"] = "critical"

                            failures.append(failure)

        return failures

    async def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate trends over time.

        Returns:
            Dictionary of trend data
        """
        trends = {
            "success_rate_trend": "stable",
            "duration_trend": "stable",
            "failure_pattern": "none",
            "improvement_areas": [],
        }

        # Load previous reports for trend analysis
        previous_reports = []
        for report_file in sorted(self.report_dir.glob("health_report_*.json"))[-7:]:
            with open(report_file) as f:
                previous_reports.append(json.load(f))

        if len(previous_reports) >= 3:
            # Analyze success rate trend
            success_rates = [
                r["metrics"].get("overall_success_rate", 0) for r in previous_reports
            ]

            if success_rates[-1] > success_rates[0] + 5:
                trends["success_rate_trend"] = "improving"
            elif success_rates[-1] < success_rates[0] - 5:
                trends["success_rate_trend"] = "declining"
                trends["improvement_areas"].append("success_rate")

            # Identify failure patterns
            recent_failures = []
            for report in previous_reports[-3:]:
                recent_failures.extend(report.get("failures", []))

            failure_types = {}
            for failure in recent_failures:
                error_type = failure.get("error_type", "unknown")
                failure_types[error_type] = failure_types.get(error_type, 0) + 1

            if failure_types:
                most_common = max(failure_types, key=failure_types.get)
                if failure_types[most_common] >= 3:
                    trends["failure_pattern"] = most_common
                    trends["improvement_areas"].append(most_common)

        return trends

    async def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate actionable recommendations.

        Returns:
            List of recommendations
        """
        recommendations = []

        # Get current metrics
        metrics = await self._collect_metrics()
        trends = await self._calculate_trends()

        # Generate recommendations based on metrics
        if metrics["overall_success_rate"] < 90:
            recommendations.append(
                {
                    "priority": "high",
                    "action": "Improve CI stability",
                    "details": f"Success rate is {metrics['overall_success_rate']:.1f}%, target is 95%+",
                    "suggestion": "Review failing tests and add retry mechanisms",
                }
            )

        if trends["failure_pattern"] == "test_failure":
            recommendations.append(
                {
                    "priority": "high",
                    "action": "Fix flaky tests",
                    "details": "Recurring test failures detected",
                    "suggestion": "Isolate and fix unstable tests, consider adding test retries",
                }
            )

        if trends["failure_pattern"] == "linting_issue":
            recommendations.append(
                {
                    "priority": "medium",
                    "action": "Standardize code formatting",
                    "details": "Frequent linting failures",
                    "suggestion": "Ensure all developers use same tool versions and pre-commit hooks",
                }
            )

        if any(d > 600 for d in metrics.get("average_duration", {}).values()):
            recommendations.append(
                {
                    "priority": "medium",
                    "action": "Optimize slow workflows",
                    "details": "Some workflows take over 10 minutes",
                    "suggestion": "Enable caching, parallelize tests, optimize Docker builds",
                }
            )

        if (
            metrics["workflow_runs"]["cancelled"]
            > metrics["workflow_runs"]["total"] * 0.1
        ):
            recommendations.append(
                {
                    "priority": "low",
                    "action": "Reduce cancelled runs",
                    "details": f"{metrics['workflow_runs']['cancelled']} runs were cancelled",
                    "suggestion": "Review workflow triggers and concurrency settings",
                }
            )

        return recommendations

    async def create_github_issues(
        self, recommendations: List[Dict[str, str]]
    ) -> List[str]:
        """Create GitHub issues for high-priority recommendations.

        Args:
            recommendations: List of recommendations

        Returns:
            List of created issue URLs
        """
        created_issues = []

        if not self.github_token:
            return created_issues

        import aiohttp

        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json",
            }

            for rec in recommendations:
                if rec.get("priority") in ["critical", "high"]:
                    # Create issue for high priority items
                    issue_data = {
                        "title": f"[CI/CD] {rec['action']}",
                        "body": f"## Automated CI/CD Health Report\n\n"
                        f"**Priority**: {rec['priority']}\n"
                        f"**Details**: {rec['details']}\n"
                        f"**Suggestion**: {rec['suggestion']}\n\n"
                        f"*This issue was automatically created by the CI/CD health monitoring system.*",
                        "labels": ["ci/cd", "automated", rec["priority"]],
                    }

                    url = f"https://api.github.com/repos/{self.repo}/issues"
                    async with session.post(
                        url, headers=headers, json=issue_data
                    ) as resp:
                        if resp.status == 201:
                            issue = await resp.json()
                            created_issues.append(issue["html_url"])

        return created_issues

    async def generate_markdown_report(self) -> str:
        """Generate a markdown-formatted health report.

        Returns:
            Markdown report string
        """
        report = await self.generate_daily_report()

        markdown = f"""# CI/CD Health Report

**Generated**: {report["timestamp"]}
**Repository**: {report["repository"]}
**Period**: Last 24 hours

## 📊 Metrics Summary

| Metric | Value |
|--------|-------|
| Total Runs | {report["metrics"]["workflow_runs"]["total"]} |
| Success Rate | {report["metrics"].get("overall_success_rate", 0):.1f}% |
| Failed Runs | {report["metrics"]["workflow_runs"]["failure"]} |
| In Progress | {report["metrics"]["workflow_runs"]["in_progress"]} |

## ⏱️ Average Durations

| Workflow | Duration |
|----------|----------|
"""

        for workflow, duration in report["metrics"].get("average_duration", {}).items():
            if isinstance(duration, (int, float)):
                markdown += f"| {workflow} | {duration:.1f}s |\n"

        markdown += f"""

## 📈 Trends

- **Success Rate Trend**: {report["trends"]["success_rate_trend"]}
- **Duration Trend**: {report["trends"]["duration_trend"]}
- **Failure Pattern**: {report["trends"]["failure_pattern"]}

## 🔴 Recent Failures

"""

        for failure in report["failures"][:5]:
            markdown += f"- [{failure['workflow']}]({failure['url']}) - {failure['error_type']} ({failure['priority']})\n"

        markdown += "\n## 💡 Recommendations\n\n"

        for rec in report["recommendations"]:
            markdown += f"### {rec['priority'].upper()}: {rec['action']}\n"
            markdown += f"- **Details**: {rec['details']}\n"
            markdown += f"- **Suggestion**: {rec['suggestion']}\n\n"

        return markdown


async def main():
    """Run daily health report generation."""
    reporter = HealthReporter()

    print("🔍 Generating CI/CD health report...")
    report = await reporter.generate_daily_report()

    print("📝 Creating markdown report...")
    markdown = await reporter.generate_markdown_report()

    # Save markdown report
    report_file = (
        Path("reports/ci-health")
        / f"health_report_{datetime.utcnow().strftime('%Y%m%d')}.md"
    )
    with open(report_file, "w") as f:
        f.write(markdown)

    print(f"✅ Report saved to {report_file}")

    # Create issues for critical items
    if report["recommendations"]:
        print("📌 Creating GitHub issues for high-priority items...")
        issues = await reporter.create_github_issues(report["recommendations"])
        if issues:
            print(f"✅ Created {len(issues)} issues")
            for issue in issues:
                print(f"  - {issue}")

    return report


if __name__ == "__main__":
    asyncio.run(main())
