"""DORA metrics collection and analysis for CI/CD pipeline.

Implements the four key DORA (DevOps Research and Assessment) metrics:
1. Deployment Frequency - How often we deploy to production
2. Lead Time for Changes - Time from commit to production
3. Change Failure Rate - Percentage of deployments causing failures
4. Mean Time to Recovery (MTTR) - How long to recover from failures
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DORAMetricsCollector:
    """Collects and analyzes DORA metrics for the CI/CD pipeline."""

    def __init__(self, repo_path: Optional[Path] = None):
        """Initialize DORA metrics collector.

        Args:
            repo_path: Path to git repository (defaults to current directory)
        """
        self.repo_path = repo_path or Path.cwd()
        self.metrics_cache: Dict[str, Any] = {}
        self.metrics_history: List[Dict[str, Any]] = []

    def collect_deployment_frequency(self, days: int = 7) -> float:
        """Calculate deployment frequency (deploys per day).

        Args:
            days: Number of days to analyze

        Returns:
            Average deployments per day
        """
        try:
            # Get commits to main branch (deployments)
            cmd = [
                "git",
                "-C",
                str(self.repo_path),
                "log",
                "--since",
                f"{days} days ago",
                "--oneline",
                "--first-parent",
                "main",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            deployments = (
                len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
            )
            frequency = deployments / days

            logger.info(
                f"Deployment frequency: {frequency:.2f} per day ({deployments} in {days} days)"
            )
            return frequency

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to calculate deployment frequency: {e}")
            return 0.0

    def collect_lead_time(self, hours: int = 168) -> float:
        """Calculate lead time for changes (hours from commit to deploy).

        Args:
            hours: Number of hours to analyze (default: 1 week)

        Returns:
            Average lead time in hours
        """
        try:
            # Get PR merge times and deployment times
            cmd = [
                "gh",
                "pr",
                "list",
                "--state",
                "merged",
                "--limit",
                "20",
                "--json",
                "mergedAt,createdAt,number",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if result.stdout:
                prs = json.loads(result.stdout)

                lead_times = []
                for pr in prs:
                    if pr.get("mergedAt") and pr.get("createdAt"):
                        created = datetime.fromisoformat(
                            pr["createdAt"].replace("Z", "+00:00")
                        )
                        merged = datetime.fromisoformat(
                            pr["mergedAt"].replace("Z", "+00:00")
                        )
                        lead_time = (merged - created).total_seconds() / 3600
                        lead_times.append(lead_time)

                if lead_times:
                    avg_lead_time = sum(lead_times) / len(lead_times)
                    logger.info(f"Average lead time: {avg_lead_time:.1f} hours")
                    return avg_lead_time

            return 0.0

        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.error(f"Failed to calculate lead time: {e}")
            return 0.0

    def collect_change_failure_rate(self, runs: int = 20) -> float:
        """Calculate change failure rate (percentage of failed deployments).

        Args:
            runs: Number of recent CI runs to analyze

        Returns:
            Failure rate as percentage (0-100)
        """
        try:
            cmd = [
                "gh",
                "run",
                "list",
                "--workflow",
                "CI",
                "--branch",
                "main",
                "--limit",
                str(runs),
                "--json",
                "conclusion,status",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if result.stdout:
                ci_runs = json.loads(result.stdout)

                failures = sum(
                    1
                    for run in ci_runs
                    if run.get("conclusion") in ["failure", "cancelled"]
                )

                if ci_runs:
                    failure_rate = (failures / len(ci_runs)) * 100
                    logger.info(
                        f"Change failure rate: {failure_rate:.1f}% ({failures}/{len(ci_runs)})"
                    )
                    return failure_rate

            return 0.0

        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.error(f"Failed to calculate change failure rate: {e}")
            return 0.0

    def collect_mttr(self) -> float:
        """Calculate mean time to recovery (MTTR) in minutes.

        Returns:
            Average recovery time in minutes
        """
        try:
            # Get recent failures and their recovery times
            cmd = [
                "gh",
                "run",
                "list",
                "--workflow",
                "CI",
                "--limit",
                "50",
                "--json",
                "conclusion,createdAt,updatedAt",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if result.stdout:
                runs = json.loads(result.stdout)

                recovery_times = []
                previous_failure = None

                for run in sorted(runs, key=lambda x: x["createdAt"]):
                    if run.get("conclusion") == "failure":
                        previous_failure = run
                    elif run.get("conclusion") == "success" and previous_failure:
                        # Calculate recovery time
                        failure_time = datetime.fromisoformat(
                            previous_failure["updatedAt"].replace("Z", "+00:00")
                        )
                        success_time = datetime.fromisoformat(
                            run["createdAt"].replace("Z", "+00:00")
                        )
                        recovery_minutes = (
                            success_time - failure_time
                        ).total_seconds() / 60

                        if recovery_minutes > 0:  # Sanity check
                            recovery_times.append(recovery_minutes)

                        previous_failure = None

                if recovery_times:
                    mttr = sum(recovery_times) / len(recovery_times)
                    logger.info(f"Mean time to recovery: {mttr:.1f} minutes")
                    return mttr

            return 30.0  # Default 30 minutes if no data

        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.error(f"Failed to calculate MTTR: {e}")
            return 30.0

    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all DORA metrics.

        Returns:
            Dictionary containing all four DORA metrics
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "deployment_frequency": self.collect_deployment_frequency(),
            "lead_time_hours": self.collect_lead_time(),
            "change_failure_rate": self.collect_change_failure_rate(),
            "mttr_minutes": self.collect_mttr(),
        }

        # Calculate performance level based on DORA research
        metrics["performance_level"] = self._calculate_performance_level(metrics)

        # Store in history
        self.metrics_history.append(metrics)
        self.metrics_cache = metrics

        return metrics

    def _calculate_performance_level(self, metrics: Dict[str, Any]) -> str:
        """Calculate performance level based on DORA research.

        Args:
            metrics: Dictionary of DORA metrics

        Returns:
            Performance level (Elite, High, Medium, Low)
        """
        # Based on DORA State of DevOps research
        df = metrics["deployment_frequency"]
        lt = metrics["lead_time_hours"]
        cfr = metrics["change_failure_rate"]
        mttr = metrics["mttr_minutes"]

        # Elite performers criteria
        if df >= 1 and lt <= 24 and cfr <= 15 and mttr <= 60:
            return "Elite"
        # High performers
        elif df >= 0.14 and lt <= 168 and cfr <= 30 and mttr <= 1440:
            return "High"
        # Medium performers
        elif df >= 0.03 and lt <= 720 and cfr <= 45 and mttr <= 10080:
            return "Medium"
        else:
            return "Low"

    def generate_report(self) -> str:
        """Generate a human-readable DORA metrics report.

        Returns:
            Formatted report string
        """
        if not self.metrics_cache:
            self.collect_all_metrics()

        m = self.metrics_cache

        report = f"""
# DORA Metrics Report
Generated: {m["timestamp"]}

## Performance Level: {m["performance_level"]}

### Metrics Summary
- **Deployment Frequency**: {m["deployment_frequency"]:.2f} deployments/day
- **Lead Time for Changes**: {m["lead_time_hours"]:.1f} hours
- **Change Failure Rate**: {m["change_failure_rate"]:.1f}%
- **Mean Time to Recovery**: {m["mttr_minutes"]:.1f} minutes

### Recommendations
"""

        # Add specific recommendations based on metrics
        if m["deployment_frequency"] < 1:
            report += "- ⚡ Increase deployment frequency with smaller, more frequent releases\n"

        if m["lead_time_hours"] > 24:
            report += "- 🔄 Reduce lead time by optimizing CI/CD pipeline and review process\n"

        if m["change_failure_rate"] > 15:
            report += "- 🛡️ Improve testing and quality gates to reduce failure rate\n"

        if m["mttr_minutes"] > 60:
            report += "- 🚨 Implement better monitoring and rollback procedures for faster recovery\n"

        return report

    def export_metrics(self, filepath: Path) -> None:
        """Export metrics history to JSON file.

        Args:
            filepath: Path to save metrics file
        """
        data = {
            "current": self.metrics_cache,
            "history": self.metrics_history,
            "generated_at": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported DORA metrics to {filepath}")


# Create singleton instance
dora_collector = DORAMetricsCollector()


def get_dora_metrics() -> Dict[str, Any]:
    """Get current DORA metrics.

    Returns:
        Dictionary of DORA metrics
    """
    return dora_collector.collect_all_metrics()


def generate_dora_report() -> str:
    """Generate DORA metrics report.

    Returns:
        Formatted report string
    """
    return dora_collector.generate_report()
