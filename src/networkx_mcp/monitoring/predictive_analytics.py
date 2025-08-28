"""Predictive analytics for CI/CD failure detection.

Uses machine learning to predict potential CI/CD failures before they occur,
enabling proactive intervention and resource optimization.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BuildFeatures:
    """Features extracted from a build for prediction."""

    # Code change features
    files_changed: int
    lines_added: int
    lines_removed: int
    file_types: List[str]

    # Timing features
    hour_of_day: int
    day_of_week: int
    time_since_last_commit: float

    # Historical features
    author_failure_rate: float
    branch_failure_rate: float
    recent_failure_count: int

    # Dependency features
    dependencies_changed: bool
    config_files_changed: bool
    test_files_changed: bool

    # Environmental features
    python_version: str
    os_type: str
    is_pull_request: bool


class CIPredictiveAnalyzer:
    """Predictive analyzer for CI/CD pipeline failures."""

    def __init__(self, history_days: int = 30):
        """Initialize predictive analyzer.

        Args:
            history_days: Days of history to analyze
        """
        self.history_days = history_days
        self.failure_patterns = self._initialize_patterns()
        self.model = None  # Will be trained on first use
        self.feature_importance = {}

    def _initialize_patterns(self) -> Dict[str, float]:
        """Initialize known failure patterns with risk scores.

        Returns:
            Dictionary of patterns and their risk scores
        """
        return {
            # High-risk patterns (0.7-1.0)
            "dependency_update": 0.8,
            "config_change": 0.75,
            "large_refactor": 0.85,
            "merge_conflict_resolution": 0.9,
            "force_push": 0.95,
            # Medium-risk patterns (0.4-0.7)
            "test_modification": 0.5,
            "ci_config_change": 0.6,
            "new_feature": 0.45,
            "documentation_only": 0.1,
            # Time-based patterns
            "friday_deployment": 0.7,
            "after_hours": 0.6,
            "monday_morning": 0.55,
            # Historical patterns
            "author_high_failure": 0.65,
            "flaky_test_area": 0.7,
            "unstable_branch": 0.6,
        }

    async def extract_build_features(
        self, commit_sha: str, repo_path: Optional[Path] = None
    ) -> BuildFeatures:
        """Extract features from a commit for prediction.

        Args:
            commit_sha: Git commit SHA
            repo_path: Repository path

        Returns:
            Extracted build features
        """
        repo_path = repo_path or Path.cwd()

        try:
            # Get commit information
            commit_info = await self._get_commit_info(commit_sha, repo_path)

            # Extract code change features
            files_changed = commit_info.get("files_changed", 0)
            lines_added = commit_info.get("lines_added", 0)
            lines_removed = commit_info.get("lines_removed", 0)
            file_types = self._extract_file_types(commit_info.get("files", []))

            # Extract timing features
            commit_time = datetime.fromisoformat(
                commit_info.get("timestamp", datetime.now().isoformat())
            )
            hour_of_day = commit_time.hour
            day_of_week = commit_time.weekday()

            # Calculate time since last commit
            last_commit_time = await self._get_last_commit_time(repo_path)
            time_since_last = (commit_time - last_commit_time).total_seconds() / 3600

            # Get historical features
            author = commit_info.get("author", "unknown")
            branch = commit_info.get("branch", "main")
            author_failure_rate = await self._get_author_failure_rate(author)
            branch_failure_rate = await self._get_branch_failure_rate(branch)
            recent_failures = await self._get_recent_failure_count()

            # Check for specific file changes
            changed_files = commit_info.get("files", [])
            dependencies_changed = any(
                "requirements" in f or "pyproject.toml" in f or "package" in f
                for f in changed_files
            )
            config_files_changed = any(
                ".yml" in f or ".yaml" in f or ".toml" in f or ".ini" in f
                for f in changed_files
            )
            test_files_changed = any("test" in f for f in changed_files)

            # Environmental features (would be passed in or detected)
            python_version = "3.12"
            os_type = "ubuntu-latest"
            is_pull_request = commit_info.get("is_pr", False)

            return BuildFeatures(
                files_changed=files_changed,
                lines_added=lines_added,
                lines_removed=lines_removed,
                file_types=file_types,
                hour_of_day=hour_of_day,
                day_of_week=day_of_week,
                time_since_last_commit=time_since_last,
                author_failure_rate=author_failure_rate,
                branch_failure_rate=branch_failure_rate,
                recent_failure_count=recent_failures,
                dependencies_changed=dependencies_changed,
                config_files_changed=config_files_changed,
                test_files_changed=test_files_changed,
                python_version=python_version,
                os_type=os_type,
                is_pull_request=is_pull_request,
            )

        except Exception as e:
            logger.error(f"Error extracting build features: {e}")
            # Return default features on error
            return BuildFeatures(
                files_changed=0,
                lines_added=0,
                lines_removed=0,
                file_types=[],
                hour_of_day=12,
                day_of_week=1,
                time_since_last_commit=24.0,
                author_failure_rate=0.1,
                branch_failure_rate=0.1,
                recent_failure_count=0,
                dependencies_changed=False,
                config_files_changed=False,
                test_files_changed=False,
                python_version="3.12",
                os_type="ubuntu-latest",
                is_pull_request=False,
            )

    async def _get_commit_info(
        self, commit_sha: str, repo_path: Path
    ) -> Dict[str, Any]:
        """Get commit information from git.

        Args:
            commit_sha: Commit SHA
            repo_path: Repository path

        Returns:
            Commit information
        """
        try:
            # Get commit details
            cmd = [
                "git",
                "-C",
                str(repo_path),
                "show",
                "--stat",
                "--format=%H|%an|%ae|%at|%s",
                commit_sha,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split("\n")

            # Parse commit info
            if lines:
                parts = lines[0].split("|")
                if len(parts) >= 5:
                    # Parse file statistics
                    files_changed = 0
                    lines_added = 0
                    lines_removed = 0
                    files = []

                    for line in lines[1:]:
                        if " file" in line and "changed" in line:
                            # Parse summary line
                            parts = line.split(",")
                            for part in parts:
                                if "file" in part:
                                    files_changed = int(part.strip().split()[0])
                                elif "insertion" in part:
                                    lines_added = int(part.strip().split()[0])
                                elif "deletion" in part:
                                    lines_removed = int(part.strip().split()[0])
                        elif "|" in line:
                            # File change line
                            file_path = line.split("|")[0].strip()
                            if file_path:
                                files.append(file_path)

                    return {
                        "sha": parts[0],
                        "author": parts[1],
                        "email": parts[2],
                        "timestamp": datetime.fromtimestamp(int(parts[3])).isoformat(),
                        "message": parts[4] if len(parts) > 4 else "",
                        "files_changed": files_changed,
                        "lines_added": lines_added,
                        "lines_removed": lines_removed,
                        "files": files,
                    }

        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting commit info: {e}")

        return {}

    def _extract_file_types(self, files: List[str]) -> List[str]:
        """Extract file types from changed files.

        Args:
            files: List of file paths

        Returns:
            List of unique file extensions
        """
        extensions = set()
        for file_path in files:
            path = Path(file_path)
            if path.suffix:
                extensions.add(path.suffix)
        return list(extensions)

    async def _get_last_commit_time(self, repo_path: Path) -> datetime:
        """Get timestamp of last commit before current.

        Args:
            repo_path: Repository path

        Returns:
            Last commit timestamp
        """
        try:
            cmd = [
                "git",
                "-C",
                str(repo_path),
                "log",
                "-2",
                "--format=%at",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            timestamps = result.stdout.strip().split("\n")

            if len(timestamps) > 1:
                return datetime.fromtimestamp(int(timestamps[1]))

        except subprocess.CalledProcessError:
            pass

        # Return 24 hours ago as default
        return datetime.now() - timedelta(hours=24)

    async def _get_author_failure_rate(self, author: str) -> float:
        """Get historical failure rate for an author.

        Args:
            author: Author name

        Returns:
            Failure rate (0-1)
        """
        try:
            # Get recent CI runs by this author
            cmd = [
                "gh",
                "run",
                "list",
                "--limit",
                "20",
                "--json",
                "conclusion,headBranch,event",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if result.stdout:
                runs = json.loads(result.stdout)
                # Simplified: return average failure rate
                failures = sum(1 for run in runs if run.get("conclusion") == "failure")
                return failures / max(len(runs), 1)

        except (subprocess.CalledProcessError, json.JSONDecodeError):
            pass

        return 0.1  # Default 10% failure rate

    async def _get_branch_failure_rate(self, branch: str) -> float:
        """Get historical failure rate for a branch.

        Args:
            branch: Branch name

        Returns:
            Failure rate (0-1)
        """
        try:
            cmd = [
                "gh",
                "run",
                "list",
                "--branch",
                branch,
                "--limit",
                "10",
                "--json",
                "conclusion",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if result.stdout:
                runs = json.loads(result.stdout)
                failures = sum(1 for run in runs if run.get("conclusion") == "failure")
                return failures / max(len(runs), 1)

        except (subprocess.CalledProcessError, json.JSONDecodeError):
            pass

        return 0.1  # Default 10% failure rate

    async def _get_recent_failure_count(self) -> int:
        """Get count of recent CI failures.

        Returns:
            Number of failures in last 24 hours
        """
        try:
            cmd = [
                "gh",
                "run",
                "list",
                "--limit",
                "10",
                "--json",
                "conclusion,createdAt",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            if result.stdout:
                runs = json.loads(result.stdout)
                cutoff = datetime.now() - timedelta(hours=24)

                recent_failures = 0
                for run in runs:
                    created_at = datetime.fromisoformat(
                        run["createdAt"].replace("Z", "+00:00")
                    )
                    if created_at > cutoff and run.get("conclusion") == "failure":
                        recent_failures += 1

                return recent_failures

        except (subprocess.CalledProcessError, json.JSONDecodeError):
            pass

        return 0

    def predict_failure_probability(
        self, features: BuildFeatures
    ) -> Tuple[float, Dict[str, float]]:
        """Predict probability of CI failure.

        Args:
            features: Build features

        Returns:
            Tuple of (failure probability, contributing factors)
        """
        # Simple rule-based prediction (would use ML model in production)
        risk_score = 0.1  # Base risk
        factors = {}

        # Code complexity factors
        if features.files_changed > 20:
            risk_score += 0.2
            factors["large_change"] = 0.2

        if features.lines_added + features.lines_removed > 500:
            risk_score += 0.15
            factors["many_lines_changed"] = 0.15

        # Dependency risk
        if features.dependencies_changed:
            risk_score += 0.25
            factors["dependency_change"] = 0.25

        # Configuration risk
        if features.config_files_changed:
            risk_score += 0.2
            factors["config_change"] = 0.2

        # Timing risk
        if features.day_of_week == 4:  # Friday
            risk_score += 0.15
            factors["friday_deployment"] = 0.15

        if features.hour_of_day < 6 or features.hour_of_day > 20:
            risk_score += 0.1
            factors["off_hours"] = 0.1

        # Historical risk
        if features.author_failure_rate > 0.3:
            risk_score += 0.2
            factors["high_author_failure_rate"] = 0.2

        if features.recent_failure_count > 3:
            risk_score += 0.15
            factors["recent_instability"] = 0.15

        # Test changes increase risk slightly
        if features.test_files_changed:
            risk_score += 0.1
            factors["test_changes"] = 0.1

        # Cap probability at 0.95
        probability = min(risk_score, 0.95)

        return probability, factors

    def generate_recommendations(
        self, probability: float, factors: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on failure prediction.

        Args:
            probability: Failure probability
            factors: Contributing factors

        Returns:
            List of recommendations
        """
        recommendations = []

        if probability > 0.7:
            recommendations.append("🚨 HIGH RISK: Consider additional manual review")
            recommendations.append("Run extended test suite locally before pushing")
        elif probability > 0.4:
            recommendations.append("⚠️ MEDIUM RISK: Monitor CI closely")

        # Factor-specific recommendations
        if "dependency_change" in factors:
            recommendations.append("Check dependency compatibility and lock files")
            recommendations.append("Run dependency audit for security vulnerabilities")

        if "large_change" in factors:
            recommendations.append("Consider breaking into smaller commits")
            recommendations.append("Request additional code review")

        if "config_change" in factors:
            recommendations.append("Validate configuration syntax")
            recommendations.append("Test in staging environment first")

        if "friday_deployment" in factors:
            recommendations.append("Consider postponing to Monday if non-critical")
            recommendations.append("Ensure on-call coverage is available")

        if "high_author_failure_rate" in factors:
            recommendations.append("Pair review with experienced team member")
            recommendations.append("Add extra test coverage for changed areas")

        if "recent_instability" in factors:
            recommendations.append("Check recent failure patterns for root cause")
            recommendations.append("Consider stabilization sprint")

        return recommendations

    async def analyze_current_risk(self) -> Dict[str, Any]:
        """Analyze current CI/CD risk level.

        Returns:
            Current risk analysis
        """
        try:
            # Get current HEAD commit
            cmd = ["git", "rev-parse", "HEAD"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            commit_sha = result.stdout.strip()

            # Extract features
            features = await self.extract_build_features(commit_sha)

            # Predict failure probability
            probability, factors = self.predict_failure_probability(features)

            # Generate recommendations
            recommendations = self.generate_recommendations(probability, factors)

            # Determine risk level
            if probability > 0.7:
                risk_level = "HIGH"
            elif probability > 0.4:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            return {
                "commit_sha": commit_sha,
                "risk_level": risk_level,
                "failure_probability": round(probability, 2),
                "contributing_factors": factors,
                "recommendations": recommendations,
                "analyzed_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error analyzing current risk: {e}")
            return {
                "error": str(e),
                "risk_level": "UNKNOWN",
                "failure_probability": 0.5,
            }


# Create singleton analyzer
predictive_analyzer = CIPredictiveAnalyzer()


async def predict_ci_failure() -> Dict[str, Any]:
    """Predict CI failure for current commit.

    Returns:
        Prediction results
    """
    return await predictive_analyzer.analyze_current_risk()
