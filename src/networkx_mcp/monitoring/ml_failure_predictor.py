"""
ML-Powered Predictive Failure Detection for CI/CD Pipelines

This module implements machine learning models to predict CI/CD failures
before they occur, enabling proactive intervention and self-healing mechanisms.
"""

import json
import logging
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import asyncio
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for build predictions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BuildFeatures:
    """Features extracted from build metadata"""
    # Code complexity features
    files_changed: int = 0
    lines_added: int = 0
    lines_deleted: int = 0
    test_files_changed: int = 0
    config_files_changed: int = 0
    
    # Historical features
    author_failure_rate: float = 0.0
    author_experience: int = 0  # Number of previous commits
    branch_failure_rate: float = 0.0
    time_since_last_build: float = 0.0  # Hours
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    
    # Temporal features
    hour_of_day: int = 0
    day_of_week: int = 0
    is_weekend: bool = False
    is_end_of_sprint: bool = False
    
    # Dependency features
    dependency_changes: int = 0
    security_vulnerabilities: int = 0
    outdated_dependencies: int = 0
    
    # Team features
    reviewer_count: int = 0
    comment_count: int = 0
    approval_count: int = 0
    time_to_merge: float = 0.0  # Hours
    
    # Environmental features
    runner_type: str = "ubuntu-latest"
    parallel_jobs: int = 1
    cache_hit_rate: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for ML model"""
        return np.array([
            self.files_changed,
            self.lines_added,
            self.lines_deleted,
            self.test_files_changed,
            self.config_files_changed,
            self.author_failure_rate,
            self.author_experience,
            self.branch_failure_rate,
            self.time_since_last_build,
            self.consecutive_successes,
            self.consecutive_failures,
            self.hour_of_day,
            self.day_of_week,
            float(self.is_weekend),
            float(self.is_end_of_sprint),
            self.dependency_changes,
            self.security_vulnerabilities,
            self.outdated_dependencies,
            self.reviewer_count,
            self.comment_count,
            self.approval_count,
            self.time_to_merge,
            self.parallel_jobs,
            self.cache_hit_rate,
            hash(self.runner_type) % 1000  # Simple hash for categorical
        ])


@dataclass
class PredictionResult:
    """Result of failure prediction"""
    failure_probability: float
    risk_level: RiskLevel
    confidence: float
    contributing_factors: List[str]
    recommended_actions: List[str]
    estimated_duration: float  # Minutes
    resource_recommendations: Dict[str, Any]
    

class SimpleMLModel:
    """Simplified ML model for environments without scikit-learn"""
    
    def __init__(self):
        self.weights = None
        self.bias = 0.0
        self.feature_importance = None
        self.threshold_high = 0.7
        self.threshold_medium = 0.4
        self.threshold_low = 0.2
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train using simple logistic regression via gradient descent"""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Simple gradient descent
        learning_rate = 0.01
        n_iterations = 1000
        
        for _ in range(n_iterations):
            # Forward propagation
            z = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(z)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
        
        # Calculate feature importance (absolute weights)
        self.feature_importance = np.abs(self.weights) / np.sum(np.abs(self.weights))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of failure"""
        z = np.dot(X, self.weights) + self.bias
        probabilities = self._sigmoid(z)
        return np.column_stack([1 - probabilities, probabilities])
    
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


class BuildFailurePredictor:
    """Main predictor class for CI/CD failure prediction"""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = SimpleMLModel()
        self.model_path = model_path or Path(".ml_models/failure_predictor.pkl")
        self.feature_names = [
            "files_changed", "lines_added", "lines_deleted", "test_files_changed",
            "config_files_changed", "author_failure_rate", "author_experience",
            "branch_failure_rate", "time_since_last_build", "consecutive_successes",
            "consecutive_failures", "hour_of_day", "day_of_week", "is_weekend",
            "is_end_of_sprint", "dependency_changes", "security_vulnerabilities",
            "outdated_dependencies", "reviewer_count", "comment_count",
            "approval_count", "time_to_merge", "parallel_jobs", "cache_hit_rate",
            "runner_type_hash"
        ]
        self.historical_data = deque(maxlen=1000)
        self.author_stats = defaultdict(lambda: {"failures": 0, "total": 0})
        self.branch_stats = defaultdict(lambda: {"failures": 0, "total": 0})
        
        # Load model if exists
        self._load_model()
    
    def extract_features(self, build_data: Dict[str, Any]) -> BuildFeatures:
        """Extract features from build metadata"""
        features = BuildFeatures()
        
        # Extract code changes
        if "changes" in build_data:
            features.files_changed = build_data["changes"].get("files", 0)
            features.lines_added = build_data["changes"].get("additions", 0)
            features.lines_deleted = build_data["changes"].get("deletions", 0)
            features.test_files_changed = build_data["changes"].get("test_files", 0)
            features.config_files_changed = build_data["changes"].get("config_files", 0)
        
        # Extract author stats
        author = build_data.get("author", "unknown")
        if author in self.author_stats:
            stats = self.author_stats[author]
            features.author_failure_rate = stats["failures"] / max(stats["total"], 1)
            features.author_experience = stats["total"]
        
        # Extract branch stats
        branch = build_data.get("branch", "main")
        if branch in self.branch_stats:
            stats = self.branch_stats[branch]
            features.branch_failure_rate = stats["failures"] / max(stats["total"], 1)
        
        # Extract temporal features
        timestamp = build_data.get("timestamp", datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        features.hour_of_day = timestamp.hour
        features.day_of_week = timestamp.weekday()
        features.is_weekend = timestamp.weekday() >= 5
        
        # Extract other features
        features.dependency_changes = build_data.get("dependency_changes", 0)
        features.security_vulnerabilities = build_data.get("vulnerabilities", 0)
        features.reviewer_count = build_data.get("reviewers", 0)
        features.comment_count = build_data.get("comments", 0)
        features.approval_count = build_data.get("approvals", 0)
        features.runner_type = build_data.get("runner", "ubuntu-latest")
        features.parallel_jobs = build_data.get("parallel_jobs", 1)
        features.cache_hit_rate = build_data.get("cache_hit_rate", 0.0)
        
        # Calculate consecutive patterns
        features.consecutive_successes = self._get_consecutive_successes(branch)
        features.consecutive_failures = self._get_consecutive_failures(branch)
        
        return features
    
    def predict(self, build_data: Dict[str, Any]) -> PredictionResult:
        """Predict build failure probability"""
        features = self.extract_features(build_data)
        feature_array = features.to_array().reshape(1, -1)
        
        # Get prediction
        probabilities = self.model.predict_proba(feature_array)
        failure_probability = probabilities[0][1]
        
        # Determine risk level
        if failure_probability >= 0.8:
            risk_level = RiskLevel.CRITICAL
        elif failure_probability >= 0.6:
            risk_level = RiskLevel.HIGH
        elif failure_probability >= 0.4:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # Calculate confidence (based on model certainty)
        confidence = abs(failure_probability - 0.5) * 2  # Maps [0.5, 1] to [0, 1]
        
        # Identify contributing factors
        contributing_factors = self._identify_contributing_factors(features, feature_array)
        
        # Generate recommendations
        recommended_actions = self._generate_recommendations(
            features, risk_level, contributing_factors
        )
        
        # Estimate build duration
        estimated_duration = self._estimate_duration(features)
        
        # Generate resource recommendations
        resource_recommendations = self._recommend_resources(features, risk_level)
        
        return PredictionResult(
            failure_probability=failure_probability,
            risk_level=risk_level,
            confidence=confidence,
            contributing_factors=contributing_factors,
            recommended_actions=recommended_actions,
            estimated_duration=estimated_duration,
            resource_recommendations=resource_recommendations
        )
    
    def _identify_contributing_factors(
        self, 
        features: BuildFeatures, 
        feature_array: np.ndarray
    ) -> List[str]:
        """Identify top contributing factors to failure risk"""
        factors = []
        
        if not hasattr(self.model, 'feature_importance') or self.model.feature_importance is None:
            # Fallback to rule-based factors
            if features.security_vulnerabilities > 0:
                factors.append(f"Security vulnerabilities detected: {features.security_vulnerabilities}")
            if features.author_failure_rate > 0.5:
                factors.append(f"High author failure rate: {features.author_failure_rate:.1%}")
            if features.consecutive_failures > 2:
                factors.append(f"Consecutive failures: {features.consecutive_failures}")
            if features.config_files_changed > 0:
                factors.append("Configuration files modified")
            if features.is_end_of_sprint:
                factors.append("End of sprint - higher risk period")
            if features.cache_hit_rate < 0.3:
                factors.append(f"Low cache hit rate: {features.cache_hit_rate:.1%}")
            if features.test_files_changed == 0 and features.files_changed > 5:
                factors.append("No test changes despite code modifications")
                
            return factors[:5]  # Top 5 factors
        
        # Use feature importance if available
        importance_indices = np.argsort(self.model.feature_importance)[::-1][:5]
        feature_values = feature_array[0]
        
        for idx in importance_indices:
            feature_name = self.feature_names[idx]
            feature_value = feature_values[idx]
            importance = self.model.feature_importance[idx]
            
            if importance > 0.05:  # Only include significant factors
                factors.append(
                    f"{feature_name.replace('_', ' ').title()}: "
                    f"{feature_value:.2f} (importance: {importance:.1%})"
                )
        
        return factors
    
    def _generate_recommendations(
        self,
        features: BuildFeatures,
        risk_level: RiskLevel,
        contributing_factors: List[str]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Critical risk recommendations
        if risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            recommendations.append("🚨 Request additional code review before merging")
            recommendations.append("🧪 Run extended test suite locally first")
            recommendations.append("👥 Pair with team member for deployment")
        
        # Factor-specific recommendations
        if features.security_vulnerabilities > 0:
            recommendations.append("🔒 Fix security vulnerabilities before proceeding")
        
        if features.test_files_changed == 0 and features.files_changed > 5:
            recommendations.append("✅ Add tests for modified code")
        
        if features.dependency_changes > 0:
            recommendations.append("📦 Verify dependency compatibility")
            recommendations.append("🔍 Run dependency vulnerability scan")
        
        if features.cache_hit_rate < 0.3:
            recommendations.append("💾 Investigate cache configuration")
        
        if features.consecutive_failures > 2:
            recommendations.append("🔧 Consider reverting recent changes")
            recommendations.append("📊 Review recent failure patterns")
        
        if features.is_weekend:
            recommendations.append("📅 Consider postponing to weekday for better support")
        
        if features.config_files_changed > 0:
            recommendations.append("⚙️ Double-check configuration changes")
            recommendations.append("🔄 Test in staging environment first")
        
        # Resource-based recommendations
        if features.parallel_jobs < 3 and features.files_changed > 20:
            recommendations.append("⚡ Increase parallel jobs for faster feedback")
        
        return recommendations[:7]  # Limit to 7 most relevant
    
    def _estimate_duration(self, features: BuildFeatures) -> float:
        """Estimate build duration in minutes"""
        # Base duration
        base_duration = 2.0
        
        # Adjust based on changes
        base_duration += features.files_changed * 0.1
        base_duration += features.test_files_changed * 0.3
        base_duration += features.dependency_changes * 2.0
        
        # Adjust based on parallelization
        base_duration = base_duration / max(features.parallel_jobs, 1)
        
        # Adjust based on cache
        base_duration *= (1 - features.cache_hit_rate * 0.3)
        
        return round(base_duration, 1)
    
    def _recommend_resources(
        self,
        features: BuildFeatures,
        risk_level: RiskLevel
    ) -> Dict[str, Any]:
        """Recommend resource allocation"""
        recommendations = {
            "runner_type": features.runner_type,
            "parallel_jobs": features.parallel_jobs,
            "timeout_minutes": 30,
            "retry_count": 1
        }
        
        # High-risk builds get more resources
        if risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            recommendations["runner_type"] = "ubuntu-latest-8-core"
            recommendations["parallel_jobs"] = min(features.parallel_jobs * 2, 10)
            recommendations["timeout_minutes"] = 45
            recommendations["retry_count"] = 2
        
        # Large changesets need more resources
        if features.files_changed > 50:
            recommendations["runner_type"] = "ubuntu-latest-8-core"
            recommendations["parallel_jobs"] = min(8, features.parallel_jobs * 2)
        
        # Dependency changes need more time
        if features.dependency_changes > 0:
            recommendations["timeout_minutes"] = 60
        
        return recommendations
    
    def update_history(self, build_data: Dict[str, Any], success: bool):
        """Update historical data with build result"""
        # Update author stats
        author = build_data.get("author", "unknown")
        self.author_stats[author]["total"] += 1
        if not success:
            self.author_stats[author]["failures"] += 1
        
        # Update branch stats
        branch = build_data.get("branch", "main")
        self.branch_stats[branch]["total"] += 1
        if not success:
            self.branch_stats[branch]["failures"] += 1
        
        # Store in history
        self.historical_data.append({
            "timestamp": datetime.now().isoformat(),
            "build_data": build_data,
            "success": success
        })
        
        # Retrain model periodically
        if len(self.historical_data) % 100 == 0:
            self._retrain_model()
    
    def _get_consecutive_successes(self, branch: str) -> int:
        """Count consecutive successes for branch"""
        count = 0
        for record in reversed(self.historical_data):
            if record["build_data"].get("branch") == branch:
                if record["success"]:
                    count += 1
                else:
                    break
        return count
    
    def _get_consecutive_failures(self, branch: str) -> int:
        """Count consecutive failures for branch"""
        count = 0
        for record in reversed(self.historical_data):
            if record["build_data"].get("branch") == branch:
                if not record["success"]:
                    count += 1
                else:
                    break
        return count
    
    def _retrain_model(self):
        """Retrain model with recent data"""
        if len(self.historical_data) < 50:
            return
        
        # Prepare training data
        X = []
        y = []
        
        for record in self.historical_data:
            features = self.extract_features(record["build_data"])
            X.append(features.to_array())
            y.append(0 if record["success"] else 1)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train model
        self.model.fit(X, y)
        
        # Save model
        self._save_model()
        
        logger.info(f"Model retrained with {len(X)} samples")
    
    def _save_model(self):
        """Save model to disk"""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'author_stats': dict(self.author_stats),
                'branch_stats': dict(self.branch_stats),
                'historical_data': list(self.historical_data)
            }, f)
    
    def _load_model(self):
        """Load model from disk"""
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.author_stats = defaultdict(
                        lambda: {"failures": 0, "total": 0},
                        data['author_stats']
                    )
                    self.branch_stats = defaultdict(
                        lambda: {"failures": 0, "total": 0},
                        data['branch_stats']
                    )
                    self.historical_data = deque(
                        data['historical_data'],
                        maxlen=1000
                    )
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")


class PredictiveMonitoringService:
    """Service for continuous predictive monitoring"""
    
    def __init__(self, predictor: BuildFailurePredictor):
        self.predictor = predictor
        self.session: Optional[aiohttp.ClientSession] = None
        self.monitoring_interval = 60  # seconds
        self.github_token = None  # Set via environment variable
    
    async def start(self):
        """Start monitoring service"""
        self.session = aiohttp.ClientSession()
        logger.info("Predictive monitoring service started")
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
    
    async def stop(self):
        """Stop monitoring service"""
        if self.session:
            await self.session.close()
        logger.info("Predictive monitoring service stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                await self._check_pending_builds()
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _check_pending_builds(self):
        """Check and predict for pending builds"""
        # This would integrate with GitHub API to check pending PRs/builds
        # For now, this is a placeholder
        pass
    
    async def analyze_pr(self, pr_data: Dict[str, Any]) -> PredictionResult:
        """Analyze a pull request for failure risk"""
        # Extract build data from PR
        build_data = {
            "author": pr_data.get("user", {}).get("login"),
            "branch": pr_data.get("head", {}).get("ref"),
            "timestamp": pr_data.get("created_at"),
            "changes": {
                "files": pr_data.get("changed_files", 0),
                "additions": pr_data.get("additions", 0),
                "deletions": pr_data.get("deletions", 0)
            },
            "reviewers": len(pr_data.get("requested_reviewers", [])),
            "comments": pr_data.get("comments", 0),
            "approvals": pr_data.get("review_comments", 0)
        }
        
        # Get prediction
        return self.predictor.predict(build_data)


# Example usage
async def main():
    """Example usage of predictive failure detection"""
    
    # Initialize predictor
    predictor = BuildFailurePredictor()
    
    # Example build data
    build_data = {
        "author": "developer123",
        "branch": "feature/new-feature",
        "timestamp": datetime.now().isoformat(),
        "changes": {
            "files": 15,
            "additions": 500,
            "deletions": 200,
            "test_files": 3,
            "config_files": 1
        },
        "dependency_changes": 2,
        "vulnerabilities": 0,
        "reviewers": 2,
        "comments": 5,
        "approvals": 1,
        "runner": "ubuntu-latest",
        "parallel_jobs": 4,
        "cache_hit_rate": 0.75
    }
    
    # Get prediction
    result = predictor.predict(build_data)
    
    print(f"Failure Probability: {result.failure_probability:.1%}")
    print(f"Risk Level: {result.risk_level.value.upper()}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Estimated Duration: {result.estimated_duration} minutes")
    print("\nContributing Factors:")
    for factor in result.contributing_factors:
        print(f"  • {factor}")
    print("\nRecommended Actions:")
    for action in result.recommended_actions:
        print(f"  {action}")
    print("\nResource Recommendations:")
    for key, value in result.resource_recommendations.items():
        print(f"  • {key}: {value}")
    
    # Update with result (for learning)
    predictor.update_history(build_data, success=True)


if __name__ == "__main__":
    asyncio.run(main())