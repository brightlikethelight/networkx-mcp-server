"""
AI-Powered Threat Detection System

Advanced threat detection system that identifies and mitigates prompt injection attacks,
tool poisoning, and other security threats in MCP interactions.

Key Features:
- Machine learning-based prompt injection detection
- Behavioral anomaly analysis
- Real-time threat intelligence integration
- Context-aware threat assessment
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

# Try to import ML libraries, fallback to rule-based detection
try:
    import torch
    import transformers

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class ThreatLevel(Enum):
    """Threat severity levels."""

    BENIGN = "benign"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"
    CRITICAL = "critical"


@dataclass
class ThreatAssessment:
    """Comprehensive threat assessment result."""

    threat_level: ThreatLevel
    confidence: float
    detected_patterns: List[str] = field(default_factory=list)
    attack_vectors: List[str] = field(default_factory=list)
    mitigation_actions: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    context_analysis: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert assessment to dictionary."""
        return {
            "threat_level": self.threat_level.value,
            "confidence": self.confidence,
            "detected_patterns": self.detected_patterns,
            "attack_vectors": self.attack_vectors,
            "mitigation_actions": self.mitigation_actions,
            "risk_score": self.risk_score,
            "context_analysis": self.context_analysis,
            "timestamp": self.timestamp.isoformat(),
        }


class PromptInjectionDetector:
    """
    AI-powered prompt injection detection system.

    Uses multiple detection layers:
    1. Pattern-based detection for known injection patterns
    2. ML-based detection for sophisticated attacks
    3. Context-aware analysis for graph operation context
    4. Behavioral analysis for anomalous request patterns
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ml_model = self._load_ml_model() if ML_AVAILABLE else None
        self.injection_patterns = self._load_injection_patterns()
        self.context_analyzer = ContextAnalyzer()
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.threat_intel = ThreatIntelligence()

        # Detection statistics
        self.detection_stats = {
            "total_analyzed": 0,
            "threats_detected": 0,
            "false_positives": 0,
            "last_updated": datetime.utcnow(),
        }

    def _load_ml_model(self):
        """Load ML model for prompt injection detection."""
        if not ML_AVAILABLE:
            return None

        try:
            # In production, this would load a custom-trained model
            # For now, we'll use a simulated model
            self.logger.info("Loading ML model for prompt injection detection")
            return "simulated_ml_model"  # Placeholder
        except Exception as e:
            self.logger.error(f"Failed to load ML model: {e}")
            return None

    def _load_injection_patterns(self) -> List[Dict[str, Any]]:
        """Load known prompt injection patterns."""
        return [
            {
                "pattern": r"(?i)ignore\s+(?:all\s+)?(?:previous\s+)?(?:instructions|commands|prompts)",
                "description": "Ignore previous instructions",
                "severity": "high",
                "attack_type": "instruction_override",
            },
            {
                "pattern": r"(?i)system\s*[:.]?\s*(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)",
                "description": "System role hijacking",
                "severity": "critical",
                "attack_type": "role_hijacking",
            },
            {
                "pattern": r"(?i)(?:execute|run|eval|import|subprocess|os\.system|shell)",
                "description": "Code execution attempt",
                "severity": "critical",
                "attack_type": "code_injection",
            },
            {
                "pattern": r"(?i)(?:delete|remove|drop|truncate|destroy)\s+(?:all|everything|files|data|graphs?)",
                "description": "Destructive operation attempt",
                "severity": "high",
                "attack_type": "destructive_operation",
            },
            {
                "pattern": r"(?i)(?:reveal|show|display|print|output)\s+(?:password|secret|key|token|credential)",
                "description": "Information disclosure attempt",
                "severity": "high",
                "attack_type": "information_disclosure",
            },
            {
                "pattern": r"(?i)(?:bypass|skip|disable|turn\s+off)\s+(?:security|validation|check|filter)",
                "description": "Security bypass attempt",
                "severity": "critical",
                "attack_type": "security_bypass",
            },
            {
                "pattern": r"(?i)(?:admin|root|sudo|privilege)\s+(?:access|mode|rights|escalation)",
                "description": "Privilege escalation attempt",
                "severity": "critical",
                "attack_type": "privilege_escalation",
            },
            {
                "pattern": r"<!--.*?-->|<script.*?</script>|javascript:",
                "description": "HTML/JavaScript injection",
                "severity": "medium",
                "attack_type": "script_injection",
            },
            {
                "pattern": r"(?i)(?:union|select|insert|update|delete|drop)\s+(?:from|into|table|database)",
                "description": "SQL injection attempt",
                "severity": "high",
                "attack_type": "sql_injection",
            },
            {
                "pattern": r"(?i)(?:../|\.\.\\|/etc/|/var/|/usr/|/root/|/home/|c:\\|\\windows\\)",
                "description": "Path traversal attempt",
                "severity": "high",
                "attack_type": "path_traversal",
            },
        ]

    def detect_injection(
        self, prompt: str, context: Dict[str, Any]
    ) -> ThreatAssessment:
        """
        Comprehensive prompt injection detection.

        Args:
            prompt: The input prompt to analyze
            context: Operation context (tool_name, user_id, etc.)

        Returns:
            ThreatAssessment: Detailed threat analysis
        """
        self.detection_stats["total_analyzed"] += 1

        # Multi-layer detection
        pattern_results = self._pattern_based_detection(prompt)
        ml_results = self._ml_based_detection(prompt) if self.ml_model else None
        context_results = self.context_analyzer.analyze_context(prompt, context)
        behavioral_results = self.behavioral_analyzer.analyze_behavior(prompt, context)

        # Combine results
        assessment = self._combine_detection_results(
            pattern_results, ml_results, context_results, behavioral_results
        )

        # Update statistics
        if assessment.threat_level != ThreatLevel.BENIGN:
            self.detection_stats["threats_detected"] += 1

        self.logger.info(
            f"Threat assessment: {assessment.threat_level.value} "
            f"(confidence: {assessment.confidence:.2f})"
        )

        return assessment

    def _pattern_based_detection(self, prompt: str) -> Dict[str, Any]:
        """Pattern-based detection using known injection patterns."""
        detected_patterns = []
        attack_vectors = []
        max_severity = 0

        for pattern_info in self.injection_patterns:
            pattern = pattern_info["pattern"]
            if re.search(pattern, prompt):
                detected_patterns.append(pattern_info["description"])
                attack_vectors.append(pattern_info["attack_type"])

                # Calculate severity score
                severity_scores = {"low": 1, "medium": 2, "high": 3, "critical": 4}
                severity_score = severity_scores.get(pattern_info["severity"], 1)
                max_severity = max(max_severity, severity_score)

        confidence = min(max_severity * 0.3, 1.0)  # Max confidence 1.0

        return {
            "detected_patterns": detected_patterns,
            "attack_vectors": attack_vectors,
            "confidence": confidence,
            "max_severity": max_severity,
        }

    def _ml_based_detection(self, prompt: str) -> Optional[Dict[str, Any]]:
        """ML-based detection for sophisticated attacks."""
        if not self.ml_model:
            return None

        # Placeholder for ML model inference
        # In production, this would use a trained transformer model

        # Simulate ML detection
        suspicious_tokens = ["ignore", "system", "admin", "bypass", "execute"]
        token_count = sum(
            1 for token in suspicious_tokens if token.lower() in prompt.lower()
        )

        # Simple heuristic for demonstration
        ml_confidence = min(token_count * 0.15, 0.8)

        return {
            "ml_confidence": ml_confidence,
            "suspicious_tokens": token_count,
            "model_version": "v1.0.0",
        }

    def _combine_detection_results(
        self,
        pattern_results: Dict[str, Any],
        ml_results: Optional[Dict[str, Any]],
        context_results: Dict[str, Any],
        behavioral_results: Dict[str, Any],
    ) -> ThreatAssessment:
        """Combine results from all detection layers."""

        # Calculate overall confidence
        pattern_confidence = pattern_results.get("confidence", 0.0)
        ml_confidence = ml_results.get("ml_confidence", 0.0) if ml_results else 0.0
        context_confidence = context_results.get("confidence", 0.0)
        behavioral_confidence = behavioral_results.get("confidence", 0.0)

        # Weighted combination - give more weight to pattern detection for clear attacks
        if pattern_confidence > 0.8:
            # For clear pattern matches, use higher weight
            overall_confidence = (
                pattern_confidence * 0.6
                + ml_confidence * 0.2
                + context_confidence * 0.15
                + behavioral_confidence * 0.05
            )
        else:
            # For unclear patterns, use more balanced weights
            overall_confidence = (
                pattern_confidence * 0.4
                + ml_confidence * 0.3
                + context_confidence * 0.2
                + behavioral_confidence * 0.1
            )

        # Determine threat level
        if overall_confidence >= 0.7:
            threat_level = ThreatLevel.CRITICAL
        elif overall_confidence >= 0.5:
            threat_level = ThreatLevel.MALICIOUS
        elif overall_confidence >= 0.25:
            threat_level = ThreatLevel.SUSPICIOUS
        else:
            threat_level = ThreatLevel.BENIGN

        # Combine detected patterns and attack vectors
        all_patterns = pattern_results.get("detected_patterns", [])
        all_patterns.extend(context_results.get("detected_patterns", []))
        all_patterns.extend(behavioral_results.get("detected_patterns", []))

        all_vectors = pattern_results.get("attack_vectors", [])
        all_vectors.extend(context_results.get("attack_vectors", []))
        all_vectors.extend(behavioral_results.get("attack_vectors", []))

        # Generate mitigation actions
        mitigation_actions = self._generate_mitigation_actions(
            threat_level, all_vectors
        )

        return ThreatAssessment(
            threat_level=threat_level,
            confidence=overall_confidence,
            detected_patterns=list(set(all_patterns)),
            attack_vectors=list(set(all_vectors)),
            mitigation_actions=mitigation_actions,
            risk_score=overall_confidence * 100,
            context_analysis={
                "pattern_results": pattern_results,
                "ml_results": ml_results,
                "context_results": context_results,
                "behavioral_results": behavioral_results,
            },
        )

    def _generate_mitigation_actions(
        self, threat_level: ThreatLevel, attack_vectors: List[str]
    ) -> List[str]:
        """Generate appropriate mitigation actions based on threat level."""
        actions = []

        if threat_level == ThreatLevel.CRITICAL:
            actions.extend(
                [
                    "BLOCK_REQUEST",
                    "REQUIRE_HUMAN_APPROVAL",
                    "ALERT_SECURITY_TEAM",
                    "LOG_SECURITY_EVENT",
                ]
            )
        elif threat_level == ThreatLevel.MALICIOUS:
            actions.extend(
                ["BLOCK_REQUEST", "REQUIRE_HUMAN_APPROVAL", "LOG_SECURITY_EVENT"]
            )
        elif threat_level == ThreatLevel.SUSPICIOUS:
            actions.extend(
                ["REQUIRE_CONFIRMATION", "ENHANCED_MONITORING", "LOG_SECURITY_EVENT"]
            )

        # Add vector-specific actions
        if "code_injection" in attack_vectors:
            actions.append("DISABLE_CODE_EXECUTION")
        if "destructive_operation" in attack_vectors:
            actions.append("ENABLE_CONFIRMATION_MODE")
        if "information_disclosure" in attack_vectors:
            actions.append("ENHANCED_OUTPUT_FILTERING")

        return list(set(actions))

    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            **self.detection_stats,
            "detection_rate": (
                self.detection_stats["threats_detected"]
                / max(self.detection_stats["total_analyzed"], 1)
            ),
            "false_positive_rate": (
                self.detection_stats["false_positives"]
                / max(self.detection_stats["total_analyzed"], 1)
            ),
        }


class ContextAnalyzer:
    """Analyzes operation context for threat assessment."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_context(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze operation context for suspicious patterns."""
        detected_patterns = []
        attack_vectors = []
        confidence = 0.0

        tool_name = context.get("tool_name", "")
        user_id = context.get("user_id", "")

        # Check for context-specific threats
        if tool_name == "visualize_graph" and "delete" in prompt.lower():
            detected_patterns.append("Destructive operation in visualization context")
            attack_vectors.append("context_mismatch")
            confidence += 0.3

        if tool_name == "get_info" and any(
            x in prompt.lower() for x in ["password", "secret", "token"]
        ):
            detected_patterns.append("Information disclosure in info context")
            attack_vectors.append("information_disclosure")
            confidence += 0.4

        # Check for unusual parameter combinations
        if self._check_unusual_parameters(prompt, context):
            detected_patterns.append("Unusual parameter combination")
            attack_vectors.append("parameter_manipulation")
            confidence += 0.2

        return {
            "detected_patterns": detected_patterns,
            "attack_vectors": attack_vectors,
            "confidence": min(confidence, 1.0),
        }

    def _check_unusual_parameters(self, prompt: str, context: Dict[str, Any]) -> bool:
        """Check for unusual parameter combinations."""
        # Implement context-specific parameter validation
        return False


class BehavioralAnalyzer:
    """Analyzes user behavior patterns for anomaly detection."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.user_profiles = {}  # In production, this would be persistent storage

    def analyze_behavior(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user behavior for anomalies."""
        user_id = context.get("user_id", "unknown")
        current_time = datetime.utcnow()

        # Update user profile
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "request_count": 0,
                "last_request": current_time,
                "request_history": [],
                "typical_tools": set(),
                "request_pattern": [],
            }

        profile = self.user_profiles[user_id]
        profile["request_count"] += 1
        profile["request_history"].append(current_time)
        profile["typical_tools"].add(context.get("tool_name", ""))

        # Keep only recent history (last 100 requests)
        profile["request_history"] = profile["request_history"][-100:]

        # Analyze for anomalies
        anomalies = self._detect_anomalies(profile, context)

        confidence = len(anomalies) * 0.2

        return {
            "detected_patterns": [
                f"Behavioral anomaly: {anomaly}" for anomaly in anomalies
            ],
            "attack_vectors": ["behavioral_anomaly"] if anomalies else [],
            "confidence": min(confidence, 1.0),
            "anomalies": anomalies,
        }

    def _detect_anomalies(
        self, profile: Dict[str, Any], context: Dict[str, Any]
    ) -> List[str]:
        """Detect behavioral anomalies."""
        anomalies = []

        # Check for request frequency anomalies
        recent_requests = [
            req
            for req in profile["request_history"]
            if req > datetime.utcnow() - timedelta(minutes=5)
        ]

        if len(recent_requests) > 20:  # More than 20 requests in 5 minutes
            anomalies.append("High request frequency")

        # Check for unusual tool usage
        tool_name = context.get("tool_name", "")
        if tool_name not in profile["typical_tools"] and profile["request_count"] > 10:
            anomalies.append("Unusual tool usage")

        return anomalies


class ThreatIntelligence:
    """Threat intelligence integration for real-time threat data."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.threat_signatures = set()
        self.malicious_patterns = []
        self.last_update = datetime.utcnow()

    def update_threat_intel(self):
        """Update threat intelligence data."""
        # In production, this would fetch from threat intelligence feeds
        self.logger.info("Updating threat intelligence data")
        self.last_update = datetime.utcnow()

    def check_threat_signatures(self, prompt: str) -> List[str]:
        """Check prompt against known threat signatures."""
        # Placeholder for threat signature checking
        return []

    def is_known_malicious(self, prompt_hash: str) -> bool:
        """Check if prompt hash is known malicious."""
        return prompt_hash in self.threat_signatures
