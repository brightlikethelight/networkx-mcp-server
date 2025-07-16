"""
Security Broker/Firewall Architecture

Central security orchestration system that coordinates all security layers
and provides human-in-the-loop controls for sensitive operations.

Key Features:
- Security policy enforcement
- Human-in-the-loop approval workflows
- Risk-based operation assessment
- Real-time security decision making
- Comprehensive audit trail
"""

import time
import json
import uuid
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from .threat_detection import PromptInjectionDetector, ThreatAssessment, ThreatLevel
from .validation import ZeroTrustValidator, ValidationResult, ValidationStatus
from ..enterprise.auth import User, Role, Permission


class OperationRisk(Enum):
    """Operation risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalStatus(Enum):
    """Human approval status."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    TIMEOUT = "timeout"


@dataclass
class AuthorizationResult:
    """Security authorization result."""
    authorized: bool
    risk_level: OperationRisk
    requires_approval: bool = False
    approval_id: Optional[str] = None
    security_actions: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    mitigation_actions: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "authorized": self.authorized,
            "risk_level": self.risk_level.value,
            "requires_approval": self.requires_approval,
            "approval_id": self.approval_id,
            "security_actions": self.security_actions,
            "risk_factors": self.risk_factors,
            "mitigation_actions": self.mitigation_actions,
            "processing_time": self.processing_time,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ApprovalRequest:
    """Human approval request."""
    request_id: str
    user: User
    operation: str
    arguments: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    threat_assessment: ThreatAssessment
    validation_result: ValidationResult
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    denial_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary."""
        return {
            "request_id": self.request_id,
            "user": {
                "id": self.user.id,
                "username": self.user.username,
                "roles": [role.value for role in self.user.roles]
            },
            "operation": self.operation,
            "arguments": self.arguments,
            "risk_assessment": self.risk_assessment,
            "threat_assessment": self.threat_assessment.to_dict(),
            "validation_result": self.validation_result.to_dict(),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "denial_reason": self.denial_reason
        }


class SecurityBroker:
    """
    Central security broker that orchestrates all security layers.
    
    The Security Broker acts as the central decision-making component that:
    1. Coordinates threat detection, validation, and risk assessment
    2. Enforces security policies and access controls
    3. Manages human-in-the-loop approval workflows
    4. Provides comprehensive audit trail
    5. Implements real-time security monitoring
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.threat_detector = PromptInjectionDetector()
        self.validator = ZeroTrustValidator()
        self.risk_assessor = OperationRiskAssessor()
        self.approval_manager = HumanInLoopManager()
        self.security_policies = self._load_security_policies()
        
        # Security statistics
        self.security_stats = {
            "total_requests": 0,
            "authorized": 0,
            "denied": 0,
            "pending_approval": 0,
            "high_risk_operations": 0,
            "threats_detected": 0,
            "last_reset": datetime.utcnow()
        }
    
    def _load_security_policies(self) -> Dict[str, Any]:
        """Load security policies configuration."""
        return {
            "high_risk_operations": [
                "import_csv",
                "visualize_graph",
                "admin_reset_limits",
                "admin_manage_keys"
            ],
            "auto_approval_threshold": 0.3,  # Risk score threshold for auto-approval
            "human_approval_timeout": 300,  # 5 minutes
            "max_concurrent_approvals": 10,
            "require_approval_for_roles": [Role.GUEST],
            "blocked_operations_for_roles": {
                Role.GUEST: ["import_csv", "admin_reset_limits", "admin_manage_keys"],
                Role.READONLY: ["create_graph", "add_nodes", "add_edges", "import_csv"]
            }
        }
    
    async def authorize_operation(
        self, 
        user: User, 
        operation: str, 
        arguments: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> AuthorizationResult:
        """
        Comprehensive operation authorization.
        
        Args:
            user: User requesting the operation
            operation: Operation name
            arguments: Operation arguments
            context: Additional context information
            
        Returns:
            AuthorizationResult: Authorization decision and metadata
        """
        start_time = time.time()
        self.security_stats["total_requests"] += 1
        
        if context is None:
            context = {}
        
        try:
            # 1. Threat Detection
            prompt = self._construct_prompt(operation, arguments)
            threat_assessment = self.threat_detector.detect_injection(prompt, {
                "tool_name": operation,
                "user_id": user.id,
                **context
            })
            
            # 2. Input Validation
            validation_result = self.validator.validate_input(operation, arguments)
            
            # 3. Risk Assessment
            risk_assessment = self.risk_assessor.assess_risk(
                user, operation, arguments, threat_assessment, validation_result
            )
            
            # 4. Policy Enforcement
            policy_result = self._enforce_security_policies(
                user, operation, risk_assessment, threat_assessment, validation_result
            )
            
            # 5. Authorization Decision
            authorization_result = await self._make_authorization_decision(
                user, operation, arguments, risk_assessment, threat_assessment, 
                validation_result, policy_result
            )
            
            # Update statistics
            if authorization_result.authorized:
                self.security_stats["authorized"] += 1
            else:
                self.security_stats["denied"] += 1
            
            if authorization_result.requires_approval:
                self.security_stats["pending_approval"] += 1
            
            if authorization_result.risk_level in [OperationRisk.HIGH, OperationRisk.CRITICAL]:
                self.security_stats["high_risk_operations"] += 1
            
            if threat_assessment.threat_level != ThreatLevel.BENIGN:
                self.security_stats["threats_detected"] += 1
            
            authorization_result.processing_time = time.time() - start_time
            
            self.logger.info(f"Authorization result: {authorization_result.authorized} "
                           f"(risk: {authorization_result.risk_level.value}, "
                           f"approval: {authorization_result.requires_approval})")
            
            return authorization_result
            
        except Exception as e:
            self.logger.error(f"Authorization error: {e}")
            return AuthorizationResult(
                authorized=False,
                risk_level=OperationRisk.CRITICAL,
                security_actions=["BLOCK_REQUEST"],
                risk_factors=[f"Authorization error: {str(e)}"],
                processing_time=time.time() - start_time
            )
    
    def _construct_prompt(self, operation: str, arguments: Dict[str, Any]) -> str:
        """Construct prompt for threat detection."""
        return f"Operation: {operation}\\nArguments: {json.dumps(arguments)}"
    
    def _enforce_security_policies(
        self, 
        user: User, 
        operation: str, 
        risk_assessment: Dict[str, Any],
        threat_assessment: ThreatAssessment,
        validation_result: ValidationResult
    ) -> Dict[str, Any]:
        """Enforce security policies."""
        policy_violations = []
        policy_actions = []
        
        # Check role-based restrictions
        blocked_ops = self.security_policies.get("blocked_operations_for_roles", {})
        for role in user.roles:
            if role in blocked_ops and operation in blocked_ops[role]:
                policy_violations.append(f"Operation {operation} blocked for role {role.value}")
                policy_actions.append("BLOCK_REQUEST")
        
        # Check high-risk operations
        if operation in self.security_policies["high_risk_operations"]:
            policy_actions.append("REQUIRE_HUMAN_APPROVAL")
        
        # Check threat level policies
        if threat_assessment.threat_level == ThreatLevel.CRITICAL:
            policy_actions.extend(["BLOCK_REQUEST", "ALERT_SECURITY_TEAM"])
        elif threat_assessment.threat_level == ThreatLevel.MALICIOUS:
            policy_actions.extend(["REQUIRE_HUMAN_APPROVAL", "ENHANCED_MONITORING"])
        
        # Check validation policies
        if validation_result.status == ValidationStatus.BLOCKED:
            policy_actions.append("BLOCK_REQUEST")
        elif validation_result.status == ValidationStatus.FAILED:
            policy_actions.append("REQUIRE_HUMAN_APPROVAL")
        
        return {
            "violations": policy_violations,
            "actions": list(set(policy_actions))
        }
    
    async def _make_authorization_decision(
        self,
        user: User,
        operation: str,
        arguments: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        threat_assessment: ThreatAssessment,
        validation_result: ValidationResult,
        policy_result: Dict[str, Any]
    ) -> AuthorizationResult:
        """Make final authorization decision."""
        
        # Check for immediate blocks
        if "BLOCK_REQUEST" in policy_result["actions"]:
            return AuthorizationResult(
                authorized=False,
                risk_level=OperationRisk.CRITICAL,
                security_actions=["BLOCK_REQUEST"],
                risk_factors=policy_result["violations"],
                mitigation_actions=["Operation blocked by security policy"]
            )
        
        # Determine risk level
        risk_level = risk_assessment["risk_level"]
        
        # Check if human approval is required
        requires_approval = (
            "REQUIRE_HUMAN_APPROVAL" in policy_result["actions"] or
            risk_level in [OperationRisk.HIGH, OperationRisk.CRITICAL] or
            any(role in self.security_policies["require_approval_for_roles"] for role in user.roles)
        )
        
        # Handle human approval workflow
        approval_id = None
        if requires_approval:
            approval_request = ApprovalRequest(
                request_id=str(uuid.uuid4()),
                user=user,
                operation=operation,
                arguments=arguments,
                risk_assessment=risk_assessment,
                threat_assessment=threat_assessment,
                validation_result=validation_result,
                expires_at=datetime.utcnow() + timedelta(
                    seconds=self.security_policies["human_approval_timeout"]
                )
            )
            
            approval_id = await self.approval_manager.create_approval_request(approval_request)
        
        # Determine authorization
        authorized = not requires_approval  # If approval required, not yet authorized
        
        # Generate security actions
        security_actions = policy_result["actions"].copy()
        if requires_approval:
            security_actions.append("REQUIRE_HUMAN_APPROVAL")
        if risk_level == OperationRisk.HIGH:
            security_actions.append("ENHANCED_MONITORING")
        
        return AuthorizationResult(
            authorized=authorized,
            risk_level=risk_level,
            requires_approval=requires_approval,
            approval_id=approval_id,
            security_actions=security_actions,
            risk_factors=risk_assessment["risk_factors"],
            mitigation_actions=risk_assessment["mitigation_actions"]
        )
    
    async def check_approval_status(self, approval_id: str) -> Optional[ApprovalRequest]:
        """Check status of approval request."""
        return await self.approval_manager.get_approval_request(approval_id)
    
    async def approve_request(self, approval_id: str, approver_id: str) -> bool:
        """Approve a pending request."""
        return await self.approval_manager.approve_request(approval_id, approver_id)
    
    async def deny_request(self, approval_id: str, approver_id: str, reason: str) -> bool:
        """Deny a pending request."""
        return await self.approval_manager.deny_request(approval_id, approver_id, reason)
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        total = self.security_stats["total_requests"]
        return {
            **self.security_stats,
            "authorization_rate": self.security_stats["authorized"] / max(total, 1),
            "denial_rate": self.security_stats["denied"] / max(total, 1),
            "high_risk_rate": self.security_stats["high_risk_operations"] / max(total, 1),
            "threat_detection_rate": self.security_stats["threats_detected"] / max(total, 1)
        }


class OperationRiskAssessor:
    """Assesses risk level of operations based on multiple factors."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.risk_factors = self._load_risk_factors()
    
    def _load_risk_factors(self) -> Dict[str, Any]:
        """Load risk assessment factors."""
        return {
            "high_risk_operations": {
                "import_csv": 0.7,
                "visualize_graph": 0.5,
                "admin_reset_limits": 0.9,
                "admin_manage_keys": 0.95
            },
            "user_risk_factors": {
                "guest_user": 0.3,
                "new_user": 0.2,
                "suspicious_behavior": 0.5
            },
            "threat_level_multipliers": {
                ThreatLevel.BENIGN: 0.0,
                ThreatLevel.SUSPICIOUS: 0.3,
                ThreatLevel.MALICIOUS: 0.7,
                ThreatLevel.CRITICAL: 1.0
            },
            "validation_multipliers": {
                ValidationStatus.PASSED: 0.0,
                ValidationStatus.SANITIZED: 0.1,
                ValidationStatus.FAILED: 0.5,
                ValidationStatus.BLOCKED: 1.0
            }
        }
    
    def assess_risk(
        self,
        user: User,
        operation: str,
        arguments: Dict[str, Any],
        threat_assessment: ThreatAssessment,
        validation_result: ValidationResult
    ) -> Dict[str, Any]:
        """Comprehensive risk assessment."""
        
        risk_factors = []
        risk_score = 0.0
        
        # 1. Operation-based risk
        base_risk = self.risk_factors["high_risk_operations"].get(operation, 0.1)
        risk_score += base_risk
        if base_risk > 0.5:
            risk_factors.append(f"High-risk operation: {operation}")
        
        # 2. User-based risk
        if Role.GUEST in user.roles:
            risk_score += 0.3
            risk_factors.append("Guest user access")
        
        # 3. Threat-based risk
        threat_multiplier = self.risk_factors["threat_level_multipliers"][threat_assessment.threat_level]
        risk_score += threat_multiplier
        if threat_multiplier > 0.0:
            risk_factors.append(f"Threat detected: {threat_assessment.threat_level.value}")
        
        # 4. Validation-based risk
        validation_multiplier = self.risk_factors["validation_multipliers"][validation_result.status]
        risk_score += validation_multiplier
        if validation_multiplier > 0.0:
            risk_factors.append(f"Validation issues: {validation_result.status.value}")
        
        # 5. Argument-based risk
        if self._has_suspicious_arguments(arguments):
            risk_score += 0.2
            risk_factors.append("Suspicious arguments detected")
        
        # Normalize risk score
        risk_score = min(risk_score, 1.0)
        
        # Determine risk level
        if risk_score >= 0.8:
            risk_level = OperationRisk.CRITICAL
        elif risk_score >= 0.6:
            risk_level = OperationRisk.HIGH
        elif risk_score >= 0.3:
            risk_level = OperationRisk.MEDIUM
        else:
            risk_level = OperationRisk.LOW
        
        # Generate mitigation actions
        mitigation_actions = self._generate_mitigation_actions(risk_level, risk_factors)
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "mitigation_actions": mitigation_actions
        }
    
    def _has_suspicious_arguments(self, arguments: Dict[str, Any]) -> bool:
        """Check for suspicious argument patterns."""
        # Check for large data volumes
        if "csv_data" in arguments and len(arguments["csv_data"]) > 1024 * 1024:
            return True
        
        # Check for unusual node/edge counts
        if "nodes" in arguments and len(arguments["nodes"]) > 1000:
            return True
        
        if "edges" in arguments and len(arguments["edges"]) > 10000:
            return True
        
        return False
    
    def _generate_mitigation_actions(self, risk_level: OperationRisk, risk_factors: List[str]) -> List[str]:
        """Generate mitigation actions based on risk level."""
        actions = []
        
        if risk_level == OperationRisk.CRITICAL:
            actions.extend([
                "Require human approval",
                "Enhanced monitoring",
                "Alert security team",
                "Full audit logging"
            ])
        elif risk_level == OperationRisk.HIGH:
            actions.extend([
                "Require human approval",
                "Enhanced monitoring",
                "Full audit logging"
            ])
        elif risk_level == OperationRisk.MEDIUM:
            actions.extend([
                "Enhanced monitoring",
                "Additional validation"
            ])
        else:
            actions.append("Standard monitoring")
        
        return actions


class HumanInLoopManager:
    """Manages human-in-the-loop approval workflows."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pending_approvals: Dict[str, ApprovalRequest] = {}
        self.approval_callbacks: Dict[str, Callable] = {}
    
    async def create_approval_request(self, request: ApprovalRequest) -> str:
        """Create new approval request."""
        self.pending_approvals[request.request_id] = request
        
        # In production, this would notify administrators
        self.logger.info(f"Approval request created: {request.request_id} "
                        f"for operation {request.operation} by user {request.user.id}")
        
        # Set timeout callback
        asyncio.create_task(self._handle_approval_timeout(request.request_id))
        
        return request.request_id
    
    async def get_approval_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get approval request by ID."""
        return self.pending_approvals.get(request_id)
    
    async def approve_request(self, request_id: str, approver_id: str) -> bool:
        """Approve a pending request."""
        if request_id not in self.pending_approvals:
            return False
        
        request = self.pending_approvals[request_id]
        if request.status != ApprovalStatus.PENDING:
            return False
        
        request.status = ApprovalStatus.APPROVED
        request.approved_by = approver_id
        request.approved_at = datetime.utcnow()
        
        self.logger.info(f"Request {request_id} approved by {approver_id}")
        
        # Notify callback if registered
        if request_id in self.approval_callbacks:
            callback = self.approval_callbacks[request_id]
            await callback(request)
        
        return True
    
    async def deny_request(self, request_id: str, approver_id: str, reason: str) -> bool:
        """Deny a pending request."""
        if request_id not in self.pending_approvals:
            return False
        
        request = self.pending_approvals[request_id]
        if request.status != ApprovalStatus.PENDING:
            return False
        
        request.status = ApprovalStatus.DENIED
        request.approved_by = approver_id
        request.approved_at = datetime.utcnow()
        request.denial_reason = reason
        
        self.logger.info(f"Request {request_id} denied by {approver_id}: {reason}")
        
        return True
    
    async def _handle_approval_timeout(self, request_id: str):
        """Handle approval timeout."""
        request = self.pending_approvals.get(request_id)
        if not request or request.status != ApprovalStatus.PENDING:
            return
        
        # Wait for timeout
        if request.expires_at:
            timeout_seconds = (request.expires_at - datetime.utcnow()).total_seconds()
            if timeout_seconds > 0:
                await asyncio.sleep(timeout_seconds)
        
        # Check if still pending
        if request.status == ApprovalStatus.PENDING:
            request.status = ApprovalStatus.TIMEOUT
            self.logger.warning(f"Approval request {request_id} timed out")
    
    def get_pending_approvals(self) -> List[ApprovalRequest]:
        """Get all pending approval requests."""
        return [
            request for request in self.pending_approvals.values()
            if request.status == ApprovalStatus.PENDING
        ]
    
    def register_approval_callback(self, request_id: str, callback: Callable):
        """Register callback for approval completion."""
        self.approval_callbacks[request_id] = callback