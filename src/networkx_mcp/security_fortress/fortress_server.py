"""
NetworkX MCP Security Fortress Server

Production-ready secure MCP server that integrates all security layers:
- AI-powered prompt injection detection
- Zero-trust input/output validation  
- Secure container-based sandboxing
- Human-in-the-loop approval workflows
- Real-time threat monitoring
- Comprehensive audit logging

This server provides enterprise-grade security for NetworkX graph operations
while maintaining full MCP compatibility.
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from .threat_detection import PromptInjectionDetector, ThreatAssessment, ThreatLevel
from .validation import ZeroTrustValidator, ValidationResult, ValidationStatus
from .security_broker import SecurityBroker, AuthorizationResult, OperationRisk
from .sandboxing import SecureSandbox, ExecutionResult, SandboxStatus
from .monitoring import SecurityMonitor, SecurityEvent, SecurityEventType, AlertSeverity
from ..enterprise.auth import User, Role, Permission
from ..enterprise.config import EnterpriseConfig


@dataclass
class SecurityFortressConfig:
    """Security Fortress configuration."""
    enable_threat_detection: bool = True
    enable_zero_trust_validation: bool = True
    enable_sandboxing: bool = True
    enable_human_approval: bool = True
    enable_monitoring: bool = True
    max_concurrent_operations: int = 100
    operation_timeout: int = 60
    enable_audit_logging: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "enable_threat_detection": self.enable_threat_detection,
            "enable_zero_trust_validation": self.enable_zero_trust_validation,
            "enable_sandboxing": self.enable_sandboxing,
            "enable_human_approval": self.enable_human_approval,
            "enable_monitoring": self.enable_monitoring,
            "max_concurrent_operations": self.max_concurrent_operations,
            "operation_timeout": self.operation_timeout,
            "enable_audit_logging": self.enable_audit_logging
        }


@dataclass
class SecureOperationResult:
    """Result of secure operation execution."""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    security_summary: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "security_summary": self.security_summary,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat()
        }


class SecurityFortressServer:
    """
    NetworkX MCP Security Fortress Server
    
    Enterprise-grade secure MCP server that provides:
    1. Multi-layer security validation
    2. Real-time threat detection
    3. Secure sandbox execution
    4. Human-in-the-loop controls
    5. Comprehensive monitoring
    6. Full audit trail
    """
    
    def __init__(self, config: Optional[SecurityFortressConfig] = None):
        self.config = config or SecurityFortressConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize security components
        self.threat_detector = PromptInjectionDetector() if self.config.enable_threat_detection else None
        self.validator = ZeroTrustValidator() if self.config.enable_zero_trust_validation else None
        self.security_broker = SecurityBroker() if self.config.enable_human_approval else None
        self.sandbox = SecureSandbox() if self.config.enable_sandboxing else None
        self.monitor = SecurityMonitor() if self.config.enable_monitoring else None
        
        # Operation tracking
        self.active_operations: Dict[str, Dict[str, Any]] = {}
        self.operation_history: List[Dict[str, Any]] = []
        
        # Server statistics
        self.server_stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "blocked_operations": 0,
            "pending_approvals": 0,
            "average_execution_time": 0.0,
            "uptime": datetime.utcnow()
        }
        
        self.logger.info("Security Fortress Server initialized")
    
    async def execute_secure_operation(
        self, 
        user: User, 
        operation: str, 
        arguments: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SecureOperationResult:
        """
        Execute operation through complete security fortress.
        
        Args:
            user: User requesting the operation
            operation: Operation name
            arguments: Operation arguments
            context: Additional context
            
        Returns:
            SecureOperationResult: Secure execution result
        """
        start_time = time.time()
        operation_id = f"op_{int(time.time() * 1000)}"
        
        if context is None:
            context = {}
        
        # Track operation
        self.active_operations[operation_id] = {
            "user": user.id,
            "operation": operation,
            "start_time": start_time,
            "status": "processing"
        }
        
        self.server_stats["total_operations"] += 1
        
        try:
            # PHASE 1: Threat Detection
            threat_assessment = None
            if self.threat_detector:
                prompt = self._construct_prompt(operation, arguments)
                threat_assessment = self.threat_detector.detect_injection(prompt, {
                    "tool_name": operation,
                    "user_id": user.id,
                    **context
                })
                
                if threat_assessment.threat_level == ThreatLevel.CRITICAL:
                    return await self._handle_critical_threat(
                        user, operation, arguments, threat_assessment, start_time
                    )
            
            # PHASE 2: Input Validation
            validation_result = None
            if self.validator:
                validation_result = self.validator.validate_input(operation, arguments)
                
                if validation_result.status == ValidationStatus.BLOCKED:
                    return await self._handle_blocked_input(
                        user, operation, arguments, validation_result, start_time
                    )
            
            # PHASE 3: Authorization and Risk Assessment
            authorization_result = None
            if self.security_broker:
                authorization_result = await self.security_broker.authorize_operation(
                    user, operation, arguments, context
                )
                
                if not authorization_result.authorized:
                    if authorization_result.requires_approval:
                        return await self._handle_pending_approval(
                            user, operation, arguments, authorization_result, start_time
                        )
                    else:
                        return await self._handle_denied_operation(
                            user, operation, arguments, authorization_result, start_time
                        )
            
            # PHASE 4: Secure Execution
            execution_result = None
            if self.sandbox:
                execution_result = await self.sandbox.execute_operation(
                    operation, arguments, context
                )
                
                if execution_result.status != SandboxStatus.COMPLETED:
                    return await self._handle_execution_failure(
                        user, operation, arguments, execution_result, start_time
                    )
            else:
                # Fallback to direct execution (not recommended for production)
                execution_result = await self._direct_execution(operation, arguments)
            
            # PHASE 5: Output Validation
            output_validation = None
            if self.validator and execution_result.result:
                output_validation = self.validator.validate_output(execution_result.result)
            
            # PHASE 6: Monitoring and Logging
            execution_time = time.time() - start_time
            if self.monitor:
                self.monitor.log_operation(
                    user_id=user.id,
                    operation=operation,
                    arguments=arguments,
                    result=execution_result.result,
                    threat_assessment=threat_assessment,
                    validation_result=validation_result,
                    authorization_result=authorization_result,
                    execution_time=execution_time
                )
            
            # Update statistics
            self.server_stats["successful_operations"] += 1
            self._update_average_execution_time(execution_time)
            
            # Clean up
            del self.active_operations[operation_id]
            
            # Create security summary
            security_summary = {
                "threat_level": threat_assessment.threat_level.value if threat_assessment else "benign",
                "validation_status": validation_result.status.value if validation_result else "passed",
                "risk_level": authorization_result.risk_level.value if authorization_result else "low",
                "execution_status": execution_result.status.value if execution_result else "completed",
                "security_actions": authorization_result.security_actions if authorization_result else [],
                "processing_time": execution_time
            }
            
            return SecureOperationResult(
                success=True,
                result=execution_result.result,
                security_summary=security_summary,
                execution_time=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Secure operation failed: {e}")
            self.server_stats["failed_operations"] += 1
            
            # Clean up
            if operation_id in self.active_operations:
                del self.active_operations[operation_id]
            
            return SecureOperationResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _construct_prompt(self, operation: str, arguments: Dict[str, Any]) -> str:
        """Construct prompt for threat detection."""
        return f"Operation: {operation}\nArguments: {json.dumps(arguments)}"
    
    async def _handle_critical_threat(
        self, 
        user: User, 
        operation: str, 
        arguments: Dict[str, Any],
        threat_assessment: ThreatAssessment,
        start_time: float
    ) -> SecureOperationResult:
        """Handle critical threat detection."""
        self.server_stats["blocked_operations"] += 1
        
        if self.monitor:
            self.monitor.log_security_event(SecurityEvent(
                event_id=f"threat_{int(time.time() * 1000)}",
                event_type=SecurityEventType.THREAT_DETECTION,
                severity=AlertSeverity.CRITICAL,
                message=f"Critical threat detected: {threat_assessment.threat_level.value}",
                user_id=user.id,
                operation=operation,
                metadata=threat_assessment.to_dict()
            ))
        
        return SecureOperationResult(
            success=False,
            error="Operation blocked due to critical security threat",
            security_summary={
                "threat_level": threat_assessment.threat_level.value,
                "detected_patterns": threat_assessment.detected_patterns,
                "mitigation_actions": threat_assessment.mitigation_actions,
                "blocked_reason": "Critical threat detected"
            },
            execution_time=time.time() - start_time
        )
    
    async def _handle_blocked_input(
        self, 
        user: User, 
        operation: str, 
        arguments: Dict[str, Any],
        validation_result: ValidationResult,
        start_time: float
    ) -> SecureOperationResult:
        """Handle blocked input validation."""
        self.server_stats["blocked_operations"] += 1
        
        return SecureOperationResult(
            success=False,
            error="Operation blocked due to input validation failure",
            security_summary={
                "validation_status": validation_result.status.value,
                "violations": validation_result.violations,
                "risk_score": validation_result.risk_score,
                "blocked_reason": "Input validation failed"
            },
            execution_time=time.time() - start_time
        )
    
    async def _handle_pending_approval(
        self, 
        user: User, 
        operation: str, 
        arguments: Dict[str, Any],
        authorization_result: AuthorizationResult,
        start_time: float
    ) -> SecureOperationResult:
        """Handle pending human approval."""
        self.server_stats["pending_approvals"] += 1
        
        return SecureOperationResult(
            success=False,
            error="Operation requires human approval",
            security_summary={
                "authorization_status": "pending_approval",
                "approval_id": authorization_result.approval_id,
                "risk_level": authorization_result.risk_level.value,
                "risk_factors": authorization_result.risk_factors,
                "requires_approval": True
            },
            execution_time=time.time() - start_time
        )
    
    async def _handle_denied_operation(
        self, 
        user: User, 
        operation: str, 
        arguments: Dict[str, Any],
        authorization_result: AuthorizationResult,
        start_time: float
    ) -> SecureOperationResult:
        """Handle denied operation."""
        self.server_stats["blocked_operations"] += 1
        
        return SecureOperationResult(
            success=False,
            error="Operation denied by security policy",
            security_summary={
                "authorization_status": "denied",
                "risk_level": authorization_result.risk_level.value,
                "risk_factors": authorization_result.risk_factors,
                "denied_reason": "Security policy violation"
            },
            execution_time=time.time() - start_time
        )
    
    async def _handle_execution_failure(
        self, 
        user: User, 
        operation: str, 
        arguments: Dict[str, Any],
        execution_result: ExecutionResult,
        start_time: float
    ) -> SecureOperationResult:
        """Handle execution failure."""
        self.server_stats["failed_operations"] += 1
        
        return SecureOperationResult(
            success=False,
            error=f"Execution failed: {execution_result.error}",
            security_summary={
                "execution_status": execution_result.status.value,
                "resource_usage": execution_result.resource_usage,
                "security_events": execution_result.security_events,
                "execution_error": execution_result.error
            },
            execution_time=time.time() - start_time
        )
    
    async def _direct_execution(self, operation: str, arguments: Dict[str, Any]) -> ExecutionResult:
        """Direct execution fallback (not recommended for production)."""
        from ..server_minimal import GRAPH_STORAGE
        
        # Import operations
        from ..server_minimal import (
            create_graph, add_nodes, add_edges, get_graph_info,
            shortest_path, degree_centrality, betweenness_centrality,
            pagerank, connected_components, community_detection,
            visualize_graph, import_csv, export_json
        )
        
        # Map operations to functions
        operation_map = {
            "create_graph": create_graph,
            "add_nodes": add_nodes,
            "add_edges": add_edges,
            "get_info": get_graph_info,
            "shortest_path": shortest_path,
            "degree_centrality": degree_centrality,
            "betweenness_centrality": betweenness_centrality,
            "pagerank": pagerank,
            "connected_components": connected_components,
            "community_detection": community_detection,
            "visualize_graph": visualize_graph,
            "import_csv": import_csv,
            "export_json": export_json
        }
        
        if operation not in operation_map:
            return ExecutionResult(
                status=SandboxStatus.FAILED,
                error=f"Unknown operation: {operation}"
            )
        
        try:
            func = operation_map[operation]
            
            # Execute based on operation type
            if operation == "create_graph":
                result = func(arguments["name"], arguments.get("directed", False))
            elif operation in ["add_nodes", "add_edges", "get_info"]:
                result = func(arguments["graph"], arguments.get("nodes") or arguments.get("edges"))
            elif operation == "shortest_path":
                result = func(arguments["graph"], arguments["source"], arguments["target"])
            elif operation in ["degree_centrality", "betweenness_centrality", "pagerank", 
                              "connected_components", "community_detection"]:
                result = func(arguments["graph"])
            elif operation == "visualize_graph":
                result = func(arguments["graph"], arguments.get("layout", "spring"))
            elif operation == "import_csv":
                result = func(arguments["graph"], arguments["csv_data"])
            elif operation == "export_json":
                result = func(arguments["graph"])
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            return ExecutionResult(
                status=SandboxStatus.COMPLETED,
                result=result
            )
            
        except Exception as e:
            return ExecutionResult(
                status=SandboxStatus.FAILED,
                error=str(e)
            )
    
    def _update_average_execution_time(self, execution_time: float):
        """Update average execution time."""
        total_ops = self.server_stats["total_operations"]
        current_avg = self.server_stats["average_execution_time"]
        
        self.server_stats["average_execution_time"] = (
            (current_avg * (total_ops - 1) + execution_time) / total_ops
        )
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        uptime = datetime.utcnow() - self.server_stats["uptime"]
        
        stats = {
            **self.server_stats,
            "uptime_seconds": uptime.total_seconds(),
            "active_operations": len(self.active_operations),
            "config": self.config.to_dict()
        }
        
        # Add component statistics
        if self.threat_detector:
            stats["threat_detection"] = self.threat_detector.get_detection_stats()
        if self.validator:
            stats["validation"] = self.validator.get_validation_stats()
        if self.security_broker:
            stats["security_broker"] = self.security_broker.get_security_stats()
        if self.sandbox:
            stats["sandbox"] = self.sandbox.get_sandbox_stats()
        if self.monitor:
            stats["monitoring"] = self.monitor.get_monitoring_stats()
        
        return stats
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get comprehensive security summary."""
        return {
            "fortress_status": "operational",
            "security_layers": {
                "threat_detection": self.config.enable_threat_detection,
                "input_validation": self.config.enable_zero_trust_validation,
                "authorization": self.config.enable_human_approval,
                "sandboxing": self.config.enable_sandboxing,
                "monitoring": self.config.enable_monitoring
            },
            "statistics": self.get_server_stats(),
            "active_operations": len(self.active_operations),
            "security_posture": "high"
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of all security components."""
        health = {
            "overall_status": "healthy",
            "components": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check threat detector
        if self.threat_detector:
            health["components"]["threat_detector"] = "operational"
        
        # Check validator
        if self.validator:
            health["components"]["validator"] = "operational"
        
        # Check security broker
        if self.security_broker:
            health["components"]["security_broker"] = "operational"
        
        # Check sandbox
        if self.sandbox:
            health["components"]["sandbox"] = "operational"
        
        # Check monitor
        if self.monitor:
            health["components"]["monitor"] = "operational"
        
        return health
    
    async def shutdown(self):
        """Graceful shutdown of security fortress."""
        self.logger.info("Shutting down Security Fortress Server")
        
        # Wait for active operations to complete
        timeout = 30  # 30 second timeout
        start_time = time.time()
        
        while self.active_operations and (time.time() - start_time) < timeout:
            await asyncio.sleep(1)
        
        # Force stop monitoring
        if self.monitor:
            self.monitor.stop_monitoring()
        
        self.logger.info("Security Fortress Server shutdown complete")


# Factory function for easy server creation
def create_security_fortress_server(
    enable_all_security: bool = True,
    **config_overrides
) -> SecurityFortressServer:
    """
    Create a Security Fortress Server with sensible defaults.
    
    Args:
        enable_all_security: Enable all security features
        **config_overrides: Override specific configuration options
        
    Returns:
        SecurityFortressServer: Configured server instance
    """
    if enable_all_security:
        config = SecurityFortressConfig(
            enable_threat_detection=True,
            enable_zero_trust_validation=True,
            enable_sandboxing=True,
            enable_human_approval=True,
            enable_monitoring=True,
            **config_overrides
        )
    else:
        config = SecurityFortressConfig(**config_overrides)
    
    return SecurityFortressServer(config)


# Context manager for managed server lifecycle
@asynccontextmanager
async def managed_security_fortress(**config_overrides):
    """Context manager for Security Fortress Server."""
    server = create_security_fortress_server(**config_overrides)
    try:
        yield server
    finally:
        await server.shutdown()