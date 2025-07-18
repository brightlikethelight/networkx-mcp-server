"""
NetworkX MCP Security Fortress

A comprehensive security framework that transforms the NetworkX MCP Server into
the reference implementation for secure graph analysis servers, addressing all
critical MCP protocol vulnerabilities.

Features:
- AI-Powered Prompt Injection Detection
- Zero-Trust Input/Output Validation
- Comprehensive Sandboxing Framework
- Real-Time Threat Detection & Monitoring
- Human-in-the-Loop Security Controls
- Tool Integrity Verification System
"""

from typing import Optional

# Core Security Components
try:
    from .fortress_server import SecureMCPHandler, SecurityFortressServer
    from .monitoring import (
        ComplianceReporter,
        ComprehensiveAuditLogger,
        SecurityMonitor,
        ThreatDetector,
    )
    from .sandboxing import (
        ContainerManager,
        ExecutionResult,
        ResourceMonitor,
        SecureSandbox,
    )
    from .security_broker import (
        AuthorizationResult,
        HumanInLoopManager,
        OperationRiskAssessor,
        SecurityBroker,
    )
    from .threat_detection import (
        BehavioralAnalyzer,
        PromptInjectionDetector,
        ThreatAssessment,
        ThreatIntelligence,
    )
    from .validation import (
        ContentSanitizer,
        DataLossPreventionEngine,
        ValidationResult,
        ZeroTrustValidator,
    )

    SECURITY_FORTRESS_AVAILABLE = True

except ImportError as e:
    SECURITY_FORTRESS_AVAILABLE = False
    _import_error = e

    # Provide stub classes for graceful degradation
    class PromptInjectionDetector:
        def __init__(self):
            raise ImportError(f"Security Fortress not available: {_import_error}")

    class ZeroTrustValidator:
        def __init__(self):
            raise ImportError(f"Security Fortress not available: {_import_error}")

    class SecurityBroker:
        def __init__(self):
            raise ImportError(f"Security Fortress not available: {_import_error}")

    class SecureSandbox:
        def __init__(self):
            raise ImportError(f"Security Fortress not available: {_import_error}")

    class SecurityMonitor:
        def __init__(self):
            raise ImportError(f"Security Fortress not available: {_import_error}")

    class SecurityFortressServer:
        def __init__(self):
            raise ImportError(f"Security Fortress not available: {_import_error}")


def check_security_fortress_availability() -> bool:
    """Check if Security Fortress dependencies are available."""
    return SECURITY_FORTRESS_AVAILABLE


def get_security_fortress_info() -> dict:
    """Get information about Security Fortress availability and features."""
    return {
        "available": SECURITY_FORTRESS_AVAILABLE,
        "version": "2.0.0",
        "features": {
            "prompt_injection_detection": SECURITY_FORTRESS_AVAILABLE,
            "zero_trust_validation": SECURITY_FORTRESS_AVAILABLE,
            "security_broker": SECURITY_FORTRESS_AVAILABLE,
            "secure_sandboxing": SECURITY_FORTRESS_AVAILABLE,
            "threat_monitoring": SECURITY_FORTRESS_AVAILABLE,
            "human_in_loop": SECURITY_FORTRESS_AVAILABLE,
            "tool_integrity": SECURITY_FORTRESS_AVAILABLE,
            "compliance_reporting": SECURITY_FORTRESS_AVAILABLE,
        },
        "error": None if SECURITY_FORTRESS_AVAILABLE else str(_import_error),
    }


# Export main classes
__all__ = [
    "PromptInjectionDetector",
    "ZeroTrustValidator",
    "SecurityBroker",
    "SecureSandbox",
    "SecurityMonitor",
    "SecurityFortressServer",
    "ThreatAssessment",
    "ValidationResult",
    "AuthorizationResult",
    "ExecutionResult",
    "check_security_fortress_availability",
    "get_security_fortress_info",
]
