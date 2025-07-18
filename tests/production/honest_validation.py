#!/usr/bin/env python3
"""Honest production validation for NetworkX MCP Server.

This validation suite tests what's actually implemented and clearly reports
what's missing before production deployment. It validates the real capabilities
rather than aspirational ones.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation test."""

    test_name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    critical: bool = False  # If this fails, deployment should be blocked


class HonestProductionValidator:
    """Production validator that tests real implementation, not aspirations."""

    def __init__(self):
        self.results: List[ValidationResult] = []
        self.critical_failures = 0
        self.warnings = 0

    def add_result(self, result: ValidationResult):
        """Add validation result and track failures."""
        self.results.append(result)

        if not result.passed:
            if result.critical:
                self.critical_failures += 1
                logger.error(f"❌ CRITICAL: {result.test_name} - {result.message}")
            else:
                self.warnings += 1
                logger.warning(f"⚠️  WARNING: {result.test_name} - {result.message}")
        else:
            logger.info(f"✅ PASS: {result.test_name} - {result.message}")

    def test_core_implementation_exists(self) -> ValidationResult:
        """Test if core implementation files actually exist."""

        required_files = [
            "src/networkx_mcp/__init__.py",
            "src/networkx_mcp/__main__.py",
            "src/networkx_mcp/core/graph_operations.py",
            "src/networkx_mcp/server.py",
        ]

        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)

        if missing_files:
            return ValidationResult(
                test_name="Core Implementation Files",
                passed=False,
                critical=True,
                message=f"Missing critical files: {missing_files}",
                details={"missing_files": missing_files},
            )

        return ValidationResult(
            test_name="Core Implementation Files",
            passed=True,
            message="All core implementation files present",
        )

    def test_mcp_protocol_implementation(self) -> ValidationResult:
        """Test if MCP protocol is actually implemented."""

        # Check for critical MCP files
        mcp_files = [
            "src/networkx_mcp/server_minimal.py",  # Referenced in __main__.py
            "src/networkx_mcp/mcp/jsonrpc_handler.py",
            "src/networkx_mcp/transport/stdio_transport.py",
        ]

        missing_critical = []
        for file_path in mcp_files:
            if not Path(file_path).exists():
                missing_critical.append(file_path)

        if missing_critical:
            return ValidationResult(
                test_name="MCP Protocol Implementation",
                passed=False,
                critical=True,
                message=f"MCP protocol not implemented - missing: {missing_critical}",
                details={
                    "missing_files": missing_critical,
                    "impact": "Cannot be used by Claude Desktop or other MCP clients",
                    "estimated_fix": "2-3 weeks of implementation work",
                },
            )

        # Test if modules can be imported
        try:
            from networkx_mcp.mcp.jsonrpc_handler import MCPJsonRpcHandler
            from networkx_mcp.transport.stdio_transport import StdioTransport

            return ValidationResult(
                test_name="MCP Protocol Implementation",
                passed=True,
                message="MCP protocol components can be imported",
            )

        except ImportError as e:
            return ValidationResult(
                test_name="MCP Protocol Implementation",
                passed=False,
                critical=True,
                message=f"MCP components cannot be imported: {e}",
                details={"import_error": str(e)},
            )

    def test_server_startup(self) -> ValidationResult:
        """Test if the server can actually start."""

        try:
            # Try to import the main server module
            from networkx_mcp.__main__ import main

            # Check if server_minimal.py exists (referenced in __main__.py)
            if not Path("src/networkx_mcp/server_minimal.py").exists():
                return ValidationResult(
                    test_name="Server Startup",
                    passed=False,
                    critical=True,
                    message="server_minimal.py missing - server cannot start",
                    details={"referenced_in": "__main__.py", "import_will_fail": True},
                )

            return ValidationResult(
                test_name="Server Startup",
                passed=True,
                message="Server module can be imported",
            )

        except ImportError as e:
            return ValidationResult(
                test_name="Server Startup",
                passed=False,
                critical=True,
                message=f"Server cannot be imported: {e}",
                details={"import_error": str(e)},
            )

    def test_graph_operations_functional(self) -> ValidationResult:
        """Test if core graph operations actually work."""

        try:
            from networkx_mcp.core.graph_operations import GraphManager

            # Test basic operations
            manager = GraphManager()

            # Create a test graph
            result = manager.create_graph("test_validation", directed=True)
            if not result.get("success"):
                raise Exception(f"Graph creation failed: {result}")

            # Add nodes
            result = manager.add_nodes("test_validation", ["A", "B", "C"])
            if not result.get("success"):
                raise Exception(f"Add nodes failed: {result}")

            # Add edges
            result = manager.add_edges("test_validation", [("A", "B"), ("B", "C")])
            if not result.get("success"):
                raise Exception(f"Add edges failed: {result}")

            # Get graph info
            info = manager.get_graph_info("test_validation")
            if info.get("nodes", 0) != 3 or info.get("edges", 0) != 2:
                raise Exception(f"Graph info incorrect: {info}")

            # Clean up
            manager.delete_graph("test_validation")

            return ValidationResult(
                test_name="Graph Operations Functional",
                passed=True,
                message="Core graph operations working correctly",
                details={
                    "tested": ["create", "add_nodes", "add_edges", "info", "delete"]
                },
            )

        except Exception as e:
            return ValidationResult(
                test_name="Graph Operations Functional",
                passed=False,
                critical=True,
                message=f"Graph operations not working: {e}",
                details={"error": str(e)},
            )

    def test_algorithms_functional(self) -> ValidationResult:
        """Test if graph algorithms actually work."""

        try:
            from networkx_mcp.advanced.algorithms import GraphAlgorithms
            from networkx_mcp.core.graph_operations import GraphManager

            # Create test graph with algorithms
            manager = GraphManager()
            algorithms = GraphAlgorithms(manager)

            # Create a small test graph
            manager.create_graph("algo_test", directed=False)
            manager.add_nodes("algo_test", ["A", "B", "C", "D"])
            manager.add_edges(
                "algo_test", [("A", "B"), ("B", "C"), ("C", "D"), ("A", "D")]
            )

            # Test shortest path
            result = algorithms.shortest_path("algo_test", "A", "C")
            if not result.get("success"):
                raise Exception(f"Shortest path failed: {result}")

            # Test centrality
            result = algorithms.centrality_measures("algo_test", ["degree"])
            if not result.get("success"):
                raise Exception(f"Centrality failed: {result}")

            # Clean up
            manager.delete_graph("algo_test")

            return ValidationResult(
                test_name="Algorithms Functional",
                passed=True,
                message="Graph algorithms working correctly",
                details={"tested": ["shortest_path", "centrality_measures"]},
            )

        except Exception as e:
            return ValidationResult(
                test_name="Algorithms Functional",
                passed=False,
                critical=False,  # Algorithms not critical for basic operation
                message=f"Graph algorithms not working: {e}",
                details={"error": str(e)},
            )

    def test_real_mcp_client_compatibility(self) -> ValidationResult:
        """Test if server can work with real MCP clients."""

        # This is the most important test - can actual MCP clients use this server?

        # Check if we can run the server in stdio mode
        try:
            # Try to start server process
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write("""
import sys
import subprocess
import time

# Try to start the MCP server
try:
    proc = subprocess.Popen([
        sys.executable, '-m', 'networkx_mcp'
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, timeout=10)

    # Send MCP initialization
    init_msg = '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'

    stdout, stderr = proc.communicate(input=init_msg + '\\n', timeout=5)

    print(f"STDOUT: {stdout}")
    print(f"STDERR: {stderr}")
    print(f"RETURNCODE: {proc.returncode}")

except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
""")
                test_script = f.name

            # Run the test
            result = subprocess.run(
                [sys.executable, test_script],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Clean up
            os.unlink(test_script)

            if "ERROR:" in result.stdout:
                return ValidationResult(
                    test_name="MCP Client Compatibility",
                    passed=False,
                    critical=True,
                    message="Server cannot handle MCP protocol requests",
                    details={
                        "test_output": result.stdout,
                        "impact": "Cannot be used by Claude Desktop or other MCP clients",
                    },
                )

            if "RETURNCODE: 0" in result.stdout:
                return ValidationResult(
                    test_name="MCP Client Compatibility",
                    passed=True,
                    message="Server responds to MCP protocol (basic test)",
                )

            return ValidationResult(
                test_name="MCP Client Compatibility",
                passed=False,
                critical=True,
                message="Server does not respond correctly to MCP protocol",
                details={"test_output": result.stdout},
            )

        except Exception as e:
            return ValidationResult(
                test_name="MCP Client Compatibility",
                passed=False,
                critical=True,
                message=f"Cannot test MCP compatibility: {e}",
                details={"error": str(e)},
            )

    def test_monitoring_infrastructure(self) -> ValidationResult:
        """Test if monitoring infrastructure works."""

        try:
            from networkx_mcp.metrics.prometheus import get_metrics

            metrics = get_metrics()

            # Test basic metrics collection
            metrics.record_request("test_method", "success", 0.1)

            return ValidationResult(
                test_name="Monitoring Infrastructure",
                passed=True,
                message="Monitoring and metrics collection working",
            )

        except Exception as e:
            return ValidationResult(
                test_name="Monitoring Infrastructure",
                passed=False,
                critical=False,  # Not critical for basic operation
                message=f"Monitoring not working: {e}",
                details={"error": str(e)},
            )

    def test_feature_flags_realistic(self) -> ValidationResult:
        """Test feature flags and get realistic status."""

        try:
            from networkx_mcp.features.realistic_feature_flags import (
                get_realistic_feature_flags,
            )

            flags = get_realistic_feature_flags()
            can_deploy, missing = flags.can_deploy_to_production()

            if can_deploy:
                return ValidationResult(
                    test_name="Feature Flags & Production Readiness",
                    passed=True,
                    message="All required features implemented for production",
                )

            return ValidationResult(
                test_name="Feature Flags & Production Readiness",
                passed=False,
                critical=True,
                message=f"Missing required features: {', '.join(missing)}",
                details={
                    "missing_features": missing,
                    "recommendation": flags.get_deployment_recommendation(),
                },
            )

        except Exception as e:
            return ValidationResult(
                test_name="Feature Flags & Production Readiness",
                passed=False,
                critical=False,
                message=f"Feature flags not working: {e}",
                details={"error": str(e)},
            )

    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests and return comprehensive report."""

        logger.info("🔍 Starting honest production validation...")

        # Core tests (critical for deployment)
        self.add_result(self.test_core_implementation_exists())
        self.add_result(self.test_mcp_protocol_implementation())
        self.add_result(self.test_server_startup())
        self.add_result(self.test_real_mcp_client_compatibility())

        # Functional tests (validate what works)
        self.add_result(self.test_graph_operations_functional())
        self.add_result(self.test_algorithms_functional())

        # Infrastructure tests (important but not critical)
        self.add_result(self.test_monitoring_infrastructure())
        self.add_result(self.test_feature_flags_realistic())

        # Generate report
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)

        critical_tests = [r for r in self.results if r.critical]
        critical_passed = sum(1 for r in critical_tests if r.passed)

        production_ready = self.critical_failures == 0

        report = {
            "validation_timestamp": time.time(),
            "production_ready": production_ready,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "critical_failures": self.critical_failures,
            "warnings": self.warnings,
            "critical_tests_passed": critical_passed,
            "critical_tests_total": len(critical_tests),
            "deployment_recommendation": self._get_deployment_recommendation(),
            "test_results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "critical": r.critical,
                    "message": r.message,
                    "details": r.details,
                }
                for r in self.results
            ],
        }

        return report

    def _get_deployment_recommendation(self) -> str:
        """Get honest deployment recommendation."""

        if self.critical_failures == 0:
            if self.warnings == 0:
                return "✅ DEPLOY: All tests passed - ready for production"
            else:
                return f"✅ DEPLOY WITH CAUTION: {self.warnings} warnings but no critical failures"

        critical_tests = [r for r in self.results if r.critical and not r.passed]

        if any("MCP protocol" in r.test_name for r in critical_tests):
            return "❌ DO NOT DEPLOY: MCP protocol not implemented - cannot be used by MCP clients"

        if any("server_minimal.py" in r.message for r in critical_tests):
            return "❌ DO NOT DEPLOY: Server cannot start - missing core implementation"

        return f"❌ DO NOT DEPLOY: {self.critical_failures} critical failures must be fixed first"

    def print_summary(self, report: Dict[str, Any]):
        """Print human-readable validation summary."""

        print("\n" + "=" * 60)
        print("🔍 NETWORKX MCP SERVER - HONEST PRODUCTION VALIDATION")
        print("=" * 60)

        # Overall status
        status_emoji = "✅" if report["production_ready"] else "❌"
        print(
            f"\n{status_emoji} PRODUCTION READINESS: {report['deployment_recommendation']}"
        )

        # Test summary
        print("\n📊 TEST SUMMARY:")
        print(f"   Total Tests: {report['total_tests']}")
        print(f"   Passed: {report['passed_tests']}")
        print(f"   Failed: {report['failed_tests']}")
        print(f"   Critical Failures: {report['critical_failures']}")
        print(f"   Warnings: {report['warnings']}")

        # Critical tests detail
        if report["critical_tests_total"] > 0:
            print(
                f"\n🚨 CRITICAL TESTS: {report['critical_tests_passed']}/{report['critical_tests_total']} passed"
            )

        # Failed tests
        failed_tests = [r for r in report["test_results"] if not r["passed"]]
        if failed_tests:
            print("\n❌ FAILED TESTS:")
            for test in failed_tests:
                emoji = "🚨" if test["critical"] else "⚠️"
                print(f"   {emoji} {test['test_name']}: {test['message']}")

                if test.get("details"):
                    for key, value in test["details"].items():
                        if key in ["missing_files", "missing_features"]:
                            print(f"      {key}: {value}")

        # What actually works
        passed_tests = [r for r in report["test_results"] if r["passed"]]
        if passed_tests:
            print("\n✅ WORKING FEATURES:")
            for test in passed_tests:
                print(f"   ✅ {test['test_name']}: {test['message']}")

        print("\n" + "=" * 60)


async def main():
    """Main validation entry point."""

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    validator = HonestProductionValidator()
    report = await validator.run_comprehensive_validation()

    # Print detailed summary
    validator.print_summary(report)

    # Save report
    report_file = f"validation_report_{int(time.time())}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Detailed report saved to: {report_file}")

    # Exit with appropriate code
    if report["production_ready"]:
        print("\n🎉 Validation passed - system ready for production!")
        sys.exit(0)
    else:
        print(
            f"\n💥 Validation failed - {report['critical_failures']} critical issues must be fixed"
        )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
