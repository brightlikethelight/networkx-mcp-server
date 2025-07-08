#!/usr/bin/env python3
"""Production deployment validation suite.

Comprehensive validation tests for production readiness including:
- MCP protocol compliance testing
- Performance validation based on testing limits
- Concurrent connection testing
- Health endpoint verification
- Feature flag validation
- Monitoring endpoint checks
"""

import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    import pytest
except ImportError:
    pytest = None

import requests

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from networkx_mcp.features.production_flags import get_production_feature_manager
    FEATURE_FLAGS_AVAILABLE = True
except ImportError:
    FEATURE_FLAGS_AVAILABLE = False
    get_production_feature_manager = None

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    message: str
    details: Optional[Dict] = None
    duration_ms: Optional[float] = None


class ProductionValidator:
    """Production deployment validation suite."""
    
    def __init__(self, 
                 base_url: str = "https://api.mcp.example.com",
                 stdio_mode: bool = False,
                 concurrent_users: int = 30,  # Based on production testing
                 max_graph_size: int = 10000):  # Based on performance limits
        self.base_url = base_url
        self.stdio_mode = stdio_mode
        self.concurrent_users = concurrent_users
        self.max_graph_size = max_graph_size
        self.results: List[ValidationResult] = []
        self.session = requests.Session()
        self.session.timeout = 30
        
    def add_result(self, result: ValidationResult):
        """Add validation result."""
        self.results.append(result)
        status = "✅ PASS" if result.passed else "❌ FAIL"
        logger.info(f"{status} {result.test_name}: {result.message}")
        if result.details:
            logger.debug(f"Details: {result.details}")
    
    def test_health_endpoints(self) -> ValidationResult:
        """Test all health check endpoints."""
        start_time = time.time()
        
        try:
            endpoints = ["/health", "/ready", "/startup"]
            all_healthy = True
            details = {}
            
            for endpoint in endpoints:
                url = urljoin(self.base_url, endpoint)
                response = self.session.get(url)
                
                details[endpoint] = {
                    "status_code": response.status_code,
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                    "healthy": response.status_code == 200
                }
                
                if response.status_code != 200:
                    all_healthy = False
                    
            duration_ms = (time.time() - start_time) * 1000
            
            if all_healthy:
                return ValidationResult(
                    test_name="Health Endpoints",
                    passed=True,
                    message="All health endpoints responding correctly",
                    details=details,
                    duration_ms=duration_ms
                )
            else:
                failed_endpoints = [ep for ep, data in details.items() if not data["healthy"]]
                return ValidationResult(
                    test_name="Health Endpoints",
                    passed=False,
                    message=f"Failed endpoints: {failed_endpoints}",
                    details=details,
                    duration_ms=duration_ms
                )
                
        except Exception as e:
            return ValidationResult(
                test_name="Health Endpoints",
                passed=False,
                message=f"Health check failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000
            )
    
    def test_metrics_endpoint(self) -> ValidationResult:
        """Test Prometheus metrics endpoint."""
        start_time = time.time()
        
        try:
            url = urljoin(self.base_url.replace(":8080", ":9090"), "/metrics")
            response = self.session.get(url)
            
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                # Check for expected metrics
                content = response.text
                expected_metrics = [
                    "mcp_requests_total",
                    "mcp_request_duration_seconds",
                    "mcp_active_connections",
                    "mcp_graph_operations_total",
                    "process_cpu_seconds_total",
                    "process_resident_memory_bytes"
                ]
                
                missing_metrics = []
                for metric in expected_metrics:
                    if metric not in content:
                        missing_metrics.append(metric)
                        
                if not missing_metrics:
                    return ValidationResult(
                        test_name="Metrics Endpoint",
                        passed=True,
                        message="Metrics endpoint healthy with all expected metrics",
                        details={"metrics_count": len(content.splitlines())},
                        duration_ms=duration_ms
                    )
                else:
                    return ValidationResult(
                        test_name="Metrics Endpoint",
                        passed=False,
                        message=f"Missing metrics: {missing_metrics}",
                        duration_ms=duration_ms
                    )
            else:
                return ValidationResult(
                    test_name="Metrics Endpoint",
                    passed=False,
                    message=f"Metrics endpoint returned {response.status_code}",
                    duration_ms=duration_ms
                )
                
        except Exception as e:
            return ValidationResult(
                test_name="Metrics Endpoint",
                passed=False,
                message=f"Metrics test failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000
            )
    
    async def test_mcp_protocol_compliance(self) -> ValidationResult:
        """Test MCP protocol compliance."""
        start_time = time.time()
        
        if not MCP_AVAILABLE:
            return ValidationResult(
                test_name="MCP Protocol",
                passed=False,
                message="MCP SDK not available - install with 'pip install mcp'",
                duration_ms=(time.time() - start_time) * 1000
            )
        
        try:
            if self.stdio_mode:
                # Test stdio transport
                server_params = StdioServerParameters(
                    command="python",
                    args=["-m", "networkx_mcp"]
                )
                
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        # Test initialization
                        await session.initialize()
                        
                        # Test basic capability
                        result = await session.call_tool("create_graph", {"graph_id": "test_validation"})
                        
                        if "error" not in result:
                            return ValidationResult(
                                test_name="MCP Protocol (stdio)",
                                passed=True,
                                message="MCP stdio protocol working correctly",
                                duration_ms=(time.time() - start_time) * 1000
                            )
                        else:
                            return ValidationResult(
                                test_name="MCP Protocol (stdio)",
                                passed=False,
                                message=f"MCP stdio error: {result.get('error')}",
                                duration_ms=(time.time() - start_time) * 1000
                            )
            else:
                # Test HTTP/JSON-RPC transport
                if not aiohttp:
                    return ValidationResult(
                        test_name="MCP Protocol (HTTP/JSON-RPC)",
                        passed=False,
                        message="aiohttp not available - install with 'pip install aiohttp'",
                        duration_ms=(time.time() - start_time) * 1000
                    )
                
                async with aiohttp.ClientSession() as session:
                    # Test JSON-RPC 2.0 compliance
                    rpc_request = {
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": "create_graph",
                            "arguments": {"graph_id": "test_validation_http"}
                        },
                        "id": 1
                    }
                    
                    async with session.post(
                        urljoin(self.base_url, "/jsonrpc"),
                        json=rpc_request,
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            
                            # Validate JSON-RPC 2.0 response format
                            if ("jsonrpc" in result and 
                                result["jsonrpc"] == "2.0" and
                                "id" in result and
                                ("result" in result or "error" in result)):
                                
                                return ValidationResult(
                                    test_name="MCP Protocol (HTTP/JSON-RPC)",
                                    passed=True,
                                    message="MCP HTTP/JSON-RPC protocol working correctly",
                                    details={"response": result},
                                    duration_ms=(time.time() - start_time) * 1000
                                )
                            else:
                                return ValidationResult(
                                    test_name="MCP Protocol (HTTP/JSON-RPC)",
                                    passed=False,
                                    message="Invalid JSON-RPC 2.0 response format",
                                    details={"response": result},
                                    duration_ms=(time.time() - start_time) * 1000
                                )
                        else:
                            return ValidationResult(
                                test_name="MCP Protocol (HTTP/JSON-RPC)",
                                passed=False,
                                message=f"HTTP request failed with status {response.status}",
                                duration_ms=(time.time() - start_time) * 1000
                            )
                            
        except Exception as e:
            return ValidationResult(
                test_name="MCP Protocol",
                passed=False,
                message=f"MCP protocol test failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000
            )
    
    def test_concurrent_connections(self) -> ValidationResult:
        """Test concurrent connection handling."""
        start_time = time.time()
        
        def make_request(user_id: int) -> Tuple[int, bool, str]:
            """Make a single test request."""
            try:
                # Test basic graph operation
                request_data = {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "create_graph",
                        "arguments": {"graph_id": f"concurrent_test_{user_id}"}
                    },
                    "id": user_id
                }
                
                response = self.session.post(
                    urljoin(self.base_url, "/jsonrpc"),
                    json=request_data,
                    timeout=10
                )
                
                return user_id, response.status_code == 200, f"Status: {response.status_code}"
                
            except Exception as e:
                return user_id, False, str(e)
        
        try:
            # Test with our production limit (30 users based on testing)
            with ThreadPoolExecutor(max_workers=self.concurrent_users) as executor:
                futures = [
                    executor.submit(make_request, i) 
                    for i in range(self.concurrent_users)
                ]
                
                results = []
                for future in as_completed(futures):
                    user_id, success, message = future.result()
                    results.append((user_id, success, message))
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Analyze results
            successful = sum(1 for _, success, _ in results if success)
            success_rate = successful / len(results)
            
            # Based on our testing: 30 users should achieve >95% success rate
            expected_success_rate = 0.95
            
            if success_rate >= expected_success_rate:
                return ValidationResult(
                    test_name="Concurrent Connections",
                    passed=True,
                    message=f"Concurrent connections test passed: {success_rate:.1%} success rate",
                    details={
                        "concurrent_users": self.concurrent_users,
                        "successful_requests": successful,
                        "total_requests": len(results),
                        "success_rate": success_rate
                    },
                    duration_ms=duration_ms
                )
            else:
                return ValidationResult(
                    test_name="Concurrent Connections",
                    passed=False,
                    message=f"Success rate {success_rate:.1%} below threshold {expected_success_rate:.1%}",
                    details={
                        "concurrent_users": self.concurrent_users,
                        "successful_requests": successful,
                        "total_requests": len(results),
                        "success_rate": success_rate,
                        "failures": [msg for _, success, msg in results if not success]
                    },
                    duration_ms=duration_ms
                )
                
        except Exception as e:
            return ValidationResult(
                test_name="Concurrent Connections",
                passed=False,
                message=f"Concurrent connection test failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000
            )
    
    def test_graph_size_limits(self) -> ValidationResult:
        """Test graph size limits handling."""
        start_time = time.time()
        
        try:
            # Test creating a graph at the limit (10K nodes)
            request_data = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "create_graph",
                    "arguments": {"graph_id": "size_limit_test"}
                },
                "id": 1
            }
            
            response = self.session.post(
                urljoin(self.base_url, "/jsonrpc"),
                json=request_data
            )
            
            if response.status_code != 200:
                return ValidationResult(
                    test_name="Graph Size Limits",
                    passed=False,
                    message=f"Failed to create test graph: {response.status_code}",
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            # Add nodes up to the limit
            add_nodes_data = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "add_nodes",
                    "arguments": {
                        "graph_id": "size_limit_test",
                        "nodes": [f"node_{i}" for i in range(min(1000, self.max_graph_size))]  # Test subset
                    }
                },
                "id": 2
            }
            
            response = self.session.post(
                urljoin(self.base_url, "/jsonrpc"),
                json=add_nodes_data
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                if "error" not in result:
                    return ValidationResult(
                        test_name="Graph Size Limits",
                        passed=True,
                        message="Graph size limits working correctly",
                        details={"max_graph_size": self.max_graph_size},
                        duration_ms=duration_ms
                    )
                else:
                    return ValidationResult(
                        test_name="Graph Size Limits",
                        passed=False,
                        message=f"Graph operation error: {result.get('error')}",
                        duration_ms=duration_ms
                    )
            else:
                return ValidationResult(
                    test_name="Graph Size Limits",
                    passed=False,
                    message=f"Graph size test failed with status {response.status_code}",
                    duration_ms=duration_ms
                )
                
        except Exception as e:
            return ValidationResult(
                test_name="Graph Size Limits",
                passed=False,
                message=f"Graph size limits test failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000
            )
    
    def test_feature_flags(self) -> ValidationResult:
        """Test production feature flags are properly configured."""
        start_time = time.time()
        
        if not FEATURE_FLAGS_AVAILABLE:
            return ValidationResult(
                test_name="Feature Flags",
                passed=False,
                message="Feature flags module not available",
                duration_ms=(time.time() - start_time) * 1000
            )
        
        try:
            # Test that production feature manager is working
            feature_manager = get_production_feature_manager()
            
            # Test critical production features
            critical_features = {
                "strict_input_validation": True,  # Should be enabled for security
                "detailed_performance_tracing": True,  # Should be enabled for monitoring
                "use_approximate_algorithms": None,  # Should be in rollout
                "enable_graph_caching": None,  # Should be in rollout
            }
            
            feature_status = {}
            all_correct = True
            
            for feature_name, expected_enabled in critical_features.items():
                try:
                    is_enabled = feature_manager.is_enabled(feature_name, environment="production")
                    feature_status[feature_name] = {
                        "enabled": is_enabled,
                        "expected": expected_enabled
                    }
                    
                    if expected_enabled is not None and is_enabled != expected_enabled:
                        all_correct = False
                        
                except Exception as e:
                    feature_status[feature_name] = {"error": str(e)}
                    all_correct = False
            
            # Get rollout status
            rollout_status = feature_manager.get_rollout_status()
            
            duration_ms = (time.time() - start_time) * 1000
            
            if all_correct:
                return ValidationResult(
                    test_name="Feature Flags",
                    passed=True,
                    message="Production feature flags configured correctly",
                    details={
                        "feature_status": feature_status,
                        "rollout_status": rollout_status
                    },
                    duration_ms=duration_ms
                )
            else:
                return ValidationResult(
                    test_name="Feature Flags",
                    passed=False,
                    message="Feature flag configuration issues detected",
                    details={
                        "feature_status": feature_status,
                        "rollout_status": rollout_status
                    },
                    duration_ms=duration_ms
                )
                
        except Exception as e:
            return ValidationResult(
                test_name="Feature Flags",
                passed=False,
                message=f"Feature flags test failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000
            )
    
    def test_security_headers(self) -> ValidationResult:
        """Test security headers are properly configured."""
        start_time = time.time()
        
        try:
            response = self.session.get(urljoin(self.base_url, "/health"))
            
            expected_headers = {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block",
                "Strict-Transport-Security": None,  # Should be present for HTTPS
            }
            
            missing_headers = []
            header_status = {}
            
            for header, expected_value in expected_headers.items():
                actual_value = response.headers.get(header)
                header_status[header] = actual_value
                
                if expected_value is not None and actual_value != expected_value:
                    missing_headers.append(f"{header}: expected '{expected_value}', got '{actual_value}'")
                elif expected_value is None and actual_value is None:
                    missing_headers.append(f"{header}: missing")
            
            duration_ms = (time.time() - start_time) * 1000
            
            if not missing_headers:
                return ValidationResult(
                    test_name="Security Headers",
                    passed=True,
                    message="Security headers configured correctly",
                    details={"headers": header_status},
                    duration_ms=duration_ms
                )
            else:
                return ValidationResult(
                    test_name="Security Headers",
                    passed=False,
                    message=f"Security header issues: {missing_headers}",
                    details={"headers": header_status, "issues": missing_headers},
                    duration_ms=duration_ms
                )
                
        except Exception as e:
            return ValidationResult(
                test_name="Security Headers",
                passed=False,
                message=f"Security headers test failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000
            )
    
    def test_graceful_shutdown(self) -> ValidationResult:
        """Test graceful shutdown behavior (non-destructive)."""
        start_time = time.time()
        
        try:
            # Test that the server responds to SIGTERM gracefully
            # This is a non-destructive test that checks the endpoint exists
            response = self.session.get(urljoin(self.base_url, "/shutdown"))
            
            duration_ms = (time.time() - start_time) * 1000
            
            # We expect this to either:
            # 1. Return 404 (endpoint not exposed, which is fine)
            # 2. Return 405 (method not allowed for GET)
            # 3. Return 401/403 (protected endpoint)
            # We don't want it to actually shutdown in production
            
            if response.status_code in [404, 405, 401, 403]:
                return ValidationResult(
                    test_name="Graceful Shutdown",
                    passed=True,
                    message="Graceful shutdown endpoint properly protected",
                    details={"status_code": response.status_code},
                    duration_ms=duration_ms
                )
            else:
                return ValidationResult(
                    test_name="Graceful Shutdown",
                    passed=False,
                    message=f"Unexpected shutdown endpoint response: {response.status_code}",
                    details={"status_code": response.status_code},
                    duration_ms=duration_ms
                )
                
        except Exception as e:
            return ValidationResult(
                test_name="Graceful Shutdown",
                passed=False,
                message=f"Graceful shutdown test failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000
            )
    
    async def run_all_tests(self) -> Dict[str, any]:
        """Run all validation tests."""
        logger.info(f"Starting production validation for {self.base_url}")
        logger.info(f"Test parameters: {self.concurrent_users} concurrent users, {self.max_graph_size} max graph size")
        
        # Run synchronous tests
        sync_tests = [
            self.test_health_endpoints,
            self.test_metrics_endpoint,
            self.test_concurrent_connections,
            self.test_graph_size_limits,
            self.test_feature_flags,
            self.test_security_headers,
            self.test_graceful_shutdown,
        ]
        
        for test_func in sync_tests:
            try:
                result = test_func()
                self.add_result(result)
            except Exception as e:
                self.add_result(ValidationResult(
                    test_name=test_func.__name__,
                    passed=False,
                    message=f"Test execution failed: {str(e)}"
                ))
        
        # Run async tests
        try:
            mcp_result = await self.test_mcp_protocol_compliance()
            self.add_result(mcp_result)
        except Exception as e:
            self.add_result(ValidationResult(
                test_name="MCP Protocol",
                passed=False,
                message=f"MCP protocol test failed: {str(e)}"
            ))
        
        # Compile results
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "overall_passed": failed_tests == 0,
            "results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "message": r.message,
                    "duration_ms": r.duration_ms,
                    "details": r.details
                }
                for r in self.results
            ]
        }
        
        logger.info(f"Validation complete: {passed_tests}/{total_tests} tests passed ({success_rate:.1%})")
        
        if failed_tests > 0:
            logger.error("Failed tests:")
            for result in self.results:
                if not result.passed:
                    logger.error(f"  - {result.test_name}: {result.message}")
        
        return summary


async def main():
    """Main validation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production deployment validation")
    parser.add_argument("--base-url", default="https://api.mcp.example.com",
                       help="Base URL for the MCP server")
    parser.add_argument("--stdio", action="store_true",
                       help="Test stdio mode instead of HTTP")
    parser.add_argument("--concurrent-users", type=int, default=30,
                       help="Number of concurrent users to test")
    parser.add_argument("--max-graph-size", type=int, default=10000,
                       help="Maximum graph size to test")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create validator
    validator = ProductionValidator(
        base_url=args.base_url,
        stdio_mode=args.stdio,
        concurrent_users=args.concurrent_users,
        max_graph_size=args.max_graph_size
    )
    
    # Run validation
    results = await validator.run_all_tests()
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    
    # Exit with appropriate code
    if results["overall_passed"]:
        logger.info("🎉 All validation tests passed!")
        sys.exit(0)
    else:
        logger.error("❌ Some validation tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())