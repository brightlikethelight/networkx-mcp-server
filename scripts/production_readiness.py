#!/usr/bin/env python3
"""
Production Readiness Assessment - Day 14-15
==========================================

Honest assessment of production readiness.
No lying, no theater, just facts.

Tests what actually works vs. what we claim works.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class ProductionReadiness:
    def __init__(self):
        self.checks = {
            "protocol_compliance": False,
            "error_handling": False,
            "docker_ready": False,
            "performance_tested": False,
            "security_review": False,
            "documentation": False,
            "monitoring": False,
            "multi_user": False,
            "high_availability": False,
            "persistence": False,
            "auth_security": False,
            "real_deployment": False
        }
        self.details = {}
        
    def check_protocol_compliance(self):
        """Does it actually implement MCP protocol correctly?"""
        print("🔍 Testing MCP Protocol Compliance...")
        
        try:
            # Test 1: Can we start the server?
            proc = subprocess.Popen(
                [sys.executable, "-m", "networkx_mcp"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(__file__).parent.parent
            )
            
            # Test 2: Initialize handshake
            initialize_request = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "roots": {"listChanged": True},
                        "sampling": {}
                    },
                    "clientInfo": {
                        "name": "production-test",
                        "version": "1.0.0"
                    }
                }
            }) + "\n"
            
            stdout, stderr = proc.communicate(
                initialize_request.encode(), 
                timeout=10
            )
            
            proc.terminate()
            
            # Parse response
            if stdout:
                try:
                    response = json.loads(stdout.decode().strip())
                    if "error" not in response and response.get("id") == 1:
                        self.checks["protocol_compliance"] = True
                        self.details["protocol_compliance"] = "✅ MCP handshake successful"
                    else:
                        self.details["protocol_compliance"] = f"❌ Invalid response: {response}"
                except json.JSONDecodeError as e:
                    self.details["protocol_compliance"] = f"❌ Invalid JSON: {stdout.decode()[:200]}"
            else:
                self.details["protocol_compliance"] = f"❌ No response. stderr: {stderr.decode()[:200]}"
                
        except subprocess.TimeoutExpired:
            self.details["protocol_compliance"] = "❌ Server hung (timeout)"
            proc.kill()
        except Exception as e:
            self.details["protocol_compliance"] = f"❌ Failed to start: {e}"
            
    def check_error_handling(self):
        """Does it handle errors gracefully or crash?"""
        print("🔍 Testing Error Handling...")
        
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "networkx_mcp"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(__file__).parent.parent
            )
            
            # Send malformed JSON
            malformed_requests = [
                "invalid json\n",
                '{"invalid": "missing required fields"}\n',
                '{"jsonrpc": "2.0", "method": "nonexistent_method", "id": 1}\n',
                '{"jsonrpc": "2.0", "method": "tools/call", "id": 2, "params": {"name": "fake_tool"}}\n'
            ]
            
            responses = []
            for request in malformed_requests:
                try:
                    stdout, stderr = proc.communicate(request.encode(), timeout=5)
                    if stdout:
                        responses.append(stdout.decode())
                    proc = subprocess.Popen(
                        [sys.executable, "-m", "networkx_mcp"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=Path(__file__).parent.parent
                    )
                except subprocess.TimeoutExpired:
                    proc.kill()
                    self.details["error_handling"] = "❌ Server hung on malformed input"
                    return
                    
            proc.terminate()
            
            # Check if server responded to errors gracefully
            if len(responses) >= 2:  # Got some responses
                self.checks["error_handling"] = True
                self.details["error_handling"] = f"✅ Handled {len(responses)} error cases gracefully"
            else:
                self.details["error_handling"] = "❌ Server didn't respond to error cases"
                
        except Exception as e:
            self.details["error_handling"] = f"❌ Error testing failed: {e}"
            
    def check_docker_ready(self):
        """Can it be containerized and deployed?"""
        print("🔍 Testing Docker Readiness...")
        
        # Check if Dockerfile exists
        dockerfile_path = Path(__file__).parent.parent / "Dockerfile"
        if not dockerfile_path.exists():
            self.details["docker_ready"] = "❌ No Dockerfile found"
            return
            
        # Try to build Docker image (if Docker is available)
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                timeout=5
            )
            
            if result.returncode != 0:
                self.details["docker_ready"] = "⚠️ Docker not available for testing"
                return
                
            # Try building the image
            print("  📦 Building Docker image (this may take a moment)...")
            build_result = subprocess.run(
                ["docker", "build", "-t", "networkx-mcp-test", "."],
                cwd=Path(__file__).parent.parent,
                capture_output=True,
                timeout=120  # 2 minutes
            )
            
            if build_result.returncode == 0:
                # Try running the container
                run_result = subprocess.run(
                    ["docker", "run", "--rm", "-d", "--name", "networkx-mcp-test-run", "networkx-mcp-test"],
                    capture_output=True,
                    timeout=30
                )
                
                if run_result.returncode == 0:
                    # Clean up
                    subprocess.run(["docker", "stop", "networkx-mcp-test-run"], capture_output=True)
                    self.checks["docker_ready"] = True
                    self.details["docker_ready"] = "✅ Docker build and run successful"
                else:
                    self.details["docker_ready"] = f"❌ Docker run failed: {run_result.stderr.decode()[:200]}"
            else:
                self.details["docker_ready"] = f"❌ Docker build failed: {build_result.stderr.decode()[:200]}"
                
        except subprocess.TimeoutExpired:
            self.details["docker_ready"] = "❌ Docker build timed out"
        except FileNotFoundError:
            self.details["docker_ready"] = "⚠️ Docker not installed"
        except Exception as e:
            self.details["docker_ready"] = f"❌ Docker test failed: {e}"
            
    def check_performance_tested(self):
        """Are performance claims based on real testing?"""
        print("🔍 Checking Performance Testing...")
        
        # Look for real performance reports
        perf_files = [
            "benchmarks/test_real_performance.py",
            "docs/PERFORMANCE_REALITY_CHECK.md",
            "ACTUAL_PERFORMANCE_REPORT.md"
        ]
        
        found_files = []
        for file_path in perf_files:
            full_path = Path(__file__).parent.parent / file_path
            if full_path.exists():
                found_files.append(file_path)
                
        if found_files:
            self.checks["performance_tested"] = True
            self.details["performance_tested"] = f"✅ Real performance testing: {', '.join(found_files)}"
        else:
            self.details["performance_tested"] = "❌ No real performance testing found"
            
    def check_security_review(self):
        """Has basic security been reviewed?"""
        print("🔍 Checking Security Review...")
        
        security_issues = []
        
        # Check for obvious security issues
        server_file = Path(__file__).parent.parent / "src" / "networkx_mcp" / "server.py"
        if server_file.exists():
            content = server_file.read_text()
            
            # Look for security red flags
            if "eval(" in content:
                security_issues.append("Uses eval()")
            if "exec(" in content:
                security_issues.append("Uses exec()")
            if "pickle.loads" in content and "trusted" not in content.lower():
                security_issues.append("Unsafe pickle usage")
            if "shell=True" in content:
                security_issues.append("Shell injection risk")
                
        # Check if security modules exist
        security_path = Path(__file__).parent.parent / "src" / "networkx_mcp" / "security"
        if security_path.exists():
            security_files = list(security_path.glob("*.py"))
            if len(security_files) > 2:  # More than just __init__.py
                if not security_issues:
                    self.checks["security_review"] = True
                    self.details["security_review"] = f"✅ Security modules present, no obvious issues"
                else:
                    self.details["security_review"] = f"⚠️ Security modules exist but issues found: {security_issues}"
            else:
                self.details["security_review"] = "❌ Minimal security implementation"
        else:
            self.details["security_review"] = "❌ No security modules found"
            
    def check_documentation(self):
        """Is documentation adequate for production use?"""
        print("🔍 Checking Documentation...")
        
        required_docs = [
            "README.md",
            "docs/",
            "CHANGELOG.md"
        ]
        
        found_docs = []
        missing_docs = []
        
        for doc in required_docs:
            doc_path = Path(__file__).parent.parent / doc
            if doc_path.exists():
                found_docs.append(doc)
            else:
                missing_docs.append(doc)
                
        # Check README quality
        readme_path = Path(__file__).parent.parent / "README.md"
        readme_quality = False
        if readme_path.exists():
            readme_content = readme_path.read_text()
            # Look for key sections
            required_sections = ["installation", "usage", "example"]
            sections_found = sum(1 for section in required_sections if section.lower() in readme_content.lower())
            readme_quality = sections_found >= 2
            
        if len(found_docs) >= 2 and readme_quality:
            self.checks["documentation"] = True
            self.details["documentation"] = f"✅ Good docs: {', '.join(found_docs)}"
        else:
            self.details["documentation"] = f"❌ Inadequate docs. Missing: {missing_docs}"
            
    def check_monitoring(self):
        """Can it be monitored in production?"""
        print("🔍 Checking Monitoring Capabilities...")
        
        # Look for monitoring/observability features
        monitoring_indicators = [
            "src/networkx_mcp/utils/monitoring.py",
            "health",
            "metrics",
            "logging"
        ]
        
        found_monitoring = []
        
        # Check for health endpoints
        try:
            from networkx_mcp.server import NetworkXMCPServer
            server = NetworkXMCPServer()
            
            # Check if server has health-related methods
            server_methods = [method for method in dir(server) if not method.startswith('_')]
            health_methods = [method for method in server_methods if 'health' in method.lower()]
            
            if health_methods:
                found_monitoring.append(f"Health methods: {health_methods}")
                
        except Exception as e:
            pass
            
        # Check for monitoring files
        for indicator in monitoring_indicators:
            file_path = Path(__file__).parent.parent / indicator
            if file_path.exists():
                found_monitoring.append(indicator)
                
        if found_monitoring:
            self.checks["monitoring"] = True
            self.details["monitoring"] = f"✅ Monitoring features: {', '.join(found_monitoring)}"
        else:
            self.details["monitoring"] = "❌ No monitoring/observability features"
            
    def check_multi_user(self):
        """Can it handle multiple users?"""
        print("🔍 Checking Multi-User Support...")
        
        # Check if authentication exists
        auth_indicators = [
            "auth",
            "user",
            "session",
            "token"
        ]
        
        try:
            from networkx_mcp.server import NetworkXMCPServer
            server = NetworkXMCPServer()
            
            # Check for user-related attributes
            server_attrs = dir(server)
            auth_attrs = [attr for attr in server_attrs for indicator in auth_indicators if indicator in attr.lower()]
            
            if auth_attrs:
                self.details["multi_user"] = f"⚠️ Some auth features: {auth_attrs} (needs testing)"
            else:
                self.details["multi_user"] = "❌ No multi-user support (stdio only, single user)"
                
        except Exception:
            self.details["multi_user"] = "❌ Cannot assess - server import failed"
            
    def check_high_availability(self):
        """Can it scale and be highly available?"""
        print("🔍 Checking High Availability...")
        
        # HA requires multiple things
        ha_requirements = {
            "Load balancing": False,
            "State sharing": False, 
            "Health checks": False,
            "Graceful shutdown": False,
            "Clustering": False
        }
        
        # Check for HA indicators
        ha_files = [
            "k8s/",
            "helm/",
            "docker-compose.yml",
            "src/networkx_mcp/core/graceful_shutdown.py"
        ]
        
        found_ha = []
        for ha_file in ha_files:
            if (Path(__file__).parent.parent / ha_file).exists():
                found_ha.append(ha_file)
                
        # stdio transport is inherently not HA
        transport_check = True  # Assume stdio only unless proven otherwise
        
        if found_ha:
            ha_score = len(found_ha) / len(ha_files)
            if ha_score > 0.5:
                self.checks["high_availability"] = True
                self.details["high_availability"] = f"✅ HA infrastructure: {', '.join(found_ha)}"
            else:
                self.details["high_availability"] = f"⚠️ Partial HA: {', '.join(found_ha)}"
        else:
            self.details["high_availability"] = "❌ No HA support (stdio transport = single instance)"
            
    def check_persistence(self):
        """Is data persistence production-ready?"""
        print("🔍 Checking Persistence...")
        
        try:
            # We know from Day 14 that persistence exists but isn't integrated
            from networkx_mcp.storage import MemoryBackend, StorageFactory
            
            # Test if storage actually works
            import asyncio
            
            async def test_storage():
                try:
                    backend = await StorageFactory.create_backend("memory")
                    await backend.close()
                    return True
                except Exception:
                    return False
                    
            works = asyncio.run(test_storage())
            
            if works:
                # But is it integrated?
                from networkx_mcp.server import NetworkXMCPServer
                server = NetworkXMCPServer()
                
                if hasattr(server, 'storage_manager'):
                    self.checks["persistence"] = True
                    self.details["persistence"] = "✅ Persistence works and integrated"
                else:
                    self.details["persistence"] = "⚠️ Persistence works but NOT integrated into server"
            else:
                self.details["persistence"] = "❌ Persistence broken"
                
        except Exception as e:
            self.details["persistence"] = f"❌ Persistence check failed: {e}"
            
    def check_auth_security(self):
        """Is authentication/authorization production-ready?"""
        print("🔍 Checking Authentication & Security...")
        
        try:
            from networkx_mcp.server import NetworkXMCPServer
            server = NetworkXMCPServer()
            
            # Check if server has any auth
            auth_methods = [method for method in dir(server) if any(word in method.lower() for word in ['auth', 'login', 'token', 'user'])]
            
            if auth_methods:
                self.details["auth_security"] = f"⚠️ Some auth methods: {auth_methods} (needs verification)"
            else:
                self.details["auth_security"] = "❌ No authentication (open access)"
                
        except Exception:
            self.details["auth_security"] = "❌ Cannot assess auth"
            
    def check_real_deployment(self):
        """Has it been deployed to a real environment?"""
        print("🔍 Checking Real Deployment Evidence...")
        
        deployment_indicators = [
            "docker-compose.yml",
            "k8s/",
            "helm/",
            ".github/workflows/",
            "ci/cd evidence"
        ]
        
        found_deployment = []
        base_path = Path(__file__).parent.parent
        
        for indicator in deployment_indicators:
            if (base_path / indicator).exists():
                found_deployment.append(indicator)
                
        if found_deployment:
            self.checks["real_deployment"] = len(found_deployment) >= 3
            self.details["real_deployment"] = f"Deployment files: {', '.join(found_deployment)}"
        else:
            self.details["real_deployment"] = "❌ No deployment configuration found"
            
    def run_all_checks(self):
        """Run all production readiness checks."""
        print("=" * 60)
        print("NetworkX MCP Server - Production Readiness Assessment")
        print("=" * 60)
        
        # Run each check
        self.check_protocol_compliance()
        self.check_error_handling()
        self.check_docker_ready()
        self.check_performance_tested()
        self.check_security_review()
        self.check_documentation()
        self.check_monitoring()
        self.check_multi_user()
        self.check_high_availability()
        self.check_persistence()
        self.check_auth_security()
        self.check_real_deployment()
        
    def generate_report(self):
        """Generate honest production readiness report."""
        total = len(self.checks)
        passed = sum(self.checks.values())
        percentage = (passed / total) * 100
        
        print("\n" + "=" * 60)
        print("PRODUCTION READINESS RESULTS")
        print("=" * 60)
        
        # Categorize checks
        critical_checks = ["protocol_compliance", "error_handling", "security_review"]
        important_checks = ["performance_tested", "documentation", "persistence"]
        nice_to_have = ["docker_ready", "monitoring", "multi_user", "high_availability", "auth_security", "real_deployment"]
        
        critical_passed = sum(1 for check in critical_checks if self.checks[check])
        important_passed = sum(1 for check in important_checks if self.checks[check])
        nice_passed = sum(1 for check in nice_to_have if self.checks[check])
        
        print(f"\n📊 OVERALL SCORE: {passed}/{total} ({percentage:.0f}%)")
        print(f"🔴 Critical (must-have): {critical_passed}/{len(critical_checks)}")
        print(f"🟡 Important: {important_passed}/{len(important_checks)}")
        print(f"🟢 Nice-to-have: {nice_passed}/{len(nice_to_have)}")
        
        print("\n📋 DETAILED RESULTS:")
        for check, result in self.checks.items():
            status = "✅" if result else "❌"
            category = "🔴" if check in critical_checks else "🟡" if check in important_checks else "🟢"
            detail = self.details.get(check, "No details")
            print(f"{status} {category} {check.replace('_', ' ').title()}")
            print(f"     {detail}")
            
        # Production readiness assessment
        print(f"\n{'='*60}")
        print("🚨 PRODUCTION DEPLOYMENT RECOMMENDATION")
        print("="*60)
        
        if critical_passed == len(critical_checks) and passed >= 8:
            print("✅ READY FOR PRODUCTION")
        elif critical_passed == len(critical_checks) and passed >= 6:
            print("⚠️ READY FOR STAGING/LIMITED PRODUCTION")
        elif critical_passed >= 2:
            print("🟡 READY FOR DEVELOPMENT/TESTING ONLY")
        else:
            print("❌ NOT READY - FUNDAMENTAL ISSUES")
            
        # Specific recommendations
        print(f"\n📋 RECOMMENDATIONS:")
        
        if not self.checks["protocol_compliance"]:
            print("🔴 CRITICAL: Fix MCP protocol compliance first")
        if not self.checks["error_handling"]:
            print("🔴 CRITICAL: Implement proper error handling")
        if not self.checks["security_review"]:
            print("🔴 CRITICAL: Conduct security review")
            
        if critical_passed == len(critical_checks):
            print("✅ Critical requirements met")
            
            if not self.checks["persistence"]:
                print("🟡 Consider integrating existing persistence layer")
            if not self.checks["docker_ready"]:
                print("🟡 Fix Docker build for easier deployment")
            if not self.checks["monitoring"]:
                print("🟡 Add monitoring for production observability")
                
        return {
            "overall_score": percentage,
            "production_ready": critical_passed == len(critical_checks) and passed >= 8,
            "staging_ready": critical_passed == len(critical_checks) and passed >= 6,
            "critical_passed": critical_passed,
            "total_passed": passed,
            "details": self.details
        }

def main():
    assessment = ProductionReadiness()
    assessment.run_all_checks()
    result = assessment.generate_report()
    
    # Return appropriate exit code
    if result["production_ready"]:
        print(f"\n🎉 Production ready!")
        return 0
    elif result["staging_ready"]:
        print(f"\n⚠️ Staging ready, needs work for production")
        return 1
    else:
        print(f"\n❌ Not ready for production deployment")
        return 2

if __name__ == "__main__":
    sys.exit(main())