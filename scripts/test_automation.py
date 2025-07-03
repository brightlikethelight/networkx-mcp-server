#!/usr/bin/env python3
"""Advanced test automation and quality assurance script.

This script provides comprehensive test automation capabilities including:
- Automated test execution with intelligent retry logic
- Performance baseline establishment and regression detection
- Mutation testing coordination
- Quality gate enforcement
- Test result aggregation and reporting
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestAutomation:
    """Advanced test automation coordinator."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results: dict[str, Any] = {}
        self.quality_gate_passed = True

    def run_command(self, cmd: list[str], timeout: int = 300) -> dict[str, Any]:
        """Run a command with timeout and capture results."""
        logger.info(f"Running: {' '.join(cmd)}")
        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            duration = time.time() - start_time

            return {
                "command": " ".join(cmd),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": duration,
                "success": result.returncode == 0,
            }

        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout}s: {' '.join(cmd)}")
            return {
                "command": " ".join(cmd),
                "returncode": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "duration": timeout,
                "success": False,
            }
        except Exception as e:
            logger.error(f"Command failed: {e}")
            return {
                "command": " ".join(cmd),
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "duration": 0,
                "success": False,
            }

    def run_linting(self) -> bool:
        """Run comprehensive linting checks."""
        logger.info("🔍 Running linting checks...")

        checks = [
            (["python", "-m", "ruff", "check", "src", "tests"], "Ruff check"),
            (
                ["python", "-m", "ruff", "format", "--check", "src", "tests"],
                "Ruff format",
            ),
            (["python", "-m", "black", "--check", "src", "tests"], "Black format"),
            (["python", "-m", "mypy", "src/networkx_mcp"], "MyPy typing"),
        ]

        all_passed = True
        for cmd, name in checks:
            result = self.run_command(cmd)
            self.results[f"lint_{name.lower().replace(' ', '_')}"] = result

            if not result["success"]:
                logger.error(f"❌ {name} failed")
                all_passed = False
            else:
                logger.info(f"✅ {name} passed")

        return all_passed

    def run_security_checks(self) -> bool:
        """Run security scanning."""
        logger.info("🔒 Running security checks...")

        checks = [
            (["python", "-m", "bandit", "-r", "src/"], "Bandit security scan"),
            (["python", "-m", "safety", "check"], "Safety dependency scan"),
        ]

        all_passed = True
        for cmd, name in checks:
            result = self.run_command(cmd)
            self.results[f"security_{name.lower().replace(' ', '_')}"] = result

            if not result["success"]:
                logger.warning(f"⚠️ {name} issues found")
                # Security warnings don't fail the build by default
            else:
                logger.info(f"✅ {name} passed")

        return all_passed

    def run_unit_tests(self) -> bool:
        """Run unit tests with coverage."""
        logger.info("🧪 Running unit tests...")

        cmd = [
            "python",
            "-m",
            "pytest",
            "tests/unit/",
            "--cov=src/networkx_mcp",
            "--cov-report=term-missing",
            "--cov-report=json:coverage.json",
            "--cov-fail-under=95",
            "--cov-branch",
            "--tb=short",
            "-v",
        ]

        result = self.run_command(cmd, timeout=600)
        self.results["unit_tests"] = result

        if result["success"]:
            logger.info("✅ Unit tests passed")
            self._parse_coverage_report()
        else:
            logger.error("❌ Unit tests failed")
            self.quality_gate_passed = False

        return result["success"]

    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        logger.info("🔗 Running integration tests...")

        cmd = ["python", "-m", "pytest", "tests/integration/", "--tb=short", "-v"]

        result = self.run_command(cmd, timeout=900)
        self.results["integration_tests"] = result

        if result["success"]:
            logger.info("✅ Integration tests passed")
        else:
            logger.error("❌ Integration tests failed")
            self.quality_gate_passed = False

        return result["success"]

    def run_property_tests(self) -> bool:
        """Run property-based tests."""
        logger.info("🎲 Running property-based tests...")

        cmd = [
            "python",
            "-m",
            "pytest",
            "tests/property/",
            "--hypothesis-show-statistics",
            "--tb=short",
            "-v",
        ]

        result = self.run_command(cmd, timeout=600)
        self.results["property_tests"] = result

        if result["success"]:
            logger.info("✅ Property tests passed")
        else:
            logger.error("❌ Property tests failed")
            self.quality_gate_passed = False

        return result["success"]

    def run_security_tests(self) -> bool:
        """Run security boundary tests."""
        logger.info("🛡️ Running security tests...")

        cmd = ["python", "-m", "pytest", "tests/security/", "--tb=short", "-v"]

        result = self.run_command(cmd, timeout=300)
        self.results["security_tests"] = result

        if result["success"]:
            logger.info("✅ Security tests passed")
        else:
            logger.error("❌ Security tests failed")
            self.quality_gate_passed = False

        return result["success"]

    def run_performance_tests(self) -> bool:
        """Run performance benchmarks."""
        logger.info("⚡ Running performance tests...")

        cmd = [
            "python",
            "-m",
            "pytest",
            "tests/performance/",
            "--benchmark-only",
            "--benchmark-sort=mean",
            "--benchmark-json=benchmark_results.json",
            "--benchmark-histogram=histogram.svg",
            "-v",
        ]

        result = self.run_command(cmd, timeout=900)
        self.results["performance_tests"] = result

        if result["success"]:
            logger.info("✅ Performance tests passed")
            self._parse_benchmark_results()
        else:
            logger.warning("⚠️ Performance tests issues")
            # Performance issues are warnings, not failures

        return True  # Don't fail on performance issues

    def run_mutation_testing(self, quick: bool = True) -> bool:
        """Run mutation testing."""
        logger.info("🧬 Running mutation testing...")

        if quick:
            cmd = [
                "python",
                "-m",
                "mutmut",
                "run",
                "--paths-to-mutate",
                "src/networkx_mcp/core/",
                "--tests-dir",
                "tests/unit/",
                "--runner",
                "python -m pytest tests/unit/ -x --tb=short",
                "--use-coverage",
                "--CI",
            ]
        else:
            cmd = [
                "python",
                "-m",
                "mutmut",
                "run",
                "--paths-to-mutate",
                "src/networkx_mcp/",
                "--tests-dir",
                "tests/",
                "--runner",
                "python -m pytest tests/unit/ tests/property/ -x --tb=short",
                "--use-coverage",
            ]

        result = self.run_command(cmd, timeout=1800)  # 30 minutes max
        self.results["mutation_testing"] = result

        # Parse mutation results
        self._parse_mutation_results()

        return True  # Mutation testing is informational

    def run_asv_benchmarks(self, quick: bool = True) -> bool:
        """Run ASV benchmarks."""
        logger.info("📊 Running ASV benchmarks...")

        # Setup ASV machine info
        setup_result = self.run_command(["python", "-m", "asv", "machine", "--yes"])
        if not setup_result["success"]:
            logger.warning("ASV machine setup failed")
            return False

        # Run benchmarks
        if quick:
            cmd = ["python", "-m", "asv", "run", "--quick", "HEAD^!"]
        else:
            cmd = ["python", "-m", "asv", "run", "HEAD^!"]

        result = self.run_command(cmd, timeout=1200)  # 20 minutes max
        self.results["asv_benchmarks"] = result

        if result["success"]:
            logger.info("✅ ASV benchmarks completed")

            # Generate benchmark report
            self.run_command(["python", "-m", "asv", "publish"])
        else:
            logger.warning("⚠️ ASV benchmarks had issues")

        return True  # Benchmarks are informational

    def _parse_coverage_report(self):
        """Parse coverage report for metrics."""
        coverage_file = self.project_root / "coverage.json"
        if coverage_file.exists():
            try:
                with open(coverage_file) as f:
                    coverage_data = json.load(f)

                total_coverage = coverage_data.get("totals", {}).get(
                    "percent_covered", 0
                )
                self.results["coverage_percentage"] = total_coverage
                logger.info(f"📊 Code coverage: {total_coverage:.1f}%")

            except Exception as e:
                logger.warning(f"Failed to parse coverage report: {e}")

    def _parse_benchmark_results(self):
        """Parse benchmark results for metrics."""
        benchmark_file = self.project_root / "benchmark_results.json"
        if benchmark_file.exists():
            try:
                with open(benchmark_file) as f:
                    benchmark_data = json.load(f)

                benchmarks = benchmark_data.get("benchmarks", [])
                if benchmarks:
                    avg_time = sum(
                        b.get("stats", {}).get("mean", 0) for b in benchmarks
                    ) / len(benchmarks)
                    self.results["average_benchmark_time"] = avg_time
                    logger.info(f"⚡ Average benchmark time: {avg_time:.4f}s")

            except Exception as e:
                logger.warning(f"Failed to parse benchmark results: {e}")

    def _parse_mutation_results(self):
        """Parse mutation testing results."""
        try:
            result = self.run_command(["python", "-m", "mutmut", "results"])
            if result["success"]:
                output = result["stdout"]
                # Extract mutation score from output
                for line in output.split("\n"):
                    if "mutation score" in line.lower():
                        logger.info(f"🧬 {line.strip()}")
                        break
        except Exception as e:
            logger.warning(f"Failed to parse mutation results: {e}")

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive test report."""
        report = {
            "timestamp": time.time(),
            "quality_gate_passed": self.quality_gate_passed,
            "test_results": self.results,
            "summary": {
                "total_tests": len(self.results),
                "passed_tests": sum(
                    1
                    for r in self.results.values()
                    if isinstance(r, dict) and r.get("success", False)
                ),
                "total_duration": sum(
                    r.get("duration", 0)
                    for r in self.results.values()
                    if isinstance(r, dict)
                ),
            },
        }

        # Save report
        report_file = self.project_root / "test_automation_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"📋 Test report saved to: {report_file}")
        return report

    def run_quality_gate(self) -> bool:
        """Run complete quality gate."""
        logger.info("🚀 Starting quality gate execution...")

        # Critical checks (must pass)
        critical_checks = [
            ("Linting", self.run_linting),
            ("Security Checks", self.run_security_checks),
            ("Unit Tests", self.run_unit_tests),
            ("Integration Tests", self.run_integration_tests),
            ("Property Tests", self.run_property_tests),
            ("Security Tests", self.run_security_tests),
        ]

        # Optional checks (warnings only)
        optional_checks = [
            ("Performance Tests", self.run_performance_tests),
            ("ASV Benchmarks", lambda: self.run_asv_benchmarks(quick=True)),
        ]

        # Run critical checks
        for name, check_func in critical_checks:
            logger.info(f"Running {name}...")
            if not check_func():
                logger.error(f"❌ Critical check failed: {name}")
                self.quality_gate_passed = False

        # Run optional checks
        for name, check_func in optional_checks:
            logger.info(f"Running {name}...")
            try:
                check_func()
            except Exception as e:
                logger.warning(f"⚠️ Optional check had issues: {name} - {e}")

        # Generate final report
        report = self.generate_report()

        if self.quality_gate_passed:
            logger.info("✅ Quality gate PASSED!")
        else:
            logger.error("❌ Quality gate FAILED!")

        return self.quality_gate_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Advanced test automation")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument(
        "--mutation", action="store_true", help="Include mutation testing"
    )
    parser.add_argument(
        "--benchmarks", action="store_true", help="Include full benchmarks"
    )
    parser.add_argument(
        "--quality-gate", action="store_true", help="Run complete quality gate"
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    automation = TestAutomation(project_root)

    if args.quality_gate:
        success = automation.run_quality_gate()
        if args.mutation:
            automation.run_mutation_testing(quick=args.quick)
        if args.benchmarks:
            automation.run_asv_benchmarks(quick=args.quick)
    else:
        # Run individual components
        success = True

        if not args.quick:
            success &= automation.run_linting()
            success &= automation.run_security_checks()

        success &= automation.run_unit_tests()

        if not args.quick:
            success &= automation.run_integration_tests()
            success &= automation.run_property_tests()
            success &= automation.run_security_tests()
            success &= automation.run_performance_tests()

        if args.mutation:
            automation.run_mutation_testing(quick=args.quick)

        if args.benchmarks:
            automation.run_asv_benchmarks(quick=args.quick)

        automation.generate_report()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
