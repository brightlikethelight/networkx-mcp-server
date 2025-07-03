#!/usr/bin/env python3
"""Advanced code quality gate for NetworkX MCP Server.

This script implements a comprehensive code quality gate that runs multiple
static analysis tools, security scans, and quality metrics to ensure
enterprise-grade code standards.
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ANSI color codes
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


@dataclass
class QualityResult:
    """Result of a quality check."""

    tool: str
    passed: bool
    score: float | None = None
    issues: int = 0
    warnings: int = 0
    errors: int = 0
    duration: float = 0.0
    output: str = ""
    details: dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class QualityGate:
    """Comprehensive code quality gate implementation."""

    def __init__(self, project_root: Path, config: dict | None = None):
        self.project_root = project_root
        self.config = config or {}
        self.results: list[QualityResult] = []
        self.overall_passed = True

        # Quality thresholds
        self.thresholds = {
            "coverage": 95.0,
            "complexity": 10,
            "duplication": 5.0,
            "maintainability": 7.0,
            "security_score": 9.0,
            "type_coverage": 90.0,
        }

        # Update thresholds from config
        if "thresholds" in self.config:
            self.thresholds.update(self.config["thresholds"])

    def run_command(self, cmd: list[str], timeout: int = 300) -> tuple[int, str, str]:
        """Run a command and return (returncode, stdout, stderr)."""
        try:
            logger.info(f"Running: {' '.join(cmd)}")
            start_time = time.time()

            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            duration = time.time() - start_time
            logger.info(f"Command completed in {duration:.2f}s")

            return result.returncode, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout}s")
            return -1, "", f"Command timed out after {timeout}s"
        except Exception as e:
            logger.error(f"Command failed: {e}")
            return -1, "", str(e)

    def check_ruff_linting(self) -> QualityResult:
        """Run Ruff linting checks."""
        start_time = time.time()

        # Run Ruff check
        returncode, stdout, stderr = self.run_command(
            [
                "python",
                "-m",
                "ruff",
                "check",
                "src",
                "tests",
                "scripts",
                "--output-format=json",
            ]
        )

        duration = time.time() - start_time

        if returncode == 0:
            issues = 0
            warnings = 0
            errors = 0
        else:
            try:
                output_data = json.loads(stdout) if stdout else []
                issues = len(output_data)
                warnings = sum(
                    1 for item in output_data if item.get("severity") == "warning"
                )
                errors = sum(
                    1 for item in output_data if item.get("severity") == "error"
                )
            except json.JSONDecodeError:
                issues = 1
                warnings = 0
                errors = 1

        return QualityResult(
            tool="ruff",
            passed=returncode == 0,
            issues=issues,
            warnings=warnings,
            errors=errors,
            duration=duration,
            output=stdout if returncode != 0 else "No issues found",
            details={"command": "ruff check"},
        )

    def check_mypy_typing(self) -> QualityResult:
        """Run MyPy type checking."""
        start_time = time.time()

        returncode, stdout, stderr = self.run_command(
            [
                "python",
                "-m",
                "mypy",
                "src/networkx_mcp",
                "--json-report",
                ".mypy_report",
            ]
        )

        duration = time.time() - start_time

        # Parse MyPy output
        errors = stdout.count("error:") if stdout else 0
        warnings = stdout.count("warning:") if stdout else 0

        # Try to get type coverage from report
        type_coverage = None
        report_file = self.project_root / ".mypy_report" / "index.txt"
        if report_file.exists():
            try:
                content = report_file.read_text()
                # Extract coverage percentage from MyPy report
                for line in content.split("\n"):
                    if "type coverage" in line.lower():
                        # Parse percentage from line
                        import re

                        match = re.search(r"(\d+(?:\.\d+)?)%", line)
                        if match:
                            type_coverage = float(match.group(1))
                            break
            except Exception:
                pass

        passed = returncode == 0 and (
            type_coverage is None or type_coverage >= self.thresholds["type_coverage"]
        )

        return QualityResult(
            tool="mypy",
            passed=passed,
            score=type_coverage,
            issues=errors + warnings,
            warnings=warnings,
            errors=errors,
            duration=duration,
            output=stdout,
            details={
                "type_coverage": type_coverage,
                "threshold": self.thresholds["type_coverage"],
            },
        )

    def check_bandit_security(self) -> QualityResult:
        """Run Bandit security analysis."""
        start_time = time.time()

        returncode, stdout, stderr = self.run_command(
            ["python", "-m", "bandit", "-r", "src", "-f", "json", "-q"]
        )

        duration = time.time() - start_time

        high_issues = 0
        medium_issues = 0
        low_issues = 0

        if returncode != 0 and stdout:
            try:
                data = json.loads(stdout)
                results = data.get("results", [])

                for result in results:
                    severity = result.get("issue_severity", "").lower()
                    if severity == "high":
                        high_issues += 1
                    elif severity == "medium":
                        medium_issues += 1
                    elif severity == "low":
                        low_issues += 1

            except json.JSONDecodeError:
                high_issues = 1  # Assume worst case

        # Calculate security score (10 - issues with weights)
        security_score = max(
            0, 10 - (high_issues * 3 + medium_issues * 1.5 + low_issues * 0.5)
        )
        passed = security_score >= self.thresholds["security_score"]

        return QualityResult(
            tool="bandit",
            passed=passed,
            score=security_score,
            issues=high_issues + medium_issues + low_issues,
            warnings=medium_issues + low_issues,
            errors=high_issues,
            duration=duration,
            output=stdout,
            details={
                "high_severity": high_issues,
                "medium_severity": medium_issues,
                "low_severity": low_issues,
                "security_score": security_score,
                "threshold": self.thresholds["security_score"],
            },
        )

    def check_complexity(self) -> QualityResult:
        """Check code complexity with radon."""
        start_time = time.time()

        # Install radon if not available
        try:
            subprocess.run(
                ["python", "-c", "import radon"], check=True, capture_output=True
            )
        except subprocess.CalledProcessError:
            subprocess.run(["python", "-m", "pip", "install", "radon"], check=True)

        returncode, stdout, stderr = self.run_command(
            ["python", "-m", "radon", "cc", "src", "-j"]
        )

        duration = time.time() - start_time

        max_complexity = 0
        avg_complexity = 0
        total_functions = 0
        high_complexity_count = 0

        if returncode == 0 and stdout:
            try:
                data = json.loads(stdout)
                complexities = []

                for file_data in data.values():
                    for item in file_data:
                        if isinstance(item, dict) and "complexity" in item:
                            complexity = item["complexity"]
                            complexities.append(complexity)
                            total_functions += 1

                            if complexity > self.thresholds["complexity"]:
                                high_complexity_count += 1

                            max_complexity = max(max_complexity, complexity)

                if complexities:
                    avg_complexity = sum(complexities) / len(complexities)

            except json.JSONDecodeError:
                pass

        passed = max_complexity <= self.thresholds["complexity"]

        return QualityResult(
            tool="radon",
            passed=passed,
            score=avg_complexity,
            issues=high_complexity_count,
            warnings=high_complexity_count,
            errors=0,
            duration=duration,
            output=stdout,
            details={
                "max_complexity": max_complexity,
                "avg_complexity": avg_complexity,
                "total_functions": total_functions,
                "high_complexity_count": high_complexity_count,
                "threshold": self.thresholds["complexity"],
            },
        )

    def check_code_duplication(self) -> QualityResult:
        """Check for code duplication."""
        start_time = time.time()

        # Try to install and run jscpd for duplication detection
        try:
            # Check if Node.js is available for jscpd
            subprocess.run(["node", "--version"], check=True, capture_output=True)

            # Try to run jscpd
            returncode, stdout, stderr = self.run_command(
                [
                    "npx",
                    "jscpd",
                    "src",
                    "--reporters",
                    "json",
                    "--output",
                    ".jscpd-report",
                ]
            )

            duplication_percentage = 0.0
            if returncode == 0:
                report_file = self.project_root / ".jscpd-report" / "jscpd-report.json"
                if report_file.exists():
                    try:
                        with open(report_file) as f:
                            data = json.load(f)
                            duplication_percentage = data.get("statistics", {}).get(
                                "percentage", 0.0
                            )
                    except Exception:
                        pass

        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: simple line-based duplication check
            duplication_percentage = self._simple_duplication_check()
            stdout = f"Estimated duplication: {duplication_percentage:.1f}%"

        duration = time.time() - start_time
        passed = duplication_percentage <= self.thresholds["duplication"]

        return QualityResult(
            tool="duplication",
            passed=passed,
            score=duplication_percentage,
            issues=1 if duplication_percentage > self.thresholds["duplication"] else 0,
            warnings=(
                1 if duplication_percentage > self.thresholds["duplication"] else 0
            ),
            errors=0,
            duration=duration,
            output=stdout,
            details={
                "duplication_percentage": duplication_percentage,
                "threshold": self.thresholds["duplication"],
            },
        )

    def _simple_duplication_check(self) -> float:
        """Simple line-based duplication detection."""
        try:
            import hashlib
            from collections import defaultdict

            line_hashes = defaultdict(list)
            total_lines = 0

            for py_file in (self.project_root / "src").rglob("*.py"):
                try:
                    with open(py_file, encoding="utf-8") as f:
                        for line_no, line in enumerate(f, 1):
                            line = line.strip()
                            if len(line) > 10 and not line.startswith(
                                "#"
                            ):  # Skip short lines and comments
                                line_hash = hashlib.md5(line.encode()).hexdigest()
                                line_hashes[line_hash].append((py_file, line_no))
                                total_lines += 1
                except Exception:
                    continue

            # Count duplicated lines
            duplicated_lines = sum(
                len(locations) - 1
                for locations in line_hashes.values()
                if len(locations) > 1
            )

            return (duplicated_lines / total_lines * 100) if total_lines > 0 else 0.0

        except Exception:
            return 0.0

    def check_test_coverage(self) -> QualityResult:
        """Check test coverage."""
        start_time = time.time()

        returncode, stdout, stderr = self.run_command(
            [
                "python",
                "-m",
                "pytest",
                "--cov=src/networkx_mcp",
                "--cov-report=json:.coverage.json",
                "--cov-report=term-missing",
                "--cov-fail-under=95",
                "tests/unit/",
                "-q",
            ]
        )

        duration = time.time() - start_time

        coverage_percentage = 0.0
        coverage_file = self.project_root / ".coverage.json"

        if coverage_file.exists():
            try:
                with open(coverage_file) as f:
                    data = json.load(f)
                    coverage_percentage = data.get("totals", {}).get(
                        "percent_covered", 0.0
                    )
            except Exception:
                pass

        passed = coverage_percentage >= self.thresholds["coverage"]

        return QualityResult(
            tool="coverage",
            passed=passed,
            score=coverage_percentage,
            issues=1 if not passed else 0,
            warnings=1 if coverage_percentage < self.thresholds["coverage"] else 0,
            errors=0,
            duration=duration,
            output=stdout,
            details={
                "coverage_percentage": coverage_percentage,
                "threshold": self.thresholds["coverage"],
            },
        )

    def check_dependency_security(self) -> QualityResult:
        """Check for security vulnerabilities in dependencies."""
        start_time = time.time()

        returncode, stdout, stderr = self.run_command(
            [
                "python",
                "-m",
                "safety",
                "check",
                "--json",
                "--ignore",
                "70612",  # Ignore specific jinja2 issue if needed
            ]
        )

        duration = time.time() - start_time

        vulnerabilities = 0
        if returncode != 0 and stdout:
            try:
                data = json.loads(stdout)
                vulnerabilities = len(data)
            except json.JSONDecodeError:
                vulnerabilities = 1

        passed = vulnerabilities == 0

        return QualityResult(
            tool="safety",
            passed=passed,
            issues=vulnerabilities,
            warnings=vulnerabilities,
            errors=0,
            duration=duration,
            output=stdout,
            details={"vulnerabilities": vulnerabilities},
        )

    def run_quality_gate(self, checks: list[str] | None = None) -> bool:
        """Run the complete quality gate."""
        logger.info(f"{Colors.BOLD}🚀 Starting Quality Gate{Colors.END}")

        available_checks = {
            "ruff": self.check_ruff_linting,
            "mypy": self.check_mypy_typing,
            "bandit": self.check_bandit_security,
            "complexity": self.check_complexity,
            "duplication": self.check_code_duplication,
            "coverage": self.check_test_coverage,
            "safety": self.check_dependency_security,
        }

        if checks is None:
            checks = list(available_checks.keys())

        # Run each check
        for check_name in checks:
            if check_name not in available_checks:
                logger.warning(f"Unknown check: {check_name}")
                continue

            logger.info(f"Running {check_name} check...")
            try:
                result = available_checks[check_name]()
                self.results.append(result)

                if result.passed:
                    logger.info(f"{Colors.GREEN}✅ {check_name}: PASSED{Colors.END}")
                else:
                    logger.error(f"{Colors.RED}❌ {check_name}: FAILED{Colors.END}")
                    self.overall_passed = False

            except Exception as e:
                logger.error(f"{Colors.RED}❌ {check_name}: ERROR - {e}{Colors.END}")
                self.results.append(
                    QualityResult(
                        tool=check_name, passed=False, errors=1, output=str(e)
                    )
                )
                self.overall_passed = False

        return self.overall_passed

    def generate_report(self) -> dict[str, Any]:
        """Generate a comprehensive quality report."""
        total_issues = sum(r.issues for r in self.results)
        total_warnings = sum(r.warnings for r in self.results)
        total_errors = sum(r.errors for r in self.results)
        total_duration = sum(r.duration for r in self.results)

        # Calculate quality score (0-100)
        passed_checks = sum(1 for r in self.results if r.passed)
        quality_score = (passed_checks / len(self.results) * 100) if self.results else 0

        report = {
            "timestamp": time.time(),
            "overall_passed": self.overall_passed,
            "quality_score": quality_score,
            "summary": {
                "total_checks": len(self.results),
                "passed_checks": passed_checks,
                "failed_checks": len(self.results) - passed_checks,
                "total_issues": total_issues,
                "total_warnings": total_warnings,
                "total_errors": total_errors,
                "total_duration": total_duration,
            },
            "thresholds": self.thresholds,
            "results": [asdict(r) for r in self.results],
        }

        return report

    def print_summary(self):
        """Print a summary of the quality gate results."""
        print(f"\n{Colors.BOLD}📊 Quality Gate Summary{Colors.END}")
        print("=" * 50)

        for result in self.results:
            status = (
                f"{Colors.GREEN}✅ PASS{Colors.END}"
                if result.passed
                else f"{Colors.RED}❌ FAIL{Colors.END}"
            )
            score_info = (
                f" (Score: {result.score:.1f})" if result.score is not None else ""
            )
            issues_info = f" - {result.issues} issues" if result.issues > 0 else ""

            print(f"{result.tool:12} {status}{score_info}{issues_info}")

        print("=" * 50)

        total_issues = sum(r.issues for r in self.results)
        passed_checks = sum(1 for r in self.results if r.passed)
        quality_score = (passed_checks / len(self.results) * 100) if self.results else 0

        print(f"Quality Score: {quality_score:.1f}%")
        print(f"Total Issues:  {total_issues}")
        print(f"Checks Passed: {passed_checks}/{len(self.results)}")

        if self.overall_passed:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 Quality Gate PASSED!{Colors.END}")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ Quality Gate FAILED!{Colors.END}")

        print("\nDetailed report saved to: quality_report.json")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Advanced code quality gate")
    parser.add_argument(
        "--checks",
        nargs="+",
        choices=[
            "ruff",
            "mypy",
            "bandit",
            "complexity",
            "duplication",
            "coverage",
            "safety",
        ],
        help="Specific checks to run",
    )
    parser.add_argument("--config", type=str, help="Configuration file")
    parser.add_argument(
        "--output", type=str, default="quality_report.json", help="Output report file"
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop on first failure"
    )
    parser.add_argument(
        "--threshold-coverage",
        type=float,
        default=95.0,
        help="Coverage threshold percentage",
    )
    parser.add_argument(
        "--threshold-complexity",
        type=int,
        default=10,
        help="Maximum complexity threshold",
    )

    args = parser.parse_args()

    # Find project root
    project_root = Path(__file__).parent.parent

    # Load configuration
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = json.load(f)

    # Override thresholds from command line
    if "thresholds" not in config:
        config["thresholds"] = {}

    config["thresholds"]["coverage"] = args.threshold_coverage
    config["thresholds"]["complexity"] = args.threshold_complexity

    # Run quality gate
    gate = QualityGate(project_root, config)

    try:
        success = gate.run_quality_gate(args.checks)

        # Generate and save report
        report = gate.generate_report()

        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        gate.print_summary()

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.error("Quality gate interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Quality gate failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
