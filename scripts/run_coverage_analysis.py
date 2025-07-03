#!/usr/bin/env python3
"""Run coverage analysis on NetworkX MCP Server."""

import subprocess
import sys


def run_coverage_analysis():
    """Run coverage analysis in smaller chunks to avoid timeouts."""

    print("🧪 NetworkX MCP Server - Coverage Analysis")
    print("=" * 50)

    # Define test groups to run separately
    test_groups = [
        ("Unit Tests - Basic", "tests/unit/test_basic.py tests/unit/test_ci_basic.py"),
        ("Unit Tests - Algorithms", "tests/unit/test_algorithms.py"),
        ("Unit Tests - Graph Operations", "tests/unit/test_graph_operations.py"),
        ("Unit Tests - Visualization", "tests/unit/test_visualization.py"),
        ("Integration - Server", "tests/integration/test_server_startup.py"),
        ("Feature Audit", "tests/test_feature_audit.py"),
        ("Core Operations", "tests/test_core_operations.py"),
    ]

    overall_passed = 0
    overall_failed = 0

    for group_name, test_path in test_groups:
        print(f"\n📊 Running {group_name}...")
        print("-" * 40)

        cmd = [
            "pytest",
            test_path,
            "-v",
            "--tb=short",
            "--timeout=10",
            "--cov=src/networkx_mcp",
            "--cov-report=term-missing:skip-covered",
            "--cov-append",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Parse output for pass/fail counts
            output_lines = result.stdout.split("\n")
            for line in output_lines:
                if "passed" in line or "failed" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed":
                            try:
                                overall_passed += int(parts[i - 1])
                            except:
                                pass
                        elif part == "failed":
                            try:
                                overall_failed += int(parts[i - 1])
                            except:
                                pass

            # Show abbreviated output
            if result.returncode != 0:
                print(f"❌ {group_name} had failures")
                # Show last few lines of error
                error_lines = result.stdout.split("\n")[-10:]
                for line in error_lines:
                    if line.strip():
                        print(f"  {line}")
            else:
                print(f"✅ {group_name} passed")

        except Exception as e:
            print(f"⚠️  Error running {group_name}: {e}")

    # Generate final coverage report
    print("\n📊 Generating Coverage Report...")
    print("=" * 50)

    subprocess.run(
        [
            "pytest",
            "--cov=src/networkx_mcp",
            "--cov-report=html",
            "--cov-report=term:skip-covered",
            "--no-cov-on-fail",
            "--co",  # Collect only, don't run
        ]
    )

    print(f"\n✅ Total Passed: {overall_passed}")
    print(f"❌ Total Failed: {overall_failed}")
    print("📊 Coverage report: htmlcov/index.html")

    return overall_failed == 0


if __name__ == "__main__":
    success = run_coverage_analysis()
    sys.exit(0 if success else 1)
