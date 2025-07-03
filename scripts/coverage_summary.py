#!/usr/bin/env python3
"""Generate a coverage summary for the NetworkX MCP Server."""

import subprocess


def run_coverage_summary():
    """Run tests and generate coverage summary."""

    print("📊 NetworkX MCP Server - Coverage Summary")
    print("=" * 60)

    # Run tests with coverage
    cmd = [
        "pytest",
        "tests/unit/test_basic.py",
        "tests/unit/test_algorithms.py",
        "tests/test_feature_audit.py",
        "-v",
        "--cov=src/networkx_mcp",
        "--cov-report=term",
        "--cov-report=html",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse coverage from output
    lines = result.stdout.split("\n")
    in_coverage = False
    coverage_lines = []

    for line in lines:
        if "Name" in line and "Stmts" in line:
            in_coverage = True
        if in_coverage:
            coverage_lines.append(line)
        if "TOTAL" in line and in_coverage:
            break

    # Print coverage table
    print("\n📊 Module Coverage:")
    print("-" * 60)

    # Show key modules
    key_modules = [
        "algorithms.py",
        "graph_operations.py",
        "server.py",
        "validation.py",
        "metrics.py",
        "logging.py",
    ]

    for line in coverage_lines:
        for module in key_modules:
            if module in line:
                print(line)

    # Print total
    for line in coverage_lines:
        if "TOTAL" in line:
            print("\n" + "=" * 60)
            print(line)
            parts = line.split()
            if len(parts) >= 5:
                coverage_pct = parts[-1]
                print(f"\n📈 Overall Coverage: {coverage_pct}")

    # Summary by category
    print("\n📋 Coverage by Category:")
    print("-" * 60)

    categories = {
        "Core": ["algorithms", "graph_operations", "base"],
        "MCP/Server": ["server", "handlers", "resources", "prompts"],
        "Security": ["validation", "auth", "middleware", "audit"],
        "Monitoring": ["metrics", "logging", "health_checks", "tracing"],
        "Advanced": ["community", "bipartite", "ml_integration", "flow"],
    }

    for category, modules in categories.items():
        total_lines = 0
        covered_lines = 0
        count = 0

        for line in coverage_lines:
            for module in modules:
                if module in line and "TOTAL" not in line:
                    parts = line.split()
                    if len(parts) >= 5 and parts[1].isdigit():
                        stmts = int(parts[1])
                        miss = int(parts[2])
                        covered = stmts - miss
                        total_lines += stmts
                        covered_lines += covered
                        count += 1
                        break

        if total_lines > 0:
            category_coverage = (covered_lines / total_lines) * 100
            print(f"{category:15} {category_coverage:5.1f}% ({count} modules)")

    print("\n✅ Coverage report generated: htmlcov/index.html")
    return True


if __name__ == "__main__":
    run_coverage_summary()
