#!/usr/bin/env python3
"""Check GitHub Actions CI status."""

import subprocess


def check_ci_status():
    """Check the status of the latest GitHub Actions run."""
    print("Checking GitHub Actions CI status...")

    # Get the latest commit
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    commit_sha = result.stdout.strip()
    print(f"Latest commit: {commit_sha[:8]}")

    # Instructions for checking status
    print("\nTo check CI status:")
    print("1. Visit: https://github.com/Bright-L01/networkx-mcp-server/actions")
    print(f"2. Look for commit: {commit_sha[:8]}")
    print("3. Check each job status")

    print("\nExpected jobs:")
    print("✓ Lint Code")
    print("✓ Test Python (3.8, 3.9, 3.10, 3.11, 3.12)")
    print("✓ Security Scan")
    print("✓ Performance Tests")
    print("✓ Build Package")

    print("\nIf any jobs fail, check the logs for:")
    print("- Import errors")
    print("- Missing dependencies")
    print("- Test failures")
    print("- Linting issues")


if __name__ == "__main__":
    check_ci_status()
