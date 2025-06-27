#!/usr/bin/env python3
"""Generate comprehensive test report for NetworkX MCP Server."""

import subprocess
import time
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def run_command(cmd, capture=True):
    """Run command and return result."""
    try:
        if capture:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
            return result.returncode, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, shell=True, timeout=300)
            return result.returncode, "", ""
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def check_dependencies():
    """Check if all dependencies are installed."""
    dependencies = [
        "networkx", "fastmcp", "pytest", "pytest-cov", "pytest-asyncio",
        "matplotlib", "plotly", "numpy", "pandas", "community"
    ]
    
    missing = []
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            if dep == "community":
                try:
                    import community as community_louvain
                except ImportError:
                    missing.append("python-louvain")
            else:
                missing.append(dep)
    
    return missing


def run_tests_with_coverage():
    """Run full test suite with coverage."""
    print("🧪 Running comprehensive test suite...")
    
    cmd = "python3 -m pytest tests/ --cov=src/networkx_mcp --cov-report=term-missing --cov-report=html --cov-report=json -v"
    returncode, stdout, stderr = run_command(cmd)
    
    return returncode, stdout, stderr


def run_linting():
    """Run code linting checks."""
    print("🔍 Running code quality checks...")
    
    # Run ruff
    ruff_code, ruff_out, ruff_err = run_command("ruff check src/ tests/ --output-format=json")
    
    # Run mypy (if available)
    mypy_code, mypy_out, mypy_err = run_command("mypy src/ --ignore-missing-imports --json-report /dev/stdout")
    
    return {
        "ruff": {"code": ruff_code, "output": ruff_out, "errors": ruff_err},
        "mypy": {"code": mypy_code, "output": mypy_out, "errors": mypy_err}
    }


def check_examples():
    """Check if examples run successfully."""
    print("📚 Validating examples...")
    
    example_files = list(Path("examples").glob("*.py"))
    results = {}
    
    for example in example_files:
        print(f"  Testing {example.name}...")
        code, out, err = run_command(f"python3 {example}")
        results[example.name] = {
            "success": code == 0,
            "output_length": len(out),
            "has_errors": bool(err)
        }
    
    return results


def get_git_info():
    """Get git repository information."""
    try:
        # Get current commit
        code, commit, _ = run_command("git rev-parse HEAD")
        if code != 0:
            return {"error": "Not a git repository"}
        
        # Get commit count
        code, count, _ = run_command("git rev-list --count HEAD")
        
        # Get last commit message
        code, message, _ = run_command("git log -1 --pretty=%B")
        
        # Get current branch
        code, branch, _ = run_command("git branch --show-current")
        
        return {
            "commit": commit.strip(),
            "commit_count": int(count.strip()) if count.strip().isdigit() else 0,
            "last_message": message.strip(),
            "branch": branch.strip()
        }
    except Exception as e:
        return {"error": str(e)}


def count_mcp_tools():
    """Count MCP tools in server.py."""
    try:
        with open("src/networkx_mcp/server.py", "r") as f:
            content = f.read()
        
        tool_count = content.count("@mcp.tool")
        return tool_count
    except Exception as e:
        return f"Error: {e}"


def generate_report():
    """Generate comprehensive test report."""
    print("📊 NetworkX MCP Server Test Report Generator")
    print("=" * 50)
    
    timestamp = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    
    # Check dependencies
    print("📦 Checking dependencies...")
    missing_deps = check_dependencies()
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("   Install with: pip install " + " ".join(missing_deps))
        return False
    else:
        print("✅ All dependencies available")
    
    # Get git info
    git_info = get_git_info()
    
    # Count MCP tools
    tool_count = count_mcp_tools()
    
    # Run tests
    test_code, test_out, test_err = run_tests_with_coverage()
    
    # Parse coverage if available
    coverage_data = {}
    try:
        with open("coverage.json", "r") as f:
            coverage_data = json.load(f)
    except FileNotFoundError:
        pass
    
    # Run linting
    lint_results = run_linting()
    
    # Check examples
    example_results = check_examples()
    
    # Generate report
    report = {
        "report_metadata": {
            "generated_at": timestamp,
            "generator": "NetworkX MCP Server Test Suite",
            "version": "1.0.0"
        },
        "git_info": git_info,
        "implementation_status": {
            "mcp_tools_count": tool_count,
            "expected_tools": 39,
            "tools_complete": tool_count >= 39
        },
        "test_execution": {
            "exit_code": test_code,
            "success": test_code == 0,
            "has_output": bool(test_out),
            "has_errors": bool(test_err)
        },
        "code_quality": lint_results,
        "examples_validation": example_results,
        "dependencies": {
            "missing": missing_deps,
            "all_available": len(missing_deps) == 0
        }
    }
    
    # Add coverage data if available
    if coverage_data:
        report["coverage"] = {
            "overall_percentage": coverage_data.get("totals", {}).get("percent_covered", 0),
            "lines_covered": coverage_data.get("totals", {}).get("covered_lines", 0),
            "lines_total": coverage_data.get("totals", {}).get("num_statements", 0)
        }
    
    # Print summary
    print("\n" + "=" * 50)
    print("📋 TEST EXECUTION SUMMARY")
    print("=" * 50)
    
    print(f"📅 Generated: {timestamp}")
    print(f"🔧 MCP Tools: {tool_count}/39 ({tool_count >= 39 and '✅' or '❌'})")
    print(f"🧪 Tests: {'✅ PASSED' if test_code == 0 else '❌ FAILED'}")
    print(f"📚 Examples: {sum(1 for r in example_results.values() if r['success'])}/{len(example_results)} working")
    print(f"📦 Dependencies: {'✅ Complete' if not missing_deps else f'❌ Missing {len(missing_deps)}'}")
    
    if coverage_data:
        coverage_pct = coverage_data.get("totals", {}).get("percent_covered", 0)
        print(f"📊 Coverage: {coverage_pct:.1f}%")
    
    # Print details if tests failed
    if test_code != 0:
        print("\n❌ TEST FAILURES:")
        print("-" * 30)
        print(test_err[:1000] + ("..." if len(test_err) > 1000 else ""))
    
    # Save detailed report
    with open("test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: test_report.json")
    
    # Overall status
    overall_success = (
        test_code == 0 and
        tool_count >= 39 and
        not missing_deps and
        sum(1 for r in example_results.values() if r['success']) >= len(example_results) * 0.8
    )
    
    print(f"\n🎯 OVERALL STATUS: {'✅ PRODUCTION READY' if overall_success else '⚠️  NEEDS ATTENTION'}")
    
    return overall_success


if __name__ == "__main__":
    success = generate_report()
    sys.exit(0 if success else 1)