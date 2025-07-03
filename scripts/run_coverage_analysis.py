#!/usr/bin/env python3
"""Comprehensive Coverage Analysis Script for NetworkX MCP Server.

This script runs comprehensive test coverage analysis for Phase 2.1
to achieve 95%+ code coverage.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False, result.stderr
    return True, result.stdout


def main():
    """Run comprehensive coverage analysis."""
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("🚀 Phase 2.1: Test Coverage Explosion - Starting Comprehensive Analysis")
    print("=" * 70)
    
    # 1. Clean previous coverage data
    print("\n1. Cleaning previous coverage data...")
    run_command(["coverage", "erase"])
    
    # 2. Run basic unit tests with coverage
    print("\n2. Running basic unit tests...")
    success, output = run_command([
        "coverage", "run", "-m", "pytest", 
        "tests/unit/test_basic.py",
        "-v", "--tb=short"
    ])
    if not success:
        print(f"Basic tests failed: {output}")
        return 1
    
    # 3. Run property-based tests
    print("\n3. Running property-based tests...")
    success, output = run_command([
        "coverage", "run", "--append", "-m", "pytest",
        "tests/property/",
        "-v", "--hypothesis-show-statistics"
    ])
    if not success:
        print(f"Property tests failed: {output}")
        return 1
    
    # 4. Run security boundary tests  
    print("\n4. Running security boundary tests...")
    success, output = run_command([
        "coverage", "run", "--append", "-m", "pytest",
        "tests/security/",
        "-v"
    ])
    if not success:
        print(f"Security tests failed: {output}")
        return 1
    
    # 5. Run existing comprehensive tests
    print("\n5. Running existing comprehensive tests...")
    success, output = run_command([
        "coverage", "run", "--append", "-m", "pytest",
        "tests/unit/test_graph_operations.py",
        "tests/unit/test_error_handling.py",
        "-v", "--tb=short"
    ])
    if not success:
        print("Some existing tests failed, but continuing...")
    
    # 6. Generate coverage report
    print("\n6. Generating coverage report...")
    success, output = run_command([
        "coverage", "report", 
        "--include=src/networkx_mcp/*",
        "--omit=src/networkx_mcp/mcp_mock.py",
        "--show-missing"
    ])
    
    if success:
        print("\n📊 COVERAGE REPORT:")
        print(output)
        
        # Extract coverage percentage
        lines = output.split('\n')
        total_line = [line for line in lines if 'TOTAL' in line]
        if total_line:
            coverage_percent = total_line[0].split()[-1].replace('%', '')
            try:
                coverage_num = float(coverage_percent)
                print(f"\n🎯 Current Coverage: {coverage_percent}%")
                
                if coverage_num >= 95:
                    print("🎉 PHASE 2.1 COMPLETE: 95%+ Coverage Achieved!")
                    return 0
                elif coverage_num >= 80:
                    print("✅ Good progress! Targeting 95%+ coverage...")
                elif coverage_num >= 60:
                    print("📈 Making progress! Need more comprehensive tests...")
                else:
                    print("⚠️  Low coverage. Need extensive test development...")
                    
            except ValueError:
                print("Could not parse coverage percentage")
    
    # 7. Generate HTML coverage report
    print("\n7. Generating HTML coverage report...")
    success, output = run_command([
        "coverage", "html",
        "--include=src/networkx_mcp/*",
        "--omit=src/networkx_mcp/mcp_mock.py",
        "--directory=htmlcov"
    ])
    
    if success:
        print("📋 HTML coverage report generated in htmlcov/")
        print("   Open htmlcov/index.html in your browser to view detailed coverage")
    
    # 8. Coverage suggestions
    print("\n🎯 PHASE 2.1 TARGETS:")
    print("   - Unit test coverage: 95%+ for all handlers")
    print("   - Property-based testing: Mathematical correctness")  
    print("   - Security boundary testing: Input validation")
    print("   - Performance monitoring: Regression detection")
    print("   - Integration testing: End-to-end workflows")
    
    print("\n📋 NEXT STEPS:")
    print("   - Review htmlcov/index.html for uncovered lines")
    print("   - Add tests for missing code paths")
    print("   - Implement mutation testing with mutmut")
    print("   - Add performance benchmarks")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())