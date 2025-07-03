"""Nox configuration for automated testing and quality assurance.

This file defines all the test automation sessions for the NetworkX MCP server,
including unit tests, integration tests, linting, type checking, security scanning,
mutation testing, and benchmarking.
"""

import nox

# Supported Python versions
PYTHON_VERSIONS = ["3.11", "3.12"]
LOCATIONS = ["src", "tests", "noxfile.py"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session):
    """Run the complete test suite with coverage."""
    session.install(".[dev]")
    session.run(
        "python",
        "-m",
        "pytest",
        "--cov=src/networkx_mcp",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-report=xml",
        "--cov-fail-under=95",
        "--cov-branch",
        *session.posargs,
    )


@nox.session(python=PYTHON_VERSIONS)
def unit_tests(session):
    """Run only unit tests for fast feedback."""
    session.install(".[dev]")
    session.run(
        "python",
        "-m",
        "pytest",
        "tests/unit/",
        "--cov=src/networkx_mcp",
        "--cov-report=term-missing",
        "--cov-fail-under=90",
        *session.posargs,
    )


@nox.session(python=PYTHON_VERSIONS)
def integration_tests(session):
    """Run integration tests."""
    session.install(".[dev]")
    session.run(
        "python",
        "-m",
        "pytest",
        "tests/integration/",
        "--cov=src/networkx_mcp",
        "--cov-report=term-missing",
        *session.posargs,
    )


@nox.session(python=PYTHON_VERSIONS)
def property_tests(session):
    """Run property-based tests with Hypothesis."""
    session.install(".[dev]")
    session.run(
        "python",
        "-m",
        "pytest",
        "tests/property/",
        "--hypothesis-show-statistics",
        *session.posargs,
    )


@nox.session(python=PYTHON_VERSIONS)
def security_tests(session):
    """Run security boundary tests."""
    session.install(".[dev]")
    session.run(
        "python", "-m", "pytest", "tests/security/", "--tb=short", *session.posargs
    )


@nox.session(python=PYTHON_VERSIONS)
def performance_tests(session):
    """Run performance monitoring tests."""
    session.install(".[dev]")
    session.run(
        "python",
        "-m",
        "pytest",
        "tests/performance/",
        "--benchmark-only",
        "--benchmark-sort=mean",
        "--benchmark-histogram=histogram",
        *session.posargs,
    )


@nox.session(python="3.11")
def lint(session):
    """Run linting with ruff."""
    session.install(".[dev]")
    session.run("python", "-m", "ruff", "check", *LOCATIONS)
    session.run("python", "-m", "ruff", "format", "--check", *LOCATIONS)


@nox.session(python="3.11")
def format(session):
    """Format code with ruff and black."""
    session.install(".[dev]")
    session.run("python", "-m", "ruff", "format", *LOCATIONS)
    session.run("python", "-m", "black", *LOCATIONS)


@nox.session(python="3.11")
def typecheck(session):
    """Run type checking with mypy."""
    session.install(".[dev]")
    session.run("python", "-m", "mypy", "src/networkx_mcp")


@nox.session(python="3.11")
def security_scan(session):
    """Run security scanning with bandit and safety."""
    session.install(".[dev]")
    session.run("python", "-m", "bandit", "-r", "src/")
    session.run("python", "-m", "safety", "check", "--json")


@nox.session(python="3.11")
def mutation_test(session):
    """Run mutation testing with mutmut."""
    session.install(".[dev]")
    # Run a subset of mutations for CI efficiency
    session.run(
        "python",
        "-m",
        "mutmut",
        "run",
        "--paths-to-mutate",
        "src/networkx_mcp/",
        "--tests-dir",
        "tests/unit/",
        "--runner",
        "python -m pytest tests/unit/ -x --tb=short",
        "--use-coverage",
        "--CI",
        *session.posargs,
    )


@nox.session(python="3.11")
def mutation_test_full(session):
    """Run full mutation testing suite (slow)."""
    session.install(".[dev]")
    session.run(
        "python",
        "-m",
        "mutmut",
        "run",
        "--paths-to-mutate",
        "src/networkx_mcp/",
        "--tests-dir",
        "tests/",
        "--runner",
        "python -m pytest tests/unit/ tests/property/ tests/security/ -x --tb=short",
        "--use-coverage",
        *session.posargs,
    )


@nox.session(python="3.11")
def benchmarks(session):
    """Run ASV benchmarks."""
    session.install(".[dev]")
    session.run("python", "-m", "asv", "machine", "--yes")
    session.run("python", "-m", "asv", "run", "--show-stderr")


@nox.session(python="3.11")
def benchmark_compare(session):
    """Compare benchmarks between commits."""
    session.install(".[dev]")
    session.run(
        "python",
        "-m",
        "asv",
        "compare",
        "HEAD~1",
        "HEAD",
        "--factor",
        "1.1",
        *session.posargs,
    )


@nox.session(python="3.11")
def benchmark_continuous(session):
    """Run continuous benchmarking for CI."""
    session.install(".[dev]")
    session.run("python", "-m", "asv", "run", "--quick")
    session.run("python", "-m", "asv", "publish")


@nox.session(python="3.11")
def docs_build(session):
    """Build documentation."""
    session.install(".[dev]")
    session.run("python", "-m", "mkdocs", "build", "--strict")


@nox.session(python="3.11")
def docs_serve(session):
    """Serve documentation locally."""
    session.install(".[dev]")
    session.run("python", "-m", "mkdocs", "serve")


@nox.session(python="3.11")
def quality_gate(session):
    """Run complete quality gate for CI/CD."""
    session.install(".[dev]")

    # Code quality
    session.run("python", "-m", "ruff", "check", *LOCATIONS)
    session.run("python", "-m", "ruff", "format", "--check", *LOCATIONS)
    session.run("python", "-m", "black", "--check", *LOCATIONS)
    session.run("python", "-m", "mypy", "src/networkx_mcp")

    # Security
    session.run("python", "-m", "bandit", "-r", "src/")
    session.run("python", "-m", "safety", "check")

    # Tests
    session.run(
        "python",
        "-m",
        "pytest",
        "--cov=src/networkx_mcp",
        "--cov-fail-under=95",
        "--cov-branch",
        "tests/unit/",
        "tests/integration/",
    )

    # Quick benchmarks
    session.run("python", "-m", "asv", "run", "--quick")


@nox.session(python="3.11")
def coverage_report(session):
    """Generate comprehensive coverage report."""
    session.install(".[dev]")
    session.run(
        "python",
        "-m",
        "pytest",
        "--cov=src/networkx_mcp",
        "--cov-report=html:htmlcov",
        "--cov-report=xml:coverage.xml",
        "--cov-report=json:coverage.json",
        "--cov-report=term-missing",
        "--cov-branch",
    )
    session.log("Coverage report generated in htmlcov/index.html")


@nox.session(python="3.11")
def install_pre_commit(session):
    """Install pre-commit hooks."""
    session.install(".[dev]")
    session.run("python", "-m", "pre_commit", "install")
    session.run("python", "-m", "pre_commit", "install", "--hook-type", "pre-push")


@nox.session(python="3.11")
def update_deps(session):
    """Update and check dependencies."""
    session.install(".[dev]")
    session.run("python", "-m", "pip", "list", "--outdated")
    session.run("python", "-m", "safety", "check")


@nox.session(python=PYTHON_VERSIONS)
def test_matrix(session):
    """Test against multiple NetworkX versions."""
    session.install(".[dev]")

    # Test with different NetworkX versions
    for nx_version in ["3.4", "3.3"]:
        session.install(f"networkx=={nx_version}")
        session.run("python", "-m", "pytest", "tests/unit/", "--tb=short", "-q")


@nox.session(python="3.11")
def performance_profile(session):
    """Profile performance of key operations."""
    session.install(".[dev]")
    session.install("py-spy", "memory-profiler")

    # Create a simple profiling script
    session.run(
        "python",
        "-c",
        """
import cProfile
import networkx as nx
from src.networkx_mcp.core.graph_operations import GraphManager

def profile_graph_ops():
    gm = GraphManager()
    G = nx.erdos_renyi_graph(1000, 0.1)
    gm.create_graph('test')
    gm.graphs['test'] = G
    return gm.get_graph_info('test')

cProfile.run('profile_graph_ops()', 'profile_stats.prof')
print('Profile saved to profile_stats.prof')
""",
    )


@nox.session(python="3.11")
def clean(session):
    """Clean up build artifacts and cache."""
    import os
    import shutil

    dirs_to_clean = [
        ".coverage",
        "htmlcov",
        "coverage.xml",
        "coverage.json",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "__pycache__",
        "build",
        "dist",
        "*.egg-info",
        ".asv",
        "benchmark.json",
        "profile_stats.prof",
        ".mutmut-cache",
        ".hypothesis",
    ]

    for item in dirs_to_clean:
        if os.path.exists(item):
            if os.path.isdir(item):
                shutil.rmtree(item)
                session.log(f"Removed directory: {item}")
            else:
                os.remove(item)
                session.log(f"Removed file: {item}")


# Configure default sessions
nox.options.sessions = ["lint", "typecheck", "tests"]
nox.options.reuse_existing_virtualenvs = True
nox.options.error_on_missing_interpreters = False
