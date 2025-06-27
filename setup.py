"""Setup script for NetworkX MCP Server."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="networkx-mcp-server",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive MCP server for NetworkX graph operations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/networkx-mcp-server",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "networkx>=3.0",
        "fastmcp>=0.1.0",
        "numpy>=1.20.0",
        "pyyaml>=6.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.20.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "full": [
            "pandas>=1.3.0",
            "scipy>=1.7.0",
            "matplotlib>=3.4.0",
            "pyvis>=0.3.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "networkx-mcp=networkx_mcp.server:main",
            "networkx-mcp-cli=networkx_mcp.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/networkx-mcp-server/issues",
        "Source": "https://github.com/yourusername/networkx-mcp-server",
        "Documentation": "https://github.com/yourusername/networkx-mcp-server/docs",
    },
)