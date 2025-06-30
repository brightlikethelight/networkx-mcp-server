#!/usr/bin/env python3
"""Update README badges with current status."""

from pathlib import Path


def update_badges():
    """Update status badges in README."""
    readme_path = Path("README.md")
    readme_content = readme_path.read_text()

    # Badge patterns to update
    badges = {
        "Python": "[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)",
        "License": "[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)",
        "CI": "[![CI](https://github.com/Bright-L01/networkx-mcp-server/actions/workflows/ci.yml/badge.svg)](https://github.com/Bright-L01/networkx-mcp-server/actions)",
        "Coverage": "[![Coverage](https://codecov.io/gh/Bright-L01/networkx-mcp-server/branch/main/graph/badge.svg)](https://codecov.io/gh/Bright-L01/networkx-mcp-server)",
        "PyPI": "[![PyPI](https://img.shields.io/pypi/v/networkx-mcp-server.svg)](https://pypi.org/project/networkx-mcp-server/)",
        "Downloads": "[![Downloads](https://pepy.tech/badge/networkx-mcp-server)](https://pepy.tech/project/networkx-mcp-server)",
        "Code style": "[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)",
        "Ruff": "[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)",
    }

    print("Current badges in README:")
    for name, badge in badges.items():
        if badge in readme_content:
            print(f"✅ {name} badge found")
        else:
            print(f"❌ {name} badge missing")

    # Add missing badges after the first line
    lines = readme_content.split("\n")
    if len(lines) > 2 and not lines[2].startswith("[!["):
        # Insert badges after title
        badge_line = " ".join(
            [
                badges["Python"],
                badges["License"],
                badges["CI"],
                badges["Code style"],
            ]
        )
        lines.insert(2, "")
        lines.insert(3, badge_line)

        updated_content = "\n".join(lines)
        readme_path.write_text(updated_content)
        print("\n✨ Added missing badges to README")
    else:
        print("\n✅ Badges already present in README")


if __name__ == "__main__":
    update_badges()
