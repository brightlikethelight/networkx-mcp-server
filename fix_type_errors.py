#!/usr/bin/env python3
"""
Script to systematically fix type errors based on common patterns.
"""

import re
from pathlib import Path


def fix_generic_type_parameters(file_path: Path) -> bool:
    """Fix missing type parameters for generic types."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return False

    original_content = content

    # Fix dict without type parameters
    content = re.sub(r"\bdict\b(?!\[)", "dict[str, Any]", content)

    # Fix list without type parameters
    content = re.sub(r"\blist\b(?!\[)", "list[Any]", content)

    # Fix tuple without type parameters
    content = re.sub(r"\btuple\b(?!\[)", "tuple[Any, ...]", content)

    # Fix set without type parameters
    content = re.sub(r"\bset\b(?!\[)", "set[Any]", content)

    # Add Any import if used and not present
    if (
        "Any" in content
        and "from typing import" in content
        and "Any" not in original_content
    ):
        # Check if Any is already imported
        if re.search(r"from typing import [^)]*\bAny\b", content):
            pass  # Already imported
        else:
            # Add Any to existing imports
            content = re.sub(
                r"from typing import ([^)]+)",
                lambda m: f"from typing import {m.group(1)}, Any"
                if "Any" not in m.group(1)
                else m.group(0),
                content,
                count=1,
            )
    elif "Any" in content and "from typing import" not in content:
        # Add typing import at the top
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith('"""') and i > 0:
                # Find end of docstring
                j = i + 1
                while j < len(lines) and not lines[j].endswith('"""'):
                    j += 1
                lines.insert(j + 1, "")
                lines.insert(j + 2, "from typing import Any")
                break
            elif line.startswith("import ") or line.startswith("from "):
                lines.insert(i, "from typing import Any")
                break
        content = "\n".join(lines)

    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True

    return False


def fix_function_annotations(file_path: Path) -> bool:
    """Add missing type annotations to function parameters."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return False

    original_content = content

    # Pattern to find function definitions with untyped parameters
    # This is a simple heuristic - add : Any to parameters without type annotations
    pattern = r"def\s+\w+\s*\(([^)]*)\)"

    def fix_params(match):
        params = match.group(1)
        if not params.strip():
            return match.group(0)

        # Split parameters and add type annotations where missing
        param_list = [p.strip() for p in params.split(",")]
        fixed_params = []

        for param in param_list:
            param = param.strip()
            if not param:
                continue

            # Skip if already has type annotation or is *args/**kwargs
            if ":" in param or param.startswith("*") or param == "self":
                fixed_params.append(param)
            elif "=" in param:
                # Has default value but no type annotation
                name, default = param.split("=", 1)
                fixed_params.append(f"{name.strip()}: Any = {default.strip()}")
            else:
                # Simple parameter without type or default
                fixed_params.append(f"{param}: Any")

        return f"def {match.group(0).split('(')[0].split('def ')[1]}({', '.join(fixed_params)})"

    content = re.sub(pattern, fix_params, content)

    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True

    return False


def main():
    """Fix type errors in all Python files."""
    src_dir = Path("src")
    fixed_count = 0

    for py_file in src_dir.rglob("*.py"):
        if fix_generic_type_parameters(py_file):
            print(f"Fixed generic types: {py_file}")
            fixed_count += 1

        if fix_function_annotations(py_file):
            print(f"Fixed function annotations: {py_file}")
            fixed_count += 1

    print(f"\nTotal fixes applied: {fixed_count}")


if __name__ == "__main__":
    main()
