#!/usr/bin/env python3
"""Generate comprehensive API documentation for all MCP tools."""

import ast
import sys
from pathlib import Path
from typing import Any

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class MCPToolExtractor:
    """Extract MCP tool information from Python source code."""

    def __init__(self):
        self.tools = []

    def extract_tools_from_file(self, file_path: Path) -> list[dict[str, Any]]:
        """Extract all MCP tools from a Python file."""
        tools = []

        try:
            with open(file_path, encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.AsyncFunctionDef):
                    if self._has_mcp_tool_decorator(node):
                        tool_info = self._extract_tool_info(node, source)
                        if tool_info:
                            tools.append(tool_info)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

        return tools

    def _has_mcp_tool_decorator(self, node: ast.AsyncFunctionDef) -> bool:
        """Check if function has @mcp.tool() decorator."""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                if (
                    isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "tool"
                ):
                    return True
            elif (
                isinstance(decorator, ast.Attribute)
                and decorator.func.id == "mcp"
                and decorator.attr == "tool"
            ):
                return True
        return False

    def _extract_tool_info(
        self, node: ast.AsyncFunctionDef, source: str
    ) -> dict[str, Any] | None:
        """Extract comprehensive information about an MCP tool."""
        try:
            # Get function name
            name = node.name

            # Extract docstring
            docstring = ast.get_docstring(node) or ""

            # Parse docstring for structured information
            description, params_doc, returns_doc, examples = self._parse_docstring(
                docstring
            )

            # Extract function signature
            parameters = self._extract_parameters(node)

            # Extract return type annotation
            return_type = self._extract_return_type(node)

            # Categorize tool
            category = self._categorize_tool(name, description)

            # Extract error handling information
            error_codes = self._extract_error_codes(node, source)

            return {
                "name": name,
                "description": description,
                "category": category,
                "parameters": parameters,
                "params_doc": params_doc,
                "returns": return_type,
                "returns_doc": returns_doc,
                "examples": examples,
                "error_codes": error_codes,
                "source_location": f"src/networkx_mcp/server.py:{node.lineno}",
            }

        except Exception as e:
            print(f"Error extracting tool info for {node.name}: {e}")
            return None

    def _parse_docstring(
        self, docstring: str
    ) -> tuple[str, dict[str, str], str, list[str]]:
        """Parse docstring into structured components."""
        lines = docstring.strip().split("\n")

        # Extract main description (everything before Args:)
        description_lines = []
        i = 0
        while i < len(lines) and not lines[i].strip().startswith(
            ("Args:", "Parameters:")
        ):
            description_lines.append(lines[i].strip())
            i += 1

        description = " ".join(description_lines).strip()

        # Extract parameters documentation
        params_doc = {}
        returns_doc = ""
        examples = []

        current_section = None
        current_param = None

        while i < len(lines):
            line = lines[i].strip()

            if line.startswith(("Args:", "Parameters:")):
                current_section = "params"
            elif line.startswith("Returns:"):
                current_section = "returns"
            elif line.startswith(("Examples:", "Example:")):
                current_section = "examples"
            elif line and current_section == "params":
                # Parse parameter documentation
                if ":" in line:
                    param_name = line.split(":")[0].strip()
                    param_desc = ":".join(line.split(":")[1:]).strip()
                    params_doc[param_name] = param_desc
                    current_param = param_name
                elif current_param and line.startswith((" ", "\t")):
                    # Continuation of previous parameter description
                    params_doc[current_param] += " " + line.strip()
            elif line and current_section == "returns":
                returns_doc += line + " "
            elif line and current_section == "examples":
                examples.append(line)

            i += 1

        return description, params_doc, returns_doc.strip(), examples

    def _extract_parameters(self, node: ast.AsyncFunctionDef) -> list[dict[str, Any]]:
        """Extract function parameters with types and defaults."""
        parameters = []

        args = node.args
        defaults = args.defaults

        # Calculate default values
        num_defaults = len(defaults)
        num_args = len(args.args)

        for i, arg in enumerate(args.args):
            param_info = {
                "name": arg.arg,
                "type": self._extract_type_annotation(arg.annotation),
                "required": i < (num_args - num_defaults),
                "default": None,
            }

            # Add default value if available
            if i >= (num_args - num_defaults):
                default_index = i - (num_args - num_defaults)
                if default_index < len(defaults):
                    param_info["default"] = self._extract_default_value(
                        defaults[default_index]
                    )

            parameters.append(param_info)

        return parameters

    def _extract_type_annotation(self, annotation) -> str:
        """Extract type annotation as string."""
        if annotation is None:
            return "Any"

        try:
            return ast.unparse(annotation)
        except Exception:
            return "Any"

    def _extract_default_value(self, default_node) -> Any:
        """Extract default value from AST node."""
        try:
            if isinstance(default_node, ast.Constant):
                return default_node.value
            elif isinstance(default_node, ast.Name):
                return default_node.id
            else:
                return ast.unparse(default_node)
        except Exception:
            return None

    def _extract_return_type(self, node: ast.AsyncFunctionDef) -> str:
        """Extract return type annotation."""
        if node.returns:
            try:
                return ast.unparse(node.returns)
            except Exception:
                return "Any"
        return "Any"

    def _categorize_tool(self, name: str, description: str) -> str:
        """Categorize tool based on name and description."""
        name_lower = name.lower()
        description.lower()

        # Core operations
        if any(
            word in name_lower
            for word in ["create", "add", "delete", "clear", "get", "list"]
        ):
            return "Core Operations"

        # Algorithms
        elif any(
            word in name_lower
            for word in ["centrality", "shortest", "path", "cluster", "component"]
        ):
            return "Graph Algorithms"

        # Advanced analytics
        elif any(
            word in name_lower
            for word in ["community", "flow", "bipartite", "robustness", "generate"]
        ):
            return "Advanced Analytics"

        # Visualization
        elif any(
            word in name_lower
            for word in ["visualize", "plot", "interactive", "specialized"]
        ):
            return "Visualization"

        # Data integration
        elif any(
            word in name_lower for word in ["import", "export", "batch", "pipeline"]
        ):
            return "Data Integration"

        # Enterprise features
        elif any(
            word in name_lower
            for word in ["audit", "backup", "permission", "enterprise"]
        ):
            return "Enterprise Features"

        else:
            return "Utilities"

    def _extract_error_codes(
        self, node: ast.AsyncFunctionDef, source: str
    ) -> list[dict[str, str]]:
        """Extract error codes and descriptions from function source."""
        error_codes = []

        # Common patterns for error handling
        common_errors = [
            {
                "code": "GRAPH_NOT_FOUND",
                "description": "The specified graph does not exist",
            },
            {
                "code": "INVALID_PARAMETER",
                "description": "One or more parameters are invalid",
            },
            {
                "code": "OPERATION_FAILED",
                "description": "The operation could not be completed",
            },
            {
                "code": "INSUFFICIENT_DATA",
                "description": "Not enough data to perform the operation",
            },
            {"code": "MEMORY_LIMIT", "description": "Operation exceeds memory limits"},
        ]

        # Extract function source
        try:
            func_start = node.lineno
            func_end = (
                node.end_lineno if hasattr(node, "end_lineno") else func_start + 50
            )

            source_lines = source.split("\n")
            func_source = "\n".join(source_lines[func_start - 1 : func_end])

            # Look for specific error patterns
            for error in common_errors:
                if error["code"].lower() in func_source.lower():
                    error_codes.append(error)

        except Exception:
            pass

        return error_codes[:3]  # Limit to 3 most relevant errors


class DocumentationGenerator:
    """Generate markdown documentation from extracted tool information."""

    def __init__(self):
        self.tools = []

    def generate_tool_documentation(self, tool: dict[str, Any]) -> str:
        """Generate markdown documentation for a single tool."""
        md = []

        # Header
        md.append(f"# {tool['name']}")
        md.append("")
        md.append(f"**Category:** {tool['category']}")
        md.append("")

        # Description
        md.append("## Description")
        md.append("")
        md.append(tool["description"])
        md.append("")

        # Parameters
        md.append("## Parameters")
        md.append("")

        if tool["parameters"]:
            md.append("| Name | Type | Required | Default | Description |")
            md.append("|------|------|----------|---------|-------------|")

            for param in tool["parameters"]:
                name = param["name"]
                param_type = param["type"]
                required = "Yes" if param["required"] else "No"
                default = str(param["default"]) if param["default"] is not None else "-"
                description = tool["params_doc"].get(name, "No description available")

                md.append(
                    f"| `{name}` | `{param_type}` | {required} | `{default}` | {description} |"
                )
        else:
            md.append("No parameters required.")

        md.append("")

        # Returns
        md.append("## Returns")
        md.append("")
        md.append(f"**Type:** `{tool['returns']}`")
        md.append("")
        if tool["returns_doc"]:
            md.append(tool["returns_doc"])
        else:
            md.append("Returns operation result with status and metadata.")
        md.append("")

        # Examples
        if tool["examples"]:
            md.append("## Examples")
            md.append("")
            for example in tool["examples"]:
                md.append("```python")
                md.append(example)
                md.append("```")
            md.append("")

        # Error Codes
        if tool["error_codes"]:
            md.append("## Error Codes")
            md.append("")
            md.append("| Code | Description |")
            md.append("|------|-------------|")
            for error in tool["error_codes"]:
                md.append(f"| `{error['code']}` | {error['description']} |")
            md.append("")

        # Source location
        md.append("## Source")
        md.append("")
        md.append(f"Located in: `{tool['source_location']}`")
        md.append("")

        return "\n".join(md)

    def generate_api_index(self, tools: list[dict[str, Any]]) -> str:
        """Generate API index with all tools organized by category."""
        md = []

        # Header
        md.append("# NetworkX MCP Server API Reference")
        md.append("")
        md.append(
            "Complete documentation for all 39 graph analysis tools available in the NetworkX MCP Server."
        )
        md.append("")

        # Quick stats
        total_tools = len(tools)
        categories = {tool["category"] for tool in tools}

        md.append("## Overview")
        md.append("")
        md.append(f"- **Total Tools:** {total_tools}")
        md.append(f"- **Categories:** {len(categories)}")
        md.append("- **Protocol:** Model Context Protocol (MCP)")
        md.append("- **Engine:** NetworkX")
        md.append("")

        # Tools by category
        md.append("## Tools by Category")
        md.append("")

        # Group tools by category
        by_category = {}
        for tool in tools:
            category = tool["category"]
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(tool)

        # Sort categories
        category_order = [
            "Core Operations",
            "Graph Algorithms",
            "Advanced Analytics",
            "Visualization",
            "Data Integration",
            "Enterprise Features",
            "Utilities",
        ]

        for category in category_order:
            if category in by_category:
                md.append(f"### {category}")
                md.append("")

                # Sort tools alphabetically within category
                category_tools = sorted(by_category[category], key=lambda x: x["name"])

                for tool in category_tools:
                    description = (
                        tool["description"][:100] + "..."
                        if len(tool["description"]) > 100
                        else tool["description"]
                    )
                    md.append(
                        f"- [`{tool['name']}`](tools/{tool['name']}.md) - {description}"
                    )

                md.append("")

        # Quick start
        md.append("## Quick Start")
        md.append("")
        md.append("```python")
        md.append("from mcp import Client")
        md.append("import asyncio")
        md.append("")
        md.append("async def example():")
        md.append("    # Connect to server")
        md.append("    client = Client()")
        md.append("    await client.connect('localhost:8765')")
        md.append("    ")
        md.append("    # Create a graph")
        md.append("    result = await client.call_tool('create_graph', {")
        md.append("        'graph_id': 'my_graph',")
        md.append("        'graph_type': 'undirected'")
        md.append("    })")
        md.append("    ")
        md.append("    # Add some nodes")
        md.append("    await client.call_tool('add_nodes', {")
        md.append("        'graph_id': 'my_graph',")
        md.append("        'nodes': ['A', 'B', 'C', 'D']")
        md.append("    })")
        md.append("    ")
        md.append("    # Add edges")
        md.append("    await client.call_tool('add_edges', {")
        md.append("        'graph_id': 'my_graph',")
        md.append("        'edges': [['A', 'B'], ['B', 'C'], ['C', 'D']]")
        md.append("    })")
        md.append("    ")
        md.append("    # Analyze")
        md.append("    centrality = await client.call_tool('centrality_measures', {")
        md.append("        'graph_id': 'my_graph',")
        md.append("        'centrality_type': 'betweenness'")
        md.append("    })")
        md.append("    ")
        md.append("    print(centrality)")
        md.append("")
        md.append("asyncio.run(example())")
        md.append("```")
        md.append("")

        # Footer
        md.append("## Need Help?")
        md.append("")
        md.append("- [Getting Started Guide](../getting-started.md)")
        md.append("- [Examples](../examples/)")
        md.append("- [Contributing](../CONTRIBUTING.md)")
        md.append(
            "- [Issue Tracker](https://github.com/yourusername/networkx-mcp-server/issues)"
        )
        md.append("")

        return "\n".join(md)


def main():
    """Generate comprehensive API documentation."""
    print("🔍 Extracting MCP tools from server.py...")

    # Extract tools from server.py
    extractor = MCPToolExtractor()
    server_file = Path("src/networkx_mcp/server.py")

    if not server_file.exists():
        print(f"❌ Server file not found: {server_file}")
        return False

    tools = extractor.extract_tools_from_file(server_file)

    if not tools:
        print("❌ No MCP tools found in server.py")
        return False

    print(f"✅ Found {len(tools)} MCP tools")

    # Create documentation directory
    docs_dir = Path("docs")
    api_dir = docs_dir / "api"
    tools_dir = api_dir / "tools"

    docs_dir.mkdir(exist_ok=True)
    api_dir.mkdir(exist_ok=True)
    tools_dir.mkdir(exist_ok=True)

    # Generate documentation
    generator = DocumentationGenerator()

    print("📝 Generating individual tool documentation...")
    for tool in tools:
        doc_content = generator.generate_tool_documentation(tool)
        doc_file = tools_dir / f"{tool['name']}.md"
        doc_file.write_text(doc_content, encoding="utf-8")
        print(f"  ✅ {tool['name']}.md")

    # Generate API index
    print("📋 Generating API index...")
    index_content = generator.generate_api_index(tools)
    index_file = api_dir / "README.md"
    index_file.write_text(index_content, encoding="utf-8")

    print("\n📊 Documentation Summary:")
    print(f"  Tools documented: {len(tools)}")
    print(f"  Categories: {len({tool['category'] for tool in tools})}")
    print(f"  Files created: {len(tools) + 1}")
    print(f"  Output directory: {api_dir}")

    # Create categories summary
    categories = {}
    for tool in tools:
        cat = tool["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(tool["name"])

    print("\n📈 Tools by Category:")
    for cat, tool_names in sorted(categories.items()):
        print(f"  {cat}: {len(tool_names)} tools")
        for name in sorted(tool_names)[:3]:  # Show first 3
            print(f"    - {name}")
        if len(tool_names) > 3:
            print(f"    - ... and {len(tool_names) - 3} more")

    print("\n🎉 API documentation generation complete!")
    print("📖 View the index at: docs/api/README.md")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
