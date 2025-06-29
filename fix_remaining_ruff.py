#!/usr/bin/env python3
"""Fix remaining ruff linting errors."""

import subprocess
import re
from pathlib import Path

def fix_magic_values():
    """Fix PLR2004 magic value errors."""
    files_to_fix = {
        "src/networkx_mcp/advanced/ml_integration.py": [
            (553, "            elif subgraph.number_of_nodes() == 2:", "            TWO_NODES = 2\n            elif subgraph.number_of_nodes() == TWO_NODES:"),
            (555, "            elif subgraph.number_of_nodes() == 3:", "            THREE_NODES = 3\n            elif subgraph.number_of_nodes() == THREE_NODES:"),
        ],
        "src/networkx_mcp/advanced/network_flow.py": [
            (75, "        if num_edges < 1000:", "        SMALL_GRAPH_EDGES = 1000\n        if num_edges < SMALL_GRAPH_EDGES:"),
            (77, "        elif num_edges > 10000:", "        elif num_edges > 10000:  # noqa: PLR2004"),
            (845, "            if abs(total_flow - max_flow) > 1e-10:", "            FLOW_TOLERANCE = 1e-10\n            if abs(total_flow - max_flow) > FLOW_TOLERANCE:"),
            (878, '                    "fraction_of_max_flow": flow_value / max_flow if max_flow > 1e-10 else 0,', '                    "fraction_of_max_flow": flow_value / max_flow if max_flow > FLOW_TOLERANCE else 0,'),
            (948, '            "fraction_of_max_flow": path_flow / max_flow if max_flow > 1e-10 else 0', '            "fraction_of_max_flow": path_flow / max_flow if max_flow > FLOW_TOLERANCE else 0'),
        ],
        "src/networkx_mcp/advanced/robustness.py": [
            (73, "        if n_nodes > 1000:", "        MAX_NODES_FOR_EXPENSIVE_ANALYSIS = 1000\n        if n_nodes > MAX_NODES_FOR_EXPENSIVE_ANALYSIS:"),
            (210, "            if graph.number_of_nodes() > 1000:", "            if graph.number_of_nodes() > MAX_NODES_FOR_EXPENSIVE_ANALYSIS:"),
            (334, '                    "failure_probability": 0.5,  # Default probability', '                    "failure_probability": 0.5,  # Default probability  # noqa: PLR2004'),
        ],
    }
    
    for file_path, fixes in files_to_fix.items():
        if Path(file_path).exists():
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Apply fixes in reverse order to maintain line numbers
            for line_num, old_content, new_content in reversed(fixes):
                if line_num <= len(lines):
                    # Adjust for 0-based indexing
                    idx = line_num - 1
                    if old_content in lines[idx]:
                        lines[idx] = lines[idx].replace(old_content, new_content)
                    else:
                        # Try to find the line nearby
                        for offset in [-2, -1, 1, 2]:
                            if 0 <= idx + offset < len(lines) and old_content in lines[idx + offset]:
                                lines[idx + offset] = lines[idx + offset].replace(old_content, new_content)
                                break
            
            with open(file_path, 'w') as f:
                f.writelines(lines)

def fix_unused_noqa():
    """Remove unused noqa directives."""
    files_to_fix = {
        "src/networkx_mcp/advanced/enterprise.py": [
            "# noqa: S324",
        ],
        "src/networkx_mcp/advanced/ml_integration.py": [
            "  # noqa: PLR2004",
        ],
    }
    
    for file_path, patterns in files_to_fix.items():
        if Path(file_path).exists():
            with open(file_path, 'r') as f:
                content = f.read()
            
            for pattern in patterns:
                content = content.replace(pattern, "")
            
            with open(file_path, 'w') as f:
                f.write(content)

def fix_other_issues():
    """Fix remaining issues."""
    # Fix ARG004 in robustness.py
    file_path = "src/networkx_mcp/advanced/robustness.py"
    if Path(file_path).exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix unused params argument
        content = content.replace("def generator_fn(n, **params):", "def generator_fn(n, **_params):")
        content = content.replace("def failure_mode_fn(graph, **params):", "def failure_mode_fn(graph, **_params):")
        
        # Add noqa for random.choice
        content = content.replace("node = random.choice(nodes)", "node = random.choice(nodes)  # noqa: S311")
        content = content.replace("edge = random.choice(edges)", "edge = random.choice(edges)  # noqa: S311")
        
        with open(file_path, 'w') as f:
            f.write(content)
    
    # Fix PLC0206 in network_flow.py
    file_path = "src/networkx_mcp/advanced/network_flow.py"
    if Path(file_path).exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix dictionary iteration
        content = re.sub(r'for partition in partitions:\s*\n\s*nodes = partitions\[partition\]', 
                        'for partition, nodes in partitions.items():', 
                        content)
        
        with open(file_path, 'w') as f:
            f.write(content)
    
    # Fix RUF010 in ml_integration.py
    file_path = "src/networkx_mcp/advanced/ml_integration.py"
    if Path(file_path).exists():
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Fix string formatting
        content = content.replace('{"error": f"Failed to extract features: {str(e)}"}', '{"error": f"Failed to extract features: {e!s}"}')
        
        with open(file_path, 'w') as f:
            f.write(content)

if __name__ == "__main__":
    print("Fixing magic values...")
    fix_magic_values()
    
    print("Removing unused noqa directives...")
    fix_unused_noqa()
    
    print("Fixing other issues...")
    fix_other_issues()
    
    print("Done! Running ruff check to verify...")
    result = subprocess.run(["ruff", "check", "src/", "--select=PLR2004,RUF100,RUF010,PLC0206,ARG004,S311"], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("All selected ruff errors fixed!")
    else:
        print(f"Remaining errors:\n{result.stdout}")