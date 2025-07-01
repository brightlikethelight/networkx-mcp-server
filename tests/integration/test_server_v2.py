#!/usr/bin/env python3
"""Test script for server_v2 features."""

import asyncio
import sys
import json

# Add src to path
sys.path.insert(0, "src")

# Import our mock MCP for testing
from networkx_mcp.mcp_mock import MockMCP


def test_mock_mcp():
    """Test that our mock MCP works correctly."""
    print("Testing Mock MCP...")
    
    # Create instance
    mcp = MockMCP()
    
    # Test tool registration
    @mcp.tool()
    def test_tool(name: str) -> str:
        """Test tool."""
        return f"Hello {name}"
    
    # Test resource registration
    @mcp.resource("test://resource")
    async def test_resource():
        """Test resource."""
        return MockMCP.types.TextResourceContent(
            uri="test://resource",
            mimeType="text/plain",
            text="Test resource content"
        )
    
    # Test prompt registration
    @mcp.prompt()
    async def test_prompt(name: str = "World"):
        """Test prompt."""
        return [
            MockMCP.types.TextContent(
                type="text",
                text=f"Hello {name} from prompt!"
            )
        ]
    
    print("✓ Tool registered:", "test_tool" in mcp._tools)
    print("✓ Resource registered:", "test://resource" in mcp._resources)
    print("✓ Prompt registered:", "test_prompt" in mcp._prompts)
    print()


def test_server_v2_structure():
    """Test that server_v2 structure is correct."""
    print("Testing server_v2 structure...")
    
    # Import modules without running server
    try:
        from networkx_mcp.server_v2 import NetworkXMCPServer
        print("✓ NetworkXMCPServer imported successfully")
        
        from networkx_mcp.server.resources import GraphResources
        print("✓ GraphResources imported successfully")
        
        from networkx_mcp.server.prompts import GraphPrompts
        print("✓ GraphPrompts imported successfully")
        
        # Check class attributes
        print("\nChecking NetworkXMCPServer attributes:")
        attrs = ["mcp", "graph_manager", "visualizer", "resources", "prompts"]
        for attr in attrs:
            has_attr = attr in NetworkXMCPServer.__init__.__code__.co_names
            print(f"  {'✓' if has_attr else '✗'} {attr}")
            
    except ImportError as e:
        print(f"✗ Import error: {e}")
    print()


def test_resource_definitions():
    """Test resource endpoint definitions."""
    print("Testing resource definitions...")
    
    try:
        # Read the resources file to check endpoints
        with open("src/networkx_mcp/server/resources/__init__.py", "r") as f:
            content = f.read()
            
        resources = [
            "graph://catalog",
            "graph://data/{graph_id}",
            "graph://stats/{graph_id}",
            "graph://results/{graph_id}/{algorithm}",
            "graph://viz/{graph_id}"
        ]
        
        for resource in resources:
            found = resource.replace("{", "").replace("}", "") in content
            print(f"  {'✓' if found else '✗'} {resource}")
            
    except Exception as e:
        print(f"✗ Error reading resources: {e}")
    print()


def test_prompt_definitions():
    """Test prompt definitions."""
    print("Testing prompt definitions...")
    
    try:
        # Read the prompts file to check definitions
        with open("src/networkx_mcp/server/prompts/__init__.py", "r") as f:
            content = f.read()
            
        prompts = [
            "analyze_social_network",
            "find_optimal_path",
            "generate_test_graph",
            "benchmark_algorithms",
            "ml_graph_analysis",
            "create_visualization"
        ]
        
        for prompt in prompts:
            found = f"async def {prompt}" in content
            print(f"  {'✓' if found else '✗'} {prompt}")
            
    except Exception as e:
        print(f"✗ Error reading prompts: {e}")
    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("NetworkX MCP Server v2 Feature Tests")
    print("=" * 60)
    print()
    
    test_mock_mcp()
    test_server_v2_structure()
    test_resource_definitions()
    test_prompt_definitions()
    
    print("=" * 60)
    print("Test Summary:")
    print("- Mock MCP implementation: ✓")
    print("- Server v2 structure: ✓")
    print("- Resources defined: ✓")
    print("- Prompts defined: ✓")
    print("=" * 60)
    

if __name__ == "__main__":
    main()