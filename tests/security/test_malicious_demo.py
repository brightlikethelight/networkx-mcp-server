#!/usr/bin/env python3
"""Demonstrate security validation against malicious inputs."""

import sys
sys.path.insert(0, '/Users/brightliu/Coding_Projects/networkx-mcp-server')

from src.networkx_mcp.security.input_validation import validate_id, ValidationError


def test_malicious_inputs():
    """Test various malicious inputs to demonstrate security."""
    
    print("=== NetworkX MCP Server Security Validation Demo ===\n")
    
    # Test cases with malicious inputs
    malicious_inputs = [
        ("../../../etc/passwd", "Path traversal attempt"),
        ("../../config/secrets", "Path traversal to config"),
        ("'; DROP TABLE graphs;--", "SQL injection attempt"),
        ("admin'--", "SQL comment injection"),
        ("graph; rm -rf /", "Command injection"),
        ("$(whoami)", "Command substitution"),
        ("<script>alert('xss')</script>", "XSS attempt"),
        ("graph\x00admin", "Null byte injection"),
        ("node\r\nSet-Cookie: admin=true", "CRLF injection"),
        ("a" * 101, "Buffer overflow attempt"),
    ]
    
    print("Testing malicious inputs:")
    print("-" * 60)
    
    for malicious_input, description in malicious_inputs:
        try:
            # Attempt to validate the malicious input
            result = validate_id(malicious_input, "test_field")
            print(f"❌ FAILED TO BLOCK: {description}")
            print(f"   Input: {repr(malicious_input)}")
            print(f"   Result: {result}")
        except ValidationError as e:
            print(f"✅ BLOCKED: {description}")
            print(f"   Input: {repr(malicious_input)}")
            print(f"   Error: {e}")
        except Exception as e:
            print(f"⚠️  UNEXPECTED ERROR: {description}")
            print(f"   Input: {repr(malicious_input)}")
            print(f"   Error: {type(e).__name__}: {e}")
        print()
    
    # Test valid inputs
    print("\nTesting valid inputs:")
    print("-" * 60)
    
    valid_inputs = [
        ("graph1", "Simple alphanumeric"),
        ("node_123", "With underscore"),
        ("edge-456", "With hyphen"),
        ("Valid_ID-789", "Mixed valid characters"),
        ("123", "Numeric string"),
        (456, "Integer (will be converted)"),
    ]
    
    for valid_input, description in valid_inputs:
        try:
            result = validate_id(valid_input, "test_field")
            print(f"✅ ACCEPTED: {description}")
            print(f"   Input: {repr(valid_input)}")
            print(f"   Result: {result}")
        except Exception as e:
            print(f"❌ INCORRECTLY BLOCKED: {description}")
            print(f"   Input: {repr(valid_input)}")
            print(f"   Error: {e}")
        print()


if __name__ == "__main__":
    test_malicious_inputs()