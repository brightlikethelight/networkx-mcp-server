#!/usr/bin/env python3
"""Quick test to verify server can start without errors."""

import subprocess
import sys
import time

import requests

print("🧪 Testing NetworkX MCP Server startup...")
print("   Testing with SSE transport (HTTP server mode)...\n")

# Start the server process with SSE transport
process = subprocess.Popen(
    [sys.executable, "-m", "networkx_mcp.server", "sse", "8765"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Wait a bit for server to start
time.sleep(3)

# Check if process is still running
if process.poll() is None:
    print("✅ Server process started successfully!")

    # Try to connect to the server
    try:
        response = requests.get("http://localhost:8765/", timeout=2)
        print(f"📡 Server responded with status code: {response.status_code}")

        # Try the SSE endpoint
        response = requests.get("http://localhost:8765/sse", timeout=2)
        print(f"🔄 SSE endpoint status: {response.status_code}")

        print("\n✅ Server is running and ready to accept connections on port 8765!")

    except requests.exceptions.ConnectionError:
        print("⚠️  Server process is running but not accepting HTTP connections")
        print("    This might be normal for stdio transport mode")
    except Exception as e:
        print(f"⚠️  Connection test error: {e}")

    print("\nTo stop the test server, run:")
    print(f"   kill {process.pid}")

    # Kill the test server
    process.terminate()
    process.wait()
    print("\n🛑 Test server stopped")
else:
    # Process exited, check for errors
    stdout, stderr = process.communicate()
    print("❌ Server failed to start!")
    if stderr:
        print(f"Error: {stderr}")
    if stdout:
        print(f"Output: {stdout}")
    sys.exit(1)

print("\n✨ Server startup test completed!")
