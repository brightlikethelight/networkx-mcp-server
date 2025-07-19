#!/usr/bin/env python3
"""Simple brutal reality test with timeout protection to prevent CI hangs."""

import json
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Optional


class TimeoutException(Exception):
    """Raised when operation times out."""

    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Operation timed out")


def test_server_reality_with_timeout():
    """Test the actual server functionality with timeout protection."""
    print("🔥 BRUTAL REALITY CHECK (TIMEOUT PROTECTED) 🔥")

    # Set overall test timeout to 30 seconds
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)

    process = None
    try:
        # Start server with timeout
        print("Starting server...")
        process = subprocess.Popen(
            [sys.executable, "-m", "networkx_mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Give server time to start
        time.sleep(2)

        if process.poll() is not None:
            # Server exited immediately
            stdout, stderr = process.communicate()
            print(f"❌ Server exited immediately with code {process.returncode}")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False

        def send_request_with_timeout(
            method: str, params: Optional[Dict[str, Any]] = None, timeout: float = 5.0
        ) -> Optional[Dict[str, Any]]:
            """Send request with timeout protection."""
            request = {"jsonrpc": "2.0", "method": method, "id": 1}
            if params:
                request["params"] = params

            try:
                request_str = json.dumps(request) + "\n"
                process.stdin.write(request_str)
                process.stdin.flush()

                # Set a timeout for reading response
                start_time = time.time()
                while time.time() - start_time < timeout:
                    if process.poll() is not None:
                        print(f"❌ Server died during {method} request")
                        return {"error": "Server died"}

                    # Try to read with small timeout
                    import select

                    ready, _, _ = select.select([process.stdout], [], [], 0.1)
                    if ready:
                        response_line = process.stdout.readline()
                        if response_line:
                            return json.loads(response_line.strip())

                print(f"⚠️ Timeout waiting for {method} response")
                return {"error": f"Timeout waiting for {method}"}

            except Exception as e:
                print(f"❌ Error in {method}: {e}")
                return {"error": str(e)}

        # Test basic functionality
        print("\n=== INITIALIZATION ===")
        init_response = send_request_with_timeout(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "brutal-test", "version": "1.0.0"},
            },
            timeout=10.0,
        )

        if init_response and "result" in init_response:
            print("✅ Server initialization successful")
        else:
            print(f"❌ Initialization failed: {init_response}")
            return False

        print("\n=== TOOLS LIST ===")
        tools_response = send_request_with_timeout("tools/list", timeout=5.0)
        if tools_response and "result" in tools_response:
            tools = tools_response["result"]["tools"]
            print(f"✅ Found {len(tools)} tools")
            return True
        else:
            print(f"❌ Failed to get tools list: {tools_response}")
            return False

    except TimeoutException:
        print("❌ Overall test timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"💥 CRITICAL ERROR: {e}")
        return False
    finally:
        # Clean up
        signal.alarm(0)  # Cancel timeout
        if process:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                process.kill()


def main():
    """Main entry point that exits with proper code for CI."""
    success = test_server_reality_with_timeout()
    if success:
        print("\n✅ Reality test PASSED")
        sys.exit(0)
    else:
        print("\n❌ Reality test FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
