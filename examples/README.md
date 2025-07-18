# NetworkX MCP Server Examples

This directory contains example code for using the NetworkX MCP Server.

## Examples

### docker_client.py

A Python client that demonstrates how to communicate with the MCP server running in a Docker container.

**Features:**

- Starts the server in a Docker container
- Sends JSON-RPC requests via stdin/stdout
- Demonstrates all major operations:
  - Server initialization
  - Creating graphs
  - Adding nodes and edges
  - Finding shortest paths
  - Getting graph information

**Usage:**

```bash
# Make sure Docker is running and the image is built
docker build -t networkx-mcp:0.1.0 ..

# Run the example
python docker_client.py
```

## Writing Your Own Client

To create your own MCP client:

1. **Start the container** with stdin enabled:

   ```python
   subprocess.Popen(['docker', 'run', '-i', '--rm', 'networkx-mcp:0.1.0'], ...)
   ```

2. **Send JSON-RPC messages** as line-delimited JSON:

   ```python
   request = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {...}}
   stdin.write(json.dumps(request) + '\n')
   ```

3. **Read responses** from stdout:

   ```python
   response = json.loads(stdout.readline())
   ```

4. **Follow the MCP protocol**:
   - Always send `initialize` first
   - Send `initialized` notification after receiving initialize response
   - Only then can you use tools via `tools/call`

## Protocol Flow

```
Client                          Server
  |                               |
  |-- initialize request -------->|
  |<-- initialize response -------|
  |                               |
  |-- initialized notification -->|
  |                               |
  |-- tools/list request -------->|
  |<-- tools/list response -------|
  |                               |
  |-- tools/call request -------->|
  |<-- tools/call response -------|
  |                               |
```
