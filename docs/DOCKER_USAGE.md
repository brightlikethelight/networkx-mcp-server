# Docker Usage for NetworkX MCP Server

This document describes how to build, run, and test the NetworkX MCP Server using Docker.

## Prerequisites

- Docker Engine 20.10 or later
- Docker Compose v2.0 or later (optional)

## Building the Image

### Quick Build
```bash
docker build -t networkx-mcp:0.1.0 .
```

### Multi-platform Build (for Apple Silicon and Intel)
```bash
docker buildx build --platform linux/amd64,linux/arm64 -t networkx-mcp:0.1.0 .
```

## Running the Container

### Basic Usage (stdio mode)

The server operates in stdio mode, reading JSON-RPC messages from stdin and writing responses to stdout.

```bash
# Interactive mode
docker run -it networkx-mcp:0.1.0

# With JSON-RPC input
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | \
  docker run -i networkx-mcp:0.1.0
```

### Using Docker Compose

```bash
# Start the service
docker-compose up -d

# Send commands to the running container
docker exec -i networkx-mcp-server python -m networkx_mcp.server
```

## Testing the Container

### Automated Testing

Run the included test script:
```bash
./test_docker.sh
```

This script will:
1. Build the Docker image
2. Run protocol compliance tests
3. Verify JSON-RPC communication
4. Test error handling

### Manual Testing

1. **Initialize the server:**
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}' | \
  docker run -i --rm networkx-mcp:0.1.0
```

2. **Send initialized notification:**
```bash
echo '{"jsonrpc":"2.0","method":"initialized"}' | \
  docker run -i --rm networkx-mcp:0.1.0
```

3. **List available tools:**
```bash
printf '%s\n%s\n%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}' \
  '{"jsonrpc":"2.0","method":"initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' | \
  docker run -i --rm networkx-mcp:0.1.0
```

## Container Configuration

### Environment Variables

- `LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `PYTHONUNBUFFERED`: Set to 1 for unbuffered output (default)
- `DOCKER_CONTAINER`: Set to true to indicate container environment

### Resource Limits

The docker-compose.yml includes resource limits:
- CPU: 1 core max, 0.25 core reserved
- Memory: 512MB max, 128MB reserved

## Current Limitations

### Stdio Mode Only
- The server only supports stdio transport
- No HTTP or WebSocket support
- Single connection at a time

### No Persistent Storage
- Graphs are stored in memory only
- Data is lost when container stops
- No volume mounts configured

### No Claude Desktop Integration
- Stdio mode requires process management
- Claude Desktop integration not yet supported
- Manual JSON-RPC communication only

## Security Considerations

1. **Non-root User**: Container runs as unprivileged `mcp` user
2. **Minimal Base Image**: Uses `python:3.11-slim` for smaller attack surface
3. **No Network Exposure**: No ports exposed by default
4. **Resource Limits**: CPU and memory limits prevent resource exhaustion

## Troubleshooting

### Container won't start
```bash
# Check logs
docker logs networkx-mcp-server

# Verify image was built
docker images | grep networkx-mcp
```

### No response to commands
- Ensure JSON is properly formatted
- Include newline after each JSON message
- Check container is still running: `docker ps`

### Permission denied errors
```bash
# Ensure script is executable
chmod +x test_docker.sh

# Check Docker daemon is running
docker info
```

## Future Enhancements

1. **Persistent Storage**: Add volume mounts for graph persistence
2. **HTTP Transport**: Implement HTTP/SSE transport for remote access
3. **Health Checks**: Add proper health check endpoints
4. **Multi-stage Build**: Optimize image size with multi-stage builds
5. **Alpine Base**: Consider Alpine Linux for even smaller images

## Example Integration

### With Python Client
```python
import subprocess
import json

# Start container
proc = subprocess.Popen(
    ['docker', 'run', '-i', '--rm', 'networkx-mcp:0.1.0'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Send initialize
request = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1.0.0"}
    }
}

proc.stdin.write(json.dumps(request) + '\n')
proc.stdin.flush()

# Read response
response = proc.stdout.readline()
print(json.loads(response))
```

### With Node.js Client
```javascript
const { spawn } = require('child_process');

const mcp = spawn('docker', ['run', '-i', '--rm', 'networkx-mcp:0.1.0']);

mcp.stdin.write(JSON.stringify({
  jsonrpc: "2.0",
  id: 1,
  method: "initialize",
  params: {}
}) + '\n');

mcp.stdout.on('data', (data) => {
  console.log(JSON.parse(data.toString()));
});
```