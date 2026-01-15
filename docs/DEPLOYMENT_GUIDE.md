# NetworkX MCP Server v2.0.0 Deployment Guide

## Overview

This guide covers deploying the enhanced NetworkX MCP Server with complete MCP specification support (Tools, Resources, and Prompts).

## Deployment Options

### 1. Local Development

```bash
# Clone the repository
git clone https://github.com/Bright-L01/networkx-mcp-server.git
cd networkx-mcp-server

# Install in development mode
pip install -e ".[dev]"

# Run the new modular server
python -m networkx_mcp.server_v2

# Or run with MCP desktop/CLI
mcp run networkx-mcp-server
```

### 2. Production Installation

```bash
# Install from PyPI (once published)
pip install networkx-mcp-server==2.0.0

# Run the server
networkx-mcp-server

# Or use the new modular entry point
python -m networkx_mcp.server_v2
```

### 3. Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY pyproject.toml .

# Install the package
RUN pip install .

# Run the server
CMD ["python", "-m", "networkx_mcp.server_v2"]
```

Build and run:

```bash
docker build -t networkx-mcp-server:2.0.0 .
docker run -p 3000:3000 networkx-mcp-server:2.0.0
```

### 4. Cloud Deployment

#### AWS Lambda

```python
# lambda_handler.py
from networkx_mcp.server_v2 import NetworkXMCPServer

def lambda_handler(event, context):
    server = NetworkXMCPServer()
    # Process MCP request
    return server.handle_request(event)
```

#### Google Cloud Functions

```python
# main.py
from networkx_mcp.server_v2 import NetworkXMCPServer

server = NetworkXMCPServer()

def handle_mcp_request(request):
    return server.handle_request(request.get_json())
```

## Configuration

### Environment Variables

```bash
# .env file
MCP_SERVER_NAME="NetworkX Graph Analysis Server"
LOG_LEVEL=INFO
REDIS_URL=redis://localhost:6379
MAX_GRAPH_SIZE=1000000
ENABLE_CACHING=true
```

### Server Configuration

```python
# config.py
from networkx_mcp.server_v2 import NetworkXMCPServer

# Custom configuration
server = NetworkXMCPServer(
    name="My Graph Analysis Server",
    max_graphs=100,
    enable_persistence=True,
    cache_results=True
)
```

## Integration Examples

### With Claude Desktop

1. Add to Claude Desktop configuration:

```json
{
  "mcpServers": {
    "networkx": {
      "command": "python",
      "args": ["-m", "networkx_mcp.server_v2"],
      "env": {
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### With LangChain

```python
from langchain.tools import MCPTool
from networkx_mcp.server_v2 import NetworkXMCPServer

# Initialize server
server = NetworkXMCPServer()

# Create LangChain tools
tools = [
    MCPTool(name=tool_name, mcp_server=server)
    for tool_name in server.list_tools()
]
```

### With Custom Applications

```python
import asyncio
from networkx_mcp.server_v2 import NetworkXMCPServer

async def main():
    server = NetworkXMCPServer()

    # Create a graph
    result = await server.invoke_tool(
        "create_graph",
        {"graph_id": "my_graph", "graph_type": "directed"}
    )

    # Add nodes and edges
    await server.invoke_tool(
        "add_nodes",
        {"graph_id": "my_graph", "nodes": ["A", "B", "C"]}
    )

    # Use resources
    catalog = await server.get_resource("graph://catalog")
    print(catalog)

asyncio.run(main())
```

## Monitoring

### Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('networkx_mcp.log'),
        logging.StreamHandler()
    ]
)
```

### Metrics

The server exposes metrics for monitoring:

- Tool invocation counts
- Response times
- Error rates
- Graph statistics

### Health Checks

```bash
# Check server health
curl http://localhost:3000/health

# Expected response
{"status": "healthy", "version": "2.0.0", "uptime": 3600}
```

## Troubleshooting

### Common Issues

1. **Import Errors**

   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   ```

2. **Memory Issues**
   - Limit graph sizes with MAX_GRAPH_SIZE
   - Enable Redis persistence for large graphs
   - Use batch processing for large operations

3. **Performance**
   - Enable caching with ENABLE_CACHING=true
   - Use appropriate algorithms for graph size
   - Consider using PyPy for better performance

### Debug Mode

```bash
# Enable debug logging
LOG_LEVEL=DEBUG python -m networkx_mcp.server_v2

# Or in code
import logging
logging.getLogger('networkx_mcp').setLevel(logging.DEBUG)
```

## Security Considerations

1. **Input Validation**: All inputs are validated
2. **Resource Limits**: Configurable limits on graph sizes
3. **Authentication**: Integrate with your auth system
4. **Sandboxing**: Run in isolated environments

## Migration from v1.0.0

If upgrading from v1.0.0:

1. Review MIGRATION_NOTES.md
2. Test with existing graphs
3. Update tool names if changed
4. Leverage new Resources and Prompts

## Support

- GitHub Issues: <https://github.com/Bright-L01/networkx-mcp-server/issues>
- Documentation: <https://networkx-mcp-server.readthedocs.io>
- Examples: See `/examples` directory

## Next Steps

1. Explore the new MCP Resources for data access
2. Try the workflow Prompts for common tasks
3. Build custom handlers for your use cases
4. Contribute improvements back to the project

---

Happy graphing! 🎯
