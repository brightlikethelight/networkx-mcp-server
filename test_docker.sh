#!/bin/bash
# test_docker.sh - Test NetworkX MCP Server in Docker container

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Testing NetworkX MCP Server in Docker...${NC}"
echo

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}✗ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Build image
echo "Building Docker image..."
if docker build -t networkx-mcp:test . > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Docker image built successfully${NC}"
else
    echo -e "${RED}✗ Docker build failed${NC}"
    exit 1
fi

echo
echo "Running MCP protocol tests..."

# Test 1: Initialize request
echo -n "Test 1 - Initialize: "
RESPONSE=$(echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}' | \
  docker run -i --rm networkx-mcp:test 2>/dev/null || echo "ERROR")

if echo "$RESPONSE" | grep -q '"protocolVersion"'; then
    echo -e "${GREEN}✓ Initialize works${NC}"
else
    echo -e "${RED}✗ Initialize failed${NC}"
    echo "Response: $RESPONSE"
fi

# Test 2: Send initialized notification (no response expected)
echo -n "Test 2 - Initialized notification: "
RESPONSE=$(echo '{"jsonrpc":"2.0","method":"initialized"}' | \
  docker run -i --rm networkx-mcp:test 2>/dev/null || echo "")

if [ -z "$RESPONSE" ]; then
    echo -e "${GREEN}✓ Initialized notification handled${NC}"
else
    echo -e "${RED}✗ Unexpected response to notification${NC}"
    echo "Response: $RESPONSE"
fi

# Test 3: Tools list (requires proper initialization sequence)
echo -n "Test 3 - Tools list: "
RESPONSE=$(printf '%s\n%s\n%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}' \
  '{"jsonrpc":"2.0","method":"initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' | \
  docker run -i --rm networkx-mcp:test 2>/dev/null | tail -n 1 || echo "ERROR")

if echo "$RESPONSE" | grep -q '"tools"'; then
    echo -e "${GREEN}✓ Tool list works${NC}"
else
    echo -e "${RED}✗ Tool list failed${NC}"
    echo "Response: $RESPONSE"
fi

# Test 4: Create graph tool
echo -n "Test 4 - Create graph: "
RESPONSE=$(printf '%s\n%s\n%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}' \
  '{"jsonrpc":"2.0","method":"initialized"}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"create_graph","arguments":{"graph_id":"test_graph"}}}' | \
  docker run -i --rm networkx-mcp:test 2>/dev/null | tail -n 1 || echo "ERROR")

if echo "$RESPONSE" | grep -q '"result"'; then
    echo -e "${GREEN}✓ Create graph works${NC}"
else
    echo -e "${RED}✗ Create graph failed${NC}"
    echo "Response: $RESPONSE"
fi

# Test 5: Error handling - Invalid JSON
echo -n "Test 5 - Invalid JSON handling: "
RESPONSE=$(echo 'invalid json' | \
  docker run -i --rm networkx-mcp:test 2>&1 | grep -i "error" > /dev/null && echo "OK" || echo "FAIL")

if [ "$RESPONSE" = "OK" ]; then
    echo -e "${GREEN}✓ Invalid JSON handled properly${NC}"
else
    echo -e "${RED}✗ Invalid JSON not handled${NC}"
fi

# Test 6: Multi-platform build check
echo -n "Test 6 - Check image info: "
docker inspect networkx-mcp:test --format='{{.Architecture}}' > /dev/null 2>&1 && \
    echo -e "${GREEN}✓ Image metadata accessible${NC}" || \
    echo -e "${RED}✗ Cannot inspect image${NC}"

echo
echo -e "${YELLOW}Docker tests complete!${NC}"

# Optional: Clean up test image
echo
read -p "Remove test image? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker rmi networkx-mcp:test > /dev/null 2>&1
    echo "Test image removed."
fi