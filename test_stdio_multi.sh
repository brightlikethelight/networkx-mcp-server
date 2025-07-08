#!/bin/bash
# Test multiple JSON-RPC messages

echo "Testing multiple sequential messages..."

# Send multiple messages
(
echo '{"jsonrpc":"2.0","id":"init","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'
sleep 0.1
echo '{"jsonrpc":"2.0","id":"list","method":"tools/list"}'
sleep 0.1
echo '{"jsonrpc":"2.0","id":"create","method":"tools/call","params":{"name":"create_graph","arguments":{"name":"test_graph","graph_type":"undirected"}}}'
sleep 0.1
echo '{"jsonrpc":"2.0","id":"info","method":"tools/call","params":{"name":"graph_info","arguments":{"graph_name":"test_graph"}}}'
) | python -m networkx_mcp --jsonrpc 2>&1 | grep "^{" | while read line; do
    echo "Response: $(echo "$line" | jq -c '{id: .id, hasResult: (.result != null), hasError: (.error != null)}')"
done