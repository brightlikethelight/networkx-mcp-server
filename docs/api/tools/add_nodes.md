# add_nodes

**Category:** Core Operations

## Description

Add nodes to a graph with support for bulk operations and attributes.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph to add nodes to |
| `nodes` | `Union[List[str], List[Dict[str, Any]]]` | Yes | `-` | Either: |
| `node_attributes` | `Optional[Dict[str, Any]]` | No | `-` | Optional default attributes to apply to all nodes |

## Returns

**Type:** `Dict[str, Any]`

Operation status including: - nodes_added: Number of nodes added - total_nodes: Total nodes in graph after operation - nodes_with_attributes: Number of nodes that have attributes - execution_time_ms: Time taken for the operation

## Source

Located in: `src/networkx_mcp/server.py:293`
