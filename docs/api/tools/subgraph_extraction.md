# subgraph_extraction

**Category:** Utilities

## Description

Extract subgraphs based on various criteria.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the source graph |
| `method` | `str` | No | `nodes` | Extraction method - 'nodes', 'edges', 'k_hop', 'largest_component', 'condition' |
| `nodes` | `Optional[List[Union[str, int]]]` | No | `-` | Node list for 'nodes' method |
| `edges` | `Optional[List[Tuple[Union[str, int], Union[str, int]]]]` | No | `-` | Edge list for 'edges' method |
| `k_hop` | `Optional[int]` | No | `-` | Number of hops for 'k_hop' method |
| `center_node` | `Optional[Union[str, int]]` | No | `-` | Center node for 'k_hop' method |
| `condition` | `Optional[str]` | No | `-` | Node/edge attribute condition (e.g., "weight > 0.5") |
| `create_new` | `bool` | No | `True` | Whether to create a new graph or return subgraph data |
| `new_graph_id` | `Optional[str]` | No | `-` | ID for the new graph (auto-generated if not provided) |

## Returns

**Type:** `Dict[str, Any]`

Subgraph information including: - subgraph_id: ID of created subgraph (if create_new=True) - num_nodes: Number of nodes in subgraph - num_edges: Number of edges in subgraph - nodes: List of nodes in subgraph - extraction_stats: Statistics about the extraction

## Source

Located in: `src/networkx_mcp/server.py:2204`
