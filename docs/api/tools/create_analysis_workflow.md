# create_analysis_workflow

**Category:** Core Operations

## Description

Execute analysis workflow with chained operations.

## Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `graph_id` | `str` | Yes | `-` | ID of the graph |
| `workflow_steps` | `List[Dict[str, Any]]` | Yes | `-` | List of workflow step configurations |
| `cache_results` | `bool` | No | `True` | Cache intermediate results |
| `params` | `Optional[Dict[str, Any]]` | No | `-` | Additional parameters |

## Returns

**Type:** `Dict[str, Any]`

Workflow execution results

## Source

Located in: `src/networkx_mcp/server.py:3211`
