# NetworkX MCP Server -- API Reference

All tools are exposed via the MCP `tools/call` JSON-RPC method.
Parameters are passed as `arguments` inside the request.

## Table of Contents

1. [Graph Management](#graph-management) -- create_graph, add_nodes, add_edges, get_info, delete_graph
2. [Algorithms](#algorithms) -- shortest_path, degree_centrality, betweenness_centrality, connected_components, pagerank, community_detection
3. [I/O and Visualization](#io-and-visualization) -- import_csv, export_json, visualize_graph
4. [Academic / Citation](#academic--citation) -- build_citation_network, analyze_author_impact, find_collaboration_patterns, detect_research_trends, export_bibtex, recommend_papers, resolve_doi
5. [CI/CD Control](#cicd-control) -- trigger_workflow, get_workflow_status, cancel_workflow, rerun_failed_jobs, get_dora_metrics, analyze_workflow_failures
6. [Monitoring](#monitoring) -- health_status

---

## Graph Management

### create_graph

Create a new graph.

| Parameter  | Type    | Required | Default | Description              |
|------------|---------|----------|---------|--------------------------|
| `name`     | string  | yes      | --      | Unique name for the graph |
| `directed` | boolean | no       | false   | Create a directed graph  |

```json
{"name": "tools/call", "arguments": {"name": "my_graph", "directed": false}}
```

### add_nodes

Add nodes to an existing graph.

| Parameter | Type            | Required | Description                |
|-----------|-----------------|----------|----------------------------|
| `graph`   | string          | yes      | Name of the target graph   |
| `nodes`   | array of string/number | yes | Node identifiers to add |

```json
{"graph": "my_graph", "nodes": ["A", "B", "C"]}
```

### add_edges

Add edges to an existing graph.

| Parameter | Type                       | Required | Description                  |
|-----------|----------------------------|----------|------------------------------|
| `graph`   | string                     | yes      | Name of the target graph     |
| `edges`   | array of [source, target]  | yes      | Pairs of node identifiers    |

```json
{"graph": "my_graph", "edges": [["A", "B"], ["B", "C"]]}
```

### get_info

Get graph information (node count, edge count, directedness).

| Parameter | Type   | Required | Description          |
|-----------|--------|----------|----------------------|
| `graph`   | string | yes      | Name of the graph    |

```json
{"graph": "my_graph"}
```

Returns: `{"nodes": 3, "edges": 2, "directed": false}`

### delete_graph

Delete a graph from storage.

| Parameter | Type   | Required | Description          |
|-----------|--------|----------|----------------------|
| `graph`   | string | yes      | Name of the graph    |

```json
{"graph": "my_graph"}
```

---

## Algorithms

### shortest_path

Find the shortest path between two nodes.

| Parameter | Type          | Required | Description      |
|-----------|---------------|----------|------------------|
| `graph`   | string        | yes      | Graph name       |
| `source`  | string/number | yes      | Start node       |
| `target`  | string/number | yes      | End node         |

```json
{"graph": "my_graph", "source": "A", "target": "C"}
```

Returns: `{"path": ["A", "B", "C"], "length": 2}`

### degree_centrality

Calculate degree centrality for all nodes.

| Parameter | Type   | Required | Description |
|-----------|--------|----------|-------------|
| `graph`   | string | yes      | Graph name  |

```json
{"graph": "my_graph"}
```

### betweenness_centrality

Calculate betweenness centrality for all nodes.

| Parameter | Type   | Required | Description |
|-----------|--------|----------|-------------|
| `graph`   | string | yes      | Graph name  |

```json
{"graph": "my_graph"}
```

### connected_components

Find connected components in the graph.

| Parameter | Type   | Required | Description |
|-----------|--------|----------|-------------|
| `graph`   | string | yes      | Graph name  |

```json
{"graph": "my_graph"}
```

### pagerank

Calculate PageRank for all nodes.

| Parameter | Type   | Required | Description |
|-----------|--------|----------|-------------|
| `graph`   | string | yes      | Graph name  |

```json
{"graph": "my_graph"}
```

### community_detection

Detect communities using the Louvain method.

| Parameter | Type   | Required | Description |
|-----------|--------|----------|-------------|
| `graph`   | string | yes      | Graph name  |

```json
{"graph": "my_graph"}
```

---

## I/O and Visualization

### import_csv

Import a graph from a CSV edge list. Each line should be `source,target`.

| Parameter  | Type    | Required | Default | Description              |
|------------|---------|----------|---------|--------------------------|
| `graph`    | string  | yes      | --      | Name for the new graph   |
| `csv_data` | string  | yes      | --      | CSV edge list as a string |
| `directed` | boolean | no       | false   | Import as directed graph |

```json
{"graph": "csv_graph", "csv_data": "A,B\nB,C\nC,A", "directed": false}
```

### export_json

Export a graph in node-link JSON format.

| Parameter | Type   | Required | Description |
|-----------|--------|----------|-------------|
| `graph`   | string | yes      | Graph name  |

```json
{"graph": "my_graph"}
```

### visualize_graph

Create a visualization of the graph (returns base64 image).

| Parameter | Type   | Required | Default  | Description                                     |
|-----------|--------|----------|----------|-------------------------------------------------|
| `graph`   | string | yes      | --       | Graph name                                      |
| `layout`  | string | no       | "spring" | Layout algorithm: `spring`, `circular`, or `kamada_kawai` |

```json
{"graph": "my_graph", "layout": "circular"}
```

---

## Academic / Citation

These tools use the CrossRef API to work with scholarly citation data.

### build_citation_network

Build a citation network from seed DOIs.

| Parameter   | Type             | Required | Default | Description                  |
|-------------|------------------|----------|---------|------------------------------|
| `graph`     | string           | yes      | --      | Name for the citation graph  |
| `seed_dois` | array of string  | yes      | --      | Starting DOIs to crawl       |
| `max_depth` | integer          | no       | 2       | Citation traversal depth     |

```json
{"graph": "citations", "seed_dois": ["10.1038/nature12373"], "max_depth": 2}
```

### analyze_author_impact

Analyze author impact metrics including h-index.

| Parameter     | Type   | Required | Description                  |
|---------------|--------|----------|------------------------------|
| `graph`       | string | yes      | Citation graph name          |
| `author_name` | string | yes      | Author to analyze            |

```json
{"graph": "citations", "author_name": "Jane Smith"}
```

### find_collaboration_patterns

Find collaboration patterns in a citation network.

| Parameter | Type   | Required | Description         |
|-----------|--------|----------|---------------------|
| `graph`   | string | yes      | Citation graph name |

```json
{"graph": "citations"}
```

### detect_research_trends

Detect research trends over time.

| Parameter     | Type    | Required | Default | Description               |
|---------------|---------|----------|---------|---------------------------|
| `graph`       | string  | yes      | --      | Citation graph name       |
| `time_window` | integer | no       | 5       | Window size in years      |

```json
{"graph": "citations", "time_window": 3}
```

### export_bibtex

Export a citation network as BibTeX.

| Parameter | Type   | Required | Description         |
|-----------|--------|----------|---------------------|
| `graph`   | string | yes      | Citation graph name |

```json
{"graph": "citations"}
```

### recommend_papers

Recommend papers based on citation network analysis.

| Parameter             | Type    | Required | Default | Description                     |
|-----------------------|---------|----------|---------|---------------------------------|
| `graph`               | string  | yes      | --      | Citation graph name             |
| `seed_doi`            | string  | yes      | --      | DOI of the reference paper      |
| `max_recommendations` | integer | no       | 10      | Maximum papers to recommend     |

```json
{"graph": "citations", "seed_doi": "10.1038/nature12373", "max_recommendations": 5}
```

### resolve_doi

Resolve a DOI to publication metadata via CrossRef. This tool does not require a graph.

| Parameter | Type   | Required | Description |
|-----------|--------|----------|-------------|
| `doi`     | string | yes      | The DOI     |

```json
{"doi": "10.1038/nature12373"}
```

---

## CI/CD Control

These tools interact with GitHub Actions. They are available when the `networkx_mcp.tools` module is installed.

### trigger_workflow

Trigger a GitHub Actions workflow.

| Parameter  | Type   | Required | Default  | Description                  |
|------------|--------|----------|----------|------------------------------|
| `workflow` | string | yes      | --       | Workflow file name           |
| `branch`   | string | no       | "main"   | Branch to run the workflow on |
| `inputs`   | string | no       | --       | JSON string of workflow inputs |

```json
{"workflow": "ci.yml", "branch": "main", "inputs": "{\"deploy\": true}"}
```

### get_workflow_status

Get the status of CI/CD workflow runs.

| Parameter | Type   | Required | Description                         |
|-----------|--------|----------|-------------------------------------|
| `run_id`  | string | no       | Specific run ID; omit for latest runs |

```json
{"run_id": "12345678"}
```

### cancel_workflow

Cancel a running workflow.

| Parameter | Type   | Required | Description |
|-----------|--------|----------|-------------|
| `run_id`  | string | yes      | Run ID      |

```json
{"run_id": "12345678"}
```

### rerun_failed_jobs

Rerun only the failed jobs in a workflow run.

| Parameter | Type   | Required | Description |
|-----------|--------|----------|-------------|
| `run_id`  | string | yes      | Run ID      |

```json
{"run_id": "12345678"}
```

### get_dora_metrics

Get DORA metrics (deployment frequency, lead time, change failure rate, MTTR) for CI/CD performance. Takes no parameters.

```json
{}
```

### analyze_workflow_failures

Analyze workflow failures with AI-powered insights.

| Parameter | Type   | Required | Description |
|-----------|--------|----------|-------------|
| `run_id`  | string | yes      | Run ID      |

```json
{"run_id": "12345678"}
```

---

## Monitoring

### health_status

Get server health and performance metrics. Only available when monitoring is enabled (`NETWORKX_MCP_MONITORING=true`). Takes no parameters.

```json
{}
```
