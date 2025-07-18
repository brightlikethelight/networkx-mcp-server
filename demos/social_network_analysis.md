# Demo 1: Social Network Analysis

This demo shows how to analyze a social network to find influencers, communities, and key connections.

## Scenario

You're analyzing a small social network to understand:

- Who are the most influential people (high centrality)
- What communities exist
- Who are the bridges between communities

## Steps with Claude

### 1. Create and Import the Network

```
Create a graph called "social_network" and import this CSV data:
Alice,Bob
Alice,Charlie
Bob,Charlie
Bob,David
Charlie,Eve
David,Eve
David,Frank
Eve,Frank
Frank,George
George,Helen
Helen,Ivan
Ivan,George
```

### 2. Analyze Influence (Centrality)

```
Calculate degree centrality for social_network
Calculate betweenness centrality for social_network
```

**Expected Insights:**

- Degree centrality shows who has the most connections
- Betweenness centrality reveals who are the "bridges" between groups

### 3. Find Communities

```
Detect communities in social_network
```

**Expected Result:**
The algorithm should identify 2-3 distinct communities in the network.

### 4. Visualize the Network

```
Visualize social_network with spring layout
```

This creates a visual representation showing the network structure.

### 5. Identify Key Players

```
Calculate PageRank for social_network
```

PageRank (Google's algorithm) shows overall importance in the network.

## Real-World Applications

- LinkedIn: Finding key connectors in professional networks
- Marketing: Identifying influencers for campaigns
- Organization: Understanding informal communication patterns
- Security: Detecting potential information flow bottlenecks
