# Demo 2: Transportation Network Routing

This demo shows how to analyze a transportation network for optimal routing and network resilience.

## Scenario
You're analyzing a city's transportation network to:
- Find shortest routes between locations
- Identify critical intersections
- Analyze network resilience

## Steps with Claude

### 1. Create the Transportation Network

```
Create a directed graph called "city_transport"

Add these nodes to city_transport: ["Airport", "Downtown", "University", "Hospital", "Mall", "Suburb1", "Suburb2", "Industrial", "Port", "Station"]

Add these edges to city_transport:
[["Airport", "Downtown"], ["Downtown", "Airport"], 
 ["Downtown", "University"], ["University", "Downtown"],
 ["Downtown", "Hospital"], ["Hospital", "Downtown"],
 ["Downtown", "Mall"], ["Mall", "Downtown"],
 ["Downtown", "Station"], ["Station", "Downtown"],
 ["Suburb1", "Downtown"], ["Downtown", "Suburb1"],
 ["Suburb2", "Mall"], ["Mall", "Suburb2"],
 ["Industrial", "Port"], ["Port", "Industrial"],
 ["Port", "Downtown"], ["Downtown", "Port"],
 ["Station", "University"], ["University", "Hospital"]]
```

### 2. Find Optimal Routes

```
Find shortest path in city_transport from Airport to Hospital
Find shortest path in city_transport from Suburb1 to University
```

### 3. Identify Critical Intersections

```
Calculate betweenness centrality for city_transport
```

**Expected Insight:**
Downtown should show highest betweenness centrality as it's the main hub.

### 4. Analyze Network Connectivity

```
Find connected components in city_transport
```

This shows if all locations are reachable from each other.

### 5. Visualize the Network

```
Visualize city_transport with circular layout
```

The circular layout often works well for transportation networks.

## Real-World Applications
- Urban Planning: Identifying traffic bottlenecks
- Emergency Services: Finding fastest routes to hospitals
- Public Transit: Optimizing bus/train routes
- Logistics: Planning delivery routes
- Infrastructure: Identifying critical nodes for maintenance priority

## Extension Ideas
- Add weights to edges representing travel time
- Model traffic congestion at different times
- Plan alternative routes when main roads are blocked
- Optimize placement of new roads or transit lines