# Demo 3: Knowledge Graph Analysis

This demo shows how to build and analyze a knowledge graph to understand concept relationships and learning paths.

## Scenario
You're building a knowledge graph for a Computer Science curriculum to:
- Understand prerequisite relationships
- Find learning paths
- Identify foundational concepts

## Steps with Claude

### 1. Build the Knowledge Graph

```
Create a directed graph called "cs_knowledge"

Import this CSV data into cs_knowledge as a directed graph:
Programming_Basics,Data_Structures
Programming_Basics,Algorithms
Data_Structures,Algorithms
Algorithms,Machine_Learning
Algorithms,Databases
Mathematics,Machine_Learning
Mathematics,Algorithms
Statistics,Machine_Learning
Databases,Web_Development
Programming_Basics,Web_Development
Web_Development,Cloud_Computing
Machine_Learning,Deep_Learning
Machine_Learning,Natural_Language_Processing
Deep_Learning,Computer_Vision
Operating_Systems,Cloud_Computing
Networking,Cloud_Computing
Data_Structures,Operating_Systems
```

### 2. Find Learning Paths

```
Find shortest path in cs_knowledge from Programming_Basics to Deep_Learning
Find shortest path in cs_knowledge from Mathematics to Cloud_Computing
```

This shows the minimum concepts needed to learn before reaching advanced topics.

### 3. Identify Foundational Concepts

```
Calculate PageRank for cs_knowledge
```

**Expected Insight:**
Concepts like Programming_Basics and Algorithms should rank highly as they're prerequisites for many other topics.

### 4. Analyze Concept Connectivity

```
Calculate degree centrality for cs_knowledge
```

This shows which concepts are most connected to others in the curriculum.

### 5. Visualize the Knowledge Structure

```
Visualize cs_knowledge with kamada_kawai layout
```

The Kamada-Kawai layout often produces clear hierarchical structures for directed graphs.

### 6. Find Concept Clusters

```
Detect communities in cs_knowledge
```

This might reveal natural groupings like "Systems", "AI/ML", "Web Technologies".

## Real-World Applications
- Education: Curriculum planning and prerequisite mapping
- Research: Understanding citation networks and influence
- Enterprise: Mapping organizational knowledge and expertise
- Content Creation: Planning tutorial series or documentation
- Skills Development: Creating personalized learning paths

## Extension Ideas
- Add weights representing difficulty or time to learn
- Identify knowledge gaps in curriculum
- Recommend next topics to learn based on current knowledge
- Find alternative learning paths for different goals
- Analyze which concepts are "gateway" topics to new areas