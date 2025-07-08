# NetworkX MCP Server - Operational Runbook

## 🚨 Emergency Response

### Critical Alert Response Time: **< 15 minutes**
### Warning Alert Response Time: **< 1 hour**

---

## 📊 Performance Baselines (From Testing)

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| **Users** | 10 users | 50 users | 100 users | >100 users |
| **Success Rate** | 98.5% | 95.2% | 87.3% | <85% |
| **Avg Response** | 145ms | 320ms | 580ms | >1000ms |
| **P95 Response** | 280ms | 650ms | 1200ms | >2000ms |
| **Memory Usage** | 125MB | 185MB | 245MB | >500MB |

| Graph Size | Memory | Algorithm Time | Performance |
|------------|--------|----------------|-------------|
| 1K nodes | 15MB | 12ms | ✅ Excellent |
| 10K nodes | 120MB | 180ms | ✅ Good |
| 50K nodes | 450MB | 2.1s | ⚠️ Acceptable |
| 100K nodes | 1.2GB | 8.5s | ❌ Poor |

---

## 🔥 Critical Issues

### 1. High Memory Usage (>1.8GB)

**Symptoms:**
- Alert: `MCPMemoryUsageCritical`
- Memory usage >90% of 2GB limit
- Potential OOM kills

**Immediate Actions:**
```bash
# 1. Check current memory usage
kubectl exec -it deployment/networkx-mcp-server -- ps aux
curl http://mcp-server:9090/metrics | grep mcp_memory_usage_bytes

# 2. Identify large graphs
curl http://mcp-server:9090/metrics | grep mcp_graph_nodes_bucket

# 3. Check recent graph operations
kubectl logs deployment/networkx-mcp-server --tail=100 | grep "graph.*nodes"

# 4. Emergency mitigation - restart high-memory pods
kubectl rollout restart deployment/networkx-mcp-server
```

**Root Cause Investigation:**
1. **Large Graph Accumulation**: Check for graphs >10K nodes
2. **Memory Leaks**: Look for steadily growing memory without graph growth
3. **Algorithm Complexity**: Check for expensive algorithms on large graphs

**Resolution:**
- **Short-term**: Scale horizontally, restart high-memory pods
- **Long-term**: Implement graph partitioning, add memory limits per graph

### 2. High Error Rate (>15%)

**Symptoms:**
- Alert: `MCPHighErrorRate` 
- Error rate exceeding 15% threshold
- Normal rate: <5%

**Immediate Actions:**
```bash
# 1. Check error distribution
curl http://mcp-server:9090/metrics | grep mcp_requests_total

# 2. Get recent error logs
kubectl logs deployment/networkx-mcp-server --tail=200 | grep -i error

# 3. Check specific error types
kubectl logs deployment/networkx-mcp-server --since=10m | grep -E "(ValidationError|ResourceLimitError|TimeoutError)"

# 4. Verify input validation
curl -X POST http://mcp-server:8080/mcp/session
```

**Common Error Patterns:**
- **Validation Errors**: Malformed requests, invalid parameters
- **Resource Limit Errors**: Graph size exceeded, memory constraints
- **Timeout Errors**: Algorithms taking too long (>20s)
- **Connection Errors**: Transport layer issues

**Resolution Steps:**
1. **Input Validation**: Check for malformed requests
2. **Resource Scaling**: Add capacity if resource-limited
3. **Algorithm Optimization**: Use approximate algorithms for large graphs
4. **Client Education**: Update client libraries if protocol issues

### 3. Connection Pool Exhaustion (>45 connections)

**Symptoms:**
- Alert: `MCPConnectionPoolExhausted`
- New connections failing
- Based on testing: 50 user limit, optimal at 45

**Immediate Actions:**
```bash
# 1. Check current connections
curl http://mcp-server:9090/metrics | grep mcp_active_connections

# 2. Identify connection distribution
kubectl exec -it deployment/networkx-mcp-server -- netstat -an | grep ESTABLISHED

# 3. Check for stuck connections
kubectl logs deployment/networkx-mcp-server --tail=100 | grep -E "(connection|session)"

# 4. Scale immediately
kubectl scale deployment networkx-mcp-server --replicas=6
```

**Investigation:**
- **Stuck Sessions**: Long-running operations not releasing connections
- **Client Behavior**: Clients not properly closing connections
- **Load Imbalance**: Uneven distribution across instances

**Resolution:**
- **Immediate**: Horizontal scaling
- **Medium-term**: Connection timeout tuning
- **Long-term**: Connection pooling optimization

### 4. Critical Response Times (P95 >5s)

**Symptoms:**
- Alert: `MCPResponseTimesCritical`
- P95 response time >5 seconds
- Normal P95: <2 seconds

**Immediate Actions:**
```bash
# 1. Check response time distribution
curl http://mcp-server:9090/metrics | grep mcp_request_duration_seconds_bucket

# 2. Identify slow operations
kubectl logs deployment/networkx-mcp-server --tail=100 | grep -E "duration.*[5-9][0-9]{3}ms"

# 3. Check algorithm performance
curl http://mcp-server:9090/metrics | grep mcp_algorithm_duration_seconds

# 4. Look for large graphs in recent operations
kubectl logs deployment/networkx-mcp-server --since=30m | grep -E "nodes.*[0-9]{4,}"
```

**Common Causes:**
1. **Large Graph Algorithms**: Operations on >10K node graphs
2. **Complex Algorithms**: Betweenness centrality, community detection
3. **Resource Contention**: CPU/memory pressure
4. **Storage Latency**: Redis backend performance

**Resolution Steps:**
1. **Algorithm Optimization**: Use approximate algorithms
2. **Graph Partitioning**: Break large graphs into smaller chunks
3. **Resource Scaling**: Add CPU/memory capacity
4. **Caching**: Implement result caching for repeated operations

---

## ⚠️ Warning Issues

### 1. High Memory Usage (>1.5GB)

**Actions:**
```bash
# Monitor memory growth trend
curl http://mcp-server:9090/metrics | grep mcp_memory_usage_bytes

# Check for memory leaks
kubectl top pods -l app=networkx-mcp

# Review recent large graph operations
kubectl logs deployment/networkx-mcp-server --since=1h | grep -E "(create_graph|add_nodes).*[0-9]{4,}"
```

**Prevention:**
- Monitor graph sizes regularly
- Implement automatic cleanup of large graphs
- Set per-graph memory limits

### 2. Moderate Error Rate (5-15%)

**Actions:**
```bash
# Analyze error patterns
kubectl logs deployment/networkx-mcp-server --since=30m | grep -i error | head -20

# Check input validation metrics
curl http://mcp-server:9090/metrics | grep mcp_validation_errors_total

# Review recent changes
kubectl rollout history deployment/networkx-mcp-server
```

### 3. Slow Responses (P95 2-5s)

**Actions:**
```bash
# Check algorithm performance
curl http://mcp-server:9090/metrics | grep mcp_algorithm_duration_seconds_bucket

# Identify slow endpoints
kubectl logs deployment/networkx-mcp-server --since=10m | grep -E "duration.*[2-4][0-9]{3}ms"

# Monitor resource usage
kubectl top pods -l app=networkx-mcp
```

---

## 🔧 Common Troubleshooting

### Memory Issues

**Check Graph Distribution:**
```bash
# Get graph size histogram
curl http://mcp-server:9090/metrics | grep mcp_graph_nodes_bucket

# Find large graphs
kubectl logs deployment/networkx-mcp-server --since=1h | grep -E "graph.*nodes.*[0-9]{4,}" | sort -k5 -nr
```

**Memory Leak Detection:**
```bash
# Memory growth over time
curl http://mcp-server:9090/metrics | grep mcp_memory_usage_bytes
# Compare with graph count
curl http://mcp-server:9090/metrics | grep mcp_graphs_total
```

**Mitigation:**
- Restart high-memory pods
- Clear large graphs manually
- Implement graph size limits

### Performance Issues

**Algorithm Performance Analysis:**
```bash
# Check algorithm distribution
curl http://mcp-server:9090/metrics | grep mcp_algorithm_duration_seconds_bucket

# Identify expensive operations
kubectl logs deployment/networkx-mcp-server --since=30m | grep -E "(betweenness|closeness|community)" | head -10
```

**Graph Size Impact:**
```bash
# Correlate response time with graph size
kubectl logs deployment/networkx-mcp-server --since=1h | grep -E "duration.*nodes" | awk '{print $X, $Y}' | sort -k2 -nr
```

**Optimization Strategies:**
1. **Use Approximate Algorithms**: For graphs >10K nodes
2. **Implement Sampling**: Random node/edge sampling
3. **Cache Results**: Store expensive computation results
4. **Partition Graphs**: Break large graphs into components

### Connection Issues

**Transport Layer Debugging:**
```bash
# Check connection states
kubectl exec -it deployment/networkx-mcp-server -- ss -tuln

# HTTP transport health
curl http://mcp-server:8080/health

# Session management
curl http://mcp-server:8080/info
```

**stdio vs HTTP Issues:**
- **stdio**: Check for broken pipes, invalid JSON
- **HTTP**: Check CORS, authentication, session timeouts

### Authentication Issues

**OAuth Debugging:**
```bash
# Check auth metrics
curl http://mcp-server:9090/metrics | grep mcp_auth_attempts_total

# Recent auth failures
kubectl logs deployment/networkx-mcp-server --since=10m | grep -i "auth.*fail"

# Token validation issues
kubectl logs deployment/networkx-mcp-server --since=10m | grep -E "(token|oauth|jwt)"
```

---

## 📈 Capacity Planning

### Scaling Decisions

**When to Scale Horizontally:**
- Active connections >35 (70% of capacity)
- Memory usage >1.5GB (75% of limit)
- P95 response time >2s consistently
- Error rate >5% for sustained period

**Scaling Commands:**
```bash
# Scale deployment
kubectl scale deployment networkx-mcp-server --replicas=6

# Check scaling status
kubectl get hpa networkx-mcp-hpa

# Monitor new pods
kubectl get pods -l app=networkx-mcp -w
```

**When to Scale Vertically:**
- Single-threaded algorithm bottlenecks
- Memory-intensive operations
- CPU-bound computations

### Performance Optimization

**Graph Size Management:**
```bash
# Implement graph size limits
# Add to tool validation:
if nodes > 10000:
    return {"error": "Graph too large, use partitioning"}

# Automatic cleanup
# Schedule cleanup of graphs >50K nodes after 1 hour
```

**Algorithm Optimization:**
```python
# Use approximate algorithms for large graphs
if graph.number_of_nodes() > 10000:
    # Use sampling for centrality measures
    sample_size = min(1000, graph.number_of_nodes() // 10)
    sampled_nodes = random.sample(list(graph.nodes()), sample_size)
    return approximate_centrality(graph, sampled_nodes)
```

---

## 🔍 Monitoring & Diagnostics

### Key Metrics to Watch

**Performance Metrics:**
- `mcp_request_duration_seconds` (P50, P95, P99)
- `mcp_requests_total` (success rate)
- `mcp_active_connections` (capacity utilization)
- `mcp_memory_usage_bytes` (resource usage)

**Business Metrics:**
- `mcp_graph_operations_total` (usage patterns)
- `mcp_algorithm_duration_seconds` (complexity analysis)
- `mcp_graph_nodes` (size distribution)

**System Metrics:**
- `mcp_cpu_usage_percent` (resource utilization)
- `mcp_file_descriptors_open` (resource leaks)
- `mcp_component_health` (system health)

### Log Analysis

**Performance Debugging:**
```bash
# Find slow requests
kubectl logs deployment/networkx-mcp-server --since=1h | grep -E "duration.*[0-9]{4,}ms" | head -10

# Memory growth patterns
kubectl logs deployment/networkx-mcp-server --since=4h | grep -E "memory.*MB" | awk '{print $1, $X}' | sort
```

**Error Pattern Analysis:**
```bash
# Error frequency by type
kubectl logs deployment/networkx-mcp-server --since=1h | grep -i error | cut -d' ' -f3- | sort | uniq -c | sort -nr

# Validation errors
kubectl logs deployment/networkx-mcp-server --since=30m | grep "ValidationError" | head -5
```

### Distributed Tracing

**Jaeger Queries:**
- Service: `networkx-mcp-server`
- Operation: `mcp.request.*`
- Tags: `mcp.performance.tier`, `algorithm.performance.anomaly`

**Common Trace Analysis:**
1. **Slow Requests**: Filter by duration >2s
2. **Error Requests**: Filter by error=true
3. **Large Graphs**: Filter by `graph.size.category=large`
4. **Algorithm Performance**: Group by `algorithm.name`

---

## 🚀 Deployment & Rollback

### Safe Deployment Process

**Pre-deployment Checks:**
```bash
# 1. Check current system health
kubectl get pods -l app=networkx-mcp
curl http://mcp-server:9090/metrics | grep mcp_performance_tier

# 2. Backup current configuration
kubectl get deployment networkx-mcp-server -o yaml > backup-deployment.yaml

# 3. Review change impact
git diff HEAD~1 HEAD --name-only | grep -E "(algorithm|core|server)"
```

**Rolling Deployment:**
```bash
# 1. Update image
kubectl set image deployment/networkx-mcp-server mcp-server=networkx-mcp:new-version

# 2. Monitor rollout
kubectl rollout status deployment/networkx-mcp-server --timeout=300s

# 3. Verify health
kubectl get pods -l app=networkx-mcp
curl http://mcp-server:8080/health
```

**Health Verification:**
```bash
# Check all pods are ready
kubectl get pods -l app=networkx-mcp | grep -c "1/1.*Running"

# Verify metrics endpoint
curl http://mcp-server:9090/metrics | grep mcp_server_info

# Test basic functionality
curl -X POST http://mcp-server:8080/mcp/session
```

### Emergency Rollback

**Immediate Rollback:**
```bash
# 1. Rollback deployment
kubectl rollout undo deployment/networkx-mcp-server

# 2. Verify rollback
kubectl rollout status deployment/networkx-mcp-server

# 3. Check pod health
kubectl get pods -l app=networkx-mcp
```

**Post-rollback Actions:**
1. Analyze failure cause
2. Update incident documentation
3. Plan fix implementation
4. Communicate to stakeholders

---

## 📞 Escalation Procedures

### Incident Severity Levels

**P0 - Critical (Response: <15 min)**
- Server completely down
- Error rate >50%
- Memory exhaustion causing crashes
- Data corruption

**P1 - High (Response: <1 hour)**
- Performance severely degraded (P95 >5s)
- Error rate >15%
- Connection pool exhausted
- Security breach

**P2 - Medium (Response: <4 hours)**
- Moderate performance degradation
- Error rate 5-15%
- Resource warnings
- Non-critical feature failures

**P3 - Low (Response: <24 hours)**
- Minor performance issues
- Non-urgent optimization needs
- Documentation updates

### Contact Information

**Primary On-call:** Platform Engineering Team
**Secondary:** DevOps Team  
**Escalation:** Engineering Manager

**Communication Channels:**
- Slack: `#incidents-mcp`
- PagerDuty: `MCP Server Issues`
- Email: `platform-engineering@company.com`

---

## 📚 Reference Links

- **Metrics Dashboard**: [Grafana MCP Dashboard](http://grafana.company.com/d/mcp-overview)
- **Tracing**: [Jaeger Traces](http://jaeger.company.com/search?service=networkx-mcp-server)
- **Logs**: [Log Aggregation](http://logs.company.com/app/mcp-server)
- **Performance Testing**: [Load Test Results](./MCP_CLIENT_COMPATIBILITY.md)
- **Architecture**: [Production Deployment](./PRODUCTION_DEPLOYMENT_SUMMARY.md)

---

## 📝 Maintenance Tasks

### Daily
- [ ] Check alert status
- [ ] Review error rates
- [ ] Monitor memory trends
- [ ] Verify backup completion

### Weekly  
- [ ] Performance trend analysis
- [ ] Capacity planning review
- [ ] Security patch assessment
- [ ] Documentation updates

### Monthly
- [ ] Load testing execution
- [ ] Disaster recovery testing
- [ ] Performance optimization review
- [ ] Incident retrospectives

---

*Last Updated: Production Deployment 2024*  
*Next Review: [Schedule quarterly review]*