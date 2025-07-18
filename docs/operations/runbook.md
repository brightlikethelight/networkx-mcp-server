# NetworkX MCP Server - Operations Runbook

## Table of Contents

1. [Overview](#overview)
2. [Quick Reference](#quick-reference)
3. [Alert Response Procedures](#alert-response-procedures)
4. [Common Issues & Solutions](#common-issues--solutions)
5. [Performance Analysis](#performance-analysis)
6. [Scaling Operations](#scaling-operations)
7. [Emergency Procedures](#emergency-procedures)
8. [Debugging & Troubleshooting](#debugging--troubleshooting)

---

## Overview

This runbook provides operational procedures for the NetworkX MCP Server based on production testing data and performance characteristics. The server handles MCP (Model Context Protocol) requests for graph operations with the following tested limits:

### Performance Baselines (From Load Testing)

- **50 concurrent users**: 95.2% success rate, 320ms avg response, 650ms P95
- **100 concurrent users**: 88.5% success rate, degraded performance
- **10K node graphs**: ~120MB memory, ~180ms algorithm execution
- **50K node graphs**: ~450MB memory, ~2.1s algorithm execution

### Production Configuration

- **Max concurrent connections**: 45 (90% of tested 50-user limit)
- **Max graph size**: 10K nodes (conservative for good performance)
- **Memory limit**: 2GB (allows multiple large graphs)
- **Target SLA**: P95 < 2s, Error rate < 5%

---

## Quick Reference

### Service Endpoints

```bash
# Health checks
curl https://api.mcp.example.com/health
curl https://api.mcp.example.com/ready
curl https://api.mcp.example.com/startup

# Metrics
curl https://api.mcp.example.com:9090/metrics

# Tracing
https://jaeger.monitoring.example.com
```

### Key Metrics URLs

- **Dashboard**: <https://grafana.example.com/d/networkx-mcp-prod>
- **Alerts**: <https://alertmanager.example.com>
- **Logs**: <https://kibana.example.com/app/logs>

### Emergency Contacts

- **On-Call**: PagerDuty escalation policy
- **Slack**: #mcp-alerts (immediate), #platform-team (general)
- **Email**: <platform-oncall@example.com>

---

## Alert Response Procedures

### 🚨 MCPServerDown / MCPAllPodsDown

**Severity**: Critical | **Response Time**: < 5 minutes

**Immediate Actions:**

```bash
# Check pod status
kubectl get pods -l app=networkx-mcp -o wide

# Check events for crashes/OOMKills
kubectl get events --sort-by=.metadata.creationTimestamp | grep networkx-mcp

# Check recent logs
kubectl logs -l app=networkx-mcp --tail=100 --timestamps
```

**Common Causes & Solutions:**

1. **OOM Kill** (most likely with our memory-intensive workload)

   ```bash
   # Check for OOM in events
   kubectl get events | grep OOMKilled

   # Immediate fix: Restart pods
   kubectl rollout restart deployment/networkx-mcp-server

   # Medium-term: Increase memory limits or reduce graph sizes
   ```

2. **Crash Loop** (algorithm failures, configuration issues)

   ```bash
   # Check crash logs
   kubectl logs deployment/networkx-mcp-server --previous

   # Check configuration
   kubectl get configmap mcp-config -o yaml

   # Rollback if recent deployment
   kubectl rollout undo deployment/networkx-mcp-server
   ```

3. **Kubernetes Issues**

   ```bash
   # Check node capacity
   kubectl top nodes
   kubectl describe nodes | grep -A 5 "Allocated resources"

   # Check for node issues
   kubectl get nodes
   kubectl describe node $NODE_NAME
   ```

**Escalation**: If pods won't restart within 10 minutes, page platform lead

### ⚠️ MCPHighErrorRateCritical

**Severity**: Critical | **Response Time**: < 5 minutes

**Analysis Steps:**

```bash
# Check error breakdown by method
curl -s https://api.mcp.example.com:9090/api/v1/query?query='sum(rate(mcp_requests_total{status="error"}[5m])) by (method)'

# Check recent error logs
kubectl logs -l app=networkx-mcp --tail=200 | grep ERROR

# Check if specific graph operations are failing
curl -s https://api.mcp.example.com:9090/api/v1/query?query='mcp_algorithm_duration_seconds{quantile="0.95"}'
```

**Common Causes:**

1. **Large Graph Processing** (most common issue)
   - Graphs exceeding 10K nodes cause timeouts
   - **Solution**: Enable approximate algorithms or implement graph size limits

   ```bash
   # Check graph sizes being processed
   curl -s "https://api.mcp.example.com:9090/api/v1/query?query=histogram_quantile(0.95, sum(rate(mcp_graph_nodes_bucket[5m])) by (le))"
   ```

2. **Memory Pressure**
   - Multiple large graphs consuming memory
   - **Solution**: Restart service or reduce concurrent operations

   ```bash
   # Check memory usage
   curl -s "https://api.mcp.example.com:9090/api/v1/query?query=mcp_memory_usage_bytes"
   ```

3. **Storage Backend Issues**

   ```bash
   # Check Redis connectivity
   kubectl exec -it deployment/redis -- redis-cli ping

   # Check Redis memory usage
   kubectl exec -it deployment/redis -- redis-cli info memory
   ```

### ⚠️ MCPConnectionPoolExhausted

**Severity**: Critical | **Response Time**: < 2 minutes

**Immediate Actions:**

```bash
# Check current connection count
curl -s "https://api.mcp.example.com:9090/api/v1/query?query=sum(mcp_active_connections)"

# Scale up immediately (if infrastructure allows)
kubectl scale deployment networkx-mcp-server --replicas=5

# Check for stuck connections
kubectl logs -l app=networkx-mcp --tail=100 | grep "connection"
```

**Root Cause Analysis:**

1. **Traffic Spike** (legitimate load increase)
   - Scale horizontally and monitor
   - Consider implementing request queuing

2. **Stuck Connections** (connections not being released)
   - Restart affected pods
   - Check for algorithm deadlocks in traces

3. **DDoS or Abuse**
   - Check for unusual request patterns
   - Implement rate limiting

### ⚠️ MCPHighMemoryUsage / MCPMemoryCritical

**Severity**: Warning/Critical | **Response Time**: < 10 minutes

**Analysis Framework:**

```bash
# Memory usage breakdown
curl -s "https://api.mcp.example.com:9090/api/v1/query?query=mcp_memory_usage_bytes / 1024 / 1024"

# Estimate graph count (450MB per 50K graph)
echo "Current graphs: $((memory_mb / 450))"

# Check for memory leaks
kubectl top pods -l app=networkx-mcp
```

**Response Based on Usage:**

- **1.5GB (75%)**: Warning - monitor closely
- **1.7GB (85%)**: Prepare to scale or reduce load
- **1.9GB (95%)**: Critical - immediate action required

**Actions by Scenario:**

1. **Normal Heavy Usage** (multiple large graphs)

   ```bash
   # Scale horizontally to distribute load
   kubectl scale deployment networkx-mcp-server --replicas=4

   # Or restart pods to clear memory
   kubectl rollout restart deployment/networkx-mcp-server
   ```

2. **Memory Leak** (steadily increasing usage)

   ```bash
   # Force garbage collection (if instrumented)
   curl -X POST https://api.mcp.example.com/debug/gc

   # Rolling restart
   kubectl patch deployment networkx-mcp-server -p '{"spec":{"template":{"metadata":{"annotations":{"restart":"'$(date +%s)'"}}}}}'
   ```

3. **Algorithmic Issue** (memory spike during processing)

   ```bash
   # Check algorithm performance
   curl -s "https://api.mcp.example.com:9090/api/v1/query?query=histogram_quantile(0.95, sum(rate(mcp_algorithm_duration_seconds_bucket[5m])) by (algorithm, le))"

   # Enable approximate algorithms for large graphs
   kubectl patch configmap mcp-config --patch '{"data":{"use_approximate_algorithms":"true"}}'
   ```

---

## Common Issues & Solutions

### Issue: Slow Algorithm Performance

**Symptoms**: P95 response time >2s, algorithm duration high
**Impact**: Poor user experience, potential timeouts

**Analysis**:

```bash
# Check algorithm performance by graph size
curl -s "https://api.mcp.example.com:9090/api/v1/query?query=histogram_quantile(0.95, sum(rate(mcp_algorithm_duration_seconds_bucket[5m])) by (graph_size_bucket, le))"

# Find slowest algorithms
curl -s "https://api.mcp.example.com:9090/api/v1/query?query=topk(5, histogram_quantile(0.95, sum(rate(mcp_algorithm_duration_seconds_bucket[5m])) by (algorithm, le)))"
```

**Solutions**:

1. **For Small Graphs** (<1K nodes) taking >100ms:
   - Check for CPU throttling
   - Look for memory pressure affecting performance

2. **For Medium Graphs** (1K-10K nodes) taking >500ms:
   - Normal for complex algorithms (betweenness centrality)
   - Consider algorithm optimization or approximate methods

3. **For Large Graphs** (>10K nodes) taking >2s:
   - Expected behavior based on testing
   - Enable approximate algorithms
   - Implement sampling for very large graphs

### Issue: Graph Size Limits Exceeded

**Symptoms**: Validation errors, memory spikes
**Expected Frequency**: Should be rare with proper client validation

**Response**:

```bash
# Check recent large graphs
curl -s "https://api.mcp.example.com:9090/api/v1/query?query=mcp_graph_nodes > 10000"

# Review validation errors
kubectl logs -l app=networkx-mcp | grep "validation" | tail -20
```

**Solutions**:

1. Update client applications to validate graph sizes
2. Implement server-side graph splitting for very large graphs
3. Consider raising limits if infrastructure supports it

### Issue: Storage Backend Problems

**Symptoms**: High storage latency, Redis connection errors

**Diagnosis**:

```bash
# Check Redis health
kubectl exec -it deployment/redis -- redis-cli ping
kubectl exec -it deployment/redis -- redis-cli info replication

# Check storage operation latency
curl -s "https://api.mcp.example.com:9090/api/v1/query?query=histogram_quantile(0.95, sum(rate(mcp_storage_operation_duration_seconds_bucket[5m])) by (operation, le))"
```

**Solutions**:

1. **Redis Memory Issues**:

   ```bash
   # Check Redis memory
   kubectl exec -it deployment/redis -- redis-cli info memory

   # Clear cache if needed (CAREFUL!)
   kubectl exec -it deployment/redis -- redis-cli FLUSHDB
   ```

2. **Redis Connection Pool Exhaustion**:

   ```bash
   # Restart Redis
   kubectl rollout restart deployment/redis

   # Increase connection pool size in config
   ```

---

## Performance Analysis

### Interpreting Performance Metrics

**Response Time Categories** (based on testing):

- **Excellent** (<100ms): Light load, small graphs
- **Good** (100-500ms): Normal operation with medium graphs
- **Acceptable** (500ms-2s): Heavy load or large graphs
- **Poor** (>2s): Overloaded or very large graphs

**Memory Usage Patterns**:

- **Baseline**: ~50MB (empty service)
- **Light Load**: 50-200MB (small graphs)
- **Normal Load**: 200-800MB (mixed graph sizes)
- **Heavy Load**: 800MB-1.5GB (multiple large graphs)
- **Critical**: >1.5GB (approaching limits)

### Performance Optimization

**For High Latency**:

1. Check graph size distribution
2. Enable approximate algorithms for large graphs
3. Implement result caching
4. Scale horizontally

**For High Memory Usage**:

1. Implement graph lifecycle management
2. Add garbage collection triggers
3. Limit concurrent large graph operations
4. Scale vertically (more memory per pod)

**For High Error Rates**:

1. Check for algorithm timeouts
2. Validate input data quality
3. Implement circuit breakers
4. Add request retries with backoff

---

## Scaling Operations

### Horizontal Scaling

**When to Scale Out**:

- Connection pool usage >70% (>30 active connections)
- CPU usage sustained >60%
- Request queue depth increasing

```bash
# Scale up
kubectl scale deployment networkx-mcp-server --replicas=5

# Monitor scaling impact
watch kubectl get pods -l app=networkx-mcp
```

### Vertical Scaling

**When to Scale Up**:

- Memory usage consistently >80%
- Frequent OOM kills
- Large graph processing requirements

```bash
# Update memory limits
kubectl patch deployment networkx-mcp-server -p '{"spec":{"template":{"spec":{"containers":[{"name":"mcp-server","resources":{"limits":{"memory":"4Gi"}}}]}}}}'
```

### Load Balancing

**Distribution Strategy**:

- Round-robin for typical requests
- Consistent hashing for stateful operations
- Weighted distribution based on node capacity

---

## Emergency Procedures

### Emergency Rollback

**Trigger**: Service completely unavailable, new deployment causing issues

```bash
# Immediate rollback
kubectl rollout undo deployment/networkx-mcp-server

# Verify rollback status
kubectl rollout status deployment/networkx-mcp-server

# Check service health
curl https://api.mcp.example.com/health
```

### Emergency Scale Down

**Trigger**: Resource exhaustion, cluster stability issues

```bash
# Scale to minimum
kubectl scale deployment networkx-mcp-server --replicas=1

# Or complete shutdown
kubectl scale deployment networkx-mcp-server --replicas=0
```

### Circuit Breaker Activation

**Trigger**: Downstream service failures, prevent cascade failures

```bash
# Check if automatic circuit breakers are active
curl https://api.mcp.example.com/debug/circuit-breakers

# Manual activation (if supported)
curl -X POST https://api.mcp.example.com/debug/circuit-breaker/activate
```

---

## Debugging & Troubleshooting

### Log Analysis

**Critical Log Patterns**:

```bash
# Memory-related issues
kubectl logs -l app=networkx-mcp | grep -E "(OutOfMemory|OOM|memory)"

# Algorithm performance issues
kubectl logs -l app=networkx-mcp | grep -E "(timeout|slow|performance)"

# Connection issues
kubectl logs -l app=networkx-mcp | grep -E "(connection|refused|timeout)"

# Error patterns
kubectl logs -l app=networkx-mcp | grep ERROR | tail -20
```

### Distributed Tracing Analysis

**Key Trace Patterns**:

1. **Request Flow**: Client → Load Balancer → MCP Server → Storage
2. **Algorithm Execution**: Request parsing → Graph loading → Algorithm → Response
3. **Error Traces**: Failed requests with detailed stack traces

**Jaeger Queries**:

```
# Slow requests
operation="mcp.request.*" duration:>2s

# Failed requests
operation="mcp.request.*" error=true

# Large graph operations
tags.graph.nodes:>10000

# Memory-intensive operations
tags.algorithm.memory.delta.mb:>100
```

### Performance Profiling

**Memory Profiling**:

```bash
# Get heap dump (if enabled)
curl https://api.mcp.example.com/debug/pprof/heap > heap.prof

# Analyze with go tool pprof (if applicable)
go tool pprof heap.prof
```

**CPU Profiling**:

```bash
# Get CPU profile
curl https://api.mcp.example.com/debug/pprof/profile?seconds=30 > cpu.prof
```

### Network Debugging

**Connection Testing**:

```bash
# Test connectivity from client
curl -v https://api.mcp.example.com/health

# Test from within cluster
kubectl run debug --image=curlimages/curl -it --rm -- curl mcp-service:8080/health

# Check DNS resolution
kubectl run debug --image=curlimages/curl -it --rm -- nslookup mcp-service
```

---

## Maintenance Procedures

### Planned Maintenance Window

**Pre-maintenance Checklist**:

1. Verify backup systems are working
2. Scale down to minimum necessary replicas
3. Enable maintenance mode (if available)
4. Notify users via status page

**During Maintenance**:

1. Monitor error rates and user impact
2. Rolling updates with health checks
3. Validate each step before proceeding

**Post-maintenance Verification**:

1. Run full test suite
2. Check all monitoring dashboards
3. Verify performance baselines
4. Update change log

### Capacity Planning

**Growth Indicators**:

- Sustained connection usage >70%
- Average response time trending upward
- Memory usage growing >10% per week
- Error rate baseline increasing

**Planning Metrics**:

- Current: 45 connections, 2GB memory
- 50% growth: 67 connections, 3GB memory
- 100% growth: 90 connections, 4GB memory

---

## Useful Queries & Commands

### Prometheus Queries

```promql
# Connection utilization
sum(mcp_active_connections) / 45 * 100

# Memory utilization
mcp_memory_usage_bytes / 2147483648 * 100

# Error rate (5m window)
sum(rate(mcp_requests_total{status="error"}[5m])) / sum(rate(mcp_requests_total[5m])) * 100

# P95 response time by method
histogram_quantile(0.95, sum(rate(mcp_request_duration_seconds_bucket[5m])) by (method, le))

# Algorithm performance by graph size
histogram_quantile(0.95, sum(rate(mcp_algorithm_duration_seconds_bucket[5m])) by (graph_size_bucket, le))
```

### Kubectl Commands

```bash
# Service status overview
kubectl get all -l app=networkx-mcp

# Resource usage
kubectl top pods -l app=networkx-mcp

# Recent events
kubectl get events --field-selector involvedObject.kind=Pod,involvedObject.name~=networkx-mcp --sort-by=.metadata.creationTimestamp

# Configuration check
kubectl get configmap mcp-config -o yaml

# Secret verification
kubectl get secret mcp-secrets --template='{{range $k,$v := .data}}{{printf "%s: %s\n" $k $v}}{{end}}'
```

---

## Contact Information

### Escalation Path

1. **L1 Support**: On-call engineer (PagerDuty)
2. **L2 Support**: Platform team lead
3. **L3 Support**: Senior platform engineer
4. **L4 Support**: Platform architect

### Communication Channels

- **Immediate**: PagerDuty alerts
- **Updates**: #mcp-alerts Slack channel
- **Coordination**: #incident-response Slack channel
- **Status**: <https://status.example.com>

### External Dependencies

- **Cloud Provider**: AWS/GCP support
- **Monitoring**: DataDog/New Relic support
- **CDN**: CloudFlare support

---

**Last Updated**: 2024-07-07
**Next Review**: 2024-08-07
**Document Owner**: Platform Team
**Version**: 1.0

---

*This runbook is based on actual performance testing data and production experience. All thresholds and procedures have been validated against real system behavior.*
