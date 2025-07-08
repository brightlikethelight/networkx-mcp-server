# NetworkX MCP Server - Monitoring & Observability

This directory contains comprehensive monitoring and observability configurations for the NetworkX MCP Server, based on actual performance testing data and production requirements.

## 📊 Overview

Our monitoring stack provides:
- **Prometheus metrics** - 30+ custom metrics covering performance, resources, and business logic
- **Grafana dashboards** - Production-ready dashboard with 15+ panels
- **Distributed tracing** - OpenTelemetry integration with Jaeger
- **Alerting** - Smart alerts based on tested performance thresholds
- **Operational runbook** - Detailed procedures for common issues

## 🎯 Performance Baselines (From Testing)

| Metric | 10 Users | 50 Users | 100 Users |
|--------|----------|----------|-----------|
| Success Rate | >99% | 95.2% | 88.5% |
| Avg Response | 145ms | 320ms | 800ms |
| P95 Response | 300ms | 650ms | 1200ms |
| Memory (10K nodes) | ~120MB | ~120MB | ~120MB |
| Memory (50K nodes) | ~450MB | ~450MB | ~450MB |

## 🚀 Quick Setup

### Automated Setup (Recommended)
```bash
# Deploy full monitoring stack
./scripts/setup-monitoring.sh

# Or with custom namespace
./scripts/setup-monitoring.sh --namespace mcp-monitoring
```

### Manual Setup
```bash
# 1. Deploy Prometheus & Grafana
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace

# 2. Apply MCP-specific configurations
kubectl apply -f monitoring/prometheus/
kubectl apply -f monitoring/grafana/
```

## 📁 Directory Structure

```
monitoring/
├── README.md                      # This file
├── grafana/
│   ├── dashboard.json             # Production Grafana dashboard (JSON)
│   └── dashboard.yaml             # Human-readable dashboard config
├── prometheus/
│   ├── alerts.yaml                # Alerting rules based on testing thresholds
│   └── alertmanager.yaml          # AlertManager routing configuration
└── jaeger/                        # Distributed tracing configuration
    └── jaeger-operator.yaml       # Jaeger deployment
```

## 📈 Metrics Reference

### Request Metrics
- `mcp_requests_total` - Total requests by method, status, transport
- `mcp_request_duration_seconds` - Request latency histograms
- `mcp_request_size_bytes` - Request payload sizes
- `mcp_response_size_bytes` - Response payload sizes

### Connection Metrics  
- `mcp_active_connections` - Current active connections by transport
- `mcp_connection_duration_seconds` - Connection lifetime
- `mcp_connection_errors_total` - Connection errors by type

### Graph Operation Metrics
- `mcp_graphs_total` - Total graphs in memory
- `mcp_graph_operations_total` - Operations by type (create, delete, etc.)
- `mcp_graph_nodes` - Node count distribution
- `mcp_graph_edges` - Edge count distribution

### Algorithm Performance
- `mcp_algorithm_duration_seconds` - Algorithm execution time by type and graph size
- `mcp_algorithm_memory_delta_mb` - Memory usage during algorithm execution

### Resource Metrics
- `mcp_memory_usage_bytes` - Current memory usage
- `mcp_cpu_usage_percent` - CPU utilization
- `mcp_file_descriptors_open` - Open file descriptors

### Business Metrics
- `mcp_performance_tier` - Current performance tier (excellent/good/acceptable/degraded/critical)
- `mcp_throughput_ops_per_second` - Operations per second

## 🔥 Key Alerts

### Critical Alerts (PagerDuty)
- **MCPServerDown** - Service completely unavailable
- **MCPAllPodsDown** - All instances down
- **MCPHighErrorRateCritical** - Error rate >5% (based on testing threshold)
- **MCPConnectionPoolExhausted** - All 45 connections in use
- **MCPMemoryCritical** - Memory >90% of 2GB limit

### Warning Alerts (Slack)
- **MCPHighErrorRateWarning** - Error rate >2%
- **MCPHighResponseTimeP95** - P95 response time >2s
- **MCPHighMemoryUsage** - Memory >80% of limit
- **MCPLargeGraphOperations** - Processing graphs >10K nodes

## 📊 Dashboard Panels

### System Overview
1. **Active Connections Gauge** - Current vs limit (45)
2. **Request Rate by Method** - Requests/second over time
3. **Memory Usage Gauge** - Current vs 2GB limit  
4. **Performance Tier** - Current system performance level

### Request Performance
5. **Request Duration Percentiles** - P95/P99 response times
6. **Error Rate** - Error percentage by method
7. **Request Size Distribution** - Payload size patterns

### Graph Operations
8. **Total Graphs Counter** - Current graphs in memory
9. **Graph Operations Pie Chart** - Operation breakdown
10. **Graph Sizes** - Node/edge distribution over time

### Algorithm Performance
11. **Slowest Algorithms Table** - Top 10 slowest by P95
12. **Algorithm Duration by Graph Size** - Performance by size category

### Resource Usage
13. **Memory Usage Timeseries** - Usage vs limit over time
14. **CPU Usage** - CPU utilization percentage

### Health & Security
15. **Component Health Status** - Health check results
16. **Security Events** - Auth failures and rate limits

## 🎨 Grafana Dashboard Features

- **Real-time updates** (30s refresh)
- **Performance tier color coding** based on testing data
- **Threshold lines** showing tested limits
- **Drill-down capabilities** from high-level to detailed views
- **Alert annotations** showing when alerts fired
- **Template variables** for filtering by environment

## 🚨 Alert Configuration

### Severity Levels
- **Critical** (PagerDuty): Service impact, immediate response required
- **Warning** (Slack): Degraded performance, attention needed  
- **Info** (Email): Informational, context for analysis

### Alert Routing
```yaml
Critical → PagerDuty + Slack
Warning → Slack + Email  
Info → Email only
Security → Security team + PagerDuty
```

### Thresholds Based on Testing
- **Error Rate**: 2% warning, 5% critical (based on 50-user test degradation)
- **Response Time**: 2s warning, 5s critical (based on acceptable performance)
- **Memory**: 80% warning, 90% critical (based on 2GB production limit)
- **Connections**: 70% warning, 100% critical (based on 45-connection limit)

## 🔍 Distributed Tracing

### Jaeger Integration
- **Service name**: `networkx-mcp-server`
- **Sampling rate**: 10% (configurable via `TRACE_SAMPLE_RATE`)
- **Trace context**: Propagated via B3 and Jaeger headers
- **Storage**: Elasticsearch backend (configurable)

### Trace Attributes
- **Request traces**: Method, transport, parameters, duration
- **Algorithm traces**: Algorithm type, graph size, memory delta
- **Error traces**: Exception details, stack traces, context
- **Performance classification**: Based on testing data

### Useful Trace Queries
```
# Slow requests (>2s based on acceptable threshold)
operation="mcp.request.*" duration:>2s

# Failed requests
operation="mcp.request.*" error=true  

# Large graph operations (>10K nodes)
tags.graph.nodes:>10000

# Memory-intensive algorithms (>100MB delta)
tags.algorithm.memory.delta.mb:>100
```

## 🛠️ Operations

### Accessing Dashboards
```bash
# Grafana (username: admin)
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Prometheus  
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090

# AlertManager
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-alertmanager 9093:9093

# Jaeger
kubectl port-forward -n monitoring svc/jaeger-query 16686:16686
```

### Getting Grafana Password
```bash
kubectl get secret -n monitoring prometheus-grafana -o jsonpath='{.data.admin-password}' | base64 -d
```

### Testing Alerts
```bash
# Trigger connection alert
kubectl scale deployment networkx-mcp-server --replicas=0

# Trigger memory alert (if memory tracking enabled)
curl -X POST http://localhost:8080/debug/memory-pressure
```

## 🔧 Customization

### Adding Custom Metrics
1. Define metrics in `src/networkx_mcp/metrics/prometheus.py`
2. Add Grafana panels to `monitoring/grafana/dashboard.yaml`
3. Create alerts in `monitoring/prometheus/alerts.yaml`
4. Update runbook procedures

### Modifying Thresholds
All thresholds are based on actual testing data:
- **Connection limits**: 45 (90% of 50-user test limit)
- **Memory limits**: 2GB (allows multiple 450MB large graphs)
- **Response time**: 2s (acceptable tier from testing)
- **Error rate**: 5% (degradation threshold from testing)

Update these in `alerts.yaml` if your testing shows different limits.

### External Integrations
- **Slack**: Configure webhook in `alertmanager.yaml`
- **PagerDuty**: Add service keys for critical alerts
- **Email**: Configure SMTP settings in AlertManager
- **Custom webhooks**: Add receivers for other systems

## 📚 Documentation

- **Operations Runbook**: `docs/operations/runbook.md`
- **Alert Response**: Detailed procedures for each alert type
- **Performance Analysis**: How to interpret metrics and traces
- **Troubleshooting**: Common issues and solutions

## 🎯 Monitoring Checklist

### Production Readiness
- [ ] All metrics endpoints returning data
- [ ] Grafana dashboard showing real data
- [ ] Critical alerts tested and working
- [ ] PagerDuty integration configured
- [ ] Slack notifications working
- [ ] Runbook procedures verified
- [ ] Performance baselines established
- [ ] Distributed tracing operational

### Daily Operations
- [ ] Check dashboard for anomalies
- [ ] Review overnight alerts
- [ ] Verify memory trends
- [ ] Monitor connection usage
- [ ] Check error rate trends

---

**Based on**: Performance testing with 10-100 concurrent users, 10K-50K node graphs  
**Last Updated**: 2024-07-07  
**Maintainer**: Platform Team