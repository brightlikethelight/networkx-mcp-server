# Post-Deployment Checklist

## Overview

This checklist ensures that every production deployment of the NetworkX MCP Server is properly validated and monitored after deployment. Use this checklist immediately after each production rollout to verify system health and performance.

## Pre-Checklist Requirements

- [ ] Deployment completed successfully
- [ ] CI/CD pipeline passed all stages
- [ ] No active incidents or alerts
- [ ] Deployment notification sent to stakeholders

---

## 1. System Health Validation (First 5 minutes)

### Core Service Health
- [ ] **Health endpoint responding**: `curl https://api.mcp.example.com/health` returns 200
- [ ] **Readiness endpoint responding**: `curl https://api.mcp.example.com/ready` returns 200
- [ ] **Startup endpoint responding**: `curl https://api.mcp.example.com/startup` returns 200
- [ ] **All pods running**: `kubectl get pods -l app=networkx-mcp` shows all pods in Running state
- [ ] **No pod restarts**: Verify restart count is 0 for new deployment

### Load Balancer & Ingress
- [ ] **Ingress healthy**: `kubectl get ingress networkx-mcp-ingress` shows healthy status
- [ ] **SSL certificate valid**: `curl -vI https://api.mcp.example.com` shows valid TLS
- [ ] **DNS resolution**: `nslookup api.mcp.example.com` resolves correctly
- [ ] **Traffic routing**: Verify requests reach new pods via load balancer

### Database & Storage
- [ ] **Redis connectivity**: Verify connection to Redis backend
- [ ] **Storage accessible**: Confirm graph storage is working
- [ ] **No data loss**: Validate existing graphs are accessible
- [ ] **Backup systems operational**: Confirm backup processes running

---

## 2. Functional Testing (First 10 minutes)

### MCP Protocol Compliance
- [ ] **JSON-RPC endpoint working**: Basic JSON-RPC 2.0 request succeeds
  ```bash
  curl -X POST https://api.mcp.example.com/jsonrpc \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer TOKEN" \
    -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"list_graphs","arguments":{}},"id":1}'
  ```
- [ ] **Stdio mode working**: If enabled, verify stdio transport works
- [ ] **Tool discovery**: `list_tools` returns expected tools
- [ ] **Resource discovery**: `list_resources` returns expected resources
- [ ] **Prompt templates**: `list_prompts` returns expected prompts

### Core Graph Operations
- [ ] **Create graph**: Successfully create a new test graph
- [ ] **Add nodes**: Add nodes to the test graph
- [ ] **Add edges**: Add edges to the test graph
- [ ] **Query graph**: Retrieve graph information
- [ ] **Delete graph**: Clean up test graph
- [ ] **List graphs**: Verify graph listing works

### Authentication & Authorization
- [ ] **Valid token accepted**: Authenticated requests work
- [ ] **Invalid token rejected**: Returns 401 for invalid tokens
- [ ] **Missing token rejected**: Returns 401 for missing tokens
- [ ] **Rate limiting working**: Verify rate limits are enforced

---

## 3. Performance Validation (First 15 minutes)

### Response Time Metrics
- [ ] **Health endpoint < 100ms**: Health checks respond quickly
- [ ] **JSON-RPC requests < 2s**: API requests within SLA (P95 < 2000ms)
- [ ] **Graph operations < 5s**: Complex graph operations complete timely
- [ ] **Memory usage stable**: Memory consumption within expected ranges

### Concurrent Load Testing
- [ ] **Concurrent connections**: Test with 10 concurrent users
  ```bash
  python tests/performance/test_load_performance.py --users 10 --duration 60
  ```
- [ ] **Success rate > 95%**: Verify high success rate under load
- [ ] **No timeout errors**: All requests complete within timeout
- [ ] **Resource limits respected**: Memory and CPU within limits

### Capacity Validation
- [ ] **Graph size limits**: Test with graphs up to size limits
- [ ] **Connection limits**: Verify connection limit enforcement
- [ ] **Memory limits**: Confirm memory usage stays within bounds
- [ ] **Storage limits**: Validate storage usage is reasonable

---

## 4. Monitoring & Observability (First 20 minutes)

### Metrics Collection
- [ ] **Prometheus scraping**: `curl https://api.mcp.example.com:9090/metrics` returns metrics
- [ ] **Key metrics present**: Verify essential metrics are being collected:
  - `mcp_requests_total`
  - `mcp_request_duration_seconds`
  - `mcp_active_connections`
  - `mcp_graph_operations_total`
  - `process_resident_memory_bytes`
  - `process_cpu_seconds_total`

### Dashboard Validation
- [ ] **Grafana dashboards**: All dashboards show data from new deployment
- [ ] **Alert manager**: Verify AlertManager is receiving metrics
- [ ] **No firing alerts**: Confirm no unexpected alerts triggered
- [ ] **Metric trends**: Verify metrics trending as expected

### Distributed Tracing
- [ ] **Jaeger traces**: Traces appearing in Jaeger UI
- [ ] **Trace sampling**: Sampling rate appropriate (10%)
- [ ] **Span details**: Request spans contain expected information
- [ ] **Error traces**: Error conditions properly traced

### Log Aggregation
- [ ] **Structured logs**: JSON-formatted logs being collected
- [ ] **Log levels**: Appropriate log levels (INFO in production)
- [ ] **No error floods**: No excessive error logging
- [ ] **Log searchability**: Logs searchable in aggregation system

---

## 5. Security Validation (First 30 minutes)

### Security Headers
- [ ] **HTTPS enforcement**: HTTP requests redirect to HTTPS
- [ ] **Security headers present**: Required security headers in responses:
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: DENY`
  - `X-XSS-Protection: 1; mode=block`
  - `Strict-Transport-Security: max-age=31536000`

### Input Validation
- [ ] **Malformed requests rejected**: Invalid JSON-RPC requests return errors
- [ ] **XSS protection**: XSS attempts properly sanitized
- [ ] **Injection protection**: SQL/NoSQL injection attempts blocked
- [ ] **Large payload handling**: Oversized requests properly rejected

### Access Control
- [ ] **RBAC functioning**: Role-based access control working
- [ ] **Network policies**: Kubernetes network policies enforced
- [ ] **Resource isolation**: Proper resource isolation between namespaces
- [ ] **Audit logging**: Security events being logged

---

## 6. Feature Flag Validation (First 30 minutes)

### Production Feature Flags
- [ ] **Critical flags enabled**: Security and monitoring flags enabled
- [ ] **Rollout flags status**: Gradual rollout flags at expected percentages
- [ ] **Performance gates**: Performance-gated features working correctly
- [ ] **Flag metrics**: Feature flag usage metrics being collected

### Rollout Status Check
```bash
# Check current rollout status
python -c "
from src.networkx_mcp.features.production_flags import get_production_feature_manager
manager = get_production_feature_manager()
print(manager.get_rollout_status())
"
```

- [ ] **Rollout percentages**: Verify expected rollout percentages
- [ ] **User targeting**: User-based targeting working correctly
- [ ] **Environment targeting**: Production-specific flags active
- [ ] **Performance monitoring**: Flag performance being tracked

---

## 7. Backup & Recovery Validation (First 45 minutes)

### Data Backup
- [ ] **Backup creation**: New backup created post-deployment
- [ ] **Backup integrity**: Backup passes integrity checks
- [ ] **Backup accessibility**: Backup can be accessed when needed
- [ ] **Retention policies**: Old backups cleaned up per policy

### Disaster Recovery
- [ ] **Recovery procedures**: Recovery runbooks up to date
- [ ] **RTO/RPO targets**: Recovery targets documented and achievable
- [ ] **Cross-region backup**: If applicable, cross-region backups working
- [ ] **Recovery testing**: Last recovery test results reviewed

---

## 8. User Acceptance Testing (First 60 minutes)

### End-to-End Workflows
- [ ] **Basic user workflow**: Complete a typical user workflow
- [ ] **Advanced features**: Test advanced graph algorithms
- [ ] **Error handling**: Verify graceful error handling
- [ ] **Performance**: User-facing performance meets expectations

### Integration Testing
- [ ] **MCP clients**: Test with common MCP client implementations
- [ ] **SDK compatibility**: Verify SDK compatibility
- [ ] **API versioning**: Confirm API version compatibility
- [ ] **Third-party tools**: Test with integrated tools

---

## 9. Communication & Documentation (First 90 minutes)

### Stakeholder Communication
- [ ] **Deployment notification**: Send deployment success notification
- [ ] **Status page update**: Update public status page if applicable
- [ ] **Team notification**: Notify relevant teams of successful deployment
- [ ] **Customer communication**: If needed, notify customers of updates

### Documentation Updates
- [ ] **Version documentation**: Update version documentation
- [ ] **API documentation**: Verify API docs are current
- [ ] **Operational runbooks**: Update operational procedures
- [ ] **Change log**: Add deployment to change log

---

## 10. Post-Deployment Monitoring (Ongoing)

### Immediate Monitoring (First 2 hours)
- [ ] **Monitor alerts**: Watch for any unusual alerts
- [ ] **Track error rates**: Monitor error rate trends
- [ ] **Watch performance**: Observe performance metrics
- [ ] **Customer feedback**: Monitor for user-reported issues

### Extended Monitoring (First 24 hours)
- [ ] **Traffic patterns**: Analyze traffic pattern changes
- [ ] **Resource utilization**: Monitor resource usage trends
- [ ] **Feature adoption**: Track new feature usage
- [ ] **Performance baselines**: Establish new performance baselines

### Weekly Follow-up
- [ ] **Performance review**: Review week-over-week performance
- [ ] **Error analysis**: Analyze any errors or issues
- [ ] **Capacity planning**: Update capacity plans based on usage
- [ ] **Lessons learned**: Document lessons learned from deployment

---

## Emergency Procedures

### If Issues Are Discovered

#### Severity 1 (Critical - Service Down)
1. **Immediate Action**: Execute emergency rollback
   ```bash
   kubectl rollout undo deployment/networkx-mcp-server
   ```
2. **Incident Response**: Activate incident response team
3. **Communication**: Notify all stakeholders immediately
4. **Investigation**: Begin root cause analysis

#### Severity 2 (Major - Degraded Performance)
1. **Assessment**: Evaluate impact and affected users
2. **Mitigation**: Apply immediate mitigations (scaling, config changes)
3. **Monitoring**: Increase monitoring frequency
4. **Decision**: Decide on rollback vs. hotfix

#### Severity 3 (Minor - Non-Critical Issues)
1. **Documentation**: Log issues for next maintenance window
2. **Monitoring**: Add specific monitoring for the issue
3. **Planning**: Plan fix for next deployment
4. **Communication**: Inform team of known issues

---

## Checklist Sign-off

### Deployment Information
- **Deployment Date**: ________________
- **Version Deployed**: ________________
- **Deployment Engineer**: ________________
- **Deployment Duration**: ________________

### Validation Results
- **Health Checks**: ✅ Pass / ❌ Fail
- **Functional Tests**: ✅ Pass / ❌ Fail  
- **Performance Tests**: ✅ Pass / ❌ Fail
- **Security Tests**: ✅ Pass / ❌ Fail
- **Monitoring Setup**: ✅ Pass / ❌ Fail

### Sign-offs
- **Deployment Engineer**: ________________ Date: ________
- **SRE Lead**: ________________ Date: ________
- **Engineering Manager**: ________________ Date: ________

### Notes
```
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________
```

---

**Last Updated**: 2024-07-07  
**Document Owner**: SRE Team  
**Review Cycle**: After each deployment  
**Template Version**: 1.0