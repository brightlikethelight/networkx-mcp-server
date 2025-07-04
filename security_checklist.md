# 🔐 Security Deployment Checklist

Use this checklist before deploying NetworkX MCP Server to production.

## Pre-Deployment Security Checklist

### 🔑 Secrets Management
- [ ] Create `.env` file from `.env.example`
- [ ] Generate strong passwords (minimum 32 characters)
  ```bash
  # Generate secure passwords
  openssl rand -base64 32  # For POSTGRES_PASSWORD
  openssl rand -base64 32  # For JWT_SECRET
  openssl rand -base64 32  # For REDIS_PASSWORD
  ```
- [ ] Set all required environment variables
- [ ] Verify `.env` is in `.gitignore`
- [ ] Never commit `.env` file
- [ ] Use secrets management service (Vault, AWS Secrets Manager, etc.)

### 🛡️ Network Security
- [ ] Deploy behind HTTPS reverse proxy
- [ ] Configure TLS certificates (Let's Encrypt recommended)
- [ ] Set up firewall rules
  ```bash
  # Example UFW rules
  sudo ufw allow 443/tcp  # HTTPS
  sudo ufw allow 80/tcp   # HTTP redirect
  sudo ufw deny 8000/tcp  # Block direct MCP access
  ```
- [ ] Enable CORS policies if needed
- [ ] Configure security headers in reverse proxy
  ```nginx
  # Nginx security headers
  add_header X-Frame-Options "SAMEORIGIN";
  add_header X-Content-Type-Options "nosniff";
  add_header X-XSS-Protection "1; mode=block";
  add_header Strict-Transport-Security "max-age=31536000";
  ```

### 🔐 Authentication & Authorization
- [ ] Implement authentication layer (required for production)
  - [ ] JWT tokens
  - [ ] API keys
  - [ ] OAuth2/OIDC
- [ ] Set up rate limiting in reverse proxy
  ```nginx
  # Nginx rate limiting
  limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
  limit_req zone=api burst=20 nodelay;
  ```
- [ ] Configure IP allowlisting if needed
- [ ] Implement user roles and permissions

### 📊 Resource Limits
- [ ] Configure memory limits
  ```bash
  export MAX_MEMORY_MB=2048        # Adjust based on server
  export MAX_GRAPH_SIZE_MB=200     # Per-graph limit
  ```
- [ ] Set operation timeouts
  ```bash
  export OPERATION_TIMEOUT=60      # 60 seconds
  ```
- [ ] Configure concurrent request limits
  ```bash
  export MAX_CONCURRENT_REQUESTS=20
  ```
- [ ] Set rate limiting
  ```bash
  export REQUESTS_PER_MINUTE=120
  ```

### 🐳 Container Security (Docker/Kubernetes)
- [ ] Use non-root user in container
  ```dockerfile
  USER 1000:1000
  ```
- [ ] Set read-only root filesystem
- [ ] Drop unnecessary capabilities
- [ ] Scan images for vulnerabilities
  ```bash
  docker scan networkx-mcp:latest
  ```
- [ ] Use minimal base images (alpine)
- [ ] Keep base images updated

### ☸️ Kubernetes Security
- [ ] Apply NetworkPolicy
- [ ] Use PodSecurityPolicy/PodSecurityStandards
- [ ] Enable RBAC
- [ ] Set resource limits and requests
- [ ] Use namespaces for isolation
- [ ] Encrypt secrets at rest
- [ ] Regular security updates

### 📝 Logging & Monitoring
- [ ] Enable security event logging
- [ ] Set up log aggregation (ELK, Splunk)
- [ ] Configure alerts for:
  - [ ] Failed authentication attempts
  - [ ] Rate limit violations
  - [ ] Memory/resource exhaustion
  - [ ] Unusual request patterns
- [ ] Enable audit logging
- [ ] Set up intrusion detection

### 🔍 Validation & Testing
- [ ] Run security tests
  ```bash
  python -m pytest tests/security/ -v
  ```
- [ ] Test input validation
  ```bash
  python tests/security/test_malicious_demo.py
  ```
- [ ] Test DoS prevention
  ```bash
  python tests/security/test_dos_prevention_demo.py
  ```
- [ ] Perform penetration testing
- [ ] Run dependency vulnerability scan
  ```bash
  pip audit
  safety check
  ```

### 🚀 Deployment Configuration
- [ ] Set production environment
  ```bash
  export APP_ENV=production
  export DEBUG=false
  ```
- [ ] Disable debug endpoints
- [ ] Remove development dependencies
- [ ] Enable production logging
- [ ] Configure health checks
- [ ] Set up automated backups (if using persistent storage)

### 📋 Operational Security
- [ ] Document security procedures
- [ ] Set up security incident response plan
- [ ] Configure automated security updates
- [ ] Regular security audits schedule
- [ ] Security training for operators
- [ ] Access control for production systems

### 🔄 Post-Deployment
- [ ] Verify all security measures active
- [ ] Run security scan
- [ ] Monitor for first 24 hours
- [ ] Document any deviations
- [ ] Schedule regular reviews

## Security Contact

Report security issues to: security@networkx-mcp.example.com

## Quick Security Test Commands

```bash
# Test authentication (should fail without auth)
curl -X POST http://localhost:8000/tool/create_graph \
  -H "Content-Type: application/json" \
  -d '{"name": "test"}'

# Test rate limiting (should be limited)
for i in {1..100}; do
  curl -X GET http://localhost:8000/tool/list_graphs &
done

# Test input validation (should be blocked)
curl -X POST http://localhost:8000/tool/create_graph \
  -H "Content-Type: application/json" \
  -d '{"name": "../../../etc/passwd"}'

# Check resource status
curl -X GET http://localhost:8000/tool/resource_status
```

## Emergency Procedures

### If Compromised:
1. Immediately shut down the service
2. Revoke all credentials
3. Review logs for extent of breach
4. Notify security team
5. Follow incident response plan

### If Under Attack:
1. Enable emergency rate limiting
2. Block attacking IPs
3. Scale down to essential services
4. Monitor resource usage
5. Prepare incident report

---

**Remember**: Security is not a one-time task. Regular reviews and updates are essential.