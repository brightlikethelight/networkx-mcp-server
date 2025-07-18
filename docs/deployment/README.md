# 🚀 NetworkX MCP Server Deployment Guide

This comprehensive guide covers all deployment options for the NetworkX MCP Server, from development to production environments.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Environment Requirements](#environment-requirements)
- [Deployment Options](#deployment-options)
- [Configuration Management](#configuration-management)
- [Security Considerations](#security-considerations)
- [Monitoring & Observability](#monitoring--observability)
- [Scaling & Performance](#scaling--performance)
- [Troubleshooting](#troubleshooting)

## 🏃‍♂️ Quick Start

### Docker Compose (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/your-org/networkx-mcp-server.git
cd networkx-mcp-server

# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f networkx-mcp
```

### Local Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Run with development settings
export APP_ENV=development
python -m networkx_mcp

# Or use the development script
./scripts/dev_setup.py --run
```

## 📊 Environment Requirements

### Minimum Requirements

| Component | Minimum | Recommended | Production |
|-----------|---------|-------------|------------|
| CPU | 2 cores | 4 cores | 8+ cores |
| RAM | 4 GB | 8 GB | 16+ GB |
| Storage | 10 GB | 50 GB | 200+ GB |
| Python | 3.11+ | 3.11+ | 3.11+ |

### Dependencies

#### Required Services

- **Redis**: Caching and session storage
- **PostgreSQL**: Primary database (optional)
- **Nginx**: Reverse proxy and load balancing

#### Optional Services

- **Prometheus**: Metrics collection
- **Grafana**: Metrics visualization
- **Jaeger**: Distributed tracing
- **ELK Stack**: Centralized logging

## 🐳 Deployment Options

### 1. Docker Compose

**Best for**: Development, testing, small production deployments

```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# With monitoring stack
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

# Scale the application
docker-compose up -d --scale networkx-mcp=3
```

#### Configuration Files

- `docker-compose.yml` - Development setup
- `docker-compose.prod.yml` - Production optimized
- `docker-compose.monitoring.yml` - Monitoring stack
- `docker-compose.override.yml` - Local customizations

### 2. Kubernetes

**Best for**: Production, high availability, auto-scaling

#### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify cluster access
kubectl cluster-info
```

#### Deploy with Kubectl

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n networkx-mcp

# Scale deployment
kubectl scale deployment networkx-mcp --replicas=5 -n networkx-mcp
```

#### Deploy with Helm

```bash
# Add Helm repository (if external)
helm repo add networkx-mcp https://charts.networkx-mcp.com
helm repo update

# Install with custom values
helm install networkx-mcp ./helm/networkx-mcp \
  --namespace networkx-mcp \
  --create-namespace \
  --values helm/networkx-mcp/values.prod.yaml

# Upgrade deployment
helm upgrade networkx-mcp ./helm/networkx-mcp \
  --namespace networkx-mcp \
  --values helm/networkx-mcp/values.prod.yaml
```

### 3. Cloud Platforms

#### AWS ECS

```bash
# Build and push to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com
docker build -t networkx-mcp .
docker tag networkx-mcp:latest 123456789012.dkr.ecr.us-west-2.amazonaws.com/networkx-mcp:latest
docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/networkx-mcp:latest

# Deploy with Terraform
cd infrastructure/aws/ecs
terraform init
terraform plan
terraform apply
```

#### Google Cloud Run

```bash
# Deploy to Cloud Run
gcloud run deploy networkx-mcp \
  --image gcr.io/PROJECT-ID/networkx-mcp \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars APP_ENV=production
```

#### Azure Container Instances

```bash
# Deploy to ACI
az container create \
  --resource-group networkx-mcp-rg \
  --name networkx-mcp \
  --image your-registry.azurecr.io/networkx-mcp:latest \
  --dns-name-label networkx-mcp \
  --ports 8000
```

## ⚙️ Configuration Management

### Environment Variables

#### Core Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `APP_ENV` | Environment (dev/staging/prod) | `production` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `HOST` | Server host | `0.0.0.0` | No |
| `PORT` | Server port | `8000` | No |
| `WORKERS` | Worker processes | `4` | No |

#### Database Configuration

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `POSTGRES_URL` | PostgreSQL connection | `postgresql://user:pass@host:5432/db` | No |
| `REDIS_URL` | Redis connection | `redis://host:6379/0` | No |

#### Security Configuration

| Variable | Description | Required |
|----------|-------------|----------|
| `JWT_SECRET` | JWT signing secret | Yes (Production) |
| `API_KEY` | API authentication key | No |
| `RATE_LIMIT_ENABLED` | Enable rate limiting | No |

### Configuration Files

#### Development (`config/development.yaml`)

```yaml
app:
  debug: true
  reload: true

server:
  workers: 1

logging:
  level: DEBUG

features:
  rate_limiting: false
  caching: false
```

#### Production (`config/production.yaml`)

```yaml
app:
  debug: false
  reload: false

server:
  workers: 4
  timeout: 30

logging:
  level: INFO
  format: json

features:
  rate_limiting: true
  caching: true
  monitoring: true
```

### Secrets Management

#### Kubernetes Secrets

```bash
# Create secrets
kubectl create secret generic networkx-mcp-secrets \
  --from-literal=jwt-secret=your-jwt-secret \
  --from-literal=postgres-password=your-password \
  -n networkx-mcp

# Use external secret management
kubectl apply -f k8s/external-secrets.yaml
```

#### Docker Secrets

```bash
# Create Docker secrets
echo "your-jwt-secret" | docker secret create jwt_secret -
echo "your-db-password" | docker secret create db_password -

# Use in compose file
version: '3.8'
services:
  networkx-mcp:
    secrets:
      - jwt_secret
      - db_password
```

## 🔒 Security Considerations

### Network Security

#### Firewall Rules

```bash
# Allow only necessary ports
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw deny 8000/tcp   # Block direct app access
```

#### TLS Configuration

```yaml
# nginx.conf
server {
    listen 443 ssl http2;
    ssl_certificate /etc/ssl/certs/networkx-mcp.crt;
    ssl_certificate_key /etc/ssl/private/networkx-mcp.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
}
```

### Application Security

#### Security Headers

```python
# Implemented in application
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'"
}
```

#### Authentication & Authorization

```bash
# Generate JWT secret
openssl rand -hex 32

# Set up API key authentication
export API_KEY=$(openssl rand -hex 16)
```

### Container Security

#### Security Scanning

```bash
# Scan container images
trivy image networkx-mcp:latest

# Continuous scanning in CI
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image networkx-mcp:latest
```

#### Runtime Security

```yaml
# Pod Security Context
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  fsGroup: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
```

## 📊 Monitoring & Observability

### Metrics Collection

#### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'networkx-mcp'
    static_configs:
      - targets: ['networkx-mcp:9090']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

#### Key Metrics

- **Application Metrics**: Request rate, response time, error rate
- **System Metrics**: CPU, memory, disk usage
- **Business Metrics**: Graph operations, algorithm executions
- **Custom Metrics**: Cache hit rate, queue depth

### Logging

#### Centralized Logging

```yaml
# filebeat.yml
filebeat.inputs:
- type: container
  paths:
    - '/var/lib/docker/containers/*/*.log'
  processors:
    - add_docker_metadata: ~

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
```

#### Log Levels

- `DEBUG`: Development debugging
- `INFO`: General information
- `WARNING`: Warning conditions
- `ERROR`: Error conditions
- `CRITICAL`: Critical errors

### Distributed Tracing

#### Jaeger Configuration

```yaml
# jaeger.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: jaeger-config
data:
  jaeger.yaml: |
    span_storage_type: elasticsearch
    es_config:
      server_urls: http://elasticsearch:9200
```

### Health Checks

#### Application Health

```bash
# Health check endpoint
curl -f http://localhost:8000/health

# Readiness check
curl -f http://localhost:8000/ready

# Detailed status
curl -f http://localhost:8000/status
```

#### Infrastructure Health

```bash
# Database connectivity
kubectl exec -it postgres-pod -- pg_isready -U networkx

# Redis connectivity
kubectl exec -it redis-pod -- redis-cli ping
```

## 📈 Scaling & Performance

### Horizontal Scaling

#### Kubernetes HPA

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: networkx-mcp-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: networkx-mcp
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

#### Docker Swarm

```bash
# Scale service
docker service scale networkx-mcp=5

# Update service with zero downtime
docker service update --image networkx-mcp:2.1.0 networkx-mcp
```

### Vertical Scaling

#### Resource Optimization

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

### Performance Tuning

#### Application Tuning

```python
# gunicorn configuration
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
```

#### Database Optimization

```sql
-- PostgreSQL optimizations
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET random_page_cost = 1.1;
SELECT pg_reload_conf();
```

#### Cache Configuration

```yaml
redis:
  maxmemory: 512mb
  maxmemory-policy: allkeys-lru
  save: "900 1 300 10 60 10000"
```

## 🔧 Troubleshooting

### Common Issues

#### Container Issues

```bash
# Check container logs
docker logs networkx-mcp

# Debug container
docker exec -it networkx-mcp /bin/sh

# Inspect container
docker inspect networkx-mcp
```

#### Kubernetes Issues

```bash
# Check pod status
kubectl describe pod networkx-mcp-xxx -n networkx-mcp

# View events
kubectl get events -n networkx-mcp --sort-by='.lastTimestamp'

# Debug networking
kubectl exec -it networkx-mcp-xxx -n networkx-mcp -- wget -qO- http://redis-service:6379
```

#### Performance Issues

```bash
# CPU profiling
docker exec networkx-mcp python -m cProfile -o profile.stats -m networkx_mcp

# Memory profiling
docker exec networkx-mcp python -m memory_profiler -m networkx_mcp

# Network debugging
docker exec networkx-mcp netstat -tulpn
```

### Debugging Tools

#### Application Debugging

```python
# Enable debug mode
import networkx_mcp
networkx_mcp.set_debug_mode(True)

# Profile specific operations
from networkx_mcp.monitoring import profile
with profile("graph_operation"):
    result = create_graph("test")
```

#### Infrastructure Debugging

```bash
# System resources
htop
iotop
nethogs

# Container resources
docker stats

# Kubernetes resources
kubectl top pods -n networkx-mcp
kubectl top nodes
```

### Recovery Procedures

#### Database Recovery

```bash
# PostgreSQL backup
pg_dump networkx_mcp > backup.sql

# PostgreSQL restore
psql networkx_mcp < backup.sql

# Redis backup
redis-cli BGSAVE
```

#### Application Recovery

```bash
# Rolling restart (Kubernetes)
kubectl rollout restart deployment/networkx-mcp -n networkx-mcp

# Graceful restart (Docker)
docker-compose restart networkx-mcp

# Emergency stop
kubectl delete pod -l app=networkx-mcp -n networkx-mcp
```

## 📚 Additional Resources

### Documentation Links

- [Architecture Overview](../ARCHITECTURE.md)
- [Development Guide](../DEVELOPMENT_GUIDE.md)
- [API Documentation](../api/)
- [Configuration Reference](../configuration.md)

### External Resources

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Production Best Practices](https://kubernetes.io/docs/setup/best-practices/)
- [Security Hardening Guide](https://kubernetes.io/docs/concepts/security/)
- [Monitoring Best Practices](https://prometheus.io/docs/practices/)

### Community & Support

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Comprehensive guides and tutorials
- **Community Forum**: Ask questions and share knowledge
- **Slack Channel**: Real-time community support

---

**Need help?** Check our [troubleshooting guide](./troubleshooting.md) or reach out to the community!
