#!/bin/bash
# Setup monitoring infrastructure for NetworkX MCP Server
# This script deploys Prometheus, Grafana, and AlertManager configurations

set -euo pipefail

# Configuration
NAMESPACE="${MONITORING_NAMESPACE:-monitoring}"
PROMETHEUS_VERSION="${PROMETHEUS_VERSION:-v2.45.0}"
GRAFANA_VERSION="${GRAFANA_VERSION:-10.0.0}"
ALERT_MANAGER_VERSION="${ALERT_MANAGER_VERSION:-v0.25.0}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    command -v kubectl >/dev/null 2>&1 || error "kubectl is required but not installed"
    command -v helm >/dev/null 2>&1 || error "helm is required but not installed"

    # Check if we can connect to cluster
    kubectl cluster-info >/dev/null 2>&1 || error "Cannot connect to Kubernetes cluster"

    log "Prerequisites check passed"
}

# Create namespace
create_namespace() {
    log "Creating monitoring namespace..."

    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

    log "Namespace $NAMESPACE ready"
}

# Deploy Prometheus
deploy_prometheus() {
    log "Deploying Prometheus..."

    # Add Prometheus Helm repo
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update

    # Create Prometheus configuration
    cat > /tmp/prometheus-values.yaml << EOF
server:
  persistentVolume:
    size: 20Gi
  retention: "30d"

  global:
    scrape_interval: 30s
    evaluation_interval: 30s

  extraFlags:
    - web.enable-lifecycle
    - web.enable-admin-api

serverFiles:
  alerting_rules.yml:
$(cat monitoring/prometheus/alerts.yaml | sed 's/^/    /')

alertmanager:
  enabled: true
  config:
$(cat monitoring/prometheus/alertmanager.yaml | sed 's/^/    /')

pushgateway:
  enabled: false

nodeExporter:
  enabled: true

kubeStateMetrics:
  enabled: true
EOF

    # Deploy Prometheus
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace "$NAMESPACE" \
        --values /tmp/prometheus-values.yaml \
        --wait

    log "Prometheus deployed successfully"
}

# Deploy Grafana Dashboard
deploy_grafana_dashboard() {
    log "Deploying Grafana dashboard..."

    # Create dashboard ConfigMap
    kubectl create configmap mcp-dashboard \
        --from-file=monitoring/grafana/dashboard.json \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -

    # Create dashboard provisioning config
    cat > /tmp/dashboard-provider.yaml << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: dashboard-provider
  namespace: $NAMESPACE
data:
  dashboards.yaml: |
    apiVersion: 1
    providers:
    - name: 'mcp-dashboards'
      orgId: 1
      folder: 'MCP'
      type: file
      disableDeletion: false
      updateIntervalSeconds: 30
      options:
        path: /var/lib/grafana/dashboards/mcp
EOF

    kubectl apply -f /tmp/dashboard-provider.yaml

    log "Grafana dashboard configuration deployed"
}

# Setup ServiceMonitor for MCP Server
setup_service_monitor() {
    log "Setting up ServiceMonitor for MCP Server..."

    cat > /tmp/mcp-service-monitor.yaml << EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: networkx-mcp-server
  namespace: $NAMESPACE
  labels:
    app: networkx-mcp
    release: prometheus
spec:
  selector:
    matchLabels:
      app: networkx-mcp
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
    timeout: 15s
  namespaceSelector:
    matchNames:
    - default
EOF

    kubectl apply -f /tmp/mcp-service-monitor.yaml

    log "ServiceMonitor configured"
}

# Setup PrometheusRule for MCP alerts
setup_prometheus_rules() {
    log "Setting up Prometheus alerting rules..."

    kubectl create configmap mcp-alerts \
        --from-file=monitoring/prometheus/alerts.yaml \
        --namespace="$NAMESPACE" \
        --dry-run=client -o yaml | kubectl apply -f -

    # Apply the rules to Prometheus
    cat > /tmp/mcp-prometheus-rule.yaml << EOF
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: networkx-mcp-alerts
  namespace: $NAMESPACE
  labels:
    app: networkx-mcp
    release: prometheus
spec:
$(cat monitoring/prometheus/alerts.yaml | sed 's/^/  /')
EOF

    kubectl apply -f /tmp/mcp-prometheus-rule.yaml

    log "Prometheus alerting rules configured"
}

# Verify deployment
verify_deployment() {
    log "Verifying monitoring deployment..."

    # Check if Prometheus is running
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=prometheus \
        --namespace="$NAMESPACE" --timeout=300s

    # Check if Grafana is running
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=grafana \
        --namespace="$NAMESPACE" --timeout=300s

    # Check if AlertManager is running
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=alertmanager \
        --namespace="$NAMESPACE" --timeout=300s

    log "All monitoring components are ready"
}

# Get access information
get_access_info() {
    log "Getting access information..."

    echo ""
    echo "=== Monitoring Access Information ==="
    echo ""

    # Prometheus
    echo "🔥 Prometheus:"
    echo "   Port-forward: kubectl port-forward -n $NAMESPACE svc/prometheus-kube-prometheus-prometheus 9090:9090"
    echo "   URL: http://localhost:9090"
    echo ""

    # Grafana
    echo "📊 Grafana:"
    echo "   Port-forward: kubectl port-forward -n $NAMESPACE svc/prometheus-grafana 3000:80"
    echo "   URL: http://localhost:3000"
    echo "   Username: admin"
    echo "   Password: $(kubectl get secret -n $NAMESPACE prometheus-grafana -o jsonpath='{.data.admin-password}' | base64 -d)"
    echo ""

    # AlertManager
    echo "🚨 AlertManager:"
    echo "   Port-forward: kubectl port-forward -n $NAMESPACE svc/prometheus-kube-prometheus-alertmanager 9093:9093"
    echo "   URL: http://localhost:9093"
    echo ""

    echo "📋 Dashboard:"
    echo "   Import the NetworkX MCP dashboard from monitoring/grafana/dashboard.json"
    echo "   Or use the auto-provisioned dashboard in the 'MCP' folder"
    echo ""

    echo "✅ Monitoring setup complete!"
    echo ""
    echo "📖 Next steps:"
    echo "   1. Configure external endpoints (Slack, PagerDuty) in AlertManager"
    echo "   2. Import dashboard in Grafana"
    echo "   3. Test alerting with: kubectl delete pod -l app=networkx-mcp"
    echo "   4. Review runbook: docs/operations/runbook.md"
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    rm -f /tmp/prometheus-values.yaml
    rm -f /tmp/dashboard-provider.yaml
    rm -f /tmp/mcp-service-monitor.yaml
    rm -f /tmp/mcp-prometheus-rule.yaml
}

# Main execution
main() {
    log "Starting NetworkX MCP Server monitoring setup..."

    check_prerequisites
    create_namespace
    deploy_prometheus
    deploy_grafana_dashboard
    setup_service_monitor
    setup_prometheus_rules
    verify_deployment
    get_access_info
    cleanup

    log "Monitoring setup completed successfully!"
}

# Handle script interruption
trap cleanup EXIT

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --help)
            echo "Setup monitoring for NetworkX MCP Server"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --namespace NAMESPACE    Monitoring namespace (default: monitoring)"
            echo "  --help                   Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  MONITORING_NAMESPACE     Override default namespace"
            echo "  PROMETHEUS_VERSION       Prometheus version (default: v2.45.0)"
            echo "  GRAFANA_VERSION          Grafana version (default: 10.0.0)"
            echo "  ALERT_MANAGER_VERSION    AlertManager version (default: v0.25.0)"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Run main function
main "$@"
