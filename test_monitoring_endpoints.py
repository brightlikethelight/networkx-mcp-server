#!/usr/bin/env python3
"""Test script to verify monitoring endpoints work correctly."""

import json
import time
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from networkx_mcp.server import (
    health_check,
    readiness_check,
    get_metrics,
    get_performance_stats,
    manage_feature_flags
)

def test_health_endpoint():
    """Test the health check endpoint."""
    print("🔍 Testing Health Check Endpoint")
    print("=" * 50)
    
    # Test with details
    result = health_check(include_details=True)
    print("Health Check (with details):")
    print(f"  Status: {result['status']}")
    print(f"  Uptime: {result['uptime_seconds']}s")
    print(f"  Total checks: {result['summary']['total_checks']}")
    print(f"  Healthy: {result['summary']['healthy']}")
    print(f"  Degraded: {result['summary']['degraded']}")
    print(f"  Unhealthy: {result['summary']['unhealthy']}")
    
    # Show component statuses
    print("\n  Component Status:")
    for component, details in result['checks'].items():
        status_emoji = "✅" if details['status'] == 'healthy' else "⚠️" if details['status'] == 'degraded' else "❌"
        print(f"    {status_emoji} {component}: {details['status']}")
        if details['status'] != 'healthy' and 'error' in details:
            print(f"       Error: {details['error']}")
    
    # Test without details
    result_minimal = health_check(include_details=False)
    print(f"\nHealth Check (minimal): {result_minimal['status']}")
    
    return result['status'] == 'healthy' or result['status'] == 'degraded'


def test_readiness_endpoint():
    """Test the readiness check endpoint."""
    print("\n🚀 Testing Readiness Check Endpoint")
    print("=" * 50)
    
    result = readiness_check()
    print(f"Readiness Status: {'✅ READY' if result['ready'] else '❌ NOT READY'}")
    print(f"Check Duration: {result['check_duration_ms']}ms")
    
    if not result['ready']:
        print("Failed Checks:")
        for check in result['failed_checks']:
            print(f"  ❌ {check}")
    
    # Show detailed check results
    print("\nDetailed Check Results:")
    for check_name, check_result in result['checks'].items():
        status_emoji = "✅" if check_result['status'] == 'ready' else "❌"
        print(f"  {status_emoji} {check_name}: {check_result['status']}")
        if 'message' in check_result:
            print(f"     {check_result['message']}")
    
    return result['ready']


def test_metrics_endpoint():
    """Test the metrics endpoint."""
    print("\n📊 Testing Metrics Endpoint")
    print("=" * 50)
    
    # Test Prometheus format
    prometheus_result = get_metrics(format="prometheus")
    print("Prometheus Metrics:")
    print(f"  Format: {prometheus_result['format']}")
    print(f"  Content-Type: {prometheus_result['content_type']}")
    
    # Show sample metrics (first 10 lines)
    metrics_lines = prometheus_result['metrics'].split('\n')
    print("  Sample metrics:")
    for i, line in enumerate(metrics_lines[:10]):
        if line.strip():
            print(f"    {line}")
    print(f"  ... (total {len([l for l in metrics_lines if l.strip()])} metric lines)")
    
    # Test JSON format
    json_result = get_metrics(format="json")
    print(f"\nJSON Metrics:")
    print(f"  Format: {json_result['format']}")
    print(f"  Content-Type: {json_result['content_type']}")
    
    # Show JSON structure
    if 'metrics' in json_result:
        metrics = json_result['metrics']
        print("  Structure:")
        for section, data in metrics.items():
            if isinstance(data, dict):
                print(f"    {section}: {len(data)} items")
            else:
                print(f"    {section}: {type(data).__name__}")
    
    return 'metrics' in prometheus_result and 'metrics' in json_result


def test_performance_stats():
    """Test the performance statistics endpoint."""
    print("\n⚡ Testing Performance Stats Endpoint")
    print("=" * 50)
    
    result = get_performance_stats()
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return False
    
    print("Performance Statistics:")
    
    # Current load
    if 'current_load' in result:
        load = result['current_load']
        print(f"  Current Load:")
        print(f"    Active operations: {load.get('active_operations', 0)}")
        print(f"    Recent ops (5min): {load.get('recent_operations_5min', 0)}")
        print(f"    Recent errors (5min): {load.get('recent_errors_5min', 0)}")
        print(f"    Error rate (5min): {load.get('error_rate_5min', 0):.2f}%")
        print(f"    Ops per minute: {load.get('operations_per_minute', 0):.2f}")
    
    # Top operations
    if 'top_operations' in result:
        top_ops = result['top_operations']
        print(f"\n  Top Operations ({len(top_ops)}):")
        for op in top_ops[:5]:  # Show top 5
            print(f"    {op['operation']}: {op['total_calls']} calls, {op['avg_duration_ms']}ms avg")
    
    # All stats summary
    if 'all_stats' in result:
        all_stats = result['all_stats']
        print(f"\n  Total tracked operations: {len(all_stats)}")
    
    return True


def test_feature_flags():
    """Test the feature flag management endpoint."""
    print("\n🏁 Testing Feature Flag Management")
    print("=" * 50)
    
    # List all features
    result = manage_feature_flags(action="list")
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return False
    
    print(f"Feature Flags Summary:")
    print(f"  Total flags: {result['total_flags']}")
    print(f"  ML enabled: {result['ml_enabled']}")
    print(f"  Experimental allowed: {result['experimental_allowed']}")
    
    # Show by category
    print("\n  By Category:")
    for category, flags in result['by_category'].items():
        enabled_count = sum(1 for f in flags if f['enabled'])
        print(f"    {category}: {enabled_count}/{len(flags)} enabled")
        for flag in flags[:3]:  # Show first 3 in each category
            status_emoji = "✅" if flag['enabled'] else "❌"
            print(f"      {status_emoji} {flag['name']}: {flag['description'][:50]}...")
    
    return True


def simulate_graph_operations():
    """Simulate some graph operations to generate performance data."""
    print("\n⚙️ Simulating Graph Operations for Performance Data")
    print("=" * 50)
    
    try:
        from networkx_mcp.server import create_graph, add_nodes, add_edges, graph_info
        
        # Create a test graph
        print("Creating test graph...")
        result = create_graph("test_monitoring", "undirected")
        if 'error' in result:
            print(f"❌ Create graph error: {result['error']}")
            return False
        
        # Add nodes
        print("Adding nodes...")
        result = add_nodes("test_monitoring", ["A", "B", "C", "D", "E"])
        if 'error' in result:
            print(f"❌ Add nodes error: {result['error']}")
            return False
        
        # Add edges
        print("Adding edges...")
        result = add_edges("test_monitoring", [["A", "B"], ["B", "C"], ["C", "D"], ["D", "E"]])
        if 'error' in result:
            print(f"❌ Add edges error: {result['error']}")
            return False
        
        # Get graph info
        print("Getting graph info...")
        result = graph_info("test_monitoring")
        if 'error' in result:
            print(f"❌ Graph info error: {result['error']}")
            return False
        
        print(f"✅ Successfully created graph with {result['nodes']} nodes and {result['edges']} edges")
        return True
        
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        return False


def test_k8s_deployment_scenario():
    """Test scenarios relevant to Kubernetes deployment."""
    print("\n☸️  Testing Kubernetes Deployment Scenarios")
    print("=" * 50)
    
    scenarios_passed = 0
    total_scenarios = 3
    
    # Scenario 1: Liveness probe simulation
    print("1. Liveness Probe Simulation:")
    health_result = health_check(include_details=False)
    if health_result['status'] in ['healthy', 'degraded']:
        print("   ✅ Liveness probe would PASS (pod stays alive)")
        scenarios_passed += 1
    else:
        print("   ❌ Liveness probe would FAIL (pod would be restarted)")
    
    # Scenario 2: Readiness probe simulation
    print("\n2. Readiness Probe Simulation:")
    readiness_result = readiness_check()
    if readiness_result['ready']:
        print("   ✅ Readiness probe would PASS (pod receives traffic)")
        scenarios_passed += 1
    else:
        print("   ❌ Readiness probe would FAIL (pod removed from service)")
        print(f"      Failed checks: {', '.join(readiness_result['failed_checks'])}")
    
    # Scenario 3: Metrics scraping simulation
    print("\n3. Prometheus Metrics Scraping Simulation:")
    try:
        metrics_result = get_metrics(format="prometheus")
        if 'metrics' in metrics_result and len(metrics_result['metrics']) > 0:
            print("   ✅ Metrics endpoint would be successfully scraped")
            scenarios_passed += 1
        else:
            print("   ❌ Metrics endpoint would return empty data")
    except Exception as e:
        print(f"   ❌ Metrics endpoint would fail: {e}")
    
    print(f"\nKubernetes Readiness: {scenarios_passed}/{total_scenarios} scenarios passed")
    
    # Generate sample Kubernetes manifests
    if scenarios_passed >= 2:
        print("\n📄 Sample Kubernetes Health Check Configuration:")
        print("""
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: networkx-mcp-server
    image: networkx-mcp-server:latest
    livenessProbe:
      exec:
        command:
        - python
        - -c
        - "from networkx_mcp.server import health_check; import sys; sys.exit(0 if health_check()['status'] in ['healthy', 'degraded'] else 1)"
      initialDelaySeconds: 30
      periodSeconds: 10
      timeoutSeconds: 5
      failureThreshold: 3
    
    readinessProbe:
      exec:
        command:
        - python
        - -c
        - "from networkx_mcp.server import readiness_check; import sys; sys.exit(0 if readiness_check()['ready'] else 1)"
      initialDelaySeconds: 10
      periodSeconds: 5
      timeoutSeconds: 3
      failureThreshold: 3
    
    ports:
    - containerPort: 8080
      name: metrics
---
apiVersion: v1
kind: Service
metadata:
  name: networkx-mcp-metrics
  labels:
    app: networkx-mcp-server
spec:
  ports:
  - port: 8080
    name: metrics
  selector:
    app: networkx-mcp-server
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: networkx-mcp-server
spec:
  selector:
    matchLabels:
      app: networkx-mcp-server
  endpoints:
  - port: metrics
    path: /metrics
    interval: 30s
""")
    
    return scenarios_passed >= 2


def main():
    """Run all monitoring tests."""
    print("🧪 NetworkX MCP Server - Monitoring Endpoints Test")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 6
    
    try:
        # Run tests
        if test_health_endpoint():
            tests_passed += 1
        
        if test_readiness_endpoint():
            tests_passed += 1
        
        if test_metrics_endpoint():
            tests_passed += 1
        
        if test_performance_stats():
            tests_passed += 1
        
        if test_feature_flags():
            tests_passed += 1
        
        # Simulate operations and test again
        simulate_graph_operations()
        time.sleep(1)  # Let performance tracker capture the operations
        
        # Re-test performance stats with data
        print("\n🔄 Re-testing Performance Stats (with simulated data)")
        if test_performance_stats():
            print("✅ Performance tracking working correctly with operational data")
        
        if test_k8s_deployment_scenario():
            tests_passed += 1
        
    except Exception as e:
        print(f"\n❌ Test execution error: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print(f"\n📋 Test Summary")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED - Monitoring system is ready for production!")
        print("\n✅ Reflection: Kubernetes Health Checks Analysis")
        print("The monitoring endpoints are properly designed for K8s deployment:")
        print("• Liveness probes check overall system health (allows degraded state)")
        print("• Readiness probes are stricter (ensures full operational capability)")
        print("• Metrics endpoint provides comprehensive telemetry")
        print("• Performance tracking captures operational statistics")
        print("• Feature flags enable runtime configuration")
        return 0
    else:
        print("⚠️  Some tests failed - review the issues above")
        return 1


if __name__ == "__main__":
    exit(main())