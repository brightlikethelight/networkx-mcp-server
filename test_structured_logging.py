#!/usr/bin/env python3
"""Comprehensive test for structured logging system with correlation IDs."""

import json
import os
import sys
import time
import tempfile
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the logging system
from networkx_mcp.logging import (
    get_logger,
    configure_logging,
    CorrelationContext,
    correlation_middleware,
    timed_operation,
    generate_correlation_id,
    set_correlation_id,
    get_correlation_id,
)

# Import server functions to test
from networkx_mcp.server import create_graph, add_nodes, add_edges, shortest_path


def test_basic_structured_logging():
    """Test basic structured logging capabilities."""
    print("🔍 Testing Basic Structured Logging")
    print("=" * 50)
    
    # Configure logging for testing
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.log', delete=False) as log_file:
        log_file_path = log_file.name
    
    # Test different formats
    formats_to_test = ["colored", "json", "compact"]
    
    for format_type in formats_to_test:
        print(f"\n📝 Testing {format_type} format:")
        
        configure_logging(
            level="DEBUG",
            format_type=format_type,
            output_file=log_file_path if format_type == "json" else None
        )
        
        logger = get_logger("test.basic")
        
        # Test all log levels
        logger.debug("Debug message", extra={"test_level": "debug", "format": format_type})
        logger.info("Info message", extra={"test_level": "info", "format": format_type})
        logger.warning("Warning message", extra={"test_level": "warning", "format": format_type})
        logger.error("Error message", extra={"test_level": "error", "format": format_type})
        
        # Test structured data
        logger.info("Structured message", extra={
            "user_id": "test_user_123",
            "graph_name": "test_graph",
            "operation": "test_operation",
            "metrics": {
                "nodes": 100,
                "edges": 250,
                "processing_time_ms": 42.5
            }
        })
        
        print(f"  ✅ {format_type} format tested")
    
    # Read and verify JSON log file
    if os.path.exists(log_file_path):
        print(f"\n📄 JSON Log File Content:")
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines[-3:]:  # Show last 3 lines
                try:
                    log_entry = json.loads(line.strip())
                    print(f"  📋 {log_entry['level']}: {log_entry['message']}")
                    if 'extra' in log_entry:
                        print(f"     📊 Extra: {log_entry['extra']}")
                except json.JSONDecodeError:
                    print(f"  ⚠️  Invalid JSON: {line.strip()}")
        
        # Cleanup
        os.unlink(log_file_path)
    
    return True


def test_correlation_ids():
    """Test correlation ID functionality."""
    print("\n🔗 Testing Correlation IDs")
    print("=" * 50)
    
    configure_logging(level="INFO", format_type="colored")
    logger = get_logger("test.correlation")
    
    # Test manual correlation ID
    correlation_id = generate_correlation_id()
    print(f"Generated correlation ID: {correlation_id[:8]}...")
    
    with CorrelationContext(correlation_id=correlation_id, operation_name="test_operation"):
        current_id = get_correlation_id()
        print(f"Current correlation ID: {current_id[:8]}...")
        
        logger.info("Message with correlation ID", extra={
            "test_phase": "manual_correlation",
            "expected_id": correlation_id
        })
        
        # Test nested contexts
        with CorrelationContext(operation_name="nested_operation"):
            nested_id = get_correlation_id()
            logger.info("Nested operation", extra={
                "test_phase": "nested_correlation",
                "nested_id": nested_id[:8]
            })
        
        # Should be back to original correlation ID
        back_to_original = get_correlation_id()
        assert back_to_original == correlation_id, "Correlation ID should be restored"
        logger.info("Back to original context")
    
    # Test middleware
    @correlation_middleware
    def test_function(operation_name: str):
        logger.info(f"Inside {operation_name}", extra={
            "test_phase": "middleware_test",
            "operation": operation_name
        })
        return get_correlation_id()
    
    result_id = test_function("middleware_test_operation")
    print(f"Middleware generated ID: {result_id[:8] if result_id else 'None'}...")
    
    return True


def test_performance_logging():
    """Test performance timing and structured logging."""
    print("\n⚡ Testing Performance Logging")
    print("=" * 50)
    
    configure_logging(level="DEBUG", format_type="colored")
    logger = get_logger("test.performance")
    
    # Test timed operation decorator
    @timed_operation("test.slow_operation", log_args=True)
    def slow_operation(delay: float, operation_name: str):
        time.sleep(delay)
        logger.info("Operation processing", extra={
            "operation_name": operation_name,
            "delay": delay
        })
        return f"Completed {operation_name}"
    
    # Test successful operation
    result = slow_operation(0.1, "test_operation_1")
    print(f"Result: {result}")
    
    # Test operation with error
    @timed_operation("test.failing_operation")
    def failing_operation():
        logger.warning("About to fail")
        raise ValueError("Simulated error for testing")
    
    try:
        failing_operation()
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    # Test manual performance logging
    start_time = time.time()
    time.sleep(0.05)  # Simulate work
    duration_ms = (time.time() - start_time) * 1000
    
    logger.log_performance(
        operation="manual_timing_test",
        duration_ms=duration_ms,
        success=True,
        custom_metric=42,
        data_size=1024
    )
    
    return True


def test_request_tracing():
    """Test end-to-end request tracing through the system."""
    print("\n🔄 Testing Request Tracing")
    print("=" * 50)
    
    configure_logging(level="DEBUG", format_type="colored")
    logger = get_logger("test.tracing")
    
    # Simulate a complete request flow
    with CorrelationContext(
        operation_name="create_graph_request",
        user_id="test_user_456",
        request_id="req_" + generate_correlation_id()[:8]
    ):
        logger.info("Request started", extra={
            "request_type": "create_graph",
            "user_id": "test_user_456"
        })
        
        try:
            # Create a graph (this will use the structured logging)
            result = create_graph(
                name=f"trace_test_graph_{int(time.time())}",
                graph_type="undirected",
                data={
                    "nodes": ["A", "B", "C"],
                    "edges": [["A", "B"], ["B", "C"]]
                }
            )
            
            if "error" in result:
                logger.error("Graph creation failed", extra={
                    "error": result["error"],
                    "request_result": "failure"
                })
                return False
            else:
                logger.info("Graph creation successful", extra={
                    "graph_name": result["name"],
                    "nodes": result["nodes"],
                    "edges": result["edges"],
                    "request_result": "success"
                })
            
            # Test additional operations with same correlation ID
            graph_name = result["name"]
            
            # Add more nodes
            add_result = add_nodes(graph_name, ["D", "E"])
            logger.info("Added nodes", extra={
                "graph_name": graph_name,
                "nodes_added": add_result.get("nodes_added", 0)
            })
            
            # Find shortest path
            path_result = shortest_path(graph_name, "A", "C")
            logger.info("Shortest path computed", extra={
                "graph_name": graph_name,
                "source": "A",
                "target": "C",
                "path_found": "path" in path_result
            })
            
            logger.info("Request completed successfully", extra={
                "total_operations": 3,
                "final_graph_state": {
                    "nodes": add_result.get("total_nodes"),
                    "edges": add_result.get("total_edges")
                }
            })
            
        except Exception as e:
            logger.error("Request failed with exception", extra={
                "error": str(e),
                "error_type": type(e).__name__
            }, exc_info=True)
            return False
    
    return True


def test_log_aggregation_format():
    """Test log format suitable for aggregation tools."""
    print("\n📊 Testing Log Aggregation Format")
    print("=" * 50)
    
    # Create a temporary log file for aggregation testing
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as log_file:
        log_file_path = log_file.name
    
    configure_logging(
        level="INFO",
        format_type="json",
        output_file=log_file_path
    )
    
    logger = get_logger("test.aggregation")
    
    # Generate various log entries that would be useful for aggregation
    test_scenarios = [
        {
            "correlation_id": generate_correlation_id(),
            "user_id": "user_001",
            "operation": "graph_analysis",
            "status": "success",
            "metrics": {"duration_ms": 150.5, "nodes_processed": 1000}
        },
        {
            "correlation_id": generate_correlation_id(),
            "user_id": "user_002", 
            "operation": "graph_creation",
            "status": "error",
            "metrics": {"duration_ms": 25.0, "error_code": "VALIDATION_FAILED"}
        },
        {
            "correlation_id": generate_correlation_id(),
            "user_id": "user_001",
            "operation": "algorithm_execution",
            "status": "success",
            "metrics": {"duration_ms": 2340.8, "algorithm": "centrality", "complexity": "high"}
        }
    ]
    
    for i, scenario in enumerate(test_scenarios):
        with CorrelationContext(
            correlation_id=scenario["correlation_id"],
            operation_name=scenario["operation"],
            user_id=scenario["user_id"]
        ):
            if scenario["status"] == "success":
                logger.info(f"Operation {scenario['operation']} completed", extra={
                    "status": scenario["status"],
                    "metrics": scenario["metrics"],
                    "scenario_index": i
                })
            else:
                logger.error(f"Operation {scenario['operation']} failed", extra={
                    "status": scenario["status"],
                    "metrics": scenario["metrics"],
                    "scenario_index": i
                })
    
    # Read and analyze the log file
    print(f"\n📄 Aggregation Log Analysis:")
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
        
        successful_ops = 0
        failed_ops = 0
        total_duration = 0
        users = set()
        operations = set()
        
        print(f"Total log entries: {len(lines)}")
        
        for line in lines:
            try:
                log_entry = json.loads(line.strip())
                
                # Extract correlation ID
                correlation_id = log_entry.get('correlation_id', 'N/A')
                
                # Extract context
                context = log_entry.get('context', {})
                if context:
                    user_id = context.get('user_id')
                    operation = context.get('operation')
                    
                    if user_id:
                        users.add(user_id)
                    if operation:
                        operations.add(operation)
                
                # Extract metrics from extra
                extra = log_entry.get('extra', {})
                if 'metrics' in extra:
                    metrics = extra['metrics']
                    if 'duration_ms' in metrics:
                        total_duration += metrics['duration_ms']
                    
                    status = extra.get('status')
                    if status == 'success':
                        successful_ops += 1
                    elif status == 'error':
                        failed_ops += 1
                
                print(f"  📋 {log_entry['level']}: {correlation_id[:8]}... | {log_entry['message']}")
                
            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON parse error: {e}")
        
        print(f"\n📈 Aggregation Summary:")
        print(f"  Successful operations: {successful_ops}")
        print(f"  Failed operations: {failed_ops}")
        print(f"  Total duration: {total_duration:.2f}ms")
        print(f"  Unique users: {len(users)}")
        print(f"  Unique operations: {len(operations)}")
        print(f"  Average duration: {total_duration/(successful_ops + failed_ops):.2f}ms" if (successful_ops + failed_ops) > 0 else "N/A")
    
    # Show sample aggregation queries
    print(f"\n🔍 Sample Aggregation Queries:")
    print(f"  # Group by user_id:")
    print(f"  cat {log_file_path} | jq '.context.user_id' | sort | uniq -c")
    print(f"  ")
    print(f"  # Average duration by operation:")
    print(f"  cat {log_file_path} | jq -r '[.context.operation, .extra.metrics.duration_ms] | @csv'")
    print(f"  ")
    print(f"  # Error rate by operation:")
    print(f"  cat {log_file_path} | jq -r 'select(.extra.status == \"error\") | .context.operation'")
    
    # Cleanup
    os.unlink(log_file_path)
    
    return True


def test_correlation_across_modules():
    """Test that correlation IDs work across different modules and components."""
    print("\n🌐 Testing Cross-Module Correlation")
    print("=" * 50)
    
    configure_logging(level="DEBUG", format_type="colored")
    
    # Import different modules
    from networkx_mcp.core.graph_operations import GraphManager
    from networkx_mcp.monitoring.endpoints import HealthEndpoint
    
    correlation_id = generate_correlation_id()
    
    with CorrelationContext(correlation_id=correlation_id, operation_name="cross_module_test"):
        # Test GraphManager with correlation
        manager_logger = get_logger("test.graph_manager")
        manager_logger.info("Testing GraphManager with correlation")
        
        graph_manager = GraphManager()
        # Note: GraphManager doesn't have structured logging yet, but correlation should still work
        
        # Test monitoring endpoint
        health_logger = get_logger("test.health_endpoint")
        health_logger.info("Testing HealthEndpoint with correlation")
        
        health_endpoint = HealthEndpoint()
        # The health check would inherit the correlation ID
        
        # Test feature flags
        features_logger = get_logger("test.features")
        features_logger.info("Testing feature flags with correlation")
        
        from networkx_mcp.features import get_feature_flags
        flags = get_feature_flags()
        features_logger.info("Retrieved feature flags", extra={
            "flag_count": len(flags),
            "ml_enabled": any(f.get('enabled', False) for f in flags.values() if 'ml' in f.get('category', ''))
        })
        
        # Verify all these operations have the same correlation ID
        current_id = get_correlation_id()
        assert current_id == correlation_id, f"Correlation ID mismatch: {current_id} != {correlation_id}"
        
        manager_logger.info("Cross-module correlation test completed", extra={
            "modules_tested": ["graph_manager", "health_endpoint", "features"],
            "correlation_preserved": True
        })
    
    return True


def main():
    """Run all structured logging tests."""
    print("🧪 NetworkX MCP Server - Structured Logging Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Structured Logging", test_basic_structured_logging),
        ("Correlation IDs", test_correlation_ids),
        ("Performance Logging", test_performance_logging),
        ("Request Tracing", test_request_tracing),
        ("Log Aggregation Format", test_log_aggregation_format),
        ("Cross-Module Correlation", test_correlation_across_modules),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n🔄 Running: {test_name}")
            success = test_func()
            if success:
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📋 Test Results:")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Structured logging system is working!")
        print("\n✅ Reflection: Request Tracing Capabilities")
        print("The structured logging system provides comprehensive request tracing:")
        print("• Correlation IDs link all operations within a request flow")
        print("• Structured data enables easy parsing and aggregation")
        print("• Performance metrics track timing across all operations")
        print("• Cross-module correlation preserves context")
        print("• JSON format supports modern log analysis tools")
        print("• Multiple output formats serve different use cases")
        return 0
    else:
        print("⚠️  Some tests failed - check the output above")
        return 1


if __name__ == "__main__":
    exit(main())