#!/usr/bin/env python3
"""Demo script for log aggregation tools testing."""

import json
import os
import sys
import tempfile
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from networkx_mcp.logging import (
    get_logger,
    configure_logging,
    CorrelationContext,
    generate_correlation_id,
)

# Import server functions
from networkx_mcp.server import create_graph, add_nodes, shortest_path


def generate_sample_logs():
    """Generate a variety of log entries for aggregation testing."""
    
    # Create temporary log file
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as log_file:
        log_file_path = log_file.name
    
    # Configure JSON logging for aggregation
    configure_logging(
        level="INFO",
        format_type="json",
        output_file=log_file_path
    )
    
    logger = get_logger("demo.aggregation")
    
    print(f"📄 Generating sample logs in: {log_file_path}")
    
    # Simulate various request scenarios
    scenarios = [
        {
            "user_id": "alice_researcher",
            "operation": "create_social_network",
            "success": True,
            "duration": 150.5,
            "nodes": 1000,
            "edges": 2500
        },
        {
            "user_id": "bob_analyst", 
            "operation": "shortest_path_analysis",
            "success": True,
            "duration": 75.2,
            "algorithm": "dijkstra"
        },
        {
            "user_id": "alice_researcher",
            "operation": "centrality_calculation",
            "success": False,
            "duration": 25.0,
            "error": "MEMORY_LIMIT_EXCEEDED"
        },
        {
            "user_id": "charlie_student",
            "operation": "create_small_graph",
            "success": True,
            "duration": 10.5,
            "nodes": 50,
            "edges": 100
        },
        {
            "user_id": "bob_analyst",
            "operation": "community_detection",
            "success": True,
            "duration": 2340.8,
            "algorithm": "louvain",
            "communities": 15
        }
    ]
    
    for i, scenario in enumerate(scenarios):
        correlation_id = generate_correlation_id()
        
        with CorrelationContext(
            correlation_id=correlation_id,
            operation_name=scenario["operation"],
            user_id=scenario["user_id"],
            request_id=f"req_{i+1:03d}"
        ):
            # Log request start
            logger.info("Request started", extra={
                "event_type": "request_start",
                "user_id": scenario["user_id"],
                "operation": scenario["operation"]
            })
            
            # Simulate some processing time
            time.sleep(0.01)
            
            # Log specific operation details
            if scenario["success"]:
                logger.info("Operation completed successfully", extra={
                    "event_type": "operation_success",
                    "duration_ms": scenario["duration"],
                    "metrics": {k: v for k, v in scenario.items() 
                              if k not in ["user_id", "operation", "success", "duration"]}
                })
            else:
                logger.error("Operation failed", extra={
                    "event_type": "operation_error",
                    "duration_ms": scenario["duration"],
                    "error_code": scenario.get("error", "UNKNOWN_ERROR"),
                    "error_details": "Simulated error for testing"
                })
            
            # Log request completion
            logger.info("Request completed", extra={
                "event_type": "request_end",
                "status": "success" if scenario["success"] else "error",
                "total_duration_ms": scenario["duration"]
            })
    
    print(f"✅ Generated {len(scenarios)} request scenarios")
    return log_file_path


def demonstrate_log_queries(log_file_path):
    """Demonstrate various log aggregation queries."""
    print(f"\n🔍 Log Aggregation Queries Demo")
    print("=" * 50)
    
    # Read all log entries
    with open(log_file_path, 'r') as f:
        log_entries = [json.loads(line.strip()) for line in f if line.strip()]
    
    print(f"Total log entries: {len(log_entries)}")
    
    # Analysis 1: Group by user
    users = {}
    for entry in log_entries:
        context = entry.get('context', {})
        user_id = context.get('user_id')
        if user_id:
            users[user_id] = users.get(user_id, 0) + 1
    
    print(f"\n📊 Requests by User:")
    for user, count in sorted(users.items()):
        print(f"  {user}: {count} requests")
    
    # Analysis 2: Operation success rates
    operations = {}
    for entry in log_entries:
        extra = entry.get('extra', {})
        if extra.get('event_type') == 'operation_success':
            context = entry.get('context', {})
            op = context.get('operation', 'unknown')
            operations[op] = operations.get(op, {'success': 0, 'total': 0})
            operations[op]['success'] += 1
            operations[op]['total'] += 1
        elif extra.get('event_type') == 'operation_error':
            context = entry.get('context', {})
            op = context.get('operation', 'unknown')
            operations[op] = operations.get(op, {'success': 0, 'total': 0})
            operations[op]['total'] += 1
    
    print(f"\n📈 Operation Success Rates:")
    for op, stats in sorted(operations.items()):
        success_rate = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"  {op}: {success_rate:.1f}% ({stats['success']}/{stats['total']})")
    
    # Analysis 3: Average duration by operation
    durations = {}
    for entry in log_entries:
        extra = entry.get('extra', {})
        if 'duration_ms' in extra:
            context = entry.get('context', {})
            op = context.get('operation', 'unknown')
            if op not in durations:
                durations[op] = []
            durations[op].append(extra['duration_ms'])
    
    print(f"\n⏱️  Average Duration by Operation:")
    for op, times in sorted(durations.items()):
        avg_duration = sum(times) / len(times)
        print(f"  {op}: {avg_duration:.1f}ms (n={len(times)})")
    
    # Analysis 4: Error analysis
    errors = []
    for entry in log_entries:
        if entry.get('level') == 'ERROR':
            extra = entry.get('extra', {})
            context = entry.get('context', {})
            errors.append({
                'operation': context.get('operation', 'unknown'),
                'error_code': extra.get('error_code', 'UNKNOWN'),
                'correlation_id': entry.get('correlation_id', 'N/A')
            })
    
    print(f"\n🚨 Error Analysis:")
    if errors:
        for error in errors:
            print(f"  {error['operation']}: {error['error_code']} (trace: {error['correlation_id'][:8]}...)")
    else:
        print("  No errors found in logs")
    
    # Analysis 5: Trace a specific request
    print(f"\n🔗 Request Trace Example (first correlation ID):")
    if log_entries:
        first_correlation_id = log_entries[0].get('correlation_id')
        if first_correlation_id:
            trace_entries = [e for e in log_entries if e.get('correlation_id') == first_correlation_id]
            for entry in trace_entries:
                timestamp = entry.get('timestamp', 'N/A')
                message = entry.get('message', 'N/A')
                extra = entry.get('extra', {})
                event_type = extra.get('event_type', 'N/A')
                print(f"  {timestamp}: {event_type} - {message}")
    
    return {
        'total_entries': len(log_entries),
        'unique_users': len(users),
        'unique_operations': len(operations),
        'error_count': len(errors)
    }


def show_aggregation_tool_examples(log_file_path):
    """Show examples for popular log aggregation tools."""
    print(f"\n🛠️  Log Aggregation Tool Examples")
    print("=" * 50)
    
    print(f"📁 Log file: {log_file_path}")
    print(f"")
    
    # jq examples (JSON processor)
    print(f"🔧 jq Examples:")
    print(f"  # Count requests by user:")
    print(f"  cat {log_file_path} | jq -r '.context.user_id' | sort | uniq -c")
    print(f"  ")
    print(f"  # Extract error messages:")
    print(f"  cat {log_file_path} | jq -r 'select(.level == \"ERROR\") | .extra.error_code'")
    print(f"  ")
    print(f"  # Average duration by operation:")
    print(f"  cat {log_file_path} | jq -r 'select(.extra.duration_ms) | [.context.operation, .extra.duration_ms] | @csv'")
    print(f"  ")
    print(f"  # Trace specific request:")
    print(f"  cat {log_file_path} | jq -r 'select(.correlation_id == \"YOUR_CORRELATION_ID\") | .message'")
    
    # Elasticsearch examples
    print(f"\n🔍 Elasticsearch Query Examples:")
    print(f'  # Group by user:')
    print(f'  {{')
    print(f'    "aggs": {{')
    print(f'      "users": {{')
    print(f'        "terms": {{ "field": "context.user_id.keyword" }}')
    print(f'      }}')
    print(f'    }}')
    print(f'  }}')
    print(f'  ')
    print(f'  # Error rate by operation:')
    print(f'  {{')
    print(f'    "query": {{ "term": {{ "level": "ERROR" }} }},')
    print(f'    "aggs": {{')
    print(f'      "operations": {{')
    print(f'        "terms": {{ "field": "context.operation.keyword" }}')
    print(f'      }}')
    print(f'    }}')
    print(f'  }}')
    
    # Grafana Loki examples  
    print(f"\n📊 Grafana Loki LogQL Examples:")
    print(f'  # All logs: {{service="networkx-mcp"}}')
    print(f'  # Errors only: {{service="networkx-mcp"}} |= "ERROR"')
    print(f'  # By user: {{service="networkx-mcp"}} | json | context_user_id="alice_researcher"')
    print(f'  # Slow operations: {{service="networkx-mcp"}} | json | extra_duration_ms > 1000')
    print(f'  # Trace request: {{service="networkx-mcp"}} | json | correlation_id="YOUR_ID"')
    
    # Splunk examples
    print(f"\n🎯 Splunk Query Examples:")
    print(f'  # Count by user: index="networkx-mcp" | stats count by context.user_id')
    print(f'  # Error analysis: index="networkx-mcp" level="ERROR" | stats count by extra.error_code')
    print(f'  # Performance trending: index="networkx-mcp" | timechart avg(extra.duration_ms) by context.operation')
    print(f'  # Request tracing: index="networkx-mcp" correlation_id="YOUR_ID" | sort _time')


def main():
    """Main demonstration function."""
    print("🧪 NetworkX MCP Server - Log Aggregation Demo")
    print("=" * 60)
    
    try:
        # Generate sample logs
        log_file_path = generate_sample_logs()
        
        # Analyze the logs
        stats = demonstrate_log_queries(log_file_path)
        
        # Show tool examples
        show_aggregation_tool_examples(log_file_path)
        
        print(f"\n📋 Summary:")
        print("=" * 60)
        print(f"✅ Generated structured logs suitable for aggregation")
        print(f"📊 Statistics:")
        print(f"  - Total log entries: {stats['total_entries']}")
        print(f"  - Unique users: {stats['unique_users']}")
        print(f"  - Unique operations: {stats['unique_operations']}")
        print(f"  - Error entries: {stats['error_count']}")
        print(f"")
        print(f"🎯 Log aggregation capabilities demonstrated:")
        print(f"  ✅ Correlation ID-based request tracing")
        print(f"  ✅ User activity analysis")
        print(f"  ✅ Operation performance metrics")
        print(f"  ✅ Error rate and failure analysis")
        print(f"  ✅ JSON-structured logs for easy parsing")
        print(f"  ✅ Ready for Elasticsearch, Splunk, Loki, etc.")
        
        # Cleanup
        os.unlink(log_file_path)
        
        return 0
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())