"""
Tests for enterprise monitoring and audit logging system.
"""

import pytest
import time
import uuid
from datetime import datetime
from unittest.mock import patch, MagicMock

# Skip all tests if enterprise dependencies not available
pytest.importorskip("structlog")
pytest.importorskip("prometheus_client")

from networkx_mcp.enterprise.monitoring import (
    AuditEvent, MetricsCollector, AuditLogger, get_metrics_collector, get_audit_logger
)
from networkx_mcp.enterprise.config import EnterpriseConfig, MonitoringConfig


@pytest.fixture
def mock_config():
    """Mock monitoring configuration."""
    config = EnterpriseConfig()
    config.monitoring.metrics_enabled = True
    config.monitoring.log_level = "INFO"
    config.monitoring.log_format = "json"
    config.monitoring.audit_enabled = True
    config.enterprise_mode = True
    config.server.transport = "stdio"
    config.security.api_key_enabled = True
    config.security.oauth_enabled = False
    return config


class TestAuditEvent:
    """Test audit event data structure."""
    
    def test_audit_event_creation(self):
        """Test creating an audit event."""
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            user_id="test_user",
            username="Test User",
            operation="create_graph",
            resource="test_graph",
            success=True,
            duration_ms=150.5,
            ip_address="192.168.1.1",
            user_agent="test-client/1.0",
            request_id="req-123",
            metadata={"nodes": 10, "edges": 15}
        )
        
        assert event.user_id == "test_user"
        assert event.operation == "create_graph"
        assert event.success == True
        assert event.duration_ms == 150.5
        assert event.metadata["nodes"] == 10
    
    def test_audit_event_to_dict(self):
        """Test converting audit event to dictionary."""
        timestamp = datetime.utcnow()
        event = AuditEvent(
            event_id="event-123",
            timestamp=timestamp,
            user_id="test_user",
            username="Test User",
            operation="create_graph",
            resource="test_graph",
            success=True,
            duration_ms=150.5,
            ip_address="192.168.1.1",
            user_agent="test-client/1.0",
            request_id="req-123",
            metadata={"nodes": 10}
        )
        
        event_dict = event.to_dict()
        
        assert event_dict["event_id"] == "event-123"
        assert event_dict["timestamp"] == timestamp.isoformat()
        assert event_dict["user_id"] == "test_user"
        assert event_dict["operation"] == "create_graph"
        assert event_dict["success"] == True
        assert event_dict["metadata"]["nodes"] == 10


class TestMetricsCollector:
    """Test Prometheus metrics collection."""
    
    def test_metrics_collector_creation(self, mock_config):
        """Test metrics collector initialization."""
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            collector = MetricsCollector()
            
            assert collector.config == mock_config
            assert collector.registry is not None
            assert collector.request_count is not None
            assert collector.request_duration is not None
    
    def test_record_request_metrics(self, mock_config):
        """Test recording request metrics."""
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            collector = MetricsCollector()
            
            # Record some requests
            collector.record_request("create_graph", 150.5, "success", 1024, 2048)
            collector.record_request("add_nodes", 75.2, "error", 512, 0)
            
            # Metrics should be recorded (we can't easily test the values without accessing internal state)
            assert collector.request_count is not None
            assert collector.request_duration is not None
    
    def test_record_graph_metrics(self, mock_config):
        """Test recording graph-related metrics."""
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            collector = MetricsCollector()
            
            collector.record_graph_metrics(graph_count=5, nodes=100, edges=200)
            
            # Verify metrics are recorded
            assert collector.graph_count is not None
            assert collector.graph_nodes is not None
            assert collector.graph_edges is not None
    
    def test_record_auth_metrics(self, mock_config):
        """Test recording authentication metrics."""
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            collector = MetricsCollector()
            
            collector.record_auth_attempt("api_key", True)
            collector.record_auth_attempt("oauth", False)
            
            assert collector.auth_attempts is not None
    
    def test_record_rate_limit_metrics(self, mock_config):
        """Test recording rate limit metrics."""
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            collector = MetricsCollector()
            
            collector.record_rate_limit_hit("user123", "per_minute")
            collector.record_rate_limit_hit("user456", "operation_visualize")
            
            assert collector.rate_limit_hits is not None
    
    def test_record_error_metrics(self, mock_config):
        """Test recording error metrics."""
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            collector = MetricsCollector()
            
            collector.record_error("create_graph", "ValidationError")
            collector.record_error("add_nodes", "SecurityError")
            
            assert collector.error_count is not None
    
    def test_get_metrics_output(self, mock_config):
        """Test getting metrics in Prometheus format."""
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            collector = MetricsCollector()
            
            # Record some metrics
            collector.record_request("test_op", 100.0, "success")
            collector.record_graph_metrics(3, 50, 75)
            
            metrics_output = collector.get_metrics()
            content_type = collector.get_content_type()
            
            assert isinstance(metrics_output, str)
            assert "networkx_mcp_requests_total" in metrics_output
            assert "networkx_mcp_server" in metrics_output
            assert content_type == "text/plain; version=0.0.4; charset=utf-8"
    
    def test_system_metrics_update(self, mock_config):
        """Test updating system resource metrics."""
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            collector = MetricsCollector()
            
            # Mock psutil
            with patch('psutil.Process') as mock_process:
                mock_memory = MagicMock()
                mock_memory.rss = 1024 * 1024 * 100  # 100MB
                mock_process.return_value.memory_info.return_value = mock_memory
                mock_process.return_value.cpu_percent.return_value = 15.5
                
                collector.update_system_metrics()
                
                # Should have called psutil methods
                mock_process.return_value.memory_info.assert_called()
                mock_process.return_value.cpu_percent.assert_called()
    
    def test_system_metrics_without_psutil(self, mock_config):
        """Test system metrics when psutil is not available."""
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            collector = MetricsCollector()
            
            # Mock ImportError for psutil
            with patch('networkx_mcp.enterprise.monitoring.psutil', side_effect=ImportError):
                # Should not raise an error
                collector.update_system_metrics()
    
    def test_auth_methods_detection(self, mock_config):
        """Test detection of enabled authentication methods."""
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            collector = MetricsCollector()
            
            auth_methods = collector._get_auth_methods()
            assert "api_key" in auth_methods
            assert "oauth" not in auth_methods
        
        # Test with OAuth enabled
        mock_config.security.oauth_enabled = True
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            collector = MetricsCollector()
            auth_methods = collector._get_auth_methods()
            assert "api_key" in auth_methods
            assert "oauth" in auth_methods
        
        # Test with no auth
        mock_config.security.api_key_enabled = False
        mock_config.security.oauth_enabled = False
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            collector = MetricsCollector()
            auth_methods = collector._get_auth_methods()
            assert auth_methods == "none"


class TestAuditLogger:
    """Test structured audit logging."""
    
    def test_audit_logger_creation(self, mock_config):
        """Test audit logger initialization."""
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            logger = AuditLogger()
            
            assert logger.config == mock_config
            assert logger.logger is not None
            assert logger.performance_logger is not None
            assert logger.security_logger is not None
    
    def test_request_context_management(self, mock_config):
        """Test request context management."""
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            logger = AuditLogger()
            
            logger.set_request_context("req-123", "user456", "192.168.1.1", "test-client")
            
            context = logger.get_request_context()
            assert context["request_id"] == "req-123"
            assert context["user_id"] == "user456"
            assert context["ip_address"] == "192.168.1.1"
            assert context["user_agent"] == "test-client"
    
    def test_authentication_logging(self, mock_config):
        """Test authentication event logging."""
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            logger = AuditLogger()
            
            # Mock the actual logger to capture calls
            logger.security_logger = MagicMock()
            
            # Log successful authentication
            logger.log_authentication("user123", "api_key", True, "192.168.1.1", {"api_key_hash": "abc123"})
            
            logger.security_logger.info.assert_called_once()
            call_args = logger.security_logger.info.call_args
            assert call_args[0][0] == "Authentication successful"
            assert call_args[1]["user_id"] == "user123"
            assert call_args[1]["method"] == "api_key"
            assert call_args[1]["success"] == True
            
            # Reset mock
            logger.security_logger.reset_mock()
            
            # Log failed authentication
            logger.log_authentication("user456", "oauth", False, "192.168.1.2")
            
            logger.security_logger.warning.assert_called_once()
            call_args = logger.security_logger.warning.call_args
            assert call_args[0][0] == "Authentication failed"
            assert call_args[1]["success"] == False
    
    def test_authorization_logging(self, mock_config):
        """Test authorization decision logging."""
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            logger = AuditLogger()
            logger.security_logger = MagicMock()
            logger.set_request_context("req-123", "user456", "192.168.1.1", "test-client")
            
            # Log allowed authorization
            logger.log_authorization("user123", "create_graph", "my_graph", True)
            
            logger.security_logger.info.assert_called_once()
            call_args = logger.security_logger.info.call_args
            assert call_args[0][0] == "Authorization granted"
            assert call_args[1]["operation"] == "create_graph"
            assert call_args[1]["allowed"] == True
            
            # Reset and test denied authorization
            logger.security_logger.reset_mock()
            logger.log_authorization("user123", "admin_access", "system", False, "insufficient_privileges")
            
            logger.security_logger.warning.assert_called_once()
            call_args = logger.security_logger.warning.call_args
            assert call_args[0][0] == "Authorization denied"
            assert call_args[1]["allowed"] == False
            assert call_args[1]["reason"] == "insufficient_privileges"
    
    def test_operation_logging(self, mock_config):
        """Test operation execution logging."""
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            logger = AuditLogger()
            logger.logger = MagicMock()
            logger.set_request_context("req-123", "user456", "192.168.1.1", "test-client")
            
            # Log successful operation
            logger.log_operation("create_graph", "my_graph", True, 150.5, {"nodes": 10})
            
            logger.logger.info.assert_called_once()
            call_args = logger.logger.info.call_args
            assert call_args[0][0] == "Operation completed"
            assert call_args[1]["operation"] == "create_graph"
            assert call_args[1]["success"] == True
            assert call_args[1]["duration_ms"] == 150.5
            
            # Reset and test failed operation
            logger.logger.reset_mock()
            logger.log_operation("add_nodes", "my_graph", False, 75.2, {"error": "validation_failed"})
            
            logger.logger.error.assert_called_once()
            call_args = logger.logger.error.call_args
            assert call_args[0][0] == "Operation failed"
            assert call_args[1]["success"] == False
    
    def test_performance_logging(self, mock_config):
        """Test performance metrics logging."""
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            logger = AuditLogger()
            logger.performance_logger = MagicMock()
            logger.set_request_context("req-123", "user456", "192.168.1.1", "test-client")
            
            logger.log_performance("pagerank", 2500.0, {"memory_mb": 50, "nodes": 1000})
            
            logger.performance_logger.info.assert_called_once()
            call_args = logger.performance_logger.info.call_args
            assert call_args[0][0] == "Performance metric"
            assert call_args[1]["operation"] == "pagerank"
            assert call_args[1]["duration_ms"] == 2500.0
            assert call_args[1]["memory_mb"] == 50
    
    def test_security_event_logging(self, mock_config):
        """Test security event logging with different severities."""
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            logger = AuditLogger()
            logger.security_logger = MagicMock()
            logger.set_request_context("req-123", "user456", "192.168.1.1", "test-client")
            
            # Test high severity
            logger.log_security_event("rate_limit_exceeded", "high", {"limit": "per_minute"})
            logger.security_logger.error.assert_called_once()
            
            # Test medium severity
            logger.security_logger.reset_mock()
            logger.log_security_event("invalid_token", "medium", {"token_type": "jwt"})
            logger.security_logger.warning.assert_called_once()
            
            # Test low severity
            logger.security_logger.reset_mock()
            logger.log_security_event("user_login", "low", {"login_method": "api_key"})
            logger.security_logger.info.assert_called_once()
    
    def test_operation_context_manager(self, mock_config):
        """Test operation context manager for automatic logging."""
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            logger = AuditLogger()
            logger.logger = MagicMock()
            logger.set_request_context("req-123", "user456", "192.168.1.1", "test-client")
            
            # Test successful operation
            with logger.operation_context("create_graph", "my_graph"):
                time.sleep(0.01)  # Small delay to test timing
            
            logger.logger.info.assert_called_once()
            call_args = logger.logger.info.call_args
            assert call_args[0][0] == "Operation completed"
            assert call_args[1]["success"] == True
            assert call_args[1]["duration_ms"] > 0
            
            # Test failed operation
            logger.logger.reset_mock()
            try:
                with logger.operation_context("add_nodes", "my_graph"):
                    raise ValueError("Test error")
            except ValueError:
                pass
            
            logger.logger.error.assert_called_once()
            call_args = logger.logger.error.call_args
            assert call_args[0][0] == "Operation failed"
            assert call_args[1]["success"] == False
            assert call_args[1]["metadata"]["error"] == "Test error"
    
    def test_log_format_configuration(self):
        """Test different log format configurations."""
        # Test JSON format
        config = EnterpriseConfig()
        config.monitoring.log_format = "json"
        
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=config):
            logger = AuditLogger()
            # Should not raise any errors
            assert logger.logger is not None
        
        # Test text format
        config.monitoring.log_format = "text"
        
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=config):
            logger = AuditLogger()
            # Should not raise any errors
            assert logger.logger is not None


class TestGlobalInstances:
    """Test global instance management."""
    
    def test_get_metrics_collector_singleton(self, mock_config):
        """Test metrics collector singleton behavior."""
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            collector1 = get_metrics_collector()
            collector2 = get_metrics_collector()
            
            # Should return the same instance
            assert collector1 is collector2
    
    def test_get_audit_logger_singleton(self, mock_config):
        """Test audit logger singleton behavior."""
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            logger1 = get_audit_logger()
            logger2 = get_audit_logger()
            
            # Should return the same instance
            assert logger1 is logger2


@pytest.mark.integration
class TestMonitoringIntegration:
    """Integration tests for monitoring system."""
    
    def test_metrics_and_audit_integration(self, mock_config):
        """Test integration between metrics and audit logging."""
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            metrics = MetricsCollector()
            audit = AuditLogger()
            
            # Set up request context
            audit.set_request_context("req-123", "user456", "192.168.1.1", "test-client")
            
            # Simulate a request workflow
            start_time = time.time()
            
            # Record authentication
            metrics.record_auth_attempt("api_key", True)
            audit.log_authentication("user456", "api_key", True, "192.168.1.1")
            
            # Record authorization
            audit.log_authorization("user456", "create_graph", "my_graph", True)
            
            # Record operation with context manager
            with audit.operation_context("create_graph", "my_graph"):
                # Simulate work
                time.sleep(0.01)
                
                # Record metrics during operation
                metrics.record_graph_metrics(1, 10, 15)
            
            # Record request completion
            duration = (time.time() - start_time) * 1000
            metrics.record_request("create_graph", duration, "success", 1024, 2048)
            
            # Get metrics output
            metrics_output = metrics.get_metrics()
            assert "networkx_mcp_requests_total" in metrics_output
    
    def test_concurrent_monitoring(self, mock_config):
        """Test monitoring under concurrent access."""
        import threading
        
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            metrics = MetricsCollector()
            audit = AuditLogger()
            
            def worker(worker_id):
                for i in range(10):
                    metrics.record_request(f"operation_{i}", 100.0, "success")
                    audit.log_operation(f"operation_{i}", f"resource_{i}", True, 100.0)
            
            # Start multiple threads
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            # Should complete without errors
            metrics_output = metrics.get_metrics()
            assert metrics_output is not None
    
    def test_monitoring_error_handling(self, mock_config):
        """Test monitoring system error handling."""
        with patch('networkx_mcp.enterprise.monitoring.get_config', return_value=mock_config):
            metrics = MetricsCollector()
            audit = AuditLogger()
            
            # Test with various invalid inputs
            metrics.record_request("", -1, "")  # Should not crash
            metrics.record_graph_metrics(-1, -1, -1)  # Should not crash
            
            audit.log_operation("", "", True, -1)  # Should not crash
            
            # System should remain functional
            metrics.record_request("valid_op", 100.0, "success")
            audit.log_operation("valid_op", "valid_resource", True, 100.0)