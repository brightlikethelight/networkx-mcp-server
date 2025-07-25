"""Comprehensive tests for utils/performance.py - Target: 100% coverage (74 lines).

This module has simple, self-contained utilities that are easy to test.
"""

import time
from unittest.mock import Mock, patch


class TestPerformanceMonitor:
    """Test the PerformanceMonitor class."""

    def test_init(self):
        """Test PerformanceMonitor initialization."""
        from networkx_mcp.utils.performance import PerformanceMonitor

        monitor = PerformanceMonitor()
        assert isinstance(monitor.metrics, dict)
        assert len(monitor.metrics) == 0

    def test_time_operation_decorator(self):
        """Test the time_operation decorator."""
        from networkx_mcp.utils.performance import PerformanceMonitor

        monitor = PerformanceMonitor()

        # Create a test function
        @monitor.time_operation("test_op")
        def test_func(x, y):
            time.sleep(0.01)  # Small delay to ensure measurable time
            return x + y

        # Call the decorated function
        result = test_func(2, 3)

        # Check result
        assert result == 5

        # Check metrics were recorded
        assert "test_op" in monitor.metrics
        assert len(monitor.metrics["test_op"]) == 1
        assert monitor.metrics["test_op"][0] > 0.01  # Should take at least 0.01s

    def test_time_operation_multiple_calls(self):
        """Test decorator with multiple calls."""
        from networkx_mcp.utils.performance import PerformanceMonitor

        monitor = PerformanceMonitor()

        @monitor.time_operation("multi_op")
        def quick_func():
            return "done"

        # Call multiple times
        for _ in range(3):
            quick_func()

        # Check metrics
        assert "multi_op" in monitor.metrics
        assert len(monitor.metrics["multi_op"]) == 3

    def test_time_operation_with_args_kwargs(self):
        """Test decorator preserves function signature."""
        from networkx_mcp.utils.performance import PerformanceMonitor

        monitor = PerformanceMonitor()

        @monitor.time_operation("complex_op")
        def complex_func(a, b, c=10, *args, **kwargs):
            return a + b + c + sum(args) + sum(kwargs.values())

        # Test with various argument combinations
        result = complex_func(1, 2, c=3, extra=4)
        assert result == 10  # 1 + 2 + 3 + 0 + 4

        result2 = complex_func(1, 2, 3, 4, 5, x=6, y=7)
        assert result2 == 28  # 1 + 2 + 3 + 4 + 5 + 6 + 7

        # Metrics should be recorded
        assert len(monitor.metrics["complex_op"]) == 2

    def test_get_memory_usage_without_psutil(self):
        """Test get_memory_usage when psutil is not available."""
        # Mock the HAS_PSUTIL flag
        with patch("networkx_mcp.utils.performance.HAS_PSUTIL", False):
            from networkx_mcp.utils.performance import PerformanceMonitor

            monitor = PerformanceMonitor()
            memory = monitor.get_memory_usage()

            # Should return dummy values
            assert memory["rss_mb"] == 0.0
            assert memory["vms_mb"] == 0.0
            assert memory["percent"] == 0.0
            assert memory["error"] == "psutil not available"

    def test_get_memory_usage_with_psutil(self):
        """Test get_memory_usage when psutil is available."""
        # Mock psutil
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB
        mock_memory_info.vms = 200 * 1024 * 1024  # 200 MB

        mock_process = Mock()
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.memory_percent.return_value = 5.5

        mock_psutil = Mock()
        mock_psutil.Process.return_value = mock_process

        with patch("networkx_mcp.utils.performance.HAS_PSUTIL", True):
            with patch("networkx_mcp.utils.performance.psutil", mock_psutil):
                from networkx_mcp.utils.performance import PerformanceMonitor

                monitor = PerformanceMonitor()
                memory = monitor.get_memory_usage()

                # Check values
                assert memory["rss_mb"] == 100.0
                assert memory["vms_mb"] == 200.0
                assert memory["percent"] == 5.5
                assert "error" not in memory

    def test_get_performance_summary_empty(self):
        """Test get_performance_summary with no metrics."""
        from networkx_mcp.utils.performance import PerformanceMonitor

        monitor = PerformanceMonitor()
        summary = monitor.get_performance_summary()

        assert isinstance(summary, dict)
        assert len(summary) == 0

    def test_get_performance_summary_with_data(self):
        """Test get_performance_summary with recorded metrics."""
        from networkx_mcp.utils.performance import PerformanceMonitor

        monitor = PerformanceMonitor()

        # Manually add some metrics
        monitor.metrics["op1"] = [0.1, 0.2, 0.15]
        monitor.metrics["op2"] = [0.5, 0.3, 0.4, 0.6, 0.2]

        summary = monitor.get_performance_summary()

        # Check op1 summary
        assert summary["op1"]["count"] == 3
        assert (
            abs(summary["op1"]["total_time"] - 0.45) < 0.0001
        )  # Handle float precision
        assert abs(summary["op1"]["avg_time"] - 0.15) < 0.0001
        assert summary["op1"]["max_time"] == 0.2
        assert summary["op1"]["min_time"] == 0.1

        # Check op2 summary
        assert summary["op2"]["count"] == 5
        assert (
            abs(summary["op2"]["total_time"] - 2.0) < 0.0001
        )  # Handle float precision
        assert abs(summary["op2"]["avg_time"] - 0.4) < 0.0001
        assert summary["op2"]["max_time"] == 0.6
        assert summary["op2"]["min_time"] == 0.2

    def test_multiple_monitors_independent(self):
        """Test that multiple monitors maintain independent metrics."""
        from networkx_mcp.utils.performance import PerformanceMonitor

        monitor1 = PerformanceMonitor()
        monitor2 = PerformanceMonitor()

        @monitor1.time_operation("op1")
        def func1():
            return 1

        @monitor2.time_operation("op2")
        def func2():
            return 2

        func1()
        func2()

        # Each monitor should only have its own metrics
        assert "op1" in monitor1.metrics
        assert "op1" not in monitor2.metrics
        assert "op2" in monitor2.metrics
        assert "op2" not in monitor1.metrics

    def test_wrapped_function_attributes(self):
        """Test that decorated functions preserve their attributes."""
        from networkx_mcp.utils.performance import PerformanceMonitor

        monitor = PerformanceMonitor()

        def original_func():
            """Original docstring."""
            return "result"

        original_func.custom_attr = "test"

        decorated = monitor.time_operation("test")(original_func)

        # Should preserve name and docstring (via @wraps)
        assert decorated.__name__ == "original_func"
        assert decorated.__doc__ == "Original docstring."

    def test_psutil_import_error_handling(self):
        """Test that the module handles psutil ImportError correctly."""
        import importlib
        import sys

        # Remove the module from sys.modules to force a fresh import
        if "networkx_mcp.utils.performance" in sys.modules:
            del sys.modules["networkx_mcp.utils.performance"]

        # Mock psutil to raise ImportError
        original_psutil = sys.modules.get("psutil")
        sys.modules["psutil"] = None  # This will cause ImportError

        try:
            # Import the module - this will trigger lines 12-14
            import networkx_mcp.utils.performance as perf_module

            # Verify HAS_PSUTIL is False
            assert perf_module.HAS_PSUTIL is False
            assert perf_module.psutil is None

            # Verify the monitor still works without psutil
            monitor = perf_module.PerformanceMonitor()
            memory = monitor.get_memory_usage()
            assert memory["error"] == "psutil not available"

        finally:
            # Restore original psutil module
            if original_psutil is not None:
                sys.modules["psutil"] = original_psutil
            else:
                sys.modules.pop("psutil", None)

            # Force reload to restore normal state
            if "networkx_mcp.utils.performance" in sys.modules:
                importlib.reload(sys.modules["networkx_mcp.utils.performance"])
