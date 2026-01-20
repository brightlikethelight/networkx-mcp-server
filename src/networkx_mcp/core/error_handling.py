"""
Comprehensive error handling and recovery system for NetworkX MCP Server.

This module provides error classification, recovery strategies, and automatic
fallback mechanisms for robust graph operations.
"""

import asyncio
import logging
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GraphErrorType(Enum):
    """Types of errors that can occur in graph operations"""

    # Resource errors
    MEMORY_EXHAUSTED = "memory_exhausted"
    GPU_OUT_OF_MEMORY = "gpu_out_of_memory"
    CPU_OVERLOAD = "cpu_overload"

    # Graph errors
    GRAPH_NOT_FOUND = "graph_not_found"
    GRAPH_TOO_LARGE = "graph_too_large"
    GRAPH_CORRUPTED = "graph_corrupted"
    INVALID_GRAPH_OPERATION = "invalid_graph_operation"

    # Computation errors
    COMPUTATION_TIMEOUT = "computation_timeout"
    ALGORITHM_FAILURE = "algorithm_failure"
    NUMERICAL_INSTABILITY = "numerical_instability"
    CONVERGENCE_FAILURE = "convergence_failure"

    # Plugin errors
    PLUGIN_NOT_FOUND = "plugin_not_found"
    PLUGIN_INITIALIZATION_FAILED = "plugin_initialization_failed"
    PLUGIN_DEPENDENCY_MISSING = "plugin_dependency_missing"
    PLUGIN_CRASHED = "plugin_crashed"

    # Network errors
    NETWORK_TIMEOUT = "network_timeout"
    CONNECTION_LOST = "connection_lost"
    API_RATE_LIMIT = "api_rate_limit"

    # Data errors
    INVALID_INPUT = "invalid_input"
    DATA_CORRUPTION = "data_corruption"
    SERIALIZATION_ERROR = "serialization_error"

    # System errors
    PERMISSION_DENIED = "permission_denied"
    DISK_FULL = "disk_full"
    SYSTEM_ERROR = "system_error"


class ErrorSeverity(Enum):
    """Error severity levels"""

    LOW = 1  # Can be ignored or logged
    MEDIUM = 2  # Should be handled but not critical
    HIGH = 3  # Requires immediate attention
    CRITICAL = 4  # System failure, requires recovery


class ErrorRecoveryStrategy(Enum):
    """Strategies for error recovery"""

    RETRY = "retry"  # Simple retry
    RETRY_WITH_BACKOFF = "retry_with_backoff"  # Exponential backoff
    FALLBACK_TO_CPU = "fallback_to_cpu"  # GPU -> CPU fallback
    REDUCE_MEMORY = "reduce_memory"  # Reduce memory usage
    CHUNK_OPERATION = "chunk_operation"  # Split into smaller operations
    USE_CACHE = "use_cache"  # Return cached result
    APPROXIMATE = "approximate"  # Use approximate algorithm
    GRACEFUL_DEGRADATION = "graceful_degradation"  # Reduced functionality
    ABORT = "abort"  # Cannot recover


@dataclass
class GraphError(Exception):
    """Enhanced exception class for graph operations"""

    error_type: GraphErrorType
    message: str
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    recoverable: bool = True
    recovery_strategies: List[ErrorRecoveryStrategy] = field(default_factory=list)
    suggested_action: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    stack_trace: Optional[str] = None

    def __post_init__(self):
        """Capture stack trace on error creation"""
        if self.stack_trace is None:
            self.stack_trace = traceback.format_exc()

        # Set default recovery strategies based on error type
        if not self.recovery_strategies:
            self.recovery_strategies = self._get_default_strategies()

    def _get_default_strategies(self) -> List[ErrorRecoveryStrategy]:
        """Get default recovery strategies for error type"""
        strategies_map = {
            GraphErrorType.MEMORY_EXHAUSTED: [
                ErrorRecoveryStrategy.REDUCE_MEMORY,
                ErrorRecoveryStrategy.CHUNK_OPERATION,
                ErrorRecoveryStrategy.USE_CACHE,
            ],
            GraphErrorType.GPU_OUT_OF_MEMORY: [
                ErrorRecoveryStrategy.FALLBACK_TO_CPU,
                ErrorRecoveryStrategy.REDUCE_MEMORY,
                ErrorRecoveryStrategy.CHUNK_OPERATION,
            ],
            GraphErrorType.COMPUTATION_TIMEOUT: [
                ErrorRecoveryStrategy.APPROXIMATE,
                ErrorRecoveryStrategy.CHUNK_OPERATION,
                ErrorRecoveryStrategy.USE_CACHE,
            ],
            GraphErrorType.NETWORK_TIMEOUT: [
                ErrorRecoveryStrategy.RETRY_WITH_BACKOFF,
                ErrorRecoveryStrategy.USE_CACHE,
            ],
            GraphErrorType.API_RATE_LIMIT: [
                ErrorRecoveryStrategy.RETRY_WITH_BACKOFF,
                ErrorRecoveryStrategy.USE_CACHE,
            ],
        }

        return strategies_map.get(self.error_type, [ErrorRecoveryStrategy.RETRY])

    def __str__(self):
        return f"GraphError({self.error_type.value}): {self.message}"


@dataclass
class ErrorMetrics:
    """Metrics for error tracking"""

    total_errors: int = 0
    errors_by_type: Dict[GraphErrorType, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    errors_by_severity: Dict[ErrorSeverity, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    last_error_time: Optional[datetime] = None
    error_rate_per_minute: float = 0.0


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = GraphError,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open

    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise GraphError(
                    error_type=GraphErrorType.SYSTEM_ERROR,
                    message="Circuit breaker is open",
                    severity=ErrorSeverity.HIGH,
                    recoverable=False,
                )

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result

        except self.expected_exception:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset the circuit"""
        if self.last_failure_time is None:
            return True

        time_since_failure = (datetime.now() - self.last_failure_time).seconds
        return time_since_failure >= self.recovery_timeout

    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = "closed"

    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )


class ErrorHandler:
    """
    Central error handling and recovery system.

    Features:
    - Error classification and severity assessment
    - Automatic recovery strategies
    - Circuit breaker pattern
    - Error metrics and monitoring
    - Fallback mechanisms
    """

    def __init__(self):
        self.metrics = ErrorMetrics()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.recovery_handlers: Dict[ErrorRecoveryStrategy, Callable] = {
            ErrorRecoveryStrategy.RETRY: self._retry_recovery,
            ErrorRecoveryStrategy.RETRY_WITH_BACKOFF: self._retry_with_backoff,
            ErrorRecoveryStrategy.FALLBACK_TO_CPU: self._fallback_to_cpu,
            ErrorRecoveryStrategy.REDUCE_MEMORY: self._reduce_memory,
            ErrorRecoveryStrategy.CHUNK_OPERATION: self._chunk_operation,
            ErrorRecoveryStrategy.USE_CACHE: self._use_cache,
            ErrorRecoveryStrategy.APPROXIMATE: self._approximate,
            ErrorRecoveryStrategy.GRACEFUL_DEGRADATION: self._graceful_degradation,
        }

        # Error history for pattern detection
        self.error_history: List[GraphError] = []
        self.max_history_size = 1000

        logger.info("Initialized error handler with recovery strategies")

    def classify_error(self, exception: Exception) -> GraphError:
        """Classify a generic exception into GraphError"""
        if isinstance(exception, GraphError):
            return exception

        # Memory errors
        if isinstance(exception, MemoryError):
            return GraphError(
                error_type=GraphErrorType.MEMORY_EXHAUSTED,
                message=str(exception),
                severity=ErrorSeverity.HIGH,
                recovery_strategies=[
                    ErrorRecoveryStrategy.REDUCE_MEMORY,
                    ErrorRecoveryStrategy.CHUNK_OPERATION,
                ],
            )

        # Timeout errors
        if isinstance(exception, asyncio.TimeoutError):
            return GraphError(
                error_type=GraphErrorType.COMPUTATION_TIMEOUT,
                message=str(exception),
                severity=ErrorSeverity.MEDIUM,
                recovery_strategies=[
                    ErrorRecoveryStrategy.RETRY,
                    ErrorRecoveryStrategy.APPROXIMATE,
                ],
            )

        # Network errors
        if "connection" in str(exception).lower():
            return GraphError(
                error_type=GraphErrorType.CONNECTION_LOST,
                message=str(exception),
                severity=ErrorSeverity.MEDIUM,
                recovery_strategies=[ErrorRecoveryStrategy.RETRY_WITH_BACKOFF],
            )

        # Generic errors
        return GraphError(
            error_type=GraphErrorType.SYSTEM_ERROR,
            message=str(exception),
            severity=ErrorSeverity.MEDIUM,
            recovery_strategies=[ErrorRecoveryStrategy.RETRY],
        )

    async def handle_error(self, error: GraphError) -> Optional[Any]:
        """
        Handle an error with automatic recovery attempts.

        Args:
            error: The error to handle

        Returns:
            Recovery result if successful, None otherwise
        """
        # Update metrics
        self.metrics.total_errors += 1
        self.metrics.errors_by_type[error.error_type] += 1
        self.metrics.errors_by_severity[error.severity] += 1
        self.metrics.last_error_time = datetime.now()

        # Add to history
        self.error_history.append(error)
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)

        # Log error
        log_method = {
            ErrorSeverity.LOW: logger.debug,
            ErrorSeverity.MEDIUM: logger.warning,
            ErrorSeverity.HIGH: logger.error,
            ErrorSeverity.CRITICAL: logger.critical,
        }[error.severity]

        log_method(f"Error occurred: {error}")

        # Check if recoverable
        if not error.recoverable:
            logger.error(f"Error is not recoverable: {error.error_type.value}")
            return None

        # Try recovery strategies
        for strategy in error.recovery_strategies:
            try:
                self.metrics.recovery_attempts += 1
                logger.info(f"Attempting recovery strategy: {strategy.value}")

                if strategy in self.recovery_handlers:
                    result = await self.recovery_handlers[strategy](error)
                    if result is not None:
                        self.metrics.successful_recoveries += 1
                        logger.info(
                            f"Recovery successful with strategy: {strategy.value}"
                        )
                        return result

            except Exception as e:
                logger.error(f"Recovery strategy {strategy.value} failed: {e}")
                self.metrics.failed_recoveries += 1

        # All strategies failed
        return None

    async def handle_with_recovery(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with error handling and recovery.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result or recovery result
        """
        try:
            return await func(*args, **kwargs)

        except Exception as e:
            error = self.classify_error(e)
            recovery_result = await self.handle_error(error)

            if recovery_result is not None:
                return recovery_result

            # Re-raise if no recovery possible
            raise error

    async def recover(
        self, error: GraphError, context: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Attempt to recover from an error with context.

        Args:
            error: The error to recover from
            context: Context information for recovery

        Returns:
            Recovery result if successful
        """
        error.context.update(context)
        return await self.handle_error(error)

    # Recovery strategy implementations

    async def _retry_recovery(self, error: GraphError) -> Optional[Any]:
        """Simple retry recovery"""
        if "retry_count" not in error.context:
            error.context["retry_count"] = 0

        if error.context["retry_count"] < 3:
            error.context["retry_count"] += 1
            await asyncio.sleep(1)
            # Would need to re-execute the original operation
            return None

        return None

    async def _retry_with_backoff(self, error: GraphError) -> Optional[Any]:
        """Retry with exponential backoff"""
        if "retry_count" not in error.context:
            error.context["retry_count"] = 0

        if error.context["retry_count"] < 5:
            delay = 2 ** error.context["retry_count"]
            error.context["retry_count"] += 1
            logger.info(f"Retrying after {delay} seconds")
            await asyncio.sleep(delay)
            # Would need to re-execute the original operation
            return None

        return None

    async def _fallback_to_cpu(self, error: GraphError) -> Optional[Any]:
        """Fallback from GPU to CPU execution"""
        logger.info("Falling back to CPU execution")
        error.context["use_gpu"] = False
        # Would need to re-execute on CPU
        return None

    async def _reduce_memory(self, error: GraphError) -> Optional[Any]:
        """Reduce memory usage and retry"""
        logger.info("Attempting to reduce memory usage")
        # Would implement memory reduction strategies
        return None

    async def _chunk_operation(self, error: GraphError) -> Optional[Any]:
        """Split operation into smaller chunks"""
        logger.info("Chunking operation for reduced memory usage")
        # Would implement operation chunking
        return None

    async def _use_cache(self, error: GraphError) -> Optional[Any]:
        """Use cached result if available"""
        logger.info("Checking for cached result")
        # Would check cache
        return None

    async def _approximate(self, error: GraphError) -> Optional[Any]:
        """Use approximate algorithm"""
        logger.info("Using approximate algorithm")
        # Would use approximate version
        return None

    async def _graceful_degradation(self, error: GraphError) -> Optional[Any]:
        """Gracefully degrade functionality"""
        logger.info("Gracefully degrading functionality")
        # Would return partial result
        return None

    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker()
        return self.circuit_breakers[name]

    def detect_error_patterns(self) -> List[Tuple[GraphErrorType, int]]:
        """Detect recurring error patterns"""
        if not self.error_history:
            return []

        # Count errors in last 5 minutes
        cutoff = datetime.now() - timedelta(minutes=5)
        recent_errors = [e for e in self.error_history if e.timestamp > cutoff]

        error_counts = defaultdict(int)
        for error in recent_errors:
            error_counts[error.error_type] += 1

        # Find patterns (more than 3 occurrences)
        patterns = [
            (error_type, count)
            for error_type, count in error_counts.items()
            if count > 3
        ]

        if patterns:
            logger.warning(f"Detected error patterns: {patterns}")

        return patterns

    def get_stats(self) -> Dict[str, Any]:
        """Get error handler statistics"""
        return {
            "total_errors": self.metrics.total_errors,
            "errors_by_type": dict(self.metrics.errors_by_type),
            "errors_by_severity": dict(self.metrics.errors_by_severity),
            "recovery_rate": self.metrics.successful_recoveries
            / max(self.metrics.recovery_attempts, 1),
            "circuit_breakers": {
                name: cb.state for name, cb in self.circuit_breakers.items()
            },
            "error_patterns": self.detect_error_patterns(),
        }


# Example usage
async def test_error_handling():
    """Test error handling system"""
    handler = ErrorHandler()

    # Simulate memory error
    error = GraphError(
        error_type=GraphErrorType.MEMORY_EXHAUSTED,
        message="Out of memory processing large graph",
        severity=ErrorSeverity.HIGH,
        context={"graph_size": 1000000},
    )

    result = await handler.handle_error(error)
    print(f"Recovery result: {result}")

    # Test circuit breaker
    breaker = handler.get_circuit_breaker("test_operation")

    async def failing_operation():
        raise GraphError(
            error_type=GraphErrorType.ALGORITHM_FAILURE,
            message="Algorithm failed to converge",
        )

    for i in range(10):
        try:
            await breaker.call(failing_operation)
        except GraphError as e:
            print(f"Attempt {i + 1}: {e}")

    # Get statistics
    stats = handler.get_stats()
    print(f"Error handler stats: {stats}")


if __name__ == "__main__":
    asyncio.run(test_error_handling())
