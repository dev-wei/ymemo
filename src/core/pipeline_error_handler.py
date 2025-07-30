"""Error handling and resilience patterns for audio processing pipeline."""

import asyncio
import logging
import traceback
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar, Generic
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from ..utils.exceptions import PipelineError, PipelineTimeoutError, ResourceCleanupError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels for pipeline operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RetryStrategy(Enum):
    """Retry strategies for failed operations."""
    NONE = "none"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIXED_DELAY = "fixed_delay"


class PipelineErrorHandler:
    """
    Centralized error handling and resilience patterns for audio processing pipeline.
    
    Provides consistent error handling, retry logic, timeout management,
    and structured logging for pipeline operations.
    """
    
    def __init__(self, 
                 default_timeout: float = 30.0,
                 max_retries: int = 3,
                 base_retry_delay: float = 1.0):
        """
        Initialize pipeline error handler.
        
        Args:
            default_timeout: Default timeout for pipeline operations
            max_retries: Maximum number of retry attempts
            base_retry_delay: Base delay between retry attempts
        """
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self.error_counts: Dict[str, int] = {}
        self.last_error_times: Dict[str, datetime] = {}
    
    @asynccontextmanager
    async def handle_pipeline_operation(self,
                                        operation_name: str,
                                        timeout: Optional[float] = None,
                                        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                                        retry_strategy: RetryStrategy = RetryStrategy.NONE,
                                        cleanup_callback: Optional[Callable[[], None]] = None):
        """
        Context manager for handling pipeline operations with consistent error handling.
        
        Args:
            operation_name: Name of the operation for logging
            timeout: Operation timeout (uses default if None)
            severity: Error severity level
            retry_strategy: Retry strategy for failures
            cleanup_callback: Optional cleanup callback for failures
        
        Usage:
            async with error_handler.handle_pipeline_operation("transcription_start"):
                await transcription_provider.start_stream()
        """
        operation_timeout = timeout or self.default_timeout
        start_time = datetime.now()
        
        logger.info(f"🔄 Pipeline: Starting operation '{operation_name}' (timeout: {operation_timeout}s)")
        
        try:
            # Execute with timeout
            await asyncio.wait_for(
                self._execute_operation(operation_name, retry_strategy),
                timeout=operation_timeout
            )
            
            # Operation succeeded
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✅ Pipeline: Operation '{operation_name}' completed successfully in {duration:.2f}s")
            
            # Reset error count on success
            self.error_counts.pop(operation_name, None)
            
            yield
            
        except asyncio.TimeoutError as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"Operation '{operation_name}' timed out after {duration:.2f}s (limit: {operation_timeout}s)"
            
            logger.error(f"⏱️ Pipeline: {error_msg}")
            self._record_error(operation_name, severity)
            
            # Execute cleanup if provided
            if cleanup_callback:
                try:
                    cleanup_callback()
                    logger.info(f"🧹 Pipeline: Cleanup executed for '{operation_name}'")
                except Exception as cleanup_error:
                    logger.error(f"❌ Pipeline: Cleanup failed for '{operation_name}': {cleanup_error}")
                    raise ResourceCleanupError(f"Cleanup failed for {operation_name}") from cleanup_error
            
            raise PipelineTimeoutError(error_msg, operation_timeout, e)
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_msg = f"Operation '{operation_name}' failed after {duration:.2f}s: {str(e)}"
            
            logger.error(f"❌ Pipeline: {error_msg}")
            logger.debug(f"❌ Pipeline: Full traceback for '{operation_name}':\n{traceback.format_exc()}")
            
            self._record_error(operation_name, severity)
            
            # Execute cleanup if provided
            if cleanup_callback:
                try:
                    cleanup_callback()
                    logger.info(f"🧹 Pipeline: Cleanup executed for '{operation_name}'")
                except Exception as cleanup_error:
                    logger.error(f"❌ Pipeline: Cleanup failed for '{operation_name}': {cleanup_error}")
                    raise ResourceCleanupError(f"Cleanup failed for {operation_name}") from cleanup_error
            
            # Wrap in pipeline error for consistent handling
            if isinstance(e, (PipelineError, PipelineTimeoutError)):
                raise
            else:
                raise PipelineError(f"Pipeline operation '{operation_name}' failed: {str(e)}") from e
    
    async def _execute_operation(self, operation_name: str, retry_strategy: RetryStrategy):
        """Execute operation with retry logic if configured."""
        if retry_strategy == RetryStrategy.NONE:
            return  # No retry, just execute once
        
        last_exception = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                return  # Operation succeeded
                
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    logger.error(f"❌ Pipeline: Operation '{operation_name}' failed after {self.max_retries} attempts")
                    break
                
                # Calculate retry delay
                delay = self._calculate_retry_delay(retry_strategy, attempt)
                
                logger.warning(f"⚠️ Pipeline: Operation '{operation_name}' failed (attempt {attempt}/{self.max_retries}), "
                             f"retrying in {delay:.2f}s: {str(e)}")
                
                await asyncio.sleep(delay)
        
        # All retries exhausted
        raise last_exception
    
    def _calculate_retry_delay(self, strategy: RetryStrategy, attempt: int) -> float:
        """Calculate retry delay based on strategy."""
        if strategy == RetryStrategy.LINEAR:
            return self.base_retry_delay * attempt
        elif strategy == RetryStrategy.EXPONENTIAL:
            return self.base_retry_delay * (2 ** (attempt - 1))
        elif strategy == RetryStrategy.FIXED_DELAY:
            return self.base_retry_delay
        else:
            return self.base_retry_delay
    
    def _record_error(self, operation_name: str, severity: ErrorSeverity):
        """Record error occurrence for monitoring."""
        self.error_counts[operation_name] = self.error_counts.get(operation_name, 0) + 1
        self.last_error_times[operation_name] = datetime.now()
        
        logger.info(f"📊 Pipeline: Error recorded for '{operation_name}' "
                   f"(count: {self.error_counts[operation_name]}, severity: {severity.value})")
    
    async def safe_cleanup(self, 
                          cleanup_operations: Dict[str, Callable],
                          timeout_per_operation: float = 5.0) -> Dict[str, bool]:
        """
        Safely execute multiple cleanup operations with individual timeouts.
        
        Args:
            cleanup_operations: Dict of operation_name -> cleanup_function
            timeout_per_operation: Timeout for each cleanup operation
            
        Returns:
            Dict of operation_name -> success_status
        """
        results = {}
        
        for operation_name, cleanup_func in cleanup_operations.items():
            try:
                logger.info(f"🧹 Pipeline: Starting cleanup for '{operation_name}'")
                
                if asyncio.iscoroutinefunction(cleanup_func):
                    await asyncio.wait_for(cleanup_func(), timeout=timeout_per_operation)
                else:
                    cleanup_func()
                
                results[operation_name] = True
                logger.info(f"✅ Pipeline: Cleanup completed for '{operation_name}'")
                
            except asyncio.TimeoutError:
                results[operation_name] = False
                logger.error(f"⏱️ Pipeline: Cleanup timeout for '{operation_name}' after {timeout_per_operation}s")
                
            except Exception as e:
                results[operation_name] = False
                logger.error(f"❌ Pipeline: Cleanup failed for '{operation_name}': {str(e)}")
                logger.debug(f"❌ Pipeline: Cleanup traceback for '{operation_name}':\n{traceback.format_exc()}")
        
        successful_cleanups = sum(results.values())
        total_cleanups = len(results)
        
        logger.info(f"🧹 Pipeline: Cleanup summary: {successful_cleanups}/{total_cleanups} operations successful")
        
        return results
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error counts and patterns."""
        return {
            'error_counts': self.error_counts.copy(),
            'last_error_times': {
                op: time.isoformat() for op, time in self.last_error_times.items()
            },
            'total_errors': sum(self.error_counts.values()),
            'operations_with_errors': len(self.error_counts)
        }
    
    def should_circuit_break(self, operation_name: str, 
                           error_threshold: int = 5, 
                           time_window_minutes: int = 5) -> bool:
        """
        Determine if operation should be circuit broken due to repeated failures.
        
        Args:
            operation_name: Name of the operation to check
            error_threshold: Number of errors to trigger circuit breaker
            time_window_minutes: Time window to consider for error counting
            
        Returns:
            True if operation should be circuit broken
        """
        error_count = self.error_counts.get(operation_name, 0)
        last_error_time = self.last_error_times.get(operation_name)
        
        if error_count < error_threshold:
            return False
        
        if not last_error_time:
            return False
        
        time_window = timedelta(minutes=time_window_minutes)
        if datetime.now() - last_error_time > time_window:
            # Errors are outside time window, reset counter
            self.error_counts[operation_name] = 0
            return False
        
        logger.warning(f"⚡ Pipeline: Circuit breaker triggered for '{operation_name}' "
                      f"({error_count} errors in {time_window_minutes} minutes)")
        
        return True