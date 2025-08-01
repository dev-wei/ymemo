"""Enhanced error handling for dual AWS Transcribe connections."""

import asyncio
import logging
import time
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

from ..utils.exceptions import AWSTranscribeError, TranscriptionProviderError


logger = logging.getLogger(__name__)


class FallbackStrategy(Enum):
    """Strategy for handling dual connection failures."""
    MONO_FALLBACK = "mono_fallback"  # Fall back to single connection
    CHANNEL_PRIORITY = "channel_priority"  # Use priority channel only
    RETRY_DUAL = "retry_dual"  # Retry dual connection
    FAIL_FAST = "fail_fast"  # Fail immediately


class ConnectionHealth(Enum):
    """Health status of individual connections."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Working but with issues
    FAILING = "failing"  # Repeated failures
    FAILED = "failed"  # Completely failed


@dataclass
class ConnectionMetrics:
    """Metrics for monitoring connection health."""
    connection_attempts: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    results_received: int = 0
    bytes_sent: int = 0
    average_latency: float = 0.0
    last_success_time: float = 0.0
    last_failure_time: float = 0.0
    consecutive_failures: int = 0
    error_history: List[str] = field(default_factory=list)


@dataclass
class DualConnectionStatus:
    """Overall status of dual connection system."""
    left_health: ConnectionHealth = ConnectionHealth.HEALTHY
    right_health: ConnectionHealth = ConnectionHealth.HEALTHY
    fallback_active: bool = False
    fallback_strategy: Optional[FallbackStrategy] = None
    active_channel: Optional[str] = None
    total_errors: int = 0
    uptime: float = 0.0


class DualConnectionErrorHandler:
    """
    Enhanced error handler for dual AWS Transcribe connections.
    
    This handler provides sophisticated error recovery, health monitoring,
    and fallback strategies for dual-channel transcription systems.
    """
    
    def __init__(
        self,
        fallback_strategy: FallbackStrategy = FallbackStrategy.MONO_FALLBACK,
        health_check_interval: float = 5.0,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        max_error_history: int = 10,
        priority_channel: Optional[str] = None
    ):
        """
        Initialize dual connection error handler.
        
        Args:
            fallback_strategy: Strategy for handling connection failures
            health_check_interval: Interval between health checks in seconds
            failure_threshold: Number of consecutive failures before marking as failed
            recovery_timeout: Time to wait before considering recovery
            max_error_history: Maximum errors to keep in history
            priority_channel: Preferred channel for fallback ('left' or 'right')
        """
        self.fallback_strategy = fallback_strategy
        self.health_check_interval = health_check_interval
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.max_error_history = max_error_history
        self.priority_channel = priority_channel
        
        # Connection metrics
        self.left_metrics = ConnectionMetrics()
        self.right_metrics = ConnectionMetrics()
        
        # System status
        self.status = DualConnectionStatus()
        self.start_time = 0.0
        
        # Error callbacks
        self.error_callback: Optional[Callable[[str, Exception], None]] = None
        self.health_change_callback: Optional[Callable[[DualConnectionStatus], None]] = None
        self.fallback_callback: Optional[Callable[[FallbackStrategy, str], None]] = None
        
        # Monitoring
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Retry management
        self.retry_delays = [1.0, 2.0, 5.0, 10.0, 30.0]  # Exponential backoff
        self.max_retry_delay = 60.0
        
        logger.info(f"🛡️ DualConnectionErrorHandler initialized:")
        logger.info(f"   📋 Strategy: {fallback_strategy.value}")
        logger.info(f"   🔍 Health check interval: {health_check_interval}s")
        logger.info(f"   🚨 Failure threshold: {failure_threshold}")
        logger.info(f"   ⏱️  Recovery timeout: {recovery_timeout}s")
    
    async def start_monitoring(self) -> None:
        """Start health monitoring."""
        if self.is_monitoring:
            logger.warning("⚠️ Error handler already monitoring")
            return
        
        logger.info("🚀 Error Handler: Starting health monitoring")
        
        self.is_monitoring = True
        self.start_time = time.time()
        
        # Start monitoring task
        self.monitor_task = asyncio.create_task(self._monitor_connections())
        
        logger.info("✅ Error Handler: Health monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        logger.info("🛑 Error Handler: Stopping health monitoring")
        
        self.is_monitoring = False
        
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        self._log_final_statistics()
        logger.info("✅ Error Handler: Health monitoring stopped")
    
    def record_connection_attempt(self, channel: str) -> None:
        """Record a connection attempt."""
        metrics = self._get_channel_metrics(channel)
        metrics.connection_attempts += 1
        
        logger.debug(f"📊 {channel.title()} channel: Connection attempt #{metrics.connection_attempts}")
    
    def record_connection_success(self, channel: str) -> None:
        """Record successful connection."""
        metrics = self._get_channel_metrics(channel)
        metrics.successful_connections += 1
        metrics.last_success_time = time.time()
        metrics.consecutive_failures = 0  # Reset failure count
        
        # Update health status
        if channel == "left":
            self.status.left_health = ConnectionHealth.HEALTHY
        else:
            self.status.right_health = ConnectionHealth.HEALTHY
        
        logger.info(f"✅ {channel.title()} channel: Connection successful")
        self._check_fallback_recovery()
    
    def record_connection_failure(self, channel: str, error: Exception) -> None:
        """Record connection failure."""
        metrics = self._get_channel_metrics(channel)
        metrics.failed_connections += 1
        metrics.last_failure_time = time.time()
        metrics.consecutive_failures += 1
        
        # Add to error history
        error_msg = str(error)
        metrics.error_history.append(error_msg)
        if len(metrics.error_history) > self.max_error_history:
            metrics.error_history.pop(0)
        
        # Update health status based on consecutive failures
        health = self._calculate_health_status(metrics.consecutive_failures)
        if channel == "left":
            self.status.left_health = health
        else:
            self.status.right_health = health
        
        logger.error(f"❌ {channel.title()} channel: Connection failed (#{metrics.consecutive_failures}): {error}")
        
        # Check if fallback is needed
        self._check_fallback_needed()
        
        # Notify error callback
        if self.error_callback:
            try:
                self.error_callback(channel, error)
            except Exception as e:
                logger.error(f"❌ Error callback failed: {e}")
    
    def record_result_received(self, channel: str, latency: Optional[float] = None) -> None:
        """Record successful result reception."""
        metrics = self._get_channel_metrics(channel)
        metrics.results_received += 1
        
        # Update latency average
        if latency is not None:
            if metrics.average_latency == 0.0:
                metrics.average_latency = latency
            else:
                # Simple moving average
                metrics.average_latency = (metrics.average_latency * 0.9) + (latency * 0.1)
        
        logger.debug(f"📝 {channel.title()} channel: Result received (#{metrics.results_received})")
    
    def record_bytes_sent(self, channel: str, bytes_count: int) -> None:
        """Record bytes sent to channel."""
        metrics = self._get_channel_metrics(channel)
        metrics.bytes_sent += bytes_count
    
    def _get_channel_metrics(self, channel: str) -> ConnectionMetrics:
        """Get metrics for specified channel."""
        if channel.lower() == "left":
            return self.left_metrics
        else:
            return self.right_metrics
    
    def _calculate_health_status(self, consecutive_failures: int) -> ConnectionHealth:
        """Calculate health status based on consecutive failures."""
        if consecutive_failures == 0:
            return ConnectionHealth.HEALTHY
        elif consecutive_failures < self.failure_threshold // 2:
            return ConnectionHealth.DEGRADED
        elif consecutive_failures < self.failure_threshold:
            return ConnectionHealth.FAILING
        else:
            return ConnectionHealth.FAILED
    
    def _check_fallback_needed(self) -> None:
        """Check if fallback mode should be activated."""
        left_failed = self.status.left_health == ConnectionHealth.FAILED
        right_failed = self.status.right_health == ConnectionHealth.FAILED
        
        if left_failed and right_failed:
            logger.error("❌ Both channels failed - complete transcription failure")
            self._activate_fallback(FallbackStrategy.FAIL_FAST, "both_channels_failed")
        elif left_failed and not self.status.fallback_active:
            logger.warning("⚠️ Left channel failed - activating right channel fallback")
            self._activate_fallback(self.fallback_strategy, "right")
        elif right_failed and not self.status.fallback_active:
            logger.warning("⚠️ Right channel failed - activating left channel fallback")  
            self._activate_fallback(self.fallback_strategy, "left")
    
    def _check_fallback_recovery(self) -> None:
        """Check if we can recover from fallback mode."""
        if not self.status.fallback_active:
            return
        
        left_healthy = self.status.left_health in [ConnectionHealth.HEALTHY, ConnectionHealth.DEGRADED]
        right_healthy = self.status.right_health in [ConnectionHealth.HEALTHY, ConnectionHealth.DEGRADED]
        
        if left_healthy and right_healthy:
            logger.info("✅ Both channels recovered - deactivating fallback mode")
            self.status.fallback_active = False
            self.status.fallback_strategy = None
            self.status.active_channel = None
            
            if self.fallback_callback:
                try:
                    self.fallback_callback(None, "recovered")
                except Exception as e:
                    logger.error(f"❌ Fallback callback error: {e}")
    
    def _activate_fallback(self, strategy: FallbackStrategy, active_channel: str) -> None:
        """Activate fallback mode with specified strategy."""
        if self.status.fallback_active:
            return  # Already in fallback mode
        
        self.status.fallback_active = True
        self.status.fallback_strategy = strategy
        self.status.active_channel = active_channel
        
        logger.warning(f"🔄 Activated fallback mode: {strategy.value} using {active_channel} channel")
        
        # Notify fallback callback
        if self.fallback_callback:
            try:
                self.fallback_callback(strategy, active_channel)
            except Exception as e:
                logger.error(f"❌ Fallback callback error: {e}")
        
        # Notify health change
        if self.health_change_callback:
            try:
                self.health_change_callback(self.status)
            except Exception as e:
                logger.error(f"❌ Health change callback error: {e}")
    
    async def _monitor_connections(self) -> None:
        """Main monitoring loop."""
        try:
            logger.info("🔍 Error Handler: Starting connection monitoring loop")
            
            while self.is_monitoring:
                current_time = time.time()
                
                # Update uptime
                self.status.uptime = current_time - self.start_time
                
                # Check for stale connections (no recent success)
                self._check_stale_connections(current_time)
                
                # Log periodic health summary
                if int(self.status.uptime) % 30 == 0:  # Every 30 seconds
                    self._log_health_summary()
                
                await asyncio.sleep(self.health_check_interval)
                
        except asyncio.CancelledError:
            logger.info("🛑 Error Handler: Connection monitoring cancelled")
        except Exception as e:
            logger.error(f"❌ Error Handler: Monitoring error: {e}")
    
    def _check_stale_connections(self, current_time: float) -> None:
        """Check for connections that haven't had recent activity."""
        stale_timeout = 60.0  # Consider stale after 60 seconds
        
        # Check left channel
        if (self.left_metrics.last_success_time > 0 and 
            current_time - self.left_metrics.last_success_time > stale_timeout and
            self.status.left_health == ConnectionHealth.HEALTHY):
            
            logger.warning(f"⚠️ Left channel: No recent activity ({current_time - self.left_metrics.last_success_time:.0f}s)")
            self.status.left_health = ConnectionHealth.DEGRADED
        
        # Check right channel
        if (self.right_metrics.last_success_time > 0 and 
            current_time - self.right_metrics.last_success_time > stale_timeout and
            self.status.right_health == ConnectionHealth.HEALTHY):
            
            logger.warning(f"⚠️ Right channel: No recent activity ({current_time - self.right_metrics.last_success_time:.0f}s)")
            self.status.right_health = ConnectionHealth.DEGRADED
    
    def _log_health_summary(self) -> None:
        """Log periodic health summary."""
        logger.info(f"🔍 Connection Health Summary (uptime: {self.status.uptime:.0f}s):")
        logger.info(f"   🎚️  Left Channel: {self.status.left_health.value} "
                   f"({self.left_metrics.results_received} results, {self.left_metrics.consecutive_failures} failures)")
        logger.info(f"   🎚️  Right Channel: {self.status.right_health.value} "
                   f"({self.right_metrics.results_received} results, {self.right_metrics.consecutive_failures} failures)")
        logger.info(f"   🔄 Fallback: {self.status.fallback_active} "
                   f"({'active on ' + self.status.active_channel if self.status.fallback_active else 'disabled'})")
    
    def _log_final_statistics(self) -> None:
        """Log final statistics."""
        logger.info(f"📊 Final Dual Connection Statistics:")
        logger.info(f"   ⏱️  Total Uptime: {self.status.uptime:.1f}s")
        logger.info(f"   📊 Left Channel: {self.left_metrics.connection_attempts} attempts, "
                   f"{self.left_metrics.successful_connections} success, {self.left_metrics.failed_connections} failed")
        logger.info(f"   📊 Right Channel: {self.right_metrics.connection_attempts} attempts, "
                   f"{self.right_metrics.successful_connections} success, {self.right_metrics.failed_connections} failed")
        logger.info(f"   📝 Total Results: Left={self.left_metrics.results_received}, Right={self.right_metrics.results_received}")
        logger.info(f"   📡 Data Sent: Left={self.left_metrics.bytes_sent:,} bytes, Right={self.right_metrics.bytes_sent:,} bytes")
        logger.info(f"   🔄 Fallback Activations: {1 if self.status.fallback_active else 0}")
    
    def get_retry_delay(self, channel: str) -> float:
        """Get appropriate retry delay for channel."""
        metrics = self._get_channel_metrics(channel)
        failure_count = min(metrics.consecutive_failures - 1, len(self.retry_delays) - 1)
        
        if failure_count < 0:
            return 0.0
        
        delay = self.retry_delays[failure_count]
        return min(delay, self.max_retry_delay)
    
    def should_retry(self, channel: str) -> bool:
        """Determine if connection should be retried."""
        if self.fallback_strategy == FallbackStrategy.FAIL_FAST:
            return False
        
        metrics = self._get_channel_metrics(channel)
        
        # Don't retry if too many consecutive failures
        if metrics.consecutive_failures > len(self.retry_delays):
            return False
        
        # Check if enough time has passed since last failure
        if metrics.last_failure_time > 0:
            time_since_failure = time.time() - metrics.last_failure_time
            required_delay = self.get_retry_delay(channel)
            
            return time_since_failure >= required_delay
        
        return True
    
    def get_status(self) -> DualConnectionStatus:
        """Get current system status."""
        return self.status
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics."""
        return {
            'status': {
                'left_health': self.status.left_health.value,
                'right_health': self.status.right_health.value,
                'fallback_active': self.status.fallback_active,
                'fallback_strategy': self.status.fallback_strategy.value if self.status.fallback_strategy else None,
                'active_channel': self.status.active_channel,
                'uptime': self.status.uptime
            },
            'left_metrics': {
                'connection_attempts': self.left_metrics.connection_attempts,
                'successful_connections': self.left_metrics.successful_connections,
                'failed_connections': self.left_metrics.failed_connections,
                'results_received': self.left_metrics.results_received,
                'bytes_sent': self.left_metrics.bytes_sent,
                'average_latency': self.left_metrics.average_latency,
                'consecutive_failures': self.left_metrics.consecutive_failures
            },
            'right_metrics': {
                'connection_attempts': self.right_metrics.connection_attempts,
                'successful_connections': self.right_metrics.successful_connections,
                'failed_connections': self.right_metrics.failed_connections,
                'results_received': self.right_metrics.results_received,
                'bytes_sent': self.right_metrics.bytes_sent,
                'average_latency': self.right_metrics.average_latency,
                'consecutive_failures': self.right_metrics.consecutive_failures
            }
        }
    
    def set_error_callback(self, callback: Callable[[str, Exception], None]) -> None:
        """Set callback for error notifications."""
        self.error_callback = callback
    
    def set_health_change_callback(self, callback: Callable[[DualConnectionStatus], None]) -> None:
        """Set callback for health status changes."""
        self.health_change_callback = callback
    
    def set_fallback_callback(self, callback: Callable[[Optional[FallbackStrategy], str], None]) -> None:
        """Set callback for fallback mode changes."""
        self.fallback_callback = callback