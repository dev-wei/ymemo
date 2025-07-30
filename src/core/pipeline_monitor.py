"""Pipeline monitoring and observability system for audio processing pipeline."""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import statistics
import traceback

from ..analytics.session_analytics import SessionAnalytics, AnalyticsEvent

logger = logging.getLogger(__name__)

# Try to import psutil, fall back gracefully if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    logger.warning("psutil not available - resource monitoring will be limited")
    PSUTIL_AVAILABLE = False


class PipelineStage(Enum):
    """Pipeline processing stages."""
    INITIALIZATION = "initialization"
    PROVIDER_SETUP = "provider_setup"
    TRANSCRIPTION_START = "transcription_start"
    AUDIO_CAPTURE_START = "audio_capture_start"
    PROCESSING_LOOP = "processing_loop"
    TRANSCRIPTION_PROCESSING = "transcription_processing"
    AUDIO_CAPTURE_PROCESSING = "audio_capture_processing"
    SHUTDOWN = "shutdown"
    CLEANUP = "cleanup"


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class PipelineMetrics:
    """Real-time pipeline performance metrics."""
    # Throughput metrics
    audio_chunks_processed: int = 0
    transcriptions_processed: int = 0
    audio_chunks_per_second: float = 0.0
    transcriptions_per_second: float = 0.0
    
    # Latency metrics
    audio_capture_latency_ms: deque = field(default_factory=lambda: deque(maxlen=100))
    transcription_latency_ms: deque = field(default_factory=lambda: deque(maxlen=100))
    end_to_end_latency_ms: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # Resource utilization
    cpu_usage_percent: deque = field(default_factory=lambda: deque(maxlen=50))
    memory_usage_mb: deque = field(default_factory=lambda: deque(maxlen=50))
    memory_usage_percent: deque = field(default_factory=lambda: deque(maxlen=50))
    
    # Error tracking
    error_count: int = 0
    error_rate: float = 0.0
    consecutive_errors: int = 0
    last_error_time: Optional[datetime] = None
    
    # Connection health
    connection_drops: int = 0
    reconnection_attempts: int = 0
    connection_uptime_seconds: float = 0.0
    
    # Queue depths (for bottleneck detection)
    audio_queue_depth: int = 0
    transcription_queue_depth: int = 0
    
    def get_avg_audio_latency(self) -> float:
        """Get average audio capture latency."""
        return statistics.mean(self.audio_capture_latency_ms) if self.audio_capture_latency_ms else 0.0
    
    def get_avg_transcription_latency(self) -> float:
        """Get average transcription latency."""
        return statistics.mean(self.transcription_latency_ms) if self.transcription_latency_ms else 0.0
    
    def get_avg_end_to_end_latency(self) -> float:
        """Get average end-to-end latency."""
        return statistics.mean(self.end_to_end_latency_ms) if self.end_to_end_latency_ms else 0.0
    
    def get_current_cpu_usage(self) -> float:
        """Get current CPU usage."""
        return self.cpu_usage_percent[-1] if self.cpu_usage_percent else 0.0
    
    def get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.memory_usage_mb[-1] if self.memory_usage_mb else 0.0
    
    def get_current_memory_percent(self) -> float:
        """Get current memory usage percentage."""
        return self.memory_usage_percent[-1] if self.memory_usage_percent else 0.0


@dataclass
class PipelineHealth:
    """Pipeline health assessment."""
    overall_status: HealthStatus
    stage_statuses: Dict[PipelineStage, HealthStatus] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    last_assessment: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'overall_status': self.overall_status.value,
            'stage_statuses': {stage.value: status.value for stage, status in self.stage_statuses.items()},
            'issues': self.issues,
            'recommendations': self.recommendations,
            'last_assessment': self.last_assessment.isoformat(),
            'issues_count': len(self.issues)
        }


class PipelineMonitor:
    """
    Comprehensive pipeline monitoring and observability system.
    
    Provides real-time metrics collection, health assessment,
    performance monitoring, and integration with analytics.
    """
    
    def __init__(self, 
                 session_analytics: Optional[SessionAnalytics] = None,
                 metrics_retention_seconds: int = 3600,  # 1 hour
                 health_check_interval_seconds: float = 30.0):
        """
        Initialize pipeline monitor.
        
        Args:
            session_analytics: Optional analytics system integration
            metrics_retention_seconds: How long to retain metrics
            health_check_interval_seconds: Health check frequency
        """
        self.session_analytics = session_analytics
        self.metrics_retention = timedelta(seconds=metrics_retention_seconds)
        self.health_check_interval = health_check_interval_seconds
        
        # Monitoring state
        self.metrics = PipelineMetrics()
        self.current_health = PipelineHealth(overall_status=HealthStatus.HEALTHY)
        self.is_monitoring = False
        self.current_session_id: Optional[str] = None
        
        # Timing and correlation tracking
        self.stage_timings: Dict[str, datetime] = {}
        self.correlation_ids: Dict[str, Dict[str, Any]] = {}
        self._last_audio_chunk_time: Optional[float] = None
        self._last_transcription_time: Optional[float] = None
        
        # Event history for trend analysis
        self.metric_history: deque = deque(maxlen=1000)
        self.health_history: deque = deque(maxlen=100)
        
        # Performance baselines (learned over time)
        self.performance_baselines: Dict[str, float] = {
            'audio_latency_baseline_ms': 50.0,
            'transcription_latency_baseline_ms': 200.0,
            'cpu_usage_baseline_percent': 30.0,
            'memory_usage_baseline_mb': 500.0
        }
        
        # Alert thresholds
        self.alert_thresholds: Dict[str, Dict[str, float]] = {
            'latency': {
                'warning_ms': 500.0,
                'critical_ms': 1000.0
            },
            'cpu': {
                'warning_percent': 70.0,
                'critical_percent': 90.0
            },
            'memory': {
                'warning_mb': 1000.0,
                'critical_mb': 2000.0
            },
            'error_rate': {
                'warning_percent': 5.0,
                'critical_percent': 15.0
            }
        }
        
        # Monitoring threads
        self._monitoring_thread: Optional[threading.Thread] = None
        self._health_check_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Callbacks for alerts and notifications
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        logger.info("PipelineMonitor initialized with comprehensive observability")
    
    def start_monitoring(self, session_id: str) -> None:
        """Start pipeline monitoring for a session."""
        if self.is_monitoring:
            logger.warning("Pipeline monitoring already active")
            return
        
        self.current_session_id = session_id
        self.is_monitoring = True
        self._stop_monitoring.clear()
        self._monitoring_start_time = time.time()
        
        # Reset metrics for new session
        self.metrics = PipelineMetrics()
        self.stage_timings.clear()
        self.correlation_ids.clear()
        self._last_audio_chunk_time = None
        self._last_transcription_time = None
        
        # Start monitoring threads
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name=f"PipelineMonitor-{session_id}",
            daemon=True
        )
        self._monitoring_thread.start()
        
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            name=f"HealthChecker-{session_id}",
            daemon=True
        )
        self._health_check_thread.start()
        
        # Track monitoring start
        if self.session_analytics:
            self.session_analytics.track_event(
                AnalyticsEvent.SESSION_STARTED,
                session_id,
                {'monitoring_enabled': True, 'monitor_version': '1.0'}
            )
        
        logger.info(f"🔍 Pipeline monitoring started for session {session_id}")
    
    def stop_monitoring(self) -> None:
        """Stop pipeline monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self._stop_monitoring.set()
        
        # Wait for threads to complete
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=2.0)
        
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=2.0)
        
        # Generate final report
        if self.current_session_id and self.session_analytics:
            self._generate_session_monitoring_report()
        
        logger.info("🛑 Pipeline monitoring stopped")
    
    def record_stage_start(self, stage: PipelineStage, correlation_id: Optional[str] = None, **context) -> str:
        """
        Record the start of a pipeline stage.
        
        Args:
            stage: Pipeline stage being started
            correlation_id: Optional correlation ID for tracking
            **context: Additional context data
            
        Returns:
            Correlation ID for this stage execution
        """
        if correlation_id is None:
            correlation_id = f"{stage.value}_{int(time.time() * 1000)}"
        
        start_time = datetime.now()
        self.stage_timings[correlation_id] = start_time
        
        self.correlation_ids[correlation_id] = {
            'stage': stage,
            'start_time': start_time,
            'context': context
        }
        
        logger.debug(f"📊 Stage started: {stage.value} [{correlation_id}]")
        
        # Track in analytics
        if self.session_analytics and self.current_session_id:
            self.session_analytics.track_performance_metric(
                self.current_session_id,
                f"{stage.value}_start",
                time.time(),
                context
            )
        
        return correlation_id
    
    def record_stage_complete(self, correlation_id: str, success: bool = True, **result_context) -> Optional[float]:
        """
        Record the completion of a pipeline stage.
        
        Args:
            correlation_id: Correlation ID from record_stage_start
            success: Whether the stage completed successfully
            **result_context: Additional result context
            
        Returns:
            Stage duration in milliseconds, or None if correlation_id not found
        """
        if correlation_id not in self.correlation_ids:
            logger.warning(f"Unknown correlation ID: {correlation_id}")
            return None
        
        end_time = datetime.now()
        start_time = self.correlation_ids[correlation_id]['start_time']
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        stage = self.correlation_ids[correlation_id]['stage']
        
        # Update stage timing
        self.correlation_ids[correlation_id].update({
            'end_time': end_time,
            'duration_ms': duration_ms,
            'success': success,
            'result_context': result_context
        })
        
        # Update metrics based on stage
        self._update_stage_metrics(stage, duration_ms, success)
        
        logger.debug(f"📊 Stage completed: {stage.value} [{correlation_id}] - {duration_ms:.1f}ms ({'✅' if success else '❌'})")
        
        # Track in analytics
        if self.session_analytics and self.current_session_id:
            self.session_analytics.track_performance_metric(
                self.current_session_id,
                f"{stage.value}_duration",
                duration_ms,
                {'success': success, **result_context}
            )
        
        return duration_ms
    
    def record_audio_chunk_processed(self, chunk_size_bytes: int, processing_time_ms: float) -> None:
        """Record processing of an audio chunk."""
        self.metrics.audio_chunks_processed += 1
        self.metrics.audio_capture_latency_ms.append(processing_time_ms)
        
        # Update throughput calculation (simplified)
        current_time = time.time()
        if self._last_audio_chunk_time is not None:
            time_diff = current_time - self._last_audio_chunk_time
            if time_diff > 0:
                self.metrics.audio_chunks_per_second = 1.0 / time_diff
        self._last_audio_chunk_time = current_time
        
        # Track queue depth if available
        if hasattr(self, '_audio_queue_size'):
            self.metrics.audio_queue_depth = self._audio_queue_size
        
        # Log chunk processing for debugging
        logger.debug(f"📊 Audio chunk: {chunk_size_bytes} bytes, {processing_time_ms:.1f}ms")
    
    def record_transcription_processed(self, text: str, confidence: float, processing_time_ms: float, is_partial: bool = False) -> None:
        """Record processing of a transcription result."""
        self.metrics.transcriptions_processed += 1
        self.metrics.transcription_latency_ms.append(processing_time_ms)
        
        # Update throughput calculation
        current_time = time.time()
        if self._last_transcription_time is not None:
            time_diff = current_time - self._last_transcription_time
            if time_diff > 0:
                self.metrics.transcriptions_per_second = 1.0 / time_diff
        self._last_transcription_time = current_time
        
        # Track in analytics
        if self.session_analytics and self.current_session_id:
            self.session_analytics.track_transcription(
                self.current_session_id,
                text,
                confidence,
                is_partial,
                processing_time_ms
            )
    
    def record_error(self, error: Exception, stage: Optional[PipelineStage] = None, **context) -> None:
        """Record a pipeline error."""
        self.metrics.error_count += 1
        self.metrics.consecutive_errors += 1
        self.metrics.last_error_time = datetime.now()
        
        # Calculate error rate (errors per minute)
        if self.metrics.audio_chunks_processed > 0:
            self.metrics.error_rate = (self.metrics.error_count / self.metrics.audio_chunks_processed) * 100
        
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'stage': stage.value if stage else 'unknown',
            'traceback': traceback.format_exc(),
            **context
        }
        
        logger.error(f"🚨 Pipeline error recorded: {error_info}")
        
        # Track in analytics
        if self.session_analytics and self.current_session_id:
            self.session_analytics.track_event(
                AnalyticsEvent.CONNECTION_ERROR,  # Generic error event
                self.current_session_id,
                error_info
            )
        
        # Trigger alerts if error rate is high
        self._check_error_rate_alerts()
    
    def record_success_operation(self) -> None:
        """Record a successful operation (resets consecutive error count)."""
        self.metrics.consecutive_errors = 0
    
    def update_queue_depths(self, audio_queue_depth: int, transcription_queue_depth: int) -> None:
        """Update queue depth metrics for bottleneck detection."""
        self.metrics.audio_queue_depth = audio_queue_depth
        self.metrics.transcription_queue_depth = transcription_queue_depth
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current pipeline metrics."""
        return {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.current_session_id,
            'throughput': {
                'audio_chunks_processed': self.metrics.audio_chunks_processed,
                'transcriptions_processed': self.metrics.transcriptions_processed,
                'audio_chunks_per_second': self.metrics.audio_chunks_per_second,
                'transcriptions_per_second': self.metrics.transcriptions_per_second
            },
            'latency': {
                'avg_audio_capture_ms': self.metrics.get_avg_audio_latency(),
                'avg_transcription_ms': self.metrics.get_avg_transcription_latency(),
                'avg_end_to_end_ms': self.metrics.get_avg_end_to_end_latency()
            },
            'resources': {
                'current_cpu_percent': self.metrics.get_current_cpu_usage(),
                'current_memory_mb': self.metrics.get_current_memory_usage(),
                'current_memory_percent': self.metrics.get_current_memory_percent()
            },
            'errors': {
                'total_errors': self.metrics.error_count,
                'error_rate_percent': self.metrics.error_rate,
                'consecutive_errors': self.metrics.consecutive_errors,
                'last_error_time': self.metrics.last_error_time.isoformat() if self.metrics.last_error_time else None
            },
            'queues': {
                'audio_queue_depth': self.metrics.audio_queue_depth,
                'transcription_queue_depth': self.metrics.transcription_queue_depth
            },
            'connection': {
                'drops': self.metrics.connection_drops,
                'reconnection_attempts': self.metrics.reconnection_attempts,
                'uptime_seconds': self.metrics.connection_uptime_seconds
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current pipeline health status."""
        return self.current_health.to_dict()
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add callback for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop that collects system metrics."""
        logger.info("📊 Pipeline monitoring loop started")
        
        while not self._stop_monitoring.is_set():
            try:
                # Collect system metrics (if psutil available)
                if PSUTIL_AVAILABLE:
                    process = psutil.Process()
                    
                    # CPU usage
                    cpu_percent = process.cpu_percent()
                    self.metrics.cpu_usage_percent.append(cpu_percent)
                    
                    # Memory usage
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
                    memory_percent = process.memory_percent()
                    
                    self.metrics.memory_usage_mb.append(memory_mb)
                    self.metrics.memory_usage_percent.append(memory_percent)
                else:
                    # Fallback values when psutil not available
                    cpu_percent = 0.0
                    memory_mb = 0.0
                    memory_percent = 0.0
                
                # Update connection uptime
                self.metrics.connection_uptime_seconds += 1.0
                
                # Store metric snapshot
                metric_snapshot = {
                    'timestamp': datetime.now(),
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'memory_percent': memory_percent,
                    'audio_queue_depth': self.metrics.audio_queue_depth,
                    'transcription_queue_depth': self.metrics.transcription_queue_depth
                }
                self.metric_history.append(metric_snapshot)
                
                # Check for performance alerts (only if we have real metrics)
                if PSUTIL_AVAILABLE:
                    self._check_performance_alerts(cpu_percent, memory_mb)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Wait before next collection
            if not self._stop_monitoring.wait(timeout=1.0):
                continue
            else:
                break
        
        logger.info("📊 Pipeline monitoring loop stopped")
    
    def _health_check_loop(self) -> None:
        """Health check loop that assesses pipeline health."""
        logger.info("🏥 Pipeline health check loop started")
        
        while not self._stop_monitoring.is_set():
            try:
                # Perform comprehensive health assessment
                self._assess_pipeline_health()
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
            
            # Wait before next health check
            if not self._stop_monitoring.wait(timeout=self.health_check_interval):
                continue
            else:
                break
        
        logger.info("🏥 Pipeline health check loop stopped")
    
    def _assess_pipeline_health(self) -> None:
        """Assess overall pipeline health."""
        health = PipelineHealth(overall_status=HealthStatus.HEALTHY)
        
        # Check error rate
        if self.metrics.error_rate > self.alert_thresholds['error_rate']['critical_percent']:
            health.overall_status = HealthStatus.CRITICAL
            health.issues.append(f"Critical error rate: {self.metrics.error_rate:.1f}%")
            health.recommendations.append("Investigate error patterns and root causes")
        elif self.metrics.error_rate > self.alert_thresholds['error_rate']['warning_percent']:
            health.overall_status = HealthStatus.DEGRADED
            health.issues.append(f"Elevated error rate: {self.metrics.error_rate:.1f}%")
            health.recommendations.append("Monitor error trends closely")
        
        # Check consecutive errors
        if self.metrics.consecutive_errors > 10:
            health.overall_status = HealthStatus.CRITICAL
            health.issues.append(f"High consecutive errors: {self.metrics.consecutive_errors}")
            health.recommendations.append("Check provider connectivity and configuration")
        elif self.metrics.consecutive_errors > 5:
            if health.overall_status == HealthStatus.HEALTHY:
                health.overall_status = HealthStatus.DEGRADED
            health.issues.append(f"Multiple consecutive errors: {self.metrics.consecutive_errors}")
        
        # Check resource usage
        current_cpu = self.metrics.get_current_cpu_usage()
        current_memory = self.metrics.get_current_memory_usage()
        
        if current_cpu > self.alert_thresholds['cpu']['critical_percent']:
            health.overall_status = HealthStatus.CRITICAL
            health.issues.append(f"Critical CPU usage: {current_cpu:.1f}%")
            health.recommendations.append("Reduce processing load or optimize performance")
        elif current_cpu > self.alert_thresholds['cpu']['warning_percent']:
            if health.overall_status == HealthStatus.HEALTHY:
                health.overall_status = HealthStatus.DEGRADED
            health.issues.append(f"High CPU usage: {current_cpu:.1f}%")
            health.recommendations.append("Monitor CPU trends")
        
        if current_memory > self.alert_thresholds['memory']['critical_mb']:
            health.overall_status = HealthStatus.CRITICAL
            health.issues.append(f"Critical memory usage: {current_memory:.1f}MB")
            health.recommendations.append("Check for memory leaks or increase available memory")
        elif current_memory > self.alert_thresholds['memory']['warning_mb']:
            if health.overall_status == HealthStatus.HEALTHY:
                health.overall_status = HealthStatus.DEGRADED
            health.issues.append(f"High memory usage: {current_memory:.1f}MB")
        
        # Check latency
        avg_transcription_latency = self.metrics.get_avg_transcription_latency()
        if avg_transcription_latency > self.alert_thresholds['latency']['critical_ms']:
            health.overall_status = HealthStatus.CRITICAL
            health.issues.append(f"Critical transcription latency: {avg_transcription_latency:.1f}ms")
            health.recommendations.append("Check network connectivity and provider performance")
        elif avg_transcription_latency > self.alert_thresholds['latency']['warning_ms']:
            if health.overall_status == HealthStatus.HEALTHY:
                health.overall_status = HealthStatus.DEGRADED
            health.issues.append(f"High transcription latency: {avg_transcription_latency:.1f}ms")
        
        # Check queue depths for bottlenecks
        if self.metrics.audio_queue_depth > 100:
            health.issues.append(f"Audio queue backlog: {self.metrics.audio_queue_depth} items")
            health.recommendations.append("Check audio processing performance")
        
        if self.metrics.transcription_queue_depth > 50:
            health.issues.append(f"Transcription queue backlog: {self.metrics.transcription_queue_depth} items")
            health.recommendations.append("Check transcription service performance")
        
        # Store health assessment
        health.last_assessment = datetime.now()
        self.current_health = health
        self.health_history.append(health)
        
        # Log health changes
        if len(self.health_history) > 1:
            previous_health = self.health_history[-2]
            if previous_health.overall_status != health.overall_status:
                logger.info(f"🏥 Pipeline health changed: {previous_health.overall_status.value} → {health.overall_status.value}")
    
    def _update_stage_metrics(self, stage: PipelineStage, duration_ms: float, success: bool) -> None:
        """Update metrics based on completed stage."""
        if stage in [PipelineStage.AUDIO_CAPTURE_PROCESSING]:
            if success:
                self.record_success_operation()
        elif stage in [PipelineStage.TRANSCRIPTION_PROCESSING]:
            if success:
                self.record_success_operation()
        
        # Log stage metrics for debugging
        logger.debug(f"📊 Stage: {stage.value} - {duration_ms:.1f}ms ({'✅' if success else '❌'})")
    
    def _check_performance_alerts(self, cpu_percent: float, memory_mb: float) -> None:
        """Check for performance-related alerts."""
        # CPU alerts
        if cpu_percent > self.alert_thresholds['cpu']['critical_percent']:
            self._trigger_alert('cpu_critical', {
                'cpu_percent': cpu_percent,
                'threshold': self.alert_thresholds['cpu']['critical_percent']
            })
        elif cpu_percent > self.alert_thresholds['cpu']['warning_percent']:
            self._trigger_alert('cpu_warning', {
                'cpu_percent': cpu_percent,
                'threshold': self.alert_thresholds['cpu']['warning_percent']
            })
        
        # Memory alerts
        if memory_mb > self.alert_thresholds['memory']['critical_mb']:
            self._trigger_alert('memory_critical', {
                'memory_mb': memory_mb,
                'threshold': self.alert_thresholds['memory']['critical_mb']
            })
        elif memory_mb > self.alert_thresholds['memory']['warning_mb']:
            self._trigger_alert('memory_warning', {
                'memory_mb': memory_mb,
                'threshold': self.alert_thresholds['memory']['warning_mb']
            })
    
    def _check_error_rate_alerts(self) -> None:
        """Check for error rate alerts."""
        if self.metrics.error_rate > self.alert_thresholds['error_rate']['critical_percent']:
            self._trigger_alert('error_rate_critical', {
                'error_rate_percent': self.metrics.error_rate,
                'threshold': self.alert_thresholds['error_rate']['critical_percent']
            })
        elif self.metrics.error_rate > self.alert_thresholds['error_rate']['warning_percent']:
            self._trigger_alert('error_rate_warning', {
                'error_rate_percent': self.metrics.error_rate,
                'threshold': self.alert_thresholds['error_rate']['warning_percent']
            })
    
    def _trigger_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        """Trigger an alert notification."""
        alert_data = {
            'alert_type': alert_type,
            'timestamp': datetime.now().isoformat(),
            'session_id': self.current_session_id,
            **data
        }
        
        logger.warning(f"🚨 Pipeline alert: {alert_type} - {data}")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def _generate_session_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report for the session."""
        current_time = time.time()
        start_time = getattr(self, '_monitoring_start_time', current_time)
        
        report = {
            'session_id': self.current_session_id,
            'monitoring_duration': current_time - start_time,
            'final_metrics': self.get_current_metrics(),
            'final_health': self.get_health_status(),
            'performance_summary': {
                'total_audio_chunks': self.metrics.audio_chunks_processed,
                'total_transcriptions': self.metrics.transcriptions_processed,
                'total_errors': self.metrics.error_count,
                'avg_cpu_usage': statistics.mean(self.metrics.cpu_usage_percent) if self.metrics.cpu_usage_percent else 0,
                'avg_memory_usage': statistics.mean(self.metrics.memory_usage_mb) if self.metrics.memory_usage_mb else 0,
                'peak_memory_usage': max(self.metrics.memory_usage_mb) if self.metrics.memory_usage_mb else 0
            }
        }
        
        if self.session_analytics:
            self.session_analytics.track_event(
                AnalyticsEvent.SESSION_ENDED,
                self.current_session_id,
                {'monitoring_report': report}
            )
        
        logger.info(f"📊 Generated monitoring report for session {self.current_session_id}")
        return report