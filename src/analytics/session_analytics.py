"""Session analytics and metrics collection system."""

import json
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class AnalyticsEvent(Enum):
    """Types of analytics events."""
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    RECORDING_STARTED = "recording_started"
    RECORDING_STOPPED = "recording_stopped"
    TRANSCRIPTION_RECEIVED = "transcription_received"
    PARTIAL_TRANSCRIPTION = "partial_transcription"
    FINAL_TRANSCRIPTION = "final_transcription"
    CONNECTION_ERROR = "connection_error"
    CONNECTION_RESTORED = "connection_restored"
    USER_ACTION = "user_action"
    PERFORMANCE_METRIC = "performance_metric"


@dataclass
class AnalyticsEventData:
    """Analytics event data structure."""
    event_type: AnalyticsEvent
    timestamp: datetime
    session_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "data": self.data
        }


@dataclass
class SessionMetrics:
    """Comprehensive session metrics."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration: float = 0.0
    recording_duration: float = 0.0
    transcription_count: int = 0
    partial_transcription_count: int = 0
    final_transcription_count: int = 0
    word_count: int = 0
    character_count: int = 0
    average_confidence: float = 0.0
    connection_errors: int = 0
    connection_downtime: float = 0.0
    device_switches: int = 0
    user_actions: int = 0
    performance_issues: int = 0
    recording_segments: List[Dict[str, Any]] = field(default_factory=list)
    transcription_quality_scores: List[float] = field(default_factory=list)
    response_times: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data


@dataclass
class PerformanceMetrics:
    """Performance-related metrics."""
    transcription_latency: deque = field(default_factory=lambda: deque(maxlen=100))
    memory_usage: deque = field(default_factory=lambda: deque(maxlen=100))
    cpu_usage: deque = field(default_factory=lambda: deque(maxlen=100))
    network_latency: deque = field(default_factory=lambda: deque(maxlen=100))
    error_rates: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def get_average_latency(self) -> float:
        """Get average transcription latency."""
        return statistics.mean(self.transcription_latency) if self.transcription_latency else 0.0
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage."""
        return max(self.memory_usage) if self.memory_usage else 0.0
    
    def get_error_rate(self) -> float:
        """Get current error rate."""
        return statistics.mean(self.error_rates) if self.error_rates else 0.0


class SessionAnalytics:
    """Comprehensive session analytics system."""
    
    def __init__(self, storage_directory: Optional[Path] = None, max_events: int = 1000):
        self.max_events = max_events
        
        # Storage
        self.storage_directory = storage_directory or Path.home() / '.ymemo' / 'analytics'
        self.storage_directory.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Current session tracking
        self._current_session_id: Optional[str] = None
        self._current_session_metrics: Optional[SessionMetrics] = None
        self._session_start_time: Optional[datetime] = None
        
        # Event storage
        self._events: deque = deque(maxlen=max_events)
        self._session_metrics_history: Dict[str, SessionMetrics] = {}
        
        # Performance tracking
        self._performance_metrics = PerformanceMetrics()
        
        # Aggregated analytics
        self._daily_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._weekly_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._monthly_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Callbacks
        self._analytics_callbacks: List[Callable[[AnalyticsEventData], None]] = []
        
        logger.info(f"SessionAnalytics initialized with storage: {self.storage_directory}")
    
    def track_event(self, event_type: AnalyticsEvent, session_id: str, data: Optional[Dict[str, Any]] = None):
        """Track an analytics event."""
        with self._lock:
            event_data = AnalyticsEventData(
                event_type=event_type,
                timestamp=datetime.now(),
                session_id=session_id,
                data=data or {}
            )
            
            self._events.append(event_data)
            
            # Update current session metrics
            if session_id == self._current_session_id and self._current_session_metrics:
                self._update_session_metrics(event_data)
            
            # Update aggregated stats
            self._update_aggregated_stats(event_data)
            
            # Notify callbacks
            for callback in self._analytics_callbacks:
                try:
                    callback(event_data)
                except Exception as e:
                    logger.error(f"Error in analytics callback: {e}")
            
            logger.debug(f"Tracked event: {event_type.value} for session {session_id}")
    
    def start_session(self, session_id: str) -> SessionMetrics:
        """Start tracking a new session."""
        with self._lock:
            # End previous session if exists
            if self._current_session_id and self._current_session_metrics:
                self.end_session(self._current_session_id)
            
            # Start new session
            self._current_session_id = session_id
            self._session_start_time = datetime.now()
            self._current_session_metrics = SessionMetrics(
                session_id=session_id,
                start_time=self._session_start_time
            )
            
            # Track session start event
            self.track_event(AnalyticsEvent.SESSION_STARTED, session_id, {
                'start_time': self._session_start_time.isoformat()
            })
            
            logger.info(f"Started analytics tracking for session {session_id}")
            return self._current_session_metrics
    
    def end_session(self, session_id: str) -> Optional[SessionMetrics]:
        """End session tracking and finalize metrics."""
        with self._lock:
            if session_id != self._current_session_id or not self._current_session_metrics:
                logger.warning(f"Attempt to end non-current session {session_id}")
                return None
            
            # Finalize session metrics
            end_time = datetime.now()
            self._current_session_metrics.end_time = end_time
            self._current_session_metrics.total_duration = (
                end_time - self._current_session_metrics.start_time
            ).total_seconds()
            
            # Calculate final statistics
            self._calculate_final_session_stats()
            
            # Store completed session metrics
            completed_metrics = self._current_session_metrics
            self._session_metrics_history[session_id] = completed_metrics
            
            # Track session end event
            self.track_event(AnalyticsEvent.SESSION_ENDED, session_id, {
                'end_time': end_time.isoformat(),
                'total_duration': completed_metrics.total_duration,
                'transcription_count': completed_metrics.transcription_count,
                'word_count': completed_metrics.word_count
            })
            
            # Save session metrics to file
            self._save_session_metrics(completed_metrics)
            
            # Clear current session
            self._current_session_id = None
            self._current_session_metrics = None
            
            logger.info(f"Ended analytics tracking for session {session_id}")
            return completed_metrics
    
    def _update_session_metrics(self, event_data: AnalyticsEventData):
        """Update current session metrics based on event."""
        if not self._current_session_metrics:
            return
        
        metrics = self._current_session_metrics
        event_type = event_data.event_type
        data = event_data.data
        
        if event_type == AnalyticsEvent.TRANSCRIPTION_RECEIVED:
            metrics.transcription_count += 1
            
            # Update word and character counts
            text = data.get('text', '')
            metrics.word_count += len(text.split())
            metrics.character_count += len(text)
            
            # Update confidence tracking
            confidence = data.get('confidence', 0.0)
            if confidence > 0:
                metrics.transcription_quality_scores.append(confidence)
                # Recalculate average
                metrics.average_confidence = statistics.mean(metrics.transcription_quality_scores)
        
        elif event_type == AnalyticsEvent.PARTIAL_TRANSCRIPTION:
            metrics.partial_transcription_count += 1
            
        elif event_type == AnalyticsEvent.FINAL_TRANSCRIPTION:
            metrics.final_transcription_count += 1
            
        elif event_type == AnalyticsEvent.CONNECTION_ERROR:
            metrics.connection_errors += 1
            
        elif event_type == AnalyticsEvent.USER_ACTION:
            metrics.user_actions += 1
            
        elif event_type == AnalyticsEvent.RECORDING_STARTED:
            segment_data = {
                'start_time': event_data.timestamp.isoformat(),
                'device': data.get('device'),
                'config': data.get('config')
            }
            metrics.recording_segments.append(segment_data)
            
        elif event_type == AnalyticsEvent.RECORDING_STOPPED:
            # Update the last recording segment
            if metrics.recording_segments:
                last_segment = metrics.recording_segments[-1]
                if 'end_time' not in last_segment:
                    last_segment['end_time'] = event_data.timestamp.isoformat()
                    
                    # Calculate segment duration
                    start_time = datetime.fromisoformat(last_segment['start_time'])
                    segment_duration = (event_data.timestamp - start_time).total_seconds()
                    last_segment['duration'] = segment_duration
                    metrics.recording_duration += segment_duration
        
        elif event_type == AnalyticsEvent.PERFORMANCE_METRIC:
            metrics.performance_issues += 1
            
            # Track response time if provided
            response_time = data.get('response_time')
            if response_time:
                metrics.response_times.append(response_time)
    
    def _calculate_final_session_stats(self):
        """Calculate final statistics for the session."""
        if not self._current_session_metrics:
            return
        
        metrics = self._current_session_metrics
        
        # Calculate recording efficiency
        if metrics.total_duration > 0:
            recording_ratio = metrics.recording_duration / metrics.total_duration
            metrics.data = metrics.__dict__.get('data', {})
            metrics.data['recording_efficiency'] = recording_ratio
        
        # Calculate transcription rate (words per minute)
        if metrics.recording_duration > 0:
            wpm = (metrics.word_count / metrics.recording_duration) * 60
            metrics.data['words_per_minute'] = wpm
        
        # Calculate error rate
        if metrics.transcription_count > 0:
            error_rate = metrics.connection_errors / metrics.transcription_count
            metrics.data['error_rate'] = error_rate
    
    def _update_aggregated_stats(self, event_data: AnalyticsEventData):
        """Update daily, weekly, and monthly aggregated statistics."""
        date_key = event_data.timestamp.date().isoformat()
        week_key = event_data.timestamp.strftime("%Y-W%U")
        month_key = event_data.timestamp.strftime("%Y-%m")
        
        # Update daily stats
        if date_key not in self._daily_stats:
            self._daily_stats[date_key] = defaultdict(int)
        self._daily_stats[date_key][event_data.event_type.value] += 1
        
        # Update weekly stats
        if week_key not in self._weekly_stats:
            self._weekly_stats[week_key] = defaultdict(int)
        self._weekly_stats[week_key][event_data.event_type.value] += 1
        
        # Update monthly stats
        if month_key not in self._monthly_stats:
            self._monthly_stats[month_key] = defaultdict(int)
        self._monthly_stats[month_key][event_data.event_type.value] += 1
    
    def track_performance_metric(self, session_id: str, metric_type: str, value: float, context: Optional[Dict] = None):
        """Track a performance-related metric."""
        with self._lock:
            # Store in performance metrics
            if metric_type == 'transcription_latency':
                self._performance_metrics.transcription_latency.append(value)
            elif metric_type == 'memory_usage':
                self._performance_metrics.memory_usage.append(value)
            elif metric_type == 'cpu_usage':
                self._performance_metrics.cpu_usage.append(value)
            elif metric_type == 'network_latency':
                self._performance_metrics.network_latency.append(value)
            elif metric_type == 'error_rate':
                self._performance_metrics.error_rates.append(value)
            
            # Track as event
            self.track_event(AnalyticsEvent.PERFORMANCE_METRIC, session_id, {
                'metric_type': metric_type,
                'value': value,
                'context': context or {}
            })
    
    def track_user_action(self, session_id: str, action: str, context: Optional[Dict] = None):
        """Track a user action."""
        self.track_event(AnalyticsEvent.USER_ACTION, session_id, {
            'action': action,
            'context': context or {}
        })
    
    def track_transcription(self, session_id: str, text: str, confidence: float, 
                          is_partial: bool, latency: Optional[float] = None):
        """Track a transcription event."""
        event_type = AnalyticsEvent.PARTIAL_TRANSCRIPTION if is_partial else AnalyticsEvent.FINAL_TRANSCRIPTION
        
        data = {
            'text': text,
            'confidence': confidence,
            'is_partial': is_partial,
            'word_count': len(text.split()),
            'character_count': len(text)
        }
        
        if latency is not None:
            data['latency'] = latency
            self.track_performance_metric(session_id, 'transcription_latency', latency)
        
        self.track_event(AnalyticsEvent.TRANSCRIPTION_RECEIVED, session_id, data)
        self.track_event(event_type, session_id, data)
    
    def get_current_session_metrics(self) -> Optional[SessionMetrics]:
        """Get metrics for the current session."""
        with self._lock:
            return self._current_session_metrics
    
    def get_session_metrics(self, session_id: str) -> Optional[SessionMetrics]:
        """Get metrics for a specific session."""
        with self._lock:
            return self._session_metrics_history.get(session_id)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance metrics summary."""
        with self._lock:
            return {
                'average_transcription_latency': self._performance_metrics.get_average_latency(),
                'peak_memory_usage': self._performance_metrics.get_peak_memory(),
                'current_error_rate': self._performance_metrics.get_error_rate(),
                'recent_events_count': len(self._events),
                'active_sessions': 1 if self._current_session_id else 0
            }
    
    def get_usage_statistics(self, period: str = 'daily') -> Dict[str, Any]:
        """Get usage statistics for specified period."""
        with self._lock:
            if period == 'daily':
                stats = dict(self._daily_stats)
            elif period == 'weekly':
                stats = dict(self._weekly_stats)
            elif period == 'monthly':
                stats = dict(self._monthly_stats)
            else:
                raise ValueError(f"Invalid period: {period}")
            
            return {
                'period': period,
                'statistics': stats,
                'total_periods': len(stats)
            }
    
    def get_recent_events(self, limit: int = 100, event_type: Optional[AnalyticsEvent] = None) -> List[Dict[str, Any]]:
        """Get recent analytics events."""
        with self._lock:
            events = list(self._events)
            
            # Filter by event type if specified
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            # Apply limit
            events = events[-limit:] if limit else events
            
            return [event.to_dict() for event in events]
    
    def generate_session_report(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Generate comprehensive report for a session."""
        metrics = self.get_session_metrics(session_id)
        if not metrics:
            return None
        
        # Get events for this session
        session_events = [e for e in self._events if e.session_id == session_id]
        
        # Calculate insights
        insights = []
        
        if metrics.average_confidence < 0.8:
            insights.append("Low average transcription confidence detected")
        
        if metrics.connection_errors > 5:
            insights.append("Multiple connection issues during session")
        
        if metrics.recording_duration < metrics.total_duration * 0.5:
            insights.append("Low recording efficiency - consider longer recording sessions")
        
        return {
            'session_id': session_id,
            'metrics': metrics.to_dict(),
            'events_count': len(session_events),
            'insights': insights,
            'performance': {
                'words_per_minute': metrics.__dict__.get('data', {}).get('words_per_minute', 0),
                'recording_efficiency': metrics.__dict__.get('data', {}).get('recording_efficiency', 0),
                'error_rate': metrics.__dict__.get('data', {}).get('error_rate', 0)
            }
        }
    
    def _save_session_metrics(self, metrics: SessionMetrics):
        """Save session metrics to file."""
        try:
            metrics_file = self.storage_directory / f"session_metrics_{metrics.session_id}.json"
            
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved session metrics to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Failed to save session metrics: {e}")
    
    def export_analytics_data(self, start_date: Optional[datetime] = None, 
                            end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Export analytics data for specified date range."""
        with self._lock:
            # Filter events by date range
            events = list(self._events)
            
            if start_date:
                events = [e for e in events if e.timestamp >= start_date]
            if end_date:
                events = [e for e in events if e.timestamp <= end_date]
            
            # Get session metrics for the period
            relevant_sessions = {}
            for session_id, metrics in self._session_metrics_history.items():
                if start_date and metrics.start_time < start_date:
                    continue
                if end_date and metrics.start_time > end_date:
                    continue
                relevant_sessions[session_id] = metrics.to_dict()
            
            return {
                'export_timestamp': datetime.now().isoformat(),
                'date_range': {
                    'start': start_date.isoformat() if start_date else None,
                    'end': end_date.isoformat() if end_date else None
                },
                'events': [e.to_dict() for e in events],
                'session_metrics': relevant_sessions,
                'performance_summary': self.get_performance_summary(),
                'usage_statistics': {
                    'daily': self.get_usage_statistics('daily'),
                    'weekly': self.get_usage_statistics('weekly'),
                    'monthly': self.get_usage_statistics('monthly')
                }
            }
    
    def add_analytics_callback(self, callback: Callable[[AnalyticsEventData], None]):
        """Add callback for analytics events."""
        with self._lock:
            self._analytics_callbacks.append(callback)
    
    def remove_analytics_callback(self, callback: Callable[[AnalyticsEventData], None]):
        """Remove analytics callback."""
        with self._lock:
            if callback in self._analytics_callbacks:
                self._analytics_callbacks.remove(callback)
    
    def cleanup(self):
        """Clean up analytics resources."""
        with self._lock:
            # End current session if active
            if self._current_session_id:
                self.end_session(self._current_session_id)
            
            # Clear callbacks
            self._analytics_callbacks.clear()
            
            logger.info("SessionAnalytics cleanup completed")