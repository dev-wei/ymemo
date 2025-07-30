#!/usr/bin/env python3
"""
Tests for pipeline monitoring and observability system.

These tests verify that pipeline monitoring correctly tracks metrics,
assesses health, and integrates with analytics.
"""

import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime, timedelta
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock psutil before importing
sys.modules['psutil'] = MagicMock()

from src.core.pipeline_monitor import (
    PipelineMonitor, PipelineStage, HealthStatus, PipelineMetrics, PipelineHealth
)
from src.analytics.session_analytics import SessionAnalytics, AnalyticsEvent


class TestPipelineMetrics(unittest.TestCase):
    """Test PipelineMetrics class."""
    
    def test_metrics_initialization(self):
        """Test pipeline metrics initialization."""
        metrics = PipelineMetrics()
        
        self.assertEqual(metrics.audio_chunks_processed, 0)
        self.assertEqual(metrics.transcriptions_processed, 0)
        self.assertEqual(metrics.audio_chunks_per_second, 0.0)
        self.assertEqual(metrics.error_count, 0)
        self.assertEqual(metrics.consecutive_errors, 0)
        self.assertIsNone(metrics.last_error_time)
    
    def test_latency_calculations(self):
        """Test latency calculation methods."""
        metrics = PipelineMetrics()
        
        # Add some latency samples
        metrics.audio_capture_latency_ms.extend([10.0, 20.0, 30.0])
        metrics.transcription_latency_ms.extend([100.0, 200.0, 300.0])
        metrics.end_to_end_latency_ms.extend([500.0, 600.0, 700.0])
        
        self.assertEqual(metrics.get_avg_audio_latency(), 20.0)
        self.assertEqual(metrics.get_avg_transcription_latency(), 200.0)
        self.assertEqual(metrics.get_avg_end_to_end_latency(), 600.0)
    
    def test_resource_usage_tracking(self):
        """Test resource usage tracking."""
        metrics = PipelineMetrics()
        
        # Add resource usage samples
        metrics.cpu_usage_percent.extend([10.0, 20.0, 30.0])
        metrics.memory_usage_mb.extend([100.0, 200.0, 300.0])
        metrics.memory_usage_percent.extend([25.0, 50.0, 75.0])
        
        self.assertEqual(metrics.get_current_cpu_usage(), 30.0)
        self.assertEqual(metrics.get_current_memory_usage(), 300.0)
        self.assertEqual(metrics.get_current_memory_percent(), 75.0)
    
    def test_empty_metrics(self):
        """Test behavior with empty metrics."""
        metrics = PipelineMetrics()
        
        self.assertEqual(metrics.get_avg_audio_latency(), 0.0)
        self.assertEqual(metrics.get_avg_transcription_latency(), 0.0)
        self.assertEqual(metrics.get_avg_end_to_end_latency(), 0.0)
        self.assertEqual(metrics.get_current_cpu_usage(), 0.0)
        self.assertEqual(metrics.get_current_memory_usage(), 0.0)


class TestPipelineHealth(unittest.TestCase):
    """Test PipelineHealth class."""
    
    def test_health_initialization(self):
        """Test pipeline health initialization."""
        health = PipelineHealth(overall_status=HealthStatus.HEALTHY)
        
        self.assertEqual(health.overall_status, HealthStatus.HEALTHY)
        self.assertEqual(len(health.issues), 0)
        self.assertEqual(len(health.recommendations), 0)
        self.assertEqual(len(health.stage_statuses), 0)
    
    def test_health_to_dict(self):
        """Test health status serialization."""
        health = PipelineHealth(
            overall_status=HealthStatus.DEGRADED,
            stage_statuses={PipelineStage.INITIALIZATION: HealthStatus.HEALTHY},
            issues=["High latency detected"],
            recommendations=["Check network connectivity"]
        )
        
        health_dict = health.to_dict()
        
        self.assertEqual(health_dict['overall_status'], 'degraded')
        self.assertEqual(health_dict['stage_statuses']['initialization'], 'healthy')
        self.assertIn("High latency detected", health_dict['issues'])
        self.assertIn("Check network connectivity", health_dict['recommendations'])
        self.assertEqual(health_dict['issues_count'], 1)


class TestPipelineMonitor(unittest.TestCase):
    """Test PipelineMonitor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_analytics = Mock(spec=SessionAnalytics)
        self.monitor = PipelineMonitor(
            session_analytics=self.mock_analytics,
            health_check_interval_seconds=0.1  # Fast for testing
        )
    
    def test_monitor_initialization(self):
        """Test pipeline monitor initialization."""
        self.assertEqual(self.monitor.session_analytics, self.mock_analytics)
        self.assertFalse(self.monitor.is_monitoring)
        self.assertIsNone(self.monitor.current_session_id)
        self.assertEqual(self.monitor.current_health.overall_status, HealthStatus.HEALTHY)
    
    def test_stage_recording(self):
        """Test pipeline stage recording."""
        # Record stage start
        correlation_id = self.monitor.record_stage_start(
            PipelineStage.INITIALIZATION,
            test_param="test_value"
        )
        
        self.assertIsNotNone(correlation_id)
        self.assertIn(correlation_id, self.monitor.correlation_ids)
        
        # Check stored stage info
        stage_info = self.monitor.correlation_ids[correlation_id]
        self.assertEqual(stage_info['stage'], PipelineStage.INITIALIZATION)
        self.assertIn('start_time', stage_info)
        self.assertEqual(stage_info['context']['test_param'], "test_value")
        
        # Record stage completion
        duration = self.monitor.record_stage_complete(correlation_id, success=True)
        
        self.assertIsNotNone(duration)
        self.assertGreater(duration, 0)
        
        # Check completion info
        self.assertTrue(stage_info['success'])
        self.assertIn('end_time', stage_info)
        self.assertIn('duration_ms', stage_info)
    
    def test_stage_recording_with_failure(self):
        """Test pipeline stage recording with failure."""
        correlation_id = self.monitor.record_stage_start(PipelineStage.PROVIDER_SETUP)
        
        # Record stage failure
        duration = self.monitor.record_stage_complete(
            correlation_id, 
            success=False, 
            error="Provider initialization failed"
        )
        
        self.assertIsNotNone(duration)
        
        stage_info = self.monitor.correlation_ids[correlation_id]
        self.assertFalse(stage_info['success'])
        self.assertEqual(stage_info['result_context']['error'], "Provider initialization failed")
    
    def test_unknown_correlation_id(self):
        """Test handling of unknown correlation ID."""
        duration = self.monitor.record_stage_complete("unknown_id", success=True)
        self.assertIsNone(duration)
    
    def test_audio_chunk_processing(self):
        """Test audio chunk processing recording."""
        initial_count = self.monitor.metrics.audio_chunks_processed
        
        self.monitor.record_audio_chunk_processed(1024, 50.0)
        
        self.assertEqual(self.monitor.metrics.audio_chunks_processed, initial_count + 1)
        self.assertIn(50.0, self.monitor.metrics.audio_capture_latency_ms)
        self.assertEqual(self.monitor.metrics.get_avg_audio_latency(), 50.0)
    
    def test_transcription_processing(self):
        """Test transcription processing recording."""
        initial_count = self.monitor.metrics.transcriptions_processed
        session_id = "test_session"
        self.monitor.current_session_id = session_id
        
        self.monitor.record_transcription_processed(
            "Hello world", 0.95, 200.0, is_partial=False
        )
        
        self.assertEqual(self.monitor.metrics.transcriptions_processed, initial_count + 1)
        self.assertIn(200.0, self.monitor.metrics.transcription_latency_ms)
        
        # Verify analytics integration
        self.mock_analytics.track_transcription.assert_called_once_with(
            session_id,
            "Hello world",
            0.95,
            False,
            200.0
        )
    
    def test_error_recording(self):
        """Test error recording."""
        test_error = RuntimeError("Test error")
        initial_error_count = self.monitor.metrics.error_count
        
        self.monitor.record_error(test_error, PipelineStage.TRANSCRIPTION_PROCESSING)
        
        self.assertEqual(self.monitor.metrics.error_count, initial_error_count + 1)
        self.assertEqual(self.monitor.metrics.consecutive_errors, 1)
        self.assertIsNotNone(self.monitor.metrics.last_error_time)
    
    def test_success_operation_resets_consecutive_errors(self):
        """Test that successful operations reset consecutive error count."""
        # Record some errors
        self.monitor.record_error(RuntimeError("Error 1"))
        self.monitor.record_error(RuntimeError("Error 2"))
        
        self.assertEqual(self.monitor.metrics.consecutive_errors, 2)
        
        # Record success
        self.monitor.record_success_operation()
        
        self.assertEqual(self.monitor.metrics.consecutive_errors, 0)
    
    def test_queue_depth_updates(self):
        """Test queue depth updates."""
        self.monitor.update_queue_depths(10, 5)
        
        self.assertEqual(self.monitor.metrics.audio_queue_depth, 10)
        self.assertEqual(self.monitor.metrics.transcription_queue_depth, 5)
    
    def test_current_metrics_retrieval(self):
        """Test current metrics retrieval."""
        # Add some test data
        self.monitor.metrics.audio_chunks_processed = 100
        self.monitor.metrics.transcriptions_processed = 50
        self.monitor.metrics.error_count = 2
        
        metrics = self.monitor.get_current_metrics()
        
        self.assertIn('timestamp', metrics)
        self.assertEqual(metrics['throughput']['audio_chunks_processed'], 100)
        self.assertEqual(metrics['throughput']['transcriptions_processed'], 50)
        self.assertEqual(metrics['errors']['total_errors'], 2)
    
    def test_health_status_retrieval(self):
        """Test health status retrieval."""
        # Set some health issues
        self.monitor.current_health.issues.append("Test issue")
        self.monitor.current_health.overall_status = HealthStatus.DEGRADED
        
        health = self.monitor.get_health_status()
        
        self.assertEqual(health['overall_status'], 'degraded')
        self.assertIn('Test issue', health['issues'])
    
    @patch('src.core.pipeline_monitor.psutil.Process')
    def test_monitoring_start_stop(self, mock_process):
        """Test monitoring start and stop."""
        # Mock psutil process
        mock_process_instance = Mock()
        mock_process_instance.cpu_percent.return_value = 25.0
        mock_process_instance.memory_info.return_value = Mock(rss=500 * 1024 * 1024)  # 500MB
        mock_process_instance.memory_percent.return_value = 50.0
        mock_process.return_value = mock_process_instance
        
        session_id = "test_session_123"
        
        # Start monitoring
        self.monitor.start_monitoring(session_id)
        
        self.assertTrue(self.monitor.is_monitoring)
        self.assertEqual(self.monitor.current_session_id, session_id)
        self.assertIsNotNone(self.monitor._monitoring_thread)
        self.assertIsNotNone(self.monitor._health_check_thread)
        
        # Let monitoring run briefly
        time.sleep(0.2)
        
        # Check that metrics were collected
        self.assertGreater(len(self.monitor.metrics.cpu_usage_percent), 0)
        self.assertGreater(len(self.monitor.metrics.memory_usage_mb), 0)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        self.assertFalse(self.monitor.is_monitoring)
        
        # Verify analytics integration
        self.mock_analytics.track_event.assert_called()
    
    def test_alert_callbacks(self):
        """Test alert callback functionality."""
        alert_received = []
        
        def alert_callback(alert_type, data):
            alert_received.append((alert_type, data))
        
        self.monitor.add_alert_callback(alert_callback)
        
        # Trigger an alert condition by setting high CPU usage
        self.monitor.metrics.cpu_usage_percent.append(95.0)  # Above critical threshold
        
        # Manually trigger performance check
        self.monitor._check_performance_alerts(95.0, 500.0)
        
        # Check that alert was triggered
        self.assertEqual(len(alert_received), 1)
        self.assertEqual(alert_received[0][0], 'cpu_critical')
    
    @patch('src.core.pipeline_monitor.psutil.Process')
    def test_health_assessment(self, mock_process):
        """Test pipeline health assessment."""
        # Mock process for monitoring
        mock_process_instance = Mock()
        mock_process_instance.cpu_percent.return_value = 15.0
        mock_process_instance.memory_info.return_value = Mock(rss=200 * 1024 * 1024)
        mock_process_instance.memory_percent.return_value = 20.0
        mock_process.return_value = mock_process_instance
        
        # Start monitoring
        self.monitor.start_monitoring("test_session")
        
        # Let health assessment run
        time.sleep(0.15)  # Wait longer than health_check_interval
        
        # Check that health was assessed
        self.assertIsNotNone(self.monitor.current_health.last_assessment)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
    
    def test_health_assessment_with_high_error_rate(self):
        """Test health assessment with high error rate."""
        # Set high error rate
        self.monitor.metrics.error_count = 20
        self.monitor.metrics.audio_chunks_processed = 100  # 20% error rate
        self.monitor.metrics.error_rate = 20.0
        
        # Run health assessment
        self.monitor._assess_pipeline_health()
        
        # Should be critical due to high error rate
        self.assertEqual(self.monitor.current_health.overall_status, HealthStatus.CRITICAL)
        self.assertTrue(any('error rate' in issue.lower() for issue in self.monitor.current_health.issues))
    
    def test_health_assessment_with_high_latency(self):
        """Test health assessment with high latency."""
        # Set high transcription latency
        self.monitor.metrics.transcription_latency_ms.extend([1500.0, 1600.0, 1700.0])
        
        # Run health assessment
        self.monitor._assess_pipeline_health()
        
        # Should be critical due to high latency
        self.assertEqual(self.monitor.current_health.overall_status, HealthStatus.CRITICAL)
        self.assertTrue(any('latency' in issue.lower() for issue in self.monitor.current_health.issues))
    
    def test_health_assessment_with_queue_backlogs(self):
        """Test health assessment with queue backlogs."""
        # Set high queue depths
        self.monitor.metrics.audio_queue_depth = 150
        self.monitor.metrics.transcription_queue_depth = 75
        
        # Run health assessment
        self.monitor._assess_pipeline_health()
        
        # Should have queue-related issues
        issues_text = ' '.join(self.monitor.current_health.issues).lower()
        self.assertTrue('queue' in issues_text or 'backlog' in issues_text)
    
    def test_analytics_integration(self):
        """Test integration with session analytics."""
        session_id = "analytics_test_session"
        
        # Record some operations with analytics integration
        self.monitor.current_session_id = session_id
        
        correlation_id = self.monitor.record_stage_start(PipelineStage.INITIALIZATION)
        self.monitor.record_stage_complete(correlation_id, success=True)
        
        # Verify analytics calls
        self.mock_analytics.track_performance_metric.assert_called()
        
        # Test error recording with analytics
        test_error = RuntimeError("Analytics test error")
        self.monitor.record_error(test_error, PipelineStage.TRANSCRIPTION_PROCESSING)
        
        self.mock_analytics.track_event.assert_called_with(
            AnalyticsEvent.CONNECTION_ERROR,
            session_id,
            unittest.mock.ANY
        )
    
    def test_metric_history_retention(self):
        """Test metric history retention."""
        # Add many metric snapshots
        for i in range(1200):  # More than maxlen of 1000
            self.monitor.metric_history.append({
                'timestamp': datetime.now(),
                'cpu_percent': float(i % 100),
                'memory_mb': float(100 + i % 100)
            })
        
        # Should be limited to maxlen
        self.assertEqual(len(self.monitor.metric_history), 1000)
    
    def test_performance_baseline_tracking(self):
        """Test performance baseline tracking."""
        # Check initial baselines
        self.assertIn('audio_latency_baseline_ms', self.monitor.performance_baselines)
        self.assertIn('transcription_latency_baseline_ms', self.monitor.performance_baselines)
        self.assertIn('cpu_usage_baseline_percent', self.monitor.performance_baselines)
        
        # Verify baseline values are reasonable
        self.assertGreater(self.monitor.performance_baselines['audio_latency_baseline_ms'], 0)
        self.assertGreater(self.monitor.performance_baselines['transcription_latency_baseline_ms'], 0)


if __name__ == '__main__':
    # Set up logging to reduce noise during tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)