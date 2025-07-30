"""Tests for session analytics system."""

import unittest
import tempfile
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from src.analytics.session_analytics import (
    SessionAnalytics, AnalyticsEvent, AnalyticsEventData, SessionMetrics,
    PerformanceMetrics
)


class TestAnalyticsEventData(unittest.TestCase):
    """Test cases for AnalyticsEventData."""
    
    def test_event_data_creation(self):
        """Test creating analytics event data."""
        timestamp = datetime.now()
        event_data = AnalyticsEventData(
            event_type=AnalyticsEvent.SESSION_STARTED,
            timestamp=timestamp,
            session_id="test_session",
            data={"test": "value"}
        )
        
        self.assertEqual(event_data.event_type, AnalyticsEvent.SESSION_STARTED)
        self.assertEqual(event_data.timestamp, timestamp)
        self.assertEqual(event_data.session_id, "test_session")
        self.assertEqual(event_data.data["test"], "value")
    
    def test_event_data_serialization(self):
        """Test event data serialization to dictionary."""
        timestamp = datetime.now()
        event_data = AnalyticsEventData(
            event_type=AnalyticsEvent.TRANSCRIPTION_RECEIVED,
            timestamp=timestamp,
            session_id="test_session",
            data={"text": "hello world", "confidence": 0.95}
        )
        
        serialized = event_data.to_dict()
        
        self.assertEqual(serialized["event_type"], "transcription_received")
        self.assertEqual(serialized["timestamp"], timestamp.isoformat())
        self.assertEqual(serialized["session_id"], "test_session")
        self.assertEqual(serialized["data"]["text"], "hello world")
        self.assertEqual(serialized["data"]["confidence"], 0.95)


class TestSessionMetrics(unittest.TestCase):
    """Test cases for SessionMetrics."""
    
    def test_metrics_creation(self):
        """Test creating session metrics."""
        start_time = datetime.now()
        metrics = SessionMetrics(
            session_id="test_123",
            start_time=start_time,
            transcription_count=10,
            word_count=50,
            average_confidence=0.85
        )
        
        self.assertEqual(metrics.session_id, "test_123")
        self.assertEqual(metrics.start_time, start_time)
        self.assertEqual(metrics.transcription_count, 10)
        self.assertEqual(metrics.word_count, 50)
        self.assertEqual(metrics.average_confidence, 0.85)
        self.assertIsNone(metrics.end_time)
    
    def test_metrics_serialization(self):
        """Test metrics serialization to dictionary."""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=5)
        
        metrics = SessionMetrics(
            session_id="test_456",
            start_time=start_time,
            end_time=end_time,
            total_duration=300.0
        )
        
        serialized = metrics.to_dict()
        
        self.assertEqual(serialized["session_id"], "test_456")
        self.assertEqual(serialized["start_time"], start_time.isoformat())
        self.assertEqual(serialized["end_time"], end_time.isoformat())
        self.assertEqual(serialized["total_duration"], 300.0)


class TestPerformanceMetrics(unittest.TestCase):
    """Test cases for PerformanceMetrics."""
    
    def test_performance_metrics_initialization(self):
        """Test performance metrics initialization."""
        metrics = PerformanceMetrics()
        
        self.assertEqual(len(metrics.transcription_latency), 0)
        self.assertEqual(len(metrics.memory_usage), 0)
        self.assertEqual(len(metrics.cpu_usage), 0)
        self.assertEqual(len(metrics.network_latency), 0)
        self.assertEqual(len(metrics.error_rates), 0)
    
    def test_average_latency_calculation(self):
        """Test average latency calculation."""
        metrics = PerformanceMetrics()
        
        # Initially should be 0
        self.assertEqual(metrics.get_average_latency(), 0.0)
        
        # Add some latency values
        metrics.transcription_latency.extend([100, 200, 150])
        
        # Should calculate average
        self.assertEqual(metrics.get_average_latency(), 150.0)
    
    def test_peak_memory_calculation(self):
        """Test peak memory calculation."""
        metrics = PerformanceMetrics()
        
        # Initially should be 0
        self.assertEqual(metrics.get_peak_memory(), 0.0)
        
        # Add memory usage values
        metrics.memory_usage.extend([50.5, 75.2, 60.8, 90.1])
        
        # Should return maximum
        self.assertEqual(metrics.get_peak_memory(), 90.1)
    
    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        metrics = PerformanceMetrics()
        
        # Initially should be 0
        self.assertEqual(metrics.get_error_rate(), 0.0)
        
        # Add error rates
        metrics.error_rates.extend([0.05, 0.10, 0.03, 0.07])
        
        # Should calculate average
        expected_avg = (0.05 + 0.10 + 0.03 + 0.07) / 4
        self.assertEqual(metrics.get_error_rate(), expected_avg)


class TestSessionAnalytics(unittest.TestCase):
    """Test cases for SessionAnalytics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name) / "test_analytics"
        self.analytics = SessionAnalytics(storage_directory=self.storage_dir)
    
    def tearDown(self):
        """Clean up after each test."""
        self.analytics.cleanup()
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test analytics system initialization."""
        self.assertIsNotNone(self.analytics.storage_directory)
        self.assertTrue(self.analytics.storage_directory.exists())
        self.assertEqual(self.analytics.max_events, 1000)
        self.assertIsNone(self.analytics._current_session_id)
    
    def test_session_lifecycle(self):
        """Test complete session lifecycle tracking."""
        session_id = "test_session_123"
        
        # Start session
        metrics = self.analytics.start_session(session_id)
        
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.session_id, session_id)
        self.assertEqual(self.analytics._current_session_id, session_id)
        
        # Track some events
        self.analytics.track_transcription(
            session_id, "Hello world", 0.95, False, 150.0
        )
        
        # End session
        final_metrics = self.analytics.end_session(session_id)
        
        self.assertIsNotNone(final_metrics)
        self.assertIsNotNone(final_metrics.end_time)
        self.assertEqual(final_metrics.transcription_count, 1)
        self.assertEqual(final_metrics.final_transcription_count, 1)
        self.assertIsNone(self.analytics._current_session_id)
    
    def test_event_tracking(self):
        """Test basic event tracking."""
        session_id = "event_test_session"
        self.analytics.start_session(session_id)
        
        # Track various events
        self.analytics.track_event(
            AnalyticsEvent.RECORDING_STARTED,
            session_id,
            {"device": "mic_1"}
        )
        
        self.analytics.track_user_action(session_id, "button_click", {"button": "start"})
        
        self.analytics.track_performance_metric(
            session_id, "transcription_latency", 120.5
        )
        
        # Check events were recorded
        recent_events = self.analytics.get_recent_events()
        
        # Should have session start + 3 tracked events
        self.assertGreaterEqual(len(recent_events), 4)
        
        # Check event types
        event_types = [e["event_type"] for e in recent_events]
        self.assertIn("session_started", event_types)
        self.assertIn("recording_started", event_types)
        self.assertIn("user_action", event_types)
        self.assertIn("performance_metric", event_types)
    
    def test_transcription_tracking(self):
        """Test transcription-specific tracking."""
        session_id = "transcription_test"
        self.analytics.start_session(session_id)
        
        # Track partial transcription
        self.analytics.track_transcription(
            session_id, "Hello", 0.8, True, 100.0
        )
        
        # Track final transcription
        self.analytics.track_transcription(
            session_id, "Hello world", 0.95, False, 150.0
        )
        
        # Check metrics were updated
        metrics = self.analytics.get_current_session_metrics()
        
        self.assertEqual(metrics.transcription_count, 2)
        self.assertEqual(metrics.partial_transcription_count, 1)
        self.assertEqual(metrics.final_transcription_count, 1)
        self.assertEqual(metrics.word_count, 3)  # "Hello" + "Hello world" = 1 + 2
        self.assertGreater(metrics.average_confidence, 0.8)
    
    def test_performance_metrics_tracking(self):
        """Test performance metrics tracking."""
        session_id = "performance_test"
        self.analytics.start_session(session_id)
        
        # Track various performance metrics
        self.analytics.track_performance_metric(
            session_id, "transcription_latency", 120.0
        )
        self.analytics.track_performance_metric(
            session_id, "memory_usage", 75.5
        )
        self.analytics.track_performance_metric(
            session_id, "cpu_usage", 45.2
        )
        
        # Check performance summary
        summary = self.analytics.get_performance_summary()
        
        self.assertEqual(summary["average_transcription_latency"], 120.0)
        self.assertEqual(summary["peak_memory_usage"], 75.5)
        self.assertEqual(summary["active_sessions"], 1)
    
    def test_usage_statistics(self):
        """Test usage statistics collection."""
        session_id = "stats_test"
        self.analytics.start_session(session_id)
        
        # Track various events to generate statistics
        for i in range(5):
            self.analytics.track_transcription(
                session_id, f"Message {i}", 0.9, False
            )
        
        self.analytics.track_user_action(session_id, "save")
        self.analytics.track_user_action(session_id, "export")
        
        # Get daily statistics
        daily_stats = self.analytics.get_usage_statistics("daily")
        
        self.assertEqual(daily_stats["period"], "daily")
        self.assertGreater(len(daily_stats["statistics"]), 0)
        
        # Check that today's statistics exist
        today = datetime.now().date().isoformat()
        if today in daily_stats["statistics"]:
            today_stats = daily_stats["statistics"][today]
            self.assertGreater(today_stats.get("transcription_received", 0), 0)
            self.assertGreater(today_stats.get("user_action", 0), 0)
    
    def test_session_report_generation(self):
        """Test session report generation."""
        session_id = "report_test"
        self.analytics.start_session(session_id)
        
        # Add some data for a meaningful report
        self.analytics.track_transcription(session_id, "Test transcription", 0.95, False)
        self.analytics.track_user_action(session_id, "save")
        self.analytics.track_event(
            AnalyticsEvent.CONNECTION_ERROR, session_id, {"error": "timeout"}
        )
        
        # End session to finalize metrics
        self.analytics.end_session(session_id)
        
        # Generate report
        report = self.analytics.generate_session_report(session_id)
        
        self.assertIsNotNone(report)
        self.assertEqual(report["session_id"], session_id)
        self.assertIn("metrics", report)
        self.assertIn("events_count", report)
        self.assertIn("insights", report)
        self.assertIn("performance", report)
        
        # Check that insights were generated
        self.assertIsInstance(report["insights"], list)
    
    def test_analytics_callbacks(self):
        """Test analytics event callbacks."""
        callback_events = []
        
        def test_callback(event_data):
            callback_events.append(event_data)
        
        self.analytics.add_analytics_callback(test_callback)
        
        session_id = "callback_test"
        self.analytics.start_session(session_id)
        
        self.analytics.track_user_action(session_id, "test_action")
        
        # Should have received callbacks
        self.assertGreaterEqual(len(callback_events), 2)  # session start + user action
        
        # Check callback data
        user_action_events = [e for e in callback_events if e.event_type == AnalyticsEvent.USER_ACTION]
        self.assertEqual(len(user_action_events), 1)
        self.assertEqual(user_action_events[0].data["action"], "test_action")
        
        # Remove callback
        self.analytics.remove_analytics_callback(test_callback)
        
        # Should not receive more callbacks
        callback_count_before = len(callback_events)
        self.analytics.track_user_action(session_id, "another_action")
        
        # Callback count should not increase (only session events, not our callback)
        self.assertEqual(len(callback_events), callback_count_before)
    
    def test_session_metrics_persistence(self):
        """Test session metrics persistence to file."""
        session_id = "persistence_test"
        self.analytics.start_session(session_id)
        
        # Add some data
        self.analytics.track_transcription(session_id, "Test data", 0.9, False)
        
        # End session (should trigger save)
        metrics = self.analytics.end_session(session_id)
        
        # Check that file was created
        metrics_file = self.storage_dir / f"session_metrics_{session_id}.json"
        self.assertTrue(metrics_file.exists())
        
        # Check file content
        with open(metrics_file, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data["session_id"], session_id)
        self.assertEqual(saved_data["transcription_count"], 1)
    
    def test_data_export(self):
        """Test analytics data export."""
        session_id = "export_test"
        self.analytics.start_session(session_id)
        
        # Add test data
        self.analytics.track_transcription(session_id, "Export test", 0.9, False)
        self.analytics.track_user_action(session_id, "export")
        
        self.analytics.end_session(session_id)
        
        # Export data
        exported_data = self.analytics.export_analytics_data()
        
        self.assertIn("export_timestamp", exported_data)
        self.assertIn("events", exported_data)
        self.assertIn("session_metrics", exported_data)
        self.assertIn("performance_summary", exported_data)
        self.assertIn("usage_statistics", exported_data)
        
        # Check that our session is included
        self.assertIn(session_id, exported_data["session_metrics"])
    
    def test_date_range_export(self):
        """Test data export with date range filtering."""
        # Create sessions on different days (simulated by setting different timestamps)
        old_session = "old_session"
        new_session = "new_session"
        
        # Start both sessions
        self.analytics.start_session(old_session)
        self.analytics.end_session(old_session)
        
        self.analytics.start_session(new_session)
        self.analytics.end_session(new_session)
        
        # Export with date range (last 1 hour)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        exported_data = self.analytics.export_analytics_data(start_date=one_hour_ago)
        
        # Should include recent sessions
        self.assertGreaterEqual(len(exported_data["session_metrics"]), 2)
    
    def test_recent_events_filtering(self):
        """Test filtering of recent events."""
        session_id = "filter_test"
        self.analytics.start_session(session_id)
        
        # Add various event types
        self.analytics.track_user_action(session_id, "action1")
        self.analytics.track_transcription(session_id, "text", 0.9, False)
        self.analytics.track_user_action(session_id, "action2")
        
        # Get all recent events
        all_events = self.analytics.get_recent_events()
        self.assertGreaterEqual(len(all_events), 4)  # session start + 3 events
        
        # Get only user action events
        user_action_events = self.analytics.get_recent_events(
            event_type=AnalyticsEvent.USER_ACTION
        )
        self.assertEqual(len(user_action_events), 2)
        
        # Get limited number of events
        limited_events = self.analytics.get_recent_events(limit=2)
        self.assertEqual(len(limited_events), 2)
    
    def test_concurrent_session_handling(self):
        """Test thread safety of session operations."""
        # Start a single session
        session_id = "thread_safety_test"
        self.analytics.start_session(session_id)
        
        errors = []
        event_counts = []
        
        def track_events_concurrently(thread_num):
            try:
                # Multiple threads tracking events on the same session
                for i in range(5):
                    self.analytics.track_transcription(
                        session_id, f"Thread {thread_num} Message {i}", 0.9, False
                    )
                    self.analytics.track_user_action(session_id, f"action_{thread_num}_{i}")
                    time.sleep(0.001)  # Small delay
                
                event_counts.append(thread_num)
                
            except Exception as e:
                errors.append(f"Thread {thread_num} error: {e}")
        
        # Run multiple threads concurrently tracking events
        threads = []
        for i in range(3):
            thread = threading.Thread(target=track_events_concurrently, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Check results
        self.assertEqual(len(errors), 0, f"Concurrent errors: {errors}")
        self.assertEqual(len(event_counts), 3, "All threads should have completed")
        
        # End session
        final_metrics = self.analytics.end_session(session_id)
        self.assertIsNotNone(final_metrics)
        
        # Verify that events from all threads were tracked
        # Each thread tracked 5 transcriptions and 5 user actions = 10 events each
        # 3 threads = 30 events total (plus session start/end events)
        self.assertGreaterEqual(final_metrics.transcription_count, 15)  # 3 threads * 5 transcriptions
        self.assertGreaterEqual(final_metrics.user_actions, 15)  # 3 threads * 5 actions
    
    def test_metrics_boundary_conditions(self):
        """Test metrics calculations with boundary conditions."""
        session_id = "boundary_test"
        self.analytics.start_session(session_id)
        
        # Test with empty transcription
        self.analytics.track_transcription(session_id, "", 0.0, False)
        
        # Test with very high confidence
        self.analytics.track_transcription(session_id, "Perfect", 1.0, False)
        
        # Test with very low confidence
        self.analytics.track_transcription(session_id, "Unclear", 0.1, False)
        
        metrics = self.analytics.get_current_session_metrics()
        
        # Should handle all cases gracefully
        self.assertEqual(metrics.transcription_count, 3)
        self.assertGreaterEqual(metrics.average_confidence, 0.0)
        self.assertLessEqual(metrics.average_confidence, 1.0)
    
    def test_cleanup_functionality(self):
        """Test cleanup functionality."""
        session_id = "cleanup_test"
        
        # Create callback to test cleanup
        callback_called = []
        def test_callback(event_data):
            callback_called.append(True)
        
        self.analytics.add_analytics_callback(test_callback)
        self.analytics.start_session(session_id)
        
        # Verify callback works
        self.analytics.track_user_action(session_id, "test")
        self.assertGreater(len(callback_called), 0)
        
        # Cleanup
        self.analytics.cleanup()
        
        # Should have ended session and cleared callbacks
        self.assertIsNone(self.analytics._current_session_id)
        self.assertEqual(len(self.analytics._analytics_callbacks), 0)


if __name__ == '__main__':
    unittest.main()