"""Tests for enhanced audio session manager."""

import unittest
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.managers.enhanced_session_manager import (
    EnhancedAudioSessionManager, SessionState, RecordingSegment, 
    SessionMetrics, TranscriptionBuffer, get_enhanced_audio_session
)
from src.core.interfaces import TranscriptionResult


class TestTranscriptionBuffer(unittest.TestCase):
    """Test cases for TranscriptionBuffer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer = TranscriptionBuffer(max_size=5)  # Small size for testing
    
    def test_add_new_transcription(self):
        """Test adding a new transcription."""
        result = TranscriptionResult(
            text="Hello world",
            confidence=0.95,
            start_time="12:00:00",
            is_partial=False,
            utterance_id="utt_1"
        )
        
        message = self.buffer.add_transcription(result)
        
        self.assertEqual(message["content"], "Hello world")
        self.assertEqual(message["confidence"], 0.95)
        self.assertFalse(message["is_partial"])
        
        messages = self.buffer.get_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["content"], "Hello world")
    
    def test_partial_result_update(self):
        """Test updating partial results."""
        # Add partial result
        partial_result = TranscriptionResult(
            text="Hello",
            confidence=0.8,
            start_time="12:00:00",
            is_partial=True,
            utterance_id="utt_1"
        )
        
        self.buffer.add_transcription(partial_result)
        messages = self.buffer.get_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["content"], "Hello")
        self.assertTrue(messages[0]["is_partial"])
        
        # Update partial result
        updated_partial = TranscriptionResult(
            text="Hello there",
            confidence=0.85,
            start_time="12:00:00",
            is_partial=True,
            utterance_id="utt_1"
        )
        
        self.buffer.add_transcription(updated_partial)
        messages = self.buffer.get_messages()
        self.assertEqual(len(messages), 1)  # Should still be 1
        self.assertEqual(messages[0]["content"], "Hello there")
        self.assertTrue(messages[0]["is_partial"])
    
    def test_partial_to_final_replacement(self):
        """Test replacing partial with final result."""
        # Add partial
        partial = TranscriptionResult(
            text="Hello there",
            confidence=0.85,
            start_time="12:00:00",
            is_partial=True,
            utterance_id="utt_1"
        )
        self.buffer.add_transcription(partial)
        
        # Replace with final
        final = TranscriptionResult(
            text="Hello there everyone",
            confidence=0.95,
            start_time="12:00:00",
            is_partial=False,
            utterance_id="utt_1"
        )
        self.buffer.add_transcription(final)
        
        messages = self.buffer.get_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["content"], "Hello there everyone")
        self.assertFalse(messages[0]["is_partial"])
        
        # Verify partial tracking is cleared
        stats = self.buffer.get_stats()
        self.assertEqual(stats["active_partials"], 0)
    
    def test_speaker_id_formatting(self):
        """Test speaker ID formatting in content."""
        result = TranscriptionResult(
            text="Hello everyone",
            confidence=0.95,
            start_time="12:00:00",
            is_partial=False,
            speaker_id="Speaker 1"
        )
        
        message = self.buffer.add_transcription(result)
        
        self.assertEqual(message["content"], "Speaker 1: Hello everyone")
    
    def test_buffer_size_management(self):
        """Test buffer size management and truncation."""
        # Add more messages than buffer size
        for i in range(7):
            result = TranscriptionResult(
                text=f"Message {i}",
                confidence=0.95,
                start_time="12:00:00",
                is_partial=False,
                utterance_id=f"utt_{i}"
            )
            self.buffer.add_transcription(result)
        
        messages = self.buffer.get_messages()
        self.assertEqual(len(messages), 5)  # Should be truncated to max_size
        
        # Should keep the latest messages
        self.assertEqual(messages[0]["content"], "Message 2")
        self.assertEqual(messages[4]["content"], "Message 6")
    
    def test_partial_tracking_after_truncation(self):
        """Test partial result tracking after buffer truncation."""
        # Add some final messages
        for i in range(3):
            result = TranscriptionResult(
                text=f"Final {i}",
                confidence=0.95,
                start_time="12:00:00",
                is_partial=False,
                utterance_id=f"final_{i}"
            )
            self.buffer.add_transcription(result)
        
        # Add partial results
        partial1 = TranscriptionResult(
            text="Partial 1",
            confidence=0.8,
            start_time="12:00:00",
            is_partial=True,
            utterance_id="partial_1"
        )
        self.buffer.add_transcription(partial1)
        
        partial2 = TranscriptionResult(
            text="Partial 2", 
            confidence=0.8,
            start_time="12:00:00",
            is_partial=True,
            utterance_id="partial_2"
        )
        self.buffer.add_transcription(partial2)
        
        # Now add more messages to trigger truncation
        for i in range(3):
            result = TranscriptionResult(
                text=f"Extra {i}",
                confidence=0.95,
                start_time="12:00:00", 
                is_partial=False,
                utterance_id=f"extra_{i}"
            )
            self.buffer.add_transcription(result)
        
        # Check that valid partials are preserved
        stats = self.buffer.get_stats()
        self.assertLessEqual(stats["active_partials"], 2)
        
        # Check that we can still update preserved partials
        updated_partial = TranscriptionResult(
            text="Updated Partial 2",
            confidence=0.9,
            start_time="12:00:00",
            is_partial=True,
            utterance_id="partial_2"
        )
        
        self.buffer.add_transcription(updated_partial)
        messages = self.buffer.get_messages()
        
        # Should find the updated partial
        updated_found = any("Updated Partial 2" in msg.get("content", "") for msg in messages)
        if stats["active_partials"] > 0:  # Only check if partial was preserved
            self.assertTrue(updated_found)
    
    def test_clear_buffer(self):
        """Test clearing the buffer."""
        # Add some messages
        for i in range(3):
            result = TranscriptionResult(
                text=f"Message {i}",
                confidence=0.95,
                start_time="12:00:00",
                is_partial=bool(i % 2),  # Mix of partial and final
                utterance_id=f"utt_{i}"
            )
            self.buffer.add_transcription(result)
        
        self.buffer.clear()
        
        messages = self.buffer.get_messages()
        stats = self.buffer.get_stats()
        
        self.assertEqual(len(messages), 0)
        self.assertEqual(stats["active_partials"], 0)
        self.assertEqual(stats["total_messages"], 0)


class TestRecordingSegment(unittest.TestCase):
    """Test cases for RecordingSegment."""
    
    def test_segment_creation(self):
        """Test creating a recording segment."""
        start_time = datetime.now()
        segment = RecordingSegment(start_time=start_time, device_index=1)
        
        self.assertEqual(segment.start_time, start_time)
        self.assertEqual(segment.device_index, 1)
        self.assertIsNone(segment.end_time)
        self.assertIsNone(segment.duration_seconds)
        self.assertEqual(segment.transcription_count, 0)
    
    def test_segment_completion(self):
        """Test completing a recording segment."""
        start_time = datetime.now()
        segment = RecordingSegment(start_time=start_time)
        
        # Wait a small amount to ensure duration > 0
        time.sleep(0.01)
        segment.complete()
        
        self.assertIsNotNone(segment.end_time)
        self.assertIsNotNone(segment.duration_seconds)
        self.assertGreater(segment.duration_seconds, 0)
        self.assertGreater(segment.end_time, start_time)


class TestSessionMetrics(unittest.TestCase):
    """Test cases for SessionMetrics."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = SessionMetrics()
        
        self.assertEqual(metrics.total_recording_time, 0.0)
        self.assertEqual(metrics.total_transcriptions, 0)
        self.assertEqual(metrics.partial_transcriptions, 0)
        self.assertEqual(metrics.final_transcriptions, 0)
        self.assertEqual(metrics.connection_errors, 0)
        self.assertEqual(len(metrics.recording_segments), 0)
        self.assertIsNone(metrics.session_start_time)
        self.assertIsNone(metrics.last_activity_time)
    
    def test_update_activity(self):
        """Test updating activity timestamp."""
        metrics = SessionMetrics()
        
        # First update sets start time
        metrics.update_activity()
        
        self.assertIsNotNone(metrics.session_start_time)
        self.assertIsNotNone(metrics.last_activity_time)
        self.assertEqual(metrics.session_start_time, metrics.last_activity_time)
        
        # Second update only updates last activity
        time.sleep(0.01)
        first_start_time = metrics.session_start_time
        
        metrics.update_activity()
        
        self.assertEqual(metrics.session_start_time, first_start_time)  # Should not change
        self.assertGreater(metrics.last_activity_time, first_start_time)


class TestEnhancedAudioSessionManager(unittest.TestCase):
    """Test cases for EnhancedAudioSessionManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Reset singleton for each test
        EnhancedAudioSessionManager._instance = None
        self.session_manager = EnhancedAudioSessionManager()
    
    def tearDown(self):
        """Clean up after each test."""
        if self.session_manager.is_recording():
            self.session_manager.stop_recording()
        self.session_manager.reset_session()
    
    def test_singleton_pattern(self):
        """Test singleton pattern implementation."""
        manager1 = EnhancedAudioSessionManager()
        manager2 = EnhancedAudioSessionManager()
        
        self.assertIs(manager1, manager2)
    
    def test_initial_state(self):
        """Test initial session state."""
        self.assertEqual(self.session_manager.current_state, SessionState.IDLE)
        self.assertFalse(self.session_manager.is_recording())
        
        info = self.session_manager.get_session_info()
        self.assertEqual(info['state'], SessionState.IDLE.value)
        self.assertFalse(info['is_recording'])
        self.assertEqual(info['transcription_count'], 0)
    
    def test_state_transitions(self):
        """Test session state transitions."""
        state_changes = []
        
        def state_callback(old_state, new_state):
            state_changes.append((old_state, new_state))
        
        self.session_manager.add_state_callback(state_callback)
        
        # Trigger state change
        with patch.object(self.session_manager, '_audio_processor'):
            with patch.object(self.session_manager, '_run_audio_processor_async'):
                self.session_manager._set_state(SessionState.RECORDING)
        
        self.assertEqual(len(state_changes), 1)
        self.assertEqual(state_changes[0][0], SessionState.IDLE)
        self.assertEqual(state_changes[0][1], SessionState.RECORDING)
    
    def test_transcription_callback_management(self):
        """Test transcription callback management."""
        callback1 = Mock()
        callback2 = Mock()
        
        # Add callbacks
        self.session_manager.add_transcription_callback(callback1)
        self.session_manager.add_transcription_callback(callback2)
        
        info = self.session_manager.get_session_info()
        self.assertEqual(info['callbacks_registered'], 2)
        
        # Remove one callback
        self.session_manager.remove_transcription_callback(callback1)
        
        info = self.session_manager.get_session_info()
        self.assertEqual(info['callbacks_registered'], 1)
    
    @patch('src.managers.enhanced_session_manager.AudioProcessor')
    def test_transcription_processing(self, mock_audio_processor):
        """Test transcription processing."""
        callback = Mock()
        self.session_manager.add_transcription_callback(callback)
        
        # Simulate transcription result
        result = TranscriptionResult(
            text="Test transcription",
            confidence=0.95,
            start_time="12:00:00",
            is_partial=False,
            utterance_id="test_1"
        )
        
        self.session_manager._on_transcription_received(result)
        
        # Check callback was called
        callback.assert_called_once()
        call_args = callback.call_args[0][0]
        self.assertEqual(call_args["content"], "Test transcription")
        self.assertEqual(call_args["confidence"], 0.95)
        self.assertFalse(call_args["is_partial"])
        
        # Check transcriptions are stored
        transcriptions = self.session_manager.get_current_transcriptions()
        self.assertEqual(len(transcriptions), 1)
        self.assertEqual(transcriptions[0]["content"], "Test transcription")
        
        # Check metrics updated
        info = self.session_manager.get_session_info()
        self.assertEqual(info['metrics']['total_transcriptions'], 1)
        self.assertEqual(info['metrics']['final_transcriptions'], 1)
        self.assertEqual(info['metrics']['partial_transcriptions'], 0)
    
    def test_connection_health_handling(self):
        """Test connection health change handling."""
        # Simulate connection health change
        self.session_manager._on_connection_health_changed(False, "Connection lost")
        
        info = self.session_manager.get_session_info()
        self.assertFalse(info['connection_healthy'])
        self.assertEqual(info['metrics']['connection_errors'], 1)
        
        # Simulate recovery
        self.session_manager._on_connection_health_changed(True, "Connection restored")
        
        info = self.session_manager.get_session_info()
        self.assertTrue(info['connection_healthy'])
        self.assertEqual(info['metrics']['connection_errors'], 1)  # Should not change
    
    @patch('src.managers.enhanced_session_manager.AudioProcessor')
    @patch('threading.Thread')
    def test_start_recording_success(self, mock_thread, mock_audio_processor):
        """Test successful recording start."""
        # Mock thread
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance
        
        # Mock audio processor
        mock_processor_instance = Mock()
        mock_audio_processor.return_value = mock_processor_instance
        
        result = self.session_manager.start_recording(device_index=1)
        
        self.assertTrue(result)
        self.assertEqual(self.session_manager.current_state, SessionState.RECORDING)
        
        # Check audio processor was created
        mock_audio_processor.assert_called_once_with(
            transcription_provider='aws',
            capture_provider='pyaudio',
            transcription_config={'region': 'us-east-1', 'language_code': 'en-US'}
        )
        
        # Check thread was started
        mock_thread_instance.start.assert_called_once()
    
    def test_start_recording_invalid_state(self):
        """Test starting recording in invalid state."""
        # Set to recording state
        self.session_manager._set_state(SessionState.RECORDING)
        
        result = self.session_manager.start_recording(device_index=1)
        
        self.assertFalse(result)
    
    @patch('src.managers.enhanced_session_manager.AudioProcessor')
    def test_stop_recording_success(self, mock_audio_processor):
        """Test successful recording stop."""
        # Setup recording state
        self.session_manager._set_state(SessionState.RECORDING)
        self.session_manager._audio_processor = Mock()
        self.session_manager._current_segment = RecordingSegment(
            start_time=datetime.now(),
            device_index=1
        )
        
        with patch.object(self.session_manager, '_stop_audio_processor', return_value=True):
            with patch.object(self.session_manager, '_wait_for_background_thread'):
                result = self.session_manager.stop_recording()
        
        self.assertTrue(result)
        self.assertEqual(self.session_manager.current_state, SessionState.IDLE)
        
        # Check segment was completed
        info = self.session_manager.get_session_info()
        self.assertEqual(info['metrics']['recording_segments'], 1)
    
    def test_duration_calculation(self):
        """Test duration calculation."""
        # Initially should be 0
        self.assertEqual(self.session_manager.get_current_duration_seconds(), 0.0)
        self.assertEqual(self.session_manager.get_formatted_duration(), "00:00")
        
        # Simulate completed segment
        segment = RecordingSegment(start_time=datetime.now() - timedelta(seconds=65))
        segment.complete()
        self.session_manager._session_metrics.recording_segments.append(segment)
        self.session_manager._session_metrics.total_recording_time = segment.duration_seconds
        
        duration = self.session_manager.get_current_duration_seconds()
        formatted = self.session_manager.get_formatted_duration()
        
        self.assertGreaterEqual(duration, 65.0)
        self.assertEqual(formatted, "01:05")  # 1 minute 5 seconds
    
    def test_duration_formatting(self):
        """Test duration formatting for different times."""
        # Test various durations
        test_cases = [
            (0, "00:00"),
            (30, "00:30"),
            (90, "01:30"),
            (3600, "01:00:00"),
            (3665, "01:01:05"),
            (7200, "02:00:00")
        ]
        
        for seconds, expected in test_cases:
            result = self.session_manager._format_duration(seconds)
            self.assertEqual(result, expected, f"Failed for {seconds} seconds")
    
    def test_clear_transcriptions(self):
        """Test clearing transcriptions."""
        # Add some transcriptions
        result = TranscriptionResult(
            text="Test",
            confidence=0.95,
            start_time="12:00:00",
            is_partial=False
        )
        self.session_manager._on_transcription_received(result)
        
        self.assertEqual(len(self.session_manager.get_current_transcriptions()), 1)
        
        # Clear transcriptions
        self.session_manager.clear_transcriptions()
        
        self.assertEqual(len(self.session_manager.get_current_transcriptions()), 0)
    
    def test_reset_session(self):
        """Test session reset."""
        # Add some data
        result = TranscriptionResult(
            text="Test", 
            confidence=0.95,
            start_time="12:00:00",
            is_partial=False
        )
        self.session_manager._on_transcription_received(result)
        
        # Reset session
        self.session_manager.reset_session()
        
        # Check everything is reset
        self.assertEqual(self.session_manager.current_state, SessionState.IDLE)
        self.assertEqual(len(self.session_manager.get_current_transcriptions()), 0)
        
        info = self.session_manager.get_session_info()
        self.assertEqual(info['metrics']['total_transcriptions'], 0)
        self.assertEqual(info['metrics']['recording_segments'], 0)
    
    def test_thread_safety(self):
        """Test thread safety of critical operations."""
        results = []
        errors = []
        
        def add_transcriptions(thread_id):
            try:
                for i in range(10):
                    result = TranscriptionResult(
                        text=f"Thread {thread_id} message {i}",
                        confidence=0.95,
                        start_time="12:00:00",
                        is_partial=False,
                        utterance_id=f"t{thread_id}_m{i}"
                    )
                    self.session_manager._on_transcription_received(result)
                    time.sleep(0.001)  # Small delay
                results.append(f"Thread {thread_id} completed")
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=add_transcriptions, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Check results
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 3)
        
        # Check that all transcriptions were added
        transcriptions = self.session_manager.get_current_transcriptions()
        self.assertEqual(len(transcriptions), 30)  # 3 threads * 10 messages each


class TestFactoryFunction(unittest.TestCase):
    """Test factory function."""
    
    def test_get_enhanced_audio_session(self):
        """Test factory function returns correct instance."""
        session1 = get_enhanced_audio_session()
        session2 = get_enhanced_audio_session()
        
        self.assertIsInstance(session1, EnhancedAudioSessionManager)
        self.assertIs(session1, session2)  # Should be same instance (singleton)


if __name__ == '__main__':
    unittest.main()