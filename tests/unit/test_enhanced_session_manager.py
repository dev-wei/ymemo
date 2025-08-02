"""Enhanced audio session manager tests using new test infrastructure.

Migrated from unittest to pytest with centralized fixtures and base classes.
Eliminates duplication and provides consistent patterns.
"""

import contextlib
import time
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.managers.enhanced_session_manager import (
    EnhancedAudioSessionManager,
    RecordingSegment,
    SessionMetrics,
    SessionState,
    TranscriptionBuffer,
    get_enhanced_audio_session,
)
from tests.base.base_test import BaseIntegrationTest, BaseTest


class TestTranscriptionBuffer(BaseTest):
    """Test cases for TranscriptionBuffer using new infrastructure."""

    def setup_method(self):
        """Set up test fixtures using base class."""
        super().setup_method()
        self.buffer = TranscriptionBuffer(max_size=5)  # Small size for testing

    @pytest.mark.unit
    def test_add_new_transcription(self, transcription_result_factory):
        """Test adding a new transcription using centralized factory."""
        result = transcription_result_factory.create_final_result(text="Hello world")
        # Customize confidence if needed
        result.confidence = 0.95

        message = self.buffer.add_transcription(result)

        assert "Hello world" in message["content"]  # Content includes speaker info
        assert message["confidence"] == 0.95
        assert not message["is_partial"]

        messages = self.buffer.get_messages()
        assert len(messages) == 1
        assert "Hello world" in messages[0]["content"]

    @pytest.mark.unit
    def test_partial_result_update(self, transcription_result_factory):
        """Test updating partial results using result factory."""
        # Add partial result
        partial_result = transcription_result_factory.create_partial_result(
            text="Hello"
        )
        partial_result.confidence = 0.8

        self.buffer.add_transcription(partial_result)
        messages = self.buffer.get_messages()
        assert len(messages) == 1
        assert "Hello" in messages[0]["content"]
        assert messages[0]["is_partial"]

        # Update partial result
        updated_partial = transcription_result_factory.create_partial_result(
            text="Hello there"
        )
        updated_partial.confidence = 0.85
        updated_partial.utterance_id = partial_result.utterance_id  # Same utterance

        self.buffer.add_transcription(updated_partial)
        messages = self.buffer.get_messages()
        assert len(messages) == 1  # Should still be 1
        assert "Hello there" in messages[0]["content"]
        assert messages[0]["is_partial"]

    @pytest.mark.unit
    def test_partial_to_final_replacement(self, transcription_result_factory):
        """Test replacing partial with final result using factory."""
        # Add partial
        partial = transcription_result_factory.create_partial_result(text="Hello there")
        partial.confidence = 0.85
        self.buffer.add_transcription(partial)

        # Replace with final
        final = transcription_result_factory.create_final_result(
            text="Hello there everyone"
        )
        final.confidence = 0.95
        final.utterance_id = partial.utterance_id  # Same utterance

        self.buffer.add_transcription(final)

        messages = self.buffer.get_messages()
        assert len(messages) == 1
        assert "Hello there everyone" in messages[0]["content"]
        assert not messages[0]["is_partial"]

    @pytest.mark.unit
    def test_buffer_max_size_enforcement(self, transcription_result_factory):
        """Test buffer size limits using result factory."""
        # Fill buffer to max capacity
        for i in range(6):  # Buffer max is 5, so 6 should overflow
            result = transcription_result_factory.create_final_result(
                text=f"Message {i}"
            )
            result.utterance_id = f"utt_{i}"  # Customize utterance ID
            self.buffer.add_transcription(result)

        messages = self.buffer.get_messages()
        assert len(messages) == 5  # Should not exceed max_size

        # Check that oldest message was removed (FIFO)
        contents = [msg["content"] for msg in messages]
        assert not any("Message 0" in content for content in contents)
        assert any("Message 5" in content for content in contents)


class TestRecordingSegment(BaseTest):
    """Test cases for RecordingSegment using new infrastructure."""

    @pytest.mark.unit
    def test_segment_creation(self):
        """Test recording segment creation with base test utilities."""
        start_time = datetime.now()
        segment = RecordingSegment(start_time=start_time, device_index=0)

        assert segment.start_time == start_time
        assert segment.device_index == 0
        assert segment.end_time is None
        assert segment.duration_seconds is None
        assert segment.transcription_count == 0

    @pytest.mark.unit
    def test_segment_completion(self):
        """Test segment completion and duration calculation."""
        start_time = datetime.now()
        segment = RecordingSegment(start_time=start_time, device_index=1)

        # Wait a bit then complete
        time.sleep(0.01)  # 10ms
        segment.complete()

        assert segment.end_time is not None
        assert segment.duration_seconds is not None
        assert segment.duration_seconds > 0

    @pytest.mark.unit
    def test_segment_transcription_tracking(self, transcription_result_factory):
        """Test transcription tracking in segments using factory."""
        segment = RecordingSegment(start_time=datetime.now(), device_index=1)

        # Initially no transcriptions
        assert segment.transcription_count == 0

        # Note: Based on the dataclass, RecordingSegment doesn't store transcriptions
        # It only tracks the count, so we'll test that the count can be updated
        segment.transcription_count = 1
        assert segment.transcription_count == 1


class TestSessionMetrics(BaseTest):
    """Test cases for SessionMetrics using new infrastructure."""

    @pytest.mark.unit
    def test_metrics_initialization(self):
        """Test metrics initialization with base test patterns."""
        metrics = SessionMetrics()

        assert metrics.session_start_time is None  # Initially None
        assert metrics.total_recording_time == 0.0
        assert metrics.total_transcriptions == 0
        assert metrics.connection_errors == 0
        assert len(metrics.recording_segments) == 0

    @pytest.mark.unit
    def test_metrics_activity_tracking(self):
        """Test activity tracking functionality."""
        metrics = SessionMetrics()

        # Initially no activity
        assert metrics.session_start_time is None
        assert metrics.last_activity_time is None

        # Update activity
        metrics.update_activity()

        assert metrics.session_start_time is not None
        assert metrics.last_activity_time is not None
        assert metrics.session_start_time == metrics.last_activity_time

    @pytest.mark.unit
    def test_metrics_transcription_counting(self):
        """Test transcription counting functionality."""
        metrics = SessionMetrics()

        # Update transcription counts
        metrics.total_transcriptions = 5
        metrics.partial_transcriptions = 2
        metrics.final_transcriptions = 3

        assert metrics.total_transcriptions == 5
        assert metrics.partial_transcriptions == 2
        assert metrics.final_transcriptions == 3

    @pytest.mark.unit
    def test_metrics_error_tracking(self):
        """Test error counting functionality."""
        metrics = SessionMetrics()

        # Update error count
        metrics.connection_errors = 2

        assert metrics.connection_errors == 2


class TestEnhancedAudioSessionManager(BaseIntegrationTest):
    """Integration tests for EnhancedAudioSessionManager using new infrastructure."""

    @pytest.mark.integration
    def test_manager_initialization(self, reset_singletons):
        """Test manager initialization with singleton management."""
        manager = EnhancedAudioSessionManager()

        assert manager is not None
        assert manager.current_state == SessionState.IDLE
        assert manager._session_metrics is not None
        assert manager._transcription_buffer is not None
        assert len(manager.get_recording_segments()) == 0

    @pytest.mark.integration
    def test_singleton_behavior(self, reset_singletons):
        """Test singleton pattern behavior."""
        manager1 = get_enhanced_audio_session()
        manager2 = get_enhanced_audio_session()

        assert manager1 is manager2  # Same instance

    @pytest.mark.integration
    @patch("src.core.processor.AudioProcessor")
    @patch("threading.Thread")
    @patch("config.audio_config.get_config")
    def test_start_recording_success(
        self,
        mock_get_config,
        mock_thread,
        mock_audio_processor_class,
        mock_audio_processor,
        clean_enhanced_session_manager,
    ):
        """Test successful recording start with centralized mocks."""
        # Mock configuration
        mock_config = Mock()
        mock_config.get_transcription_config.return_value = {
            "region": "us-east-1",
            "language_code": "en-US",
        }
        mock_config.transcription_provider = "aws"
        mock_get_config.return_value = mock_config

        # Mock thread
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance

        # Mock AudioProcessor
        mock_audio_processor_class.return_value = mock_audio_processor

        success = clean_enhanced_session_manager.start_recording(device_index=0)

        assert success
        assert clean_enhanced_session_manager.current_state == SessionState.RECORDING
        assert clean_enhanced_session_manager._audio_processor is not None

    @pytest.mark.integration
    def test_stop_recording_success(
        self, clean_enhanced_session_manager, mock_audio_processor
    ):
        """Test successful recording stop with proper cleanup."""
        # Set up recording state
        clean_enhanced_session_manager._set_state(SessionState.RECORDING)
        clean_enhanced_session_manager._audio_processor = mock_audio_processor
        clean_enhanced_session_manager._background_thread = Mock()
        clean_enhanced_session_manager._background_thread.is_alive.return_value = False

        with patch("asyncio.run_coroutine_threadsafe") as mock_run_coroutine:
            mock_future = Mock()
            mock_future.result.return_value = None
            mock_run_coroutine.return_value = mock_future

            success = clean_enhanced_session_manager.stop_recording()

            assert success
            assert clean_enhanced_session_manager.current_state == SessionState.IDLE

    @pytest.mark.integration
    def test_transcription_processing(
        self, clean_enhanced_session_manager, transcription_result_factory
    ):
        """Test transcription result processing integration."""
        result = transcription_result_factory.create_final_result(
            "Integration test transcription"
        )

        clean_enhanced_session_manager._on_transcription_received(result)

        # Check buffer
        messages = clean_enhanced_session_manager._transcription_buffer.get_messages()
        assert len(messages) == 1
        assert "Integration test transcription" in messages[0]["content"]

        # Check metrics
        assert clean_enhanced_session_manager._session_metrics.total_transcriptions == 1


class TestFactoryFunction(BaseTest):
    """Test cases for factory function using new infrastructure."""

    @pytest.mark.unit
    def test_get_enhanced_audio_session(self, reset_singletons):
        """Test factory function returns singleton correctly."""
        session1 = get_enhanced_audio_session()
        session2 = get_enhanced_audio_session()

        assert session1 is not None
        assert session1 is session2
        assert isinstance(session1, EnhancedAudioSessionManager)


# Custom fixture for clean enhanced session manager
@pytest.fixture
def clean_enhanced_session_manager(reset_singletons):
    """Fresh EnhancedAudioSessionManager instance for testing."""
    manager = EnhancedAudioSessionManager()
    # State is already IDLE by default, cannot set directly due to property
    manager._audio_processor = None
    manager._background_thread = None

    yield manager

    # Cleanup
    if hasattr(manager, "_audio_processor") and manager._audio_processor:
        with contextlib.suppress(Exception):
            manager.stop_recording()
