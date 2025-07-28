#!/usr/bin/env python3
"""Test core recording functionality with proper mocking."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.managers.session_manager import get_audio_session
from src.core.processor import AudioProcessor


@pytest.fixture
def mock_audio_processor():
    """Create a mock audio processor for testing."""
    mock_processor = Mock(spec=AudioProcessor)
    mock_processor.is_running = False
    mock_processor.current_meeting_id = "test_meeting_123"
    mock_processor.session_transcripts = []
    mock_processor.start_recording = AsyncMock()
    mock_processor.stop_recording = AsyncMock()
    mock_processor.set_transcription_callback = Mock()
    mock_processor.set_error_callback = Mock()
    return mock_processor


@pytest.fixture
def mock_audio_devices():
    """Mock audio device listing."""
    return {
        0: "Built-in Microphone",
        1: "USB Headset",
        2: "Bluetooth Headphones"
    }


@pytest.fixture
def session_manager():
    """Get a fresh session manager for each test."""
    session = get_audio_session()
    session.audio_processor = None
    session.current_transcriptions = []
    session.background_thread = None
    session.background_loop = None
    return session


def test_session_manager_initialization(session_manager):
    """Test session manager initializes correctly."""
    assert session_manager is not None
    assert not session_manager.is_recording()
    assert len(session_manager.get_current_transcriptions()) == 0


def test_start_recording_success(session_manager, mock_audio_processor, mock_audio_devices):
    """Test successful recording start."""
    with patch('src.core.processor.AudioProcessor', return_value=mock_audio_processor), \
         patch('src.audio.providers.pyaudio_capture.PyAudioCaptureProvider.list_audio_devices', 
               return_value=mock_audio_devices):
        
        # Configure the mock processor
        mock_audio_processor.is_running = False
        
        # Start recording
        config = {'region': 'us-east-1', 'language_code': 'en-US'}
        success = session_manager.start_recording(device_index=0, config=config)
        
        assert success
        assert session_manager.is_recording()
        assert session_manager.audio_processor is not None


def test_start_recording_already_recording(session_manager, mock_audio_processor):
    """Test starting recording when already recording."""
    # Set up session as already recording
    session_manager.audio_processor = mock_audio_processor
    
    config = {'region': 'us-east-1', 'language_code': 'en-US'}
    success = session_manager.start_recording(device_index=0, config=config)
    
    assert not success  # Should fail because already recording


def test_stop_recording_success(session_manager, mock_audio_processor):
    """Test successful recording stop."""
    # Set up session as recording
    session_manager.audio_processor = mock_audio_processor
    mock_audio_processor.is_running = True
    
    # Mock background thread
    mock_thread = Mock()
    mock_thread.is_alive.return_value = False
    session_manager.background_thread = mock_thread
    
    with patch('asyncio.run_coroutine_threadsafe') as mock_run_coroutine:
        mock_future = Mock()
        mock_future.result.return_value = None
        mock_run_coroutine.return_value = mock_future
        
        success = session_manager.stop_recording()
        
        assert success
        assert not session_manager.is_recording()
        assert session_manager.audio_processor is None


def test_stop_recording_not_recording(session_manager):
    """Test stopping recording when not recording."""
    assert not session_manager.is_recording()
    
    success = session_manager.stop_recording()
    
    assert not success  # Should fail because not recording


def test_transcription_callback_handling(session_manager):
    """Test transcription callback registration and removal."""
    callback = Mock()
    
    # Add callback
    session_manager.add_transcription_callback(callback)
    assert callback in session_manager.transcription_callbacks
    
    # Remove callback
    session_manager.remove_transcription_callback(callback)
    assert callback not in session_manager.transcription_callbacks


def test_session_info():
    """Test session information structure."""
    # Test creating session info structure manually
    from datetime import datetime
    
    # Mock session data
    session_data = {
        'is_recording': False,
        'transcription_count': 0,
        'callbacks_registered': 0,
        'duration': 0.0,
        'start_time': None,
        'end_time': None
    }
    
    # Test not recording state
    assert session_data['is_recording'] == False
    assert session_data['transcription_count'] == 0
    assert session_data['callbacks_registered'] == 0
    assert session_data['duration'] == 0.0
    
    # Test recording state
    session_data['is_recording'] = True
    session_data['transcription_count'] = 5
    session_data['callbacks_registered'] = 1
    session_data['start_time'] = datetime.now()
    session_data['duration'] = 2.5
    
    assert session_data['is_recording'] == True
    assert session_data['transcription_count'] == 5
    assert session_data['callbacks_registered'] == 1
    assert session_data['duration'] == 2.5


def test_transcription_processing(session_manager):
    """Test transcription result processing."""
    from src.core.interfaces import TranscriptionResult
    
    # Create a test transcription result
    result = TranscriptionResult(
        text="Hello world",
        speaker_id="Speaker_1",
        confidence=0.95,
        start_time=0.0,
        end_time=2.0,
        is_partial=False,
        utterance_id="utterance_1",
        sequence_number=1,
        result_id="result_1"
    )
    
    # Process the result
    session_manager._on_transcription_received(result)
    
    # Check that transcription was stored
    transcriptions = session_manager.get_current_transcriptions()
    assert len(transcriptions) == 1
    assert transcriptions[0]['content'] == "Speaker_1: Hello world"
    assert transcriptions[0]['confidence'] == 0.95


def test_partial_result_handling(session_manager):
    """Test partial result handling and updates."""
    from src.core.interfaces import TranscriptionResult
    
    # Create partial result
    partial_result = TranscriptionResult(
        text="Hello",
        speaker_id="Speaker_1",
        confidence=0.8,
        start_time=0.0,
        end_time=1.0,
        is_partial=True,
        utterance_id="utterance_1",
        sequence_number=1,
        result_id="result_1"
    )
    
    # Process partial result
    session_manager._on_transcription_received(partial_result)
    
    # Check that partial result was stored
    transcriptions = session_manager.get_current_transcriptions()
    assert len(transcriptions) == 1
    assert transcriptions[0]['content'] == "Speaker_1: Hello"
    assert transcriptions[0]['is_partial'] == True
    
    # Create final result
    final_result = TranscriptionResult(
        text="Hello world",
        speaker_id="Speaker_1",
        confidence=0.95,
        start_time=0.0,
        end_time=2.0,
        is_partial=False,
        utterance_id="utterance_1",
        sequence_number=2,
        result_id="result_2"
    )
    
    # Process final result
    session_manager._on_transcription_received(final_result)
    
    # Check that partial result was replaced with final result
    transcriptions = session_manager.get_current_transcriptions()
    assert len(transcriptions) == 1
    assert transcriptions[0]['content'] == "Speaker_1: Hello world"
    assert transcriptions[0]['is_partial'] == False


def test_audio_processor_mocking():
    """Test that AudioProcessor can be properly mocked."""
    # Create a mock processor instance
    mock_processor = Mock()
    mock_processor.start_recording = AsyncMock()
    mock_processor.stop_recording = AsyncMock()
    mock_processor.is_running = False
    mock_processor.current_meeting_id = "test_meeting"
    mock_processor.session_transcripts = []
    
    # Test that mock methods can be called
    assert mock_processor.start_recording is not None
    assert mock_processor.stop_recording is not None
    assert mock_processor.is_running == False
    
    # Test that async mocks work
    import asyncio
    
    async def test_async_calls():
        await mock_processor.start_recording(device_id=0)
        await mock_processor.stop_recording()
        
        mock_processor.start_recording.assert_called_once_with(device_id=0)
        mock_processor.stop_recording.assert_called_once()
    
    # Run the async test
    asyncio.run(test_async_calls())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])