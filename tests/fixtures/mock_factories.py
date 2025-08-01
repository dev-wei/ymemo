"""Centralized mock creation factories for test suite.

This module provides factory functions for creating properly configured mock objects
that can be reused across all test files, reducing duplication and ensuring consistency.
"""

import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, List, Optional, Any

from src.core.processor import AudioProcessor
from src.core.interfaces import AudioConfig, TranscriptionResult
from src.managers.session_manager import AudioSessionManager


class MockAudioProcessorFactory:
    """Factory for creating AudioProcessor mocks with standard configurations."""
    
    @staticmethod
    def create_basic_mock() -> Mock:
        """Create a basic AudioProcessor mock with essential attributes."""
        mock_processor = Mock(spec=AudioProcessor)
        
        # Basic state attributes
        mock_processor.is_running = False
        mock_processor.current_meeting_id = "test_meeting_123"
        mock_processor.session_transcripts = []
        
        # Provider attributes
        mock_processor.capture_provider = Mock()
        mock_processor.capture_provider.__class__.__name__ = "PyAudioCaptureProvider"
        mock_processor.transcription_provider = Mock()
        
        # Async methods
        mock_processor.start_recording = AsyncMock()
        mock_processor.stop_recording = AsyncMock()
        
        # Callback methods
        mock_processor.set_transcription_callback = Mock()
        mock_processor.set_connection_health_callback = Mock()
        mock_processor.set_error_callback = Mock()
        
        return mock_processor
    
    @staticmethod
    def create_running_mock() -> Mock:
        """Create an AudioProcessor mock in running state."""
        mock_processor = MockAudioProcessorFactory.create_basic_mock()
        mock_processor.is_running = True
        return mock_processor
    
    @staticmethod
    def create_with_providers(capture_provider: Mock = None, transcription_provider: Mock = None) -> Mock:
        """Create AudioProcessor mock with custom providers."""
        mock_processor = MockAudioProcessorFactory.create_basic_mock()
        
        if capture_provider:
            mock_processor.capture_provider = capture_provider
        if transcription_provider:
            mock_processor.transcription_provider = transcription_provider
            
        return mock_processor
    
    @staticmethod
    def create_with_error_simulation() -> Mock:
        """Create AudioProcessor mock that simulates errors."""
        mock_processor = MockAudioProcessorFactory.create_basic_mock()
        
        # Configure methods to raise exceptions
        mock_processor.start_recording.side_effect = Exception("Simulated start error")
        mock_processor.stop_recording.side_effect = Exception("Simulated stop error")
        
        return mock_processor


class MockProviderFactory:
    """Factory for creating provider mocks (AWS, PyAudio, File)."""
    
    @staticmethod
    def create_pyaudio_provider_mock() -> Mock:
        """Create a comprehensive PyAudio provider mock."""
        mock_provider = Mock()
        mock_provider.__class__.__name__ = "PyAudioCaptureProvider"
        
        # State attributes
        mock_provider._is_active = False
        mock_provider._stop_event = Mock()
        mock_provider._stop_event.is_set.return_value = False
        mock_provider._stop_event.set = Mock()
        mock_provider.stream = None
        mock_provider._capture_thread = None
        
        # Async methods
        mock_provider.start_capture = AsyncMock()
        mock_provider.stop_capture = AsyncMock()
        mock_provider.get_audio_stream = AsyncMock()
        
        # Sync methods
        mock_provider.list_audio_devices = Mock(return_value={
            0: "Built-in Microphone",
            1: "USB Headset",
            2: "Bluetooth Device"
        })
        mock_provider.set_audio_callback = Mock()
        
        return mock_provider
    
    @staticmethod
    def create_aws_provider_mock() -> Mock:
        """Create a comprehensive AWS Transcribe provider mock."""
        mock_provider = Mock()
        mock_provider.__class__.__name__ = "AWSTranscribeProvider"
        
        # Configuration attributes
        mock_provider.region = "us-east-1"
        mock_provider.language_code = "en-US"
        mock_provider.profile_name = None
        
        # State attributes
        mock_provider.client = None
        mock_provider.stream = None
        mock_provider.result_queue = None
        mock_provider._current_event_loop = None
        mock_provider.is_connected = False
        
        # Async methods
        mock_provider.start_stream = AsyncMock()
        mock_provider.stop_stream = AsyncMock()
        mock_provider.send_audio = AsyncMock()
        mock_provider.get_transcription = AsyncMock()
        
        # Callback methods
        mock_provider.set_connection_health_callback = Mock()
        
        return mock_provider
    
    @staticmethod
    def create_file_provider_mock(file_path: str = "/tmp/test_audio.wav") -> Mock:
        """Create a File audio provider mock."""
        mock_provider = Mock()
        mock_provider.__class__.__name__ = "FileAudioCaptureProvider"
        
        # Configuration
        mock_provider.file_path = file_path
        
        # State attributes
        mock_provider._is_active = False
        mock_provider._stop_event = Mock()
        
        # Async methods
        mock_provider.start_capture = AsyncMock()
        mock_provider.stop_capture = AsyncMock()
        mock_provider.get_audio_stream = AsyncMock()
        
        # Sync methods
        mock_provider.list_audio_devices = Mock(return_value={0: "File Audio Source"})
        
        return mock_provider


class MockSessionManagerFactory:
    """Factory for creating SessionManager mocks with proper state."""
    
    @staticmethod
    def create_basic_mock() -> Mock:
        """Create a basic session manager mock."""
        mock_manager = Mock(spec=AudioSessionManager)
        
        # State attributes
        mock_manager._recording_active = False
        mock_manager.current_transcriptions = []
        mock_manager.background_thread = None
        mock_manager.background_loop = None
        mock_manager.audio_processor = MockAudioProcessorFactory.create_basic_mock()
        mock_manager.transcription_callbacks = []
        
        # Methods
        mock_manager.start_recording = Mock(return_value=True)
        mock_manager.stop_recording = Mock(return_value=True)
        mock_manager.is_recording = Mock(return_value=False)
        mock_manager.get_current_transcriptions = Mock(return_value=[])
        mock_manager.add_transcription_callback = Mock()
        mock_manager.remove_transcription_callback = Mock()
        
        return mock_manager
    
    @staticmethod
    def create_recording_mock() -> Mock:
        """Create a session manager mock in recording state."""
        mock_manager = MockSessionManagerFactory.create_basic_mock()
        mock_manager._recording_active = True
        mock_manager.is_recording.return_value = True
        mock_manager.audio_processor.is_running = True
        return mock_manager
    
    @staticmethod
    def create_with_transcriptions(transcriptions: List[Dict[str, Any]]) -> Mock:
        """Create session manager mock with existing transcriptions."""
        mock_manager = MockSessionManagerFactory.create_basic_mock()
        mock_manager.current_transcriptions = transcriptions
        mock_manager.get_current_transcriptions.return_value = transcriptions
        return mock_manager


class MockAudioConfigFactory:
    """Factory for creating AudioConfig instances with standard test values."""
    
    @staticmethod
    def create_default() -> AudioConfig:
        """Create default test AudioConfig."""
        return AudioConfig(
            sample_rate=16000,
            channels=1,
            chunk_size=1024,
            format='int16'
        )
    
    @staticmethod
    def create_high_quality() -> AudioConfig:
        """Create high-quality AudioConfig for performance tests."""
        return AudioConfig(
            sample_rate=44100,
            channels=2,
            chunk_size=2048,
            format='int24'
        )
    
    @staticmethod
    def create_low_quality() -> AudioConfig:
        """Create low-quality AudioConfig for basic tests."""
        return AudioConfig(
            sample_rate=8000,
            channels=1,
            chunk_size=512,
            format='int16'
        )


class MockTranscriptionResultFactory:
    """Factory for creating TranscriptionResult objects for testing."""
    
    @staticmethod
    def create_basic_result(text: str = "Test transcription", is_partial: bool = False) -> TranscriptionResult:
        """Create a basic TranscriptionResult."""
        return TranscriptionResult(
            text=text,
            speaker_id="Speaker1",
            confidence=0.95,
            start_time=1.0,
            end_time=2.0,
            is_partial=is_partial,
            utterance_id="utterance_001",
            sequence_number=1,
            result_id="result_001"
        )
    
    @staticmethod
    def create_partial_result(text: str = "Partial text") -> TranscriptionResult:
        """Create a partial TranscriptionResult."""
        return MockTranscriptionResultFactory.create_basic_result(text=text, is_partial=True)
    
    @staticmethod
    def create_final_result(text: str = "Final complete text") -> TranscriptionResult:
        """Create a final TranscriptionResult."""
        return MockTranscriptionResultFactory.create_basic_result(text=text, is_partial=False)
    
    @staticmethod
    def create_sequence(utterance_id: str, texts: List[str], final_text: str) -> List[TranscriptionResult]:
        """Create a sequence of partial results followed by final result."""
        results = []
        
        # Add partial results
        for i, text in enumerate(texts):
            result = TranscriptionResult(
                text=text,
                speaker_id="Speaker1",
                confidence=0.8 + i * 0.05,  # Increasing confidence
                start_time=1.0,
                end_time=2.0 + i * 0.1,
                is_partial=True,
                utterance_id=utterance_id,
                sequence_number=i + 1,
                result_id=f"partial_{i+1}"
            )
            results.append(result)
        
        # Add final result
        final_result = TranscriptionResult(
            text=final_text,
            speaker_id="Speaker1",
            confidence=0.98,
            start_time=1.0,
            end_time=2.0 + len(texts) * 0.1,
            is_partial=False,
            utterance_id=utterance_id,
            sequence_number=len(texts) + 1,
            result_id="final_result"
        )
        results.append(final_result)
        
        return results


class MockThreadFactory:
    """Factory for creating thread mocks with proper behavior."""
    
    @staticmethod
    def create_basic_mock() -> Mock:
        """Create a basic thread mock."""
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True
        mock_thread.daemon = True
        mock_thread.name = "MockThread"
        
        # Mock join that simulates proper termination
        def mock_join(timeout=None):
            mock_thread.is_alive.return_value = False
        
        mock_thread.join.side_effect = mock_join
        return mock_thread
    
    @staticmethod
    def create_hanging_mock() -> Mock:
        """Create a thread mock that hangs (doesn't terminate)."""
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True  # Always alive
        mock_thread.daemon = True
        mock_thread.name = "HangingMockThread"
        mock_thread.join = Mock()  # join doesn't change is_alive
        return mock_thread


class MockPyAudioFactory:
    """Factory for creating PyAudio mocks."""
    
    @staticmethod
    def create_full_mock():
        """Create comprehensive PyAudio mock with all components."""
        mock_pyaudio_class = Mock()
        mock_pyaudio = Mock()
        mock_stream = Mock()
        
        # Configure PyAudio instance
        mock_pyaudio_class.return_value = mock_pyaudio
        mock_pyaudio.get_device_count.return_value = 3
        mock_pyaudio.get_device_info_by_index.return_value = {
            'name': 'Mock Audio Device',
            'maxInputChannels': 2,
            'maxOutputChannels': 0,
            'defaultSampleRate': 44100.0,
            'index': 0
        }
        mock_pyaudio.get_default_input_device_info.return_value = {
            'index': 0,
            'name': 'Mock Default Device'
        }
        
        # Configure stream
        mock_pyaudio.open.return_value = mock_stream
        mock_stream.read.return_value = b'\x00' * 2048  # Mock audio data
        mock_stream.stop_stream = Mock()
        mock_stream.close = Mock()
        mock_stream.is_active.return_value = True
        
        return mock_pyaudio_class, mock_pyaudio, mock_stream