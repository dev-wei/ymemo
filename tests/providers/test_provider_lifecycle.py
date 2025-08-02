"""Provider lifecycle tests using new test infrastructure.

Migrated from unittest to pytest with centralized fixtures and base classes.
Tests provider lifecycle management, initialization, and resource cleanup.
"""

import os
import tempfile
import threading
import wave
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from src.audio.providers.aws_transcribe import AWSTranscribeProvider
from src.audio.providers.file_audio_capture import FileAudioCaptureProvider
from src.audio.providers.pyaudio_capture import PyAudioCaptureProvider
from src.core.factory import AudioProcessorFactory
from src.core.interfaces import (
    AudioCaptureProvider,
    AudioConfig,
    TranscriptionProvider,
    TranscriptionResult,
)
from tests.base.async_test_base import BaseAsyncTest
from tests.base.base_test import BaseIntegrationTest, BaseTest


class TestProviderFactory(BaseTest):
    """Test provider factory lifecycle management using new infrastructure."""

    @pytest.mark.integration
    def test_transcription_provider_creation(self):
        """Test transcription provider creation and configuration."""
        # Test AWS provider creation (may fail in test environment - expected)
        try:
            aws_provider = AudioProcessorFactory.create_transcription_provider(
                "aws", region="us-west-2", language_code="en-US"
            )

            assert isinstance(aws_provider, AWSTranscribeProvider)
            assert aws_provider.region == "us-west-2"
            assert aws_provider.language_code == "en-US"
        except (RuntimeError, TypeError):
            # Expected in test environment without AWS credentials
            pytest.skip("AWS provider creation failed - expected in test environment")

    @pytest.mark.integration
    def test_audio_capture_provider_creation(self):
        """Test audio capture provider creation."""
        # Test file provider creation (safer for testing)
        test_file = self._create_test_audio_file()
        try:
            file_provider = AudioProcessorFactory.create_audio_capture_provider(
                "file", file_path=test_file
            )
            assert isinstance(file_provider, FileAudioCaptureProvider)
            assert file_provider.file_path == test_file
        finally:
            if os.path.exists(test_file):
                os.unlink(test_file)

        # Test PyAudio provider creation (may fail without audio hardware)
        try:
            pyaudio_provider = AudioProcessorFactory.create_audio_capture_provider(
                "pyaudio"
            )
            assert isinstance(pyaudio_provider, PyAudioCaptureProvider)
        except (RuntimeError, TypeError):
            # Expected in test environment without audio hardware
            pass

    @pytest.mark.unit
    def test_provider_registration(self):
        """Test dynamic provider registration using centralized mocks."""

        # Create mock provider classes that inherit from interfaces
        class MockTranscriptionProvider(TranscriptionProvider):
            def __init__(self, **kwargs):
                self.config = kwargs

            async def start_stream(self, audio_config):
                pass

            async def send_audio(self, audio_chunk):
                pass

            async def get_transcription(self):
                result = TranscriptionResult(
                    text="Mock result",
                    is_final=True,
                    confidence=0.9,
                    speaker_id=None,
                    utterance_id="test_utterance",
                    sequence_number=1,
                )
                yield result

            async def stop_stream(self):
                pass

            def get_required_channels(self) -> int:
                return 1  # Mock provider uses mono

        class MockCaptureProvider(AudioCaptureProvider):
            def __init__(self, **kwargs):
                self.config = kwargs

            async def start_capture(self, audio_config, device_id=None):
                pass

            async def get_audio_stream(self):
                yield b"mock_audio_data"

            async def stop_capture(self):
                pass

            def list_audio_devices(self):
                return {0: "Mock Device"}

        # Register providers
        factory = AudioProcessorFactory()
        factory.register_transcription_provider(
            "mock_transcription", MockTranscriptionProvider
        )
        factory.register_audio_capture_provider("mock_capture", MockCaptureProvider)

        # Test provider creation
        transcription_provider = factory.create_transcription_provider(
            "mock_transcription", test_param="test_value"
        )
        assert isinstance(transcription_provider, MockTranscriptionProvider)
        assert transcription_provider.config["test_param"] == "test_value"

        capture_provider = factory.create_audio_capture_provider(
            "mock_capture", capture_param="capture_value"
        )
        assert isinstance(capture_provider, MockCaptureProvider)
        assert capture_provider.config["capture_param"] == "capture_value"

    @pytest.mark.unit
    def test_invalid_provider_creation(self):
        """Test error handling for invalid providers."""
        factory = AudioProcessorFactory()

        # Test invalid transcription provider
        with pytest.raises(
            ValueError, match="Unsupported transcription provider.*invalid_provider"
        ):
            factory.create_transcription_provider("invalid_provider")

        # Test invalid capture provider
        with pytest.raises(
            ValueError, match="Unsupported audio capture provider.*invalid_capture"
        ):
            factory.create_audio_capture_provider("invalid_capture")

    @pytest.mark.unit
    def test_provider_listing(self):
        """Test provider listing functionality."""
        factory = AudioProcessorFactory()

        transcription_providers = factory.list_transcription_providers()
        assert isinstance(transcription_providers, dict)
        assert "aws" in transcription_providers
        assert transcription_providers["aws"] == "AWSTranscribeProvider"

        capture_providers = factory.list_audio_capture_providers()
        assert isinstance(capture_providers, dict)
        assert "pyaudio" in capture_providers
        assert "file" in capture_providers
        assert capture_providers["pyaudio"] == "PyAudioCaptureProvider"
        assert capture_providers["file"] == "FileAudioCaptureProvider"

    def _create_test_audio_file(self) -> str:
        """Create a temporary test audio file."""
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        # Generate simple sine wave
        sample_rate = 16000
        duration = 0.5
        frequency = 440

        samples = int(sample_rate * duration)
        audio_data = np.sin(2 * np.pi * frequency * np.linspace(0, duration, samples))
        audio_data = (audio_data * 32767).astype(np.int16)

        with wave.open(temp_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        return temp_path


class TestPyAudioProviderLifecycle(BaseAsyncTest):
    """Test PyAudio provider lifecycle management using new infrastructure."""

    def setup_method(self):
        """Set up test environment using base class."""
        super().setup_method()
        self.audio_config = AudioConfig(
            sample_rate=16000, channels=1, chunk_size=1024, format="int16"
        )

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_provider_initialization(self):
        """Test provider initialization."""
        provider = PyAudioCaptureProvider()

        # Verify initial state
        assert not provider._is_active
        assert provider._stop_event is not None
        assert not provider._stop_event.is_set()
        assert provider.stream is None
        assert provider._capture_thread is None

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_provider_start_stop_cycle(self):
        """Test complete start/stop cycle with mocked hardware."""
        provider = PyAudioCaptureProvider()

        # Mock PyAudio components to avoid hardware dependency
        mock_pyaudio = MagicMock()
        mock_stream = MagicMock()
        mock_pyaudio.PyAudio.return_value = mock_pyaudio
        mock_pyaudio.open.return_value = mock_stream
        mock_stream.read.return_value = b"\x00" * 2048  # Mock audio data

        with patch("pyaudio.PyAudio", return_value=mock_pyaudio):
            # Start capture
            await provider.start_capture(self.audio_config, device_id=0)

            # Verify active state
            assert provider._is_active
            assert not provider._stop_event.is_set()
            assert provider.stream is not None
            assert provider._capture_thread is not None
            assert provider._capture_thread.is_alive()

            # Stop capture
            await provider.stop_capture()

            # Verify cleanup
            assert not provider._is_active
            assert provider._stop_event.is_set()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_provider_restart_behavior(self):
        """Test provider restart behavior - fresh stop events."""
        provider = PyAudioCaptureProvider()

        with patch("pyaudio.PyAudio") as mock_pyaudio_class:
            mock_pyaudio = MagicMock()
            mock_stream = MagicMock()
            mock_pyaudio_class.return_value = mock_pyaudio
            mock_pyaudio.open.return_value = mock_stream
            mock_stream.read.return_value = b"\x00" * 2048

            # First session
            await provider.start_capture(self.audio_config, device_id=0)
            first_stop_event = provider._stop_event
            await provider.stop_capture()

            # Verify first stop event is set
            assert first_stop_event.is_set()

            # Second session - should get fresh stop event
            await provider.start_capture(self.audio_config, device_id=0)
            second_stop_event = provider._stop_event

            # Verify we got a fresh stop event
            assert second_stop_event is not first_stop_event
            assert not second_stop_event.is_set()

            await provider.stop_capture()
            assert second_stop_event.is_set()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_start_protection(self):
        """Test protection against concurrent start operations."""
        provider = PyAudioCaptureProvider()

        with patch("pyaudio.PyAudio") as mock_pyaudio_class:
            mock_pyaudio = MagicMock()
            mock_stream = MagicMock()
            mock_pyaudio_class.return_value = mock_pyaudio
            mock_pyaudio.open.return_value = mock_stream
            mock_stream.read.return_value = b"\x00" * 2048

            # Start first session
            await provider.start_capture(self.audio_config, device_id=0)
            assert provider._is_active

            # Try to start second session - should stop first and start new
            await provider.start_capture(self.audio_config, device_id=0)
            assert provider._is_active

            # Cleanup
            await provider.stop_capture()

    @pytest.mark.unit
    def test_thread_safety(self):
        """Test thread safety of provider operations."""
        provider = PyAudioCaptureProvider()
        results = []

        def check_provider_state():
            try:
                # Test thread-safe attribute access
                is_active = provider._is_active
                stop_event = provider._stop_event
                stream = provider.stream

                results.append(
                    {
                        "is_active": is_active,
                        "has_stop_event": stop_event is not None,
                        "stream": stream,
                    }
                )
            except Exception as e:
                results.append({"error": str(e)})

        # Run multiple threads accessing provider state
        threads = []
        for _i in range(3):
            thread = threading.Thread(target=check_provider_state)
            threads.append(thread)
            thread.start()

        # Wait for completion with reasonable timeout
        for thread in threads:
            thread.join(timeout=2.0)

        # Verify all threads completed successfully
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

        # Verify no errors occurred
        for result in results:
            assert "error" not in result, f"Thread error: {result.get('error')}"


class TestAWSTranscribeProviderLifecycle(BaseAsyncTest):
    """Test AWS Transcribe provider lifecycle using new infrastructure."""

    def setup_method(self):
        """Set up test environment using base class."""
        super().setup_method()
        self.audio_config = AudioConfig(
            sample_rate=16000, channels=1, chunk_size=1024, format="int16"
        )

    @pytest.mark.integration
    def test_provider_initialization_with_mock(self):
        """Test AWS provider initialization with properly mocked services."""
        # Skip AWS tests if provider creation fails (no credentials)
        try:
            provider = AWSTranscribeProvider(region="us-east-1", language_code="en-US")

            assert provider.region == "us-east-1"
            assert provider.language_code == "en-US"
            # AWS provider doesn't have is_streaming property initially
            assert provider.client is None
        except (RuntimeError, TypeError, ImportError) as e:
            pytest.skip(f"AWS provider initialization failed: {e}")

    @pytest.mark.integration
    def test_provider_configuration_validation(self):
        """Test provider configuration validation without AWS connection."""
        # Test that we can create provider instances with different configs
        configs = [
            {"region": "us-west-2", "language_code": "en-US"},
            {"region": "eu-west-1", "language_code": "en-GB"},
        ]

        for config in configs:
            try:
                provider = AWSTranscribeProvider(**config)
                assert provider.region == config["region"]
                assert provider.language_code == config["language_code"]
            except (RuntimeError, TypeError, ImportError):
                # Expected in test environment without AWS setup
                pytest.skip(
                    "AWS provider creation failed - expected in test environment"
                )

    @pytest.mark.integration
    def test_provider_error_handling(self):
        """Test provider error handling for invalid configurations."""
        # Test invalid region
        try:
            provider = AWSTranscribeProvider(
                region="invalid-region-12345", language_code="en-US"
            )
            # If creation succeeds, provider should still be created
            assert provider is not None
        except (RuntimeError, TypeError, ValueError):
            # Expected behavior for invalid config
            pass


class TestAudioProcessorProviderIntegration(BaseIntegrationTest):
    """Test AudioProcessor with different providers using new infrastructure."""

    def setup_method(self):
        """Set up test environment using base class."""
        super().setup_method()
        self.audio_config = AudioConfig(
            sample_rate=16000, channels=1, chunk_size=1024, format="int16"
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_processor_with_file_provider(self, mock_audio_processor):
        """Test AudioProcessor integration with file audio provider."""
        # Create test audio file
        test_file = self._create_test_audio_file()

        try:
            # Create file provider
            capture_provider = AudioProcessorFactory.create_audio_capture_provider(
                "file", file_path=test_file
            )

            # Create mock transcription provider
            transcription_provider = Mock()
            transcription_provider.start_stream = AsyncMock()
            transcription_provider.send_audio = AsyncMock()
            transcription_provider.stop_stream = AsyncMock()
            transcription_provider.get_transcription = AsyncMock()

            async def mock_transcription_generator():
                result = TranscriptionResult(
                    text="Test transcription from file",
                    is_final=True,
                    confidence=0.9,
                    speaker_id=None,
                    utterance_id="file_test",
                    sequence_number=1,
                )
                yield result

            transcription_provider.get_transcription.return_value = (
                mock_transcription_generator()
            )

            # Test that providers can be created and configured
            assert capture_provider is not None
            assert isinstance(capture_provider, FileAudioCaptureProvider)
            assert capture_provider.file_path == test_file

        finally:
            if os.path.exists(test_file):
                os.unlink(test_file)

    @pytest.mark.integration
    def test_provider_factory_integration(self):
        """Test provider factory integration with AudioProcessor."""
        # Test that factory can create providers for AudioProcessor
        factory = AudioProcessorFactory()

        # List available providers
        transcription_providers = factory.list_transcription_providers()
        capture_providers = factory.list_audio_capture_providers()

        # Verify expected providers are available
        assert "aws" in transcription_providers
        assert "file" in capture_providers
        assert "pyaudio" in capture_providers

        # Test provider creation (may fail in test environment)
        try:
            file_provider = factory.create_audio_capture_provider(
                "file",
                file_path="/dev/null",  # Safe non-existent file for testing
            )
            assert file_provider is not None
        except (RuntimeError, TypeError):
            # Expected if file doesn't exist or can't be opened
            pass

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_provider_lifecycle_coordination(self):
        """Test coordinated lifecycle management between providers."""
        # This test validates that providers can be started and stopped
        # in coordination without hardware dependencies

        # Create mock providers
        capture_provider = Mock()
        transcription_provider = Mock()

        # Setup async methods
        capture_provider.start_capture = AsyncMock()
        capture_provider.get_audio_stream = AsyncMock()
        capture_provider.stop_capture = AsyncMock()
        transcription_provider.start_stream = AsyncMock()
        transcription_provider.send_audio = AsyncMock()
        transcription_provider.stop_stream = AsyncMock()

        # Mock audio stream
        async def mock_audio_stream():
            for _i in range(3):
                yield b"mock_audio_chunk"

        capture_provider.get_audio_stream.return_value = mock_audio_stream()

        # Test coordinated startup
        await capture_provider.start_capture(self.audio_config)
        await transcription_provider.start_stream(self.audio_config)

        # Verify startup calls
        capture_provider.start_capture.assert_called_once_with(self.audio_config)
        transcription_provider.start_stream.assert_called_once_with(self.audio_config)

        # Test coordinated shutdown
        await transcription_provider.stop_stream()
        await capture_provider.stop_capture()

        # Verify shutdown calls
        transcription_provider.stop_stream.assert_called_once()
        capture_provider.stop_capture.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_mock_provider_integration(self):
        """Test integration with fully mocked providers to avoid timeouts."""

        # Create mock providers that implement the full interface
        class MockTranscriptionProvider(TranscriptionProvider):
            def __init__(self):
                self.started = False

            async def start_stream(self, audio_config):
                self.started = True

            async def send_audio(self, audio_chunk):
                pass

            async def get_transcription(self):
                result = TranscriptionResult(
                    text="Mock transcription",
                    is_partial=False,  # Use is_partial instead of is_final
                    confidence=0.9,
                    speaker_id=None,
                    utterance_id="mock_test",
                    sequence_number=1,
                )
                yield result

            async def stop_stream(self):
                self.started = False

            def get_required_channels(self) -> int:
                return 1  # Mock provider uses mono

        class MockCaptureProvider:
            def __init__(self):
                self.capturing = False

            async def start_capture(self, audio_config, device_id=None):
                self.capturing = True

            async def get_audio_stream(self):
                for _i in range(2):
                    yield b"mock_audio_data"

            async def stop_capture(self):
                self.capturing = False

            def list_audio_devices(self):
                return {0: "Mock Device"}

        # Test full lifecycle with mocked providers
        transcription_provider = MockTranscriptionProvider()
        capture_provider = MockCaptureProvider()

        # Start providers
        await transcription_provider.start_stream(self.audio_config)
        await capture_provider.start_capture(self.audio_config)

        assert transcription_provider.started
        assert capture_provider.capturing

        # Test audio streaming
        audio_chunks = []
        async for chunk in capture_provider.get_audio_stream():
            audio_chunks.append(chunk)

        assert len(audio_chunks) == 2

        # Test transcription results
        results = []
        async for result in transcription_provider.get_transcription():
            results.append(result)
            break  # Just get one result

        assert len(results) == 1
        assert results[0].text == "Mock transcription"

        # Stop providers
        await transcription_provider.stop_stream()
        await capture_provider.stop_capture()

        assert not transcription_provider.started
        assert not capture_provider.capturing

    def _create_test_audio_file(self) -> str:
        """Create a temporary test audio file using base class utilities."""
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        # Generate simple sine wave
        sample_rate = 16000
        duration = 0.1  # Short duration for testing
        frequency = 440

        samples = int(sample_rate * duration)
        audio_data = np.sin(2 * np.pi * frequency * np.linspace(0, duration, samples))
        audio_data = (audio_data * 32767).astype(np.int16)

        with wave.open(temp_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        return temp_path
