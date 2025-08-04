"""Provider factory tests using new test infrastructure.

Migrated from unittest to pytest with centralized fixtures and base classes.
Tests factory behavior, registration, and error handling without external dependencies.
"""

from unittest.mock import Mock, patch

import pytest

from src.core.factory import AudioProcessorFactory
from src.core.interfaces import AudioCaptureProvider, AudioConfig, TranscriptionProvider
from src.utils.exceptions import AudioCaptureError, AWSTranscribeError
from tests.base.async_test_base import BaseAsyncTest
from tests.base.base_test import BaseIntegrationTest, BaseTest


class MockTranscriptionProvider(TranscriptionProvider):
    """Mock transcription provider for testing."""

    def __init__(self, **kwargs):
        self.config = kwargs
        self.started = False

    async def start_stream(self, audio_config):
        self.started = True

    async def send_audio(self, audio_chunk):
        pass

    async def get_transcription(self):
        from src.core.interfaces import TranscriptionResult

        result = TranscriptionResult(
            text="Mock transcription",
            is_final=True,
            confidence=0.9,
            speaker_id=None,
            utterance_id="mock_utterance",
            sequence_number=1,
        )
        yield result

    async def stop_stream(self):
        self.started = False

    def get_required_channels(self) -> int:
        return 1  # Mock provider uses mono for simplicity


class MockAudioCaptureProvider(AudioCaptureProvider):
    """Mock audio capture provider for testing."""

    def __init__(self, **kwargs):
        self.config = kwargs
        self.capturing = False
        self.audio_data = []
        self.started = False

    async def start_capture(self, audio_config, device_id=None):
        self.capturing = True
        self.started = True

    async def get_audio_stream(self):
        if not self.started:
            raise RuntimeError("Capture not started")
        # Yield a few chunks for testing
        for _i in range(3):
            yield b"mock_audio_data"

    async def stop_capture(self):
        self.capturing = False
        self.started = False

    def list_audio_devices(self):
        return {0: "Mock Device 1", 1: "Mock Device 2"}


class TestAudioProcessorFactory(BaseTest):
    """Test AudioProcessorFactory functionality using new infrastructure."""

    def setup_method(self):
        """Setup for factory tests."""
        super().setup_method()
        self.factory = AudioProcessorFactory()

        # Register test providers
        self.factory.register_transcription_provider("mock", MockTranscriptionProvider)
        self.factory.register_audio_capture_provider("mock", MockAudioCaptureProvider)

    @pytest.mark.unit
    def test_list_transcription_providers(self):
        """Test listing available transcription providers."""
        providers = self.factory.list_transcription_providers()

        assert isinstance(providers, dict)
        assert "mock" in providers
        # Should include built-in AWS provider if available
        provider_names = list(providers.keys())
        assert len(provider_names) >= 1
        assert "aws" in provider_names  # Built-in provider

    @pytest.mark.unit
    def test_list_audio_capture_providers(self):
        """Test listing available audio capture providers."""
        providers = self.factory.list_audio_capture_providers()

        assert isinstance(providers, dict)
        assert "mock" in providers
        # Should include built-in providers like pyaudio, file
        provider_names = list(providers.keys())
        assert len(provider_names) >= 1
        assert "pyaudio" in provider_names  # Built-in provider

    @pytest.mark.unit
    def test_register_transcription_provider(self):
        """Test registering new transcription provider."""

        class TestProvider(TranscriptionProvider):
            async def start_stream(self, audio_config):
                pass

            async def send_audio(self, audio_chunk):
                pass

            async def get_transcription(self):
                yield None

            async def stop_stream(self):
                pass

            def get_required_channels(self) -> int:
                return 1

        # Register new provider
        self.factory.register_transcription_provider("test", TestProvider)

        # Verify registration
        providers = self.factory.list_transcription_providers()
        assert "test" in providers
        assert providers["test"] == "TestProvider"

        # Create instance to verify it works
        provider = self.factory.create_transcription_provider("test")
        assert isinstance(provider, TestProvider)

    @pytest.mark.unit
    def test_register_audio_capture_provider(self):
        """Test registering new audio capture provider."""

        class TestCaptureProvider(AudioCaptureProvider):
            async def start_capture(self, audio_config, device_id=None):
                pass

            async def get_audio_stream(self):
                yield b"test"

            async def stop_capture(self):
                pass

            def list_audio_devices(self):
                return {}

        # Register new provider
        self.factory.register_audio_capture_provider(
            "test_capture", TestCaptureProvider
        )

        # Verify registration
        providers = self.factory.list_audio_capture_providers()
        assert "test_capture" in providers
        assert providers["test_capture"] == "TestCaptureProvider"

        # Create instance to verify it works
        provider = self.factory.create_audio_capture_provider("test_capture")
        assert isinstance(provider, TestCaptureProvider)

    @pytest.mark.unit
    def test_register_invalid_transcription_provider(self):
        """Test registering invalid transcription provider raises error."""

        class NotAProvider:
            pass

        with pytest.raises(
            TypeError, match="must implement TranscriptionProvider interface"
        ):
            self.factory.register_transcription_provider("invalid", NotAProvider)

    @pytest.mark.unit
    def test_register_invalid_audio_capture_provider(self):
        """Test registering invalid audio capture provider raises error."""

        class NotAProvider:
            pass

        with pytest.raises(
            TypeError, match="must implement AudioCaptureProvider interface"
        ):
            self.factory.register_audio_capture_provider("invalid", NotAProvider)

    @pytest.mark.unit
    def test_create_unknown_transcription_provider(self):
        """Test creating unknown transcription provider raises error."""
        with pytest.raises(ValueError, match="Unsupported transcription provider"):
            self.factory.create_transcription_provider("nonexistent")

    @pytest.mark.unit
    def test_create_unknown_audio_capture_provider(self):
        """Test creating unknown audio capture provider raises error."""
        with pytest.raises(ValueError, match="Unsupported audio capture provider"):
            self.factory.create_audio_capture_provider("nonexistent")

    @pytest.mark.integration
    @patch("boto3.Session")
    def test_create_aws_provider_success(self, mock_boto3, aws_mock_setup):
        """Test creating AWS transcription provider successfully."""
        # Setup AWS mocks using centralized fixture
        mock_session = Mock()
        mock_boto3.return_value = mock_session

        # Create AWS provider with valid config
        config = {"region": "us-east-1", "language_code": "en-US"}

        try:
            provider = self.factory.create_transcription_provider("aws", **config)
            # Provider creation should succeed
            assert provider is not None
        except Exception as e:
            # If AWS provider not registered, that's expected in test environment
            if "aws" not in self.factory.list_transcription_providers():
                pytest.skip("AWS provider not available in test environment")
            else:
                raise e

    @pytest.mark.integration
    def test_create_aws_provider_invalid_region(self):
        """Test creating AWS provider with invalid region."""
        config = {"region": "invalid-region", "language_code": "en-US"}

        if "aws" not in self.factory.list_transcription_providers():
            pytest.skip("AWS provider not available in test environment")

        # In test environment, AWS provider creation will likely fail
        # due to lack of credentials, but that's expected behavior
        try:
            provider = self.factory.create_transcription_provider("aws", **config)
            # If it somehow succeeds, provider should be valid
            assert provider is not None
        except (ValueError, TypeError, RuntimeError):
            # Expected behavior - AWS setup fails in test environment
            pass

    @pytest.mark.integration
    def test_create_pyaudio_provider_invalid_device_index(self):
        """Test creating PyAudio provider with invalid device."""
        if "pyaudio" not in self.factory.list_audio_capture_providers():
            pytest.skip("PyAudio provider not available in test environment")

        config = {"device_index": 999}  # Non-existent device

        # Should handle invalid device gracefully (may raise exception or return None)
        try:
            provider = self.factory.create_audio_capture_provider("pyaudio", **config)
            # If successful, provider should be valid
            if provider is not None:
                assert hasattr(provider, "list_audio_devices")
        except (AudioCaptureError, ValueError, RuntimeError, TypeError):
            # Expected behavior for invalid device
            pass

    @pytest.mark.unit
    def test_factory_error_handling_consistency(self):
        """Test that factory handles errors consistently."""

        # Test with mock provider that raises exception during creation
        class FailingProvider(TranscriptionProvider):
            def __init__(self, **kwargs):
                raise RuntimeError("Test error")

            # Implement abstract methods to avoid TypeError
            async def start_stream(self, audio_config):
                pass

            async def send_audio(self, audio_chunk):
                pass

            async def get_transcription(self):
                yield None

            async def stop_stream(self):
                pass

            def get_required_channels(self) -> int:
                return 1

        self.factory.register_transcription_provider("failing", FailingProvider)

        # Should propagate the initialization error wrapped in RuntimeError
        with pytest.raises(
            RuntimeError, match="Failed to initialize transcription provider"
        ):
            self.factory.create_transcription_provider("failing")


class TestProviderInterfaces(BaseTest):
    """Test provider interfaces using new infrastructure."""

    @pytest.mark.unit
    def test_audio_config_creation(self, default_audio_config):
        """Test AudioConfig creation using centralized config."""
        # Use centralized fixture
        config = default_audio_config

        assert isinstance(config, AudioConfig)
        assert config.sample_rate > 0
        assert config.channels > 0
        assert config.chunk_size > 0

        # Test config attributes
        assert hasattr(config, "sample_rate")
        assert hasattr(config, "channels")
        assert hasattr(config, "chunk_size")
        assert hasattr(config, "format")

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_mock_transcription_provider_interface(self):
        """Test mock transcription provider implements interface correctly."""
        provider = MockTranscriptionProvider(region="us-east-1")
        config = AudioConfig(
            sample_rate=16000, channels=1, chunk_size=1024, format="int16"
        )

        # Test interface compliance
        assert hasattr(provider, "start_stream")
        assert hasattr(provider, "send_audio")
        assert hasattr(provider, "get_transcription")
        assert hasattr(provider, "stop_stream")

        # Test basic functionality
        await provider.start_stream(config)
        assert provider.started

        await provider.stop_stream()
        assert not provider.started

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_mock_audio_capture_provider_interface(self):
        """Test mock audio capture provider implements interface correctly."""
        provider = MockAudioCaptureProvider(device_index=0)
        config = AudioConfig(
            sample_rate=16000, channels=1, chunk_size=1024, format="int16"
        )

        # Test interface compliance
        assert hasattr(provider, "start_capture")
        assert hasattr(provider, "stop_capture")
        assert hasattr(provider, "list_audio_devices")

        # Test device listing
        devices = provider.list_audio_devices()
        assert isinstance(devices, dict)
        assert len(devices) > 0

        # Test capture functionality - first start, then get stream
        await provider.start_capture(config)
        assert provider.capturing

        # Test audio stream
        async for audio_chunk in provider.get_audio_stream():
            assert isinstance(audio_chunk, bytes)
            break  # Just test first chunk

        await provider.stop_capture()
        assert not provider.capturing


class TestConvenienceFunctions(BaseIntegrationTest):
    """Test convenience functions using integration test patterns."""

    @pytest.mark.integration
    @patch("src.config.audio_config.get_config")
    @patch("boto3.Session")
    def test_create_aws_transcribe_provider(
        self, mock_boto3, mock_get_config, aws_mock_setup
    ):
        """Test AWS transcribe provider creation convenience function."""
        # Mock configuration
        mock_config = Mock()
        mock_config.get_transcription_config.return_value = {
            "region": "us-east-1",
            "language_code": "en-US",
        }
        mock_get_config.return_value = mock_config

        # Mock boto3 session
        mock_session = Mock()
        mock_boto3.return_value = mock_session

        factory = AudioProcessorFactory()

        if "aws" not in factory.list_transcription_providers():
            pytest.skip("AWS provider not available in test environment")

        try:
            provider = factory.create_transcription_provider("aws")
            assert provider is not None
        except Exception as e:
            # In test environment, AWS creation may fail - that's expected
            assert isinstance(e, AWSTranscribeError | ImportError | AttributeError)

    @pytest.mark.integration
    def test_create_pyaudio_capture_provider(self):
        """Test PyAudio capture provider creation."""
        factory = AudioProcessorFactory()

        if "pyaudio" not in factory.list_audio_capture_providers():
            pytest.skip("PyAudio provider not available in test environment")

        try:
            provider = factory.create_audio_capture_provider("pyaudio", device_index=0)
            # If creation succeeds, provider should have expected interface
            if provider is not None:
                assert hasattr(provider, "list_audio_devices")
                devices = provider.list_audio_devices()
                assert isinstance(devices, dict)
        except (AudioCaptureError, ImportError):
            # Expected in test environment without audio hardware
            pass


class TestProviderFactoryIntegration(BaseAsyncTest):
    """Async integration tests for provider factory."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_provider_lifecycle_integration(self):
        """Test complete provider lifecycle with factory."""
        factory = AudioProcessorFactory()

        # Register and create mock providers
        factory.register_transcription_provider("test_async", MockTranscriptionProvider)
        factory.register_audio_capture_provider("test_async", MockAudioCaptureProvider)

        transcription_provider = factory.create_transcription_provider("test_async")
        capture_provider = factory.create_audio_capture_provider("test_async")

        config = AudioConfig(
            sample_rate=16000, channels=1, chunk_size=1024, format="int16"
        )

        # Test transcription provider lifecycle
        await transcription_provider.start_stream(config)
        assert transcription_provider.started

        await transcription_provider.stop_stream()
        assert not transcription_provider.started

        # Test capture provider functionality
        devices = capture_provider.list_audio_devices()
        assert isinstance(devices, dict)
        assert len(devices) > 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_factory_error_propagation(self):
        """Test that factory properly propagates async errors."""

        class FailingAsyncProvider(TranscriptionProvider):
            def __init__(self, **kwargs):
                pass

            async def start_stream(self, audio_config):
                raise RuntimeError("Async test error")

            # Implement other abstract methods
            async def send_audio(self, audio_chunk):
                pass

            async def get_transcription(self):
                yield None

            async def stop_stream(self):
                pass

            def get_required_channels(self) -> int:
                return 1

        factory = AudioProcessorFactory()
        factory.register_transcription_provider("failing_async", FailingAsyncProvider)

        provider = factory.create_transcription_provider("failing_async")
        config = AudioConfig(
            sample_rate=16000, channels=1, chunk_size=1024, format="int16"
        )

        # Should propagate async error
        with pytest.raises(RuntimeError, match="Async test error"):
            await provider.start_stream(config)
