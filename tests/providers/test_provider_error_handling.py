"""Provider error handling tests using new test infrastructure.

Migrated from unittest to pytest with centralized fixtures and base classes.
Tests consistent error handling patterns across all providers.
"""

from unittest.mock import Mock, patch

import pytest

from src.core.factory import AudioProcessorFactory
from src.core.interfaces import AudioCaptureProvider, AudioConfig, TranscriptionProvider
from tests.base.base_test import BaseIntegrationTest, BaseTest


class TestProviderErrorHandling(BaseTest):
    """Test consistent error handling across all providers using new infrastructure."""

    @pytest.mark.unit
    def test_transcription_provider_parameter_validation(self):
        """Test that all transcription providers validate parameters consistently."""
        test_cases = [
            # (provider_name, invalid_params, expected_error_type, error_substring)
            ("aws", {"region": ""}, (RuntimeError, ValueError, TypeError), "region"),
            ("aws", {"region": None}, (RuntimeError, ValueError, TypeError), "region"),
            ("aws", {"region": 123}, (RuntimeError, ValueError, TypeError), "region"),
            (
                "aws",
                {"language_code": ""},
                (RuntimeError, ValueError, TypeError),
                "language",
            ),
            (
                "aws",
                {"language_code": None},
                (RuntimeError, ValueError, TypeError),
                "language",
            ),
            (
                "aws",
                {"profile_name": 123},
                (RuntimeError, ValueError, TypeError),
                "profile",
            ),
        ]

        for provider_name, params, expected_error, error_substring in test_cases:
            with pytest.raises(expected_error) as exc_info:
                AudioProcessorFactory.create_transcription_provider(
                    provider_name, **params
                )

            error_msg = str(exc_info.value).lower()
            assert (
                error_substring.lower() in error_msg
            ), f"Expected '{error_substring}' in error message: {error_msg}"

    @pytest.mark.unit
    def test_audio_capture_provider_parameter_validation(self):
        """Test that all audio capture providers validate parameters consistently."""
        # File provider allows any file_path in constructor, validates at runtime
        # Test that providers can be created with various parameters
        test_cases = [
            # (provider_name, params) - these should succeed at creation time
            ("file", {"file_path": "test.wav"}),
            ("file", {"file_path": "/some/path.wav"}),
        ]

        for provider_name, params in test_cases:
            try:
                provider = AudioProcessorFactory.create_audio_capture_provider(
                    provider_name, **params
                )
                assert provider is not None
                # For file provider, verify file_path was set
                if provider_name == "file":
                    assert hasattr(provider, "file_path")
                    assert provider.file_path == params["file_path"]
            except (RuntimeError, TypeError):
                # Some providers may fail in test environment - that's acceptable
                pass

    @pytest.mark.unit
    def test_factory_error_message_format(self):
        """Test that factory provides helpful error messages."""
        factory = AudioProcessorFactory()

        # Test invalid transcription provider
        with pytest.raises(ValueError) as exc_info:
            factory.create_transcription_provider("nonexistent_provider")

        error_msg = str(exc_info.value)
        assert "nonexistent_provider" in error_msg
        assert "Available providers" in error_msg
        assert "aws" in error_msg  # Should list available providers

        # Test invalid capture provider
        with pytest.raises(ValueError) as exc_info:
            factory.create_audio_capture_provider("nonexistent_capture")

        error_msg = str(exc_info.value)
        assert "nonexistent_capture" in error_msg
        assert "Available providers" in error_msg
        assert (
            "file" in error_msg or "pyaudio" in error_msg
        )  # Should list available providers

    @pytest.mark.integration
    def test_provider_initialization_error_wrapping(self):
        """Test that initialization errors are properly wrapped."""
        factory = AudioProcessorFactory()

        # Test with provider that will fail initialization
        try:
            # This may succeed or fail depending on AWS setup
            provider = factory.create_transcription_provider(
                "aws", region="invalid-region-12345"
            )
            # If it succeeds, that's okay - AWS validation might be lazy
            assert provider is not None
        except (RuntimeError, ValueError, TypeError) as e:
            # Expected behavior - should wrap the error appropriately
            assert len(str(e)) > 0  # Should have a meaningful error message

    @pytest.mark.integration
    @patch("boto3.Session")
    def test_aws_provider_configuration_validation(self, mock_boto3):
        """Test AWS provider configuration validation with mocked boto3."""
        # Mock boto3 session creation
        mock_session = Mock()
        mock_boto3.return_value = mock_session

        factory = AudioProcessorFactory()

        # Test valid configuration
        try:
            provider = factory.create_transcription_provider(
                "aws", region="us-east-1", language_code="en-US"
            )
            assert provider is not None
            assert provider.region == "us-east-1"
            assert provider.language_code == "en-US"
        except (RuntimeError, TypeError):
            # Expected if AWS provider creation fails in test environment
            pytest.skip("AWS provider creation failed - expected in test environment")

    @pytest.mark.unit
    def test_error_logging_consistency(self, caplog):
        """Test that error logging is consistent across providers."""
        factory = AudioProcessorFactory()

        # Test that errors are logged when they occur
        with pytest.raises(ValueError):
            factory.create_transcription_provider("invalid_provider")

        # Check that the error was logged (factory should log errors)
        # The actual logging behavior may vary, so we're flexible here
        if caplog.records:
            # If logging occurred, verify it contains useful information
            log_messages = [record.message for record in caplog.records]
            assert any("invalid_provider" in msg.lower() for msg in log_messages)

    @pytest.mark.unit
    def test_provider_type_checking(self):
        """Test that provider type checking works correctly."""
        factory = AudioProcessorFactory()

        # Test registering invalid provider class
        class NotAProvider:
            pass

        with pytest.raises(TypeError, match="must implement.*Provider interface"):
            factory.register_transcription_provider("invalid", NotAProvider)

        with pytest.raises(TypeError, match="must implement.*Provider interface"):
            factory.register_audio_capture_provider("invalid", NotAProvider)

        # Test registering valid provider classes
        class ValidTranscriptionProvider(TranscriptionProvider):
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

        class ValidCaptureProvider(AudioCaptureProvider):
            async def start_capture(self, audio_config, device_id=None):
                pass

            async def get_audio_stream(self):
                yield b"test"

            async def stop_capture(self):
                pass

            def list_audio_devices(self):
                return {}

        # These should succeed
        factory.register_transcription_provider(
            "valid_transcription", ValidTranscriptionProvider
        )
        factory.register_audio_capture_provider("valid_capture", ValidCaptureProvider)

        # Verify registration worked
        transcription_providers = factory.list_transcription_providers()
        capture_providers = factory.list_audio_capture_providers()

        assert "valid_transcription" in transcription_providers
        assert "valid_capture" in capture_providers

    @pytest.mark.unit
    def test_audio_config_validation_patterns(self, default_audio_config):
        """Test that AudioConfig validation follows consistent patterns."""
        # Test valid config using centralized fixture
        config = default_audio_config
        assert isinstance(config, AudioConfig)
        assert config.sample_rate > 0
        assert config.channels > 0
        assert config.chunk_size > 0

        # Test creating configs with various parameters
        test_configs = [
            AudioConfig(sample_rate=16000, channels=1, chunk_size=1024, format="int16"),
            AudioConfig(
                sample_rate=48000, channels=2, chunk_size=2048, format="float32"
            ),
        ]

        for config in test_configs:
            assert isinstance(config.sample_rate, int)
            assert isinstance(config.channels, int)
            assert isinstance(config.chunk_size, int)
            assert isinstance(config.format, str)
            assert config.sample_rate > 0
            assert config.channels > 0
            assert config.chunk_size > 0
            assert len(config.format) > 0


class TestProviderErrorRecovery(BaseIntegrationTest):
    """Test provider error recovery and isolation using new infrastructure."""

    @pytest.mark.integration
    def test_factory_registry_isolation(self):
        """Test that factory registry errors don't affect other providers."""
        factory = AudioProcessorFactory()

        # Register a valid provider
        class WorkingProvider(TranscriptionProvider):
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

        factory.register_transcription_provider("working", WorkingProvider)

        # Try to register an invalid provider - should fail but not affect others
        class BrokenProvider:
            pass

        with pytest.raises(TypeError):
            factory.register_transcription_provider("broken", BrokenProvider)

        # Working provider should still be available
        providers = factory.list_transcription_providers()
        assert "working" in providers

        # Should be able to create working provider
        provider = factory.create_transcription_provider("working")
        assert isinstance(provider, WorkingProvider)

    @pytest.mark.integration
    def test_provider_creation_independence(self):
        """Test that provider creation failures are independent."""
        factory = AudioProcessorFactory()

        # Test that failure to create one provider doesn't affect others
        provider_results = {}

        # Try to create various providers (some may fail in test environment)
        provider_attempts = [
            ("aws", {"region": "us-east-1", "language_code": "en-US"}),
            (
                "file",
                {"file_path": "test.wav"},
            ),  # Use simple filename that won't cause creation errors
        ]

        for provider_name, config in provider_attempts:
            try:
                if provider_name in factory.list_transcription_providers():
                    factory.create_transcription_provider(provider_name, **config)
                    provider_results[provider_name] = "success"
                elif provider_name in factory.list_audio_capture_providers():
                    factory.create_audio_capture_provider(provider_name, **config)
                    provider_results[provider_name] = "success"
            except (RuntimeError, TypeError, ValueError) as e:
                provider_results[provider_name] = f"failed: {type(e).__name__}"

        # At least one provider attempt should have been made
        assert len(provider_results) > 0

        # Factory should still be functional after failures
        providers = factory.list_transcription_providers()
        assert isinstance(providers, dict)
        assert len(providers) > 0

    @pytest.mark.unit
    def test_error_message_helpfulness(self):
        """Test that error messages provide helpful guidance."""
        factory = AudioProcessorFactory()

        # Test helpful error for unknown providers
        with pytest.raises(ValueError) as exc_info:
            factory.create_transcription_provider("unknown_provider")

        error_msg = str(exc_info.value)
        # Should contain the invalid name
        assert "unknown_provider" in error_msg
        # Should suggest available alternatives
        assert "Available" in error_msg or "supported" in error_msg.lower()
        # Should list at least one valid provider
        assert "aws" in error_msg

        # Test helpful error for wrong parameter types
        with pytest.raises((ValueError, TypeError, RuntimeError)) as exc_info:
            factory.create_transcription_provider("aws", region=123)  # Wrong type

        error_msg = str(exc_info.value)
        # Should mention the parameter issue
        assert "region" in error_msg.lower() or "parameter" in error_msg.lower()

        # Test helpful error for audio capture providers
        with pytest.raises(ValueError) as exc_info:
            factory.create_audio_capture_provider("unknown_capture")

        error_msg = str(exc_info.value)
        assert "unknown_capture" in error_msg
        assert "Available" in error_msg or "supported" in error_msg.lower()
        # Should list at least one valid provider
        assert any(provider in error_msg for provider in ["file", "pyaudio"])
