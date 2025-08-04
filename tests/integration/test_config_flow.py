"""Integration tests for configuration flow from .env to AWS provider.

Tests that validate the complete configuration pipeline:
1. Environment variables loading from .env
2. Configuration flow through get_config() → get_transcription_config()
3. AWS provider initialization with correct parameters
4. Audio saving component initialization

Migrated from root directory test_config_validation.py
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config.audio_config import get_config
from src.core.factory import AudioProcessorFactory
from src.core.processor import AudioProcessor
from tests.base.base_test import BaseIntegrationTest


class TestConfigurationFlow(BaseIntegrationTest):
    """Test complete configuration flow from environment to components."""

    @pytest.fixture(autouse=True)
    def setup_test_env(self):
        """Set up test environment variables."""
        self.test_env_vars = {
            # Use new provider-agnostic variables
            "SAVE_RAW_AUDIO": "true",
            "SAVE_SPLIT_AUDIO": "true",
            "AUDIO_SAVE_PATH": "./debug_audio/",
            "AUDIO_SAVE_DURATION": "30",
            "LOG_LEVEL": "INFO",
        }

        # Patch environment
        self.env_patches = []
        for key, value in self.test_env_vars.items():
            patcher = patch.dict(os.environ, {key: value})
            patcher.start()
            self.env_patches.append(patcher)

        yield

        # Clean up patches
        for patcher in self.env_patches:
            patcher.stop()

    def test_environment_variables_loading(self):
        """Test that environment variables are loaded correctly."""
        # Test critical environment variables
        critical_vars = ["SAVE_SPLIT_AUDIO", "SAVE_RAW_AUDIO"]

        for var in critical_vars:
            value = os.getenv(var)
            assert value is not None, f"Critical environment variable {var} not set"
            assert (
                value == self.test_env_vars[var]
            ), f"Environment variable {var} has wrong value"

    def test_audio_config_loading(self):
        """Test AudioSystemConfig loading from environment variables."""
        config = get_config()

        # Verify key configuration values
        assert config.transcription_provider == "aws"
        # New provider-agnostic configuration
        assert config.save_split_audio is True
        assert config.save_raw_audio is True
        assert config.audio_save_path == "./debug_audio/"
        assert config.audio_save_duration == 30

    def test_transcription_config_extraction(self):
        """Test transcription config extraction from system config."""
        config = get_config()
        transcription_config = config.get_transcription_config()

        # Verify all required keys are present
        required_keys = [
            "region",
            "language_code",
            "dual_fallback_enabled",
            "channel_balance_threshold",
        ]

        for key in required_keys:
            assert key in transcription_config, f"Missing key: {key}"

        # Verify critical values
        assert transcription_config["dual_fallback_enabled"] is True
        assert transcription_config["channel_balance_threshold"] == 0.3

    @patch("src.audio.providers.aws_transcribe.boto3")
    def test_aws_provider_creation_with_config(self, mock_boto3):
        """Test AWS provider creation receives correct configuration."""
        # Mock boto3 to avoid AWS calls
        mock_boto3.Session.return_value.client.return_value = MagicMock()

        config = get_config()
        transcription_config = config.get_transcription_config()

        # Create AWS provider with configuration
        factory = AudioProcessorFactory()
        aws_provider = factory.create_transcription_provider(
            "aws", **transcription_config
        )

        # Verify provider was created
        assert aws_provider is not None

        # Verify AWS provider configuration (audio saving parameters removed)
        assert hasattr(aws_provider, "region")
        assert aws_provider.region == "us-east-1"
        assert hasattr(aws_provider, "language_code")
        assert aws_provider.language_code == "en-US"
        # Note: dual_save_split_audio parameter removed - audio saving now handled at pipeline level

    @patch("src.audio.providers.aws_transcribe.boto3")
    @patch("src.audio.providers.pyaudio_capture.pyaudio.PyAudio")
    def test_audio_processor_integration(self, mock_pyaudio, mock_boto3):
        """Test AudioProcessor creation like the real application does."""
        # Mock external dependencies
        mock_boto3.Session.return_value.client.return_value = MagicMock()
        mock_pyaudio.return_value = MagicMock()

        # Replicate how session_manager.py creates AudioProcessor
        system_config = get_config()

        audio_processor = AudioProcessor(
            transcription_provider=system_config.transcription_provider,
            capture_provider=system_config.capture_provider,
            transcription_config=system_config.get_transcription_config(),
        )

        # Verify AudioProcessor was created
        assert audio_processor is not None

        # Verify transcription provider has correct configuration
        provider = audio_processor.transcription_provider
        assert provider is not None

        # Audio saving parameters removed from transcription provider
        # Audio saving is now handled at pipeline level by AudioSaver component
        assert hasattr(provider, "region")
        assert hasattr(provider, "language_code")

    def test_directory_setup_validation(self):
        """Test debug directory setup and permissions."""
        config = get_config()
        debug_dir = Path(config.audio_save_path)

        # Directory doesn't need to exist initially - it gets created automatically
        # But if it exists, it should be writable
        if debug_dir.exists():
            assert os.access(debug_dir, os.W_OK), "Debug directory is not writable"

    def test_configuration_consistency(self):
        """Test that configuration is consistent across multiple calls."""
        # Multiple calls should return consistent configuration
        config1 = get_config()
        config2 = get_config()

        # Compare key attributes
        assert config1.save_split_audio == config2.save_split_audio
        assert config1.save_raw_audio == config2.save_raw_audio
        assert config1.audio_save_path == config2.audio_save_path

        # Transcription configs should also be consistent
        trans_config1 = config1.get_transcription_config()
        trans_config2 = config2.get_transcription_config()

        for key in trans_config1:
            assert (
                trans_config1[key] == trans_config2[key]
            ), f"Inconsistent value for {key}"


class TestConfigurationErrorHandling(BaseIntegrationTest):
    """Test configuration error handling and validation."""

    def test_missing_critical_environment_variables(self):
        """Test behavior when critical environment variables are missing."""
        # Test with minimal environment
        with patch.dict(os.environ, {}, clear=True):
            config = get_config()

            # Should still work with defaults
            assert config is not None
            assert config.transcription_provider == "aws"  # Default value

            # Should have sensible defaults for audio saving
            assert config.save_split_audio is False  # Default disabled

    def test_invalid_environment_variable_values(self):
        """Test behavior with invalid environment variable values."""
        invalid_env = {
            "SAVE_SPLIT_AUDIO": "invalid_boolean",
            "AUDIO_SAVE_DURATION": "not_a_number",
        }

        with patch.dict(os.environ, invalid_env):
            config = get_config()

            # Should handle invalid values gracefully with defaults
            assert config.save_split_audio in [
                True,
                False,
            ]  # Should be a boolean
            assert isinstance(config.audio_save_duration, int)  # Should be an integer

    @patch("src.audio.providers.aws_transcribe.boto3")
    def test_aws_provider_creation_error_handling(self, mock_boto3):
        """Test error handling during AWS provider creation."""
        # Mock boto3 to raise an exception
        mock_boto3.Session.side_effect = Exception("AWS credentials not configured")

        config = get_config()
        transcription_config = config.get_transcription_config()

        factory = AudioProcessorFactory()

        # Should raise an appropriate exception
        with pytest.raises(Exception):
            factory.create_transcription_provider("aws", **transcription_config)
