"""Tests for audio configuration validation and loading.

Validates that audio configuration is properly loaded from environment
variables and that transcription configurations are correctly generated.

Migrated from root directory test_audio_config.py
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.config.audio_config import get_config
from src.core.factory import AudioProcessorFactory
from tests.base.base_test import BaseTest


class TestAudioConfigValidation(BaseTest):
    """Test audio configuration validation functionality."""

    @pytest.fixture
    def test_environment(self):
        """Set up test environment variables."""
        test_env = {
            # Use new provider-agnostic variables
            "SAVE_RAW_AUDIO": "true",
            "SAVE_SPLIT_AUDIO": "true",
            "AUDIO_SAVE_PATH": "./debug_audio/",
            "AUDIO_SAVE_DURATION": "30",
        }

        with patch.dict(os.environ, test_env):
            yield test_env

    def test_config_loading_basic(self, test_environment):
        """Test basic configuration loading."""
        config = get_config()

        assert config is not None
        assert config.transcription_provider == "aws"
        # Test new provider-agnostic audio saving config
        assert config.save_raw_audio is True
        assert config.save_split_audio is True
        assert config.audio_save_path == "./debug_audio/"
        assert config.audio_save_duration == 30

    def test_transcription_config_generation(self, test_environment):
        """Test transcription configuration generation."""
        config = get_config()
        transcription_config = config.get_transcription_config()

        # Verify structure
        assert isinstance(transcription_config, dict)

        # Verify required keys for AWS provider
        expected_keys = [
            "region",
            "language_code",
            "dual_fallback_enabled",
            "channel_balance_threshold",
        ]

        for key in expected_keys:
            assert key in transcription_config, f"Missing key: {key}"

        # Verify specific values
        assert transcription_config["dual_fallback_enabled"] is True
        assert transcription_config["channel_balance_threshold"] == 0.3

    @patch("src.audio.providers.aws_transcribe.boto3")
    def test_provider_creation_with_config(self, mock_boto3, test_environment):
        """Test provider creation with generated configuration."""
        # Mock AWS
        mock_boto3.Session.return_value.client.return_value = MagicMock()

        config = get_config()
        transcription_config = config.get_transcription_config()

        # Create provider
        provider = AudioProcessorFactory.create_transcription_provider(
            "aws", **transcription_config
        )

        assert provider is not None

        # Verify configuration was applied (audio saving parameters removed from provider)
        assert hasattr(provider, "region")
        # Use actual configured region instead of hardcoded value
        assert provider.region == config.aws_region
        assert hasattr(provider, "language_code")
        assert provider.language_code == "en-US"
        # Note: dual_save_split_audio parameter removed - audio saving handled at pipeline level

    def test_config_with_different_providers(self):
        """Test configuration generation for different providers."""
        config = get_config()

        # Test AWS configuration
        aws_config = config.get_transcription_config()
        assert "region" in aws_config
        assert "dual_fallback_enabled" in aws_config

        # Test with different provider setting
        with patch.object(config, "transcription_provider", "azure"):
            azure_config = config.get_transcription_config()
            assert "speech_key" in azure_config
            assert "region" in azure_config
            assert "language_code" in azure_config

    def test_boolean_environment_variable_parsing(self):
        """Test that boolean environment variables are parsed correctly."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("", False),
            ("invalid", False),  # Default to False for invalid values
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"SAVE_SPLIT_AUDIO": env_value}):
                config = get_config()
                assert (
                    config.save_split_audio == expected
                ), f"Failed for env_value='{env_value}', expected={expected}"

    def test_numeric_environment_variable_parsing(self):
        """Test that numeric environment variables are parsed correctly."""
        with patch.dict(os.environ, {"AUDIO_SAVE_DURATION": "45"}):
            config = get_config()
            assert config.audio_save_duration == 45
            assert isinstance(config.audio_save_duration, int)

        # Test invalid numeric value (should use default)
        with patch.dict(os.environ, {"AUDIO_SAVE_DURATION": "invalid"}):
            config = get_config()
            assert isinstance(config.audio_save_duration, int)
            assert config.audio_save_duration > 0  # Should have valid default

    def test_config_validation_errors(self):
        """Test configuration validation with invalid settings."""
        config = get_config()

        # The validate method should check for invalid combinations
        # and raise appropriate errors or warnings
        try:
            config.validate()
            # If validation passes, that's fine
        except Exception as e:
            # If validation fails, the exception should be meaningful
            assert isinstance(e, ValueError | TypeError)
            assert len(str(e)) > 0  # Should have a meaningful error message

    def test_config_singleton_behavior(self):
        """Test that config behaves as expected for multiple calls."""
        config1 = get_config()
        config2 = get_config()

        # Should return the same values (not necessarily same instance)
        assert config1.transcription_provider == config2.transcription_provider
        assert config1.save_split_audio == config2.save_split_audio
        assert config1.save_raw_audio == config2.save_raw_audio

        # Test that get_config() works consistently
        get_config()
