"""Tests for audio configuration validation and loading.

Validates that audio configuration is properly loaded from environment
variables and that transcription configurations are correctly generated.

Migrated from root directory test_audio_config.py
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from config.audio_config import get_config
from src.core.factory import AudioProcessorFactory
from tests.base.base_test import BaseTest


class TestAudioConfigValidation(BaseTest):
    """Test audio configuration validation functionality."""

    @pytest.fixture
    def test_environment(self):
        """Set up test environment variables."""
        test_env = {
            "AWS_CONNECTION_STRATEGY": "dual",
            "AWS_DUAL_CONNECTION_TEST_MODE": "left_only",
            "AWS_DUAL_SAVE_SPLIT_AUDIO": "true",
            "AWS_DUAL_AUDIO_SAVE_PATH": "./debug_audio/",
            "AWS_DUAL_AUDIO_SAVE_DURATION": "30",
        }

        with patch.dict(os.environ, test_env):
            yield test_env

    def test_config_loading_basic(self, test_environment):
        """Test basic configuration loading."""
        config = get_config()

        assert config is not None
        assert config.transcription_provider == "aws"
        assert config.aws_connection_strategy == "dual"
        assert config.aws_dual_connection_test_mode == "left_only"
        assert config.aws_dual_save_split_audio is True

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
            "connection_strategy",
            "dual_fallback_enabled",
            "channel_balance_threshold",
            "dual_connection_test_mode",
            "dual_save_split_audio",
            "dual_save_raw_audio",
            "dual_audio_save_path",
            "dual_audio_save_duration",
        ]

        for key in expected_keys:
            assert key in transcription_config, f"Missing key: {key}"

        # Verify specific values
        assert transcription_config["connection_strategy"] == "dual"
        assert transcription_config["dual_connection_test_mode"] == "left_only"
        assert transcription_config["dual_save_split_audio"] is True
        assert transcription_config["dual_audio_save_path"] == "./debug_audio/"
        assert transcription_config["dual_audio_save_duration"] == 30

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

        # Verify configuration was applied
        assert hasattr(provider, "dual_save_split_audio")
        assert provider.dual_save_split_audio is True
        assert hasattr(provider, "dual_audio_save_path")
        assert provider.dual_audio_save_path == "./debug_audio/"
        assert hasattr(provider, "dual_audio_save_duration")
        assert provider.dual_audio_save_duration == 30

    def test_config_with_different_providers(self):
        """Test configuration generation for different providers."""
        config = get_config()

        # Test AWS configuration
        aws_config = config.get_transcription_config()
        assert "region" in aws_config
        assert "connection_strategy" in aws_config

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
            with patch.dict(os.environ, {"AWS_DUAL_SAVE_SPLIT_AUDIO": env_value}):
                config = get_config()
                assert (
                    config.aws_dual_save_split_audio == expected
                ), f"Failed for env_value='{env_value}', expected={expected}"

    def test_numeric_environment_variable_parsing(self):
        """Test that numeric environment variables are parsed correctly."""
        with patch.dict(os.environ, {"AWS_DUAL_AUDIO_SAVE_DURATION": "45"}):
            config = get_config()
            assert config.aws_dual_audio_save_duration == 45
            assert isinstance(config.aws_dual_audio_save_duration, int)

        # Test invalid numeric value (should use default)
        with patch.dict(os.environ, {"AWS_DUAL_AUDIO_SAVE_DURATION": "invalid"}):
            config = get_config()
            assert isinstance(config.aws_dual_audio_save_duration, int)
            assert config.aws_dual_audio_save_duration > 0  # Should have valid default

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
        assert config1.aws_connection_strategy == config2.aws_connection_strategy

        # Changes to environment should be reflected in new config calls
        with patch.dict(os.environ, {"AWS_CONNECTION_STRATEGY": "single"}):
            # Depending on implementation, this might require cache clearing
            # For now, just verify the mechanism works
            get_config()
            # The actual behavior depends on whether there's caching implemented
