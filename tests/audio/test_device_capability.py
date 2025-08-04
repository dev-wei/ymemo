"""Tests for device capability detection and audio configuration optimization.

Tests the new device-aware functionality for automatic channel detection and configuration optimization.
"""

from unittest.mock import patch

import pytest

from src.core.interfaces import AudioConfig
from src.utils.device_utils import (
    AudioDeviceInfo,
    device_manager,
    get_device_max_channels,
    get_optimal_channels,
    validate_device_config,
)
from tests.base.base_test import BaseIntegrationTest, BaseTest


class TestDeviceCapabilityDetection(BaseTest):
    """Test device capability detection functions."""

    @pytest.mark.unit
    def test_get_device_max_channels_valid_device(self):
        """Test getting max channels for a valid device."""
        # Mock a device with 2 channels
        mock_device = AudioDeviceInfo(
            index=1,
            name="Test Device",
            max_input_channels=2,
            max_output_channels=2,
            default_sample_rate=44100.0,
            is_default_input=False,
        )

        with patch.object(
            device_manager, "get_device_by_index", return_value=mock_device
        ):
            max_channels = get_device_max_channels(1)
            assert max_channels == 2

    @pytest.mark.unit
    def test_get_device_max_channels_invalid_device(self):
        """Test getting max channels for an invalid device."""
        with patch.object(device_manager, "get_device_by_index", return_value=None):
            max_channels = get_device_max_channels(999)
            assert max_channels == 1  # Fallback to 1 channel

    @pytest.mark.unit
    def test_get_device_max_channels_exception(self):
        """Test getting max channels when an exception occurs."""
        with patch.object(
            device_manager, "get_device_by_index", side_effect=Exception("Test error")
        ):
            max_channels = get_device_max_channels(1)
            assert max_channels == 1  # Fallback to 1 channel

    @pytest.mark.unit
    def test_get_optimal_channels_within_limit(self):
        """Test getting optimal channels when requested is within device limit."""
        mock_device = AudioDeviceInfo(
            index=1,
            name="Test Device",
            max_input_channels=4,
            max_output_channels=2,
            default_sample_rate=44100.0,
        )

        with patch.object(
            device_manager, "get_device_by_index", return_value=mock_device
        ):
            optimal = get_optimal_channels(1, 2)
            assert optimal == 2  # Should return requested since it's within limit

    @pytest.mark.unit
    def test_get_optimal_channels_exceeds_limit(self):
        """Test getting optimal channels when requested exceeds device limit."""
        mock_device = AudioDeviceInfo(
            index=1,
            name="Test Device",
            max_input_channels=1,
            max_output_channels=2,
            default_sample_rate=44100.0,
        )

        with patch.object(
            device_manager, "get_device_by_index", return_value=mock_device
        ):
            optimal = get_optimal_channels(1, 4)
            assert optimal == 1  # Should return device max

    @pytest.mark.unit
    def test_get_optimal_channels_error_handling(self):
        """Test getting optimal channels with error handling."""
        with patch.object(
            device_manager, "get_device_by_index", side_effect=Exception("Test error")
        ):
            optimal = get_optimal_channels(1, 4)
            assert optimal == 1  # Should fallback to mono


class TestDeviceConfigValidation(BaseTest):
    """Test device configuration validation."""

    @pytest.mark.unit
    def test_validate_device_config_valid_device(self):
        """Test device config validation with a valid device."""
        mock_device = AudioDeviceInfo(
            index=1,
            name="Test Device",
            max_input_channels=2,
            max_output_channels=2,
            default_sample_rate=48000.0,
            is_default_input=False,
        )

        with patch.object(
            device_manager, "get_device_by_index", return_value=mock_device
        ):
            result = validate_device_config(1, 2, 48000)

            assert result["channels"] == 2
            assert result["sample_rate"] == 48000
            assert result["device_info"]["name"] == "Test Device"
            assert result["device_info"]["max_input_channels"] == 2
            assert len(result["warnings"]) == 0  # No warnings expected

    @pytest.mark.unit
    def test_validate_device_config_channel_reduction(self):
        """Test device config validation with channel reduction needed."""
        mock_device = AudioDeviceInfo(
            index=1,
            name="Mono Device",
            max_input_channels=1,
            max_output_channels=1,
            default_sample_rate=44100.0,
            is_default_input=True,
        )

        with patch.object(
            device_manager, "get_device_by_index", return_value=mock_device
        ):
            result = validate_device_config(1, 4, 44100)

            assert result["channels"] == 1  # Should be reduced to 1
            assert result["sample_rate"] == 44100
            assert result["device_info"]["name"] == "Mono Device"
            assert any(
                "Channel count reduced from 4 to 1" in warning
                for warning in result["warnings"]
            )

    @pytest.mark.unit
    def test_validate_device_config_sample_rate_warning(self):
        """Test device config validation with sample rate warning."""
        mock_device = AudioDeviceInfo(
            index=1,
            name="Test Device",
            max_input_channels=2,
            max_output_channels=2,
            default_sample_rate=48000.0,
        )

        with patch.object(
            device_manager, "get_device_by_index", return_value=mock_device
        ):
            result = validate_device_config(1, 2, 16000)

            assert result["channels"] == 2
            assert result["sample_rate"] == 16000
            assert any(
                "sample rate 16000Hz differs from device default 48000Hz" in warning
                for warning in result["warnings"]
            )

    @pytest.mark.unit
    def test_validate_device_config_device_not_found(self):
        """Test device config validation when device is not found."""
        with patch.object(device_manager, "get_device_by_index", return_value=None):
            result = validate_device_config(999, 2, 44100)

            assert result["channels"] == 1  # Fallback
            assert result["sample_rate"] == 44100
            assert result["device_info"] == {}
            assert any(
                "Device 999 not found" in warning for warning in result["warnings"]
            )

    @pytest.mark.unit
    def test_validate_device_config_exception(self):
        """Test device config validation with exception handling."""
        with patch.object(
            device_manager, "get_device_by_index", side_effect=Exception("Test error")
        ):
            result = validate_device_config(1, 2, 44100)

            assert result["channels"] == 1  # Fallback
            assert result["sample_rate"] == 44100
            assert result["device_info"] == {}
            assert any(
                "Device validation failed: Test error" in warning
                for warning in result["warnings"]
            )


class TestAudioConfigOptimization(BaseTest):
    """Test audio configuration optimization functionality."""

    @pytest.mark.unit
    def test_device_optimized_audio_config_no_device(self):
        """Test device-optimized audio config with no device specified."""
        from src.config.audio_config import AudioSystemConfig

        config = AudioSystemConfig(channels=2, sample_rate=44100)
        optimized = config.get_device_optimized_audio_config(None)

        # Should return base config when no device specified
        assert optimized.channels == 2
        assert optimized.sample_rate == 44100

    @pytest.mark.unit
    def test_device_optimized_audio_config_channel_reduction(self):
        """Test device-optimized audio config with channel reduction."""
        from src.config.audio_config import AudioSystemConfig

        config = AudioSystemConfig(channels=4, sample_rate=16000)

        # Mock validate_device_config to return reduced channels
        mock_result = {
            "channels": 1,
            "sample_rate": 16000,
            "device_info": {"name": "Test Device", "max_input_channels": 1},
            "warnings": ["Channel count reduced from 4 to 1 due to device limitations"],
        }

        with patch(
            "src.utils.device_utils.validate_device_config", return_value=mock_result
        ):
            optimized = config.get_device_optimized_audio_config(1)

            assert optimized.channels == 1  # Should be optimized
            assert optimized.sample_rate == 16000
            assert (
                optimized.chunk_size == config.chunk_size
            )  # Should preserve other settings
            assert optimized.format == config.audio_format

    @pytest.mark.unit
    def test_device_optimized_audio_config_exception_fallback(self):
        """Test device-optimized audio config with exception fallback."""
        from src.config.audio_config import AudioSystemConfig

        config = AudioSystemConfig(channels=4, sample_rate=16000)

        # Mock validate_device_config to raise an exception
        with patch(
            "src.utils.device_utils.validate_device_config",
            side_effect=Exception("Test error"),
        ):
            optimized = config.get_device_optimized_audio_config(1)

            # Should fallback to safe mono configuration
            assert optimized.channels == 1
            assert optimized.sample_rate == 16000
            assert optimized.chunk_size == config.chunk_size
            assert optimized.format == config.audio_format


class TestPyAudioProviderOptimization(BaseIntegrationTest):
    """Test PyAudio provider with configuration optimization."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pyaudio_config_optimization(self):
        """Test that PyAudio provider optimizes configuration for device."""
        from src.audio.providers.pyaudio_capture import PyAudioCaptureProvider

        # Mock the optimization method to avoid actual device detection
        provider = PyAudioCaptureProvider()

        original_config = AudioConfig(
            sample_rate=16000, channels=4, chunk_size=1024, format="int16"
        )

        optimized_config = AudioConfig(
            sample_rate=16000, channels=1, chunk_size=1024, format="int16"
        )

        # Mock the optimization method
        with patch.object(
            provider, "_optimize_config_for_device", return_value=optimized_config
        ):
            actual_optimized = await provider._optimize_config_for_device(
                original_config, 1
            )

            assert actual_optimized.channels == 1  # Should be optimized
            assert actual_optimized.sample_rate == 16000
            assert actual_optimized.chunk_size == 1024
            assert actual_optimized.format == "int16"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pyaudio_optimization_no_device(self):
        """Test that PyAudio provider returns original config when no device specified."""
        from src.audio.providers.pyaudio_capture import PyAudioCaptureProvider

        provider = PyAudioCaptureProvider()

        original_config = AudioConfig(
            sample_rate=16000, channels=4, chunk_size=1024, format="int16"
        )

        # Test with no device ID
        optimized = await provider._optimize_config_for_device(original_config, None)

        # Should return original config unchanged
        assert optimized.channels == original_config.channels
        assert optimized.sample_rate == original_config.sample_rate
        assert optimized.chunk_size == original_config.chunk_size
        assert optimized.format == original_config.format

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_pyaudio_optimization_exception_fallback(self):
        """Test that PyAudio provider falls back to safe config on exception."""
        from src.audio.providers.pyaudio_capture import PyAudioCaptureProvider

        provider = PyAudioCaptureProvider()

        original_config = AudioConfig(
            sample_rate=16000, channels=4, chunk_size=1024, format="int16"
        )

        # Mock validate_device_config to raise exception
        with patch(
            "src.utils.device_utils.validate_device_config",
            side_effect=Exception("Test error"),
        ):
            optimized = await provider._optimize_config_for_device(original_config, 1)

            # Should fallback to safe mono configuration
            assert optimized.channels == 1
            assert optimized.sample_rate == original_config.sample_rate
            assert optimized.chunk_size == original_config.chunk_size
            assert optimized.format == original_config.format
