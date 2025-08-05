"""Tests for audio quality configuration functions."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Import the functions we want to test
from src.config.audio_config import (
    QUALITY_DISPLAY_AVERAGE,
    QUALITY_DISPLAY_HIGH,
    SAMPLE_RATE_AVERAGE,
    SAMPLE_RATE_HIGH,
    get_audio_quality_choices,
    get_current_audio_quality_from_sample_rate,
    get_current_audio_quality_info,
    get_default_audio_quality,
    get_sample_rate_from_quality,
)
from src.ui.audio_quality_handlers import (
    get_current_audio_quality_info_html,
    handle_audio_quality_change,
    update_audio_quality_configuration,
    validate_audio_quality_compatibility,
)
from tests.base.base_test import BaseTest


class TestAudioQualityConfig(BaseTest):
    """Test audio quality configuration functions."""

    def test_get_audio_quality_choices(self):
        """Test that quality choices are returned correctly."""
        choices = get_audio_quality_choices()
        expected = [QUALITY_DISPLAY_HIGH, QUALITY_DISPLAY_AVERAGE]
        assert choices == expected
        assert isinstance(choices, list)
        assert len(choices) == 2

    def test_get_default_audio_quality(self):
        """Test that default quality is Average."""
        default = get_default_audio_quality()
        assert default == QUALITY_DISPLAY_AVERAGE
        assert isinstance(default, str)

    def test_get_current_audio_quality_from_sample_rate(self):
        """Test sample rate to quality mapping."""
        # Test high quality
        quality_high = get_current_audio_quality_from_sample_rate(SAMPLE_RATE_HIGH)
        assert quality_high == QUALITY_DISPLAY_HIGH

        # Test average quality
        quality_avg = get_current_audio_quality_from_sample_rate(SAMPLE_RATE_AVERAGE)
        assert quality_avg == QUALITY_DISPLAY_AVERAGE

        # Test custom high sample rate
        quality_custom_high = get_current_audio_quality_from_sample_rate(48000)
        assert quality_custom_high == QUALITY_DISPLAY_HIGH

        # Test custom low sample rate
        quality_custom_low = get_current_audio_quality_from_sample_rate(8000)
        assert quality_custom_low == QUALITY_DISPLAY_AVERAGE

    def test_get_sample_rate_from_quality(self):
        """Test quality to sample rate mapping."""
        # Test high quality
        rate_high = get_sample_rate_from_quality(QUALITY_DISPLAY_HIGH)
        assert rate_high == SAMPLE_RATE_HIGH

        # Test average quality
        rate_avg = get_sample_rate_from_quality(QUALITY_DISPLAY_AVERAGE)
        assert rate_avg == SAMPLE_RATE_AVERAGE

        # Test case insensitive
        rate_high_lower = get_sample_rate_from_quality("high")
        assert rate_high_lower == SAMPLE_RATE_HIGH

        # Test invalid quality defaults to average
        rate_invalid = get_sample_rate_from_quality("invalid")
        assert rate_invalid == SAMPLE_RATE_AVERAGE

    @patch('src.config.audio_config.get_config')
    def test_get_current_audio_quality_info(self, mock_get_config):
        """Test current quality info generation."""
        # Mock config with high quality
        mock_config = MagicMock()
        mock_config.sample_rate = SAMPLE_RATE_HIGH
        mock_get_config.return_value = mock_config

        info = get_current_audio_quality_info()

        assert isinstance(info, dict)
        assert 'quality' in info
        assert 'sample_rate' in info
        assert 'description' in info
        assert info['quality'] == QUALITY_DISPLAY_HIGH
        assert info['sample_rate'] == f"{SAMPLE_RATE_HIGH:,} Hz"


class TestAudioQualityHandlers(BaseTest):
    """Test audio quality UI handlers."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        # Store original environment
        self.original_env = os.environ.copy()

    def tearDown(self):
        """Clean up test environment."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
        super().tearDown()

    def test_update_audio_quality_configuration_valid(self):
        """Test updating audio quality configuration with valid input."""
        # Test high quality
        result = update_audio_quality_configuration(QUALITY_DISPLAY_HIGH)
        assert result is True
        assert os.environ.get('AUDIO_QUALITY') == 'high'

        # Test average quality
        result = update_audio_quality_configuration(QUALITY_DISPLAY_AVERAGE)
        assert result is True
        assert os.environ.get('AUDIO_QUALITY') == 'average'

    def test_update_audio_quality_configuration_invalid(self):
        """Test updating audio quality configuration with invalid input."""
        # Test invalid quality
        result = update_audio_quality_configuration("Invalid")
        assert result is False

        # Test empty quality
        result = update_audio_quality_configuration("")
        assert result is False

        # Test None quality
        result = update_audio_quality_configuration(None)
        assert result is False

    def test_validate_audio_quality_compatibility_aws(self):
        """Test validation with AWS provider."""
        with patch.dict(os.environ, {'TRANSCRIPTION_PROVIDER': 'aws'}):
            is_compatible, warnings = validate_audio_quality_compatibility(
                QUALITY_DISPLAY_HIGH
            )

            assert is_compatible is True
            assert isinstance(warnings, list)
            # Should have warnings about cost and device support
            assert len(warnings) >= 1

    def test_validate_audio_quality_compatibility_azure(self):
        """Test validation with Azure provider."""
        with patch.dict(os.environ, {'TRANSCRIPTION_PROVIDER': 'azure'}):
            is_compatible, warnings = validate_audio_quality_compatibility(
                QUALITY_DISPLAY_HIGH
            )

            assert is_compatible is True
            assert isinstance(warnings, list)

    @patch('src.ui.audio_quality_handlers.get_current_audio_quality_info_html')
    @patch('src.ui.audio_quality_handlers.update_audio_quality_configuration')
    @patch('src.ui.audio_quality_handlers.validate_audio_quality_compatibility')
    def test_handle_audio_quality_change_success(
        self, mock_validate, mock_update, mock_html
    ):
        """Test successful audio quality change."""
        # Mock successful validation and update
        mock_validate.return_value = (True, [])
        mock_update.return_value = True
        mock_html.return_value = "<div>Quality info</div>"

        status, html = handle_audio_quality_change(QUALITY_DISPLAY_HIGH)

        assert "✅" in status
        assert QUALITY_DISPLAY_HIGH in status
        assert html == "<div>Quality info</div>"

    @patch('src.ui.audio_quality_handlers.get_current_audio_quality_info_html')
    @patch('src.ui.audio_quality_handlers.validate_audio_quality_compatibility')
    def test_handle_audio_quality_change_incompatible(self, mock_validate, mock_html):
        """Test audio quality change with incompatible settings."""
        # Mock incompatible validation
        mock_validate.return_value = (False, ["Some error"])
        mock_html.return_value = "<div>Quality info</div>"

        status, html = handle_audio_quality_change(QUALITY_DISPLAY_HIGH)

        assert "compatibility issues" in status
        assert status == "Audio quality change failed due to compatibility issues"

    @patch('src.ui.audio_quality_handlers.get_current_audio_quality_info')
    def test_get_current_audio_quality_info_html_success(self, mock_get_info):
        """Test HTML generation for quality info."""
        # Mock quality info
        mock_get_info.return_value = {
            'quality': QUALITY_DISPLAY_HIGH,
            'sample_rate': '44,100 Hz',
            'description': 'High Quality (44,100 Hz) - CD-quality audio capture',
        }

        html = get_current_audio_quality_info_html()

        assert QUALITY_DISPLAY_HIGH in html
        assert '44,100 Hz' in html
        assert 'CD-quality' in html
        assert '<div' in html

    @patch('src.ui.audio_quality_handlers.get_current_audio_quality_info')
    def test_get_current_audio_quality_info_html_error(self, mock_get_info):
        """Test HTML generation when info retrieval fails."""
        # Mock exception
        mock_get_info.side_effect = Exception("Test error")

        html = get_current_audio_quality_info_html()

        assert "Unable to load" in html
        assert "Error retrieving" in html


if __name__ == '__main__':
    unittest.main()
