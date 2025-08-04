"""Audio quality selection event handlers for the UI interface."""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from src.config.audio_config import (
    QUALITY_DISPLAY_AVERAGE,
    QUALITY_DISPLAY_HIGH,
    get_audio_quality_choices,
    get_current_audio_quality_info,
    get_sample_rate_from_quality,
)

logger = logging.getLogger(__name__)

# Constants for repeated strings
ERROR_PREFIX = "❌"
SUCCESS_PREFIX = "✅"
WARNING_PREFIX = "⚠️"
QUALITY_CHANGE_EMOJI = "🎚️"

# HTML styling constants
HTML_STYLE_BASE = """
<div style='font-size: 0.9em; color: #444;'>
    <strong>Current Quality:</strong> <span style='color: #666;'>{quality}</span><br>
    <strong>Sample Rate:</strong> <span style='color: #666;'>{sample_rate}</span><br>
    <span style='color: #666; font-size: 0.8em;'>{description}</span>
</div>
""".strip()

HTML_STYLE_ERROR = """
<div style='font-size: 0.9em; color: #444;'>
    <strong>Audio Quality:</strong> Unable to load<br>
    <span style='color: #666; font-size: 0.8em;'>Error retrieving quality information</span>
</div>
""".strip()


def update_audio_quality_configuration(selected_quality: str) -> bool:
    """Update audio configuration with selected quality setting.

    Args:
        selected_quality: The selected audio quality (QUALITY_DISPLAY_HIGH or QUALITY_DISPLAY_AVERAGE)

    Returns:
        True if configuration was updated successfully, False otherwise
    """
    if not selected_quality or selected_quality not in [
        QUALITY_DISPLAY_HIGH,
        QUALITY_DISPLAY_AVERAGE,
    ]:
        logger.error(f"{ERROR_PREFIX} Invalid audio quality: {selected_quality}")
        return False

    try:
        # Map quality to sample rate
        sample_rate = get_sample_rate_from_quality(selected_quality)

        # Update environment variable
        os.environ['AUDIO_QUALITY'] = selected_quality.lower()

        logger.info(
            f"📊 Audio quality updated to: {selected_quality} ({sample_rate:,} Hz)"
        )
        return True

    except (ValueError, TypeError, KeyError) as e:
        logger.error(
            f"{ERROR_PREFIX} Failed to update audio quality configuration: {e}"
        )
        return False
    except Exception as e:
        logger.error(f"{ERROR_PREFIX} Unexpected error updating audio quality: {e}")
        return False


def validate_audio_quality_compatibility(quality: str) -> Tuple[bool, List[str]]:
    """Validate if the selected audio quality is compatible with current settings.

    Args:
        quality: Selected audio quality setting

    Returns:
        Tuple of (is_compatible, list_of_warnings)
    """
    warnings: List[str] = []
    is_compatible: bool = True

    try:
        sample_rate: int = get_sample_rate_from_quality(quality)
        provider: str = os.getenv('TRANSCRIPTION_PROVIDER', 'aws').lower()

        # Provider-specific validation
        validation_rules: Dict[str, Any] = {
            'aws': _validate_aws_compatibility,
            'azure': _validate_azure_compatibility,
            'whisper': _validate_whisper_compatibility,
            'google': _validate_google_compatibility,
        }

        validator = validation_rules.get(provider, _validate_default_compatibility)
        provider_warnings: List[str] = validator(quality, sample_rate)
        warnings.extend(provider_warnings)

        # General device compatibility check
        if quality == QUALITY_DISPLAY_HIGH:
            warnings.append("Ensure your audio device supports 44,100 Hz sample rate")

    except Exception as e:
        logger.error(f"{ERROR_PREFIX} Audio quality validation error: {e}")
        warnings.append("Unable to validate audio quality compatibility")
        is_compatible = False

    return is_compatible, warnings


def _validate_aws_compatibility(quality: str, sample_rate: int) -> List[str]:
    """Validate AWS Transcribe compatibility.

    Args:
        quality: Audio quality setting
        sample_rate: Sample rate in Hz

    Returns:
        List of warning messages
    """
    warnings: List[str] = []
    aws_supported_rates: List[int] = [8000, 16000, 22050, 44100, 48000]

    if sample_rate not in aws_supported_rates:
        warnings.append(
            f"Sample rate {sample_rate}Hz may not be optimal for AWS Transcribe"
        )
    elif quality == QUALITY_DISPLAY_HIGH:
        warnings.append("High quality audio may increase AWS Transcribe costs")

    return warnings


def _validate_azure_compatibility(quality: str, sample_rate: int) -> List[str]:
    """Validate Azure Speech Service compatibility.

    Args:
        quality: Audio quality setting
        sample_rate: Sample rate in Hz

    Returns:
        List of warning messages
    """
    warnings: List[str] = []
    if quality == QUALITY_DISPLAY_HIGH:
        warnings.append("High quality audio may increase Azure Speech Service costs")
    return warnings


def _validate_whisper_compatibility(quality: str, sample_rate: int) -> List[str]:
    """Validate Whisper provider compatibility.

    Args:
        quality: Audio quality setting
        sample_rate: Sample rate in Hz

    Returns:
        List of warning messages
    """
    warnings: List[str] = []
    if quality == QUALITY_DISPLAY_HIGH:
        warnings.append("High quality audio may increase processing time for whisper")
    return warnings


def _validate_google_compatibility(quality: str, sample_rate: int) -> List[str]:
    """Validate Google provider compatibility.

    Args:
        quality: Audio quality setting
        sample_rate: Sample rate in Hz

    Returns:
        List of warning messages
    """
    warnings: List[str] = []
    if quality == QUALITY_DISPLAY_HIGH:
        warnings.append("High quality audio may increase processing time for google")
    return warnings


def _validate_default_compatibility(quality: str, sample_rate: int) -> List[str]:
    """Default validation for unknown providers.

    Args:
        quality: Audio quality setting
        sample_rate: Sample rate in Hz

    Returns:
        List of warning messages
    """
    warnings: List[str] = []
    if quality == QUALITY_DISPLAY_HIGH:
        warnings.append("High quality audio may affect performance")
    return warnings


def handle_audio_quality_change(selected_quality: str) -> Tuple[str, str]:
    """Handle audio quality selection change event.

    Args:
        selected_quality: The newly selected audio quality

    Returns:
        Tuple of (status_message, quality_info_html)
    """
    try:
        logger.info(
            f"{QUALITY_CHANGE_EMOJI} Audio quality change requested: {selected_quality}"
        )

        # Validate compatibility
        is_compatible, warnings = validate_audio_quality_compatibility(selected_quality)

        if not is_compatible:
            error_msg = f"{ERROR_PREFIX} Audio quality change failed - compatibility issues detected"
            logger.error(error_msg)
            return error_msg, get_current_audio_quality_info_html()

        # Update configuration
        if update_audio_quality_configuration(selected_quality):
            success_msg = (
                f"{SUCCESS_PREFIX} Audio quality updated to {selected_quality}"
            )

            # Add warnings if any
            if warnings:
                warning_text = "<br>".join(f"{WARNING_PREFIX} {w}" for w in warnings)
                success_msg += f"<br><small>{warning_text}</small>"

            logger.info(
                f"{SUCCESS_PREFIX} Audio quality successfully changed to: {selected_quality}"
            )
            return success_msg, get_current_audio_quality_info_html()
        else:
            error_msg = f"{ERROR_PREFIX} Failed to update audio quality configuration"
            logger.error(error_msg)
            return error_msg, get_current_audio_quality_info_html()

    except Exception as e:
        error_msg = f"{ERROR_PREFIX} Audio quality change error: {str(e)}"
        logger.error(error_msg)
        return error_msg, get_current_audio_quality_info_html()


def get_current_audio_quality_info_html() -> str:
    """Get current audio quality information formatted as HTML.

    Returns:
        HTML string with current audio quality information
    """
    try:
        info = get_current_audio_quality_info()
        return HTML_STYLE_BASE.format(
            quality=info['quality'],
            sample_rate=info['sample_rate'],
            description=info['description'],
        )

    except Exception as e:
        logger.error(f"{ERROR_PREFIX} Error getting audio quality info: {e}")
        return HTML_STYLE_ERROR


def get_current_audio_quality() -> str:
    """Get the current audio quality setting for UI initialization.

    Returns:
        Current audio quality setting (QUALITY_DISPLAY_HIGH, QUALITY_DISPLAY_AVERAGE, or default)
    """
    try:
        info = get_current_audio_quality_info()
        return info['quality']
    except Exception as e:
        logger.error(f"{ERROR_PREFIX} Error getting current audio quality: {e}")
        return QUALITY_DISPLAY_HIGH  # Default fallback
