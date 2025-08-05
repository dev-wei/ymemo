"""Language selection event handlers for the UI interface."""

import logging
import os
from typing import Tuple

import gradio as gr

from src.config.language_config import (
    get_display_name,
    get_language_code,
    is_language_supported,
)
from src.utils.status_manager import status_manager

logger = logging.getLogger(__name__)


def update_language_configuration(selected_language: str) -> bool:
    """Update audio configuration with selected language codes.

    Args:
        selected_language: The selected language key (e.g., "English (US)")

    Returns:
        True if configuration was updated successfully, False otherwise
    """
    try:
        # Get provider-specific language codes
        aws_code = get_language_code(selected_language, "aws")
        azure_code = get_language_code(selected_language, "azure")

        if not aws_code and not azure_code:
            logger.warning(
                f"⚠️ No provider codes found for language: {selected_language}"
            )
            return False

        # Update environment variables to persist language selection
        updated = False
        if aws_code:
            os.environ['AWS_LANGUAGE_CODE'] = aws_code
            logger.info(f"🌐 Updated AWS_LANGUAGE_CODE to: {aws_code}")
            updated = True

        if azure_code:
            os.environ['AZURE_SPEECH_LANGUAGE'] = azure_code
            logger.info(f"🌐 Updated AZURE_SPEECH_LANGUAGE to: {azure_code}")
            updated = True

        return updated

    except Exception as e:
        logger.error(f"❌ Error updating language configuration: {e}")
        return False


def handle_language_change(selected_language: str) -> Tuple[str, str]:
    """Handle language dropdown change event.

    Args:
        selected_language: The selected language key (e.g., "English (US)")

    Returns:
        Tuple of (status_message, confirmation_message)
    """
    try:
        logger.info(f"🌐 Language change requested: {selected_language}")

        # Validate the selected language
        if not selected_language:
            logger.warning("⚠️ No language selected")
            gr.Warning("No language selected ⚠️", duration=3)
            return status_manager.get_status_message(), "Language selection required"

        # Get display name
        display_name = get_display_name(selected_language)

        # Check if language is supported by current providers
        aws_code = get_language_code(selected_language, "aws")
        azure_code = get_language_code(selected_language, "azure")

        if not aws_code and not azure_code:
            logger.warning(
                f"❌ Language '{selected_language}' is not supported by any provider"
            )
            gr.Warning(
                f"Language '{selected_language}' is not supported by any provider ⚠️",
                duration=5,
            )
            return status_manager.get_status_message(), "Language not supported"

        # Check provider-specific support
        provider_support = []
        if aws_code:
            provider_support.append(f"AWS ({aws_code})")
        if azure_code:
            provider_support.append(f"Azure ({azure_code})")

        support_info = ", ".join(provider_support)

        # Update language configuration
        config_updated = update_language_configuration(selected_language)
        if not config_updated:
            logger.warning(f"⚠️ Failed to update configuration for {selected_language}")

        success_msg = f"Language changed to {display_name}"
        logger.info(f"✅ {success_msg} - Supported by: {support_info}")

        status_manager.set_status(
            status_manager.current_status, f"Language: {display_name}"
        )

        confirmation_msg = f"✅ Language set to: {display_name}"
        if config_updated:
            confirmation_msg += " (will apply to next recording session)"

        return status_manager.get_status_message(), confirmation_msg

    except Exception as e:
        error_msg = f"System error changing language: {str(e)}"
        logger.error(f"❌ {error_msg}", exc_info=True)
        raise gr.Error(error_msg, duration=5)


def get_current_language_info(selected_language: str) -> str:
    """Get detailed information about the currently selected language.

    Args:
        selected_language: The selected language key

    Returns:
        HTML formatted language information
    """
    try:
        if not selected_language:
            return "<span style='color: #666;'>No language selected</span>"

        # Get provider codes
        aws_code = get_language_code(selected_language, "aws")
        azure_code = get_language_code(selected_language, "azure")

        if not aws_code and not azure_code:
            return "<span style='color: red;'>⚠️ Language not supported</span>"

        # Build support information
        support_parts = []
        if aws_code:
            support_parts.append(f"AWS: {aws_code}")
        if azure_code:
            support_parts.append(f"Azure: {azure_code}")

        support_text = " | ".join(support_parts)
        display_name = get_display_name(selected_language)

        return f"""
        <div style='font-size: 0.9em; color: #444;'>
            <strong>{display_name}</strong><br/>
            <span style='color: #666;'>{support_text}</span><br/>
            <span style='color: #888; font-size: 0.8em;'>Changes apply to next recording</span>
        </div>
        """

    except Exception as e:
        logger.error(f"❌ Error getting language info: {e}")
        return "<span style='color: red;'>Error getting language info</span>"


def validate_language_for_provider(
    language_key: str, provider: str
) -> Tuple[bool, str]:
    """Validate if a language is supported by a specific provider.

    Args:
        language_key: Language key to validate
        provider: Provider name ("aws" or "azure")

    Returns:
        Tuple of (is_supported, status_message)
    """
    try:
        if not is_language_supported(language_key, provider):
            return (
                False,
                f"Language '{language_key}' not supported by {provider.upper()}",
            )

        language_code = get_language_code(language_key, provider)
        display_name = get_display_name(language_key)

        return True, f"{display_name} supported by {provider.upper()} ({language_code})"

    except Exception as e:
        error_msg = (
            f"System error validating language {language_key} for {provider}: {str(e)}"
        )
        logger.error(f"❌ {error_msg}", exc_info=True)
        raise gr.Error(error_msg, duration=5)


def get_language_status_update(selected_language: str) -> str:
    """Get status update for language selection.

    Args:
        selected_language: Currently selected language key

    Returns:
        Status message HTML
    """
    try:
        if not selected_language:
            return "Select a language for transcription"

        display_name = get_display_name(selected_language)

        # Check provider support
        aws_supported = is_language_supported(selected_language, "aws")
        azure_supported = is_language_supported(selected_language, "azure")

        if not aws_supported and not azure_supported:
            return f"⚠️ {display_name} is not supported by any provider"

        # Build support status
        support_status = []
        if aws_supported:
            support_status.append("AWS")
        if azure_supported:
            support_status.append("Azure")

        providers_text = " & ".join(support_status)
        return f"🌐 Language: {display_name} (supported by {providers_text}) - applies to next recording"

    except Exception as e:
        logger.error(f"❌ Error getting language status: {e}")
        return "Error checking language status"


def refresh_language_dropdown() -> gr.Dropdown:
    """Refresh the language dropdown with current language options.

    Returns:
        Updated Gradio dropdown component
    """
    try:
        from src.config.language_config import (
            get_default_language,
            get_language_choices,
        )

        choices = get_language_choices()
        default = get_default_language()

        logger.info(f"🔄 Refreshing language dropdown with {len(choices)} languages")

        return gr.Dropdown(choices=choices, value=default, interactive=True)

    except Exception as e:
        logger.error(f"❌ Error refreshing language dropdown: {e}")
        # Return a fallback dropdown
        return gr.Dropdown(
            choices=["English (US)"], value="English (US)", interactive=True
        )
