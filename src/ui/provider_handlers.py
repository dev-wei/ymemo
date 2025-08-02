"""Provider selection event handlers for the UI interface."""

import logging
import os
from typing import Tuple

import gradio as gr

from src.config.provider_config import (
    ProviderStatus,
    check_provider_status,
    get_display_name_from_key,
    get_provider_info_html,
    get_provider_key_from_display_name,
    is_provider_language_compatible,
    validate_provider_availability,
)
from src.utils.status_manager import status_manager

logger = logging.getLogger(__name__)


def update_provider_configuration(provider_display_name: str) -> bool:
    """Update provider configuration in environment variables.

    Args:
        provider_display_name: Display name of selected provider (e.g., "AWS Transcribe")

    Returns:
        True if configuration was updated successfully, False otherwise
    """
    try:
        # Convert display name to provider key
        provider_key = get_provider_key_from_display_name(provider_display_name)
        if not provider_key:
            logger.warning(f"⚠️ Unknown provider display name: {provider_display_name}")
            return False

        # Validate provider availability
        if not validate_provider_availability(provider_key):
            logger.warning(f"⚠️ Provider not available: {provider_key}")
            return False

        # Update environment variable
        old_provider = os.environ.get('TRANSCRIPTION_PROVIDER', 'aws')
        os.environ['TRANSCRIPTION_PROVIDER'] = provider_key

        logger.info(
            f"🔄 Updated TRANSCRIPTION_PROVIDER: {old_provider} → {provider_key}"
        )
        return True

    except Exception as e:
        logger.error(f"❌ Error updating provider configuration: {e}")
        return False


def handle_provider_change(
    selected_provider_display: str, current_language: str = None
) -> Tuple[str, str]:
    """Handle provider dropdown change event.

    Args:
        selected_provider_display: The selected provider display name
        current_language: Current language selection for compatibility checking

    Returns:
        Tuple of (status_message, confirmation_message)
    """
    try:
        logger.info(f"🔄 Provider change requested: {selected_provider_display}")

        # Validate the selected provider
        if not selected_provider_display:
            error_msg = "No provider selected"
            logger.warning(f"⚠️ {error_msg}")
            status_manager.set_error(
                Exception(error_msg), "Provider selection required"
            )
            return status_manager.get_status_message(), "Provider selection required"

        # Convert display name to key
        provider_key = get_provider_key_from_display_name(selected_provider_display)
        if not provider_key:
            error_msg = f"Unknown provider: {selected_provider_display}"
            logger.error(f"❌ {error_msg}")
            status_manager.set_error(Exception(error_msg), "Invalid provider")
            return status_manager.get_status_message(), "❌ Invalid provider selection"

        # Check provider status
        provider_status = check_provider_status(provider_key)

        # If provider has errors, don't allow switching
        if provider_status["status"] == ProviderStatus.ERROR:
            error_msg = f"Cannot switch to {selected_provider_display}: {provider_status['message']}"
            logger.warning(f"⚠️ {error_msg}")
            status_manager.set_error(Exception(error_msg), "Provider not available")
            return (
                status_manager.get_status_message(),
                f"❌ {provider_status['message']}",
            )

        # Check language compatibility if language is provided
        if current_language and not is_provider_language_compatible(
            provider_key, current_language
        ):
            warning_msg = (
                f"{selected_provider_display} may not support {current_language}"
            )
            logger.warning(f"⚠️ {warning_msg}")

        # Update provider configuration
        config_updated = update_provider_configuration(selected_provider_display)
        if not config_updated:
            error_msg = (
                f"Failed to update configuration for {selected_provider_display}"
            )
            logger.error(f"❌ {error_msg}")
            status_manager.set_error(
                Exception(error_msg), "Configuration update failed"
            )
            return status_manager.get_status_message(), "❌ Configuration update failed"

        # Success message based on provider status
        if provider_status["status"] == ProviderStatus.WARNING:
            confirmation_msg = f"⚠️ Switched to: {selected_provider_display} ({provider_status['message']})"
            status_msg = f"Provider: {selected_provider_display} (warning: {provider_status['message']})"
        else:
            confirmation_msg = f"✅ Switched to: {selected_provider_display}"
            status_msg = f"Provider: {selected_provider_display}"

        confirmation_msg += " (will apply to next recording session)"

        logger.info(f"✅ Provider switched successfully to {provider_key}")

        status_manager.set_status(status_manager.current_status, status_msg)

        return status_manager.get_status_message(), confirmation_msg

    except Exception as e:
        error_msg = f"Error changing provider: {str(e)}"
        logger.error(f"❌ {error_msg}")
        status_manager.set_error(e, "Provider change failed")
        return status_manager.get_status_message(), "❌ Provider change failed"


def get_current_provider_info(selected_provider_display: str) -> str:
    """Get detailed information about the currently selected provider.

    Args:
        selected_provider_display: The selected provider display name

    Returns:
        HTML formatted provider information
    """
    try:
        if not selected_provider_display:
            return "<span style='color: #666;'>No provider selected</span>"

        # Convert display name to key
        provider_key = get_provider_key_from_display_name(selected_provider_display)
        if not provider_key:
            return "<span style='color: red;'>⚠️ Unknown provider</span>"

        # Get provider info HTML
        return get_provider_info_html(provider_key)

    except Exception as e:
        logger.error(f"❌ Error getting provider info: {e}")
        return "<span style='color: red;'>Error getting provider info</span>"


def validate_provider_for_language(
    provider_display: str, language_key: str
) -> Tuple[bool, str]:
    """Validate if a provider supports a specific language.

    Args:
        provider_display: Provider display name
        language_key: Language key to validate

    Returns:
        Tuple of (is_supported, status_message)
    """
    try:
        provider_key = get_provider_key_from_display_name(provider_display)
        if not provider_key:
            return False, f"Unknown provider: {provider_display}"

        if not is_provider_language_compatible(provider_key, language_key):
            return False, f"{provider_display} doesn't support {language_key}"

        return True, f"{provider_display} supports {language_key}"

    except Exception as e:
        logger.error(f"❌ Error validating provider language compatibility: {e}")
        return False, f"Validation error: {str(e)}"


def get_provider_status_update(selected_provider_display: str) -> str:
    """Get status update for provider selection.

    Args:
        selected_provider_display: Currently selected provider display name

    Returns:
        Status message text
    """
    try:
        if not selected_provider_display:
            return "Select a transcription provider"

        provider_key = get_provider_key_from_display_name(selected_provider_display)
        if not provider_key:
            return f"⚠️ Unknown provider: {selected_provider_display}"

        # Check provider status
        provider_status = check_provider_status(provider_key)

        status_icon = provider_status["icon"]
        status_msg = provider_status["message"]

        if provider_status["status"] == ProviderStatus.READY:
            return f"{status_icon} Provider: {selected_provider_display} ({status_msg})"
        elif provider_status["status"] == ProviderStatus.WARNING:
            return f"{status_icon} Provider: {selected_provider_display} - {status_msg}"
        elif provider_status["status"] == ProviderStatus.ERROR:
            return f"{status_icon} Provider: {selected_provider_display} - {status_msg}"
        else:  # NOT_IMPLEMENTED
            return f"{status_icon} Provider: {selected_provider_display} - {status_msg}"

    except Exception as e:
        logger.error(f"❌ Error getting provider status: {e}")
        return "Error checking provider status"


def refresh_provider_dropdown() -> gr.Dropdown:
    """Refresh the provider dropdown with current provider options.

    Returns:
        Updated Gradio dropdown component
    """
    try:
        from src.config.provider_config import (
            get_current_provider_from_env,
            get_display_name_from_key,
            get_provider_choices,
        )

        choices = get_provider_choices()
        current_provider_key = get_current_provider_from_env()
        current_display_name = get_display_name_from_key(current_provider_key)

        logger.info(f"🔄 Refreshing provider dropdown with {len(choices)} providers")
        logger.info(
            f"🎯 Current provider: {current_display_name} ({current_provider_key})"
        )

        return gr.Dropdown(
            choices=choices, value=current_display_name, interactive=True
        )

    except Exception as e:
        logger.error(f"❌ Error refreshing provider dropdown: {e}")
        # Return a fallback dropdown
        return gr.Dropdown(
            choices=["AWS Transcribe"], value="AWS Transcribe", interactive=True
        )


def get_provider_requirements_info(provider_display: str) -> str:
    """Get provider requirements information for display.

    Args:
        provider_display: Provider display name

    Returns:
        HTML formatted requirements information
    """
    try:
        provider_key = get_provider_key_from_display_name(provider_display)
        if not provider_key:
            return "<span style='color: red;'>Unknown provider</span>"

        from src.config.provider_config import get_provider_requirements

        requirements = get_provider_requirements(provider_key)

        if not requirements:
            return "<span style='color: green;'>No special requirements</span>"

        req_list = "<br/>".join([f"• {req}" for req in requirements])

        return f"""
        <div style='font-size: 0.8em; color: #666;'>
            <strong>Requirements:</strong><br/>
            {req_list}
        </div>
        """

    except Exception as e:
        logger.error(f"❌ Error getting provider requirements: {e}")
        return "<span style='color: red;'>Error getting requirements</span>"
