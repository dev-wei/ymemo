"""Provider selection event handlers for the UI interface.

Refactored to use the new service-based architecture with improved validation,
type safety, and error handling.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import gradio as gr

# Import provider_config first to trigger initialization of default providers
from src.config import provider_config  # This triggers _initialize_default_providers()
from src.config.provider_registry import ProviderStatus, ProviderStatusInfo
from src.exceptions.provider_exceptions import (
    ProviderConfigurationError,
    ProviderCredentialsError,
    ProviderNotAvailableError,
)
from src.services.provider_service import get_provider_service
from src.utils.status_manager import status_manager

logger = logging.getLogger(__name__)


@dataclass
class ProviderChangeResult:
    """Result of a provider change operation."""

    success: bool
    status_message: str
    confirmation_message: str
    provider_key: Optional[str] = None
    status_info: Optional[ProviderStatusInfo] = None
    warnings: Optional[list[str]] = None


class UIValidationError(Exception):
    """Raised when UI validation fails."""

    pass


def update_provider_configuration(provider_display_name: str) -> bool:
    """Update provider configuration using the service layer.

    Args:
        provider_display_name: Display name of selected provider (e.g., "AWS Transcribe")

    Returns:
        True if configuration was updated successfully, False otherwise
    """
    try:
        service = get_provider_service()
        success = service.update_environment_provider(provider_display_name)

        if success:
            provider_key = service.get_provider_key_from_display_name(
                provider_display_name
            )
            logger.info(f"✅ Provider configuration updated to: {provider_key}")
        else:
            logger.warning(
                f"⚠️ Failed to update provider configuration: {provider_display_name}"
            )

        return success

    except (ProviderNotAvailableError, ProviderConfigurationError) as e:
        logger.error(f"❌ Provider configuration error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error updating provider configuration: {e}")
        return False


def handle_provider_change(
    selected_provider_display: str, current_language: Optional[str] = None
) -> Tuple[str, str]:
    """Handle provider dropdown change event using service layer.

    Args:
        selected_provider_display: The selected provider display name
        current_language: Current language selection for compatibility checking

    Returns:
        Tuple of (status_message, confirmation_message)
    """
    try:
        logger.info(f"🔄 Provider change requested: {selected_provider_display}")

        result = _process_provider_change(selected_provider_display, current_language)

        # Update status manager based on result
        if result.success:
            status_manager.set_status(
                status_manager.current_status, result.status_message
            )
        else:
            error = UIValidationError(result.confirmation_message)
            status_manager.set_error(error, "Provider change failed")

        return status_manager.get_status_message(), result.confirmation_message

    except Exception as e:
        error_msg = f"Unexpected error changing provider: {str(e)}"
        logger.error(f"❌ {error_msg}")
        status_manager.set_error(e, "Provider change failed")
        return status_manager.get_status_message(), "❌ Provider change failed"


def _process_provider_change(
    selected_provider_display: str, current_language: Optional[str] = None
) -> ProviderChangeResult:
    """Process provider change with comprehensive validation.

    Args:
        selected_provider_display: The selected provider display name
        current_language: Current language selection for compatibility checking

    Returns:
        ProviderChangeResult with success status and messages
    """
    service = get_provider_service()

    # Validate selection
    if not selected_provider_display:
        return ProviderChangeResult(
            success=False,
            status_message="Provider selection required",
            confirmation_message="Provider selection required",
        )

    # Validate provider switch possibility
    is_valid, reason, status_info = service.validate_provider_switch(
        selected_provider_display, current_language
    )

    if not is_valid:
        logger.warning(f"⚠️ Provider switch validation failed: {reason}")
        return ProviderChangeResult(
            success=False,
            status_message=f"Provider validation failed: {reason}",
            confirmation_message=f"❌ {reason}",
            status_info=status_info,
        )

    # Update configuration
    config_updated = service.update_environment_provider(selected_provider_display)
    if not config_updated:
        error_msg = f"Failed to update configuration for {selected_provider_display}"
        logger.error(f"❌ {error_msg}")
        return ProviderChangeResult(
            success=False,
            status_message=error_msg,
            confirmation_message="❌ Configuration update failed",
        )

    # Success - generate messages based on status
    provider_key = service.get_provider_key_from_display_name(selected_provider_display)

    if status_info.status == ProviderStatus.WARNING:
        confirmation_msg = (
            f"⚠️ Switched to: {selected_provider_display} ({status_info.message})"
        )
        status_msg = (
            f"Provider: {selected_provider_display} (warning: {status_info.message})"
        )
    else:
        confirmation_msg = f"✅ Switched to: {selected_provider_display}"
        status_msg = f"Provider: {selected_provider_display}"

    confirmation_msg += " (will apply to next recording session)"
    logger.info(f"✅ Provider switched successfully to {provider_key}")

    return ProviderChangeResult(
        success=True,
        status_message=status_msg,
        confirmation_message=confirmation_msg,
        provider_key=provider_key,
        status_info=status_info,
    )


def get_current_provider_info(selected_provider_display: str) -> str:
    """Get detailed information about the currently selected provider using service layer.

    Args:
        selected_provider_display: The selected provider display name

    Returns:
        HTML formatted provider information
    """
    try:
        if not selected_provider_display:
            return "<span style='color: #666;'>No provider selected</span>"

        service = get_provider_service()

        # Convert display name to key
        provider_key = service.get_provider_key_from_display_name(
            selected_provider_display
        )
        if not provider_key:
            return "<span style='color: red;'>⚠️ Unknown provider</span>"

        # Get provider info HTML using service
        return service.get_provider_info_html(provider_key)

    except ProviderNotAvailableError:
        return "<span style='color: red;'>⚠️ Provider not available</span>"
    except Exception as e:
        logger.error(f"❌ Error getting provider info: {e}")
        return "<span style='color: red;'>Error getting provider info</span>"


def validate_provider_for_language(
    provider_display: str, language_key: str
) -> Tuple[bool, str]:
    """Validate if a provider supports a specific language using service layer.

    Args:
        provider_display: Provider display name
        language_key: Language key to validate

    Returns:
        Tuple of (is_supported, status_message)
    """
    try:
        service = get_provider_service()

        provider_key = service.get_provider_key_from_display_name(provider_display)
        if not provider_key:
            return False, f"Unknown provider: {provider_display}"

        is_compatible = service.check_provider_language_compatibility(
            provider_key, language_key
        )

        if not is_compatible:
            logger.warning(
                f"❌ Provider {provider_display} doesn't support {language_key}"
            )
            gr.Warning(
                f"{provider_display} doesn't support {language_key} ⚠️", duration=5
            )
            return False, f"{provider_display} doesn't support {language_key}"

        return True, f"{provider_display} supports {language_key}"

    except ProviderNotAvailableError:
        logger.warning(f"❌ Provider {provider_display} not available")
        gr.Warning(f"Provider {provider_display} not available ⚠️", duration=5)
        return False, f"Provider {provider_display} not available"
    except Exception as e:
        error_msg = f"System error validating provider compatibility: {str(e)}"
        logger.error(f"❌ {error_msg}", exc_info=True)
        raise gr.Error(error_msg, duration=5)


def get_provider_status_update(selected_provider_display: str) -> str:
    """Get status update for provider selection using service layer.

    Args:
        selected_provider_display: Currently selected provider display name

    Returns:
        Status message text with proper formatting
    """
    try:
        if not selected_provider_display:
            return "Select a transcription provider"

        service = get_provider_service()

        provider_key = service.get_provider_key_from_display_name(
            selected_provider_display
        )
        if not provider_key:
            return f"⚠️ Unknown provider: {selected_provider_display}"

        # Get status info from service (uses caching)
        status_info = service._get_cached_status(provider_key)

        status_icon = status_info.icon
        status_msg = status_info.message

        if status_info.status == ProviderStatus.READY:
            return f"{status_icon} Provider: {selected_provider_display} ({status_msg})"
        elif status_info.status == ProviderStatus.WARNING:
            return f"{status_icon} Provider: {selected_provider_display} - {status_msg}"
        elif status_info.status == ProviderStatus.ERROR:
            return f"{status_icon} Provider: {selected_provider_display} - {status_msg}"
        else:  # NOT_IMPLEMENTED
            return f"{status_icon} Provider: {selected_provider_display} - {status_msg}"

    except ProviderNotAvailableError:
        return f"⚠️ Provider {selected_provider_display} not available"
    except Exception as e:
        logger.error(f"❌ Error getting provider status: {e}")
        return "Error checking provider status"


def refresh_provider_dropdown() -> gr.Dropdown:
    """Refresh the provider dropdown with current provider options using service layer.

    Returns:
        Updated Gradio dropdown component
    """
    try:
        service = get_provider_service()

        choices = service.get_provider_choices()
        current_provider_key = service.get_current_provider_from_env()
        current_display_name = service.get_display_name_from_key(current_provider_key)

        logger.info(f"🔄 Refreshing provider dropdown with {len(choices)} providers")
        logger.info(
            f"🎯 Current provider: {current_display_name} ({current_provider_key})"
        )

        return gr.Dropdown(
            choices=choices, value=current_display_name, interactive=True
        )

    except Exception as e:
        logger.error(f"❌ Error refreshing provider dropdown: {e}")
        # Return a fallback dropdown with AWS Transcribe
        return gr.Dropdown(
            choices=["AWS Transcribe"], value="AWS Transcribe", interactive=True
        )


def get_provider_requirements_info(provider_display: str) -> str:
    """Get provider requirements information for display using service layer.

    Args:
        provider_display: Provider display name

    Returns:
        HTML formatted requirements information
    """
    try:
        service = get_provider_service()

        provider_key = service.get_provider_key_from_display_name(provider_display)
        if not provider_key:
            return "<span style='color: red;'>Unknown provider</span>"

        requirements = service.get_provider_requirements(provider_key)

        if not requirements:
            return "<span style='color: green;'>No special requirements</span>"

        req_list = "<br/>".join([f"• {req}" for req in requirements])

        return f"""
        <div style='font-size: 0.8em; color: #666;'>
            <strong>Requirements:</strong><br/>
            {req_list}
        </div>
        """

    except ProviderNotAvailableError:
        return "<span style='color: red;'>Provider not available</span>"
    except Exception as e:
        logger.error(f"❌ Error getting provider requirements: {e}")
        return "<span style='color: red;'>Error getting requirements</span>"


def get_provider_service_metrics() -> Dict[str, Any]:
    """Get service performance metrics for debugging and monitoring.

    Returns:
        Dictionary with service metrics and status information
    """
    try:
        service = get_provider_service()
        return service.get_service_metrics()
    except Exception as e:
        logger.error(f"❌ Error getting service metrics: {e}")
        return {"error": str(e)}


def clear_provider_status_cache() -> bool:
    """Clear the provider status cache to force fresh status checks.

    Returns:
        True if cache was cleared successfully
    """
    try:
        service = get_provider_service()
        service.clear_status_cache()
        logger.info("🧹 Provider status cache cleared successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error clearing provider status cache: {e}")
        return False


def get_available_provider_features() -> Dict[str, list[str]]:
    """Get features supported by all available providers.

    Returns:
        Dictionary mapping provider keys to their supported feature lists
    """
    try:
        service = get_provider_service()
        available_providers = service.get_available_providers(include_warnings=True)

        features_map = {}
        for provider in available_providers:
            provider_key = provider["key"]
            try:
                # Get provider config from registry to access features
                registry = service._registry
                config = registry.get_provider(provider_key)
                features_map[provider_key] = config.get_feature_list()
            except Exception:
                features_map[provider_key] = []

        return features_map
    except Exception as e:
        logger.error(f"❌ Error getting provider features: {e}")
        return {}


def validate_ui_handler_dependencies() -> Tuple[bool, list[str]]:
    """Validate that all required dependencies for UI handlers are available.

    Returns:
        Tuple of (all_valid, list_of_issues)
    """
    issues = []

    try:
        # Check service availability
        service = get_provider_service()

        # Check registry health
        registry = service._registry
        registry_issues = registry.validate_registry()
        if registry_issues:
            issues.extend([f"Registry: {issue}" for issue in registry_issues])

        # Check service metrics
        try:
            metrics = service.get_service_metrics()
            if "error" in metrics:
                issues.append(f"Service metrics error: {metrics['error']}")
        except Exception as e:
            issues.append(f"Cannot get service metrics: {e}")

        # Check basic functionality
        try:
            choices = service.get_provider_choices()
            if not choices:
                issues.append("No provider choices available")
        except Exception as e:
            issues.append(f"Cannot get provider choices: {e}")

        logger.info(f"🔍 UI handler validation complete: {len(issues)} issues found")
        return len(issues) == 0, issues

    except Exception as e:
        issues.append(f"Critical error during validation: {e}")
        logger.error(f"❌ UI handler validation failed: {e}")
        return False, issues
