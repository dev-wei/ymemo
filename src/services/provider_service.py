"""Provider service facade with caching and validation."""

import logging
import os
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from ..config.provider_registry import (
    ProviderConfig,
    ProviderFeature,
    ProviderLatency,
    ProviderRegistry,
    ProviderStatus,
    ProviderStatusInfo,
    get_registry,
)

# Note: provider_config.py imports this module, so we avoid circular import
# The registry will be initialized when provider_config is imported elsewhere
from ..exceptions.provider_exceptions import (
    ProviderConfigurationError,
    ProviderCredentialsError,
    ProviderNotAvailableError,
    ProviderStatusCheckError,
)

logger = logging.getLogger(__name__)


class ProviderService:
    """High-level service for provider management with caching and validation."""

    def __init__(self, registry: Optional[ProviderRegistry] = None):
        self._registry = registry or get_registry()
        self._status_cache: Dict[str, Tuple[ProviderStatusInfo, float]] = {}
        self._cache_ttl = 30.0  # Cache status checks for 30 seconds
        logger.info("🚀 ProviderService: Initialized with caching enabled")

    def get_available_providers(
        self, include_warnings: bool = True
    ) -> List[Dict[str, str]]:
        """Get list of available providers for UI display."""
        providers = []

        for provider_config in self._registry.list_providers():
            status_info = self._get_cached_status(provider_config.key)

            # Skip providers with errors unless specifically requested
            if status_info.status == ProviderStatus.ERROR:
                continue

            # Skip providers with warnings if not requested
            if status_info.status == ProviderStatus.WARNING and not include_warnings:
                continue

            providers.append(
                {
                    "key": provider_config.key,
                    "display_name": provider_config.display_name,
                    "description": provider_config.description,
                    "status": status_info.status.value,
                    "status_message": status_info.message,
                    "status_icon": status_info.icon,
                }
            )

        logger.debug(f"🔍 ProviderService: Found {len(providers)} available providers")
        return providers

    def get_provider_choices(self) -> List[str]:
        """Get provider display names for dropdown UI."""
        available = self.get_available_providers(include_warnings=True)
        choices = [provider["display_name"] for provider in available]
        logger.debug(f"📋 ProviderService: Generated {len(choices)} provider choices")
        return choices

    def get_provider_by_display_name(
        self, display_name: str
    ) -> Optional[ProviderConfig]:
        """Find provider config by display name."""
        for provider_config in self._registry.list_providers():
            if provider_config.display_name == display_name:
                return provider_config
        return None

    def get_provider_key_from_display_name(self, display_name: str) -> Optional[str]:
        """Convert display name to provider key."""
        provider_config = self.get_provider_by_display_name(display_name)
        return provider_config.key if provider_config else None

    def get_display_name_from_key(self, provider_key: str) -> str:
        """Convert provider key to display name."""
        try:
            provider_config = self._registry.get_provider(provider_key)
            return provider_config.display_name
        except ProviderNotAvailableError:
            logger.warning(f"⚠️ ProviderService: Unknown provider key: {provider_key}")
            return provider_key.upper()

    def validate_provider_switch(
        self, target_provider_display: str, current_language: Optional[str] = None
    ) -> Tuple[bool, str, ProviderStatusInfo]:
        """Validate if provider switch is possible."""
        if not target_provider_display:
            return (
                False,
                "No provider selected",
                ProviderStatusInfo(
                    status=ProviderStatus.ERROR,
                    message="Provider selection required",
                    icon="❌",
                ),
            )

        # Get provider config
        provider_config = self.get_provider_by_display_name(target_provider_display)
        if not provider_config:
            return (
                False,
                f"Unknown provider: {target_provider_display}",
                ProviderStatusInfo(
                    status=ProviderStatus.ERROR, message="Invalid provider", icon="❌"
                ),
            )

        # Check provider status
        status_info = self._get_cached_status(provider_config.key)

        # Block switching to providers with errors
        if status_info.status == ProviderStatus.ERROR:
            return (
                False,
                f"Cannot switch to {target_provider_display}: {status_info.message}",
                status_info,
            )

        # Check language compatibility
        if current_language and not provider_config.supports_language(current_language):
            logger.warning(
                f"⚠️ ProviderService: {target_provider_display} may not support {current_language}"
            )

        return True, "Switch allowed", status_info

    def update_environment_provider(self, provider_display_name: str) -> bool:
        """Update the environment variable for provider selection."""
        provider_key = self.get_provider_key_from_display_name(provider_display_name)
        if not provider_key:
            logger.error(
                f"❌ ProviderService: Unknown provider display name: {provider_display_name}"
            )
            return False

        # Validate provider availability
        try:
            provider_config = self._registry.get_provider(provider_key)
            if not provider_config.implemented:
                logger.error(
                    f"❌ ProviderService: Provider not implemented: {provider_key}"
                )
                return False
        except ProviderNotAvailableError:
            logger.error(f"❌ ProviderService: Provider not available: {provider_key}")
            return False

        # Update environment variable
        old_provider = os.environ.get('TRANSCRIPTION_PROVIDER', 'aws')
        os.environ['TRANSCRIPTION_PROVIDER'] = provider_key

        logger.info(
            f"🔄 ProviderService: Updated TRANSCRIPTION_PROVIDER: {old_provider} → {provider_key}"
        )

        # Clear status cache for the new provider to ensure fresh status
        self._invalidate_status_cache(provider_key)

        return True

    def get_current_provider_from_env(self) -> str:
        """Get current provider from environment variable."""
        return os.environ.get('TRANSCRIPTION_PROVIDER', 'aws')

    def get_provider_info_html(self, provider_key: str) -> str:
        """Generate HTML information for a provider."""
        try:
            provider_config = self._registry.get_provider(provider_key)
            status_info = self._get_cached_status(provider_key)

            # Determine status color
            status_colors = {
                ProviderStatus.READY: "green",
                ProviderStatus.WARNING: "orange",
                ProviderStatus.ERROR: "red",
                ProviderStatus.NOT_IMPLEMENTED: "gray",
            }
            status_color = status_colors.get(status_info.status, "gray")

            # Format features (show first 3)
            features = provider_config.get_feature_list()[:3]
            features_text = ", ".join(features) if features else "None listed"

            return f"""
            <div style='font-size: 0.9em; color: #444;'>
                <strong>{provider_config.display_name}</strong><br/>
                <span style='color: #666;'>{provider_config.description}</span><br/>
                <span style='color: {status_color}; font-size: 0.8em;'>
                    {status_info.icon} {status_info.message}
                </span><br/>
                <span style='color: #888; font-size: 0.8em;'>
                    Features: {features_text}
                </span>
            </div>
            """

        except ProviderNotAvailableError:
            return "<span style='color: red;'>Unknown provider</span>"
        except Exception as e:
            logger.error(
                f"❌ ProviderService: Error generating provider info HTML: {e}"
            )
            return "<span style='color: red;'>Error getting provider info</span>"

    def get_provider_requirements(self, provider_key: str) -> List[str]:
        """Get list of requirements for a provider."""
        try:
            provider_config = self._registry.get_provider(provider_key)
            return provider_config.requirements
        except ProviderNotAvailableError:
            return ["Unknown provider"]

    def check_provider_language_compatibility(
        self, provider_key: str, language: str
    ) -> bool:
        """Check if provider supports a specific language."""
        try:
            provider_config = self._registry.get_provider(provider_key)
            return provider_config.supports_language(language)
        except ProviderNotAvailableError:
            return False

    def get_providers_by_feature(self, feature: ProviderFeature) -> List[str]:
        """Get provider keys that support a specific feature."""
        providers = self._registry.get_providers_by_feature(feature)
        return [p.key for p in providers]

    def clear_status_cache(self) -> None:
        """Clear all cached status information."""
        self._status_cache.clear()
        self._registry.clear_status_cache()
        logger.info("🧹 ProviderService: All caches cleared")

    def get_service_metrics(self) -> Dict[str, any]:
        """Get service performance metrics."""
        registry_summary = self._registry.get_registry_summary()

        return {
            "registry": registry_summary,
            "cache": {
                "status_cache_size": len(self._status_cache),
                "cache_ttl_seconds": self._cache_ttl,
            },
            "validation": {
                "registry_issues": len(self._registry.validate_registry()),
            },
        }

    def _get_cached_status(self, provider_key: str) -> ProviderStatusInfo:
        """Get provider status with caching."""
        current_time = time.time()

        # Check cache
        if provider_key in self._status_cache:
            cached_status, cached_time = self._status_cache[provider_key]
            if current_time - cached_time < self._cache_ttl:
                return cached_status

        # Get fresh status
        status_info = self._registry.check_provider_status(provider_key)
        status_info.checked_at = current_time

        # Cache the result
        self._status_cache[provider_key] = (status_info, current_time)

        logger.debug(
            f"🔍 ProviderService: Cached status for '{provider_key}': {status_info.status.value}"
        )
        return status_info

    def _invalidate_status_cache(self, provider_key: str) -> None:
        """Invalidate cache for a specific provider."""
        self._status_cache.pop(provider_key, None)
        logger.debug(
            f"🧹 ProviderService: Invalidated cache for provider '{provider_key}'"
        )


# Global service instance
_service = ProviderService()


def get_provider_service() -> ProviderService:
    """Get the global provider service instance."""
    return _service
