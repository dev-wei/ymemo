"""Provider configuration registry with dataclass-based system."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Set

from ..exceptions.provider_exceptions import (
    ProviderConfigurationError,
    ProviderNotAvailableError,
    ProviderRegistrationError,
)

logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    """Provider availability status."""

    READY = "ready"
    WARNING = "warning"
    ERROR = "error"
    NOT_IMPLEMENTED = "not_implemented"


class ProviderFeature(Enum):
    """Provider capability features."""

    STREAMING = "streaming"
    BATCH = "batch"
    SPEAKER_DIARIZATION = "speaker_diarization"
    REAL_TIME = "real_time"
    DUAL_CHANNEL = "dual_channel"
    CUSTOM_MODELS = "custom_models"
    MULTILINGUAL = "multilingual"
    OFFLINE = "offline"
    HIGH_ACCURACY = "high_accuracy"
    AUTO_PUNCTUATION = "auto_punctuation"


class ProviderLatency(Enum):
    """Provider latency classifications."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ProviderConfig:
    """Immutable configuration for a transcription provider."""

    # Basic identification
    key: str
    display_name: str
    description: str

    # Implementation status
    implemented: bool

    # Features and capabilities
    features: Set[ProviderFeature] = field(default_factory=set)
    supported_regions: List[str] = field(default_factory=list)
    supported_languages: Set[str] = field(default_factory=set)

    # Technical specifications
    max_audio_length: Optional[str] = None
    latency: ProviderLatency = ProviderLatency.MEDIUM

    # Requirements and dependencies
    requirements: List[str] = field(default_factory=list)
    status_check_function: Optional[str] = None

    # Metadata
    version: str = "1.0.0"
    vendor: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.key:
            raise ProviderConfigurationError(self.key, "Provider key cannot be empty")
        if not self.display_name:
            raise ProviderConfigurationError(self.key, "Display name cannot be empty")
        if self.implemented and not self.features:
            logger.warning(
                f"Provider '{self.key}' is marked as implemented but has no features"
            )

    def has_feature(self, feature: ProviderFeature) -> bool:
        """Check if provider supports a specific feature."""
        return feature in self.features

    def supports_region(self, region: str) -> bool:
        """Check if provider supports a specific region."""
        return not self.supported_regions or region in self.supported_regions

    def supports_language(self, language: str) -> bool:
        """Check if provider supports a specific language."""
        return not self.supported_languages or language in self.supported_languages

    def get_feature_list(self) -> List[str]:
        """Get list of feature names for display."""
        return [feature.value for feature in self.features]

    def is_real_time_capable(self) -> bool:
        """Check if provider supports real-time processing."""
        return (
            ProviderFeature.REAL_TIME in self.features
            and ProviderFeature.STREAMING in self.features
        )


@dataclass
class ProviderStatusInfo:
    """Provider status information with validation results."""

    status: ProviderStatus
    message: str
    icon: str
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: Optional[float] = None

    def is_available(self) -> bool:
        """Check if provider is ready for use."""
        return self.status == ProviderStatus.READY

    def has_warnings(self) -> bool:
        """Check if provider has warnings."""
        return self.status == ProviderStatus.WARNING

    def has_errors(self) -> bool:
        """Check if provider has errors."""
        return self.status == ProviderStatus.ERROR


class ProviderRegistry:
    """Registry for managing provider configurations."""

    def __init__(self):
        self._providers: Dict[str, ProviderConfig] = {}
        self._status_checkers: Dict[str, Callable[[str], ProviderStatusInfo]] = {}
        logger.info("🏭 ProviderRegistry: Initialized empty registry")

    def register_provider(
        self,
        config: ProviderConfig,
        status_checker: Optional[Callable[[str], ProviderStatusInfo]] = None,
    ) -> None:
        """Register a provider with optional status checker."""
        if config.key in self._providers:
            raise ProviderRegistrationError(
                config.key, f"Provider '{config.key}' is already registered"
            )

        self._providers[config.key] = config

        if status_checker:
            self._status_checkers[config.key] = status_checker

        logger.info(
            f"✅ ProviderRegistry: Registered provider '{config.key}' ({config.display_name})"
        )
        logger.debug(
            f"🔧 ProviderRegistry: Provider features: {config.get_feature_list()}"
        )

    def unregister_provider(self, provider_key: str) -> None:
        """Remove a provider from the registry."""
        if provider_key not in self._providers:
            logger.warning(
                f"⚠️ ProviderRegistry: Provider '{provider_key}' not found for unregistration"
            )
            return

        del self._providers[provider_key]
        self._status_checkers.pop(provider_key, None)
        logger.info(f"🗑️ ProviderRegistry: Unregistered provider '{provider_key}'")

    def get_provider(self, provider_key: str) -> ProviderConfig:
        """Get provider configuration by key."""
        if provider_key not in self._providers:
            raise ProviderNotAvailableError(provider_key, "Provider not registered")
        return self._providers[provider_key]

    def list_providers(
        self, include_unimplemented: bool = False
    ) -> List[ProviderConfig]:
        """Get list of provider configurations."""
        providers = list(self._providers.values())
        if not include_unimplemented:
            providers = [p for p in providers if p.implemented]
        return providers

    def list_provider_keys(self, include_unimplemented: bool = False) -> List[str]:
        """Get list of provider keys."""
        return [p.key for p in self.list_providers(include_unimplemented)]

    def get_providers_by_feature(
        self, feature: ProviderFeature
    ) -> List[ProviderConfig]:
        """Get providers that support a specific feature."""
        return [p for p in self.list_providers() if p.has_feature(feature)]

    def get_providers_by_region(self, region: str) -> List[ProviderConfig]:
        """Get providers that support a specific region."""
        return [p for p in self.list_providers() if p.supports_region(region)]

    @lru_cache(maxsize=32)
    def check_provider_status(self, provider_key: str) -> ProviderStatusInfo:
        """Check provider status with caching."""
        try:
            if provider_key not in self._providers:
                return ProviderStatusInfo(
                    status=ProviderStatus.ERROR,
                    message=f"Unknown provider: {provider_key}",
                    icon="❌",
                )

            provider_config = self._providers[provider_key]

            # Check if implemented
            if not provider_config.implemented:
                return ProviderStatusInfo(
                    status=ProviderStatus.NOT_IMPLEMENTED,
                    message="Coming soon - not yet implemented",
                    icon="🚧",
                )

            # Use custom status checker if available
            if provider_key in self._status_checkers:
                return self._status_checkers[provider_key](provider_key)

            # Default to ready if no specific checker
            return ProviderStatusInfo(
                status=ProviderStatus.READY, message="Provider available", icon="✅"
            )

        except Exception as e:
            logger.error(
                f"❌ ProviderRegistry: Status check failed for '{provider_key}': {e}"
            )
            return ProviderStatusInfo(
                status=ProviderStatus.ERROR,
                message=f"Status check failed: {str(e)[:50]}...",
                icon="❌",
            )

    def clear_status_cache(self) -> None:
        """Clear the status check cache."""
        self.check_provider_status.cache_clear()
        logger.info("🧹 ProviderRegistry: Status cache cleared")

    def validate_registry(self) -> List[str]:
        """Validate all registered providers and return list of issues."""
        issues = []

        for provider_key, config in self._providers.items():
            try:
                # Check for duplicate display names
                display_names = [p.display_name for p in self._providers.values()]
                if display_names.count(config.display_name) > 1:
                    issues.append(
                        f"Duplicate display name '{config.display_name}' for provider '{provider_key}'"
                    )

                # Validate implemented providers have required features
                if config.implemented and not config.features:
                    issues.append(
                        f"Implemented provider '{provider_key}' has no defined features"
                    )

                # Check status checker availability for implemented providers
                if (
                    config.implemented
                    and config.status_check_function
                    and provider_key not in self._status_checkers
                ):
                    issues.append(
                        f"Provider '{provider_key}' specifies status check but no checker registered"
                    )

            except Exception as e:
                issues.append(f"Validation error for provider '{provider_key}': {e}")

        return issues

    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary of registry state."""
        providers = list(self._providers.values())
        implemented = [p for p in providers if p.implemented]

        return {
            "total_providers": len(providers),
            "implemented_providers": len(implemented),
            "unimplemented_providers": len(providers) - len(implemented),
            "providers_with_status_checkers": len(self._status_checkers),
            "available_features": list(set().union(*(p.features for p in implemented))),
            "supported_regions": list(
                set().union(
                    *(
                        set(p.supported_regions)
                        for p in implemented
                        if p.supported_regions
                    )
                )
            ),
        }


# Global registry instance
_registry = ProviderRegistry()


def get_registry() -> ProviderRegistry:
    """Get the global provider registry instance."""
    return _registry
