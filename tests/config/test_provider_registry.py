"""Comprehensive unit tests for the provider registry system."""

from dataclasses import FrozenInstanceError
from typing import Set
from unittest.mock import MagicMock, patch

import pytest

from src.config.provider_registry import (
    ProviderConfig,
    ProviderFeature,
    ProviderLatency,
    ProviderRegistry,
    ProviderStatus,
    ProviderStatusInfo,
    get_registry,
)
from src.exceptions.provider_exceptions import (
    ProviderConfigurationError,
    ProviderNotAvailableError,
    ProviderRegistrationError,
)


class TestProviderConfig:
    """Test the ProviderConfig dataclass."""

    def test_basic_provider_config_creation(self):
        """Test creating a basic provider configuration."""
        config = ProviderConfig(
            key="test",
            display_name="Test Provider",
            description="A test provider",
            implemented=True,
        )

        assert config.key == "test"
        assert config.display_name == "Test Provider"
        assert config.description == "A test provider"
        assert config.implemented is True
        assert config.version == "1.0.0"  # Default value
        assert config.latency == ProviderLatency.MEDIUM  # Default value
        assert len(config.features) == 0  # Default empty set

    def test_provider_config_with_features(self):
        """Test provider config with features."""
        features = {ProviderFeature.STREAMING, ProviderFeature.REAL_TIME}
        config = ProviderConfig(
            key="aws",
            display_name="AWS Transcribe",
            description="Amazon transcription",
            implemented=True,
            features=features,
            supported_regions=["us-east-1", "us-west-2"],
            requirements=["AWS credentials"],
            latency=ProviderLatency.LOW,
        )

        assert config.features == features
        assert config.supported_regions == ["us-east-1", "us-west-2"]
        assert config.requirements == ["AWS credentials"]
        assert config.latency == ProviderLatency.LOW

    def test_provider_config_immutability(self):
        """Test that ProviderConfig is immutable (frozen)."""
        config = ProviderConfig(
            key="test",
            display_name="Test Provider",
            description="A test provider",
            implemented=True,
        )

        with pytest.raises(FrozenInstanceError):
            config.key = "modified"

    def test_provider_config_validation_empty_key(self):
        """Test validation fails with empty key."""
        with pytest.raises(ProviderConfigurationError) as exc_info:
            ProviderConfig(
                key="",
                display_name="Test Provider",
                description="A test provider",
                implemented=True,
            )
        assert "Provider key cannot be empty" in str(exc_info.value)

    def test_provider_config_validation_empty_display_name(self):
        """Test validation fails with empty display name."""
        with pytest.raises(ProviderConfigurationError) as exc_info:
            ProviderConfig(
                key="test",
                display_name="",
                description="A test provider",
                implemented=True,
            )
        assert "Display name cannot be empty" in str(exc_info.value)

    def test_has_feature_method(self):
        """Test the has_feature method."""
        features = {ProviderFeature.STREAMING, ProviderFeature.REAL_TIME}
        config = ProviderConfig(
            key="test",
            display_name="Test Provider",
            description="A test provider",
            implemented=True,
            features=features,
        )

        assert config.has_feature(ProviderFeature.STREAMING) is True
        assert config.has_feature(ProviderFeature.REAL_TIME) is True
        assert config.has_feature(ProviderFeature.BATCH) is False

    def test_supports_region_method(self):
        """Test the supports_region method."""
        config = ProviderConfig(
            key="test",
            display_name="Test Provider",
            description="A test provider",
            implemented=True,
            supported_regions=["us-east-1", "us-west-2"],
        )

        assert config.supports_region("us-east-1") is True
        assert config.supports_region("us-west-2") is True
        assert config.supports_region("eu-west-1") is False

        # Test empty regions (should support all)
        config_all = ProviderConfig(
            key="test2",
            display_name="Test Provider 2",
            description="Supports all regions",
            implemented=True,
        )
        assert config_all.supports_region("any-region") is True

    def test_supports_language_method(self):
        """Test the supports_language method."""
        config = ProviderConfig(
            key="test",
            display_name="Test Provider",
            description="A test provider",
            implemented=True,
            supported_languages={"en-US", "es-ES"},
        )

        assert config.supports_language("en-US") is True
        assert config.supports_language("es-ES") is True
        assert config.supports_language("fr-FR") is False

        # Test empty languages (should support all)
        config_all = ProviderConfig(
            key="test2",
            display_name="Test Provider 2",
            description="Supports all languages",
            implemented=True,
        )
        assert config_all.supports_language("any-language") is True

    def test_get_feature_list_method(self):
        """Test the get_feature_list method."""
        features = {ProviderFeature.STREAMING, ProviderFeature.REAL_TIME}
        config = ProviderConfig(
            key="test",
            display_name="Test Provider",
            description="A test provider",
            implemented=True,
            features=features,
        )

        feature_list = config.get_feature_list()
        assert isinstance(feature_list, list)
        assert len(feature_list) == 2
        assert "streaming" in feature_list
        assert "real_time" in feature_list

    def test_is_real_time_capable_method(self):
        """Test the is_real_time_capable method."""
        # Real-time capable (has both REAL_TIME and STREAMING)
        config_capable = ProviderConfig(
            key="test",
            display_name="Test Provider",
            description="A test provider",
            implemented=True,
            features={ProviderFeature.REAL_TIME, ProviderFeature.STREAMING},
        )
        assert config_capable.is_real_time_capable() is True

        # Not real-time capable (missing STREAMING)
        config_not_capable = ProviderConfig(
            key="test2",
            display_name="Test Provider 2",
            description="Not real-time",
            implemented=True,
            features={ProviderFeature.REAL_TIME, ProviderFeature.BATCH},
        )
        assert config_not_capable.is_real_time_capable() is False


class TestProviderStatusInfo:
    """Test the ProviderStatusInfo dataclass."""

    def test_provider_status_info_creation(self):
        """Test creating provider status info."""
        status_info = ProviderStatusInfo(
            status=ProviderStatus.READY,
            message="Provider is ready",
            icon="✅",
        )

        assert status_info.status == ProviderStatus.READY
        assert status_info.message == "Provider is ready"
        assert status_info.icon == "✅"
        assert status_info.details == {}  # Default empty dict
        assert status_info.checked_at is None

    def test_is_available_method(self):
        """Test the is_available method."""
        ready_status = ProviderStatusInfo(
            status=ProviderStatus.READY,
            message="Ready",
            icon="✅",
        )
        assert ready_status.is_available() is True

        error_status = ProviderStatusInfo(
            status=ProviderStatus.ERROR,
            message="Error",
            icon="❌",
        )
        assert error_status.is_available() is False

    def test_has_warnings_method(self):
        """Test the has_warnings method."""
        warning_status = ProviderStatusInfo(
            status=ProviderStatus.WARNING,
            message="Warning",
            icon="⚠️",
        )
        assert warning_status.has_warnings() is True

        ready_status = ProviderStatusInfo(
            status=ProviderStatus.READY,
            message="Ready",
            icon="✅",
        )
        assert ready_status.has_warnings() is False

    def test_has_errors_method(self):
        """Test the has_errors method."""
        error_status = ProviderStatusInfo(
            status=ProviderStatus.ERROR,
            message="Error",
            icon="❌",
        )
        assert error_status.has_errors() is True

        ready_status = ProviderStatusInfo(
            status=ProviderStatus.READY,
            message="Ready",
            icon="✅",
        )
        assert ready_status.has_errors() is False


class TestProviderRegistry:
    """Test the ProviderRegistry class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = ProviderRegistry()

        self.aws_config = ProviderConfig(
            key="aws",
            display_name="AWS Transcribe",
            description="Amazon transcription service",
            implemented=True,
            features={ProviderFeature.STREAMING, ProviderFeature.REAL_TIME},
            supported_regions=["us-east-1", "us-west-2"],
            requirements=["AWS credentials"],
        )

        self.azure_config = ProviderConfig(
            key="azure",
            display_name="Azure Speech Service",
            description="Microsoft transcription service",
            implemented=True,
            features={ProviderFeature.STREAMING, ProviderFeature.SPEAKER_DIARIZATION},
            supported_regions=["eastus", "westus2"],
            requirements=["Azure API key"],
        )

        self.whisper_config = ProviderConfig(
            key="whisper",
            display_name="OpenAI Whisper",
            description="OpenAI offline transcription",
            implemented=False,
            features={ProviderFeature.BATCH, ProviderFeature.OFFLINE},
            requirements=["Local model files"],
        )

    def test_registry_initialization(self):
        """Test registry is properly initialized."""
        assert len(self.registry._providers) == 0
        assert len(self.registry._status_checkers) == 0

    def test_register_provider_success(self):
        """Test successful provider registration."""

        # Mock status checker
        def mock_status_checker(provider_key: str) -> ProviderStatusInfo:
            return ProviderStatusInfo(
                status=ProviderStatus.READY,
                message="Mock status",
                icon="✅",
            )

        self.registry.register_provider(self.aws_config, mock_status_checker)

        assert len(self.registry._providers) == 1
        assert "aws" in self.registry._providers
        assert self.registry._providers["aws"] == self.aws_config
        assert "aws" in self.registry._status_checkers

    def test_register_provider_without_status_checker(self):
        """Test registering provider without status checker."""
        self.registry.register_provider(self.whisper_config)

        assert len(self.registry._providers) == 1
        assert "whisper" in self.registry._providers
        assert "whisper" not in self.registry._status_checkers

    def test_register_provider_duplicate_key_error(self):
        """Test error when registering duplicate provider key."""
        self.registry.register_provider(self.aws_config)

        with pytest.raises(ProviderRegistrationError) as exc_info:
            self.registry.register_provider(self.aws_config)

        assert "already registered" in str(exc_info.value)
        assert exc_info.value.provider_key == "aws"

    def test_unregister_provider_success(self):
        """Test successful provider unregistration."""
        self.registry.register_provider(self.aws_config)
        assert len(self.registry._providers) == 1

        self.registry.unregister_provider("aws")
        assert len(self.registry._providers) == 0
        assert "aws" not in self.registry._status_checkers

    def test_unregister_nonexistent_provider(self):
        """Test unregistering a provider that doesn't exist."""
        # Should not raise an error, just log a warning
        self.registry.unregister_provider("nonexistent")
        assert len(self.registry._providers) == 0

    def test_get_provider_success(self):
        """Test successfully getting a provider."""
        self.registry.register_provider(self.aws_config)

        retrieved_config = self.registry.get_provider("aws")
        assert retrieved_config == self.aws_config

    def test_get_provider_not_found_error(self):
        """Test error when getting nonexistent provider."""
        with pytest.raises(ProviderNotAvailableError) as exc_info:
            self.registry.get_provider("nonexistent")

        assert "Provider not registered" in str(exc_info.value)
        assert exc_info.value.provider_key == "nonexistent"

    def test_list_providers(self):
        """Test listing providers."""
        self.registry.register_provider(self.aws_config)
        self.registry.register_provider(self.azure_config)
        self.registry.register_provider(self.whisper_config)

        # Test including unimplemented
        all_providers = self.registry.list_providers(include_unimplemented=True)
        assert len(all_providers) == 3

        # Test excluding unimplemented
        implemented_only = self.registry.list_providers(include_unimplemented=False)
        assert len(implemented_only) == 2

        implemented_keys = [p.key for p in implemented_only]
        assert "aws" in implemented_keys
        assert "azure" in implemented_keys
        assert "whisper" not in implemented_keys

    def test_list_provider_keys(self):
        """Test listing provider keys."""
        self.registry.register_provider(self.aws_config)
        self.registry.register_provider(self.whisper_config)

        # Test including unimplemented
        all_keys = self.registry.list_provider_keys(include_unimplemented=True)
        assert len(all_keys) == 2
        assert "aws" in all_keys
        assert "whisper" in all_keys

        # Test excluding unimplemented
        implemented_keys = self.registry.list_provider_keys(include_unimplemented=False)
        assert len(implemented_keys) == 1
        assert "aws" in implemented_keys
        assert "whisper" not in implemented_keys

    def test_get_providers_by_feature(self):
        """Test getting providers by feature."""
        self.registry.register_provider(self.aws_config)
        self.registry.register_provider(self.azure_config)
        self.registry.register_provider(self.whisper_config)

        streaming_providers = self.registry.get_providers_by_feature(
            ProviderFeature.STREAMING
        )
        assert (
            len(streaming_providers) == 2
        )  # AWS and Azure (Whisper is not implemented)

        streaming_keys = [p.key for p in streaming_providers]
        assert "aws" in streaming_keys
        assert "azure" in streaming_keys
        assert "whisper" not in streaming_keys  # Not implemented

        offline_providers = self.registry.get_providers_by_feature(
            ProviderFeature.OFFLINE
        )
        assert len(offline_providers) == 0  # Whisper has OFFLINE but is not implemented

    def test_get_providers_by_region(self):
        """Test getting providers by region."""
        self.registry.register_provider(self.aws_config)
        self.registry.register_provider(self.azure_config)

        us_east_providers = self.registry.get_providers_by_region("us-east-1")
        assert len(us_east_providers) == 1
        assert us_east_providers[0].key == "aws"

        eastus_providers = self.registry.get_providers_by_region("eastus")
        assert len(eastus_providers) == 1
        assert eastus_providers[0].key == "azure"

    @patch('src.config.provider_registry.logger')
    def test_check_provider_status_with_custom_checker(self, mock_logger):
        """Test provider status check with custom checker."""

        def mock_status_checker(provider_key: str) -> ProviderStatusInfo:
            return ProviderStatusInfo(
                status=ProviderStatus.READY,
                message=f"Mock status for {provider_key}",
                icon="✅",
            )

        self.registry.register_provider(self.aws_config, mock_status_checker)

        status_info = self.registry.check_provider_status("aws")
        assert status_info.status == ProviderStatus.READY
        assert "Mock status for aws" in status_info.message
        assert status_info.icon == "✅"

    def test_check_provider_status_not_implemented(self):
        """Test status check for unimplemented provider."""
        self.registry.register_provider(self.whisper_config)

        status_info = self.registry.check_provider_status("whisper")
        assert status_info.status == ProviderStatus.NOT_IMPLEMENTED
        assert "Coming soon" in status_info.message
        assert status_info.icon == "🚧"

    def test_check_provider_status_unknown_provider(self):
        """Test status check for unknown provider."""
        status_info = self.registry.check_provider_status("unknown")
        assert status_info.status == ProviderStatus.ERROR
        assert "Unknown provider" in status_info.message
        assert status_info.icon == "❌"

    def test_check_provider_status_default_ready(self):
        """Test status check for provider without custom checker."""
        self.registry.register_provider(self.aws_config)  # No status checker

        status_info = self.registry.check_provider_status("aws")
        assert status_info.status == ProviderStatus.READY
        assert status_info.message == "Provider available"
        assert status_info.icon == "✅"

    def test_clear_status_cache(self):
        """Test clearing the status cache."""
        self.registry.register_provider(self.aws_config)

        # Call check_provider_status to populate cache
        self.registry.check_provider_status("aws")

        # Clear cache (should not raise error)
        self.registry.clear_status_cache()

    def test_validate_registry(self):
        """Test registry validation."""
        # Register providers with potential issues
        self.registry.register_provider(self.aws_config)

        # Create a duplicate display name issue
        duplicate_display_config = ProviderConfig(
            key="aws_duplicate",
            display_name="AWS Transcribe",  # Same as aws_config
            description="Duplicate display name",
            implemented=True,
        )
        self.registry.register_provider(duplicate_display_config)

        issues = self.registry.validate_registry()
        assert len(issues) > 0

        # Check for duplicate display name issue
        duplicate_issues = [
            issue for issue in issues if "Duplicate display name" in issue
        ]
        assert len(duplicate_issues) > 0

    def test_get_registry_summary(self):
        """Test getting registry summary."""
        self.registry.register_provider(self.aws_config)
        self.registry.register_provider(self.azure_config)
        self.registry.register_provider(self.whisper_config)

        summary = self.registry.get_registry_summary()

        assert summary["total_providers"] == 3
        assert summary["implemented_providers"] == 2
        assert summary["unimplemented_providers"] == 1
        assert summary["providers_with_status_checkers"] == 0

        # Check that features and regions are collected
        assert len(summary["available_features"]) > 0
        assert len(summary["supported_regions"]) > 0


class TestRegistryGlobalInstance:
    """Test the global registry instance."""

    def test_get_registry_singleton(self):
        """Test that get_registry returns the same instance."""
        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2
        assert isinstance(registry1, ProviderRegistry)
