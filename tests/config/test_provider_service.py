"""Comprehensive unit tests for the provider service layer."""

import time
from typing import Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.config.provider_registry import (
    ProviderConfig,
    ProviderFeature,
    ProviderLatency,
    ProviderRegistry,
    ProviderStatus,
    ProviderStatusInfo,
)
from src.exceptions.provider_exceptions import (
    ProviderConfigurationError,
    ProviderCredentialsError,
    ProviderNotAvailableError,
)
from src.services.provider_service import ProviderService, get_provider_service


class TestProviderService:
    """Test the ProviderService class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_registry = Mock(spec=ProviderRegistry)
        self.service = ProviderService(registry=self.mock_registry)

        # Create sample provider configs
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

    def test_service_initialization(self):
        """Test service is properly initialized."""
        assert self.service._registry == self.mock_registry
        assert isinstance(self.service._status_cache, dict)
        assert self.service._cache_ttl == 30.0

    def test_service_initialization_with_default_registry(self):
        """Test service initialization with default registry."""
        with patch('src.services.provider_service.get_registry') as mock_get_registry:
            mock_default_registry = Mock(spec=ProviderRegistry)
            mock_get_registry.return_value = mock_default_registry

            service = ProviderService()
            assert service._registry == mock_default_registry

    def test_get_available_providers_all_ready(self):
        """Test getting available providers when all are ready."""
        # Mock registry to return sample providers
        self.mock_registry.list_providers.return_value = [
            self.aws_config,
            self.azure_config,
        ]

        # Mock status checks
        def mock_check_status(provider_key):
            return ProviderStatusInfo(
                status=ProviderStatus.READY,
                message=f"{provider_key} is ready",
                icon="✅",
            )

        self.mock_registry.check_provider_status.side_effect = mock_check_status

        providers = self.service.get_available_providers()

        assert len(providers) == 2
        assert providers[0]["key"] == "aws"
        assert providers[0]["display_name"] == "AWS Transcribe"
        assert providers[0]["status"] == "ready"
        assert providers[1]["key"] == "azure"
        assert providers[1]["display_name"] == "Azure Speech Service"

    def test_get_available_providers_exclude_errors(self):
        """Test that providers with errors are excluded."""
        self.mock_registry.list_providers.return_value = [
            self.aws_config,
            self.azure_config,
        ]

        def mock_check_status(provider_key):
            if provider_key == "aws":
                return ProviderStatusInfo(
                    status=ProviderStatus.READY,
                    message="AWS is ready",
                    icon="✅",
                )
            else:  # azure
                return ProviderStatusInfo(
                    status=ProviderStatus.ERROR,
                    message="Azure has error",
                    icon="❌",
                )

        self.mock_registry.check_provider_status.side_effect = mock_check_status

        providers = self.service.get_available_providers()

        assert len(providers) == 1
        assert providers[0]["key"] == "aws"

    def test_get_available_providers_exclude_warnings(self):
        """Test excluding providers with warnings."""
        self.mock_registry.list_providers.return_value = [
            self.aws_config,
            self.azure_config,
        ]

        def mock_check_status(provider_key):
            if provider_key == "aws":
                return ProviderStatusInfo(
                    status=ProviderStatus.READY,
                    message="AWS is ready",
                    icon="✅",
                )
            else:  # azure
                return ProviderStatusInfo(
                    status=ProviderStatus.WARNING,
                    message="Azure has warning",
                    icon="⚠️",
                )

        self.mock_registry.check_provider_status.side_effect = mock_check_status

        # Exclude warnings
        providers = self.service.get_available_providers(include_warnings=False)
        assert len(providers) == 1
        assert providers[0]["key"] == "aws"

        # Include warnings
        providers = self.service.get_available_providers(include_warnings=True)
        assert len(providers) == 2

    def test_get_provider_choices(self):
        """Test getting provider choices for UI dropdown."""
        self.mock_registry.list_providers.return_value = [
            self.aws_config,
            self.azure_config,
        ]

        def mock_check_status(provider_key):
            return ProviderStatusInfo(
                status=ProviderStatus.READY,
                message=f"{provider_key} is ready",
                icon="✅",
            )

        self.mock_registry.check_provider_status.side_effect = mock_check_status

        choices = self.service.get_provider_choices()

        assert len(choices) == 2
        assert "AWS Transcribe" in choices
        assert "Azure Speech Service" in choices

    def test_get_provider_by_display_name_success(self):
        """Test successfully finding provider by display name."""
        self.mock_registry.list_providers.return_value = [
            self.aws_config,
            self.azure_config,
        ]

        provider = self.service.get_provider_by_display_name("AWS Transcribe")
        assert provider == self.aws_config

        provider = self.service.get_provider_by_display_name("Azure Speech Service")
        assert provider == self.azure_config

    def test_get_provider_by_display_name_not_found(self):
        """Test getting provider by display name when not found."""
        self.mock_registry.list_providers.return_value = [self.aws_config]

        provider = self.service.get_provider_by_display_name("Unknown Provider")
        assert provider is None

    def test_get_provider_key_from_display_name(self):
        """Test converting display name to provider key."""
        self.mock_registry.list_providers.return_value = [
            self.aws_config,
            self.azure_config,
        ]

        key = self.service.get_provider_key_from_display_name("AWS Transcribe")
        assert key == "aws"

        key = self.service.get_provider_key_from_display_name("Azure Speech Service")
        assert key == "azure"

        key = self.service.get_provider_key_from_display_name("Unknown Provider")
        assert key is None

    def test_get_display_name_from_key_success(self):
        """Test converting provider key to display name."""
        self.mock_registry.get_provider.return_value = self.aws_config

        display_name = self.service.get_display_name_from_key("aws")
        assert display_name == "AWS Transcribe"

    def test_get_display_name_from_key_not_found(self):
        """Test getting display name for unknown key."""
        self.mock_registry.get_provider.side_effect = ProviderNotAvailableError(
            "unknown", "Not found"
        )

        display_name = self.service.get_display_name_from_key("unknown")
        assert display_name == "UNKNOWN"  # Fallback behavior

    def test_validate_provider_switch_success(self):
        """Test successful provider switch validation."""
        self.mock_registry.list_providers.return_value = [self.aws_config]

        def mock_check_status(provider_key):
            return ProviderStatusInfo(
                status=ProviderStatus.READY,
                message="Ready",
                icon="✅",
            )

        self.mock_registry.check_provider_status.side_effect = mock_check_status

        is_valid, reason, status_info = self.service.validate_provider_switch(
            "AWS Transcribe"
        )

        assert is_valid is True
        assert reason == "Switch allowed"
        assert status_info.status == ProviderStatus.READY

    def test_validate_provider_switch_empty_provider(self):
        """Test validation fails with empty provider."""
        is_valid, reason, status_info = self.service.validate_provider_switch("")

        assert is_valid is False
        assert reason == "No provider selected"
        assert status_info.status == ProviderStatus.ERROR

    def test_validate_provider_switch_unknown_provider(self):
        """Test validation fails with unknown provider."""
        self.mock_registry.list_providers.return_value = []

        is_valid, reason, status_info = self.service.validate_provider_switch(
            "Unknown Provider"
        )

        assert is_valid is False
        assert reason == "Unknown provider: Unknown Provider"
        assert status_info.status == ProviderStatus.ERROR

    def test_validate_provider_switch_error_status(self):
        """Test validation fails with provider error status."""
        self.mock_registry.list_providers.return_value = [self.aws_config]

        def mock_check_status(provider_key):
            return ProviderStatusInfo(
                status=ProviderStatus.ERROR,
                message="Provider has errors",
                icon="❌",
            )

        self.mock_registry.check_provider_status.side_effect = mock_check_status

        is_valid, reason, status_info = self.service.validate_provider_switch(
            "AWS Transcribe"
        )

        assert is_valid is False
        assert "Cannot switch to AWS Transcribe" in reason
        assert status_info.status == ProviderStatus.ERROR

    @patch.dict('os.environ', {'TRANSCRIPTION_PROVIDER': 'old_provider'})
    def test_update_environment_provider_success(self):
        """Test successful environment provider update."""
        self.mock_registry.list_providers.return_value = [self.aws_config]
        self.mock_registry.get_provider.return_value = self.aws_config

        with patch(
            'os.environ', {'TRANSCRIPTION_PROVIDER': 'old_provider'}
        ) as mock_env:
            success = self.service.update_environment_provider("AWS Transcribe")

            assert success is True
            assert mock_env['TRANSCRIPTION_PROVIDER'] == 'aws'

    def test_update_environment_provider_unknown_display_name(self):
        """Test environment update fails with unknown display name."""
        self.mock_registry.list_providers.return_value = []

        success = self.service.update_environment_provider("Unknown Provider")
        assert success is False

    def test_update_environment_provider_not_implemented(self):
        """Test environment update fails with unimplemented provider."""
        self.mock_registry.list_providers.return_value = [self.whisper_config]
        self.mock_registry.get_provider.return_value = self.whisper_config

        success = self.service.update_environment_provider("OpenAI Whisper")
        assert success is False

    def test_update_environment_provider_not_available(self):
        """Test environment update fails when provider not available."""
        self.mock_registry.list_providers.return_value = []
        self.mock_registry.get_provider.side_effect = ProviderNotAvailableError(
            "unknown", "Not found"
        )

        success = self.service.update_environment_provider("Unknown Provider")
        assert success is False

    @patch.dict('os.environ', {'TRANSCRIPTION_PROVIDER': 'azure'})
    def test_get_current_provider_from_env(self):
        """Test getting current provider from environment."""
        current = self.service.get_current_provider_from_env()
        assert current == 'azure'

    @patch.dict('os.environ', {}, clear=True)
    def test_get_current_provider_from_env_default(self):
        """Test getting current provider with default value."""
        current = self.service.get_current_provider_from_env()
        assert current == 'aws'  # Default value

    def test_get_provider_info_html_success(self):
        """Test generating provider info HTML."""
        self.mock_registry.get_provider.return_value = self.aws_config

        def mock_check_status(provider_key):
            return ProviderStatusInfo(
                status=ProviderStatus.READY,
                message="Provider ready",
                icon="✅",
            )

        self.mock_registry.check_provider_status.side_effect = mock_check_status

        html = self.service.get_provider_info_html("aws")

        assert "AWS Transcribe" in html
        assert "Amazon transcription service" in html
        assert "✅" in html
        assert "Provider ready" in html

    def test_get_provider_info_html_not_available(self):
        """Test generating HTML for unavailable provider."""
        self.mock_registry.get_provider.side_effect = ProviderNotAvailableError(
            "unknown", "Not found"
        )

        html = self.service.get_provider_info_html("unknown")
        assert "Unknown provider" in html

    def test_get_provider_requirements_success(self):
        """Test getting provider requirements."""
        self.mock_registry.get_provider.return_value = self.aws_config

        requirements = self.service.get_provider_requirements("aws")
        assert requirements == ["AWS credentials"]

    def test_get_provider_requirements_not_available(self):
        """Test getting requirements for unavailable provider."""
        self.mock_registry.get_provider.side_effect = ProviderNotAvailableError(
            "unknown", "Not found"
        )

        requirements = self.service.get_provider_requirements("unknown")
        assert requirements == ["Unknown provider"]

    def test_check_provider_language_compatibility_success(self):
        """Test checking provider language compatibility."""
        mock_config = Mock()
        mock_config.supports_language.return_value = True

        self.mock_registry.get_provider.return_value = mock_config

        is_compatible = self.service.check_provider_language_compatibility(
            "aws", "en-US"
        )
        assert is_compatible is True
        mock_config.supports_language.assert_called_once_with("en-US")

    def test_check_provider_language_compatibility_not_available(self):
        """Test language compatibility for unavailable provider."""
        self.mock_registry.get_provider.side_effect = ProviderNotAvailableError(
            "unknown", "Not found"
        )

        is_compatible = self.service.check_provider_language_compatibility(
            "unknown", "en-US"
        )
        assert is_compatible is False

    def test_get_providers_by_feature(self):
        """Test getting providers by feature."""
        mock_providers = [self.aws_config, self.azure_config]
        self.mock_registry.get_providers_by_feature.return_value = mock_providers

        provider_keys = self.service.get_providers_by_feature(ProviderFeature.STREAMING)

        assert provider_keys == ["aws", "azure"]
        self.mock_registry.get_providers_by_feature.assert_called_once_with(
            ProviderFeature.STREAMING
        )

    def test_clear_status_cache(self):
        """Test clearing status cache."""
        # Populate cache first
        self.service._status_cache["aws"] = (
            ProviderStatusInfo(ProviderStatus.READY, "Ready", "✅"),
            time.time(),
        )

        self.service.clear_status_cache()

        assert len(self.service._status_cache) == 0
        self.mock_registry.clear_status_cache.assert_called_once()

    def test_get_service_metrics(self):
        """Test getting service performance metrics."""
        mock_registry_summary = {
            "total_providers": 2,
            "implemented_providers": 2,
        }
        self.mock_registry.get_registry_summary.return_value = mock_registry_summary
        self.mock_registry.validate_registry.return_value = []

        metrics = self.service.get_service_metrics()

        assert "registry" in metrics
        assert "cache" in metrics
        assert "validation" in metrics
        assert metrics["registry"] == mock_registry_summary
        assert metrics["cache"]["status_cache_size"] == 0
        assert metrics["cache"]["cache_ttl_seconds"] == 30.0

    def test_cached_status_cache_hit(self):
        """Test status cache hit behavior."""
        # Mock a fresh status info
        mock_status = ProviderStatusInfo(ProviderStatus.READY, "Ready", "✅")
        self.mock_registry.check_provider_status.return_value = mock_status

        # First call should populate cache
        status1 = self.service._get_cached_status("aws")

        # Second call should use cache (registry should not be called again)
        self.mock_registry.check_provider_status.reset_mock()
        status2 = self.service._get_cached_status("aws")

        assert status1 == status2
        self.mock_registry.check_provider_status.assert_not_called()

    def test_cached_status_cache_miss_expired(self):
        """Test status cache miss due to expiration."""
        mock_status = ProviderStatusInfo(ProviderStatus.READY, "Ready", "✅")
        self.mock_registry.check_provider_status.return_value = mock_status

        # Manually add an expired entry
        expired_time = time.time() - self.service._cache_ttl - 1
        self.service._status_cache["aws"] = (mock_status, expired_time)

        # Call should fetch fresh status
        status = self.service._get_cached_status("aws")

        assert status == mock_status
        self.mock_registry.check_provider_status.assert_called_once_with("aws")

    def test_invalidate_status_cache_specific_provider(self):
        """Test invalidating cache for specific provider."""
        # Populate cache
        self.service._status_cache["aws"] = (
            ProviderStatusInfo(ProviderStatus.READY, "Ready", "✅"),
            time.time(),
        )
        self.service._status_cache["azure"] = (
            ProviderStatusInfo(ProviderStatus.READY, "Ready", "✅"),
            time.time(),
        )

        # Invalidate specific provider
        self.service._invalidate_status_cache("aws")

        assert "aws" not in self.service._status_cache
        assert "azure" in self.service._status_cache


class TestProviderServiceGlobalInstance:
    """Test the global provider service instance."""

    def test_get_provider_service_singleton(self):
        """Test that get_provider_service returns the same instance."""
        service1 = get_provider_service()
        service2 = get_provider_service()

        assert service1 is service2
        assert isinstance(service1, ProviderService)

    @patch('src.services.provider_service._service')
    def test_get_provider_service_returns_global_instance(self, mock_service):
        """Test that get_provider_service returns the global _service instance."""
        mock_instance = Mock(spec=ProviderService)
        mock_service.__instance = mock_instance

        # We need to patch at the module level for this test
        with patch('src.services.provider_service._service', mock_instance):
            service = get_provider_service()
            assert service == mock_instance
