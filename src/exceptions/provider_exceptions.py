"""Custom exceptions for provider management system."""


class ProviderError(Exception):
    """Base exception for all provider-related errors."""

    def __init__(
        self, message: str, provider_key: str = None, original_error: Exception = None
    ):
        super().__init__(message)
        self.provider_key = provider_key
        self.original_error = original_error


class ProviderNotAvailableError(ProviderError):
    """Raised when a provider is not available or not implemented."""

    def __init__(self, provider_key: str, reason: str = None):
        message = f"Provider '{provider_key}' is not available"
        if reason:
            message += f": {reason}"
        super().__init__(message, provider_key)


class ProviderConfigurationError(ProviderError):
    """Raised when provider configuration is invalid or missing."""

    def __init__(
        self, provider_key: str, config_issue: str, original_error: Exception = None
    ):
        message = f"Configuration error for provider '{provider_key}': {config_issue}"
        super().__init__(message, provider_key, original_error)


class ProviderCredentialsError(ProviderError):
    """Raised when provider credentials are missing or invalid."""

    def __init__(self, provider_key: str, credential_issue: str):
        message = f"Credentials error for provider '{provider_key}': {credential_issue}"
        super().__init__(message, provider_key)


class ProviderStatusCheckError(ProviderError):
    """Raised when provider status check fails."""

    def __init__(
        self, provider_key: str, check_error: str, original_error: Exception = None
    ):
        message = f"Status check failed for provider '{provider_key}': {check_error}"
        super().__init__(message, provider_key, original_error)


class ProviderRegistrationError(ProviderError):
    """Raised when provider registration fails."""

    def __init__(self, provider_key: str, registration_issue: str):
        message = (
            f"Registration failed for provider '{provider_key}': {registration_issue}"
        )
        super().__init__(message, provider_key)
