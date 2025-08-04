"""Transcription provider configuration and management.

This module provides backward compatibility with the old dictionary-based system
while using the new registry-based architecture under the hood.
"""

import logging
import os
from typing import Dict, List, Optional

from ..exceptions.provider_exceptions import ProviderStatusCheckError
from ..services.provider_service import get_provider_service

# Import new system components
from .provider_registry import (
    ProviderConfig,
    ProviderFeature,
    ProviderLatency,
)
from .provider_registry import ProviderStatus as NewProviderStatus
from .provider_registry import (
    get_registry,
)

logger = logging.getLogger(__name__)


def _check_aws_status(provider_key: str) -> 'ProviderStatusInfo':
    """Check AWS provider status."""
    from .provider_registry import ProviderStatusInfo

    try:
        # Check if AWS credentials are available
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError

        try:
            # Try to create a session to validate credentials
            session = boto3.Session()
            credentials = session.get_credentials()

            if credentials is None:
                return ProviderStatusInfo(
                    status=NewProviderStatus.ERROR,
                    message="AWS credentials not found",
                    icon="❌",
                )

            # Check if transcribe service is accessible
            region = os.environ.get('AWS_REGION', 'us-east-1')
            transcribe_client = session.client('transcribe', region_name=region)

            # Try to list transcription jobs (this validates permissions)
            transcribe_client.list_transcription_jobs(MaxResults=1)

            return ProviderStatusInfo(
                status=NewProviderStatus.READY,
                message=f"Ready (region: {region})",
                icon="✅",
            )

        except NoCredentialsError:
            return ProviderStatusInfo(
                status=NewProviderStatus.ERROR,
                message="AWS credentials not configured",
                icon="❌",
            )
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'UnauthorizedOperation':
                return ProviderStatusInfo(
                    status=NewProviderStatus.WARNING,
                    message="AWS credentials found but may lack Transcribe permissions",
                    icon="⚠️",
                )
            return ProviderStatusInfo(
                status=NewProviderStatus.WARNING,
                message=f"AWS available but service issue: {error_code}",
                icon="⚠️",
            )

    except ImportError:
        return ProviderStatusInfo(
            status=NewProviderStatus.ERROR,
            message="boto3 library not installed",
            icon="❌",
        )
    except Exception as e:
        logger.warning(f"AWS status check failed: {e}")
        return ProviderStatusInfo(
            status=NewProviderStatus.WARNING,
            message=f"Status check failed: {str(e)[:50]}...",
            icon="⚠️",
        )


def _check_azure_status(provider_key: str) -> 'ProviderStatusInfo':
    """Check Azure provider status."""
    from .provider_registry import ProviderStatusInfo

    try:
        speech_key = os.environ.get('AZURE_SPEECH_KEY')
        speech_region = os.environ.get('AZURE_SPEECH_REGION', 'eastus')

        if not speech_key:
            return ProviderStatusInfo(
                status=NewProviderStatus.ERROR,
                message="Azure Speech API key required (AZURE_SPEECH_KEY)",
                icon="❌",
            )

        # Basic validation - key should be 32 characters
        if len(speech_key) != 32:
            return ProviderStatusInfo(
                status=NewProviderStatus.WARNING,
                message="Azure API key format may be invalid",
                icon="⚠️",
            )

        return ProviderStatusInfo(
            status=NewProviderStatus.READY,
            message=f"Ready (region: {speech_region})",
            icon="✅",
        )

    except Exception as e:
        logger.warning(f"Azure status check failed: {e}")
        return ProviderStatusInfo(
            status=NewProviderStatus.WARNING,
            message=f"Status check failed: {str(e)[:50]}...",
            icon="⚠️",
        )


# Initialize the registry with default providers on first import
def _initialize_default_providers():
    """Initialize the registry with default provider configurations."""
    registry = get_registry()

    # Check if already initialized
    if registry.list_providers():
        return

    logger.info("🏭 Initializing default provider configurations...")

    # AWS Transcribe Provider
    aws_config = ProviderConfig(
        key="aws",
        display_name="AWS Transcribe",
        description="Amazon Web Services real-time transcription with speaker diarization",
        implemented=True,
        features={
            ProviderFeature.STREAMING,
            ProviderFeature.BATCH,
            ProviderFeature.SPEAKER_DIARIZATION,
            ProviderFeature.REAL_TIME,
            ProviderFeature.DUAL_CHANNEL,
        },
        supported_regions=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-2"],
        max_audio_length="4 hours",
        latency=ProviderLatency.LOW,
        requirements=["AWS credentials"],
        status_check_function="check_aws_credentials",
        vendor="Amazon Web Services",
    )

    # Azure Speech Service Provider
    azure_config = ProviderConfig(
        key="azure",
        display_name="Azure Speech Service",
        description="Microsoft Azure cognitive speech-to-text with speaker identification",
        implemented=True,
        features={
            ProviderFeature.STREAMING,
            ProviderFeature.SPEAKER_DIARIZATION,
            ProviderFeature.REAL_TIME,
            ProviderFeature.CUSTOM_MODELS,
        },
        supported_regions=["eastus", "westus2", "westeurope", "eastasia"],
        max_audio_length="10 minutes per request",
        latency=ProviderLatency.LOW,
        requirements=["Azure Speech API key", "Azure region"],
        status_check_function="check_azure_credentials",
        vendor="Microsoft Azure",
    )

    # OpenAI Whisper Provider (Future)
    whisper_config = ProviderConfig(
        key="whisper",
        display_name="OpenAI Whisper",
        description="OpenAI's robust offline speech recognition model",
        implemented=False,
        features={
            ProviderFeature.BATCH,
            ProviderFeature.MULTILINGUAL,
            ProviderFeature.OFFLINE,
            ProviderFeature.HIGH_ACCURACY,
        },
        supported_regions=["local"],
        max_audio_length="unlimited",
        latency=ProviderLatency.HIGH,
        requirements=["Local model files", "Python whisper package"],
        status_check_function="check_whisper_availability",
        vendor="OpenAI",
    )

    # Google Cloud Speech Provider (Future)
    google_config = ProviderConfig(
        key="google",
        display_name="Google Cloud Speech",
        description="Google Cloud Platform speech-to-text API",
        implemented=False,
        features={
            ProviderFeature.STREAMING,
            ProviderFeature.BATCH,
            ProviderFeature.SPEAKER_DIARIZATION,
            ProviderFeature.AUTO_PUNCTUATION,
        },
        supported_regions=["global", "us-central1", "europe-west1"],
        max_audio_length="unlimited",
        latency=ProviderLatency.LOW,
        requirements=["Google Cloud credentials", "Speech API enabled"],
        status_check_function="check_google_credentials",
        vendor="Google Cloud Platform",
    )

    # Register all providers with status checkers
    registry.register_provider(aws_config, _check_aws_status)
    registry.register_provider(azure_config, _check_azure_status)
    registry.register_provider(whisper_config)  # No status checker yet
    registry.register_provider(google_config)  # No status checker yet

    logger.info("✅ Default provider configurations initialized")


# Initialize on module import
_initialize_default_providers()

# Legacy compatibility - maintain the old dictionary structure for existing code
AVAILABLE_PROVIDERS = {
    "aws": {
        "display_name": "AWS Transcribe",
        "description": "Amazon Web Services real-time transcription with speaker diarization",
        "requires": ["AWS credentials"],
        "features": [
            "streaming",
            "batch",
            "speaker_diarization",
            "real_time",
            "dual_channel",
        ],
        "implemented": True,
        "status_check": "check_aws_credentials",
        "regions": ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-2"],
        "language_support": "extensive",
        "max_audio_length": "4 hours",
        "latency": "low",
    },
    "azure": {
        "display_name": "Azure Speech Service",
        "description": "Microsoft Azure cognitive speech-to-text with speaker identification",
        "requires": ["Azure Speech API key", "Azure region"],
        "features": ["streaming", "speaker_diarization", "real_time", "custom_models"],
        "implemented": True,
        "status_check": "check_azure_credentials",
        "regions": ["eastus", "westus2", "westeurope", "eastasia"],
        "language_support": "extensive",
        "max_audio_length": "10 minutes per request",
        "latency": "low",
    },
    "whisper": {
        "display_name": "OpenAI Whisper",
        "description": "OpenAI's robust offline speech recognition model",
        "requires": ["Local model files", "Python whisper package"],
        "features": ["batch", "multilingual", "offline", "high_accuracy"],
        "implemented": False,
        "status_check": "check_whisper_availability",
        "regions": ["local"],
        "language_support": "99 languages",
        "max_audio_length": "unlimited",
        "latency": "high",
    },
    "google": {
        "display_name": "Google Cloud Speech",
        "description": "Google Cloud Platform speech-to-text API",
        "requires": ["Google Cloud credentials", "Speech API enabled"],
        "features": ["streaming", "batch", "speaker_diarization", "auto_punctuation"],
        "implemented": False,
        "status_check": "check_google_credentials",
        "regions": ["global", "us-central1", "europe-west1"],
        "language_support": "125+ languages",
        "max_audio_length": "unlimited",
        "latency": "low",
    },
}


# Provider status types - maintain backward compatibility
class ProviderStatus:
    READY = "ready"
    WARNING = "warning"
    ERROR = "error"
    NOT_IMPLEMENTED = "not_implemented"


def get_available_providers(include_unimplemented: bool = False) -> List[str]:
    """Get list of available provider keys.

    Args:
        include_unimplemented: Whether to include providers not yet implemented

    Returns:
        List of provider keys
    """
    service = get_provider_service()
    registry = get_registry()
    return registry.list_provider_keys(include_unimplemented)


def get_provider_choices() -> List[str]:
    """Get list of provider display names for dropdown."""
    service = get_provider_service()
    return service.get_provider_choices()


def get_provider_key_from_display_name(display_name: str) -> Optional[str]:
    """Get provider key from display name.

    Args:
        display_name: Display name (e.g., "AWS Transcribe")

    Returns:
        Provider key (e.g., "aws") or None if not found
    """
    service = get_provider_service()
    return service.get_provider_key_from_display_name(display_name)


def get_display_name_from_key(provider_key: str) -> str:
    """Get display name from provider key.

    Args:
        provider_key: Provider key (e.g., "aws")

    Returns:
        Display name (e.g., "AWS Transcribe")
    """
    service = get_provider_service()
    return service.get_display_name_from_key(provider_key)


def get_default_provider() -> str:
    """Get the default provider key."""
    return "aws"  # Default to AWS


def get_current_provider_from_env() -> str:
    """Get current provider from environment variable."""
    service = get_provider_service()
    return service.get_current_provider_from_env()


def validate_provider_availability(provider_key: str) -> bool:
    """Check if provider is available and implemented.

    Args:
        provider_key: Provider key to validate

    Returns:
        True if provider is available and implemented
    """
    try:
        registry = get_registry()
        provider_config = registry.get_provider(provider_key)
        return provider_config.implemented
    except:
        return False


def check_provider_status(provider_key: str) -> Dict[str, any]:
    """Check the status of a specific provider.

    Args:
        provider_key: Provider key to check

    Returns:
        Dict with status information
    """
    service = get_provider_service()
    status_info = service._get_cached_status(provider_key)

    # Convert to legacy format
    return {
        "status": status_info.status.value,
        "message": status_info.message,
        "icon": status_info.icon,
    }


def get_provider_info_html(provider_key: str) -> str:
    """Get HTML formatted provider information.

    Args:
        provider_key: Provider key to get info for

    Returns:
        HTML string with provider information
    """
    service = get_provider_service()
    return service.get_provider_info_html(provider_key)


def is_provider_language_compatible(provider_key: str, language_key: str) -> bool:
    """Check if provider supports the selected language.

    Args:
        provider_key: Provider to check
        language_key: Language key (e.g., "Chinese (Simplified)")

    Returns:
        True if compatible (for now, assume all implemented providers support all languages)
    """
    service = get_provider_service()
    return service.check_provider_language_compatibility(provider_key, language_key)


def get_provider_requirements(provider_key: str) -> List[str]:
    """Get list of requirements for a provider.

    Args:
        provider_key: Provider key

    Returns:
        List of requirements
    """
    service = get_provider_service()
    return service.get_provider_requirements(provider_key)
