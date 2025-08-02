"""Transcription provider configuration and management."""

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Comprehensive provider definitions
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
        "language_support": "extensive",  # 100+ languages
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
        "language_support": "extensive",  # 100+ languages
        "max_audio_length": "10 minutes per request",
        "latency": "low",
    },
    "whisper": {
        "display_name": "OpenAI Whisper",
        "description": "OpenAI's robust offline speech recognition model",
        "requires": ["Local model files", "Python whisper package"],
        "features": ["batch", "multilingual", "offline", "high_accuracy"],
        "implemented": False,  # Future implementation
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
        "implemented": False,  # Future implementation
        "status_check": "check_google_credentials",
        "regions": ["global", "us-central1", "europe-west1"],
        "language_support": "125+ languages",
        "max_audio_length": "unlimited",
        "latency": "low",
    },
}


# Provider status types
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
    if include_unimplemented:
        return list(AVAILABLE_PROVIDERS.keys())

    return [
        key
        for key, config in AVAILABLE_PROVIDERS.items()
        if config.get("implemented", False)
    ]


def get_provider_choices() -> List[str]:
    """Get list of provider display names for dropdown."""
    implemented_providers = get_available_providers()
    return [AVAILABLE_PROVIDERS[key]["display_name"] for key in implemented_providers]


def get_provider_key_from_display_name(display_name: str) -> Optional[str]:
    """Get provider key from display name.

    Args:
        display_name: Display name (e.g., "AWS Transcribe")

    Returns:
        Provider key (e.g., "aws") or None if not found
    """
    for key, config in AVAILABLE_PROVIDERS.items():
        if config["display_name"] == display_name:
            return key
    return None


def get_display_name_from_key(provider_key: str) -> str:
    """Get display name from provider key.

    Args:
        provider_key: Provider key (e.g., "aws")

    Returns:
        Display name (e.g., "AWS Transcribe")
    """
    if provider_key in AVAILABLE_PROVIDERS:
        return AVAILABLE_PROVIDERS[provider_key]["display_name"]
    return provider_key.upper()


def get_default_provider() -> str:
    """Get the default provider key."""
    return "aws"  # Default to AWS


def get_current_provider_from_env() -> str:
    """Get current provider from environment variable."""
    return os.environ.get('TRANSCRIPTION_PROVIDER', get_default_provider())


def validate_provider_availability(provider_key: str) -> bool:
    """Check if provider is available and implemented.

    Args:
        provider_key: Provider key to validate

    Returns:
        True if provider is available and implemented
    """
    if provider_key not in AVAILABLE_PROVIDERS:
        return False

    return AVAILABLE_PROVIDERS[provider_key].get("implemented", False)


def check_provider_status(provider_key: str) -> Dict[str, any]:
    """Check the status of a specific provider.

    Args:
        provider_key: Provider key to check

    Returns:
        Dict with status information
    """
    if provider_key not in AVAILABLE_PROVIDERS:
        return {
            "status": ProviderStatus.ERROR,
            "message": f"Unknown provider: {provider_key}",
            "icon": "❌",
        }

    provider_config = AVAILABLE_PROVIDERS[provider_key]

    # Check if implemented
    if not provider_config.get("implemented", False):
        return {
            "status": ProviderStatus.NOT_IMPLEMENTED,
            "message": "Coming soon - not yet implemented",
            "icon": "🚧",
        }

    # Check provider-specific status
    status_check_func = provider_config.get("status_check")
    if status_check_func == "check_aws_credentials":
        return _check_aws_status()
    elif status_check_func == "check_azure_credentials":
        return _check_azure_status()

    # Default to ready if no specific check
    return {
        "status": ProviderStatus.READY,
        "message": "Provider available",
        "icon": "✅",
    }


def _check_aws_status() -> Dict[str, any]:
    """Check AWS provider status."""
    try:
        # Check if AWS credentials are available
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError

        try:
            # Try to create a session to validate credentials
            session = boto3.Session()
            credentials = session.get_credentials()

            if credentials is None:
                return {
                    "status": ProviderStatus.ERROR,
                    "message": "AWS credentials not found",
                    "icon": "❌",
                }

            # Check if transcribe service is accessible
            region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
            transcribe_client = session.client('transcribe', region_name=region)

            # Try to list transcription jobs (this validates permissions)
            transcribe_client.list_transcription_jobs(MaxResults=1)

            return {
                "status": ProviderStatus.READY,
                "message": f"Ready (region: {region})",
                "icon": "✅",
            }

        except NoCredentialsError:
            return {
                "status": ProviderStatus.ERROR,
                "message": "AWS credentials not configured",
                "icon": "❌",
            }
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'UnauthorizedOperation':
                return {
                    "status": ProviderStatus.WARNING,
                    "message": "AWS credentials found but may lack Transcribe permissions",
                    "icon": "⚠️",
                }
            return {
                "status": ProviderStatus.WARNING,
                "message": f"AWS available but service issue: {error_code}",
                "icon": "⚠️",
            }

    except ImportError:
        return {
            "status": ProviderStatus.ERROR,
            "message": "boto3 library not installed",
            "icon": "❌",
        }
    except Exception as e:
        logger.warning(f"AWS status check failed: {e}")
        return {
            "status": ProviderStatus.WARNING,
            "message": f"Status check failed: {str(e)[:50]}...",
            "icon": "⚠️",
        }


def _check_azure_status() -> Dict[str, any]:
    """Check Azure provider status."""
    try:
        speech_key = os.environ.get('AZURE_SPEECH_KEY')
        speech_region = os.environ.get('AZURE_SPEECH_REGION', 'eastus')

        if not speech_key:
            return {
                "status": ProviderStatus.ERROR,
                "message": "Azure Speech API key required (AZURE_SPEECH_KEY)",
                "icon": "❌",
            }

        # Basic validation - key should be 32 characters
        if len(speech_key) != 32:
            return {
                "status": ProviderStatus.WARNING,
                "message": "Azure API key format may be invalid",
                "icon": "⚠️",
            }

        return {
            "status": ProviderStatus.READY,
            "message": f"Ready (region: {speech_region})",
            "icon": "✅",
        }

    except Exception as e:
        logger.warning(f"Azure status check failed: {e}")
        return {
            "status": ProviderStatus.WARNING,
            "message": f"Status check failed: {str(e)[:50]}...",
            "icon": "⚠️",
        }


def get_provider_info_html(provider_key: str) -> str:
    """Get HTML formatted provider information.

    Args:
        provider_key: Provider key to get info for

    Returns:
        HTML string with provider information
    """
    if provider_key not in AVAILABLE_PROVIDERS:
        return "<span style='color: red;'>Unknown provider</span>"

    provider_config = AVAILABLE_PROVIDERS[provider_key]
    status_info = check_provider_status(provider_key)

    display_name = provider_config["display_name"]
    description = provider_config["description"]
    features = ", ".join(provider_config["features"][:3])  # Show first 3 features

    status_color = {
        ProviderStatus.READY: "green",
        ProviderStatus.WARNING: "orange",
        ProviderStatus.ERROR: "red",
        ProviderStatus.NOT_IMPLEMENTED: "gray",
    }.get(status_info["status"], "gray")

    return f"""
    <div style='font-size: 0.9em; color: #444;'>
        <strong>{display_name}</strong><br/>
        <span style='color: #666;'>{description}</span><br/>
        <span style='color: {status_color}; font-size: 0.8em;'>
            {status_info["icon"]} {status_info["message"]}
        </span><br/>
        <span style='color: #888; font-size: 0.8em;'>
            Features: {features}
        </span>
    </div>
    """


def is_provider_language_compatible(provider_key: str, language_key: str) -> bool:
    """Check if provider supports the selected language.

    Args:
        provider_key: Provider to check
        language_key: Language key (e.g., "Chinese (Simplified)")

    Returns:
        True if compatible (for now, assume all implemented providers support all languages)
    """
    # For now, assume all implemented providers support the languages we have configured
    # In the future, this could check against provider-specific language matrices
    return validate_provider_availability(provider_key)


def get_provider_requirements(provider_key: str) -> List[str]:
    """Get list of requirements for a provider.

    Args:
        provider_key: Provider key

    Returns:
        List of requirements
    """
    if provider_key not in AVAILABLE_PROVIDERS:
        return ["Unknown provider"]

    return AVAILABLE_PROVIDERS[provider_key].get("requires", [])
