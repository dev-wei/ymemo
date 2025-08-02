"""Language configuration and mapping for transcription providers."""

from typing import Dict, List, Optional

# Comprehensive language mappings for AWS Transcribe and Azure Speech Service
LANGUAGE_MAPPINGS = {
    # Major English variants
    "English (US)": {
        "aws": "en-US",
        "azure": "en-US",
        "display": "English (United States)",
        "default": True,  # Mark default language
    },
    "English (UK)": {
        "aws": "en-GB",
        "azure": "en-GB",
        "display": "English (United Kingdom)",
    },
    "English (Australia)": {
        "aws": "en-AU",
        "azure": "en-AU",
        "display": "English (Australia)",
    },
    "English (Canada)": {
        "aws": "en-CA",
        "azure": "en-CA",
        "display": "English (Canada)",
    },
    "English (India)": {"aws": "en-IN", "azure": "en-IN", "display": "English (India)"},
    # Spanish variants
    "Spanish (Spain)": {"aws": "es-ES", "azure": "es-ES", "display": "Spanish (Spain)"},
    "Spanish (US)": {
        "aws": "es-US",
        "azure": "es-MX",  # Azure uses Mexico for US Spanish
        "display": "Spanish (United States)",
    },
    # French variants
    "French (France)": {"aws": "fr-FR", "azure": "fr-FR", "display": "French (France)"},
    "French (Canada)": {"aws": "fr-CA", "azure": "fr-CA", "display": "French (Canada)"},
    # German variants
    "German (Germany)": {
        "aws": "de-DE",
        "azure": "de-DE",
        "display": "German (Germany)",
    },
    "German (Switzerland)": {
        "aws": "de-CH",
        "azure": "de-CH",
        "display": "German (Switzerland)",
    },
    # Chinese variants
    "Chinese (Simplified)": {
        "aws": "zh-CN",
        "azure": "zh-CN",
        "display": "Chinese (Simplified)",
    },
    "Chinese (Traditional)": {
        "aws": "zh-TW",
        "azure": "zh-TW",
        "display": "Chinese (Traditional)",
    },
    "Chinese (Cantonese)": {
        "aws": "zh-HK",
        "azure": "zh-HK",
        "display": "Chinese (Cantonese, Hong Kong)",
    },
    # Portuguese variants
    "Portuguese (Brazil)": {
        "aws": "pt-BR",
        "azure": "pt-BR",
        "display": "Portuguese (Brazil)",
    },
    "Portuguese (Portugal)": {
        "aws": "pt-PT",
        "azure": "pt-PT",
        "display": "Portuguese (Portugal)",
    },
    # Major Asian languages
    "Japanese": {"aws": "ja-JP", "azure": "ja-JP", "display": "Japanese"},
    "Korean": {"aws": "ko-KR", "azure": "ko-KR", "display": "Korean"},
    "Hindi": {"aws": "hi-IN", "azure": "hi-IN", "display": "Hindi (India)"},
    # Major European languages
    "Italian": {"aws": "it-IT", "azure": "it-IT", "display": "Italian"},
    "Russian": {"aws": "ru-RU", "azure": "ru-RU", "display": "Russian"},
    "Dutch": {"aws": "nl-NL", "azure": "nl-NL", "display": "Dutch (Netherlands)"},
    "Swedish": {"aws": "sv-SE", "azure": "sv-SE", "display": "Swedish"},
    "Norwegian": {
        "aws": "no-NO",
        "azure": "nb-NO",  # Azure uses nb-NO for Norwegian Bokmål
        "display": "Norwegian",
    },
    "Danish": {"aws": "da-DK", "azure": "da-DK", "display": "Danish"},
    "Finnish": {"aws": "fi-FI", "azure": "fi-FI", "display": "Finnish"},
    "Polish": {"aws": "pl-PL", "azure": "pl-PL", "display": "Polish"},
    # Arabic variants
    "Arabic (Saudi Arabia)": {
        "aws": "ar-SA",
        "azure": "ar-SA",
        "display": "Arabic (Saudi Arabia)",
    },
    "Arabic (UAE)": {
        "aws": "ar-AE",
        "azure": "ar-AE",
        "display": "Arabic (United Arab Emirates)",
    },
    # Other commonly used languages
    "Turkish": {"aws": "tr-TR", "azure": "tr-TR", "display": "Turkish"},
    "Thai": {"aws": "th-TH", "azure": "th-TH", "display": "Thai"},
    "Vietnamese": {"aws": "vi-VN", "azure": "vi-VN", "display": "Vietnamese"},
    "Indonesian": {"aws": "id-ID", "azure": "id-ID", "display": "Indonesian"},
    "Malay": {"aws": "ms-MY", "azure": "ms-MY", "display": "Malay (Malaysia)"},
}


def get_language_choices() -> List[str]:
    """Get list of language display names for dropdown."""
    return sorted(LANGUAGE_MAPPINGS.keys())


def get_default_language() -> str:
    """Get the default language key."""
    for key, value in LANGUAGE_MAPPINGS.items():
        if value.get("default", False):
            return key
    return "English (US)"  # Fallback


def get_language_code(language_key: str, provider: str) -> Optional[str]:
    """Get provider-specific language code for a language key.

    Args:
        language_key: Language key (e.g., "English (US)")
        provider: Provider name ("aws" or "azure")

    Returns:
        Provider-specific language code or None if not found
    """
    if language_key not in LANGUAGE_MAPPINGS:
        return None

    return LANGUAGE_MAPPINGS[language_key].get(provider)


def get_display_name(language_key: str) -> str:
    """Get display name for a language key.

    Args:
        language_key: Language key (e.g., "English (US)")

    Returns:
        Display name or the key itself if not found
    """
    if language_key not in LANGUAGE_MAPPINGS:
        return language_key

    return LANGUAGE_MAPPINGS[language_key].get("display", language_key)


def find_language_by_code(language_code: str, provider: str) -> Optional[str]:
    """Find language key by provider-specific code.

    Args:
        language_code: Provider-specific language code (e.g., "en-US")
        provider: Provider name ("aws" or "azure")

    Returns:
        Language key or None if not found
    """
    for key, value in LANGUAGE_MAPPINGS.items():
        if value.get(provider) == language_code:
            return key
    return None


def is_language_supported(language_key: str, provider: str) -> bool:
    """Check if a language is supported by a provider.

    Args:
        language_key: Language key (e.g., "English (US)")
        provider: Provider name ("aws" or "azure")

    Returns:
        True if supported, False otherwise
    """
    return (
        language_key in LANGUAGE_MAPPINGS
        and provider in LANGUAGE_MAPPINGS[language_key]
    )


def get_supported_languages(provider: str) -> List[str]:
    """Get list of language keys supported by a provider.

    Args:
        provider: Provider name ("aws" or "azure")

    Returns:
        List of supported language keys
    """
    supported = []
    for key, value in LANGUAGE_MAPPINGS.items():
        if provider in value:
            supported.append(key)
    return sorted(supported)
