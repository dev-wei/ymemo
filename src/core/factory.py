"""
Factory for creating audio processing providers.

This module provides a centralized factory for creating transcription and audio capture
providers. The factory pattern allows for easy swapping between different providers
(AWS Transcribe, Azure Speech, PyAudio, etc.) without changing client code.

Example Usage:
    # Create transcription providers
    aws_provider = AudioProcessorFactory.create_transcription_provider('aws',
                                                                      region='us-west-2')
    azure_provider = AudioProcessorFactory.create_transcription_provider('azure',
                                                                        speech_key='key',
                                                                        region='eastus')

    # Create audio capture providers
    mic_provider = AudioProcessorFactory.create_audio_capture_provider('pyaudio')
    file_provider = AudioProcessorFactory.create_audio_capture_provider('file',
                                                                       file_path='test.wav')

    # List available providers
    transcription_providers = AudioProcessorFactory.list_transcription_providers()
    capture_providers = AudioProcessorFactory.list_audio_capture_providers()
"""

import logging

from ..audio.providers.aws_transcribe import AWSTranscribeProvider
from ..audio.providers.azure_speech import AzureSpeechProvider
from ..audio.providers.file_audio_capture import FileAudioCaptureProvider
from ..audio.providers.pyaudio_capture import PyAudioCaptureProvider
from .interfaces import AudioCaptureProvider, TranscriptionProvider

logger = logging.getLogger(__name__)


class AudioProcessorFactory:
    """
    Factory for creating audio processing providers with easy swapping.

    This factory provides a centralized way to create transcription and audio capture
    providers. It supports:
    - Dynamic provider selection by name
    - Consistent error handling and logging
    - Runtime provider registration
    - Provider discovery and listing

    The factory ensures all providers implement the appropriate interface contracts
    and provides clear error messages for debugging.
    """

    # Registry of available transcription providers
    TRANSCRIPTION_PROVIDERS: dict[str, type[TranscriptionProvider]] = {
        "aws": AWSTranscribeProvider,  # Now handles both single and dual connections intelligently
        "azure": AzureSpeechProvider,
    }

    # Registry of available audio capture providers
    CAPTURE_PROVIDERS: dict[str, type[AudioCaptureProvider]] = {
        "pyaudio": PyAudioCaptureProvider,
        "file": FileAudioCaptureProvider,
    }

    @classmethod
    def create_transcription_provider(
        cls, provider_name: str, **config
    ) -> TranscriptionProvider:
        """
        Create a transcription provider instance.

        This method creates and configures a transcription provider based on the
        specified provider name and configuration parameters.

        Args:
            provider_name: Name of the provider. Currently supported:
                         - 'aws': AWS Transcribe service (intelligent single/dual connection switching)
                         - 'azure': Azure Speech service
            **config: Provider-specific configuration parameters:
                     For AWS: region, language_code, profile_name, connection_strategy, dual_fallback_enabled, channel_balance_threshold, dual_connection_test_mode
                     For Azure: speech_key, region, language_code, enable_speaker_diarization

        Returns:
            TranscriptionProvider: Fully configured provider instance

        Raises:
            ValueError: If provider_name is not supported or invalid
            TypeError: If required configuration parameters are missing
            RuntimeError: If provider initialization fails

        Example:
            # Create AWS provider using system config
            from config.audio_config import get_config
            config = get_config()
            aws_provider = AudioProcessorFactory.create_transcription_provider(
                'aws', **config.get_transcription_config()
            )

            # Create Azure provider
            azure_provider = AudioProcessorFactory.create_transcription_provider(
                'azure', speech_key='your-key', region='eastus'
            )
        """
        # Validate provider name
        if provider_name not in cls.TRANSCRIPTION_PROVIDERS:
            available = ", ".join(cls.TRANSCRIPTION_PROVIDERS.keys())
            raise ValueError(
                f"Unsupported transcription provider '{provider_name}'. "
                f"Available providers: {available}. "
                f"To add a new provider, use register_transcription_provider()."
            )

        provider_class = cls.TRANSCRIPTION_PROVIDERS[provider_name]

        try:
            logger.info(
                f"🏭 Factory: Creating transcription provider '{provider_name}' with config keys: {list(config.keys())}"
            )

            # Enhanced logging for AWS provider configuration
            if provider_name == "aws" and config:
                audio_saving_enabled = config.get(
                    "dual_save_split_audio"
                ) or config.get("dual_save_raw_audio")
                if audio_saving_enabled:
                    logger.info("🎵 Factory: AWS provider audio saving configuration:")
                    logger.info(
                        f"   🔀 Split audio: {config.get('dual_save_split_audio', False)}"
                    )
                    logger.info(
                        f"   🎵 Raw audio: {config.get('dual_save_raw_audio', False)}"
                    )
                    logger.info(
                        f"   📁 Save path: {config.get('dual_audio_save_path', 'N/A')}"
                    )
                    logger.info(
                        f"   ⏱️  Duration: {config.get('dual_audio_save_duration', 'N/A')}s"
                    )
                    logger.info(
                        f"   🧪 Test mode: {config.get('dual_connection_test_mode', 'N/A')}"
                    )
                else:
                    logger.info("🎵 Factory: AWS provider audio saving is DISABLED")

            instance = provider_class(**config)
            logger.info(
                f"✅ Factory: Successfully created {provider_name} transcription provider"
            )
            return instance
        except TypeError as e:
            logger.error(f"❌ Factory: Invalid configuration for {provider_name}: {e}")
            raise TypeError(
                f"Invalid configuration for transcription provider '{provider_name}': {e}"
            )
        except Exception as e:
            logger.error(
                f"❌ Factory: Failed to create transcription provider '{provider_name}': {e}"
            )
            raise RuntimeError(
                f"Failed to initialize transcription provider '{provider_name}': {e}"
            )

    @classmethod
    def create_audio_capture_provider(
        cls, provider_name: str, **config
    ) -> AudioCaptureProvider:
        """
        Create an audio capture provider instance.

        This method creates and configures an audio capture provider for recording
        audio from various sources (microphone, files, etc.).

        Args:
            provider_name: Name of the provider. Currently supported:
                         - 'pyaudio': PyAudio microphone capture
                         - 'file': File-based audio source for testing
            **config: Provider-specific configuration parameters:
                     For PyAudio: device_index (optional)
                     For File: file_path (required), loop (optional)

        Returns:
            AudioCaptureProvider: Fully configured provider instance

        Raises:
            ValueError: If provider_name is not supported or invalid
            TypeError: If required configuration parameters are missing
            RuntimeError: If provider initialization fails

        Example:
            # Create microphone capture provider
            mic_provider = AudioProcessorFactory.create_audio_capture_provider('pyaudio')

            # Create file-based provider for testing
            file_provider = AudioProcessorFactory.create_audio_capture_provider(
                'file', file_path='test_audio.wav', loop=True
            )
        """
        # Validate provider name
        if provider_name not in cls.CAPTURE_PROVIDERS:
            available = ", ".join(cls.CAPTURE_PROVIDERS.keys())
            raise ValueError(
                f"Unsupported audio capture provider '{provider_name}'. "
                f"Available providers: {available}. "
                f"To add a new provider, use register_audio_capture_provider()."
            )

        provider_class = cls.CAPTURE_PROVIDERS[provider_name]

        try:
            logger.info(
                f"🏭 Factory: Creating audio capture provider '{provider_name}' with config keys: {list(config.keys())}"
            )
            provider_instance = provider_class(**config)

            # Log instance details if available
            if hasattr(provider_instance, "_instance_id"):
                logger.info(
                    f"✅ Factory: Created {provider_name} provider instance {provider_instance._instance_id}"
                )
            else:
                logger.info(
                    f"✅ Factory: Successfully created {provider_name} audio capture provider"
                )

            return provider_instance
        except TypeError as e:
            logger.error(f"❌ Factory: Invalid configuration for {provider_name}: {e}")
            raise TypeError(
                f"Invalid configuration for audio capture provider '{provider_name}': {e}"
            )
        except Exception as e:
            logger.error(
                f"❌ Factory: Failed to create audio capture provider '{provider_name}': {e}"
            )
            raise RuntimeError(
                f"Failed to initialize audio capture provider '{provider_name}': {e}"
            )

    @classmethod
    def register_transcription_provider(
        cls, name: str, provider_class: type[TranscriptionProvider]
    ) -> None:
        """
        Register a new transcription provider for runtime use.

        This allows third-party or custom providers to be added to the factory
        without modifying the core code.

        Args:
            name: Unique name to register the provider under (e.g., 'whisper', 'google')
            provider_class: Provider class that implements TranscriptionProvider interface

        Raises:
            TypeError: If provider_class doesn't implement TranscriptionProvider interface

        Example:
            class CustomProvider(TranscriptionProvider):
                # Implementation here
                pass

            AudioProcessorFactory.register_transcription_provider('custom', CustomProvider)
        """
        # Validate that the provider implements the interface
        if not issubclass(provider_class, TranscriptionProvider):
            raise TypeError(
                f"Provider class {provider_class.__name__} must implement TranscriptionProvider interface"
            )

        cls.TRANSCRIPTION_PROVIDERS[name] = provider_class
        logger.info(
            f"✅ Factory: Registered transcription provider '{name}' -> {provider_class.__name__}"
        )

    @classmethod
    def register_audio_capture_provider(
        cls, name: str, provider_class: type[AudioCaptureProvider]
    ) -> None:
        """
        Register a new audio capture provider for runtime use.

        This allows third-party or custom providers to be added to the factory
        without modifying the core code.

        Args:
            name: Unique name to register the provider under (e.g., 'sounddevice', 'custom')
            provider_class: Provider class that implements AudioCaptureProvider interface

        Raises:
            TypeError: If provider_class doesn't implement AudioCaptureProvider interface

        Example:
            class CustomCaptureProvider(AudioCaptureProvider):
                # Implementation here
                pass

            AudioProcessorFactory.register_audio_capture_provider('custom', CustomCaptureProvider)
        """
        # Validate that the provider implements the interface
        if not issubclass(provider_class, AudioCaptureProvider):
            raise TypeError(
                f"Provider class {provider_class.__name__} must implement AudioCaptureProvider interface"
            )

        cls.CAPTURE_PROVIDERS[name] = provider_class
        logger.info(
            f"✅ Factory: Registered audio capture provider '{name}' -> {provider_class.__name__}"
        )

    @classmethod
    def list_transcription_providers(cls) -> dict[str, str]:
        """
        List all available transcription providers.

        Returns:
            Dictionary mapping provider names to their class names.

        Example:
            providers = AudioProcessorFactory.list_transcription_providers()
            print(providers)  # {'aws': 'AWSTranscribeProvider', 'azure': 'AzureSpeechProvider'}
        """
        return {
            name: provider_class.__name__
            for name, provider_class in cls.TRANSCRIPTION_PROVIDERS.items()
        }

    @classmethod
    def list_audio_capture_providers(cls) -> dict[str, str]:
        """
        List all available audio capture providers.

        Returns:
            Dictionary mapping provider names to their class names.

        Example:
            providers = AudioProcessorFactory.list_audio_capture_providers()
            print(providers)  # {'pyaudio': 'PyAudioCaptureProvider', 'file': 'FileAudioCaptureProvider'}
        """
        return {
            name: provider_class.__name__
            for name, provider_class in cls.CAPTURE_PROVIDERS.items()
        }


# =============================================================================
# Convenience Functions for Easy Provider Creation
# =============================================================================
# These functions provide simplified interfaces for creating common providers
# with sensible defaults, reducing boilerplate code for typical use cases.


def create_aws_transcribe_provider(
    region: str | None = None,
    language_code: str | None = None,
    profile_name: str | None = None,
) -> TranscriptionProvider:
    """
    Create AWS Transcribe provider using system configuration defaults.

    This is a convenience function that uses centralized configuration
    with optional parameter overrides.

    Args:
        region: AWS region for Transcribe service (default: 'us-east-1')
        language_code: Language code for transcription (default: 'en-US')
        profile_name: AWS profile name for authentication (default: None, uses default profile)

    Returns:
        TranscriptionProvider: Configured AWS Transcribe provider

    Example:
        # Use defaults
        provider = create_aws_transcribe_provider()

        # Customize region and language
        provider = create_aws_transcribe_provider(
            region='us-west-2',
            language_code='es-US'
        )
    """
    # Use system configuration as defaults if not provided
    from config.audio_config import get_config

    system_config = get_config()

    # Use provided values or fall back to system config
    final_region = region or system_config.aws_region
    final_language = language_code or system_config.aws_language_code
    final_profile = profile_name  # This can stay None if not provided

    return AudioProcessorFactory.create_transcription_provider(
        "aws",
        region=final_region,
        language_code=final_language,
        profile_name=final_profile,
    )


def create_azure_speech_provider(
    speech_key: str,
    region: str = "eastus",
    language_code: str = "en-US",
    endpoint: str | None = None,
    enable_speaker_diarization: bool = False,
    max_speakers: int = 4,
    timeout: int = 30,
) -> TranscriptionProvider:
    """
    Create Azure Speech Service provider with common defaults.

    This is a convenience function that simplifies creating Azure Speech Service providers
    with commonly used configuration values.

    Args:
        speech_key: Azure Speech Service subscription key (required)
        region: Azure region for Speech service (default: 'eastus')
        language_code: Language code for transcription (default: 'en-US')
        endpoint: Custom endpoint URL (default: None, uses standard endpoint)
        enable_speaker_diarization: Enable speaker identification (default: False)
        max_speakers: Maximum number of speakers to identify (default: 4)
        timeout: Connection timeout in seconds (default: 30)

    Returns:
        TranscriptionProvider: Configured Azure Speech Service provider

    Example:
        # Basic setup
        provider = create_azure_speech_provider(speech_key='your-key')

        # With speaker diarization
        provider = create_azure_speech_provider(
            speech_key='your-key',
            enable_speaker_diarization=True,
            max_speakers=6
        )
    """
    return AudioProcessorFactory.create_transcription_provider(
        "azure",
        speech_key=speech_key,
        region=region,
        language_code=language_code,
        endpoint=endpoint,
        enable_speaker_diarization=enable_speaker_diarization,
        max_speakers=max_speakers,
        timeout=timeout,
    )


def create_pyaudio_capture_provider(
    device_index: int | None = None,
) -> AudioCaptureProvider:
    """
    Create PyAudio microphone capture provider.

    This is a convenience function for creating PyAudio providers to capture
    audio from system microphones.

    Args:
        device_index: Specific audio device index to use (default: None, uses system default)

    Returns:
        AudioCaptureProvider: Configured PyAudio capture provider

    Example:
        # Use default microphone
        provider = create_pyaudio_capture_provider()

        # Use specific device
        provider = create_pyaudio_capture_provider(device_index=2)
    """
    config = {}
    if device_index is not None:
        config["device_index"] = device_index

    return AudioProcessorFactory.create_audio_capture_provider("pyaudio", **config)
