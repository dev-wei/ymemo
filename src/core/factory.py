"""Factory for creating audio processing providers."""

import logging
from typing import Dict, Type, Any, Optional

from .interfaces import TranscriptionProvider, AudioCaptureProvider
from ..audio.providers.aws_transcribe import AWSTranscribeProvider
from ..audio.providers.pyaudio_capture import PyAudioCaptureProvider
from ..audio.providers.file_audio_capture import FileAudioCaptureProvider


logger = logging.getLogger(__name__)


class AudioProcessorFactory:
    """Factory for creating audio processing providers with easy swapping."""
    
    # Registry of available transcription providers
    TRANSCRIPTION_PROVIDERS: Dict[str, Type[TranscriptionProvider]] = {
        'aws': AWSTranscribeProvider,
    }
    
    # Registry of available audio capture providers
    CAPTURE_PROVIDERS: Dict[str, Type[AudioCaptureProvider]] = {
        'pyaudio': PyAudioCaptureProvider,
        'file': FileAudioCaptureProvider,
    }
    
    @classmethod
    def create_transcription_provider(
        cls, 
        provider_name: str, 
        **config
    ) -> TranscriptionProvider:
        """Create a transcription provider instance.
        
        Args:
            provider_name: Name of the provider ('aws', 'whisper', 'google', etc.)
            **config: Configuration parameters for the provider
            
        Returns:
            TranscriptionProvider instance
            
        Raises:
            ValueError: If provider_name is not supported
        """
        if provider_name not in cls.TRANSCRIPTION_PROVIDERS:
            available = ', '.join(cls.TRANSCRIPTION_PROVIDERS.keys())
            raise ValueError(f"Unknown transcription provider: {provider_name}. "
                           f"Available providers: {available}")
        
        provider_class = cls.TRANSCRIPTION_PROVIDERS[provider_name]
        
        try:
            logger.info(f"Creating transcription provider: {provider_name}")
            return provider_class(**config)
        except Exception as e:
            logger.error(f"Failed to create transcription provider {provider_name}: {e}")
            raise
    
    @classmethod
    def create_audio_capture_provider(
        cls, 
        provider_name: str, 
        **config
    ) -> AudioCaptureProvider:
        """Create an audio capture provider instance.
        
        Args:
            provider_name: Name of the provider ('pyaudio', 'sounddevice', etc.)
            **config: Configuration parameters for the provider
            
        Returns:
            AudioCaptureProvider instance
            
        Raises:
            ValueError: If provider_name is not supported
        """
        if provider_name not in cls.CAPTURE_PROVIDERS:
            available = ', '.join(cls.CAPTURE_PROVIDERS.keys())
            raise ValueError(f"Unknown audio capture provider: {provider_name}. "
                           f"Available providers: {available}")
        
        provider_class = cls.CAPTURE_PROVIDERS[provider_name]
        
        try:
            logger.info(f"Creating audio capture provider: {provider_name}")
            provider_instance = provider_class(**config)
            if hasattr(provider_instance, '_instance_id'):
                logger.info(f"🏭 Factory: Created {provider_name} provider instance {provider_instance._instance_id}")
            return provider_instance
        except Exception as e:
            logger.error(f"Failed to create audio capture provider {provider_name}: {e}")
            raise
    
    @classmethod
    def register_transcription_provider(
        cls, 
        name: str, 
        provider_class: Type[TranscriptionProvider]
    ) -> None:
        """Register a new transcription provider.
        
        Args:
            name: Name to register the provider under
            provider_class: Provider class that implements TranscriptionProvider
        """
        cls.TRANSCRIPTION_PROVIDERS[name] = provider_class
        logger.info(f"Registered transcription provider: {name}")
    
    @classmethod
    def register_audio_capture_provider(
        cls, 
        name: str, 
        provider_class: Type[AudioCaptureProvider]
    ) -> None:
        """Register a new audio capture provider.
        
        Args:
            name: Name to register the provider under
            provider_class: Provider class that implements AudioCaptureProvider
        """
        cls.CAPTURE_PROVIDERS[name] = provider_class
        logger.info(f"Registered audio capture provider: {name}")
    
    @classmethod
    def list_transcription_providers(cls) -> Dict[str, str]:
        """List available transcription providers.
        
        Returns:
            Dictionary mapping provider names to class names
        """
        return {
            name: provider_class.__name__ 
            for name, provider_class in cls.TRANSCRIPTION_PROVIDERS.items()
        }
    
    @classmethod
    def list_audio_capture_providers(cls) -> Dict[str, str]:
        """List available audio capture providers.
        
        Returns:
            Dictionary mapping provider names to class names
        """
        return {
            name: provider_class.__name__ 
            for name, provider_class in cls.CAPTURE_PROVIDERS.items()
        }


# Convenience functions for easy provider creation
def create_aws_transcribe_provider(
    region: str = 'us-east-1', 
    language_code: str = 'en-US',
    profile_name: Optional[str] = None
) -> TranscriptionProvider:
    """Create AWS Transcribe provider with common defaults."""
    return AudioProcessorFactory.create_transcription_provider(
        'aws', 
        region=region, 
        language_code=language_code,
        profile_name=profile_name
    )


def create_pyaudio_capture_provider() -> AudioCaptureProvider:
    """Create PyAudio capture provider."""
    return AudioProcessorFactory.create_audio_capture_provider('pyaudio')