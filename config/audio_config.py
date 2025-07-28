"""Configuration system for audio processing providers."""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class AudioSystemConfig:
    """Comprehensive configuration for the entire audio processing system."""
    
    # Provider selection
    transcription_provider: str = 'aws'
    capture_provider: str = 'pyaudio'
    
    # Audio settings
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    audio_format: str = 'int16'
    
    # AWS Transcribe settings
    aws_region: str = 'us-east-1'
    aws_language_code: str = 'en-US'
    aws_max_speakers: int = 10
    
    # Performance settings
    max_latency_ms: int = 300
    enable_partial_results: bool = True
    
    # Partial result handling
    partial_result_handling: str = 'replace'  # 'replace', 'append', 'final_only'
    partial_result_timeout: float = 2.0  # Seconds before treating partial as final
    confidence_threshold: float = 0.0  # Minimum confidence to show result
    
    # Fallback providers
    fallback_providers: list = None
    
    def __post_init__(self):
        if self.fallback_providers is None:
            self.fallback_providers = ['whisper', 'google']
    
    @classmethod
    def from_env(cls) -> 'AudioSystemConfig':
        """Create configuration from environment variables."""
        return cls(
            transcription_provider=os.getenv('TRANSCRIPTION_PROVIDER', 'aws'),
            capture_provider=os.getenv('CAPTURE_PROVIDER', 'pyaudio'),
            sample_rate=int(os.getenv('AUDIO_SAMPLE_RATE', '16000')),
            channels=int(os.getenv('AUDIO_CHANNELS', '1')),
            chunk_size=int(os.getenv('AUDIO_CHUNK_SIZE', '1024')),
            audio_format=os.getenv('AUDIO_FORMAT', 'int16'),
            aws_region=os.getenv('AWS_REGION', 'us-east-1'),
            aws_language_code=os.getenv('AWS_LANGUAGE_CODE', 'en-US'),
            aws_max_speakers=int(os.getenv('AWS_MAX_SPEAKERS', '10')),
            max_latency_ms=int(os.getenv('MAX_LATENCY_MS', '300')),
            enable_partial_results=os.getenv('ENABLE_PARTIAL_RESULTS', 'true').lower() == 'true',
            partial_result_handling=os.getenv('PARTIAL_RESULT_HANDLING', 'replace'),
            partial_result_timeout=float(os.getenv('PARTIAL_RESULT_TIMEOUT', '2.0')),
            confidence_threshold=float(os.getenv('CONFIDENCE_THRESHOLD', '0.0'))
        )
    
    def get_transcription_config(self) -> Dict[str, Any]:
        """Get configuration for transcription provider."""
        if self.transcription_provider == 'aws':
            return {
                'region': self.aws_region,
                'language_code': self.aws_language_code
            }
        elif self.transcription_provider == 'whisper':
            return {
                'model_size': os.getenv('WHISPER_MODEL_SIZE', 'base'),
                'device': os.getenv('WHISPER_DEVICE', 'auto')
            }
        elif self.transcription_provider == 'google':
            return {
                'language_code': self.aws_language_code,
                'credentials_path': os.getenv('GOOGLE_CREDENTIALS_PATH')
            }
        else:
            return {}
    
    def get_capture_config(self) -> Dict[str, Any]:
        """Get configuration for audio capture provider."""
        return {
            # PyAudio or other capture providers may need specific config
        }
    
    def get_audio_config(self) -> 'AudioConfig':
        """Get audio configuration as AudioConfig object."""
        from src.core.interfaces import AudioConfig
        return AudioConfig(
            sample_rate=self.sample_rate,
            channels=self.channels,
            chunk_size=self.chunk_size,
            format=self.audio_format
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)


# Default configuration instance
default_config = AudioSystemConfig()


def get_config() -> AudioSystemConfig:
    """Get audio system configuration.
    
    Priority order:
    1. Environment variables
    2. Default configuration
    """
    return AudioSystemConfig.from_env()


# Provider-specific configuration templates
PROVIDER_CONFIGS = {
    'aws_transcribe_streaming': {
        'transcription_provider': 'aws',
        'aws_region': 'us-east-1',
        'aws_language_code': 'en-US',
        'max_latency_ms': 200,
        'enable_partial_results': True
    },
    
    'whisper_local': {
        'transcription_provider': 'whisper',
        'max_latency_ms': 500,
        'enable_partial_results': False
    },
    
    'google_streaming': {
        'transcription_provider': 'google',
        'max_latency_ms': 300,
        'enable_partial_results': True
    },
    
    'high_performance': {
        'transcription_provider': 'aws',
        'sample_rate': 16000,
        'chunk_size': 512,  # Smaller chunks for lower latency
        'max_latency_ms': 100,
        'enable_partial_results': True
    },
    
    'cost_optimized': {
        'transcription_provider': 'whisper',
        'sample_rate': 16000,
        'chunk_size': 2048,  # Larger chunks for efficiency
        'max_latency_ms': 1000,
        'enable_partial_results': False
    }
}


def get_preset_config(preset_name: str) -> AudioSystemConfig:
    """Get a preset configuration.
    
    Args:
        preset_name: Name of the preset configuration
        
    Returns:
        AudioSystemConfig with preset values
        
    Raises:
        ValueError: If preset_name is not found
    """
    if preset_name not in PROVIDER_CONFIGS:
        available = ', '.join(PROVIDER_CONFIGS.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
    
    preset = PROVIDER_CONFIGS[preset_name]
    config = AudioSystemConfig()
    
    # Update config with preset values
    for key, value in preset.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config