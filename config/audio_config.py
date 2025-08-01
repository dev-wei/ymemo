"""Configuration system for audio processing providers."""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


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
    
    # AWS Connection Strategy settings  
    aws_connection_strategy: str = 'auto'  # 'auto', 'single', 'dual'
    aws_dual_fallback_enabled: bool = True
    aws_channel_balance_threshold: float = 0.3  # Threshold for detecting severe channel imbalance
    
    # AWS Dual Connection Test Mode settings
    aws_dual_connection_test_mode: str = 'full'  # 'left_only', 'right_only', 'full'
    
    # AWS Dual Connection Audio Saving settings (for debugging)
    aws_dual_save_split_audio: bool = False
    aws_dual_save_raw_audio: bool = False  # Save raw PyAudio input before splitting
    aws_dual_audio_save_path: str = './debug_audio/'
    aws_dual_audio_save_duration: int = 30  # seconds
    
    # Azure Speech Service settings
    azure_speech_key: str = ''
    azure_speech_region: str = 'eastus'
    azure_speech_language: str = 'en-US'
    azure_speech_endpoint: Optional[str] = None
    azure_enable_speaker_diarization: bool = False
    azure_max_speakers: int = 4
    azure_speech_timeout: int = 30
    
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
        
        # Validate configuration after initialization
        self.validate()
    
    @classmethod
    def _safe_int(cls, value: str, default: int) -> int:
        """Safely parse integer value with fallback to default."""
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid integer value '{value}', using default {default}")
            return default
    
    @classmethod
    def _safe_float(cls, value: str, default: float) -> float:
        """Safely parse float value with fallback to default."""
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid float value '{value}', using default {default}")
            return default
    
    @classmethod
    def _safe_bool(cls, value: str) -> bool:
        """Safely parse boolean value."""
        if value is None:
            return False
        return str(value).lower() in ('true', '1', 'yes', 'on')

    @classmethod
    def from_env(cls) -> 'AudioSystemConfig':
        """Create configuration from environment variables."""
        return cls(
            transcription_provider=os.getenv('TRANSCRIPTION_PROVIDER', 'aws'),
            capture_provider=os.getenv('CAPTURE_PROVIDER', 'pyaudio'),
            sample_rate=cls._safe_int(os.getenv('AUDIO_SAMPLE_RATE', '16000'), 16000),
            channels=cls._safe_int(os.getenv('AUDIO_CHANNELS', '1'), 1),
            chunk_size=cls._safe_int(os.getenv('AUDIO_CHUNK_SIZE', '1024'), 1024),
            audio_format=os.getenv('AUDIO_FORMAT', 'int16'),
            aws_region=os.getenv('AWS_REGION', 'us-east-1'),
            aws_language_code=os.getenv('AWS_LANGUAGE_CODE', 'en-US'),
            aws_max_speakers=cls._safe_int(os.getenv('AWS_MAX_SPEAKERS', '10'), 10),
            aws_connection_strategy=os.getenv('AWS_CONNECTION_STRATEGY', 'auto'),
            aws_dual_fallback_enabled=cls._safe_bool(os.getenv('AWS_DUAL_FALLBACK_ENABLED', 'true')),
            aws_channel_balance_threshold=cls._safe_float(os.getenv('AWS_CHANNEL_BALANCE_THRESHOLD', '0.3'), 0.3),
            aws_dual_connection_test_mode=os.getenv('AWS_DUAL_CONNECTION_TEST_MODE', 'full'),
            aws_dual_save_split_audio=cls._safe_bool(os.getenv('AWS_DUAL_SAVE_SPLIT_AUDIO', 'false')),
            aws_dual_save_raw_audio=cls._safe_bool(os.getenv('AWS_DUAL_SAVE_RAW_AUDIO', 'false')),
            aws_dual_audio_save_path=os.getenv('AWS_DUAL_AUDIO_SAVE_PATH', './debug_audio/'),
            aws_dual_audio_save_duration=cls._safe_int(os.getenv('AWS_DUAL_AUDIO_SAVE_DURATION', '30'), 30),
            azure_speech_key=os.getenv('AZURE_SPEECH_KEY', ''),
            azure_speech_region=os.getenv('AZURE_SPEECH_REGION', 'eastus'),
            azure_speech_language=os.getenv('AZURE_SPEECH_LANGUAGE', 'en-US'),
            azure_speech_endpoint=os.getenv('AZURE_SPEECH_ENDPOINT'),
            azure_enable_speaker_diarization=cls._safe_bool(os.getenv('AZURE_ENABLE_SPEAKER_DIARIZATION', 'false')),
            azure_max_speakers=cls._safe_int(os.getenv('AZURE_MAX_SPEAKERS', '4'), 4),
            azure_speech_timeout=cls._safe_int(os.getenv('AZURE_SPEECH_TIMEOUT', '30'), 30),
            max_latency_ms=cls._safe_int(os.getenv('MAX_LATENCY_MS', '300'), 300),
            enable_partial_results=cls._safe_bool(os.getenv('ENABLE_PARTIAL_RESULTS', 'true')),
            partial_result_handling=os.getenv('PARTIAL_RESULT_HANDLING', 'replace'),
            partial_result_timeout=cls._safe_float(os.getenv('PARTIAL_RESULT_TIMEOUT', '2.0'), 2.0),
            confidence_threshold=cls._safe_float(os.getenv('CONFIDENCE_THRESHOLD', '0.0'), 0.0)
        )
    
    def get_transcription_config(self) -> Dict[str, Any]:
        """Get configuration for transcription provider."""
        if self.transcription_provider == 'aws':
            return {
                'region': self.aws_region,
                'language_code': self.aws_language_code,
                'connection_strategy': self.aws_connection_strategy,
                'dual_fallback_enabled': self.aws_dual_fallback_enabled,
                'channel_balance_threshold': self.aws_channel_balance_threshold,
                'dual_connection_test_mode': self.aws_dual_connection_test_mode,
                'dual_save_split_audio': self.aws_dual_save_split_audio,
                'dual_save_raw_audio': self.aws_dual_save_raw_audio,
                'dual_audio_save_path': self.aws_dual_audio_save_path,
                'dual_audio_save_duration': self.aws_dual_audio_save_duration
            }
        elif self.transcription_provider == 'azure':
            return {
                'speech_key': self.azure_speech_key,
                'region': self.azure_speech_region,
                'language_code': self.azure_speech_language,
                'endpoint': self.azure_speech_endpoint,
                'enable_speaker_diarization': self.azure_enable_speaker_diarization,
                'max_speakers': self.azure_max_speakers,
                'timeout': self.azure_speech_timeout
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
    
    def get_device_optimized_audio_config(self, device_id: Optional[int] = None) -> 'AudioConfig':
        """Get audio configuration optimized for a specific device.
        
        Args:
            device_id: Target audio device ID for optimization
            
        Returns:
            AudioConfig optimized for the specified device
        """
        from src.core.interfaces import AudioConfig
        
        # Start with base configuration
        base_config = self.get_audio_config()
        
        # If no device specified, return base config
        if device_id is None:
            logger.debug("🔧 AudioConfig: No device specified, using base configuration")
            return base_config
        
        try:
            from src.utils.device_utils import validate_device_config, get_device_max_channels
            
            # Get device's maximum channels
            device_max_channels = get_device_max_channels(device_id)
            
            # For 1-2 channel support, use device capability but cap at 2
            preferred_channels = min(device_max_channels, 2)
            
            logger.info(f"🔧 AudioConfig: Device {device_id} supports {device_max_channels} channels, "
                       f"using: {preferred_channels} (capped at 2 for direct AWS processing)")
            
            # Validate configuration for the device
            validation_result = validate_device_config(
                device_index=device_id,
                channels=preferred_channels,
                sample_rate=self.sample_rate
            )
            
            # Log device information and warnings
            device_info = validation_result.get('device_info', {})
            if device_info:
                logger.info(f"🎤 AudioConfig: Using device {device_id} - {device_info.get('name', 'Unknown')} "
                           f"({device_info.get('max_input_channels', 'unknown')} channels available)")
            
            for warning in validation_result.get('warnings', []):
                logger.warning(f"⚠️ AudioConfig: {warning}")
            
            # Create optimized configuration
            optimized_config = AudioConfig(
                sample_rate=validation_result.get('sample_rate', self.sample_rate),
                channels=validation_result.get('channels', self.channels),
                chunk_size=self.chunk_size,
                format=self.audio_format
            )
            
            if optimized_config.channels != base_config.channels:
                logger.info(f"🔧 AudioConfig: Channel count set to {optimized_config.channels} for device {device_id}")
            
            return optimized_config
            
        except Exception as e:
            logger.error(f"❌ AudioConfig: Device optimization failed for device {device_id}: {e}")
            logger.info("🔧 AudioConfig: Falling back to safe mono configuration")
            
            # Return safe fallback configuration
            return AudioConfig(
                sample_rate=self.sample_rate,
                channels=1,  # Safe fallback to mono
                chunk_size=self.chunk_size,
                format=self.audio_format
            )
    
    def validate(self) -> None:
        """Validate configuration values and raise descriptive errors."""
        errors = []
        
        # Validate provider selection
        valid_transcription_providers = ['aws', 'azure', 'whisper', 'google']
        if self.transcription_provider not in valid_transcription_providers:
            errors.append(
                f"Invalid transcription_provider '{self.transcription_provider}'. "
                f"Valid options: {', '.join(valid_transcription_providers)}"
            )
        
        valid_capture_providers = ['pyaudio', 'file']
        if self.capture_provider not in valid_capture_providers:
            errors.append(
                f"Invalid capture_provider '{self.capture_provider}'. "
                f"Valid options: {', '.join(valid_capture_providers)}"
            )
        
        # Validate audio settings
        if self.sample_rate <= 0:
            errors.append(f"Sample rate must be positive, got {self.sample_rate}")
        
        if self.channels <= 0:
            errors.append(f"Number of channels must be positive, got {self.channels}")
        
        if self.chunk_size <= 0:
            errors.append(f"Chunk size must be positive, got {self.chunk_size}")
        
        valid_formats = ['int16', 'int24', 'int32', 'float32']
        if self.audio_format not in valid_formats:
            errors.append(
                f"Invalid audio_format '{self.audio_format}'. "
                f"Valid options: {', '.join(valid_formats)}"
            )
        
        # Validate AWS settings if using AWS
        if self.transcription_provider == 'aws':
            if not self.aws_region:
                errors.append("AWS region is required when using AWS transcription provider")
            
            if not self.aws_language_code:
                errors.append("AWS language code is required when using AWS transcription provider")
            
            # Validate AWS connection strategy
            valid_strategies = ['auto', 'single', 'dual']
            if self.aws_connection_strategy not in valid_strategies:
                errors.append(
                    f"Invalid aws_connection_strategy '{self.aws_connection_strategy}'. "
                    f"Valid options: {', '.join(valid_strategies)}"
                )
            
            # Validate AWS dual connection test mode
            valid_test_modes = ['left_only', 'right_only', 'full']
            if self.aws_dual_connection_test_mode not in valid_test_modes:
                errors.append(
                    f"Invalid aws_dual_connection_test_mode '{self.aws_dual_connection_test_mode}'. "
                    f"Valid options: {', '.join(valid_test_modes)}"
                )
            
            # Validate audio saving settings
            if not isinstance(self.aws_dual_save_split_audio, bool):
                errors.append("aws_dual_save_split_audio must be a boolean")
            
            if not isinstance(self.aws_dual_save_raw_audio, bool):
                errors.append("aws_dual_save_raw_audio must be a boolean")
            
            if self.aws_dual_audio_save_duration <= 0:
                errors.append(f"aws_dual_audio_save_duration must be positive, got {self.aws_dual_audio_save_duration}")
            
            if not self.aws_dual_audio_save_path or not isinstance(self.aws_dual_audio_save_path, str):
                errors.append("aws_dual_audio_save_path must be a non-empty string")
            
            # Validate channel balance threshold
            if not (0.0 <= self.aws_channel_balance_threshold <= 1.0):
                errors.append(
                    f"aws_channel_balance_threshold must be between 0.0 and 1.0, got {self.aws_channel_balance_threshold}"
                )
            
            # Note: AWS provider now handles both single and dual connections automatically
        
        # Validate Azure settings if using Azure
        if self.transcription_provider == 'azure':
            if not self.azure_speech_key:
                errors.append("Azure speech key is required when using Azure transcription provider")
            
            if not self.azure_speech_region:
                errors.append("Azure speech region is required when using Azure transcription provider")
        
        # Validate performance settings
        if self.max_latency_ms <= 0:
            errors.append(f"Max latency must be positive, got {self.max_latency_ms}")
        
        if self.partial_result_timeout <= 0:
            errors.append(f"Partial result timeout must be positive, got {self.partial_result_timeout}")
        
        if not (0.0 <= self.confidence_threshold <= 1.0):
            errors.append(f"Confidence threshold must be between 0.0 and 1.0, got {self.confidence_threshold}")
        
        # Log warnings for potentially problematic configurations
        if self.sample_rate != 16000:
            logger.warning(f"Non-standard sample rate {self.sample_rate}Hz may cause issues with transcription providers")
        
        if self.channels > 2:
            logger.warning(f"Multi-channel audio ({self.channels} channels) not supported. Only 1-2 channels allowed.")
        
        if errors:
            error_message = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ValueError(error_message)
        
        logger.debug("Configuration validation passed")
    
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
    config = AudioSystemConfig.from_env()
    
    # Log configuration for debugging (mask sensitive values)
    masked_config = config.to_dict().copy()
    if masked_config.get('azure_speech_key'):
        masked_config['azure_speech_key'] = '***masked***'
    
    logger.info("🔧 Loaded audio system configuration:")
    logger.info(f"  - Transcription Provider: {config.transcription_provider}")
    logger.info(f"  - Capture Provider: {config.capture_provider}")
    logger.info(f"  - AWS Region: {config.aws_region}")
    logger.info(f"  - AWS Language: {config.aws_language_code}")
    logger.info(f"  - AWS Connection Strategy: {config.aws_connection_strategy}")
    logger.info(f"  - AWS Dual Test Mode: {config.aws_dual_connection_test_mode}")
    
    # Enhanced audio saving configuration logging
    if config.aws_dual_save_split_audio or config.aws_dual_save_raw_audio:
        logger.info(f"  - AWS Audio Saving: ✅ ENABLED")
        logger.info(f"    📁 Save Path: {config.aws_dual_audio_save_path}")
        logger.info(f"    ⏱️  Save Duration: {config.aws_dual_audio_save_duration}s")
        if config.aws_dual_save_split_audio:
            logger.info(f"    🔀 Split audio saving: ✅ ENABLED")
        if config.aws_dual_save_raw_audio:
            logger.info(f"    🎵 Raw audio saving: ✅ ENABLED")
        
        # Validate directory exists
        import os
        if not os.path.exists(config.aws_dual_audio_save_path):
            logger.warning(f"    ⚠️  Save directory does not exist: {config.aws_dual_audio_save_path}")
            try:
                os.makedirs(config.aws_dual_audio_save_path, exist_ok=True)
                logger.info(f"    📁 Created save directory: {config.aws_dual_audio_save_path}")
            except Exception as e:
                logger.error(f"    ❌ Failed to create save directory: {e}")
    else:
        logger.info(f"  - AWS Audio Saving: ❌ DISABLED")
        logger.info(f"    💡 To enable: set AWS_DUAL_SAVE_SPLIT_AUDIO=true or AWS_DUAL_SAVE_RAW_AUDIO=true")
    
    logger.info(f"  - Audio Format: {config.sample_rate}Hz, {config.channels}ch, {config.audio_format}")
    logger.info(f"  - Chunk Size: {config.chunk_size}")
    logger.info(f"  - Max Latency: {config.max_latency_ms}ms")
    logger.info(f"  - Partial Results: {config.enable_partial_results}")
    
    logger.debug(f"Full configuration (sensitive values masked): {masked_config}")
    
    return config


def print_config_summary() -> None:
    """Print a human-readable configuration summary for debugging."""
    config = get_config()
    print("=== Audio System Configuration Summary ===")
    print(f"Transcription Provider: {config.transcription_provider}")
    print(f"Capture Provider: {config.capture_provider}")
    print(f"")
    print(f"Audio Settings:")
    print(f"  - Sample Rate: {config.sample_rate} Hz")
    print(f"  - Channels: {config.channels}")
    print(f"  - Format: {config.audio_format}")
    print(f"  - Chunk Size: {config.chunk_size}")
    print(f"")
    print(f"AWS Configuration:")
    print(f"  - Region: {config.aws_region}")
    print(f"  - Language: {config.aws_language_code}")
    print(f"  - Connection Strategy: {config.aws_connection_strategy}")
    print(f"  - Dual Connection Test Mode: {config.aws_dual_connection_test_mode}")
    if config.aws_dual_save_split_audio or config.aws_dual_save_raw_audio:
        print(f"  - Audio Saving: ENABLED (path: {config.aws_dual_audio_save_path}, duration: {config.aws_dual_audio_save_duration}s)")
        if config.aws_dual_save_split_audio:
            print(f"    - Split audio: ENABLED")
        if config.aws_dual_save_raw_audio:
            print(f"    - Raw audio: ENABLED")
    print(f"")
    print(f"Performance Settings:")
    print(f"  - Max Latency: {config.max_latency_ms} ms")
    print(f"  - Partial Results: {config.enable_partial_results}")
    print(f"  - Confidence Threshold: {config.confidence_threshold}")
    print("=============================================")


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