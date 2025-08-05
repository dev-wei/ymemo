"""Configuration system for audio processing providers."""

import logging
import os
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.interfaces import AudioConfig

logger = logging.getLogger(__name__)

# Audio quality constants
AUDIO_QUALITY_HIGH = "high"
AUDIO_QUALITY_AVERAGE = "average"
SAMPLE_RATE_HIGH = 44100  # CD-quality audio
SAMPLE_RATE_AVERAGE = 16000  # Speech-optimized
SAMPLE_RATE_THRESHOLD = 32000  # Threshold for determining quality from custom rates

# Audio quality mappings
QUALITY_SAMPLE_RATE_MAP = {
    AUDIO_QUALITY_HIGH: SAMPLE_RATE_HIGH,
    AUDIO_QUALITY_AVERAGE: SAMPLE_RATE_AVERAGE,
}

# UI display constants
QUALITY_DISPLAY_HIGH = "High"
QUALITY_DISPLAY_AVERAGE = "Average"

QUALITY_DISPLAY_MAP = {
    AUDIO_QUALITY_HIGH: QUALITY_DISPLAY_HIGH,
    AUDIO_QUALITY_AVERAGE: QUALITY_DISPLAY_AVERAGE,
}


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

    # AWS Connection Strategy: Now automatically determined based on device channels
    # - 1 channel device → Single AWS connection
    # - 2+ channel device → Dual AWS connections
    aws_dual_fallback_enabled: bool = True
    aws_channel_balance_threshold: float = (
        0.3  # Threshold for detecting severe channel imbalance
    )

    # General Audio Saving settings (provider-agnostic)
    save_raw_audio: bool = False  # Save raw audio input to WAV file
    save_split_audio: bool = False  # Save left/right channels separately (stereo only)
    audio_save_path: str = './debug_audio/'  # Directory to save audio files
    audio_save_duration: int = 30  # Maximum recording duration in seconds

    # Azure Speech Service settings
    azure_speech_key: str = ''
    azure_speech_region: str = 'eastus'
    azure_speech_language: str = 'en-US'
    azure_speech_endpoint: str | None = None
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

    # Silence detection and auto-stop settings
    silence_timeout_seconds: int = (
        300  # Auto-stop recording after this many seconds of silence (0 = disabled)
    )

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
    def _safe_bool(cls, value: str | None) -> bool:
        """Safely parse boolean value."""
        if value is None:
            return False
        return str(value).lower() in ('true', '1', 'yes', 'on')

    @classmethod
    def from_env(cls) -> 'AudioSystemConfig':
        """Create configuration from environment variables."""
        # Handle audio quality presets first, then fall back to sample rate
        audio_quality = os.getenv('AUDIO_QUALITY', '').lower()
        sample_rate = QUALITY_SAMPLE_RATE_MAP.get(
            audio_quality,
            cls._safe_int(
                os.getenv('AUDIO_SAMPLE_RATE', str(SAMPLE_RATE_AVERAGE)),
                SAMPLE_RATE_AVERAGE,
            ),
        )

        # Show informational message if user still has deprecated AWS_CONNECTION_STRATEGY
        if 'AWS_CONNECTION_STRATEGY' in os.environ:
            logger.warning(
                "⚠️ AWS_CONNECTION_STRATEGY is no longer supported and will be ignored. "
                "Connection strategy is now automatically determined based on device channels: "
                "1 channel = single connection, 2+ channels = dual connections."
            )

        return cls(
            transcription_provider=os.getenv('TRANSCRIPTION_PROVIDER', 'aws'),
            capture_provider=os.getenv('CAPTURE_PROVIDER', 'pyaudio'),
            sample_rate=sample_rate,
            channels=1,  # Default fallback - actual channels determined by device detection
            chunk_size=cls._safe_int(os.getenv('AUDIO_CHUNK_SIZE', '1024'), 1024),
            audio_format=os.getenv('AUDIO_FORMAT', 'int16'),
            aws_region=os.getenv('AWS_REGION', 'us-east-1'),
            aws_language_code=os.getenv('AWS_LANGUAGE_CODE', 'en-US'),
            aws_max_speakers=cls._safe_int(os.getenv('AWS_MAX_SPEAKERS', '10'), 10),
            aws_dual_fallback_enabled=cls._safe_bool(
                os.getenv('AWS_DUAL_FALLBACK_ENABLED', 'true')
            ),
            aws_channel_balance_threshold=cls._safe_float(
                os.getenv('AWS_CHANNEL_BALANCE_THRESHOLD', '0.3'), 0.3
            ),
            # General audio saving settings (provider-agnostic)
            save_raw_audio=cls._safe_bool(os.getenv('SAVE_RAW_AUDIO', 'false')),
            save_split_audio=cls._safe_bool(os.getenv('SAVE_SPLIT_AUDIO', 'false')),
            audio_save_path=os.getenv('AUDIO_SAVE_PATH', './debug_audio/'),
            audio_save_duration=cls._safe_int(
                os.getenv('AUDIO_SAVE_DURATION', '30'), 30
            ),
            azure_speech_key=os.getenv('AZURE_SPEECH_KEY', ''),
            azure_speech_region=os.getenv('AZURE_SPEECH_REGION', 'eastus'),
            azure_speech_language=os.getenv('AZURE_SPEECH_LANGUAGE', 'en-US'),
            azure_speech_endpoint=os.getenv('AZURE_SPEECH_ENDPOINT'),
            azure_enable_speaker_diarization=cls._safe_bool(
                os.getenv('AZURE_ENABLE_SPEAKER_DIARIZATION', 'false')
            ),
            azure_max_speakers=cls._safe_int(os.getenv('AZURE_MAX_SPEAKERS', '4'), 4),
            azure_speech_timeout=cls._safe_int(
                os.getenv('AZURE_SPEECH_TIMEOUT', '30'), 30
            ),
            max_latency_ms=cls._safe_int(os.getenv('MAX_LATENCY_MS', '300'), 300),
            enable_partial_results=cls._safe_bool(
                os.getenv('ENABLE_PARTIAL_RESULTS', 'true')
            ),
            partial_result_handling=os.getenv('PARTIAL_RESULT_HANDLING', 'replace'),
            partial_result_timeout=cls._safe_float(
                os.getenv('PARTIAL_RESULT_TIMEOUT', '2.0'), 2.0
            ),
            confidence_threshold=cls._safe_float(
                os.getenv('CONFIDENCE_THRESHOLD', '0.0'), 0.0
            ),
            silence_timeout_seconds=cls._safe_int(
                os.getenv('SILENCE_TIMEOUT_SECONDS', '300'), 300
            ),
        )

    def get_transcription_config(self) -> dict[str, Any]:
        """Get configuration for transcription provider."""
        if self.transcription_provider == 'aws':
            return {
                'region': self.aws_region,
                'language_code': self.aws_language_code,
                # Note: connection_strategy removed - now auto-detected based on device channels
                'dual_fallback_enabled': self.aws_dual_fallback_enabled,
                'channel_balance_threshold': self.aws_channel_balance_threshold,
                # NOTE: Audio saving is now handled at pipeline level by AudioSaver component
            }
        if self.transcription_provider == 'azure':
            return {
                'speech_key': self.azure_speech_key,
                'region': self.azure_speech_region,
                'language_code': self.azure_speech_language,
                'endpoint': self.azure_speech_endpoint,
                'enable_speaker_diarization': self.azure_enable_speaker_diarization,
                'max_speakers': self.azure_max_speakers,
                'timeout': self.azure_speech_timeout,
            }
        if self.transcription_provider == 'whisper':
            return {
                'model_size': os.getenv('WHISPER_MODEL_SIZE', 'base'),
                'device': os.getenv('WHISPER_DEVICE', 'auto'),
            }
        if self.transcription_provider == 'google':
            return {
                'language_code': self.aws_language_code,
                'credentials_path': os.getenv('GOOGLE_CREDENTIALS_PATH'),
            }
        return {}

    def get_capture_config(self) -> dict[str, Any]:
        """Get configuration for audio capture provider."""
        return {
            # PyAudio or other capture providers may need specific config
        }

    def get_audio_saving_config(self) -> dict[str, Any]:
        """Get configuration for audio saving component."""
        return {
            'enabled': self.save_raw_audio,
            'save_split_audio': self.save_split_audio,
            'save_path': self.audio_save_path,
            'max_duration': self.audio_save_duration,
        }

    def get_audio_config(self) -> 'AudioConfig':
        """Get audio configuration as AudioConfig object."""
        from src.core.interfaces import AudioConfig

        return AudioConfig(
            sample_rate=self.sample_rate,
            channels=self.channels,
            chunk_size=self.chunk_size,
            format=self.audio_format,
        )

    def get_device_optimized_audio_config(
        self, device_id: int | None = None
    ) -> 'AudioConfig':
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
            logger.debug(
                "🔧 AudioConfig: No device specified, using base configuration"
            )
            return base_config

        try:
            from src.utils.device_utils import (
                get_device_max_channels,
                validate_device_config,
            )

            # Get device's maximum channels
            device_max_channels = get_device_max_channels(device_id)

            # For 1-2 channel support, use device capability but cap at 2
            preferred_channels = min(device_max_channels, 2)

            logger.info(
                f"🔧 AudioConfig: Device {device_id} supports {device_max_channels} channels, "
                f"using: {preferred_channels} (capped at 2 for direct AWS processing)"
            )

            # Validate configuration for the device
            validation_result = validate_device_config(
                device_index=device_id,
                channels=preferred_channels,
                sample_rate=self.sample_rate,
            )

            # Log device sample rate compatibility for audio debugging
            device_default_rate = validation_result.get('device_default_sample_rate')
            if device_default_rate:
                logger.info(f"🔍 AudioConfig: Device sample rate analysis:")
                logger.info(f"   🎚️ System requests: {self.sample_rate}Hz")
                logger.info(f"   🎛️ Device default: {device_default_rate}Hz")
                if abs(self.sample_rate - device_default_rate) > 1000:
                    logger.warning(
                        f"   ⚠️ Large mismatch may cause audio quality issues!"
                    )
                else:
                    logger.info(f"   ✅ Sample rates are compatible")

            # Log device information and warnings
            device_info = validation_result.get('device_info', {})
            if device_info:
                logger.info(
                    f"🎤 AudioConfig: Using device {device_id} - {device_info.get('name', 'Unknown')} "
                    f"({device_info.get('max_input_channels', 'unknown')} channels available)"
                )

            for warning in validation_result.get('warnings', []):
                logger.warning(f"⚠️ AudioConfig: {warning}")

            # Create optimized configuration
            optimized_config = AudioConfig(
                sample_rate=validation_result.get('sample_rate', self.sample_rate),
                channels=validation_result.get('channels', self.channels),
                chunk_size=self.chunk_size,
                format=self.audio_format,
            )

            if optimized_config.channels != base_config.channels:
                logger.info(
                    f"🔧 AudioConfig: Channel count set to {optimized_config.channels} for device {device_id}"
                )

            return optimized_config

        except Exception as e:
            logger.error(
                f"❌ AudioConfig: Device optimization failed for device {device_id}: {e}"
            )
            logger.info("🔧 AudioConfig: Falling back to safe mono configuration")

            # Return safe fallback configuration
            return AudioConfig(
                sample_rate=self.sample_rate,
                channels=1,  # Safe fallback to mono
                chunk_size=self.chunk_size,
                format=self.audio_format,
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
                errors.append(
                    "AWS region is required when using AWS transcription provider"
                )

            if not self.aws_language_code:
                errors.append(
                    "AWS language code is required when using AWS transcription provider"
                )

            # Validate channel balance threshold
            if not (0.0 <= self.aws_channel_balance_threshold <= 1.0):
                errors.append(
                    f"aws_channel_balance_threshold must be between 0.0 and 1.0, got {self.aws_channel_balance_threshold}"
                )

            # Note: AWS provider now handles both single and dual connections automatically

        # Validate general audio saving settings
        if not isinstance(self.save_raw_audio, bool):
            errors.append("save_raw_audio must be a boolean")

        if not isinstance(self.save_split_audio, bool):
            errors.append("save_split_audio must be a boolean")

        if self.audio_save_duration <= 0:
            errors.append(
                f"audio_save_duration must be positive, got {self.audio_save_duration}"
            )

        if not self.audio_save_path or not isinstance(self.audio_save_path, str):
            errors.append("audio_save_path must be a non-empty string")

        # Validate Azure settings if using Azure
        if self.transcription_provider == 'azure':
            if not self.azure_speech_key:
                errors.append(
                    "Azure speech key is required when using Azure transcription provider"
                )

            if not self.azure_speech_region:
                errors.append(
                    "Azure speech region is required when using Azure transcription provider"
                )

        # Validate performance settings
        if self.max_latency_ms <= 0:
            errors.append(f"Max latency must be positive, got {self.max_latency_ms}")

        if self.partial_result_timeout <= 0:
            errors.append(
                f"Partial result timeout must be positive, got {self.partial_result_timeout}"
            )

        if not (0.0 <= self.confidence_threshold <= 1.0):
            errors.append(
                f"Confidence threshold must be between 0.0 and 1.0, got {self.confidence_threshold}"
            )

        # Validate silence detection settings
        if self.silence_timeout_seconds < 0:
            errors.append(
                f"Silence timeout must be non-negative (0 disables feature), got {self.silence_timeout_seconds}"
            )

        # Log warnings for potentially problematic configurations
        if self.sample_rate != 16000:
            logger.warning(
                f"Non-standard sample rate {self.sample_rate}Hz may cause issues with transcription providers"
            )

        if self.channels > 2:
            logger.warning(
                f"Multi-channel audio ({self.channels} channels) not supported. Only 1-2 channels allowed."
            )

        if errors:
            error_message = "Configuration validation failed:\n" + "\n".join(
                f"  - {error}" for error in errors
            )
            raise ValueError(error_message)

        logger.debug("Configuration validation passed")

    def to_dict(self) -> dict[str, Any]:
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
    logger.info(f"  - AWS Connection Strategy: auto-detected based on device channels")

    # General audio saving configuration logging
    if config.save_raw_audio or config.save_split_audio:
        logger.info("  - Audio Saving: ✅ ENABLED")
        logger.info(f"    📁 Save Path: {config.audio_save_path}")
        logger.info(f"    ⏱️  Save Duration: {config.audio_save_duration}s")

        if config.save_raw_audio:
            logger.info("    🎵 Raw audio saving: ✅ ENABLED (provider-agnostic)")

        if config.save_split_audio:
            logger.info("    🔀 Split audio saving: ✅ ENABLED (provider-agnostic)")

        # Validate directory exists
        import os

        if not os.path.exists(config.audio_save_path):
            logger.warning(
                f"    ⚠️  Save directory does not exist: {config.audio_save_path}"
            )
            try:
                os.makedirs(config.audio_save_path, exist_ok=True)
                logger.info(f"    📁 Created save directory: {config.audio_save_path}")
            except Exception as e:
                logger.error(f"    ❌ Failed to create save directory: {e}")
    else:
        logger.info("  - Audio Saving: ❌ DISABLED")
        logger.info("    💡 To enable: set SAVE_RAW_AUDIO=true")

    logger.info(
        f"  - Audio Format: {config.sample_rate}Hz, {config.channels}ch, {config.audio_format}"
    )
    logger.info(f"  - Chunk Size: {config.chunk_size}")
    logger.info(f"  - Max Latency: {config.max_latency_ms}ms")
    logger.info(f"  - Partial Results: {config.enable_partial_results}")

    if config.silence_timeout_seconds > 0:
        logger.info(
            f"  - Silence Auto-Stop: ✅ ENABLED ({config.silence_timeout_seconds}s timeout)"
        )
    else:
        logger.info("  - Silence Auto-Stop: ❌ DISABLED")

    logger.debug(f"Full configuration (sensitive values masked): {masked_config}")

    return config


def print_config_summary() -> None:
    """Print a human-readable configuration summary for debugging."""
    config = get_config()
    print("=== Audio System Configuration Summary ===")
    print(f"Transcription Provider: {config.transcription_provider}")
    print(f"Capture Provider: {config.capture_provider}")
    print("")
    print("Audio Settings:")
    print(f"  - Sample Rate: {config.sample_rate} Hz")
    print(f"  - Channels: {config.channels}")
    print(f"  - Format: {config.audio_format}")
    print(f"  - Chunk Size: {config.chunk_size}")
    print("")
    print("AWS Configuration:")
    print(f"  - Region: {config.aws_region}")
    print(f"  - Language: {config.aws_language_code}")
    print(f"  - Connection Strategy: auto-detected based on device channels")
    print("")
    print("Audio Saving Configuration:")
    if config.save_raw_audio:
        print(
            f"  - Raw Audio Saving: ENABLED (path: {config.audio_save_path}, duration: {config.audio_save_duration}s)"
        )
    else:
        print("  - Raw Audio Saving: DISABLED")

    if config.save_split_audio:
        print(f"  - Split Audio Saving: ENABLED (stereo channels saved separately)")
    else:
        print("  - Split Audio Saving: DISABLED")
    print("")
    print("Performance Settings:")
    print(f"  - Max Latency: {config.max_latency_ms} ms")
    print(f"  - Partial Results: {config.enable_partial_results}")
    print(f"  - Confidence Threshold: {config.confidence_threshold}")
    print("")
    print("Safety Settings:")
    if config.silence_timeout_seconds > 0:
        print(
            f"  - Silence Auto-Stop: ENABLED ({config.silence_timeout_seconds}s timeout)"
        )
    else:
        print("  - Silence Auto-Stop: DISABLED")
    print("=============================================")


# Provider-specific configuration templates
PROVIDER_CONFIGS = {
    'aws_transcribe_streaming': {
        'transcription_provider': 'aws',
        'aws_region': 'us-east-1',
        'aws_language_code': 'en-US',
        'max_latency_ms': 200,
        'enable_partial_results': True,
    },
    'whisper_local': {
        'transcription_provider': 'whisper',
        'max_latency_ms': 500,
        'enable_partial_results': False,
    },
    'google_streaming': {
        'transcription_provider': 'google',
        'max_latency_ms': 300,
        'enable_partial_results': True,
    },
    'high_performance': {
        'transcription_provider': 'aws',
        'sample_rate': 16000,
        'chunk_size': 512,  # Smaller chunks for lower latency
        'max_latency_ms': 100,
        'enable_partial_results': True,
    },
    'cost_optimized': {
        'transcription_provider': 'whisper',
        'sample_rate': 16000,
        'chunk_size': 2048,  # Larger chunks for efficiency
        'max_latency_ms': 1000,
        'enable_partial_results': False,
    },
    # Audio quality presets
    'audio_quality_high': {
        'sample_rate': 44100,  # CD-quality audio
        'chunk_size': 1024,
        'max_latency_ms': 300,
        'enable_partial_results': True,
    },
    'audio_quality_average': {
        'sample_rate': 16000,  # Speech-optimized
        'chunk_size': 1024,
        'max_latency_ms': 300,
        'enable_partial_results': True,
    },
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


# Audio quality configuration functions
def get_audio_quality_choices() -> list[str]:
    """Get available audio quality choices for UI dropdown."""
    return [QUALITY_DISPLAY_HIGH, QUALITY_DISPLAY_AVERAGE]


def get_default_audio_quality() -> str:
    """Get default audio quality setting."""
    return QUALITY_DISPLAY_AVERAGE


def get_current_audio_quality_from_sample_rate(sample_rate: int) -> str:
    """Get audio quality label from sample rate."""
    if sample_rate == SAMPLE_RATE_HIGH:
        return QUALITY_DISPLAY_HIGH
    elif sample_rate == SAMPLE_RATE_AVERAGE:
        return QUALITY_DISPLAY_AVERAGE
    else:
        # For custom sample rates, determine closest quality
        return (
            QUALITY_DISPLAY_HIGH
            if sample_rate >= SAMPLE_RATE_THRESHOLD
            else QUALITY_DISPLAY_AVERAGE
        )


def get_sample_rate_from_quality(quality: str) -> int:
    """Get sample rate from audio quality setting."""
    quality_lower = quality.lower()
    return QUALITY_SAMPLE_RATE_MAP.get(quality_lower, SAMPLE_RATE_AVERAGE)


def get_current_audio_quality_info() -> dict[str, str]:
    """Get current audio quality information for UI display."""
    config = get_config()
    current_quality = get_current_audio_quality_from_sample_rate(config.sample_rate)

    if current_quality == QUALITY_DISPLAY_HIGH:
        description = (
            f"High Quality ({SAMPLE_RATE_HIGH:,} Hz) - CD-quality audio capture"
        )
    elif current_quality == QUALITY_DISPLAY_AVERAGE:
        description = f"Average Quality ({SAMPLE_RATE_AVERAGE:,} Hz) - Speech-optimized"
    else:
        description = f"Custom ({config.sample_rate:,} Hz)"

    return {
        "quality": current_quality,
        "sample_rate": f"{config.sample_rate:,} Hz",
        "description": description,
    }
