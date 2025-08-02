"""Standard test configurations for consistent testing across the suite.

This module provides centralized configuration objects that can be reused
across all test files, ensuring consistency and reducing duplication.
"""

from typing import Any

from src.core.interfaces import AudioConfig


class TestAudioConfigs:
    """Standard AudioConfig instances for testing."""

    # Default configuration used in most tests
    DEFAULT = AudioConfig(
        sample_rate=16000, channels=1, chunk_size=1024, format="int16"
    )

    # High-quality configuration for performance tests
    HIGH_QUALITY = AudioConfig(
        sample_rate=44100, channels=2, chunk_size=2048, format="int24"
    )

    # Low-quality configuration for basic tests
    LOW_QUALITY = AudioConfig(
        sample_rate=8000, channels=1, chunk_size=512, format="int16"
    )

    # Configuration for AWS Transcribe tests
    AWS_COMPATIBLE = AudioConfig(
        sample_rate=16000,  # AWS Transcribe requirement
        channels=1,  # Mono audio
        chunk_size=1024,
        format="int16",
    )

    # Configuration for file-based testing
    FILE_TEST = AudioConfig(
        sample_rate=22050, channels=1, chunk_size=1024, format="int16"
    )


class TestTranscriptionConfigs:
    """Standard transcription provider configurations."""

    # Default AWS configuration
    AWS_DEFAULT = {
        "region": "us-east-1",
        "language_code": "en-US",
        "profile_name": None,
    }

    # AWS configuration for different regions
    AWS_US_WEST = {
        "region": "us-west-2",
        "language_code": "en-US",
        "profile_name": None,
    }

    # AWS configuration for different languages
    AWS_SPANISH = {
        "region": "us-east-1",
        "language_code": "es-US",
        "profile_name": None,
    }

    # Configuration for testing with custom profile
    AWS_WITH_PROFILE = {
        "region": "us-east-1",
        "language_code": "en-US",
        "profile_name": "test-profile",
    }


class TestCaptureConfigs:
    """Standard audio capture provider configurations."""

    # Default PyAudio configuration
    PYAUDIO_DEFAULT = {"device_index": 0}

    # PyAudio configuration with specific device
    PYAUDIO_SPECIFIC_DEVICE = {"device_index": 1}

    # File capture configuration
    FILE_CAPTURE = {"file_path": "/tmp/test_audio.wav"}

    # File capture with custom settings
    FILE_CAPTURE_CUSTOM = {
        "file_path": "/tmp/custom_test.wav",
        "loop": True,
        "speed_multiplier": 1.5,
    }


class TestEnvironmentConfigs:
    """Environment variable configurations for testing."""

    # Default test environment
    DEFAULT_ENV = {
        "LOG_LEVEL": "WARNING",
        "TRANSCRIPTION_PROVIDER": "aws",
        "CAPTURE_PROVIDER": "pyaudio",
        "AWS_REGION": "us-east-1",
        "AWS_LANGUAGE_CODE": "en-US",
    }

    # Debug environment
    DEBUG_ENV = {
        "LOG_LEVEL": "DEBUG",
        "TRANSCRIPTION_PROVIDER": "aws",
        "CAPTURE_PROVIDER": "file",
        "AWS_REGION": "us-east-1",
        "AWS_LANGUAGE_CODE": "en-US",
    }

    # File-based testing environment
    FILE_TEST_ENV = {
        "LOG_LEVEL": "INFO",
        "TRANSCRIPTION_PROVIDER": "aws",
        "CAPTURE_PROVIDER": "file",
        "AWS_REGION": "us-west-2",
        "AWS_LANGUAGE_CODE": "en-US",
        "AUDIO_SAMPLE_RATE": "22050",
    }

    # Performance testing environment
    PERFORMANCE_ENV = {
        "LOG_LEVEL": "ERROR",  # Minimal logging for performance
        "TRANSCRIPTION_PROVIDER": "aws",
        "CAPTURE_PROVIDER": "pyaudio",
        "AWS_REGION": "us-east-1",
        "AWS_LANGUAGE_CODE": "en-US",
        "AUDIO_SAMPLE_RATE": "16000",
        "AUDIO_CHUNK_SIZE": "2048",
    }


class TestSessionConfigs:
    """Session manager configurations for testing."""

    # Basic session configuration
    BASIC_SESSION = {"region": "us-east-1", "language_code": "en-US"}

    # Session with custom timeout
    TIMEOUT_SESSION = {"region": "us-east-1", "language_code": "en-US", "timeout": 30.0}

    # Session for long-running tests
    LONG_RUNNING_SESSION = {
        "region": "us-east-1",
        "language_code": "en-US",
        "timeout": 300.0,  # 5 minutes
        "auto_restart": True,
    }


class TestDeviceConfigs:
    """Mock device configurations for testing."""

    # Standard mock devices
    MOCK_DEVICES = {
        0: "Built-in Microphone",
        1: "USB Headset",
        2: "Bluetooth Headphones",
        3: "External Audio Interface",
    }

    # Single device setup
    SINGLE_DEVICE = {0: "Default Audio Device"}

    # No devices (error scenario)
    NO_DEVICES = {}

    # Devices with problematic names
    PROBLEMATIC_DEVICES = {
        0: "Device with (special) characters",
        1: "Device with very long name that exceeds normal limits",
        2: "Device with unicode: 🎤 microphone",
        3: "",  # Empty name
    }


class TestTimeouts:
    """Standard timeout values for different test scenarios."""

    # Basic operation timeouts
    FAST_OPERATION = 0.5  # 500ms for fast operations
    NORMAL_OPERATION = 2.0  # 2s for normal operations
    SLOW_OPERATION = 5.0  # 5s for slow operations

    # Provider-specific timeouts
    AWS_CONNECTION = 10.0  # 10s for AWS connection
    PYAUDIO_STARTUP = 3.0  # 3s for PyAudio initialization
    FILE_PROCESSING = 1.0  # 1s for file operations

    # Session management timeouts
    SESSION_START = 5.0  # 5s for session start
    SESSION_STOP = 3.0  # 3s for session stop
    SESSION_CLEANUP = 2.0  # 2s for cleanup

    # Performance test timeouts
    PERFORMANCE_STARTUP = 1.0  # 1s max startup time
    PERFORMANCE_SHUTDOWN = 3.0  # 3s max shutdown time
    PERFORMANCE_RESPONSE = 0.1  # 100ms max response time

    # Integration test timeouts
    INTEGRATION_TIMEOUT = 15.0  # 15s for integration tests
    END_TO_END_TIMEOUT = 30.0  # 30s for end-to-end tests


class TestDataSizes:
    """Standard data sizes for testing."""

    # Audio chunk sizes
    SMALL_CHUNK = 512
    NORMAL_CHUNK = 1024
    LARGE_CHUNK = 2048
    HUGE_CHUNK = 8192

    # Test durations (in seconds)
    SHORT_AUDIO = 1.0
    MEDIUM_AUDIO = 5.0
    LONG_AUDIO = 30.0

    # Memory limits (in bytes)
    MEMORY_LIMIT_LOW = 50 * 1024 * 1024  # 50MB
    MEMORY_LIMIT_NORMAL = 150 * 1024 * 1024  # 150MB
    MEMORY_LIMIT_HIGH = 500 * 1024 * 1024  # 500MB


def get_test_config(config_name: str) -> dict[str, Any]:
    """Get test configuration by name."""
    configs = {
        "default_audio": TestAudioConfigs.DEFAULT.__dict__,
        "aws_transcription": TestTranscriptionConfigs.AWS_DEFAULT,
        "pyaudio_capture": TestCaptureConfigs.PYAUDIO_DEFAULT,
        "default_env": TestEnvironmentConfigs.DEFAULT_ENV,
        "basic_session": TestSessionConfigs.BASIC_SESSION,
        "mock_devices": TestDeviceConfigs.MOCK_DEVICES,
    }

    return configs.get(config_name, {})


def get_timeout(timeout_name: str) -> float:
    """Get timeout value by name."""
    return getattr(TestTimeouts, timeout_name.upper(), 5.0)
