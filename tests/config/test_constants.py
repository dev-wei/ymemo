"""Test constants and sample data for consistent testing.

This module provides constant values, sample data, and test fixtures
that are used across multiple test files.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
TESTS_DIR = PROJECT_ROOT / "tests"
SRC_DIR = PROJECT_ROOT / "src"
TEMP_DIR = Path("/tmp") / "ymemo_tests"

# Ensure temp directory exists
TEMP_DIR.mkdir(exist_ok=True)


class TestConstants:
    """General test constants."""

    # Test identifiers
    DEFAULT_MEETING_ID = "test_meeting_123"
    DEFAULT_SESSION_ID = "test_session_456"
    DEFAULT_USER_ID = "test_user_789"

    # Audio parameters
    DEFAULT_SAMPLE_RATE = 16000
    DEFAULT_CHANNELS = 1
    DEFAULT_CHUNK_SIZE = 1024
    DEFAULT_FORMAT = "int16"

    # Timing constants
    SHORT_DELAY = 0.1
    MEDIUM_DELAY = 0.5
    LONG_DELAY = 1.0

    # Test data sizes
    SMALL_BUFFER_SIZE = 512
    NORMAL_BUFFER_SIZE = 1024
    LARGE_BUFFER_SIZE = 2048

    # File extensions
    AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg"]
    CONFIG_EXTENSIONS = [".json", ".yaml", ".yml"]

    # Error messages for testing
    GENERIC_ERROR_MSG = "Test error occurred"
    CONNECTION_ERROR_MSG = "Connection failed"
    TIMEOUT_ERROR_MSG = "Operation timed out"
    PERMISSION_ERROR_MSG = "Permission denied"


class SampleAudioData:
    """Sample audio data for testing."""

    # Silence samples (16-bit mono)
    SILENCE_1SEC = b"\x00\x00" * (16000 // 2)  # 1 second of silence
    SILENCE_100MS = b"\x00\x00" * (1600 // 2)  # 100ms of silence

    # Noise samples (16-bit mono)
    WHITE_NOISE_100MS = bytes([i % 256 for i in range(3200)])  # Simple noise pattern

    # Audio chunk samples of various sizes
    CHUNK_512 = b"\x00\x01" * 256  # 512 bytes
    CHUNK_1024 = b"\x00\x01" * 512  # 1024 bytes
    CHUNK_2048 = b"\x00\x01" * 1024  # 2048 bytes

    @staticmethod
    def generate_sine_wave(
        frequency: int = 440, duration: float = 1.0, sample_rate: int = 16000
    ) -> bytes:
        """Generate sine wave audio data."""
        import math

        samples = int(sample_rate * duration)
        audio_data = []

        for i in range(samples):
            sample = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
            # Convert to 16-bit little-endian
            audio_data.extend([(sample & 0xFF), ((sample >> 8) & 0xFF)])

        return bytes(audio_data)

    @staticmethod
    def generate_test_chunks(
        chunk_size: int = 1024, num_chunks: int = 10
    ) -> list[bytes]:
        """Generate list of test audio chunks."""
        chunks = []
        for i in range(num_chunks):
            # Create unique pattern for each chunk
            chunk = bytes([(i * j) % 256 for j in range(chunk_size)])
            chunks.append(chunk)
        return chunks


class SampleTranscriptionResults:
    """Sample transcription results for testing."""

    # Basic transcription samples
    HELLO_WORLD = {
        "text": "Hello world",
        "speaker_id": "Speaker1",
        "confidence": 0.95,
        "start_time": 0.0,
        "end_time": 1.0,
        "is_partial": False,
        "utterance_id": "utterance_001",
        "sequence_number": 1,
        "result_id": "result_001",
    }

    PARTIAL_HELLO = {
        "text": "Hello",
        "speaker_id": "Speaker1",
        "confidence": 0.8,
        "start_time": 0.0,
        "end_time": 0.5,
        "is_partial": True,
        "utterance_id": "utterance_001",
        "sequence_number": 1,
        "result_id": "partial_001",
    }

    LONG_SENTENCE = {
        "text": "This is a longer sentence that might be used to test transcription handling of extended speech.",
        "speaker_id": "Speaker1",
        "confidence": 0.92,
        "start_time": 0.0,
        "end_time": 5.0,
        "is_partial": False,
        "utterance_id": "utterance_002",
        "sequence_number": 1,
        "result_id": "result_002",
    }

    # Multiple speakers
    CONVERSATION = [
        {
            "text": "How are you doing today?",
            "speaker_id": "Speaker1",
            "confidence": 0.93,
            "start_time": 0.0,
            "end_time": 2.0,
            "is_partial": False,
            "utterance_id": "utterance_003",
            "sequence_number": 1,
            "result_id": "result_003",
        },
        {
            "text": "I am doing great, thank you for asking.",
            "speaker_id": "Speaker2",
            "confidence": 0.91,
            "start_time": 2.5,
            "end_time": 4.5,
            "is_partial": False,
            "utterance_id": "utterance_004",
            "sequence_number": 1,
            "result_id": "result_004",
        },
    ]

    # Partial result sequence
    PARTIAL_SEQUENCE = [
        {
            "text": "The",
            "speaker_id": "Speaker1",
            "confidence": 0.7,
            "start_time": 0.0,
            "end_time": 0.2,
            "is_partial": True,
            "utterance_id": "utterance_005",
            "sequence_number": 1,
            "result_id": "partial_005_1",
        },
        {
            "text": "The weather",
            "speaker_id": "Speaker1",
            "confidence": 0.85,
            "start_time": 0.0,
            "end_time": 0.7,
            "is_partial": True,
            "utterance_id": "utterance_005",
            "sequence_number": 2,
            "result_id": "partial_005_2",
        },
        {
            "text": "The weather is nice today",
            "speaker_id": "Speaker1",
            "confidence": 0.94,
            "start_time": 0.0,
            "end_time": 2.0,
            "is_partial": False,
            "utterance_id": "utterance_005",
            "sequence_number": 3,
            "result_id": "result_005",
        },
    ]


class SampleDeviceInfo:
    """Sample device information for testing."""

    BUILTIN_MIC = {
        "index": 0,
        "name": "Built-in Microphone",
        "maxInputChannels": 1,
        "maxOutputChannels": 0,
        "defaultSampleRate": 44100.0,
        "hostApi": 0,
    }

    USB_HEADSET = {
        "index": 1,
        "name": "USB Audio Device",
        "maxInputChannels": 1,
        "maxOutputChannels": 2,
        "defaultSampleRate": 48000.0,
        "hostApi": 0,
    }

    BLUETOOTH_DEVICE = {
        "index": 2,
        "name": "Bluetooth Audio",
        "maxInputChannels": 1,
        "maxOutputChannels": 2,
        "defaultSampleRate": 16000.0,
        "hostApi": 1,
    }

    PROFESSIONAL_INTERFACE = {
        "index": 3,
        "name": "Audio Interface Pro",
        "maxInputChannels": 8,
        "maxOutputChannels": 8,
        "defaultSampleRate": 96000.0,
        "hostApi": 0,
    }


class SampleErrorScenarios:
    """Sample error scenarios for testing."""

    AWS_CONNECTION_ERROR = {
        "error_type": "ConnectionError",
        "message": "Unable to connect to AWS Transcribe",
        "details": "Network timeout after 30 seconds",
    }

    PYAUDIO_DEVICE_ERROR = {
        "error_type": "IOError",
        "message": "Audio device not available",
        "details": "Device index 5 does not exist",
    }

    PERMISSION_ERROR = {
        "error_type": "PermissionError",
        "message": "Microphone access denied",
        "details": "Application does not have microphone permissions",
    }

    MEMORY_ERROR = {
        "error_type": "MemoryError",
        "message": "Insufficient memory for audio processing",
        "details": "Unable to allocate 256MB for audio buffer",
    }

    TIMEOUT_ERROR = {
        "error_type": "TimeoutError",
        "message": "Operation timed out",
        "details": "Session start took longer than 30 seconds",
    }


class TestFilenames:
    """Standard test filenames."""

    # Audio files
    TEST_AUDIO_WAV = "test_audio.wav"
    TEST_AUDIO_LONG = "test_long_audio.wav"
    TEST_AUDIO_SILENT = "test_silent.wav"
    TEST_AUDIO_NOISE = "test_noise.wav"

    # Config files
    TEST_CONFIG_JSON = "test_config.json"
    TEST_CONFIG_YAML = "test_config.yaml"

    # Log files
    TEST_LOG_FILE = "test_run.log"
    ERROR_LOG_FILE = "test_errors.log"
    PERFORMANCE_LOG = "performance_metrics.log"

    # Temporary files
    TEMP_AUDIO = str(TEMP_DIR / "temp_audio.wav")
    TEMP_CONFIG = str(TEMP_DIR / "temp_config.json")
    TEMP_LOG = str(TEMP_DIR / "temp_test.log")


class TestMessages:
    """Standard test messages and formatting."""

    # Success messages
    TEST_PASSED = "✅ Test completed successfully"
    SETUP_COMPLETE = "🏗️ Test setup completed"
    CLEANUP_COMPLETE = "🧹 Test cleanup completed"

    # Progress messages
    STARTING_TEST = "🧪 Starting test: {test_name}"
    TEST_PROGRESS = "⏳ Test progress: {step} ({progress}%)"

    # Error messages
    TEST_FAILED = "❌ Test failed: {reason}"
    SETUP_FAILED = "💥 Test setup failed: {reason}"
    CLEANUP_FAILED = "⚠️ Test cleanup failed: {reason}"

    # Performance messages
    PERFORMANCE_BASELINE = "📊 Performance baseline: {metric} = {value}"
    PERFORMANCE_RESULT = (
        "🎯 Performance result: {metric} = {value} (baseline: {baseline})"
    )
    PERFORMANCE_THRESHOLD = "⚡ Performance threshold: {metric} must be < {threshold}"


def get_test_file_path(filename: str) -> str:
    """Get full path for test file."""
    return str(TEMP_DIR / filename)


def cleanup_test_files():
    """Clean up all temporary test files."""
    import shutil

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
        TEMP_DIR.mkdir(exist_ok=True)
