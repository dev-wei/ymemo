"""
Abstract interfaces for audio processing components.

This module defines the core interfaces that all audio processing providers must implement.
The interfaces use abstract base classes (ABC) to ensure consistent implementation across
different providers (AWS Transcribe, Azure Speech, PyAudio, etc.).

Key Interfaces:
- TranscriptionProvider: For speech-to-text services
- AudioCaptureProvider: For audio input sources
- DiarizationProvider: For speaker identification

Data Models:
- AudioConfig: Configuration for audio processing
- TranscriptionResult: Structured transcription output

Example Implementation:
    class MyTranscriptionProvider(TranscriptionProvider):
        async def start_stream(self, audio_config: AudioConfig) -> None:
            # Initialize transcription service
            pass

        async def send_audio(self, audio_chunk: bytes) -> None:
            # Send audio to service
            pass

        async def get_transcription(self) -> AsyncGenerator[TranscriptionResult, None]:
            # Yield transcription results
            yield TranscriptionResult(text="Hello", confidence=0.95)

        async def stop_stream(self) -> None:
            # Cleanup resources
            pass
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any


@dataclass
class AudioConfig:
    """
    Configuration for audio capture and processing.

    This dataclass defines the audio parameters used throughout the system.
    All providers should use these settings for consistent audio processing.

    Attributes:
        sample_rate: Audio sample rate in Hz (default: 16000 - optimal for speech)
        channels: Number of audio channels (default: 1 - mono for speech recognition)
        chunk_size: Size of audio chunks in samples (default: 1024 - good balance of latency/throughput)
        format: Audio format specification (default: 'int16' - 16-bit signed integer)

    Example:
        # Default configuration for speech recognition
        config = AudioConfig()

        # High-quality configuration
        config = AudioConfig(
            sample_rate=48000,
            channels=2,
            chunk_size=2048,
            format='float32'
        )
    """

    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format: str = "int16"


@dataclass
class TranscriptionResult:
    """
    Result from transcription processing.

    This dataclass represents a single transcription result from a speech-to-text
    provider. It includes the transcribed text along with metadata for timing,
    confidence, speaker identification, and partial result handling.

    Attributes:
        text: The transcribed text content (required)
        speaker_id: Speaker identifier (e.g., "Speaker 1", "John", None for no diarization)
        confidence: Confidence score 0.0-1.0 (higher = more confident)
        start_time: Start time of audio segment in seconds
        end_time: End time of audio segment in seconds
        is_partial: Whether this is a partial/interim result (will be updated)
        result_id: Provider-specific result identifier for grouping
        utterance_id: Groups related partial results for the same utterance
        sequence_number: Order within an utterance for partial results

    Example:
        # Final transcription result
        result = TranscriptionResult(
            text="Hello, how are you?",
            speaker_id="Speaker 1",
            confidence=0.95,
            start_time=1.2,
            end_time=3.4,
            is_partial=False
        )

        # Partial result that will be updated
        partial = TranscriptionResult(
            text="Hello, how are",
            confidence=0.85,
            is_partial=True,
            utterance_id="utterance_1",
            sequence_number=1
        )
    """

    text: str
    speaker_id: str | None = None
    confidence: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    is_partial: bool = False
    result_id: str | None = None  # Track result groups from AWS
    utterance_id: str | None = None  # Group related partial results
    sequence_number: int = 0  # Order within utterance


class TranscriptionProvider(ABC):
    """
    Abstract base class for speech-to-text transcription providers.

    This interface defines the contract that all transcription providers must implement.
    It supports streaming transcription with real-time results, partial results,
    and proper resource management.

    Providers implementing this interface include:
    - AWSTranscribeProvider: AWS Transcribe streaming service
    - AzureSpeechProvider: Azure Speech Service
    - (Future) OpenAIWhisperProvider, GoogleSpeechProvider, etc.

    Usage Pattern:
        1. start_stream() - Initialize transcription service
        2. send_audio() - Send audio chunks continuously
        3. get_transcription() - Receive results asynchronously
        4. stop_stream() - Clean up resources

    Example:
        provider = MyTranscriptionProvider()

        # Initialize
        await provider.start_stream(AudioConfig())

        # Stream audio and get results
        async def process():
            # Send audio in background
            asyncio.create_task(send_audio_continuously(provider))

            # Receive transcriptions
            async for result in provider.get_transcription():
                print(f"Transcribed: {result.text}")

        # Cleanup
        await provider.stop_stream()
    """

    @abstractmethod
    async def start_stream(self, audio_config: AudioConfig) -> None:
        """
        Start the transcription stream and initialize the service.

        This method should establish connection to the transcription service,
        configure audio parameters, and prepare to receive audio data.

        Args:
            audio_config: Audio configuration specifying sample rate, format, etc.

        Raises:
            ConnectionError: If unable to connect to transcription service
            ValueError: If audio configuration is invalid
            RuntimeError: If service initialization fails

        Example:
            config = AudioConfig(sample_rate=16000, channels=1)
            await provider.start_stream(config)
        """

    @abstractmethod
    async def send_audio(self, audio_chunk: bytes) -> None:
        """
        Send audio data to the transcription service.

        This method should be called continuously with audio chunks during recording.
        The provider should handle buffering and streaming to the service.

        Args:
            audio_chunk: Raw audio data bytes matching the AudioConfig format

        Raises:
            ConnectionError: If connection to service is lost
            ValueError: If audio chunk format is invalid
            RuntimeError: If stream is not started or already stopped

        Note:
            - Audio chunks should match the format specified in start_stream()
            - This method should be non-blocking for real-time performance
            - Providers should handle internal buffering as needed

        Example:
            # In a loop during recording
            audio_data = await capture_audio_chunk()
            await provider.send_audio(audio_data)
        """

    @abstractmethod
    async def get_transcription(self) -> AsyncGenerator[TranscriptionResult, None]:
        """
        Get transcription results as they become available.

        This async generator yields transcription results in real-time as the
        service processes audio. Results may include partial (interim) results
        that get updated, followed by final results.

        Yields:
            TranscriptionResult: Objects containing transcribed text and metadata

        Raises:
            ConnectionError: If connection to service is lost
            RuntimeError: If stream is not started

        Note:
            - Partial results (is_partial=True) may be updated with better text
            - Final results (is_partial=False) are the definitive transcription
            - Providers should handle result ordering and deduplication
            - Generator continues until stop_stream() is called

        Example:
            async for result in provider.get_transcription():
                if result.is_partial:
                    print(f"Partial: {result.text}")
                else:
                    print(f"Final: {result.text}")
        """

    @abstractmethod
    async def stop_stream(self) -> None:
        """
        Stop the transcription stream and cleanup resources.

        This method should gracefully close the connection to the transcription
        service, flush any remaining results, and release all resources.

        Raises:
            RuntimeError: If cleanup fails or resources cannot be released

        Note:
            - Should be called even if errors occurred during streaming
            - Should be idempotent (safe to call multiple times)
            - Should wait for any final results before closing
            - Should release network connections, file handles, etc.

        Example:
            try:
                # Transcription work
                await provider.start_stream(config)
                # ... streaming ...
            finally:
                await provider.stop_stream()  # Always cleanup
        """

    @abstractmethod
    def get_required_channels(self) -> int:
        """
        Get the number of audio channels required by this transcription provider.

        This method indicates how many audio channels the provider can effectively
        utilize for transcription. It helps the audio processing pipeline determine
        optimal channel conversion strategies.

        Returns:
            int: Number of channels the provider supports/requires
                - 1: Mono transcription (most providers)
                - 2: Dual-channel with speaker separation (AWS Transcribe, Azure)
                - >2: Multi-channel support (rare, advanced providers)

        Note:
            - This is used by the AudioChannelProcessor to optimize channel conversion
            - For providers supporting channel identification (speaker separation),
              returning 2 enables intelligent channel grouping for better speaker isolation
            - Most speech-to-text services work best with 1 or 2 channels

        Example:
            # AWS Transcribe with channel identification
            def get_required_channels(self) -> int:
                return 2  # Supports dual-channel with speaker separation

            # OpenAI Whisper (mono only)
            def get_required_channels(self) -> int:
                return 1  # Mono transcription only
        """


class AudioCaptureProvider(ABC):
    """
    Abstract base class for audio capture providers.

    This interface defines the contract for capturing audio from various sources
    such as microphones, files, network streams, etc. It provides a unified
    interface for real-time audio streaming.

    Providers implementing this interface include:
    - PyAudioCaptureProvider: Microphone capture via PyAudio
    - FileAudioCaptureProvider: File-based audio source for testing
    - (Future) NetworkCaptureProvider, USBCaptureProvider, etc.

    Usage Pattern:
        1. list_audio_devices() - Discover available devices
        2. start_capture() - Initialize audio capture
        3. get_audio_stream() - Receive audio data continuously
        4. stop_capture() - Clean up resources

    Example:
        provider = MyAudioCaptureProvider()

        # List available devices
        devices = provider.list_audio_devices()

        # Start capture
        config = AudioConfig(sample_rate=16000)
        await provider.start_capture(config, device_id=1)

        # Stream audio
        async for audio_chunk in provider.get_audio_stream():
            process_audio(audio_chunk)

        # Cleanup
        await provider.stop_capture()
    """

    @abstractmethod
    async def start_capture(
        self, audio_config: AudioConfig, device_id: int | None = None
    ) -> None:
        """
        Start audio capture from specified device.

        This method initializes the audio capture system and prepares to stream
        audio data according to the specified configuration.

        Args:
            audio_config: Audio configuration (sample rate, channels, format, etc.)
            device_id: Specific device ID to use (None = use system default)

        Raises:
            DeviceError: If the specified device is not available or cannot be opened
            ValueError: If audio configuration is invalid or unsupported
            RuntimeError: If capture system initialization fails

        Example:
            # Use default device with standard settings
            config = AudioConfig(sample_rate=16000, channels=1)
            await provider.start_capture(config)

            # Use specific device
            await provider.start_capture(config, device_id=2)
        """

    @abstractmethod
    async def get_audio_stream(self) -> AsyncGenerator[bytes, None]:
        """
        Get continuous stream of audio data.

        This async generator yields audio chunks continuously during capture.
        The chunks match the format and size specified in AudioConfig.

        Yields:
            bytes: Raw audio data chunks in the configured format

        Raises:
            RuntimeError: If capture is not started or has been stopped
            DeviceError: If audio device disconnected or encountered error

        Note:
            - Audio chunks are in the format specified during start_capture()
            - Chunk size is determined by AudioConfig.chunk_size
            - Generator continues until stop_capture() is called
            - Should provide real-time streaming with minimal latency

        Example:
            async for audio_chunk in provider.get_audio_stream():
                # Process audio in real-time
                transcription_service.send_audio(audio_chunk)
        """

    @abstractmethod
    async def stop_capture(self) -> None:
        """
        Stop audio capture and cleanup resources.

        This method gracefully stops audio capture, flushes any remaining
        audio data, and releases all system resources.

        Raises:
            RuntimeError: If cleanup fails or resources cannot be released

        Note:
            - Should be called even if errors occurred during capture
            - Should be idempotent (safe to call multiple times)
            - Should release audio devices, file handles, network connections
            - Should wait for any remaining audio data to be processed

        Example:
            try:
                await provider.start_capture(config)
                # ... audio streaming ...
            finally:
                await provider.stop_capture()  # Always cleanup
        """

    @abstractmethod
    def list_audio_devices(self) -> dict[int, str]:
        """
        List available audio input devices.

        This method discovers and returns all available audio input devices
        that can be used for capture. Device IDs can be used with start_capture().

        Returns:
            Dictionary mapping device ID to human-readable device name

        Raises:
            RuntimeError: If device enumeration fails

        Note:
            - Device IDs should be stable during application lifetime
            - Device names should be user-friendly for display in UI
            - Should include only input devices capable of capture
            - May exclude devices that are in use by other applications

        Example:
            devices = provider.list_audio_devices()
            # {0: "Built-in Microphone", 1: "USB Headset", 2: "Blue Yeti"}

            # Let user select device
            for device_id, name in devices.items():
                print(f"{device_id}: {name}")
        """


class DiarizationProvider(ABC):
    """
    Abstract base class for speaker diarization providers.

    This interface defines the contract for identifying and separating different
    speakers in audio. Speaker diarization answers "who spoke when" by analyzing
    audio characteristics and grouping speech segments by speaker.

    Note: Many transcription providers (like Azure Speech) include built-in
    diarization, so this interface may be used less frequently as a standalone
    component.

    Providers implementing this interface include:
    - (Future) PyannoteProvider: Pyannote.audio for speaker diarization
    - (Future) ResembleAIProvider: Resemble.ai diarization service
    - Built-in diarization in transcription providers (preferred)

    Usage Pattern:
        1. identify_speakers() - Analyze audio segment for speakers
        2. Process results to map speakers to speech segments

    Example:
        provider = MyDiarizationProvider()

        # Analyze audio segment
        audio_data = get_audio_segment()
        config = AudioConfig()

        speaker_info = await provider.identify_speakers(audio_data, config)

        # Process speaker mapping
        for segment in speaker_info['segments']:
            speaker_id = segment['speaker']
            start_time = segment['start']
            print(f"Speaker {speaker_id} spoke at {start_time}s")
    """

    @abstractmethod
    async def identify_speakers(
        self, audio_segment: bytes, audio_config: AudioConfig
    ) -> dict[str, Any]:
        """
        Identify and separate speakers in an audio segment.

        This method analyzes audio data to identify distinct speakers and
        provides timing information for when each speaker was active.

        Args:
            audio_segment: Raw audio data to analyze
            audio_config: Audio format configuration for the segment

        Returns:
            Dictionary containing speaker analysis results with structure:
            {
                'speakers': List of unique speaker identifiers,
                'segments': List of dicts with speaker, start_time, end_time,
                'confidence': Overall confidence in speaker identification
            }

        Raises:
            ValueError: If audio_segment format doesn't match audio_config
            RuntimeError: If speaker identification fails

        Note:
            - Audio segment should be long enough for meaningful analysis (>1-2 seconds)
            - Speaker IDs should be consistent within the same audio session
            - Confidence scores help indicate reliability of speaker assignments
            - Some providers may require minimum segment length or audio quality

        Example:
            result = await provider.identify_speakers(audio_data, config)

            # Process results
            print(f"Found {len(result['speakers'])} speakers")

            for segment in result['segments']:
                speaker = segment['speaker']
                start = segment['start_time']
                end = segment['end_time']
                print(f"Speaker {speaker}: {start:.2f}s - {end:.2f}s")
        """
