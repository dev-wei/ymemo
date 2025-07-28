"""Abstract interfaces for audio processing components."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AudioConfig:
    """Configuration for audio capture and processing."""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format: str = 'int16'


@dataclass
class TranscriptionResult:
    """Result from transcription processing."""
    text: str
    speaker_id: Optional[str] = None
    confidence: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    is_partial: bool = False
    result_id: Optional[str] = None  # Track result groups from AWS
    utterance_id: Optional[str] = None  # Group related partial results
    sequence_number: int = 0  # Order within utterance


class TranscriptionProvider(ABC):
    """Abstract base class for transcription providers."""
    
    @abstractmethod
    async def start_stream(self, audio_config: AudioConfig) -> None:
        """Start the transcription stream.
        
        Args:
            audio_config: Configuration for audio processing
        """
        pass
    
    @abstractmethod
    async def send_audio(self, audio_chunk: bytes) -> None:
        """Send audio data to the transcription service.
        
        Args:
            audio_chunk: Raw audio data bytes
        """
        pass
    
    @abstractmethod
    async def get_transcription(self) -> AsyncGenerator[TranscriptionResult, None]:
        """Get transcription results as they become available.
        
        Yields:
            TranscriptionResult objects with text and metadata
        """
        pass
    
    @abstractmethod
    async def stop_stream(self) -> None:
        """Stop the transcription stream and cleanup resources."""
        pass


class AudioCaptureProvider(ABC):
    """Abstract base class for audio capture providers."""
    
    @abstractmethod
    async def start_capture(self, audio_config: AudioConfig, device_id: Optional[int] = None) -> None:
        """Start audio capture from specified device.
        
        Args:
            audio_config: Configuration for audio capture
            device_id: Optional specific device ID to use
        """
        pass
    
    @abstractmethod
    async def get_audio_stream(self) -> AsyncGenerator[bytes, None]:
        """Get audio data stream.
        
        Yields:
            Raw audio data bytes
        """
        pass
    
    @abstractmethod
    async def stop_capture(self) -> None:
        """Stop audio capture and cleanup resources."""
        pass
    
    @abstractmethod
    def list_audio_devices(self) -> Dict[int, str]:
        """List available audio input devices.
        
        Returns:
            Dictionary mapping device ID to device name
        """
        pass


class DiarizationProvider(ABC):
    """Abstract base class for speaker diarization providers."""
    
    @abstractmethod
    async def identify_speakers(self, audio_segment: bytes, audio_config: AudioConfig) -> Dict[str, Any]:
        """Identify speakers in audio segment.
        
        Args:
            audio_segment: Raw audio data
            audio_config: Audio configuration
            
        Returns:
            Dictionary with speaker information
        """
        pass