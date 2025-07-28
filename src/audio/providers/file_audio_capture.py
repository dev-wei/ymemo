"""Audio capture provider that plays back from a WAV file for testing."""

import asyncio
import wave
import logging
from typing import AsyncGenerator, Optional, Dict
from ...core.interfaces import AudioCaptureProvider, AudioConfig

logger = logging.getLogger(__name__)


class FileAudioCaptureProvider(AudioCaptureProvider):
    """Audio capture provider that reads from a WAV file."""
    
    def __init__(self, file_path: str, chunk_size: int = 1024):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.is_running = False
        self.wav_file: Optional[wave.Wave_read] = None
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.chunk_duration = 0.0
        
    async def start_capture(self, audio_config: AudioConfig, device_id: Optional[int] = None) -> None:
        """Start capturing audio from WAV file."""
        self.is_running = True
        
        try:
            logger.info(f"🎵 Starting file audio capture from: {self.file_path}")
            
            # Open WAV file
            self.wav_file = wave.open(self.file_path, 'rb')
            
            # Log file properties
            channels = self.wav_file.getnchannels()
            sample_width = self.wav_file.getsampwidth()
            sample_rate = self.wav_file.getframerate()
            
            logger.info(f"📁 WAV file properties: {channels} channels, "
                       f"{sample_width} bytes/sample, {sample_rate} Hz")
            
            # Calculate timing for realistic playback
            bytes_per_second = sample_rate * channels * sample_width
            self.chunk_duration = (self.chunk_size * sample_width) / bytes_per_second
            
            logger.info(f"🎯 Chunk size: {self.chunk_size} frames, "
                       f"duration: {self.chunk_duration:.3f}s")
                
        except Exception as e:
            logger.error(f"❌ Error in file audio capture: {e}")
            raise
    
    async def get_audio_stream(self) -> AsyncGenerator[bytes, None]:
        """Get audio data stream from WAV file."""
        if not self.wav_file or not self.is_running:
            logger.error("❌ Audio capture not started")
            return
        
        try:
            # Read and yield audio chunks
            while self.is_running:
                # Read audio chunk
                frames = self.wav_file.readframes(self.chunk_size)
                
                if not frames:
                    logger.info("📄 Reached end of audio file")
                    break
                
                # Yield the audio data
                yield frames
                
                # Sleep to simulate real-time playback
                await asyncio.sleep(self.chunk_duration)
                
        except Exception as e:
            logger.error(f"❌ Error in file audio stream: {e}")
            raise
    
    async def stop_capture(self):
        """Stop audio capture."""
        self.is_running = False
        
        if self.wav_file:
            try:
                self.wav_file.close()
                logger.info("📁 WAV file closed")
            except Exception as e:
                logger.error(f"Error closing WAV file: {e}")
            finally:
                self.wav_file = None
        
        logger.info("🛑 File audio capture stopped")
    
    def list_audio_devices(self) -> Dict[int, str]:
        """List available audio input devices (mock for file)."""
        return {
            0: f"File Audio ({self.file_path})"
        }