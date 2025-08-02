"""Audio file writer utility for saving split audio channels during debugging."""

import contextlib
import logging
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AudioFileWriter:
    """
    Utility class for writing audio data to WAV files for debugging purposes.

    This class handles writing PCM audio data to proper WAV files with correct
    headers, sample rates, and formats. It's designed for debugging channel
    splitting in dual connection modes.
    """

    def __init__(
        self,
        file_path: str,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,  # 2 bytes = 16-bit
        max_duration: int = 30,  # seconds
    ):
        """
        Initialize audio file writer.

        Args:
            file_path: Path where the WAV file will be saved
            sample_rate: Audio sample rate in Hz (default: 16000)
            channels: Number of audio channels (default: 1 for mono)
            sample_width: Number of bytes per sample (default: 2 for int16)
            max_duration: Maximum recording duration in seconds
        """
        self.file_path = Path(file_path)
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.max_duration = max_duration

        # Create directory if it doesn't exist
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # State tracking
        self.is_recording = False
        self.start_time = 0.0
        self.bytes_written = 0
        self.total_samples = 0
        self._wave_file = None
        self._lock = threading.Lock()

        # Calculate maximum bytes based on duration
        self.max_bytes = sample_rate * channels * sample_width * max_duration

        logger.info(f"🎵 AudioFileWriter: Initialized for {file_path}")
        logger.info(f"   📊 Format: {sample_rate}Hz, {channels}ch, {sample_width*8}-bit")
        logger.info(f"   ⏱️  Max duration: {max_duration}s ({self.max_bytes:,} bytes)")

    def start_recording(self) -> bool:
        """
        Start recording audio to the file.

        Returns:
            bool: True if recording started successfully, False otherwise
        """
        with self._lock:
            if self.is_recording:
                logger.warning(
                    f"⚠️ AudioFileWriter: Already recording to {self.file_path}"
                )
                return False

            try:
                # Open WAV file for writing
                self._wave_file = wave.open(str(self.file_path), "wb")
                self._wave_file.setnchannels(self.channels)
                self._wave_file.setsampwidth(self.sample_width)
                self._wave_file.setframerate(self.sample_rate)

                self.is_recording = True
                self.start_time = time.time()
                self.bytes_written = 0
                self.total_samples = 0

                logger.info(f"🎵 AudioFileWriter: Started recording to {self.file_path}")
                return True

            except Exception as e:
                logger.error(f"❌ AudioFileWriter: Failed to start recording: {e}")
                if self._wave_file:
                    with contextlib.suppress(Exception):
                        self._wave_file.close()
                    self._wave_file = None
                return False

    def write_audio_data(self, audio_data: bytes) -> bool:
        """
        Write audio data to the file.

        Args:
            audio_data: Raw PCM audio data bytes

        Returns:
            bool: True if data was written successfully, False otherwise
        """
        if not self.is_recording or not self._wave_file:
            return False

        with self._lock:
            try:
                # Check if we've exceeded maximum duration/size
                if self.bytes_written + len(audio_data) > self.max_bytes:
                    remaining_bytes = self.max_bytes - self.bytes_written
                    if remaining_bytes > 0:
                        # Write partial data to reach exactly the limit
                        audio_data = audio_data[:remaining_bytes]
                        self._wave_file.writeframes(audio_data)
                        self.bytes_written += len(audio_data)
                        self.total_samples += len(audio_data) // (
                            self.channels * self.sample_width
                        )

                    logger.info(
                        "🎵 AudioFileWriter: Maximum duration reached, stopping recording"
                    )
                    self._stop_recording_internal()
                    return False

                # Write the audio data
                self._wave_file.writeframes(audio_data)
                self.bytes_written += len(audio_data)
                self.total_samples += len(audio_data) // (
                    self.channels * self.sample_width
                )

                # Log progress periodically
                if (
                    self.bytes_written % (self.sample_rate * self.sample_width * 5) == 0
                ):  # Every ~5 seconds
                    elapsed_time = time.time() - self.start_time
                    logger.debug(
                        f"🎵 AudioFileWriter: {elapsed_time:.1f}s recorded, {self.bytes_written:,} bytes written"
                    )

                return True

            except Exception as e:
                logger.error(f"❌ AudioFileWriter: Failed to write audio data: {e}")
                return False

    def stop_recording(self) -> dict[str, Any]:
        """
        Stop recording and close the file.

        Returns:
            dict: Recording statistics including file path, duration, bytes written
        """
        with self._lock:
            return self._stop_recording_internal()

    def _stop_recording_internal(self) -> dict[str, Any]:
        """Internal method to stop recording (assumes lock is held)."""
        if not self.is_recording:
            return {"error": "Not recording"}

        try:
            # Close the WAV file
            if self._wave_file:
                self._wave_file.close()
                self._wave_file = None

            end_time = time.time()
            wall_clock_duration = end_time - self.start_time
            # Calculate actual audio duration from samples (this is what matters for audio content)
            duration = self.total_samples / self.sample_rate

            # Calculate statistics
            stats = {
                "file_path": str(self.file_path),
                "duration_seconds": duration,
                "bytes_written": self.bytes_written,
                "total_samples": self.total_samples,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "sample_width": self.sample_width,
                "file_exists": self.file_path.exists(),
                "file_size_bytes": (
                    self.file_path.stat().st_size if self.file_path.exists() else 0
                ),
            }

            self.is_recording = False

            logger.info("🎵 AudioFileWriter: Recording stopped")
            logger.info(f"   📁 File: {self.file_path}")
            logger.info(f"   ⏱️  Audio Duration: {duration:.2f}s")
            logger.info(f"   ⏱️  Wall Clock Time: {wall_clock_duration:.2f}s")
            logger.info(
                f"   📊 Data: {self.bytes_written:,} bytes, {self.total_samples:,} samples"
            )
            logger.info(f"   💾 File size: {stats['file_size_bytes']:,} bytes")

            # Validate file integrity
            if self.file_path.exists():
                try:
                    with wave.open(str(self.file_path), "rb") as test_wave:
                        test_frames = test_wave.getnframes()
                        test_rate = test_wave.getframerate()
                        test_channels = test_wave.getnchannels()
                        logger.info(
                            f"   ✅ File validation: {test_frames} frames, {test_rate}Hz, {test_channels}ch"
                        )
                except Exception as e:
                    logger.warning(f"   ⚠️ File validation failed: {e}")
            else:
                logger.error("   ❌ File was not created successfully")

            return stats

        except Exception as e:
            logger.error(f"❌ AudioFileWriter: Error stopping recording: {e}")
            self.is_recording = False
            return {"error": str(e)}

    def is_active(self) -> bool:
        """Check if currently recording."""
        return self.is_recording

    def get_statistics(self) -> dict[str, Any]:
        """Get current recording statistics."""
        with self._lock:
            elapsed_time = time.time() - self.start_time if self.is_recording else 0
            return {
                "is_recording": self.is_recording,
                "file_path": str(self.file_path),
                "elapsed_seconds": elapsed_time,
                "bytes_written": self.bytes_written,
                "total_samples": self.total_samples,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "progress_percent": (
                    (self.bytes_written / self.max_bytes * 100)
                    if self.max_bytes > 0
                    else 0
                ),
            }


class DualChannelAudioSaver:
    """
    Manager for saving audio from both channels during dual connection debugging.

    This class manages separate AudioFileWriter instances for left and right
    channels, with synchronized start/stop and automatic file naming.
    """

    def __init__(
        self,
        save_path: str = "./debug_audio/",
        sample_rate: int = 16000,
        duration: int = 30,
    ):
        """
        Initialize dual channel audio saver.

        Args:
            save_path: Directory where audio files will be saved
            sample_rate: Audio sample rate in Hz
            duration: Maximum recording duration in seconds
        """
        self.save_path = Path(save_path)
        self.sample_rate = sample_rate
        self.duration = duration

        # Create directory if needed
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Generate timestamp-based filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        left_file = self.save_path / f"left_channel_{timestamp}.wav"
        right_file = self.save_path / f"right_channel_{timestamp}.wav"

        # Create writers for each channel
        self.left_writer = AudioFileWriter(
            str(left_file), sample_rate, channels=1, max_duration=duration
        )
        self.right_writer = AudioFileWriter(
            str(right_file), sample_rate, channels=1, max_duration=duration
        )

        self.is_active = False

        logger.info("🎵 DualChannelAudioSaver: Initialized")
        logger.info(f"   📁 Left: {left_file}")
        logger.info(f"   📁 Right: {right_file}")

    def start_recording(self) -> bool:
        """Start recording both channels."""
        if self.is_active:
            return True

        left_started = self.left_writer.start_recording()
        right_started = self.right_writer.start_recording()

        if left_started and right_started:
            self.is_active = True
            logger.info("🎵 DualChannelAudioSaver: Both channels recording started")
            return True
        logger.error("❌ DualChannelAudioSaver: Failed to start recording")
        # Clean up any successful starts
        if left_started:
            self.left_writer.stop_recording()
        if right_started:
            self.right_writer.stop_recording()
        return False

    def write_left_audio(self, audio_data: bytes) -> bool:
        """Write audio data to left channel file."""
        if not self.is_active:
            return False
        return self.left_writer.write_audio_data(audio_data)

    def write_right_audio(self, audio_data: bytes) -> bool:
        """Write audio data to right channel file."""
        if not self.is_active:
            return False
        return self.right_writer.write_audio_data(audio_data)

    def stop_recording(self) -> dict[str, Any]:
        """Stop recording both channels and return statistics."""
        if not self.is_active:
            return {"error": "Not recording"}

        left_stats = self.left_writer.stop_recording()
        right_stats = self.right_writer.stop_recording()

        self.is_active = False

        logger.info("🎵 DualChannelAudioSaver: Recording stopped for both channels")

        return {
            "left_channel": left_stats,
            "right_channel": right_stats,
            "total_files": 2,
        }

    def get_file_paths(self) -> dict[str, str]:
        """Get the file paths for both channels."""
        return {
            "left": str(self.left_writer.file_path),
            "right": str(self.right_writer.file_path),
        }
