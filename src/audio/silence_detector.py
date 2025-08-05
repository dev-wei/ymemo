"""Audio silence detection system for automatic recording termination.

Provides configurable silence detection with support for multi-channel audio
and different sample formats, designed for real-time processing with minimal latency.
"""

import logging
import struct
import threading
import time
from typing import Any, Callable, Optional, Union

logger = logging.getLogger(__name__)


class SilenceDetector:
    """Real-time silence detection for automatic recording termination.

    Monitors audio levels across all channels and tracks silence duration.
    When silence exceeds configured threshold, triggers auto-stop callback.
    Thread-safe implementation suitable for concurrent audio processing.
    """

    def __init__(
        self,
        silence_timeout_seconds: int = 300,
        silence_threshold: Optional[float] = None,
        auto_stop_callback: Optional[Callable[[], None]] = None,
    ):
        """Initialize silence detector.

        Args:
            silence_timeout_seconds: Duration of silence (seconds) before triggering auto-stop.
                                   Set to 0 to disable silence detection.
            silence_threshold: Amplitude threshold below which audio is considered silence.
                             If None, uses format-appropriate defaults.
            auto_stop_callback: Function to call when silence timeout exceeded.
                              Should be async-compatible.
        """
        self.silence_timeout_seconds = silence_timeout_seconds
        self.silence_threshold = silence_threshold  # Will be set per audio format
        self.auto_stop_callback = auto_stop_callback

        # Thread-safe state tracking
        self._lock = threading.Lock()
        self._silence_start_time = None
        self._last_activity_time = time.time()
        self._total_chunks_analyzed = 0
        self._silence_chunks_detected = 0
        self._audio_format = None
        self._channels = None

        # Format-specific amplitude processors
        self._format_processors = {
            "int16": self._analyze_int16_amplitude,
            "int24": self._analyze_int24_amplitude,
            "int32": self._analyze_int32_amplitude,
            "float32": self._analyze_float32_amplitude,
        }

        # Default silence thresholds for different formats
        self._format_thresholds = {
            "int16": 100,  # ~0.3% of 16-bit range
            "int24": 8388,  # ~0.1% of 24-bit range
            "int32": 214748,  # ~0.01% of 32-bit range
            "float32": 0.001,  # 0.1% of float range
        }

        # Enabled state
        self._enabled = silence_timeout_seconds > 0

        if self._enabled:
            logger.info(
                f"🔇 SilenceDetector: Initialized with {silence_timeout_seconds}s timeout"
            )
        else:
            logger.info("🔇 SilenceDetector: Disabled (timeout = 0)")

    def configure_audio_format(self, audio_format: str, channels: int) -> None:
        """Configure detector for specific audio format and channel count.

        Args:
            audio_format: Audio format string ('int16', 'int24', 'int32', 'float32')
            channels: Number of audio channels
        """
        with self._lock:
            self._audio_format = audio_format
            self._channels = channels

            # Set format-appropriate silence threshold if not explicitly provided
            if self.silence_threshold is None:
                self.silence_threshold = self._format_thresholds.get(
                    audio_format, self._format_thresholds["int16"]
                )

            logger.info(
                f"🔇 SilenceDetector: Configured for {audio_format} format, "
                f"{channels} channels, threshold={self.silence_threshold}"
            )

    def analyze_audio_chunk(self, audio_data: bytes) -> bool:
        """Analyze audio chunk for silence and update state.

        Args:
            audio_data: Raw audio data bytes

        Returns:
            True if silence timeout has been exceeded (auto-stop should trigger)
            False otherwise
        """
        if not self._enabled:
            return False

        if not audio_data or not self._audio_format:
            return False

        try:
            # Analyze amplitude across all channels
            max_amplitude = self._get_max_amplitude(audio_data)
            current_time = time.time()

            with self._lock:
                self._total_chunks_analyzed += 1

                # Check if this chunk contains significant audio activity
                is_silence = max_amplitude < self.silence_threshold

                if is_silence:
                    self._silence_chunks_detected += 1

                    # Start tracking silence period if not already started
                    if self._silence_start_time is None:
                        self._silence_start_time = current_time
                        logger.info(
                            f"🔇 SilenceDetector: Silence period started (threshold={self.silence_threshold}, "
                            f"max_amplitude={max_amplitude:.1f})"
                        )

                    # Check if silence duration exceeds timeout
                    silence_duration = current_time - self._silence_start_time

                    # Log silence progress periodically
                    if (
                        self._total_chunks_analyzed % 100 == 0
                    ):  # Every ~10 seconds at typical chunk rates
                        logger.info(
                            f"🔇 SilenceDetector: Silence duration {silence_duration:.1f}s / "
                            f"{self.silence_timeout_seconds}s"
                        )

                    if silence_duration >= self.silence_timeout_seconds:
                        logger.warning(
                            f"🔇 SilenceDetector: Silence timeout exceeded! "
                            f"Duration: {silence_duration:.1f}s, Threshold: {self.silence_timeout_seconds}s"
                        )
                        logger.info(
                            f"🔇 SilenceDetector: Analysis summary - Total chunks: {self._total_chunks_analyzed}, "
                            f"Silent chunks: {self._silence_chunks_detected} "
                            f"({100 * self._silence_chunks_detected / self._total_chunks_analyzed:.1f}%)"
                        )

                        # Trigger auto-stop callback
                        if self.auto_stop_callback:
                            try:
                                self.auto_stop_callback()
                            except Exception as e:
                                logger.error(
                                    f"❌ SilenceDetector: Error in auto-stop callback: {e}"
                                )

                        return True
                else:
                    # Audio activity detected - reset silence tracking
                    if self._silence_start_time is not None:
                        silence_duration = current_time - self._silence_start_time
                        logger.info(
                            f"🔇 SilenceDetector: Audio activity detected after {silence_duration:.1f}s silence "
                            f"(max_amplitude={max_amplitude:.1f})"
                        )

                    self._silence_start_time = None
                    self._last_activity_time = current_time

                return False

        except Exception as e:
            logger.error(f"❌ SilenceDetector: Error analyzing audio chunk: {e}")
            return False

    def _get_max_amplitude(self, audio_data: bytes) -> float:
        """Get maximum amplitude across all channels in audio data.

        Args:
            audio_data: Raw audio data bytes

        Returns:
            Maximum amplitude value found in any channel
        """
        if self._audio_format not in self._format_processors:
            logger.error(
                f"❌ SilenceDetector: Unsupported audio format: {self._audio_format}"
            )
            return 0.0

        processor = self._format_processors[self._audio_format]
        return processor(audio_data)

    def _analyze_int16_amplitude(self, audio_data: bytes) -> float:
        """Analyze amplitude for 16-bit integer audio data."""
        try:
            sample_count = len(audio_data) // 2  # 2 bytes per int16 sample
            if sample_count == 0:
                return 0.0

            samples = struct.unpack(f"<{sample_count}h", audio_data)
            return float(max(abs(sample) for sample in samples))

        except Exception as e:
            logger.error(f"❌ SilenceDetector: Error analyzing int16 amplitude: {e}")
            return 0.0

    def _analyze_int24_amplitude(self, audio_data: bytes) -> float:
        """Analyze amplitude for 24-bit integer audio data."""
        # 24-bit is complex due to non-standard byte alignment
        # For now, treat as int32 with reduced threshold
        try:
            # Pad to 32-bit alignment (simple approximation)
            padded_samples = []
            for i in range(0, len(audio_data), 3):
                if i + 2 < len(audio_data):
                    # Convert 3-byte 24-bit to 4-byte 32-bit (left-justified)
                    bytes_24 = audio_data[i : i + 3]
                    bytes_32 = bytes_24 + b'\x00'  # Pad with zero byte
                    sample = (
                        struct.unpack("<i", bytes_32)[0] >> 8
                    )  # Right-shift to get 24-bit value
                    padded_samples.append(abs(sample))

            return float(max(padded_samples)) if padded_samples else 0.0

        except Exception as e:
            logger.error(f"❌ SilenceDetector: Error analyzing int24 amplitude: {e}")
            return 0.0

    def _analyze_int32_amplitude(self, audio_data: bytes) -> float:
        """Analyze amplitude for 32-bit integer audio data."""
        try:
            sample_count = len(audio_data) // 4  # 4 bytes per int32 sample
            if sample_count == 0:
                return 0.0

            samples = struct.unpack(f"<{sample_count}i", audio_data)
            return float(max(abs(sample) for sample in samples))

        except Exception as e:
            logger.error(f"❌ SilenceDetector: Error analyzing int32 amplitude: {e}")
            return 0.0

    def _analyze_float32_amplitude(self, audio_data: bytes) -> float:
        """Analyze amplitude for 32-bit float audio data."""
        try:
            sample_count = len(audio_data) // 4  # 4 bytes per float32 sample
            if sample_count == 0:
                return 0.0

            samples = struct.unpack(f"<{sample_count}f", audio_data)
            return max(abs(sample) for sample in samples)

        except Exception as e:
            logger.error(f"❌ SilenceDetector: Error analyzing float32 amplitude: {e}")
            return 0.0

    def reset_silence_tracking(self) -> None:
        """Reset silence tracking state (called when recording starts/stops)."""
        with self._lock:
            self._silence_start_time = None
            self._last_activity_time = time.time()
            self._total_chunks_analyzed = 0
            self._silence_chunks_detected = 0

        if self._enabled:
            logger.info("🔇 SilenceDetector: Silence tracking reset")

    def set_auto_stop_callback(self, callback: Callable[[], None]) -> None:
        """Set or update the auto-stop callback function.

        Args:
            callback: Function to call when silence timeout exceeded
        """
        with self._lock:
            self.auto_stop_callback = callback

        logger.info("🔇 SilenceDetector: Auto-stop callback updated")

    def get_silence_stats(self) -> dict[str, Any]:
        """Get current silence detection statistics.

        Returns:
            Dictionary with silence detection statistics
        """
        with self._lock:
            current_time = time.time()

            stats = {
                "enabled": self._enabled,
                "silence_timeout_seconds": self.silence_timeout_seconds,
                "silence_threshold": self.silence_threshold,
                "audio_format": self._audio_format,
                "channels": self._channels,
                "total_chunks_analyzed": self._total_chunks_analyzed,
                "silence_chunks_detected": self._silence_chunks_detected,
                "last_activity_time": self._last_activity_time,
                "current_silence_duration": (
                    current_time - self._silence_start_time
                    if self._silence_start_time is not None
                    else 0.0
                ),
                "is_currently_silent": self._silence_start_time is not None,
            }

            # Calculate silence percentage
            if self._total_chunks_analyzed > 0:
                stats["silence_percentage"] = (
                    100.0 * self._silence_chunks_detected / self._total_chunks_analyzed
                )
            else:
                stats["silence_percentage"] = 0.0

            return stats

    def is_enabled(self) -> bool:
        """Check if silence detection is enabled.

        Returns:
            True if silence detection is active, False otherwise
        """
        return self._enabled
