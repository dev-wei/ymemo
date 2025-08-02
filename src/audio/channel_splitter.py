"""Audio channel splitting utilities for dual-stream processing."""

import asyncio
import logging
import struct
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from .audio_file_writer import DualChannelAudioSaver

logger = logging.getLogger(__name__)


@dataclass
class ChannelMetrics:
    """Metrics for a single audio channel."""

    sample_count: int = 0
    max_amplitude: int = 0
    avg_amplitude: float = 0.0
    rms_amplitude: float = 0.0
    is_silent: bool = True
    activity_level: str = "silent"


@dataclass
class SplitResult:
    """Result of channel splitting operation."""

    left_channel: bytes
    right_channel: bytes
    left_metrics: ChannelMetrics
    right_metrics: ChannelMetrics
    original_size: int
    split_successful: bool = True
    error_message: str | None = None


class AudioChannelSplitter:
    """
    Splits stereo audio into separate left and right mono streams.

    This class provides the core functionality for dual AWS Transcribe architecture,
    converting stereo audio into two independent mono streams that can be processed
    by separate transcription services.
    """

    def __init__(
        self,
        audio_format: str = "int16",
        silence_threshold: int = 50,
        enable_audio_saving: bool = False,
        audio_save_path: str = "./debug_audio/",
        sample_rate: int = 16000,
        save_duration: int = 30,
    ):
        """
        Initialize the channel splitter.

        Args:
            audio_format: Audio format ('int16', 'int24', 'int32', 'float32')
            silence_threshold: Amplitude threshold below which channel is considered silent
            enable_audio_saving: Enable saving split audio to files for debugging
            audio_save_path: Directory path for saving audio files
            sample_rate: Audio sample rate in Hz (needed for proper WAV file creation)
            save_duration: Maximum duration to save audio in seconds
        """
        self.audio_format = audio_format
        self.silence_threshold = silence_threshold
        self.enable_audio_saving = enable_audio_saving
        self.sample_rate = sample_rate

        # Format configuration
        self.format_config = {
            "int16": {"bytes_per_sample": 2, "struct_format": "h", "max_value": 32767},
            "int24": {
                "bytes_per_sample": 3,
                "struct_format": "i",
                "max_value": 8388607,
            },  # Note: 24-bit handling is complex
            "int32": {
                "bytes_per_sample": 4,
                "struct_format": "i",
                "max_value": 2147483647,
            },
            "float32": {"bytes_per_sample": 4, "struct_format": "f", "max_value": 1.0},
        }

        if audio_format not in self.format_config:
            raise ValueError(
                f"Unsupported audio format: {audio_format}. Supported: {list(self.format_config.keys())}"
            )

        self.config = self.format_config[audio_format]
        self.bytes_per_sample = self.config["bytes_per_sample"]
        self.struct_format = self.config["struct_format"]

        # Statistics
        self.total_chunks_processed = 0
        self.total_bytes_processed = 0
        self.left_silent_chunks = 0
        self.right_silent_chunks = 0

        # Audio saving for debugging
        self.audio_saver = None
        if self.enable_audio_saving:
            try:
                self.audio_saver = DualChannelAudioSaver(
                    save_path=audio_save_path,
                    sample_rate=sample_rate,
                    duration=save_duration,
                )
                logger.info("🎵 AudioChannelSplitter: Audio saving ENABLED")
                logger.info(f"   📁 Save path: {audio_save_path}")
                logger.info(f"   ⏱️  Duration: {save_duration}s")
            except Exception as e:
                logger.error(
                    f"❌ AudioChannelSplitter: Failed to initialize audio saver: {e}"
                )
                self.audio_saver = None
                self.enable_audio_saving = False

        logger.info(
            f"🔀 AudioChannelSplitter initialized: format={audio_format}, "
            f"bytes_per_sample={self.bytes_per_sample}, silence_threshold={silence_threshold}"
        )
        if self.enable_audio_saving:
            logger.info("🎵 AudioChannelSplitter: Audio saving enabled for debugging")

    def split_stereo_chunk(self, audio_chunk: bytes) -> SplitResult:
        """
        Split a stereo audio chunk into separate left and right mono chunks.

        Args:
            audio_chunk: Stereo audio data in specified format

        Returns:
            SplitResult containing left/right channels and metrics
        """
        try:
            chunk_size = len(audio_chunk)
            self.total_chunks_processed += 1
            self.total_bytes_processed += chunk_size

            # Enhanced validation with detailed logging
            bytes_per_stereo_sample = self.bytes_per_sample * 2  # Left + Right
            if chunk_size % bytes_per_stereo_sample != 0:
                error_msg = f"Invalid chunk size {chunk_size} for stereo {self.audio_format} (expected multiple of {bytes_per_stereo_sample})"
                logger.error(f"❌ CHANNEL SPLIT VALIDATION: {error_msg}")
                logger.error(f"   📊 Chunk size: {chunk_size} bytes")
                logger.error(f"   📊 Bytes per sample: {self.bytes_per_sample}")
                logger.error(
                    f"   📊 Expected stereo sample size: {bytes_per_stereo_sample} bytes"
                )
                logger.error(
                    f"   📊 Remainder: {chunk_size % bytes_per_stereo_sample} bytes"
                )
                return SplitResult(
                    left_channel=b"",
                    right_channel=b"",
                    left_metrics=ChannelMetrics(),
                    right_metrics=ChannelMetrics(),
                    original_size=chunk_size,
                    split_successful=False,
                    error_message=error_msg,
                )

            # Log validation success for first few chunks
            if self.total_chunks_processed <= 5:
                stereo_sample_count = chunk_size // bytes_per_stereo_sample
                logger.info(
                    f"✅ CHANNEL SPLIT VALIDATION (chunk #{self.total_chunks_processed}):"
                )
                logger.info(f"   📊 Chunk size: {chunk_size} bytes")
                logger.info(f"   📊 Stereo samples: {stereo_sample_count}")
                logger.info(f"   📊 Expected mono output: {chunk_size // 2} bytes each")

            stereo_sample_count = chunk_size // bytes_per_stereo_sample

            # Unpack stereo samples
            if self.audio_format == "int24":
                # Special handling for 24-bit audio (complex unpacking)
                samples = self._unpack_int24_stereo(audio_chunk)
            else:
                # Standard unpacking for other formats
                format_str = f"<{stereo_sample_count * 2}{self.struct_format}"
                samples = struct.unpack(format_str, audio_chunk)

            # Split into left and right channels
            left_samples = samples[0::2]  # Every other sample starting from 0
            right_samples = samples[1::2]  # Every other sample starting from 1

            # Analyze each channel
            left_metrics = self._analyze_channel(left_samples, "Left")
            right_metrics = self._analyze_channel(right_samples, "Right")

            # Update silence statistics
            if left_metrics.is_silent:
                self.left_silent_chunks += 1
            if right_metrics.is_silent:
                self.right_silent_chunks += 1

            # Pack back to mono chunks
            left_channel = self._pack_mono_samples(left_samples)
            right_channel = self._pack_mono_samples(right_samples)

            # Create split result for logging
            split_result = SplitResult(
                left_channel=left_channel,
                right_channel=right_channel,
                left_metrics=left_metrics,
                right_metrics=right_metrics,
                original_size=chunk_size,
                split_successful=True,
            )

            # Enhanced debugging: Log split results
            self._log_split_results(
                split_result, left_metrics, right_metrics, chunk_size
            )

            # Save split audio for debugging if enabled
            if self.enable_audio_saving and self.audio_saver:
                if self.total_chunks_processed == 1:  # Start recording on first chunk
                    if self.audio_saver.start_recording():
                        logger.info(
                            "🎵 AudioChannelSplitter: Started saving split audio to files"
                        )
                        file_paths = self.audio_saver.get_file_paths()
                        logger.info(f"   📁 Left channel: {file_paths['left']}")
                        logger.info(f"   📁 Right channel: {file_paths['right']}")

                # Write audio data to files with validation
                if self.audio_saver.is_active:
                    left_written = self.audio_saver.write_left_audio(left_channel)
                    right_written = self.audio_saver.write_right_audio(right_channel)

                    # Log write failures
                    if not left_written:
                        logger.warning(
                            f"⚠️ AudioChannelSplitter: Failed to write left channel audio (chunk #{self.total_chunks_processed})"
                        )
                    if not right_written:
                        logger.warning(
                            f"⚠️ AudioChannelSplitter: Failed to write right channel audio (chunk #{self.total_chunks_processed})"
                        )

            # Log detailed analysis for first few chunks
            if self.total_chunks_processed <= 10:
                logger.info(f"🔀 Channel split #{self.total_chunks_processed}:")
                logger.info(
                    f"   📊 Original: {chunk_size} bytes → Left: {len(left_channel)} bytes, Right: {len(right_channel)} bytes"
                )
                logger.info(
                    f"   🎚️  Left: {left_metrics.activity_level} (max: {left_metrics.max_amplitude}, avg: {left_metrics.avg_amplitude:.1f})"
                )
                logger.info(
                    f"   🎚️  Right: {right_metrics.activity_level} (max: {right_metrics.max_amplitude}, avg: {right_metrics.avg_amplitude:.1f})"
                )
                if self.enable_audio_saving:
                    logger.info(
                        f"   🎵 Audio saving: {'ACTIVE' if self.audio_saver and self.audio_saver.is_active else 'INACTIVE'}"
                    )

            return split_result

        except Exception as e:
            error_msg = f"Channel splitting failed: {e}"
            logger.error(f"❌ {error_msg}")
            return SplitResult(
                left_channel=b"",
                right_channel=b"",
                left_metrics=ChannelMetrics(),
                right_metrics=ChannelMetrics(),
                original_size=len(audio_chunk) if audio_chunk else 0,
                split_successful=False,
                error_message=error_msg,
            )

    def _analyze_channel(self, samples: tuple, channel_name: str) -> ChannelMetrics:
        """Analyze a single channel's audio characteristics."""
        if not samples:
            return ChannelMetrics()

        # Convert to absolute values for amplitude analysis
        if self.audio_format == "float32":
            abs_samples = [abs(s) for s in samples]
            max_amp = (
                max(abs_samples) * self.config["max_value"]
            )  # Scale to int equivalent
            avg_amp = sum(abs_samples) / len(abs_samples) * self.config["max_value"]
            rms_amp = (sum(s * s for s in samples) / len(samples)) ** 0.5 * self.config[
                "max_value"
            ]
        else:
            abs_samples = [abs(s) for s in samples]
            max_amp = max(abs_samples)
            avg_amp = sum(abs_samples) / len(abs_samples)
            rms_amp = (sum(s * s for s in samples) / len(samples)) ** 0.5

        # Determine activity level and silence
        is_silent = max_amp < self.silence_threshold

        if max_amp < 50:
            activity_level = "silent"
        elif max_amp < 500:
            activity_level = "very_quiet"
        elif max_amp < 2000:
            activity_level = "quiet"
        elif max_amp < 8000:
            activity_level = "normal"
        elif max_amp < 20000:
            activity_level = "loud"
        else:
            activity_level = "very_loud"

        return ChannelMetrics(
            sample_count=len(samples),
            max_amplitude=int(max_amp),
            avg_amplitude=avg_amp,
            rms_amplitude=rms_amp,
            is_silent=is_silent,
            activity_level=activity_level,
        )

    def _pack_mono_samples(self, samples: tuple) -> bytes:
        """Pack mono samples back to bytes."""
        if self.audio_format == "int24":
            return self._pack_int24_mono(samples)
        format_str = f"<{len(samples)}{self.struct_format}"
        return struct.pack(format_str, *samples)

    def _unpack_int24_stereo(self, audio_chunk: bytes) -> tuple:
        """Special handling for 24-bit stereo unpacking."""
        # 24-bit is typically stored as 3 bytes per sample, but Python struct doesn't have native 24-bit
        # We'll read as bytes and convert manually
        samples = []
        for i in range(0, len(audio_chunk), 3):
            if i + 2 < len(audio_chunk):
                # Read 3 bytes and convert to int
                bytes_sample = audio_chunk[i : i + 3]
                # Convert 3-byte little-endian to signed int
                value = int.from_bytes(bytes_sample, byteorder="little", signed=True)
                samples.append(value)
        return tuple(samples)

    def _pack_int24_mono(self, samples: tuple) -> bytes:
        """Special handling for 24-bit mono packing."""
        result = b""
        for sample in samples:
            # Convert int to 3-byte little-endian
            sample_bytes = sample.to_bytes(3, byteorder="little", signed=True)
            result += sample_bytes
        return result

    async def split_audio_stream(
        self, audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[tuple[bytes, bytes, SplitResult], None]:
        """
        Split an async audio stream into left and right channels.

        Args:
            audio_stream: Async generator of stereo audio chunks

        Yields:
            Tuples of (left_channel_bytes, right_channel_bytes, split_result)
        """
        chunk_count = 0

        try:
            async for audio_chunk in audio_stream:
                chunk_count += 1

                # Split the chunk
                split_result = self.split_stereo_chunk(audio_chunk)

                if split_result.split_successful:
                    yield (
                        split_result.left_channel,
                        split_result.right_channel,
                        split_result,
                    )

                    # Periodic logging
                    if chunk_count % 100 == 0:
                        self._log_split_statistics(chunk_count)
                else:
                    logger.error(
                        f"❌ Failed to split chunk #{chunk_count}: {split_result.error_message}"
                    )

        except asyncio.CancelledError:
            logger.info(
                f"🛑 Channel splitter cancelled after processing {chunk_count} chunks"
            )
            raise
        except Exception as e:
            logger.error(f"❌ Channel splitter error after {chunk_count} chunks: {e}")
            raise
        finally:
            logger.info(
                f"🔀 Channel splitter completed: {chunk_count} chunks processed"
            )
            self._log_final_statistics()

            # Stop audio saving if active
            if (
                self.enable_audio_saving
                and self.audio_saver
                and self.audio_saver.is_active
            ):
                try:
                    save_stats = self.audio_saver.stop_recording()
                    logger.info("🎵 AudioChannelSplitter: Audio saving completed")
                    if "left_channel" in save_stats:
                        logger.info(
                            f"   📁 Left file: {save_stats['left_channel'].get('file_path', 'N/A')}"
                        )
                        logger.info(
                            f"   📁 Right file: {save_stats['right_channel'].get('file_path', 'N/A')}"
                        )
                except Exception as e:
                    logger.error(
                        f"❌ AudioChannelSplitter: Error stopping audio saver: {e}"
                    )

    def _log_split_statistics(self, chunk_count: int) -> None:
        """Log periodic splitting statistics."""
        left_silence_rate = (
            (self.left_silent_chunks / chunk_count) * 100 if chunk_count > 0 else 0
        )
        right_silence_rate = (
            (self.right_silent_chunks / chunk_count) * 100 if chunk_count > 0 else 0
        )

        logger.info(f"🔀 Channel Split Stats (chunk #{chunk_count}):")
        logger.info(f"   📊 Total processed: {self.total_bytes_processed:,} bytes")
        logger.info(
            f"   🔇 Left channel silence: {left_silence_rate:.1f}% ({self.left_silent_chunks}/{chunk_count})"
        )
        logger.info(
            f"   🔇 Right channel silence: {right_silence_rate:.1f}% ({self.right_silent_chunks}/{chunk_count})"
        )

        # Warn about potential issues
        if left_silence_rate > 80:
            logger.warning(
                f"⚠️ Left channel mostly silent ({left_silence_rate:.1f}%) - check Source A connection"
            )
        if right_silence_rate > 80:
            logger.warning(
                f"⚠️ Right channel mostly silent ({right_silence_rate:.1f}%) - check Source B connection"
            )

    def _log_final_statistics(self) -> None:
        """Log final splitting statistics."""
        if self.total_chunks_processed == 0:
            return

        left_silence_rate = (
            self.left_silent_chunks / self.total_chunks_processed
        ) * 100
        right_silence_rate = (
            self.right_silent_chunks / self.total_chunks_processed
        ) * 100

        logger.info("🔀 Final Channel Split Statistics:")
        logger.info(f"   📊 Total chunks: {self.total_chunks_processed}")
        logger.info(f"   📊 Total bytes: {self.total_bytes_processed:,}")
        logger.info(f"   🎚️  Left silence rate: {left_silence_rate:.1f}%")
        logger.info(f"   🎚️  Right silence rate: {right_silence_rate:.1f}%")

        # Final recommendations
        if left_silence_rate > 50 and right_silence_rate > 50:
            logger.warning(
                "⚠️ Both channels have high silence rates - check audio sources"
            )
        elif left_silence_rate > 80:
            logger.warning(
                "⚠️ Left channel (Source A) mostly silent - consider using only right channel"
            )
        elif right_silence_rate > 80:
            logger.warning(
                "⚠️ Right channel (Source B) mostly silent - consider using only left channel"
            )
        else:
            logger.info(
                "✅ Both channels have reasonable activity levels for dual transcription"
            )

    def get_statistics(self) -> dict[str, Any]:
        """Get current splitting statistics."""
        stats = {
            "total_chunks_processed": self.total_chunks_processed,
            "total_bytes_processed": self.total_bytes_processed,
            "left_silent_chunks": self.left_silent_chunks,
            "right_silent_chunks": self.right_silent_chunks,
            "left_silence_rate": (
                (self.left_silent_chunks / self.total_chunks_processed * 100)
                if self.total_chunks_processed > 0
                else 0
            ),
            "right_silence_rate": (
                (self.right_silent_chunks / self.total_chunks_processed * 100)
                if self.total_chunks_processed > 0
                else 0
            ),
            "audio_format": self.audio_format,
            "silence_threshold": self.silence_threshold,
            "audio_saving_enabled": self.enable_audio_saving,
        }

        # Add audio saving statistics if available
        if self.enable_audio_saving and self.audio_saver:
            stats["audio_save_files"] = self.audio_saver.get_file_paths()
            stats["audio_save_active"] = self.audio_saver.is_active

        return stats

    def stop_audio_saving(self) -> dict[str, Any] | None:
        """Manually stop audio saving and return statistics."""
        if self.enable_audio_saving and self.audio_saver and self.audio_saver.is_active:
            try:
                return self.audio_saver.stop_recording()
            except Exception as e:
                logger.error(
                    f"❌ AudioChannelSplitter: Error stopping audio saving: {e}"
                )
                return None
        return None

    def _log_split_results(
        self,
        split_result: "SplitResult",
        left_metrics: "ChannelMetrics",
        right_metrics: "ChannelMetrics",
        original_size: int,
    ) -> None:
        """
        Log detailed information about split results for debugging.

        Args:
            split_result: Result of the channel splitting operation
            left_metrics: Metrics for left channel
            right_metrics: Metrics for right channel
            original_size: Original chunk size before splitting
        """
        # Log every 25 chunks for detailed debugging
        if self.total_chunks_processed % 25 == 0:
            logger.info(
                f"🔀 CHANNEL SPLIT ANALYSIS (chunk #{self.total_chunks_processed}):"
            )
            logger.info(f"   📊 Input: {original_size} bytes")
            logger.info(
                f"   📊 Output: Left={len(split_result.left_channel)} bytes, Right={len(split_result.right_channel)} bytes"
            )
            logger.info(f"   ✅ Split successful: {split_result.split_successful}")

            if split_result.error_message:
                logger.warning(f"   ❌ Error: {split_result.error_message}")

            # Detailed channel analysis
            logger.info("   🎚️  LEFT CHANNEL:")
            logger.info(f"      - Activity: {left_metrics.activity_level}")
            logger.info(f"      - Max amplitude: {left_metrics.max_amplitude}")
            logger.info(f"      - Avg amplitude: {left_metrics.avg_amplitude:.1f}")
            logger.info(
                f"      - Is silent: {left_metrics.is_silent} (threshold: {self.silence_threshold})"
            )
            logger.info(f"      - Sample count: {left_metrics.sample_count}")

            logger.info("   🎚️  RIGHT CHANNEL:")
            logger.info(f"      - Activity: {right_metrics.activity_level}")
            logger.info(f"      - Max amplitude: {right_metrics.max_amplitude}")
            logger.info(f"      - Avg amplitude: {right_metrics.avg_amplitude:.1f}")
            logger.info(
                f"      - Is silent: {right_metrics.is_silent} (threshold: {self.silence_threshold})"
            )
            logger.info(f"      - Sample count: {right_metrics.sample_count}")

            # Audio saving status
            if self.enable_audio_saving and self.audio_saver:
                logger.info("   🎵 AUDIO SAVING:")
                logger.info(f"      - Active: {self.audio_saver.is_active}")
                if self.audio_saver.is_active:
                    left_stats = self.audio_saver.left_writer.get_statistics()
                    right_stats = self.audio_saver.right_writer.get_statistics()
                    logger.info(
                        f"      - Left file: {left_stats['bytes_written']:,} bytes written"
                    )
                    logger.info(
                        f"      - Right file: {right_stats['bytes_written']:,} bytes written"
                    )
                    logger.info(
                        f"      - Duration: {left_stats['elapsed_seconds']:.2f}s"
                    )

            # Warning for potential issues
            if left_metrics.is_silent and right_metrics.is_silent:
                logger.warning(
                    "   ⚠️  BOTH CHANNELS SILENT - This explains why WAV files have no audio!"
                )
            elif left_metrics.is_silent:
                logger.warning(
                    "   ⚠️  LEFT CHANNEL SILENT - Only right channel has audio"
                )
            elif right_metrics.is_silent:
                logger.warning(
                    "   ⚠️  RIGHT CHANNEL SILENT - Only left channel has audio"
                )
