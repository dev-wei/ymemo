"""Audio saver component - independent consumer of audio stream."""

import logging
from datetime import datetime
from pathlib import Path

from ..core.interfaces import AudioConfig
from .audio_file_writer import AudioFileWriter
from .channel_splitter import AudioChannelSplitter

logger = logging.getLogger(__name__)


class AudioSaver:
    """
    Independent consumer of audio stream that saves raw audio to WAV files.

    This component runs in parallel with transcription providers, consuming
    the same raw audio stream and saving it to disk for debugging or archival.
    It's a terminal consumer - it doesn't pass data to any next stage.
    """

    def __init__(
        self,
        enabled: bool,
        save_path: str,
        max_duration: int,
        audio_config: AudioConfig,
        save_split_audio: bool = False,
    ):
        """
        Initialize audio saver consumer.

        Args:
            enabled: Whether audio saving is enabled
            save_path: Directory path where audio files will be saved
            max_duration: Maximum recording duration in seconds
            audio_config: Audio configuration (sample rate, channels, format)
            save_split_audio: Whether to save left/right channels separately for stereo input
        """
        self.enabled = enabled
        self.save_path = save_path
        self.max_duration = max_duration
        self.audio_config = audio_config
        self.save_split_audio = save_split_audio
        self.audio_writer = None
        self.channel_splitter = None
        self.is_active = False

        # Initialize channel splitter if split audio saving is enabled and input is stereo
        if self.save_split_audio and audio_config.channels == 2:
            logger.info(
                "🔀 AudioSaver: Initializing channel splitter for split audio saving (stereo input detected)"
            )
            logger.info(
                "   💡 AudioSaver is the PRIMARY component responsible for split audio saving"
            )

            self.channel_splitter = AudioChannelSplitter(
                audio_format=audio_config.format,
                silence_threshold=50,
                enable_audio_saving=True,
                audio_save_path=save_path,
                sample_rate=audio_config.sample_rate,
                save_duration=max_duration,
            )
            logger.info(f"🔀 AudioSaver: Channel splitter initialized for stereo input")
        elif self.save_split_audio and audio_config.channels == 1:
            logger.info(
                "🔀 AudioSaver: Split audio saving requested for mono input - this is the correct behavior"
            )
            logger.info(
                "   💡 For mono input, AudioSaver will save a single raw audio file (no splitting needed)"
            )

        # Add comprehensive configuration debugging
        logger.info("🔧 AudioSaver: Configuration Debug Summary:")
        logger.info(f"   ✅ enabled: {self.enabled}")
        logger.info(f"   🔀 save_split_audio: {self.save_split_audio}")
        logger.info(f"   🎚️  audio_config.channels: {self.audio_config.channels}")
        logger.info(f"   📁 save_path: {self.save_path}")
        logger.info(f"   ⏱️  max_duration: {self.max_duration}s")
        logger.info(
            f"   🎛️  channel_splitter initialized: {self.channel_splitter is not None}"
        )

        if self.enabled:
            logger.info(
                f"🎵 AudioSaver: Initialized - path: {save_path}, max_duration: {max_duration}s"
            )
            logger.info(
                "   🎯 AudioSaver is the ONLY component that should save audio files in production"
            )

            if self.save_split_audio:
                if audio_config.channels == 2:
                    logger.info(
                        "🔀 AudioSaver: Split audio saving ENABLED for stereo input"
                    )
                    logger.info("   📁 Will create: 1 raw file + 2 split channel files")
                    logger.info(
                        f"   🔀 Channel splitter status: {'✅ initialized' if self.channel_splitter else '❌ failed'}"
                    )
                else:
                    logger.info(
                        "🔀 AudioSaver: Split audio saving requested but input is mono"
                    )
                    logger.info(
                        "   📁 Will create: 1 raw file only (no splitting needed)"
                    )
            else:
                logger.info("🔀 AudioSaver: Split audio saving DISABLED")
                logger.info("   📁 Will create: 1 raw file only")
        else:
            logger.info("🎵 AudioSaver: DISABLED - no audio files will be saved")
            logger.info(
                "   💡 To enable: set SAVE_RAW_AUDIO=true or SAVE_SPLIT_AUDIO=true"
            )

    def start_saving(self) -> None:
        """Start audio saving session."""
        if not self.enabled:
            logger.debug(
                "🎵 AudioSaver: Skipping start_saving - AudioSaver is disabled"
            )
            return

        # Validate that previous session is completely cleaned up
        if self.is_active or self.audio_writer is not None:
            logger.warning("⚠️ AudioSaver: Previous session not properly cleaned up!")
            logger.warning(
                f"   🔍 is_active: {self.is_active}, audio_writer exists: {self.audio_writer is not None}"
            )
            logger.warning(
                "   🧹 Performing emergency cleanup before starting new session"
            )

            # Emergency cleanup
            if self.audio_writer:
                try:
                    self.audio_writer.stop_recording()
                except Exception as e:
                    logger.error(f"❌ Emergency cleanup error: {e}")
                finally:
                    self.audio_writer = None
            self.is_active = False

        # Add device switch validation logging
        logger.info("🔧 AudioSaver: Device Switch Validation for start_saving():")
        logger.info(
            f"   🎛️  Current audio_config: {self.audio_config.channels}ch, {self.audio_config.sample_rate}Hz"
        )
        logger.info(f"   🔀 Split audio enabled: {self.save_split_audio}")
        logger.info(
            f"   📁 Channel splitter ready: {self.channel_splitter is not None}"
        )

        # Validate channel splitter state consistency
        if self.save_split_audio and self.audio_config.channels == 2:
            if not self.channel_splitter:
                logger.error(
                    "❌ AudioSaver: INCONSISTENT STATE - Split audio enabled for 2ch but no channel splitter!"
                )
                logger.error(
                    "   💡 This will cause only 1 raw file instead of 3 files (raw + left + right)"
                )
                logger.error(
                    "   🔧 Attempting to fix by re-initializing channel splitter..."
                )
                try:
                    self.channel_splitter = AudioChannelSplitter(
                        audio_format=self.audio_config.format,
                        silence_threshold=50,
                        enable_audio_saving=True,
                        audio_save_path=self.save_path,
                        sample_rate=self.audio_config.sample_rate,
                        save_duration=self.max_duration,
                    )
                    logger.info(
                        "✅ AudioSaver: Channel splitter emergency re-initialization successful"
                    )
                except Exception as e:
                    logger.error(
                        f"❌ AudioSaver: Channel splitter emergency re-initialization failed: {e}"
                    )
            else:
                logger.info(
                    "✅ AudioSaver: Channel splitter state is consistent for 2ch device"
                )

        elif self.save_split_audio and self.audio_config.channels == 1:
            logger.info(
                "✅ AudioSaver: Mono device with split audio - will save 1 raw file (expected)"
            )
        elif not self.save_split_audio:
            logger.info(
                "✅ AudioSaver: Split audio disabled - will save 1 raw file (expected)"
            )

        logger.info("🎵 AudioSaver: Starting new audio saving session")

        try:
            # Create save directory if it doesn't exist
            save_dir = Path(self.save_path)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"
            file_path = save_dir / filename

            # Initialize audio file writer with comprehensive logging
            logger.info(
                "🎵 AudioSaver: Initializing AudioFileWriter with current configuration:"
            )
            logger.info(f"   📁 File path: {file_path}")
            logger.info(f"   📊 Sample rate: {self.audio_config.sample_rate}Hz")
            logger.info(f"   🎛️ Channels: {self.audio_config.channels}")
            logger.info(f"   📋 Sample width: 2 bytes (int16)")
            logger.info(f"   ⏱️ Max duration: {self.max_duration}s")

            # Validate audio configuration before creating writer
            if self.audio_config.sample_rate <= 0:
                logger.error(
                    f"❌ AudioSaver: Invalid sample rate: {self.audio_config.sample_rate}Hz"
                )
                self.enabled = False
                return

            if self.audio_config.channels <= 0 or self.audio_config.channels > 2:
                logger.error(
                    f"❌ AudioSaver: Invalid channel count: {self.audio_config.channels}"
                )
                self.enabled = False
                return

            # Critical: Ensure we're creating a fresh AudioFileWriter for the current configuration
            logger.info(
                "🔧 AudioSaver: Creating fresh AudioFileWriter instance for current device configuration"
            )
            logger.info(
                "   💡 This ensures no format conflicts from previous device sessions"
            )

            self.audio_writer = AudioFileWriter(
                file_path=str(file_path),
                sample_rate=self.audio_config.sample_rate,
                channels=self.audio_config.channels,
                sample_width=2,  # int16 = 2 bytes
                max_duration=self.max_duration,
            )

            # Validate AudioFileWriter was created with correct configuration
            logger.info("✅ AudioSaver: AudioFileWriter Configuration Validation:")
            logger.info(
                f"   📊 Created sample rate: {getattr(self.audio_writer, 'sample_rate', 'unknown')}"
            )
            logger.info(
                f"   🎛️ Created channels: {getattr(self.audio_writer, 'channels', 'unknown')}"
            )
            logger.info(
                f"   📁 Target file: {getattr(self.audio_writer, 'file_path', 'unknown')}"
            )

            # Validate that AudioFileWriter config matches AudioSaver config
            writer_channels = getattr(self.audio_writer, 'channels', None)
            writer_sample_rate = getattr(self.audio_writer, 'sample_rate', None)

            if writer_channels != self.audio_config.channels:
                logger.error(f"❌ AudioSaver: AudioFileWriter channel mismatch!")
                logger.error(
                    f"   Expected: {self.audio_config.channels}ch, Got: {writer_channels}ch"
                )
                logger.error(
                    "   💡 This WILL cause audio corruption - aborting initialization"
                )
                self.audio_writer = None
                self.enabled = False
                return

            if writer_sample_rate != self.audio_config.sample_rate:
                logger.error(f"❌ AudioSaver: AudioFileWriter sample rate mismatch!")
                logger.error(
                    f"   Expected: {self.audio_config.sample_rate}Hz, Got: {writer_sample_rate}Hz"
                )
                logger.error(
                    "   💡 This WILL cause audio corruption - aborting initialization"
                )
                self.audio_writer = None
                self.enabled = False
                return

            logger.info(
                "✅ AudioSaver: AudioFileWriter configuration validation passed"
            )

            # Start the recording session - this is critical!
            logger.info("🎵 AudioSaver: Starting AudioFileWriter recording session...")
            if not self.audio_writer.start_recording():
                logger.error(f"❌ AudioSaver: Failed to start recording session")
                logger.error(
                    "   💡 Check file permissions, disk space, or audio format compatibility"
                )
                self.audio_writer = None
                self.enabled = False
                return

            logger.info(
                "✅ AudioSaver: AudioFileWriter recording session started successfully"
            )

            self.is_active = True
            logger.info(f"🎵 AudioSaver: Started saving to {file_path}")

        except Exception as e:
            logger.error(f"❌ AudioSaver: Failed to start saving: {e}")
            self.enabled = False

    async def save_audio_chunk(self, audio_chunk: bytes) -> None:
        """
        Consume audio chunk and save to file.

        This is a terminal consumer - it processes the audio but doesn't
        pass it to any next stage in the pipeline. Supports both raw audio
        saving and split channel saving for stereo input.

        Args:
            audio_chunk: Raw audio data to save
        """
        if not self.enabled or not self.is_active:
            return

        try:
            # Initialize audio data validation tracking
            if not hasattr(self, '_debug_chunk_count'):
                self._debug_chunk_count = 0
                self._audio_data_validated = False

            self._debug_chunk_count += 1

            # Validate audio data format on first chunk
            if not self._audio_data_validated and self.audio_writer:
                logger.info(
                    "🔍 AudioSaver: Validating audio data format compatibility:"
                )
                logger.info(f"   📊 Received chunk size: {len(audio_chunk)} bytes")

                # Calculate expected chunk properties
                bytes_per_sample = 2  # int16 = 2 bytes
                samples_per_chunk = len(audio_chunk) // bytes_per_sample
                expected_samples_per_channel = (
                    samples_per_chunk // self.audio_config.channels
                    if self.audio_config.channels > 0
                    else 0
                )

                logger.info(
                    f"   🎚️ Expected format: {self.audio_config.channels}ch, {self.audio_config.sample_rate}Hz, int16"
                )
                logger.info(
                    f"   📈 Calculated: {samples_per_chunk} total samples, {expected_samples_per_channel} per channel"
                )

                # Validate chunk size alignment
                if (
                    len(audio_chunk) % (bytes_per_sample * self.audio_config.channels)
                    != 0
                ):
                    logger.error(
                        f"❌ AudioSaver: Audio data not properly aligned for {self.audio_config.channels} channels"
                    )
                    logger.error(
                        f"   💡 Chunk size {len(audio_chunk)} not divisible by {bytes_per_sample * self.audio_config.channels}"
                    )
                    logger.error(
                        "   🔧 This could cause audio corruption - check device configuration"
                    )
                else:
                    logger.info("✅ AudioSaver: Audio data format validation passed")

                self._audio_data_validated = True

            # Save raw audio to main file
            if self.audio_writer:
                self.audio_writer.write_audio_data(audio_chunk)

            if self._debug_chunk_count % 100 == 0:
                logger.info(f"🔍 AudioSaver Debug (chunk #{self._debug_chunk_count}):")
                logger.info(
                    f"   🎛️  channel_splitter exists: {self.channel_splitter is not None}"
                )
                logger.info(f"   🔀 save_split_audio: {self.save_split_audio}")
                logger.info(
                    f"   🎚️  audio_config.channels: {self.audio_config.channels}"
                )
                condition_result = (
                    self.channel_splitter
                    and self.save_split_audio
                    and self.audio_config.channels == 2
                )
                logger.info(f"   ✅ Split condition result: {condition_result}")

                if not condition_result:
                    if not self.channel_splitter:
                        logger.info("   ❌ Reason: channel_splitter is None")
                    elif not self.save_split_audio:
                        logger.info("   ❌ Reason: save_split_audio is False")
                    elif self.audio_config.channels != 2:
                        logger.info(
                            f"   ❌ Reason: channels != 2 (actual: {self.audio_config.channels})"
                        )

            # Save split channels if enabled and stereo input
            if (
                self.channel_splitter
                and self.save_split_audio
                and self.audio_config.channels == 2
            ):
                split_result = self.channel_splitter.split_stereo_chunk(audio_chunk)
                if not split_result.split_successful:
                    logger.warning(
                        f"⚠️ AudioSaver: Channel splitting failed: {split_result.error_message}"
                    )
                    logger.warning(
                        "   💡 This may indicate mono audio is being sent to AudioSaver configured for stereo splitting"
                    )
                # Add success logging every 100 chunks
                elif self._debug_chunk_count % 100 == 0:
                    logger.info("✅ AudioSaver: Channel splitting successful")
            elif self.save_split_audio and self.audio_config.channels == 1:
                # Log when mono audio is correctly handled (no splitting needed)
                if not hasattr(self, '_mono_split_logged'):
                    logger.info(
                        "🔀 AudioSaver: Processing mono audio - no channel splitting needed (expected behavior)"
                    )
                    self._mono_split_logged = True

        except Exception as e:
            logger.error(f"❌ AudioSaver: Error saving audio chunk: {e}")
            # Don't disable on single chunk failure, continue trying

    def stop_saving(self) -> dict:
        """
        Stop audio saving and return statistics.

        Returns:
            dict: Statistics about the saving session, including split channel stats if applicable
        """
        logger.info("🛑 AudioSaver: stop_saving() called")
        logger.info(
            f"   🔍 Current state - enabled: {self.enabled}, is_active: {self.is_active}"
        )
        logger.info(f"   📊 audio_writer exists: {self.audio_writer is not None}")

        if not self.enabled or not self.is_active:
            logger.info("🛑 AudioSaver: Not active, returning empty stats")
            return {}

        try:
            logger.info("🛑 AudioSaver: Stopping active audio saving session...")
            self.is_active = False
            stats = {}

            # Stop main audio writer
            if self.audio_writer:
                main_stats = {
                    "file_path": getattr(self.audio_writer, "file_path", "unknown"),
                    "duration_saved": getattr(self.audio_writer, "duration", 0),
                    "bytes_written": getattr(self.audio_writer, "bytes_written", 0),
                }
                stats["main_audio"] = main_stats

                # Stop the audio writer
                self.audio_writer.stop_recording()
                self.audio_writer = None

                logger.info(
                    f"🎵 AudioSaver: Stopped main audio - saved {main_stats['duration_saved']:.1f}s "
                    f"({main_stats['bytes_written']:,} bytes) to {main_stats['file_path']}"
                )

            # Stop channel splitter if active
            if self.channel_splitter and self.save_split_audio:
                split_stats = self.channel_splitter.stop_audio_saving()
                if split_stats:
                    stats["split_audio"] = split_stats
                    logger.info("🔀 AudioSaver: Stopped split audio saving")
                    if "left_channel" in split_stats:
                        logger.info(
                            f"   📁 Left: {split_stats['left_channel'].get('file_path', 'N/A')}"
                        )
                    if "right_channel" in split_stats:
                        logger.info(
                            f"   📁 Right: {split_stats['right_channel'].get('file_path', 'N/A')}"
                        )

            return stats if stats else {}

        except Exception as e:
            logger.error(f"❌ AudioSaver: Error stopping audio saver: {e}")
            return {}

    def update_audio_config(self, new_audio_config: AudioConfig) -> None:
        """
        Update audio configuration for device switching.

        This method allows AudioSaver to adapt when the user switches between
        devices with different channel counts (e.g., mono to stereo).

        Args:
            new_audio_config: Updated audio configuration from device optimization
        """
        if self.is_active:
            logger.warning(
                "⚠️ AudioSaver: Cannot update configuration while actively saving"
            )
            return

        old_channels = self.audio_config.channels
        old_sample_rate = self.audio_config.sample_rate
        new_channels = new_audio_config.channels
        new_sample_rate = new_audio_config.sample_rate

        # Update the audio configuration
        self.audio_config = new_audio_config

        # Check if ANY audio parameter changed that would require AudioFileWriter recreation
        config_changed = (
            old_channels != new_channels or old_sample_rate != new_sample_rate
        )

        if config_changed:
            logger.info(
                f"🔄 AudioSaver: Audio configuration changed for device switching:"
            )
            logger.info(f"   🎛️ Channels: {old_channels}ch → {new_channels}ch")
            logger.info(f"   📊 Sample Rate: {old_sample_rate}Hz → {new_sample_rate}Hz")
            logger.info(f"   🔀 Split audio setting: {self.save_split_audio}")

            # CRITICAL: Clean up any existing AudioFileWriter to prevent format conflicts
            if self.audio_writer:
                logger.warning(
                    "🧹 AudioSaver: Cleaning up existing AudioFileWriter from previous device"
                )
                logger.warning(
                    "   💡 This is necessary to prevent audio format corruption between devices"
                )
                try:
                    # Force stop any active recording session
                    if (
                        hasattr(self.audio_writer, 'is_recording')
                        and self.audio_writer.is_recording
                    ):
                        logger.info(
                            "   🛑 Stopping active AudioFileWriter recording session"
                        )
                        self.audio_writer.stop_recording()
                    logger.info("   ✅ AudioFileWriter cleaned up successfully")
                except Exception as e:
                    logger.error(f"   ❌ Error cleaning AudioFileWriter: {e}")
                finally:
                    self.audio_writer = None
                    logger.info("   🧹 AudioFileWriter reference cleared")

            logger.info(f"   📋 Expected file creation for new device:")

            # Re-initialize channel splitter based on new configuration
            old_splitter = self.channel_splitter
            self.channel_splitter = None

            if self.save_split_audio and new_channels == 2:
                logger.info("   📁 Will create: 1 raw file + 2 split channel files")
                logger.info(
                    "🔀 AudioSaver: Initializing channel splitter for new stereo device"
                )
                try:
                    self.channel_splitter = AudioChannelSplitter(
                        audio_format=new_audio_config.format,
                        silence_threshold=50,
                        enable_audio_saving=True,
                        audio_save_path=self.save_path,
                        sample_rate=new_audio_config.sample_rate,
                        save_duration=self.max_duration,
                    )
                    logger.info(
                        "✅ AudioSaver: Channel splitter re-initialized for stereo device"
                    )
                    logger.info(
                        "   🎛️  Channel splitter will save left/right files during recording"
                    )
                except Exception as e:
                    logger.error(
                        f"❌ AudioSaver: Failed to initialize channel splitter: {e}"
                    )
                    logger.error(
                        "   💡 Split channel files will NOT be created due to initialization failure"
                    )
                    self.channel_splitter = None

            elif self.save_split_audio and new_channels == 1:
                logger.info(
                    "   📁 Will create: 1 raw file only (mono - no splitting possible)"
                )
                logger.info(
                    "🔀 AudioSaver: No channel splitter needed for mono device (expected)"
                )
            elif not self.save_split_audio:
                logger.info("   📁 Will create: 1 raw file only (split audio disabled)")
                logger.info(
                    "🔀 AudioSaver: Split audio disabled - no channel splitter needed"
                )

            # Clean up old splitter if it existed
            if old_splitter:
                logger.info(
                    "🧹 AudioSaver: Cleaned up old channel splitter from previous device"
                )

        else:
            logger.debug(
                f"🔄 AudioSaver: Audio config updated, no significant changes ({new_channels}ch, {new_sample_rate}Hz)"
            )

    def is_saving_active(self) -> bool:
        """Check if audio saving is currently active."""
        return self.enabled and self.is_active and self.audio_writer is not None

    def get_current_file_path(self) -> str | None:
        """Get the path of the currently active audio file."""
        if self.audio_writer and hasattr(self.audio_writer, 'file_path'):
            return str(self.audio_writer.file_path)
        return None
