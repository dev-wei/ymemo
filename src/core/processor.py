"""Real-time audio processing pipeline."""

import asyncio
import logging
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from src.config.audio_config import get_config

from ..analytics.session_analytics import SessionAnalytics
from ..audio.audio_saver import AudioSaver
from ..audio.silence_detector import SilenceDetector
from ..utils.exceptions import PipelineError, PipelineTimeoutError
from .factory import AudioProcessorFactory
from .interfaces import AudioCaptureProvider, TranscriptionProvider, TranscriptionResult
from .pipeline_error_handler import ErrorSeverity, PipelineErrorHandler
from .pipeline_monitor import PipelineMonitor, PipelineStage
from .resource_manager import ResourceManager

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Real-time audio processing pipeline coordinator."""

    def __init__(
        self,
        transcription_provider: str = "aws",
        capture_provider: str = "pyaudio",
        transcription_config: dict[str, Any] | None = None,
        capture_config: dict[str, Any] | None = None,
        error_handler_config: dict[str, Any] | None = None,
        session_analytics: SessionAnalytics | None = None,
    ):
        logger.info(
            f"🏗️  AudioProcessor: Initializing with transcription={transcription_provider}, capture={capture_provider}"
        )
        logger.debug(f"🔧 AudioProcessor: Transcription config: {transcription_config}")
        logger.debug(f"🔧 AudioProcessor: Capture config: {capture_config}")

        self.transcription_provider_name = transcription_provider
        self.capture_provider_name = capture_provider
        self.transcription_config = transcription_config or {}
        self.capture_config = capture_config or {}

        # Providers - initialize immediately for app lifecycle
        self.transcription_provider: TranscriptionProvider | None = None
        self.capture_provider: AudioCaptureProvider | None = None
        self._providers_initialized = False

        # Configuration - get from system config or use default
        system_config = get_config()
        self.audio_config = system_config.get_audio_config()
        logger.debug(
            f"🎚️  AudioProcessor: Audio config - sample_rate={self.audio_config.sample_rate}, channels={self.audio_config.channels}, format={self.audio_config.format}"
        )

        # Initialize audio saver as independent consumer
        audio_saving_config = system_config.get_audio_saving_config()
        self.audio_saver = self._initialize_audio_saver(audio_saving_config)

        # Initialize silence detector for auto-stop functionality
        self.silence_detector = self._initialize_silence_detector(system_config)

        # State
        self.is_running = False
        self.transcription_callback: Callable[[TranscriptionResult], None] | None = None
        self.error_callback: Callable[[Exception], None] | None = None
        self.connection_health_callback: Callable[[bool, str], None] | None = None
        self.silence_auto_stop_callback: Callable[[], None] | None = (
            None  # Callback for silence-triggered auto-stop
        )
        self._silence_auto_stop_requested = (
            False  # Flag for silence-triggered auto-stop
        )

        # Error handling
        error_config = error_handler_config or {}
        self.error_handler = PipelineErrorHandler(
            default_timeout=error_config.get("default_timeout", 30.0),
            max_retries=error_config.get("max_retries", 3),
            base_retry_delay=error_config.get("base_retry_delay", 1.0),
        )

        # Resource management
        self.resource_manager = ResourceManager(
            default_resource_timeout=error_config.get("resource_timeout", 5.0)
        )

        # Pipeline monitoring
        self.session_analytics = session_analytics
        self.pipeline_monitor = PipelineMonitor(
            session_analytics=session_analytics,
            metrics_retention_seconds=error_config.get("metrics_retention", 3600),
            health_check_interval_seconds=error_config.get(
                "health_check_interval", 30.0
            ),
        )

        # Tasks (managed by resource manager)
        self._capture_task: asyncio.Task | None = None
        self._transcription_task: asyncio.Task | None = None

        # Session data
        self.session_transcripts: list[TranscriptionResult] = []
        self.current_meeting_id: str | None = None

        # Initialize providers immediately for single-instance lifecycle
        self._initialize_providers_sync()

        logger.debug("✅ AudioProcessor: Initialization complete")

    def _initialize_providers_sync(self) -> None:
        """Initialize providers synchronously during constructor."""
        if self._providers_initialized:
            return

        try:
            logger.info(
                "🏭 AudioProcessor: Initializing providers for app lifecycle..."
            )

            # Create transcription provider
            logger.info(
                f"🏭 AudioProcessor: Creating transcription provider '{self.transcription_provider_name}'"
            )
            self.transcription_provider = (
                AudioProcessorFactory.create_transcription_provider(
                    self.transcription_provider_name, **self.transcription_config
                )
            )

            # Create audio capture provider
            logger.info(
                f"🎤 AudioProcessor: Creating capture provider '{self.capture_provider_name}'"
            )
            self.capture_provider = AudioProcessorFactory.create_audio_capture_provider(
                self.capture_provider_name, **self.capture_config
            )

            # Log provider instance details
            if hasattr(self.capture_provider, "_instance_id"):
                logger.info(
                    f"🔧 AudioProcessor: Created capture provider instance {self.capture_provider._instance_id}"
                )

            # Register providers with resource manager for cleanup on app shutdown
            self.resource_manager.register_resource(
                "transcription_provider",
                self.transcription_provider,
                cleanup_func=self._cleanup_transcription_provider,
                timeout=8.0,
            )

            self.resource_manager.register_resource(
                "capture_provider",
                self.capture_provider,
                cleanup_func=self._cleanup_capture_provider,
                timeout=5.0,
            )

            self._providers_initialized = True
            logger.info(
                "✅ AudioProcessor: Providers initialized successfully for app lifecycle"
            )

        except Exception as e:
            logger.error(f"❌ AudioProcessor: Provider initialization failed: {e}")
            raise RuntimeError(
                f"Failed to initialize audio processor providers: {e}"
            ) from e

    def _initialize_audio_saver(
        self, audio_saving_config: dict[str, Any]
    ) -> AudioSaver | None:
        """Initialize audio saver as independent consumer."""
        try:
            # Comprehensive validation of audio saving configuration
            enabled = audio_saving_config["enabled"]
            save_split_audio = audio_saving_config.get("save_split_audio", False)

            # Log comprehensive audio saving configuration
            logger.info("🎵 AudioProcessor: Validating audio saving configuration")
            logger.info(f"   🎵 Raw audio saving: {enabled}")
            logger.info(f"   🔀 Split audio saving: {save_split_audio}")
            logger.info(f"   📁 Save path: {audio_saving_config['save_path']}")
            logger.info(f"   ⏱️  Max duration: {audio_saving_config['max_duration']}s")
            logger.info(f"   🎚️  Input channels: {self.audio_config.channels}")
            logger.info(f"   📊 Sample rate: {self.audio_config.sample_rate}Hz")

            # Validate configuration consistency
            if save_split_audio and self.audio_config.channels == 1:
                logger.info(
                    "✅ AudioProcessor: Split audio requested for mono input - AudioSaver will handle correctly"
                )
                logger.info(
                    "   💡 Mono input + split audio = 1 raw file (no actual splitting needed)"
                )
            elif save_split_audio and self.audio_config.channels == 2:
                logger.info(
                    "✅ AudioProcessor: Split audio requested for stereo input - AudioSaver will create split files"
                )
                logger.info(
                    "   💡 Stereo input + split audio = 1 raw file + 2 split channel files"
                )
            elif not save_split_audio:
                logger.info(
                    "✅ AudioProcessor: Split audio disabled - AudioSaver will create raw file only"
                )
                logger.info(
                    f"   💡 {self.audio_config.channels}-channel input + no split = 1 raw file"
                )

            # Validate that AudioSaver is the ONLY component saving audio files
            logger.info(
                "🎯 AudioProcessor: AudioSaver is configured as the SINGLE audio saving component"
            )
            logger.info(
                "   ⚠️ No other components (AWS provider, channel splitter) should save audio files independently"
            )

            # Use device-agnostic placeholder configuration for AudioSaver initialization
            # The actual device-optimized configuration will be set during start_recording()
            from .interfaces import AudioConfig

            placeholder_audio_config = AudioConfig(
                sample_rate=self.audio_config.sample_rate,
                channels=1,  # Placeholder - will be updated with actual device channels during recording
                chunk_size=self.audio_config.chunk_size,
                format=self.audio_config.format,
            )

            logger.info(
                "🎵 AudioProcessor: Initializing AudioSaver with device-agnostic placeholder config"
            )
            logger.info(
                f"   📋 Placeholder config: {placeholder_audio_config.sample_rate}Hz, {placeholder_audio_config.channels}ch"
            )
            logger.info(
                "   💡 Real device configuration will be applied during start_recording()"
            )

            audio_saver = AudioSaver(
                enabled=enabled,
                save_path=audio_saving_config["save_path"],
                max_duration=audio_saving_config["max_duration"],
                audio_config=placeholder_audio_config,
                save_split_audio=save_split_audio,
            )

            if audio_saver.enabled:
                logger.info(
                    "🎵 AudioProcessor: Audio saver initialized as parallel consumer"
                )
                if save_split_audio:
                    logger.info(
                        "🔀 AudioProcessor: Split audio saving enabled for AudioSaver"
                    )
            else:
                logger.debug("🎵 AudioProcessor: Audio saver disabled")

            return audio_saver

        except Exception as e:
            logger.error(f"❌ AudioProcessor: Audio saver initialization failed: {e}")
            # Return disabled audio saver rather than failing the entire processor
            return AudioSaver(
                enabled=False,
                save_path="./debug_audio/",
                max_duration=30,
                audio_config=self.audio_config,
                save_split_audio=False,
            )

    def _initialize_silence_detector(self, system_config) -> SilenceDetector:
        """Initialize silence detector for auto-stop functionality."""
        try:
            silence_timeout = system_config.silence_timeout_seconds
            logger.info(
                f"🔇 AudioProcessor: Initializing SilenceDetector with {silence_timeout}s timeout"
            )

            # Create silence detector with auto-stop callback
            silence_detector = SilenceDetector(
                silence_timeout_seconds=silence_timeout,
                auto_stop_callback=self._on_silence_auto_stop,
            )

            if silence_detector.is_enabled():
                logger.info(
                    "🔇 AudioProcessor: Silence detection enabled as safety feature"
                )
            else:
                logger.info("🔇 AudioProcessor: Silence detection disabled")

            return silence_detector

        except Exception as e:
            logger.error(
                f"❌ AudioProcessor: Silence detector initialization failed: {e}"
            )
            # Return disabled detector rather than failing the entire processor
            return SilenceDetector(
                silence_timeout_seconds=0,  # Disabled
            )

    def _on_silence_auto_stop(self) -> None:
        """Handle automatic stop triggered by silence detection."""
        logger.warning(
            "🔇 AudioProcessor: Auto-stopping recording due to prolonged silence"
        )

        # We need to stop recording asynchronously since this is called from audio processing thread
        # Set a flag that will be checked by the main processing loop
        self._silence_auto_stop_requested = True

        # Notify the SessionManager about the silence-triggered stop
        if self.silence_auto_stop_callback:
            try:
                self.silence_auto_stop_callback()
            except Exception as e:
                logger.error(
                    f"❌ AudioProcessor: Error in silence auto-stop callback: {e}"
                )

    async def _distribute_audio_to_consumers(self, audio_chunk: bytes) -> None:
        """
        Distribute audio chunk to parallel consumers using fan-out pattern.

        Both transcription provider and audio saver receive the same raw audio
        simultaneously as independent terminal consumers. Also performs silence
        detection for auto-stop functionality.
        """
        tasks = []
        consumer_names = []

        # Consumer 1: Transcription Provider (time-sensitive, highest priority)
        if self.transcription_provider:
            tasks.append(self.transcription_provider.send_audio(audio_chunk))
            consumer_names.append("Transcription Provider")

        # Consumer 2: Audio Saver (non-time-sensitive, lower priority)
        if self.audio_saver and self.audio_saver.is_saving_active():
            tasks.append(self.audio_saver.save_audio_chunk(audio_chunk))
            consumer_names.append("Audio Saver")

        # Consumer 3: Silence Detector (safety feature, lowest priority)
        if self.silence_detector and self.silence_detector.is_enabled():
            # Analyze audio chunk for silence (non-blocking, synchronous)
            try:
                silence_timeout_exceeded = self.silence_detector.analyze_audio_chunk(
                    audio_chunk
                )
                if silence_timeout_exceeded:
                    logger.warning(
                        "🔇 AudioProcessor: Silence timeout exceeded, stopping recording"
                    )
                    # The silence detector will have already called our callback
                    # which sets the _silence_auto_stop_requested flag
            except Exception as e:
                logger.error(f"❌ AudioProcessor: Silence detection error: {e}")

            consumer_names.append("Silence Detector")

        # Log audio distribution for validation (periodically to avoid spam)
        if not hasattr(self, '_audio_distribution_count'):
            self._audio_distribution_count = 0

        self._audio_distribution_count += 1

        # Log every 100 chunks for validation
        if self._audio_distribution_count % 100 == 0:
            logger.info(
                f"🔀 AudioProcessor: Audio chunk #{self._audio_distribution_count} distributed to {len(tasks)} consumers"
            )
            logger.info(f"   🎯 Active consumers: {', '.join(consumer_names)}")
            logger.info(f"   📊 Chunk size: {len(audio_chunk)} bytes")

            # Validate single audio saving responsibility
            audio_saving_active = (
                self.audio_saver and self.audio_saver.is_saving_active()
            )
            if audio_saving_active:
                logger.info("   ✅ AudioSaver is the ONLY component saving audio files")
            else:
                logger.info("   🔇 No audio saving active")

        # Execute all consumers in parallel
        if tasks:
            try:
                await asyncio.gather(*tasks)
            except Exception as e:
                # Enhanced error logging for validation
                logger.error(f"❌ AudioProcessor: Error in audio consumer: {e}")
                logger.error(f"   🎯 Failed consumers: {consumer_names}")
                logger.error(
                    f"   📊 Chunk: #{self._audio_distribution_count}, {len(audio_chunk)} bytes"
                )

                # Re-raise if it was a transcription error (critical)
                if len(tasks) == 1 or not self.audio_saver:
                    raise

    async def initialize(self) -> None:
        """Verify providers are initialized and set up connection monitoring."""
        init_correlation_id = self.pipeline_monitor.record_stage_start(
            PipelineStage.INITIALIZATION,
            provider_count=2,
            transcription_provider=self.transcription_provider_name,
            capture_provider=self.capture_provider_name,
        )

        async with self.error_handler.handle_pipeline_operation(
            "provider_initialization", timeout=15.0, severity=ErrorSeverity.CRITICAL
        ):
            try:
                # Verify providers are already initialized
                if (
                    not self._providers_initialized
                    or not self.transcription_provider
                    or not self.capture_provider
                ):
                    raise PipelineError(
                        "Providers should already be initialized in constructor"
                    )

                logger.info("✅ AudioProcessor: Using pre-initialized providers")
                logger.info(
                    f"🏭 AudioProcessor: Transcription provider: {type(self.transcription_provider).__name__}"
                )
                logger.info(
                    f"🎤 AudioProcessor: Capture provider: {type(self.capture_provider).__name__}"
                )

                if hasattr(self.capture_provider, "_instance_id"):
                    logger.info(
                        f"🔧 AudioProcessor: Using capture provider instance {self.capture_provider._instance_id}"
                    )

                # Set up connection health monitoring for AWS Transcribe
                if (
                    hasattr(
                        self.transcription_provider, "set_connection_health_callback"
                    )
                    and self.connection_health_callback
                ):
                    self.transcription_provider.set_connection_health_callback(
                        self.connection_health_callback
                    )
                    logger.info(
                        "🔍 AudioProcessor: Connection health monitoring enabled"
                    )

                # Verify transcription callback is set for this session
                if self.transcription_callback:
                    logger.info(
                        "📱 AudioProcessor: Transcription callback is configured and ready"
                    )
                else:
                    logger.warning(
                        "⚠️ AudioProcessor: No transcription callback set - UI may not receive results"
                    )

                logger.info(
                    "✅ AudioProcessor: Provider initialization completed successfully"
                )
                self.pipeline_monitor.record_stage_complete(
                    init_correlation_id,
                    success=True,
                    transcription_provider_type=type(
                        self.transcription_provider
                    ).__name__,
                    capture_provider_type=type(self.capture_provider).__name__,
                )

            except Exception as e:
                logger.error(f"❌ AudioProcessor: Provider initialization failed: {e}")
                self.pipeline_monitor.record_stage_complete(
                    init_correlation_id,
                    success=False,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                self.pipeline_monitor.record_error(e, PipelineStage.INITIALIZATION)
                raise PipelineError(
                    f"Failed to initialize audio processor providers: {e}"
                ) from e

    def update_transcription_provider(
        self, provider_name: str, config: dict[str, Any] | None = None
    ) -> None:
        """Hot-swap transcription provider with fresh configuration.

        This method allows updating the transcription provider (e.g., with new language settings)
        without disrupting the entire AudioProcessor. Must be called when not recording.

        Args:
            provider_name: Name of the transcription provider ('aws', 'azure', etc.)
            config: Configuration dict for the new provider (uses fresh system config if None)

        Raises:
            RuntimeError: If called while recording is active
            ValueError: If provider creation fails
        """
        if self.is_running:
            logger.error(
                "🚫 Cannot update transcription provider while recording is active"
            )
            raise RuntimeError(
                "Cannot update transcription provider while recording is active"
            )

        try:
            logger.info(f"🔄 Updating transcription provider to: {provider_name}")

            # Get fresh configuration if not provided
            if config is None:
                fresh_config = get_config()
                if provider_name == "aws":
                    config = {
                        "region": fresh_config.aws_region,
                        "language_code": fresh_config.aws_language_code,  # Fresh language!
                        # Note: connection_strategy removed - now auto-detected based on device channels
                        "dual_fallback_enabled": fresh_config.aws_dual_fallback_enabled,
                        "channel_balance_threshold": fresh_config.aws_channel_balance_threshold,
                        # Note: Audio saving parameters removed - now handled at pipeline level
                    }
                elif provider_name == "azure":
                    config = {
                        "speech_key": fresh_config.azure_speech_key,
                        "region": fresh_config.azure_speech_region,
                        "language_code": fresh_config.azure_speech_language,  # Fresh language!
                        "endpoint": fresh_config.azure_speech_endpoint,
                        "enable_speaker_diarization": fresh_config.azure_enable_speaker_diarization,
                        "max_speakers": fresh_config.azure_max_speakers,
                        "timeout": fresh_config.azure_speech_timeout,
                    }
                else:
                    config = {}

            logger.info(
                f"🌐 New language code will be: {config.get('language_code', 'not specified')}"
            )

            # Clean up old provider if it exists
            if self.transcription_provider:
                try:
                    # Unregister from resource manager
                    self.resource_manager.unregister_resource("transcription_provider")
                    logger.debug("🧹 Cleaned up old transcription provider")
                except Exception as e:
                    logger.warning(
                        f"⚠️ Error cleaning up old transcription provider: {e}"
                    )

            # Create new provider with fresh config
            self.transcription_provider = (
                AudioProcessorFactory.create_transcription_provider(
                    provider_name, **config
                )
            )

            # Register new provider with resource manager
            self.resource_manager.register_resource(
                "transcription_provider",
                self.transcription_provider,
                cleanup_func=self._cleanup_transcription_provider,
                timeout=8.0,
            )

            # Update provider name for future reference
            self.transcription_provider_name = provider_name
            self.transcription_config = config

            logger.info(
                f"✅ Transcription provider updated successfully to {provider_name}"
            )
            logger.info(
                f"🌐 Active language code: {config.get('language_code', 'unknown')}"
            )

        except Exception as e:
            logger.error(f"❌ Failed to update transcription provider: {e}")
            raise ValueError(f"Failed to update transcription provider: {e}") from e

    async def start_recording(self, device_id: int | None = None) -> None:
        """Start real-time audio recording and transcription.

        Args:
            device_id: Optional specific audio device ID
        """
        if self.is_running:
            logger.warning("Audio processor is already running")
            return

        # Providers should already be initialized - verify this
        if (
            not self._providers_initialized
            or not self.transcription_provider
            or not self.capture_provider
        ):
            logger.debug(
                "🔄 AudioProcessor: Running initialize() to verify providers..."
            )
            await self.initialize()

        try:
            # Start new session
            self.current_meeting_id = (
                f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            self.session_transcripts.clear()
            self._silence_auto_stop_requested = False  # Reset silence auto-stop flag
            logger.info(
                f"🆔 AudioProcessor: Created meeting session: {self.current_meeting_id}"
            )

            # Start pipeline monitoring for this session
            self.pipeline_monitor.start_monitoring(self.current_meeting_id)

            # Create device-optimized audio config first
            system_config = get_config()
            optimized_audio_config = system_config.get_device_optimized_audio_config(
                device_id
            )

            logger.info("🎛️ AudioProcessor: Audio Configuration Summary:")
            logger.info(f"   🎚️ Device {device_id} optimized config:")
            logger.info(f"      📊 Sample Rate: {optimized_audio_config.sample_rate}Hz")
            logger.info(f"      🎛️ Channels: {optimized_audio_config.channels}")
            logger.info(f"      📋 Format: {optimized_audio_config.format}")
            logger.info(f"      🔢 Chunk Size: {optimized_audio_config.chunk_size}")

            # Compare with system base configuration
            logger.info(f"   📊 System base config comparison:")
            logger.info(
                f"      📊 Sample Rate: {self.audio_config.sample_rate}Hz {'✅' if self.audio_config.sample_rate == optimized_audio_config.sample_rate else '⚠️'}"
            )
            logger.info(
                f"      🎛️ Channels: {self.audio_config.channels} {'✅' if self.audio_config.channels == optimized_audio_config.channels else '⚠️'}"
            )
            logger.info(
                f"      📋 Format: {self.audio_config.format} {'✅' if self.audio_config.format == optimized_audio_config.format else '⚠️'}"
            )

            if (
                optimized_audio_config.sample_rate != self.audio_config.sample_rate
                or optimized_audio_config.channels != self.audio_config.channels
            ):
                logger.warning(
                    "⚠️ Device optimization changed audio parameters - AudioSaver will be updated"
                )

            # Log channel processing strategy info
            logger.info(
                "🔧 AudioProcessor: Automatic channel detection - 1ch→single connection, 2ch→dual connections, 3-4ch→dual connections, >4ch→error"
            )

            # Start transcription stream with error handling
            async with self.error_handler.handle_pipeline_operation(
                "transcription_start",
                timeout=10.0,
                severity=ErrorSeverity.HIGH,
                cleanup_callback=lambda: self._emergency_cleanup_transcription(),
            ):
                logger.debug("🎯 AudioProcessor: Starting transcription stream...")

                # Determine processed channel count based on automatic device channel detection
                capture_channels = optimized_audio_config.channels

                if capture_channels == 1:
                    # 1 channel → always mono (single AWS connection)
                    processed_channels = 1
                    logger.info(
                        "🎤 AudioProcessor: 1-channel device detected → using single AWS Transcribe connection"
                    )
                elif capture_channels == 2:
                    # 2 channels → automatic dual connections for optimal transcription
                    processed_channels = 2
                    logger.info(
                        "🎤 AudioProcessor: 2-channel device detected → using dual AWS Transcribe connections"
                    )
                elif capture_channels <= 4:
                    # 3-4 channels → will be processed to 2 channels (dual-channel)
                    processed_channels = 2
                    logger.info(
                        f"🎤 AudioProcessor: {capture_channels}-channel device detected → processing to dual connections"
                    )
                else:
                    # >4 channels → not supported, should have been caught earlier
                    raise ValueError(
                        f"Unsupported channel count: {capture_channels}. Maximum 4 channels supported."
                    )

                # Create transcription config with processed channel count
                from .interfaces import AudioConfig

                transcription_config = AudioConfig(
                    sample_rate=optimized_audio_config.sample_rate,
                    channels=processed_channels,  # Use processed channel count
                    chunk_size=optimized_audio_config.chunk_size,
                    format=optimized_audio_config.format,
                )

                logger.info(
                    f"🔧 AudioProcessor: Transcription config - capture: {capture_channels}ch → processed: {processed_channels}ch"
                )
                await self.transcription_provider.start_stream(transcription_config)

                # Configure silence detector with optimized audio configuration
                if self.silence_detector and self.silence_detector.is_enabled():
                    logger.info(
                        "🔇 AudioProcessor: Configuring silence detector for recording session"
                    )
                    self.silence_detector.configure_audio_format(
                        optimized_audio_config.format, optimized_audio_config.channels
                    )
                    self.silence_detector.reset_silence_tracking()
                    logger.info(
                        "🔇 AudioProcessor: Silence detection ready for safety monitoring"
                    )

                # Update audio saver configuration for device switching with proper stop-then-restart sequence
                logger.info("🔧 AudioProcessor: AudioSaver Device Switch Debug:")
                logger.info(f"   🎵 audio_saver exists: {self.audio_saver is not None}")
                logger.info(
                    f"   🎛️  optimized device channels: {optimized_audio_config.channels}"
                )
                logger.info(
                    f"   📊 optimized device sample rate: {optimized_audio_config.sample_rate}Hz"
                )

                if self.audio_saver:
                    logger.info(
                        f"   ✅ audio_saver.enabled: {self.audio_saver.enabled}"
                    )
                    logger.info(
                        f"   🔀 audio_saver.save_split_audio: {self.audio_saver.save_split_audio}"
                    )
                    logger.info(
                        f"   🎚️  current audio_saver.audio_config.channels: {self.audio_saver.audio_config.channels}"
                    )
                    logger.info(
                        f"   📊 current audio_saver.audio_config.sample_rate: {self.audio_saver.audio_config.sample_rate}Hz"
                    )
                    logger.info(
                        f"   🟢 current audio_saver.is_saving_active(): {self.audio_saver.is_saving_active()}"
                    )

                    # Create new device configuration
                    audio_saver_config = AudioConfig(
                        sample_rate=optimized_audio_config.sample_rate,
                        channels=optimized_audio_config.channels,  # Use actual device channels
                        chunk_size=optimized_audio_config.chunk_size,
                        format=optimized_audio_config.format,
                    )

                    # Detect configuration changes for proper device switching handling
                    old_channels = self.audio_saver.audio_config.channels
                    old_sample_rate = self.audio_saver.audio_config.sample_rate
                    new_channels = audio_saver_config.channels
                    new_sample_rate = audio_saver_config.sample_rate

                    config_changed = (
                        old_channels != new_channels
                        or old_sample_rate != new_sample_rate
                    )

                    logger.info(
                        f"🔄 AudioProcessor: Target AudioSaver config: {audio_saver_config.channels}ch, {audio_saver_config.sample_rate}Hz"
                    )

                    if config_changed:
                        logger.info(f"   📋 Device configuration change detected:")
                        logger.info(
                            f"      🎛️  Channels: {old_channels}ch → {new_channels}ch"
                        )
                        logger.info(
                            f"      📊 Sample Rate: {old_sample_rate}Hz → {new_sample_rate}Hz"
                        )
                        logger.info(
                            f"      🔀 Split audio will be: {'ENABLED' if self.audio_saver.save_split_audio and new_channels == 2 else 'DISABLED/MONO'}"
                        )

                        # CRITICAL: Stop AudioSaver BEFORE reconfiguration to avoid timing issue
                        if self.audio_saver.is_saving_active():
                            logger.info(
                                "🛑 AudioProcessor: AudioSaver is active - stopping for device configuration change..."
                            )
                            logger.info(
                                "   💡 This is necessary to prevent 'Cannot update configuration while actively saving' error"
                            )

                            try:
                                # Stop the active recording session
                                stop_stats = self.audio_saver.stop_saving()
                                logger.info(
                                    f"✅ AudioProcessor: AudioSaver stopped for reconfiguration - stats: {stop_stats}"
                                )

                                # Small delay to ensure complete stop and cleanup
                                await asyncio.sleep(0.1)

                                # Verify AudioSaver is actually stopped
                                if self.audio_saver.is_saving_active():
                                    logger.error(
                                        "❌ AudioProcessor: AudioSaver still active after stop - this should not happen!"
                                    )
                                    logger.error(
                                        "   🚨 Device switching may fail due to timing issue"
                                    )
                                else:
                                    logger.info(
                                        "✅ AudioProcessor: AudioSaver confirmed stopped - safe to reconfigure"
                                    )

                            except Exception as e:
                                logger.error(
                                    f"❌ AudioProcessor: Failed to stop AudioSaver for reconfiguration: {e}"
                                )
                                logger.error(
                                    "   💡 Continuing with reconfiguration attempt anyway..."
                                )
                        else:
                            logger.info(
                                "🔧 AudioProcessor: AudioSaver not currently active - safe to reconfigure directly"
                            )

                    else:
                        logger.info(
                            f"   📋 Device configuration unchanged: {new_channels}ch, {new_sample_rate}Hz"
                        )
                        logger.info(
                            "   🔧 No stop-restart needed - configuration update only"
                        )

                    # NOW update configuration - AudioSaver is guaranteed to be inactive for device changes
                    logger.info(
                        "🔄 AudioProcessor: Applying device configuration to AudioSaver..."
                    )
                    try:
                        self.audio_saver.update_audio_config(audio_saver_config)
                        logger.info(
                            "✅ AudioProcessor: AudioSaver configuration updated successfully"
                        )

                        # Verify the configuration was applied correctly
                        if (
                            self.audio_saver.audio_config.channels == new_channels
                            and self.audio_saver.audio_config.sample_rate
                            == new_sample_rate
                        ):
                            logger.info(
                                f"   ✅ Configuration verified: {self.audio_saver.audio_config.channels}ch, {self.audio_saver.audio_config.sample_rate}Hz"
                            )
                        else:
                            logger.error(f"   ❌ Configuration mismatch after update!")
                            logger.error(
                                f"      Expected: {new_channels}ch, {new_sample_rate}Hz"
                            )
                            logger.error(
                                f"      Actual: {self.audio_saver.audio_config.channels}ch, {self.audio_saver.audio_config.sample_rate}Hz"
                            )

                    except Exception as e:
                        logger.error(
                            f"❌ AudioProcessor: AudioSaver configuration update failed: {e}"
                        )
                        logger.error(
                            "   💡 AudioSaver may be in inconsistent state - audio saving may not work correctly"
                        )

                    # Start AudioSaver with new configuration if enabled
                    if self.audio_saver.enabled:
                        logger.info(
                            "🚀 AudioProcessor: Starting AudioSaver with new device configuration..."
                        )
                        try:
                            self.audio_saver.start_saving()

                            # Verify AudioSaver started correctly
                            if self.audio_saver.is_saving_active():
                                logger.info(
                                    "✅ AudioProcessor: AudioSaver started successfully as parallel consumer"
                                )

                                # Log expected file creation based on final configuration
                                if (
                                    self.audio_saver.save_split_audio
                                    and new_channels == 2
                                ):
                                    logger.info(
                                        "   📁 Expected files: 1 raw + 2 split channel files (3 total)"
                                    )
                                else:
                                    logger.info("   📁 Expected files: 1 raw file only")
                            else:
                                logger.error(
                                    "❌ AudioProcessor: AudioSaver failed to start - is_saving_active() returned False"
                                )

                        except Exception as e:
                            logger.error(
                                f"❌ AudioProcessor: Failed to start AudioSaver: {e}"
                            )
                            logger.error(
                                "   💡 No audio files will be saved for this recording session"
                            )
                    else:
                        logger.info(
                            "⚠️ AudioProcessor: AudioSaver is disabled - configuration updated but no saving will occur"
                        )
                        logger.info(
                            "   💡 AudioSaver disabled due to initialization error or configuration"
                        )
                else:
                    logger.info(
                        "⚠️ AudioProcessor: AudioSaver is None - no audio files will be saved"
                    )
                    logger.info(
                        "   💡 Check SAVE_RAW_AUDIO or SAVE_SPLIT_AUDIO environment variables"
                    )

            # Start audio capture with error handling
            async with self.error_handler.handle_pipeline_operation(
                "audio_capture_start",
                timeout=8.0,
                severity=ErrorSeverity.HIGH,
                cleanup_callback=lambda: self._emergency_cleanup_capture(),
            ):
                logger.debug("🎤 AudioProcessor: Starting audio capture...")

                # Validate provider state before starting
                if (
                    hasattr(self.capture_provider, "is_active")
                    and self.capture_provider.is_active()
                ):
                    logger.warning(
                        "⚠️ AudioProcessor: Capture provider already active, will be reset automatically"
                    )

                await self.capture_provider.start_capture(
                    optimized_audio_config, device_id
                )

            # Create managed tasks with proper lifecycle control
            logger.debug("🔄 AudioProcessor: Creating managed async tasks...")

            capture_task = self.resource_manager.create_task(
                "audio_capture",
                self._audio_capture_loop(),
                timeout=None,  # No timeout for main processing loop
                cleanup_on_cancel=self._cleanup_capture_on_cancel,
            )

            transcription_task = self.resource_manager.create_task(
                "transcription_processing",
                self._transcription_loop(),
                timeout=None,  # No timeout for main processing loop
                cleanup_on_cancel=self._cleanup_transcription_on_cancel,
            )

            # Store task references for compatibility
            self._capture_task = capture_task.task
            self._transcription_task = transcription_task.task

            self.is_running = True
            logger.info(
                f"✅ AudioProcessor: Started recording for meeting: {self.current_meeting_id}"
            )

            # Wait for tasks to complete (this keeps the function running)
            logger.debug(
                "⏳ AudioProcessor: Waiting for processing tasks to complete..."
            )
            try:
                await asyncio.gather(self._capture_task, self._transcription_task)
            except asyncio.CancelledError:
                logger.info("🛑 AudioProcessor: Processing tasks cancelled")
                raise
            except Exception as e:
                logger.error(f"❌ AudioProcessor: Error in processing tasks: {e}")
                raise

        except (PipelineError, PipelineTimeoutError) as e:
            logger.error(
                f"❌ AudioProcessor: Pipeline error during recording start: {e}"
            )
            await self.stop_recording()
            if self.error_callback:
                self.error_callback(e)
            raise
        except Exception as e:
            logger.error(
                f"❌ AudioProcessor: Unexpected error during recording start: {e}"
            )
            import traceback

            traceback.print_exc()
            await self.stop_recording()
            if self.error_callback:
                self.error_callback(PipelineError(f"Recording start failed: {e}", e))
            raise PipelineError(f"Failed to start recording: {e}") from e

    async def stop_recording(self) -> None:
        """Stop audio recording and transcription with improved error handling."""
        logger.info("🛑 AudioProcessor: stop_recording() called")
        logger.info(f"🛑 AudioProcessor: Current is_running state: {self.is_running}")

        if not self.is_running:
            logger.debug("🛑 AudioProcessor: Already stopped, nothing to do")
            return

        logger.info("🛑 AudioProcessor: Stopping audio recording...")
        self.is_running = False

        # Stop recording streams but keep providers alive for reuse
        try:
            # Cancel tasks first to stop the processing loops
            tasks_to_cancel = []
            if self._capture_task and not self._capture_task.done():
                tasks_to_cancel.append(self._capture_task)
            if self._transcription_task and not self._transcription_task.done():
                tasks_to_cancel.append(self._transcription_task)

            if tasks_to_cancel:
                logger.info(
                    f"🛑 AudioProcessor: Cancelling {len(tasks_to_cancel)} active tasks first..."
                )
                for task in tasks_to_cancel:
                    task.cancel()

            # Then stop provider streams to interrupt any blocking calls
            logger.info("🛑 AudioProcessor: Stopping provider streams...")

            # Stop capture stream first (more critical)
            if self.capture_provider:
                try:
                    logger.info("🛑 AudioProcessor: Stopping capture stream...")
                    logger.info(
                        f"🛑 AudioProcessor: Capture provider active state: {getattr(self.capture_provider, '_is_active', 'unknown')}"
                    )
                    await asyncio.wait_for(
                        self.capture_provider.stop_capture(), timeout=2.0
                    )
                    logger.info("✅ AudioProcessor: Capture stream stopped")
                except Exception as e:
                    logger.warning(
                        f"⚠️ AudioProcessor: Error stopping capture stream: {e}"
                    )

            # Stop transcription stream
            if self.transcription_provider:
                try:
                    logger.info("🛑 AudioProcessor: Stopping transcription stream...")
                    await asyncio.wait_for(
                        self.transcription_provider.stop_stream(), timeout=2.0
                    )
                    logger.info("✅ AudioProcessor: Transcription stream stopped")
                except Exception as e:
                    logger.warning(
                        f"⚠️ AudioProcessor: Error stopping transcription stream: {e}"
                    )

            # Stop audio saver
            if self.audio_saver and self.audio_saver.is_saving_active():
                try:
                    logger.info("🛑 AudioProcessor: Stopping audio saver...")
                    stats = self.audio_saver.stop_saving()
                    if stats:
                        logger.info(f"✅ AudioProcessor: Audio saver stopped - {stats}")
                    else:
                        logger.info("✅ AudioProcessor: Audio saver stopped")
                except Exception as e:
                    logger.warning(f"⚠️ AudioProcessor: Error stopping audio saver: {e}")

            # Wait for task cancellation with shorter timeout
            if tasks_to_cancel:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                        timeout=1.0,
                    )
                    logger.info("✅ AudioProcessor: Tasks cancelled successfully")
                except TimeoutError:
                    logger.warning(
                        "⚠️ AudioProcessor: Task cancellation timed out - continuing cleanup"
                    )

            # Clear task references after cleanup
            self._capture_task = None
            self._transcription_task = None

            logger.info(
                "✅ AudioProcessor: Audio recording stopped successfully (providers remain alive for reuse)"
            )

            # Stop pipeline monitoring
            self.pipeline_monitor.stop_monitoring()

        except Exception as e:
            logger.error(f"❌ AudioProcessor: Error during stop_recording cleanup: {e}")
            # Log error and resource manager status for debugging
            error_summary = self.error_handler.get_error_summary()
            resource_status = self.resource_manager.get_status()
            logger.error(f"📊 AudioProcessor: Error handler summary: {error_summary}")
            logger.error(
                f"📊 AudioProcessor: Resource manager status: {resource_status}"
            )
            raise PipelineError(f"Failed to stop recording properly: {e}") from e

    async def _audio_capture_loop(self) -> None:
        """Main audio capture loop."""
        try:
            chunk_count = 0
            logger.info("🔄 AudioProcessor: Starting audio capture loop...")

            async for audio_chunk in self.capture_provider.get_audio_stream():
                if not self.is_running:
                    logger.info(
                        "🛑 AudioProcessor: is_running=False, breaking capture loop"
                    )
                    break

                chunk_count += 1

                # Record audio chunk processing and distribute to parallel consumers
                processing_start = time.time()
                await self._distribute_audio_to_consumers(audio_chunk)
                processing_time_ms = (time.time() - processing_start) * 1000

                self.pipeline_monitor.record_audio_chunk_processed(
                    len(audio_chunk), processing_time_ms
                )

                # Check is_running after sending audio (in case stop was called)
                if not self.is_running:
                    logger.info(
                        "🛑 AudioProcessor: is_running=False after send_audio, breaking capture loop"
                    )
                    break

                # Check for silence-triggered auto-stop
                if self._silence_auto_stop_requested:
                    logger.warning(
                        "🔇 AudioProcessor: Silence auto-stop requested, breaking capture loop"
                    )
                    # The silence callback will have already been triggered to notify the SessionManager
                    # Just break the loop - the AudioProcessor stop will be handled normally
                    break

                # Log every 50 chunks to monitor flow
                if chunk_count % 50 == 0:
                    logger.info(
                        f"🔄 AudioProcessor: Processed {chunk_count} audio chunks through transcription pipeline"
                    )

        except asyncio.CancelledError:
            logger.info("🛑 AudioProcessor: Audio capture loop cancelled")
            raise
        except Exception as e:
            logger.error(f"❌ AudioProcessor: Audio capture loop error: {e}")
            if self.error_callback:
                self.error_callback(PipelineError(f"Audio capture loop failed: {e}", e))
        finally:
            logger.info(
                f"🔄 AudioProcessor: Audio capture loop stopped after processing {chunk_count} chunks"
            )

    async def _transcription_loop(self) -> None:
        """Main transcription processing loop."""
        try:
            transcription_count = 0
            logger.info("📝 AudioProcessor: Starting transcription processing loop...")

            async for result in self.transcription_provider.get_transcription():
                if not self.is_running:
                    logger.info(
                        "🛑 AudioProcessor: is_running=False, breaking transcription loop"
                    )
                    break

                transcription_count += 1

                # Record transcription processing
                processing_time_ms = (
                    100.0  # Placeholder since we don't have actual processing time
                )
                self.pipeline_monitor.record_transcription_processed(
                    result.text,
                    result.confidence,
                    processing_time_ms,
                    result.is_partial,
                )

                # Store transcript
                self.session_transcripts.append(result)

                # Callback to UI
                if self.transcription_callback:
                    logger.info(
                        f"📱 AudioProcessor: Sending transcription #{transcription_count} to UI: '{result.text}'"
                    )
                    self.transcription_callback(result)

                logger.info(
                    f"📝 AudioProcessor: Transcription #{transcription_count}: {result.speaker_id or 'Unknown'}: '{result.text}' (confidence: {result.confidence:.2f})"
                )

                # Check is_running after processing (in case stop was called)
                if not self.is_running:
                    logger.info(
                        "🛑 AudioProcessor: is_running=False after processing, breaking transcription loop"
                    )
                    break

        except asyncio.CancelledError:
            logger.info("🛑 AudioProcessor: Transcription loop cancelled")
            raise
        except Exception as e:
            logger.error(f"❌ AudioProcessor: Transcription loop error: {e}")
            if self.error_callback:
                self.error_callback(PipelineError(f"Transcription loop failed: {e}", e))
        finally:
            logger.info(
                f"📝 Transcription loop stopped after processing {transcription_count} transcriptions"
            )

    async def _cleanup_transcription_provider(self, provider) -> None:
        """Final cleanup function for transcription provider (app shutdown only)."""
        logger.info(
            f"🧹 AudioProcessor: Final cleanup of transcription provider - Type: {type(provider).__name__}"
        )
        try:
            await asyncio.wait_for(provider.stop_stream(), timeout=5.0)
            logger.info(
                "✅ AudioProcessor: Transcription provider final cleanup completed"
            )
        except Exception as e:
            logger.warning(
                f"⚠️ AudioProcessor: Error in transcription provider final cleanup: {e}"
            )

    async def _cleanup_capture_provider(self, provider) -> None:
        """Final cleanup function for capture provider (app shutdown only)."""
        logger.info(
            f"🧹 AudioProcessor: Final cleanup of capture provider - Type: {type(provider).__name__}"
        )

        # Log provider instance details
        if hasattr(provider, "_instance_id"):
            logger.info(
                f"🧹 AudioProcessor: Final cleanup of provider instance {provider._instance_id}"
            )

        try:
            await asyncio.wait_for(provider.stop_capture(), timeout=5.0)
            logger.info("✅ AudioProcessor: Capture provider final cleanup completed")
        except Exception as e:
            logger.warning(
                f"⚠️ AudioProcessor: Error in capture provider final cleanup: {e}"
            )

    def _cleanup_capture_on_cancel(self) -> None:
        """Cleanup function called when capture task is cancelled."""
        logger.info("🧹 AudioProcessor: Capture task cleanup on cancellation")
        # Any non-async cleanup can be done here
        # Async cleanup is handled by the resource manager

    def _cleanup_transcription_on_cancel(self) -> None:
        """Cleanup function called when transcription task is cancelled."""
        logger.info("🧹 AudioProcessor: Transcription task cleanup on cancellation")
        # Any non-async cleanup can be done here
        # Async cleanup is handled by the resource manager

    def _emergency_cleanup_transcription(self) -> None:
        """Emergency cleanup for transcription provider (non-async)."""
        logger.warning("🚨 AudioProcessor: Emergency transcription cleanup triggered")
        if hasattr(self.transcription_provider, "emergency_stop"):
            self.transcription_provider.emergency_stop()
        else:
            logger.warning(
                "⚠️ AudioProcessor: No emergency_stop method on transcription provider"
            )

    def _emergency_cleanup_capture(self) -> None:
        """Emergency cleanup for capture provider (non-async)."""
        logger.warning("🚨 AudioProcessor: Emergency capture cleanup triggered")
        if hasattr(self.capture_provider, "emergency_stop"):
            self.capture_provider.emergency_stop()
        else:
            logger.warning(
                "⚠️ AudioProcessor: No emergency_stop method on capture provider"
            )

    async def _cleanup_providers(self) -> None:
        """Clean up providers during initialization failure."""
        if self.transcription_provider:
            logger.info(
                "🧹 AudioProcessor: Cleaning up transcription provider after init failure"
            )
            try:
                await asyncio.wait_for(
                    self.transcription_provider.stop_stream(), timeout=2.0
                )
            except Exception as e:
                logger.warning(
                    f"⚠️ AudioProcessor: Error cleaning transcription provider: {e}"
                )
            finally:
                self.transcription_provider = None

        if self.capture_provider:
            logger.info(
                "🧹 AudioProcessor: Cleaning up capture provider after init failure"
            )
            try:
                await asyncio.wait_for(
                    self.capture_provider.stop_capture(), timeout=2.0
                )
            except Exception as e:
                logger.warning(
                    f"⚠️ AudioProcessor: Error cleaning capture provider: {e}"
                )
            finally:
                self.capture_provider = None

    def set_transcription_callback(
        self, callback: Callable[[TranscriptionResult], None]
    ) -> None:
        """Set callback function for new transcription results."""
        self.transcription_callback = callback

    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Set callback function for error handling."""
        self.error_callback = callback

    def set_connection_health_callback(
        self, callback: Callable[[bool, str], None]
    ) -> None:
        """Set callback for connection health notifications.

        Args:
            callback: Function to call with (is_healthy, message) when connection status changes
        """
        self.connection_health_callback = callback

        # If transcription provider exists and supports health callbacks, set it immediately
        if (
            hasattr(self.transcription_provider, "set_connection_health_callback")
            and self.transcription_provider
        ):
            self.transcription_provider.set_connection_health_callback(callback)
            logger.debug(
                "🔍 AudioProcessor: Connection health callback set on transcription provider"
            )

    def set_silence_auto_stop_callback(self, callback: Callable[[], None]) -> None:
        """Set callback for silence-triggered auto-stop notifications.

        Args:
            callback: Function to call when silence timeout is exceeded
        """
        self.silence_auto_stop_callback = callback
        logger.debug("🔇 AudioProcessor: Silence auto-stop callback set")

    def get_available_devices(self) -> dict[int, str]:
        """Get list of available audio input devices using existing provider."""
        # Providers should already be initialized - verify this
        if not self._providers_initialized or not self.capture_provider:
            raise RuntimeError(
                "Audio capture provider not initialized - this should not happen"
            )

        logger.info(
            f"🔧 AudioProcessor: Using existing provider instance {getattr(self.capture_provider, '_instance_id', 'unknown')} for device listing"
        )
        devices = self.capture_provider.list_audio_devices()
        logger.info(
            f"✅ AudioProcessor: Retrieved {len(devices)} devices from existing provider"
        )
        return devices

    def get_session_transcripts(self) -> list[TranscriptionResult]:
        """Get all transcripts from current session."""
        return self.session_transcripts.copy()

    def export_session(self) -> dict[str, Any]:
        """Export current session data."""
        return {
            "meeting_id": self.current_meeting_id,
            "start_time": datetime.now().isoformat(),
            "transcripts": [
                {
                    "text": t.text,
                    "speaker_id": t.speaker_id,
                    "confidence": t.confidence,
                    "start_time": t.start_time,
                    "end_time": t.end_time,
                    "is_partial": t.is_partial,
                }
                for t in self.session_transcripts
            ],
            "error_summary": self.error_handler.get_error_summary(),
            "resource_summary": self.resource_manager.get_status(),
            "monitoring_metrics": self.pipeline_monitor.get_current_metrics(),
            "pipeline_health": self.pipeline_monitor.get_health_status(),
            "silence_stats": (
                self.silence_detector.get_silence_stats()
                if self.silence_detector
                else None
            ),
        }

    def get_pipeline_health(self) -> dict[str, Any]:
        """Get pipeline health status and error information."""
        return {
            "is_running": self.is_running,
            "has_providers": {
                "transcription": self.transcription_provider is not None,
                "capture": self.capture_provider is not None,
            },
            "has_tasks": {
                "capture_task": self._capture_task is not None
                and not self._capture_task.done(),
                "transcription_task": self._transcription_task is not None
                and not self._transcription_task.done(),
            },
            "session_info": {
                "meeting_id": self.current_meeting_id,
                "transcript_count": len(self.session_transcripts),
            },
            "error_handler": self.error_handler.get_error_summary(),
            "resource_manager": self.resource_manager.get_status(),
            "pipeline_monitor": self.pipeline_monitor.get_health_status(),
            "monitoring_metrics": self.pipeline_monitor.get_current_metrics(),
            "silence_detector": {
                "enabled": (
                    self.silence_detector.is_enabled()
                    if self.silence_detector
                    else False
                ),
                "stats": (
                    self.silence_detector.get_silence_stats()
                    if self.silence_detector
                    else None
                ),
            },
        }
