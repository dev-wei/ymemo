"""AWS Transcribe Streaming provider implementation."""

import asyncio
import logging
import os
import struct
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from typing import Any

import boto3
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

from ...core.interfaces import AudioConfig, TranscriptionProvider, TranscriptionResult
from ...utils.exceptions import AWSTranscribeError
from ..channel_splitter import AudioChannelSplitter
from ..dual_connection_error_handler import DualConnectionErrorHandler
from ..result_merger import DualChannelResultMerger

logger = logging.getLogger(__name__)


class AWSTranscribeHandler(TranscriptResultStreamHandler):
    """Handler for AWS Transcribe streaming events."""

    def __init__(
        self,
        transcript_result_stream,
        result_queue: asyncio.Queue,
        parent_provider=None,
    ):
        super().__init__(transcript_result_stream)
        self.result_queue = result_queue
        self.parent_provider = (
            parent_provider  # Reference to AWSTranscribeProvider for health tracking
        )

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        """Handle incoming transcript events using AWS handler pattern."""
        logger.debug("📥 AWS Handler: Received transcript event from AWS Transcribe")

        # Enhanced event analysis for debugging
        self._analyze_transcript_event(transcript_event)

        results = transcript_event.transcript.results
        logger.debug(
            f"📋 AWS Handler: Processing {len(results)} results from transcript event"
        )

        # If no results, log detailed event information for debugging
        if len(results) == 0:
            self._log_empty_result_analysis(transcript_event)

        for result in results:
            if not result.alternatives:
                logger.debug("⚠️ AWS Handler: No alternatives in result, skipping")
                continue

            alternative = result.alternatives[0]
            text = alternative.transcript if hasattr(alternative, "transcript") else ""

            if not text.strip():
                logger.debug("⚠️ AWS Handler: Empty text result, skipping")
                continue

            # Extract speaker information using AWS dual-channel pattern
            speaker_id = None

            # Method 1: Check for channel_labels (AWS dual-channel standard)
            if hasattr(result, "channel_labels") and result.channel_labels:
                logger.debug("🎙️ AWS Handler: Found channel_labels in result")
                for channel in result.channel_labels.channels:
                    if hasattr(channel, "channel_label"):
                        # Map AWS channel labels to our speaker naming
                        if channel.channel_label == "0":
                            speaker_id = "Speaker A"  # Ch1+2 from AudioChannelProcessor
                        elif channel.channel_label == "1":
                            speaker_id = "Speaker B"  # Ch3+4 from AudioChannelProcessor
                        else:
                            speaker_id = f"Speaker-{channel.channel_label}"
                        logger.info(
                            f"🎙️ AWS Handler: Channel {channel.channel_label} → {speaker_id}"
                        )
                        break

            # Method 2: Fallback to channel_id (alternative AWS approach)
            elif hasattr(result, "channel_id") and result.channel_id is not None:
                if result.channel_id == "ch_0":
                    speaker_id = "Speaker A"
                elif result.channel_id == "ch_1":
                    speaker_id = "Speaker B"
                else:
                    speaker_id = f"Speaker-{result.channel_id}"
                logger.info(
                    f"🎙️ AWS Handler: Channel ID {result.channel_id} → {speaker_id}"
                )

            # Method 3: Fallback to item-level speaker labels (speaker diarization)
            elif hasattr(alternative, "items") and alternative.items:
                for item in alternative.items:
                    if hasattr(item, "speaker") and item.speaker:
                        speaker_id = f"Speaker-{item.speaker}"
                        logger.debug(
                            f"🎙️ AWS Handler: Item speaker {item.speaker} → {speaker_id}"
                        )
                        break

            # Generate result ID for partial result tracking
            result_id = getattr(result, "result_id", str(uuid.uuid4()))
            is_partial = result.is_partial if hasattr(result, "is_partial") else False
            confidence = getattr(alternative, "confidence", 0.0)

            logger.info(
                f"💬 AWS Handler: '{text}' (partial: {is_partial}, confidence: {confidence:.2f}, speaker: {speaker_id})"
            )

            transcription_result = TranscriptionResult(
                text=text,
                speaker_id=speaker_id,
                confidence=confidence,
                start_time=getattr(result, "start_time", 0.0),
                end_time=getattr(result, "end_time", 0.0),
                is_partial=is_partial,
                result_id=result_id,
                utterance_id=result_id,  # Use result_id as utterance_id for simplicity
                sequence_number=1,
            )

            # Put result in queue for main processor
            if self.result_queue:
                await self.result_queue.put(transcription_result)
                logger.debug(f"✅ AWS Handler: Added result to queue: '{text}'")

                # Update parent provider's connection health tracking
                if self.parent_provider:
                    self.parent_provider.last_result_time = time.time()
            else:
                logger.error("❌ AWS Handler: No result queue available")

    def _analyze_transcript_event(self, transcript_event: TranscriptEvent):
        """Enhanced transcript event analysis with dual-channel focus."""
        try:
            # Check event structure
            has_transcript = hasattr(transcript_event, "transcript")
            has_results = has_transcript and hasattr(
                transcript_event.transcript, "results"
            )
            result_count = (
                len(transcript_event.transcript.results) if has_results else 0
            )

            # Track event frequency (every 25 events for more detailed logging)
            if not hasattr(self, "_event_count"):
                self._event_count = 0
                self._last_result_time = time.time()
                self._last_non_empty_result_time = time.time()

            self._event_count += 1
            current_time = time.time()

            # Update timing if we got results
            if result_count > 0:
                self._last_result_time = current_time
                self._last_non_empty_result_time = current_time

            # Log every 25 events or if we get results after a period of empty results
            should_log_analysis = (self._event_count % 25 == 0) or (
                result_count > 0
                and current_time - self._last_non_empty_result_time > 5.0
            )

            if should_log_analysis or result_count > 0:
                logger.info(f"📊 AWS Event Analysis (#{self._event_count}):")
                logger.info(
                    f"   📅 Time since last result: {current_time - self._last_result_time:.1f}s"
                )
                logger.info(
                    f"   📅 Time since non-empty result: {current_time - self._last_non_empty_result_time:.1f}s"
                )
                logger.info(
                    f"   📋 Event structure: transcript={has_transcript}, results={has_results}"
                )
                logger.info(f"   📈 Result count: {result_count}")

                # Detailed transcript structure analysis
                if has_transcript:
                    transcript = transcript_event.transcript
                    logger.info(f"   🔍 Transcript object: {type(transcript).__name__}")

                    if has_results and result_count > 0:
                        # Detailed result analysis for dual-channel debugging
                        self._log_detailed_results(transcript_event.transcript.results)
                    elif has_results:
                        logger.info("   📋 Results array exists but is empty")
                        # Check for other transcript properties when results are empty
                        self._log_empty_transcript_details(transcript)

        except Exception as e:
            logger.warning(f"⚠️ AWS Event analysis error: {e}")

    def _log_detailed_results(self, results):
        """Log detailed information about AWS transcript results."""
        try:
            for i, result in enumerate(results[:3]):  # Log first 3 results max
                logger.info(f"   🎯 Result #{i}: type={type(result).__name__}")

                # Basic result properties
                if hasattr(result, "is_partial"):
                    logger.info(f"     📝 Partial: {result.is_partial}")
                if hasattr(result, "result_id"):
                    logger.info(f"     🆔 Result ID: {result.result_id}")

                # Channel identification information (key for dual-channel)
                self._log_channel_identification_details(result, i)

                # Alternative analysis
                if hasattr(result, "alternatives") and result.alternatives:
                    alt = result.alternatives[0]
                    transcript_text = getattr(alt, "transcript", "")
                    confidence = getattr(alt, "confidence", 0.0)
                    logger.info(
                        f"     💬 Text: '{transcript_text}' (conf: {confidence:.3f})"
                    )

                    # Item-level analysis for speaker info
                    if hasattr(alt, "items") and alt.items:
                        logger.info(f"     📑 Items: {len(alt.items)} items")
                        # Log first few items
                        for j, item in enumerate(alt.items[:2]):
                            item_info = f"Item {j}: "
                            if hasattr(item, "content"):
                                item_info += f"'{item.content}' "
                            if hasattr(item, "speaker"):
                                item_info += f"(speaker: {item.speaker}) "
                            if hasattr(item, "confidence"):
                                item_info += f"(conf: {item.confidence:.3f})"
                            logger.info(f"       📄 {item_info}")
                else:
                    logger.warning(f"     ⚠️ No alternatives in result #{i}")

        except Exception as e:
            logger.warning(f"⚠️ Detailed result logging error: {e}")

    def _log_channel_identification_details(self, result, result_index: int):
        """Log detailed channel identification information from AWS result."""
        try:
            # Method 1: Check for channel_labels (primary dual-channel method)
            if hasattr(result, "channel_labels") and result.channel_labels:
                logger.info("     🎙️ Channel Labels Found!")
                if hasattr(result.channel_labels, "channels"):
                    for j, channel in enumerate(result.channel_labels.channels):
                        channel_info = f"Channel {j}: "
                        if hasattr(channel, "channel_label"):
                            channel_info += f"label={channel.channel_label} "
                        if hasattr(channel, "items") and channel.items:
                            channel_info += f"({len(channel.items)} items) "
                            # Show first item content if available
                            first_item = channel.items[0]
                            if hasattr(first_item, "content"):
                                channel_info += f"'{first_item.content}'"
                        logger.info(f"       🎚️  {channel_info}")

            # Method 2: Check for channel_id (alternative method)
            elif hasattr(result, "channel_id") and result.channel_id is not None:
                logger.info(f"     🎙️ Channel ID: {result.channel_id}")

            # Method 3: Check for other channel-related attributes
            else:
                # Look for any channel-related attributes
                channel_attrs = [
                    attr
                    for attr in dir(result)
                    if "channel" in attr.lower() and not attr.startswith("_")
                ]
                if channel_attrs:
                    logger.info(f"     🔍 Channel-related attributes: {channel_attrs}")
                    for attr in channel_attrs:
                        try:
                            value = getattr(result, attr)
                            logger.info(f"       📄 {attr}: {value}")
                        except Exception:
                            pass
                else:
                    logger.info(
                        f"     ⚠️ No channel identification found in result #{result_index}"
                    )

        except Exception as e:
            logger.warning(f"⚠️ Channel identification logging error: {e}")

    def _log_empty_transcript_details(self, transcript):
        """Log details when transcript exists but has no results."""
        try:
            # Look for any properties that might explain why results are empty
            all_attrs = [attr for attr in dir(transcript) if not attr.startswith("_")]
            logger.info(f"   🔧 Available transcript attributes: {all_attrs}")

            # Check specific attributes that might give clues
            interesting_attrs = [
                "status",
                "error",
                "message",
                "results",
                "partial_results",
            ]
            for attr in interesting_attrs:
                if hasattr(transcript, attr):
                    try:
                        value = getattr(transcript, attr)
                        logger.info(f"     📄 {attr}: {value}")
                    except Exception as e:
                        logger.info(f"     📄 {attr}: <error accessing: {e}>")

        except Exception as e:
            logger.warning(f"⚠️ Empty transcript logging error: {e}")

    def _log_empty_result_analysis(self, transcript_event: TranscriptEvent):
        """Log detailed analysis when AWS returns empty results."""
        try:
            # Increment empty result counter
            if not hasattr(self, "_empty_result_count"):
                self._empty_result_count = 0
                self._first_empty_result_time = time.time()

            self._empty_result_count += 1
            current_time = time.time()
            time_since_first_empty = current_time - self._first_empty_result_time

            # Log every 100 empty results or after significant time periods
            should_log = (
                (self._empty_result_count % 100 == 0)
                or (self._empty_result_count <= 10)
                or (
                    time_since_first_empty > 30.0 and self._empty_result_count % 25 == 0
                )
            )

            if should_log:
                logger.info(
                    f"🔍 AWS Empty Result Analysis (#{self._empty_result_count}):"
                )
                logger.info(
                    f"   ⏱️  Duration: {time_since_first_empty:.1f}s of empty results"
                )
                logger.info(
                    f"   📊 Rate: {self._empty_result_count / time_since_first_empty:.1f} empty results/second"
                )

                # Check transcript event structure when empty
                if hasattr(transcript_event, "transcript"):
                    transcript = transcript_event.transcript
                    logger.info("   📋 Transcript exists but results empty")

                    # Check for any other properties in the transcript
                    if hasattr(transcript, "__dict__"):
                        all_attrs = {
                            k: v
                            for k, v in transcript.__dict__.items()
                            if not k.startswith("_")
                        }
                        if all_attrs:
                            logger.info(f"   🔧 Transcript properties: {all_attrs}")

                    # Check if there are any hidden attributes that might indicate status
                    transcript_type_attrs = [
                        attr
                        for attr in dir(transcript)
                        if not attr.startswith("_")
                        and not callable(getattr(transcript, attr))
                    ]
                    if transcript_type_attrs:
                        logger.info(
                            f"   📝 Available attributes: {transcript_type_attrs}"
                        )

                        for attr in transcript_type_attrs[
                            :5
                        ]:  # Check first 5 non-method attributes
                            try:
                                value = getattr(transcript, attr)
                                logger.info(
                                    f"      - {attr}: {value} ({type(value).__name__})"
                                )
                            except Exception as e:
                                logger.info(f"      - {attr}: <error accessing: {e}>")
                else:
                    logger.warning(
                        "   ❌ Transcript event has no transcript attribute!"
                    )

                # Critical warning if too many consecutive empty results
                if self._empty_result_count > 200:
                    logger.warning(
                        f"⚠️ AWS Handler: {self._empty_result_count} consecutive empty results! "
                        f"This suggests a serious issue with audio processing or AWS configuration."
                    )
                    logger.warning("   💡 Possible causes:")
                    logger.warning("      - Audio is completely silent")
                    logger.warning("      - Audio format is corrupted/invalid")
                    logger.warning(
                        "      - AWS dual-channel configuration is incorrect"
                    )
                    logger.warning("      - Network/connection issues with AWS")

        except Exception as e:
            logger.warning(f"⚠️ Empty result analysis error: {e}")


class AWSTranscribeProvider(TranscriptionProvider):
    """
    AWS Transcribe Streaming transcription provider.

    This provider uses Amazon Transcribe Streaming API for real-time speech-to-text
    conversion with support for partial results and speaker identification.
    """

    def __init__(
        self,
        region: str = "us-east-1",
        language_code: str = "en-US",
        profile_name: str | None = None,
        connection_strategy: str = "auto",
        dual_fallback_enabled: bool = True,
        channel_balance_threshold: float = 0.3,
        dual_connection_test_mode: str = "full",
    ):
        """
        Initialize AWS Transcribe provider with intelligent connection strategy.

        Args:
            region: AWS region for Transcribe service (default: 'us-east-1')
            language_code: Language code for transcription (default: 'en-US')
            profile_name: AWS profile name for authentication (default: None, uses default)
            connection_strategy: Connection strategy - 'auto', 'single', 'dual' (default: 'auto')
            dual_fallback_enabled: Enable fallback to dual connections (default: True)
            channel_balance_threshold: Threshold for channel imbalance detection (default: 0.3)

        Raises:
            ValueError: If parameters are invalid
            AWSTranscribeError: If AWS configuration is invalid
        """
        # Validate required parameters
        if not region or not isinstance(region, str):
            raise ValueError("AWS region must be a non-empty string")
        if not language_code or not isinstance(language_code, str):
            raise ValueError("Language code must be a non-empty string")
        if profile_name is not None and not isinstance(profile_name, str):
            raise ValueError("Profile name must be a string or None")

        # Validate connection strategy parameters
        valid_strategies = ["auto", "single", "dual"]
        if connection_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid connection_strategy '{connection_strategy}'. Valid options: {valid_strategies}"
            )
        if not isinstance(dual_fallback_enabled, bool):
            raise ValueError("dual_fallback_enabled must be a boolean")
        if not isinstance(channel_balance_threshold, int | float) or not (
            0.0 <= channel_balance_threshold <= 1.0
        ):
            raise ValueError(
                "channel_balance_threshold must be a number between 0.0 and 1.0"
            )

        # Validate dual connection test mode
        valid_test_modes = ["left_only", "right_only", "full"]
        if dual_connection_test_mode not in valid_test_modes:
            raise ValueError(
                f"Invalid dual_connection_test_mode '{dual_connection_test_mode}'. Valid options: {valid_test_modes}"
            )

        # Store configuration
        self.region = region.strip()
        self.language_code = language_code.strip()
        self.profile_name = profile_name or os.getenv("AWS_PROFILE")

        # Store connection strategy configuration
        self.connection_strategy = connection_strategy
        self.dual_fallback_enabled = dual_fallback_enabled
        self.channel_balance_threshold = channel_balance_threshold
        self.dual_connection_test_mode = dual_connection_test_mode
        self._connection_mode = (
            None  # Will be set to 'single_connection' or 'dual_connection'
        )

        # Initialize state
        self.client = None
        self.stream = None
        self.handler = None  # AWS Transcribe handler instance
        self.result_queue = None  # Will be created fresh for each session
        self._streaming_task = None
        self._current_event_loop = None  # Track current event loop

        logger.info(
            f"🏗️ AWS Transcribe: Initialized provider with region={self.region}, language={self.language_code}"
        )
        logger.info(
            f"🔧 AWS Transcribe: Connection strategy={self.connection_strategy}, dual_fallback={self.dual_fallback_enabled}, balance_threshold={self.channel_balance_threshold}"
        )
        logger.info(
            f"🧪 AWS Transcribe: Dual connection test mode={self.dual_connection_test_mode}"
        )

        # Validate AWS configuration early (skip in test environment)
        # Use environment-first approach for maximum CI compatibility
        skip_aws_validation = (
            os.environ.get("SKIP_AWS_VALIDATION", "").lower() == "true"
            or os.environ.get("MOCK_SERVICES", "").lower() == "true"
            or os.environ.get("CI") is not None
            or os.environ.get("TESTING", "").lower() == "true"
            or os.environ.get("PYTEST_RUNNING", "").lower() == "true"
            or os.environ.get("PYTEST_CURRENT_TEST") is not None
        )

        if not skip_aws_validation:
            try:
                self._validate_aws_configuration()
            except Exception as e:
                logger.error(f"❌ AWS Transcribe: Configuration validation failed: {e}")
                raise AWSTranscribeError(f"AWS configuration invalid: {e}") from e
        else:
            logger.debug(
                "🔧 AWS Transcribe: Skipping AWS configuration validation (test environment)"
            )

        # Track utterances for proper partial result handling
        self.active_utterances: dict[str, int] = {}  # result_id -> sequence_number
        self.result_to_utterance: dict[str, str] = {}  # result_id -> utterance_id
        self.utterance_counter = 0

        # Connection health monitoring
        self.last_result_time = 0.0
        self.last_audio_sent_time = 0.0
        self.connection_timeout = 30.0  # 30 seconds without results = disconnected
        self.is_connected = False
        self.connection_health_callback: Callable[[bool, str], None] | None = None
        self.retry_count = 0
        self.max_retries = 3
        self.retry_delay = 1.0  # Start with 1 second delay
        self.max_retry_delay = 60.0  # Cap at 60 seconds
        self._health_check_task = None

        # Channel configuration for AWS Transcribe adaptive channel support
        self.enable_channel_identification = (
            True  # Enable AWS channel ID feature when applicable
        )
        self.required_channels = (
            1  # Default to mono, but can handle dual-channel from 3-4ch devices
        )

        # Audio quality monitoring
        self._audio_chunk_count = 0
        self._total_audio_samples_analyzed = 0
        self._silence_chunks = 0
        self._audio_level_sum = 0.0

        # Dual connection state (initialized when needed)
        self._dual_connection_components = (
            None  # Will store dual connection components when activated
        )

    def _validate_aws_configuration(self) -> None:
        """
        Validate AWS configuration and credentials.

        Raises:
            AWSTranscribeError: If configuration is invalid
        """
        try:
            # Test AWS credentials and region by creating a session
            session = boto3.Session(
                profile_name=self.profile_name, region_name=self.region
            )

            # Verify credentials are available
            credentials = session.get_credentials()
            if not credentials:
                raise AWSTranscribeError(
                    "AWS credentials not found. Please configure AWS credentials."
                )

            # Test that the region is valid by attempting to create a client
            session.client("transcribe", region_name=self.region)

            logger.debug(
                f"✅ AWS Transcribe: Configuration validated for region {self.region}"
            )

        except Exception as e:
            if isinstance(e, AWSTranscribeError):
                raise
            raise AWSTranscribeError(f"AWS configuration validation failed: {e}") from e

    def _determine_connection_strategy(self, audio_config) -> str:
        """
        Determine the optimal connection strategy based on configuration and audio setup.

        Args:
            audio_config: AudioConfig object with channel and device information

        Returns:
            Connection mode: 'single_connection' or 'dual_connection'
        """
        logger.info(
            f"🔍 AWS Connection Strategy: Analyzing audio config - channels={audio_config.channels}, strategy={self.connection_strategy}"
        )

        # If explicitly set to single or dual, use that
        if self.connection_strategy == "single":
            logger.info("🔧 AWS Connection Strategy: Forced single connection mode")
            return "single_connection"
        if self.connection_strategy == "dual":
            if audio_config.channels < 2:
                logger.warning(
                    "⚠️ AWS Connection Strategy: Dual mode requested but audio is mono - falling back to single"
                )
                return "single_connection"
            logger.info("🔧 AWS Connection Strategy: Forced dual connection mode")
            return "dual_connection"

        # Auto mode - intelligent detection
        logger.info(
            "🤖 AWS Connection Strategy: Auto mode - analyzing audio characteristics"
        )

        # Single channel always uses single connection
        if audio_config.channels == 1:
            logger.info(
                "🔧 AWS Connection Strategy: Single channel detected → single connection"
            )
            return "single_connection"

        # For stereo (2 channels), default to single connection with AWS dual-channel support
        # This will be the primary mode that uses AWS's built-in channel identification
        if audio_config.channels == 2:
            logger.info(
                "🔧 AWS Connection Strategy: Stereo detected → single connection with AWS dual-channel support"
            )
            logger.info(
                "🔧 AWS Connection Strategy: Dual connection available as fallback if channel imbalance detected"
            )
            return "single_connection"

        # More than 2 channels - not currently supported
        logger.error(
            f"❌ AWS Connection Strategy: Unsupported channel count: {audio_config.channels}"
        )
        raise ValueError(
            f"Audio configuration with {audio_config.channels} channels is not supported. Use 1-2 channels."
        )

    def _initialize_dual_connection_components(self, audio_config) -> None:
        """
        Initialize dual connection components when dual mode is activated.

        Args:
            audio_config: AudioConfig object with audio format information
        """
        # Validate that input is actually stereo before initializing dual connection components
        if audio_config.channels != 2:
            error_msg = f"Dual connection components require stereo input (2 channels), got {audio_config.channels} channels"
            logger.error(f"❌ AWS Dual Connection: {error_msg}")
            raise ValueError(error_msg)

        if self._dual_connection_components is not None:
            logger.debug("🔧 AWS Dual Connection: Components already initialized")
            return

        logger.info("🏗️ AWS Dual Connection: Initializing dual connection components...")
        logger.info(
            f"✅ AWS Dual Connection: Validated stereo input - {audio_config.channels} channels"
        )

        # Create channel splitter for stereo audio processing (audio saving now handled at pipeline level)
        enable_saving = False  # Audio saving moved to AudioSaver component
        save_path = "./debug_audio/"  # Legacy path for backwards compatibility
        save_duration = 30  # Legacy duration for backwards compatibility

        logger.info(
            f"🔧 AWS Dual Connection: Channel splitter config - enable_split_saving={enable_saving}"
        )
        logger.info(
            f"🔧 AWS Dual Connection: Audio saving disabled (handled by AudioSaver component at pipeline level)"
        )

        channel_splitter = AudioChannelSplitter(
            audio_format=audio_config.format,
            enable_audio_saving=enable_saving,
            audio_save_path=save_path,
            sample_rate=audio_config.sample_rate,
            save_duration=save_duration,
        )

        # Create result merger for synchronizing dual stream results
        from ..result_merger import MergeStrategy

        result_merger = DualChannelResultMerger(
            merge_strategy=MergeStrategy.TIMESTAMP_ORDER,
            buffer_window=0.1,  # 100ms window for result synchronization
            max_buffer_size=100,
            confidence_threshold=0.0,
            priority_channel="left",  # Prefer left channel (Speaker A) for conflicts
        )

        # Create enhanced error handler for dual connections
        from ..dual_connection_error_handler import FallbackStrategy

        error_handler = DualConnectionErrorHandler(
            fallback_strategy=FallbackStrategy.MONO_FALLBACK,
            health_check_interval=5.0,
            failure_threshold=3,
            recovery_timeout=30.0,
            priority_channel="left",  # Prefer left channel (Speaker A) for fallback
        )

        # Store components in a container
        self._dual_connection_components = {
            "channel_splitter": channel_splitter,
            "result_merger": result_merger,
            "error_handler": error_handler,
            "left_provider": None,  # Will be created when stream starts
            "right_provider": None,  # Will be created when stream starts
            "left_queue": None,  # Will be created when stream starts
            "right_queue": None,  # Will be created when stream starts
        }

        logger.info("✅ AWS Dual Connection: Components initialized successfully")

    def set_connection_health_callback(
        self, callback: Callable[[bool, str], None]
    ) -> None:
        """Set callback for connection health notifications.

        Args:
            callback: Function to call with (is_healthy, message) when connection status changes
        """
        self.connection_health_callback = callback

    async def _monitor_connection_health(self) -> None:
        """Monitor connection health and detect timeouts."""
        try:
            logger.info("🔍 AWS Transcribe: Starting connection health monitor...")

            while (
                self.stream and self._streaming_task and not self._streaming_task.done()
            ):
                current_time = time.time()

                # Check if we've been sending audio but not receiving results
                if self.last_audio_sent_time > 0 and self.last_result_time > 0:
                    time_since_last_result = current_time - self.last_result_time
                    time_since_last_audio = current_time - self.last_audio_sent_time

                    # If we've sent audio recently but haven't received results, check timeout
                    if (
                        time_since_last_audio < 5.0
                        and time_since_last_result > self.connection_timeout
                    ):
                        if self.is_connected:
                            logger.warning(
                                f"⚠️ AWS Transcribe: Connection timeout detected - no results for {time_since_last_result:.1f}s"
                            )
                            self.is_connected = False
                            if self.connection_health_callback:
                                self.connection_health_callback(
                                    False,
                                    f"No transcription results for {time_since_last_result:.0f}s",
                                )
                    elif (
                        time_since_last_result < self.connection_timeout
                        and not self.is_connected
                    ):
                        # Connection recovered
                        logger.info(
                            "✅ AWS Transcribe: Connection recovered - receiving results again"
                        )
                        self.is_connected = True
                        self.retry_count = (
                            0  # Reset retry count on successful connection
                        )
                        if self.connection_health_callback:
                            self.connection_health_callback(
                                True, "Connection recovered"
                            )

                # Sleep for 5 seconds between health checks
                await asyncio.sleep(5.0)

        except asyncio.CancelledError:
            logger.info("🛑 AWS Transcribe: Connection health monitor cancelled")
            raise
        except Exception as e:
            logger.error(f"❌ AWS Transcribe: Error in connection health monitor: {e}")

    async def _calculate_retry_delay(self) -> float:
        """Calculate retry delay with exponential backoff.

        Returns:
            Delay in seconds for next retry attempt
        """
        delay = self.retry_delay * (2**self.retry_count)
        return min(delay, self.max_retry_delay)

    def _analyze_audio_content(self, audio_chunk: bytes) -> dict[str, any]:
        """Analyze audio content with detailed dual-channel analysis.

        Args:
            audio_chunk: Raw audio data bytes (assumed to be int16 PCM)

        Returns:
            Dict containing comprehensive analysis results for dual-channel audio
        """
        try:
            # Assume int16 PCM format (2 bytes per sample)
            sample_count = len(audio_chunk) // 2
            if sample_count == 0:
                return {"error": "Empty audio chunk"}

            # Unpack audio samples as signed 16-bit integers
            samples = struct.unpack(f"<{sample_count}h", audio_chunk)

            # Calculate basic statistics
            max_amplitude = max(abs(s) for s in samples)
            avg_amplitude = sum(abs(s) for s in samples) / sample_count

            # Detect silence (very low amplitude)
            silence_threshold = 100  # Adjust based on testing
            is_silent = max_amplitude < silence_threshold

            # Enhanced dual-channel analysis for Source A (Left) and Source B (Right)
            channel_analysis = self._analyze_dual_channel_audio(samples, sample_count)

            return {
                "sample_count": sample_count,
                "max_amplitude": max_amplitude,
                "avg_amplitude": avg_amplitude,
                "is_silent": is_silent,
                "silence_threshold": silence_threshold,
                "chunk_size_bytes": len(audio_chunk),
                "dual_channel_analysis": channel_analysis,
            }

        except Exception as e:
            return {"error": f"Audio analysis failed: {e}"}

    def _analyze_dual_channel_audio(self, samples, sample_count: int) -> dict[str, any]:
        """Detailed analysis of dual-channel audio for Source A (Left) and Source B (Right).

        Args:
            samples: Unpacked audio samples
            sample_count: Total number of samples

        Returns:
            Dict with detailed per-channel analysis
        """
        try:
            # For dual-channel (stereo), samples are interleaved L-R-L-R
            if sample_count >= 2 and sample_count % 2 == 0:
                # Extract Left (Source A) and Right (Source B) channels
                left_samples = samples[
                    0::2
                ]  # Source A (every other sample starting from 0)
                right_samples = samples[
                    1::2
                ]  # Source B (every other sample starting from 1)

                # Analyze Source A (Left Channel)
                source_a_analysis = self._analyze_single_channel(
                    left_samples, "Source A (Left)"
                )

                # Analyze Source B (Right Channel)
                source_b_analysis = self._analyze_single_channel(
                    right_samples, "Source B (Right)"
                )

                # Calculate channel balance and relationships
                balance_analysis = self._analyze_channel_balance(
                    source_a_analysis, source_b_analysis
                )

                return {
                    "is_dual_channel": True,
                    "source_a": source_a_analysis,
                    "source_b": source_b_analysis,
                    "balance": balance_analysis,
                    "interleaving_valid": len(left_samples) == len(right_samples),
                }
            # Not dual-channel or invalid sample count
            return {
                "is_dual_channel": False,
                "error": f"Invalid dual-channel format: {sample_count} samples (should be even)",
                "sample_count": sample_count,
            }

        except Exception as e:
            return {"error": f"Dual-channel analysis failed: {e}"}

    def _analyze_single_channel(
        self, channel_samples, channel_name: str
    ) -> dict[str, any]:
        """Analyze a single audio channel.

        Args:
            channel_samples: Audio samples for this channel
            channel_name: Name/description of the channel

        Returns:
            Dict with single-channel analysis
        """
        if not channel_samples:
            return {"error": f"No samples for {channel_name}"}

        max_amp = max(abs(s) for s in channel_samples)
        avg_amp = sum(abs(s) for s in channel_samples) / len(channel_samples)
        rms_amp = (sum(s * s for s in channel_samples) / len(channel_samples)) ** 0.5

        # Silence detection for this channel
        silence_threshold = 50  # Lower threshold for individual channels
        is_silent = max_amp < silence_threshold

        # Activity level classification
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

        return {
            "channel_name": channel_name,
            "sample_count": len(channel_samples),
            "max_amplitude": max_amp,
            "avg_amplitude": avg_amp,
            "rms_amplitude": rms_amp,
            "is_silent": is_silent,
            "activity_level": activity_level,
            "silence_threshold": silence_threshold,
        }

    def _analyze_channel_balance(
        self, source_a: dict[str, any], source_b: dict[str, any]
    ) -> dict[str, any]:
        """Analyze balance and relationships between Source A and Source B.

        Args:
            source_a: Analysis results for Source A (Left)
            source_b: Analysis results for Source B (Right)

        Returns:
            Dict with balance analysis
        """
        try:
            a_avg = source_a.get("avg_amplitude", 0)
            b_avg = source_b.get("avg_amplitude", 0)

            # Calculate balance ratio (0.5 = perfectly balanced)
            total_avg = a_avg + b_avg
            if total_avg > 0:
                balance_ratio = a_avg / total_avg
            else:
                balance_ratio = 0.5  # Default to balanced if both silent

            # Determine balance status
            if abs(balance_ratio - 0.5) < 0.1:
                balance_status = "well_balanced"
            elif balance_ratio > 0.8:
                balance_status = "source_a_dominant"
            elif balance_ratio < 0.2:
                balance_status = "source_b_dominant"
            elif balance_ratio > 0.6:
                balance_status = "source_a_louder"
            else:
                balance_status = "source_b_louder"

            # Check for problematic situations
            issues = []
            if source_a.get("is_silent", False) and not source_b.get(
                "is_silent", False
            ):
                issues.append(
                    "Source A (Left) is silent - AWS may only process Source B"
                )
            elif source_b.get("is_silent", False) and not source_a.get(
                "is_silent", False
            ):
                issues.append(
                    "Source B (Right) is silent - AWS may only process Source A"
                )
            elif source_a.get("is_silent", False) and source_b.get("is_silent", False):
                issues.append("Both channels are silent - no audio to process")

            # Check for significant level imbalance
            if total_avg > 0:
                max_ratio = max(a_avg, b_avg) / total_avg
                if max_ratio > 0.9:
                    issues.append(
                        f"Severe channel imbalance ({max_ratio:.1%}) - AWS may ignore quieter channel"
                    )

            return {
                "balance_ratio": balance_ratio,
                "balance_status": balance_status,
                "source_a_activity": source_a.get("activity_level", "unknown"),
                "source_b_activity": source_b.get("activity_level", "unknown"),
                "amplitude_difference": abs(a_avg - b_avg),
                "issues": issues,
                "recommendation": self._get_balance_recommendation(
                    balance_status, issues
                ),
            }

        except Exception as e:
            return {"error": f"Balance analysis failed: {e}"}

    def _get_balance_recommendation(self, balance_status: str, issues: list) -> str:
        """Get recommendation based on channel balance analysis."""
        if issues:
            return f"Address channel issues: {'; '.join(issues[:2])}"
        if balance_status == "well_balanced":
            return "Channel balance is optimal for AWS dual-channel processing"
        if "dominant" in balance_status:
            return "Consider adjusting audio levels - one source is much louder than the other"
        return "Channel balance is acceptable but could be improved"

    def _validate_and_align_chunk(self, audio_chunk: bytes) -> tuple[bytes, dict]:
        """Validate and align audio chunk for dual-channel processing.

        AWS requires dual-channel PCM chunks to be multiples of 4 bytes
        (2 bytes per sample × 2 channels = 4 bytes per sample pair).

        Args:
            audio_chunk: Raw audio data bytes

        Returns:
            Tuple of (aligned_chunk, alignment_info_dict)
        """
        chunk_size = len(audio_chunk)

        # Check if chunk size is valid for dual-channel int16 PCM
        alignment_info = {
            "original_size": chunk_size,
            "is_aligned": True,
            "padding_added": 0,
            "warnings": [],
        }

        # For dual-channel int16 PCM: each sample pair = 4 bytes (L sample + R sample)
        if chunk_size % 4 != 0:
            alignment_info["is_aligned"] = False

            # Calculate padding needed
            padding_needed = 4 - (chunk_size % 4)
            alignment_info["padding_added"] = padding_needed

            # Add zero padding to align to 4-byte boundary
            aligned_chunk = audio_chunk + b"\x00" * padding_needed

            alignment_info["warnings"].append(
                f"Chunk size {chunk_size} not divisible by 4 - added {padding_needed} padding bytes"
            )
            alignment_info["aligned_size"] = len(aligned_chunk)

            # Log alignment issue for first 20 chunks
            if self._audio_chunk_count <= 20:
                logger.warning(
                    f"⚠️ AWS Chunk Alignment: Original {chunk_size} bytes → "
                    f"Aligned {len(aligned_chunk)} bytes (+{padding_needed} padding)"
                )
        else:
            # Already aligned
            aligned_chunk = audio_chunk
            alignment_info["aligned_size"] = chunk_size

        # Validate sample count for dual-channel
        total_samples = len(aligned_chunk) // 2  # int16 = 2 bytes per sample
        sample_pairs = total_samples // 2  # dual-channel = 2 samples per pair

        if total_samples % 2 != 0:
            alignment_info["warnings"].append(
                f"Odd number of samples ({total_samples}) - may indicate single-channel audio"
            )

        # Additional validation
        alignment_info.update(
            {
                "total_samples": total_samples,
                "sample_pairs": sample_pairs,
                "expected_dual_channel": total_samples % 2 == 0,
                "chunk_valid_for_aws": len(aligned_chunk) % 4 == 0,
            }
        )

        return aligned_chunk, alignment_info

    def _add_dual_channel_optimizations(self, stream_params: dict) -> None:
        """Add optimization parameters for dual-channel AWS Transcribe processing.

        Args:
            stream_params: Stream parameters dict to modify
        """
        # Add parameters that may improve dual-channel processing
        optimizations_added = []

        # Enable partial results stabilization for better real-time experience
        stream_params["enable_partial_results_stabilization"] = True
        optimizations_added.append("partial_results_stabilization")

        # Set vocabulary filter mode to mask instead of remove for better channel continuity
        # (Only if vocabulary filtering is being used)
        if "vocabulary_filter_name" in stream_params:
            stream_params["vocabulary_filter_method"] = "mask"
            optimizations_added.append("vocabulary_filter_masking")

        # Add content redaction settings for dual-channel (if needed)
        # This can help with processing consistency across channels
        # stream_params['content_redaction_type'] = 'PII'  # Uncomment if needed

        # Log the optimizations applied
        if optimizations_added:
            logger.info(
                f"🚀 AWS Dual-Channel Optimizations: {', '.join(optimizations_added)}"
            )
        else:
            logger.info(
                "🚀 AWS Dual-Channel: Using standard channel identification parameters"
            )

        # Add detailed parameter validation for dual-channel
        self._validate_dual_channel_config(stream_params)

    def _validate_dual_channel_config(self, stream_params: dict) -> None:
        """Validate dual-channel specific configuration."""
        issues = []
        recommendations = []

        # Check required parameters are present
        if not stream_params.get("enable_channel_identification"):
            issues.append("enable_channel_identification is False")

        if stream_params.get("number_of_channels") != 2:
            issues.append(
                f"number_of_channels is {stream_params.get('number_of_channels', 'N/A')}, expected 2"
            )

        # Check for conflicting parameters
        if stream_params.get("show_speaker_label"):
            issues.append(
                "show_speaker_label=True conflicts with enable_channel_identification=True"
            )
            recommendations.append(
                "Use either channel identification OR speaker labels, not both"
            )

        # Media encoding validation
        if stream_params.get("media_encoding") != "pcm":
            issues.append(
                f"media_encoding is {stream_params.get('media_encoding')}, PCM is recommended for dual-channel"
            )

        # Sample rate validation for dual-channel
        sample_rate = stream_params.get("media_sample_rate_hz", 0)
        if sample_rate not in [16000, 44100, 48000]:
            recommendations.append(
                f"Sample rate {sample_rate}Hz may not be optimal for dual-channel (consider 16000Hz)"
            )

        # Log validation results
        if issues:
            logger.warning(f"⚠️ AWS Dual-Channel Config Issues: {'; '.join(issues)}")

        if recommendations:
            logger.info(
                f"💡 AWS Dual-Channel Recommendations: {'; '.join(recommendations)}"
            )

        if not issues:
            logger.info("✅ AWS Dual-Channel configuration validation passed")

    def _log_channel_quality_summary(self, dual_channel_analysis: dict) -> None:
        """Log summary of channel quality for monitoring purposes.

        Args:
            dual_channel_analysis: Results from dual-channel analysis
        """
        try:
            source_a = dual_channel_analysis.get("source_a", {})
            source_b = dual_channel_analysis.get("source_b", {})
            balance = dual_channel_analysis.get("balance", {})

            # Channel activity summary
            source_a_activity = source_a.get("activity_level", "unknown")
            source_b_activity = source_b.get("activity_level", "unknown")

            logger.info(
                f"   🎚️  Channel Activity: Source A = {source_a_activity}, Source B = {source_b_activity}"
            )

            # Balance status
            balance_status = balance.get("balance_status", "unknown")
            balance_ratio = balance.get("balance_ratio", 0.5)

            if balance_status == "well_balanced":
                logger.info(
                    f"   ⚖️  Channel Balance: ✅ {balance_status} ({balance_ratio:.3f})"
                )
            else:
                logger.info(
                    f"   ⚖️  Channel Balance: ⚠️  {balance_status} ({balance_ratio:.3f})"
                )

            # Critical issues that could explain incomplete transcription
            issues = balance.get("issues", [])
            if issues:
                logger.warning("   🚨 Channel Issues Affecting AWS Transcription:")
                for issue in issues:
                    logger.warning(f"      - {issue}")

            # Amplitude comparison
            source_a_avg = source_a.get("avg_amplitude", 0)
            source_b_avg = source_b.get("avg_amplitude", 0)

            if source_a_avg > 0 and source_b_avg > 0:
                amp_ratio = max(source_a_avg, source_b_avg) / min(
                    source_a_avg, source_b_avg
                )
                if amp_ratio > 5.0:  # One channel 5x louder than other
                    logger.warning(
                        f"   📊 Amplitude Imbalance: {amp_ratio:.1f}x difference may cause AWS to ignore quieter channel"
                    )
                else:
                    logger.info(
                        f"   📊 Amplitude Balance: {amp_ratio:.1f}x difference (acceptable)"
                    )

            # Channel silence warnings
            if source_a.get("is_silent", False):
                logger.warning(
                    "   🔇 Source A (Left) Silent: AWS will only process Source B (Right) channel"
                )
            elif source_b.get("is_silent", False):
                logger.warning(
                    "   🔇 Source B (Right) Silent: AWS will only process Source A (Left) channel"
                )

            # Overall recommendation
            recommendation = balance.get("recommendation", "")
            if recommendation and "optimal" not in recommendation.lower():
                logger.info(f"   💡 Channel Recommendation: {recommendation}")

        except Exception as e:
            logger.warning(f"⚠️ Channel quality summary error: {e}")

    def _log_aws_stream_configuration(
        self, stream_params: dict, audio_config: AudioConfig
    ) -> None:
        """Log comprehensive AWS stream configuration for debugging.

        Args:
            stream_params: Parameters being sent to AWS
            audio_config: Audio configuration being used
        """
        logger.info("📋 AWS Transcribe Stream Configuration:")
        logger.info("   🏷️  Provider: AWS Transcribe Streaming")
        logger.info(f"   🌍 Region: {self.region}")
        logger.info(f"   🗣️  Language: {stream_params.get('language_code', 'N/A')}")
        logger.info(
            f"   📡 Sample Rate: {stream_params.get('media_sample_rate_hz', 'N/A')} Hz"
        )
        logger.info(
            f"   🎵 Media Encoding: {stream_params.get('media_encoding', 'N/A')}"
        )
        logger.info(f"   🎚️  Channels: {stream_params.get('number_of_channels', 1)}")
        logger.info(
            f"   🔍 Channel ID Enabled: {stream_params.get('enable_channel_identification', False)}"
        )

        # Log additional AWS-specific parameters if present
        aws_features = []
        if stream_params.get("enable_channel_identification"):
            aws_features.append("Channel Identification")
        if stream_params.get("show_speaker_label"):
            aws_features.append("Speaker Labeling")
        if stream_params.get("vocabulary_name"):
            aws_features.append(f"Custom Vocab: {stream_params['vocabulary_name']}")
        if stream_params.get("enable_partial_results_stabilization"):
            aws_features.append("Partial Results Stabilization")

        if aws_features:
            logger.info(f"   ✨ Features: {', '.join(aws_features)}")
        else:
            logger.info("   ✨ Features: Basic transcription only")

        # Log original audio input configuration
        logger.info("   📦 Original Audio Config:")
        logger.info(f"      - Sample Rate: {audio_config.sample_rate} Hz")
        logger.info(f"      - Channels: {audio_config.channels}")
        logger.info(f"      - Format: {audio_config.format}")
        logger.info(f"      - Chunk Size: {audio_config.chunk_size} samples")

        # Calculate expected data rates
        bytes_per_sample = 2 if audio_config.format == "int16" else 4
        expected_bytes_per_chunk = (
            audio_config.chunk_size * audio_config.channels * bytes_per_sample
        )
        chunks_per_second = audio_config.sample_rate / audio_config.chunk_size
        bytes_per_second = expected_bytes_per_chunk * chunks_per_second

        logger.info("   📈 Expected Data Rates:")
        logger.info(f"      - Bytes per chunk: {expected_bytes_per_chunk}")
        logger.info(f"      - Chunks per second: {chunks_per_second:.1f}")
        logger.info(f"      - Bytes per second: {bytes_per_second:,.0f}")

        # Log AWS service endpoint info
        logger.info(
            f"   🔗 AWS Service: transcribestreaming.{self.region}.amazonaws.com"
        )
        logger.info(f"   🔑 Profile: {self.profile_name or 'default'}")

    def _validate_aws_stream_params(self, stream_params: dict) -> dict:
        """Validate AWS stream parameters before sending to service.

        Args:
            stream_params: Parameters to validate

        Returns:
            Dict with 'errors' and 'warnings' lists
        """
        errors = []
        warnings = []

        # Validate required parameters
        if not stream_params.get("language_code"):
            errors.append("language_code is required")

        if not stream_params.get("media_sample_rate_hz"):
            errors.append("media_sample_rate_hz is required")
        elif not isinstance(stream_params["media_sample_rate_hz"], int):
            errors.append(
                f"media_sample_rate_hz must be integer, got {type(stream_params['media_sample_rate_hz'])}"
            )
        elif stream_params["media_sample_rate_hz"] not in [
            8000,
            16000,
            22050,
            44100,
            48000,
        ]:
            warnings.append(
                f"Sample rate {stream_params['media_sample_rate_hz']} Hz may not be optimal for transcription"
            )

        if not stream_params.get("media_encoding"):
            errors.append("media_encoding is required")
        elif stream_params["media_encoding"] not in ["pcm", "ogg-opus", "flac"]:
            errors.append(
                f"Unsupported media_encoding: {stream_params['media_encoding']}"
            )

        # Validate channel identification configuration
        if stream_params.get("enable_channel_identification"):
            if not stream_params.get("number_of_channels"):
                errors.append(
                    "number_of_channels is required when enable_channel_identification=True"
                )
            elif stream_params["number_of_channels"] > 2:
                errors.append(
                    f"AWS Transcribe supports maximum 2 channels for channel identification, got {stream_params['number_of_channels']}"
                )
            elif stream_params["number_of_channels"] < 2:
                warnings.append(
                    f"Channel identification enabled but only {stream_params['number_of_channels']} channel(s) provided"
                )

        # Validate language code format
        language_code = stream_params.get("language_code", "")
        if language_code and not (len(language_code) == 5 and language_code[2] == "-"):
            warnings.append(
                f"Language code '{language_code}' may not be in correct format (expected: xx-XX)"
            )

        # Check for dual-channel configuration consistency
        if (
            stream_params.get("enable_channel_identification")
            and stream_params.get("number_of_channels") == 2
            and stream_params.get("media_encoding") == "pcm"
        ):
            logger.info(
                "✅ Dual-channel PCM configuration detected - optimal for speaker separation"
            )

        return {
            "errors": errors,
            "warnings": warnings,
            "validation_passed": len(errors) == 0,
        }

    async def _attempt_reconnection(self, audio_config: AudioConfig) -> bool:
        """Attempt to reconnect to AWS Transcribe with retry logic.

        Args:
            audio_config: Audio configuration for the stream

        Returns:
            True if reconnection successful, False otherwise
        """
        if self.retry_count >= self.max_retries:
            logger.error(
                f"❌ AWS Transcribe: Maximum retry attempts ({self.max_retries}) exceeded"
            )
            if self.connection_health_callback:
                self.connection_health_callback(
                    False, f"Max retries ({self.max_retries}) exceeded"
                )
            return False

        self.retry_count += 1
        delay = await self._calculate_retry_delay()

        logger.info(
            f"🔄 AWS Transcribe: Attempting reconnection #{self.retry_count}/{self.max_retries} after {delay:.1f}s delay..."
        )
        if self.connection_health_callback:
            self.connection_health_callback(
                False,
                f"Reconnecting... (attempt {self.retry_count}/{self.max_retries})",
            )

        await asyncio.sleep(delay)

        try:
            # Stop existing stream cleanly
            await self.stop_stream()

            # Wait a bit before restart
            await asyncio.sleep(1.0)

            # Restart stream
            await self.start_stream(audio_config)

            logger.info(
                f"✅ AWS Transcribe: Reconnection attempt #{self.retry_count} successful"
            )
            return True

        except Exception as e:
            logger.error(
                f"❌ AWS Transcribe: Reconnection attempt #{self.retry_count} failed: {e}"
            )
            return False

    async def start_stream(self, audio_config: AudioConfig) -> None:
        """
        Start the AWS Transcribe streaming session.

        Args:
            audio_config: Audio configuration for the stream

        Raises:
            AWSTranscribeError: If stream initialization fails
            ConnectionError: If unable to connect to AWS
            ValueError: If audio configuration is invalid
        """
        try:
            logger.info(
                f"🚀 AWS Transcribe: Starting stream with config: {audio_config}"
            )

            # Reset connection state for new session
            self.is_connected = False
            self.retry_count = 0
            self._reset_fallback_tracking()
            logger.info("🔄 AWS Transcribe: Reset connection state for new session")

            # Create fresh result queue for this session in current event loop
            current_loop = asyncio.get_event_loop()
            logger.info(f"🔄 AWS Transcribe: Current event loop ID: {id(current_loop)}")

            if self._current_event_loop != current_loop:
                logger.info(
                    f"🔄 AWS Transcribe: Event loop changed (old: {id(self._current_event_loop) if self._current_event_loop else 'None'}, new: {id(current_loop)})"
                )
                self._current_event_loop = current_loop

            # Always create fresh queue for each session to avoid event loop binding issues
            old_queue_id = id(self.result_queue) if self.result_queue else "None"
            self.result_queue = asyncio.Queue()
            logger.info(
                f"🔄 AWS Transcribe: Created fresh result queue (old: {old_queue_id}, new: {id(self.result_queue)})"
            )
            logger.info(
                f"🔄 AWS Transcribe: Queue created in event loop: {id(current_loop)}"
            )

            # Validate audio configuration
            if not isinstance(audio_config, AudioConfig):
                raise ValueError("audio_config must be an AudioConfig instance")

            # Determine connection strategy and route accordingly
            connection_mode = self._determine_connection_strategy(audio_config)
            self._connection_mode = connection_mode
            logger.info(
                f"🎯 AWS Transcribe: Selected connection mode: {connection_mode}"
            )

            if connection_mode == "dual_connection":
                # Route to dual connection implementation
                logger.info("🔀 AWS Transcribe: Routing to dual connection stream")
                return await self._start_dual_connection_stream(audio_config)
            # Continue with single connection logic
            logger.info("🔀 AWS Transcribe: Routing to single connection stream")
            return await self._start_single_connection_stream(audio_config)

        except Exception as e:
            logger.error(f"❌ AWS Transcribe: Stream start failed: {e}")
            raise AWSTranscribeError(
                f"Failed to start AWS Transcribe stream: {e}"
            ) from e

    async def _start_single_connection_stream(self, audio_config: AudioConfig) -> None:
        """
        Start single connection AWS Transcribe stream (existing logic).

        Args:
            audio_config: Audio configuration for the stream
        """
        try:
            # Create boto3 session with profile if specified
            if self.profile_name:
                logger.info(
                    f"🔑 AWS Transcribe: Using AWS profile: {self.profile_name}"
                )
                boto3.Session(profile_name=self.profile_name)
            else:
                logger.info("🔑 AWS Transcribe: Using default AWS credentials")
                boto3.Session()

            logger.info(
                f"🚀 Initializing AWS Transcribe client (region: {self.region})"
            )
            self.client = TranscribeStreamingClient(region=self.region)

            logger.info(
                f"🎯 Starting stream transcription (language: {self.language_code}, sample_rate: {audio_config.sample_rate}, channels: {audio_config.channels})"
            )

            # Configure stream transcription parameters
            stream_params = {
                "language_code": self.language_code,
                "media_sample_rate_hz": audio_config.sample_rate,
                "media_encoding": "pcm",
            }

            # Configure channel identification based on input channels
            if audio_config.channels == 1:
                # Mono input - standard transcription without channel identification
                logger.info(
                    "🎯 AWS Transcribe: Mono input - standard transcription mode"
                )
            elif audio_config.channels == 2 and self.enable_channel_identification:
                # Dual-channel input - enable speaker separation via AWS channel identification
                stream_params["enable_channel_identification"] = True
                stream_params["number_of_channels"] = (
                    audio_config.channels
                )  # Required for channel identification

                # Add optimization parameters for dual-channel processing
                self._add_dual_channel_optimizations(stream_params)

                logger.info(
                    "🎯 AWS Transcribe: Dual-channel input - enabled channel identification for speaker separation"
                )
            elif audio_config.channels > 2:
                # This shouldn't happen with device filtering to 1-2 channels only
                logger.warning(
                    f"⚠️ AWS Transcribe: Received {audio_config.channels} channels. Only 1-2 channels supported."
                )

            # Log complete AWS stream configuration for debugging
            self._log_aws_stream_configuration(stream_params, audio_config)

            # Validate AWS configuration before sending
            validation_result = self._validate_aws_stream_params(stream_params)
            if validation_result.get("errors"):
                error_msg = f"AWS stream parameter validation failed: {validation_result['errors']}"
                logger.error(f"❌ {error_msg}")
                raise AWSTranscribeError(error_msg)

            if validation_result.get("warnings"):
                for warning in validation_result["warnings"]:
                    logger.warning(f"⚠️ AWS stream parameter warning: {warning}")

            # Start stream transcription with configured parameters
            logger.info("🚀 AWS Transcribe: Sending stream parameters to AWS...")
            self.stream = await self.client.start_stream_transcription(**stream_params)

            # Log successful connection
            logger.info("🎯 AWS Transcribe: Stream parameters accepted by AWS service")

            logger.info("✅ AWS Transcribe stream connection established")

            # Initialize connection health tracking
            self.is_connected = True
            self.last_result_time = time.time()
            self.last_audio_sent_time = time.time()

            # Reset audio analysis counters for this session
            self._audio_chunk_count = 0
            self._total_audio_samples_analyzed = 0
            self._silence_chunks = 0
            self._audio_level_sum = 0.0

            # Create AWS Transcribe handler using proper AWS pattern
            logger.info("🔄 AWS Transcribe: Creating handler for transcript events")
            self.handler = AWSTranscribeHandler(
                self.stream.output_stream, self.result_queue, self
            )

            # Start the AWS handler event processing task (AWS recommended pattern)
            self._streaming_task = asyncio.create_task(self.handler.handle_events())

            # Start the health monitoring task (with fixed stream checking)
            self._health_check_task = asyncio.create_task(
                self._monitor_connection_health()
            )

            logger.info(
                "🔄 AWS Transcribe: Handler and health monitor started using AWS pattern"
            )

        except Exception as e:
            logger.error(f"❌ Failed to start AWS Transcribe stream: {e}")
            logger.error(f"❌ Error details: {str(e)}")
            raise AWSTranscribeError(
                f"Failed to start AWS Transcribe stream: {e}"
            ) from e

    async def send_audio(self, audio_chunk: bytes) -> None:
        """Send audio data to AWS Transcribe (strategy-aware routing)."""
        # Route based on current connection mode
        if self._connection_mode == "dual_connection":
            return await self._send_audio_dual_connection(audio_chunk)
        return await self._send_audio_single_connection(audio_chunk)

    async def _send_audio_single_connection(self, audio_chunk: bytes) -> None:
        """Send audio data to single AWS Transcribe connection with comprehensive audio analysis."""
        if self.stream and self.stream.input_stream:
            try:
                # Increment chunk counter
                self._audio_chunk_count += 1

                # Validate and align chunk for dual-channel processing
                aligned_chunk, alignment_info = self._validate_and_align_chunk(
                    audio_chunk
                )

                # Analyze audio content for debugging
                audio_analysis = self._analyze_audio_content(aligned_chunk)

                # Add alignment info to analysis
                audio_analysis["alignment_info"] = alignment_info

                # Check for fallback to dual connection if enabled
                if (
                    self.dual_fallback_enabled
                    and self._connection_mode == "single_connection"
                    and self._should_fallback_to_dual_connection(audio_analysis)
                ):
                    logger.warning(
                        "🔄 AWS Fallback: Fallback conditions detected, will attempt to switch to dual connection"
                    )
                    # Store current audio config for fallback attempt
                    if not hasattr(self, "_stored_audio_config"):
                        # We'll need the audio config for fallback, but we don't have it here
                        # Log the need for fallback but don't attempt it in send_audio to avoid blocking
                        logger.warning(
                            "⚠️ AWS Fallback: Fallback needed but cannot switch during send_audio - consider switching at stream level"
                        )

                # Track silence statistics
                if audio_analysis.get("is_silent", False):
                    self._silence_chunks += 1

                # Update running statistics
                if "avg_amplitude" in audio_analysis:
                    self._audio_level_sum += audio_analysis["avg_amplitude"]
                    self._total_audio_samples_analyzed += audio_analysis.get(
                        "sample_count", 0
                    )

                # Send to AWS Transcribe (using aligned chunk)
                await self.stream.input_stream.send_audio_event(
                    audio_chunk=aligned_chunk
                )

                # Enhanced logging with audio analysis
                chunk_size = len(audio_chunk)
                logger.debug(f"📡 AWS Transcribe: Sent audio chunk {chunk_size} bytes")

                # Detailed logging every 10 chunks for initial debugging
                if self._audio_chunk_count <= 100 and self._audio_chunk_count % 10 == 0:
                    if "error" in audio_analysis:
                        logger.warning(
                            f"⚠️ AWS Audio Analysis Error: {audio_analysis['error']}"
                        )
                    else:
                        logger.info(
                            f"🎵 AWS Audio Analysis (chunk #{self._audio_chunk_count}):"
                        )
                        logger.info(
                            f"   📊 Overall - Max: {audio_analysis.get('max_amplitude', 'N/A')}, "
                            f"Avg: {audio_analysis.get('avg_amplitude', 'N/A'):.1f}"
                        )
                        logger.info(
                            f"   🔇 Silent: {audio_analysis.get('is_silent', 'N/A')} "
                            f"(threshold: {audio_analysis.get('silence_threshold', 'N/A')})"
                        )
                        logger.info(
                            f"   📦 Samples: {audio_analysis.get('sample_count', 'N/A')}, "
                            f"Bytes: {audio_analysis.get('chunk_size_bytes', 'N/A')}"
                        )

                        # Log chunk alignment information
                        alignment = audio_analysis.get("alignment_info", {})
                        if alignment:
                            if not alignment.get("is_aligned", True):
                                logger.warning(
                                    f"   ⚠️ Alignment: {alignment['original_size']} → "
                                    f"{alignment['aligned_size']} bytes (+{alignment['padding_added']} padding)"
                                )
                            elif alignment.get("chunk_valid_for_aws"):
                                logger.debug(
                                    f"   ✅ Chunk aligned: {alignment['aligned_size']} bytes "
                                    f"({alignment['sample_pairs']} sample pairs)"
                                )

                            # Log alignment warnings
                            for warning in alignment.get("warnings", []):
                                logger.warning(f"   ⚠️ {warning}")

                        # Enhanced dual-channel logging
                        dual_channel = audio_analysis.get("dual_channel_analysis", {})
                        if dual_channel.get("is_dual_channel"):
                            source_a = dual_channel.get("source_a", {})
                            source_b = dual_channel.get("source_b", {})
                            balance = dual_channel.get("balance", {})

                            logger.info(
                                f"   🎚️  Source A (Left): {source_a.get('activity_level', 'N/A')} - "
                                f"Max: {source_a.get('max_amplitude', 'N/A')}, "
                                f"Avg: {source_a.get('avg_amplitude', 'N/A'):.1f}"
                            )
                            logger.info(
                                f"   🎚️  Source B (Right): {source_b.get('activity_level', 'N/A')} - "
                                f"Max: {source_b.get('max_amplitude', 'N/A')}, "
                                f"Avg: {source_b.get('avg_amplitude', 'N/A'):.1f}"
                            )
                            logger.info(
                                f"   ⚖️  Balance: {balance.get('balance_status', 'N/A')} "
                                f"(ratio: {balance.get('balance_ratio', 0):.3f})"
                            )

                            # Log critical issues
                            issues = balance.get("issues", [])
                            if issues:
                                logger.warning(
                                    f"   ⚠️ Channel Issues: {', '.join(issues)}"
                                )

                            recommendation = balance.get("recommendation", "")
                            if (
                                recommendation
                                and "optimal" not in recommendation.lower()
                            ):
                                logger.info(f"   💡 Recommendation: {recommendation}")
                        elif "error" in dual_channel:
                            logger.warning(
                                f"   ⚠️ Dual-channel analysis error: {dual_channel['error']}"
                            )
                        else:
                            logger.info("   🔇 Single-channel audio detected")

                # Periodic summary every 100 chunks
                if self._audio_chunk_count % 100 == 0:
                    silence_rate = (
                        self._silence_chunks / self._audio_chunk_count
                    ) * 100
                    avg_level = (
                        self._audio_level_sum / self._audio_chunk_count
                        if self._audio_chunk_count > 0
                        else 0
                    )

                    logger.info(
                        f"📊 AWS Audio Summary (chunk #{self._audio_chunk_count}):"
                    )
                    logger.info(
                        f"   🔇 Overall silence rate: {silence_rate:.1f}% ({self._silence_chunks}/{self._audio_chunk_count})"
                    )
                    logger.info(f"   📈 Average audio level: {avg_level:.1f}")
                    logger.info(
                        f"   🎵 Total samples analyzed: {self._total_audio_samples_analyzed:,}"
                    )

                    # Enhanced dual-channel quality monitoring
                    dual_channel = audio_analysis.get("dual_channel_analysis", {})
                    if dual_channel.get("is_dual_channel"):
                        self._log_channel_quality_summary(dual_channel)

                    # Critical warning if too much silence
                    if silence_rate > 80:
                        logger.warning(
                            f"⚠️ AWS Transcribe: High silence rate ({silence_rate:.1f}%) - "
                            f"This may explain why AWS is returning 0 results!"
                        )

                    # Alignment status summary
                    alignment = audio_analysis.get("alignment_info", {})
                    if alignment and not alignment.get("is_aligned", True):
                        total_padding = alignment.get("padding_added", 0) * (
                            self._audio_chunk_count / 100
                        )
                        logger.info(
                            f"   ⚠️ Alignment: ~{total_padding:.0f} padding bytes added in last 100 chunks"
                        )

                    logger.info(
                        f"📡 AWS Transcribe: Audio chunk #{self._audio_chunk_count} - {chunk_size} bytes sent directly to AWS"
                    )

                # Update audio send time for connection health monitoring
                self.last_audio_sent_time = time.time()

            except Exception as e:
                logger.error(f"❌ Failed to send audio to AWS Transcribe: {e}")
                logger.error(f"❌ Send error details: {str(e)}")

                # Mark connection as unhealthy on send errors
                if self.is_connected:
                    self.is_connected = False
                    if self.connection_health_callback:
                        self.connection_health_callback(
                            False, f"Audio send error: {str(e)}"
                        )

                raise AWSTranscribeError(
                    f"Failed to send audio to AWS Transcribe: {e}"
                ) from e
        else:
            logger.warning(
                f"⚠️  Cannot send audio - stream not available (stream: {self.stream is not None}, input_stream: {self.stream.input_stream is not None if self.stream else False})"
            )

            # Mark connection as unhealthy if stream is not available
            if self.is_connected:
                self.is_connected = False
                if self.connection_health_callback:
                    self.connection_health_callback(False, "Stream not available")

    async def get_transcription(self) -> AsyncGenerator[TranscriptionResult, None]:
        """Get transcription results as they become available (strategy-aware)."""
        # Route based on current connection mode
        if self._connection_mode == "dual_connection":
            async for result in self._get_transcription_dual_connection():
                yield result
        else:
            async for result in self._get_transcription_single_connection():
                yield result

    async def _get_transcription_single_connection(
        self,
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """Get transcription results from single connection."""
        logger.info(
            f"🔊 AWS Transcribe: Starting transcription generator with queue {id(self.result_queue) if self.result_queue else 'None'}"
        )

        while True:
            try:
                # Validate that queue exists and is accessible
                if not self.result_queue:
                    logger.error("❌ AWS Transcribe: No result queue available")
                    break

                # Wait for results with timeout to allow for graceful shutdown
                result = await asyncio.wait_for(self.result_queue.get(), timeout=0.1)
                logger.debug(
                    f"🔊 AWS Transcribe: Got result from queue {id(self.result_queue)}: '{result.text}'"
                )

                # Track transcription quality for fallback decision making
                self._track_transcription_quality(result)

                yield result
            except TimeoutError:
                # Continue polling for results
                continue
            except asyncio.CancelledError:
                logger.info("🛑 AWS Transcribe: Transcription generator cancelled")
                break
            except Exception as e:
                logger.error(
                    f"❌ AWS Transcribe: Error getting transcription result: {e}"
                )
                logger.error(
                    f"❌ AWS Transcribe: Queue state - exists: {self.result_queue is not None}, queue: {self.result_queue}"
                )
                if self.result_queue:
                    logger.error(
                        f"❌ AWS Transcribe: Queue ID: {id(self.result_queue)}, size: {self.result_queue.qsize()}"
                    )
                break

        logger.info("🛑 AWS Transcribe: Transcription generator stopped")

    async def stop_stream(self) -> None:
        """Stop the transcription stream and cleanup resources (strategy-aware)."""
        # Route based on current connection mode
        if self._connection_mode == "dual_connection":
            return await self._stop_dual_connection_stream()
        return await self._stop_single_connection_stream()

    async def _stop_single_connection_stream(self) -> None:
        """Stop single connection transcription stream and cleanup resources."""
        logger.info("🛑 AWS Transcribe: Stopping stream...")

        try:
            # Step 1: Stop the input stream
            if self.stream and self.stream.input_stream:
                try:
                    logger.info("🛑 AWS Transcribe: Ending input stream...")
                    await asyncio.wait_for(
                        self.stream.input_stream.end_stream(), timeout=1.0
                    )
                    logger.info("✅ AWS Transcribe: Input stream ended")
                except TimeoutError:
                    logger.warning("⚠️ AWS Transcribe: Input stream end timed out")
                except Exception as e:
                    logger.warning(f"⚠️ AWS Transcribe: Error ending input stream: {e}")

            # Step 2: Cancel the health monitoring task
            if self._health_check_task and not self._health_check_task.done():
                try:
                    logger.info("🛑 AWS Transcribe: Cancelling health monitor task...")
                    self._health_check_task.cancel()
                    await asyncio.wait_for(self._health_check_task, timeout=0.5)
                    logger.info("✅ AWS Transcribe: Health monitor task cancelled")
                except asyncio.CancelledError:
                    logger.info("✅ AWS Transcribe: Health monitor task cancelled")
                except TimeoutError:
                    logger.warning(
                        "⚠️ AWS Transcribe: Health monitor task cancellation timed out"
                    )
                except Exception as e:
                    logger.warning(
                        f"⚠️ AWS Transcribe: Error cancelling health monitor task: {e}"
                    )

            # Step 3: Cancel the streaming task
            if self._streaming_task and not self._streaming_task.done():
                try:
                    logger.info("🛑 AWS Transcribe: Cancelling streaming task...")
                    self._streaming_task.cancel()
                    await asyncio.wait_for(self._streaming_task, timeout=0.5)
                    logger.info("✅ AWS Transcribe: Streaming task cancelled")
                except asyncio.CancelledError:
                    logger.info("✅ AWS Transcribe: Streaming task cancelled")
                except TimeoutError:
                    logger.warning(
                        "⚠️ AWS Transcribe: Streaming task cancellation timed out"
                    )
                except Exception as e:
                    logger.warning(
                        f"⚠️ AWS Transcribe: Error cancelling streaming task: {e}"
                    )

            logger.info("✅ AWS Transcribe: Stream stopped successfully")

        except Exception as e:
            logger.error(f"❌ AWS Transcribe: Error stopping stream: {e}")
            # Don't re-raise - we want cleanup to always complete
        finally:
            # Always clear references
            self.stream = None
            self.client = None
            self.handler = None
            self._streaming_task = None
            self._health_check_task = None

            # Clear result queue to prevent stale results from carrying over
            if self.result_queue:
                queue_size = self.result_queue.qsize()
                if queue_size > 0:
                    logger.info(
                        f"🗑️ AWS Transcribe: Clearing {queue_size} items from result queue"
                    )
                    # Drain the queue
                    try:
                        while not self.result_queue.empty():
                            try:
                                self.result_queue.get_nowait()
                            except Exception:
                                break
                    except Exception as e:
                        logger.warning(
                            f"⚠️ AWS Transcribe: Error clearing result queue: {e}"
                        )

                logger.info(
                    f"🗑️ AWS Transcribe: Cleared result queue {id(self.result_queue)}"
                )

            # Don't set result_queue to None - let it be recreated fresh in next session

            # Reset connection health
            self.is_connected = False
            self.last_result_time = 0.0
            self.last_audio_sent_time = 0.0

            logger.info("🛑 AWS Transcribe: Cleanup completed")

    def get_required_channels(self) -> int:
        """
        Get the number of audio channels required by AWS Transcribe.

        Returns 1 (mono) as the default requirement since:
        - 1-2 channel devices → processed to 1 channel (mono)
        - 3-4 channel devices → processed to 2 channels (dual-channel with speaker separation)

        AWS Transcribe adaptively handles both:
        - 1 channel: Standard mono transcription
        - 2 channels: Dual-channel transcription with speaker identification

        Returns:
            int: 1 channel (mono) as the primary requirement
        """
        return self.required_channels

    # ===================================================================
    # Dual Connection Strategy Methods
    # ===================================================================

    async def _start_dual_connection_stream(self, audio_config: AudioConfig) -> None:
        """
        Start dual connection AWS Transcribe stream using separate mono connections.

        Args:
            audio_config: Audio configuration (must be stereo - 2 channels)
        """
        test_mode = self.dual_connection_test_mode
        logger.info(
            f"🚀 AWS Dual Connection: Starting dual connection stream in test mode: {test_mode}"
        )

        try:
            # Initialize dual connection components if not already done
            self._initialize_dual_connection_components(audio_config)
            components = self._dual_connection_components

            # Create separate mono audio configs for left/right channels
            left_config = AudioConfig(
                sample_rate=audio_config.sample_rate,
                channels=1,  # Mono for left channel
                chunk_size=audio_config.chunk_size,
                format=audio_config.format,
            )

            right_config = AudioConfig(
                sample_rate=audio_config.sample_rate,
                channels=1,  # Mono for right channel
                chunk_size=audio_config.chunk_size,
                format=audio_config.format,
            )

            # Create AWS Transcribe providers based on test mode
            if test_mode in ["left_only", "full"]:
                logger.info("🏗️ AWS Dual Connection: Creating left channel provider...")
                components["left_provider"] = AWSTranscribeProvider(
                    region=self.region,
                    language_code=self.language_code,
                    profile_name=self.profile_name,
                    connection_strategy="single",  # Force single connection mode for individual channels
                    dual_fallback_enabled=False,
                )
            else:
                logger.info(
                    "🧪 AWS Dual Connection: Left channel provider DISABLED in test mode"
                )
                components["left_provider"] = None

            if test_mode in ["right_only", "full"]:
                logger.info("🏗️ AWS Dual Connection: Creating right channel provider...")
                components["right_provider"] = AWSTranscribeProvider(
                    region=self.region,
                    language_code=self.language_code,
                    profile_name=self.profile_name,
                    connection_strategy="single",  # Force single connection mode for individual channels
                    dual_fallback_enabled=False,
                )
            else:
                logger.info(
                    "🧪 AWS Dual Connection: Right channel provider DISABLED in test mode"
                )
                components["right_provider"] = None

            # Create separate result queues for channel synchronization
            components["left_queue"] = asyncio.Queue()
            components["right_queue"] = asyncio.Queue()

            # Start channel streams based on test mode
            if components["left_provider"]:
                logger.info("🚀 AWS Dual Connection: Starting left channel stream...")
                await components["left_provider"].start_stream(left_config)
            else:
                logger.info(
                    "🧪 AWS Dual Connection: Left channel stream SKIPPED in test mode"
                )

            if components["right_provider"]:
                logger.info("🚀 AWS Dual Connection: Starting right channel stream...")
                await components["right_provider"].start_stream(right_config)
            else:
                logger.info(
                    "🧪 AWS Dual Connection: Right channel stream SKIPPED in test mode"
                )

            # Start error handler monitoring
            logger.info("🔍 AWS Dual Connection: Starting connection monitoring...")
            await components["error_handler"].start_monitoring()

            # Start result merger
            logger.info("🔀 AWS Dual Connection: Starting result merger...")
            await components["result_merger"].start()

            # Log test mode status
            if test_mode == "left_only":
                logger.info(
                    "🧪 AWS Dual Connection: TEST MODE - Only left channel (Source A) is active"
                )
            elif test_mode == "right_only":
                logger.info(
                    "🧪 AWS Dual Connection: TEST MODE - Only right channel (Source B) is active"
                )
            else:
                logger.info("✅ AWS Dual Connection: FULL MODE - Both channels active")

            # Split audio saving moved to AudioSaver component at pipeline level
            logger.info(
                "🎵 AWS Dual Connection: Audio saving now handled by AudioSaver component"
            )

            logger.info(
                "✅ AWS Dual Connection: Dual connection stream started successfully"
            )

        except Exception as e:
            logger.error(
                f"❌ AWS Dual Connection: Failed to start dual connection stream: {e}"
            )
            # Cleanup any partially initialized components
            await self._cleanup_dual_connection_components()
            raise AWSTranscribeError(
                f"Failed to start dual connection stream: {e}"
            ) from e

    async def _send_audio_dual_connection(self, audio_chunk: bytes) -> None:
        """
        Send audio data to active dual connection channels after splitting.

        Args:
            audio_chunk: Stereo audio chunk to be split and sent to active channels
        """
        if not self._dual_connection_components:
            logger.error("❌ AWS Dual Connection: Components not initialized")
            return

        components = self._dual_connection_components
        test_mode = self.dual_connection_test_mode

        try:
            # Log raw audio input for debugging
            self._log_raw_audio_input(audio_chunk)

            # Always split stereo audio to test the channel splitter
            split_result = components["channel_splitter"].split_stereo_chunk(
                audio_chunk
            )

            if not split_result.split_successful:
                logger.error(
                    f"❌ AWS Dual Connection: Channel splitting failed: {split_result.error_message}"
                )
                return

            # Send left channel audio based on test mode
            if test_mode in ["left_only", "full"] and components["left_provider"]:
                try:
                    await components["left_provider"].send_audio(
                        split_result.left_channel
                    )
                    # Record successful transmission for error handler
                    components["error_handler"].record_bytes_sent(
                        "left", len(split_result.left_channel)
                    )
                except Exception as e:
                    logger.error(
                        f"❌ AWS Dual Connection: Left channel send failed: {e}"
                    )
                    components["error_handler"].record_connection_failure("left", e)
            elif test_mode in ["left_only", "full"]:
                logger.debug(
                    "🧪 AWS Dual Connection: Left channel audio dropped (provider not active)"
                )

            # Send right channel audio based on test mode
            if test_mode in ["right_only", "full"] and components["right_provider"]:
                try:
                    await components["right_provider"].send_audio(
                        split_result.right_channel
                    )
                    # Record successful transmission for error handler
                    components["error_handler"].record_bytes_sent(
                        "right", len(split_result.right_channel)
                    )
                except Exception as e:
                    logger.error(
                        f"❌ AWS Dual Connection: Right channel send failed: {e}"
                    )
                    components["error_handler"].record_connection_failure("right", e)
            elif test_mode in ["right_only", "full"]:
                logger.debug(
                    "🧪 AWS Dual Connection: Right channel audio dropped (provider not active)"
                )

            # Log audio activity levels occasionally for debugging
            if hasattr(self, "_dual_audio_chunk_count"):
                self._dual_audio_chunk_count += 1
            else:
                self._dual_audio_chunk_count = 1

            if self._dual_audio_chunk_count % 100 == 0:  # Every 100 chunks
                active_channels = []
                if test_mode in ["left_only", "full"]:
                    active_channels.append(
                        f"Left: {split_result.left_metrics.activity_level}"
                    )
                if test_mode in ["right_only", "full"]:
                    active_channels.append(
                        f"Right: {split_result.right_metrics.activity_level}"
                    )

                logger.info(
                    f"🧪 AWS Dual Connection (#{self._dual_audio_chunk_count}): Test mode={test_mode}, Active channels: {', '.join(active_channels)}"
                )

                # Audio saving status moved to AudioSaver component
                # Note: Split audio saving is now handled by AudioSaver at pipeline level

        except Exception as e:
            logger.error(f"❌ AWS Dual Connection: Audio send failed: {e}")
            # Don't raise exception to avoid breaking the main audio loop

    async def _get_transcription_dual_connection(
        self,
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """
        Get transcription results from active dual connection channels with synchronization.

        Yields merged transcription results from active channels based on test mode.
        """
        if not self._dual_connection_components:
            logger.error("❌ AWS Dual Connection: Components not initialized")
            return

        components = self._dual_connection_components
        result_merger = components["result_merger"]
        test_mode = self.dual_connection_test_mode

        logger.info(
            f"🔊 AWS Dual Connection: Starting dual transcription generator in test mode: {test_mode}"
        )

        # Create tasks to collect results from active channels only
        async def collect_left_results():
            """Collect results from left channel provider."""
            if not components["left_provider"]:
                logger.info(
                    "🧪 AWS Dual Connection: Left channel collection SKIPPED (provider not active)"
                )
                return

            try:
                logger.info(
                    "🔊 AWS Dual Connection: Starting left channel result collection..."
                )
                async for result in components["left_provider"].get_transcription():
                    # Record result reception for error handler
                    components["error_handler"].record_result_received("left")
                    # Add to merger as left channel result
                    await result_merger.add_left_result(result)
                    logger.debug(
                        f"🧪 AWS Dual Connection: Left result: '{result.text}' (confidence: {result.confidence:.2f})"
                    )
            except Exception as e:
                logger.error(
                    f"❌ AWS Dual Connection: Left channel collection failed: {e}"
                )
                components["error_handler"].record_connection_failure("left", e)

        async def collect_right_results():
            """Collect results from right channel provider."""
            if not components["right_provider"]:
                logger.info(
                    "🧪 AWS Dual Connection: Right channel collection SKIPPED (provider not active)"
                )
                return

            try:
                logger.info(
                    "🔊 AWS Dual Connection: Starting right channel result collection..."
                )
                async for result in components["right_provider"].get_transcription():
                    # Record result reception for error handler
                    components["error_handler"].record_result_received("right")
                    # Add to merger as right channel result
                    await result_merger.add_right_result(result)
                    logger.debug(
                        f"🧪 AWS Dual Connection: Right result: '{result.text}' (confidence: {result.confidence:.2f})"
                    )
            except Exception as e:
                logger.error(
                    f"❌ AWS Dual Connection: Right channel collection failed: {e}"
                )
                components["error_handler"].record_connection_failure("right", e)

        # Start collection tasks based on test mode
        active_tasks = []
        if test_mode in ["left_only", "full"]:
            left_task = asyncio.create_task(collect_left_results())
            active_tasks.append(left_task)
        if test_mode in ["right_only", "full"]:
            right_task = asyncio.create_task(collect_right_results())
            active_tasks.append(right_task)

        if not active_tasks:
            logger.error(
                "❌ AWS Dual Connection: No active channels configured for result collection"
            )
            return

        logger.info(
            f"🧪 AWS Dual Connection: Started {len(active_tasks)} result collection tasks"
        )

        try:
            # Yield merged results as they become available
            async for merged_result in result_merger.get_merged_results():
                logger.info(
                    f"🧪 AWS Dual Connection: Test mode result: {merged_result.speaker_id}: '{merged_result.text}' (confidence: {merged_result.confidence:.2f})"
                )
                yield merged_result

        except asyncio.CancelledError:
            logger.info("🛑 AWS Dual Connection: Transcription collection cancelled")
            # Cancel active collection tasks
            for task in active_tasks:
                if not task.done():
                    task.cancel()
            raise
        except Exception as e:
            logger.error(
                f"❌ AWS Dual Connection: Transcription collection failed: {e}"
            )
            # Cancel active collection tasks on error
            for task in active_tasks:
                if not task.done():
                    task.cancel()
        finally:
            logger.info("🔊 AWS Dual Connection: Transcription generator stopped")

    async def _stop_dual_connection_stream(self) -> None:
        """
        Stop dual connection transcription streams and cleanup resources.
        """
        test_mode = self.dual_connection_test_mode
        logger.info(
            f"🛑 AWS Dual Connection: Stopping dual connection stream (test mode: {test_mode})..."
        )

        if not self._dual_connection_components:
            logger.debug("🛑 AWS Dual Connection: No components to stop")
            return

        components = self._dual_connection_components

        try:
            # Stop result merger first to prevent new results
            if components["result_merger"]:
                try:
                    logger.info("🛑 AWS Dual Connection: Stopping result merger...")
                    await components["result_merger"].stop()
                except Exception as e:
                    logger.warning(
                        f"⚠️ AWS Dual Connection: Error stopping result merger: {e}"
                    )

            # Stop active channel providers based on test mode
            if components["left_provider"] and test_mode in ["left_only", "full"]:
                try:
                    logger.info(
                        "🛑 AWS Dual Connection: Stopping left channel provider..."
                    )
                    await components["left_provider"].stop_stream()
                except Exception as e:
                    logger.warning(
                        f"⚠️ AWS Dual Connection: Error stopping left provider: {e}"
                    )
            elif components["left_provider"]:
                logger.info(
                    "🧪 AWS Dual Connection: Left provider was inactive (test mode)"
                )

            if components["right_provider"] and test_mode in ["right_only", "full"]:
                try:
                    logger.info(
                        "🛑 AWS Dual Connection: Stopping right channel provider..."
                    )
                    await components["right_provider"].stop_stream()
                except Exception as e:
                    logger.warning(
                        f"⚠️ AWS Dual Connection: Error stopping right provider: {e}"
                    )
            elif components["right_provider"]:
                logger.info(
                    "🧪 AWS Dual Connection: Right provider was inactive (test mode)"
                )

            # Stop error handler monitoring
            if components["error_handler"]:
                try:
                    logger.info("🛑 AWS Dual Connection: Stopping error handler...")
                    await components["error_handler"].stop_monitoring()
                except Exception as e:
                    logger.warning(
                        f"⚠️ AWS Dual Connection: Error stopping error handler: {e}"
                    )

            logger.info(
                "✅ AWS Dual Connection: Dual connection stream stopped successfully"
            )

        except Exception as e:
            logger.error(
                f"❌ AWS Dual Connection: Error during dual connection stop: {e}"
            )
            raise AWSTranscribeError(
                f"Failed to stop dual connection stream properly: {e}"
            ) from e
        finally:
            # Stop audio saving if active before cleanup
            if (
                self._dual_connection_components
                and self._dual_connection_components.get("channel_splitter")
                and self._dual_connection_components[
                    "channel_splitter"
                ].enable_audio_saving
            ):
                try:
                    save_stats = self._dual_connection_components[
                        "channel_splitter"
                    ].stop_audio_saving()
                    if save_stats:
                        logger.info(
                            "🎵 AWS Dual Connection: Split audio saving stopped during cleanup"
                        )
                except Exception as e:
                    logger.error(
                        f"❌ AWS Dual Connection: Error stopping split audio saving: {e}"
                    )

            # Always cleanup components references
            await self._cleanup_dual_connection_components()

    def _log_raw_audio_input(self, audio_chunk: bytes) -> None:
        """
        Log detailed information about raw audio input before channel splitting.

        Args:
            audio_chunk: Raw audio chunk received from PyAudio
        """
        if not hasattr(self, "_raw_audio_chunk_count"):
            self._raw_audio_chunk_count = 0
            self._raw_audio_total_bytes = 0
            self._raw_audio_start_time = time.time()

        self._raw_audio_chunk_count += 1
        self._raw_audio_total_bytes += len(audio_chunk)

        # Analyze raw audio chunk
        chunk_size = len(audio_chunk)
        current_time = time.time()
        elapsed_time = current_time - self._raw_audio_start_time

        # Calculate expected vs actual data rates
        expected_bytes_per_second = (
            16000 * 2 * 2
        )  # 16kHz * 2 channels * 2 bytes (int16)
        actual_bytes_per_second = (
            self._raw_audio_total_bytes / elapsed_time if elapsed_time > 0 else 0
        )

        # Analyze audio content
        audio_analysis = self._analyze_raw_audio_chunk(audio_chunk)

        # Log every 50 chunks for the first 500 chunks, then every 100
        log_interval = 50 if self._raw_audio_chunk_count <= 500 else 100

        if self._raw_audio_chunk_count % log_interval == 0:
            logger.info(f"📡 RAW AUDIO INPUT (chunk #{self._raw_audio_chunk_count}):")
            logger.info(f"   ⏱️  Elapsed time: {elapsed_time:.2f}s")
            logger.info(f"   📁 Chunk size: {chunk_size} bytes")
            logger.info(f"   📊 Total bytes: {self._raw_audio_total_bytes:,} bytes")
            logger.info(
                f"   📊 Data rate: {actual_bytes_per_second:,.0f} bytes/sec (expected: {expected_bytes_per_second:,})"
            )

            if audio_analysis:
                logger.info("   🔊 Audio analysis:")
                logger.info(
                    f"      - Max amplitude: {audio_analysis.get('max_amplitude', 'N/A')}"
                )
                logger.info(
                    f"      - Avg amplitude: {audio_analysis.get('avg_amplitude', 'N/A'):.1f}"
                )
                logger.info(
                    f"      - Is silent: {audio_analysis.get('is_silent', 'N/A')}"
                )
                logger.info(
                    f"      - Sample count: {audio_analysis.get('sample_count', 'N/A')}"
                )

                # Detailed channel analysis if available
                dual_analysis = audio_analysis.get("dual_channel_analysis", {})
                if dual_analysis.get("is_dual_channel"):
                    source_a = dual_analysis.get("source_a", {})
                    source_b = dual_analysis.get("source_b", {})
                    logger.info(
                        f"      - Left channel: {source_a.get('activity_level', 'N/A')} (max: {source_a.get('max_amplitude', 'N/A')})"
                    )
                    logger.info(
                        f"      - Right channel: {source_b.get('activity_level', 'N/A')} (max: {source_b.get('max_amplitude', 'N/A')})"
                    )
                else:
                    logger.warning("      - ⚠️  Audio appears to be MONO, not stereo!")

    def _analyze_raw_audio_chunk(self, audio_chunk: bytes) -> dict[str, Any] | None:
        """
        Analyze raw audio chunk to understand its characteristics.

        Args:
            audio_chunk: Raw audio data bytes

        Returns:
            Dictionary with audio analysis results
        """
        try:
            # Use the same analysis method as the channel splitter for consistency
            return self._analyze_audio_content(audio_chunk)
        except Exception as e:
            logger.error(f"❌ RAW AUDIO: Analysis failed: {e}")
            return None

    async def _cleanup_dual_connection_components(self) -> None:
        """
        Cleanup dual connection components and reset state.
        """
        logger.info("🧹 AWS Dual Connection: Cleaning up components...")

        if self._dual_connection_components:
            components = self._dual_connection_components

            # Clear component references
            components["left_provider"] = None
            components["right_provider"] = None
            components["left_queue"] = None
            components["right_queue"] = None
            # Keep channel_splitter, result_merger, error_handler for reuse

            logger.info("🧹 AWS Dual Connection: Components cleaned up")

        # Reset connection mode
        self._connection_mode = None

        logger.info("✅ AWS Dual Connection: Cleanup completed")

    # ===================================================================
    # Intelligent Fallback Logic
    # ===================================================================

    def _should_fallback_to_dual_connection(
        self, audio_analysis: dict[str, Any]
    ) -> bool:
        """
        Determine if we should fallback from single to dual connection mode.

        Args:
            audio_analysis: Analysis results from _analyze_dual_channel_audio

        Returns:
            bool: True if fallback to dual connection is recommended
        """
        if not self.dual_fallback_enabled:
            return False

        if self._connection_mode == "dual_connection":
            return False  # Already in dual connection mode

        # Only check fallback conditions periodically to avoid spam
        if not hasattr(self, "_last_fallback_check"):
            self._last_fallback_check = 0.0
            self._fallback_check_interval = 30.0  # Only check every 30 seconds

        current_time = time.time()
        if current_time - self._last_fallback_check < self._fallback_check_interval:
            return False  # Too soon to check again

        self._last_fallback_check = current_time

        # Check for severe channel imbalance in dual-channel audio analysis
        dual_channel = audio_analysis.get("dual_channel_analysis", {})
        if dual_channel.get("is_dual_channel"):
            balance = dual_channel.get("balance", {})

            # Check for severe channel imbalance
            balance_ratio = balance.get("balance_ratio", 0.5)
            imbalance_ratio = abs(balance_ratio - 0.5)  # Perfect balance is 0.5

            if imbalance_ratio > self.channel_balance_threshold:
                logger.warning(
                    f"⚠️ AWS Fallback: Severe channel imbalance detected: {balance_ratio:.3f} (threshold: {self.channel_balance_threshold})"
                )
                logger.warning(
                    "⚠️ AWS Fallback: Will attempt fallback to dual connection after this send_audio cycle"
                )
                return True

            # Check for one channel being completely silent
            source_a = dual_channel.get("source_a", {})
            source_b = dual_channel.get("source_b", {})

            source_a_silent = source_a.get("is_silent", False)
            source_b_silent = source_b.get("is_silent", False)

            if source_a_silent and not source_b_silent:
                logger.warning(
                    "⚠️ AWS Fallback: Source A (left) is silent while Source B (right) has audio"
                )
                logger.warning(
                    "⚠️ AWS Fallback: Will attempt fallback to dual connection after this send_audio cycle"
                )
                return True
            if source_b_silent and not source_a_silent:
                logger.warning(
                    "⚠️ AWS Fallback: Source B (right) is silent while Source A (left) has audio"
                )
                logger.warning(
                    "⚠️ AWS Fallback: Will attempt fallback to dual connection after this send_audio cycle"
                )
                return True

        # REMOVED: Transcription quality check as AWS often returns 0.000 confidence for valid results
        # This was causing the fallback to trigger continuously

        return False

    async def _attempt_fallback_to_dual_connection(
        self, audio_config: AudioConfig
    ) -> bool:
        """
        Attempt to fallback from single to dual connection mode.

        Args:
            audio_config: Current audio configuration

        Returns:
            bool: True if fallback was successful, False otherwise
        """
        if not self.dual_fallback_enabled:
            logger.info("🔄 AWS Fallback: Dual fallback disabled, cannot switch")
            return False

        if self._connection_mode == "dual_connection":
            logger.info("🔄 AWS Fallback: Already in dual connection mode")
            return True

        if audio_config.channels < 2:
            logger.warning(
                "⚠️ AWS Fallback: Cannot fallback to dual connection with mono audio"
            )
            return False

        logger.warning(
            "🔄 AWS Fallback: Attempting fallback from single to dual connection..."
        )

        try:
            # Stop current single connection stream
            logger.info("🛑 AWS Fallback: Stopping single connection stream...")
            await self._stop_single_connection_stream()

            # Switch connection mode
            self._connection_mode = "dual_connection"
            logger.info("🔄 AWS Fallback: Switched to dual connection mode")

            # Start dual connection stream
            logger.info("🚀 AWS Fallback: Starting dual connection stream...")
            await self._start_dual_connection_stream(audio_config)

            logger.info(
                "✅ AWS Fallback: Successfully switched to dual connection mode"
            )
            return True

        except Exception as e:
            logger.error(f"❌ AWS Fallback: Failed to switch to dual connection: {e}")

            # Attempt to restore single connection
            try:
                logger.info(
                    "🔄 AWS Fallback: Attempting to restore single connection..."
                )
                self._connection_mode = "single_connection"
                await self._start_single_connection_stream(audio_config)
                logger.info(
                    "✅ AWS Fallback: Restored single connection after failed fallback"
                )
            except Exception as restore_error:
                logger.error(
                    f"❌ AWS Fallback: Failed to restore single connection: {restore_error}"
                )

            return False

    def _track_transcription_quality(self, result: TranscriptionResult) -> None:
        """
        Track transcription result quality for fallback decision making.

        Args:
            result: Transcription result to analyze
        """
        # DISABLED: Quality tracking for fallback decisions
        # AWS Transcribe often returns 0.000 confidence even for valid transcriptions,
        # causing aggressive fallback triggering. We now rely on audio-level analysis only.

        # Just log the result quality for debugging without using it for fallback decisions
        if result.confidence is not None and result.confidence > 0.0:
            logger.debug(
                f"📊 AWS Quality: Got result with confidence {result.confidence:.3f}: '{result.text}'"
            )
        elif result.text.strip():
            logger.debug(
                f"📊 AWS Quality: Got result with 0.000 confidence (normal for AWS): '{result.text}'"
            )

    def _reset_fallback_tracking(self) -> None:
        """Reset fallback-related tracking variables."""
        if hasattr(self, "_recent_result_quality"):
            self._recent_result_quality.clear()
        logger.debug("🔄 AWS Fallback: Reset fallback tracking")
