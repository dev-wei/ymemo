"""Azure Speech Service provider implementation."""

import asyncio
import logging
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Callable

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    speechsdk = None
    logging.warning(
        "Azure Speech SDK not available. Install azure-cognitiveservices-speech to use Azure provider."
    )

from ...core.interfaces import AudioConfig, TranscriptionProvider, TranscriptionResult
from ...utils.exceptions import (
    AzureSpeechAuthenticationError,
    AzureSpeechConfigurationError,
    AzureSpeechConnectionError,
)

logger = logging.getLogger(__name__)


class AzureSpeechProvider(TranscriptionProvider):
    """Azure Speech Service transcription provider."""

    def __init__(
        self,
        speech_key: str,
        region: str = "eastus",
        language_code: str = "en-US",
        endpoint: str | None = None,
        enable_speaker_diarization: bool = False,
        max_speakers: int = 4,
        timeout: int = 30,
    ):
        """Initialize Azure Speech Service provider.

        Args:
            speech_key: Azure Speech Service API key
            region: Azure region (e.g., 'eastus', 'westus2')
            language_code: Language code (e.g., 'en-US', 'es-ES')
            endpoint: Custom endpoint URL (optional)
            enable_speaker_diarization: Enable speaker identification
            max_speakers: Maximum number of speakers to detect
            timeout: Connection timeout in seconds
        """
        if speechsdk is None:
            raise AzureSpeechConfigurationError(
                "Azure Speech SDK not available. Install with: pip install azure-cognitiveservices-speech"
            )

        if not speech_key:
            raise AzureSpeechAuthenticationError("Azure Speech Service key is required")

        self.speech_key = speech_key
        self.region = region
        self.language_code = language_code
        self.endpoint = endpoint
        self.enable_speaker_diarization = enable_speaker_diarization
        self.max_speakers = max_speakers
        self.timeout = timeout

        # Azure Speech Service components
        self.speech_config = None
        self.audio_config = None
        self.speech_recognizer = None
        self.push_stream = None

        # Async event handling
        self.result_queue = asyncio.Queue()
        self._recognizing_lock = threading.Lock()
        self._is_connected = False
        self._is_running = False
        self._stop_event = threading.Event()

        # Connection health monitoring
        self.last_result_time = 0.0
        self.connection_health_callback: Callable[[bool, str], None] | None = None
        self.retry_count = 0
        self.max_retries = 3

        # Track utterances for proper partial result handling
        self.active_utterances: dict[str, int] = {}
        self.result_to_utterance: dict[str, str] = {}
        self.utterance_counter = 0

        logger.info(
            f"🔵 AzureSpeechProvider initialized: region={region}, language={language_code}, diarization={enable_speaker_diarization}"
        )

    def set_connection_health_callback(
        self, callback: Callable[[bool, str], None]
    ) -> None:
        """Set callback for connection health notifications."""
        self.connection_health_callback = callback

    async def start_stream(self, audio_config: AudioConfig) -> None:
        """Start the Azure Speech recognition stream."""
        try:
            logger.info(
                f"🚀 Starting Azure Speech stream (language: {self.language_code}, sample_rate: {audio_config.sample_rate})"
            )

            # Create speech configuration
            if self.endpoint:
                self.speech_config = speechsdk.SpeechConfig(
                    subscription=self.speech_key, endpoint=self.endpoint
                )
            else:
                self.speech_config = speechsdk.SpeechConfig(
                    subscription=self.speech_key, region=self.region
                )

            # Set language and audio format
            self.speech_config.speech_recognition_language = self.language_code
            self.speech_config.set_property(
                speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
                "Continuous",
            )

            # Enable speaker diarization if requested
            if self.enable_speaker_diarization:
                self.speech_config.set_property(
                    speechsdk.PropertyId.SpeechServiceConnection_SpeakerDiarizationMode,
                    "Identity",
                )
                logger.info(
                    f"🎙️ Azure speaker diarization enabled with max {self.max_speakers} speakers"
                )

            # Create push audio stream for real-time audio
            stream_format = speechsdk.AudioStreamFormat(
                samples_per_second=audio_config.sample_rate,
                bits_per_sample=16,
                channels=audio_config.channels,
            )
            self.push_stream = speechsdk.audio.PushAudioInputStream(stream_format)
            self.audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)

            # Create speech recognizer
            self.speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config, audio_config=self.audio_config
            )

            # Set up event handlers
            self._setup_event_handlers()

            # Start continuous recognition
            self.speech_recognizer.start_continuous_recognition()
            self._is_connected = True
            self._is_running = True
            self.last_result_time = time.time()

            if self.connection_health_callback:
                self.connection_health_callback(True, "Azure Speech Service connected")

            logger.info("✅ Azure Speech stream connection established")

        except Exception as e:
            logger.error(f"❌ Failed to start Azure Speech stream: {e}")
            await self._handle_error(e, "Failed to start Azure Speech stream")
            raise AzureSpeechConnectionError(
                f"Failed to start Azure Speech stream: {e}"
            ) from e

    def _setup_event_handlers(self):
        """Set up Azure Speech Service event handlers."""

        def recognizing_handler(evt):
            """Handle intermediate recognition results (partial)."""
            try:
                if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech:
                    text = evt.result.text
                    if text.strip():
                        # Generate utterance ID and sequence number
                        result_id = str(uuid.uuid4())

                        if result_id not in self.active_utterances:
                            self.utterance_counter += 1
                            utterance_id = f"utterance_{self.utterance_counter}"
                            self.active_utterances[result_id] = 0
                            self.result_to_utterance[result_id] = utterance_id
                        else:
                            utterance_id = self.result_to_utterance[result_id]

                        self.active_utterances[result_id] += 1
                        sequence_number = self.active_utterances[result_id]

                        # Extract speaker information if available
                        speaker_id = self._extract_speaker_id(evt.result)

                        transcription_result = TranscriptionResult(
                            text=text,
                            speaker_id=speaker_id,
                            confidence=0.0,  # Azure doesn't provide confidence for partial results
                            start_time=0.0,
                            end_time=0.0,
                            is_partial=True,
                            result_id=result_id,
                            utterance_id=utterance_id,
                            sequence_number=sequence_number,
                        )

                        # Put result in queue for async consumption
                        asyncio.create_task(self._queue_result(transcription_result))
                        logger.debug(
                            f"🔵 Azure partial result: '{text}' (speaker: {speaker_id})"
                        )

            except Exception as e:
                logger.error(f"❌ Error in recognizing handler: {e}")

        def recognized_handler(evt):
            """Handle final recognition results."""
            try:
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    text = evt.result.text
                    if text.strip():
                        # Generate final result
                        result_id = str(uuid.uuid4())
                        self.utterance_counter += 1
                        utterance_id = f"utterance_{self.utterance_counter}"

                        # Clean up any partial results for this utterance
                        if result_id in self.active_utterances:
                            del self.active_utterances[result_id]
                            del self.result_to_utterance[result_id]

                        # Extract speaker information
                        speaker_id = self._extract_speaker_id(evt.result)

                        # Get confidence if available
                        confidence = self._extract_confidence(evt.result)

                        transcription_result = TranscriptionResult(
                            text=text,
                            speaker_id=speaker_id,
                            confidence=confidence,
                            start_time=0.0,  # Azure provides timing in different format
                            end_time=0.0,
                            is_partial=False,
                            result_id=result_id,
                            utterance_id=utterance_id,
                            sequence_number=1,
                        )

                        # Update connection health
                        self.last_result_time = time.time()

                        # Put result in queue for async consumption
                        asyncio.create_task(self._queue_result(transcription_result))
                        logger.info(
                            f"💬 Azure final result: '{text}' (speaker: {speaker_id}, confidence: {confidence:.2f})"
                        )

                elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                    logger.debug("🔵 Azure: No speech could be recognized")

            except Exception as e:
                logger.error(f"❌ Error in recognized handler: {e}")

        def session_stopped_handler(evt):
            """Handle session stopped events."""
            logger.info("🛑 Azure Speech session stopped")
            self._is_connected = False
            if self.connection_health_callback:
                self.connection_health_callback(False, "Azure Speech session stopped")

        def canceled_handler(evt):
            """Handle cancellation events."""
            logger.warning(
                f"🚫 Azure Speech recognition canceled: {evt.result.cancellation_details}"
            )
            self._is_connected = False
            error_message = (
                f"Recognition canceled: {evt.result.cancellation_details.reason}"
            )
            if self.connection_health_callback:
                self.connection_health_callback(False, error_message)

        # Connect event handlers
        self.speech_recognizer.recognizing.connect(recognizing_handler)
        self.speech_recognizer.recognized.connect(recognized_handler)
        self.speech_recognizer.session_stopped.connect(session_stopped_handler)
        self.speech_recognizer.canceled.connect(canceled_handler)

    def _extract_speaker_id(self, result) -> str | None:
        """Extract speaker ID from Azure result."""
        try:
            if self.enable_speaker_diarization and hasattr(result, "speaker_id"):
                speaker_id = result.speaker_id
                if speaker_id:
                    # Convert Azure format to user-friendly format
                    if speaker_id.startswith("Speaker"):
                        return speaker_id
                    # Assume numeric format and convert
                    return f"Speaker {int(speaker_id) + 1}"
            return None
        except Exception as e:
            logger.debug(f"Could not extract speaker ID: {e}")
            return None

    def _extract_confidence(self, result) -> float:
        """Extract confidence score from Azure result."""
        try:
            # Azure confidence is typically available in detailed results
            # This is a simplified implementation
            return 0.9  # Default confidence for Azure results
        except Exception as e:
            logger.debug(f"Could not extract confidence: {e}")
            return 0.0

    async def _queue_result(self, result: TranscriptionResult):
        """Queue transcription result for async consumption."""
        try:
            await self.result_queue.put(result)
        except Exception as e:
            logger.error(f"❌ Error queuing result: {e}")

    async def send_audio(self, audio_chunk: bytes) -> None:
        """Send audio data to Azure Speech Service."""
        if not self._is_running or not self.push_stream:
            logger.warning("⚠️ Cannot send audio - Azure Speech stream not running")
            return

        try:
            # Send audio to push stream
            self.push_stream.write(audio_chunk)
            logger.debug(
                f"📡 Sent audio chunk to Azure Speech: {len(audio_chunk)} bytes"
            )

        except Exception as e:
            logger.error(f"❌ Failed to send audio to Azure Speech: {e}")
            if self._is_connected:
                self._is_connected = False
                if self.connection_health_callback:
                    self.connection_health_callback(
                        False, f"Audio send error: {str(e)}"
                    )
            raise AzureSpeechConnectionError(f"Failed to send audio: {e}") from e

    async def get_transcription(self) -> AsyncGenerator[TranscriptionResult, None]:
        """Get transcription results as they become available."""
        while self._is_running or not self.result_queue.empty():
            try:
                # Wait for results with timeout to allow for graceful shutdown
                result = await asyncio.wait_for(self.result_queue.get(), timeout=0.1)
                yield result
            except TimeoutError:
                # Continue polling for results
                continue
            except asyncio.CancelledError:
                logger.info("🛑 Azure Speech: Transcription generator cancelled")
                break
            except Exception as e:
                logger.error(
                    f"❌ Azure Speech: Error getting transcription result: {e}"
                )
                break

    async def stop_stream(self) -> None:
        """Stop the transcription stream and cleanup resources."""
        logger.info("🛑 Azure Speech: Stopping stream...")

        try:
            self._is_running = False
            self._stop_event.set()

            # Stop speech recognition
            if self.speech_recognizer:
                try:
                    self.speech_recognizer.stop_continuous_recognition()
                    logger.info("✅ Azure Speech: Recognition stopped")
                except Exception as e:
                    logger.warning(f"⚠️ Azure Speech: Error stopping recognition: {e}")

            # Close push stream
            if self.push_stream:
                try:
                    self.push_stream.close()
                    logger.info("✅ Azure Speech: Push stream closed")
                except Exception as e:
                    logger.warning(f"⚠️ Azure Speech: Error closing push stream: {e}")

            logger.info("✅ Azure Speech: Stream stopped successfully")

        except Exception as e:
            logger.error(f"❌ Azure Speech: Error stopping stream: {e}")
        finally:
            # Always clear references
            self.speech_recognizer = None
            self.push_stream = None
            self.audio_config = None
            self.speech_config = None
            self._is_connected = False

            logger.info("🛑 Azure Speech: Cleanup completed")

    async def _handle_error(self, error: Exception, context: str):
        """Handle errors and notify via callback."""
        if self.connection_health_callback:
            self.connection_health_callback(False, f"{context}: {str(error)}")

        # Mark as disconnected
        self._is_connected = False
