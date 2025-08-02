"""PyAudio-based audio capture provider implementation."""

import asyncio
import logging
import queue
import threading
from collections.abc import AsyncGenerator

import pyaudio

from ...core.interfaces import AudioCaptureProvider, AudioConfig
from ...utils.exceptions import AudioCaptureError

logger = logging.getLogger(__name__)


class PyAudioCaptureProvider(AudioCaptureProvider):
    """
    PyAudio-based audio capture implementation.

    This provider uses PyAudio to capture audio from system microphones
    for real-time transcription processing.
    """

    def __init__(self, device_index: int | None = None):
        """
        Initialize PyAudio capture provider.

        Args:
            device_index: Specific audio device index to use (default: None, uses system default)

        Raises:
            AudioCaptureError: If PyAudio initialization fails
        """
        # Validate parameters
        if device_index is not None and not isinstance(device_index, int):
            raise ValueError("device_index must be an integer or None")
        if device_index is not None and device_index < 0:
            raise ValueError("device_index must be non-negative")

        # Store configuration
        self.default_device_index = device_index

        # Initialize state
        self.audio = None
        self.stream = None
        self.audio_queue = queue.Queue()  # Use thread-safe queue
        self._capture_thread = None
        self._stop_event = threading.Event()
        self._is_active = False  # Track active state

        # Store source channels for direct audio streaming
        self._source_channels = None  # Will be set from audio config

        # Instance tracking for debugging
        self._instance_id = id(self)
        logger.info(
            f"🏗️ PyAudio: Created new instance {self._instance_id} with default_device={device_index}"
        )

        # Validate PyAudio availability early
        try:
            self._validate_pyaudio_availability()
        except Exception as e:
            logger.error(f"❌ PyAudio: Initialization validation failed: {e}")
            raise AudioCaptureError(f"PyAudio initialization failed: {e}") from e

    def _validate_pyaudio_availability(self) -> None:
        """
        Validate that PyAudio is available and working.

        Raises:
            AudioCaptureError: If PyAudio is not available or not working
        """
        try:
            # Test PyAudio initialization
            test_audio = pyaudio.PyAudio()
            test_audio.terminate()  # Clean up immediately
            logger.debug("✅ PyAudio: Availability validation successful")
        except Exception as e:
            raise AudioCaptureError(f"PyAudio not available or not working: {e}") from e

    async def _optimize_config_for_device(
        self, audio_config: AudioConfig, device_id: int | None
    ) -> AudioConfig:
        """
        Optimize audio configuration for specific device capabilities.

        Args:
            audio_config: Original audio configuration
            device_id: Target device ID

        Returns:
            Optimized AudioConfig with device-appropriate settings
        """
        try:
            from ...utils.device_utils import validate_device_config

            # If no specific device, use original config
            if device_id is None:
                logger.info("🔧 PyAudio: No specific device ID, using original config")
                return audio_config

            # Validate and optimize configuration for the device
            validation_result = validate_device_config(
                device_index=device_id,
                channels=audio_config.channels,
                sample_rate=audio_config.sample_rate,
            )

            # Log any warnings
            for warning in validation_result["warnings"]:
                logger.warning(f"⚠️ PyAudio: {warning}")

            # Log device information
            device_info = validation_result["device_info"]
            if device_info:
                logger.info(
                    f"🎤 PyAudio: Target device info - {device_info['name']} "
                    f"(max {device_info['max_input_channels']} channels, "
                    f"default {int(device_info['default_sample_rate'])}Hz)"
                )

            # Create optimized config
            return AudioConfig(
                sample_rate=validation_result["sample_rate"],
                channels=validation_result["channels"],
                chunk_size=audio_config.chunk_size,
                format=audio_config.format,
            )

        except Exception as e:
            logger.error(f"❌ PyAudio: Config optimization failed: {e}")
            logger.info("🔧 PyAudio: Falling back to mono configuration")
            # Fallback to safe mono configuration
            return AudioConfig(
                sample_rate=audio_config.sample_rate,
                channels=1,  # Safe fallback
                chunk_size=audio_config.chunk_size,
                format=audio_config.format,
            )

    async def start_capture(
        self, audio_config: AudioConfig, device_id: int | None = None
    ) -> None:
        """
        Start audio capture from specified device.

        Args:
            audio_config: Audio configuration for capture
            device_id: Specific device ID to use (overrides constructor default)

        Raises:
            AudioCaptureError: If capture initialization fails
            AudioDeviceError: If specified device is not available
            ValueError: If audio configuration is invalid
        """
        try:
            logger.info(f"🚀 PyAudio: Starting capture with config: {audio_config}")
            logger.info(
                f"🚀 PyAudio: Instance {self._instance_id} - Current active state: {self._is_active}"
            )

            # Check if already active - stop existing session first
            if self._is_active:
                logger.warning(
                    f"⚠️ PyAudio: Instance {self._instance_id} already active, stopping existing session first"
                )
                await self.stop_capture()

            # Validate audio configuration
            if not isinstance(audio_config, AudioConfig):
                raise ValueError("audio_config must be an AudioConfig instance")

            # Determine device to use
            target_device = (
                device_id if device_id is not None else self.default_device_index
            )

            logger.info(
                f"🎤 PyAudio: Initializing capture on device_id={target_device}"
            )

            # Note: Audio config should already be optimized by AudioProcessor
            logger.debug(
                f"🎛️ PyAudio: Using provided config: {audio_config.channels} channels, {audio_config.sample_rate}Hz"
            )

            # Store source channels for audio processing
            self._source_channels = audio_config.channels

            # Initialize PyAudio if not already done
            if not self.audio:
                self.audio = pyaudio.PyAudio()

            # Configure audio format
            format_map = {
                "int16": pyaudio.paInt16,
                "int24": pyaudio.paInt24,
                "int32": pyaudio.paInt32,
                "float32": pyaudio.paFloat32,
            }

            audio_format = format_map.get(audio_config.format, pyaudio.paInt16)

            # Open audio stream
            self.stream = self.audio.open(
                format=audio_format,
                channels=audio_config.channels,
                rate=audio_config.sample_rate,
                input=True,
                input_device_index=device_id,
                frames_per_buffer=audio_config.chunk_size,
                stream_callback=None,  # We'll use blocking read
            )

            # Create fresh stop event for this session (critical for thread safety)
            self._stop_event = threading.Event()
            logger.info(
                f"🎤 PyAudio: Created fresh stop event with ID: {id(self._stop_event)}"
            )

            # Start capture thread
            self._capture_thread = threading.Thread(
                target=self._capture_audio_thread, args=(audio_config.chunk_size,)
            )
            self._capture_thread.daemon = True
            self._capture_thread.start()

            # Mark as active
            self._is_active = True

            logger.info(
                f"🎤 PyAudio: Audio capture started - Instance: {self._instance_id}, Device: {device_id}, "
                f"Sample Rate: {audio_config.sample_rate}Hz, "
                f"Channels: {audio_config.channels}"
            )
            logger.info(
                f"🎤 PyAudio: Capture thread started - Instance: {self._instance_id}, Thread: {self._capture_thread.name}"
            )

        except Exception as e:
            error_msg = str(e)

            # Enhanced error messages for common issues
            if "Invalid number of channels" in error_msg or "-9998" in error_msg:
                from ...utils.device_utils import get_device_max_channels

                try:
                    max_channels = (
                        get_device_max_channels(device_id)
                        if device_id is not None
                        else "unknown"
                    )
                    enhanced_msg = (
                        f"Audio device channel mismatch: Requested {audio_config.channels} channels, "
                        f"but device {device_id} supports maximum {max_channels} channels. "
                        f"Original error: {error_msg}"
                    )
                except Exception:
                    enhanced_msg = f"Audio device channel mismatch: {error_msg}"

                logger.error(f"❌ PyAudio: {enhanced_msg}")
            elif "Invalid device" in error_msg or "-9996" in error_msg:
                enhanced_msg = f"Audio device not available: Device {device_id} may be disconnected or in use by another application. Original error: {error_msg}"
                logger.error(f"❌ PyAudio: {enhanced_msg}")
            elif "Invalid sample rate" in error_msg or "-9997" in error_msg:
                enhanced_msg = f"Audio sample rate not supported: Device {device_id} doesn't support {audio_config.sample_rate}Hz. Original error: {error_msg}"
                logger.error(f"❌ PyAudio: {enhanced_msg}")
            else:
                enhanced_msg = f"Audio capture initialization failed: {error_msg}"
                logger.error(f"❌ PyAudio: {enhanced_msg}")

            await self._cleanup()
            raise AudioCaptureError(enhanced_msg) from e

    def _capture_audio_thread(self, chunk_size: int) -> None:
        """Background thread for audio capture."""
        try:
            audio_chunk_count = 0
            logger.info(
                f"🎤 PyAudio Thread: Starting audio capture thread (chunk size: {chunk_size})"
            )
            logger.info(f"🎤 PyAudio Thread: Instance ID: {self._instance_id}")
            logger.info(
                f"🎤 PyAudio Thread: Stop event object ID: {id(self._stop_event)}"
            )
            logger.info(
                f"🎤 PyAudio Thread: Thread name: {threading.current_thread().name}"
            )

            while not self._stop_event.is_set() and self.stream:
                try:
                    # Check if stream is still active before reading
                    if not self.stream.is_active():
                        logger.info(
                            "🎤 Stream no longer active, stopping capture thread"
                        )
                        break

                    # Check stop event before potentially blocking read
                    if self._stop_event.is_set():
                        logger.info(
                            "🛑 PyAudio Thread: Stop event detected before read, breaking"
                        )
                        break

                    # Double-check stream is still valid
                    if not self.stream:
                        logger.info(
                            "🛑 PyAudio Thread: Stream reference cleared, breaking"
                        )
                        break

                    # Read audio data (this is the blocking call)
                    # Use non-blocking read to allow stop event checking
                    try:
                        # Use a smaller chunk size for more responsive stopping
                        audio_data = self.stream.read(
                            chunk_size, exception_on_overflow=False
                        )
                    except Exception as e:
                        # If stream is closed or stopped, read will throw exception
                        if self._stop_event.is_set():
                            logger.info(
                                "🛑 PyAudio Thread: Stream read exception after stop event, breaking"
                            )
                            break
                        logger.error(f"❌ PyAudio Thread: Stream read error: {e}")
                        break

                    audio_chunk_count += 1

                    # Check stop event after reading (critical check)
                    if self._stop_event.is_set():
                        logger.info(
                            "🛑 PyAudio Thread: Stop event detected after read, breaking"
                        )
                        break

                    # Send audio data directly without any channel processing
                    # 1-channel devices: Send mono audio to AWS Transcribe
                    # 2-channel devices: Send stereo audio to AWS Transcribe for dual-channel processing

                    # Put data in queue (thread-safe) - only if not stopping
                    if not self._stop_event.is_set():
                        self.audio_queue.put(audio_data)
                    else:
                        logger.info(
                            "🛑 PyAudio Thread: Stop event detected, not queuing audio data"
                        )
                        break

                    # Log every 100 chunks to avoid spam
                    if audio_chunk_count % 100 == 0:
                        logger.info(
                            f"🎤 PyAudio Thread: Captured {audio_chunk_count} audio chunks ({len(audio_data)} bytes each)"
                        )
                        logger.info(
                            f"🎤 PyAudio Thread: Instance {self._instance_id} - Stop event state at chunk {audio_chunk_count}: {self._stop_event.is_set()}"
                        )
                        logger.info(
                            f"🎤 PyAudio Thread: Stop event object ID: {id(self._stop_event)}"
                        )

                except Exception as e:
                    if not self._stop_event.is_set():
                        logger.error(
                            f"❌ PyAudio Thread: Error reading audio data: {e}"
                        )
                    else:
                        logger.info(
                            "🛑 PyAudio Thread: Exception during read after stop event - expected behavior"
                        )
                    break

        except Exception as e:
            logger.error(f"❌ PyAudio Thread: Audio capture thread error: {e}")
        finally:
            logger.info(
                f"🎤 PyAudio Thread: Audio capture thread stopped after {audio_chunk_count} chunks"
            )
            logger.info(
                f"🎤 PyAudio Thread: Instance {self._instance_id} - Final stop event state: {self._stop_event.is_set()}"
            )
            logger.info(
                f"🎤 PyAudio Thread: Final stop event object ID: {id(self._stop_event)}"
            )

    async def get_audio_stream(self) -> AsyncGenerator[bytes, None]:
        """Get audio data stream."""
        logger.info(
            f"🔊 PyAudio: Starting audio stream generator for instance {self._instance_id}"
        )
        logger.info(
            f"🔊 PyAudio: Stream generator - active: {self._is_active}, stop_event ID: {id(self._stop_event)}"
        )

        while self._is_active and not self._stop_event.is_set():
            try:
                # Wait for audio data with timeout (non-blocking)
                audio_data = self.audio_queue.get(timeout=0.1)

                # Triple-check before yielding
                if self._is_active and not self._stop_event.is_set():
                    yield audio_data
                else:
                    logger.debug(
                        f"🛑 PyAudio: Stop condition met (active: {self._is_active}, stop_event: {self._stop_event.is_set()}), breaking audio stream"
                    )
                    break

            except queue.Empty:
                # Continue polling with stop event check
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                continue
            except Exception as e:
                logger.error(f"Error in audio stream: {e}")
                break

        logger.info("🛑 PyAudio: Audio stream generator stopped")

    async def stop_capture(self) -> None:
        """Stop audio capture and cleanup resources."""
        logger.info(
            f"🛑 PyAudio: Stopping audio capture for instance {self._instance_id}..."
        )
        logger.info(
            f"🛑 PyAudio: Initial state - active: {self._is_active}, stream: {self.stream is not None}, thread: {self._capture_thread is not None if hasattr(self, '_capture_thread') else 'N/A'}"
        )
        logger.info(f"🛑 PyAudio: Stop event object ID: {id(self._stop_event)}")

        # If not active, nothing to stop
        if not self._is_active:
            logger.info(
                f"🛑 PyAudio: Instance {self._instance_id} not active, nothing to stop"
            )
            return

        # Log detailed capture thread info
        if hasattr(self, "_capture_thread") and self._capture_thread:
            logger.info(f"🛑 PyAudio: Capture thread name: {self._capture_thread.name}")
            logger.info(
                f"🛑 PyAudio: Capture thread is_alive: {self._capture_thread.is_alive()}"
            )
        else:
            logger.info("🛑 PyAudio: No capture thread found")

        # Signal stop to all components FIRST
        self._stop_event.set()
        logger.info("🛑 PyAudio: Stop event set")
        logger.info(f"🛑 PyAudio: Stop event is_set(): {self._stop_event.is_set()}")

        # Add a small delay to ensure thread sees the stop event
        await asyncio.sleep(0.1)

        # Immediately stop the PyAudio stream to interrupt any blocking reads
        try:
            if self.stream:
                if self.stream.is_active():
                    logger.info("🛑 PyAudio: Stopping active stream...")
                    self.stream.stop_stream()
                    logger.info("🛑 PyAudio: Stream stopped")
                else:
                    logger.info("🛑 PyAudio: Stream was already inactive")

                # Close the stream to make sure it's fully terminated
                logger.info("🛑 PyAudio: Closing stream...")
                self.stream.close()
                logger.info("🛑 PyAudio: Stream closed")
                # Set stream to None to ensure thread loop breaks
                self.stream = None
                logger.info("🛑 PyAudio: Stream reference cleared")
            else:
                logger.warning("⚠️ PyAudio: No stream to stop")
        except Exception as e:
            logger.error(f"❌ PyAudio: Error stopping/closing stream: {e}")
            import traceback

            traceback.print_exc()

        # Wait for capture thread to finish with timeout
        if (
            hasattr(self, "_capture_thread")
            and self._capture_thread
            and self._capture_thread.is_alive()
        ):
            logger.info(
                f"🛑 PyAudio: Waiting for capture thread to finish... (instance: {self._instance_id}, thread: {self._capture_thread.name})"
            )

            # Brief wait for normal termination
            self._capture_thread.join(timeout=0.2)
            if self._capture_thread.is_alive():
                logger.info(
                    "🛑 PyAudio: Capture thread still alive - abandoning as daemon thread"
                )
                logger.info(
                    f"🛑 PyAudio: Thread details: {self._capture_thread.name}, daemon: {self._capture_thread.daemon}"
                )
                # Don't wait longer - daemon threads will be cleaned up automatically
            else:
                logger.info("✅ PyAudio: Capture thread finished successfully")
        else:
            logger.info("🛑 PyAudio: No capture thread to wait for")

        # Clear thread reference immediately to prevent access
        self._capture_thread = None

        # Mark as inactive
        self._is_active = False

        await self._cleanup()
        logger.info("🛑 PyAudio: Stop capture complete")

    async def _cleanup(self) -> None:
        """Cleanup audio resources with improved safety."""
        try:
            # Stream cleanup (may already be done in stop_capture)
            if self.stream:
                try:
                    # Check if stream is still active before stopping
                    if hasattr(self.stream, "is_active") and self.stream.is_active():
                        logger.info("🛑 PyAudio: Stream is active, stopping...")
                        self.stream.stop_stream()

                    # Close the stream
                    if hasattr(self.stream, "close"):
                        logger.info("🛑 PyAudio: Closing stream...")
                        self.stream.close()

                    logger.info("🛑 PyAudio: Stream cleanup completed")
                except Exception as e:
                    logger.warning(f"⚠️ PyAudio: Error cleaning up stream: {e}")
                finally:
                    self.stream = None

            # PyAudio cleanup - this is often where segfaults occur
            if self.audio:
                try:
                    logger.info("🛑 PyAudio: Terminating PyAudio instance...")
                    # Add a small delay before termination to prevent race conditions
                    await asyncio.sleep(0.05)
                    self.audio.terminate()
                    logger.info("🛑 PyAudio: PyAudio instance terminated")
                except Exception as e:
                    logger.warning(f"⚠️ PyAudio: Error terminating audio: {e}")
                finally:
                    self.audio = None

            # Clear any remaining audio data in queue
            cleared_count = 0
            try:
                while not self.audio_queue.empty():
                    try:
                        self.audio_queue.get_nowait()
                        cleared_count += 1
                        # Prevent infinite loop
                        if cleared_count > 1000:
                            logger.warning(
                                "⚠️ PyAudio: Too many items in queue, stopping cleanup"
                            )
                            break
                    except queue.Empty:
                        break
            except Exception as e:
                logger.warning(f"⚠️ PyAudio: Error clearing queue: {e}")

            if cleared_count > 0:
                logger.info(
                    f"🛑 PyAudio: Cleared {cleared_count} remaining audio chunks from queue"
                )

        except Exception as e:
            logger.error(f"❌ PyAudio: Error during audio cleanup: {e}")
            # Don't re-raise - we want cleanup to always complete

    def list_audio_devices(self) -> dict[int, str]:
        """List available audio input devices."""
        devices = {}

        try:
            if not self.audio:
                self.audio = pyaudio.PyAudio()

            device_count = self.audio.get_device_count()

            for i in range(device_count):
                try:
                    device_info = self.audio.get_device_info_by_index(i)

                    # Only include input devices
                    if device_info["maxInputChannels"] > 0:
                        device_name = device_info["name"]
                        devices[i] = device_name

                except Exception as e:
                    logger.warning(f"Could not get info for device {i}: {e}")

        except Exception as e:
            logger.error(f"Error listing audio devices: {e}")

        return devices

    def is_active(self) -> bool:
        """Check if the provider is currently active."""
        return self._is_active
