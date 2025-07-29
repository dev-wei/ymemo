"""Real-time audio processing pipeline."""

import asyncio
import logging
from typing import Optional, Callable, List, Dict, Any
from datetime import datetime

from .interfaces import TranscriptionProvider, AudioCaptureProvider, AudioConfig, TranscriptionResult
from .factory import AudioProcessorFactory
from config.audio_config import get_config


logger = logging.getLogger(__name__)


class AudioProcessor:
    """Real-time audio processing pipeline coordinator."""
    
    def __init__(
        self,
        transcription_provider: str = 'aws',
        capture_provider: str = 'pyaudio',
        transcription_config: Optional[Dict[str, Any]] = None,
        capture_config: Optional[Dict[str, Any]] = None
    ):
        logger.info(f"🏗️  AudioProcessor: Initializing with transcription={transcription_provider}, capture={capture_provider}")
        logger.debug(f"🔧 AudioProcessor: Transcription config: {transcription_config}")
        logger.debug(f"🔧 AudioProcessor: Capture config: {capture_config}")
        
        self.transcription_provider_name = transcription_provider
        self.capture_provider_name = capture_provider
        self.transcription_config = transcription_config or {}
        self.capture_config = capture_config or {}
        
        # Providers
        self.transcription_provider: Optional[TranscriptionProvider] = None
        self.capture_provider: Optional[AudioCaptureProvider] = None
        
        # Configuration - get from system config or use default
        system_config = get_config()
        self.audio_config = system_config.get_audio_config()
        logger.debug(f"🎚️  AudioProcessor: Audio config - sample_rate={self.audio_config.sample_rate}, channels={self.audio_config.channels}, format={self.audio_config.format}")
        
        # State
        self.is_running = False
        self.transcription_callback: Optional[Callable[[TranscriptionResult], None]] = None
        self.error_callback: Optional[Callable[[Exception], None]] = None
        self.connection_health_callback: Optional[Callable[[bool, str], None]] = None
        
        # Tasks
        self._capture_task: Optional[asyncio.Task] = None
        self._transcription_task: Optional[asyncio.Task] = None
        
        # Session data
        self.session_transcripts: List[TranscriptionResult] = []
        self.current_meeting_id: Optional[str] = None
        
        logger.debug("✅ AudioProcessor: Initialization complete")
    
    async def initialize(self) -> None:
        """Initialize audio processing providers."""
        try:
            # Create transcription provider
            self.transcription_provider = AudioProcessorFactory.create_transcription_provider(
                self.transcription_provider_name,
                **self.transcription_config
            )
            
            # Create audio capture provider
            self.capture_provider = AudioProcessorFactory.create_audio_capture_provider(
                self.capture_provider_name,
                **self.capture_config
            )
            
            # Log provider instance details
            if hasattr(self.capture_provider, '_instance_id'):
                logger.info(f"🔧 AudioProcessor: Using capture provider instance {self.capture_provider._instance_id}")
            
            # Set up connection health monitoring for AWS Transcribe
            if hasattr(self.transcription_provider, 'set_connection_health_callback') and self.connection_health_callback:
                self.transcription_provider.set_connection_health_callback(self.connection_health_callback)
                logger.info("🔍 AudioProcessor: Connection health monitoring enabled")
            
            logger.info("Audio processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize audio processor: {e}")
            raise
    
    async def start_recording(self, device_id: Optional[int] = None) -> None:
        """Start real-time audio recording and transcription.
        
        Args:
            device_id: Optional specific audio device ID
        """
        if self.is_running:
            logger.warning("Audio processor is already running")
            return
        
        if not self.transcription_provider or not self.capture_provider:
            logger.debug("🔄 AudioProcessor: Initializing providers...")
            await self.initialize()
        
        try:
            # Start new session
            self.current_meeting_id = f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.session_transcripts.clear()
            logger.info(f"🆔 AudioProcessor: Created meeting session: {self.current_meeting_id}")
            
            # Start transcription stream
            logger.debug("🎯 AudioProcessor: Starting transcription stream...")
            await self.transcription_provider.start_stream(self.audio_config)
            
            # Start audio capture
            logger.debug("🎤 AudioProcessor: Starting audio capture...")
            await self.capture_provider.start_capture(self.audio_config, device_id)
            
            # Start processing tasks
            logger.debug("🔄 AudioProcessor: Creating async tasks...")
            self._capture_task = asyncio.create_task(self._audio_capture_loop())
            self._transcription_task = asyncio.create_task(self._transcription_loop())
            
            self.is_running = True
            logger.info(f"✅ AudioProcessor: Started recording for meeting: {self.current_meeting_id}")
            
            # Wait for tasks to complete (this keeps the function running)
            logger.debug("⏳ AudioProcessor: Waiting for processing tasks to complete...")
            try:
                await asyncio.gather(self._capture_task, self._transcription_task)
            except asyncio.CancelledError:
                logger.info("🛑 AudioProcessor: Processing tasks cancelled")
                raise
            except Exception as e:
                logger.error(f"❌ AudioProcessor: Error in processing tasks: {e}")
                raise
            
        except Exception as e:
            logger.error(f"❌ AudioProcessor: Failed to start recording: {e}")
            import traceback
            traceback.print_exc()
            await self.stop_recording()
            if self.error_callback:
                self.error_callback(e)
            raise
    
    async def stop_recording(self) -> None:
        """Stop audio recording and transcription."""
        logger.info("🛑 AudioProcessor: stop_recording() called")
        logger.info(f"🛑 AudioProcessor: Current is_running state: {self.is_running}")
        
        if not self.is_running:
            logger.debug("🛑 AudioProcessor: Already stopped, nothing to do")
            return
        
        logger.info("🛑 AudioProcessor: Stopping audio recording...")
        self.is_running = False
        
        try:
            # PRIORITY 1: Stop PyAudio capture provider immediately to prevent more audio data
            logger.info("🛑 AudioProcessor: Stopping PyAudio capture provider first...")
            try:
                if self.capture_provider:
                    logger.info(f"🛑 AudioProcessor: Stopping capture provider - Type: {type(self.capture_provider).__name__}")
                    logger.info(f"🛑 AudioProcessor: Capture provider state: {hasattr(self.capture_provider, '_stop_event')}")
                    
                    # Log provider instance details
                    if hasattr(self.capture_provider, '_instance_id'):
                        logger.info(f"🛑 AudioProcessor: Calling stop_capture() on provider instance {self.capture_provider._instance_id}")
                        if hasattr(self.capture_provider, '_stop_event'):
                            logger.info(f"🛑 AudioProcessor: Provider instance {self.capture_provider._instance_id} stop event ID: {id(self.capture_provider._stop_event)}")
                    
                    await self.capture_provider.stop_capture()
                    logger.info("🛑 AudioProcessor: Capture provider stopped")
                else:
                    logger.warning("⚠️ AudioProcessor: No capture provider to stop")
            except Exception as e:
                logger.error(f"❌ AudioProcessor: Error stopping capture provider: {e}")
                import traceback
                traceback.print_exc()
            
            # PRIORITY 2: Cancel capture task (should be quick now that provider is stopped)
            logger.info("🛑 AudioProcessor: Cancelling capture task...")
            if self._capture_task and not self._capture_task.done():
                logger.info("🛑 AudioProcessor: Cancelling capture task")
                self._capture_task.cancel()
                try:
                    # Wait for capture task with timeout to prevent hanging
                    await asyncio.wait_for(self._capture_task, timeout=1.0)
                    logger.info("🛑 AudioProcessor: Capture task cancelled successfully")
                except asyncio.CancelledError:
                    logger.info("🛑 AudioProcessor: Capture task cancelled")
                except asyncio.TimeoutError:
                    logger.warning("⚠️ AudioProcessor: Capture task cancellation timed out")
                except Exception as e:
                    logger.warning(f"⚠️ AudioProcessor: Error cancelling capture task: {e}")
            
            # Always clear the task reference to prevent "Task was destroyed but it is pending" error
            self._capture_task = None
            
            # PRIORITY 3: Handle transcription cleanup (may hang, but PyAudio is already stopped)
            logger.info("🛑 AudioProcessor: Handling transcription cleanup...")
            if self._transcription_task and not self._transcription_task.done():
                logger.info("🛑 AudioProcessor: Cancelling transcription task")
                self._transcription_task.cancel()
                try:
                    # Wait for transcription task with timeout to prevent hanging
                    await asyncio.wait_for(self._transcription_task, timeout=2.0)
                    logger.info("🛑 AudioProcessor: Transcription task cancelled successfully")
                except asyncio.CancelledError:
                    logger.info("🛑 AudioProcessor: Transcription task cancelled")
                except asyncio.TimeoutError:
                    logger.warning("⚠️ AudioProcessor: Transcription task cancellation timed out")
                except Exception as e:
                    logger.warning(f"⚠️ AudioProcessor: Error cancelling transcription task: {e}")
            
            # Always clear the task reference to prevent "Task was destroyed but it is pending" error
            self._transcription_task = None
            
            # PRIORITY 4: Stop transcription provider (may hang, but PyAudio is already stopped)
            try:
                if self.transcription_provider:
                    logger.info(f"🛑 AudioProcessor: Stopping transcription provider - Type: {type(self.transcription_provider).__name__}")
                    await asyncio.wait_for(self.transcription_provider.stop_stream(), timeout=2.0)
                    logger.info("🛑 AudioProcessor: Transcription provider stopped")
                else:
                    logger.warning("⚠️ AudioProcessor: No transcription provider to stop")
            except asyncio.TimeoutError:
                logger.warning("⚠️ AudioProcessor: Transcription provider stop timed out")
            except Exception as e:
                logger.error(f"❌ AudioProcessor: Error stopping transcription provider: {e}")
                import traceback
                traceback.print_exc()
            
            logger.info("✅ AudioProcessor: Audio recording stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ AudioProcessor: Error in stop_recording: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def _audio_capture_loop(self) -> None:
        """Main audio capture loop."""
        try:
            chunk_count = 0
            logger.info("🔄 AudioProcessor: Starting audio capture loop...")
            
            async for audio_chunk in self.capture_provider.get_audio_stream():
                if not self.is_running:
                    logger.info("🛑 AudioProcessor: is_running=False, breaking capture loop")
                    break
                
                chunk_count += 1
                
                # Send audio to transcription service
                await self.transcription_provider.send_audio(audio_chunk)
                
                # Check is_running after sending audio (in case stop was called)
                if not self.is_running:
                    logger.info("🛑 AudioProcessor: is_running=False after send_audio, breaking capture loop")
                    break
                
                # Log every 50 chunks to monitor flow
                if chunk_count % 50 == 0:
                    logger.info(f"🔄 AudioProcessor: Processed {chunk_count} audio chunks through transcription pipeline")
                
        except asyncio.CancelledError:
            logger.info("🛑 AudioProcessor: Audio capture loop cancelled")
            raise
        except Exception as e:
            logger.error(f"❌ AudioProcessor: Audio capture loop error: {e}")
            if self.error_callback:
                self.error_callback(e)
        finally:
            logger.info(f"🔄 AudioProcessor: Audio capture loop stopped after processing {chunk_count} chunks")
    
    async def _transcription_loop(self) -> None:
        """Main transcription processing loop."""
        try:
            transcription_count = 0
            logger.info("📝 AudioProcessor: Starting transcription processing loop...")
            
            async for result in self.transcription_provider.get_transcription():
                if not self.is_running:
                    logger.info("🛑 AudioProcessor: is_running=False, breaking transcription loop")
                    break
                
                transcription_count += 1
                
                # Store transcript
                self.session_transcripts.append(result)
                
                # Callback to UI
                if self.transcription_callback:
                    logger.info(f"📱 AudioProcessor: Sending transcription #{transcription_count} to UI: '{result.text}'")
                    self.transcription_callback(result)
                
                logger.info(f"📝 AudioProcessor: Transcription #{transcription_count}: {result.speaker_id or 'Unknown'}: '{result.text}' (confidence: {result.confidence:.2f})")
                
                # Check is_running after processing (in case stop was called)
                if not self.is_running:
                    logger.info("🛑 AudioProcessor: is_running=False after processing, breaking transcription loop")
                    break
                
        except asyncio.CancelledError:
            logger.info("🛑 AudioProcessor: Transcription loop cancelled")
            raise
        except Exception as e:
            logger.error(f"❌ AudioProcessor: Transcription loop error: {e}")
            if self.error_callback:
                self.error_callback(e)
        finally:
            logger.info(f"📝 Transcription loop stopped after processing {transcription_count} transcriptions")
    
    def set_transcription_callback(self, callback: Callable[[TranscriptionResult], None]) -> None:
        """Set callback function for new transcription results."""
        self.transcription_callback = callback
    
    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Set callback function for error handling."""
        self.error_callback = callback
    
    def set_connection_health_callback(self, callback: Callable[[bool, str], None]) -> None:
        """Set callback for connection health notifications.
        
        Args:
            callback: Function to call with (is_healthy, message) when connection status changes
        """
        self.connection_health_callback = callback
    
    def get_available_devices(self) -> Dict[int, str]:
        """Get list of available audio input devices."""
        if not self.capture_provider:
            # Create temporary provider to list devices
            logger.info("🔧 AudioProcessor: Creating temporary provider for device listing")
            temp_provider = AudioProcessorFactory.create_audio_capture_provider(
                self.capture_provider_name
            )
            if hasattr(temp_provider, '_instance_id'):
                logger.info(f"🔧 AudioProcessor: Using temporary provider instance {temp_provider._instance_id} for device listing")
            return temp_provider.list_audio_devices()
        
        logger.info(f"🔧 AudioProcessor: Using existing provider instance {getattr(self.capture_provider, '_instance_id', 'unknown')} for device listing")
        return self.capture_provider.list_audio_devices()
    
    def get_session_transcripts(self) -> List[TranscriptionResult]:
        """Get all transcripts from current session."""
        return self.session_transcripts.copy()
    
    def export_session(self) -> Dict[str, Any]:
        """Export current session data."""
        return {
            'meeting_id': self.current_meeting_id,
            'start_time': datetime.now().isoformat(),
            'transcripts': [
                {
                    'text': t.text,
                    'speaker_id': t.speaker_id,
                    'confidence': t.confidence,
                    'start_time': t.start_time,
                    'end_time': t.end_time,
                    'is_partial': t.is_partial
                }
                for t in self.session_transcripts
            ]
        }