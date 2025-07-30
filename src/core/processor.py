"""Real-time audio processing pipeline."""

import asyncio
import logging
from typing import Optional, Callable, List, Dict, Any
from datetime import datetime

from .interfaces import TranscriptionProvider, AudioCaptureProvider, AudioConfig, TranscriptionResult
from .factory import AudioProcessorFactory
from .pipeline_error_handler import PipelineErrorHandler, ErrorSeverity, RetryStrategy
from config.audio_config import get_config
from ..utils.exceptions import PipelineError, PipelineTimeoutError


logger = logging.getLogger(__name__)


class AudioProcessor:
    """Real-time audio processing pipeline coordinator."""
    
    def __init__(
        self,
        transcription_provider: str = 'aws',
        capture_provider: str = 'pyaudio',
        transcription_config: Optional[Dict[str, Any]] = None,
        capture_config: Optional[Dict[str, Any]] = None,
        error_handler_config: Optional[Dict[str, Any]] = None
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
        
        # Error handling
        error_config = error_handler_config or {}
        self.error_handler = PipelineErrorHandler(
            default_timeout=error_config.get('default_timeout', 30.0),
            max_retries=error_config.get('max_retries', 3),
            base_retry_delay=error_config.get('base_retry_delay', 1.0)
        )
        
        # Tasks
        self._capture_task: Optional[asyncio.Task] = None
        self._transcription_task: Optional[asyncio.Task] = None
        
        # Session data
        self.session_transcripts: List[TranscriptionResult] = []
        self.current_meeting_id: Optional[str] = None
        
        logger.debug("✅ AudioProcessor: Initialization complete")
    
    async def initialize(self) -> None:
        """Initialize audio processing providers with error handling."""
        async with self.error_handler.handle_pipeline_operation(
            "provider_initialization",
            timeout=15.0,
            severity=ErrorSeverity.CRITICAL
        ):
            try:
                # Create transcription provider
                logger.info(f"🏭 AudioProcessor: Creating transcription provider '{self.transcription_provider_name}'")
                self.transcription_provider = AudioProcessorFactory.create_transcription_provider(
                    self.transcription_provider_name,
                    **self.transcription_config
                )
                
                # Create audio capture provider  
                logger.info(f"🎤 AudioProcessor: Creating capture provider '{self.capture_provider_name}'")
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
                
                logger.info("✅ AudioProcessor: Provider initialization completed successfully")
                
            except Exception as e:
                logger.error(f"❌ AudioProcessor: Provider initialization failed: {e}")
                # Clean up any partially initialized providers
                await self._cleanup_providers()
                raise PipelineError(f"Failed to initialize audio processor providers: {e}") from e
    
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
            
            # Start transcription stream with error handling
            async with self.error_handler.handle_pipeline_operation(
                "transcription_start",
                timeout=10.0,
                severity=ErrorSeverity.HIGH,
                cleanup_callback=lambda: self._emergency_cleanup_transcription()
            ):
                logger.debug("🎯 AudioProcessor: Starting transcription stream...")
                await self.transcription_provider.start_stream(self.audio_config)
            
            # Start audio capture with error handling
            async with self.error_handler.handle_pipeline_operation(
                "audio_capture_start",
                timeout=8.0,
                severity=ErrorSeverity.HIGH,
                cleanup_callback=lambda: self._emergency_cleanup_capture()
            ):
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
            
        except (PipelineError, PipelineTimeoutError) as e:
            logger.error(f"❌ AudioProcessor: Pipeline error during recording start: {e}")
            await self.stop_recording()
            if self.error_callback:
                self.error_callback(e)
            raise
        except Exception as e:
            logger.error(f"❌ AudioProcessor: Unexpected error during recording start: {e}")
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
        
        # Use error handler for safe cleanup with individual timeouts
        cleanup_operations = {
            "capture_provider_stop": self._stop_capture_provider,
            "capture_task_cancel": self._cancel_capture_task,
            "transcription_task_cancel": self._cancel_transcription_task,
            "transcription_provider_stop": self._stop_transcription_provider
        }
        
        try:
            cleanup_results = await self.error_handler.safe_cleanup(
                cleanup_operations,
                timeout_per_operation=3.0
            )
            
            # Check if any critical cleanup failed
            failed_cleanups = [op for op, success in cleanup_results.items() if not success]
            
            if failed_cleanups:
                logger.warning(f"⚠️ AudioProcessor: Some cleanup operations failed: {failed_cleanups}")
                # Don't raise exception, as partial cleanup is better than no cleanup
            
            logger.info("✅ AudioProcessor: Audio recording stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ AudioProcessor: Error during stop_recording cleanup: {e}")
            # Log error summary for debugging
            error_summary = self.error_handler.get_error_summary()
            logger.error(f"📊 AudioProcessor: Error handler summary: {error_summary}")
            raise PipelineError(f"Failed to stop recording properly: {e}") from e
    
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
                self.error_callback(PipelineError(f"Audio capture loop failed: {e}", e))
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
                self.error_callback(PipelineError(f"Transcription loop failed: {e}", e))
        finally:
            logger.info(f"📝 Transcription loop stopped after processing {transcription_count} transcriptions")
    
    async def _stop_capture_provider(self) -> None:
        """Stop audio capture provider."""
        if not self.capture_provider:
            logger.warning("⚠️ AudioProcessor: No capture provider to stop")
            return
        
        logger.info(f"🛑 AudioProcessor: Stopping capture provider - Type: {type(self.capture_provider).__name__}")
        
        # Log provider instance details
        if hasattr(self.capture_provider, '_instance_id'):
            logger.info(f"🛑 AudioProcessor: Calling stop_capture() on provider instance {self.capture_provider._instance_id}")
        
        await self.capture_provider.stop_capture()
        logger.info("🛑 AudioProcessor: Capture provider stopped")
    
    async def _cancel_capture_task(self) -> None:
        """Cancel audio capture task."""
        if not self._capture_task or self._capture_task.done():
            logger.debug("🛑 AudioProcessor: No capture task to cancel")
            return
        
        logger.info("🛑 AudioProcessor: Cancelling capture task")
        self._capture_task.cancel()
        
        try:
            await asyncio.wait_for(self._capture_task, timeout=1.0)
            logger.info("🛑 AudioProcessor: Capture task cancelled successfully")
        except asyncio.CancelledError:
            logger.info("🛑 AudioProcessor: Capture task cancelled")
        except asyncio.TimeoutError:
            logger.warning("⚠️ AudioProcessor: Capture task cancellation timed out")
        finally:
            self._capture_task = None
    
    async def _cancel_transcription_task(self) -> None:
        """Cancel transcription processing task."""
        if not self._transcription_task or self._transcription_task.done():
            logger.debug("🛑 AudioProcessor: No transcription task to cancel")
            return
        
        logger.info("🛑 AudioProcessor: Cancelling transcription task")
        self._transcription_task.cancel()
        
        try:
            await asyncio.wait_for(self._transcription_task, timeout=2.0)
            logger.info("🛑 AudioProcessor: Transcription task cancelled successfully")
        except asyncio.CancelledError:
            logger.info("🛑 AudioProcessor: Transcription task cancelled")
        except asyncio.TimeoutError:
            logger.warning("⚠️ AudioProcessor: Transcription task cancellation timed out")
        finally:
            self._transcription_task = None
    
    async def _stop_transcription_provider(self) -> None:
        """Stop transcription provider."""
        if not self.transcription_provider:
            logger.warning("⚠️ AudioProcessor: No transcription provider to stop")
            return
        
        logger.info(f"🛑 AudioProcessor: Stopping transcription provider - Type: {type(self.transcription_provider).__name__}")
        await self.transcription_provider.stop_stream()
        logger.info("🛑 AudioProcessor: Transcription provider stopped")
    
    def _emergency_cleanup_transcription(self) -> None:
        """Emergency cleanup for transcription provider (non-async)."""
        logger.warning("🚨 AudioProcessor: Emergency transcription cleanup triggered")
        if hasattr(self.transcription_provider, 'emergency_stop'):
            self.transcription_provider.emergency_stop()
        else:
            logger.warning("⚠️ AudioProcessor: No emergency_stop method on transcription provider")
    
    def _emergency_cleanup_capture(self) -> None:
        """Emergency cleanup for capture provider (non-async)."""
        logger.warning("🚨 AudioProcessor: Emergency capture cleanup triggered")
        if hasattr(self.capture_provider, 'emergency_stop'):
            self.capture_provider.emergency_stop()
        else:
            logger.warning("⚠️ AudioProcessor: No emergency_stop method on capture provider")
    
    async def _cleanup_providers(self) -> None:
        """Clean up providers during initialization failure."""
        cleanup_tasks = []
        
        if self.transcription_provider:
            logger.info("🧹 AudioProcessor: Cleaning up transcription provider after init failure")
            try:
                await asyncio.wait_for(self.transcription_provider.stop_stream(), timeout=2.0)
            except Exception as e:
                logger.warning(f"⚠️ AudioProcessor: Error cleaning transcription provider: {e}")
            finally:
                self.transcription_provider = None
        
        if self.capture_provider:
            logger.info("🧹 AudioProcessor: Cleaning up capture provider after init failure")
            try:
                await asyncio.wait_for(self.capture_provider.stop_capture(), timeout=2.0)
            except Exception as e:
                logger.warning(f"⚠️ AudioProcessor: Error cleaning capture provider: {e}")
            finally:
                self.capture_provider = None
    
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
            ],
            'error_summary': self.error_handler.get_error_summary()
        }
    
    def get_pipeline_health(self) -> Dict[str, Any]:
        """Get pipeline health status and error information."""
        return {
            'is_running': self.is_running,
            'has_providers': {
                'transcription': self.transcription_provider is not None,
                'capture': self.capture_provider is not None
            },
            'has_tasks': {
                'capture_task': self._capture_task is not None and not self._capture_task.done(),
                'transcription_task': self._transcription_task is not None and not self._transcription_task.done()
            },
            'session_info': {
                'meeting_id': self.current_meeting_id,
                'transcript_count': len(self.session_transcripts)
            },
            'error_handler': self.error_handler.get_error_summary()
        }