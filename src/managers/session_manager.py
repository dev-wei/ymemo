"""Audio session management using singleton pattern."""

import threading
import asyncio
import logging
from typing import Optional, List, Callable
from concurrent.futures import TimeoutError
from datetime import datetime
from ..core.processor import AudioProcessor
from ..core.interfaces import TranscriptionResult
from ..utils.exceptions import SessionManagerError

logger = logging.getLogger(__name__)


class AudioSessionManager:
    """Singleton class to manage audio recording sessions."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.audio_processor: Optional[AudioProcessor] = None
        self.current_transcriptions: List[dict] = []
        self.transcription_callbacks: List[Callable[[dict], None]] = []
        self.background_thread: Optional[threading.Thread] = None
        self.background_loop: Optional[asyncio.AbstractEventLoop] = None
        self._session_lock = threading.Lock()
        self._stop_event = threading.Event()  # Simple signal for stopping
        
        # Track active partial results for smart updating
        self.active_partial_results: dict = {}  # utterance_id -> message_index
        self.partial_result_timeout = 2.0  # seconds
        
        # Session timing
        self.session_start_time = None
        self.session_end_time = None
        
    def add_transcription_callback(self, callback: Callable[[dict], None]) -> None:
        """Add a callback for transcription updates."""
        with self._session_lock:
            self.transcription_callbacks.append(callback)
    
    def remove_transcription_callback(self, callback: Callable[[dict], None]) -> None:
        """Remove a transcription callback."""
        with self._session_lock:
            if callback in self.transcription_callbacks:
                self.transcription_callbacks.remove(callback)
    
    def _on_transcription_received(self, result: TranscriptionResult) -> None:
        """Handle new transcription results with smart partial result management."""
        with self._session_lock:
            # Format the transcription message
            if result.speaker_id:
                content = f"{result.speaker_id}: {result.text}"
            else:
                content = result.text  # No speaker prefix when speaker ID is unknown
            
            message = {
                "role": "assistant", 
                "content": content,
                "timestamp": result.start_time,
                "confidence": result.confidence,
                "is_partial": result.is_partial,
                "utterance_id": result.utterance_id,
                "sequence_number": result.sequence_number,
                "result_id": result.result_id
            }
            
            logger.info(f"🎯 SessionManager received transcription: '{result.text}' (confidence: {result.confidence:.2f}, partial: {result.is_partial}, utterance: {result.utterance_id})")
            
            # Smart partial result handling
            if result.is_partial and result.utterance_id:
                # Check if we already have a partial result for this utterance
                if result.utterance_id in self.active_partial_results:
                    # Update existing partial result
                    existing_index = self.active_partial_results[result.utterance_id]
                    if existing_index < len(self.current_transcriptions):
                        self.current_transcriptions[existing_index] = message
                        logger.debug(f"📝 Updated partial result for utterance {result.utterance_id} at index {existing_index}")
                    else:
                        # Index is out of bounds, add new message
                        self.current_transcriptions.append(message)
                        self.active_partial_results[result.utterance_id] = len(self.current_transcriptions) - 1
                        logger.debug(f"📝 Added new partial result for utterance {result.utterance_id}")
                else:
                    # New partial result
                    self.current_transcriptions.append(message)
                    self.active_partial_results[result.utterance_id] = len(self.current_transcriptions) - 1
                    logger.debug(f"📝 Added new partial result for utterance {result.utterance_id}")
            else:
                # Final result or no utterance tracking
                if result.utterance_id and result.utterance_id in self.active_partial_results:
                    # Replace the partial result with the final result
                    existing_index = self.active_partial_results[result.utterance_id]
                    if existing_index < len(self.current_transcriptions):
                        self.current_transcriptions[existing_index] = message
                        logger.debug(f"✅ Finalized result for utterance {result.utterance_id} at index {existing_index}")
                    else:
                        # Index is out of bounds, add new message
                        self.current_transcriptions.append(message)
                        logger.debug(f"✅ Added final result for utterance {result.utterance_id}")
                    
                    # Clean up tracking
                    del self.active_partial_results[result.utterance_id]
                else:
                    # No partial result to replace, add new message
                    self.current_transcriptions.append(message)
                    logger.debug(f"✅ Added new final result")
            
            # Keep only last 100 messages to prevent memory issues
            if len(self.current_transcriptions) > 100:
                self.current_transcriptions = self.current_transcriptions[-100:]
                # Update partial result indices after truncation
                self.active_partial_results = {
                    utterance_id: max(0, index - (len(self.current_transcriptions) - 100))
                    for utterance_id, index in self.active_partial_results.items()
                    if index >= (len(self.current_transcriptions) - 100)
                }
            
            logger.debug(f"💾 Total transcriptions stored: {len(self.current_transcriptions)}, active partials: {len(self.active_partial_results)}")
            
            # Notify all callbacks
            logger.debug(f"🔔 Notifying {len(self.transcription_callbacks)} UI callbacks")
            for i, callback in enumerate(self.transcription_callbacks):
                try:
                    callback(message)
                    logger.debug(f"✅ Callback #{i+1} executed successfully")
                except Exception as e:
                    logger.error(f"❌ Error in callback #{i+1}: {e}")
    
    def _run_audio_processor_async(self, device_index: int) -> None:
        """Run audio processor in background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.background_loop = loop  # Store reference to the loop
        
        try:
            logger.debug(f"🔄 SessionManager: Starting async audio processing for device {device_index}")
            if self.audio_processor:
                # Run the audio processor - it will handle its own stopping
                loop.run_until_complete(self.audio_processor.start_recording(device_index))
            else:
                logger.error("❌ SessionManager: No audio processor available")
        except asyncio.CancelledError:
            logger.info("🛑 SessionManager: Background thread cancelled")
        except Exception as e:
            logger.error(f"❌ SessionManager: Audio processing error: {e}", exc_info=True)
        finally:
            self.background_loop = None  # Clear reference
            try:
                loop.close()
            except Exception as e:
                logger.warning(f"⚠️ SessionManager: Error closing background loop: {e}")
            logger.debug("🔄 SessionManager: Async audio processing loop closed")
    
    def start_recording(self, device_index: int, config: Optional[dict] = None) -> bool:
        """Start recording session.
        
        Args:
            device_index: Audio device index to use
            config: Optional configuration for transcription
            
        Returns:
            True if successfully started, False otherwise
        """
        with self._session_lock:
            if self.audio_processor is not None:
                logger.warning("⚠️  Recording already in progress, ignoring start request")
                return False  # Already recording
            
            try:
                logger.info(f"🎯 SessionManager: Starting recording with device {device_index}")
                
                # Clear stop event for new recording session
                self._stop_event.clear()
                
                # Clear previous session data
                self.current_transcriptions.clear()
                self.active_partial_results.clear()
                self.session_start_time = datetime.now()
                self.session_end_time = None
                
                # Create audio processor with default or provided config
                default_config = {
                    'region': 'us-east-1',
                    'language_code': 'en-US'
                }
                transcription_config = config or default_config
                logger.debug(f"🔧 SessionManager: Using transcription config: {transcription_config}")
                
                self.audio_processor = AudioProcessor(
                    transcription_provider='aws',
                    capture_provider='pyaudio',
                    transcription_config=transcription_config
                )
                logger.debug("✅ SessionManager: AudioProcessor created successfully")
                
                # Set up callbacks
                self.audio_processor.set_transcription_callback(self._on_transcription_received)
                logger.debug("✅ SessionManager: Transcription callback set")
                
                # Start recording in background thread
                self.background_thread = threading.Thread(
                    target=self._run_audio_processor_async,
                    args=(device_index,),
                    daemon=True
                )
                self.background_thread.start()
                logger.debug("✅ SessionManager: Background thread started")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ SessionManager: Failed to start recording: {e}", exc_info=True)
                self.audio_processor = None
                return False
    
    def stop_recording(self) -> bool:
        """Stop recording session.
        
        Returns:
            True if successfully stopped, False otherwise
        """
        with self._session_lock:
            if self.audio_processor is None:
                return False  # Not recording
            
            try:
                logger.info("🛑 SessionManager: Initiating stop sequence")
                
                # First, signal the audio processor to stop
                logger.info("🛑 SessionManager: Stopping AudioProcessor...")
                logger.info(f"🛑 SessionManager: AudioProcessor is_running before stop: {self.audio_processor.is_running}")
                
                # Stop the audio processor using the background loop if available
                # Note: stop_recording() will handle setting is_running = False
                stop_success = False
                stop_task = None
                
                # Use a shorter timeout to prevent hanging
                timeout = 2.0
                
                try:
                    if self.background_loop and not self.background_loop.is_closed():
                        logger.info("🛑 SessionManager: Using background loop to stop AudioProcessor")
                        logger.info(f"🛑 SessionManager: Background loop state: {self.background_loop}, closed: {self.background_loop.is_closed()}")
                        # Schedule the stop on the background loop
                        future = asyncio.run_coroutine_threadsafe(
                            self.audio_processor.stop_recording(),
                            self.background_loop
                        )
                        # Wait for it to complete with timeout
                        try:
                            future.result(timeout=timeout)
                            stop_success = True
                            logger.info("✅ SessionManager: AudioProcessor stopped via background loop")
                        except Exception as e:
                            logger.warning(f"⚠️ SessionManager: Future result error: {e}")
                            # Don't cancel the future - let it complete naturally
                            stop_success = False  # Mark as failed but continue cleanup
                    else:
                        logger.info("🛑 SessionManager: Background loop not available, using new loop")
                        logger.info(f"🛑 SessionManager: Background loop state: {self.background_loop}")
                        # Fallback to new event loop with better error handling
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            stop_task = loop.create_task(self.audio_processor.stop_recording())
                            loop.run_until_complete(asyncio.wait_for(stop_task, timeout=timeout))
                            stop_success = True
                            logger.info("✅ SessionManager: AudioProcessor stopped via new loop")
                        except Exception as e:
                            logger.warning(f"⚠️ SessionManager: Stop task error: {e}")
                            if stop_task and not stop_task.done():
                                try:
                                    stop_task.cancel()
                                    loop.run_until_complete(asyncio.wait_for(stop_task, timeout=0.5))
                                except (asyncio.CancelledError, asyncio.TimeoutError):
                                    pass
                                except Exception as cleanup_error:
                                    logger.warning(f"⚠️ SessionManager: Task cleanup error: {cleanup_error}")
                        finally:
                            try:
                                loop.close()
                            except Exception as loop_error:
                                logger.warning(f"⚠️ SessionManager: Loop close error: {loop_error}")
                    
                    if stop_success:
                        logger.info("✅ SessionManager: AudioProcessor stopped successfully")
                        logger.info(f"🛑 SessionManager: AudioProcessor is_running after stop: {self.audio_processor.is_running}")
                        logger.info(f"🛑 SessionManager: AudioProcessor provider type: {type(self.audio_processor.capture_provider).__name__ if self.audio_processor.capture_provider else 'None'}")
                        
                except asyncio.TimeoutError:
                    logger.warning("⚠️ SessionManager: AudioProcessor stop timeout, forcing cleanup")
                    # Don't cancel futures - let them complete naturally to avoid event loop issues
                    stop_success = False
                except Exception as e:
                    logger.warning(f"⚠️ SessionManager: Error stopping AudioProcessor: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue with cleanup even if stop fails
                
                # Wait for background thread to finish with shorter timeout
                if self.background_thread and self.background_thread.is_alive():
                    logger.info("🛑 SessionManager: Waiting for background thread to finish")
                    self.background_thread.join(timeout=0.5)  # Even shorter timeout
                    if self.background_thread.is_alive():
                        logger.info("🛑 SessionManager: Background thread still running - abandoning as daemon thread")
                        logger.info(f"🛑 SessionManager: Thread details: {self.background_thread.name}, daemon: {getattr(self.background_thread, 'daemon', 'unknown')}")
                        # Don't wait longer - daemon threads will be cleaned up automatically
                    else:
                        logger.info("✅ SessionManager: Background thread finished successfully")
                
                # Always clear background thread reference
                self.background_thread = None
                
                # Clean up - always clear references regardless of stop success
                # Add small delay to ensure all cleanup completes
                import time
                time.sleep(0.1)
                
                self.audio_processor = None
                self.background_loop = None
                self.session_end_time = datetime.now()
                logger.info("✅ SessionManager: Recording stopped successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to stop recording: {e}", exc_info=True)
                # Force cleanup even if there was an error
                self.audio_processor = None
                self.background_loop = None
                self.background_thread = None
                return False
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        with self._session_lock:
            return self.audio_processor is not None
    
    def get_current_transcriptions(self) -> List[dict]:
        """Get current transcriptions."""
        with self._session_lock:
            return self.current_transcriptions.copy()
    
    def clear_transcriptions(self) -> None:
        """Clear all transcriptions."""
        with self._session_lock:
            self.current_transcriptions.clear()
    
    def get_session_info(self) -> dict:
        """Get current session information."""
        with self._session_lock:
            # Calculate duration
            duration = 0.0
            if self.session_start_time:
                end_time = self.session_end_time or datetime.now()
                duration = (end_time - self.session_start_time).total_seconds() / 60.0  # Convert to minutes
            
            return {
                'is_recording': self.is_recording(),
                'transcription_count': len(self.current_transcriptions),
                'callbacks_registered': len(self.transcription_callbacks),
                'duration': duration,
                'start_time': self.session_start_time,
                'end_time': self.session_end_time
            }


# Convenience function to get the singleton instance
def get_audio_session() -> AudioSessionManager:
    """Get the global audio session manager instance."""
    return AudioSessionManager()