"""Enhanced audio session management with improved thread safety and lifecycle management."""

import threading
import asyncio
import logging
import time
from typing import Optional, List, Callable, Dict, Any
from datetime import datetime, timedelta
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum

from ..core.processor import AudioProcessor
from ..core.interfaces import TranscriptionResult
from ..utils.exceptions import SessionManagerError
from ..utils.status_manager import status_manager

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Session lifecycle states."""
    IDLE = "idle"
    INITIALIZING = "initializing" 
    CONNECTING = "connecting"
    RECORDING = "recording"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class RecordingSegment:
    """Recording segment information."""
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    device_index: Optional[int] = None
    transcription_count: int = 0
    
    def complete(self) -> None:
        """Mark segment as complete."""
        if self.end_time is None:
            self.end_time = datetime.now()
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()


@dataclass
class SessionMetrics:
    """Session analytics and metrics."""
    total_recording_time: float = 0.0
    total_transcriptions: int = 0
    partial_transcriptions: int = 0
    final_transcriptions: int = 0
    connection_errors: int = 0
    recording_segments: List[RecordingSegment] = field(default_factory=list)
    session_start_time: Optional[datetime] = None
    last_activity_time: Optional[datetime] = None
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity_time = datetime.now()
        if self.session_start_time is None:
            self.session_start_time = self.last_activity_time


class TranscriptionBuffer:
    """Thread-safe transcription buffer with smart partial result handling."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._messages: List[Dict[str, Any]] = []
        self._active_partials: Dict[str, int] = {}  # utterance_id -> message_index
        self._lock = threading.RLock()
        
    @contextmanager
    def _thread_safe(self):
        """Context manager for thread-safe operations."""
        with self._lock:
            yield
            
    def add_transcription(self, result: TranscriptionResult) -> Dict[str, Any]:
        """Add transcription result with smart partial handling."""
        with self._thread_safe():
            # Format message
            if result.speaker_id:
                content = f"{result.speaker_id}: {result.text}"
            else:
                content = result.text
                
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
            
            # Smart partial result handling
            if result.is_partial and result.utterance_id:
                self._handle_partial_result(message, result.utterance_id)
            else:
                self._handle_final_result(message, result.utterance_id)
                
            # Manage buffer size
            self._manage_buffer_size()
            
            return message
    
    def _handle_partial_result(self, message: Dict[str, Any], utterance_id: str) -> None:
        """Handle partial transcription result."""
        if utterance_id in self._active_partials:
            # Update existing partial
            existing_index = self._active_partials[utterance_id]
            if self._is_valid_index(existing_index, utterance_id):
                self._messages[existing_index] = message
                logger.debug(f"Updated partial result for {utterance_id} at index {existing_index}")
            else:
                # Index is invalid, add as new
                self._add_new_message(message, utterance_id)
        else:
            # New partial result
            self._add_new_message(message, utterance_id)
    
    def _handle_final_result(self, message: Dict[str, Any], utterance_id: Optional[str]) -> None:
        """Handle final transcription result."""
        if utterance_id and utterance_id in self._active_partials:
            # Replace partial with final
            existing_index = self._active_partials[utterance_id]
            if self._is_valid_index(existing_index, utterance_id):
                self._messages[existing_index] = message
                logger.debug(f"Finalized result for {utterance_id} at index {existing_index}")
            else:
                self._messages.append(message)
            
            # Clean up partial tracking
            del self._active_partials[utterance_id]
        else:
            # New final result
            self._messages.append(message)
    
    def _add_new_message(self, message: Dict[str, Any], utterance_id: Optional[str]) -> None:
        """Add new message and track if partial."""
        self._messages.append(message)
        if message.get("is_partial") and utterance_id:
            self._active_partials[utterance_id] = len(self._messages) - 1
            logger.debug(f"Added new partial result for {utterance_id}")
    
    def _is_valid_index(self, index: int, expected_utterance_id: str) -> bool:
        """Validate that index points to correct utterance."""
        if 0 <= index < len(self._messages):
            actual_utterance_id = self._messages[index].get('utterance_id')
            return actual_utterance_id == expected_utterance_id
        return False
    
    def _manage_buffer_size(self) -> None:
        """Manage buffer size and update indices."""
        if len(self._messages) <= self.max_size:
            return
            
        # Calculate items to remove
        items_to_remove = len(self._messages) - self.max_size
        logger.info(f"Truncating transcription buffer: removing {items_to_remove} items")
        
        # Truncate messages
        self._messages = self._messages[-self.max_size:]
        
        # Update partial indices
        old_partials = self._active_partials.copy()
        self._active_partials.clear()
        
        for utterance_id, old_index in old_partials.items():
            new_index = old_index - items_to_remove
            if new_index >= 0 and self._is_valid_index(new_index, utterance_id):
                self._active_partials[utterance_id] = new_index
                logger.debug(f"Preserved partial {utterance_id} at new index {new_index}")
            else:
                logger.debug(f"Dropped partial {utterance_id} after truncation")
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get copy of all messages."""
        with self._thread_safe():
            return self._messages.copy()
    
    def clear(self) -> None:
        """Clear all messages and partial tracking."""
        with self._thread_safe():
            self._messages.clear()
            self._active_partials.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get buffer statistics."""
        with self._thread_safe():
            return {
                "total_messages": len(self._messages),
                "active_partials": len(self._active_partials),
                "buffer_capacity": self.max_size
            }


class EnhancedAudioSessionManager:
    """Enhanced audio session manager with improved thread safety and lifecycle management."""
    
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
        if getattr(self, '_initialized', False):
            return
            
        self._initialized = True
        
        # Core components
        self._audio_processor: Optional[AudioProcessor] = None
        self._transcription_buffer = TranscriptionBuffer()
        self._session_metrics = SessionMetrics()
        
        # Thread safety
        self._session_lock = threading.RLock()
        self._state_lock = threading.Lock()
        
        # State management
        self._current_state = SessionState.IDLE
        self._state_callbacks: List[Callable[[SessionState, SessionState], None]] = []
        
        # Threading
        self._background_thread: Optional[threading.Thread] = None
        self._background_loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event = threading.Event()
        self._shutdown_timeout = 3.0
        
        # Callbacks
        self._transcription_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._connection_health = True
        
        # Current recording segment
        self._current_segment: Optional[RecordingSegment] = None
        
        logger.info("Enhanced AudioSessionManager initialized")
    
    @property
    def current_state(self) -> SessionState:
        """Get current session state."""
        with self._state_lock:
            return self._current_state
    
    def _set_state(self, new_state: SessionState) -> None:
        """Set session state and notify callbacks."""
        with self._state_lock:
            if self._current_state == new_state:
                return
                
            old_state = self._current_state
            self._current_state = new_state
            
            logger.info(f"Session state changed: {old_state.value} -> {new_state.value}")
            
            # Update metrics
            self._session_metrics.update_activity()
            
            # Notify state callbacks
            for callback in self._state_callbacks:
                try:
                    callback(old_state, new_state)
                except Exception as e:
                    logger.error(f"Error in state callback: {e}")
    
    def add_state_callback(self, callback: Callable[[SessionState, SessionState], None]) -> None:
        """Add state change callback."""
        with self._state_lock:
            self._state_callbacks.append(callback)
    
    def remove_state_callback(self, callback: Callable[[SessionState, SessionState], None]) -> None:
        """Remove state change callback."""
        with self._state_lock:
            if callback in self._state_callbacks:
                self._state_callbacks.remove(callback)
    
    def add_transcription_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add transcription callback."""
        with self._session_lock:
            self._transcription_callbacks.append(callback)
    
    def remove_transcription_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Remove transcription callback."""
        with self._session_lock:
            if callback in self._transcription_callbacks:
                self._transcription_callbacks.remove(callback)
    
    def _on_transcription_received(self, result: TranscriptionResult) -> None:
        """Handle transcription result."""
        with self._session_lock:
            try:
                # Add to buffer
                message = self._transcription_buffer.add_transcription(result)
                
                # Update metrics
                self._session_metrics.total_transcriptions += 1
                if result.is_partial:
                    self._session_metrics.partial_transcriptions += 1
                else:
                    self._session_metrics.final_transcriptions += 1
                
                # Update current segment
                if self._current_segment:
                    self._current_segment.transcription_count += 1
                
                self._session_metrics.update_activity()
                
                logger.debug(f"Received transcription: '{result.text}' (partial: {result.is_partial})")
                
                # Notify callbacks
                for callback in self._transcription_callbacks:
                    try:
                        callback(message)
                    except Exception as e:
                        logger.error(f"Error in transcription callback: {e}")
                        
            except Exception as e:
                logger.error(f"Error processing transcription: {e}")
    
    def _on_connection_health_changed(self, is_healthy: bool, message: str) -> None:
        """Handle connection health changes."""
        with self._session_lock:
            logger.info(f"Connection health changed: {is_healthy}, message: {message}")
            
            if is_healthy != self._connection_health:
                self._connection_health = is_healthy
                
                if not is_healthy:
                    self._session_metrics.connection_errors += 1
                
                # Update status manager
                if self.is_recording():
                    if is_healthy:
                        status_manager.set_recording()
                    else:
                        status_manager.set_transcription_disconnected(message)
    
    def _run_audio_processor_async(self, device_index: int) -> None:
        """Run audio processor in background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        with self._session_lock:
            self._background_loop = loop
        
        try:
            logger.debug(f"Starting audio processing for device {device_index}")
            
            if self._audio_processor:
                # Set callbacks before starting
                self._audio_processor.set_transcription_callback(self._on_transcription_received)
                self._audio_processor.set_connection_health_callback(self._on_connection_health_changed)
                
                # Run audio processor
                loop.run_until_complete(self._audio_processor.start_recording(device_index))
            else:
                logger.error("No audio processor available")
                
        except asyncio.CancelledError:
            logger.info("Audio processing cancelled")
        except Exception as e:
            logger.error(f"Audio processing error: {e}", exc_info=True)
            self._set_state(SessionState.ERROR)
        finally:
            with self._session_lock:
                self._background_loop = None
            try:
                loop.close()
            except Exception as e:
                logger.warning(f"Error closing event loop: {e}")
    
    def start_recording(self, device_index: int, config: Optional[Dict[str, Any]] = None) -> bool:
        """Start recording session."""
        with self._session_lock:
            if self._current_state not in [SessionState.IDLE, SessionState.ERROR]:
                logger.warning(f"Cannot start recording in state {self._current_state}")
                return False
            
            try:
                self._set_state(SessionState.INITIALIZING)
                
                # Clear stop event
                self._stop_event.clear()
                
                # Start new recording segment
                self._current_segment = RecordingSegment(
                    start_time=datetime.now(),
                    device_index=device_index
                )
                
                # Create audio processor using centralized configuration
                from config.audio_config import get_config
                system_config = get_config()
                transcription_config = config or system_config.get_transcription_config()
                
                self._audio_processor = AudioProcessor(
                    transcription_provider='aws',
                    capture_provider='pyaudio',
                    transcription_config=transcription_config
                )
                
                self._set_state(SessionState.CONNECTING)
                
                # Start background processing
                self._background_thread = threading.Thread(
                    target=self._run_audio_processor_async,
                    args=(device_index,),
                    daemon=True
                )
                self._background_thread.start()
                
                # Give it a moment to start
                time.sleep(0.1)
                
                self._set_state(SessionState.RECORDING)
                
                logger.info(f"Recording started on device {device_index}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to start recording: {e}", exc_info=True)
                self._cleanup_recording()
                self._set_state(SessionState.ERROR)
                return False
    
    def stop_recording(self) -> bool:
        """Stop recording session."""
        with self._session_lock:
            if self._current_state != SessionState.RECORDING:
                logger.warning(f"Cannot stop recording in state {self._current_state}")
                return False
            
            try:
                self._set_state(SessionState.STOPPING)
                
                # Stop audio processor
                stop_success = self._stop_audio_processor()
                
                # Wait for background thread
                self._wait_for_background_thread()
                
                # Complete current segment
                if self._current_segment:
                    self._current_segment.complete()
                    self._session_metrics.recording_segments.append(self._current_segment)
                    self._session_metrics.total_recording_time += self._current_segment.duration_seconds or 0.0
                    self._current_segment = None
                
                # Cleanup
                self._cleanup_recording()
                
                self._set_state(SessionState.IDLE)
                
                logger.info("Recording stopped successfully")
                return True
                
            except Exception as e:
                logger.error(f"Error stopping recording: {e}", exc_info=True)
                self._cleanup_recording()
                self._set_state(SessionState.ERROR)
                return False
    
    def _stop_audio_processor(self) -> bool:
        """Stop the audio processor gracefully."""
        if not self._audio_processor:
            return True
            
        try:
            if self._background_loop and not self._background_loop.is_closed():
                # Use background loop
                future = asyncio.run_coroutine_threadsafe(
                    self._audio_processor.stop_recording(),
                    self._background_loop
                )
                future.result(timeout=self._shutdown_timeout)
                return True
            else:
                # Use new loop
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    task = loop.create_task(self._audio_processor.stop_recording())
                    loop.run_until_complete(asyncio.wait_for(task, timeout=self._shutdown_timeout))
                    return True
                finally:
                    try:
                        loop.close()
                    except Exception:
                        pass
                        
        except Exception as e:
            logger.warning(f"Error stopping audio processor: {e}")
            return False
    
    def _wait_for_background_thread(self) -> None:
        """Wait for background thread to finish."""
        if self._background_thread and self._background_thread.is_alive():
            logger.debug("Waiting for background thread to finish")
            self._background_thread.join(timeout=1.0)
            
            if self._background_thread.is_alive():
                logger.warning("Background thread still running after timeout")
    
    def _cleanup_recording(self) -> None:
        """Clean up recording resources."""
        self._audio_processor = None
        self._background_thread = None
        self._background_loop = None
        self._stop_event.set()
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._current_state == SessionState.RECORDING
    
    def get_current_transcriptions(self) -> List[Dict[str, Any]]:
        """Get current transcriptions."""
        return self._transcription_buffer.get_messages()
    
    def clear_transcriptions(self) -> None:
        """Clear all transcriptions."""
        with self._session_lock:
            self._transcription_buffer.clear()
            logger.info("Transcriptions cleared")
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information."""
        with self._session_lock:
            buffer_stats = self._transcription_buffer.get_stats()
            
            return {
                'state': self._current_state.value,
                'is_recording': self.is_recording(),
                'transcription_count': buffer_stats['total_messages'],
                'active_partials': buffer_stats['active_partials'],
                'callbacks_registered': len(self._transcription_callbacks),
                'connection_healthy': self._connection_health,
                'metrics': {
                    'total_recording_time': self._session_metrics.total_recording_time,
                    'total_transcriptions': self._session_metrics.total_transcriptions,
                    'partial_transcriptions': self._session_metrics.partial_transcriptions,
                    'final_transcriptions': self._session_metrics.final_transcriptions,
                    'connection_errors': self._session_metrics.connection_errors,
                    'recording_segments': len(self._session_metrics.recording_segments),
                    'session_start_time': self._session_metrics.session_start_time,
                    'last_activity_time': self._session_metrics.last_activity_time
                }
            }
    
    def get_current_duration_seconds(self) -> float:
        """Get current total recording duration in seconds."""
        with self._session_lock:
            total_duration = self._session_metrics.total_recording_time
            
            # Add current segment duration if recording
            if self._current_segment and self._current_segment.end_time is None:
                current_duration = (datetime.now() - self._current_segment.start_time).total_seconds()
                total_duration += current_duration
            
            return total_duration
    
    def get_formatted_duration(self) -> str:
        """Get formatted duration string."""
        total_seconds = self.get_current_duration_seconds()
        return self._format_duration(total_seconds)
    
    def _format_duration(self, total_seconds: float) -> str:
        """Format seconds as MM:SS or HH:MM:SS."""
        if total_seconds < 0:
            total_seconds = 0
        
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def reset_session(self) -> None:
        """Reset session to clean state."""
        with self._session_lock:
            if self.is_recording():
                self.stop_recording()
            
            self._transcription_buffer.clear()
            self._session_metrics = SessionMetrics()
            self._current_segment = None
            self._connection_health = True
            
            self._set_state(SessionState.IDLE)
            
            logger.info("Session reset complete")
    
    def get_recording_segments(self) -> List[RecordingSegment]:
        """Get recording segments for analytics."""
        with self._session_lock:
            return self._session_metrics.recording_segments.copy()


# Factory function for backward compatibility
def get_enhanced_audio_session() -> EnhancedAudioSessionManager:
    """Get the enhanced audio session manager instance."""
    return EnhancedAudioSessionManager()