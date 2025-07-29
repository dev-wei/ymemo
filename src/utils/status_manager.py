"""Audio processing status management system."""

import logging
from enum import Enum
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


class AudioStatus(Enum):
    """Audio processing status states."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    READY = "ready"
    CONNECTING = "connecting"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    TRANSCRIPTION_DISCONNECTED = "transcription_disconnected"
    RECONNECTING = "reconnecting"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class StatusInfo:
    """Information about current status."""
    status: AudioStatus
    message: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None


class AudioStatusManager:
    """Manages audio processing status and provides user-friendly messages."""
    
    # Status messages for UI display
    STATUS_MESSAGES = {
        AudioStatus.IDLE: "Ready to start",
        AudioStatus.INITIALIZING: "Initializing audio system...",
        AudioStatus.READY: "Ready to record",
        AudioStatus.CONNECTING: "Connecting to transcription service...",
        AudioStatus.RECORDING: "Recording audio...",
        AudioStatus.TRANSCRIBING: "Processing speech...",
        AudioStatus.TRANSCRIPTION_DISCONNECTED: "⚠️ Transcription disconnected - audio recording continues",
        AudioStatus.RECONNECTING: "🔄 Reconnecting to transcription service...",
        AudioStatus.STOPPING: "Stopping recording...",
        AudioStatus.STOPPED: "Recording stopped",
        AudioStatus.ERROR: "Error occurred"
    }
    
    # Status colors for UI styling
    STATUS_COLORS = {
        AudioStatus.IDLE: "gray",
        AudioStatus.INITIALIZING: "blue",
        AudioStatus.READY: "green",
        AudioStatus.CONNECTING: "blue",
        AudioStatus.RECORDING: "red",
        AudioStatus.TRANSCRIBING: "orange",
        AudioStatus.STOPPING: "yellow",
        AudioStatus.STOPPED: "gray",
        AudioStatus.ERROR: "red"
    }
    
    def __init__(self):
        self.current_status = AudioStatus.IDLE
        self.status_history: list[StatusInfo] = []
        self.status_callbacks: list[Callable[[StatusInfo], None]] = []
        self.error_callbacks: list[Callable[[Exception], None]] = []
        
    def set_status(
        self, 
        status: AudioStatus, 
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ) -> None:
        """Set the current status and notify callbacks.
        
        Args:
            status: New status
            message: Optional custom message (uses default if None)
            details: Optional additional details
            error: Optional error information
        """
        if message is None:
            message = self.STATUS_MESSAGES.get(status, str(status))
        
        # Add error details to message if present
        if error and status == AudioStatus.ERROR:
            message = f"{message}: {str(error)}"
        
        status_info = StatusInfo(
            status=status,
            message=message,
            timestamp=datetime.now(),
            details=details,
            error=error
        )
        
        self.current_status = status
        self.status_history.append(status_info)
        
        # Keep only last 100 status entries
        if len(self.status_history) > 100:
            self.status_history = self.status_history[-100:]
        
        logger.info(f"Status changed to {status.value}: {message}")
        
        # Notify callbacks
        for callback in self.status_callbacks:
            try:
                callback(status_info)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")
        
        # Notify error callbacks if this is an error status
        if status == AudioStatus.ERROR and error:
            for callback in self.error_callbacks:
                try:
                    callback(error)
                except Exception as e:
                    logger.error(f"Error in error callback: {e}")
    
    def get_current_status(self) -> StatusInfo:
        """Get the current status information.
        
        Returns:
            Current StatusInfo object
        """
        if self.status_history:
            return self.status_history[-1]
        
        # Return default status if no history
        return StatusInfo(
            status=AudioStatus.IDLE,
            message=self.STATUS_MESSAGES[AudioStatus.IDLE],
            timestamp=datetime.now()
        )
    
    def get_status_message(self) -> str:
        """Get the current status message for UI display.
        
        Returns:
            User-friendly status message
        """
        return self.get_current_status().message
    
    def get_status_color(self) -> str:
        """Get the current status color for UI styling.
        
        Returns:
            Color string for the current status
        """
        return self.STATUS_COLORS.get(self.current_status, "gray")
    
    def is_recording(self) -> bool:
        """Check if currently recording.
        
        Returns:
            True if in recording state
        """
        return self.current_status in [
            AudioStatus.RECORDING,
            AudioStatus.TRANSCRIBING
        ]
    
    def is_ready(self) -> bool:
        """Check if ready to start recording.
        
        Returns:
            True if ready to start
        """
        return self.current_status in [
            AudioStatus.IDLE,
            AudioStatus.READY,
            AudioStatus.STOPPED
        ]
    
    def is_error(self) -> bool:
        """Check if in error state.
        
        Returns:
            True if in error state
        """
        return self.current_status == AudioStatus.ERROR
    
    def add_status_callback(self, callback: Callable[[StatusInfo], None]) -> None:
        """Add a callback for status changes.
        
        Args:
            callback: Function to call when status changes
        """
        self.status_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Add a callback for error events.
        
        Args:
            callback: Function to call when errors occur
        """
        self.error_callbacks.append(callback)
    
    def remove_status_callback(self, callback: Callable[[StatusInfo], None]) -> None:
        """Remove a status callback.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self.status_callbacks:
            self.status_callbacks.remove(callback)
    
    def remove_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Remove an error callback.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self.error_callbacks:
            self.error_callbacks.remove(callback)
    
    def clear_callbacks(self) -> None:
        """Clear all callbacks."""
        self.status_callbacks.clear()
        self.error_callbacks.clear()
    
    def reset(self) -> None:
        """Reset status to idle and clear history."""
        self.current_status = AudioStatus.IDLE
        self.status_history.clear()
        self.set_status(AudioStatus.IDLE)
    
    def get_status_history(self, limit: int = 10) -> list[StatusInfo]:
        """Get recent status history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of recent StatusInfo objects
        """
        return self.status_history[-limit:] if self.status_history else []
    
    # Convenience methods for common status transitions
    def set_idle(self) -> None:
        """Set status to idle."""
        self.set_status(AudioStatus.IDLE)
    
    def set_initializing(self) -> None:
        """Set status to initializing."""
        self.set_status(AudioStatus.INITIALIZING)
    
    def set_ready(self) -> None:
        """Set status to ready."""
        self.set_status(AudioStatus.READY)
    
    def set_connecting(self) -> None:
        """Set status to connecting."""
        self.set_status(AudioStatus.CONNECTING)
    
    def set_recording(self) -> None:
        """Set status to recording."""
        self.set_status(AudioStatus.RECORDING)
    
    def set_transcribing(self) -> None:
        """Set status to transcribing."""
        self.set_status(AudioStatus.TRANSCRIBING)
    
    def set_stopping(self) -> None:
        """Set status to stopping."""
        self.set_status(AudioStatus.STOPPING)
    
    def set_stopped(self) -> None:
        """Set status to stopped."""
        self.set_status(AudioStatus.STOPPED)
    
    def set_transcription_disconnected(self, message: Optional[str] = None) -> None:
        """Set status to transcription disconnected.
        
        Args:
            message: Optional custom message about the disconnection
        """
        status_message = message or self.STATUS_MESSAGES[AudioStatus.TRANSCRIPTION_DISCONNECTED]
        self.set_status(AudioStatus.TRANSCRIPTION_DISCONNECTED, status_message)
    
    def set_reconnecting(self, attempt: int = 1) -> None:
        """Set status to reconnecting.
        
        Args:
            attempt: Reconnection attempt number
        """
        status_message = f"🔄 Reconnecting to transcription service... (attempt {attempt})"
        self.set_status(AudioStatus.RECONNECTING, status_message)
    
    def set_error(self, error: Exception, message: Optional[str] = None) -> None:
        """Set status to error.
        
        Args:
            error: Exception that occurred
            message: Optional custom error message
        """
        self.set_status(AudioStatus.ERROR, message=message, error=error)


# Global status manager instance
status_manager = AudioStatusManager()


def get_current_status() -> str:
    """Get the current status message.
    
    Returns:
        Current status message string
    """
    return status_manager.get_status_message()


def is_recording() -> bool:
    """Check if currently recording.
    
    Returns:
        True if recording is active
    """
    return status_manager.is_recording()


def is_ready() -> bool:
    """Check if ready to start recording.
    
    Returns:
        True if ready to start
    """
    return status_manager.is_ready()