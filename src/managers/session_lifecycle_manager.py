"""Session lifecycle management with state persistence and recovery."""

import json
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum
import threading
import uuid

from .enhanced_session_manager import EnhancedAudioSessionManager, SessionState

logger = logging.getLogger(__name__)


class SessionPersistenceLevel(Enum):
    """Levels of session data persistence."""
    NONE = "none"  # No persistence
    METADATA_ONLY = "metadata_only"  # Only session metadata
    TRANSCRIPTIONS = "transcriptions"  # Include transcriptions
    FULL = "full"  # Everything including analytics


@dataclass
class SessionSnapshot:
    """Snapshot of session state for persistence and recovery."""
    session_id: str
    created_at: datetime
    last_updated: datetime
    state: str
    total_recording_time: float
    total_transcriptions: int
    transcriptions: List[Dict[str, Any]] = field(default_factory=list)
    recording_segments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        data['created_at'] = self.created_at.isoformat()
        data['last_updated'] = self.last_updated.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionSnapshot':
        """Create from dictionary (JSON deserialization)."""
        # Convert ISO strings back to datetime objects
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('last_updated'), str):
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        
        return cls(**data)


class SessionRecoveryStrategy(Enum):
    """Strategies for session recovery after interruption."""
    DISCARD = "discard"  # Discard incomplete session
    PRESERVE_METADATA = "preserve_metadata"  # Keep metadata, discard transcriptions
    FULL_RESTORE = "full_restore"  # Restore everything


@dataclass
class SessionLifecycleConfig:
    """Configuration for session lifecycle management."""
    persistence_level: SessionPersistenceLevel = SessionPersistenceLevel.TRANSCRIPTIONS
    auto_save_interval: float = 30.0  # seconds
    recovery_strategy: SessionRecoveryStrategy = SessionRecoveryStrategy.PRESERVE_METADATA
    max_session_age: timedelta = timedelta(hours=24)
    storage_directory: Optional[Path] = None
    max_stored_sessions: int = 10
    enable_crash_recovery: bool = True
    
    def __post_init__(self):
        if self.storage_directory is None:
            self.storage_directory = Path.home() / '.ymemo' / 'sessions'
        
        # Ensure storage directory exists
        self.storage_directory.mkdir(parents=True, exist_ok=True)


class SessionLifecycleManager:
    """Manages session lifecycle with state persistence and recovery capabilities."""
    
    def __init__(self, config: Optional[SessionLifecycleConfig] = None):
        self.config = config or SessionLifecycleConfig()
        self.session_manager = EnhancedAudioSessionManager()
        
        # State management
        self._current_session_id: Optional[str] = None
        self._session_metadata: Dict[str, Any] = {}
        self._last_save_time: Optional[datetime] = None
        
        # Threading
        self._lock = threading.RLock()
        self._auto_save_timer: Optional[threading.Timer] = None
        
        # Callbacks
        self._lifecycle_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # Initialize
        self._setup_session_manager_callbacks()
        self._recovery_check()
        
        logger.info(f"SessionLifecycleManager initialized with {self.config.persistence_level.value} persistence")
    
    def _setup_session_manager_callbacks(self):
        """Set up callbacks to monitor session manager state changes."""
        self.session_manager.add_state_callback(self._on_session_state_changed)
    
    def _on_session_state_changed(self, old_state: SessionState, new_state: SessionState):
        """Handle session manager state changes."""
        with self._lock:
            logger.info(f"Session state changed: {old_state.value} -> {new_state.value}")
            
            # Handle state-specific lifecycle events
            if new_state == SessionState.RECORDING and old_state != SessionState.RECORDING:
                self._on_recording_started()
            elif old_state == SessionState.RECORDING and new_state != SessionState.RECORDING:
                self._on_recording_stopped()
            elif new_state == SessionState.ERROR:
                self._on_session_error()
            
            # Schedule auto-save if enabled
            if self.config.persistence_level != SessionPersistenceLevel.NONE:
                self._schedule_auto_save()
    
    def _on_recording_started(self):
        """Handle recording start lifecycle event."""
        if self._current_session_id is None:
            self._start_new_session()
        
        self._session_metadata['last_recording_start'] = datetime.now().isoformat()
        self._notify_lifecycle_event('recording_started', {'session_id': self._current_session_id})
        
        logger.info(f"Recording started for session {self._current_session_id}")
    
    def _on_recording_stopped(self):
        """Handle recording stop lifecycle event."""
        if self._current_session_id:
            self._session_metadata['last_recording_stop'] = datetime.now().isoformat()
            self._save_session_state()
            self._notify_lifecycle_event('recording_stopped', {'session_id': self._current_session_id})
            
            logger.info(f"Recording stopped for session {self._current_session_id}")
    
    def _on_session_error(self):
        """Handle session error lifecycle event."""
        if self._current_session_id:
            self._session_metadata['last_error'] = datetime.now().isoformat()
            self._session_metadata['error_count'] = self._session_metadata.get('error_count', 0) + 1
            
            # Save state for crash recovery
            if self.config.enable_crash_recovery:
                self._save_session_state(force=True)
            
            self._notify_lifecycle_event('session_error', {
                'session_id': self._current_session_id,
                'error_count': self._session_metadata['error_count']
            })
            
            logger.warning(f"Session error occurred for {self._current_session_id}")
    
    def _start_new_session(self) -> str:
        """Start a new session with unique ID."""
        session_id = str(uuid.uuid4())
        self._current_session_id = session_id
        
        self._session_metadata = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'version': '2.0',
            'lifecycle_manager': True
        }
        
        logger.info(f"Started new session: {session_id}")
        return session_id
    
    def _schedule_auto_save(self):
        """Schedule automatic session save."""
        if self._auto_save_timer:
            self._auto_save_timer.cancel()
        
        self._auto_save_timer = threading.Timer(
            self.config.auto_save_interval,
            self._auto_save_callback
        )
        self._auto_save_timer.daemon = True
        self._auto_save_timer.start()
    
    def _auto_save_callback(self):
        """Automatic save callback."""
        try:
            with self._lock:
                if self._current_session_id and self.session_manager.current_state != SessionState.IDLE:
                    self._save_session_state()
                    logger.debug(f"Auto-saved session {self._current_session_id}")
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")
    
    def _save_session_state(self, force: bool = False):
        """Save current session state to persistent storage."""
        if self.config.persistence_level == SessionPersistenceLevel.NONE:
            return
        
        if not self._current_session_id:
            return
        
        # Check if save is needed
        now = datetime.now()
        if not force and self._last_save_time:
            time_since_save = (now - self._last_save_time).total_seconds()
            if time_since_save < self.config.auto_save_interval / 2:
                return  # Too soon since last save
        
        try:
            # Create snapshot
            snapshot = self._create_session_snapshot()
            
            # Save to file
            session_file = self.config.storage_directory / f"session_{self._current_session_id}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(snapshot.to_dict(), f, indent=2, ensure_ascii=False)
            
            self._last_save_time = now
            
            # Clean up old sessions
            self._cleanup_old_sessions()
            
            logger.debug(f"Saved session state to {session_file}")
            
        except Exception as e:
            logger.error(f"Failed to save session state: {e}")
    
    def _create_session_snapshot(self) -> SessionSnapshot:
        """Create a snapshot of current session state."""
        session_info = self.session_manager.get_session_info()
        
        # Get transcriptions based on persistence level
        transcriptions = []
        if self.config.persistence_level in [SessionPersistenceLevel.TRANSCRIPTIONS, SessionPersistenceLevel.FULL]:
            transcriptions = self.session_manager.get_current_transcriptions()
        
        # Get recording segments
        recording_segments = []
        if self.config.persistence_level == SessionPersistenceLevel.FULL:
            segments = self.session_manager.get_recording_segments()
            recording_segments = [
                {
                    'start_time': seg.start_time.isoformat() if seg.start_time else None,
                    'end_time': seg.end_time.isoformat() if seg.end_time else None,
                    'duration_seconds': seg.duration_seconds,
                    'device_index': seg.device_index,
                    'transcription_count': seg.transcription_count
                }
                for seg in segments
            ]
        
        # Combine metadata
        combined_metadata = {**self._session_metadata}
        if self.config.persistence_level == SessionPersistenceLevel.FULL:
            combined_metadata.update(session_info.get('metrics', {}))
        
        return SessionSnapshot(
            session_id=self._current_session_id,
            created_at=datetime.fromisoformat(self._session_metadata['created_at']),
            last_updated=datetime.now(),
            state=session_info['state'],
            total_recording_time=session_info['metrics']['total_recording_time'],
            total_transcriptions=session_info['metrics']['total_transcriptions'],
            transcriptions=transcriptions,
            recording_segments=recording_segments,
            metadata=combined_metadata
        )
    
    def _recovery_check(self):
        """Check for recoverable sessions and handle according to strategy."""
        if not self.config.enable_crash_recovery:
            return
        
        try:
            session_files = list(self.config.storage_directory.glob("session_*.json"))
            
            for session_file in session_files:
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    snapshot = SessionSnapshot.from_dict(data)
                    
                    # Check if session is recoverable
                    if self._should_recover_session(snapshot):
                        self._recover_session(snapshot)
                        logger.info(f"Recovered session {snapshot.session_id}")
                    else:
                        # Clean up old session file
                        session_file.unlink(missing_ok=True)
                        logger.debug(f"Cleaned up old session file {session_file}")
                        
                except Exception as e:
                    logger.warning(f"Failed to process session file {session_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Recovery check failed: {e}")
    
    def _should_recover_session(self, snapshot: SessionSnapshot) -> bool:
        """Determine if a session should be recovered."""
        # Check age
        age = datetime.now() - snapshot.last_updated
        if age > self.config.max_session_age:
            return False
        
        # Check if session was in a recoverable state
        if snapshot.state in ['idle', 'error']:
            return False
        
        # Check recovery strategy
        if self.config.recovery_strategy == SessionRecoveryStrategy.DISCARD:
            return False
        
        return True
    
    def _recover_session(self, snapshot: SessionSnapshot):
        """Recover a session from snapshot."""
        try:
            self._current_session_id = snapshot.session_id
            self._session_metadata = snapshot.metadata.copy()
            
            # Restore transcriptions based on strategy
            if self.config.recovery_strategy == SessionRecoveryStrategy.FULL_RESTORE:
                # Clear current session and restore transcriptions
                self.session_manager.clear_transcriptions()
                
                # Add transcriptions back (this is complex due to the way transcription buffer works)
                # For now, we'll just restore the metadata and let the user know
                self._session_metadata['recovered'] = True
                self._session_metadata['recovery_time'] = datetime.now().isoformat()
                self._session_metadata['original_transcription_count'] = len(snapshot.transcriptions)
                
            self._notify_lifecycle_event('session_recovered', {
                'session_id': snapshot.session_id,
                'original_state': snapshot.state,
                'transcription_count': len(snapshot.transcriptions),
                'recovery_strategy': self.config.recovery_strategy.value
            })
            
        except Exception as e:
            logger.error(f"Failed to recover session {snapshot.session_id}: {e}")
    
    def _cleanup_old_sessions(self):
        """Clean up old session files."""
        try:
            session_files = list(self.config.storage_directory.glob("session_*.json"))
            
            # Sort by modification time (newest first)
            session_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Keep only the most recent sessions
            for old_file in session_files[self.config.max_stored_sessions:]:
                try:
                    old_file.unlink()
                    logger.debug(f"Cleaned up old session file {old_file}")
                except Exception as e:
                    logger.warning(f"Failed to delete old session file {old_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to clean up old sessions: {e}")
    
    def _notify_lifecycle_event(self, event_type: str, data: Dict[str, Any]):
        """Notify lifecycle event callbacks."""
        for callback in self._lifecycle_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in lifecycle callback: {e}")
    
    def add_lifecycle_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add lifecycle event callback."""
        with self._lock:
            self._lifecycle_callbacks.append(callback)
    
    def remove_lifecycle_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Remove lifecycle event callback."""
        with self._lock:
            if callback in self._lifecycle_callbacks:
                self._lifecycle_callbacks.remove(callback)
    
    def get_current_session_info(self) -> Dict[str, Any]:
        """Get current session lifecycle information."""
        with self._lock:
            base_info = self.session_manager.get_session_info()
            
            return {
                **base_info,
                'session_id': self._current_session_id,
                'session_metadata': self._session_metadata.copy(),
                'persistence_level': self.config.persistence_level.value,
                'last_save_time': self._last_save_time.isoformat() if self._last_save_time else None,
                'auto_save_enabled': self.config.persistence_level != SessionPersistenceLevel.NONE,
                'crash_recovery_enabled': self.config.enable_crash_recovery
            }
    
    def force_save(self) -> bool:
        """Force immediate save of current session state."""
        try:
            with self._lock:
                if self._current_session_id:
                    self._save_session_state(force=True)
                    return True
                return False
        except Exception as e:
            logger.error(f"Force save failed: {e}")
            return False
    
    def end_current_session(self):
        """Properly end the current session."""
        with self._lock:
            if self._current_session_id:
                # Stop recording if active
                if self.session_manager.is_recording():
                    self.session_manager.stop_recording()
                
                # Final save
                self._session_metadata['ended_at'] = datetime.now().isoformat()
                self._save_session_state(force=True)
                
                # Clean up
                session_id = self._current_session_id
                self._current_session_id = None
                self._session_metadata = {}
                
                # Cancel auto-save timer
                if self._auto_save_timer:
                    self._auto_save_timer.cancel()
                    self._auto_save_timer = None
                
                self._notify_lifecycle_event('session_ended', {'session_id': session_id})
                
                logger.info(f"Ended session {session_id}")
    
    def list_stored_sessions(self) -> List[Dict[str, Any]]:
        """List all stored sessions."""
        sessions = []
        
        try:
            session_files = list(self.config.storage_directory.glob("session_*.json"))
            
            for session_file in session_files:
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    snapshot = SessionSnapshot.from_dict(data)
                    
                    sessions.append({
                        'session_id': snapshot.session_id,
                        'created_at': snapshot.created_at,
                        'last_updated': snapshot.last_updated,
                        'state': snapshot.state,
                        'total_recording_time': snapshot.total_recording_time,
                        'total_transcriptions': snapshot.total_transcriptions,
                        'file_path': str(session_file)
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to read session file {session_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to list stored sessions: {e}")
        
        # Sort by creation time (newest first)
        sessions.sort(key=lambda s: s['created_at'], reverse=True)
        
        return sessions
    
    def load_session(self, session_id: str) -> Optional[SessionSnapshot]:
        """Load a specific session snapshot."""
        try:
            session_file = self.config.storage_directory / f"session_{session_id}.json"
            
            if not session_file.exists():
                return None
            
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return SessionSnapshot.from_dict(data)
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def delete_stored_session(self, session_id: str) -> bool:
        """Delete a stored session."""
        try:
            session_file = self.config.storage_directory / f"session_{session_id}.json"
            
            if session_file.exists():
                session_file.unlink()
                logger.info(f"Deleted stored session {session_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    def cleanup(self):
        """Clean up lifecycle manager resources."""
        with self._lock:
            # End current session
            self.end_current_session()
            
            # Cancel auto-save timer
            if self._auto_save_timer:
                self._auto_save_timer.cancel()
                self._auto_save_timer = None
            
            # Clear callbacks
            self._lifecycle_callbacks.clear()
            
            logger.info("SessionLifecycleManager cleanup completed")