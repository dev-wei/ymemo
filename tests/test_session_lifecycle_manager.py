"""Tests for session lifecycle manager."""

import unittest
import tempfile
import json
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

from src.managers.session_lifecycle_manager import (
    SessionLifecycleManager, SessionLifecycleConfig, SessionSnapshot,
    SessionPersistenceLevel, SessionRecoveryStrategy
)
from src.managers.enhanced_session_manager import SessionState


class TestSessionSnapshot(unittest.TestCase):
    """Test cases for SessionSnapshot."""
    
    def test_snapshot_creation(self):
        """Test creating a session snapshot."""
        now = datetime.now()
        snapshot = SessionSnapshot(
            session_id="test_123",
            created_at=now,
            last_updated=now,
            state="recording",
            total_recording_time=120.5,
            total_transcriptions=25,
            transcriptions=[{"content": "test"}],
            metadata={"test": "value"}
        )
        
        self.assertEqual(snapshot.session_id, "test_123")
        self.assertEqual(snapshot.created_at, now)
        self.assertEqual(snapshot.state, "recording")
        self.assertEqual(snapshot.total_recording_time, 120.5)
        self.assertEqual(snapshot.total_transcriptions, 25)
        self.assertEqual(len(snapshot.transcriptions), 1)
        self.assertEqual(snapshot.metadata["test"], "value")
    
    def test_snapshot_serialization(self):
        """Test snapshot serialization to dictionary."""
        now = datetime.now()
        snapshot = SessionSnapshot(
            session_id="test_123",
            created_at=now,
            last_updated=now,
            state="idle",
            total_recording_time=60.0,
            total_transcriptions=10
        )
        
        data = snapshot.to_dict()
        
        self.assertEqual(data["session_id"], "test_123")
        self.assertEqual(data["created_at"], now.isoformat())
        self.assertEqual(data["last_updated"], now.isoformat())
        self.assertEqual(data["state"], "idle")
        self.assertEqual(data["total_recording_time"], 60.0)
    
    def test_snapshot_deserialization(self):
        """Test snapshot deserialization from dictionary."""
        now = datetime.now()
        data = {
            "session_id": "test_456",
            "created_at": now.isoformat(),
            "last_updated": now.isoformat(),
            "state": "recording",
            "total_recording_time": 90.0,
            "total_transcriptions": 15,
            "transcriptions": [{"content": "hello"}],
            "recording_segments": [],
            "metadata": {"version": "2.0"}
        }
        
        snapshot = SessionSnapshot.from_dict(data)
        
        self.assertEqual(snapshot.session_id, "test_456")
        self.assertEqual(snapshot.created_at, now)
        self.assertEqual(snapshot.last_updated, now)
        self.assertEqual(snapshot.state, "recording")
        self.assertEqual(snapshot.total_recording_time, 90.0)
        self.assertEqual(len(snapshot.transcriptions), 1)
        self.assertEqual(snapshot.metadata["version"], "2.0")


class TestSessionLifecycleConfig(unittest.TestCase):
    """Test cases for SessionLifecycleConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SessionLifecycleConfig()
        
        self.assertEqual(config.persistence_level, SessionPersistenceLevel.TRANSCRIPTIONS)
        self.assertEqual(config.auto_save_interval, 30.0)
        self.assertEqual(config.recovery_strategy, SessionRecoveryStrategy.PRESERVE_METADATA)
        self.assertEqual(config.max_session_age, timedelta(hours=24))
        self.assertEqual(config.max_stored_sessions, 10)
        self.assertTrue(config.enable_crash_recovery)
        
        # Storage directory should be created
        self.assertIsNotNone(config.storage_directory)
        self.assertTrue(config.storage_directory.exists())
    
    def test_custom_config(self):
        """Test custom configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_dir = Path(temp_dir) / "custom_sessions"
            
            config = SessionLifecycleConfig(
                persistence_level=SessionPersistenceLevel.FULL,
                auto_save_interval=15.0,
                recovery_strategy=SessionRecoveryStrategy.FULL_RESTORE,
                max_session_age=timedelta(hours=12),
                storage_directory=storage_dir,
                max_stored_sessions=5,
                enable_crash_recovery=False
            )
            
            self.assertEqual(config.persistence_level, SessionPersistenceLevel.FULL)
            self.assertEqual(config.auto_save_interval, 15.0)
            self.assertEqual(config.recovery_strategy, SessionRecoveryStrategy.FULL_RESTORE)
            self.assertEqual(config.max_session_age, timedelta(hours=12))
            self.assertEqual(config.storage_directory, storage_dir)
            self.assertEqual(config.max_stored_sessions, 5)
            self.assertFalse(config.enable_crash_recovery)
            
            # Storage directory should be created
            self.assertTrue(storage_dir.exists())


class TestSessionLifecycleManager(unittest.TestCase):
    """Test cases for SessionLifecycleManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name) / "test_sessions"
        
        self.config = SessionLifecycleConfig(
            persistence_level=SessionPersistenceLevel.TRANSCRIPTIONS,
            auto_save_interval=1.0,  # Short interval for testing
            storage_directory=self.storage_dir,
            enable_crash_recovery=True  # Enable for error handling test
        )
        
        # Mock the enhanced session manager to avoid actual audio processing
        with patch('src.managers.session_lifecycle_manager.EnhancedAudioSessionManager') as mock_session_manager:
            self.mock_session_manager_instance = Mock()
            mock_session_manager.return_value = self.mock_session_manager_instance
            
            # Set up default mock behavior
            self.mock_session_manager_instance.current_state = SessionState.IDLE
            self.mock_session_manager_instance.is_recording.return_value = False
            self.mock_session_manager_instance.get_session_info.return_value = {
                'state': 'idle',
                'metrics': {
                    'total_recording_time': 0.0,
                    'total_transcriptions': 0,
                    'partial_transcriptions': 0,
                    'final_transcriptions': 0,
                    'connection_errors': 0
                }
            }
            self.mock_session_manager_instance.get_current_transcriptions.return_value = []
            self.mock_session_manager_instance.get_recording_segments.return_value = []
            
            self.lifecycle_manager = SessionLifecycleManager(self.config)
    
    def tearDown(self):
        """Clean up after each test."""
        try:
            self.lifecycle_manager.cleanup()
        except:
            pass
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test lifecycle manager initialization."""
        self.assertEqual(self.lifecycle_manager.config, self.config)
        self.assertIsNotNone(self.lifecycle_manager.session_manager)
        
        # Should have set up callbacks
        self.mock_session_manager_instance.add_state_callback.assert_called()
    
    def test_session_creation_on_recording_start(self):
        """Test that new session is created when recording starts."""
        # Simulate state change to recording
        callback = self.mock_session_manager_instance.add_state_callback.call_args[0][0]
        callback(SessionState.IDLE, SessionState.RECORDING)
        
        # Should have created a session
        info = self.lifecycle_manager.get_current_session_info()
        self.assertIsNotNone(info['session_id'])
        self.assertIsNotNone(info['session_metadata'])
        self.assertIn('created_at', info['session_metadata'])
    
    def test_session_persistence_metadata_only(self):
        """Test session persistence with metadata only."""
        self.config.persistence_level = SessionPersistenceLevel.METADATA_ONLY
        
        # Start a session
        callback = self.mock_session_manager_instance.add_state_callback.call_args[0][0]
        callback(SessionState.IDLE, SessionState.RECORDING)
        
        # Get session info
        info = self.lifecycle_manager.get_current_session_info()
        session_id = info['session_id']
        
        # Force save
        self.lifecycle_manager.force_save()
        
        # Check that file was created
        session_file = self.storage_dir / f"session_{session_id}.json"
        self.assertTrue(session_file.exists())
        
        # Check file content
        with open(session_file, 'r') as f:
            data = json.load(f)
        
        # Should have metadata but no transcriptions
        self.assertEqual(data['session_id'], session_id)
        self.assertEqual(len(data['transcriptions']), 0)
        self.assertIsInstance(data['metadata'], dict)
    
    def test_session_persistence_with_transcriptions(self):
        """Test session persistence including transcriptions."""
        # Mock transcriptions
        mock_transcriptions = [
            {"content": "Hello world", "timestamp": "12:00:00"},
            {"content": "This is a test", "timestamp": "12:00:05"}
        ]
        self.mock_session_manager_instance.get_current_transcriptions.return_value = mock_transcriptions
        
        # Start session and add transcriptions
        callback = self.mock_session_manager_instance.add_state_callback.call_args[0][0]
        callback(SessionState.IDLE, SessionState.RECORDING)
        
        session_id = self.lifecycle_manager.get_current_session_info()['session_id']
        
        # Force save
        self.lifecycle_manager.force_save()
        
        # Check file content
        session_file = self.storage_dir / f"session_{session_id}.json"
        with open(session_file, 'r') as f:
            data = json.load(f)
        
        # Should include transcriptions
        self.assertEqual(len(data['transcriptions']), 2)
        self.assertEqual(data['transcriptions'][0]['content'], "Hello world")
    
    def test_auto_save_functionality(self):
        """Test automatic saving functionality."""
        # Use very short auto-save interval
        self.config.auto_save_interval = 0.1
        
        # Mock recording state so auto-save will trigger
        self.mock_session_manager_instance.current_state = SessionState.RECORDING
        
        # Start recording to trigger session creation
        callback = self.mock_session_manager_instance.add_state_callback.call_args[0][0]
        callback(SessionState.IDLE, SessionState.RECORDING)
        
        session_id = self.lifecycle_manager.get_current_session_info()['session_id']
        session_file = self.storage_dir / f"session_{session_id}.json"
        
        # Wait for auto-save to trigger
        time.sleep(0.3)
        
        # File should exist due to auto-save
        self.assertTrue(session_file.exists())
    
    def test_lifecycle_callbacks(self):
        """Test lifecycle event callbacks."""
        callback_events = []
        
        def test_callback(event_type, data):
            callback_events.append((event_type, data))
        
        self.lifecycle_manager.add_lifecycle_callback(test_callback)
        
        # Trigger recording start
        state_callback = self.mock_session_manager_instance.add_state_callback.call_args[0][0]
        state_callback(SessionState.IDLE, SessionState.RECORDING)
        
        # Should have received callback
        self.assertEqual(len(callback_events), 1)
        self.assertEqual(callback_events[0][0], 'recording_started')
        self.assertIn('session_id', callback_events[0][1])
        
        # Trigger recording stop
        state_callback(SessionState.RECORDING, SessionState.IDLE)
        
        # Should have received another callback
        self.assertEqual(len(callback_events), 2)
        self.assertEqual(callback_events[1][0], 'recording_stopped')
    
    def test_session_error_handling(self):
        """Test session error handling."""
        callback_events = []
        
        def test_callback(event_type, data):
            callback_events.append((event_type, data))
        
        self.lifecycle_manager.add_lifecycle_callback(test_callback)
        
        # Start session
        state_callback = self.mock_session_manager_instance.add_state_callback.call_args[0][0]
        state_callback(SessionState.IDLE, SessionState.RECORDING)
        
        # Verify session was created
        info = self.lifecycle_manager.get_current_session_info()
        session_id = info['session_id']
        self.assertIsNotNone(session_id)
        
        # Directly call the error handler to test the logic
        self.lifecycle_manager._on_session_error()
        
        # Check metadata was updated
        info = self.lifecycle_manager.get_current_session_info()
        self.assertIn('error_count', info['session_metadata'])
        self.assertEqual(info['session_metadata']['error_count'], 1)
        self.assertIn('last_error', info['session_metadata'])
        
        # Should have received error callback
        error_events = [e for e in callback_events if e[0] == 'session_error']
        self.assertEqual(len(error_events), 1)
        self.assertIn('error_count', error_events[0][1])
        self.assertEqual(error_events[0][1]['session_id'], session_id)
    
    def test_session_cleanup(self):
        """Test session cleanup functionality."""
        # Create multiple sessions to test cleanup
        for i in range(12):  # More than max_stored_sessions (10)
            session_file = self.storage_dir / f"session_old_{i}.json"
            session_data = {
                "session_id": f"old_{i}",
                "created_at": (datetime.now() - timedelta(hours=i)).isoformat(),
                "last_updated": (datetime.now() - timedelta(hours=i)).isoformat(),
                "state": "idle",
                "total_recording_time": 0.0,
                "total_transcriptions": 0,
                "transcriptions": [],
                "recording_segments": [],
                "metadata": {}
            }
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f)
        
        # Trigger cleanup by saving a new session
        callback = self.mock_session_manager_instance.add_state_callback.call_args[0][0]
        callback(SessionState.IDLE, SessionState.RECORDING)
        self.lifecycle_manager.force_save()
        
        # Check that old sessions were cleaned up
        remaining_files = list(self.storage_dir.glob("session_*.json"))
        self.assertLessEqual(len(remaining_files), self.config.max_stored_sessions)
    
    def test_session_ending(self):
        """Test proper session ending."""
        callback_events = []
        
        def test_callback(event_type, data):
            callback_events.append((event_type, data))
        
        self.lifecycle_manager.add_lifecycle_callback(test_callback)
        
        # Start session
        state_callback = self.mock_session_manager_instance.add_state_callback.call_args[0][0]
        state_callback(SessionState.IDLE, SessionState.RECORDING)
        
        session_id = self.lifecycle_manager.get_current_session_info()['session_id']
        
        # End session
        self.lifecycle_manager.end_current_session()
        
        # Should have no current session
        info = self.lifecycle_manager.get_current_session_info()
        self.assertIsNone(info['session_id'])
        
        # Should have received ended callback
        ended_events = [e for e in callback_events if e[0] == 'session_ended']
        self.assertEqual(len(ended_events), 1)
        self.assertEqual(ended_events[0][1]['session_id'], session_id)
    
    def test_stored_session_management(self):
        """Test stored session listing and loading."""
        # Create and save a session
        callback = self.mock_session_manager_instance.add_state_callback.call_args[0][0]
        callback(SessionState.IDLE, SessionState.RECORDING)
        
        session_id = self.lifecycle_manager.get_current_session_info()['session_id']
        self.lifecycle_manager.force_save()
        
        # List stored sessions
        stored_sessions = self.lifecycle_manager.list_stored_sessions()
        self.assertEqual(len(stored_sessions), 1)
        self.assertEqual(stored_sessions[0]['session_id'], session_id)
        
        # Load specific session
        snapshot = self.lifecycle_manager.load_session(session_id)
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot.session_id, session_id)
        
        # Delete session
        success = self.lifecycle_manager.delete_stored_session(session_id)
        self.assertTrue(success)
        
        # Should be gone now
        snapshot = self.lifecycle_manager.load_session(session_id)
        self.assertIsNone(snapshot)
    
    def test_no_persistence_mode(self):
        """Test operation with no persistence."""
        self.config.persistence_level = SessionPersistenceLevel.NONE
        
        # Start session
        callback = self.mock_session_manager_instance.add_state_callback.call_args[0][0]
        callback(SessionState.IDLE, SessionState.RECORDING)
        
        session_id = self.lifecycle_manager.get_current_session_info()['session_id']
        
        # Force save should not create file
        self.lifecycle_manager.force_save()
        
        session_file = self.storage_dir / f"session_{session_id}.json"
        self.assertFalse(session_file.exists())
    
    def test_thread_safety(self):
        """Test thread safety of lifecycle operations."""
        callback_count = []
        errors = []
        
        def test_callback(event_type, data):
            callback_count.append(event_type)
        
        self.lifecycle_manager.add_lifecycle_callback(test_callback)
        
        def trigger_events(thread_id):
            try:
                state_callback = self.mock_session_manager_instance.add_state_callback.call_args[0][0]
                
                # Simulate rapid state changes
                for i in range(5):
                    state_callback(SessionState.IDLE, SessionState.RECORDING)
                    time.sleep(0.01)
                    state_callback(SessionState.RECORDING, SessionState.IDLE)
                    time.sleep(0.01)
                    
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")
        
        # Run multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=trigger_events, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Check for errors
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        
        # Should have received callbacks from all threads
        self.assertGreater(len(callback_count), 0)


class TestSessionRecovery(unittest.TestCase):
    """Test session recovery functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self.temp_dir.name) / "recovery_test"
        
        self.config = SessionLifecycleConfig(
            storage_directory=self.storage_dir,
            enable_crash_recovery=True,
            recovery_strategy=SessionRecoveryStrategy.PRESERVE_METADATA
        )
    
    def tearDown(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()
    
    def test_recovery_disabled(self):
        """Test operation with recovery disabled."""
        self.config.enable_crash_recovery = False
        
        # Create an old session file
        old_session = {
            "session_id": "old_session",
            "created_at": datetime.now().isoformat(),
            "last_updated": (datetime.now() - timedelta(minutes=5)).isoformat(),
            "state": "recording",
            "total_recording_time": 60.0,
            "total_transcriptions": 5,
            "transcriptions": [],
            "recording_segments": [],
            "metadata": {}
        }
        
        session_file = self.storage_dir / "session_old_session.json"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        with open(session_file, 'w') as f:
            json.dump(old_session, f)
        
        # Initialize manager - should not recover
        with patch('src.managers.session_lifecycle_manager.EnhancedAudioSessionManager'):
            lifecycle_manager = SessionLifecycleManager(self.config)
            
            info = lifecycle_manager.get_current_session_info()
            self.assertIsNone(info['session_id'])  # Should not have recovered
    
    def test_recovery_age_limit(self):
        """Test that very old sessions are not recovered."""
        # Create an old session file (older than max age)
        old_session = {
            "session_id": "very_old_session",
            "created_at": (datetime.now() - timedelta(days=2)).isoformat(),
            "last_updated": (datetime.now() - timedelta(days=2)).isoformat(),
            "state": "recording",
            "total_recording_time": 60.0,
            "total_transcriptions": 5,
            "transcriptions": [],
            "recording_segments": [],
            "metadata": {}
        }
        
        session_file = self.storage_dir / "session_very_old_session.json"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        with open(session_file, 'w') as f:
            json.dump(old_session, f)
        
        # Initialize manager - should clean up old session
        with patch('src.managers.session_lifecycle_manager.EnhancedAudioSessionManager'):
            lifecycle_manager = SessionLifecycleManager(self.config)
            
            # File should be cleaned up
            self.assertFalse(session_file.exists())
    
    def test_recovery_strategy_discard(self):
        """Test discard recovery strategy."""
        self.config.recovery_strategy = SessionRecoveryStrategy.DISCARD
        
        # Create a recent session file
        recent_session = {
            "session_id": "recent_session",
            "created_at": (datetime.now() - timedelta(minutes=5)).isoformat(),
            "last_updated": (datetime.now() - timedelta(minutes=5)).isoformat(),
            "state": "recording",
            "total_recording_time": 60.0,
            "total_transcriptions": 5,
            "transcriptions": [{"content": "test"}],
            "recording_segments": [],
            "metadata": {}
        }
        
        session_file = self.storage_dir / "session_recent_session.json"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        with open(session_file, 'w') as f:
            json.dump(recent_session, f)
        
        # Initialize manager - should discard
        with patch('src.managers.session_lifecycle_manager.EnhancedAudioSessionManager'):
            lifecycle_manager = SessionLifecycleManager(self.config)
            
            info = lifecycle_manager.get_current_session_info()
            self.assertIsNone(info['session_id'])  # Should not have recovered


if __name__ == '__main__':
    unittest.main()