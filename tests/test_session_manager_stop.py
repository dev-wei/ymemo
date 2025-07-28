#!/usr/bin/env python3
"""Focused tests for session manager stop functionality."""

import sys
import time
import threading
import asyncio
import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import logging
sys.path.append('/Users/mweiwei/src/ymemo')

from src.managers.session_manager import AudioSessionManager
from src.core.processor import AudioProcessor

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSessionManagerStop(unittest.TestCase):
    """Test session manager stop functionality specifically."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a fresh session manager instance
        self.session_manager = AudioSessionManager()
        self.session_manager.audio_processor = None
        self.session_manager.background_thread = None
        self.session_manager.background_loop = None
    
    def tearDown(self):
        """Clean up after tests."""
        try:
            self.session_manager.stop_recording()
        except:
            pass
    
    def test_stop_recording_calls_audio_processor_stop(self):
        """Test that stop_recording properly calls AudioProcessor.stop_recording()."""
        logger.info("🧪 Testing stop_recording calls audio_processor.stop_recording()")
        
        # Mock audio processor with PyAudio provider
        mock_processor = Mock()
        mock_processor.is_running = True
        
        # Mock stop_recording to set is_running = False when called
        async def mock_stop_recording():
            mock_processor.is_running = False
        
        mock_processor.stop_recording = AsyncMock(side_effect=mock_stop_recording)
        
        # Mock PyAudio capture provider to test stop sequence
        mock_pyaudio_provider = Mock()
        mock_pyaudio_provider.stop_capture = AsyncMock()
        mock_pyaudio_provider._stop_event = Mock()
        mock_pyaudio_provider._stop_event.is_set = Mock(return_value=False)
        mock_pyaudio_provider._stop_event.set = Mock()
        
        mock_processor.capture_provider = mock_pyaudio_provider
        
        # Set up session manager with mock processor
        self.session_manager.audio_processor = mock_processor
        
        # Mock background thread
        mock_thread = Mock()
        mock_thread.is_alive.return_value = False
        self.session_manager.background_thread = mock_thread
        
        # Call stop_recording
        result = self.session_manager.stop_recording()
        
        # Verify results
        self.assertTrue(result)
        self.assertFalse(mock_processor.is_running)
        mock_processor.stop_recording.assert_called_once()
        self.assertIsNone(self.session_manager.audio_processor)
        
        logger.info("✅ stop_recording calls audio_processor.stop_recording() - PASSED")
    
    def test_stop_recording_handles_async_timeout(self):
        """Test that stop_recording handles timeout from AudioProcessor.stop_recording()."""
        logger.info("🧪 Testing stop_recording handles async timeout")
        
        # Mock audio processor with slow stop_recording
        mock_processor = Mock()
        mock_processor.is_running = True
        
        async def slow_stop():
            await asyncio.sleep(5)  # Longer than 3 second timeout
        
        mock_processor.stop_recording = slow_stop
        
        # Set up session manager with mock processor
        self.session_manager.audio_processor = mock_processor
        
        # Mock background thread
        mock_thread = Mock()
        mock_thread.is_alive.return_value = False
        self.session_manager.background_thread = mock_thread
        
        # Call stop_recording with timeout
        start_time = time.time()
        result = self.session_manager.stop_recording()
        end_time = time.time()
        
        # Verify results
        self.assertTrue(result)  # Should still return True
        self.assertLess(end_time - start_time, 6.0)  # Should timeout around 5 seconds
        self.assertIsNone(self.session_manager.audio_processor)  # Should still clean up
        
        logger.info("✅ stop_recording handles async timeout - PASSED")
    
    def test_stop_recording_handles_async_exception(self):
        """Test that stop_recording handles exception from AudioProcessor.stop_recording()."""
        logger.info("🧪 Testing stop_recording handles async exception")
        
        # Mock audio processor with failing stop_recording
        mock_processor = Mock()
        mock_processor.is_running = True
        
        async def failing_stop():
            raise RuntimeError("Test exception in stop_recording")
        
        mock_processor.stop_recording = failing_stop
        
        # Set up session manager with mock processor
        self.session_manager.audio_processor = mock_processor
        
        # Mock background thread
        mock_thread = Mock()
        mock_thread.is_alive.return_value = False
        self.session_manager.background_thread = mock_thread
        
        # Call stop_recording - should handle exception gracefully
        result = self.session_manager.stop_recording()
        
        # Verify results
        self.assertTrue(result)  # Should still return True
        self.assertIsNone(self.session_manager.audio_processor)  # Should still clean up
        
        logger.info("✅ stop_recording handles async exception - PASSED")
    
    def test_stop_recording_waits_for_background_thread(self):
        """Test that stop_recording waits for background thread to finish."""
        logger.info("🧪 Testing stop_recording waits for background thread")
        
        # Mock audio processor
        mock_processor = Mock()
        mock_processor.is_running = True
        mock_processor.stop_recording = AsyncMock()
        
        # Set up session manager with mock processor
        self.session_manager.audio_processor = mock_processor
        
        # Mock background thread that takes time to finish
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True
        mock_thread.join = Mock()
        self.session_manager.background_thread = mock_thread
        
        # Call stop_recording
        result = self.session_manager.stop_recording()
        
        # Verify results
        self.assertTrue(result)
        mock_thread.join.assert_called_once_with(timeout=0.5)
        self.assertIsNone(self.session_manager.background_thread)
        
        logger.info("✅ stop_recording waits for background thread - PASSED")
    
    def test_stop_recording_thread_safety(self):
        """Test that stop_recording is thread-safe."""
        logger.info("🧪 Testing stop_recording thread safety")
        
        # Mock audio processor
        mock_processor = Mock()
        mock_processor.is_running = True
        mock_processor.stop_recording = AsyncMock()
        
        # Set up session manager with mock processor
        self.session_manager.audio_processor = mock_processor
        
        # Mock background thread
        mock_thread = Mock()
        mock_thread.is_alive.return_value = False
        self.session_manager.background_thread = mock_thread
        
        # Call stop_recording from multiple threads
        results = []
        exceptions = []
        
        def stop_worker():
            try:
                result = self.session_manager.stop_recording()
                results.append(result)
            except Exception as e:
                exceptions.append(e)
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=stop_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        self.assertEqual(len(exceptions), 0)  # No exceptions
        self.assertEqual(len(results), 5)     # All threads completed
        self.assertEqual(sum(results), 1)     # Only one successful stop
        
        # stop_recording should only be called once
        mock_processor.stop_recording.assert_called_once()
        
        logger.info("✅ stop_recording thread safety - PASSED")
    
    def test_stop_recording_cleans_up_properly(self):
        """Test that stop_recording cleans up all resources properly."""
        logger.info("🧪 Testing stop_recording cleans up properly")
        
        # Mock audio processor
        mock_processor = Mock()
        mock_processor.is_running = True
        mock_processor.stop_recording = AsyncMock()
        
        # Mock background thread and loop
        mock_thread = Mock()
        mock_thread.is_alive.return_value = False
        mock_loop = Mock()
        
        # Set up session manager
        self.session_manager.audio_processor = mock_processor
        self.session_manager.background_thread = mock_thread
        self.session_manager.background_loop = mock_loop
        
        # Call stop_recording
        result = self.session_manager.stop_recording()
        
        # Verify cleanup
        self.assertTrue(result)
        self.assertIsNone(self.session_manager.audio_processor)
        self.assertIsNone(self.session_manager.background_thread)
        self.assertIsNone(self.session_manager.background_loop)
        
        logger.info("✅ stop_recording cleans up properly - PASSED")
    
    def test_stop_recording_pyaudio_provider_stop_sequence(self):
        """Test that stop_recording properly calls PyAudio provider stop sequence."""
        logger.info("🧪 Testing stop_recording PyAudio provider stop sequence")
        
        # Mock PyAudio stream
        mock_stream = Mock()
        mock_stream.is_active.return_value = True
        mock_stream.stop_stream = Mock()
        mock_stream.close = Mock()
        
        # Mock PyAudio capture provider 
        mock_pyaudio_provider = Mock()
        mock_pyaudio_provider.stop_capture = AsyncMock()
        mock_pyaudio_provider.stream = mock_stream
        mock_pyaudio_provider._stop_event = Mock()
        mock_pyaudio_provider._stop_event.set = Mock()
        mock_pyaudio_provider._capture_thread = Mock()
        mock_pyaudio_provider._capture_thread.is_alive.return_value = False
        
        # Mock audio processor
        mock_processor = Mock()
        mock_processor.is_running = True
        mock_processor.capture_provider = mock_pyaudio_provider
        
        # Mock the stop_recording to simulate calling provider stop
        async def mock_stop_recording():
            mock_processor.is_running = False
            await mock_pyaudio_provider.stop_capture()
        
        mock_processor.stop_recording = AsyncMock(side_effect=mock_stop_recording)
        
        # Set up session manager
        self.session_manager.audio_processor = mock_processor
        
        # Mock background thread
        mock_thread = Mock()
        mock_thread.is_alive.return_value = False
        self.session_manager.background_thread = mock_thread
        
        # Call stop_recording
        result = self.session_manager.stop_recording()
        
        # Verify results
        self.assertTrue(result)
        self.assertFalse(mock_processor.is_running)
        mock_pyaudio_provider.stop_capture.assert_called_once()
        self.assertIsNone(self.session_manager.audio_processor)
        
        logger.info("✅ stop_recording PyAudio provider stop sequence - PASSED")
    
    def test_stop_recording_background_thread_termination(self):
        """Test that stop_recording properly terminates background thread."""
        logger.info("🧪 Testing stop_recording background thread termination")
        
        # Create a mock background thread that simulates proper termination
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True
        mock_thread.join = Mock()
        
        # Mock audio processor
        mock_processor = Mock()
        mock_processor.is_running = True
        mock_processor.stop_recording = AsyncMock()
        
        # Set up session manager
        self.session_manager.audio_processor = mock_processor
        self.session_manager.background_thread = mock_thread
        
        # Simulate thread finishing after join
        def mock_join(timeout=None):
            # After join is called, thread should no longer be alive
            mock_thread.is_alive.return_value = False
        
        mock_thread.join.side_effect = mock_join
        
        # Call stop_recording
        result = self.session_manager.stop_recording()
        
        # Verify results
        self.assertTrue(result)
        mock_processor.stop_recording.assert_called_once()
        mock_thread.join.assert_called_once()
        self.assertIsNone(self.session_manager.background_thread)
        
        logger.info("✅ stop_recording background thread termination - PASSED")
    
    def test_stop_recording_hanging_background_thread(self):
        """Test that stop_recording handles hanging background thread."""
        logger.info("🧪 Testing stop_recording hanging background thread")
        
        # Create a mock background thread that hangs
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True  # Always alive (hanging)
        mock_thread.join = Mock()
        
        # Mock audio processor
        mock_processor = Mock()
        mock_processor.is_running = True
        mock_processor.stop_recording = AsyncMock()
        
        # Set up session manager
        self.session_manager.audio_processor = mock_processor
        self.session_manager.background_thread = mock_thread
        
        # Call stop_recording
        result = self.session_manager.stop_recording()
        
        # Verify results
        self.assertTrue(result)  # Should still return True
        mock_processor.stop_recording.assert_called_once()
        mock_thread.join.assert_called_once()
        self.assertIsNone(self.session_manager.background_thread)  # Should clear reference
        
        logger.info("✅ stop_recording hanging background thread - PASSED")


def run_session_manager_stop_tests():
    """Run session manager stop tests."""
    print("🎯 Running Session Manager Stop Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTest(TestSessionManagerStop('test_stop_recording_calls_audio_processor_stop'))
    suite.addTest(TestSessionManagerStop('test_stop_recording_handles_async_timeout'))
    suite.addTest(TestSessionManagerStop('test_stop_recording_handles_async_exception'))
    suite.addTest(TestSessionManagerStop('test_stop_recording_waits_for_background_thread'))
    suite.addTest(TestSessionManagerStop('test_stop_recording_thread_safety'))
    suite.addTest(TestSessionManagerStop('test_stop_recording_cleans_up_properly'))
    suite.addTest(TestSessionManagerStop('test_stop_recording_pyaudio_provider_stop_sequence'))
    suite.addTest(TestSessionManagerStop('test_stop_recording_background_thread_termination'))
    suite.addTest(TestSessionManagerStop('test_stop_recording_hanging_background_thread'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"🎉 SESSION MANAGER STOP TESTS RESULTS:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, trace in result.failures:
            print(f"   {test}: {trace}")
    
    if result.errors:
        print("\n❌ ERRORS:")
        for test, trace in result.errors:
            print(f"   {test}: {trace}")
    
    if result.wasSuccessful():
        print("\n✅ ALL SESSION MANAGER STOP TESTS PASSED!")
        return True
    else:
        print("\n❌ SOME TESTS FAILED!")
        return False


if __name__ == "__main__":
    success = run_session_manager_stop_tests()
    sys.exit(0 if success else 1)