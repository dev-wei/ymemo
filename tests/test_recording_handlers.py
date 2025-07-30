"""Tests for RecordingHandler."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from src.ui.recording_handlers import RecordingHandler
from src.utils.status_manager import AudioStatus


class TestRecordingHandler(unittest.TestCase):
    """Test cases for RecordingHandler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = RecordingHandler()
    
    def test_validate_device_selection_with_int(self):
        """Test device validation with integer input."""
        with patch.object(self.handler, '_get_device_choices_and_default', return_value=([("Device 1", 0), ("Device 2", 1)], 0)):
            is_valid, device_index, error_msg = self.handler._validate_device_selection(0)
            
            self.assertTrue(is_valid)
            self.assertEqual(device_index, 0)
            self.assertEqual(error_msg, "")
    
    def test_validate_device_selection_with_string_number(self):
        """Test device validation with string number input."""
        with patch.object(self.handler, '_get_device_choices_and_default', return_value=([("Device 1", 0), ("Device 2", 1)], 0)):
            is_valid, device_index, error_msg = self.handler._validate_device_selection("1")
            
            self.assertTrue(is_valid)
            self.assertEqual(device_index, 1)
            self.assertEqual(error_msg, "")
    
    def test_validate_device_selection_with_device_name(self):
        """Test device validation with device name."""
        with patch.object(self.handler, '_get_device_choices_and_default', return_value=([("Device 1", 0), ("Device 2", 1)], 0)):
            is_valid, device_index, error_msg = self.handler._validate_device_selection("Device 2")
            
            self.assertTrue(is_valid)
            self.assertEqual(device_index, 1)
            self.assertEqual(error_msg, "")
    
    def test_validate_device_selection_invalid_index(self):
        """Test device validation with invalid index."""
        with patch.object(self.handler, '_get_device_choices_and_default', return_value=([("Device 1", 0), ("Device 2", 1)], 0)):
            is_valid, device_index, error_msg = self.handler._validate_device_selection(-1)
            
            self.assertFalse(is_valid)
            self.assertEqual(device_index, -1)
            self.assertIn("Invalid device selection", error_msg)
    
    def test_validate_device_selection_unavailable_device(self):
        """Test device validation with unavailable device index."""
        with patch.object(self.handler, '_get_device_choices_and_default', return_value=([("Device 1", 0), ("Device 2", 1)], 0)):
            is_valid, device_index, error_msg = self.handler._validate_device_selection(5)
            
            self.assertFalse(is_valid)
            self.assertEqual(device_index, 5)
            self.assertIn("not available", error_msg)
    
    def test_prepare_gradio_messages(self):
        """Test conversion of state to Gradio messages."""
        preserved_state = [
            {"content": "Speaker 1: Hello"},
            {"content": "Speaker 2: Hi there"},
        ]
        
        gradio_messages = self.handler._prepare_gradio_messages(preserved_state)
        
        self.assertEqual(len(gradio_messages), 2)
        self.assertEqual(gradio_messages[0]["role"], "assistant")
        self.assertEqual(gradio_messages[0]["content"], "Speaker 1: Hello")
        self.assertEqual(gradio_messages[1]["role"], "assistant")
        self.assertEqual(gradio_messages[1]["content"], "Speaker 2: Hi there")
    
    def test_prepare_gradio_messages_empty_state(self):
        """Test conversion of empty state to Gradio messages."""
        gradio_messages = self.handler._prepare_gradio_messages([])
        
        self.assertEqual(len(gradio_messages), 0)
        self.assertIsInstance(gradio_messages, list)
    
    @patch('src.ui.recording_handlers.status_manager')
    @patch('src.ui.recording_handlers.get_audio_session')
    def test_start_recording_success(self, mock_get_session, mock_status_manager):
        """Test successful recording start."""
        # Mock audio session
        mock_session = Mock()
        mock_session.is_recording.return_value = False
        mock_session.start_recording.return_value = True
        mock_get_session.return_value = mock_session
        
        # Mock device validation
        with patch.object(self.handler, '_validate_device_selection', return_value=(True, 0, "")):
            with patch.object(self.handler, '_get_device_choices_and_default', return_value=([("Device 1", 0)], 0)):
                with patch.object(self.handler, '_update_button_states', return_value=(Mock(), Mock(), Mock())):
                    
                    result = self.handler.start_recording(0, [])
                    
                    # Should return 6 elements
                    self.assertEqual(len(result), 6)
                    
                    # Verify session methods were called
                    mock_session.start_recording.assert_called_once()
                    mock_status_manager.set_initializing.assert_called_once()
                    mock_status_manager.set_connecting.assert_called_once()
                    mock_status_manager.set_recording.assert_called_once()
    
    @patch('src.ui.recording_handlers.status_manager')
    @patch('src.ui.recording_handlers.get_audio_session')
    def test_start_recording_already_recording(self, mock_get_session, mock_status_manager):
        """Test start recording when already recording."""
        # Mock audio session - already recording
        mock_session = Mock()
        mock_session.is_recording.return_value = True
        mock_get_session.return_value = mock_session
        
        # Mock device validation
        with patch.object(self.handler, '_validate_device_selection', return_value=(True, 0, "")):
            with patch.object(self.handler, '_get_device_choices_and_default', return_value=([("Device 1", 0)], 0)):
                with patch.object(self.handler, '_update_button_states', return_value=(Mock(), Mock(), Mock())):
                    
                    result = self.handler.start_recording(0, [])
                    
                    # Should return 6 elements
                    self.assertEqual(len(result), 6)
                    
                    # Should not try to start recording
                    mock_session.start_recording.assert_not_called()
                    
                    # Should set error status
                    mock_status_manager.set_error.assert_called()
    
    @patch('src.ui.recording_handlers.status_manager')
    @patch('src.ui.recording_handlers.get_audio_session')
    def test_start_recording_invalid_device(self, mock_get_session, mock_status_manager):
        """Test start recording with invalid device."""
        # Mock audio session
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        
        # Mock invalid device validation
        with patch.object(self.handler, '_validate_device_selection', return_value=(False, -1, "Invalid device")):
            with patch.object(self.handler, '_update_button_states', return_value=(Mock(), Mock(), Mock())):
                
                result = self.handler.start_recording(-1, [])
                
                # Should return 6 elements
                self.assertEqual(len(result), 6)
                
                # Should not try to start recording
                mock_session.start_recording.assert_not_called()
                
                # Should set error status
                mock_status_manager.set_error.assert_called()
    
    @patch('src.ui.recording_handlers.status_manager')
    @patch('src.ui.recording_handlers.get_audio_session')
    def test_stop_recording_success(self, mock_get_session, mock_status_manager):
        """Test successful recording stop."""
        # Mock audio session
        mock_session = Mock()
        mock_session.stop_recording.return_value = True
        mock_get_session.return_value = mock_session
        
        with patch.object(self.handler, '_update_button_states', return_value=(Mock(), Mock(), Mock())):
            result = self.handler.stop_recording()
            
            # Should return 4 elements
            self.assertEqual(len(result), 4)
            
            # Verify session methods were called
            mock_session.stop_recording.assert_called_once()
            mock_status_manager.set_stopping.assert_called_once()
            mock_status_manager.set_stopped.assert_called_once()
    
    @patch('src.ui.recording_handlers.status_manager')
    @patch('src.ui.recording_handlers.get_audio_session')
    def test_stop_recording_failure(self, mock_get_session, mock_status_manager):
        """Test recording stop failure."""
        # Mock audio session - stop fails
        mock_session = Mock()
        mock_session.stop_recording.return_value = False
        mock_get_session.return_value = mock_session
        
        with patch.object(self.handler, '_update_button_states', return_value=(Mock(), Mock(), Mock())):
            result = self.handler.stop_recording()
            
            # Should return 4 elements
            self.assertEqual(len(result), 4)
            
            # Verify session methods were called
            mock_session.stop_recording.assert_called_once()
            mock_status_manager.set_stopping.assert_called_once()
            
            # Should set error status instead of stopped
            mock_status_manager.set_error.assert_called()
            mock_status_manager.set_stopped.assert_not_called()
    
    @patch('src.ui.recording_handlers.status_manager')
    def test_stop_recording_exception(self, mock_status_manager):
        """Test recording stop with exception."""
        # Mock get_audio_session to raise exception
        with patch.object(self.handler, '_get_audio_session', side_effect=Exception("Test error")):
            with patch.object(self.handler, '_update_button_states', return_value=(Mock(), Mock(), Mock())):
                result = self.handler.stop_recording()
                
                # Should return 4 elements
                self.assertEqual(len(result), 4)
                
                # Should set error status
                mock_status_manager.set_error.assert_called()
    
    def test_get_device_choices_and_default_success(self):
        """Test successful device choices retrieval."""
        with patch('src.ui.recording_handlers.get_audio_devices', return_value=[("Device 1", 0), ("Device 2", 1)]):
            with patch('src.ui.recording_handlers.get_default_device_index', return_value=0):
                
                devices, default = self.handler._get_device_choices_and_default()
                
                self.assertEqual(len(devices), 2)
                self.assertEqual(default, 0)
                self.assertEqual(devices[0][0], "Device 1")
                self.assertEqual(devices[0][1], 0)
    
    def test_get_device_choices_and_default_no_devices(self):
        """Test device choices retrieval with no devices."""
        with patch('src.ui.recording_handlers.get_audio_devices', return_value=[]):
            
            devices, default = self.handler._get_device_choices_and_default()
            
            self.assertEqual(len(devices), 1)
            self.assertEqual(devices[0], ("No devices available", -1))
            self.assertEqual(default, -1)
    
    def test_get_device_choices_and_default_exception(self):
        """Test device choices retrieval with exception."""
        with patch('src.ui.recording_handlers.get_audio_devices', side_effect=Exception("Test error")):
            
            devices, default = self.handler._get_device_choices_and_default()
            
            self.assertEqual(len(devices), 1)
            self.assertIn("Error", devices[0][0])
            self.assertEqual(default, -1)


if __name__ == '__main__':
    unittest.main()