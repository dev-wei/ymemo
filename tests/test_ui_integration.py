"""Integration tests for UI component separation and interaction."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import gradio as gr
from src.utils.status_manager import AudioStatus


class TestUIComponentSeparation(unittest.TestCase):
    """Test that UI components work together correctly after modularization."""
    
    @patch('src.ui.recording_handlers.get_audio_session')
    @patch('src.ui.recording_handlers.status_manager')
    @patch('src.ui.recording_handlers.button_state_manager')
    @patch('src.ui.recording_handlers.get_audio_devices')
    @patch('src.ui.recording_handlers.get_default_device_index')
    def test_recording_handler_integration(self, mock_default_device, mock_devices, mock_button_mgr, mock_status_mgr, mock_session):
        """Test that recording handler integrates properly with other components."""
        from src.ui.recording_handlers import RecordingHandler
        
        # Setup device mocks
        mock_devices.return_value = [("Test Device", 1), ("Another Device", 2)]
        mock_default_device.return_value = 1
        
        # Setup session mocks
        mock_audio_session = Mock()
        mock_session.return_value = mock_audio_session
        mock_audio_session.start_recording.return_value = True
        mock_audio_session.is_recording.return_value = False  # Not currently recording
        
        # Setup status manager mocks
        mock_status_mgr.set_initializing = Mock()
        mock_status_mgr.set_connecting = Mock() 
        mock_status_mgr.set_recording = Mock()
        mock_status_mgr.get_status_message.return_value = "Recording started"
        
        # Setup button manager mocks
        mock_button_mgr.get_safe_fallback_updates.return_value = {
            "start_btn": gr.update(interactive=False),
            "stop_btn": gr.update(interactive=True),
            "save_btn": gr.update(interactive=False)
        }
        
        # The _update_button_states method returns a tuple, so we need to mock it properly
        def mock_update_button_states(handler_self):
            return (
                gr.update(interactive=False),  # start_btn
                gr.update(interactive=True),   # stop_btn
                gr.update(interactive=False)   # save_btn
            )
        
        # Patch the _update_button_states method specifically
        with patch.object(RecordingHandler, '_update_button_states', mock_update_button_states):
            handler = RecordingHandler()
            
            # Test start recording integration
            result = handler.start_recording(1, [])
            
            # Should return tuple with 6 elements
            self.assertEqual(len(result), 6)
            
            # Should call session manager with device index and config
            mock_audio_session.start_recording.assert_called_once()
            call_args = mock_audio_session.start_recording.call_args[0]
            self.assertEqual(call_args[0], 1)  # device_index
            
            # Should update status through various states
            mock_status_mgr.set_initializing.assert_called_once()
            mock_status_mgr.set_connecting.assert_called_once()
            mock_status_mgr.set_recording.assert_called_once()
            
            # Should check if already recording
            mock_audio_session.is_recording.assert_called()
    
    @patch('src.ui.meeting_handlers.save_meeting_to_database')
    @patch('src.ui.meeting_handlers.load_meetings_data')  
    def test_meeting_handler_integration(self, mock_load_meetings, mock_save_meeting):
        """Test that meeting handler integrates with database operations."""
        from src.ui.meeting_handlers import MeetingHandler
        
        # Setup mocks
        mock_save_meeting.return_value = (True, "Meeting saved successfully")
        mock_load_meetings.return_value = [["1", "Test Meeting", "2025-01-30", "02:30", "Test content"]]
        
        handler = MeetingHandler()
        
        # Test dialog messages in different formats
        dialog_messages = [
            {"role": "assistant", "content": "Speaker 1: Hello"},
            {"role": "assistant", "content": "Speaker 2: Hi there"}
        ]
        
        with patch('gradio.Info') as mock_info:
            result = handler.submit_new_meeting("Integration Test", "02:30", dialog_messages)
            
            # Should return status message and meeting list
            self.assertEqual(len(result), 2)
            status_msg, meeting_list = result
            
            # Should save to database
            mock_save_meeting.assert_called_once()
            save_call = mock_save_meeting.call_args[1]
            self.assertEqual(save_call['meeting_name'], "Integration Test")
            self.assertEqual(save_call['duration'], 2.5)  # 02:30 in minutes
            self.assertIn("Speaker 1: Hello", save_call['transcription'])
            self.assertIn("Speaker 2: Hi there", save_call['transcription'])
            
            # Should show success notification
            mock_info.assert_called_once()
            
            # Should refresh meeting list
            mock_load_meetings.assert_called_once()
    
    @patch('src.ui.interface_dialog_handlers.conditional_update')
    @patch('src.ui.interface_dialog_handlers.update_download_button_visibility')
    @patch('src.ui.interface_dialog_handlers.get_current_duration_display')
    def test_dialog_handler_timer_integration(self, mock_duration, mock_download, mock_conditional):
        """Test that dialog handlers integrate with timer updates."""
        from src.ui.interface_dialog_handlers import UIUpdateManager
        
        # Setup mocks
        mock_conditional.return_value = (["msg1", "msg2"], [{"role": "assistant", "content": "msg1"}])
        mock_download.return_value = gr.update(visible=True)
        mock_duration.return_value = "03:45"
        
        manager = UIUpdateManager()
        result = manager.combined_update()
        
        # Should integrate all timer updates
        self.assertEqual(len(result), 4)
        
        # Check each component was called
        mock_conditional.assert_called_once()
        mock_download.assert_called_once()
        mock_duration.assert_called_once()
        
        # Check return values
        self.assertEqual(result[0], ["msg1", "msg2"])  # dialog_state
        self.assertEqual(result[1], [{"role": "assistant", "content": "msg1"}])  # dialog_output
        self.assertEqual(result[2]["visible"], True)  # download_button (gr.update includes __type__)
        self.assertEqual(result[3], "03:45")  # duration_display
    
    def test_button_state_manager_audio_status_integration(self):
        """Test ButtonStateManager integrates with AudioStatus enum."""
        from src.ui.button_state_manager import ButtonStateManager
        
        manager = ButtonStateManager()
        
        # Test all AudioStatus values are handled
        for status in AudioStatus:
            configs = manager.get_button_configs(status)
            self.assertIn('start_btn', configs)
            self.assertIn('stop_btn', configs) 
            self.assertIn('save_btn', configs)
            
            # Each config should have required attributes
            for btn_name, config in configs.items():
                self.assertTrue(hasattr(config, 'text'))
                self.assertTrue(hasattr(config, 'variant'))
                self.assertTrue(hasattr(config, 'interactive'))
    
    def test_cross_module_imports(self):
        """Test that modules can import from each other without circular dependencies."""
        try:
            # Test recording handlers can import from interface_handlers
            from src.ui.recording_handlers import RecordingHandler
            
            # Test meeting handlers can import utilities
            from src.ui.meeting_handlers import MeetingHandler
            
            # Test interface_handlers can import from specialized modules
            from src.ui.interface_handlers import start_recording, stop_recording
            
            # Test interface can import from all modules
            from src.ui.interface import create_interface
            
            # Test dialog handlers work independently  
            from src.ui.interface_dialog_handlers import DialogStateManager, UIUpdateManager
            
        except ImportError as e:
            self.fail(f"Import error detected: {e}")


class TestUIStateConsistency(unittest.TestCase):
    """Test state consistency across UI components."""
    
    def test_dialog_state_message_formats(self):
        """Test that dialog state handles consistent message formats."""
        from src.ui.interface_dialog_handlers import DialogStateManager
        from src.ui.meeting_handlers import MeetingHandler
        
        dialog_manager = DialogStateManager()
        meeting_manager = MeetingHandler()
        
        # Create dialog messages in standard format
        messages = []
        test_message = {"content": "Test transcription", "utterance_id": "123", "is_partial": False}
        
        messages, gradio_msgs = dialog_manager.update_dialog_state(messages, test_message)
        
        # Meeting handler should be able to extract transcription from these messages
        transcription = meeting_manager.extract_transcription_from_dialog(gradio_msgs)
        
        self.assertEqual(transcription, "Test transcription")
        
        # Test with partial update
        partial_message = {"content": "Test partial", "utterance_id": "123", "is_partial": True}
        messages, gradio_msgs = dialog_manager.update_dialog_state(messages, partial_message)
        
        # Should replace previous message
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["content"], "Test partial")
    
    @patch('src.ui.interface_handlers.get_audio_session')
    def test_session_manager_consistency(self, mock_get_session):
        """Test that session manager state is consistent across handlers."""
        from src.ui.recording_handlers import RecordingHandler
        from src.ui.interface_handlers import get_latest_dialog_state
        
        # Setup mock session
        mock_session = Mock()
        mock_get_session.return_value = mock_session
        mock_session.get_current_transcriptions.return_value = [
            {"content": "Test message 1"},
            {"content": "Test message 2"}
        ]
        
        # Both handlers should get same session state
        handler = RecordingHandler()
        
        # Get state through interface handler
        state1, gradio1 = get_latest_dialog_state()
        
        # Verify consistency
        self.assertEqual(len(state1), 2)
        self.assertEqual(len(gradio1), 2)
        self.assertEqual(state1[0]["content"], "Test message 1")
        self.assertEqual(gradio1[1]["content"], "Test message 2")


class TestErrorHandling(unittest.TestCase):
    """Test error handling across UI components."""
    
    def test_recording_handler_error_propagation(self):
        """Test that recording handler properly handles and logs errors."""
        from src.ui.recording_handlers import RecordingHandler
        
        handler = RecordingHandler()
        
        with patch('src.ui.recording_handlers.get_audio_session') as mock_session:
            mock_session.side_effect = Exception("Session error")
            
            with patch('src.ui.recording_handlers.logger') as mock_logger:
                result = handler.start_recording(1, [])
                
                # Should handle error gracefully
                self.assertIsNotNone(result)
                
                # Should log error
                mock_logger.error.assert_called()
    
    def test_meeting_handler_validation_errors(self):
        """Test meeting handler validation and error messages."""
        from src.ui.meeting_handlers import MeetingHandler
        
        handler = MeetingHandler()
        
        # Test empty name validation
        result = handler.submit_new_meeting("", "02:30", [])
        status_msg, meeting_list = result
        
        self.assertIn("Meeting name cannot be empty", status_msg.value)
        
        # Test invalid duration handling
        duration_minutes = handler.parse_duration_to_minutes("invalid:format:too:many:colons")
        self.assertEqual(duration_minutes, 0.0)
    
    def test_dialog_handler_error_recovery(self):
        """Test dialog handler recovers from errors gracefully."""
        from src.ui.interface_dialog_handlers import DialogStateManager
        
        manager = DialogStateManager()
        
        # Test with None message (should not crash)
        current_messages = [{"content": "existing"}]
        messages, gradio = manager.update_dialog_state(current_messages, None)
        
        # Should return original state on error
        self.assertEqual(messages, current_messages)
        self.assertEqual(gradio, [])


class TestPerformanceIntegration(unittest.TestCase):
    """Test performance aspects of UI integration."""
    
    def test_button_state_caching(self):
        """Test that button state manager efficiently handles repeated calls."""
        from src.ui.button_state_manager import ButtonStateManager
        
        manager = ButtonStateManager()
        
        # Multiple calls should be efficient (no complex computation each time)
        import time
        start_time = time.time()
        
        for _ in range(100):
            configs = manager.get_button_configs(AudioStatus.IDLE)
            updates = manager.get_button_update_tuple(AudioStatus.RECORDING)
        
        elapsed = time.time() - start_time
        
        # Should complete quickly (< 1 second for 200 operations)
        self.assertLess(elapsed, 1.0)
    
    def test_dialog_state_memory_efficiency(self):
        """Test that dialog state doesn't grow unboundedly."""
        from src.ui.interface_dialog_handlers import DialogStateManager
        
        manager = DialogStateManager()
        messages = []
        
        # Add many messages with same utterance_id (should replace, not accumulate)
        for i in range(100):
            test_message = {
                "content": f"Message {i}",
                "utterance_id": "test_utterance",
                "is_partial": True
            }
            messages, _ = manager.update_dialog_state(messages, test_message)
        
        # Should only have 1 message (latest replacement)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["content"], "Message 99")


if __name__ == '__main__':
    unittest.main()