"""Tests for interface dialog handlers."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from src.ui.interface_dialog_handlers import DialogStateManager, UIUpdateManager, update_dialog_state, combined_update, handle_download_click
import gradio as gr


class TestDialogStateManager(unittest.TestCase):
    """Test cases for DialogStateManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = DialogStateManager()
    
    def test_update_dialog_state_new_message(self):
        """Test adding a new message to empty dialog state."""
        current_messages = []
        new_message = {"content": "Hello world", "timestamp": "2025-01-30 12:00:00"}
        
        updated_messages, gradio_messages = self.manager.update_dialog_state(current_messages, new_message)
        
        self.assertEqual(len(updated_messages), 1)
        self.assertEqual(updated_messages[0]["content"], "Hello world")
        
        self.assertEqual(len(gradio_messages), 1)
        self.assertEqual(gradio_messages[0]["role"], "assistant")
        self.assertEqual(gradio_messages[0]["content"], "Hello world")
    
    def test_update_dialog_state_partial_update(self):
        """Test updating a partial message with same utterance_id."""
        current_messages = [
            {"utterance_id": "123", "content": "Hello", "is_partial": True}
        ]
        new_message = {"utterance_id": "123", "content": "Hello world", "is_partial": True}
        
        updated_messages, gradio_messages = self.manager.update_dialog_state(current_messages, new_message)
        
        self.assertEqual(len(updated_messages), 1)
        self.assertEqual(updated_messages[0]["content"], "Hello world")
        self.assertTrue(updated_messages[0]["is_partial"])
    
    def test_update_dialog_state_final_replaces_partial(self):
        """Test that final result replaces partial result with same utterance_id."""
        current_messages = [
            {"utterance_id": "123", "content": "Hello", "is_partial": True}
        ]
        new_message = {"utterance_id": "123", "content": "Hello world!", "is_partial": False}
        
        updated_messages, gradio_messages = self.manager.update_dialog_state(current_messages, new_message)
        
        self.assertEqual(len(updated_messages), 1)
        self.assertEqual(updated_messages[0]["content"], "Hello world!")
        self.assertFalse(updated_messages[0]["is_partial"])
    
    def test_update_dialog_state_no_utterance_id(self):
        """Test adding message without utterance tracking."""
        current_messages = [
            {"content": "First message"}
        ]
        new_message = {"content": "Second message"}
        
        updated_messages, gradio_messages = self.manager.update_dialog_state(current_messages, new_message)
        
        self.assertEqual(len(updated_messages), 2)
        self.assertEqual(updated_messages[0]["content"], "First message")
        self.assertEqual(updated_messages[1]["content"], "Second message")
    
    def test_update_dialog_state_error_handling(self):
        """Test error handling in dialog state update."""
        current_messages = [{"content": "Existing message"}]
        
        # Simulate error by passing invalid new_message
        with patch('src.ui.interface_dialog_handlers.logger') as mock_logger:
            updated_messages, gradio_messages = self.manager.update_dialog_state(current_messages, None)
            
            # Should return original messages on error
            self.assertEqual(updated_messages, current_messages)
            self.assertEqual(gradio_messages, [])
            mock_logger.error.assert_called_once()
    
    def test_convert_to_gradio_format_dict_messages(self):
        """Test conversion of dict messages to Gradio format."""
        messages = [
            {"content": "First message", "role": "assistant"},
            {"content": "Second message", "timestamp": "12:00"}
        ]
        
        gradio_messages = self.manager._convert_to_gradio_format(messages)
        
        self.assertEqual(len(gradio_messages), 2)
        for msg in gradio_messages:
            self.assertEqual(msg["role"], "assistant")
        self.assertEqual(gradio_messages[0]["content"], "First message")
        self.assertEqual(gradio_messages[1]["content"], "Second message")
    
    def test_convert_to_gradio_format_empty_list(self):
        """Test conversion of empty message list."""
        gradio_messages = self.manager._convert_to_gradio_format([])
        
        self.assertEqual(gradio_messages, [])
    
    def test_convert_to_gradio_format_invalid_messages(self):
        """Test conversion handles invalid message formats gracefully."""
        messages = [
            {"content": "Valid message"},
            {"no_content": "Invalid"},  # No content key
            "string_message",  # String instead of dict
            None  # None value
        ]
        
        gradio_messages = self.manager._convert_to_gradio_format(messages)
        
        # Should only include valid messages with content
        self.assertEqual(len(gradio_messages), 1)
        self.assertEqual(gradio_messages[0]["content"], "Valid message")


class TestUIUpdateManager(unittest.TestCase):
    """Test cases for UIUpdateManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = UIUpdateManager()
    
    @patch('src.ui.interface_dialog_handlers.conditional_update')
    @patch('src.ui.interface_dialog_handlers.update_download_button_visibility')
    @patch('src.ui.interface_dialog_handlers.get_current_duration_display')
    def test_combined_update_success(self, mock_duration, mock_download, mock_conditional):
        """Test successful combined update."""
        # Mock return values
        mock_conditional.return_value = ("dialog_state", "dialog_output")
        mock_download.return_value = {"visible": True}
        mock_duration.return_value = "02:30"
        
        result = self.manager.combined_update()
        
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], "dialog_state")
        self.assertEqual(result[1], "dialog_output")
        self.assertEqual(result[2], {"visible": True})
        self.assertEqual(result[3], "02:30")
        
        mock_conditional.assert_called_once()
        mock_download.assert_called_once()
        mock_duration.assert_called_once()
    
    @patch('src.ui.interface_dialog_handlers.conditional_update')
    @patch('src.ui.interface_dialog_handlers.update_download_button_visibility')
    @patch('src.ui.interface_dialog_handlers.get_current_duration_display')
    def test_combined_update_error_handling(self, mock_duration, mock_download, mock_conditional):
        """Test error handling in combined update."""
        # Make one of the functions raise an exception
        mock_conditional.side_effect = Exception("Test error")
        
        with patch('src.ui.interface_dialog_handlers.logger') as mock_logger:
            result = self.manager.combined_update()
            
            # Should return safe defaults
            self.assertEqual(len(result), 4)
            # Check that gr.skip() is returned (can't test exact object due to Gradio internals)
            self.assertIsNotNone(result[0])
            self.assertIsNotNone(result[1])
            self.assertIsNotNone(result[2])
            self.assertEqual(result[3], "00:00")
            
            mock_logger.error.assert_called_once()
    
    @patch('src.ui.interface_dialog_handlers.download_transcript')
    @patch('src.ui.interface_dialog_handlers.create_download_button')
    def test_handle_download_click_success(self, mock_create_button, mock_download):
        """Test successful download click handling."""
        mock_download.return_value = "/path/to/transcript.txt"
        mock_button = Mock()
        mock_create_button.return_value = mock_button
        
        result = self.manager.handle_download_click()
        
        self.assertEqual(result, mock_button)
        mock_download.assert_called_once()
        mock_create_button.assert_called_once_with("/path/to/transcript.txt")
    
    @patch('src.ui.interface_dialog_handlers.download_transcript')
    def test_handle_download_click_error_handling(self, mock_download):
        """Test error handling in download click."""
        mock_download.side_effect = Exception("Download failed")
        
        with patch('src.ui.interface_dialog_handlers.logger') as mock_logger:
            result = self.manager.handle_download_click()
            
            # Should return default download button
            self.assertIsInstance(result, gr.DownloadButton)
            self.assertEqual(result.label, "Download Transcript")
            self.assertEqual(result.variant, "secondary")
            self.assertFalse(result.visible)
            
            mock_logger.error.assert_called_once()


class TestModuleLevelFunctions(unittest.TestCase):
    """Test module-level wrapper functions."""
    
    @patch('src.ui.interface_dialog_handlers.dialog_state_manager')
    def test_update_dialog_state_wrapper(self, mock_manager):
        """Test that module-level function calls manager method."""
        mock_manager.update_dialog_state.return_value = ("updated", "gradio")
        
        result = update_dialog_state([], {"content": "test"})
        
        self.assertEqual(result, ("updated", "gradio"))
        mock_manager.update_dialog_state.assert_called_once_with([], {"content": "test"})
    
    @patch('src.ui.interface_dialog_handlers.ui_update_manager')
    def test_combined_update_wrapper(self, mock_manager):
        """Test that module-level function calls manager method."""
        mock_manager.combined_update.return_value = ("a", "b", "c", "d")
        
        result = combined_update()
        
        self.assertEqual(result, ("a", "b", "c", "d"))
        mock_manager.combined_update.assert_called_once()
    
    @patch('src.ui.interface_dialog_handlers.ui_update_manager')
    def test_handle_download_click_wrapper(self, mock_manager):
        """Test that module-level function calls manager method."""
        mock_button = Mock()
        mock_manager.handle_download_click.return_value = mock_button
        
        result = handle_download_click()
        
        self.assertEqual(result, mock_button)
        mock_manager.handle_download_click.assert_called_once()


class TestDialogStateComplexScenarios(unittest.TestCase):
    """Test complex dialog state scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = DialogStateManager()
    
    def test_multiple_utterances_interleaved(self):
        """Test handling multiple interleaved utterances."""
        current_messages = []
        
        # Add first utterance partial
        messages1, _ = self.manager.update_dialog_state(
            current_messages, 
            {"utterance_id": "utt1", "content": "Hello", "is_partial": True}
        )
        
        # Add second utterance partial
        messages2, _ = self.manager.update_dialog_state(
            messages1,
            {"utterance_id": "utt2", "content": "World", "is_partial": True}
        )
        
        # Update first utterance final
        messages3, gradio_msgs = self.manager.update_dialog_state(
            messages2,
            {"utterance_id": "utt1", "content": "Hello there", "is_partial": False}
        )
        
        self.assertEqual(len(messages3), 2)
        self.assertEqual(messages3[0]["content"], "Hello there")
        self.assertFalse(messages3[0]["is_partial"])
        self.assertEqual(messages3[1]["content"], "World")
        self.assertTrue(messages3[1]["is_partial"])
    
    def test_gradio_format_preservation(self):
        """Test that Gradio format is correctly maintained."""
        current_messages = [
            {"utterance_id": "1", "content": "First", "timestamp": "12:00"},
            {"utterance_id": "2", "content": "Second", "speaker": "User"}
        ]
        
        _, gradio_messages = self.manager.update_dialog_state(
            current_messages,
            {"content": "Third message"}
        )
        
        # All messages should have assistant role
        self.assertTrue(all(msg["role"] == "assistant" for msg in gradio_messages))
        # Content should be preserved
        contents = [msg["content"] for msg in gradio_messages]
        self.assertIn("First", contents)
        self.assertIn("Second", contents)
        self.assertIn("Third message", contents)


if __name__ == '__main__':
    unittest.main()