"""Tests for MeetingHandler."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from src.ui.meeting_handlers import MeetingHandler
import gradio as gr


class TestMeetingHandler(unittest.TestCase):
    """Test cases for MeetingHandler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = MeetingHandler()
    
    def test_create_success_message(self):
        """Test creation of success message HTML."""
        message = "Meeting saved successfully"
        result = self.handler.create_success_message(message)
        
        self.assertIsInstance(result, gr.HTML)
        # Check that the message contains success styling and content
        html_content = result.value
        self.assertIn("✅ Success:", html_content)
        self.assertIn(message, html_content)
        self.assertIn("#155724", html_content)  # Success color
    
    def test_create_error_message(self):
        """Test creation of error message HTML."""
        message = "Meeting name cannot be empty"
        result = self.handler.create_error_message(message)
        
        self.assertIsInstance(result, gr.HTML)
        # Check that the message contains error styling and content
        html_content = result.value
        self.assertIn("❌ Error:", html_content)
        self.assertIn(message, html_content)
        self.assertIn("#721c24", html_content)  # Error color
    
    def test_extract_transcription_from_dialog_dict_format(self):
        """Test extraction of transcription from dict message format."""
        dialog_messages = [
            {"role": "assistant", "content": "Speaker 1: Hello"},
            {"role": "assistant", "content": "Speaker 2: Hi there"},
            {"role": "assistant", "content": ""}  # Empty content should be ignored
        ]
        
        result = self.handler.extract_transcription_from_dialog(dialog_messages)
        
        self.assertEqual(result, "Speaker 1: Hello\nSpeaker 2: Hi there")
    
    def test_extract_transcription_from_dialog_string_format(self):
        """Test extraction of transcription from string message format."""
        dialog_messages = ["First message", "Second message", ""]
        
        result = self.handler.extract_transcription_from_dialog(dialog_messages)
        
        self.assertEqual(result, "First message\nSecond message")
    
    def test_extract_transcription_from_dialog_empty(self):
        """Test extraction from empty dialog messages."""
        result = self.handler.extract_transcription_from_dialog([])
        
        self.assertEqual(result, "")
    
    def test_extract_transcription_from_dialog_none(self):
        """Test extraction from None dialog messages."""
        result = self.handler.extract_transcription_from_dialog(None)
        
        self.assertEqual(result, "")
    
    def test_parse_duration_to_minutes_mm_ss(self):
        """Test parsing MM:SS duration format."""
        result = self.handler.parse_duration_to_minutes("02:30")
        
        self.assertAlmostEqual(result, 2.5, places=2)
    
    def test_parse_duration_to_minutes_hh_mm_ss(self):
        """Test parsing HH:MM:SS duration format."""
        result = self.handler.parse_duration_to_minutes("1:23:45")
        
        expected = 60 + 23 + 45/60  # 83.75 minutes
        self.assertAlmostEqual(result, expected, places=2)
    
    def test_parse_duration_to_minutes_zero(self):
        """Test parsing zero duration."""
        result = self.handler.parse_duration_to_minutes("00:00")
        
        self.assertEqual(result, 0.0)
    
    def test_parse_duration_to_minutes_empty(self):
        """Test parsing empty duration."""
        result = self.handler.parse_duration_to_minutes("")
        
        self.assertEqual(result, 0.0)
    
    def test_parse_duration_to_minutes_invalid_format(self):
        """Test parsing invalid duration format."""
        result = self.handler.parse_duration_to_minutes("invalid:format:too:many")
        
        self.assertEqual(result, 0.0)
    
    def test_parse_duration_to_minutes_invalid_numbers(self):
        """Test parsing duration with invalid numbers."""
        result = self.handler.parse_duration_to_minutes("ab:cd")
        
        self.assertEqual(result, 0.0)
    
    @patch('src.ui.meeting_handlers.save_meeting_to_database')
    @patch('src.ui.meeting_handlers.load_meetings_data')
    def test_submit_new_meeting_success(self, mock_load_meetings, mock_save_meeting):
        """Test successful meeting submission."""
        # Mock successful save
        mock_save_meeting.return_value = (True, "Meeting saved")
        mock_load_meetings.return_value = [["1", "Test Meeting", "2025-01-30", "02:30", "50 words"]]
        
        # Mock dialog messages
        dialog_messages = [
            {"role": "assistant", "content": "Speaker 1: Test transcription"}
        ]
        
        with patch('gradio.Info') as mock_info:
            result = self.handler.submit_new_meeting("Test Meeting", "02:30", dialog_messages)
            
            # Should return (status_message, meeting_list)
            self.assertEqual(len(result), 2)
            status_msg, meeting_list = result
            
            # Status should be empty HTML (success case)
            self.assertIsInstance(status_msg, gr.HTML)
            self.assertEqual(status_msg.value, "")
            
            # Should have called save function
            mock_save_meeting.assert_called_once()
            
            # Should show info notification
            mock_info.assert_called_once()
    
    @patch('src.ui.meeting_handlers.save_meeting_to_database')
    def test_submit_new_meeting_empty_name(self, mock_save_meeting):
        """Test meeting submission with empty name."""
        result = self.handler.submit_new_meeting("", "02:30", [])
        
        # Should return error message
        status_msg, meeting_list = result
        self.assertIsInstance(status_msg, gr.HTML)
        self.assertIn("Meeting name cannot be empty", status_msg.value)
        
        # Should not call save function
        mock_save_meeting.assert_not_called()
    
    @patch('src.ui.meeting_handlers.save_meeting_to_database')
    def test_submit_new_meeting_save_failure(self, mock_save_meeting):
        """Test meeting submission with save failure."""
        # Mock failed save
        mock_save_meeting.return_value = (False, "Database error")
        
        dialog_messages = [{"role": "assistant", "content": "Test"}]
        
        result = self.handler.submit_new_meeting("Test Meeting", "02:30", dialog_messages)
        
        # Should return error message
        status_msg, meeting_list = result
        self.assertIsInstance(status_msg, gr.HTML)
        self.assertIn("Failed to save meeting", status_msg.value)
    
    @patch('src.ui.meeting_handlers.delete_meeting_by_id')
    @patch('src.ui.meeting_handlers.load_meetings_data')
    def test_delete_meeting_by_id_input_success(self, mock_load_meetings, mock_delete_meeting):
        """Test successful meeting deletion by ID."""
        # Mock successful deletion
        mock_delete_meeting.return_value = True
        mock_load_meetings.return_value = [["2", "Other Meeting", "2025-01-30", "01:15", "30 words"]]
        
        with patch('gradio.Info') as mock_info:
            result = self.handler.delete_meeting_by_id_input("1")
            
            # Should return (meeting_list, status_message)
            self.assertEqual(len(result), 2)
            meeting_list, status_msg = result
            
            # Should have called delete function with correct ID
            mock_delete_meeting.assert_called_once_with(1)
            
            # Should show success info notification
            mock_info.assert_called_once()
            
            # Status message should indicate success
            self.assertIn("successfully", status_msg["value"])
    
    @patch('src.ui.meeting_handlers.load_meetings_data')
    def test_delete_meeting_by_id_input_empty(self, mock_load_meetings):
        """Test meeting deletion with empty ID input."""
        mock_load_meetings.return_value = []
        
        result = self.handler.delete_meeting_by_id_input("")
        
        # Should return error status
        meeting_list, status_msg = result
        self.assertIn("Please enter a meeting ID", status_msg["value"])
    
    @patch('src.ui.meeting_handlers.load_meetings_data')
    def test_delete_meeting_by_id_input_invalid_id(self, mock_load_meetings):
        """Test meeting deletion with invalid ID format."""
        mock_load_meetings.return_value = []
        
        result = self.handler.delete_meeting_by_id_input("abc")
        
        # Should return error status
        meeting_list, status_msg = result
        self.assertIn("Invalid meeting ID", status_msg["value"])
    
    @patch('src.ui.meeting_handlers.delete_meeting_by_id')
    @patch('src.ui.meeting_handlers.load_meetings_data')
    def test_delete_meeting_by_id_input_not_found(self, mock_load_meetings, mock_delete_meeting):
        """Test meeting deletion when meeting is not found."""
        # Mock deletion failure
        mock_delete_meeting.return_value = False
        mock_load_meetings.return_value = []
        
        result = self.handler.delete_meeting_by_id_input("999")
        
        # Should return error status
        meeting_list, status_msg = result
        self.assertIn("not found", status_msg["value"])
    
    def test_handle_meeting_row_selection(self):
        """Test meeting row selection handling."""
        # Mock event
        mock_evt = Mock()
        mock_evt.index = 0
        
        # Should not raise any errors (currently just logs)
        result = self.handler.handle_meeting_row_selection(mock_evt)
        
        # Should return None (just logging for now)
        self.assertIsNone(result)
    
    def test_reset_meeting_duration(self):
        """Test meeting duration reset."""
        result = self.handler.reset_meeting_duration()
        
        self.assertEqual(result, "00:00")
    
    @patch('src.ui.meeting_handlers.load_meetings_data')
    def test_delete_meeting_with_confirmation_no_selection(self, mock_load_meetings):
        """Test bulk deletion with no selections."""
        mock_load_meetings.return_value = []
        
        result = self.handler.delete_meeting_with_confirmation([])
        
        # Should return appropriate message
        meeting_list, status_msg = result
        self.assertIn("No meetings selected", status_msg)
    
    def test_extract_transcription_mixed_formats(self):
        """Test extraction with mixed message formats."""
        dialog_messages = [
            {"role": "assistant", "content": "First message"},
            "Second message",  # String format
            {"role": "user", "content": "User message"},  # Should be included
            {"content": "No role"},  # Dict without role - should be ignored
            {"role": "assistant", "content": "   "},  # Whitespace only - should be ignored
        ]
        
        result = self.handler.extract_transcription_from_dialog(dialog_messages)
        
        # Should include all messages with content, including those without roles
        expected_lines = ["First message", "Second message", "User message", "No role"]
        self.assertEqual(result, "\n".join(expected_lines))


if __name__ == '__main__':
    unittest.main()