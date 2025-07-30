"""Tests for ButtonStateManager."""

import unittest
from src.ui.button_state_manager import ButtonStateManager, ButtonConfig
from src.utils.status_manager import AudioStatus


class TestButtonStateManager(unittest.TestCase):
    """Test cases for ButtonStateManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = ButtonStateManager()
    
    def test_idle_state_configuration(self):
        """Test button configuration for IDLE state."""
        configs = self.manager.get_button_configs(AudioStatus.IDLE)
        
        # Start button should be interactive and primary
        self.assertEqual(configs["start_btn"].text, "🎤 Start Recording")
        self.assertEqual(configs["start_btn"].variant, "primary")
        self.assertTrue(configs["start_btn"].interactive)
        self.assertTrue(configs["start_btn"].visible)
        
        # Stop button should not be interactive
        self.assertEqual(configs["stop_btn"].text, "⏹️ Stop Recording")
        self.assertEqual(configs["stop_btn"].variant, "secondary")
        self.assertFalse(configs["stop_btn"].interactive)
        
        # Save button should not be interactive
        self.assertEqual(configs["save_btn"].text, "💾 Save as New Meeting")
        self.assertEqual(configs["save_btn"].variant, "secondary")
        self.assertFalse(configs["save_btn"].interactive)
    
    def test_recording_state_configuration(self):
        """Test button configuration for RECORDING state."""
        configs = self.manager.get_button_configs(AudioStatus.RECORDING)
        
        # Start button should not be interactive
        self.assertFalse(configs["start_btn"].interactive)
        self.assertEqual(configs["start_btn"].variant, "secondary")
        
        # Stop button should be interactive and primary
        self.assertTrue(configs["stop_btn"].interactive)
        self.assertEqual(configs["stop_btn"].variant, "primary")
        
        # Save button should not be interactive
        self.assertFalse(configs["save_btn"].interactive)
        self.assertEqual(configs["save_btn"].variant, "secondary")
    
    def test_stopped_state_configuration(self):
        """Test button configuration for STOPPED state."""
        configs = self.manager.get_button_configs(AudioStatus.STOPPED)
        
        # Start button should be interactive but secondary
        self.assertTrue(configs["start_btn"].interactive)
        self.assertEqual(configs["start_btn"].variant, "secondary")
        
        # Stop button should not be interactive
        self.assertFalse(configs["stop_btn"].interactive)
        self.assertEqual(configs["stop_btn"].variant, "secondary")
        
        # Save button should be interactive and primary (highlighted)
        self.assertTrue(configs["save_btn"].interactive)
        self.assertEqual(configs["save_btn"].variant, "primary")
    
    def test_stopping_state_configuration(self):
        """Test button configuration for STOPPING state."""
        configs = self.manager.get_button_configs(AudioStatus.STOPPING)
        
        # All buttons should be non-interactive during stopping
        self.assertFalse(configs["start_btn"].interactive)
        self.assertFalse(configs["stop_btn"].interactive)
        self.assertFalse(configs["save_btn"].interactive)
        
        # Stop button should show "stopping" text
        self.assertEqual(configs["stop_btn"].text, "⏳ Stopping...")
    
    def test_error_state_configuration(self):
        """Test button configuration for ERROR state."""
        configs = self.manager.get_button_configs(AudioStatus.ERROR)
        
        # Start button should be interactive to allow restart
        self.assertTrue(configs["start_btn"].interactive)
        self.assertEqual(configs["start_btn"].variant, "secondary")
        
        # Stop and save buttons should not be interactive
        self.assertFalse(configs["stop_btn"].interactive)
        self.assertFalse(configs["save_btn"].interactive)
    
    def test_gradio_updates_generation(self):
        """Test generation of Gradio update objects."""
        updates = self.manager.get_gradio_updates(AudioStatus.IDLE)
        
        # Should have updates for all three buttons
        self.assertIn("start_btn", updates)
        self.assertIn("stop_btn", updates)
        self.assertIn("save_btn", updates)
        
        # Each update should have the required fields
        for button_name, update in updates.items():
            self.assertIsInstance(update, dict)
            self.assertIn("value", update)
            self.assertIn("variant", update)
            self.assertIn("interactive", update)
            self.assertIn("visible", update)
    
    def test_button_update_tuple(self):
        """Test generation of button update tuple."""
        tuple_result = self.manager.get_button_update_tuple(AudioStatus.RECORDING)
        
        # Should return tuple of exactly 3 elements
        self.assertEqual(len(tuple_result), 3)
        
        # Each element should be a Gradio update dictionary
        for update in tuple_result:
            self.assertIsInstance(update, dict)
    
    def test_is_button_interactive(self):
        """Test individual button interactivity checking."""
        # In IDLE state, start should be interactive, others not
        self.assertTrue(self.manager.is_button_interactive(AudioStatus.IDLE, "start_btn"))
        self.assertFalse(self.manager.is_button_interactive(AudioStatus.IDLE, "stop_btn"))
        self.assertFalse(self.manager.is_button_interactive(AudioStatus.IDLE, "save_btn"))
        
        # In RECORDING state, stop should be interactive, others not
        self.assertFalse(self.manager.is_button_interactive(AudioStatus.RECORDING, "start_btn"))
        self.assertTrue(self.manager.is_button_interactive(AudioStatus.RECORDING, "stop_btn"))
        self.assertFalse(self.manager.is_button_interactive(AudioStatus.RECORDING, "save_btn"))
        
        # In STOPPED state, start and save should be interactive
        self.assertTrue(self.manager.is_button_interactive(AudioStatus.STOPPED, "start_btn"))
        self.assertFalse(self.manager.is_button_interactive(AudioStatus.STOPPED, "stop_btn"))
        self.assertTrue(self.manager.is_button_interactive(AudioStatus.STOPPED, "save_btn"))
    
    def test_unknown_status_fallback(self):
        """Test handling of unknown status values."""
        # Create a mock status that doesn't exist in the mapping
        class UnknownStatus:
            pass
        
        unknown_status = UnknownStatus()
        
        # Should fall back to IDLE state configuration
        configs = self.manager.get_button_configs(unknown_status)
        expected_configs = self.manager.get_button_configs(AudioStatus.IDLE)
        
        self.assertEqual(configs["start_btn"].text, expected_configs["start_btn"].text)
        self.assertEqual(configs["start_btn"].interactive, expected_configs["start_btn"].interactive)
    
    def test_safe_fallback_updates(self):
        """Test safe fallback updates for error scenarios."""
        safe_updates = self.manager.get_safe_fallback_updates()
        
        # Should provide safe defaults for all buttons
        self.assertIn("start_btn", safe_updates)
        self.assertIn("stop_btn", safe_updates)
        self.assertIn("save_btn", safe_updates)
        
        # Start button should be interactive in fallback
        self.assertTrue(safe_updates["start_btn"]["interactive"])
        self.assertEqual(safe_updates["start_btn"]["variant"], "primary")
        
        # Other buttons should not be interactive
        self.assertFalse(safe_updates["stop_btn"]["interactive"])
        self.assertFalse(safe_updates["save_btn"]["interactive"])
    
    def test_all_status_states_covered(self):
        """Test that all AudioStatus states are covered in the configuration."""
        # Get all AudioStatus enum values
        all_statuses = list(AudioStatus)
        
        # Each status should have a configuration
        for status in all_statuses:
            configs = self.manager.get_button_configs(status)
            
            # Should have all required button configs
            self.assertIn("start_btn", configs)
            self.assertIn("stop_btn", configs)
            self.assertIn("save_btn", configs)
            
            # Each config should be a ButtonConfig instance
            for button_name, config in configs.items():
                self.assertIsInstance(config, ButtonConfig)
                self.assertIsInstance(config.text, str)
                self.assertIsInstance(config.variant, str)
                self.assertIsInstance(config.interactive, bool)
                self.assertIsInstance(config.visible, bool)


if __name__ == '__main__':
    unittest.main()