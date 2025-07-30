"""Dialog and UI update handlers for the interface."""

import logging
from typing import List, Dict, Tuple, Any

import gradio as gr

from .interface_handlers import conditional_update, update_download_button_visibility, get_current_duration_display, download_transcript, create_download_button

logger = logging.getLogger(__name__)


class DialogStateManager:
    """Manages dialog state updates and message handling."""
    
    def __init__(self):
        """Initialize the dialog state manager."""
        pass
    
    def update_dialog_state(self, current_messages: List[Dict], new_message: Dict) -> Tuple[List[Dict], List[Dict]]:
        """Update dialog state with new transcription message.
        
        Args:
            current_messages: List of current dialog messages
            new_message: New message to add or update
            
        Returns:
            Tuple of (updated_messages, gradio_formatted_messages)
        """
        try:
            logger.debug(f"UI: Updating dialog state with message: {new_message}")
            
            # Create a copy of current messages
            updated_messages = current_messages.copy() if current_messages else []
            
            # Handle partial result updates
            if new_message.get("utterance_id") and new_message.get("is_partial"):
                # Find existing message with same utterance_id
                existing_index = None
                for i, msg in enumerate(updated_messages):
                    if msg.get("utterance_id") == new_message["utterance_id"]:
                        existing_index = i
                        break
                
                if existing_index is not None:
                    # Update existing message
                    updated_messages[existing_index] = new_message
                    logger.debug(f"UI: Updated partial message at index {existing_index}")
                else:
                    # Add new partial message
                    updated_messages.append(new_message)
                    logger.debug(f"UI: Added new partial message")
            else:
                # Final result or no utterance tracking
                if new_message.get("utterance_id"):
                    # Replace partial result with final result
                    existing_index = None
                    for i, msg in enumerate(updated_messages):
                        if msg.get("utterance_id") == new_message["utterance_id"]:
                            existing_index = i
                            break
                    
                    if existing_index is not None:
                        updated_messages[existing_index] = new_message
                        logger.debug(f"UI: Finalized message at index {existing_index}")
                    else:
                        updated_messages.append(new_message)
                        logger.debug(f"UI: Added new final message")
                else:
                    # No utterance tracking, just append
                    updated_messages.append(new_message)
                    logger.debug(f"UI: Added message without utterance tracking")
                    
            logger.debug(f"UI: Dialog now has {len(updated_messages)} messages")
            
            # Convert to Gradio format
            gradio_messages = self._convert_to_gradio_format(updated_messages)
            
            return updated_messages, gradio_messages
            
        except Exception as e:
            logger.error(f"Error updating dialog state: {e}")
            return current_messages if current_messages else [], []
    
    def _convert_to_gradio_format(self, messages: List[Dict]) -> List[Dict]:
        """Convert internal message format to Gradio chatbot format.
        
        Args:
            messages: List of internal message dictionaries
            
        Returns:
            List of Gradio-formatted messages
        """
        gradio_messages = []
        for msg in messages:
            if isinstance(msg, dict) and "content" in msg:
                gradio_messages.append({
                    "role": "assistant",
                    "content": msg["content"]
                })
        return gradio_messages


class UIUpdateManager:
    """Manages combined UI updates and event handling."""
    
    def __init__(self):
        """Initialize the UI update manager."""
        pass
    
    def combined_update(self) -> Tuple[Any, Any, Any, str]:
        """Update dialog, download button visibility, and duration display.
        
        Returns:
            Tuple of (dialog_state_result, dialog_output_result, download_button_result, duration_display_result)
        """
        try:
            dialog_state_result, dialog_output_result = conditional_update()
            download_button_result = update_download_button_visibility()
            duration_display_result = get_current_duration_display()
            return dialog_state_result, dialog_output_result, download_button_result, duration_display_result
        except Exception as e:
            logger.error(f"Error in combined update: {e}")
            # Return safe defaults
            return gr.skip(), gr.skip(), gr.skip(), "00:00"
    
    def handle_download_click(self) -> gr.DownloadButton:
        """Handle download button click - generate file and return DownloadButton with value.
        
        Returns:
            Updated DownloadButton component with file path
        """
        try:
            file_path = download_transcript()
            return create_download_button(file_path)
        except Exception as e:
            logger.error(f"Error handling download click: {e}")
            # Return default download button state
            return gr.DownloadButton(
                label="Download Transcript",
                variant="secondary",
                visible=False
            )


# Global instances for consistent state management
dialog_state_manager = DialogStateManager()
ui_update_manager = UIUpdateManager()


# Module-level functions for backward compatibility
def update_dialog_state(current_messages: List[Dict], new_message: Dict) -> Tuple[List[Dict], List[Dict]]:
    """Update dialog state with new transcription message."""
    return dialog_state_manager.update_dialog_state(current_messages, new_message)


def combined_update() -> Tuple[Any, Any, Any, str]:
    """Update dialog, download button visibility, and duration display."""
    return ui_update_manager.combined_update()


def handle_download_click() -> gr.DownloadButton:
    """Handle download button click - generate file and return DownloadButton with value."""
    return ui_update_manager.handle_download_click()