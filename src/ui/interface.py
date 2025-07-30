"""Main interface creation for the Voice Meeting App."""

import logging
from typing import Optional, List, Tuple
from datetime import datetime

import gradio as gr

from src.utils.device_utils import get_audio_devices, get_default_device_index
from src.managers.session_manager import get_audio_session
from src.utils.status_manager import status_manager, AudioStatus
from src.managers.meeting_repository import get_all_meetings, create_meeting, MeetingRepositoryError
from src.core.models import Meeting
from .interface_utils import load_meetings_data, refresh_meetings_list, save_meeting_to_database
from .interface_constants import (
    AVAILABLE_THEMES, DEFAULT_THEME, BUTTON_TEXT, UI_TEXT, PLACEHOLDER_TEXT,
    UI_DIMENSIONS, TABLE_HEADERS, FORM_LABELS, DEFAULT_VALUES, AUDIO_CONFIG,
    COPY_CONFIG, DURATION_FORMAT
)
from .interface_styles import APP_CSS, APP_JS
from .button_state_manager import button_state_manager
from .interface_handlers import (
    refresh_devices, start_recording, stop_recording, handle_transcription_update,
    get_latest_dialog_state, conditional_update, submit_new_meeting,
    immediate_transcription_update, get_device_choices_and_default,
    download_transcript, update_download_button_visibility, create_download_button, clear_dialog,
    handle_copy_event, get_current_duration_display, reset_meeting_duration,
    delete_meeting_by_id_input
)

logger = logging.getLogger(__name__)


# Utility functions moved to interface_utils.py


def update_button_states():
    """Update all button states based on current recording status.
    
    Returns:
        Tuple of button updates for (start_btn, stop_btn, save_btn)
    """
    try:
        current_status = status_manager.current_status
        return button_state_manager.get_button_update_tuple(current_status)
    except Exception as e:
        logger.error(f"Error updating button states: {e}")
        # Return safe defaults using the manager
        safe_updates = button_state_manager.get_safe_fallback_updates()
        return (
            safe_updates["start_btn"],
            safe_updates["stop_btn"],
            safe_updates["save_btn"]
        )


# Use themes from constants
THEMES = AVAILABLE_THEMES

def create_header():
    """Create the header section of the interface."""
    with gr.Row():
        gr.Markdown(
            f"""
            # {UI_TEXT["app_title"]}
            {UI_TEXT["app_subtitle"]}
            """,
            elem_classes=["header-text"],
        )


def create_meeting_list():
    """Create the meeting list panel with delete functionality."""
    with gr.Column(elem_classes=["meeting-list-container"]):
        gr.Markdown(UI_TEXT["meeting_list_title"])
        
        # Meeting list dataframe in its own container
        with gr.Column(elem_classes=["meeting-panel"]):
            meeting_list = gr.Dataframe(
                headers=TABLE_HEADERS["meeting_list"],
                datatype=["number", "str", "str", "str", "str"],  # number for ID
                value=load_meetings_data(),
                interactive=False,           # Make completely readonly 
                show_search="search",        # Enable search functionality
                show_fullscreen_button=True, # Allow fullscreen viewing  
                show_copy_button=True,       # Enable copying data
                show_row_numbers=True,       # Show row numbers for additional clarity
                wrap=True                    # Enable text wrapping if needed
            )
        
        # Delete section - positioned below the table, outside the fixed-height panel
        with gr.Column(elem_classes=["delete-controls-section"]):
            with gr.Row():
                meeting_id_input = gr.Textbox(
                    label="Meeting ID to Delete",
                    placeholder="Enter meeting ID (e.g., 1, 2, 3)",
                    scale=3
                )
                delete_meeting_btn = gr.Button(
                    "🗑️ Delete Meeting", 
                    variant="stop", 
                    scale=1,
                    size="sm"
                )
            
            # Status message for delete operations
            delete_status = gr.HTML(
                value="💡 Enter a meeting ID from the table above and click Delete",
                visible=True
            )
        
        return meeting_list, meeting_id_input, delete_meeting_btn, delete_status


def create_dialog_panel():
    """Create the dialog panel with meeting fields and chatbot."""
    with gr.Column(scale=4, elem_classes=["dialog-panel"]):
        gr.Markdown(UI_TEXT["live_dialog_title"])
        
        # Meeting fields
        with gr.Row():
            meeting_name_field = gr.Textbox(
                label=FORM_LABELS["meeting_name"],
                placeholder=PLACEHOLDER_TEXT["meeting_name"],
                value=""
            )
            duration_field = gr.Textbox(
                label=FORM_LABELS["duration"],
                value=DEFAULT_VALUES["duration_display"],
                interactive=False
            )
        
        dialog_output = gr.Chatbot(
            value=[],  # Start with empty dialog
            type="messages",
            show_label=False,
            placeholder=PLACEHOLDER_TEXT["transcription_dialog"],
            height=UI_DIMENSIONS["dialog_height"],  # Set chatbot height
            show_copy_button=True,  # Enable individual message copy buttons
            show_copy_all_button=True,  # Enable copy all messages button
            watermark=COPY_CONFIG["watermark"]  # Add watermark to copied content
        )
        
        return meeting_name_field, duration_field, dialog_output


# get_device_choices_and_default moved to interface_handlers.py


def create_controls():
    """Create the audio controls panel."""
    
    with gr.Column(scale=2, elem_classes=["control-panel"]):
        gr.Markdown(UI_TEXT["audio_controls_title"])
        
        # Audio device selection
        device_choices, initial_device_index = get_device_choices_and_default()
        
        device_dropdown = gr.Dropdown(
            label=FORM_LABELS["audio_device"],
            choices=device_choices,
            value=initial_device_index,
            interactive=True,
            allow_custom_value=False  # Disable custom values to prevent invalid indices
        )
        
        # Device refresh button
        refresh_btn = gr.Button(
            BUTTON_TEXT["refresh_devices"], 
            size="sm",
            variant="secondary"
        )
        
        # Recording status
        status_text = gr.Textbox(
            label=FORM_LABELS["status"],
            value=status_manager.get_status_message(),
            interactive=False
        )
        
        # Control buttons - Initialize with proper states using ButtonStateManager
        current_status = status_manager.current_status
        logger.info(f"🔍 Current status: {current_status}")
        button_configs = button_state_manager.get_button_configs(current_status)
        logger.info(f"🔍 Start button interactive: {button_configs['start_btn'].interactive}")
        
        with gr.Row():
            start_btn = gr.Button(
                button_configs["start_btn"].text,
                variant=button_configs["start_btn"].variant,
                interactive=button_configs["start_btn"].interactive
            )
            stop_btn = gr.Button(
                button_configs["stop_btn"].text,
                variant=button_configs["stop_btn"].variant,
                interactive=button_configs["stop_btn"].interactive
            )
        
        # Save meeting button
        save_meeting_btn = gr.Button(
            button_configs["save_btn"].text,
            variant=button_configs["save_btn"].variant,
            interactive=button_configs["save_btn"].interactive
        )
        
        # Download transcript button
        download_transcript_btn = gr.DownloadButton(
            label=BUTTON_TEXT["download_transcript"],
            variant="secondary",
            visible=False  # Initially hidden until transcript is available
        )
        
        return device_dropdown, refresh_btn, status_text, start_btn, stop_btn, save_meeting_btn, download_transcript_btn


def create_interface(theme_name: str = DEFAULT_THEME) -> gr.Blocks:
    """Create the main Gradio interface.
    
    Args:
        theme_name: Name of the theme to use
        
    Returns:
        Gradio Blocks interface
    """
    # Get theme
    theme = THEMES.get(theme_name, THEMES[DEFAULT_THEME])
    
    # Initialize audio devices - function moved to create_controls()
    # Use styles from separate file
    css = APP_CSS
    js_func = APP_JS

    with gr.Blocks(
        title="Voice Meeting App", 
        theme=theme, 
        css=css, 
        js=js_func,
    ) as demo:
        
        # Header
        create_header()

        # Responsive layout structure
        # Desktop: [Meeting List] [Live Dialog] [Audio Controls]
        # Mobile: [Meeting List - Full Width] then [Live Dialog] [Audio Controls]
        
        # Meeting List - Full width on mobile, partial on desktop
        meeting_list, meeting_id_input, delete_meeting_btn, delete_status = create_meeting_list()
        
        # Dialog and Controls - Side by side on all screens, but different proportions
        with gr.Row(elem_classes=["main-content-row"]):
            # Center panel - Live Dialog
            meeting_name_field, duration_field, dialog_output = create_dialog_panel()
            
            # Right panel - Audio Controls
            device_dropdown, refresh_btn, status_text, start_btn, stop_btn, save_meeting_btn, download_transcript_btn = create_controls()
        
        # Status message component for user feedback (placed near save button)
        with gr.Row():
            save_status_message = gr.HTML(visible=False)
        
        # Get audio session manager
        audio_session = get_audio_session()
        
        # State management for real-time updates
        dialog_state = gr.State([])
        
        def update_dialog_state(current_messages, new_message):
            """Update dialog state with new transcription message."""
            try:
                logger.debug(f"UI: Updating dialog state with message: {new_message}")
                
                # Create a copy of current messages
                updated_messages = current_messages.copy()
                
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
                gradio_messages = []
                for msg in updated_messages:
                    gradio_messages.append({
                        "role": "assistant",
                        "content": msg["content"]
                    })
                
                return updated_messages, gradio_messages
                
            except Exception as e:
                logger.error(f"Error updating dialog state: {e}")
                return current_messages, []
        
        # Event handlers moved to interface_handlers.py
        
        # Register callback with session manager
        audio_session.add_transcription_callback(immediate_transcription_update)
        
        # Timer for dialog updates only (not button updates)
        timer = gr.Timer(value=UI_DIMENSIONS["timer_interval"])  # Check for updates every 500ms
        
        # Wire up event handlers
        refresh_btn.click(
            fn=refresh_devices,
            outputs=[device_dropdown, status_text]
        )
        
        start_btn.click(
            fn=start_recording,
            inputs=[device_dropdown, dialog_state],
            outputs=[status_text, dialog_state, dialog_output, start_btn, stop_btn, save_meeting_btn]
        )
        
        stop_btn.click(
            fn=stop_recording,
            outputs=[status_text, start_btn, stop_btn, save_meeting_btn]
        )
        
        # Combined update function for dialog, download button, and duration
        def combined_update():
            """Update dialog, download button visibility, and duration display."""
            dialog_state_result, dialog_output_result = conditional_update()
            download_button_result = update_download_button_visibility()
            duration_display_result = get_current_duration_display()
            return dialog_state_result, dialog_output_result, download_button_result, duration_display_result
        
        # Timer for dialog, download button, and duration updates  
        timer.tick(
            fn=combined_update,
            outputs=[dialog_state, dialog_output, download_transcript_btn, duration_field]
        )
        
        # Download transcript button - following the working Gradio pattern
        def handle_download_click():
            """Handle download button click - generate file and return DownloadButton with value."""
            file_path = download_transcript()
            return create_download_button(file_path)
        
        download_transcript_btn.click(
            fn=handle_download_click,
            outputs=[download_transcript_btn]
        )
        
        # Clear dialog functionality - wire up chatbot's built-in clear event
        dialog_output.clear(
            fn=clear_dialog,
            outputs=[dialog_state, dialog_output],
            queue=False  # Immediate response for clearing
        )
        
        # Copy event handler - wire up chatbot's copy event for analytics
        dialog_output.copy(
            fn=handle_copy_event,
            queue=False  # Immediate response for copying
        )
        
        # Direct save functionality (replaces old sliding panel system)
        save_meeting_btn.click(
            fn=submit_new_meeting,
            inputs=[meeting_name_field, duration_field, dialog_output],
            outputs=[save_status_message, meeting_list]
        ).then(
            # Show the status message after submission
            fn=lambda: gr.update(visible=True),
            outputs=[save_status_message]
        )
        
        # Simple ID-based delete functionality
        delete_meeting_btn.click(
            fn=delete_meeting_by_id_input,
            inputs=[meeting_id_input],
            outputs=[meeting_list, delete_status]
        ).then(
            # Clear the input field after successful operation
            fn=lambda: "",
            outputs=[meeting_id_input]
        )
        
        # Note: Removed automatic button updates to prevent interference with clicks
        # Buttons are updated manually in the event handlers when needed
    
    return demo