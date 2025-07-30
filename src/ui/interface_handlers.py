"""Event handlers for the UI interface."""

import logging
import tempfile
import os
from typing import List, Tuple, Optional
from datetime import datetime

import gradio as gr

from src.utils.device_utils import get_audio_devices, get_default_device_index
from src.managers.session_manager import get_audio_session
from src.utils.status_manager import status_manager
from .interface_utils import load_meetings_data, save_meeting_to_database
from .interface_constants import DEFAULT_VALUES, AUDIO_CONFIG
from .button_state_manager import button_state_manager
from .recording_handlers import start_recording, stop_recording
from .meeting_handlers import (
    submit_new_meeting, delete_meeting_by_id_input, handle_meeting_row_selection,
    reset_meeting_duration, delete_meeting_with_confirmation, create_success_message,
    create_error_message, extract_transcription_from_dialog, parse_duration_to_minutes
)
from src.managers.meeting_repository import get_all_meetings, delete_meeting_by_id

logger = logging.getLogger(__name__)


def get_device_choices_and_default():
    """Get current audio device choices and default selection."""
    try:
        devices = get_audio_devices(refresh=True)
        if not devices:
            return [(DEFAULT_VALUES["no_devices"], -1)], -1
        
        device_index = get_default_device_index()
        default_device_index = None
        
        # Find default device index in the list
        for display_name, index in devices:
            if index == device_index:
                default_device_index = index
                break
        
        # If default not found, use first device index
        if default_device_index is None:
            default_device_index = devices[0][1]  # Use index, not name
        
        return devices, default_device_index
    except Exception as e:
        error_choice = [(f"Error: {str(e)}", -1)]
        return error_choice, -1


def update_button_states():
    """Update all button states based on current recording status."""
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


def update_download_button_visibility():
    """Update download button visibility based on transcript availability."""
    try:
        # Get audio session manager
        audio_session = get_audio_session()
        
        # Check if there are any transcriptions available
        current_transcriptions = audio_session.get_current_transcriptions()
        has_transcript = len(current_transcriptions) > 0
        
        return gr.update(visible=has_transcript)
    except Exception as e:
        logger.error(f"Error updating download button visibility: {e}")
        return gr.update(visible=False)


def clear_dialog():
    """Clear dialog messages and session transcriptions."""
    try:
        logger.info("🗑️ Clear dialog button clicked")
        
        # Clear session manager transcriptions
        audio_session = get_audio_session()
        audio_session.clear_transcriptions()
        
        logger.info("✅ Dialog cleared - transcriptions and UI messages removed")
        
        # Return empty states for both dialog components
        return [], []
    except Exception as e:
        logger.error(f"❌ Error clearing dialog: {e}")
        # Return empty states even on error to ensure UI is cleared
        return [], []


# Device Management Handlers
def refresh_devices():
    """Refresh audio device list."""
    try:
        devices, current_device_index = get_device_choices_and_default()
        logger.info(f"🔄 Refreshed devices: {devices}, default index: {current_device_index}")
        status_manager.set_status(
            status_manager.current_status,
            "Devices refreshed"
        )
        return (
            gr.Dropdown(choices=devices, value=current_device_index),
            status_manager.get_status_message()
        )
    except Exception as e:
        logger.error(f"❌ Failed to refresh devices: {e}")
        status_manager.set_error(e, "Failed to refresh devices")
        error_choice = [(f"Error: {str(e)}", -1)]
        return (
            gr.Dropdown(choices=error_choice, value=-1),
            status_manager.get_status_message()
        )


# Recording Control Handlers
# Recording functions moved to recording_handlers.py for better modularity


# Transcription Handlers
def handle_transcription_update(current_state, message):
    """Handle new transcription message and update dialog state."""
    try:
        logger.debug(f"UI: Handling transcription update: {message}")
        from .interface_dialog_handlers import update_dialog_state
        updated_state, gradio_messages = update_dialog_state(current_state, message)
        return updated_state, gradio_messages
    except Exception as e:
        logger.error(f"Error handling transcription update: {e}")
        return current_state, []


def get_latest_dialog_state():
    """Get the latest dialog state from session manager."""
    try:
        # Get audio session manager
        audio_session = get_audio_session()
        
        # Get current transcriptions from session manager
        current_transcriptions = audio_session.get_current_transcriptions()
        
        # Convert to Gradio format
        gradio_messages = []
        for msg in current_transcriptions:
            gradio_messages.append({
                "role": "assistant",
                "content": msg["content"]
            })
        
        return current_transcriptions, gradio_messages
    except Exception as e:
        logger.error(f"Error getting dialog state: {e}")
        return [], []


def conditional_update():
    """Only update if recording is active or there are messages."""
    try:
        # Get audio session manager
        audio_session = get_audio_session()
        
        # Get current transcriptions
        current_transcriptions = audio_session.get_current_transcriptions()
        
        # If no transcriptions and not recording, return None (no update)
        if not current_transcriptions and not audio_session.is_recording():
            return gr.skip(), gr.skip()
        
        # Convert to Gradio format
        gradio_messages = []
        for msg in current_transcriptions:
            gradio_messages.append({
                "role": "assistant",
                "content": msg["content"]
            })
        
        return current_transcriptions, gradio_messages
    except Exception as e:
        logger.error(f"Error in conditional update: {e}")
        return gr.skip(), gr.skip()


# Meeting management functions moved to meeting_handlers.py for better modularity


# Helper functions moved to meeting_handlers.py for better modularity

def create_warning_message(text):
    """Create yellow warning message."""
    return gr.HTML(f'''
        <div style="color: #856404; padding: 12px; border: 1px solid #ffeaa7; border-radius: 6px; background-color: #fff3cd; margin: 10px 0;">
            <strong>⚠️ Warning:</strong> {text}  
        </div>
    ''')


# Utility handler functions
def immediate_transcription_update(message):
    """Immediately handle transcription update."""
    logger.debug(f"UI: Immediate transcription update: {message}")
    # This will be handled by the session manager directly
    pass


def setup_save_callback():
    """Setup JavaScript callback for save action."""
    return gr.HTML("""
        <script>
            window.gradioSaveMeeting = function(meetingName, transcription, duration) {
                // Find the save button in Gradio and trigger it
                const saveBtn = document.querySelector('#save-meeting-trigger');
                if (saveBtn) {
                    // Update hidden inputs with form data
                    const nameInput = document.querySelector('#meeting-name-input textarea');
                    const transcInput = document.querySelector('#transcription-preview textarea');
                    const durationInput = document.querySelector('#duration-display textarea');
                    
                    if (nameInput) nameInput.value = meetingName;
                    if (transcInput) transcInput.value = transcription;
                    if (durationInput) durationInput.value = duration;
                    
                    saveBtn.click();
                }
            };
        </script>
    """)


def download_transcript():
    """Generate and return transcript file for download."""
    try:
        logger.info("🔽 Download transcript button clicked")
        
        # Get audio session manager
        audio_session = get_audio_session()
        
        # Get current transcriptions from session manager
        current_transcriptions = audio_session.get_current_transcriptions()
        
        if not current_transcriptions:
            logger.info("📄 No transcript available for download")
            # Create a file with a message indicating no transcript
            transcript_content = "No transcript available.\n\nPlease start recording to generate a transcript."
        else:
            logger.info(f"📄 Generating transcript file with {len(current_transcriptions)} transcriptions")
            
            # Get session info for duration and timing
            session_info = audio_session.get_session_info()
            duration = session_info.get('duration', 0.0)
            start_time = session_info.get('start_time')
            
            # Format transcript content
            transcript_lines = []
            
            # Add header
            transcript_lines.append("Voice Meeting Transcript")
            transcript_lines.append("=" * 50)
            transcript_lines.append("")
            
            if start_time:
                transcript_lines.append(f"Session Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            transcript_lines.append(f"Duration: {duration:.1f} minutes")
            transcript_lines.append("")
            transcript_lines.append("Transcript:")
            transcript_lines.append("-" * 20)
            transcript_lines.append("")
            
            # Add transcript content
            for i, msg in enumerate(current_transcriptions, 1):
                content = msg.get("content", "")
                timestamp = msg.get("timestamp", "")
                
                if timestamp:
                    transcript_lines.append(f"[{timestamp}] {content}")
                else:
                    transcript_lines.append(f"[{i}] {content}")
                transcript_lines.append("")
            
            transcript_content = "\n".join(transcript_lines)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"transcript_{timestamp}.txt"
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.txt',
            prefix='transcript_',
            delete=False,
            encoding='utf-8'
        )
        
        try:
            temp_file.write(transcript_content)
            temp_file.flush()
            temp_file_path = temp_file.name
        finally:
            temp_file.close()
        
        logger.info(f"📄 Transcript file created: {temp_file_path}")
        logger.info(f"📄 Content length: {len(transcript_content)} characters")
        
        return temp_file_path
        
    except Exception as e:
        logger.error(f"❌ Error generating transcript download: {e}")
        import traceback
        traceback.print_exc()
        
        # Create an error file
        error_content = f"Error generating transcript: {str(e)}\n\nPlease try again or contact support."
        
        error_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.txt',
            prefix='transcript_error_',
            delete=False,
            encoding='utf-8'
        )
        
        try:
            error_file.write(error_content)
            error_file.flush()
            error_file_path = error_file.name
        finally:
            error_file.close()
        
        return error_file_path


def create_download_button(file_path):
    """Create a DownloadButton with the given file path to trigger download."""
    from .interface_constants import BUTTON_TEXT
    
    if file_path:
        return gr.DownloadButton(
            label=BUTTON_TEXT["download_transcript"],
            value=file_path,
            variant="secondary",
            visible=True
        )
    else:
        return gr.DownloadButton(
            label=BUTTON_TEXT["download_transcript"],
            variant="secondary",
            visible=False
        )


def handle_copy_event(copy_data):
    """Handle copy events from the chatbot."""
    try:
        logger.info(f"📋 Copy event triggered")
        logger.info(f"📋 Copied content length: {len(copy_data.value) if copy_data.value else 0} characters")
        
        # Log copy usage for analytics (optional)
        if copy_data.value:
            content_preview = copy_data.value[:100] + "..." if len(copy_data.value) > 100 else copy_data.value
            logger.info(f"📋 Copy preview: {content_preview}")
        
        # Return the copied value (required by Gradio copy event)
        return copy_data.value
        
    except Exception as e:
        logger.error(f"❌ Error handling copy event: {e}")
        # Return original value even on error
        return copy_data.value if hasattr(copy_data, 'value') else ""


def get_current_duration_display():
    """Get formatted current duration for display."""
    try:
        audio_session = get_audio_session()
        formatted_duration = audio_session.get_formatted_duration()
        
        logger.debug(f"⏱️ Duration display: {formatted_duration}")
        return formatted_duration
        
    except Exception as e:
        logger.error(f"❌ Error getting duration display: {e}")
        from .interface_constants import DURATION_FORMAT
        return DURATION_FORMAT["default_display"]


# Meeting duration reset function moved to meeting_handlers.py


def get_duration_analytics():
    """Get duration analytics for current session."""
    try:
        audio_session = get_audio_session()
        
        analytics = {
            'total_duration_seconds': audio_session.get_current_duration_seconds(),
            'total_duration_formatted': audio_session.get_formatted_duration(),
            'recording_segments': audio_session.get_recording_segments(),
            'segment_count': len(audio_session.get_recording_segments()),
            'is_recording': audio_session.is_recording()
        }
        
        logger.info(f"📊 Duration analytics: {analytics['total_duration_formatted']} across {analytics['segment_count']} segments")
        
        return analytics
        
    except Exception as e:
        logger.error(f"❌ Error getting duration analytics: {e}")
        return {
            'total_duration_seconds': 0.0,
            'total_duration_formatted': "00:00",
            'recording_segments': [],
            'segment_count': 0,
            'is_recording': False
        }


# Meeting List Management handlers moved to meeting_handlers.py for better modularity