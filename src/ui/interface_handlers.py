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
def start_recording(device_selection, current_state):
    """Start recording with selected device."""
    try:
        logger.info(f"🎤 START RECORDING CLICKED")
        logger.info(f"🎤 Device selection: {device_selection} (type: {type(device_selection)})")
        logger.info(f"🎤 Current state: {len(current_state) if current_state else 0} messages")
        
        # Get audio session manager
        audio_session = get_audio_session()
        
        # Preserve existing dialog state instead of clearing
        preserved_state = current_state if current_state is not None else []
        logger.info(f"🎤 Preserving {len(preserved_state)} existing messages")
        
        # Log current available devices for debugging
        current_devices, current_default = get_device_choices_and_default()
        logger.info(f"🎤 Current available devices: {current_devices}")
        logger.info(f"🎤 Current default device index: {current_default}")
        
        # Convert preserved state to Gradio format for visual display
        gradio_messages = []
        for msg in preserved_state:
            gradio_messages.append({
                "role": "assistant",
                "content": msg["content"]
            })
        logger.info(f"🎤 Converted {len(gradio_messages)} messages to Gradio format")
        
        # Handle device selection - should be device index directly
        device_index = None
        if isinstance(device_selection, int):
            device_index = device_selection
            logger.info(f"🎤 Using device index directly: {device_index}")
        elif isinstance(device_selection, str):
            # Fallback: try to parse as integer first
            try:
                device_index = int(device_selection)
                logger.info(f"🎤 Parsed device index from string: {device_index}")
            except ValueError:
                # If not a number, try to find by name (legacy support)
                logger.warning(f"🎤 Received device name instead of index: '{device_selection}', attempting name lookup")
                devices, _ = get_device_choices_and_default()
                for name, index in devices:
                    if name == device_selection:
                        device_index = index
                        logger.info(f"🎤 Found device index by name: {name} -> {device_index}")
                        break
        
        if device_index is None or device_index == -1:
            error_msg = f"Invalid device selection: {device_selection}"
            logger.error(f"❌ {error_msg}")
            status_manager.set_error(
                Exception("Invalid device"),
                error_msg
            )
            start_btn_state, stop_btn_state, save_btn_state = update_button_states()
            return status_manager.get_status_message(), preserved_state, gradio_messages, start_btn_state, stop_btn_state, save_btn_state
        
        # Validate that the device index exists in the current device list
        devices, _ = get_device_choices_and_default()
        valid_indices = [index for name, index in devices]
        if device_index not in valid_indices:
            error_msg = f"Device index {device_index} is not available. Available devices: {valid_indices}"
            logger.error(f"❌ {error_msg}")
            status_manager.set_error(
                Exception("Device not available"),
                f"Selected device is no longer available"
            )
            start_btn_state, stop_btn_state, save_btn_state = update_button_states()
            return status_manager.get_status_message(), preserved_state, gradio_messages, start_btn_state, stop_btn_state, save_btn_state
        
        # Check if already recording
        if audio_session.is_recording():
            status_manager.set_error(
                Exception("Already recording"),
                "Recording already in progress"
            )
            start_btn_state, stop_btn_state, save_btn_state = update_button_states()
            return status_manager.get_status_message(), preserved_state, gradio_messages, start_btn_state, stop_btn_state, save_btn_state
        
        # Start recording using session manager
        status_manager.set_initializing()
        start_btn_state, stop_btn_state, save_btn_state = update_button_states()
        
        config = AUDIO_CONFIG
        
        status_manager.set_connecting()
        
        if audio_session.start_recording(device_index, config):
            status_manager.set_recording()
        else:
            status_manager.set_error(
                Exception("Failed to start"),
                "Could not start recording"
            )
        
        # Update button states based on final status
        start_btn_state, stop_btn_state, save_btn_state = update_button_states()
        return status_manager.get_status_message(), preserved_state, gradio_messages, start_btn_state, stop_btn_state, save_btn_state
        
    except Exception as e:
        logger.error(f"❌ START RECORDING ERROR: {e}")
        import traceback
        traceback.print_exc()
        status_manager.set_error(e, "Failed to start recording")
        start_btn_state, stop_btn_state, save_btn_state = update_button_states()
        return status_manager.get_status_message(), preserved_state, gradio_messages, start_btn_state, stop_btn_state, save_btn_state


def stop_recording():
    """Stop recording."""
    try:
        # Get audio session manager
        audio_session = get_audio_session()
        
        status_manager.set_stopping()
        start_btn_state, stop_btn_state, save_btn_state = update_button_states()
        
        if audio_session.stop_recording():
            status_manager.set_stopped()
        else:
            status_manager.set_error(
                Exception("Failed to stop"),
                "Could not stop recording"
            )
        
        # Update button states based on final status
        start_btn_state, stop_btn_state, save_btn_state = update_button_states()
        return status_manager.get_status_message(), start_btn_state, stop_btn_state, save_btn_state
        
    except Exception as e:
        status_manager.set_error(e, "Failed to stop recording")
        start_btn_state, stop_btn_state, save_btn_state = update_button_states()
        return status_manager.get_status_message(), start_btn_state, stop_btn_state, save_btn_state


# Transcription Handlers
def handle_transcription_update(current_state, message):
    """Handle new transcription message and update dialog state."""
    try:
        logger.debug(f"UI: Handling transcription update: {message}")
        from .interface import update_dialog_state
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


# Direct Save Meeting Handler
def submit_new_meeting(meeting_name, duration_display, dialog_messages):
    """Submit a new meeting directly to database with validation."""
    try:
        logger.info(f"💾 Submitting new meeting: '{meeting_name}', duration: '{duration_display}'")
        
        # Validation - Meeting name cannot be empty
        if not meeting_name or not meeting_name.strip():
            error_msg = "Meeting name cannot be empty"
            logger.warning(f"❌ Validation failed: {error_msg}")
            return create_error_message(error_msg)
        
        # Extract transcription from dialog messages
        transcription_text = extract_transcription_from_dialog(dialog_messages)
        logger.info(f"💾 Extracted transcription length: {len(transcription_text)} characters")
        
        # Parse duration (from "MM:SS" or "HH:MM:SS" format to float minutes)
        duration_minutes = parse_duration_to_minutes(duration_display)
        logger.info(f"💾 Parsed duration: {duration_minutes} minutes")
        
        # Validation - Duration should be > 0 (but allow saving empty recordings with warning)
        if duration_minutes <= 0:
            warning_msg = "No recording duration found, but saving anyway"
            logger.warning(f"⚠️ {warning_msg}")
        
        # Validation - Warn if transcription is empty but allow saving
        if not transcription_text.strip():
            logger.warning("⚠️ Empty transcription, but saving anyway")
        
        # Save to database
        success, message = save_meeting_to_database(
            meeting_name=meeting_name.strip(),
            duration=duration_minutes,
            transcription=transcription_text,
            audio_file_path=None  # As specified - keep empty for now
        )
        
        logger.info(f"💾 Save result: success={success}, message='{message}'")
        
        if success:
            success_msg = f"Meeting '{meeting_name.strip()}' saved successfully! ℹ️"
            logger.info(f"✅ {success_msg}")
            
            # Show Gradio info notification
            gr.Info(success_msg, duration=5)
            
            # Refresh meeting list data
            refreshed_meetings = load_meetings_data()
            logger.info(f"📋 Refreshed meeting list with {len(refreshed_meetings)} meetings")
            
            # Return empty status message and refreshed meeting list
            return gr.HTML(""), refreshed_meetings
        else:
            error_msg = f"Failed to save meeting: {message}"
            logger.error(f"❌ {error_msg}")
            # Keep existing meeting list unchanged on error
            return create_error_message(error_msg), gr.update()
            
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"❌ Error submitting meeting: {e}", exc_info=True)
        # Keep existing meeting list unchanged on error
        return create_error_message(error_msg), gr.update()


# Helper Functions for Message Creation and Data Parsing
def create_success_message(text):
    """Create green success message."""
    return gr.HTML(f'''
        <div style="color: #155724; padding: 12px; border: 1px solid #c3e6cb; border-radius: 6px; background-color: #d4edda; margin: 10px 0;">
            <strong>✅ Success:</strong> {text}
        </div>
    ''')


def create_error_message(text):
    """Create red error message.""" 
    return gr.HTML(f'''
        <div style="color: #721c24; padding: 12px; border: 1px solid #f5c6cb; border-radius: 6px; background-color: #f8d7da; margin: 10px 0;">
            <strong>❌ Error:</strong> {text}
        </div>
    ''')


def create_warning_message(text):
    """Create yellow warning message."""
    return gr.HTML(f'''
        <div style="color: #856404; padding: 12px; border: 1px solid #ffeaa7; border-radius: 6px; background-color: #fff3cd; margin: 10px 0;">
            <strong>⚠️ Warning:</strong> {text}  
        </div>
    ''')


def extract_transcription_from_dialog(dialog_messages):
    """Extract transcription text from dialog messages."""
    if not dialog_messages:
        logger.debug("No dialog messages to extract transcription from")
        return ""
    
    transcription_parts = []
    
    try:
        # Handle Gradio chatbot message format - list of message objects
        for message in dialog_messages:
            if isinstance(message, dict):
                # New message format: {"role": "user/assistant", "content": "text"}
                if 'content' in message and message.get('role') in ['user', 'assistant']:
                    content = message['content']
                    if isinstance(content, str) and content.strip():
                        transcription_parts.append(content.strip())
            elif isinstance(message, (list, tuple)) and len(message) >= 2:
                # Legacy tuple format: [user_message, assistant_message] 
                assistant_msg = message[1] if len(message) > 1 else message[0]
                if isinstance(assistant_msg, str) and assistant_msg.strip():
                    transcription_parts.append(assistant_msg.strip())
    
        result = "\n".join(transcription_parts)
        logger.debug(f"Extracted {len(transcription_parts)} transcription parts, total length: {len(result)}")
        return result
        
    except Exception as e:
        logger.error(f"Error extracting transcription from dialog: {e}")
        return ""


def parse_duration_to_minutes(duration_display):
    """Parse duration from MM:SS or HH:MM:SS to float minutes."""
    if not duration_display:
        return 0.0
        
    try:
        # Remove any extra whitespace
        duration_display = duration_display.strip()
        
        # Split by colon
        parts = duration_display.split(':')
        
        if len(parts) == 2:  # MM:SS format
            minutes = int(parts[0])
            seconds = int(parts[1])
            total_minutes = minutes + seconds / 60.0
        elif len(parts) == 3:  # HH:MM:SS format
            hours = int(parts[0])
            minutes = int(parts[1])  
            seconds = int(parts[2])
            total_minutes = hours * 60 + minutes + seconds / 60.0
        else:
            logger.warning(f"Unexpected duration format: {duration_display}")
            return 0.0
            
        logger.debug(f"Parsed duration '{duration_display}' to {total_minutes:.2f} minutes")
        return total_minutes
        
    except (ValueError, IndexError) as e:
        logger.error(f"Error parsing duration '{duration_display}': {e}")
        return 0.0


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


def reset_meeting_duration():
    """Reset duration tracking for a new meeting."""
    try:
        logger.info("🔄 Reset meeting duration requested")
        
        audio_session = get_audio_session()
        audio_session.reset_duration_tracking()
        
        logger.info("✅ Meeting duration reset successfully")
        
        # Return updated duration display
        return get_current_duration_display()
        
    except Exception as e:
        logger.error(f"❌ Error resetting meeting duration: {e}")
        from .interface_constants import DURATION_FORMAT
        return DURATION_FORMAT["default_display"]


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


# Meeting List Management Handlers
def handle_meeting_row_selection(evt: gr.SelectData):
    """Handle meeting row selection - enable delete button and store meeting ID."""
    try:
        logger.info(f"🎯 Meeting row selected at index: {evt.index}")
        
        row_index = evt.index
        meetings = get_all_meetings()
        
        if 0 <= row_index < len(meetings):
            selected_meeting = meetings[row_index]
            logger.info(f"✅ Selected meeting: {selected_meeting.name} (ID: {selected_meeting.id})")
            
            return (
                gr.update(interactive=True, variant="stop"),  # Enable delete button
                selected_meeting.id,  # Store meeting ID
                f"Selected: {selected_meeting.name}"  # Status message (optional)
            )
        else:
            logger.warning(f"⚠️ Invalid row index: {row_index}, total meetings: {len(meetings)}")
            return gr.update(interactive=False), None, ""
            
    except Exception as e:
        logger.error(f"❌ Error in meeting selection: {e}")
        return gr.update(interactive=False), None, "Selection error"


def delete_meeting_by_id_input(meeting_id_text):
    """Delete a meeting by ID entered in text field."""
    try:
        logger.info(f"🗑️ Delete meeting by ID requested: '{meeting_id_text}'")
        
        # Validate input
        if not meeting_id_text or not meeting_id_text.strip():
            error_msg = "Please enter a meeting ID"
            logger.warning(f"❌ {error_msg}")
            return (
                load_meetings_data(),  # Keep current meeting list
                gr.update(value=error_msg, visible=True)
            )
        
        # Parse meeting ID
        try:
            meeting_id = int(meeting_id_text.strip())
            logger.info(f"🗑️ Parsed meeting ID: {meeting_id}")
        except ValueError:
            error_msg = f"Invalid meeting ID: '{meeting_id_text}'. Please enter a valid number."
            logger.error(f"❌ {error_msg}")
            return (
                load_meetings_data(),  # Keep current meeting list
                gr.update(value=f"❌ {error_msg}", visible=True)
            )
        
        # Attempt to delete the meeting
        try:
            success = delete_meeting_by_id(meeting_id)
            
            if success:
                success_msg = f"Meeting ID {meeting_id} deleted successfully! 🗑️"
                logger.info(f"✅ {success_msg}")
                
                # Show success notification
                gr.Info(success_msg, duration=3)
                
                # Refresh meeting list and clear input
                refreshed_data = load_meetings_data()
                logger.info(f"📋 Refreshed meeting list with {len(refreshed_data)} meetings")
                
                return (
                    refreshed_data,  # Refresh meeting list
                    gr.update(value="✅ Meeting deleted successfully", visible=True)
                )
            else:
                error_msg = f"Meeting ID {meeting_id} not found or could not be deleted"
                logger.error(f"❌ {error_msg}")
                return (
                    load_meetings_data(),  # Keep current meeting list
                    gr.update(value=f"❌ {error_msg}", visible=True)
                )
                
        except Exception as delete_error:
            error_msg = f"Error deleting meeting ID {meeting_id}: {str(delete_error)}"
            logger.error(f"❌ {error_msg}")
            return (
                load_meetings_data(),  # Keep current meeting list
                gr.update(value=f"❌ {error_msg}", visible=True)
            )
            
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"❌ Error in delete_meeting_by_id_input: {e}")
        return (
            load_meetings_data(),  # Keep current meeting list
            gr.update(value=f"❌ {error_msg}", visible=True)
        )


def delete_meeting_with_confirmation(selected_indices: list):
    """Delete selected meetings and refresh the list, clearing all checkboxes."""
    try:
        logger.info(f"🗑️ Delete button clicked for indices: {selected_indices}")
        
        if not selected_indices:
            logger.warning("⚠️ No meetings selected for deletion")
            return gr.update(), gr.update(), gr.update(value="No meetings selected"), []
        
        # Get current meetings to map indices to IDs
        meetings = get_all_meetings()
        deleted_count = 0
        failed_count = 0
        
        # Delete each selected meeting
        for index in selected_indices:
            if 0 <= index < len(meetings):
                meeting = meetings[index]
                try:
                    if delete_meeting_by_id(meeting.id):
                        logger.info(f"✅ Meeting {meeting.id} ({meeting.name}) deleted successfully")
                        deleted_count += 1
                    else:
                        logger.error(f"❌ Failed to delete meeting {meeting.id} ({meeting.name})")
                        failed_count += 1
                except Exception as e:
                    logger.error(f"❌ Error deleting meeting {meeting.id}: {e}")
                    failed_count += 1
            else:
                logger.warning(f"⚠️ Invalid index: {index}")
                failed_count += 1
        
        # Refresh meeting list (all checkboxes will be unchecked)
        refreshed_data = load_meetings_data()
        logger.info(f"📋 Refreshed meeting list with {len(refreshed_data)} meetings")
        
        # Show appropriate success/error notification
        if deleted_count > 0 and failed_count == 0:
            message = f"{deleted_count} meeting{'s' if deleted_count > 1 else ''} deleted successfully! 🗑️"
            gr.Info(message, duration=3)
        elif deleted_count > 0 and failed_count > 0:
            message = f"{deleted_count} deleted, {failed_count} failed"
            gr.Info(message, duration=4)
        else:
            message = f"Failed to delete {failed_count} meeting{'s' if failed_count > 1 else ''}"
            
        return (
            refreshed_data,  # Refresh meeting list with unchecked checkboxes
            gr.update(
                interactive=False,
                value="🗑️ Delete Selected"  # Reset button text
            ),  # Disable delete button
            gr.update(value="💡 Check boxes to select meetings for deletion", visible=True),  # Reset status
            []  # Clear stored indices
        )
            
    except Exception as e:
        logger.error(f"❌ Error deleting meetings: {e}")
        return gr.update(), gr.update(), gr.update(value=f"Error: {str(e)}"), []