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
    from .interface import get_button_states
    
    try:
        current_status = status_manager.current_status
        button_configs = get_button_states(current_status)
        
        return (
            gr.update(
                value=button_configs["start_btn"]["text"],
                variant=button_configs["start_btn"]["variant"],
                interactive=button_configs["start_btn"]["interactive"],
                visible=button_configs["start_btn"]["visible"]
            ),
            gr.update(
                value=button_configs["stop_btn"]["text"],
                variant=button_configs["stop_btn"]["variant"],
                interactive=button_configs["stop_btn"]["interactive"],
                visible=button_configs["stop_btn"]["visible"]
            ),
            gr.update(
                value=button_configs["save_btn"]["text"],
                variant=button_configs["save_btn"]["variant"],
                interactive=button_configs["save_btn"]["interactive"],
                visible=button_configs["save_btn"]["visible"]
            )
        )
    except Exception as e:
        logger.error(f"Error updating button states: {e}")
        from .interface_constants import BUTTON_TEXT
        # Return safe defaults
        return (
            gr.update(value=BUTTON_TEXT["start_recording"], variant="primary", interactive=True),
            gr.update(value=BUTTON_TEXT["stop_recording"], variant="secondary", interactive=False),
            gr.update(value=BUTTON_TEXT["save_meeting"], variant="secondary", interactive=False)
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


# Save Meeting Handlers
def open_save_panel():
    """Open the save meeting panel with current recording data."""
    try:
        logger.info("🔓 Save panel button clicked")
        
        # Get audio session manager
        audio_session = get_audio_session()
        
        # Get current transcription from session manager
        current_transcriptions = audio_session.get_current_transcriptions()
        
        # Combine all transcriptions into one text
        current_transcription = ""
        if current_transcriptions:
            transcription_parts = []
            for msg in current_transcriptions:
                transcription_parts.append(msg["content"])
            current_transcription = "\n".join(transcription_parts)
        
        # Get session info for duration
        session_info = audio_session.get_session_info()
        duration = session_info.get('duration', 0.0)
        
        # Format duration for display
        duration_str = f"{duration:.1f} min" if duration > 0 else "0.0 min"
        
        logger.info(f"✅ Opening save panel with {len(current_transcriptions)} transcriptions")
        logger.info(f"✅ Transcription preview: {current_transcription[:100]}...")
        logger.info(f"✅ Duration string: {duration_str}")
        
        # Generate a meaningful default meeting name
        default_name = f"Meeting {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Return JavaScript to show panel and populate form
        return gr.HTML(f"""
            <script>
                setTimeout(function() {{
                    showSavePanel();
                    populateSavePanel('{default_name}', '{datetime.now().strftime("%Y-%m-%d")}', '{duration_str}', {repr(current_transcription)});
                    hideSaveStatus();
                }}, 100);
            </script>
        """)
        
    except Exception as e:
        logger.error(f"❌ Error opening save panel: {e}")
        import traceback
        traceback.print_exc()
        return gr.HTML(f"""
            <script>
                setTimeout(function() {{
                    showSaveStatus('Error: {str(e)}', true);
                }}, 100);
            </script>
        """)


def save_meeting(meeting_name, transcription, duration_str):
    """Save the meeting to database."""
    try:
        logger.info(f"💾 Saving meeting: '{meeting_name}', duration: '{duration_str}'")
        logger.info(f"💾 Transcription length: {len(transcription)} characters")
        
        # Parse duration from string
        duration = float(duration_str.replace(" min", "").replace(" sec", ""))
        logger.info(f"💾 Parsed duration: {duration}")
        
        # Save to database
        success, message = save_meeting_to_database(
            meeting_name=meeting_name,
            duration=duration,
            transcription=transcription,
            audio_file_path=None  # TODO: Add audio file path when available
        )
        
        logger.info(f"💾 Save result: success={success}, message='{message}'")
        
        if success:
            # Close panel and refresh meeting list
            return (
                gr.update(value=load_meetings_data()),  # meeting_list
                gr.HTML(f"""
                    <script>
                        setTimeout(function() {{
                            hideSavePanel();
                            showSaveStatus('{message}', false);
                        }}, 100);
                    </script>
                """)
            )
        else:
            # Show error but keep panel open
            return (
                gr.update(),  # meeting_list (no change)
                gr.HTML(f"""
                    <script>
                        setTimeout(function() {{
                            showSaveStatus('{message}', true);
                        }}, 100);
                    </script>
                """)
            )
            
    except Exception as e:
        logger.error(f"Error saving meeting: {e}")
        return (
            gr.update(),  # meeting_list (no change)
            gr.HTML(f"""
                <script>
                    setTimeout(function() {{
                        showSaveStatus('Error: {str(e)}', true);
                    }}, 100);
                </script>
            """)
        )


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