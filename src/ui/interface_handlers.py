"""Event handlers for the UI interface."""

import logging
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
            return [(DEFAULT_VALUES["no_devices"], -1)], DEFAULT_VALUES["no_devices"]
        
        device_index = get_default_device_index()
        default_device = None
        
        # Find default device in the list
        for display_name, index in devices:
            if index == device_index:
                default_device = display_name
                break
        
        # If default not found, use first device
        if default_device is None:
            default_device = devices[0][0]
        
        return devices, default_device
    except Exception as e:
        error_choice = [(f"Error: {str(e)}", -1)]
        return error_choice, error_choice[0][0]


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


# Device Management Handlers
def refresh_devices():
    """Refresh audio device list."""
    try:
        devices, current_device = get_device_choices_and_default()
        status_manager.set_status(
            status_manager.current_status,
            "Devices refreshed"
        )
        return (
            gr.Dropdown(choices=devices, value=current_device),
            status_manager.get_status_message()
        )
    except Exception as e:
        status_manager.set_error(e, "Failed to refresh devices")
        error_choice = [(f"Error: {str(e)}", -1)]
        return (
            gr.Dropdown(choices=error_choice, value=error_choice[0][0]),
            status_manager.get_status_message()
        )


# Recording Control Handlers
def start_recording(device_name, current_state):
    """Start recording with selected device."""
    try:
        logger.info(f"🎤 START RECORDING CLICKED - Device: {device_name}")
        logger.info(f"🎤 Current state: {current_state}")
        
        # Get audio session manager
        audio_session = get_audio_session()
        
        # Preserve existing dialog state instead of clearing
        preserved_state = current_state if current_state is not None else []
        logger.info(f"🎤 Preserving {len(preserved_state)} existing messages")
        
        # Convert preserved state to Gradio format for visual display
        gradio_messages = []
        for msg in preserved_state:
            gradio_messages.append({
                "role": "assistant",
                "content": msg["content"]
            })
        logger.info(f"🎤 Converted {len(gradio_messages)} messages to Gradio format")
        
        # Find device index from name
        devices, _ = get_device_choices_and_default()
        device_index = -1
        for name, index in devices:
            if name == device_name:
                device_index = index
                break
        
        if device_index == -1:
            status_manager.set_error(
                Exception("Device not found"),
                "Selected device not available"
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