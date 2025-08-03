"""Event handlers for the UI interface."""

import logging
import tempfile
from datetime import datetime

import gradio as gr

from src.managers.session_manager import get_audio_session
from src.utils.device_utils import get_default_device_index, get_supported_audio_devices
from src.utils.status_manager import status_manager

from .button_state_manager import button_state_manager
from .interface_constants import DEFAULT_VALUES

logger = logging.getLogger(__name__)


def get_device_choices_and_default():
    """Get current audio device choices and default selection."""
    try:
        devices = get_supported_audio_devices(refresh=True)
        if not devices:
            return [(DEFAULT_VALUES["no_devices"], -1)], -1

        device_index = get_default_device_index()
        default_device_index = None

        # Find default device index in the list
        for _display_name, index in devices:
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
            safe_updates["save_btn"],
        )


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
        logger.info(
            f"🔄 Refreshed devices: {devices}, default index: {current_device_index}"
        )
        status_manager.set_status(status_manager.current_status, "Devices refreshed")
        return (
            gr.Dropdown(choices=devices, value=current_device_index),
            status_manager.get_status_message(),
        )
    except Exception as e:
        logger.error(f"❌ Failed to refresh devices: {e}")
        status_manager.set_error(e, "Failed to refresh devices")
        error_choice = [(f"Error: {str(e)}", -1)]
        return (
            gr.Dropdown(choices=error_choice, value=-1),
            status_manager.get_status_message(),
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
            gradio_messages.append({"role": "assistant", "content": msg["content"]})

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
            gradio_messages.append({"role": "assistant", "content": msg["content"]})

        return current_transcriptions, gradio_messages
    except Exception as e:
        logger.error(f"Error in conditional update: {e}")
        return gr.skip(), gr.skip()


# Meeting management functions moved to meeting_handlers.py for better modularity


# Helper functions moved to meeting_handlers.py for better modularity


def create_warning_message(text):
    """Create yellow warning message."""
    return gr.HTML(
        f"""
        <div style="color: #856404; padding: 12px; border: 1px solid #ffeaa7; border-radius: 6px; background-color: #fff3cd; margin: 10px 0;">
            <strong>⚠️ Warning:</strong> {text}
        </div>
    """
    )


# Utility handler functions
def immediate_transcription_update(message):
    """Immediately handle transcription update."""
    logger.debug(f"UI: Immediate transcription update: {message}")
    # This will be handled by the session manager directly


def setup_save_callback():
    """Setup JavaScript callback for save action."""
    return gr.HTML(
        """
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
    """
    )


def handle_copy_event(copy_data=None):
    """Handle copy events from the chatbot."""
    try:
        logger.info("📋 Copy event triggered")
        if copy_data and hasattr(copy_data, 'value'):
            logger.info(
                f"📋 Copied content length: {len(copy_data.value) if copy_data.value else 0} characters"
            )
        else:
            logger.info("📋 Copy event occurred (no data available)")

        # Log copy usage for analytics (optional)
        if copy_data and hasattr(copy_data, 'value') and copy_data.value:
            content_preview = (
                copy_data.value[:100] + "..."
                if len(copy_data.value) > 100
                else copy_data.value
            )
            logger.info(f"📋 Copy preview: {content_preview}")

        # Return the copied value (required by Gradio copy event)
        return copy_data.value

    except Exception as e:
        logger.error(f"❌ Error handling copy event: {e}")
        # Return original value even on error
        return copy_data.value if hasattr(copy_data, "value") else ""


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
            "total_duration_seconds": audio_session.get_current_duration_seconds(),
            "total_duration_formatted": audio_session.get_formatted_duration(),
            "recording_segments": audio_session.get_recording_segments(),
            "segment_count": len(audio_session.get_recording_segments()),
            "is_recording": audio_session.is_recording(),
        }

        logger.info(
            f"📊 Duration analytics: {analytics['total_duration_formatted']} across {analytics['segment_count']} segments"
        )

        return analytics

    except Exception as e:
        logger.error(f"❌ Error getting duration analytics: {e}")
        return {
            "total_duration_seconds": 0.0,
            "total_duration_formatted": "00:00",
            "recording_segments": [],
            "segment_count": 0,
            "is_recording": False,
        }


# Meeting List Management handlers moved to meeting_handlers.py for better modularity
