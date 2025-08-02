"""Recording-related event handlers for the UI interface."""

import logging

import gradio as gr

from src.managers.session_manager import get_audio_session
from src.utils.device_utils import get_audio_devices, get_default_device_index
from src.utils.status_manager import status_manager

from .button_state_manager import button_state_manager
from .interface_constants import AUDIO_CONFIG

logger = logging.getLogger(__name__)


class RecordingHandler:
    """Handles recording-related operations and state management."""

    def __init__(self):
        """Initialize the recording handler."""
        self.audio_session = None

    def _get_audio_session(self):
        """Get or initialize the audio session manager."""
        if self.audio_session is None:
            self.audio_session = get_audio_session()
        return self.audio_session

    def _get_device_choices_and_default(self):
        """Get current audio device choices and default selection."""
        try:
            devices = get_audio_devices(refresh=True)
            if not devices:
                return [("No devices available", -1)], -1

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

    def _update_button_states(self):
        """Update button states based on current recording status."""
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

    def _validate_device_selection(self, device_selection) -> tuple[bool, int, str]:
        """Validate and extract device index from selection.

        Args:
            device_selection: Device selection from UI (int or str)

        Returns:
            Tuple of (is_valid, device_index, error_message)
        """
        device_index = None

        # Handle device selection - should be device index directly
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
                logger.warning(
                    f"🎤 Received device name instead of index: '{device_selection}', attempting name lookup"
                )
                devices, _ = self._get_device_choices_and_default()
                for name, index in devices:
                    if name == device_selection:
                        device_index = index
                        logger.info(
                            f"🎤 Found device index by name: {name} -> {device_index}"
                        )
                        break

        if device_index is None or device_index == -1:
            return False, -1, f"Invalid device selection: {device_selection}"

        # Validate that the device index exists in the current device list
        devices, _ = self._get_device_choices_and_default()
        valid_indices = [index for name, index in devices]
        if device_index not in valid_indices:
            return (
                False,
                device_index,
                f"Device index {device_index} is not available. Available devices: {valid_indices}",
            )

        return True, device_index, ""

    def _prepare_gradio_messages(self, preserved_state: list[dict]) -> list[dict]:
        """Convert preserved state to Gradio message format.

        Args:
            preserved_state: List of message dictionaries

        Returns:
            List of Gradio-formatted messages
        """
        gradio_messages = []
        for msg in preserved_state:
            gradio_messages.append({"role": "assistant", "content": msg["content"]})
        logger.info(f"🎤 Converted {len(gradio_messages)} messages to Gradio format")
        return gradio_messages

    def start_recording(
        self, device_selection, current_state
    ) -> tuple[str, list[dict], list[dict], gr.update, gr.update, gr.update]:
        """Start recording with selected device.

        Args:
            device_selection: Selected audio device (index or name)
            current_state: Current dialog state to preserve

        Returns:
            Tuple of (status_message, preserved_state, gradio_messages, start_btn_state, stop_btn_state, save_btn_state)
        """
        try:
            logger.info("🎤 START RECORDING CLICKED")
            logger.info(
                f"🎤 Device selection: {device_selection} (type: {type(device_selection)})"
            )
            logger.info(
                f"🎤 Current state: {len(current_state) if current_state else 0} messages"
            )

            # Get audio session manager
            audio_session = self._get_audio_session()

            # Preserve existing dialog state instead of clearing
            preserved_state = current_state if current_state is not None else []
            logger.info(f"🎤 Preserving {len(preserved_state)} existing messages")

            # Log current available devices for debugging
            current_devices, current_default = self._get_device_choices_and_default()
            logger.info(f"🎤 Current available devices: {current_devices}")
            logger.info(f"🎤 Current default device index: {current_default}")

            # Convert preserved state to Gradio format for visual display
            gradio_messages = self._prepare_gradio_messages(preserved_state)

            # Validate device selection
            is_valid, device_index, error_msg = self._validate_device_selection(
                device_selection
            )
            if not is_valid:
                logger.error(f"❌ {error_msg}")
                status_manager.set_error(Exception("Invalid device"), error_msg)
                (
                    start_btn_state,
                    stop_btn_state,
                    save_btn_state,
                ) = self._update_button_states()
                return (
                    status_manager.get_status_message(),
                    preserved_state,
                    gradio_messages,
                    start_btn_state,
                    stop_btn_state,
                    save_btn_state,
                )

            # Check if already recording
            if audio_session.is_recording():
                status_manager.set_error(
                    Exception("Already recording"), "Recording already in progress"
                )
                (
                    start_btn_state,
                    stop_btn_state,
                    save_btn_state,
                ) = self._update_button_states()
                return (
                    status_manager.get_status_message(),
                    preserved_state,
                    gradio_messages,
                    start_btn_state,
                    stop_btn_state,
                    save_btn_state,
                )

            # Start recording using session manager
            status_manager.set_initializing()
            (
                start_btn_state,
                stop_btn_state,
                save_btn_state,
            ) = self._update_button_states()

            config = AUDIO_CONFIG

            status_manager.set_connecting()

            if audio_session.start_recording(device_index, config):
                status_manager.set_recording()
            else:
                status_manager.set_error(
                    Exception("Failed to start"), "Could not start recording"
                )

            # Update button states based on final status
            (
                start_btn_state,
                stop_btn_state,
                save_btn_state,
            ) = self._update_button_states()
            return (
                status_manager.get_status_message(),
                preserved_state,
                gradio_messages,
                start_btn_state,
                stop_btn_state,
                save_btn_state,
            )

        except Exception as e:
            logger.error(f"❌ START RECORDING ERROR: {e}")
            import traceback

            traceback.print_exc()
            status_manager.set_error(e, "Failed to start recording")

            # Ensure we have fallback values for preserved_state and gradio_messages
            preserved_state = current_state if current_state is not None else []
            gradio_messages = self._prepare_gradio_messages(preserved_state)
            (
                start_btn_state,
                stop_btn_state,
                save_btn_state,
            ) = self._update_button_states()
            return (
                status_manager.get_status_message(),
                preserved_state,
                gradio_messages,
                start_btn_state,
                stop_btn_state,
                save_btn_state,
            )

    def stop_recording(self) -> tuple[str, gr.update, gr.update, gr.update]:
        """Stop recording.

        Returns:
            Tuple of (status_message, start_btn_state, stop_btn_state, save_btn_state)
        """
        try:
            # Get audio session manager
            audio_session = self._get_audio_session()

            status_manager.set_stopping()
            (
                start_btn_state,
                stop_btn_state,
                save_btn_state,
            ) = self._update_button_states()

            if audio_session.stop_recording():
                status_manager.set_stopped()
            else:
                status_manager.set_error(
                    Exception("Failed to stop"), "Could not stop recording"
                )

            # Update button states based on final status
            (
                start_btn_state,
                stop_btn_state,
                save_btn_state,
            ) = self._update_button_states()
            return (
                status_manager.get_status_message(),
                start_btn_state,
                stop_btn_state,
                save_btn_state,
            )

        except Exception as e:
            status_manager.set_error(e, "Failed to stop recording")
            (
                start_btn_state,
                stop_btn_state,
                save_btn_state,
            ) = self._update_button_states()
            return (
                status_manager.get_status_message(),
                start_btn_state,
                stop_btn_state,
                save_btn_state,
            )


# Global instance for consistent recording operations
recording_handler = RecordingHandler()


# Wrapper functions to maintain compatibility with existing interface code
def start_recording(device_selection, current_state):
    """Start recording with selected device."""
    return recording_handler.start_recording(device_selection, current_state)


def stop_recording():
    """Stop recording."""
    return recording_handler.stop_recording()
