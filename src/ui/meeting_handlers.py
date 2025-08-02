"""Meeting management event handlers for the UI interface."""

import logging
from typing import Any

import gradio as gr

from src.managers.meeting_repository import delete_meeting_by_id

from .interface_utils import load_meetings_data, save_meeting_to_database

logger = logging.getLogger(__name__)


class MeetingHandler:
    """Handles meeting-related operations like saving, deleting, and managing meeting data."""

    def __init__(self):
        """Initialize the meeting handler."""

    def create_success_message(self, text: str) -> gr.HTML:
        """Create green success message.

        Args:
            text: Success message text

        Returns:
            Gradio HTML component with success styling
        """
        return gr.HTML(
            f"""
            <div style="color: #155724; padding: 12px; border: 1px solid #c3e6cb; border-radius: 6px; background-color: #d4edda; margin: 10px 0;">
                <strong>✅ Success:</strong> {text}
            </div>
        """
        )

    def create_error_message(self, text: str) -> gr.HTML:
        """Create red error message.

        Args:
            text: Error message text

        Returns:
            Gradio HTML component with error styling
        """
        return gr.HTML(
            f"""
            <div style="color: #721c24; padding: 12px; border: 1px solid #f5c6cb; border-radius: 6px; background-color: #f8d7da; margin: 10px 0;">
                <strong>❌ Error:</strong> {text}
            </div>
        """
        )

    def extract_transcription_from_dialog(self, dialog_messages: list[dict]) -> str:
        """Extract transcription text from dialog messages.

        Args:
            dialog_messages: List of dialog messages from Gradio chatbot

        Returns:
            Concatenated transcription text
        """
        if not dialog_messages:
            return ""

        # Extract content from each message
        transcription_parts = []
        for message in dialog_messages:
            if isinstance(message, dict) and "content" in message:
                content = message["content"].strip()
                if content:
                    transcription_parts.append(content)
            elif isinstance(message, str):
                # Handle direct string messages
                content = message.strip()
                if content:
                    transcription_parts.append(content)

        return "\n".join(transcription_parts)

    def parse_duration_to_minutes(self, duration_display: str) -> float:
        """Parse duration display string to minutes.

        Args:
            duration_display: Duration string like "02:35" or "1:23:45"

        Returns:
            Duration in minutes as float
        """
        try:
            if not duration_display or duration_display.strip() == "00:00":
                return 0.0

            # Split by colons
            parts = duration_display.strip().split(":")

            if len(parts) == 2:  # MM:SS format
                minutes, seconds = map(int, parts)
                return minutes + (seconds / 60.0)
            if len(parts) == 3:  # HH:MM:SS format
                hours, minutes, seconds = map(int, parts)
                return (hours * 60) + minutes + (seconds / 60.0)
            logger.warning(f"⚠️ Unknown duration format: {duration_display}")
            return 0.0

        except (ValueError, AttributeError) as e:
            logger.warning(f"⚠️ Could not parse duration '{duration_display}': {e}")
            return 0.0

    def submit_new_meeting(
        self, meeting_name: str, duration_display: str, dialog_messages: list[dict]
    ) -> tuple[gr.HTML, Any]:
        """Submit a new meeting directly to database with validation.

        Args:
            meeting_name: Name for the meeting
            duration_display: Duration string from UI
            dialog_messages: List of dialog messages from chatbot

        Returns:
            Tuple of (status_message, updated_meeting_list)
        """
        try:
            logger.info(
                f"💾 Submitting new meeting: '{meeting_name}', duration: '{duration_display}'"
            )

            # Validation - Meeting name cannot be empty
            if not meeting_name or not meeting_name.strip():
                error_msg = "Meeting name cannot be empty"
                logger.warning(f"❌ Validation failed: {error_msg}")
                return self.create_error_message(error_msg), gr.update()

            # Extract transcription from dialog messages
            transcription_text = self.extract_transcription_from_dialog(dialog_messages)
            logger.info(
                f"💾 Extracted transcription length: {len(transcription_text)} characters"
            )

            # Parse duration (from "MM:SS" or "HH:MM:SS" format to float minutes)
            duration_minutes = self.parse_duration_to_minutes(duration_display)
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
                audio_file_path=None,  # As specified - keep empty for now
            )

            logger.info(f"💾 Save result: success={success}, message='{message}'")

            if success:
                success_msg = f"Meeting '{meeting_name.strip()}' saved successfully! ℹ️"
                logger.info(f"✅ {success_msg}")

                # Show Gradio info notification
                gr.Info(success_msg, duration=5)

                # Refresh meeting list data
                refreshed_meetings = load_meetings_data()
                logger.info(
                    f"📋 Refreshed meeting list with {len(refreshed_meetings)} meetings"
                )

                # Return empty status message and refreshed meeting list
                return gr.HTML(""), refreshed_meetings
            error_msg = f"Failed to save meeting: {message}"
            logger.error(f"❌ {error_msg}")
            # Keep existing meeting list unchanged on error
            return self.create_error_message(error_msg), gr.update()

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"❌ Error submitting meeting: {e}", exc_info=True)
            # Keep existing meeting list unchanged on error
            return self.create_error_message(error_msg), gr.update()

    def delete_meeting_by_id_input(self, meeting_id_text: str) -> tuple[Any, gr.update]:
        """Delete a meeting by ID entered in text field.

        Args:
            meeting_id_text: Meeting ID as string from text input

        Returns:
            Tuple of (updated_meeting_list, status_message)
        """
        try:
            logger.info(f"🗑️ Delete meeting by ID requested: '{meeting_id_text}'")

            # Validate input
            if not meeting_id_text or not meeting_id_text.strip():
                error_msg = "Please enter a meeting ID"
                logger.warning(f"❌ {error_msg}")
                return (
                    load_meetings_data(),  # Keep current meeting list
                    gr.update(value=error_msg, visible=True),
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
                    gr.update(value=f"❌ {error_msg}", visible=True),
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
                    logger.info(
                        f"📋 Refreshed meeting list with {len(refreshed_data)} meetings"
                    )

                    return (
                        refreshed_data,  # Refresh meeting list
                        gr.update(
                            value="✅ Meeting deleted successfully", visible=True
                        ),
                    )
                error_msg = f"Meeting ID {meeting_id} not found or could not be deleted"
                logger.error(f"❌ {error_msg}")
                return (
                    load_meetings_data(),  # Keep current meeting list
                    gr.update(value=f"❌ {error_msg}", visible=True),
                )

            except Exception as e:
                error_msg = f"Database error: {str(e)}"
                logger.error(f"❌ Database error deleting meeting {meeting_id}: {e}")
                return (
                    load_meetings_data(),  # Keep current meeting list
                    gr.update(value=f"❌ {error_msg}", visible=True),
                )

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"❌ Error in delete operation: {e}", exc_info=True)
            return (
                load_meetings_data(),  # Keep current meeting list
                gr.update(value=f"❌ {error_msg}", visible=True),
            )

    def handle_meeting_row_selection(self, evt) -> None:
        """Handle meeting row selection events.

        Args:
            evt: Gradio SelectData event
        """
        logger.info(f"📋 Meeting row selected: {evt}")
        # For now, just log the selection - could be extended for future features

    def reset_meeting_duration(self) -> str:
        """Reset meeting duration display.

        Returns:
            Default duration string "00:00"
        """
        logger.info("⏰ Resetting meeting duration")
        return "00:00"

    def delete_meeting_with_confirmation(
        self, selected_indices: list[int]
    ) -> tuple[Any, str]:
        """Delete meetings with confirmation (legacy function for compatibility).

        Args:
            selected_indices: List of selected meeting indices

        Returns:
            Tuple of (updated_meeting_list, status_message)
        """
        logger.warning(
            "🗑️ Legacy delete function called - this should not be used in current implementation"
        )

        if not selected_indices:
            return load_meetings_data(), "No meetings selected for deletion"

        try:
            for index in selected_indices:
                # This would need to be implemented based on actual requirements
                # For now, just log and return unchanged data
                logger.info(f"Would delete meeting at index: {index}")

            return (
                load_meetings_data(),
                f"Would have deleted {len(selected_indices)} meetings",
            )

        except Exception as e:
            error_msg = f"Error in bulk deletion: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return load_meetings_data(), error_msg


# Global instance for consistent meeting operations
meeting_handler = MeetingHandler()


# Wrapper functions to maintain compatibility with existing interface code
def submit_new_meeting(
    meeting_name: str, duration_display: str, dialog_messages: list[dict]
) -> tuple[gr.HTML, Any]:
    """Submit a new meeting directly to database with validation."""
    return meeting_handler.submit_new_meeting(
        meeting_name, duration_display, dialog_messages
    )


def delete_meeting_by_id_input(meeting_id_text: str) -> tuple[Any, gr.update]:
    """Delete a meeting by ID entered in text field."""
    return meeting_handler.delete_meeting_by_id_input(meeting_id_text)


def handle_meeting_row_selection(evt) -> None:
    """Handle meeting row selection events."""
    return meeting_handler.handle_meeting_row_selection(evt)


def reset_meeting_duration() -> str:
    """Reset meeting duration display."""
    return meeting_handler.reset_meeting_duration()


def delete_meeting_with_confirmation(selected_indices: list[int]) -> tuple[Any, str]:
    """Delete meetings with confirmation (legacy function)."""
    return meeting_handler.delete_meeting_with_confirmation(selected_indices)


def create_success_message(text: str) -> gr.HTML:
    """Create green success message."""
    return meeting_handler.create_success_message(text)


def create_error_message(text: str) -> gr.HTML:
    """Create red error message."""
    return meeting_handler.create_error_message(text)


def extract_transcription_from_dialog(dialog_messages: list[dict]) -> str:
    """Extract transcription text from dialog messages."""
    return meeting_handler.extract_transcription_from_dialog(dialog_messages)


def parse_duration_to_minutes(duration_display: str) -> float:
    """Parse duration display string to minutes."""
    return meeting_handler.parse_duration_to_minutes(duration_display)
