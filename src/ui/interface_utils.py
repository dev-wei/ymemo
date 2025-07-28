"""Utility functions for the UI interface."""

import logging
from typing import List
from src.managers.meeting_repository import get_all_meetings, create_meeting, MeetingRepositoryError

logger = logging.getLogger(__name__)


def load_meetings_data() -> List[List[str]]:
    """Load meetings from database and format for Gradio Dataframe."""
    try:
        meetings = get_all_meetings()
        if not meetings:
            return [["No meetings yet", "", ""]]
        
        # Convert meetings to display format
        meeting_data = []
        for meeting in meetings:
            meeting_data.append(meeting.to_display_row())
        
        return meeting_data
    except Exception as e:
        logger.error(f"Failed to load meetings: {e}")
        return [["Error loading meetings", "", ""]]


def refresh_meetings_list():
    """Refresh the meetings list."""
    return load_meetings_data()


def save_meeting_to_database(meeting_name: str, duration: float, transcription: str, audio_file_path: str = None):
    """Save a meeting to the database."""
    try:
        if not meeting_name or not meeting_name.strip():
            return False, "Meeting name cannot be empty"
        
        if duration <= 0:
            return False, "Invalid recording duration"
        
        if not transcription or not transcription.strip():
            return False, "No transcription available to save"
        
        # Create meeting in database
        meeting = create_meeting(
            name=meeting_name.strip(),
            duration=duration,
            transcription=transcription.strip(),
            audio_file_path=audio_file_path
        )
        
        logger.info(f"Successfully saved meeting: {meeting.name}")
        return True, f"Meeting '{meeting.name}' saved successfully"
        
    except MeetingRepositoryError as e:
        logger.error(f"Failed to save meeting: {e}")
        return False, f"Failed to save meeting: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error saving meeting: {e}")
        return False, f"Unexpected error: {str(e)}"