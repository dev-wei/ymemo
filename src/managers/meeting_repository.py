"""Meeting repository for database operations."""

import os
import logging
from typing import List, Optional
from datetime import datetime

from ..utils.database import get_supabase_client
from ..core.models import Meeting
from ..utils.exceptions import AudioProcessingError

logger = logging.getLogger(__name__)


class MeetingRepositoryError(AudioProcessingError):
    """Exception raised for meeting repository errors."""
    pass


class MeetingRepository:
    """Repository for meeting database operations."""
    
    def __init__(self):
        self.client = get_supabase_client()
        self.table_name = 'ymemo'
    
    def get_all_meetings(self) -> List[Meeting]:
        """Fetch all meetings from the database, ordered by created_at DESC."""
        try:
            logger.info("🔍 Fetching all meetings from database")
            
            response = self.client.table(self.table_name).select(
                'id, name, duration, transcription, created_at, audio_file_path'
            ).order('created_at', desc=True).execute()
            
            if not response.data:
                logger.info("📝 No meetings found in database")
                return []
            
            meetings = []
            for row in response.data:
                try:
                    meeting = Meeting.from_dict(row)
                    meetings.append(meeting)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse meeting row {row.get('id')}: {e}")
                    continue
            
            logger.info(f"✅ Successfully fetched {len(meetings)} meetings")
            return meetings
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch meetings: {e}")
            raise MeetingRepositoryError(f"Failed to fetch meetings: {e}")
    
    def create_meeting(
        self, 
        name: str, 
        duration: float, 
        transcription: str, 
        audio_file_path: Optional[str] = None
    ) -> Meeting:
        """Create a new meeting in the database."""
        try:
            logger.info(f"💾 Creating new meeting: {name}")
            
            # Validate input
            if not name or not name.strip():
                raise ValueError("Meeting name cannot be empty")
            
            if duration <= 0:
                raise ValueError("Duration must be greater than 0")
            
            if not transcription or not transcription.strip():
                raise ValueError("Transcription cannot be empty")
            
            # Prepare data for insertion
            meeting_data = {
                'name': name.strip(),
                'duration': duration,
                'transcription': transcription.strip(),
                'audio_file_path': audio_file_path
            }
            
            # Insert into database
            response = self.client.table(self.table_name).insert(meeting_data).execute()
            
            if not response.data:
                raise MeetingRepositoryError("Failed to create meeting: No data returned")
            
            # Convert response to Meeting object
            meeting = Meeting.from_dict(response.data[0])
            
            logger.info(f"✅ Successfully created meeting with ID: {meeting.id}")
            return meeting
            
        except ValueError as e:
            logger.error(f"❌ Invalid meeting data: {e}")
            raise MeetingRepositoryError(f"Invalid meeting data: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to create meeting: {e}")
            raise MeetingRepositoryError(f"Failed to create meeting: {e}")
    
    def get_meeting_by_id(self, meeting_id: int) -> Optional[Meeting]:
        """Get a specific meeting by ID."""
        try:
            logger.info(f"🔍 Fetching meeting with ID: {meeting_id}")
            
            response = self.client.table(self.table_name).select(
                'id, name, duration, transcription, created_at, audio_file_path'
            ).eq('id', meeting_id).execute()
            
            if not response.data:
                logger.info(f"📝 No meeting found with ID: {meeting_id}")
                return None
            
            meeting = Meeting.from_dict(response.data[0])
            logger.info(f"✅ Successfully fetched meeting: {meeting.name}")
            return meeting
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch meeting {meeting_id}: {e}")
            raise MeetingRepositoryError(f"Failed to fetch meeting {meeting_id}: {e}")
    
    def get_meetings_count(self) -> int:
        """Get the total number of meetings."""
        try:
            logger.info("🔢 Counting meetings in database")
            
            response = self.client.table(self.table_name).select('id', count='exact').execute()
            count = response.count or 0
            
            logger.info(f"✅ Total meetings count: {count}")
            return count
            
        except Exception as e:
            logger.error(f"❌ Failed to count meetings: {e}")
            raise MeetingRepositoryError(f"Failed to count meetings: {e}")
    
    def test_connection(self) -> bool:
        """Test the database connection."""
        try:
            logger.info("🔌 Testing database connection")
            
            # Try to perform a simple query
            response = self.client.table(self.table_name).select('id').limit(1).execute()
            
            logger.info("✅ Database connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False
    
    def get_recent_meetings(self, limit: int = 10) -> List[Meeting]:
        """Get recent meetings with a limit."""
        try:
            logger.info(f"🔍 Fetching {limit} recent meetings")
            
            response = self.client.table(self.table_name).select(
                'id, name, duration, transcription, created_at, audio_file_path'
            ).order('created_at', desc=True).limit(limit).execute()
            
            if not response.data:
                logger.info("📝 No recent meetings found")
                return []
            
            meetings = []
            for row in response.data:
                try:
                    meeting = Meeting.from_dict(row)
                    meetings.append(meeting)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to parse meeting row {row.get('id')}: {e}")
                    continue
            
            logger.info(f"✅ Successfully fetched {len(meetings)} recent meetings")
            return meetings
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch recent meetings: {e}")
            raise MeetingRepositoryError(f"Failed to fetch recent meetings: {e}")


# Global repository instance
_meeting_repository = None


def get_meeting_repository() -> MeetingRepository:
    """Get the global meeting repository instance."""
    global _meeting_repository
    if _meeting_repository is None:
        _meeting_repository = MeetingRepository()
    return _meeting_repository


# Convenience functions
def get_all_meetings() -> List[Meeting]:
    """Get all meetings."""
    return get_meeting_repository().get_all_meetings()


def create_meeting(name: str, duration: float, transcription: str, audio_file_path: Optional[str] = None) -> Meeting:
    """Create a new meeting."""
    return get_meeting_repository().create_meeting(name, duration, transcription, audio_file_path)


def get_meeting_by_id(meeting_id: int) -> Optional[Meeting]:
    """Get a meeting by ID."""
    return get_meeting_repository().get_meeting_by_id(meeting_id)


def test_database_connection() -> bool:
    """Test database connection."""
    return get_meeting_repository().test_connection()