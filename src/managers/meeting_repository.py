"""Meeting repository for database operations."""

import logging

from ..core.models import Meeting
from ..utils.database import get_postgresql_client
from ..utils.exceptions import AudioProcessingError

logger = logging.getLogger(__name__)


class MeetingRepositoryError(AudioProcessingError):
    """Exception raised for meeting repository errors."""


class MeetingRepository:
    """Repository for meeting database operations."""

    def __init__(self):
        self.client = get_postgresql_client()
        self.table_name = "ymemo"

    def get_all_meetings(self) -> list[Meeting]:
        """Fetch all meetings from the database, ordered by created_at DESC."""
        try:
            logger.info("🔍 Fetching all meetings from database")

            query = """
                SELECT id, name, duration, transcription, created_at, updated_at, audio_file_path
                FROM {}
                ORDER BY created_at DESC
            """.format(
                self.table_name
            )

            results = self.client.execute_query(query)

            if not results:
                logger.info("📝 No meetings found in database")
                return []

            meetings = []
            for row in results:
                try:
                    meeting = Meeting.from_dict(dict(row))
                    meetings.append(meeting)
                except Exception as e:
                    logger.warning(
                        f"⚠️ Failed to parse meeting row {row.get('id')}: {e}"
                    )
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
        audio_file_path: str | None = None,
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

            # Prepare parameters
            name = name.strip()
            transcription = transcription.strip()

            # Insert into database
            query = """
                INSERT INTO {} (name, duration, transcription, audio_file_path)
                VALUES (%s, %s, %s, %s)
                RETURNING id, name, duration, transcription, created_at, updated_at, audio_file_path
            """.format(
                self.table_name
            )

            result = self.client.execute_insert_returning(
                query, (name, duration, transcription, audio_file_path)
            )

            # Convert response to Meeting object
            meeting = Meeting.from_dict(result)

            logger.info(f"✅ Successfully created meeting with ID: {meeting.id}")
            return meeting

        except ValueError as e:
            logger.error(f"❌ Invalid meeting data: {e}")
            raise MeetingRepositoryError(f"Invalid meeting data: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to create meeting: {e}")
            raise MeetingRepositoryError(f"Failed to create meeting: {e}")

    def get_meeting_by_id(self, meeting_id: int) -> Meeting | None:
        """Get a specific meeting by ID."""
        try:
            logger.info(f"🔍 Fetching meeting with ID: {meeting_id}")

            query = """
                SELECT id, name, duration, transcription, created_at, updated_at, audio_file_path
                FROM {}
                WHERE id = %s
            """.format(
                self.table_name
            )

            results = self.client.execute_query(query, (meeting_id,))

            if not results:
                logger.info(f"📝 No meeting found with ID: {meeting_id}")
                return None

            meeting = Meeting.from_dict(dict(results[0]))
            logger.info(f"✅ Successfully fetched meeting: {meeting.name}")
            return meeting

        except Exception as e:
            logger.error(f"❌ Failed to fetch meeting {meeting_id}: {e}")
            raise MeetingRepositoryError(f"Failed to fetch meeting {meeting_id}: {e}")

    def get_meetings_count(self) -> int:
        """Get the total number of meetings."""
        try:
            logger.info("🔢 Counting meetings in database")

            query = "SELECT COUNT(*) as count FROM {}".format(self.table_name)

            results = self.client.execute_query(query)
            count = results[0]["count"] if results else 0

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
            query = "SELECT 1 FROM {} LIMIT 1".format(self.table_name)
            self.client.execute_query(query)

            logger.info("✅ Database connection test successful")
            return True

        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False

    def get_recent_meetings(self, limit: int = 10) -> list[Meeting]:
        """Get recent meetings with a limit."""
        try:
            logger.info(f"🔍 Fetching {limit} recent meetings")

            query = """
                SELECT id, name, duration, transcription, created_at, updated_at, audio_file_path
                FROM {}
                ORDER BY created_at DESC
                LIMIT %s
            """.format(
                self.table_name
            )

            results = self.client.execute_query(query, (limit,))

            if not results:
                logger.info("📝 No recent meetings found")
                return []

            meetings = []
            for row in results:
                try:
                    meeting = Meeting.from_dict(dict(row))
                    meetings.append(meeting)
                except Exception as e:
                    logger.warning(
                        f"⚠️ Failed to parse meeting row {row.get('id')}: {e}"
                    )
                    continue

            logger.info(f"✅ Successfully fetched {len(meetings)} recent meetings")
            return meetings

        except Exception as e:
            logger.error(f"❌ Failed to fetch recent meetings: {e}")
            raise MeetingRepositoryError(f"Failed to fetch recent meetings: {e}")

    def delete_meeting(self, meeting_id: int) -> bool:
        """Delete a meeting from the database by ID."""
        try:
            logger.info(f"🗑️ Deleting meeting with ID: {meeting_id}")

            # Validate meeting ID
            if not meeting_id or meeting_id <= 0:
                raise ValueError("Invalid meeting ID")

            # Delete from database
            query = "DELETE FROM {} WHERE id = %s".format(self.table_name)

            rows_affected = self.client.execute_update(query, (meeting_id,))

            if rows_affected > 0:
                logger.info(f"✅ Successfully deleted meeting {meeting_id}")
                return True

            logger.warning(f"⚠️ No meeting found with ID {meeting_id}")
            return False

        except ValueError as e:
            logger.error(f"❌ Invalid meeting ID {meeting_id}: {e}")
            raise MeetingRepositoryError(f"Invalid meeting ID: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to delete meeting {meeting_id}: {e}")
            raise MeetingRepositoryError(f"Failed to delete meeting: {e}")


# Global repository instance
_meeting_repository = None


def get_meeting_repository() -> MeetingRepository:
    """Get the global meeting repository instance."""
    global _meeting_repository
    if _meeting_repository is None:
        _meeting_repository = MeetingRepository()
    return _meeting_repository


# Convenience functions
def get_all_meetings() -> list[Meeting]:
    """Get all meetings."""
    return get_meeting_repository().get_all_meetings()


def create_meeting(
    name: str,
    duration: float,
    transcription: str,
    audio_file_path: str | None = None,
) -> Meeting:
    """Create a new meeting."""
    return get_meeting_repository().create_meeting(
        name, duration, transcription, audio_file_path
    )


def get_meeting_by_id(meeting_id: int) -> Meeting | None:
    """Get a meeting by ID."""
    return get_meeting_repository().get_meeting_by_id(meeting_id)


def delete_meeting_by_id(meeting_id: int) -> bool:
    """Delete a meeting by ID."""
    return get_meeting_repository().delete_meeting(meeting_id)


def test_database_connection() -> bool:
    """Test database connection."""
    return get_meeting_repository().test_connection()
