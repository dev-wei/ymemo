"""Data models for the application."""

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass
class Meeting:
    """Data model for a meeting record from the ymemo table."""

    id: int
    name: str
    duration: float | None = None
    transcription: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    audio_file_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Meeting":
        """Create a Meeting instance from a dictionary (database row)."""
        # Handle datetime conversion for created_at
        created_at = data.get("created_at")
        if created_at and isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                created_at = None

        # Handle datetime conversion for updated_at
        updated_at = data.get("updated_at")
        if updated_at and isinstance(updated_at, str):
            try:
                updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            except ValueError:
                updated_at = None

        return cls(
            id=data.get("id"),
            name=data.get("name"),
            duration=data.get("duration"),
            transcription=data.get("transcription"),
            created_at=created_at,
            updated_at=updated_at,
            audio_file_path=data.get("audio_file_path"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert Meeting instance to dictionary for database operations."""
        data = asdict(self)

        # Handle datetime conversion for created_at
        if self.created_at:
            data["created_at"] = self.created_at.isoformat()

        # Handle datetime conversion for updated_at
        if self.updated_at:
            data["updated_at"] = self.updated_at.isoformat()

        return data

    def to_display_row(self) -> list:
        """Convert Meeting to display format for Gradio Dataframe with ID column."""
        # Format date for display
        date_str = ""
        if self.created_at:
            date_str = self.created_at.strftime("%Y-%m-%d")

        # Format duration for display
        duration_str = ""
        if self.duration is not None:
            duration_str = f"{self.duration:.1f} min"

        # Return meeting data with ID as first column
        return [
            self.id,  # Meeting ID column
            self.name or "Unnamed Meeting",
            date_str,
            duration_str,
            self.get_word_count_display(),
        ]

    def get_formatted_duration(self) -> str:
        """Get formatted duration string."""
        if self.duration is None:
            return "N/A"

        if self.duration < 1:
            return f"{self.duration * 60:.0f} sec"
        return f"{self.duration:.1f} min"

    def get_transcription_preview(self, max_length: int = 100) -> str:
        """Get a preview of the transcription."""
        if not self.transcription:
            return "No transcription available"

        if len(self.transcription) <= max_length:
            return self.transcription

        return self.transcription[:max_length] + "..."

    def get_word_count(self) -> int:
        """Get the word count of the transcription."""
        if not self.transcription:
            return 0
        # Simple but accurate word counting
        return len(self.transcription.strip().split())

    def get_word_count_display(self) -> str:
        """Get formatted word count for display."""
        count = self.get_word_count()
        if count == 0:
            return "0 words"
        if count == 1:
            return "1 word"
        return f"{count} words"

    def __str__(self) -> str:
        """String representation of Meeting."""
        return f"Meeting(id={self.id}, name='{self.name}', duration={self.duration}min)"

    def __repr__(self) -> str:
        """Detailed string representation of Meeting."""
        return (
            f"Meeting(id={self.id}, name='{self.name}', duration={self.duration}, "
            f"created_at={self.created_at}, audio_file_path='{self.audio_file_path}')"
        )


@dataclass
class RecordingSession:
    """Data model for an active recording session."""

    duration: float = 0.0
    transcription: str = ""
    audio_file_path: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None

    def to_meeting(self, name: str) -> Meeting:
        """Convert RecordingSession to Meeting for saving."""
        return Meeting(
            id=0,  # Will be set by database
            name=name,
            duration=self.duration,
            transcription=self.transcription,
            audio_file_path=self.audio_file_path,
            created_at=datetime.now(),
        )

    def get_duration_minutes(self) -> float:
        """Get duration in minutes."""
        return self.duration / 60.0 if self.duration else 0.0

    def is_valid_for_saving(self) -> bool:
        """Check if session has minimum data required for saving."""
        return bool(self.transcription and self.duration > 0)

    def clear(self) -> None:
        """Clear the session data."""
        self.duration = 0.0
        self.transcription = ""
        self.audio_file_path = None
        self.start_time = None
        self.end_time = None


@dataclass
class Persona:
    """Data model for a persona record from the ymemo_persona table."""

    id: int
    name: str
    description: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Persona":
        """Create a Persona instance from a dictionary (database row)."""
        # Handle datetime conversion
        created_at = data.get("created_at")
        if created_at and isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                created_at = None

        updated_at = data.get("updated_at")
        if updated_at and isinstance(updated_at, str):
            try:
                updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            except ValueError:
                updated_at = None

        return cls(
            id=data.get("id"),
            name=data.get("name"),
            description=data.get("description"),
            created_at=created_at,
            updated_at=updated_at,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert Persona instance to dictionary for database operations."""
        data = asdict(self)

        # Handle datetime conversion
        if self.created_at:
            data["created_at"] = self.created_at.isoformat()
        if self.updated_at:
            data["updated_at"] = self.updated_at.isoformat()

        return data

    def to_display_row(self) -> list:
        """Convert Persona to display format for Gradio Dataframe with ID column."""
        # Format date for display
        created_date_str = ""
        if self.created_at:
            created_date_str = self.created_at.strftime("%Y-%m-%d")

        updated_date_str = ""
        if self.updated_at:
            updated_date_str = self.updated_at.strftime("%Y-%m-%d %H:%M")

        # Return simplified persona data with ID as first column
        return [
            self.id,  # Persona ID column
            self.name or "Unnamed Persona",
            self.description or "No description",
            created_date_str,
            updated_date_str,
        ]

    def get_description_summary(self) -> str:
        """Get a formatted summary of the persona."""
        if self.description:
            return self.description[:100] + (
                "..." if len(self.description) > 100 else ""
            )
        return "No description available"

    def __str__(self) -> str:
        """String representation of Persona."""
        return f"Persona(id={self.id}, name='{self.name}')"

    def __repr__(self) -> str:
        """Detailed string representation of Persona."""
        return (
            f"Persona(id={self.id}, name='{self.name}', "
            f"description='{self.description}', created_at={self.created_at})"
        )
