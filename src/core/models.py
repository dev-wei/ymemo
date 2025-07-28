"""Data models for the application."""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any
import json


@dataclass
class Meeting:
    """Data model for a meeting record from the ymemo table."""
    
    id: int
    name: str
    duration: Optional[float] = None
    transcription: Optional[str] = None
    created_at: Optional[datetime] = None
    audio_file_path: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Meeting':
        """Create a Meeting instance from a dictionary (database row)."""
        # Handle datetime conversion
        created_at = data.get('created_at')
        if created_at and isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            except ValueError:
                created_at = None
        
        return cls(
            id=data.get('id'),
            name=data.get('name'),
            duration=data.get('duration'),
            transcription=data.get('transcription'),
            created_at=created_at,
            audio_file_path=data.get('audio_file_path')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Meeting instance to dictionary for database operations."""
        data = asdict(self)
        
        # Handle datetime conversion
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        
        return data
    
    def to_display_row(self) -> list:
        """Convert Meeting to display format for Gradio Dataframe."""
        # Format date for display
        date_str = ""
        if self.created_at:
            date_str = self.created_at.strftime("%Y-%m-%d")
        
        # Format duration for display
        duration_str = ""
        if self.duration is not None:
            duration_str = f"{self.duration:.1f} min"
        
        return [self.name or "Unnamed Meeting", date_str, duration_str]
    
    def get_formatted_duration(self) -> str:
        """Get formatted duration string."""
        if self.duration is None:
            return "N/A"
        
        if self.duration < 1:
            return f"{self.duration * 60:.0f} sec"
        else:
            return f"{self.duration:.1f} min"
    
    def get_transcription_preview(self, max_length: int = 100) -> str:
        """Get a preview of the transcription."""
        if not self.transcription:
            return "No transcription available"
        
        if len(self.transcription) <= max_length:
            return self.transcription
        
        return self.transcription[:max_length] + "..."
    
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
    audio_file_path: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def to_meeting(self, name: str) -> Meeting:
        """Convert RecordingSession to Meeting for saving."""
        return Meeting(
            id=0,  # Will be set by database
            name=name,
            duration=self.duration,
            transcription=self.transcription,
            audio_file_path=self.audio_file_path,
            created_at=datetime.now()
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