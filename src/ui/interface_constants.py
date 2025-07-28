"""Constants for the UI interface."""

import gradio as gr

# Theme configurations
AVAILABLE_THEMES = {
    "Default": gr.themes.Default(),
    "Soft": gr.themes.Soft(),
    "Monochrome": gr.themes.Monochrome(),
    "Glass": gr.themes.Glass(),
    "Origin": gr.themes.Origin(),
    "Citrus": gr.themes.Citrus(),
    "Ocean": gr.themes.Ocean(),
    "Base": gr.themes.Base()
}

# Default theme
DEFAULT_THEME = "Ocean"

# Button text constants
BUTTON_TEXT = {
    "start_recording": "🎤 Start Recording",
    "stop_recording": "⏹️ Stop Recording",
    "save_meeting": "💾 Save as New Meeting",
    "starting": "🔄 Starting...",
    "stopping": "⏳ Stopping...",
    "refresh_devices": "🔄 Refresh Devices"
}

# UI text constants
UI_TEXT = {
    "app_title": "🎤 Voice Meeting App",
    "app_subtitle": "### Real-time speech transcription with speaker identification",
    "meeting_list_title": "### Meeting List",
    "live_dialog_title": "### Live Dialog",
    "audio_controls_title": "### Audio Controls"
}

# Placeholder text constants
PLACEHOLDER_TEXT = {
    "meeting_name": "Enter meeting name...",
    "transcription_dialog": "Transcription will appear here when recording starts...",
    "live_transcription": "Transcription will appear here..."
}

# UI dimensions
UI_DIMENSIONS = {
    "dialog_height": 800,
    "timer_interval": 0.5
}

# Table headers
TABLE_HEADERS = {
    "meeting_list": ["Meeting", "Date", "Duration"]
}

# Form labels
FORM_LABELS = {
    "meeting_name": "Meeting Name",
    "duration": "Duration",
    "live_transcription": "Live Transcription",
    "audio_device": "Audio Device",
    "status": "Status"
}

# Default values
DEFAULT_VALUES = {
    "duration_display": "0.0 min",
    "no_devices": "No devices"
}

# Audio configuration
AUDIO_CONFIG = {
    "region": "us-east-1",
    "language_code": "en-US"
}