# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YMemo is a real-time voice meeting transcription application built with Gradio and multiple transcription services (AWS Transcribe and Azure Speech Service). The application captures audio from microphones, sends it to your chosen transcription provider for real-time speech-to-text conversion, and displays the results in a responsive web interface with speaker diarization support.

## Development Setup

**IMPORTANT: Always use the virtual environment (.venv) for all Python operations:**

```bash
# Activate virtual environment before any Python commands
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## Key Commands

### Running the Application
```bash
source .venv/bin/activate && python main.py
```

### Running Tests
```bash
# Run all tests
source .venv/bin/activate && python -m pytest tests/

# Run specific test
source .venv/bin/activate && python tests/test_file_audio_capture.py

# Run core functionality test (no AWS dependency)
source .venv/bin/activate && python tests/test_core_functionality.py
```

### Create Test Audio File
```bash
source .venv/bin/activate && python tests/create_test_audio.py
```

### Test Azure Speech Provider
```bash
source .venv/bin/activate && python test_azure_speech_provider.py
```

## Architecture Overview

### Core Components

**Audio Processing Pipeline:**
- `AudioProcessor` - Main coordinator that orchestrates audio capture and transcription
- `AudioSessionManager` - Singleton that manages recording sessions and UI callbacks
- `AudioProcessorFactory` - Factory pattern for creating transcription and capture providers

**Provider Pattern:**
- `TranscriptionProvider` - Interface for speech-to-text services (AWS Transcribe, Azure Speech Service)
- `AudioCaptureProvider` - Interface for audio input sources (PyAudio, File)
- Providers are swappable via factory configuration and environment variables

**UI Architecture:**
- `src/ui/interface.py` - Gradio-based responsive web interface
- Uses Timer component for real-time updates instead of deprecated polling
- Responsive design with mobile-friendly stacking layout

### Key Design Patterns

**Singleton Pattern:**
- `AudioSessionManager` ensures single recording session
- Thread-safe with proper locking mechanisms

**Factory Pattern:**
- `AudioProcessorFactory` creates providers based on string names
- Supports transcription providers ('aws', 'azure') and capture providers ('pyaudio', 'file')
- Easy provider swapping via TRANSCRIPTION_PROVIDER environment variable

**Provider Pattern:**
- Abstract interfaces in `interfaces.py` allow swapping implementations
- `FileAudioCaptureProvider` for testing without microphone hardware
- `AWSTranscribeProvider` for AWS real-time speech recognition with speaker diarization
- `AzureSpeechProvider` for Azure Speech Service with speaker diarization support

**Smart Partial Results:**
- Tracks `utterance_id` and `sequence_number` to update partial results in-place
- Prevents duplicate entries in Live Dialog panel
- Configurable timeout for treating partial results as final

## Configuration

**Environment Variables:**
- `LOG_LEVEL` - Set logging level (DEBUG, INFO, WARNING, ERROR)
- `TRANSCRIPTION_PROVIDER` - Choose transcription provider ('aws' or 'azure', default: 'aws')
- `CAPTURE_PROVIDER` - Choose audio capture provider (default: 'pyaudio')
- `ENABLE_SPEAKER_DIARIZATION` - Enable speaker identification for AWS (true/false)

**AWS Configuration:**
- Requires AWS credentials configured (via ~/.aws/credentials or environment)
- Default region: us-east-1
- Default language: en-US
- Speaker diarization: Set `ENABLE_SPEAKER_DIARIZATION=true`

**Azure Speech Service Configuration:**
- `AZURE_SPEECH_KEY` - Azure Speech Service API key (required)
- `AZURE_SPEECH_REGION` - Azure region (default: 'eastus')
- `AZURE_SPEECH_LANGUAGE` - Language code (default: 'en-US')
- `AZURE_ENABLE_SPEAKER_DIARIZATION` - Enable speaker identification (default: false)
- `AZURE_MAX_SPEAKERS` - Maximum speakers to detect (default: 4)
- `AZURE_SPEECH_TIMEOUT` - Connection timeout in seconds (default: 30)
- Requires `azure-cognitiveservices-speech>=1.45.0` dependency

## Testing Strategy

**File-Based Testing:**
- Use `FileAudioCaptureProvider` with pre-recorded audio files
- Avoids AWS timeout issues in automated testing
- Test audio file created via `tests/create_test_audio.py`

**Core Functionality Testing:**
- `test_core_functionality.py` - Tests session management without AWS dependency
- `test_file_audio_capture.py` - Tests file-based audio capture directly
- Avoid tests that require live microphone input or AWS streaming
- Always use .venv to run any python code or tests. Always try to use existing test as much as possible, to enrich them to cover more edge cases, rather than always creating new test cases
- Always follow these rules in building test cases: Tests should NOT be included in the automated test suite if they require:
  1. A working audio device
  2. AWS credentials
  3. Network connectivity to AWS Transcribe

**Stop Recording Testing:**
- `test_session_manager_stop.py` - Comprehensive session manager stop functionality tests
- `test_stop_recording_comprehensive.py` - Full stop recording test suite
- `test_audio_processor_stop_integration.py` - Integration tests with real audio processor
- `test_all_stop_recording.py` - Master test runner for all stop recording tests

**Test Coverage:**
- Session manager stop functionality (mocked and real)
- Audio processor stop sequence validation
- Thread safety and concurrency testing
- Error handling and timeout scenarios
- Resource cleanup verification
- State transition validation

## File Structure

```
src/
├── audio/
│   └── providers/           # Audio capture and transcription providers
│       ├── aws_transcribe.py
│       ├── azure_speech.py
│       ├── file_audio_capture.py
│       └── pyaudio_capture.py
├── core/                    # Core business logic and interfaces
│   ├── interfaces.py        # Abstract interfaces for all providers
│   ├── factory.py           # Provider factory with registries
│   └── processor.py         # Main audio processing pipeline
├── managers/                # Management classes
│   └── session_manager.py   # Singleton session management
├── ui/                      # User interface
│   └── interface.py         # Gradio web interface with responsive design
└── utils/                   # Utility modules
    ├── device_utils.py      # Audio device utilities
    ├── exceptions.py        # Custom exceptions
    └── status_manager.py    # Status management
```

## Key Files

- `main.py` - Application entry point with CLI arguments
- `src/core/interfaces.py` - Abstract interfaces for all providers
- `src/core/factory.py` - Provider factory with registries
- `src/core/processor.py` - Main audio processing pipeline
- `src/managers/session_manager.py` - Singleton session management
- `src/ui/interface.py` - Gradio web interface with responsive design
- `config/audio_config.py` - System configuration classes

## Threading and Async

**Threading Model:**
- UI runs in main thread
- Audio processing runs in background thread with separate event loop
- `threading.Event` used for cross-thread signaling (not complex asyncio patterns)

**Critical: Stop Recording Implementation:**
- Uses simple `threading.Event` signaling rather than complex asyncio cross-thread operations
- Avoids "different event loop" errors by keeping asyncio operations within single thread
- Background thread joins with reasonable timeout (2.0 seconds)

## AWS Transcribe Integration

**Streaming Configuration:**
- Uses `amazon-transcribe` library for real-time streaming
- Partial results enabled by default for responsive UI
- Smart partial result handling prevents duplicate entries

**Partial Result Handling:**
- Results grouped by `utterance_id` and ordered by `sequence_number`
- Partial results replace previous partials for same utterance
- Final results replace all partials for that utterance

## Azure Speech Service Integration

**Provider Swapping:**
- Set `TRANSCRIPTION_PROVIDER=azure` to use Azure instead of AWS
- Seamless backend switching without code changes
- Both providers implement the same `TranscriptionProvider` interface

**Azure SDK Integration:**
- Uses `azure-cognitiveservices-speech` SDK for real-time streaming
- Event-driven architecture bridging Azure callbacks to async/await
- Push audio stream for continuous recognition
- Speaker diarization with configurable speaker limits

**Azure-Specific Features:**
- Real-time speech recognition with partial and final results
- Speaker identification in format "Speaker 1", "Speaker 2", etc.
- Connection health monitoring and automatic retry logic
- Comprehensive error handling with Azure-specific exception classes

## Common Issues

**Event Loop Conflicts:**
- Always use `threading.Event` for stop signaling
- Avoid `asyncio.run_coroutine_threadsafe` with different event loops

**AWS Timeout in Tests:**
- Never test with live AWS streaming in automation
- Use file-based audio capture for reliable testing

**Mobile Responsiveness:**
- CSS media queries stack Live Dialog and Audio Controls vertically on narrow screens
- Use `!important` declarations for mobile layout overrides
- Meeting list layout restructured for proper vertical stacking of delete controls

**UI Layout Issues:**
- Meeting list delete controls positioned using column-based layout to prevent horizontal wrapping
- Dataframe and delete controls separated into distinct containers with proper height constraints
- Responsive design ensures consistent layout across all screen sizes

**Application Execution:**
- Never run python main.py, because it will hang as it is a website
- Use uv pip when you can 