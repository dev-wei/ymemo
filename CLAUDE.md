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

**IMPORTANT: Test suite has been fully migrated to pytest-based infrastructure (2024)**

```bash
# Run all migrated tests (95 tests, <6 seconds, hardware-free)
source .venv/bin/activate && python -m pytest tests/providers/ tests/aws/ tests/audio/ tests/unit/test_enhanced_session_manager.py tests/unit/test_session_manager_stop.py -v

# Run by test category
source .venv/bin/activate && python -m pytest tests/providers/ -v      # Provider tests (47 tests)
source .venv/bin/activate && python -m pytest tests/aws/ -v           # AWS integration (9 tests)  
source .venv/bin/activate && python -m pytest tests/audio/ -v         # Audio/device tests (10 tests)
source .venv/bin/activate && python -m pytest tests/unit/ -v          # Core unit tests (29 tests)

# Run with coverage
source .venv/bin/activate && python -m pytest tests/providers/ tests/aws/ tests/audio/ tests/unit/test_enhanced_session_manager.py tests/unit/test_session_manager_stop.py --cov=src --cov-report=html

# Legacy test commands (deprecated, use above instead)
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

**Centralized Configuration System:**
- All configuration is managed through `config/audio_config.py`
- Configuration is loaded from environment variables with sensible defaults
- Automatic validation with helpful error messages
- Debug logging shows loaded configuration values

**Key Environment Variables:**

*Provider Selection:*
- `TRANSCRIPTION_PROVIDER` - Choose transcription provider ('aws', 'azure', 'whisper', 'google', default: 'aws')
- `CAPTURE_PROVIDER` - Choose audio capture provider ('pyaudio', 'file', default: 'pyaudio')

*Audio Settings:*
- `AUDIO_SAMPLE_RATE` - Sample rate in Hz (default: 16000)
- `AUDIO_CHANNELS` - Number of audio channels (default: 1)
- `AUDIO_CHUNK_SIZE` - Audio chunk size (default: 1024)
- `AUDIO_FORMAT` - Audio format ('int16', 'int24', 'int32', 'float32', default: 'int16')

*AWS Configuration:*
- `AWS_REGION` - AWS region (default: 'us-east-1')
- `AWS_LANGUAGE_CODE` - Language code (default: 'en-US')
- `AWS_MAX_SPEAKERS` - Maximum speakers for diarization (default: 10)
- `ENABLE_SPEAKER_DIARIZATION` - Enable speaker identification (true/false)

*Performance Settings:*
- `MAX_LATENCY_MS` - Maximum latency in milliseconds (default: 300)
- `ENABLE_PARTIAL_RESULTS` - Enable partial results (default: true)
- `PARTIAL_RESULT_HANDLING` - How to handle partials ('replace', 'append', 'final_only', default: 'replace')
- `CONFIDENCE_THRESHOLD` - Minimum confidence threshold (default: 0.0)

*Other Settings:*
- `LOG_LEVEL` - Set logging level (DEBUG, INFO, WARNING, ERROR)

**Configuration Debugging:**
```python
from config.audio_config import print_config_summary
print_config_summary()  # Shows current configuration
```

**AWS Setup:**
- Requires AWS credentials configured (via ~/.aws/credentials or environment)
- Uses centralized configuration for region and language settings

**Azure Speech Service Configuration:**
- `AZURE_SPEECH_KEY` - Azure Speech Service API key (required)
- `AZURE_SPEECH_REGION` - Azure region (default: 'eastus')
- `AZURE_SPEECH_LANGUAGE` - Language code (default: 'en-US')
- `AZURE_ENABLE_SPEAKER_DIARIZATION` - Enable speaker identification (default: false)
- `AZURE_MAX_SPEAKERS` - Maximum speakers to detect (default: 4)
- `AZURE_SPEECH_TIMEOUT` - Connection timeout in seconds (default: 30)
- Requires `azure-cognitiveservices-speech>=1.45.0` dependency

## Testing Strategy

**MIGRATED PYTEST INFRASTRUCTURE (2024):**
- **95 tests** across 7 core files, **100% pass rate**, **<6 seconds execution**
- **Zero hardware dependencies** - all tests run without PyAudio/AWS/device access
- **Centralized infrastructure** with base classes, fixtures, and mock factories
- **CI/CD ready** - tests run consistently in any environment

**Test Architecture:**
```
tests/
├── providers/     (47 tests) - Provider functionality tests
│   ├── test_provider_factory.py      # Factory pattern & registration (19 tests)
│   ├── test_provider_lifecycle.py    # Lifecycle management (17 tests)
│   └── test_provider_error_handling.py # Error handling patterns (11 tests)
├── aws/          (9 tests)  - AWS integration tests  
│   └── test_aws_connection.py        # AWS connection & streaming mocking (9 tests)
├── audio/        (10 tests) - Audio device tests
│   └── test_device_selection.py      # Device selection & validation (10 tests)
├── unit/         (29 tests) - Core unit tests
│   ├── test_enhanced_session_manager.py  # Enhanced session management (17 tests)
│   └── test_session_manager_stop.py      # Stop functionality (12 tests)
├── base/         - Test infrastructure (BaseTest, BaseIntegrationTest, BaseAsyncTest)
├── fixtures/     - Mock factories, AWS mocks, async utilities
└── conftest.py   - Central pytest configuration with fixtures
```

**Key Test Principles:**
- **Hardware Independence**: All PyAudio calls mocked, no AWS credentials needed, no device access
- **Consistent Patterns**: All tests inherit from base classes with standard fixtures
- **Comprehensive Mocking**: Centralized mock factories for AudioProcessor, Providers, AWS services
- **Performance Focus**: Fast execution through effective mocking and parallel-safe design

**Base Test Classes:**
- `BaseTest` - Unit tests with singleton cleanup and mock factory access
- `BaseIntegrationTest` - Integration tests with extended timeout handling  
- `BaseAsyncTest` - Async tests with proper event loop management

**Mock Strategy:**
- `MockAudioProcessorFactory` - Standardized AudioProcessor mocks
- `MockProviderFactory` - Provider mocks with proper interface compliance
- `MockSessionManagerFactory` - Session manager mocks with state management
- AWS mocking patterns for transcription without actual service calls

**Legacy Test Files (Deprecated):**
- Use migrated pytest versions instead of legacy unittest files
- `test_core_functionality.py` - Use `tests/unit/` instead
- `test_file_audio_capture.py` - Use `tests/audio/test_device_selection.py` instead
- Always follow these rules: Tests should NOT require hardware devices, AWS credentials, or network connectivity

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