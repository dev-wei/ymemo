"""Custom exceptions for audio processing system."""


class AudioProcessingError(Exception):
    """Base exception for audio processing errors."""

    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message)
        self.cause = cause


class AudioDeviceError(AudioProcessingError):
    """Raised when there's an issue with audio device access."""


class TranscriptionProviderError(AudioProcessingError):
    """Raised when there's an issue with transcription provider."""


class AWSTranscribeError(TranscriptionProviderError):
    """Raised when there's an AWS Transcribe specific error."""


class AzureSpeechError(TranscriptionProviderError):
    """Raised when there's an Azure Speech Service specific error."""


class AzureSpeechConnectionError(AzureSpeechError):
    """Raised when there's an Azure Speech Service connection error."""


class AzureSpeechAuthenticationError(AzureSpeechError):
    """Raised when there's an Azure Speech Service authentication error."""


class AzureSpeechConfigurationError(AzureSpeechError):
    """Raised when there's an Azure Speech Service configuration error."""


class AudioCaptureError(AudioProcessingError):
    """Raised when there's an issue with audio capture."""


class SessionManagerError(AudioProcessingError):
    """Raised when there's an issue with session management."""


class ConfigurationError(AudioProcessingError):
    """Raised when there's an issue with configuration."""


class PipelineError(AudioProcessingError):
    """Raised when there's an issue with the audio processing pipeline."""


class PipelineTimeoutError(PipelineError):
    """Raised when pipeline operations exceed timeout limits."""

    def __init__(self, message: str, timeout_seconds: float, cause: Exception = None):
        super().__init__(message, cause)
        self.timeout_seconds = timeout_seconds


class ResourceCleanupError(PipelineError):
    """Raised when resource cleanup fails during pipeline operations."""
