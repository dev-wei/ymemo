"""Custom exceptions for audio processing system."""


class AudioProcessingError(Exception):
    """Base exception for audio processing errors."""
    
    def __init__(self, message: str, cause: Exception = None):
        super().__init__(message)
        self.cause = cause


class AudioDeviceError(AudioProcessingError):
    """Raised when there's an issue with audio device access."""
    pass


class TranscriptionProviderError(AudioProcessingError):
    """Raised when there's an issue with transcription provider."""
    pass


class AWSTranscribeError(TranscriptionProviderError):
    """Raised when there's an AWS Transcribe specific error."""
    pass


class AzureSpeechError(TranscriptionProviderError):
    """Raised when there's an Azure Speech Service specific error."""
    pass


class AzureSpeechConnectionError(AzureSpeechError):
    """Raised when there's an Azure Speech Service connection error."""
    pass


class AzureSpeechAuthenticationError(AzureSpeechError):
    """Raised when there's an Azure Speech Service authentication error."""
    pass


class AzureSpeechConfigurationError(AzureSpeechError):
    """Raised when there's an Azure Speech Service configuration error."""
    pass


class AudioCaptureError(AudioProcessingError):
    """Raised when there's an issue with audio capture."""
    pass


class SessionManagerError(AudioProcessingError):
    """Raised when there's an issue with session management."""
    pass


class ConfigurationError(AudioProcessingError):
    """Raised when there's an issue with configuration."""
    pass


class PipelineError(AudioProcessingError):
    """Raised when there's an issue with the audio processing pipeline."""
    pass


class PipelineTimeoutError(PipelineError):
    """Raised when pipeline operations exceed timeout limits."""
    
    def __init__(self, message: str, timeout_seconds: float, cause: Exception = None):
        super().__init__(message, cause)
        self.timeout_seconds = timeout_seconds


class ResourceCleanupError(PipelineError):
    """Raised when resource cleanup fails during pipeline operations."""
    pass