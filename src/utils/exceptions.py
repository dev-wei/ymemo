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