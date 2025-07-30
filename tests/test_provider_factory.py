#!/usr/bin/env python3
"""
Comprehensive tests for provider factory patterns.

These tests focus on the factory behavior, registration, error handling,
and consistency patterns without requiring external dependencies.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.factory import AudioProcessorFactory
from src.core.interfaces import TranscriptionProvider, AudioCaptureProvider, AudioConfig
from src.utils.exceptions import AWSTranscribeError, AudioCaptureError


class MockTranscriptionProvider(TranscriptionProvider):
    """Mock transcription provider for testing."""
    
    def __init__(self, **kwargs):
        self.config = kwargs
        self.started = False
    
    async def start_stream(self, audio_config):
        self.started = True
    
    async def send_audio(self, audio_chunk):
        pass
    
    async def get_transcription(self):
        yield None
    
    async def stop_stream(self):
        self.started = False


class MockAudioCaptureProvider(AudioCaptureProvider):
    """Mock audio capture provider for testing."""
    
    def __init__(self, **kwargs):
        self.config = kwargs
        self.capturing = False
    
    async def start_capture(self, audio_config, device_id=None):
        self.capturing = True
    
    async def get_audio_stream(self):
        yield b"mock_audio_data"
    
    async def stop_capture(self):
        self.capturing = False
    
    def list_audio_devices(self):
        return {0: "Mock Device 1", 1: "Mock Device 2"}


class TestAudioProcessorFactory(unittest.TestCase):
    """Test the AudioProcessorFactory class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Store original registries
        self.original_transcription_providers = AudioProcessorFactory.TRANSCRIPTION_PROVIDERS.copy()
        self.original_capture_providers = AudioProcessorFactory.CAPTURE_PROVIDERS.copy()
    
    def tearDown(self):
        """Clean up after tests."""
        # Restore original registries
        AudioProcessorFactory.TRANSCRIPTION_PROVIDERS = self.original_transcription_providers
        AudioProcessorFactory.CAPTURE_PROVIDERS = self.original_capture_providers
    
    def test_list_transcription_providers(self):
        """Test listing available transcription providers."""
        providers = AudioProcessorFactory.list_transcription_providers()
        
        # Should return a dictionary
        self.assertIsInstance(providers, dict)
        
        # Should include built-in providers
        self.assertIn('aws', providers)
        self.assertIn('azure', providers)
        
        # Values should be class names
        self.assertEqual(providers['aws'], 'AWSTranscribeProvider')
        self.assertEqual(providers['azure'], 'AzureSpeechProvider')
    
    def test_list_audio_capture_providers(self):
        """Test listing available audio capture providers."""
        providers = AudioProcessorFactory.list_audio_capture_providers()
        
        # Should return a dictionary
        self.assertIsInstance(providers, dict)
        
        # Should include built-in providers
        self.assertIn('pyaudio', providers)
        self.assertIn('file', providers)
        
        # Values should be class names
        self.assertEqual(providers['pyaudio'], 'PyAudioCaptureProvider')
        self.assertEqual(providers['file'], 'FileAudioCaptureProvider')
    
    def test_register_transcription_provider(self):
        """Test registering a custom transcription provider."""
        # Register mock provider
        AudioProcessorFactory.register_transcription_provider('mock', MockTranscriptionProvider)
        
        # Should be in the registry
        providers = AudioProcessorFactory.list_transcription_providers()
        self.assertIn('mock', providers)
        self.assertEqual(providers['mock'], 'MockTranscriptionProvider')
        
        # Should be able to create instance
        provider = AudioProcessorFactory.create_transcription_provider('mock', test_param='value')
        self.assertIsInstance(provider, MockTranscriptionProvider)
        self.assertEqual(provider.config['test_param'], 'value')
    
    def test_register_audio_capture_provider(self):
        """Test registering a custom audio capture provider."""
        # Register mock provider
        AudioProcessorFactory.register_audio_capture_provider('mock', MockAudioCaptureProvider)
        
        # Should be in the registry
        providers = AudioProcessorFactory.list_audio_capture_providers()
        self.assertIn('mock', providers)
        self.assertEqual(providers['mock'], 'MockAudioCaptureProvider')
        
        # Should be able to create instance
        provider = AudioProcessorFactory.create_audio_capture_provider('mock', test_param='value')
        self.assertIsInstance(provider, MockAudioCaptureProvider)
        self.assertEqual(provider.config['test_param'], 'value')
    
    def test_register_invalid_transcription_provider(self):
        """Test registering invalid transcription provider fails."""
        class InvalidProvider:
            pass
        
        with self.assertRaises(TypeError) as cm:
            AudioProcessorFactory.register_transcription_provider('invalid', InvalidProvider)
        
        self.assertIn('must implement TranscriptionProvider interface', str(cm.exception))
    
    def test_register_invalid_audio_capture_provider(self):
        """Test registering invalid audio capture provider fails."""
        class InvalidProvider:
            pass
        
        with self.assertRaises(TypeError) as cm:
            AudioProcessorFactory.register_audio_capture_provider('invalid', InvalidProvider)
        
        self.assertIn('must implement AudioCaptureProvider interface', str(cm.exception))
    
    def test_create_unknown_transcription_provider(self):
        """Test creating unknown transcription provider fails with helpful error."""
        with self.assertRaises(ValueError) as cm:
            AudioProcessorFactory.create_transcription_provider('unknown')
        
        error_msg = str(cm.exception)
        self.assertIn('Unsupported transcription provider', error_msg)
        self.assertIn('Available providers:', error_msg)
        self.assertIn('aws', error_msg)
        self.assertIn('azure', error_msg)
    
    def test_create_unknown_audio_capture_provider(self):
        """Test creating unknown audio capture provider fails with helpful error."""
        with self.assertRaises(ValueError) as cm:
            AudioProcessorFactory.create_audio_capture_provider('unknown')
        
        error_msg = str(cm.exception)
        self.assertIn('Unsupported audio capture provider', error_msg)
        self.assertIn('Available providers:', error_msg)
        self.assertIn('pyaudio', error_msg)
        self.assertIn('file', error_msg)
    
    @patch('src.audio.providers.aws_transcribe.boto3')
    def test_create_aws_provider_success(self, mock_boto3):
        """Test creating AWS provider with valid configuration."""
        # Mock boto3 session and credentials
        mock_session = Mock()
        mock_credentials = Mock()
        mock_credentials.access_key = 'test_key'
        mock_session.get_credentials.return_value = mock_credentials
        mock_boto3.Session.return_value = mock_session
        
        # Should create provider successfully
        provider = AudioProcessorFactory.create_transcription_provider(
            'aws', region='us-east-1', language_code='en-US'
        )
        
        # Verify it's the correct type
        from src.audio.providers.aws_transcribe import AWSTranscribeProvider
        self.assertIsInstance(provider, AWSTranscribeProvider)
        self.assertEqual(provider.region, 'us-east-1')
        self.assertEqual(provider.language_code, 'en-US')
    
    def test_create_aws_provider_invalid_region(self):
        """Test creating AWS provider with invalid region fails."""
        with self.assertRaises(RuntimeError) as cm:
            AudioProcessorFactory.create_transcription_provider('aws', region='', language_code='en-US')
        
        error_msg = str(cm.exception)
        self.assertIn('Failed to initialize transcription provider', error_msg)
        self.assertIn('AWS region must be a non-empty string', error_msg)
    
    def test_create_pyaudio_provider_invalid_device_index(self):
        """Test creating PyAudio provider with invalid device index fails."""
        with self.assertRaises(RuntimeError) as cm:
            AudioProcessorFactory.create_audio_capture_provider('pyaudio', device_index='invalid')
        
        error_msg = str(cm.exception)
        self.assertIn('Failed to initialize audio capture provider', error_msg)
        self.assertIn('device_index must be an integer', error_msg)
    
    def test_factory_error_handling_consistency(self):
        """Test that factory error handling is consistent across provider types."""
        # Test transcription provider error handling
        with self.assertRaises(ValueError) as cm1:
            AudioProcessorFactory.create_transcription_provider('nonexistent')
        
        # Test audio capture provider error handling
        with self.assertRaises(ValueError) as cm2:
            AudioProcessorFactory.create_audio_capture_provider('nonexistent')
        
        # Both should have similar error message structure
        error1 = str(cm1.exception)
        error2 = str(cm2.exception)
        
        self.assertIn('Unsupported', error1)
        self.assertIn('Unsupported', error2)
        self.assertIn('Available providers:', error1)
        self.assertIn('Available providers:', error2)
        self.assertIn('register_', error1)  # Should suggest registration
        self.assertIn('register_', error2)  # Should suggest registration


class TestProviderInterfaces(unittest.TestCase):
    """Test provider interface implementations."""
    
    def test_audio_config_creation(self):
        """Test AudioConfig creation with various parameters."""
        # Test default configuration
        config1 = AudioConfig()
        self.assertEqual(config1.sample_rate, 16000)
        self.assertEqual(config1.channels, 1)
        self.assertEqual(config1.chunk_size, 1024)
        self.assertEqual(config1.format, 'int16')
        
        # Test custom configuration
        config2 = AudioConfig(
            sample_rate=48000,
            channels=2,
            chunk_size=2048,
            format='float32'
        )
        self.assertEqual(config2.sample_rate, 48000)
        self.assertEqual(config2.channels, 2)
        self.assertEqual(config2.chunk_size, 2048)
        self.assertEqual(config2.format, 'float32')
    
    def test_mock_transcription_provider_interface(self):
        """Test that mock provider properly implements interface."""
        provider = MockTranscriptionProvider(test_param='value')
        
        # Should store configuration
        self.assertEqual(provider.config['test_param'], 'value')
        self.assertFalse(provider.started)
        
        # Should implement interface methods (basic test)
        self.assertTrue(hasattr(provider, 'start_stream'))
        self.assertTrue(hasattr(provider, 'send_audio'))
        self.assertTrue(hasattr(provider, 'get_transcription'))
        self.assertTrue(hasattr(provider, 'stop_stream'))
    
    def test_mock_audio_capture_provider_interface(self):
        """Test that mock provider properly implements interface."""
        provider = MockAudioCaptureProvider(test_param='value')
        
        # Should store configuration
        self.assertEqual(provider.config['test_param'], 'value')
        self.assertFalse(provider.capturing)
        
        # Should implement interface methods
        self.assertTrue(hasattr(provider, 'start_capture'))
        self.assertTrue(hasattr(provider, 'get_audio_stream'))
        self.assertTrue(hasattr(provider, 'stop_capture'))
        self.assertTrue(hasattr(provider, 'list_audio_devices'))
        
        # Test device listing
        devices = provider.list_audio_devices()
        self.assertIsInstance(devices, dict)
        self.assertIn(0, devices)
        self.assertIn(1, devices)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions for provider creation."""
    
    @patch('src.audio.providers.aws_transcribe.boto3')
    def test_create_aws_transcribe_provider(self, mock_boto3):
        """Test AWS convenience function."""
        # Mock boto3 session and credentials
        mock_session = Mock()
        mock_credentials = Mock()
        mock_credentials.access_key = 'test_key'
        mock_session.get_credentials.return_value = mock_credentials
        mock_boto3.Session.return_value = mock_session
        
        from src.core.factory import create_aws_transcribe_provider
        
        # Test with defaults
        provider1 = create_aws_transcribe_provider()
        self.assertEqual(provider1.region, 'us-east-1')
        self.assertEqual(provider1.language_code, 'en-US')
        
        # Test with custom parameters
        provider2 = create_aws_transcribe_provider(
            region='us-west-2',
            language_code='es-US',
            profile_name='test-profile'
        )
        self.assertEqual(provider2.region, 'us-west-2')
        self.assertEqual(provider2.language_code, 'es-US')
        self.assertEqual(provider2.profile_name, 'test-profile')
    
    def test_create_pyaudio_capture_provider(self):
        """Test PyAudio convenience function."""
        from src.core.factory import create_pyaudio_capture_provider
        
        try:
            # Test with defaults (may fail without PyAudio, that's OK)
            provider = create_pyaudio_capture_provider()
            self.assertIsNotNone(provider)
        except Exception:
            # Expected if PyAudio not available
            pass
        
        # Test parameter validation
        with self.assertRaises(RuntimeError):
            create_pyaudio_capture_provider(device_index='invalid')


if __name__ == '__main__':
    # Set up logging to reduce noise during tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)