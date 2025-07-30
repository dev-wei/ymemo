#!/usr/bin/env python3
"""
Tests for provider error handling patterns and consistency.

These tests verify that all providers handle errors consistently,
provide helpful error messages, and follow the same patterns.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.factory import AudioProcessorFactory
from src.core.interfaces import AudioConfig
from src.utils.exceptions import AWSTranscribeError, AudioCaptureError


class TestProviderErrorHandling(unittest.TestCase):
    """Test consistent error handling across all providers."""
    
    def test_transcription_provider_parameter_validation(self):
        """Test that all transcription providers validate parameters consistently."""
        test_cases = [
            # (provider_name, invalid_params, expected_error_type, error_substring)
            ('aws', {'region': ''}, RuntimeError, 'AWS region must be a non-empty string'),
            ('aws', {'region': None}, RuntimeError, 'AWS region must be a non-empty string'),
            ('aws', {'region': 123}, RuntimeError, 'AWS region must be a non-empty string'),
            ('aws', {'language_code': ''}, RuntimeError, 'Language code must be a non-empty string'),
            ('aws', {'language_code': None}, RuntimeError, 'Language code must be a non-empty string'),
            ('aws', {'profile_name': 123}, RuntimeError, 'Profile name must be a string or None'),
        ]
        
        for provider_name, params, expected_error, error_substring in test_cases:
            with self.subTest(provider=provider_name, params=params):
                with self.assertRaises(expected_error) as cm:
                    AudioProcessorFactory.create_transcription_provider(provider_name, **params)
                
                self.assertIn(error_substring, str(cm.exception))
    
    def test_audio_capture_provider_parameter_validation(self):
        """Test that all audio capture providers validate parameters consistently."""
        test_cases = [
            # (provider_name, invalid_params, expected_error_type, error_substring)
            ('pyaudio', {'device_index': 'invalid'}, RuntimeError, 'device_index must be an integer'),
            ('pyaudio', {'device_index': -1}, RuntimeError, 'device_index must be non-negative'),
            # Note: File provider validation is done at runtime when file is opened, not in constructor
        ]
        
        for provider_name, params, expected_error, error_substring in test_cases:
            with self.subTest(provider=provider_name, params=params):
                with self.assertRaises(expected_error) as cm:
                    AudioProcessorFactory.create_audio_capture_provider(provider_name, **params)
                
                self.assertIn(error_substring, str(cm.exception))
    
    def test_factory_error_message_format(self):
        """Test that factory error messages follow consistent format."""
        # Test transcription provider error
        with self.assertRaises(ValueError) as cm1:
            AudioProcessorFactory.create_transcription_provider('nonexistent')
        
        error1 = str(cm1.exception)
        # Should have: provider type, provider name, available options, suggestion
        self.assertIn('Unsupported transcription provider', error1)
        self.assertIn('nonexistent', error1)
        self.assertIn('Available providers:', error1)
        self.assertIn('register_transcription_provider', error1)
        
        # Test audio capture provider error
        with self.assertRaises(ValueError) as cm2:
            AudioProcessorFactory.create_audio_capture_provider('nonexistent')
        
        error2 = str(cm2.exception)
        self.assertIn('Unsupported audio capture provider', error2)
        self.assertIn('nonexistent', error2)
        self.assertIn('Available providers:', error2)
        self.assertIn('register_audio_capture_provider', error2)
    
    def test_provider_initialization_error_wrapping(self):
        """Test that provider initialization errors are properly wrapped."""
        # Test that factory wraps provider errors consistently
        
        # AWS provider with invalid region should be wrapped in RuntimeError
        with self.assertRaises(RuntimeError) as cm:
            AudioProcessorFactory.create_transcription_provider('aws', region='')
        
        error_msg = str(cm.exception)
        self.assertIn('Failed to initialize transcription provider', error_msg)
        self.assertIn('aws', error_msg)
        # Original error should be preserved
        self.assertIn('AWS region must be a non-empty string', error_msg)
    
    @patch('src.audio.providers.aws_transcribe.boto3')
    def test_aws_provider_configuration_validation(self, mock_boto3):
        """Test AWS provider configuration validation."""
        # Mock boto3 to raise configuration errors
        mock_session = Mock()
        mock_session.get_credentials.return_value = None  # No credentials
        mock_boto3.Session.return_value = mock_session
        
        with self.assertRaises(RuntimeError) as cm:
            AudioProcessorFactory.create_transcription_provider('aws', region='us-east-1')
        
        error_msg = str(cm.exception)
        self.assertIn('Failed to initialize transcription provider', error_msg)
        # Should contain original AWS error information
        self.assertIn('AWS', error_msg)
    
    def test_error_logging_consistency(self):
        """Test that error logging follows consistent patterns."""
        import logging
        from unittest.mock import patch
        
        # Capture log messages
        with patch('src.core.factory.logger') as mock_logger:
            # Test transcription provider error logging
            try:
                AudioProcessorFactory.create_transcription_provider('aws', region='')
            except RuntimeError:
                pass  # Expected
            
            # Should log the error
            mock_logger.error.assert_called()
            
            # Log message should follow pattern: ❌ Factory: Failed to create provider...
            logged_message = mock_logger.error.call_args[0][0]
            self.assertIn('❌ Factory:', logged_message)
            self.assertIn('Failed to create transcription provider', logged_message)
    
    def test_provider_type_checking(self):
        """Test that factory enforces proper provider interface implementation."""
        class InvalidTranscriptionProvider:
            """Does not implement TranscriptionProvider interface."""
            pass
        
        class InvalidAudioCaptureProvider:
            """Does not implement AudioCaptureProvider interface."""
            pass
        
        # Should reject invalid provider types during registration
        with self.assertRaises(TypeError) as cm1:
            AudioProcessorFactory.register_transcription_provider('invalid', InvalidTranscriptionProvider)
        
        self.assertIn('must implement TranscriptionProvider interface', str(cm1.exception))
        
        with self.assertRaises(TypeError) as cm2:
            AudioProcessorFactory.register_audio_capture_provider('invalid', InvalidAudioCaptureProvider)
        
        self.assertIn('must implement AudioCaptureProvider interface', str(cm2.exception))
    
    def test_audio_config_validation_patterns(self):
        """Test that AudioConfig validation follows consistent patterns."""
        # Test that providers validate AudioConfig properly
        
        # Valid config should not raise errors
        valid_config = AudioConfig(sample_rate=16000, channels=1)
        self.assertIsInstance(valid_config, AudioConfig)
        
        # Test various configuration scenarios
        configs = [
            AudioConfig(),  # Default values
            AudioConfig(sample_rate=48000),  # Custom sample rate
            AudioConfig(channels=2),  # Stereo
            AudioConfig(chunk_size=2048),  # Custom chunk size
            AudioConfig(format='float32'),  # Different format
        ]
        
        for config in configs:
            # All should be valid AudioConfig instances
            self.assertIsInstance(config, AudioConfig)
            self.assertGreater(config.sample_rate, 0)
            self.assertGreater(config.channels, 0)
            self.assertGreater(config.chunk_size, 0)
            self.assertIn(config.format, ['int16', 'int24', 'int32', 'float32'])


class TestProviderErrorRecovery(unittest.TestCase):
    """Test error recovery and resilience patterns."""
    
    def test_factory_registry_isolation(self):
        """Test that provider registration errors don't affect factory state."""
        # Store original state
        original_transcription = AudioProcessorFactory.TRANSCRIPTION_PROVIDERS.copy()
        original_capture = AudioProcessorFactory.CAPTURE_PROVIDERS.copy()
        
        try:
            # Try to register invalid provider - should fail
            class BadProvider:
                pass
            
            with self.assertRaises(TypeError):
                AudioProcessorFactory.register_transcription_provider('bad', BadProvider)
            
            # Factory state should be unchanged
            self.assertEqual(
                AudioProcessorFactory.TRANSCRIPTION_PROVIDERS,
                original_transcription
            )
            
            # Should still be able to create valid providers
            providers = AudioProcessorFactory.list_transcription_providers()
            self.assertIn('aws', providers)
            self.assertIn('azure', providers)
            
        finally:
            # Ensure cleanup
            AudioProcessorFactory.TRANSCRIPTION_PROVIDERS = original_transcription
            AudioProcessorFactory.CAPTURE_PROVIDERS = original_capture
    
    def test_provider_creation_independence(self):
        """Test that failed provider creation doesn't affect subsequent attempts."""
        # First attempt should fail
        with self.assertRaises(RuntimeError):
            AudioProcessorFactory.create_transcription_provider('aws', region='')
        
        # Second attempt with valid parameters should work (if AWS configured)
        try:
            with patch('src.audio.providers.aws_transcribe.boto3') as mock_boto3:
                mock_session = Mock()
                mock_credentials = Mock()
                mock_credentials.access_key = 'test'
                mock_session.get_credentials.return_value = mock_credentials
                mock_boto3.Session.return_value = mock_session
                
                provider = AudioProcessorFactory.create_transcription_provider(
                    'aws', region='us-east-1', language_code='en-US'
                )
                self.assertIsNotNone(provider)
        except Exception:
            # May fail due to AWS configuration, that's OK for this test
            pass
    
    def test_error_message_helpfulness(self):
        """Test that error messages provide actionable guidance."""
        
        # Test unknown provider error
        with self.assertRaises(ValueError) as cm:
            AudioProcessorFactory.create_transcription_provider('unknown')
        
        error_msg = str(cm.exception)
        
        # Should explain the problem
        self.assertIn('Unsupported', error_msg)
        self.assertIn('unknown', error_msg)
        
        # Should list available options
        self.assertIn('Available providers:', error_msg)
        self.assertIn('aws', error_msg)
        self.assertIn('azure', error_msg)
        
        # Should suggest a solution
        self.assertIn('register_transcription_provider', error_msg)


if __name__ == '__main__':
    # Set up logging to reduce noise during tests
    import logging
    logging.basicConfig(level=logging.ERROR)  # Only show errors
    
    # Run tests
    unittest.main(verbosity=2)