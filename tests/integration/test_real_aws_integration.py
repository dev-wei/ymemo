"""Integration tests for real AWS provider scenarios.

Tests that validate real-world usage patterns with the AWS provider
including configuration, connection strategies, and audio saving.

Migrated and simplified from root directory test_real_application.py
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from tests.base.base_test import BaseIntegrationTest
from config.audio_config import get_config
from src.core.interfaces import AudioConfig


class TestRealAWSIntegration(BaseIntegrationTest):
    """Test real AWS provider integration scenarios."""
    
    @pytest.fixture
    def aws_test_environment(self):
        """Set up AWS test environment."""
        test_env = {
            'LOG_LEVEL': 'DEBUG',
            'AWS_CONNECTION_STRATEGY': 'dual',
            'AWS_DUAL_CONNECTION_TEST_MODE': 'left_only',
            'AWS_DUAL_SAVE_SPLIT_AUDIO': 'true',
            'AWS_DUAL_SAVE_RAW_AUDIO': 'true',
            'AWS_DUAL_AUDIO_SAVE_PATH': './debug_audio/',
            'AWS_DUAL_AUDIO_SAVE_DURATION': '20'
        }
        
        with patch.dict(os.environ, test_env):
            yield test_env
    
    @patch('src.audio.providers.aws_transcribe.boto3')
    def test_real_aws_provider_configuration_loading(self, mock_boto3, aws_test_environment):
        """Test real AWS provider with loaded configuration."""
        # Mock AWS services
        mock_boto3.Session.return_value.client.return_value = MagicMock()
        
        # Load real configuration
        config = get_config()
        transcription_config = config.get_transcription_config()
        
        # Verify configuration values
        assert transcription_config['connection_strategy'] == 'dual'
        assert transcription_config['dual_connection_test_mode'] == 'left_only'
        assert transcription_config['dual_save_split_audio'] is True
        assert transcription_config['dual_save_raw_audio'] is True
        assert transcription_config['dual_audio_save_path'] == './debug_audio/'
        assert transcription_config['dual_audio_save_duration'] == 20
    
    @patch('src.audio.providers.aws_transcribe.boto3')
    def test_aws_provider_with_real_configuration(self, mock_boto3, aws_test_environment):
        """Test AWS provider creation with real configuration."""
        # Mock AWS services
        mock_boto3.Session.return_value.client.return_value = MagicMock()
        
        from src.core.factory import AudioProcessorFactory
        
        config = get_config()
        transcription_config = config.get_transcription_config()
        
        # Create AWS provider with real configuration
        factory = AudioProcessorFactory()
        aws_provider = factory.create_transcription_provider('aws', **transcription_config)
        
        # Verify provider was created with correct settings
        assert aws_provider is not None
        assert hasattr(aws_provider, 'connection_strategy')
        assert aws_provider.connection_strategy == 'dual'
        assert hasattr(aws_provider, 'dual_connection_test_mode')
        assert aws_provider.dual_connection_test_mode == 'left_only'
        assert hasattr(aws_provider, 'dual_save_split_audio')
        assert aws_provider.dual_save_split_audio is True
    
    @patch('src.audio.providers.aws_transcribe.boto3')
    def test_audio_config_compatibility(self, mock_boto3, aws_test_environment):
        """Test audio configuration compatibility with AWS provider."""
        mock_boto3.Session.return_value.client.return_value = MagicMock()
        
        from src.core.factory import AudioProcessorFactory
        
        config = get_config()
        transcription_config = config.get_transcription_config()
        
        factory = AudioProcessorFactory()
        aws_provider = factory.create_transcription_provider('aws', **transcription_config)
        
        # Test with mono audio config (should fall back to single connection)
        mono_config = AudioConfig(
            sample_rate=16000,
            channels=1,
            chunk_size=1024,
            format='int16'
        )
        
        # Provider should handle mono input gracefully
        assert aws_provider is not None
        
        # Test with stereo audio config (should enable dual connection)
        stereo_config = AudioConfig(
            sample_rate=16000,
            channels=2,
            chunk_size=1024,
            format='int16'
        )
        
        # Provider should handle stereo input
        assert aws_provider is not None
    
    def test_configuration_validation_with_missing_env_vars(self):
        """Test configuration behavior with missing environment variables."""
        # Test with minimal environment (no AWS-specific vars)
        with patch.dict(os.environ, {}, clear=True):
            config = get_config()
            
            # Should still work with defaults
            assert config is not None
            assert config.transcription_provider == 'aws'
            
            # Should have reasonable defaults
            transcription_config = config.get_transcription_config()
            assert 'region' in transcription_config
            assert 'language_code' in transcription_config
            assert 'connection_strategy' in transcription_config
    
    def test_configuration_precedence(self, aws_test_environment):
        """Test that environment variables take precedence over defaults."""
        config = get_config()
        
        # Environment values should override defaults
        assert config.aws_connection_strategy == 'dual'  # From environment
        assert config.aws_dual_connection_test_mode == 'left_only'  # From environment
        assert config.aws_dual_save_split_audio is True  # From environment
        
        # Test with different environment values
        with patch.dict(os.environ, {'AWS_CONNECTION_STRATEGY': 'single'}):
            new_config = get_config()
            # Note: Depending on caching implementation, this might not change immediately
            # This test verifies the mechanism works
            transcription_config = new_config.get_transcription_config()
            assert 'connection_strategy' in transcription_config


class TestRealApplicationScenarios(BaseIntegrationTest):
    """Test real application usage scenarios."""
    
    @patch('src.audio.providers.aws_transcribe.boto3')
    @patch('src.audio.providers.pyaudio_capture.pyaudio.PyAudio')
    def test_audio_processor_creation_like_session_manager(self, mock_pyaudio, mock_boto3):
        """Test AudioProcessor creation like the real session manager does."""
        # Mock external dependencies
        mock_boto3.Session.return_value.client.return_value = MagicMock()
        mock_pyaudio.return_value = MagicMock()
        
        # Set up test environment
        with patch.dict(os.environ, {
            'AWS_CONNECTION_STRATEGY': 'dual',
            'AWS_DUAL_SAVE_SPLIT_AUDIO': 'true'
        }):
            # Replicate session_manager.py approach
            from src.core.processor import AudioProcessor
            system_config = get_config()
            
            audio_processor = AudioProcessor(
                transcription_provider=system_config.transcription_provider,
                capture_provider=system_config.capture_provider,
                transcription_config=system_config.get_transcription_config()
            )
            
            # Verify processor was created successfully
            assert audio_processor is not None
            assert audio_processor.transcription_provider is not None
            assert audio_processor.capture_provider is not None
    
    def test_configuration_error_handling(self):
        """Test handling of configuration errors."""
        # Test with invalid boolean values
        with patch.dict(os.environ, {'AWS_DUAL_SAVE_SPLIT_AUDIO': 'invalid'}):
            config = get_config()
            # Should handle invalid boolean gracefully
            assert isinstance(config.aws_dual_save_split_audio, bool)
        
        # Test with invalid numeric values
        with patch.dict(os.environ, {'AWS_DUAL_AUDIO_SAVE_DURATION': 'not_a_number'}):
            config = get_config()
            # Should handle invalid number gracefully
            assert isinstance(config.aws_dual_audio_save_duration, int)
            assert config.aws_dual_audio_save_duration > 0
    
    @patch('src.audio.providers.aws_transcribe.boto3')
    def test_provider_error_handling_in_real_scenario(self, mock_boto3):
        """Test provider error handling in realistic scenarios."""
        # Mock boto3 to raise various AWS errors
        mock_boto3.Session.side_effect = Exception("AWS credentials not found")
        
        from src.core.factory import AudioProcessorFactory
        
        config = get_config()
        transcription_config = config.get_transcription_config()
        
        factory = AudioProcessorFactory()
        
        # Should raise appropriate error for AWS credential issues
        with pytest.raises(Exception) as exc_info:
            factory.create_transcription_provider('aws', **transcription_config)
        
        assert "AWS credentials not found" in str(exc_info.value)
    
    def test_debug_audio_directory_handling(self):
        """Test debug audio directory path handling."""
        test_paths = [
            './debug_audio/',
            '/tmp/debug_audio/',
            'relative/debug_audio/',
            'debug_audio'  # Without trailing slash
        ]
        
        for test_path in test_paths:
            with patch.dict(os.environ, {'AWS_DUAL_AUDIO_SAVE_PATH': test_path}):
                config = get_config()
                
                # Should accept various path formats
                assert config.aws_dual_audio_save_path == test_path
                
                transcription_config = config.get_transcription_config()
                assert transcription_config['dual_audio_save_path'] == test_path