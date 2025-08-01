"""Central pytest configuration and fixtures for the test suite.

This file provides centralized pytest fixtures that are available to all test files,
reducing duplication and ensuring consistent test setup across the suite.
"""

import sys
import os
import tempfile
import asyncio
from pathlib import Path
from typing import Dict, Any, Generator

import pytest
from unittest.mock import patch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.interfaces import AudioConfig, TranscriptionResult
from src.managers.session_manager import AudioSessionManager
from src.managers.enhanced_session_manager import EnhancedAudioSessionManager

from tests.fixtures.mock_factories import (
    MockAudioProcessorFactory,
    MockSessionManagerFactory,
    MockAudioConfigFactory,
    MockTranscriptionResultFactory,
    MockProviderFactory,
    MockPyAudioFactory
)
from tests.fixtures.async_mocks import AsyncIteratorMock
from tests.fixtures.aws_mocks import AWSMockFactory
from tests.config.test_configs import TestAudioConfigs, TestTranscriptionConfigs
from tests.config.test_constants import TestConstants, SampleAudioData


# ============================================================================
# Session-scoped fixtures (expensive setup, shared across tests)
# ============================================================================

@pytest.fixture(scope="session")
def project_root_path():
    """Path to project root directory."""
    return project_root


@pytest.fixture(scope="session")
def temp_dir():
    """Session-scoped temporary directory for test files."""
    temp_path = tempfile.mkdtemp(prefix="ymemo_tests_")
    yield temp_path
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_path, ignore_errors=True)


# ============================================================================
# Function-scoped fixtures (fresh instances for each test)
# ============================================================================

@pytest.fixture
def reset_singletons():
    """Reset all singleton instances before and after each test."""
    # Reset before test
    singletons = [
        (AudioSessionManager, '_instance'),
        (EnhancedAudioSessionManager, '_instance'),
    ]
    
    for singleton_class, instance_attr in singletons:
        if hasattr(singleton_class, instance_attr):
            setattr(singleton_class, instance_attr, None)
    
    yield
    
    # Reset after test
    for singleton_class, instance_attr in singletons:
        if hasattr(singleton_class, instance_attr):
            setattr(singleton_class, instance_attr, None)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file for testing."""
    def _create_temp_file(suffix: str = '.tmp', content: bytes = None) -> str:
        fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=temp_dir)
        os.close(fd)
        
        if content:
            with open(temp_path, 'wb') as f:
                f.write(content)
        
        return temp_path
    
    return _create_temp_file


# ============================================================================
# Audio Configuration Fixtures
# ============================================================================

@pytest.fixture
def default_audio_config():
    """Default AudioConfig for testing."""
    return TestAudioConfigs.DEFAULT


@pytest.fixture
def high_quality_audio_config():
    """High-quality AudioConfig for performance testing."""
    return TestAudioConfigs.HIGH_QUALITY


@pytest.fixture
def aws_compatible_audio_config():
    """AudioConfig compatible with AWS Transcribe."""
    return TestAudioConfigs.AWS_COMPATIBLE


# ============================================================================
# Mock Factory Fixtures
# ============================================================================

@pytest.fixture
def audio_processor_factory():
    """MockAudioProcessorFactory instance."""
    return MockAudioProcessorFactory()


@pytest.fixture
def session_manager_factory():
    """MockSessionManagerFactory instance."""
    return MockSessionManagerFactory()


@pytest.fixture
def audio_config_factory():
    """MockAudioConfigFactory instance."""
    return MockAudioConfigFactory()


@pytest.fixture
def transcription_result_factory():
    """MockTranscriptionResultFactory instance."""
    return MockTranscriptionResultFactory()


@pytest.fixture
def provider_factory():
    """MockProviderFactory instance."""
    return MockProviderFactory()


# ============================================================================
# Common Mock Objects
# ============================================================================

@pytest.fixture
def mock_audio_processor(audio_processor_factory):
    """Basic mock AudioProcessor."""
    return audio_processor_factory.create_basic_mock()


@pytest.fixture
def mock_running_audio_processor(audio_processor_factory):
    """Mock AudioProcessor in running state."""
    return audio_processor_factory.create_running_mock()


@pytest.fixture
def mock_session_manager(session_manager_factory):
    """Basic mock session manager."""
    return session_manager_factory.create_basic_mock()


@pytest.fixture
def mock_recording_session_manager(session_manager_factory):
    """Mock session manager in recording state."""
    return session_manager_factory.create_recording_mock()


@pytest.fixture
def mock_pyaudio_provider(provider_factory):
    """Mock PyAudio provider."""
    return provider_factory.create_pyaudio_provider_mock()


@pytest.fixture
def mock_aws_provider(provider_factory):
    """Mock AWS Transcribe provider."""
    return provider_factory.create_aws_provider_mock()


@pytest.fixture
def mock_file_provider(provider_factory, temp_file):
    """Mock File audio provider."""
    test_audio_file = temp_file('.wav', SampleAudioData.SILENCE_100MS)
    return provider_factory.create_file_provider_mock(test_audio_file)


# ============================================================================
# PyAudio Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_pyaudio():
    """Complete PyAudio mock setup."""
    return MockPyAudioFactory.create_full_mock()


@pytest.fixture
def mock_pyaudio_devices():
    """Standard mock audio device listing."""
    return {
        0: "Built-in Microphone",
        1: "USB Headset", 
        2: "Bluetooth Headphones"
    }


# ============================================================================
# AWS Mock Fixtures
# ============================================================================

@pytest.fixture
def aws_mock_setup():
    """Complete AWS Transcribe mock setup."""
    return AWSMockFactory.create_full_transcribe_setup()


@pytest.fixture
def aws_error_setup():
    """AWS mock setup with error scenarios."""
    return AWSMockFactory.create_error_scenario_setup()


@pytest.fixture
def aws_partial_results_setup():
    """AWS mock setup with partial results."""
    return AWSMockFactory.create_partial_results_scenario()


# ============================================================================
# Session Manager Fixtures
# ============================================================================

@pytest.fixture
def clean_session_manager(reset_singletons):
    """Fresh AudioSessionManager instance."""
    session_mgr = AudioSessionManager()
    session_mgr._recording_active = False
    session_mgr.background_thread = None
    session_mgr.background_loop = None
    
    yield session_mgr


@pytest.fixture
def enhanced_session_manager(reset_singletons):
    """Fresh EnhancedAudioSessionManager instance."""
    return EnhancedAudioSessionManager()


# ============================================================================
# Audio Data Fixtures
# ============================================================================

@pytest.fixture
def sample_audio_chunk():
    """Sample audio chunk for testing."""
    return SampleAudioData.CHUNK_1024


@pytest.fixture
def sample_audio_chunks():
    """List of sample audio chunks."""
    return SampleAudioData.generate_test_chunks(chunk_size=1024, num_chunks=5)


@pytest.fixture
def sine_wave_audio():
    """Generated sine wave audio data."""
    return SampleAudioData.generate_sine_wave(frequency=440, duration=1.0)


# ============================================================================
# Transcription Result Fixtures
# ============================================================================

@pytest.fixture
def basic_transcription_result(transcription_result_factory):
    """Basic transcription result."""
    return transcription_result_factory.create_basic_result()


@pytest.fixture
def partial_transcription_result(transcription_result_factory):
    """Partial transcription result."""
    return transcription_result_factory.create_partial_result()


@pytest.fixture
def transcription_sequence(transcription_result_factory):
    """Sequence of partial to final transcription results."""
    return transcription_result_factory.create_sequence(
        utterance_id="test_utterance",
        texts=["Hello", "Hello there", "Hello there how"],
        final_text="Hello there how are you?"
    )


# ============================================================================
# Environment and Configuration Fixtures
# ============================================================================

@pytest.fixture
def test_environment():
    """Patch environment with test configuration."""
    env_vars = {
        'LOG_LEVEL': 'WARNING',
        'TRANSCRIPTION_PROVIDER': 'aws',
        'CAPTURE_PROVIDER': 'pyaudio',
        'AWS_REGION': 'us-east-1',
        'AWS_LANGUAGE_CODE': 'en-US'
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def debug_environment():
    """Patch environment with debug configuration."""
    env_vars = {
        'LOG_LEVEL': 'DEBUG',
        'TRANSCRIPTION_PROVIDER': 'aws',
        'CAPTURE_PROVIDER': 'file',
        'AWS_REGION': 'us-east-1',
        'AWS_LANGUAGE_CODE': 'en-US'
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars


# ============================================================================
# Async Test Fixtures
# ============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def async_audio_stream():
    """Mock async audio stream."""
    chunks = [b'\x00' * 1024 for _ in range(5)]
    return AsyncIteratorMock(chunks)


@pytest.fixture
def async_transcription_stream(transcription_sequence):
    """Mock async transcription stream."""
    return AsyncIteratorMock(transcription_sequence)


# ============================================================================
# Performance Test Fixtures
# ============================================================================

@pytest.fixture
def performance_thresholds():
    """Performance thresholds for testing."""
    return {
        'startup_time': 1.0,
        'shutdown_time': 3.0,
        'memory_usage': 150 * 1024 * 1024,  # 150MB
        'response_time': 0.1
    }


# ============================================================================
# Parametrized Fixtures
# ============================================================================

@pytest.fixture(params=['aws', 'mock'])
def transcription_provider_type(request):
    """Parametrized transcription provider types."""
    return request.param


@pytest.fixture(params=['pyaudio', 'file'])
def capture_provider_type(request):
    """Parametrized capture provider types."""
    return request.param


@pytest.fixture(params=[16000, 22050, 44100])
def sample_rate(request):
    """Parametrized sample rates for testing."""
    return request.param


@pytest.fixture(params=[1, 2])
def channel_count(request):
    """Parametrized channel counts."""
    return request.param


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: fast unit tests")
    config.addinivalue_line("markers", "integration: integration tests with multiple components")
    config.addinivalue_line("markers", "performance: performance and resource usage tests")
    config.addinivalue_line("markers", "slow: tests that take longer than 1 second")
    config.addinivalue_line("markers", "aws: tests that require AWS mocking")
    config.addinivalue_line("markers", "pyaudio: tests that require PyAudio mocking")
    config.addinivalue_line("markers", "ui: tests that involve UI components")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location and name."""
    for item in items:
        # Mark slow tests
        if "slow" in item.name.lower() or "performance" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Mark AWS tests
        if "aws" in item.name.lower() or "transcribe" in item.name.lower():
            item.add_marker(pytest.mark.aws)
        
        # Mark PyAudio tests
        if "pyaudio" in item.name.lower() or "audio_device" in item.name.lower():
            item.add_marker(pytest.mark.pyaudio)
        
        # Mark tests by directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)


# ============================================================================
# Logging Configuration for Tests
# ============================================================================

@pytest.fixture(autouse=True)
def configure_logging():
    """Configure logging for all tests."""
    import logging
    
    # Set test-appropriate log level
    log_level = os.getenv('TEST_LOG_LEVEL', 'WARNING')
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(name)s - %(levelname)s - %(message)s',
        force=True
    )
    
    # Suppress noisy loggers
    logging.getLogger('boto3').setLevel(logging.ERROR)
    logging.getLogger('botocore').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)