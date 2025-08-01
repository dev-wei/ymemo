"""Base test class providing common functionality for all tests.

This module provides a standardized base class that handles common test setup,
teardown, singleton management, and utility methods that are needed across
all test files.
"""

import sys
import os
import logging
import threading
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import patch

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.managers.session_manager import AudioSessionManager
from src.managers.enhanced_session_manager import EnhancedAudioSessionManager
from tests.fixtures.mock_factories import (
    MockAudioProcessorFactory,
    MockSessionManagerFactory,
    MockAudioConfigFactory
)


class BaseTest:
    """Base test class with common functionality for all tests."""
    
    def setup_method(self):
        """Standard setup method called before each test."""
        # Configure logging
        self.setup_test_logging()
        
        # Reset singleton instances
        self.reset_singletons()
        
        # Initialize test data
        self.test_data = {}
        self.temp_files = []
        
        # Setup mock factories
        self.audio_processor_factory = MockAudioProcessorFactory()
        self.session_manager_factory = MockSessionManagerFactory()
        self.audio_config_factory = MockAudioConfigFactory()
        
        # Test timing
        self.test_start_time = time.time()
    
    def teardown_method(self):
        """Standard teardown method called after each test."""
        # Cleanup temporary files
        self.cleanup_temp_files()
        
        # Reset singletons
        self.reset_singletons()
        
        # Log test completion time
        test_duration = time.time() - self.test_start_time
        logging.debug(f"Test completed in {test_duration:.3f}s")
    
    def setup_test_logging(self):
        """Configure logging for test environment."""
        log_level = os.getenv('TEST_LOG_LEVEL', 'WARNING')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        
        # Suppress noisy loggers during tests
        logging.getLogger('boto3').setLevel(logging.WARNING)
        logging.getLogger('botocore').setLevel(logging.WARNING)
        logging.getLogger('pyaudio').setLevel(logging.ERROR)
    
    def reset_singletons(self):
        """Reset all singleton instances to ensure test isolation."""
        singletons = [
            (AudioSessionManager, '_instance'),
            (EnhancedAudioSessionManager, '_instance'),
        ]
        
        for singleton_class, instance_attr in singletons:
            if hasattr(singleton_class, instance_attr):
                setattr(singleton_class, instance_attr, None)
    
    def create_temp_file(self, suffix: str = '.tmp', content: bytes = None) -> str:
        """Create temporary file that will be cleaned up automatically."""
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        
        if content:
            with open(temp_path, 'wb') as f:
                f.write(content)
        
        self.temp_files.append(temp_path)
        return temp_path
    
    def cleanup_temp_files(self):
        """Clean up all temporary files created during test."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logging.warning(f"Failed to cleanup temp file {temp_file}: {e}")
        self.temp_files.clear()
    
    def assert_mock_called_with_timeout(self, mock_obj, timeout: float = 1.0, *args, **kwargs):
        """Assert that mock was called with specific arguments within timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if mock_obj.called:
                mock_obj.assert_called_with(*args, **kwargs)
                return
            time.sleep(0.01)
        
        raise AssertionError(f"Mock was not called within {timeout}s timeout")
    
    def wait_for_condition(self, condition_func, timeout: float = 1.0, interval: float = 0.01):
        """Wait for condition to become true within timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            time.sleep(interval)
        
        raise AssertionError(f"Condition not met within {timeout}s timeout")
    
    def assert_state_transition(self, obj, attr_name: str, expected_sequence: List[Any], 
                              timeout: float = 1.0):
        """Assert that object attribute transitions through expected sequence."""
        observed_sequence = []
        start_time = time.time()
        
        while len(observed_sequence) < len(expected_sequence) and time.time() - start_time < timeout:
            current_value = getattr(obj, attr_name)
            if not observed_sequence or current_value != observed_sequence[-1]:
                observed_sequence.append(current_value)
            time.sleep(0.01)
        
        if observed_sequence != expected_sequence:
            raise AssertionError(
                f"Expected state sequence {expected_sequence}, "
                f"got {observed_sequence}"
            )
    
    def patch_environment(self, env_vars: Dict[str, str]):
        """Context manager for patching environment variables."""
        return patch.dict(os.environ, env_vars)
    
    def create_default_audio_config(self):
        """Create default AudioConfig for testing."""
        return self.audio_config_factory.create_default()
    
    def create_mock_audio_processor(self, **kwargs):
        """Create mock AudioProcessor with optional customization."""
        if kwargs:
            # Create basic mock and apply customizations
            mock_processor = self.audio_processor_factory.create_basic_mock()
            for attr, value in kwargs.items():
                setattr(mock_processor, attr, value)
            return mock_processor
        return self.audio_processor_factory.create_basic_mock()
    
    def create_mock_session_manager(self, **kwargs):
        """Create mock SessionManager with optional customization."""
        if kwargs:
            mock_manager = self.session_manager_factory.create_basic_mock()
            for attr, value in kwargs.items():
                setattr(mock_manager, attr, value)
            return mock_manager
        return self.session_manager_factory.create_basic_mock()


class BaseIntegrationTest(BaseTest):
    """Base class for integration tests with additional setup."""
    
    def setup_method(self):
        """Setup for integration tests."""
        super().setup_method()
        
        # Additional integration test setup
        self.integration_timeout = 5.0  # Longer timeout for integration tests
        self.setup_provider_mocks()
    
    def setup_provider_mocks(self):
        """Setup provider mocks for integration testing."""
        # This will be implemented with common provider mock setups
        pass
    
    def verify_resource_cleanup(self, resources: List[Any]):
        """Verify that resources were properly cleaned up."""
        for resource in resources:
            if hasattr(resource, 'is_active'):
                assert not resource.is_active, f"Resource {resource} not properly cleaned up"
            if hasattr(resource, '_stop_event') and resource._stop_event:
                assert resource._stop_event.is_set(), f"Stop event not set for {resource}"


class BasePerformanceTest(BaseTest):
    """Base class for performance tests with timing utilities."""
    
    def setup_method(self):
        """Setup for performance tests."""
        super().setup_method()
        
        # Performance test configuration
        self.performance_thresholds = {
            'startup_time': 1.0,      # seconds
            'shutdown_time': 3.0,     # seconds
            'memory_usage': 150 * 1024 * 1024,  # 150MB in bytes
        }
        
        self.timing_data = {}
    
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        
        class TimingContext:
            def __init__(self, test_instance, name):
                self.test_instance = test_instance
                self.name = name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.test_instance.timing_data[self.name] = duration
        
        return TimingContext(self, operation_name)
    
    def assert_performance_threshold(self, operation_name: str, threshold: float = None):
        """Assert that operation completed within performance threshold."""
        if operation_name not in self.timing_data:
            raise AssertionError(f"No timing data for operation '{operation_name}'")
        
        actual_time = self.timing_data[operation_name]
        expected_threshold = threshold or self.performance_thresholds.get(operation_name, 1.0)
        
        if actual_time > expected_threshold:
            raise AssertionError(
                f"Operation '{operation_name}' took {actual_time:.3f}s, "
                f"exceeding threshold of {expected_threshold:.3f}s"
            )
    
    def get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    
    def assert_memory_threshold(self, threshold: int = None):
        """Assert that memory usage is within threshold."""
        current_usage = self.get_memory_usage()
        expected_threshold = threshold or self.performance_thresholds['memory_usage']
        
        if current_usage > expected_threshold:
            raise AssertionError(
                f"Memory usage {current_usage / 1024 / 1024:.1f}MB "
                f"exceeds threshold of {expected_threshold / 1024 / 1024:.1f}MB"
            )


class BaseUITest(BaseTest):
    """Base class for UI-related tests."""
    
    def setup_method(self):
        """Setup for UI tests."""
        super().setup_method()
        
        # UI test configuration
        self.ui_timeout = 2.0
        self.setup_gradio_mocks()
    
    def setup_gradio_mocks(self):
        """Setup Gradio component mocks."""
        # This will be implemented when needed
        pass


# Utility functions for test discovery and execution
def get_test_categories() -> Dict[str, str]:
    """Get available test categories and their descriptions."""
    return {
        'unit': 'Fast unit tests with minimal dependencies',
        'integration': 'Integration tests with multiple components',
        'performance': 'Performance and resource usage tests',
        'ui': 'User interface interaction tests',
        'slow': 'Tests that take longer than 1 second',
        'aws': 'Tests that require AWS mocking',
        'pyaudio': 'Tests that require PyAudio mocking'
    }