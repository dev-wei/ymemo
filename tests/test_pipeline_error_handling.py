#!/usr/bin/env python3
"""
Tests for pipeline error handling and resilience patterns.

These tests verify that the audio processing pipeline handles errors
consistently, provides proper cleanup, and maintains system stability.
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
import os
from datetime import datetime, timedelta
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.pipeline_error_handler import (
    PipelineErrorHandler, ErrorSeverity, RetryStrategy
)
from src.core.processor import AudioProcessor
from src.core.interfaces import AudioConfig
from src.utils.exceptions import PipelineError, PipelineTimeoutError, ResourceCleanupError


class TestPipelineErrorHandler(unittest.TestCase):
    """Test the PipelineErrorHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.error_handler = PipelineErrorHandler(
            default_timeout=5.0,
            max_retries=2,
            base_retry_delay=0.1  # Fast for testing
        )
    
    def test_error_handler_initialization(self):
        """Test error handler initialization with custom config."""
        handler = PipelineErrorHandler(
            default_timeout=10.0,
            max_retries=5,
            base_retry_delay=0.5
        )
        
        self.assertEqual(handler.default_timeout, 10.0)
        self.assertEqual(handler.max_retries, 5)
        self.assertEqual(handler.base_retry_delay, 0.5)
        self.assertEqual(handler.error_counts, {})
    
    @pytest.mark.asyncio
    async def test_successful_operation(self):
        """Test successful operation handling."""
        operation_executed = False
        
        async def successful_operation():
            nonlocal operation_executed
            operation_executed = True
        
        async with self.error_handler.handle_pipeline_operation("test_success"):
            await successful_operation()
        
        self.assertTrue(operation_executed)
        self.assertEqual(self.error_handler.error_counts, {})
    
    @pytest.mark.asyncio
    async def test_timeout_error_handling(self):
        """Test timeout error handling."""
        cleanup_called = False
        
        def cleanup_callback():
            nonlocal cleanup_called
            cleanup_called = True
        
        async def slow_operation():
            await asyncio.sleep(10)  # Longer than timeout
        
        with self.assertRaises(PipelineTimeoutError) as cm:
            async with self.error_handler.handle_pipeline_operation(
                "test_timeout",
                timeout=0.1,
                cleanup_callback=cleanup_callback
            ):
                await slow_operation()
        
        self.assertTrue(cleanup_called)
        self.assertIn("timed out", str(cm.exception))
        self.assertEqual(cm.exception.timeout_seconds, 0.1)
        self.assertEqual(self.error_handler.error_counts["test_timeout"], 1)
    
    @pytest.mark.asyncio
    async def test_operation_exception_handling(self):
        """Test general exception handling."""
        cleanup_called = False
        
        def cleanup_callback():
            nonlocal cleanup_called
            cleanup_called = True
        
        async def failing_operation():
            raise ValueError("Test error")
        
        with self.assertRaises(PipelineError) as cm:
            async with self.error_handler.handle_pipeline_operation(
                "test_exception",
                cleanup_callback=cleanup_callback
            ):
                await failing_operation()
        
        self.assertTrue(cleanup_called)
        self.assertIn("Test error", str(cm.exception))
        self.assertEqual(self.error_handler.error_counts["test_exception"], 1)
    
    @pytest.mark.asyncio
    async def test_cleanup_failure_handling(self):
        """Test handling of cleanup failures."""
        def failing_cleanup():
            raise RuntimeError("Cleanup failed")
        
        async def failing_operation():
            raise ValueError("Operation failed")
        
        with self.assertRaises(ResourceCleanupError):
            async with self.error_handler.handle_pipeline_operation(
                "test_cleanup_failure",
                cleanup_callback=failing_cleanup
            ):
                await failing_operation()
    
    @pytest.mark.asyncio
    async def test_retry_strategies(self):
        """Test different retry strategies."""
        # Test no retry
        attempt_count = 0
        
        async def failing_operation():
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("Always fails")
        
        handler = PipelineErrorHandler(max_retries=3, base_retry_delay=0.01)
        
        with self.assertRaises(PipelineError):
            async with handler.handle_pipeline_operation(
                "test_no_retry",
                retry_strategy=RetryStrategy.NONE
            ):
                await failing_operation()
        
        # Should only attempt once with no retry
        self.assertEqual(attempt_count, 1)
    
    def test_retry_delay_calculation(self):
        """Test retry delay calculations."""
        # Linear strategy
        delay = self.error_handler._calculate_retry_delay(RetryStrategy.LINEAR, 3)
        self.assertAlmostEqual(delay, 0.3, places=5)  # base_retry_delay * attempt
        
        # Exponential strategy
        delay = self.error_handler._calculate_retry_delay(RetryStrategy.EXPONENTIAL, 3)
        self.assertAlmostEqual(delay, 0.4, places=5)  # base_retry_delay * (2 ** (attempt - 1))
        
        # Fixed delay
        delay = self.error_handler._calculate_retry_delay(RetryStrategy.FIXED_DELAY, 5)
        self.assertAlmostEqual(delay, 0.1, places=5)  # Always base_retry_delay
    
    @pytest.mark.asyncio
    async def test_safe_cleanup_operations(self):
        """Test safe cleanup of multiple operations."""
        cleanup1_called = False
        cleanup2_called = False
        
        def successful_cleanup():
            nonlocal cleanup1_called
            cleanup1_called = True
        
        def failing_cleanup():
            nonlocal cleanup2_called
            cleanup2_called = True
            raise RuntimeError("Cleanup failed")
        
        cleanup_operations = {
            "operation1": successful_cleanup,
            "operation2": failing_cleanup
        }
        
        results = await self.error_handler.safe_cleanup(
            cleanup_operations,
            timeout_per_operation=1.0
        )
        
        self.assertTrue(cleanup1_called)
        self.assertTrue(cleanup2_called)
        self.assertTrue(results["operation1"])
        self.assertFalse(results["operation2"])
    
    @pytest.mark.asyncio
    async def test_async_cleanup_operations(self):
        """Test safe cleanup with async operations."""
        cleanup_called = False
        
        async def async_cleanup():
            nonlocal cleanup_called
            cleanup_called = True
            await asyncio.sleep(0.01)
        
        cleanup_operations = {"async_op": async_cleanup}
        
        results = await self.error_handler.safe_cleanup(cleanup_operations)
        
        self.assertTrue(cleanup_called)
        self.assertTrue(results["async_op"])
    
    def test_error_summary(self):
        """Test error summary generation."""
        # Record some errors
        self.error_handler._record_error("operation1", ErrorSeverity.HIGH)
        self.error_handler._record_error("operation1", ErrorSeverity.HIGH)
        self.error_handler._record_error("operation2", ErrorSeverity.MEDIUM)
        
        summary = self.error_handler.get_error_summary()
        
        self.assertEqual(summary['error_counts']['operation1'], 2)
        self.assertEqual(summary['error_counts']['operation2'], 1)
        self.assertEqual(summary['total_errors'], 3)
        self.assertEqual(summary['operations_with_errors'], 2)
        self.assertIn('last_error_times', summary)
    
    def test_circuit_breaker_logic(self):
        """Test circuit breaker functionality."""
        operation_name = "test_circuit_breaker"
        
        # Record errors below threshold
        for _ in range(3):
            self.error_handler._record_error(operation_name, ErrorSeverity.HIGH)
        
        # Should not circuit break yet
        self.assertFalse(self.error_handler.should_circuit_break(operation_name, error_threshold=5))
        
        # Record more errors to exceed threshold
        for _ in range(3):
            self.error_handler._record_error(operation_name, ErrorSeverity.HIGH)
        
        # Should circuit break now
        self.assertTrue(self.error_handler.should_circuit_break(operation_name, error_threshold=5))
    
    def test_circuit_breaker_time_window(self):
        """Test circuit breaker time window logic."""
        operation_name = "test_time_window"
        
        # Record old error
        self.error_handler.error_counts[operation_name] = 10
        self.error_handler.last_error_times[operation_name] = datetime.now() - timedelta(minutes=10)
        
        # Should not circuit break due to old timestamp
        self.assertFalse(self.error_handler.should_circuit_break(
            operation_name, 
            error_threshold=5, 
            time_window_minutes=5
        ))
        
        # Error count should be reset
        self.assertEqual(self.error_handler.error_counts[operation_name], 0)


class TestAudioProcessorErrorHandling(unittest.TestCase):
    """Test AudioProcessor error handling integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_transcription_provider = AsyncMock()
        self.mock_capture_provider = AsyncMock()
        
        # Create processor with mocked providers
        self.processor = AudioProcessor(
            transcription_provider='mock_transcription',
            capture_provider='mock_capture',
            error_handler_config={
                'default_timeout': 2.0,
                'max_retries': 2,
                'base_retry_delay': 0.1
            }
        )
    
    @pytest.mark.asyncio
    async def test_processor_initialization_error_handling(self):
        """Test processor initialization with provider failures."""
        
        with patch('src.core.factory.AudioProcessorFactory.create_transcription_provider') as mock_transcription:
            with patch('src.core.factory.AudioProcessorFactory.create_audio_capture_provider') as mock_capture:
                # Mock provider creation failure
                mock_transcription.side_effect = RuntimeError("Provider creation failed")
                
                with self.assertRaises(PipelineError) as cm:
                    await self.processor.initialize()
                
                self.assertIn("Provider creation failed", str(cm.exception))
    
    @pytest.mark.asyncio
    async def test_processor_start_recording_error_handling(self):
        """Test recording start with provider failures."""
        
        # Mock successful initialization
        self.processor.transcription_provider = self.mock_transcription_provider
        self.processor.capture_provider = self.mock_capture_provider
        
        # Mock transcription start failure
        self.mock_transcription_provider.start_stream.side_effect = RuntimeError("Transcription start failed")
        
        with self.assertRaises(PipelineError):
            await self.processor.start_recording(device_id=0)
    
    @pytest.mark.asyncio
    async def test_processor_stop_recording_cleanup(self):
        """Test recording stop with cleanup operations."""
        
        # Set up processor in running state
        self.processor.is_running = True
        self.processor.transcription_provider = self.mock_transcription_provider
        self.processor.capture_provider = self.mock_capture_provider
        self.processor._capture_task = AsyncMock()
        self.processor._transcription_task = AsyncMock()
        
        # Mock provider stop methods
        self.mock_capture_provider.stop_capture = AsyncMock()
        self.mock_transcription_provider.stop_stream = AsyncMock()
        
        # Should complete without error
        await self.processor.stop_recording()
        
        # Verify cleanup was called
        self.mock_capture_provider.stop_capture.assert_called_once()
        self.mock_transcription_provider.stop_stream.assert_called_once()
        self.assertFalse(self.processor.is_running)
    
    def test_processor_pipeline_health_status(self):
        """Test pipeline health status reporting."""
        
        # Set up processor state
        self.processor.is_running = True
        self.processor.transcription_provider = self.mock_transcription_provider
        self.processor.capture_provider = self.mock_capture_provider
        self.processor.current_meeting_id = "test_meeting"
        self.processor.session_transcripts = [Mock(), Mock()]
        
        health = self.processor.get_pipeline_health()
        
        self.assertTrue(health['is_running'])
        self.assertTrue(health['has_providers']['transcription'])
        self.assertTrue(health['has_providers']['capture'])
        self.assertEqual(health['session_info']['meeting_id'], "test_meeting")
        self.assertEqual(health['session_info']['transcript_count'], 2)
        self.assertIn('error_handler', health)
    
    def test_processor_export_session_includes_errors(self):
        """Test that session export includes error information."""
        
        self.processor.current_meeting_id = "test_meeting"
        self.processor.session_transcripts = []
        
        # Record some errors
        self.processor.error_handler._record_error("test_operation", ErrorSeverity.HIGH)
        
        export_data = self.processor.export_session()
        
        self.assertEqual(export_data['meeting_id'], "test_meeting")
        self.assertIn('error_summary', export_data)
        self.assertEqual(export_data['error_summary']['total_errors'], 1)


if __name__ == '__main__':
    # Set up logging to reduce noise during tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests with asyncio support
    unittest.main(verbosity=2)