#!/usr/bin/env python3
"""
Tests for resource management and task lifecycle patterns.

These tests verify that resources and tasks are properly managed,
cleaned up, and monitored throughout their lifecycle.
"""

import unittest
import asyncio
from unittest.mock import Mock, AsyncMock
import sys
import os
from datetime import datetime, timedelta
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.resource_manager import (
    ResourceManager, ManagedResource, ManagedTask,
    ResourceState, TaskState
)
from src.core.processor import AudioProcessor


class TestManagedResource(unittest.TestCase):
    """Test ManagedResource class."""
    
    def test_managed_resource_initialization(self):
        """Test managed resource initialization."""
        mock_resource = Mock()
        cleanup_func = Mock()
        
        managed = ManagedResource(
            resource_id="test_resource",
            resource=mock_resource,
            cleanup_func=cleanup_func,
            timeout=10.0
        )
        
        self.assertEqual(managed.resource_id, "test_resource")
        self.assertEqual(managed.resource, mock_resource)
        self.assertEqual(managed.cleanup_func, cleanup_func)
        self.assertEqual(managed.timeout, 10.0)
        self.assertEqual(managed.state, ResourceState.ACTIVE)
        self.assertEqual(managed.cleanup_attempts, 0)
    
    def test_resource_access_tracking(self):
        """Test resource access tracking."""
        mock_resource = Mock()
        managed = ManagedResource("test", mock_resource)
        
        initial_access = managed.last_access
        
        # Access the resource
        accessed = managed.access()
        
        self.assertEqual(accessed, mock_resource)
        self.assertGreater(managed.last_access, initial_access)
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_success(self):
        """Test successful resource cleanup."""
        mock_resource = Mock()
        cleanup_func = AsyncMock()
        
        managed = ManagedResource(
            "test_resource",
            mock_resource,
            cleanup_func=cleanup_func,
            timeout=5.0
        )
        
        success = await managed.cleanup()
        
        self.assertTrue(success)
        self.assertEqual(managed.state, ResourceState.STOPPED)
        self.assertEqual(managed.cleanup_attempts, 1)
        cleanup_func.assert_called_once_with(mock_resource)
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_sync_function(self):
        """Test resource cleanup with sync function."""
        mock_resource = Mock()
        cleanup_func = Mock()
        
        managed = ManagedResource(
            "test_resource",
            mock_resource,
            cleanup_func=cleanup_func
        )
        
        success = await managed.cleanup()
        
        self.assertTrue(success)
        self.assertEqual(managed.state, ResourceState.STOPPED)
        cleanup_func.assert_called_once_with(mock_resource)
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_timeout(self):
        """Test resource cleanup timeout handling."""
        mock_resource = Mock()
        
        async def slow_cleanup(resource):
            await asyncio.sleep(10)  # Longer than timeout
        
        managed = ManagedResource(
            "test_resource",
            mock_resource,
            cleanup_func=slow_cleanup,
            timeout=0.1
        )
        
        success = await managed.cleanup()
        
        self.assertFalse(success)
        self.assertEqual(managed.state, ResourceState.ERROR)
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_exception(self):
        """Test resource cleanup exception handling."""
        mock_resource = Mock()
        
        async def failing_cleanup(resource):
            raise RuntimeError("Cleanup failed")
        
        managed = ManagedResource(
            "test_resource",
            mock_resource,
            cleanup_func=failing_cleanup
        )
        
        success = await managed.cleanup()
        
        self.assertFalse(success)
        self.assertEqual(managed.state, ResourceState.ERROR)
    
    def test_resource_staleness_detection(self):
        """Test resource staleness detection."""
        mock_resource = Mock()
        managed = ManagedResource("test", mock_resource)
        
        # Fresh resource should not be stale
        self.assertFalse(managed.is_stale(max_age_minutes=30))
        
        # Make resource appear old
        managed.last_access = datetime.now() - timedelta(minutes=45)
        self.assertTrue(managed.is_stale(max_age_minutes=30))
    
    def test_resource_info_generation(self):
        """Test resource info generation for monitoring."""
        mock_resource = Mock()
        managed = ManagedResource("test_resource", mock_resource)
        
        info = managed.get_info()
        
        self.assertEqual(info['resource_id'], "test_resource")
        self.assertEqual(info['state'], ResourceState.ACTIVE.value)
        self.assertEqual(info['type'], "Mock")
        self.assertIn('created_at', info)
        self.assertIn('last_access', info)
        self.assertIn('age_seconds', info)


class TestManagedTask(unittest.TestCase):
    """Test ManagedTask class."""
    
    @pytest.mark.asyncio
    async def test_successful_task_execution(self):
        """Test successful task execution."""
        result_value = "task_result"
        
        async def successful_task():
            await asyncio.sleep(0.01)
            return result_value
        
        managed_task = ManagedTask("test_task", successful_task())
        
        # Wait for task completion
        result = await managed_task.task
        
        self.assertEqual(result, result_value)
        self.assertEqual(managed_task.state, TaskState.COMPLETED)
        self.assertIsNotNone(managed_task.started_at)
        self.assertIsNotNone(managed_task.completed_at)
    
    @pytest.mark.asyncio
    async def test_task_timeout_handling(self):
        """Test task timeout handling."""
        async def slow_task():
            await asyncio.sleep(10)  # Longer than timeout
        
        managed_task = ManagedTask("test_task", slow_task(), timeout=0.1)
        
        with self.assertRaises(asyncio.TimeoutError):
            await managed_task.task
        
        self.assertEqual(managed_task.state, TaskState.FAILED)
        self.assertIsInstance(managed_task.error, asyncio.TimeoutError)
    
    @pytest.mark.asyncio
    async def test_task_exception_handling(self):
        """Test task exception handling."""
        async def failing_task():
            raise ValueError("Task failed")
        
        managed_task = ManagedTask("test_task", failing_task())
        
        with self.assertRaises(ValueError):
            await managed_task.task
        
        self.assertEqual(managed_task.state, TaskState.FAILED)
        self.assertIsInstance(managed_task.error, ValueError)
    
    @pytest.mark.asyncio
    async def test_task_cancellation(self):
        """Test task cancellation."""
        cleanup_called = False
        
        def cleanup_callback():
            nonlocal cleanup_called
            cleanup_called = True
        
        async def long_running_task():
            await asyncio.sleep(10)
        
        managed_task = ManagedTask(
            "test_task", 
            long_running_task(),
            cleanup_on_cancel=cleanup_callback
        )
        
        # Give task time to start
        await asyncio.sleep(0.01)
        
        success = await managed_task.cancel(timeout=1.0)
        
        self.assertTrue(success)
        self.assertTrue(cleanup_called)
        self.assertEqual(managed_task.state, TaskState.COMPLETED)
        self.assertTrue(managed_task.task.cancelled())
    
    @pytest.mark.asyncio
    async def test_async_cleanup_on_cancel(self):
        """Test async cleanup on task cancellation."""
        cleanup_called = False
        
        async def async_cleanup():
            nonlocal cleanup_called
            cleanup_called = True
            await asyncio.sleep(0.01)
        
        async def long_running_task():
            await asyncio.sleep(10)
        
        managed_task = ManagedTask(
            "test_task",
            long_running_task(),
            cleanup_on_cancel=async_cleanup
        )
        
        # Give task time to start
        await asyncio.sleep(0.01)
        
        success = await managed_task.cancel(timeout=1.0)
        
        self.assertTrue(success)
        self.assertTrue(cleanup_called)
    
    @pytest.mark.asyncio
    async def test_task_info_generation(self):
        """Test task info generation for monitoring."""
        async def dummy_task():
            await asyncio.sleep(0.01)
            return "done"
        
        managed_task = ManagedTask("test_task", dummy_task(), timeout=10.0)
        
        # Wait a bit for task to complete
        await asyncio.sleep(0.02)
        
        info = managed_task.get_info()
        
        self.assertEqual(info['task_id'], "test_task")
        self.assertEqual(info['timeout'], 10.0)
        self.assertIn('created_at', info)
        self.assertIn('state', info)
        self.assertIn('done', info)
        self.assertIn('cancelled', info)


class TestResourceManager(unittest.TestCase):
    """Test ResourceManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.resource_manager = ResourceManager(default_resource_timeout=2.0)
    
    def test_resource_manager_initialization(self):
        """Test resource manager initialization."""
        self.assertEqual(self.resource_manager.default_resource_timeout, 2.0)
        self.assertEqual(len(self.resource_manager.resources), 0)
        self.assertEqual(len(self.resource_manager.tasks), 0)
        self.assertEqual(len(self.resource_manager.cleanup_hooks), 0)
    
    def test_resource_registration(self):
        """Test resource registration."""
        mock_resource = Mock()
        cleanup_func = Mock()
        
        managed = self.resource_manager.register_resource(
            "test_resource",
            mock_resource,
            cleanup_func=cleanup_func,
            timeout=5.0
        )
        
        self.assertIsInstance(managed, ManagedResource)
        self.assertEqual(managed.resource_id, "test_resource")
        self.assertEqual(managed.resource, mock_resource)
        self.assertEqual(managed.timeout, 5.0)
        
        # Should be in registry
        self.assertIn("test_resource", self.resource_manager.resources)
    
    def test_resource_retrieval(self):
        """Test resource retrieval."""
        mock_resource = Mock()
        
        self.resource_manager.register_resource("test_resource", mock_resource)
        
        retrieved = self.resource_manager.get_resource("test_resource")
        self.assertEqual(retrieved, mock_resource)
        
        # Non-existent resource
        self.assertIsNone(self.resource_manager.get_resource("nonexistent"))
    
    @pytest.mark.asyncio
    async def test_task_creation(self):
        """Test managed task creation."""
        async def dummy_task():
            await asyncio.sleep(0.01)
            return "result"
        
        managed_task = self.resource_manager.create_task(
            "test_task",
            dummy_task(),
            timeout=5.0
        )
        
        self.assertIsInstance(managed_task, ManagedTask)
        self.assertEqual(managed_task.task_id, "test_task")
        self.assertEqual(managed_task.timeout, 5.0)
        
        # Should be in registry
        self.assertIn("test_task", self.resource_manager.tasks)
        
        # Clean up task
        await managed_task.cancel()
    
    @pytest.mark.asyncio
    async def test_duplicate_task_handling(self):
        """Test handling of duplicate task IDs."""
        async def dummy_task():
            await asyncio.sleep(10)
        
        # Create first task
        task1 = self.resource_manager.create_task("duplicate_task", dummy_task())
        
        # Give first task time to start
        await asyncio.sleep(0.01)
        
        # Create second task with same ID (should cancel first)
        task2 = self.resource_manager.create_task("duplicate_task", dummy_task())
        
        self.assertNotEqual(task1.task, task2.task)
        self.assertEqual(len(self.resource_manager.tasks), 1)
        
        # Clean up
        await task2.cancel()
    
    @pytest.mark.asyncio
    async def test_resource_cleanup(self):
        """Test individual resource cleanup."""
        mock_resource = Mock()
        cleanup_func = AsyncMock()
        
        self.resource_manager.register_resource(
            "test_resource",
            mock_resource,
            cleanup_func=cleanup_func
        )
        
        success = await self.resource_manager.cleanup_resource("test_resource")
        
        self.assertTrue(success)
        cleanup_func.assert_called_once_with(mock_resource)
        self.assertNotIn("test_resource", self.resource_manager.resources)
    
    @pytest.mark.asyncio
    async def test_task_cancellation(self):
        """Test individual task cancellation."""
        async def long_task():
            await asyncio.sleep(10)
        
        self.resource_manager.create_task("test_task", long_task())
        
        # Give task time to start
        await asyncio.sleep(0.01)
        
        success = await self.resource_manager.cancel_task("test_task", timeout=1.0)
        
        self.assertTrue(success)
        self.assertNotIn("test_task", self.resource_manager.tasks)
    
    @pytest.mark.asyncio
    async def test_cleanup_all_resources_and_tasks(self):
        """Test cleanup of all resources and tasks."""
        # Register resources
        resource1 = Mock()
        resource2 = Mock()
        cleanup1 = AsyncMock()
        cleanup2 = AsyncMock()
        
        self.resource_manager.register_resource("resource1", resource1, cleanup1)
        self.resource_manager.register_resource("resource2", resource2, cleanup2)
        
        # Create tasks
        async def long_task():
            await asyncio.sleep(10)
        
        self.resource_manager.create_task("task1", long_task())
        self.resource_manager.create_task("task2", long_task())
        
        # Give tasks time to start
        await asyncio.sleep(0.01)
        
        results = await self.resource_manager.cleanup_all(timeout_per_operation=2.0)
        
        # All operations should succeed
        self.assertTrue(all(results.values()))
        
        # All registries should be empty
        self.assertEqual(len(self.resource_manager.resources), 0)
        self.assertEqual(len(self.resource_manager.tasks), 0)
        
        # Cleanup functions should be called
        cleanup1.assert_called_once()
        cleanup2.assert_called_once()
    
    def test_cleanup_hook_management(self):
        """Test cleanup hook management."""
        hook1 = Mock()
        hook2 = Mock()
        
        self.resource_manager.add_cleanup_hook(hook1)
        self.resource_manager.add_cleanup_hook(hook2)
        
        self.assertEqual(len(self.resource_manager.cleanup_hooks), 2)
    
    @pytest.mark.asyncio
    async def test_cleanup_hooks_execution(self):
        """Test cleanup hooks execution during cleanup_all."""
        sync_hook = Mock()
        async_hook = AsyncMock()
        
        self.resource_manager.add_cleanup_hook(sync_hook)
        self.resource_manager.add_cleanup_hook(async_hook)
        
        await self.resource_manager.cleanup_all()
        
        sync_hook.assert_called_once()
        async_hook.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stale_resource_detection(self):
        """Test stale resource detection and cleanup."""
        # Create fresh resource
        fresh_resource = Mock()
        self.resource_manager.register_resource("fresh", fresh_resource)
        
        # Create stale resource
        stale_resource = Mock()
        managed_stale = self.resource_manager.register_resource("stale", stale_resource)
        
        # Make it appear stale
        managed_stale.last_access = datetime.now() - timedelta(minutes=45)
        
        stale_count = self.resource_manager.cleanup_stale_resources(max_age_minutes=30)
        
        self.assertEqual(stale_count, 1)
        # Give time for async cleanup to potentially complete
        await asyncio.sleep(0.01)
    
    @pytest.mark.asyncio
    async def test_status_reporting(self):
        """Test comprehensive status reporting."""
        # Add some resources and tasks
        self.resource_manager.register_resource("resource1", Mock())
        
        async def dummy_task():
            await asyncio.sleep(0.01)
        
        task = self.resource_manager.create_task("task1", dummy_task())
        self.resource_manager.add_cleanup_hook(Mock())
        
        status = self.resource_manager.get_status()
        
        self.assertIn('manager_id', status)
        self.assertEqual(status['resources']['count'], 1)
        self.assertEqual(status['tasks']['count'], 1)
        self.assertEqual(status['cleanup_hooks'], 1)
        self.assertIn('by_state', status['resources'])
        self.assertIn('by_state', status['tasks'])
        self.assertIn('details', status['resources'])
        self.assertIn('details', status['tasks'])
        
        # Clean up
        await task.cancel()
    
    @pytest.mark.asyncio
    async def test_managed_lifecycle_context(self):
        """Test managed lifecycle context manager."""
        cleanup_executed = False
        
        async def test_lifecycle():
            nonlocal cleanup_executed
            
            async with self.resource_manager.managed_lifecycle() as manager:
                # Register a resource
                manager.register_resource("test", Mock(), cleanup_func=lambda r: setattr(self, 'cleanup_executed', True))
            
            # After context exit, resources should be cleaned up
            cleanup_executed = True
        
        await test_lifecycle()
        # Resources should be cleaned up automatically


class TestAudioProcessorResourceIntegration(unittest.TestCase):
    """Test AudioProcessor integration with resource management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = AudioProcessor(
            transcription_provider='mock_transcription',
            capture_provider='mock_capture',
            error_handler_config={'resource_timeout': 3.0}
        )
    
    def test_processor_resource_manager_initialization(self):
        """Test that processor initializes resource manager."""
        self.assertIsNotNone(self.processor.resource_manager)
        self.assertEqual(self.processor.resource_manager.default_resource_timeout, 3.0)
    
    def test_pipeline_health_includes_resource_status(self):
        """Test that pipeline health includes resource manager status."""
        health = self.processor.get_pipeline_health()
        
        self.assertIn('resource_manager', health)
        self.assertIn('manager_id', health['resource_manager'])
        self.assertIn('resources', health['resource_manager'])
        self.assertIn('tasks', health['resource_manager'])
    
    def test_session_export_includes_resource_summary(self):
        """Test that session export includes resource summary."""
        export_data = self.processor.export_session()
        
        self.assertIn('resource_summary', export_data)
        self.assertIn('manager_id', export_data['resource_summary'])


if __name__ == '__main__':
    # Set up logging to reduce noise during tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)