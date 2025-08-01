"""Base class for async tests with proper event loop and resource management.

This module provides a standardized base class for testing async operations,
ensuring proper event loop handling, resource cleanup, and timeout management.
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import List, Any, Callable, Optional
from contextlib import asynccontextmanager

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from .base_test import BaseTest
from tests.fixtures.async_mocks import (
    AsyncIteratorMock, 
    AsyncContextManagerMock,
    AsyncProviderMock
)


class BaseAsyncTest(BaseTest):
    """Base class for async tests with event loop management."""
    
    def setup_method(self):
        """Setup for async tests."""
        super().setup_method()
        
        # Event loop management
        self.event_loop = None
        self.async_tasks = []
        self.async_resources = []
        
        # Async test configuration
        self.async_timeout = 5.0
        self.cleanup_timeout = 2.0
        
    def teardown_method(self):
        """Cleanup for async tests."""
        # Cancel all running tasks
        self.cleanup_async_tasks()
        
        # Cleanup async resources
        self.cleanup_async_resources()
        
        super().teardown_method()
    
    def cleanup_async_tasks(self):
        """Cancel and cleanup all async tasks created during test."""
        if not self.async_tasks:
            return
            
        # Get current event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop available, tasks likely already cleaned up
            return
        
        # Cancel all tasks
        for task in self.async_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for cancellation with timeout
        if self.async_tasks:
            try:
                loop.run_until_complete(
                    asyncio.wait_for(
                        asyncio.gather(*self.async_tasks, return_exceptions=True),
                        timeout=self.cleanup_timeout
                    )
                )
            except asyncio.TimeoutError:
                # Force cleanup if tasks don't cancel gracefully
                for task in self.async_tasks:
                    if not task.done():
                        task.cancel()
        
        self.async_tasks.clear()
    
    def cleanup_async_resources(self):
        """Cleanup async resources like streams and connections."""
        for resource in self.async_resources:
            try:
                if hasattr(resource, 'close') and callable(resource.close):
                    if asyncio.iscoroutinefunction(resource.close):
                        # Async cleanup
                        loop = asyncio.get_event_loop()
                        loop.run_until_complete(resource.close())
                    else:
                        # Sync cleanup
                        resource.close()
            except Exception as e:
                # Log but don't fail test on cleanup errors
                import logging
                logging.warning(f"Failed to cleanup async resource {resource}: {e}")
        
        self.async_resources.clear()
    
    def create_task(self, coro, *, name: str = None) -> asyncio.Task:
        """Create and track an async task for automatic cleanup."""
        task = asyncio.create_task(coro, name=name)
        self.async_tasks.append(task)
        return task
    
    def register_async_resource(self, resource: Any):
        """Register async resource for automatic cleanup."""
        self.async_resources.append(resource)
        return resource
    
    async def wait_for_async_condition(
        self, 
        condition_coro: Callable[[], Any], 
        timeout: float = None,
        interval: float = 0.01
    ) -> Any:
        """Wait for async condition to become true/return truthy value."""
        timeout = timeout or self.async_timeout
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            result = await condition_coro()
            if result:
                return result
            await asyncio.sleep(interval)
        
        raise asyncio.TimeoutError(f"Async condition not met within {timeout}s")
    
    async def assert_async_state_transition(
        self,
        obj: Any,
        attr_name: str,
        expected_sequence: List[Any],
        timeout: float = None
    ):
        """Assert that object attribute transitions through expected sequence asynchronously."""
        timeout = timeout or self.async_timeout
        observed_sequence = []
        start_time = asyncio.get_event_loop().time()
        
        while (len(observed_sequence) < len(expected_sequence) and 
               asyncio.get_event_loop().time() - start_time < timeout):
            
            current_value = getattr(obj, attr_name)
            if not observed_sequence or current_value != observed_sequence[-1]:
                observed_sequence.append(current_value)
                
            if len(observed_sequence) >= len(expected_sequence):
                break
                
            await asyncio.sleep(0.01)
        
        if observed_sequence != expected_sequence:
            raise AssertionError(
                f"Expected async state sequence {expected_sequence}, "
                f"got {observed_sequence}"
            )
    
    @asynccontextmanager
    async def async_timeout_context(self, timeout: float = None):
        """Context manager for async operations with timeout."""
        timeout = timeout or self.async_timeout
        
        try:
            yield
        except asyncio.TimeoutError:
            raise AssertionError(f"Async operation timed out after {timeout}s")
    
    async def run_with_timeout(self, coro, timeout: float = None):
        """Run coroutine with timeout."""
        timeout = timeout or self.async_timeout
        
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise AssertionError(f"Async operation timed out after {timeout}s")
    
    def create_async_iterator_mock(self, items: List[Any]) -> AsyncIteratorMock:
        """Create async iterator mock for testing."""
        return AsyncIteratorMock(items)
    
    def create_async_context_manager_mock(
        self, 
        enter_result: Any = None, 
        exit_result: bool = False
    ) -> AsyncContextManagerMock:
        """Create async context manager mock for testing."""
        return AsyncContextManagerMock(enter_result, exit_result)
    
    def create_async_provider_mock(self, name: str = "TestProvider") -> AsyncProviderMock:
        """Create async provider mock for testing."""
        mock_provider = AsyncProviderMock(name)
        self.register_async_resource(mock_provider)
        return mock_provider


class BaseStreamTest(BaseAsyncTest):
    """Base class for testing streaming operations."""
    
    def setup_method(self):
        """Setup for stream tests."""
        super().setup_method()
        
        # Stream test configuration
        self.stream_timeout = 3.0
        self.stream_chunk_size = 1024
        self.active_streams = []
    
    def teardown_method(self):
        """Cleanup for stream tests."""
        # Close all active streams
        self.cleanup_streams()
        super().teardown_method()
    
    def cleanup_streams(self):
        """Close all active streams."""
        for stream in self.active_streams:
            try:
                if hasattr(stream, 'close'):
                    if asyncio.iscoroutinefunction(stream.close):
                        loop = asyncio.get_event_loop()
                        loop.run_until_complete(stream.close())
                    else:
                        stream.close()
            except Exception as e:
                import logging
                logging.warning(f"Failed to close stream {stream}: {e}")
        
        self.active_streams.clear()
    
    def register_stream(self, stream: Any) -> Any:
        """Register stream for automatic cleanup."""
        self.active_streams.append(stream)
        return stream
    
    async def consume_async_stream(
        self, 
        async_iterator, 
        max_items: int = 10,
        timeout: float = None
    ) -> List[Any]:
        """Consume items from async iterator with limits."""
        timeout = timeout or self.stream_timeout
        items = []
        
        start_time = asyncio.get_event_loop().time()
        
        async for item in async_iterator:
            items.append(item)
            
            if len(items) >= max_items:
                break
                
            if asyncio.get_event_loop().time() - start_time > timeout:
                break
        
        return items
    
    async def assert_stream_produces_items(
        self,
        async_iterator,
        expected_count: int,
        timeout: float = None
    ):
        """Assert that stream produces expected number of items."""
        items = await self.consume_async_stream(
            async_iterator, 
            max_items=expected_count + 1,  # Allow one extra to check for overproduction
            timeout=timeout
        )
        
        if len(items) != expected_count:
            raise AssertionError(
                f"Stream produced {len(items)} items, expected {expected_count}"
            )
    
    async def assert_stream_empty(self, async_iterator, timeout: float = 0.5):
        """Assert that stream produces no items within timeout."""
        items = await self.consume_async_stream(
            async_iterator,
            max_items=1,
            timeout=timeout
        )
        
        if items:
            raise AssertionError(f"Stream expected to be empty, but produced: {items}")


class BaseConcurrencyTest(BaseAsyncTest):
    """Base class for testing concurrent operations."""
    
    def setup_method(self):
        """Setup for concurrency tests."""
        super().setup_method()
        
        # Concurrency test configuration
        self.max_concurrent_operations = 10
        self.concurrent_timeout = 10.0
        self.operation_counters = {}
    
    async def run_concurrent_operations(
        self,
        operation_coro: Callable,
        count: int,
        *args,
        **kwargs
    ) -> List[Any]:
        """Run multiple concurrent operations."""
        if count > self.max_concurrent_operations:
            raise ValueError(f"Too many concurrent operations: {count}")
        
        tasks = []
        for i in range(count):
            coro = operation_coro(*args, **kwargs)
            task = self.create_task(coro, name=f"concurrent_op_{i}")
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def assert_concurrent_operation_success(
        self,
        operation_coro: Callable,
        count: int,
        *args,
        **kwargs
    ):
        """Assert that concurrent operations all succeed."""
        results = await self.run_concurrent_operations(
            operation_coro, count, *args, **kwargs
        )
        
        exceptions = [r for r in results if isinstance(r, Exception)]
        if exceptions:
            raise AssertionError(
                f"Concurrent operations failed: {exceptions}"
            )
    
    def increment_operation_counter(self, operation_name: str):
        """Increment counter for operation tracking."""
        self.operation_counters[operation_name] = self.operation_counters.get(operation_name, 0) + 1
    
    def assert_operation_count(self, operation_name: str, expected_count: int):
        """Assert that operation was called expected number of times."""
        actual_count = self.operation_counters.get(operation_name, 0)
        if actual_count != expected_count:
            raise AssertionError(
                f"Operation '{operation_name}' called {actual_count} times, "
                f"expected {expected_count}"
            )