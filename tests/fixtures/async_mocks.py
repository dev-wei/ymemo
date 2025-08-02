"""Specialized async mock utilities for testing async operations.

This module provides mock classes and utilities specifically designed for testing
async operations, generators, and context managers.
"""

import asyncio
from typing import Any, List
from unittest.mock import AsyncMock


class AsyncIteratorMock:
    """Mock for async iterators (async generators)."""

    def __init__(self, items: List[Any]):
        """Initialize with items to yield."""
        self.items = items
        self.index = 0

    def __aiter__(self):
        """Return self as async iterator."""
        return self

    async def __anext__(self):
        """Return next item or raise StopAsyncIteration."""
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


class AsyncContextManagerMock:
    """Mock for async context managers."""

    def __init__(self, enter_result: Any = None, exit_result: bool = False):
        """Initialize with enter and exit behavior."""
        self.enter_result = enter_result
        self.exit_result = exit_result
        self.entered = False
        self.exited = False
        self.exception_info = None

    async def __aenter__(self):
        """Async enter method."""
        self.entered = True
        return self.enter_result

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async exit method."""
        self.exited = True
        self.exception_info = (exc_type, exc_val, exc_tb)
        return self.exit_result


class AsyncCallbackMock:
    """Mock for testing async callbacks."""

    def __init__(self):
        """Initialize callback tracking."""
        self.calls = []
        self.call_count = 0

    async def __call__(self, *args, **kwargs):
        """Record async callback call."""
        self.call_count += 1
        self.calls.append((args, kwargs))
        return None

    def assert_called_once(self):
        """Assert callback was called exactly once."""
        if self.call_count != 1:
            raise AssertionError(f"Expected 1 call, got {self.call_count}")

    def assert_called_with(self, *args, **kwargs):
        """Assert callback was called with specific arguments."""
        if not self.calls:
            raise AssertionError("Callback was not called")
        last_call = self.calls[-1]
        if last_call != (args, kwargs):
            raise AssertionError(f"Expected call {(args, kwargs)}, got {last_call}")


class MockAsyncGenerator:
    """Factory for creating async generator mocks."""

    @staticmethod
    def create_audio_stream(
        chunk_count: int = 5, chunk_size: int = 1024
    ) -> AsyncIteratorMock:
        """Create mock audio stream generator."""
        chunks = [b"\x00" * chunk_size for _ in range(chunk_count)]
        return AsyncIteratorMock(chunks)

    @staticmethod
    def create_transcription_stream(results: List[Any]) -> AsyncIteratorMock:
        """Create mock transcription result stream."""
        return AsyncIteratorMock(results)

    @staticmethod
    def create_empty_stream() -> AsyncIteratorMock:
        """Create empty async stream."""
        return AsyncIteratorMock([])

    @staticmethod
    def create_infinite_stream(item: Any, delay: float = 0.01):
        """Create infinite async stream (for stress testing)."""

        class InfiniteAsyncIterator:
            def __init__(self, item, delay):
                self.item = item
                self.delay = delay

            def __aiter__(self):
                return self

            async def __anext__(self):
                await asyncio.sleep(self.delay)
                return self.item

        return InfiniteAsyncIterator(item, delay)


class AsyncProviderMock:
    """Enhanced mock for async providers with realistic behavior."""

    def __init__(self, name: str = "MockProvider"):
        """Initialize async provider mock."""
        self.name = name
        self.is_active = False
        self.start_calls = []
        self.stop_calls = []
        self.stream_data = []

    async def start_stream(self, *args, **kwargs):
        """Mock start stream method."""
        self.start_calls.append((args, kwargs))
        self.is_active = True
        return None

    async def stop_stream(self):
        """Mock stop stream method."""
        self.stop_calls.append(())
        self.is_active = False
        return None

    async def get_stream(self):
        """Mock get stream method."""
        for item in self.stream_data:
            yield item

    def set_stream_data(self, data: List[Any]):
        """Set data for stream to yield."""
        self.stream_data = data


class AsyncMockWithState:
    """AsyncMock with state tracking for complex testing scenarios."""

    def __init__(self, spec=None, side_effect=None, return_value=None):
        """Initialize with state tracking."""
        self.mock = AsyncMock(
            spec=spec, side_effect=side_effect, return_value=return_value
        )
        self.call_history = []
        self.state_history = []
        self.current_state = {}

    async def __call__(self, *args, **kwargs):
        """Track calls and state changes."""
        # Record call
        self.call_history.append(
            {
                "args": args,
                "kwargs": kwargs,
                "timestamp": asyncio.get_event_loop().time(),
                "state_before": self.current_state.copy(),
            }
        )

        # Execute mock
        result = await self.mock(*args, **kwargs)

        # Record state after
        self.call_history[-1]["state_after"] = self.current_state.copy()

        return result

    def set_state(self, key: str, value: Any):
        """Set state value."""
        self.current_state[key] = value
        self.state_history.append(
            {"key": key, "value": value, "timestamp": asyncio.get_event_loop().time()}
        )

    def get_state(self, key: str, default: Any = None):
        """Get state value."""
        return self.current_state.get(key, default)

    def assert_state_sequence(self, expected_states: List[dict]):
        """Assert that state changes happened in expected sequence."""
        if len(self.state_history) != len(expected_states):
            raise AssertionError(
                f"Expected {len(expected_states)} state changes, "
                f"got {len(self.state_history)}"
            )

        for i, (actual, expected) in enumerate(
            zip(self.state_history, expected_states)
        ):
            for key, value in expected.items():
                if key not in actual or actual[key] != value:
                    raise AssertionError(
                        f"State change {i}: expected {key}={value}, "
                        f"got {actual.get(key)}"
                    )


class TimeoutMock:
    """Mock for testing timeout scenarios."""

    def __init__(self, timeout_after: float = 1.0):
        """Initialize with timeout duration."""
        self.timeout_after = timeout_after
        self.start_time = None

    async def __call__(self, *args, **kwargs):
        """Simulate operation that times out."""
        if self.start_time is None:
            self.start_time = asyncio.get_event_loop().time()

        await asyncio.sleep(self.timeout_after + 0.1)  # Exceed timeout
        return "Should not reach here"


class ConcurrentCallMock:
    """Mock for testing concurrent call scenarios."""

    def __init__(self):
        """Initialize concurrent call tracking."""
        self.concurrent_calls = 0
        self.max_concurrent_calls = 0
        self.call_log = []

    async def __call__(self, *args, **kwargs):
        """Track concurrent calls."""
        self.concurrent_calls += 1
        self.max_concurrent_calls = max(
            self.max_concurrent_calls, self.concurrent_calls
        )

        call_info = {
            "start_time": asyncio.get_event_loop().time(),
            "args": args,
            "kwargs": kwargs,
            "concurrent_count": self.concurrent_calls,
        }
        self.call_log.append(call_info)

        try:
            # Simulate some work
            await asyncio.sleep(0.1)
            return f"Completed call {len(self.call_log)}"
        finally:
            self.concurrent_calls -= 1
            call_info["end_time"] = asyncio.get_event_loop().time()

    def assert_max_concurrent_calls(self, expected_max: int):
        """Assert maximum concurrent calls."""
        if self.max_concurrent_calls != expected_max:
            raise AssertionError(
                f"Expected max {expected_max} concurrent calls, "
                f"got {self.max_concurrent_calls}"
            )
