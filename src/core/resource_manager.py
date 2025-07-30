"""Resource management and task lifecycle controller for audio processing pipeline."""

import asyncio
import logging
import weakref
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from ..utils.exceptions import ResourceCleanupError, PipelineError

logger = logging.getLogger(__name__)


class ResourceState(Enum):
    """Resource lifecycle states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class TaskState(Enum):
    """Task lifecycle states."""
    PENDING = "pending"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"


class ManagedResource:
    """
    Wrapper for managing resource lifecycle and cleanup.
    
    Tracks resource state, provides cleanup capabilities,
    and ensures proper resource disposal.
    """
    
    def __init__(self, 
                 resource_id: str, 
                 resource: Any,
                 cleanup_func: Optional[Callable] = None,
                 timeout: float = 5.0):
        """
        Initialize managed resource.
        
        Args:
            resource_id: Unique identifier for the resource
            resource: The actual resource object
            cleanup_func: Optional cleanup function (async or sync)
            timeout: Timeout for cleanup operations
        """
        self.resource_id = resource_id
        self.resource = resource
        self.cleanup_func = cleanup_func
        self.timeout = timeout
        self.state = ResourceState.ACTIVE
        self.created_at = datetime.now()
        self.last_access = datetime.now()
        self.cleanup_attempts = 0
        self.max_cleanup_attempts = 3
        
        logger.debug(f"🏗️ Resource: Created managed resource '{resource_id}'")
    
    def access(self):
        """Mark resource as accessed (for tracking)."""
        self.last_access = datetime.now()
        return self.resource
    
    async def cleanup(self) -> bool:
        """
        Clean up the resource safely.
        
        Returns:
            True if cleanup successful, False otherwise
        """
        if self.state in [ResourceState.STOPPED, ResourceState.STOPPING]:
            logger.debug(f"🧹 Resource: '{self.resource_id}' already stopped/stopping")
            return True
        
        self.state = ResourceState.STOPPING
        self.cleanup_attempts += 1
        
        logger.info(f"🧹 Resource: Cleaning up '{self.resource_id}' (attempt {self.cleanup_attempts})")
        
        try:
            if self.cleanup_func:
                if asyncio.iscoroutinefunction(self.cleanup_func):
                    await asyncio.wait_for(self.cleanup_func(self.resource), timeout=self.timeout)
                else:
                    self.cleanup_func(self.resource)
            
            self.state = ResourceState.STOPPED
            logger.info(f"✅ Resource: Successfully cleaned up '{self.resource_id}'")
            return True
            
        except asyncio.TimeoutError:
            self.state = ResourceState.ERROR
            logger.error(f"⏱️ Resource: Cleanup timeout for '{self.resource_id}' after {self.timeout}s")
            return False
            
        except Exception as e:
            self.state = ResourceState.ERROR
            logger.error(f"❌ Resource: Cleanup failed for '{self.resource_id}': {e}")
            return False
    
    def is_stale(self, max_age_minutes: int = 30) -> bool:
        """Check if resource is stale (unused for too long)."""
        age = datetime.now() - self.last_access
        return age > timedelta(minutes=max_age_minutes)
    
    def get_info(self) -> Dict[str, Any]:
        """Get resource information for monitoring."""
        return {
            'resource_id': self.resource_id,
            'state': self.state.value,
            'type': type(self.resource).__name__,
            'created_at': self.created_at.isoformat(),
            'last_access': self.last_access.isoformat(),
            'age_seconds': (datetime.now() - self.created_at).total_seconds(),
            'cleanup_attempts': self.cleanup_attempts
        }


class ManagedTask:
    """
    Wrapper for managing asyncio task lifecycle.
    
    Provides controlled task execution, cancellation,
    and monitoring capabilities.
    """
    
    def __init__(self,
                 task_id: str,
                 coro,
                 timeout: Optional[float] = None,
                 cleanup_on_cancel: Optional[Callable] = None):
        """
        Initialize managed task.
        
        Args:
            task_id: Unique identifier for the task
            coro: Coroutine to execute
            timeout: Optional task timeout
            cleanup_on_cancel: Optional cleanup function when cancelled
        """
        self.task_id = task_id
        self.timeout = timeout
        self.cleanup_on_cancel = cleanup_on_cancel
        self.state = TaskState.PENDING
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error: Optional[Exception] = None
        
        # Create the actual asyncio task
        self.task: asyncio.Task = asyncio.create_task(self._execute_with_monitoring(coro))
        
        logger.debug(f"🚀 Task: Created managed task '{task_id}'")
    
    async def _execute_with_monitoring(self, coro):
        """Execute coroutine with monitoring and timeout."""
        try:
            self.state = TaskState.RUNNING
            self.started_at = datetime.now()
            
            logger.info(f"🏃 Task: Starting execution of '{self.task_id}'")
            
            if self.timeout:
                result = await asyncio.wait_for(coro, timeout=self.timeout)
            else:
                result = await coro
            
            self.state = TaskState.COMPLETED
            self.completed_at = datetime.now()
            
            duration = (self.completed_at - self.started_at).total_seconds()
            logger.info(f"✅ Task: '{self.task_id}' completed successfully in {duration:.2f}s")
            
            return result
            
        except asyncio.CancelledError:
            self.state = TaskState.COMPLETED  # Cancelled is a form of completion
            self.completed_at = datetime.now()
            
            logger.info(f"🛑 Task: '{self.task_id}' was cancelled")
            
            # Execute cleanup if provided
            if self.cleanup_on_cancel:
                try:
                    if asyncio.iscoroutinefunction(self.cleanup_on_cancel):
                        await self.cleanup_on_cancel()
                    else:
                        self.cleanup_on_cancel()
                    logger.info(f"🧹 Task: Cleanup completed for cancelled task '{self.task_id}'")
                except Exception as e:
                    logger.error(f"❌ Task: Cleanup failed for cancelled task '{self.task_id}': {e}")
            
            raise
            
        except asyncio.TimeoutError as e:
            self.state = TaskState.FAILED
            self.completed_at = datetime.now()
            self.error = e
            
            logger.error(f"⏱️ Task: '{self.task_id}' timed out after {self.timeout}s")
            raise
            
        except Exception as e:
            self.state = TaskState.FAILED
            self.completed_at = datetime.now()
            self.error = e
            
            logger.error(f"❌ Task: '{self.task_id}' failed with error: {e}")
            raise
    
    async def cancel(self, timeout: float = 2.0) -> bool:
        """
        Cancel the task gracefully.
        
        Args:
            timeout: How long to wait for cancellation
            
        Returns:
            True if task was cancelled successfully
        """
        if self.state in [TaskState.COMPLETED, TaskState.FAILED]:
            logger.debug(f"🛑 Task: '{self.task_id}' already completed/failed")
            return True
        
        self.state = TaskState.CANCELLING
        logger.info(f"🛑 Task: Cancelling '{self.task_id}'")
        
        self.task.cancel()
        
        try:
            await asyncio.wait_for(self.task, timeout=timeout)
            logger.info(f"✅ Task: '{self.task_id}' cancelled successfully")
            return True
        except asyncio.TimeoutError:
            logger.error(f"⏱️ Task: Cancellation timeout for '{self.task_id}' after {timeout}s")
            return False
        except asyncio.CancelledError:
            logger.info(f"✅ Task: '{self.task_id}' cancelled successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Task: Error during cancellation of '{self.task_id}': {e}")
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get task information for monitoring."""
        info = {
            'task_id': self.task_id,
            'state': self.state.value,
            'created_at': self.created_at.isoformat(),
            'timeout': self.timeout,
            'done': self.task.done(),
            'cancelled': self.task.cancelled()
        }
        
        if self.started_at:
            info['started_at'] = self.started_at.isoformat()
            
        if self.completed_at:
            info['completed_at'] = self.completed_at.isoformat()
            info['duration_seconds'] = (self.completed_at - (self.started_at or self.created_at)).total_seconds()
            
        if self.error:
            info['error'] = str(self.error)
            info['error_type'] = type(self.error).__name__
        
        return info


class ResourceManager:
    """
    Centralized resource management and task lifecycle controller.
    
    Manages resources, tasks, and their cleanup in a coordinated manner
    to ensure proper resource disposal and prevent resource leaks.
    """
    
    def __init__(self, default_resource_timeout: float = 5.0):
        """
        Initialize resource manager.
        
        Args:
            default_resource_timeout: Default timeout for resource cleanup
        """
        self.default_resource_timeout = default_resource_timeout
        self.resources: Dict[str, ManagedResource] = {}
        self.tasks: Dict[str, ManagedTask] = {}
        self.cleanup_hooks: List[Callable] = []
        self._manager_id = id(self)
        
        logger.info(f"🏗️ ResourceManager: Initialized manager {self._manager_id}")
    
    def register_resource(self,
                         resource_id: str,
                         resource: Any,
                         cleanup_func: Optional[Callable] = None,
                         timeout: Optional[float] = None) -> ManagedResource:
        """
        Register a resource for management.
        
        Args:
            resource_id: Unique identifier for the resource
            resource: The resource object to manage
            cleanup_func: Optional cleanup function
            timeout: Resource cleanup timeout
            
        Returns:
            ManagedResource wrapper
        """
        if resource_id in self.resources:
            logger.warning(f"⚠️ ResourceManager: Resource '{resource_id}' already registered, replacing")
        
        managed = ManagedResource(
            resource_id=resource_id,
            resource=resource,
            cleanup_func=cleanup_func,
            timeout=timeout or self.default_resource_timeout
        )
        
        self.resources[resource_id] = managed
        logger.info(f"📋 ResourceManager: Registered resource '{resource_id}'")
        
        return managed
    
    def get_resource(self, resource_id: str) -> Optional[Any]:
        """
        Get a managed resource by ID.
        
        Args:
            resource_id: Resource identifier
            
        Returns:
            The resource object if found, None otherwise
        """
        if resource_id in self.resources:
            return self.resources[resource_id].access()
        return None
    
    def create_task(self,
                   task_id: str,
                   coro,
                   timeout: Optional[float] = None,
                   cleanup_on_cancel: Optional[Callable] = None) -> ManagedTask:
        """
        Create and register a managed task.
        
        Args:
            task_id: Unique identifier for the task
            coro: Coroutine to execute
            timeout: Optional task timeout
            cleanup_on_cancel: Optional cleanup function on cancellation
            
        Returns:
            ManagedTask wrapper
        """
        if task_id in self.tasks:
            logger.warning(f"⚠️ ResourceManager: Task '{task_id}' already exists, cancelling old task")
            # Cancel old task
            asyncio.create_task(self.tasks[task_id].cancel())
        
        managed_task = ManagedTask(
            task_id=task_id,
            coro=coro,
            timeout=timeout,
            cleanup_on_cancel=cleanup_on_cancel
        )
        
        self.tasks[task_id] = managed_task
        logger.info(f"📋 ResourceManager: Created managed task '{task_id}'")
        
        return managed_task
    
    async def cleanup_resource(self, resource_id: str) -> bool:
        """
        Clean up a specific resource.
        
        Args:
            resource_id: Resource to clean up
            
        Returns:
            True if cleanup successful
        """
        if resource_id not in self.resources:
            logger.warning(f"⚠️ ResourceManager: Resource '{resource_id}' not found for cleanup")
            return True
        
        success = await self.resources[resource_id].cleanup()
        
        if success:
            del self.resources[resource_id]
            logger.info(f"🗑️ ResourceManager: Removed resource '{resource_id}' from registry")
        
        return success
    
    async def cancel_task(self, task_id: str, timeout: float = 2.0) -> bool:
        """
        Cancel a specific task.
        
        Args:
            task_id: Task to cancel
            timeout: Cancellation timeout
            
        Returns:
            True if cancellation successful
        """
        if task_id not in self.tasks:
            logger.warning(f"⚠️ ResourceManager: Task '{task_id}' not found for cancellation")
            return True
        
        success = await self.tasks[task_id].cancel(timeout)
        
        # Always remove from registry after cancellation attempt
        del self.tasks[task_id]
        logger.info(f"🗑️ ResourceManager: Removed task '{task_id}' from registry")
        
        return success
    
    async def cleanup_all(self, timeout_per_operation: float = 3.0) -> Dict[str, bool]:
        """
        Clean up all managed resources and tasks.
        
        Args:
            timeout_per_operation: Timeout for each cleanup operation
            
        Returns:
            Dict of operation_id -> success_status
        """
        results = {}
        
        # Cancel all tasks first
        logger.info(f"🛑 ResourceManager: Cancelling {len(self.tasks)} tasks...")
        task_cancellations = []
        
        for task_id in list(self.tasks.keys()):
            task_cancellations.append(self.cancel_task(task_id, timeout_per_operation))
        
        if task_cancellations:
            task_results = await asyncio.gather(*task_cancellations, return_exceptions=True)
            
            for i, (task_id, result) in enumerate(zip(list(self.tasks.keys()), task_results)):
                if isinstance(result, Exception):
                    results[f"task_{task_id}"] = False
                    logger.error(f"❌ ResourceManager: Task cancellation failed for '{task_id}': {result}")
                else:
                    results[f"task_{task_id}"] = result
        
        # Clean up all resources
        logger.info(f"🧹 ResourceManager: Cleaning up {len(self.resources)} resources...")
        resource_cleanups = []
        
        for resource_id in list(self.resources.keys()):
            resource_cleanups.append(self.cleanup_resource(resource_id))
        
        if resource_cleanups:
            resource_results = await asyncio.gather(*resource_cleanups, return_exceptions=True)
            
            for i, (resource_id, result) in enumerate(zip(list(self.resources.keys()), resource_results)):
                if isinstance(result, Exception):
                    results[f"resource_{resource_id}"] = False
                    logger.error(f"❌ ResourceManager: Resource cleanup failed for '{resource_id}': {result}")
                else:
                    results[f"resource_{resource_id}"] = result
        
        # Execute cleanup hooks
        for hook in self.cleanup_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await asyncio.wait_for(hook(), timeout=timeout_per_operation)
                else:
                    hook()
                results[f"hook_{id(hook)}"] = True
            except Exception as e:
                logger.error(f"❌ ResourceManager: Cleanup hook failed: {e}")
                results[f"hook_{id(hook)}"] = False
        
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        logger.info(f"🧹 ResourceManager: Cleanup completed - {successful}/{total} operations successful")
        
        return results
    
    def cleanup_stale_resources(self, max_age_minutes: int = 30) -> int:
        """
        Clean up stale (unused) resources.
        
        Args:
            max_age_minutes: Maximum age before considering resource stale
            
        Returns:
            Number of stale resources found
        """
        stale_resources = [
            resource_id for resource_id, resource in self.resources.items()
            if resource.is_stale(max_age_minutes)
        ]
        
        if stale_resources:
            logger.info(f"🕰️ ResourceManager: Found {len(stale_resources)} stale resources")
            # Schedule cleanup (don't await here as this might be called from sync context)
            for resource_id in stale_resources:
                asyncio.create_task(self.cleanup_resource(resource_id))
        
        return len(stale_resources)
    
    def add_cleanup_hook(self, hook: Callable):
        """Add a cleanup hook to be executed during shutdown."""
        self.cleanup_hooks.append(hook)
        logger.debug(f"📌 ResourceManager: Added cleanup hook {id(hook)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive resource manager status."""
        return {
            'manager_id': self._manager_id,
            'resources': {
                'count': len(self.resources),
                'by_state': self._count_by_state(self.resources, lambda r: r.state),
                'details': [resource.get_info() for resource in self.resources.values()]
            },
            'tasks': {
                'count': len(self.tasks),
                'by_state': self._count_by_state(self.tasks, lambda t: t.state),
                'details': [task.get_info() for task in self.tasks.values()]
            },
            'cleanup_hooks': len(self.cleanup_hooks)
        }
    
    def _count_by_state(self, items: Dict, state_getter: Callable) -> Dict[str, int]:
        """Count items by their state."""
        counts = {}
        for item in items.values():
            state = state_getter(item).value
            counts[state] = counts.get(state, 0) + 1
        return counts
    
    @asynccontextmanager
    async def managed_lifecycle(self):
        """Context manager for automatic cleanup on exit."""
        try:
            logger.info(f"🚀 ResourceManager: Starting managed lifecycle")
            yield self
        finally:
            logger.info(f"🛑 ResourceManager: Ending managed lifecycle, cleaning up...")
            await self.cleanup_all()
            logger.info(f"✅ ResourceManager: Managed lifecycle cleanup completed")