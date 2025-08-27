"""
Task Lifecycle Manager for T-Bot Trading System.

This module provides comprehensive task management including proper
cancellation handling, resource cleanup, and memory leak prevention.
"""

import asyncio
import logging
import time
import weakref
from collections import defaultdict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .resource_manager import ResourceType, get_resource_manager


class TaskState(Enum):
    """Task lifecycle states."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    TIMEOUT = "timeout"


class TaskPriority(Enum):
    """Task priority levels."""

    CRITICAL = "critical"  # Trading operations
    HIGH = "high"  # Risk management
    NORMAL = "normal"  # Market data processing
    LOW = "low"  # Background tasks
    CLEANUP = "cleanup"  # Resource cleanup tasks


@dataclass
class TaskInfo:
    """Information about a managed task."""

    task_id: str
    name: str
    priority: TaskPriority
    state: TaskState
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    timeout: float | None = None
    cleanup_callback: Callable[[], None] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Exception | None = None
    retries: int = 0
    max_retries: int = 0


class TaskManager:
    """
    Comprehensive task lifecycle manager.

    Features:
    - Task registration and tracking
    - Priority-based execution
    - Automatic timeout handling
    - Proper cancellation with cleanup
    - Resource leak prevention
    - Task metrics and monitoring
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Task tracking
        self._tasks: dict[str, TaskInfo] = {}
        self._task_refs: weakref.WeakValueDictionary[str, Any] = weakref.WeakValueDictionary()
        self._task_lock = asyncio.Lock()

        # Task queues by priority
        self._task_queues: dict[TaskPriority, asyncio.Queue[Any]] = {
            priority: asyncio.Queue() for priority in TaskPriority
        }

        # Worker management
        self._workers: list[asyncio.Task[Any]] = []
        self._worker_count = 5
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Configuration
        self._default_timeout = 300.0  # 5 minutes
        self._cleanup_interval = 60  # seconds
        self._max_completed_tasks = 1000  # Keep for debugging

        # Background tasks
        self._cleanup_task: asyncio.Task | None = None
        self._monitor_task: asyncio.Task | None = None

        # Statistics
        self._stats = {
            "tasks_created": 0,
            "tasks_completed": 0,
            "tasks_cancelled": 0,
            "tasks_failed": 0,
            "tasks_timeout": 0,
            "cleanup_runs": 0,
        }

        # Resource manager integration
        self._resource_manager = get_resource_manager()

    async def start(self):
        """Start the task manager."""
        if self._running:
            return

        self._running = True
        self._shutdown_event.clear()

        # Start worker tasks
        self._workers = [
            asyncio.create_task(self._worker_loop(f"worker-{i}")) for i in range(self._worker_count)
        ]

        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._monitor_task = asyncio.create_task(self._monitor_loop())

        self.logger.info(f"Task manager started with {len(self._workers)} workers")

    async def stop(self):
        """Stop the task manager and cleanup all tasks."""
        self.logger.info("Stopping task manager...")

        # Signal shutdown
        self._running = False
        self._shutdown_event.set()

        # Cancel all pending tasks
        await self._cancel_all_tasks()

        # Stop workers
        if self._workers:
            for worker in self._workers:
                if not worker.done():
                    worker.cancel()

            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True), timeout=30.0
                )
            except asyncio.TimeoutError:
                self.logger.warning("Some workers did not complete within timeout")

            self._workers.clear()

        # Stop background tasks
        for bg_task in [self._cleanup_task, self._monitor_task]:
            if bg_task and not bg_task.done():
                bg_task.cancel()
                try:
                    await asyncio.wait_for(bg_task, timeout=10.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        self.logger.info("Task manager stopped")

    async def create_task(
        self,
        coro: Coroutine,
        name: str | None = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: float | None = None,
        cleanup_callback: Callable[[], None] | None = None,
        metadata: dict[str, Any] | None = None,
        max_retries: int = 0,
    ) -> str:
        """Create and register a managed task."""
        if not self._running:
            raise RuntimeError("Task manager is not running")

        # Generate task ID
        task_id = f"{name or 'task'}_{int(time.time() * 1000)}_{id(coro)}"

        # Create task info
        task_info = TaskInfo(
            task_id=task_id,
            name=name or "unnamed_task",
            priority=priority,
            state=TaskState.CREATED,
            created_at=datetime.now(timezone.utc),
            timeout=timeout or self._default_timeout,
            cleanup_callback=cleanup_callback,
            metadata=metadata or {},
            max_retries=max_retries,
        )

        # Register with resource manager
        self._resource_manager.register_resource(
            coro,
            ResourceType.ASYNCIO_TASK,
            task_id,
            cleanup_callback=lambda: self._cleanup_task_resource(task_id),
            metadata={"name": task_info.name, "priority": priority.value},
        )

        async with self._task_lock:
            self._tasks[task_id] = task_info

            # Queue for execution
            await self._task_queues[priority].put((task_id, coro))

            self._stats["tasks_created"] += 1

        self.logger.debug(f"Created task: {task_id} ({name}) with priority {priority.value}")
        return task_id

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a specific task."""
        async with self._task_lock:
            task_info = self._tasks.get(task_id)
            if not task_info:
                return False

            task_ref = self._task_refs.get(task_id)

            if task_ref and not task_ref.done():
                task_ref.cancel()
                task_info.state = TaskState.CANCELLED
                task_info.completed_at = datetime.now(timezone.utc)
                self._stats["tasks_cancelled"] += 1

                # Execute cleanup callback
                if task_info.cleanup_callback:
                    try:
                        task_info.cleanup_callback()
                    except Exception as e:
                        self.logger.error(f"Error in cleanup callback for {task_id}: {e}")

                self.logger.debug(f"Cancelled task: {task_id}")
                return True

        return False

    async def _cancel_all_tasks(self):
        """Cancel all pending and running tasks."""
        async with self._task_lock:
            task_ids = list(self._tasks.keys())

        cancelled_count = 0
        for task_id in task_ids:
            if await self.cancel_task(task_id):
                cancelled_count += 1

        # Clear all queues
        for queue in self._task_queues.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    break

        self.logger.info(f"Cancelled {cancelled_count} tasks")

    def _cleanup_task_resource(self, task_id: str):
        """Cleanup callback for task resources."""
        # This will be called by the resource manager
        pass

    async def _worker_loop(self, worker_name: str):
        """Worker loop that processes tasks from priority queues."""
        self.logger.debug(f"Starting worker: {worker_name}")

        try:
            while self._running and not self._shutdown_event.is_set():
                task_executed = False

                # Process tasks by priority (highest first)
                for priority in [
                    TaskPriority.CRITICAL,
                    TaskPriority.HIGH,
                    TaskPriority.NORMAL,
                    TaskPriority.LOW,
                    TaskPriority.CLEANUP,
                ]:
                    queue = self._task_queues[priority]

                    try:
                        # Try to get a task with short timeout
                        task_id, coro = await asyncio.wait_for(queue.get(), timeout=0.1)

                        try:
                            await self._execute_task(task_id, coro, worker_name)
                            task_executed = True
                        finally:
                            queue.task_done()

                        break  # Move to next iteration after executing a task

                    except asyncio.TimeoutError:
                        continue  # Try next priority queue
                    except asyncio.CancelledError:
                        # Worker cancelled
                        return
                    except Exception as e:
                        self.logger.error(f"Error getting task from queue {priority}: {e}")
                        continue

                # If no task was executed, sleep briefly
                if not task_executed:
                    try:
                        await asyncio.sleep(0.1)
                    except asyncio.CancelledError:
                        break

        except asyncio.CancelledError:
            self.logger.debug(f"Worker {worker_name} cancelled")
        except Exception as e:
            self.logger.error(f"Worker {worker_name} failed: {e}")
        finally:
            self.logger.debug(f"Worker {worker_name} stopped")

    async def _execute_task(self, task_id: str, coro: Coroutine, worker_name: str):
        """Execute a single task with proper lifecycle management."""
        async with self._task_lock:
            task_info = self._tasks.get(task_id)
            if not task_info:
                self.logger.warning(f"Task {task_id} not found in registry")
                return

            task_info.state = TaskState.RUNNING
            task_info.started_at = datetime.now(timezone.utc)

        # Create and track the actual asyncio task
        actual_task = asyncio.create_task(coro)
        self._task_refs[task_id] = actual_task

        # Touch resource to indicate activity
        self._resource_manager.touch_resource(task_id)

        try:
            if task_info.timeout:
                await asyncio.wait_for(actual_task, timeout=task_info.timeout)
            else:
                await actual_task

            # Task completed successfully
            async with self._task_lock:
                task_info.state = TaskState.COMPLETED
                task_info.completed_at = datetime.now(timezone.utc)
                self._stats["tasks_completed"] += 1

            self.logger.debug(f"Worker {worker_name} completed task: {task_id}")

        except asyncio.TimeoutError:
            # Task timed out
            actual_task.cancel()
            try:
                await actual_task
            except asyncio.CancelledError:
                pass

            async with self._task_lock:
                task_info.state = TaskState.TIMEOUT
                task_info.completed_at = datetime.now(timezone.utc)
                self._stats["tasks_timeout"] += 1

            self.logger.warning(f"Task {task_id} timed out after {task_info.timeout}s")

        except asyncio.CancelledError:
            # Task was cancelled
            async with self._task_lock:
                task_info.state = TaskState.CANCELLED
                task_info.completed_at = datetime.now(timezone.utc)
                self._stats["tasks_cancelled"] += 1

            self.logger.debug(f"Task {task_id} cancelled")

        except Exception as e:
            # Task failed
            async with self._task_lock:
                task_info.state = TaskState.FAILED
                task_info.error = e
                task_info.completed_at = datetime.now(timezone.utc)
                self._stats["tasks_failed"] += 1

            # Check if we should retry
            if task_info.retries < task_info.max_retries:
                task_info.retries += 1
                task_info.state = TaskState.CREATED
                task_info.started_at = None
                task_info.completed_at = None

                # Re-queue for retry
                await self._task_queues[task_info.priority].put((task_id, coro))
                self.logger.info(f"Retrying task {task_id} (attempt {task_info.retries + 1})")
            else:
                self.logger.error(f"Task {task_id} failed: {e}")

        finally:
            # Execute cleanup callback
            if task_info.cleanup_callback:
                try:
                    task_info.cleanup_callback()
                except Exception as e:
                    self.logger.error(f"Error in cleanup callback for {task_id}: {e}")

            # Unregister from resource manager
            self._resource_manager.unregister_resource(task_id)

    async def _cleanup_loop(self):
        """Background cleanup of completed tasks."""
        try:
            while self._running and not self._shutdown_event.is_set():
                try:
                    await self._cleanup_completed_tasks()
                    self._stats["cleanup_runs"] += 1

                    await asyncio.wait_for(
                        asyncio.sleep(self._cleanup_interval), timeout=self._cleanup_interval + 1.0
                    )

                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in cleanup loop: {e}")
                    await asyncio.sleep(self._cleanup_interval)

        except asyncio.CancelledError:
            pass
        finally:
            self.logger.debug("Cleanup loop stopped")

    async def _monitor_loop(self):
        """Background monitoring and reporting."""
        try:
            while self._running and not self._shutdown_event.is_set():
                try:
                    self._log_task_stats()

                    await asyncio.wait_for(
                        asyncio.sleep(300),  # Monitor every 5 minutes
                        timeout=301.0,
                    )

                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in monitor loop: {e}")
                    await asyncio.sleep(300)

        except asyncio.CancelledError:
            pass
        finally:
            self.logger.debug("Monitor loop stopped")

    async def _cleanup_completed_tasks(self):
        """Clean up old completed task records."""
        to_remove = []

        async with self._task_lock:
            # Keep only recent completed tasks
            if len(self._tasks) > self._max_completed_tasks:
                completed_tasks = [
                    (task_id, task_info)
                    for task_id, task_info in self._tasks.items()
                    if task_info.state
                    in [
                        TaskState.COMPLETED,
                        TaskState.CANCELLED,
                        TaskState.FAILED,
                        TaskState.TIMEOUT,
                    ]
                ]

                # Sort by completion time, oldest first
                completed_tasks.sort(key=lambda x: x[1].completed_at or x[1].created_at)

                # Remove oldest tasks beyond limit
                excess_count = len(completed_tasks) - (self._max_completed_tasks // 2)
                if excess_count > 0:
                    to_remove = [task_id for task_id, _ in completed_tasks[:excess_count]]

        # Remove old tasks
        for task_id in to_remove:
            async with self._task_lock:
                self._tasks.pop(task_id, None)
                self._task_refs.pop(task_id, None)

        if to_remove:
            self.logger.debug(f"Cleaned up {len(to_remove)} old task records")

    def _log_task_stats(self):
        """Log task statistics."""
        state_counts: dict[str, int] = defaultdict(int)
        priority_counts: dict[str, int] = defaultdict(int)

        # Use snapshot to avoid async lock in sync method
        tasks_snapshot = dict(self._tasks)
        for task_info in tasks_snapshot.values():
            state_counts[task_info.state.value] += 1
            priority_counts[task_info.priority.value] += 1

        queue_sizes = {
            priority.value: queue.qsize() for priority, queue in self._task_queues.items()
        }

        self.logger.info(
            f"Task Stats - Total: {len(self._tasks)}, "
            f"States: {dict(state_counts)}, "
            f"Queued: {dict(queue_sizes)}, "
            f"Lifecycle: {self._stats}"
        )

    def get_task_stats(self) -> dict[str, Any]:
        """Get comprehensive task statistics."""
        state_counts: dict[str, int] = defaultdict(int)
        priority_counts: dict[str, int] = defaultdict(int)

        # Since this is a sync method, we can't use async lock
        # Use a snapshot of the data instead
        tasks_snapshot = dict(self._tasks)

        for task_info in tasks_snapshot.values():
            state_counts[task_info.state.value] += 1
            priority_counts[task_info.priority.value] += 1

        queue_sizes = {
            priority.value: queue.qsize() for priority, queue in self._task_queues.items()
        }

        return {
            "total_tasks": len(tasks_snapshot),
            "by_state": dict(state_counts),
            "by_priority": dict(priority_counts),
            "queue_sizes": queue_sizes,
            "workers": len(self._workers),
            "running": self._running,
            "stats": self._stats.copy(),
        }

    def get_task_info(self, task_id: str) -> dict[str, Any] | None:
        """Get information about a specific task."""
        task_info = self._tasks.get(task_id)
        if not task_info:
            return None

        return {
            "task_id": task_info.task_id,
            "name": task_info.name,
            "state": task_info.state.value,
            "priority": task_info.priority.value,
            "created_at": task_info.created_at.isoformat(),
            "started_at": task_info.started_at.isoformat() if task_info.started_at else None,
            "completed_at": task_info.completed_at.isoformat() if task_info.completed_at else None,
            "retries": task_info.retries,
            "max_retries": task_info.max_retries,
            "error": str(task_info.error) if task_info.error else None,
            "metadata": task_info.metadata,
        }


# Global task manager instance
_task_manager: TaskManager | None = None


def get_task_manager() -> TaskManager:
    """Get or create the global task manager."""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager


async def initialize_task_manager():
    """Initialize the global task manager."""
    manager = get_task_manager()
    await manager.start()
    return manager


async def shutdown_task_manager():
    """Shutdown the global task manager."""
    global _task_manager
    if _task_manager:
        await _task_manager.stop()
        _task_manager = None
