"""Task management utilities for analytics module."""

import asyncio
import uuid
from collections.abc import Callable, Coroutine
from datetime import datetime, timedelta
from typing import Any

from src.core.base import BaseComponent


class TaskManager(BaseComponent):
    """Centralized task management utilities for analytics."""

    def __init__(self):
        super().__init__()
        self._background_tasks: dict[str, asyncio.Task] = {}
        self._scheduled_tasks: dict[str, dict[str, Any]] = {}
        self._cleanup_interval = 300  # 5 minutes

    async def create_background_task(
        self,
        coro: Callable[[], Coroutine[Any, Any, Any]] | Coroutine[Any, Any, Any],
        task_name: str | None = None,
        cleanup_on_complete: bool = True,
    ) -> str:
        """Create and track a background task.

        Args:
            coro: Coroutine to run as background task
            task_name: Optional name for the task
            cleanup_on_complete: Whether to auto-cleanup completed tasks

        Returns:
            Task ID for tracking
        """
        task_id = task_name or f"task_{uuid.uuid4().hex[:8]}"

        task = asyncio.create_task(coro if asyncio.iscoroutine(coro) else coro())
        self._background_tasks[task_id] = task

        if cleanup_on_complete:
            task.add_done_callback(lambda t: self._cleanup_task(task_id))

        self.logger.debug(f"Created background task: {task_id}")
        return task_id

    def _cleanup_task(self, task_id: str) -> None:
        """Clean up completed task."""
        if task_id in self._background_tasks:
            task = self._background_tasks.pop(task_id)
            if task.exception():
                self.logger.error(f"Background task {task_id} failed: {task.exception()}")
            else:
                self.logger.debug(f"Background task {task_id} completed successfully")

    async def cancel_background_task(self, task_id: str) -> bool:
        """Cancel a background task.

        Args:
            task_id: ID of task to cancel

        Returns:
            True if task was cancelled, False if not found
        """
        if task_id not in self._background_tasks:
            return False

        task = self._background_tasks[task_id]
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            self.logger.debug(f"Background task {task_id} cancelled")

        self._cleanup_task(task_id)
        return True

    async def wait_for_tasks(
        self, task_ids: list[str] | None = None, timeout: float | None = None
    ) -> dict[str, Any]:
        """Wait for specific tasks or all tasks to complete.

        Args:
            task_ids: Specific task IDs to wait for, or None for all
            timeout: Maximum time to wait

        Returns:
            Dictionary of task results
        """
        if task_ids:
            tasks_to_wait = {
                tid: task for tid, task in self._background_tasks.items() if tid in task_ids
            }
        else:
            tasks_to_wait = self._background_tasks.copy()

        if not tasks_to_wait:
            return {}

        results = {}
        try:
            completed_tasks = await asyncio.wait_for(
                asyncio.gather(*tasks_to_wait.values(), return_exceptions=True), timeout=timeout
            )

            for task_id, result in zip(tasks_to_wait.keys(), completed_tasks, strict=False):
                results[task_id] = result
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout waiting for tasks: {list(tasks_to_wait.keys())}")
            for task_id, task in tasks_to_wait.items():
                if not task.done():
                    results[task_id] = asyncio.TimeoutError()

        return results

    def get_task_status(self, task_id: str) -> str | None:
        """Get status of a background task.

        Args:
            task_id: ID of task to check

        Returns:
            Task status string or None if not found
        """
        if task_id not in self._background_tasks:
            return None

        task = self._background_tasks[task_id]
        if task.done():
            if task.cancelled():
                return "cancelled"
            elif task.exception():
                return "failed"
            else:
                return "completed"
        else:
            return "running"

    def list_active_tasks(self) -> dict[str, str | None]:
        """List all active background tasks with their status."""
        return {task_id: self.get_task_status(task_id) for task_id in self._background_tasks.keys()}

    async def schedule_periodic_task(
        self,
        coro_factory: Callable[[], Coroutine[Any, Any, Any]],
        interval_seconds: float,
        task_name: str | None = None,
        max_runs: int | None = None,
    ) -> str:
        """Schedule a periodic task.

        Args:
            coro_factory: Function that returns coroutine to execute
            interval_seconds: Interval between executions
            task_name: Optional name for the scheduled task
            max_runs: Maximum number of runs (None for infinite)

        Returns:
            Scheduled task ID
        """
        task_id = task_name or f"periodic_{uuid.uuid4().hex[:8]}"

        self._scheduled_tasks[task_id] = {
            "coro_factory": coro_factory,
            "interval": interval_seconds,
            "max_runs": max_runs,
            "run_count": 0,
            "next_run": datetime.now() + timedelta(seconds=interval_seconds),
            "active": True,
        }

        # Start the periodic scheduler if not already running
        await self._start_periodic_scheduler()

        self.logger.debug(f"Scheduled periodic task: {task_id}")
        return task_id

    async def _start_periodic_scheduler(self) -> None:
        """Start the periodic task scheduler if not already running."""
        if "periodic_scheduler" not in self._background_tasks:
            await self.create_background_task(
                self._periodic_scheduler_loop(), "periodic_scheduler", cleanup_on_complete=False
            )

    async def _periodic_scheduler_loop(self) -> None:
        """Main loop for periodic task scheduling."""
        while True:
            current_time = datetime.now()

            for task_id, task_info in list(self._scheduled_tasks.items()):
                if not task_info["active"]:
                    continue

                if current_time >= task_info["next_run"]:
                    # Check if max runs reached
                    if task_info["max_runs"] and task_info["run_count"] >= task_info["max_runs"]:
                        task_info["active"] = False
                        continue

                    # Execute the task
                    try:
                        coro = task_info["coro_factory"]()
                        await self.create_background_task(
                            coro, f"{task_id}_run_{task_info['run_count']}"
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to execute periodic task {task_id}: {e}")

                    # Update for next run
                    task_info["run_count"] += 1
                    task_info["next_run"] = current_time + timedelta(seconds=task_info["interval"])

            # Sleep for a short interval
            await asyncio.sleep(1)

    def cancel_periodic_task(self, task_id: str) -> bool:
        """Cancel a periodic task.

        Args:
            task_id: ID of periodic task to cancel

        Returns:
            True if task was found and cancelled
        """
        if task_id in self._scheduled_tasks:
            self._scheduled_tasks[task_id]["active"] = False
            self.logger.debug(f"Cancelled periodic task: {task_id}")
            return True
        return False

    async def cleanup_completed_tasks(self) -> int:
        """Clean up all completed tasks.

        Returns:
            Number of tasks cleaned up
        """
        completed_task_ids = [
            task_id for task_id, task in self._background_tasks.items() if task.done()
        ]

        for task_id in completed_task_ids:
            self._cleanup_task(task_id)

        self.logger.debug(f"Cleaned up {len(completed_task_ids)} completed tasks")
        return len(completed_task_ids)

    async def shutdown(self) -> None:
        """Shutdown all tasks gracefully."""
        self.logger.info("Shutting down task manager...")

        # Cancel all background tasks
        for task_id in list(self._background_tasks.keys()):
            await self.cancel_background_task(task_id)

        # Deactivate all periodic tasks
        for task_info in self._scheduled_tasks.values():
            task_info["active"] = False

        self.logger.info("Task manager shutdown complete")
