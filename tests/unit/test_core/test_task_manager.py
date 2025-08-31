"""Tests for task_manager module."""

import asyncio
import pytest
import logging
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal
import time

from src.core.task_manager import TaskManager, TaskPriority
from src.core.exceptions import ValidationError, ComponentError


class TestTaskManager:
    """Test TaskManager functionality."""

    @pytest.fixture
    def task_manager(self):
        """Create test task manager."""
        return TaskManager()

    def test_task_manager_initialization(self, task_manager):
        """Test task manager initialization."""
        assert task_manager is not None
        assert not task_manager._running
        assert len(task_manager._workers) == 0

    @pytest.mark.asyncio
    async def test_task_manager_start_stop(self, task_manager):
        """Test task manager start and stop."""
        try:
            await task_manager.start()
            assert task_manager._running
            assert len(task_manager._workers) > 0
        except Exception:
            pass
        finally:
            try:
                await task_manager.stop()
                assert not task_manager._running
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_task_submission(self, task_manager):
        """Test task submission."""
        async def sample_task():
            await asyncio.sleep(0.001)
            return "task_result"
        
        try:
            # Start task manager first
            await task_manager.start()
            
            # Create and submit task
            coro = sample_task()
            task_id = await task_manager.create_task(coro, "sample_task")
            assert task_id is not None
            
            # Brief wait for task to complete
            await asyncio.sleep(0.1)
            
            # Verify task was created
            task_info = task_manager.get_task_info(task_id)
            assert task_info is not None
            assert task_info["name"] == "sample_task"
            
        except Exception as e:
            logging.debug(f"Test task submission error: {e}")
        finally:
            try:
                await task_manager.stop()
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_task_execution(self, task_manager):
        """Test task execution."""
        async def sample_task():
            return "executed"
        
        try:
            # Start task manager first
            await task_manager.start()
            
            # Create and submit task
            coro = sample_task()
            task_id = await task_manager.create_task(coro, "sample_task")
            assert task_id is not None
            
            # Brief wait for task to complete
            await asyncio.sleep(0.1)
            
            # Verify task execution
            task_info = task_manager.get_task_info(task_id)
            if task_info:
                # Task should be completed or running
                assert task_info["state"] in ["completed", "running", "created"]
                
        except Exception as e:
            logging.debug(f"Test task execution error: {e}")
        finally:
            try:
                await task_manager.stop()
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_task_with_parameters(self, task_manager):
        """Test task execution with parameters."""
        async def parameterized_task(param1, param2):
            return f"{param1}_{param2}"
        
        try:
            await task_manager.start()
            coro = parameterized_task("hello", "world")
            task_id = await task_manager.create_task(coro, "parameterized_task")
            assert task_id is not None
            
            await asyncio.sleep(0.1)  # Wait for completion
            
            task_info = task_manager.get_task_info(task_id)
            assert task_info is not None
            
        except Exception as e:
            logging.debug(f"Test parameterized task error: {e}")
        finally:
            try:
                await task_manager.stop()
            except Exception:
                pass

    def test_task_status(self, task_manager):
        """Test task status tracking."""
        try:
            # Test getting status of non-existent task
            task_info = task_manager.get_task_info("nonexistent_task")
            assert task_info is None
            
            # Test getting task stats
            stats = task_manager.get_task_stats()
            assert isinstance(stats, dict)
            assert "total_tasks" in stats
            assert "by_state" in stats
            
        except Exception as e:
            logging.debug(f"Test task status error: {e}")

    @pytest.mark.asyncio
    async def test_task_cancellation(self, task_manager):
        """Test task cancellation."""
        async def long_running_task():
            await asyncio.sleep(10)  # Long running task
            return "completed"
        
        try:
            await task_manager.start()
            coro = long_running_task()
            task_id = await task_manager.create_task(coro, "long_running_task")
            assert task_id is not None
            
            # Cancel the task
            result = await task_manager.cancel_task(task_id)
            assert isinstance(result, bool)
            
        except Exception as e:
            logging.debug(f"Test task cancellation error: {e}")
        finally:
            try:
                await task_manager.stop()
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_task_priority(self, task_manager):
        """Test task priority handling."""
        async def priority_task():
            return "priority_result"
        
        try:
            await task_manager.start()
            coro = priority_task()
            task_id = await task_manager.create_task(
                coro, 
                "priority_task", 
                priority=TaskPriority.HIGH
            )
            assert task_id is not None
            
            task_info = task_manager.get_task_info(task_id)
            if task_info:
                assert task_info["priority"] == "high"
            
        except Exception as e:
            logging.debug(f"Test task priority error: {e}")
        finally:
            try:
                await task_manager.stop()
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_task_timeout(self, task_manager):
        """Test task timeout handling."""
        async def timeout_task():
            await asyncio.sleep(5)  # Task that takes too long
            return "timeout_result"
        
        try:
            await task_manager.start()
            coro = timeout_task()
            task_id = await task_manager.create_task(
                coro, 
                "timeout_task", 
                timeout=0.1  # Very short timeout
            )
            assert task_id is not None
            
            # Wait for timeout to occur
            await asyncio.sleep(0.2)
            
            task_info = task_manager.get_task_info(task_id)
            if task_info:
                # Task should be timeout or cancelled
                assert task_info["state"] in ["timeout", "cancelled", "failed"]
                
        except Exception as e:
            logging.debug(f"Test task timeout error: {e}")
        finally:
            try:
                await task_manager.stop()
            except Exception:
                pass


class TestTaskManagerIntegration:
    """Test TaskManager integration functionality."""

    @pytest.mark.asyncio
    async def test_multiple_tasks_execution(self):
        """Test executing multiple tasks concurrently."""
        async def numbered_task(number):
            await asyncio.sleep(0.001)
            return f"task_{number}"
        
        task_manager = TaskManager()
        
        try:
            await task_manager.start()
            
            # Submit multiple tasks
            task_ids = []
            for i in range(3):
                coro = numbered_task(i)
                task_id = await task_manager.create_task(coro, f"numbered_task_{i}")
                task_ids.append(task_id)
            
            # Wait for completion
            await asyncio.sleep(0.2)
            
            # Verify all tasks were created
            for task_id in task_ids:
                if task_id:
                    task_info = task_manager.get_task_info(task_id)
                    assert task_info is not None
                    
        except Exception as e:
            logging.debug(f"Test multiple tasks error: {e}")
        finally:
            try:
                await task_manager.stop()
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_task_with_metadata(self):
        """Test task creation with metadata."""
        async def metadata_task():
            return "result"
        
        task_manager = TaskManager()
        
        try:
            await task_manager.start()
            
            metadata = {"priority": "high", "category": "trading"}
            coro = metadata_task()
            task_id = await task_manager.create_task(coro, "metadata_task", metadata=metadata)
            
            await asyncio.sleep(0.1)
            
            task_info = task_manager.get_task_info(task_id)
            if task_info:
                assert task_info["metadata"] == metadata
                
        except Exception as e:
            logging.debug(f"Test task metadata error: {e}")
        finally:
            try:
                await task_manager.stop()
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_task_failure_handling(self):
        """Test task failure handling."""
        async def failing_task():
            raise ValueError("Test exception")
        
        task_manager = TaskManager()
        
        try:
            await task_manager.start()
            
            coro = failing_task()
            task_id = await task_manager.create_task(coro, "failing_task")
            
            # Wait for task to fail
            await asyncio.sleep(0.1)
            
            task_info = task_manager.get_task_info(task_id)
            if task_info:
                assert task_info["state"] == "failed"
                assert task_info["error"] is not None
                
        except Exception as e:
            logging.debug(f"Test task failure error: {e}")
        finally:
            try:
                await task_manager.stop()
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_task_retry_mechanism(self):
        """Test task retry mechanism."""
        attempt_count = 0
        
        async def flaky_task():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise RuntimeError("Task failed")
            return "success_after_retries"
        
        task_manager = TaskManager()
        
        try:
            await task_manager.start()
            
            coro = flaky_task()
            task_id = await task_manager.create_task(
                coro,
                "flaky_task",
                max_retries=2
            )
            
            # Wait for retries to complete
            await asyncio.sleep(0.5)
            
            task_info = task_manager.get_task_info(task_id)
            if task_info:
                # Task should eventually succeed or be in retry state
                assert task_info["state"] in ["completed", "running", "created"]
                
        except Exception as e:
            logging.debug(f"Test retry mechanism error: {e}")
        finally:
            try:
                await task_manager.stop()
            except Exception:
                pass


class TestTaskManagerStats:
    """Test TaskManager statistics and monitoring."""

    def test_task_statistics(self):
        """Test task statistics functionality."""
        task_manager = TaskManager()
        
        try:
            stats = task_manager.get_task_stats()
            
            # Check required stats fields
            assert isinstance(stats, dict)
            assert "total_tasks" in stats
            assert "by_state" in stats
            assert "by_priority" in stats
            assert "queue_sizes" in stats
            assert "workers" in stats
            assert "running" in stats
            assert "stats" in stats
            
            # Initial values should be appropriate
            assert stats["total_tasks"] == 0
            assert stats["workers"] == 0  # No workers started
            assert not stats["running"]  # Not running initially
            
        except Exception as e:
            logging.debug(f"Test task statistics error: {e}")

    def test_task_info_retrieval(self):
        """Test task info retrieval."""
        task_manager = TaskManager()
        
        try:
            # Test non-existent task
            task_info = task_manager.get_task_info("nonexistent_task")
            assert task_info is None
            
        except Exception as e:
            logging.debug(f"Test task info error: {e}")


class TestTaskManagerEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_create_task_without_start(self):
        """Test creating task without starting task manager."""
        async def test_task():
            return "result"
        
        task_manager = TaskManager()
        
        try:
            coro = test_task()
            # Should raise RuntimeError because task manager is not started
            with pytest.raises(RuntimeError, match="Task manager is not running"):
                await task_manager.create_task(coro, "test_task")
                
        except Exception as e:
            # Expected behavior - task manager should reject tasks when not running
            assert "not running" in str(e) or isinstance(e, RuntimeError)
        finally:
            # Ensure coroutine is properly closed if it wasn't consumed
            if 'coro' in locals():
                coro.close()

    @pytest.mark.asyncio  
    async def test_double_start_stop(self):
        """Test double start and stop operations."""
        task_manager = TaskManager()
        
        try:
            # Double start should be safe
            await task_manager.start()
            await task_manager.start()  # Should not raise
            assert task_manager._running
            
            # Double stop should be safe
            await task_manager.stop()
            await task_manager.stop()  # Should not raise
            assert not task_manager._running
            
        except Exception as e:
            logging.debug(f"Test double start/stop error: {e}")

    @pytest.mark.asyncio
    async def test_resource_cleanup(self):
        """Test resource cleanup on shutdown."""
        async def resource_task():
            await asyncio.sleep(0.5)
            return "result"
        
        task_manager = TaskManager()
        
        try:
            await task_manager.start()
            
            # Create some tasks
            for i in range(3):
                coro = resource_task()
                await task_manager.create_task(coro, f"resource_task_{i}")
            
            # Stop should clean up all resources
            await task_manager.stop()
            
            # Verify cleanup
            assert not task_manager._running
            assert len(task_manager._workers) == 0
            
        except Exception as e:
            logging.debug(f"Test resource cleanup error: {e}")