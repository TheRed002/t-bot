"""
Cache warming strategies for the T-Bot trading system.

This module implements intelligent cache warming to preload critical data
into Redis before it's needed, optimizing performance for trading operations.
Strategies are designed specifically for financial trading data patterns.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from src.core.base.component import BaseComponent
from src.core.exceptions import CacheError

from .cache_keys import CacheKeys
from .cache_manager import get_cache_manager


class WarmingStrategy(Enum):
    """Cache warming strategy types."""

    IMMEDIATE = "immediate"  # Warm immediately on startup
    SCHEDULED = "scheduled"  # Warm on schedule (market hours, etc.)
    PROGRESSIVE = "progressive"  # Gradually warm over time
    DEMAND_BASED = "demand_based"  # Warm based on predicted demand
    MARKET_HOURS = "market_hours"  # Warm before market opens


class WarmingPriority(Enum):
    """Warming priority levels."""

    CRITICAL = "critical"  # Must warm (trading-critical data)
    HIGH = "high"  # Important data (risk metrics, positions)
    NORMAL = "normal"  # Useful data (historical data)
    LOW = "low"  # Nice-to-have data


@dataclass
class WarmingTask:
    """Represents a cache warming task."""

    task_id: str
    name: str
    strategy: WarmingStrategy
    priority: WarmingPriority
    warming_function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    cache_key: str = ""
    namespace: str = "warm"
    ttl: int = 3600

    # Scheduling
    schedule_cron: str = ""  # Cron expression
    market_offset_minutes: int = -30  # Minutes before market open
    interval_minutes: int = 0  # Interval for repeated warming

    # Control
    enabled: bool = True
    max_retries: int = 3
    retry_delay: float = 5.0
    timeout: float = 30.0

    # Status
    last_run: datetime | None = None
    last_success: datetime | None = None
    failure_count: int = 0
    total_runs: int = 0


class CacheWarmer(BaseComponent):
    """
    Intelligent cache warming system for trading data.

    Features:
    - Multiple warming strategies (immediate, scheduled, progressive)
    - Priority-based execution
    - Market-aware scheduling
    - Failure handling and retries
    - Performance monitoring
    - Batch warming for efficiency
    """

    def __init__(self, config: Any | None = None):
        super().__init__()
        self.config = config
        self.cache_manager = get_cache_manager(config=config)

        # Warming tasks registry
        self._warming_tasks: dict[str, WarmingTask] = {}
        self._active_warmers: dict[str, asyncio.Task] = {}

        # Queue management for proper task lifecycle
        self._task_queue: asyncio.Queue[WarmingTask] = asyncio.Queue()
        self._worker_tasks: list[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        # Warming state
        self._warming_active = False
        self._startup_warming_complete = False
        self._scheduler_task: asyncio.Task | None = None

        # Configuration
        self.batch_size = 10
        self.concurrent_limit = 5
        self.warming_delay = 0.1  # Delay between batch operations

        # Market hours (configurable)
        self.market_open_utc = 14  # 9:30 AM EST in UTC (approx)
        self.market_close_utc = 21  # 4:00 PM EST in UTC (approx)

        # Performance tracking
        self._warming_stats: dict[str, int | float | None] = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "cache_hits_generated": 0,
            "total_warming_time": 0.0,
            "last_full_warming": None,
        }

    async def start_warming(self) -> None:
        """Start the cache warming system with proper queue management."""
        if self._warming_active:
            return

        self._warming_active = True
        self._shutdown_event.clear()

        # Start worker tasks for queue processing
        self._worker_tasks = [
            asyncio.create_task(self._queue_worker(f"worker-{i}"))
            for i in range(self.concurrent_limit)
        ]

        # Start immediate warming tasks
        await self._run_immediate_warming()

        # Start scheduled warming loop
        self._scheduler_task = asyncio.create_task(self._warming_scheduler())

        self.logger.info(f"Cache warming system started with {len(self._worker_tasks)} workers")

    async def stop_warming(self) -> None:
        """Stop cache warming system with proper cleanup."""
        self.logger.info("Stopping cache warming system...")

        # Signal shutdown
        self._warming_active = False
        self._shutdown_event.set()

        # Cancel scheduler task
        scheduler_task = self._scheduler_task
        if scheduler_task and not scheduler_task.done():
            scheduler_task.cancel()
            try:
                await asyncio.wait_for(scheduler_task, timeout=10.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            finally:
                self._scheduler_task = None

        # Stop worker tasks gracefully
        worker_tasks = self._worker_tasks.copy()
        if worker_tasks:
            # Cancel all workers
            for worker in worker_tasks:
                if not worker.done():
                    worker.cancel()

            try:
                # Wait for workers to finish
                await asyncio.wait_for(
                    asyncio.gather(*worker_tasks, return_exceptions=True), timeout=15.0
                )
            except asyncio.TimeoutError:
                self.logger.warning("Some worker tasks did not complete within timeout")
            finally:
                self._worker_tasks.clear()

        # Cancel active warming tasks
        active_warmers = dict(self._active_warmers)
        if active_warmers:
            for _task_id, task in active_warmers.items():
                if not task.done():
                    task.cancel()

            try:
                await asyncio.wait_for(
                    asyncio.gather(*active_warmers.values(), return_exceptions=True),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                self.logger.warning("Some warming tasks did not complete within timeout")
            finally:
                self._active_warmers.clear()

        # Clear any remaining queue items
        while not self._task_queue.empty():
            try:
                self._task_queue.get_nowait()
                self._task_queue.task_done()
            except asyncio.QueueEmpty:
                break

        self.logger.info("Cache warming system stopped")

    async def _queue_worker(self, worker_name: str) -> None:
        """Worker task that processes warming tasks from the queue."""
        self.logger.debug(f"Starting queue worker: {worker_name}")

        try:
            while not self._shutdown_event.is_set():
                try:
                    # Get task from queue with timeout
                    task = await asyncio.wait_for(
                        self._task_queue.get(),
                        timeout=1.0,  # Check shutdown every second
                    )

                    try:
                        # Execute the warming task
                        self.logger.debug(f"Worker {worker_name} executing task: {task.name}")
                        success = await self._execute_single_warming_task(task)

                        # Update stats
                        if success:
                            current = self._warming_stats.get("successful_tasks", 0)
                            self._warming_stats["successful_tasks"] = (current or 0) + 1
                        else:
                            current = self._warming_stats.get("failed_tasks", 0)
                            self._warming_stats["failed_tasks"] = (current or 0) + 1

                    except Exception as e:
                        self.logger.error(
                            f"Worker {worker_name} failed to execute task {task.name}: {e}"
                        )
                        current = self._warming_stats.get("failed_tasks", 0)
                        self._warming_stats["failed_tasks"] = (current or 0) + 1
                    finally:
                        # CRITICAL: Always mark task as done
                        self._task_queue.task_done()

                except asyncio.TimeoutError:
                    # Timeout is expected when checking for shutdown
                    continue
                except asyncio.CancelledError:
                    # Worker was cancelled - clean shutdown
                    self.logger.debug(f"Worker {worker_name} cancelled")
                    break
                except Exception as e:
                    self.logger.error(f"Unexpected error in worker {worker_name}: {e}")
                    # Continue processing to prevent worker death

        except asyncio.CancelledError:
            self.logger.debug(f"Worker {worker_name} cancelled during shutdown")
        finally:
            self.logger.debug(f"Queue worker {worker_name} stopped")

    def register_warming_task(self, task: WarmingTask) -> None:
        """Register a cache warming task."""
        if not callable(task.warming_function):
            raise CacheError(f"Warming function must be callable for task {task.task_id}")

        self._warming_tasks[task.task_id] = task
        self.logger.info(f"Registered warming task: {task.name}", task_id=task.task_id)

    def register_market_data_warming(self, symbols: list[str], exchange: str = "all") -> None:
        """Register market data warming for trading symbols."""
        for symbol in symbols:
            # Latest price warming
            price_task = WarmingTask(
                task_id=f"price_{symbol}_{exchange}",
                name=f"Latest Price - {symbol}",
                strategy=WarmingStrategy.MARKET_HOURS,
                priority=WarmingPriority.CRITICAL,
                warming_function=self._warm_latest_price,
                args=(symbol, exchange),
                cache_key=CacheKeys.market_price(symbol, exchange),
                namespace="market_data",
                ttl=5,
                market_offset_minutes=-15,  # 15 minutes before market open
                interval_minutes=5,  # Refresh every 5 minutes during market hours
            )
            self.register_warming_task(price_task)

            # Order book warming
            orderbook_task = WarmingTask(
                task_id=f"orderbook_{symbol}_{exchange}",
                name=f"Order Book - {symbol}",
                strategy=WarmingStrategy.MARKET_HOURS,
                priority=WarmingPriority.HIGH,
                warming_function=self._warm_order_book,
                args=(symbol, exchange),
                cache_key=CacheKeys.order_book(symbol, exchange),
                namespace="market_data",
                ttl=10,
                market_offset_minutes=-10,
                interval_minutes=10,
            )
            self.register_warming_task(orderbook_task)

    def register_bot_state_warming(self, bot_ids: list[str]) -> None:
        """Register bot state warming for active bots."""
        for bot_id in bot_ids:
            # Bot status warming
            status_task = WarmingTask(
                task_id=f"bot_status_{bot_id}",
                name=f"Bot Status - {bot_id}",
                strategy=WarmingStrategy.IMMEDIATE,
                priority=WarmingPriority.CRITICAL,
                warming_function=self._warm_bot_status,
                args=(bot_id,),
                cache_key=CacheKeys.bot_status(bot_id),
                namespace="bot",
                ttl=30,
                interval_minutes=1,  # Refresh every minute
            )
            self.register_warming_task(status_task)

            # Bot configuration warming
            config_task = WarmingTask(
                task_id=f"bot_config_{bot_id}",
                name=f"Bot Config - {bot_id}",
                strategy=WarmingStrategy.IMMEDIATE,
                priority=WarmingPriority.HIGH,
                warming_function=self._warm_bot_config,
                args=(bot_id,),
                cache_key=CacheKeys.bot_config(bot_id),
                namespace="bot",
                ttl=3600,
            )
            self.register_warming_task(config_task)

    def register_risk_metrics_warming(
        self, bot_ids: list[str], timeframes: list[str] | None = None
    ) -> None:
        """Register risk metrics warming."""
        if timeframes is None:
            timeframes = ["1h", "1d"]
        for bot_id in bot_ids:
            for timeframe in timeframes:
                risk_task = WarmingTask(
                    task_id=f"risk_metrics_{bot_id}_{timeframe}",
                    name=f"Risk Metrics - {bot_id} ({timeframe})",
                    strategy=WarmingStrategy.SCHEDULED,
                    priority=WarmingPriority.HIGH,
                    warming_function=self._warm_risk_metrics,
                    args=(bot_id, timeframe),
                    cache_key=CacheKeys.risk_metrics(bot_id, timeframe),
                    namespace="risk",
                    ttl=300,  # 5 minutes
                    interval_minutes=15,  # Refresh every 15 minutes
                )
                self.register_warming_task(risk_task)

    def register_strategy_performance_warming(self, strategy_ids: list[str]) -> None:
        """Register strategy performance warming."""
        for strategy_id in strategy_ids:
            perf_task = WarmingTask(
                task_id=f"strategy_performance_{strategy_id}",
                name=f"Strategy Performance - {strategy_id}",
                strategy=WarmingStrategy.SCHEDULED,
                priority=WarmingPriority.NORMAL,
                warming_function=self._warm_strategy_performance,
                args=(strategy_id,),
                cache_key=CacheKeys.strategy_performance(strategy_id),
                namespace="strategy",
                ttl=900,  # 15 minutes
                interval_minutes=30,
            )
            self.register_warming_task(perf_task)

    async def _run_immediate_warming(self) -> None:
        """Run all immediate warming tasks."""
        immediate_tasks = [
            task
            for task in self._warming_tasks.values()
            if task.strategy == WarmingStrategy.IMMEDIATE and task.enabled
        ]

        if not immediate_tasks:
            return

        self.logger.info(f"Starting immediate warming for {len(immediate_tasks)} tasks")

        # Sort by priority
        immediate_tasks.sort(key=lambda t: list(WarmingPriority).index(t.priority))

        # Execute in batches using the queue
        for i in range(0, len(immediate_tasks), self.batch_size):
            batch = immediate_tasks[i : i + self.batch_size]
            await self._queue_warming_batch(batch)

            if self.warming_delay > 0:
                await asyncio.sleep(self.warming_delay)

        self._startup_warming_complete = True
        self.logger.info("Immediate warming completed")

    async def _queue_warming_batch(self, tasks: list[WarmingTask]) -> None:
        """Queue a batch of warming tasks for processing."""
        for task in tasks:
            await self._task_queue.put(task)

        self.logger.debug(f"Queued {len(tasks)} tasks for warming")

    async def _warming_scheduler(self) -> None:
        """Main warming scheduler loop with proper queue management."""
        try:
            while self._warming_active and not self._shutdown_event.is_set():
                try:
                    current_time = datetime.now(timezone.utc)
                    tasks_to_schedule = []

                    # Check for scheduled tasks
                    for task in self._warming_tasks.values():
                        if not task.enabled:
                            continue

                        should_run = False

                        if task.strategy == WarmingStrategy.MARKET_HOURS:
                            should_run = await self._should_run_market_hours_task(
                                task, current_time
                            )
                        elif task.strategy == WarmingStrategy.SCHEDULED:
                            should_run = await self._should_run_scheduled_task(task, current_time)
                        elif task.strategy == WarmingStrategy.PROGRESSIVE:
                            should_run = await self._should_run_progressive_task(task, current_time)

                        if should_run and task.task_id not in self._active_warmers:
                            tasks_to_schedule.append(task)

                    # Queue scheduled tasks
                    if tasks_to_schedule:
                        await self._queue_warming_batch(tasks_to_schedule)

                    # Clean up completed tasks
                    completed = [
                        task_id for task_id, task in self._active_warmers.items() if task.done()
                    ]
                    for task_id in completed:
                        del self._active_warmers[task_id]

                    # Use cancellable sleep
                    try:
                        await asyncio.wait_for(
                            asyncio.sleep(60),  # Check every minute
                            timeout=61.0,
                        )
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in warming scheduler: {e}")
                    try:
                        await asyncio.sleep(60)
                    except asyncio.CancelledError:
                        break

        except asyncio.CancelledError:
            self.logger.debug("Warming scheduler cancelled")
        finally:
            self.logger.debug("Warming scheduler stopped")

    async def _should_run_market_hours_task(
        self, task: WarmingTask, current_time: datetime
    ) -> bool:
        """Check if market hours task should run."""
        current_hour = current_time.hour

        # Calculate market open time with offset
        market_open_with_offset = self.market_open_utc + (task.market_offset_minutes / 60.0)

        # Check if we're in the warming window
        if task.interval_minutes > 0:
            # Recurring task during market hours
            if self.market_open_utc <= current_hour < self.market_close_utc:
                if (
                    task.last_run is None
                    or (current_time - task.last_run).total_seconds() >= task.interval_minutes * 60
                ):
                    return True
        else:
            # One-time warming before market open
            if abs(current_hour - market_open_with_offset) < 0.5 and (  # Within 30 minutes
                task.last_run is None or task.last_run.date() != current_time.date()
            ):  # Not run today
                return True

        return False

    async def _should_run_scheduled_task(self, task: WarmingTask, current_time: datetime) -> bool:
        """Check if scheduled task should run."""
        if task.interval_minutes <= 0:
            return False

        if (
            task.last_run is None
            or (current_time - task.last_run).total_seconds() >= task.interval_minutes * 60
        ):
            return True

        return False

    async def _should_run_progressive_task(self, task: WarmingTask, current_time: datetime) -> bool:
        """Check if progressive warming task should run."""
        # Progressive tasks run with increasing intervals after failures
        if task.last_run is None:
            return True

        base_interval = task.interval_minutes or 60
        # Increase interval based on failure count
        actual_interval = base_interval * (1.5 ** min(task.failure_count, 5))

        if (current_time - task.last_run).total_seconds() >= actual_interval * 60:
            return True

        return False

    async def _execute_warming_batch(self, tasks: list[WarmingTask]) -> None:
        """Execute a batch of warming tasks concurrently."""
        # Limit concurrency
        semaphore = asyncio.Semaphore(self.concurrent_limit)

        async def execute_with_semaphore(task):
            async with semaphore:
                return await self._execute_single_warming_task(task)

        # Execute batch
        results = await asyncio.gather(
            *[execute_with_semaphore(task) for task in tasks], return_exceptions=True
        )

        # Log results
        successful = sum(1 for r in results if r is True)
        failed = len(results) - successful

        if failed > 0:
            self.logger.warning(
                f"Warming batch completed: {successful} successful, {failed} failed"
            )
        else:
            self.logger.info(f"Warming batch completed successfully: {successful} tasks")

    async def _execute_single_warming_task(self, task: WarmingTask) -> bool:
        """Execute a single warming task."""
        start_time = asyncio.get_event_loop().time()

        try:
            task.last_run = datetime.now(timezone.utc)
            task.total_runs += 1

            # Execute warming function
            if asyncio.iscoroutinefunction(task.warming_function):
                result = await asyncio.wait_for(
                    task.warming_function(*task.args, **task.kwargs), timeout=task.timeout
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, task.warming_function, *task.args, **task.kwargs
                    ),
                    timeout=task.timeout,
                )

            # Cache the result if warming function returned data
            if result is not None and task.cache_key:
                await self.cache_manager.set(
                    task.cache_key,
                    result,
                    namespace=task.namespace,
                    ttl=task.ttl,
                    data_type=task.namespace,
                )

            # Update success tracking
            task.last_success = datetime.now(timezone.utc)
            task.failure_count = 0

            # Update stats
            current_success = self._warming_stats.get("successful_tasks", 0)
            self._warming_stats["successful_tasks"] = (current_success or 0) + 1

            current_hits = self._warming_stats.get("cache_hits_generated", 0)
            self._warming_stats["cache_hits_generated"] = (current_hits or 0) + 1

            execution_time = asyncio.get_event_loop().time() - start_time
            current_time = self._warming_stats.get("total_warming_time", 0.0)
            self._warming_stats["total_warming_time"] = (current_time or 0.0) + execution_time

            self.logger.debug(f"Warming task completed: {task.name}", execution_time=execution_time)
            return True

        except asyncio.TimeoutError:
            self.logger.warning(f"Warming task timeout: {task.name}")
            task.failure_count += 1
            current = self._warming_stats.get("failed_tasks", 0)
            self._warming_stats["failed_tasks"] = (current or 0) + 1
            return False

        except Exception as e:
            self.logger.error(f"Warming task failed: {task.name}", error=str(e))
            task.failure_count += 1
            current = self._warming_stats.get("failed_tasks", 0)
            self._warming_stats["failed_tasks"] = (current or 0) + 1

            # Retry logic
            if task.failure_count <= task.max_retries:
                await asyncio.sleep(task.retry_delay)
                return await self._execute_single_warming_task(task)

            return False

    # Warming function implementations for common data types
    async def _warm_latest_price(self, symbol: str, exchange: str = "all") -> dict[str, Any] | None:
        """Warm latest price data."""
        # This would integrate with your data service
        # For now, return mock data structure
        return {
            "symbol": symbol,
            "exchange": exchange,
            "price": 100.0,  # Would fetch real price
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "volume": 1000.0,
        }

    async def _warm_order_book(
        self, symbol: str, exchange: str, depth: int = 20
    ) -> dict[str, Any] | None:
        """Warm order book data."""
        return {
            "symbol": symbol,
            "exchange": exchange,
            "bids": [],  # Would fetch real order book
            "asks": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def _warm_bot_status(self, bot_id: str) -> dict[str, Any] | None:
        """Warm bot status data."""
        return {
            "bot_id": bot_id,
            "status": "running",  # Would fetch real status
            "last_update": datetime.now(timezone.utc).isoformat(),
            "health": "healthy",
        }

    async def _warm_bot_config(self, bot_id: str) -> dict[str, Any] | None:
        """Warm bot configuration data."""
        return {
            "bot_id": bot_id,
            "config": {},  # Would fetch real config
            "last_modified": datetime.now(timezone.utc).isoformat(),
        }

    async def _warm_risk_metrics(self, bot_id: str, timeframe: str) -> dict[str, Any] | None:
        """Warm risk metrics data."""
        return {
            "bot_id": bot_id,
            "timeframe": timeframe,
            "metrics": {},  # Would calculate real metrics
            "calculated_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _warm_strategy_performance(self, strategy_id: str) -> dict[str, Any] | None:
        """Warm strategy performance data."""
        return {
            "strategy_id": strategy_id,
            "performance": {},  # Would calculate real performance
            "calculated_at": datetime.now(timezone.utc).isoformat(),
        }

    async def get_warming_status(self) -> dict[str, Any]:
        """Get warming system status and statistics."""
        active_tasks = len([t for t in self._warming_tasks.values() if t.enabled])

        return {
            "warming_active": self._warming_active,
            "startup_warming_complete": self._startup_warming_complete,
            "registered_tasks": len(self._warming_tasks),
            "active_tasks": active_tasks,
            "running_warmers": len(self._active_warmers),
            "stats": self._warming_stats.copy(),
            "task_summary": {
                task_id: {
                    "name": task.name,
                    "strategy": task.strategy.value,
                    "priority": task.priority.value,
                    "enabled": task.enabled,
                    "last_run": task.last_run.isoformat() if task.last_run else None,
                    "last_success": task.last_success.isoformat() if task.last_success else None,
                    "failure_count": task.failure_count,
                    "total_runs": task.total_runs,
                }
                for task_id, task in self._warming_tasks.items()
            },
        }

    async def warm_critical_data_now(self) -> dict[str, Any]:
        """Immediately warm all critical priority data."""
        critical_tasks = [
            task
            for task in self._warming_tasks.values()
            if task.priority == WarmingPriority.CRITICAL and task.enabled
        ]

        if not critical_tasks:
            return {"message": "No critical warming tasks found", "warmed": 0}

        self.logger.info(f"Warming {len(critical_tasks)} critical tasks immediately")

        await self._queue_warming_batch(critical_tasks)

        # Wait for tasks to complete
        await self._task_queue.join()

        return {
            "message": f"Warmed {len(critical_tasks)} critical tasks",
            "warmed": len(critical_tasks),
            "tasks": [task.name for task in critical_tasks],
        }


# Global warmer instance
_cache_warmer: CacheWarmer | None = None


def get_cache_warmer(config: Any | None = None) -> CacheWarmer:
    """Get or create global cache warmer instance."""
    global _cache_warmer
    if _cache_warmer is None:
        _cache_warmer = CacheWarmer(config)
    return _cache_warmer
