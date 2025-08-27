"""
Database performance integration tests.

Tests database operations under load and measures performance metrics
to ensure the database layer can handle production workloads.
"""

import asyncio
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import List

import pytest
import pytest_asyncio

from src.database.models.bot import Bot, Strategy
from src.database.models.market_data import MarketDataRecord
from src.database.models.trading import Order, Position
from src.database.repository.bot import BotRepository, StrategyRepository
from src.database.repository.market_data import MarketDataRepository
from src.database.repository.trading import OrderRepository, PositionRepository


@pytest.mark.asyncio
class TestDatabasePerformanceIntegration:
    """Test database performance under various load conditions."""

    @pytest.fixture
    def repositories(self, async_session):
        """Create repository instances for performance testing."""
        return {
            'bot': BotRepository(async_session),
            'strategy': StrategyRepository(async_session),
            'market_data': MarketDataRepository(async_session),
            'order': OrderRepository(async_session),
            'position': PositionRepository(async_session)
        }

    async def test_bulk_market_data_insertion(self, repositories):
        """Test inserting large amounts of market data efficiently."""
        # Create 100 market data records
        records = []
        base_time = datetime.now(timezone.utc)
        
        start_time = time.time()
        
        for i in range(100):
            record = MarketDataRecord(
                id=uuid.uuid4(),
                symbol=f"TEST{i % 10}USDT",  # 10 different symbols
                exchange="binance",
                data_timestamp=base_time,
                open_price=Decimal(f"100{i}.00"),
                high_price=Decimal(f"105{i}.00"),
                low_price=Decimal(f"95{i}.00"),
                close_price=Decimal(f"102{i}.00"),
                volume=Decimal(f"1000.{i}"),
                interval="1m",
                source="performance_test"
            )
            created_record = await repositories['market_data'].create(record)
            records.append(created_record)
        
        creation_time = time.time() - start_time
        
        # Performance assertion - should create 100 records in under 5 seconds
        assert creation_time < 5.0, f"Bulk creation took {creation_time:.2f} seconds, expected < 5.0"
        assert len(records) == 100
        
        # Test bulk querying performance
        start_time = time.time()
        
        all_test_data = await repositories['market_data'].get_all(
            filters={"source": "performance_test"}
        )
        
        query_time = time.time() - start_time
        
        # Should retrieve all records quickly
        assert query_time < 1.0, f"Bulk query took {query_time:.2f} seconds, expected < 1.0"
        assert len(all_test_data) == 100
        
        # Cleanup - test cleanup performance too
        start_time = time.time()
        
        cleanup_tasks = [repositories['market_data'].delete(record.id) for record in records]
        await asyncio.gather(*cleanup_tasks)
        
        cleanup_time = time.time() - start_time
        assert cleanup_time < 3.0, f"Bulk cleanup took {cleanup_time:.2f} seconds, expected < 3.0"

    async def test_concurrent_bot_operations(self, repositories):
        """Test concurrent bot creation and updates."""
        # Create 20 bots concurrently
        bot_tasks = []
        bot_ids = []
        
        start_time = time.time()
        
        for i in range(20):
            bot_id = uuid.uuid4()
            bot_ids.append(bot_id)
            bot = Bot(
                id=bot_id,
                name=f"Concurrent Bot {i}",
                description=f"Performance test bot {i}",
                status="RUNNING" if i % 2 == 0 else "STOPPED",
                exchange="binance" if i % 3 == 0 else "coinbase",
                allocated_capital=1000.0 + i * 100
            )
            bot_tasks.append(repositories['bot'].create(bot))
        
        created_bots = await asyncio.gather(*bot_tasks)
        creation_time = time.time() - start_time
        
        # Should create 20 bots concurrently in under 3 seconds
        assert creation_time < 3.0, f"Concurrent creation took {creation_time:.2f} seconds"
        assert len(created_bots) == 20
        assert all(bot is not None for bot in created_bots)
        
        # Test concurrent updates
        start_time = time.time()
        
        update_tasks = []
        for i, bot in enumerate(created_bots):
            bot.allocated_capital = 2000.0 + i * 50
            bot.current_balance = 1500.0 + i * 25
            update_tasks.append(repositories['bot'].update(bot))
        
        updated_bots = await asyncio.gather(*update_tasks)
        update_time = time.time() - start_time
        
        assert update_time < 2.0, f"Concurrent updates took {update_time:.2f} seconds"
        assert len(updated_bots) == 20
        
        # Test concurrent reads
        start_time = time.time()
        
        read_tasks = [repositories['bot'].get(bot_id) for bot_id in bot_ids]
        read_bots = await asyncio.gather(*read_tasks)
        
        read_time = time.time() - start_time
        assert read_time < 1.0, f"Concurrent reads took {read_time:.2f} seconds"
        assert len(read_bots) == 20
        
        # Cleanup
        cleanup_tasks = [repositories['bot'].delete(bot_id) for bot_id in bot_ids]
        await asyncio.gather(*cleanup_tasks)

    async def test_complex_query_performance(self, repositories):
        """Test performance of complex queries with filtering and ordering."""
        # Create test data
        bots = []
        strategies = []
        orders = []
        
        # Create 10 bots
        for i in range(10):
            bot = Bot(
                id=uuid.uuid4(),
                name=f"Query Test Bot {i}",
                description=f"Bot for query performance testing {i}",
                status="RUNNING" if i % 2 == 0 else "STOPPED",
                exchange="binance" if i % 2 == 0 else "coinbase",
                allocated_capital=5000.0 + i * 1000,
                total_trades=i * 10,
                winning_trades=i * 6,
                total_pnl=i * 500.0
            )
            created_bot = await repositories['bot'].create(bot)
            bots.append(created_bot)
        
        # Create 3 strategies per bot (30 total)
        for bot in bots:
            for j in range(3):
                strategy = Strategy(
                    id=uuid.uuid4(),
                    name=f"Strategy {j} for {bot.name}",
                    type=["trend_following", "arbitrage", "market_making"][j],
                    status="ACTIVE",
                    bot_id=bot.id,
                    total_signals=j * 20,
                    executed_signals=j * 15,
                    successful_signals=j * 10
                )
                created_strategy = await repositories['strategy'].create(strategy)
                strategies.append(created_strategy)
        
        # Create 5 orders per strategy (150 total)
        symbols = ["BTCUSD", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
        for strategy in strategies:
            for k, symbol in enumerate(symbols):
                order = Order(
                    id=uuid.uuid4(),
                    exchange="binance",
                    symbol=symbol,
                    side="BUY" if k % 2 == 0 else "SELL",
                    type="LIMIT",
                    status=["FILLED", "OPEN", "CANCELLED"][k % 3],
                    price=Decimal(f"{1000 + k * 100}.00"),
                    quantity=Decimal("1.0"),
                    filled_quantity=Decimal("1.0") if k % 3 == 0 else Decimal("0.0"),
                    bot_id=strategy.bot_id,
                    strategy_id=strategy.id
                )
                created_order = await repositories['order'].create(order)
                orders.append(created_order)
        
        # Test complex filtering performance
        start_time = time.time()
        
        # Query 1: Get all running bots with high capital
        running_bots = await repositories['bot'].get_all(
            filters={"status": "RUNNING"},
            order_by="-allocated_capital"
        )
        
        query1_time = time.time() - start_time
        assert query1_time < 0.5, f"Running bots query took {query1_time:.2f} seconds"
        assert len(running_bots) == 5  # Half of the bots are running
        
        # Query 2: Get all active strategies for specific bot type
        start_time = time.time()
        
        active_strategies = await repositories['strategy'].get_all(
            filters={"status": "ACTIVE", "type": "arbitrage"}
        )
        
        query2_time = time.time() - start_time
        assert query2_time < 0.5, f"Active strategies query took {query2_time:.2f} seconds"
        assert len(active_strategies) == 10  # One arbitrage strategy per bot
        
        # Query 3: Get filled orders for specific symbol
        start_time = time.time()
        
        btc_filled_orders = await repositories['order'].get_all(
            filters={"symbol": "BTCUSD", "status": "FILLED"}
        )
        
        query3_time = time.time() - start_time
        assert query3_time < 0.5, f"BTC filled orders query took {query3_time:.2f} seconds"
        
        # Cleanup all test data
        cleanup_start = time.time()
        
        cleanup_tasks = []
        for order in orders:
            cleanup_tasks.append(repositories['order'].delete(order.id))
        for strategy in strategies:
            cleanup_tasks.append(repositories['strategy'].delete(strategy.id))
        for bot in bots:
            cleanup_tasks.append(repositories['bot'].delete(bot.id))
        
        await asyncio.gather(*cleanup_tasks)
        
        cleanup_time = time.time() - cleanup_start
        assert cleanup_time < 5.0, f"Cleanup took {cleanup_time:.2f} seconds"

    async def test_memory_usage_under_load(self, repositories):
        """Test memory usage doesn't grow excessively under load."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and delete many entities in batches
        for batch in range(5):
            # Create 50 market data records
            records = []
            base_time = datetime.now(timezone.utc)
            
            for i in range(50):
                record = MarketDataRecord(
                    id=uuid.uuid4(),
                    symbol=f"BATCH{batch}_{i % 5}USDT",
                    exchange="binance",
                    data_timestamp=base_time,
                    open_price=Decimal(f"100{i}.00"),
                    high_price=Decimal(f"105{i}.00"),
                    low_price=Decimal(f"95{i}.00"),
                    close_price=Decimal(f"102{i}.00"),
                    volume=Decimal("1000.0"),
                    interval="1m",
                    source=f"memory_test_batch_{batch}"
                )
                created_record = await repositories['market_data'].create(record)
                records.append(created_record)
            
            # Delete all records in this batch
            delete_tasks = [repositories['market_data'].delete(record.id) for record in records]
            await asyncio.gather(*delete_tasks)
            
            # Force garbage collection
            import gc
            gc.collect()
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 50MB for this test)
        assert memory_growth < 50, f"Memory grew by {memory_growth:.2f}MB, expected < 50MB"

    async def test_repository_cache_performance(self, repositories):
        """Test repository caching mechanisms if implemented."""
        # Create a bot that we'll query multiple times
        bot = Bot(
            id=uuid.uuid4(),
            name="Cache Test Bot",
            description="Testing cache performance",
            status="RUNNING",
            exchange="binance",
            allocated_capital=10000.0
        )
        created_bot = await repositories['bot'].create(bot)
        
        # First query - should hit database
        start_time = time.time()
        bot1 = await repositories['bot'].get(created_bot.id)
        first_query_time = time.time() - start_time
        
        # Second query - might be cached if caching is implemented
        start_time = time.time()
        bot2 = await repositories['bot'].get(created_bot.id)
        second_query_time = time.time() - start_time
        
        # Both queries should return the same data
        assert bot1.id == bot2.id
        assert bot1.name == bot2.name
        
        # If caching is implemented, second query should be faster
        # This is more of an observation than assertion since caching might not be implemented
        cache_improvement = first_query_time - second_query_time
        print(f"First query: {first_query_time:.4f}s, Second query: {second_query_time:.4f}s")
        if cache_improvement > 0:
            print(f"Cache improvement: {cache_improvement:.4f}s")
        
        # Multiple rapid queries should all succeed
        rapid_query_start = time.time()
        rapid_query_tasks = [repositories['bot'].get(created_bot.id) for _ in range(10)]
        rapid_results = await asyncio.gather(*rapid_query_tasks)
        rapid_query_time = time.time() - rapid_query_start
        
        assert len(rapid_results) == 10
        assert all(bot.id == created_bot.id for bot in rapid_results)
        assert rapid_query_time < 1.0, f"10 rapid queries took {rapid_query_time:.2f} seconds"
        
        # Cleanup
        await repositories['bot'].delete(created_bot.id)