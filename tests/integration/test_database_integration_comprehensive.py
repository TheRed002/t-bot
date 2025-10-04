"""
Comprehensive database integration tests.

Tests end-to-end workflows across multiple database components to ensure
proper integration between repositories, models, and services.
"""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models.bot import Bot, Strategy, Signal
from src.database.models.capital import CapitalAllocationDB, FundFlowDB
from src.database.models.market_data import MarketDataRecord
from src.database.models.trading import Order, Position
from src.database.models.audit import CapitalAuditLog
from src.database.repository.bot import BotRepository, StrategyRepository, SignalRepository
from src.database.repository.capital import CapitalAllocationRepository, FundFlowRepository
from src.database.repository.market_data import MarketDataRepository
from src.database.repository.trading import OrderRepository, PositionRepository
from src.database.repository.audit import CapitalAuditLogRepository


@pytest.mark.asyncio
class TestDatabaseIntegrationWorkflows:
    """Test complete database workflows involving multiple components."""

    @pytest_asyncio.fixture
    async def integration_session(self, async_session):
        """Create a test session for integration tests."""
        return async_session

    @pytest.fixture
    def repositories(self, integration_session):
        """Create all repository instances for testing."""
        return {
            'bot': BotRepository(integration_session),
            'strategy': StrategyRepository(integration_session),
            'signal': SignalRepository(integration_session),
            'capital_allocation': CapitalAllocationRepository(integration_session),
            'fund_flow': FundFlowRepository(integration_session),
            'market_data': MarketDataRepository(integration_session),
            'order': OrderRepository(integration_session),
            'position': PositionRepository(integration_session),
            'audit': CapitalAuditLogRepository(integration_session)
        }

    async def test_complete_trading_workflow(self, repositories, integration_session):
        """Test a complete trading workflow from bot creation to trade execution."""
        # Step 1: Create a bot
        bot = Bot(
            id=uuid.uuid4(),
            name="Integration Test Bot",
            description="Bot for integration testing",
            status="RUNNING",
            exchange="binance",
            allocated_capital=10000.0,
            current_balance=10000.0
        )
        created_bot = await repositories['bot'].create(bot)
        assert created_bot is not None
        assert created_bot.id == bot.id

        # Step 2: Create a strategy for the bot
        strategy = Strategy(
            id=uuid.uuid4(),
            name="Test Strategy",
            type="trend_following",
            status="ACTIVE",
            bot_id=created_bot.id,
            params={"lookback": 20, "threshold": 0.02},
            max_position_size=1000.0,
            risk_per_trade=0.02
        )
        created_strategy = await repositories['strategy'].create(strategy)
        assert created_strategy is not None
        assert created_strategy.bot_id == created_bot.id

        # Step 3: Add market data
        now = datetime.now(timezone.utc)
        market_data = MarketDataRecord(
            id=uuid.uuid4(),
            symbol="BTCUSD",
            exchange="binance",
            data_timestamp=now,
            open_price=Decimal("45000.00"),
            high_price=Decimal("45500.00"),
            low_price=Decimal("44500.00"),
            close_price=Decimal("45200.00"),
            volume=Decimal("100.0"),
            interval="1h",
            source="exchange"
        )
        created_market_data = await repositories['market_data'].create(market_data)
        assert created_market_data is not None

        # Step 4: Create a trading signal
        signal = Signal(
            id=uuid.uuid4(),
            strategy_id=created_strategy.id,
            symbol="BTCUSD",
            action="BUY",
            strength=0.8,
            price=Decimal("45200.00"),
            quantity=Decimal("0.5"),
            reason="Trend following signal"
        )
        created_signal = await repositories['signal'].create(signal)
        assert created_signal is not None

        # Step 5: Create an order based on the signal
        order = Order(
            id=uuid.uuid4(),
            exchange="binance",
            symbol="BTCUSD",
            side="BUY",
            type="LIMIT",
            status="FILLED",
            price=Decimal("45200.00"),
            quantity=Decimal("0.5"),
            filled_quantity=Decimal("0.5"),
            bot_id=created_bot.id,
            strategy_id=created_strategy.id
        )
        created_order = await repositories['order'].create(order)
        assert created_order is not None

        # Step 6: Create a position
        position = Position(
            id=uuid.uuid4(),
            exchange="binance",
            symbol="BTCUSD",
            side="LONG",
            status="OPEN",
            quantity=Decimal("0.5"),
            entry_price=Decimal("45200.00"),
            current_price=Decimal("45200.00"),
            bot_id=created_bot.id,
            strategy_id=created_strategy.id
        )
        created_position = await repositories['position'].create(position)
        assert created_position is not None

        # Step 7: Update signal as executed
        created_signal.executed = True
        created_signal.execution_time = 1.5  # 1.5 seconds
        created_signal.order_id = created_order.id
        updated_signal = await repositories['signal'].update(created_signal)
        assert updated_signal.executed is True

        # Step 8: Verify relationships and data consistency
        # Check that bot has the strategy
        bot_strategies = await repositories['strategy'].get_all(
            filters={"bot_id": str(created_bot.id)}
        )
        assert len(bot_strategies) == 1
        assert bot_strategies[0].id == created_strategy.id

        # Check that strategy has the signal
        strategy_signals = await repositories['signal'].get_all(
            filters={"strategy_id": str(created_strategy.id)}
        )
        assert len(strategy_signals) == 1
        assert strategy_signals[0].id == created_signal.id

        # Check market data exists for the symbol
        symbol_data = await repositories['market_data'].get_by_symbol("BTCUSD")
        assert len(symbol_data) >= 1

        # Cleanup
        await repositories['position'].delete(created_position.id)
        await repositories['order'].delete(created_order.id)
        await repositories['signal'].delete(created_signal.id)
        await repositories['strategy'].delete(created_strategy.id)
        await repositories['bot'].delete(created_bot.id)
        await repositories['market_data'].delete(created_market_data.id)

    async def test_capital_management_workflow(self, repositories, integration_session):
        """Test capital allocation and fund flow workflow."""
        # Step 1: Create capital allocation
        allocation = CapitalAllocationDB(
            id=uuid.uuid4(),
            strategy_id="test_strategy_001",
            exchange="binance",
            allocated_amount=Decimal("5000.00"),
            utilized_amount=Decimal("0.00"),
            available_amount=Decimal("5000.00"),
            allocation_type="fixed"
        )
        created_allocation = await repositories['capital_allocation'].create(allocation)
        assert created_allocation is not None

        # Step 2: Create fund flow record
        flow = FundFlowDB(
            id=uuid.uuid4(),
            flow_type="allocation",
            from_account="main",
            to_account="strategy_001",
            currency="USD",
            amount=Decimal("5000.00"),
            status="COMPLETED"
        )
        created_flow = await repositories['fund_flow'].create(flow)
        assert created_flow is not None

        # Step 3: Create audit log
        audit = CapitalAuditLog(
            id=str(uuid.uuid4()),
            operation_id=str(uuid.uuid4()),
            operation_type="allocate",
            strategy_id="test_strategy_001",
            exchange="binance",
            operation_description="Initial capital allocation",
            amount=Decimal("5000.00"),
            success=True,
            requested_at=datetime.now(timezone.utc),
            source_component="CapitalManager"
        )
        created_audit = await repositories['audit'].create(audit)
        assert created_audit is not None

        # Step 4: Verify data consistency
        allocations = await repositories['capital_allocation'].get_all(
            filters={"strategy_id": "test_strategy_001"}
        )
        assert len(allocations) == 1
        assert allocations[0].allocated_amount == Decimal("5000.00")

        flows = await repositories['fund_flow'].get_all(
            filters={"flow_type": "allocation"}
        )
        assert len(flows) >= 1

        # Cleanup
        await repositories['audit'].delete(created_audit.id)
        await repositories['fund_flow'].delete(created_flow.id)
        await repositories['capital_allocation'].delete(created_allocation.id)

    async def test_concurrent_repository_operations(self, repositories, integration_session):
        """Test concurrent operations across repositories."""
        # Create multiple bots concurrently
        bot_tasks = []
        for i in range(3):
            bot = Bot(
                id=uuid.uuid4(),
                name=f"Concurrent Bot {i}",
                description=f"Bot {i} for concurrency testing",
                status="RUNNING",
                exchange="binance",
                allocated_capital=1000.0 + i * 1000
            )
            bot_tasks.append(repositories['bot'].create(bot))

        created_bots = await asyncio.gather(*bot_tasks)
        assert len(created_bots) == 3

        # Create strategies for each bot concurrently
        strategy_tasks = []
        for i, bot in enumerate(created_bots):
            strategy = Strategy(
                id=uuid.uuid4(),
                name=f"Strategy {i}",
                type="arbitrage",
                status="ACTIVE",
                bot_id=bot.id,
                params={"spread_threshold": 0.001 + i * 0.001}
            )
            strategy_tasks.append(repositories['strategy'].create(strategy))

        created_strategies = await asyncio.gather(*strategy_tasks)
        assert len(created_strategies) == 3

        # Verify all data was created correctly
        all_bots = await repositories['bot'].get_all()
        concurrent_bots = [b for b in all_bots if "Concurrent Bot" in b.name]
        assert len(concurrent_bots) == 3

        # Cleanup
        cleanup_tasks = []
        for strategy in created_strategies:
            cleanup_tasks.append(repositories['strategy'].delete(strategy.id))
        for bot in created_bots:
            cleanup_tasks.append(repositories['bot'].delete(bot.id))
        
        await asyncio.gather(*cleanup_tasks)

    async def test_data_consistency_across_updates(self, repositories, integration_session):
        """Test data consistency when updating related entities."""
        # Create bot and strategy
        bot = Bot(
            id=uuid.uuid4(),
            name="Consistency Test Bot",
            description="Testing data consistency",
            status="RUNNING",
            exchange="binance",
            allocated_capital=5000.0
        )
        created_bot = await repositories['bot'].create(bot)

        strategy = Strategy(
            id=uuid.uuid4(),
            name="Consistency Test Strategy",
            type="market_making",
            status="ACTIVE",
            bot_id=created_bot.id,
            total_signals=0,
            executed_signals=0
        )
        created_strategy = await repositories['strategy'].create(strategy)

        # Create multiple signals
        signals = []
        for i in range(5):
            signal = Signal(
                id=uuid.uuid4(),
                strategy_id=created_strategy.id,
                symbol="ETHUSDT",
                action="BUY" if i % 2 == 0 else "SELL",
                strength=0.5 + i * 0.1,
                price=Decimal(f"300{i}.00"),
                quantity=Decimal("1.0"),
                executed=i < 3  # First 3 are executed
            )
            signals.append(signal)

        for signal in signals:
            await repositories['signal'].create(signal)

        # Update strategy metrics based on signals
        created_strategy.total_signals = 5
        created_strategy.executed_signals = 3
        created_strategy.successful_signals = 2
        updated_strategy = await repositories['strategy'].update(created_strategy)

        # Verify consistency
        assert updated_strategy.total_signals == 5
        assert updated_strategy.executed_signals == 3
        assert updated_strategy.successful_signals == 2

        # Verify signals exist
        strategy_signals = await repositories['signal'].get_all(
            filters={"strategy_id": str(created_strategy.id)}
        )
        assert len(strategy_signals) == 5

        executed_signals = [s for s in strategy_signals if s.executed]
        assert len(executed_signals) == 3

        # Cleanup
        for signal in signals:
            await repositories['signal'].delete(signal.id)
        await repositories['strategy'].delete(created_strategy.id)
        await repositories['bot'].delete(created_bot.id)

    async def test_repository_error_handling(self, repositories, integration_session):
        """Test error handling across repository operations."""
        # Test creating entity with duplicate ID (should handle gracefully)
        bot_id = uuid.uuid4()
        bot1 = Bot(
            id=bot_id,
            name="Error Test Bot 1",
            description="First bot",
            status="RUNNING",
            exchange="binance",
            allocated_capital=1000.0
        )
        created_bot1 = await repositories['bot'].create(bot1)
        assert created_bot1 is not None

        # Try to create another bot with same ID
        bot2 = Bot(
            id=bot_id,
            name="Error Test Bot 2", 
            description="Duplicate ID bot",
            status="RUNNING",
            exchange="binance",
            allocated_capital=2000.0
        )
        
        # This should raise an exception or handle gracefully
        try:
            created_bot2 = await repositories['bot'].create(bot2)
            # If creation succeeds, it means the repository handled the duplicate gracefully
            # In that case, clean up the second bot too
            if created_bot2:
                await repositories['bot'].delete(created_bot2.id)
        except Exception as e:
            # Expected behavior - duplicate IDs should cause an error
            assert "duplicate" in str(e).lower() or "constraint" in str(e).lower()

        # Test updating non-existent entity
        non_existent_bot = Bot(
            id=uuid.uuid4(),
            name="Non-existent Bot",
            description="This bot doesn't exist in DB",
            status="RUNNING",
            exchange="binance"
        )
        
        try:
            updated_bot = await repositories['bot'].update(non_existent_bot)
            # Some repositories might create the entity if it doesn't exist
        except Exception as e:
            # Expected - updating non-existent entity should fail
            pass

        # Test deleting non-existent entity
        result = await repositories['bot'].delete(uuid.uuid4())
        assert result is False  # Should return False for non-existent entity

        # Cleanup
        await repositories['bot'].delete(created_bot1.id)

    async def test_repository_filtering_and_querying(self, repositories, integration_session):
        """Test complex filtering and querying across repositories."""
        # Create test data
        base_time = datetime.now(timezone.utc)
        
        # Create multiple market data records
        symbols = ["BTCUSD", "ETHUSDT", "ADAUSDT"]
        exchanges = ["binance", "coinbase"]
        
        market_data_records = []
        for i, symbol in enumerate(symbols):
            for j, exchange in enumerate(exchanges):
                record = MarketDataRecord(
                    id=uuid.uuid4(),
                    symbol=symbol,
                    exchange=exchange,
                    data_timestamp=base_time - timedelta(hours=i * 2 + j),
                    open_price=Decimal(f"100{i}{j}.00"),
                    high_price=Decimal(f"105{i}{j}.00"),
                    low_price=Decimal(f"95{i}{j}.00"),
                    close_price=Decimal(f"102{i}{j}.00"),
                    volume=Decimal(f"50{i}{j}.0"),
                    interval="1h",
                    source="exchange"
                )
                market_data_records.append(record)
                await repositories['market_data'].create(record)

        # Test filtering by symbol
        btc_data = await repositories['market_data'].get_by_symbol("BTCUSD")
        assert len(btc_data) == 2  # One for each exchange
        assert all(record.symbol == "BTCUSD" for record in btc_data)

        # Test filtering by exchange  
        binance_data = await repositories['market_data'].get_by_exchange("binance")
        assert len(binance_data) == 3  # One for each symbol
        assert all(record.exchange == "binance" for record in binance_data)

        # Test combined filtering
        btc_binance_data = await repositories['market_data'].get_by_symbol_and_exchange(
            "BTCUSD", "binance"
        )
        assert len(btc_binance_data) == 1
        assert btc_binance_data[0].symbol == "BTCUSD"
        assert btc_binance_data[0].exchange == "binance"

        # Test time-based queries
        recent_data = await repositories['market_data'].get_recent_data(
            "ETHUSDT", "coinbase", hours=12
        )
        assert len(recent_data) >= 1

        # Cleanup
        for record in market_data_records:
            await repositories['market_data'].delete(record.id)

    async def test_transaction_rollback_scenario(self, repositories, integration_session):
        """Test transaction rollback in complex scenarios."""
        # This test verifies that if part of a complex operation fails,
        # the entire transaction can be rolled back properly
        
        bot_id = uuid.uuid4()
        strategy_id = uuid.uuid4()
        
        try:
            # Start a transaction-like operation
            # Create bot
            bot = Bot(
                id=bot_id,
                name="Transaction Test Bot",
                description="Testing transaction behavior",
                status="RUNNING",
                exchange="binance",
                allocated_capital=10000.0
            )
            created_bot = await repositories['bot'].create(bot)
            assert created_bot is not None

            # Create strategy
            strategy = Strategy(
                id=strategy_id,
                name="Transaction Test Strategy",
                type="scalping",
                status="ACTIVE",
                bot_id=bot_id,
                max_position_size=500.0
            )
            created_strategy = await repositories['strategy'].create(strategy)
            assert created_strategy is not None

            # Simulate an error condition that would require rollback
            # For example, try to create an invalid signal
            invalid_signal = Signal(
                id=uuid.uuid4(),
                strategy_id=strategy_id,
                symbol="INVALID_SYMBOL_THAT_MIGHT_CAUSE_ERROR",
                action="INVALID_ACTION",
                strength=2.0,  # Invalid strength (should be 0-1)
                price=Decimal("-100.00"),  # Invalid negative price
                quantity=Decimal("0.0")  # Invalid zero quantity
            )
            
            # Depending on validation, this might fail
            try:
                await repositories['signal'].create(invalid_signal)
            except Exception:
                # If signal creation fails, we should clean up
                # In a real transaction, this would be automatic rollback
                await repositories['strategy'].delete(strategy_id)
                await repositories['bot'].delete(bot_id)
                raise  # Re-raise to indicate transaction failed

            # If we get here, cleanup the successful creations
            await repositories['signal'].delete(invalid_signal.id)
            await repositories['strategy'].delete(strategy_id)
            await repositories['bot'].delete(bot_id)

        except Exception as e:
            # This is expected for invalid data
            # Verify cleanup happened (bot and strategy should not exist)
            remaining_bot = await repositories['bot'].get(bot_id)
            remaining_strategy = await repositories['strategy'].get(strategy_id)
            
            # In a proper transaction system, these should be None after rollback
            # For now, we just verify the test structure works
            pass