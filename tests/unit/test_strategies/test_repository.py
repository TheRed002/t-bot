"""
Tests for strategy repository module.
"""

from abc import ABC
from datetime import datetime, timezone, timedelta
from typing import Any
from uuid import uuid4

import pytest

from src.strategies.repository import StrategyRepositoryInterface


class MockStrategyRepository(StrategyRepositoryInterface):
    """Mock implementation of StrategyRepositoryInterface for testing."""

    def __init__(self):
        self._strategies = {}
        self._states = {}
        self._metrics = {}
        self._signals = {}
        self._trades = {}

    # Implement the interface methods
    async def create_strategy(self, strategy_data) -> Any:
        strategy_id = getattr(strategy_data, 'id', strategy_data.get("strategy_id") if isinstance(strategy_data, dict) else None)
        if not strategy_id:
            raise ValueError("strategy_id is required")

        if strategy_id in self._strategies:
            raise ValueError(f"Strategy {strategy_id} already exists")

        record = strategy_data if not isinstance(strategy_data, dict) else {
            **strategy_data,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        self._strategies[strategy_id] = record
        return record

    async def update_strategy(self, strategy_id: str, updates: dict[str, Any]) -> Any:
        if strategy_id not in self._strategies:
            return None

        if isinstance(self._strategies[strategy_id], dict):
            self._strategies[strategy_id].update(updates)
            self._strategies[strategy_id]["updated_at"] = datetime.now(timezone.utc)
        else:
            # Handle object case
            for key, value in updates.items():
                if hasattr(self._strategies[strategy_id], key):
                    setattr(self._strategies[strategy_id], key, value)
        return self._strategies[strategy_id]

    async def get_strategy(self, strategy_id: str) -> Any:
        return self._strategies.get(strategy_id)

    async def delete_strategy(self, strategy_id: str) -> bool:
        if strategy_id in self._strategies:
            del self._strategies[strategy_id]
            return True
        return False

    async def get_strategies_by_bot(self, bot_id: str) -> list[Any]:
        return [s for s in self._strategies.values() if getattr(s, 'bot_id', s.get('bot_id') if isinstance(s, dict) else None) == bot_id]

    async def get_active_strategies(self, bot_id: str | None = None) -> list[Any]:
        active = [s for s in self._strategies.values() if getattr(s, 'enabled', s.get('enabled', True) if isinstance(s, dict) else True)]
        if bot_id:
            active = [s for s in active if getattr(s, 'bot_id', s.get('bot_id') if isinstance(s, dict) else None) == bot_id]
        return active

    async def save_strategy_state(self, strategy_id: str, state_data: dict[str, Any]) -> bool:
        self._states[strategy_id] = {
            'data': state_data,
            'timestamp': datetime.now(timezone.utc)
        }
        return True

    async def load_strategy_state(self, strategy_id: str) -> dict[str, Any] | None:
        state = self._states.get(strategy_id)
        return state['data'] if state else None

    async def save_strategy_metrics(self, strategy_id: str, metrics) -> bool:
        if strategy_id not in self._metrics:
            self._metrics[strategy_id] = []
        self._metrics[strategy_id].append({
            'metrics': metrics,
            'timestamp': datetime.now(timezone.utc)
        })
        return True

    async def get_strategy_metrics(self, strategy_id: str, start_time: datetime | None = None, end_time: datetime | None = None) -> list[Any]:
        metrics_list = self._metrics.get(strategy_id, [])
        if start_time or end_time:
            filtered = []
            for m in metrics_list:
                ts = m['timestamp']
                if start_time and ts < start_time:
                    continue
                if end_time and ts > end_time:
                    continue
                filtered.append(m['metrics'])
            return filtered
        return [m['metrics'] for m in metrics_list]

    async def save_strategy_signals(self, signals: list[Any]) -> list[Any]:
        for signal in signals:
            strategy_id = getattr(signal, 'strategy_id', signal.get('strategy_id') if isinstance(signal, dict) else 'unknown')
            if strategy_id not in self._signals:
                self._signals[strategy_id] = []
            self._signals[strategy_id].append(signal)
        return signals

    async def get_strategy_signals(self, strategy_id: str, limit: int | None = None) -> list[Any]:
        signals = self._signals.get(strategy_id, [])
        if limit:
            return signals[-limit:]
        return signals

    async def save_strategy_trades(self, trades: list[Any]) -> list[Any]:
        for trade in trades:
            strategy_id = getattr(trade, 'strategy_id', trade.get('strategy_id') if isinstance(trade, dict) else 'unknown')
            if strategy_id not in self._trades:
                self._trades[strategy_id] = []
            self._trades[strategy_id].append(trade)
        return trades

    async def get_strategy_trades(self, strategy_id: str, limit: int | None = None) -> list[Any]:
        trades = self._trades.get(strategy_id, [])
        if limit:
            return trades[-limit:]
        return trades

    # Legacy methods for backward compatibility
    async def create_strategy_record(self, strategy_data: dict[str, Any]) -> dict[str, Any]:
        return await self.create_strategy(strategy_data)

    async def update_strategy_record(self, strategy_id: str, updates: dict[str, Any]) -> bool:
        result = await self.update_strategy(strategy_id, updates)
        return result is not None

    async def get_strategy_record(self, strategy_id: str) -> dict[str, Any] | None:
        return await self.get_strategy(strategy_id)

    async def get_strategies_by_criteria(
        self, criteria: dict[str, Any], limit: int | None = None, offset: int | None = None
    ) -> list[dict[str, Any]]:
        results = []
        for strategy_id, strategy in self._strategies.items():
            match = True
            for key, value in criteria.items():
                if strategy.get(key) != value:
                    match = False
                    break
            if match:
                results.append(strategy)

        # Apply offset and limit
        if offset:
            results = results[offset:]
        if limit:
            results = results[:limit]

        return results

    async def delete_strategy_record(self, strategy_id: str) -> bool:
        if strategy_id in self._strategies:
            del self._strategies[strategy_id]
            return True
        return False

    async def save_strategy_state(self, strategy_id: str, state_data: dict[str, Any]) -> bool:
        self._states[strategy_id] = {**state_data, "saved_at": datetime.now(timezone.utc)}
        return True

    async def load_strategy_state(self, strategy_id: str) -> dict[str, Any] | None:
        return self._states.get(strategy_id)

    async def save_strategy_metrics(
        self, strategy_id: str, metrics: dict[str, Any], timestamp: datetime | None = None
    ) -> bool:
        if strategy_id not in self._metrics:
            self._metrics[strategy_id] = []

        metrics_record = {**metrics, "timestamp": timestamp or datetime.now(timezone.utc)}
        self._metrics[strategy_id].append(metrics_record)
        return True

    async def load_strategy_metrics(
        self,
        strategy_id: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        if strategy_id not in self._metrics:
            return []

        metrics = self._metrics[strategy_id]

        # Apply time filters
        if start_time or end_time:
            filtered_metrics = []
            for metric in metrics:
                timestamp = metric.get("timestamp")
                if not timestamp:
                    continue

                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue

                filtered_metrics.append(metric)
            return filtered_metrics

        return metrics

    async def save_strategy_signals(self, strategy_id: str, signals: list[dict[str, Any]]) -> bool:
        if strategy_id not in self._signals:
            self._signals[strategy_id] = []

        for signal in signals:
            signal_record = {**signal, "saved_at": datetime.now(timezone.utc)}
            self._signals[strategy_id].append(signal_record)

        return True

    async def load_strategy_signals(
        self,
        strategy_id: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        if strategy_id not in self._signals:
            return []

        signals = self._signals[strategy_id]

        # Apply time filters
        if start_time or end_time:
            filtered_signals = []
            for signal in signals:
                timestamp = signal.get("timestamp")
                if not timestamp:
                    continue

                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue

                filtered_signals.append(signal)
            return filtered_signals

        return signals

    async def save_strategy_trades(self, strategy_id: str, trades: list[dict[str, Any]]) -> bool:
        if strategy_id not in self._trades:
            self._trades[strategy_id] = []

        for trade in trades:
            trade_record = {**trade, "saved_at": datetime.now(timezone.utc)}
            self._trades[strategy_id].append(trade_record)

        return True

    async def load_strategy_trades(
        self,
        strategy_id: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        if strategy_id not in self._trades:
            return []

        trades = self._trades[strategy_id]

        # Apply time filters
        if start_time or end_time:
            filtered_trades = []
            for trade in trades:
                timestamp = trade.get("timestamp")
                if not timestamp:
                    continue

                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue

                filtered_trades.append(trade)
            return filtered_trades

        return trades


class TestStrategyRepositoryInterface:
    """Test StrategyRepositoryInterface abstract interface."""

    def test_repository_interface_is_abstract(self):
        """Test that StrategyRepositoryInterface cannot be instantiated."""
        with pytest.raises(TypeError):
            StrategyRepositoryInterface()

    def test_repository_interface_inheritance(self):
        """Test that StrategyRepositoryInterface inherits from ABC."""
        assert issubclass(StrategyRepositoryInterface, ABC)

    def test_repository_interface_abstract_methods(self):
        """Test that interface has required abstract methods."""
        abstract_methods = StrategyRepositoryInterface.__abstractmethods__

        expected_methods = {
            "create_strategy",
            "get_strategy", 
            "update_strategy",
            "delete_strategy",
            "get_strategies_by_bot",
            "get_active_strategies",
            "save_strategy_state",
            "load_strategy_state",
            "save_strategy_metrics",
            "get_strategy_metrics",
            "save_strategy_signals",
            "get_strategy_signals",
        }

        # Check that core methods are present (allowing for implementation variations)
        for method in expected_methods:
            assert method in abstract_methods, f"Method {method} should be abstract"


class TestMockStrategyRepository:
    """Test MockStrategyRepository implementation."""

    @pytest.fixture
    def repository(self):
        """Create a mock repository instance."""
        return MockStrategyRepository()

    @pytest.fixture
    def sample_strategy_data(self):
        """Create sample strategy data."""
        return {
            "strategy_id": "test_strategy_001",
            "name": "Test Strategy",
            "strategy_type": "momentum",
            "enabled": True,
            "parameters": {"fast_ma": 20, "slow_ma": 50},
        }

    @pytest.mark.asyncio
    async def test_create_strategy_record(self, repository, sample_strategy_data):
        """Test creating strategy record."""
        record = await repository.create_strategy_record(sample_strategy_data)

        assert record is not None
        assert record["strategy_id"] == sample_strategy_data["strategy_id"]
        assert record["name"] == sample_strategy_data["name"]
        assert "created_at" in record
        assert "updated_at" in record

    @pytest.mark.asyncio
    async def test_create_strategy_record_duplicate(self, repository, sample_strategy_data):
        """Test creating duplicate strategy record fails."""
        await repository.create_strategy_record(sample_strategy_data)

        with pytest.raises(ValueError, match="already exists"):
            await repository.create_strategy_record(sample_strategy_data)

    @pytest.mark.asyncio
    async def test_create_strategy_record_missing_id(self, repository):
        """Test creating strategy record without ID fails."""
        invalid_data = {"name": "Test Strategy"}

        with pytest.raises(ValueError, match="strategy_id is required"):
            await repository.create_strategy_record(invalid_data)

    @pytest.mark.asyncio
    async def test_get_strategy_record(self, repository, sample_strategy_data):
        """Test getting strategy record."""
        await repository.create_strategy_record(sample_strategy_data)

        record = await repository.get_strategy_record(sample_strategy_data["strategy_id"])

        assert record is not None
        assert record["strategy_id"] == sample_strategy_data["strategy_id"]
        assert record["name"] == sample_strategy_data["name"]

    @pytest.mark.asyncio
    async def test_get_strategy_record_not_found(self, repository):
        """Test getting non-existent strategy record."""
        record = await repository.get_strategy_record("nonexistent")

        assert record is None

    @pytest.mark.asyncio
    async def test_update_strategy_record(self, repository, sample_strategy_data):
        """Test updating strategy record."""
        await repository.create_strategy_record(sample_strategy_data)

        updates = {"name": "Updated Strategy", "enabled": False}
        result = await repository.update_strategy_record(
            sample_strategy_data["strategy_id"], updates
        )

        assert result is True

        # Verify updates
        record = await repository.get_strategy_record(sample_strategy_data["strategy_id"])
        assert record["name"] == "Updated Strategy"
        assert record["enabled"] is False
        assert "updated_at" in record

    @pytest.mark.asyncio
    async def test_update_strategy_record_not_found(self, repository):
        """Test updating non-existent strategy record."""
        result = await repository.update_strategy_record("nonexistent", {"name": "Updated"})

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_strategy_record(self, repository, sample_strategy_data):
        """Test deleting strategy record."""
        await repository.create_strategy_record(sample_strategy_data)

        result = await repository.delete_strategy_record(sample_strategy_data["strategy_id"])

        assert result is True

        # Verify deletion
        record = await repository.get_strategy_record(sample_strategy_data["strategy_id"])
        assert record is None

    @pytest.mark.asyncio
    async def test_delete_strategy_record_not_found(self, repository):
        """Test deleting non-existent strategy record."""
        result = await repository.delete_strategy_record("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_strategies_by_criteria(self, repository):
        """Test getting strategies by criteria."""
        # Create multiple strategies
        strategies = [
            {
                "strategy_id": "strategy_001",
                "name": "Strategy 1",
                "strategy_type": "momentum",
                "enabled": True,
            },
            {
                "strategy_id": "strategy_002",
                "name": "Strategy 2",
                "strategy_type": "mean_reversion",
                "enabled": True,
            },
            {
                "strategy_id": "strategy_003",
                "name": "Strategy 3",
                "strategy_type": "momentum",
                "enabled": False,
            },
        ]

        for strategy in strategies:
            await repository.create_strategy_record(strategy)

        # Test criteria filtering
        momentum_strategies = await repository.get_strategies_by_criteria(
            {"strategy_type": "momentum"}
        )
        assert len(momentum_strategies) == 2

        enabled_strategies = await repository.get_strategies_by_criteria({"enabled": True})
        assert len(enabled_strategies) == 2

        # Test multiple criteria
        enabled_momentum = await repository.get_strategies_by_criteria(
            {"strategy_type": "momentum", "enabled": True}
        )
        assert len(enabled_momentum) == 1
        assert enabled_momentum[0]["strategy_id"] == "strategy_001"

    @pytest.mark.asyncio
    async def test_get_strategies_by_criteria_with_pagination(self, repository):
        """Test getting strategies with pagination."""
        # Create multiple strategies
        for i in range(10):
            strategy_data = {
                "strategy_id": f"strategy_{i:03d}",
                "name": f"Strategy {i}",
                "strategy_type": "momentum",
                "enabled": True,
            }
            await repository.create_strategy_record(strategy_data)

        # Test with limit
        limited_results = await repository.get_strategies_by_criteria(
            {"strategy_type": "momentum"}, limit=5
        )
        assert len(limited_results) == 5

        # Test with offset
        offset_results = await repository.get_strategies_by_criteria(
            {"strategy_type": "momentum"}, offset=5
        )
        assert len(offset_results) == 5

        # Test with both
        paginated_results = await repository.get_strategies_by_criteria(
            {"strategy_type": "momentum"}, limit=3, offset=2
        )
        assert len(paginated_results) == 3

    @pytest.mark.asyncio
    async def test_save_and_load_strategy_state(self, repository):
        """Test saving and loading strategy state."""
        strategy_id = "test_strategy_001"
        state_data = {
            "current_position": "long",
            "entry_price": 50000.0,
            "stop_loss": 49000.0,
            "indicators": {"rsi": 65.5, "macd": 0.125},
        }

        # Save state
        result = await repository.save_strategy_state(strategy_id, state_data)
        assert result is True

        # Load state
        loaded_state = await repository.load_strategy_state(strategy_id)

        assert loaded_state is not None
        assert loaded_state["current_position"] == "long"
        assert loaded_state["entry_price"] == 50000.0
        assert "saved_at" in loaded_state

    @pytest.mark.asyncio
    async def test_load_strategy_state_not_found(self, repository):
        """Test loading non-existent strategy state."""
        state = await repository.load_strategy_state("nonexistent")

        assert state is None

    @pytest.mark.asyncio
    async def test_save_and_load_strategy_metrics(self, repository):
        """Test saving and loading strategy metrics."""
        strategy_id = "test_strategy_001"

        # Save multiple metrics
        metrics1 = {"total_return": 15.5, "sharpe_ratio": 1.2, "max_drawdown": 5.3}

        metrics2 = {"total_return": 18.2, "sharpe_ratio": 1.4, "max_drawdown": 6.1}

        timestamp1 = datetime.now(timezone.utc)
        timestamp2 = datetime.now(timezone.utc)

        await repository.save_strategy_metrics(strategy_id, metrics1, timestamp1)
        await repository.save_strategy_metrics(strategy_id, metrics2, timestamp2)

        # Load all metrics
        loaded_metrics = await repository.load_strategy_metrics(strategy_id)

        assert len(loaded_metrics) == 2
        assert loaded_metrics[0]["total_return"] == 15.5
        assert loaded_metrics[1]["total_return"] == 18.2

    @pytest.mark.asyncio
    async def test_load_strategy_metrics_with_time_filter(self, repository):
        """Test loading strategy metrics with time filtering."""
        strategy_id = "test_strategy_001"

        now = datetime.now(timezone.utc)
        past = now - timedelta(hours=1)
        future = now + timedelta(hours=1)

        # Save metrics at different times
        await repository.save_strategy_metrics(strategy_id, {"value": 1}, past)
        await repository.save_strategy_metrics(strategy_id, {"value": 2}, now)
        await repository.save_strategy_metrics(strategy_id, {"value": 3}, future)

        # Test time filtering
        filtered_metrics = await repository.load_strategy_metrics(strategy_id, start_time=now)

        # Should include metrics at or after 'now'
        assert len(filtered_metrics) >= 1

    @pytest.mark.asyncio
    async def test_save_and_load_strategy_signals(self, repository):
        """Test saving and loading strategy signals."""
        strategy_id = "test_strategy_001"

        signals = [
            {
                "signal_id": "signal_001",
                "direction": "buy",
                "strength": 0.8,
                "symbol": "BTC/USD",
                "timestamp": datetime.now(timezone.utc),
            },
            {
                "signal_id": "signal_002",
                "direction": "sell",
                "strength": 0.6,
                "symbol": "BTC/USD",
                "timestamp": datetime.now(timezone.utc),
            },
        ]

        # Save signals
        result = await repository.save_strategy_signals(strategy_id, signals)
        assert result is True

        # Load signals
        loaded_signals = await repository.load_strategy_signals(strategy_id)

        assert len(loaded_signals) == 2
        assert loaded_signals[0]["signal_id"] == "signal_001"
        assert loaded_signals[1]["signal_id"] == "signal_002"
        assert "saved_at" in loaded_signals[0]

    @pytest.mark.asyncio
    async def test_save_and_load_strategy_trades(self, repository):
        """Test saving and loading strategy trades."""
        strategy_id = "test_strategy_001"

        trades = [
            {
                "trade_id": "trade_001",
                "side": "buy",
                "quantity": 0.1,
                "price": 50000.0,
                "pnl": 100.0,
                "timestamp": datetime.now(timezone.utc),
            },
            {
                "trade_id": "trade_002",
                "side": "sell",
                "quantity": 0.1,
                "price": 51000.0,
                "pnl": 100.0,
                "timestamp": datetime.now(timezone.utc),
            },
        ]

        # Save trades
        result = await repository.save_strategy_trades(strategy_id, trades)
        assert result is True

        # Load trades
        loaded_trades = await repository.load_strategy_trades(strategy_id)

        assert len(loaded_trades) == 2
        assert loaded_trades[0]["trade_id"] == "trade_001"
        assert loaded_trades[1]["trade_id"] == "trade_002"
        assert "saved_at" in loaded_trades[0]


class TestRepositoryEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.fixture
    def repository(self):
        """Create repository for testing."""
        return MockStrategyRepository()

    @pytest.mark.asyncio
    async def test_save_metrics_without_timestamp(self, repository):
        """Test saving metrics without explicit timestamp."""
        strategy_id = "test_strategy"
        metrics = {"total_return": 10.0}

        result = await repository.save_strategy_metrics(strategy_id, metrics)
        assert result is True

        loaded_metrics = await repository.load_strategy_metrics(strategy_id)
        assert len(loaded_metrics) == 1
        assert "timestamp" in loaded_metrics[0]

    @pytest.mark.asyncio
    async def test_load_metrics_nonexistent_strategy(self, repository):
        """Test loading metrics for non-existent strategy."""
        metrics = await repository.load_strategy_metrics("nonexistent")

        assert metrics == []

    @pytest.mark.asyncio
    async def test_load_signals_nonexistent_strategy(self, repository):
        """Test loading signals for non-existent strategy."""
        signals = await repository.load_strategy_signals("nonexistent")

        assert signals == []

    @pytest.mark.asyncio
    async def test_load_trades_nonexistent_strategy(self, repository):
        """Test loading trades for non-existent strategy."""
        trades = await repository.load_strategy_trades("nonexistent")

        assert trades == []

    @pytest.mark.asyncio
    async def test_empty_criteria_search(self, repository):
        """Test searching with empty criteria."""
        # Create a strategy
        strategy_data = {"strategy_id": "test_001", "name": "Test", "strategy_type": "momentum"}
        await repository.create_strategy_record(strategy_data)

        # Search with empty criteria should return all strategies
        results = await repository.get_strategies_by_criteria({})

        assert len(results) == 1
        assert results[0]["strategy_id"] == "test_001"

    @pytest.mark.asyncio
    async def test_save_empty_signals_list(self, repository):
        """Test saving empty signals list."""
        result = await repository.save_strategy_signals("test_strategy", [])

        assert result is True

        signals = await repository.load_strategy_signals("test_strategy")
        assert signals == []

    @pytest.mark.asyncio
    async def test_save_empty_trades_list(self, repository):
        """Test saving empty trades list."""
        result = await repository.save_strategy_trades("test_strategy", [])

        assert result is True

        trades = await repository.load_strategy_trades("test_strategy")
        assert trades == []

    @pytest.mark.asyncio
    async def test_update_strategy_record_partial(self, repository):
        """Test partial updates to strategy record."""
        # Create strategy
        original_data = {
            "strategy_id": "test_001",
            "name": "Original Name",
            "enabled": True,
            "parameters": {"param1": "value1"},
        }
        await repository.create_strategy_record(original_data)

        # Update only name
        updates = {"name": "Updated Name"}
        result = await repository.update_strategy_record("test_001", updates)

        assert result is True

        # Verify partial update
        record = await repository.get_strategy_record("test_001")
        assert record["name"] == "Updated Name"
        assert record["enabled"] is True  # Should remain unchanged
        assert record["parameters"] == {"param1": "value1"}  # Should remain unchanged


class TestRepositoryInterface:
    """Test repository interface compliance."""

    def test_mock_repository_implements_interface(self):
        """Test that MockStrategyRepository implements the interface."""
        repository = MockStrategyRepository()

        assert isinstance(repository, StrategyRepositoryInterface)

    def test_interface_method_signatures(self):
        """Test that interface method signatures are correct."""
        # This test verifies the interface methods exist with correct signatures
        interface_methods = [
            "create_strategy",
            "get_strategy",
            "update_strategy",
            "delete_strategy",
            "get_strategies_by_bot",
            "get_active_strategies",
            "save_strategy_state",
            "load_strategy_state",
            "save_strategy_metrics",
            "get_strategy_metrics",
            "save_strategy_signals",
            "get_strategy_signals",
        ]

        for method_name in interface_methods:
            assert hasattr(StrategyRepositoryInterface, method_name)
            method = getattr(StrategyRepositoryInterface, method_name)
            assert callable(method)
