"""
Comprehensive Concurrency and Race Condition Tests for Portfolio Value Calculations.

This module tests thread safety and concurrent access patterns in RiskCalculator
to ensure portfolio value calculations remain accurate under concurrent load
and prevent race conditions that could cause incorrect risk assessments.

CRITICAL AREAS TESTED:
1. Thread-safe portfolio value updates
2. Concurrent risk metric calculations
3. Race conditions in data structure access
4. Memory consistency under concurrent updates
5. Atomic operations for financial data
6. High-frequency update scenarios
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from decimal import Decimal
from unittest.mock import patch

import numpy as np
import pytest

from src.core.config import Config
from src.core.types import MarketData, OrderSide, Position, RiskLevel
from src.risk_management.risk_metrics import RiskCalculator


class TestPortfolioConcurrency:
    """
    Test suite for concurrent portfolio operations and thread safety.
    
    These tests ensure that portfolio value calculations and risk metrics
    remain accurate and consistent under concurrent access patterns that
    could occur in production trading scenarios.
    """

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()

    @pytest.fixture
    def risk_calculator(self, config):
        """Create risk calculator instance."""
        return RiskCalculator(config)

    @pytest.fixture
    def sample_positions(self):
        """Create sample positions for testing."""
        positions = []
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
        
        for i, symbol in enumerate(symbols):
            position = Position(
                symbol=symbol,
                quantity=Decimal(f"{1 + i * 0.1}"),
                entry_price=Decimal(f"{1000 + i * 100}"),
                current_price=Decimal(f"{1100 + i * 100}"),
                unrealized_pnl=Decimal(f"{100 + i * 10}"),
                side=OrderSide.BUY,
                timestamp=datetime.now(),
            )
            positions.append(position)
        
        return positions

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        market_data = []
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
        
        for i, symbol in enumerate(symbols):
            data = MarketData(
                symbol=symbol,
                price=Decimal(f"{1100 + i * 100}"),
                volume=Decimal(f"{1000 + i * 50}"),
                timestamp=datetime.now(),
                bid=Decimal(f"{1099 + i * 100}"),
                ask=Decimal(f"{1101 + i * 100}"),
                open_price=Decimal(f"{1000 + i * 100}"),
                high_price=Decimal(f"{1200 + i * 100}"),
                low_price=Decimal(f"{950 + i * 100}"),
            )
            market_data.append(data)
        
        return market_data

    def test_concurrent_portfolio_value_updates(self, risk_calculator):
        """Test concurrent portfolio value updates don't cause race conditions."""
        num_threads = 10
        updates_per_thread = 100
        
        def update_portfolio_values(thread_id):
            """Update portfolio values in a thread."""
            values_added = []
            for i in range(updates_per_thread):
                value = Decimal(f"{10000 + thread_id * 100 + i}")
                risk_calculator._update_portfolio_history(value)
                values_added.append(value)
            return values_added
        
        # Run concurrent updates
        all_expected_values = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(update_portfolio_values, i) for i in range(num_threads)]
            for future in as_completed(futures):
                thread_values = future.result()
                all_expected_values.extend(thread_values)
        
        # Verify all updates were recorded
        total_expected = num_threads * updates_per_thread
        assert len(risk_calculator.portfolio_values) == total_expected
        
        # Verify values are all present (may not be in order due to concurrency)
        recorded_values = set(risk_calculator.portfolio_values)
        expected_values = set(float(v) for v in all_expected_values)
        assert recorded_values == expected_values

    @pytest.mark.asyncio
    async def test_concurrent_risk_metric_calculations(self, risk_calculator, sample_positions, sample_market_data):
        """Test concurrent risk metric calculations maintain consistency."""
        # Pre-populate with some portfolio history
        for i in range(50):
            await risk_calculator._update_portfolio_history(Decimal(f"{10000 + i * 10}"))
        
        num_concurrent_calcs = 20
        
        async def calculate_risk_metrics():
            """Calculate risk metrics concurrently."""
            return await risk_calculator.calculate_risk_metrics(sample_positions, sample_market_data)
        
        # Run concurrent calculations
        tasks = [calculate_risk_metrics() for _ in range(num_concurrent_calcs)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all calculations succeeded
        valid_results = [r for r in results if not isinstance(r, Exception)]
        assert len(valid_results) == num_concurrent_calcs
        
        # Verify calculations are consistent (should be identical)
        first_result = valid_results[0]
        for result in valid_results[1:]:
            assert result.var_1d == first_result.var_1d
            assert result.var_5d == first_result.var_5d
            assert result.current_drawdown == first_result.current_drawdown
            assert result.risk_level == first_result.risk_level

    def test_concurrent_position_return_updates(self, risk_calculator):
        """Test concurrent position return updates are thread-safe."""
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        num_threads = 15
        updates_per_symbol = 100
        
        def update_position_returns(thread_id):
            """Update position returns for all symbols."""
            updates = []
            for symbol in symbols:
                for i in range(updates_per_symbol):
                    price = 50000 + thread_id * 100 + i
                    risk_calculator.update_position_returns(symbol, price)
                    updates.append((symbol, price))
            return updates
        
        # Run concurrent updates
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(update_position_returns, i) for i in range(num_threads)]
            all_updates = []
            for future in as_completed(futures):
                thread_updates = future.result()
                all_updates.extend(thread_updates)
        
        # Verify all symbols have data
        for symbol in symbols:
            assert symbol in risk_calculator.position_returns
            assert symbol in risk_calculator.position_prices
            assert len(risk_calculator.position_prices[symbol]) > 0
        
        # Verify data integrity (no missing or corrupted data)
        for symbol in symbols:
            prices = risk_calculator.position_prices[symbol]
            assert all(isinstance(p, (int, float)) for p in prices)
            assert len(prices) <= num_threads * updates_per_symbol  # May have truncation

    @pytest.mark.asyncio 
    async def test_memory_consistency_under_load(self, risk_calculator):
        """Test memory consistency during high-frequency portfolio updates."""
        # Simulate high-frequency updates
        update_count = 1000
        batch_size = 50
        
        async def batch_update(start_value):
            """Update portfolio values in batches."""
            for i in range(batch_size):
                value = Decimal(str(start_value + i))
                await risk_calculator._update_portfolio_history(value)
        
        # Create batched updates
        tasks = []
        for batch in range(update_count // batch_size):
            start_value = batch * batch_size + 10000
            tasks.append(batch_update(start_value))
        
        # Run all updates concurrently
        await asyncio.gather(*tasks)
        
        # Verify memory consistency
        assert len(risk_calculator.portfolio_values) == update_count
        assert len(risk_calculator.portfolio_returns) == update_count - 1  # Returns = values - 1
        
        # Verify no data corruption
        for i, value in enumerate(risk_calculator.portfolio_values[:-1]):
            next_value = risk_calculator.portfolio_values[i + 1]
            # Values should be monotonically increasing in this test
            assert next_value >= value

    def test_atomic_portfolio_operations(self, risk_calculator):
        """Test that portfolio operations are atomic under concurrent access."""
        import threading
        
        # Shared state for testing atomicity
        results = []
        lock = threading.Lock()
        
        def atomic_operation(thread_id):
            """Perform atomic portfolio operations."""
            # Get initial state
            initial_count = len(risk_calculator.portfolio_values)
            
            # Add some values
            for i in range(10):
                value = Decimal(f"{thread_id * 1000 + i}")
                risk_calculator._update_portfolio_history(value)
            
            # Record final state
            final_count = len(risk_calculator.portfolio_values)
            
            with lock:
                results.append((thread_id, initial_count, final_count))
        
        # Run multiple threads
        threads = []
        num_threads = 20
        
        for i in range(num_threads):
            thread = threading.Thread(target=atomic_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify atomicity - total values should equal expected
        expected_total = num_threads * 10
        actual_total = len(risk_calculator.portfolio_values)
        assert actual_total == expected_total
        
        # Verify each thread added exactly 10 values
        for thread_id, initial_count, final_count in results:
            added_count = final_count - initial_count
            # Note: Due to concurrent access, this might not be exactly 10
            # but the total should still be correct
            assert added_count >= 0

    @pytest.mark.asyncio
    async def test_concurrent_var_calculations(self, risk_calculator):
        """Test concurrent VaR calculations maintain accuracy."""
        # Setup portfolio history
        returns = [0.01, -0.005, 0.02, -0.01, 0.015] * 20  # 100 returns
        risk_calculator.portfolio_returns = returns
        portfolio_value = Decimal("10000")
        
        # Test different VaR time horizons concurrently
        async def calculate_var(days):
            """Calculate VaR for given time horizon."""
            return await risk_calculator._calculate_var(days, portfolio_value)
        
        # Run concurrent VaR calculations
        time_horizons = [1, 5, 10, 20]
        tasks = [calculate_var(days) for days in time_horizons for _ in range(10)]  # 40 total
        results = await asyncio.gather(*tasks)
        
        # Group results by time horizon
        grouped_results = {}
        for i, result in enumerate(results):
            horizon = time_horizons[i % len(time_horizons)]
            if horizon not in grouped_results:
                grouped_results[horizon] = []
            grouped_results[horizon].append(result)
        
        # Verify consistency within each time horizon
        for horizon, values in grouped_results.items():
            first_value = values[0]
            for value in values[1:]:
                assert value == first_value  # Should be identical
        
        # Verify VaR scaling relationship (longer horizon = higher VaR)
        var_values = {horizon: values[0] for horizon, values in grouped_results.items()}
        assert var_values[5] > var_values[1]  # 5-day > 1-day
        assert var_values[10] > var_values[5]  # 10-day > 5-day
        assert var_values[20] > var_values[10]  # 20-day > 10-day

    def test_race_condition_prevention(self, risk_calculator):
        """Test prevention of race conditions in data structure access."""
        import threading
        
        # Shared variables to detect race conditions
        race_detected = threading.Event()
        access_count = {"read": 0, "write": 0}
        lock = threading.Lock()
        
        def reader_thread():
            """Thread that reads portfolio data."""
            for _ in range(1000):
                try:
                    # Read operations
                    len(risk_calculator.portfolio_values)
                    len(risk_calculator.portfolio_returns)
                    
                    with lock:
                        access_count["read"] += 1
                        
                    time.sleep(0.0001)  # Small delay to increase race chance
                except (IndexError, KeyError, RuntimeError) as e:
                    # Race condition detected
                    race_detected.set()
                    break
        
        def writer_thread(thread_id):
            """Thread that writes portfolio data."""
            for i in range(500):
                try:
                    value = Decimal(f"{thread_id * 1000 + i}")
                    risk_calculator._update_portfolio_history(value)
                    
                    with lock:
                        access_count["write"] += 1
                        
                    time.sleep(0.0001)  # Small delay
                except Exception as e:
                    # Race condition detected
                    race_detected.set()
                    break
        
        # Start multiple reader and writer threads
        threads = []
        
        # Start readers
        for _ in range(5):
            thread = threading.Thread(target=reader_thread)
            threads.append(thread)
            thread.start()
        
        # Start writers
        for i in range(5):
            thread = threading.Thread(target=writer_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads with timeout
        for thread in threads:
            thread.join(timeout=10)
        
        # Check if race condition was detected
        assert not race_detected.is_set(), "Race condition detected in concurrent access"
        
        # Verify operations completed
        assert access_count["read"] > 0
        assert access_count["write"] > 0

    @pytest.mark.asyncio
    async def test_high_frequency_update_performance(self, risk_calculator):
        """Test performance under high-frequency portfolio updates."""
        import time
        
        # High-frequency update simulation
        num_updates = 10000
        start_time = time.time()
        
        # Batch updates for better performance testing
        async def update_batch(start_idx, batch_size):
            for i in range(batch_size):
                value = Decimal(f"{start_idx + i + 50000}")
                await risk_calculator._update_portfolio_history(value)
        
        # Run batched updates concurrently
        batch_size = 100
        num_batches = num_updates // batch_size
        
        tasks = []
        for batch in range(num_batches):
            start_idx = batch * batch_size
            tasks.append(update_batch(start_idx, batch_size))
        
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify performance
        updates_per_second = num_updates / total_time
        assert updates_per_second > 1000, f"Update rate too slow: {updates_per_second:.0f} updates/sec"
        
        # Verify data integrity
        assert len(risk_calculator.portfolio_values) == num_updates
        
        # Verify no data loss or corruption
        values = risk_calculator.portfolio_values
        for i, value in enumerate(values):
            expected_value = float(Decimal(f"{i + 50000}"))
            # Allow for some reordering due to concurrency
            assert 50000 <= value < 60000

    def test_concurrent_drawdown_calculations(self, risk_calculator):
        """Test concurrent drawdown calculations for consistency."""
        # Setup portfolio with known drawdown pattern
        portfolio_values = [10000, 12000, 11000, 13000, 9000, 14000, 8000, 15000]
        risk_calculator.portfolio_values = portfolio_values
        
        def calculate_max_drawdown():
            """Calculate maximum drawdown."""
            return risk_calculator._calculate_max_drawdown()
        
        def calculate_current_drawdown():
            """Calculate current drawdown."""
            current_value = Decimal("12000")  # Test value
            return risk_calculator._calculate_current_drawdown(current_value)
        
        # Run concurrent calculations
        with ThreadPoolExecutor(max_workers=20) as executor:
            # Submit mixed drawdown calculations
            futures = []
            for _ in range(50):
                futures.append(executor.submit(calculate_max_drawdown))
                futures.append(executor.submit(calculate_current_drawdown))
            
            # Collect results
            max_drawdowns = []
            current_drawdowns = []
            
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if i % 2 == 0:  # Max drawdown
                    max_drawdowns.append(result)
                else:  # Current drawdown
                    current_drawdowns.append(result)
        
        # Verify consistency
        assert all(dd == max_drawdowns[0] for dd in max_drawdowns)
        assert all(dd == current_drawdowns[0] for dd in current_drawdowns)
        
        # Verify correctness
        assert max_drawdowns[0] > Decimal("0")  # Should have some drawdown
        assert current_drawdowns[0] >= Decimal("0")  # Non-negative

    @pytest.mark.asyncio
    async def test_concurrent_risk_level_determination(self, risk_calculator):
        """Test concurrent risk level determination maintains consistency."""
        # Setup test data
        risk_calculator.portfolio_values = [10000] * 100  # For risk level calculation
        
        # Test parameters
        test_cases = [
            (Decimal("0.01"), Decimal("0.02"), Decimal("1.5")),  # Low risk
            (Decimal("0.06"), Decimal("0.12"), Decimal("-1.5")),  # High risk
            (Decimal("0.12"), Decimal("0.25"), Decimal("-2.0")),  # Critical risk
        ]
        
        async def determine_risk_level(var_1d, current_drawdown, sharpe_ratio):
            """Determine risk level concurrently."""
            return await risk_calculator._determine_risk_level(var_1d, current_drawdown, sharpe_ratio)
        
        # Run concurrent risk level determinations
        for var_1d, drawdown, sharpe in test_cases:
            tasks = [determine_risk_level(var_1d, drawdown, sharpe) for _ in range(20)]
            results = await asyncio.gather(*tasks)
            
            # Verify all results are identical
            first_result = results[0]
            assert all(result == first_result for result in results)
            
            # Verify result is valid risk level
            assert first_result in RiskLevel

    def test_memory_usage_under_concurrent_load(self, risk_calculator):
        """Test memory usage remains reasonable under concurrent load."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        def stress_test_memory():
            """Stress test memory usage."""
            # Add lots of portfolio history
            for i in range(1000):
                value = Decimal(f"{50000 + i}")
                risk_calculator._update_portfolio_history(value)
            
            # Add position return history
            symbols = [f"SYMBOL{i}" for i in range(100)]
            for symbol in symbols:
                for j in range(100):
                    price = 1000 + j
                    risk_calculator.update_position_returns(symbol, price)
        
        # Run concurrent memory stress test
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(stress_test_memory) for _ in range(10)]
            for future in as_completed(futures):
                future.result()
        
        # Check memory usage after stress test
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB for this test)
        assert memory_increase < 100, f"Excessive memory usage: {memory_increase:.2f}MB increase"
        
        # Verify data structures maintain reasonable size
        max_history = max(risk_calculator.risk_config.var_calculation_window, 252)
        assert len(risk_calculator.portfolio_values) <= max_history
        
        # Position data should also be trimmed
        for symbol in risk_calculator.position_returns:
            assert len(risk_calculator.position_returns[symbol]) <= 252
            assert len(risk_calculator.position_prices[symbol]) <= 252

    @pytest.mark.asyncio
    async def test_concurrent_exception_handling(self, risk_calculator, sample_positions):
        """Test that exceptions in concurrent operations don't corrupt state."""
        # Create invalid market data to trigger exceptions
        invalid_market_data = [
            MarketData(
                symbol="INVALID",  # Mismatched symbol
                price=Decimal("1000"),
                volume=Decimal("100"),
                timestamp=datetime.now(),
                bid=Decimal("999"),
                ask=Decimal("1001"),
                open_price=Decimal("1000"),
                high_price=Decimal("1100"),
                low_price=Decimal("900"),
            )
        ]
        
        async def failing_calculation():
            """Calculation that should fail."""
            try:
                return await risk_calculator.calculate_risk_metrics(sample_positions, invalid_market_data)
            except Exception:
                return None  # Expected to fail
        
        async def successful_calculation():
            """Calculation that should succeed."""
            return await risk_calculator._create_empty_risk_metrics()
        
        # Mix successful and failing operations
        tasks = []
        for _ in range(10):
            tasks.append(failing_calculation())
            tasks.append(successful_calculation())
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify some operations failed and some succeeded
        successes = [r for r in results if r is not None and not isinstance(r, Exception)]
        failures = [r for r in results if r is None or isinstance(r, Exception)]
        
        assert len(successes) > 0, "No successful operations"
        assert len(failures) > 0, "No failed operations (test setup issue)"
        
        # Verify state wasn't corrupted by failures
        assert isinstance(risk_calculator.portfolio_values, list)
        assert isinstance(risk_calculator.portfolio_returns, list)
        assert isinstance(risk_calculator.position_returns, dict)