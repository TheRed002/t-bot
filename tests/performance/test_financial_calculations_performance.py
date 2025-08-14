"""
Performance Benchmarks for High-Frequency Financial Calculations.

This module benchmarks critical financial calculations to ensure they meet
performance requirements for high-frequency trading operations and can handle
production-scale throughput without latency issues.

PERFORMANCE TARGETS:
1. Fee calculations: < 0.1ms per operation (10,000 ops/sec)
2. Kelly Criterion: < 1ms per calculation (1,000 calcs/sec)  
3. VaR calculations: < 5ms per calculation (200 calcs/sec)
4. Portfolio updates: < 0.05ms per update (20,000 updates/sec)
5. Risk metric calculations: < 10ms per full calculation (100 calcs/sec)
6. Stop-loss calculations: < 0.5ms per calculation (2,000 calcs/sec)
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from decimal import Decimal
from statistics import mean, median, stdev

import numpy as np
import pytest

from src.core.config import Config
from src.core.types import (
    MarketData, OrderRequest, OrderSide, OrderType, Position,
    PositionSizeMethod, Signal, SignalDirection, MarketRegime
)
from src.exchanges.binance_orders import BinanceOrderManager
from src.risk_management.adaptive_risk import AdaptiveRiskManager
from src.risk_management.position_sizing import PositionSizer
from src.risk_management.regime_detection import MarketRegimeDetector
from src.risk_management.risk_metrics import RiskCalculator


class TestFinancialCalculationsPerformance:
    """
    Performance benchmark suite for critical financial calculations.
    
    These benchmarks ensure all financial calculations can handle
    production-scale high-frequency trading operations.
    """

    @pytest.fixture(scope="class")
    def config(self):
        """Create test configuration."""
        return Config()

    @pytest.fixture(scope="class")
    def binance_order_manager(self, config):
        """Create BinanceOrderManager for performance testing."""
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        return BinanceOrderManager(config, mock_client)

    @pytest.fixture(scope="class")
    def position_sizer(self, config):
        """Create PositionSizer for performance testing."""
        sizer = PositionSizer(config)
        # Pre-populate with return history for Kelly calculations
        returns = [0.01, -0.005, 0.02, -0.01, 0.015] * 10  # 50 returns
        sizer.return_history["BTCUSDT"] = returns
        return sizer

    @pytest.fixture(scope="class") 
    def risk_calculator(self, config):
        """Create RiskCalculator for performance testing."""
        calculator = RiskCalculator(config)
        # Pre-populate with portfolio history
        for i in range(100):
            calculator.portfolio_returns.append(0.001 * (1 if i % 2 == 0 else -1))
        return calculator

    @pytest.fixture(scope="class")
    def adaptive_risk_manager(self, config):
        """Create AdaptiveRiskManager for performance testing."""
        detector_config = {"volatility_window": 20, "trend_window": 50}
        regime_detector = MarketRegimeDetector(detector_config)
        return AdaptiveRiskManager(config.__dict__, regime_detector)

    def benchmark_function(self, func, *args, iterations=1000, **kwargs):
        """
        Benchmark a function with statistical analysis.
        
        Returns:
            dict: Performance statistics
        """
        times = []
        
        # Warm-up runs
        for _ in range(10):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
        
        # Actual benchmark runs
        for _ in range(iterations):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to milliseconds
        
        return {
            "mean_ms": mean(times),
            "median_ms": median(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "std_ms": stdev(times) if len(times) > 1 else 0,
            "p95_ms": np.percentile(times, 95),
            "p99_ms": np.percentile(times, 99),
            "iterations": iterations,
            "ops_per_second": 1000 / mean(times),
        }

    async def async_benchmark_function(self, func, *args, iterations=1000, **kwargs):
        """
        Benchmark an async function with statistical analysis.
        
        Returns:
            dict: Performance statistics
        """
        times = []
        
        # Warm-up runs
        for _ in range(10):
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            end = time.perf_counter()
        
        # Actual benchmark runs
        for _ in range(iterations):
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to milliseconds
        
        return {
            "mean_ms": mean(times),
            "median_ms": median(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "std_ms": stdev(times) if len(times) > 1 else 0,
            "p95_ms": np.percentile(times, 95),
            "p99_ms": np.percentile(times, 99),
            "iterations": iterations,
            "ops_per_second": 1000 / mean(times),
        }

    def test_fee_calculation_performance(self, binance_order_manager):
        """Benchmark fee calculation performance (Target: < 0.1ms per operation)."""
        order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.12345678"),
        )
        fill_price = Decimal("50000.12345678")
        
        stats = self.benchmark_function(
            binance_order_manager.calculate_fees,
            order, fill_price,
            iterations=10000
        )
        
        print(f"\nFee Calculation Performance:")
        print(f"  Mean: {stats['mean_ms']:.3f}ms")
        print(f"  P95: {stats['p95_ms']:.3f}ms")  
        print(f"  P99: {stats['p99_ms']:.3f}ms")
        print(f"  Ops/sec: {stats['ops_per_second']:.0f}")
        
        # Performance assertion
        assert stats['mean_ms'] < 0.1, f"Fee calculation too slow: {stats['mean_ms']:.3f}ms (target: < 0.1ms)"
        assert stats['p95_ms'] < 0.2, f"P95 too slow: {stats['p95_ms']:.3f}ms (target: < 0.2ms)"
        assert stats['ops_per_second'] > 5000, f"Throughput too low: {stats['ops_per_second']:.0f} ops/sec"

    @pytest.mark.asyncio
    async def test_kelly_criterion_performance(self, position_sizer):
        """Benchmark Kelly Criterion calculation performance (Target: < 1ms per calculation)."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(),
            symbol="BTCUSDT", 
            strategy_name="test_strategy",
        )
        portfolio_value = Decimal("10000")
        
        stats = await self.async_benchmark_function(
            position_sizer._kelly_criterion_sizing,
            signal, portfolio_value,
            iterations=1000
        )
        
        print(f"\nKelly Criterion Performance:")
        print(f"  Mean: {stats['mean_ms']:.3f}ms")
        print(f"  P95: {stats['p95_ms']:.3f}ms")
        print(f"  P99: {stats['p99_ms']:.3f}ms") 
        print(f"  Ops/sec: {stats['ops_per_second']:.0f}")
        
        # Performance assertion
        assert stats['mean_ms'] < 1.0, f"Kelly calculation too slow: {stats['mean_ms']:.3f}ms (target: < 1ms)"
        assert stats['p95_ms'] < 2.0, f"P95 too slow: {stats['p95_ms']:.3f}ms (target: < 2ms)"
        assert stats['ops_per_second'] > 500, f"Throughput too low: {stats['ops_per_second']:.0f} ops/sec"

    @pytest.mark.asyncio
    async def test_var_calculation_performance(self, risk_calculator):
        """Benchmark VaR calculation performance (Target: < 5ms per calculation)."""
        portfolio_value = Decimal("10000")
        
        stats = await self.async_benchmark_function(
            risk_calculator._calculate_var,
            1, portfolio_value,
            iterations=500
        )
        
        print(f"\nVaR Calculation Performance:")
        print(f"  Mean: {stats['mean_ms']:.3f}ms")
        print(f"  P95: {stats['p95_ms']:.3f}ms")
        print(f"  P99: {stats['p99_ms']:.3f}ms")
        print(f"  Ops/sec: {stats['ops_per_second']:.0f}")
        
        # Performance assertion
        assert stats['mean_ms'] < 5.0, f"VaR calculation too slow: {stats['mean_ms']:.3f}ms (target: < 5ms)"
        assert stats['p95_ms'] < 10.0, f"P95 too slow: {stats['p95_ms']:.3f}ms (target: < 10ms)"
        assert stats['ops_per_second'] > 100, f"Throughput too low: {stats['ops_per_second']:.0f} ops/sec"

    @pytest.mark.asyncio  
    async def test_portfolio_update_performance(self, risk_calculator):
        """Benchmark portfolio update performance (Target: < 0.05ms per update)."""
        portfolio_value = Decimal("10000")
        
        stats = await self.async_benchmark_function(
            risk_calculator._update_portfolio_history,
            portfolio_value,
            iterations=20000
        )
        
        print(f"\nPortfolio Update Performance:")
        print(f"  Mean: {stats['mean_ms']:.3f}ms")
        print(f"  P95: {stats['p95_ms']:.3f}ms")
        print(f"  P99: {stats['p99_ms']:.3f}ms")
        print(f"  Ops/sec: {stats['ops_per_second']:.0f}")
        
        # Performance assertion
        assert stats['mean_ms'] < 0.05, f"Portfolio update too slow: {stats['mean_ms']:.3f}ms (target: < 0.05ms)"
        assert stats['p95_ms'] < 0.1, f"P95 too slow: {stats['p95_ms']:.3f}ms (target: < 0.1ms)"
        assert stats['ops_per_second'] > 10000, f"Throughput too low: {stats['ops_per_second']:.0f} ops/sec"

    @pytest.mark.asyncio
    async def test_full_risk_metrics_performance(self, risk_calculator):
        """Benchmark full risk metrics calculation performance (Target: < 10ms per calculation)."""
        # Setup test data
        position = Position(
            symbol="BTCUSDT",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"), 
            unrealized_pnl=Decimal("1000"),
            side=OrderSide.BUY,
            timestamp=datetime.now(),
        )
        
        market_data = MarketData(
            symbol="BTCUSDT",
            price=Decimal("51000"),
            volume=Decimal("1000"),
            timestamp=datetime.now(),
            bid=Decimal("50990"),
            ask=Decimal("51010"),
            open_price=Decimal("50000"),
            high_price=Decimal("52000"),
            low_price=Decimal("49000"),
        )
        
        positions = [position]
        market_data_list = [market_data]
        
        stats = await self.async_benchmark_function(
            risk_calculator.calculate_risk_metrics,
            positions, market_data_list,
            iterations=100
        )
        
        print(f"\nFull Risk Metrics Performance:")
        print(f"  Mean: {stats['mean_ms']:.3f}ms")
        print(f"  P95: {stats['p95_ms']:.3f}ms")
        print(f"  P99: {stats['p99_ms']:.3f}ms")
        print(f"  Ops/sec: {stats['ops_per_second']:.0f}")
        
        # Performance assertion
        assert stats['mean_ms'] < 10.0, f"Risk metrics too slow: {stats['mean_ms']:.3f}ms (target: < 10ms)"
        assert stats['p95_ms'] < 20.0, f"P95 too slow: {stats['p95_ms']:.3f}ms (target: < 20ms)"
        assert stats['ops_per_second'] > 50, f"Throughput too low: {stats['ops_per_second']:.0f} ops/sec"

    @pytest.mark.asyncio
    async def test_stop_loss_calculation_performance(self, adaptive_risk_manager):
        """Benchmark stop-loss calculation performance (Target: < 0.5ms per calculation)."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            strategy_name="test_strategy",
            metadata={},
        )
        entry_price = Decimal("50000")
        
        stats = await self.async_benchmark_function(
            adaptive_risk_manager.calculate_adaptive_stop_loss,
            signal, MarketRegime.MEDIUM_VOLATILITY, entry_price,
            iterations=2000
        )
        
        print(f"\nStop-Loss Calculation Performance:")
        print(f"  Mean: {stats['mean_ms']:.3f}ms")
        print(f"  P95: {stats['p95_ms']:.3f}ms")
        print(f"  P99: {stats['p99_ms']:.3f}ms")
        print(f"  Ops/sec: {stats['ops_per_second']:.0f}")
        
        # Performance assertion  
        assert stats['mean_ms'] < 0.5, f"Stop-loss calculation too slow: {stats['mean_ms']:.3f}ms (target: < 0.5ms)"
        assert stats['p95_ms'] < 1.0, f"P95 too slow: {stats['p95_ms']:.3f}ms (target: < 1ms)"
        assert stats['ops_per_second'] > 1000, f"Throughput too low: {stats['ops_per_second']:.0f} ops/sec"

    def test_concurrent_fee_calculation_performance(self, binance_order_manager):
        """Benchmark concurrent fee calculation performance."""
        order = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )
        fill_price = Decimal("50000")
        
        def calculate_fees():
            return binance_order_manager.calculate_fees(order, fill_price)
        
        # Benchmark concurrent execution
        num_threads = 10
        operations_per_thread = 1000
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for _ in range(num_threads):
                for _ in range(operations_per_thread):
                    futures.append(executor.submit(calculate_fees))
            
            # Wait for all to complete
            for future in as_completed(futures):
                result = future.result()
                assert isinstance(result, Decimal)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        total_operations = num_threads * operations_per_thread
        
        ops_per_second = total_operations / total_time
        avg_time_ms = (total_time / total_operations) * 1000
        
        print(f"\nConcurrent Fee Calculation Performance:")
        print(f"  Total operations: {total_operations}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Avg time: {avg_time_ms:.3f}ms")
        print(f"  Ops/sec: {ops_per_second:.0f}")
        
        # Performance assertion for concurrent execution
        assert ops_per_second > 5000, f"Concurrent throughput too low: {ops_per_second:.0f} ops/sec"
        assert avg_time_ms < 0.2, f"Concurrent average time too high: {avg_time_ms:.3f}ms"

    @pytest.mark.asyncio
    async def test_batch_risk_calculation_performance(self, risk_calculator):
        """Benchmark batch risk calculations for multiple positions."""
        # Create multiple positions
        positions = []
        market_data = []
        
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"] * 4  # 20 positions
        
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
        
        stats = await self.async_benchmark_function(
            risk_calculator.calculate_risk_metrics,
            positions, market_data,
            iterations=50
        )
        
        print(f"\nBatch Risk Calculation Performance (20 positions):")
        print(f"  Mean: {stats['mean_ms']:.3f}ms")
        print(f"  P95: {stats['p95_ms']:.3f}ms")
        print(f"  P99: {stats['p99_ms']:.3f}ms")
        print(f"  Portfolios/sec: {stats['ops_per_second']:.0f}")
        
        # Performance assertion for batch processing
        assert stats['mean_ms'] < 50.0, f"Batch risk calculation too slow: {stats['mean_ms']:.3f}ms (target: < 50ms)"
        assert stats['ops_per_second'] > 10, f"Batch throughput too low: {stats['ops_per_second']:.0f} portfolios/sec"

    @pytest.mark.asyncio
    async def test_position_sizing_method_comparison_performance(self, position_sizer):
        """Benchmark performance comparison across different position sizing methods."""
        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.8,
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            strategy_name="test_strategy",
        )
        portfolio_value = Decimal("10000")
        
        methods = [
            PositionSizeMethod.FIXED_PCT,
            PositionSizeMethod.KELLY_CRITERION,
            PositionSizeMethod.VOLATILITY_ADJUSTED,
            PositionSizeMethod.CONFIDENCE_WEIGHTED,
        ]
        
        method_stats = {}
        
        for method in methods:
            stats = await self.async_benchmark_function(
                position_sizer.calculate_position_size,
                signal, portfolio_value, method,
                iterations=1000
            )
            method_stats[method.value] = stats
            
            print(f"\n{method.value} Performance:")
            print(f"  Mean: {stats['mean_ms']:.3f}ms")
            print(f"  P95: {stats['p95_ms']:.3f}ms")
            print(f"  Ops/sec: {stats['ops_per_second']:.0f}")
        
        # Verify all methods meet performance requirements
        for method, stats in method_stats.items():
            if method == "kelly_criterion":
                assert stats['mean_ms'] < 2.0, f"{method} too slow: {stats['mean_ms']:.3f}ms"
            else:
                assert stats['mean_ms'] < 1.0, f"{method} too slow: {stats['mean_ms']:.3f}ms"

    def test_memory_usage_performance(self, risk_calculator):
        """Benchmark memory usage during intensive calculations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Get initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform intensive calculations
        num_iterations = 10000
        
        start_time = time.time()
        
        for i in range(num_iterations):
            # Add portfolio history
            risk_calculator._update_portfolio_history(Decimal(f"{50000 + i}"))
            
            # Add position returns
            risk_calculator.update_position_returns("BTCUSDT", 50000 + i)
            
            if i % 1000 == 0:
                # Periodic memory check
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                # Memory should not grow excessively
                assert memory_increase < 50, f"Excessive memory growth: {memory_increase:.2f}MB at iteration {i}"
        
        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        total_time = end_time - start_time
        memory_increase = final_memory - initial_memory
        ops_per_second = num_iterations / total_time
        
        print(f"\nMemory Usage Performance:")
        print(f"  Initial memory: {initial_memory:.2f}MB")
        print(f"  Final memory: {final_memory:.2f}MB") 
        print(f"  Memory increase: {memory_increase:.2f}MB")
        print(f"  Operations: {num_iterations}")
        print(f"  Time: {total_time:.2f}s")
        print(f"  Ops/sec: {ops_per_second:.0f}")
        
        # Performance assertions
        assert memory_increase < 20, f"Excessive memory usage: {memory_increase:.2f}MB"
        assert ops_per_second > 1000, f"Operation rate too low: {ops_per_second:.0f} ops/sec"

    def test_decimal_precision_performance_impact(self, binance_order_manager):
        """Benchmark performance impact of using Decimal vs float for precision."""
        
        # Test with Decimal (current implementation)
        order_decimal = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.12345678"),
        )
        fill_price_decimal = Decimal("50000.12345678")
        
        decimal_stats = self.benchmark_function(
            binance_order_manager.calculate_fees,
            order_decimal, fill_price_decimal,
            iterations=5000
        )
        
        print(f"\nDecimal Precision Performance Impact:")
        print(f"  Decimal Mean: {decimal_stats['mean_ms']:.3f}ms")
        print(f"  Decimal Ops/sec: {decimal_stats['ops_per_second']:.0f}")
        
        # Decimal precision should still meet performance requirements
        assert decimal_stats['mean_ms'] < 0.2, f"Decimal precision too slow: {decimal_stats['mean_ms']:.3f}ms"
        assert decimal_stats['ops_per_second'] > 2000, f"Decimal throughput too low: {decimal_stats['ops_per_second']:.0f}"

    @pytest.mark.asyncio
    async def test_stress_test_performance(self, risk_calculator):
        """Stress test performance under extreme load."""
        # Create stress test scenario
        num_concurrent_operations = 1000
        
        async def stress_operation():
            """Single stress test operation."""
            # Portfolio update
            await risk_calculator._update_portfolio_history(Decimal("50000"))
            
            # VaR calculation  
            var_result = await risk_calculator._calculate_var(1, Decimal("10000"))
            
            # Position return update
            await risk_calculator.update_position_returns("BTCUSDT", 50000)
            
            return var_result
        
        # Run stress test
        start_time = time.perf_counter()
        
        tasks = [stress_operation() for _ in range(num_concurrent_operations)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000  # Convert to ms
        
        ops_per_second = num_concurrent_operations / (total_time / 1000)
        avg_time_per_op = total_time / num_concurrent_operations
        
        print(f"\nStress Test Performance:")
        print(f"  Operations: {num_concurrent_operations}")
        print(f"  Total time: {total_time:.3f}ms")
        print(f"  Avg time/op: {avg_time_per_op:.3f}ms")
        print(f"  Ops/sec: {ops_per_second:.0f}")
        
        # Verify all operations completed successfully
        assert len(results) == num_concurrent_operations
        assert all(isinstance(result, Decimal) for result in results)
        
        # Performance assertion for stress test
        assert avg_time_per_op < 20.0, f"Stress test too slow: {avg_time_per_op:.3f}ms per operation"
        assert ops_per_second > 50, f"Stress test throughput too low: {ops_per_second:.0f} ops/sec"