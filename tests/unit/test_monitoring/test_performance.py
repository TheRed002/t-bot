"""
Unit tests for performance monitoring and optimization.

Tests the performance profiling infrastructure including:
- PerformanceProfiler functionality
- Latency statistics tracking
- System resource monitoring
- Trading-specific performance metrics
- Performance decorators
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timezone, timedelta

from src.monitoring.performance import (
    PerformanceProfiler,
    PerformanceMetrics,
    QueryMetrics,
    CacheMetrics,
    QueryOptimizer,
    CacheOptimizer,
    LatencyStats,
    ThroughputStats,
    SystemResourceStats,
    GCStats,
    get_performance_profiler,
    set_global_profiler,
    initialize_performance_monitoring,
    profile_async,
    profile_sync
)
from src.monitoring.metrics import MetricsCollector
from src.monitoring.alerting import AlertManager
from src.core.types import OrderType


class TestPerformanceMetrics:
    """Test performance metrics data structures."""
    
    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            latency_p50=1.5,
            latency_p95=2.0,
            latency_p99=3.0,
            throughput=1000.0,
            error_rate=0.01,
            cpu_usage=50.0,
            memory_usage=1024.0,
            timestamp=datetime.now(timezone.utc)
        )
        
        assert metrics.latency_p50 == 1.5
        assert metrics.latency_p95 == 2.0
        assert metrics.latency_p99 == 3.0
        assert metrics.throughput == 1000.0
        assert metrics.error_rate == 0.01
        assert metrics.cpu_usage == 50.0
        assert metrics.memory_usage == 1024.0
    
    def test_query_metrics_creation(self):
        """Test creating query metrics."""
        metrics = QueryMetrics(
            query_time_ms=500.0,
            rows_processed=100,
            cache_hits=80,
            cache_misses=20,
            connection_pool_usage=0.5,
            timestamp=datetime.now(timezone.utc)
        )
        
        assert metrics.query_time_ms == 500.0
        assert metrics.rows_processed == 100
        assert metrics.cache_hits == 80
        assert metrics.cache_misses == 20
        assert metrics.connection_pool_usage == 0.5
    
    def test_cache_metrics_creation(self):
        """Test creating cache metrics."""
        metrics = CacheMetrics(
            hit_rate=0.85,
            miss_rate=0.15,
            eviction_rate=0.05,
            memory_usage=1024.0,
            total_keys=100,
            timestamp=datetime.now(timezone.utc)
        )
        
        assert metrics.hit_rate == 0.85
        assert metrics.miss_rate == 0.15
        assert metrics.eviction_rate == 0.05
        assert metrics.memory_usage == 1024.0
        assert metrics.total_keys == 100


class TestLatencyStats:
    """Test LatencyStats functionality."""
    
    def test_latency_stats_from_values(self):
        """Test creating latency stats from values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        stats = LatencyStats.from_values(values)
        
        assert stats.count == 10
        assert stats.min_value == 1.0
        assert stats.max_value == 10.0
        assert stats.avg == 5.5
        assert stats.sum_value == 55.0
        assert stats.p50 > 0
        assert stats.p95 > stats.p50
        assert stats.p99 >= stats.p95
    
    def test_latency_stats_empty_values(self):
        """Test creating latency stats from empty values."""
        stats = LatencyStats.from_values([])
        
        assert stats.count == 0
        assert stats.p50 == 0.0
        assert stats.p95 == 0.0
        assert stats.p99 == 0.0
        assert stats.min_value == 0.0
        assert stats.max_value == 0.0
        assert stats.avg == 0.0


class TestPerformanceProfiler:
    """Test PerformanceProfiler functionality."""
    
    @pytest.fixture
    def profiler(self):
        """Create a test performance profiler."""
        metrics_collector = Mock(spec=MetricsCollector)
        alert_manager = Mock(spec=AlertManager)
        return PerformanceProfiler(
            metrics_collector=metrics_collector,
            alert_manager=alert_manager,
            max_samples=1000,
            collection_interval=1.0,
            anomaly_detection=False
        )
    
    def test_profiler_initialization(self, profiler):
        """Test profiler initialization."""
        assert profiler.max_samples == 1000
        assert profiler.collection_interval == 1.0
        assert profiler.anomaly_detection is False
        assert profiler._running is False
        assert profiler.metrics_collector is not None
        assert profiler.alert_manager is not None
    
    def test_profile_function_context_manager(self, profiler):
        """Test profile_function context manager."""
        with profiler.profile_function("test_function", "test_module", {"env": "test"}):
            time.sleep(0.01)  # Small operation
        
        # Check that histogram was observed
        profiler.metrics_collector.observe_histogram.assert_called()
        call_args = profiler.metrics_collector.observe_histogram.call_args[0]
        assert call_args[0] == "function_execution_latency_seconds"
        assert call_args[1] > 0  # Duration should be positive
    
    @pytest.mark.asyncio
    async def test_profile_async_function_context_manager(self, profiler):
        """Test profile_async_function context manager."""
        async with profiler.profile_async_function("test_async", "test_module", {"env": "test"}):
            await asyncio.sleep(0.01)  # Small operation
        
        # Check that histogram was observed
        profiler.metrics_collector.observe_histogram.assert_called()
        call_args = profiler.metrics_collector.observe_histogram.call_args[0]
        assert call_args[0] == "async_function_execution_latency_seconds"
        assert call_args[1] > 0  # Duration should be positive
    
    def test_record_order_execution(self, profiler):
        """Test recording order execution metrics."""
        profiler.record_order_execution(
            exchange="binance",
            order_type=OrderType.MARKET,
            symbol="BTCUSDT",
            latency_ms=50.0,
            fill_rate=0.95,
            slippage_bps=2.5
        )
        
        # Check metrics were recorded
        assert profiler.metrics_collector.observe_histogram.called
        assert profiler.metrics_collector.set_gauge.called
    
    def test_record_market_data_processing(self, profiler):
        """Test recording market data processing metrics."""
        profiler.record_market_data_processing(
            exchange="binance",
            data_type="ticker",
            processing_time_ms=5.0,
            message_count=100
        )
        
        # Check metrics were recorded
        assert profiler.metrics_collector.observe_histogram.called
        assert profiler.metrics_collector.set_gauge.called
    
    def test_record_websocket_latency(self, profiler):
        """Test recording WebSocket latency metrics."""
        profiler.record_websocket_latency(
            exchange="binance",
            message_type="order_update",
            latency_ms=10.0
        )
        
        # Check metrics were recorded
        assert profiler.metrics_collector.observe_histogram.called
        assert profiler.metrics_collector.set_gauge.called
    
    def test_record_database_query(self, profiler):
        """Test recording database query metrics."""
        profiler.record_database_query(
            database="trading_db",
            operation="select",
            table="orders",
            query_time_ms=25.0
        )
        
        # Check metrics were recorded
        assert profiler.metrics_collector.observe_histogram.called
    
    def test_record_strategy_performance(self, profiler):
        """Test recording strategy performance metrics."""
        profiler.record_strategy_performance(
            strategy="momentum",
            symbol="BTCUSDT",
            execution_time_ms=100.0,
            signal_accuracy=0.75,
            sharpe_ratio=1.8,
            timeframe="1h"
        )
        
        # Check metrics were recorded
        assert profiler.metrics_collector.observe_histogram.called
        assert profiler.metrics_collector.set_gauge.call_count >= 2
    
    def test_get_latency_stats(self, profiler):
        """Test getting latency statistics."""
        # Add some sample data
        metric_name = "test_metric"
        profiler._latency_data[metric_name].extend([1.0, 2.0, 3.0, 4.0, 5.0])
        
        stats = profiler.get_latency_stats(metric_name)
        assert stats is not None
        assert stats.count == 5
        assert stats.avg == 3.0
    
    def test_get_performance_summary(self, profiler):
        """Test getting performance summary."""
        summary = profiler.get_performance_summary()
        
        assert "timestamp" in summary
        assert "metrics_collected" in summary
        assert "system_resources" in summary
        assert "latency_stats" in summary
        assert "throughput_stats" in summary
        assert "gc_stats" in summary
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, profiler):
        """Test starting and stopping performance monitoring."""
        # Mock psutil to avoid system dependencies
        with patch('src.monitoring.performance.psutil') as mock_psutil:
            mock_psutil.cpu_percent.return_value = 50.0
            mock_psutil.virtual_memory.return_value = Mock(
                used=1000000, total=2000000, percent=50.0
            )
            mock_process = Mock()
            mock_process.num_threads.return_value = 10
            mock_psutil.Process.return_value = mock_process
            
            await profiler.start()
            assert profiler._running is True
            
            # Let it run briefly
            await asyncio.sleep(0.1)
            
            await profiler.stop()
            assert profiler._running is False


class TestQueryOptimizer:
    """Test QueryOptimizer functionality."""
    
    def test_query_optimizer_initialization(self):
        """Test query optimizer initialization."""
        optimizer = QueryOptimizer()
        assert optimizer.optimizations_applied == []
    
    def test_analyze_query(self):
        """Test analyzing query performance."""
        optimizer = QueryOptimizer()
        analysis = optimizer.analyze_query("SELECT * FROM orders WHERE status = 'open'")
        
        assert "complexity" in analysis
        assert "estimated_cost" in analysis
        assert "recommendations" in analysis
    
    def test_optimize_query(self):
        """Test optimizing a query."""
        optimizer = QueryOptimizer()
        original_query = "SELECT * FROM orders"
        optimized_query = optimizer.optimize_query(original_query)
        
        # Basic implementation just returns the same query
        assert optimized_query == original_query


class TestCacheOptimizer:
    """Test CacheOptimizer functionality."""
    
    def test_cache_optimizer_initialization(self):
        """Test cache optimizer initialization."""
        optimizer = CacheOptimizer()
        assert optimizer.cache_stats == {}
    
    def test_analyze_cache_performance(self):
        """Test analyzing cache performance."""
        optimizer = CacheOptimizer()
        analysis = optimizer.analyze_cache_performance()
        
        assert "hit_rate" in analysis
        assert "miss_rate" in analysis
        assert "recommendations" in analysis


class TestDecorators:
    """Test performance monitoring decorators."""
    
    @pytest.mark.asyncio
    async def test_profile_async_decorator(self):
        """Test async profiling decorator."""
        # Set up a mock profiler
        mock_profiler = Mock(spec=PerformanceProfiler)
        # Create a context manager mock that properly awaits
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=None)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        mock_profiler.profile_async_function = Mock(return_value=mock_context)
        set_global_profiler(mock_profiler)
        
        @profile_async(function_name="test_async_func", module_name="test_module")
        async def test_function():
            await asyncio.sleep(0.01)
            return "result"
        
        result = await test_function()
        assert result == "result"
        
        # Check that profile_async_function was called
        mock_profiler.profile_async_function.assert_called_once()
    
    def test_profile_sync_decorator(self):
        """Test sync profiling decorator."""
        # Set up a mock profiler
        mock_profiler = Mock(spec=PerformanceProfiler)
        # Create a context manager mock
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=None)
        mock_context.__exit__ = Mock(return_value=None)
        mock_profiler.profile_function = Mock(return_value=mock_context)
        set_global_profiler(mock_profiler)
        
        @profile_sync(function_name="test_sync_func", module_name="test_module")
        def test_function():
            time.sleep(0.01)
            return "result"
        
        result = test_function()
        assert result == "result"
        
        # Check that profile_function was called
        mock_profiler.profile_function.assert_called_once()


class TestGlobalFunctions:
    """Test global functions."""
    
    def test_get_set_global_profiler(self):
        """Test getting and setting global profiler."""
        profiler = Mock(spec=PerformanceProfiler)
        set_global_profiler(profiler)
        
        retrieved = get_performance_profiler()
        assert retrieved is profiler
    
    def test_initialize_performance_monitoring(self):
        """Test initializing performance monitoring."""
        metrics_collector = Mock(spec=MetricsCollector)
        alert_manager = Mock(spec=AlertManager)
        
        profiler = initialize_performance_monitoring(
            metrics_collector=metrics_collector,
            alert_manager=alert_manager,
            max_samples=5000
        )
        
        assert profiler is not None
        assert profiler.max_samples == 5000
        assert get_performance_profiler() is profiler


class TestPerformanceIntegration:
    """Integration tests for performance monitoring."""
    
    @pytest.mark.asyncio
    async def test_full_performance_workflow(self):
        """Test complete performance monitoring workflow."""
        # Create profiler with mock dependencies
        metrics_collector = Mock(spec=MetricsCollector)
        metrics_collector.observe_histogram = Mock()
        metrics_collector.set_gauge = Mock()
        metrics_collector.increment_counter = Mock()
        
        profiler = PerformanceProfiler(
            metrics_collector=metrics_collector,
            alert_manager=None,
            max_samples=100
        )
        
        # Record various metrics
        profiler.record_order_execution(
            exchange="binance",
            order_type=OrderType.LIMIT,
            symbol="BTCUSDT",
            latency_ms=45.0,
            fill_rate=0.98,
            slippage_bps=1.5
        )
        
        profiler.record_market_data_processing(
            exchange="binance",
            data_type="orderbook",
            processing_time_ms=2.5,
            message_count=50
        )
        
        profiler.record_database_query(
            database="trading_db",
            operation="insert",
            table="trades",
            query_time_ms=15.0
        )
        
        # Get performance summary
        summary = profiler.get_performance_summary()
        assert summary is not None
        assert "timestamp" in summary
    
    @pytest.mark.asyncio
    async def test_concurrent_performance_recording(self):
        """Test concurrent performance metric recording."""
        # Create a profiler with mocked dependencies
        metrics_collector = Mock(spec=MetricsCollector)
        alert_manager = Mock(spec=AlertManager)
        profiler = PerformanceProfiler(
            metrics_collector=metrics_collector,
            alert_manager=alert_manager
        )
        
        async def record_metrics():
            for i in range(10):
                profiler.record_order_execution(
                    exchange="test_exchange",
                    order_type=OrderType.MARKET,
                    symbol="BTCUSDT",
                    latency_ms=float(i),
                    fill_rate=0.95,
                    slippage_bps=1.0
                )
                await asyncio.sleep(0.001)
        
        # Run multiple concurrent tasks
        tasks = [record_metrics() for _ in range(5)]
        await asyncio.gather(*tasks)
        
        # Should have recorded metrics without errors
        stats = profiler.get_latency_stats("order_execution.test_exchange.market.BTCUSDT")
        assert stats is not None
        assert stats.count > 0