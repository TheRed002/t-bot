"""
Unit tests for performance monitoring and optimization.

Tests the performance profiling infrastructure including:
- PerformanceProfiler functionality
- Query optimization
- Cache performance analysis
- Memory monitoring
- Performance decorators
"""

import asyncio
import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from collections import deque

from src.monitoring.performance import (
    PerformanceProfiler,
    PerformanceMetrics,
    QueryMetrics,
    CacheMetrics,
    QueryOptimizer,
    CacheOptimizer,
    get_performance_profiler,
    set_global_profiler,
    profile_async,
    profile_sync
)
from src.core.exceptions import MonitoringError


class TestPerformanceMetrics:
    """Test performance metrics data structures."""
    
    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            function_name="test_function",
            module_name="test_module",
            execution_time=1.5,
            memory_usage=1024,
            cpu_time=0.8,
            call_count=1,
            timestamp=datetime.now(),
            thread_id=12345
        )
        
        assert metrics.function_name == "test_function"
        assert metrics.module_name == "test_module"
        assert metrics.execution_time == 1.5
        assert metrics.memory_usage == 1024
        assert metrics.cpu_time == 0.8
        assert metrics.call_count == 1
        assert metrics.thread_id == 12345
    
    def test_query_metrics_creation(self):
        """Test creating query metrics."""
        metrics = QueryMetrics(
            query="SELECT * FROM orders",
            execution_time=0.5,
            rows_affected=100,
            database="trading_db",
            timestamp=datetime.now(),
            slow_query=True
        )
        
        assert metrics.query == "SELECT * FROM orders"
        assert metrics.execution_time == 0.5
        assert metrics.rows_affected == 100
        assert metrics.database == "trading_db"
        assert metrics.slow_query is True
    
    def test_cache_metrics_creation(self):
        """Test creating cache metrics."""
        metrics = CacheMetrics(
            cache_name="order_cache",
            operation="get",
            hit=True,
            execution_time=0.001,
            key_size=64,
            value_size=512,
            timestamp=datetime.now()
        )
        
        assert metrics.cache_name == "order_cache"
        assert metrics.operation == "get"
        assert metrics.hit is True
        assert metrics.execution_time == 0.001
        assert metrics.key_size == 64
        assert metrics.value_size == 512


class TestPerformanceProfiler:
    """Test PerformanceProfiler functionality."""
    
    @pytest.fixture
    def profiler(self):
        """Create a test performance profiler."""
        return PerformanceProfiler(
            enable_memory_tracking=True,
            enable_cpu_profiling=True
        )
    
    def test_profiler_initialization(self, profiler):
        """Test profiler initialization."""
        assert profiler.enable_memory_tracking is True
        assert profiler.enable_cpu_profiling is True
        assert len(profiler._performance_data) == 0
        assert len(profiler._query_data) == 0
        assert len(profiler._cache_data) == 0
        assert profiler._monitoring_active is False
        assert profiler.slow_query_threshold == 1.0
    
    @patch('src.monitoring.performance.tracemalloc')
    def test_profile_function_context_manager(self, mock_tracemalloc, profiler):
        """Test profile_function context manager."""
        # Mock memory tracking
        mock_tracemalloc.get_traced_memory.side_effect = [
            (1000, 2000),  # Before
            (1500, 2000)   # After
        ]
        
        with profiler.profile_function("test_function", "test_module"):
            time.sleep(0.01)  # Small operation
        
        assert len(profiler._performance_data) == 1
        metrics = profiler._performance_data[0]
        assert metrics.function_name == "test_function"
        assert metrics.module_name == "test_module"
        assert metrics.execution_time > 0
        assert metrics.memory_usage >= 0
    
    def test_profile_function_without_memory_tracking(self):
        """Test profiling without memory tracking."""
        profiler = PerformanceProfiler(enable_memory_tracking=False)
        
        with profiler.profile_function("test_function"):
            pass
        
        assert len(profiler._performance_data) == 1
        metrics = profiler._performance_data[0]
        assert metrics.memory_usage == 0
    
    def test_record_query_performance(self, profiler):
        """Test recording query performance."""
        profiler.record_query_performance(
            query="SELECT * FROM orders WHERE status = 'open'",
            execution_time=0.5,
            rows_affected=50,
            database="trading_db",
            trace_id="trace123"
        )
        
        assert len(profiler._query_data) == 1
        query_metrics = profiler._query_data[0]
        assert query_metrics.query.startswith("SELECT * FROM orders")
        assert query_metrics.execution_time == 0.5
        assert query_metrics.rows_affected == 50
        assert query_metrics.database == "trading_db"
        assert query_metrics.trace_id == "trace123"
        assert query_metrics.slow_query is False
    
    def test_record_slow_query(self, profiler):
        """Test recording slow query performance."""
        profiler.record_query_performance(
            query="SELECT * FROM large_table",
            execution_time=2.5,  # Exceeds slow_query_threshold
            rows_affected=10000,
            database="trading_db"
        )
        
        query_metrics = profiler._query_data[0]
        assert query_metrics.slow_query is True
    
    def test_record_cache_performance(self, profiler):
        """Test recording cache performance."""
        profiler.record_cache_performance(
            cache_name="redis_cache",
            operation="get",
            hit=True,
            execution_time=0.002,
            key_size=32,
            value_size=256
        )
        
        assert len(profiler._cache_data) == 1
        cache_metrics = profiler._cache_data[0]
        assert cache_metrics.cache_name == "redis_cache"
        assert cache_metrics.operation == "get"
        assert cache_metrics.hit is True
        assert cache_metrics.execution_time == 0.002
    
    @patch('src.monitoring.performance.psutil.Process')
    def test_get_performance_summary(self, mock_process, profiler):
        """Test getting performance summary."""
        # Mock process info
        mock_proc = Mock()
        mock_proc.cpu_percent.return_value = 50.0
        mock_proc.memory_info.return_value = Mock(rss=100000, vms=200000)
        mock_proc.num_fds.return_value = 10
        mock_proc.num_threads.return_value = 5
        mock_process.return_value = mock_proc
        
        # Add some test data
        profiler._performance_data.append(PerformanceMetrics(
            function_name="test_func",
            module_name="test_module",
            execution_time=1.0,
            memory_usage=1024,
            cpu_time=0.5,
            call_count=1,
            timestamp=datetime.now(),
            thread_id=1
        ))
        
        profiler._query_data.append(QueryMetrics(
            query="SELECT * FROM test",
            execution_time=0.5,
            rows_affected=10,
            database="test_db",
            timestamp=datetime.now()
        ))
        
        profiler._cache_data.append(CacheMetrics(
            cache_name="test_cache",
            operation="get",
            hit=True,
            execution_time=0.01,
            key_size=32,
            value_size=128,
            timestamp=datetime.now()
        ))
        
        summary = profiler.get_performance_summary(timeframe_minutes=60)
        
        assert "function_performance" in summary
        assert "query_performance" in summary
        assert "cache_performance" in summary
        assert "system_resources" in summary
        
        # Check function performance
        func_perf = summary["function_performance"]
        assert "test_module.test_func" in func_perf
        
        # Check query performance
        query_perf = summary["query_performance"]
        assert query_perf["total_queries"] == 1
        assert query_perf["slow_queries"] == 0
        
        # Check cache performance
        cache_perf = summary["cache_performance"]
        assert cache_perf["total_operations"] == 1
        assert cache_perf["cache_hits"] == 1
        assert cache_perf["hit_rate_percentage"] == 100.0
    
    def test_get_slow_queries(self, profiler):
        """Test getting slow queries."""
        # Add slow and fast queries (slow is determined by execution time vs threshold)
        profiler.record_query_performance("SLOW QUERY", 2.0, 100, "db")  # 2.0s > 1.0s threshold
        profiler.record_query_performance("FAST QUERY", 0.1, 10, "db")   # 0.1s < 1.0s threshold
        profiler.record_query_performance("ANOTHER SLOW", 3.0, 1000, "db")  # 3.0s > 1.0s threshold
        
        slow_queries = profiler.get_slow_queries(limit=5)
        
        assert len(slow_queries) == 2
        # Should be sorted by execution time (descending)
        assert slow_queries[0].execution_time >= slow_queries[1].execution_time
    
    @patch('src.monitoring.performance.tracemalloc')
    @patch('src.monitoring.performance.psutil.Process')
    def test_get_memory_usage_report(self, mock_process, mock_tracemalloc, profiler):
        """Test getting memory usage report."""
        # Mock tracemalloc
        mock_tracemalloc.get_traced_memory.return_value = (5000000, 8000000)
        mock_snapshot = Mock()
        mock_snapshot.statistics.return_value = [
            Mock(
                traceback=Mock(format=lambda: ["test.py:10"]),
                size=1000000,
                count=100
            )
        ]
        mock_tracemalloc.take_snapshot.return_value = mock_snapshot
        
        # Mock process info
        mock_proc = Mock()
        mock_proc.memory_info.return_value = Mock(rss=50000000, vms=100000000)
        mock_process.return_value = mock_proc
        
        # Mock gc
        with patch('src.monitoring.performance.gc') as mock_gc:
            mock_gc.get_count.return_value = [100, 10, 1]
            mock_gc.get_objects.return_value = range(1000)
            
            report = profiler.get_memory_usage_report()
        
        assert "traced_memory" in report
        assert "process_memory" in report
        assert "top_allocations" in report
        assert "gc_stats" in report
        
        traced = report["traced_memory"]
        assert traced["current_bytes"] == 5000000
        assert traced["peak_bytes"] == 8000000
    
    @patch('src.monitoring.performance.tracemalloc')
    @patch('src.monitoring.performance.psutil.Process')
    @patch('src.monitoring.performance.gc')
    def test_optimize_memory(self, mock_gc, mock_process, mock_tracemalloc, profiler):
        """Test memory optimization."""
        # Mock memory tracking
        mock_tracemalloc.get_traced_memory.side_effect = [
            (10000000, 12000000),  # Before
            (8000000, 12000000)    # After
        ]
        
        # Mock process
        mock_proc = Mock()
        mock_proc.memory_info.side_effect = [
            Mock(rss=50000000),  # Before
            Mock(rss=45000000)   # After
        ]
        mock_process.return_value = mock_proc
        
        # Mock garbage collection
        mock_gc.collect.return_value = 50
        
        # Add lots of test data to trigger cleanup
        for i in range(6000):
            profiler._performance_data.append(PerformanceMetrics(
                function_name=f"func_{i}",
                module_name="test",
                execution_time=1.0,
                memory_usage=1024,
                cpu_time=0.5,
                call_count=1,
                timestamp=datetime.now(),
                thread_id=1
            ))
        
        result = profiler.optimize_memory()
        
        assert "gc_collected" in result
        assert "memory_freed_bytes" in result
        assert "rss_freed_bytes" in result
        assert result["gc_collected"] == 50
        assert result["memory_freed_bytes"] == 2000000
        assert result["rss_freed_bytes"] == 5000000
        
        # Check that data was trimmed
        assert len(profiler._performance_data) <= 3000
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, profiler):
        """Test starting and stopping monitoring."""
        await profiler.start_monitoring()
        assert profiler._monitoring_active is True
        assert profiler._monitoring_task is not None
        
        await profiler.stop_monitoring()
        assert profiler._monitoring_active is False
        assert profiler._monitoring_task is None


class TestDecorators:
    """Test performance profiling decorators."""
    
    @pytest.fixture
    def mock_profiler(self):
        """Create a mock profiler."""
        profiler = Mock(spec=PerformanceProfiler)
        profiler.profile_function.return_value.__enter__ = Mock()
        profiler.profile_function.return_value.__exit__ = Mock(return_value=False)
        
        with patch('src.monitoring.performance.get_performance_profiler', return_value=profiler):
            yield profiler
    
    @pytest.mark.asyncio
    async def test_profile_async_decorator(self, mock_profiler):
        """Test async function profiling decorator."""
        @profile_async("test_async_function")
        async def test_function():
            await asyncio.sleep(0.01)
            return "test_result"
        
        result = await test_function()
        
        assert result == "test_result"
        mock_profiler.profile_function.assert_called_once_with("test_async_function", "tests.unit.test_monitoring.test_performance")
    
    def test_profile_sync_decorator(self, mock_profiler):
        """Test sync function profiling decorator."""
        @profile_sync("test_sync_function")
        def test_function():
            time.sleep(0.01)
            return "test_result"
        
        result = test_function()
        
        assert result == "test_result"
        mock_profiler.profile_function.assert_called_once_with("test_sync_function", "tests.unit.test_monitoring.test_performance")
    
    def test_profile_decorator_without_profiler(self):
        """Test decorators when no profiler is available."""
        with patch('src.monitoring.performance.get_performance_profiler', return_value=None):
            @profile_sync("test_function")
            def test_function():
                return "test_result"
            
            result = test_function()
            assert result == "test_result"


class TestQueryOptimizer:
    """Test query optimization functionality."""
    
    @pytest.fixture
    def query_optimizer(self):
        """Create a query optimizer."""
        profiler = Mock(spec=PerformanceProfiler)
        return QueryOptimizer(profiler)
    
    def test_query_optimizer_initialization(self, query_optimizer):
        """Test query optimizer initialization."""
        assert query_optimizer.profiler is not None
        assert query_optimizer._query_cache == {}
        assert query_optimizer._prepared_statements == {}
    
    def test_analyze_slow_queries(self, query_optimizer):
        """Test analyzing slow queries."""
        # Mock slow queries
        slow_queries = [
            QueryMetrics(
                query="SELECT * FROM large_table WHERE condition",
                execution_time=2.5,
                rows_affected=10000,
                database="test_db",
                timestamp=datetime.now()
            ),
            QueryMetrics(
                query="SELECT id, name FROM table WHERE name LIKE '%%pattern%%'",
                execution_time=1.8,
                rows_affected=500,
                database="test_db",
                timestamp=datetime.now()
            )
        ]
        
        query_optimizer.profiler.get_slow_queries.return_value = slow_queries
        
        analysis = query_optimizer.analyze_slow_queries()
        
        assert len(analysis) == 2
        assert "optimization_suggestions" in analysis[0]
        assert "optimization_suggestions" in analysis[1]
        
        # Check for SELECT * suggestion
        suggestions = analysis[0]["optimization_suggestions"]
        assert any("SELECT *" in suggestion for suggestion in suggestions)
        
        # Check for LIKE %% suggestion
        suggestions = analysis[1]["optimization_suggestions"]
        assert any("leading wildcards" in suggestion for suggestion in suggestions)
    
    def test_analyze_query_patterns(self, query_optimizer):
        """Test query pattern analysis."""
        # Test various anti-patterns
        test_queries = [
            "SELECT * FROM orders",
            "UPDATE orders SET status = 'closed'",  # No WHERE clause
            "SELECT id FROM orders ORDER BY created_at",  # No LIMIT
            "SELECT o.* FROM orders o JOIN customers c ON o.customer_id = c.id JOIN products p ON o.product_id = p.id JOIN categories cat ON p.category_id = cat.id",  # Many JOINs
            "SELECT id FROM orders WHERE name LIKE '%%test%%'",  # Leading wildcard
            "SELECT id FROM orders WHERE status = 'open' OR status = 'pending'"  # OR condition
        ]
        
        for query in test_queries:
            suggestions = query_optimizer._analyze_query(query)
            assert isinstance(suggestions, list)
            assert len(suggestions) > 0


class TestCacheOptimizer:
    """Test cache optimization functionality."""
    
    @pytest.fixture
    def cache_optimizer(self):
        """Create a cache optimizer."""
        profiler = Mock(spec=PerformanceProfiler)
        return CacheOptimizer(profiler)
    
    def test_cache_optimizer_initialization(self, cache_optimizer):
        """Test cache optimizer initialization."""
        assert cache_optimizer.profiler is not None
    
    def test_analyze_cache_performance(self, cache_optimizer):
        """Test analyzing cache performance."""
        # Mock cache data
        cache_data = [
            CacheMetrics("redis_cache", "get", True, 0.001, 32, 128, datetime.now()),
            CacheMetrics("redis_cache", "get", False, 0.002, 32, 0, datetime.now()),
            CacheMetrics("redis_cache", "set", True, 0.003, 32, 256, datetime.now()),
            CacheMetrics("memory_cache", "get", True, 0.0005, 16, 64, datetime.now()),
            CacheMetrics("memory_cache", "get", True, 0.0008, 16, 64, datetime.now())
        ]
        
        cache_optimizer.profiler._cache_data = deque(cache_data)
        
        with patch('src.monitoring.performance.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now()
            
            analysis = cache_optimizer.analyze_cache_performance()
        
        assert "cache_statistics" in analysis
        assert "recommendations" in analysis
        assert "overall_metrics" in analysis
        
        cache_stats = analysis["cache_statistics"]
        assert "redis_cache" in cache_stats
        assert "memory_cache" in cache_stats
        
        redis_stats = cache_stats["redis_cache"]
        assert redis_stats["total_operations"] == 3
        assert redis_stats["hits"] == 2
        assert redis_stats["hit_rate"] == pytest.approx(66.67, rel=1e-2)
        
        overall = analysis["overall_metrics"]
        assert overall["total_operations"] == 5
        assert overall["overall_hit_rate"] == 80.0


class TestGlobalFunctions:
    """Test global profiler functions."""
    
    def test_global_profiler_functions(self):
        """Test global profiler get/set functions."""
        profiler = PerformanceProfiler()
        set_global_profiler(profiler)
        
        retrieved_profiler = get_performance_profiler()
        assert retrieved_profiler is profiler


class TestPerformanceIntegration:
    """Integration tests for performance monitoring."""
    
    @pytest.mark.asyncio
    async def test_full_performance_workflow(self):
        """Test complete performance monitoring workflow."""
        profiler = PerformanceProfiler(
            enable_memory_tracking=False,  # Disable for test stability
            enable_cpu_profiling=True
        )
        
        # Record various performance data
        with profiler.profile_function("test_function", "test_module"):
            time.sleep(0.01)
        
        profiler.record_query_performance(
            "SELECT * FROM test_table",
            0.5,
            100,
            "test_db"
        )
        
        profiler.record_cache_performance(
            "test_cache",
            "get",
            True,
            0.001
        )
        
        # Get performance summary
        summary = profiler.get_performance_summary(60)
        
        assert summary["function_performance"]["test_module.test_function"]["call_count"] == 1
        assert summary["query_performance"]["total_queries"] == 1
        assert summary["cache_performance"]["total_operations"] == 1
        assert summary["cache_performance"]["hit_rate_percentage"] == 100.0
    
    def test_concurrent_performance_recording(self):
        """Test concurrent performance data recording."""
        profiler = PerformanceProfiler(enable_memory_tracking=False)
        
        def record_data():
            for i in range(50):
                with profiler.profile_function(f"func_{i}", "test"):
                    pass
                
                profiler.record_query_performance(
                    f"SELECT {i}",
                    0.1,
                    1,
                    "db"
                )
                
                profiler.record_cache_performance(
                    "cache",
                    "get",
                    i % 2 == 0,  # 50% hit rate
                    0.001
                )
        
        # Run in multiple threads
        threads = [threading.Thread(target=record_data) for _ in range(3)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Check that all data was recorded
        assert len(profiler._performance_data) == 150  # 3 threads * 50 operations
        assert len(profiler._query_data) == 150
        assert len(profiler._cache_data) == 150
        
        # Get summary
        summary = profiler.get_performance_summary(60)
        assert summary["cache_performance"]["hit_rate_percentage"] == 50.0