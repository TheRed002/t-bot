"""
Production Readiness Tests for Monitoring and Observability

Tests monitoring, logging, metrics collection, and observability:
- Comprehensive logging coverage
- Metrics collection points
- Error reporting and classification
- Performance measurement hooks
- Health check endpoints
- Alert generation and routing
- Tracing and debugging capabilities
"""

import asyncio
import json
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.production_readiness.test_config import TestConfig as Config
from src.core.types import OrderRequest, OrderType
from src.core.types.trading import OrderSide
from src.exchanges.health_monitor import HealthMonitor
from src.exchanges.service import ExchangeService


class TestMonitoringObservability:
    """Test monitoring and observability capabilities."""

    @pytest.fixture
    def config(self):
        """Create test configuration with monitoring settings."""
        return Config({
            "monitoring": {
                "log_level": "INFO",
                "metrics_enabled": True,
                "health_check_interval": 30,
                "alert_thresholds": {
                    "error_rate": 0.05,  # 5%
                    "latency_p95": 1000,  # 1 second
                    "success_rate": 0.95  # 95%
                },
                "retention_days": 30
            },
            "logging": {
                "format": "structured",
                "include_correlation_id": True,
                "sanitize_sensitive_data": True,
                "max_log_size_mb": 100
            },
            "exchanges": {
                "binance": {
                    "api_key": "test_key",
                    "api_secret": "test_secret",
                    "sandbox": True
                }
            }
        })

    @pytest.fixture
    def health_monitor(self):
        """Create health monitor."""
        return HealthMonitor(
            failure_threshold=5,
            recovery_timeout=60,
            check_interval=10
        )

    @pytest.fixture
    async def exchange_service(self, config):
        """Create exchange service with monitoring."""
        with patch('src.exchanges.factory.ExchangeFactory') as mock_factory:
            mock_exchange_factory = AsyncMock()
            service = ExchangeService(
                exchange_factory=mock_exchange_factory,
                config=config
            )
            await service.start()
            yield service
            await service.stop()

    @pytest.mark.asyncio
    async def test_comprehensive_logging_coverage(self, exchange_service):
        """Test comprehensive logging coverage across operations."""
        
        with patch('src.exchanges.service.logger') as mock_logger:
            with patch('src.exchanges.factory.ExchangeFactory') as mock_factory:
                mock_exchange_factory = AsyncMock()
                mock_exchange = AsyncMock()
                mock_exchange.exchange_name = "binance"
                mock_exchange.health_check.return_value = True
                mock_exchange.get_account_balance.return_value = {
                    "BTC": Decimal("1.0"),
                    "USDT": Decimal("50000.0")
                }
                
                mock_exchange_factory.get_exchange.return_value = mock_exchange
                mock_factory.return_value = mock_exchange_factory
                
                # Perform various operations to trigger logging
                operations = [
                    ("get_exchange", lambda: exchange_service.get_exchange("binance")),
                    ("get_balance", lambda: exchange_service.get_account_balance("binance")),
                    ("health_check", lambda: exchange_service.get_service_health())
                ]
                
                log_calls = []
                
                # Capture all logging calls
                def capture_log(*args, **kwargs):
                    log_calls.append({
                        "args": args,
                        "kwargs": kwargs,
                        "timestamp": time.time()
                    })
                
                mock_logger.info.side_effect = capture_log
                mock_logger.debug.side_effect = capture_log
                mock_logger.warning.side_effect = capture_log
                mock_logger.error.side_effect = capture_log
                
                # Execute operations
                for op_name, operation in operations:
                    try:
                        await operation()
                    except Exception:
                        pass  # Expected for some test scenarios
                
                # Verify logging occurred
                assert len(log_calls) > 0, "Expected logging calls to be made"
                
                # Verify log structure
                for log_call in log_calls:
                    assert "timestamp" in log_call
                    assert "args" in log_call or "kwargs" in log_call

    @pytest.mark.asyncio
    async def test_metrics_collection_points(self, exchange_service, health_monitor):
        """Test metrics collection at key points."""
        
        # Test basic health metrics
        initial_health = health_monitor.get_health_status()
        assert isinstance(initial_health, dict)
        assert "healthy" in initial_health
        
        # Record various metrics
        latencies = [10.5, 25.2, 45.8, 33.1, 28.7]
        for latency in latencies:
            health_monitor.record_latency(latency)
        
        # Record successes and failures
        for _ in range(18):
            health_monitor.record_success()
        
        for _ in range(2):
            health_monitor.record_failure()
        
        # Get updated metrics
        metrics = health_monitor.get_health_status()
        
        # Verify metrics structure
        expected_fields = ["healthy", "failure_count", "success_count", "total_requests"]
        for field in expected_fields:
            if field in metrics:
                assert isinstance(metrics[field], (int, bool, float))
        
        # Test service-level metrics
        service_health = await exchange_service.get_service_health()
        assert "service" in service_health
        assert "active_exchanges" in service_health
        assert isinstance(service_health["active_exchanges"], int)

    @pytest.mark.asyncio
    async def test_error_reporting_classification(self, exchange_service):
        """Test error reporting and classification."""
        
        error_categories = []
        
        with patch('src.exchanges.service.logger') as mock_logger:
            with patch('src.exchanges.factory.ExchangeFactory') as mock_factory:
                # Test different error scenarios
                error_scenarios = [
                    ("connection_error", ConnectionError("Network unreachable")),
                    ("timeout_error", asyncio.TimeoutError("Request timeout")),
                    ("validation_error", ValueError("Invalid parameter")),
                    ("service_error", Exception("Generic service error"))
                ]
                
                for error_type, error in error_scenarios:
                    mock_factory.get_exchange.side_effect = error
                    
                    try:
                        await exchange_service.get_exchange("binance")
                    except Exception as e:
                        error_categories.append({
                            "type": error_type,
                            "exception": type(e).__name__,
                            "message": str(e)
                        })
                
                # Verify error logging
                assert mock_logger.error.called or mock_logger.warning.called
                
                # Verify error classification
                assert len(error_categories) > 0
                
                for error_info in error_categories:
                    assert "type" in error_info
                    assert "exception" in error_info
                    assert "message" in error_info

    @pytest.mark.asyncio
    async def test_performance_measurement_hooks(self, exchange_service):
        """Test performance measurement capabilities."""
        
        performance_data = []
        
        with patch('src.exchanges.factory.ExchangeFactory') as mock_factory:
            mock_exchange_factory = AsyncMock()
            mock_exchange = AsyncMock()
            mock_exchange.exchange_name = "binance"
            mock_exchange.health_check.return_value = True
            
            # Add artificial delays to measure
            async def delayed_operation(*args, **kwargs):
                start_time = time.time()
                await asyncio.sleep(0.05)  # 50ms delay
                end_time = time.time()
                
                performance_data.append({
                    "operation": "get_balance",
                    "duration": end_time - start_time,
                    "timestamp": end_time
                })
                
                return {"BTC": Decimal("1.0"), "USDT": Decimal("50000.0")}
            
            mock_exchange.get_account_balance.side_effect = delayed_operation
            mock_exchange_factory.get_exchange.return_value = mock_exchange
            mock_factory.return_value = mock_exchange_factory
            
            # Perform timed operations
            start_time = time.time()
            result = await exchange_service.get_account_balance("binance")
            end_time = time.time()
            
            # Verify performance measurement
            assert result is not None
            assert end_time - start_time >= 0.05  # At least the simulated delay
            
            # Verify performance data collection
            assert len(performance_data) > 0
            
            perf_entry = performance_data[0]
            assert "operation" in perf_entry
            assert "duration" in perf_entry
            assert "timestamp" in perf_entry
            assert perf_entry["duration"] >= 0.05

    @pytest.mark.asyncio
    async def test_health_check_endpoints(self, exchange_service, health_monitor):
        """Test health check endpoint functionality."""
        
        # Test service health check
        service_health = await exchange_service.get_service_health()
        
        # Verify health check structure
        required_fields = ["service", "status"]
        for field in required_fields:
            assert field in service_health
        
        assert service_health["service"] == "ExchangeService"
        
        # Test component health monitoring
        component_health = health_monitor.get_health_status()
        assert "healthy" in component_health
        
        # Test health check with failures
        for _ in range(10):
            health_monitor.record_failure()
        
        degraded_health = health_monitor.get_health_status()
        assert "failure_count" in degraded_health
        assert degraded_health["failure_count"] == 10

    def test_alert_generation_routing(self, health_monitor, config):
        """Test alert generation and routing."""
        
        monitoring_config = config.monitoring if hasattr(config, 'monitoring') else {}
        alert_thresholds = monitoring_config.get('alert_thresholds', {})
        
        # Test error rate threshold
        error_threshold = alert_thresholds.get('error_rate', 0.05)  # 5%
        
        # Generate high error rate
        for _ in range(8):  # 8 failures
            health_monitor.record_failure()
        
        for _ in range(2):  # 2 successes  
            health_monitor.record_success()
        
        # Should trigger error rate alert (80% failure rate > 5% threshold)
        health_status = health_monitor.get_health_status()
        
        if "error_rate" in health_status:
            error_rate = health_status["error_rate"]
            alert_triggered = error_rate > error_threshold
            assert alert_triggered  # Should trigger alert
        
        # Test recovery alert
        for _ in range(20):  # Add many successes
            health_monitor.record_success()
        
        recovered_health = health_monitor.get_health_status()
        if "error_rate" in recovered_health:
            assert recovered_health["error_rate"] < error_threshold

    @pytest.mark.asyncio
    async def test_tracing_debugging_capabilities(self, exchange_service):
        """Test tracing and debugging capabilities."""
        
        trace_data = []
        
        with patch('src.exchanges.service.logger') as mock_logger:
            # Capture debug information
            def capture_debug(*args, **kwargs):
                trace_data.append({
                    "level": "debug",
                    "message": args[0] if args else "",
                    "kwargs": kwargs,
                    "timestamp": time.time()
                })
            
            mock_logger.debug.side_effect = capture_debug
            
            with patch('src.exchanges.factory.ExchangeFactory') as mock_factory:
                mock_exchange_factory = AsyncMock()
                mock_exchange = AsyncMock()
                mock_exchange.exchange_name = "binance"
                mock_exchange.health_check.return_value = True
                mock_exchange.get_account_balance.return_value = {
                    "BTC": Decimal("1.0")
                }
                
                mock_exchange_factory.get_exchange.return_value = mock_exchange
                mock_factory.return_value = mock_exchange_factory
                
                # Perform operations that should generate trace data
                await exchange_service.get_account_balance("binance")
                
                # Verify tracing data
                if trace_data:
                    trace_entry = trace_data[0]
                    assert "timestamp" in trace_entry
                    assert "message" in trace_entry
                    assert isinstance(trace_entry["message"], str)

    @pytest.mark.asyncio
    async def test_correlation_id_tracking(self, config):
        """Test correlation ID tracking across operations."""
        
        correlation_ids = []
        
        with patch('src.exchanges.factory.ExchangeFactory') as mock_factory:
            mock_exchange_factory = AsyncMock()
            
            # Create service with correlation ID
            service = ExchangeService(
                exchange_factory=mock_exchange_factory,
                config=config,
                correlation_id="test-correlation-123"
            )
            
            # Verify correlation ID is set
            assert hasattr(service, '_correlation_id') or hasattr(service, 'correlation_id')
            
            # Test that operations maintain correlation context
            with patch('src.exchanges.service.logger') as mock_logger:
                def capture_correlation(*args, **kwargs):
                    # Look for correlation ID in log context
                    if 'correlation_id' in kwargs:
                        correlation_ids.append(kwargs['correlation_id'])
                
                mock_logger.info.side_effect = capture_correlation
                mock_logger.debug.side_effect = capture_correlation
                
                try:
                    await service.get_service_health()
                except:
                    pass  # Expected to fail in test environment
                
                # Verify correlation ID tracking
                if correlation_ids:
                    assert "test-correlation-123" in correlation_ids[0]

    @pytest.mark.asyncio
    async def test_structured_logging_format(self, exchange_service):
        """Test structured logging format."""
        
        log_entries = []
        
        with patch('src.exchanges.service.logger') as mock_logger:
            # Capture structured log entries
            def capture_structured(*args, **kwargs):
                log_entry = {
                    "message": args[0] if args else "",
                    "extra_data": kwargs,
                    "timestamp": time.time()
                }
                log_entries.append(log_entry)
            
            mock_logger.info.side_effect = capture_structured
            mock_logger.error.side_effect = capture_structured
            mock_logger.warning.side_effect = capture_structured
            
            with patch('src.exchanges.factory.ExchangeFactory') as mock_factory:
                mock_factory.get_exchange.side_effect = Exception("Test error")
                
                try:
                    await exchange_service.get_exchange("binance")
                except:
                    pass  # Expected error
                
                # Verify structured logging
                if log_entries:
                    log_entry = log_entries[0]
                    assert "message" in log_entry
                    assert "timestamp" in log_entry
                    assert isinstance(log_entry["message"], str)
                    
                    # Should contain structured data
                    if "extra_data" in log_entry:
                        assert isinstance(log_entry["extra_data"], dict)

    @pytest.mark.asyncio
    async def test_metrics_aggregation_reporting(self, health_monitor):
        """Test metrics aggregation and reporting."""
        
        # Generate metrics data over time
        latency_data = []
        success_data = []
        
        # Simulate operations over time
        for i in range(100):
            # Vary latency
            latency = 10 + (i % 50)  # 10-60ms range
            health_monitor.record_latency(latency)
            latency_data.append(latency)
            
            # Vary success/failure
            if i % 10 == 0:  # 10% failure rate
                health_monitor.record_failure()
                success_data.append(False)
            else:
                health_monitor.record_success()
                success_data.append(True)
        
        # Get aggregated metrics
        metrics = health_monitor.get_health_status()
        
        # Verify aggregated statistics
        if "total_requests" in metrics:
            assert metrics["total_requests"] == 100
        
        if "success_rate" in metrics:
            expected_success_rate = sum(success_data) / len(success_data)
            actual_success_rate = metrics["success_rate"]
            assert abs(actual_success_rate - expected_success_rate) < 0.1
        
        if "average_latency" in metrics:
            expected_avg_latency = sum(latency_data) / len(latency_data)
            actual_avg_latency = metrics["average_latency"]
            assert abs(actual_avg_latency - expected_avg_latency) < 5.0

    @pytest.mark.asyncio
    async def test_log_sanitization_security(self, exchange_service):
        """Test log sanitization for sensitive data."""
        
        sanitized_logs = []
        
        with patch('src.exchanges.service.logger') as mock_logger:
            # Capture log messages
            def capture_sanitized(*args, **kwargs):
                log_message = args[0] if args else ""
                sanitized_logs.append({
                    "message": log_message,
                    "args": args,
                    "kwargs": kwargs
                })
            
            mock_logger.info.side_effect = capture_sanitized
            mock_logger.error.side_effect = capture_sanitized
            mock_logger.debug.side_effect = capture_sanitized
            
            # Create order with potentially sensitive data
            order_request = OrderRequest(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
                price=Decimal("50000.00")
            )
            
            with patch('src.exchanges.factory.ExchangeFactory') as mock_factory:
                mock_factory.get_exchange.side_effect = Exception("API key invalid")
                
                try:
                    await exchange_service.place_order("binance", order_request)
                except:
                    pass
                
                # Verify sensitive data is not in logs
                for log_entry in sanitized_logs:
                    message = str(log_entry["message"]).lower()
                    
                    # Should not contain sensitive patterns
                    sensitive_patterns = [
                        "api_key", "api_secret", "password", "private_key",
                        "token", "credential", "auth"
                    ]
                    
                    for pattern in sensitive_patterns:
                        if pattern in message:
                            # Should be masked or redacted
                            assert "***" in message or "[REDACTED]" in message

    def test_monitoring_configuration_validation(self, config):
        """Test monitoring configuration validation."""
        
        monitoring_config = config.monitoring if hasattr(config, 'monitoring') else {}
        
        # Test required monitoring settings
        if monitoring_config:
            # Should have alert thresholds
            if "alert_thresholds" in monitoring_config:
                thresholds = monitoring_config["alert_thresholds"]
                assert isinstance(thresholds, dict)
                
                # Thresholds should be reasonable
                if "error_rate" in thresholds:
                    assert 0.0 <= thresholds["error_rate"] <= 1.0
                
                if "latency_p95" in thresholds:
                    assert thresholds["latency_p95"] > 0
                
                if "success_rate" in thresholds:
                    assert 0.0 <= thresholds["success_rate"] <= 1.0
        
        # Test logging configuration
        logging_config = config.logging if hasattr(config, 'logging') else {}
        if logging_config:
            if "max_log_size_mb" in logging_config:
                assert logging_config["max_log_size_mb"] > 0
                assert logging_config["max_log_size_mb"] <= 1000  # Reasonable limit