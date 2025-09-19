"""Helper functions for analytics tests."""

from decimal import Decimal
from datetime import datetime, timezone
from src.analytics.types import (
    PortfolioMetrics,
    PositionMetrics,
    RiskMetrics,
    OperationalMetrics,
    StrategyMetrics,
)
from src.core.types import PositionSide, PositionStatus


def create_test_portfolio_metrics(
    total_value: Decimal = Decimal('10000'),
    cash: Decimal = Decimal('5000'),
    invested_capital: Decimal = Decimal('5000'),
    unrealized_pnl: Decimal = Decimal('100'),
    realized_pnl: Decimal = Decimal('50'),
    **kwargs
) -> PortfolioMetrics:
    """Create test portfolio metrics with all required fields."""
    # Handle None values in calculation
    total_pnl = (unrealized_pnl or Decimal('0')) + (realized_pnl or Decimal('0'))

    defaults = {
        'timestamp': datetime.now(timezone.utc),
        'total_value': total_value,
        'cash': cash,
        'invested_capital': invested_capital,
        'unrealized_pnl': unrealized_pnl,
        'realized_pnl': realized_pnl,
        'total_pnl': total_pnl,
    }
    defaults.update(kwargs)
    return PortfolioMetrics(**defaults)


def create_test_position_metrics(
    symbol: str = 'BTC/USDT',
    current_value: Decimal = Decimal('5000'),
    **kwargs
) -> PositionMetrics:
    """Create test position metrics with all required fields."""
    defaults = {
        'timestamp': datetime.now(timezone.utc),
        'symbol': symbol,
        'exchange': 'binance',
        'side': 'long',
        'quantity': Decimal('0.1'),
        'entry_price': Decimal('49000'),
        'current_price': Decimal('50000'),
        'market_value': current_value,
        'unrealized_pnl': Decimal('100'),
        'unrealized_pnl_percent': Decimal('0.02'),
        'realized_pnl': Decimal('50'),
        'total_pnl': Decimal('150'),
        'weight': Decimal('0.1'),
        'duration_hours': Decimal('24'),
        'fees_paid': Decimal('5'),
        'strategy': 'momentum',
    }
    defaults.update(kwargs)
    return PositionMetrics(**defaults)


def create_test_risk_metrics(
    value_at_risk_95: Decimal = Decimal('500'),
    **kwargs
) -> RiskMetrics:
    """Create test risk metrics with all required fields."""
    defaults = {
        'timestamp': datetime.now(timezone.utc),
        'portfolio_var_95': value_at_risk_95,  # Use correct field name
        'max_drawdown': Decimal('200'),
        'sharpe_ratio': Decimal('1.5'),
        'volatility': Decimal('0.15'),
        'portfolio_var_99': Decimal('750'),
        'current_drawdown': Decimal('50'),
        'downside_deviation': Decimal('0.12'),
    }
    defaults.update(kwargs)
    return RiskMetrics(**defaults)


def create_test_operational_metrics(
    active_strategies: int = 5,
    **kwargs
) -> OperationalMetrics:
    """Create test operational metrics with all required fields."""
    # Handle legacy parameter names from tests
    if 'total_orders' in kwargs:
        kwargs['orders_placed_today'] = kwargs.pop('total_orders')
    if 'filled_orders' in kwargs:
        kwargs['orders_filled_today'] = kwargs.pop('filled_orders')
    if 'failed_orders' in kwargs:
        kwargs.pop('failed_orders')  # Calculated from placed - filled
    if 'uptime_percentage' in kwargs:
        kwargs['websocket_uptime_percent'] = kwargs.pop('uptime_percentage')

    defaults = {
        'timestamp': datetime.now(timezone.utc),
        'system_uptime': Decimal('24.5'),
        'strategies_active': active_strategies,
        'strategies_total': active_strategies + 2,
        'exchanges_connected': 3,
        'exchanges_total': 5,
        'orders_placed_today': 100,
        'orders_filled_today': 95,
        'order_fill_rate': Decimal('0.95'),
        'avg_order_execution_time': Decimal('50'),
        'avg_order_slippage': Decimal('5'),
        'api_call_success_rate': Decimal('0.98'),
        'websocket_uptime_percent': Decimal('99.5'),
        'data_latency_p50': Decimal('25'),
        'data_latency_p95': Decimal('75'),
        'error_rate': Decimal('0.02'),
        'critical_errors_today': 0,
        'memory_usage_percent': Decimal('65.5'),
        'cpu_usage_percent': Decimal('45.2'),
        'disk_usage_percent': Decimal('30.1'),
        'network_throughput_mbps': Decimal('100.5'),
        'database_connections_active': 10,
        'database_query_avg_time': Decimal('15'),
        'cache_hit_rate': Decimal('0.85'),
        'backup_status': 'successful',
        'last_backup_timestamp': datetime.now(timezone.utc),
        'compliance_checks_passed': 25,
        'compliance_checks_failed': 0,
        'risk_limit_breaches': 0,
        'circuit_breaker_triggers': 0,
        'performance_degradation_events': 1,
        'data_quality_issues': 0,
        'exchange_outages': 0,
        'recovery_time_minutes': Decimal('0.5'),
    }
    defaults.update(kwargs)
    return OperationalMetrics(**defaults)


def create_test_strategy_metrics(
    strategy_name: str = 'test_strategy',
    **kwargs
) -> StrategyMetrics:
    """Create test strategy metrics with all required fields."""
    defaults = {
        'timestamp': datetime.now(timezone.utc),
        'strategy_name': strategy_name,
        'total_pnl': Decimal('50'),
        'unrealized_pnl': Decimal('25'),
        'realized_pnl': Decimal('25'),
        'total_return': Decimal('0.05'),
        'daily_return': Decimal('0.002'),
        'volatility': Decimal('0.15'),
        'sharpe_ratio': Decimal('1.2'),
        'max_drawdown': Decimal('0.1'),
        'win_rate': Decimal('0.8'),
        'total_trades': 10,
        'winning_trades': 8,
        'losing_trades': 2,
        'capital_allocated': Decimal('10000'),
        'capital_utilized': Decimal('8000'),
        'utilization_rate': Decimal('0.8'),
        'active_positions': 3,
        'exposure': Decimal('7500'),
        'fees_paid': Decimal('10'),
    }
    defaults.update(kwargs)
    return StrategyMetrics(**defaults)