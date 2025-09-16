"""
Unit tests for Analytics API endpoints.

Tests all analytics endpoints including portfolio metrics, risk analysis,
alerts, and reporting functionality.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from src.core.exceptions import ServiceError, ValidationError
from src.web_interface.api.analytics import (
    PortfolioMetricsResponse,
    RiskMetricsResponse,
    StrategyMetricsResponse,
    VaRRequest,
    StressTestRequest,
    GenerateReportRequest,
    AlertAcknowledgeRequest,
    router,
)


@pytest.fixture
def mock_analytics_service():
    """Create a mock analytics service."""
    service = MagicMock()
    service.get_portfolio_metrics = AsyncMock()
    service.get_risk_metrics = AsyncMock()
    service.get_strategy_metrics = AsyncMock()
    service.get_operational_metrics = AsyncMock()
    service.get_alerts = AsyncMock()
    service.generate_report = AsyncMock()
    service.export_data = AsyncMock()
    service.calculate_var = AsyncMock()
    service.run_stress_test = AsyncMock()
    service.get_portfolio_composition = AsyncMock()
    service.get_correlation_matrix = AsyncMock()
    service.acknowledge_alert = AsyncMock()
    return service


@pytest.fixture
def mock_current_user():
    """Create a mock authenticated user."""
    user = MagicMock()
    user.username = "test_user"
    user.user_id = "user_123"
    user.roles = ["user", "trading"]
    return user


@pytest.fixture
def mock_admin_user():
    """Create a mock admin user."""
    user = MagicMock()
    user.username = "admin_user"
    user.user_id = "admin_123"
    user.roles = ["admin"]
    return user


class TestPortfolioMetricsEndpoints:
    """Test portfolio metrics endpoints."""

    @pytest.mark.asyncio
    async def test_get_portfolio_metrics_success(self, mock_analytics_service, mock_current_user):
        """Test successful portfolio metrics retrieval."""
        # Setup mock response
        mock_metrics = {
            "total_value": "10000.00",
            "total_pnl": "123.45", 
            "total_pnl_percentage": "1.25",
            "win_rate": "0.60",
            "sharpe_ratio": "1.85",
            "max_drawdown": "0.08",
            "positions_count": 5,
            "active_strategies": 2,
            "timestamp": datetime.now(timezone.utc)
        }
        mock_analytics_service.get_portfolio_metrics = AsyncMock(return_value=mock_metrics)

        # Import the endpoint function directly
        from src.web_interface.api.analytics import get_portfolio_metrics
        
        # Call endpoint
        result = await get_portfolio_metrics(
            current_user=mock_current_user,
            web_analytics_service=mock_analytics_service
        )

        # Assertions
        assert result is not None
        assert result.total_value == "10000.00"
        assert result.total_pnl == "123.45"
        assert result.sharpe_ratio == "1.85"
        mock_analytics_service.get_portfolio_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_portfolio_metrics_service_error(self, mock_analytics_service, mock_current_user):
        """Test portfolio metrics retrieval with service error."""
        # Setup mock to raise error
        mock_analytics_service.get_portfolio_metrics.side_effect = ServiceError("Database connection failed")

        from src.web_interface.api.analytics import get_portfolio_metrics

        # Call endpoint and expect exception
        with pytest.raises(HTTPException) as exc_info:
            await get_portfolio_metrics(
                current_user=mock_current_user,
                web_analytics_service=mock_analytics_service
            )

        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to retrieve portfolio metrics" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_get_portfolio_metrics_validation_error(self, mock_analytics_service, mock_current_user):
        """Test portfolio metrics with validation error."""
        from src.web_interface.api.analytics import get_portfolio_metrics

        # Setup mock to raise validation error
        mock_analytics_service.get_portfolio_metrics.side_effect = ValidationError("Invalid portfolio data")

        with pytest.raises(HTTPException) as exc_info:
            await get_portfolio_metrics(
                current_user=mock_current_user,
                web_analytics_service=mock_analytics_service
            )

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST


class TestRiskMetricsEndpoints:
    """Test risk metrics endpoints."""

    @pytest.mark.asyncio
    async def test_get_risk_metrics_success(self, mock_analytics_service, mock_current_user):
        """Test successful risk metrics retrieval."""
        # Setup mock response
        mock_metrics = {
            "portfolio_var": {"1d": "500.00", "5d": "1200.00"},
            "portfolio_volatility": "0.15",
            "portfolio_beta": "1.15",
            "correlation_risk": "0.65",
            "concentration_risk": "0.45",
            "leverage_ratio": "1.2",
            "margin_usage": "0.25",
            "timestamp": datetime.now(timezone.utc)
        }
        mock_analytics_service.get_risk_metrics = AsyncMock(return_value=mock_metrics)

        from src.web_interface.api.analytics import get_risk_metrics

        # Call endpoint
        result = await get_risk_metrics(
            current_user=mock_current_user,
            web_analytics_service=mock_analytics_service
        )

        # Assertions
        assert result is not None
        assert result.portfolio_var["1d"] == "500.00"
        assert result.portfolio_beta == "1.15"
        assert result.correlation_risk == "0.65"
        mock_analytics_service.get_risk_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_calculate_var_success(self, mock_analytics_service, mock_current_user):
        """Test VaR calculation endpoint."""
        # Setup mock response
        mock_var_result = {
            "var_95": Decimal("450.00"),
            "var_99": Decimal("680.00"),
            "cvar_95": Decimal("550.00"),
            "cvar_99": Decimal("780.00"),
            "confidence_level": Decimal("0.95"),
            "time_horizon": 1,
            "method": "historical",
            "calculated_at": datetime.now(timezone.utc)
        }
        mock_analytics_service.calculate_var = AsyncMock(return_value=mock_var_result)

        from src.web_interface.api.analytics import calculate_var

        # Call endpoint
        result = await calculate_var(
            request=VaRRequest(confidence_level=0.95, time_horizon=1, method="historical"),
            current_user=mock_current_user,
            web_analytics_service=mock_analytics_service
        )

        # Assertions
        assert result["var_results"]["var_95"] == Decimal("450.00")
        assert result["parameters"]["method"] == "historical"
        mock_analytics_service.calculate_var.assert_called_once_with(
            confidence_level=Decimal("0.95"),
            time_horizon=1,
            method="historical"
        )

    @pytest.mark.asyncio
    async def test_run_stress_test_success(self, mock_analytics_service, mock_admin_user):
        """Test stress testing endpoint (admin only)."""
        # Setup mock admin user with proper role
        mock_admin_user.get = lambda x: ["admin"] if x == "role" else mock_admin_user.get(x)
        
        # Setup mock response
        mock_stress_result = {
            "scenario": "market_crash",
            "parameters": {
                "market_drop": Decimal("-0.20"),
                "volatility_spike": Decimal("2.5"),
                "correlation_increase": Decimal("0.30")
            },
            "impact": {
                "portfolio_loss": Decimal("-2500.00"),
                "max_drawdown": Decimal("0.25"),
                "var_increase": Decimal("1.8")
            },
            "recommendations": [
                "Reduce position sizes",
                "Increase hedging",
                "Diversify holdings"
            ],
            "test_date": datetime.now(timezone.utc)
        }
        mock_analytics_service.run_stress_test = AsyncMock(return_value=mock_stress_result)

        from src.web_interface.api.analytics import run_stress_test

        # Call endpoint
        result = await run_stress_test(
            request=StressTestRequest(
                scenario_name="market_crash",
                scenario_params={
                    "market_drop": -0.20,
                    "volatility_spike": 2.5
                }
            ),
            current_user=mock_admin_user,
            web_analytics_service=mock_analytics_service
        )

        # Assertions
        assert result["scenario"] == "market_crash"
        assert result["results"]["impact"]["portfolio_loss"] == Decimal("-2500.00")
        assert len(result["results"]["recommendations"]) == 3


class TestAlertEndpoints:
    """Test alert management endpoints."""

    @pytest.mark.asyncio
    async def test_get_alerts_success(self, mock_analytics_service, mock_current_user):
        """Test successful alerts retrieval."""
        # Setup mock response
        mock_alerts = [
            {
                "alert_id": "alert_001",
                "severity": "high",
                "type": "risk_limit",
                "title": "VaR Limit Exceeded",
                "message": "Portfolio VaR exceeds 95% confidence limit",
                "metric_name": "portfolio_var_1d",
                "metric_value": Decimal("1100.00"),
                "threshold": Decimal("1000.00"),
                "created_at": datetime.now(timezone.utc),
                "acknowledged": False
            },
            {
                "alert_id": "alert_002",
                "severity": "medium",
                "type": "performance",
                "title": "Drawdown Warning",
                "message": "Current drawdown approaching limit",
                "metric_name": "current_drawdown",
                "metric_value": Decimal("0.08"),
                "threshold": Decimal("0.10"),
                "created_at": datetime.now(timezone.utc) - timedelta(hours=1),
                "acknowledged": True,
                "acknowledged_by": "user_456",
                "acknowledged_at": datetime.now(timezone.utc) - timedelta(minutes=30)
            }
        ]
        mock_analytics_service.get_active_alerts = AsyncMock(return_value={"alerts": mock_alerts, "count": len(mock_alerts)})

        # Call endpoint (using get_active_alerts which exists)
        from src.web_interface.api.analytics import get_active_alerts
        
        result = await get_active_alerts(
            current_user=mock_current_user,
            web_analytics_service=mock_analytics_service
        )

        # Assertions
        assert len(result["alerts"]) == 2
        assert result["alerts"][0]["alert_id"] == "alert_001"
        assert result["alerts"][0]["severity"] == "high"
        mock_analytics_service.get_active_alerts.assert_called_once()

    @pytest.mark.asyncio
    async def test_acknowledge_alert_success(self, mock_analytics_service, mock_current_user):
        """Test alert acknowledgment."""
        # Setup mock response
        mock_analytics_service.acknowledge_alert = AsyncMock(return_value=True)

        from src.web_interface.api.analytics import acknowledge_alert

        # Call endpoint
        result = await acknowledge_alert(
            alert_id="alert_001",
            request=AlertAcknowledgeRequest(acknowledged_by="test_user"),
            current_user=mock_current_user,
            web_analytics_service=mock_analytics_service
        )

        # Assertions
        assert result["status"] == "acknowledged"
        assert result["alert_id"] == "alert_001"
        mock_analytics_service.acknowledge_alert.assert_called_once_with(
            alert_id="alert_001",
            acknowledged_by="test_user",
            notes=None
        )


class TestReportingEndpoints:
    """Test reporting endpoints."""

    @pytest.mark.asyncio
    async def test_generate_report_success(self, mock_analytics_service, mock_current_user):
        """Test report generation."""
        # Setup mock response
        mock_report = {
            "report_id": "report_20240101_120000",
            "report_type": "daily",
            "period_start": datetime.now(timezone.utc) - timedelta(days=1),
            "period_end": datetime.now(timezone.utc),
            "sections": {
                "summary": {
                    "total_trades": 25,
                    "winning_trades": 15,
                    "total_pnl": Decimal("456.78"),
                    "roi": Decimal("4.57")
                },
                "portfolio_metrics": {
                    "end_value": Decimal("10456.78"),
                    "max_value": Decimal("10600.00"),
                    "min_value": Decimal("10100.00")
                },
                "risk_metrics": {
                    "max_var": Decimal("520.00"),
                    "average_var": Decimal("480.00"),
                    "risk_events": 2
                }
            },
            "generated_at": datetime.now(timezone.utc)
        }
        mock_analytics_service.generate_report = AsyncMock(return_value=mock_report)

        from src.web_interface.api.analytics import generate_report

        # Call endpoint function directly with service dependency
        result = await generate_report(
            report_type="daily",
            start_date=datetime.now(timezone.utc) - timedelta(days=1),
            end_date=datetime.now(timezone.utc),
            current_user=mock_current_user,
            web_analytics_service=mock_analytics_service
        )

        # Assertions
        assert result["report_type"] == "daily"
        assert "sections" in result
        assert result["sections"]["summary"]["total_pnl"] == Decimal("456.78")


class TestExportEndpoints:
    """Test data export endpoints."""

    @pytest.mark.asyncio
    async def test_export_portfolio_data_json(self, mock_analytics_service, mock_current_user):
        """Test portfolio data export in JSON format."""
        # Setup mock response
        mock_export = {
            "format": "json",
            "data": {
                "portfolio": {
                    "total_value": "10000.00",
                    "positions": [
                        {"symbol": "BTC/USDT", "quantity": "0.1", "value": "4500.00"},
                        {"symbol": "ETH/USDT", "quantity": "2.0", "value": "5000.00"}
                    ]
                },
                "metadata": {
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "user": "test_user"
                }
            }
        }
        mock_analytics_service.export_data = AsyncMock(return_value=mock_export)

        from src.web_interface.api.analytics import export_portfolio_data

        # Call endpoint function directly with service dependency
        result = await export_portfolio_data(
            format="json",
            include_metadata=True,
            current_user=mock_current_user,
            web_analytics_service=mock_analytics_service
        )

        # Assertions
        assert result["format"] == "json"
        assert "data" in result
        assert "metadata" in result["data"]
        mock_analytics_service.export_data.assert_called_once_with(
            data_type="portfolio",
            format="json",
            include_metadata=True
        )

    @pytest.mark.asyncio
    async def test_export_portfolio_data_csv(self, mock_analytics_service, mock_current_user):
        """Test portfolio data export in CSV format."""
        # Setup mock response for CSV
        mock_export = {
            "format": "csv",
            "filename": "portfolio_20240101_120000.csv",
            "size_bytes": 2048,
            "download_url": "/downloads/portfolio_20240101_120000.csv"
        }
        mock_analytics_service.export_data = AsyncMock(return_value=mock_export)

        from src.web_interface.api.analytics import export_portfolio_data

        # Call endpoint function directly with service dependency
        result = await export_portfolio_data(
            format="csv",
            include_metadata=False,
            current_user=mock_current_user,
            web_analytics_service=mock_analytics_service
        )

        # Assertions
        assert result["format"] == "csv"
        assert "filename" in result
        assert "download_url" in result


class TestStrategyMetricsEndpoints:
    """Test strategy metrics endpoints."""

    @pytest.mark.asyncio
    async def test_get_strategy_metrics_success(self, mock_analytics_service, mock_current_user):
        """Test strategy metrics retrieval."""
        # Setup mock response
        mock_metrics = [
            {
                "strategy_name": "momentum_v1",
                "total_trades": 50,
                "winning_trades": 30,
                "win_rate": "0.60",
                "total_pnl": "1234.56",
                "average_pnl": "24.69",
                "sharpe_ratio": "1.92",
                "max_drawdown": "0.06",
                "active": True,
                "last_trade": datetime.now(timezone.utc) - timedelta(hours=2)
            },
            {
                "strategy_name": "mean_reversion_v2",
                "total_trades": 35,
                "winning_trades": 22,
                "win_rate": "0.63",
                "total_pnl": "890.12",
                "average_pnl": "25.43",
                "sharpe_ratio": "1.78",
                "max_drawdown": "0.05",
                "active": True,
                "last_trade": datetime.now(timezone.utc) - timedelta(hours=1)
            }
        ]
        mock_analytics_service.get_strategy_metrics = AsyncMock(return_value=mock_metrics)

        from src.web_interface.api.analytics import get_strategy_metrics

        # Call endpoint function directly with service dependency
        result = await get_strategy_metrics(
            strategy_name=None,
            active_only=True,
            current_user=mock_current_user,
            web_analytics_service=mock_analytics_service
        )

        # Assertions
        assert len(result) == 2
        assert result[0]["strategy_name"] == "momentum_v1"
        assert result[0]["win_rate"] == "0.60"
        assert result[1]["total_pnl"] == "890.12"


class TestOperationalMetricsEndpoints:
    """Test operational metrics endpoints."""

    @pytest.mark.asyncio
    async def test_get_operational_metrics_success(self, mock_analytics_service, mock_admin_user):
        """Test operational metrics retrieval (admin only)."""
        # Setup mock response
        mock_metrics = {
            "system_uptime": 86400,  # 1 day in seconds
            "total_api_calls": 15000,
            "average_response_time_ms": 45.2,
            "error_rate": 0.002,
            "database_connections": 25,
            "cache_hit_rate": 0.85,
            "message_queue_depth": 150,
            "active_websocket_connections": 42,
            "cpu_usage_percent": 35.5,
            "memory_usage_percent": 62.3,
            "disk_usage_percent": 45.8,
            "last_updated": datetime.now(timezone.utc)
        }
        mock_analytics_service.get_operational_metrics = AsyncMock(return_value=mock_metrics)

        from src.web_interface.api.analytics import get_operational_metrics

        # Call endpoint function directly with service dependency
        result = await get_operational_metrics(
            current_user=mock_admin_user,
            web_analytics_service=mock_analytics_service
        )

        # Assertions
        assert result["system_uptime"] == 86400
        assert result["cache_hit_rate"] == 0.85
        assert result["active_websocket_connections"] == 42
        mock_analytics_service.get_operational_metrics.assert_called_once()


class TestPortfolioCompositionEndpoints:
    """Test portfolio composition endpoints."""

    @pytest.mark.asyncio
    async def test_get_portfolio_composition_success(self, mock_analytics_service, mock_current_user):
        """Test portfolio composition retrieval."""
        # Setup mock response
        mock_composition = {
            "by_asset": {
                "BTC": Decimal("0.45"),
                "ETH": Decimal("0.35"),
                "USDT": Decimal("0.20")
            },
            "by_sector": {
                "cryptocurrency": Decimal("0.80"),
                "stablecoins": Decimal("0.20")
            },
            "by_exchange": {
                "binance": Decimal("0.60"),
                "coinbase": Decimal("0.40")
            },
            "concentration_risk": Decimal("0.45"),
            "diversification_ratio": Decimal("2.2"),
            "effective_assets": 3,
            "last_updated": datetime.now(timezone.utc)
        }
        mock_analytics_service.get_portfolio_composition = AsyncMock(return_value=mock_composition)

        from src.web_interface.api.analytics import get_portfolio_composition

        # Call endpoint function directly with service dependency
        result = await get_portfolio_composition(
            current_user=mock_current_user,
            web_analytics_service=mock_analytics_service
        )

        # Assertions
        assert "by_asset" in result
        assert result["by_asset"]["BTC"] == Decimal("0.45")
        assert result["concentration_risk"] == Decimal("0.45")
        assert result["effective_assets"] == 3


class TestCorrelationMatrixEndpoints:
    """Test correlation matrix endpoints."""

    @pytest.mark.asyncio
    async def test_get_correlation_matrix_success(self, mock_analytics_service, mock_current_user):
        """Test correlation matrix retrieval."""
        # Setup mock response
        mock_matrix = {
            "assets": ["BTC", "ETH", "BNB"],
            "matrix": [
                [Decimal("1.00"), Decimal("0.85"), Decimal("0.72")],
                [Decimal("0.85"), Decimal("1.00"), Decimal("0.78")],
                [Decimal("0.72"), Decimal("0.78"), Decimal("1.00")]
            ],
            "period_days": 30,
            "calculation_method": "pearson",
            "last_updated": datetime.now(timezone.utc)
        }
        mock_analytics_service.get_correlation_matrix = AsyncMock(return_value=mock_matrix)

        from src.web_interface.api.analytics import get_correlation_matrix

        # Call endpoint function directly with service dependency
        result = await get_correlation_matrix(
            assets=["BTC", "ETH", "BNB"],
            period_days=30,
            current_user=mock_current_user,
            web_analytics_service=mock_analytics_service
        )

        # Assertions
        assert "matrix" in result
        assert len(result["matrix"]) == 3
        assert result["matrix"][0][1] == Decimal("0.85")
        assert result["calculation_method"] == "pearson"


# Integration test class
class TestAnalyticsAPIIntegration:
    """Integration tests for Analytics API."""

    @pytest.mark.asyncio
    async def test_full_analytics_workflow(self, mock_analytics_service, mock_current_user):
        """Test complete analytics workflow."""
        # This would test a complete workflow:
        # 1. Get portfolio metrics
        # 2. Check for alerts
        # 3. Generate report if needed
        # 4. Export data
        
        # Setup all mocks
        mock_analytics_service.get_portfolio_metrics = AsyncMock(return_value={
            "total_value": "10000.00",
            "total_pnl": "123.45",
            "total_pnl_percentage": "1.25",
            "win_rate": "0.60",
            "sharpe_ratio": "1.85",
            "max_drawdown": "0.08",
            "positions_count": 5,
            "active_strategies": 2,
            "timestamp": datetime.now(timezone.utc)
        })
        
        mock_analytics_service.get_alerts = AsyncMock(return_value=[
            {"alert_id": "alert_001", "severity": "high"}
        ])
        
        mock_analytics_service.generate_report = AsyncMock(return_value={
            "report_id": "report_001",
            "report_type": "alert_triggered"
        })
        
        mock_analytics_service.export_data = AsyncMock(return_value={
            "format": "json",
            "data": {"report": "data"}
        })

        from src.web_interface.api.analytics import (
            get_portfolio_metrics,
            get_alerts,
            generate_report,
            export_portfolio_data
        )

        # Call functions directly with service dependency
        # Step 1: Get metrics
        metrics = await get_portfolio_metrics(
            current_user=mock_current_user,
            web_analytics_service=mock_analytics_service
        )
        assert hasattr(metrics, 'total_value')  # It's a Pydantic model, not a dict
        
        # Step 2: Check alerts
        alerts = await get_alerts(
            severity="high", 
            acknowledged=False, 
            limit=10, 
            current_user=mock_current_user,
            web_analytics_service=mock_analytics_service
        )
        assert len(alerts) == 1
        
        # Step 3: Generate report due to high severity alert
        if alerts and alerts[0]["severity"] == "high":
            report = await generate_report(
                report_type="alert_triggered",
                start_date=datetime.now(timezone.utc) - timedelta(days=1),
                end_date=datetime.now(timezone.utc),
                current_user=mock_current_user,
                web_analytics_service=mock_analytics_service
            )
            assert report["report_type"] == "alert_triggered"
        
        # Step 4: Export data
        export = await export_portfolio_data(
            format="json", 
            include_metadata=True, 
            current_user=mock_current_user,
            web_analytics_service=mock_analytics_service
        )
        assert export["format"] == "json"