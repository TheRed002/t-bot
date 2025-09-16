"""
Unit tests for Capital Management API endpoints.

Tests capital allocation, fund flow management, currency operations,
and capital tracking functionality.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, status

from src.core.exceptions import CapitalAllocationError, ServiceError, ValidationError
from src.web_interface.api.capital import (
    CapitalAllocationRequest,
    CapitalReleaseRequest,
    CapitalUtilizationUpdate,
    CurrencyHedgeRequest,
    FundFlowRequest,
    CapitalLimitsUpdate,
    router,
)


@pytest.fixture
def mock_capital_service():
    """Create a mock capital management service."""
    service = MagicMock()
    service.allocate_capital = AsyncMock()
    service.release_capital = AsyncMock()
    service.get_capital_status = AsyncMock()
    service.get_allocation_history = AsyncMock()
    service.transfer_funds = AsyncMock()
    service.get_fund_flows = AsyncMock()
    service.convert_currency = AsyncMock()
    service.get_exchange_rates = AsyncMock()
    service.rebalance_portfolio = AsyncMock()
    service.set_allocation_limits = AsyncMock()
    service.get_allocation_limits = AsyncMock()
    service.calculate_optimal_allocation = AsyncMock()
    service.get_capital_efficiency = AsyncMock()
    service.reserve_capital = AsyncMock()
    service.get_reserved_capital = AsyncMock()
    return service


@pytest.fixture
def mock_current_user():
    """Create a mock authenticated user with trading permissions."""
    user = MagicMock()
    user.username = "trader_user"
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


class TestCapitalAllocationEndpoints:
    """Test capital allocation endpoints."""

    @pytest.mark.asyncio
    async def test_allocate_capital_success(self, mock_capital_service, mock_current_user):
        """Test successful capital allocation."""
        # Setup mock response to match CapitalAllocationResponse
        mock_allocation = {
            "allocation_id": "alloc_001",
            "strategy_id": "momentum_v1",
            "exchange": "binance",
            "allocated_amount": "5000.00",
            "utilized_amount": "0.00",
            "available_amount": "5000.00",
            "utilization_ratio": 0.0,
            "created_at": datetime.now(timezone.utc),
            "last_updated": datetime.now(timezone.utc),
            "status": "allocated"
        }
        mock_capital_service.allocate_capital = AsyncMock(return_value=mock_allocation)

        from src.web_interface.api.capital import allocate_capital

        # Call endpoint
        result = await allocate_capital(
            request=CapitalAllocationRequest(
                strategy_id="momentum_v1",
                exchange="binance",
                amount="5000.00"
            ),
            current_user=mock_current_user,
            web_capital_service=mock_capital_service
        )

        # Assertions
        assert result.allocation_id == "alloc_001"
        assert result.allocated_amount == "5000.00"
        assert result.status == "allocated"
        assert result.utilization_ratio == 0.0

    @pytest.mark.asyncio
    async def test_allocate_capital_insufficient_funds(self, mock_capital_service, mock_current_user):
        """Test capital allocation with insufficient funds."""
        # Setup mock to raise error
        mock_capital_service.allocate_capital.side_effect = CapitalAllocationError(
            "Insufficient capital: requested 10000.00, available 5000.00"
        )

        from src.web_interface.api.capital import allocate_capital

        # Call endpoint and expect exception
        with pytest.raises(HTTPException) as exc_info:
            await allocate_capital(
                request=CapitalAllocationRequest(
                    strategy_id="momentum_v1",
                    exchange="binance",
                    amount="10000.00"
                ),
                current_user=mock_current_user,
                web_capital_service=mock_capital_service
            )

        assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
        assert "Insufficient capital" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_release_capital_success(self, mock_capital_service, mock_current_user):
        """Test successful capital release."""
        # Setup mock response
        mock_capital_service.release_capital = AsyncMock(return_value=True)

        from src.web_interface.api.capital import release_capital

        # Call endpoint
        result = await release_capital(
            request=CapitalReleaseRequest(
                strategy_id="momentum_v1",
                exchange="binance", 
                amount="5000.00"
            ),
            current_user=mock_current_user,
            web_capital_service=mock_capital_service
        )

        # Assertions
        assert result["status"] == "success"
        assert result["message"] == "Capital released successfully"

    @pytest.mark.asyncio
    async def test_get_capital_metrics_success(self, mock_capital_service, mock_current_user):
        """Test capital metrics retrieval."""
        # Setup mock response
        mock_metrics = {
            "total_capital": "20000.00",
            "allocated_capital": "5000.00",
            "utilized_capital": "3000.00",
            "available_capital": "15000.00",
            "allocation_ratio": 0.25,
            "utilization_ratio": 0.60,
            "currency": "USDT",
            "last_updated": datetime.now(timezone.utc)
        }
        mock_capital_service.get_capital_metrics = AsyncMock(return_value=mock_metrics)

        from src.web_interface.api.capital import get_capital_metrics

        # Call endpoint
        result = await get_capital_metrics(
            current_user=mock_current_user,
            web_capital_service=mock_capital_service
        )

        # Assertions
        assert result.total_capital == "20000.00"
        assert result.allocation_ratio == 0.25
        assert result.utilization_ratio == 0.60
        assert result.currency == "USDT"


class TestFundFlowEndpoints:
    """Test fund flow management endpoints."""

    @pytest.mark.asyncio
    async def test_record_fund_flow_success(self, mock_capital_service, mock_admin_user):
        """Test successful fund flow recording."""
        # Setup mock response
        mock_transfer = {
            "transfer_id": "transfer_001",
            "from_account": "main_account",
            "to_account": "trading_account",
            "amount": Decimal("10000.00"),
            "currency": "USDT",
            "status": "completed",
            "transferred_at": datetime.now(timezone.utc),
            "fee": Decimal("10.00"),
            "net_amount": Decimal("9990.00")
        }
        mock_capital_service.transfer_funds = AsyncMock(return_value=mock_transfer)

        # Import the correct function
        from src.web_interface.api.capital import record_fund_flow
        
        # Setup mock for fund flow recording
        mock_capital_service.record_fund_flow = AsyncMock(return_value={"id": "flow_001"})
        
        # Call endpoint
        result = await record_fund_flow(
            request=FundFlowRequest(
                flow_type="transfer",
                amount="10000.00",
                currency="USDT",
                source="main_account",
                destination="trading_account"
            ),
            current_user=mock_admin_user,
            web_capital_service=mock_capital_service
        )

        # Assertions
        assert result["status"] == "success"
        assert result["flow_id"] == "flow_001"

    @pytest.mark.asyncio
    async def test_get_fund_flows_success(self, mock_capital_service, mock_current_user):
        """Test fund flow history retrieval."""
        # Setup mock response
        mock_flows = [
            {
                "flow_id": "flow_001",
                "type": "deposit",
                "amount": Decimal("5000.00"),
                "currency": "USDT",
                "from_account": "external",
                "to_account": "main_account",
                "timestamp": datetime.now(timezone.utc) - timedelta(days=2),
                "status": "completed"
            },
            {
                "flow_id": "flow_002",
                "type": "allocation",
                "amount": Decimal("3000.00"),
                "currency": "USDT",
                "from_account": "main_account",
                "to_account": "strategy_momentum_v1",
                "timestamp": datetime.now(timezone.utc) - timedelta(days=1),
                "status": "completed"
            },
            {
                "flow_id": "flow_003",
                "type": "withdrawal",
                "amount": Decimal("1000.00"),
                "currency": "USDT",
                "from_account": "main_account",
                "to_account": "external",
                "timestamp": datetime.now(timezone.utc) - timedelta(hours=12),
                "status": "pending"
            }
        ]
        mock_capital_service.get_fund_flows = AsyncMock(return_value=mock_flows)

        from src.web_interface.api.capital import get_fund_flows

        # Call endpoint
        result = await get_fund_flows(
            days=7,
            flow_type=None,
            current_user=mock_current_user,
            web_capital_service=mock_capital_service
        )

        # Assertions
        assert len(result["flows"]) == 3  # Should be 3 flows in the mock data
        assert result["flows"][0]["type"] == "deposit"
        assert result["flows"][1]["amount"] == Decimal("3000.00")
        assert result["flows"][2]["status"] == "pending"
        assert result["count"] == 3
        assert result["period_days"] == 7
        assert result["flow_type"] is None


class TestCurrencyManagementEndpoints:
    """Test currency management endpoints."""

    @pytest.mark.asyncio
    async def test_convert_currency_success(self, mock_capital_service, mock_current_user):
        """Test successful currency conversion."""
        # Setup mock response
        mock_conversion = {
            "conversion_id": "conv_001",
            "from_currency": "USDT",
            "to_currency": "BTC",
            "from_amount": Decimal("10000.00"),
            "to_amount": Decimal("0.22222"),
            "exchange_rate": Decimal("0.000022222"),
            "fee": Decimal("10.00"),
            "net_amount": Decimal("0.22200"),
            "converted_at": datetime.now(timezone.utc)
        }
        mock_capital_service.convert_currency = AsyncMock(return_value=mock_conversion)

        from src.web_interface.api.capital import convert_currency

        # Call endpoint function directly with service dependency
        result = await convert_currency(
            request={
                "from_currency": "USDT",
                "to_currency": "BTC",
                "amount": "10000.00"
            },
            current_user=mock_current_user,
            web_capital_service=mock_capital_service
        )

        # Assertions
        assert result["conversion_id"] == "conv_001"
        assert result["from_amount"] == Decimal("10000.00")
        assert result["to_amount"] == Decimal("0.22222")
        assert result["exchange_rate"] == Decimal("0.000022222")

    @pytest.mark.asyncio
    async def test_get_exchange_rates_success(self, mock_capital_service, mock_current_user):
        """Test exchange rates retrieval."""
        # Setup mock response
        mock_rates = {
            "base_currency": "USDT",
            "rates": {
                "BTC": Decimal("0.000022222"),
                "ETH": Decimal("0.0004"),
                "BNB": Decimal("0.003"),
                "USDC": Decimal("1.0001")
            },
            "last_updated": datetime.now(timezone.utc),
            "source": "binance"
        }
        mock_capital_service.get_exchange_rates = AsyncMock(return_value=mock_rates)

        from src.web_interface.api.capital import get_exchange_rates

        # Call endpoint function directly with service dependency
        result = await get_exchange_rates(
            base_currency="USDT",
            current_user=mock_current_user,
            web_capital_service=mock_capital_service
        )

        # Assertions
        assert result["base_currency"] == "USDT"
        assert "rates" in result
        assert result["rates"]["BTC"] == Decimal("0.000022222")
        assert result["source"] == "binance"


class TestPortfolioRebalancingEndpoints:
    """Test portfolio rebalancing endpoints."""

    @pytest.mark.asyncio
    async def test_rebalance_portfolio_success(self, mock_capital_service, mock_admin_user):
        """Test portfolio rebalancing (admin only)."""
        # Setup mock response
        mock_rebalance = {
            "rebalance_id": "rebal_001",
            "status": "completed",
            "changes": [
                {
                    "strategy_id": "momentum_v1",
                    "old_allocation": Decimal("5000.00"),
                    "new_allocation": Decimal("4000.00"),
                    "change": Decimal("-1000.00")
                },
                {
                    "strategy_id": "mean_reversion_v2",
                    "old_allocation": Decimal("3000.00"),
                    "new_allocation": Decimal("4000.00"),
                    "change": Decimal("1000.00")
                }
            ],
            "total_reallocated": Decimal("1000.00"),
            "rebalanced_at": datetime.now(timezone.utc)
        }
        mock_capital_service.rebalance_portfolio = AsyncMock(return_value=mock_rebalance)

        from src.web_interface.api.capital import rebalance_portfolio

        # Call endpoint function directly with service dependency
        result = await rebalance_portfolio(
            request={
                "target_allocations": {
                    "momentum_v1": "0.40",
                    "mean_reversion_v2": "0.40",
                    "arbitrage_v1": "0.20"
                },
                "dry_run": False
            },
            current_user=mock_admin_user,
            web_capital_service=mock_capital_service
        )

        # Assertions
        assert result["rebalance_id"] == "rebal_001"
        assert result["status"] == "completed"
        assert len(result["changes"]) == 2
        assert result["total_reallocated"] == Decimal("1000.00")


class TestAllocationLimitsEndpoints:
    """Test allocation limits endpoints."""

    @pytest.mark.asyncio
    async def test_set_allocation_limits_success(self, mock_capital_service, mock_admin_user):
        """Test setting allocation limits (admin only)."""
        # Setup mock response
        mock_limits = {
            "limits_id": "limits_001",
            "strategy_limits": {
                "momentum_v1": {
                    "max_allocation": Decimal("10000.00"),
                    "max_percentage": Decimal("0.50")
                },
                "mean_reversion_v2": {
                    "max_allocation": Decimal("8000.00"),
                    "max_percentage": Decimal("0.40")
                }
            },
            "global_limits": {
                "max_single_allocation": Decimal("10000.00"),
                "max_total_allocation": Decimal("18000.00"),
                "max_utilization_rate": Decimal("0.90")
            },
            "updated_at": datetime.now(timezone.utc)
        }
        mock_capital_service.set_allocation_limits = AsyncMock(return_value=mock_limits)

        from src.web_interface.api.capital import set_allocation_limits

        # Call endpoint function directly with service dependency
        result = await set_allocation_limits(
            request={
                "strategy_limits": {
                    "momentum_v1": {
                        "max_allocation": "10000.00",
                        "max_percentage": "0.50"
                    }
                },
                "global_limits": {
                    "max_single_allocation": "10000.00",
                    "max_utilization_rate": "0.90"
                }
            },
            current_user=mock_admin_user,
            web_capital_service=mock_capital_service
        )

        # Assertions
        assert result["limits_id"] == "limits_001"
        assert "strategy_limits" in result
        assert "global_limits" in result

    @pytest.mark.asyncio
    async def test_get_allocation_limits_success(self, mock_capital_service, mock_current_user):
        """Test retrieving allocation limits."""
        # Setup mock response
        mock_limits = {
            "strategy_limits": {
                "momentum_v1": {
                    "max_allocation": Decimal("10000.00"),
                    "max_percentage": Decimal("0.50"),
                    "current_allocation": Decimal("4000.00"),
                    "remaining_capacity": Decimal("6000.00")
                }
            },
            "global_limits": {
                "max_total_allocation": Decimal("18000.00"),
                "current_total_allocation": Decimal("8000.00"),
                "remaining_capacity": Decimal("10000.00")
            }
        }
        mock_capital_service.get_allocation_limits = AsyncMock(return_value=mock_limits)

        from src.web_interface.api.capital import get_allocation_limits

        # Call endpoint function directly with service dependency
        result = await get_allocation_limits(
            current_user=mock_current_user,
            web_capital_service=mock_capital_service
        )

        # Assertions
        assert "strategy_limits" in result
        assert "global_limits" in result
        assert result["global_limits"]["remaining_capacity"] == Decimal("10000.00")


class TestCapitalOptimizationEndpoints:
    """Test capital optimization endpoints."""

    @pytest.mark.asyncio
    async def test_calculate_optimal_allocation_success(self, mock_capital_service, mock_current_user):
        """Test optimal allocation calculation."""
        # Setup mock response
        mock_optimization = {
            "optimization_id": "opt_001",
            "method": "mean_variance",
            "optimal_allocations": {
                "momentum_v1": Decimal("0.35"),
                "mean_reversion_v2": Decimal("0.40"),
                "arbitrage_v1": Decimal("0.25")
            },
            "expected_return": Decimal("0.12"),
            "expected_risk": Decimal("0.08"),
            "sharpe_ratio": Decimal("1.50"),
            "confidence_level": Decimal("0.95"),
            "calculated_at": datetime.now(timezone.utc)
        }
        mock_capital_service.calculate_optimal_allocation = AsyncMock(return_value=mock_optimization)

        from src.web_interface.api.capital import calculate_optimal_allocation

        # Call endpoint function directly with service dependency
        result = await calculate_optimal_allocation(
            risk_tolerance="moderate",
            optimization_method="mean_variance",
            current_user=mock_current_user,
            web_capital_service=mock_capital_service
        )

        # Assertions
        assert result["optimization_id"] == "opt_001"
        assert result["method"] == "mean_variance"
        assert result["sharpe_ratio"] == Decimal("1.50")
        assert sum(result["optimal_allocations"].values()) == Decimal("1.00")

    @pytest.mark.asyncio
    async def test_get_capital_efficiency_success(self, mock_capital_service, mock_current_user):
        """Test capital efficiency metrics retrieval."""
        # Setup mock response
        mock_efficiency = {
            "overall_efficiency": Decimal("0.82"),
            "utilization_rate": Decimal("0.75"),
            "return_on_capital": Decimal("0.15"),
            "capital_turnover": Decimal("2.5"),
            "strategy_efficiency": {
                "momentum_v1": {
                    "efficiency": Decimal("0.88"),
                    "roi": Decimal("0.18"),
                    "turnover": Decimal("3.0")
                },
                "mean_reversion_v2": {
                    "efficiency": Decimal("0.76"),
                    "roi": Decimal("0.12"),
                    "turnover": Decimal("2.0")
                }
            },
            "recommendations": [
                "Increase allocation to momentum_v1",
                "Review mean_reversion_v2 performance"
            ],
            "calculated_at": datetime.now(timezone.utc)
        }
        mock_capital_service.get_capital_efficiency = AsyncMock(return_value=mock_efficiency)

        from src.web_interface.api.capital import get_capital_efficiency

        # Call endpoint function directly with service dependency
        result = await get_capital_efficiency(
            current_user=mock_current_user,
            web_capital_service=mock_capital_service
        )

        # Assertions
        assert result["overall_efficiency"] == Decimal("0.82")
        assert result["return_on_capital"] == Decimal("0.15")
        assert "strategy_efficiency" in result
        assert len(result["recommendations"]) == 2


class TestCapitalReservationEndpoints:
    """Test capital reservation endpoints."""

    @pytest.mark.asyncio
    async def test_reserve_capital_success(self, mock_capital_service, mock_current_user):
        """Test capital reservation."""
        # Setup mock response
        mock_reservation = {
            "reservation_id": "reserve_001",
            "purpose": "upcoming_trade",
            "amount": Decimal("2000.00"),
            "currency": "USDT",
            "reserved_until": datetime.now(timezone.utc) + timedelta(hours=1),
            "status": "reserved",
            "created_at": datetime.now(timezone.utc)
        }
        mock_capital_service.reserve_capital = AsyncMock(return_value=mock_reservation)

        from src.web_interface.api.capital import reserve_capital

        # Call endpoint function directly with service dependency
        result = await reserve_capital(
            request={
                "amount": "2000.00",
                "currency": "USDT",
                "purpose": "upcoming_trade",
                "duration_minutes": 60
            },
            current_user=mock_current_user,
            web_capital_service=mock_capital_service
        )

        # Assertions
        assert result["reservation_id"] == "reserve_001"
        assert result["amount"] == Decimal("2000.00")
        assert result["status"] == "reserved"

    @pytest.mark.asyncio
    async def test_get_reserved_capital_success(self, mock_capital_service, mock_current_user):
        """Test reserved capital retrieval."""
        # Setup mock response
        mock_reserved = {
            "total_reserved": Decimal("5000.00"),
            "reservations": [
                {
                    "reservation_id": "reserve_001",
                    "amount": Decimal("2000.00"),
                    "currency": "USDT",
                    "purpose": "upcoming_trade",
                    "reserved_until": datetime.now(timezone.utc) + timedelta(minutes=30)
                },
                {
                    "reservation_id": "reserve_002",
                    "amount": Decimal("3000.00"),
                    "currency": "USDT",
                    "purpose": "risk_buffer",
                    "reserved_until": datetime.now(timezone.utc) + timedelta(hours=2)
                }
            ]
        }
        mock_capital_service.get_reserved_capital = AsyncMock(return_value=mock_reserved)

        from src.web_interface.api.capital import get_reserved_capital

        # Call endpoint function directly with service dependency
        result = await get_reserved_capital(
            current_user=mock_current_user,
            web_capital_service=mock_capital_service
        )

        # Assertions
        assert result["total_reserved"] == Decimal("5000.00")
        assert len(result["reservations"]) == 2