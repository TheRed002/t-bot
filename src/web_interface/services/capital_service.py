"""
Web Capital Management Service Implementation.

This service provides a web-specific interface to the capital management system,
handling data transformation, formatting, validation, and web-specific business logic.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from src.capital_management.interfaces import (
    AbstractCurrencyManagementService,
    AbstractFundFlowManagementService,
    CapitalServiceProtocol,
)
from src.core.base import BaseService
from src.core.exceptions import ServiceError, ValidationError
from src.core.logging import get_logger
from src.core.types import CapitalAllocation
from src.utils.decorators import cached, monitored

logger = get_logger(__name__)


class WebCapitalService(BaseService):
    """
    Web interface service for capital management operations.

    This service wraps the capital management service and provides web-specific
    formatting, validation, and business logic.
    """

    def __init__(
        self,
        capital_service: CapitalServiceProtocol,
        currency_service: AbstractCurrencyManagementService | None = None,
        fund_flow_service: AbstractFundFlowManagementService | None = None,
    ):
        """Initialize web capital service with dependencies."""
        super().__init__("WebCapitalService")
        self.capital_service = capital_service
        self.currency_service = currency_service
        self.fund_flow_service = fund_flow_service
        logger.info("Web capital service initialized")

    async def _do_start(self) -> None:
        """Start the web capital service."""
        logger.info("Starting web capital service")

    async def _do_stop(self) -> None:
        """Stop the web capital service."""
        logger.info("Stopping web capital service")

    # Capital Allocation Methods
    @monitored()
    async def allocate_capital(
        self,
        strategy_id: str,
        exchange: str,
        amount: Decimal,
        bot_id: str | None = None,
        user_id: str | None = None,
        risk_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Allocate capital to a strategy with web validation and formatting.

        Args:
            strategy_id: Strategy identifier
            exchange: Exchange name
            amount: Amount to allocate (Decimal)
            bot_id: Optional bot identifier
            user_id: User making the allocation
            risk_context: Optional risk context

        Returns:
            Formatted allocation response
        """
        try:
            # Validate amount
            if amount <= Decimal("0"):
                raise ValidationError("Allocation amount must be positive")

            # Check if amount exceeds available capital
            available = await self._get_available_capital_amount()
            if amount > available:
                raise ValidationError(
                    f"Requested amount {amount} exceeds available capital {available}"
                )

            # Perform allocation through service
            allocation = await self.capital_service.allocate_capital(
                strategy_id=strategy_id,
                exchange=exchange,
                requested_amount=amount,
                bot_id=bot_id,
                authorized_by=user_id,
                risk_context=risk_context,
            )

            # Format for web response
            return self._format_allocation(allocation)

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error allocating capital: {e}")
            raise ServiceError(f"Failed to allocate capital: {e!s}")

    @monitored()
    async def release_capital(
        self,
        strategy_id: str,
        exchange: str,
        amount: Decimal,
        bot_id: str | None = None,
        user_id: str | None = None,
        reason: str | None = None,
    ) -> bool:
        """Release allocated capital from a strategy."""
        try:
            # Validate amount
            if amount <= Decimal("0"):
                raise ValidationError("Release amount must be positive")

            # Check current allocation
            allocation = await self.get_strategy_allocation(strategy_id, exchange)
            if not allocation:
                raise ValidationError(f"No allocation found for strategy {strategy_id}")

            allocated = Decimal(allocation["allocated_amount"])
            if amount > allocated:
                raise ValidationError(f"Cannot release {amount}, only {allocated} is allocated")

            # Perform release through service
            success = await self.capital_service.release_capital(
                strategy_id=strategy_id,
                exchange=exchange,
                release_amount=amount,
                bot_id=bot_id,
                authorized_by=user_id,
            )

            # Log the reason if provided
            if success and reason:
                logger.info(f"Capital released: {reason}")

            return success

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error releasing capital: {e}")
            raise ServiceError(f"Failed to release capital: {e!s}")

    @monitored()
    async def update_utilization(
        self,
        strategy_id: str,
        exchange: str,
        utilized_amount: Decimal,
        bot_id: str | None = None,
    ) -> bool:
        """Update capital utilization for a strategy."""
        try:
            # Validate amount
            if utilized_amount < Decimal("0"):
                raise ValidationError("Utilized amount cannot be negative")

            # Check allocation exists
            allocation = await self.get_strategy_allocation(strategy_id, exchange)
            if not allocation:
                raise ValidationError(f"No allocation found for strategy {strategy_id}")

            allocated = Decimal(allocation["allocated_amount"])
            if utilized_amount > allocated:
                raise ValidationError(
                    f"Utilized amount {utilized_amount} exceeds allocated {allocated}"
                )

            # Update through service
            return await self.capital_service.update_utilization(
                strategy_id=strategy_id,
                exchange=exchange,
                utilized_amount=utilized_amount,
                bot_id=bot_id,
            )

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error updating utilization: {e}")
            raise ServiceError(f"Failed to update utilization: {e!s}")

    @cached(ttl=30)  # Cache for 30 seconds
    async def get_allocations(
        self,
        strategy_id: str | None = None,
        exchange: str | None = None,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Get capital allocations with filtering."""
        try:
            if strategy_id:
                allocations = await self.capital_service.get_allocations_by_strategy(strategy_id)
                # Filter by exchange if specified
                if exchange:
                    allocations = [a for a in allocations if a.exchange == exchange]
            else:
                # Get all allocations
                allocations = await self.capital_service.get_all_allocations()
                # Filter by exchange if specified
                if exchange:
                    allocations = [a for a in allocations if a.exchange == exchange]

            # Filter active if requested
            if active_only:
                allocations = [a for a in allocations if a.status == "active"]

            # Format for web
            return [self._format_allocation(a) for a in allocations]

        except Exception as e:
            logger.error(f"Error getting allocations: {e}")
            raise ServiceError(f"Failed to retrieve allocations: {e!s}")

    async def get_strategy_allocation(
        self, strategy_id: str, exchange: str | None = None
    ) -> dict[str, Any] | None:
        """Get capital allocation for a specific strategy."""
        try:
            allocations = await self.get_allocations(strategy_id=strategy_id, exchange=exchange)
            return allocations[0] if allocations else None

        except Exception as e:
            logger.error(f"Error getting strategy allocation: {e}")
            raise ServiceError(f"Failed to retrieve strategy allocation: {e!s}")

    # Capital Metrics Methods
    @cached(ttl=60)  # Cache for 1 minute
    @monitored()
    async def get_capital_metrics(self) -> dict[str, Any]:
        """Get overall capital metrics."""
        try:
            metrics = await self.capital_service.get_capital_metrics()

            return {
                "total_capital": str(metrics.total_capital),
                "allocated_capital": str(metrics.allocated_capital),
                "utilized_capital": str(metrics.utilized_capital),
                "available_capital": str(metrics.available_capital),
                "allocation_ratio": float(metrics.allocation_ratio),
                "utilization_ratio": float(metrics.utilization_ratio),
                "currency": metrics.currency,
                "last_updated": metrics.last_updated or datetime.utcnow(),
            }

        except Exception as e:
            logger.error(f"Error getting capital metrics: {e}")
            raise ServiceError(f"Failed to retrieve capital metrics: {e!s}")

    async def get_utilization_breakdown(self, by: str = "strategy") -> dict[str, Any]:
        """Get capital utilization breakdown."""
        try:
            metrics = await self.capital_service.get_capital_metrics()
            allocations = await self.capital_service.get_all_allocations()

            breakdown = {}

            if by == "strategy":
                # Group by strategy
                for alloc in allocations:
                    strategy = alloc.strategy_id
                    if strategy not in breakdown:
                        breakdown[strategy] = {
                            "allocated": Decimal("0"),
                            "utilized": Decimal("0"),
                            "available": Decimal("0"),
                        }
                    breakdown[strategy]["allocated"] += alloc.allocated_amount
                    breakdown[strategy]["utilized"] += alloc.utilized_amount
                    breakdown[strategy]["available"] += alloc.available_amount

            elif by == "exchange":
                # Group by exchange
                for alloc in allocations:
                    exchange = alloc.exchange
                    if exchange not in breakdown:
                        breakdown[exchange] = {
                            "allocated": Decimal("0"),
                            "utilized": Decimal("0"),
                            "available": Decimal("0"),
                        }
                    breakdown[exchange]["allocated"] += alloc.allocated_amount
                    breakdown[exchange]["utilized"] += alloc.utilized_amount
                    breakdown[exchange]["available"] += alloc.available_amount

            else:  # global
                breakdown["global"] = {
                    "allocated": metrics.allocated_capital,
                    "utilized": metrics.utilized_capital,
                    "available": metrics.available_capital,
                }

            # Format for web
            return {
                key: {
                    "allocated": str(values["allocated"]),
                    "utilized": str(values["utilized"]),
                    "available": str(values["available"]),
                    "utilization_ratio": float(
                        values["utilized"] / values["allocated"] if values["allocated"] > 0 else 0
                    ),
                }
                for key, values in breakdown.items()
            }

        except Exception as e:
            logger.error(f"Error getting utilization breakdown: {e}")
            raise ServiceError(f"Failed to retrieve utilization breakdown: {e!s}")

    async def get_available_capital(
        self, strategy_id: str | None = None, exchange: str | None = None
    ) -> dict[str, Any]:
        """Get available capital for allocation."""
        try:
            metrics = await self.capital_service.get_capital_metrics()

            # If filters provided, calculate available for specific strategy/exchange
            if strategy_id or exchange:
                allocations = await self.get_allocations(strategy_id, exchange)
                allocated = sum(Decimal(a["allocated_amount"]) for a in allocations)
                available = metrics.total_capital - allocated
            else:
                available = metrics.available_capital

            return {
                "amount": available,
                "currency": metrics.currency,
            }

        except Exception as e:
            logger.error(f"Error getting available capital: {e}")
            raise ServiceError(f"Failed to retrieve available capital: {e!s}")

    async def get_capital_exposure(self) -> dict[str, Any]:
        """Get capital exposure analysis."""
        try:
            allocations = await self.capital_service.get_all_allocations()
            metrics = await self.capital_service.get_capital_metrics()

            # Calculate exposure by various dimensions
            exposure_by_strategy = {}
            exposure_by_exchange = {}
            exposure_by_currency = {}

            for alloc in allocations:
                # By strategy
                if alloc.strategy_id not in exposure_by_strategy:
                    exposure_by_strategy[alloc.strategy_id] = Decimal("0")
                exposure_by_strategy[alloc.strategy_id] += alloc.utilized_amount

                # By exchange
                if alloc.exchange not in exposure_by_exchange:
                    exposure_by_exchange[alloc.exchange] = Decimal("0")
                exposure_by_exchange[alloc.exchange] += alloc.utilized_amount

                # By currency (assuming USD for now)
                currency = "USD"  # Would get from allocation
                if currency not in exposure_by_currency:
                    exposure_by_currency[currency] = Decimal("0")
                exposure_by_currency[currency] += alloc.utilized_amount

            return {
                "total_exposure": str(metrics.utilized_capital),
                "exposure_by_strategy": {k: str(v) for k, v in exposure_by_strategy.items()},
                "exposure_by_exchange": {k: str(v) for k, v in exposure_by_exchange.items()},
                "exposure_by_currency": {k: str(v) for k, v in exposure_by_currency.items()},
                "concentration_risk": self._calculate_concentration_risk(exposure_by_strategy),
                "timestamp": datetime.utcnow(),
            }

        except Exception as e:
            logger.error(f"Error getting capital exposure: {e}")
            raise ServiceError(f"Failed to retrieve capital exposure: {e!s}")

    # Currency Management Methods
    async def get_currency_exposure(self) -> dict[str, Any]:
        """Get currency exposure breakdown."""
        try:
            if not self.currency_service:
                raise ServiceError("Currency service not available")

            # Get current portfolio balances for exposure calculation
            allocations = await self.capital_service.get_all_allocations()

            # Group balances by currency (assuming USD default)
            balances = {"USD": {}}
            for alloc in allocations:
                balances["USD"][alloc.exchange] = alloc.utilized_amount

            # Update currency exposures
            exposures = await self.currency_service.update_currency_exposures(balances)

            # Format for web response
            exposure_data = {}
            total_exposure = Decimal("0")

            for currency, exposure in exposures.items():
                amount = exposure.exposure_amount
                total_exposure += amount
                exposure_data[currency] = {
                    "amount": str(amount),
                    "hedge_amount": str(exposure.hedge_amount),
                    "net_exposure": str(amount - exposure.hedge_amount),
                }

            # Calculate percentages
            for currency, data in exposure_data.items():
                amount = Decimal(data["amount"])
                data["percentage"] = (
                    (amount / total_exposure * Decimal("100")).quantize(Decimal("0.01"))
                    if total_exposure > 0
                    else Decimal("0.00")
                )

            return {
                "base_currency": "USD",
                "exposures": exposure_data,
                "total_exposure": str(total_exposure),
                "timestamp": datetime.utcnow(),
            }

        except Exception as e:
            logger.error(f"Error getting currency exposure: {e}")
            raise ServiceError(f"Failed to retrieve currency exposure: {e!s}")

    async def create_currency_hedge(
        self,
        base_currency: str,
        quote_currency: str,
        exposure_amount: Decimal,
        hedge_ratio: float,
        strategy: str,
        created_by: str,
    ) -> dict[str, Any]:
        """Create a currency hedge."""
        try:
            if not self.currency_service:
                raise ServiceError("Currency service not available")

            # Calculate hedge amount
            hedge_amount = exposure_amount * Decimal(str(hedge_ratio))

            # Update hedge position through service
            await self.currency_service.update_hedge_position(base_currency, hedge_amount)

            return {
                "hedge_id": f"hedge_{datetime.utcnow().timestamp()}",
                "base_currency": base_currency,
                "quote_currency": quote_currency,
                "exposure_amount": str(exposure_amount),
                "hedge_amount": str(hedge_amount),
                "hedge_ratio": hedge_ratio,
                "strategy": strategy,
                "status": "active",
                "created_by": created_by,
                "created_at": datetime.utcnow(),
            }

        except Exception as e:
            logger.error(f"Error creating currency hedge: {e}")
            raise ServiceError(f"Failed to create currency hedge: {e!s}")

    async def get_currency_rates(self, base_currency: str = "USD") -> dict[str, float]:
        """Get current currency exchange rates."""
        try:
            if not self.currency_service:
                raise ServiceError("Currency service not available")

            # Get rates from currency service
            currencies = ["EUR", "GBP", "JPY", "BTC", "ETH"]
            rates = {}

            for currency in currencies:
                rate = await self.currency_service.get_exchange_rate(base_currency, currency)
                if rate is not None:
                    rates[currency] = float(rate)

            return rates

        except Exception as e:
            logger.error(f"Error getting currency rates: {e}")
            raise ServiceError(f"Failed to retrieve currency rates: {e!s}")

    # Fund Flow Methods
    async def get_fund_flows(
        self, days: int = 30, flow_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Get fund flow history."""
        try:
            if not self.fund_flow_service:
                raise ServiceError("Fund flow service not available")

            # Get flow history from service
            flows = await self.fund_flow_service.get_flow_history(days)

            # Filter by flow type if specified
            if flow_type:
                flows = [f for f in flows if f.flow_type == flow_type]

            # Format for web response
            return [
                {
                    "id": f.flow_id,
                    "flow_type": f.flow_type,
                    "amount": str(f.amount),
                    "currency": f.currency,
                    "exchange": f.exchange,
                    "timestamp": f.timestamp,
                    "status": f.status,
                }
                for f in flows
            ]

        except Exception as e:
            logger.error(f"Error getting fund flows: {e}")
            raise ServiceError(f"Failed to retrieve fund flows: {e!s}")

    async def record_fund_flow(
        self,
        flow_type: str,
        amount: Decimal,
        currency: str,
        source: str | None,
        destination: str | None,
        reference: str | None,
        notes: str | None,
        recorded_by: str,
    ) -> dict[str, Any]:
        """Record a fund flow transaction."""
        try:
            if not self.fund_flow_service:
                raise ServiceError("Fund flow service not available")

            # Process based on flow type
            if flow_type.lower() == "deposit":
                flow = await self.fund_flow_service.process_deposit(
                    amount=amount,
                    currency=currency,
                    exchange=destination or "binance",  # Default exchange
                )
            elif flow_type.lower() == "withdrawal":
                flow = await self.fund_flow_service.process_withdrawal(
                    amount=amount,
                    currency=currency,
                    exchange=source or "binance",  # Default exchange
                    reason=notes or "withdrawal",
                )
            else:
                raise ValidationError(f"Unsupported flow type: {flow_type}")

            return {
                "id": flow.flow_id,
                "flow_type": flow.flow_type,
                "amount": str(flow.amount),
                "currency": flow.currency,
                "exchange": flow.exchange,
                "timestamp": flow.timestamp,
                "status": flow.status,
                "recorded_by": recorded_by,
                "reference": reference,
                "notes": notes,
            }

        except Exception as e:
            logger.error(f"Error recording fund flow: {e}")
            raise ServiceError(f"Failed to record fund flow: {e!s}")

    async def generate_fund_flow_report(
        self,
        start_date: datetime | None,
        end_date: datetime | None,
        format: str,
    ) -> dict[str, Any]:
        """Generate fund flow report."""
        try:
            if not self.fund_flow_service:
                raise ServiceError("Fund flow service not available")

            # Calculate days between dates
            end = end_date or datetime.utcnow()
            start = start_date or (end - timedelta(days=30))
            days = (end - start).days

            # Get flow summary from service
            summary = await self.fund_flow_service.get_flow_summary(days)

            return {
                "report_id": f"report_{datetime.utcnow().timestamp()}",
                "start_date": start,
                "end_date": end,
                "format": format,
                "status": "generated",
                "summary": summary,
                "url": f"/api/capital/flows/report/download/{datetime.utcnow().timestamp()}",
            }

        except Exception as e:
            logger.error(f"Error generating fund flow report: {e}")
            raise ServiceError(f"Failed to generate fund flow report: {e!s}")

    # Limits & Controls Methods
    async def get_capital_limits(self, limit_type: str | None = None) -> list[dict[str, Any]]:
        """Get configured capital limits."""
        try:
            # This would typically fetch from limit service
            # For now, returning mock limits
            limits = [
                {
                    "limit_id": "global_max",
                    "limit_type": "global",
                    "max_allocation": "1000000.00",
                    "max_utilization_ratio": 0.8,
                    "enabled": True,
                },
                {
                    "limit_id": "strategy_trend_following",
                    "limit_type": "strategy",
                    "max_allocation": "100000.00",
                    "max_utilization_ratio": 0.6,
                    "enabled": True,
                },
            ]

            if limit_type:
                limits = [l for l in limits if l["limit_type"] == limit_type]

            return limits

        except Exception as e:
            logger.error(f"Error getting capital limits: {e}")
            raise ServiceError(f"Failed to retrieve capital limits: {e!s}")

    async def update_capital_limits(
        self,
        limit_type: str,
        limit_id: str,
        max_allocation: Decimal | None,
        max_utilization_ratio: float | None,
        max_concentration: float | None,
        enabled: bool,
        updated_by: str,
    ) -> bool:
        """Update capital limits configuration."""
        try:
            # This would typically update through limit service
            logger.info(f"Updating capital limit {limit_id} of type {limit_type} by {updated_by}")
            return True

        except Exception as e:
            logger.error(f"Error updating capital limits: {e}")
            raise ServiceError(f"Failed to update capital limits: {e!s}")

    async def get_limit_breaches(
        self, hours: int = 24, severity: str | None = None
    ) -> list[dict[str, Any]]:
        """Get recent capital limit breaches."""
        try:
            # This would typically fetch from breach monitoring service
            # For now, returning empty list
            return []

        except Exception as e:
            logger.error(f"Error getting limit breaches: {e}")
            raise ServiceError(f"Failed to retrieve limit breaches: {e!s}")

    # Helper Methods
    def _format_allocation(self, allocation: CapitalAllocation) -> dict[str, Any]:
        """Format capital allocation for web response."""
        return {
            "allocation_id": allocation.allocation_id,
            "strategy_id": allocation.strategy_id,
            "exchange": allocation.exchange,
            "allocated_amount": str(allocation.allocated_amount),
            "utilized_amount": str(allocation.utilized_amount),
            "available_amount": str(allocation.available_amount),
            "utilization_ratio": float(allocation.utilization_ratio),
            "created_at": allocation.created_at,
            "last_updated": allocation.last_updated,
            "status": allocation.status,
        }

    async def _get_available_capital_amount(self) -> Decimal:
        """Get available capital amount as Decimal."""
        metrics = await self.capital_service.get_capital_metrics()
        return metrics.available_capital

    def _calculate_concentration_risk(self, exposure_by_strategy: dict[str, Decimal]) -> float:
        """Calculate concentration risk score (0-1)."""
        if not exposure_by_strategy:
            return 0.0

        total = sum(exposure_by_strategy.values())
        if total == 0:
            return 0.0

        # Calculate Herfindahl-Hirschman Index (HHI)
        hhi = sum((value / total) ** 2 for value in exposure_by_strategy.values())

        # Normalize to 0-1 scale
        # HHI ranges from 1/n (perfect distribution) to 1 (complete concentration)
        n = len(exposure_by_strategy)
        min_hhi = 1 / n if n > 0 else 1

        return float((hhi - min_hhi) / (1 - min_hhi) if min_hhi < 1 else 0)
