"""
Portfolio service for web interface business logic.

This service handles all portfolio-related business logic that was previously
embedded in controllers, ensuring proper separation of concerns.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from src.core.base import BaseComponent
from src.core.exceptions import ServiceError
from src.core.logging import get_logger
from src.web_interface.interfaces import WebPortfolioServiceInterface

logger = get_logger(__name__)


class WebPortfolioService(BaseComponent):
    """Service handling portfolio business logic for web interface."""

    def __init__(self, portfolio_facade=None):
        super().__init__()
        self.portfolio_facade = portfolio_facade

    async def initialize(self) -> None:
        """Initialize the service."""
        self.logger.info("Web portfolio service initialized")

    async def cleanup(self) -> None:
        """Cleanup the service."""
        self.logger.info("Web portfolio service cleaned up")

    async def get_portfolio_summary_data(self) -> dict[str, Any]:
        """Get processed portfolio summary data with business logic."""
        try:
            if self.portfolio_facade:
                portfolio_summary = await self.portfolio_facade.get_portfolio_summary()
            else:
                # Mock data for development
                portfolio_summary = {
                    "total_value": Decimal("10000.00"),
                    "available_balance": Decimal("5000.00"),
                    "unrealized_pnl": Decimal("123.45"),
                    "daily_pnl": Decimal("67.89"),
                    "daily_pnl_percent": Decimal("0.68"),
                    "positions": [
                        {
                            "symbol": "BTC/USDT",
                            "size": Decimal("0.1"),
                            "entry_price": Decimal("45000"),
                            "current_price": Decimal("46000"),
                            "pnl": Decimal("100.00"),
                        }
                    ],
                    "running_bots": 2,
                    "total_trades": 25,
                    "average_win_rate": 0.72,
                }

            # Business logic: process and enrich the data
            total_value = portfolio_summary.get("total_value", Decimal("0"))
            total_pnl = portfolio_summary.get("unrealized_pnl", Decimal("0"))
            total_pnl_percentage = (
                (total_pnl / total_value * 100) if total_value > 0 else Decimal("0")
            )

            positions = portfolio_summary.get("positions", [])
            positions_count = len([pos for pos in positions if pos.get("symbol")])
            active_bots = portfolio_summary.get("running_bots", 0)

            return {
                "total_value": total_value,
                "total_pnl": total_pnl,
                "total_pnl_percentage": total_pnl_percentage,
                "positions_count": positions_count,
                "active_bots": active_bots,
                "raw_data": portfolio_summary,
            }

        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            raise ServiceError(f"Failed to get portfolio summary: {e}")

    async def calculate_pnl_periods(
        self, total_pnl: Decimal, total_trades: int, win_rate: float
    ) -> dict[str, dict[str, Any]]:
        """Calculate P&L data for different periods with business logic."""
        try:
            periods = {
                "daily": {
                    "multiplier": Decimal("0.1"),
                    "trade_divisor": 7,
                    "max_drawdown_percent": -2.5,
                    "sharpe_ratio": 1.2,
                    "profit_factor": 2.0,
                },
                "weekly": {
                    "multiplier": Decimal("0.3"),
                    "trade_divisor": 4,
                    "max_drawdown_percent": -4.2,
                    "sharpe_ratio": 1.35,
                    "profit_factor": 2.2,
                },
                "monthly": {
                    "multiplier": Decimal("1.0"),
                    "trade_divisor": 1,
                    "max_drawdown_percent": -8.3,
                    "sharpe_ratio": 1.45,
                    "profit_factor": 2.1,
                },
            }

            results = {}
            for period_name, config in periods.items():
                period_pnl = total_pnl * Decimal(str(config["multiplier"]))
                period_trades = total_trades // int(config["trade_divisor"])
                winning_trades = int(period_trades * win_rate)
                losing_trades = period_trades - winning_trades

                # Business logic for calculating averages
                base_win = Decimal("150.00")
                base_loss = Decimal("-75.00")
                multiplier = Decimal(str(config["multiplier"]))

                average_win = base_win * (Decimal("1") + multiplier)
                average_loss = base_loss * (Decimal("1") + multiplier)

                results[period_name] = {
                    "total_pnl": period_pnl,
                    "realized_pnl": period_pnl * Decimal("0.8"),
                    "unrealized_pnl": period_pnl * Decimal("0.2"),
                    "number_of_trades": period_trades,
                    "winning_trades": winning_trades,
                    "losing_trades": losing_trades,
                    "win_rate": win_rate,
                    "average_win": average_win,
                    "average_loss": average_loss,
                    "profit_factor": config["profit_factor"],
                    "sharpe_ratio": config["sharpe_ratio"],
                    "max_drawdown": period_pnl * Decimal("-0.15"),
                    "max_drawdown_percentage": config["max_drawdown_percent"],
                }

            return results

        except Exception as e:
            self.logger.error(f"Error calculating P&L periods: {e}")
            raise ServiceError(f"Failed to calculate P&L periods: {e}")

    async def get_processed_positions(
        self, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Get positions with business logic processing and filtering."""
        try:
            filters = filters or {}
            positions = []

            if self.portfolio_facade:
                portfolio_summary = await self.portfolio_facade.get_portfolio_summary()
                bot_positions_list = portfolio_summary.get("positions", [])
            else:
                # Mock data for development
                bot_positions_list = [
                    {
                        "symbol": "BTCUSDT",
                        "exchange": "binance",
                        "side": "long",
                        "quantity": Decimal("0.1"),
                        "entry_price": Decimal("45000.00"),
                        "current_price": Decimal("47000.00"),
                        "created_at": datetime.now(timezone.utc) - timedelta(hours=2),
                    },
                    {
                        "symbol": "ETHUSDT",
                        "exchange": "binance",
                        "side": "long",
                        "quantity": Decimal("2.5"),
                        "entry_price": Decimal("3000.00"),
                        "current_price": Decimal("3100.00"),
                        "created_at": datetime.now(timezone.utc) - timedelta(hours=1),
                    },
                ]

            # Business logic: process positions and apply filters
            for i, position in enumerate(bot_positions_list):
                # Apply filters
                if filters.get("exchange") and position.get("exchange") != filters["exchange"]:
                    continue
                if filters.get("symbol") and position.get("symbol") != filters["symbol"]:
                    continue
                if filters.get("bot_id") and f"bot_{i:03d}" != filters["bot_id"]:
                    continue

                # Business logic: calculate position metrics
                quantity = position.get("quantity", Decimal("0"))
                entry_price = position.get("entry_price", Decimal("0"))
                current_price = position.get("current_price", Decimal("0"))

                market_value = quantity * current_price
                cost_basis = quantity * entry_price
                unrealized_pnl = market_value - cost_basis
                unrealized_pnl_percentage = (
                    (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else Decimal("0")
                )

                processed_position = {
                    "symbol": position.get("symbol", ""),
                    "exchange": position.get("exchange", ""),
                    "side": position.get("side", "long"),
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "market_value": market_value,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_percentage": unrealized_pnl_percentage,
                    "cost_basis": cost_basis,
                    "created_at": position.get("created_at", datetime.now(timezone.utc)),
                    "updated_at": datetime.now(timezone.utc),
                    "bot_id": f"bot_{i:03d}",
                }
                positions.append(processed_position)

            return positions

        except Exception as e:
            self.logger.error(f"Error processing positions: {e}")
            raise ServiceError(f"Failed to process positions: {e}")

    async def calculate_pnl_metrics(self, period: str) -> dict[str, Any]:
        """Calculate P&L metrics for a specific period with business logic."""
        try:
            # Calculate date range based on period
            end_date = datetime.now(timezone.utc)
            period_days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "1y": 365}
            days = period_days.get(period, 30)
            start_date = end_date - timedelta(days=days)

            if self.portfolio_facade:
                pnl_report = await self.portfolio_facade.get_pnl_report(start_date, end_date)
            else:
                # Mock data for development
                pnl_report = {
                    "total_pnl": Decimal("567.89"),
                    "total_trades": 25,
                    "win_rate": 0.72,
                    "max_drawdown": Decimal("-123.45"),
                    "sharpe_ratio": Decimal("1.25"),
                    "total_allocated_capital": Decimal("100000"),
                }

            # Business logic: calculate period-specific metrics
            total_pnl = pnl_report.get("total_pnl", Decimal("0"))
            total_trades = pnl_report.get("total_trades", 0)
            win_rate = pnl_report.get("win_rate", 0.0)

            # Adjust metrics based on period
            period_multipliers = {"1d": 0.1, "7d": 0.3, "30d": 1.0, "90d": 3.0, "1y": 12.0}
            multiplier = period_multipliers.get(period, 1.0)
            period_pnl = total_pnl * Decimal(str(multiplier))
            period_trades = int(total_trades * multiplier)

            # Calculate win/loss distribution
            winning_trades = int(period_trades * win_rate)
            losing_trades = period_trades - winning_trades

            # Calculate profit metrics
            realized_pnl = period_pnl * Decimal("0.8")
            unrealized_pnl = period_pnl * Decimal("0.2")

            allocated_capital = pnl_report.get("total_allocated_capital", Decimal("100000"))
            total_return_percentage = (
                (period_pnl / allocated_capital * 100) if allocated_capital > 0 else Decimal("0")
            )

            average_win = Decimal("200.00") * Decimal(str(multiplier))
            average_loss = Decimal("-100.00") * Decimal(str(multiplier))
            profit_factor = abs(average_win / average_loss) if average_loss != 0 else Decimal("0")

            max_drawdown = period_pnl * Decimal("-0.15")  # 15% of gains as max drawdown
            max_drawdown_percentage = (
                (max_drawdown / allocated_capital * 100) if allocated_capital > 0 else Decimal("0")
            )

            return {
                "period": period,
                "total_pnl": period_pnl,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_return_percentage": total_return_percentage,
                "number_of_trades": period_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "average_win": average_win,
                "average_loss": average_loss,
                "profit_factor": profit_factor,
                "sharpe_ratio": 1.2 + (multiplier * 0.1),  # Mock Sharpe ratio
                "max_drawdown": max_drawdown,
                "max_drawdown_percentage": max_drawdown_percentage,
            }

        except Exception as e:
            self.logger.error(f"Error calculating P&L metrics: {e}")
            raise ServiceError(f"Failed to calculate P&L metrics: {e}")

    def generate_mock_balances(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Generate mock balance data (business logic for development/testing)."""
        filters = filters or {}

        mock_balances = [
            {
                "exchange": "binance",
                "currency": "USDT",
                "total_balance": Decimal("50000.00"),
                "available_balance": Decimal("45000.00"),
                "locked_balance": Decimal("5000.00"),
                "usd_value": Decimal("50000.00"),
            },
            {
                "exchange": "binance",
                "currency": "BTC",
                "total_balance": Decimal("1.5"),
                "available_balance": Decimal("1.3"),
                "locked_balance": Decimal("0.2"),
                "usd_value": Decimal("70500.00"),
            },
            {
                "exchange": "binance",
                "currency": "ETH",
                "total_balance": Decimal("10.0"),
                "available_balance": Decimal("8.5"),
                "locked_balance": Decimal("1.5"),
                "usd_value": Decimal("31000.00"),
            },
            {
                "exchange": "coinbase",
                "currency": "USD",
                "total_balance": Decimal("25000.00"),
                "available_balance": Decimal("23000.00"),
                "locked_balance": Decimal("2000.00"),
                "usd_value": Decimal("25000.00"),
            },
        ]

        # Apply filters
        filtered_balances = []
        for balance_data in mock_balances:
            if filters.get("exchange") and balance_data["exchange"] != filters["exchange"]:
                continue
            if filters.get("currency") and balance_data["currency"] != filters["currency"]:
                continue
            filtered_balances.append(balance_data)

        return filtered_balances

    def calculate_asset_allocation(self) -> list[dict[str, Any]]:
        """Calculate asset allocation with business logic."""
        # Mock asset allocation calculation
        total_value = Decimal("176500.00")  # Sum of all asset values

        allocations = [
            {
                "asset": "BTC",
                "value": Decimal("70500.00"),
                "percentage": (Decimal("70500.00") / total_value * 100),
                "positions": 2,
            },
            {
                "asset": "ETH",
                "value": Decimal("31000.00"),
                "percentage": (Decimal("31000.00") / total_value * 100),
                "positions": 3,
            },
            {
                "asset": "USDT",
                "value": Decimal("50000.00"),
                "percentage": (Decimal("50000.00") / total_value * 100),
                "positions": 1,
            },
            {
                "asset": "USD",
                "value": Decimal("25000.00"),
                "percentage": (Decimal("25000.00") / total_value * 100),
                "positions": 1,
            },
        ]

        return allocations

    def generate_performance_chart_data(self, period: str, resolution: str) -> dict[str, Any]:
        """Generate performance chart data with business logic."""
        import random

        # Calculate number of data points
        resolution_minutes = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        period_hours = {"1d": 24, "7d": 168, "30d": 720, "90d": 2160, "1y": 8760}

        minutes_per_point = resolution_minutes.get(resolution, 60)
        hours_in_period = period_hours.get(period, 720)
        points = hours_in_period * 60 // minutes_per_point

        # Generate mock data with business logic
        start_time = datetime.now(timezone.utc) - timedelta(hours=hours_in_period)
        start_value = Decimal("100000.00")  # Starting portfolio value

        data_points = []
        current_value = start_value

        for i in range(points):
            timestamp = start_time + timedelta(minutes=i * minutes_per_point)

            # Business logic: simulate portfolio value changes
            change_percent = Decimal(str(random.uniform(-0.5, 0.7)))  # Slight upward bias
            current_value *= Decimal("1") + change_percent / Decimal("100")

            data_points.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "portfolio_value": current_value.quantize(Decimal("0.01")),
                    "pnl": (current_value - start_value).quantize(Decimal("0.01")),
                    "pnl_percentage": (
                        (current_value - start_value) / start_value * Decimal("100")
                    ).quantize(Decimal("0.0001")),
                }
            )

        return {
            "period": period,
            "resolution": resolution,
            "data_points": len(data_points),
            "start_value": start_value,
            "end_value": current_value,
            "total_return": (current_value - start_value).quantize(Decimal("0.01")),
            "total_return_percentage": (
                (current_value - start_value) / start_value * Decimal("100")
            ).quantize(Decimal("0.0001")),
            "data": data_points,
        }
