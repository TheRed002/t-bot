"""
Performance Attribution for Backtesting.

This module provides detailed performance attribution analysis to understand
the sources of returns and risk in trading strategies.
"""

from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from src.core.base.component import BaseComponent
from src.utils.attribution_structures import (
    create_empty_attribution_structure,
)
from src.utils.config_conversion import convert_config_to_dict
from src.utils.decimal_utils import to_decimal
from src.utils.decorators import time_execution
from src.utils.financial_constants import DEFAULT_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


class PerformanceAttributor(BaseComponent):
    """
    Analyzes and attributes performance to various factors.

    Provides:
    - Return attribution by symbol/sector
    - Risk factor decomposition
    - Timing vs selection analysis
    - Cost analysis
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize performance attributor."""
        # Convert config to dict using shared utility
        config_dict = convert_config_to_dict(config)
        super().__init__(name="PerformanceAttributor", config=config_dict)  # type: ignore
        self.config = config
        self.logger.info("PerformanceAttributor initialized")

    @time_execution
    def attribute_returns(
        self,
        trades: list[dict[str, Any]],
        market_returns: dict[str, pd.Series],
        risk_free_rate: Decimal | None = None,
    ) -> dict[str, Any]:
        """
        Perform comprehensive return attribution.

        Args:
            trades: List of executed trades
            market_returns: Market returns for each symbol
            risk_free_rate: Risk-free rate for excess return calculation

        Returns:
            Attribution analysis results
        """
        if risk_free_rate is None:
            risk_free_rate = to_decimal(str(DEFAULT_RISK_FREE_RATE))

        if not trades:
            return self._empty_attribution()

        self.logger.info("Starting return attribution", num_trades=len(trades))

        # Group trades by symbol
        try:
            symbol_trades: dict[str, list[dict[str, Any]]] = self._group_trades_by_symbol(trades)
        except (KeyError, TypeError) as e:
            self.logger.error(f"Failed to group trades by symbol: {e}")
            return self._empty_attribution()

        # Calculate attributions
        try:
            symbol_attribution = self._attribute_by_symbol(symbol_trades)
            timing_attribution = self._timing_vs_selection(trades, market_returns)
            factor_attribution = self._factor_decomposition(trades, market_returns)
            cost_attribution = self._cost_analysis(trades)
        except Exception as e:
            self.logger.error(f"Attribution calculation failed: {e}")
            raise

        # Aggregate results
        try:
            results = {
                "symbol_attribution": symbol_attribution,
                "timing_attribution": timing_attribution,
                "factor_attribution": factor_attribution,
                "cost_attribution": cost_attribution,
                "summary": self._calculate_summary(
                    symbol_attribution, timing_attribution, cost_attribution
                ),
            }
        except Exception as e:
            self.logger.error(f"Failed to aggregate attribution results: {e}")
            raise

        self.logger.info("Return attribution completed")
        return results

    def _empty_attribution(self) -> dict[str, Any]:
        """Return empty attribution structure."""
        return create_empty_attribution_structure()

    def _group_trades_by_symbol(
        self, trades: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Group trades by symbol."""
        symbol_trades: dict[str, list[dict[str, Any]]] = {}

        for trade in trades:
            symbol = trade["symbol"]
            if symbol not in symbol_trades:
                symbol_trades[symbol] = []
            symbol_trades[symbol].append(trade)

        return symbol_trades

    def _attribute_by_symbol(
        self, symbol_trades: dict[str, list[dict[str, Any]]]
    ) -> dict[str, Any]:
        """Attribute returns by symbol."""
        attribution = {}

        total_pnl = 0.0
        total_trades = 0

        for symbol, trades in symbol_trades.items():
            symbol_pnl = sum(to_decimal(str(t["pnl"])) for t in trades)
            symbol_return = (
                symbol_pnl / to_decimal(str(abs(trades[0]["size"]))) if trades else to_decimal("0")
            )

            winning_trades = [t for t in trades if to_decimal(str(t["pnl"])) > 0]
            losing_trades = [t for t in trades if to_decimal(str(t["pnl"])) <= 0]

            attribution[symbol] = {
                "total_pnl": symbol_pnl,
                "return_pct": symbol_return * to_decimal("100"),
                "num_trades": len(trades),
                "win_rate": to_decimal(len(winning_trades)) / to_decimal(len(trades))
                if trades
                else to_decimal("0"),
                "avg_win": to_decimal(str(np.mean([float(t["pnl"]) for t in winning_trades])))
                if winning_trades
                else to_decimal("0"),
                "avg_loss": to_decimal(str(np.mean([float(t["pnl"]) for t in losing_trades])))
                if losing_trades
                else to_decimal("0"),
                "contribution_pct": to_decimal("0"),  # Will be calculated after
            }

            total_pnl += symbol_pnl
            total_trades += len(trades)

        # Calculate contribution percentages
        if total_pnl != 0:
            for symbol in attribution:
                attribution[symbol]["contribution_pct"] = (
                    attribution[symbol]["total_pnl"] / abs(total_pnl) * to_decimal("100")
                )

        # Add summary
        filtered_attribution = {k: v for k, v in attribution.items() if k != "_summary"}
        top_contributor: str | None = None
        worst_performer: str | None = None

        if filtered_attribution:
            top_contributor = max(filtered_attribution.items(), key=lambda x: x[1]["total_pnl"])[0]
            worst_performer = min(filtered_attribution.items(), key=lambda x: x[1]["total_pnl"])[0]

        attribution["_summary"] = {
            "total_pnl": float(total_pnl),
            "total_trades": total_trades,
            "top_contributor": top_contributor or "",  # type: ignore
            "worst_performer": worst_performer or "",  # type: ignore
        }

        return attribution

    def _timing_vs_selection(
        self,
        trades: list[dict[str, Any]],
        market_returns: dict[str, pd.Series],
    ) -> dict[str, Any]:
        """Analyze timing vs selection skill."""
        if not trades or not market_returns:
            return {}

        # Calculate average holding period returns
        strategy_returns = []
        market_period_returns = []

        for trade in trades:
            symbol = trade["symbol"]

            if symbol not in market_returns:
                continue

            # Strategy return
            strategy_return = to_decimal(str(trade["pnl"])) / to_decimal(str(trade["size"]))
            strategy_returns.append(strategy_return)

            # Market return for same period
            entry_time = trade["entry_time"]
            exit_time = trade["exit_time"]

            market_data = market_returns[symbol]

            # Find closest timestamps
            entry_idx = market_data.index.get_indexer([entry_time], method="nearest")[0]
            exit_idx = market_data.index.get_indexer([exit_time], method="nearest")[0]

            if exit_idx > entry_idx:
                period_return = to_decimal(
                    str(
                        (market_data.iloc[exit_idx] - market_data.iloc[entry_idx])
                        / market_data.iloc[entry_idx]
                    )
                )
            else:
                period_return = to_decimal("0.0")

            market_period_returns.append(period_return)

        # Calculate timing and selection components
        avg_strategy_return = (
            to_decimal(str(np.mean([float(r) for r in strategy_returns])))
            if strategy_returns
            else to_decimal("0")
        )
        avg_market_return = (
            to_decimal(str(np.mean([float(r) for r in market_period_returns])))
            if market_period_returns
            else to_decimal("0")
        )

        # Timing: Ability to enter/exit at right times
        timing_skill = avg_strategy_return - avg_market_return

        # Selection: Ability to pick outperforming assets
        # (Simplified - would need benchmark for proper calculation)
        selection_skill = avg_market_return

        return {
            "average_strategy_return": avg_strategy_return * to_decimal("100"),
            "average_market_return": avg_market_return * to_decimal("100"),
            "timing_skill": timing_skill * to_decimal("100"),
            "selection_skill": selection_skill * to_decimal("100"),
            "timing_contribution_pct": (
                abs(timing_skill) / (abs(timing_skill) + abs(selection_skill)) * to_decimal("100")
                if (timing_skill + selection_skill) != 0
                else to_decimal("50.0")
            ),
        }

    def _factor_decomposition(
        self,
        trades: list[dict[str, Any]],
        market_returns: dict[str, pd.Series],
    ) -> dict[str, Any]:
        """Decompose returns by risk factors."""
        if not trades:
            return {}

        # Simplified factor model
        # In production, would use actual factor returns (momentum, value, etc.)

        returns = [float(to_decimal(str(t["pnl"])) / to_decimal(str(t["size"]))) for t in trades]

        if not returns:
            return {}

        # Calculate basic statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Estimate factor exposures (simplified)
        # Market factor (beta)
        market_vals = []
        if market_returns:
            for _symbol, data in market_returns.items():
                if len(data) > 1:
                    market_vals.extend(data.pct_change().dropna().tolist())

            if market_vals and len(returns) == len(market_vals):
                beta = np.cov(returns, market_vals)[0, 1] / np.var(market_vals)
            else:
                beta = 1.0
        else:
            beta = 1.0

        # Alpha (excess return)
        daily_risk_free_rate = (
            float(DEFAULT_RISK_FREE_RATE) / TRADING_DAYS_PER_YEAR
        )  # Daily risk-free rate
        market_return = np.mean(market_vals) if market_vals else mean_return
        alpha = mean_return - (daily_risk_free_rate + beta * (market_return - daily_risk_free_rate))

        # Decomposition
        market_contribution = beta * market_return
        alpha_contribution = alpha
        residual = mean_return - (market_contribution + alpha_contribution)

        return {
            "total_return": mean_return * to_decimal("100"),
            "market_contribution": market_contribution * to_decimal("100"),
            "alpha_contribution": alpha_contribution * to_decimal("100"),
            "residual": residual * to_decimal("100"),
            "beta": beta,
            "alpha_annual": alpha * to_decimal(str(TRADING_DAYS_PER_YEAR)) * to_decimal("100"),
            "information_ratio": to_decimal(
                str(alpha / std_return * np.sqrt(TRADING_DAYS_PER_YEAR))
            )
            if std_return > 0
            else to_decimal("0"),
        }

    def _cost_analysis(self, trades: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze trading costs and their impact."""
        if not trades:
            return {}

        total_pnl = sum(to_decimal(str(t["pnl"])) for t in trades)

        # Extract cost components (if available in trade data)
        commission_costs = to_decimal("0.0")
        slippage_costs = to_decimal("0.0")
        spread_costs = to_decimal("0.0")

        for trade in trades:
            # Estimate costs (would be tracked in real system)
            size = to_decimal(str(trade["size"]))

            # Commission (assumed 0.1%)
            commission = size * to_decimal("0.001")
            commission_costs += commission

            # Slippage (if available)
            if "slippage" in trade:
                slippage_costs += to_decimal(str(trade["slippage"])) * size

            # Spread (estimated)
            spread = size * to_decimal("0.0005")
            spread_costs += spread

        total_costs = commission_costs + slippage_costs + spread_costs
        gross_pnl = total_pnl + total_costs

        return {
            "gross_pnl": gross_pnl,
            "total_costs": total_costs,
            "net_pnl": total_pnl,
            "commission_costs": commission_costs,
            "slippage_costs": slippage_costs,
            "spread_costs": spread_costs,
            "cost_as_pct_of_gross": total_costs / abs(gross_pnl) * to_decimal("100")
            if gross_pnl != 0
            else to_decimal("0"),
            "cost_per_trade": total_costs / to_decimal(len(trades)) if trades else to_decimal("0"),
        }

    def _calculate_summary(
        self,
        symbol_attr: dict[str, Any],
        timing_attr: dict[str, Any],
        cost_attr: dict[str, Any],
    ) -> dict[str, Any]:
        """Calculate summary statistics."""
        total_return = 0
        alpha = 0
        beta = 1.0

        # Get total return from symbol attribution
        if "_summary" in symbol_attr:
            total_pnl = symbol_attr["_summary"].get("total_pnl", 0)
            # Assuming initial capital of 10000 for percentage
            total_return = total_pnl / 10000 * 100

        # Get alpha from timing attribution
        if timing_attr:
            alpha = timing_attr.get("timing_skill", 0)

        return {
            "total_return": to_decimal(str(total_return)),
            "alpha": to_decimal(str(alpha)),
            "beta": to_decimal(str(beta)),
            "sharpe_ratio": to_decimal("0"),  # Would need full return series
            "information_ratio": to_decimal("0"),  # Would need benchmark
        }

    def calculate_rolling_attribution(
        self,
        trades: list[dict[str, Any]],
        window_days: int = 30,
    ) -> pd.DataFrame:
        """
        Calculate rolling attribution metrics.

        Args:
            trades: List of trades
            window_days: Rolling window size in days

        Returns:
            DataFrame with rolling attribution metrics
        """
        if not trades:
            return pd.DataFrame()

        # Convert trades to DataFrame
        df = pd.DataFrame(trades)
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        df.set_index("exit_time", inplace=True)
        df.sort_index(inplace=True)

        # Calculate daily P&L
        daily_pnl = df.groupby(df.index.date)["pnl"].sum()
        daily_pnl.index = pd.to_datetime(daily_pnl.index)

        # Create rolling windows
        rolling = daily_pnl.rolling(window=f"{window_days}D")

        # Calculate rolling metrics
        results = pd.DataFrame(index=daily_pnl.index)
        results["rolling_pnl"] = rolling.sum()
        results["rolling_return"] = rolling.mean() * TRADING_DAYS_PER_YEAR  # Annualized
        results["rolling_volatility"] = rolling.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        results["rolling_sharpe"] = results["rolling_return"] / results["rolling_volatility"]

        # Rolling win rate
        df["is_win"] = df["pnl"] > 0
        daily_wins = df.groupby(df.index.date)["is_win"].mean()
        daily_wins.index = pd.to_datetime(daily_wins.index)
        results["rolling_win_rate"] = daily_wins.rolling(window=f"{window_days}D").mean()

        return results

    def generate_attribution_report(self, attribution_results: dict[str, Any]) -> str:
        """
        Generate a text report of attribution results.

        Args:
            attribution_results: Attribution analysis results

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("PERFORMANCE ATTRIBUTION REPORT")
        report.append("=" * 60)

        # Symbol Attribution
        if "symbol_attribution" in attribution_results:
            report.append("\n--- Symbol Attribution ---")
            symbol_attr = attribution_results["symbol_attribution"]

            if "_summary" in symbol_attr:
                summary = symbol_attr["_summary"]
                report.append(f"Total P&L: ${summary.get('total_pnl', 0):,.2f}")
                report.append(f"Total Trades: {summary.get('total_trades', 0)}")
                report.append(f"Top Contributor: {summary.get('top_contributor', 'N/A')}")
                report.append(f"Worst Performer: {summary.get('worst_performer', 'N/A')}")

            report.append("\nPer Symbol:")
            for symbol, metrics in symbol_attr.items():
                if symbol != "_summary":
                    report.append(f"  {symbol}:")
                    report.append(f"    Return: {metrics['return_pct']:.2f}%")
                    report.append(f"    Win Rate: {metrics['win_rate'] * 100:.1f}%")
                    report.append(f"    Contribution: {metrics['contribution_pct']:.1f}%")

        # Timing Attribution
        if "timing_attribution" in attribution_results:
            report.append("\n--- Timing vs Selection ---")
            timing = attribution_results["timing_attribution"]

            if timing:
                report.append(f"Strategy Return: {timing.get('average_strategy_return', 0):.2f}%")
                report.append(f"Market Return: {timing.get('average_market_return', 0):.2f}%")
                report.append(f"Timing Skill: {timing.get('timing_skill', 0):.2f}%")
                report.append(f"Selection Skill: {timing.get('selection_skill', 0):.2f}%")

        # Cost Analysis
        if "cost_attribution" in attribution_results:
            report.append("\n--- Cost Analysis ---")
            costs = attribution_results["cost_attribution"]

            if costs:
                report.append(f"Gross P&L: ${costs.get('gross_pnl', 0):,.2f}")
                report.append(f"Total Costs: ${costs.get('total_costs', 0):,.2f}")
                report.append(f"Net P&L: ${costs.get('net_pnl', 0):,.2f}")
                report.append(f"Costs as % of Gross: {costs.get('cost_as_pct_of_gross', 0):.2f}%")

        report.append("\n" + "=" * 60)

        return "\n".join(report)

    async def analyze(
        self, simulation_result: dict[str, Any], market_data: dict[str, pd.DataFrame]
    ) -> dict[str, Any]:
        """
        Analyze performance attribution for service interface.

        Args:
            simulation_result: Simulation results with trades and equity curve
            market_data: Historical market data

        Returns:
            Dictionary with comprehensive attribution analysis
        """
        trades = simulation_result.get("trades", [])

        if not trades:
            return {
                "asset_allocation": {},
                "security_selection": {},
                "timing_effects": {},
                "factor_attribution": {},
                "regime_analysis": {},
            }

        # Convert market data to the expected format
        market_returns = {}
        for symbol, df in market_data.items():
            if "close" in df.columns:
                market_returns[symbol] = df["close"].pct_change().dropna()

        # Use existing attribution methods
        attribution_results = self.attribute_returns(trades, market_returns)

        # Reformat to expected structure
        return {
            "asset_allocation": attribution_results.get("symbol_attribution", {}),
            "security_selection": attribution_results.get("timing_attribution", {}),
            "timing_effects": attribution_results.get("timing_attribution", {}),
            "factor_attribution": attribution_results.get("factor_attribution", {}),
            "regime_analysis": {"summary": attribution_results.get("summary", {})},
        }
