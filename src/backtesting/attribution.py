"""
Performance Attribution for Backtesting.

This module provides detailed performance attribution analysis to understand
the sources of returns and risk in trading strategies.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from src.core.logging import get_logger
from src.utils.decorators import time_execution

logger = get_logger(__name__)


class PerformanceAttributor:
    """
    Analyzes and attributes performance to various factors.
    
    Provides:
    - Return attribution by symbol/sector
    - Risk factor decomposition
    - Timing vs selection analysis
    - Cost analysis
    """

    def __init__(self):
        """Initialize performance attributor."""
        logger.info("PerformanceAttributor initialized")

    @time_execution
    def attribute_returns(
        self,
        trades: List[Dict[str, Any]],
        market_returns: Dict[str, pd.Series],
        risk_free_rate: float = 0.02,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive return attribution.

        Args:
            trades: List of executed trades
            market_returns: Market returns for each symbol
            risk_free_rate: Risk-free rate for excess return calculation

        Returns:
            Attribution analysis results
        """
        if not trades:
            return self._empty_attribution()

        logger.info("Starting return attribution", num_trades=len(trades))

        # Group trades by symbol
        symbol_trades = self._group_trades_by_symbol(trades)

        # Calculate attributions
        symbol_attribution = self._attribute_by_symbol(symbol_trades)
        timing_attribution = self._timing_vs_selection(trades, market_returns)
        factor_attribution = self._factor_decomposition(trades, market_returns)
        cost_attribution = self._cost_analysis(trades)

        # Aggregate results
        results = {
            "symbol_attribution": symbol_attribution,
            "timing_attribution": timing_attribution,
            "factor_attribution": factor_attribution,
            "cost_attribution": cost_attribution,
            "summary": self._calculate_summary(
                symbol_attribution, timing_attribution, cost_attribution
            ),
        }

        logger.info("Return attribution completed")
        return results

    def _empty_attribution(self) -> Dict[str, Any]:
        """Return empty attribution structure."""
        return {
            "symbol_attribution": {},
            "timing_attribution": {},
            "factor_attribution": {},
            "cost_attribution": {},
            "summary": {
                "total_return": 0.0,
                "alpha": 0.0,
                "beta": 0.0,
            },
        }

    def _group_trades_by_symbol(
        self, trades: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group trades by symbol."""
        symbol_trades = {}
        
        for trade in trades:
            symbol = trade["symbol"]
            if symbol not in symbol_trades:
                symbol_trades[symbol] = []
            symbol_trades[symbol].append(trade)
        
        return symbol_trades

    def _attribute_by_symbol(
        self, symbol_trades: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Attribute returns by symbol."""
        attribution = {}
        
        total_pnl = 0
        total_trades = 0
        
        for symbol, trades in symbol_trades.items():
            symbol_pnl = sum(t["pnl"] for t in trades)
            symbol_return = symbol_pnl / abs(trades[0]["size"]) if trades else 0
            
            winning_trades = [t for t in trades if t["pnl"] > 0]
            losing_trades = [t for t in trades if t["pnl"] <= 0]
            
            attribution[symbol] = {
                "total_pnl": float(symbol_pnl),
                "return_pct": float(symbol_return * 100),
                "num_trades": len(trades),
                "win_rate": len(winning_trades) / len(trades) if trades else 0,
                "avg_win": np.mean([t["pnl"] for t in winning_trades]) if winning_trades else 0,
                "avg_loss": np.mean([t["pnl"] for t in losing_trades]) if losing_trades else 0,
                "contribution_pct": 0,  # Will be calculated after
            }
            
            total_pnl += symbol_pnl
            total_trades += len(trades)

        # Calculate contribution percentages
        if total_pnl != 0:
            for symbol in attribution:
                attribution[symbol]["contribution_pct"] = (
                    attribution[symbol]["total_pnl"] / abs(total_pnl) * 100
                )

        # Add summary
        attribution["_summary"] = {
            "total_pnl": float(total_pnl),
            "total_trades": total_trades,
            "top_contributor": max(attribution.items(), key=lambda x: x[1]["total_pnl"])[0]
            if attribution else None,
            "worst_performer": min(attribution.items(), key=lambda x: x[1]["total_pnl"])[0]
            if attribution else None,
        }

        return attribution

    def _timing_vs_selection(
        self,
        trades: List[Dict[str, Any]],
        market_returns: Dict[str, pd.Series],
    ) -> Dict[str, Any]:
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
            strategy_return = trade["pnl"] / trade["size"]
            strategy_returns.append(strategy_return)
            
            # Market return for same period
            entry_time = trade["entry_time"]
            exit_time = trade["exit_time"]
            
            market_data = market_returns[symbol]
            
            # Find closest timestamps
            entry_idx = market_data.index.get_indexer([entry_time], method="nearest")[0]
            exit_idx = market_data.index.get_indexer([exit_time], method="nearest")[0]
            
            if exit_idx > entry_idx:
                period_return = (
                    market_data.iloc[exit_idx] - market_data.iloc[entry_idx]
                ) / market_data.iloc[entry_idx]
            else:
                period_return = 0
            
            market_period_returns.append(period_return)

        # Calculate timing and selection components
        avg_strategy_return = np.mean(strategy_returns) if strategy_returns else 0
        avg_market_return = np.mean(market_period_returns) if market_period_returns else 0
        
        # Timing: Ability to enter/exit at right times
        timing_skill = avg_strategy_return - avg_market_return
        
        # Selection: Ability to pick outperforming assets
        # (Simplified - would need benchmark for proper calculation)
        selection_skill = avg_market_return
        
        return {
            "average_strategy_return": float(avg_strategy_return * 100),
            "average_market_return": float(avg_market_return * 100),
            "timing_skill": float(timing_skill * 100),
            "selection_skill": float(selection_skill * 100),
            "timing_contribution_pct": float(
                abs(timing_skill) / (abs(timing_skill) + abs(selection_skill)) * 100
                if (timing_skill + selection_skill) != 0 else 50
            ),
        }

    def _factor_decomposition(
        self,
        trades: List[Dict[str, Any]],
        market_returns: Dict[str, pd.Series],
    ) -> Dict[str, Any]:
        """Decompose returns by risk factors."""
        if not trades:
            return {}

        # Simplified factor model
        # In production, would use actual factor returns (momentum, value, etc.)
        
        returns = [t["pnl"] / t["size"] for t in trades]
        
        if not returns:
            return {}

        # Calculate basic statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Estimate factor exposures (simplified)
        # Market factor (beta)
        if market_returns:
            market_vals = []
            for symbol, data in market_returns.items():
                if len(data) > 1:
                    market_vals.extend(data.pct_change().dropna().tolist())
            
            if market_vals and len(returns) == len(market_vals):
                beta = np.cov(returns, market_vals)[0, 1] / np.var(market_vals)
            else:
                beta = 1.0
        else:
            beta = 1.0

        # Alpha (excess return)
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        market_return = np.mean(market_vals) if market_vals else mean_return
        alpha = mean_return - (risk_free_rate + beta * (market_return - risk_free_rate))

        # Decomposition
        market_contribution = beta * market_return
        alpha_contribution = alpha
        residual = mean_return - (market_contribution + alpha_contribution)

        return {
            "total_return": float(mean_return * 100),
            "market_contribution": float(market_contribution * 100),
            "alpha_contribution": float(alpha_contribution * 100),
            "residual": float(residual * 100),
            "beta": float(beta),
            "alpha_annual": float(alpha * 252 * 100),
            "information_ratio": float(alpha / std_return * np.sqrt(252)) if std_return > 0 else 0,
        }

    def _cost_analysis(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trading costs and their impact."""
        if not trades:
            return {}

        total_pnl = sum(t["pnl"] for t in trades)
        
        # Extract cost components (if available in trade data)
        commission_costs = 0
        slippage_costs = 0
        spread_costs = 0
        
        for trade in trades:
            # Estimate costs (would be tracked in real system)
            size = trade["size"]
            
            # Commission (assumed 0.1%)
            commission = size * 0.001
            commission_costs += commission
            
            # Slippage (if available)
            if "slippage" in trade:
                slippage_costs += trade["slippage"] * size
            
            # Spread (estimated)
            spread = size * 0.0005
            spread_costs += spread

        total_costs = commission_costs + slippage_costs + spread_costs
        gross_pnl = total_pnl + total_costs

        return {
            "gross_pnl": float(gross_pnl),
            "total_costs": float(total_costs),
            "net_pnl": float(total_pnl),
            "commission_costs": float(commission_costs),
            "slippage_costs": float(slippage_costs),
            "spread_costs": float(spread_costs),
            "cost_as_pct_of_gross": float(
                total_costs / abs(gross_pnl) * 100 if gross_pnl != 0 else 0
            ),
            "cost_per_trade": float(total_costs / len(trades)) if trades else 0,
        }

    def _calculate_summary(
        self,
        symbol_attr: Dict[str, Any],
        timing_attr: Dict[str, Any],
        cost_attr: Dict[str, Any],
    ) -> Dict[str, Any]:
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
            "total_return": float(total_return),
            "alpha": float(alpha),
            "beta": float(beta),
            "sharpe_ratio": 0,  # Would need full return series
            "information_ratio": 0,  # Would need benchmark
        }

    def calculate_rolling_attribution(
        self,
        trades: List[Dict[str, Any]],
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
        results["rolling_return"] = rolling.mean() * 252  # Annualized
        results["rolling_volatility"] = rolling.std() * np.sqrt(252)
        results["rolling_sharpe"] = (
            results["rolling_return"] / results["rolling_volatility"]
        )

        # Rolling win rate
        df["is_win"] = df["pnl"] > 0
        daily_wins = df.groupby(df.index.date)["is_win"].mean()
        daily_wins.index = pd.to_datetime(daily_wins.index)
        results["rolling_win_rate"] = daily_wins.rolling(window=f"{window_days}D").mean()

        return results

    def generate_attribution_report(
        self, attribution_results: Dict[str, Any]
    ) -> str:
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
                    report.append(f"    Win Rate: {metrics['win_rate']*100:.1f}%")
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