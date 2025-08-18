"""
Results Analysis and Visualization for Optimization.

This module provides comprehensive analysis tools for optimization results,
including performance metrics calculation, sensitivity analysis, parameter
importance ranking, and stability analysis across different market conditions.

Key Features:
- Comprehensive performance metrics calculation
- Parameter sensitivity and importance analysis
- Stability analysis across time periods and market regimes
- Risk-adjusted performance evaluation
- Correlation analysis between parameters and outcomes
- Statistical significance testing for parameter effects
- Visualization-ready data preparation

Critical for Financial Applications:
- Risk-adjusted metrics (Sharpe, Sortino, Calmar ratios)
- Drawdown analysis and recovery metrics
- Market regime aware analysis
- Transaction cost considerations
- Liquidity impact assessment
- Regulatory compliance metrics
"""

import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field
from scipy import stats
from scipy.stats import pearsonr

from src.core.logging import get_logger
from src.utils.decorators import time_execution

logger = get_logger(__name__)


class PerformanceMetrics(BaseModel):
    """
    Comprehensive performance metrics for trading strategies.

    Contains both basic and advanced metrics for evaluating
    trading strategy performance with proper risk adjustment.
    """

    # Basic return metrics
    total_return: Decimal = Field(description="Total return over the period")
    annualized_return: Decimal = Field(description="Annualized return")
    excess_return: Decimal = Field(description="Return above risk-free rate")

    # Risk metrics
    volatility: Decimal = Field(description="Annualized volatility")
    downside_volatility: Decimal = Field(description="Downside volatility")
    max_drawdown: Decimal = Field(description="Maximum drawdown")
    current_drawdown: Decimal = Field(description="Current drawdown")

    # Risk-adjusted metrics
    sharpe_ratio: Decimal = Field(description="Sharpe ratio")
    sortino_ratio: Decimal = Field(description="Sortino ratio")
    calmar_ratio: Decimal = Field(description="Calmar ratio")
    omega_ratio: Decimal = Field(description="Omega ratio")

    # Trading specific metrics
    win_rate: Decimal = Field(description="Percentage of winning trades")
    profit_factor: Decimal = Field(description="Gross profit / Gross loss")
    average_win: Decimal = Field(description="Average winning trade")
    average_loss: Decimal = Field(description="Average losing trade")
    largest_win: Decimal = Field(description="Largest winning trade")
    largest_loss: Decimal = Field(description="Largest losing trade")

    # Advanced metrics
    value_at_risk_95: Decimal = Field(description="95% Value at Risk")
    conditional_var_95: Decimal = Field(description="95% Conditional VaR")
    skewness: Decimal = Field(description="Return distribution skewness")
    kurtosis: Decimal = Field(description="Return distribution kurtosis")

    # Recovery and stability
    recovery_factor: Decimal = Field(description="Recovery factor")
    stability_ratio: Decimal = Field(description="Stability of returns")
    consistency_score: Decimal = Field(description="Consistency of performance")

    # Transaction costs
    total_fees: Decimal = Field(description="Total transaction fees")
    fee_adjusted_return: Decimal = Field(description="Return after fees")
    turnover_ratio: Decimal = Field(description="Portfolio turnover ratio")

    # Time-based metrics
    periods_analyzed: int = Field(description="Number of periods analyzed")
    start_date: datetime = Field(description="Analysis start date")
    end_date: datetime = Field(description="Analysis end date")

    def get_risk_score(self) -> Decimal:
        """Calculate overall risk score (0-1, lower is better)."""
        risk_components = [
            min(self.volatility / Decimal("0.5"), Decimal("1")),  # Normalize volatility
            min(self.max_drawdown / Decimal("0.3"), Decimal("1")),  # Normalize drawdown
            max(Decimal("0"), Decimal("1") - self.sharpe_ratio / Decimal("2")),  # Inverse Sharpe
        ]
        return sum(risk_components) / len(risk_components)

    def get_quality_score(self) -> Decimal:
        """Calculate overall quality score (0-1, higher is better)."""
        quality_components = [
            (
                min(self.sharpe_ratio / Decimal("2"), Decimal("1"))
                if self.sharpe_ratio > 0
                else Decimal("0")
            ),
            (
                min(self.sortino_ratio / Decimal("3"), Decimal("1"))
                if self.sortino_ratio > 0
                else Decimal("0")
            ),
            self.win_rate,
            (
                min(self.profit_factor / Decimal("2"), Decimal("1"))
                if self.profit_factor > 0
                else Decimal("0")
            ),
            self.consistency_score,
        ]
        return sum(quality_components) / len(quality_components)


class SensitivityAnalysis(BaseModel):
    """
    Parameter sensitivity analysis results.

    Measures how sensitive the optimization results are
    to changes in individual parameters.
    """

    parameter_name: str = Field(description="Parameter name")
    sensitivity_score: Decimal = Field(
        description="Sensitivity score (0-1, higher = more sensitive)"
    )
    correlation_with_performance: Decimal = Field(
        description="Correlation between parameter and performance"
    )
    importance_rank: int = Field(description="Importance ranking among all parameters")

    # Statistical metrics
    p_value: Decimal | None = Field(
        default=None, description="Statistical significance of parameter effect"
    )
    confidence_interval: tuple[Decimal, Decimal] | None = Field(
        default=None, description="Confidence interval for parameter effect"
    )

    # Practical metrics
    optimal_range: tuple[Decimal, Decimal] = Field(description="Optimal parameter range")
    stability_across_periods: Decimal = Field(
        description="Parameter stability across different time periods"
    )

    # Interaction effects
    interaction_partners: list[str] = Field(
        default_factory=list, description="Parameters with significant interaction effects"
    )
    interaction_strength: dict[str, Decimal] = Field(
        default_factory=dict, description="Strength of interaction with other parameters"
    )


class StabilityAnalysis(BaseModel):
    """
    Stability analysis across different conditions.

    Analyzes how stable the optimization results are
    across different time periods and market conditions.
    """

    # Temporal stability
    period_consistency: Decimal = Field(description="Consistency across different time periods")
    regime_consistency: Decimal = Field(description="Consistency across different market regimes")

    # Performance stability
    parameter_stability: dict[str, Decimal] = Field(
        description="Stability of individual parameters"
    )
    performance_stability: Decimal = Field(description="Stability of performance metrics")

    # Market condition analysis
    bull_market_performance: Decimal | None = Field(
        default=None, description="Performance during bull markets"
    )
    bear_market_performance: Decimal | None = Field(
        default=None, description="Performance during bear markets"
    )
    sideways_market_performance: Decimal | None = Field(
        default=None, description="Performance during sideways markets"
    )

    # Volatility regime analysis
    low_vol_performance: Decimal | None = Field(
        default=None, description="Performance during low volatility periods"
    )
    high_vol_performance: Decimal | None = Field(
        default=None, description="Performance during high volatility periods"
    )

    # Stress testing
    worst_period_performance: Decimal = Field(
        description="Performance during worst historical period"
    )
    crisis_resilience: Decimal = Field(description="Resilience during market crises")

    def get_overall_stability_score(self) -> Decimal:
        """Calculate overall stability score."""
        components = [self.period_consistency, self.regime_consistency, self.performance_stability]

        # Add market condition components if available
        market_components = [
            self.bull_market_performance,
            self.bear_market_performance,
            self.sideways_market_performance,
        ]

        valid_market_components = [c for c in market_components if c is not None]
        if valid_market_components:
            # Check consistency across market conditions
            if len(valid_market_components) > 1:
                market_std = Decimal(
                    str(statistics.stdev([float(c) for c in valid_market_components]))
                )
                market_mean = sum(valid_market_components) / len(valid_market_components)
                if market_mean != 0:
                    market_consistency = Decimal("1") / (
                        Decimal("1") + abs(market_std / market_mean)
                    )
                else:
                    market_consistency = Decimal("0")
                components.append(market_consistency)

        return sum(components) / len(components)


class ParameterImportanceAnalyzer:
    """
    Analyzes parameter importance and interactions.

    Uses various statistical methods to determine which parameters
    have the most significant impact on optimization results.
    """

    def __init__(self):
        """Initialize parameter importance analyzer."""
        logger.info("ParameterImportanceAnalyzer initialized")

    def analyze_parameter_importance(
        self, optimization_results: list[dict[str, Any]], parameter_names: list[str]
    ) -> list[SensitivityAnalysis]:
        """
        Analyze importance of each parameter.

        Args:
            optimization_results: List of optimization result dictionaries
            parameter_names: List of parameter names to analyze

        Returns:
            List of sensitivity analysis results
        """
        if not optimization_results or not parameter_names:
            return []

        importance_results = []

        # Extract data
        parameter_data = self._extract_parameter_data(optimization_results, parameter_names)
        performance_data = self._extract_performance_data(optimization_results)

        if not parameter_data or not performance_data:
            return []

        # Analyze each parameter
        for param_name in parameter_names:
            if param_name not in parameter_data:
                continue

            analysis = self._analyze_single_parameter(
                param_name, parameter_data[param_name], performance_data, parameter_data
            )

            if analysis:
                importance_results.append(analysis)

        # Rank parameters by importance
        importance_results.sort(key=lambda x: x.sensitivity_score, reverse=True)
        for i, result in enumerate(importance_results):
            result.importance_rank = i + 1

        logger.info(f"Analyzed importance of {len(importance_results)} parameters")
        return importance_results

    def _extract_parameter_data(
        self, optimization_results: list[dict[str, Any]], parameter_names: list[str]
    ) -> dict[str, list[float]]:
        """Extract parameter data from optimization results."""
        parameter_data = {name: [] for name in parameter_names}

        for result in optimization_results:
            parameters = result.get("parameters", {})

            for param_name in parameter_names:
                if param_name in parameters:
                    value = parameters[param_name]

                    # Convert to float for analysis
                    if isinstance(value, int | float | Decimal):
                        parameter_data[param_name].append(float(value))
                    elif isinstance(value, bool):
                        parameter_data[param_name].append(1.0 if value else 0.0)
                    elif isinstance(value, str):
                        # For categorical parameters, we'll handle separately
                        parameter_data[param_name].append(hash(value) % 1000 / 1000.0)
                    else:
                        parameter_data[param_name].append(0.0)

        return parameter_data

    def _extract_performance_data(self, optimization_results: list[dict[str, Any]]) -> list[float]:
        """Extract performance data from optimization results."""
        performance_data = []

        for result in optimization_results:
            # Try different keys for performance
            performance = None

            for key in ["objective_value", "performance", "total_return", "sharpe_ratio"]:
                if key in result:
                    performance = result[key]
                    break

            if performance is not None:
                if isinstance(performance, int | float | Decimal):
                    performance_data.append(float(performance))
                else:
                    performance_data.append(0.0)
            else:
                performance_data.append(0.0)

        return performance_data

    def _analyze_single_parameter(
        self,
        param_name: str,
        param_values: list[float],
        performance_values: list[float],
        all_parameter_data: dict[str, list[float]],
    ) -> SensitivityAnalysis | None:
        """Analyze a single parameter's importance."""
        if len(param_values) != len(performance_values) or len(param_values) < 3:
            return None

        try:
            # Calculate correlation
            correlation, p_value = pearsonr(param_values, performance_values)

            # Calculate sensitivity score
            sensitivity_score = abs(correlation)

            # Calculate optimal range (based on top quartile performance)
            combined_data = list(zip(param_values, performance_values, strict=False))
            combined_data.sort(key=lambda x: x[1], reverse=True)

            top_quartile_size = max(1, len(combined_data) // 4)
            top_quartile_params = [x[0] for x in combined_data[:top_quartile_size]]

            if top_quartile_params:
                optimal_min = Decimal(str(min(top_quartile_params)))
                optimal_max = Decimal(str(max(top_quartile_params)))
            else:
                optimal_min = Decimal(str(min(param_values)))
                optimal_max = Decimal(str(max(param_values)))

            # Calculate stability (consistency across different data subsets)
            stability = self._calculate_parameter_stability(param_values, performance_values)

            # Find interaction partners
            interaction_partners, interaction_strengths = self._find_interaction_partners(
                param_name, all_parameter_data, performance_values
            )

            return SensitivityAnalysis(
                parameter_name=param_name,
                sensitivity_score=Decimal(str(sensitivity_score)),
                correlation_with_performance=Decimal(str(correlation)),
                importance_rank=0,  # Will be set later
                p_value=Decimal(str(p_value)),
                optimal_range=(optimal_min, optimal_max),
                stability_across_periods=Decimal(str(stability)),
                interaction_partners=interaction_partners,
                interaction_strength=interaction_strengths,
            )

        except Exception as e:
            logger.warning(f"Failed to analyze parameter {param_name}: {e!s}")
            return None

    def _calculate_parameter_stability(
        self, param_values: list[float], performance_values: list[float]
    ) -> float:
        """Calculate stability of parameter across different subsets."""
        if len(param_values) < 10:
            return 0.0

        # Split data into thirds and calculate correlations
        n = len(param_values)
        third = n // 3

        correlations = []

        for i in range(3):
            start_idx = i * third
            end_idx = (i + 1) * third if i < 2 else n

            subset_params = param_values[start_idx:end_idx]
            subset_performance = performance_values[start_idx:end_idx]

            if len(subset_params) > 2:
                try:
                    corr, _ = pearsonr(subset_params, subset_performance)
                    if not math.isnan(corr):
                        correlations.append(corr)
                except:
                    pass

        if len(correlations) < 2:
            return 0.0

        # Stability is inverse of standard deviation
        std_corr = statistics.stdev(correlations)
        stability = 1.0 / (1.0 + std_corr)

        return stability

    def _find_interaction_partners(
        self,
        param_name: str,
        all_parameter_data: dict[str, list[float]],
        performance_values: list[float],
    ) -> tuple[list[str], dict[str, Decimal]]:
        """Find parameters that interact significantly with the given parameter."""
        interaction_partners = []
        interaction_strengths = {}

        target_param_values = all_parameter_data.get(param_name, [])

        if not target_param_values:
            return interaction_partners, interaction_strengths

        for other_param, other_values in all_parameter_data.items():
            if other_param == param_name or len(other_values) != len(target_param_values):
                continue

            try:
                # Calculate interaction effect using correlation of products
                interaction_values = [
                    a * b for a, b in zip(target_param_values, other_values, strict=False)
                ]

                # Correlation between interaction term and performance
                interaction_corr, interaction_p = pearsonr(interaction_values, performance_values)

                # Check if interaction is significant
                if abs(interaction_corr) > 0.3 and interaction_p < 0.05:
                    interaction_partners.append(other_param)
                    interaction_strengths[other_param] = Decimal(str(abs(interaction_corr)))

            except Exception as e:
                logger.debug(
                    f"Failed to calculate interaction between {param_name} and {other_param}: {e!s}"
                )

        return interaction_partners, interaction_strengths


class PerformanceAnalyzer:
    """
    Analyzes trading strategy performance metrics.

    Calculates comprehensive performance and risk metrics
    for trading strategies with proper risk adjustment.
    """

    def __init__(self, risk_free_rate: Decimal = Decimal("0.02")):
        """
        Initialize performance analyzer.

        Args:
            risk_free_rate: Annual risk-free rate for calculations
        """
        self.risk_free_rate = risk_free_rate

        logger.info("PerformanceAnalyzer initialized", risk_free_rate=float(risk_free_rate))

    def calculate_performance_metrics(
        self,
        returns: list[Decimal],
        trades: list[dict[str, Any]],
        start_date: datetime,
        end_date: datetime,
        initial_capital: Decimal = Decimal("100000"),
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.

        Args:
            returns: List of period returns
            trades: List of individual trades
            start_date: Analysis start date
            end_date: Analysis end date
            initial_capital: Initial capital amount

        Returns:
            Comprehensive performance metrics
        """
        if not returns:
            return self._create_empty_metrics(start_date, end_date)

        # Convert to float for calculations
        float_returns = [float(r) for r in returns]

        # Basic return calculations
        total_return = self._calculate_total_return(float_returns)
        annualized_return = self._calculate_annualized_return(total_return, start_date, end_date)

        # Risk calculations
        volatility = self._calculate_volatility(float_returns)
        downside_volatility = self._calculate_downside_volatility(float_returns)
        max_drawdown, current_drawdown = self._calculate_drawdowns(float_returns)

        # Risk-adjusted metrics
        sharpe_ratio = self._calculate_sharpe_ratio(annualized_return, volatility)
        sortino_ratio = self._calculate_sortino_ratio(annualized_return, downside_volatility)
        calmar_ratio = self._calculate_calmar_ratio(annualized_return, max_drawdown)
        omega_ratio = self._calculate_omega_ratio(float_returns)

        # Trading metrics
        trade_metrics = self._calculate_trade_metrics(trades)

        # Advanced risk metrics
        var_95 = self._calculate_var(float_returns, 0.95)
        cvar_95 = self._calculate_conditional_var(float_returns, 0.95)
        skewness = self._calculate_skewness(float_returns)
        kurtosis = self._calculate_kurtosis(float_returns)

        # Recovery and stability
        recovery_factor = self._calculate_recovery_factor(total_return, max_drawdown)
        stability_ratio = self._calculate_stability_ratio(float_returns)
        consistency_score = self._calculate_consistency_score(float_returns)

        # Fee calculations
        total_fees = sum(Decimal(str(trade.get("fee", 0))) for trade in trades)
        fee_adjusted_return = total_return - (total_fees / initial_capital)
        turnover_ratio = self._calculate_turnover_ratio(trades, initial_capital)

        return PerformanceMetrics(
            total_return=Decimal(str(total_return)),
            annualized_return=Decimal(str(annualized_return)),
            excess_return=Decimal(str(annualized_return - float(self.risk_free_rate))),
            volatility=Decimal(str(volatility)),
            downside_volatility=Decimal(str(downside_volatility)),
            max_drawdown=Decimal(str(max_drawdown)),
            current_drawdown=Decimal(str(current_drawdown)),
            sharpe_ratio=Decimal(str(sharpe_ratio)),
            sortino_ratio=Decimal(str(sortino_ratio)),
            calmar_ratio=Decimal(str(calmar_ratio)),
            omega_ratio=Decimal(str(omega_ratio)),
            win_rate=trade_metrics["win_rate"],
            profit_factor=trade_metrics["profit_factor"],
            average_win=trade_metrics["average_win"],
            average_loss=trade_metrics["average_loss"],
            largest_win=trade_metrics["largest_win"],
            largest_loss=trade_metrics["largest_loss"],
            value_at_risk_95=Decimal(str(var_95)),
            conditional_var_95=Decimal(str(cvar_95)),
            skewness=Decimal(str(skewness)),
            kurtosis=Decimal(str(kurtosis)),
            recovery_factor=Decimal(str(recovery_factor)),
            stability_ratio=Decimal(str(stability_ratio)),
            consistency_score=Decimal(str(consistency_score)),
            total_fees=total_fees,
            fee_adjusted_return=Decimal(str(fee_adjusted_return)),
            turnover_ratio=turnover_ratio,
            periods_analyzed=len(returns),
            start_date=start_date,
            end_date=end_date,
        )

    def _create_empty_metrics(self, start_date: datetime, end_date: datetime) -> PerformanceMetrics:
        """Create empty performance metrics."""
        return PerformanceMetrics(
            total_return=Decimal("0"),
            annualized_return=Decimal("0"),
            excess_return=Decimal("0"),
            volatility=Decimal("0"),
            downside_volatility=Decimal("0"),
            max_drawdown=Decimal("0"),
            current_drawdown=Decimal("0"),
            sharpe_ratio=Decimal("0"),
            sortino_ratio=Decimal("0"),
            calmar_ratio=Decimal("0"),
            omega_ratio=Decimal("0"),
            win_rate=Decimal("0"),
            profit_factor=Decimal("0"),
            average_win=Decimal("0"),
            average_loss=Decimal("0"),
            largest_win=Decimal("0"),
            largest_loss=Decimal("0"),
            value_at_risk_95=Decimal("0"),
            conditional_var_95=Decimal("0"),
            skewness=Decimal("0"),
            kurtosis=Decimal("0"),
            recovery_factor=Decimal("0"),
            stability_ratio=Decimal("0"),
            consistency_score=Decimal("0"),
            total_fees=Decimal("0"),
            fee_adjusted_return=Decimal("0"),
            turnover_ratio=Decimal("0"),
            periods_analyzed=0,
            start_date=start_date,
            end_date=end_date,
        )

    def _calculate_total_return(self, returns: list[float]) -> float:
        """Calculate total return from period returns."""
        if not returns:
            return 0.0

        # Compound returns
        total = 1.0
        for r in returns:
            total *= 1.0 + r

        return total - 1.0

    def _calculate_annualized_return(
        self, total_return: float, start_date: datetime, end_date: datetime
    ) -> float:
        """Calculate annualized return."""
        if total_return <= -1.0:
            return -1.0

        days = (end_date - start_date).days
        if days <= 0:
            return 0.0

        years = days / 365.25
        if years <= 0:
            return 0.0

        return (1.0 + total_return) ** (1.0 / years) - 1.0

    def _calculate_volatility(self, returns: list[float]) -> float:
        """Calculate annualized volatility."""
        if len(returns) < 2:
            return 0.0

        # Calculate standard deviation
        std_dev = statistics.stdev(returns)

        # Annualize (assuming daily returns)
        return std_dev * math.sqrt(252)

    def _calculate_downside_volatility(self, returns: list[float]) -> float:
        """Calculate downside volatility."""
        if len(returns) < 2:
            return 0.0

        # Only consider negative returns
        downside_returns = [r for r in returns if r < 0]

        if len(downside_returns) < 2:
            return 0.0

        # Calculate standard deviation of downside returns
        std_dev = statistics.stdev(downside_returns)

        # Annualize
        return std_dev * math.sqrt(252)

    def _calculate_drawdowns(self, returns: list[float]) -> tuple[float, float]:
        """Calculate maximum and current drawdown."""
        if not returns:
            return 0.0, 0.0

        # Calculate cumulative returns
        cumulative = [1.0]
        for r in returns:
            cumulative.append(cumulative[-1] * (1.0 + r))

        # Calculate drawdowns
        running_max = cumulative[0]
        max_drawdown = 0.0
        drawdowns = []

        for value in cumulative:
            if value > running_max:
                running_max = value

            drawdown = (running_max - value) / running_max
            drawdowns.append(drawdown)
            max_drawdown = max(max_drawdown, drawdown)

        current_drawdown = drawdowns[-1] if drawdowns else 0.0

        return max_drawdown, current_drawdown

    def _calculate_sharpe_ratio(self, annualized_return: float, volatility: float) -> float:
        """Calculate Sharpe ratio."""
        if volatility == 0:
            return 0.0

        excess_return = annualized_return - float(self.risk_free_rate)
        return excess_return / volatility

    def _calculate_sortino_ratio(
        self, annualized_return: float, downside_volatility: float
    ) -> float:
        """Calculate Sortino ratio."""
        if downside_volatility == 0:
            return 0.0

        excess_return = annualized_return - float(self.risk_free_rate)
        return excess_return / downside_volatility

    def _calculate_calmar_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        if max_drawdown == 0:
            return 0.0

        return annualized_return / max_drawdown

    def _calculate_omega_ratio(self, returns: list[float]) -> float:
        """Calculate Omega ratio."""
        if not returns:
            return 0.0

        threshold = 0.0  # Use zero as threshold

        gains = sum(max(0, r - threshold) for r in returns)
        losses = sum(max(0, threshold - r) for r in returns)

        if losses == 0:
            return float("inf") if gains > 0 else 0.0

        return gains / losses

    def _calculate_trade_metrics(self, trades: list[dict[str, Any]]) -> dict[str, Decimal]:
        """Calculate trade-specific metrics."""
        if not trades:
            return {
                "win_rate": Decimal("0"),
                "profit_factor": Decimal("0"),
                "average_win": Decimal("0"),
                "average_loss": Decimal("0"),
                "largest_win": Decimal("0"),
                "largest_loss": Decimal("0"),
            }

        # Extract PnL from trades
        pnls = []
        for trade in trades:
            pnl = trade.get("pnl", trade.get("profit_loss", 0))
            pnls.append(Decimal(str(pnl)))

        # Separate wins and losses
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]

        # Calculate metrics
        win_rate = Decimal(str(len(wins) / len(pnls))) if pnls else Decimal("0")

        average_win = sum(wins) / len(wins) if wins else Decimal("0")
        average_loss = sum(losses) / len(losses) if losses else Decimal("0")

        largest_win = max(wins) if wins else Decimal("0")
        largest_loss = min(losses) if losses else Decimal("0")

        # Profit factor
        gross_profit = sum(wins) if wins else Decimal("0")
        gross_loss = abs(sum(losses)) if losses else Decimal("0")

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else Decimal("0")

        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "average_win": average_win,
            "average_loss": average_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
        }

    def _calculate_var(self, returns: list[float], confidence: float) -> float:
        """Calculate Value at Risk."""
        if not returns:
            return 0.0

        # Sort returns
        sorted_returns = sorted(returns)

        # Find VaR at confidence level
        index = int((1 - confidence) * len(sorted_returns))
        index = max(0, min(index, len(sorted_returns) - 1))

        return abs(sorted_returns[index])

    def _calculate_conditional_var(self, returns: list[float], confidence: float) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        if not returns:
            return 0.0

        # Sort returns
        sorted_returns = sorted(returns)

        # Find tail returns
        index = int((1 - confidence) * len(sorted_returns))
        tail_returns = sorted_returns[:index] if index > 0 else [sorted_returns[0]]

        # Average of tail returns
        return abs(sum(tail_returns) / len(tail_returns)) if tail_returns else 0.0

    def _calculate_skewness(self, returns: list[float]) -> float:
        """Calculate return distribution skewness."""
        if len(returns) < 3:
            return 0.0

        try:
            return stats.skew(returns)
        except:
            return 0.0

    def _calculate_kurtosis(self, returns: list[float]) -> float:
        """Calculate return distribution kurtosis."""
        if len(returns) < 4:
            return 0.0

        try:
            return stats.kurtosis(returns)
        except:
            return 0.0

    def _calculate_recovery_factor(self, total_return: float, max_drawdown: float) -> float:
        """Calculate recovery factor."""
        if max_drawdown == 0:
            return 0.0

        return total_return / max_drawdown

    def _calculate_stability_ratio(self, returns: list[float]) -> float:
        """Calculate stability ratio."""
        if len(returns) < 12:  # Need sufficient data
            return 0.0

        # Split returns into periods and calculate correlation
        n = len(returns)
        mid = n // 2

        first_half = returns[:mid]
        second_half = returns[mid:]

        try:
            correlation, _ = pearsonr(first_half, second_half[: len(first_half)])
            return max(0.0, correlation)
        except:
            return 0.0

    def _calculate_consistency_score(self, returns: list[float]) -> float:
        """Calculate consistency score."""
        if len(returns) < 12:
            return 0.0

        # Calculate rolling 12-period returns
        rolling_returns = []
        for i in range(12, len(returns) + 1):
            period_return = self._calculate_total_return(returns[i - 12 : i])
            rolling_returns.append(period_return)

        if not rolling_returns:
            return 0.0

        # Consistency is based on positive rolling returns
        positive_periods = sum(1 for r in rolling_returns if r > 0)
        return positive_periods / len(rolling_returns)

    def _calculate_turnover_ratio(
        self, trades: list[dict[str, Any]], initial_capital: Decimal
    ) -> Decimal:
        """Calculate portfolio turnover ratio."""
        if not trades or initial_capital <= 0:
            return Decimal("0")

        # Calculate total trading volume
        total_volume = Decimal("0")

        for trade in trades:
            quantity = Decimal(str(trade.get("quantity", 0)))
            price = Decimal(str(trade.get("price", 0)))
            volume = abs(quantity * price)
            total_volume += volume

        # Turnover ratio = total volume / initial capital
        return total_volume / initial_capital


class ResultsAnalyzer:
    """
    Main results analyzer that orchestrates all analysis components.

    Provides comprehensive analysis of optimization results including
    performance metrics, parameter importance, and stability analysis.
    """

    def __init__(self, risk_free_rate: Decimal = Decimal("0.02")):
        """
        Initialize results analyzer.

        Args:
            risk_free_rate: Annual risk-free rate
        """
        self.performance_analyzer = PerformanceAnalyzer(risk_free_rate)
        self.importance_analyzer = ParameterImportanceAnalyzer()

        logger.info("ResultsAnalyzer initialized")

    @time_execution
    def analyze_optimization_results(
        self,
        optimization_results: list[dict[str, Any]],
        parameter_names: list[str],
        best_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Perform comprehensive analysis of optimization results.

        Args:
            optimization_results: List of optimization result dictionaries
            parameter_names: List of parameter names
            best_result: Best optimization result (optional)

        Returns:
            Comprehensive analysis results
        """
        analysis_results = {}

        try:
            # 1. Parameter importance analysis
            logger.info("Analyzing parameter importance")
            importance_results = self.importance_analyzer.analyze_parameter_importance(
                optimization_results, parameter_names
            )
            analysis_results["parameter_importance"] = importance_results

            # 2. Performance distribution analysis
            logger.info("Analyzing performance distribution")
            performance_stats = self._analyze_performance_distribution(optimization_results)
            analysis_results["performance_distribution"] = performance_stats

            # 3. Parameter correlation analysis
            logger.info("Analyzing parameter correlations")
            correlation_matrix = self._calculate_parameter_correlations(
                optimization_results, parameter_names
            )
            analysis_results["parameter_correlations"] = correlation_matrix

            # 4. Optimization landscape analysis
            logger.info("Analyzing optimization landscape")
            landscape_analysis = self._analyze_optimization_landscape(optimization_results)
            analysis_results["landscape_analysis"] = landscape_analysis

            # 5. Best result detailed analysis
            if best_result:
                logger.info("Analyzing best result")
                best_analysis = self._analyze_best_result(best_result)
                analysis_results["best_result_analysis"] = best_analysis

            # 6. Convergence analysis
            logger.info("Analyzing optimization convergence")
            convergence_analysis = self._analyze_convergence(optimization_results)
            analysis_results["convergence_analysis"] = convergence_analysis

            # 7. Summary statistics
            analysis_results["summary"] = self._create_analysis_summary(
                analysis_results, len(optimization_results)
            )

            logger.info("Comprehensive optimization analysis completed")

        except Exception as e:
            logger.error(f"Optimization analysis failed: {e!s}")
            analysis_results["error"] = str(e)

        return analysis_results

    def _analyze_performance_distribution(
        self, optimization_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze distribution of performance across all results."""
        performance_values = []

        for result in optimization_results:
            performance = None
            for key in ["objective_value", "performance", "total_return"]:
                if key in result:
                    performance = result[key]
                    break

            if performance is not None:
                performance_values.append(float(performance))

        if not performance_values:
            return {}

        return {
            "count": len(performance_values),
            "mean": statistics.mean(performance_values),
            "median": statistics.median(performance_values),
            "std": statistics.stdev(performance_values) if len(performance_values) > 1 else 0,
            "min": min(performance_values),
            "max": max(performance_values),
            "q25": (
                statistics.quantiles(performance_values, n=4)[0]
                if len(performance_values) >= 4
                else min(performance_values)
            ),
            "q75": (
                statistics.quantiles(performance_values, n=4)[2]
                if len(performance_values) >= 4
                else max(performance_values)
            ),
            "skewness": stats.skew(performance_values) if len(performance_values) > 2 else 0,
            "kurtosis": stats.kurtosis(performance_values) if len(performance_values) > 3 else 0,
        }

    def _calculate_parameter_correlations(
        self, optimization_results: list[dict[str, Any]], parameter_names: list[str]
    ) -> dict[str, dict[str, float]]:
        """Calculate correlation matrix between parameters."""
        parameter_data = {}

        # Extract parameter data
        for param_name in parameter_names:
            parameter_data[param_name] = []

            for result in optimization_results:
                parameters = result.get("parameters", {})
                value = parameters.get(param_name, 0)

                if isinstance(value, int | float | Decimal):
                    parameter_data[param_name].append(float(value))
                elif isinstance(value, bool):
                    parameter_data[param_name].append(1.0 if value else 0.0)
                else:
                    parameter_data[param_name].append(0.0)

        # Calculate correlation matrix
        correlation_matrix = {}

        for param1 in parameter_names:
            correlation_matrix[param1] = {}

            for param2 in parameter_names:
                if (
                    param1 in parameter_data
                    and param2 in parameter_data
                    and len(parameter_data[param1]) == len(parameter_data[param2])
                    and len(parameter_data[param1]) > 1
                ):
                    try:
                        corr, _ = pearsonr(parameter_data[param1], parameter_data[param2])
                        correlation_matrix[param1][param2] = corr if not math.isnan(corr) else 0.0
                    except:
                        correlation_matrix[param1][param2] = 0.0
                else:
                    correlation_matrix[param1][param2] = 0.0

        return correlation_matrix

    def _analyze_optimization_landscape(
        self, optimization_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze the optimization landscape."""
        if not optimization_results:
            return {}

        # Extract performance values
        performance_values = []
        for result in optimization_results:
            performance = None
            for key in ["objective_value", "performance", "total_return"]:
                if key in result:
                    performance = result[key]
                    break

            if performance is not None:
                performance_values.append(float(performance))

        if not performance_values:
            return {}

        # Sort results by performance
        sorted_performance = sorted(performance_values, reverse=True)

        # Analyze landscape characteristics
        return {
            "ruggedness": self._calculate_landscape_ruggedness(performance_values),
            "multimodality": self._detect_multimodality(sorted_performance),
            "convergence_rate": self._calculate_convergence_rate(sorted_performance),
            "plateau_detection": self._detect_performance_plateaus(sorted_performance),
            "improvement_potential": self._assess_improvement_potential(sorted_performance),
        }

    def _calculate_landscape_ruggedness(self, performance_values: list[float]) -> float:
        """Calculate how rugged (noisy) the optimization landscape is."""
        if len(performance_values) < 3:
            return 0.0

        # Calculate autocorrelation
        try:
            mean_perf = statistics.mean(performance_values)
            normalized = [p - mean_perf for p in performance_values]

            # Calculate lag-1 autocorrelation
            autocorr = sum(normalized[i] * normalized[i + 1] for i in range(len(normalized) - 1))
            autocorr /= sum(x**2 for x in normalized[:-1])

            # Ruggedness is inverse of autocorrelation
            return max(0.0, 1.0 - autocorr)
        except:
            return 0.0

    def _detect_multimodality(self, sorted_performance: list[float]) -> dict[str, Any]:
        """Detect if the optimization landscape has multiple modes."""
        if len(sorted_performance) < 10:
            return {"has_multiple_modes": False, "mode_count": 1}

        # Simple mode detection using performance gaps
        gaps = []
        for i in range(1, len(sorted_performance)):
            gap = sorted_performance[i - 1] - sorted_performance[i]
            gaps.append(gap)

        # Find significant gaps (larger than 2 standard deviations)
        if len(gaps) > 1:
            gap_std = statistics.stdev(gaps)
            gap_mean = statistics.mean(gaps)
            threshold = gap_mean + 2 * gap_std

            significant_gaps = sum(1 for gap in gaps if gap > threshold)
            mode_count = significant_gaps + 1

            return {
                "has_multiple_modes": mode_count > 1,
                "mode_count": mode_count,
                "gap_threshold": threshold,
            }

        return {"has_multiple_modes": False, "mode_count": 1}

    def _calculate_convergence_rate(self, sorted_performance: list[float]) -> float:
        """Calculate how quickly the optimization converged."""
        if len(sorted_performance) < 10:
            return 0.0

        # Take top 10% of results
        top_10_pct = int(len(sorted_performance) * 0.1)
        top_10_pct = max(1, top_10_pct)

        top_performance = sorted_performance[:top_10_pct]
        median_performance = statistics.median(sorted_performance)

        if len(top_performance) > 0 and median_performance != 0:
            performance_ratio = statistics.mean(top_performance) / median_performance
            # Normalize to 0-1 scale
            return min(1.0, max(0.0, (performance_ratio - 1.0) / 2.0))

        return 0.0

    def _detect_performance_plateaus(self, sorted_performance: list[float]) -> dict[str, Any]:
        """Detect performance plateaus in the results."""
        if len(sorted_performance) < 5:
            return {"has_plateaus": False, "plateau_count": 0}

        # Find regions with minimal performance variation
        plateaus = []
        current_plateau = [sorted_performance[0]]

        tolerance = (
            statistics.stdev(sorted_performance) * 0.1 if len(sorted_performance) > 1 else 0.01
        )

        for i in range(1, len(sorted_performance)):
            if abs(sorted_performance[i] - current_plateau[-1]) <= tolerance:
                current_plateau.append(sorted_performance[i])
            else:
                if len(current_plateau) >= 3:  # Minimum plateau size
                    plateaus.append(current_plateau)
                current_plateau = [sorted_performance[i]]

        # Check last plateau
        if len(current_plateau) >= 3:
            plateaus.append(current_plateau)

        return {
            "has_plateaus": len(plateaus) > 0,
            "plateau_count": len(plateaus),
            "largest_plateau_size": max(len(p) for p in plateaus) if plateaus else 0,
        }

    def _assess_improvement_potential(self, sorted_performance: list[float]) -> dict[str, Any]:
        """Assess potential for further improvement."""
        if len(sorted_performance) < 5:
            return {"improvement_potential": "unknown"}

        # Analyze trend in top results
        top_20_pct = int(len(sorted_performance) * 0.2)
        top_20_pct = max(5, top_20_pct)

        top_results = sorted_performance[:top_20_pct]

        # Calculate trend
        x = list(range(len(top_results)))
        try:
            slope, _, r_value, p_value, _ = stats.linregress(x, top_results)

            if p_value < 0.05:  # Significant trend
                if slope < -0.01:  # Declining trend
                    potential = "high"
                elif slope > 0.01:  # Improving trend (should not happen with sorted data)
                    potential = "low"
                else:
                    potential = "medium"
            else:
                potential = "medium"

            return {
                "improvement_potential": potential,
                "trend_slope": slope,
                "trend_significance": p_value,
            }

        except:
            return {"improvement_potential": "medium"}

    def _analyze_best_result(self, best_result: dict[str, Any]) -> dict[str, Any]:
        """Analyze the best optimization result in detail."""
        analysis = {
            "parameters": best_result.get("parameters", {}),
            "performance": best_result.get("objective_value", 0),
            "metadata": best_result.get("metadata", {}),
        }

        # Add parameter analysis
        parameters = best_result.get("parameters", {})
        if parameters:
            analysis["parameter_summary"] = {
                "parameter_count": len(parameters),
                "parameter_types": self._categorize_parameter_types(parameters),
            }

        return analysis

    def _categorize_parameter_types(self, parameters: dict[str, Any]) -> dict[str, int]:
        """Categorize parameter types."""
        type_counts = defaultdict(int)

        for value in parameters.values():
            if isinstance(value, bool):
                type_counts["boolean"] += 1
            elif isinstance(value, int):
                type_counts["integer"] += 1
            elif isinstance(value, float | Decimal):
                type_counts["continuous"] += 1
            elif isinstance(value, str):
                type_counts["categorical"] += 1
            else:
                type_counts["other"] += 1

        return dict(type_counts)

    def _analyze_convergence(self, optimization_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze optimization convergence characteristics."""
        if not optimization_results:
            return {}

        # Extract performance values in order
        performance_sequence = []

        for result in optimization_results:
            performance = None
            for key in ["objective_value", "performance", "total_return"]:
                if key in result:
                    performance = result[key]
                    break

            if performance is not None:
                performance_sequence.append(float(performance))

        if len(performance_sequence) < 2:
            return {}

        # Calculate running maximum
        running_max = []
        current_max = performance_sequence[0]

        for perf in performance_sequence:
            current_max = max(current_max, perf)
            running_max.append(current_max)

        # Analyze convergence
        final_max = running_max[-1]

        # Find when 90% of final performance was reached
        target_performance = final_max * 0.9
        convergence_point = len(running_max)

        for i, perf in enumerate(running_max):
            if perf >= target_performance:
                convergence_point = i + 1
                break

        return {
            "total_evaluations": len(performance_sequence),
            "final_performance": final_max,
            "convergence_point_90pct": convergence_point,
            "convergence_efficiency": convergence_point / len(performance_sequence),
            "improvement_curve": (
                running_max[-20:] if len(running_max) >= 20 else running_max
            ),  # Last 20 points
        }

    def _create_analysis_summary(
        self, analysis_results: dict[str, Any], total_evaluations: int
    ) -> dict[str, Any]:
        """Create summary of analysis results."""
        summary = {
            "total_evaluations": total_evaluations,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Add key insights
        if "parameter_importance" in analysis_results:
            importance_results = analysis_results["parameter_importance"]
            if importance_results:
                most_important = max(importance_results, key=lambda x: x.sensitivity_score)
                summary["most_important_parameter"] = {
                    "name": most_important.parameter_name,
                    "sensitivity_score": float(most_important.sensitivity_score),
                }

        if "performance_distribution" in analysis_results:
            perf_dist = analysis_results["performance_distribution"]
            summary["performance_summary"] = {
                "best_performance": perf_dist.get("max", 0),
                "median_performance": perf_dist.get("median", 0),
                "performance_std": perf_dist.get("std", 0),
            }

        if "landscape_analysis" in analysis_results:
            landscape = analysis_results["landscape_analysis"]
            summary["landscape_characteristics"] = {
                "ruggedness": landscape.get("ruggedness", 0),
                "has_multiple_modes": landscape.get("multimodality", {}).get(
                    "has_multiple_modes", False
                ),
                "improvement_potential": landscape.get("improvement_potential", {}).get(
                    "improvement_potential", "unknown"
                ),
            }

        return summary
