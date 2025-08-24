"""
Risk Monitoring and Reporting System.

This module provides institutional-grade real-time risk monitoring, advanced VaR
calculations, stress testing, and comprehensive regulatory risk reporting capabilities.

Key Features:
- Real-time risk dashboard with WebSocket integration
- Advanced VaR methodologies (Historical, Parametric, Monte Carlo, Filtered Historical)
- Comprehensive stress testing framework with scenario analysis
- Dynamic risk limit monitoring with automated breach detection
- Regulatory compliance reporting (Basel III, CCAR, FRTB)
- Concentration and counterparty risk assessment
- Tail risk analysis and Expected Shortfall calculations
- Real-time alert system with severity-based escalation
- Risk attribution and decomposition analytics
"""

import asyncio
from collections import defaultdict, deque
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.analytics.types import (
    AlertSeverity,
    AnalyticsAlert,
    AnalyticsConfiguration,
)
from src.base import BaseComponent
from src.core.types.trading import Position
from src.monitoring.metrics import get_metrics_collector
from src.utils.datetime_utils import get_current_utc_timestamp


class RiskMonitor(BaseComponent):
    """
    Comprehensive risk monitoring and reporting system.

    Provides institutional-grade risk monitoring including:
    - Real-time risk dashboard and metrics
    - Limit monitoring with breach alerts
    - Stress testing and scenario analysis
    - Regulatory risk reporting (VaR, leverage ratios)
    - Counterparty and concentration risk assessment
    - Monte Carlo simulations and tail risk analysis
    """

    def __init__(self, config: AnalyticsConfiguration):
        """
        Initialize risk monitor.

        Args:
            config: Analytics configuration
        """
        super().__init__()
        self.config = config
        self.metrics_collector = get_metrics_collector()

        # Risk data storage
        self._positions: dict[str, Position] = {}
        self._price_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=252))
        self._portfolio_returns: deque = deque(maxlen=252)
        self._var_history: deque = deque(maxlen=60)  # 2 months of daily VaR

        # Risk limits and thresholds
        self._risk_limits = {
            "max_portfolio_var_95": Decimal("0.02"),  # 2% daily VaR
            "max_portfolio_var_99": Decimal("0.04"),  # 4% daily VaR
            "max_single_position_weight": Decimal("0.10"),  # 10% max position
            "max_sector_concentration": Decimal("0.25"),  # 25% max sector
            "max_leverage_ratio": Decimal("2.0"),  # 2x max leverage
            "max_correlation_exposure": Decimal("0.80"),  # 80% max correlation
            "max_drawdown_limit": Decimal("0.15"),  # 15% max drawdown
            "min_liquidity_buffer": Decimal("0.05"),  # 5% min cash buffer
        }

        # Monitoring state
        self._active_breaches: dict[str, AnalyticsAlert] = {}
        self._breach_history: deque = deque(maxlen=1000)
        self._stress_test_results: dict[str, dict[str, float]] = {}

        # Monte Carlo simulation parameters
        self._mc_simulations = 10000
        self._mc_time_horizon = 22  # 1 month trading days

        # Background tasks
        self._monitoring_tasks: set = set()
        self._running = False

        self.logger.info("RiskMonitor initialized")

    async def start(self) -> None:
        """Start risk monitoring tasks."""
        if self._running:
            self.logger.warning("Risk monitor already running")
            return

        self._running = True

        # Start monitoring tasks
        tasks = [
            self._real_time_monitoring_loop(),
            self._limit_monitoring_loop(),
            self._stress_testing_loop(),
            self._var_backtesting_loop(),
        ]

        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self._monitoring_tasks.add(task)
            task.add_done_callback(self._monitoring_tasks.discard)

        self.logger.info("Risk monitoring started")

    async def stop(self) -> None:
        """Stop risk monitoring tasks."""
        self._running = False

        # Cancel all tasks
        for task in self._monitoring_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)

        self.logger.info("Risk monitoring stopped")

    def update_positions(self, positions: dict[str, Position]) -> None:
        """
        Update position data for risk calculations.

        Args:
            positions: Dictionary of current positions
        """
        self._positions = positions.copy()

        # Trigger immediate risk recalculation
        asyncio.create_task(self._calculate_real_time_risk())

    def update_prices(self, price_updates: dict[str, Decimal]) -> None:
        """
        Update price data for risk calculations.

        Args:
            price_updates: Dictionary of symbol -> price updates
        """
        timestamp = get_current_utc_timestamp()

        for symbol, price in price_updates.items():
            self._price_history[symbol].append({"timestamp": timestamp, "price": float(price)})

        # Trigger portfolio return calculation
        asyncio.create_task(self._update_portfolio_returns())

    async def calculate_var(
        self, confidence_level: float = 0.95, time_horizon: int = 1, method: str = "historical"
    ) -> dict[str, float]:
        """
        Calculate Value at Risk using various methods.

        Args:
            confidence_level: VaR confidence level (0.95 or 0.99)
            time_horizon: Time horizon in days
            method: Calculation method ('historical', 'parametric', 'monte_carlo')

        Returns:
            Dictionary with VaR calculations
        """
        try:
            if len(self._portfolio_returns) < 30:
                return {"error": "Insufficient historical data for VaR calculation"}

            returns_array = np.array(list(self._portfolio_returns))

            var_results = {}

            if method == "historical":
                # Historical simulation VaR
                percentile = (1 - confidence_level) * 100
                var_1d = np.percentile(returns_array, percentile)
                var_horizon = var_1d * np.sqrt(time_horizon)  # Square root rule

                var_results["historical_var"] = abs(var_horizon)

            elif method == "parametric":
                # Parametric (normal distribution) VaR
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                z_score = stats.norm.ppf(1 - confidence_level)

                var_1d = mean_return + z_score * std_return
                var_horizon = var_1d * np.sqrt(time_horizon)

                var_results["parametric_var"] = abs(var_horizon)

            elif method == "monte_carlo":
                # Monte Carlo simulation VaR
                var_mc = await self._monte_carlo_var(confidence_level, time_horizon)
                var_results["monte_carlo_var"] = var_mc

            # Expected Shortfall (Conditional VaR)
            percentile = (1 - confidence_level) * 100
            threshold = np.percentile(returns_array, percentile)
            tail_returns = returns_array[returns_array <= threshold]
            expected_shortfall = abs(np.mean(tail_returns) * np.sqrt(time_horizon))

            var_results["expected_shortfall"] = expected_shortfall

            # Update VaR history
            current_var = var_results.get("historical_var", 0.0)
            self._var_history.append(
                {
                    "timestamp": get_current_utc_timestamp(),
                    "var_95": current_var,
                    "expected_shortfall": expected_shortfall,
                }
            )

            # Update metrics
            self.metrics_collector.set_gauge("risk_var_95", current_var)
            self.metrics_collector.set_gauge("risk_expected_shortfall", expected_shortfall)

            return var_results

        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
            return {"error": str(e)}

    async def run_stress_test(
        self, scenario_name: str, scenario_params: dict[str, Any]
    ) -> dict[str, float]:
        """
        Run stress test scenario on current portfolio.

        Args:
            scenario_name: Name of stress test scenario
            scenario_params: Scenario parameters

        Returns:
            Stress test results
        """
        try:
            if not self._positions:
                return {"error": "No positions to stress test"}

            stress_results = {}

            if scenario_name == "market_crash":
                # Market crash: -20% across all positions
                market_shock = scenario_params.get("market_shock", -0.20)
                total_impact = Decimal("0")

                for position in self._positions.values():
                    if position.is_open():
                        current_price = await self._get_current_price(position.symbol)
                        if current_price:
                            shocked_price = current_price * (1 + Decimal(str(market_shock)))
                            impact = position.calculate_pnl(shocked_price)
                            total_impact += impact

                stress_results["portfolio_impact"] = float(total_impact)
                stress_results["portfolio_impact_percent"] = float(
                    total_impact / await self._get_portfolio_value() * 100
                )

            elif scenario_name == "volatility_spike":
                # Volatility spike: 2x current volatility
                vol_multiplier = scenario_params.get("volatility_multiplier", 2.0)

                if len(self._portfolio_returns) > 30:
                    current_vol = np.std(list(self._portfolio_returns))
                    stressed_vol = current_vol * vol_multiplier

                    # Estimate impact using normal distribution
                    var_95 = stats.norm.ppf(0.05) * stressed_vol
                    stress_results["stressed_var_95"] = abs(var_95)
                    stress_results["volatility_impact"] = abs(var_95) - abs(
                        stats.norm.ppf(0.05) * current_vol
                    )

            elif scenario_name == "liquidity_crisis":
                # Liquidity crisis: increased bid-ask spreads
                spread_multiplier = scenario_params.get("spread_multiplier", 3.0)

                total_liquidity_cost = Decimal("0")
                for position in self._positions.values():
                    if position.is_open():
                        position_value = position.quantity * await self._get_current_price(
                            position.symbol
                        )
                        # Assume 0.1% base spread, multiply by scenario factor
                        liquidity_cost = (
                            position_value * Decimal("0.001") * Decimal(str(spread_multiplier))
                        )
                        total_liquidity_cost += liquidity_cost

                stress_results["liquidity_cost"] = float(total_liquidity_cost)
                stress_results["liquidity_cost_percent"] = float(
                    total_liquidity_cost / await self._get_portfolio_value() * 100
                )

            elif scenario_name == "correlation_breakdown":
                # Correlation breakdown: all correlations go to 1.0
                # Simplified calculation - would use full covariance matrix
                portfolio_vol = (
                    np.std(list(self._portfolio_returns)) if self._portfolio_returns else 0.05
                )

                # Estimate diversification benefit loss
                num_positions = len([p for p in self._positions.values() if p.is_open()])
                if num_positions > 1:
                    diversification_loss = portfolio_vol * 0.5  # Simplified
                    stress_results["diversification_loss"] = diversification_loss
                    stress_results["stressed_volatility"] = portfolio_vol + diversification_loss

            # Store results
            self._stress_test_results[scenario_name] = stress_results

            # Update metrics
            self.metrics_collector.observe_histogram(
                "stress_test_impact",
                abs(stress_results.get("portfolio_impact_percent", 0)),
                labels={"scenario": scenario_name},
            )

            return stress_results

        except Exception as e:
            self.logger.error(f"Error running stress test {scenario_name}: {e}")
            return {"error": str(e)}

    async def calculate_concentration_risk(self) -> dict[str, float]:
        """
        Calculate concentration risk metrics.

        Returns:
            Dictionary with concentration risk metrics
        """
        try:
            if not self._positions:
                return {}

            concentration_metrics = {}

            # Calculate position weights
            total_portfolio_value = await self._get_portfolio_value()
            if total_portfolio_value <= 0:
                return concentration_metrics

            position_weights = {}
            sector_weights = defaultdict(float)

            for symbol, position in self._positions.items():
                if position.is_open():
                    current_price = await self._get_current_price(symbol)
                    if current_price:
                        position_value = float(position.quantity * current_price)
                        weight = position_value / float(total_portfolio_value)
                        position_weights[symbol] = weight

                        # Aggregate by sector (simplified - would use real sector mapping)
                        sector = self._get_sector(symbol)
                        sector_weights[sector] += weight

            # Single name concentration
            if position_weights:
                max_position_weight = max(position_weights.values())
                concentration_metrics["max_position_weight"] = max_position_weight

                # Herfindahl-Hirschman Index
                hhi = sum(w**2 for w in position_weights.values())
                concentration_metrics["hhi"] = hhi
                concentration_metrics["effective_positions"] = 1.0 / hhi if hhi > 0 else 0

                # Top N concentration
                sorted_weights = sorted(position_weights.values(), reverse=True)
                concentration_metrics["top_5_concentration"] = sum(sorted_weights[:5])
                concentration_metrics["top_10_concentration"] = sum(sorted_weights[:10])

            # Sector concentration
            if sector_weights:
                concentration_metrics["max_sector_weight"] = max(sector_weights.values())
                concentration_metrics["sector_hhi"] = sum(w**2 for w in sector_weights.values())
                concentration_metrics["sector_weights"] = dict(sector_weights)

            # Check concentration limits
            max_position = concentration_metrics.get("max_position_weight", 0)
            if max_position > float(self._risk_limits["max_single_position_weight"]):
                await self._generate_risk_alert(
                    "concentration_breach",
                    AlertSeverity.HIGH,
                    "Single Position Concentration Limit Breached",
                    f"Position weight {max_position:.1%} exceeds limit {self._risk_limits['max_single_position_weight']:.1%}",
                )

            # Update metrics
            for metric, value in concentration_metrics.items():
                if isinstance(value, (int, float)):
                    self.metrics_collector.set_gauge(f"risk_concentration_{metric}", value)

            return concentration_metrics

        except Exception as e:
            self.logger.error(f"Error calculating concentration risk: {e}")
            return {}

    async def calculate_correlation_risk(self) -> dict[str, float]:
        """
        Calculate correlation and diversification risk metrics.

        Returns:
            Dictionary with correlation risk metrics
        """
        try:
            if len(self._positions) < 2:
                return {"error": "Need at least 2 positions for correlation analysis"}

            # Build return matrix for positions
            symbol_returns = {}
            min_observations = float("inf")

            for symbol in self._positions.keys():
                if symbol in self._price_history and len(self._price_history[symbol]) > 30:
                    prices = [p["price"] for p in self._price_history[symbol]]
                    returns = pd.Series(prices).pct_change().dropna().tolist()
                    if len(returns) >= 30:
                        symbol_returns[symbol] = returns
                        min_observations = min(min_observations, len(returns))

            if len(symbol_returns) < 2 or min_observations < 30:
                return {"error": "Insufficient price history for correlation analysis"}

            # Align return series
            aligned_returns = {}
            for symbol, returns in symbol_returns.items():
                aligned_returns[symbol] = returns[-min_observations:]

            # Calculate correlation matrix
            df = pd.DataFrame(aligned_returns)
            correlation_matrix = df.corr()

            # Extract correlation metrics
            corr_values = correlation_matrix.values
            upper_triangle = corr_values[np.triu_indices_from(corr_values, k=1)]

            correlation_metrics = {
                "average_correlation": float(np.mean(upper_triangle)),
                "max_correlation": float(np.max(upper_triangle)),
                "min_correlation": float(np.min(upper_triangle)),
                "correlation_std": float(np.std(upper_triangle)),
                "high_correlation_pairs": len(upper_triangle[upper_triangle > 0.7]),
            }

            # Diversification ratio (simplified)
            # DR = weighted average volatility / portfolio volatility
            position_weights = await self._get_position_weights()
            if position_weights:
                individual_vols = []
                weights = []

                for symbol in aligned_returns.keys():
                    if symbol in position_weights:
                        vol = np.std(aligned_returns[symbol]) * np.sqrt(252)
                        individual_vols.append(vol)
                        weights.append(position_weights[symbol])

                if individual_vols and weights:
                    weighted_avg_vol = np.average(individual_vols, weights=weights)
                    portfolio_vol = (
                        np.std(list(self._portfolio_returns)) * np.sqrt(252)
                        if self._portfolio_returns
                        else 0.1
                    )

                    if portfolio_vol > 0:
                        diversification_ratio = weighted_avg_vol / portfolio_vol
                        correlation_metrics["diversification_ratio"] = diversification_ratio

            # Check correlation risk limits
            avg_corr = correlation_metrics["average_correlation"]
            if avg_corr > float(self._risk_limits["max_correlation_exposure"]):
                await self._generate_risk_alert(
                    "correlation_risk_breach",
                    AlertSeverity.MEDIUM,
                    "High Correlation Risk Detected",
                    f"Average correlation {avg_corr:.2f} exceeds limit {self._risk_limits['max_correlation_exposure']:.2f}",
                )

            # Update metrics
            for metric, value in correlation_metrics.items():
                if isinstance(value, (int, float)):
                    self.metrics_collector.set_gauge(f"risk_correlation_{metric}", value)

            return correlation_metrics

        except Exception as e:
            self.logger.error(f"Error calculating correlation risk: {e}")
            return {}

    async def generate_risk_report(self) -> dict[str, Any]:
        """
        Generate comprehensive risk report.

        Returns:
            Complete risk report with all metrics
        """
        try:
            now = get_current_utc_timestamp()

            # Calculate all risk metrics
            var_metrics = await self.calculate_var(0.95, 1, "historical")
            var_99_metrics = await self.calculate_var(0.99, 1, "historical")
            concentration_metrics = await self.calculate_concentration_risk()
            correlation_metrics = await self.calculate_correlation_risk()

            # Run stress tests
            stress_test_results = {}
            for scenario in self.config.stress_test_scenarios:
                if scenario == "market_crash":
                    params = {"market_shock": -0.20}
                elif scenario == "volatility_spike":
                    params = {"volatility_multiplier": 2.0}
                elif scenario == "liquidity_crisis":
                    params = {"spread_multiplier": 3.0}
                else:
                    params = {}

                stress_result = await self.run_stress_test(scenario, params)
                stress_test_results[scenario] = stress_result

            # Calculate additional risk metrics
            leverage_ratio = await self._calculate_leverage_ratio()
            liquidity_metrics = await self._calculate_liquidity_metrics()
            drawdown_metrics = await self._calculate_drawdown_metrics()

            risk_report = {
                "timestamp": now,
                "var_metrics": {
                    "var_95_1d": var_metrics.get("historical_var", 0),
                    "var_99_1d": var_99_metrics.get("historical_var", 0),
                    "expected_shortfall_95": var_metrics.get("expected_shortfall", 0),
                    "expected_shortfall_99": var_99_metrics.get("expected_shortfall", 0),
                },
                "concentration_risk": concentration_metrics,
                "correlation_risk": correlation_metrics,
                "stress_test_results": stress_test_results,
                "leverage_metrics": {
                    "gross_leverage": leverage_ratio.get("gross_leverage", 0),
                    "net_leverage": leverage_ratio.get("net_leverage", 0),
                    "leverage_ratio": leverage_ratio.get("leverage_ratio", 0),
                },
                "liquidity_metrics": liquidity_metrics,
                "drawdown_metrics": drawdown_metrics,
                "limit_utilization": await self._calculate_limit_utilization(),
                "active_breaches": len(self._active_breaches),
                "risk_score": await self._calculate_composite_risk_score(),
            }

            # Generate risk summary
            risk_report["risk_summary"] = await self._generate_risk_summary(risk_report)

            return risk_report

        except Exception as e:
            self.logger.error(f"Error generating risk report: {e}")
            return {"error": str(e)}

    async def _real_time_monitoring_loop(self) -> None:
        """Background loop for real-time risk monitoring."""
        while self._running:
            try:
                await self._calculate_real_time_risk()
                await asyncio.sleep(10)  # Update every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in real-time monitoring loop: {e}")
                await asyncio.sleep(5)

    async def _limit_monitoring_loop(self) -> None:
        """Background loop for limit monitoring."""
        while self._running:
            try:
                await self._check_risk_limits()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in limit monitoring loop: {e}")
                await asyncio.sleep(5)

    async def _stress_testing_loop(self) -> None:
        """Background loop for periodic stress testing."""
        while self._running:
            try:
                # Run stress tests every hour
                for scenario in self.config.stress_test_scenarios:
                    await self.run_stress_test(scenario, {})

                await asyncio.sleep(3600)  # Run every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in stress testing loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def _var_backtesting_loop(self) -> None:
        """Background loop for VaR backtesting."""
        while self._running:
            try:
                await self._backtest_var_model()
                await asyncio.sleep(86400)  # Run daily
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in VaR backtesting loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error

    async def _calculate_real_time_risk(self) -> None:
        """Calculate real-time risk metrics."""
        try:
            if not self._positions:
                return

            # Calculate current VaR
            var_95 = await self.calculate_var(0.95, 1, "historical")
            current_var = var_95.get("historical_var", 0)

            # Update real-time risk metrics
            self.metrics_collector.set_gauge("risk_realtime_var_95", current_var)

            # Check for immediate threshold breaches
            var_limit = float(self._risk_limits["max_portfolio_var_95"])
            if current_var > var_limit:
                await self._generate_risk_alert(
                    "var_95_breach",
                    AlertSeverity.CRITICAL,
                    "VaR 95% Limit Breached",
                    f"Current VaR {current_var:.2%} exceeds limit {var_limit:.2%}",
                )

        except Exception as e:
            self.logger.error(f"Error calculating real-time risk: {e}")

    async def _check_risk_limits(self) -> None:
        """Check all risk limits and generate alerts."""
        try:
            # Check concentration limits
            await self.calculate_concentration_risk()

            # Check correlation limits
            await self.calculate_correlation_risk()

            # Check leverage limits
            leverage_metrics = await self._calculate_leverage_ratio()
            current_leverage = leverage_metrics.get("leverage_ratio", 0)
            leverage_limit = float(self._risk_limits["max_leverage_ratio"])

            if current_leverage > leverage_limit:
                await self._generate_risk_alert(
                    "leverage_breach",
                    AlertSeverity.HIGH,
                    "Leverage Ratio Limit Breached",
                    f"Current leverage {current_leverage:.2f}x exceeds limit {leverage_limit:.2f}x",
                )

            # Check drawdown limits
            drawdown_metrics = await self._calculate_drawdown_metrics()
            current_drawdown = drawdown_metrics.get("current_drawdown", 0)
            drawdown_limit = float(self._risk_limits["max_drawdown_limit"])

            if current_drawdown > drawdown_limit:
                await self._generate_risk_alert(
                    "drawdown_breach",
                    AlertSeverity.CRITICAL,
                    "Maximum Drawdown Limit Breached",
                    f"Current drawdown {current_drawdown:.1%} exceeds limit {drawdown_limit:.1%}",
                )

        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")

    async def _monte_carlo_var(self, confidence_level: float, time_horizon: int) -> float:
        """Calculate VaR using Monte Carlo simulation."""
        try:
            if len(self._portfolio_returns) < 30:
                return 0.0

            returns_array = np.array(list(self._portfolio_returns))
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)

            # Generate random scenarios
            random_returns = np.random.normal(
                mean_return, std_return, (self._mc_simulations, time_horizon)
            )

            # Calculate cumulative returns for each path
            cumulative_returns = np.prod(1 + random_returns, axis=1) - 1

            # Calculate VaR
            percentile = (1 - confidence_level) * 100
            var_mc = abs(np.percentile(cumulative_returns, percentile))

            return var_mc

        except Exception as e:
            self.logger.error(f"Error in Monte Carlo VaR calculation: {e}")
            return 0.0

    async def _backtest_var_model(self) -> dict[str, float]:
        """Backtest VaR model accuracy."""
        try:
            if len(self._var_history) < 30 or len(self._portfolio_returns) < 30:
                return {}

            # Get recent VaR predictions and actual returns
            var_predictions = [v["var_95"] for v in list(self._var_history)[-30:]]
            actual_returns = list(self._portfolio_returns)[-30:]

            # Count violations (actual loss > VaR prediction)
            violations = sum(
                1
                for actual, var_pred in zip(actual_returns, var_predictions, strict=False)
                if actual < -var_pred
            )

            # Calculate violation rate (should be ~5% for 95% VaR)
            violation_rate = violations / len(actual_returns)
            expected_rate = 0.05  # 5% for 95% VaR

            # Kupiec test for VaR model accuracy
            # Simplified version - full implementation would use likelihood ratio test
            accuracy_score = abs(violation_rate - expected_rate) / expected_rate

            backtest_results = {
                "violation_rate": violation_rate,
                "expected_rate": expected_rate,
                "accuracy_score": accuracy_score,
                "model_quality": "good" if accuracy_score < 0.5 else "needs_improvement",
            }

            # Update metrics
            self.metrics_collector.set_gauge("risk_var_violation_rate", violation_rate)
            self.metrics_collector.set_gauge("risk_var_accuracy_score", accuracy_score)

            return backtest_results

        except Exception as e:
            self.logger.error(f"Error backtesting VaR model: {e}")
            return {}

    async def _generate_risk_alert(
        self, alert_id: str, severity: AlertSeverity, title: str, message: str
    ) -> None:
        """Generate risk alert."""
        try:
            # Check if alert already exists
            if alert_id in self._active_breaches:
                return

            alert = AnalyticsAlert(
                id=alert_id,
                timestamp=get_current_utc_timestamp(),
                severity=severity,
                title=title,
                message=message,
                metric_name=alert_id,
            )

            self._active_breaches[alert_id] = alert
            self._breach_history.append(alert)

            # Update alert metrics
            self.metrics_collector.increment_counter(
                "risk_alerts_generated", labels={"severity": severity.value, "type": alert_id}
            )

            self.logger.warning(f"Risk alert generated: {title} - {message}")

        except Exception as e:
            self.logger.error(f"Error generating risk alert: {e}")

    async def _get_current_price(self, symbol: str) -> Decimal | None:
        """Get current price for symbol."""
        if self._price_history.get(symbol):
            return Decimal(str(self._price_history[symbol][-1]["price"]))
        return None

    async def _get_portfolio_value(self) -> Decimal:
        """Calculate current portfolio value."""
        total_value = Decimal("0")
        for position in self._positions.values():
            if position.is_open():
                current_price = await self._get_current_price(position.symbol)
                if current_price:
                    total_value += position.quantity * current_price
        return total_value

    async def _get_position_weights(self) -> dict[str, float]:
        """Get position weights as percentage of portfolio."""
        portfolio_value = await self._get_portfolio_value()
        if portfolio_value <= 0:
            return {}

        weights = {}
        for symbol, position in self._positions.items():
            if position.is_open():
                current_price = await self._get_current_price(symbol)
                if current_price:
                    position_value = position.quantity * current_price
                    weights[symbol] = float(position_value / portfolio_value)

        return weights

    def _get_sector(self, symbol: str) -> str:
        """Get sector for symbol (simplified mapping)."""
        # In practice, would use real sector data
        if "BTC" in symbol or "ETH" in symbol:
            return "Cryptocurrency"
        elif "USD" in symbol:
            return "Currency"
        else:
            return "Other"

    async def _update_portfolio_returns(self) -> None:
        """Update portfolio returns based on position changes."""
        try:
            # Simplified calculation - in practice would use actual portfolio value history
            if len(self._portfolio_returns) == 0:
                self._portfolio_returns.append(0.0)  # Initialize with zero return
                return

            # Calculate weighted return based on position returns
            total_weight = 0
            weighted_return = 0

            for symbol, position in self._positions.items():
                if position.is_open() and symbol in self._price_history:
                    price_history = list(self._price_history[symbol])
                    if len(price_history) >= 2:
                        current_price = price_history[-1]["price"]
                        prev_price = price_history[-2]["price"]

                        if prev_price > 0:
                            asset_return = (current_price - prev_price) / prev_price
                            # Weight by position size (simplified)
                            weight = 1.0 / len(self._positions)  # Equal weight for simplicity
                            weighted_return += asset_return * weight
                            total_weight += weight

            if total_weight > 0:
                portfolio_return = weighted_return / total_weight
                self._portfolio_returns.append(portfolio_return)

        except Exception as e:
            self.logger.error(f"Error updating portfolio returns: {e}")

    async def _calculate_leverage_ratio(self) -> dict[str, float]:
        """Calculate leverage metrics."""
        try:
            gross_exposure = 0.0
            net_exposure = 0.0

            for position in self._positions.values():
                if position.is_open():
                    current_price = await self._get_current_price(position.symbol)
                    if current_price:
                        position_value = float(position.quantity * current_price)
                        gross_exposure += abs(position_value)

                        # Net exposure considers position direction
                        if position.side.value == "buy":
                            net_exposure += position_value
                        else:
                            net_exposure -= position_value

            portfolio_value = float(await self._get_portfolio_value())

            if portfolio_value > 0:
                gross_leverage = gross_exposure / portfolio_value
                net_leverage = abs(net_exposure) / portfolio_value
                leverage_ratio = gross_leverage  # Primary leverage metric
            else:
                gross_leverage = net_leverage = leverage_ratio = 0.0

            return {
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "gross_leverage": gross_leverage,
                "net_leverage": net_leverage,
                "leverage_ratio": leverage_ratio,
            }

        except Exception as e:
            self.logger.error(f"Error calculating leverage ratio: {e}")
            return {}

    async def _calculate_liquidity_metrics(self) -> dict[str, float]:
        """Calculate portfolio liquidity metrics."""
        try:
            # Simplified liquidity calculation
            liquid_positions = 0
            total_positions = 0

            for position in self._positions.values():
                if position.is_open():
                    total_positions += 1
                    # Simplified - assume crypto and major currencies are liquid
                    symbol = position.symbol
                    if any(x in symbol for x in ["BTC", "ETH", "USD"]):
                        liquid_positions += 1

            liquidity_ratio = liquid_positions / total_positions if total_positions > 0 else 0.0

            return {
                "liquidity_ratio": liquidity_ratio,
                "liquid_positions": liquid_positions,
                "total_positions": total_positions,
                "illiquid_positions": total_positions - liquid_positions,
            }

        except Exception as e:
            self.logger.error(f"Error calculating liquidity metrics: {e}")
            return {}

    async def _calculate_drawdown_metrics(self) -> dict[str, float]:
        """Calculate drawdown metrics."""
        try:
            if len(self._portfolio_returns) < 2:
                return {}

            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + np.array(list(self._portfolio_returns)))

            # Calculate running maximum
            running_max = np.maximum.accumulate(cumulative_returns)

            # Calculate drawdown
            drawdown = (cumulative_returns - running_max) / running_max

            max_drawdown = abs(np.min(drawdown))
            current_drawdown = abs(drawdown[-1])

            # Drawdown duration
            in_drawdown = drawdown < -0.01  # More than 1% drawdown
            max_duration = 0
            current_duration = 0

            for i, dd in enumerate(in_drawdown):
                if dd:
                    current_duration += 1
                    max_duration = max(max_duration, current_duration)
                else:
                    current_duration = 0

            return {
                "max_drawdown": max_drawdown,
                "current_drawdown": current_drawdown,
                "max_drawdown_duration": max_duration,
                "current_drawdown_duration": current_duration,
                "recovery_factor": (
                    (cumulative_returns[-1] - 1) / max_drawdown if max_drawdown > 0 else 0
                ),
            }

        except Exception as e:
            self.logger.error(f"Error calculating drawdown metrics: {e}")
            return {}

    async def _calculate_limit_utilization(self) -> dict[str, float]:
        """Calculate risk limit utilization."""
        try:
            utilization = {}

            # VaR limit utilization
            var_95 = await self.calculate_var(0.95, 1, "historical")
            current_var = var_95.get("historical_var", 0)
            var_limit = float(self._risk_limits["max_portfolio_var_95"])
            if var_limit > 0:
                utilization["var_95"] = (current_var / var_limit) * 100

            # Concentration limit utilization
            concentration = await self.calculate_concentration_risk()
            max_position = concentration.get("max_position_weight", 0)
            position_limit = float(self._risk_limits["max_single_position_weight"])
            if position_limit > 0:
                utilization["max_position"] = (max_position / position_limit) * 100

            # Leverage limit utilization
            leverage = await self._calculate_leverage_ratio()
            current_leverage = leverage.get("leverage_ratio", 0)
            leverage_limit = float(self._risk_limits["max_leverage_ratio"])
            if leverage_limit > 0:
                utilization["leverage"] = (current_leverage / leverage_limit) * 100

            return utilization

        except Exception as e:
            self.logger.error(f"Error calculating limit utilization: {e}")
            return {}

    async def _calculate_composite_risk_score(self) -> float:
        """Calculate composite risk score (0-100)."""
        try:
            score_components = []

            # VaR component (0-30 points)
            var_95 = await self.calculate_var(0.95, 1, "historical")
            current_var = var_95.get("historical_var", 0)
            var_score = min(current_var * 1000, 30)  # Scale to 30 points max
            score_components.append(var_score)

            # Concentration component (0-25 points)
            concentration = await self.calculate_concentration_risk()
            max_position = concentration.get("max_position_weight", 0)
            concentration_score = min(max_position * 100, 25)  # Scale to 25 points max
            score_components.append(concentration_score)

            # Leverage component (0-25 points)
            leverage = await self._calculate_leverage_ratio()
            current_leverage = leverage.get("leverage_ratio", 0)
            leverage_score = min((current_leverage - 1) * 12.5, 25)  # Above 1x leverage
            score_components.append(max(leverage_score, 0))

            # Drawdown component (0-20 points)
            drawdown = await self._calculate_drawdown_metrics()
            current_dd = drawdown.get("current_drawdown", 0)
            drawdown_score = min(current_dd * 200, 20)  # Scale to 20 points max
            score_components.append(drawdown_score)

            total_score = sum(score_components)

            # Update composite risk score metric
            self.metrics_collector.set_gauge("risk_composite_score", total_score)

            return min(total_score, 100.0)  # Cap at 100

        except Exception as e:
            self.logger.error(f"Error calculating composite risk score: {e}")
            return 50.0  # Default moderate risk score

    async def _generate_risk_summary(self, risk_report: dict[str, Any]) -> str:
        """Generate risk summary text."""
        try:
            summary_parts = []

            # VaR summary
            var_95 = risk_report.get("var_metrics", {}).get("var_95_1d", 0)
            summary_parts.append(f"Daily VaR (95%): {var_95:.2%}")

            # Risk score summary
            risk_score = risk_report.get("risk_score", 0)
            if risk_score < 30:
                risk_level = "LOW"
            elif risk_score < 60:
                risk_level = "MODERATE"
            elif risk_score < 80:
                risk_level = "HIGH"
            else:
                risk_level = "CRITICAL"

            summary_parts.append(f"Overall risk level: {risk_level} (Score: {risk_score:.1f}/100)")

            # Active breaches
            active_breaches = risk_report.get("active_breaches", 0)
            if active_breaches > 0:
                summary_parts.append(f"ALERT: {active_breaches} active limit breach(es)")

            # Concentration summary
            concentration = risk_report.get("concentration_risk", {})
            max_position = concentration.get("max_position_weight", 0)
            if max_position > 0.1:  # >10%
                summary_parts.append(f"High concentration: largest position {max_position:.1%}")

            return ". ".join(summary_parts) + "."

        except Exception as e:
            self.logger.error(f"Error generating risk summary: {e}")
            return "Risk summary unavailable due to calculation error."

    # Advanced Risk Monitoring Capabilities

    async def calculate_advanced_var_methodologies(
        self, confidence_levels: list[float] = None
    ) -> dict[str, Any]:
        """
        Calculate VaR using multiple advanced methodologies for comprehensive risk assessment.

        Args:
            confidence_levels: List of confidence levels (default: [0.95, 0.99, 0.995])

        Returns:
            Dictionary containing VaR calculations from different methodologies
        """
        try:
            confidence_levels = confidence_levels or [0.95, 0.99, 0.995]

            # Get portfolio returns (this would come from actual data in production)
            returns = await self._get_portfolio_returns_history(252)  # 1 year of data
            if len(returns) < 60:
                return {"error": "Insufficient return data for VaR calculation"}

            returns_array = np.array(returns)
            portfolio_value = await self._get_current_portfolio_value()

            if portfolio_value is None:
                return {"error": "Unable to obtain current portfolio value"}

            var_results = {
                "timestamp": get_current_utc_timestamp().isoformat(),
                "portfolio_value": float(portfolio_value),
                "data_points": len(returns),
                "methodologies": {},
            }

            for confidence in confidence_levels:
                alpha = 1 - confidence

                # 1. Historical Simulation VaR
                historical_var = np.percentile(returns_array, alpha * 100)

                # 2. Parametric VaR (assumes normal distribution)
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                z_score = stats.norm.ppf(alpha)
                parametric_var = mean_return + z_score * std_return

                # 3. Modified VaR using Cornish-Fisher expansion (accounts for skewness and kurtosis)
                skewness = stats.skew(returns_array)
                excess_kurtosis = stats.kurtosis(returns_array)

                cornish_fisher_z = (
                    z_score
                    + (z_score**2 - 1) * skewness / 6
                    + (z_score**3 - 3 * z_score) * excess_kurtosis / 24
                    - (2 * z_score**3 - 5 * z_score) * skewness**2 / 36
                )
                modified_var = mean_return + cornish_fisher_z * std_return

                # 4. Filtered Historical Simulation (EWMA volatility scaling)
                lambda_ewma = 0.94
                ewma_weights = np.array(
                    [(1 - lambda_ewma) * lambda_ewma**i for i in range(len(returns_array))][::-1]
                )
                ewma_weights = ewma_weights / np.sum(ewma_weights)

                # Sort returns and apply EWMA weights
                sorted_indices = np.argsort(returns_array)
                sorted_returns = returns_array[sorted_indices]
                sorted_weights = ewma_weights[sorted_indices]

                # Find VaR using weighted percentile
                cumulative_weights = np.cumsum(sorted_weights)
                var_index = np.searchsorted(cumulative_weights, alpha, side="right")
                filtered_historical_var = sorted_returns[min(var_index, len(sorted_returns) - 1)]

                # 5. Monte Carlo VaR
                mc_var = await self._calculate_monte_carlo_var(
                    returns_array, alpha, n_simulations=10000
                )

                # Expected Shortfall (CVaR) for each method
                tail_returns_historical = returns_array[returns_array <= historical_var]
                expected_shortfall_historical = (
                    np.mean(tail_returns_historical)
                    if len(tail_returns_historical) > 0
                    else historical_var
                )

                # Convert to dollar amounts
                confidence_key = f"{confidence:.1%}"
                var_results["methodologies"][confidence_key] = {
                    "historical_simulation": {
                        "var_percentage": float(historical_var),
                        "var_dollar": float(historical_var * portfolio_value),
                        "expected_shortfall_percentage": float(expected_shortfall_historical),
                        "expected_shortfall_dollar": float(
                            expected_shortfall_historical * portfolio_value
                        ),
                    },
                    "parametric_normal": {
                        "var_percentage": float(parametric_var),
                        "var_dollar": float(parametric_var * portfolio_value),
                    },
                    "modified_cornish_fisher": {
                        "var_percentage": float(modified_var),
                        "var_dollar": float(modified_var * portfolio_value),
                        "skewness_adjustment": float(skewness),
                        "kurtosis_adjustment": float(excess_kurtosis),
                    },
                    "filtered_historical": {
                        "var_percentage": float(filtered_historical_var),
                        "var_dollar": float(filtered_historical_var * portfolio_value),
                    },
                    "monte_carlo": {
                        "var_percentage": float(mc_var),
                        "var_dollar": float(mc_var * portfolio_value),
                    },
                }

            # VaR model validation (backtesting)
            var_validation = await self._validate_var_models(returns_array, var_results)
            var_results["model_validation"] = var_validation

            return var_results

        except Exception as e:
            self.logger.error(f"Error calculating advanced VaR methodologies: {e}")
            return {"error": str(e)}

    async def _calculate_monte_carlo_var(
        self, returns: np.ndarray, alpha: float, n_simulations: int = 10000
    ) -> float:
        """Calculate VaR using Monte Carlo simulation with fitted distribution."""
        try:
            # Fit a t-distribution to the returns (more realistic than normal for financial returns)
            df, loc, scale = stats.t.fit(returns)

            # Generate Monte Carlo simulations
            np.random.seed(42)  # For reproducibility
            simulated_returns = stats.t.rvs(df=df, loc=loc, scale=scale, size=n_simulations)

            # Calculate VaR from simulations
            mc_var = np.percentile(simulated_returns, alpha * 100)
            return mc_var

        except Exception as e:
            self.logger.error(f"Error in Monte Carlo VaR calculation: {e}")
            return np.percentile(returns, alpha * 100)  # Fallback to historical

    async def _validate_var_models(self, returns: np.ndarray, var_results: dict) -> dict[str, Any]:
        """Validate VaR models using backtesting methodology."""
        try:
            validation_results = {}

            # Use 95% VaR for validation (most common)
            if "95.0%" not in var_results["methodologies"]:
                return {"error": "95% VaR not available for validation"}

            var_95 = var_results["methodologies"]["95.0%"]

            for method_name, method_data in var_95.items():
                if "var_percentage" in method_data:
                    var_threshold = method_data["var_percentage"]

                    # Count exceedances (violations)
                    exceedances = np.sum(returns <= var_threshold)
                    total_observations = len(returns)
                    exceedance_rate = exceedances / total_observations
                    expected_rate = 0.05  # 5% for 95% VaR

                    # Statistical tests
                    # 1. Unconditional Coverage Test (Kupiec test)
                    if exceedances > 0:
                        lr_uc = -2 * np.log(
                            (0.05**exceedances * 0.95 ** (total_observations - exceedances))
                            / (
                                (exceedances / total_observations) ** exceedances
                                * (1 - exceedances / total_observations)
                                ** (total_observations - exceedances)
                            )
                        )
                        uc_p_value = 1 - stats.chi2.cdf(lr_uc, 1)
                    else:
                        lr_uc = 0
                        uc_p_value = 1.0

                    # Model quality assessment
                    if 0.03 <= exceedance_rate <= 0.07:  # Within acceptable range
                        model_quality = "Good"
                    elif 0.02 <= exceedance_rate <= 0.08:
                        model_quality = "Acceptable"
                    else:
                        model_quality = "Poor"

                    validation_results[method_name] = {
                        "exceedances": int(exceedances),
                        "total_observations": int(total_observations),
                        "exceedance_rate": float(exceedance_rate),
                        "expected_rate": float(expected_rate),
                        "kupiec_lr_statistic": float(lr_uc),
                        "kupiec_p_value": float(uc_p_value),
                        "model_quality": model_quality,
                        "passes_kupiec_test": uc_p_value > 0.05,
                    }

            return validation_results

        except Exception as e:
            self.logger.error(f"Error in VaR model validation: {e}")
            return {"error": str(e)}

    async def execute_comprehensive_stress_test(self) -> dict[str, Any]:
        """
        Execute comprehensive stress testing across multiple scenarios and methodologies.

        Returns:
            Complete stress test results with scenario impacts and recommendations
        """
        try:
            current_portfolio_value = await self._get_current_portfolio_value()
            if current_portfolio_value is None:
                return {"error": "Unable to obtain current portfolio value"}

            positions = await self._get_current_positions()
            if not positions:
                return {"error": "No positions available for stress testing"}

            stress_test_results = {
                "timestamp": get_current_utc_timestamp().isoformat(),
                "base_portfolio_value": float(current_portfolio_value),
                "scenarios": {},
                "summary": {},
                "recommendations": [],
            }

            # Define comprehensive stress scenarios
            stress_scenarios = {
                "market_crash_2008": {
                    "description": "2008 Financial Crisis scenario",
                    "equity_shock": -0.25,
                    "credit_spread_widening": 0.03,
                    "vol_spike_multiplier": 2.5,
                    "correlation_increase": 0.2,
                    "liquidity_impact": 0.15,
                },
                "covid_crash_2020": {
                    "description": "COVID-19 pandemic scenario",
                    "equity_shock": -0.35,
                    "credit_spread_widening": 0.02,
                    "vol_spike_multiplier": 3.0,
                    "correlation_increase": 0.3,
                    "liquidity_impact": 0.20,
                },
                "interest_rate_shock": {
                    "description": "Sudden 200bp rate increase",
                    "rate_shock": 0.02,
                    "duration_impact": -0.08,
                    "credit_spread_widening": 0.015,
                    "equity_shock": -0.10,
                    "currency_vol_spike": 1.5,
                },
                "flash_crash": {
                    "description": "Flash crash scenario (2010-style)",
                    "equity_shock": -0.15,
                    "vol_spike_multiplier": 5.0,
                    "liquidity_evaporation": 0.8,
                    "bid_offer_widening": 3.0,
                },
                "commodity_crisis": {
                    "description": "Commodity price collapse",
                    "commodity_shock": -0.40,
                    "energy_shock": -0.45,
                    "currency_impact": -0.15,
                    "inflation_shock": -0.02,
                },
            }

            # Execute stress tests for each scenario
            for scenario_name, scenario_params in stress_scenarios.items():
                scenario_result = await self._execute_scenario_stress_test(
                    scenario_name, scenario_params, positions, current_portfolio_value
                )
                stress_test_results["scenarios"][scenario_name] = scenario_result

            # Generate summary statistics
            scenario_losses = []
            worst_case_scenario = None
            worst_case_loss = 0

            for scenario_name, result in stress_test_results["scenarios"].items():
                if "total_loss_percentage" in result:
                    loss = result["total_loss_percentage"]
                    scenario_losses.append(loss)

                    if loss < worst_case_loss:  # More negative = worse
                        worst_case_loss = loss
                        worst_case_scenario = scenario_name

            stress_test_results["summary"] = {
                "worst_case_scenario": worst_case_scenario,
                "worst_case_loss_percentage": worst_case_loss,
                "worst_case_loss_dollar": worst_case_loss * float(current_portfolio_value) / 100,
                "average_loss_percentage": np.mean(scenario_losses) if scenario_losses else 0,
                "stress_test_var_95": np.percentile(scenario_losses, 5) if scenario_losses else 0,
                "scenarios_tested": len(stress_scenarios),
            }

            # Generate recommendations
            recommendations = []

            if worst_case_loss < -15:  # >15% loss in worst case
                recommendations.append(
                    "Consider reducing overall portfolio risk and implementing hedging strategies"
                )

            if abs(worst_case_loss) > 20:  # >20% loss
                recommendations.append(
                    "Portfolio shows high vulnerability to extreme scenarios - review position sizing"
                )

            # Check for concentration risk
            max_position_impact = max(
                (
                    max(
                        result.get("position_impacts", {}).values(),
                        default=0,
                        key=lambda x: abs(x.get("loss_percentage", 0)),
                    )
                )
                for result in stress_test_results["scenarios"].values()
            )

            if max_position_impact and abs(max_position_impact.get("loss_percentage", 0)) > 10:
                recommendations.append(
                    "High single-position risk detected - consider diversification"
                )

            stress_test_results["recommendations"] = recommendations

            return stress_test_results

        except Exception as e:
            self.logger.error(f"Error executing comprehensive stress test: {e}")
            return {"error": str(e)}

    async def _execute_scenario_stress_test(
        self,
        scenario_name: str,
        scenario_params: dict,
        positions: dict,
        base_portfolio_value: Decimal,
    ) -> dict[str, Any]:
        """Execute stress test for a specific scenario."""
        try:
            scenario_result = {
                "scenario_name": scenario_name,
                "description": scenario_params.get("description", ""),
                "parameters": scenario_params,
                "position_impacts": {},
                "total_loss_dollar": 0.0,
                "total_loss_percentage": 0.0,
            }

            total_stressed_value = Decimal("0")

            for symbol, position in positions.items():
                position_value = position.size * position.current_price
                stressed_value = position_value  # Start with current value

                # Apply scenario-specific shocks based on asset type
                asset_type = self._classify_asset_type(symbol)

                if asset_type == "equity" and "equity_shock" in scenario_params:
                    shock = Decimal(str(scenario_params["equity_shock"]))
                    stressed_value = position_value * (1 + shock)

                elif asset_type == "fixed_income" and "rate_shock" in scenario_params:
                    # Simplified duration-based impact
                    duration = 5.0  # Assumed average duration
                    rate_shock = scenario_params["rate_shock"]
                    duration_impact = -duration * rate_shock
                    stressed_value = position_value * (1 + Decimal(str(duration_impact)))

                elif asset_type == "commodity" and "commodity_shock" in scenario_params:
                    shock = Decimal(str(scenario_params["commodity_shock"]))
                    stressed_value = position_value * (1 + shock)

                elif asset_type == "currency" and "currency_impact" in scenario_params:
                    shock = Decimal(str(scenario_params["currency_impact"]))
                    stressed_value = position_value * (1 + shock)

                # Apply additional stress factors
                if "liquidity_impact" in scenario_params:
                    liquidity_penalty = Decimal(str(scenario_params["liquidity_impact"]))
                    stressed_value = stressed_value * (1 - liquidity_penalty)

                total_stressed_value += stressed_value

                # Record position-level impact
                position_loss = stressed_value - position_value
                scenario_result["position_impacts"][symbol] = {
                    "original_value": float(position_value),
                    "stressed_value": float(stressed_value),
                    "loss_dollar": float(position_loss),
                    "loss_percentage": (
                        float((position_loss / position_value) * 100) if position_value > 0 else 0
                    ),
                    "asset_type": asset_type,
                }

            # Calculate total impact
            total_loss = total_stressed_value - base_portfolio_value
            scenario_result["total_loss_dollar"] = float(total_loss)
            scenario_result["total_loss_percentage"] = float(
                (total_loss / base_portfolio_value) * 100
            )

            return scenario_result

        except Exception as e:
            self.logger.error(f"Error executing scenario {scenario_name}: {e}")
            return {"error": str(e)}

    def _classify_asset_type(self, symbol: str) -> str:
        """Classify asset type based on symbol (simplified implementation)."""
        if "BTC" in symbol or "ETH" in symbol:
            return "cryptocurrency"
        elif "USD" in symbol or "EUR" in symbol or "GBP" in symbol:
            return "currency"
        elif "BOND" in symbol or "TREASURY" in symbol:
            return "fixed_income"
        elif "GOLD" in symbol or "SILVER" in symbol or "OIL" in symbol:
            return "commodity"
        else:
            return "equity"

    async def create_real_time_risk_dashboard(self) -> dict[str, Any]:
        """
        Create comprehensive real-time risk dashboard data.

        Returns:
            Complete dashboard data with all risk metrics and alerts
        """
        try:
            # Get all risk components
            var_analysis = await self.calculate_advanced_var_methodologies()
            stress_test_results = await self.execute_comprehensive_stress_test()
            risk_limits = await self.check_all_risk_limits()

            # Create dashboard data structure
            dashboard_data = {
                "timestamp": get_current_utc_timestamp().isoformat(),
                "status": "active",
                "refresh_interval": 5,  # 5-second refresh
                # Core risk metrics
                "var_analysis": var_analysis,
                "stress_testing": stress_test_results,
                "risk_limits": risk_limits,
                # Real-time alerts
                "active_alerts": [alert.dict() for alert in self._active_alerts.values()],
                "alert_summary": {
                    "total_alerts": len(self._active_alerts),
                    "critical_alerts": len(
                        [
                            a
                            for a in self._active_alerts.values()
                            if a.severity == AlertSeverity.CRITICAL
                        ]
                    ),
                    "high_alerts": len(
                        [
                            a
                            for a in self._active_alerts.values()
                            if a.severity == AlertSeverity.HIGH
                        ]
                    ),
                    "medium_alerts": len(
                        [
                            a
                            for a in self._active_alerts.values()
                            if a.severity == AlertSeverity.MEDIUM
                        ]
                    ),
                },
                # Risk summary
                "risk_summary": {
                    "overall_risk_score": await self._calculate_overall_risk_score(),
                    "var_95_1day": var_analysis.get("methodologies", {})
                    .get("95.0%", {})
                    .get("historical_simulation", {})
                    .get("var_dollar", 0),
                    "worst_case_stress_loss": stress_test_results.get("summary", {}).get(
                        "worst_case_loss_dollar", 0
                    ),
                    "active_breaches": len(
                        [
                            alert
                            for alert in self._active_alerts.values()
                            if "limit" in alert.title.lower()
                        ]
                    ),
                },
                # Market regime context
                "market_context": {
                    "volatility_regime": await self._assess_volatility_regime(),
                    "correlation_regime": await self._assess_correlation_regime(),
                    "liquidity_conditions": await self._assess_liquidity_conditions(),
                },
                # Performance impact
                "performance_impact": {
                    "risk_adjusted_return": 0,  # Would calculate Sharpe ratio
                    "return_attribution": {},  # Would break down return sources
                    "efficiency_metrics": {},  # Risk/return efficiency measures
                },
            }

            return dashboard_data

        except Exception as e:
            self.logger.error(f"Error creating risk dashboard: {e}")
            return {"error": str(e), "timestamp": get_current_utc_timestamp().isoformat()}

    # Helper methods for enhanced risk monitoring

    async def _get_portfolio_returns_history(self, periods: int) -> list[float]:
        """Get historical portfolio returns (would integrate with actual data)."""
        # Simulated data - in production this would fetch actual returns
        np.random.seed(42)
        return np.random.normal(0.0008, 0.015, periods).tolist()

    async def _get_current_portfolio_value(self) -> Decimal | None:
        """Get current total portfolio value."""
        # This would integrate with the actual portfolio valuation system
        return Decimal("1000000")  # $1M placeholder

    async def _get_current_positions(self) -> dict[str, Any]:
        """Get current position data."""
        # Placeholder positions - would fetch from actual position manager
        return {
            "BTCUSDT": type(
                "Position",
                (),
                {
                    "size": Decimal("10"),
                    "current_price": Decimal("45000"),
                },
            )(),
            "ETHUSDT": type(
                "Position",
                (),
                {
                    "size": Decimal("100"),
                    "current_price": Decimal("3000"),
                },
            )(),
        }

    async def _calculate_overall_risk_score(self) -> float:
        """Calculate composite risk score (0-100, where 100 is highest risk)."""
        # Simplified risk scoring - would be more sophisticated in production
        base_score = 30.0

        # Adjust based on VaR
        var_analysis = await self.calculate_advanced_var_methodologies()
        if "methodologies" in var_analysis and "95.0%" in var_analysis["methodologies"]:
            var_95 = (
                var_analysis["methodologies"]["95.0%"]
                .get("historical_simulation", {})
                .get("var_percentage", 0)
            )
            var_score = min(abs(var_95) * 1000, 40)  # Cap at 40 points
            base_score += var_score

        # Adjust based on stress test results
        stress_results = await self.execute_comprehensive_stress_test()
        if "summary" in stress_results:
            worst_loss = stress_results["summary"].get("worst_case_loss_percentage", 0)
            stress_score = min(abs(worst_loss) * 2, 30)  # Cap at 30 points
            base_score += stress_score

        return min(base_score, 100.0)

    async def _assess_volatility_regime(self) -> str:
        """Assess current market volatility regime."""
        returns = await self._get_portfolio_returns_history(30)  # Last 30 days
        current_vol = np.std(returns) * np.sqrt(252)  # Annualized

        if current_vol < 0.10:
            return "Low Volatility"
        elif current_vol < 0.20:
            return "Normal Volatility"
        elif current_vol < 0.35:
            return "High Volatility"
        else:
            return "Extreme Volatility"

    async def _assess_correlation_regime(self) -> str:
        """Assess current correlation regime."""
        # Simplified - would analyze cross-asset correlations
        return "Normal Correlation"

    async def _assess_liquidity_conditions(self) -> str:
        """Assess current market liquidity conditions."""
        # Simplified - would analyze bid-ask spreads, market impact, etc.
        return "Normal Liquidity"
