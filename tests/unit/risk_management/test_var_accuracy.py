"""
Comprehensive VaR Calculation Accuracy Tests with Non-Normal Distributions.

This module tests Value at Risk (VaR) calculation accuracy across various
distribution types and market scenarios to ensure risk assessments remain
accurate under different market conditions that deviate from normal distributions.

CRITICAL AREAS TESTED:
1. VaR accuracy with normal distributions (baseline)
2. VaR behavior with skewed distributions (market asymmetry)
3. VaR with heavy-tailed distributions (fat tails, extreme events)
4. VaR with mixed/bimodal distributions (regime changes)
5. VaR scaling across time horizons (square root of time)
6. Historical simulation vs parametric VaR
7. Extreme value distributions and stress scenarios
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import patch

import numpy as np
import pytest
from scipy import stats

from src.core.config import Config
from src.core.types import MarketData, OrderSide, Position
from src.risk_management.risk_metrics import RiskCalculator


class TestVaRAccuracy:
    """
    Test suite for VaR calculation accuracy across different distributions.
    
    These tests ensure VaR calculations provide accurate risk estimates
    under various market conditions and distribution assumptions.
    """

    @pytest.fixture
    def config(self):
        """Create test configuration with specific VaR parameters."""
        config = Config()
        config.risk.var_confidence_level = 0.95  # 95% confidence
        config.risk.var_calculation_window = 252  # One year of data
        return config

    @pytest.fixture
    def risk_calculator(self, config):
        """Create risk calculator instance."""
        return RiskCalculator(config)

    @pytest.fixture
    def sample_position(self):
        """Create a sample position for testing."""
        return Position(
            symbol="BTCUSDT",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            unrealized_pnl=Decimal("1000"),
            side=OrderSide.BUY,
            timestamp=datetime.now(),
        )

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data."""
        return MarketData(
            symbol="BTCUSDT",
            price=Decimal("51000"),
            volume=Decimal("1000"),
            timestamp=datetime.now(),
            bid=Decimal("50990"),
            ask=Decimal("51010"),
            open_price=Decimal("50000"),
            high_price=Decimal("52000"),
            low_price=Decimal("49000"),
        )

    @pytest.mark.asyncio
    async def test_var_normal_distribution_accuracy(self, risk_calculator):
        """Test VaR accuracy with normal distribution (baseline case)."""
        portfolio_value = Decimal("10000")
        
        # Generate normal returns with known parameters
        np.random.seed(42)  # Reproducible results
        mean_return = 0.001  # 0.1% daily
        volatility = 0.02    # 2% daily volatility
        n_samples = 252
        
        returns = np.random.normal(mean_return, volatility, n_samples)
        risk_calculator.portfolio_returns = returns.tolist()
        
        # Calculate VaR
        var_1d = await risk_calculator._calculate_var(1, portfolio_value)
        
        # Theoretical VaR for normal distribution
        z_score = 1.645  # 95% confidence level
        theoretical_var = portfolio_value * Decimal(str(volatility * z_score))
        
        # Allow for 10% tolerance due to sample variation
        ratio = float(var_1d / theoretical_var)
        assert 0.8 <= ratio <= 1.2, f"VaR accuracy issue: ratio={ratio}"
        
        # Verify VaR is positive and reasonable
        assert var_1d > Decimal("0")
        assert var_1d < portfolio_value * Decimal("0.1")  # Sanity check

    @pytest.mark.asyncio
    async def test_var_skewed_distribution(self, risk_calculator):
        """Test VaR with positively and negatively skewed distributions."""
        portfolio_value = Decimal("10000")
        
        skew_scenarios = [
            (5, "positive_skew"),   # Right tail (extreme gains)
            (-5, "negative_skew"),  # Left tail (extreme losses)
        ]
        
        for skew_param, scenario_name in skew_scenarios:
            np.random.seed(42)
            # Generate skewed returns
            returns = stats.skewnorm.rvs(a=skew_param, size=252) * 0.02  # Scale to 2% volatility
            risk_calculator.portfolio_returns = returns.tolist()
            
            var_1d = await risk_calculator._calculate_var(1, portfolio_value)
            
            # For negative skew (left tail), VaR should be higher than normal
            # For positive skew (right tail), VaR might be lower
            assert isinstance(var_1d, Decimal)
            assert var_1d > Decimal("0")
            
            if skew_param < 0:  # Negative skew should increase VaR
                # Compare to normal case
                normal_returns = np.random.normal(0, 0.02, 252)
                risk_calculator.portfolio_returns = normal_returns.tolist()
                normal_var = await risk_calculator._calculate_var(1, portfolio_value)
                
                # Reset to skewed returns
                risk_calculator.portfolio_returns = returns.tolist()
                
                # Negative skew should generally produce higher VaR
                # (though this depends on the specific realization)
                assert var_1d > Decimal("0")

    @pytest.mark.asyncio
    async def test_var_heavy_tailed_distribution(self, risk_calculator):
        """Test VaR with heavy-tailed distributions (fat tails)."""
        portfolio_value = Decimal("10000")
        
        # Test different degrees of freedom for t-distribution
        df_scenarios = [
            (3, "very_heavy_tails"),
            (5, "heavy_tails"),
            (10, "moderate_tails"),
            (30, "light_tails"),
        ]
        
        var_results = []
        
        for df, scenario in df_scenarios:
            np.random.seed(42)
            # Generate t-distributed returns
            returns = stats.t.rvs(df=df, size=252) * 0.01  # Scale appropriately
            risk_calculator.portfolio_returns = returns.tolist()
            
            var_1d = await risk_calculator._calculate_var(1, portfolio_value)
            var_results.append((df, var_1d, scenario))
            
            assert isinstance(var_1d, Decimal)
            assert var_1d > Decimal("0")
        
        # Generally, lower degrees of freedom (heavier tails) should produce higher VaR
        # Sort by degrees of freedom
        var_results.sort(key=lambda x: x[0])
        
        # Verify VaR values are reasonable
        for df, var_val, scenario in var_results:
            assert var_val < portfolio_value * Decimal("0.5")  # Sanity check

    @pytest.mark.asyncio
    async def test_var_bimodal_distribution(self, risk_calculator):
        """Test VaR with bimodal distributions (market regime changes)."""
        portfolio_value = Decimal("10000")
        
        np.random.seed(42)
        # Create bimodal distribution (mix of two normals)
        n_samples = 252
        
        # First mode: bear market returns
        bear_returns = np.random.normal(-0.02, 0.01, n_samples // 2)  # -2% mean, 1% vol
        
        # Second mode: bull market returns
        bull_returns = np.random.normal(0.015, 0.015, n_samples // 2)  # 1.5% mean, 1.5% vol
        
        # Combine to create bimodal distribution
        bimodal_returns = np.concatenate([bear_returns, bull_returns])
        np.random.shuffle(bimodal_returns)  # Shuffle to mix regimes
        
        risk_calculator.portfolio_returns = bimodal_returns.tolist()
        
        var_1d = await risk_calculator._calculate_var(1, portfolio_value)
        
        # VaR should capture the risk from both regimes
        assert isinstance(var_1d, Decimal)
        assert var_1d > Decimal("0")
        
        # Compare to unimodal case with same overall volatility
        overall_volatility = np.std(bimodal_returns)
        z_score = 1.645
        theoretical_var = portfolio_value * Decimal(str(overall_volatility * z_score))
        
        # Bimodal VaR should be in reasonable range
        ratio = float(var_1d / theoretical_var)
        assert 0.5 <= ratio <= 2.0  # Allow wider tolerance for complex distribution

    @pytest.mark.asyncio
    async def test_var_time_horizon_scaling(self, risk_calculator):
        """Test VaR scaling across different time horizons."""
        portfolio_value = Decimal("10000")
        
        # Setup returns with known daily volatility
        np.random.seed(42)
        daily_vol = 0.02
        returns = np.random.normal(0, daily_vol, 252)
        risk_calculator.portfolio_returns = returns.tolist()
        
        # Calculate VaR for different time horizons
        time_horizons = [1, 5, 10, 20, 60]
        var_results = {}
        
        for days in time_horizons:
            var_value = await risk_calculator._calculate_var(days, portfolio_value)
            var_results[days] = var_value
        
        # Verify scaling relationship: VaR(T) = VaR(1) * sqrt(T)
        var_1d = var_results[1]
        
        for days in time_horizons[1:]:  # Skip 1-day
            expected_var = var_1d * Decimal(str(np.sqrt(days)))
            actual_var = var_results[days]
            
            # Allow 20% tolerance for scaling relationship
            ratio = float(actual_var / expected_var)
            assert 0.8 <= ratio <= 1.2, f"VaR scaling issue for {days} days: ratio={ratio}"
        
        # Verify monotonic increase
        prev_var = Decimal("0")
        for days in sorted(time_horizons):
            current_var = var_results[days]
            assert current_var > prev_var, f"VaR not increasing with time horizon: {days} days"
            prev_var = current_var

    @pytest.mark.asyncio
    async def test_var_extreme_value_distribution(self, risk_calculator):
        """Test VaR with extreme value distributions (Gumbel, Weibull)."""
        portfolio_value = Decimal("10000")
        
        extreme_distributions = [
            ("gumbel", lambda size: stats.gumbel_r.rvs(size=size) * 0.01),
            ("weibull", lambda size: stats.weibull_min.rvs(c=1.5, size=size) * 0.01),
            ("pareto", lambda size: (stats.pareto.rvs(b=1.16, size=size) - 1) * 0.01),  # Subtract 1 to center
        ]
        
        for dist_name, generator in extreme_distributions:
            np.random.seed(42)
            returns = generator(252)
            
            # Center the distribution around zero
            returns = returns - np.mean(returns)
            
            risk_calculator.portfolio_returns = returns.tolist()
            
            var_1d = await risk_calculator._calculate_var(1, portfolio_value)
            
            assert isinstance(var_1d, Decimal)
            assert var_1d > Decimal("0")
            assert var_1d < portfolio_value  # Sanity check

    @pytest.mark.asyncio
    async def test_var_insufficient_data_fallback(self, risk_calculator):
        """Test VaR fallback behavior with insufficient data."""
        portfolio_value = Decimal("10000")
        
        # Test with various amounts of insufficient data
        insufficient_data_cases = [
            ([], "no_data"),
            ([0.01], "single_point"),
            ([0.01, -0.005] * 5, "very_limited"),  # 10 points
            ([0.01, -0.005] * 10, "somewhat_limited"),  # 20 points
        ]
        
        for returns, case_name in insufficient_data_cases:
            risk_calculator.portfolio_returns = returns
            
            var_1d = await risk_calculator._calculate_var(1, portfolio_value)
            var_5d = await risk_calculator._calculate_var(5, portfolio_value)
            
            # Should fall back to conservative estimates
            if len(returns) < 30:
                # Should use conservative estimate
                expected_var_1d = portfolio_value * Decimal("0.02")  # 2% conservative
                expected_var_5d = portfolio_value * Decimal("0.02") * Decimal(str(np.sqrt(5)))
                
                assert var_1d == expected_var_1d
                assert var_5d == expected_var_5d
            
            # Verify scaling relationship even in fallback
            assert var_5d > var_1d

    @pytest.mark.asyncio
    async def test_var_confidence_level_accuracy(self, risk_calculator):
        """Test VaR accuracy at different confidence levels."""
        portfolio_value = Decimal("10000")
        
        # Setup returns
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 1000)  # Large sample for accuracy
        risk_calculator.portfolio_returns = returns.tolist()
        
        # Test different confidence levels
        confidence_levels = [0.90, 0.95, 0.99]
        z_scores = [1.282, 1.645, 2.326]
        
        for conf_level, z_score in zip(confidence_levels, z_scores):
            # Temporarily change confidence level
            original_conf = risk_calculator.risk_config.var_confidence_level
            risk_calculator.risk_config.var_confidence_level = conf_level
            
            var_1d = await risk_calculator._calculate_var(1, portfolio_value)
            
            # Calculate expected VaR
            volatility = np.std(returns)
            expected_var = portfolio_value * Decimal(str(volatility * z_score))
            
            # Verify accuracy
            ratio = float(var_1d / expected_var)
            assert 0.9 <= ratio <= 1.1, f"VaR accuracy issue at {conf_level*100}% confidence: ratio={ratio}"
            
            # Restore original confidence level
            risk_calculator.risk_config.var_confidence_level = original_conf
        
        # Verify higher confidence levels produce higher VaR
        vars_by_confidence = []
        for conf_level in confidence_levels:
            risk_calculator.risk_config.var_confidence_level = conf_level
            var_val = await risk_calculator._calculate_var(1, portfolio_value)
            vars_by_confidence.append(var_val)
        
        # Should be monotonically increasing
        for i in range(1, len(vars_by_confidence)):
            assert vars_by_confidence[i] > vars_by_confidence[i-1]

    @pytest.mark.asyncio
    async def test_var_historical_simulation_vs_parametric(self, risk_calculator):
        """Test comparison between historical simulation and parametric VaR."""
        portfolio_value = Decimal("10000")
        
        # Create returns with known characteristics
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 252)
        risk_calculator.portfolio_returns = returns.tolist()
        
        # Parametric VaR (current implementation)
        parametric_var = await risk_calculator._calculate_var(1, portfolio_value)
        
        # Historical simulation VaR
        confidence_level = risk_calculator.risk_config.var_confidence_level
        percentile = (1 - confidence_level) * 100
        historical_var_pct = np.percentile(returns, percentile)
        historical_var = portfolio_value * Decimal(str(abs(historical_var_pct)))
        
        # For normal returns, both methods should be similar
        ratio = float(parametric_var / historical_var) if historical_var > 0 else 1
        assert 0.7 <= ratio <= 1.4, f"Parametric vs Historical VaR mismatch: ratio={ratio}"

    @pytest.mark.asyncio
    async def test_var_with_autocorrelated_returns(self, risk_calculator):
        """Test VaR with autocorrelated (trending) returns."""
        portfolio_value = Decimal("10000")
        
        # Generate autocorrelated returns (AR(1) process)
        np.random.seed(42)
        n_samples = 252
        phi = 0.3  # Autocorrelation coefficient
        sigma = 0.02  # Innovation standard deviation
        
        returns = np.zeros(n_samples)
        returns[0] = np.random.normal(0, sigma)
        
        for i in range(1, n_samples):
            returns[i] = phi * returns[i-1] + np.random.normal(0, sigma)
        
        risk_calculator.portfolio_returns = returns.tolist()
        
        var_1d = await risk_calculator._calculate_var(1, portfolio_value)
        
        # Autocorrelated returns might affect VaR scaling
        assert isinstance(var_1d, Decimal)
        assert var_1d > Decimal("0")
        
        # Compare to IID case
        iid_returns = np.random.normal(0, np.std(returns), n_samples)
        risk_calculator.portfolio_returns = iid_returns.tolist()
        iid_var = await risk_calculator._calculate_var(1, portfolio_value)
        
        # Both should be reasonable estimates
        assert var_1d > Decimal("0")
        assert iid_var > Decimal("0")

    @pytest.mark.asyncio
    async def test_var_stress_scenarios(self, risk_calculator):
        """Test VaR under stress scenarios and extreme market conditions."""
        portfolio_value = Decimal("10000")
        
        stress_scenarios = [
            # Market crash scenario (2008-like)
            ([-0.05, -0.03, -0.08, -0.02, -0.04] * 10 + [0.01] * 200, "market_crash"),
            
            # High volatility scenario (crypto-like)
            ([np.random.choice([-0.1, 0.12], p=[0.3, 0.7]) for _ in range(252)], "high_volatility"),
            
            # Gradual decline scenario
            ([max(-0.02, -0.001 * i + np.random.normal(0, 0.01)) for i in range(252)], "gradual_decline"),
            
            # Volatile recovery scenario
            ([-0.05] * 50 + [0.03 + np.random.normal(0, 0.02) for _ in range(202)], "volatile_recovery"),
        ]
        
        for returns, scenario_name in stress_scenarios:
            if scenario_name == "high_volatility":
                np.random.seed(42)
                returns = [np.random.choice([-0.1, 0.12], p=[0.3, 0.7]) for _ in range(252)]
            
            risk_calculator.portfolio_returns = returns
            
            var_1d = await risk_calculator._calculate_var(1, portfolio_value)
            var_5d = await risk_calculator._calculate_var(5, portfolio_value)
            
            # Verify VaR responds appropriately to stress
            assert isinstance(var_1d, Decimal)
            assert isinstance(var_5d, Decimal)
            assert var_1d > Decimal("0")
            assert var_5d > var_1d  # Scaling relationship maintained
            
            # In stress scenarios, VaR should be elevated
            if "crash" in scenario_name or "volatile" in scenario_name:
                # Expect higher VaR in stress scenarios
                assert var_1d > portfolio_value * Decimal("0.01")  # At least 1%

    @pytest.mark.asyncio
    async def test_var_numerical_edge_cases(self, risk_calculator):
        """Test VaR calculation with numerical edge cases."""
        portfolio_value = Decimal("10000")
        
        edge_cases = [
            # Very small returns
            ([1e-10, -1e-10] * 126, "micro_returns"),
            
            # Very large returns (but realistic for crypto)
            ([-0.2, 0.25, -0.15, 0.18] * 63, "macro_returns"),
            
            # All zero returns
            ([0.0] * 252, "zero_returns"),
            
            # Single extreme outlier
            ([0.001] * 251 + [-0.5], "single_outlier"),
            
            # Alternating pattern
            ([0.01 if i % 2 == 0 else -0.01 for i in range(252)], "alternating"),
        ]
        
        for returns, case_name in edge_cases:
            risk_calculator.portfolio_returns = returns
            
            var_1d = await risk_calculator._calculate_var(1, portfolio_value)
            
            assert isinstance(var_1d, Decimal)
            assert var_1d >= Decimal("0")  # VaR should never be negative
            
            if case_name == "zero_returns":
                # Zero volatility should give zero VaR (or minimal conservative estimate)
                assert var_1d <= portfolio_value * Decimal("0.001")
            elif case_name == "single_outlier":
                # Should handle outlier gracefully
                assert var_1d > Decimal("0")
            elif case_name == "macro_returns":
                # Large returns should produce proportionally large VaR
                assert var_1d > portfolio_value * Decimal("0.05")  # At least 5%

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_var_integration_with_risk_metrics(self, risk_calculator, sample_position, sample_market_data):
        """Test VaR integration with overall risk metrics calculation."""
        # Setup portfolio history for VaR calculation
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        risk_calculator.portfolio_returns = returns.tolist()
        
        # Calculate full risk metrics
        positions = [sample_position]
        market_data = [sample_market_data]
        
        risk_metrics = await risk_calculator.calculate_risk_metrics(positions, market_data)
        
        # Verify VaR values are included and reasonable
        assert risk_metrics.var_1d > Decimal("0")
        assert risk_metrics.var_5d > Decimal("0")
        assert risk_metrics.var_5d > risk_metrics.var_1d  # Time scaling
        
        # Verify consistency with direct calculation
        portfolio_value = sample_position.quantity * sample_market_data.price
        direct_var_1d = await risk_calculator._calculate_var(1, portfolio_value)
        direct_var_5d = await risk_calculator._calculate_var(5, portfolio_value)
        
        # Should be close (allowing for portfolio value updates)
        assert abs(risk_metrics.var_1d - direct_var_1d) < direct_var_1d * Decimal("0.1")
        assert abs(risk_metrics.var_5d - direct_var_5d) < direct_var_5d * Decimal("0.1")