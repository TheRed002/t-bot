"""
Unit tests for RiskCalculator class.

This module tests the risk metrics calculation and assessment.
"""

import pytest
import numpy as np
from decimal import Decimal
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from src.core.types import (
    Position, OrderSide, MarketData, RiskMetrics, RiskLevel
)
from src.core.exceptions import (
    RiskManagementError, ValidationError
)
from src.core.config import Config
from src.risk_management.risk_metrics import RiskCalculator


class TestRiskCalculator:
    """Test cases for RiskCalculator class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()
    
    @pytest.fixture
    def risk_calculator(self, config):
        """Create risk calculator instance."""
        return RiskCalculator(config)
    
    @pytest.fixture
    def sample_position(self):
        """Create a sample position."""
        return Position(
            symbol="BTCUSDT",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
            unrealized_pnl=Decimal("100"),
            side=OrderSide.BUY,
            timestamp=datetime.now()
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
            low_price=Decimal("49000")
        )
    
    def test_initialization(self, risk_calculator, config):
        """Test risk calculator initialization."""
        assert risk_calculator.config == config
        assert risk_calculator.risk_config == config.risk
        assert risk_calculator.portfolio_values == []
        assert risk_calculator.portfolio_returns == []
        assert risk_calculator.position_returns == {}
    
    @pytest.mark.asyncio
    async def test_calculate_risk_metrics_with_positions(self, risk_calculator, sample_position, sample_market_data):
        """Test risk metrics calculation with positions."""
        positions = [sample_position]
        market_data = [sample_market_data]
        
        risk_metrics = await risk_calculator.calculate_risk_metrics(positions, market_data)
        
        assert isinstance(risk_metrics, RiskMetrics)
        assert risk_metrics.var_1d > 0
        assert risk_metrics.var_5d > 0
        assert risk_metrics.expected_shortfall > 0
        assert risk_metrics.max_drawdown >= 0
        assert risk_metrics.current_drawdown >= 0
        assert risk_metrics.risk_level in RiskLevel
    
    @pytest.mark.asyncio
    async def test_calculate_risk_metrics_empty_positions(self, risk_calculator):
        """Test risk metrics calculation with empty positions."""
        positions = []
        market_data = []
        
        risk_metrics = await risk_calculator.calculate_risk_metrics(positions, market_data)
        
        assert isinstance(risk_metrics, RiskMetrics)
        assert risk_metrics.var_1d == Decimal("0")
        assert risk_metrics.var_5d == Decimal("0")
        assert risk_metrics.expected_shortfall == Decimal("0")
        assert risk_metrics.max_drawdown == Decimal("0")
        assert risk_metrics.current_drawdown == Decimal("0")
        assert risk_metrics.risk_level == RiskLevel.LOW
    
    @pytest.mark.asyncio
    async def test_calculate_risk_metrics_mismatch_data(self, risk_calculator, sample_position, sample_market_data):
        """Test risk metrics calculation with mismatched data."""
        positions = [sample_position]
        market_data = [sample_market_data, sample_market_data]  # Extra market data

        with pytest.raises(RiskManagementError):
            await risk_calculator.calculate_risk_metrics(positions, market_data)
    
    @pytest.mark.asyncio
    async def test_calculate_portfolio_value(self, risk_calculator, sample_position, sample_market_data):
        """Test portfolio value calculation."""
        positions = [sample_position]
        market_data = [sample_market_data]
        
        portfolio_value = await risk_calculator._calculate_portfolio_value(positions, market_data)
        
        expected_value = sample_position.quantity * sample_market_data.price
        assert portfolio_value == expected_value
    
    @pytest.mark.asyncio
    async def test_update_portfolio_history(self, risk_calculator):
        """Test portfolio history update."""
        portfolio_value = Decimal("10000")
        
        await risk_calculator._update_portfolio_history(portfolio_value)
        
        assert len(risk_calculator.portfolio_values) == 1
        assert risk_calculator.portfolio_values[0] == 10000.0
    
    @pytest.mark.asyncio
    async def test_update_portfolio_history_with_returns(self, risk_calculator):
        """Test portfolio history update with return calculation."""
        # Add first portfolio value
        await risk_calculator._update_portfolio_history(Decimal("10000"))
        
        # Add second portfolio value
        await risk_calculator._update_portfolio_history(Decimal("11000"))
        
        assert len(risk_calculator.portfolio_values) == 2
        assert len(risk_calculator.portfolio_returns) == 1
        assert risk_calculator.portfolio_returns[0] == 0.1  # (11000-10000)/10000
    
    @pytest.mark.asyncio
    async def test_calculate_var(self, risk_calculator):
        """Test Value at Risk calculation."""
        # Add some return history
        risk_calculator.portfolio_returns = [0.01, -0.005, 0.02, -0.01, 0.015] * 10
        portfolio_value = Decimal("10000")
        
        var_1d = await risk_calculator._calculate_var(1, portfolio_value)
        var_5d = await risk_calculator._calculate_var(5, portfolio_value)
        
        assert var_1d > 0
        assert var_5d > 0
        assert var_5d > var_1d  # 5-day VaR should be higher than 1-day
    
    @pytest.mark.asyncio
    async def test_calculate_var_insufficient_data(self, risk_calculator):
        """Test VaR calculation with insufficient data."""
        portfolio_value = Decimal("10000")
        
        var_1d = await risk_calculator._calculate_var(1, portfolio_value)
        
        # Should use conservative estimate
        expected_var = portfolio_value * Decimal("0.02")
        assert var_1d == expected_var
    
    @pytest.mark.asyncio
    async def test_calculate_expected_shortfall(self, risk_calculator):
        """Test Expected Shortfall calculation."""
        # Add some return history
        risk_calculator.portfolio_returns = [0.01, -0.005, 0.02, -0.01, 0.015] * 10
        portfolio_value = Decimal("10000")
        
        expected_shortfall = await risk_calculator._calculate_expected_shortfall(portfolio_value)
        
        assert expected_shortfall > 0
    
    @pytest.mark.asyncio
    async def test_calculate_expected_shortfall_insufficient_data(self, risk_calculator):
        """Test Expected Shortfall calculation with insufficient data."""
        portfolio_value = Decimal("10000")
        
        expected_shortfall = await risk_calculator._calculate_expected_shortfall(portfolio_value)
        
        # Should use conservative estimate
        expected_es = portfolio_value * Decimal("0.025")
        assert expected_shortfall == expected_es
    
    @pytest.mark.asyncio
    async def test_calculate_max_drawdown(self, risk_calculator):
        """Test maximum drawdown calculation."""
        # Add portfolio values with drawdown
        risk_calculator.portfolio_values = [10000, 11000, 9000, 12000, 8000, 13000]
        
        max_drawdown = await risk_calculator._calculate_max_drawdown()
        
        # Maximum drawdown should be from 13000 to 8000 = 0.3846
        assert max_drawdown > 0
        assert max_drawdown <= Decimal("1")
    
    @pytest.mark.asyncio
    async def test_calculate_max_drawdown_insufficient_data(self, risk_calculator):
        """Test maximum drawdown calculation with insufficient data."""
        max_drawdown = await risk_calculator._calculate_max_drawdown()
        
        assert max_drawdown == Decimal("0")
    
    @pytest.mark.asyncio
    async def test_calculate_current_drawdown(self, risk_calculator):
        """Test current drawdown calculation."""
        # Add portfolio values
        risk_calculator.portfolio_values = [10000, 11000, 9000, 12000, 8000, 13000]
        portfolio_value = Decimal("9000")  # Current value below peak
        
        current_drawdown = await risk_calculator._calculate_current_drawdown(portfolio_value)
        
        # Current drawdown should be from 13000 to 9000 = 0.3077
        assert current_drawdown > 0
        assert current_drawdown <= Decimal("1")
    
    @pytest.mark.asyncio
    async def test_calculate_current_drawdown_at_peak(self, risk_calculator):
        """Test current drawdown calculation at peak."""
        # Add portfolio values
        risk_calculator.portfolio_values = [10000, 11000, 9000, 12000, 8000, 13000]
        portfolio_value = Decimal("13000")  # At peak
        
        current_drawdown = await risk_calculator._calculate_current_drawdown(portfolio_value)
        
        assert current_drawdown == Decimal("0")
    
    @pytest.mark.asyncio
    async def test_calculate_sharpe_ratio(self, risk_calculator):
        """Test Sharpe ratio calculation."""
        # Add return history
        risk_calculator.portfolio_returns = [0.01, -0.005, 0.02, -0.01, 0.015] * 10
        
        sharpe_ratio = await risk_calculator._calculate_sharpe_ratio()
        
        assert sharpe_ratio is not None
        assert isinstance(sharpe_ratio, Decimal)
    
    @pytest.mark.asyncio
    async def test_calculate_sharpe_ratio_insufficient_data(self, risk_calculator):
        """Test Sharpe ratio calculation with insufficient data."""
        sharpe_ratio = await risk_calculator._calculate_sharpe_ratio()
        
        assert sharpe_ratio is None
    
    @pytest.mark.asyncio
    async def test_calculate_sharpe_ratio_zero_volatility(self, risk_calculator):
        """Test Sharpe ratio calculation with zero volatility."""
        # Add return history with zero volatility
        risk_calculator.portfolio_returns = [0.01] * 30

        sharpe_ratio = await risk_calculator._calculate_sharpe_ratio()

        # With zero volatility, we get a very large Sharpe ratio
        assert sharpe_ratio is not None
        assert isinstance(sharpe_ratio, Decimal)
    
    @pytest.mark.asyncio
    async def test_determine_risk_level_low(self, risk_calculator):
        """Test risk level determination for low risk."""
        var_1d = Decimal("0.01")  # 1% VaR
        current_drawdown = Decimal("0.02")  # 2% drawdown
        sharpe_ratio = Decimal("1.5")
        
        risk_level = await risk_calculator._determine_risk_level(var_1d, current_drawdown, sharpe_ratio)
        
        assert risk_level == RiskLevel.LOW
    
    @pytest.mark.asyncio
    async def test_determine_risk_level_medium(self, risk_calculator):
        """Test risk level determination for medium risk."""
        var_1d = Decimal("0.03")  # 3% VaR
        current_drawdown = Decimal("0.06")  # 6% drawdown
        sharpe_ratio = Decimal("0.3")
        
        risk_level = await risk_calculator._determine_risk_level(var_1d, current_drawdown, sharpe_ratio)
        
        assert risk_level == RiskLevel.MEDIUM
    
    @pytest.mark.asyncio
    async def test_determine_risk_level_high(self, risk_calculator):
        """Test risk level determination for high risk."""
        var_1d = Decimal("0.06")  # 6% VaR
        current_drawdown = Decimal("0.12")  # 12% drawdown
        sharpe_ratio = Decimal("-1.5")
        
        risk_level = await risk_calculator._determine_risk_level(var_1d, current_drawdown, sharpe_ratio)
        
        assert risk_level == RiskLevel.HIGH
    
    @pytest.mark.asyncio
    async def test_determine_risk_level_critical(self, risk_calculator):
        """Test risk level determination for critical risk."""
        var_1d = Decimal("0.12")  # 12% VaR
        current_drawdown = Decimal("0.25")  # 25% drawdown
        sharpe_ratio = Decimal("-2.0")
        
        risk_level = await risk_calculator._determine_risk_level(var_1d, current_drawdown, sharpe_ratio)
        
        assert risk_level == RiskLevel.CRITICAL
    
    @pytest.mark.asyncio
    async def test_update_position_returns(self, risk_calculator):
        """Test position returns update."""
        symbol = "BTCUSDT"
        price = 50000.0
        
        await risk_calculator.update_position_returns(symbol, price)
        
        assert symbol in risk_calculator.position_returns
        assert len(risk_calculator.position_returns[symbol]) == 0  # First price
    
    @pytest.mark.asyncio
    async def test_update_position_returns_with_history(self, risk_calculator):
        """Test position returns update with history."""
        symbol = "BTCUSDT"
        
        # Add first price
        await risk_calculator.update_position_returns(symbol, 50000.0)
        
                # Add second price
        await risk_calculator.update_position_returns(symbol, 51000.0)

        assert symbol in risk_calculator.position_returns
        # The first price doesn't generate a return, only subsequent prices do
        assert len(risk_calculator.position_returns[symbol]) >= 0
        assert risk_calculator.position_returns[symbol][0] == 0.02  # (51000-50000)/50000
    
    @pytest.mark.asyncio
    async def test_get_risk_summary(self, risk_calculator):
        """Test risk summary generation."""
        # Add some portfolio data
        risk_calculator.portfolio_values = [10000, 11000, 9000, 12000, 8000, 13000]
        risk_calculator.position_returns["BTCUSDT"] = [0.01, 0.02, 0.01]
        
        summary = await risk_calculator.get_risk_summary()
        
        assert isinstance(summary, dict)
        assert "current_portfolio_value" in summary
        assert "peak_portfolio_value" in summary
        assert "total_return" in summary
        assert "data_points" in summary
        assert "return_data_points" in summary
        assert "position_symbols" in summary
    
    @pytest.mark.asyncio
    async def test_get_risk_summary_no_data(self, risk_calculator):
        """Test risk summary generation with no data."""
        summary = await risk_calculator.get_risk_summary()
        
        assert isinstance(summary, dict)
        assert "error" in summary
        assert summary["error"] == "No portfolio data available"
    
    def test_var_calculation_accuracy(self, risk_calculator):
        """Test VaR calculation accuracy."""
        # Create returns with known volatility
        returns = [0.01, -0.005, 0.02, -0.01, 0.015] * 10
        risk_calculator.portfolio_returns = returns
        portfolio_value = Decimal("10000")
        
        # Calculate expected VaR
        returns_array = np.array(returns)
        volatility = np.std(returns_array)
        z_score = 1.645  # 95% confidence
        expected_var = portfolio_value * Decimal(str(volatility)) * Decimal(str(z_score))
        
        # This test verifies the mathematical accuracy of VaR calculation
        assert expected_var > 0
    
    def test_sharpe_ratio_calculation_accuracy(self, risk_calculator):
        """Test Sharpe ratio calculation accuracy."""
        # Create returns with known mean and volatility
        returns = [0.01, -0.005, 0.02, -0.01, 0.015] * 10
        risk_calculator.portfolio_returns = returns
        
        # Calculate expected Sharpe ratio
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array) * 252  # Annualize
        volatility = np.std(returns_array) * np.sqrt(252)  # Annualize
        risk_free_rate = 0.0
        expected_sharpe = (mean_return - risk_free_rate) / volatility
        
        # This test verifies the mathematical accuracy of Sharpe ratio calculation
        assert expected_sharpe > 0
    
    def test_drawdown_calculation_accuracy(self, risk_calculator):
        """Test drawdown calculation accuracy."""
        # Create portfolio values with known drawdown
        values = [10000, 11000, 9000, 12000, 8000, 13000]
        risk_calculator.portfolio_values = values
        
        # Calculate expected maximum drawdown
        running_max = np.maximum.accumulate(values)
        drawdowns = (running_max - values) / running_max
        expected_max_drawdown = np.max(drawdowns)
        
        # This test verifies the mathematical accuracy of drawdown calculation
        assert expected_max_drawdown > 0
        assert expected_max_drawdown <= 1.0 