"""
Comprehensive tests for optimization analysis module.

This module focuses on achieving high coverage for the analysis.py module
by testing the currently uncovered functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from datetime import datetime
import numpy as np

from src.optimization.analysis import (
    PerformanceMetrics,
    SensitivityAnalysis,
    StabilityAnalysis,
    ParameterImportanceAnalyzer,
    PerformanceAnalyzer,
    ResultsAnalyzer,
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics class methods."""

    @pytest.fixture
    def sample_metrics(self):
        """Create sample performance metrics."""
        return PerformanceMetrics(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            total_return=Decimal("0.25"),
            annualized_return=Decimal("0.15"),
            excess_return=Decimal("0.10"),
            volatility=Decimal("0.12"),
            downside_volatility=Decimal("0.08"),
            max_drawdown=Decimal("0.05"),
            current_drawdown=Decimal("0.02"),
            sharpe_ratio=Decimal("1.5"),
            sortino_ratio=Decimal("2.0"),
            calmar_ratio=Decimal("3.0"),
            omega_ratio=Decimal("1.8"),
            win_rate=Decimal("0.65"),
            profit_factor=Decimal("1.4"),
            average_win=Decimal("0.02"),
            average_loss=Decimal("-0.015"),
            largest_win=Decimal("0.08"),
            largest_loss=Decimal("-0.04"),
            value_at_risk_95=Decimal("0.02"),
            conditional_var_95=Decimal("0.03"),
            skewness=Decimal("0.1"),
            kurtosis=Decimal("3.5"),
            recovery_factor=Decimal("5.0"),
            stability_ratio=Decimal("0.85"),
            consistency_score=Decimal("0.75"),
            total_fees=Decimal("0.005"),
            fee_adjusted_return=Decimal("0.145"),
            turnover_ratio=Decimal("2.5"),
            periods_analyzed=365
        )

    def test_get_risk_score(self, sample_metrics):
        """Test risk score calculation."""
        risk_score = sample_metrics.get_risk_score()

        assert isinstance(risk_score, Decimal)
        assert Decimal("0") <= risk_score <= Decimal("1")

    def test_get_quality_score(self, sample_metrics):
        """Test quality score calculation."""
        quality_score = sample_metrics.get_quality_score()

        assert isinstance(quality_score, Decimal)
        assert Decimal("0") <= quality_score <= Decimal("1")

    def test_get_risk_score_edge_cases(self):
        """Test risk score with edge case values."""
        # High risk metrics
        high_risk_metrics = PerformanceMetrics(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            total_return=Decimal("-0.5"),
            annualized_return=Decimal("-0.3"),
            excess_return=Decimal("-0.35"),
            volatility=Decimal("0.8"),
            downside_volatility=Decimal("0.6"),
            max_drawdown=Decimal("0.4"),
            current_drawdown=Decimal("0.3"),
            sharpe_ratio=Decimal("-2.0"),
            sortino_ratio=Decimal("-1.5"),
            calmar_ratio=Decimal("-0.75"),
            omega_ratio=Decimal("0.5"),
            win_rate=Decimal("0.25"),
            profit_factor=Decimal("0.5"),
            average_win=Decimal("0.01"),
            average_loss=Decimal("-0.05"),
            largest_win=Decimal("0.02"),
            largest_loss=Decimal("-0.2"),
            value_at_risk_95=Decimal("0.15"),
            conditional_var_95=Decimal("0.2"),
            skewness=Decimal("-2.0"),
            kurtosis=Decimal("10.0"),
            recovery_factor=Decimal("0.5"),
            stability_ratio=Decimal("0.2"),
            consistency_score=Decimal("0.1"),
            total_fees=Decimal("0.02"),
            fee_adjusted_return=Decimal("-0.32"),
            turnover_ratio=Decimal("10.0"),
            periods_analyzed=365
        )

        risk_score = high_risk_metrics.get_risk_score()
        assert isinstance(risk_score, Decimal)
        assert risk_score > Decimal("0.5")  # Should indicate high risk (score is 0-1)

    def test_get_quality_score_edge_cases(self):
        """Test quality score with edge case values."""
        # Low quality metrics
        low_quality_metrics = PerformanceMetrics(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            total_return=Decimal("0.01"),
            annualized_return=Decimal("0.005"),
            excess_return=Decimal("0.001"),
            volatility=Decimal("0.25"),
            downside_volatility=Decimal("0.2"),
            max_drawdown=Decimal("0.15"),
            current_drawdown=Decimal("0.08"),
            sharpe_ratio=Decimal("0.2"),
            sortino_ratio=Decimal("0.3"),
            calmar_ratio=Decimal("0.1"),
            omega_ratio=Decimal("1.1"),
            win_rate=Decimal("0.45"),
            profit_factor=Decimal("1.05"),
            average_win=Decimal("0.008"),
            average_loss=Decimal("-0.009"),
            largest_win=Decimal("0.03"),
            largest_loss=Decimal("-0.06"),
            value_at_risk_95=Decimal("0.05"),
            conditional_var_95=Decimal("0.08"),
            skewness=Decimal("-1.0"),
            kurtosis=Decimal("6.0"),
            recovery_factor=Decimal("1.2"),
            stability_ratio=Decimal("0.4"),
            consistency_score=Decimal("0.3"),
            total_fees=Decimal("0.01"),
            fee_adjusted_return=Decimal("-0.005"),
            turnover_ratio=Decimal("5.0"),
            periods_analyzed=365
        )

        quality_score = low_quality_metrics.get_quality_score()
        assert isinstance(quality_score, Decimal)
        assert quality_score < Decimal("0.5")  # Should indicate low quality (score is 0-1)


class TestStabilityAnalysis:
    """Test StabilityAnalysis class methods."""

    @pytest.fixture
    def sample_stability(self):
        """Create sample stability analysis."""
        return StabilityAnalysis(
            period_consistency=Decimal("0.85"),
            regime_consistency=Decimal("0.78"),
            parameter_stability={"param1": Decimal("0.9"), "param2": Decimal("0.7")},
            performance_stability=Decimal("0.82"),
            bull_market_performance=Decimal("0.15"),
            bear_market_performance=Decimal("0.08"),
            sideways_market_performance=Decimal("0.12"),
            low_vol_performance=Decimal("0.14"),
            high_vol_performance=Decimal("0.09"),
            worst_period_performance=Decimal("0.03"),
            crisis_resilience=Decimal("0.6")
        )

    def test_get_overall_stability_score(self, sample_stability):
        """Test overall stability score calculation."""
        overall_score = sample_stability.get_overall_stability_score()

        assert isinstance(overall_score, Decimal)
        assert Decimal("0") <= overall_score <= Decimal("1")

    def test_get_overall_stability_score_edge_cases(self):
        """Test stability score with edge case values."""
        # Low stability scenario
        low_stability = StabilityAnalysis(
            period_consistency=Decimal("0.2"),
            regime_consistency=Decimal("0.15"),
            parameter_stability={"param1": Decimal("0.3"), "param2": Decimal("0.25")},
            performance_stability=Decimal("0.18"),
            bull_market_performance=Decimal("0.02"),
            bear_market_performance=Decimal("-0.05"),
            sideways_market_performance=Decimal("0.01"),
            worst_period_performance=Decimal("-0.15"),
            crisis_resilience=Decimal("0.1")
        )

        overall_score = low_stability.get_overall_stability_score()
        assert isinstance(overall_score, Decimal)
        assert overall_score < Decimal("0.5")  # Should indicate low stability


class TestParameterImportanceAnalyzer:
    """Test ParameterImportanceAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create ParameterImportanceAnalyzer instance."""
        return ParameterImportanceAnalyzer()

    @pytest.fixture
    def sample_optimization_results(self):
        """Create sample optimization results."""
        return [
            {
                "parameters": {"param1": 0.1, "param2": 0.2, "param3": 0.3},
                "objective_value": 1.5,
                "objective_values": {"profit": 100.0, "sharpe": 1.5},
                "performance": {"return": 0.15, "volatility": 0.12}
            },
            {
                "parameters": {"param1": 0.2, "param2": 0.3, "param3": 0.1},
                "objective_value": 1.8,
                "objective_values": {"profit": 120.0, "sharpe": 1.8},
                "performance": {"return": 0.18, "volatility": 0.14}
            },
            {
                "parameters": {"param1": 0.3, "param2": 0.1, "param3": 0.2},
                "objective_value": 1.2,
                "objective_values": {"profit": 80.0, "sharpe": 1.2},
                "performance": {"return": 0.12, "volatility": 0.10}
            },
            {
                "parameters": {"param1": 0.15, "param2": 0.25, "param3": 0.35},
                "objective_value": 1.6,
                "objective_values": {"profit": 110.0, "sharpe": 1.6},
                "performance": {"return": 0.16, "volatility": 0.13}
            },
            {
                "parameters": {"param1": 0.25, "param2": 0.15, "param3": 0.05},
                "objective_value": 1.3,
                "objective_values": {"profit": 90.0, "sharpe": 1.3},
                "performance": {"return": 0.13, "volatility": 0.11}
            },
            {
                "parameters": {"param1": 0.35, "param2": 0.05, "param3": 0.15},
                "objective_value": 1.1,
                "objective_values": {"profit": 85.0, "sharpe": 1.1},
                "performance": {"return": 0.11, "volatility": 0.09}
            },
            {
                "parameters": {"param1": 0.05, "param2": 0.35, "param3": 0.25},
                "objective_value": 1.7,
                "objective_values": {"profit": 125.0, "sharpe": 1.7},
                "performance": {"return": 0.17, "volatility": 0.15}
            }
        ]

    def test_analyze_parameter_importance_success(self, analyzer, sample_optimization_results):
        """Test successful parameter importance analysis."""
        parameter_names = ["param1", "param2", "param3"]

        results = analyzer.analyze_parameter_importance(
            optimization_results=sample_optimization_results,
            parameter_names=parameter_names
        )

        assert isinstance(results, list)
        assert len(results) == len(parameter_names)

        for result in results:
            assert isinstance(result, SensitivityAnalysis)

    def test_analyze_parameter_importance_empty_results(self, analyzer):
        """Test parameter importance analysis with empty results."""
        results = analyzer.analyze_parameter_importance(
            optimization_results=[],
            parameter_names=["param1"]
        )

        assert isinstance(results, list)
        assert len(results) == 0

    def test_analyze_parameter_importance_single_result(self, analyzer):
        """Test parameter importance with single optimization result."""
        single_result = [{
            "parameters": {"param1": 0.1},
            "objective_values": {"profit": 100.0},
            "performance": {"return": 0.15}
        }]

        results = analyzer.analyze_parameter_importance(
            optimization_results=single_result,
            parameter_names=["param1"]
        )

        assert isinstance(results, list)
        # Should handle single result gracefully (may return empty or single analysis)

    def test_extract_parameter_data(self, analyzer, sample_optimization_results):
        """Test parameter data extraction."""
        parameter_names = ["param1", "param2"]

        param_data = analyzer._extract_parameter_data(
            optimization_results=sample_optimization_results,
            parameter_names=parameter_names
        )

        assert isinstance(param_data, dict)
        assert len(param_data) == len(parameter_names)

        for param_name in parameter_names:
            assert param_name in param_data
            assert isinstance(param_data[param_name], list)
            assert len(param_data[param_name]) == len(sample_optimization_results)

    def test_extract_performance_data(self, analyzer, sample_optimization_results):
        """Test performance data extraction."""
        performance_data = analyzer._extract_performance_data(
            optimization_results=sample_optimization_results
        )

        assert isinstance(performance_data, list)
        assert len(performance_data) == len(sample_optimization_results)

        for value in performance_data:
            assert isinstance(value, Decimal)

    def test_analyze_single_parameter(self, analyzer, sample_optimization_results):
        """Test single parameter analysis."""
        param_values = [0.1, 0.2, 0.3, 0.15, 0.25]
        performance_values = [100.0, 120.0, 80.0, 110.0, 90.0]

        analysis = analyzer._analyze_single_parameter(
            param_name="test_param",
            param_values=param_values,
            performance_values=performance_values,
            all_parameter_data={"test_param": param_values}
        )

        # Should return SensitivityAnalysis or None
        if analysis is not None:
            assert isinstance(analysis, SensitivityAnalysis)

    def test_calculate_parameter_stability(self, analyzer):
        """Test parameter stability calculation."""
        param_values = [0.1, 0.2, 0.3, 0.15, 0.25]
        performance_values = [100.0, 120.0, 80.0, 110.0, 90.0]

        stability = analyzer._calculate_parameter_stability(
            param_values=param_values,
            performance_values=performance_values
        )

        assert isinstance(stability, Decimal)
        assert 0.0 <= stability <= 1.0

    def test_find_interaction_partners(self, analyzer):
        """Test interaction partner finding."""
        all_param_data = {
            "param1": [0.1, 0.2, 0.3],
            "param2": [0.2, 0.3, 0.1],
            "param3": [0.3, 0.1, 0.2]
        }
        performance_values = [100.0, 120.0, 80.0]

        partners, interactions = analyzer._find_interaction_partners(
            param_name="param1",
            all_parameter_data=all_param_data,
            performance_values=performance_values
        )

        assert isinstance(partners, list)
        assert isinstance(interactions, dict)


class TestPerformanceAnalyzer:
    """Test PerformanceAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create PerformanceAnalyzer instance."""
        return PerformanceAnalyzer(risk_free_rate=Decimal("0.02"))

    @pytest.fixture
    def sample_returns(self):
        """Create sample return data."""
        return [0.01, -0.005, 0.02, 0.015, -0.01, 0.008, -0.003, 0.012, 0.005, -0.008]

    @pytest.fixture
    def sample_trades(self):
        """Create sample trade data."""
        return [
            {"return": 0.02, "pnl": 200.0, "side": "long", "size": 1000.0},
            {"return": -0.01, "pnl": -100.0, "side": "short", "size": 1000.0},
            {"return": 0.015, "pnl": 150.0, "side": "long", "size": 1000.0},
            {"return": -0.008, "pnl": -80.0, "side": "short", "size": 1000.0},
            {"return": 0.025, "pnl": 250.0, "side": "long", "size": 1000.0}
        ]

    def test_calculate_performance_metrics_success(
        self, analyzer, sample_returns, sample_trades
    ):
        """Test successful performance metrics calculation."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        metrics = analyzer.calculate_performance_metrics(
            returns=sample_returns,
            start_date=start_date,
            end_date=end_date,
            trades=sample_trades,
            initial_capital=Decimal("10000")
        )

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.start_date == start_date
        assert metrics.end_date == end_date

    def test_calculate_performance_metrics_empty_data(self, analyzer):
        """Test performance metrics with empty data."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        metrics = analyzer.calculate_performance_metrics(
            returns=[],
            start_date=start_date,
            end_date=end_date,
            trades=[],
            initial_capital=Decimal("10000")
        )

        assert isinstance(metrics, PerformanceMetrics)

    def test_create_empty_metrics(self, analyzer):
        """Test empty metrics creation."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        metrics = analyzer._create_empty_metrics(start_date, end_date)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.start_date == start_date
        assert metrics.end_date == end_date
        assert metrics.total_return == Decimal("0")

    def test_calculate_total_return(self, analyzer, sample_returns):
        """Test total return calculation."""
        total_return = analyzer._calculate_total_return(sample_returns)

        assert isinstance(total_return, Decimal)

    def test_calculate_annualized_return(self, analyzer):
        """Test annualized return calculation."""
        total_return = 0.25  # 25% total return
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        annualized_return = analyzer._calculate_annualized_return(
            total_return, start_date, end_date
        )

        assert isinstance(annualized_return, Decimal)

    def test_calculate_volatility(self, analyzer, sample_returns):
        """Test volatility calculation."""
        volatility = analyzer._calculate_volatility(sample_returns)

        assert isinstance(volatility, Decimal)
        assert volatility >= 0.0

    def test_calculate_downside_volatility(self, analyzer, sample_returns):
        """Test downside volatility calculation."""
        downside_vol = analyzer._calculate_downside_volatility(sample_returns)

        assert isinstance(downside_vol, Decimal)
        assert downside_vol >= 0.0

    def test_calculate_drawdowns(self, analyzer, sample_returns):
        """Test drawdown calculations."""
        max_dd, duration = analyzer._calculate_drawdowns(sample_returns)

        assert isinstance(max_dd, Decimal)
        assert isinstance(duration, Decimal)
        assert max_dd >= 0.0  # Drawdown should be positive or zero

    def test_calculate_sharpe_ratio(self, analyzer):
        """Test Sharpe ratio calculation."""
        sharpe = analyzer._calculate_sharpe_ratio(0.15, 0.12)

        assert isinstance(sharpe, Decimal)

    def test_calculate_sortino_ratio(self, analyzer):
        """Test Sortino ratio calculation."""
        sortino = analyzer._calculate_sortino_ratio(0.15, 0.08)

        assert isinstance(sortino, Decimal)

    def test_calculate_calmar_ratio(self, analyzer):
        """Test Calmar ratio calculation."""
        calmar = analyzer._calculate_calmar_ratio(0.15, -0.05)

        assert isinstance(calmar, Decimal)

    def test_calculate_omega_ratio(self, analyzer, sample_returns):
        """Test Omega ratio calculation."""
        omega = analyzer._calculate_omega_ratio(sample_returns)

        assert isinstance(omega, Decimal)
        assert omega >= 0.0

    def test_calculate_trade_metrics(self, analyzer, sample_trades):
        """Test trade metrics calculation."""
        metrics = analyzer._calculate_trade_metrics(sample_trades)

        assert isinstance(metrics, dict)
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "average_win" in metrics
        assert "average_loss" in metrics

    def test_calculate_var(self, analyzer, sample_returns):
        """Test VaR calculation."""
        var = analyzer._calculate_var(sample_returns, confidence=0.95)

        assert isinstance(var, Decimal)

    def test_calculate_conditional_var(self, analyzer, sample_returns):
        """Test Conditional VaR calculation."""
        cvar = analyzer._calculate_conditional_var(sample_returns, confidence=0.95)

        assert isinstance(cvar, Decimal)

    def test_calculate_skewness(self, analyzer, sample_returns):
        """Test skewness calculation."""
        skewness = analyzer._calculate_skewness(sample_returns)

        assert isinstance(skewness, Decimal)

    def test_calculate_kurtosis(self, analyzer, sample_returns):
        """Test kurtosis calculation."""
        kurtosis = analyzer._calculate_kurtosis(sample_returns)

        assert isinstance(kurtosis, Decimal)

    def test_calculate_recovery_factor(self, analyzer):
        """Test recovery factor calculation."""
        recovery = analyzer._calculate_recovery_factor(0.25, 0.05)

        assert isinstance(recovery, Decimal)
        assert recovery >= 0.0

    def test_calculate_stability_ratio(self, analyzer, sample_returns):
        """Test stability ratio calculation."""
        stability = analyzer._calculate_stability_ratio(sample_returns)

        assert isinstance(stability, Decimal)
        assert 0.0 <= stability <= 1.0

    def test_calculate_consistency_score(self, analyzer, sample_returns):
        """Test consistency score calculation."""
        consistency = analyzer._calculate_consistency_score(sample_returns)

        assert isinstance(consistency, Decimal)
        assert 0.0 <= consistency <= 1.0

    def test_calculate_turnover_ratio(self, analyzer, sample_trades):
        """Test turnover ratio calculation."""
        turnover = analyzer._calculate_turnover_ratio(sample_trades, Decimal("10000"))

        assert isinstance(turnover, Decimal)
        assert turnover >= Decimal("0")


class TestResultsAnalyzer:
    """Test ResultsAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create ResultsAnalyzer instance."""
        return ResultsAnalyzer(risk_free_rate=Decimal("0.02"))

    @pytest.fixture
    def sample_optimization_results(self):
        """Create comprehensive sample optimization results."""
        return [
            {
                "parameters": {"param1": 0.1, "param2": 0.2},
                "objective_values": {"profit": 100.0, "sharpe": 1.5},
                "performance": {"return": 0.15, "volatility": 0.12, "drawdown": -0.03},
                "trades": [{"return": 0.02, "pnl": 200.0}],
                "returns": [0.01, 0.02, -0.005]
            },
            {
                "parameters": {"param1": 0.2, "param2": 0.3},
                "objective_values": {"profit": 120.0, "sharpe": 1.8},
                "performance": {"return": 0.18, "volatility": 0.14, "drawdown": -0.04},
                "trades": [{"return": 0.025, "pnl": 250.0}],
                "returns": [0.015, 0.025, -0.008]
            },
            {
                "parameters": {"param1": 0.3, "param2": 0.1},
                "objective_values": {"profit": 80.0, "sharpe": 1.2},
                "performance": {"return": 0.12, "volatility": 0.10, "drawdown": -0.02},
                "trades": [{"return": 0.015, "pnl": 150.0}],
                "returns": [0.008, 0.015, -0.003]
            }
        ]

    def test_analyze_optimization_results_success(
        self, analyzer, sample_optimization_results
    ):
        """Test successful optimization results analysis."""
        parameter_names = ["param1", "param2"]

        results = analyzer.analyze_optimization_results(
            optimization_results=sample_optimization_results,
            parameter_names=parameter_names,
            best_result=sample_optimization_results[1]
        )

        assert isinstance(results, dict)
        assert "performance_distribution" in results
        assert "parameter_correlations" in results
        assert "optimization_landscape" in results
        assert "best_result_analysis" in results
        assert "convergence_analysis" in results
        assert "summary" in results

    def test_analyze_optimization_results_minimal(self, analyzer):
        """Test analysis with minimal data."""
        minimal_results = [{
            "parameters": {"param1": 0.1},
            "objective_values": {"profit": 100.0}
        }]

        results = analyzer.analyze_optimization_results(
            optimization_results=minimal_results,
            parameter_names=["param1"]
        )

        assert isinstance(results, dict)

    def test_analyze_performance_distribution(self, analyzer, sample_optimization_results):
        """Test performance distribution analysis."""
        distribution = analyzer._analyze_performance_distribution(sample_optimization_results)

        assert isinstance(distribution, dict)
        assert "mean" in distribution
        assert "std" in distribution
        assert "min" in distribution
        assert "max" in distribution

    def test_calculate_parameter_correlations(
        self, analyzer, sample_optimization_results
    ):
        """Test parameter correlation calculation."""
        parameter_names = ["param1", "param2"]

        correlations = analyzer._calculate_parameter_correlations(
            sample_optimization_results, parameter_names
        )

        assert isinstance(correlations, dict)

    def test_analyze_optimization_landscape(self, analyzer, sample_optimization_results):
        """Test optimization landscape analysis."""
        landscape = analyzer._analyze_optimization_landscape(sample_optimization_results)

        assert isinstance(landscape, dict)
        assert "ruggedness" in landscape
        assert "multimodality" in landscape

    def test_calculate_landscape_ruggedness(self, analyzer):
        """Test landscape ruggedness calculation."""
        performance_values = [100.0, 120.0, 80.0, 110.0, 90.0]

        ruggedness = analyzer._calculate_landscape_ruggedness(performance_values)

        assert isinstance(ruggedness, float)
        assert ruggedness >= 0.0

    def test_detect_multimodality(self, analyzer):
        """Test multimodality detection."""
        performance_values = [100.0, 120.0, 80.0, 110.0, 90.0, 115.0]
        sorted_performance = sorted(performance_values, reverse=True)

        multimodality = analyzer._detect_multimodality(sorted_performance)

        assert isinstance(multimodality, dict)
        assert "has_multiple_modes" in multimodality

    def test_calculate_convergence_rate(self, analyzer):
        """Test convergence rate calculation."""
        sorted_performance = [120.0, 115.0, 110.0, 100.0, 90.0, 80.0]

        convergence_rate = analyzer._calculate_convergence_rate(sorted_performance)

        assert isinstance(convergence_rate, float)

    def test_detect_performance_plateaus(self, analyzer):
        """Test performance plateau detection."""
        sorted_performance = [120.0, 119.0, 118.0, 115.0, 100.0, 80.0]

        plateaus = analyzer._detect_performance_plateaus(sorted_performance)

        assert isinstance(plateaus, dict)

    def test_assess_improvement_potential(self, analyzer):
        """Test improvement potential assessment."""
        sorted_performance = [120.0, 115.0, 110.0, 100.0, 90.0]

        potential = analyzer._assess_improvement_potential(sorted_performance)

        assert isinstance(potential, dict)
        assert "improvement_potential" in potential

    def test_analyze_best_result(self, analyzer, sample_optimization_results):
        """Test best result analysis."""
        best_analysis = analyzer._analyze_best_result(sample_optimization_results[1])

        assert isinstance(best_analysis, dict)

    def test_categorize_parameter_types(self, analyzer):
        """Test parameter type categorization."""
        parameters = {"continuous": 0.1, "discrete": 5, "categorical": "option_a"}

        categorization = analyzer._categorize_parameter_types(parameters)

        assert isinstance(categorization, dict)

    def test_analyze_convergence(self, analyzer, sample_optimization_results):
        """Test convergence analysis."""
        convergence = analyzer._analyze_convergence(sample_optimization_results)

        assert isinstance(convergence, dict)
        assert "convergence_efficiency" in convergence

    def test_create_analysis_summary(self, analyzer):
        """Test analysis summary creation."""
        analysis_results = {
            "performance_distribution": {"mean": 100.0},
            "optimization_landscape": {"ruggedness": 0.5},
            "best_result_analysis": {"parameters": {"param1": 0.2}}
        }

        summary = analyzer._create_analysis_summary(analysis_results, total_evaluations=100)

        assert isinstance(summary, dict)
        assert "total_evaluations" in summary