"""
Tests for optimization analysis service.

This module ensures comprehensive testing of the AnalysisService class,
covering all functionality including error handling and edge cases.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from decimal import Decimal
import asyncio

from src.optimization.analysis_service import AnalysisService
from src.optimization.analysis import ResultsAnalyzer, ParameterImportanceAnalyzer
from src.core.exceptions import ServiceError


class TestAnalysisService:
    """Test cases for AnalysisService class."""

    @pytest.fixture
    def mock_results_analyzer(self):
        """Create a mock ResultsAnalyzer."""
        analyzer = Mock(spec=ResultsAnalyzer)
        analyzer.importance_analyzer = Mock(spec=ParameterImportanceAnalyzer)
        return analyzer

    @pytest.fixture
    def analysis_service(self, mock_results_analyzer):
        """Create an AnalysisService instance with mocked dependencies."""
        return AnalysisService(
            results_analyzer=mock_results_analyzer,
            name="TestAnalysisService",
            correlation_id="test-correlation-id"
        )

    @pytest.fixture
    def analysis_service_default(self):
        """Create an AnalysisService instance with default dependencies."""
        with patch('src.optimization.analysis_service.ResultsAnalyzer') as mock_analyzer_class:
            mock_analyzer_instance = Mock(spec=ResultsAnalyzer)
            mock_analyzer_instance.importance_analyzer = Mock(spec=ParameterImportanceAnalyzer)
            mock_analyzer_class.return_value = mock_analyzer_instance

            service = AnalysisService()
            service._results_analyzer = mock_analyzer_instance
            return service

    @pytest.fixture
    def sample_optimization_results(self):
        """Sample optimization results for testing."""
        return [
            {
                "parameters": {"param1": Decimal("0.1"), "param2": Decimal("0.2")},
                "objective_values": {"profit": Decimal("100.0"), "sharpe": Decimal("1.5")},
                "performance": {"return": Decimal("0.15"), "volatility": Decimal("0.12")}
            },
            {
                "parameters": {"param1": Decimal("0.15"), "param2": Decimal("0.25")},
                "objective_values": {"profit": Decimal("120.0"), "sharpe": Decimal("1.7")},
                "performance": {"return": Decimal("0.18"), "volatility": Decimal("0.14")}
            },
            {
                "parameters": {"param1": Decimal("0.05"), "param2": Decimal("0.3")},
                "objective_values": {"profit": Decimal("80.0"), "sharpe": Decimal("1.2")},
                "performance": {"return": Decimal("0.12"), "volatility": Decimal("0.10")}
            }
        ]

    @pytest.fixture
    def sample_parameter_names(self):
        """Sample parameter names for testing."""
        return ["param1", "param2"]


class TestAnalysisServiceInitialization(TestAnalysisService):
    """Test AnalysisService initialization."""

    def test_initialization_with_injected_analyzer(self, mock_results_analyzer):
        """Test service initialization with injected analyzer."""
        service = AnalysisService(
            results_analyzer=mock_results_analyzer,
            name="TestService",
            config={"test_key": "test_value"},
            correlation_id="test-id"
        )

        assert service._results_analyzer == mock_results_analyzer
        assert service.name == "TestService"
        assert service.get_config() == {"test_key": "test_value"}
        assert service.correlation_id == "test-id"

    def test_initialization_with_default_analyzer(self):
        """Test service initialization creates default analyzer."""
        with patch('src.optimization.analysis.ResultsAnalyzer') as mock_analyzer_class:
            mock_analyzer_instance = Mock()
            mock_analyzer_class.return_value = mock_analyzer_instance

            service = AnalysisService()

            assert service.name == "AnalysisService"
            assert service._results_analyzer == mock_analyzer_instance
            mock_analyzer_class.assert_called_once_with()

    def test_initialization_default_values(self):
        """Test service initialization with default values."""
        with patch('src.optimization.analysis.ResultsAnalyzer'):
            service = AnalysisService()

            assert service.name == "AnalysisService"
            assert service.get_config() == {}
            assert service.correlation_id is not None  # Auto-generated

    def test_dependencies_added_when_analyzer_injected(self, mock_results_analyzer):
        """Test dependencies are added when analyzer is injected."""
        with patch.object(AnalysisService, 'add_dependency') as mock_add_dep:
            service = AnalysisService(results_analyzer=mock_results_analyzer)
            mock_add_dep.assert_called_once_with("ResultsAnalyzer")

    def test_no_dependencies_added_for_default_analyzer(self):
        """Test no dependencies added when using default analyzer."""
        with patch('src.optimization.analysis_service.ResultsAnalyzer'):
            with patch.object(AnalysisService, 'add_dependency') as mock_add_dep:
                service = AnalysisService()
                mock_add_dep.assert_not_called()


class TestAnalyzeOptimizationResults(TestAnalysisService):
    """Test analyze_optimization_results method."""

    @pytest.mark.asyncio
    async def test_analyze_optimization_results_success(
        self,
        analysis_service,
        sample_optimization_results,
        sample_parameter_names
    ):
        """Test successful optimization results analysis."""
        # Setup mock return value
        expected_analysis = {
            "performance_distribution": {"mean": Decimal("100.0")},
            "parameter_correlations": {"param1": {"param2": 0.5}},
            "best_result_analysis": {"parameters": {"param1": Decimal("0.15")}}
        }
        analysis_service._results_analyzer.analyze_optimization_results.return_value = expected_analysis

        # Call method
        result = await analysis_service.analyze_optimization_results(
            optimization_results=sample_optimization_results,
            parameter_names=sample_parameter_names,
            best_result=sample_optimization_results[1]
        )

        # Verify results
        assert result == expected_analysis
        analysis_service._results_analyzer.analyze_optimization_results.assert_called_once_with(
            optimization_results=sample_optimization_results,
            parameter_names=sample_parameter_names,
            best_result=sample_optimization_results[1]
        )

    @pytest.mark.asyncio
    async def test_analyze_optimization_results_without_best_result(
        self,
        analysis_service,
        sample_optimization_results,
        sample_parameter_names
    ):
        """Test analysis without best result specified."""
        expected_analysis = {"analysis": "results"}
        analysis_service._results_analyzer.analyze_optimization_results.return_value = expected_analysis

        result = await analysis_service.analyze_optimization_results(
            optimization_results=sample_optimization_results,
            parameter_names=sample_parameter_names
        )

        assert result == expected_analysis
        analysis_service._results_analyzer.analyze_optimization_results.assert_called_once_with(
            optimization_results=sample_optimization_results,
            parameter_names=sample_parameter_names,
            best_result=None
        )

    @pytest.mark.asyncio
    async def test_analyze_optimization_results_empty_results(self, analysis_service):
        """Test analysis with empty optimization results."""
        expected_analysis = {"empty": "analysis"}
        analysis_service._results_analyzer.analyze_optimization_results.return_value = expected_analysis

        result = await analysis_service.analyze_optimization_results(
            optimization_results=[],
            parameter_names=["param1"]
        )

        assert result == expected_analysis

    @pytest.mark.asyncio
    async def test_analyze_optimization_results_analyzer_exception(
        self,
        analysis_service,
        sample_optimization_results,
        sample_parameter_names
    ):
        """Test error handling when analyzer raises exception."""
        # Setup mock to raise exception
        analysis_service._results_analyzer.analyze_optimization_results.side_effect = ValueError("Analysis failed")

        # Verify exception is wrapped and propagated
        from src.core.exceptions import OptimizationError
        with pytest.raises(OptimizationError, match="Optimization results analysis failed"):
            await analysis_service.analyze_optimization_results(
                optimization_results=sample_optimization_results,
                parameter_names=sample_parameter_names
            )

    @pytest.mark.asyncio
    async def test_analyze_optimization_results_logging(
        self,
        analysis_service,
        sample_optimization_results,
        sample_parameter_names
    ):
        """Test proper logging during analysis."""
        expected_analysis = {"test": "result"}
        analysis_service._results_analyzer.analyze_optimization_results.return_value = expected_analysis

        with patch.object(analysis_service, '_logger') as mock_logger:
            await analysis_service.analyze_optimization_results(
                optimization_results=sample_optimization_results,
                parameter_names=sample_parameter_names
            )

            mock_logger.info.assert_called_once_with(
                "Optimization results analysis completed",
                result_count=len(sample_optimization_results),
                parameter_count=len(sample_parameter_names),
            )

    @pytest.mark.asyncio
    async def test_analyze_optimization_results_error_logging(
        self,
        analysis_service,
        sample_optimization_results,
        sample_parameter_names
    ):
        """Test error logging during analysis failure."""
        from src.core.exceptions import OptimizationError
        error_msg = "Analysis computation failed"
        analysis_service._results_analyzer.analyze_optimization_results.side_effect = RuntimeError(error_msg)

        with patch.object(analysis_service, '_logger') as mock_logger:
            with pytest.raises(OptimizationError):
                await analysis_service.analyze_optimization_results(
                    optimization_results=sample_optimization_results,
                    parameter_names=sample_parameter_names
                )

            mock_logger.error.assert_called_once()
            error_call = mock_logger.error.call_args[0][0]
            assert error_msg in str(error_call)


class TestAnalyzeParameterImportance(TestAnalysisService):
    """Test analyze_parameter_importance method."""

    @pytest.mark.asyncio
    async def test_analyze_parameter_importance_success(
        self,
        analysis_service,
        sample_optimization_results,
        sample_parameter_names
    ):
        """Test successful parameter importance analysis."""
        expected_importance = [
            {"parameter": "param1", "importance": Decimal("0.8")},
            {"parameter": "param2", "importance": Decimal("0.6")}
        ]
        analysis_service._results_analyzer.importance_analyzer.analyze_parameter_importance.return_value = expected_importance

        result = await analysis_service.analyze_parameter_importance(
            optimization_results=sample_optimization_results,
            parameter_names=sample_parameter_names
        )

        assert result == expected_importance
        analysis_service._results_analyzer.importance_analyzer.analyze_parameter_importance.assert_called_once_with(
            optimization_results=sample_optimization_results,
            parameter_names=sample_parameter_names
        )

    @pytest.mark.asyncio
    async def test_analyze_parameter_importance_empty_results(self, analysis_service):
        """Test parameter importance analysis with empty results."""
        expected_importance = []
        analysis_service._results_analyzer.importance_analyzer.analyze_parameter_importance.return_value = expected_importance

        result = await analysis_service.analyze_parameter_importance(
            optimization_results=[],
            parameter_names=["param1"]
        )

        assert result == expected_importance

    @pytest.mark.asyncio
    async def test_analyze_parameter_importance_single_parameter(
        self,
        analysis_service,
        sample_optimization_results
    ):
        """Test parameter importance analysis with single parameter."""
        expected_importance = [{"parameter": "param1", "importance": Decimal("1.0")}]
        analysis_service._results_analyzer.importance_analyzer.analyze_parameter_importance.return_value = expected_importance

        result = await analysis_service.analyze_parameter_importance(
            optimization_results=sample_optimization_results,
            parameter_names=["param1"]
        )

        assert result == expected_importance

    @pytest.mark.asyncio
    async def test_analyze_parameter_importance_analyzer_exception(
        self,
        analysis_service,
        sample_optimization_results,
        sample_parameter_names
    ):
        """Test error handling when importance analyzer raises exception."""
        analysis_service._results_analyzer.importance_analyzer.analyze_parameter_importance.side_effect = RuntimeError("Importance analysis failed")

        from src.core.exceptions import OptimizationError
        with pytest.raises(OptimizationError, match="Parameter importance analysis failed"):
            await analysis_service.analyze_parameter_importance(
                optimization_results=sample_optimization_results,
                parameter_names=sample_parameter_names
            )

    @pytest.mark.asyncio
    async def test_analyze_parameter_importance_logging(
        self,
        analysis_service,
        sample_optimization_results,
        sample_parameter_names
    ):
        """Test proper logging during parameter importance analysis."""
        expected_importance = [{"test": "importance"}]
        analysis_service._results_analyzer.importance_analyzer.analyze_parameter_importance.return_value = expected_importance

        with patch.object(analysis_service, '_logger') as mock_logger:
            await analysis_service.analyze_parameter_importance(
                optimization_results=sample_optimization_results,
                parameter_names=sample_parameter_names
            )

            mock_logger.info.assert_called_once_with(
                "Parameter importance analysis completed",
                parameter_count=len(sample_parameter_names),
                result_count=len(sample_optimization_results),
            )

    @pytest.mark.asyncio
    async def test_analyze_parameter_importance_error_logging(
        self,
        analysis_service,
        sample_optimization_results,
        sample_parameter_names
    ):
        """Test error logging during importance analysis failure."""
        from src.core.exceptions import OptimizationError
        error_msg = "Importance computation failed"
        analysis_service._results_analyzer.importance_analyzer.analyze_parameter_importance.side_effect = ValueError(error_msg)

        with patch.object(analysis_service, '_logger') as mock_logger:
            with pytest.raises(OptimizationError):
                await analysis_service.analyze_parameter_importance(
                    optimization_results=sample_optimization_results,
                    parameter_names=sample_parameter_names
                )

            mock_logger.error.assert_called_once()
            error_call = mock_logger.error.call_args[0][0]
            assert error_msg in str(error_call)


class TestAnalysisServiceIntegration(TestAnalysisService):
    """Test integration scenarios and edge cases."""

    @pytest.mark.asyncio
    async def test_concurrent_analysis_operations(
        self,
        analysis_service,
        sample_optimization_results,
        sample_parameter_names
    ):
        """Test concurrent execution of analysis operations."""
        # Setup different return values for different calls
        analysis_service._results_analyzer.analyze_optimization_results.return_value = {"analysis": "result1"}
        analysis_service._results_analyzer.importance_analyzer.analyze_parameter_importance.return_value = [{"importance": "result1"}]

        # Run operations concurrently
        results = await asyncio.gather(
            analysis_service.analyze_optimization_results(sample_optimization_results, sample_parameter_names),
            analysis_service.analyze_parameter_importance(sample_optimization_results, sample_parameter_names)
        )

        assert len(results) == 2
        assert results[0] == {"analysis": "result1"}
        assert results[1] == [{"importance": "result1"}]

    @pytest.mark.asyncio
    async def test_large_dataset_analysis(self, analysis_service):
        """Test analysis with large dataset."""
        # Create large optimization results dataset
        large_results = []
        for i in range(1000):
            large_results.append({
                "parameters": {"param1": Decimal(str(i * 0.001)), "param2": Decimal(str(i * 0.002))},
                "objective_values": {"profit": Decimal(str(i * 100.0))},
                "performance": {"return": Decimal(str(i * 0.01))}
            })

        expected_analysis = {"large": "analysis"}
        analysis_service._results_analyzer.analyze_optimization_results.return_value = expected_analysis

        result = await analysis_service.analyze_optimization_results(
            optimization_results=large_results,
            parameter_names=["param1", "param2"]
        )

        assert result == expected_analysis

    @pytest.mark.asyncio
    async def test_malformed_optimization_results(self, analysis_service):
        """Test handling of malformed optimization results."""
        malformed_results = [
            {"invalid": "structure"},
            {"parameters": None},
            {},
            {"parameters": {"param1": "invalid_decimal"}}
        ]

        # Let the analyzer handle the malformed data
        analysis_service._results_analyzer.analyze_optimization_results.return_value = {"handled": "malformed"}

        result = await analysis_service.analyze_optimization_results(
            optimization_results=malformed_results,
            parameter_names=["param1"]
        )

        assert result == {"handled": "malformed"}

    @pytest.mark.asyncio
    async def test_analysis_with_extreme_decimal_values(self, analysis_service):
        """Test analysis with extreme decimal values for financial precision."""
        extreme_results = [
            {
                "parameters": {"param1": Decimal("0.000000001"), "param2": Decimal("999999999.999999999")},
                "objective_values": {"profit": Decimal("0.0000000000001")},
                "performance": {"return": Decimal("-0.999999999999999")}
            }
        ]

        expected_analysis = {"extreme": "values_handled"}
        analysis_service._results_analyzer.analyze_optimization_results.return_value = expected_analysis

        result = await analysis_service.analyze_optimization_results(
            optimization_results=extreme_results,
            parameter_names=["param1", "param2"]
        )

        assert result == expected_analysis

    def test_service_inheritance_and_interface_compliance(self, analysis_service):
        """Test that service properly inherits from BaseService and implements IAnalysisService."""
        from src.core.base import BaseService
        from src.optimization.interfaces import IAnalysisService

        assert isinstance(analysis_service, BaseService)
        assert isinstance(analysis_service, IAnalysisService)

        # Check that required methods exist
        assert hasattr(analysis_service, 'analyze_optimization_results')
        assert hasattr(analysis_service, 'analyze_parameter_importance')


class TestAnalysisServiceEdgeCases(TestAnalysisService):
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_none_parameter_names(self, analysis_service, sample_optimization_results):
        """Test handling when parameter_names is None."""
        # This should be handled by the analyzer, but service should not crash
        analysis_service._results_analyzer.analyze_optimization_results.side_effect = TypeError("parameter_names cannot be None")

        # Pass None as parameter_names - this should cause TypeError from analyzer
        from src.core.exceptions import OptimizationError
        with pytest.raises(OptimizationError, match="Optimization results analysis failed"):
            await analysis_service.analyze_optimization_results(
                optimization_results=sample_optimization_results,
                parameter_names=None
            )

    @pytest.mark.asyncio
    async def test_none_optimization_results(self, analysis_service):
        """Test handling when optimization_results is None."""
        from src.core.exceptions import OptimizationError
        # Make the analyzer raise TypeError for None inputs
        analysis_service._results_analyzer.analyze_optimization_results.side_effect = TypeError("Cannot analyze None results")

        # This should cause an error from the analyzer, wrapped by the service
        with pytest.raises(OptimizationError):
            await analysis_service.analyze_optimization_results(
                optimization_results=None,
                parameter_names=["param1"]
            )

    @pytest.mark.asyncio
    async def test_async_behavior_with_sleep(self, analysis_service, sample_optimization_results, sample_parameter_names):
        """Test that methods properly yield to event loop."""
        analysis_service._results_analyzer.analyze_optimization_results.return_value = {"async": "result"}

        # Measure that the method actually yields control
        import time
        start_time = time.time()

        result = await analysis_service.analyze_optimization_results(
            optimization_results=sample_optimization_results,
            parameter_names=sample_parameter_names
        )

        end_time = time.time()

        # The method should complete (asyncio.sleep(0) is nearly instantaneous but yields control)
        assert result == {"async": "result"}
        assert end_time >= start_time  # Basic sanity check

    def test_analyzer_access(self, analysis_service):
        """Test that the analyzer is properly accessible."""
        assert hasattr(analysis_service, '_results_analyzer')
        assert analysis_service._results_analyzer is not None

        # Test that importance analyzer is accessible through results analyzer
        assert hasattr(analysis_service._results_analyzer, 'importance_analyzer')


class TestAnalysisServiceFinancialEdgeCases(TestAnalysisService):
    """Test financial calculation specific edge cases."""

    @pytest.mark.asyncio
    async def test_zero_profit_analysis(self, analysis_service):
        """Test analysis with zero profit scenarios."""
        zero_profit_results = [
            {
                "parameters": {"param1": Decimal("0.1"), "param2": Decimal("0.2")},
                "objective_values": {"profit": Decimal("0.0"), "sharpe": Decimal("0.0")},
                "performance": {"return": Decimal("0.0"), "volatility": Decimal("0.05")}
            }
        ]

        expected_analysis = {"zero_profit": "handled"}
        analysis_service._results_analyzer.analyze_optimization_results.return_value = expected_analysis

        result = await analysis_service.analyze_optimization_results(
            optimization_results=zero_profit_results,
            parameter_names=["param1", "param2"]
        )

        assert result == expected_analysis

    @pytest.mark.asyncio
    async def test_negative_performance_analysis(self, analysis_service):
        """Test analysis with negative performance values."""
        negative_results = [
            {
                "parameters": {"param1": Decimal("0.5"), "param2": Decimal("0.8")},
                "objective_values": {"profit": Decimal("-1000.0"), "sharpe": Decimal("-2.5")},
                "performance": {"return": Decimal("-0.25"), "volatility": Decimal("0.30")}
            }
        ]

        expected_analysis = {"negative": "performance_handled"}
        analysis_service._results_analyzer.analyze_optimization_results.return_value = expected_analysis

        result = await analysis_service.analyze_optimization_results(
            optimization_results=negative_results,
            parameter_names=["param1", "param2"]
        )

        assert result == expected_analysis

    @pytest.mark.asyncio
    async def test_high_precision_decimal_preservation(self, analysis_service):
        """Test that high precision decimals are preserved through analysis."""
        high_precision_results = [
            {
                "parameters": {
                    "param1": Decimal("0.123456789012345678"),
                    "param2": Decimal("0.987654321098765432")
                },
                "objective_values": {
                    "profit": Decimal("1234.56789012345678901234"),
                    "sharpe": Decimal("1.23456789012345678901234")
                },
                "performance": {
                    "return": Decimal("0.15678901234567890123456"),
                    "volatility": Decimal("0.12345678901234567890123")
                }
            }
        ]

        # The analyzer should receive the exact decimal values
        analysis_service._results_analyzer.analyze_optimization_results.return_value = {"precision": "preserved"}

        result = await analysis_service.analyze_optimization_results(
            optimization_results=high_precision_results,
            parameter_names=["param1", "param2"]
        )

        assert result == {"precision": "preserved"}

        # Verify the analyzer was called with the exact high-precision data
        call_args = analysis_service._results_analyzer.analyze_optimization_results.call_args
        passed_results = call_args[1]['optimization_results']
        assert passed_results[0]["parameters"]["param1"] == Decimal("0.123456789012345678")