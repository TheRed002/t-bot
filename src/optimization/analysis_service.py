"""
Optimization analysis service implementation.

This module provides a service layer wrapper around the analysis components,
following proper service layer patterns and dependency injection.
"""

import asyncio
from typing import Any

from src.core.base import BaseService
from src.optimization.analysis import ResultsAnalyzer
from src.optimization.interfaces import IAnalysisService


class AnalysisService(BaseService, IAnalysisService):
    """
    Service for optimization result analysis.

    Provides clean abstraction for analysis operations while maintaining
    proper separation of concerns between analysis components.
    """

    def __init__(
        self,
        results_analyzer: ResultsAnalyzer | None = None,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ):
        """
        Initialize analysis service.

        Args:
            results_analyzer: Results analyzer instance (optional, will create if None)
            name: Service name for identification
            config: Service configuration
            correlation_id: Request correlation ID
        """
        super().__init__(name or "AnalysisService", config, correlation_id)

        # Use injected analyzer with fallback for backward compatibility
        if results_analyzer is None:
            # Fallback for cases where DI hasn't been fully set up yet
            self._logger.warning("ResultsAnalyzer not injected, creating default instance")
            from src.optimization.analysis import ResultsAnalyzer

            self._results_analyzer = ResultsAnalyzer()
        else:
            self._results_analyzer = results_analyzer

        # Add dependencies
        if results_analyzer:
            self.add_dependency("ResultsAnalyzer")

        self._logger.info("AnalysisService initialized")

    async def analyze_optimization_results(
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
        try:
            # Yield control to event loop
            await asyncio.sleep(0)

            # Delegate to results analyzer
            analysis = self._results_analyzer.analyze_optimization_results(
                optimization_results=optimization_results,
                parameter_names=parameter_names,
                best_result=best_result,
            )

            self._logger.info(
                "Optimization results analysis completed",
                result_count=len(optimization_results) if optimization_results is not None else 0,
                parameter_count=len(parameter_names) if parameter_names is not None else 0,
            )

            return analysis

        except Exception as e:
            self._logger.error(f"Analysis failed: {e}")
            # Re-raise with better context for upstream error handling
            from src.core.exceptions import OptimizationError

            raise OptimizationError(
                f"Optimization results analysis failed: {e}",
                error_code="OPT_ANALYSIS_001",
                optimization_stage="results_analysis",
            ) from e

    async def analyze_parameter_importance(
        self,
        optimization_results: list[dict[str, Any]],
        parameter_names: list[str],
    ) -> list[Any]:
        """
        Analyze parameter importance.

        Args:
            optimization_results: List of optimization result dictionaries
            parameter_names: List of parameter names

        Returns:
            List of parameter importance analysis results
        """
        try:
            # Yield control to event loop
            await asyncio.sleep(0)

            # Use the importance analyzer from results analyzer
            importance_results = (
                self._results_analyzer.importance_analyzer.analyze_parameter_importance(
                    optimization_results=optimization_results,
                    parameter_names=parameter_names,
                )
            )

            self._logger.info(
                "Parameter importance analysis completed",
                parameter_count=len(parameter_names),
                result_count=len(optimization_results),
            )

            return importance_results

        except Exception as e:
            self._logger.error(f"Parameter importance analysis failed: {e}")
            # Re-raise with better context for upstream error handling
            from src.core.exceptions import OptimizationError

            raise OptimizationError(
                f"Parameter importance analysis failed: {e}",
                error_code="OPT_ANALYSIS_002",
                optimization_stage="parameter_importance_analysis",
            ) from e
