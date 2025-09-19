"""
Result transformation service for optimization module.

This module provides specialized services for transforming optimization results
between different formats and for different consuming modules.
"""

from decimal import Decimal
from typing import Any

from src.core.base import BaseService
from src.optimization.core import OptimizationResult


class ResultTransformationService(BaseService):
    """
    Service for optimization result transformations.

    Handles transforming optimization results for different consumers
    while maintaining data integrity and consistency.
    """

    def __init__(
        self,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ):
        """
        Initialize result transformation service.

        Args:
            name: Service name for identification
            config: Service configuration
            correlation_id: Request correlation ID
        """
        super().__init__(name or "ResultTransformationService", config, correlation_id)
        self._logger.info("ResultTransformationService initialized")

    def transform_for_strategies_module(
        self,
        optimization_result: dict[str, Any],
        current_parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Transform optimization result for strategies module compatibility.

        Args:
            optimization_result: Raw optimization result
            current_parameters: Original parameters for fallback

        Returns:
            Transformed result compatible with strategies module
        """
        result = optimization_result.get("optimization_result")
        if result:
            # Calculate performance improvement (strategies module expects this)
            baseline_performance = Decimal("0.1")  # Default baseline
            performance_improvement = self._calculate_performance_improvement(
                result.optimal_objective_value, baseline_performance
            )

            return {
                "success": True,
                "optimized_parameters": result.optimal_parameters,
                "performance_improvement": float(performance_improvement),
                "iterations_completed": result.iterations_completed,
                "performance_metrics": {
                    "objective_value": result.optimal_objective_value,
                    "convergence_achieved": result.convergence_achieved,
                    "iterations_completed": result.iterations_completed,
                },
                "optimization_metadata": optimization_result.get("metadata", {}),
                "analysis": optimization_result.get("analysis", {}),
            }
        else:
            return {
                "success": False,
                "error": "No optimization result available",
                "optimized_parameters": current_parameters,
            }

    def transform_for_web_interface(
        self, optimization_result: OptimizationResult
    ) -> dict[str, Any]:
        """
        Transform optimization result for web interface display.

        Args:
            optimization_result: Optimization result to transform

        Returns:
            Web interface compatible result
        """
        return {
            "id": optimization_result.optimization_id,
            "algorithm": optimization_result.algorithm_name,
            "status": "completed" if optimization_result.convergence_achieved else "partial",
            "optimal_parameters": optimization_result.optimal_parameters,
            "optimal_value": float(optimization_result.optimal_objective_value),
            "iterations": optimization_result.iterations_completed,
            "evaluations": optimization_result.evaluations_completed,
            "convergence": optimization_result.convergence_achieved,
            "duration_seconds": optimization_result.total_duration_seconds,
            "start_time": optimization_result.start_time.isoformat() if optimization_result.start_time else None,
            "end_time": optimization_result.end_time.isoformat() if optimization_result.end_time else None,
            "objective_values": {
                k: float(v) if isinstance(v, Decimal) else v
                for k, v in (optimization_result.objective_values or {}).items()
            },
            "validation_score": float(optimization_result.validation_score) if optimization_result.validation_score else None,
            "overfitting_score": float(optimization_result.overfitting_score) if optimization_result.overfitting_score else None,
            "robustness_score": float(optimization_result.robustness_score) if optimization_result.robustness_score else None,
            "statistical_significance": float(optimization_result.statistical_significance) if optimization_result.statistical_significance else None,
            "confidence_interval": [
                float(optimization_result.confidence_interval[0]),
                float(optimization_result.confidence_interval[1])
            ] if optimization_result.confidence_interval else None,
            "warnings": optimization_result.warnings or [],
        }

    def transform_for_analytics(
        self, optimization_result: OptimizationResult
    ) -> dict[str, Any]:
        """
        Transform optimization result for analytics processing.

        Args:
            optimization_result: Optimization result to transform

        Returns:
            Analytics-ready result format
        """
        return {
            "optimization_id": optimization_result.optimization_id,
            "algorithm_name": optimization_result.algorithm_name,
            "performance_metrics": {
                "optimal_objective_value": optimization_result.optimal_objective_value,
                "objective_values": optimization_result.objective_values or {},
                "validation_score": optimization_result.validation_score,
                "overfitting_score": optimization_result.overfitting_score,
                "robustness_score": optimization_result.robustness_score,
                "statistical_significance": optimization_result.statistical_significance,
            },
            "execution_metrics": {
                "iterations_completed": optimization_result.iterations_completed,
                "evaluations_completed": optimization_result.evaluations_completed,
                "convergence_achieved": optimization_result.convergence_achieved,
                "total_duration_seconds": optimization_result.total_duration_seconds,
            },
            "parameter_analysis": {
                "optimal_parameters": optimization_result.optimal_parameters,
                "parameter_stability": optimization_result.parameter_stability or {},
                "sensitivity_analysis": optimization_result.sensitivity_analysis or {},
            },
            "quality_indicators": {
                "confidence_interval": optimization_result.confidence_interval,
                "warnings": optimization_result.warnings or [],
                "config_used": optimization_result.config_used or {},
            },
            "timestamps": {
                "start_time": optimization_result.start_time,
                "end_time": optimization_result.end_time,
            },
        }

    def _calculate_performance_improvement(
        self, optimal_value: Decimal, baseline: Decimal
    ) -> Decimal:
        """
        Calculate performance improvement percentage.

        Args:
            optimal_value: Optimal objective value achieved
            baseline: Baseline performance value

        Returns:
            Performance improvement as decimal percentage
        """
        if optimal_value > baseline and baseline > 0:
            return (optimal_value - baseline) / baseline
        else:
            return Decimal("0")

    def standardize_financial_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Standardize financial data for consistent cross-module usage.

        Args:
            data: Data containing financial values

        Returns:
            Standardized data with consistent financial formatting
        """
        financial_fields = [
            "optimal_objective_value", "objective_value", "total_return",
            "sharpe_ratio", "max_drawdown", "volatility", "initial_capital"
        ]

        standardized = data.copy()

        for field in financial_fields:
            if field in standardized and standardized[field] is not None:
                try:
                    # Convert to Decimal for precision, then to string for transport
                    if isinstance(standardized[field], (int, float)):
                        standardized[field] = str(Decimal(str(standardized[field])))
                    elif isinstance(standardized[field], Decimal):
                        standardized[field] = str(standardized[field])
                except (ValueError, TypeError):
                    # Leave as-is if conversion fails
                    self._logger.warning(f"Could not standardize financial field {field}: {standardized[field]}")

        return standardized

    def extract_summary_metrics(self, optimization_result: OptimizationResult) -> dict[str, Any]:
        """
        Extract key summary metrics from optimization result.

        Args:
            optimization_result: Optimization result to summarize

        Returns:
            Summary metrics dictionary
        """
        return {
            "optimization_id": optimization_result.optimization_id,
            "algorithm": optimization_result.algorithm_name,
            "optimal_value": optimization_result.optimal_objective_value,
            "convergence": optimization_result.convergence_achieved,
            "iterations": optimization_result.iterations_completed,
            "duration_seconds": optimization_result.total_duration_seconds,
            "parameter_count": len(optimization_result.optimal_parameters),
            "has_validation": optimization_result.validation_score is not None,
            "has_warnings": bool(optimization_result.warnings),
            "quality_score": self._calculate_quality_score(optimization_result),
        }

    def _calculate_quality_score(self, optimization_result: OptimizationResult) -> float:
        """
        Calculate overall quality score for optimization result.

        Args:
            optimization_result: Optimization result to score

        Returns:
            Quality score between 0 and 1
        """
        score = 0.0

        # Base score from convergence
        if optimization_result.convergence_achieved:
            score += 0.3

        # Score from validation metrics
        if optimization_result.validation_score:
            score += min(float(optimization_result.validation_score) * 0.2, 0.2)

        # Score from robustness
        if optimization_result.robustness_score:
            score += min(float(optimization_result.robustness_score) * 0.2, 0.2)

        # Score from statistical significance
        if optimization_result.statistical_significance:
            score += min(float(optimization_result.statistical_significance) * 0.2, 0.2)

        # Penalty for warnings
        if optimization_result.warnings:
            score -= min(len(optimization_result.warnings) * 0.05, 0.1)

        # Bonus for low overfitting
        if optimization_result.overfitting_score:
            overfitting_penalty = min(float(optimization_result.overfitting_score) * 0.1, 0.1)
            score -= overfitting_penalty

        return max(0.0, min(1.0, score))