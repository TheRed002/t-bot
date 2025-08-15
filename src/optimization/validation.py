"""
Validation and Overfitting Prevention for Optimization.

This module implements comprehensive validation techniques to prevent
overfitting and ensure robust parameter selection in trading strategy
optimization. Includes walk-forward analysis, cross-validation,
statistical significance testing, and robustness analysis.

Key Features:
- Time-series aware cross-validation
- Walk-forward analysis with expanding/rolling windows
- Out-of-sample testing with statistical significance
- Monte Carlo robustness testing
- Overfitting detection with performance degradation metrics
- Multiple hypothesis testing correction
- Bootstrap confidence intervals

Critical for Financial Applications:
- Proper time series validation (no lookahead bias)
- Robust statistical testing
- Multiple time period validation
- Market regime change detection
- Conservative significance thresholds
- Comprehensive audit trails
"""

import asyncio
import math
import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

from src.backtesting.engine import BacktestEngine, BacktestResult
from src.core.exceptions import ValidationError, OptimizationError
from src.core.logging import get_logger
from src.core.types import TradingMode
from src.optimization.core import OptimizationObjective, OptimizationConfig
from src.utils.decimal_utils import decimal_to_float, float_to_decimal
from src.utils.decorators import time_execution, memory_usage

logger = get_logger(__name__)


class ValidationMetrics(BaseModel):
    """
    Comprehensive validation metrics for optimization results.
    
    Contains in-sample, out-of-sample, and robustness metrics
    to assess optimization quality and overfitting risk.
    """
    
    # Basic metrics
    in_sample_score: Decimal = Field(description="In-sample performance score")
    out_of_sample_score: Decimal = Field(description="Out-of-sample performance score")
    validation_score: Decimal = Field(description="Cross-validation score")
    
    # Overfitting metrics
    overfitting_ratio: Decimal = Field(
        description="Ratio of out-of-sample to in-sample performance"
    )
    performance_degradation: Decimal = Field(
        description="Performance degradation from in-sample to out-of-sample"
    )
    
    # Statistical significance
    p_value: Optional[Decimal] = Field(
        default=None, 
        description="Statistical significance p-value"
    )
    confidence_interval: Optional[Tuple[Decimal, Decimal]] = Field(
        default=None, 
        description="Confidence interval for out-of-sample performance"
    )
    
    # Robustness metrics
    stability_score: Decimal = Field(
        description="Parameter stability across validation folds"
    )
    robustness_score: Decimal = Field(
        description="Performance robustness to parameter changes"
    )
    worst_case_performance: Decimal = Field(
        description="Worst-case performance across validation periods"
    )
    
    # Time series specific
    walk_forward_scores: List[Decimal] = Field(
        default_factory=list,
        description="Performance scores from walk-forward analysis"
    )
    regime_consistency: Optional[Decimal] = Field(
        default=None,
        description="Consistency across different market regimes"
    )
    
    # Quality flags
    is_statistically_significant: bool = Field(
        description="Whether results are statistically significant"
    )
    is_robust: bool = Field(
        description="Whether results pass robustness tests"
    )
    has_overfitting: bool = Field(
        description="Whether overfitting is detected"
    )
    
    def get_overall_quality_score(self) -> Decimal:
        """Calculate overall quality score (0-1)."""
        scores = []
        
        # Statistical significance (0.3 weight)
        if self.is_statistically_significant:
            scores.append(Decimal("0.3"))
        
        # Robustness (0.3 weight)
        if self.is_robust:
            scores.append(Decimal("0.3"))
        
        # No overfitting (0.2 weight)
        if not self.has_overfitting:
            scores.append(Decimal("0.2"))
        
        # Stability (0.2 weight)
        stability_component = min(self.stability_score, Decimal("1")) * Decimal("0.2")
        scores.append(stability_component)
        
        return sum(scores)


class ValidationConfig(BaseModel):
    """
    Configuration for validation and overfitting prevention.
    
    Defines all validation strategies, thresholds, and
    statistical testing parameters.
    """
    
    # Cross-validation settings
    enable_cross_validation: bool = Field(
        default=True, 
        description="Enable cross-validation"
    )
    cv_method: str = Field(
        default="time_series", 
        description="Cross-validation method"
    )
    cv_folds: int = Field(
        default=5, 
        ge=2, 
        description="Number of cross-validation folds"
    )
    cv_gap: int = Field(
        default=0, 
        ge=0, 
        description="Gap between train and test in CV (days)"
    )
    
    # Walk-forward analysis
    enable_walk_forward: bool = Field(
        default=True, 
        description="Enable walk-forward analysis"
    )
    walk_forward_periods: int = Field(
        default=12, 
        ge=2, 
        description="Number of walk-forward periods"
    )
    walk_forward_step_days: int = Field(
        default=30, 
        ge=1, 
        description="Step size for walk-forward analysis"
    )
    walk_forward_window_type: str = Field(
        default="expanding", 
        description="Window type: 'expanding' or 'rolling'"
    )
    walk_forward_min_train_days: int = Field(
        default=180, 
        ge=30, 
        description="Minimum training period for walk-forward"
    )
    
    # Out-of-sample testing
    out_of_sample_ratio: float = Field(
        default=0.25, 
        gt=0, 
        lt=0.5, 
        description="Ratio of data reserved for out-of-sample testing"
    )
    require_oos_significance: bool = Field(
        default=True, 
        description="Require statistical significance on OOS data"
    )
    
    # Statistical testing
    significance_level: Decimal = Field(
        default=Decimal("0.05"), 
        gt=Decimal("0"), 
        lt=Decimal("1"), 
        description="Statistical significance level"
    )
    multiple_testing_correction: str = Field(
        default="bonferroni", 
        description="Multiple testing correction method"
    )
    bootstrap_samples: int = Field(
        default=10000, 
        ge=1000, 
        description="Number of bootstrap samples"
    )
    
    # Overfitting detection
    overfitting_threshold: Decimal = Field(
        default=Decimal("0.15"), 
        gt=Decimal("0"), 
        description="Threshold for overfitting detection"
    )
    degradation_threshold: Decimal = Field(
        default=Decimal("0.10"), 
        gt=Decimal("0"), 
        description="Acceptable performance degradation"
    )
    stability_threshold: Decimal = Field(
        default=Decimal("0.7"), 
        gt=Decimal("0"), 
        le=Decimal("1"), 
        description="Minimum stability score required"
    )
    
    # Robustness testing
    enable_robustness_testing: bool = Field(
        default=True, 
        description="Enable robustness testing"
    )
    robustness_perturbation: float = Field(
        default=0.05, 
        gt=0, 
        lt=0.5, 
        description="Parameter perturbation for robustness testing"
    )
    robustness_samples: int = Field(
        default=100, 
        ge=10, 
        description="Number of samples for robustness testing"
    )
    
    # Monte Carlo validation
    monte_carlo_runs: int = Field(
        default=1000, 
        ge=100, 
        description="Number of Monte Carlo simulation runs"
    )
    random_seed: Optional[int] = Field(
        default=42, 
        description="Random seed for reproducibility"
    )
    
    @field_validator("cv_method")
    @classmethod
    def validate_cv_method(cls, v):
        """Validate cross-validation method."""
        valid_methods = ["time_series", "blocked", "gap", "combinatorial"]
        if v not in valid_methods:
            raise ValueError(f"CV method must be one of {valid_methods}")
        return v
    
    @field_validator("walk_forward_window_type")
    @classmethod
    def validate_window_type(cls, v):
        """Validate walk-forward window type."""
        valid_types = ["expanding", "rolling"]
        if v not in valid_types:
            raise ValueError(f"Window type must be one of {valid_types}")
        return v
    
    @field_validator("multiple_testing_correction")
    @classmethod
    def validate_correction_method(cls, v):
        """Validate multiple testing correction method."""
        valid_methods = ["bonferroni", "holm", "fdr_bh", "none"]
        if v not in valid_methods:
            raise ValueError(f"Correction method must be one of {valid_methods}")
        return v


class TimeSeriesValidator:
    """
    Time series specific validation with proper temporal splits.
    
    Ensures no lookahead bias and respects temporal ordering
    in validation procedures.
    """
    
    def __init__(self, config: ValidationConfig):
        """
        Initialize time series validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        
        logger.info(
            "TimeSeriesValidator initialized",
            cv_method=config.cv_method,
            cv_folds=config.cv_folds
        )
    
    def create_time_series_splits(
        self, 
        data_length: int, 
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Create time series splits for cross-validation.
        
        Args:
            data_length: Length of the dataset
            start_date: Start date of the data
            end_date: End date of the data
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        if self.config.cv_method == "time_series":
            return self._create_expanding_window_splits(data_length)
        elif self.config.cv_method == "blocked":
            return self._create_blocked_splits(data_length)
        elif self.config.cv_method == "gap":
            return self._create_gap_splits(data_length)
        else:
            return self._create_expanding_window_splits(data_length)
    
    def _create_expanding_window_splits(self, data_length: int) -> List[Tuple[List[int], List[int]]]:
        """Create expanding window splits."""
        splits = []
        
        # Calculate split points
        total_test_size = data_length // (self.config.cv_folds + 1)
        min_train_size = max(30, data_length // (self.config.cv_folds * 2))
        
        for fold in range(self.config.cv_folds):
            # Expanding training window
            train_end = min_train_size + fold * total_test_size
            test_start = train_end + self.config.cv_gap
            test_end = min(test_start + total_test_size, data_length)
            
            if test_start >= data_length:
                break
            
            train_indices = list(range(train_end))
            test_indices = list(range(test_start, test_end))
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))
        
        logger.info(f"Created {len(splits)} expanding window splits")
        return splits
    
    def _create_blocked_splits(self, data_length: int) -> List[Tuple[List[int], List[int]]]:
        """Create blocked time series splits."""
        splits = []
        block_size = data_length // self.config.cv_folds
        
        for fold in range(self.config.cv_folds):
            test_start = fold * block_size
            test_end = min((fold + 1) * block_size, data_length)
            
            # Training data comes before test block
            train_indices = list(range(test_start))
            test_indices = list(range(test_start, test_end))
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))
        
        logger.info(f"Created {len(splits)} blocked splits")
        return splits
    
    def _create_gap_splits(self, data_length: int) -> List[Tuple[List[int], List[int]]]:
        """Create splits with gaps to prevent data leakage."""
        splits = []
        
        # Use sklearn's TimeSeriesSplit with modifications
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds, gap=self.config.cv_gap)
        
        for train_indices, test_indices in tscv.split(range(data_length)):
            splits.append((train_indices.tolist(), test_indices.tolist()))
        
        logger.info(f"Created {len(splits)} gap splits")
        return splits


class WalkForwardValidator:
    """
    Walk-forward analysis for time series validation.
    
    Implements proper walk-forward testing with expanding
    or rolling windows to simulate realistic trading conditions.
    """
    
    def __init__(self, config: ValidationConfig):
        """
        Initialize walk-forward validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        
        logger.info(
            "WalkForwardValidator initialized",
            periods=config.walk_forward_periods,
            step_days=config.walk_forward_step_days,
            window_type=config.walk_forward_window_type
        )
    
    async def run_walk_forward_analysis(
        self,
        objective_function: Callable,
        parameters: Dict[str, Any],
        start_date: datetime,
        end_date: datetime
    ) -> List[Decimal]:
        """
        Run walk-forward analysis.
        
        Args:
            objective_function: Function to evaluate
            parameters: Parameters to test
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            List of performance scores for each period
        """
        period_scores = []
        current_date = start_date
        
        step_delta = timedelta(days=self.config.walk_forward_step_days)
        min_train_delta = timedelta(days=self.config.walk_forward_min_train_days)
        
        for period in range(self.config.walk_forward_periods):
            # Calculate training and testing periods
            if self.config.walk_forward_window_type == "expanding":
                train_start = start_date
            else:  # rolling
                train_start = max(start_date, current_date - min_train_delta)
            
            train_end = current_date
            test_start = current_date
            test_end = min(current_date + step_delta, end_date)
            
            if test_start >= end_date:
                break
            
            # Ensure minimum training period
            if (train_end - train_start).days < self.config.walk_forward_min_train_days:
                current_date += step_delta
                continue
            
            try:
                # Run evaluation for this period
                score = await self._evaluate_period(
                    objective_function,
                    parameters,
                    train_start,
                    train_end,
                    test_start,
                    test_end
                )
                
                if score is not None:
                    period_scores.append(score)
                    logger.debug(
                        f"Walk-forward period {period + 1} completed",
                        score=float(score),
                        train_period=f"{train_start.date()} to {train_end.date()}",
                        test_period=f"{test_start.date()} to {test_end.date()}"
                    )
                
            except Exception as e:
                logger.warning(
                    f"Walk-forward period {period + 1} failed: {str(e)}"
                )
            
            current_date += step_delta
        
        logger.info(
            f"Walk-forward analysis completed",
            periods_evaluated=len(period_scores),
            avg_score=float(sum(period_scores) / len(period_scores)) if period_scores else 0
        )
        
        return period_scores
    
    async def _evaluate_period(
        self,
        objective_function: Callable,
        parameters: Dict[str, Any],
        train_start: datetime,
        train_end: datetime,
        test_start: datetime,
        test_end: datetime
    ) -> Optional[Decimal]:
        """Evaluate parameters for a specific time period."""
        try:
            # Create period-specific parameters
            period_params = parameters.copy()
            period_params.update({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'validation_mode': 'walk_forward'
            })
            
            # Run objective function
            if asyncio.iscoroutinefunction(objective_function):
                result = await objective_function(period_params)
            else:
                result = objective_function(period_params)
            
            if isinstance(result, dict):
                # Extract primary metric
                return Decimal(str(result.get('total_return', 0)))
            else:
                return Decimal(str(result))
                
        except Exception as e:
            logger.error(f"Period evaluation failed: {str(e)}")
            return None


class OverfittingDetector:
    """
    Detects overfitting in optimization results.
    
    Uses multiple techniques to identify when optimization
    has fitted to noise rather than signal.
    """
    
    def __init__(self, config: ValidationConfig):
        """
        Initialize overfitting detector.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        
        logger.info("OverfittingDetector initialized")
    
    def detect_overfitting(
        self,
        in_sample_score: Decimal,
        out_of_sample_score: Decimal,
        validation_scores: List[Decimal]
    ) -> Tuple[bool, Dict[str, Decimal]]:
        """
        Detect overfitting using multiple metrics.
        
        Args:
            in_sample_score: In-sample performance
            out_of_sample_score: Out-of-sample performance
            validation_scores: Cross-validation scores
            
        Returns:
            Tuple of (has_overfitting, metrics_dict)
        """
        metrics = {}
        
        # Performance degradation test
        if in_sample_score != 0:
            degradation = (in_sample_score - out_of_sample_score) / abs(in_sample_score)
            metrics['performance_degradation'] = degradation
        else:
            degradation = Decimal("0")
            metrics['performance_degradation'] = degradation
        
        # Overfitting ratio test
        if out_of_sample_score != 0:
            overfitting_ratio = in_sample_score / out_of_sample_score
            metrics['overfitting_ratio'] = overfitting_ratio
        else:
            overfitting_ratio = Decimal("1")
            metrics['overfitting_ratio'] = overfitting_ratio
        
        # Validation consistency test
        if validation_scores:
            validation_mean = sum(validation_scores) / len(validation_scores)
            validation_std = self._calculate_std(validation_scores)
            
            metrics['validation_mean'] = validation_mean
            metrics['validation_std'] = validation_std
            
            # Coefficient of variation
            if validation_mean != 0:
                cv = validation_std / abs(validation_mean)
                metrics['validation_cv'] = cv
            else:
                cv = Decimal("0")
                metrics['validation_cv'] = cv
        else:
            cv = Decimal("0")
        
        # Determine overfitting
        has_overfitting = (
            degradation > self.config.overfitting_threshold or
            overfitting_ratio > (Decimal("1") + self.config.overfitting_threshold) or
            cv > Decimal("0.5")  # High variability in validation scores
        )
        
        logger.info(
            "Overfitting analysis completed",
            has_overfitting=has_overfitting,
            degradation=float(degradation),
            overfitting_ratio=float(overfitting_ratio)
        )
        
        return has_overfitting, metrics
    
    def _calculate_std(self, values: List[Decimal]) -> Decimal:
        """Calculate standard deviation of decimal values."""
        if len(values) < 2:
            return Decimal("0")
        
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        
        return Decimal(str(math.sqrt(float(variance))))


class StatisticalTester:
    """
    Statistical significance testing for optimization results.
    
    Implements various statistical tests to determine if
    optimization results are statistically significant.
    """
    
    def __init__(self, config: ValidationConfig):
        """
        Initialize statistical tester.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        
        # Set random seed for reproducibility
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
        
        logger.info(
            "StatisticalTester initialized",
            significance_level=float(config.significance_level),
            bootstrap_samples=config.bootstrap_samples
        )
    
    async def test_significance(
        self,
        optimization_results: List[Decimal],
        baseline_results: Optional[List[Decimal]] = None
    ) -> Tuple[Decimal, Tuple[Decimal, Decimal], bool]:
        """
        Test statistical significance of optimization results.
        
        Args:
            optimization_results: Results from optimized parameters
            baseline_results: Baseline results for comparison
            
        Returns:
            Tuple of (p_value, confidence_interval, is_significant)
        """
        if not optimization_results:
            return Decimal("1"), (Decimal("0"), Decimal("0")), False
        
        # Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_confidence_interval(optimization_results)
        
        # Statistical test
        if baseline_results and len(baseline_results) > 1:
            # Two-sample test
            p_value = self._two_sample_test(optimization_results, baseline_results)
        else:
            # One-sample test against zero
            p_value = self._one_sample_test(optimization_results)
        
        # Apply multiple testing correction
        corrected_p_value = self._apply_multiple_testing_correction(p_value)
        
        # Determine significance
        is_significant = corrected_p_value < self.config.significance_level
        
        logger.info(
            "Statistical significance test completed",
            p_value=float(corrected_p_value),
            is_significant=is_significant,
            confidence_interval=(float(ci_lower), float(ci_upper))
        )
        
        return corrected_p_value, (ci_lower, ci_upper), is_significant
    
    def _bootstrap_confidence_interval(
        self, 
        data: List[Decimal]
    ) -> Tuple[Decimal, Decimal]:
        """Calculate bootstrap confidence interval."""
        bootstrap_means = []
        
        for _ in range(self.config.bootstrap_samples):
            # Bootstrap sample
            sample = [random.choice(data) for _ in range(len(data))]
            bootstrap_means.append(sum(sample) / len(sample))
        
        bootstrap_means.sort()
        
        # Calculate confidence interval
        alpha = float(self.config.significance_level)
        lower_idx = int(alpha / 2 * len(bootstrap_means))
        upper_idx = int((1 - alpha / 2) * len(bootstrap_means))
        
        lower_bound = bootstrap_means[lower_idx]
        upper_bound = bootstrap_means[upper_idx]
        
        return lower_bound, upper_bound
    
    def _one_sample_test(self, data: List[Decimal]) -> Decimal:
        """Perform one-sample t-test against zero."""
        float_data = [float(d) for d in data]
        
        if len(float_data) < 2:
            return Decimal("1")
        
        # One-sample t-test
        t_stat, p_value = stats.ttest_1samp(float_data, 0)
        
        return Decimal(str(p_value))
    
    def _two_sample_test(
        self, 
        data1: List[Decimal], 
        data2: List[Decimal]
    ) -> Decimal:
        """Perform two-sample t-test."""
        float_data1 = [float(d) for d in data1]
        float_data2 = [float(d) for d in data2]
        
        if len(float_data1) < 2 or len(float_data2) < 2:
            return Decimal("1")
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(float_data1, float_data2)
        
        return Decimal(str(p_value))
    
    def _apply_multiple_testing_correction(self, p_value: Decimal) -> Decimal:
        """Apply multiple testing correction."""
        if self.config.multiple_testing_correction == "bonferroni":
            # Conservative Bonferroni correction
            # Assume testing 10 different parameter combinations
            corrected_p = p_value * 10
            return min(corrected_p, Decimal("1"))
        elif self.config.multiple_testing_correction == "holm":
            # Holm-Bonferroni method (less conservative)
            corrected_p = p_value * 5
            return min(corrected_p, Decimal("1"))
        else:
            return p_value


class RobustnessAnalyzer:
    """
    Analyzes robustness of optimization results.
    
    Tests sensitivity to parameter changes and market conditions
    to ensure results are not overly dependent on specific settings.
    """
    
    def __init__(self, config: ValidationConfig):
        """
        Initialize robustness analyzer.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        
        logger.info(
            "RobustnessAnalyzer initialized",
            perturbation_level=config.robustness_perturbation,
            samples=config.robustness_samples
        )
    
    async def analyze_robustness(
        self,
        objective_function: Callable,
        optimal_parameters: Dict[str, Any],
        parameter_space: Any  # ParameterSpace type
    ) -> Tuple[Decimal, Dict[str, Any]]:
        """
        Analyze robustness of optimal parameters.
        
        Args:
            objective_function: Function to evaluate
            optimal_parameters: Optimal parameters found
            parameter_space: Parameter space definition
            
        Returns:
            Tuple of (robustness_score, detailed_metrics)
        """
        perturbation_results = []
        parameter_sensitivities = {}
        
        # Test parameter perturbations
        for sample in range(self.config.robustness_samples):
            perturbed_params = self._perturb_parameters(
                optimal_parameters, 
                parameter_space
            )
            
            try:
                # Evaluate perturbed parameters
                if asyncio.iscoroutinefunction(objective_function):
                    result = await objective_function(perturbed_params)
                else:
                    result = objective_function(perturbed_params)
                
                if isinstance(result, dict):
                    score = Decimal(str(result.get('total_return', 0)))
                else:
                    score = Decimal(str(result))
                
                perturbation_results.append(score)
                
                # Track parameter-specific sensitivity
                for param_name, param_value in perturbed_params.items():
                    if param_name not in parameter_sensitivities:
                        parameter_sensitivities[param_name] = []
                    parameter_sensitivities[param_name].append((param_value, score))
                
            except Exception as e:
                logger.warning(f"Robustness evaluation failed: {str(e)}")
        
        # Calculate robustness metrics
        if perturbation_results:
            mean_perturbed = sum(perturbation_results) / len(perturbation_results)
            std_perturbed = self._calculate_std(perturbation_results)
            
            # Get original performance
            try:
                if asyncio.iscoroutinefunction(objective_function):
                    original_result = await objective_function(optimal_parameters)
                else:
                    original_result = objective_function(optimal_parameters)
                
                if isinstance(original_result, dict):
                    original_score = Decimal(str(original_result.get('total_return', 0)))
                else:
                    original_score = Decimal(str(original_result))
            except:
                original_score = Decimal("0")
            
            # Robustness score (higher is better)
            if original_score != 0:
                robustness_score = mean_perturbed / original_score
            else:
                robustness_score = Decimal("0")
            
            # Detailed metrics
            detailed_metrics = {
                'original_score': original_score,
                'perturbed_mean': mean_perturbed,
                'perturbed_std': std_perturbed,
                'perturbation_results': perturbation_results,
                'parameter_sensitivities': self._calculate_parameter_sensitivities(
                    parameter_sensitivities
                ),
                'worst_case_performance': min(perturbation_results),
                'best_case_performance': max(perturbation_results)
            }
        else:
            robustness_score = Decimal("0")
            detailed_metrics = {}
        
        logger.info(
            "Robustness analysis completed",
            robustness_score=float(robustness_score),
            samples_evaluated=len(perturbation_results)
        )
        
        return robustness_score, detailed_metrics
    
    def _perturb_parameters(
        self, 
        parameters: Dict[str, Any], 
        parameter_space: Any
    ) -> Dict[str, Any]:
        """Perturb parameters for robustness testing."""
        perturbed = parameters.copy()
        
        for param_name, value in parameters.items():
            if param_name in parameter_space.parameters:
                param_def = parameter_space.parameters[param_name]
                
                if param_def.parameter_type.value == "continuous":
                    # Add Gaussian noise
                    min_val, max_val = param_def.get_bounds()
                    range_size = max_val - min_val
                    noise = random.gauss(0, float(range_size) * self.config.robustness_perturbation)
                    new_value = Decimal(str(value)) + Decimal(str(noise))
                    perturbed[param_name] = param_def.clip_value(new_value)
                
                elif param_def.parameter_type.value == "discrete":
                    # Add discrete noise
                    if random.random() < self.config.robustness_perturbation:
                        valid_values = param_def.get_valid_values()
                        perturbed[param_name] = random.choice(valid_values)
                
                elif param_def.parameter_type.value == "categorical":
                    # Random category change
                    if random.random() < self.config.robustness_perturbation:
                        choices = getattr(param_def, 'choices', [])
                        perturbed[param_name] = random.choice(choices)
        
        return perturbed
    
    def _calculate_parameter_sensitivities(
        self, 
        parameter_sensitivities: Dict[str, List[Tuple[Any, Decimal]]]
    ) -> Dict[str, Decimal]:
        """Calculate sensitivity scores for each parameter."""
        sensitivities = {}
        
        for param_name, data in parameter_sensitivities.items():
            if len(data) < 2:
                sensitivities[param_name] = Decimal("0")
                continue
            
            # Calculate correlation between parameter value and performance
            values = [float(d[0]) if isinstance(d[0], (int, float, Decimal)) else 0 for d in data]
            scores = [float(d[1]) for d in data]
            
            if len(set(values)) > 1:  # Avoid division by zero
                correlation = np.corrcoef(values, scores)[0, 1]
                sensitivities[param_name] = Decimal(str(abs(correlation)))
            else:
                sensitivities[param_name] = Decimal("0")
        
        return sensitivities
    
    def _calculate_std(self, values: List[Decimal]) -> Decimal:
        """Calculate standard deviation."""
        if len(values) < 2:
            return Decimal("0")
        
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        
        return Decimal(str(math.sqrt(float(variance))))


class ValidationEngine:
    """
    Main validation engine that orchestrates all validation techniques.
    
    Provides comprehensive validation and overfitting prevention
    for optimization results.
    """
    
    def __init__(self, config: ValidationConfig):
        """
        Initialize validation engine.
        
        Args:
            config: Validation configuration
        """
        self.config = config
        
        # Initialize validators
        self.ts_validator = TimeSeriesValidator(config)
        self.wf_validator = WalkForwardValidator(config)
        self.overfitting_detector = OverfittingDetector(config)
        self.statistical_tester = StatisticalTester(config)
        self.robustness_analyzer = RobustnessAnalyzer(config)
        
        logger.info("ValidationEngine initialized with comprehensive validation suite")
    
    @time_execution
    async def validate_optimization_result(
        self,
        objective_function: Callable,
        optimal_parameters: Dict[str, Any],
        parameter_space: Any,
        optimization_history: Optional[List[Dict[str, Any]]] = None,
        data_start_date: Optional[datetime] = None,
        data_end_date: Optional[datetime] = None
    ) -> ValidationMetrics:
        """
        Perform comprehensive validation of optimization results.
        
        Args:
            objective_function: Function that was optimized
            optimal_parameters: Optimal parameters found
            parameter_space: Parameter space definition
            optimization_history: History of optimization process
            data_start_date: Start date of available data
            data_end_date: End date of available data
            
        Returns:
            Comprehensive validation metrics
        """
        logger.info("Starting comprehensive optimization validation")
        
        # 1. In-sample vs Out-of-sample evaluation
        in_sample_score, out_of_sample_score = await self._evaluate_in_out_sample(
            objective_function, optimal_parameters, data_start_date, data_end_date
        )
        
        # 2. Cross-validation
        validation_scores = await self._run_cross_validation(
            objective_function, optimal_parameters, data_start_date, data_end_date
        )
        validation_score = sum(validation_scores) / len(validation_scores) if validation_scores else Decimal("0")
        
        # 3. Walk-forward analysis
        walk_forward_scores = []
        if self.config.enable_walk_forward and data_start_date and data_end_date:
            walk_forward_scores = await self.wf_validator.run_walk_forward_analysis(
                objective_function, optimal_parameters, data_start_date, data_end_date
            )
        
        # 4. Overfitting detection
        has_overfitting, overfitting_metrics = self.overfitting_detector.detect_overfitting(
            in_sample_score, out_of_sample_score, validation_scores
        )
        
        # 5. Statistical significance testing
        optimization_scores = [out_of_sample_score] + validation_scores + walk_forward_scores
        p_value, confidence_interval, is_significant = await self.statistical_tester.test_significance(
            optimization_scores
        )
        
        # 6. Robustness analysis
        robustness_score, robustness_details = await self.robustness_analyzer.analyze_robustness(
            objective_function, optimal_parameters, parameter_space
        )
        
        # 7. Calculate stability score
        stability_score = self._calculate_stability_score(
            validation_scores, walk_forward_scores, robustness_details
        )
        
        # 8. Determine worst-case performance
        all_scores = optimization_scores
        worst_case = min(all_scores) if all_scores else Decimal("0")
        
        # 9. Create comprehensive metrics
        metrics = ValidationMetrics(
            in_sample_score=in_sample_score,
            out_of_sample_score=out_of_sample_score,
            validation_score=validation_score,
            overfitting_ratio=overfitting_metrics.get('overfitting_ratio', Decimal("1")),
            performance_degradation=overfitting_metrics.get('performance_degradation', Decimal("0")),
            p_value=p_value,
            confidence_interval=confidence_interval,
            stability_score=stability_score,
            robustness_score=robustness_score,
            worst_case_performance=worst_case,
            walk_forward_scores=walk_forward_scores,
            is_statistically_significant=is_significant,
            is_robust=robustness_score >= Decimal("0.7"),  # Threshold for robustness
            has_overfitting=has_overfitting
        )
        
        logger.info(
            "Comprehensive validation completed",
            overall_quality=float(metrics.get_overall_quality_score()),
            is_significant=is_significant,
            has_overfitting=has_overfitting,
            robustness_score=float(robustness_score)
        )
        
        return metrics
    
    async def _evaluate_in_out_sample(
        self,
        objective_function: Callable,
        parameters: Dict[str, Any],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> Tuple[Decimal, Decimal]:
        """Evaluate in-sample vs out-of-sample performance."""
        if not start_date or not end_date:
            # No date information, use full evaluation
            try:
                if asyncio.iscoroutinefunction(objective_function):
                    result = await objective_function(parameters)
                else:
                    result = objective_function(parameters)
                
                if isinstance(result, dict):
                    score = Decimal(str(result.get('total_return', 0)))
                else:
                    score = Decimal(str(result))
                
                return score, score  # Same for both if no date split
            except:
                return Decimal("0"), Decimal("0")
        
        # Calculate split point
        total_days = (end_date - start_date).days
        oos_days = int(total_days * self.config.out_of_sample_ratio)
        split_date = end_date - timedelta(days=oos_days)
        
        # In-sample evaluation
        try:
            in_sample_params = parameters.copy()
            in_sample_params.update({
                'start_date': start_date,
                'end_date': split_date,
                'validation_mode': 'in_sample'
            })
            
            if asyncio.iscoroutinefunction(objective_function):
                in_result = await objective_function(in_sample_params)
            else:
                in_result = objective_function(in_sample_params)
            
            if isinstance(in_result, dict):
                in_sample_score = Decimal(str(in_result.get('total_return', 0)))
            else:
                in_sample_score = Decimal(str(in_result))
        except Exception as e:
            logger.warning(f"In-sample evaluation failed: {str(e)}")
            in_sample_score = Decimal("0")
        
        # Out-of-sample evaluation
        try:
            oos_params = parameters.copy()
            oos_params.update({
                'start_date': split_date,
                'end_date': end_date,
                'validation_mode': 'out_of_sample'
            })
            
            if asyncio.iscoroutinefunction(objective_function):
                oos_result = await objective_function(oos_params)
            else:
                oos_result = objective_function(oos_params)
            
            if isinstance(oos_result, dict):
                out_of_sample_score = Decimal(str(oos_result.get('total_return', 0)))
            else:
                out_of_sample_score = Decimal(str(oos_result))
        except Exception as e:
            logger.warning(f"Out-of-sample evaluation failed: {str(e)}")
            out_of_sample_score = Decimal("0")
        
        return in_sample_score, out_of_sample_score
    
    async def _run_cross_validation(
        self,
        objective_function: Callable,
        parameters: Dict[str, Any],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[Decimal]:
        """Run cross-validation."""
        if not self.config.enable_cross_validation:
            return []
        
        if not start_date or not end_date:
            return []
        
        # Create time series splits
        total_days = (end_date - start_date).days
        splits = self.ts_validator.create_time_series_splits(total_days, start_date, end_date)
        
        cv_scores = []
        
        for fold, (train_indices, test_indices) in enumerate(splits):
            try:
                # Calculate dates for this fold
                train_start = start_date + timedelta(days=min(train_indices))
                train_end = start_date + timedelta(days=max(train_indices))
                test_start = start_date + timedelta(days=min(test_indices))
                test_end = start_date + timedelta(days=max(test_indices))
                
                # Evaluate fold
                fold_params = parameters.copy()
                fold_params.update({
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'validation_mode': 'cross_validation',
                    'fold': fold
                })
                
                if asyncio.iscoroutinefunction(objective_function):
                    result = await objective_function(fold_params)
                else:
                    result = objective_function(fold_params)
                
                if isinstance(result, dict):
                    score = Decimal(str(result.get('total_return', 0)))
                else:
                    score = Decimal(str(result))
                
                cv_scores.append(score)
                
                logger.debug(f"CV fold {fold + 1} completed, score: {float(score)}")
                
            except Exception as e:
                logger.warning(f"CV fold {fold + 1} failed: {str(e)}")
        
        return cv_scores
    
    def _calculate_stability_score(
        self,
        validation_scores: List[Decimal],
        walk_forward_scores: List[Decimal],
        robustness_details: Dict[str, Any]
    ) -> Decimal:
        """Calculate overall stability score."""
        stability_components = []
        
        # CV stability
        if validation_scores and len(validation_scores) > 1:
            mean_cv = sum(validation_scores) / len(validation_scores)
            std_cv = self._calculate_std(validation_scores)
            
            if mean_cv != 0:
                cv_stability = Decimal("1") / (Decimal("1") + abs(std_cv / mean_cv))
            else:
                cv_stability = Decimal("0")
            
            stability_components.append(cv_stability)
        
        # Walk-forward stability
        if walk_forward_scores and len(walk_forward_scores) > 1:
            mean_wf = sum(walk_forward_scores) / len(walk_forward_scores)
            std_wf = self._calculate_std(walk_forward_scores)
            
            if mean_wf != 0:
                wf_stability = Decimal("1") / (Decimal("1") + abs(std_wf / mean_wf))
            else:
                wf_stability = Decimal("0")
            
            stability_components.append(wf_stability)
        
        # Robustness stability
        robustness_score = robustness_details.get('robustness_score', Decimal("0"))
        if robustness_score > 0:
            stability_components.append(min(robustness_score, Decimal("1")))
        
        # Overall stability
        if stability_components:
            return sum(stability_components) / len(stability_components)
        else:
            return Decimal("0")
    
    def _calculate_std(self, values: List[Decimal]) -> Decimal:
        """Calculate standard deviation."""
        if len(values) < 2:
            return Decimal("0")
        
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        
        return Decimal(str(math.sqrt(float(variance))))