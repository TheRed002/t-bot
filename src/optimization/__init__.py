"""
Comprehensive Optimization Module for T-Bot Trading System.

This module provides advanced optimization capabilities for trading strategies,
including brute force parameter search, Bayesian optimization, genetic algorithms,
and robust overfitting prevention mechanisms.

Key Features:
- Grid search with intelligent sampling
- Bayesian optimization for efficient parameter space exploration
- Genetic algorithms for complex non-linear optimization
- Walk-forward analysis for time-series validation
- Statistical significance testing
- Overfitting prevention with multiple validation techniques
- Performance optimization with parallel processing
- Comprehensive results analysis and visualization

Critical for Financial Applications:
- Decimal precision for all calculations
- Robust validation to prevent curve fitting
- Statistical significance testing
- Multiple time period validation
- Monte Carlo robustness testing
- Proper cross-validation techniques

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
- P-007A: Utility decorators
- P-013: Evolutionary strategies (integration)
- P-016: Backtesting engine (for validation)
"""

from .brute_force import (
    BruteForceOptimizer,
    GridSearchConfig,
    OptimizationResult,
    ParameterSpace,
    ValidationConfig,
)
from .bayesian import (
    BayesianOptimizer,
    BayesianConfig,
    AcquisitionFunction,
    GaussianProcessConfig,
)
from .parameter_space import (
    ParameterDefinition,
    ParameterType,
    ContinuousParameter,
    DiscreteParameter,
    CategoricalParameter,
    ConditionalParameter,
    ParameterSpaceBuilder,
)
from .validation import (
    ValidationEngine,
    ValidationMetrics,
    ValidationConfig,
    TimeSeriesValidator,
    WalkForwardValidator,
    OverfittingDetector,
    StatisticalTester,
    RobustnessAnalyzer,
)
from .analysis import (
    ResultsAnalyzer,
    PerformanceMetrics,
    SensitivityAnalysis,
    StabilityAnalysis,
    ParameterImportanceAnalyzer,
    PerformanceAnalyzer,
)
from .core import (
    OptimizationEngine,
    OptimizationObjective,
    OptimizationConstraint,
    OptimizationStatus,
    OptimizationConfig,
    ObjectiveDirection,
)

__all__ = [
    # Core optimization
    "OptimizationEngine",
    "OptimizationObjective", 
    "OptimizationConstraint",
    "OptimizationStatus",
    "OptimizationConfig",
    "ObjectiveDirection",
    
    # Brute force optimization
    "BruteForceOptimizer",
    "GridSearchConfig",
    "OptimizationResult",
    "ParameterSpace",
    "ValidationConfig",
    
    # Bayesian optimization
    "BayesianOptimizer",
    "BayesianConfig",
    "AcquisitionFunction",
    "GaussianProcessConfig",
    
    # Parameter space definition
    "ParameterDefinition",
    "ParameterType",
    "ContinuousParameter",
    "DiscreteParameter", 
    "CategoricalParameter",
    "ConditionalParameter",
    "ParameterSpaceBuilder",
    
    # Validation and overfitting prevention
    "ValidationEngine",
    "ValidationMetrics",
    "ValidationConfig",
    "TimeSeriesValidator",
    "WalkForwardValidator",
    "OverfittingDetector",
    "StatisticalTester",
    "RobustnessAnalyzer",
    
    # Results analysis
    "ResultsAnalyzer",
    "PerformanceMetrics",
    "SensitivityAnalysis",
    "StabilityAnalysis",
    "ParameterImportanceAnalyzer",
    "PerformanceAnalyzer",
]

# Version info
__version__ = "1.0.0"
__author__ = "T-Bot Development Team"
__description__ = "Comprehensive optimization module for trading strategies"