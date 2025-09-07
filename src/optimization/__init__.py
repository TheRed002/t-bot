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

from .analysis import (
    ParameterImportanceAnalyzer,
    PerformanceAnalyzer,
    PerformanceMetrics,
    ResultsAnalyzer,
    SensitivityAnalysis,
    StabilityAnalysis,
)
from .backtesting_integration import BacktestIntegrationService
from .bayesian import (
    AcquisitionFunction,
    BayesianConfig,
    BayesianOptimizer,
    GaussianProcessConfig,
)
from .brute_force import (
    BruteForceOptimizer,
    GridSearchConfig,
    OptimizationResult,
    ParameterSpace,
    ValidationConfig as BruteForceValidationConfig,
)
from .controller import OptimizationController
from .core import (
    ObjectiveDirection,
    OptimizationConfig,
    OptimizationConstraint,
    OptimizationEngine,
    OptimizationObjective,
    OptimizationStatus,
)
from .factory import (
    OptimizationFactory,
    create_optimization_controller,
    create_optimization_service,
    create_optimization_stack,
)
from .interfaces import (
    BacktestIntegrationProtocol,
    IBacktestIntegrationService,
    IOptimizationService,
    OptimizationAnalysisProtocol,
    OptimizationRepositoryProtocol,
    OptimizationServiceProtocol,
)
from .parameter_space import (
    CategoricalParameter,
    ConditionalParameter,
    ContinuousParameter,
    DiscreteParameter,
    ParameterDefinition,
    ParameterSpaceBuilder,
    ParameterType,
)
from .repository import OptimizationRepository
from .service import OptimizationService
from .validation import (
    OverfittingDetector,
    RobustnessAnalyzer,
    StatisticalTester,
    TimeSeriesValidator,
    ValidationConfig as OptimizationValidationConfig,
    ValidationEngine,
    ValidationMetrics,
    WalkForwardValidator,
)

__all__ = [
    "AcquisitionFunction",
    "BacktestIntegrationProtocol",
    "BacktestIntegrationService",
    "BayesianConfig",
    "BayesianOptimizer",
    "BruteForceOptimizer",
    "BruteForceValidationConfig",
    "CategoricalParameter",
    "ConditionalParameter",
    "ContinuousParameter",
    "DiscreteParameter",
    "GaussianProcessConfig",
    "GridSearchConfig",
    "IBacktestIntegrationService",
    "IOptimizationService",
    "ObjectiveDirection",
    "OptimizationAnalysisProtocol",
    "OptimizationConfig",
    "OptimizationConstraint",
    "OptimizationController",
    "OptimizationEngine",
    "OptimizationFactory",
    "OptimizationObjective",
    "OptimizationRepository",
    "OptimizationRepositoryProtocol",
    "OptimizationResult",
    "OptimizationService",
    "OptimizationServiceProtocol",
    "OptimizationStatus",
    "OptimizationValidationConfig",
    "OverfittingDetector",
    "ParameterDefinition",
    "ParameterImportanceAnalyzer",
    "ParameterSpace",
    "ParameterSpaceBuilder",
    "ParameterType",
    "PerformanceAnalyzer",
    "PerformanceMetrics",
    "ResultsAnalyzer",
    "RobustnessAnalyzer",
    "SensitivityAnalysis",
    "StabilityAnalysis",
    "StatisticalTester",
    "TimeSeriesValidator",
    "ValidationEngine",
    "ValidationMetrics",
    "WalkForwardValidator",
    "create_optimization_controller",
    "create_optimization_service",
    "create_optimization_stack",
]

# Version info
__version__ = "1.0.0"
__author__ = "T-Bot Development Team"
__description__ = "Comprehensive optimization module for trading strategies"
