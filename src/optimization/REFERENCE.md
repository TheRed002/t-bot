# OPTIMIZATION Module Reference

## INTEGRATION
**Dependencies**: backtesting, core, database, utils
**Used By**: None
**Provides**: AnalysisService, BacktestIntegrationService, IAnalysisService, IBacktestIntegrationService, IOptimizationService, OptimizationController, OptimizationEngine, OptimizationService, ValidationEngine
**Patterns**: Async Operations, Component Architecture, Service Layer

## DETECTED PATTERNS
**Financial**:
- Decimal precision arithmetic
- Database decimal columns
- Financial data handling
**Performance**:
- Parallel execution
- Parallel execution
**Architecture**:
- AnalysisService inherits from base architecture
- BacktestIntegrationService inherits from base architecture
- OptimizationController inherits from base architecture

## MODULE OVERVIEW
**Files**: 14 Python files
**Classes**: 58
**Functions**: 15

## COMPLETE API REFERENCE

## PROTOCOLS & INTERFACES

### Protocol: `OptimizationServiceProtocol`

**Purpose**: Protocol for optimization services

**Required Methods:**
- `async optimize_strategy(self, ...) -> dict[str, Any]`
- `async optimize_parameters(self, ...) -> OptimizationResult`

### Protocol: `BacktestIntegrationProtocol`

**Purpose**: Protocol for backtesting integration

**Required Methods:**
- `async evaluate_strategy(self, ...) -> dict[str, Decimal]`
- `create_objective_function(self, ...) -> Callable[[dict[str, Any]], Any]`

### Protocol: `OptimizationAnalysisProtocol`

**Purpose**: Protocol for optimization result analysis

**Required Methods:**
- `analyze_results(self, ...) -> dict[str, Any]`
- `calculate_parameter_importance(self, optimization_history: list[dict[str, Any]], parameter_names: list[str]) -> dict[str, Decimal]`

### Protocol: `OptimizationRepositoryProtocol`

**Purpose**: Protocol for optimization result storage

**Required Methods:**
- `async save_optimization_result(self, result: OptimizationResult, metadata: dict[str, Any] | None = None) -> str`
- `async get_optimization_result(self, optimization_id: str) -> OptimizationResult | None`
- `async list_optimization_results(self, strategy_name: str | None = None, limit: int = 100) -> list[OptimizationResult]`

### Protocol: `AnalysisServiceProtocol`

**Purpose**: Protocol for optimization analysis services

**Required Methods:**
- `async analyze_optimization_results(self, ...) -> dict[str, Any]`
- `async analyze_parameter_importance(self, optimization_results: list[dict[str, Any]], parameter_names: list[str]) -> list[Any]`

## IMPLEMENTATIONS

### Implementation: `PerformanceMetrics` ✅

**Inherits**: BaseModel
**Purpose**: Comprehensive performance metrics for trading strategies
**Status**: Complete

**Implemented Methods:**
- `get_risk_score(self) -> Decimal` - Line 100
- `get_quality_score(self) -> Decimal` - Line 109

### Implementation: `SensitivityAnalysis` ✅

**Inherits**: BaseModel
**Purpose**: Parameter sensitivity analysis results
**Status**: Complete

### Implementation: `StabilityAnalysis` ✅

**Inherits**: BaseModel
**Purpose**: Stability analysis across different conditions
**Status**: Complete

**Implemented Methods:**
- `get_overall_stability_score(self) -> Decimal` - Line 216

### Implementation: `ParameterImportanceAnalyzer` ✅

**Purpose**: Analyzes parameter importance and interactions
**Status**: Complete

**Implemented Methods:**
- `analyze_parameter_importance(self, optimization_results: list[dict[str, Any]], parameter_names: list[str]) -> list[SensitivityAnalysis]` - Line 258

### Implementation: `PerformanceAnalyzer` ✅

**Purpose**: Analyzes trading strategy performance metrics
**Status**: Complete

**Implemented Methods:**
- `calculate_performance_metrics(self, ...) -> PerformanceMetrics` - Line 523

### Implementation: `ResultsAnalyzer` ✅

**Purpose**: Main results analyzer that orchestrates all analysis components
**Status**: Complete

**Implemented Methods:**
- `analyze_optimization_results(self, ...) -> dict[str, Any]` - Line 1032

### Implementation: `AnalysisService` ✅

**Inherits**: BaseService, IAnalysisService
**Purpose**: Service for optimization result analysis
**Status**: Complete

**Implemented Methods:**
- `async analyze_optimization_results(self, ...) -> dict[str, Any]` - Line 58
- `async analyze_parameter_importance(self, optimization_results: list[dict[str, Any]], parameter_names: list[str]) -> list[Any]` - Line 104

### Implementation: `BacktestIntegrationService` ✅

**Inherits**: BaseService, IBacktestIntegrationService
**Purpose**: Service for integrating optimization with backtesting
**Status**: Complete

**Implemented Methods:**
- `async evaluate_strategy(self, ...) -> dict[str, Decimal]` - Line 56
- `create_objective_function(self, ...) -> Callable[[dict[str, Any]], Any]` - Line 103

### Implementation: `AcquisitionFunction` ✅

**Inherits**: BaseModel
**Purpose**: Acquisition function configuration for Bayesian optimization
**Status**: Complete

**Implemented Methods:**
- `validate_acquisition_function(cls, v)` - Line 97

### Implementation: `GaussianProcessConfig` ✅

**Inherits**: BaseModel
**Purpose**: Configuration for Gaussian Process surrogate model
**Status**: Complete

**Implemented Methods:**
- `validate_kernel_type(cls, v)` - Line 140
- `validate_bounds(cls, v)` - Line 149

### Implementation: `BayesianConfig` ✅

**Inherits**: BaseModel
**Purpose**: Configuration for Bayesian optimization
**Status**: Complete

**Implemented Methods:**
- `validate_batch_strategy(cls, v)` - Line 196

### Implementation: `BayesianPoint` ✅

**Inherits**: BaseModel
**Purpose**: Represents a point in the Bayesian optimization process
**Status**: Complete

**Implemented Methods:**
- `mark_evaluated(self, objective_value: Decimal, objective_std: Decimal | None = None) -> None` - Line 235

### Implementation: `GaussianProcessModel` ✅

**Purpose**: Gaussian Process surrogate model for Bayesian optimization
**Status**: Complete

**Implemented Methods:**
- `fit(self, points: list[BayesianPoint]) -> None` - Line 359
- `predict(self, parameters_list: list[dict[str, Any]], return_std: bool = True) -> tuple[np.ndarray, np.ndarray | None]` - Line 408
- `get_model_info(self) -> dict[str, Any]` - Line 448

### Implementation: `AcquisitionOptimizer` ✅

**Purpose**: Optimizes acquisition functions to select next evaluation points
**Status**: Complete

**Implemented Methods:**
- `optimize_acquisition(self, current_points: list[BayesianPoint], n_points: int = 1) -> list[dict[str, Any]]` - Line 491

### Implementation: `BayesianOptimizer` ✅

**Inherits**: OptimizationEngine
**Purpose**: Bayesian optimization engine with Gaussian Process surrogate models
**Status**: Complete

**Implemented Methods:**
- `async optimize(self, ...) -> OptimizationResult` - Line 703
- `get_next_parameters(self) -> dict[str, Any] | None` - Line 977
- `get_gp_predictions(self, parameters_list: list[dict[str, Any]]) -> tuple[list[Decimal], list[Decimal]]` - Line 985
- `get_optimization_summary(self) -> dict[str, Any]` - Line 1001

### Implementation: `GridSearchConfig` ✅

**Inherits**: BaseModel
**Purpose**: Configuration for grid search optimization
**Status**: Complete

**Implemented Methods:**
- `validate_grid_resolution(cls, v)` - Line 125

### Implementation: `OptimizationCandidate` ✅

**Inherits**: BaseModel
**Purpose**: Represents a single parameter combination candidate for evaluation
**Status**: Complete

**Implemented Methods:**
- `mark_started(self) -> None` - Line 169
- `mark_completed(self, ...) -> None` - Line 174
- `mark_failed(self, error_message: str) -> None` - Line 191

### Implementation: `GridGenerator` ✅

**Purpose**: Generates parameter grids for brute force optimization
**Status**: Complete

**Implemented Methods:**
- `generate_initial_grid(self) -> list[dict[str, Any]]` - Line 229
- `generate_refined_grid(self, best_candidates: list[OptimizationCandidate], refinement_factor: Decimal) -> list[dict[str, Any]]` - Line 436

### Implementation: `BruteForceOptimizer` ✅

**Inherits**: OptimizationEngine
**Purpose**: Brute force optimization engine with grid search and intelligent sampling
**Status**: Complete

**Implemented Methods:**
- `async optimize(self, ...) -> OptimizationResult` - Line 588
- `get_next_parameters(self) -> dict[str, Any] | None` - Line 1166

### Implementation: `OptimizationController` ✅

**Inherits**: BaseComponent
**Purpose**: Controller for optimization operations
**Status**: Complete

**Implemented Methods:**
- `async optimize_strategy(self, ...) -> dict[str, Any]` - Line 49
- `async optimize_parameters(self, ...) -> dict[str, Any]` - Line 102

### Implementation: `OptimizationStatus` ✅

**Inherits**: Enum
**Purpose**: Status enumeration for optimization processes
**Status**: Complete

### Implementation: `ObjectiveDirection` ✅

**Inherits**: Enum
**Purpose**: Direction for optimization objectives
**Status**: Complete

### Implementation: `OptimizationObjective` ✅

**Inherits**: BaseModel
**Purpose**: Optimization objective definition
**Status**: Complete

**Implemented Methods:**
- `validate_weight(cls, v: Decimal) -> Decimal` - Line 84
- `validate_constraints(cls, v: Decimal | None) -> Decimal | None` - Line 92
- `is_better(self, value1: Decimal, value2: Decimal) -> bool` - Line 97
- `satisfies_constraints(self, value: Decimal) -> bool` - Line 113
- `distance_to_target(self, value: Decimal) -> Decimal` - Line 129

### Implementation: `OptimizationConstraint` ✅

**Inherits**: BaseModel
**Purpose**: Optimization constraint definition
**Status**: Complete

**Implemented Methods:**

### Implementation: `OptimizationProgress` ✅

**Inherits**: BaseModel
**Purpose**: Progress tracking for optimization processes
**Status**: Complete

**Implemented Methods:**
- `update_progress(self, ...) -> None` - Line 224
- `add_warning(self, warning: str) -> None` - Line 257
- `estimate_completion_time(self) -> None` - Line 265

### Implementation: `OptimizationConfig` ✅

**Inherits**: BaseModel
**Purpose**: Base configuration for optimization algorithms
**Status**: Complete

### Implementation: `OptimizationResult` ✅

**Inherits**: BaseModel
**Purpose**: Result of an optimization process
**Status**: Complete

**Implemented Methods:**
- `is_statistically_significant(self, significance_level: Decimal = Any) -> bool` - Line 412
- `get_summary(self) -> dict[str, Any]` - Line 426

### Implementation: `OptimizationEngine` 🔧

**Inherits**: ABC
**Purpose**: Abstract base class for optimization engines
**Status**: Abstract Base Class

**Implemented Methods:**
- `async optimize(self, ...) -> OptimizationResult` - Line 522
- `get_next_parameters(self) -> dict[str, Any] | None` - Line 542
- `update_progress(self, ...) -> None` - Line 551
- `check_convergence(self, recent_values: list[Decimal]) -> bool` - Line 583
- `evaluate_constraints(self, parameters: dict[str, Any]) -> dict[str, Decimal]` - Line 600
- `is_feasible(self, parameters: dict[str, Any]) -> bool` - Line 620
- `get_progress(self) -> OptimizationProgress` - Line 634
- `get_result(self) -> OptimizationResult | None` - Line 638
- `async stop(self) -> None` - Line 642

### Implementation: `OptimizationFactory` ✅

**Inherits**: BaseFactory[Any]
**Purpose**: Factory for creating optimization components
**Status**: Complete

**Implemented Methods:**
- `create_complete_optimization_stack(self) -> dict[str, Any]` - Line 298

### Implementation: `OptimizationComponentFactory` ✅

**Purpose**: Composite factory for all optimization components
**Status**: Complete

**Implemented Methods:**
- `create_service(self, **kwargs) -> 'IOptimizationService'` - Line 333
- `create_controller(self, **kwargs) -> 'OptimizationController'` - Line 337
- `create_repository(self, **kwargs) -> 'OptimizationRepositoryProtocol'` - Line 341
- `create_backtest_integration(self, **kwargs) -> 'IBacktestIntegrationService'` - Line 345
- `create_analysis_service(self, **kwargs) -> 'IAnalysisService'` - Line 349
- `register_factories(self, container: Any) -> None` - Line 353

### Implementation: `IOptimizationService` 🔧

**Inherits**: ABC
**Purpose**: Abstract base class for optimization services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async optimize_strategy(self, ...) -> dict[str, Any]` - Line 118
- `async optimize_parameters(self, ...) -> OptimizationResult` - Line 130
- `async analyze_optimization_results(self, optimization_result: OptimizationResult, parameter_space: ParameterSpace) -> dict[str, Any]` - Line 142

### Implementation: `IBacktestIntegrationService` 🔧

**Inherits**: ABC
**Purpose**: Abstract base class for backtesting integration services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async evaluate_strategy(self, ...) -> dict[str, Decimal]` - Line 155
- `create_objective_function(self, ...) -> Callable[[dict[str, Any]], Any]` - Line 166

### Implementation: `IAnalysisService` 🔧

**Inherits**: ABC
**Purpose**: Abstract base class for optimization analysis services
**Status**: Abstract Base Class

**Implemented Methods:**
- `async analyze_optimization_results(self, ...) -> dict[str, Any]` - Line 181
- `async analyze_parameter_importance(self, optimization_results: list[dict[str, Any]], parameter_names: list[str]) -> list[Any]` - Line 191

### Implementation: `ParameterType` ✅

**Inherits**: Enum
**Purpose**: Parameter type enumeration
**Status**: Complete

### Implementation: `SamplingStrategy` ✅

**Inherits**: Enum
**Purpose**: Sampling strategy for parameter space exploration
**Status**: Complete

### Implementation: `ParameterDefinition` 🔧

**Inherits**: BaseModel, ABC
**Purpose**: Abstract base class for parameter definitions
**Status**: Abstract Base Class

**Implemented Methods:**
- `sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> Any` - Line 75
- `validate_value(self, value: Any) -> bool` - Line 88
- `clip_value(self, value: Any) -> Any` - Line 101
- `get_bounds(self) -> tuple[Any, Any]` - Line 114
- `is_active(self, context: dict[str, Any]) -> bool` - Line 123

### Implementation: `ContinuousParameter` ✅

**Inherits**: ParameterDefinition
**Purpose**: Continuous parameter definition for real-valued parameters
**Status**: Complete

**Implemented Methods:**
- `validate_bounds(cls, v, values)` - Line 180
- `validate_default(cls, v, values)` - Line 188
- `sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> Decimal` - Line 197
- `validate_value(self, value: Any) -> bool` - Line 272
- `clip_value(self, value: Any) -> Decimal` - Line 280
- `get_bounds(self) -> tuple[Decimal, Decimal]` - Line 288
- `get_range(self) -> Decimal` - Line 292

### Implementation: `DiscreteParameter` ✅

**Inherits**: ParameterDefinition
**Purpose**: Discrete parameter definition for integer-valued parameters
**Status**: Complete

**Implemented Methods:**
- `validate_bounds(cls, v, values)` - Line 313
- `validate_default(cls, v, values)` - Line 321
- `sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> int` - Line 336
- `validate_value(self, value: Any) -> bool` - Line 352
- `clip_value(self, value: Any) -> int` - Line 362
- `get_bounds(self) -> tuple[int, int]` - Line 381
- `get_valid_values(self) -> list[int]` - Line 385

### Implementation: `CategoricalParameter` ✅

**Inherits**: ParameterDefinition
**Purpose**: Categorical parameter definition for discrete choice parameters
**Status**: Complete

**Implemented Methods:**
- `validate_choices(cls, v)` - Line 406
- `validate_weights(cls, v, values)` - Line 416
- `validate_default(cls, v, values)` - Line 428
- `sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> Any` - Line 436
- `validate_value(self, value: Any) -> bool` - Line 445
- `clip_value(self, value: Any) -> Any` - Line 449
- `get_bounds(self) -> tuple[Any, Any]` - Line 455
- `get_choice_index(self, value: Any) -> int` - Line 459

### Implementation: `BooleanParameter` ✅

**Inherits**: ParameterDefinition
**Purpose**: Boolean parameter definition for binary choice parameters
**Status**: Complete

**Implemented Methods:**
- `sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> bool` - Line 483
- `validate_value(self, value: Any) -> bool` - Line 487
- `clip_value(self, value: Any) -> bool` - Line 491
- `get_bounds(self) -> tuple[bool, bool]` - Line 499

### Implementation: `ConditionalParameter` ✅

**Inherits**: ParameterDefinition
**Purpose**: Conditional parameter that depends on other parameters
**Status**: Complete

**Implemented Methods:**
- `sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> Any` - Line 520
- `validate_value(self, value: Any) -> bool` - Line 524
- `clip_value(self, value: Any) -> Any` - Line 528
- `get_bounds(self) -> tuple[Any, Any]` - Line 532

### Implementation: `ParameterSpace` ✅

**Inherits**: BaseModel
**Purpose**: Complete parameter space definition
**Status**: Complete

**Implemented Methods:**
- `validate_parameters(cls, v)` - Line 555
- `sample(self, ...) -> dict[str, Any]` - Line 592
- `validate_parameter_values(self, parameters: dict[str, Any]) -> dict[str, bool]` - Line 629
- `validate_parameter_set(self, parameters: dict[str, Any]) -> dict[str, bool]` - Line 650
- `clip_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]` - Line 662
- `get_active_parameters(self, context: dict[str, Any]) -> set[str]` - Line 682
- `get_bounds(self) -> dict[str, tuple[Any, Any]]` - Line 700
- `get_dimensionality(self) -> int` - Line 750
- `get_parameter_info(self) -> dict[str, dict[str, Any]]` - Line 754

### Implementation: `ParameterSpaceBuilder` ✅

**Purpose**: Builder class for constructing parameter spaces
**Status**: Complete

**Implemented Methods:**
- `add_continuous(self, ...) -> 'ParameterSpaceBuilder'` - Line 823
- `add_discrete(self, ...) -> 'ParameterSpaceBuilder'` - Line 848
- `add_categorical(self, ...) -> 'ParameterSpaceBuilder'` - Line 871
- `add_boolean(self, ...) -> 'ParameterSpaceBuilder'` - Line 892
- `add_conditional(self, ...) -> 'ParameterSpaceBuilder'` - Line 911
- `add_constraint(self, constraint: str) -> 'ParameterSpaceBuilder'` - Line 930
- `set_metadata(self, key: str, value: Any) -> 'ParameterSpaceBuilder'` - Line 935
- `build(self) -> ParameterSpace` - Line 940

### Implementation: `OptimizationRepository` ✅

**Inherits**: BaseComponent, OptimizationRepositoryProtocol
**Purpose**: Repository for optimization result persistence using database models
**Status**: Complete

**Implemented Methods:**
- `async save_optimization_result(self, result: OptimizationResult, metadata: dict[str, Any] | None = None) -> str` - Line 63
- `async get_optimization_result(self, optimization_id: str) -> OptimizationResult | None` - Line 190
- `async list_optimization_results(self, strategy_name: str | None = None, limit: int = 100, offset: int = 0) -> list[OptimizationResult]` - Line 252
- `async delete_optimization_result(self, optimization_id: str) -> bool` - Line 327
- `async save_parameter_set(self, ...) -> str` - Line 368
- `async get_parameter_sets(self, optimization_id: str, limit: int | None = None) -> list[dict[str, Any]]` - Line 441

### Implementation: `OptimizationService` ✅

**Inherits**: BaseService, IOptimizationService, ErrorPropagationMixin
**Purpose**: Main optimization service implementation
**Status**: Complete

**Implemented Methods:**
- `async optimize_strategy(self, ...) -> dict[str, Any]` - Line 86
- `async optimize_parameters(self, ...) -> OptimizationResult` - Line 273
- `async optimize_parameters_with_config(self, ...) -> dict[str, Any]` - Line 313
- `async analyze_optimization_results(self, optimization_result: OptimizationResult, parameter_space: ParameterSpace) -> dict[str, Any]` - Line 354
- `async shutdown(self) -> None` - Line 704

### Implementation: `ValidationMetrics` ✅

**Inherits**: BaseModel
**Purpose**: Comprehensive validation metrics for optimization results
**Status**: Complete

**Implemented Methods:**
- `get_overall_quality_score(self) -> Decimal` - Line 104

### Implementation: `ValidationConfig` ✅

**Inherits**: BaseModel
**Purpose**: Configuration for validation and overfitting prevention
**Status**: Complete

**Implemented Methods:**
- `validate_cv_method(cls, v)` - Line 218
- `validate_window_type(cls, v)` - Line 227
- `validate_correction_method(cls, v)` - Line 236

### Implementation: `TimeSeriesValidator` ✅

**Purpose**: Time series specific validation with proper temporal splits
**Status**: Complete

**Implemented Methods:**
- `create_time_series_splits(self, data_length: int, start_date: datetime, end_date: datetime) -> list[tuple[list[int], list[int]]]` - Line 265

### Implementation: `WalkForwardValidator` ✅

**Purpose**: Walk-forward analysis for time series validation
**Status**: Complete

**Implemented Methods:**
- `async run_walk_forward_analysis(self, ...) -> list[Decimal]` - Line 373

### Implementation: `OverfittingDetector` ✅

**Purpose**: Detects overfitting in optimization results
**Status**: Complete

**Implemented Methods:**
- `detect_overfitting(self, ...) -> tuple[bool, dict[str, Decimal]]` - Line 526

### Implementation: `StatisticalTester` ✅

**Purpose**: Statistical significance testing for optimization results
**Status**: Complete

**Implemented Methods:**
- `async test_significance(self, ...) -> tuple[Decimal, tuple[Decimal, Decimal], bool]` - Line 625

### Implementation: `RobustnessAnalyzer` ✅

**Purpose**: Analyzes robustness of optimization results
**Status**: Complete

**Implemented Methods:**
- `async analyze_robustness(self, ...) -> tuple[Decimal, dict[str, Any]]` - Line 756

### Implementation: `ValidationEngine` ✅

**Purpose**: Main validation engine that orchestrates all validation techniques
**Status**: Complete

**Implemented Methods:**
- `async validate_optimization_result(self, ...) -> ValidationMetrics` - Line 981

## COMPLETE API REFERENCE

### File: analysis.py

**Key Imports:**
- `from src.core.exceptions import DataProcessingError`
- `from src.core.logging import get_logger`
- `from src.utils.decorators import time_execution`

#### Class: `PerformanceMetrics`

**Inherits**: BaseModel
**Purpose**: Comprehensive performance metrics for trading strategies

```python
class PerformanceMetrics(BaseModel):
    def get_risk_score(self) -> Decimal  # Line 100
    def get_quality_score(self) -> Decimal  # Line 109
```

#### Class: `SensitivityAnalysis`

**Inherits**: BaseModel
**Purpose**: Parameter sensitivity analysis results

```python
class SensitivityAnalysis(BaseModel):
```

#### Class: `StabilityAnalysis`

**Inherits**: BaseModel
**Purpose**: Stability analysis across different conditions

```python
class StabilityAnalysis(BaseModel):
    def get_overall_stability_score(self) -> Decimal  # Line 216
```

#### Class: `ParameterImportanceAnalyzer`

**Purpose**: Analyzes parameter importance and interactions

```python
class ParameterImportanceAnalyzer:
    def __init__(self)  # Line 254
    def analyze_parameter_importance(self, optimization_results: list[dict[str, Any]], parameter_names: list[str]) -> list[SensitivityAnalysis]  # Line 258
    def _extract_parameter_data(self, optimization_results: list[dict[str, Any]], parameter_names: list[str]) -> dict[str, list[Decimal]]  # Line 303
    def _extract_performance_data(self, optimization_results: list[dict[str, Any]]) -> list[Decimal]  # Line 330
    def _analyze_single_parameter(self, ...) -> SensitivityAnalysis | None  # Line 353
    def _calculate_parameter_stability(self, param_values: list[Decimal], performance_values: list[Decimal]) -> Decimal  # Line 414
    def _find_interaction_partners(self, ...) -> tuple[list[str], dict[str, Decimal]]  # Line 458
```

#### Class: `PerformanceAnalyzer`

**Purpose**: Analyzes trading strategy performance metrics

```python
class PerformanceAnalyzer:
    def __init__(self, risk_free_rate: Decimal = Any)  # Line 512
    def calculate_performance_metrics(self, ...) -> PerformanceMetrics  # Line 523
    def _create_empty_metrics(self, start_date: datetime, end_date: datetime) -> PerformanceMetrics  # Line 614
    def _calculate_total_return(self, returns: list[Decimal]) -> Decimal  # Line 649
    def _calculate_annualized_return(self, total_return: Decimal, start_date: datetime, end_date: datetime) -> Decimal  # Line 661
    def _calculate_volatility(self, returns: list[Decimal]) -> Decimal  # Line 678
    def _calculate_downside_volatility(self, returns: list[Decimal]) -> Decimal  # Line 695
    def _calculate_drawdowns(self, returns: list[Decimal]) -> tuple[Decimal, Decimal]  # Line 718
    def _calculate_sharpe_ratio(self, annualized_return: Decimal, volatility: Decimal) -> Decimal  # Line 745
    def _calculate_sortino_ratio(self, annualized_return: Decimal, downside_volatility: Decimal) -> Decimal  # Line 753
    def _calculate_calmar_ratio(self, annualized_return: Decimal, max_drawdown: Decimal) -> Decimal  # Line 763
    def _calculate_omega_ratio(self, returns: list[Decimal]) -> Decimal  # Line 770
    def _calculate_trade_metrics(self, trades: list[dict[str, Any]]) -> dict[str, Decimal]  # Line 785
    def _calculate_var(self, returns: list[Decimal], confidence: Decimal) -> Decimal  # Line 831
    def _calculate_conditional_var(self, returns: list[Decimal], confidence: Decimal) -> Decimal  # Line 845
    def _calculate_skewness(self, returns: list[Decimal]) -> Decimal  # Line 860
    def _calculate_kurtosis(self, returns: list[Decimal]) -> Decimal  # Line 889
    def _calculate_recovery_factor(self, total_return: Decimal, max_drawdown: Decimal) -> Decimal  # Line 919
    def _calculate_stability_ratio(self, returns: list[Decimal]) -> Decimal  # Line 926
    def _calculate_consistency_score(self, returns: list[Decimal]) -> Decimal  # Line 973
    def _calculate_turnover_ratio(self, trades: list[dict[str, Any]], initial_capital: Decimal) -> Decimal  # Line 991
```

#### Class: `ResultsAnalyzer`

**Purpose**: Main results analyzer that orchestrates all analysis components

```python
class ResultsAnalyzer:
    def __init__(self, risk_free_rate: Decimal = Any)  # Line 1019
    def analyze_optimization_results(self, ...) -> dict[str, Any]  # Line 1032
    def _analyze_performance_distribution(self, optimization_results: list[dict[str, Any]]) -> dict[str, Any]  # Line 1103
    def _calculate_parameter_correlations(self, optimization_results: list[dict[str, Any]], parameter_names: list[str]) -> dict[str, dict[str, Decimal]]  # Line 1143
    def _analyze_optimization_landscape(self, optimization_results: list[dict[str, Any]]) -> dict[str, Any]  # Line 1196
    def _calculate_landscape_ruggedness(self, performance_values: list[float]) -> float  # Line 1230
    def _detect_multimodality(self, sorted_performance: list[float]) -> dict[str, Any]  # Line 1253
    def _calculate_convergence_rate(self, sorted_performance: list[float]) -> float  # Line 1281
    def _detect_performance_plateaus(self, sorted_performance: list[float]) -> dict[str, Any]  # Line 1300
    def _assess_improvement_potential(self, sorted_performance: list[float]) -> dict[str, Any]  # Line 1331
    def _analyze_best_result(self, best_result: dict[str, Any]) -> dict[str, Any]  # Line 1374
    def _categorize_parameter_types(self, parameters: dict[str, Any]) -> dict[str, int]  # Line 1392
    def _analyze_convergence(self, optimization_results: list[dict[str, Any]]) -> dict[str, Any]  # Line 1410
    def _create_analysis_summary(self, analysis_results: dict[str, Any], total_evaluations: int) -> dict[str, Any]  # Line 1461
```

### File: analysis_service.py

**Key Imports:**
- `from src.core.base import BaseService`
- `from src.core.exceptions import ValidationError`
- `from src.optimization.analysis import ResultsAnalyzer`
- `from src.optimization.interfaces import IAnalysisService`

#### Class: `AnalysisService`

**Inherits**: BaseService, IAnalysisService
**Purpose**: Service for optimization result analysis

```python
class AnalysisService(BaseService, IAnalysisService):
    def __init__(self, ...)  # Line 25
    async def analyze_optimization_results(self, ...) -> dict[str, Any]  # Line 58
    async def analyze_parameter_importance(self, optimization_results: list[dict[str, Any]], parameter_names: list[str]) -> list[Any]  # Line 104
```

### File: backtesting_integration.py

**Key Imports:**
- `from src.backtesting.interfaces import BacktestServiceInterface`
- `from src.backtesting.service import BacktestRequest`
- `from src.core.base import BaseService`
- `from src.core.exceptions import OptimizationError`
- `from src.core.exceptions import ServiceError`

#### Class: `BacktestIntegrationService`

**Inherits**: BaseService, IBacktestIntegrationService
**Purpose**: Service for integrating optimization with backtesting

```python
class BacktestIntegrationService(BaseService, IBacktestIntegrationService):
    def __init__(self, ...)  # Line 31
    async def evaluate_strategy(self, ...) -> dict[str, Decimal]  # Line 56
    def create_objective_function(self, ...) -> Callable[[dict[str, Any]], Any]  # Line 103
    async def _run_backtest(self, ...) -> Any  # Line 172
    def _extract_performance_metrics(self, backtest_result: Any) -> dict[str, Decimal]  # Line 229
    def _simulate_performance(self, parameters: dict[str, Any]) -> dict[str, Decimal]  # Line 274
```

### File: bayesian.py

**Key Imports:**
- `from src.core.exceptions import OptimizationError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.optimization.core import OptimizationConfig`
- `from src.optimization.core import OptimizationEngine`

#### Class: `AcquisitionFunction`

**Inherits**: BaseModel
**Purpose**: Acquisition function configuration for Bayesian optimization

```python
class AcquisitionFunction(BaseModel):
    def validate_acquisition_function(cls, v)  # Line 97
```

#### Class: `GaussianProcessConfig`

**Inherits**: BaseModel
**Purpose**: Configuration for Gaussian Process surrogate model

```python
class GaussianProcessConfig(BaseModel):
    def validate_kernel_type(cls, v)  # Line 140
    def validate_bounds(cls, v)  # Line 149
```

#### Class: `BayesianConfig`

**Inherits**: BaseModel
**Purpose**: Configuration for Bayesian optimization

```python
class BayesianConfig(BaseModel):
    def validate_batch_strategy(cls, v)  # Line 196
```

#### Class: `BayesianPoint`

**Inherits**: BaseModel
**Purpose**: Represents a point in the Bayesian optimization process

```python
class BayesianPoint(BaseModel):
    def mark_evaluated(self, objective_value: Decimal, objective_std: Decimal | None = None) -> None  # Line 235
```

#### Class: `GaussianProcessModel`

**Purpose**: Gaussian Process surrogate model for Bayesian optimization

```python
class GaussianProcessModel:
    def __init__(self, config: GaussianProcessConfig, parameter_space: ParameterSpace)  # Line 253
    def _create_kernel(self) -> Any  # Line 290
    def _encode_parameters(self, parameters_list: list[dict[str, Any]]) -> np.ndarray  # Line 323
    def fit(self, points: list[BayesianPoint]) -> None  # Line 359
    def predict(self, parameters_list: list[dict[str, Any]], return_std: bool = True) -> tuple[np.ndarray, np.ndarray | None]  # Line 408
    def get_model_info(self) -> dict[str, Any]  # Line 448
```

#### Class: `AcquisitionOptimizer`

**Purpose**: Optimizes acquisition functions to select next evaluation points

```python
class AcquisitionOptimizer:
    def __init__(self, ...)  # Line 468
    def optimize_acquisition(self, current_points: list[BayesianPoint], n_points: int = 1) -> list[dict[str, Any]]  # Line 491
    def _optimize_single_point(self, current_points: list[BayesianPoint]) -> dict[str, Any]  # Line 509
    def _optimize_batch(self, current_points: list[BayesianPoint], n_points: int) -> list[dict[str, Any]]  # Line 549
    def _decode_parameters(self, x_encoded: np.ndarray) -> dict[str, Any]  # Line 569
    def _calculate_acquisition(self, mean: float, std: float, best_value: float) -> float  # Line 599
    def _expected_improvement(self, mean: float, std: float, best_value: float) -> float  # Line 615
    def _upper_confidence_bound(self, mean: float, std: float) -> float  # Line 627
    def _probability_of_improvement(self, mean: float, std: float, best_value: float) -> float  # Line 632
    def _lower_confidence_bound(self, mean: float, std: float) -> float  # Line 643
```

#### Class: `BayesianOptimizer`

**Inherits**: OptimizationEngine
**Purpose**: Bayesian optimization engine with Gaussian Process surrogate models

```python
class BayesianOptimizer(OptimizationEngine):
    def __init__(self, ...)  # Line 657
    async def optimize(self, ...) -> OptimizationResult  # Line 703
    async def _generate_initial_points(self, ...) -> None  # Line 803
    async def _optimization_iteration(self, objective_function: Callable) -> None  # Line 831
    async def _evaluate_point(self, point: BayesianPoint, objective_function: Callable) -> None  # Line 862
    def _update_best_point(self) -> None  # Line 904
    def _check_convergence(self) -> bool  # Line 933
    async def _finalize_optimization(self) -> OptimizationResult  # Line 948
    def get_next_parameters(self) -> dict[str, Any] | None  # Line 977
    def get_gp_predictions(self, parameters_list: list[dict[str, Any]]) -> tuple[list[Decimal], list[Decimal]]  # Line 985
    def get_optimization_summary(self) -> dict[str, Any]  # Line 1001
```

### File: brute_force.py

**Key Imports:**
- `from src.core.exceptions import OptimizationError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.optimization.core import OptimizationConfig`
- `from src.optimization.core import OptimizationEngine`

#### Class: `GridSearchConfig`

**Inherits**: BaseModel
**Purpose**: Configuration for grid search optimization

```python
class GridSearchConfig(BaseModel):
    def validate_grid_resolution(cls, v)  # Line 125
```

#### Class: `OptimizationCandidate`

**Inherits**: BaseModel
**Purpose**: Represents a single parameter combination candidate for evaluation

```python
class OptimizationCandidate(BaseModel):
    def mark_started(self) -> None  # Line 169
    def mark_completed(self, ...) -> None  # Line 174
    def mark_failed(self, error_message: str) -> None  # Line 191
```

#### Class: `GridGenerator`

**Purpose**: Generates parameter grids for brute force optimization

```python
class GridGenerator:
    def __init__(self, parameter_space: ParameterSpace, config: GridSearchConfig)  # Line 210
    def generate_initial_grid(self) -> list[dict[str, Any]]  # Line 229
    def _generate_uniform_grid(self) -> list[dict[str, Any]]  # Line 249
    def _generate_random_grid(self) -> list[dict[str, Any]]  # Line 297
    def _generate_latin_hypercube(self) -> list[dict[str, Any]]  # Line 311
    def _generate_sobol_sequence(self) -> list[dict[str, Any]]  # Line 356
    def _generate_halton_sequence(self) -> list[dict[str, Any]]  # Line 396
    def generate_refined_grid(self, best_candidates: list[OptimizationCandidate], refinement_factor: Decimal) -> list[dict[str, Any]]  # Line 436
```

#### Class: `BruteForceOptimizer`

**Inherits**: OptimizationEngine
**Purpose**: Brute force optimization engine with grid search and intelligent sampling

```python
class BruteForceOptimizer(OptimizationEngine):
    def __init__(self, ...)  # Line 533
    async def optimize(self, ...) -> OptimizationResult  # Line 588
    def _create_candidates(self, parameter_combinations: list[dict[str, Any]]) -> None  # Line 686
    async def _evaluate_candidates_in_batches(self, objective_function: Callable) -> None  # Line 697
    async def _evaluate_batch(self, batch: list[OptimizationCandidate], objective_function: Callable) -> None  # Line 725
    async def _evaluate_candidate_with_semaphore(self, candidate: OptimizationCandidate, objective_function: Callable) -> None  # Line 768
    async def _evaluate_candidate(self, candidate: OptimizationCandidate, objective_function: Callable) -> None  # Line 775
    def _is_better_candidate(self, candidate: OptimizationCandidate) -> bool  # Line 843
    def _is_duplicate(self, ...) -> bool  # Line 856
    def _parameters_equal(self, params1: dict[str, Any], params2: dict[str, Any]) -> bool  # Line 875
    def _validate_candidate(self, candidate: OptimizationCandidate) -> bool  # Line 895
    async def _validate_candidate_performance(self, candidate: OptimizationCandidate, objective_function: Callable) -> Decimal | None  # Line 900
    def _perturb_parameters(self, parameters: dict[str, Any], noise_factor: float) -> dict[str, Any]  # Line 940
    def _should_stop_early(self) -> bool  # Line 973
    async def _run_adaptive_refinement(self, objective_function: Callable) -> None  # Line 994
    def _get_top_candidates(self, count: int) -> list[OptimizationCandidate]  # Line 1027
    async def _finalize_optimization(self) -> OptimizationResult  # Line 1040
    async def _calculate_statistical_significance(self) -> Decimal | None  # Line 1078
    def _analyze_parameter_stability(self) -> dict[str, Decimal]  # Line 1124
    def get_next_parameters(self) -> dict[str, Any] | None  # Line 1166
```

### File: controller.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.exceptions import ValidationError`
- `from src.optimization.interfaces import IOptimizationService`

#### Class: `OptimizationController`

**Inherits**: BaseComponent
**Purpose**: Controller for optimization operations

```python
class OptimizationController(BaseComponent):
    def __init__(self, ...)  # Line 25
    async def optimize_strategy(self, ...) -> dict[str, Any]  # Line 49
    async def optimize_parameters(self, ...) -> dict[str, Any]  # Line 102
    def _validate_strategy_optimization_request(self, ...) -> None  # Line 139
    def _validate_parameter_optimization_request(self, ...) -> None  # Line 158
```

### File: core.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.core.types import TradingMode`

#### Class: `OptimizationStatus`

**Inherits**: Enum
**Purpose**: Status enumeration for optimization processes

```python
class OptimizationStatus(Enum):
```

#### Class: `ObjectiveDirection`

**Inherits**: Enum
**Purpose**: Direction for optimization objectives

```python
class ObjectiveDirection(Enum):
```

#### Class: `OptimizationObjective`

**Inherits**: BaseModel
**Purpose**: Optimization objective definition

```python
class OptimizationObjective(BaseModel):
    def validate_weight(cls, v: Decimal) -> Decimal  # Line 84
    def validate_constraints(cls, v: Decimal | None) -> Decimal | None  # Line 92
    def is_better(self, value1: Decimal, value2: Decimal) -> bool  # Line 97
    def satisfies_constraints(self, value: Decimal) -> bool  # Line 113
    def distance_to_target(self, value: Decimal) -> Decimal  # Line 129
```

#### Class: `OptimizationConstraint`

**Inherits**: BaseModel
**Purpose**: Optimization constraint definition

```python
class OptimizationConstraint(BaseModel):
    def __init__(self, **data)  # Line 170
```

#### Class: `OptimizationProgress`

**Inherits**: BaseModel
**Purpose**: Progress tracking for optimization processes

```python
class OptimizationProgress(BaseModel):
    def update_progress(self, ...) -> None  # Line 224
    def add_warning(self, warning: str) -> None  # Line 257
    def estimate_completion_time(self) -> None  # Line 265
```

#### Class: `OptimizationConfig`

**Inherits**: BaseModel
**Purpose**: Base configuration for optimization algorithms

```python
class OptimizationConfig(BaseModel):
```

#### Class: `OptimizationResult`

**Inherits**: BaseModel
**Purpose**: Result of an optimization process

```python
class OptimizationResult(BaseModel):
    def is_statistically_significant(self, significance_level: Decimal = Any) -> bool  # Line 412
    def get_summary(self) -> dict[str, Any]  # Line 426
```

#### Class: `OptimizationEngine`

**Inherits**: ABC
**Purpose**: Abstract base class for optimization engines

```python
class OptimizationEngine(ABC):
    def __init__(self, ...)  # Line 455
    def _validate_configuration(self) -> None  # Line 498
    async def optimize(self, ...) -> OptimizationResult  # Line 522
    def get_next_parameters(self) -> dict[str, Any] | None  # Line 542
    def update_progress(self, ...) -> None  # Line 551
    def check_convergence(self, recent_values: list[Decimal]) -> bool  # Line 583
    def evaluate_constraints(self, parameters: dict[str, Any]) -> dict[str, Decimal]  # Line 600
    def is_feasible(self, parameters: dict[str, Any]) -> bool  # Line 620
    def get_progress(self) -> OptimizationProgress  # Line 634
    def get_result(self) -> OptimizationResult | None  # Line 638
    async def stop(self) -> None  # Line 642
    def _get_primary_objective_value(self, objective_values: dict[str, Decimal]) -> Decimal  # Line 647
    async def _run_objective_function(self, ...) -> Any  # Line 669
```

#### Functions:

```python
def create_profit_maximization_objective() -> OptimizationObjective  # Line 717
def create_risk_minimization_objective() -> OptimizationObjective  # Line 729
def create_sharpe_ratio_objective() -> OptimizationObjective  # Line 741
def create_standard_trading_objectives() -> list[OptimizationObjective]  # Line 753
```

### File: di_registration.py

**Key Imports:**
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.logging import get_logger`

#### Functions:

```python
def register_optimization_dependencies(injector: DependencyInjector) -> None  # Line 25
def configure_optimization_module(injector: DependencyInjector, config: dict[str, Any] | None = None) -> None  # Line 134
def get_optimization_service(injector: DependencyInjector) -> 'IOptimizationService'  # Line 154
def get_optimization_controller(injector: DependencyInjector)  # Line 159
def get_optimization_repository(injector: DependencyInjector)  # Line 164
```

### File: factory.py

**Key Imports:**
- `from src.core.base.factory import BaseFactory`
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.exceptions import FactoryError`
- `from src.core.logging import get_logger`

#### Class: `OptimizationFactory`

**Inherits**: BaseFactory[Any]
**Purpose**: Factory for creating optimization components

```python
class OptimizationFactory(BaseFactory[Any]):
    def __init__(self, ...)  # Line 36
    def _create_optimization_repository(self, database_session: Any = None, **kwargs) -> 'OptimizationRepositoryProtocol'  # Line 70
    def _create_backtest_integration(self, backtest_service: Any = None, **kwargs) -> 'IBacktestIntegrationService'  # Line 112
    def _create_analysis_service(self, results_analyzer: Any = None, **kwargs) -> 'IAnalysisService'  # Line 150
    def _create_optimization_service(self, ...) -> 'IOptimizationService'  # Line 188
    def _create_optimization_controller(self, optimization_service: Any = None, **kwargs) -> 'OptimizationController'  # Line 256
    def create_complete_optimization_stack(self) -> dict[str, Any]  # Line 298
```

#### Class: `OptimizationComponentFactory`

**Purpose**: Composite factory for all optimization components

```python
class OptimizationComponentFactory:
    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None)  # Line 329
    def create_service(self, **kwargs) -> 'IOptimizationService'  # Line 333
    def create_controller(self, **kwargs) -> 'OptimizationController'  # Line 337
    def create_repository(self, **kwargs) -> 'OptimizationRepositoryProtocol'  # Line 341
    def create_backtest_integration(self, **kwargs) -> 'IBacktestIntegrationService'  # Line 345
    def create_analysis_service(self, **kwargs) -> 'IAnalysisService'  # Line 349
    def register_factories(self, container: Any) -> None  # Line 353
```

#### Functions:

```python
def create_optimization_service(injector: DependencyInjector | None = None) -> 'IOptimizationService'  # Line 363
def create_optimization_controller(injector: DependencyInjector | None = None) -> 'OptimizationController'  # Line 371
def create_optimization_stack(injector: DependencyInjector | None = None) -> dict[str, Any]  # Line 379
```

### File: interfaces.py

**Key Imports:**
- `from src.core.types import StrategyConfig`
- `from src.optimization.core import OptimizationObjective`
- `from src.optimization.core import OptimizationResult`
- `from src.optimization.parameter_space import ParameterSpace`

#### Class: `IOptimizationService`

**Inherits**: ABC
**Purpose**: Abstract base class for optimization services

```python
class IOptimizationService(ABC):
    async def optimize_strategy(self, ...) -> dict[str, Any]  # Line 118
    async def optimize_parameters(self, ...) -> OptimizationResult  # Line 130
    async def analyze_optimization_results(self, optimization_result: OptimizationResult, parameter_space: ParameterSpace) -> dict[str, Any]  # Line 142
```

#### Class: `IBacktestIntegrationService`

**Inherits**: ABC
**Purpose**: Abstract base class for backtesting integration services

```python
class IBacktestIntegrationService(ABC):
    async def evaluate_strategy(self, ...) -> dict[str, Decimal]  # Line 155
    def create_objective_function(self, ...) -> Callable[[dict[str, Any]], Any]  # Line 166
```

#### Class: `IAnalysisService`

**Inherits**: ABC
**Purpose**: Abstract base class for optimization analysis services

```python
class IAnalysisService(ABC):
    async def analyze_optimization_results(self, ...) -> dict[str, Any]  # Line 181
    async def analyze_parameter_importance(self, optimization_results: list[dict[str, Any]], parameter_names: list[str]) -> list[Any]  # Line 191
```

### File: parameter_space.py

#### Class: `ParameterType`

**Inherits**: Enum
**Purpose**: Parameter type enumeration

```python
class ParameterType(Enum):
```

#### Class: `SamplingStrategy`

**Inherits**: Enum
**Purpose**: Sampling strategy for parameter space exploration

```python
class SamplingStrategy(Enum):
```

#### Class: `ParameterDefinition`

**Inherits**: BaseModel, ABC
**Purpose**: Abstract base class for parameter definitions

```python
class ParameterDefinition(BaseModel, ABC):
    def sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> Any  # Line 75
    def validate_value(self, value: Any) -> bool  # Line 88
    def clip_value(self, value: Any) -> Any  # Line 101
    def get_bounds(self) -> tuple[Any, Any]  # Line 114
    def is_active(self, context: dict[str, Any]) -> bool  # Line 123
```

#### Class: `ContinuousParameter`

**Inherits**: ParameterDefinition
**Purpose**: Continuous parameter definition for real-valued parameters

```python
class ContinuousParameter(ParameterDefinition):
    def validate_bounds(cls, v, values)  # Line 180
    def validate_default(cls, v, values)  # Line 188
    def sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> Decimal  # Line 197
    def validate_value(self, value: Any) -> bool  # Line 272
    def clip_value(self, value: Any) -> Decimal  # Line 280
    def get_bounds(self) -> tuple[Decimal, Decimal]  # Line 288
    def get_range(self) -> Decimal  # Line 292
```

#### Class: `DiscreteParameter`

**Inherits**: ParameterDefinition
**Purpose**: Discrete parameter definition for integer-valued parameters

```python
class DiscreteParameter(ParameterDefinition):
    def validate_bounds(cls, v, values)  # Line 313
    def validate_default(cls, v, values)  # Line 321
    def sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> int  # Line 336
    def validate_value(self, value: Any) -> bool  # Line 352
    def clip_value(self, value: Any) -> int  # Line 362
    def get_bounds(self) -> tuple[int, int]  # Line 381
    def get_valid_values(self) -> list[int]  # Line 385
```

#### Class: `CategoricalParameter`

**Inherits**: ParameterDefinition
**Purpose**: Categorical parameter definition for discrete choice parameters

```python
class CategoricalParameter(ParameterDefinition):
    def validate_choices(cls, v)  # Line 406
    def validate_weights(cls, v, values)  # Line 416
    def validate_default(cls, v, values)  # Line 428
    def sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> Any  # Line 436
    def validate_value(self, value: Any) -> bool  # Line 445
    def clip_value(self, value: Any) -> Any  # Line 449
    def get_bounds(self) -> tuple[Any, Any]  # Line 455
    def get_choice_index(self, value: Any) -> int  # Line 459
```

#### Class: `BooleanParameter`

**Inherits**: ParameterDefinition
**Purpose**: Boolean parameter definition for binary choice parameters

```python
class BooleanParameter(ParameterDefinition):
    def sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> bool  # Line 483
    def validate_value(self, value: Any) -> bool  # Line 487
    def clip_value(self, value: Any) -> bool  # Line 491
    def get_bounds(self) -> tuple[bool, bool]  # Line 499
```

#### Class: `ConditionalParameter`

**Inherits**: ParameterDefinition
**Purpose**: Conditional parameter that depends on other parameters

```python
class ConditionalParameter(ParameterDefinition):
    def __init__(self, **data)  # Line 515
    def sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> Any  # Line 520
    def validate_value(self, value: Any) -> bool  # Line 524
    def clip_value(self, value: Any) -> Any  # Line 528
    def get_bounds(self) -> tuple[Any, Any]  # Line 532
```

#### Class: `ParameterSpace`

**Inherits**: BaseModel
**Purpose**: Complete parameter space definition

```python
class ParameterSpace(BaseModel):
    def validate_parameters(cls, v)  # Line 555
    def sample(self, ...) -> dict[str, Any]  # Line 592
    def validate_parameter_values(self, parameters: dict[str, Any]) -> dict[str, bool]  # Line 629
    def validate_parameter_set(self, parameters: dict[str, Any]) -> dict[str, bool]  # Line 650
    def clip_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]  # Line 662
    def get_active_parameters(self, context: dict[str, Any]) -> set[str]  # Line 682
    def get_bounds(self) -> dict[str, tuple[Any, Any]]  # Line 700
    def _topological_sort(self) -> list[str]  # Line 714
    def get_dimensionality(self) -> int  # Line 750
    def get_parameter_info(self) -> dict[str, dict[str, Any]]  # Line 754
```

#### Class: `ParameterSpaceBuilder`

**Purpose**: Builder class for constructing parameter spaces

```python
class ParameterSpaceBuilder:
    def __init__(self)  # Line 817
    def add_continuous(self, ...) -> 'ParameterSpaceBuilder'  # Line 823
    def add_discrete(self, ...) -> 'ParameterSpaceBuilder'  # Line 848
    def add_categorical(self, ...) -> 'ParameterSpaceBuilder'  # Line 871
    def add_boolean(self, ...) -> 'ParameterSpaceBuilder'  # Line 892
    def add_conditional(self, ...) -> 'ParameterSpaceBuilder'  # Line 911
    def add_constraint(self, constraint: str) -> 'ParameterSpaceBuilder'  # Line 930
    def set_metadata(self, key: str, value: Any) -> 'ParameterSpaceBuilder'  # Line 935
    def build(self) -> ParameterSpace  # Line 940
```

#### Functions:

```python
def create_trading_strategy_space() -> ParameterSpace  # Line 950
def create_ml_model_space() -> ParameterSpace  # Line 1012
def create_risk_management_space() -> ParameterSpace  # Line 1087
```

### File: service.py

**Key Imports:**
- `from src.core.base import BaseService`
- `from src.core.event_constants import OptimizationEvents`
- `from src.core.exceptions import OptimizationError`
- `from src.core.exceptions import ValidationError`
- `from src.optimization.bayesian import BayesianConfig`

#### Class: `OptimizationService`

**Inherits**: BaseService, IOptimizationService, ErrorPropagationMixin
**Purpose**: Main optimization service implementation

```python
class OptimizationService(BaseService, IOptimizationService, ErrorPropagationMixin):
    def __init__(self, ...)  # Line 40
    async def optimize_strategy(self, ...) -> dict[str, Any]  # Line 86
    async def optimize_parameters(self, ...) -> OptimizationResult  # Line 273
    async def optimize_parameters_with_config(self, ...) -> dict[str, Any]  # Line 313
    async def analyze_optimization_results(self, optimization_result: OptimizationResult, parameter_space: ParameterSpace) -> dict[str, Any]  # Line 354
    async def _create_objective_function(self, ...) -> Callable[[dict[str, Any]], Any]  # Line 412
    def _create_simulation_objective_function(self, ...) -> Callable[[dict[str, Any]], Any]  # Line 444
    def _create_standard_trading_objectives(self) -> list[OptimizationObjective]  # Line 487
    async def _create_brute_force_optimizer(self, ...) -> Any  # Line 523
    async def _create_bayesian_optimizer(self, ...) -> Any  # Line 564
    def _build_parameter_space(self, config: dict[str, Any]) -> ParameterSpace  # Line 606
    def _build_objectives(self, objectives_config: list[dict[str, Any]]) -> list[OptimizationObjective]  # Line 649
    def _create_objective_function_by_name(self, function_name: str) -> Callable[[dict[str, Any]], Any]  # Line 689
    async def shutdown(self) -> None  # Line 704
```

### File: validation.py

**Key Imports:**
- `from src.core.exceptions import OptimizationError`
- `from src.core.logging import get_logger`
- `from src.utils.decorators import time_execution`
- `from src.utils.financial_calculations import calculate_std`

#### Class: `ValidationMetrics`

**Inherits**: BaseModel
**Purpose**: Comprehensive validation metrics for optimization results

```python
class ValidationMetrics(BaseModel):
    def get_overall_quality_score(self) -> Decimal  # Line 104
```

#### Class: `ValidationConfig`

**Inherits**: BaseModel
**Purpose**: Configuration for validation and overfitting prevention

```python
class ValidationConfig(BaseModel):
    def validate_cv_method(cls, v)  # Line 218
    def validate_window_type(cls, v)  # Line 227
    def validate_correction_method(cls, v)  # Line 236
```

#### Class: `TimeSeriesValidator`

**Purpose**: Time series specific validation with proper temporal splits

```python
class TimeSeriesValidator:
    def __init__(self, config: ValidationConfig)  # Line 252
    def create_time_series_splits(self, data_length: int, start_date: datetime, end_date: datetime) -> list[tuple[list[int], list[int]]]  # Line 265
    def _create_expanding_window_splits(self, data_length: int) -> list[tuple[list[int], list[int]]]  # Line 288
    def _create_blocked_splits(self, data_length: int) -> list[tuple[list[int], list[int]]]  # Line 316
    def _create_gap_splits(self, data_length: int) -> list[tuple[list[int], list[int]]]  # Line 335
```

#### Class: `WalkForwardValidator`

**Purpose**: Walk-forward analysis for time series validation

```python
class WalkForwardValidator:
    def __init__(self, config: ValidationConfig)  # Line 357
    async def run_walk_forward_analysis(self, ...) -> list[Decimal]  # Line 373
    async def _evaluate_period(self, ...) -> Decimal | None  # Line 449
```

#### Class: `OverfittingDetector`

**Purpose**: Detects overfitting in optimization results

```python
class OverfittingDetector:
    def __init__(self, config: ValidationConfig)  # Line 515
    def detect_overfitting(self, ...) -> tuple[bool, dict[str, Decimal]]  # Line 526
```

#### Class: `StatisticalTester`

**Purpose**: Statistical significance testing for optimization results

```python
class StatisticalTester:
    def __init__(self, config: ValidationConfig)  # Line 605
    async def test_significance(self, ...) -> tuple[Decimal, tuple[Decimal, Decimal], bool]  # Line 625
    def _bootstrap_confidence_interval(self, data: list[Decimal]) -> tuple[Decimal, Decimal]  # Line 672
    def _one_sample_test(self, data: list[Decimal]) -> Decimal  # Line 693
    def _two_sample_test(self, data1: list[Decimal], data2: list[Decimal]) -> Decimal  # Line 705
    def _apply_multiple_testing_correction(self, p_value: Decimal) -> Decimal  # Line 718
```

#### Class: `RobustnessAnalyzer`

**Purpose**: Analyzes robustness of optimization results

```python
class RobustnessAnalyzer:
    def __init__(self, config: ValidationConfig)  # Line 741
    async def analyze_robustness(self, ...) -> tuple[Decimal, dict[str, Any]]  # Line 756
    def _perturb_parameters(self, parameters: dict[str, Any], parameter_space: Any) -> dict[str, Any]  # Line 896
    def _calculate_parameter_sensitivities(self, parameter_sensitivities: dict[str, list[tuple[Any, Decimal]]]) -> dict[str, Decimal]  # Line 930
```

#### Class: `ValidationEngine`

**Purpose**: Main validation engine that orchestrates all validation techniques

```python
class ValidationEngine:
    def __init__(self, config: ValidationConfig)  # Line 962
    async def validate_optimization_result(self, ...) -> ValidationMetrics  # Line 981
    async def _evaluate_in_out_sample(self, ...) -> tuple[Decimal, Decimal]  # Line 1105
    async def _run_cross_validation(self, ...) -> list[Decimal]  # Line 1221
    def _calculate_stability_score(self, ...) -> Decimal  # Line 1292
```

---
**Generated**: Complete reference for optimization module
**Total Classes**: 58
**Total Functions**: 15