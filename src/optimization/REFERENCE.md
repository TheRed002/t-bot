# OPTIMIZATION Module Reference

## INTEGRATION
**Dependencies**: backtesting, core, database, utils
**Used By**: None
**Provides**: AnalysisService, BacktestIntegrationService, IAnalysisService, IBacktestIntegrationService, IOptimizationService, OptimizationController, OptimizationEngine, OptimizationService, ParameterSpaceService, ResultTransformationService, ValidationEngine
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
**Files**: 17 Python files
**Classes**: 61
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
- `get_risk_score(self) -> Decimal` - Line 115
- `get_quality_score(self) -> Decimal` - Line 124

### Implementation: `SensitivityAnalysis` ✅

**Inherits**: BaseModel
**Purpose**: Parameter sensitivity analysis results
**Status**: Complete

### Implementation: `StabilityAnalysis` ✅

**Inherits**: BaseModel
**Purpose**: Stability analysis across different conditions
**Status**: Complete

**Implemented Methods:**
- `get_overall_stability_score(self) -> Decimal` - Line 231

### Implementation: `ParameterImportanceAnalyzer` ✅

**Purpose**: Analyzes parameter importance and interactions
**Status**: Complete

**Implemented Methods:**
- `analyze_parameter_importance(self, optimization_results: list[dict[str, Any]], parameter_names: list[str]) -> list[SensitivityAnalysis]` - Line 273

### Implementation: `PerformanceAnalyzer` ✅

**Purpose**: Analyzes trading strategy performance metrics
**Status**: Complete

**Implemented Methods:**
- `calculate_performance_metrics(self, ...) -> PerformanceMetrics` - Line 543

### Implementation: `ResultsAnalyzer` ✅

**Purpose**: Main results analyzer that orchestrates all analysis components
**Status**: Complete

**Implemented Methods:**
- `analyze_optimization_results(self, ...) -> dict[str, Any]` - Line 1113

### Implementation: `AnalysisService` ✅

**Inherits**: BaseService, IAnalysisService
**Purpose**: Service for optimization result analysis
**Status**: Complete

**Implemented Methods:**
- `async analyze_optimization_results(self, ...) -> dict[str, Any]` - Line 60
- `async analyze_parameter_importance(self, optimization_results: list[dict[str, Any]], parameter_names: list[str]) -> list[Any]` - Line 105

### Implementation: `BacktestIntegrationService` ✅

**Inherits**: BaseService, IBacktestIntegrationService
**Purpose**: Service for integrating optimization with backtesting
**Status**: Complete

**Implemented Methods:**
- `async evaluate_strategy(self, ...) -> dict[str, Decimal]` - Line 57
- `create_objective_function(self, ...) -> Callable[[dict[str, Any]], Any]` - Line 119

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
- `validate_kernel_type(cls, v)` - Line 145
- `validate_bounds(cls, v)` - Line 154

### Implementation: `BayesianConfig` ✅

**Inherits**: BaseModel
**Purpose**: Configuration for Bayesian optimization
**Status**: Complete

**Implemented Methods:**
- `validate_batch_strategy(cls, v)` - Line 201

### Implementation: `BayesianPoint` ✅

**Inherits**: BaseModel
**Purpose**: Represents a point in the Bayesian optimization process
**Status**: Complete

**Implemented Methods:**
- `mark_evaluated(self, objective_value: Decimal, objective_std: Decimal | None = None) -> None` - Line 240

### Implementation: `GaussianProcessModel` ✅

**Purpose**: Gaussian Process surrogate model for Bayesian optimization
**Status**: Complete

**Implemented Methods:**
- `fit(self, points: list[BayesianPoint]) -> None` - Line 364
- `predict(self, parameters_list: list[dict[str, Any]], return_std: bool = True) -> tuple[np.ndarray, np.ndarray | None]` - Line 413
- `get_model_info(self) -> dict[str, Any]` - Line 453

### Implementation: `AcquisitionOptimizer` ✅

**Purpose**: Optimizes acquisition functions to select next evaluation points
**Status**: Complete

**Implemented Methods:**
- `optimize_acquisition(self, current_points: list[BayesianPoint], n_points: int = 1) -> list[dict[str, Any]]` - Line 496

### Implementation: `BayesianOptimizer` ✅

**Inherits**: OptimizationEngine
**Purpose**: Bayesian optimization engine with Gaussian Process surrogate models
**Status**: Complete

**Implemented Methods:**
- `async optimize(self, ...) -> OptimizationResult` - Line 708
- `get_next_parameters(self) -> dict[str, Any] | None` - Line 982
- `get_gp_predictions(self, parameters_list: list[dict[str, Any]]) -> tuple[list[Decimal], list[Decimal]]` - Line 990
- `get_optimization_summary(self) -> dict[str, Any]` - Line 1006

### Implementation: `GridSearchConfig` ✅

**Inherits**: BaseModel
**Purpose**: Configuration for grid search optimization
**Status**: Complete

**Implemented Methods:**
- `validate_grid_resolution(cls, v)` - Line 132

### Implementation: `OptimizationCandidate` ✅

**Inherits**: BaseModel
**Purpose**: Represents a single parameter combination candidate for evaluation
**Status**: Complete

**Implemented Methods:**
- `mark_started(self) -> None` - Line 176
- `mark_completed(self, ...) -> None` - Line 181
- `mark_failed(self, error_message: str) -> None` - Line 198

### Implementation: `GridGenerator` ✅

**Purpose**: Generates parameter grids for brute force optimization
**Status**: Complete

**Implemented Methods:**
- `generate_initial_grid(self) -> list[dict[str, Any]]` - Line 236
- `generate_refined_grid(self, best_candidates: list[OptimizationCandidate], refinement_factor: Decimal) -> list[dict[str, Any]]` - Line 443

### Implementation: `BruteForceOptimizer` ✅

**Inherits**: OptimizationEngine
**Purpose**: Brute force optimization engine with grid search and intelligent sampling
**Status**: Complete

**Implemented Methods:**
- `async optimize(self, ...) -> OptimizationResult` - Line 595
- `get_next_parameters(self) -> dict[str, Any] | None` - Line 1181

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
- `estimate_completion_time(self) -> None` - Line 263

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

### Implementation: `OptimizationDataTransformer` ✅

**Purpose**: Handles consistent data transformation for optimization module
**Status**: Complete

**Implemented Methods:**
- `transform_optimization_result_to_event_data(result: OptimizationResult, metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 23
- `transform_parameter_set_to_event_data(parameters: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 63
- `transform_objective_values_to_event_data(objective_values: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]` - Line 91
- `validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]` - Line 114
- `ensure_boundary_fields(data: dict[str, Any], source: str = 'optimization') -> dict[str, Any]` - Line 154
- `transform_for_req_reply(cls, request_type: str, data: Any, correlation_id: str | None = None) -> dict[str, Any]` - Line 181
- `transform_for_batch_processing(cls, ...) -> dict[str, Any]` - Line 236
- `align_processing_paradigm(cls, data: dict[str, Any], target_mode: str = 'batch') -> dict[str, Any]` - Line 301
- `apply_cross_module_validation(cls, ...) -> dict[str, Any]` - Line 380

### Implementation: `OptimizationFactory` ✅

**Inherits**: BaseFactory[Any]
**Purpose**: Factory for creating optimization components
**Status**: Complete

**Implemented Methods:**
- `create_complete_optimization_stack(self) -> dict[str, Any]` - Line 292

### Implementation: `OptimizationComponentFactory` ✅

**Purpose**: Composite factory for all optimization components
**Status**: Complete

**Implemented Methods:**
- `create_service(self, **kwargs) -> 'IOptimizationService'` - Line 325
- `create_controller(self, **kwargs) -> 'OptimizationController'` - Line 329
- `create_repository(self, **kwargs) -> 'OptimizationRepositoryProtocol'` - Line 333
- `create_backtest_integration(self, **kwargs) -> 'IBacktestIntegrationService'` - Line 337
- `create_analysis_service(self, **kwargs) -> 'IAnalysisService'` - Line 341
- `register_factories(self, container: Any) -> None` - Line 345

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
- `sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> Any` - Line 74
- `validate_value(self, value: Any) -> bool` - Line 87
- `clip_value(self, value: Any) -> Any` - Line 100
- `get_bounds(self) -> tuple[Any, Any]` - Line 113
- `is_active(self, context: dict[str, Any]) -> bool` - Line 122

### Implementation: `ContinuousParameter` ✅

**Inherits**: ParameterDefinition
**Purpose**: Continuous parameter definition for real-valued parameters
**Status**: Complete

**Implemented Methods:**
- `validate_bounds(cls, v, values)` - Line 179
- `validate_default(cls, v, values)` - Line 187
- `sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> Decimal` - Line 196
- `validate_value(self, value: Any) -> bool` - Line 271
- `clip_value(self, value: Any) -> Decimal` - Line 279
- `get_bounds(self) -> tuple[Decimal, Decimal]` - Line 287
- `get_range(self) -> Decimal` - Line 291

### Implementation: `DiscreteParameter` ✅

**Inherits**: ParameterDefinition
**Purpose**: Discrete parameter definition for integer-valued parameters
**Status**: Complete

**Implemented Methods:**
- `validate_bounds(cls, v, values)` - Line 312
- `validate_default(cls, v, values)` - Line 320
- `sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> int` - Line 335
- `validate_value(self, value: Any) -> bool` - Line 351
- `clip_value(self, value: Any) -> int` - Line 361
- `get_bounds(self) -> tuple[int, int]` - Line 380
- `get_valid_values(self) -> list[int]` - Line 384

### Implementation: `CategoricalParameter` ✅

**Inherits**: ParameterDefinition
**Purpose**: Categorical parameter definition for discrete choice parameters
**Status**: Complete

**Implemented Methods:**
- `validate_choices(cls, v)` - Line 405
- `validate_weights(cls, v, values)` - Line 415
- `validate_default(cls, v, values)` - Line 427
- `sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> Any` - Line 435
- `validate_value(self, value: Any) -> bool` - Line 444
- `clip_value(self, value: Any) -> Any` - Line 448
- `get_bounds(self) -> tuple[Any, Any]` - Line 454
- `get_choice_index(self, value: Any) -> int` - Line 458

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

### Implementation: `ParameterSpaceService` ✅

**Inherits**: BaseService
**Purpose**: Service for parameter space operations
**Status**: Complete

**Implemented Methods:**
- `build_parameter_space(self, config: dict[str, Any]) -> ParameterSpace` - Line 40
- `build_parameter_space_from_current(self, current_parameters: dict[str, Any]) -> dict[str, Any]` - Line 89
- `validate_parameter_space_config(self, config: dict[str, Any]) -> bool` - Line 155

### Implementation: `OptimizationRepository` ✅

**Inherits**: BaseComponent
**Purpose**: Repository for optimization result persistence using database models
**Status**: Complete

**Implemented Methods:**
- `async save_optimization_result(self, result: OptimizationResult, metadata: dict[str, Any] | None = None) -> str` - Line 66
- `async get_optimization_result(self, optimization_id: str) -> OptimizationResult | None` - Line 182
- `async list_optimization_results(self, strategy_name: str | None = None, limit: int = 100, offset: int = 0) -> list[OptimizationResult]` - Line 249
- `async delete_optimization_result(self, optimization_id: str) -> bool` - Line 330
- `async save_parameter_set(self, ...) -> str` - Line 371
- `async get_parameter_sets(self, optimization_id: str, limit: int | None = None) -> list[dict[str, Any]]` - Line 444

### Implementation: `ResultTransformationService` ✅

**Inherits**: BaseService
**Purpose**: Service for optimization result transformations
**Status**: Complete

**Implemented Methods:**
- `transform_for_strategies_module(self, optimization_result: dict[str, Any], current_parameters: dict[str, Any]) -> dict[str, Any]` - Line 40
- `transform_for_web_interface(self, optimization_result: OptimizationResult) -> dict[str, Any]` - Line 83
- `transform_for_analytics(self, optimization_result: OptimizationResult) -> dict[str, Any]` - Line 122
- `standardize_financial_data(self, data: dict[str, Any]) -> dict[str, Any]` - Line 185
- `extract_summary_metrics(self, optimization_result: OptimizationResult) -> dict[str, Any]` - Line 216

### Implementation: `OptimizationService` ✅

**Inherits**: BaseService, IOptimizationService, ErrorPropagationMixin
**Purpose**: Main optimization service implementation
**Status**: Complete

**Implemented Methods:**
- `async optimize_strategy(self, ...) -> dict[str, Any]` - Line 96
- `async optimize_strategy_parameters(self, strategy_id: str, optimization_request: dict[str, Any]) -> dict[str, Any]` - Line 303
- `async optimize_parameters(self, ...) -> OptimizationResult` - Line 359
- `async optimize_parameters_with_config(self, ...) -> dict[str, Any]` - Line 399
- `async analyze_optimization_results(self, optimization_result: OptimizationResult, parameter_space: ParameterSpace) -> dict[str, Any]` - Line 440
- `async shutdown(self) -> None` - Line 858

### Implementation: `ValidationMetrics` ✅

**Inherits**: BaseModel
**Purpose**: Comprehensive validation metrics for optimization results
**Status**: Complete

**Implemented Methods:**
- `get_overall_quality_score(self) -> Decimal` - Line 103

### Implementation: `ValidationConfig` ✅

**Inherits**: BaseModel
**Purpose**: Configuration for validation and overfitting prevention
**Status**: Complete

**Implemented Methods:**
- `validate_cv_method(cls, v)` - Line 217
- `validate_window_type(cls, v)` - Line 226
- `validate_correction_method(cls, v)` - Line 235

### Implementation: `TimeSeriesValidator` ✅

**Purpose**: Time series specific validation with proper temporal splits
**Status**: Complete

**Implemented Methods:**
- `create_time_series_splits(self, data_length: int, start_date: datetime, end_date: datetime) -> list[tuple[list[int], list[int]]]` - Line 264

### Implementation: `WalkForwardValidator` ✅

**Purpose**: Walk-forward analysis for time series validation
**Status**: Complete

**Implemented Methods:**
- `async run_walk_forward_analysis(self, ...) -> list[Decimal]` - Line 372

### Implementation: `OverfittingDetector` ✅

**Purpose**: Detects overfitting in optimization results
**Status**: Complete

**Implemented Methods:**
- `detect_overfitting(self, ...) -> tuple[bool, dict[str, Decimal]]` - Line 525

### Implementation: `StatisticalTester` ✅

**Purpose**: Statistical significance testing for optimization results
**Status**: Complete

**Implemented Methods:**
- `async test_significance(self, ...) -> tuple[Decimal, tuple[Decimal, Decimal], bool]` - Line 623

### Implementation: `RobustnessAnalyzer` ✅

**Purpose**: Analyzes robustness of optimization results
**Status**: Complete

**Implemented Methods:**
- `async analyze_robustness(self, ...) -> tuple[Decimal, dict[str, Any]]` - Line 754

### Implementation: `ValidationEngine` ✅

**Purpose**: Main validation engine that orchestrates all validation techniques
**Status**: Complete

**Implemented Methods:**
- `async validate_optimization_result(self, ...) -> ValidationMetrics` - Line 979

## COMPLETE API REFERENCE

### File: analysis.py

**Key Imports:**
- `from src.core.exceptions import DataProcessingError`
- `from src.core.logging import get_logger`
- `from src.utils.decorators import time_execution`
- `from src.utils.financial_calculations import calculate_var_cvar`
- `from src.utils.financial_calculations import calculate_volatility`

#### Class: `PerformanceMetrics`

**Inherits**: BaseModel
**Purpose**: Comprehensive performance metrics for trading strategies

```python
class PerformanceMetrics(BaseModel):
    def get_risk_score(self) -> Decimal  # Line 115
    def get_quality_score(self) -> Decimal  # Line 124
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
    def get_overall_stability_score(self) -> Decimal  # Line 231
```

#### Class: `ParameterImportanceAnalyzer`

**Purpose**: Analyzes parameter importance and interactions

```python
class ParameterImportanceAnalyzer:
    def __init__(self)  # Line 269
    def analyze_parameter_importance(self, optimization_results: list[dict[str, Any]], parameter_names: list[str]) -> list[SensitivityAnalysis]  # Line 273
    def _extract_parameter_data(self, optimization_results: list[dict[str, Any]], parameter_names: list[str]) -> dict[str, list[Decimal]]  # Line 318
    def _extract_performance_data(self, optimization_results: list[dict[str, Any]]) -> list[Decimal]  # Line 345
    def _analyze_single_parameter(self, ...) -> SensitivityAnalysis | None  # Line 370
    def _calculate_parameter_stability(self, param_values: list[Decimal], performance_values: list[Decimal]) -> Decimal  # Line 431
    def _find_interaction_partners(self, ...) -> tuple[list[str], dict[str, Decimal]]  # Line 475
```

#### Class: `PerformanceAnalyzer`

**Purpose**: Analyzes trading strategy performance metrics

```python
class PerformanceAnalyzer:
    def __init__(self, risk_free_rate: Decimal = DEFAULT_RISK_FREE_RATE)  # Line 532
    def calculate_performance_metrics(self, ...) -> PerformanceMetrics  # Line 543
    def _create_empty_metrics(self, start_date: datetime, end_date: datetime) -> PerformanceMetrics  # Line 634
    def _calculate_total_return(self, returns: list[Decimal]) -> Decimal  # Line 669
    def _calculate_annualized_return(self, total_return: Decimal, start_date: datetime, end_date: datetime) -> Decimal  # Line 683
    def _calculate_volatility(self, returns: list[Decimal]) -> Decimal  # Line 705
    def _calculate_downside_volatility(self, returns: list[Decimal]) -> Decimal  # Line 712
    def _calculate_drawdowns(self, returns: list[Decimal]) -> tuple[Decimal, Decimal]  # Line 738
    def _calculate_sharpe_ratio(self, annualized_return: Decimal, volatility: Decimal) -> Decimal  # Line 767
    def _calculate_sortino_ratio(self, annualized_return: Decimal, downside_volatility: Decimal) -> Decimal  # Line 784
    def _calculate_calmar_ratio(self, annualized_return: Decimal, max_drawdown: Decimal) -> Decimal  # Line 806
    def _calculate_omega_ratio(self, returns: list[Decimal]) -> Decimal  # Line 823
    def _calculate_trade_metrics(self, trades: list[dict[str, Any]]) -> dict[str, Decimal]  # Line 841
    def _calculate_var(self, returns: list[Decimal], confidence: Decimal) -> Decimal  # Line 887
    def _calculate_conditional_var(self, returns: list[Decimal], confidence: Decimal) -> Decimal  # Line 895
    def _calculate_skewness(self, returns: list[Decimal]) -> Decimal  # Line 920
    def _calculate_kurtosis(self, returns: list[Decimal]) -> Decimal  # Line 949
    def _calculate_recovery_factor(self, total_return: Decimal, max_drawdown: Decimal) -> Decimal  # Line 984
    def _calculate_stability_ratio(self, returns: list[Decimal]) -> Decimal  # Line 999
    def _calculate_consistency_score(self, returns: list[Decimal]) -> Decimal  # Line 1051
    def _calculate_turnover_ratio(self, trades: list[dict[str, Any]], initial_capital: Decimal) -> Decimal  # Line 1072
```

#### Class: `ResultsAnalyzer`

**Purpose**: Main results analyzer that orchestrates all analysis components

```python
class ResultsAnalyzer:
    def __init__(self, risk_free_rate: Decimal = DEFAULT_RISK_FREE_RATE)  # Line 1100
    def analyze_optimization_results(self, ...) -> dict[str, Any]  # Line 1113
    def _analyze_performance_distribution(self, optimization_results: list[dict[str, Any]]) -> dict[str, Any]  # Line 1184
    def _calculate_parameter_correlations(self, optimization_results: list[dict[str, Any]], parameter_names: list[str]) -> dict[str, dict[str, Decimal]]  # Line 1249
    def _analyze_optimization_landscape(self, optimization_results: list[dict[str, Any]]) -> dict[str, Any]  # Line 1304
    def _calculate_landscape_ruggedness(self, performance_values: list[float]) -> float  # Line 1363
    def _detect_multimodality(self, sorted_performance: list[float]) -> dict[str, Any]  # Line 1386
    def _calculate_convergence_rate(self, sorted_performance: list[float]) -> float  # Line 1414
    def _detect_performance_plateaus(self, sorted_performance: list[float]) -> dict[str, Any]  # Line 1433
    def _assess_improvement_potential(self, sorted_performance: list[float]) -> dict[str, Any]  # Line 1464
    def _analyze_best_result(self, best_result: dict[str, Any]) -> dict[str, Any]  # Line 1507
    def _categorize_parameter_types(self, parameters: dict[str, Any]) -> dict[str, int]  # Line 1525
    def _analyze_convergence(self, optimization_results: list[dict[str, Any]]) -> dict[str, Any]  # Line 1543
    def _create_analysis_summary(self, analysis_results: dict[str, Any], total_evaluations: int) -> dict[str, Any]  # Line 1619
```

### File: analysis_service.py

**Key Imports:**
- `from src.core.base import BaseService`
- `from src.core.exceptions import OptimizationError`
- `from src.optimization.analysis import ResultsAnalyzer`
- `from src.optimization.interfaces import IAnalysisService`
- `from src.utils.messaging_patterns import ErrorPropagationMixin`

#### Class: `AnalysisService`

**Inherits**: BaseService, IAnalysisService
**Purpose**: Service for optimization result analysis

```python
class AnalysisService(BaseService, IAnalysisService):
    def __init__(self, ...)  # Line 26
    async def analyze_optimization_results(self, ...) -> dict[str, Any]  # Line 60
    async def analyze_parameter_importance(self, optimization_results: list[dict[str, Any]], parameter_names: list[str]) -> list[Any]  # Line 105
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
    def __init__(self, ...)  # Line 32
    async def evaluate_strategy(self, ...) -> dict[str, Decimal]  # Line 57
    def create_objective_function(self, ...) -> Callable[[dict[str, Any]], Any]  # Line 119
    async def _run_backtest(self, ...) -> Any  # Line 194
    def _extract_performance_metrics(self, backtest_result: Any) -> dict[str, Decimal]  # Line 253
    def _simulate_performance(self, parameters: dict[str, Any]) -> dict[str, Decimal]  # Line 308
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
    def validate_kernel_type(cls, v)  # Line 145
    def validate_bounds(cls, v)  # Line 154
```

#### Class: `BayesianConfig`

**Inherits**: BaseModel
**Purpose**: Configuration for Bayesian optimization

```python
class BayesianConfig(BaseModel):
    def validate_batch_strategy(cls, v)  # Line 201
```

#### Class: `BayesianPoint`

**Inherits**: BaseModel
**Purpose**: Represents a point in the Bayesian optimization process

```python
class BayesianPoint(BaseModel):
    def mark_evaluated(self, objective_value: Decimal, objective_std: Decimal | None = None) -> None  # Line 240
```

#### Class: `GaussianProcessModel`

**Purpose**: Gaussian Process surrogate model for Bayesian optimization

```python
class GaussianProcessModel:
    def __init__(self, config: GaussianProcessConfig, parameter_space: ParameterSpace)  # Line 258
    def _create_kernel(self) -> Any  # Line 295
    def _encode_parameters(self, parameters_list: list[dict[str, Any]]) -> np.ndarray  # Line 328
    def fit(self, points: list[BayesianPoint]) -> None  # Line 364
    def predict(self, parameters_list: list[dict[str, Any]], return_std: bool = True) -> tuple[np.ndarray, np.ndarray | None]  # Line 413
    def get_model_info(self) -> dict[str, Any]  # Line 453
```

#### Class: `AcquisitionOptimizer`

**Purpose**: Optimizes acquisition functions to select next evaluation points

```python
class AcquisitionOptimizer:
    def __init__(self, ...)  # Line 473
    def optimize_acquisition(self, current_points: list[BayesianPoint], n_points: int = 1) -> list[dict[str, Any]]  # Line 496
    def _optimize_single_point(self, current_points: list[BayesianPoint]) -> dict[str, Any]  # Line 514
    def _optimize_batch(self, current_points: list[BayesianPoint], n_points: int) -> list[dict[str, Any]]  # Line 554
    def _decode_parameters(self, x_encoded: np.ndarray) -> dict[str, Any]  # Line 574
    def _calculate_acquisition(self, mean: float, std: float, best_value: float) -> float  # Line 604
    def _expected_improvement(self, mean: float, std: float, best_value: float) -> float  # Line 620
    def _upper_confidence_bound(self, mean: float, std: float) -> float  # Line 632
    def _probability_of_improvement(self, mean: float, std: float, best_value: float) -> float  # Line 637
    def _lower_confidence_bound(self, mean: float, std: float) -> float  # Line 648
```

#### Class: `BayesianOptimizer`

**Inherits**: OptimizationEngine
**Purpose**: Bayesian optimization engine with Gaussian Process surrogate models

```python
class BayesianOptimizer(OptimizationEngine):
    def __init__(self, ...)  # Line 662
    async def optimize(self, ...) -> OptimizationResult  # Line 708
    async def _generate_initial_points(self, ...) -> None  # Line 808
    async def _optimization_iteration(self, objective_function: Callable) -> None  # Line 836
    async def _evaluate_point(self, point: BayesianPoint, objective_function: Callable) -> None  # Line 867
    def _update_best_point(self) -> None  # Line 909
    def _check_convergence(self) -> bool  # Line 938
    async def _finalize_optimization(self) -> OptimizationResult  # Line 953
    def get_next_parameters(self) -> dict[str, Any] | None  # Line 982
    def get_gp_predictions(self, parameters_list: list[dict[str, Any]]) -> tuple[list[Decimal], list[Decimal]]  # Line 990
    def get_optimization_summary(self) -> dict[str, Any]  # Line 1006
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
    def validate_grid_resolution(cls, v)  # Line 132
```

#### Class: `OptimizationCandidate`

**Inherits**: BaseModel
**Purpose**: Represents a single parameter combination candidate for evaluation

```python
class OptimizationCandidate(BaseModel):
    def mark_started(self) -> None  # Line 176
    def mark_completed(self, ...) -> None  # Line 181
    def mark_failed(self, error_message: str) -> None  # Line 198
```

#### Class: `GridGenerator`

**Purpose**: Generates parameter grids for brute force optimization

```python
class GridGenerator:
    def __init__(self, parameter_space: ParameterSpace, config: GridSearchConfig)  # Line 217
    def generate_initial_grid(self) -> list[dict[str, Any]]  # Line 236
    def _generate_uniform_grid(self) -> list[dict[str, Any]]  # Line 256
    def _generate_random_grid(self) -> list[dict[str, Any]]  # Line 304
    def _generate_latin_hypercube(self) -> list[dict[str, Any]]  # Line 318
    def _generate_sobol_sequence(self) -> list[dict[str, Any]]  # Line 363
    def _generate_halton_sequence(self) -> list[dict[str, Any]]  # Line 403
    def generate_refined_grid(self, best_candidates: list[OptimizationCandidate], refinement_factor: Decimal) -> list[dict[str, Any]]  # Line 443
```

#### Class: `BruteForceOptimizer`

**Inherits**: OptimizationEngine
**Purpose**: Brute force optimization engine with grid search and intelligent sampling

```python
class BruteForceOptimizer(OptimizationEngine):
    def __init__(self, ...)  # Line 540
    async def optimize(self, ...) -> OptimizationResult  # Line 595
    def _create_candidates(self, parameter_combinations: list[dict[str, Any]]) -> None  # Line 693
    async def _evaluate_candidates_in_batches(self, objective_function: Callable) -> None  # Line 704
    async def _evaluate_batch(self, batch: list[OptimizationCandidate], objective_function: Callable) -> None  # Line 732
    async def _evaluate_candidate_with_semaphore(self, candidate: OptimizationCandidate, objective_function: Callable) -> None  # Line 779
    async def _evaluate_candidate(self, candidate: OptimizationCandidate, objective_function: Callable) -> None  # Line 786
    def _is_better_candidate(self, candidate: OptimizationCandidate) -> bool  # Line 854
    def _is_duplicate(self, ...) -> bool  # Line 867
    def _parameters_equal(self, params1: dict[str, Any], params2: dict[str, Any]) -> bool  # Line 886
    def _validate_candidate(self, candidate: OptimizationCandidate) -> bool  # Line 906
    async def _validate_candidate_performance(self, candidate: OptimizationCandidate, objective_function: Callable) -> Decimal | None  # Line 911
    def _perturb_parameters(self, parameters: dict[str, Any], noise_factor: float) -> dict[str, Any]  # Line 955
    def _should_stop_early(self) -> bool  # Line 988
    async def _run_adaptive_refinement(self, objective_function: Callable) -> None  # Line 1009
    def _get_top_candidates(self, count: int) -> list[OptimizationCandidate]  # Line 1042
    async def _finalize_optimization(self) -> OptimizationResult  # Line 1055
    async def _calculate_statistical_significance(self) -> Decimal | None  # Line 1093
    def _analyze_parameter_stability(self) -> dict[str, Decimal]  # Line 1139
    def get_next_parameters(self) -> dict[str, Any] | None  # Line 1181
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
    def estimate_completion_time(self) -> None  # Line 263
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
def create_profit_maximization_objective() -> OptimizationObjective  # Line 716
def create_risk_minimization_objective() -> OptimizationObjective  # Line 728
def create_sharpe_ratio_objective() -> OptimizationObjective  # Line 740
def create_standard_trading_objectives() -> list[OptimizationObjective]  # Line 752
```

### File: data_transformer.py

**Key Imports:**
- `from src.optimization.core import OptimizationResult`
- `from src.utils.decimal_utils import to_decimal`
- `from src.utils.messaging_patterns import MessagePattern`

#### Class: `OptimizationDataTransformer`

**Purpose**: Handles consistent data transformation for optimization module

```python
class OptimizationDataTransformer:
    def transform_optimization_result_to_event_data(result: OptimizationResult, metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 23
    def transform_parameter_set_to_event_data(parameters: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 63
    def transform_objective_values_to_event_data(objective_values: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]  # Line 91
    def validate_financial_precision(data: dict[str, Any]) -> dict[str, Any]  # Line 114
    def ensure_boundary_fields(data: dict[str, Any], source: str = 'optimization') -> dict[str, Any]  # Line 154
    def transform_for_req_reply(cls, request_type: str, data: Any, correlation_id: str | None = None) -> dict[str, Any]  # Line 181
    def transform_for_batch_processing(cls, ...) -> dict[str, Any]  # Line 236
    def align_processing_paradigm(cls, data: dict[str, Any], target_mode: str = 'batch') -> dict[str, Any]  # Line 301
    def apply_cross_module_validation(cls, ...) -> dict[str, Any]  # Line 380
    def _apply_backtesting_boundary_validation(cls, data: dict[str, Any], source_module: str, target_module: str) -> dict[str, Any]  # Line 469
```

### File: di_registration.py

**Key Imports:**
- `from src.core.dependency_injection import DependencyInjector`
- `from src.core.logging import get_logger`

#### Functions:

```python
def register_optimization_dependencies(injector: DependencyInjector) -> None  # Line 25
def configure_optimization_module(injector: DependencyInjector, config: dict[str, Any] | None = None) -> None  # Line 193
def get_optimization_service(injector: DependencyInjector) -> 'IOptimizationService'  # Line 213
def get_optimization_controller(injector: DependencyInjector)  # Line 218
def get_optimization_repository(injector: DependencyInjector)  # Line 223
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
    def __init__(self, ...)  # Line 35
    def _create_optimization_repository(self, database_session: Any = None, **kwargs) -> 'OptimizationRepositoryProtocol'  # Line 68
    def _create_backtest_integration(self, backtest_service: Any = None, **kwargs) -> 'IBacktestIntegrationService'  # Line 108
    def _create_analysis_service(self, results_analyzer: Any = None, **kwargs) -> 'IAnalysisService'  # Line 144
    def _create_optimization_service(self, ...) -> 'IOptimizationService'  # Line 180
    def _create_optimization_controller(self, optimization_service: Any = None, **kwargs) -> 'OptimizationController'  # Line 254
    def create_complete_optimization_stack(self) -> dict[str, Any]  # Line 292
```

#### Class: `OptimizationComponentFactory`

**Purpose**: Composite factory for all optimization components

```python
class OptimizationComponentFactory:
    def __init__(self, dependency_container: Any = None, correlation_id: str | None = None)  # Line 321
    def create_service(self, **kwargs) -> 'IOptimizationService'  # Line 325
    def create_controller(self, **kwargs) -> 'OptimizationController'  # Line 329
    def create_repository(self, **kwargs) -> 'OptimizationRepositoryProtocol'  # Line 333
    def create_backtest_integration(self, **kwargs) -> 'IBacktestIntegrationService'  # Line 337
    def create_analysis_service(self, **kwargs) -> 'IAnalysisService'  # Line 341
    def register_factories(self, container: Any) -> None  # Line 345
```

#### Functions:

```python
def create_optimization_service(injector: DependencyInjector | None = None) -> 'IOptimizationService'  # Line 351
def create_optimization_controller(injector: DependencyInjector | None = None) -> 'OptimizationController'  # Line 359
def create_optimization_stack(injector: DependencyInjector | None = None) -> dict[str, Any]  # Line 367
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
    def sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> Any  # Line 74
    def validate_value(self, value: Any) -> bool  # Line 87
    def clip_value(self, value: Any) -> Any  # Line 100
    def get_bounds(self) -> tuple[Any, Any]  # Line 113
    def is_active(self, context: dict[str, Any]) -> bool  # Line 122
```

#### Class: `ContinuousParameter`

**Inherits**: ParameterDefinition
**Purpose**: Continuous parameter definition for real-valued parameters

```python
class ContinuousParameter(ParameterDefinition):
    def validate_bounds(cls, v, values)  # Line 179
    def validate_default(cls, v, values)  # Line 187
    def sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> Decimal  # Line 196
    def validate_value(self, value: Any) -> bool  # Line 271
    def clip_value(self, value: Any) -> Decimal  # Line 279
    def get_bounds(self) -> tuple[Decimal, Decimal]  # Line 287
    def get_range(self) -> Decimal  # Line 291
```

#### Class: `DiscreteParameter`

**Inherits**: ParameterDefinition
**Purpose**: Discrete parameter definition for integer-valued parameters

```python
class DiscreteParameter(ParameterDefinition):
    def validate_bounds(cls, v, values)  # Line 312
    def validate_default(cls, v, values)  # Line 320
    def sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> int  # Line 335
    def validate_value(self, value: Any) -> bool  # Line 351
    def clip_value(self, value: Any) -> int  # Line 361
    def get_bounds(self) -> tuple[int, int]  # Line 380
    def get_valid_values(self) -> list[int]  # Line 384
```

#### Class: `CategoricalParameter`

**Inherits**: ParameterDefinition
**Purpose**: Categorical parameter definition for discrete choice parameters

```python
class CategoricalParameter(ParameterDefinition):
    def validate_choices(cls, v)  # Line 405
    def validate_weights(cls, v, values)  # Line 415
    def validate_default(cls, v, values)  # Line 427
    def sample(self, strategy: SamplingStrategy = SamplingStrategy.UNIFORM) -> Any  # Line 435
    def validate_value(self, value: Any) -> bool  # Line 444
    def clip_value(self, value: Any) -> Any  # Line 448
    def get_bounds(self) -> tuple[Any, Any]  # Line 454
    def get_choice_index(self, value: Any) -> int  # Line 458
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

### File: parameter_space_service.py

**Key Imports:**
- `from src.core.base import BaseService`
- `from src.core.exceptions import ValidationError`
- `from src.optimization.parameter_space import ParameterSpace`
- `from src.optimization.parameter_space import ParameterSpaceBuilder`

#### Class: `ParameterSpaceService`

**Inherits**: BaseService
**Purpose**: Service for parameter space operations

```python
class ParameterSpaceService(BaseService):
    def __init__(self, ...)  # Line 23
    def build_parameter_space(self, config: dict[str, Any]) -> ParameterSpace  # Line 40
    def build_parameter_space_from_current(self, current_parameters: dict[str, Any]) -> dict[str, Any]  # Line 89
    def _get_default_trading_parameter_space(self) -> dict[str, Any]  # Line 132
    def validate_parameter_space_config(self, config: dict[str, Any]) -> bool  # Line 155
    def _validate_parameter_type_config(self, param_name: str, param_type: str, param_config: dict[str, Any]) -> None  # Line 202
```

### File: repository.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.event_constants import OptimizationEvents`
- `from src.core.exceptions import RepositoryError`
- `from src.database.models.optimization import OptimizationResult`
- `from src.database.models.optimization import OptimizationRun`

#### Class: `OptimizationRepository`

**Inherits**: BaseComponent
**Purpose**: Repository for optimization result persistence using database models

```python
class OptimizationRepository(BaseComponent):
    def __init__(self, ...)  # Line 41
    async def save_optimization_result(self, result: OptimizationResult, metadata: dict[str, Any] | None = None) -> str  # Line 66
    async def get_optimization_result(self, optimization_id: str) -> OptimizationResult | None  # Line 182
    async def list_optimization_results(self, strategy_name: str | None = None, limit: int = 100, offset: int = 0) -> list[OptimizationResult]  # Line 249
    async def delete_optimization_result(self, optimization_id: str) -> bool  # Line 330
    async def save_parameter_set(self, ...) -> str  # Line 371
    async def get_parameter_sets(self, optimization_id: str, limit: int | None = None) -> list[dict[str, Any]]  # Line 444
```

### File: result_transformation_service.py

**Key Imports:**
- `from src.core.base import BaseService`
- `from src.optimization.core import OptimizationResult`

#### Class: `ResultTransformationService`

**Inherits**: BaseService
**Purpose**: Service for optimization result transformations

```python
class ResultTransformationService(BaseService):
    def __init__(self, ...)  # Line 23
    def transform_for_strategies_module(self, optimization_result: dict[str, Any], current_parameters: dict[str, Any]) -> dict[str, Any]  # Line 40
    def transform_for_web_interface(self, optimization_result: OptimizationResult) -> dict[str, Any]  # Line 83
    def transform_for_analytics(self, optimization_result: OptimizationResult) -> dict[str, Any]  # Line 122
    def _calculate_performance_improvement(self, optimal_value: Decimal, baseline: Decimal) -> Decimal  # Line 167
    def standardize_financial_data(self, data: dict[str, Any]) -> dict[str, Any]  # Line 185
    def extract_summary_metrics(self, optimization_result: OptimizationResult) -> dict[str, Any]  # Line 216
    def _calculate_quality_score(self, optimization_result: OptimizationResult) -> float  # Line 239
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
    def __init__(self, ...)  # Line 55
    async def optimize_strategy(self, ...) -> dict[str, Any]  # Line 96
    async def optimize_strategy_parameters(self, strategy_id: str, optimization_request: dict[str, Any]) -> dict[str, Any]  # Line 303
    async def optimize_parameters(self, ...) -> OptimizationResult  # Line 359
    async def optimize_parameters_with_config(self, ...) -> dict[str, Any]  # Line 399
    async def analyze_optimization_results(self, optimization_result: OptimizationResult, parameter_space: ParameterSpace) -> dict[str, Any]  # Line 440
    async def _create_objective_function(self, ...) -> Callable[[dict[str, Any]], Any]  # Line 498
    def _create_simulation_objective_function(self, ...) -> Callable[[dict[str, Any]], Any]  # Line 530
    def _create_standard_trading_objectives(self) -> list[OptimizationObjective]  # Line 579
    async def _create_brute_force_optimizer(self, ...) -> Any  # Line 615
    async def _create_bayesian_optimizer(self, ...) -> Any  # Line 653
    def _build_objectives(self, objectives_config: list[dict[str, Any]]) -> list[OptimizationObjective]  # Line 694
    def _create_objective_function_by_name(self, function_name: str) -> Callable[[dict[str, Any]], Any]  # Line 734
    async def _emit_result_saved_event(self, optimization_result: OptimizationResult, result_id: str) -> None  # Line 749
    def _determine_target_processing_mode(self, result: OptimizationResult) -> str  # Line 781
    def _determine_target_processing_mode_for_strategy(self, strategy_name: str) -> str  # Line 802
    def _apply_cross_module_alignment(self, data: dict[str, Any]) -> None  # Line 827
    async def shutdown(self) -> None  # Line 858
    def _transform_event_data(self, data: dict[str, Any]) -> dict[str, Any]  # Line 872
    def _validate_event_data(self, data: Any) -> None  # Line 952
    def _propagate_error_with_boundary_validation(self, ...) -> None  # Line 970
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
    def get_overall_quality_score(self) -> Decimal  # Line 103
```

#### Class: `ValidationConfig`

**Inherits**: BaseModel
**Purpose**: Configuration for validation and overfitting prevention

```python
class ValidationConfig(BaseModel):
    def validate_cv_method(cls, v)  # Line 217
    def validate_window_type(cls, v)  # Line 226
    def validate_correction_method(cls, v)  # Line 235
```

#### Class: `TimeSeriesValidator`

**Purpose**: Time series specific validation with proper temporal splits

```python
class TimeSeriesValidator:
    def __init__(self, config: ValidationConfig)  # Line 251
    def create_time_series_splits(self, data_length: int, start_date: datetime, end_date: datetime) -> list[tuple[list[int], list[int]]]  # Line 264
    def _create_expanding_window_splits(self, data_length: int) -> list[tuple[list[int], list[int]]]  # Line 287
    def _create_blocked_splits(self, data_length: int) -> list[tuple[list[int], list[int]]]  # Line 315
    def _create_gap_splits(self, data_length: int) -> list[tuple[list[int], list[int]]]  # Line 334
```

#### Class: `WalkForwardValidator`

**Purpose**: Walk-forward analysis for time series validation

```python
class WalkForwardValidator:
    def __init__(self, config: ValidationConfig)  # Line 356
    async def run_walk_forward_analysis(self, ...) -> list[Decimal]  # Line 372
    async def _evaluate_period(self, ...) -> Decimal | None  # Line 448
```

#### Class: `OverfittingDetector`

**Purpose**: Detects overfitting in optimization results

```python
class OverfittingDetector:
    def __init__(self, config: ValidationConfig)  # Line 514
    def detect_overfitting(self, ...) -> tuple[bool, dict[str, Decimal]]  # Line 525
```

#### Class: `StatisticalTester`

**Purpose**: Statistical significance testing for optimization results

```python
class StatisticalTester:
    def __init__(self, config: ValidationConfig)  # Line 603
    async def test_significance(self, ...) -> tuple[Decimal, tuple[Decimal, Decimal], bool]  # Line 623
    def _bootstrap_confidence_interval(self, data: list[Decimal]) -> tuple[Decimal, Decimal]  # Line 670
    def _one_sample_test(self, data: list[Decimal]) -> Decimal  # Line 691
    def _two_sample_test(self, data1: list[Decimal], data2: list[Decimal]) -> Decimal  # Line 703
    def _apply_multiple_testing_correction(self, p_value: Decimal) -> Decimal  # Line 716
```

#### Class: `RobustnessAnalyzer`

**Purpose**: Analyzes robustness of optimization results

```python
class RobustnessAnalyzer:
    def __init__(self, config: ValidationConfig)  # Line 739
    async def analyze_robustness(self, ...) -> tuple[Decimal, dict[str, Any]]  # Line 754
    def _perturb_parameters(self, parameters: dict[str, Any], parameter_space: Any) -> dict[str, Any]  # Line 894
    def _calculate_parameter_sensitivities(self, parameter_sensitivities: dict[str, list[tuple[Any, Decimal]]]) -> dict[str, Decimal]  # Line 928
```

#### Class: `ValidationEngine`

**Purpose**: Main validation engine that orchestrates all validation techniques

```python
class ValidationEngine:
    def __init__(self, config: ValidationConfig)  # Line 960
    async def validate_optimization_result(self, ...) -> ValidationMetrics  # Line 979
    async def _evaluate_in_out_sample(self, ...) -> tuple[Decimal, Decimal]  # Line 1103
    async def _run_cross_validation(self, ...) -> list[Decimal]  # Line 1216
    def _calculate_stability_score(self, ...) -> Decimal  # Line 1286
```

---
**Generated**: Complete reference for optimization module
**Total Classes**: 61
**Total Functions**: 15