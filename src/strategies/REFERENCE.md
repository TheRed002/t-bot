# STRATEGIES Module Reference

## INTEGRATION
**Dependencies**: analytics, capital_management, core, data, database, error_handling, execution, ml, monitoring, optimization, risk_management, state, utils
**Used By**: None
**Provides**: EnvironmentAwareStrategyManager, InventoryManager, ParetoFrontierManager, SpeciationManager, StrategyConfigurationManager, StrategyController, StrategyService, TechnicalRuleEngine
**Patterns**: Async Operations, Circuit Breaker, Component Architecture, Service Layer

## DETECTED PATTERNS
**Financial**:
- Decimal precision arithmetic
- Database decimal columns
- Financial data handling
**Performance**:
- Parallel execution
- Caching
- Parallel execution
**Architecture**:
- BaseStrategy inherits from base architecture
- StrategyController inherits from base architecture
- StrategyServiceContainerBuilder inherits from base architecture

## MODULE OVERVIEW
**Files**: 37 Python files
**Classes**: 108
**Functions**: 7

## COMPLETE API REFERENCE

## IMPLEMENTATIONS

### Implementation: `BaseStrategy` 🔧

**Inherits**: BaseComponent, BaseStrategyInterface
**Purpose**: Base strategy interface that ALL strategies must inherit from
**Status**: Abstract Base Class

**Implemented Methods:**
- `strategy_type(self) -> StrategyType` - Line 160
- `name(self) -> str` - Line 165
- `version(self) -> str` - Line 170
- `status(self) -> StrategyStatus` - Line 175
- `async generate_signals(self, data: MarketData) -> list[Signal]` - Line 197
- `async validate_signal(self, signal: Signal) -> bool` - Line 335
- `get_position_size(self, signal: Signal) -> Decimal` - Line 391
- `should_exit(self, position: Position, data: MarketData) -> bool` - Line 442
- `async pre_trade_validation(self, signal: Signal) -> bool` - Line 458
- `async post_trade_processing(self, trade_result: Any) -> None` - Line 484
- `set_risk_manager(self, risk_manager: Any) -> None` - Line 530
- `set_exchange(self, exchange: Any) -> None` - Line 538
- `set_data_service(self, data_service: Any) -> None` - Line 547
- `set_validation_framework(self, validation_framework: ValidationFramework) -> None` - Line 555
- `set_metrics_collector(self, metrics_collector: MetricsCollector) -> None` - Line 564
- `get_strategy_info(self) -> dict[str, Any]` - Line 573
- `async initialize(self, config: StrategyConfig) -> None` - Line 588
- `async start(self) -> bool` - Line 600
- `async stop(self) -> bool` - Line 653
- `async pause(self) -> None` - Line 679
- `async resume(self) -> None` - Line 685
- `async prepare_for_backtest(self, config: dict[str, Any]) -> None` - Line 692
- `async process_historical_data(self, data: MarketData) -> list[Signal]` - Line 708
- `async get_backtest_metrics(self) -> dict[str, Any]` - Line 724
- `get_real_time_metrics(self) -> dict[str, Any]` - Line 737
- `update_config(self, new_config: dict[str, Any]) -> None` - Line 783
- `async get_state(self) -> dict[str, Any]` - Line 798
- `get_performance_summary(self) -> dict[str, Any]` - Line 815
- `cleanup(self) -> None` - Line 845
- `async get_market_data(self, symbol: str) -> MarketData | None` - Line 879
- `async get_historical_data(self, symbol: str, timeframe: str, limit: int = 100) -> list[MarketData]` - Line 901
- `async execute_order(self, signal: Signal) -> Any | None` - Line 929
- `async save_state(self, state_data: dict[str, Any]) -> bool` - Line 975
- `async load_state(self) -> dict[str, Any] | None` - Line 998
- `get_metrics(self) -> dict[str, Any]` - Line 1078
- `is_healthy(self) -> bool` - Line 1094
- `async reset(self) -> bool` - Line 1119
- `set_execution_service(self, execution_service: Any) -> None` - Line 1191
- `get_status(self) -> StrategyStatus` - Line 1197
- `get_status_string(self) -> str` - Line 1201
- `async validate_market_data(self, data: MarketData | None) -> None` - Line 1252
- `async get_sma(self, symbol: str, period: int) -> Decimal | None` - Line 1286
- `async get_ema(self, symbol: str, period: int) -> Decimal | None` - Line 1307
- `async get_rsi(self, symbol: str, period: int = 14) -> Decimal | None` - Line 1326
- `async get_volatility(self, symbol: str, period: int) -> Decimal | None` - Line 1345
- `async get_atr(self, symbol: str, period: int) -> Decimal | None` - Line 1364
- `async get_volume_ratio(self, symbol: str, period: int) -> Decimal | None` - Line 1383
- `async get_bollinger_bands(self, symbol: str, period: int = 20, std_dev: float = 2.0) -> dict[str, Decimal] | None` - Line 1402
- `async get_macd(self, ...) -> dict[str, Decimal] | None` - Line 1429
- `async execute_with_algorithm(self, ...) -> dict[str, Any] | None` - Line 1458
- `async optimize_parameters(self, optimization_config: dict[str, Any] | None) -> dict[str, Any]` - Line 1506
- `async enhance_signals_with_ml(self, signals: list[Signal]) -> list[Signal]` - Line 1580
- `async get_allocated_capital(self) -> Decimal` - Line 1686
- `async execute_large_order(self, order_request: OrderRequest, max_position_size: Decimal | None = None) -> dict[str, Any] | None` - Line 1731
- `async get_execution_algorithms_status(self) -> dict[str, Any]` - Line 1793

### Implementation: `StrategyConfigurationManager` ✅

**Purpose**: Manager for strategy configuration handling
**Status**: Complete

**Implemented Methods:**
- `load_strategy_config(self, strategy_name: str) -> StrategyConfig` - Line 113
- `save_strategy_config(self, strategy_name: str, config: StrategyConfig) -> None` - Line 244
- `validate_config(self, config: dict[str, Any]) -> bool` - Line 299
- `get_available_strategies(self) -> list[str]` - Line 322
- `get_config_schema(self) -> dict[str, Any]` - Line 342
- `update_config_parameter(self, strategy_name: str, parameter: str, value: Any) -> bool` - Line 351
- `create_strategy_config(self, strategy_name: str, strategy_type: StrategyType, symbol: str, **kwargs) -> StrategyConfig` - Line 415
- `delete_strategy_config(self, strategy_name: str) -> bool` - Line 487
- `get_config_summary(self) -> dict[str, Any]` - Line 530

### Implementation: `StrategyConfigTemplates` ✅

**Purpose**: Comprehensive strategy configuration templates for production deployment
**Status**: Complete

**Implemented Methods:**
- `get_arbitrage_scanner_config(risk_level, ...) -> dict[str, Any]` - Line 35
- `get_mean_reversion_config(timeframe: str = '1h', risk_level: str = 'medium') -> dict[str, Any]` - Line 139
- `get_trend_following_config(timeframe: str = '1h', trend_strength: str = 'medium') -> dict[str, Any]` - Line 266
- `get_market_making_config(symbol, ...) -> dict[str, Any]` - Line 383
- `get_volatility_breakout_config(volatility_regime: str = 'medium', breakout_type: str = 'range') -> dict[str, Any]` - Line 499
- `get_ensemble_config(strategy_types, ...) -> dict[str, Any]` - Line 605
- `get_all_templates(cls) -> dict[str, dict[str, Any]]` - Line 688
- `get_template_by_name(cls, template_name: str) -> dict[str, Any]` - Line 733
- `list_available_templates(cls) -> list[str]` - Line 754
- `get_templates_by_strategy_type(cls, strategy_type: str) -> dict[str, dict[str, Any]]` - Line 764
- `validate_template(cls, template: dict[str, Any]) -> tuple[bool, list[str]]` - Line 782

### Implementation: `StrategyController` ✅

**Inherits**: BaseComponent
**Purpose**: Controller for strategy operations
**Status**: Complete

**Implemented Methods:**
- `async register_strategy(self, request_data: dict[str, Any]) -> dict[str, Any]` - Line 29
- `async start_strategy(self, strategy_id: str) -> dict[str, Any]` - Line 64
- `async stop_strategy(self, strategy_id: str) -> dict[str, Any]` - Line 90
- `async process_market_data(self, market_data_dict: dict[str, Any]) -> dict[str, Any]` - Line 116
- `async get_strategy_performance(self, strategy_id: str) -> dict[str, Any]` - Line 146
- `async get_all_strategies(self) -> dict[str, Any]` - Line 168
- `async cleanup_strategy(self, strategy_id: str) -> dict[str, Any]` - Line 184

### Implementation: `StrategyServiceContainer` ✅

**Purpose**: Container for all services required by strategies
**Status**: Complete

**Implemented Methods:**
- `is_ready(self) -> bool` - Line 75
- `get_service_status(self) -> dict[str, bool]` - Line 84

### Implementation: `StrategyServiceContainerBuilder` ✅

**Inherits**: BaseComponent
**Purpose**: Builder for creating properly configured StrategyServiceContainer
**Status**: Complete

**Implemented Methods:**
- `with_risk_service(self, risk_service: 'RiskManagementService') -> 'StrategyServiceContainerBuilder'` - Line 110
- `with_data_service(self, data_service: 'DataService') -> 'StrategyServiceContainerBuilder'` - Line 116
- `with_execution_service(self, execution_service: 'ExecutionService') -> 'StrategyServiceContainerBuilder'` - Line 122
- `with_monitoring_service(self, monitoring_service: 'MonitoringService') -> 'StrategyServiceContainerBuilder'` - Line 128
- `with_state_service(self, state_service: 'StateService') -> 'StrategyServiceContainerBuilder'` - Line 134
- `with_capital_service(self, capital_service: 'CapitalManagementService') -> 'StrategyServiceContainerBuilder'` - Line 140
- `with_ml_service(self, ml_service: 'MLService') -> 'StrategyServiceContainerBuilder'` - Line 147
- `with_analytics_service(self, analytics_service: 'AnalyticsService') -> 'StrategyServiceContainerBuilder'` - Line 153
- `with_optimization_service(self, optimization_service: 'OptimizationService') -> 'StrategyServiceContainerBuilder'` - Line 159
- `build(self) -> StrategyServiceContainer` - Line 165

### Implementation: `AdaptiveMomentumStrategy` ✅

**Inherits**: BaseStrategy
**Purpose**: Enhanced Adaptive Momentum Strategy with service layer integration
**Status**: Complete

**Implemented Methods:**
- `name(self) -> str` - Line 61
- `name(self, value: str) -> None` - Line 70
- `version(self) -> str` - Line 75
- `version(self, value: str) -> None` - Line 80
- `status(self) -> str` - Line 85
- `status(self, value: str) -> None` - Line 90
- `strategy_type(self) -> StrategyType` - Line 145
- `set_technical_indicators(self, technical_indicators: TechnicalIndicators) -> None` - Line 149
- `set_strategy_service(self, strategy_service: 'StrategyService') -> None` - Line 154
- `set_regime_detector(self, regime_detector: MarketRegimeDetector) -> None` - Line 159
- `set_adaptive_risk_manager(self, adaptive_risk_manager: AdaptiveRiskManager) -> None` - Line 164
- `async validate_signal(self, signal: Signal) -> bool` - Line 657
- `get_position_size(self, signal: Signal) -> Decimal` - Line 738
- `should_exit(self, position: Position, data: MarketData) -> bool` - Line 831
- `get_strategy_info(self) -> dict[str, Any]` - Line 912
- `cleanup(self) -> None` - Line 996

### Implementation: `DynamicStrategyFactory` ✅

**Inherits**: BaseComponent
**Purpose**: Factory for creating dynamic strategies with service layer integration
**Status**: Complete

**Implemented Methods:**
- `async create_strategy(self, strategy_name: str, config: dict[str, Any]) -> BaseStrategy | None` - Line 94
- `get_available_strategies(self) -> dict[str, str]` - Line 354
- `get_strategy_requirements(self, strategy_name: str) -> dict[str, Any]` - Line 361
- `async create_multiple_strategies(self, strategy_configs: dict[str, dict[str, Any]]) -> dict[str, BaseStrategy | None]` - Line 399

### Implementation: `VolatilityBreakoutStrategy` ✅

**Inherits**: BaseStrategy
**Purpose**: Enhanced Volatility Breakout Strategy with service layer integration
**Status**: Complete

**Implemented Methods:**
- `name(self) -> str` - Line 60
- `name(self, value: str) -> None` - Line 69
- `version(self) -> str` - Line 74
- `version(self, value: str) -> None` - Line 79
- `status(self) -> str` - Line 84
- `status(self, value: str) -> None` - Line 89
- `strategy_type(self) -> StrategyType` - Line 149
- `set_technical_indicators(self, technical_indicators: TechnicalIndicators) -> None` - Line 153
- `set_strategy_service(self, strategy_service: 'StrategyService') -> None` - Line 158
- `set_regime_detector(self, regime_detector: MarketRegimeDetector) -> None` - Line 163
- `set_adaptive_risk_manager(self, adaptive_risk_manager: AdaptiveRiskManager) -> None` - Line 168
- `async validate_signal(self, signal: Signal) -> bool` - Line 1041
- `get_position_size(self, signal: Signal) -> Decimal` - Line 1156
- `should_exit(self, position: Position, data: MarketData) -> bool` - Line 1269
- `get_strategy_info(self) -> dict[str, Any]` - Line 1348
- `cleanup(self) -> None` - Line 1436

### Implementation: `StrategyMode` ✅

**Inherits**: Enum
**Purpose**: Strategy operation modes for different environments
**Status**: Complete

### Implementation: `EnvironmentAwareStrategyConfiguration` ✅

**Purpose**: Environment-specific strategy configuration
**Status**: Complete

**Implemented Methods:**
- `get_sandbox_strategy_config() -> dict[str, Any]` - Line 42
- `get_live_strategy_config() -> dict[str, Any]` - Line 66

### Implementation: `EnvironmentAwareStrategyManager` ✅

**Inherits**: EnvironmentAwareServiceMixin
**Purpose**: Environment-aware strategy management functionality
**Status**: Complete

**Implemented Methods:**
- `get_environment_strategy_config(self, exchange: str) -> dict[str, Any]` - Line 140
- `async deploy_environment_aware_strategy(self, strategy_config: StrategyConfig, exchange: str, force_deploy: bool = False) -> bool` - Line 153
- `async validate_strategy_for_environment(self, strategy_config: StrategyConfig, exchange: str) -> bool` - Line 220
- `async generate_environment_aware_signal(self, strategy_name: str, market_data: MarketData, exchange: str) -> 'Signal | None'` - Line 362
- `async update_strategy_performance(self, ...) -> None` - Line 545
- `get_environment_strategy_metrics(self, exchange: str) -> dict[str, Any]` - Line 577
- `async rebalance_strategies_for_environment(self, exchange: str) -> dict[str, Any]` - Line 607

### Implementation: `FitnessFunction` 🔧

**Inherits**: ABC
**Purpose**: Abstract base class for fitness functions
**Status**: Abstract Base Class

**Implemented Methods:**
- `calculate(self, result: OptimizationResult) -> float` - Line 18

### Implementation: `SharpeFitness` ✅

**Inherits**: FitnessFunction
**Purpose**: Fitness based on Sharpe ratio
**Status**: Complete

**Implemented Methods:**
- `calculate(self, result: OptimizationResult) -> float` - Line 34

### Implementation: `ReturnFitness` ✅

**Inherits**: FitnessFunction
**Purpose**: Fitness based on total return
**Status**: Complete

**Implemented Methods:**
- `calculate(self, result: OptimizationResult) -> float` - Line 42

### Implementation: `CompositeFitness` ✅

**Inherits**: FitnessFunction
**Purpose**: Composite fitness combining multiple metrics
**Status**: Complete

**Implemented Methods:**
- `calculate(self, result: OptimizationResult) -> float` - Line 73

### Implementation: `FitnessEvaluator` ✅

**Purpose**: Evaluates fitness of trading strategies
**Status**: Complete

**Implemented Methods:**
- `evaluate(self, result: OptimizationResult) -> float` - Line 143
- `evaluate_multi_objective(self, result: OptimizationResult) -> dict[str, float]` - Line 165
- `compare(self, result1: OptimizationResult, result2: OptimizationResult) -> int` - Line 218
- `rank(self, results: list[OptimizationResult]) -> list[int]` - Line 239

### Implementation: `AdaptiveFitness` ✅

**Inherits**: FitnessFunction
**Purpose**: Adaptive fitness that changes based on market conditions
**Status**: Complete

**Implemented Methods:**
- `set_market_regime(self, regime: str) -> None` - Line 277
- `calculate(self, result: OptimizationResult) -> float` - Line 284

### Implementation: `GeneticConfig` ✅

**Inherits**: BaseModel
**Purpose**: Configuration for genetic algorithm
**Status**: Complete

### Implementation: `GeneticAlgorithm` ✅

**Purpose**: Genetic Algorithm for evolving trading strategies
**Status**: Complete

**Implemented Methods:**
- `async evolve(self) -> Individual` - Line 109
- `get_evolution_summary(self) -> dict[str, Any]` - Line 452

### Implementation: `MutationOperator` ✅

**Purpose**: Mutation operators for genetic algorithms
**Status**: Complete

**Implemented Methods:**
- `mutate(self, genes: dict[str, Any], parameter_ranges: dict[str, tuple[Any, Any]]) -> dict[str, Any]` - Line 40
- `increment_generation(self) -> None` - Line 112

### Implementation: `CrossoverOperator` ✅

**Purpose**: Crossover operators for genetic algorithms
**Status**: Complete

**Implemented Methods:**
- `crossover(self, parent1_genes: dict[str, Any], parent2_genes: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]` - Line 139

### Implementation: `AdvancedMutationOperator` ✅

**Inherits**: MutationOperator
**Purpose**: Advanced mutation operator with multiple strategies
**Status**: Complete

**Implemented Methods:**
- `mutate(self, genes: dict[str, Any], parameter_ranges: dict[str, tuple[Any, Any]]) -> dict[str, Any]` - Line 324

### Implementation: `ActivationType` ✅

**Inherits**: Enum
**Purpose**: Supported activation function types for neural networks
**Status**: Complete

### Implementation: `NodeType` ✅

**Inherits**: Enum
**Purpose**: Neural network node types for NEAT genome representation
**Status**: Complete

### Implementation: `ConnectionType` ✅

**Inherits**: Enum
**Purpose**: Connection types for network topology
**Status**: Complete

### Implementation: `NodeGene` ✅

**Purpose**: Represents a node in the NEAT genome
**Status**: Complete

**Implemented Methods:**
- `copy(self) -> 'NodeGene'` - Line 119

### Implementation: `ConnectionGene` ✅

**Purpose**: Represents a connection in the NEAT genome
**Status**: Complete

**Implemented Methods:**
- `copy(self) -> 'ConnectionGene'` - Line 151

### Implementation: `InnovationTracker` ✅

**Purpose**: Tracks innovation numbers for topology mutations in NEAT algorithm
**Status**: Complete

**Implemented Methods:**
- `get_connection_innovation(self, from_node: int, to_node: int) -> int` - Line 177
- `get_node_innovation(self, connection_id: int) -> int` - Line 193
- `get_next_node_id(self) -> int` - Line 207

### Implementation: `NEATGenome` ✅

**Purpose**: NEAT genome representation for evolving neural network topologies
**Status**: Complete

**Implemented Methods:**
- `add_node_mutation(self) -> bool` - Line 292
- `add_connection_mutation(self, allow_recurrent: bool = True) -> bool` - Line 359
- `mutate_weights(self, mutation_rate: float = 0.8, mutation_strength: float = 0.1) -> None` - Line 430
- `mutate_activation_functions(self, mutation_rate: float = 0.1) -> None` - Line 456
- `enable_disable_mutation(self, enable_rate: float = 0.01, disable_rate: float = 0.01) -> None` - Line 472
- `calculate_compatibility_distance(self, other: 'NEATGenome', c1: float = 1.0, c2: float = 1.0, c3: float = 0.4) -> float` - Line 498
- `crossover(self, other: 'NEATGenome', fitness_equal: bool = False) -> 'NEATGenome'` - Line 550
- `copy(self) -> 'NEATGenome'` - Line 613

### Implementation: `NeuroNetwork` ✅

**Purpose**: PyTorch implementation of evolvable neural network from NEAT genome
**Status**: Complete

**Implemented Methods:**
- `forward(self, x: 'Tensor', reset_hidden: bool = False) -> 'Tensor'` - Line 712
- `predict_signal(self, market_data: MarketData) -> Signal` - Line 797

### Implementation: `Species` ✅

**Purpose**: Represents a species in NEAT population for diversity preservation
**Status**: Complete

**Implemented Methods:**
- `add_member(self, genome: NEATGenome) -> None` - Line 917
- `calculate_average_fitness(self) -> None` - Line 922
- `select_parents(self, selection_pressure: float = 0.5) -> list[NEATGenome]` - Line 929
- `remove_worst_genomes(self, keep_ratio: float = 0.5) -> None` - Line 948

### Implementation: `SpeciationManager` ✅

**Purpose**: Manages species formation and evolution in NEAT population
**Status**: Complete

**Implemented Methods:**
- `speciate_population(self, population: list[NEATGenome]) -> None` - Line 987
- `allocate_offspring(self, total_offspring: int) -> dict[int, int]` - Line 1071

### Implementation: `NeuroEvolutionConfig` ✅

**Purpose**: Configuration for neuroevolution strategy
**Status**: Complete

### Implementation: `NeuroEvolutionStrategy` ✅

**Inherits**: BaseStrategy
**Purpose**: Neural network evolution strategy for trading decisions
**Status**: Complete

**Implemented Methods:**
- `async adapt_networks(self) -> None` - Line 1293
- `async evolve_population(self, fitness_evaluator: FitnessEvaluator) -> None` - Line 1374
- `async validate_signal(self, signal: Signal) -> bool` - Line 1526
- `get_position_size(self, signal: Signal) -> Decimal` - Line 1550
- `should_exit(self, position: Position, data: MarketData) -> bool` - Line 1574
- `get_strategy_info(self) -> dict[str, Any]` - Line 1628
- `get_evolution_summary(self) -> dict[str, Any]` - Line 1658
- `async save_population(self, filepath: str) -> None` - Line 1687
- `async load_population(self, filepath: str) -> None` - Line 1718

### Implementation: `OptimizationObjective` ✅

**Inherits**: BaseModel
**Purpose**: Optimization objective definition
**Status**: Complete

**Implemented Methods:**
- `validate_direction(self) -> None` - Line 56

### Implementation: `MultiObjectiveConfig` ✅

**Inherits**: BaseModel
**Purpose**: Configuration for multi-objective optimization
**Status**: Complete

### Implementation: `ParetoSolution` ✅

**Purpose**: Represents a solution in the Pareto frontier
**Status**: Complete

### Implementation: `ConstraintHandler` ✅

**Purpose**: Handles constraints in multi-objective optimization
**Status**: Complete

**Implemented Methods:**
- `evaluate_constraints(self, objectives: dict[str, float]) -> dict[str, float]` - Line 135
- `is_feasible(self, objectives: dict[str, float], tolerance: float = 0.01) -> bool` - Line 168
- `apply_penalty(self, objectives: dict[str, float], constraint_violations: dict[str, float]) -> dict[str, float]` - Line 183

### Implementation: `DominanceComparator` ✅

**Purpose**: Implements dominance comparison for multi-objective optimization
**Status**: Complete

**Implemented Methods:**
- `dominates(self, solution1: dict[str, float], solution2: dict[str, float]) -> bool` - Line 240
- `non_dominated_sort(self, solutions: list[dict[str, float]]) -> list[list[int]]` - Line 280

### Implementation: `CrowdingDistanceCalculator` ✅

**Purpose**: Calculates crowding distance for diversity preservation
**Status**: Complete

**Implemented Methods:**
- `calculate_crowding_distance(self, solutions: list[dict[str, float]], front_indices: list[int]) -> list[float]` - Line 353

### Implementation: `ParetoFrontierManager` ✅

**Purpose**: Manages the Pareto frontier and provides analysis tools
**Status**: Complete

**Implemented Methods:**
- `update_frontier(self, solutions: list[ParetoSolution]) -> None` - Line 435
- `get_frontier_summary(self) -> dict[str, Any]` - Line 617

### Implementation: `NSGAIIOptimizer` ✅

**Purpose**: NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation
**Status**: Complete

**Implemented Methods:**
- `async optimize(self) -> list[ParetoSolution]` - Line 712
- `get_optimization_summary(self) -> dict[str, Any]` - Line 1157

### Implementation: `MultiObjectiveOptimizer` ✅

**Purpose**: Main interface for multi-objective optimization of trading strategies
**Status**: Complete

**Implemented Methods:**
- `async optimize_strategy(self, ...) -> list[ParetoSolution]` - Line 1218
- `get_pareto_frontier_data(self) -> dict[str, Any]` - Line 1270
- `export_results(self, filepath: str) -> None` - Line 1309

### Implementation: `Individual` ✅

**Purpose**: Represents an individual in the population
**Status**: Complete

**Implemented Methods:**
- `copy(self) -> 'Individual'` - Line 32

### Implementation: `Population` ✅

**Purpose**: Manages a population of individuals
**Status**: Complete

**Implemented Methods:**
- `get_best(self) -> Individual | None` - Line 69
- `get_worst(self) -> Individual | None` - Line 76
- `get_top_n(self, n: int) -> list[Individual]` - Line 83
- `get_bottom_n(self, n: int) -> list[Individual]` - Line 88
- `get_bottom_n_indices(self, n: int) -> list[int]` - Line 93
- `get_statistics(self) -> dict[str, float]` - Line 98
- `add(self, individual: Individual) -> None` - Line 119
- `remove(self, individual: Individual) -> None` - Line 124
- `replace(self, index: int, individual: Individual) -> None` - Line 130

### Implementation: `StrategyFactory` ✅

**Inherits**: StrategyFactoryInterface
**Purpose**: Factory for creating strategies with proper dependency injection
**Status**: Complete

**Implemented Methods:**
- `register_strategy_type(self, strategy_type: StrategyType, strategy_class: type) -> None` - Line 227
- `async create_strategy(self, strategy_type: StrategyType, config: StrategyConfig) -> BaseStrategyInterface` - Line 252
- `get_supported_strategies(self) -> list[StrategyType]` - Line 538
- `validate_strategy_requirements(self, strategy_type: StrategyType, config: StrategyConfig) -> bool` - Line 562
- `async create_strategy_with_validation(self, ...) -> BaseStrategyInterface` - Line 745
- `get_strategy_info(self, strategy_type: StrategyType) -> dict[str, Any]` - Line 863
- `list_available_strategies(self) -> dict[str, Any]` - Line 888

### Implementation: `StrategyPerformanceTracker` ✅

**Purpose**: Tracks individual strategy performance within the ensemble
**Status**: Complete

**Implemented Methods:**
- `add_signal(self, signal: Signal) -> None` - Line 61
- `add_trade_result(self, return_pct: float, trade_info: dict[str, Any]) -> None` - Line 72
- `get_recent_performance(self, window: int = 10) -> float` - Line 121
- `get_performance_score(self) -> float` - Line 128
- `get_metrics(self) -> dict[str, Any]` - Line 146

### Implementation: `CorrelationAnalyzer` ✅

**Purpose**: Analyzes correlations between strategies for diversity maintenance
**Status**: Complete

**Implemented Methods:**
- `add_strategy_return(self, strategy_name: str, return_value: float) -> None` - Line 171
- `calculate_correlation_matrix(self) -> dict[str, dict[str, float]]` - Line 184
- `get_diversity_score(self) -> float` - Line 225
- `identify_redundant_strategies(self, threshold: float = 0.8) -> list[tuple[str, str]]` - Line 249

### Implementation: `VotingMechanism` ✅

**Purpose**: Implements different voting mechanisms for ensemble decisions
**Status**: Complete

**Implemented Methods:**
- `majority_vote(signals: list[tuple[str, Signal]]) -> Signal | None` - Line 268
- `weighted_vote(signals: list[tuple[str, Signal]], weights: dict[str, float]) -> Signal | None` - Line 307
- `confidence_weighted_vote(signals: list[tuple[str, Signal]], performance_scores: dict[str, float]) -> Signal | None` - Line 365

### Implementation: `EnsembleStrategy` ✅

**Inherits**: BaseStrategy
**Purpose**: Adaptive ensemble strategy that combines multiple trading strategies
**Status**: Complete

**Implemented Methods:**
- `add_strategy(self, strategy: BaseStrategy, initial_weight: float = 1.0) -> None` - Line 476
- `remove_strategy(self, strategy_name: str) -> None` - Line 506
- `async validate_signal(self, signal: Signal) -> bool` - Line 732
- `get_position_size(self, signal: Signal) -> Decimal` - Line 762
- `should_exit(self, position: Position, data: MarketData) -> bool` - Line 793
- `update_strategy_performance(self, strategy_name: str, return_pct: float, trade_info: dict[str, Any]) -> None` - Line 885
- `get_ensemble_statistics(self) -> dict[str, Any]` - Line 905
- `get_strategy_info(self) -> dict[str, Any]` - Line 918

### Implementation: `FallbackMode` ✅

**Inherits**: Enum
**Purpose**: Fallback operation modes
**Status**: Complete

### Implementation: `FailureType` ✅

**Inherits**: Enum
**Purpose**: Types of failures that can trigger fallback
**Status**: Complete

### Implementation: `FailureDetector` ✅

**Purpose**: Detects various types of strategy failures
**Status**: Complete

**Implemented Methods:**
- `add_trade_result(self, return_pct: float, timestamp: datetime) -> None` - Line 93
- `add_error(self, error_type: str, timestamp: datetime) -> None` - Line 103
- `add_timeout(self, duration: float, timestamp: datetime) -> None` - Line 113
- `add_signal(self, signal: Signal) -> None` - Line 123
- `detect_performance_degradation(self) -> dict[str, Any]` - Line 131
- `detect_technical_issues(self) -> dict[str, Any]` - Line 187
- `detect_confidence_issues(self) -> dict[str, Any]` - Line 231
- `detect_market_conditions(self, market_data: list[MarketData]) -> dict[str, Any]` - Line 250

### Implementation: `SafeModeStrategy` ✅

**Purpose**: Emergency safe mode strategy with minimal risk
**Status**: Complete

**Implemented Methods:**
- `async generate_signal(self, data: MarketData, price_history: list[float]) -> Signal | None` - Line 283

### Implementation: `DegradedModeStrategy` ✅

**Purpose**: Degraded mode strategy with simplified logic
**Status**: Complete

**Implemented Methods:**
- `async generate_signal(self, data: MarketData, price_history: list[float]) -> Signal | None` - Line 350

### Implementation: `FallbackStrategy` ✅

**Inherits**: BaseStrategy
**Purpose**: Intelligent fallback strategy with automatic failure detection and recovery
**Status**: Complete

**Implemented Methods:**
- `set_primary_strategy(self, strategy: BaseStrategy) -> None` - Line 477
- `async validate_signal(self, signal: Signal) -> bool` - Line 819
- `get_position_size(self, signal: Signal) -> Decimal` - Line 859
- `should_exit(self, position: Position, data: MarketData) -> bool` - Line 900
- `update_trade_result(self, return_pct: float, trade_info: dict[str, Any]) -> None` - Line 952
- `get_fallback_statistics(self) -> dict[str, Any]` - Line 968
- `get_strategy_info(self) -> dict[str, Any]` - Line 981

### Implementation: `TechnicalRuleEngine` ✅

**Purpose**: Traditional technical analysis rule engine
**Status**: Complete

**Implemented Methods:**
- `async calculate_rsi(self, symbol: str) -> float` - Line 76
- `async calculate_moving_averages(self, symbol: str, current_price: Decimal) -> tuple[Decimal, Decimal]` - Line 96
- `async evaluate_rules(self, ...) -> dict[str, Any]` - Line 117
- `update_rule_performance(self, rule: str, performance: float) -> None` - Line 203
- `adjust_rule_weights(self) -> None` - Line 212

### Implementation: `AIPredictor` ✅

**Purpose**: AI prediction component using machine learning
**Status**: Complete

**Implemented Methods:**
- `prepare_features(self, price_history: list[float], volume_history: list[float]) -> np.ndarray` - Line 244
- `async train_model(self, training_data: list[dict[str, Any]]) -> None` - Line 306
- `async predict(self, price_history: list[float], volume_history: list[float]) -> dict[str, Any]` - Line 354
- `update_performance(self, prediction: dict[str, Any], actual_outcome: float) -> None` - Line 408
- `get_performance_metrics(self) -> dict[str, float]` - Line 418

### Implementation: `RuleBasedAIStrategy` ✅

**Inherits**: BaseStrategy
**Purpose**: Hybrid strategy combining traditional technical analysis rules with AI predictions
**Status**: Complete

**Implemented Methods:**
- `strategy_type(self) -> StrategyType` - Line 494
- `async validate_signal(self, signal: Signal) -> bool` - Line 722
- `get_position_size(self, signal: Signal) -> Decimal` - Line 751
- `async should_exit(self, position: Position, data: MarketData) -> bool` - Line 783
- `adjust_component_weights(self) -> None` - Line 883
- `get_strategy_statistics(self) -> dict[str, Any]` - Line 931
- `get_strategy_stats(self) -> dict[str, Any]` - Line 946

### Implementation: `BacktestingInterface` ✅

**Inherits**: Protocol
**Purpose**: Protocol for backtesting integration with strategies
**Status**: Complete

**Implemented Methods:**
- `async prepare_for_backtest(self, config: dict[str, Any]) -> None` - Line 34
- `async process_historical_data(self, data: MarketData) -> list[Signal]` - Line 43
- `async simulate_trade_execution(self, signal: Signal, market_data: MarketData) -> dict[str, Any]` - Line 55
- `async get_backtest_metrics(self) -> dict[str, Any]` - Line 70
- `async reset_backtest_state(self) -> None` - Line 79

### Implementation: `PerformanceMonitoringInterface` ✅

**Inherits**: Protocol
**Purpose**: Protocol for strategy performance monitoring
**Status**: Complete

**Implemented Methods:**
- `update_performance_metrics(self, trade_result: dict[str, Any]) -> None` - Line 87
- `get_real_time_metrics(self) -> dict[str, Any]` - Line 96
- `calculate_risk_adjusted_returns(self) -> dict[str, Decimal]` - Line 105
- `get_drawdown_analysis(self) -> dict[str, Any]` - Line 114

### Implementation: `RiskManagementInterface` ✅

**Inherits**: Protocol
**Purpose**: Protocol for strategy risk management integration
**Status**: Complete

**Implemented Methods:**
- `async validate_risk_limits(self, signal: Signal) -> bool` - Line 127
- `calculate_position_size(self, signal: Signal, account_balance: Decimal) -> Decimal` - Line 139
- `should_close_position(self, position: Position, current_data: MarketData) -> bool` - Line 152

### Implementation: `BaseStrategyInterface` 🔧

**Inherits**: ABC
**Purpose**: Base interface that all strategies must implement
**Status**: Abstract Base Class

**Implemented Methods:**
- `strategy_type(self) -> StrategyType` - Line 176
- `name(self) -> str` - Line 182
- `version(self) -> str` - Line 188
- `status(self) -> StrategyStatus` - Line 194
- `async initialize(self, config: StrategyConfig) -> None` - Line 200
- `async generate_signals(self, data: MarketData) -> list[Signal]` - Line 210
- `async validate_signal(self, signal: Signal) -> bool` - Line 223
- `get_position_size(self, signal: Signal) -> Decimal` - Line 236
- `should_exit(self, position: Position, data: MarketData) -> bool` - Line 249
- `async start(self) -> None` - Line 264
- `async stop(self) -> None` - Line 269
- `async pause(self) -> None` - Line 274
- `async resume(self) -> None` - Line 279
- `async prepare_for_backtest(self, config: dict[str, Any]) -> None` - Line 285
- `async process_historical_data(self, data: MarketData) -> list[Signal]` - Line 290
- `async get_backtest_metrics(self) -> dict[str, Any]` - Line 295
- `get_performance_summary(self) -> dict[str, Any]` - Line 301
- `get_real_time_metrics(self) -> dict[str, Any]` - Line 306
- `async get_state(self) -> dict[str, Any]` - Line 311

### Implementation: `TrendStrategyInterface` 🔧

**Inherits**: BaseStrategyInterface
**Purpose**: Interface for trend-following strategies
**Status**: Abstract Base Class

**Implemented Methods:**
- `calculate_trend_strength(self, data: MarketData) -> float` - Line 325
- `identify_trend_direction(self, data: MarketData) -> int` - Line 338
- `get_trend_confirmation(self, data: MarketData) -> bool` - Line 351

### Implementation: `MeanReversionStrategyInterface` 🔧

**Inherits**: BaseStrategyInterface
**Purpose**: Interface for mean reversion strategies
**Status**: Abstract Base Class

**Implemented Methods:**
- `calculate_mean_deviation(self, data: MarketData) -> float` - Line 368
- `is_oversold(self, data: MarketData) -> bool` - Line 381
- `is_overbought(self, data: MarketData) -> bool` - Line 394
- `calculate_reversion_probability(self, data: MarketData) -> float` - Line 407

### Implementation: `ArbitrageStrategyInterface` 🔧

**Inherits**: BaseStrategyInterface
**Purpose**: Interface for arbitrage strategies
**Status**: Abstract Base Class

**Implemented Methods:**
- `async identify_arbitrage_opportunities(self, market_data_sources: list[MarketData]) -> list[dict[str, Any]]` - Line 424
- `calculate_profit_potential(self, opportunity: dict[str, Any]) -> Decimal` - Line 439
- `validate_arbitrage_execution(self, opportunity: dict[str, Any]) -> bool` - Line 452

### Implementation: `MarketMakingStrategyInterface` 🔧

**Inherits**: BaseStrategyInterface
**Purpose**: Interface for market making strategies
**Status**: Abstract Base Class

**Implemented Methods:**
- `calculate_optimal_spread(self, data: MarketData) -> tuple[Decimal, Decimal]` - Line 469
- `manage_inventory(self, current_position: Position) -> dict[str, Any]` - Line 482
- `calculate_quote_adjustment(self, market_impact: float) -> float` - Line 495

### Implementation: `StrategyFactoryInterface` 🔧

**Inherits**: ABC
**Purpose**: Interface for strategy factories
**Status**: Abstract Base Class

**Implemented Methods:**
- `async create_strategy(self, strategy_type: StrategyType, config: StrategyConfig) -> BaseStrategyInterface` - Line 512
- `get_supported_strategies(self) -> list[StrategyType]` - Line 528
- `validate_strategy_requirements(self, strategy_type: StrategyType, config: StrategyConfig) -> bool` - Line 538

### Implementation: `BacktestingServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for backtesting service integration
**Status**: Complete

**Implemented Methods:**
- `async run_backtest(self, strategy: BaseStrategyInterface, config: dict[str, Any]) -> dict[str, Any]` - Line 557
- `async validate_backtest_config(self, config: dict[str, Any]) -> bool` - Line 572
- `async get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> list[MarketData]` - Line 584

### Implementation: `StrategyRegistryInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for strategy registry
**Status**: Complete

**Implemented Methods:**
- `register_strategy(self, strategy_id: str, strategy: BaseStrategyInterface) -> None` - Line 604
- `get_strategy(self, strategy_id: str) -> BaseStrategyInterface | None` - Line 614
- `list_strategies(self) -> list[str]` - Line 626
- `remove_strategy(self, strategy_id: str) -> bool` - Line 635

### Implementation: `MarketDataProviderInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for market data access
**Status**: Complete

**Implemented Methods:**
- `async get_current_price(self, symbol: str) -> Decimal | None` - Line 651
- `async get_market_regime(self, symbol: str) -> MarketRegime` - Line 663

### Implementation: `StrategyDataRepositoryInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for strategy data persistence
**Status**: Complete

**Implemented Methods:**
- `async load_strategy_state(self, strategy_id: str) -> dict[str, Any] | None` - Line 684
- `async save_strategy_state(self, strategy_id: str, state: dict[str, Any]) -> None` - Line 696
- `async get_strategy_trades(self, ...) -> list[dict[str, Any]]` - Line 706
- `async save_trade(self, strategy_id: str, trade: dict[str, Any]) -> None` - Line 722
- `async get_strategy_positions(self, strategy_id: str) -> list[dict[str, Any]]` - Line 732
- `async save_performance_metrics(self, ...) -> None` - Line 744
- `async load_performance_history(self, ...) -> list[dict[str, Any]]` - Line 757

### Implementation: `StrategyServiceInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for strategy service business logic operations
**Status**: Complete

**Implemented Methods:**
- `async register_strategy(self, strategy_id: str, strategy_instance: Any, config: StrategyConfig) -> None` - Line 777
- `async start_strategy(self, strategy_id: str) -> None` - Line 783
- `async stop_strategy(self, strategy_id: str) -> None` - Line 787
- `async process_market_data(self, market_data: MarketData) -> dict[str, list[Signal]]` - Line 791
- `async validate_signal(self, strategy_id: str, signal: Signal) -> bool` - Line 795
- `async get_strategy_performance(self, strategy_id: str) -> dict[str, Any]` - Line 799
- `async get_all_strategies(self) -> dict[str, dict[str, Any]]` - Line 803
- `async cleanup_strategy(self, strategy_id: str) -> None` - Line 807

### Implementation: `OptimizationResult` ✅

**Purpose**: Generic optimization result that doesn't depend on backtesting module
**Status**: Complete

**Implemented Methods:**

### Implementation: `OptimizationConfig` ✅

**Purpose**: Generic optimization configuration that doesn't depend on backtesting module
**Status**: Complete

**Implemented Methods:**

### Implementation: `OptimizationEngineInterface` ✅

**Inherits**: Protocol
**Purpose**: Interface for optimization engines that genetic algorithms can use
**Status**: Complete

**Implemented Methods:**
- `async run_optimization(self, strategy: Any, config: OptimizationConfig) -> OptimizationResult` - Line 857

### Implementation: `PerformanceMetrics` ✅

**Inherits**: BaseModel
**Purpose**: Comprehensive performance metrics for strategies
**Status**: Complete

### Implementation: `MetricsCalculator` ✅

**Purpose**: Calculator for strategy performance metrics
**Status**: Complete

**Implemented Methods:**
- `async calculate_comprehensive_metrics(self, ...) -> PerformanceMetrics` - Line 107

### Implementation: `RealTimeMetricsTracker` ✅

**Purpose**: Real-time metrics tracker for live strategy monitoring
**Status**: Complete

**Implemented Methods:**
- `async update_equity(self, equity: float, timestamp: datetime | None = None) -> None` - Line 466
- `async add_trade(self, trade_data: dict[str, Any]) -> None` - Line 490
- `async add_signal(self, signal: Signal) -> None` - Line 505
- `get_current_metrics(self) -> PerformanceMetrics` - Line 547
- `get_metrics_summary(self) -> dict[str, Any]` - Line 556
- `async reset_metrics(self) -> None` - Line 575

### Implementation: `StrategyComparator` ✅

**Purpose**: Compare performance between multiple strategies
**Status**: Complete

**Implemented Methods:**
- `async compare_strategies(self, strategy_metrics: dict[str, PerformanceMetrics]) -> dict[str, Any]` - Line 598

### Implementation: `PerformanceMetrics` ✅

**Purpose**: Comprehensive performance metrics for a trading strategy
**Status**: Complete

**Implemented Methods:**

### Implementation: `PerformanceMonitor` ✅

**Purpose**: Comprehensive performance monitoring system for trading strategies
**Status**: Complete

**Implemented Methods:**
- `async add_strategy(self, strategy: BaseStrategyInterface) -> None` - Line 177
- `async remove_strategy(self, strategy_name: str) -> None` - Line 219
- `async start_monitoring(self) -> None` - Line 239
- `async stop_monitoring(self) -> None` - Line 247
- `async get_strategy_performance(self, strategy_name: str) -> dict[str, Any]` - Line 789
- `async get_comparative_analysis(self) -> dict[str, Any]` - Line 881

### Implementation: `StrategyAllocation` ✅

**Purpose**: Represents allocation for a single strategy
**Status**: Complete

**Implemented Methods:**

### Implementation: `PortfolioAllocator` ✅

**Purpose**: Dynamic portfolio allocator for trading strategies
**Status**: Complete

**Implemented Methods:**
- `async add_strategy(self, strategy: BaseStrategyInterface, initial_weight: float = 0.1) -> bool` - Line 165
- `async rebalance_portfolio(self) -> dict[str, Any]` - Line 324
- `async update_market_regime(self, new_regime: MarketRegime) -> None` - Line 752
- `async remove_strategy(self, strategy_name: str, reason: str = 'manual') -> bool` - Line 774
- `get_strategy_allocation(self, strategy: BaseStrategyInterface) -> StrategyAllocation | None` - Line 818
- `async calculate_optimal_weights(self) -> dict[str, float]` - Line 832
- `update_strategy_performance(self, strategy: BaseStrategyInterface, performance_data: dict[str, float]) -> bool` - Line 841
- `get_allocation_status(self) -> dict[str, Any]` - Line 874
- `async should_rebalance(self) -> bool` - Line 929

### Implementation: `StrategyRepositoryInterface` 🔧

**Inherits**: ABC
**Purpose**: Interface for strategy data repository operations
**Status**: Abstract Base Class

**Implemented Methods:**
- `async create_strategy(self, strategy: Strategy) -> Strategy` - Line 37
- `async get_strategy(self, strategy_id: str) -> Strategy | None` - Line 42
- `async update_strategy(self, strategy_id: str, updates: dict[str, Any]) -> Strategy | None` - Line 47
- `async delete_strategy(self, strategy_id: str) -> bool` - Line 52
- `async get_strategies_by_bot(self, bot_id: str) -> list[Strategy]` - Line 57
- `async get_active_strategies(self, bot_id: str | None = None) -> list[Strategy]` - Line 62
- `async save_strategy_state(self, strategy_id: str, state_data: dict[str, Any]) -> bool` - Line 67
- `async load_strategy_state(self, strategy_id: str) -> dict[str, Any] | None` - Line 72
- `async save_strategy_metrics(self, strategy_id: str, metrics: StrategyMetrics) -> bool` - Line 77
- `async get_strategy_metrics(self, ...) -> list[AnalyticsStrategyMetrics]` - Line 82
- `async save_strategy_signals(self, signals: list[Signal]) -> list[Signal]` - Line 89
- `async get_strategy_signals(self, strategy_id: str, limit: int | None = None) -> list[Signal]` - Line 94

### Implementation: `StrategyRepository` ✅

**Inherits**: DatabaseRepository, StrategyRepositoryInterface
**Purpose**: Strategy repository with database integration using UoW pattern
**Status**: Complete

**Implemented Methods:**
- `async create_strategy(self, strategy: Strategy) -> Strategy` - Line 114
- `async get_strategy(self, strategy_id: str) -> Strategy | None` - Line 126
- `async update_strategy(self, strategy_id: str, updates: dict[str, Any]) -> Strategy | None` - Line 163
- `async delete_strategy(self, strategy_id: str) -> bool` - Line 204
- `async get_strategies_by_bot(self, bot_id: str) -> list[Strategy]` - Line 220
- `async get_active_strategies(self, bot_id: str | None = None) -> list[Strategy]` - Line 246
- `async save_strategy_state(self, strategy_id: str, state_data: dict[str, Any]) -> bool` - Line 273
- `async load_strategy_state(self, strategy_id: str) -> dict[str, Any] | None` - Line 305
- `async save_strategy_metrics(self, strategy_id: str, metrics: StrategyMetrics) -> bool` - Line 342
- `async get_strategy_metrics(self, ...) -> list[AnalyticsStrategyMetrics]` - Line 382
- `async save_strategy_signals(self, signals: list[Signal]) -> list[Signal]` - Line 414
- `async get_strategy_signals(self, strategy_id: str, limit: int | None = None) -> list[Signal]` - Line 435
- `async get_strategy_trades(self, ...) -> list[Trade]` - Line 465
- `async get_strategy_performance_summary(self, strategy_id: str) -> dict[str, Any]` - Line 497

### Implementation: `StrategyService` ✅

**Inherits**: BaseService, StrategyServiceInterface
**Purpose**: Service layer for strategy operations and management
**Status**: Complete

**Implemented Methods:**
- `async register_strategy(self, strategy_id: str, strategy_instance: Any, config: StrategyConfig) -> None` - Line 234
- `async start_strategy(self, strategy_id: str) -> None` - Line 331
- `async stop_strategy(self, strategy_id: str) -> None` - Line 361
- `async process_market_data(self, market_data: MarketData) -> dict[str, list[Signal]]` - Line 385
- `async validate_signal(self, strategy_id: str, signal: Signal) -> bool` - Line 500
- `async validate_strategy_config(self, config: StrategyConfig) -> bool` - Line 542
- `async get_strategy_performance(self, strategy_id: str) -> dict[str, Any]` - Line 782
- `async get_cached_strategy_metrics(self, strategy_id: str) -> dict[str, Any] | None` - Line 826
- `async get_strategy_performance_with_cache(self, strategy_id: str) -> dict[str, Any]` - Line 847
- `async get_all_strategies(self) -> dict[str, dict[str, Any]]` - Line 892
- `async cleanup_strategy(self, strategy_id: str) -> None` - Line 907
- `get_metrics(self) -> dict[str, Any]` - Line 978
- `resolve_dependency(self, dependency_name: str) -> Any` - Line 1008

### Implementation: `StrategyIntegratedBase` 🔧

**Inherits**: ABC
**Purpose**: Enhanced base class that provides comprehensive module integration
**Status**: Abstract Base Class

**Implemented Methods:**
- `async initialize_validation_service(self) -> None` - Line 112
- `set_monitoring_services(self, ...)` - Line 143
- `async validate_market_data_comprehensive(self, data: MarketData) -> tuple[bool, list[str]]` - Line 171
- `async calculate_technical_indicators(self, data: MarketData, indicators: list[str], periods: dict[str, int] = None) -> 'dict[str, Decimal | None]'` - Line 238
- `format_signal_metadata(self, signal: Signal, additional_data: dict[str, Any] = None) -> dict[str, Any]` - Line 354
- `async record_signal_metrics(self, ...)` - Line 395
- `async safe_execute_with_monitoring(self, operation_name: str, operation_func, *args, **kwargs) -> Any` - Line 441
- `get_comprehensive_status(self) -> dict[str, Any]` - Line 516
- `async cleanup_resources(self)` - Line 532

### Implementation: `StrategyDataAccessMixin` ✅

**Purpose**: Mixin providing data access patterns for strategies
**Status**: Complete

**Implemented Methods:**
- `async get_indicator_data(self, symbol: str, indicator: str, period: int) -> 'Decimal | None'` - Line 569

### Implementation: `ArbitrageOpportunity` ✅

**Inherits**: BaseStrategy
**Purpose**: Arbitrage opportunity scanner for detecting and prioritizing arbitrage opportunities
**Status**: Complete

**Implemented Methods:**
- `strategy_type(self) -> StrategyType` - Line 93
- `async validate_signal(self, signal: Signal) -> bool` - Line 352
- `get_position_size(self, signal: Signal) -> Decimal` - Line 421
- `async should_exit(self, position: Position, data: MarketData) -> bool` - Line 476
- `async post_trade_processing(self, trade_result: dict[str, Any]) -> None` - Line 817

### Implementation: `BreakoutStrategy` ✅

**Inherits**: BaseStrategy
**Purpose**: Breakout Strategy Implementation
**Status**: Complete

**Implemented Methods:**
- `strategy_type(self) -> StrategyType` - Line 106
- `async validate_signal(self, signal: Signal) -> bool` - Line 608
- `get_position_size(self, signal: Signal) -> Decimal` - Line 671
- `async should_exit(self, position: Position, data: MarketData) -> bool` - Line 722
- `get_strategy_info(self) -> dict[str, Any]` - Line 800

### Implementation: `CrossExchangeArbitrageStrategy` ✅

**Inherits**: BaseStrategy
**Purpose**: Cross-exchange arbitrage strategy for detecting and executing price differences
**Status**: Complete

**Implemented Methods:**
- `strategy_type(self) -> StrategyType` - Line 87
- `name(self) -> str` - Line 92
- `name(self, value: str) -> None` - Line 98
- `version(self) -> str` - Line 103
- `version(self, value: str) -> None` - Line 109
- `status(self) -> StrategyStatus` - Line 114
- `status(self, value: StrategyStatus) -> None` - Line 120
- `async validate_signal(self, signal: Signal) -> bool` - Line 391
- `get_position_size(self, signal: Signal) -> Decimal` - Line 448
- `async should_exit(self, position: Position, data: MarketData) -> bool` - Line 541
- `async post_trade_processing(self, trade_result: dict[str, Any]) -> None` - Line 644

### Implementation: `InventoryManager` ✅

**Purpose**: Inventory Manager for Market Making Strategy
**Status**: Complete

**Implemented Methods:**
- `async update_inventory(self, position: Position) -> None` - Line 88
- `async should_rebalance(self) -> bool` - Line 116
- `async calculate_rebalance_orders(self, current_price: Decimal) -> list[OrderRequest]` - Line 163
- `async should_emergency_liquidate(self) -> bool` - Line 238
- `async calculate_emergency_orders(self, current_price: Decimal) -> list[OrderRequest]` - Line 270
- `async calculate_spread_adjustment(self, base_spread: Decimal) -> Decimal` - Line 326
- `async calculate_size_adjustment(self, base_size: Decimal) -> Decimal` - Line 362
- `async record_rebalance(self, cost: Decimal) -> None` - Line 398
- `async record_emergency(self, cost: Decimal) -> None` - Line 421
- `get_inventory_summary(self) -> dict[str, Any]` - Line 442
- `async validate_inventory_limits(self, new_position: Position) -> bool` - Line 471

### Implementation: `OrderLevel` ✅

**Purpose**: Represents a single order level in the market making strategy
**Status**: Complete

### Implementation: `InventoryState` ✅

**Purpose**: Current inventory state for the market making strategy
**Status**: Complete

### Implementation: `MarketMakingStrategy` ✅

**Inherits**: BaseStrategy
**Purpose**: Market Making Strategy Implementation
**Status**: Complete

**Implemented Methods:**
- `strategy_type(self) -> StrategyType` - Line 163
- `async validate_signal(self, signal: Signal) -> bool` - Line 362
- `get_position_size(self, signal: Signal) -> Decimal` - Line 463
- `async should_exit(self, position: Position, data: MarketData) -> bool` - Line 500
- `async update_inventory_state(self, new_position: Position) -> None` - Line 603
- `async update_performance_metrics(self, trade_result: dict[str, Any]) -> None` - Line 632
- `get_strategy_info(self) -> dict[str, Any]` - Line 677

### Implementation: `MeanReversionStrategy` ✅

**Inherits**: BaseStrategy
**Purpose**: Mean Reversion Strategy Implementation
**Status**: Complete

**Implemented Methods:**
- `strategy_type(self) -> StrategyType` - Line 117
- `async validate_signal(self, signal: Signal) -> bool` - Line 333
- `get_position_size(self, signal: Signal) -> Decimal` - Line 391
- `async should_exit(self, position: Position, data: MarketData) -> bool` - Line 438
- `get_strategy_info(self) -> dict[str, Any]` - Line 505

### Implementation: `SpreadOptimizer` ✅

**Purpose**: Spread Optimizer for Market Making Strategy
**Status**: Complete

**Implemented Methods:**
- `async optimize_spread(self, ...) -> Decimal` - Line 93
- `async calculate_optimal_spread(self, ...) -> tuple[Decimal, Decimal]` - Line 398
- `async should_widen_spread(self, market_data: MarketData) -> bool` - Line 448
- `get_optimization_summary(self) -> dict[str, Any]` - Line 498

### Implementation: `TrendFollowingStrategy` ✅

**Inherits**: BaseStrategy
**Purpose**: Trend Following Strategy Implementation
**Status**: Complete

**Implemented Methods:**
- `set_technical_indicators(self, technical_indicators: TechnicalIndicators) -> None` - Line 101
- `strategy_type(self) -> StrategyType` - Line 107
- `async validate_signal(self, signal: Signal) -> bool` - Line 449
- `get_position_size(self, signal: Signal) -> Decimal` - Line 505
- `async should_exit(self, position: Position, data: MarketData) -> bool` - Line 561
- `get_strategy_info(self) -> dict[str, Any]` - Line 688

### Implementation: `TriangularArbitrageStrategy` ✅

**Inherits**: BaseStrategy
**Purpose**: Triangular arbitrage strategy for detecting and executing three-pair arbitrage
**Status**: Complete

**Implemented Methods:**
- `strategy_type(self) -> StrategyType` - Line 82
- `async validate_signal(self, signal: Signal) -> bool` - Line 408
- `get_position_size(self, signal: Signal) -> Decimal` - Line 472
- `async should_exit(self, position: Position, data: MarketData) -> bool` - Line 558
- `async post_trade_processing(self, trade_result: dict[str, Any]) -> None` - Line 619

### Implementation: `ValidationResult` ✅

**Inherits**: BaseModel
**Purpose**: Result of a validation operation
**Status**: Complete

**Implemented Methods:**
- `add_error(self, message: str) -> None` - Line 43
- `add_warning(self, message: str) -> None` - Line 48
- `merge(self, other: 'ValidationResult') -> None` - Line 52

### Implementation: `BaseValidator` 🔧

**Inherits**: ABC
**Purpose**: Base class for all validators
**Status**: Abstract Base Class

**Implemented Methods:**
- `async validate(self, target: Any, context: dict[str, Any] | None = None) -> ValidationResult` - Line 77

### Implementation: `SignalValidator` ✅

**Inherits**: BaseValidator
**Purpose**: Validator for trading signals
**Status**: Complete

**Implemented Methods:**
- `async validate(self, signal: Signal, context: dict[str, Any] | None = None) -> ValidationResult` - Line 105

### Implementation: `StrategyConfigValidator` ✅

**Inherits**: BaseValidator
**Purpose**: Validator for strategy configurations
**Status**: Complete

**Implemented Methods:**
- `async validate(self, config: StrategyConfig, context: dict[str, Any] | None = None) -> ValidationResult` - Line 225

### Implementation: `MarketConditionValidator` ✅

**Inherits**: BaseValidator
**Purpose**: Validator for market conditions and trading environment
**Status**: Complete

**Implemented Methods:**
- `async validate(self, market_data: MarketData, context: dict[str, Any] | None = None) -> ValidationResult` - Line 355

### Implementation: `CompositeValidator` ✅

**Inherits**: BaseValidator
**Purpose**: Composite validator that runs multiple validators
**Status**: Complete

**Implemented Methods:**
- `async validate(self, target: Any, context: dict[str, Any] | None = None) -> ValidationResult` - Line 440

### Implementation: `ValidationFramework` ✅

**Purpose**: Main validation framework for strategies
**Status**: Complete

**Implemented Methods:**
- `async validate_signal(self, signal: Signal, market_data: MarketData | None = None) -> ValidationResult` - Line 497
- `async validate_strategy_config(self, config: StrategyConfig) -> ValidationResult` - Line 516
- `async validate_market_conditions(self, market_data: MarketData) -> ValidationResult` - Line 528
- `async validate_for_trading(self, signal: Signal, market_data: MarketData) -> ValidationResult` - Line 540
- `async batch_validate_signals(self, signals: list[Signal], market_data: MarketData | None = None) -> list[tuple[Signal, ValidationResult]]` - Line 556
- `add_custom_validator(self, validator: BaseValidator, validator_type: str = 'custom') -> None` - Line 578
- `get_validation_stats(self) -> dict[str, Any]` - Line 598

## COMPLETE API REFERENCE

### File: base.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.types import MarketData`
- `from src.core.types import OrderRequest`
- `from src.core.types import Position`
- `from src.core.types import Signal`

#### Class: `BaseStrategy`

**Inherits**: BaseComponent, BaseStrategyInterface
**Purpose**: Base strategy interface that ALL strategies must inherit from

```python
class BaseStrategy(BaseComponent, BaseStrategyInterface):
    def __init__(self, config: dict[str, Any], services: StrategyServiceContainer | None = None)  # Line 96
    def strategy_type(self) -> StrategyType  # Line 160
    def name(self) -> str  # Line 165
    def version(self) -> str  # Line 170
    def status(self) -> StrategyStatus  # Line 175
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]  # Line 180
    async def generate_signals(self, data: MarketData) -> list[Signal]  # Line 197
    async def validate_signal(self, signal: Signal) -> bool  # Line 335
    async def _validate_and_process_signal(self, signal: Signal, market_data: MarketData) -> bool  # Line 348
    def get_position_size(self, signal: Signal) -> Decimal  # Line 391
    def _get_account_balance(self) -> Decimal  # Line 429
    def should_exit(self, position: Position, data: MarketData) -> bool  # Line 442
    async def pre_trade_validation(self, signal: Signal) -> bool  # Line 458
    async def post_trade_processing(self, trade_result: Any) -> None  # Line 484
    def set_risk_manager(self, risk_manager: Any) -> None  # Line 530
    def set_exchange(self, exchange: Any) -> None  # Line 538
    def set_data_service(self, data_service: Any) -> None  # Line 547
    def set_validation_framework(self, validation_framework: ValidationFramework) -> None  # Line 555
    def set_metrics_collector(self, metrics_collector: MetricsCollector) -> None  # Line 564
    def get_strategy_info(self) -> dict[str, Any]  # Line 573
    async def initialize(self, config: StrategyConfig) -> None  # Line 588
    async def start(self) -> bool  # Line 600
    async def stop(self) -> bool  # Line 653
    async def pause(self) -> None  # Line 679
    async def resume(self) -> None  # Line 685
    async def prepare_for_backtest(self, config: dict[str, Any]) -> None  # Line 692
    async def process_historical_data(self, data: MarketData) -> list[Signal]  # Line 708
    async def get_backtest_metrics(self) -> dict[str, Any]  # Line 724
    def get_real_time_metrics(self) -> dict[str, Any]  # Line 737
    def _add_to_signal_history(self, signals: list[Signal]) -> None  # Line 754
    async def _on_initialize(self) -> None  # Line 767
    async def _on_start(self) -> None  # Line 771
    async def _on_stop(self) -> None  # Line 775
    async def _on_backtest_prepare(self) -> None  # Line 779
    def update_config(self, new_config: dict[str, Any]) -> None  # Line 783
    async def get_state(self) -> dict[str, Any]  # Line 798
    def get_performance_summary(self) -> dict[str, Any]  # Line 815
    def cleanup(self) -> None  # Line 845
    async def get_market_data(self, symbol: str) -> MarketData | None  # Line 879
    async def get_historical_data(self, symbol: str, timeframe: str, limit: int = 100) -> list[MarketData]  # Line 901
    async def execute_order(self, signal: Signal) -> Any | None  # Line 929
    async def save_state(self, state_data: dict[str, Any]) -> bool  # Line 975
    async def load_state(self) -> dict[str, Any] | None  # Line 998
    def _update_metrics(self, metrics: dict[str, Any]) -> None  # Line 1021
    def _log_signal(self, signal: Signal) -> None  # Line 1034
    async def _handle_error(self, error: Exception, severity: ErrorSeverity, context: dict[str, Any]) -> None  # Line 1044
    def get_metrics(self) -> dict[str, Any]  # Line 1078
    def is_healthy(self) -> bool  # Line 1094
    async def reset(self) -> bool  # Line 1119
    def _calculate_win_rate(self) -> float  # Line 1129
    def _calculate_sharpe_ratio(self) -> float  # Line 1144
    def _calculate_max_drawdown(self) -> float  # Line 1164
    def set_execution_service(self, execution_service: Any) -> None  # Line 1191
    def get_status(self) -> StrategyStatus  # Line 1197
    def get_status_string(self) -> str  # Line 1201
    async def _cleanup_resources(self) -> None  # Line 1205
    async def _persist_strategy_state(self) -> None  # Line 1213
    def _validate_config(config: dict[str, Any]) -> None  # Line 1228
    async def validate_market_data(self, data: MarketData | None) -> None  # Line 1252
    def __str__(self) -> str  # Line 1274
    def __repr__(self) -> str  # Line 1278
    async def get_sma(self, symbol: str, period: int) -> Decimal | None  # Line 1286
    async def get_ema(self, symbol: str, period: int) -> Decimal | None  # Line 1307
    async def get_rsi(self, symbol: str, period: int = 14) -> Decimal | None  # Line 1326
    async def get_volatility(self, symbol: str, period: int) -> Decimal | None  # Line 1345
    async def get_atr(self, symbol: str, period: int) -> Decimal | None  # Line 1364
    async def get_volume_ratio(self, symbol: str, period: int) -> Decimal | None  # Line 1383
    async def get_bollinger_bands(self, symbol: str, period: int = 20, std_dev: float = 2.0) -> dict[str, Decimal] | None  # Line 1402
    async def get_macd(self, ...) -> dict[str, Decimal] | None  # Line 1429
    async def execute_with_algorithm(self, ...) -> dict[str, Any] | None  # Line 1458
    async def optimize_parameters(self, optimization_config: dict[str, Any] | None) -> dict[str, Any]  # Line 1506
    async def enhance_signals_with_ml(self, signals: list[Signal]) -> list[Signal]  # Line 1580
    async def _get_market_context_for_ml(self) -> dict[str, Any]  # Line 1643
    async def get_allocated_capital(self) -> Decimal  # Line 1686
    async def execute_large_order(self, order_request: OrderRequest, max_position_size: Decimal | None = None) -> dict[str, Any] | None  # Line 1731
    async def get_execution_algorithms_status(self) -> dict[str, Any]  # Line 1793
```

### File: config.py

**Key Imports:**
- `from src.core.exceptions import ConfigurationError`
- `from src.core.logging import get_logger`
- `from src.core.types import StrategyConfig`
- `from src.core.types import StrategyType`
- `from src.utils.decorators import time_execution`

#### Class: `StrategyConfigurationManager`

**Purpose**: Manager for strategy configuration handling

```python
class StrategyConfigurationManager:
    def __init__(self, config_dir: str = 'config/strategies')  # Line 28
    def _initialize_default_configs(self) -> dict[str, dict[str, Any]]  # Line 45
    def load_strategy_config(self, strategy_name: str) -> StrategyConfig  # Line 113
    def _load_config_file(self, config_file: Path) -> dict[str, Any]  # Line 193
    def _get_default_config(self, strategy_name: str) -> dict[str, Any]  # Line 229
    def save_strategy_config(self, strategy_name: str, config: StrategyConfig) -> None  # Line 244
    def validate_config(self, config: dict[str, Any]) -> bool  # Line 299
    def get_available_strategies(self) -> list[str]  # Line 322
    def get_config_schema(self) -> dict[str, Any]  # Line 342
    def update_config_parameter(self, strategy_name: str, parameter: str, value: Any) -> bool  # Line 351
    def create_strategy_config(self, strategy_name: str, strategy_type: StrategyType, symbol: str, **kwargs) -> StrategyConfig  # Line 415
    def delete_strategy_config(self, strategy_name: str) -> bool  # Line 487
    def get_config_summary(self) -> dict[str, Any]  # Line 530
```

### File: config_templates.py

**Key Imports:**
- `from src.core.types import StrategyType`

#### Class: `StrategyConfigTemplates`

**Purpose**: Comprehensive strategy configuration templates for production deployment

```python
class StrategyConfigTemplates:
    def get_arbitrage_scanner_config(risk_level, ...) -> dict[str, Any]  # Line 35
    def get_mean_reversion_config(timeframe: str = '1h', risk_level: str = 'medium') -> dict[str, Any]  # Line 139
    def get_trend_following_config(timeframe: str = '1h', trend_strength: str = 'medium') -> dict[str, Any]  # Line 266
    def get_market_making_config(symbol, ...) -> dict[str, Any]  # Line 383
    def get_volatility_breakout_config(volatility_regime: str = 'medium', breakout_type: str = 'range') -> dict[str, Any]  # Line 499
    def get_ensemble_config(strategy_types, ...) -> dict[str, Any]  # Line 605
    def get_all_templates(cls) -> dict[str, dict[str, Any]]  # Line 688
    def get_template_by_name(cls, template_name: str) -> dict[str, Any]  # Line 733
    def list_available_templates(cls) -> list[str]  # Line 754
    def get_templates_by_strategy_type(cls, strategy_type: str) -> dict[str, dict[str, Any]]  # Line 764
    def validate_template(cls, template: dict[str, Any]) -> tuple[bool, list[str]]  # Line 782
```

### File: controller.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import MarketData`
- `from src.core.types import StrategyConfig`

#### Class: `StrategyController`

**Inherits**: BaseComponent
**Purpose**: Controller for strategy operations

```python
class StrategyController(BaseComponent):
    def __init__(self, strategy_service: StrategyServiceInterface)  # Line 24
    async def register_strategy(self, request_data: dict[str, Any]) -> dict[str, Any]  # Line 29
    async def start_strategy(self, strategy_id: str) -> dict[str, Any]  # Line 64
    async def stop_strategy(self, strategy_id: str) -> dict[str, Any]  # Line 90
    async def process_market_data(self, market_data_dict: dict[str, Any]) -> dict[str, Any]  # Line 116
    async def get_strategy_performance(self, strategy_id: str) -> dict[str, Any]  # Line 146
    async def get_all_strategies(self) -> dict[str, Any]  # Line 168
    async def cleanup_strategy(self, strategy_id: str) -> dict[str, Any]  # Line 184
```

### File: dependencies.py

**Key Imports:**
- `from src.core.base import BaseComponent`
- `from src.core.logging import get_logger`

#### Class: `StrategyServiceContainer`

**Purpose**: Container for all services required by strategies

```python
class StrategyServiceContainer:
    def __post_init__(self)  # Line 57
    def is_ready(self) -> bool  # Line 75
    def get_service_status(self) -> dict[str, bool]  # Line 84
```

#### Class: `StrategyServiceContainerBuilder`

**Inherits**: BaseComponent
**Purpose**: Builder for creating properly configured StrategyServiceContainer

```python
class StrategyServiceContainerBuilder(BaseComponent):
    def __init__(self)  # Line 106
    def with_risk_service(self, risk_service: 'RiskManagementService') -> 'StrategyServiceContainerBuilder'  # Line 110
    def with_data_service(self, data_service: 'DataService') -> 'StrategyServiceContainerBuilder'  # Line 116
    def with_execution_service(self, execution_service: 'ExecutionService') -> 'StrategyServiceContainerBuilder'  # Line 122
    def with_monitoring_service(self, monitoring_service: 'MonitoringService') -> 'StrategyServiceContainerBuilder'  # Line 128
    def with_state_service(self, state_service: 'StateService') -> 'StrategyServiceContainerBuilder'  # Line 134
    def with_capital_service(self, capital_service: 'CapitalManagementService') -> 'StrategyServiceContainerBuilder'  # Line 140
    def with_ml_service(self, ml_service: 'MLService') -> 'StrategyServiceContainerBuilder'  # Line 147
    def with_analytics_service(self, analytics_service: 'AnalyticsService') -> 'StrategyServiceContainerBuilder'  # Line 153
    def with_optimization_service(self, optimization_service: 'OptimizationService') -> 'StrategyServiceContainerBuilder'  # Line 159
    def build(self) -> StrategyServiceContainer  # Line 165
```

#### Functions:

```python
def create_strategy_service_container(...) -> StrategyServiceContainer  # Line 181
```

### File: di_registration.py

**Key Imports:**
- `from src.core.dependency_injection import DependencyContainer`
- `from src.core.logging import get_logger`
- `from src.strategies.factory import StrategyFactory`
- `from src.strategies.dynamic.strategy_factory import DynamicStrategyFactory`
- `from src.strategies.repository import StrategyRepository`

#### Functions:

```python
def register_strategies_dependencies(container: DependencyContainer) -> None  # Line 15
```

### File: adaptive_momentum.py

**Key Imports:**
- `from src.core.types import MarketData`
- `from src.core.types import MarketRegime`
- `from src.core.types import Position`
- `from src.core.types import Signal`
- `from src.core.types import SignalDirection`

#### Class: `AdaptiveMomentumStrategy`

**Inherits**: BaseStrategy
**Purpose**: Enhanced Adaptive Momentum Strategy with service layer integration

```python
class AdaptiveMomentumStrategy(BaseStrategy):
    def name(self) -> str  # Line 61
    def name(self, value: str) -> None  # Line 70
    def version(self) -> str  # Line 75
    def version(self, value: str) -> None  # Line 80
    def status(self) -> str  # Line 85
    def status(self, value: str) -> None  # Line 90
    def __init__(self, config: dict[str, Any], services: 'StrategyServiceContainer | None' = None)  # Line 94
    def strategy_type(self) -> StrategyType  # Line 145
    def set_technical_indicators(self, technical_indicators: TechnicalIndicators) -> None  # Line 149
    def set_strategy_service(self, strategy_service: 'StrategyService') -> None  # Line 154
    def set_regime_detector(self, regime_detector: MarketRegimeDetector) -> None  # Line 159
    def set_adaptive_risk_manager(self, adaptive_risk_manager: AdaptiveRiskManager) -> None  # Line 164
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]  # Line 170
    async def _validate_data_availability(self, symbol: str) -> bool  # Line 248
    async def _get_current_regime_via_service(self, symbol: str) -> 'MarketRegime | None'  # Line 269
    async def _calculate_momentum_indicators_via_service(self, symbol: str, current_data: MarketData) -> 'dict[str, Any] | None'  # Line 318
    def _calculate_rsi_score_from_value(self, rsi: float) -> float  # Line 410
    async def _generate_momentum_signals(self, ...) -> list[Signal]  # Line 419
    def _calculate_confidence(self, ...) -> float  # Line 491
    async def _apply_confidence_adjustments(self, signals: list[Signal], current_regime: 'MarketRegime | None') -> list[Signal]  # Line 524
    async def _update_strategy_state(self, ...) -> None  # Line 576
    def _get_regime_confidence_multiplier(self, regime: 'MarketRegime | None') -> float  # Line 637
    async def validate_signal(self, signal: Signal) -> bool  # Line 657
    def get_position_size(self, signal: Signal) -> Decimal  # Line 738
    def should_exit(self, position: Position, data: MarketData) -> bool  # Line 831
    def get_strategy_info(self) -> dict[str, Any]  # Line 912
    async def _on_start(self) -> None  # Line 954
    async def _on_stop(self) -> None  # Line 976
    def cleanup(self) -> None  # Line 996
```

### File: strategy_factory.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.types import StrategyType`
- `from src.data.features.technical_indicators import TechnicalIndicators`
- `from src.risk_management.adaptive_risk import AdaptiveRiskManager`
- `from src.risk_management.regime_detection import MarketRegimeDetector`

#### Class: `DynamicStrategyFactory`

**Inherits**: BaseComponent
**Purpose**: Factory for creating dynamic strategies with service layer integration

```python
class DynamicStrategyFactory(BaseComponent):
    def __init__(self, ...)  # Line 43
    async def create_strategy(self, strategy_name: str, config: dict[str, Any]) -> BaseStrategy | None  # Line 94
    def _resolve_strategy_class(self, strategy_name: str)  # Line 163
    async def _enhance_configuration(self, strategy_name: str, config: dict[str, Any]) -> dict[str, Any]  # Line 189
    async def _inject_dependencies(self, strategy: BaseStrategy, strategy_name: str) -> None  # Line 251
    async def _validate_strategy_setup(self, strategy: BaseStrategy) -> bool  # Line 301
    def get_available_strategies(self) -> dict[str, str]  # Line 354
    def get_strategy_requirements(self, strategy_name: str) -> dict[str, Any]  # Line 361
    async def create_multiple_strategies(self, strategy_configs: dict[str, dict[str, Any]]) -> dict[str, BaseStrategy | None]  # Line 399
```

### File: volatility_breakout.py

**Key Imports:**
- `from src.core.types import MarketData`
- `from src.core.types import MarketRegime`
- `from src.core.types import Position`
- `from src.core.types import Signal`
- `from src.core.types import SignalDirection`

#### Class: `VolatilityBreakoutStrategy`

**Inherits**: BaseStrategy
**Purpose**: Enhanced Volatility Breakout Strategy with service layer integration

```python
class VolatilityBreakoutStrategy(BaseStrategy):
    def name(self) -> str  # Line 60
    def name(self, value: str) -> None  # Line 69
    def version(self) -> str  # Line 74
    def version(self, value: str) -> None  # Line 79
    def status(self) -> str  # Line 84
    def status(self, value: str) -> None  # Line 89
    def __init__(self, config: dict[str, Any], services: 'StrategyServiceContainer | None' = None)  # Line 93
    def strategy_type(self) -> StrategyType  # Line 149
    def set_technical_indicators(self, technical_indicators: TechnicalIndicators) -> None  # Line 153
    def set_strategy_service(self, strategy_service: 'StrategyService') -> None  # Line 158
    def set_regime_detector(self, regime_detector: MarketRegimeDetector) -> None  # Line 163
    def set_adaptive_risk_manager(self, adaptive_risk_manager: AdaptiveRiskManager) -> None  # Line 168
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]  # Line 174
    async def _validate_data_availability(self, symbol: str) -> bool  # Line 270
    async def _is_in_breakout_cooldown(self, symbol: str) -> bool  # Line 290
    async def _get_current_regime_via_service(self, symbol: str) -> 'MarketRegime | None'  # Line 312
    async def _calculate_volatility_indicators_via_service(self, symbol: str, current_data: MarketData) -> 'dict[str, Any] | None'  # Line 358
    async def _calculate_consolidation_score(self, symbol: str, price_data: list[MarketData]) -> float  # Line 445
    def _is_bollinger_squeeze(self, bb_data: dict, current_price: Decimal) -> bool  # Line 510
    def _get_bb_position(self, bb_data: dict, current_price: Decimal) -> str  # Line 529
    async def _calculate_breakout_levels(self, ...) -> dict[str, float]  # Line 553
    def _get_regime_breakout_adjustment(self, regime: 'MarketRegime | None') -> float  # Line 614
    async def _generate_breakout_signals(self, ...) -> list[Signal]  # Line 633
    def _calculate_breakout_confidence(self, ...) -> float  # Line 729
    def _get_regime_confidence_multiplier(self, regime: 'MarketRegime | None') -> float  # Line 781
    async def _apply_regime_filtering(self, signals: list[Signal], current_regime: 'MarketRegime | None') -> list[Signal]  # Line 800
    def _is_signal_valid_for_regime(self, signal: Signal, regime: MarketRegime) -> bool  # Line 860
    async def _apply_time_decay(self, signals: list[Signal], symbol: str) -> list[Signal]  # Line 908
    async def _update_strategy_state(self, ...) -> None  # Line 986
    async def validate_signal(self, signal: Signal) -> bool  # Line 1041
    def get_position_size(self, signal: Signal) -> Decimal  # Line 1156
    def _get_regime_position_adjustment(self, regime: MarketRegime) -> float  # Line 1253
    def should_exit(self, position: Position, data: MarketData) -> bool  # Line 1269
    def get_strategy_info(self) -> dict[str, Any]  # Line 1348
    async def _on_start(self) -> None  # Line 1394
    async def _on_stop(self) -> None  # Line 1416
    def cleanup(self) -> None  # Line 1436
```

### File: environment_integration.py

**Key Imports:**
- `from src.core.exceptions import StrategyError`
- `from src.core.integration.environment_aware_service import EnvironmentAwareServiceMixin`
- `from src.core.integration.environment_aware_service import EnvironmentContext`
- `from src.core.logging import get_logger`
- `from src.core.types import MarketData`

#### Class: `StrategyMode`

**Inherits**: Enum
**Purpose**: Strategy operation modes for different environments

```python
class StrategyMode(Enum):
```

#### Class: `EnvironmentAwareStrategyConfiguration`

**Purpose**: Environment-specific strategy configuration

```python
class EnvironmentAwareStrategyConfiguration:
    def get_sandbox_strategy_config() -> dict[str, Any]  # Line 42
    def get_live_strategy_config() -> dict[str, Any]  # Line 66
```

#### Class: `EnvironmentAwareStrategyManager`

**Inherits**: EnvironmentAwareServiceMixin
**Purpose**: Environment-aware strategy management functionality

```python
class EnvironmentAwareStrategyManager(EnvironmentAwareServiceMixin):
    def __init__(self, *args: Any, **kwargs: Any) -> None  # Line 98
    async def _update_service_environment(self, context: EnvironmentContext) -> None  # Line 105
    def get_environment_strategy_config(self, exchange: str) -> dict[str, Any]  # Line 140
    async def deploy_environment_aware_strategy(self, strategy_config: StrategyConfig, exchange: str, force_deploy: bool = False) -> bool  # Line 153
    async def validate_strategy_for_environment(self, strategy_config: StrategyConfig, exchange: str) -> bool  # Line 220
    async def _validate_production_strategy(self, strategy_config: StrategyConfig, env_config: dict[str, Any]) -> bool  # Line 240
    async def _validate_sandbox_strategy(self, strategy_config: StrategyConfig, env_config: dict[str, Any]) -> bool  # Line 278
    async def _validate_common_strategy_rules(self, strategy_config: StrategyConfig, exchange: str, env_config: dict[str, Any]) -> bool  # Line 299
    async def _apply_environment_strategy_adjustments(self, strategy_config: StrategyConfig, exchange: str) -> StrategyConfig  # Line 326
    async def generate_environment_aware_signal(self, strategy_name: str, market_data: MarketData, exchange: str) -> 'Signal | None'  # Line 362
    async def _apply_environment_signal_filters(self, signal: Signal, exchange: str, env_config: dict[str, Any]) -> 'Signal | None'  # Line 394
    async def _validate_signal_for_environment(self, signal: Signal, exchange: str) -> bool  # Line 425
    async def _deploy_strategy_with_config(self, strategy_config: StrategyConfig, exchange: str) -> bool  # Line 449
    async def _generate_base_signal(self, strategy_name: str, market_data: MarketData, exchange: str) -> 'Signal | None'  # Line 457
    async def _verify_strategy_backtest(self, strategy_config: StrategyConfig, exchange: str) -> bool  # Line 479
    async def _initialize_strategy_tracking(self, strategy_name: str, exchange: str) -> None  # Line 509
    async def _update_signal_tracking(self, strategy_name: str, exchange: str, signal: Signal) -> None  # Line 518
    async def _disable_experimental_strategies(self, exchange: str) -> None  # Line 529
    async def _is_high_volatility_period(self, symbol: str, exchange: str) -> bool  # Line 540
    async def update_strategy_performance(self, ...) -> None  # Line 545
    def get_environment_strategy_metrics(self, exchange: str) -> dict[str, Any]  # Line 577
    async def rebalance_strategies_for_environment(self, exchange: str) -> dict[str, Any]  # Line 607
```

### File: fitness.py

**Key Imports:**
- `from src.strategies.interfaces import OptimizationResult`

#### Class: `FitnessFunction`

**Inherits**: ABC
**Purpose**: Abstract base class for fitness functions

```python
class FitnessFunction(ABC):
    def calculate(self, result: OptimizationResult) -> float  # Line 18
```

#### Class: `SharpeFitness`

**Inherits**: FitnessFunction
**Purpose**: Fitness based on Sharpe ratio

```python
class SharpeFitness(FitnessFunction):
    def calculate(self, result: OptimizationResult) -> float  # Line 34
```

#### Class: `ReturnFitness`

**Inherits**: FitnessFunction
**Purpose**: Fitness based on total return

```python
class ReturnFitness(FitnessFunction):
    def calculate(self, result: OptimizationResult) -> float  # Line 42
```

#### Class: `CompositeFitness`

**Inherits**: FitnessFunction
**Purpose**: Composite fitness combining multiple metrics

```python
class CompositeFitness(FitnessFunction):
    def __init__(self, ...)  # Line 50
    def calculate(self, result: OptimizationResult) -> float  # Line 73
```

#### Class: `FitnessEvaluator`

**Purpose**: Evaluates fitness of trading strategies

```python
class FitnessEvaluator:
    def __init__(self, ...)  # Line 121
    def evaluate(self, result: OptimizationResult) -> float  # Line 143
    def evaluate_multi_objective(self, result: OptimizationResult) -> dict[str, float]  # Line 165
    def _apply_constraints(self, result: OptimizationResult, fitness: float) -> float  # Line 198
    def compare(self, result1: OptimizationResult, result2: OptimizationResult) -> int  # Line 218
    def rank(self, results: list[OptimizationResult]) -> list[int]  # Line 239
```

#### Class: `AdaptiveFitness`

**Inherits**: FitnessFunction
**Purpose**: Adaptive fitness that changes based on market conditions

```python
class AdaptiveFitness(FitnessFunction):
    def __init__(self)  # Line 256
    def set_market_regime(self, regime: str) -> None  # Line 277
    def calculate(self, result: OptimizationResult) -> float  # Line 284
```

### File: genetic.py

**Key Imports:**
- `from src.core.exceptions import OptimizationError`
- `from src.strategies.evolutionary.fitness import FitnessEvaluator`
- `from src.strategies.evolutionary.mutations import CrossoverOperator`
- `from src.strategies.evolutionary.mutations import MutationOperator`
- `from src.strategies.evolutionary.population import Individual`

#### Class: `GeneticConfig`

**Inherits**: BaseModel
**Purpose**: Configuration for genetic algorithm

```python
class GeneticConfig(BaseModel):
```

#### Class: `GeneticAlgorithm`

**Purpose**: Genetic Algorithm for evolving trading strategies

```python
class GeneticAlgorithm:
    def __init__(self, ...)  # Line 64
    async def evolve(self) -> Individual  # Line 109
    def _initialize_population(self) -> Population  # Line 186
    async def _evaluate_population(self, population: Population) -> None  # Line 219
    async def _evaluate_individual(self, individual: Individual) -> None  # Line 245
    def _selection(self, population: Population) -> list[Individual]  # Line 277
    def _tournament_selection(self, population: Population) -> Individual  # Line 293
    def _create_offspring(self, parents: list[Individual]) -> Population  # Line 302
    def _replacement(self, population: Population, offspring: Population) -> Population  # Line 342
    def _calculate_diversity(self) -> float  # Line 357
    def _gene_distance(self, genes1: dict[str, Any], genes2: dict[str, Any]) -> float  # Line 374
    def _inject_diversity(self) -> None  # Line 405
    def _record_generation_stats(self) -> None  # Line 434
    def get_evolution_summary(self) -> dict[str, Any]  # Line 452
```

### File: mutations.py

#### Class: `MutationOperator`

**Purpose**: Mutation operators for genetic algorithms

```python
class MutationOperator:
    def __init__(self, ...)  # Line 21
    def mutate(self, genes: dict[str, Any], parameter_ranges: dict[str, tuple[Any, Any]]) -> dict[str, Any]  # Line 40
    def _mutate_parameter(self, value: Any, param_range: tuple[Any, Any]) -> Any  # Line 70
    def _adaptive_rate(self) -> float  # Line 103
    def increment_generation(self) -> None  # Line 112
```

#### Class: `CrossoverOperator`

**Purpose**: Crossover operators for genetic algorithms

```python
class CrossoverOperator:
    def __init__(self, crossover_rate: float = 0.7, crossover_type: str = 'uniform')  # Line 124
    def crossover(self, parent1_genes: dict[str, Any], parent2_genes: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]  # Line 139
    def _uniform_crossover(self, parent1_genes: dict[str, Any], parent2_genes: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]  # Line 170
    def _single_point_crossover(self, parent1_genes: dict[str, Any], parent2_genes: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]  # Line 193
    def _two_point_crossover(self, parent1_genes: dict[str, Any], parent2_genes: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]  # Line 224
    def _blend_crossover(self, ...) -> tuple[dict[str, Any], dict[str, Any]]  # Line 256
```

#### Class: `AdvancedMutationOperator`

**Inherits**: MutationOperator
**Purpose**: Advanced mutation operator with multiple strategies

```python
class AdvancedMutationOperator(MutationOperator):
    def __init__(self, ...)  # Line 307
    def mutate(self, genes: dict[str, Any], parameter_ranges: dict[str, tuple[Any, Any]]) -> dict[str, Any]  # Line 324
    def _gaussian_mutation(self, value: Any, param_range: tuple[Any, Any]) -> Any  # Line 348
    def _uniform_mutation(self, value: Any, param_range: tuple[Any, Any]) -> Any  # Line 367
    def _polynomial_mutation(self, value: Any, param_range: tuple[Any, Any], eta: float = 20) -> Any  # Line 379
```

### File: neuroevolution.py

**Key Imports:**
- `from src.core.exceptions import OptimizationError`
- `from src.core.types import MarketData`
- `from src.core.types import Position`
- `from src.core.types import Signal`
- `from src.core.types import SignalDirection`

#### Class: `ActivationType`

**Inherits**: Enum
**Purpose**: Supported activation function types for neural networks

```python
class ActivationType(Enum):
```

#### Class: `NodeType`

**Inherits**: Enum
**Purpose**: Neural network node types for NEAT genome representation

```python
class NodeType(Enum):
```

#### Class: `ConnectionType`

**Inherits**: Enum
**Purpose**: Connection types for network topology

```python
class ConnectionType(Enum):
```

#### Class: `NodeGene`

**Purpose**: Represents a node in the NEAT genome

```python
class NodeGene:
    def copy(self) -> 'NodeGene'  # Line 119
```

#### Class: `ConnectionGene`

**Purpose**: Represents a connection in the NEAT genome

```python
class ConnectionGene:
    def copy(self) -> 'ConnectionGene'  # Line 151
```

#### Class: `InnovationTracker`

**Purpose**: Tracks innovation numbers for topology mutations in NEAT algorithm

```python
class InnovationTracker:
    def __init__(self)  # Line 170
    def get_connection_innovation(self, from_node: int, to_node: int) -> int  # Line 177
    def get_node_innovation(self, connection_id: int) -> int  # Line 193
    def get_next_node_id(self) -> int  # Line 207
```

#### Class: `NEATGenome`

**Purpose**: NEAT genome representation for evolving neural network topologies

```python
class NEATGenome:
    def __init__(self, ...)  # Line 221
    def _initialize_minimal_topology(self) -> None  # Line 256
    def add_node_mutation(self) -> bool  # Line 292
    def add_connection_mutation(self, allow_recurrent: bool = True) -> bool  # Line 359
    def mutate_weights(self, mutation_rate: float = 0.8, mutation_strength: float = 0.1) -> None  # Line 430
    def mutate_activation_functions(self, mutation_rate: float = 0.1) -> None  # Line 456
    def enable_disable_mutation(self, enable_rate: float = 0.01, disable_rate: float = 0.01) -> None  # Line 472
    def _shift_layers_from(self, start_layer: int, shift_amount: int) -> None  # Line 487
    def calculate_compatibility_distance(self, other: 'NEATGenome', c1: float = 1.0, c2: float = 1.0, c3: float = 0.4) -> float  # Line 498
    def crossover(self, other: 'NEATGenome', fitness_equal: bool = False) -> 'NEATGenome'  # Line 550
    def copy(self) -> 'NEATGenome'  # Line 613
```

#### Class: `NeuroNetwork`

**Purpose**: PyTorch implementation of evolvable neural network from NEAT genome

```python
class NeuroNetwork:
    def __init__(self, genome: NEATGenome, device: str = 'cpu')  # Line 649
    def _build_network(self) -> None  # Line 669
    def _reset_hidden_states(self) -> None  # Line 706
    def forward(self, x: 'Tensor', reset_hidden: bool = False) -> 'Tensor'  # Line 712
    def predict_signal(self, market_data: MarketData) -> Signal  # Line 797
    def _extract_features(self, market_data: MarketData) -> list[float]  # Line 843
```

#### Class: `Species`

**Purpose**: Represents a species in NEAT population for diversity preservation

```python
class Species:
    def __init__(self, species_id: int, representative: NEATGenome)  # Line 902
    def add_member(self, genome: NEATGenome) -> None  # Line 917
    def calculate_average_fitness(self) -> None  # Line 922
    def select_parents(self, selection_pressure: float = 0.5) -> list[NEATGenome]  # Line 929
    def remove_worst_genomes(self, keep_ratio: float = 0.5) -> None  # Line 948
```

#### Class: `SpeciationManager`

**Purpose**: Manages species formation and evolution in NEAT population

```python
class SpeciationManager:
    def __init__(self, ...)  # Line 968
    def speciate_population(self, population: list[NEATGenome]) -> None  # Line 987
    def _calculate_fitness_sharing(self) -> None  # Line 1032
    def _remove_stagnant_species(self) -> None  # Line 1039
    def _adjust_compatibility_threshold(self) -> None  # Line 1057
    def allocate_offspring(self, total_offspring: int) -> dict[int, int]  # Line 1071
```

#### Class: `NeuroEvolutionConfig`

**Purpose**: Configuration for neuroevolution strategy

```python
class NeuroEvolutionConfig:
```

#### Class: `NeuroEvolutionStrategy`

**Inherits**: BaseStrategy
**Purpose**: Neural network evolution strategy for trading decisions

```python
class NeuroEvolutionStrategy(BaseStrategy):
    def __init__(self, config: dict[str, Any], services: StrategyServiceContainer | None = None)  # Line 1165
    def _initialize_population(self) -> None  # Line 1205
    def _update_best_network(self) -> None  # Line 1232
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]  # Line 1240
    async def _maybe_adapt(self) -> None  # Line 1282
    async def adapt_networks(self) -> None  # Line 1293
    async def _micro_evolution(self) -> None  # Line 1341
    async def evolve_population(self, fitness_evaluator: FitnessEvaluator) -> None  # Line 1374
    def _create_offspring(self, species: Species, parents: list[NEATGenome]) -> NEATGenome  # Line 1450
    def _mutate_genome(self, genome: NEATGenome) -> None  # Line 1478
    def _record_generation_stats(self) -> None  # Line 1505
    async def validate_signal(self, signal: Signal) -> bool  # Line 1526
    def get_position_size(self, signal: Signal) -> Decimal  # Line 1550
    def should_exit(self, position: Position, data: MarketData) -> bool  # Line 1574
    def _check_standard_exits(self, position: Position, data: MarketData) -> bool  # Line 1604
    def get_strategy_info(self) -> dict[str, Any]  # Line 1628
    def get_evolution_summary(self) -> dict[str, Any]  # Line 1658
    async def save_population(self, filepath: str) -> None  # Line 1687
    async def load_population(self, filepath: str) -> None  # Line 1718
```

### File: optimization.py

**Key Imports:**
- `from src.core.exceptions import OptimizationError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.strategies.evolutionary.fitness import FitnessEvaluator`
- `from src.strategies.evolutionary.population import Individual`

#### Class: `OptimizationObjective`

**Inherits**: BaseModel
**Purpose**: Optimization objective definition

```python
class OptimizationObjective(BaseModel):
    def validate_direction(self) -> None  # Line 56
```

#### Class: `MultiObjectiveConfig`

**Inherits**: BaseModel
**Purpose**: Configuration for multi-objective optimization

```python
class MultiObjectiveConfig(BaseModel):
```

#### Class: `ParetoSolution`

**Purpose**: Represents a solution in the Pareto frontier

```python
class ParetoSolution:
```

#### Class: `ConstraintHandler`

**Purpose**: Handles constraints in multi-objective optimization

```python
class ConstraintHandler:
    def __init__(self, constraints: list[OptimizationObjective])  # Line 118
    def evaluate_constraints(self, objectives: dict[str, float]) -> dict[str, float]  # Line 135
    def is_feasible(self, objectives: dict[str, float], tolerance: float = 0.01) -> bool  # Line 168
    def apply_penalty(self, objectives: dict[str, float], constraint_violations: dict[str, float]) -> dict[str, float]  # Line 183
```

#### Class: `DominanceComparator`

**Purpose**: Implements dominance comparison for multi-objective optimization

```python
class DominanceComparator:
    def __init__(self, objectives: list[OptimizationObjective])  # Line 228
    def dominates(self, solution1: dict[str, float], solution2: dict[str, float]) -> bool  # Line 240
    def non_dominated_sort(self, solutions: list[dict[str, float]]) -> list[list[int]]  # Line 280
```

#### Class: `CrowdingDistanceCalculator`

**Purpose**: Calculates crowding distance for diversity preservation

```python
class CrowdingDistanceCalculator:
    def __init__(self, objectives: list[OptimizationObjective])  # Line 341
    def calculate_crowding_distance(self, solutions: list[dict[str, float]], front_indices: list[int]) -> list[float]  # Line 353
```

#### Class: `ParetoFrontierManager`

**Purpose**: Manages the Pareto frontier and provides analysis tools

```python
class ParetoFrontierManager:
    def __init__(self, config: MultiObjectiveConfig)  # Line 412
    def update_frontier(self, solutions: list[ParetoSolution]) -> None  # Line 435
    def _calculate_convergence_metrics(self) -> None  # Line 484
    def _calculate_hypervolume(self, solutions: list[ParetoSolution]) -> float  # Line 502
    def _calculate_spread(self, solutions: list[ParetoSolution]) -> float  # Line 539
    def _calculate_convergence(self, current: list[ParetoSolution], previous: list[ParetoSolution]) -> float  # Line 562
    def _solution_distance(self, sol1: ParetoSolution, sol2: ParetoSolution) -> float  # Line 595
    def get_frontier_summary(self) -> dict[str, Any]  # Line 617
```

#### Class: `NSGAIIOptimizer`

**Purpose**: NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation

```python
class NSGAIIOptimizer:
    def __init__(self, ...)  # Line 672
    async def optimize(self) -> list[ParetoSolution]  # Line 712
    async def _initialize_population(self) -> Population  # Line 781
    async def _evaluate_population(self, population: Population) -> list[ParetoSolution]  # Line 808
    async def _evaluate_individual(self, individual: Individual) -> ParetoSolution | None  # Line 849
    async def _simulate_objectives(self, individual: Individual) -> dict[str, float]  # Line 889
    async def _create_offspring(self) -> Population  # Line 930
    def _tournament_selection(self) -> ParetoSolution  # Line 966
    def _create_random_solution(self) -> ParetoSolution  # Line 984
    def _crossover(self, genes1: dict[str, Any], genes2: dict[str, Any]) -> dict[str, Any]  # Line 1008
    def _mutate(self, genes: dict[str, Any]) -> dict[str, Any]  # Line 1033
    def _environmental_selection(self, solutions: list[ParetoSolution]) -> list[ParetoSolution]  # Line 1065
    def _check_convergence(self) -> bool  # Line 1118
    def _record_generation_stats(self) -> None  # Line 1142
    def get_optimization_summary(self) -> dict[str, Any]  # Line 1157
```

#### Class: `MultiObjectiveOptimizer`

**Purpose**: Main interface for multi-objective optimization of trading strategies

```python
class MultiObjectiveOptimizer:
    def __init__(self, config: MultiObjectiveConfig)  # Line 1194
    async def optimize_strategy(self, ...) -> list[ParetoSolution]  # Line 1218
    def get_pareto_frontier_data(self) -> dict[str, Any]  # Line 1270
    def export_results(self, filepath: str) -> None  # Line 1309
```

#### Functions:

```python
def create_trading_objectives() -> list[OptimizationObjective]  # Line 1338
def create_default_config(objectives: list[OptimizationObjective] | None = None) -> MultiObjectiveConfig  # Line 1377
```

### File: population.py

#### Class: `Individual`

**Purpose**: Represents an individual in the population

```python
class Individual:
    def __lt__(self, other)  # Line 24
    def __repr__(self)  # Line 28
    def copy(self) -> 'Individual'  # Line 32
```

#### Class: `Population`

**Purpose**: Manages a population of individuals

```python
class Population:
    def __init__(self, individuals: list[Individual])  # Line 45
    def __len__(self) -> int  # Line 55
    def __iter__(self)  # Line 59
    def _ensure_sorted(self) -> None  # Line 63
    def get_best(self) -> Individual | None  # Line 69
    def get_worst(self) -> Individual | None  # Line 76
    def get_top_n(self, n: int) -> list[Individual]  # Line 83
    def get_bottom_n(self, n: int) -> list[Individual]  # Line 88
    def get_bottom_n_indices(self, n: int) -> list[int]  # Line 93
    def get_statistics(self) -> dict[str, float]  # Line 98
    def add(self, individual: Individual) -> None  # Line 119
    def remove(self, individual: Individual) -> None  # Line 124
    def replace(self, index: int, individual: Individual) -> None  # Line 130
```

### File: factory.py

**Key Imports:**
- `from src.core.exceptions import StrategyError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.core.types import StrategyConfig`
- `from src.core.types import StrategyType`

#### Class: `StrategyFactory`

**Inherits**: StrategyFactoryInterface
**Purpose**: Factory for creating strategies with proper dependency injection

```python
class StrategyFactory(StrategyFactoryInterface):
    def __init__(self, ...)  # Line 63
    def _register_builtin_strategies(self) -> None  # Line 142
    def _lazy_load_strategy_class(self, strategy_type: StrategyType) -> type | None  # Line 148
    def register_strategy_type(self, strategy_type: StrategyType, strategy_class: type) -> None  # Line 227
    async def create_strategy(self, strategy_type: StrategyType, config: StrategyConfig) -> BaseStrategyInterface  # Line 252
    async def _create_comprehensive_service_container(self, config: StrategyConfig) -> StrategyServiceContainer  # Line 316
    async def _enhance_strategy_with_integrations(self, strategy, config: StrategyConfig) -> None  # Line 399
    def _validate_configuration_parameters(self, config: StrategyConfig) -> bool  # Line 423
    def _get_integration_status(self, strategy) -> dict[str, bool]  # Line 471
    async def _inject_dependencies(self, strategy: BaseStrategyInterface, config: StrategyConfig) -> None  # Line 482
    def get_supported_strategies(self) -> list[StrategyType]  # Line 538
    def validate_strategy_requirements(self, strategy_type: StrategyType, config: StrategyConfig) -> bool  # Line 562
    def _validate_strategy_specific_requirements(self, strategy_type: StrategyType, config: StrategyConfig) -> bool  # Line 595
    def _get_required_parameters(self, strategy_type: StrategyType) -> list[str]  # Line 628
    def _validate_momentum_strategy_config(self, config: StrategyConfig) -> bool  # Line 655
    def _validate_mean_reversion_strategy_config(self, config: StrategyConfig) -> bool  # Line 679
    def _validate_arbitrage_strategy_config(self, config: StrategyConfig) -> bool  # Line 703
    def _validate_volatility_strategy_config(self, config: StrategyConfig) -> bool  # Line 719
    async def create_strategy_with_validation(self, ...) -> BaseStrategyInterface  # Line 745
    def _validate_dependency_availability_sync(self, config: StrategyConfig) -> bool  # Line 787
    async def _validate_created_strategy(self, strategy: BaseStrategyInterface) -> bool  # Line 816
    def get_strategy_info(self, strategy_type: StrategyType) -> dict[str, Any]  # Line 863
    def list_available_strategies(self) -> dict[str, Any]  # Line 888
```

### File: ensemble.py

**Key Imports:**
- `from src.core.exceptions import StrategyError`
- `from src.core.types import MarketData`
- `from src.core.types import Position`
- `from src.core.types import Signal`
- `from src.core.types import SignalDirection`

#### Class: `StrategyPerformanceTracker`

**Purpose**: Tracks individual strategy performance within the ensemble

```python
class StrategyPerformanceTracker:
    def __init__(self, strategy_name: str)  # Line 44
    def add_signal(self, signal: Signal) -> None  # Line 61
    def add_trade_result(self, return_pct: float, trade_info: dict[str, Any]) -> None  # Line 72
    def _update_metrics(self) -> None  # Line 92
    def get_recent_performance(self, window: int = 10) -> float  # Line 121
    def get_performance_score(self) -> float  # Line 128
    def get_metrics(self) -> dict[str, Any]  # Line 146
```

#### Class: `CorrelationAnalyzer`

**Purpose**: Analyzes correlations between strategies for diversity maintenance

```python
class CorrelationAnalyzer:
    def __init__(self, window_size: int = 50)  # Line 166
    def add_strategy_return(self, strategy_name: str, return_value: float) -> None  # Line 171
    def calculate_correlation_matrix(self) -> dict[str, dict[str, float]]  # Line 184
    def get_diversity_score(self) -> float  # Line 225
    def identify_redundant_strategies(self, threshold: float = 0.8) -> list[tuple[str, str]]  # Line 249
```

#### Class: `VotingMechanism`

**Purpose**: Implements different voting mechanisms for ensemble decisions

```python
class VotingMechanism:
    def majority_vote(signals: list[tuple[str, Signal]]) -> Signal | None  # Line 268
    def weighted_vote(signals: list[tuple[str, Signal]], weights: dict[str, float]) -> Signal | None  # Line 307
    def confidence_weighted_vote(signals: list[tuple[str, Signal]], performance_scores: dict[str, float]) -> Signal | None  # Line 365
```

#### Class: `EnsembleStrategy`

**Inherits**: BaseStrategy
**Purpose**: Adaptive ensemble strategy that combines multiple trading strategies

```python
class EnsembleStrategy(BaseStrategy):
    def __init__(self, config: dict[str, Any], services: StrategyServiceContainer | None = None)  # Line 436
    def add_strategy(self, strategy: BaseStrategy, initial_weight: float = 1.0) -> None  # Line 476
    def remove_strategy(self, strategy_name: str) -> None  # Line 506
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]  # Line 528
    async def _vote_on_signals(self, component_signals: list[tuple[str, Signal]], regime: Any) -> Signal | None  # Line 596
    async def _filter_signals_by_regime(self, signals: list[tuple[str, Signal]], regime: Any) -> list[tuple[str, Signal]]  # Line 639
    async def _check_and_rebalance(self) -> None  # Line 647
    async def _rebalance_strategies(self) -> None  # Line 663
    async def validate_signal(self, signal: Signal) -> bool  # Line 732
    def get_position_size(self, signal: Signal) -> Decimal  # Line 762
    def should_exit(self, position: Position, data: MarketData) -> bool  # Line 793
    async def _on_start(self) -> None  # Line 859
    async def _on_stop(self) -> None  # Line 872
    def update_strategy_performance(self, strategy_name: str, return_pct: float, trade_info: dict[str, Any]) -> None  # Line 885
    def get_ensemble_statistics(self) -> dict[str, Any]  # Line 905
    def get_strategy_info(self) -> dict[str, Any]  # Line 918
```

### File: fallback.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.core.types import MarketData`
- `from src.core.types import Position`
- `from src.core.types import Signal`
- `from src.core.types import SignalDirection`

#### Class: `FallbackMode`

**Inherits**: Enum
**Purpose**: Fallback operation modes

```python
class FallbackMode(Enum):
```

#### Class: `FailureType`

**Inherits**: Enum
**Purpose**: Types of failures that can trigger fallback

```python
class FailureType(Enum):
```

#### Class: `FailureDetector`

**Purpose**: Detects various types of strategy failures

```python
class FailureDetector:
    def __init__(self, config: dict[str, Any], services: StrategyServiceContainer | None = None)  # Line 67
    def add_trade_result(self, return_pct: float, timestamp: datetime) -> None  # Line 93
    def add_error(self, error_type: str, timestamp: datetime) -> None  # Line 103
    def add_timeout(self, duration: float, timestamp: datetime) -> None  # Line 113
    def add_signal(self, signal: Signal) -> None  # Line 123
    def detect_performance_degradation(self) -> dict[str, Any]  # Line 131
    def detect_technical_issues(self) -> dict[str, Any]  # Line 187
    def detect_confidence_issues(self) -> dict[str, Any]  # Line 231
    def detect_market_conditions(self, market_data: list[MarketData]) -> dict[str, Any]  # Line 250
```

#### Class: `SafeModeStrategy`

**Purpose**: Emergency safe mode strategy with minimal risk

```python
class SafeModeStrategy:
    def __init__(self, config: dict[str, Any], services: StrategyServiceContainer | None = None)  # Line 275
    async def generate_signal(self, data: MarketData, price_history: list[float]) -> Signal | None  # Line 283
```

#### Class: `DegradedModeStrategy`

**Purpose**: Degraded mode strategy with simplified logic

```python
class DegradedModeStrategy:
    def __init__(self, config: dict[str, Any], services: StrategyServiceContainer | None = None)  # Line 343
    async def generate_signal(self, data: MarketData, price_history: list[float]) -> Signal | None  # Line 350
```

#### Class: `FallbackStrategy`

**Inherits**: BaseStrategy
**Purpose**: Intelligent fallback strategy with automatic failure detection and recovery

```python
class FallbackStrategy(BaseStrategy):
    def __init__(self, config: dict[str, Any], services: StrategyServiceContainer | None = None)  # Line 435
    def set_primary_strategy(self, strategy: BaseStrategy) -> None  # Line 477
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]  # Line 483
    async def _check_and_update_mode(self, data: MarketData) -> None  # Line 532
    async def _handle_failure(self, failure_info: dict[str, Any]) -> None  # Line 583
    def _determine_fallback_mode(self, failure_info: dict[str, Any]) -> FallbackMode  # Line 606
    async def _switch_mode(self, new_mode: FallbackMode, failure_info: dict[str, Any]) -> None  # Line 642
    async def _check_recovery_conditions(self) -> None  # Line 683
    async def _evaluate_recovery_performance(self) -> bool  # Line 729
    async def _generate_signal_for_current_mode(self, data: MarketData) -> Signal | None  # Line 762
    async def validate_signal(self, signal: Signal) -> bool  # Line 819
    def get_position_size(self, signal: Signal) -> Decimal  # Line 859
    def should_exit(self, position: Position, data: MarketData) -> bool  # Line 900
    def update_trade_result(self, return_pct: float, trade_info: dict[str, Any]) -> None  # Line 952
    def get_fallback_statistics(self) -> dict[str, Any]  # Line 968
    def get_strategy_info(self) -> dict[str, Any]  # Line 981
```

### File: rule_based_ai.py

**Key Imports:**
- `from src.core.exceptions import StrategyError`
- `from src.core.logging import get_logger`
- `from src.core.types import MarketData`
- `from src.core.types import Position`
- `from src.core.types import PositionSide`

#### Class: `TechnicalRuleEngine`

**Purpose**: Traditional technical analysis rule engine

```python
class TechnicalRuleEngine:
    def __init__(self, config: dict[str, Any], strategy_instance: Optional['BaseStrategy'] = None)  # Line 48
    async def calculate_rsi(self, symbol: str) -> float  # Line 76
    async def calculate_moving_averages(self, symbol: str, current_price: Decimal) -> tuple[Decimal, Decimal]  # Line 96
    async def evaluate_rules(self, ...) -> dict[str, Any]  # Line 117
    def update_rule_performance(self, rule: str, performance: float) -> None  # Line 203
    def adjust_rule_weights(self) -> None  # Line 212
```

#### Class: `AIPredictor`

**Purpose**: AI prediction component using machine learning

```python
class AIPredictor:
    def __init__(self, config: dict[str, Any], services: StrategyServiceContainer | None = None)  # Line 225
    def prepare_features(self, price_history: list[float], volume_history: list[float]) -> np.ndarray  # Line 244
    async def train_model(self, training_data: list[dict[str, Any]]) -> None  # Line 306
    async def predict(self, price_history: list[float], volume_history: list[float]) -> dict[str, Any]  # Line 354
    def update_performance(self, prediction: dict[str, Any], actual_outcome: float) -> None  # Line 408
    def get_performance_metrics(self) -> dict[str, float]  # Line 418
```

#### Class: `RuleBasedAIStrategy`

**Inherits**: BaseStrategy
**Purpose**: Hybrid strategy combining traditional technical analysis rules with AI predictions

```python
class RuleBasedAIStrategy(BaseStrategy):
    def __init__(self, config: dict[str, Any], services: StrategyServiceContainer | None = None)  # Line 448
    def strategy_type(self) -> StrategyType  # Line 494
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]  # Line 499
    async def _resolve_conflicts(self, ...) -> Signal | None  # Line 573
    def _weighted_average_resolution(self, ...) -> dict[str, Any] | None  # Line 659
    def _highest_confidence_resolution(self, ...) -> dict[str, Any] | None  # Line 693
    def _consensus_resolution(self, ...) -> dict[str, Any] | None  # Line 706
    async def validate_signal(self, signal: Signal) -> bool  # Line 722
    def get_position_size(self, signal: Signal) -> Decimal  # Line 751
    async def should_exit(self, position: Position, data: MarketData) -> bool  # Line 783
    async def _on_start(self) -> None  # Line 838
    async def _retrain_ai_model(self) -> None  # Line 846
    def adjust_component_weights(self) -> None  # Line 883
    def get_strategy_statistics(self) -> dict[str, Any]  # Line 931
    def get_strategy_stats(self) -> dict[str, Any]  # Line 946
    async def _get_ml_service_prediction(self, symbol: str, data: MarketData) -> dict[str, Any] | None  # Line 961
    def _combine_predictions(self, local_prediction: dict[str, Any], ml_prediction: dict[str, Any]) -> dict[str, Any]  # Line 1013
```

### File: interfaces.py

**Key Imports:**
- `from src.core.types import MarketData`
- `from src.core.types import MarketRegime`
- `from src.core.types import Position`
- `from src.core.types import Signal`
- `from src.core.types import StrategyConfig`

#### Class: `BacktestingInterface`

**Inherits**: Protocol
**Purpose**: Protocol for backtesting integration with strategies

```python
class BacktestingInterface(Protocol):
    async def prepare_for_backtest(self, config: dict[str, Any]) -> None  # Line 34
    async def process_historical_data(self, data: MarketData) -> list[Signal]  # Line 43
    async def simulate_trade_execution(self, signal: Signal, market_data: MarketData) -> dict[str, Any]  # Line 55
    async def get_backtest_metrics(self) -> dict[str, Any]  # Line 70
    async def reset_backtest_state(self) -> None  # Line 79
```

#### Class: `PerformanceMonitoringInterface`

**Inherits**: Protocol
**Purpose**: Protocol for strategy performance monitoring

```python
class PerformanceMonitoringInterface(Protocol):
    def update_performance_metrics(self, trade_result: dict[str, Any]) -> None  # Line 87
    def get_real_time_metrics(self) -> dict[str, Any]  # Line 96
    def calculate_risk_adjusted_returns(self) -> dict[str, Decimal]  # Line 105
    def get_drawdown_analysis(self) -> dict[str, Any]  # Line 114
```

#### Class: `RiskManagementInterface`

**Inherits**: Protocol
**Purpose**: Protocol for strategy risk management integration

```python
class RiskManagementInterface(Protocol):
    async def validate_risk_limits(self, signal: Signal) -> bool  # Line 127
    def calculate_position_size(self, signal: Signal, account_balance: Decimal) -> Decimal  # Line 139
    def should_close_position(self, position: Position, current_data: MarketData) -> bool  # Line 152
```

#### Class: `BaseStrategyInterface`

**Inherits**: ABC
**Purpose**: Base interface that all strategies must implement

```python
class BaseStrategyInterface(ABC):
    def strategy_type(self) -> StrategyType  # Line 176
    def name(self) -> str  # Line 182
    def version(self) -> str  # Line 188
    def status(self) -> StrategyStatus  # Line 194
    async def initialize(self, config: StrategyConfig) -> None  # Line 200
    async def generate_signals(self, data: MarketData) -> list[Signal]  # Line 210
    async def validate_signal(self, signal: Signal) -> bool  # Line 223
    def get_position_size(self, signal: Signal) -> Decimal  # Line 236
    def should_exit(self, position: Position, data: MarketData) -> bool  # Line 249
    async def start(self) -> None  # Line 264
    async def stop(self) -> None  # Line 269
    async def pause(self) -> None  # Line 274
    async def resume(self) -> None  # Line 279
    async def prepare_for_backtest(self, config: dict[str, Any]) -> None  # Line 285
    async def process_historical_data(self, data: MarketData) -> list[Signal]  # Line 290
    async def get_backtest_metrics(self) -> dict[str, Any]  # Line 295
    def get_performance_summary(self) -> dict[str, Any]  # Line 301
    def get_real_time_metrics(self) -> dict[str, Any]  # Line 306
    async def get_state(self) -> dict[str, Any]  # Line 311
```

#### Class: `TrendStrategyInterface`

**Inherits**: BaseStrategyInterface
**Purpose**: Interface for trend-following strategies

```python
class TrendStrategyInterface(BaseStrategyInterface):
    def calculate_trend_strength(self, data: MarketData) -> float  # Line 325
    def identify_trend_direction(self, data: MarketData) -> int  # Line 338
    def get_trend_confirmation(self, data: MarketData) -> bool  # Line 351
```

#### Class: `MeanReversionStrategyInterface`

**Inherits**: BaseStrategyInterface
**Purpose**: Interface for mean reversion strategies

```python
class MeanReversionStrategyInterface(BaseStrategyInterface):
    def calculate_mean_deviation(self, data: MarketData) -> float  # Line 368
    def is_oversold(self, data: MarketData) -> bool  # Line 381
    def is_overbought(self, data: MarketData) -> bool  # Line 394
    def calculate_reversion_probability(self, data: MarketData) -> float  # Line 407
```

#### Class: `ArbitrageStrategyInterface`

**Inherits**: BaseStrategyInterface
**Purpose**: Interface for arbitrage strategies

```python
class ArbitrageStrategyInterface(BaseStrategyInterface):
    async def identify_arbitrage_opportunities(self, market_data_sources: list[MarketData]) -> list[dict[str, Any]]  # Line 424
    def calculate_profit_potential(self, opportunity: dict[str, Any]) -> Decimal  # Line 439
    def validate_arbitrage_execution(self, opportunity: dict[str, Any]) -> bool  # Line 452
```

#### Class: `MarketMakingStrategyInterface`

**Inherits**: BaseStrategyInterface
**Purpose**: Interface for market making strategies

```python
class MarketMakingStrategyInterface(BaseStrategyInterface):
    def calculate_optimal_spread(self, data: MarketData) -> tuple[Decimal, Decimal]  # Line 469
    def manage_inventory(self, current_position: Position) -> dict[str, Any]  # Line 482
    def calculate_quote_adjustment(self, market_impact: float) -> float  # Line 495
```

#### Class: `StrategyFactoryInterface`

**Inherits**: ABC
**Purpose**: Interface for strategy factories

```python
class StrategyFactoryInterface(ABC):
    async def create_strategy(self, strategy_type: StrategyType, config: StrategyConfig) -> BaseStrategyInterface  # Line 512
    def get_supported_strategies(self) -> list[StrategyType]  # Line 528
    def validate_strategy_requirements(self, strategy_type: StrategyType, config: StrategyConfig) -> bool  # Line 538
```

#### Class: `BacktestingServiceInterface`

**Inherits**: Protocol
**Purpose**: Interface for backtesting service integration

```python
class BacktestingServiceInterface(Protocol):
    async def run_backtest(self, strategy: BaseStrategyInterface, config: dict[str, Any]) -> dict[str, Any]  # Line 557
    async def validate_backtest_config(self, config: dict[str, Any]) -> bool  # Line 572
    async def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> list[MarketData]  # Line 584
```

#### Class: `StrategyRegistryInterface`

**Inherits**: Protocol
**Purpose**: Interface for strategy registry

```python
class StrategyRegistryInterface(Protocol):
    def register_strategy(self, strategy_id: str, strategy: BaseStrategyInterface) -> None  # Line 604
    def get_strategy(self, strategy_id: str) -> BaseStrategyInterface | None  # Line 614
    def list_strategies(self) -> list[str]  # Line 626
    def remove_strategy(self, strategy_id: str) -> bool  # Line 635
```

#### Class: `MarketDataProviderInterface`

**Inherits**: Protocol
**Purpose**: Interface for market data access

```python
class MarketDataProviderInterface(Protocol):
    async def get_current_price(self, symbol: str) -> Decimal | None  # Line 651
    async def get_market_regime(self, symbol: str) -> MarketRegime  # Line 663
```

#### Class: `StrategyDataRepositoryInterface`

**Inherits**: Protocol
**Purpose**: Interface for strategy data persistence

```python
class StrategyDataRepositoryInterface(Protocol):
    async def load_strategy_state(self, strategy_id: str) -> dict[str, Any] | None  # Line 684
    async def save_strategy_state(self, strategy_id: str, state: dict[str, Any]) -> None  # Line 696
    async def get_strategy_trades(self, ...) -> list[dict[str, Any]]  # Line 706
    async def save_trade(self, strategy_id: str, trade: dict[str, Any]) -> None  # Line 722
    async def get_strategy_positions(self, strategy_id: str) -> list[dict[str, Any]]  # Line 732
    async def save_performance_metrics(self, ...) -> None  # Line 744
    async def load_performance_history(self, ...) -> list[dict[str, Any]]  # Line 757
```

#### Class: `StrategyServiceInterface`

**Inherits**: Protocol
**Purpose**: Interface for strategy service business logic operations

```python
class StrategyServiceInterface(Protocol):
    async def register_strategy(self, strategy_id: str, strategy_instance: Any, config: StrategyConfig) -> None  # Line 777
    async def start_strategy(self, strategy_id: str) -> None  # Line 783
    async def stop_strategy(self, strategy_id: str) -> None  # Line 787
    async def process_market_data(self, market_data: MarketData) -> dict[str, list[Signal]]  # Line 791
    async def validate_signal(self, strategy_id: str, signal: Signal) -> bool  # Line 795
    async def get_strategy_performance(self, strategy_id: str) -> dict[str, Any]  # Line 799
    async def get_all_strategies(self) -> dict[str, dict[str, Any]]  # Line 803
    async def cleanup_strategy(self, strategy_id: str) -> None  # Line 807
```

#### Class: `OptimizationResult`

**Purpose**: Generic optimization result that doesn't depend on backtesting module

```python
class OptimizationResult:
    def __init__(self, ...)  # Line 817
```

#### Class: `OptimizationConfig`

**Purpose**: Generic optimization configuration that doesn't depend on backtesting module

```python
class OptimizationConfig:
    def __init__(self, ...)  # Line 837
```

#### Class: `OptimizationEngineInterface`

**Inherits**: Protocol
**Purpose**: Interface for optimization engines that genetic algorithms can use

```python
class OptimizationEngineInterface(Protocol):
    async def run_optimization(self, strategy: Any, config: OptimizationConfig) -> OptimizationResult  # Line 857
```

### File: metrics.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.core.types import Signal`
- `from src.utils.decorators import time_execution`

#### Class: `PerformanceMetrics`

**Inherits**: BaseModel
**Purpose**: Comprehensive performance metrics for strategies

```python
class PerformanceMetrics(BaseModel):
```

#### Class: `MetricsCalculator`

**Purpose**: Calculator for strategy performance metrics

```python
class MetricsCalculator:
    def __init__(self, config: dict[str, Any] | None = None)  # Line 87
    async def calculate_comprehensive_metrics(self, ...) -> PerformanceMetrics  # Line 107
    async def _calculate_basic_metrics(self, equity_curve: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]  # Line 167
    async def _calculate_risk_metrics(self, equity_curve: list[dict[str, Any]], trades: list[dict[str, Any]]) -> dict[str, Any]  # Line 195
    async def _calculate_drawdown_metrics(self, equity_curve: list[dict[str, Any]]) -> dict[str, Any]  # Line 263
    async def _calculate_trade_metrics(self, trades: list[dict[str, Any]]) -> dict[str, Any]  # Line 331
    async def _calculate_signal_metrics(self, signals: list[Signal], trades: list[dict[str, Any]]) -> dict[str, Any]  # Line 367
    async def _calculate_timing_metrics(self, trades: list[dict[str, Any]], signals: list[Signal]) -> dict[str, Any]  # Line 404
```

#### Class: `RealTimeMetricsTracker`

**Purpose**: Real-time metrics tracker for live strategy monitoring

```python
class RealTimeMetricsTracker:
    def __init__(self, strategy_id: str, config: dict[str, Any] | None = None)  # Line 436
    async def update_equity(self, equity: float, timestamp: datetime | None = None) -> None  # Line 466
    async def add_trade(self, trade_data: dict[str, Any]) -> None  # Line 490
    async def add_signal(self, signal: Signal) -> None  # Line 505
    async def _check_and_update_metrics(self) -> None  # Line 520
    async def _update_metrics(self) -> None  # Line 527
    def get_current_metrics(self) -> PerformanceMetrics  # Line 547
    def get_metrics_summary(self) -> dict[str, Any]  # Line 556
    async def reset_metrics(self) -> None  # Line 575
```

#### Class: `StrategyComparator`

**Purpose**: Compare performance between multiple strategies

```python
class StrategyComparator:
    def __init__(self)  # Line 594
    async def compare_strategies(self, strategy_metrics: dict[str, PerformanceMetrics]) -> dict[str, Any]  # Line 598
```

### File: performance_monitor.py

**Key Imports:**
- `from src.core.exceptions import PerformanceError`
- `from src.core.logging import get_logger`
- `from src.core.types import MarketRegime`
- `from src.core.types import Position`
- `from src.core.types import Trade`

#### Class: `PerformanceMetrics`

**Purpose**: Comprehensive performance metrics for a trading strategy

```python
class PerformanceMetrics:
    def __init__(self, strategy_name: str)  # Line 41
```

#### Class: `PerformanceMonitor`

**Purpose**: Comprehensive performance monitoring system for trading strategies

```python
class PerformanceMonitor:
    def __init__(self, ...)  # Line 129
    async def add_strategy(self, strategy: BaseStrategyInterface) -> None  # Line 177
    async def remove_strategy(self, strategy_name: str) -> None  # Line 219
    async def start_monitoring(self) -> None  # Line 239
    async def stop_monitoring(self) -> None  # Line 247
    async def _monitoring_loop(self) -> None  # Line 263
    async def _update_all_metrics(self) -> None  # Line 322
    async def _update_strategy_metrics(self, strategy_name: str) -> None  # Line 342
    def _update_trade_statistics(self, metrics: PerformanceMetrics, trades: list[Trade]) -> None  # Line 387
    def _calculate_consecutive_trades(self, metrics: PerformanceMetrics, trades: list[Trade]) -> None  # Line 446
    async def _update_pnl_metrics(self, ...) -> None  # Line 485
    def _calculate_risk_ratios(self, metrics: PerformanceMetrics) -> None  # Line 559
    def _update_drawdown_analysis(self, metrics: PerformanceMetrics) -> None  # Line 599
    def _update_time_metrics(self, metrics: PerformanceMetrics, trades: list[Trade]) -> None  # Line 626
    def _update_exposure_metrics(self, ...) -> None  # Line 649
    def _calculate_risk_metrics(self, metrics: PerformanceMetrics) -> None  # Line 682
    async def _check_performance_alerts(self) -> None  # Line 707
    def _update_strategy_rankings(self) -> None  # Line 736
    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float  # Line 752
    async def get_strategy_performance(self, strategy_name: str) -> dict[str, Any]  # Line 789
    async def get_comparative_analysis(self) -> dict[str, Any]  # Line 881
    async def _calculate_portfolio_metrics(self) -> dict[str, Any]  # Line 921
    async def _get_current_positions(self, strategy_name: str) -> list[Position]  # Line 996
    async def _get_recent_trades(self, strategy_name: str, limit: int = 1000, offset: int = 0) -> list[Trade]  # Line 1075
    def _validate_trade_query_params(self, strategy_name: str, limit: int, offset: int) -> None  # Line 1100
    async def _fetch_trade_data(self, strategy_name: str) -> list[dict[str, Any]]  # Line 1111
    def _convert_trade_dicts_to_objects(self, trade_dicts: list[dict[str, Any]]) -> list[Trade]  # Line 1126
    def _create_trade_from_dict(self, trade_dict: dict[str, Any]) -> 'Trade | None'  # Line 1141
    def _map_legacy_trade_fields(self, trade_dict: dict[str, Any]) -> dict[str, Any]  # Line 1208
    async def _get_current_price(self, symbol: str) -> 'Decimal | None'  # Line 1239
    def _calculate_position_pnl(self, position: Position, current_price: Decimal) -> Decimal  # Line 1259
    async def _load_historical_performance(self, strategy_name: str) -> None  # Line 1268
    async def _save_performance_metrics(self, strategy_name: str) -> None  # Line 1293
    async def _persist_metrics(self) -> None  # Line 1330
    async def _send_performance_alerts(self, strategy_name: str, alerts: list[str]) -> None  # Line 1368
```

### File: portfolio_allocator.py

**Key Imports:**
- `from src.core.exceptions import AllocationError`
- `from src.core.logging import get_logger`
- `from src.core.types import MarketRegime`
- `from src.core.types import Signal`
- `from src.core.types import SignalDirection`

#### Class: `StrategyAllocation`

**Purpose**: Represents allocation for a single strategy

```python
class StrategyAllocation:
    def __init__(self, ...)  # Line 43
```

#### Class: `PortfolioAllocator`

**Purpose**: Dynamic portfolio allocator for trading strategies

```python
class PortfolioAllocator:
    def __init__(self, ...)  # Line 98
    async def add_strategy(self, strategy: BaseStrategyInterface, initial_weight: float = 0.1) -> bool  # Line 165
    async def _validate_strategy(self, strategy: BaseStrategyInterface) -> bool  # Line 225
    async def _calculate_strategy_correlation(self, new_strategy: BaseStrategyInterface) -> float  # Line 278
    async def rebalance_portfolio(self) -> dict[str, Any]  # Line 324
    async def _update_strategy_metrics(self) -> None  # Line 367
    async def _calculate_optimal_weights(self) -> dict[str, float]  # Line 404
    def _build_returns_matrix(self, strategies: list[str]) -> ndarray | None  # Line 454
    def _optimize_sharpe_ratio(self, expected_returns: np.ndarray, cov_matrix: np.ndarray, n_strategies: int) -> np.ndarray  # Line 486
    def _apply_weight_constraints(self, weights: np.ndarray) -> np.ndarray  # Line 539
    def _calculate_performance_based_weights(self) -> dict[str, float]  # Line 550
    def _apply_regime_adjustments(self, weights: dict[str, float]) -> dict[str, float]  # Line 592
    async def _execute_rebalancing(self, target_weights: dict[str, float]) -> list[dict[str, Any]]  # Line 629
    async def _calculate_portfolio_metrics(self) -> dict[str, Any]  # Line 680
    async def update_market_regime(self, new_regime: MarketRegime) -> None  # Line 752
    async def remove_strategy(self, strategy_name: str, reason: str = 'manual') -> bool  # Line 774
    def get_strategy_allocation(self, strategy: BaseStrategyInterface) -> StrategyAllocation | None  # Line 818
    async def calculate_optimal_weights(self) -> dict[str, float]  # Line 832
    def update_strategy_performance(self, strategy: BaseStrategyInterface, performance_data: dict[str, float]) -> bool  # Line 841
    def get_allocation_status(self) -> dict[str, Any]  # Line 874
    async def should_rebalance(self) -> bool  # Line 929
```

### File: repository.py

**Key Imports:**
- `from src.core.exceptions import RepositoryError`
- `from src.core.exceptions import ValidationError`
- `from src.core.logging import get_logger`
- `from src.core.types.strategy import StrategyMetrics`
- `from src.database.models import AnalyticsStrategyMetrics`

#### Class: `StrategyRepositoryInterface`

**Inherits**: ABC
**Purpose**: Interface for strategy data repository operations

```python
class StrategyRepositoryInterface(ABC):
    async def create_strategy(self, strategy: Strategy) -> Strategy  # Line 37
    async def get_strategy(self, strategy_id: str) -> Strategy | None  # Line 42
    async def update_strategy(self, strategy_id: str, updates: dict[str, Any]) -> Strategy | None  # Line 47
    async def delete_strategy(self, strategy_id: str) -> bool  # Line 52
    async def get_strategies_by_bot(self, bot_id: str) -> list[Strategy]  # Line 57
    async def get_active_strategies(self, bot_id: str | None = None) -> list[Strategy]  # Line 62
    async def save_strategy_state(self, strategy_id: str, state_data: dict[str, Any]) -> bool  # Line 67
    async def load_strategy_state(self, strategy_id: str) -> dict[str, Any] | None  # Line 72
    async def save_strategy_metrics(self, strategy_id: str, metrics: StrategyMetrics) -> bool  # Line 77
    async def get_strategy_metrics(self, ...) -> list[AnalyticsStrategyMetrics]  # Line 82
    async def save_strategy_signals(self, signals: list[Signal]) -> list[Signal]  # Line 89
    async def get_strategy_signals(self, strategy_id: str, limit: int | None = None) -> list[Signal]  # Line 94
```

#### Class: `StrategyRepository`

**Inherits**: DatabaseRepository, StrategyRepositoryInterface
**Purpose**: Strategy repository with database integration using UoW pattern

```python
class StrategyRepository(DatabaseRepository, StrategyRepositoryInterface):
    def __init__(self, session: AsyncSession)  # Line 104
    async def create_strategy(self, strategy: Strategy) -> Strategy  # Line 114
    async def get_strategy(self, strategy_id: str) -> Strategy | None  # Line 126
    async def update_strategy(self, strategy_id: str, updates: dict[str, Any]) -> Strategy | None  # Line 163
    async def delete_strategy(self, strategy_id: str) -> bool  # Line 204
    async def get_strategies_by_bot(self, bot_id: str) -> list[Strategy]  # Line 220
    async def get_active_strategies(self, bot_id: str | None = None) -> list[Strategy]  # Line 246
    async def save_strategy_state(self, strategy_id: str, state_data: dict[str, Any]) -> bool  # Line 273
    async def load_strategy_state(self, strategy_id: str) -> dict[str, Any] | None  # Line 305
    async def save_strategy_metrics(self, strategy_id: str, metrics: StrategyMetrics) -> bool  # Line 342
    async def get_strategy_metrics(self, ...) -> list[AnalyticsStrategyMetrics]  # Line 382
    async def save_strategy_signals(self, signals: list[Signal]) -> list[Signal]  # Line 414
    async def get_strategy_signals(self, strategy_id: str, limit: int | None = None) -> list[Signal]  # Line 435
    async def get_strategy_trades(self, ...) -> list[Trade]  # Line 465
    async def get_strategy_performance_summary(self, strategy_id: str) -> dict[str, Any]  # Line 497
```

### File: service.py

**Key Imports:**
- `from src.core.base.service import BaseService`
- `from src.core.caching.cache_decorators import cached`
- `from src.core.caching.cache_manager import get_cache_manager`
- `from src.core.exceptions import ServiceError`
- `from src.core.exceptions import StrategyError`

#### Class: `StrategyService`

**Inherits**: BaseService, StrategyServiceInterface
**Purpose**: Service layer for strategy operations and management

```python
class StrategyService(BaseService, StrategyServiceInterface):
    def __init__(self, ...)  # Line 68
    async def _do_start(self) -> None  # Line 126
    async def _build_strategy_service_container(self) -> StrategyServiceContainer  # Line 154
    async def register_strategy(self, strategy_id: str, strategy_instance: Any, config: StrategyConfig) -> None  # Line 234
    async def _register_strategy_impl(self, strategy_id: str, strategy_instance: Any, config: StrategyConfig) -> None  # Line 256
    async def start_strategy(self, strategy_id: str) -> None  # Line 331
    async def _start_strategy_impl(self, strategy_id: str) -> None  # Line 345
    async def stop_strategy(self, strategy_id: str) -> None  # Line 361
    async def _stop_strategy_impl(self, strategy_id: str) -> None  # Line 372
    async def process_market_data(self, market_data: MarketData) -> dict[str, list[Signal]]  # Line 385
    async def _process_market_data_impl(self, market_data: MarketData) -> dict[str, list[Signal]]  # Line 399
    async def validate_signal(self, strategy_id: str, signal: Signal) -> bool  # Line 500
    async def validate_strategy_config(self, config: StrategyConfig) -> bool  # Line 542
    async def _validate_strategy_specific_config(self, config: StrategyConfig) -> bool  # Line 573
    async def _validate_start_conditions(self, strategy_id: str) -> bool  # Line 578
    async def _update_strategy_metrics(self, strategy_id: str, signals: list[Signal]) -> None  # Line 594
    async def _record_strategy_analytics(self, strategy_id: str, signals: list[Signal]) -> None  # Line 634
    async def _calculate_win_rate(self, strategy_id: str, signal_history: list[Signal]) -> float  # Line 693
    async def _calculate_sharpe_ratio(self, strategy_id: str) -> float  # Line 716
    async def _calculate_max_drawdown(self, strategy_id: str) -> float  # Line 747
    async def get_strategy_performance(self, strategy_id: str) -> dict[str, Any]  # Line 782
    async def _get_strategy_performance_impl(self, strategy_id: str) -> dict[str, Any]  # Line 796
    async def get_cached_strategy_metrics(self, strategy_id: str) -> dict[str, Any] | None  # Line 826
    async def get_strategy_performance_with_cache(self, strategy_id: str) -> dict[str, Any]  # Line 847
    async def get_all_strategies(self) -> dict[str, dict[str, Any]]  # Line 892
    async def cleanup_strategy(self, strategy_id: str) -> None  # Line 907
    async def _cleanup_strategy_impl(self, strategy_id: str) -> None  # Line 918
    async def _service_health_check(self) -> Any  # Line 947
    def get_metrics(self) -> dict[str, Any]  # Line 978
    def resolve_dependency(self, dependency_name: str) -> Any  # Line 1008
```

#### Functions:

```python
def cache_strategy_signals(strategy_id_arg_name: str, ttl: int = DEFAULT_CACHE_TTL) -> Callable  # Line 46
```

### File: shared_utilities.py

**Key Imports:**
- `from src.core.caching.cache_decorators import cached`
- `from src.core.caching.cache_manager import CacheManager`
- `from src.core.caching.cache_manager import get_cache_manager`
- `from src.core.exceptions import ExecutionError`
- `from src.core.exceptions import ServiceError`

#### Class: `StrategyIntegratedBase`

**Inherits**: ABC
**Purpose**: Enhanced base class that provides comprehensive module integration

```python
class StrategyIntegratedBase(ABC):
    def __init__(self, ...)  # Line 66
    async def initialize_validation_service(self) -> None  # Line 112
    def set_monitoring_services(self, ...)  # Line 143
    def _get_available_integrations(self) -> dict[str, bool]  # Line 157
    async def validate_market_data_comprehensive(self, data: MarketData) -> tuple[bool, list[str]]  # Line 171
    async def calculate_technical_indicators(self, data: MarketData, indicators: list[str], periods: dict[str, int] = None) -> 'dict[str, Decimal | None]'  # Line 238
    def format_signal_metadata(self, signal: Signal, additional_data: dict[str, Any] = None) -> dict[str, Any]  # Line 354
    async def record_signal_metrics(self, ...)  # Line 395
    async def safe_execute_with_monitoring(self, operation_name: str, operation_func, *args, **kwargs) -> Any  # Line 441
    def get_comprehensive_status(self) -> dict[str, Any]  # Line 516
    async def cleanup_resources(self)  # Line 532
```

#### Class: `StrategyDataAccessMixin`

**Purpose**: Mixin providing data access patterns for strategies

```python
class StrategyDataAccessMixin:
    def __init__(self, data_service: 'DataServiceInterface | None' = None)  # Line 565
    async def get_indicator_data(self, symbol: str, indicator: str, period: int) -> 'Decimal | None'  # Line 569
```

#### Functions:

```python
def create_comprehensive_signal(...) -> Signal  # Line 605
async def calculate_position_size_comprehensive(...) -> Decimal  # Line 646
```

### File: arbitrage_scanner.py

**Key Imports:**
- `from src.core.exceptions import ArbitrageError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import MarketData`
- `from src.core.types import Position`
- `from src.core.types import Signal`

#### Class: `ArbitrageOpportunity`

**Inherits**: BaseStrategy
**Purpose**: Arbitrage opportunity scanner for detecting and prioritizing arbitrage opportunities

```python
class ArbitrageOpportunity(BaseStrategy):
    def __init__(self, config: dict, services: 'StrategyServiceContainer')  # Line 45
    def strategy_type(self) -> StrategyType  # Line 93
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]  # Line 98
    async def _scan_arbitrage_opportunities(self) -> list[Signal]  # Line 139
    async def _scan_cross_exchange_opportunities(self) -> list[Signal]  # Line 175
    async def _scan_triangular_opportunities(self) -> list[Signal]  # Line 264
    async def validate_signal(self, signal: Signal) -> bool  # Line 352
    def get_position_size(self, signal: Signal) -> Decimal  # Line 421
    async def should_exit(self, position: Position, data: MarketData) -> bool  # Line 476
    async def _get_current_cross_exchange_spread(self, symbol: str, buy_exchange: str, sell_exchange: str) -> Decimal  # Line 545
    def _calculate_cross_exchange_fees(self, buy_price: Decimal, sell_price: Decimal) -> Decimal  # Line 590
    def _calculate_triangular_fees(self, rate1: Decimal, rate2: Decimal, rate3: Decimal) -> Decimal  # Line 628
    def _calculate_priority(self, profit_percentage: float, arbitrage_type: str) -> float  # Line 667
    def _prioritize_opportunities(self, signals: list[Signal]) -> list[Signal]  # Line 712
    async def _check_triangular_path(self, path: list[str]) -> 'Signal | None'  # Line 743
    async def post_trade_processing(self, trade_result: dict[str, Any]) -> None  # Line 817
```

### File: breakout.py

**Key Imports:**
- `from src.core.types import MarketData`
- `from src.core.types import Position`
- `from src.core.types import Signal`
- `from src.core.types import SignalDirection`
- `from src.core.types import StrategyType`

#### Class: `BreakoutStrategy`

**Inherits**: BaseStrategy
**Purpose**: Breakout Strategy Implementation

```python
class BreakoutStrategy(BaseStrategy):
    def __init__(self, config: dict[str, Any], services: 'StrategyServiceContainer | None' = None)  # Line 56
    def strategy_type(self) -> StrategyType  # Line 106
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]  # Line 115
    async def _update_support_resistance_levels(self, data: MarketData) -> None  # Line 196
    async def _check_consolidation_period(self, data: MarketData) -> bool  # Line 228
    async def _check_resistance_breakout(self, data: MarketData) -> 'dict[str, Any] | None'  # Line 263
    async def _check_support_breakout(self, data: MarketData) -> 'dict[str, Any] | None'  # Line 315
    async def _check_volume_confirmation(self, data: MarketData) -> bool  # Line 347
    def _check_false_breakout(self, data: MarketData) -> 'dict[str, Any] | None'  # Line 373
    async def _generate_bullish_breakout_signal(self, data: MarketData, breakout_info: dict[str, Any]) -> 'Signal | None'  # Line 412
    async def _generate_bearish_breakout_signal(self, data: MarketData, breakout_info: dict[str, Any]) -> 'Signal | None'  # Line 483
    async def _generate_false_breakout_exit_signal(self, data: MarketData, false_breakout_info: dict[str, Any]) -> 'Signal | None'  # Line 556
    async def validate_signal(self, signal: Signal) -> bool  # Line 608
    def get_position_size(self, signal: Signal) -> Decimal  # Line 671
    async def should_exit(self, position: Position, data: MarketData) -> bool  # Line 722
    def get_strategy_info(self) -> dict[str, Any]  # Line 800
```

### File: cross_exchange_arbitrage.py

**Key Imports:**
- `from src.core.exceptions import ArbitrageError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import MarketData`
- `from src.core.types import Position`
- `from src.core.types import Signal`

#### Class: `CrossExchangeArbitrageStrategy`

**Inherits**: BaseStrategy
**Purpose**: Cross-exchange arbitrage strategy for detecting and executing price differences

```python
class CrossExchangeArbitrageStrategy(BaseStrategy):
    def __init__(self, config: dict, services: 'StrategyServiceContainer')  # Line 53
    def strategy_type(self) -> StrategyType  # Line 87
    def name(self) -> str  # Line 92
    def name(self, value: str) -> None  # Line 98
    def version(self) -> str  # Line 103
    def version(self, value: str) -> None  # Line 109
    def status(self) -> StrategyStatus  # Line 114
    def status(self, value: StrategyStatus) -> None  # Line 120
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]  # Line 125
    async def _detect_arbitrage_opportunities(self, symbol: str) -> list[Signal]  # Line 167
    def _calculate_total_fees(self, buy_price: Decimal, sell_price: Decimal) -> Decimal  # Line 270
    async def _validate_execution_timing(self, symbol: str) -> bool  # Line 336
    async def validate_signal(self, signal: Signal) -> bool  # Line 391
    def get_position_size(self, signal: Signal) -> Decimal  # Line 448
    async def should_exit(self, position: Position, data: MarketData) -> bool  # Line 541
    async def _get_current_spread(self, symbol: str, buy_exchange: str, sell_exchange: str) -> Decimal  # Line 602
    async def post_trade_processing(self, trade_result: dict[str, Any]) -> None  # Line 644
    async def _process_trade_result(self, trade_result: dict[str, Any]) -> None  # Line 663
```

### File: inventory_manager.py

**Key Imports:**
- `from src.core.exceptions import RiskManagementError`
- `from src.core.logging import get_logger`
- `from src.core.types import OrderRequest`
- `from src.core.types import OrderSide`
- `from src.core.types import OrderType`

#### Class: `InventoryManager`

**Purpose**: Inventory Manager for Market Making Strategy

```python
class InventoryManager:
    def __init__(self, config: dict[str, Any], services: 'StrategyServiceContainer | None' = None)  # Line 41
    async def update_inventory(self, position: Position) -> None  # Line 88
    async def should_rebalance(self) -> bool  # Line 116
    async def calculate_rebalance_orders(self, current_price: Decimal) -> list[OrderRequest]  # Line 163
    async def should_emergency_liquidate(self) -> bool  # Line 238
    async def calculate_emergency_orders(self, current_price: Decimal) -> list[OrderRequest]  # Line 270
    async def calculate_spread_adjustment(self, base_spread: Decimal) -> Decimal  # Line 326
    async def calculate_size_adjustment(self, base_size: Decimal) -> Decimal  # Line 362
    async def record_rebalance(self, cost: Decimal) -> None  # Line 398
    async def record_emergency(self, cost: Decimal) -> None  # Line 421
    def get_inventory_summary(self) -> dict[str, Any]  # Line 442
    async def validate_inventory_limits(self, new_position: Position) -> bool  # Line 471
```

### File: market_making.py

**Key Imports:**
- `from src.core.exceptions import ValidationError`
- `from src.core.types import MarketData`
- `from src.core.types import Position`
- `from src.core.types import Signal`
- `from src.core.types import SignalDirection`

#### Class: `OrderLevel`

**Purpose**: Represents a single order level in the market making strategy

```python
class OrderLevel:
```

#### Class: `InventoryState`

**Purpose**: Current inventory state for the market making strategy

```python
class InventoryState:
```

#### Class: `MarketMakingStrategy`

**Inherits**: BaseStrategy
**Purpose**: Market Making Strategy Implementation

```python
class MarketMakingStrategy(BaseStrategy):
    def __init__(self, config: dict[str, Any], services: 'StrategyServiceContainer | None' = None)  # Line 85
    def strategy_type(self) -> StrategyType  # Line 163
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]  # Line 171
    def _calculate_level_spread(self, level: int, base_spread: Decimal, volatility: float) -> Decimal  # Line 281
    def _calculate_correlation_risk_factor(self) -> Decimal  # Line 319
    def _calculate_level_size(self, level: int) -> Decimal  # Line 339
    async def validate_signal(self, signal: Signal) -> bool  # Line 362
    def _check_inventory_limits(self, signal: Signal) -> bool  # Line 440
    def get_position_size(self, signal: Signal) -> Decimal  # Line 463
    async def should_exit(self, position: Position, data: MarketData) -> bool  # Line 500
    async def _should_rebalance_inventory(self, position: Position) -> bool  # Line 556
    async def update_inventory_state(self, new_position: Position) -> None  # Line 603
    async def update_performance_metrics(self, trade_result: dict[str, Any]) -> None  # Line 632
    def get_strategy_info(self) -> dict[str, Any]  # Line 677
```

### File: mean_reversion.py

**Key Imports:**
- `from src.core.caching.cache_decorators import cached`
- `from src.core.caching.cache_manager import get_cache_manager`
- `from src.core.types import MarketData`
- `from src.core.types import Position`
- `from src.core.types import Signal`

#### Class: `MeanReversionStrategy`

**Inherits**: BaseStrategy
**Purpose**: Mean Reversion Strategy Implementation

```python
class MeanReversionStrategy(BaseStrategy):
    def __init__(self, config: dict[str, Any], services: 'StrategyServiceContainer | None' = None)  # Line 63
    def strategy_type(self) -> StrategyType  # Line 117
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]  # Line 127
    async def validate_signal(self, signal: Signal) -> bool  # Line 333
    def get_position_size(self, signal: Signal) -> Decimal  # Line 391
    async def should_exit(self, position: Position, data: MarketData) -> bool  # Line 438
    def get_strategy_info(self) -> dict[str, Any]  # Line 505
    async def _record_strategy_analytics(self, signal: Signal, market_data: MarketData, metrics: dict[str, Any]) -> None  # Line 535
```

### File: spread_optimizer.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.core.types import MarketData`
- `from src.core.types import OrderBook`
- `from src.strategies.dependencies import StrategyServiceContainer`
- `from src.utils.decorators import time_execution`

#### Class: `SpreadOptimizer`

**Purpose**: Spread Optimizer for Market Making Strategy

```python
class SpreadOptimizer:
    def __init__(self, config: dict[str, Any], services: 'StrategyServiceContainer | None' = None)  # Line 40
    async def optimize_spread(self, ...) -> Decimal  # Line 93
    async def _calculate_volatility_adjustment(self, base_spread: Decimal) -> Decimal  # Line 167
    async def _calculate_imbalance_adjustment(self, base_spread: Decimal, order_book: 'OrderBook | None') -> Decimal  # Line 212
    async def _calculate_competitor_adjustment(self, base_spread: Decimal, competitor_spreads: 'list[float] | None') -> Decimal  # Line 265
    async def _calculate_impact_adjustment(self, base_spread: Decimal) -> Decimal  # Line 321
    def _update_history(self, market_data: MarketData) -> None  # Line 359
    async def calculate_optimal_spread(self, ...) -> tuple[Decimal, Decimal]  # Line 398
    async def should_widen_spread(self, market_data: MarketData) -> bool  # Line 448
    def get_optimization_summary(self) -> dict[str, Any]  # Line 498
```

### File: trend_following.py

**Key Imports:**
- `from src.core.types import MarketData`
- `from src.core.types import Position`
- `from src.core.types import Signal`
- `from src.core.types import SignalDirection`
- `from src.core.types import StrategyType`

#### Class: `TrendFollowingStrategy`

**Inherits**: BaseStrategy
**Purpose**: Trend Following Strategy Implementation

```python
class TrendFollowingStrategy(BaseStrategy):
    def __init__(self, config: dict[str, Any], services: 'StrategyServiceContainer | None' = None)  # Line 62
    def set_technical_indicators(self, technical_indicators: TechnicalIndicators) -> None  # Line 101
    def strategy_type(self) -> StrategyType  # Line 107
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]  # Line 115
    async def _check_volume_confirmation(self, data: MarketData) -> bool  # Line 232
    async def _generate_bullish_signal(self, data: MarketData, fast_ma: float, slow_ma: float, rsi: float) -> 'Signal | None'  # Line 262
    async def _generate_bearish_signal(self, data: MarketData, fast_ma: float, slow_ma: float, rsi: float) -> 'Signal | None'  # Line 333
    async def _generate_exit_signal(self, data: MarketData, direction: SignalDirection, reason: str) -> 'Signal | None'  # Line 404
    async def validate_signal(self, signal: Signal) -> bool  # Line 449
    def get_position_size(self, signal: Signal) -> Decimal  # Line 505
    async def should_exit(self, position: Position, data: MarketData) -> bool  # Line 561
    def _should_exit_by_time(self, position: Position) -> bool  # Line 631
    def _should_exit_by_trailing_stop(self, position: Position, data: MarketData) -> bool  # Line 654
    def get_strategy_info(self) -> dict[str, Any]  # Line 688
```

### File: triangular_arbitrage.py

**Key Imports:**
- `from src.core.exceptions import ArbitrageError`
- `from src.core.exceptions import ValidationError`
- `from src.core.types import MarketData`
- `from src.core.types import Position`
- `from src.core.types import Signal`

#### Class: `TriangularArbitrageStrategy`

**Inherits**: BaseStrategy
**Purpose**: Triangular arbitrage strategy for detecting and executing three-pair arbitrage

```python
class TriangularArbitrageStrategy(BaseStrategy):
    def __init__(self, config: dict, services: 'StrategyServiceContainer')  # Line 46
    def strategy_type(self) -> StrategyType  # Line 82
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]  # Line 87
    async def _detect_triangular_opportunities(self, symbol: str) -> list[Signal]  # Line 125
    async def _check_triangular_path(self, path: list[str]) -> 'Signal | None'  # Line 155
    def _calculate_triangular_fees(self, rate1: Decimal, rate2: Decimal, rate3: Decimal) -> Decimal  # Line 269
    async def _validate_triangular_timing(self, path: list[str]) -> bool  # Line 352
    async def validate_signal(self, signal: Signal) -> bool  # Line 408
    def get_position_size(self, signal: Signal) -> Decimal  # Line 472
    async def should_exit(self, position: Position, data: MarketData) -> bool  # Line 558
    async def post_trade_processing(self, trade_result: dict[str, Any]) -> None  # Line 619
    async def _process_trade_result(self, trade_result: dict[str, Any]) -> None  # Line 631
```

### File: validation.py

**Key Imports:**
- `from src.core.logging import get_logger`
- `from src.core.types import MarketData`
- `from src.core.types import Signal`
- `from src.core.types import SignalDirection`
- `from src.core.types import StrategyConfig`

#### Class: `ValidationResult`

**Inherits**: BaseModel
**Purpose**: Result of a validation operation

```python
class ValidationResult(BaseModel):
    def add_error(self, message: str) -> None  # Line 43
    def add_warning(self, message: str) -> None  # Line 48
    def merge(self, other: 'ValidationResult') -> None  # Line 52
```

#### Class: `BaseValidator`

**Inherits**: ABC
**Purpose**: Base class for all validators

```python
class BaseValidator(ABC):
    def __init__(self, name: str, config: dict[str, Any] | None = None)  # Line 64
    async def validate(self, target: Any, context: dict[str, Any] | None = None) -> ValidationResult  # Line 77
```

#### Class: `SignalValidator`

**Inherits**: BaseValidator
**Purpose**: Validator for trading signals

```python
class SignalValidator(BaseValidator):
    def __init__(self, config: dict[str, Any] | None = None)  # Line 96
    async def validate(self, signal: Signal, context: dict[str, Any] | None = None) -> ValidationResult  # Line 105
    async def _validate_required_fields(self, signal: Signal, result: ValidationResult) -> None  # Line 146
    async def _validate_strength(self, signal: Signal, result: ValidationResult) -> None  # Line 152
    async def _validate_timestamp(self, signal: Signal, result: ValidationResult) -> None  # Line 163
    async def _validate_direction(self, signal: Signal, result: ValidationResult) -> None  # Line 174
    async def _validate_symbol(self, signal: Signal, result: ValidationResult) -> None  # Line 180
    async def _validate_with_context(self, signal: Signal, context: dict[str, Any], result: ValidationResult) -> None  # Line 189
```

#### Class: `StrategyConfigValidator`

**Inherits**: BaseValidator
**Purpose**: Validator for strategy configurations

```python
class StrategyConfigValidator(BaseValidator):
    def __init__(self, config: dict[str, Any] | None = None)  # Line 213
    async def validate(self, config: StrategyConfig, context: dict[str, Any] | None = None) -> ValidationResult  # Line 225
    async def _validate_basic_config(self, config: StrategyConfig, result: ValidationResult) -> None  # Line 259
    async def _validate_strategy_specific(self, config: StrategyConfig, result: ValidationResult) -> None  # Line 275
    async def _validate_parameter_values(self, config: StrategyConfig, result: ValidationResult) -> None  # Line 289
    async def _validate_risk_parameters(self, config: StrategyConfig, result: ValidationResult) -> None  # Line 316
    async def _validate_exchange_compatibility(self, config: StrategyConfig, result: ValidationResult) -> None  # Line 333
```

#### Class: `MarketConditionValidator`

**Inherits**: BaseValidator
**Purpose**: Validator for market conditions and trading environment

```python
class MarketConditionValidator(BaseValidator):
    def __init__(self, config: dict[str, Any] | None = None)  # Line 348
    async def validate(self, market_data: MarketData, context: dict[str, Any] | None = None) -> ValidationResult  # Line 355
    async def _validate_volume(self, market_data: MarketData, result: ValidationResult) -> None  # Line 389
    async def _validate_spread(self, market_data: MarketData, result: ValidationResult) -> None  # Line 394
    async def _validate_price_data(self, market_data: MarketData, result: ValidationResult) -> None  # Line 402
    async def _validate_market_hours(self, market_data: MarketData, result: ValidationResult) -> None  # Line 414
```

#### Class: `CompositeValidator`

**Inherits**: BaseValidator
**Purpose**: Composite validator that runs multiple validators

```python
class CompositeValidator(BaseValidator):
    def __init__(self, validators: list[BaseValidator], name: str = 'CompositeValidator')  # Line 429
    async def validate(self, target: Any, context: dict[str, Any] | None = None) -> ValidationResult  # Line 440
```

#### Class: `ValidationFramework`

**Purpose**: Main validation framework for strategies

```python
class ValidationFramework:
    def __init__(self, config: dict[str, Any] | None = None)  # Line 476
    async def validate_signal(self, signal: Signal, market_data: MarketData | None = None) -> ValidationResult  # Line 497
    async def validate_strategy_config(self, config: StrategyConfig) -> ValidationResult  # Line 516
    async def validate_market_conditions(self, market_data: MarketData) -> ValidationResult  # Line 528
    async def validate_for_trading(self, signal: Signal, market_data: MarketData) -> ValidationResult  # Line 540
    async def batch_validate_signals(self, signals: list[Signal], market_data: MarketData | None = None) -> list[tuple[Signal, ValidationResult]]  # Line 556
    def add_custom_validator(self, validator: BaseValidator, validator_type: str = 'custom') -> None  # Line 578
    def get_validation_stats(self) -> dict[str, Any]  # Line 598
```

---
**Generated**: Complete reference for strategies module
**Total Classes**: 108
**Total Functions**: 7