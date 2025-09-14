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
- Caching
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
- `strategy_type(self) -> StrategyType` - Line 157
- `name(self) -> str` - Line 162
- `version(self) -> str` - Line 167
- `status(self) -> StrategyStatus` - Line 172
- `async generate_signals(self, data: MarketData) -> list[Signal]` - Line 194
- `async validate_signal(self, signal: Signal) -> bool` - Line 329
- `get_position_size(self, signal: Signal) -> Decimal` - Line 385
- `should_exit(self, position: Position, data: MarketData) -> bool` - Line 436
- `async pre_trade_validation(self, signal: Signal) -> bool` - Line 452
- `async post_trade_processing(self, trade_result: Any) -> None` - Line 478
- `set_risk_manager(self, risk_manager: Any) -> None` - Line 524
- `set_exchange(self, exchange: Any) -> None` - Line 532
- `set_data_service(self, data_service: Any) -> None` - Line 541
- `set_validation_framework(self, validation_framework: ValidationFramework) -> None` - Line 549
- `set_metrics_collector(self, metrics_collector: MetricsCollector) -> None` - Line 558
- `get_strategy_info(self) -> dict[str, Any]` - Line 567
- `async initialize(self, config: StrategyConfig) -> None` - Line 582
- `async start(self) -> bool` - Line 594
- `async stop(self) -> bool` - Line 644
- `async pause(self) -> None` - Line 670
- `async resume(self) -> None` - Line 676
- `async prepare_for_backtest(self, config: dict[str, Any]) -> None` - Line 683
- `async process_historical_data(self, data: MarketData) -> list[Signal]` - Line 699
- `async get_backtest_metrics(self) -> dict[str, Any]` - Line 715
- `get_real_time_metrics(self) -> dict[str, Any]` - Line 728
- `update_config(self, new_config: dict[str, Any]) -> None` - Line 774
- `async get_state(self) -> dict[str, Any]` - Line 789
- `get_performance_summary(self) -> dict[str, Any]` - Line 806
- `cleanup(self) -> None` - Line 836
- `async get_market_data(self, symbol: str) -> MarketData | None` - Line 870
- `async get_historical_data(self, symbol: str, timeframe: str, limit: int = 100) -> list[MarketData]` - Line 892
- `async execute_order(self, signal: Signal) -> Any | None` - Line 918
- `async save_state(self, state_data: dict[str, Any]) -> bool` - Line 964
- `async load_state(self) -> dict[str, Any] | None` - Line 987
- `get_metrics(self) -> dict[str, Any]` - Line 1064
- `is_healthy(self) -> bool` - Line 1080
- `async reset(self) -> bool` - Line 1104
- `set_execution_service(self, execution_service: Any) -> None` - Line 1172
- `get_status(self) -> StrategyStatus` - Line 1178
- `get_status_string(self) -> str` - Line 1182
- `async validate_market_data(self, data: MarketData | None) -> None` - Line 1219
- `async get_sma(self, symbol: str, period: int) -> Decimal | None` - Line 1250
- `async get_ema(self, symbol: str, period: int) -> Decimal | None` - Line 1271
- `async get_rsi(self, symbol: str, period: int = 14) -> Decimal | None` - Line 1290
- `async get_volatility(self, symbol: str, period: int) -> Decimal | None` - Line 1309
- `async get_atr(self, symbol: str, period: int) -> Decimal | None` - Line 1328
- `async get_volume_ratio(self, symbol: str, period: int) -> Decimal | None` - Line 1347
- `async get_bollinger_bands(self, symbol: str, period: int = 20, std_dev: float = 2.0) -> dict[str, Decimal] | None` - Line 1366
- `async get_macd(self, ...) -> dict[str, Decimal] | None` - Line 1393
- `async execute_with_algorithm(self, ...) -> dict[str, Any] | None` - Line 1422
- `async optimize_parameters(self, optimization_config: dict[str, Any] | None) -> dict[str, Any]` - Line 1470
- `async enhance_signals_with_ml(self, signals: list[Signal]) -> list[Signal]` - Line 1544
- `async get_allocated_capital(self) -> Decimal` - Line 1650
- `async execute_large_order(self, order_request: OrderRequest, max_position_size: Decimal | None = None) -> dict[str, Any] | None` - Line 1755
- `async get_execution_algorithms_status(self) -> dict[str, Any]` - Line 1817

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
- `delete_strategy_config(self, strategy_name: str) -> bool` - Line 486
- `get_config_summary(self) -> dict[str, Any]` - Line 529

### Implementation: `StrategyConfigTemplates` ✅

**Purpose**: Comprehensive strategy configuration templates for production deployment
**Status**: Complete

**Implemented Methods:**
- `get_arbitrage_scanner_config(risk_level, ...) -> dict[str, Any]` - Line 35
- `get_mean_reversion_config(timeframe: str = '1h', risk_level: str = 'medium') -> dict[str, Any]` - Line 139
- `get_trend_following_config(timeframe: str = '1h', trend_strength: str = 'medium') -> dict[str, Any]` - Line 263
- `get_market_making_config(symbol, ...) -> dict[str, Any]` - Line 380
- `get_volatility_breakout_config(volatility_regime: str = 'medium', breakout_type: str = 'range') -> dict[str, Any]` - Line 496
- `get_ensemble_config(strategy_types, ...) -> dict[str, Any]` - Line 602
- `get_all_templates(cls) -> dict[str, dict[str, Any]]` - Line 685
- `get_template_by_name(cls, template_name: str) -> dict[str, Any]` - Line 730
- `list_available_templates(cls) -> list[str]` - Line 751
- `get_templates_by_strategy_type(cls, strategy_type: str) -> dict[str, dict[str, Any]]` - Line 761
- `validate_template(cls, template: dict[str, Any]) -> tuple[bool, list[str]]` - Line 779

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
- `async validate_signal(self, signal: Signal) -> bool` - Line 681
- `get_position_size(self, signal: Signal) -> Decimal` - Line 762
- `should_exit(self, position: Position, data: MarketData) -> bool` - Line 855
- `get_strategy_info(self) -> dict[str, Any]` - Line 936
- `cleanup(self) -> None` - Line 1074

### Implementation: `EnhancedDynamicStrategyFactory` ✅

**Inherits**: BaseComponent
**Purpose**: Enhanced factory for creating dynamic strategies with service layer integration
**Status**: Complete

**Implemented Methods:**
- `async create_strategy(self, strategy_name: str, config: dict[str, Any], use_enhanced: bool = True) -> BaseStrategy | None` - Line 94
- `get_available_strategies(self) -> dict[str, str]` - Line 346
- `get_strategy_requirements(self, strategy_name: str) -> dict[str, Any]` - Line 355
- `async create_multiple_strategies(self, strategy_configs: dict[str, dict[str, Any]], use_enhanced: bool = True) -> dict[str, BaseStrategy | None]` - Line 393

### Implementation: `VolatilityBreakoutStrategy` ✅

**Inherits**: BaseStrategy
**Purpose**: Enhanced Volatility Breakout Strategy with service layer integration
**Status**: Complete

**Implemented Methods:**
- `name(self) -> str` - Line 58
- `name(self, value: str) -> None` - Line 67
- `version(self) -> str` - Line 72
- `version(self, value: str) -> None` - Line 77
- `status(self) -> str` - Line 82
- `status(self, value: str) -> None` - Line 87
- `strategy_type(self) -> StrategyType` - Line 147
- `set_technical_indicators(self, technical_indicators: TechnicalIndicators) -> None` - Line 151
- `set_strategy_service(self, strategy_service: StrategyService) -> None` - Line 156
- `set_regime_detector(self, regime_detector: MarketRegimeDetector) -> None` - Line 161
- `set_adaptive_risk_manager(self, adaptive_risk_manager: AdaptiveRiskManager) -> None` - Line 166
- `async validate_signal(self, signal: Signal) -> bool` - Line 1052
- `get_position_size(self, signal: Signal) -> Decimal` - Line 1167
- `should_exit(self, position: Position, data: MarketData) -> bool` - Line 1280
- `get_strategy_info(self) -> dict[str, Any]` - Line 1359
- `cleanup(self) -> None` - Line 1445

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
- `async generate_environment_aware_signal(self, strategy_name: str, market_data: MarketData, exchange: str) -> Signal | None` - Line 362
- `async update_strategy_performance(self, ...) -> None` - Line 535
- `get_environment_strategy_metrics(self, exchange: str) -> dict[str, Any]` - Line 567
- `async rebalance_strategies_for_environment(self, exchange: str) -> dict[str, Any]` - Line 601

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
- `add_member(self, genome: NEATGenome) -> None` - Line 914
- `calculate_average_fitness(self) -> None` - Line 919
- `select_parents(self, selection_pressure: float = 0.5) -> list[NEATGenome]` - Line 926
- `remove_worst_genomes(self, keep_ratio: float = 0.5) -> None` - Line 945

### Implementation: `SpeciationManager` ✅

**Purpose**: Manages species formation and evolution in NEAT population
**Status**: Complete

**Implemented Methods:**
- `speciate_population(self, population: list[NEATGenome]) -> None` - Line 984
- `allocate_offspring(self, total_offspring: int) -> dict[int, int]` - Line 1068

### Implementation: `NeuroEvolutionConfig` ✅

**Purpose**: Configuration for neuroevolution strategy
**Status**: Complete

### Implementation: `NeuroEvolutionStrategy` ✅

**Inherits**: BaseStrategy
**Purpose**: Neural network evolution strategy for trading decisions
**Status**: Complete

**Implemented Methods:**
- `async adapt_networks(self) -> None` - Line 1290
- `async evolve_population(self, fitness_evaluator: FitnessEvaluator) -> None` - Line 1371
- `async validate_signal(self, signal: Signal) -> bool` - Line 1523
- `get_position_size(self, signal: Signal) -> Decimal` - Line 1547
- `should_exit(self, position: Position, data: MarketData) -> bool` - Line 1571
- `get_strategy_info(self) -> dict[str, Any]` - Line 1625
- `get_evolution_summary(self) -> dict[str, Any]` - Line 1655
- `async save_population(self, filepath: str) -> None` - Line 1684
- `async load_population(self, filepath: str) -> None` - Line 1715

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

**Implemented Methods:**

### Implementation: `ConstraintHandler` ✅

**Purpose**: Handles constraints in multi-objective optimization
**Status**: Complete

**Implemented Methods:**
- `evaluate_constraints(self, objectives: dict[str, float]) -> dict[str, float]` - Line 139
- `is_feasible(self, objectives: dict[str, float], tolerance: float = 0.01) -> bool` - Line 172
- `apply_penalty(self, objectives: dict[str, float], constraint_violations: dict[str, float]) -> dict[str, float]` - Line 187

### Implementation: `DominanceComparator` ✅

**Purpose**: Implements dominance comparison for multi-objective optimization
**Status**: Complete

**Implemented Methods:**
- `dominates(self, solution1: dict[str, float], solution2: dict[str, float]) -> bool` - Line 244
- `non_dominated_sort(self, solutions: list[dict[str, float]]) -> list[list[int]]` - Line 284

### Implementation: `CrowdingDistanceCalculator` ✅

**Purpose**: Calculates crowding distance for diversity preservation
**Status**: Complete

**Implemented Methods:**
- `calculate_crowding_distance(self, solutions: list[dict[str, float]], front_indices: list[int]) -> list[float]` - Line 357

### Implementation: `ParetoFrontierManager` ✅

**Purpose**: Manages the Pareto frontier and provides analysis tools
**Status**: Complete

**Implemented Methods:**
- `update_frontier(self, solutions: list[ParetoSolution]) -> None` - Line 439
- `get_frontier_summary(self) -> dict[str, Any]` - Line 621

### Implementation: `NSGAIIOptimizer` ✅

**Purpose**: NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation
**Status**: Complete

**Implemented Methods:**
- `async optimize(self) -> list[ParetoSolution]` - Line 716
- `get_optimization_summary(self) -> dict[str, Any]` - Line 1161

### Implementation: `MultiObjectiveOptimizer` ✅

**Purpose**: Main interface for multi-objective optimization of trading strategies
**Status**: Complete

**Implemented Methods:**
- `async optimize_strategy(self, ...) -> list[ParetoSolution]` - Line 1222
- `get_pareto_frontier_data(self) -> dict[str, Any]` - Line 1274
- `export_results(self, filepath: str) -> None` - Line 1313

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
- `get_supported_strategies(self) -> list[StrategyType]` - Line 517
- `validate_strategy_requirements(self, strategy_type: StrategyType, config: StrategyConfig) -> bool` - Line 541
- `async create_strategy_with_validation(self, ...) -> BaseStrategyInterface` - Line 724
- `get_strategy_info(self, strategy_type: StrategyType) -> dict[str, Any]` - Line 842
- `list_available_strategies(self) -> dict[str, Any]` - Line 867

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
- `get_strategy_info(self) -> dict[str, Any]` - Line 972

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

### Implementation: `TechnicalRuleEngine` ✅

**Purpose**: Traditional technical analysis rule engine
**Status**: Complete

**Implemented Methods:**
- `async calculate_rsi(self, symbol: str) -> float` - Line 76
- `async calculate_moving_averages(self, symbol: str, current_price: float) -> tuple[float, float]` - Line 96
- `async evaluate_rules(self, ...) -> dict[str, Any]` - Line 117
- `update_rule_performance(self, rule: str, performance: float) -> None` - Line 202
- `adjust_rule_weights(self) -> None` - Line 211

### Implementation: `AIPredictor` ✅

**Purpose**: AI prediction component using machine learning
**Status**: Complete

**Implemented Methods:**
- `prepare_features(self, price_history: list[float], volume_history: list[float]) -> np.ndarray` - Line 243
- `async train_model(self, training_data: list[dict[str, Any]]) -> None` - Line 305
- `async predict(self, price_history: list[float], volume_history: list[float]) -> dict[str, Any]` - Line 353
- `update_performance(self, prediction: dict[str, Any], actual_outcome: float) -> None` - Line 407
- `get_performance_metrics(self) -> dict[str, float]` - Line 417

### Implementation: `RuleBasedAIStrategy` ✅

**Inherits**: BaseStrategy
**Purpose**: Hybrid strategy combining traditional technical analysis rules with AI predictions
**Status**: Complete

**Implemented Methods:**
- `strategy_type(self) -> StrategyType` - Line 493
- `async validate_signal(self, signal: Signal) -> bool` - Line 721
- `get_position_size(self, signal: Signal) -> Decimal` - Line 750
- `async should_exit(self, position: Position, data: MarketData) -> bool` - Line 782
- `adjust_component_weights(self) -> None` - Line 882
- `get_strategy_statistics(self) -> dict[str, Any]` - Line 930
- `get_strategy_stats(self) -> dict[str, Any]` - Line 999

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
- `async get_strategy_performance(self, strategy_name: str) -> dict[str, Any]` - Line 766
- `async get_comparative_analysis(self) -> dict[str, Any]` - Line 858

### Implementation: `StrategyAllocation` ✅

**Purpose**: Represents allocation for a single strategy
**Status**: Complete

**Implemented Methods:**

### Implementation: `PortfolioAllocator` ✅

**Purpose**: Dynamic portfolio allocator for trading strategies
**Status**: Complete

**Implemented Methods:**
- `async add_strategy(self, strategy: BaseStrategyInterface, initial_weight: float = 0.1) -> bool` - Line 163
- `async rebalance_portfolio(self) -> dict[str, Any]` - Line 310
- `async update_market_regime(self, new_regime: MarketRegime) -> None` - Line 738
- `async remove_strategy(self, strategy_name: str, reason: str = 'manual') -> bool` - Line 760
- `get_allocation_status(self) -> dict[str, Any]` - Line 804
- `async should_rebalance(self) -> bool` - Line 859

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
- `async update_strategy(self, strategy_id: str, updates: dict[str, Any]) -> Strategy | None` - Line 160
- `async delete_strategy(self, strategy_id: str) -> bool` - Line 201
- `async get_strategies_by_bot(self, bot_id: str) -> list[Strategy]` - Line 217
- `async get_active_strategies(self, bot_id: str | None = None) -> list[Strategy]` - Line 240
- `async save_strategy_state(self, strategy_id: str, state_data: dict[str, Any]) -> bool` - Line 264
- `async load_strategy_state(self, strategy_id: str) -> dict[str, Any] | None` - Line 293
- `async save_strategy_metrics(self, strategy_id: str, metrics: StrategyMetrics) -> bool` - Line 327
- `async get_strategy_metrics(self, ...) -> list[AnalyticsStrategyMetrics]` - Line 364
- `async save_strategy_signals(self, signals: list[Signal]) -> list[Signal]` - Line 393
- `async get_strategy_signals(self, strategy_id: str, limit: int | None = None) -> list[Signal]` - Line 414
- `async get_strategy_trades(self, ...) -> list[Trade]` - Line 441
- `async get_strategy_performance_summary(self, strategy_id: str) -> dict[str, Any]` - Line 470

### Implementation: `StrategyService` ✅

**Inherits**: BaseService, StrategyServiceInterface
**Purpose**: Service layer for strategy operations and management
**Status**: Complete

**Implemented Methods:**
- `async register_strategy(self, strategy_id: str, strategy_instance: Any, config: StrategyConfig) -> None` - Line 230
- `async start_strategy(self, strategy_id: str) -> None` - Line 327
- `async stop_strategy(self, strategy_id: str) -> None` - Line 357
- `async process_market_data(self, market_data: MarketData) -> dict[str, list[Signal]]` - Line 381
- `async validate_signal(self, strategy_id: str, signal: Signal) -> bool` - Line 442
- `async validate_strategy_config(self, config: StrategyConfig) -> bool` - Line 484
- `async get_strategy_performance(self, strategy_id: str) -> dict[str, Any]` - Line 717
- `async get_cached_strategy_metrics(self, strategy_id: str) -> dict[str, Any] | None` - Line 761
- `async get_strategy_performance_with_cache(self, strategy_id: str) -> dict[str, Any]` - Line 782
- `async get_all_strategies(self) -> dict[str, dict[str, Any]]` - Line 827
- `async cleanup_strategy(self, strategy_id: str) -> None` - Line 842
- `get_metrics(self) -> dict[str, Any]` - Line 906
- `resolve_dependency(self, dependency_name: str) -> Any` - Line 936

### Implementation: `StrategyIntegratedBase` 🔧

**Inherits**: ABC
**Purpose**: Enhanced base class that provides comprehensive module integration
**Status**: Abstract Base Class

**Implemented Methods:**
- `async initialize_validation_service(self) -> None` - Line 122
- `set_monitoring_services(self, ...)` - Line 154
- `async validate_market_data_comprehensive(self, data: MarketData) -> tuple[bool, list[str]]` - Line 182
- `async calculate_technical_indicators(self, data: MarketData, indicators: list[str], periods: dict[str, int] = None) -> dict[str, Decimal | None]` - Line 249
- `format_signal_metadata(self, signal: Signal, additional_data: dict[str, Any] = None) -> dict[str, Any]` - Line 351
- `async record_signal_metrics(self, ...)` - Line 392
- `async safe_execute_with_monitoring(self, operation_name: str, operation_func, *args, **kwargs) -> Any` - Line 438
- `get_comprehensive_status(self) -> dict[str, Any]` - Line 513
- `async cleanup_resources(self)` - Line 529

### Implementation: `StrategyDataAccessMixin` ✅

**Purpose**: Mixin providing data access patterns for strategies
**Status**: Complete

**Implemented Methods:**
- `async get_indicator_data(self, symbol: str, indicator: str, period: int) -> Decimal | None` - Line 566

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
- `get_strategy_info(self) -> dict[str, Any]` - Line 855

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
- `async update_inventory(self, position: Position) -> None` - Line 85
- `async should_rebalance(self) -> bool` - Line 113
- `async calculate_rebalance_orders(self, current_price: Decimal) -> list[OrderRequest]` - Line 160
- `async should_emergency_liquidate(self) -> bool` - Line 235
- `async calculate_emergency_orders(self, current_price: Decimal) -> list[OrderRequest]` - Line 267
- `async calculate_spread_adjustment(self, base_spread: Decimal) -> Decimal` - Line 323
- `async calculate_size_adjustment(self, base_size: Decimal) -> Decimal` - Line 359
- `async record_rebalance(self, cost: Decimal) -> None` - Line 395
- `async record_emergency(self, cost: Decimal) -> None` - Line 418
- `get_inventory_summary(self) -> dict[str, Any]` - Line 439
- `async validate_inventory_limits(self, new_position: Position) -> bool` - Line 468

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
- `get_strategy_info(self) -> dict[str, Any]` - Line 731

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
- `strategy_type(self) -> StrategyType` - Line 99
- `async validate_signal(self, signal: Signal) -> bool` - Line 441
- `get_position_size(self, signal: Signal) -> Decimal` - Line 497
- `async should_exit(self, position: Position, data: MarketData) -> bool` - Line 553
- `get_strategy_info(self) -> dict[str, Any]` - Line 677
- `get_strategy_info(self) -> dict[str, Any]` - Line 746

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
- `async validate(self, target: Any, context: dict[str, Any] | None = None) -> ValidationResult` - Line 441

### Implementation: `ValidationFramework` ✅

**Purpose**: Main validation framework for strategies
**Status**: Complete

**Implemented Methods:**
- `async validate_signal(self, signal: Signal, market_data: MarketData | None = None) -> ValidationResult` - Line 498
- `async validate_strategy_config(self, config: StrategyConfig) -> ValidationResult` - Line 517
- `async validate_market_conditions(self, market_data: MarketData) -> ValidationResult` - Line 529
- `async validate_for_trading(self, signal: Signal, market_data: MarketData) -> ValidationResult` - Line 541
- `async batch_validate_signals(self, signals: list[Signal], market_data: MarketData | None = None) -> list[tuple[Signal, ValidationResult]]` - Line 557
- `add_custom_validator(self, validator: BaseValidator, validator_type: str = 'custom') -> None` - Line 579
- `get_validation_stats(self) -> dict[str, Any]` - Line 599

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
    def __init__(self, config: dict[str, Any], services: StrategyServiceContainer | None = None)  # Line 94
    def strategy_type(self) -> StrategyType  # Line 157
    def name(self) -> str  # Line 162
    def version(self) -> str  # Line 167
    def status(self) -> StrategyStatus  # Line 172
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]  # Line 177
    async def generate_signals(self, data: MarketData) -> list[Signal]  # Line 194
    async def validate_signal(self, signal: Signal) -> bool  # Line 329
    async def _validate_and_process_signal(self, signal: Signal, market_data: MarketData) -> bool  # Line 342
    def get_position_size(self, signal: Signal) -> Decimal  # Line 385
    def _get_account_balance(self) -> Decimal  # Line 423
    def should_exit(self, position: Position, data: MarketData) -> bool  # Line 436
    async def pre_trade_validation(self, signal: Signal) -> bool  # Line 452
    async def post_trade_processing(self, trade_result: Any) -> None  # Line 478
    def set_risk_manager(self, risk_manager: Any) -> None  # Line 524
    def set_exchange(self, exchange: Any) -> None  # Line 532
    def set_data_service(self, data_service: Any) -> None  # Line 541
    def set_validation_framework(self, validation_framework: ValidationFramework) -> None  # Line 549
    def set_metrics_collector(self, metrics_collector: MetricsCollector) -> None  # Line 558
    def get_strategy_info(self) -> dict[str, Any]  # Line 567
    async def initialize(self, config: StrategyConfig) -> None  # Line 582
    async def start(self) -> bool  # Line 594
    async def stop(self) -> bool  # Line 644
    async def pause(self) -> None  # Line 670
    async def resume(self) -> None  # Line 676
    async def prepare_for_backtest(self, config: dict[str, Any]) -> None  # Line 683
    async def process_historical_data(self, data: MarketData) -> list[Signal]  # Line 699
    async def get_backtest_metrics(self) -> dict[str, Any]  # Line 715
    def get_real_time_metrics(self) -> dict[str, Any]  # Line 728
    def _add_to_signal_history(self, signals: list[Signal]) -> None  # Line 745
    async def _on_initialize(self) -> None  # Line 758
    async def _on_start(self) -> None  # Line 762
    async def _on_stop(self) -> None  # Line 766
    async def _on_backtest_prepare(self) -> None  # Line 770
    def update_config(self, new_config: dict[str, Any]) -> None  # Line 774
    async def get_state(self) -> dict[str, Any]  # Line 789
    def get_performance_summary(self) -> dict[str, Any]  # Line 806
    def cleanup(self) -> None  # Line 836
    async def get_market_data(self, symbol: str) -> MarketData | None  # Line 870
    async def get_historical_data(self, symbol: str, timeframe: str, limit: int = 100) -> list[MarketData]  # Line 892
    async def execute_order(self, signal: Signal) -> Any | None  # Line 918
    async def save_state(self, state_data: dict[str, Any]) -> bool  # Line 964
    async def load_state(self) -> dict[str, Any] | None  # Line 987
    def _update_metrics(self, metrics: dict[str, Any]) -> None  # Line 1010
    def _log_signal(self, signal: Signal) -> None  # Line 1023
    async def _handle_error(self, error: Exception, severity: ErrorSeverity, context: dict[str, Any]) -> None  # Line 1033
    def get_metrics(self) -> dict[str, Any]  # Line 1064
    def is_healthy(self) -> bool  # Line 1080
    async def reset(self) -> bool  # Line 1104
    def _calculate_win_rate(self) -> float  # Line 1113
    def _calculate_sharpe_ratio(self) -> float  # Line 1127
    def _calculate_max_drawdown(self) -> float  # Line 1146
    def set_execution_service(self, execution_service: Any) -> None  # Line 1172
    def get_status(self) -> StrategyStatus  # Line 1178
    def get_status_string(self) -> str  # Line 1182
    async def _cleanup_resources(self) -> None  # Line 1186
    def _validate_config(config: dict[str, Any]) -> None  # Line 1195
    async def validate_market_data(self, data: MarketData | None) -> None  # Line 1219
    def __str__(self) -> str  # Line 1241
    def __repr__(self) -> str  # Line 1245
    async def get_sma(self, symbol: str, period: int) -> Decimal | None  # Line 1250
    async def get_ema(self, symbol: str, period: int) -> Decimal | None  # Line 1271
    async def get_rsi(self, symbol: str, period: int = 14) -> Decimal | None  # Line 1290
    async def get_volatility(self, symbol: str, period: int) -> Decimal | None  # Line 1309
    async def get_atr(self, symbol: str, period: int) -> Decimal | None  # Line 1328
    async def get_volume_ratio(self, symbol: str, period: int) -> Decimal | None  # Line 1347
    async def get_bollinger_bands(self, symbol: str, period: int = 20, std_dev: float = 2.0) -> dict[str, Decimal] | None  # Line 1366
    async def get_macd(self, ...) -> dict[str, Decimal] | None  # Line 1393
    async def execute_with_algorithm(self, ...) -> dict[str, Any] | None  # Line 1422
    async def optimize_parameters(self, optimization_config: dict[str, Any] | None) -> dict[str, Any]  # Line 1470
    async def enhance_signals_with_ml(self, signals: list[Signal]) -> list[Signal]  # Line 1544
    async def _get_market_context_for_ml(self) -> dict[str, Any]  # Line 1607
    async def get_allocated_capital(self) -> Decimal  # Line 1650
    async def execute_large_order(self, order_request: OrderRequest, max_position_size: Decimal | None = None) -> dict[str, Any] | None  # Line 1755
    async def get_execution_algorithms_status(self) -> dict[str, Any]  # Line 1817
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
    def delete_strategy_config(self, strategy_name: str) -> bool  # Line 486
    def get_config_summary(self) -> dict[str, Any]  # Line 529
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
    def get_trend_following_config(timeframe: str = '1h', trend_strength: str = 'medium') -> dict[str, Any]  # Line 263
    def get_market_making_config(symbol, ...) -> dict[str, Any]  # Line 380
    def get_volatility_breakout_config(volatility_regime: str = 'medium', breakout_type: str = 'range') -> dict[str, Any]  # Line 496
    def get_ensemble_config(strategy_types, ...) -> dict[str, Any]  # Line 602
    def get_all_templates(cls) -> dict[str, dict[str, Any]]  # Line 685
    def get_template_by_name(cls, template_name: str) -> dict[str, Any]  # Line 730
    def list_available_templates(cls) -> list[str]  # Line 751
    def get_templates_by_strategy_type(cls, strategy_type: str) -> dict[str, dict[str, Any]]  # Line 761
    def validate_template(cls, template: dict[str, Any]) -> tuple[bool, list[str]]  # Line 779
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
- `from src.strategies.repository import StrategyRepository`
- `from src.strategies.service import StrategyService`

#### Functions:

```python
def register_strategies_dependencies(container: DependencyContainer) -> None  # Line 13
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
    def __init__(self, config: dict[str, Any], services: StrategyServiceContainer | None = None)  # Line 94
    def strategy_type(self) -> StrategyType  # Line 145
    def set_technical_indicators(self, technical_indicators: TechnicalIndicators) -> None  # Line 149
    def set_strategy_service(self, strategy_service: 'StrategyService') -> None  # Line 154
    def set_regime_detector(self, regime_detector: MarketRegimeDetector) -> None  # Line 159
    def set_adaptive_risk_manager(self, adaptive_risk_manager: AdaptiveRiskManager) -> None  # Line 164
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]  # Line 170
    async def _validate_data_availability(self, symbol: str) -> bool  # Line 248
    async def _get_current_regime_via_service(self, symbol: str) -> MarketRegime | None  # Line 269
    async def _calculate_momentum_indicators_via_service(self, symbol: str, current_data: MarketData) -> dict[str, Any] | None  # Line 318
    def _calculate_rsi_score_from_value(self, rsi: float) -> float  # Line 410
    async def _generate_momentum_signals_enhanced(self, ...) -> list[Signal]  # Line 419
    def _calculate_enhanced_confidence(self, ...) -> float  # Line 494
    async def _apply_enhanced_confidence_adjustments(self, signals: list[Signal], current_regime: MarketRegime | None) -> list[Signal]  # Line 527
    async def _update_strategy_state(self, ...) -> None  # Line 587
    async def _persist_strategy_state(self) -> None  # Line 647
    def _get_regime_confidence_multiplier(self, regime: MarketRegime | None) -> float  # Line 661
    async def validate_signal(self, signal: Signal) -> bool  # Line 681
    def get_position_size(self, signal: Signal) -> Decimal  # Line 762
    def should_exit(self, position: Position, data: MarketData) -> bool  # Line 855
    def get_strategy_info(self) -> dict[str, Any]  # Line 936
    async def _on_start(self) -> None  # Line 978
    async def _on_stop(self) -> None  # Line 1000
    async def _get_sma(self, symbol: str, period: int) -> Decimal | None  # Line 1019
    async def _get_rsi(self, symbol: str, period: int) -> Decimal | None  # Line 1030
    async def _get_volatility(self, symbol: str, period: int) -> Decimal | None  # Line 1041
    async def _get_atr(self, symbol: str, period: int) -> Decimal | None  # Line 1052
    async def _get_volume_ratio(self, symbol: str, period: int) -> Decimal | None  # Line 1063
    def cleanup(self) -> None  # Line 1074
```

### File: strategy_factory.py

**Key Imports:**
- `from src.core.base.component import BaseComponent`
- `from src.core.types import StrategyType`
- `from src.data.features.technical_indicators import TechnicalIndicators`
- `from src.risk_management.adaptive_risk import AdaptiveRiskManager`
- `from src.risk_management.regime_detection import MarketRegimeDetector`

#### Class: `EnhancedDynamicStrategyFactory`

**Inherits**: BaseComponent
**Purpose**: Enhanced factory for creating dynamic strategies with service layer integration

```python
class EnhancedDynamicStrategyFactory(BaseComponent):
    def __init__(self, ...)  # Line 43
    async def create_strategy(self, strategy_name: str, config: dict[str, Any], use_enhanced: bool = True) -> BaseStrategy | None  # Line 94
    def _resolve_strategy_class(self, strategy_name: str, use_enhanced: bool)  # Line 161
    async def _enhance_configuration(self, strategy_name: str, config: dict[str, Any]) -> dict[str, Any]  # Line 189
    async def _inject_dependencies(self, strategy: BaseStrategy, strategy_name: str) -> None  # Line 251
    async def _validate_strategy_setup(self, strategy: BaseStrategy) -> bool  # Line 293
    def get_available_strategies(self) -> dict[str, str]  # Line 346
    def get_strategy_requirements(self, strategy_name: str) -> dict[str, Any]  # Line 355
    async def create_multiple_strategies(self, strategy_configs: dict[str, dict[str, Any]], use_enhanced: bool = True) -> dict[str, BaseStrategy | None]  # Line 393
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
    def name(self) -> str  # Line 58
    def name(self, value: str) -> None  # Line 67
    def version(self) -> str  # Line 72
    def version(self, value: str) -> None  # Line 77
    def status(self) -> str  # Line 82
    def status(self, value: str) -> None  # Line 87
    def __init__(self, config: dict[str, Any], services: StrategyServiceContainer | None = None)  # Line 91
    def strategy_type(self) -> StrategyType  # Line 147
    def set_technical_indicators(self, technical_indicators: TechnicalIndicators) -> None  # Line 151
    def set_strategy_service(self, strategy_service: StrategyService) -> None  # Line 156
    def set_regime_detector(self, regime_detector: MarketRegimeDetector) -> None  # Line 161
    def set_adaptive_risk_manager(self, adaptive_risk_manager: AdaptiveRiskManager) -> None  # Line 166
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]  # Line 172
    async def _validate_data_availability(self, symbol: str) -> bool  # Line 268
    async def _is_in_breakout_cooldown(self, symbol: str) -> bool  # Line 288
    async def _get_current_regime_via_service(self, symbol: str) -> MarketRegime | None  # Line 310
    async def _calculate_volatility_indicators_via_service(self, symbol: str, current_data: MarketData) -> dict[str, Any] | None  # Line 356
    async def _calculate_consolidation_score_enhanced(self, symbol: str, price_data: list[MarketData]) -> float  # Line 443
    def _is_bollinger_squeeze(self, bb_data: dict, current_price: Decimal) -> bool  # Line 508
    def _get_bb_position(self, bb_data: dict, current_price: Decimal) -> str  # Line 527
    async def _calculate_breakout_levels_enhanced(self, ...) -> dict[str, float]  # Line 551
    def _get_regime_breakout_adjustment(self, regime: MarketRegime | None) -> float  # Line 612
    async def _generate_breakout_signals_enhanced(self, ...) -> list[Signal]  # Line 631
    def _calculate_enhanced_breakout_confidence(self, ...) -> float  # Line 727
    def _get_regime_confidence_multiplier(self, regime: MarketRegime | None) -> float  # Line 779
    async def _apply_enhanced_regime_filtering(self, signals: list[Signal], current_regime: MarketRegime | None) -> list[Signal]  # Line 798
    def _is_signal_valid_for_regime_enhanced(self, signal: Signal, regime: MarketRegime) -> bool  # Line 858
    async def _apply_enhanced_time_decay(self, signals: list[Signal], symbol: str) -> list[Signal]  # Line 906
    async def _update_strategy_state(self, ...) -> None  # Line 984
    async def _persist_strategy_state(self) -> None  # Line 1037
    async def validate_signal(self, signal: Signal) -> bool  # Line 1052
    def get_position_size(self, signal: Signal) -> Decimal  # Line 1167
    def _get_regime_position_adjustment(self, regime: MarketRegime) -> float  # Line 1264
    def should_exit(self, position: Position, data: MarketData) -> bool  # Line 1280
    def get_strategy_info(self) -> dict[str, Any]  # Line 1359
    async def _on_start(self) -> None  # Line 1405
    async def _on_stop(self) -> None  # Line 1427
    def cleanup(self) -> None  # Line 1445
    async def _get_sma(self, symbol: str, period: int) -> Decimal | None  # Line 1450
    async def _get_rsi(self, symbol: str, period: int) -> Decimal | None  # Line 1461
    async def _get_volatility(self, symbol: str, period: int) -> Decimal | None  # Line 1472
    async def _get_atr(self, symbol: str, period: int) -> Decimal | None  # Line 1483
    async def _get_volume_ratio(self, symbol: str, period: int) -> Decimal | None  # Line 1494
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
    def __init__(self, *args, **kwargs)  # Line 98
    async def _update_service_environment(self, context: EnvironmentContext) -> None  # Line 105
    def get_environment_strategy_config(self, exchange: str) -> dict[str, Any]  # Line 140
    async def deploy_environment_aware_strategy(self, strategy_config: StrategyConfig, exchange: str, force_deploy: bool = False) -> bool  # Line 153
    async def validate_strategy_for_environment(self, strategy_config: StrategyConfig, exchange: str) -> bool  # Line 220
    async def _validate_production_strategy(self, strategy_config: StrategyConfig, env_config: dict[str, Any]) -> bool  # Line 240
    async def _validate_sandbox_strategy(self, strategy_config: StrategyConfig, env_config: dict[str, Any]) -> bool  # Line 278
    async def _validate_common_strategy_rules(self, strategy_config: StrategyConfig, exchange: str, env_config: dict[str, Any]) -> bool  # Line 299
    async def _apply_environment_strategy_adjustments(self, strategy_config: StrategyConfig, exchange: str) -> StrategyConfig  # Line 326
    async def generate_environment_aware_signal(self, strategy_name: str, market_data: MarketData, exchange: str) -> Signal | None  # Line 362
    async def _apply_environment_signal_filters(self, signal: Signal, exchange: str, env_config: dict[str, Any]) -> Signal | None  # Line 394
    async def _validate_signal_for_environment(self, signal: Signal, exchange: str) -> bool  # Line 422
    async def _deploy_strategy_with_config(self, strategy_config: StrategyConfig, exchange: str) -> bool  # Line 446
    async def _generate_base_signal(self, strategy_name: str, market_data: MarketData, exchange: str) -> Signal | None  # Line 454
    async def _verify_strategy_backtest(self, strategy_config: StrategyConfig, exchange: str) -> bool  # Line 469
    async def _initialize_strategy_tracking(self, strategy_name: str, exchange: str) -> None  # Line 499
    async def _update_signal_tracking(self, strategy_name: str, exchange: str, signal: Signal) -> None  # Line 508
    async def _disable_experimental_strategies(self, exchange: str) -> None  # Line 519
    async def _is_high_volatility_period(self, symbol: str, exchange: str) -> bool  # Line 530
    async def update_strategy_performance(self, ...) -> None  # Line 535
    def get_environment_strategy_metrics(self, exchange: str) -> dict[str, Any]  # Line 567
    async def rebalance_strategies_for_environment(self, exchange: str) -> dict[str, Any]  # Line 601
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
    def __init__(self, species_id: int, representative: NEATGenome)  # Line 899
    def add_member(self, genome: NEATGenome) -> None  # Line 914
    def calculate_average_fitness(self) -> None  # Line 919
    def select_parents(self, selection_pressure: float = 0.5) -> list[NEATGenome]  # Line 926
    def remove_worst_genomes(self, keep_ratio: float = 0.5) -> None  # Line 945
```

#### Class: `SpeciationManager`

**Purpose**: Manages species formation and evolution in NEAT population

```python
class SpeciationManager:
    def __init__(self, ...)  # Line 965
    def speciate_population(self, population: list[NEATGenome]) -> None  # Line 984
    def _calculate_fitness_sharing(self) -> None  # Line 1029
    def _remove_stagnant_species(self) -> None  # Line 1036
    def _adjust_compatibility_threshold(self) -> None  # Line 1054
    def allocate_offspring(self, total_offspring: int) -> dict[int, int]  # Line 1068
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
    def __init__(self, config: dict[str, Any], services: StrategyServiceContainer | None = None)  # Line 1162
    def _initialize_population(self) -> None  # Line 1202
    def _update_best_network(self) -> None  # Line 1229
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]  # Line 1237
    async def _maybe_adapt(self) -> None  # Line 1279
    async def adapt_networks(self) -> None  # Line 1290
    async def _micro_evolution(self) -> None  # Line 1338
    async def evolve_population(self, fitness_evaluator: FitnessEvaluator) -> None  # Line 1371
    def _create_offspring(self, species: Species, parents: list[NEATGenome]) -> NEATGenome  # Line 1447
    def _mutate_genome(self, genome: NEATGenome) -> None  # Line 1475
    def _record_generation_stats(self) -> None  # Line 1502
    async def validate_signal(self, signal: Signal) -> bool  # Line 1523
    def get_position_size(self, signal: Signal) -> Decimal  # Line 1547
    def should_exit(self, position: Position, data: MarketData) -> bool  # Line 1571
    def _check_standard_exits(self, position: Position, data: MarketData) -> bool  # Line 1601
    def get_strategy_info(self) -> dict[str, Any]  # Line 1625
    def get_evolution_summary(self) -> dict[str, Any]  # Line 1655
    async def save_population(self, filepath: str) -> None  # Line 1684
    async def load_population(self, filepath: str) -> None  # Line 1715
    async def _get_sma(self, symbol: str, period: int) -> Decimal | None  # Line 1754
    async def _get_rsi(self, symbol: str, period: int) -> Decimal | None  # Line 1765
    async def _get_volatility(self, symbol: str, period: int) -> Decimal | None  # Line 1776
    async def _get_atr(self, symbol: str, period: int) -> Decimal | None  # Line 1787
    async def _get_volume_ratio(self, symbol: str, period: int) -> Decimal | None  # Line 1798
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
    def __post_init__(self)  # Line 109
```

#### Class: `ConstraintHandler`

**Purpose**: Handles constraints in multi-objective optimization

```python
class ConstraintHandler:
    def __init__(self, constraints: list[OptimizationObjective])  # Line 122
    def evaluate_constraints(self, objectives: dict[str, float]) -> dict[str, float]  # Line 139
    def is_feasible(self, objectives: dict[str, float], tolerance: float = 0.01) -> bool  # Line 172
    def apply_penalty(self, objectives: dict[str, float], constraint_violations: dict[str, float]) -> dict[str, float]  # Line 187
```

#### Class: `DominanceComparator`

**Purpose**: Implements dominance comparison for multi-objective optimization

```python
class DominanceComparator:
    def __init__(self, objectives: list[OptimizationObjective])  # Line 232
    def dominates(self, solution1: dict[str, float], solution2: dict[str, float]) -> bool  # Line 244
    def non_dominated_sort(self, solutions: list[dict[str, float]]) -> list[list[int]]  # Line 284
```

#### Class: `CrowdingDistanceCalculator`

**Purpose**: Calculates crowding distance for diversity preservation

```python
class CrowdingDistanceCalculator:
    def __init__(self, objectives: list[OptimizationObjective])  # Line 345
    def calculate_crowding_distance(self, solutions: list[dict[str, float]], front_indices: list[int]) -> list[float]  # Line 357
```

#### Class: `ParetoFrontierManager`

**Purpose**: Manages the Pareto frontier and provides analysis tools

```python
class ParetoFrontierManager:
    def __init__(self, config: MultiObjectiveConfig)  # Line 416
    def update_frontier(self, solutions: list[ParetoSolution]) -> None  # Line 439
    def _calculate_convergence_metrics(self) -> None  # Line 488
    def _calculate_hypervolume(self, solutions: list[ParetoSolution]) -> float  # Line 506
    def _calculate_spread(self, solutions: list[ParetoSolution]) -> float  # Line 543
    def _calculate_convergence(self, current: list[ParetoSolution], previous: list[ParetoSolution]) -> float  # Line 566
    def _solution_distance(self, sol1: ParetoSolution, sol2: ParetoSolution) -> float  # Line 599
    def get_frontier_summary(self) -> dict[str, Any]  # Line 621
```

#### Class: `NSGAIIOptimizer`

**Purpose**: NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation

```python
class NSGAIIOptimizer:
    def __init__(self, ...)  # Line 676
    async def optimize(self) -> list[ParetoSolution]  # Line 716
    async def _initialize_population(self) -> Population  # Line 785
    async def _evaluate_population(self, population: Population) -> list[ParetoSolution]  # Line 812
    async def _evaluate_individual(self, individual: Individual) -> ParetoSolution | None  # Line 853
    async def _simulate_objectives(self, individual: Individual) -> dict[str, float]  # Line 893
    async def _create_offspring(self) -> Population  # Line 934
    def _tournament_selection(self) -> ParetoSolution  # Line 970
    def _create_random_solution(self) -> ParetoSolution  # Line 988
    def _crossover(self, genes1: dict[str, Any], genes2: dict[str, Any]) -> dict[str, Any]  # Line 1012
    def _mutate(self, genes: dict[str, Any]) -> dict[str, Any]  # Line 1037
    def _environmental_selection(self, solutions: list[ParetoSolution]) -> list[ParetoSolution]  # Line 1069
    def _check_convergence(self) -> bool  # Line 1122
    def _record_generation_stats(self) -> None  # Line 1146
    def get_optimization_summary(self) -> dict[str, Any]  # Line 1161
```

#### Class: `MultiObjectiveOptimizer`

**Purpose**: Main interface for multi-objective optimization of trading strategies

```python
class MultiObjectiveOptimizer:
    def __init__(self, config: MultiObjectiveConfig)  # Line 1198
    async def optimize_strategy(self, ...) -> list[ParetoSolution]  # Line 1222
    def get_pareto_frontier_data(self) -> dict[str, Any]  # Line 1274
    def export_results(self, filepath: str) -> None  # Line 1313
```

#### Functions:

```python
def create_trading_objectives() -> list[OptimizationObjective]  # Line 1341
def create_default_config(objectives: list[OptimizationObjective] | None = None) -> MultiObjectiveConfig  # Line 1380
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
    def __init__(self, ...)  # Line 64
    def _register_builtin_strategies(self) -> None  # Line 142
    def _lazy_load_strategy_class(self, strategy_type: StrategyType) -> type | None  # Line 148
    def register_strategy_type(self, strategy_type: StrategyType, strategy_class: type) -> None  # Line 227
    async def create_strategy(self, strategy_type: StrategyType, config: StrategyConfig) -> BaseStrategyInterface  # Line 252
    async def _create_comprehensive_service_container(self, config: StrategyConfig) -> StrategyServiceContainer  # Line 312
    async def _enhance_strategy_with_integrations(self, strategy, config: StrategyConfig) -> None  # Line 388
    def _validate_configuration_parameters(self, config: StrategyConfig) -> bool  # Line 412
    def _get_integration_status(self, strategy) -> dict[str, bool]  # Line 458
    async def _inject_dependencies(self, strategy: BaseStrategyInterface, config: StrategyConfig) -> None  # Line 469
    def get_supported_strategies(self) -> list[StrategyType]  # Line 517
    def validate_strategy_requirements(self, strategy_type: StrategyType, config: StrategyConfig) -> bool  # Line 541
    def _validate_strategy_specific_requirements(self, strategy_type: StrategyType, config: StrategyConfig) -> bool  # Line 574
    def _get_required_parameters(self, strategy_type: StrategyType) -> list[str]  # Line 607
    def _validate_momentum_strategy_config(self, config: StrategyConfig) -> bool  # Line 634
    def _validate_mean_reversion_strategy_config(self, config: StrategyConfig) -> bool  # Line 658
    def _validate_arbitrage_strategy_config(self, config: StrategyConfig) -> bool  # Line 682
    def _validate_volatility_strategy_config(self, config: StrategyConfig) -> bool  # Line 698
    async def create_strategy_with_validation(self, ...) -> BaseStrategyInterface  # Line 724
    def _validate_dependency_availability_sync(self, config: StrategyConfig) -> bool  # Line 766
    async def _validate_created_strategy(self, strategy: BaseStrategyInterface) -> bool  # Line 795
    def get_strategy_info(self, strategy_type: StrategyType) -> dict[str, Any]  # Line 842
    def list_available_strategies(self) -> dict[str, Any]  # Line 867
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
    async def _get_sma(self, symbol: str, period: int) -> Decimal | None  # Line 917
    async def _get_rsi(self, symbol: str, period: int) -> Decimal | None  # Line 928
    async def _get_volatility(self, symbol: str, period: int) -> Decimal | None  # Line 939
    async def _get_atr(self, symbol: str, period: int) -> Decimal | None  # Line 950
    async def _get_volume_ratio(self, symbol: str, period: int) -> Decimal | None  # Line 961
    def get_strategy_info(self) -> dict[str, Any]  # Line 972
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
    async def _get_sma(self, symbol: str, period: int) -> Decimal | None  # Line 981
    async def _get_rsi(self, symbol: str, period: int) -> Decimal | None  # Line 992
    async def _get_volatility(self, symbol: str, period: int) -> Decimal | None  # Line 1003
    async def _get_atr(self, symbol: str, period: int) -> Decimal | None  # Line 1014
    async def _get_volume_ratio(self, symbol: str, period: int) -> Decimal | None  # Line 1025
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
    async def calculate_moving_averages(self, symbol: str, current_price: float) -> tuple[float, float]  # Line 96
    async def evaluate_rules(self, ...) -> dict[str, Any]  # Line 117
    def update_rule_performance(self, rule: str, performance: float) -> None  # Line 202
    def adjust_rule_weights(self) -> None  # Line 211
```

#### Class: `AIPredictor`

**Purpose**: AI prediction component using machine learning

```python
class AIPredictor:
    def __init__(self, config: dict[str, Any], services: StrategyServiceContainer | None = None)  # Line 224
    def prepare_features(self, price_history: list[float], volume_history: list[float]) -> np.ndarray  # Line 243
    async def train_model(self, training_data: list[dict[str, Any]]) -> None  # Line 305
    async def predict(self, price_history: list[float], volume_history: list[float]) -> dict[str, Any]  # Line 353
    def update_performance(self, prediction: dict[str, Any], actual_outcome: float) -> None  # Line 407
    def get_performance_metrics(self) -> dict[str, float]  # Line 417
```

#### Class: `RuleBasedAIStrategy`

**Inherits**: BaseStrategy
**Purpose**: Hybrid strategy combining traditional technical analysis rules with AI predictions

```python
class RuleBasedAIStrategy(BaseStrategy):
    def __init__(self, config: dict[str, Any], services: StrategyServiceContainer | None = None)  # Line 447
    def strategy_type(self) -> StrategyType  # Line 493
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]  # Line 498
    async def _resolve_conflicts(self, ...) -> Signal | None  # Line 572
    def _weighted_average_resolution(self, ...) -> dict[str, Any] | None  # Line 658
    def _highest_confidence_resolution(self, ...) -> dict[str, Any] | None  # Line 692
    def _consensus_resolution(self, ...) -> dict[str, Any] | None  # Line 705
    async def validate_signal(self, signal: Signal) -> bool  # Line 721
    def get_position_size(self, signal: Signal) -> Decimal  # Line 750
    async def should_exit(self, position: Position, data: MarketData) -> bool  # Line 782
    async def _on_start(self) -> None  # Line 837
    async def _retrain_ai_model(self) -> None  # Line 845
    def adjust_component_weights(self) -> None  # Line 882
    def get_strategy_statistics(self) -> dict[str, Any]  # Line 930
    async def _get_sma(self, symbol: str, period: int) -> Decimal | None  # Line 944
    async def _get_rsi(self, symbol: str, period: int) -> Decimal | None  # Line 955
    async def _get_volatility(self, symbol: str, period: int) -> Decimal | None  # Line 966
    async def _get_atr(self, symbol: str, period: int) -> Decimal | None  # Line 977
    async def _get_volume_ratio(self, symbol: str, period: int) -> Decimal | None  # Line 988
    def get_strategy_stats(self) -> dict[str, Any]  # Line 999
    async def _get_ml_service_prediction(self, symbol: str, data: MarketData) -> dict[str, Any] | None  # Line 1014
    def _combine_predictions(self, local_prediction: dict[str, Any], ml_prediction: dict[str, Any]) -> dict[str, Any]  # Line 1066
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
    def __init__(self, strategy_name: str)  # Line 42
```

#### Class: `PerformanceMonitor`

**Purpose**: Comprehensive performance monitoring system for trading strategies

```python
class PerformanceMonitor:
    def __init__(self, ...)  # Line 130
    async def add_strategy(self, strategy: BaseStrategyInterface) -> None  # Line 177
    async def remove_strategy(self, strategy_name: str) -> None  # Line 219
    async def start_monitoring(self) -> None  # Line 239
    async def stop_monitoring(self) -> None  # Line 247
    async def _monitoring_loop(self) -> None  # Line 263
    async def _update_all_metrics(self) -> None  # Line 322
    async def _update_strategy_metrics(self, strategy_name: str) -> None  # Line 333
    def _update_trade_statistics(self, metrics: PerformanceMetrics, trades: list[Trade]) -> None  # Line 380
    def _calculate_consecutive_trades(self, metrics: PerformanceMetrics, trades: list[Trade]) -> None  # Line 439
    async def _update_pnl_metrics(self, ...) -> None  # Line 478
    def _calculate_risk_ratios(self, metrics: PerformanceMetrics) -> None  # Line 536
    def _update_drawdown_analysis(self, metrics: PerformanceMetrics) -> None  # Line 576
    def _update_time_metrics(self, metrics: PerformanceMetrics, trades: list[Trade]) -> None  # Line 603
    def _update_exposure_metrics(self, ...) -> None  # Line 626
    def _calculate_risk_metrics(self, metrics: PerformanceMetrics) -> None  # Line 659
    async def _check_performance_alerts(self) -> None  # Line 684
    def _update_strategy_rankings(self) -> None  # Line 713
    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float  # Line 729
    async def get_strategy_performance(self, strategy_name: str) -> dict[str, Any]  # Line 766
    async def get_comparative_analysis(self) -> dict[str, Any]  # Line 858
    async def _calculate_portfolio_metrics(self) -> dict[str, Any]  # Line 898
    async def _get_current_positions(self, strategy_name: str) -> list[Position]  # Line 973
    async def _get_recent_trades(self, strategy_name: str, limit: int = 1000, offset: int = 0) -> list[Trade]  # Line 1049
    def _validate_trade_query_params(self, strategy_name: str, limit: int, offset: int) -> None  # Line 1074
    async def _fetch_trade_data(self, strategy_name: str) -> list[dict[str, Any]]  # Line 1085
    def _convert_trade_dicts_to_objects(self, trade_dicts: list[dict[str, Any]]) -> list[Trade]  # Line 1096
    def _create_trade_from_dict(self, trade_dict: dict[str, Any]) -> Trade | None  # Line 1111
    def _map_legacy_trade_fields(self, trade_dict: dict[str, Any]) -> dict[str, Any]  # Line 1178
    async def _get_current_price(self, symbol: str) -> Decimal | None  # Line 1209
    def _calculate_position_pnl(self, position: Position, current_price: Decimal) -> Decimal  # Line 1229
    async def _load_historical_performance(self, strategy_name: str) -> None  # Line 1238
    async def _save_performance_metrics(self, strategy_name: str) -> None  # Line 1260
    async def _persist_metrics(self) -> None  # Line 1293
    async def _send_performance_alerts(self, strategy_name: str, alerts: list[str]) -> None  # Line 1311
```

### File: portfolio_allocator.py

**Key Imports:**
- `from src.core.exceptions import AllocationError`
- `from src.core.types import MarketRegime`
- `from src.core.types import Signal`
- `from src.core.types import SignalDirection`
- `from src.core.types import StrategyStatus`

#### Class: `StrategyAllocation`

**Purpose**: Represents allocation for a single strategy

```python
class StrategyAllocation:
    def __init__(self, ...)  # Line 42
```

#### Class: `PortfolioAllocator`

**Purpose**: Dynamic portfolio allocator for trading strategies

```python
class PortfolioAllocator:
    def __init__(self, ...)  # Line 97
    async def add_strategy(self, strategy: BaseStrategyInterface, initial_weight: float = 0.1) -> bool  # Line 163
    async def _validate_strategy(self, strategy: BaseStrategyInterface) -> bool  # Line 217
    async def _calculate_strategy_correlation(self, new_strategy: BaseStrategyInterface) -> float  # Line 265
    async def rebalance_portfolio(self) -> dict[str, Any]  # Line 310
    async def _update_strategy_metrics(self) -> None  # Line 353
    async def _calculate_optimal_weights(self) -> dict[str, float]  # Line 390
    def _build_returns_matrix(self, strategies: list[str]) -> ndarray | None  # Line 440
    def _optimize_sharpe_ratio(self, expected_returns: np.ndarray, cov_matrix: np.ndarray, n_strategies: int) -> np.ndarray  # Line 472
    def _apply_weight_constraints(self, weights: np.ndarray) -> np.ndarray  # Line 525
    def _calculate_performance_based_weights(self) -> dict[str, float]  # Line 536
    def _apply_regime_adjustments(self, weights: dict[str, float]) -> dict[str, float]  # Line 578
    async def _execute_rebalancing(self, target_weights: dict[str, float]) -> list[dict[str, Any]]  # Line 615
    async def _calculate_portfolio_metrics(self) -> dict[str, Any]  # Line 666
    async def update_market_regime(self, new_regime: MarketRegime) -> None  # Line 738
    async def remove_strategy(self, strategy_name: str, reason: str = 'manual') -> bool  # Line 760
    def get_allocation_status(self) -> dict[str, Any]  # Line 804
    async def should_rebalance(self) -> bool  # Line 859
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
    async def update_strategy(self, strategy_id: str, updates: dict[str, Any]) -> Strategy | None  # Line 160
    async def delete_strategy(self, strategy_id: str) -> bool  # Line 201
    async def get_strategies_by_bot(self, bot_id: str) -> list[Strategy]  # Line 217
    async def get_active_strategies(self, bot_id: str | None = None) -> list[Strategy]  # Line 240
    async def save_strategy_state(self, strategy_id: str, state_data: dict[str, Any]) -> bool  # Line 264
    async def load_strategy_state(self, strategy_id: str) -> dict[str, Any] | None  # Line 293
    async def save_strategy_metrics(self, strategy_id: str, metrics: StrategyMetrics) -> bool  # Line 327
    async def get_strategy_metrics(self, ...) -> list[AnalyticsStrategyMetrics]  # Line 364
    async def save_strategy_signals(self, signals: list[Signal]) -> list[Signal]  # Line 393
    async def get_strategy_signals(self, strategy_id: str, limit: int | None = None) -> list[Signal]  # Line 414
    async def get_strategy_trades(self, ...) -> list[Trade]  # Line 441
    async def get_strategy_performance_summary(self, strategy_id: str) -> dict[str, Any]  # Line 470
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
    def __init__(self, ...)  # Line 67
    async def _do_start(self) -> None  # Line 125
    async def _build_strategy_service_container(self) -> StrategyServiceContainer  # Line 150
    async def register_strategy(self, strategy_id: str, strategy_instance: Any, config: StrategyConfig) -> None  # Line 230
    async def _register_strategy_impl(self, strategy_id: str, strategy_instance: Any, config: StrategyConfig) -> None  # Line 252
    async def start_strategy(self, strategy_id: str) -> None  # Line 327
    async def _start_strategy_impl(self, strategy_id: str) -> None  # Line 341
    async def stop_strategy(self, strategy_id: str) -> None  # Line 357
    async def _stop_strategy_impl(self, strategy_id: str) -> None  # Line 368
    async def process_market_data(self, market_data: MarketData) -> dict[str, list[Signal]]  # Line 381
    async def _process_market_data_impl(self, market_data: MarketData) -> dict[str, list[Signal]]  # Line 395
    async def validate_signal(self, strategy_id: str, signal: Signal) -> bool  # Line 442
    async def validate_strategy_config(self, config: StrategyConfig) -> bool  # Line 484
    async def _validate_strategy_specific_config(self, config: StrategyConfig) -> bool  # Line 515
    async def _validate_start_conditions(self, strategy_id: str) -> bool  # Line 520
    async def _update_strategy_metrics(self, strategy_id: str, signals: list[Signal]) -> None  # Line 536
    async def _record_strategy_analytics(self, strategy_id: str, signals: list[Signal]) -> None  # Line 569
    async def _calculate_win_rate(self, strategy_id: str, signal_history: list[Signal]) -> float  # Line 628
    async def _calculate_sharpe_ratio(self, strategy_id: str) -> float  # Line 651
    async def _calculate_max_drawdown(self, strategy_id: str) -> float  # Line 682
    async def get_strategy_performance(self, strategy_id: str) -> dict[str, Any]  # Line 717
    async def _get_strategy_performance_impl(self, strategy_id: str) -> dict[str, Any]  # Line 731
    async def get_cached_strategy_metrics(self, strategy_id: str) -> dict[str, Any] | None  # Line 761
    async def get_strategy_performance_with_cache(self, strategy_id: str) -> dict[str, Any]  # Line 782
    async def get_all_strategies(self) -> dict[str, dict[str, Any]]  # Line 827
    async def cleanup_strategy(self, strategy_id: str) -> None  # Line 842
    async def _cleanup_strategy_impl(self, strategy_id: str) -> None  # Line 853
    async def _service_health_check(self) -> Any  # Line 875
    def get_metrics(self) -> dict[str, Any]  # Line 906
    def resolve_dependency(self, dependency_name: str) -> Any  # Line 936
```

#### Functions:

```python
def cache_strategy_signals(strategy_id_arg_name: str, ttl: int = DEFAULT_CACHE_TTL) -> Callable  # Line 45
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
    def __init__(self, ...)  # Line 72
    async def initialize_validation_service(self) -> None  # Line 122
    def set_monitoring_services(self, ...)  # Line 154
    def _get_available_integrations(self) -> dict[str, bool]  # Line 168
    async def validate_market_data_comprehensive(self, data: MarketData) -> tuple[bool, list[str]]  # Line 182
    async def calculate_technical_indicators(self, data: MarketData, indicators: list[str], periods: dict[str, int] = None) -> dict[str, Decimal | None]  # Line 249
    def format_signal_metadata(self, signal: Signal, additional_data: dict[str, Any] = None) -> dict[str, Any]  # Line 351
    async def record_signal_metrics(self, ...)  # Line 392
    async def safe_execute_with_monitoring(self, operation_name: str, operation_func, *args, **kwargs) -> Any  # Line 438
    def get_comprehensive_status(self) -> dict[str, Any]  # Line 513
    async def cleanup_resources(self)  # Line 529
```

#### Class: `StrategyDataAccessMixin`

**Purpose**: Mixin providing data access patterns for strategies

```python
class StrategyDataAccessMixin:
    def __init__(self, data_service: DataServiceInterface | None = None)  # Line 562
    async def get_indicator_data(self, symbol: str, indicator: str, period: int) -> Decimal | None  # Line 566
```

#### Functions:

```python
def create_comprehensive_signal(...) -> Signal  # Line 602
async def calculate_position_size_comprehensive(...) -> Decimal  # Line 642
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
    async def _check_triangular_path(self, path: list[str]) -> Signal | None  # Line 743
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
    def __init__(self, config: dict[str, Any], services: StrategyServiceContainer | None = None)  # Line 56
    def strategy_type(self) -> StrategyType  # Line 106
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]  # Line 115
    async def _update_support_resistance_levels(self, data: MarketData) -> None  # Line 196
    async def _check_consolidation_period(self, data: MarketData) -> bool  # Line 228
    async def _check_resistance_breakout(self, data: MarketData) -> dict[str, Any] | None  # Line 263
    async def _check_support_breakout(self, data: MarketData) -> dict[str, Any] | None  # Line 315
    async def _check_volume_confirmation(self, data: MarketData) -> bool  # Line 347
    def _check_false_breakout(self, data: MarketData) -> dict[str, Any] | None  # Line 373
    async def _generate_bullish_breakout_signal(self, data: MarketData, breakout_info: dict[str, Any]) -> Signal | None  # Line 412
    async def _generate_bearish_breakout_signal(self, data: MarketData, breakout_info: dict[str, Any]) -> Signal | None  # Line 483
    async def _generate_false_breakout_exit_signal(self, data: MarketData, false_breakout_info: dict[str, Any]) -> Signal | None  # Line 556
    async def validate_signal(self, signal: Signal) -> bool  # Line 608
    def get_position_size(self, signal: Signal) -> Decimal  # Line 671
    async def should_exit(self, position: Position, data: MarketData) -> bool  # Line 722
    async def _get_sma(self, symbol: str, period: int) -> Decimal | None  # Line 800
    async def _get_rsi(self, symbol: str, period: int) -> Decimal | None  # Line 811
    async def _get_volatility(self, symbol: str, period: int) -> Decimal | None  # Line 822
    async def _get_atr(self, symbol: str, period: int) -> Decimal | None  # Line 833
    async def _get_volume_ratio(self, symbol: str, period: int) -> Decimal | None  # Line 844
    def get_strategy_info(self) -> dict[str, Any]  # Line 855
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
    async def _get_sma(self, symbol: str, period: int) -> Decimal | None  # Line 662
    async def _get_rsi(self, symbol: str, period: int) -> Decimal | None  # Line 673
    async def _get_volatility(self, symbol: str, period: int) -> Decimal | None  # Line 684
    async def _get_atr(self, symbol: str, period: int) -> Decimal | None  # Line 695
    async def _get_volume_ratio(self, symbol: str, period: int) -> Decimal | None  # Line 706
    async def _process_trade_result(self, trade_result: dict[str, Any]) -> None  # Line 717
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
    def __init__(self, config: dict[str, Any], services: StrategyServiceContainer | None = None)  # Line 41
    async def update_inventory(self, position: Position) -> None  # Line 85
    async def should_rebalance(self) -> bool  # Line 113
    async def calculate_rebalance_orders(self, current_price: Decimal) -> list[OrderRequest]  # Line 160
    async def should_emergency_liquidate(self) -> bool  # Line 235
    async def calculate_emergency_orders(self, current_price: Decimal) -> list[OrderRequest]  # Line 267
    async def calculate_spread_adjustment(self, base_spread: Decimal) -> Decimal  # Line 323
    async def calculate_size_adjustment(self, base_size: Decimal) -> Decimal  # Line 359
    async def record_rebalance(self, cost: Decimal) -> None  # Line 395
    async def record_emergency(self, cost: Decimal) -> None  # Line 418
    def get_inventory_summary(self) -> dict[str, Any]  # Line 439
    async def validate_inventory_limits(self, new_position: Position) -> bool  # Line 468
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
    def __init__(self, config: dict[str, Any], services: StrategyServiceContainer | None = None)  # Line 85
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
    async def _get_sma(self, symbol: str, period: int) -> Decimal | None  # Line 676
    async def _get_rsi(self, symbol: str, period: int) -> Decimal | None  # Line 687
    async def _get_volatility(self, symbol: str, period: int) -> Decimal | None  # Line 698
    async def _get_atr(self, symbol: str, period: int) -> Decimal | None  # Line 709
    async def _get_volume_ratio(self, symbol: str, period: int) -> Decimal | None  # Line 720
    def get_strategy_info(self) -> dict[str, Any]  # Line 731
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
    def __init__(self, config: dict[str, Any], services: StrategyServiceContainer | None = None)  # Line 63
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
    def __init__(self, config: dict[str, Any], services: StrategyServiceContainer | None = None)  # Line 40
    async def optimize_spread(self, ...) -> Decimal  # Line 93
    async def _calculate_volatility_adjustment(self, base_spread: Decimal) -> Decimal  # Line 167
    async def _calculate_imbalance_adjustment(self, base_spread: Decimal, order_book: OrderBook | None) -> Decimal  # Line 212
    async def _calculate_competitor_adjustment(self, base_spread: Decimal, competitor_spreads: list[float] | None) -> Decimal  # Line 265
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
    def __init__(self, config: dict[str, Any], services: StrategyServiceContainer | None = None)  # Line 59
    def strategy_type(self) -> StrategyType  # Line 99
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]  # Line 107
    async def _check_volume_confirmation(self, data: MarketData) -> bool  # Line 224
    async def _generate_bullish_signal(self, data: MarketData, fast_ma: float, slow_ma: float, rsi: float) -> Signal | None  # Line 254
    async def _generate_bearish_signal(self, data: MarketData, fast_ma: float, slow_ma: float, rsi: float) -> Signal | None  # Line 325
    async def _generate_exit_signal(self, data: MarketData, direction: SignalDirection, reason: str) -> Signal | None  # Line 396
    async def validate_signal(self, signal: Signal) -> bool  # Line 441
    def get_position_size(self, signal: Signal) -> Decimal  # Line 497
    async def should_exit(self, position: Position, data: MarketData) -> bool  # Line 553
    def _should_exit_by_time(self, position: Position) -> bool  # Line 623
    def _should_exit_by_trailing_stop(self, position: Position, data: MarketData) -> bool  # Line 646
    def get_strategy_info(self) -> dict[str, Any]  # Line 677
    async def _get_sma(self, symbol: str, period: int) -> Decimal | None  # Line 691
    async def _get_rsi(self, symbol: str, period: int) -> Decimal | None  # Line 702
    async def _get_volatility(self, symbol: str, period: int) -> Decimal | None  # Line 713
    async def _get_atr(self, symbol: str, period: int) -> Decimal | None  # Line 724
    async def _get_volume_ratio(self, symbol: str, period: int) -> Decimal | None  # Line 735
    def get_strategy_info(self) -> dict[str, Any]  # Line 746
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
    async def _check_triangular_path(self, path: list[str]) -> Signal | None  # Line 155
    def _calculate_triangular_fees(self, rate1: Decimal, rate2: Decimal, rate3: Decimal) -> Decimal  # Line 269
    async def _validate_triangular_timing(self, path: list[str]) -> bool  # Line 352
    async def validate_signal(self, signal: Signal) -> bool  # Line 408
    def get_position_size(self, signal: Signal) -> Decimal  # Line 472
    async def should_exit(self, position: Position, data: MarketData) -> bool  # Line 558
    async def post_trade_processing(self, trade_result: dict[str, Any]) -> None  # Line 619
    async def _get_sma(self, symbol: str, period: int) -> Decimal | None  # Line 631
    async def _get_rsi(self, symbol: str, period: int) -> Decimal | None  # Line 642
    async def _get_volatility(self, symbol: str, period: int) -> Decimal | None  # Line 653
    async def _get_atr(self, symbol: str, period: int) -> Decimal | None  # Line 664
    async def _get_volume_ratio(self, symbol: str, period: int) -> Decimal | None  # Line 675
    async def _process_trade_result(self, trade_result: dict[str, Any]) -> None  # Line 686
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
    async def _validate_price_data(self, market_data: MarketData, result: ValidationResult) -> None  # Line 403
    async def _validate_market_hours(self, market_data: MarketData, result: ValidationResult) -> None  # Line 415
```

#### Class: `CompositeValidator`

**Inherits**: BaseValidator
**Purpose**: Composite validator that runs multiple validators

```python
class CompositeValidator(BaseValidator):
    def __init__(self, validators: list[BaseValidator], name: str = 'CompositeValidator')  # Line 430
    async def validate(self, target: Any, context: dict[str, Any] | None = None) -> ValidationResult  # Line 441
```

#### Class: `ValidationFramework`

**Purpose**: Main validation framework for strategies

```python
class ValidationFramework:
    def __init__(self, config: dict[str, Any] | None = None)  # Line 477
    async def validate_signal(self, signal: Signal, market_data: MarketData | None = None) -> ValidationResult  # Line 498
    async def validate_strategy_config(self, config: StrategyConfig) -> ValidationResult  # Line 517
    async def validate_market_conditions(self, market_data: MarketData) -> ValidationResult  # Line 529
    async def validate_for_trading(self, signal: Signal, market_data: MarketData) -> ValidationResult  # Line 541
    async def batch_validate_signals(self, signals: list[Signal], market_data: MarketData | None = None) -> list[tuple[Signal, ValidationResult]]  # Line 557
    def add_custom_validator(self, validator: BaseValidator, validator_type: str = 'custom') -> None  # Line 579
    def get_validation_stats(self) -> dict[str, Any]  # Line 599
```

---
**Generated**: Complete reference for strategies module
**Total Classes**: 108
**Total Functions**: 7