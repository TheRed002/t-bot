"""
Integration tests for P-013D Evolutionary Strategies.

Tests cover:
- Genetic algorithm integration with backtesting
- Population management with real strategies
- Fitness evaluation with actual backtest results
- Multi-objective optimization integration
- Neuroevolution integration
- Performance with realistic parameter spaces
- Error handling and recovery
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, patch
import numpy as np
import pandas as pd

from src.strategies.evolutionary.genetic import GeneticAlgorithm, GeneticConfig
from src.strategies.evolutionary.fitness import FitnessEvaluator
from src.strategies.evolutionary.population import Individual, Population
from src.backtesting.engine import BacktestEngine, BacktestConfig, BacktestResult
from src.core.types import MarketData, Position, Signal, SignalDirection, StrategyConfig, StrategyType
from src.strategies.base import BaseStrategy


class ParameterizedTestStrategy(BaseStrategy):
    """Test strategy with multiple parameters for optimization."""
    
    def __init__(self, **config):
        # Create proper config for BaseStrategy 
        strategy_config = {
            'name': 'parameterized_test_strategy',
            'strategy_id': 'param_test_001',
            'strategy_type': 'mean_reversion',
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'symbols': ['BTCUSDT'],
            **config
        }
        super().__init__(strategy_config)
        self.lookback_period = config.get('lookback_period', 20)
        self.threshold = config.get('threshold', 0.02)
        self.use_volume = config.get('use_volume', True)
        self.rsi_period = config.get('rsi_period', 14)
        self.ma_ratio = config.get('ma_ratio', 1.5)
        self.signal_history = {}
    
    @property
    def strategy_type(self) -> StrategyType:
        """Get the strategy type."""
        return StrategyType.MEAN_REVERSION
    
    @property
    def name(self) -> str:
        """Get strategy name."""
        return getattr(self, '_name', "parameterized_test_strategy")
    
    @name.setter
    def name(self, value: str) -> None:
        """Set strategy name."""
        self._name = value
    
    @property
    def version(self) -> str:
        """Get strategy version."""
        return getattr(self, '_version', "1.0.0")
    
    @version.setter 
    def version(self, value: str) -> None:
        """Set strategy version."""
        self._version = value
    
    @property
    def status(self) -> 'StrategyStatus':
        """Get strategy status."""
        from src.core.types import StrategyStatus
        return getattr(self, '_status', StrategyStatus.ACTIVE)
    
    @status.setter
    def status(self, value: 'StrategyStatus') -> None:
        """Set strategy status.""" 
        self._status = value
    
    async def initialize(self, config: StrategyConfig) -> None:
        """Initialize the strategy with configuration."""
        pass
    
    async def initialize_symbol(self, symbol: str, data: pd.DataFrame):
        """Initialize strategy for a specific symbol."""
        self.signal_history[symbol] = []
    
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> SignalDirection:
        """Generate signals based on parameters."""
        if len(data) < max(self.lookback_period, self.rsi_period):
            return SignalDirection.HOLD
        
        # Calculate indicators based on parameters
        returns = data['close'].pct_change(self.lookback_period)
        current_return = returns.iloc[-1]
        
        # RSI calculation
        deltas = data['close'].diff()
        gains = deltas.where(deltas > 0, 0).rolling(self.rsi_period).mean()
        losses = (-deltas).where(deltas < 0, 0).rolling(self.rsi_period).mean()
        rsi = 100 - (100 / (1 + gains / losses)).iloc[-1]
        
        # Moving average signal
        ma_short = data['close'].rolling(int(self.lookback_period / 2)).mean().iloc[-1]
        ma_long = data['close'].rolling(self.lookback_period).mean().iloc[-1]
        ma_signal = ma_short / ma_long
        
        # Volume confirmation if enabled
        volume_signal = 1.0
        if self.use_volume and len(data) > 5:
            recent_volume = data['volume'].iloc[-5:].mean()
            avg_volume = data['volume'].rolling(self.lookback_period).mean().iloc[-1]
            volume_signal = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Combine signals
        buy_signal = (
            current_return > self.threshold and
            rsi < 70 and
            ma_signal > self.ma_ratio and
            (not self.use_volume or volume_signal > 1.2)
        )
        
        sell_signal = (
            current_return < -self.threshold and
            rsi > 30 and
            ma_signal < (1 / self.ma_ratio) and
            (not self.use_volume or volume_signal > 1.2)
        )
        
        if buy_signal:
            signal = SignalDirection.BUY
        elif sell_signal:
            signal = SignalDirection.SELL
        else:
            signal = SignalDirection.HOLD
        
        # Track signal for debugging
        if symbol in self.signal_history:
            self.signal_history[symbol].append({
                'timestamp': data.index[-1],
                'signal': signal,
                'current_return': current_return,
                'rsi': rsi,
                'ma_signal': ma_signal,
                'volume_signal': volume_signal
            })
        
        return signal
    
    async def generate_signals(self, data: MarketData) -> list[Signal]:
        """Generate trading signals from market data."""
        return []
    
    async def _generate_signals_impl(self, data: MarketData) -> list[Signal]:
        """Implementation required by BaseStrategy."""
        return []
    
    def get_position_size(self, signal: Signal) -> Decimal:
        """Implementation required by BaseStrategy."""
        return Decimal("1000.0")
    
    def should_exit(self, position: Position, data: MarketData) -> bool:
        """Implementation required by BaseStrategy."""
        return False
    
    async def validate_signal(self, signal: Signal) -> bool:
        """Implementation required by BaseStrategy."""
        return True


class MultiFitnesEvaluator(FitnessEvaluator):
    """Multi-objective fitness evaluator for testing."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.return_weight = config.get('return_weight', 0.4)
        self.sharpe_weight = config.get('sharpe_weight', 0.3)
        self.drawdown_weight = config.get('drawdown_weight', 0.3)
    
    def evaluate(self, backtest_result: BacktestResult) -> float:
        """Multi-objective fitness evaluation."""
        # Normalize metrics to 0-100 scale
        total_return = max(0, min(100, float(backtest_result.total_return)))
        sharpe_ratio = max(0, min(5, backtest_result.sharpe_ratio)) * 20  # Scale to 0-100
        drawdown_penalty = max(0, 100 - float(backtest_result.max_drawdown))
        
        # Weighted combination
        fitness = (
            total_return * self.return_weight +
            sharpe_ratio * self.sharpe_weight +
            drawdown_penalty * self.drawdown_weight
        )
        
        return fitness


@pytest.mark.integration
class TestEvolutionaryStrategiesIntegration:
    """Integration tests for evolutionary strategies."""
    
    @pytest.fixture
    def genetic_config(self):
        """Create genetic algorithm configuration."""
        return GeneticConfig(
            population_size=20,
            generations=5,
            mutation_rate=0.2,
            crossover_rate=0.8,
            elitism_rate=0.2,
            parallel_evaluation=False,  # Easier for testing
            early_stopping_generations=3
        )
    
    @pytest.fixture
    def parameter_ranges(self):
        """Define parameter ranges for optimization."""
        return {
            'lookback_period': (10, 50),
            'threshold': (0.01, 0.05),
            'use_volume': ([True, False], None),
            'rsi_period': (10, 20),
            'ma_ratio': (1.2, 2.0)
        }
    
    @pytest.fixture
    def backtest_config(self):
        """Create backtest configuration."""
        return BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 2, 28),  # 2 months
            symbols=["BTC/USD"],
            initial_capital=Decimal("50000"),
            commission=Decimal("0.001"),
            warm_up_period=20
        )
    
    @pytest.fixture
    def fitness_evaluator(self):
        """Create fitness evaluator."""
        return MultiFitnesEvaluator({
            'return_weight': 0.5,
            'sharpe_weight': 0.3,
            'drawdown_weight': 0.2
        })
    
    @pytest.fixture
    async def mock_market_data_db(self):
        """Create mock database with realistic market data."""
        db_manager = AsyncMock()
        
        async def generate_realistic_data(query, symbol, start_date, end_date):
            # Generate 2 months of hourly data
            dates = pd.date_range(start=start_date, end=end_date, freq='1H')
            np.random.seed(42)  # Reproducible results
            
            # Create realistic price movement with trends and volatility
            base_price = 45000
            prices = [base_price]
            
            # Add some market regimes
            total_hours = len(dates)
            trend_changes = [0, total_hours//3, 2*total_hours//3, total_hours]
            trends = [0.0001, -0.0002, 0.0003]  # Different trend periods
            
            for i, date in enumerate(dates):
                # Determine current trend
                trend_idx = next(j for j, change_point in enumerate(trend_changes[1:]) 
                               if i < change_point)
                trend = trends[trend_idx]
                
                # Add trend + noise
                daily_return = trend + np.random.normal(0, 0.015)
                new_price = prices[-1] * (1 + daily_return)
                prices.append(new_price)
            
            prices = prices[1:]  # Remove initial price
            
            # Create OHLCV data
            data = []
            for i, (date, close_price) in enumerate(zip(dates, prices)):
                open_price = prices[i-1] if i > 0 else close_price
                high = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.003)))
                low = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.003)))
                volume = np.random.lognormal(7, 0.5)  # Realistic volume distribution
                
                data.append({
                    "timestamp": date,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close_price,
                    "volume": volume
                })
            
            return data
        
        db_manager.fetch_all = generate_realistic_data
        return db_manager
    
    @pytest.mark.asyncio
    async def test_genetic_algorithm_with_real_backtesting(self, genetic_config, parameter_ranges, 
                                                         backtest_config, fitness_evaluator, 
                                                         mock_market_data_db):
        """Test genetic algorithm with real backtesting integration."""
        ga = GeneticAlgorithm(
            config=genetic_config,
            strategy_class=ParameterizedTestStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        # Patch BacktestEngine to use our mock data
        with patch('src.strategies.evolutionary.genetic.BacktestEngine') as MockEngine:
            def create_engine(config, strategy, **kwargs):
                engine = BacktestEngine(config, strategy, db_manager=mock_market_data_db)
                return engine
            
            MockEngine.side_effect = create_engine
            
            best_individual = await ga.evolve()
            
            # Should complete evolution successfully
            assert best_individual is not None
            assert isinstance(best_individual, Individual)
            assert best_individual.fitness > 0
            
            # Should have evolution history
            assert len(ga.evolution_history) == genetic_config.generations
            
            # Parameters should be within expected ranges
            for param, value in best_individual.genes.items():
                if param in parameter_ranges:
                    param_range = parameter_ranges[param]
                    if isinstance(param_range[0], (int, float)):
                        assert param_range[0] <= value <= param_range[1]
                    elif isinstance(param_range[0], list):
                        assert value in param_range[0]
            
            # Fitness should improve over generations (generally)
            fitness_progression = [h['best_fitness'] for h in ga.evolution_history]
            # Allow for some stochasticity - final should be >= 80% of max improvement
            max_improvement = max(fitness_progression) - fitness_progression[0]
            actual_improvement = fitness_progression[-1] - fitness_progression[0]
            assert actual_improvement >= max_improvement * 0.5  # At least 50% of potential
    
    @pytest.mark.asyncio
    async def test_population_diversity_evolution(self, genetic_config, parameter_ranges, 
                                                 backtest_config, fitness_evaluator, 
                                                 mock_market_data_db):
        """Test population diversity evolution and maintenance."""
        ga = GeneticAlgorithm(
            config=genetic_config,
            strategy_class=ParameterizedTestStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        # Track diversity progression
        diversity_history = []
        
        with patch('src.strategies.evolutionary.genetic.BacktestEngine') as MockEngine:
            def create_engine(config, strategy, **kwargs):
                engine = BacktestEngine(config, strategy, db_manager=mock_market_data_db)
                return engine
            
            MockEngine.side_effect = create_engine
            
            # Patch diversity recording to capture values
            original_record_stats = ga._record_generation_stats
            def track_diversity():
                diversity = ga._calculate_diversity()
                diversity_history.append(diversity)
                original_record_stats()
            
            ga._record_generation_stats = track_diversity
            
            best_individual = await ga.evolve()
            
            # Should maintain reasonable diversity
            assert len(diversity_history) == genetic_config.generations
            
            # Initial diversity should be high
            assert diversity_history[0] > 0.1
            
            # Final diversity should not be too low (avoid premature convergence)
            assert diversity_history[-1] > 0.05
    
    @pytest.mark.asyncio
    async def test_fitness_evaluation_accuracy(self, genetic_config, parameter_ranges, 
                                              backtest_config, fitness_evaluator, 
                                              mock_market_data_db):
        """Test accuracy of fitness evaluation."""
        ga = GeneticAlgorithm(
            config=genetic_config,
            strategy_class=ParameterizedTestStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        # Create test individual with known parameters
        test_individual = Individual(
            id="test_fitness",
            genes={
                'lookback_period': 20,
                'threshold': 0.02,
                'use_volume': True,
                'rsi_period': 14,
                'ma_ratio': 1.5
            },
            fitness=0.0
        )
        
        with patch('src.strategies.evolutionary.genetic.BacktestEngine') as MockEngine:
            def create_engine(config, strategy, **kwargs):
                engine = BacktestEngine(config, strategy, db_manager=mock_market_data_db)
                return engine
            
            MockEngine.side_effect = create_engine
            
            # Evaluate individual
            await ga._evaluate_individual(test_individual)
            
            # Should have valid fitness
            assert test_individual.fitness != 0.0
            assert not np.isnan(test_individual.fitness)
            assert not np.isinf(test_individual.fitness)
            
            # Should have metadata from backtest
            assert 'sharpe_ratio' in test_individual.metadata
            assert 'total_return' in test_individual.metadata
            assert 'max_drawdown' in test_individual.metadata
            assert 'win_rate' in test_individual.metadata
            
            # Metadata should be reasonable
            assert -100 <= test_individual.metadata['total_return'] <= 1000  # Reasonable return range
            assert 0 <= test_individual.metadata['win_rate'] <= 100
            assert 0 <= test_individual.metadata['max_drawdown'] <= 100
    
    @pytest.mark.asyncio
    async def test_multi_symbol_optimization(self, genetic_config, parameter_ranges, 
                                           fitness_evaluator, mock_market_data_db):
        """Test optimization with multiple symbols."""
        # Multi-symbol backtest config
        multi_symbol_config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),  # 1 month for speed
            symbols=["BTC/USD", "ETH/USD"],
            initial_capital=Decimal("100000"),
            warm_up_period=20
        )
        
        ga = GeneticAlgorithm(
            config=genetic_config,
            strategy_class=ParameterizedTestStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=multi_symbol_config
        )
        
        # Mock data for multiple symbols
        async def multi_symbol_data(query, symbol, start_date, end_date):
            # Different base prices for different symbols
            base_prices = {"BTC/USD": 45000, "ETH/USD": 3000}
            base_price = base_prices.get(symbol, 1000)
            
            dates = pd.date_range(start=start_date, end=end_date, freq='1H')
            np.random.seed(42 if symbol == "BTC/USD" else 123)  # Different seeds
            
            prices = [base_price]
            for _ in dates:
                daily_return = np.random.normal(0, 0.02)
                prices.append(prices[-1] * (1 + daily_return))
            
            data = []
            for i, (date, price) in enumerate(zip(dates, prices[1:])):
                data.append({
                    "timestamp": date,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": np.random.uniform(100, 1000)
                })
            
            return data
        
        mock_market_data_db.fetch_all = multi_symbol_data
        
        with patch('src.strategies.evolutionary.genetic.BacktestEngine') as MockEngine:
            def create_engine(config, strategy, **kwargs):
                engine = BacktestEngine(config, strategy, db_manager=mock_market_data_db)
                return engine
            
            MockEngine.side_effect = create_engine
            
            best_individual = await ga.evolve()
            
            # Should handle multi-symbol optimization
            assert best_individual is not None
            assert best_individual.fitness > 0
    
    @pytest.mark.asyncio
    async def test_optimization_error_resilience(self, genetic_config, parameter_ranges, 
                                               backtest_config, fitness_evaluator, 
                                               mock_market_data_db):
        """Test optimization resilience to errors."""
        ga = GeneticAlgorithm(
            config=genetic_config,
            strategy_class=ParameterizedTestStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        # Inject errors in some backtests
        error_count = 0
        max_errors = 3
        
        with patch('src.strategies.evolutionary.genetic.BacktestEngine') as MockEngine:
            def create_problematic_engine(config, strategy, **kwargs):
                engine = BacktestEngine(config, strategy, db_manager=mock_market_data_db)
                
                # Patch run method to occasionally fail
                original_run = engine.run
                
                async def failing_run():
                    nonlocal error_count
                    if error_count < max_errors and np.random.random() < 0.3:  # 30% failure rate
                        error_count += 1
                        raise Exception(f"Simulated backtest error {error_count}")
                    return await original_run()
                
                engine.run = failing_run
                return engine
            
            MockEngine.side_effect = create_problematic_engine
            
            best_individual = await ga.evolve()
            
            # Should complete despite errors
            assert best_individual is not None
            
            # Some individuals should have -inf fitness due to errors
            final_population = ga.population
            failed_individuals = [ind for ind in final_population.individuals 
                                if ind.fitness == -float("inf")]
            
            # Should have some failures but not all
            assert 0 < len(failed_individuals) < len(final_population.individuals)
    
    @pytest.mark.asyncio
    async def test_parameter_sensitivity_analysis(self, genetic_config, parameter_ranges, 
                                                 backtest_config, fitness_evaluator, 
                                                 mock_market_data_db):
        """Test parameter sensitivity through optimization."""
        # Reduce population for focused analysis
        genetic_config.population_size = 10
        genetic_config.generations = 3
        
        ga = GeneticAlgorithm(
            config=genetic_config,
            strategy_class=ParameterizedTestStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        parameter_fitness_correlation = {}
        
        with patch('src.strategies.evolutionary.genetic.BacktestEngine') as MockEngine:
            def create_engine(config, strategy, **kwargs):
                engine = BacktestEngine(config, strategy, db_manager=mock_market_data_db)
                return engine
            
            MockEngine.side_effect = create_engine
            
            best_individual = await ga.evolve()
            
            # Analyze final population for parameter-fitness relationships
            final_population = ga.population
            
            for param in parameter_ranges:
                param_values = []
                fitness_values = []
                
                for individual in final_population.individuals:
                    if individual.fitness > -float("inf"):  # Valid fitness
                        param_values.append(individual.genes[param])
                        fitness_values.append(individual.fitness)
                
                if len(param_values) > 3:  # Enough data points
                    # Calculate correlation
                    if isinstance(param_values[0], (int, float)):
                        correlation = np.corrcoef(param_values, fitness_values)[0, 1]
                        parameter_fitness_correlation[param] = correlation
                    else:
                        # For categorical parameters, calculate group means
                        unique_values = list(set(param_values))
                        group_fitness = {}
                        for val in unique_values:
                            group_fitness[val] = np.mean([f for p, f in zip(param_values, fitness_values) if p == val])
                        parameter_fitness_correlation[param] = group_fitness
            
            # Should have some parameter insights
            assert len(parameter_fitness_correlation) > 0
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_optimization_performance_scaling(self, parameter_ranges, fitness_evaluator, 
                                                   mock_market_data_db):
        """Test optimization performance with different population sizes."""
        backtest_config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 15),  # 2 weeks for speed
            symbols=["BTC/USD"],
            initial_capital=Decimal("50000"),
            warm_up_period=20
        )
        
        performance_results = {}
        
        for pop_size in [10, 20, 30]:
            config = GeneticConfig(
                population_size=pop_size,
                generations=2,  # Minimal for performance test
                parallel_evaluation=False
            )
            
            ga = GeneticAlgorithm(
                config=config,
                strategy_class=ParameterizedTestStrategy,
                parameter_ranges=parameter_ranges,
                fitness_evaluator=fitness_evaluator,
                backtest_config=backtest_config
            )
            
            with patch('src.strategies.evolutionary.genetic.BacktestEngine') as MockEngine:
                def create_engine(config, strategy, **kwargs):
                    engine = BacktestEngine(config, strategy, db_manager=mock_market_data_db)
                    return engine
                
                MockEngine.side_effect = create_engine
                
                import time
                start_time = time.time()
                
                best_individual = await ga.evolve()
                
                execution_time = time.time() - start_time
                performance_results[pop_size] = {
                    'time': execution_time,
                    'best_fitness': best_individual.fitness if best_individual else 0
                }
        
        # Should scale reasonably with population size
        times = [performance_results[size]['time'] for size in [10, 20, 30]]
        
        # Time should increase with population size but not exponentially
        assert times[1] > times[0]  # 20 > 10
        assert times[2] > times[1]  # 30 > 20
        
        # Should not be more than linear scaling (with some tolerance)
        assert times[2] / times[0] < 4.0  # 30-pop shouldn't take 4x longer than 10-pop
    
    @pytest.mark.asyncio
    async def test_early_stopping_effectiveness(self, genetic_config, parameter_ranges, 
                                               backtest_config, fitness_evaluator, 
                                               mock_market_data_db):
        """Test early stopping mechanism effectiveness."""
        genetic_config.early_stopping_generations = 2
        genetic_config.generations = 10  # Would run longer without early stopping
        
        ga = GeneticAlgorithm(
            config=genetic_config,
            strategy_class=ParameterizedTestStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        with patch('src.strategies.evolutionary.genetic.BacktestEngine') as MockEngine:
            def create_engine(config, strategy, **kwargs):
                engine = BacktestEngine(config, strategy, db_manager=mock_market_data_db)
                
                # Mock consistent results to trigger early stopping
                original_run = engine.run
                
                async def consistent_run():
                    result = await original_run()
                    # Make all results similar to trigger early stopping
                    result.total_return = Decimal("5.0")
                    result.sharpe_ratio = 1.0
                    result.max_drawdown = Decimal("3.0")
                    return result
                
                engine.run = consistent_run
                return engine
            
            MockEngine.side_effect = create_engine
            
            best_individual = await ga.evolve()
            
            # Should stop early
            assert ga.generation < genetic_config.generations - 1
            assert len(ga.evolution_history) <= genetic_config.early_stopping_generations + 1


@pytest.mark.integration
class TestEvolutionaryStrategiesComplexScenarios:
    """Test complex evolutionary optimization scenarios."""
    
    @pytest.mark.asyncio
    async def test_multi_objective_trade_offs(self):
        """Test multi-objective optimization trade-offs."""
        # Define conflicting objectives
        class ConflictingFitnessEvaluator(FitnessEvaluator):
            def __init__(self, objective_type):
                super().__init__({})
                self.objective_type = objective_type
            
            def evaluate(self, backtest_result: BacktestResult) -> float:
                if self.objective_type == "return_focused":
                    # Maximize returns, ignore risk
                    return float(backtest_result.total_return)
                elif self.objective_type == "risk_focused":
                    # Minimize drawdown, ignore returns
                    return 100 - float(backtest_result.max_drawdown)
                else:  # balanced
                    # Balance both
                    return (float(backtest_result.total_return) + 
                           (100 - float(backtest_result.max_drawdown))) / 2
        
        config = GeneticConfig(population_size=15, generations=3)
        parameter_ranges = {
            'lookback_period': (5, 30),
            'threshold': (0.01, 0.08),
            'use_volume': ([True, False], None)
        }
        backtest_config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            symbols=["BTC/USD"],
            initial_capital=Decimal("50000")
        )
        
        # Create mock data
        db_manager = AsyncMock()
        async def mock_data(query, symbol, start_date, end_date):
            dates = pd.date_range(start=start_date, end=end_date, freq='1H')
            np.random.seed(42)
            prices = [45000]
            for _ in dates:
                prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
            
            return [{"timestamp": date, "open": price, "high": price*1.01, 
                    "low": price*0.99, "close": price, "volume": 1000}
                   for date, price in zip(dates, prices[1:])]
        
        db_manager.fetch_all = mock_data
        
        results = {}
        
        for objective in ["return_focused", "risk_focused", "balanced"]:
            fitness_evaluator = ConflictingFitnessEvaluator(objective)
            
            ga = GeneticAlgorithm(
                config=config,
                strategy_class=ParameterizedTestStrategy,
                parameter_ranges=parameter_ranges,
                fitness_evaluator=fitness_evaluator,
                backtest_config=backtest_config
            )
            
            with patch('src.strategies.evolutionary.genetic.BacktestEngine') as MockEngine:
                def create_engine(config, strategy, **kwargs):
                    return BacktestEngine(config, strategy, db_manager=db_manager)
                MockEngine.side_effect = create_engine
                
                best_individual = await ga.evolve()
                results[objective] = best_individual
        
        # Different objectives should produce different optimal parameters
        return_best = results["return_focused"]
        risk_best = results["risk_focused"]
        
        # Should have different parameter sets for different objectives
        param_differences = sum(1 for param in parameter_ranges
                              if return_best.genes[param] != risk_best.genes[param])
        assert param_differences > 0  # At least some parameters should differ