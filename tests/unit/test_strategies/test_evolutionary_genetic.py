"""
Unit tests for P-013D Evolutionary Strategies - Genetic Algorithm.

Tests cover:
- GeneticConfig validation
- Population initialization and management
- Selection mechanisms (tournament, elitism)
- Crossover and mutation operations
- Fitness evaluation and evolution process
- Diversity management and injection
- Early stopping criteria
- Error handling and edge cases
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np

from src.strategies.evolutionary.genetic import GeneticAlgorithm, GeneticConfig
from src.strategies.evolutionary.population import Individual, Population
from src.strategies.evolutionary.fitness import FitnessEvaluator
from src.strategies.evolutionary.mutations import MutationOperator, CrossoverOperator
from src.backtesting.engine import BacktestConfig, BacktestResult
from src.core.exceptions import OptimizationError
from src.strategies.base import BaseStrategy


class MockStrategy(BaseStrategy):
    """Mock strategy for genetic algorithm testing."""
    
    def __init__(self, **kwargs):
        # Extract custom parameters
        self.param1 = kwargs.get('param1', 1.0)
        self.param2 = kwargs.get('param2', 10)
        self.param3 = kwargs.get('param3', 'option1')
        
        # Create proper config for BaseStrategy
        config = {
            'name': 'test_strategy',
            'strategy_type': 'static',  # Use valid StrategyType enum value
            'symbols': kwargs.get('symbols', ['BTC/USD']),
            'parameters': {
                'param1': self.param1,
                'param2': self.param2,
                'param3': self.param3
            }
        }
        super().__init__(config)
    
    async def generate_signal(self, symbol: str, data):
        return None
    
    async def _generate_signals_impl(self, data):
        """Implementation required by BaseStrategy."""
        return []
    
    def get_position_size(self, signal, portfolio_value):
        """Implementation required by BaseStrategy."""
        return 1000.0
    
    def should_exit(self, position, current_data):
        """Implementation required by BaseStrategy."""
        return False
    
    async def validate_signal(self, signal):
        """Implementation required by BaseStrategy."""
        return True


class MockFitnessEvaluator(FitnessEvaluator):
    """Mock fitness evaluator for testing."""
    
    def __init__(self):
        super().__init__({})
    
    def evaluate(self, backtest_result: BacktestResult) -> float:
        # Simple fitness based on total return
        return float(backtest_result.total_return)


class TestGeneticConfig:
    """Test GeneticConfig validation and functionality."""
    
    def test_valid_config_creation(self):
        """Test creating a valid genetic algorithm configuration."""
        config = GeneticConfig(
            population_size=100,
            generations=50,
            mutation_rate=0.15,
            crossover_rate=0.8,
            elitism_rate=0.1,
            tournament_size=5
        )
        
        assert config.population_size == 100
        assert config.generations == 50
        assert config.mutation_rate == 0.15
        assert config.crossover_rate == 0.8
        assert config.elitism_rate == 0.1
        assert config.tournament_size == 5
    
    def test_default_values(self):
        """Test default configuration values."""
        config = GeneticConfig()
        
        assert config.population_size == 50
        assert config.generations == 100
        assert config.mutation_rate == 0.1
        assert config.crossover_rate == 0.7
        assert config.elitism_rate == 0.1
        assert config.tournament_size == 3
        assert config.parallel_evaluation is True
        assert config.early_stopping_generations == 10
        assert config.diversity_threshold == 0.1
    
    def test_config_validation_ranges(self):
        """Test configuration parameter validation."""
        # Valid ranges should work
        config = GeneticConfig(
            population_size=10,
            generations=5,
            mutation_rate=0.0,
            crossover_rate=1.0,
            elitism_rate=0.0
        )
        assert config.population_size == 10
        
        # Test edge cases
        config = GeneticConfig(
            mutation_rate=0.99,
            crossover_rate=0.01,
            elitism_rate=0.99
        )
        assert config.mutation_rate == 0.99


class TestGeneticAlgorithm:
    """Test GeneticAlgorithm functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test genetic algorithm configuration."""
        return GeneticConfig(
            population_size=20,
            generations=5,
            mutation_rate=0.2,
            crossover_rate=0.8,
            elitism_rate=0.2,
            tournament_size=3,
            parallel_evaluation=False,  # Easier to test
            early_stopping_generations=3
        )
    
    @pytest.fixture
    def parameter_ranges(self):
        """Create test parameter ranges."""
        return {
            'param1': (0.1, 2.0),  # Float parameter
            'param2': (5, 20),     # Integer parameter
            'param3': (['option1', 'option2', 'option3'], None)  # Categorical
        }
    
    @pytest.fixture
    def backtest_config(self):
        """Create test backtest configuration."""
        return BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            symbols=["BTC/USD"],
            initial_capital=10000
        )
    
    @pytest.fixture
    def fitness_evaluator(self):
        """Create mock fitness evaluator."""
        return MockFitnessEvaluator()
    
    @pytest.fixture
    def mock_backtest_result(self):
        """Create mock backtest result."""
        return BacktestResult(
            total_return=15.5,
            annual_return=18.2,
            sharpe_ratio=1.25,
            sortino_ratio=1.45,
            max_drawdown=8.5,
            win_rate=65.5,
            total_trades=100,
            winning_trades=65,
            losing_trades=35,
            avg_win=150.25,
            avg_loss=85.75,
            profit_factor=2.15,
            volatility=0.12,
            var_95=250.50,
            cvar_95=320.75,
            equity_curve=[],
            trades=[],
            daily_returns=[]
        )
    
    def test_genetic_algorithm_initialization(self, config, parameter_ranges, 
                                           fitness_evaluator, backtest_config):
        """Test genetic algorithm initialization."""
        ga = GeneticAlgorithm(
            config=config,
            strategy_class=MockStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        assert ga.config == config
        assert ga.strategy_class == MockStrategy
        assert ga.parameter_ranges == parameter_ranges
        assert ga.fitness_evaluator == fitness_evaluator
        assert ga.backtest_config == backtest_config
        assert ga.population is None
        assert ga.generation == 0
        assert ga.best_individual is None
        assert len(ga.evolution_history) == 0
    
    def test_population_initialization(self, config, parameter_ranges, 
                                     fitness_evaluator, backtest_config):
        """Test population initialization with random individuals."""
        ga = GeneticAlgorithm(
            config=config,
            strategy_class=MockStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        population = ga._initialize_population()
        
        assert isinstance(population, Population)
        assert len(population.individuals) == config.population_size
        
        # Check individuals have correct gene structure
        for individual in population.individuals:
            assert isinstance(individual, Individual)
            assert 'param1' in individual.genes
            assert 'param2' in individual.genes
            assert 'param3' in individual.genes
            
            # Check parameter ranges
            assert parameter_ranges['param1'][0] <= individual.genes['param1'] <= parameter_ranges['param1'][1]
            assert parameter_ranges['param2'][0] <= individual.genes['param2'] <= parameter_ranges['param2'][1]
            assert individual.genes['param3'] in parameter_ranges['param3'][0]
            
            # Check types
            assert isinstance(individual.genes['param1'], float)
            assert isinstance(individual.genes['param2'], int)
            assert isinstance(individual.genes['param3'], str)
    
    @pytest.mark.asyncio
    async def test_individual_evaluation(self, config, parameter_ranges, 
                                       fitness_evaluator, backtest_config, mock_backtest_result):
        """Test individual fitness evaluation."""
        ga = GeneticAlgorithm(
            config=config,
            strategy_class=MockStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        # Create test individual
        individual = Individual(
            id="test_ind",
            genes={'param1': 1.5, 'param2': 15, 'param3': 'option2'},
            fitness=0.0
        )
        
        with patch('src.strategies.evolutionary.genetic.BacktestEngine') as MockEngine:
            mock_engine = MockEngine.return_value
            mock_engine.run = AsyncMock(return_value=mock_backtest_result)
            
            await ga._evaluate_individual(individual)
            
            assert individual.fitness == float(mock_backtest_result.total_return)
            assert 'sharpe_ratio' in individual.metadata
            assert 'total_return' in individual.metadata
            assert 'max_drawdown' in individual.metadata
            assert 'win_rate' in individual.metadata
    
    @pytest.mark.asyncio
    async def test_individual_evaluation_error_handling(self, config, parameter_ranges, 
                                                      fitness_evaluator, backtest_config):
        """Test error handling during individual evaluation."""
        ga = GeneticAlgorithm(
            config=config,
            strategy_class=MockStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        individual = Individual(
            id="test_ind",
            genes={'param1': 1.5, 'param2': 15, 'param3': 'option2'},
            fitness=0.0
        )
        
        with patch('src.strategies.evolutionary.genetic.BacktestEngine') as MockEngine:
            mock_engine = MockEngine.return_value
            mock_engine.run = AsyncMock(side_effect=Exception("Backtest failed"))
            
            await ga._evaluate_individual(individual)
            
            assert individual.fitness == -float("inf")
    
    @pytest.mark.asyncio
    async def test_population_evaluation_sequential(self, config, parameter_ranges, 
                                                  fitness_evaluator, backtest_config):
        """Test sequential population evaluation."""
        config.parallel_evaluation = False
        
        ga = GeneticAlgorithm(
            config=config,
            strategy_class=MockStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        population = ga._initialize_population()
        
        with patch.object(ga, '_evaluate_individual', new_callable=AsyncMock) as mock_eval:
            mock_eval.return_value = None  # Don't actually evaluate
            
            await ga._evaluate_population(population)
            
            # Should call evaluate_individual for each individual
            assert mock_eval.call_count == len(population.individuals)
    
    @pytest.mark.asyncio
    async def test_population_evaluation_parallel(self, config, parameter_ranges, 
                                                fitness_evaluator, backtest_config):
        """Test parallel population evaluation."""
        config.parallel_evaluation = True
        
        ga = GeneticAlgorithm(
            config=config,
            strategy_class=MockStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        population = ga._initialize_population()
        
        with patch.object(ga, '_evaluate_individual', new_callable=AsyncMock) as mock_eval:
            mock_eval.return_value = None
            
            await ga._evaluate_population(population)
            
            # Should still evaluate all individuals
            assert mock_eval.call_count == len(population.individuals)
    
    def test_tournament_selection(self, config, parameter_ranges, 
                                 fitness_evaluator, backtest_config):
        """Test tournament selection mechanism."""
        ga = GeneticAlgorithm(
            config=config,
            strategy_class=MockStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        # Create population with known fitness values
        individuals = []
        for i in range(10):
            ind = Individual(
                id=f"ind_{i}",
                genes={'param1': 1.0, 'param2': 10, 'param3': 'option1'},
                fitness=float(i)  # Fitness 0-9
            )
            individuals.append(ind)
        
        population = Population(individuals)
        
        # Run tournament selection multiple times
        winners = []
        for _ in range(100):
            winner = ga._tournament_selection(population)
            winners.append(winner.fitness)
        
        # Higher fitness individuals should be selected more often
        avg_fitness = np.mean(winners)
        assert avg_fitness > 5.0  # Should bias toward higher fitness
    
    def test_selection_with_elitism(self, config, parameter_ranges, 
                                   fitness_evaluator, backtest_config):
        """Test selection with elitism."""
        ga = GeneticAlgorithm(
            config=config,
            strategy_class=MockStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        # Create population with known fitness values
        individuals = []
        for i in range(config.population_size):
            ind = Individual(
                id=f"ind_{i}",
                genes={'param1': 1.0, 'param2': 10, 'param3': 'option1'},
                fitness=float(i)
            )
            individuals.append(ind)
        
        population = Population(individuals)
        parents = ga._selection(population)
        
        assert len(parents) == config.population_size
        
        # Check that elite individuals are included
        num_elite = int(config.population_size * config.elitism_rate)
        top_fitness_values = sorted([ind.fitness for ind in individuals], reverse=True)[:num_elite]
        
        selected_fitness_values = [p.fitness for p in parents]
        for top_fitness in top_fitness_values:
            assert top_fitness in selected_fitness_values
    
    def test_offspring_creation(self, config, parameter_ranges, 
                               fitness_evaluator, backtest_config):
        """Test offspring creation through crossover and mutation."""
        ga = GeneticAlgorithm(
            config=config,
            strategy_class=MockStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        # Create parent population
        parents = []
        for i in range(config.population_size):
            ind = Individual(
                id=f"parent_{i}",
                genes={'param1': 1.0 + i*0.1, 'param2': 10+i, 'param3': 'option1'},
                fitness=float(i)
            )
            parents.append(ind)
        
        offspring_pop = ga._create_offspring(parents)
        
        assert isinstance(offspring_pop, Population)
        assert len(offspring_pop.individuals) == config.population_size
        
        # Check offspring have valid genes
        for offspring in offspring_pop.individuals:
            assert 'param1' in offspring.genes
            assert 'param2' in offspring.genes
            assert 'param3' in offspring.genes
            assert offspring.fitness == 0.0  # Not evaluated yet
    
    def test_replacement_strategy(self, config, parameter_ranges, 
                                 fitness_evaluator, backtest_config):
        """Test population replacement with elitism."""
        ga = GeneticAlgorithm(
            config=config,
            strategy_class=MockStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        # Create current population
        current_individuals = []
        for i in range(config.population_size):
            ind = Individual(
                id=f"current_{i}",
                genes={'param1': 1.0, 'param2': 10, 'param3': 'option1'},
                fitness=float(i)
            )
            current_individuals.append(ind)
        current_pop = Population(current_individuals)
        
        # Create offspring population
        offspring_individuals = []
        for i in range(config.population_size):
            ind = Individual(
                id=f"offspring_{i}",
                genes={'param1': 1.5, 'param2': 15, 'param3': 'option2'},
                fitness=float(i + 5)  # Higher fitness
            )
            offspring_individuals.append(ind)
        offspring_pop = Population(offspring_individuals)
        
        new_pop = ga._replacement(current_pop, offspring_pop)
        
        assert len(new_pop.individuals) == config.population_size
        
        # Elite from current population should be preserved
        num_elite = int(config.population_size * config.elitism_rate)
        top_current_fitness = sorted([ind.fitness for ind in current_individuals], reverse=True)[:num_elite]
        
        new_fitness_values = [ind.fitness for ind in new_pop.individuals]
        for elite_fitness in top_current_fitness:
            assert elite_fitness in new_fitness_values
    
    def test_diversity_calculation(self, config, parameter_ranges, 
                                  fitness_evaluator, backtest_config):
        """Test population diversity calculation."""
        ga = GeneticAlgorithm(
            config=config,
            strategy_class=MockStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        # Create population with high diversity
        diverse_individuals = []
        for i in range(5):
            ind = Individual(
                id=f"diverse_{i}",
                genes={
                    'param1': 0.1 + i * 0.4,  # Spread across range
                    'param2': 5 + i * 3,
                    'param3': ['option1', 'option2', 'option3'][i % 3]
                },
                fitness=1.0
            )
            diverse_individuals.append(ind)
        
        ga.population = Population(diverse_individuals)
        diversity_high = ga._calculate_diversity()
        
        # Create population with low diversity
        similar_individuals = []
        for i in range(5):
            ind = Individual(
                id=f"similar_{i}",
                genes={'param1': 1.0, 'param2': 10, 'param3': 'option1'},  # All same
                fitness=1.0
            )
            similar_individuals.append(ind)
        
        ga.population = Population(similar_individuals)
        diversity_low = ga._calculate_diversity()
        
        assert diversity_high > diversity_low
        assert diversity_low == 0.0  # Identical individuals
    
    def test_gene_distance_calculation(self, config, parameter_ranges, 
                                      fitness_evaluator, backtest_config):
        """Test gene distance calculation between individuals."""
        ga = GeneticAlgorithm(
            config=config,
            strategy_class=MockStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        genes1 = {'param1': 0.5, 'param2': 10, 'param3': 'option1'}
        genes2 = {'param1': 1.5, 'param2': 15, 'param3': 'option2'}
        
        distance = ga._gene_distance(genes1, genes2)
        
        assert distance > 0
        assert distance <= 1.0  # Normalized distance
        
        # Test identical genes
        distance_identical = ga._gene_distance(genes1, genes1)
        assert distance_identical == 0.0
    
    def test_diversity_injection(self, config, parameter_ranges, 
                                fitness_evaluator, backtest_config):
        """Test diversity injection mechanism."""
        ga = GeneticAlgorithm(
            config=config,
            strategy_class=MockStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        # Create population with low diversity
        individuals = []
        for i in range(config.population_size):
            ind = Individual(
                id=f"ind_{i}",
                genes={'param1': 1.0, 'param2': 10, 'param3': 'option1'},
                fitness=float(i)  # Different fitness, same genes
            )
            individuals.append(ind)
        
        ga.population = Population(individuals)
        ga.generation = 5
        
        # Store original genes of bottom individuals
        original_bottom = ga.population.get_bottom_n_indices(int(config.population_size * 0.2))
        
        ga._inject_diversity()
        
        # Check that bottom individuals were replaced with diverse genes
        for idx in original_bottom:
            new_individual = ga.population.individuals[idx]
            assert new_individual.fitness == 0.0  # Reset fitness
            assert new_individual.metadata.get('injected') is True
    
    def test_generation_stats_recording(self, config, parameter_ranges, 
                                       fitness_evaluator, backtest_config):
        """Test recording of generation statistics."""
        ga = GeneticAlgorithm(
            config=config,
            strategy_class=MockStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        # Create population with known fitness values
        individuals = []
        fitness_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for i, fitness in enumerate(fitness_values):
            ind = Individual(
                id=f"ind_{i}",
                genes={'param1': 1.0, 'param2': 10, 'param3': 'option1'},
                fitness=fitness
            )
            individuals.append(ind)
        
        ga.population = Population(individuals)
        ga.generation = 3
        
        ga._record_generation_stats()
        
        assert len(ga.evolution_history) == 1
        stats = ga.evolution_history[0]
        
        assert stats['generation'] == 3
        assert stats['best_fitness'] == 5.0
        assert stats['avg_fitness'] == 3.0
        assert 'std_fitness' in stats
        assert 'diversity' in stats
        assert 'timestamp' in stats
    
    @pytest.mark.asyncio
    async def test_early_stopping(self, config, parameter_ranges, 
                                 fitness_evaluator, backtest_config):
        """Test early stopping mechanism."""
        config.early_stopping_generations = 2
        
        ga = GeneticAlgorithm(
            config=config,
            strategy_class=MockStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        with patch.object(ga, '_evaluate_population', new_callable=AsyncMock) as mock_eval:
            # Mock population evaluation to return same fitness
            def mock_eval_func(population):
                for i, ind in enumerate(population.individuals):
                    ind.fitness = 5.0  # Same fitness every generation
            
            mock_eval.side_effect = mock_eval_func
            
            result = await ga.evolve()
            
            # Should stop early due to no improvement
            assert ga.generation < config.generations - 1
    
    @pytest.mark.asyncio
    async def test_full_evolution_process(self, config, parameter_ranges, 
                                        fitness_evaluator, backtest_config, 
                                        mock_backtest_result):
        """Test complete evolution process."""
        ga = GeneticAlgorithm(
            config=config,
            strategy_class=MockStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        with patch('src.strategies.evolutionary.genetic.BacktestEngine') as MockEngine:
            mock_engine = MockEngine.return_value
            mock_engine.run = AsyncMock(return_value=mock_backtest_result)
            
            best_individual = await ga.evolve()
            
            assert isinstance(best_individual, Individual)
            assert best_individual.fitness > 0
            assert len(ga.evolution_history) > 0
            assert ga.generation >= 0
    
    @pytest.mark.asyncio
    async def test_evolution_error_handling(self, config, parameter_ranges, 
                                          fitness_evaluator, backtest_config):
        """Test error handling during evolution."""
        ga = GeneticAlgorithm(
            config=config,
            strategy_class=MockStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        with patch.object(ga, '_initialize_population', side_effect=Exception("Init error")):
            with pytest.raises(OptimizationError, match="Evolution failed"):
                await ga.evolve()
    
    def test_evolution_summary(self, config, parameter_ranges, 
                              fitness_evaluator, backtest_config):
        """Test evolution summary generation."""
        ga = GeneticAlgorithm(
            config=config,
            strategy_class=MockStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        # Set up some evolution state
        ga.best_individual = Individual(
            id="best",
            genes={'param1': 1.5, 'param2': 15, 'param3': 'option2'},
            fitness=10.0
        )
        
        ga.evolution_history = [
            {'best_fitness': 5.0, 'diversity': 0.8, 'avg_fitness': 3.0, 'std_fitness': 1.0},
            {'best_fitness': 8.0, 'diversity': 0.6, 'avg_fitness': 4.0, 'std_fitness': 1.2},
            {'best_fitness': 10.0, 'diversity': 0.4, 'avg_fitness': 5.0, 'std_fitness': 1.5},
        ]
        
        summary = ga.get_evolution_summary()
        
        assert summary['generations_run'] == 3
        assert summary['best_fitness'] == 10.0
        assert summary['best_parameters'] == {'param1': 1.5, 'param2': 15, 'param3': 'option2'}
        assert summary['fitness_progression'] == [5.0, 8.0, 10.0]
        assert summary['diversity_progression'] == [0.8, 0.6, 0.4]
        assert 'final_population_stats' in summary
    
    def test_evolution_summary_empty(self, config, parameter_ranges, 
                                    fitness_evaluator, backtest_config):
        """Test evolution summary with no evolution history."""
        ga = GeneticAlgorithm(
            config=config,
            strategy_class=MockStrategy,
            parameter_ranges=parameter_ranges,
            fitness_evaluator=fitness_evaluator,
            backtest_config=backtest_config
        )
        
        summary = ga.get_evolution_summary()
        assert summary == {}


# Integration tests
@pytest.mark.asyncio
async def test_genetic_algorithm_integration():
    """Integration test of genetic algorithm with real components."""
    config = GeneticConfig(
        population_size=10,
        generations=3,
        mutation_rate=0.3,
        crossover_rate=0.8,
        parallel_evaluation=False
    )
    
    parameter_ranges = {
        'lookback_period': (5, 20),
        'threshold': (0.1, 0.5),
        'use_volume': ([True, False], None)
    }
    
    backtest_config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 10),
        symbols=["BTC/USD"],
        initial_capital=10000
    )
    
    fitness_evaluator = MockFitnessEvaluator()
    
    ga = GeneticAlgorithm(
        config=config,
        strategy_class=MockStrategy,
        parameter_ranges=parameter_ranges,
        fitness_evaluator=fitness_evaluator,
        backtest_config=backtest_config
    )
    
    # Mock the backtest engine to return consistent results
    with patch('src.strategies.evolutionary.genetic.BacktestEngine') as MockEngine:
        def create_mock_result(strategy):
            # Vary fitness based on parameters to test evolution
            param_value = strategy.genes.get('threshold', 0.3)
            fitness_value = param_value * 100  # Simple fitness function
            
            return BacktestResult(
                total_return=fitness_value,
                annual_return=fitness_value * 1.2,
                sharpe_ratio=1.0,
                sortino_ratio=1.1,
                max_drawdown=5.0,
                win_rate=60.0,
                total_trades=50,
                winning_trades=30,
                losing_trades=20,
                avg_win=100.0,
                avg_loss=50.0,
                profit_factor=2.0,
                volatility=0.15,
                var_95=200.0,
                cvar_95=300.0,
                equity_curve=[],
                trades=[],
                daily_returns=[]
            )
        
        MockEngine.return_value.run = AsyncMock(side_effect=lambda: create_mock_result(MockEngine.return_value.strategy))
        
        best_individual = await ga.evolve()
        
        assert isinstance(best_individual, Individual)
        assert best_individual.fitness > 0
        assert len(ga.evolution_history) == config.generations
        
        # Evolution should improve fitness over generations
        fitness_progression = [h['best_fitness'] for h in ga.evolution_history]
        # Allow for some stochasticity - final fitness should be >= initial
        assert fitness_progression[-1] >= fitness_progression[0] * 0.8


@pytest.mark.performance
@pytest.mark.asyncio
async def test_genetic_algorithm_performance():
    """Test genetic algorithm performance with larger population."""
    config = GeneticConfig(
        population_size=50,
        generations=10,
        parallel_evaluation=True
    )
    
    parameter_ranges = {
        'param1': (0.1, 2.0),
        'param2': (5, 50),
        'param3': (['A', 'B', 'C', 'D'], None)
    }
    
    backtest_config = BacktestConfig(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 31),
        symbols=["BTC/USD"],
        initial_capital=10000
    )
    
    fitness_evaluator = MockFitnessEvaluator()
    
    ga = GeneticAlgorithm(
        config=config,
        strategy_class=MockStrategy,
        parameter_ranges=parameter_ranges,
        fitness_evaluator=fitness_evaluator,
        backtest_config=backtest_config
    )
    
    # Mock fast evaluation
    with patch.object(ga, '_evaluate_individual', new_callable=AsyncMock) as mock_eval:
        async def fast_eval(individual):
            individual.fitness = np.random.uniform(0, 100)
        
        mock_eval.side_effect = fast_eval
        
        import time
        start_time = time.time()
        
        await ga.evolve()
        
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert execution_time < 10.0  # 10 seconds for 50x10 = 500 evaluations