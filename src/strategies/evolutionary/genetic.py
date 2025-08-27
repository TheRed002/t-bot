"""
Genetic Algorithm Implementation for Strategy Evolution.

This module provides the core genetic algorithm for evolving trading strategies
through selection, crossover, and mutation operations.
"""

import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.core.exceptions import OptimizationError
from src.utils.decorators import time_execution

from .fitness import FitnessEvaluator
from .mutations import CrossoverOperator, MutationOperator
from .population import Individual, Population


class GeneticConfig(BaseModel):
    """Configuration for genetic algorithm."""

    population_size: int = Field(default=50, description="Size of population")
    generations: int = Field(default=100, description="Number of generations")
    mutation_rate: float = Field(default=0.1, description="Mutation probability")
    crossover_rate: float = Field(default=0.7, description="Crossover probability")
    elitism_rate: float = Field(default=0.1, description="Elite preservation rate")
    tournament_size: int = Field(default=3, description="Tournament selection size")
    parallel_evaluation: bool = Field(default=True, description="Evaluate population in parallel")
    early_stopping_generations: int = Field(
        default=10, description="Generations without improvement for early stop"
    )
    diversity_threshold: float = Field(default=0.1, description="Minimum population diversity")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "population_size": 100,
                "generations": 50,
                "mutation_rate": 0.15,
                "crossover_rate": 0.8,
            }
        }
    )


class GeneticAlgorithm:
    """
    Genetic Algorithm for evolving trading strategies.

    Features:
    - Configurable selection methods
    - Advanced crossover and mutation operators
    - Elitism and diversity preservation
    - Parallel fitness evaluation
    - Early stopping criteria
    """

    def __init__(
        self,
        config: GeneticConfig,
        strategy_class: type,
        parameter_ranges: dict[str, tuple[Any, Any]],
        fitness_evaluator: FitnessEvaluator,
        backtest_config: BacktestConfig,
    ):
        """
        Initialize genetic algorithm.

        Args:
            config: GA configuration
            strategy_class: Strategy class to evolve
            parameter_ranges: Parameter ranges for evolution
            fitness_evaluator: Fitness evaluation function
            backtest_config: Backtesting configuration
        """
        self.config = config
        self.strategy_class = strategy_class
        self.parameter_ranges = parameter_ranges
        self.fitness_evaluator = fitness_evaluator
        self.backtest_config = backtest_config

        # Initialize operators
        self.mutation_operator = MutationOperator(mutation_rate=config.mutation_rate)
        self.crossover_operator = CrossoverOperator(crossover_rate=config.crossover_rate)

        # Evolution state
        self.population: Population | None = None
        self.generation = 0
        self.best_individual: Individual | None = None
        self.evolution_history: list[dict[str, Any]] = []

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        self.logger.info(
            f"GeneticAlgorithm initialized with strategy {strategy_class.__name__}, parameters {list(parameter_ranges.keys())}, population_size {config.population_size}"
        )

    @time_execution
    async def evolve(self) -> Individual:
        """
        Run the genetic algorithm evolution.

        Returns:
            Best individual found during evolution
        """
        try:
            self.logger.info(f"Starting evolution with {self.config.generations} generations")

            # Initialize population
            self.population = self._initialize_population()

            # Evaluate initial population
            await self._evaluate_population(self.population)

            # Evolution loop
            generations_without_improvement = 0
            previous_best_fitness = float("-inf")

            for generation in range(self.config.generations):
                self.generation = generation
                self.logger.info(f"Generation {generation + 1}/{self.config.generations}")

                # Select parents
                parents = self._selection(self.population)

                # Create offspring
                offspring = self._create_offspring(parents)

                # Evaluate offspring
                await self._evaluate_population(offspring)

                # Replace population
                self.population = self._replacement(self.population, offspring)

                # Track best individual
                current_best = self.population.get_best()
                if (
                    self.best_individual is None
                    or current_best.fitness > self.best_individual.fitness
                ):
                    self.best_individual = current_best
                    self.logger.info(
                        f"New best individual found with fitness {current_best.fitness} at generation {generation}"
                    )

                # Record history
                self._record_generation_stats()

                # Check early stopping
                if current_best.fitness > previous_best_fitness:
                    previous_best_fitness = current_best.fitness
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1

                if generations_without_improvement >= self.config.early_stopping_generations:
                    self.logger.info("Early stopping triggered")
                    break

                # Check diversity
                diversity = self._calculate_diversity()
                if diversity < self.config.diversity_threshold:
                    self.logger.warning(f"Low diversity: {diversity:.3f}")
                    self._inject_diversity()

            self.logger.info(
                f"Evolution completed with best_fitness {self.best_individual.fitness if self.best_individual else 0}, generations_run {self.generation + 1}"
            )

            return self.best_individual

        except Exception as e:
            self.logger.error(f"Evolution failed: {e!s}")
            raise OptimizationError(f"Evolution failed: {e!s}")

    def _initialize_population(self) -> Population:
        """Initialize random population."""
        individuals = []

        for i in range(self.config.population_size):
            # Generate random parameters
            genes = {}
            for param, (min_val, max_val) in self.parameter_ranges.items():
                if isinstance(min_val, int | float):
                    # Numeric parameter
                    if isinstance(min_val, int):
                        value = random.randint(min_val, max_val)
                    else:
                        value = random.uniform(min_val, max_val)
                elif isinstance(min_val, list):
                    # Categorical parameter
                    value = random.choice(min_val)
                else:
                    # Boolean or other
                    value = random.choice([min_val, max_val])

                genes[param] = value

            individual = Individual(
                id=f"gen0_ind{i}",
                genes=genes,
                fitness=0.0,
                metadata={"generation": 0},
            )
            individuals.append(individual)

        return Population(individuals)

    async def _evaluate_population(self, population: Population) -> None:
        """Evaluate fitness of all individuals in population."""
        self.logger.debug(f"Evaluating {len(population.individuals)} individuals")

        if self.config.parallel_evaluation:
            # Parallel evaluation
            tasks = []
            for individual in population.individuals:
                if individual.fitness == 0.0:  # Not evaluated yet
                    task = self._evaluate_individual(individual)
                    tasks.append(task)

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Handle results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Evaluation failed for individual: {result!s}")
                        population.individuals[i].fitness = -float("inf")
        else:
            # Sequential evaluation
            for individual in population.individuals:
                if individual.fitness == 0.0:
                    await self._evaluate_individual(individual)

    async def _evaluate_individual(self, individual: Individual) -> None:
        """Evaluate fitness of a single individual."""
        try:
            # Create strategy with individual's parameters
            strategy = self.strategy_class(**individual.genes)

            # Run backtest
            engine = BacktestEngine(
                config=self.backtest_config,
                strategy=strategy,
            )

            result = await engine.run()

            # Calculate fitness
            fitness = self.fitness_evaluator.evaluate(result)
            individual.fitness = fitness

            # Store additional metrics
            individual.metadata.update(
                {
                    "sharpe_ratio": result.sharpe_ratio,
                    "total_return": float(result.total_return),
                    "max_drawdown": float(result.max_drawdown),
                    "win_rate": result.win_rate,
                }
            )

        except Exception as e:
            self.logger.error(f"Failed to evaluate individual: {e!s}")
            individual.fitness = -float("inf")

    def _selection(self, population: Population) -> list[Individual]:
        """Select parents for reproduction."""
        parents = []

        # Elite selection
        num_elite = int(self.config.population_size * self.config.elitism_rate)
        elite = population.get_top_n(num_elite)
        parents.extend(elite)

        # Tournament selection for remaining
        while len(parents) < self.config.population_size:
            winner = self._tournament_selection(population)
            parents.append(winner)

        return parents

    def _tournament_selection(self, population: Population) -> Individual:
        """Perform tournament selection."""
        tournament = random.sample(
            population.individuals,
            min(self.config.tournament_size, len(population.individuals)),
        )

        return max(tournament, key=lambda x: x.fitness)

    def _create_offspring(self, parents: list[Individual]) -> Population:
        """Create offspring through crossover and mutation."""
        offspring = []

        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            # Crossover
            if random.random() < self.config.crossover_rate:
                child1_genes, child2_genes = self.crossover_operator.crossover(
                    parent1.genes, parent2.genes
                )
            else:
                child1_genes = parent1.genes.copy()
                child2_genes = parent2.genes.copy()

            # Mutation
            child1_genes = self.mutation_operator.mutate(child1_genes, self.parameter_ranges)
            child2_genes = self.mutation_operator.mutate(child2_genes, self.parameter_ranges)

            # Create offspring individuals
            child1 = Individual(
                id=f"gen{self.generation + 1}_ind{i}",
                genes=child1_genes,
                fitness=0.0,
                metadata={"generation": self.generation + 1},
            )

            child2 = Individual(
                id=f"gen{self.generation + 1}_ind{i + 1}",
                genes=child2_genes,
                fitness=0.0,
                metadata={"generation": self.generation + 1},
            )

            offspring.extend([child1, child2])

        return Population(offspring[: self.config.population_size])

    def _replacement(self, population: Population, offspring: Population) -> Population:
        """Replace population with offspring using elitism."""
        # Keep elite from current population
        num_elite = int(self.config.population_size * self.config.elitism_rate)
        elite = population.get_top_n(num_elite)

        # Fill rest with best offspring
        remaining = self.config.population_size - num_elite
        best_offspring = offspring.get_top_n(remaining)

        # Combine
        new_individuals = elite + best_offspring

        return Population(new_individuals)

    def _calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if not self.population or len(self.population.individuals) < 2:
            return 0.0

        # Calculate pairwise distances
        distances = []
        individuals = self.population.individuals

        for i in range(len(individuals)):
            for j in range(i + 1, len(individuals)):
                distance = self._gene_distance(individuals[i].genes, individuals[j].genes)
                distances.append(distance)

        # Return average distance as diversity measure
        return np.mean(distances) if distances else 0.0

    def _gene_distance(self, genes1: dict[str, Any], genes2: dict[str, Any]) -> float:
        """Calculate distance between two gene sets."""
        distance = 0.0
        num_params = 0

        for param in genes1:
            if param in genes2:
                val1 = genes1[param]
                val2 = genes2[param]

                # Check if this is a categorical parameter
                param_range = self.parameter_ranges.get(param, (0, 1))
                is_categorical = len(param_range) == 2 and param_range[1] is None

                if (
                    not is_categorical
                    and isinstance(val1, int | float)
                    and not isinstance(val1, bool)
                ):
                    # Numeric distance (normalized)
                    range_size = param_range[1] - param_range[0]
                    if range_size > 0:
                        distance += abs(val1 - val2) / range_size
                else:
                    # Categorical distance (including boolean)
                    distance += 0 if val1 == val2 else 1

                num_params += 1

        return distance / num_params if num_params > 0 else 0.0

    def _inject_diversity(self) -> None:
        """Inject diversity into population."""
        self.logger.info("Injecting diversity into population")

        # Replace bottom 20% with random individuals
        num_replace = int(self.config.population_size * 0.2)
        bottom_indices = self.population.get_bottom_n_indices(num_replace)

        for idx in bottom_indices:
            # Generate random individual
            genes = {}
            for param, (min_val, max_val) in self.parameter_ranges.items():
                if isinstance(min_val, int | float):
                    if isinstance(min_val, int):
                        value = random.randint(min_val, max_val)
                    else:
                        value = random.uniform(min_val, max_val)
                else:
                    value = random.choice([min_val, max_val])

                genes[param] = value

            self.population.individuals[idx] = Individual(
                id=f"gen{self.generation}_div{idx}",
                genes=genes,
                fitness=0.0,
                metadata={"generation": self.generation, "injected": True},
            )

    def _record_generation_stats(self) -> None:
        """Record statistics for current generation."""
        if not self.population:
            return

        fitnesses = [ind.fitness for ind in self.population.individuals]

        stats = {
            "generation": self.generation,
            "best_fitness": max(fitnesses),
            "avg_fitness": np.mean(fitnesses),
            "std_fitness": np.std(fitnesses),
            "diversity": self._calculate_diversity(),
            "timestamp": datetime.now(timezone.utc),
        }

        self.evolution_history.append(stats)

    def get_evolution_summary(self) -> dict[str, Any]:
        """Get summary of evolution process."""
        if not self.evolution_history:
            return {}

        return {
            "generations_run": len(self.evolution_history),
            "best_fitness": self.best_individual.fitness if self.best_individual else 0,
            "best_parameters": self.best_individual.genes if self.best_individual else {},
            "fitness_progression": [h["best_fitness"] for h in self.evolution_history],
            "diversity_progression": [h["diversity"] for h in self.evolution_history],
            "final_population_stats": {
                "avg_fitness": (
                    self.evolution_history[-1]["avg_fitness"] if self.evolution_history else 0
                ),
                "std_fitness": (
                    self.evolution_history[-1]["std_fitness"] if self.evolution_history else 0
                ),
            },
        }
