"""
Multi-Objective Optimization for Evolutionary Trading Strategies.

This module implements NSGA-II (Non-dominated Sorting Genetic Algorithm II) and
multi-objective optimization for trading strategy evolution. It optimizes multiple
conflicting objectives simultaneously while maintaining diversity through Pareto
frontier exploration.

Key Features:
- NSGA-II implementation with non-dominated sorting
- Pareto frontier calculation and maintenance
- Multiple optimization objectives (profit, risk, Sharpe ratio, drawdown, etc.)
- Crowding distance calculation for diversity preservation
- Constraint handling for trading constraints
- Visualization-ready Pareto frontier data
- Integration with existing evolutionary components

Dependencies:
- P-001: Core types, exceptions, logging
- P-002A: Error handling framework
- P-007A: Utility decorators
- P-013A-C: Existing evolutionary components
"""

import asyncio
import math
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from src.core.exceptions import OptimizationError, ValidationError
from src.core.logging import get_logger
from src.strategies.evolutionary.fitness import FitnessEvaluator
from src.strategies.evolutionary.population import Individual, Population
from src.utils.decorators import memory_usage, retry, time_execution

logger = get_logger(__name__)


class OptimizationObjective(BaseModel):
    """
    Optimization objective definition.

    Defines a single objective to optimize with its direction and constraints.
    """

    name: str = Field(description="Objective name")
    direction: str = Field(description="Optimization direction: 'maximize' or 'minimize'")
    weight: float = Field(default=1.0, ge=0.0, description="Objective weight")
    constraint_min: float | None = Field(default=None, description="Minimum constraint value")
    constraint_max: float | None = Field(default=None, description="Maximum constraint value")
    description: str = Field(default="", description="Objective description")

    def validate_direction(self) -> None:
        """Validate optimization direction."""
        if self.direction not in ["maximize", "minimize"]:
            raise ValidationError(f"Invalid direction: {self.direction}")


class MultiObjectiveConfig(BaseModel):
    """Configuration for multi-objective optimization."""

    objectives: list[OptimizationObjective] = Field(description="List of optimization objectives")
    population_size: int = Field(default=100, ge=10, description="Population size")
    generations: int = Field(default=50, ge=1, description="Number of generations")
    crossover_probability: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Crossover probability"
    )
    mutation_probability: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Mutation probability"
    )
    crowding_distance_threshold: float = Field(
        default=0.1, ge=0.0, description="Minimum crowding distance"
    )
    constraint_tolerance: float = Field(
        default=0.01, ge=0.0, description="Constraint violation tolerance"
    )
    elitism_ratio: float = Field(
        default=0.1, ge=0.0, le=0.5, description="Elite preservation ratio"
    )
    diversity_preservation: bool = Field(default=True, description="Enable diversity preservation")
    parallel_evaluation: bool = Field(
        default=True, description="Enable parallel fitness evaluation"
    )
    convergence_threshold: float = Field(default=1e-6, ge=0.0, description="Convergence threshold")
    max_stagnation_generations: int = Field(
        default=10, ge=1, description="Max generations without improvement"
    )


@dataclass
class ParetoSolution:
    """
    Represents a solution in the Pareto frontier.

    Contains the individual, objective values, and metadata.
    """

    individual: Individual
    objectives: dict[str, float]
    constraint_violations: dict[str, float]
    crowding_distance: float = 0.0
    rank: int = 0
    is_feasible: bool = True
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ConstraintHandler:
    """
    Handles constraints in multi-objective optimization.

    Implements constraint handling techniques for trading-specific constraints
    like maximum drawdown, minimum Sharpe ratio, etc.
    """

    def __init__(self, constraints: list[OptimizationObjective]):
        """
        Initialize constraint handler.

        Args:
            constraints: List of constraint objectives
        """
        self.constraints = constraints
        self.penalty_factor = 1000.0  # Penalty factor for constraint violations

        logger.info(
            "ConstraintHandler initialized",
            constraint_count=len(constraints),
            constraint_names=[c.name for c in constraints],
        )

    def evaluate_constraints(self, objectives: dict[str, float]) -> dict[str, float]:
        """
        Evaluate constraint violations.

        Args:
            objectives: Dictionary of objective values

        Returns:
            Dictionary of constraint violations
        """
        violations = {}

        for constraint in self.constraints:
            if constraint.name not in objectives:
                continue

            value = objectives[constraint.name]
            violation = 0.0

            # Check minimum constraint
            if constraint.constraint_min is not None:
                if value < constraint.constraint_min:
                    violation += constraint.constraint_min - value

            # Check maximum constraint
            if constraint.constraint_max is not None:
                if value > constraint.constraint_max:
                    violation += value - constraint.constraint_max

            violations[constraint.name] = violation

        return violations

    def is_feasible(self, objectives: dict[str, float], tolerance: float = 0.01) -> bool:
        """
        Check if solution is feasible (satisfies all constraints).

        Args:
            objectives: Dictionary of objective values
            tolerance: Constraint violation tolerance

        Returns:
            True if solution is feasible
        """
        violations = self.evaluate_constraints(objectives)
        total_violation = sum(violations.values())
        return total_violation <= tolerance

    def apply_penalty(
        self, objectives: dict[str, float], constraint_violations: dict[str, float]
    ) -> dict[str, float]:
        """
        Apply penalty method for constraint violations.

        Args:
            objectives: Original objective values
            constraint_violations: Constraint violation amounts

        Returns:
            Penalized objective values
        """
        penalized = objectives.copy()
        total_violation = sum(constraint_violations.values())

        if total_violation > 0:
            # Apply penalty to all objectives
            for obj_name in penalized:
                # For maximization objectives, subtract penalty
                # For minimization objectives, add penalty
                penalty = self.penalty_factor * total_violation

                # Find objective direction
                direction = "maximize"  # Default
                for constraint in self.constraints:
                    if constraint.name == obj_name:
                        direction = constraint.direction
                        break

                if direction == "maximize":
                    penalized[obj_name] -= penalty
                else:
                    penalized[obj_name] += penalty

        return penalized


class DominanceComparator:
    """
    Implements dominance comparison for multi-objective optimization.

    Provides methods to compare solutions based on Pareto dominance.
    """

    def __init__(self, objectives: list[OptimizationObjective]):
        """
        Initialize dominance comparator.

        Args:
            objectives: List of optimization objectives
        """
        self.objectives = {obj.name: obj for obj in objectives}

        logger.debug("DominanceComparator initialized", objective_count=len(objectives))

    def dominates(self, solution1: dict[str, float], solution2: dict[str, float]) -> bool:
        """
        Check if solution1 dominates solution2.

        Solution1 dominates solution2 if:
        1. Solution1 is at least as good as solution2 in all objectives
        2. Solution1 is strictly better than solution2 in at least one objective

        Args:
            solution1: First solution's objective values
            solution2: Second solution's objective values

        Returns:
            True if solution1 dominates solution2
        """
        at_least_as_good = True
        strictly_better = False

        for obj_name, obj_config in self.objectives.items():
            if obj_name not in solution1 or obj_name not in solution2:
                continue

            val1 = solution1[obj_name]
            val2 = solution2[obj_name]

            if obj_config.direction == "maximize":
                if val1 < val2:
                    at_least_as_good = False
                    break
                elif val1 > val2:
                    strictly_better = True
            else:  # minimize
                if val1 > val2:
                    at_least_as_good = False
                    break
                elif val1 < val2:
                    strictly_better = True

        return at_least_as_good and strictly_better

    def non_dominated_sort(self, solutions: list[dict[str, float]]) -> list[list[int]]:
        """
        Perform non-dominated sorting on solutions.

        Returns fronts where front[0] contains indices of non-dominated solutions,
        front[1] contains solutions dominated only by front[0], etc.

        Args:
            solutions: List of solution objective values

        Returns:
            List of fronts (lists of solution indices)
        """
        n = len(solutions)
        domination_count = [0] * n  # Number of solutions that dominate each solution
        dominated_solutions = [[] for _ in range(n)]  # Solutions dominated by each solution
        fronts = [[]]

        # Calculate domination relationships
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self.dominates(solutions[i], solutions[j]):
                        dominated_solutions[i].append(j)
                    elif self.dominates(solutions[j], solutions[i]):
                        domination_count[i] += 1

            # If solution is not dominated by any other, it belongs to first front
            if domination_count[i] == 0:
                fronts[0].append(i)

        # Build subsequent fronts
        front_index = 0
        while len(fronts[front_index]) > 0:
            next_front = []

            for i in fronts[front_index]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)

            if next_front:
                fronts.append(next_front)
            front_index += 1

        # Remove empty fronts
        fronts = [front for front in fronts if front]

        return fronts


class CrowdingDistanceCalculator:
    """
    Calculates crowding distance for diversity preservation.

    Crowding distance measures the density of solutions around a particular
    solution in the objective space. Higher crowding distance indicates
    less crowded regions.
    """

    def __init__(self, objectives: list[OptimizationObjective]):
        """
        Initialize crowding distance calculator.

        Args:
            objectives: List of optimization objectives
        """
        self.objectives = objectives

        logger.debug("CrowdingDistanceCalculator initialized", objective_count=len(objectives))

    def calculate_crowding_distance(
        self, solutions: list[dict[str, float]], front_indices: list[int]
    ) -> list[float]:
        """
        Calculate crowding distance for solutions in a front.

        Args:
            solutions: List of all solution objective values
            front_indices: Indices of solutions in the current front

        Returns:
            List of crowding distances for solutions in the front
        """
        if len(front_indices) <= 2:
            # Boundary solutions get infinite distance
            return [float("inf")] * len(front_indices)

        distances = [0.0] * len(front_indices)

        # Calculate distance for each objective
        for objective in self.objectives:
            obj_name = objective.name

            # Skip if objective not present in solutions
            if not all(obj_name in solutions[i] for i in front_indices):
                continue

            # Sort solutions by objective value
            sorted_indices = sorted(
                range(len(front_indices)), key=lambda i: solutions[front_indices[i]][obj_name]
            )

            # Get objective values
            obj_values = [solutions[front_indices[i]][obj_name] for i in sorted_indices]

            # Boundary solutions get infinite distance
            distances[sorted_indices[0]] = float("inf")
            distances[sorted_indices[-1]] = float("inf")

            # Calculate range
            obj_range = obj_values[-1] - obj_values[0]
            if obj_range == 0:
                continue

            # Calculate crowding distance for intermediate solutions
            for i in range(1, len(sorted_indices) - 1):
                distances[sorted_indices[i]] += (obj_values[i + 1] - obj_values[i - 1]) / obj_range

        return distances


class ParetoFrontierManager:
    """
    Manages the Pareto frontier and provides analysis tools.

    Maintains the current Pareto frontier, tracks convergence, and provides
    methods for analyzing the frontier properties.
    """

    def __init__(self, config: MultiObjectiveConfig):
        """
        Initialize Pareto frontier manager.

        Args:
            config: Multi-objective optimization configuration
        """
        self.config = config
        self.current_frontier: list[ParetoSolution] = []
        self.frontier_history: list[list[ParetoSolution]] = []
        self.convergence_metrics: list[dict[str, float]] = []

        # Initialize components
        self.dominance_comparator = DominanceComparator(config.objectives)
        self.crowding_calculator = CrowdingDistanceCalculator(config.objectives)
        self.constraint_handler = ConstraintHandler(config.objectives)

        logger.info("ParetoFrontierManager initialized", objective_count=len(config.objectives))

    @time_execution
    def update_frontier(self, solutions: list[ParetoSolution]) -> None:
        """
        Update the Pareto frontier with new solutions.

        Args:
            solutions: New solutions to consider for the frontier
        """
        # Combine current frontier with new solutions
        all_solutions = self.current_frontier + solutions

        if not all_solutions:
            return

        # Extract objective values
        objective_values = [sol.objectives for sol in all_solutions]

        # Perform non-dominated sorting
        fronts = self.dominance_comparator.non_dominated_sort(objective_values)

        if not fronts or not fronts[0]:
            return

        # Update frontier with first front (non-dominated solutions)
        new_frontier = []
        for idx in fronts[0]:
            solution = all_solutions[idx]
            solution.rank = 0
            new_frontier.append(solution)

        # Calculate crowding distances
        if len(new_frontier) > 1:
            distances = self.crowding_calculator.calculate_crowding_distance(
                objective_values, fronts[0]
            )
            for i, solution in enumerate(new_frontier):
                solution.crowding_distance = distances[i]

        self.current_frontier = new_frontier
        self.frontier_history.append(self.current_frontier.copy())

        # Calculate convergence metrics
        self._calculate_convergence_metrics()

        logger.info(
            "Pareto frontier updated",
            frontier_size=len(self.current_frontier),
            total_solutions=len(all_solutions),
        )

    def _calculate_convergence_metrics(self) -> None:
        """Calculate convergence metrics for the frontier."""
        if len(self.frontier_history) < 2:
            return

        current = self.current_frontier
        previous = self.frontier_history[-2] if len(self.frontier_history) >= 2 else []

        metrics = {
            "frontier_size": len(current),
            "hypervolume": self._calculate_hypervolume(current),
            "spread": self._calculate_spread(current),
            "convergence": self._calculate_convergence(current, previous),
            "timestamp": datetime.now().isoformat(),
        }

        self.convergence_metrics.append(metrics)

    def _calculate_hypervolume(self, solutions: list[ParetoSolution]) -> float:
        """
        Calculate hypervolume indicator.

        Simplified hypervolume calculation using reference point method.

        Args:
            solutions: Solutions to calculate hypervolume for

        Returns:
            Hypervolume value
        """
        if not solutions:
            return 0.0

        # Use simple reference point approach
        # In practice, more sophisticated methods would be used
        objective_names = [obj.name for obj in self.config.objectives]

        if not objective_names:
            return 0.0

        # Calculate volume contribution of each solution
        total_volume = 0.0

        for solution in solutions:
            volume = 1.0
            for obj_name in objective_names:
                if obj_name in solution.objectives:
                    # Normalize objective value (simplified)
                    value = abs(solution.objectives[obj_name])
                    volume *= max(value, 0.001)  # Avoid zero volume

            total_volume += volume

        return total_volume

    def _calculate_spread(self, solutions: list[ParetoSolution]) -> float:
        """
        Calculate spread (diversity) of solutions.

        Args:
            solutions: Solutions to calculate spread for

        Returns:
            Spread value
        """
        if len(solutions) < 2:
            return 0.0

        # Calculate average crowding distance
        distances = [
            sol.crowding_distance for sol in solutions if not math.isinf(sol.crowding_distance)
        ]

        if not distances:
            return 0.0

        return np.mean(distances)

    def _calculate_convergence(
        self, current: list[ParetoSolution], previous: list[ParetoSolution]
    ) -> float:
        """
        Calculate convergence metric comparing current and previous frontiers.

        Args:
            current: Current frontier solutions
            previous: Previous frontier solutions

        Returns:
            Convergence metric (lower is better)
        """
        if not current or not previous:
            return float("inf")

        # Calculate average distance between frontiers
        total_distance = 0.0
        count = 0

        for curr_sol in current:
            min_distance = float("inf")

            for prev_sol in previous:
                distance = self._solution_distance(curr_sol, prev_sol)
                min_distance = min(min_distance, distance)

            if not math.isinf(min_distance):
                total_distance += min_distance
                count += 1

        return total_distance / count if count > 0 else float("inf")

    def _solution_distance(self, sol1: ParetoSolution, sol2: ParetoSolution) -> float:
        """
        Calculate Euclidean distance between two solutions in objective space.

        Args:
            sol1: First solution
            sol2: Second solution

        Returns:
            Distance between solutions
        """
        distance = 0.0
        count = 0

        for obj_name in sol1.objectives:
            if obj_name in sol2.objectives:
                diff = sol1.objectives[obj_name] - sol2.objectives[obj_name]
                distance += diff * diff
                count += 1

        return math.sqrt(distance) if count > 0 else float("inf")

    def get_frontier_summary(self) -> dict[str, Any]:
        """
        Get summary statistics of the current Pareto frontier.

        Returns:
            Dictionary containing frontier summary
        """
        if not self.current_frontier:
            return {"frontier_size": 0, "objectives": {}}

        # Calculate objective statistics
        objective_stats = {}
        objective_names = [obj.name for obj in self.config.objectives]

        for obj_name in objective_names:
            values = []
            for solution in self.current_frontier:
                if obj_name in solution.objectives:
                    values.append(solution.objectives[obj_name])

            if values:
                objective_stats[obj_name] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                }

        return {
            "frontier_size": len(self.current_frontier),
            "objectives": objective_stats,
            "average_crowding_distance": (
                np.mean(
                    [
                        sol.crowding_distance
                        for sol in self.current_frontier
                        if not math.isinf(sol.crowding_distance)
                    ]
                )
                if self.current_frontier
                else 0.0
            ),
            "feasible_solutions": sum(1 for sol in self.current_frontier if sol.is_feasible),
            "convergence_metrics": self.convergence_metrics[-1] if self.convergence_metrics else {},
        }


class NSGAIIOptimizer:
    """
    NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation.

    Implements the complete NSGA-II algorithm for multi-objective optimization
    of trading strategies with constraint handling and diversity preservation.
    """

    def __init__(
        self,
        config: MultiObjectiveConfig,
        fitness_evaluator: FitnessEvaluator,
        strategy_class: type,
        parameter_ranges: dict[str, tuple[Any, Any]],
    ):
        """
        Initialize NSGA-II optimizer.

        Args:
            config: Multi-objective optimization configuration
            fitness_evaluator: Fitness evaluation function
            strategy_class: Strategy class to optimize
            parameter_ranges: Parameter ranges for optimization
        """
        self.config = config
        self.fitness_evaluator = fitness_evaluator
        self.strategy_class = strategy_class
        self.parameter_ranges = parameter_ranges

        # Initialize components
        self.frontier_manager = ParetoFrontierManager(config)
        self.population: Population | None = None
        self.generation = 0
        self.stagnation_count = 0

        # Evolution history
        self.evolution_history: list[dict[str, Any]] = []

        logger.info(
            "NSGAIIOptimizer initialized",
            strategy=strategy_class.__name__,
            objective_count=len(config.objectives),
            population_size=config.population_size,
        )

    @time_execution
    @memory_usage
    async def optimize(self) -> list[ParetoSolution]:
        """
        Run the NSGA-II optimization algorithm.

        Returns:
            List of Pareto optimal solutions
        """
        try:
            logger.info(
                "Starting NSGA-II optimization",
                generations=self.config.generations,
                population_size=self.config.population_size,
            )

            # Initialize population
            self.population = await self._initialize_population()

            # Evaluate initial population
            solutions = await self._evaluate_population(self.population)
            self.frontier_manager.update_frontier(solutions)

            # Evolution loop
            for generation in range(self.config.generations):
                self.generation = generation

                logger.info(
                    f"Generation {generation + 1}/{self.config.generations}",
                    frontier_size=len(self.frontier_manager.current_frontier),
                )

                # Create offspring population
                offspring_population = await self._create_offspring()

                # Evaluate offspring
                offspring_solutions = await self._evaluate_population(offspring_population)

                # Combine parent and offspring populations
                combined_solutions = solutions + offspring_solutions

                # Environmental selection
                solutions = self._environmental_selection(combined_solutions)

                # Update Pareto frontier
                self.frontier_manager.update_frontier(solutions)

                # Record generation statistics
                self._record_generation_stats()

                # Check convergence
                if self._check_convergence():
                    logger.info(
                        "Optimization converged",
                        generation=generation + 1,
                        stagnation_count=self.stagnation_count,
                    )
                    break

            logger.info(
                "NSGA-II optimization completed",
                final_frontier_size=len(self.frontier_manager.current_frontier),
                generations_run=self.generation + 1,
            )

            return self.frontier_manager.current_frontier

        except Exception as e:
            logger.error("NSGA-II optimization failed", error=str(e))
            raise OptimizationError(f"NSGA-II optimization failed: {e!s}")

    async def _initialize_population(self) -> Population:
        """Initialize random population."""
        individuals = []

        for i in range(self.config.population_size):
            # Generate random parameters
            genes = {}
            for param, (min_val, max_val) in self.parameter_ranges.items():
                if isinstance(min_val, int | float):
                    if isinstance(min_val, int):
                        value = random.randint(min_val, max_val)
                    else:
                        value = random.uniform(min_val, max_val)
                elif isinstance(min_val, list):
                    value = random.choice(min_val)
                else:
                    value = random.choice([min_val, max_val])

                genes[param] = value

            individual = Individual(
                id=f"gen0_ind{i}", genes=genes, fitness=0.0, metadata={"generation": 0}
            )
            individuals.append(individual)

        return Population(individuals)

    async def _evaluate_population(self, population: Population) -> list[ParetoSolution]:
        """
        Evaluate population and create Pareto solutions.

        Args:
            population: Population to evaluate

        Returns:
            List of evaluated Pareto solutions
        """
        solutions = []

        if self.config.parallel_evaluation:
            # Parallel evaluation
            tasks = []
            for individual in population.individuals:
                task = self._evaluate_individual(individual)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        "Individual evaluation failed",
                        individual_id=population.individuals[i].id,
                        error=str(result),
                    )
                    continue

                if result:
                    solutions.append(result)
        else:
            # Sequential evaluation
            for individual in population.individuals:
                solution = await self._evaluate_individual(individual)
                if solution:
                    solutions.append(solution)

        return solutions

    async def _evaluate_individual(self, individual: Individual) -> ParetoSolution | None:
        """
        Evaluate a single individual and create Pareto solution.

        Args:
            individual: Individual to evaluate

        Returns:
            Pareto solution or None if evaluation fails
        """
        try:
            # Create strategy with individual's parameters
            self.strategy_class(**individual.genes)

            # This would be replaced with actual backtesting
            # For now, we'll simulate objective values
            objectives = await self._simulate_objectives(individual)

            # Evaluate constraints
            constraint_violations = self.frontier_manager.constraint_handler.evaluate_constraints(
                objectives
            )
            is_feasible = self.frontier_manager.constraint_handler.is_feasible(
                objectives, self.config.constraint_tolerance
            )

            return ParetoSolution(
                individual=individual,
                objectives=objectives,
                constraint_violations=constraint_violations,
                is_feasible=is_feasible,
                metadata={"evaluation_timestamp": datetime.now()},
            )

        except Exception as e:
            logger.error("Individual evaluation failed", individual_id=individual.id, error=str(e))
            return None

    async def _simulate_objectives(self, individual: Individual) -> dict[str, float]:
        """
        Simulate objective values for testing.

        In a real implementation, this would run backtesting and extract
        objective values from the results.

        Args:
            individual: Individual to evaluate

        Returns:
            Dictionary of objective values
        """
        # Simulate realistic trading strategy objectives
        objectives = {}

        for objective in self.config.objectives:
            if objective.name == "total_return":
                # Simulate return between -50% to 200%
                objectives[objective.name] = random.uniform(-0.5, 2.0)
            elif objective.name == "sharpe_ratio":
                # Simulate Sharpe ratio between -2 to 4
                objectives[objective.name] = random.uniform(-2.0, 4.0)
            elif objective.name == "max_drawdown":
                # Simulate drawdown between 0% to 60%
                objectives[objective.name] = random.uniform(0.0, 0.6)
            elif objective.name == "win_rate":
                # Simulate win rate between 20% to 80%
                objectives[objective.name] = random.uniform(0.2, 0.8)
            elif objective.name == "profit_factor":
                # Simulate profit factor between 0.5 to 3.0
                objectives[objective.name] = random.uniform(0.5, 3.0)
            elif objective.name == "volatility":
                # Simulate volatility between 5% to 50%
                objectives[objective.name] = random.uniform(0.05, 0.5)
            else:
                # Default random value
                objectives[objective.name] = random.uniform(0.0, 1.0)

        return objectives

    async def _create_offspring(self) -> Population:
        """
        Create offspring population through selection, crossover, and mutation.

        Returns:
            Offspring population
        """
        offspring = []

        # Tournament selection and reproduction
        for _ in range(self.config.population_size):
            # Select parents using tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            # Create offspring
            if random.random() < self.config.crossover_probability:
                child_genes = self._crossover(parent1.individual.genes, parent2.individual.genes)
            else:
                child_genes = parent1.individual.genes.copy()

            # Apply mutation
            if random.random() < self.config.mutation_probability:
                child_genes = self._mutate(child_genes)

            # Create offspring individual
            child = Individual(
                id=f"gen{self.generation + 1}_off{len(offspring)}",
                genes=child_genes,
                fitness=0.0,
                metadata={"generation": self.generation + 1},
            )
            offspring.append(child)

        return Population(offspring)

    def _tournament_selection(self) -> ParetoSolution:
        """
        Perform tournament selection based on Pareto rank and crowding distance.

        Returns:
            Selected Pareto solution
        """
        tournament_size = min(3, len(self.frontier_manager.current_frontier))
        if tournament_size == 0:
            # Fallback: create random solution
            return self._create_random_solution()

        tournament = random.sample(self.frontier_manager.current_frontier, tournament_size)

        # Select best based on rank and crowding distance
        best = min(tournament, key=lambda x: (x.rank, -x.crowding_distance))
        return best

    def _create_random_solution(self) -> ParetoSolution:
        """Create a random solution as fallback."""
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

        individual = Individual(
            id=f"random_{random.randint(1000, 9999)}", genes=genes, fitness=0.0, metadata={}
        )

        return ParetoSolution(
            individual=individual,
            objectives={obj.name: 0.0 for obj in self.config.objectives},
            constraint_violations={},
            is_feasible=True,
        )

    def _crossover(self, genes1: dict[str, Any], genes2: dict[str, Any]) -> dict[str, Any]:
        """
        Perform uniform crossover between two gene sets.

        Args:
            genes1: First parent's genes
            genes2: Second parent's genes

        Returns:
            Offspring genes
        """
        offspring_genes = {}

        for param in genes1:
            if param in genes2:
                # Uniform crossover
                if random.random() < 0.5:
                    offspring_genes[param] = genes1[param]
                else:
                    offspring_genes[param] = genes2[param]
            else:
                offspring_genes[param] = genes1[param]

        return offspring_genes

    def _mutate(self, genes: dict[str, Any]) -> dict[str, Any]:
        """
        Apply mutation to genes.

        Args:
            genes: Genes to mutate

        Returns:
            Mutated genes
        """
        mutated_genes = genes.copy()

        for param, value in genes.items():
            if param in self.parameter_ranges:
                min_val, max_val = self.parameter_ranges[param]

                if isinstance(value, int | float):
                    # Gaussian mutation for numeric parameters
                    if isinstance(value, int):
                        mutation = random.gauss(0, (max_val - min_val) * 0.1)
                        new_value = int(value + mutation)
                        mutated_genes[param] = max(min_val, min(max_val, new_value))
                    else:
                        mutation = random.gauss(0, (max_val - min_val) * 0.1)
                        new_value = value + mutation
                        mutated_genes[param] = max(min_val, min(max_val, new_value))
                elif isinstance(min_val, list):
                    # Random mutation for categorical parameters
                    mutated_genes[param] = random.choice(min_val)

        return mutated_genes

    def _environmental_selection(self, solutions: list[ParetoSolution]) -> list[ParetoSolution]:
        """
        Perform environmental selection to maintain population size.

        Args:
            solutions: Combined parent and offspring solutions

        Returns:
            Selected solutions for next generation
        """
        if len(solutions) <= self.config.population_size:
            return solutions

        # Extract objective values for sorting
        objective_values = [sol.objectives for sol in solutions]

        # Perform non-dominated sorting
        fronts = self.frontier_manager.dominance_comparator.non_dominated_sort(objective_values)

        selected = []

        # Add complete fronts until population size is reached
        for front_indices in fronts:
            if len(selected) + len(front_indices) <= self.config.population_size:
                # Add entire front
                for idx in front_indices:
                    solutions[idx].rank = len(selected) // len(front_indices)
                    selected.append(solutions[idx])
            else:
                # Add partial front based on crowding distance
                remaining = self.config.population_size - len(selected)

                # Calculate crowding distances for this front
                distances = self.frontier_manager.crowding_calculator.calculate_crowding_distance(
                    objective_values, front_indices
                )

                # Sort by crowding distance (descending)
                sorted_indices = sorted(
                    range(len(front_indices)), key=lambda i: distances[i], reverse=True
                )

                # Add best solutions from this front
                for i in range(remaining):
                    idx = front_indices[sorted_indices[i]]
                    solutions[idx].rank = len(selected) // remaining
                    solutions[idx].crowding_distance = distances[sorted_indices[i]]
                    selected.append(solutions[idx])

                break

        return selected

    def _check_convergence(self) -> bool:
        """
        Check if optimization has converged.

        Returns:
            True if converged
        """
        if len(self.frontier_manager.convergence_metrics) < 2:
            return False

        current_metrics = self.frontier_manager.convergence_metrics[-1]
        previous_metrics = self.frontier_manager.convergence_metrics[-2]

        # Check if convergence metric improved
        current_conv = current_metrics.get("convergence", float("inf"))
        previous_conv = previous_metrics.get("convergence", float("inf"))

        if current_conv >= previous_conv - self.config.convergence_threshold:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0

        return self.stagnation_count >= self.config.max_stagnation_generations

    def _record_generation_stats(self) -> None:
        """Record statistics for current generation."""
        frontier_summary = self.frontier_manager.get_frontier_summary()

        stats = {
            "generation": self.generation,
            "frontier_size": frontier_summary["frontier_size"],
            "feasible_solutions": frontier_summary["feasible_solutions"],
            "average_crowding_distance": frontier_summary["average_crowding_distance"],
            "objective_stats": frontier_summary["objectives"],
            "timestamp": datetime.now().isoformat(),
        }

        self.evolution_history.append(stats)

    def get_optimization_summary(self) -> dict[str, Any]:
        """
        Get summary of optimization process.

        Returns:
            Dictionary containing optimization summary
        """
        frontier_summary = self.frontier_manager.get_frontier_summary()

        return {
            "generations_run": self.generation + 1,
            "final_frontier_size": frontier_summary["frontier_size"],
            "objectives_optimized": [obj.name for obj in self.config.objectives],
            "convergence_achieved": self.stagnation_count >= self.config.max_stagnation_generations,
            "frontier_summary": frontier_summary,
            "evolution_history": self.evolution_history,
            "pareto_solutions": [
                {
                    "individual_id": sol.individual.id,
                    "objectives": sol.objectives,
                    "is_feasible": sol.is_feasible,
                    "rank": sol.rank,
                    "crowding_distance": sol.crowding_distance,
                }
                for sol in self.frontier_manager.current_frontier
            ],
        }


class MultiObjectiveOptimizer:
    """
    Main interface for multi-objective optimization of trading strategies.

    Provides high-level interface for running multi-objective optimization
    with NSGA-II and other algorithms.
    """

    def __init__(self, config: MultiObjectiveConfig):
        """
        Initialize multi-objective optimizer.

        Args:
            config: Multi-objective optimization configuration
        """
        self.config = config
        self.optimizer: NSGAIIOptimizer | None = None

        # Validate configuration
        for objective in config.objectives:
            objective.validate_direction()

        logger.info(
            "MultiObjectiveOptimizer initialized",
            objective_count=len(config.objectives),
            algorithm="NSGA-II",
        )

    @time_execution
    @retry(max_attempts=3, exceptions=(OptimizationError,))
    async def optimize_strategy(
        self,
        strategy_class: type,
        parameter_ranges: dict[str, tuple[Any, Any]],
        fitness_evaluator: FitnessEvaluator,
    ) -> list[ParetoSolution]:
        """
        Optimize trading strategy using multi-objective optimization.

        Args:
            strategy_class: Strategy class to optimize
            parameter_ranges: Parameter ranges for optimization
            fitness_evaluator: Fitness evaluation function

        Returns:
            List of Pareto optimal solutions
        """
        try:
            logger.info(
                "Starting multi-objective strategy optimization",
                strategy=strategy_class.__name__,
                parameter_count=len(parameter_ranges),
                objective_count=len(self.config.objectives),
            )

            # Initialize NSGA-II optimizer
            self.optimizer = NSGAIIOptimizer(
                config=self.config,
                fitness_evaluator=fitness_evaluator,
                strategy_class=strategy_class,
                parameter_ranges=parameter_ranges,
            )

            # Run optimization
            pareto_solutions = await self.optimizer.optimize()

            logger.info(
                "Multi-objective optimization completed",
                pareto_solutions_count=len(pareto_solutions),
                strategy=strategy_class.__name__,
            )

            return pareto_solutions

        except Exception as e:
            logger.error(
                "Multi-objective optimization failed",
                strategy=strategy_class.__name__,
                error=str(e),
            )
            raise OptimizationError(f"Multi-objective optimization failed: {e!s}")

    def get_pareto_frontier_data(self) -> dict[str, Any]:
        """
        Get Pareto frontier data for visualization.

        Returns:
            Dictionary containing frontier data suitable for plotting
        """
        if not self.optimizer:
            return {"error": "No optimization has been run"}

        frontier = self.optimizer.frontier_manager.current_frontier

        if not frontier:
            return {"error": "No Pareto frontier available"}

        # Prepare data for visualization
        objective_names = [obj.name for obj in self.config.objectives]

        data = {
            "objective_names": objective_names,
            "solutions": [],
            "frontier_size": len(frontier),
            "summary": self.optimizer.get_optimization_summary(),
        }

        for solution in frontier:
            solution_data = {
                "id": solution.individual.id,
                "objectives": solution.objectives,
                "parameters": solution.individual.genes,
                "is_feasible": solution.is_feasible,
                "rank": solution.rank,
                "crowding_distance": solution.crowding_distance,
                "constraint_violations": solution.constraint_violations,
            }
            data["solutions"].append(solution_data)

        return data

    def export_results(self, filepath: str) -> None:
        """
        Export optimization results to file.

        Args:
            filepath: Path to export file
        """
        import json

        if not self.optimizer:
            raise ValueError("No optimization has been run")

        results = {
            "config": self.config.dict(),
            "summary": self.optimizer.get_optimization_summary(),
            "pareto_frontier": self.get_pareto_frontier_data(),
        }

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("Optimization results exported", filepath=filepath)


# Example usage and factory functions
def create_trading_objectives() -> list[OptimizationObjective]:
    """
    Create standard trading strategy optimization objectives.

    Returns:
        List of common trading objectives
    """
    return [
        OptimizationObjective(
            name="total_return",
            direction="maximize",
            weight=0.3,
            constraint_min=0.0,
            description="Total portfolio return",
        ),
        OptimizationObjective(
            name="sharpe_ratio",
            direction="maximize",
            weight=0.3,
            constraint_min=1.0,
            description="Risk-adjusted return (Sharpe ratio)",
        ),
        OptimizationObjective(
            name="max_drawdown",
            direction="minimize",
            weight=0.2,
            constraint_max=0.2,
            description="Maximum drawdown",
        ),
        OptimizationObjective(
            name="volatility",
            direction="minimize",
            weight=0.2,
            constraint_max=0.3,
            description="Portfolio volatility",
        ),
    ]


def create_default_config(
    objectives: list[OptimizationObjective] | None = None,
) -> MultiObjectiveConfig:
    """
    Create default multi-objective optimization configuration.

    Args:
        objectives: List of objectives (uses default if None)

    Returns:
        Default configuration
    """
    if objectives is None:
        objectives = create_trading_objectives()

    return MultiObjectiveConfig(
        objectives=objectives,
        population_size=50,
        generations=30,
        crossover_probability=0.9,
        mutation_probability=0.1,
        crowding_distance_threshold=0.1,
        constraint_tolerance=0.01,
        elitism_ratio=0.1,
        diversity_preservation=True,
        parallel_evaluation=True,
        convergence_threshold=1e-6,
        max_stagnation_generations=5,
    )
