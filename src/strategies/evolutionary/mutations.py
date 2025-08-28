"""
Mutation and Crossover Operators for Genetic Algorithms.

This module provides various mutation and crossover operators for evolving
trading strategy parameters.
"""

import random
from typing import Any

import numpy as np


class MutationOperator:
    """
    Mutation operators for genetic algorithms.

    Supports various mutation strategies for different parameter types.
    """

    def __init__(
        self,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.2,
        adaptive: bool = False,
    ):
        """
        Initialize mutation operator.

        Args:
            mutation_rate: Probability of mutation
            mutation_strength: Strength of mutations
            adaptive: Whether to use adaptive mutation
        """
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.adaptive = adaptive
        self.generation_count = 0

    def mutate(
        self,
        genes: dict[str, Any],
        parameter_ranges: dict[str, tuple[Any, Any]],
    ) -> dict[str, Any]:
        """
        Mutate genes based on parameter types.

        Args:
            genes: Current gene values
            parameter_ranges: Valid ranges for parameters

        Returns:
            Mutated genes
        """
        mutated = genes.copy()

        # Adaptive mutation rate
        if self.adaptive:
            current_rate = self._adaptive_rate()
        else:
            current_rate = self.mutation_rate

        for param, value in genes.items():
            if random.random() < current_rate:
                if param in parameter_ranges:
                    mutated[param] = self._mutate_parameter(value, parameter_ranges[param])

        return mutated

    def _mutate_parameter(self, value: Any, param_range: tuple[Any, Any]) -> Any:
        """Mutate a single parameter based on its type."""
        min_val, max_val = param_range

        if isinstance(value, bool):
            # Boolean mutation: flip
            return not value

        elif isinstance(value, int):
            # Integer mutation: add Gaussian noise
            std = (max_val - min_val) * self.mutation_strength
            noise = int(np.random.normal(0, std))
            new_value = value + noise
            return max(min_val, min(max_val, new_value))

        elif isinstance(value, float):
            # Float mutation: add Gaussian noise
            std = (max_val - min_val) * self.mutation_strength
            noise = np.random.normal(0, std)
            new_value = value + noise
            return max(min_val, min(max_val, new_value))

        elif isinstance(value, str):
            # Categorical mutation: random choice
            if isinstance(min_val, list):
                return random.choice(min_val)
            else:
                return value

        else:
            # Unknown type: no mutation
            return value

    def _adaptive_rate(self) -> float:
        """Calculate adaptive mutation rate."""
        # Decrease mutation rate over generations
        min_rate = self.mutation_rate * 0.1
        decay_factor = 0.95

        adaptive_rate = self.mutation_rate * (decay_factor**self.generation_count)
        return max(min_rate, adaptive_rate)

    def increment_generation(self) -> None:
        """Increment generation counter for adaptive mutation."""
        self.generation_count += 1


class CrossoverOperator:
    """
    Crossover operators for genetic algorithms.

    Provides various crossover strategies for combining parent genes.
    """

    def __init__(
        self,
        crossover_rate: float = 0.7,
        crossover_type: str = "uniform",
    ):
        """
        Initialize crossover operator.

        Args:
            crossover_rate: Probability of crossover
            crossover_type: Type of crossover (uniform, single_point, two_point, blend)
        """
        self.crossover_rate = crossover_rate
        self.crossover_type = crossover_type

    def crossover(
        self,
        parent1_genes: dict[str, Any],
        parent2_genes: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Perform crossover between two parents.

        Args:
            parent1_genes: First parent's genes
            parent2_genes: Second parent's genes

        Returns:
            Two offspring gene sets
        """
        if random.random() > self.crossover_rate:
            # No crossover
            return parent1_genes.copy(), parent2_genes.copy()

        if self.crossover_type == "uniform":
            return self._uniform_crossover(parent1_genes, parent2_genes)
        elif self.crossover_type == "single_point":
            return self._single_point_crossover(parent1_genes, parent2_genes)
        elif self.crossover_type == "two_point":
            return self._two_point_crossover(parent1_genes, parent2_genes)
        elif self.crossover_type == "blend":
            return self._blend_crossover(parent1_genes, parent2_genes)
        else:
            # Default to uniform
            return self._uniform_crossover(parent1_genes, parent2_genes)

    def _uniform_crossover(
        self,
        parent1_genes: dict[str, Any],
        parent2_genes: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Uniform crossover: randomly select from each parent."""
        child1_genes: dict[str, Any] = {}
        child2_genes: dict[str, Any] = {}

        for param in parent1_genes:
            if param in parent2_genes:
                if random.random() < 0.5:
                    child1_genes[param] = parent1_genes[param]
                    child2_genes[param] = parent2_genes[param]
                else:
                    child1_genes[param] = parent2_genes[param]
                    child2_genes[param] = parent1_genes[param]
            else:
                child1_genes[param] = parent1_genes[param]
                child2_genes[param] = parent1_genes[param]

        return child1_genes, child2_genes

    def _single_point_crossover(
        self,
        parent1_genes: dict[str, Any],
        parent2_genes: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Single-point crossover: split at random point."""
        params = list(parent1_genes.keys())

        if len(params) <= 1:
            return parent1_genes.copy(), parent2_genes.copy()

        # Random crossover point
        crossover_point = random.randint(1, len(params) - 1)

        child1_genes: dict[str, Any] = {}
        child2_genes: dict[str, Any] = {}

        for i, param in enumerate(params):
            if param in parent2_genes:
                if i < crossover_point:
                    child1_genes[param] = parent1_genes[param]
                    child2_genes[param] = parent2_genes[param]
                else:
                    child1_genes[param] = parent2_genes[param]
                    child2_genes[param] = parent1_genes[param]
            else:
                child1_genes[param] = parent1_genes[param]
                child2_genes[param] = parent1_genes[param]

        return child1_genes, child2_genes

    def _two_point_crossover(
        self,
        parent1_genes: dict[str, Any],
        parent2_genes: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Two-point crossover: swap middle segment."""
        params = list(parent1_genes.keys())

        if len(params) <= 2:
            return self._single_point_crossover(parent1_genes, parent2_genes)

        # Random crossover points
        point1 = random.randint(1, len(params) - 2)
        point2 = random.randint(point1 + 1, len(params) - 1)

        child1_genes: dict[str, Any] = {}
        child2_genes: dict[str, Any] = {}

        for i, param in enumerate(params):
            if param in parent2_genes:
                if i < point1 or i >= point2:
                    child1_genes[param] = parent1_genes[param]
                    child2_genes[param] = parent2_genes[param]
                else:
                    child1_genes[param] = parent2_genes[param]
                    child2_genes[param] = parent1_genes[param]
            else:
                child1_genes[param] = parent1_genes[param]
                child2_genes[param] = parent1_genes[param]

        return child1_genes, child2_genes

    def _blend_crossover(
        self,
        parent1_genes: dict[str, Any],
        parent2_genes: dict[str, Any],
        alpha: float = 0.5,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Blend crossover: blend numeric values."""
        child1_genes: dict[str, Any] = {}
        child2_genes: dict[str, Any] = {}

        for param in parent1_genes:
            if param in parent2_genes:
                val1 = parent1_genes[param]
                val2 = parent2_genes[param]

                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Blend numeric values
                    min_val = min(val1, val2)
                    max_val = max(val1, val2)
                    range_val = max_val - min_val

                    # Extended range for exploration
                    lower = min_val - alpha * range_val
                    upper = max_val + alpha * range_val

                    if isinstance(val1, int) and isinstance(val2, int):
                        # Both are integers, keep result as integer
                        child1_genes[param] = int(random.uniform(lower, upper))
                        child2_genes[param] = int(random.uniform(lower, upper))
                    else:
                        # At least one is float, result should be float
                        child1_genes[param] = random.uniform(lower, upper)
                        child2_genes[param] = random.uniform(lower, upper)
                else:
                    # Non-numeric: use uniform crossover
                    if random.random() < 0.5:
                        child1_genes[param] = val1
                        child2_genes[param] = val2
                    else:
                        child1_genes[param] = val2
                        child2_genes[param] = val1
            else:
                child1_genes[param] = parent1_genes[param]
                child2_genes[param] = parent1_genes[param]

        return child1_genes, child2_genes


class AdvancedMutationOperator(MutationOperator):
    """Advanced mutation operator with multiple strategies."""

    def __init__(
        self,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.2,
        strategies: list[str] | None = None,
    ):
        """
        Initialize advanced mutation operator.

        Args:
            mutation_rate: Base mutation probability
            mutation_strength: Strength of mutations
            strategies: List of mutation strategies to use
        """
        super().__init__(mutation_rate, mutation_strength, adaptive=True)
        self.strategies = strategies or ["gaussian", "uniform", "polynomial"]

    def mutate(
        self,
        genes: dict[str, Any],
        parameter_ranges: dict[str, tuple[Any, Any]],
    ) -> dict[str, Any]:
        """Apply advanced mutation strategies."""
        mutated = genes.copy()

        for param, value in genes.items():
            if random.random() < self.mutation_rate:
                strategy = random.choice(self.strategies)

                if param in parameter_ranges:
                    if strategy == "gaussian":
                        mutated[param] = self._gaussian_mutation(value, parameter_ranges[param])
                    elif strategy == "uniform":
                        mutated[param] = self._uniform_mutation(value, parameter_ranges[param])
                    elif strategy == "polynomial":
                        mutated[param] = self._polynomial_mutation(value, parameter_ranges[param])
                    else:
                        mutated[param] = self._mutate_parameter(value, parameter_ranges[param])

        return mutated

    def _gaussian_mutation(self, value: Any, param_range: tuple[Any, Any]) -> Any:
        """Gaussian mutation with adaptive variance."""
        if not isinstance(value, int | float):
            return value

        min_val, max_val = param_range
        std = (max_val - min_val) * self.mutation_strength

        # Adaptive variance based on generation
        std *= np.exp(-0.01 * self.generation_count)

        noise = np.random.normal(0, std)
        new_value = value + noise

        if isinstance(value, int):
            new_value = int(new_value)

        return max(min_val, min(max_val, new_value))

    def _uniform_mutation(self, value: Any, param_range: tuple[Any, Any]) -> Any:
        """Uniform random mutation within range."""
        if not isinstance(value, int | float):
            return value

        min_val, max_val = param_range

        if isinstance(value, int):
            return random.randint(min_val, max_val)
        else:
            return random.uniform(min_val, max_val)

    def _polynomial_mutation(
        self, value: Any, param_range: tuple[Any, Any], eta: float = 20
    ) -> Any:
        """Polynomial bounded mutation."""
        if not isinstance(value, int | float):
            return value

        min_val, max_val = param_range

        # Normalize value
        normalized = (value - min_val) / (max_val - min_val)

        # Polynomial mutation
        u = random.random()

        if u < 0.5:
            delta = (2 * u) ** (1 / (eta + 1)) - 1
        else:
            delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))

        # Apply mutation
        new_normalized = normalized + delta * self.mutation_strength
        new_normalized = max(0, min(1, new_normalized))

        # Denormalize
        new_value = min_val + new_normalized * (max_val - min_val)

        if isinstance(value, int):
            new_value = int(new_value)

        return new_value
