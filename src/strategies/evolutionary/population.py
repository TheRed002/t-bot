"""
Population Management for Genetic Algorithms.

This module manages individuals and populations for evolutionary optimization.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Individual:
    """Represents an individual in the population."""

    id: str
    genes: dict[str, Any]
    fitness: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """Compare individuals by fitness."""
        return self.fitness < other.fitness

    def __repr__(self):
        """String representation."""
        return f"Individual(id={self.id}, fitness={self.fitness:.4f})"

    def copy(self) -> "Individual":
        """Create a deep copy of the individual."""
        return Individual(
            id=self.id,
            genes=self.genes.copy(),
            fitness=self.fitness,
            metadata=self.metadata.copy(),
        )


class Population:
    """Manages a population of individuals."""

    def __init__(self, individuals: list[Individual]):
        """
        Initialize population.

        Args:
            individuals: List of individuals
        """
        self.individuals = individuals
        self._sorted = False

    def __len__(self) -> int:
        """Get population size."""
        return len(self.individuals)

    def __iter__(self):
        """Iterate over individuals."""
        return iter(self.individuals)

    def _ensure_sorted(self) -> None:
        """Ensure population is sorted by fitness."""
        if not self._sorted:
            self.individuals.sort(key=lambda x: x.fitness, reverse=True)
            self._sorted = True

    def get_best(self) -> Individual | None:
        """Get the best individual."""
        if not self.individuals:
            return None
        self._ensure_sorted()
        return self.individuals[0]

    def get_worst(self) -> Individual | None:
        """Get the worst individual."""
        if not self.individuals:
            return None
        self._ensure_sorted()
        return self.individuals[-1]

    def get_top_n(self, n: int) -> list[Individual]:
        """Get top n individuals by fitness."""
        self._ensure_sorted()
        return self.individuals[:n]

    def get_bottom_n(self, n: int) -> list[Individual]:
        """Get bottom n individuals by fitness."""
        self._ensure_sorted()
        return self.individuals[-n:]

    def get_bottom_n_indices(self, n: int) -> list[int]:
        """Get indices of bottom n individuals."""
        self._ensure_sorted()
        return list(range(len(self.individuals) - n, len(self.individuals)))

    def get_statistics(self) -> dict[str, float]:
        """Get population statistics."""
        if not self.individuals:
            return {
                "size": 0,
                "mean_fitness": 0,
                "std_fitness": 0,
                "min_fitness": 0,
                "max_fitness": 0,
            }

        fitnesses = [ind.fitness for ind in self.individuals]

        return {
            "size": len(self.individuals),
            "mean_fitness": float(np.mean(fitnesses)),
            "std_fitness": float(np.std(fitnesses)),
            "min_fitness": float(np.min(fitnesses)),
            "max_fitness": float(np.max(fitnesses)),
        }

    def add(self, individual: Individual) -> None:
        """Add an individual to the population."""
        self.individuals.append(individual)
        self._sorted = False

    def remove(self, individual: Individual) -> None:
        """Remove an individual from the population."""
        if individual in self.individuals:
            self.individuals.remove(individual)
            self._sorted = False

    def replace(self, index: int, individual: Individual) -> None:
        """Replace individual at index."""
        if 0 <= index < len(self.individuals):
            self.individuals[index] = individual
            self._sorted = False
