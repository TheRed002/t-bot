"""
Evolutionary Trading Strategies Framework (P-013D).

This module provides genetic algorithm-based strategy evolution,
neuroevolution capabilities, and multi-objective optimization for
developing adaptive trading strategies.

CRITICAL: Integrates with P-001 (types), P-011 (base strategies),
and P-013C (backtesting) for comprehensive strategy development.
"""

from .fitness import FitnessEvaluator
from .genetic import GeneticAlgorithm, GeneticConfig
from .mutations import CrossoverOperator, MutationOperator
from .neuroevolution import NeuroEvolutionConfig, NeuroEvolutionStrategy
from .population import Individual, Population

__all__ = [
    "CrossoverOperator",
    # Fitness Evaluation
    "FitnessEvaluator",
    # Genetic Algorithm
    "GeneticAlgorithm",
    "GeneticConfig",
    "Individual",
    # Mutation and Crossover
    "MutationOperator",
    "NeuroEvolutionConfig",
    # Neuroevolution
    "NeuroEvolutionStrategy",
    # Population Management
    "Population",
]

# Version information
__version__ = "1.0.0"
__author__ = "Trading Bot Framework"
__description__ = "Evolutionary Trading Strategies with ML Integration (P-013D)"
