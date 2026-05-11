"""
NeuralForge — Regularized Evolution Search
===========================================

Evolutionary architecture search following the AmoebaNet algorithm
(Real et al., ICML 2019). The key innovation over vanilla evolution
is the *age regularization*: younger models are favored in tournament
selection, preventing the population from stagnating on early
architectures.

Algorithm overview:
    1. Initialize a population of random architectures.
    2. Evaluate each by training for a fixed number of steps.
    3. Sample a tournament of size T, pick the best.
    4. Mutate the winner to create a new child architecture.
    5. Evaluate the child.
    6. Add child to population, remove the oldest.
    7. Repeat steps 3-6 until budget exhausted.

This is surprisingly competitive with DARTS while being much simpler
and more general (works with any search space, no grad requirement).

References:
    - Regularized Evolution for Image Classifier Architecture Search
      (Real et al., AAAI 2019 / AmoebaNet)
    - NAS-Bench-101 (Ying et al., 2019)
    - Deep Neural Network Ensembles from Evolution (Wong et al., 2020)
"""

from __future__ import annotations
import copy
import logging
import random
import time
import heapq
from typing import (
    Optional, List, Tuple, Dict, Any, Callable,
)
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

from neuralforge.algorithms.base import (
    SearchAlgorithm,
    SearchConfig,
)
from neuralforge.core.search_space import CellSearchSpace
from neuralforge.core.genotypes import (
    Genotype,
    random_genotype,
    mutate_genotype,
    crossover_genotypes,
)
from neuralforge.core.operations import PRIMITIVES

logger = logging.getLogger(__name__)


@dataclass
class Individual:
    """A single architecture in the evolutionary population.
    
    Attributes:
        genotype: The architecture definition.
        fitness: Validation accuracy (higher is better).
        age: Number of generations this individual has lived.
        birth_generation: When this individual was created.
        eval_time: Time taken to evaluate (seconds).
    """
    genotype: Genotype
    fitness: float = 0.0
    age: int = 0
    birth_generation: int = 0
    eval_time: float = 0.0

    def __lt__(self, other: "Individual") -> bool:
        """For priority queue ordering (max-heap by fitness)."""
        return self.fitness > other.fitness  # Reverse for max-heap

    def __repr__(self) -> str:
        return (
            f"Individual(fitness={self.fitness:.4f}, "
            f"age={self.age}, "
            f"params={self.genotype.params:.1f}M)"
        )


@dataclass
class EvolutionConfig(SearchConfig):
    """Configuration for evolutionary search.

    Attributes:
        population_size: Number of individuals in the population.
        tournament_size: Number of individuals in each tournament.
        mutation_rate: Per-edge mutation probability.
        cycles: Number of evolution cycles (generations).
        eval_epochs: Number of epochs to evaluate each architecture.
        crossover_prob: Probability of crossover vs. mutation.
        keep_top_k: Number of top individuals to keep for final output.
        surrogate: Whether to use surrogate predictor for speed.
    """
    population_size: int = 50
    tournament_size: int = 10
    mutation_rate: float = 0.3
    cycles: int = 200
    eval_epochs: int = 5
    crossover_prob: float = 0.2
    keep_top_k: int = 5
    surrogate: bool = False


class EvolutionSearch(SearchAlgorithm):
    """Regularized Evolution for Neural Architecture Search.

    Usage:
        >>> space = CellSearchSpace()
        >>> config = EvolutionConfig(cycles=100, population_size=30)
        >>> algo = EvolutionSearch(space, config)
        >>> best_genotype = algo.search()
        >>> print(best_genotype.to_compact_string())
    """

    def __init__(
        self,
        search_space: CellSearchSpace,
        config: Optional[EvolutionConfig] = None,
    ):
        if config is None:
            config = EvolutionConfig()
        self._evo_config = config
        super().__init__(search_space, config)

        self._population: List[Individual] = []
        self._age_history: List[int] = []

    def search(
        self,
        train_data=None,
        evaluate_fn: Optional[Callable[[Genotype], float]] = None,
        **kwargs,
    ) -> Genotype:
        """Run evolutionary architecture search.
        
        Args:
            train_data: Not used directly (evaluation is via evaluate_fn).
            evaluate_fn: Function that takes a Genotype and returns
                         validation accuracy. If None, uses a simple
                         random fitness (for testing/demo).
            **kwargs: Additional parameters.
        
        Returns:
            Best discovered Genotype.
        """
        logger.info(
            f"Starting Regularized Evolution search "
            f"(pop={self._evo_config.population_size}, "
            f"cycles={self._evo_config.cycles})..."
        )
        self._start_time = time.time()

        if evaluate_fn is None:
            logger.warning(
                "No evaluate_fn provided — using simulated fitness "
                "for demonstration only!"
            )
            evaluate_fn = self._dummy_evaluate

        primitive_set = self.search_space.config.primitive_set
        nodes = self.search_space.config.nodes

        # Phase 1: Initialize population with random architectures
        logger.info("Phase 1: Initializing population...")
        self._population = []

        for i in tqdm(
            range(self._evo_config.population_size),
            desc="Initializing",
        ):
            genotype = random_genotype(primitive_set, nodes)
            fitness = evaluate_fn(genotype)
            ind = Individual(
                genotype=genotype,
                fitness=fitness,
                birth_generation=0,
            )
            self._population.append(ind)
            self._log_metrics(i, 0, 0.0, fitness, "init")

        # Sort by fitness descending
        self._population.sort(key=lambda x: x.fitness, reverse=True)
        best_fitness = self._population[0].fitness
        logger.info(
            f"Initial population ready. "
            f"Best fitness: {best_fitness:.4f}"
        )

        # Phase 2: Evolution cycles
        logger.info("Phase 2: Evolution...")
        for cycle in tqdm(
            range(self._evo_config.cycles),
            desc="Evolution",
        ):
            # Age the population
            for ind in self._population:
                ind.age += 1

            # Tournament selection
            parent = self._tournament_select()

            # Crossover or mutation
            if random.random() < self._evo_config.crossover_prob:
                parent2 = self._tournament_select()
                child_geno = crossover_genotypes(parent.genotype, parent2.genotype)
            else:
                child_geno = mutate_genotype(
                    parent.genotype,
                    primitive_set,
                    self._evo_config.mutation_rate,
                )

            # Evaluate child
            eval_start = time.time()
            child_fitness = evaluate_fn(child_geno)
            eval_time = time.time() - eval_start

            child = Individual(
                genotype=child_geno,
                fitness=child_fitness,
                age=0,
                birth_generation=cycle + 1,
                eval_time=eval_time,
            )

            # Age regularization: remove oldest individual
            oldest = max(self._population, key=lambda x: x.age)
            self._population.remove(oldest)

            # Add child
            self._population.append(child)

            # Track best
            if child_fitness > best_fitness:
                best_fitness = child_fitness
                logger.info(
                    f"Cycle {cycle + 1}: New best fitness = {best_fitness:.4f} "
                    f"(params={child_geno.params:.1f}M)"
                )

            # Logging
            self._age_history.append(oldest.age)
            if (cycle + 1) % 20 == 0:
                avg_fitness = sum(
                    ind.fitness for ind in self._population
                ) / len(self._population)
                logger.info(
                    f"Cycle {cycle + 1}: best={best_fitness:.4f}, "
                    f"avg={avg_fitness:.4f}, "
                    f"oldest_removed_age={oldest.age}"
                )

        # Final: return best individual
        best_individual = max(self._population, key=lambda x: x.fitness)
        self._best_genotype = best_individual.genotype
        self._best_genotype.accuracy = best_individual.fitness

        logger.info(
            f"Evolution complete. Best fitness: {best_individual.fitness:.4f}\n"
            f"{self._best_genotype.to_compact_string()}"
        )
        return self._best_genotype

    def _tournament_select(self) -> Individual:
        """Select an individual via tournament selection.
        
        Samples `tournament_size` individuals uniformly from the
        population and returns the one with highest fitness.
        """
        tournament = random.sample(
            self._population,
            min(self._evo_config.tournament_size, len(self._population)),
        )
        return max(tournament, key=lambda x: x.fitness)

    def _dummy_evaluate(self, genotype: Genotype) -> float:
        """Dummy evaluation for testing without a GPU.
        
        Returns a simulated fitness based on genotype complexity.
        Real usage requires training the architecture.
        """
        # Simulate: architectures with more diverse ops tend to be better
        ops_used = set(op for op, _ in genotype.normal + genotype.reduce)
        diversity_score = len(ops_used) / len(self.search_space.config.primitive_set)
        # Add noise
        noise = np.random.normal(0, 0.02)
        # Penalize huge architectures slightly
        param_penalty = max(0, genotype.params - 10.0) * 0.01
        return min(1.0, max(0.0, 0.7 + 0.2 * diversity_score + noise - param_penalty))

    @property
    def population(self) -> List[Individual]:
        """Current population (sorted by fitness descending)."""
        return sorted(self._population, key=lambda x: x.fitness, reverse=True)

    def top_k(self, k: int = 5) -> List[Individual]:
        """Return the top-k individuals."""
        return self.population[:k]
