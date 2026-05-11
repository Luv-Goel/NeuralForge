"""
NeuralForge — Random Search Baseline
=====================================

Simple random search baseline for architecture search. Surprisingly
effective — often within 1-2% of sophisticated methods while being
orders of magnitude simpler.

The key insight (from Li & Talwalkar, 2020): most published NAS methods
barely beat random search with the same compute budget. Always compare
against this baseline!

References:
    - Random Search and Reproducibility for Neural Architecture Search
      (Li & Talwalkar, UAI 2020)
    - NAS-Bench-101: Towards Reproducible NAS (Ying et al., 2019)
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import numpy as np
from tqdm import tqdm

from neuralforge.algorithms.base import SearchAlgorithm, SearchConfig
from neuralforge.core.genotypes import Genotype, random_genotype
from neuralforge.core.search_space import CellSearchSpace

logger = logging.getLogger(__name__)


class RandomSearch(SearchAlgorithm):
    """Random architecture search baseline.

    Samples architectures uniformly at random from the search space
    and evaluates each. Returns the best found.

    Usage:
        >>> space = CellSearchSpace()
        >>> algo = RandomSearch(space, n_samples=100)
        >>> best = algo.search(evaluate_fn=my_evaluator)
    """

    def __init__(
        self,
        search_space: CellSearchSpace,
        config: Optional[SearchConfig] = None,
        n_samples: int = 100,
    ):
        super().__init__(search_space, config)
        self._n_samples = n_samples

    def search(
        self,
        train_data=None,
        evaluate_fn: Optional[Callable[[Genotype], float]] = None,
        **kwargs,
    ) -> Genotype:
        """Run random search.

        Args:
            train_data: Not used.
            evaluate_fn: Function that takes Genotype and returns fitness.
                         If None, uses dummy evaluation.
            **kwargs: Additional parameters.

        Returns:
            Best Genotype found.
        """
        logger.info(f"Starting random search ({self._n_samples} samples)...")
        self._start_time = time.time()

        if evaluate_fn is None:
            logger.warning("No evaluate_fn — using dummy evaluation.")
            evaluate_fn = self._dummy_evaluate

        primitive_set = self.search_space.config.primitive_set
        nodes = self.search_space.config.nodes

        best_geno = None
        best_fitness = -float("inf")

        for i in tqdm(range(self._n_samples), desc="Random search"):
            genotype = random_genotype(primitive_set, nodes)
            fitness = evaluate_fn(genotype)

            if fitness > best_fitness:
                best_fitness = fitness
                best_geno = genotype
                logger.debug(f"Sample {i + 1}: new best = {fitness:.4f}")

            self._log_metrics(i, 0, 0.0, fitness, "sample")

        self._best_genotype = best_geno
        self._best_genotype.accuracy = best_fitness

        logger.info(f"Random search complete. Best fitness: {best_fitness:.4f}")
        return self._best_genotype

    def _dummy_evaluate(self, genotype: Genotype) -> float:
        """Dummy fitness — uses operation diversity + noise."""
        ops_used = set(op for op, _ in genotype.normal + genotype.reduce)
        diversity = len(ops_used) / len(self.search_space.config.primitive_set)
        noise = np.random.normal(0, 0.05)
        return min(1.0, max(0.0, 0.5 + 0.4 * diversity + noise))
