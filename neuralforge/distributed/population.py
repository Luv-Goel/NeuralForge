"""
NeuralForge — Distributed Population Management
================================================

Distributed population-based training for evolutionary search.
Manages a population of architectures evaluated across multiple
workers (GPUs or machines).

Architecture:
    - PopulationServer: Central coordinator
    - PopulationWorker: Evaluates architectures
    - Uses simple file-based or Redis-based communication

This enables scaling evolutionary search across multiple GPUs.
"""

from __future__ import annotations
import json
import logging
import time
import os
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass

import numpy as np

from neuralforge.core.genotypes import Genotype
from neuralforge.algorithms.evolution import Individual

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    backend: str = "file"  # "file" or "redis"
    work_dir: str = "./distributed_population"
    heartbeat_interval: int = 10
    max_stale_workers: int = 3


class DistributedPopulation:
    """Distributed population manager for evolution."""
    
    def __init__(
        self,
        config: Optional[DistributedConfig] = None,
    ):
        self.config = config or DistributedConfig()
        self._work_dir = Path(self.config.work_dir)
        self._work_dir.mkdir(parents=True, exist_ok=True)
        
        self._individuals: Dict[str, Individual] = {}
        self._workers: Dict[str, float] = {}  # worker_id -> last heartbeat
    
    def add_individual(
        self,
        individual: Individual,
        worker_id: str = "local",
    ) -> None:
        key = f"{worker_id}_{individual.genotype.to_json()}"
        self._individuals[key] = individual
        
        if self.config.backend == "file":
            self._save_to_file(individual, worker_id)
    
    def get_best(self, k: int = 5) -> List[Individual]:
        sorted_indivs = sorted(
            self._individuals.values(),
            key=lambda x: x.fitness,
            reverse=True,
        )
        return sorted_indivs[:k]
    
    def _save_to_file(self, individual: Individual, worker_id: str) -> None:
        path = self._work_dir / f"indiv_{worker_id}_{int(time.time())}.json"
        data = {
            "worker_id": worker_id,
            "genotype": individual.genotype.to_dict(),
            "fitness": individual.fitness,
            "age": individual.age,
            "timestamp": time.time(),
        }
        path.write_text(json.dumps(data))
    
    def worker_heartbeat(self, worker_id: str) -> None:
        self._workers[worker_id] = time.time()
    
    @property
    def size(self) -> int:
        return len(self._individuals)
    
    @property
    def active_workers(self) -> List[str]:
        now = time.time()
        return [
            wid for wid, last in self._workers.items()
            if now - last < self.config.heartbeat_interval * 3
        ]
