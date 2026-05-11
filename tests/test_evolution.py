"""
Tests for the Evolution search algorithm.
Tests individual operators and search initialization.
"""

from neuralforge.algorithms.evolution import (
    EvolutionConfig,
    EvolutionSearch,
    Individual,
)
from neuralforge.core.genotypes import Genotype, random_genotype
from neuralforge.core.operations import PRIMITIVES
from neuralforge.core.search_space import CellSearchSpace


class TestIndividual:
    def test_individual_creation(self):
        """Individual should store genotype and fitness."""
        geno = random_genotype(PRIMITIVES, nodes=4)
        ind = Individual(genotype=geno, fitness=0.85)
        assert ind.fitness == 0.85
        assert ind.age == 0
        assert ind.birth_generation == 0

    def test_individual_ordering(self):
        """Individuals should sort by fitness descending (max-heap)."""
        geno = random_genotype(PRIMITIVES, nodes=4)
        a = Individual(genotype=geno, fitness=0.5)
        b = Individual(genotype=geno, fitness=0.9)
        c = Individual(genotype=geno, fitness=0.7)
        # Python's sort uses __lt__, which returns self.fitness > other.fitness
        # So sorted() should put highest fitness first
        sorted_inds = sorted([a, b, c])
        assert sorted_inds[0].fitness == 0.9
        assert sorted_inds[-1].fitness == 0.5

    def test_individual_repr(self):
        """Individual repr should contain key info."""
        geno = random_genotype(PRIMITIVES, nodes=4)
        ind = Individual(genotype=geno, fitness=0.85)
        r = repr(ind)
        assert "0.8500" in r
        assert "age=" in r


class TestEvolutionConfig:
    def test_default_config(self):
        """EvolutionConfig should have sensible defaults."""
        config = EvolutionConfig()
        assert config.population_size == 50
        assert config.tournament_size == 10
        assert config.cycles == 200
        assert config.mutation_rate == 0.3

    def test_inherits_from_search_config(self):
        """EvolutionConfig should inherit SearchConfig fields."""
        config = EvolutionConfig()
        assert hasattr(config, "epochs")
        assert hasattr(config, "batch_size")
        assert hasattr(config, "device")

    def test_custom_config(self):
        """Custom values should override defaults."""
        config = EvolutionConfig(
            population_size=20,
            cycles=50,
            mutation_rate=0.5,
        )
        assert config.population_size == 20
        assert config.cycles == 50
        assert config.mutation_rate == 0.5


class TestEvolutionSearch:
    def test_algorithm_creation(self):
        """EvolutionSearch should initialize without error."""
        space = CellSearchSpace()
        algo = EvolutionSearch(space)
        assert algo is not None

    def test_search_no_evaluator(self):
        """Search should work with dummy evaluator."""
        space = CellSearchSpace()
        config = EvolutionConfig(
            population_size=5,
            cycles=3,
            tournament_size=2,
        )
        algo = EvolutionSearch(space, config)
        best = algo.search()
        assert isinstance(best, Genotype)
        assert best.accuracy > 0

    def test_search_with_custom_evaluator(self):
        """Custom evaluator should be used during search."""
        space = CellSearchSpace()
        config = EvolutionConfig(
            population_size=5,
            cycles=3,
            tournament_size=2,
        )
        algo = EvolutionSearch(space, config)

        call_count = 0

        def my_evaluator(genotype):
            nonlocal call_count
            call_count += 1
            return 0.5 + (call_count % 5) * 0.1

        best = algo.search(evaluate_fn=my_evaluator)
        assert isinstance(best, Genotype)
        # 5 initial evaluations + 3 cycle evaluations = 8
        assert call_count == 8

    def test_tournament_select(self):
        """Tournament selection should return an individual."""
        space = CellSearchSpace()
        config = EvolutionConfig(
            population_size=10,
            tournament_size=3,
            cycles=0,  # No cycles, just init
        )
        algo = EvolutionSearch(space, config)
        # Seed population manually
        for i in range(10):
            geno = random_genotype(PRIMITIVES, nodes=4)
            algo._population.append(Individual(genotype=geno, fitness=0.1 * i))
        selected = algo._tournament_select()
        assert selected.fitness >= 0  # should be one of the 10

    def test_top_k(self):
        """Top-k should return k best individuals."""
        space = CellSearchSpace()
        config = EvolutionConfig(population_size=10, cycles=0)
        algo = EvolutionSearch(space, config)
        # Seed population with varied fitness
        for i in range(10):
            geno = random_genotype(PRIMITIVES, nodes=4)
            algo._population.append(Individual(genotype=geno, fitness=0.1 * i))
        top = algo.top_k(3)
        assert len(top) == 3
        # Should be sorted descending
        assert top[0].fitness >= top[1].fitness >= top[2].fitness

    def test_population_property(self):
        """Population property should return sorted list."""
        space = CellSearchSpace()
        algo = EvolutionSearch(space)
        algo._population = []
        for i in range(5):
            geno = random_genotype(PRIMITIVES, nodes=4)
            algo._population.append(Individual(genotype=geno, fitness=float(i)))
        pop = algo.population
        assert pop[0].fitness == 4.0
        assert pop[-1].fitness == 0.0
