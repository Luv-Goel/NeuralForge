import json

from neuralforge.core.genotypes import (
    Genotype,
    crossover_genotypes,
    mutate_genotype,
    random_genotype,
)
from neuralforge.core.operations import PRIMITIVES


class TestGenotype:
    def test_create_genotype(self, sample_genotype):
        assert len(sample_genotype.normal) > 0
        assert len(sample_genotype.reduce) > 0

    def test_to_dict_roundtrip(self, sample_genotype):
        d = sample_genotype.to_dict()
        restored = Genotype.from_dict(d)
        assert restored.normal == sample_genotype.normal
        assert restored.reduce == sample_genotype.reduce
        assert restored.accuracy == sample_genotype.accuracy

    def test_to_json(self, sample_genotype):
        json_str = sample_genotype.to_json()
        d = json.loads(json_str)
        assert len(d["normal"]) == len(sample_genotype.normal)

    def test_compact_string(self, sample_genotype):
        s = sample_genotype.to_compact_string()
        assert "Normal:" in s
        assert "Reduce:" in s


class TestRandomGenotype:
    def test_random_genotype_is_valid(self):
        for _ in range(20):
            g = random_genotype(PRIMITIVES, nodes=4)
            assert len(g.normal) > 0
            total_edges = sum(min(2, 2 + i) for i in range(4))
            assert (
                len(g.normal) == total_edges
            ), f"Expected {total_edges} edges, got {len(g.normal)}"

    def test_random_genotype_uses_valid_ops(self):
        for _ in range(20):
            g = random_genotype(PRIMITIVES, nodes=4)
            for op, _ in g.normal + g.reduce:
                assert op in PRIMITIVES, f"Invalid op: {op}"


class TestMutateGenotype:
    def test_mutation_changes_ops(self):
        original = random_genotype(PRIMITIVES, nodes=4)
        mutated = mutate_genotype(original, PRIMITIVES, mutation_rate=1.0)
        orig_ops = [op for op, _ in original.normal + original.reduce]
        mut_ops = [op for op, _ in mutated.normal + mutated.reduce]
        # With enough ops, at least one should differ
        assert orig_ops != mut_ops or len(set(PRIMITIVES)) <= 1

    def test_mutation_preserves_structure(self):
        original = random_genotype(PRIMITIVES, nodes=4)
        mutated = mutate_genotype(original, PRIMITIVES, mutation_rate=0.5)
        assert len(original.normal) == len(mutated.normal)
        assert len(original.reduce) == len(mutated.reduce)


class TestCrossoverGenotypes:
    def test_crossover(self):
        parent_a = random_genotype(PRIMITIVES, nodes=4)
        parent_b = random_genotype(PRIMITIVES, nodes=4)
        child = crossover_genotypes(parent_a, parent_b)
        assert len(child.normal) == len(parent_a.normal)
        assert len(child.reduce) == len(parent_a.reduce)
