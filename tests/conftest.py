import pytest

from neuralforge.core.genotypes import Genotype, random_genotype
from neuralforge.core.operations import PRIMITIVES
from neuralforge.core.search_space import CellSearchSpace


@pytest.fixture
def search_space():
    return CellSearchSpace()


@pytest.fixture
def random_geno():
    return random_genotype(PRIMITIVES, nodes=4)


@pytest.fixture
def sample_genotype():
    """A known genotype for testing."""
    return Genotype(
        normal=[
            ("sep_conv_3x3", 0),
            ("sep_conv_3x3", 1),
            ("sep_conv_5x5", 0),
            ("skip_connect", 0),
            ("dil_conv_3x3", 2),
            ("max_pool_3x3", 1),
            ("dil_conv_5x5", 3),
            ("avg_pool_3x3", 0),
        ],
        normal_concat=[2, 3, 4, 5],
        reduce=[
            ("avg_pool_3x3", 0),
            ("skip_connect", 1),
            ("dil_conv_3x3", 1),
            ("max_pool_3x3", 0),
            ("sep_conv_5x5", 2),
            ("sep_conv_3x3", 1),
            ("dil_conv_3x3", 2),
            ("skip_connect", 3),
        ],
        reduce_concat=[2, 3, 4, 5],
    )
