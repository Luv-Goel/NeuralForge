# API Reference

## Core

### `CellSearchSpace(config)`
The search space defining valid architectures for NAS.

**Parameters:**
- `config` (SearchSpaceConfig, optional): Search space configuration.
  Defaults to standard DARTS CIFAR-10 space.

**Properties:**
- `num_operations`: Number of primitive operations available.
- `total_edges`: Total edges in the DAG (normal + reduction cells).

**Methods:**
- `build_search_network()` → `SearchSpaceCNN`: Build the search network.
- `build_discrete_network(genotype)` → `nn.Module`: Build a discrete network from a genotype.

### `SearchSpaceConfig`
Configuration dataclass for the cell-based search space.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `c_in` | int | 3 | Input channels (3 for RGB) |
| `init_channels` | int | 16 | Channels after stem conv |
| `num_classes` | int | 10 | Output classes |
| `layers` | int | 8 | Number of cells |
| `nodes` | int | 4 | Intermediate nodes per cell |
| `primitive_set` | List[str] | PRIMITIVES | Available operations |
| `auxiliary` | bool | False | Use auxiliary head |
| `auxiliary_weight` | float | 0.4 | Auxiliary loss weight |
| `drop_path_prob` | float | 0.2 | Drop-path probability |

### `Genotype(normal, reduce)`
A discoverable architecture — compact, serializable.

**Parameters:**
- `normal`: List of (op_name, from_node) for the normal cell.
- `reduce`: List of (op_name, from_node) for the reduction cell.
- `normal_concat`, `reduce_concat`: Node indices to concatenate.

**Methods:**
- `to_dict()` → dict: JSON-serializable dict.
- `from_dict(data)` → Genotype: Create from dict.
- `to_json()` → str: Serialize to JSON.
- `to_compact_string()` → str: Human-readable format.

**Utilities:**
- `random_genotype(primitive_set, nodes)` → Genotype
- `mutate_genotype(genotype, primitive_set, mutation_rate)` → Genotype
- `crossover_genotypes(parent_a, parent_b)` → Genotype
- `build_network_from_genotype(genotype, ...)` → DiscreteNetwork

## Algorithms

### `DARTSSearch(space, config)`
Differentiable Architecture Search (Liu et al., ICLR 2019).

**Parameters:**
- `search_space` (CellSearchSpace): The search space.
- `config` (SearchConfig, optional): Algorithm configuration.
- `unrolled` (bool): Use second-order gradients (default: False).

**Methods:**
- `search(train_data)` → Genotype: Run architecture search.

### `EvolutionSearch(space, config)`
Regularized Evolution search (Real et al., ICML 2019 / AmoebaNet).

**Parameters:**
- `search_space` (CellSearchSpace): The search space.
- `config` (EvolutionConfig, optional): Evolution configuration.

**Key config:**
- `population_size`: Number of individuals (default: 50).
- `tournament_size`: Tournament selection size (default: 10).
- `cycles`: Number of generations (default: 200).
- `mutation_rate`: Per-edge mutation probability (default: 0.3).

### `RandomSearch(space, config, n_samples)`
Random search baseline (Li & Talwalkar, UAI 2020).

## Operations

### `PRIMITIVES`
List of available primitive operations:
- `none`, `max_pool_3x3`, `avg_pool_3x3`, `skip_connect`
- `sep_conv_3x3`, `sep_conv_5x5`, `dil_conv_3x3`, `dil_conv_5x5`

### `OPS`
Dictionary mapping operation names to factory functions.

### `get_op(name, c_in, c_out, stride, affine)`
Construct an operation module by name.

## Utilities

### Profiling (`neuralforge.utils.profiling`)
- `profile_model(model, input_shape)` → dict: Full model profile.
- `count_parameters(model)` → dict: Parameter counts.
- `count_macs(model, input_shape)` → int: MAC computation.
- `estimate_memory(model, input_shape)` → dict: Memory estimate.

### Visualization (`neuralforge.utils.visualization`)
- `render_cell_graph(genotype, fmt)` → str/display: Cell graph rendering.
- `plot_architecture(genotype)` → figure: Architecture visualization.
- `plot_search_trajectory(history)` → figure: Search progress plot.
- `plot_operation_importance(genotype)` → figure: Op importance chart.

### Augmentations (`neuralforge.utils.augmentations`)
- `Cutout(length, fill_value)`: Cutout augmentation (DeVries & Taylor, 2017).

## Contrib

### Lightning (`neuralforge.contrib.lightning`)
- `NeuralForgeLightning`: PyTorch Lightning integration module.
