"""Tensor parallelism patterns for distributed inference.

Implements weight sharding and collective communication patterns
used in vLLM for multi-GPU inference.

References:
    - Megatron-LM: https://arxiv.org/abs/1909.08053
    - vLLM Distributed: https://github.com/vllm-project/vllm

Rust cross-reference:
    repartir::ShardedExecutor implements tensor parallelism.
    repartir::collective::AllReduce handles cross-GPU communication.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ParallelConfig:
    """Configuration for parallel execution.

    Args:
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        pipeline_parallel_size: Number of GPUs for pipeline parallelism.

    Examples:
        >>> config = ParallelConfig(tensor_parallel_size=4)
        >>> config.world_size
        4

    Rust equivalent:
        repartir::Topology with data and model axes.

    """

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    @property
    def world_size(self) -> int:
        """Total number of parallel workers."""
        return self.tensor_parallel_size * self.pipeline_parallel_size


def shard_model_weights(
    weights: dict[str, NDArray[np.float32]],
    config: ParallelConfig,
    shard_dim: int = 1,
) -> list[dict[str, NDArray[np.float32]]]:
    """Shard model weights across tensor parallel workers.

    Splits each weight tensor along the specified dimension,
    creating one shard per tensor parallel rank.

    Args:
        weights: Dictionary of weight tensors.
        config: Parallel configuration.
        shard_dim: Dimension to shard along (0=row, 1=column).

    Returns:
        List of weight dictionaries, one per worker.

    Examples:
        >>> config = ParallelConfig(tensor_parallel_size=2)
        >>> weights = {"w": np.ones((4, 8))}
        >>> shards = shard_model_weights(weights, config, shard_dim=1)
        >>> shards[0]["w"].shape
        (4, 4)

    Rust equivalent:
        repartir::ShardedExecutor::shard_params

    """
    tp_size = config.tensor_parallel_size

    if tp_size == 1:
        return [weights]

    shards: list[dict[str, NDArray[np.float32]]] = [{} for _ in range(tp_size)]

    for name, tensor in weights.items():
        chunks = np.array_split(tensor, tp_size, axis=shard_dim)
        for i, chunk in enumerate(chunks):
            shards[i][name] = chunk

    return shards


def tensor_parallel_linear(
    x: NDArray[np.float32],
    w_shard: NDArray[np.float32],
    bias: NDArray[np.float32] | None = None,
    mode: str = "column",
) -> NDArray[np.float32]:
    """Compute linear layer with tensor-parallel weight shard.

    In column parallel mode, each worker has columns of the weight.
    In row parallel mode, each worker has rows of the weight.

    Args:
        x: Input activations.
        w_shard: Weight shard for this worker.
        bias: Optional bias (only added in column mode).
        mode: "column" or "row" parallelism.

    Returns:
        Partial output (needs gather or all-reduce to complete).

    Examples:
        >>> x = np.ones((2, 4))
        >>> w = np.ones((4, 2))
        >>> out = tensor_parallel_linear(x, w, mode="column")
        >>> out.shape
        (2, 2)

    Rust equivalent:
        repartir::tensor_parallel::matmul with ColumnParallel
        or RowParallel strategy.

    """
    output = np.matmul(x, w_shard)

    if mode == "column" and bias is not None:
        output = output + bias

    return output


def gather_outputs(
    shards: list[NDArray[np.float32]],
    dim: int = 1,
) -> NDArray[np.float32]:
    """Gather output shards from tensor parallel workers.

    Concatenates partial outputs along the sharding dimension.
    Used after column-parallel linear layers.

    Args:
        shards: List of output shards from each worker.
        dim: Dimension to concatenate along.

    Returns:
        Gathered full output.

    Examples:
        >>> shards = [np.ones((2, 2)), np.ones((2, 2))]
        >>> gathered = gather_outputs(shards, dim=1)
        >>> gathered.shape
        (2, 4)

    Rust equivalent:
        repartir::collective::AllGather

    """
    return np.concatenate(shards, axis=dim)


def all_reduce_sum(
    partials: list[NDArray[np.float32]],
) -> NDArray[np.float32]:
    """Sum partial results across all workers.

    Used after row-parallel linear layers where each worker
    computes a partial sum that must be combined.

    Args:
        partials: List of partial results from each worker.

    Returns:
        Sum of all partial results.

    Examples:
        >>> partials = [np.array([1, 2]), np.array([3, 4])]
        >>> all_reduce_sum(partials)
        array([4, 6])

    Rust equivalent:
        repartir::collective::AllReduce::sum using NCCL
        or tree-reduction for O(log n) latency.

    """
    return np.sum(np.stack(partials, axis=0), axis=0)
