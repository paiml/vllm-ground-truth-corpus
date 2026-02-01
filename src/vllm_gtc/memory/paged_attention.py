"""PagedAttention and KV-cache management patterns.

Implements the core memory management concepts from vLLM's PagedAttention
paper. These patterns map to realizar's attention kernels and memory pools.

References:
    - vLLM PagedAttention: https://arxiv.org/abs/2309.06180
    - vLLM GitHub: https://github.com/vllm-project/vllm

Rust cross-reference:
    realizar::attention::AttentionKernel provides GPU-accelerated
    tiled attention with block-based KV-cache management.
    trueno::arena::Arena provides the underlying memory pool.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class PagedAttentionConfig:
    """Configuration for PagedAttention.

    Args:
        block_size: Number of tokens per block (default: 16).
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.

    Examples:
        >>> config = PagedAttentionConfig(block_size=16, num_heads=32, head_dim=128)
        >>> config.tokens_per_block
        16

    Rust equivalent:
        realizar::attention::AttentionConfig with block_size field.

    """

    block_size: int = 16
    num_heads: int = 32
    head_dim: int = 128

    @property
    def tokens_per_block(self) -> int:
        """Number of tokens that fit in one block."""
        return self.block_size


@dataclass
class BlockAllocator:
    """Manages allocation and freeing of KV-cache blocks.

    Implements a simple free-list allocator for cache blocks.
    Each block can store block_size tokens worth of KV pairs.

    Args:
        num_blocks: Total number of blocks available.
        block_size: Tokens per block.

    Examples:
        >>> allocator = BlockAllocator(num_blocks=100, block_size=16)
        >>> block_id = allocator.allocate()
        >>> allocator.free(block_id)

    Rust equivalent:
        trueno::arena::Arena with fixed-size block allocation.
        realizar::memory::BlockManager wraps this for KV-cache.

    """

    num_blocks: int
    block_size: int
    _free_blocks: list[int] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Initialize free block list."""
        self._free_blocks = list(range(self.num_blocks))

    def allocate(self) -> int | None:
        """Allocate a single block.

        Returns:
            Block ID if available, None if no blocks free.

        """
        if not self._free_blocks:
            return None
        return self._free_blocks.pop()

    def free(self, block_id: int) -> None:
        """Return a block to the free list.

        Args:
            block_id: Block to free.

        """
        self._free_blocks.append(block_id)

    @property
    def num_free_blocks(self) -> int:
        """Number of currently free blocks."""
        return len(self._free_blocks)


@dataclass
class BlockTable:
    """Maps sequence positions to physical blocks.

    A block table tracks which physical blocks hold the KV-cache
    for a particular sequence. As sequences grow, new blocks are
    appended.

    Examples:
        >>> table = BlockTable()
        >>> table.append(0)  # First block
        >>> table.append(5)  # Second block
        >>> table.block_ids
        [0, 5]

    Rust equivalent:
        realizar::memory::BlockTable as Vec<BlockId>.

    """

    _blocks: list[int] = field(default_factory=list, repr=False)

    def append(self, block_id: int) -> None:
        """Append a new block to the table."""
        self._blocks.append(block_id)

    def __len__(self) -> int:
        """Number of blocks in table."""
        return len(self._blocks)

    def __getitem__(self, idx: int) -> int:
        """Get block ID at index."""
        return self._blocks[idx]

    @property
    def block_ids(self) -> list[int]:
        """List of all block IDs."""
        return list(self._blocks)


class KVCache:
    """Paged Key-Value cache for transformer attention.

    Stores K and V tensors in fixed-size blocks. Each block holds
    block_size tokens worth of KV pairs for all attention heads.

    Args:
        config: PagedAttention configuration.
        num_blocks: Number of blocks to allocate.

    Examples:
        >>> config = PagedAttentionConfig(block_size=16, num_heads=4, head_dim=64)
        >>> cache = KVCache(config, num_blocks=100)
        >>> k = np.zeros((4, 64))  # (num_heads, head_dim)
        >>> v = np.zeros((4, 64))
        >>> cache.write(block_id=0, position=0, k=k, v=v)

    Rust equivalent:
        realizar::cache::PagedKVCache backed by trueno tensors.
        GPU memory managed via wgpu buffer pools.

    """

    def __init__(self, config: PagedAttentionConfig, num_blocks: int) -> None:
        """Initialize KV cache with given configuration."""
        self.config = config
        self.num_blocks = num_blocks

        # Shape: (num_blocks, num_heads, block_size, head_dim)
        self.k_cache: NDArray[np.float32] = np.zeros(
            (num_blocks, config.num_heads, config.block_size, config.head_dim),
            dtype=np.float32,
        )
        self.v_cache: NDArray[np.float32] = np.zeros(
            (num_blocks, config.num_heads, config.block_size, config.head_dim),
            dtype=np.float32,
        )

    def write(
        self,
        block_id: int,
        position: int,
        k: NDArray[np.float32],
        v: NDArray[np.float32],
    ) -> None:
        """Write KV pair to cache at given block and position.

        Args:
            block_id: Physical block index.
            position: Position within block (0 to block_size-1).
            k: Key tensor (num_heads, head_dim).
            v: Value tensor (num_heads, head_dim).

        """
        self.k_cache[block_id, :, position, :] = k
        self.v_cache[block_id, :, position, :] = v

    def read(self, block_id: int, position: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Read KV pair from cache.

        Args:
            block_id: Physical block index.
            position: Position within block.

        Returns:
            Tuple of (k, v) tensors, each (num_heads, head_dim).

        """
        k = self.k_cache[block_id, :, position, :]
        v = self.v_cache[block_id, :, position, :]
        return k, v


def allocate_blocks(allocator: BlockAllocator, num_tokens: int) -> list[int] | None:
    """Allocate blocks for a sequence of given length.

    Args:
        allocator: Block allocator to use.
        num_tokens: Number of tokens needing cache space.

    Returns:
        List of allocated block IDs, or None if insufficient blocks.

    Examples:
        >>> allocator = BlockAllocator(num_blocks=10, block_size=16)
        >>> blocks = allocate_blocks(allocator, num_tokens=20)
        >>> len(blocks)
        2

    Rust equivalent:
        realizar::memory::BlockManager::allocate_sequence

    """
    num_blocks_needed = (num_tokens + allocator.block_size - 1) // allocator.block_size

    if allocator.num_free_blocks < num_blocks_needed:
        return None

    blocks = []
    for _ in range(num_blocks_needed):
        block_id = allocator.allocate()
        if block_id is None:
            # Shouldn't happen given check above, but be safe
            for b in blocks:
                allocator.free(b)
            return None
        blocks.append(block_id)

    return blocks


def free_blocks(allocator: BlockAllocator, table: BlockTable) -> None:
    """Free all blocks in a block table.

    Args:
        allocator: Block allocator.
        table: Block table to free.

    Examples:
        >>> allocator = BlockAllocator(num_blocks=10, block_size=16)
        >>> table = BlockTable()
        >>> table.append(allocator.allocate())
        >>> free_blocks(allocator, table)

    Rust equivalent:
        realizar::memory::BlockManager::free_sequence

    """
    for block_id in table.block_ids:
        allocator.free(block_id)


def compute_paged_attention(
    query: NDArray[np.float32],
    cache: KVCache,
    block_table: BlockTable,
    seq_len: int,
    scale: float | None = None,
) -> NDArray[np.float32]:
    """Compute attention over paged KV-cache.

    Implements scaled dot-product attention reading from a paged
    KV-cache. This is the core operation in PagedAttention.

    Args:
        query: Query tensor (num_heads, head_dim).
        cache: KV cache to read from.
        block_table: Maps sequence positions to blocks.
        seq_len: Number of tokens to attend over.
        scale: Attention scale factor (default: 1/sqrt(head_dim)).

    Returns:
        Attention output (num_heads, head_dim).

    Examples:
        >>> config = PagedAttentionConfig(block_size=4, num_heads=2, head_dim=8)
        >>> cache = KVCache(config, num_blocks=4)
        >>> # ... fill cache ...
        >>> query = np.random.randn(2, 8).astype(np.float32)
        >>> output = compute_paged_attention(query, cache, block_table, seq_len=4)

    Rust equivalent:
        realizar::attention::AttentionKernel::forward with PagedKVCache.
        Uses tiled computation for memory efficiency on GPU.

    """
    config = cache.config
    num_heads = config.num_heads
    head_dim = config.head_dim
    block_size = config.block_size

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    # Gather all K, V from blocks
    keys = []
    values = []

    for pos in range(seq_len):
        block_idx = pos // block_size
        block_pos = pos % block_size
        block_id = block_table[block_idx]
        k, v = cache.read(block_id, block_pos)
        keys.append(k)
        values.append(v)

    # Stack: (seq_len, num_heads, head_dim)
    k_all = np.stack(keys, axis=0)
    v_all = np.stack(values, axis=0)

    # Compute attention per head
    # query: (num_heads, head_dim)
    # k_all: (seq_len, num_heads, head_dim)
    # v_all: (seq_len, num_heads, head_dim)

    output = np.zeros((num_heads, head_dim), dtype=np.float32)

    for h in range(num_heads):
        q_h = query[h]  # (head_dim,)
        k_h = k_all[:, h, :]  # (seq_len, head_dim)
        v_h = v_all[:, h, :]  # (seq_len, head_dim)

        # Attention scores
        scores = np.dot(k_h, q_h) * scale  # (seq_len,)
        weights = _softmax(scores)  # (seq_len,)

        # Weighted sum of values
        output[h] = np.dot(weights, v_h)  # (head_dim,)

    return output


def _softmax(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Numerically stable softmax."""
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)
