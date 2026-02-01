"""Tests for vllm_gtc.memory module."""

from __future__ import annotations

import numpy as np


class TestPagedAttentionConfig:
    """Tests for PagedAttentionConfig."""

    def test_default_config(self):
        from vllm_gtc.memory import PagedAttentionConfig

        config = PagedAttentionConfig()
        assert config.block_size == 16
        assert config.num_heads > 0
        assert config.head_dim > 0

    def test_custom_config(self):
        from vllm_gtc.memory import PagedAttentionConfig

        config = PagedAttentionConfig(block_size=32, num_heads=8, head_dim=64)
        assert config.block_size == 32
        assert config.num_heads == 8
        assert config.head_dim == 64

    def test_tokens_per_block(self):
        from vllm_gtc.memory import PagedAttentionConfig

        config = PagedAttentionConfig(block_size=16)
        assert config.tokens_per_block == 16


class TestBlockAllocator:
    """Tests for BlockAllocator."""

    def test_allocate_single_block(self):
        from vllm_gtc.memory import BlockAllocator

        allocator = BlockAllocator(num_blocks=10, block_size=16)
        block_id = allocator.allocate()
        assert block_id is not None
        assert 0 <= block_id < 10

    def test_allocate_multiple_blocks(self):
        from vllm_gtc.memory import BlockAllocator

        allocator = BlockAllocator(num_blocks=10, block_size=16)
        blocks = [allocator.allocate() for _ in range(5)]
        assert len(set(blocks)) == 5  # All unique

    def test_free_block(self):
        from vllm_gtc.memory import BlockAllocator

        allocator = BlockAllocator(num_blocks=2, block_size=16)
        b1 = allocator.allocate()
        _ = allocator.allocate()  # Second block
        assert allocator.num_free_blocks == 0

        allocator.free(b1)
        assert allocator.num_free_blocks == 1

    def test_allocate_when_full_returns_none(self):
        from vllm_gtc.memory import BlockAllocator

        allocator = BlockAllocator(num_blocks=2, block_size=16)
        allocator.allocate()
        allocator.allocate()
        assert allocator.allocate() is None

    def test_num_free_blocks(self):
        from vllm_gtc.memory import BlockAllocator

        allocator = BlockAllocator(num_blocks=5, block_size=16)
        assert allocator.num_free_blocks == 5
        allocator.allocate()
        assert allocator.num_free_blocks == 4


class TestBlockTable:
    """Tests for BlockTable."""

    def test_create_empty(self):
        from vllm_gtc.memory import BlockTable

        table = BlockTable()
        assert len(table) == 0

    def test_append_block(self):
        from vllm_gtc.memory import BlockTable

        table = BlockTable()
        table.append(0)
        table.append(1)
        assert len(table) == 2
        assert table[0] == 0
        assert table[1] == 1

    def test_get_block_ids(self):
        from vllm_gtc.memory import BlockTable

        table = BlockTable()
        table.append(5)
        table.append(3)
        assert table.block_ids == [5, 3]


class TestKVCache:
    """Tests for KVCache."""

    def test_create_cache(self):
        from vllm_gtc.memory import KVCache, PagedAttentionConfig

        config = PagedAttentionConfig(block_size=16, num_heads=4, head_dim=32)
        cache = KVCache(config, num_blocks=10)
        assert cache.num_blocks == 10

    def test_cache_shape(self):
        from vllm_gtc.memory import KVCache, PagedAttentionConfig

        config = PagedAttentionConfig(block_size=16, num_heads=4, head_dim=32)
        cache = KVCache(config, num_blocks=10)
        # Shape: (num_blocks, 2, num_heads, block_size, head_dim)
        # 2 for key and value
        assert cache.k_cache.shape == (10, 4, 16, 32)
        assert cache.v_cache.shape == (10, 4, 16, 32)

    def test_write_and_read(self):
        from vllm_gtc.memory import KVCache, PagedAttentionConfig

        config = PagedAttentionConfig(block_size=16, num_heads=2, head_dim=8)
        cache = KVCache(config, num_blocks=4)

        # Write to block 0, position 0
        k = np.ones((2, 8))  # (num_heads, head_dim)
        v = np.ones((2, 8)) * 2
        cache.write(block_id=0, position=0, k=k, v=v)

        # Read back
        k_read, v_read = cache.read(block_id=0, position=0)
        assert np.allclose(k_read, k)
        assert np.allclose(v_read, v)


class TestAllocateBlocks:
    """Tests for allocate_blocks helper."""

    def test_allocate_for_sequence(self):
        from vllm_gtc.memory import BlockAllocator, allocate_blocks

        allocator = BlockAllocator(num_blocks=10, block_size=16)
        # 20 tokens needs 2 blocks (16 tokens per block)
        blocks = allocate_blocks(allocator, num_tokens=20)
        assert len(blocks) == 2

    def test_allocate_exact_block(self):
        from vllm_gtc.memory import BlockAllocator, allocate_blocks

        allocator = BlockAllocator(num_blocks=10, block_size=16)
        blocks = allocate_blocks(allocator, num_tokens=16)
        assert len(blocks) == 1

    def test_allocate_insufficient_blocks(self):
        from vllm_gtc.memory import BlockAllocator, allocate_blocks

        allocator = BlockAllocator(num_blocks=1, block_size=16)
        blocks = allocate_blocks(allocator, num_tokens=32)
        assert blocks is None  # Not enough blocks


class TestFreeBlocks:
    """Tests for free_blocks helper."""

    def test_free_all_blocks(self):
        from vllm_gtc.memory import BlockAllocator, BlockTable, free_blocks

        allocator = BlockAllocator(num_blocks=5, block_size=16)
        table = BlockTable()
        table.append(allocator.allocate())
        table.append(allocator.allocate())
        assert allocator.num_free_blocks == 3

        free_blocks(allocator, table)
        assert allocator.num_free_blocks == 5


class TestComputePagedAttention:
    """Tests for compute_paged_attention."""

    def test_single_block_attention(self):
        from vllm_gtc.memory import (
            BlockTable,
            KVCache,
            PagedAttentionConfig,
            compute_paged_attention,
        )

        config = PagedAttentionConfig(block_size=4, num_heads=2, head_dim=8)
        cache = KVCache(config, num_blocks=4)

        # Fill block 0 with some KV
        for pos in range(4):
            k = np.random.randn(2, 8).astype(np.float32)
            v = np.random.randn(2, 8).astype(np.float32)
            cache.write(block_id=0, position=pos, k=k, v=v)

        block_table = BlockTable()
        block_table.append(0)

        query = np.random.randn(2, 8).astype(np.float32)  # (num_heads, head_dim)
        output = compute_paged_attention(query, cache, block_table, seq_len=4)

        assert output.shape == (2, 8)  # (num_heads, head_dim)

    def test_multi_block_attention(self):
        from vllm_gtc.memory import (
            BlockTable,
            KVCache,
            PagedAttentionConfig,
            compute_paged_attention,
        )

        config = PagedAttentionConfig(block_size=4, num_heads=2, head_dim=8)
        cache = KVCache(config, num_blocks=4)

        # Fill blocks 0 and 1
        for block_id in [0, 1]:
            for pos in range(4):
                k = np.random.randn(2, 8).astype(np.float32)
                v = np.random.randn(2, 8).astype(np.float32)
                cache.write(block_id=block_id, position=pos, k=k, v=v)

        block_table = BlockTable()
        block_table.append(0)
        block_table.append(1)

        query = np.random.randn(2, 8).astype(np.float32)
        output = compute_paged_attention(query, cache, block_table, seq_len=8)

        assert output.shape == (2, 8)
