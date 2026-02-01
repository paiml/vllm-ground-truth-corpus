"""Tests for vllm_gtc.serving module."""

from __future__ import annotations

import numpy as np


class TestParallelConfig:
    """Tests for ParallelConfig."""

    def test_default_config(self):
        from vllm_gtc.serving import ParallelConfig

        config = ParallelConfig()
        assert config.tensor_parallel_size == 1
        assert config.pipeline_parallel_size == 1

    def test_custom_config(self):
        from vllm_gtc.serving import ParallelConfig

        config = ParallelConfig(tensor_parallel_size=4, pipeline_parallel_size=2)
        assert config.tensor_parallel_size == 4
        assert config.pipeline_parallel_size == 2

    def test_world_size(self):
        from vllm_gtc.serving import ParallelConfig

        config = ParallelConfig(tensor_parallel_size=4, pipeline_parallel_size=2)
        assert config.world_size == 8


class TestShardModelWeights:
    """Tests for shard_model_weights."""

    def test_shard_by_column(self):
        from vllm_gtc.serving import ParallelConfig, shard_model_weights

        config = ParallelConfig(tensor_parallel_size=2)
        weights = {"layer.weight": np.ones((4, 8))}

        shards = shard_model_weights(weights, config, shard_dim=1)
        assert shards[0]["layer.weight"].shape == (4, 4)
        assert shards[1]["layer.weight"].shape == (4, 4)

    def test_shard_by_row(self):
        from vllm_gtc.serving import ParallelConfig, shard_model_weights

        config = ParallelConfig(tensor_parallel_size=2)
        weights = {"layer.weight": np.ones((8, 4))}

        shards = shard_model_weights(weights, config, shard_dim=0)
        assert shards[0]["layer.weight"].shape == (4, 4)
        assert shards[1]["layer.weight"].shape == (4, 4)

    def test_shard_preserves_values(self):
        from vllm_gtc.serving import ParallelConfig, shard_model_weights

        config = ParallelConfig(tensor_parallel_size=2)
        weights = {"w": np.arange(8).reshape(2, 4).astype(np.float32)}

        shards = shard_model_weights(weights, config, shard_dim=1)
        # First shard: columns 0-1, Second shard: columns 2-3
        assert np.allclose(shards[0]["w"], [[0, 1], [4, 5]])
        assert np.allclose(shards[1]["w"], [[2, 3], [6, 7]])

    def test_single_shard_unchanged(self):
        from vllm_gtc.serving import ParallelConfig, shard_model_weights

        config = ParallelConfig(tensor_parallel_size=1)
        weights = {"w": np.ones((4, 4))}

        shards = shard_model_weights(weights, config, shard_dim=1)
        assert len(shards) == 1
        assert np.allclose(shards[0]["w"], weights["w"])


class TestTensorParallelLinear:
    """Tests for tensor_parallel_linear."""

    def test_column_parallel(self):
        from vllm_gtc.serving import tensor_parallel_linear

        x = np.ones((2, 4))
        w_shard = np.ones((4, 2))  # Shard of larger weight

        output = tensor_parallel_linear(x, w_shard, bias=None, mode="column")
        assert output.shape == (2, 2)

    def test_row_parallel(self):
        from vllm_gtc.serving import tensor_parallel_linear

        x = np.ones((2, 4))
        w_shard = np.ones((4, 4))

        output = tensor_parallel_linear(x, w_shard, bias=None, mode="row")
        assert output.shape == (2, 4)

    def test_with_bias(self):
        from vllm_gtc.serving import tensor_parallel_linear

        x = np.ones((2, 4))
        w = np.ones((4, 2))
        b = np.array([1.0, 2.0])

        output = tensor_parallel_linear(x, w, bias=b, mode="column")
        # x @ w = [[4, 4], [4, 4]], + bias = [[5, 6], [5, 6]]
        expected = np.array([[5.0, 6.0], [5.0, 6.0]])
        assert np.allclose(output, expected)


class TestGatherOutputs:
    """Tests for gather_outputs."""

    def test_gather_column_shards(self):
        from vllm_gtc.serving import gather_outputs

        shards = [np.ones((2, 2)), np.ones((2, 2)) * 2]
        gathered = gather_outputs(shards, dim=1)
        assert gathered.shape == (2, 4)
        assert np.allclose(gathered[:, :2], 1.0)
        assert np.allclose(gathered[:, 2:], 2.0)

    def test_gather_row_shards(self):
        from vllm_gtc.serving import gather_outputs

        shards = [np.ones((2, 4)), np.ones((2, 4)) * 2]
        gathered = gather_outputs(shards, dim=0)
        assert gathered.shape == (4, 4)


class TestAllReduceSum:
    """Tests for all_reduce_sum."""

    def test_sum_across_workers(self):
        from vllm_gtc.serving import all_reduce_sum

        partials = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        result = all_reduce_sum(partials)
        assert np.allclose(result, [4.0, 6.0])

    def test_single_worker(self):
        from vllm_gtc.serving import all_reduce_sum

        partials = [np.array([1.0, 2.0, 3.0])]
        result = all_reduce_sum(partials)
        assert np.allclose(result, [1.0, 2.0, 3.0])
