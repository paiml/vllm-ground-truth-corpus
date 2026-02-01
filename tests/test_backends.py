"""Tests for vllm_gtc.backends module."""

from __future__ import annotations

import numpy as np


class TestBackendType:
    """Tests for BackendType enum."""

    def test_backend_types_exist(self):
        from vllm_gtc.backends import BackendType

        assert BackendType.CPU is not None
        assert BackendType.GPU is not None
        assert BackendType.AUTO is not None


class TestCPUBackend:
    """Tests for CPUBackend."""

    def test_device_info(self):
        from vllm_gtc.backends import BackendType, CPUBackend

        backend = CPUBackend()
        info = backend.device_info
        assert info.name == "CPU"
        assert info.backend == BackendType.CPU
        assert info.memory_bytes > 0

    def test_matmul(self):
        from vllm_gtc.backends import CPUBackend

        backend = CPUBackend()
        a = np.ones((2, 3), dtype=np.float32)
        b = np.ones((3, 4), dtype=np.float32)
        result = backend.matmul(a, b)
        assert result.shape == (2, 4)
        assert np.allclose(result, 3.0)

    def test_softmax(self):
        from vllm_gtc.backends import CPUBackend

        backend = CPUBackend()
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = backend.softmax(x, axis=-1)
        assert result.shape == (1, 3)
        assert np.allclose(result.sum(axis=-1), 1.0)

    def test_layer_norm(self):
        from vllm_gtc.backends import CPUBackend

        backend = CPUBackend()
        x = np.random.randn(2, 4, 8).astype(np.float32)
        weight = np.ones(8, dtype=np.float32)
        bias = np.zeros(8, dtype=np.float32)
        result = backend.layer_norm(x, weight, bias)
        assert result.shape == x.shape
        # After layer norm with identity weight/bias, mean should be ~0
        assert np.abs(result.mean(axis=-1)).max() < 1e-5

    def test_attention(self):
        from vllm_gtc.backends import CPUBackend

        backend = CPUBackend()
        batch, heads, seq, dim = 1, 2, 4, 8
        q = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        k = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        v = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        result = backend.attention(q, k, v)
        assert result.shape == (batch, heads, seq, dim)


class TestGPUBackend:
    """Tests for GPUBackend."""

    def test_device_info(self):
        from vllm_gtc.backends import BackendType, GPUBackend

        backend = GPUBackend(device_id=0)
        info = backend.device_info
        assert info.name == "GPU:0"
        assert info.backend == BackendType.GPU
        assert info.compute_capability is not None

    def test_matmul(self):
        from vllm_gtc.backends import GPUBackend

        backend = GPUBackend()
        a = np.ones((2, 3), dtype=np.float32)
        b = np.ones((3, 4), dtype=np.float32)
        result = backend.matmul(a, b)
        assert result.shape == (2, 4)

    def test_memory_check(self):
        import pytest

        from vllm_gtc.backends import GPUBackend

        # Create backend with very small memory
        backend = GPUBackend(memory_bytes=100)
        a = np.ones((1000, 1000), dtype=np.float32)  # 4MB
        b = np.ones((1000, 1000), dtype=np.float32)
        with pytest.raises(MemoryError):
            backend.matmul(a, b)

    def test_synchronize(self):
        from vllm_gtc.backends import GPUBackend

        backend = GPUBackend()
        # Should not raise
        backend.synchronize()


class TestGetBackend:
    """Tests for get_backend factory."""

    def test_get_cpu_backend(self):
        from vllm_gtc.backends import BackendType, CPUBackend, get_backend

        backend = get_backend(BackendType.CPU)
        assert isinstance(backend, CPUBackend)

    def test_get_gpu_backend(self):
        from vllm_gtc.backends import BackendType, GPUBackend, get_backend

        backend = get_backend(BackendType.GPU)
        assert isinstance(backend, GPUBackend)

    def test_auto_defaults_to_cpu(self):
        from vllm_gtc.backends import BackendType, CPUBackend, get_backend

        backend = get_backend(BackendType.AUTO)
        # In simulation, AUTO defaults to CPU
        assert isinstance(backend, CPUBackend)


class TestAttentionKernel:
    """Tests for attention_kernel."""

    def test_basic_attention(self):
        from vllm_gtc.backends import attention_kernel

        q = np.random.randn(1, 2, 4, 8).astype(np.float32)
        k = np.random.randn(1, 2, 4, 8).astype(np.float32)
        v = np.random.randn(1, 2, 4, 8).astype(np.float32)
        result = attention_kernel(q, k, v)
        assert result.shape == (1, 2, 4, 8)

    def test_custom_scale(self):
        from vllm_gtc.backends import attention_kernel

        q = np.ones((1, 1, 2, 4), dtype=np.float32)
        k = np.ones((1, 1, 2, 4), dtype=np.float32)
        v = np.ones((1, 1, 2, 4), dtype=np.float32)
        result = attention_kernel(q, k, v, scale=0.5)
        assert result.shape == (1, 1, 2, 4)


class TestFlashAttentionKernel:
    """Tests for flash_attention_kernel."""

    def test_matches_standard_attention(self):
        from vllm_gtc.backends import attention_kernel, flash_attention_kernel

        np.random.seed(42)
        q = np.random.randn(1, 2, 8, 16).astype(np.float32)
        k = np.random.randn(1, 2, 8, 16).astype(np.float32)
        v = np.random.randn(1, 2, 8, 16).astype(np.float32)

        standard = attention_kernel(q, k, v)
        flash = flash_attention_kernel(q, k, v, block_size=4)

        assert np.allclose(standard, flash, rtol=1e-4, atol=1e-4)

    def test_different_block_sizes(self):
        from vllm_gtc.backends import flash_attention_kernel

        q = np.random.randn(1, 2, 16, 8).astype(np.float32)
        k = np.random.randn(1, 2, 16, 8).astype(np.float32)
        v = np.random.randn(1, 2, 16, 8).astype(np.float32)

        result_4 = flash_attention_kernel(q, k, v, block_size=4)
        result_8 = flash_attention_kernel(q, k, v, block_size=8)

        # Different block sizes should give same result
        assert np.allclose(result_4, result_8, rtol=1e-4, atol=1e-4)


class TestSoftmaxKernel:
    """Tests for softmax_kernel."""

    def test_sums_to_one(self):
        from vllm_gtc.backends import softmax_kernel

        x = np.random.randn(2, 3, 4).astype(np.float32)
        result = softmax_kernel(x, axis=-1)
        assert np.allclose(result.sum(axis=-1), 1.0)

    def test_numerical_stability(self):
        from vllm_gtc.backends import softmax_kernel

        # Large values that would overflow without max subtraction
        x = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
        result = softmax_kernel(x, axis=-1)
        assert np.allclose(result.sum(axis=-1), 1.0)
        assert not np.any(np.isnan(result))


class TestLayernormKernel:
    """Tests for layernorm_kernel."""

    def test_basic_layernorm(self):
        from vllm_gtc.backends import layernorm_kernel

        x = np.random.randn(2, 4, 8).astype(np.float32)
        weight = np.ones(8, dtype=np.float32)
        bias = np.zeros(8, dtype=np.float32)
        result = layernorm_kernel(x, weight, bias)
        assert result.shape == x.shape

    def test_identity_transform(self):
        from vllm_gtc.backends import layernorm_kernel

        x = np.random.randn(2, 8).astype(np.float32)
        weight = np.ones(8, dtype=np.float32)
        bias = np.zeros(8, dtype=np.float32)
        result = layernorm_kernel(x, weight, bias)

        # Mean should be ~0, std should be ~1
        assert np.abs(result.mean(axis=-1)).max() < 1e-5
        assert np.abs(result.std(axis=-1) - 1.0).max() < 1e-5

    def test_with_affine(self):
        from vllm_gtc.backends import layernorm_kernel

        x = np.random.randn(2, 4).astype(np.float32)
        weight = np.ones(4, dtype=np.float32) * 2.0
        bias = np.ones(4, dtype=np.float32) * 0.5
        result = layernorm_kernel(x, weight, bias)

        # After scaling by 2 and adding 0.5
        # Mean should be ~0.5 (after scaling normalized values centered at 0)
        assert result.shape == x.shape
