"""Compute kernel patterns for GPU and CPU.

Demonstrates kernel implementation patterns used in vLLM including
tiled computation, memory coalescing, and fusion strategies.

References:
    - FlashAttention: https://arxiv.org/abs/2205.14135
    - Triton tutorials: https://triton-lang.org/main/getting-started/tutorials/

Rust cross-reference:
    realizar::kernels provides WGSL compute shaders for GPU execution.
    trueno::simd provides AVX2/AVX-512/NEON CPU implementations.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class KernelConfig:
    """Configuration for kernel execution.

    Controls tiling, memory layout, and execution parameters.

    Rust equivalent:
        realizar::kernels::KernelConfig with workgroup sizes.

    """

    block_size: int = 64
    num_warps: int = 4
    num_stages: int = 2
    use_tensor_cores: bool = True


def attention_kernel(
    q: NDArray[np.float32],
    k: NDArray[np.float32],
    v: NDArray[np.float32],
    scale: float | None = None,
    config: KernelConfig | None = None,
) -> NDArray[np.float32]:
    """Standard attention kernel (CPU reference implementation).

    Computes scaled dot-product attention with explicit loops to
    demonstrate the computation pattern used in GPU kernels.

    Args:
        q: Query tensor (batch, heads, seq_q, head_dim).
        k: Key tensor (batch, heads, seq_k, head_dim).
        v: Value tensor (batch, heads, seq_k, head_dim).
        scale: Attention scale (default: 1/sqrt(head_dim)).
        config: Kernel configuration (unused in reference impl).

    Returns:
        Attention output (batch, heads, seq_q, head_dim).

    Examples:
        >>> q = np.random.randn(1, 4, 8, 64).astype(np.float32)
        >>> k = np.random.randn(1, 4, 8, 64).astype(np.float32)
        >>> v = np.random.randn(1, 4, 8, 64).astype(np.float32)
        >>> out = attention_kernel(q, k, v)
        >>> out.shape
        (1, 4, 8, 64)

    Rust equivalent:
        realizar::kernels::AttentionKernel with naive implementation.
        GPU version uses shared memory tiling for O(N) memory.

    """
    if scale is None:
        scale = 1.0 / np.sqrt(q.shape[-1])

    batch, heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    output = np.zeros((batch, heads, seq_q, head_dim), dtype=np.float32)

    for b in range(batch):
        for h in range(heads):
            # Compute attention scores
            scores = np.zeros((seq_q, seq_k), dtype=np.float32)
            for i in range(seq_q):
                for j in range(seq_k):
                    scores[i, j] = np.dot(q[b, h, i], k[b, h, j]) * scale

            # Softmax
            scores_max = np.max(scores, axis=-1, keepdims=True)
            scores_exp = np.exp(scores - scores_max)
            scores_sum = np.sum(scores_exp, axis=-1, keepdims=True)
            weights = scores_exp / scores_sum

            # Weighted sum
            for i in range(seq_q):
                for j in range(seq_k):
                    output[b, h, i] += weights[i, j] * v[b, h, j]

    return output


def flash_attention_kernel(
    q: NDArray[np.float32],
    k: NDArray[np.float32],
    v: NDArray[np.float32],
    scale: float | None = None,
    block_size: int = 64,
) -> NDArray[np.float32]:
    """FlashAttention-style tiled attention kernel.

    Implements the online softmax algorithm from FlashAttention to
    compute attention with O(N) memory instead of O(N^2).

    The algorithm processes K/V in blocks, maintaining running
    statistics (max, sum) for numerically stable online softmax.

    Args:
        q: Query tensor (batch, heads, seq_q, head_dim).
        k: Key tensor (batch, heads, seq_k, head_dim).
        v: Value tensor (batch, heads, seq_k, head_dim).
        scale: Attention scale (default: 1/sqrt(head_dim)).
        block_size: Tile size for K/V processing.

    Returns:
        Attention output (batch, heads, seq_q, head_dim).

    Examples:
        >>> q = np.random.randn(1, 4, 128, 64).astype(np.float32)
        >>> k = np.random.randn(1, 4, 128, 64).astype(np.float32)
        >>> v = np.random.randn(1, 4, 128, 64).astype(np.float32)
        >>> out = flash_attention_kernel(q, k, v, block_size=32)
        >>> out.shape
        (1, 4, 128, 64)

    Rust equivalent:
        realizar::kernels::FlashAttentionKernel with WGSL compute shader.
        Uses workgroup shared memory for K/V tiles.

    """
    if scale is None:
        scale = 1.0 / np.sqrt(q.shape[-1])

    batch, heads, seq_q, head_dim = q.shape
    _, _, seq_k, _ = k.shape

    output = np.zeros((batch, heads, seq_q, head_dim), dtype=np.float32)

    for b in range(batch):
        for h in range(heads):
            for i in range(seq_q):
                # Online softmax state
                m_i = -np.inf  # Running max
                l_i = 0.0  # Running sum
                acc = np.zeros(head_dim, dtype=np.float32)

                # Process K/V in blocks
                for block_start in range(0, seq_k, block_size):
                    block_end = min(block_start + block_size, seq_k)

                    # Compute scores for this block
                    scores_block = np.zeros(block_end - block_start, dtype=np.float32)
                    for j in range(block_start, block_end):
                        scores_block[j - block_start] = np.dot(q[b, h, i], k[b, h, j]) * scale

                    # Online softmax update
                    m_block = np.max(scores_block)
                    m_new = max(m_i, m_block)

                    # Rescale previous accumulator
                    if m_i > -np.inf:
                        acc = acc * np.exp(m_i - m_new)
                        l_i = l_i * np.exp(m_i - m_new)

                    # Add contribution from this block
                    exp_scores = np.exp(scores_block - m_new)
                    l_i += np.sum(exp_scores)

                    for j in range(block_start, block_end):
                        acc += exp_scores[j - block_start] * v[b, h, j]

                    m_i = m_new

                # Normalize
                output[b, h, i] = acc / l_i

    return output


def softmax_kernel(
    x: NDArray[np.float32],
    axis: int = -1,
) -> NDArray[np.float32]:
    """Optimized softmax kernel with online computation.

    Uses the two-pass algorithm: first pass computes max,
    second pass computes exp and sum. This is numerically stable
    and maps well to GPU parallel reduction patterns.

    Args:
        x: Input tensor.
        axis: Axis to apply softmax.

    Returns:
        Softmax probabilities.

    Examples:
        >>> x = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        >>> probs = softmax_kernel(x, axis=-1)
        >>> np.allclose(probs.sum(axis=-1), 1.0)
        True

    Rust equivalent:
        realizar::kernels::SoftmaxKernel with warp shuffle reduction.
        trueno::simd::softmax for CPU SIMD implementation.

    """
    # Pass 1: Find max (for numerical stability)
    x_max = np.max(x, axis=axis, keepdims=True)

    # Pass 2: Compute exp and sum
    exp_x = np.exp(x - x_max)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)

    return exp_x / sum_exp


def layernorm_kernel(
    x: NDArray[np.float32],
    weight: NDArray[np.float32],
    bias: NDArray[np.float32],
    eps: float = 1e-5,
) -> NDArray[np.float32]:
    """Fused layer normalization kernel.

    Computes mean and variance in a single pass using Welford's
    algorithm, then applies normalization and affine transform.

    Args:
        x: Input tensor (..., hidden_size).
        weight: Scale parameter (hidden_size,).
        bias: Shift parameter (hidden_size,).
        eps: Numerical stability constant.

    Returns:
        Normalized tensor.

    Examples:
        >>> x = np.random.randn(2, 4, 256).astype(np.float32)
        >>> w = np.ones(256, dtype=np.float32)
        >>> b = np.zeros(256, dtype=np.float32)
        >>> out = layernorm_kernel(x, w, b)
        >>> out.shape
        (2, 4, 256)

    Rust equivalent:
        realizar::kernels::LayerNormKernel with fused computation.
        trueno::simd::layernorm for CPU SIMD implementation.

    """
    # Welford's online algorithm for mean and variance
    # More numerically stable than two-pass
    hidden_size = x.shape[-1]

    # Reshape for easier iteration
    x_flat = x.reshape(-1, hidden_size)
    output = np.zeros_like(x_flat)

    for i in range(x_flat.shape[0]):
        mean = 0.0
        m2 = 0.0

        for j in range(hidden_size):
            delta = x_flat[i, j] - mean
            mean += delta / (j + 1)
            delta2 = x_flat[i, j] - mean
            m2 += delta * delta2

        var = m2 / hidden_size
        inv_std = 1.0 / np.sqrt(var + eps)

        for j in range(hidden_size):
            output[i, j] = (x_flat[i, j] - mean) * inv_std * weight[j] + bias[j]

    return output.reshape(x.shape)
