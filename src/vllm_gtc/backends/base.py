"""Backend abstraction for CPU and GPU execution.

Provides a unified interface for running compute kernels on different
hardware backends. The pattern mirrors vLLM's executor abstraction.

References:
    - vLLM Executor: https://github.com/vllm-project/vllm
    - CUDA/CPU dispatch: Similar to PyTorch's dispatch mechanism

Rust cross-reference:
    repartir::executor provides CPU (Rayon), GPU (wgpu), and Remote
    executors with automatic backend selection based on workload.
    trueno::Backend enum provides Scalar/Simd/Gpu variants.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class BackendType(Enum):
    """Available compute backends."""

    CPU = auto()
    GPU = auto()
    AUTO = auto()  # Automatic selection based on availability


@dataclass
class DeviceInfo:
    """Information about a compute device.

    Rust equivalent:
        trueno::device::DeviceInfo

    """

    name: str
    backend: BackendType
    memory_bytes: int
    compute_capability: tuple[int, int] | None = None  # For GPU


class Backend(ABC):
    """Abstract base class for compute backends.

    Defines the interface for executing kernels on different hardware.
    Implementations handle memory management and kernel dispatch.

    Examples:
        >>> backend = get_backend(BackendType.CPU)
        >>> result = backend.matmul(a, b)

    Rust equivalent:
        repartir::executor::Executor trait with execute() method.

    """

    @property
    @abstractmethod
    def device_info(self) -> DeviceInfo:
        """Get information about the backend device."""

    @abstractmethod
    def matmul(
        self,
        a: NDArray[np.float32],
        b: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Matrix multiplication.

        Args:
            a: Left matrix (M, K).
            b: Right matrix (K, N).

        Returns:
            Result matrix (M, N).

        """

    @abstractmethod
    def softmax(
        self,
        x: NDArray[np.float32],
        axis: int = -1,
    ) -> NDArray[np.float32]:
        """Softmax along specified axis.

        Args:
            x: Input tensor.
            axis: Axis to apply softmax.

        Returns:
            Normalized probabilities.

        """

    @abstractmethod
    def layer_norm(
        self,
        x: NDArray[np.float32],
        weight: NDArray[np.float32],
        bias: NDArray[np.float32],
        eps: float = 1e-5,
    ) -> NDArray[np.float32]:
        """Layer normalization.

        Args:
            x: Input tensor (..., hidden_size).
            weight: Scale parameter (hidden_size,).
            bias: Shift parameter (hidden_size,).
            eps: Numerical stability constant.

        Returns:
            Normalized tensor.

        """

    @abstractmethod
    def attention(
        self,
        q: NDArray[np.float32],
        k: NDArray[np.float32],
        v: NDArray[np.float32],
        scale: float | None = None,
    ) -> NDArray[np.float32]:
        """Scaled dot-product attention.

        Args:
            q: Query tensor (batch, heads, seq_q, head_dim).
            k: Key tensor (batch, heads, seq_k, head_dim).
            v: Value tensor (batch, heads, seq_k, head_dim).
            scale: Attention scale (default: 1/sqrt(head_dim)).

        Returns:
            Attention output (batch, heads, seq_q, head_dim).

        """

    def synchronize(self) -> None:
        """Synchronize backend (wait for async operations).

        Default implementation is a no-op for CPU.
        GPU backends override to call cudaDeviceSynchronize or similar.

        """


class CPUBackend(Backend):
    """CPU backend using NumPy operations.

    Provides baseline implementations using NumPy. Suitable for
    development, testing, and systems without GPU.

    Examples:
        >>> backend = CPUBackend()
        >>> result = backend.matmul(a, b)

    Rust equivalent:
        repartir::executor::CpuExecutor using Rayon for parallelism.
        trueno::Backend::Simd for SIMD-accelerated operations.

    """

    def __init__(self, num_threads: int | None = None) -> None:
        """Initialize CPU backend.

        Args:
            num_threads: Number of threads (None = auto).

        """
        self._num_threads = num_threads

    @property
    def device_info(self) -> DeviceInfo:
        """Get CPU device information."""
        import os

        return DeviceInfo(
            name="CPU",
            backend=BackendType.CPU,
            memory_bytes=os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
        )

    def matmul(
        self,
        a: NDArray[np.float32],
        b: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Matrix multiplication using NumPy."""
        return np.matmul(a, b)

    def softmax(
        self,
        x: NDArray[np.float32],
        axis: int = -1,
    ) -> NDArray[np.float32]:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def layer_norm(
        self,
        x: NDArray[np.float32],
        weight: NDArray[np.float32],
        bias: NDArray[np.float32],
        eps: float = 1e-5,
    ) -> NDArray[np.float32]:
        """Layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return weight * x_norm + bias

    def attention(
        self,
        q: NDArray[np.float32],
        k: NDArray[np.float32],
        v: NDArray[np.float32],
        scale: float | None = None,
    ) -> NDArray[np.float32]:
        """Scaled dot-product attention."""
        if scale is None:
            scale = 1.0 / np.sqrt(q.shape[-1])

        # (batch, heads, seq_q, seq_k)
        scores = np.matmul(q, k.swapaxes(-2, -1)) * scale
        weights = self.softmax(scores, axis=-1)
        return np.matmul(weights, v)


class GPUBackend(Backend):
    """Simulated GPU backend for pattern demonstration.

    This implementation simulates GPU execution patterns without
    requiring actual CUDA. It demonstrates the API and memory
    management patterns used in production GPU backends.

    In production, this would use:
    - CUDA via CuPy or PyTorch
    - wgpu for cross-platform GPU compute
    - Triton for custom kernels

    Examples:
        >>> backend = GPUBackend(device_id=0)
        >>> result = backend.matmul(a, b)
        >>> backend.synchronize()

    Rust equivalent:
        repartir::executor::GpuExecutor using wgpu.
        trueno::Backend::Gpu with compute shader dispatch.

    """

    def __init__(self, device_id: int = 0, memory_bytes: int = 8 * 1024**3) -> None:
        """Initialize GPU backend.

        Args:
            device_id: GPU device index.
            memory_bytes: Simulated GPU memory (default: 8GB).

        """
        self._device_id = device_id
        self._memory_bytes = memory_bytes
        self._allocated: dict[int, int] = {}  # Track allocations

    @property
    def device_info(self) -> DeviceInfo:
        """Get GPU device information."""
        return DeviceInfo(
            name=f"GPU:{self._device_id}",
            backend=BackendType.GPU,
            memory_bytes=self._memory_bytes,
            compute_capability=(8, 0),  # Simulated Ampere
        )

    def _check_memory(self, size_bytes: int) -> None:
        """Check if allocation fits in GPU memory."""
        used = sum(self._allocated.values())
        if used + size_bytes > self._memory_bytes:
            msg = f"GPU OOM: need {size_bytes}, have {self._memory_bytes - used}"
            raise MemoryError(msg)

    def matmul(
        self,
        a: NDArray[np.float32],
        b: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Matrix multiplication (simulated GPU).

        In production: cuBLAS GEMM or custom tiled kernel.

        """
        # Simulate memory allocation check
        result_size = a.shape[0] * b.shape[1] * 4  # float32
        self._check_memory(result_size)

        # Use NumPy (in production: GPU kernel)
        return np.matmul(a, b)

    def softmax(
        self,
        x: NDArray[np.float32],
        axis: int = -1,
    ) -> NDArray[np.float32]:
        """Softmax (simulated GPU).

        In production: Fused softmax kernel with online computation.

        """
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def layer_norm(
        self,
        x: NDArray[np.float32],
        weight: NDArray[np.float32],
        bias: NDArray[np.float32],
        eps: float = 1e-5,
    ) -> NDArray[np.float32]:
        """Layer normalization (simulated GPU).

        In production: Fused kernel computing mean/var in one pass.

        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return weight * x_norm + bias

    def attention(
        self,
        q: NDArray[np.float32],
        k: NDArray[np.float32],
        v: NDArray[np.float32],
        scale: float | None = None,
    ) -> NDArray[np.float32]:
        """Scaled dot-product attention (simulated GPU).

        In production: FlashAttention or PagedAttention kernel.

        """
        if scale is None:
            scale = 1.0 / np.sqrt(q.shape[-1])

        scores = np.matmul(q, k.swapaxes(-2, -1)) * scale
        weights = self.softmax(scores, axis=-1)
        return np.matmul(weights, v)

    def synchronize(self) -> None:
        """Synchronize GPU operations.

        In production: cudaDeviceSynchronize() or wgpu queue.submit().

        """
        pass  # Simulated - no async ops in NumPy


def get_backend(backend_type: BackendType = BackendType.AUTO) -> Backend:
    """Get a compute backend instance.

    Args:
        backend_type: Desired backend type.

    Returns:
        Backend instance.

    Examples:
        >>> backend = get_backend(BackendType.CPU)
        >>> backend = get_backend(BackendType.AUTO)  # Picks best available

    Rust equivalent:
        repartir::Pool::new() with automatic executor selection.

    """
    if backend_type == BackendType.CPU:
        return CPUBackend()
    if backend_type == BackendType.GPU:
        return GPUBackend()

    # AUTO: Try GPU first, fall back to CPU
    # In production, would check CUDA/ROCm/Metal availability
    return CPUBackend()  # Default to CPU for simulation
