"""Backend abstraction for CPU and GPU execution."""

from vllm_gtc.backends.base import (
    Backend,
    BackendType,
    CPUBackend,
    GPUBackend,
    get_backend,
)
from vllm_gtc.backends.kernels import (
    attention_kernel,
    flash_attention_kernel,
    layernorm_kernel,
    softmax_kernel,
)

__all__ = [
    "Backend",
    "BackendType",
    "CPUBackend",
    "GPUBackend",
    "get_backend",
    "attention_kernel",
    "flash_attention_kernel",
    "softmax_kernel",
    "layernorm_kernel",
]
