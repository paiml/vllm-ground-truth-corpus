"""Sampling strategies for text generation.

Implements temperature scaling, top-k, and top-p (nucleus) sampling
as used in vLLM and other LLM inference engines.

References:
    - Top-p (Nucleus) Sampling: https://arxiv.org/abs/1904.09751
    - vLLM SamplingParams: https://github.com/vllm-project/vllm

Rust cross-reference:
    realizar::sampling::SamplingConfig provides equivalent
    temperature, top_k, top_p parameters.
    realizar::sampling::Sampler implements the sampling loop.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class SamplingParams:
    """Parameters controlling token sampling.

    Args:
        temperature: Softmax temperature (higher = more random).
        top_p: Nucleus sampling threshold (1.0 = disabled).
        top_k: Top-k filtering (-1 = disabled).
        max_tokens: Maximum tokens to generate.

    Examples:
        >>> params = SamplingParams(temperature=0.7, top_p=0.9)
        >>> params = SamplingParams.greedy()  # Deterministic

    Rust equivalent:
        realizar::sampling::SamplingConfig

    """

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 16

    @classmethod
    def greedy(cls) -> SamplingParams:
        """Create params for greedy (deterministic) decoding."""
        return cls(temperature=0.0, top_k=1)


def apply_temperature(logits: NDArray[np.float32], temperature: float) -> NDArray[np.float32]:
    """Apply temperature scaling to logits.

    Temperature controls randomness:
    - T < 1: Sharper distribution (more confident)
    - T = 1: Unchanged
    - T > 1: Smoother distribution (more random)

    Args:
        logits: Raw model output logits.
        temperature: Temperature value (must be > 0).

    Returns:
        Scaled logits.

    Raises:
        ValueError: If temperature <= 0.

    Examples:
        >>> logits = np.array([1.0, 2.0, 3.0])
        >>> apply_temperature(logits, 0.5)  # Sharper
        array([2., 4., 6.])

    Rust equivalent:
        realizar::sampling::apply_temperature

    """
    if temperature <= 0:
        msg = "temperature must be > 0 (use greedy sampling for deterministic)"
        raise ValueError(msg)

    return logits / temperature


def top_k_filter(logits: NDArray[np.float32], k: int) -> NDArray[np.float32]:
    """Filter logits to keep only top-k values.

    Sets all logits outside the top-k to -inf, effectively
    removing them from sampling consideration.

    Args:
        logits: Raw logits.
        k: Number of top values to keep (-1 to disable).

    Returns:
        Filtered logits with non-top-k set to -inf.

    Examples:
        >>> logits = np.array([1.0, 5.0, 3.0])
        >>> top_k_filter(logits, k=2)
        array([-inf,  5.,  3.])

    Rust equivalent:
        realizar::sampling::top_k_filter

    """
    if k < 0 or k >= len(logits):
        return logits.copy()

    result = np.full_like(logits, -np.inf)
    top_k_indices = np.argpartition(logits, -k)[-k:]
    result[top_k_indices] = logits[top_k_indices]
    return result


def top_p_filter(logits: NDArray[np.float32], p: float) -> NDArray[np.float32]:
    """Filter logits using nucleus (top-p) sampling.

    Keeps the smallest set of tokens whose cumulative probability
    exceeds p. This adapts the number of tokens based on the
    distribution shape.

    Args:
        logits: Raw logits.
        p: Cumulative probability threshold (0 to 1).

    Returns:
        Filtered logits with tail tokens set to -inf.

    Examples:
        >>> logits = np.array([1.0, 2.0, 10.0])  # Token 2 dominates
        >>> top_p_filter(logits, p=0.9)  # Keeps mainly token 2

    Rust equivalent:
        realizar::sampling::top_p_filter (nucleus sampling)

    """
    if p >= 1.0:
        return logits.copy()

    # Convert to probabilities
    probs = _softmax(logits)

    # Sort by probability descending
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    # Find cumulative probability cutoff
    cumsum = np.cumsum(sorted_probs)
    cutoff_idx = np.searchsorted(cumsum, p) + 1

    # Keep tokens up to cutoff
    result = np.full_like(logits, -np.inf)
    keep_indices = sorted_indices[:cutoff_idx]
    result[keep_indices] = logits[keep_indices]

    return result


def sample_token(logits: NDArray[np.float32], params: SamplingParams) -> int:
    """Sample a single token from logits.

    Applies temperature, top-k, and top-p filtering in sequence,
    then samples from the resulting distribution.

    Args:
        logits: Raw model output logits.
        params: Sampling parameters.

    Returns:
        Sampled token index.

    Examples:
        >>> logits = np.array([1.0, 2.0, 3.0])
        >>> params = SamplingParams(temperature=0.7, top_k=2)
        >>> token = sample_token(logits, params)

    Rust equivalent:
        realizar::sampling::Sampler::sample

    """
    # Greedy decoding: just return argmax
    if params.temperature == 0.0 or params.top_k == 1:
        return int(np.argmax(logits))

    # Apply temperature
    scaled = apply_temperature(logits, params.temperature)

    # Apply top-k
    if params.top_k > 0:
        scaled = top_k_filter(scaled, params.top_k)

    # Apply top-p
    if params.top_p < 1.0:
        scaled = top_p_filter(scaled, params.top_p)

    # Convert to probabilities and sample
    probs = _softmax(scaled)
    return int(np.random.choice(len(probs), p=probs))


def _softmax(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Numerically stable softmax."""
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)
