"""Tests for vllm_gtc.sampling module."""

from __future__ import annotations

import numpy as np
import pytest


class TestSamplingParams:
    """Tests for SamplingParams."""

    def test_default_params(self):
        from vllm_gtc.sampling import SamplingParams

        params = SamplingParams()
        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.top_k == -1
        assert params.max_tokens == 16

    def test_custom_params(self):
        from vllm_gtc.sampling import SamplingParams

        params = SamplingParams(temperature=0.7, top_p=0.9, top_k=50, max_tokens=100)
        assert params.temperature == 0.7
        assert params.top_p == 0.9
        assert params.top_k == 50
        assert params.max_tokens == 100

    def test_greedy_decoding(self):
        from vllm_gtc.sampling import SamplingParams

        params = SamplingParams.greedy()
        assert params.temperature == 0.0
        assert params.top_k == 1


class TestApplyTemperature:
    """Tests for apply_temperature."""

    def test_temperature_one_unchanged(self):
        from vllm_gtc.sampling import apply_temperature

        logits = np.array([1.0, 2.0, 3.0])
        result = apply_temperature(logits, temperature=1.0)
        assert np.allclose(result, logits)

    def test_temperature_sharpens(self):
        from vllm_gtc.sampling import apply_temperature

        logits = np.array([1.0, 2.0, 3.0])
        result = apply_temperature(logits, temperature=0.5)
        # Lower temp = sharper distribution (larger differences)
        assert result[2] - result[1] > logits[2] - logits[1]

    def test_temperature_smooths(self):
        from vllm_gtc.sampling import apply_temperature

        logits = np.array([1.0, 2.0, 3.0])
        result = apply_temperature(logits, temperature=2.0)
        # Higher temp = smoother distribution (smaller differences)
        assert result[2] - result[1] < logits[2] - logits[1]

    def test_temperature_zero_raises(self):
        from vllm_gtc.sampling import apply_temperature

        logits = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="temperature"):
            apply_temperature(logits, temperature=0.0)


class TestTopKFilter:
    """Tests for top_k_filter."""

    def test_top_k_keeps_k_values(self):
        from vllm_gtc.sampling import top_k_filter

        logits = np.array([1.0, 5.0, 3.0, 4.0, 2.0])
        result = top_k_filter(logits, k=3)
        # Should keep top 3: indices 1, 3, 2 (values 5, 4, 3)
        non_neg_inf = result[result > -np.inf]
        assert len(non_neg_inf) == 3

    def test_top_k_preserves_values(self):
        from vllm_gtc.sampling import top_k_filter

        logits = np.array([1.0, 5.0, 3.0])
        result = top_k_filter(logits, k=2)
        # Top 2 are indices 1 and 2
        assert result[1] == 5.0
        assert result[2] == 3.0
        assert result[0] == -np.inf

    def test_top_k_negative_disables(self):
        from vllm_gtc.sampling import top_k_filter

        logits = np.array([1.0, 2.0, 3.0])
        result = top_k_filter(logits, k=-1)
        assert np.allclose(result, logits)

    def test_top_k_larger_than_vocab(self):
        from vllm_gtc.sampling import top_k_filter

        logits = np.array([1.0, 2.0, 3.0])
        result = top_k_filter(logits, k=10)
        assert np.allclose(result, logits)


class TestTopPFilter:
    """Tests for top_p_filter (nucleus sampling)."""

    def test_top_p_one_unchanged(self):
        from vllm_gtc.sampling import top_p_filter

        logits = np.array([1.0, 2.0, 3.0])
        result = top_p_filter(logits, p=1.0)
        assert np.allclose(result, logits)

    def test_top_p_filters_tail(self):
        from vllm_gtc.sampling import top_p_filter

        # Create logits where top token has most probability
        logits = np.array([0.0, 0.0, 10.0])  # Token 2 dominates
        result = top_p_filter(logits, p=0.5)
        # Should keep mainly the top token
        assert result[2] == 10.0

    def test_top_p_keeps_cumulative(self):
        from vllm_gtc.sampling import top_p_filter

        # Uniform-ish logits
        logits = np.array([1.0, 1.0, 1.0, 1.0])
        result = top_p_filter(logits, p=0.5)
        # Should keep approximately half the tokens
        non_neg_inf = result[result > -np.inf]
        assert len(non_neg_inf) >= 1


class TestSampleToken:
    """Tests for sample_token."""

    def test_greedy_sampling(self):
        from vllm_gtc.sampling import SamplingParams, sample_token

        logits = np.array([1.0, 5.0, 3.0])
        params = SamplingParams.greedy()
        token = sample_token(logits, params)
        assert token == 1  # Highest logit

    def test_sampling_returns_valid_token(self):
        from vllm_gtc.sampling import SamplingParams, sample_token

        logits = np.array([1.0, 2.0, 3.0])
        params = SamplingParams(temperature=1.0)
        token = sample_token(logits, params)
        assert 0 <= token < len(logits)

    def test_sampling_with_top_k(self):
        from vllm_gtc.sampling import SamplingParams, sample_token

        logits = np.array([1.0, 5.0, 3.0, 4.0, 2.0])
        params = SamplingParams(temperature=1.0, top_k=2)
        # Should only sample from top 2 (indices 1 and 3)
        tokens = [sample_token(logits, params) for _ in range(100)]
        unique = set(tokens)
        assert unique.issubset({1, 3})

    def test_sampling_with_top_p(self):
        from vllm_gtc.sampling import SamplingParams, sample_token

        # Token 0 has 99% probability after softmax
        logits = np.array([10.0, 0.0, 0.0])
        params = SamplingParams(temperature=1.0, top_p=0.9)
        tokens = [sample_token(logits, params) for _ in range(50)]
        # Should mostly sample token 0
        assert tokens.count(0) > 40

    def test_sampling_reproducible_with_seed(self):
        from vllm_gtc.sampling import SamplingParams, sample_token

        logits = np.array([1.0, 2.0, 3.0])
        params = SamplingParams(temperature=1.0)

        np.random.seed(42)
        t1 = sample_token(logits, params)

        np.random.seed(42)
        t2 = sample_token(logits, params)

        assert t1 == t2
