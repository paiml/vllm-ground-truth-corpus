# vLLM Ground Truth Corpus

Production-ready Python recipes for vLLM high-performance inference patterns with Sovereign AI Stack mappings.

## Overview

This corpus provides reference implementations of key vLLM concepts:

| Module | vLLM Concept | Rust Equivalent |
|--------|--------------|-----------------|
| `memory` | PagedAttention, KV-cache | `realizar::attention::AttentionKernel` |
| `engine` | Continuous batching | `realizar::scheduler::RequestScheduler` |
| `sampling` | Temperature, top-k/p | `realizar::sampling::SamplingConfig` |
| `serving` | Tensor parallelism | `repartir::ShardedExecutor` |

## Installation

```bash
uv sync
```

## Usage

```python
from vllm_gtc.memory import PagedAttentionConfig, KVCache, BlockAllocator
from vllm_gtc.sampling import SamplingParams, sample_token
from vllm_gtc.engine import Scheduler, Request

# Configure PagedAttention
config = PagedAttentionConfig(block_size=16, num_heads=32, head_dim=128)
cache = KVCache(config, num_blocks=1000)

# Sampling
params = SamplingParams(temperature=0.7, top_p=0.9)
token = sample_token(logits, params)

# Continuous batching
scheduler = Scheduler(SchedulerConfig(max_batch_size=256))
scheduler.add_request(Request("req-1", prompt_tokens=[1,2,3], max_tokens=100))
output = scheduler.schedule()
```

## Development

```bash
make test       # Run tests with coverage
make lint       # Ruff linting
make fmt        # Auto-format
make typecheck  # Mypy strict mode
```

## Sovereign Stack Mapping

These patterns inform Rust implementations in:

- **realizar** - Inference engine with GPU kernels
- **repartir** - Distributed compute with tensor/pipeline parallelism
- **trueno** - SIMD/GPU primitives and memory management

## License

MIT
