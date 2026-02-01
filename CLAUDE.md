# CLAUDE.md

Ground truth corpus for vLLM high-performance inference patterns with Sovereign AI Stack mappings.

## Project Overview

Production-ready Python recipes demonstrating vLLM patterns that map to Rust implementations in the Sovereign AI Stack (realizar, repartir, aprender, trueno).

## Build Commands

```bash
uv sync              # Install dependencies
make test            # Run tests with coverage
make lint            # Ruff linting
make fmt             # Auto-format
make typecheck       # Mypy strict mode
```

## Module Structure

| Module | vLLM Concept | Rust Equivalent |
|--------|--------------|-----------------|
| `vllm_gtc.memory` | PagedAttention, KV-cache | `realizar::attention::AttentionKernel` |
| `vllm_gtc.engine` | LLMEngine, continuous batching | `realizar::engine::InferenceEngine` |
| `vllm_gtc.sampling` | SamplingParams, logits | `realizar::sampling::SamplingConfig` |
| `vllm_gtc.serving` | Request scheduling | `realizar::scheduler::RequestScheduler` |

## Sovereign Stack Mappings

### realizar (Inference)
- `PagedAttention` → `AttentionKernel` with tiled KV-cache
- `BlockSpaceManager` → Arena-based memory pools
- `Scheduler` → `RequestScheduler` with continuous batching
- `SamplingParams` → `SamplingConfig` (temp, top_p, top_k)

### repartir (Distributed)
- Tensor parallelism → `ShardedExecutor::shard_params`
- Pipeline parallelism → `Pipeline::from_stages`
- Ray workers → `RemoteExecutor` TCP/TLS

### aprender/entrenar (Quantization)
- AWQ/GPTQ → `QuantizeKernel` (Q4_K, Q5_K, Q6_K)
- FP8 → `entrenar::quantization::fp8`

### trueno (Primitives)
- Block allocator → `trueno::arena::Arena`
- GPU memory tracking → `trueno::device::MemoryPool`

## Quality Standards

- 95% minimum test coverage
- Strict mypy typing
- All patterns include Rust cross-references
- Docstrings with examples
