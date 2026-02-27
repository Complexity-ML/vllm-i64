# vllm-i64

Integer-first inference engine for token-routed language models.

**All control flow is integer. Float exists only inside expert MLP compute.**

## Architecture

```
Text → tokenize → i64 token IDs
  → i64 routing (token_id & mask → expert_id)
  → i64 scatter (group tokens by expert, integer indices)
  → FP16 expert MLP (SwiGLU — the ONLY float operation)
  → i64 gather (restore token order, integer indices)
  → i64 sampling (argmax over logits → i64 token ID)
→ detokenize → Text
```

## What is integer

| Component | Data type | Float ops |
|-----------|-----------|-----------|
| Token routing | i64 bit mask | 0 |
| Scatter/gather | i32 indices | 0 |
| KV cache management | i32 block table | 0 |
| Scheduling | i32/i64 counters | 0 |
| Sampling | i64 argmax | 0 |
| **Expert MLP (SwiGLU)** | **FP16** | **yes** |

## Project structure

```
csrc/
  i64_router.cu          # CUDA routing + scatter/gather kernels
  i64_expert_dispatch.cu  # CUDA expert MLP dispatch (cuBLAS HGEMM)

vllm_i64/
  kernels/i64_ops.py     # Pure PyTorch fallback (CPU, no CUDA required)
  engine/i64_scheduler.py # Integer-first continuous batching scheduler
  engine/i64_engine.py    # Main inference engine
  api/server.py           # OpenAI-compatible API server
  models/                 # Model implementations

tests/test_i64_pipeline.py    # Full test suite + integer purity audit
benchmarks/bench_i64_routing.py  # i64 routing vs float routing benchmark
```

## Quick start

```python
from vllm_i64.engine.i64_engine import I64Engine

engine = I64Engine(model=None, num_experts=4, vocab_size=32000)
result = engine.generate(prompt_token_ids=[1, 2, 3, 4, 5], max_new_tokens=20)
print(result.output_tokens)
```

## Tests

```bash
pytest tests/ -v
```

## Benchmark

```bash
python benchmarks/bench_i64_routing.py
```

## License

INL - 2025
