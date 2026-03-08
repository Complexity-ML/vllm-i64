# vllm-i64

**Integer-first inference engine for token-routed language models.**

All control flow is integer (`i64`/`i32`). Float exists only inside `model.forward()`.

[![CI](https://github.com/Complexity-ML/vllm-i64/actions/workflows/ci.yml/badge.svg)](https://github.com/Complexity-ML/vllm-i64/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## Features

- **Async continuous batching** — multiple requests batched per forward pass
- **Paged KV cache** — block-level memory management with LRU eviction
- **Chunked prefill** — long prompts split across steps, mixed with decode
- **OpenAI-compatible API** — `/v1/completions`, `/v1/chat/completions`, SSE streaming, WebSocket
- **CPU engine** — dedicated CPU inference path, no CUDA required
- **GPU kernels** — Triton fused experts, CUDA FP8 tensor cores, INT8/INT4 quantization
- **Dense model support** — Llama, Mistral, Mixtral, Qwen2 (HuggingFace checkpoints)
- **Structured output** — JSON mode, regex constraints, stop sequences
- **Sampling** — temperature, top-k, top-p, min-p, typical-p, repetition/frequency/presence penalties
- **Speculative decoding** — draft+verify (opt-in via `engine.enable_speculative()`)
- **LoRA** — load/unload adapters at runtime (opt-in via `engine.enable_lora()`)

- **RAG** — native retrieval pipeline (chunk → embed → FAISS → retrieve → generate)
- **Agentic tool use** — ReAct agent loop with parallel tool execution
- **Observability** — JSON metrics, latency percentiles, usage tracking, request logs
- **Security** — token-routed partition isolation, no session tokens, no data leak possible

## Quick start

```bash
pip install -e .
```

```python
from vllm_i64.engine.i64_engine import I64Engine

engine = I64Engine(model=my_model, num_experts=4, vocab_size=32000)
result = engine.generate(prompt_token_ids=[1, 2, 3, 4, 5], max_new_tokens=100)
print(result.output_tokens)
```

### Serve

```bash
# GPU
python -m vllm_i64.cli serve my-model --checkpoint ./model --port 8000

# CPU (no CUDA required)
python -m vllm_i64.cli serve my-model --checkpoint ./model --port 8000 --no-cuda-graphs

# Limit VRAM
python -m vllm_i64.cli serve my-model --checkpoint ./model --max-kv-blocks 128
```

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "my-model", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/completions` | Text completion (sync + streaming) |
| POST | `/v1/chat/completions` | Chat completion |
| POST | `/v1/batch` | Batch multiple prompts |
| POST | `/v1/cancel/{id}` | Cancel a running request |
| GET | `/v1/ws/completions` | WebSocket streaming |
| GET | `/v1/models` | List models |
| GET | `/v1/models/{id}` | Model details |
| GET | `/health` | Health check + diagnostics |
| GET | `/v1/metrics` | Latency & usage metrics |

| POST | `/v1/rag/index` | Index documents for RAG |
| POST | `/v1/rag/search` | Search indexed documents |
| GET | `/docs` | OpenAPI 3.0 spec |

## Security — Token-Routed Isolation

```
partition = sha256(api_key ∥ user_id) mod N
```

The same deterministic routing that drives MoE inference is applied to user data isolation. Search history, context, and session data are partitioned per-identity — **no cross-user access path exists in the code**.

- **No shared cache** — each identity routes to its own isolated partition
- **No session tokens** — auth is stateless (API key + user_id per request), eliminating session hijacking
- **Team key safe** — shared API keys are split by `user_id`, so Alice never sees Bob's history
- **Blast radius = 1** — a compromised key only accesses its own partition

- **No data leak possible** — if you can't address a partition, you can't read it

## Architecture

```
text --> tokenize --> i64 token IDs
  --> i64 routing    (token_id & mask --> expert_id)
  --> i64 scatter    (group by expert, integer indices)
  --> fp16 forward   (expert MLP + attention)
  --> i64 sampling   (top-k/top-p/argmax --> i64 token ID)
  --> detokenize --> text
```

| Component | Type | Float? |
|-----------|------|--------|
| Token routing | `i64` bitmask | No |
| KV cache mgmt | `i32` block table | No |
| Scheduling | `i32`/`i64` counters | No |
| Sampling | `i64` argmax | No |
| **Model forward** | **fp16** | **Yes** |

## Project structure

```
vllm_i64/
  engine/
    i64_engine.py      # Sync + async engine, continuous batching
    i64_scheduler.py   # Integer-first scheduler with preemption
  cpu/
    engine.py          # Dedicated CPU engine (no CUDA, thread executor)
  api/
    server.py          # aiohttp OpenAI-compatible server
    middleware.py      # Auth, CORS, rate limiting, load shedding
    tracking.py        # Usage, latency, logging, priority
  core/
    kv_cache.py        # Paged KV cache with LRU eviction
    sampling.py        # All sampling strategies
    loader.py          # Checkpoint loading (FP16, INT8, INT4)
    compile.py         # torch.compile integration
  kernels/
    cuda/              # CUDA kernels (FP8, INT8, attention)
    triton/            # Triton fused expert kernels
  layers/
    attention.py       # GQA attention (flash, paged, naive)
    rmsnorm.py         # RMSNorm (float + integer paths)
    rotary.py          # RoPE (float + integer Q14 LUT)
  models/
    complexity_deep/   # Token-routed MoE (Pacific-Prime / INL)
    llama/             # Llama-family dense models
    mistral/           # Mistral
    mixtral/           # Mixtral MoE
    qwen2/             # Qwen2
tests/                 # 650+ tests
```

## Tests

```bash
pytest tests/ -v
```

## License

Apache 2.0 — INL / Complexity-ML, 2025
