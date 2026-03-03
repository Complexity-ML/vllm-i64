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
- **Speculative decoding** — draft+verify for faster generation
- **OpenAI-compatible API** — `/v1/completions`, `/v1/chat/completions`, SSE streaming, WebSocket
- **Structured output** — JSON mode, regex constraints, stop sequences
- **LoRA hot-swap** — load/unload adapters at runtime via API
- **Sampling** — temperature, top-k, top-p, min-p, typical-p, repetition/frequency/presence penalties, beam search
- **Observability** — Prometheus metrics, latency percentiles, usage tracking, request logs

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
python -m vllm_i64.cli serve my-model --checkpoint ./model --port 8000
```

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_tokens": 50}'
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
| GET | `/docs` | OpenAPI 3.0 spec |

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
  api/
    server.py          # aiohttp OpenAI-compatible server
    middleware.py       # Auth, CORS, rate limiting, load shedding
    tracking.py         # Usage, latency, logging, priority
  core/
    kv_cache.py        # Paged KV cache with prefix caching
    sampling.py        # All sampling strategies + beam search
    speculative.py     # Speculative decoding
    cuda_graph.py      # CUDA graph capture for decode
  layers/
    lora.py            # LoRA adapter hot-swap
  models/              # Model implementations
tests/                 # 530+ tests
```

## Tests

```bash
pytest tests/ -v
```

## License

Apache 2.0 — INL / Complexity-ML, 2025
