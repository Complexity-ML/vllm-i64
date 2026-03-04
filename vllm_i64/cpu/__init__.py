"""vllm-i64 :: CPU inference package"""
from vllm_i64.cpu.engine import CPUEngine, AsyncCPUEngine

__all__ = ["CPUEngine", "AsyncCPUEngine"]
