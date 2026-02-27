"""
vllm-i64 :: Generic layers for token-routed inference.
Model-agnostic — any token-routed architecture can use these.
"""

from vllm_i64.layers.token_routed_mlp import TokenRoutedMLP
from vllm_i64.layers.rmsnorm import RMSNorm
from vllm_i64.layers.rotary import RotaryEmbedding, apply_rotary
