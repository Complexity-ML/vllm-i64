"""
vllm-i64 :: Sparse MoE Layers

Two routing strategies, same experts:

  MixtralMoE   — softmax gate (learned routing, HF Mixtral weights)
                  supports integer_mode: fixed-point INT32 softmax
  IntegerMoE   — token_id % num_experts (i64 integer routing, no gate)

Expert weights follow HuggingFace naming:
    w1 = gate projection (hidden → inter)
    w2 = down projection (inter → hidden)
    w3 = up projection   (hidden → inter)

Forward: y = w2(silu(w1(x)) * w3(x))  — same SwiGLU as dense.

Integer softmax (fixed-point):
    logits_i32 = round(logits * 2^15)           # Q15 fixed-point
    shifted = logits_i32 - max(logits_i32)       # numerical stability
    exp_i32 = lut_exp(shifted)                   # lookup table exp
    weights_i32 = exp_i32 / sum(exp_i32)         # integer division
    Same top-k selection, same expert dispatch.

INL - 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# Fixed-point integer softmax
# =========================================================================
#
# Q7 input quantization (scale=128), 1025-entry LUT for exp().
# Covers exp(-8.0) to exp(0.0). exp(-8) ≈ 0.0003 — negligible for softmax.
# Output scaled to Q16 (65536) for integer precision during normalization.

_Q_IN = 128             # 2^7 — input quantization scale
_Q_OUT = 1 << 16        # 2^16 — output scale for exp LUT values
_LUT_MIN = -1024        # minimum shifted value (= -8.0 in float at Q7)
_LUT_SIZE = -_LUT_MIN + 1  # 1025 entries: [-1024, 0]

def _build_exp_lut() -> torch.Tensor:
    """Build exp() LUT: integer index in [-1024, 0] → exp(index/128) * 2^16."""
    indices = torch.arange(_LUT_MIN, 1, dtype=torch.float32)
    return (torch.exp(indices / _Q_IN) * _Q_OUT).to(torch.int32)

_EXP_LUT = _build_exp_lut()


def softmax_integer(logits: torch.Tensor) -> torch.Tensor:
    """
    Fixed-point INT32 softmax — drop-in replacement for F.softmax(x, dim=-1).

    1. Quantize float logits to Q7 (×128 → INT32)
    2. Subtract row-max for stability (all values ≤ 0)
    3. Clamp to [-1024, 0] — below that, exp() ≈ 0
    4. exp() via 1025-entry LUT (Q16 output)
    5. Normalize: weight_i = exp_i / sum(exp)
    6. Return float (experts still compute in float)
    """
    # Q7 quantization — float32 for precision (bf16/fp16 mantissa too short)
    logits_i32 = (logits.float() * _Q_IN).round().to(torch.int32)

    # Subtract row max → all values ≤ 0
    row_max = logits_i32.max(dim=-1, keepdim=True).values
    shifted = logits_i32 - row_max

    # Clamp to LUT range — values below -1024 map to exp(-8)≈0
    shifted = shifted.clamp(min=_LUT_MIN)

    # LUT lookup: index = shifted - LUT_MIN maps [-1024,0] → [0,1024]
    lut = _EXP_LUT.to(shifted.device)
    table_idx = (shifted - _LUT_MIN).long()
    exp_vals = lut[table_idx]  # INT32, Q16 scaled

    # Normalize — integer division then back to float
    exp_sum = exp_vals.sum(dim=-1, keepdim=True).clamp(min=1)
    weights = exp_vals.float() / exp_sum.float()

    return weights


# =========================================================================
# Fixed-point SiLU LUT
# =========================================================================
#
# silu(x) = x * sigmoid(x). LUT covers [-8.0, 8.0] at Q7 resolution.
# Outside this range: silu(x) ≈ x for x >> 0, silu(x) ≈ 0 for x << 0.

_SILU_LUT_MIN = -1024    # -8.0 at Q7
_SILU_LUT_MAX = 1024     # +8.0 at Q7
_SILU_LUT_SIZE = _SILU_LUT_MAX - _SILU_LUT_MIN + 1  # 2049 entries

def _build_silu_lut() -> torch.Tensor:
    """Build SiLU LUT: integer index in [-1024, 1024] → silu(index/128) * 128."""
    indices = torch.arange(_SILU_LUT_MIN, _SILU_LUT_MAX + 1, dtype=torch.float32)
    x = indices / _Q_IN
    return (F.silu(x) * _Q_IN).round().to(torch.int32)

_SILU_LUT = _build_silu_lut()


def silu_integer(x_q7: torch.Tensor) -> torch.Tensor:
    """
    Fixed-point SiLU via LUT. Input/output in Q7 scale.

    Values outside [-8, 8]: silu(x) ≈ x for x > 8, ≈ 0 for x < -8.
    """
    lut = _SILU_LUT.to(x_q7.device)
    # Clamp to LUT range
    clamped = x_q7.clamp(_SILU_LUT_MIN, _SILU_LUT_MAX)
    indices = (clamped - _SILU_LUT_MIN).long()
    result = lut[indices]
    # Outside LUT: x > 8 → silu ≈ x, x < -8 → silu ≈ 0
    result = torch.where(x_q7 > _SILU_LUT_MAX, x_q7, result)
    result = torch.where(x_q7 < _SILU_LUT_MIN, torch.zeros_like(result), result)
    return result


def silu_multiply_integer(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Integer SiLU + multiply: silu(gate) * up, all in fixed-point INT32.

    Replaces F.silu(gate) * up with:
        1. Quantize gate/up to Q7 (×128, round to INT32)
        2. SiLU via 2049-entry LUT on gate
        3. Multiply in INT32 (Q7 × Q7 → Q14)
        4. Dequantize back to float (÷ 128²)

    Q7 covers [-8.0, 8.0] at 1/128 resolution.
    Outside: silu(x) ≈ x for x >> 0, ≈ 0 for x << 0.
    """
    # float32 for quantization — bf16 mantissa (8-bit) loses precision at Q7 scale
    gate_q7 = (gate.float() * _Q_IN).round().to(torch.int32)
    silu_q7 = silu_integer(gate_q7)
    up_q7 = (up.float() * _Q_IN).round().to(torch.int32)
    inter_q14 = silu_q7 * up_q7
    return inter_q14.float() / (_Q_IN * _Q_IN)


class MoEExpert(nn.Module):
    """
    Single expert MLP (SwiGLU). HF naming: w1/w2/w3.

    INT8 paths (checked in order):
      w13_int8  → fused gate+up: 1 quantization + 1 matmul (fast)
      w1_int8   → separate gate/up: 2 quantizations + 2 matmuls (fallback)
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'w13_int8'):
            return self._forward_int8_fused(x)
        if hasattr(self, 'w1_int8'):
            return self._forward_int8(x)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def _forward_int8_fused(self, x: torch.Tensor) -> torch.Tensor:
        """Fused gate+up: 1 quantization + 1 matmul for w1/w3, then w2.
        Integer SiLU LUT + INT32 gate*up multiply."""
        from vllm_i64.core.quantization import int8_fused_gate_up_native, int8_linear_native
        gate, up = int8_fused_gate_up_native(
            x, self.w13_int8, self.w13_scale, self.w13_inter,
        )
        inter = silu_multiply_integer(gate, up)
        return int8_linear_native(inter, self.w2_int8, self.w2_scale)

    def _forward_int8(self, x: torch.Tensor) -> torch.Tensor:
        """Separate gate/up INT8 matmuls (no fused weights available).
        Integer SiLU LUT + INT32 gate*up multiply."""
        from vllm_i64.core.quantization import int8_linear_native
        gate = int8_linear_native(x, self.w1_int8, self.w1_scale)
        up = int8_linear_native(x, self.w3_int8, self.w3_scale)
        inter = silu_multiply_integer(gate, up)
        return int8_linear_native(inter, self.w2_int8, self.w2_scale)


class MixtralMoE(nn.Module):
    """
    Sparse Mixture of Experts — top-k routing with softmax gate.

    Follows vllm/HuggingFace Mixtral conventions:
      - gate: Linear(hidden, num_experts) — router
      - experts: ModuleList of MoEExpert — individual MLPs
      - top_k: number of experts per token (typically 2)

    Weight names match HF checkpoints:
      block_sparse_moe.gate.weight
      block_sparse_moe.experts.{id}.w1.weight
      block_sparse_moe.experts.{id}.w2.weight
      block_sparse_moe.experts.{id}.w3.weight
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        integer_mode: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.integer_mode = integer_mode

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            MoEExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Sparse MoE forward.

        For each token, route to top_k experts via softmax gate,
        compute weighted sum of expert outputs.

        **kwargs absorbs token_ids, expert_ids, mu — MoE uses its own routing.
        """
        original_shape = x.shape
        # Flatten to [tokens, hidden]
        if x.dim() == 3:
            x = x.view(-1, x.shape[-1])

        num_tokens = x.shape[0]

        # Router: gate matmul (float, or native INT8 when quantized)
        if hasattr(self, 'gate_int8'):
            from vllm_i64.core.quantization import int8_linear_native
            router_logits = int8_linear_native(x, self.gate_int8, self.gate_scale)
        else:
            router_logits = self.gate(x)                       # [tokens, num_experts]
        if self.integer_mode:
            routing_weights = softmax_integer(router_logits)   # INT32 fixed-point
        else:
            routing_weights = F.softmax(router_logits, dim=-1) # float
        top_weights, top_indices = torch.topk(
            routing_weights, self.top_k, dim=-1,
        )                                                      # [tokens, top_k]
        # Renormalize selected weights
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        # Sorted expert dispatch — contiguous memory access per expert
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_indices[:, k]   # [tokens]
            expert_wt = top_weights[:, k]    # [tokens]

            # Sort tokens by expert assignment → contiguous slices
            sorted_order = expert_idx.argsort()
            sorted_x = x[sorted_order]
            sorted_expert = expert_idx[sorted_order]

            counts = torch.bincount(sorted_expert, minlength=self.num_experts)
            offset = 0
            sorted_out = torch.zeros_like(sorted_x)

            for e in range(self.num_experts):
                n = counts[e].item()
                if n == 0:
                    continue
                sorted_out[offset:offset + n] = self.experts[e](
                    sorted_x[offset:offset + n]
                )
                offset += n

            # Unsort back + weighted accumulate
            out[sorted_order] += expert_wt[sorted_order].unsqueeze(-1) * sorted_out

        if len(original_shape) == 3:
            out = out.view(original_shape)
        return out


def quantize_moe_int8(moe: "MixtralMoE") -> None:
    """
    Quantize a MixtralMoE in-place to INT8.

    Quantizes:
      - gate linear (router)
      - all expert w1/w2/w3 weights
      - pre-computes fused w13 = cat([w1, w3]) for fused gate+up matmul
    Sets integer_mode=True automatically.
    """
    from vllm_i64.core.quantization import quantize_int8

    # Gate (router)
    gate_int8, gate_scale = quantize_int8(moe.gate.weight.data)
    moe.gate_int8 = gate_int8
    moe.gate_scale = gate_scale

    # Experts — quantize + fuse gate/up
    for expert in moe.experts:
        quants = {}
        for name in ('w1', 'w2', 'w3'):
            layer = getattr(expert, name)
            q, s = quantize_int8(layer.weight.data)
            quants[name] = (q, s)
            expert.register_buffer(f'{name}_int8', q)
            expert.register_buffer(f'{name}_scale', s)
            layer.weight = None

        # Fused gate+up: cat([w1, w3], dim=0) → single matmul
        w1_q, w1_s = quants['w1']
        w3_q, w3_s = quants['w3']
        expert.register_buffer('w13_int8', torch.cat([w1_q, w3_q], dim=0))
        expert.register_buffer('w13_scale', torch.cat([w1_s, w3_s]))
        expert.w13_inter = w1_q.shape[0]  # split point = intermediate_size

    # Free gate float weights & enable integer mode
    moe.gate.weight = None
    moe.integer_mode = True


class IntegerMoE(nn.Module):
    """
    Integer-routed Mixture of Experts — i64 modulo routing.

    expert_id = token_id % num_experts

    No learned gate, no softmax. Each token goes to exactly one expert
    (top_k=1) determined by its token_id modulo num_experts.
    Reuses the same MoEExpert modules as MixtralMoE.

    Can load Mixtral expert weights — only the gate is discarded.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            MoEExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Integer MoE forward.

        token_ids: [tokens] — required for modulo routing.
        If not provided, falls back to positional index.
        """
        original_shape = x.shape
        if x.dim() == 3:
            x = x.view(-1, x.shape[-1])

        num_tokens = x.shape[0]

        # Integer routing: expert_id = token_id % num_experts
        if token_ids is not None:
            if token_ids.dim() > 1:
                token_ids = token_ids.view(-1)
            expert_ids = token_ids % self.num_experts
        else:
            expert_ids = torch.arange(num_tokens, device=x.device) % self.num_experts

        # Dispatch — each token to exactly one expert, weight = 1.0
        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            mask = expert_ids == e
            if not mask.any():
                continue
            out[mask] = self.experts[e](x[mask])

        if len(original_shape) == 3:
            out = out.view(original_shape)
        return out
