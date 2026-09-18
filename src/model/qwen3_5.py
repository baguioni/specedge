"""Qwen3.5 dense (text-only) model for SpecEdge tree decoding.

Qwen3.5 is a hybrid model: three of every four decoder layers are Gated DeltaNet
linear-attention layers, the rest are gated full-attention layers. The full
attention layers use a regular KV cache. The linear-attention layers keep a
recurrent state, which does not fit tree-structured speculation directly, so
``HybridCache`` stores the following:

* ``S``: the recurrent state after the committed prefix ``[0, commit_len)``.
* Per cache position ``j >= commit_len`` (the uncommitted draft / verify tree):
  the key ``k_j``, the delta-rule value ``v_new_j`` (relative to ``S``), the log
  decay ``g_j`` and the cumulative log decay ``G_j`` along the path from
  ``commit_len`` to ``j``.
* Per cache position: the pre-convolution projection ``x_j`` for the short
  causal convolution.

With these, a token ``i`` whose ancestors are given by its attention-mask row
has (all sums run over uncommitted ancestors along the tree path)::

    G_i     = sum_{j <= i} g_j
    v_new_i = beta_i * (v_i - exp(G_i) S^T k_i
                        - sum_{j < i} exp(G_i - G_j) (k_i . k_j) v_new_j)
    o_i     = exp(G_i) S^T q_i + sum_{j <= i} exp(G_i - G_j) (q_i . k_j) v_new_j

This is the chunked gated delta rule with the causal mask replaced by the tree
ancestor mask, so the output is exact for every tree node. New tokens of one
forward that are ancestors of each other (server-side verification of a whole
tree) are solved with a small unit lower-triangular system. This requires that
a token's ancestors come before it in the input order.

When ``HybridCache.gather`` compacts the cache to the verified path
(``dest_indices`` starting at 0), the path is folded into ``S`` and
``commit_len`` moves forward.

Reference: transformers/models/qwen3_5/modeling_qwen3_5.py
"""

import json
import os
from typing import Optional

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig

from model.cache import HybridCache


class Qwen3_5TextConfig(PretrainedConfig):
    model_type = "qwen3_5_text"

    def __init__(
        self,
        vocab_size: int = 248320,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: int = 256,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rope_parameters: Optional[dict] = None,
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        layer_types: Optional[list[str]] = None,
        full_attention_interval: int = 4,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_parameters = rope_parameters or {
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        }
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        if layer_types is None:
            layer_types = [
                "full_attention"
                if (i + 1) % full_attention_interval == 0
                else "linear_attention"
                for i in range(num_hidden_layers)
            ]
        self.layer_types = layer_types
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @classmethod
    def from_json_file_or_dict(cls, config: dict) -> "Qwen3_5TextConfig":
        """Build from a Qwen3.5 ``config.json`` (multimodal or text-only)."""
        text = dict(config.get("text_config", config))
        # The top-level flag wins for the multimodal checkpoints.
        text["tie_word_embeddings"] = config.get(
            "tie_word_embeddings", text.get("tie_word_embeddings", False)
        )
        for key in ("model_type", "architectures", "transformers_version", "dtype"):
            text.pop(key, None)
        return cls(**text)


class Qwen3_5RMSNorm(nn.Module):
    """Zero-centered RMSNorm: ``x * (1 + weight)``."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        output = x.float()
        output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


class Qwen3_5MLP(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3_5RotaryEmbedding(nn.Module):
    """Text-only RoPE.

    Qwen3.5 uses interleaved M-RoPE, but for text the temporal, height and width
    position ids are identical, so it reduces to plain partial RoPE.
    """

    def __init__(self, config: Qwen3_5TextConfig, device=None):
        super().__init__()
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        dim = int(config.head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        freqs = position_ids[..., None].float() * self.inv_freq.float()
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Partial RoPE: only the first ``cos.shape[-1]`` dims are rotated."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    chunk_size: int = 64,
):
    """Causal chunked gated delta rule, used for prefill.

    Inputs are fp32 in ``[B, H, T, D]`` (``g`` and ``beta``: ``[B, H, T]``) with
    ``query`` already L2-normalized and scaled, ``key`` L2-normalized.
    Returns the output ``[B, H, T, Dv]`` and the final state ``[B, H, Dk, Dv]``.
    """
    batch_size, num_heads, sequence_length, _ = key.shape
    v_head_dim = value.shape[-1]

    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, g = (F.pad(x, (0, pad_size)) for x in (beta, g))
    num_chunks = (sequence_length + pad_size) // chunk_size

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = (
        x.reshape(batch_size, num_heads, num_chunks, chunk_size, x.shape[-1])
        for x in (query, key, k_beta, v_beta)
    )
    g = g.reshape(batch_size, num_heads, num_chunks, chunk_size)

    strictly_upper = torch.ones(
        chunk_size, chunk_size, dtype=torch.bool, device=query.device
    ).triu(1)
    cum_g = g.cumsum(dim=-1)
    pairwise_decay = (cum_g.unsqueeze(-1) - cum_g.unsqueeze(-2)).masked_fill(
        strictly_upper, float("-inf")
    )
    pairwise_decay = pairwise_decay.exp()

    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_g.exp().unsqueeze(-1)
    new_values = torch.linalg.solve_triangular(
        ut_system, v_beta, upper=False, unitriangular=True
    )
    k_cumdecay = torch.linalg.solve_triangular(
        ut_system, decayed_k_beta, upper=False, unitriangular=True
    )

    state = initial_state.to(torch.float32)
    out = torch.zeros_like(new_values)

    query = query * cum_g.exp().unsqueeze(-1)
    key = key * (cum_g[..., -1:] - cum_g).exp().unsqueeze(-1)
    chunk_decay = cum_g[..., -1].exp()[..., None, None]

    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
        out[:, :, i] = query[:, :, i] @ state + intra_chunk_attn[:, :, i] @ v_new
        state = state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new

    out = out.reshape(batch_size, num_heads, -1, v_head_dim)[:, :, :sequence_length]
    return out, state


class Qwen3_5Attention(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int, cache_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # Index of this layer among the full-attention layers (KV cache slot).
        self.cache_idx = cache_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim * 2,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_batch_indices: torch.LongTensor,
        cache_seq_indices: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: HybridCache,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # q_proj interleaves [query_h | gate_h] per head.
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        key_states, value_states = past_key_values.update(
            key_states,
            value_states,
            self.cache_idx,
            cache_batch_indices,
            cache_seq_indices,
        )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(
            query_states.contiguous(),
            key_states.contiguous(),
            value_states.contiguous(),
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.scaling,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
        attn_output = attn_output * torch.sigmoid(gate)

        return self.o_proj(attn_output)


class TreeMeta:
    """Layer-independent tree bookkeeping, computed once per forward."""

    def __init__(
        self,
        attention_mask: torch.Tensor,
        cache_seq_indices: torch.Tensor,
        commit_len: torch.Tensor,
        conv_kernel_size: int,
    ):
        batch_size, _, n, max_len = attention_mask.shape
        # ancestors-or-self of every new token, over all cache positions
        ancestors = attention_mask[:, 0] == 0  # [B, n, L]
        cols = torch.arange(max_len, device=attention_mask.device)

        # Uncommitted ancestors: the columns held in the per-position stores.
        self.window_mask = ancestors & (cols >= commit_len[:, None, None])

        # Nearest ancestors along the path for the causal conv (k=0 is self).
        rank = ancestors.cumsum(dim=-1)
        depth = rank[..., -1:]
        conv_indices = []
        conv_valid = []
        for k in range(conv_kernel_size):
            target = depth - k
            onehot = ancestors & (rank == target)
            conv_indices.append((onehot * cols).sum(dim=-1))
            conv_valid.append(target[..., 0] >= 1)
        self.conv_indices = torch.stack(conv_indices, dim=-1)  # [B, n, K]
        self.conv_valid = torch.stack(conv_valid, dim=-1)  # [B, n, K]

        self.seq_indices = cache_seq_indices.view(batch_size, n)  # [B, n]
        self.batch_arange = torch.arange(batch_size, device=attention_mask.device)


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int, cache_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        # Index of this layer among the linear-attention layers (state slot).
        self.cache_idx = cache_idx

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

    def _split_qkv(self, mixed_qkv: torch.Tensor):
        """[B, n, conv_dim] -> fp32 q, k: [B, Hk, n, Dk], v: [B, Hv, n, Dv]."""
        batch_size, n, _ = mixed_qkv.shape
        query, key, value = torch.split(
            mixed_qkv.float(), [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = l2norm(query.reshape(batch_size, n, -1, self.head_k_dim))
        key = l2norm(key.reshape(batch_size, n, -1, self.head_k_dim))
        query = query * self.head_k_dim**-0.5
        value = value.reshape(batch_size, n, -1, self.head_v_dim)
        return query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

    def _group(self, x: torch.Tensor):
        """[B, Hv, m, D] -> [B, Hk, r * m, D]: value head h reads key head h // r."""
        batch_size, _, _, d = x.shape
        return x.reshape(batch_size, self.num_k_heads, -1, d)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_batch_indices: torch.LongTensor,
        cache_seq_indices: torch.LongTensor,
        past_key_values: HybridCache,
        tree_meta: Optional[TreeMeta],
    ):
        batch_size, n, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).reshape(batch_size, n, -1, self.head_v_dim)
        beta = self.in_proj_b(hidden_states).float().sigmoid().transpose(1, 2)
        g = (
            -self.A_log.float().exp()
            * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias.float())
        ).transpose(1, 2)  # [B, Hv, n]

        cache = past_key_values
        cache.x_store[self.cache_idx, cache_batch_indices, cache_seq_indices] = (
            mixed_qkv.reshape(batch_size * n, -1)
        )
        # conv weight [C, 1, K]; tap k multiplies x_{t-k}.
        conv_weight = self.conv1d.weight[:, 0, :].flip(-1).float()

        if tree_meta is None:
            core_attn_out = self._prefill(mixed_qkv, g, beta, conv_weight, cache)
        else:
            core_attn_out = self._tree(
                g,
                beta,
                conv_weight,
                cache,
                cache_batch_indices,
                cache_seq_indices,
                tree_meta,
            )

        core_attn_out = core_attn_out.transpose(1, 2).to(hidden_states.dtype)
        core_attn_out = self.norm(
            core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
        )
        return self.out_proj(core_attn_out.reshape(batch_size, n, -1))

    def _prefill(self, mixed_qkv, g, beta, conv_weight, cache: HybridCache):
        """Causal chain ``[commit_len, commit_len + n)`` starting from ``S``."""
        batch_size = mixed_qkv.shape[0]
        tail_len = self.conv_kernel_size - 1
        start = int(cache.commit_len[0].item())

        tail = mixed_qkv.new_zeros(batch_size, tail_len, self.conv_dim)
        lo = max(start - tail_len, 0)
        if start > 0:
            tail[:, tail_len - (start - lo) :] = cache.x_store[
                self.cache_idx, :, lo:start
            ]
        xs = torch.cat([tail, mixed_qkv], dim=1).float().transpose(1, 2)
        conv_out = F.conv1d(xs, conv_weight.flip(-1).unsqueeze(1), groups=self.conv_dim)
        conv_out = F.silu(conv_out).to(mixed_qkv.dtype).transpose(1, 2)

        query, key, value = self._split_qkv(conv_out)
        ratio = self.num_v_heads // self.num_k_heads
        query = query.repeat_interleave(ratio, dim=1)
        key = key.repeat_interleave(ratio, dim=1)

        state_slot = cache.state[self.cache_idx]
        out, final_state = chunk_gated_delta_rule(
            query, key, value, g, beta, initial_state=state_slot
        )
        state_slot.copy_(final_state)
        return out

    def _tree(
        self,
        g: torch.Tensor,
        beta: torch.Tensor,
        conv_weight: torch.Tensor,
        cache: HybridCache,
        cache_batch_indices: torch.Tensor,
        cache_seq_indices: torch.Tensor,
        meta: TreeMeta,
    ):
        idx = self.cache_idx
        batch_size, _, n = g.shape

        # Causal conv over the path: gather the K nearest ancestors' inputs.
        x_store = cache.x_store[idx]  # [B, L, C]
        xs = x_store[meta.batch_arange[:, None, None], meta.conv_indices]
        xs = xs.float() * meta.conv_valid[..., None]
        conv_out = F.silu((xs * conv_weight.T).sum(dim=2)).to(x_store.dtype)
        query, key, value = self._split_qkv(conv_out)  # q, k: Hk heads

        # Cumulative decay along the path from commit_len.
        g_store = cache.g_store[idx]  # [B, Hv, L]
        g_store[cache_batch_indices, :, cache_seq_indices] = g.transpose(1, 2).reshape(
            batch_size * n, -1
        )
        window = meta.window_mask.to(torch.float32)  # [B, n, L]
        G = torch.einsum("bnl,bhl->bhn", window, g_store)  # [B, Hv, n]

        k_store = cache.k_store[idx]  # [B, Hk, L, Dk]
        v_store = cache.v_store[idx]  # [B, Hv, L, Dv]
        G_store = cache.G_store[idx]  # [B, Hv, L]
        k_store[cache_batch_indices, :, cache_seq_indices] = key.transpose(
            1, 2
        ).reshape(batch_size * n, self.num_k_heads, -1)
        G_store[cache_batch_indices, :, cache_seq_indices] = G.transpose(1, 2).reshape(
            batch_size * n, -1
        )
        # New tokens enter the history sums through the triangular solve below.
        # (a device tensor, not a python scalar, so CUDA graph capture works)
        v_store[cache_batch_indices, :, cache_seq_indices] = v_store.new_zeros(())

        ratio = self.num_v_heads // self.num_k_heads
        key_v = key.repeat_interleave(ratio, dim=1)  # [B, Hv, n, Dk]
        query_v = query.repeat_interleave(ratio, dim=1)

        # Rows 0..n-1 read with q (output), rows n..2n-1 with k (delta rule).
        readers = torch.cat([query_v, key_v], dim=2)  # [B, Hv, 2n, Dk]
        G2 = torch.cat([G, G], dim=2)  # [B, Hv, 2n]
        scores = self._group(readers) @ k_store.transpose(-1, -2)
        scores = scores.view(batch_size, self.num_v_heads, 2 * n, -1)  # [B, Hv, 2n, L]
        decay = (G2[..., None] - G_store[:, :, None, :]).clamp(max=0.0)
        window2 = torch.cat([meta.window_mask, meta.window_mask], dim=1)[:, None]
        attn = torch.where(window2, scores * decay.exp(), 0.0)

        state = cache.state[idx]  # [B, Hv, Dk, Dv]
        from_state = (readers * G2[..., None].exp()) @ state
        from_history = attn @ v_store
        base = from_state + from_history  # [B, Hv, 2n, Dv]

        # Solve for v_new of the new tokens: (I + L) v_new = rhs.
        new_cols = meta.seq_indices[:, None, None, :].expand(
            batch_size, self.num_v_heads, 2 * n, n
        )
        attn_new = attn.gather(-1, new_cols)  # [B, Hv, 2n, n]
        rhs = beta[..., None] * (value - base[:, :, n:])
        lower = beta[..., None] * attn_new[:, :, n:]
        v_new = torch.linalg.solve_triangular(
            lower, rhs, upper=False, unitriangular=True
        )
        v_store[cache_batch_indices, :, cache_seq_indices] = v_new.transpose(
            1, 2
        ).reshape(batch_size * n, self.num_v_heads, -1)

        return base[:, :, :n] + attn_new[:, :, :n] @ v_new  # [B, Hv, n, Dv]


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int, cache_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx, cache_idx)
        else:
            self.self_attn = Qwen3_5Attention(config, layer_idx, cache_idx)
        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_batch_indices: torch.LongTensor,
        cache_seq_indices: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: HybridCache,
        tree_meta: Optional[TreeMeta] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices,
                past_key_values=past_key_values,
                tree_meta=tree_meta,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states,)


class Qwen3_5Model(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        layers = []
        n_linear = n_full = 0
        for layer_idx, layer_type in enumerate(config.layer_types):
            if layer_type == "linear_attention":
                layers.append(Qwen3_5DecoderLayer(config, layer_idx, n_linear))
                n_linear += 1
            else:
                layers.append(Qwen3_5DecoderLayer(config, layer_idx, n_full))
                n_full += 1
        self.layers = nn.ModuleList(layers)
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5RotaryEmbedding(config)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        cache_batch_indices: torch.LongTensor,
        cache_seq_indices: torch.LongTensor,
        attention_mask: torch.Tensor,
        past_key_values: HybridCache,
    ):
        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if past_key_values.prefilling:
            tree_meta = None
        else:
            tree_meta = TreeMeta(
                attention_mask,
                cache_seq_indices,
                past_key_values.commit_len,
                self.config.linear_conv_kernel_dim,
            )

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                tree_meta=tree_meta,
            )[0]

        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values


class Qwen3_5ForCausalLM(nn.Module):
    cache_cls = HybridCache

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @property
    def device(self) -> torch.device:
        return self.lm_head.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.lm_head.weight.dtype

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        cache_batch_indices: torch.LongTensor,
        cache_seq_indices: torch.LongTensor,
        attention_mask: torch.Tensor,
        past_key_values: HybridCache,
    ):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_batch_indices=cache_batch_indices,
            cache_seq_indices=cache_seq_indices,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(hidden_states)
        return logits, past_key_values

    @classmethod
    def from_pretrained(
        cls,
        name: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map="cpu",
        **kwargs,
    ) -> "Qwen3_5ForCausalLM":
        """Load the text decoder of a Qwen3.5 checkpoint.

        Vision tower (``model.visual.*``) and MTP head (``mtp.*``) weights are
        skipped.
        """
        path = (
            name
            if os.path.isdir(name)
            else snapshot_download(name, allow_patterns=["*.json", "*.safetensors"])
        )
        with open(os.path.join(path, "config.json")) as f:
            config = Qwen3_5TextConfig.from_json_file_or_dict(json.load(f))
        device = torch.device(device_map)

        with torch.device("meta"):
            model = cls(config)

        state_dict = {}
        for file in sorted(os.listdir(path)):
            if not file.endswith(".safetensors"):
                continue
            with safe_open(os.path.join(path, file), framework="pt") as f:
                for key in f.keys():
                    if key.startswith("model.language_model."):
                        new_key = "model." + key[len("model.language_model.") :]
                    elif key.startswith("model.") and not key.startswith(
                        "model.visual."
                    ):
                        new_key = key
                    elif key == "lm_head.weight":
                        new_key = key
                    else:
                        continue
                    state_dict[new_key] = f.get_tensor(key).to(
                        device=device, dtype=torch_dtype
                    )

        missing, unexpected = model.load_state_dict(
            state_dict, strict=False, assign=True
        )
        if config.tie_word_embeddings:
            model.lm_head.weight = model.model.embed_tokens.weight
            missing = [k for k in missing if k != "lm_head.weight"]
        if missing or unexpected:
            raise RuntimeError(
                f"Failed to load {name}: missing={missing}, unexpected={unexpected}"
            )

        # Non-persistent buffers are not in the checkpoint.
        model.model.rotary_emb = Qwen3_5RotaryEmbedding(config, device=device)
        return model
