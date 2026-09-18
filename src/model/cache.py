from contextlib import contextmanager
from typing import Optional

import torch
from transformers.models.llama.configuration_llama import LlamaConfig


class KVCache:
    """
    KVCache is a key-value cache that stores the key-value pairs in the memory.
    It is used to store KV cache in custom llama model
    due to CUDA Graph capture limitation.
    """

    def __init__(
        self,
        config: LlamaConfig,
        max_n_beams: int,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
    ) -> None:
        self.model_config = config
        self.max_len = max_len
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size

        self.num_layers = self.model_config.num_hidden_layers
        self.head_dim = getattr(
            config,
            "head_dim",
            self.model_config.hidden_size // self.model_config.num_attention_heads,
        )
        self._offset = torch.zeros(
            self.batch_size, device=self.device, dtype=torch.long
        )

        self.seq_indices = torch.arange(
            max_n_beams, device=self.device, dtype=torch.long
        ).repeat(batch_size)

        self.k_cache = torch.zeros(
            (
                self.num_layers,
                self.batch_size,
                self.model_config.num_key_value_heads,
                self.max_len,
                self.head_dim,
            ),
            device=self.device,
            dtype=self.dtype,
        )

        self.v_cache = torch.zeros(
            (
                self.num_layers,
                self.batch_size,
                self.model_config.num_key_value_heads,
                self.max_len,
                self.head_dim,
            ),
            device=self.device,
            dtype=self.dtype,
        )

    def gather(
        self, batch_idx: int, src_indices: torch.Tensor, dest_indices: torch.Tensor
    ):
        """
        Remove the key-value pairs that are not used in the current batch.
        """
        src_indices = src_indices.to(self.device)
        dest_indices = dest_indices.to(self.device)

        if src_indices.dtype == torch.bool:
            src_indices = torch.where(src_indices)[0]

        self.k_cache[:, batch_idx, :, dest_indices, :] = self.k_cache[
            :, batch_idx, :, src_indices, :
        ]
        self.v_cache[:, batch_idx, :, dest_indices, :] = self.v_cache[
            :, batch_idx, :, src_indices, :
        ]

        self.k_cache[:, batch_idx, :, dest_indices.max() + 1 :, :].zero_()
        self.v_cache[:, batch_idx, :, dest_indices.max() + 1 :, :].zero_()

    def update(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        layer_idx: int,
        cache_batch_indices: torch.Tensor,
        cache_seq_indices: torch.Tensor,
    ):
        """
        Update the cache with the given key and value tensors.
        """
        self.k_cache[layer_idx, cache_batch_indices, :, cache_seq_indices, :] = k_cache[
            cache_batch_indices, :, self.seq_indices, :
        ]
        self.v_cache[layer_idx, cache_batch_indices, :, cache_seq_indices, :] = v_cache[
            cache_batch_indices, :, self.seq_indices, :
        ]

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def clear(self, batch_indices: Optional[torch.Tensor] = None):
        if batch_indices is not None:
            batch_indices = batch_indices.to(self.device)
            self.k_cache[:, batch_indices, ...].zero_()
            self.v_cache[:, batch_indices, ...].zero_()
            self._offset[batch_indices].zero_()
        else:
            self.k_cache.zero_()
            self.v_cache.zero_()

    @contextmanager
    def prefill_context(self, n_beams: int, batch_idx: int):
        prev_seq_indices = self.seq_indices
        prev_k_cache = self.k_cache
        prev_v_cache = self.v_cache

        self.seq_indices = torch.arange(n_beams, device=self.device)
        self.k_cache = prev_k_cache.select(1, batch_idx).clone().unsqueeze(1)
        self.v_cache = prev_v_cache.select(1, batch_idx).clone().unsqueeze(1)

        yield

        prev_k_cache[:, batch_idx : batch_idx + 1] = self.k_cache
        prev_v_cache[:, batch_idx : batch_idx + 1] = self.v_cache

        self.seq_indices = prev_seq_indices
        self.k_cache = prev_k_cache
        self.v_cache = prev_v_cache


class HybridCache:
    """Cache for hybrid models (Qwen3.5): KV for full-attention layers plus
    tree-aware recurrent state for Gated DeltaNet linear-attention layers.

    Positions ``[0, commit_len)`` are the committed prefix, folded into
    ``state``. Positions ``>= commit_len`` hold per-token delta-rule terms
    (``k_store``, ``v_store``, ``g_store``, ``G_store``) for the uncommitted
    tree. See ``model/qwen3_5.py`` for the math.

    Committing happens in ``gather`` when ``dest_indices`` starts at 0, which
    is how the engines compact the cache to the verified path. ``prefill``
    commits the prompt directly.
    """

    def __init__(
        self,
        config,
        max_n_beams: int,
        max_len: int,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        store_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model_config = config
        self.max_len = max_len
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size

        layer_types = config.layer_types
        self.num_attn_layers = sum(t == "full_attention" for t in layer_types)
        self.num_linear_layers = sum(t == "linear_attention" for t in layer_types)
        self.head_dim = config.head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        num_k_heads = config.linear_num_key_heads
        num_v_heads = config.linear_num_value_heads
        head_k_dim = config.linear_key_head_dim
        head_v_dim = config.linear_value_head_dim
        conv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim

        # Kept for GraphEngine, which swaps it while capturing graphs.
        self.seq_indices = torch.arange(
            max_n_beams, device=self.device, dtype=torch.long
        ).repeat(batch_size)
        self.prefilling = False

        kv_shape = (
            self.num_attn_layers,
            batch_size,
            config.num_key_value_heads,
            max_len,
            self.head_dim,
        )
        self.k_cache = torch.zeros(kv_shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(kv_shape, device=device, dtype=dtype)

        n_lin = self.num_linear_layers
        self.commit_len = torch.zeros(batch_size, device=device, dtype=torch.long)
        self.state = torch.zeros(
            (n_lin, batch_size, num_v_heads, head_k_dim, head_v_dim),
            device=device,
            dtype=torch.float32,
        )
        self.x_store = torch.zeros(
            (n_lin, batch_size, max_len, conv_dim), device=device, dtype=dtype
        )
        self.k_store = torch.zeros(
            (n_lin, batch_size, num_k_heads, max_len, head_k_dim),
            device=device,
            dtype=store_dtype,
        )
        self.v_store = torch.zeros(
            (n_lin, batch_size, num_v_heads, max_len, head_v_dim),
            device=device,
            dtype=store_dtype,
        )
        self.g_store = torch.zeros(
            (n_lin, batch_size, num_v_heads, max_len),
            device=device,
            dtype=torch.float32,
        )
        self.G_store = torch.zeros_like(self.g_store)

    # (tensor, dim of the sequence axis) for everything indexed by position.
    def _positional(self):
        return (
            (self.k_cache, 3),
            (self.v_cache, 3),
            (self.x_store, 2),
            (self.k_store, 3),
            (self.v_store, 3),
            (self.g_store, 3),
            (self.G_store, 3),
        )

    def _all(self):
        return [t for t, _ in self._positional()] + [self.state, self.commit_len]

    def update(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        layer_idx: int,
        cache_batch_indices: torch.Tensor,
        cache_seq_indices: torch.Tensor,
    ):
        """Write K/V ``[B, H, n, D]`` of a full-attention layer (``layer_idx``
        indexes the full-attention layers only)."""
        batch_size, num_heads, n, head_dim = k_cache.shape
        self.k_cache[layer_idx, cache_batch_indices, :, cache_seq_indices, :] = (
            k_cache.transpose(1, 2).reshape(batch_size * n, num_heads, head_dim)
        )
        self.v_cache[layer_idx, cache_batch_indices, :, cache_seq_indices, :] = (
            v_cache.transpose(1, 2).reshape(batch_size * n, num_heads, head_dim)
        )
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def gather(
        self, batch_idx: int, src_indices: torch.Tensor, dest_indices: torch.Tensor
    ):
        """Move positions ``src`` to ``dest`` and drop everything after
        ``dest.max()``. If ``dest`` starts at 0, the moved positions are the
        verified path and are committed into the recurrent state."""
        src_indices = src_indices.to(self.device)
        dest_indices = dest_indices.to(self.device)

        if src_indices.dtype == torch.bool:
            src_indices = torch.where(src_indices)[0]

        end = int(dest_indices.max().item()) + 1
        for tensor, dim in self._positional():
            t = tensor.select(1, batch_idx)
            seq_dim = dim - 1
            moved = t.index_select(seq_dim, src_indices)
            t.index_copy_(seq_dim, dest_indices, moved)
            t.narrow(seq_dim, end, self.max_len - end).zero_()

        if int(dest_indices.min().item()) == 0:
            self.commit(batch_idx, end)

    def commit(self, batch_idx: int, new_len: int):
        """Fold the path ``[commit_len, new_len)`` into the recurrent state."""
        start = int(self.commit_len[batch_idx].item())
        if new_len == start:
            return
        if new_len < start:
            raise ValueError(
                f"Cannot roll back committed state from {start} to {new_len}"
            )

        k = self.k_store[:, batch_idx, :, start:new_len]  # [N, Hk, m, Dk]
        v = self.v_store[:, batch_idx, :, start:new_len]  # [N, Hv, m, Dv]
        G = self.G_store[:, batch_idx, :, start:new_len]  # [N, Hv, m]
        ratio = v.shape[1] // k.shape[1]
        k = k.float().repeat_interleave(ratio, dim=1)
        G_last = G[..., -1:]

        state = self.state[:, batch_idx]
        state.mul_(G_last.exp()[..., None])
        state.add_(
            (k * (G_last - G).exp()[..., None]).transpose(-1, -2) @ v.float()
        )
        self.commit_len[batch_idx] = new_len

        for store in (self.k_store, self.v_store, self.g_store, self.G_store):
            store[:, batch_idx].zero_()

    def clear(self, batch_indices: Optional[torch.Tensor] = None):
        if batch_indices is not None:
            batch_indices = batch_indices.to(self.device)
            for tensor, _ in self._positional():
                tensor[:, batch_indices] = 0
            self.state[:, batch_indices] = 0
            self.commit_len[batch_indices] = 0
        else:
            for tensor in self._all():
                tensor.zero_()

    @contextmanager
    def prefill_context(self, n_beams: int, batch_idx: int):
        """Run a causal prefill of ``[commit_len, commit_len + n_beams)`` on
        ``batch_idx``. Views alias the full cache, so writes land in place."""
        names = (
            "k_cache", "v_cache", "x_store", "k_store", "v_store", "g_store",
            "G_store", "state",
        )
        prev = {name: getattr(self, name) for name in names}
        prev_seq_indices = self.seq_indices
        prev_commit_len = self.commit_len

        for name in names:
            setattr(self, name, prev[name].narrow(1, batch_idx, 1))
        self.commit_len = prev_commit_len.narrow(0, batch_idx, 1)
        self.seq_indices = torch.arange(n_beams, device=self.device)
        self.prefilling = True

        try:
            yield
            self.commit_len += n_beams
        finally:
            for name in names:
                setattr(self, name, prev[name])
            self.commit_len = prev_commit_len
            self.seq_indices = prev_seq_indices
            self.prefilling = False

    def snapshot(self, batch_idx: int) -> dict:
        """Committed state of one batch slot, for per-client offloading."""
        length = int(self.commit_len[batch_idx].item())
        tail = max(length - (self.conv_kernel_size - 1), 0)
        return {
            "commit_len": length,
            "k_cache": self.k_cache[:, batch_idx, :, :length].clone(),
            "v_cache": self.v_cache[:, batch_idx, :, :length].clone(),
            "x_tail": self.x_store[:, batch_idx, tail:length].clone(),
            "state": self.state[:, batch_idx].clone(),
        }

    def restore(self, batch_idx: int, snap: dict):
        """Load a ``snapshot`` into an empty batch slot."""
        length = snap["commit_len"]
        tail = length - snap["x_tail"].size(1)
        self.k_cache[:, batch_idx, :, :length].copy_(snap["k_cache"])
        self.v_cache[:, batch_idx, :, :length].copy_(snap["v_cache"])
        self.x_store[:, batch_idx, tail:length].copy_(snap["x_tail"])
        self.state[:, batch_idx].copy_(snap["state"])
        self.commit_len[batch_idx] = length
