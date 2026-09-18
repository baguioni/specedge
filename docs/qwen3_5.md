# Qwen3.5 dense support

This document describes the changes that add Qwen3.5 dense models
(`Qwen/Qwen3.5-0.8B`, `-2B`, `-4B`, `-9B`, `-27B`) as draft or target models in
SpecEdge.

Summary:

| File | Change |
|---|---|
| `src/model/qwen3_5.py` | New. Qwen3.5 text decoder, config and weight loader. |
| `src/model/cache.py` | New `HybridCache` class (KV cache + tree-aware linear-attention state). |
| `src/specedge/engine/graph.py` | Engines take the cache class from the model (`model.cache_cls`). |
| `src/util.py` | `load_graph_model` sends names containing `qwen3.5` / `qwen3_5` to the new model. |
| `src/strategy/server_verify/specexec/grpc.py` | Per-client state and prefill cache for hybrid models. |

`src/model/qwen3.py` is unchanged. Qwen3 models still use it.

---

## 1. Why Qwen3.5 needs more than a new model file

Qwen3.5 is a hybrid model. With `full_attention_interval: 4`, three of every
four decoder layers are **Gated DeltaNet** (linear attention) layers. Only every
fourth layer is a regular full-attention layer.

A Gated DeltaNet layer has no KV cache. It keeps a recurrent state
`S` (`[heads, k_dim, v_dim]`) and updates it token by token:

```
S_t = exp(g_t) * S_{t-1} + k_t * (beta_t * (v_t - (exp(g_t) * S_{t-1})^T k_t))^T
o_t = S_t^T q_t
```

It also has a short causal convolution (kernel 4) over the input projection.

SpecEdge verifies a *tree* of draft tokens in one forward pass and uses an
attention mask to say which tokens are ancestors of which. A single recurrent
state cannot represent a tree: every branch needs its own state, and the state
has to be rolled back to the accepted path after verification. The existing
`KVCache` handles attention layers only, so the linear-attention layers need a
new cache design.

## 2. Model: `src/model/qwen3_5.py`

The file follows the structure of `src/model/qwen3.py` and keeps the same
forward signature, so `GraphEngine`, `BatchGraphEngine` and the strategies
call it without changes:

```python
logits, cache = model(input_ids, position_ids, cache_batch_indices,
                      cache_seq_indices, attention_mask, past_key_values)
```

Reference implementation:
`transformers/models/qwen3_5/modeling_qwen3_5.py` (transformers `main`). The
installed transformers (4.57) has no `qwen3_5` module, so nothing is imported
from it.

Components and their differences from Qwen3:

| Component | Qwen3.5 detail |
|---|---|
| `Qwen3_5TextConfig` | Parses the multimodal `config.json` (`text_config` block). Builds `layer_types` from `full_attention_interval` if missing. |
| `Qwen3_5RMSNorm` | Zero-centered: `x_norm * (1 + weight)`. |
| `Qwen3_5RMSNormGated` | RMSNorm followed by `* silu(z)`, used at the output of the linear-attention layer. |
| `Qwen3_5RotaryEmbedding` | Partial RoPE (`partial_rotary_factor: 0.25`, so 64 of 256 head dims). The checkpoint uses interleaved M-RoPE, but for text-only input the three position ids are identical, so it reduces to plain RoPE. |
| `Qwen3_5Attention` | `q_proj` outputs a query and a gate per head. The output is `attn_output * sigmoid(gate)`. Uses the KV part of `HybridCache`. |
| `Qwen3_5GatedDeltaNet` | Linear-attention layer: `in_proj_qkv` → causal conv → L2-normalized q/k → gated delta rule → gated RMSNorm → `out_proj`. |
| `Qwen3_5ForCausalLM` | Has `cache_cls = HybridCache`, and `device` / `dtype` properties for the engines. |

### Weight loading

`Qwen3_5ForCausalLM.from_pretrained(name, torch_dtype, device_map)`:

1. Downloads `*.json` and `*.safetensors` with `snapshot_download` (a local
   directory also works).
2. Builds the model on the `meta` device.
3. Reads the safetensors files and renames `model.language_model.*` to
   `model.*`. It skips `model.visual.*` (vision tower) and `mtp.*`
   (multi-token-prediction head).
4. Loads with `assign=True`, ties `lm_head` to `embed_tokens` when
   `tie_word_embeddings` is true (0.8B, 2B, 4B, 9B), and raises on missing or
   unexpected keys.
5. Rebuilds the RoPE buffer on the target device (non-persistent buffers are
   not in the checkpoint).

All parameters are cast to `torch_dtype`, the same as HF. The checkpoint stores
`A_log` and the gated-norm weight in fp32, but HF also casts them.

The tokenizer needs no change. The checkpoints use `Qwen2Tokenizer`, which
transformers 4.57 loads through `AutoTokenizer`.

## 3. Exact tree decoding for the linear-attention layers

### Math

Let `c = commit_len` be the length of the committed prefix, and `S` the
recurrent state after it. For a token `i`, its uncommitted ancestors are the
positions `j >= c` that are set in its attention-mask row (the row includes
`i`). Define the cumulative log decay along the path:

```
G_i = sum over uncommitted ancestors j of i (including i) of g_j
```

The chunked form of the gated delta rule then gives, for every tree node:

```
v_new_i = beta_i * ( v_i
                     - exp(G_i) * S^T k_i
                     - sum over strict ancestors j:  exp(G_i - G_j) * (k_i . k_j) * v_new_j )

o_i     = exp(G_i) * S^T q_i
          + sum over ancestors j (including i):  exp(G_i - G_j) * (q_i . k_j) * v_new_j
```

This is the chunked gated delta rule with the causal mask replaced by the tree
ancestor mask. It is exact because the ancestors of a node form a chain, and
`v_new_j` of an ancestor depends only on that ancestor's own chain.

### Algorithm per forward (`Qwen3_5GatedDeltaNet._tree`)

1. Compute `in_proj_qkv`, `z`, `beta`, `g` for the new tokens and write the
   pre-conv input `x` to the store.
2. **Causal conv along the path.** `TreeMeta` finds the 4 nearest ancestors of
   each new token from its mask row (rank by cumulative sum). The conv gathers
   their `x` rows. This works for ancestors below `commit_len` too.
3. Write `g`, then compute `G_i` as a masked sum over the stored `g`.
4. Write `k` and `G` for the new tokens and set their `v_new` to 0.
5. One attention-like pass over the stores with `q` (for the output) and `k`
   (for the delta rule) as readers. This gives the contribution of `S` and of
   the already-stored ancestors.
6. New tokens that are ancestors of each other (for example, a whole tree
   verified in one forward on the server) form a small unit lower-triangular
   system `(I + L) v_new = rhs`, solved with `torch.linalg.solve_triangular`.
   **This requires that a token's ancestors come before it in the input
   order.** The existing trees already add parents before children.
7. Store `v_new` and return `o`.

All shapes are static and there is no host sync, so the forward works in CUDA
graphs.

### Prefill (`Qwen3_5GatedDeltaNet._prefill`)

Prefill processes a causal chain `[commit_len, commit_len + n)`. It uses the
standard chunked gated delta rule (ported from HF), with the conv left context
taken from the stored `x`. The final state goes directly into `S`, and
`commit_len` advances by `n` when `prefill_context` exits.

## 4. `HybridCache` (`src/model/cache.py`)

Constructor has the same arguments as `KVCache` (plus optional `store_dtype`),
so the engines create it the same way.

| Tensor | Shape | Purpose |
|---|---|---|
| `k_cache`, `v_cache` | `[n_attn_layers, B, kv_heads, max_len, head_dim]` | KV cache for full-attention layers only (1/4 of the layers). |
| `state` | `[n_linear_layers, B, Hv, Dk, Dv]` fp32 | Recurrent state `S` of the committed prefix. |
| `commit_len` | `[B]` long | Length of the committed prefix. |
| `x_store` | `[n_linear, B, max_len, conv_dim]` | Pre-conv input per position (conv context). |
| `k_store` | `[n_linear, B, Hk, max_len, Dk]` | Key per uncommitted position. |
| `v_store` | `[n_linear, B, Hv, max_len, Dv]` | `v_new` per uncommitted position. |
| `g_store`, `G_store` | `[n_linear, B, Hv, max_len]` fp32 | Log decay and cumulative log decay per position. |

Methods:

- `update(...)`: writes attention K/V, same contract as `KVCache.update`.
- `gather(batch_idx, src, dest)`: moves every per-position tensor from `src`
  to `dest` and zeros everything after `dest.max()`, the same as
  `KVCache.gather`. **If `dest` starts at 0, the moved positions are the
  verified path, and `commit()` folds them into `S`.** All callers compact to
  the verified path this way (`reorder_to_verified_path`, the edge-verify
  `_reorder_by_sequence`, the server's `_reorder_kv_cache`). Budget trimming
  uses `dest` starting at `prefix_len`, so it only moves data.
- `commit(batch_idx, new_len)`:
  `S = exp(G_last) * S + sum_j (k_j * exp(G_last - G_j))^T v_new_j` over the
  path, then `commit_len = new_len`. Rolling back below `commit_len` raises an
  error.
- `prefill_context(n, batch_idx)`: narrows all tensors to one batch slot
  (views, so writes go straight to the cache) and sets `prefilling = True`.
- `clear(batch_indices=None)`: zeros everything, including `state` and
  `commit_len`.
- `snapshot(batch_idx)` / `restore(batch_idx, snap)`: save and load the
  committed state of one slot (KV up to `commit_len`, the last 3 conv inputs,
  `S`, `commit_len`).

## 5. Integration changes

### `src/specedge/engine/graph.py`

`GraphEngine` and `BatchGraphEngine` now create the cache with
`getattr(model, "cache_cls", KVCache)(...)`. Llama and Qwen3 have no
`cache_cls`, so they still get `KVCache`.

### `src/util.py`

`load_graph_model` checks for `qwen3.5` / `qwen3_5` in the model name before
the `qwen3` check and loads `Qwen3_5ForCausalLM`.

### `src/strategy/server_verify/specexec/grpc.py`

The server clears the engine cache before every batch and restores each
client from per-client `k_cache` / `v_cache` buffers. For a hybrid model that
would drop the recurrent state. When the engine cache is a `HybridCache`
(`self._hybrid`):

- Per-client state is a `snapshot()` dict in `self._client_states`, taken
  after `_reorder_kv_cache` commits the accepted path, and restored with
  `restore()` before the next forward.
- The per-client `k_cache` / `v_cache` buffers are allocated with 0 layers.
- With `cache_prefill: true`, the prompt prefill is saved as a snapshot in
  `<req_idx>_hybrid_state.pt` (separate from the existing
  `_key_cache.pt` / `_value_cache.pt` files).
- `_begin_experiment` also clears `self._client_states`.

The non-hybrid code path is unchanged.

## 6. Verification

Run on an A100 (vast.ai). The reference is HF transformers `main`
(`5.18.0.dev0`) installed in a separate directory (`/workspace/hfref`, used
with `PYTHONPATH`), so the project venv is not changed.

### Logits vs HF (Qwen3.5-0.8B, fp32)

Three sequences share a 14-token prompt and branch into a 65-node tree.

| Test | Max abs logit diff | Argmax agreement |
|---|---|---|
| Prefill of the full sequence | 2.6e-5 | 100% |
| Whole tree in one forward (server style) | 2.9e-5 | 100% |
| Level by level (draft style) | 4.6e-5 | 100% |
| Commit via `gather`, then decode | 3.1e-5 | 100% |
| Trim-style `gather` (no commit), recompute leaf | 2.1e-5 | 100% |
| `GraphEngine` with CUDA graphs, level by level | 4.6e-5 | 100% |
| `BatchGraphEngine` with CUDA graphs, whole tree | 2.9e-5 | 100% |
| `snapshot` / `clear` / `restore` every step | 2.9e-5 | 100% |

### bf16

On 0.8B, the bf16 error of this implementation vs HF fp32 (max 0.14 to 0.22)
is the same size as HF bf16 vs HF fp32 (0.16 to 0.20).

On 4B, teacher-forced over a 69-token generation:

| | Mean abs diff vs HF fp32 | Max abs diff |
|---|---|---|
| This implementation, bf16 | 0.0189 | 0.186 |
| HF, bf16 | 0.0196 | 0.314 |

Greedy decoding in fp32 gives the same tokens as HF. In bf16 both 4B and 9B
split from HF at generated token 3, where the fp32 top-1 / top-2 logit gap is
only 0.028 (a near-tie).

### End to end

`script/server_only.sh` with a Qwen3.5-0.8B draft and a Qwen3.5-9B target
(bf16, temperature 0) produced coherent output, about 2.6 accepted tokens per
step. The speculative output is identical, token for token, to plain greedy
decoding with the same model, so the tree logic is lossless.

## 7. Performance

Forward latency, `GraphEngine` with CUDA graphs, bf16, `max_len` 2048,
300-token prefix (the GPU was shared with another process, so absolute numbers
are rough):

| Model | 1 beam | 8 beams | 32 beams |
|---|---|---|---|
| Qwen3.5-0.8B | 9.3 ms | 13.3 ms | 21.8 ms |
| Qwen3-0.6B | 9.4 ms | 10.2 ms | 10.6 ms |

The extra cost at many beams comes from the linear-attention tree terms. They
scan all `max_len` positions of the stores, masked to `>= commit_len`. The
planned fix is to read only a fixed window `[commit_len, commit_len + W)`,
where `W` must cover the largest uncommitted tree.

Memory: the per-position stores add roughly 1 GB for the 0.8B model and about
5 GB for 27B per batch slot at `max_len` 2048. `store_dtype=torch.bfloat16`
halves the `k_store` / `v_store` part at some precision cost.

## 8. Limitations and notes

- A Qwen3.5 target needs a Qwen3.5 draft. The vocabularies differ (Qwen3:
  151,936 tokens, Qwen3.5: 248,320). `config/specedge.example.yaml` currently
  pairs `Qwen/Qwen3-1.7B` with `Qwen/Qwen3.5-27B`, which cannot work.
- The committed state cannot be rolled back. A `gather` with `dest` starting
  at 0 must keep the whole committed prefix (`src[:commit_len]` equal to
  `arange(commit_len)`).
- New tokens in one forward must be in topological order (ancestors first).
- Not tested with hybrid models: the proactive and Saguaro splice paths
  (`splice_scratch_branch`, `reopen_leaves`) and a full `batch_server` + gRPC
  client run.
- `layer_split.py` and the layer-split scripts assume Qwen3 layers and are
  not adapted.
- Existing bug, unrelated to this change: `script/server_only.sh` does not set
  `SPECEDGE_CACHE_PREFILL`. Run it with `SPECEDGE_CACHE_PREFILL=False`.

## 9. Usage

Set Qwen3.5 models in the config:

```yaml
base:
  dtype: bf16
server:
  target_model: Qwen/Qwen3.5-9B
client:
  draft_model: Qwen/Qwen3.5-0.8B
```

Run as usual, for example:

```bash
SPECEDGE_CACHE_PREFILL=False bash script/server_only.sh -f config/q35_server_only.yaml
```
