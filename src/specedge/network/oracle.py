"""Server-free validator that answers from a recorded verifier trace.

`OracleValidator` is a drop-in for `GrpcClientController`: same `request()`
signature, same `(selection, prefill_cnt)` return. Instead of running the
target model it looks the answer up in a `trace.jsonl` written by
`src/gen_trace.py` (see `server-trace/`), so draft-side experiments can be
rerun without a GPU verifier.

Selection for a node at absolute position p is `reference[p + 1]`, where
`reference = prompt_tokens + output_tokens`. The oracle never sees the tree:
a node off the reference path gets an answer too, but it cannot be accepted
because its parent was already rejected. Past the end of the reference it
answers EOS. This is exact for a greedy (temperature 0) target, which is how
the traces were recorded.

Latency is simulated: every request sleeps `server_ms + rtt_ms` (or
`prefill_ms + rtt_ms` on prefill) on the event loop, so the overlap strategy
still gets a real window to draft in. A `server.jsonl` with the simulated
server span is written next to the client log so `src/metric/specedge.py`
works unchanged.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

import log


def load_trace(path: str | Path) -> dict[int, list[int]]:
    """Map req_idx -> full reference sequence (prompt tokens + output tokens)."""
    refs = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                refs[rec["req_idx"]] = rec["prompt_tokens"] + rec["output_tokens"]
    return refs


class OracleValidator:
    def __init__(
        self,
        trace_path: str | Path,
        device: torch.device,
        eos_token_id: int,
        server_ms: float,
        prefill_ms: float,
        rtt_ms: float,
        server_log_path: Optional[str | Path] = None,
        references: Optional[dict[int, list[int]]] = None,
    ) -> None:
        self._logger = log.get_logger()
        self._device = device
        self._eos = eos_token_id
        self._server_ms = server_ms
        self._prefill_ms = prefill_ms
        self._rtt_ms = rtt_ms
        self._refs = references if references is not None else load_trace(trace_path)

        # Reference tensors on device, built lazily per request.
        self._ref_tensors: dict[int, torch.Tensor] = {}
        self._overruns: set[int] = set()

        self._server_log = None
        if server_log_path is not None:
            Path(server_log_path).parent.mkdir(parents=True, exist_ok=True)
            self._server_log = open(server_log_path, "a")

    def has(self, req_idx: int) -> bool:
        return req_idx in self._refs

    def check_prompt(self, req_idx: int, prompt_tokens: torch.Tensor) -> None:
        """Fail fast if the client tokenized a different prompt than the trace."""
        if req_idx not in self._refs:
            raise KeyError(f"req_idx {req_idx} is not in the replay trace")
        ref = self._refs[req_idx]
        got = prompt_tokens.flatten().tolist()
        if ref[: len(got)] != got:
            raise ValueError(
                f"req_idx {req_idx}: client prompt tokens differ from the trace "
                "(check dataset, chat template / reasoning flag and max_len)"
            )

    def _reference(self, req_idx: int, min_len: int) -> torch.Tensor:
        ref = self._ref_tensors.get(req_idx)
        if ref is None or ref.numel() < min_len:
            tokens = self._refs[req_idx]
            size = max(min_len, len(tokens) + 1)
            ref = torch.full((size,), self._eos, dtype=torch.long)
            ref[: len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            ref = ref.to(self._device)
            self._ref_tensors[req_idx] = ref
        return ref

    def select(self, req_idx: int, position_ids: torch.Tensor) -> torch.Tensor:
        """Target choice for every input node: reference[position + 1]."""
        next_pos = position_ids.flatten().to(self._device, torch.long) + 1
        max_pos = int(next_pos.max().item())
        n_ref = len(self._refs[req_idx])
        if max_pos >= n_ref and req_idx not in self._overruns:
            self._overruns.add(req_idx)
            self._logger.warning(
                "req_idx %d: draft reached past the recorded completion "
                "(%d tokens); answering EOS there",
                req_idx,
                n_ref,
            )
        return self._reference(req_idx, max_pos + 1)[next_pos]

    async def request(
        self,
        client_idx: int,
        req_idx: int,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_seq_indices: torch.Tensor,
        attention_mask: torch.Tensor,
        parent_indices: torch.Tensor,
        prefill: bool = False,
        prefix: Optional[str] = None,
    ):
        if prefill and prefix is None:
            raise ValueError("Prefix must be provided for prefill requests.")

        selection = self.select(req_idx, position_ids)

        server_ms = self._prefill_ms if prefill else self._server_ms
        await asyncio.sleep((server_ms + self._rtt_ms) / 1000)

        if self._server_log is not None:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(
                    timespec="milliseconds"
                ),
                "target": {
                    "forward_t": server_ms,
                    "server_end_to_end_t": server_ms,
                    "prefill": int(prefill),
                },
                "replay": True,
            }
            self._server_log.write(json.dumps(record) + "\n")
            self._server_log.flush()

        # The real server reports how many requests in its batch were prefilled.
        return selection, int(prefill)
