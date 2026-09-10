"""Per-step tree trace for the Saguaro overlap strategy.

Every decoding step appends one ``"step"`` record to
``<result_path>/<exp_name>/<process_name>.tree_trace.jsonl`` holding three
views of the tree:

1. ``draft``   -- the committed chain (``chain_len`` tokens, ``chain_new`` of
   them new since the previous step) and the draft tree grown from its last
   token, as sent for verification. ``reused`` marks nodes carried over from
   the previous step's spliced Saguaro branch.
2. ``saguaro`` -- the fan-out planted while verification is in flight: the
   candidate exit nodes with their guess budget, every guessed bonus token,
   and the scratch forest drafted under the guesses.
3. ``verify``  -- the verified path: the accepted draft tokens, the server's
   bonus token, and whether a Saguaro branch was spliced into the next step.

The file starts with a ``"run"`` record (configuration) and every request
with a ``"request"`` record (prompt). Tokens are stored as ids;
``src/script/render_tree_trace.py`` decodes and draws them. Node ``idx``
values are tree slots, which the reorder after verification compacts, so they
are only comparable within one step record.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

from specedge.client.saguaro.cache import Outcome


def trace_path(result_path, exp_name: str, process_name: str) -> Path:
    return Path(result_path) / exp_name / f"{process_name}.tree_trace.jsonl"


@functools.cache
def _open(path: Path):
    # One file per process, shared by the per-request clients and kept open
    # until the process exits.
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "w")


def snapshot(tree, start: int, end: int) -> list[dict]:
    """Tree slots ``[start, end)`` as JSON-ready node dicts."""
    if end <= start:
        return []
    rows = zip(
        range(start, end),
        tree.tokens[start:end].tolist(),
        tree.parents[start:end].tolist(),
        tree.logprobs[start:end].tolist(),
        tree.positions[start:end].tolist(),
        strict=True,
    )
    return [
        {"idx": i, "token": t, "parent": p, "logprob": round(lp, 4), "position": pos}
        for i, t, p, lp, pos in rows
    ]


def token_paths(nodes: list[dict], root: int) -> dict[int, tuple[int, ...]]:
    """Token sequence from ``root`` (exclusive) down to every node.

    The budget trim reorders tree slots, so a node carried over from a splice
    is recognised by its token path rather than its index.
    """
    by_idx = {n["idx"]: n for n in nodes}
    paths: dict[int, tuple[int, ...]] = {root: ()}

    def path(idx: int) -> tuple[int, ...]:
        if idx not in paths:
            node = by_idx.get(idx)
            # A parent outside ``nodes`` should not happen; anchor it at root.
            paths[idx] = (
                () if node is None else path(node["parent"]) + (node["token"],)
            )
        return paths[idx]

    for node in nodes:
        path(node["idx"])
    return paths


class TreeTracer:
    """Records the three tree views of every decoding step (see module doc).

    ``SpecExecClient`` calls, per request, ``begin_request`` and then per step
    ``begin_step`` -> ``log_draft`` -> ``log_speculation`` -> ``end_step``.
    """

    def __init__(
        self, tree, strategy, path: Path, client_idx: int, run_info: dict
    ) -> None:
        self._tree = tree
        self._strategy = strategy
        self._client_idx = client_idx
        self._file = _open(Path(path))
        if self._file.tell() == 0:
            self._write({"type": "run", **run_info})

        self._chain_len = 0
        self._carried: set[tuple[int, ...]] = set()
        self._record: dict = {}

    def begin_request(self, req_idx: int) -> None:
        prompt = self._tree.tokens[: int(self._tree.prefix_len)].tolist()
        self._chain_len = len(prompt)
        self._write(
            {
                "type": "request",
                "client_idx": self._client_idx,
                "req_idx": req_idx,
                "prompt": prompt,
            }
        )

    def begin_step(self, req_idx: int, step_idx: int, prefill: bool) -> None:
        """Call before the draft tree is grown."""
        tree = self._tree
        prefix_len = int(tree.prefix_len)
        root = prefix_len - 1

        # Anything already hanging off the root was spliced in from the
        # previous step's Saguaro branch.
        carried = snapshot(tree, root + 1, int(tree.end))
        self._carried = set(token_paths(carried, root).values()) - {()}

        self._record = {
            "type": "step",
            "client_idx": self._client_idx,
            "req_idx": req_idx,
            "step_idx": step_idx,
            "prefill": prefill,
            "chain_len": prefix_len,
            "chain_new": tree.tokens[self._chain_len : prefix_len].tolist(),
        }
        self._chain_len = prefix_len

    def log_draft(self) -> None:
        """Stage 1. Call after the draft tree is grown (and trimmed)."""
        tree = self._tree
        root = int(tree.prefix_len) - 1
        nodes = snapshot(tree, root, int(tree.end))
        paths = token_paths(nodes, root)
        for node in nodes:
            node["reused"] = (
                node["idx"] != root and paths[node["idx"]] in self._carried
            )
        self._record["draft"] = {"root": root, "nodes": nodes}

    def log_speculation(self) -> None:
        """Stage 2. Call right after ``speculate()``: the reconcile step
        overwrites the scratch forest past ``tree.end``."""
        prediction = self._strategy.prediction
        cache = self._strategy.cache
        forest = self._strategy.forest
        roots = (
            forest[2] if forest is not None else [None] * len(prediction.exit_nodes)
        )

        bets = []
        for exit_idx, bonus, logprob, root in zip(
            prediction.exit_nodes,
            prediction.bonus_tokens,
            prediction.bonus_logprobs,
            roots,
            strict=True,
        ):
            spec = cache.get(Outcome(exit_idx, bonus)) if cache is not None else None
            bets.append(
                {
                    "exit": exit_idx,
                    "bonus": bonus,
                    "logprob": round(logprob, 4),
                    "root": root,
                    "nodes": spec.node_indices.tolist() if spec is not None else [],
                    "has_frontier": spec is not None and spec.has_frontier,
                }
            )

        self._record["saguaro"] = {
            "candidates": [
                {"node": node, "fan": fan, "excluded": excluded}
                for node, fan, excluded in zip(
                    prediction.candidates,
                    prediction.fan,
                    prediction.excluded,
                    strict=True,
                )
            ],
            "bets": bets,
            "forest": (
                snapshot(self._tree, forest[0], forest[1]) if forest is not None else []
            ),
        }

    def end_step(
        self, *, accepted_idx, accepted_ids, exit_idx: int, bonus: int, result
    ) -> None:
        """Stage 3. Call after reconcile; writes the step record.

        ``accepted_idx`` are pre-reorder slots, so they index into this
        record's ``draft`` nodes.
        """
        self._record["verify"] = {
            "accepted": accepted_idx.tolist(),
            "accepted_tokens": accepted_ids.tolist(),
            "exit": exit_idx,
            "bonus": bonus,
            "cache_hit": result.cache_hit,
            "spliced": result.spliced,
            "n_reused": result.n_reused,
        }
        self._write(self._record)
        self._record = {}

    def _write(self, record: dict) -> None:
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
