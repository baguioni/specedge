"""SaguaroStrategy: a fan-out of shallow bets across predicted verification
outcomes, cached and spliced on a cache hit.

Cost parity with proactive drafting: one draft forward for outcome prediction
plus ``branch_len`` forwards to grow the (multi-root) scratch forest -- the
forest is grown with the existing proactive beam/budget machinery, so each
forward is merely wider, not repeated ``B`` times.
"""

from __future__ import annotations

import torch

import log
from specedge.client.overlap import OverlapResult, OverlapStrategy
from specedge.client.proactive import SpecExecProactiveDraft
from specedge.client.reorder import (
    append_bonus_token,
    reopen_leaves,
    reorder_to_verified_path,
    splice_scratch_branch,
)
from specedge.client.saguaro.cache import (
    CachedSpeculation,
    Outcome,
    SpeculationCache,
)
from specedge.client.saguaro.outcomes import (
    OutcomePrediction,
    predict_outcome_details,
)


def build_speculation_cache(
    tree, forest, exit_nodes, bonus_tokens, device
) -> SpeculationCache:
    _, _, root_indices, root_of = forest
    post_candidate = int(tree.POST_CANDIDATE)

    by_root: dict[int, list[int]] = {int(r): [] for r in root_indices}
    for idx, root in root_of.items():
        if root in by_root:
            by_root[root].append(idx)

    cache = SpeculationCache()
    for exit_idx, bonus, root_idx in zip(
        exit_nodes, bonus_tokens, root_indices, strict=True
    ):
        nodes = sorted(by_root[int(root_idx)])
        # Without a CANDIDATE frontier the next _grow_tree has nothing to
        # extend after a splice, so reconcile re-opens the branch's leaves.
        has_frontier = any(int(tree.status[n].item()) == post_candidate for n in nodes)
        cache.put(
            Outcome(int(exit_idx), int(bonus)),
            CachedSpeculation(
                root_scratch_idx=int(root_idx),
                node_indices=torch.tensor(nodes, dtype=torch.long, device=device),
                n_tokens=len(nodes),
                has_frontier=has_frontier,
            ),
        )
    return cache


class SaguaroStrategy(OverlapStrategy):
    name = "saguaro"

    def __init__(self, tree, engine, device, dtype, cfg) -> None:
        super().__init__(tree, engine, device, dtype)
        self._logger = log.get_logger()

        self._pd = SpecExecProactiveDraft(tree=tree, engine=engine, max_len=cfg.max_len)
        self._budget = int(cfg.saguaro_budget)
        self._branch_len = int(cfg.saguaro_branch_len)
        self._fan_out = cfg.saguaro_fan_out
        self._max_n_beams = int(cfg.proactive_max_n_beams)
        # Fixed a_p for the geometric fan-out (Theorem 12), matching the paper
        # and the ssd reference: profiled/known ahead of time, not adapted online.
        self._acceptance_rate = float(cfg.saguaro_acceptance_rate)

        # Give the scratch forest its own budget so branches stay deep enough
        # to leave a CANDIDATE frontier after a splice (see build_speculation_cache).
        self._forest_budget = max(
            int(cfg.proactive_max_budget), self._budget * (self._branch_len + 1)
        )

        if str(cfg.saguaro_exit_mode) == "auto":
            self._exit_mode = "trunk" if int(cfg.max_branch_width) == 1 else "leaf"
        else:
            self._exit_mode = str(cfg.saguaro_exit_mode)
            if self._exit_mode not in ("leaf", "trunk"):
                raise ValueError(
                    f"saguaro_exit_mode must be 'auto', 'leaf' or 'trunk', "
                    f"got {self._exit_mode!r}"
                )

        self._cache: SpeculationCache | None = None
        # Last speculate() round, kept for the tree trace.
        self._prediction = OutcomePrediction()
        self._forest = None

    @property
    def depth_gain(self) -> int:
        return self._branch_len

    @property
    def prediction(self) -> OutcomePrediction:
        """Outcomes predicted by the last ``speculate()``."""
        return self._prediction

    @property
    def forest(self):  # -> tuple | None
        """``draft_forest``'s ``(forest_start, forest_end, root_indices,
        root_of)`` from the last ``speculate()``, or ``None``."""
        return self._forest

    @property
    def cache(self) -> SpeculationCache | None:
        return self._cache

    def speculate(self) -> None:
        self._cache = None
        self._forest = None

        self._prediction = predict_outcome_details(
            self._tree,
            self._engine,
            budget=self._budget,
            max_n_beams=self._max_n_beams,
            acceptance_rate=self._acceptance_rate,
            fan_out=self._fan_out,
            exit_mode=self._exit_mode,
        )
        exit_nodes = self._prediction.exit_nodes
        bonus_tokens = self._prediction.bonus_tokens
        if not exit_nodes:
            return

        forest = self._pd.draft_forest(
            exit_nodes,
            bonus_tokens,
            self._branch_len,
            forest_budget=self._forest_budget,
        )
        if forest is None:
            return

        self._forest = forest
        self._cache = build_speculation_cache(
            self._tree, forest, exit_nodes, bonus_tokens, self._device
        )

    def reconcile(
        self,
        *,
        seq_mask: torch.Tensor,
        last_accepted_token_idx: int,
        extra_token_id: torch.Tensor,
    ) -> OverlapResult:
        n_hyp = len(self._cache) if self._cache is not None else 0
        bonus = int(extra_token_id.flatten()[0].item())
        hit = (
            self._cache.get(Outcome(int(last_accepted_token_idx), bonus))
            if self._cache is not None
            else None
        )

        # No hit -> plain reorder + bonus token (same tokens, no reuse).
        if hit is None:
            reorder_to_verified_path(self._tree, self._engine, self._device, seq_mask)
            append_bonus_token(self._tree, extra_token_id, self._device)
            return OverlapResult(
                spliced=False, cache_hit=False, n_reused=0, n_hypotheses=n_hyp
            )

        splice_scratch_branch(
            self._tree,
            self._engine,
            self._device,
            self._dtype,
            seq_mask,
            hit.node_indices,
        )
        # A branch cut short by the forest budget has every node expanded and
        # no CANDIDATE left; re-open its leaves so _grow_tree can extend it
        # (otherwise the next verification request would be undersized).
        if not hit.has_frontier:
            reopen_leaves(self._tree)
        return OverlapResult(
            spliced=True,
            cache_hit=True,
            n_reused=hit.n_tokens,
            n_hypotheses=n_hyp,
        )
