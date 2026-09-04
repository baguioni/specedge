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
    reorder_to_verified_path,
    splice_scratch_branch,
)
from specedge.client.saguaro.cache import (
    CachedSpeculation,
    Outcome,
    SpeculationCache,
)
from specedge.client.saguaro.outcomes import predict_outcomes

_ACCEPT_EMA = 0.1


def build_speculation_cache(
    forest, exit_nodes, bonus_tokens, device
) -> SpeculationCache:
    _, _, root_indices, root_of = forest

    by_root: dict[int, list[int]] = {int(r): [] for r in root_indices}
    for idx, root in root_of.items():
        if root in by_root:
            by_root[root].append(idx)

    cache = SpeculationCache()
    for exit_idx, bonus, root_idx in zip(
        exit_nodes, bonus_tokens, root_indices, strict=True
    ):
        nodes = sorted(by_root[int(root_idx)])
        cache.put(
            Outcome(int(exit_idx), int(bonus)),
            CachedSpeculation(
                root_scratch_idx=int(root_idx),
                node_indices=torch.tensor(nodes, dtype=torch.long, device=device),
                n_tokens=len(nodes),
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
        self._max_depth = max(1, int(cfg.max_beam_len))
        self._accept_rate = float(cfg.saguaro_init_accept_rate)

        if str(cfg.saguaro_linear) == "auto":
            self._linear = int(cfg.max_branch_width) == 1
        else:
            self._linear = str(cfg.saguaro_linear) == "True"

        self._cache: SpeculationCache | None = None

    @property
    def depth_gain(self) -> int:
        return self._branch_len

    def speculate(self) -> None:
        self._cache = None

        exit_nodes, bonus_tokens = predict_outcomes(
            self._tree,
            self._engine,
            budget=self._budget,
            max_n_beams=self._max_n_beams,
            acceptance_rate=self._accept_rate,
            fan_out=self._fan_out,
            linear=self._linear,
        )
        if not exit_nodes:
            return

        forest = self._pd.draft_forest(exit_nodes, bonus_tokens, self._branch_len)
        if forest is None:
            return

        self._cache = build_speculation_cache(
            forest, exit_nodes, bonus_tokens, self._device
        )

    def reconcile(
        self,
        *,
        seq_mask: torch.Tensor,
        last_accepted_token_idx: int,
        extra_token_id: torch.Tensor,
    ) -> OverlapResult:
        self._observe_acceptance(seq_mask)

        n_hyp = len(self._cache) if self._cache is not None else 0
        bonus = int(extra_token_id.flatten()[0].item())
        hit = (
            self._cache.get(Outcome(int(last_accepted_token_idx), bonus))
            if self._cache is not None
            else None
        )

        if hit is None or hit.n_tokens == 0:
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
        return OverlapResult(
            spliced=True,
            cache_hit=True,
            n_reused=hit.n_tokens,
            n_hypotheses=n_hyp,
        )

    def _observe_acceptance(self, seq_mask: torch.Tensor) -> None:
        n_accepted = int(seq_mask[int(self._tree.prefix_len) :].sum().item())
        observed = min(max(n_accepted / self._max_depth, 1e-3), 1.0 - 1e-3)
        self._accept_rate = (
            1.0 - _ACCEPT_EMA
        ) * self._accept_rate + _ACCEPT_EMA * observed
