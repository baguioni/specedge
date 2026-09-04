"""Overlap-strategy interface: extra edge drafting that runs during the
server-verification RPC.

Two implementations are selectable via ``client.proactive.strategy``:

* ``proactive`` -- SpecEdge proactive draft generation (one deep bet from the
  single highest-cumulative-logprob leaf, spliced on "complete draft alignment").
* ``saguaro``   -- Saguaro speculation cache (a fan-out of shallow bets across
  predicted verification outcomes, spliced on cache hit).

``disabled`` returns ``None`` and the client falls back to a plain
verified-path reorder + bonus token.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class OverlapResult:
    """What happened when the strategy reconciled with the verification result."""

    spliced: bool  # pre-drafted work was reused this cycle
    cache_hit: bool  # the predicted outcome matched (== spliced for proactive)
    n_reused: int  # scratch nodes carried into the next draft round
    n_hypotheses: int = 1  # bets made this cycle (B for saguaro, 1 for proactive)


class OverlapStrategy(ABC):
    name = "overlap"

    def __init__(self, tree, engine, device, dtype) -> None:
        self._tree = tree
        self._engine = engine
        self._device = device
        self._dtype = dtype

    @property
    def depth_gain(self) -> int:
        """Draft depth already produced during overlap (used by the
        ``proactive.type == "included"`` depth accounting)."""
        return 0

    @abstractmethod
    def speculate(self) -> None:
        """Runs right after the Validate RPC is dispatched, before awaiting it.
        Grows scratch structure(s) into the tree past ``tree.end``."""

    @abstractmethod
    def reconcile(
        self,
        *,
        seq_mask: torch.Tensor,
        last_accepted_token_idx: int,
        extra_token_id: torch.Tensor,
    ) -> OverlapResult:
        """Runs after the RPC returns and the verified path is known.
        Performs the tree + KV reorder (splice or plain) and reports the result."""


def build_overlap_strategy(
    name: str, *, tree, engine, device, dtype, cfg
):  # -> OverlapStrategy | None
    name = (name or "disabled").lower()
    if name == "disabled":
        return None
    if name == "proactive":
        from specedge.client.proactive import ProactiveStrategy

        return ProactiveStrategy(tree, engine, device, dtype, cfg)
    if name == "saguaro":
        from specedge.client.saguaro.strategy import SaguaroStrategy

        return SaguaroStrategy(tree, engine, device, dtype, cfg)
    raise ValueError(f"Invalid overlap strategy: {name}")
