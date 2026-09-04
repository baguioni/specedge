"""Speculation cache (Definition 4 in Kumar et al. 2026), tree-structured.

A verification outcome in SpecEdge is *which leaf the verified path exits at*
plus the server's bonus token -- ``Outcome(exit_node_idx, bonus)``. The cached
continuation is a branch of the proactive scratch forest, referenced by the
node indices it occupies (its draft KV already lives at those same indices in
the engine cache until it is spliced or discarded).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass(frozen=True)
class Outcome:
    """A verification outcome ``v = (exit_node_idx, bonus_token_id)``.

    ``exit_node_idx`` is the pre-reorder tree index of the last accepted token
    (``SpecExecClient``'s ``last_accepted_token_idx``); ``bonus`` is the token
    the server sampled after it.
    """

    exit_node_idx: int
    bonus: int


@dataclass
class CachedSpeculation:
    """A pre-drafted continuation sitting in the scratch forest.

    Attributes:
        root_scratch_idx: index of this branch's root (the bonus-token node).
        node_indices: sorted absolute indices of every node in the branch.
        n_tokens: ``node_indices.numel()`` -- tokens reused on a cache hit.
    """

    root_scratch_idx: int
    node_indices: torch.Tensor
    n_tokens: int


@dataclass
class SpeculationCache:
    """Dict-like mapping from :class:`Outcome` to :class:`CachedSpeculation`."""

    entries: dict[Outcome, CachedSpeculation] = field(default_factory=dict)

    def put(self, outcome: Outcome, value: CachedSpeculation) -> None:
        self.entries[outcome] = value

    def get(self, outcome: Outcome) -> Optional[CachedSpeculation]:
        return self.entries.get(outcome)

    def __contains__(self, outcome: Outcome) -> bool:
        return outcome in self.entries

    def __len__(self) -> int:
        return len(self.entries)
