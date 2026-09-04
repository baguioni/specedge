"""Saguaro speculation cache for SpecEdge (adapted from Kumar et al. 2026).

Ported concepts, tree-structured for SpecEdge:

* ``Outcome(k_accepted, bonus)``        -> ``Outcome(exit_node_idx, bonus)``
* fan-out over K+1 accept positions     -> fan-out over candidate exit leaves
* ``predict_outcomes`` top-F_k draft    -> same, per exit leaf / chain node
* ``SpeculationCache``                  -> same, values point at scratch-forest branches

Not ported: Saguaro sampling (the draft-distribution ``C`` down-weight) -- the
bonus token is sampled by the server's target model, which the edge cannot bias.
"""

from specedge.client.saguaro.cache import (
    CachedSpeculation,
    Outcome,
    SpeculationCache,
)
from specedge.client.saguaro.outcomes import (
    geometric_fan_out,
    predict_outcomes,
    uniform_fan_out,
)

__all__ = [
    "CachedSpeculation",
    "Outcome",
    "SpeculationCache",
    "geometric_fan_out",
    "predict_outcomes",
    "uniform_fan_out",
]
