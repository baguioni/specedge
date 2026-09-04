"""SpeculationCache / Outcome behaviour."""

import dataclasses

import pytest
import torch

from specedge.client.saguaro.cache import (
    CachedSpeculation,
    Outcome,
    SpeculationCache,
)


def _spec(nodes):
    return CachedSpeculation(
        root_scratch_idx=nodes[0],
        node_indices=torch.tensor(nodes, dtype=torch.long),
        n_tokens=len(nodes),
    )


def test_outcome_is_hashable_and_frozen():
    o = Outcome(exit_node_idx=5, bonus=42)
    assert o == Outcome(5, 42)
    assert hash(o) == hash(Outcome(5, 42))
    assert {o: 1}[Outcome(5, 42)] == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.bonus = 7  # type: ignore[misc]


def test_cache_put_get_contains_len():
    cache = SpeculationCache()
    assert len(cache) == 0

    a, b = Outcome(3, 10), Outcome(3, 11)
    cache.put(a, _spec([12, 13]))
    cache.put(b, _spec([14, 15, 16]))

    assert len(cache) == 2
    assert a in cache
    assert Outcome(9, 9) not in cache
    assert cache.get(a).n_tokens == 2
    assert cache.get(b).n_tokens == 3
    assert cache.get(Outcome(9, 9)) is None


def test_distinct_exit_same_bonus_are_distinct_keys():
    cache = SpeculationCache()
    cache.put(Outcome(3, 10), _spec([20]))
    cache.put(Outcome(4, 10), _spec([21]))
    assert len(cache) == 2
