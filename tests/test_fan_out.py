"""Fan-out allocation (Theorem 12 / uniform), ported from ssd/tests."""

import pytest

from specedge.client.saguaro.outcomes import geometric_fan_out, uniform_fan_out


@pytest.mark.parametrize("K", [0, 1, 3, 7])
@pytest.mark.parametrize("B", [0, 1, 5, 12, 32])
def test_uniform_sums_to_budget(K, B):
    fan = uniform_fan_out(K, B)
    assert len(fan) == K + 1
    assert sum(fan) == B
    assert max(fan) - min(fan) <= 1


@pytest.mark.parametrize("K", [0, 1, 3, 7])
@pytest.mark.parametrize("B", [1, 5, 12, 32])
def test_geometric_sums_to_budget(K, B):
    fan = geometric_fan_out(K, B, acceptance_rate=0.6)
    assert len(fan) == K + 1
    assert sum(fan) == B
    assert all(f >= 0 for f in fan)


def test_geometric_front_loaded():
    # deeper positions (less likely reached) get no more guesses than shallower
    fan = geometric_fan_out(K=6, B=40, acceptance_rate=0.5)
    body = fan[:-1]  # exclude the final "all-accepted" bump
    assert body == sorted(body, reverse=True)


def test_geometric_degenerate_rates():
    assert geometric_fan_out(4, 10, acceptance_rate=0.0) == [10, 0, 0, 0, 0]
    assert geometric_fan_out(4, 10, acceptance_rate=1.0) == [0, 0, 0, 0, 10]
    assert geometric_fan_out(4, 0, acceptance_rate=0.5) == [0, 0, 0, 0, 0]


def test_single_candidate_gets_all():
    assert geometric_fan_out(0, 9, acceptance_rate=0.5) == [9]
    assert uniform_fan_out(0, 9) == [9]
