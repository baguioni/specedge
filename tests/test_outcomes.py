"""Outcome enumeration from draft log-probs (no model needed)."""

import torch

from specedge.client.saguaro.outcomes import outcomes_from_logprobs


def _logp(rows):
    """rows: list of dicts token_id -> score; missing tokens get a low score."""
    vocab = 10
    t = torch.full((len(rows), vocab), -20.0)
    for i, row in enumerate(rows):
        for tok, val in row.items():
            t[i, tok] = val
    return torch.log_softmax(t, dim=-1)


def test_respects_fan_out_and_exclusions():
    exit_nodes = [7, 8, 9]
    logp = _logp(
        [
            {1: 5.0, 2: 4.0, 3: 3.0, 4: 2.0},  # node 7
            {5: 5.0, 6: 4.0},  # node 8
            {0: 9.0},  # node 9  (fan_out 0 -> skipped)
        ]
    )
    fan_out = [2, 1, 0]
    excluded = [2, None, 5]  # node 7 excludes token 2; node 9 excluded ignored (f=0)

    nodes, bonus = outcomes_from_logprobs(exit_nodes, logp, fan_out, excluded)

    # node 7: top tokens [1,2,3,...]; drop 2 -> [1,3]; node 8: [5]; node 9: none
    assert list(zip(nodes, bonus, strict=True)) == [(7, 1), (7, 3), (8, 5)]


def test_zero_budget_returns_nothing():
    logp = _logp([{1: 5.0}])
    nodes, bonus = outcomes_from_logprobs([4], logp, [0], [None])
    assert nodes == [] and bonus == []


def test_mismatched_shapes_raise():
    logp = _logp([{1: 1.0}, {2: 1.0}])
    try:
        outcomes_from_logprobs([1, 2], logp, [1], [None])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on fan_out length mismatch")


def test_exclusion_can_free_a_slot_for_next_token():
    # fan_out 2, but top-1 is excluded -> should still return 2 real tokens
    logp = _logp([{1: 9.0, 2: 8.0, 3: 7.0}])
    nodes, bonus = outcomes_from_logprobs([5], logp, [2], [1])
    assert list(zip(nodes, bonus, strict=True)) == [(5, 2), (5, 3)]
