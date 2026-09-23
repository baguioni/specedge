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


def test_excludes_every_existing_child_not_just_the_first():
    # With max_branch_width > 1 the node's children are the top-W of this same
    # distribution, so all of them must be skipped or the fan-out "guesses" a
    # token the main tree already covers.
    logp = _logp([{1: 9.0, 2: 8.0, 3: 7.0, 4: 6.0}])
    nodes, bonus = outcomes_from_logprobs([5], logp, [2], [[1, 2]])
    assert list(zip(nodes, bonus, strict=True)) == [(5, 3), (5, 4)]


def test_empty_exclusion_list_behaves_like_none():
    logp = _logp([{1: 9.0, 2: 8.0}])
    assert outcomes_from_logprobs([5], logp, [1], [[]]) == ([5], [1])
    assert outcomes_from_logprobs([5], logp, [1], [None]) == ([5], [1])


def _wide_tree():
    """Prompt [1, 2, 3] plus a 2-wide, 3-deep draft tree (9 nodes > 4 beams)."""
    from specedge.tree import Tree

    tree = Tree(
        prefix_tokens=torch.tensor([[1, 2, 3]]),
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_len=32,
    )
    parent = torch.tensor([2, 2])
    lp = torch.tensor([-0.1, -2.0])
    for depth in range(3):
        tree.add(
            token_ids=torch.tensor([10 + depth, 20 + depth]),
            token_positions=torch.full((2,), 3 + depth),
            parent_indices=parent,
            logprobs=lp,
            token_status=tree.CANDIDATE,
        )
        parent = torch.tensor([tree.end - 2, tree.end - 1])
        lp = lp - 1.0
    return tree


def test_trunk_exit_nodes_ordered_by_depth_after_topk():
    from specedge.client.saguaro.outcomes import select_exit_nodes

    tree = _wide_tree()
    nodes = select_exit_nodes(tree, max_n_beams=4, exit_mode="trunk")

    # Top 4 by log-prob: last prompt token (0.0), depth-1 best (-0.1),
    # depth-2 best (-1.1), depth-1 second (-2.0). Log-prob order would be
    # [2, 3, 5, 4]; the fan-out needs accept-depth order.
    assert nodes.tolist() == [2, 3, 4, 5]
    depths = tree.positions[nodes].tolist()
    assert depths == sorted(depths)
    # Siblings at one depth: higher log-prob first.
    assert tree.logprobs[3] > tree.logprobs[4]
