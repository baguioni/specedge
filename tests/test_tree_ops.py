"""label_forest_roots + splice_scratch_branch on a CPU Tree (no model)."""

import torch

from specedge.client.reorder import label_forest_roots, splice_scratch_branch
from specedge.tree import Tree

CPU = torch.device("cpu")
F32 = torch.float32


class _FakeParents:
    def __init__(self, parents):
        self.parents = torch.tensor(parents, dtype=torch.long)


class _FakeEngine:
    def __init__(self):
        self.calls = []

    def gather(self, src, dst):
        self.calls.append((src.tolist(), dst.tolist()))


def test_label_forest_roots_multi_branch():
    #  idx:      0 1 2 3   4    5    6    7    8
    #  role:     <-prefix-> root root cont cont cont
    #  parent:   0 0 1 2   2    3    4    4    5
    fake = _FakeParents([0, 0, 1, 2, 2, 3, 4, 4, 5])
    root_of = label_forest_roots(fake, 4, 9, root_indices=[4, 5])
    assert root_of == {4: 4, 5: 5, 6: 4, 7: 4, 8: 5}


def test_label_forest_roots_orphan_is_minus_one():
    fake = _FakeParents([0, 0, 0, 0, 2, 1])  # node 5's parent (1) is not in forest
    root_of = label_forest_roots(fake, 4, 6, root_indices=[4])
    assert root_of == {4: 4, 5: -1}


def _fresh_tree():
    tree = Tree(
        prefix_tokens=torch.tensor([10, 11, 12, 13]),
        device=CPU,
        dtype=F32,
        max_len=64,
    )
    # a finished draft round: winning path prefix -> 4 -> 5, node 6 a dead branch
    tree.add(
        token_ids=torch.tensor([20]),
        token_positions=torch.tensor([4]),
        parent_indices=torch.tensor([3]),
        logprobs=torch.tensor([-1.0]),
        token_status=tree.PROCESSED,
    )
    tree.add(
        token_ids=torch.tensor([21]),
        token_positions=torch.tensor([5]),
        parent_indices=torch.tensor([4]),
        logprobs=torch.tensor([-2.0]),
        token_status=tree.PROCESSED,
    )
    tree.add(
        token_ids=torch.tensor([99]),
        token_positions=torch.tensor([4]),
        parent_indices=torch.tensor([3]),
        logprobs=torch.tensor([-5.0]),
        token_status=tree.CANDIDATE,
    )
    # scratch forest: winning branch [7,8,9] off exit leaf 5, losing root 10 off 4
    for tok, pos, par in [(30, 6, 5), (31, 7, 7), (32, 8, 8), (40, 5, 4)]:
        tree.add(
            token_ids=torch.tensor([tok]),
            token_positions=torch.tensor([pos]),
            parent_indices=torch.tensor([par]),
            logprobs=torch.tensor([-3.0]),
            token_status=tree.POST_CANDIDATE,
        )
    return tree


def test_splice_scratch_branch_reuses_winning_branch():
    tree = _fresh_tree()
    engine = _FakeEngine()

    seq_mask = torch.zeros(tree.end, dtype=torch.bool)
    seq_mask[:6] = True  # prefix 0..3 plus accepted 4,5

    splice_scratch_branch(
        tree,
        engine,
        CPU,
        F32,
        seq_mask,
        branch_src=torch.tensor([7, 8, 9]),
    )

    # verified path (6 nodes) then the 3 spliced scratch nodes
    assert int(tree.end) == 9
    assert int(tree.prefix_len) == 7

    assert tree.tokens[6].item() == 30  # bonus token node
    assert tree.tokens[7].item() == 31
    assert tree.tokens[8].item() == 32

    assert tree.parents[6].item() == 5
    assert tree.parents[7].item() == 6
    assert tree.parents[8].item() == 7

    # bonus + first spliced node are settled; the deeper node is the new frontier
    assert tree.status[6].item() == tree.PROCESSED.item()
    assert tree.status[7].item() == tree.PROCESSED.item()
    assert tree.status[8].item() == tree.CANDIDATE.item()
    assert torch.all(tree.status[:6] == tree.PROMPT)

    # losing branch + rejected nodes wiped
    assert tree.tokens[9:].sum().item() == 0
    assert not torch.any(tree.status == tree.POST_CANDIDATE)

    # KV cache compacted to the accepted prefix
    assert engine.calls == [([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])]


def test_splice_scratch_branch_matches_contiguous_proactive_range():
    # A single contiguous branch [7,8,9] must behave identically whether passed
    # as an explicit list or an arange (the proactive call site).
    a = _fresh_tree()
    b = _fresh_tree()
    m = torch.zeros(a.end, dtype=torch.bool)
    m[:6] = True

    branch = torch.tensor([7, 8, 9])
    splice_scratch_branch(a, _FakeEngine(), CPU, F32, m.clone(), branch)
    splice_scratch_branch(b, _FakeEngine(), CPU, F32, m.clone(), torch.arange(7, 10))

    assert torch.equal(a._data[:, : a.end], b._data[:, : b.end])
    assert int(a.end) == int(b.end)
    assert int(a.prefix_len) == int(b.prefix_len)
