"""Tree + KV-cache reconciliation primitives shared by the overlap strategies.

These are extracted (and generalised) from SpecExecClient so that both the
proactive drafter and the Saguaro speculation cache can splice pre-drafted work
back into the tree after server verification.
"""

from __future__ import annotations

import torch


def reorder_to_verified_path(tree, engine, device, seq_mask: torch.Tensor) -> None:
    """Compact the tree and draft KV cache down to the verified path.

    Equivalent to the former ``SpecExecClient._reorder_by_sequence``.
    """
    seq_indices = torch.where(seq_mask != 0)[0]
    engine.gather(seq_indices, torch.arange(seq_indices.size(-1), device=device))
    tree.reorder_by_sequence(seq_mask, seq_indices)


def append_bonus_token(
    tree, extra_token_id: torch.Tensor, device: torch.device
) -> None:
    """Append the server's bonus token as the new CANDIDATE root of the tree."""
    tree.add(
        token_ids=extra_token_id,
        token_positions=tree.positions[tree.end - 1] + 1,
        parent_indices=torch.tensor([tree.end - 1], device=device),
        logprobs=torch.tensor([0.0], device=device),
    )
    tree.prefix_len = tree.end
    tree.status[: tree.prefix_len - 1] = tree.PROMPT


def label_forest_roots(
    tree, forest_start: int, forest_end: int, root_indices: list[int]
) -> dict[int, int]:
    """Map every scratch-forest node index to the root index of its branch.

    Nodes are added parent-before-child and roots first, so a single ascending
    pass over ``[forest_start, forest_end)`` is enough. A node whose parent lies
    outside the forest (should not happen for non-roots) is labelled ``-1``.
    """
    root_set = {int(r) for r in root_indices}
    root_of: dict[int, int] = {}
    for idx in range(int(forest_start), int(forest_end)):
        if idx in root_set:
            root_of[idx] = idx
        else:
            parent = int(tree.parents[idx].item())
            root_of[idx] = root_of.get(parent, -1)
    return root_of


def splice_scratch_branch(
    tree,
    engine,
    device: torch.device,
    dtype: torch.dtype,
    seq_mask: torch.Tensor,
    branch_src: torch.Tensor,
) -> None:
    """Keep the verified path, then append the scratch nodes in ``branch_src``.

    Generalises the former ``SpecExecClient._reorder_by_sequence_proactive`` from
    a single contiguous proactive subtree to an arbitrary (sorted,
    parent-before-child) branch of the Saguaro scratch forest. Passing
    ``torch.arange(proactive_tree_prefix_len, proactive_tree_end)`` reproduces
    the proactive behaviour exactly.
    """
    branch_src = branch_src.to(device=device, dtype=torch.long)
    seq_indices = torch.where(seq_mask != 0)[0]

    map_size = int(branch_src.max().item()) + 1
    mapping = torch.full((map_size,), -1, dtype=torch.long, device=device)

    old_prefix_len = int(tree.prefix_len)
    new_prefix_len = int(torch.sum(seq_mask).item())

    # (a) remap the accepted portion of the submitted tree
    if torch.any(seq_mask[old_prefix_len:]):
        src = seq_indices[seq_indices >= old_prefix_len]
        dst = torch.arange(old_prefix_len, new_prefix_len, device=device)
        mapping[src] = dst

        tree.tokens[dst] = tree.tokens[src]
        tree.positions[dst] = dst
        tree.parents[dst] = dst - 1
        tree.status[dst] = tree.GENERATED

    # (b) append the scratch branch
    n_branch = int(branch_src.numel())
    dst = torch.arange(new_prefix_len, new_prefix_len + n_branch, device=device)
    mapping[branch_src] = dst

    tree.tokens[dst] = tree.tokens[branch_src]
    tree.positions[dst] = tree.positions[branch_src]
    tree.parents[dst] = mapping[tree.parents[branch_src]]
    tree.status[dst] = tree.status[branch_src]
    tree.logprobs[dst] = tree.logprobs[branch_src]
    tree.amask[..., dst.unsqueeze(-1), dst] = tree.amask[
        ..., branch_src.unsqueeze(-1), branch_src
    ]

    tree.end = new_prefix_len + n_branch
    tree.prefix_len = new_prefix_len + 1

    tree.status[: tree.prefix_len - 1] = tree.PROMPT
    tree.status[tree.prefix_len - 1 : tree.prefix_len + 1] = tree.PROCESSED
    tree.status[tree.status == tree.POST_CANDIDATE] = tree.CANDIDATE
    tree.status[tree.status == tree.POST_PROCESSED] = tree.PROCESSED

    tree.logprobs[tree.end :].zero_()
    tree._data[:, tree.end :].zero_()

    causal = torch.tril(
        torch.ones(tree.prefix_len, tree.prefix_len, dtype=dtype, device=device)
    )
    tree.amask[..., : tree.prefix_len, : tree.prefix_len] = causal
    tree.amask[..., tree.prefix_len : tree.end, : tree.prefix_len] = 1.0

    src = torch.where(seq_mask[: tree.prefix_len])[0]
    engine.gather(src, torch.arange(src.size(-1), device=device))
