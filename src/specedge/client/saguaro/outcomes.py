"""Verification-outcome prediction (§4.1 of Kumar et al. 2026), tree-structured.

Given a fan-out budget ``B``, propose the most likely
``(exit_node_idx, bonus_token)`` outcomes to populate the speculation cache.

In *tree* mode the candidate exit points are the draft tree's leaf nodes,
ranked by cumulative log-prob. In *linear* mode (SpecEdge run with
``max_branch_width == 1``) the last confirmed token and every node on the chain
are candidate exit points, which recovers the original per-position
(accept-depth) enumeration over ``k = 0 .. K``.

Note that tree mode still omits the zero-accept outcome: the last confirmed
token is a parent, so the leaf filter drops it.
"""

from __future__ import annotations

import torch


def uniform_fan_out(K: int, B: int) -> list[int]:
    """Distribute budget ``B`` as evenly as possible across ``K + 1`` positions."""
    if K < 0:
        return []
    base, rem = divmod(B, K + 1)
    return [base + (1 if i < rem else 0) for i in range(K + 1)]


def geometric_fan_out(
    K: int, B: int, acceptance_rate: float, r: float = 1.0
) -> list[int]:
    """Capped geometric fan-out (Theorem 12, Kumar et al. 2026).

    Optimal ``F_k`` under a budget constraint when the rejection rate falls as
    ``1 / F**r``::

        F_k = F_0 * a_p ** (k / (1 + r))                       for k < K
        F_K = F_0 * a_p ** (K / (1 + r)) * (1 - a_p) ** (-1 / (1 + r))

    ``F_0`` is chosen so ``sum(F_k) == B``. Positions deeper in the speculation
    are less likely to be reached, so they get fewer guesses.
    """
    if K < 0:
        return []
    if B <= 0:
        return [0] * (K + 1)
    if acceptance_rate <= 0:
        return [B] + [0] * K
    if acceptance_rate >= 1:
        return [0] * K + [B]

    exp = 1.0 / (1.0 + r)
    alpha = acceptance_rate**exp
    beta = (1.0 - acceptance_rate) ** exp

    weights = [alpha**k for k in range(K)] + [alpha**K / beta]
    total = sum(weights)
    raw = [w * B / total for w in weights]

    fan = [round(x) for x in raw]
    diff = B - sum(fan)
    if diff != 0:
        residues = sorted(
            range(len(fan)), key=lambda i: raw[i] - fan[i], reverse=(diff > 0)
        )
        step = 1 if diff > 0 else -1
        for i in residues[: abs(diff)]:
            fan[i] += step
    return [max(0, f) for f in fan]


def select_exit_nodes(tree, max_n_beams: int, linear: bool) -> torch.Tensor:
    """Candidate exit points where the verified path may leave the draft tree."""
    device = tree.tokens.device

    if linear:
        # ``prefix_len - 1`` holds the last confirmed token, which is the exit
        # point for the zero-accept outcome (``k = 0`` in Kumar et al.) -- the
        # server reports it as ``last_accepted_token_idx = prefix_len - 1``.
        # Including it both restores that outcome (the most likely single one,
        # with probability ``1 - a_p``) and keeps the fan-out schedule indexed
        # by accept depth, so ``fan[k]`` is the budget for ``k`` accepted tokens.
        start = max(0, int(tree.prefix_len) - 1)
        nodes = torch.arange(start, int(tree.end), device=device)
    else:
        rng = torch.arange(int(tree.prefix_len), int(tree.end), device=device)
        if rng.numel() == 0:
            return rng
        parents = torch.unique(tree.parents[: int(tree.end)])
        nodes = rng[~torch.isin(rng, parents)]

    if nodes.numel() == 0:
        return nodes

    if nodes.numel() > max_n_beams:
        top = torch.topk(tree.logprobs[nodes], k=int(max_n_beams), sorted=False).indices
        nodes = nodes[top]
    return nodes


def existing_child_token(tree, node_idx: int):  # -> int | None
    """Token already hanging off ``node_idx`` in the submitted tree, if any.

    That token cannot be the server's bonus token at this exit point (it is
    already covered by the main tree), so it is excluded from the fan-out --
    the SpecEdge analogue of Saguaro's ``excluded = spec_tokens[k]``.
    """
    end = int(tree.end)
    child = torch.where(tree.parents[:end] == node_idx)[0]
    child = child[child >= int(tree.prefix_len)]
    if child.numel() == 0:
        return None
    return int(tree.tokens[int(child[0])].item())


def outcomes_from_logprobs(
    exit_nodes: list[int],
    logp: torch.Tensor,  # (len(exit_nodes), V)
    fan_out: list[int],  # len(exit_nodes)
    excluded: list,  # len(exit_nodes), each int | None
) -> tuple[list[int], list[int]]:
    """Pick the top-``F`` draft tokens at each exit node as bonus candidates."""
    if logp.shape[0] != len(fan_out):
        raise ValueError(f"logp rows ({logp.shape[0]}) != fan_out ({len(fan_out)})")
    vocab = int(logp.shape[-1])
    out_nodes: list[int] = []
    out_bonus: list[int] = []

    for pos, (node_idx, f) in enumerate(zip(exit_nodes, fan_out, strict=True)):
        if f <= 0:
            continue
        cand = torch.topk(logp[pos], k=min(f + 1, vocab)).indices.tolist()
        skip = excluded[pos]
        taken = 0
        for tok in cand:
            if tok == skip:
                continue
            out_nodes.append(int(node_idx))
            out_bonus.append(int(tok))
            taken += 1
            if taken >= f:
                break
    return out_nodes, out_bonus


@torch.inference_mode()
def predict_outcomes(
    tree,
    engine,
    *,
    budget: int,
    max_n_beams: int,
    acceptance_rate: float,
    fan_out: str = "geometric",
    linear: bool = False,
) -> tuple[list[int], list[int]]:
    """One draft forward over the candidate exit nodes -> outcome list.

    Returns ``(exit_nodes, bonus_tokens)`` -- parallel lists, one entry per
    ``Outcome`` to plant a scratch branch for.
    """
    exit_idx = select_exit_nodes(tree, max_n_beams, linear)
    if exit_idx.numel() == 0:
        return [], []

    if not linear:
        # Tree leaves sit at varying depths, so rank them by cumulative log-prob.
        # In linear mode ``exit_idx`` is already in accept-depth order -- which is
        # what ``fan`` is indexed by -- so sorting would only re-derive that order
        # from the (monotonically decreasing) cumulative log-probs along the chain.
        order = torch.argsort(tree.logprobs[exit_idx], descending=True)
        exit_idx = exit_idx[order]

    K = int(exit_idx.numel()) - 1
    budget = min(int(budget), int(max_n_beams))
    if fan_out == "uniform":
        fan = uniform_fan_out(K, budget)
    else:
        fan = geometric_fan_out(K, budget, float(acceptance_rate))

    logits = engine.forward(
        input_ids=tree.tokens[exit_idx].unsqueeze(0),
        position_ids=tree.positions[exit_idx].unsqueeze(0),
        cache_batch_indices=torch.zeros_like(exit_idx),
        cache_seq_indices=exit_idx,
        attention_mask=tree.amask[..., exit_idx, :],
    )
    logp = torch.log_softmax(logits[0, -exit_idx.numel() :, :], dim=-1)

    excluded = [existing_child_token(tree, int(n)) for n in exit_idx.tolist()]
    return outcomes_from_logprobs(exit_idx.tolist(), logp, fan, excluded)
