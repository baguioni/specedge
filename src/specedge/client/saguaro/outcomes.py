"""Verification-outcome prediction (§4.1 of Kumar et al. 2026), tree-structured.

Given a fan-out budget ``B``, propose the most likely
``(exit_node_idx, bonus_token)`` outcomes to populate the speculation cache.

Both modes walk the same draft tree; they differ only in which of its nodes
count as candidate exit points.

In *leaf* mode the candidates are the draft tree's leaf nodes, ranked by
cumulative log-prob. In *trunk* mode (SpecEdge run with
``max_branch_width == 1``, so the tree is a single trunk) the last confirmed
token and every node on the trunk are candidate exit points, which recovers
the original per-position (accept-depth) enumeration over ``k = 0 .. K``.

Note that leaf mode still omits the zero-accept outcome: the last confirmed
token is a parent, so the leaf filter drops it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


def geometric_reverse_fan_out(
    K: int, B: int, acceptance_rate: float, r: float = 1.0
) -> list[int]:
    """:func:`geometric_fan_out` back to front: the deepest positions get the
    largest share. Not from the paper; an experiment in betting on long accepts
    rather than early exits.
    """
    return geometric_fan_out(K, B, acceptance_rate, r)[::-1]


def select_exit_nodes(tree, max_n_beams: int, exit_mode: str) -> torch.Tensor:
    """Candidate exit points where the verified path may leave the draft tree."""
    device = tree.tokens.device

    if exit_mode == "trunk":
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
        if exit_mode == "trunk":
            # topk(sorted=False) returns an arbitrary order, but the trunk
            # fan-out is indexed by accept depth. Restore that order: shallowest
            # first, ties (siblings at one depth) by cumulative log-prob.
            nodes = nodes[torch.argsort(tree.logprobs[nodes], descending=True)]
            nodes = nodes[torch.argsort(tree.positions[nodes], stable=True)]
    return nodes


def existing_child_tokens(tree, node_idx: int) -> list[int]:
    """Tokens already hanging off ``node_idx`` in the submitted tree.

    None of them can be the server's bonus token at this exit point: exiting
    here means the server rejected this node's drafted children, and the
    residual it samples the bonus from has their mass removed. They are the
    SpecEdge analogue of Saguaro's ``excluded = spec_tokens[k]``.

    Every child must be excluded, not just the first. With
    ``max_branch_width > 1`` a node's children are the top-W of the very
    distribution the fan-out ranks, so excluding one leaves the fan-out free
    to "guess" ranks 2..W -- outcomes the main tree already covers and that
    therefore can never be hit.
    """
    end = int(tree.end)
    child = torch.where(tree.parents[:end] == node_idx)[0]
    child = child[child >= int(tree.prefix_len)]
    if child.numel() == 0:
        return []
    return sorted({int(t) for t in tree.tokens[child].tolist()})


def _as_skip_set(excluded) -> set[int]:
    """Normalise one ``excluded`` entry: ``None``, a token id, or an iterable."""
    if excluded is None:
        return set()
    if isinstance(excluded, int):
        return {excluded}
    return {int(t) for t in excluded}


def outcomes_from_logprobs(
    exit_nodes: list[int],
    logp: torch.Tensor,  # (len(exit_nodes), V)
    fan_out: list[int],  # len(exit_nodes)
    excluded: list,  # len(exit_nodes), each None | int | iterable of int
) -> tuple[list[int], list[int]]:
    """Pick the top-``F`` draft tokens at each exit node as bonus candidates.

    Tokens in that node's ``excluded`` entry are skipped, and the top-k is
    widened by as many, so a node with several excluded children still yields
    its full ``F`` guesses.
    """
    if logp.shape[0] != len(fan_out):
        raise ValueError(f"logp rows ({logp.shape[0]}) != fan_out ({len(fan_out)})")
    vocab = int(logp.shape[-1])
    out_nodes: list[int] = []
    out_bonus: list[int] = []

    for pos, (node_idx, f) in enumerate(zip(exit_nodes, fan_out, strict=True)):
        if f <= 0:
            continue
        skip = _as_skip_set(excluded[pos])
        cand = torch.topk(logp[pos], k=min(f + len(skip), vocab)).indices.tolist()
        taken = 0
        for tok in cand:
            if tok in skip:
                continue
            out_nodes.append(int(node_idx))
            out_bonus.append(int(tok))
            taken += 1
            if taken >= f:
                break
    return out_nodes, out_bonus


@dataclass
class OutcomePrediction:
    """Everything one ``predict_outcomes`` call decided (kept for tracing).

    Attributes:
        candidates: candidate exit nodes, in fan-out order.
        fan: guesses budgeted at each candidate (parallel to ``candidates``).
        excluded: tokens already hanging off each candidate (empty when none).
        exit_nodes: exit node of each predicted outcome.
        bonus_tokens: guessed bonus token of each outcome (parallel to
            ``exit_nodes``).
        bonus_logprobs: draft log-prob of each guessed bonus token.
    """

    candidates: list[int] = field(default_factory=list)
    fan: list[int] = field(default_factory=list)
    excluded: list = field(default_factory=list)
    exit_nodes: list[int] = field(default_factory=list)
    bonus_tokens: list[int] = field(default_factory=list)
    bonus_logprobs: list[float] = field(default_factory=list)


def predict_outcomes(tree, engine, **kwargs) -> tuple[list[int], list[int]]:
    """One draft forward over the candidate exit nodes -> outcome list.

    Returns ``(exit_nodes, bonus_tokens)`` -- parallel lists, one entry per
    ``Outcome`` to plant a scratch branch for. Keyword arguments are those of
    :func:`predict_outcome_details`.
    """
    prediction = predict_outcome_details(tree, engine, **kwargs)
    return prediction.exit_nodes, prediction.bonus_tokens


@torch.inference_mode()
def predict_outcome_details(
    tree,
    engine,
    *,
    budget: int,
    max_n_beams: int,
    acceptance_rate: float,
    fan_out: str = "geometric",
    exit_mode: str = "leaf",
) -> OutcomePrediction:
    """:func:`predict_outcomes`, returning the full :class:`OutcomePrediction`."""
    exit_idx = select_exit_nodes(tree, max_n_beams, exit_mode)
    if exit_idx.numel() == 0:
        return OutcomePrediction()

    if exit_mode != "trunk":
        # Tree leaves sit at varying depths, so rank them by cumulative log-prob.
        # In trunk mode ``exit_idx`` is already in accept-depth order -- which is
        # what ``fan`` is indexed by -- so sorting would only re-derive that order
        # from the (monotonically decreasing) cumulative log-probs along the trunk.
        order = torch.argsort(tree.logprobs[exit_idx], descending=True)
        exit_idx = exit_idx[order]

    K = int(exit_idx.numel()) - 1
    budget = min(int(budget), int(max_n_beams))
    if fan_out == "uniform":
        fan = uniform_fan_out(K, budget)
    elif fan_out == "geometric":
        fan = geometric_fan_out(K, budget, float(acceptance_rate))
    elif fan_out == "geometric_reverse":
        fan = geometric_reverse_fan_out(K, budget, float(acceptance_rate))
    else:
        raise ValueError(
            "saguaro fan_out must be 'uniform', 'geometric' or "
            f"'geometric_reverse', got {fan_out!r}"
        )

    logits = engine.forward(
        input_ids=tree.tokens[exit_idx].unsqueeze(0),
        position_ids=tree.positions[exit_idx].unsqueeze(0),
        cache_batch_indices=torch.zeros_like(exit_idx),
        cache_seq_indices=exit_idx,
        attention_mask=tree.amask[..., exit_idx, :],
    )
    logp = torch.log_softmax(logits[0, -exit_idx.numel() :, :], dim=-1)

    candidates = exit_idx.tolist()
    excluded = [existing_child_tokens(tree, int(n)) for n in candidates]
    nodes, bonus = outcomes_from_logprobs(candidates, logp, fan, excluded)

    row = {int(n): i for i, n in enumerate(candidates)}
    bonus_logp = (
        logp[
            torch.tensor([row[n] for n in nodes], device=logp.device),
            torch.tensor(bonus, device=logp.device),
        ].tolist()
        if nodes
        else []
    )
    return OutcomePrediction(
        candidates=[int(n) for n in candidates],
        fan=[int(f) for f in fan],
        excluded=excluded,
        exit_nodes=nodes,
        bonus_tokens=bonus,
        bonus_logprobs=bonus_logp,
    )
