"""Visualise one SpecEdge cycle under different overlap strategies.

Runs a single prompt through prefill + first draft tree -> verification, with
the overlap strategy drafting past the tree while verification is in flight,
exactly as ``SpecExecClient._validate_tree`` does. For each strategy it records

  1. the first draft tree (token ids, parents, cumulative scores),
  2. the overlap extension: every guessed (exit node, bonus token) pair and the
     scratch branch drafted under it,
  3. the verifier's result: accepted path, exit node, bonus token, and whether
     a scratch branch was spliced into the next round,

then renders them into one self-contained HTML page (+ a JSON trace).

Draft and target run in this one process -- no gRPC server. The target model
stands in for the batch server with the same prefill / tree forward / sampling
(``InferenceController._inference``), memoised so every strategy is compared
against the identical verification result.

Example (GPU box, draft and target on separate GPUs):

    python src/script/visualize_overlap.py \\
        --draft-model Qwen/Qwen3-1.7B --target-model Qwen/Qwen3-14B \\
        --device cuda:0 --target-device cuda:1 --req-idx 0

Small local run (Apple silicon / CPU):

    python src/script/visualize_overlap.py \\
        --draft-model Qwen/Qwen3-0.6B --target-model Qwen/Qwen3-1.7B --device mps
"""

import argparse
import asyncio
import html
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)  # util.load_dataset reads data/ relative to the repo root

import torch  # noqa: E402

import util  # noqa: E402
from config import SpecEdgeClientConfig  # noqa: E402
from model.cache import KVCache  # noqa: E402
from specedge.client.proactive import ProactiveStrategy  # noqa: E402
from specedge.client.saguaro import outcomes as saguaro_outcomes  # noqa: E402
from specedge.client.specexec import SpecExecClient  # noqa: E402

STRATEGIES = {
    "proactive": {"SPECEDGE_OVERLAP_STRATEGY": "proactive"},
    "saguaro_uniform": {
        "SPECEDGE_OVERLAP_STRATEGY": "saguaro",
        "SPECEDGE_SAGUARO_FAN_OUT": "uniform",
        "SPECEDGE_SAGUARO_EXIT_MODE": "leaf",
    },
    "saguaro_uniform_linear": {
        "SPECEDGE_OVERLAP_STRATEGY": "saguaro",
        "SPECEDGE_SAGUARO_FAN_OUT": "uniform",
        "SPECEDGE_SAGUARO_EXIT_MODE": "trunk",
    },
    "saguaro_geom": {
        "SPECEDGE_OVERLAP_STRATEGY": "saguaro",
        "SPECEDGE_SAGUARO_FAN_OUT": "geometric",
        "SPECEDGE_SAGUARO_EXIT_MODE": "leaf",
    },
    "saguaro_geom_linear": {
        "SPECEDGE_OVERLAP_STRATEGY": "saguaro",
        "SPECEDGE_SAGUARO_FAN_OUT": "geometric",
        "SPECEDGE_SAGUARO_EXIT_MODE": "trunk",
    },
}

STRATEGY_BLURB = {
    "proactive": (
        "One deep bet. Picks the draft leaf with the best cumulative score, "
        "guesses the target's bonus token after it, and drafts a subtree from "
        "that guess. Reused only if the verifier exits at that exact leaf "
        "and samples that exact token."
    ),
    "saguaro_uniform": (
        "A fan-out of shallow bets. Splits the guess budget B evenly across the "
        "draft tree's leaves (best-scored first), guesses the top draft tokens "
        "at each, and grows a short branch under every guess. Reused if any "
        "(exit leaf, bonus token) pair matches."
    ),
    "saguaro_uniform_linear": (
        "Same uniform fan-out, but every node of the draft tree is a candidate "
        "exit point, including the last prompt token (the zero-accept outcome), "
        "not only leaves."
    ),
    "saguaro_geom": (
        "Fan-out of shallow bets over the draft tree's leaves, with the budget "
        "split geometrically (Theorem 12): better-ranked exit points get more "
        "guesses."
    ),
    "saguaro_geom_linear": (
        "Geometric fan-out with every draft-tree node (including the last "
        "prompt token) as a candidate exit point."
    ),
}


# --------------------------------------------------------------------------- #
# Engines and the in-process verifier
# --------------------------------------------------------------------------- #
class EagerEngine:
    """``GraphEngine``'s interface without CUDA-graph capture, so the script
    also runs on MPS / CPU. One cycle is too short for capture to pay off."""

    def __init__(self, model, max_len: int) -> None:
        self.max_len = max_len
        self._model = model
        self._device = model.device
        self._dtype = model.dtype
        self._past_key_values = KVCache(
            config=model.config,
            max_n_beams=1,
            max_len=max_len,
            device=model.device,
            dtype=model.dtype,
            batch_size=1,
        )

    @torch.inference_mode()
    def forward(
        self,
        input_ids,
        position_ids,
        cache_batch_indices,
        cache_seq_indices,
        attention_mask,
    ):
        # KVCache.update picks the new K/V rows through seq_indices, which the
        # graph engine bakes per beam count at capture time; set it per call.
        self._past_key_values.seq_indices = torch.arange(
            input_ids.size(1), device=self._device
        )
        return self._model.forward(
            input_ids=input_ids.to(self._device),
            position_ids=position_ids.to(self._device),
            cache_batch_indices=cache_batch_indices.to(self._device),
            cache_seq_indices=cache_seq_indices.to(self._device),
            attention_mask=util.invert_mask(
                attention_mask.to(self._device, self._dtype)
            ),
            past_key_values=self._past_key_values,
        )[0]

    @torch.inference_mode()
    def prefill(
        self, input_ids, position_ids, batch_idx, cache_seq_indices, attention_mask
    ):
        attention_mask = util.invert_mask(attention_mask.to(self._device, self._dtype))
        cache_batch_indices = torch.zeros(
            (input_ids.size(-1),), dtype=torch.long, device=self._device
        )
        with self._past_key_values.prefill_context(input_ids.size(-1), batch_idx):
            self._model.forward(
                input_ids=input_ids.to(self._device),
                position_ids=position_ids.to(self._device),
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices.to(self._device),
                attention_mask=attention_mask,
                past_key_values=self._past_key_values,
            )

    def gather(self, src_indices, dest_indices):
        self._past_key_values.gather(0, src_indices, dest_indices)

    def reset(self):
        self._past_key_values.clear()


class LocalVerifier:
    """Stands in for the batch server's ``Validate`` RPC (prefill cycle only)."""

    def __init__(self, engine, tokenizer, temperature, seed, client_device) -> None:
        self._engine = engine
        self._tokenizer = tokenizer
        self._temperature = temperature
        self._seed = seed
        self._client_device = client_device
        self._memo: dict[tuple, torch.Tensor] = {}
        self.last: dict | None = None

    async def request(
        self,
        client_idx,
        req_idx,
        input_ids,
        position_ids,
        cache_seq_indices,
        attention_mask,
        parent_indices,
        prefill=False,
        prefix=None,
    ):
        if not prefill:
            raise NotImplementedError("LocalVerifier only runs the prefill cycle")

        key = (
            tuple(input_ids.flatten().tolist()),
            tuple(parent_indices.flatten().tolist()),
            prefix,
        )
        if key not in self._memo:
            self._memo[key] = self._verify(
                input_ids, position_ids, cache_seq_indices, attention_mask, prefix
            )
        selection = self._memo[key]
        self.last = {
            "indices": cache_seq_indices.flatten().tolist(),
            "selection": selection.flatten().tolist(),
        }
        return selection.to(self._client_device), 1

    @torch.inference_mode()
    def _verify(self, input_ids, position_ids, cache_seq_indices, attention_mask, prefix):
        engine = self._engine
        dev = engine._device
        engine.reset()

        # Same as the server's runtime prefill: every prompt token but the last,
        # which arrives as the root of the draft tree.
        ids = self._tokenizer.encode(prefix, return_tensors="pt").to(dev)[..., :-1]
        n = ids.size(-1)
        pos = torch.arange(n, device=dev)
        engine.prefill(
            input_ids=ids,
            position_ids=pos.unsqueeze(0),
            batch_idx=0,
            cache_seq_indices=pos,
            attention_mask=torch.ones(
                (1, 1, n, engine.max_len), dtype=engine._dtype, device=dev
            ).tril_(),
        )

        logits = engine.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_batch_indices=torch.zeros(
                input_ids.size(-1), dtype=torch.long, device=dev
            ),
            cache_seq_indices=cache_seq_indices,
            attention_mask=attention_mask,
        )
        util.set_seed(self._seed)
        return util.sampler_from_logits(logits, temperature=self._temperature)[0]


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #
_saguaro_capture: dict = {}
_orig_outcomes_from_logprobs = saguaro_outcomes.outcomes_from_logprobs


def _recording_outcomes_from_logprobs(exit_nodes, logp, fan_out, excluded):
    """Wraps the Saguaro outcome picker to record the exact exit-point order,
    fan-out and each guess's draft log-prob (predict_outcomes looks the picker
    up in module globals, so patching the module attribute is enough)."""
    nodes, bonus = _orig_outcomes_from_logprobs(exit_nodes, logp, fan_out, excluded)
    row = {int(n): i for i, n in enumerate(exit_nodes)}
    _saguaro_capture.update(
        exit_nodes=[int(n) for n in exit_nodes],
        fan=[int(f) for f in fan_out],
        excluded=list(excluded),
        guess_logp={
            (int(n), int(b)): float(logp[row[int(n)], int(b)])
            for n, b in zip(nodes, bonus, strict=True)
        },
    )
    return nodes, bonus


saguaro_outcomes.outcomes_from_logprobs = _recording_outcomes_from_logprobs


def snapshot(tree, indices, tokenizer) -> dict[int, dict]:
    idx = torch.tensor(sorted(indices), dtype=torch.long, device=tree.tokens.device)
    if idx.numel() == 0:
        return {}
    tokens = tree.tokens[idx].tolist()
    parents = tree.parents[idx].tolist()
    logprobs = tree.logprobs[idx].tolist()
    status = tree.status[idx].tolist()
    positions = tree.positions[idx].tolist()
    return {
        i: {
            "idx": i,
            "token_id": t,
            "text": tokenizer.decode([t]),
            "parent": p,
            "logprob": lp,
            "status": s,
            "position": pos,
        }
        for i, t, p, lp, s, pos in zip(
            idx.tolist(), tokens, parents, logprobs, status, positions, strict=True
        )
    }


def capture_bets(overlap, tree) -> tuple[list[dict], list[dict]]:
    """(bets, exit-point candidates) right after ``speculate()``, before the
    reconcile step reorders the tree. Scratch data past ``tree.end`` is intact."""
    if isinstance(overlap, ProactiveStrategy):
        leaf, tok, p_prefix, p_end = overlap._pending
        if leaf is None:
            return [], []
        bet = {
            "exit": int(leaf),
            "bonus": int(tok),
            "root": int(p_prefix),
            "nodes": list(range(int(p_prefix), int(p_end))),
            "has_frontier": True,
            "guess_logp": None,
        }
        return [bet], [{"node": int(leaf), "fan": 1}]

    cache = overlap._cache
    bets = []
    if cache is not None:
        guess_logp = _saguaro_capture.get("guess_logp", {})
        for outcome, spec in cache.entries.items():
            bets.append(
                {
                    "exit": int(outcome.exit_node_idx),
                    "bonus": int(outcome.bonus),
                    "root": int(spec.root_scratch_idx),
                    "nodes": [int(i) for i in spec.node_indices.tolist()],
                    "has_frontier": bool(spec.has_frontier),
                    "guess_logp": guess_logp.get(
                        (int(outcome.exit_node_idx), int(outcome.bonus))
                    ),
                }
            )
    candidates = [
        {"node": n, "fan": f}
        for n, f in zip(
            _saguaro_capture.get("exit_nodes", []),
            _saguaro_capture.get("fan", []),
            strict=True,
        )
    ]
    return bets, candidates


async def run_strategy(name, env, args, draft_engine, tokenizer, verifier, prompt):
    os.environ.update(env)
    SpecEdgeClientConfig.reset()
    _saguaro_capture.clear()

    client = SpecExecClient(
        engine=draft_engine, tokenizer=tokenizer, prompt=prompt, max_len=args.max_len
    )
    client._validator = verifier
    tree = client._tree
    overlap = client._overlap

    util.set_seed(args.seed)
    client._grow_tree(prefill=True)

    root = int(tree.prefix_len) - 1
    draft = {
        "root": root,
        "prefix_len": int(tree.prefix_len),
        "end": int(tree.end),
        "prompt_tail": snapshot(tree, range(max(0, root - 3), root), tokenizer),
        "nodes": snapshot(tree, range(root, int(tree.end)), tokenizer),
    }

    rec: dict = {}
    speculate, reconcile = overlap.speculate, overlap.reconcile

    def recording_speculate():
        speculate()
        bets, candidates = capture_bets(overlap, tree)
        scratch_idx = {i for bet in bets for i in bet["nodes"]}
        rec.update(
            bets=bets, candidates=candidates, scratch=snapshot(tree, scratch_idx, tokenizer)
        )

    def recording_reconcile(**kw):
        rec["seq_mask"] = kw["seq_mask"].detach().to("cpu", torch.bool).clone()
        rec["exit"] = int(kw["last_accepted_token_idx"])
        rec["bonus"] = int(kw["extra_token_id"].flatten()[0].item())
        result = reconcile(**kw)
        rec["result"] = result
        return result

    overlap.speculate = recording_speculate
    overlap.reconcile = recording_reconcile

    await client._validate_tree(req_idx=args.req_idx, prefill=True)

    prefix_len = draft["prefix_len"]
    accepted = [i for i in torch.where(rec["seq_mask"])[0].tolist() if i >= prefix_len]
    target_choice = dict(
        zip(verifier.last["indices"], verifier.last["selection"], strict=True)
    )
    result = rec["result"]
    matched = next(
        (
            b
            for b in rec.get("bets", [])
            if b["exit"] == rec["exit"] and b["bonus"] == rec["bonus"]
        ),
        None,
    )

    return {
        "name": name,
        "draft": draft,
        "bets": rec.get("bets", []),
        "candidates": rec.get("candidates", []),
        "scratch": rec.get("scratch", {}),
        "verify": {
            "exit": rec["exit"],
            "bonus": rec["bonus"],
            "bonus_text": tokenizer.decode([rec["bonus"]]),
            "accepted": accepted,
            "target_choice": target_choice,
        },
        "result": {
            "spliced": bool(result.spliced),
            "cache_hit": bool(result.cache_hit),
            "n_reused": int(result.n_reused),
            "n_hypotheses": int(result.n_hypotheses),
            "matched_root": matched["root"] if matched else None,
        },
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
COL_W, NODE_W, NODE_H, ROW_H = 118, 88, 20, 28
PAD_X, PAD_TOP, PAD_BOTTOM = 16, 30, 14
LABEL_CHARS = 11


def show_token(text: str) -> str:
    return text.replace(" ", "␣").replace("\n", "↵").replace("\t", "⇥") or "∅"


def label(text: str) -> str:
    s = show_token(text)
    return s if len(s) <= LABEL_CHARS else s[: LABEL_CHARS - 1] + "…"


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def render_tree(nodes: dict, root, *, aria: str, axis_origin: int = 0) -> str:
    """Horizontal tidy tree. ``nodes``: id -> {parent, text, title, cls,
    edge_cls, order, badge}. Depth runs left to right, one leaf per row."""
    children = defaultdict(list)
    for nid, n in nodes.items():
        if nid != root:
            children[n["parent"]].append(nid)
    for kids in children.values():
        kids.sort(key=lambda c: nodes[c]["order"])

    pos: dict = {}
    row = 0

    def place(nid, depth):
        nonlocal row
        kids = [k for k in children.get(nid, []) if k in nodes]
        if not kids:
            pos[nid] = (depth, row)
            row += 1
            return
        for k in kids:
            place(k, depth + 1)
        pos[nid] = (depth, (pos[kids[0]][1] + pos[kids[-1]][1]) / 2)

    place(root, 0)

    max_d = max(d for d, _ in pos.values())
    width = PAD_X * 2 + max_d * COL_W + NODE_W + 8
    height = PAD_TOP + (row - 1) * ROW_H + NODE_H + PAD_BOTTOM

    def xy(nid):
        d, r = pos[nid]
        return PAD_X + d * COL_W, PAD_TOP + r * ROW_H

    out = [
        f'<svg class="tree" role="img" aria-label="{esc(aria)}" '
        f'viewBox="0 0 {width} {height:g}" width="{width}" height="{height:g}">'
    ]

    out.append('<g class="axis">')
    for d in range(max_d + 1):
        rel = d - axis_origin
        if rel < 0:
            continue
        txt = "prompt end" if rel == 0 else f"+{rel}"
        out.append(
            f'<text x="{PAD_X + d * COL_W + NODE_W / 2:g}" y="14" '
            f'text-anchor="middle">{txt}</text>'
        )
    out.append("</g>")

    for nid in pos:
        if nid == root:
            continue
        px, py = xy(nodes[nid]["parent"])
        cx, cy = xy(nid)
        x1, y1 = px + NODE_W, py + NODE_H / 2
        x2, y2 = cx, cy + NODE_H / 2
        xm = (x1 + x2) / 2
        out.append(
            f'<path class="e {nodes[nid].get("edge_cls", "")}" '
            f'd="M{x1:g} {y1:g} C{xm:g} {y1:g} {xm:g} {y2:g} {x2:g} {y2:g}"/>'
        )

    for nid in pos:
        n = nodes[nid]
        x, y = xy(nid)
        out.append(f'<g class="n {n.get("cls", "")}">')
        out.append(f"<title>{esc(n.get('title', ''))}</title>")
        out.append(
            f'<rect x="{x:g}" y="{y:g}" width="{NODE_W}" height="{NODE_H}" rx="3"/>'
        )
        out.append(f'<text x="{x + 6:g}" y="{y + 14:g}">{esc(label(n["text"]))}</text>')
        out.append("</g>")
        badge = n.get("badge")
        if badge is not None:
            text, cls = badge
            out.append(
                f'<g class="badge {cls}"><circle cx="{x + NODE_W:g}" cy="{y:g}" r="7"/>'
                f'<text x="{x + NODE_W:g}" y="{y + 3:g}" text-anchor="middle">'
                f"{esc(text)}</text></g>"
            )

    out.append("</svg>")
    return "".join(out)


def node_title(rec: dict, extra: str = "") -> str:
    lines = [
        f"#{rec['idx']}  {show_token(rec['text'])!r}  (id {rec['token_id']})",
        f"cumulative score {rec['logprob']:.2f}",
    ]
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def main_tree_nodes(run: dict, *, with_verify: bool) -> dict:
    draft, verify = run["draft"], run["verify"]
    root = draft["root"]
    accepted = set(verify["accepted"]) if with_verify else set()
    nodes = {}
    for idx, rec in draft["nodes"].items():
        cls = ["prompt"] if idx == root else []
        extra = ""
        if with_verify:
            if idx in accepted:
                cls.append("accepted")
            if idx == verify["exit"]:
                cls.append("exit")
            tc = verify["target_choice"].get(idx)
            if tc is not None:
                extra = f"target's next token here: id {tc}"
        nodes[idx] = {
            "parent": rec["parent"],
            "text": rec["text"],
            "title": node_title(rec, extra),
            "cls": " ".join(cls),
            "edge_cls": "accepted" if idx in accepted else "",
            "order": (0, -rec["logprob"], idx),
        }
    return nodes


def strategy_tree(run: dict) -> str:
    nodes = main_tree_nodes(run, with_verify=True)
    verify, res, scratch = run["verify"], run["result"], run["scratch"]
    matched_root = res["matched_root"]

    for rank, bet in enumerate(run["bets"]):
        is_match = bet["root"] == matched_root
        reused = is_match and res["spliced"]
        for idx in bet["nodes"]:
            rec = scratch[idx]
            cls = ["reused" if reused else "ext"]
            parent = rec["parent"]
            extra = ""
            if idx == bet["root"]:
                cls.append("guess")
                parent = bet["exit"]
                extra = f"guessed bonus after #{bet['exit']}"
                if bet["guess_logp"] is not None:
                    extra += f"  (draft log p {bet['guess_logp']:.2f})"
                if is_match:
                    cls.append("hitroot")
                    extra += "  -- matches the verifier's bonus"
                    if not res["spliced"]:
                        extra += ", but the branch has no frontier: not reused"
            nodes[idx] = {
                "parent": parent,
                "text": rec["text"],
                "title": node_title(rec, extra),
                "cls": " ".join(cls),
                "edge_cls": "reused" if reused else "ext",
                "order": (1, rank, -rec["logprob"], idx),
            }

    if matched_root is None:
        nodes["ghost"] = {
            "parent": verify["exit"],
            "text": verify["bonus_text"],
            "title": (
                f"verifier's bonus token {show_token(verify['bonus_text'])!r} "
                f"(id {verify['bonus']}) after #{verify['exit']} -- no guess matched"
            ),
            "cls": "ghost",
            "edge_cls": "ghost",
            "order": (2, 0, 0),
        }

    for cand in run["candidates"]:
        if cand["node"] in nodes:
            nodes[cand["node"]]["badge"] = (
                (str(cand["fan"]), "") if cand["fan"] > 0 else ("", "zero")
            )

    return render_tree(
        nodes,
        run["draft"]["root"],
        aria=f"{run['name']}: first draft tree with the overlap extension",
    )


def chain_nodes(run: dict) -> tuple[dict, object, int]:
    """Prompt tail -> last prompt token -> accepted tokens, linked in a line."""
    draft, verify = run["draft"], run["verify"]
    nodes = {}
    tail = sorted(draft["prompt_tail"])
    prev = None
    for idx in tail:
        rec = draft["prompt_tail"][idx]
        nodes[f"p{idx}"] = {
            "parent": prev,
            "text": rec["text"],
            "title": f"prompt token {show_token(rec['text'])!r}",
            "cls": "prompt faint",
            "order": (0,),
        }
        prev = f"p{idx}"
    root = draft["root"]
    root_rec = draft["nodes"][root]
    nodes[root] = {
        "parent": prev,
        "text": root_rec["text"],
        "title": "last prompt token (root of the draft tree)",
        "cls": "prompt",
        "order": (0,),
    }
    prev = root
    for idx in verify["accepted"]:
        rec = draft["nodes"][idx]
        nodes[idx] = {
            "parent": prev,
            "text": rec["text"],
            "title": node_title(rec, "accepted by the target"),
            "cls": "accepted",
            "edge_cls": "accepted",
            "order": (0,),
        }
        prev = idx
    first = f"p{tail[0]}" if tail else root
    return nodes, first, len(tail)


def verifier_chain(run: dict) -> str:
    nodes, first, origin = chain_nodes(run)
    draft, verify = run["draft"], run["verify"]
    exit_idx = verify["exit"]
    nodes["bonus"] = {
        "parent": exit_idx,
        "text": verify["bonus_text"],
        "title": f"bonus token sampled by the target (id {verify['bonus']})",
        "cls": "bonus",
        "edge_cls": "bonus",
        "order": (0,),
    }
    rejected = sorted(
        (r for r in draft["nodes"].values() if r["parent"] == exit_idx and r["idx"] != draft["root"]),
        key=lambda r: -r["logprob"],
    )
    for rec in rejected[:5]:
        nodes[f"r{rec['idx']}"] = {
            "parent": exit_idx,
            "text": rec["text"],
            "title": node_title(rec, "draft child here -- rejected by the target"),
            "cls": "rejected",
            "edge_cls": "rejected",
            "order": (1, -rec["logprob"]),
        }
    if len(rejected) > 5:
        nodes["rmore"] = {
            "parent": exit_idx,
            "text": f"+{len(rejected) - 5} more",
            "title": "more rejected draft children",
            "cls": "rejected more",
            "edge_cls": "rejected",
            "order": (2,),
        }
    return render_tree(
        nodes, first, aria="verifier's accepted path as a chain", axis_origin=origin
    )


def carried_chain(run: dict) -> str:
    nodes, first, origin = chain_nodes(run)
    verify, res = run["verify"], run["result"]
    bet = next((b for b in run["bets"] if b["root"] == res["matched_root"]), None)
    if bet is not None and res["spliced"]:
        for idx in bet["nodes"]:
            rec = run["scratch"][idx]
            is_root = idx == bet["root"]
            nodes[idx] = {
                "parent": verify["exit"] if is_root else rec["parent"],
                "text": rec["text"],
                "title": node_title(
                    rec, "bonus token = spliced guess" if is_root else "reused draft"
                ),
                "cls": "reused hitroot" if is_root else "reused",
                "edge_cls": "bonus" if is_root else "reused",
                "order": (0, -rec["logprob"], idx),
            }
    else:
        nodes["bonus"] = {
            "parent": verify["exit"],
            "text": verify["bonus_text"],
            "title": "bonus token -- round 2 drafts from here",
            "cls": "bonus",
            "edge_cls": "bonus",
            "order": (0,),
        }
    return render_tree(
        nodes, first, aria=f"{run['name']}: what round 2 starts from", axis_origin=origin
    )


def describe_outcome(run: dict) -> tuple[str, str, str]:
    """(pill class, pill text, sentence)."""
    verify, res = run["verify"], run["result"]
    exit_rec = run["draft"]["nodes"][verify["exit"]]
    exit_txt = f"#{verify['exit']} <code>{esc(show_token(exit_rec['text']))}</code>"
    bonus_txt = f"<code>{esc(show_token(verify['bonus_text']))}</code>"
    at_exit = [b for b in run["bets"] if b["exit"] == verify["exit"]]

    if res["spliced"]:
        return (
            "hit",
            f"hit · {res['n_reused']} reused",
            f"The verifier exited at {exit_txt} and sampled {bonus_txt}, which "
            f"was one of the guesses. Its branch ({res['n_reused']} tokens) is "
            f"spliced in, so round 2 starts with that draft already built.",
        )
    if res["matched_root"] is not None:
        return (
            "partial",
            "hit · not reused",
            f"A guess matched ({exit_txt} then {bonus_txt}), but its branch "
            f"has no open frontier left to extend, so the client falls back to "
            f"a plain reorder.",
        )
    if at_exit:
        guesses = ", ".join(
            f"<code>{esc(show_token(run['scratch'][b['root']]['text']))}</code>"
            for b in at_exit
        )
        return (
            "miss",
            "miss · wrong token",
            f"The exit point {exit_txt} was covered ({guesses}), but the target "
            f"sampled {bonus_txt}. Nothing is reused.",
        )
    return (
        "miss",
        "miss · wrong exit",
        f"No guess sits at the verifier's exit point {exit_txt}. Nothing is "
        f"reused. Round 2 drafts from {bonus_txt}.",
    )


CSS = """
:root {
  --bg: #f5f7f3; --surface: #ffffff; --ink: #1b201c; --muted: #58635a;
  --line: #d3dacf; --node-line: #a9b4a5;
  --accept: #3d7d51; --accept-tint: #e3efe3;
  --bonus: #a9641f; --bonus-tint: #f2e3d0;
  --reject: #9e3b3b; --reject-tint: #f4e2e0;
  --ext: #3b6a9e; --ext-tint: #e2eaf4; --on-ext: #ffffff;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #131612; --surface: #1b1f1a; --ink: #e7ebe4; --muted: #98a79b;
    --line: #333c34; --node-line: #56625a;
    --accept: #6cb582; --accept-tint: #23342a;
    --bonus: #d99f60; --bonus-tint: #382c1c;
    --reject: #cf6d6d; --reject-tint: #3a2222;
    --ext: #86b0de; --ext-tint: #1e2a38; --on-ext: #10151c;
  }
}
:root[data-theme="dark"] {
  --bg: #131612; --surface: #1b1f1a; --ink: #e7ebe4; --muted: #98a79b;
  --line: #333c34; --node-line: #56625a;
  --accept: #6cb582; --accept-tint: #23342a;
  --bonus: #d99f60; --bonus-tint: #382c1c;
  --reject: #cf6d6d; --reject-tint: #3a2222;
  --ext: #86b0de; --ext-tint: #1e2a38; --on-ext: #10151c;
}
* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--ink); margin: 0;
  font: 15px/1.6 "IBM Plex Sans", system-ui, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1120px; margin: 0 auto; padding-inline: 20px; padding-block: 40px 72px; }
h1, h2, h3 { font-family: Fraunces, Georgia, serif; font-weight: 500; text-wrap: balance; margin: 0; }
h1 { font-size: 34px; line-height: 1.15; }
h2 { font-size: 24px; margin-top: 56px; }
h3 { font-size: 20px; }
p { max-width: 68ch; margin: 0; }
code, .mono { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.92em; }
.lede { color: var(--muted); margin-top: 12px; }
.stack { display: flex; flex-direction: column; gap: 14px; }
.eyebrow { font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.meta {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 10px 28px; margin: 28px 0 0; padding: 18px 0; border-block: 1px solid var(--line);
}
.meta div { display: flex; flex-direction: column; }
.meta dt { font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
.meta dd { margin: 0; font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 13px; overflow-wrap: anywhere; }
details.prompt { margin-top: 18px; }
details.prompt summary { cursor: pointer; color: var(--muted); }
details.prompt pre {
  white-space: pre-wrap; background: var(--surface); border: 1px solid var(--line);
  padding: 12px 14px; border-radius: 4px; font-size: 13px; max-height: 260px; overflow: auto;
}
.table-scroll { overflow-x: auto; margin-top: 18px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 9px 14px 9px 0; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); font-weight: 500; }
td.num { text-align: right; padding-right: 22px; }
th.num { text-align: right; padding-right: 22px; }
a { color: var(--ext); }
a:focus-visible, summary:focus-visible { outline: 2px solid var(--ext); outline-offset: 2px; }
.pill {
  display: inline-block; font-size: 12px; font-weight: 500; padding: 2px 9px;
  border-radius: 999px; white-space: nowrap; border: 1px solid;
}
.pill.hit { color: var(--accept); background: var(--accept-tint); border-color: var(--accept); }
.pill.partial { color: var(--bonus); background: var(--bonus-tint); border-color: var(--bonus); }
.pill.miss { color: var(--reject); background: var(--reject-tint); border-color: var(--reject); }
.legend { display: flex; flex-wrap: wrap; gap: 8px 22px; margin-top: 16px; font-size: 13px; color: var(--muted); }
.legend span { display: inline-flex; align-items: center; gap: 7px; }
.sw { width: 22px; height: 13px; border-radius: 3px; border: 1px solid var(--node-line); background: var(--surface); }
.sw.accepted { background: var(--accept-tint); border: 1.5px solid var(--accept); }
.sw.exit { border: 2.5px solid var(--accept); background: var(--accept-tint); }
.sw.guess { background: var(--ext-tint); border: 1.5px solid var(--ext); }
.sw.ext { background: var(--ext-tint); border: 1px dashed var(--ext); }
.sw.reused { background: var(--ext); border-color: var(--ext); }
.sw.bonus { background: var(--bonus-tint); border: 1.5px solid var(--bonus); }
.sw.ghost { background: transparent; border: 1.5px dashed var(--bonus); }
.sw.rejected { background: var(--reject-tint); border: 1px dashed var(--reject); }
.sw.badge { width: 14px; height: 14px; border-radius: 50%; background: var(--ext); border: none; }
figure { margin: 0; display: flex; flex-direction: column; gap: 8px; }
figcaption { font-size: 13px; color: var(--muted); max-width: 80ch; }
.scroll { overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 4px; }
.scroll.tall { max-height: 78vh; overflow-y: auto; }
.strategy { margin-top: 40px; padding-top: 28px; border-top: 1px solid var(--line); display: flex; flex-direction: column; gap: 16px; }
.strategy header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 14px; }
.strategy header code { font-size: 13px; color: var(--muted); }
.facts { display: flex; flex-wrap: wrap; gap: 6px 26px; margin: 0; font-size: 13px; }
.facts div { display: flex; gap: 6px; }
.facts dt { color: var(--muted); }
.facts dd { margin: 0; font-variant-numeric: tabular-nums; font-weight: 500; }
svg.tree { display: block; max-width: none; }
svg.tree text { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px; }
svg.tree .axis text { fill: var(--muted); font-family: "IBM Plex Sans", system-ui, sans-serif; font-size: 10.5px; letter-spacing: 0.04em; }
.e { fill: none; stroke: var(--node-line); stroke-width: 1; }
.e.accepted { stroke: var(--accept); stroke-width: 2; }
.e.ext { stroke: var(--ext); stroke-dasharray: 3 2; }
.e.reused { stroke: var(--ext); stroke-width: 2; }
.e.bonus { stroke: var(--bonus); stroke-width: 2; }
.e.ghost { stroke: var(--bonus); stroke-width: 1.5; stroke-dasharray: 4 3; }
.e.rejected { stroke: var(--reject); stroke-dasharray: 2 2; }
.n rect { fill: var(--surface); stroke: var(--node-line); stroke-width: 1; }
.n text { fill: var(--ink); }
.n.prompt rect { fill: var(--bg); stroke: var(--line); }
.n.prompt text { fill: var(--muted); }
.n.faint { opacity: 0.6; }
.n.accepted rect { fill: var(--accept-tint); stroke: var(--accept); stroke-width: 1.5; }
.n.exit rect { stroke: var(--accept); stroke-width: 3; }
.n.ext rect { fill: var(--ext-tint); stroke: var(--ext); stroke-dasharray: 3 2; }
.n.ext.guess rect { stroke-dasharray: none; stroke-width: 1.5; }
.n.reused rect { fill: var(--ext); stroke: var(--ext); }
.n.reused text { fill: var(--on-ext); }
.n.hitroot rect { stroke: var(--bonus); stroke-width: 3; stroke-dasharray: none; }
.n.bonus rect { fill: var(--bonus-tint); stroke: var(--bonus); stroke-width: 1.5; }
.n.ghost rect { fill: var(--bg); stroke: var(--bonus); stroke-width: 1.5; stroke-dasharray: 4 3; }
.n.ghost text { fill: var(--bonus); }
.n.rejected rect { fill: var(--reject-tint); stroke: var(--reject); stroke-dasharray: 2 2; }
.n.rejected text { fill: var(--reject); text-decoration: line-through; }
.n.rejected.more text { text-decoration: none; }
.badge circle { fill: var(--ext); }
.badge text { fill: var(--on-ext); font-size: 9px !important; font-family: "IBM Plex Sans", system-ui, sans-serif !important; font-weight: 600; }
.badge.zero circle { r: 3.5; fill: var(--surface); stroke: var(--node-line); }
.warn { color: var(--reject); }
"""


LEGEND = """
<div class="legend" aria-label="legend">
  <span><i class="sw"></i>draft token</span>
  <span><i class="sw accepted"></i>accepted by target</span>
  <span><i class="sw exit"></i>exit point (last accepted)</span>
  <span><i class="sw bonus"></i>target's bonus token</span>
  <span><i class="sw rejected"></i>rejected draft child</span>
  <span><i class="sw guess"></i>overlap guess (bonus bet)</span>
  <span><i class="sw ext"></i>overlap continuation</span>
  <span><i class="sw reused"></i>spliced into round 2</span>
  <span><i class="sw ghost"></i>actual bonus, unguessed</span>
  <span><i class="sw badge"></i>guesses planted at node</span>
</div>
"""


def render_page(runs: list[dict], args, prompt: str, notes: list[str]) -> str:
    base = runs[0]
    draft, verify = base["draft"], base["verify"]
    n_draft = len(draft["nodes"]) - 1
    n_acc = len(verify["accepted"])
    exit_rec = draft["nodes"][verify["exit"]]

    meta = [
        ("draft model", args.draft_model),
        ("target model", args.target_model),
        ("device", f"{args.device} / {args.target_device} · {args.dtype}"),
        ("request", f"{args.dataset}[{args.req_idx}]" if not args.prompt else "custom prompt"),
        ("temperature · seed", f"{args.temperature} · {args.seed}"),
        (
            "draft tree",
            f"beams {args.max_n_beams} · depth {args.max_beam_len} · "
            f"width {args.max_branch_width} · budget {args.max_budget}",
        ),
        (
            "proactive",
            f"depth {args.proactive_max_beam_len} · width "
            f"{args.proactive_max_branch_width} · budget {args.proactive_max_budget}",
        ),
        (
            "saguaro",
            f"B {args.saguaro_budget} · branch {args.saguaro_branch_len} · "
            f"a_p {args.saguaro_acceptance_rate}",
        ),
    ]
    meta_html = "".join(
        f"<div><dt>{esc(k)}</dt><dd>{esc(v)}</dd></div>" for k, v in meta
    )

    rows = []
    for run in runs:
        pill_cls, pill_txt, _ = describe_outcome(run)
        funded = sum(1 for c in run["candidates"] if c["fan"] > 0)
        rows.append(
            f"<tr><td><a href='#s-{esc(run['name'])}'><code>{esc(run['name'])}</code></a></td>"
            f"<td class='num'>{len(run['bets'])}</td>"
            f"<td class='num'>{funded} / {len(run['candidates'])}</td>"
            f"<td class='num'>{len(run['scratch'])}</td>"
            f"<td><span class='pill {pill_cls}'>{esc(pill_txt)}</span></td>"
            f"<td class='num'>{run['result']['n_reused']}</td></tr>"
        )

    first_tree = render_tree(
        main_tree_nodes(base, with_verify=False),
        draft["root"],
        aria="first draft tree",
    )

    sections = []
    for run in runs:
        pill_cls, pill_txt, sentence = describe_outcome(run)
        funded = sum(1 for c in run["candidates"] if c["fan"] > 0)
        sections.append(
            f"""
<section class="strategy" id="s-{esc(run['name'])}">
  <header><h3>{esc(run['name'].replace('_', ' '))}</h3>
    <span class="pill {pill_cls}">{esc(pill_txt)}</span></header>
  <p>{esc(STRATEGY_BLURB.get(run['name'], ''))}</p>
  <dl class="facts">
    <div><dt>guesses</dt><dd>{len(run['bets'])}</dd></div>
    <div><dt>exit points funded</dt><dd>{funded} of {len(run['candidates'])} considered</dd></div>
    <div><dt>overlap tokens drafted</dt><dd>{len(run['scratch'])}</dd></div>
    <div><dt>tokens reused</dt><dd>{run['result']['n_reused']}</dd></div>
  </dl>
  <p>{sentence}</p>
  <figure>
    <div class="scroll tall">{strategy_tree(run)}</div>
    <figcaption>The same draft tree, with everything {esc(run['name'])} drafted while the
    target was verifying (blue). Numbered dots are the guesses planted at each exit
    point; hollow dots are candidates that got no budget. Green is the path the target
    accepted.</figcaption>
  </figure>
  <figure>
    <div class="scroll">{carried_chain(run)}</div>
    <figcaption>What round 2 starts from: the accepted chain plus the bonus token,
    and, on a hit, the spliced branch that is already drafted.</figcaption>
  </figure>
</section>"""
        )

    notes_html = "".join(f"<p class='warn'>{esc(n)}</p>" for n in notes)
    exit_desc = (
        "the last prompt token (nothing accepted)"
        if verify["exit"] == draft["root"]
        else f"node #{verify['exit']} <code>{esc(show_token(exit_rec['text']))}</code>"
    )

    return f"""<title>SpecEdge Overlap Replay</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
  <div class="stack">
    <span class="eyebrow">SpecEdge · one prefill cycle</span>
    <h1>How each overlap strategy drafts past a tree that is still being verified</h1>
    <p class="lede">The edge drafts a token tree and sends it to the target. While
    the target verifies it, the overlap strategy keeps drafting past the tree's
    edge, betting on where verification will stop and which bonus token comes next.
    When the result returns, a correct bet's branch is spliced into round 2.
    All strategies below share the same draft tree and the same verification
    result.</p>
    {notes_html}
  </div>
  <dl class="meta">{meta_html}</dl>
  <details class="prompt"><summary>Prompt ({draft['prefix_len']} tokens)</summary>
    <pre>{esc(prompt)}</pre></details>

  <h2>Comparison</h2>
  <div class="table-scroll"><table>
    <thead><tr><th>strategy</th><th class="num">guesses</th><th class="num">exit points funded</th>
    <th class="num">overlap tokens</th><th>outcome</th><th class="num">reused</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table></div>

  <h2>1 · First draft tree</h2>
  <p class="lede">{n_draft} draft tokens grown from the last prompt token. Children
  are ordered by cumulative draft score, best on top. Hover a token for its index,
  id and score.</p>
  <figure style="margin-top:16px">
    <div class="scroll tall">{first_tree}</div>
    <figcaption>The tree the edge sends for verification. Column labels count
    tokens past the prompt.</figcaption>
  </figure>

  <h2>2 · What the verifier accepted</h2>
  <p class="lede">The target accepted {n_acc} draft token{'s' if n_acc != 1 else ''} and
  exited at {exit_desc}. It then sampled the bonus token
  <code>{esc(show_token(verify['bonus_text']))}</code>, so this cycle yields
  {n_acc + 1} tokens.</p>
  <figure style="margin-top:16px">
    <div class="scroll">{verifier_chain(base)}</div>
    <figcaption>The accepted path as a chain. Red tokens are the draft's other
    children at the exit point, which the target rejected in favour of the bonus
    token.</figcaption>
  </figure>

  <h2>3 · How each strategy extends the tree</h2>
  {LEGEND}
  {''.join(sections)}
</div>
"""


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def base_env(args, out_dir: Path) -> dict[str, str]:
    return {
        "SPECEDGE_RESULT_PATH": str(out_dir),
        "SPECEDGE_EXP_NAME": "visualize_overlap",
        "SPECEDGE_PROCESS_NAME": "client",
        "SPECEDGE_SEED": str(args.seed),
        "SPECEDGE_OPTIMIZATION": "2",  # no-sync timers: no CUDA calls on MPS/CPU
        "SPECEDGE_MAX_LEN": str(args.max_len),
        "SPECEDGE_DRAFT_MODEL": args.draft_model,
        "SPECEDGE_DEVICE": args.device,
        "SPECEDGE_DTYPE": args.dtype,
        "SPECEDGE_REASONING": "False",
        "SPECEDGE_DATASET": args.dataset,
        "SPECEDGE_MAX_N_BEAMS": str(args.max_n_beams),
        "SPECEDGE_MAX_BEAM_LEN": str(args.max_beam_len),
        "SPECEDGE_MAX_BRANCH_WIDTH": str(args.max_branch_width),
        "SPECEDGE_MAX_BUDGET": str(args.max_budget),
        "SPECEDGE_PROACTIVE_TYPE": "excluded",
        "SPECEDGE_PROACTIVE_MAX_N_BEAMS": str(args.proactive_max_n_beams),
        "SPECEDGE_PROACTIVE_MAX_BEAM_LEN": str(args.proactive_max_beam_len),
        "SPECEDGE_PROACTIVE_MAX_BRANCH_WIDTH": str(args.proactive_max_branch_width),
        "SPECEDGE_PROACTIVE_MAX_BUDGET": str(args.proactive_max_budget),
        "SPECEDGE_SAGUARO_BUDGET": str(args.saguaro_budget),
        "SPECEDGE_SAGUARO_BRANCH_LEN": str(args.saguaro_branch_len),
        "SPECEDGE_SAGUARO_ACCEPTANCE_RATE": str(args.saguaro_acceptance_rate),
        "SPECEDGE_MAX_NEW_TOKENS": "64",
        "SPECEDGE_MAX_REQUEST_NUM": "-1",
        "SPECEDGE_REQ_OFFSET": "0",
        "SPECEDGE_SAMPLE_REQ_CNT": "1",
        "SPECEDGE_HOST": "127.0.0.1:0",  # unused: the verifier runs in-process
        "SPECEDGE_CLIENT_IDX": "0",
    }


def parse_args():
    if torch.cuda.is_available():
        default_device = "cuda:0"
    elif torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--draft-model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--target-model", default="Qwen/Qwen3-14B")
    p.add_argument("--device", default=default_device, help="draft device")
    p.add_argument("--target-device", default=None, help="default: --device")
    p.add_argument("--dtype", default=None, help="fp16 | bf16 | fp32 (default: fp32 on cpu, else fp16)")
    p.add_argument("--dataset", default="specbench")
    p.add_argument("--req-idx", type=int, default=0)
    p.add_argument("--prompt", default=None, help="raw user message; overrides --dataset/--req-idx")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--max-n-beams", type=int, default=32)
    p.add_argument("--max-beam-len", type=int, default=4)
    p.add_argument("--max-branch-width", type=int, default=16)
    p.add_argument("--max-budget", type=int, default=32)
    p.add_argument("--proactive-max-n-beams", type=int, default=32)
    p.add_argument("--proactive-max-beam-len", type=int, default=3)
    p.add_argument("--proactive-max-branch-width", type=int, default=16)
    p.add_argument("--proactive-max-budget", type=int, default=32)
    p.add_argument("--saguaro-budget", type=int, default=8)
    p.add_argument("--saguaro-branch-len", type=int, default=3)
    p.add_argument("--saguaro-acceptance-rate", type=float, default=0.5)
    p.add_argument(
        "--strategies",
        default="proactive,saguaro_uniform,saguaro_uniform_linear",
        help=f"comma list of: {', '.join(STRATEGIES)}",
    )
    p.add_argument("--out", default="result/visualize_overlap")
    args = p.parse_args()

    args.target_device = args.target_device or args.device
    if args.dtype is None:
        args.dtype = "fp32" if args.device == "cpu" else "fp16"
    args.strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [s for s in args.strategies if s not in STRATEGIES]
    if unknown:
        p.error(f"unknown strategies: {unknown}")
    return args


def load_prompt(args, tokenizer) -> str:
    if args.prompt:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            enable_thinking=False,
            add_generation_prompt=True,
        )
    dataset = util.load_dataset(args.dataset, model_name=args.draft_model, reasoning=False)
    return dataset[args.req_idx]


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


async def amain(args):
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(base_env(args, out_dir))

    dtype = util.convert_dtype(args.dtype)
    print(f"loading draft {args.draft_model} on {args.device} ({args.dtype})")
    draft_model = util.load_graph_model(args.draft_model, torch.device(args.device), dtype)
    print(f"loading target {args.target_model} on {args.target_device} ({args.dtype})")
    target_model = util.load_graph_model(
        args.target_model, torch.device(args.target_device), dtype
    )

    tokenizer = util.load_tokenizer(args.draft_model)
    target_tokenizer = util.load_tokenizer(args.target_model)
    prompt = load_prompt(args, tokenizer)

    draft_engine = EagerEngine(draft_model, args.max_len)
    verifier = LocalVerifier(
        EagerEngine(target_model, args.max_len),
        target_tokenizer,
        args.temperature,
        args.seed,
        torch.device(args.device),
    )

    runs = []
    for name in args.strategies:
        print(f"running {name}")
        runs.append(
            await run_strategy(
                name, STRATEGIES[name], args, draft_engine, tokenizer, verifier, prompt
            )
        )

    notes = []
    ref = [(r["token_id"], r["parent"]) for r in runs[0]["draft"]["nodes"].values()]
    for run in runs[1:]:
        cur = [(r["token_id"], r["parent"]) for r in run["draft"]["nodes"].values()]
        if cur != ref:
            notes.append(
                f"Draft tree for {run['name']} differs from {runs[0]['name']}'s; "
                "its panel shows its own tree."
            )
    if len(verifier._memo) > 1:
        notes.append("Verification inputs differed between strategies.")

    stem = f"overlap_{args.dataset}_{args.req_idx}" if not args.prompt else "overlap_custom"
    html_path = out_dir / f"{stem}.html"
    json_path = out_dir / f"{stem}.json"
    html_path.write_text(render_page(runs, args, prompt, notes))
    json_path.write_text(
        json.dumps(
            jsonable({"args": vars(args), "prompt": prompt, "runs": runs}),
            indent=1,
            ensure_ascii=False,
        )
    )

    v = runs[0]["verify"]
    print(
        f"\ndraft tree {len(runs[0]['draft']['nodes']) - 1} tokens · accepted "
        f"{len(v['accepted'])} · exit #{v['exit']} · bonus {show_token(v['bonus_text'])!r}"
    )
    for run in runs:
        _, pill, _ = describe_outcome(run)
        print(
            f"  {run['name']:<24} guesses {len(run['bets']):>2} · overlap tokens "
            f"{len(run['scratch']):>3} · {pill}"
        )
    for n in notes:
        print("  note:", n)
    print(f"\nwrote {html_path.relative_to(REPO)}\nwrote {json_path.relative_to(REPO)}")


if __name__ == "__main__":
    asyncio.run(amain(parse_args()))
