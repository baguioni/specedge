"""Render a Saguaro tree trace (``<process_name>.tree_trace.jsonl``).

The trace is written by ``specedge.client.saguaro.trace.TreeTracer`` when
``client.proactive.saguaro.trace: true``. Every decoding step shows three
views of the tree:

  1. the committed chain and the draft tree sent for verification,
  2. the Saguaro fan-out: guessed bonus tokens and the branches under them,
  3. the verified chain: accepted draft tokens plus the server's bonus token.

Token ids are decoded with the draft model's tokenizer (taken from the trace,
or ``--tokenizer``).

    # one self-contained HTML page with a request picker and a step slider
    python src/script/render_tree_trace.py result/demo/specedge/client_0.tree_trace.jsonl

    # plain-text trees on stdout, e.g. over SSH
    python src/script/render_tree_trace.py TRACE --text --req 3 --steps 0-4
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

TAIL = 6  # committed tokens shown before the root


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_trace(path: Path) -> tuple[dict, list[dict]]:
    """``(run, requests)``; each request is ``{..., "steps": [step records]}``."""
    run: dict = {}
    requests: list[dict] = []
    current = None
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            # The client may have been killed mid-write.
            print(f"warning: {path}:{line_no}: unreadable record skipped", file=sys.stderr)
            continue
        kind = rec.get("type")
        if kind == "run":
            run = rec
        elif kind == "request":
            current = {**rec, "steps": []}
            requests.append(current)
        elif kind == "step":
            if current is None or rec["req_idx"] != current["req_idx"]:
                raise ValueError(f"{path}:{line_no}: step record before its request")
            current["steps"].append(rec)
    return run, requests


def annotate(requests: list[dict]) -> None:
    """Add renderer-only fields (prefixed ``_``) to every step record."""
    for req in requests:
        chain = list(req["prompt"])
        for step in req["steps"]:
            chain += step["chain_new"]
            if len(chain) != step["chain_len"]:
                print(
                    f"warning: req {req['req_idx']} step {step['step_idx']}: rebuilt "
                    f"chain has {len(chain)} tokens, trace says {step['chain_len']}",
                    file=sys.stderr,
                )
            root = step["chain_len"] - 1
            step["_tail"] = chain[max(0, root - TAIL) : root]
            step["_outcome"] = outcome(step)
        last = req["steps"][-1] if req["steps"] else None
        if last is not None:
            chain += last["verify"]["accepted_tokens"] + [last["verify"]["bonus"]]
        req["_generated"] = chain[len(req["prompt"]) :]


def outcome(step: dict) -> dict:
    """Did a Saguaro guess match the verifier's (exit node, bonus token)?"""
    v = step["verify"]
    at_exit = [b for b in step["saguaro"]["bets"] if b["exit"] == v["exit"]]
    match = next((b for b in at_exit if b["bonus"] == v["bonus"]), None)
    if v["spliced"]:
        kind = "hit"
    elif match is not None:
        kind = "partial"  # matched, but the branch had no frontier to extend
    elif at_exit:
        kind = "miss_token"
    else:
        kind = "miss_exit"
    return {
        "kind": kind,
        "hit_root": match["root"] if match is not None else None,
        "guesses_at_exit": [b["bonus"] for b in at_exit],
    }


def token_ids(requests: list[dict]) -> set[int]:
    ids: set[int] = set()
    for req in requests:
        for step in req["steps"]:
            ids.update(step["_tail"])
            ids.update(n["token"] for n in step["draft"]["nodes"])
            ids.update(n["token"] for n in step["saguaro"]["forest"])
            ids.update(b["bonus"] for b in step["saguaro"]["bets"])
            ids.update(
                c["excluded"]
                for c in step["saguaro"]["candidates"]
                if c["excluded"] is not None
            )
            ids.update(step["verify"]["accepted_tokens"])
            ids.add(step["verify"]["bonus"])
    return ids


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #
def show(text: str) -> str:
    return text.replace(" ", "␣").replace("\n", "↵").replace("\t", "⇥") or "∅"


def ascii_tree(root, children: dict, label) -> list[str]:
    lines = [label(root)]

    def walk(node, prefix: str) -> None:
        kids = children.get(node, [])
        for i, kid in enumerate(kids):
            last = i == len(kids) - 1
            lines.append(prefix + ("└── " if last else "├── ") + label(kid))
            walk(kid, prefix + ("    " if last else "│   "))

    walk(root, "")
    return lines


def text_step(step: dict, vocab: dict[int, str]) -> list[str]:
    def tok(token_id: int) -> str:
        return repr(show(vocab[token_id]))

    draft, sag, v = step["draft"], step["saguaro"], step["verify"]
    root = draft["root"]
    nodes = {n["idx"]: n for n in draft["nodes"]}
    n_reused = sum(n["reused"] for n in draft["nodes"])
    tail = " ".join(tok(t) for t in step["_tail"])
    new = " ".join(tok(t) for t in step["chain_new"])

    out = [
        f"=== req {step['req_idx']} · step {step['step_idx']}"
        + (" (prefill)" if step["prefill"] else ""),
        f"chain: {step['chain_len']} tokens"
        + (f", +{len(step['chain_new'])} new: {new}" if step["chain_new"] else ""),
        "",
        f"[1] draft tree: {len(nodes) - 1} nodes"
        + (f", {n_reused} reused from the previous step (*)" if n_reused else ""),
        f"    …{tail}",
    ]

    def by_score(ids):
        return sorted(ids, key=lambda i: (-nodes[i]["logprob"], i))

    children = defaultdict(list)
    for idx, n in nodes.items():
        if idx != root:
            children[n["parent"]].append(idx)
    children = {k: by_score(v) for k, v in children.items()}

    def draft_label(idx):
        n = nodes[idx]
        mark = " *" if n["reused"] else ""
        score = "" if idx == root else f"  {n['logprob']:.2f}"
        return f"#{idx} {tok(n['token'])}{score}{mark}"

    out += ["    " + line for line in ascii_tree(root, children, draft_label)]

    # stage 2: the draft tree plus the scratch forest hanging off it
    forest = {n["idx"]: n for n in sag["forest"]}
    bets = {b["root"]: b for b in sag["bets"] if b["root"] is not None}
    fan = {c["node"]: c["fan"] for c in sag["candidates"]}
    n_exits = sum(1 for f in fan.values() if f > 0)
    out += [
        "",
        f"[2] saguaro fan-out: {len(sag['bets'])} guesses on {n_exits} exit nodes, "
        f"{len(forest)} scratch tokens",
    ]
    ext_children = defaultdict(list)
    for idx, n in forest.items():
        ext_children[n["parent"]].append(idx)
    merged = {
        k: children.get(k, [])
        + sorted(
            ext_children.get(k, []),
            key=lambda i: (i not in bets, -forest[i]["logprob"], i),
        )
        for k in set(children) | set(ext_children)
    }

    def fan_label(idx):
        if idx in forest:
            n = forest[idx]
            if idx in bets:
                return f"★ {tok(n['token'])}  guess, log p {bets[idx]['logprob']:.2f}"
            return f"+ {tok(n['token'])}"
        label = draft_label(idx).removesuffix(" *")
        return label + (f"  [fan {fan[idx]}]" if idx in fan else "")

    out += ["    " + line for line in ascii_tree(root, merged, fan_label)]

    # stage 3: the verified chain
    chain = " → ".join(tok(t) for t in v["accepted_tokens"])
    kind = step["_outcome"]["kind"]
    result = {
        "hit": f"cache hit, {v['n_reused']} tokens reused",
        "partial": "guess matched, but its branch has no frontier: not reused",
        "miss_token": "miss: exit covered, bonus not guessed ("
        + ", ".join(tok(t) for t in step["_outcome"]["guesses_at_exit"])
        + ")",
        "miss_exit": "miss: no guess at this exit node",
    }[kind]
    out += [
        "",
        f"[3] verified: {len(v['accepted'])} accepted + bonus · exit #{v['exit']} · {result}",
        f"    …{tail} {tok(nodes[root]['token'])}"
        + (f" → {chain}" if chain else "")
        + f" ⇒ bonus {tok(v['bonus'])}",
        "",
    ]
    return out


def render_text(requests: list[dict], vocab: dict[int, str], steps=None) -> str:
    out = []
    for req in requests:
        for step in req["steps"]:
            if steps is None or step["step_idx"] in steps:
                out += text_step(step, vocab)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def render_html(run: dict, requests: list[dict], vocab: dict[int, str], name: str) -> str:
    data = {
        "name": name,
        "run": run,
        "requests": requests,
        "vocab": {str(k): v for k, v in vocab.items()},
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return (
        PAGE.replace("__TITLE__", html.escape(name))
        .replace("__CSS__", CSS)
        .replace("__JS__", JS)
        .replace("__DATA__", payload)
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
.wrap { max-width: 1180px; margin: 0 auto; padding-inline: 20px; padding-block: 36px 72px; }
h1, h2 { font-family: Fraunces, Georgia, serif; font-weight: 500; margin: 0; text-wrap: balance; }
h1 { font-size: 30px; line-height: 1.2; }
h2 { font-size: 22px; }
p { max-width: 72ch; margin: 0; }
code, .mono { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.92em; }
.eyebrow { font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.lede { color: var(--muted); }
.meta {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 8px 28px; margin: 22px 0 0; padding: 14px 0; border-block: 1px solid var(--line);
}
.meta div { display: flex; flex-direction: column; }
.meta dt { font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
.meta dd { margin: 0; font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 13px; overflow-wrap: anywhere; }
.controls {
  position: sticky; top: 0; z-index: 2; background: var(--bg);
  display: flex; flex-wrap: wrap; align-items: center; gap: 10px 14px;
  padding-block: 14px; border-bottom: 1px solid var(--line);
}
.controls label { display: inline-flex; align-items: center; gap: 8px; color: var(--muted); font-size: 13px; }
select, button {
  font: inherit; font-size: 14px; color: var(--ink); background: var(--surface);
  border: 1px solid var(--node-line); border-radius: 4px; padding: 4px 10px;
}
button { cursor: pointer; min-width: 36px; }
input[type=range] { flex: 1 1 200px; min-width: 0; accent-color: var(--ext); }
#steplabel { font-size: 13px; white-space: nowrap; }
:focus-visible { outline: 2px solid var(--ext); outline-offset: 2px; }
.strip { display: flex; flex-wrap: wrap; gap: 3px; margin-top: 12px; }
.cell {
  width: 24px; height: 24px; border-radius: 3px; border: 1px solid var(--node-line);
  background: var(--surface); color: var(--muted); font-size: 11px; line-height: 22px;
  text-align: center; cursor: pointer; padding: 0; min-width: 0;
  font-variant-numeric: tabular-nums;
}
.cell.hit { background: var(--accept-tint); border-color: var(--accept); color: var(--accept); }
.cell.partial { background: var(--bonus-tint); border-color: var(--bonus); color: var(--bonus); }
.cell.current { outline: 2px solid var(--ink); outline-offset: 1px; }
.strip-note { font-size: 12px; color: var(--muted); margin-top: 6px; }
.summary { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 14px; margin-top: 18px; }
.pill {
  display: inline-block; font-size: 12px; font-weight: 500; padding: 2px 9px;
  border-radius: 999px; white-space: nowrap; border: 1px solid;
}
.pill.hit { color: var(--accept); background: var(--accept-tint); border-color: var(--accept); }
.pill.partial { color: var(--bonus); background: var(--bonus-tint); border-color: var(--bonus); }
.pill.miss { color: var(--reject); background: var(--reject-tint); border-color: var(--reject); }
section.stage { margin-top: 36px; display: flex; flex-direction: column; gap: 10px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 20px; font-size: 13px; color: var(--muted); }
.legend span { display: inline-flex; align-items: center; gap: 7px; }
.sw { width: 22px; height: 13px; border-radius: 3px; border: 1px solid var(--node-line); background: var(--surface); }
.sw.chain { background: var(--bg); border-color: var(--line); }
.sw.reused { background: var(--ext); border-color: var(--ext); }
.sw.guess { background: var(--ext-tint); border: 1.5px solid var(--ext); }
.sw.ext { background: var(--ext-tint); border: 1px dashed var(--ext); }
.sw.hitroot { background: var(--ext-tint); border: 2.5px solid var(--bonus); }
.sw.accepted { background: var(--accept-tint); border: 1.5px solid var(--accept); }
.sw.bonus { background: var(--bonus-tint); border: 1.5px solid var(--bonus); }
.sw.badge { width: 14px; height: 14px; border-radius: 50%; background: var(--ext); border: none; }
.scroll { overflow: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 4px; }
.scroll.tall { max-height: 72vh; }
.empty { padding: 14px 16px; color: var(--muted); font-size: 14px; }
details { margin-top: 36px; }
summary { cursor: pointer; color: var(--muted); }
pre {
  white-space: pre-wrap; background: var(--surface); border: 1px solid var(--line);
  padding: 12px 14px; border-radius: 4px; font-size: 13px; max-height: 320px; overflow: auto;
}
svg.tree { display: block; max-width: none; }
svg.tree text { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px; }
svg.tree .axis text { fill: var(--muted); font-family: "IBM Plex Sans", system-ui, sans-serif; font-size: 10.5px; letter-spacing: 0.04em; }
.e { fill: none; stroke: var(--node-line); stroke-width: 1; }
.e.chain { stroke: var(--line); }
.e.reused { stroke: var(--ext); stroke-width: 2; }
.e.ext { stroke: var(--ext); stroke-dasharray: 3 2; }
.e.accepted { stroke: var(--accept); stroke-width: 2; }
.e.bonus { stroke: var(--bonus); stroke-width: 2; }
.n rect { fill: var(--surface); stroke: var(--node-line); stroke-width: 1; }
.n text { fill: var(--ink); }
.n.chain rect { fill: var(--bg); stroke: var(--line); }
.n.chain text { fill: var(--muted); }
.n.root rect { fill: var(--bg); stroke: var(--ink); stroke-width: 1.5; }
.n.reused rect { fill: var(--ext); stroke: var(--ext); }
.n.reused text { fill: var(--on-ext); }
.n.ext rect { fill: var(--ext-tint); stroke: var(--ext); stroke-dasharray: 3 2; }
.n.ext.guess rect { stroke-dasharray: none; stroke-width: 1.5; }
.n.hitroot rect { stroke: var(--bonus); stroke-width: 3; stroke-dasharray: none; }
.n.accepted rect { fill: var(--accept-tint); stroke: var(--accept); stroke-width: 1.5; }
.n.bonus rect { fill: var(--bonus-tint); stroke: var(--bonus); stroke-width: 1.5; }
.badge circle { fill: var(--ext); }
.badge text { fill: var(--on-ext); font-size: 9px !important; font-family: "IBM Plex Sans", system-ui, sans-serif !important; font-weight: 600; }
.badge.zero circle { r: 3.5; fill: var(--surface); stroke: var(--node-line); }
"""

JS = r"""
const DATA = JSON.parse(document.getElementById("trace-data").textContent);
const V = DATA.vocab;
const $ = (id) => document.getElementById(id);

const tok = (id) => (id in V ? V[id] : `<${id}>`);
const show = (s) => s.replace(/ /g, "␣").replace(/\n/g, "↵").replace(/\t/g, "⇥") || "∅";
const LABEL_CHARS = 11;
const label = (s) => { s = show(s); return s.length <= LABEL_CHARS ? s : s.slice(0, LABEL_CHARS - 1) + "…"; };
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const q = (id) => `<code>${esc(show(tok(id)))}</code>`;

function cmp(a, b) {
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const x = a[i] ?? -Infinity, y = b[i] ?? -Infinity;
    if (x < y) return -1;
    if (x > y) return 1;
  }
  return 0;
}

// Horizontal tidy tree: depth left to right, one leaf per row.
// nodes: Map id -> {parent, text, title, cls, edge, order, badge}
const COL_W = 118, NODE_W = 88, NODE_H = 20, ROW_H = 28, PAD_X = 16, PAD_TOP = 30, PAD_BOTTOM = 14;

function renderTree(nodes, root, aria, axisOrigin = 0) {
  const children = new Map();
  for (const [id, n] of nodes) {
    if (id === root) continue;
    if (!children.has(n.parent)) children.set(n.parent, []);
    children.get(n.parent).push(id);
  }
  for (const kids of children.values()) kids.sort((a, b) => cmp(nodes.get(a).order, nodes.get(b).order));

  const pos = new Map();
  let row = 0;
  const place = (id, depth) => {
    const kids = children.get(id) || [];
    if (!kids.length) { pos.set(id, [depth, row++]); return; }
    for (const k of kids) place(k, depth + 1);
    pos.set(id, [depth, (pos.get(kids[0])[1] + pos.get(kids[kids.length - 1])[1]) / 2]);
  };
  place(root, 0);

  let maxD = 0;
  for (const [d] of pos.values()) maxD = Math.max(maxD, d);
  const width = PAD_X * 2 + maxD * COL_W + NODE_W + 8;
  const height = PAD_TOP + (row - 1) * ROW_H + NODE_H + PAD_BOTTOM;
  const xy = (id) => { const [d, r] = pos.get(id); return [PAD_X + d * COL_W, PAD_TOP + r * ROW_H]; };

  const out = [`<svg class="tree" role="img" aria-label="${esc(aria)}" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}">`, `<g class="axis">`];
  for (let d = 0; d <= maxD; d++) {
    const rel = d - axisOrigin;
    if (rel < 0) continue;
    out.push(`<text x="${PAD_X + d * COL_W + NODE_W / 2}" y="14" text-anchor="middle">${rel === 0 ? "root" : "+" + rel}</text>`);
  }
  out.push("</g>");
  for (const id of pos.keys()) {
    if (id === root) continue;
    const [px, py] = xy(nodes.get(id).parent), [cx, cy] = xy(id);
    const x1 = px + NODE_W, y1 = py + NODE_H / 2, x2 = cx, y2 = cy + NODE_H / 2, xm = (x1 + x2) / 2;
    out.push(`<path class="e ${nodes.get(id).edge || ""}" d="M${x1} ${y1} C${xm} ${y1} ${xm} ${y2} ${x2} ${y2}"/>`);
  }
  for (const id of pos.keys()) {
    const n = nodes.get(id), [x, y] = xy(id);
    out.push(`<g class="n ${n.cls || ""}"><title>${esc(n.title || "")}</title>`,
      `<rect x="${x}" y="${y}" width="${NODE_W}" height="${NODE_H}" rx="3"/>`,
      `<text x="${x + 6}" y="${y + 14}">${esc(label(n.text))}</text></g>`);
    if (n.badge) {
      const [text, cls] = n.badge;
      out.push(`<g class="badge ${cls}"><circle cx="${x + NODE_W}" cy="${y}" r="7"/><text x="${x + NODE_W}" y="${y + 3}" text-anchor="middle">${esc(text)}</text></g>`);
    }
  }
  out.push("</svg>");
  return out.join("");
}

const nodeTitle = (n, extra) =>
  [`#${n.idx}  ${JSON.stringify(show(tok(n.token)))}  (id ${n.token})`, `cumulative score ${n.logprob.toFixed(2)}`, extra]
    .filter(Boolean).join("\n");

// Committed tokens before the root, drawn as a faint chain.
function chainTail(step, nodes) {
  let prev = null;
  step._tail.forEach((t, i) => {
    const id = "c" + i;
    nodes.set(id, { parent: prev, text: tok(t), title: `committed token ${JSON.stringify(show(tok(t)))}`, cls: "chain", edge: "chain", order: [0] });
    prev = id;
  });
  return { first: step._tail.length ? "c0" : null, prev, origin: step._tail.length };
}

function rootNode(step, parent) {
  const n = step.draft.nodes.find((m) => m.idx === step.draft.root);
  return { parent, text: tok(n.token), title: nodeTitle(n, "last committed token: root of the draft tree"), cls: "root", edge: "chain", order: [0] };
}

function stage1(step) {
  const nodes = new Map();
  const { first, prev, origin } = chainTail(step, nodes);
  const root = String(step.draft.root);
  nodes.set(root, rootNode(step, prev));
  for (const n of step.draft.nodes) {
    if (n.idx === step.draft.root) continue;
    nodes.set(String(n.idx), {
      parent: String(n.parent), text: tok(n.token),
      title: nodeTitle(n, n.reused ? "carried over from the previous step's spliced Saguaro branch" : "drafted this step"),
      cls: n.reused ? "reused" : "", edge: n.reused ? "reused" : "", order: [0, -n.logprob, n.idx],
    });
  }
  return renderTree(nodes, first ?? root, "draft tree sent for verification", origin);
}

function stage2(step) {
  const sag = step.saguaro;
  const nodes = new Map();
  const root = String(step.draft.root);
  for (const n of step.draft.nodes) {
    const isRoot = n.idx === step.draft.root;
    nodes.set(String(n.idx), isRoot ? rootNode(step, null) : {
      parent: String(n.parent), text: tok(n.token), title: nodeTitle(n, ""), order: [0, -n.logprob, n.idx],
    });
  }
  const bets = new Map();
  sag.bets.forEach((b, rank) => { if (b.root !== null) bets.set(b.root, { ...b, rank }); });
  for (const n of sag.forest) {
    const bet = bets.get(n.idx);
    let title = nodeTitle(n, "scratch continuation");
    let cls = "ext";
    if (bet) {
      title = nodeTitle(n, `guessed bonus after #${bet.exit}, draft log p ${bet.logprob.toFixed(2)}` +
        (bet.has_frontier ? "" : "\nbranch has no open frontier: unusable on a hit"));
      cls = "ext guess";
      if (n.idx === step._outcome.hit_root) {
        cls += " hitroot";
        title += "\nmatched the verifier's exit node and bonus token";
      }
    }
    nodes.set(String(n.idx), { parent: String(n.parent), text: tok(n.token), title, cls, edge: "ext", order: bet ? [1, bet.rank] : [1, 1e9, -n.logprob, n.idx] });
  }
  for (const c of sag.candidates) {
    const node = nodes.get(String(c.node));
    if (!node) continue;
    node.badge = c.fan > 0 ? [String(c.fan), ""] : ["", "zero"];
    node.title += `\ncandidate exit node: ${c.fan} guess${c.fan === 1 ? "" : "es"}` +
      (c.excluded !== null ? `, skips ${JSON.stringify(show(tok(c.excluded)))} (already a draft child)` : "");
  }
  return renderTree(nodes, root, "draft tree with the Saguaro fan-out");
}

function stage3(step) {
  const v = step.verify;
  const nodes = new Map();
  const { first, prev, origin } = chainTail(step, nodes);
  const root = String(step.draft.root);
  nodes.set(root, rootNode(step, prev));
  let parent = root;
  v.accepted.forEach((idx, i) => {
    const id = "a" + i;
    nodes.set(id, { parent, text: tok(v.accepted_tokens[i]), title: `accepted draft token (slot #${idx})`, cls: "accepted", edge: "accepted", order: [0] });
    parent = id;
  });
  nodes.set("bonus", {
    parent, text: tok(v.bonus), cls: "bonus" + (v.spliced ? " hitroot" : ""), edge: "bonus", order: [0],
    title: `bonus token sampled by the target (id ${v.bonus})` + (v.spliced ? "\nSaguaro guessed it: its branch is spliced into the next step" : ""),
  });
  return renderTree(nodes, first ?? root, "verified chain", origin);
}

function outcomeText(step) {
  const v = step.verify, o = step._outcome;
  const exit = v.exit === step.draft.root ? "the root (nothing accepted)" : `#${v.exit}`;
  switch (o.kind) {
    case "hit": return ["hit", `cache hit · ${v.n_reused} reused`, `Saguaro guessed ${q(v.bonus)} after ${exit}. Its ${v.n_reused}-token branch is spliced in, so the next step's draft tree starts from it (blue in stage 1 of step ${step.step_idx + 1}).`];
    case "partial": return ["partial", "hit · not reused", `A guess matched (${q(v.bonus)} after ${exit}), but its branch has no open frontier, so the client falls back to a plain reorder.`];
    case "miss_token": return ["miss", "miss · wrong token", `The exit ${exit} had guesses (${o.guesses_at_exit.map(q).join(", ")}), but the target sampled ${q(v.bonus)}.`];
    default: return ["miss", "miss · wrong exit", `No guess sat at the exit ${exit}. The target sampled ${q(v.bonus)}.`];
  }
}

// ---- page state -----------------------------------------------------------
let reqPos = 0, stepPos = 0;

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  const r = DATA.requests.findIndex((x) => String(x.req_idx) === h.get("req"));
  if (r >= 0) reqPos = r;
  const steps = DATA.requests[reqPos]?.steps || [];
  const s = steps.findIndex((x) => String(x.step_idx) === h.get("step"));
  stepPos = s >= 0 ? s : 0;
}

function go(r, s) {
  reqPos = r;
  const steps = DATA.requests[reqPos].steps;
  stepPos = Math.max(0, Math.min(s, steps.length - 1));
  history.replaceState(null, "", `#req=${DATA.requests[reqPos].req_idx}&step=${steps[stepPos]?.step_idx ?? 0}`);
  draw();
}

function drawRequest() {
  const req = DATA.requests[reqPos];
  $("req").value = String(reqPos);
  $("step").max = String(Math.max(0, req.steps.length - 1));
  $("strip").innerHTML = req.steps.map((s, i) => {
    const kind = s._outcome.kind === "hit" ? "hit" : s._outcome.kind === "partial" ? "partial" : "miss";
    const n = s.verify.accepted.length + 1;
    return `<button class="cell ${kind}" data-i="${i}" title="step ${s.step_idx}: ${n} tokens, ${esc(outcomeText(s)[1])}" aria-label="step ${s.step_idx}">${n}</button>`;
  }).join("");
  $("gen").textContent = req._generated_text ?? req._generated.map(tok).join("");
  const hits = req.steps.filter((s) => s._outcome.kind === "hit").length;
  $("reqnote").textContent = `${req.steps.length} steps · ${req._generated.length} tokens generated · ${hits} cache hits`;
}

let drawnReq = -1;
function draw() {
  const req = DATA.requests[reqPos];
  if (drawnReq !== reqPos) { drawRequest(); drawnReq = reqPos; }
  document.querySelectorAll(".cell").forEach((c) => c.classList.toggle("current", Number(c.dataset.i) === stepPos));
  const step = req.steps[stepPos];
  if (!step) { for (const id of ["s1", "s2", "s3"]) $(id).innerHTML = `<p class="empty">No steps recorded.</p>`; return; }
  $("step").value = String(stepPos);
  $("steplabel").textContent = `step ${step.step_idx} of ${req.steps[req.steps.length - 1].step_idx}` + (step.prefill ? " · prefill" : "");

  const [pill, pillText, sentence] = outcomeText(step);
  const v = step.verify;
  $("summary").innerHTML = `<span class="pill ${pill}">${esc(pillText)}</span><span>${v.accepted.length} accepted + bonus = <b>${v.accepted.length + 1}</b> tokens</span><span class="lede">chain ${step.chain_len} → ${step.chain_len + v.accepted.length + 1} tokens</span>`;

  const nReused = step.draft.nodes.filter((n) => n.reused).length;
  $("s1lede").innerHTML = `${step.draft.nodes.length - 1} draft tokens grown from the last committed token ${q(step.draft.nodes.find((n) => n.idx === step.draft.root).token)}` +
    (nReused ? `, ${nReused} of them carried over from the previous step's Saguaro branch (blue).` : ".") +
    (step.chain_new.length ? ` New in the chain since the last step: ${step.chain_new.map(q).join(" ")}.` : "");
  $("s1").innerHTML = stage1(step);

  const sag = step.saguaro;
  const funded = sag.candidates.filter((c) => c.fan > 0).length;
  $("s2lede").innerHTML = sag.bets.length
    ? `${sag.bets.length} guessed bonus tokens on ${funded} of ${sag.candidates.length} candidate exit nodes, ${sag.forest.length} scratch tokens drafted while the target verified.`
    : "Saguaro made no guesses this step.";
  $("s2").innerHTML = sag.bets.length ? stage2(step) : `<p class="empty">Nothing to show.</p>`;

  $("s3lede").innerHTML = sentence;
  $("s3").innerHTML = stage3(step);
}

function init() {
  const run = DATA.run;
  const meta = [
    ["draft model", run.draft_model],
    ["draft tree", `beams ${run.max_n_beams} · depth ${run.max_beam_len} · width ${run.max_branch_width} · budget ${run.max_budget}`],
    ["saguaro", `B ${run.saguaro_budget} · branch ${run.saguaro_branch_len} · ${run.saguaro_fan_out} · a_p ${run.saguaro_acceptance_rate} · linear ${run.saguaro_linear}`],
    ["requests", String(DATA.requests.length)],
  ];
  $("meta").innerHTML = meta.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v ?? "?")}</dd></div>`).join("");
  $("req").innerHTML = DATA.requests.map((r, i) => `<option value="${i}">req ${r.req_idx} (${r.steps.length} steps)</option>`).join("");
  $("req").onchange = (e) => go(Number(e.target.value), 0);
  $("step").oninput = (e) => go(reqPos, Number(e.target.value));
  $("prev").onclick = () => go(reqPos, stepPos - 1);
  $("next").onclick = () => go(reqPos, stepPos + 1);
  $("strip").onclick = (e) => { const c = e.target.closest(".cell"); if (c) go(reqPos, Number(c.dataset.i)); };
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "ArrowLeft") { e.preventDefault(); go(reqPos, stepPos - 1); }
    if (e.key === "ArrowRight") { e.preventDefault(); go(reqPos, stepPos + 1); }
  });
  if (!DATA.requests.length) { $("summary").textContent = "The trace has no requests."; return; }
  readHash();
  go(reqPos, stepPos);
}
init();
"""

PAGE = """<title>Saguaro Tree Trace</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>__CSS__</style>
<div class="wrap">
  <span class="eyebrow">SpecEdge · Saguaro tree trace</span>
  <h1>__TITLE__</h1>
  <dl class="meta" id="meta"></dl>

  <nav class="controls" aria-label="step navigation">
    <label>Request <select id="req"></select></label>
    <button id="prev" aria-label="previous step">←</button>
    <input type="range" id="step" min="0" value="0" aria-label="step">
    <button id="next" aria-label="next step">→</button>
    <span id="steplabel" class="mono"></span>
  </nav>
  <div class="strip" id="strip"></div>
  <p class="strip-note"><span id="reqnote"></span> · each cell is one step: tokens gained, green on a cache hit. ← → keys step through.</p>
  <div class="summary" id="summary"></div>

  <section class="stage">
    <h2>1 · Chain and draft tree</h2>
    <p class="lede" id="s1lede"></p>
    <div class="legend"><span><i class="sw chain"></i>committed chain</span><span><i class="sw"></i>drafted this step</span><span><i class="sw reused"></i>reused from last step's Saguaro branch</span></div>
    <div class="scroll tall" id="s1"></div>
  </section>

  <section class="stage">
    <h2>2 · Saguaro fan-out</h2>
    <p class="lede" id="s2lede"></p>
    <div class="legend"><span><i class="sw badge"></i>guesses at a candidate exit node</span><span><i class="sw guess"></i>guessed bonus token</span><span><i class="sw ext"></i>scratch continuation</span><span><i class="sw hitroot"></i>guess the verifier confirmed</span></div>
    <div class="scroll tall" id="s2"></div>
  </section>

  <section class="stage">
    <h2>3 · Verified chain</h2>
    <p class="lede" id="s3lede"></p>
    <div class="legend"><span><i class="sw accepted"></i>accepted draft token</span><span><i class="sw bonus"></i>target's bonus token</span></div>
    <div class="scroll" id="s3"></div>
  </section>

  <details><summary>Generated text for this request</summary><pre id="gen"></pre></details>
</div>
<script id="trace-data" type="application/json">__DATA__</script>
<script>__JS__</script>
"""


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_steps(spec: str | None):
    if spec is None:
        return None
    steps: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        steps.update(range(int(lo), int(hi or lo) + 1))
    return steps


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("trace", type=Path, help="<process_name>.tree_trace.jsonl")
    p.add_argument("--tokenizer", default=None, help="default: draft_model from the trace")
    p.add_argument("--out", type=Path, default=None, help="default: next to the trace, .html")
    p.add_argument("--req", type=int, nargs="*", default=None, help="only these req_idx")
    p.add_argument("--text", action="store_true", help="print text trees instead of HTML")
    p.add_argument("--steps", default=None, help="text mode: step range, e.g. 0-4,9")
    return p.parse_args()


def main():
    args = parse_args()
    run, requests = load_trace(args.trace)
    if args.req is not None:
        requests = [r for r in requests if r["req_idx"] in args.req]
    annotate(requests)

    name = args.tokenizer or run.get("draft_model")
    if not name:
        raise SystemExit("trace has no draft_model; pass --tokenizer")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    vocab = {i: tokenizer.decode([i]) for i in sorted(token_ids(requests))}
    for req in requests:
        req["_generated_text"] = tokenizer.decode(req["_generated"])

    if args.text:
        print(render_text(requests, vocab, parse_steps(args.steps)))
        return

    out = args.out or args.trace.with_name(
        args.trace.name.removesuffix(".jsonl") + ".html"
    )
    stem = args.trace.name.removesuffix(".tree_trace.jsonl")
    out.write_text(render_html(run, requests, vocab, f"{stem} · {len(requests)} requests"))
    n_steps = sum(len(r["steps"]) for r in requests)
    print(f"wrote {out} ({len(requests)} requests, {n_steps} steps)")


if __name__ == "__main__":
    main()
