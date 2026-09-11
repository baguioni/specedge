"""TreeTracer records on a CPU Tree (no model), plus predict_outcome_details."""

import json
import logging

import torch

from specedge.client.overlap import OverlapResult, OverlapStrategy
from specedge.client.proactive import ProactiveStrategy, SpecExecProactiveDraft
from specedge.client.reorder import (
    append_bonus_token,
    label_forest_roots,
    reorder_to_verified_path,
    splice_scratch_branch,
)
from specedge.client.saguaro.outcomes import (
    OutcomePrediction,
    predict_outcome_details,
    predict_outcomes,
)
from specedge.client.saguaro.strategy import build_speculation_cache
from specedge.client.saguaro.trace import TreeTracer, token_paths
from specedge.tree import Tree

CPU = torch.device("cpu")
F32 = torch.float32


def _t(values, dtype=torch.long):
    return torch.tensor(values, dtype=dtype, device=CPU)


def _add(tree, tokens, parents, logprobs, status=None):
    tree.add(
        token_ids=_t(tokens),
        token_positions=tree.positions[_t(parents)] + 1,
        parent_indices=_t(parents),
        logprobs=_t(logprobs, F32),
        token_status=status,
    )


class _FakeStrategy:
    name = "saguaro"

    def __init__(self):
        self.prediction = OutcomePrediction()
        self.forest = None
        self.cache = None


def _plant_forest(tree, strategy, exit_nodes, bonus_tokens, children):
    """Scratch forest past ``tree.end``: one root per outcome, plus
    ``children`` as (root position in the outcome list, token) pairs."""
    end, prefix_len = tree.end, tree.prefix_len
    tree.prefix_len = tree.end
    start = int(tree.end)
    roots = []
    for exit_idx, bonus in zip(exit_nodes, bonus_tokens, strict=True):
        _add(tree, [bonus], [exit_idx], [0.0], tree.POST_CANDIDATE)
        roots.append(int(tree.end) - 1)
    for root_pos, token in children:
        _add(tree, [token], [roots[root_pos]], [-0.3], tree.POST_CANDIDATE)
    forest = (start, int(tree.end), roots, label_forest_roots(tree, start, int(tree.end), roots))
    tree.end, tree.prefix_len = end, prefix_len

    strategy.prediction = OutcomePrediction(
        candidates=sorted(set(exit_nodes)),
        fan=[exit_nodes.count(n) for n in sorted(set(exit_nodes))],
        excluded=[None] * len(set(exit_nodes)),
        exit_nodes=list(exit_nodes),
        bonus_tokens=list(bonus_tokens),
        bonus_logprobs=[-0.25] * len(exit_nodes),
    )
    strategy.forest = forest
    strategy.cache = build_speculation_cache(tree, forest, exit_nodes, bonus_tokens, CPU)
    return roots


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_token_paths_survive_slot_permutation():
    nodes = [
        {"idx": 5, "token": 9, "parent": 3},  # child listed before its parent
        {"idx": 3, "token": 7, "parent": 2},
        {"idx": 4, "token": 8, "parent": 2},
    ]
    paths = token_paths(nodes, root=2)
    assert paths == {2: (), 3: (7,), 4: (8,), 5: (7, 9)}


def test_three_stages_and_reuse_across_a_splice(tmp_path):
    path = tmp_path / "client_0.tree_trace.jsonl"
    tree = Tree(_t([[100, 101, 102]]), CPU, F32, max_len=64)
    strategy = _FakeStrategy()
    tracer = TreeTracer(tree, strategy, path, client_idx=0, run_info={"k": 1})

    # ---- step 0: draft tree off root #2 ------------------------------------
    tracer.begin_request(req_idx=7)
    tracer.begin_step(req_idx=7, step_idx=0, prefill=True)
    _add(tree, [10, 11], [2, 2], [-0.1, -0.9])  # #3, #4
    _add(tree, [12], [3], [-0.4])  # #5
    tree.status[3:6] = tree.PROCESSED
    tracer.log_draft()

    # fan-out: two guesses after #5, one after #4
    roots = _plant_forest(
        tree, strategy, [5, 5, 4], [20, 21, 22], children=[(0, 30), (0, 31), (2, 32)]
    )
    tracer.log_speculation()

    # verifier accepts #3 -> #5 and samples bonus 20: hits the first guess
    seq_mask = torch.zeros(tree.end, dtype=torch.bool)
    seq_mask[[0, 1, 2, 3, 5]] = True
    hit = strategy.cache.get(next(iter(strategy.cache.entries)))
    splice_scratch_branch(tree, _NoEngine(), CPU, F32, seq_mask, hit.node_indices)
    tracer.end_step(
        accepted_idx=_t([3, 5]),
        accepted_ids=_t([10, 12]),
        exit_idx=5,
        bonus=20,
        result=OverlapResult(spliced=True, cache_hit=True, n_reused=hit.n_tokens),
    )

    # ---- step 1: spliced branch + one fresh node ---------------------------
    tracer.begin_step(req_idx=7, step_idx=1, prefill=False)
    root = int(tree.prefix_len) - 1
    fresh_parent = next(
        i for i in range(root + 1, tree.end) if tree.tokens[i].item() == 30
    )
    _add(tree, [40], [fresh_parent], [-0.8])
    tracer.log_draft()

    run, request, step0 = _records(path)
    assert run == {"type": "run", "k": 1}
    assert request == {"type": "request", "client_idx": 0, "req_idx": 7, "prompt": [100, 101, 102]}

    # stage 1
    assert step0["chain_len"] == 3 and step0["chain_new"] == []
    draft = step0["draft"]
    assert draft["root"] == 2
    assert [(n["idx"], n["token"], n["parent"]) for n in draft["nodes"]] == [
        (2, 102, 1), (3, 10, 2), (4, 11, 2), (5, 12, 3)
    ]
    assert not any(n["reused"] for n in draft["nodes"])

    # stage 2
    sag = step0["overlap"]
    assert sag["strategy"] == "saguaro"
    assert sag["candidates"] == [
        {"node": 4, "fan": 1, "excluded": None},
        {"node": 5, "fan": 2, "excluded": None},
    ]
    assert [(b["exit"], b["bonus"], b["root"]) for b in sag["bets"]] == [
        (5, 20, roots[0]), (5, 21, roots[1]), (4, 22, roots[2])
    ]
    forest = {n["idx"]: n for n in sag["forest"]}
    assert set(sag["bets"][0]["nodes"]) == {roots[0]} | {
        i for i, n in forest.items() if n["parent"] == roots[0]
    }
    assert {forest[i]["token"] for i in sag["bets"][0]["nodes"]} == {20, 30, 31}

    # stage 3
    assert step0["verify"] == {
        "accepted": [3, 5],
        "accepted_tokens": [10, 12],
        "exit": 5,
        "bonus": 20,
        "cache_hit": True,
        "spliced": True,
        "n_reused": 3,
    }

    # step 1 (not yet written): chain grew by accepted + bonus; the spliced
    # branch is marked reused, the freshly drafted node is not.
    rec = tracer._record
    assert rec["chain_new"] == [10, 12, 20]
    reused = {n["token"]: n["reused"] for n in rec["draft"]["nodes"]}
    assert reused == {20: False, 30: True, 31: True, 40: False}  # root is never reused


def test_empty_prediction_logs_no_bets(tmp_path):
    tree = Tree(_t([[1, 2]]), CPU, F32, max_len=16)
    tracer = TreeTracer(tree, _FakeStrategy(), tmp_path / "t.jsonl", 0, {})
    tracer.begin_step(0, 0, True)
    tracer.log_draft()
    tracer.log_speculation()
    assert tracer._record["overlap"] == {
        "strategy": "saguaro", "candidates": [], "bets": [], "forest": []
    }


def test_renderer_reads_tracer_output(tmp_path):
    from script.render_tree_trace import (
        annotate,
        load_trace,
        render_html,
        render_text,
        token_ids,
    )

    path = tmp_path / "client_0.tree_trace.jsonl"
    tree = Tree(_t([[100, 101, 102]]), CPU, F32, max_len=64)
    strategy = _FakeStrategy()
    tracer = TreeTracer(tree, strategy, path, 0, {"draft_model": "x"})
    tracer.begin_request(3)
    tracer.begin_step(3, 0, True)
    _add(tree, [10, 11], [2, 2], [-0.1, -0.9])
    tree.status[3:5] = tree.PROCESSED
    tracer.log_draft()
    _plant_forest(tree, strategy, [3], [20], children=[(0, 30)])
    tracer.log_speculation()
    tracer.end_step(
        accepted_idx=_t([3]),
        accepted_ids=_t([10]),
        exit_idx=3,
        bonus=21,
        result=OverlapResult(spliced=False, cache_hit=False, n_reused=0),
    )

    run, requests = load_trace(path)
    annotate(requests)
    step = requests[0]["steps"][0]
    assert step["_tail"] == [100, 101]
    assert step["_outcome"] == {
        "kind": "miss_token", "hit_root": None, "guesses_at_exit": [20]
    }
    assert requests[0]["_generated"] == [10, 21]

    vocab = {i: f"t{i}" for i in token_ids(requests)}
    text = render_text(requests, vocab)
    assert "★ 't20'" in text and "[fan 1]" in text and "⇒ bonus 't21'" in text
    page = render_html(run, requests, vocab, "client_0")
    assert page.startswith("<title>SpecEdge Tree Trace</title>")
    assert "__DATA__" not in page and "__JS__" not in page


class _NoEngine:
    def gather(self, src, dst):
        pass


class _FixedLogitsEngine:
    def __init__(self, logits):
        self._logits = logits

    def forward(self, input_ids, **_):
        assert input_ids.size(-1) == self._logits.size(0)
        return self._logits.unsqueeze(0)


def test_predict_outcome_details_matches_predict_outcomes():
    tree = Tree(_t([[0, 1, 2]]), CPU, F32, max_len=16)
    _add(tree, [5, 6], [2, 2], [-0.1, -0.5])  # leaves #3 (better), #4

    logits = torch.full((2, 10), -10.0)
    logits[0, 7], logits[0, 8] = 5.0, 4.0  # after #3
    logits[1, 9] = 5.0  # after #4
    engine = _FixedLogitsEngine(logits)
    kwargs = {"budget": 3, "max_n_beams": 8, "acceptance_rate": 0.5, "fan_out": "uniform"}

    details = predict_outcome_details(tree, engine, **kwargs)
    assert details.candidates == [3, 4]
    assert details.fan == [2, 1]
    assert details.excluded == [None, None]
    assert list(zip(details.exit_nodes, details.bonus_tokens, strict=True)) == [
        (3, 7), (3, 8), (4, 9)
    ]
    logp = torch.log_softmax(logits, dim=-1)
    expected = [logp[0, 7].item(), logp[0, 8].item(), logp[1, 9].item()]
    assert details.bonus_logprobs == expected

    assert predict_outcomes(tree, engine, **kwargs) == (
        details.exit_nodes, details.bonus_tokens
    )


class _ScriptedEngine:
    """Draft engine stub: forward ``n`` returns ``calls[n]``, one
    ``{token: logit}`` dict per input row (every other logit is -10)."""

    VOCAB = 1100  # proactive drafting takes the top 1024 tokens per leaf

    def __init__(self, calls):
        self._calls = list(calls)
        self.logits = []

    def forward(self, input_ids, **_):
        rows = self._calls.pop(0)
        assert input_ids.size(-1) == len(rows)
        logits = torch.full((1, len(rows), self.VOCAB), -10.0)
        for i, row in enumerate(rows):
            for token, value in row.items():
                logits[0, i, token] = value
        self.logits.append(logits[0])
        return logits

    def gather(self, src, dst):
        pass


def _proactive(tree, engine, beam_len):
    """ProactiveStrategy on a CPU tree, built without the env-backed config."""
    drafter = SpecExecProactiveDraft.__new__(SpecExecProactiveDraft)
    drafter._logger = logging.getLogger(__name__)
    drafter._engine, drafter._tree = engine, tree
    drafter._device, drafter._dtype = CPU, F32
    drafter._max_n_beams, drafter._max_beam_len = 8, beam_len
    drafter._max_branch_width, drafter._max_budget, drafter._max_len = 2, 8, 64
    drafter.last_candidates, drafter.last_bonus_logprob = _t([]), None

    strategy = ProactiveStrategy.__new__(ProactiveStrategy)
    OverlapStrategy.__init__(strategy, tree, engine, CPU, F32)
    strategy._pd, strategy._depth_gain = drafter, beam_len
    strategy._pending = (None, None, None, None)
    return strategy


def test_proactive_logs_its_single_bet_and_the_splice(tmp_path):
    path = tmp_path / "client_0.tree_trace.jsonl"
    tree = Tree(_t([[100, 101, 102]]), CPU, F32, max_len=64)
    engine = _ScriptedEngine([
        [{60: 5.0}, {70: 5.0}],  # bonus scores after leaves #4 and #5
        [{80: 5.0, 81: 4.0}],  # one level grown under the bet
    ])
    strategy = _proactive(tree, engine, beam_len=1)
    tracer = TreeTracer(tree, strategy, path, 0, {"overlap_strategy": "proactive"})

    tracer.begin_request(req_idx=1)
    tracer.begin_step(req_idx=1, step_idx=0, prefill=True)
    _add(tree, [10, 11], [2, 2], [-0.1, -0.9])  # #3, #4 (leaf)
    _add(tree, [12], [3], [-0.4])  # #5 (the better leaf)
    tree.status[3:6] = tree.PROCESSED
    tracer.log_draft()

    strategy.speculate()
    tracer.log_speculation()

    ov = tracer._record["overlap"]
    assert ov["strategy"] == "proactive"
    assert ov["candidates"] == [
        {"node": 5, "fan": 1, "excluded": None},
        {"node": 4, "fan": 0, "excluded": None},
    ]
    (bet,) = ov["bets"]
    assert (bet["exit"], bet["bonus"], bet["root"]) == (5, 70, 6)
    expected = torch.log_softmax(engine.logits[0][1], dim=-1)[70].item()
    assert bet["logprob"] == round(expected, 4)
    assert bet["nodes"] == [6, 7, 8] and bet["has_frontier"]
    assert [(n["idx"], n["token"], n["parent"]) for n in ov["forest"]] == [
        (6, 70, 5), (7, 80, 6), (8, 81, 6)
    ]

    # verifier accepts #3 -> #5 and samples the bonus proactive bet on
    seq_mask = torch.zeros(tree.end, dtype=torch.bool)
    seq_mask[[0, 1, 2, 3, 5]] = True
    result = strategy.reconcile(
        seq_mask=seq_mask, last_accepted_token_idx=5, extra_token_id=_t([70])
    )
    tracer.end_step(
        accepted_idx=_t([3, 5]), accepted_ids=_t([10, 12]), exit_idx=5, bonus=70,
        result=result,
    )
    step0 = _records(path)[-1]
    assert step0["verify"]["spliced"] and step0["verify"]["n_reused"] == 3

    # next step starts from the spliced branch
    tracer.begin_step(req_idx=1, step_idx=1, prefill=False)
    tracer.log_draft()
    reused = {n["token"]: n["reused"] for n in tracer._record["draft"]["nodes"]}
    assert reused == {70: False, 80: True, 81: True}


def test_proactive_without_leaves_logs_no_bet(tmp_path):
    tree = Tree(_t([[1, 2]]), CPU, F32, max_len=16)
    strategy = _proactive(tree, _ScriptedEngine([]), beam_len=1)
    tracer = TreeTracer(tree, strategy, tmp_path / "t.jsonl", 0, {})
    tracer.begin_step(0, 0, True)
    tracer.log_draft()
    strategy.speculate()
    tracer.log_speculation()
    assert tracer._record["overlap"] == {
        "strategy": "proactive", "candidates": [], "bets": [], "forest": []
    }


def test_disabled_logs_an_empty_overlap_and_renders(tmp_path):
    from script.render_tree_trace import (
        annotate,
        load_trace,
        render_html,
        render_text,
        token_ids,
    )

    path = tmp_path / "client_0.tree_trace.jsonl"
    tree = Tree(_t([[100, 101, 102]]), CPU, F32, max_len=64)
    tracer = TreeTracer(tree, None, path, 0, {"overlap_strategy": "disabled"})
    tracer.begin_request(2)
    tracer.begin_step(2, 0, True)
    _add(tree, [10, 11], [2, 2], [-0.1, -0.9])
    tree.status[3:5] = tree.PROCESSED
    tracer.log_draft()
    tracer.log_speculation()
    assert tracer._record["overlap"] == {
        "strategy": "disabled", "candidates": [], "bets": [], "forest": []
    }

    seq_mask = torch.zeros(tree.end, dtype=torch.bool)
    seq_mask[[0, 1, 2, 3]] = True
    reorder_to_verified_path(tree, _NoEngine(), CPU, seq_mask)
    append_bonus_token(tree, _t([21]), CPU)
    tracer.end_step(
        accepted_idx=_t([3]), accepted_ids=_t([10]), exit_idx=3, bonus=21,
        result=OverlapResult(spliced=False, cache_hit=False, n_reused=0, n_hypotheses=0),
    )
    tracer.begin_step(2, 1, False)
    tracer.log_draft()
    assert not any(n["reused"] for n in tracer._record["draft"]["nodes"])

    run, requests = load_trace(path)
    annotate(requests)
    assert run["overlap_strategy"] == "disabled"
    assert requests[0]["steps"][0]["_outcome"]["kind"] == "none"
    vocab = {i: f"t{i}" for i in token_ids(requests)}
    text = render_text(requests, vocab)
    assert "[2] overlap disabled" in text and "no speculation" in text
    page = render_html(run, requests, vocab, "client_0")
    assert "__DATA__" not in page and "__JS__" not in page


def test_renderer_reads_traces_from_before_overlap_key(tmp_path):
    from script.render_tree_trace import annotate, load_trace

    root = {"idx": 1, "token": 2, "parent": 0, "logprob": 0.0, "position": 1, "reused": False}
    step = {
        "type": "step", "client_idx": 0, "req_idx": 0, "step_idx": 0,
        "prefill": True, "chain_len": 2, "chain_new": [],
        "draft": {"root": 1, "nodes": [root]},
        "saguaro": {"candidates": [], "bets": [], "forest": []},
        "verify": {
            "accepted": [], "accepted_tokens": [], "exit": 1, "bonus": 5,
            "cache_hit": False, "spliced": False, "n_reused": 0,
        },
    }
    records = [
        {"type": "run", "draft_model": "x"},
        {"type": "request", "client_idx": 0, "req_idx": 0, "prompt": [1, 2]},
        step,
    ]
    path = tmp_path / "old.tree_trace.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))

    run, requests = load_trace(path)
    annotate(requests)
    assert run["overlap_strategy"] == "saguaro"
    assert requests[0]["steps"][0]["overlap"]["strategy"] == "saguaro"
    assert "saguaro" not in requests[0]["steps"][0]
