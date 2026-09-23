import asyncio
import json

import pytest
import torch

from specedge.network.oracle import OracleValidator, load_trace

EOS = 99


def _oracle(refs, **kw):
    kw.setdefault("server_ms", 0.0)
    kw.setdefault("prefill_ms", 0.0)
    kw.setdefault("rtt_ms", 0.0)
    return OracleValidator(
        trace_path=None,
        device=torch.device("cpu"),
        eos_token_id=EOS,
        references=refs,
        **kw,
    )


def test_select_is_next_reference_token_by_position():
    # prompt [10, 11, 12], output [20, 21, 22]
    oracle = _oracle({7: [10, 11, 12, 20, 21, 22]})
    # last prompt token (pos 2) plus two siblings at pos 3 and one child at pos 4
    positions = torch.tensor([[2, 3, 3, 4]])
    sel = oracle.select(7, positions)
    assert sel.tolist() == [20, 21, 21, 22]


def test_select_answers_eos_past_reference_end():
    oracle = _oracle({0: [1, 2, 3]})
    sel = oracle.select(0, torch.tensor([[1, 2, 3, 6]]))
    assert sel.tolist() == [3, EOS, EOS, EOS]


def test_request_matches_grpc_contract():
    oracle = _oracle({3: [5, 6, 7, 8]})
    input_ids = torch.tensor([[6, 7, 9]])
    positions = torch.tensor([[1, 2, 2]])
    sel, prefill_cnt = asyncio.run(
        oracle.request(
            client_idx=0,
            req_idx=3,
            input_ids=input_ids,
            position_ids=positions,
            cache_seq_indices=positions.flatten(),
            attention_mask=torch.ones(1, 1, 3, 3),
            parent_indices=torch.tensor([0, 0]),
            prefill=True,
            prefix="p",
        )
    )
    assert sel.shape == (input_ids.size(-1),)
    assert sel.dtype == torch.long
    assert sel.tolist() == [7, 8, 8]
    assert prefill_cnt == 1


def test_request_sleeps_simulated_latency():
    oracle = _oracle({0: [1, 2, 3]}, server_ms=30.0, rtt_ms=20.0)

    async def run():
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await oracle.request(
            client_idx=0,
            req_idx=0,
            input_ids=torch.tensor([[2]]),
            position_ids=torch.tensor([[1]]),
            cache_seq_indices=torch.tensor([1]),
            attention_mask=torch.ones(1, 1, 1, 1),
            parent_indices=torch.tensor([], dtype=torch.long),
        )
        return (loop.time() - t0) * 1000

    assert asyncio.run(run()) >= 49.0


def test_server_log_is_metric_compatible(tmp_path):
    log_path = tmp_path / "exp" / "server.jsonl"
    oracle = _oracle({0: [1, 2, 3]}, server_ms=42.0, server_log_path=log_path)
    for prefill in (True, False):
        asyncio.run(
            oracle.request(
                client_idx=0,
                req_idx=0,
                input_ids=torch.tensor([[2]]),
                position_ids=torch.tensor([[1]]),
                cache_seq_indices=torch.tensor([1]),
                attention_mask=torch.ones(1, 1, 1, 1),
                parent_indices=torch.tensor([], dtype=torch.long),
                prefill=prefill,
                prefix="p",
            )
        )
    rows = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [r["target"]["prefill"] for r in rows] == [1, 0]
    assert rows[1]["target"]["server_end_to_end_t"] == 42.0
    assert all("timestamp" in r for r in rows)


def test_check_prompt():
    oracle = _oracle({4: [1, 2, 3, 4]})
    oracle.check_prompt(4, torch.tensor([[1, 2]]))
    with pytest.raises(ValueError):
        oracle.check_prompt(4, torch.tensor([[1, 5]]))
    with pytest.raises(KeyError):
        oracle.check_prompt(5, torch.tensor([[1]]))


def test_load_trace(tmp_path):
    p = tmp_path / "trace.jsonl"
    p.write_text(
        json.dumps({"req_idx": 2, "prompt_tokens": [1, 2], "output_tokens": [3]})
        + "\n"
    )
    assert load_trace(p) == {2: [1, 2, 3]}
