"""Per-request generation trace, for diffing two runs prompt-for-prompt.

Aggregate metrics cannot answer "are these two systems decoding the same
text?" -- two runs can agree on mean accept rate and throughput while
starting from different prompts and producing different tokens. This
records, per request, the exact prompt that was tokenized, the tokens
generated from it, and why generation stopped.

Two files land in the run's log directory:

  trace.jsonl  one record per request, streamed as each request finishes,
               so an interrupted run still leaves a usable trace.
  trace.txt    the same records sorted by req_idx, written at close(). The
               two runs visit requests in different orders (CPython's
               random.shuffle vs the port's std::mt19937), so sorting is
               what lets them line up, and every value is JSON-escaped onto
               a single line so a newline inside a prompt cannot break the
               alignment. Diff these directly:

                   diff serverA/trace.txt mobile/trace.txt

mobile-specedge/src/script/client.cpp writes byte-identical formatting from
the same field list, so that diff is a decision-level comparison of this
engine against the llama.cpp port. Field order and the ", " separator below
are part of that contract: changing either here means changing it there
too, or every record diffs on formatting alone.
"""

import atexit
import json
from pathlib import Path
from typing import Sequence

# Shared with the C++ writer. Fixed order, fixed 13-column key gutter.
_FIELDS = (
    "stop_reason",
    "prompt_len",
    "n_generated",
    "prompt_text",
    "output_text",
    "prompt_tokens",
    "output_tokens",
)


def _fmt(value) -> str:
    """Render one value as a single line, matching nlohmann's dump()."""
    if isinstance(value, str):
        # ensure_ascii=False keeps UTF-8 raw, as nlohmann::json::dump() does.
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(str(int(v)) for v in value) + "]"
    return str(value)


class TraceWriter:
    def __init__(self, log_dir: Path, dataset: Sequence[str], tokenizer):
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._records: dict[int, dict] = {}

        log_dir.mkdir(parents=True, exist_ok=True)
        self._txt_path = log_dir / "trace.txt"
        self._jsonl = open(log_dir / "trace.jsonl", "w")

        # A run killed partway through is the common case while debugging a
        # divergence; close() is idempotent, so registering it here means
        # trace.txt exists for whatever requests did finish.
        atexit.register(self.close)

    def add(
        self,
        req_idx: int,
        prompt_token_ids: list[int],
        output_token_ids: list[int],
        stop_reason: str,
    ):
        record = {
            "req_idx": int(req_idx),
            "stop_reason": stop_reason,
            "prompt_len": len(prompt_token_ids),
            "n_generated": len(output_token_ids),
            # The prompt string as handed to the tokenizer, not a decode of
            # the ids: a chat-template difference is exactly what this file
            # exists to catch, and decoding back would hide whitespace.
            "prompt_text": self._dataset[req_idx],
            "output_text": self._tokenizer.decode(
                output_token_ids, skip_special_tokens=False
            ),
            "prompt_tokens": [int(t) for t in prompt_token_ids],
            "output_tokens": [int(t) for t in output_token_ids],
        }

        self._records[record["req_idx"]] = record
        # Compact separators and insertion order match nlohmann's
        # ordered_json::dump() on the C++ side, so trace.jsonl is
        # diffable after sorting too, not just trace.txt.
        self._jsonl.write(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        self._jsonl.flush()

    def close(self):
        if self._jsonl.closed:
            return
        self._jsonl.close()

        with open(self._txt_path, "w") as f:
            for req_idx in sorted(self._records):
                record = self._records[req_idx]
                f.write(f"==== req_idx={req_idx} ====\n")
                for field in _FIELDS:
                    f.write(f"{field:<13} {_fmt(record[field])}\n")
