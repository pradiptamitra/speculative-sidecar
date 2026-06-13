"""Generate and cache the canonical summary for each document.

The canonical summary is the TARGET model's greedy continuation of the bare
summary prompt. It is the single fixed token sequence y_1..y_S that every
acceptance condition (B / E2 / G0) teacher-forces against in acceptance.py, so
we generate it once, deterministically, and freeze it to disk.

Greedy (do_sample=False) is deliberate: alpha must reflect the draft/target
distribution mismatch, not sampling noise, so the reference sequence must be
reproducible.

We store the generated token IDs (not just text): the acceptance math compares
p_target vs p_draft at the exact same token positions, and decode->re-encode is
not guaranteed to round-trip, so the IDs are the source of truth.

    python canonical_summaries.py --n 10 --max-chars 3000   # quick local run
    python canonical_summaries.py --n 100                    # full snapshot
"""

import argparse
import json
import os

import torch

from documents import load_documents
from models import load_pair
from prompts import build_summary_messages


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10, help="number of documents")
    ap.add_argument("--max-new-tokens", type=int, default=200, help="summary cap")
    ap.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="truncate each article to this many chars (local-loop speed); None = full",
    )
    ap.add_argument("--out", default="data/canonical_summaries.jsonl")
    args = ap.parse_args()

    target, draft, tokenizer, cfg = load_pair()
    del draft  # only the target generates the canonical summary; free its memory
    device = cfg["device"]

    # EOS ids per the model's generation config (Qwen stops on <|im_end|> and/or
    # <|endoftext|>); normalize int|list|None to a set for membership testing.
    eos = target.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos)

    docs = load_documents(args.n, max_chars=args.max_chars)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    records = []
    for i, doc in enumerate(docs):
        msgs = build_summary_messages(doc["article"])  # bare prompt, suffix=None
        enc = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            out = target.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # greedy -> deterministic canonical summary
            )
        summary_ids = out[0, prompt_len:].tolist()
        ended_eos = bool(summary_ids) and summary_ids[-1] in eos_ids
        summary_text = tokenizer.decode(summary_ids, skip_special_tokens=True)

        records.append(
            {
                "id": doc["id"],
                "prompt_len": prompt_len,  # = D+s in the note's notation
                "summary_ids": summary_ids,  # y_1..y_S (may include trailing EOS)
                "summary_len": len(summary_ids),  # ~ S
                "ended_eos": ended_eos,  # did greedy stop on EOS vs hit the cap?
                "summary_text": summary_text,
            }
        )
        print(
            f"[{i + 1}/{len(docs)}] {doc['id']}  "
            f"prompt={prompt_len}tok summary={len(summary_ids)}tok "
            f"eos={ended_eos}\n    {summary_text[:160]!r}"
        )

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(records)} canonical summaries to {args.out}")


if __name__ == "__main__":
    main()
