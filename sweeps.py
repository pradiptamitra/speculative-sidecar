"""Two follow-up analyses for the acceptance experiment, in one model load:

1. n-SWEEP: how the pure RoPE/position cost (alpha(baseline) - alpha(masked))
   grows with the suffix length n. Because `masked` MASKS the inserted span, its
   content is irrelevant -- only its length matters (it shifts the summary by n
   position slots). So we insert n filler tokens at the suffix's insertion point
   and mask them. The n=25 point should reproduce acceptance.py's `masked`
   exactly (masked tokens contribute nothing), which cross-checks the construction.

2. DEPTH PROFILE: how alpha(baseline)-alpha(masked) and
   alpha(baseline)-alpha(visible) vary with depth t into the summary (using the
   real n=25 suffix). Tests the prediction that the RoPE effect is concentrated
   in the first few summary tokens and washes out.

(Conditions match acceptance.py: `baseline` = bare prompt, `visible` = compound
prompt with the suffix visible, `masked` = compound prompt with the suffix masked
and naive positions.)

    DEVICE=cpu  MODEL_TIER=dev  python sweeps.py --n 10 --max-chars 3000
    DEVICE=mps  MODEL_TIER=prod PYTORCH_ENABLE_MPS_FALLBACK=1 \
        python sweeps.py --n 10 --max-chars 3000 \
        --summaries data/canonical_summaries_prod.jsonl
"""

import argparse
import csv
import json

import torch

from acceptance import (
    causal_4d,
    common_prefix_len,
    common_suffix_len,
    overlap,
    summary_dist,
    template_ids,
)
from documents import load_documents
from models import load_pair
from prompts import HASHTAG_SUFFIX, build_summary_messages


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10, help="number of documents")
    ap.add_argument("--max-chars", type=int, default=3000)
    ap.add_argument("--summaries", default="data/canonical_summaries.jsonl")
    ap.add_argument("--n-sweep", default="25,50,100,200,400",
                    help="comma-separated suffix lengths for the RoPE sweep")
    ap.add_argument("--depth-cap", type=int, default=24,
                    help="print per-position depth rows up to this t")
    args = ap.parse_args()
    sweep_ns = [int(x) for x in args.n_sweep.split(",")]

    target, draft, tokenizer, cfg = load_pair()
    device, dtype = cfg["device"], cfg["dtype"]
    vocab = min(target.config.vocab_size, draft.config.vocab_size)
    # filler token for the masked sweep span; identity is irrelevant (masked),
    # only needs to be a valid id.
    filler_id = tokenizer(" the", add_special_tokens=False)["input_ids"][0]

    with open(args.summaries) as f:
        canon = [json.loads(line) for line in f][: args.n]
    docs = load_documents(args.n, max_chars=args.max_chars)

    # n-sweep accumulators: pooled per-position overlaps for baseline and masked(n)
    sweep_base = []
    sweep_masked = {n: [] for n in sweep_ns}
    # depth profile accumulators: sum and count, by position t, of
    # (baseline - masked) and (baseline - visible)
    depth_mask = {}  # t -> [sum, count]
    depth_vis = {}

    for rec, doc in zip(canon, docs):
        assert rec["id"] == doc["id"], "summary/doc order mismatch"
        summary_ids = rec["summary_ids"]
        S = len(summary_ids)

        bare_ids = template_ids(tokenizer, build_summary_messages(doc["article"]))
        comp_ids = template_ids(
            tokenizer, build_summary_messages(doc["article"], suffix=HASHTAG_SUFFIX)
        )
        assert len(bare_ids) == rec["prompt_len"], "prompt_len mismatch (max-chars?)"
        lcp = common_prefix_len(bare_ids, comp_ids)
        lcs = common_suffix_len(bare_ids[lcp:], comp_ids[lcp:])
        assert len(bare_ids) == lcp + lcs, "bare is not comp-minus-suffix"
        insert_at = lcp  # where the suffix (or filler) goes
        suffix_lo, suffix_hi = lcp, len(comp_ids) - lcs

        seq_bare = bare_ids + summary_ids
        seq_comp = comp_ids + summary_ids
        P_bare, P_comp = len(bare_ids), len(comp_ids)

        # target p (bare) reused everywhere; draft baseline reused everywhere
        p = summary_dist(target, seq_bare, P_bare, S, device, vocab)
        q_base = summary_dist(draft, seq_bare, P_bare, S, device, vocab)
        a_base = overlap(p, q_base)
        sweep_base.append(a_base)

        # --- depth profile: real-suffix `visible` and `masked` (n=25) ---
        pos_comp = torch.arange(len(seq_comp), device=device)[None]
        q_vis = summary_dist(draft, seq_comp, P_comp, S, device, vocab)
        q_mask = summary_dist(
            draft, seq_comp, P_comp, S, device, vocab,
            attn_mask=causal_4d(len(seq_comp), dtype, device, block=(suffix_lo, suffix_hi)),
            position_ids=pos_comp,
        )
        d_mask = (a_base - overlap(p, q_mask)).tolist()
        d_vis = (a_base - overlap(p, q_vis)).tolist()
        for t in range(S):
            depth_mask.setdefault(t, [0.0, 0]); depth_mask[t][0] += d_mask[t]; depth_mask[t][1] += 1
            depth_vis.setdefault(t, [0.0, 0]); depth_vis[t][0] += d_vis[t]; depth_vis[t][1] += 1

        # --- n-sweep: masked filler of length n at insert_at ---
        for n in sweep_ns:
            seq = bare_ids[:insert_at] + [filler_id] * n + bare_ids[insert_at:] + summary_ids
            Pn = P_bare + n
            mask = causal_4d(len(seq), dtype, device, block=(insert_at, insert_at + n))
            pos = torch.arange(len(seq), device=device)[None]
            q_mask_n = summary_dist(draft, seq, Pn, S, device, vocab,
                                    attn_mask=mask, position_ids=pos)
            sweep_masked[n].append(overlap(p, q_mask_n))

        print(f"  {doc['id'][:12]}  S={S:3d}  done")

    # ===== n-sweep report =====
    m_base = torch.cat(sweep_base).mean().item()
    print("\n=== n-sweep: pure RoPE/position effect vs suffix length ===")
    print(f"  alpha(baseline) = {m_base:.4f}  (n-independent reference)")
    print(f"  {'n':>5} {'alpha(masked)':>14} {'base-masked':>12}")
    sweep_rows = []
    for n in sweep_ns:
        m_mask = torch.cat(sweep_masked[n]).mean().item()
        print(f"  {n:>5} {m_mask:>14.4f} {m_base - m_mask:>+12.4f}")
        sweep_rows.append((n, m_mask, m_base - m_mask))

    # ===== depth profile report =====
    print("\n=== depth profile: effect vs position t into the summary "
          "(real n=25) ===")
    print(f"  {'t':>4} {'docs':>5} {'base-masked':>12} {'base-visible':>13}")
    depth_rows = []
    for t in sorted(depth_mask):
        sm, cm = depth_mask[t]
        sv, cv = depth_vis[t]
        row = (t, cm, sm / cm, sv / cv)
        depth_rows.append(row)
        if t < args.depth_cap:
            print(f"  {t:>4} {cm:>5} {sm / cm:>+12.4f} {sv / cv:>+13.4f}")

    # ===== CSVs for later plotting =====
    tier = cfg["tier"]
    with open(f"data/nsweep_{tier}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "alpha_masked", "baseline_minus_masked"])
        w.writerows(sweep_rows)
    with open(f"data/depth_{tier}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "docs", "baseline_minus_masked", "baseline_minus_visible"])
        w.writerows(depth_rows)
    print(f"\nwrote data/nsweep_{tier}.csv and data/depth_{tier}.csv")


if __name__ == "__main__":
    main()
