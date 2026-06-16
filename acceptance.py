"""Teacher-forced acceptance-rate experiment: baseline vs visible vs masked.

For each document we have a frozen canonical summary y_1..y_S (the target's
greedy continuation of the bare prompt; see canonical_summaries.py). We measure,
for three draft conditions, the per-position acceptance rate

    alpha_t = sum_v min( p(v | ctx_t), q(v | ctx_t) )      # distribution overlap

where p is the (condition-independent) target distribution and q is the draft
distribution under the condition. alpha = mean_t alpha_t, averaged over docs.

Conditions (all teacher-forced over the SAME canonical y_1..y_S):
  baseline  bare prompt                          -> reference alpha
  visible   compound prompt, suffix visible      -> content leak + position shift
  masked    compound prompt, suffix MASKED,
            naive (un-rebased) positions         -> position shift only

Decomposition:
  alpha(baseline) - alpha(masked)  = pure RoPE / position effect
  alpha(masked)   - alpha(visible) = pure content-leak effect

Design notes that keep the decomposition clean:
  * visible and masked use the identical token sequence; they differ only in the
    mask, so their gap isolates content visibility with positions held fixed.
  * The bare prompt is exactly the compound prompt with the suffix tokens removed
    (the suffix is whitespace-delimited, so BPE tokenizes the junction the same
    way). We locate the suffix span by diffing the two token id lists. Masking
    the suffix therefore reproduces the baseline's content bit-for-bit; the only
    baseline<->masked difference is the +n position shift.

    DEVICE=cpu MODEL_TIER=small python acceptance.py --n 10 --max-chars 3000
"""

import argparse
import json

import torch

from documents import load_documents
from models import load_pair
from prompts import HASHTAG_SUFFIX, build_summary_messages


def template_ids(tokenizer, messages) -> list[int]:
    """Chat-template -> flat list of token ids.

    transformers 5.x returns a BatchEncoding here, not a bare list, so we pull
    input_ids out and normalize tensor/nested forms to a plain list[int].
    """
    out = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=True
    )
    ids = out["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):  # batched [[...]] -> flat
        ids = ids[0]
    return ids


def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def common_suffix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(reversed(a), reversed(b)):
        if x != y:
            break
        n += 1
    return n


def causal_4d(seq_len, dtype, device, block=None):
    """Additive [1,1,L,L] mask: 0 where attendable, finfo.min where not.

    Causal (query i may attend to key j<=i). If block=(lo,hi) is given, those key
    columns are additionally forbidden to every query (the masked suffix span).
    """
    neg = torch.finfo(dtype).min
    m = torch.full((seq_len, seq_len), neg, dtype=dtype, device=device)
    m.masked_fill_(torch.tril(torch.ones(seq_len, seq_len, device=device)).bool(), 0.0)
    if block is not None:
        lo, hi = block
        m[:, lo:hi] = neg
    return m[None, None]


@torch.no_grad()
def summary_dist(model, seq_ids, prompt_len, n_summary, device, vocab,
                 attn_mask=None, position_ids=None):
    """Softmax distributions predicting the S summary tokens.

    Off-by-one: logits[i] predicts token i+1, so the distributions for the S
    summary tokens (at absolute indices prompt_len .. prompt_len+S-1) come from
    logits indices prompt_len-1 .. prompt_len+S-2.

    vocab: truncate to the common real-vocab width before softmax. Qwen2.5 models
    pad their output matrix to different rounded sizes (e.g. 7B=152064 vs
    1.5B=151936); the extra columns are padding for token ids that never occur.
    Slicing both to min(vocab) BEFORE softmax makes them normalize over the same
    token set so the overlap is well defined.
    """
    input_ids = torch.tensor([seq_ids], device=device)
    kwargs = {}
    if attn_mask is not None:
        kwargs["attention_mask"] = attn_mask
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    logits = model(input_ids, **kwargs).logits[0]  # [L, V]
    sl = logits[prompt_len - 1 : prompt_len - 1 + n_summary, :vocab]  # [S, vocab]
    return torch.softmax(sl.float(), dim=-1)  # fp32 for stable overlap


def overlap(p, q):
    """Per-position acceptance alpha_t = sum_v min(p_v, q_v) -> [S]."""
    return torch.minimum(p, q).sum(dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-chars", type=int, default=3000,
                    help="MUST match the value used in canonical_summaries.py")
    ap.add_argument("--summaries", default="data/canonical_summaries.jsonl")
    args = ap.parse_args()

    target, draft, tokenizer, cfg = load_pair()
    device, dtype = cfg["device"], cfg["dtype"]
    # common real-vocab width (models pad output matrices to different sizes)
    vocab = min(target.config.vocab_size, draft.config.vocab_size)

    with open(args.summaries) as f:
        canon = [json.loads(line) for line in f][: args.n]
    docs = load_documents(args.n, max_chars=args.max_chars)

    # per-position alphas pooled across docs (token-weighted), per condition
    pooled = {"baseline": [], "visible": [], "masked": []}
    per_doc = []
    max_4d_check = 0.0

    for rec, doc in zip(canon, docs):
        assert rec["id"] == doc["id"], "summary/doc order mismatch"
        summary_ids = rec["summary_ids"]
        S = len(summary_ids)

        bare_ids = template_ids(tokenizer, build_summary_messages(doc["article"]))
        comp_ids = template_ids(
            tokenizer, build_summary_messages(doc["article"], suffix=HASHTAG_SUFFIX)
        )
        assert len(bare_ids) == rec["prompt_len"], (
            f"prompt_len mismatch ({len(bare_ids)} vs {rec['prompt_len']}); "
            "did --max-chars match canonical_summaries.py?"
        )

        # locate the suffix span inside comp_ids by diffing against bare_ids.
        # compute the common suffix over the post-prefix remainder so a shared
        # boundary token (both prompts end in ".") can't be counted twice.
        lcp = common_prefix_len(bare_ids, comp_ids)
        lcs = common_suffix_len(bare_ids[lcp:], comp_ids[lcp:])
        assert len(bare_ids) == lcp + lcs, "bare is not comp-minus-suffix"
        suffix_lo, suffix_hi = lcp, len(comp_ids) - lcs  # span in comp_ids

        seq_bare = bare_ids + summary_ids
        seq_comp = comp_ids + summary_ids
        P_bare, P_comp = len(bare_ids), len(comp_ids)

        # target distribution p (bare, condition-independent), reused for all three
        p = summary_dist(target, seq_bare, P_bare, S, device, vocab)

        # baseline: draft, bare, native causal mask
        q_base = summary_dist(draft, seq_bare, P_bare, S, device, vocab)

        # visible: draft, compound, native causal mask (suffix visible)
        q_vis = summary_dist(draft, seq_comp, P_comp, S, device, vocab)

        # validate the custom-4D-mask path: recompute `visible` with a hand-built
        # 4D causal mask (no block) and confirm it matches the native path
        pos_comp = torch.arange(len(seq_comp), device=device)[None]
        q_vis_4d = summary_dist(
            draft, seq_comp, P_comp, S, device, vocab,
            attn_mask=causal_4d(len(seq_comp), dtype, device),
            position_ids=pos_comp,
        )
        max_4d_check = max(max_4d_check, (q_vis - q_vis_4d).abs().max().item())

        # masked: draft, compound, suffix MASKED, naive positions
        q_mask = summary_dist(
            draft, seq_comp, P_comp, S, device, vocab,
            attn_mask=causal_4d(len(seq_comp), dtype, device, block=(suffix_lo, suffix_hi)),
            position_ids=pos_comp,
        )

        a_base, a_vis, a_mask = overlap(p, q_base), overlap(p, q_vis), overlap(p, q_mask)
        pooled["baseline"].append(a_base)
        pooled["visible"].append(a_vis)
        pooled["masked"].append(a_mask)
        per_doc.append((doc["id"], S, suffix_hi - suffix_lo,
                        a_base.mean().item(), a_vis.mean().item(), a_mask.mean().item()))
        print(f"  {doc['id'][:12]}  S={S:3d} n={suffix_hi - suffix_lo:2d}  "
              f"baseline={a_base.mean():.4f}  visible={a_vis.mean():.4f}  "
              f"masked={a_mask.mean():.4f}")

    cat = {k: torch.cat(v) for k, v in pooled.items()}
    m_base, m_vis, m_mask = (cat[k].mean().item() for k in ("baseline", "visible", "masked"))

    print("\n=== acceptance (token-weighted over all summary positions) ===")
    print(f"  alpha(baseline) = {m_base:.4f}   bare prompt (reference)")
    print(f"  alpha(visible)  = {m_vis:.4f}   compound prompt (content leak + position shift)")
    print(f"  alpha(masked)   = {m_mask:.4f}   suffix masked, naive positions (position shift only)")
    print("\n=== decomposition ===")
    print(f"  alpha(baseline) - alpha(masked)  = {m_base - m_mask:+.4f}   pure RoPE / position effect")
    print(f"  alpha(masked)   - alpha(visible) = {m_mask - m_vis:+.4f}   pure content-leak effect")
    print(f"  alpha(baseline) - alpha(visible) = {m_base - m_vis:+.4f}   total leak (visible vs baseline)")
    print(f"\n[4D-mask self-check] max|visible_native - visible_4dcausal| = {max_4d_check:.2e} "
          "(should be ~0)")


if __name__ == "__main__":
    main()
