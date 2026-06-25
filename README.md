# speculative-sidecar

Code accompanying the *Speculative Decoding for Fun and Profit* series:

- **[Part I — the acceptance leak](https://pradiptamitra.github.io/2026/06/17/speculative-decoding-for-fun-and-profit-i/)**
- **[Part II — masking and the RoPE position shift](https://pradiptamitra.github.io/2026/06/25/speculative-decoding-for-fun-and-profit-ii/)**

The series asks: if you reuse a speculative-decoding **draft** model for a second,
prompt-defined task (generating hashtags after a summary), what does it cost the
draft's **acceptance rate**? Part I measures the cost of giving the draft the
compound ("now also output hashtags") prompt. Part II measures the cost of the
fix — masking the suffix during the summary — and the RoPE position shift that
masking introduces. This repo reproduces the numbers behind both posts.

Everything is **teacher-forced**: we generate the target's greedy summary once,
then read both models' next-token distributions over those same tokens and
average the overlap $\sum_v \min(p,q)$ — the exact expected acceptance rate. There
is no stochastic accept/reject loop. See the posts for the derivation.

## The three conditions

`acceptance.py` scores the draft under three setups, all against the *same* target
summary (so only the draft's setup varies):

| condition | draft prompt | suffix in context | summary positions |
|---|---|---|---|
| `baseline` | bare "summarize" | absent | natural (matches target) |
| `visible`  | compound ("…then hashtags") | present, **attended** | shifted by `n` |
| `masked`   | compound | present but **masked** | shifted by `n` |

- **Part I** is `baseline` vs `visible` — the content leak.
- **Part II** is `baseline` vs `masked` — the pure position (RoPE) cost, since
  `masked` has baseline-identical content and differs only by the `+n` shift —
  plus the n-sweep from `sweeps.py`.

## What you'll reproduce

All over 100 CNN/DailyMail articles. (Exact figures vary slightly with
hardware/dtype.)

**Part I — content leak** (`baseline` vs `visible`):

| pair | α(baseline) | α(visible) | relative drop |
|---|---|---|---|
| small (1.5B / 0.5B) | ~0.649 | ~0.616 | ~−5.1% |
| big (7B / 1.5B) | ~0.636 | ~0.627 | ~−1.4% |

**Part II — position cost** (`baseline` vs `masked`, real `n ≈ 25` suffix):

| pair | α(baseline) | α(masked) | position cost (baseline − masked) | relative |
|---|---|---|---|---|
| small (1.5B / 0.5B) | ~0.649 | ~0.645 | ~+0.0045 | ~+0.7% |
| big (7B / 1.5B) | ~0.636 | ~0.639 | ~−0.0030 | ~−0.5% |

**Part II — n-sweep** (position cost `baseline − masked` vs suffix length, from
`sweeps.py`):

| n | small | big |
|---|---|---|
| 25 | +0.0045 | −0.0030 |
| 50 | +0.0068 | −0.0037 |
| 100 | +0.0091 | −0.0081 |
| 200 | +0.0123 | −0.0099 |
| 400 | +0.0150 | −0.0093 |

## Setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1   # Mac
```

Models (Qwen2.5-Instruct, Apache-2.0, ungated) download from the HF Hub on first
use. Two env-var knobs select the run:

- `DEVICE` = `cpu` | `mps` | `cuda`
- `MODEL_TIER` = `small` (1.5B target / 0.5B draft) | `big` (7B target / 1.5B draft)

## Reproduce

One `acceptance.py` run per pair yields **all three conditions** — i.e. both
posts' headline numbers. `sweeps.py` adds Part II's n-sweep.

```bash
# 0. one-time: snapshot 100 CNN/DailyMail articles -> data/cnndm_test.jsonl
python fetch_documents.py --n 100

# --- big pair (Qwen2.5-7B target / 1.5B draft) ---
DEVICE=mps MODEL_TIER=big python canonical_summaries.py \
    --n 100 --max-chars 3000 --out data/canonical_summaries_big.jsonl
DEVICE=mps MODEL_TIER=big python acceptance.py \
    --n 100 --max-chars 3000 --summaries data/canonical_summaries_big.jsonl
DEVICE=mps MODEL_TIER=big python sweeps.py \
    --n 100 --max-chars 3000 --summaries data/canonical_summaries_big.jsonl

# --- small pair (Qwen2.5-1.5B target / 0.5B draft) ---
DEVICE=mps MODEL_TIER=small python canonical_summaries.py \
    --n 100 --max-chars 3000 --out data/canonical_summaries_small.jsonl
DEVICE=mps MODEL_TIER=small python acceptance.py \
    --n 100 --max-chars 3000 --summaries data/canonical_summaries_small.jsonl
DEVICE=mps MODEL_TIER=small python sweeps.py \
    --n 100 --max-chars 3000 --summaries data/canonical_summaries_small.jsonl
```

Reading off the numbers:

- **Part I:** from `acceptance.py`, `alpha(baseline)` and `alpha(visible)` (and
  their difference, the `total leak` line).
- **Part II:** from `acceptance.py`, the `alpha(baseline) - alpha(masked)` line
  (printed as the *pure RoPE / position effect*). From `sweeps.py`, the
  `base-masked` column across `n` (also written to `data/nsweep_{tier}.csv`).

Swap `DEVICE=mps` for `DEVICE=cpu` for a slower but device-independent run. The
`--max-chars` value **must match** between `canonical_summaries.py`, `acceptance.py`,
and `sweeps.py` (an assertion enforces it).

## Files

| file | role |
|---|---|
| `fetch_documents.py` | snapshot CNN/DailyMail to `data/` (one-time) |
| `documents.py` | read the snapshot |
| `models.py` | load the target/draft pair (env-var tier + device) |
| `prompts.py` | shared summary + hashtag-suffix prompts |
| `canonical_summaries.py` | generate the target's greedy summaries |
| `acceptance.py` | teacher-forced acceptance: `baseline` / `visible` / `masked` + decomposition (Parts I & II) |
| `sweeps.py` | Part II n-sweep (position cost vs `n`) + depth profile; writes `data/nsweep_{tier}.csv`, `data/depth_{tier}.csv` |
