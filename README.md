# speculative-sidecar

Code accompanying the post **[Speculative Decoding for Fun and Profit I](https://pradiptamitra.github.io/2026/06/17/speculative-decoding-for-fun-and-profit-i/)**.

That post asks: if you reuse a speculative-decoding **draft** model for a second,
prompt-defined task (generating hashtags after a summary), does appending that
task to the draft's prompt hurt its **acceptance rate**? This repo reproduces the
one experiment behind that post — the drop in acceptance when the draft is given
the compound ("now also output hashtags") prompt instead of the plain one.

> The repo contains more code than this README uses (`sweeps.py` and the `masked`
> condition are for a later post). For now this README covers only the first
> post's result.

## What you'll reproduce

For each model pair, acceptance with **identical** draft/target prompts vs. a
**compound** draft prompt (target prompt unchanged), over 100 CNN/DailyMail
articles:

| pair | α (identical prompts) | α (compound draft prompt) | relative drop |
|---|---|---|---|
| small (1.5B / 0.5B) | ~0.649 | ~0.616 | ~−5.1% |
| big (7B / 1.5B) | ~0.636 | ~0.627 | ~−1.4% |

(Exact figures vary slightly with hardware/dtype.) In `acceptance.py`'s output,
"identical prompts" is the **`baseline`** condition and "compound draft prompt"
is **`visible`**; ignore the `masked` column for this post.

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

```bash
# 0. one-time: snapshot 100 CNN/DailyMail articles -> data/cnndm_test.jsonl
python fetch_documents.py --n 100

# --- big pair (Qwen2.5-7B target / 1.5B draft) ---
DEVICE=mps MODEL_TIER=big python canonical_summaries.py \
    --n 100 --max-chars 3000 --out data/canonical_summaries_big.jsonl
DEVICE=mps MODEL_TIER=big python acceptance.py \
    --n 100 --max-chars 3000 --summaries data/canonical_summaries_big.jsonl

# --- small pair (Qwen2.5-1.5B target / 0.5B draft) ---
DEVICE=mps MODEL_TIER=small python canonical_summaries.py \
    --n 100 --max-chars 3000 --out data/canonical_summaries_small.jsonl
DEVICE=mps MODEL_TIER=small python acceptance.py \
    --n 100 --max-chars 3000 --summaries data/canonical_summaries_small.jsonl
```

Each `acceptance.py` run prints the per-condition α and the decomposition; read
off `alpha(baseline)` and `alpha(visible)` and their difference. Swap
`DEVICE=mps` for `DEVICE=cpu` for a slower but device-independent run. The
`--max-chars` value **must match** between `canonical_summaries.py` and
`acceptance.py` (an assertion enforces it).

## How it works (one line)

We don't run the stochastic accept/reject loop. We generate the target's greedy
summary once, then teacher-force both models over it in a single pass each and
average the distribution overlap $\sum_v \min(p,q)$ — the exact expected
acceptance rate. See the post for the derivation.

## Files

| file | role |
|---|---|
| `fetch_documents.py` | snapshot CNN/DailyMail to `data/` (one-time) |
| `documents.py` | read the snapshot |
| `models.py` | load the target/draft pair (env-var tier + device) |
| `prompts.py` | shared summary + hashtag-suffix prompts |
| `canonical_summaries.py` | generate the target's greedy summaries |
| `acceptance.py` | teacher-forced acceptance (`baseline` / `visible` / `masked`) |
| `sweeps.py` | extended analysis (later post): n-sweep + depth profile |
