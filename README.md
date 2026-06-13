# speculative-sidecar

Code for measuring whether reusing a speculative-decoding **draft** model for a
second, prompt-defined task hurts the speculation it's there for. We measure the
draft's **acceptance rate** during summary generation under three conditions and
attribute any drop to either *content leak* or *RoPE position shift*.

## Setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1   # Mac
```

Models (Qwen2.5-Instruct, Apache-2.0, ungated) download from the HF Hub on first
use. Device/size are env-var knobs:

- `DEVICE` = `cpu` | `mps` | `cuda`
- `MODEL_TIER` = `small` (1.5B target / 0.5B draft) | `big` (7B target / 1.5B draft)

## Reproduce

```bash
# 1. snapshot N documents from CNN/DailyMail -> data/cnndm_test.jsonl
python fetch_documents.py --n 100

# 2. target's greedy summaries (the reference path)
DEVICE=mps MODEL_TIER=big python canonical_summaries.py \
    --n 100 --max-chars 3000 --out data/canonical_summaries_big.jsonl

# 3. acceptance experiment: baseline vs visible vs masked
DEVICE=mps MODEL_TIER=big python acceptance.py \
    --n 100 --max-chars 3000 --summaries data/canonical_summaries_big.jsonl

# 4. n-sweep (RoPE vs suffix length) + depth profile
DEVICE=mps MODEL_TIER=big python sweeps.py \
    --n 100 --max-chars 3000 --summaries data/canonical_summaries_big.jsonl
```

For a fast local correctness loop, run all three steps under
`DEVICE=cpu MODEL_TIER=small`, saving the summaries to
`data/canonical_summaries_small.jsonl` and passing that same path as
`--summaries` to `acceptance.py` and `sweeps.py`.

## Files

| file | role |
|---|---|
| `fetch_documents.py` | snapshot CNN/DailyMail to `data/` (one-time) |
| `documents.py` | read the snapshot |
| `models.py` | load the target/draft pair (env-var tier + device) |
| `prompts.py` | shared summary + hashtag-suffix prompts |
| `canonical_summaries.py` | generate the target's greedy summaries |
| `acceptance.py` | teacher-forced acceptance: baseline / visible / masked |
| `sweeps.py` | n-sweep + per-position depth profile |
