"""Model-pair loader for the speculative-decoding sidecar experiments.

Same code, two tiers and three devices, selected by env vars (per the research
note's section 5.3):

    MODEL_TIER = dev | prod      # which (target, draft) pair
    DEVICE     = cpu | mps | cuda

    dev  : target Qwen2.5-1.5B-Instruct, draft Qwen2.5-0.5B-Instruct  (CPU loop)
    prod : target Qwen2.5-7B-Instruct,   draft Qwen2.5-1.5B-Instruct  (MPS/cloud)

Weights are pulled from the HF Hub on first use (Qwen2.5 is Apache-2.0 and
ungated, no token needed) and cached under ~/.cache/huggingface afterwards.

Notes that matter for the experiments:
- attn_implementation="eager" so we can pass a custom 4D additive attention mask
  (Variation G's suffix mask). flex_attention is CUDA-oriented and skipped here.
- target and draft share the Qwen2.5 tokenizer, which speculative decoding
  requires; we load it once and return it.
"""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PAIRS = {
    "dev": ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    "prod": ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
}


def get_config() -> dict:
    """Resolve tier/device/dtype from the environment."""
    tier = os.environ.get("MODEL_TIER", "dev")
    device = os.environ.get("DEVICE", "cpu")
    if tier not in PAIRS:
        raise ValueError(f"MODEL_TIER must be one of {list(PAIRS)}, got {tier!r}")
    # bf16 on accelerators (Qwen2.5's native dtype; fp32 range avoids the
    # inf/NaN overflow fp16 hits in attention on the 7B); fp32 on CPU.
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    target_id, draft_id = PAIRS[tier]
    return {
        "tier": tier,
        "device": device,
        "dtype": dtype,
        "target_id": target_id,
        "draft_id": draft_id,
    }


def _load_model(model_id: str, device: str, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,  # transformers 5.x kwarg; torch_dtype is deprecated
        attn_implementation="eager",  # required for custom 4D attention masks
    )
    model.to(device)
    model.eval()
    return model


def load_pair():
    """Load (target, draft, tokenizer, config) for the active tier/device."""
    cfg = get_config()
    print(
        f"[models] tier={cfg['tier']} device={cfg['device']} dtype={cfg['dtype']}\n"
        f"[models]   target={cfg['target_id']}\n"
        f"[models]   draft ={cfg['draft_id']}"
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["target_id"])
    target = _load_model(cfg["target_id"], cfg["device"], cfg["dtype"])
    draft = _load_model(cfg["draft_id"], cfg["device"], cfg["dtype"])
    return target, draft, tokenizer, cfg


if __name__ == "__main__":
    # Smoke test: load the pair and run one tiny forward pass on each.
    target, draft, tokenizer, cfg = load_pair()
    msgs = [{"role": "user", "content": "Say hello in one word."}]
    # transformers 5.x returns a BatchEncoding (input_ids + attention_mask) here,
    # not a bare tensor, so unpack it into the model call with **enc.
    enc = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    enc = {k: v.to(cfg["device"]) for k, v in enc.items()}
    with torch.no_grad():
        for name, m in [("target", target), ("draft", draft)]:
            out = m(**enc)
            nxt = out.logits[0, -1].argmax().item()
            print(f"[smoke] {name} next-token id={nxt} -> {tokenizer.decode([nxt])!r}")
