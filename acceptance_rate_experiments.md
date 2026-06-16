# Acceptance-Rate Experiments: Hashtag-Suffix Leak and RoPE Mismatch

Scoping note for the first two experiments.

## Goal

Measure how much the draft model's acceptance rate **α** degrades, during the
summary phase, when we move from a bare draft prompt toward the sidecar designs.
We want to attribute any degradation to one of two distinct causes:

1. **Content leak** — the draft can *see* the hashtag suffix and so conditions
   its summary tokens on it (the `visible` condition below).
2. **Position mismatch** — the draft's summary tokens sit at the wrong RoPE
   positions because the prefilled suffix pushed them `n` slots later (the naive
   layout, before position rebasing; the `masked` condition below).

These are normally tangled together. The trick here is to run an intermediate
condition that has the position shift but *not* the content leak, which isolates
each effect.

## Notation

All are token counts of prefilled segments:

- **D** — document length
- **s** — summary-instruction length (`"Summarize."`)
- **n** — hashtag-suffix length (`"Now output 3–5 hashtags… as JSON."`)
- **S** — generated-summary length (uppercase; distinct from lowercase `s`)

The target model always runs the **bare** prompt `[document]["Summarize."]`; its
summary tokens live at positions `D+s .. D+s+S-1`. Everything below changes only
the **draft** side. α is measured during the **summary phase** only (the sidecar
phase is pure draft decode and has no acceptance rate).

## The four conditions

| # | Condition | Suffix in draft context? | Summary token positions | Isolates |
|---|---|---|---|---|
| baseline  | **Baseline** — bare `"Summarize."` | no | `D+s ..` (matches target) | — (reference α) |
| visible | **Compound prompt** — `"Summarize. Now hashtags…"`, no masking | yes, **visible** | `D+s+n ..` (naive) | content leak **+** position shift |
| masked | **Naive (un-rebased)** — suffix prefilled but **masked** during summary; positions naive | yes, but masked out | `D+s+n ..` (naive) | position shift **only** |
| rebased | **Full (rebased)** — suffix masked **and** positions rebased | yes, but masked out | `D+s ..` (rebased, overlaps suffix) | should ≈ baseline |

The user's two requested experiments are **baseline vs visible** and **baseline vs masked**. rebased is the
natural completion (the fix) and cheap to add once masked is in place; included here
so the table tells the whole story, but it is not required for round one.

## The decomposition

Because the three conditions differ by exactly one factor at a time:

```
α(baseline)  − α(masked)  =  pure RoPE / position effect      (content identical to baseline, only positions shifted)
α(masked) − α(visible)  =  pure content-leak effect          (positions identical, suffix becomes visible)
α(baseline)  − α(visible)  =  total visible leak  (= sum of the two above)
α(rebased) − α(baseline)   =  residual after the position fix    (expected ≈ 0; sanity check)
```

This is the payoff of running masked alongside visible: without it, visible's α drop is a
single number that confounds "the model staged for the downstream task" with
"RoPE saw the wrong positions." With it, we get the breakdown.

Note that **masked has content attention bit-for-bit identical to baseline** (the suffix is
masked, so the masked summary tokens attend to exactly `[document]["Summarize."]`
just like the baseline). The *only* difference from baseline is that each summary
token's query/key rotation uses a position index that is `n` larger. So any
α(baseline) − α(masked) gap is attributable to RoPE alone — this is the clean experiment the
user asked for.

## Experiment 1 — Hashtag-suffix content effect (visible)

**Question:** does telling the draft "you'll also produce hashtags afterward"
change how it drafts the summary, costing acceptance?

**Assumption under test:** for an idealized instruction-follower
`P(summary | "do A then B") = P(summary | "do A")`, so α(visible) ≈ α(baseline). We
expect this to *leak* in practice (length/structure conditioning, format
bleed-through, attention reallocation), more so because the draft is the small
model.

**Conditions:** baseline vs visible. Optionally add a third, irrelevant compound suffix
(`"Summarize. Then translate to French."`) to separate "any future instruction
shifts drafting" from "hashtag-specific shifts drafting."

## Experiment 2 — RoPE position effect (masked)

**Question:** if we mask the suffix so the draft's *content* conditioning matches
the target exactly, but leave the generated summary tokens at their naive
positions `D+s+n ..`, how much does α drop purely from the position offset?

**Conditions:** baseline vs masked. The magnitude of α(baseline) − α(masked) tells us how serious the
RoPE problem is and therefore how much the position-rebasing machinery (rebased) is
buying us. If this gap is ~0, the rebasing complexity may be unnecessary; if it
is large, it justifies rebased.

**Knob worth sweeping:** `n`. The position offset *is* `n`, so the RoPE effect
should grow with suffix length. Running masked at a few suffix lengths (short / our
real hashtag suffix / padded-long) turns this from one number into a curve.

## Measurement methodology

Two ways to get α; recommend doing the teacher-forced one as primary.

### Primary: teacher-forced expected acceptance (low variance)

The canonical summary supplies the **conditioning path** — the sequence of
contexts a greedy-target speculative loop actually walks. At each context we
compute the *expected* acceptance probability analytically from the two full
next-token distributions, rather than flipping the accept/reject coin.

The acceptance rate at a context is the expectation over what the draft might
propose (`t ~ q`):

```
α_context = E_{t~q}[ min(1, p(t)/q(t)) ]
          = Σ_v q(v)·min(1, p(v)/q(v))
          = Σ_v min( q(v), p(v) )        # overlap of the two distributions
          = 1 − TV(p, q)                  # one minus total-variation distance
```

This is the standard Leviathan α. Note it marginalizes over *all* tokens the
draft could propose; it is **not** `min(1, p(y_t)/q(y_t))` for the single
canonical token `y_t` (that is the integrand before the expectation, and answers
a different question — "would the target's own token be accepted").

Procedure:

1. Generate one **canonical summary** per document from the *target* (greedy, bare
   prompt). Call it `y_1..y_S`. This fixes the conditioning path only.
2. Teacher-force the **target** once over `y_1..y_S` (bare prompt) to get the full
   distribution `p(·| prompt, y_<t)` at every position `t`. Condition-independent.
3. For each condition, teacher-force the **draft** over the same `y_1..y_S` (with
   that condition's prompt / mask / positions) to get `q(·| prompt', y_<t)`.
4. Per-position `α_t = Σ_v min(p_v, q_v)`. Report mean `α_t` over positions and
   documents as the condition's α.

Implementation consequence: we need the **full softmax distributions** over the
vocab at each position from both models (logits → softmax → elementwise min →
sum), not just the gathered probability of the canonical token.

Why this and not just running the loop: only the draft distribution `q` changes
between conditions (the target `p` and the canonical path are held fixed), so the
α deltas isolate exactly the draft-side perturbation. And computing the closed-
form expectation on a fixed path removes the path-divergence noise that a
stochastic loop would inject, so small α gaps stay visible.

### Confirmation: empirical α from the real loop

Run actual speculative decoding (draft proposes `k`, target verifies, rejection
sample) and report accepted-tokens-per-round / `k`, or mean accepted run length.
Use this to confirm the teacher-forced numbers and to get a wall-clock feel. Use
the same fixed documents and a fixed seed.

### Reporting

Per condition: mean α, and the decomposition deltas above with bootstrap CIs over
documents. A small table + the α-vs-`n` curve for masked is enough for a first pass.

## Models and config

Model tiers:

- **dev** (Mac CPU correctness loop): target Qwen2.5-1.5B-Instruct, draft
  Qwen2.5-0.5B-Instruct.
- **prod** (cloud GPU, reportable numbers): target Qwen2.5-7B-Instruct, draft
  Qwen2.5-1.5B-Instruct.

Same Qwen2.5 tokenizer across sizes (required for speculation). Framework:
HuggingFace `transformers` + PyTorch; we need `position_ids` and a custom
attention mask exposed, which is why we hand-roll rather than use
`assisted_generation`. Greedy / fixed low temperature and a fixed seed so α is
not polluted by sampling noise. Use ~50–100 documents (CNN/DailyMail or XSum) for
the first pass.

## What the results would mean

- **α(visible) ≈ α(baseline):** the compound prompt is essentially free; the sidecar can ride
  on visible and most of the position-rebasing machinery is unnecessary. Strong,
  simple result.
- **α(visible) < α(baseline), and α(masked) ≈ α(baseline):** the leak is pure content conditioning, not
  position. Masking (without needing position rebasing) is the fix.
- **α(masked) < α(baseline):** RoPE mismatch is real and `n`-dependent; this directly
  motivates the position-rebasing in rebased and gives us the validation target
  (α(rebased) should recover α(baseline)).
- The two deltas rarely both being zero is itself the interesting finding for the
  writeup — it quantifies instruction-compositionality vs. positional-sensitivity
  in a small draft model.
