"""Shared prompt definitions for the sidecar experiments.

These MUST be identical between canonical-summary generation
(canonical_summaries.py) and acceptance teacher-forcing (acceptance.py): the
acceptance math compares p_target/p_draft on the exact same token positions, so
any drift in the prompt text would misalign the sequences and corrupt alpha.
Hence a single source of truth.
"""

SUMMARY_SYSTEM = "You are a concise assistant."

SUMMARY_INSTRUCTION = (
    "Summarize the document in 3 bullets. Keep each bullet under 18 words."
)

# The hashtag sidecar suffix, appended AFTER the summary instruction. Used by the
# E2 and Variation-G acceptance conditions (not by the bare baseline, and not by
# canonical summary generation). "above" refers to the summary that will sit in
# the KV cache by the time this suffix is active.
HASHTAG_SUFFIX = (
    "Then output 3-5 short topical hashtags for the summary above, "
    "as a JSON array of lowercase hyphenated strings."
)

# An irrelevant compound suffix, for the control condition that isolates "any
# future instruction shifts drafting" from "hashtag-specific shifts drafting".
FRENCH_SUFFIX = "Then translate the summary above into French."


def build_summary_messages(document: str, suffix: str | None = None) -> list[dict]:
    """Chat messages for the summarization turn.

    Layout follows the research note: [document][summary instruction][suffix],
    with the suffix LAST so it is a contiguous trailing block (maskable for
    Variation G) and so "the summary above" reads correctly once generation
    begins after it.

    suffix=None -> the bare summary prompt (target's actual prompt, and the
    baseline draft prompt). A non-None suffix appends a compound instruction
    after the summary instruction (the E2 / control draft prompts).
    """
    user = f"Document:\n{document}\n\n{SUMMARY_INSTRUCTION}"
    if suffix is not None:
        user = f"{user} {suffix}"
    return [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user", "content": user},
    ]
