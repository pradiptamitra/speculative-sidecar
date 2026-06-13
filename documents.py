"""Read the frozen CNN/DailyMail snapshot produced by fetch_documents.py.

Library module (import load_documents); the __main__ block is just a smoke test.

Single source of truth for both tiers: dev/test passes a small n (e.g. 10) and
gets the first n rows; real runs pass a larger n. The small subset is therefore
always a prefix-subset of the full snapshot — same documents, held fixed.
"""

import json
import os

SNAPSHOT = os.environ.get("DOC_SNAPSHOT", "data/cnndm_test.jsonl")


def load_documents(n: int, max_chars: int | None = None) -> list[dict]:
    """Return the first n articles from the snapshot.

    max_chars optionally truncates each article (useful for the CPU dev loop,
    where a full ~2k-token forward pass is slow). None = full article.
    """
    if not os.path.exists(SNAPSHOT):
        raise FileNotFoundError(
            f"{SNAPSHOT} not found. Run: python fetch_documents.py --n {n}"
        )

    docs = []
    with open(SNAPSHOT) as f:
        for line in f:
            row = json.loads(line)
            article = row["article"]
            if max_chars is not None:
                article = article[:max_chars]
            docs.append({"id": row["id"], "article": article})
            if len(docs) >= n:
                break

    if len(docs) < n:
        raise ValueError(
            f"snapshot has {len(docs)} docs but {n} requested; re-run "
            f"fetch_documents.py --n {n}"
        )
    return docs


if __name__ == "__main__":
    sample = load_documents(3, max_chars=200)
    for d in sample:
        print(f"[{d['id']}] {d['article']!r}\n")
