"""One-time fetch: snapshot a fixed subset of CNN/DailyMail to local JSONL.

CNN/DailyMail is hosted on the HF Hub as parquet, so there is no plain-text
curl. This script streams the test split (no full ~1.2GB download), takes the
first N articles in their stable split order, and writes them to disk. The
snapshot file is the frozen corpus for all experiments; downstream code reads it
with no network access. Re-run only to change N.

    python fetch_documents.py                 # 100 articles -> data/cnndm_test.jsonl
    python fetch_documents.py --n 200 --out data/cnndm_test.jsonl

The dev/test subset is just the first K rows of the same file (see documents.py),
so it is always a true subset of the full set.
"""

import argparse
import json
import os

from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100, help="number of articles to snapshot")
    ap.add_argument("--out", default="data/cnndm_test.jsonl", help="output JSONL path")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # streaming=True avoids downloading the whole dataset; we take the first n in
    # the split's stable order, so the snapshot is reproducible.
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test", streaming=True)

    written = 0
    with open(args.out, "w") as f:
        for row in ds:
            # we only need the article text as a summarization prompt; gold
            # highlights are irrelevant for acceptance-rate work.
            record = {"id": row["id"], "article": row["article"]}
            f.write(json.dumps(record) + "\n")
            written += 1
            if written >= args.n:
                break

    print(f"wrote {written} articles to {args.out}")


if __name__ == "__main__":
    main()
