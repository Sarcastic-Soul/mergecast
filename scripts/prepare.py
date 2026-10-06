"""Turn raw PR JSONL into a modeling table: data/prs.parquet.

See mergecast/features.py for the features and labels.

Usage: uv run scripts/prepare.py
"""

import json
from pathlib import Path

import pandas as pd

from mergecast.features import build_table

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "prs.parquet"


def load_raw() -> pd.DataFrame:
    rows = []
    for f in sorted(RAW.glob("*.jsonl")):
        for line in f.open():
            d = json.loads(line)
            if "_done" not in d:
                rows.append(d)
    return pd.DataFrame(rows).drop_duplicates(["repo", "number"])


def main():
    df = load_raw()
    print(f"raw PRs: {len(df)} from {df.repo.nunique()} repos")
    df = build_table(df)
    df.to_parquet(OUT, index=False)
    print(f"kept {len(df)} PRs, merge rate {df.merged_30d.mean():.3f}")
    print(df.groupby("repo")["merged_30d"].agg(["size", "mean"]).round(3).to_string())


if __name__ == "__main__":
    main()
