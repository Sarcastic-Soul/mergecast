"""Score a live GitHub PR with TabPFN-3.5.

No model is trained ahead of time. Each forecast puts a sample of the training
table, plus the PR's own repo history fetched live, into TabPFN's context and
predicts in one call. A repo the model has never seen still gets a forecast
tuned to it, because its own recent PRs are part of the context.

The outcome is one of four classes, so a single call gives the whole picture:
merged within a day, within a week, within 30 days, or not within 30 days.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from mergecast.features import HORIZON, build_table
from mergecast.github import GitHub

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "data" / "prs.parquet"

OUTCOMES = ["within 1 day", "1-7 days", "7-30 days", "not within 30 days"]

TEXT = ["title", "body", "file_paths"]
CATEGORICAL = ["repo", "author_association", "title_prefix", "main_ext"]
NUMERIC = [
    "additions", "deletions", "changed_files", "log_churn", "is_cross_repo",
    "base_is_default", "touches_tests", "docs_only", "n_top_dirs",
    "frac_test_files", "body_len", "title_len", "links_issue", "has_checklist",
    "hour_utc", "weekday", "repo_recent_merge_rate", "repo_recent_n",
    "repo_recent_median_hours", "author_prior_prs", "author_prior_merge_rate",
    "created_ts",
]
FEATURES = CATEGORICAL + NUMERIC + TEXT


def outcome_class(df: pd.DataFrame) -> pd.Series:
    h = df["hours_to_merge"]
    cls = np.select([df["merged_30d"] == 0, h <= 24, h <= 24 * 7], [3, 0, 1], default=2)
    return pd.Series(cls, index=df.index)


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype(str).astype("category")
    for c in NUMERIC:
        X[c] = X[c].astype(float)
    for c in TEXT:
        X[c] = X[c].fillna("").astype("string")
    return X


# What-if edits: things an author can actually change. Each returns the
# fields to overwrite on a copy of the PR's row.
WHAT_IFS = {
    "Link the issue it fixes": lambda r: dict(
        links_issue=True, body=r["body"] + "\n\nFixes #123"),
    "Add or update tests": lambda r: dict(
        touches_tests=True, frac_test_files=max(float(r["frac_test_files"]), 0.3)),
    "Split it into a PR half the size": lambda r: dict(
        additions=r["additions"] // 2, deletions=r["deletions"] // 2,
        changed_files=max(1, r["changed_files"] // 2),
        log_churn=np.log1p((r["additions"] + r["deletions"]) // 2)),
}


def apply_edit(row: pd.Series, edit) -> pd.Series:
    row = row.copy()
    for k, v in edit(row).items():
        row[k] = v
    return row


@dataclass
class Forecast:
    pr: dict
    probs: dict
    merge_prob: float
    what_ifs: list = field(default_factory=list)
    context_rows: int = 0
    repo_rows: int = 0
    repo_seen_in_training: bool = False


@lru_cache(maxsize=1)
def training_table() -> pd.DataFrame:
    """The shipped feature table, loaded once with only the columns we use."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = FEATURES + ["number", "created_at", "merged_30d", "hours_to_merge"]
    # Arrow-backed strings keep the text columns small (the web demo runs in 512MB).
    arrow_str = pd.StringDtype("pyarrow")
    return pq.read_table(TABLE, columns=cols).to_pandas(
        types_mapper={pa.string(): arrow_str, pa.large_string(): arrow_str}.get,
        self_destruct=True, split_blocks=True)


def context_table(repo: str, live_hist: pd.DataFrame, now: pd.Timestamp,
                  n_context: int, seed=0) -> pd.DataFrame:
    """Training rows + this repo's settled live history, all with known labels."""
    base = training_table()
    base = base[base.created_at < now - HORIZON]
    other = base[base.repo != repo]
    per_repo = max(20, n_context // max(1, other.repo.nunique()))
    other = other.groupby("repo", group_keys=False).apply(
        lambda g: g.sample(min(len(g), per_repo), random_state=seed))
    settled = live_hist[(live_hist.created_at < now - HORIZON) | (live_hist.merged_30d == 1)]
    own = pd.concat([base[base.repo == repo], settled]).drop_duplicates(["repo", "number"])
    return pd.concat([other.sample(min(len(other), n_context), random_state=seed), own])


def forecast(url: str, n_context: int = 3000, what_ifs: bool = True,
             model_path: str = "v3.5_default", progress=lambda step, detail="": None) -> Forecast:
    from tabpfn_client import TabPFNClassifier

    gh = GitHub()
    progress("pr", "Reading the pull request")
    pr = gh.pull_request(url)
    created = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
    progress("history", f"Fetching {pr['repo']} PRs from the 90 days before it opened")
    hist = gh.history(pr["repo"], created)
    raw = pd.DataFrame([h for h in hist if h["number"] != pr["number"]] + [pr])
    raw["author_type"] = raw["author_type"].where(raw["number"] != pr["number"], "User")
    table = build_table(raw)
    target = table[table.number == pr["number"]].iloc[[0]]
    live_hist = table[table.number != pr["number"]]

    now = pd.Timestamp(datetime.now(timezone.utc))
    ctx = context_table(pr["repo"], live_hist, now, n_context)
    progress("context", f"{len(ctx)} PRs in context, {int((ctx.repo == pr['repo']).sum())} from this repo")
    base = target.iloc[0]
    rows = [base] + ([apply_edit(base, f) for f in WHAT_IFS.values()] if what_ifs else [])
    X_test = pd.DataFrame(rows)

    # Encode context and test rows together so categories line up.
    X_all = feature_frame(pd.concat([ctx, X_test], ignore_index=True))
    X_ctx, X_te = X_all.iloc[:len(ctx)], X_all.iloc[len(ctx):]
    progress("tabpfn", "TabPFN-3.5 is reading the context and forecasting")
    clf = TabPFNClassifier(model_path=model_path)
    clf.fit(X_ctx, outcome_class(ctx).values)
    P = clf.predict_proba(X_te)

    probs = dict(zip(OUTCOMES, P[0].round(4).tolist()))
    merge_prob = float(1 - P[0][3])
    wi = [{"change": name, "merge_prob": float(1 - p[3]), "delta": float((1 - p[3]) - merge_prob)}
          for name, p in zip(WHAT_IFS, P[1:])]
    seen = bool((training_table().repo == pr["repo"]).any())
    return Forecast(pr=pr, probs=probs, merge_prob=merge_prob,
                    what_ifs=sorted(wi, key=lambda w: -w["delta"]),
                    context_rows=len(ctx), repo_rows=int((ctx.repo == pr["repo"]).sum()),
                    repo_seen_in_training=seen)
