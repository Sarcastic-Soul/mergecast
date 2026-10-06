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


CONTEXT_FILE = ROOT / "data" / "context.parquet"
POOL_FILE = ROOT / "data" / "context_pool.parquet"
COLUMNS = FEATURES + ["number", "created_at", "merged_30d", "hours_to_merge"]
POOL_PER_REPO = 400


def build_context_file() -> Path:
    """Copy of the training table sorted by repo in small row groups, so one
    repo's rows can be read without loading the whole table (the web demo
    runs in 512MB)."""
    import pyarrow.parquet as pq

    table = pq.read_table(TABLE, columns=COLUMNS).sort_by("repo")
    pq.write_table(table, CONTEXT_FILE, row_group_size=500)
    # A fixed random sample of every repo, for the "other repos" part of the context.
    df = table.to_pandas()
    pool = df.groupby("repo", group_keys=False).apply(
        lambda g: g.sample(min(len(g), POOL_PER_REPO), random_state=0))
    pool.to_parquet(POOL_FILE, index=False)
    return CONTEXT_FILE


def _read(filters=None, columns=COLUMNS, path=CONTEXT_FILE) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not (CONTEXT_FILE.exists() and POOL_FILE.exists()):
        build_context_file()
    arrow_str = pd.StringDtype("pyarrow")
    return pq.read_table(path, columns=columns, filters=filters).to_pandas(
        types_mapper={pa.string(): arrow_str, pa.large_string(): arrow_str}.get)


@lru_cache(maxsize=1)
def known_repos() -> frozenset:
    return frozenset(_read(columns=["repo"]).repo.unique())


def repo_rows(repo: str) -> pd.DataFrame:
    return _read(filters=[("repo", "==", repo)])


@lru_cache(maxsize=1)
def others_pool() -> pd.DataFrame:
    return _read(path=POOL_FILE)


def context_table(repo: str, live_hist: pd.DataFrame, now: pd.Timestamp,
                  n_context: int, seed=0) -> pd.DataFrame:
    """Training rows + this repo's settled live history, all with known labels."""
    cutoff = now - HORIZON
    other = others_pool()
    other = other[(other.repo != repo) & (other.created_at < cutoff)]
    per_repo = max(20, n_context // max(1, other.repo.nunique()))
    other = other.groupby("repo", group_keys=False).apply(
        lambda g: g.sample(min(len(g), per_repo), random_state=seed))
    own = repo_rows(repo) if repo in known_repos() else live_hist.iloc[:0]
    own = own[own.created_at < cutoff]
    settled = live_hist[(live_hist.created_at < cutoff) | (live_hist.merged_30d == 1)]
    own = pd.concat([own, settled]).drop_duplicates(["repo", "number"])
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
    seen = pr["repo"] in known_repos()
    return Forecast(pr=pr, probs=probs, merge_prob=merge_prob,
                    what_ifs=sorted(wi, key=lambda w: -w["delta"]),
                    context_rows=len(ctx), repo_rows=int((ctx.repo == pr["repo"]).sum()),
                    repo_seen_in_training=seen)
