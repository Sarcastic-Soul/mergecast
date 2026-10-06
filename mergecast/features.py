"""Feature building shared by training (scripts/prepare.py) and live scoring.

Every feature describes a PR as it looked when it was opened, or the repo and
author as they looked before that moment.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

TEXT_MAX = 2000  # Plus allows 2,500 chars per text value
HORIZON = pd.Timedelta(days=30)

TEST_RE = re.compile(r"(^|/)(tests?|testing|__tests__|spec)(/|$)|(_test|\.test|\.spec|test_)[^/]*$", re.I)
DOC_RE = re.compile(r"(^|/)(docs?|documentation)(/|$)|\.(md|rst|txt|adoc)$", re.I)
ISSUE_RE = re.compile(r"\b(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\s+(?:#\d+|https://github\.com/\S+/issues/\d+)", re.I)
COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
PREFIX_RE = re.compile(r"^\s*(\[[^\]]{1,20}\]|[a-zA-Z]{2,12}(\([^)]{0,30}\))?!?:|gh-\d+:|bpo-\d+:)")


def clean_body(body: str) -> str:
    body = COMMENT_RE.sub("", body or "")  # drop untouched PR template hints
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body[:TEXT_MAX]


def title_prefix(title: str) -> str:
    m = PREFIX_RE.match(title or "")
    if not m:
        return "none"
    p = m.group(1).lower().strip("[]:! ")
    p = re.sub(r"\(.*\)", "", p)
    return re.sub(r"\d+", "N", p)[:20]


def file_features(files) -> dict:
    paths = [f[0] for f in files]
    if not paths:
        return dict(file_paths="", touches_tests=False, docs_only=False,
                    n_top_dirs=0, main_ext="none", frac_test_files=0.0)
    tests = [bool(TEST_RE.search(p)) for p in paths]
    docs = [bool(DOC_RE.search(p)) for p in paths]
    exts = [Path(p).suffix.lower() or "noext" for p in paths]
    tops = {p.split("/")[0] for p in paths}
    return dict(
        file_paths=" ".join(paths)[:TEXT_MAX],
        touches_tests=any(tests),
        docs_only=all(docs),
        n_top_dirs=len(tops),
        main_ext=pd.Series(exts).mode().iloc[0],
        frac_test_files=float(np.mean(tests)),
    )


def history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Repo and author track record using only PRs resolved before each PR opened.

    A past PR's merged_30d label is only known once 30 days have passed (or it
    merged earlier), so we only count PRs whose outcome was settled by then.
    The repo rate uses PRs opened 30-60 days earlier: all of them are settled,
    so the rate is not tilted toward fast merges. Author history looks back
    90 days, the same reach the live scorer has (see mergecast/github.py).
    """
    df = df.copy()
    df["settled_at"] = np.where(
        df["merged_30d"] == 1, df["merged_at"], df["created_at"] + HORIZON
    )
    df["settled_at"] = pd.to_datetime(df["settled_at"], utc=True)

    out = {k: np.full(len(df), np.nan) for k in (
        "repo_recent_merge_rate", "repo_recent_n", "repo_recent_median_hours",
        "author_prior_prs", "author_prior_merge_rate")}
    repo_lo, repo_hi = np.timedelta64(60, "D"), np.timedelta64(30, "D")
    author_lookback = np.timedelta64(90, "D")

    for repo, g in df.groupby("repo"):
        settled = g.sort_values("settled_at")
        s_time = settled["settled_at"].values
        s_created = settled["created_at"].values
        s_label = settled["merged_30d"].values
        s_hours = settled["hours_to_merge"].values
        s_author = settled["author"].values
        for i, row in g.iterrows():
            t = row["created_at"].to_datetime64()
            known = s_time < t
            recent = (s_created >= t - repo_lo) & (s_created < t - repo_hi)
            n = recent.sum()
            out["repo_recent_n"][i] = n
            if n >= 5:
                out["repo_recent_merge_rate"][i] = s_label[recent].mean()
                h = s_hours[recent & (s_label == 1)]
                if len(h):
                    out["repo_recent_median_hours"][i] = np.median(h)
            mine = known & (s_author == row["author"]) & (s_created >= t - author_lookback)
            out["author_prior_prs"][i] = mine.sum()
            if mine.sum():
                out["author_prior_merge_rate"][i] = s_label[mine].mean()
    for k, v in out.items():
        df[k] = v
    return df.drop(columns="settled_at")


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    """Raw PR rows (as written by scripts/collect.py) -> modeling table.

    Labels: merged_30d (merged within 30 days of opening) and hours_to_merge.
    Rows that are still open get merged_30d=0; they only count as settled
    history once 30 days have passed.
    """
    for c in ("created_at", "merged_at", "closed_at"):
        df[c] = pd.to_datetime(df[c], utc=True)
    df = df.sort_values("created_at").reset_index(drop=True)
    df = df[df["author_type"] != "Bot"]
    df = df[~df["author"].fillna("").str.contains(r"\[bot\]|dependabot|renovate", case=False)]

    df["merged_30d"] = (
        df["merged_at"].notna() & (df["merged_at"] - df["created_at"] <= HORIZON)
    ).astype(int)
    df["hours_to_merge"] = (df["merged_at"] - df["created_at"]).dt.total_seconds() / 3600

    # GitHub reports the author's association as of today, and a merged PR
    # turns its author into a CONTRIBUTOR. Only the maintainer roles are stable
    # enough to use; everyone else is "external" and prior contributions come
    # from the leak-free author history features below.
    df["author_association"] = df["author_association"].where(
        df["author_association"].isin(["OWNER", "MEMBER", "COLLABORATOR"]), "EXTERNAL")

    base_mode = df.groupby("repo")["base_ref"].agg(lambda s: s.mode().iloc[0])
    feats = pd.DataFrame([file_features(f) for f in df["files"]], index=df.index)
    df = pd.concat([df, feats], axis=1)
    df["body"] = df["body"].map(clean_body)
    df["title_prefix"] = df["title"].map(title_prefix)
    df["base_is_default"] = df["base_ref"] == df["repo"].map(base_mode)
    df["log_churn"] = np.log1p(df["additions"] + df["deletions"])
    df["body_len"] = df["body"].str.len()
    df["title_len"] = df["title"].str.len()
    df["links_issue"] = df["body"].str.contains(ISSUE_RE)
    df["has_checklist"] = df["body"].str.contains(r"- \[[ xX]\]")
    df["hour_utc"] = df["created_at"].dt.hour
    df["weekday"] = df["created_at"].dt.weekday
    df["created_ts"] = df["created_at"].astype("int64") // 10**9

    df = history_features(df.reset_index(drop=True))
    return df.drop(columns=["files", "labels", "closed_at", "author_type"])
