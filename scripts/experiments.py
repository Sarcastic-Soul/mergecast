"""Extra experiments beyond the main benchmark (scripts/evaluate.py).

  coldstart  Hold out whole repos. The model sees every other repo plus only
             the N most recent PRs of each held-out repo opened before the
             cutoff, then predicts that repo's later PRs. Shows how quickly
             in-context learning adapts to a repo it has never seen.
  text       Plus with and without the free-text columns (title, body, paths).
  timing     The 4-way outcome (merged <1 day, 1-7 days, 7-30 days, not in
             30 days), scored with multi-class log loss. Plus and Thinking.
  context    How big the context should be for live forecasts. Each repo's
             test PRs are predicted the way the live tool does it: all of the
             repo's own earlier PRs plus an even sample of N PRs from the
             other repos.

Results land in results/<experiment>.json. TabPFN calls are cached in data/preds/.

Usage: uv run scripts/experiments.py coldstart text timing context
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import CATEGORICAL, NUMERIC, PREDS, ROOT, TARGET, TEXT, feature_frame, load_env  # noqa: E402

from mergecast.predict import outcome_class  # noqa: E402

CUTOFF = pd.Timestamp("2026-07-01", tz="UTC")
HELDOUT = ["python/cpython", "vercel/next.js", "kubernetes/kubernetes",
           "ollama/ollama", "grafana/grafana", "astral-sh/ruff"]
SHOTS = [0, 25, 100, 400, 800]
OUT = ROOT / "results"


THINKING = dict(thinking_mode=True, thinking_metric="log_loss", thinking_effort="high",
                group_col="repo", group_time_col="created_ts")
CONTEXT_SIZES = [1000, 3000, 5000, 10000]


def cached_tabpfn(tag, train, test, y, text=True, classes=False, **cfg):
    """Fit TabPFN-3.5 (Plus unless cfg says otherwise) on train, predict test;
    cache by tag + row ids."""
    from tabpfn_client import TabPFNClassifier
    ids = ",".join(train.repo + "#" + train.number.astype(str)) + "|" + \
          ",".join(test.repo + "#" + test.number.astype(str))
    path = PREDS / f"{tag}_{hashlib.sha1(ids.encode()).hexdigest()[:10]}.npy"
    if path.exists():
        return np.load(path)
    both = feature_frame(pd.concat([train, test], ignore_index=True), text=text)
    Xtr, Xte = both.iloc[:len(train)], both.iloc[len(train):]
    clf = TabPFNClassifier(model_path="v3.5_default", **cfg)
    clf.fit(Xtr, y)
    P = clf.predict_proba(Xte)
    P = P if classes else P[:, 1]
    PREDS.mkdir(parents=True, exist_ok=True)
    np.save(path, P)
    return P


def lgbm(train, test, y, classes=False):
    import lightgbm as lgb
    both = feature_frame(pd.concat([train, test], ignore_index=True), text=False)
    Xtr, Xte = both.iloc[:len(train)], both.iloc[len(train):]
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m.fit(Xtr, y)
    P = m.predict_proba(Xte)
    return P if classes else P[:, 1]


def coldstart(df):
    others = df[~df.repo.isin(HELDOUT) & (df.created_at < CUTOFF)]
    others = others.sample(min(len(others), 20000), random_state=0)
    test = df[df.repo.isin(HELDOUT) & (df.created_at >= CUTOFF)]
    test = test.groupby("repo", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 400), random_state=0))
    hist = df[df.repo.isin(HELDOUT) & (df.created_at < CUTOFF - pd.Timedelta(days=30))]
    out = {}
    for n in SHOTS:
        shots = hist.sort_values("created_at").groupby("repo").tail(n) if n else hist.iloc[:0]
        train = pd.concat([others, shots])
        res = {}
        for name, fn in [("tabpfn_plus", lambda: cached_tabpfn(f"cold{n}", train, test, train[TARGET].values)),
                         ("lgbm", lambda: lgbm(train, test, train[TARGET].values))]:
            p = fn()
            per_repo = {r: roc_auc_score(g[TARGET], p[test.repo.values == r])
                        for r, g in test.groupby("repo")}
            res[name] = {"roc_auc": roc_auc_score(test[TARGET], p),
                         "log_loss": log_loss(test[TARGET], np.clip(p, 1e-6, 1 - 1e-6)),
                         "mean_repo_auc": float(np.mean(list(per_repo.values()))),
                         "per_repo_auc": per_repo}
        out[n] = res
        print(f"  N={n:4d}  tabpfn AUC {res['tabpfn_plus']['roc_auc']:.4f} "
              f"(per-repo {res['tabpfn_plus']['mean_repo_auc']:.4f})  "
              f"lgbm AUC {res['lgbm']['roc_auc']:.4f} (per-repo {res['lgbm']['mean_repo_auc']:.4f})")
    return {"heldout": HELDOUT, "test_rows": len(test), "context_rows_other": len(others), "by_shots": out}


def split(df, n_train=30000, n_test=5000):
    train = df[df.created_at < CUTOFF]
    test = df[df.created_at >= CUTOFF]
    train = train.sample(min(len(train), n_train), random_state=0)
    test = test.sample(min(len(test), n_test), random_state=0)
    return train, test


def text_ablation(df):
    train, test = split(df)
    out = {}
    for name, text in [("with_text", True), ("without_text", False)]:
        p = cached_tabpfn(f"text_{name}", train, test, train[TARGET].values, text=text)
        out[name] = {"roc_auc": roc_auc_score(test[TARGET], p),
                     "log_loss": log_loss(test[TARGET], np.clip(p, 1e-6, 1 - 1e-6))}
        print(f"  {name}: AUC {out[name]['roc_auc']:.4f}  log loss {out[name]['log_loss']:.4f}")
    return out


def timing(df):
    train, test = split(df)
    ytr, yte = outcome_class(train).values, outcome_class(test).values
    out = {"class_share_test": np.bincount(yte, minlength=4).tolist()}
    prior = np.bincount(ytr, minlength=4) / len(ytr)
    out["base_rate"] = {"log_loss": log_loss(yte, np.tile(prior, (len(yte), 1)), labels=range(4))}
    for name, fn in [("tabpfn_plus", lambda: cached_tabpfn("timing", train, test, ytr, classes=True)),
                     ("tabpfn_thinking", lambda: cached_tabpfn("timing_thinking", train, test, ytr,
                                                               classes=True, **THINKING)),
                     ("lgbm", lambda: lgbm(train, test, ytr, classes=True))]:
        P = fn()
        out[name] = {"log_loss": log_loss(yte, P, labels=range(4)),
                     "accuracy": float((P.argmax(1) == yte).mean())}
    for k in ("base_rate", "tabpfn_plus", "tabpfn_thinking", "lgbm"):
        print(f"  {k}: " + ", ".join(f"{m}={v:.4f}" for m, v in out[k].items()))
    return out


def context_size(df):
    from concurrent.futures import ThreadPoolExecutor
    train, test = split(df)
    yte = outcome_class(test).values
    out = {}
    for n in CONTEXT_SIZES:
        def one(repo):
            own = train[train.repo == repo]
            other = train[train.repo != repo]
            per_repo = max(20, n // other.repo.nunique())
            other = other.groupby("repo", group_keys=False).apply(
                lambda g: g.sample(min(len(g), per_repo), random_state=0))
            other = other.sample(min(len(other), n), random_state=0)
            ctx = pd.concat([other, own])
            rows = test[test.repo == repo]
            P = cached_tabpfn(f"context{n}", ctx, rows, outcome_class(ctx).values, classes=True)
            return rows.index, P, len(ctx)
        with ThreadPoolExecutor(4) as pool:
            parts = list(pool.map(one, sorted(test.repo.unique())))
        P = pd.DataFrame(np.vstack([p for _, p, _ in parts]),
                         index=np.concatenate([i for i, _, _ in parts])).loc[test.index].values
        merge = 1 - P[:, 3]
        out[str(n)] = {"roc_auc": roc_auc_score(test[TARGET], merge),
                       "log_loss": log_loss(test[TARGET], np.clip(merge, 1e-6, 1 - 1e-6)),
                       "timing_log_loss": log_loss(yte, P, labels=range(4)),
                       "mean_context_rows": float(np.mean([c for _, _, c in parts]))}
        print(f"  N={n:>6}: " + ", ".join(f"{m}={v:.4f}" for m, v in out[str(n)].items()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+", choices=["coldstart", "text", "timing", "context"])
    args = ap.parse_args()
    load_env()
    df = pd.read_parquet(ROOT / "data" / "prs.parquet")
    OUT.mkdir(exist_ok=True)
    for name in args.experiments:
        print(f"== {name}")
        res = {"coldstart": coldstart, "text": text_ablation, "timing": timing,
               "context": context_size}[name](df)
        (OUT / f"{name}.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
