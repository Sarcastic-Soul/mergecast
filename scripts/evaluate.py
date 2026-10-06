"""Compare TabPFN-3.5 against LightGBM baselines on predicting PR merges.

Split: train on PRs opened before --cutoff, test on PRs opened after it
(same repos, later time). Every TabPFN API result is cached in data/preds/,
so re-running only pays for configurations that have not been run yet.

Usage:
  uv run scripts/evaluate.py --estimate        # token cost only, no API spend
  uv run scripts/evaluate.py --models lgbm lgbm_text plus thinking
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
PREDS = ROOT / "data" / "preds"

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
TARGET = "merged_30d"


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def feature_frame(df: pd.DataFrame, text=True) -> pd.DataFrame:
    X = df[CATEGORICAL + NUMERIC + (TEXT if text else [])].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    for c in NUMERIC:
        X[c] = X[c].astype(float)
    for c in TEXT if text else []:
        X[c] = X[c].fillna("").astype("string")
    return X


def split(df, cutoff, max_train, max_test, seed=0):
    cut = pd.Timestamp(cutoff, tz="UTC")
    train = df[df.created_at < cut]
    test = df[df.created_at >= cut]
    if len(train) > max_train:
        train = train.groupby("repo", group_keys=False).apply(
            lambda g: g.sample(min(len(g), max_train // df.repo.nunique() + 1), random_state=seed))
        train = train.sample(min(len(train), max_train), random_state=seed)
    if len(test) > max_test:
        test = test.sample(max_test, random_state=seed)
    return train.sort_values("created_at"), test.sort_values("created_at")


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"roc_auc": roc_auc_score(y, p), "log_loss": log_loss(y, p),
            "brier": brier_score_loss(y, p)}


# ---------------------------------------------------------------- baselines

def run_lgbm(train, test, text):
    import lightgbm as lgb
    Xtr, Xte = feature_frame(train, text=False), feature_frame(test, text=False)
    Xte = Xte.astype({c: pd.CategoricalDtype(Xtr[c].cat.categories) for c in CATEGORICAL})
    if text:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        joined = lambda d: d.title.fillna("") + " " + d.body.fillna("") + " " + d.file_paths.fillna("")
        tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        svd = TruncatedSVD(64, random_state=0)
        Ttr = svd.fit_transform(tfidf.fit_transform(joined(train)))
        Tte = svd.transform(tfidf.transform(joined(test)))
        for i in range(Ttr.shape[1]):
            Xtr[f"svd{i}"], Xte[f"svd{i}"] = Ttr[:, i], Tte[:, i]
    model = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                               min_child_samples=20, verbose=-1)
    model.fit(Xtr, train[TARGET])
    return model.predict_proba(Xte)[:, 1]


# ---------------------------------------------------------------- TabPFN API

TABPFN_CONFIGS = {
    "plus": dict(model_path="v3.5_default"),
    "fast": dict(model_path="v3.5-fast_default"),
    "thinking": dict(model_path="v3.5_default", thinking_mode=True, thinking_metric="log_loss",
                     thinking_effort="high",
                     group_col="repo", group_time_col="created_ts"),
}


def cache_key(name, train, test):
    ids = ",".join(train.repo + "#" + train.number.astype(str)) + "|" + \
          ",".join(test.repo + "#" + test.number.astype(str))
    return f"{name}_{hashlib.sha1(ids.encode()).hexdigest()[:10]}"


def run_tabpfn(name, train, test, estimate_only=False):
    from tabpfn_client import TabPFNClassifier, estimate_cost
    cfg = TABPFN_CONFIGS[name]
    Xtr, Xte = feature_frame(train), feature_frame(test)
    if estimate_only:
        est = estimate_cost(X_train=Xtr, X_test=Xte, task="classification",
                            model_path=cfg["model_path"],
                            thinking_mode=cfg.get("thinking_mode", False))
        return est
    key = cache_key(name, train, test)
    path = PREDS / f"{key}.npy"
    if path.exists():
        print(f"  cached {path.name}")
        return np.load(path)
    clf = TabPFNClassifier(**cfg)
    clf.fit(Xtr, train[TARGET])
    p = clf.predict_proba(Xte)[:, 1]
    PREDS.mkdir(parents=True, exist_ok=True)
    np.save(path, p)
    clf.save_model(str(PREDS / f"{key}.model.json"))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lgbm", "lgbm_text"])
    ap.add_argument("--cutoff", default="2026-07-01")
    ap.add_argument("--max-train", type=int, default=8000)
    ap.add_argument("--max-test", type=int, default=2000)
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args()
    load_env()

    df = pd.read_parquet(ROOT / "data" / "prs.parquet")
    train, test = split(df, args.cutoff, args.max_train, args.max_test)
    print(f"train {len(train)} ({train[TARGET].mean():.2f} merged), "
          f"test {len(test)} ({test[TARGET].mean():.2f} merged), "
          f"{df.repo.nunique()} repos")

    if args.estimate:
        for name in TABPFN_CONFIGS:
            try:
                print(name, run_tabpfn(name, train, test, estimate_only=True))
            except Exception as e:  # estimate signature differs across client versions
                print(name, "estimate failed:", e)
        return

    results, preds = {}, {}
    for name in args.models:
        print(f"running {name}")
        if name == "lgbm":
            p = run_lgbm(train, test, text=False)
        elif name == "lgbm_text":
            p = run_lgbm(train, test, text=True)
        else:
            p = run_tabpfn(name, train, test)
        preds[name] = p
        results[name] = metrics(test[TARGET].values, p)
        print(f"  {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in results[name].items()))

    # Merge into earlier runs so models can be run one at a time.
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    mpath, ppath = out / "metrics.json", out / "test_preds.parquet"
    allm = json.loads(mpath.read_text()) if mpath.exists() else {}
    allm.update(results)
    mpath.write_text(json.dumps(allm, indent=2))
    tp = pd.DataFrame({"repo": test.repo.values, "number": test.number.values,
                       "created_at": test.created_at.values, "y": test[TARGET].values})
    if ppath.exists():
        old = pd.read_parquet(ppath)
        if len(old) == len(tp) and (old.number.values == tp.number.values).all():
            tp = old
    for name, p in preds.items():
        tp[name] = p
    tp.to_parquet(ppath)

if __name__ == "__main__":
    main()
