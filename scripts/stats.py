"""Bootstrap confidence intervals and calibration from saved test predictions.

Reads results/test_preds.parquet (written by evaluate.py), spends no API tokens,
and writes results/stats.json.

Usage: uv run scripts/stats.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MODELS = ["lgbm", "lgbm_text", "plus", "thinking"]
PAIRS = [("plus", "lgbm_text"), ("plus", "lgbm"), ("thinking", "lgbm_text")]
N_BOOT = 2000
N_BINS = 10
EPS = 1e-6


def metrics(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return roc_auc_score(y, p), log_loss(y, p, labels=[0, 1])


def ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return [float(lo), float(hi)]


def calibration(y, p):
    """Equal-count bins: mean predicted vs observed rate, plus ECE."""
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, N_BINS)
    rows, ece = [], 0.0
    for b in bins:
        pred, obs = float(p[b].mean()), float(y[b].mean())
        rows.append({"n": int(len(b)), "predicted": pred, "observed": obs})
        ece += len(b) / len(p) * abs(pred - obs)
    return rows, float(ece)


def main():
    df = pd.read_parquet(RES / "test_preds.parquet")
    models = [m for m in MODELS if m in df]
    y = df["y"].to_numpy()
    P = {m: df[m].to_numpy() for m in models}

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))
    boot = {m: np.array([metrics(y[i], P[m][i]) for i in idx]) for m in models}

    out = {"n_test": int(len(y)), "n_boot": N_BOOT, "models": {}, "pairs": {}}
    for m in models:
        auc, ll = metrics(y, P[m])
        bins, ece = calibration(y, P[m])
        out["models"][m] = {
            "roc_auc": float(auc), "roc_auc_ci": ci(boot[m][:, 0]),
            "log_loss": float(ll), "log_loss_ci": ci(boot[m][:, 1]),
            "ece": ece, "calibration": bins,
        }
    for a, b in PAIRS:
        if a not in boot or b not in boot:
            continue
        d_auc = boot[a][:, 0] - boot[b][:, 0]
        d_ll = boot[a][:, 1] - boot[b][:, 1]
        out["pairs"][f"{a}_vs_{b}"] = {
            "auc_diff": float(out["models"][a]["roc_auc"] - out["models"][b]["roc_auc"]),
            "auc_diff_ci": ci(d_auc), "auc_win_share": float((d_auc > 0).mean()),
            "log_loss_diff": float(out["models"][a]["log_loss"] - out["models"][b]["log_loss"]),
            "log_loss_diff_ci": ci(d_ll), "log_loss_win_share": float((d_ll < 0).mean()),
        }

    (RES / "stats.json").write_text(json.dumps(out, indent=2))
    for m, r in out["models"].items():
        print(f"{m:10s} AUC {r['roc_auc']:.4f} [{r['roc_auc_ci'][0]:.4f}, {r['roc_auc_ci'][1]:.4f}]  "
              f"log loss {r['log_loss']:.4f} [{r['log_loss_ci'][0]:.4f}, {r['log_loss_ci'][1]:.4f}]  "
              f"ECE {r['ece']:.4f}")
    for k, r in out["pairs"].items():
        print(f"{k:22s} AUC diff {r['auc_diff']:+.4f} [{r['auc_diff_ci'][0]:+.4f}, {r['auc_diff_ci'][1]:+.4f}] "
              f"wins {r['auc_win_share']:.1%}  log loss diff {r['log_loss_diff']:+.4f} "
              f"[{r['log_loss_diff_ci'][0]:+.4f}, {r['log_loss_diff_ci'][1]:+.4f}] wins {r['log_loss_win_share']:.1%}")


if __name__ == "__main__":
    main()
