#!/usr/bin/env python3
"""
Stage 4: embedding-free routers (kNN, K-Means) trained on exactly the anchors of each LOCUS run.

For every finished run runs/main/a<A>_s<S> the anchor queries (meta.json["anchors_idx"]) and their
labels for all 112 models are the only training data, as for LOCUS. Both routers output a probability
matrix p [N, 112] (paper App. A):
  kNN      p[n, m] = mean correctness of model m on the k anchors nearest to query n (cosine, mpnet)
  K-Means  p[n, m] = mean correctness of model m in the cluster of query n (K-Means on anchor encodings)
k / K are not stated in the paper: they are chosen per run by val hit@1 (ties -> smaller value) and the
test numbers at that value are reported. Metrics come from common.metrics (same code as LOCUS, including
argsort tie-breaking, which matters here because many models share the same mean).

Outputs (results/baselines/):
  knn_kmeans_grid.csv     every run x method x k/K: val and test hit@1 / correctness accuracy
  knn_kmeans_runs.csv     selected k/K per run, overall and per-dataset test metrics
  knn_kmeans_summary.md   mean +- std over seeds vs. paper Tables 2, 5, 6

Usage (project root):  python repro/baselines/knn_kmeans.py [--anchors 256 512 1024] [--seeds 42]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")  # joblib otherwise shells out to wmic to count cores
from sklearn.cluster import KMeans  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from repro.common import RESULTS, RUNS, load_data, meta_anchors, metrics, summarize  # noqa: E402

OUT = RESULTS / "baselines"
KS_KNN = [1, 3, 5, 10, 20, 50, 100, 200, 500]
KS_KMEANS = [2, 4, 8, 16, 32, 64]

DATASETS = ["mathqa", "logiqa", "medmcqa", "piqa", "truthfulqa_mc1", "MMLU", "gsm8k", "GPQA", "asdiv", "social_iqa"]
PAPER_COLS = ["MathQA", "LogiQA", "MedQA", "PIQA", "TruthQA", "MMLU", "GSM8k", "GPQA", "ASDiv", "SoQA"]
# Paper Tables 5 (routing) and 6 (correctness): [overall, *PAPER_COLS]
PAPER = {
    ("knn", "routing"): {
        256: [60.50, 59.92, 47.06, 75.42, 85.07, 41.89, 78.27, 89.29, 29.00, 10.10, 30.86],
        512: [61.37, 57.38, 45.10, 68.64, 85.82, 48.65, 80.48, 89.29, 26.80, 29.80, 29.01],
        1024: [62.90, 57.81, 45.10, 72.60, 86.57, 55.41, 82.09, 89.29, 21.60, 39.39, 37.04],
    },
    ("kmeans", "routing"): {
        256: [61.33, 58.65, 49.02, 62.99, 86.57, 52.70, 84.72, 89.29, 25.80, 10.10, 31.48],
        512: [64.47, 57.81, 52.94, 75.42, 87.31, 47.30, 81.32, 89.29, 23.60, 57.58, 37.65],
        1024: [66.20, 56.12, 49.02, 74.86, 87.31, 52.70, 85.14, 89.29, 24.80, 60.10, 37.65],
    },
    ("knn", "correctness"): {
        256: [60.60, 65.25, 65.35, 59.73, 26.33, 66.31, 52.08, 62.65, 74.43, 88.25, 63.94],
        512: [67.90, 60.47, 55.02, 60.10, 78.38, 50.86, 63.82, 73.92, 78.21, 89.84, 65.90],
        1024: [69.27, 62.81, 59.58, 60.55, 76.32, 65.82, 64.54, 74.00, 78.93, 94.19, 67.48],
    },
    ("kmeans", "correctness"): {
        256: [66.34, 66.15, 55.74, 59.39, 65.51, 35.34, 61.33, 58.45, 78.76, 95.57, 67.74],
        512: [68.32, 66.09, 51.98, 61.29, 79.32, 44.63, 63.25, 73.71, 76.37, 96.69, 67.47],
        1024: [69.57, 65.59, 55.57, 61.20, 79.64, 62.90, 64.16, 74.04, 78.67, 96.74, 67.85],
    },
}


def knn_probs(Xa: np.ndarray, Ya: np.ndarray, X: np.ndarray, ks: list[int]) -> dict[int, np.ndarray]:
    """p_k for every k from one neighbour search. Encodings are unit-norm, so dot = cosine similarity."""
    kmax = min(max(ks), len(Xa))
    sims = X @ Xa.T                                                    # [N, A]
    nbr = np.argpartition(-sims, kmax - 1, axis=1)[:, :kmax]
    nbr = np.take_along_axis(nbr, np.argsort(-np.take_along_axis(sims, nbr, 1), axis=1, kind="stable"), 1)
    cum = np.cumsum(Ya[nbr], axis=1)                                   # [N, kmax, M]
    return {k: cum[:, k - 1] / k for k in ks if k <= kmax}


def kmeans_probs(Xa: np.ndarray, Ya: np.ndarray, Xs: dict[str, np.ndarray], K: int, seed: int) -> dict[str, np.ndarray]:
    km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(Xa)
    centers_p = np.stack([Ya[km.labels_ == c].mean(axis=0) for c in range(K)])  # [K, M]
    return {s: centers_p[km.predict(X)] for s, X in Xs.items()}


def overall(out: dict) -> dict:
    return {"hit1": out["rt_overall"]["hit@1"] * 100, "acc": out["bp_overall"]["correctness_pred_acc"] * 100}


def run_one(data, A: int, seed: int, anchors: np.ndarray) -> tuple[list[dict], list[dict]]:
    Xa = data.train.X[anchors]
    Ya = data.train.Y[anchors].astype(np.float32)
    splits = {"val": data.val, "test": data.test}
    grid, chosen = [], []

    candidates = {"knn": {}, "kmeans": {}}
    probs_val = knn_probs(Xa, Ya, data.val.X, KS_KNN)
    probs_test = knn_probs(Xa, Ya, data.test.X, KS_KNN)
    for k in probs_val:
        candidates["knn"][k] = {"val": probs_val[k], "test": probs_test[k]}
    for K in KS_KMEANS:
        if K <= len(Xa):
            candidates["kmeans"][K] = kmeans_probs(Xa, Ya, {s: d.X for s, d in splits.items()}, K, seed)

    for method, by_param in candidates.items():
        outs = {}
        for param, p in by_param.items():
            outs[param] = {s: metrics(p[s], d.Y, d.category, kind="probs") for s, d in splits.items()}
            v, t = overall(outs[param]["val"]), overall(outs[param]["test"])
            grid.append({"anchors": A, "seed": seed, "method": method, "param": param,
                         "val_hit1": v["hit1"], "val_acc": v["acc"], "test_hit1": t["hit1"], "test_acc": t["acc"]})
        # max() keeps the first maximum: on equal val hit@1 the smaller k/K wins
        best = max(outs, key=lambda q: outs[q]["val"]["rt_overall"]["hit@1"])
        s = summarize(outs[best]["test"])
        row = {"anchors": A, "seed": seed, "method": method, "param": best,
               "val_hit1": overall(outs[best]["val"])["hit1"],
               "routing": s["overall"]["routing"]["hit@1"] * 100,
               "correctness": s["overall"]["correctness_prediction"]["correctness_pred_acc"] * 100,
               "oracle_test_hit1": max(overall(o["test"])["hit1"] for o in outs.values())}
        for r in s["by_dataset"]["routing"]:
            row[f"rt_{r['dataset']}"] = r["hit@1"] * 100
        for r in s["by_dataset"]["correctness_prediction"]:
            row[f"bp_{r['dataset']}"] = r["correctness_pred_acc"] * 100
        chosen.append(row)
    return grid, chosen


def to_md(df: pd.DataFrame) -> str:
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.2f}"
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def summary_tables(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t2 = []
    for (method, A), g in runs.groupby(["method", "anchors"]):
        row = {"method": method, "anchors": A, "seeds": len(g), "k/K": ",".join(map(str, g["param"]))}
        for metric in ("routing", "correctness"):
            v = g[metric]
            p = PAPER[(method, metric)].get(A, [np.nan])[0]
            row[f"{metric}"] = v.mean()
            row[f"{metric}_std"] = v.std(ddof=1) if len(v) > 1 else np.nan
            row[f"{metric}_paper"] = p
            row[f"{metric}_diff"] = v.mean() - p
        t2.append(row)
    t56 = []
    for (method, A), g in runs.groupby(["method", "anchors"]):
        for metric, prefix in (("routing", "rt_"), ("correctness", "bp_")):
            paper = PAPER[(method, metric)].get(A)
            if paper is None:
                continue
            row = {"method": method, "anchors": A, "metric": metric}
            for ds, col, p in zip(DATASETS, PAPER_COLS, paper[1:]):
                ours = g[prefix + ds].mean()
                row[col] = f"{ours:.1f} ({ours - p:+.1f})"
            t56.append(row)
    return pd.DataFrame(t2), pd.DataFrame(t56)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--anchors", type=int, nargs="*", default=None, help="default: every finished run")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=4, help="BLAS/OpenMP threads (leave CPU for a training run)")
    args = ap.parse_args()

    run_dirs = []
    for d in sorted((RUNS / "main").glob("a*_s*")):
        m = re.fullmatch(r"a(\d+)_s(\d+)", d.name)
        if not m or not (d / "meta.json").exists():
            continue
        A, seed = int(m[1]), int(m[2])
        if (args.anchors and A not in args.anchors) or (args.seeds and seed not in args.seeds):
            continue
        run_dirs.append((A, seed, d))
    if not run_dirs:
        raise SystemExit("no matching runs under runs/main")

    data = load_data()
    grid, runs = [], []
    with threadpool_limits(args.threads):
        for A, seed, d in run_dirs:
            meta = json.loads((d / "meta.json").read_text())
            if meta["emb_col"] != data.emb_col or meta.get("model_indices", {}).get("train"):
                raise SystemExit(f"{d.name}: expected the default encoder and all 112 models")
            t0 = time.time()
            g, r = run_one(data, A, seed, meta_anchors(meta))
            grid += g
            runs += r
            msg = " | ".join(f"{x['method']} {x['param']:>3}: test hit@1 {x['routing']:.2f} acc {x['correctness']:.2f}" for x in r)
            print(f"{d.name:<11} {msg}  ({time.time() - t0:.0f} s)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    grid, runs = pd.DataFrame(grid), pd.DataFrame(runs)
    t2, t56 = summary_tables(runs)
    grid.to_csv(OUT / "knn_kmeans_grid.csv", index=False)
    runs.to_csv(OUT / "knn_kmeans_runs.csv", index=False)
    tuning = grid.groupby(["method", "anchors", "param"])[["val_hit1", "test_hit1", "test_acc"]].mean().reset_index()
    md = [
        "# Stage 4: kNN and K-Means routers", "",
        "Trained on the anchors of each LOCUS run in `runs/main`; k/K chosen per run by val hit@1. Test metrics, %.", "",
        "## Table 2 (overall): mean over seeds vs. paper", "", to_md(t2), "",
        "## Tables 5-6 (per dataset): mean over seeds (difference to paper)", "", to_md(t56), "",
        "## Tuning curves: mean over seeds", "", to_md(tuning), "",
    ]
    (OUT / "knn_kmeans_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:7]))


if __name__ == "__main__":
    main()
