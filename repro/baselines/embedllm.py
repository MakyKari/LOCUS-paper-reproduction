#!/usr/bin/env python3
"""
Stage 4: EmbedLLM and EmbedLLM+ baselines trained on exactly the anchors of each LOCUS run.

Model = TextMF from the official code (github.com/richardzhuang0412/EmbedLLM, algorithm/mf.py, 9b27630):
    p_m = P[m]                       learned model embedding, dim 232
    u   = W (phi(x) + alpha * noise) 768 -> 232, noise only in training (alpha 0.05)
    logits = classifier(p_m * u)     2 classes, cross-entropy; Adam, weight decay 1e-5, batch 2048
EmbedLLM+ ("more hidden layers in decoder", paper App. A): classifier = MLP with 1 or 2 hidden layers
(width 232, ReLU) instead of one Linear. Score for routing / correctness = logit[1] - logit[0].

Training pairs: every (anchor, model) pair of the LOCUS run (meta.json["anchors_idx"]), i.e. the same
data LOCUS sees. phi = the same mpnet encodings (the official code also uses all-mpnet-base-v2).
Protocol, same as LOCUS: after every epoch evaluate val hit@1, keep the best epoch, report test there.
Learning rate is chosen per run on val from LRS. Differences to the official script: shuffled batches
(official: shuffle=False), val-based epoch selection instead of a fixed 50 epochs.

Outputs (results/baselines/):
  embedllm_grid.csv     run x variant x lr: best epoch, val/test hit@1, test accuracy
  embedllm_runs.csv     selected lr per run and variant: overall and per-dataset test metrics
  embedllm_summary.md   mean +- std over seeds vs. paper Tables 2, 5, 6
  embedllm_emb/<variant>_a<A>_s<S>.npy   learned model embeddings P [112, 232] (Stage 11)

Usage (project root):  python repro/baselines/embedllm.py [--anchors 256 512 1024] [--seeds 42] [--device cpu]
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from repro.baselines.knn_kmeans import DATASETS, PAPER_COLS, to_md  # noqa: E402
from repro.common import RESULTS, RUNS, load_data, meta_anchors, metrics, summarize  # noqa: E402

OUT = RESULTS / "baselines"
VARIANTS = {"embedllm": 0, "embedllm+1": 1, "embedllm+2": 2}  # name -> hidden layers in the classifier
LRS = [1e-4, 1e-3]
DIM, ALPHA, BATCH, WD = 232, 0.05, 2048, 1e-5

# Paper Tables 5 (routing) and 6 (correctness): [overall, *PAPER_COLS]
PAPER = {
    ("embedllm", "routing"): {
        256: [58.80, 58.23, 41.18, 72.03, 79.85, 35.14, 75.55, 88.39, 28.20, 19.19, 30.25],
        512: [59.47, 59.49, 45.10, 74.29, 77.61, 43.24, 76.83, 89.29, 28.80, 12.12, 29.63],
        1024: [59.60, 58.65, 50.98, 72.88, 79.10, 47.30, 76.83, 89.29, 23.80, 25.25, 30.86],
    },
    ("embedllm+", "routing"): {
        256: [59.17, 49.37, 35.29, 62.99, 62.69, 51.35, 77.08, 89.29, 24.20, 54.55, 35.80],
        512: [61.57, 57.81, 45.10, 73.45, 81.34, 41.89, 78.27, 89.29, 29.00, 33.84, 32.72],
        1024: [59.73, 57.38, 45.10, 71.75, 79.85, 32.43, 78.69, 89.29, 26.20, 21.72, 29.01],
    },
    ("embedllm", "correctness"): {
        256: [67.33, 65.95, 55.58, 60.39, 57.24, 40.08, 64.01, 62.32, 79.03, 96.14, 65.24],
        512: [68.12, 65.75, 55.92, 58.78, 76.86, 39.18, 64.80, 65.51, 78.73, 95.68, 61.40],
        1024: [69.47, 65.50, 61.38, 60.45, 76.61, 60.56, 65.62, 70.07, 79.02, 94.95, 62.67],
    },
    ("embedllm+", "correctness"): {
        256: [66.14, 65.09, 53.15, 58.83, 47.88, 38.04, 62.77, 65.95, 78.69, 95.28, 66.01],
        512: [68.09, 64.29, 57.63, 58.89, 68.97, 51.35, 65.19, 68.53, 78.08, 94.56, 61.63],
        1024: [69.33, 65.22, 63.25, 60.19, 74.41, 56.38, 65.55, 70.94, 78.47, 95.72, 64.84],
    },
}
# Fig. 1 reads EmbedLLM at 2048 evaluations per model from the plot (approx.)
PAPER_FIG1_2048 = {"embedllm": 62.6}


class TextMF(nn.Module):
    """Official TextMF; the frozen question table is replaced by passing phi(x) directly."""

    def __init__(self, num_models: int, dim: int = DIM, alpha: float = ALPHA, text_dim: int = 768, hidden: int = 0):
        super().__init__()
        self.P = nn.Embedding(num_models, dim)
        self.text_proj = nn.Linear(text_dim, dim)
        self.alpha = alpha
        layers = []
        for _ in range(hidden):
            layers += [nn.Linear(dim, dim), nn.ReLU()]
        self.classifier = nn.Sequential(*layers, nn.Linear(dim, 2))

    def forward(self, model: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p = self.P(model)
        if self.training:
            q = q + torch.randn_like(q) * self.alpha
        return self.classifier(p * self.text_proj(q))

    @torch.no_grad()
    def score_matrix(self, X: torch.Tensor, chunk: int = 256) -> np.ndarray:
        """S [N, M] = logit[1] - logit[0] for every (query, model) pair."""
        self.eval()
        P = self.P.weight                                       # [M, d]
        out = []
        for i in range(0, len(X), chunk):
            u = self.text_proj(X[i:i + chunk])                  # [n, d]
            logits = self.classifier(u[:, None, :] * P[None])   # [n, M, 2]
            out.append((logits[..., 1] - logits[..., 0]).cpu().numpy())
        return np.concatenate(out)


def hit1(S: np.ndarray, Y: np.ndarray) -> float:
    """Overall routing hit@1 with the same tie-breaking as evaluation.py (first index of argsort(-s))."""
    best = np.argsort(-S, axis=1)[:, 0]
    return float(Y[np.arange(len(Y)), best].mean())


def train_one(Xa, Ya, Xv, Yv, hidden: int, lr: float, seed: int, device: str,
              max_epochs: int, patience: int) -> dict:
    torch.manual_seed(seed)
    A, M = Ya.shape
    pairs_m = torch.arange(M).repeat(A)                         # pair k = (anchor k // M, model k % M)
    pairs_q = torch.arange(A).repeat_interleave(M)
    labels = torch.from_numpy(Ya.reshape(-1).astype(np.int64))
    Xa_t, Xv_t = torch.from_numpy(Xa).to(device), torch.from_numpy(Xv).to(device)

    net = TextMF(M, hidden=hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=WD)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)
    best = {"val_hit1": -1.0, "epoch": -1, "state": None}
    for epoch in range(max_epochs):
        net.train()
        for idx in torch.randperm(len(labels), generator=gen).split(BATCH):
            m, q, y = pairs_m[idx].to(device), Xa_t[pairs_q[idx]], labels[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss_fn(net(m, q), y).backward()
            opt.step()
        v = hit1(net.score_matrix(Xv_t), Yv)
        if v > best["val_hit1"]:  # strict: the first epoch reaching the max is kept, as in training_script.py
            best = {"val_hit1": v, "epoch": epoch, "state": copy.deepcopy(net.state_dict())}
        elif epoch - best["epoch"] >= patience:
            break
    net.load_state_dict(best["state"])
    return {"net": net, "best_epoch": best["epoch"], "stopped_epoch": epoch, "val_hit1": best["val_hit1"] * 100}


def run_one(data, A: int, seed: int, anchors: np.ndarray, args) -> tuple[list[dict], list[dict]]:
    Xa, Ya = data.train.X[anchors], data.train.Y[anchors]
    Xt = torch.from_numpy(data.test.X).to(args.device)
    grid, chosen = [], []
    for variant, hidden in VARIANTS.items():
        fits = {}
        for lr in LRS:
            fit = train_one(Xa, Ya, data.val.X, data.val.Y, hidden, lr, seed, args.device,
                            args.max_epochs, args.patience)
            S_test = fit["net"].score_matrix(Xt)
            fit["test"] = metrics(S_test, data.test.Y, data.test.category)
            fits[lr] = fit
            grid.append({"anchors": A, "seed": seed, "variant": variant, "lr": lr,
                         "best_epoch": fit["best_epoch"], "stopped_epoch": fit["stopped_epoch"],
                         "val_hit1": fit["val_hit1"], "test_hit1": fit["test"]["rt_overall"]["hit@1"] * 100,
                         "test_acc": fit["test"]["bp_overall"]["correctness_pred_acc"] * 100})
        lr = max(fits, key=lambda k: fits[k]["val_hit1"])     # first max: on a tie the smaller lr wins
        fit = fits[lr]
        s = summarize(fit["test"])
        row = {"anchors": A, "seed": seed, "variant": variant, "lr": lr, "best_epoch": fit["best_epoch"],
               "val_hit1": fit["val_hit1"],
               "routing": s["overall"]["routing"]["hit@1"] * 100,
               "correctness": s["overall"]["correctness_prediction"]["correctness_pred_acc"] * 100}
        for r in s["by_dataset"]["routing"]:
            row[f"rt_{r['dataset']}"] = r["hit@1"] * 100
        for r in s["by_dataset"]["correctness_prediction"]:
            row[f"bp_{r['dataset']}"] = r["correctness_pred_acc"] * 100
        chosen.append(row)
        np.save(OUT / "embedllm_emb" / f"{variant}_a{A}_s{seed}.npy", fit["net"].P.weight.detach().cpu().numpy())
    return grid, chosen


def summary_tables(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """EmbedLLM+ is reported as the better of 1 / 2 hidden layers by mean val hit@1 per budget."""
    plus = runs[runs.variant.str.startswith("embedllm+")]
    pick = plus.groupby(["anchors", "variant"])["val_hit1"].mean().reset_index()
    pick = pick.loc[pick.groupby("anchors")["val_hit1"].idxmax(), ["anchors", "variant"]]
    rows = pd.concat([runs[runs.variant == "embedllm"].assign(method="embedllm"),
                      plus.merge(pick, on=["anchors", "variant"]).assign(method="embedllm+")])
    t2, t56 = [], []
    for (method, A), g in rows.groupby(["method", "anchors"]):
        row = {"method": method, "variant": g.variant.iloc[0], "anchors": A, "seeds": len(g),
               "lr": ",".join(f"{x:g}" for x in g["lr"])}
        for metric in ("routing", "correctness"):
            v = g[metric]
            p = PAPER[(method, metric)].get(A, [np.nan])[0]
            if metric == "routing" and np.isnan(p):
                p = PAPER_FIG1_2048.get(method, np.nan) if A == 2048 else np.nan
            row[metric] = v.mean()
            row[f"{metric}_std"] = v.std(ddof=1) if len(v) > 1 else np.nan
            row[f"{metric}_paper"] = p
            row[f"{metric}_diff"] = v.mean() - p
        t2.append(row)
        for metric, prefix in (("routing", "rt_"), ("correctness", "bp_")):
            paper = PAPER[(method, metric)].get(A)
            if paper is None:
                continue
            r = {"method": method, "anchors": A, "metric": metric}
            for ds, col, p in zip(DATASETS, PAPER_COLS, paper[1:]):
                ours = g[prefix + ds].mean()
                r[col] = f"{ours:.1f} ({ours - p:+.1f})"
            t56.append(r)
    return pd.DataFrame(t2), pd.DataFrame(t56)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--anchors", type=int, nargs="*", default=None, help="default: every finished run")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--device", default="cpu", help="cpu by default: the GPU is often busy with LOCUS runs")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    run_dirs = []
    for d in sorted((RUNS / "main").glob("a*_s*"), key=lambda p: [int(x) for x in re.findall(r"\d+", p.name)]):
        m = re.fullmatch(r"a(\d+)_s(\d+)", d.name)
        if not m or not (d / "meta.json").exists():
            continue
        A, seed = int(m[1]), int(m[2])
        if (args.anchors and A not in args.anchors) or (args.seeds and seed not in args.seeds):
            continue
        run_dirs.append((A, seed, d))
    if not run_dirs:
        raise SystemExit("no matching runs under runs/main")

    (OUT / "embedllm_emb").mkdir(parents=True, exist_ok=True)
    data = load_data()
    grid, runs = [], []
    for A, seed, d in run_dirs:
        meta = json.loads((d / "meta.json").read_text())
        if meta["emb_col"] != data.emb_col or meta.get("model_indices", {}).get("train"):
            raise SystemExit(f"{d.name}: expected the default encoder and all 112 models")
        t0 = time.time()
        g, r = run_one(data, A, seed, meta_anchors(meta), args)
        grid += g
        runs += r
        msg = " | ".join(f"{x['variant']} lr {x['lr']:g} ep {x['best_epoch']}: hit@1 {x['routing']:.2f} acc {x['correctness']:.2f}"
                         for x in r)
        print(f"{d.name:<11} {msg}  ({time.time() - t0:.0f} s)", flush=True)
        # write after every run, so an interrupted sweep keeps its results
        pd.DataFrame(grid).to_csv(OUT / "embedllm_grid.csv", index=False)
        pd.DataFrame(runs).to_csv(OUT / "embedllm_runs.csv", index=False)

    grid, runs = pd.DataFrame(grid), pd.DataFrame(runs)
    t2, t56 = summary_tables(runs)
    md = [
        "# Stage 4: EmbedLLM and EmbedLLM+", "",
        "Official TextMF trained on the anchors of each LOCUS run in `runs/main`; lr and epoch chosen per run by val hit@1.",
        "EmbedLLM+ = the better of 1 / 2 hidden classifier layers by mean val hit@1. Test metrics, %.", "",
        "## Table 2 (overall): mean over seeds vs. paper", "", to_md(t2), "",
        "## Tables 5-6 (per dataset): mean over seeds (difference to paper)", "", to_md(t56), "",
        "## All variants: mean over seeds", "",
        to_md(runs.groupby(["variant", "anchors"])[["val_hit1", "routing", "correctness", "best_epoch"]].mean().reset_index()), "",
        "## Grid: mean over seeds", "",
        to_md(grid.groupby(["variant", "anchors", "lr"])[["best_epoch", "val_hit1", "test_hit1", "test_acc"]].mean().reset_index()), "",
    ]
    (OUT / "embedllm_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:9]))


if __name__ == "__main__":
    main()
