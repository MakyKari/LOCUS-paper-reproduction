#!/usr/bin/env python3
"""
Stage 4: IRT-Net baseline (Chen et al., 2025, arXiv:2510.00844), trained on the anchors of each LOCUS run.

Re-implemented from the official repository (github.com/JianhaoChen-nju/IrtNet, commit 6cf5eec,
src/modules.py and pipeline.sh; the repository has no licence file, so no code is copied):
  h(x)   = shared expert(x) + sum_i softmax(g(x))_i * expert_i(x)   experts: 768 -> 512 -> ReLU -> dropout -> 256
           39 routed experts with top_k = 39 (dense; the load-balancing bias only affects top-k selection,
           so with top_k = all it has no effect and is omitted)
  a(x)   = Linear(256, 232) (discrimination), b(x) = Linear(256, 1) (difficulty)
  logit  = a(x) . theta_m - b(x),   theta_m = Embedding(M, 232)
Query embeddings get N(0, 0.05) noise in training. BCE, Adam lr 1e-4, weight decay 1e-4, batch 2048 shuffled
pairs, dropout 0.5, ReduceLROnPlateau(val loss, factor 0.1, patience 2), at most 30 epochs, early stopping
(patience 5) and checkpoint selection by val correctness accuracy, all as in pipeline.sh.

Evaluation uses the same metrics as LOCUS (repro.common.metrics). In eval mode a(x), b(x) depend only on x,
so the full [N, M] logit matrix is a(X) theta^T - b(X).

Outputs (results/baselines/): irtnet_runs.csv, irtnet_summary.md

Usage (project root):  python repro/baselines/irtnet.py --device cuda [--anchors 256 512 1024] [--seeds 42]
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
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro.baselines.knn_kmeans import to_md  # noqa: E402
from repro.common import RESULTS, RUNS, load_data, meta_anchors, metrics, summarize  # noqa: E402

OUT = RESULTS / "baselines"
DIM, N_EXPERTS, HIDDEN, OUT_DIM = 232, 39, 512, 256
LR, WD, BATCH, DROPOUT, NOISE = 1e-4, 1e-4, 2048, 0.5, 0.05
EPOCHS, PATIENCE = 30, 5
# Paper Tables 5 (routing) and 6 (correctness), IRT-Net rows: [overall, *PAPER_COLS]
PAPER = {
    "routing": {256: 59.57, 512: 60.17, 1024: 63.37},
    "correctness": {256: 67.38, 512: 69.07, 1024: 70.12},
}
PAPER_FIG1_2048 = 64.1  # read from Fig. 1 (approx.)


def expert(d_in: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(HIDDEN, OUT_DIM))


class IrtNet(nn.Module):
    def __init__(self, num_models: int, q_dim: int = 768):
        super().__init__()
        self.theta = nn.Embedding(num_models, DIM)
        self.shared = expert(q_dim)
        self.gate = nn.Linear(q_dim, N_EXPERTS)
        self.experts = nn.ModuleList(expert(q_dim) for _ in range(N_EXPERTS))
        self.disc = nn.Linear(OUT_DIM, DIM)
        self.diff = nn.Linear(OUT_DIM, 1)

    def query(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            x = x + torch.randn_like(x) * NOISE
        w = torch.softmax(self.gate(x), dim=-1)                                  # [B, E]
        routed = torch.stack([e(x) for e in self.experts], dim=1)                # [B, E, 256]
        h = self.shared(x) + (w.unsqueeze(-1) * routed).sum(1)
        return self.disc(h), self.diff(h).squeeze(-1)

    def forward(self, m: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        a, b = self.query(x)
        return (a * self.theta(m)).sum(-1) - b

    @torch.no_grad()
    def logits(self, X: torch.Tensor, chunk: int = 1024) -> np.ndarray:
        """[N, M] logits for every (query, model) pair."""
        self.eval()
        out = []
        for i in range(0, len(X), chunk):
            a, b = self.query(X[i:i + chunk])
            out.append((a @ self.theta.weight.T - b[:, None]).cpu().numpy())
        return np.concatenate(out)


def train_one(Xa, Ya, Xv, Yv, seed: int, device: str, max_epochs: int) -> dict:
    torch.manual_seed(seed)
    A, M = Ya.shape
    pairs_m = torch.arange(M).repeat(A)
    pairs_q = torch.arange(A).repeat_interleave(M)
    labels = torch.from_numpy(Ya.reshape(-1).astype(np.float32))
    Xa_t, Xv_t = torch.from_numpy(Xa).to(device), torch.from_numpy(Xv).to(device)
    Yv_t = torch.from_numpy(Yv.astype(np.float32))

    net = IrtNet(M, Xa.shape[1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=2)
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    best = {"val_acc": -1.0, "epoch": -1, "state": None}
    stale = 0
    for epoch in range(max_epochs):
        net.train()
        for idx in torch.randperm(len(labels), generator=gen).split(BATCH):
            opt.zero_grad(set_to_none=True)
            logit = net(pairs_m[idx].to(device), Xa_t[pairs_q[idx].to(device)])
            loss_fn(logit, labels[idx].to(device)).backward()
            opt.step()
        S = torch.from_numpy(net.logits(Xv_t))
        val_loss = loss_fn(S, Yv_t).item()
        val_acc = float(((S >= 0).float() == Yv_t).float().mean())
        sched.step(val_loss)
        if val_acc > best["val_acc"]:
            best = {"val_acc": val_acc, "epoch": epoch, "state": copy.deepcopy(net.state_dict())}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    net.load_state_dict(best["state"])
    return {"net": net, "best_epoch": best["epoch"], "stopped_epoch": epoch, "val_acc": best["val_acc"] * 100}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--anchors", type=int, nargs="*", default=None, help="default: every finished run")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    run_dirs = []
    for d in sorted((RUNS / "main").glob("a*_s*"), key=lambda p: [int(x) for x in re.findall(r"\d+", p.name)]):
        m = re.fullmatch(r"a(\d+)_s(\d+)", d.name)
        if not m or not d.is_dir() or not (d / "meta.json").exists():
            continue
        A, seed = int(m[1]), int(m[2])
        if (args.anchors and A not in args.anchors) or (args.seeds and seed not in args.seeds):
            continue
        run_dirs.append((A, seed, d))
    if not run_dirs:
        raise SystemExit("no matching runs under runs/main")

    data = load_data()
    Xt = torch.from_numpy(data.test.X).to(args.device)
    rows = []
    for A, seed, d in run_dirs:
        t0 = time.time()
        anchors = meta_anchors(json.loads((d / "meta.json").read_text()))
        fit = train_one(data.train.X[anchors], data.train.Y[anchors], data.val.X, data.val.Y, seed, args.device, args.max_epochs)
        s = summarize(metrics(fit["net"].logits(Xt), data.test.Y, data.test.category))
        row = {"anchors": A, "seed": seed, "best_epoch": fit["best_epoch"], "stopped_epoch": fit["stopped_epoch"],
               "val_acc": fit["val_acc"], "routing": s["overall"]["routing"]["hit@1"] * 100,
               "correctness": s["overall"]["correctness_prediction"]["correctness_pred_acc"] * 100}
        for r in s["by_dataset"]["routing"]:
            row[f"rt_{r['dataset']}"] = r["hit@1"] * 100
        for r in s["by_dataset"]["correctness_prediction"]:
            row[f"bp_{r['dataset']}"] = r["correctness_pred_acc"] * 100
        rows.append(row)
        print(f"{d.name:<11} ep {fit['best_epoch']}/{fit['stopped_epoch']}: hit@1 {row['routing']:.2f} acc {row['correctness']:.2f} "
              f"({time.time() - t0:.0f} s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "irtnet_runs.csv", index=False)

    runs = pd.DataFrame(rows)
    g = runs.groupby("anchors")[["routing", "correctness"]].agg(["mean", "std", "count"])
    t = pd.DataFrame({"anchors": g.index,
                      "seeds": g[("routing", "count")].to_numpy(),
                      "routing": g[("routing", "mean")].to_numpy(), "routing_std": g[("routing", "std")].to_numpy(),
                      "routing_paper": [PAPER["routing"].get(a, PAPER_FIG1_2048 if a == 2048 else np.nan) for a in g.index],
                      "correctness": g[("correctness", "mean")].to_numpy(), "correctness_std": g[("correctness", "std")].to_numpy(),
                      "correctness_paper": [PAPER["correctness"].get(a, np.nan) for a in g.index]})
    t["routing_diff"] = t.routing - t.routing_paper
    t["correctness_diff"] = t.correctness - t.correctness_paper
    md = ["# Stage 4: IRT-Net", "",
          "Official architecture and hyperparameters (pipeline.sh), trained on each LOCUS run's anchors; "
          "checkpoint chosen by val correctness accuracy (as in the official code). Test metrics, %.", "",
          to_md(t), "", "## All runs", "",
          to_md(runs[["anchors", "seed", "best_epoch", "stopped_epoch", "val_acc", "routing", "correctness"]]), ""]
    (OUT / "irtnet_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
