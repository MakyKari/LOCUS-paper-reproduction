#!/usr/bin/env python3
"""
EmbedLLM under the official training protocol (algorithm/mf.py defaults), to explain the gap
between our EmbedLLM (embedllm.py: lr chosen from {1e-4, 1e-3}, epoch chosen by val hit@1)
and the paper's EmbedLLM numbers.

Official protocol: lr 1e-4, 50 epochs, shuffle=False, the last epoch is reported (no val selection).
Pairs are fed anchor-major (all models of anchor 0, then anchor 1, ...), as in embedllm.py.
Per-epoch val/test hit@1 is logged, so "last epoch" and "best val epoch" come from the same fit.

Outputs (results/baselines/):
  embedllm_official_curves.csv   per (run, variant, epoch): val/test hit@1
  embedllm_official.md           test hit@1 at epoch 50 vs. best-val epoch vs. paper

Usage (project root):  python repro/baselines/embedllm_official.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro.baselines.embedllm import BATCH, OUT, PAPER, VARIANTS, WD, TextMF, hit1  # noqa: E402
from repro.collect_results import to_md  # noqa: E402
from repro.common import RUNS, load_data, meta_anchors  # noqa: E402

LR, EPOCHS = 1e-4, 50


def fit_curve(Xa, Ya, Xv, Yv, Xt, Yt, hidden: int, seed: int, device: str, shuffle: bool) -> list[dict]:
    torch.manual_seed(seed)
    A, M = Ya.shape
    pairs_m = torch.arange(M).repeat(A)
    pairs_q = torch.arange(A).repeat_interleave(M)
    labels = torch.from_numpy(Ya.reshape(-1).astype(np.int64))
    Xa_t = torch.from_numpy(Xa).to(device)
    Xv_t, Xt_t = torch.from_numpy(Xv).to(device), torch.from_numpy(Xt).to(device)
    net = TextMF(M, hidden=hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)
    curve = []
    for epoch in range(EPOCHS):
        net.train()
        order = torch.randperm(len(labels), generator=gen) if shuffle else torch.arange(len(labels))
        for idx in order.split(BATCH):
            opt.zero_grad(set_to_none=True)
            loss_fn(net(pairs_m[idx].to(device), Xa_t[pairs_q[idx]]), labels[idx].to(device)).backward()
            opt.step()
        curve.append({"epoch": epoch + 1, "val_hit1": hit1(net.score_matrix(Xv_t), Yv) * 100,
                      "test_hit1": hit1(net.score_matrix(Xt_t), Yt) * 100})
    return curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--anchors", type=int, nargs="*", default=None)
    args = ap.parse_args()

    data = load_data()
    rows = []
    for d in sorted((RUNS / "main").glob("a*_s*"), key=lambda p: [int(x) for x in re.findall(r"\d+", p.name)]):
        m = re.fullmatch(r"a(\d+)_s(\d+)", d.name)
        if not m or not d.is_dir():  # skip the a*_s*.log files
            continue
        A, seed = int(m[1]), int(m[2])
        if args.anchors and A not in args.anchors:
            continue
        anchors = meta_anchors(json.loads((d / "meta.json").read_text()))
        Xa, Ya = data.train.X[anchors], data.train.Y[anchors]
        for variant, hidden in VARIANTS.items():
            for shuffle in (False, True):
                for r in fit_curve(Xa, Ya, data.val.X, data.val.Y, data.test.X, data.test.Y,
                                   hidden, seed, args.device, shuffle):
                    rows.append({"anchors": A, "seed": seed, "variant": variant, "shuffle": shuffle, **r})
        last = [r for r in rows if r["anchors"] == A and r["seed"] == seed and r["epoch"] == EPOCHS]
        print(d.name, " | ".join(f"{r['variant']} sh={int(r['shuffle'])}: {r['test_hit1']:.2f}" for r in last), flush=True)
        pd.DataFrame(rows).to_csv(OUT / "embedllm_official_curves.csv", index=False)

    df = pd.DataFrame(rows)
    keys = ["anchors", "seed", "variant", "shuffle"]
    last = df[df.epoch == EPOCHS].set_index(keys).test_hit1.rename("test@50")
    bv = df.loc[df.groupby(keys).val_hit1.idxmax()].set_index(keys)
    t = pd.concat([last, bv.test_hit1.rename("test@best_val"), bv.epoch.rename("best_val_epoch")], axis=1)
    t = t.groupby(["variant", "shuffle", "anchors"]).mean().reset_index()
    t["paper"] = [PAPER.get(("embedllm" if v == "embedllm" else "embedllm+", "routing"), {}).get(a, [np.nan])[0]
                  for v, a in zip(t.variant, t.anchors)]
    md = ["# EmbedLLM under the official protocol", "",
          f"lr {LR:g}, {EPOCHS} epochs, batch {BATCH}; test hit@1 in %, mean over seeds.", "",
          "`test@50` = official (last epoch, no selection); `test@best_val` = same fit, epoch chosen by val hit@1.", "",
          to_md(t), ""]
    (OUT / "embedllm_official.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
