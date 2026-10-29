#!/usr/bin/env python3
"""
Stage 6: embedding bank from the reference encoder (runs/ref/a4096_s42).

For each of the 112 models, z_ref is computed from the reference anchor set (the 4096 train
queries the run used, shared by all models), then the decoder scores every val/test query.

Outputs (results/bank/):
  Z_ref.npy         [112, 128] model embeddings
  logits_val.npy    [3000, 112] correctness logits
  logits_test.npy   [3000, 112]
  anchors_idx.npy   [4096] train-split query indices
  model_names.json  names aligned with the rows of Z_ref
  bank_check.md     recomputed metrics vs. training_summary.json

Usage (project root):  python repro/experiments/make_bank.py [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro.common import RESULTS, RUNS, embed, load_data, load_locus, meta_anchors, metrics, score, training_summary  # noqa: E402

REF = RUNS / "ref" / "a4096_s42"
OUT = RESULTS / "bank"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(REF))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)

    data = load_data()
    model, meta = load_locus(args.run, device=args.device)
    anchors = meta_anchors(meta)
    Z = embed(model, data.train.X, data.train.Y, anchors, device=args.device, batch_models=16)
    logits = {s: score(model, Z, getattr(data, s).X, device=args.device) for s in ("val", "test")}

    np.save(OUT / "Z_ref.npy", Z.float().cpu().numpy())
    np.save(OUT / "anchors_idx.npy", anchors)
    for s, L in logits.items():
        np.save(OUT / f"logits_{s}.npy", np.asarray(L, dtype=np.float32))
    (OUT / "model_names.json").write_text(json.dumps(data.model_names, indent=1))

    best = training_summary(args.run)["best_eval"]
    lines = ["# Stage 6 bank check", "", f"run `{Path(args.run).name}`, {len(anchors)} anchors, "
             f"Z {tuple(Z.shape)}, device {args.device}", "",
             "| split | metric | bank | training_summary |", "|---|---|---|---|"]
    for s, L in logits.items():
        out = metrics(L, getattr(data, s).Y, getattr(data, s).category)
        ref = best[s]["overall"]
        lines.append(f"| {s} | hit@1 | {out['rt_overall']['hit@1'] * 100:.2f} | {ref['routing']['hit@1'] * 100:.2f} |")
        lines.append(f"| {s} | correctness | {out['bp_overall']['correctness_pred_acc'] * 100:.2f} | "
                     f"{ref['correctness_prediction']['correctness_pred_acc'] * 100:.2f} |")
    (OUT / "bank_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
