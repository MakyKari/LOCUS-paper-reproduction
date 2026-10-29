#!/usr/bin/env python3
"""
Stage 5: collect the ablation runs (runs/ablation/<tag>_a<A>_s42) into Tables 7-8 (sentence encoder)
and Tables 9-10 (embedding dimension), overall test routing hit@1 and correctness accuracy vs. the paper.

The default configuration (mpnet, d=128) comes from Stage 3 (runs/main/a<A>_s42); its spread over seeds
42-44 is reported as the noise scale, since every ablation cell is a single seed. Per-dataset cells
are not compared: the minilm / distilroberta rows of Tables 7-8 have permuted columns in the paper.

Outputs (results/): table7_8_encoder.csv, table9_10_dim.csv, stage5_summary.md

Usage (project root):  python repro/collect_ablations.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repro.collect_results import to_md  # noqa: E402
from repro.common import RESULTS, RUNS  # noqa: E402

ANCHORS = [256, 512, 1024]
# paper, overall (routing, correctness) per anchors 256 / 512 / 1024
PAPER_ENC = {
    "mpnet": ([61.90, 62.97, 64.70], [68.31, 68.33, 70.03]),
    "albert": ([59.00, 61.40, 60.97], [59.42, 66.61, 67.69]),
    "minilm": ([61.60, 63.20, 62.27], [66.12, 68.14, 69.28]),
    "distilroberta": ([61.83, 62.60, 63.33], [67.50, 68.17, 70.34]),
}
PAPER_DIM = {
    16: ([60.97, 62.93, 63.87], [67.94, 68.32, 69.97]),
    32: ([62.13, 62.17, 62.90], [67.75, 68.47, 69.43]),
    64: ([61.77, 61.60, 62.87], [67.82, 68.53, 69.73]),
    128: ([61.90, 62.97, 64.70], [68.31, 68.33, 70.03]),
    256: ([61.87, 63.27, 64.43], [68.42, 68.36, 70.34]),
    512: ([62.10, 64.07, 62.70], [67.93, 68.35, 69.89]),
}


def result(run: Path) -> dict | None:
    f = run / "training_summary.json"
    if not f.exists():
        return None
    s = json.loads(f.read_text())["best_eval"]["test"]["overall"]
    return {"routing": s["routing"]["hit@1"] * 100, "correctness": s["correctness_prediction"]["correctness_pred_acc"] * 100}


def table(kind: str, paper: dict) -> pd.DataFrame:
    rows = []
    for key, (p_rt, p_bp) in paper.items():
        for A, prt, pbp in zip(ANCHORS, p_rt, p_bp):
            default = key in ("mpnet", 128)
            r = result(RUNS / "main" / f"a{A}_s42") if default else \
                result(RUNS / "ablation" / f"{'enc_' if kind == 'encoder' else 'z'}{key}_a{A}_s42")
            row = {kind: key, "anchors": A, "routing": np.nan, "routing_paper": prt,
                   "correctness": np.nan, "correctness_paper": pbp}
            if r:
                row.update(r)
            rows.append(row)
    df = pd.DataFrame(rows)
    df["routing_diff"] = df.routing - df.routing_paper
    df["correctness_diff"] = df.correctness - df.correctness_paper
    return df[[kind, "anchors", "routing", "routing_paper", "routing_diff", "correctness", "correctness_paper", "correctness_diff"]]


def seed_noise() -> pd.DataFrame:
    rows = []
    for A in ANCHORS:
        rs = [result(RUNS / "main" / f"a{A}_s{s}") for s in (42, 43, 44)]
        rs = [r for r in rs if r]
        rt, bp = [r["routing"] for r in rs], [r["correctness"] for r in rs]
        rows.append({"anchors": A, "seeds": len(rs), "routing_min": min(rt), "routing_max": max(rt),
                     "correctness_min": min(bp), "correctness_max": max(bp)})
    return pd.DataFrame(rows)


def ranking(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Per anchors, the variant with the best routing: ours vs. paper."""
    rows = []
    for A, g in df.groupby("anchors"):
        g = g.dropna(subset=["routing"])
        if len(g) < len(df[df.anchors == A]):
            continue
        rows.append({"anchors": A, "best_ours": g.loc[g.routing.idxmax(), kind],
                     "best_paper": g.loc[g.routing_paper.idxmax(), kind],
                     "spread_ours": g.routing.max() - g.routing.min(),
                     "spread_paper": g.routing_paper.max() - g.routing_paper.min()})
    return pd.DataFrame(rows)


def main() -> None:
    enc, dim = table("encoder", PAPER_ENC), table("z_dim", PAPER_DIM)
    enc.to_csv(RESULTS / "table7_8_encoder.csv", index=False)
    dim.to_csv(RESULTS / "table9_10_dim.csv", index=False)
    noise = seed_noise()
    done = int(enc.routing.notna().sum() + dim.routing.notna().sum())
    md = ["# Stage 5: ablations (seed 42)", "",
          f"{done} of {len(enc) + len(dim)} cells available (mpnet / d=128 cells are the Stage 3 runs). Test metrics, %.", "",
          "## Noise scale: default configuration (mpnet, d=128) over seeds 42-44", "", to_md(noise), "",
          "## Tables 7-8: sentence encoder", "", to_md(enc), "",
          "## Tables 9-10: embedding dimension", "", to_md(dim), "",
          "## Best routing variant per budget", "",
          to_md(pd.concat([ranking(enc, "encoder").assign(ablation="encoder"),
                           ranking(dim, "z_dim").assign(ablation="z_dim")], ignore_index=True)), ""]
    (RESULTS / "stage5_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
