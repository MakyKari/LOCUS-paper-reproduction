#!/usr/bin/env python3
"""
Stage 4: does the paper's EmbedLLM behave like a router that always picks the single strongest model?

Compares the paper's EmbedLLM routing (Table 5, overall and per dataset) with
  - always Llama-3-70B-Instruct (the best model by train accuracy), per dataset on test;
  - "best on the anchors": the model with the highest accuracy on each LOCUS run's anchor set,
    always chosen (what a router collapsing to main effects would do), mean over seeds;
  - our EmbedLLM (results/baselines/embedllm_runs.csv), mean over seeds.

Outputs (results/baselines/): embedllm_paper_check.md, embedllm_paper_check.csv

Usage (project root):  python repro/baselines/embedllm_paper_check.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro.baselines.embedllm import PAPER  # noqa: E402
from repro.baselines.knn_kmeans import DATASETS, PAPER_COLS, to_md  # noqa: E402
from repro.common import RESULTS, RUNS, _merge_dataset_names, load_data, meta_anchors  # noqa: E402

OUT = RESULTS / "baselines"
BUDGETS = [256, 512, 1024]


def per_dataset(acc: np.ndarray, category: np.ndarray) -> list[float]:
    """Overall and per-dataset mean of a [N] 0/1 vector, in PAPER_COLS order."""
    ds = _merge_dataset_names(category)
    return [acc.mean() * 100] + [acc[ds == d].mean() * 100 for d in DATASETS]


def main() -> None:
    data = load_data()
    Yt, cat = data.test.Y, data.test.category
    best = int(data.train.Y.mean(0).argmax())
    cols = ["overall"] + PAPER_COLS
    rows = [dict(zip(cols, per_dataset(Yt[:, best].astype(float), cat)), router=f"always {data.model_names[best]}", anchors="-")]

    emb = pd.read_csv(OUT / "embedllm_runs.csv")
    emb = emb[emb.variant == "embedllm"]
    for A in BUDGETS:
        rows.append(dict(zip(cols, PAPER[("embedllm", "routing")][A]), router="EmbedLLM (paper)", anchors=A))
        picks = []
        for d in sorted((RUNS / "main").glob(f"a{A}_s*")):
            if not re.fullmatch(rf"a{A}_s\d+", d.name) or not (d / "meta.json").exists():
                continue
            idx = meta_anchors(json.loads((d / "meta.json").read_text()))
            m = int(data.train.Y[idx].mean(0).argmax())
            picks.append((d.name, data.model_names[m], per_dataset(Yt[:, m].astype(float), cat)))
        rows.append(dict(zip(cols, np.mean([p[2] for p in picks], axis=0)), anchors=A,
                         router="always the best model on the anchors: " + ", ".join(f"{n}={name}" for n, name, _ in picks)))
        e = emb[emb.anchors == A]
        rt = ["routing"] + [f"rt_{d}" for d in DATASETS]
        rows.append(dict(zip(cols, e[rt].mean().to_numpy()), router="EmbedLLM (ours)", anchors=A))

    df = pd.DataFrame(rows)[["router", "anchors"] + cols]
    df.to_csv(OUT / "embedllm_paper_check.csv", index=False)

    trivial = df.iloc[0][cols].to_numpy(float)
    gap = []
    for A in BUDGETS:
        paper = df[(df.router == "EmbedLLM (paper)") & (df.anchors == A)][cols].to_numpy(float)[0]
        ours = df[(df.router == "EmbedLLM (ours)") & (df.anchors == A)][cols].to_numpy(float)[0]
        gap.append({"anchors": A, "paper_vs_always_best_mean_abs": np.abs(paper[1:] - trivial[1:]).mean(),
                    "ours_vs_always_best_mean_abs": np.abs(ours[1:] - trivial[1:]).mean()})
    md = ["# Is the paper's EmbedLLM a 'single best model' router?", "",
          "Test routing hit@1, %, overall and per dataset. Rows 'best on the anchors' always route to the model "
          "with the highest accuracy on the run's anchors (mean over seeds 42-44).", "",
          to_md(df.round(2)), "", "## Mean |difference| to 'always the best model' over the 10 datasets, pp", "",
          to_md(pd.DataFrame(gap).round(2)), ""]
    (OUT / "embedllm_paper_check.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
