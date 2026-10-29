#!/usr/bin/env python3
"""
Stage 3: collect LOCUS runs (runs/main/a<A>_s<S>) into tables and compare with the paper.

Outputs (results/):
  table2_locus.csv          overall routing / correctness per anchor budget: mean, std, min, max, paper, diff
  table5_6_locus.csv        per-dataset routing (Table 5) and correctness (Table 6), mean over seeds vs paper
  selection_protocols.csv   test hit@1 under three checkpoint-selection rules (from the per-epoch CSV logs)
  stage3_summary.md         the same tables in Markdown
  fig1_locus.png            routing vs. anchors per model, ours (mean ± std over seeds) vs. paper

Usage (project root):  python repro/collect_results.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "main"
OUT = ROOT / "results"

# Paper Table 2 / Fig. 1 (LOCUS).  Fig. 1 points without a table value are read from the plot (approx.).
PAPER_ROUTING = {128: 60.3, 256: 61.90, 512: 62.97, 768: 63.65, 1024: 64.70, 2048: 65.0}
PAPER_CORRECT = {256: 68.31, 512: 68.33, 1024: 70.03}
FROM_PLOT = {128, 768, 2048}

DATASETS = ["mathqa", "logiqa", "medmcqa", "piqa", "truthfulqa_mc1", "MMLU", "gsm8k", "GPQA", "asdiv", "social_iqa"]
PAPER_COLS = ["MathQA", "LogiQA", "MedQA", "PIQA", "TruthQA", "MMLU", "GSM8k", "GPQA", "ASDiv", "SoQA"]
PAPER_T5 = {  # routing, LOCUS rows of Table 5 (per-dataset, column order = PAPER_COLS)
    256: [39.66, 41.18, 55.65, 85.82, 33.78, 86.25, 75.89, 25.20, 65.66, 29.63],
    512: [49.79, 43.14, 57.91, 86.57, 52.70, 83.70, 85.71, 27.40, 60.61, 30.86],
    1024: [52.74, 49.02, 72.60, 85.07, 41.89, 81.83, 89.29, 29.20, 64.65, 31.48],
}
PAPER_T6 = {  # correctness, LOCUS rows of Table 6
    256: [65.38, 64.88, 59.36, 72.21, 69.12, 62.99, 66.18, 79.17, 96.46, 61.76],
    512: [64.09, 59.84, 60.03, 65.78, 69.12, 64.64, 71.75, 79.18, 96.05, 54.08],
    1024: [65.22, 67.75, 60.89, 73.03, 67.43, 65.39, 72.85, 79.42, 96.50, 66.83],
}


def load_runs() -> pd.DataFrame:
    rows = []
    for d in sorted(RUNS.glob("a*_s*")):
        m = re.fullmatch(r"a(\d+)_s(\d+)", d.name)
        summ = d / "training_summary.json"
        if not m or not summ.exists():
            continue
        s = json.loads(summ.read_text())
        test = s["best_eval"]["test"]
        row = {
            "run": d.name, "anchors": int(m[1]), "seed": int(m[2]),
            "best_epoch": s["best_epoch"], "stopped_epoch": s["stopped_epoch"],
            "val_hit1": s["best_eval"]["val"]["overall"]["routing"]["hit@1"] * 100,
            "routing": test["overall"]["routing"]["hit@1"] * 100,
            "correctness": test["overall"]["correctness_prediction"]["correctness_pred_acc"] * 100,
        }
        for r in test["by_dataset"]["routing"]:
            row[f"rt_{r['dataset']}"] = r["hit@1"] * 100
        for r in test["by_dataset"]["correctness_prediction"]:
            row[f"bp_{r['dataset']}"] = r["correctness_pred_acc"] * 100
        # selection protocols from the per-epoch log
        log = pd.read_csv(d / f"val_metrics_train_{row['anchors']}.csv")
        first_best = log["val_hit1"].idxmax()  # training script keeps the FIRST epoch reaching the max (strict >)
        row["sel_test_at_best_val"] = log.loc[first_best, "test_hit1"] * 100
        row["sel_best_test_any_epoch"] = log["test_hit1"].max() * 100
        row["sel_last_epoch"] = log["test_hit1"].iloc[-1] * 100
        rows.append(row)
    if not rows:
        raise SystemExit(f"no finished runs under {RUNS}")
    return pd.DataFrame(rows)


def table2(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for a, g in df.groupby("anchors"):
        for metric, paper in (("routing", PAPER_ROUTING), ("correctness", PAPER_CORRECT)):
            v = g[metric]
            p = paper.get(a)
            out.append({
                "anchors": a, "metric": metric, "seeds": len(v),
                "mean": v.mean(), "std": v.std(ddof=1) if len(v) > 1 else np.nan, "min": v.min(), "max": v.max(),
                "paper": p, "paper_source": ("plot ≈" if a in FROM_PLOT else "table") if p is not None else "",
                "diff_mean": v.mean() - p if p is not None else np.nan,
                "paper_within_range": (v.min() <= p <= v.max()) if p is not None else None,
            })
    return pd.DataFrame(out)


def table56(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for a in sorted(PAPER_T5):
        g = df[df.anchors == a]
        if g.empty:
            continue
        for kind, prefix, paper in (("routing", "rt_", PAPER_T5), ("correctness", "bp_", PAPER_T6)):
            row = {"anchors": a, "metric": kind}
            for ds, col, p in zip(DATASETS, PAPER_COLS, paper[a]):
                ours = g[prefix + ds].mean()
                row[col] = f"{ours:.1f} ({ours - p:+.1f})"
            out.append(row)
    return pd.DataFrame(out)


def protocols(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["sel_test_at_best_val", "sel_best_test_any_epoch", "sel_last_epoch"]
    g = df.groupby("anchors")[cols].mean().reset_index()
    g["paper"] = g["anchors"].map(PAPER_ROUTING)
    return g


def to_md(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def plot_fig1(t2: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = t2[t2.metric == "routing"].sort_values("anchors")
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    ax.errorbar(r.anchors, r["mean"], yerr=r["std"], marker="o", capsize=3, color="#2253C4",
                label="ours, mean ± std (3 seeds)")
    xs = sorted(PAPER_ROUTING)
    ax.plot(xs, [PAPER_ROUTING[x] for x in xs], marker="s", linestyle="--", color="#B0501A", label="paper (LOCUS)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.set_xlabel("query evaluations per model (anchors)")
    ax.set_ylabel("test routing accuracy, hit@1 (%)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title("Fig. 1 reproduction: LOCUS")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_locus.png")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = load_runs()
    t2, t56, sel = table2(df), table56(df), protocols(df)
    df.to_csv(OUT / "stage3_runs.csv", index=False)
    t2.to_csv(OUT / "table2_locus.csv", index=False)
    t56.to_csv(OUT / "table5_6_locus.csv", index=False)
    sel.to_csv(OUT / "selection_protocols.csv", index=False)
    plot_fig1(t2)

    md = [
        "# Stage 3: LOCUS main results", "",
        f"{len(df)} runs from `runs/main`. Test metrics at the best-val-hit@1 epoch, in %.", "",
        "## Table 2 (overall)", "", to_md(t2), "",
        "## Tables 5-6 (per dataset): mean over seeds (difference to paper)", "", to_md(t56), "",
        "## Checkpoint selection rules: test hit@1, mean over seeds", "",
        "`sel_test_at_best_val` is what training_script.py reports.", "", to_md(sel), "",
        "## All runs", "", to_md(df[["run", "best_epoch", "stopped_epoch", "val_hit1", "routing", "correctness"]]), "",
    ]
    (OUT / "stage3_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
