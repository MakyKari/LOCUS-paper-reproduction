#!/usr/bin/env python3
"""
Stage 4 summary: LOCUS vs. EmbedLLM / EmbedLLM+ / IRT-Net / K-Means / kNN on the same anchor sets.

Every baseline was trained on the anchors of the matching LOCUS run (runs/main/a<A>_s<S>), so runs
with the same (anchors, seed) are paired.  EmbedLLM+ = the 1- or 2-hidden-layer variant with the
higher mean val hit@1 per budget (same rule as repro/baselines/embedllm.py).

Inputs:  results/stage3_runs.csv, results/baselines/{embedllm,irtnet,knn_kmeans}_runs.csv
Outputs: results/stage4_comparison.md, results/stage4_comparison.csv, results/fig1_methods.png

Usage (project root):  python repro/compare_methods.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repro.collect_results import PAPER_ROUTING, to_md  # noqa: E402
from repro.common import RESULTS  # noqa: E402

KEYS = ["anchors", "seed"]
PAPER_EMBEDLLM = {256: 58.80, 512: 59.47, 1024: 59.60, 2048: 62.6}  # Table 2; 2048 read from Fig. 1 (approx.)
PAPER_IRTNET = {256: 59.57, 512: 60.17, 1024: 63.37, 2048: 64.1}  # Table 2; 2048 read from Fig. 1 (approx.)
PAPER_BASELINE = {"EmbedLLM": PAPER_EMBEDLLM, "IRT-Net": PAPER_IRTNET}


def load() -> pd.DataFrame:
    locus = pd.read_csv(RESULTS / "stage3_runs.csv").assign(method="LOCUS")
    emb = pd.read_csv(RESULTS / "baselines" / "embedllm_runs.csv")
    plus = emb[emb.variant != "embedllm"]
    pick = plus.groupby(["anchors", "variant"]).val_hit1.mean().reset_index()
    pick = pick.loc[pick.groupby("anchors").val_hit1.idxmax(), ["anchors", "variant"]]
    plus = plus.merge(pick, on=["anchors", "variant"]).assign(method="EmbedLLM+")
    emb = emb[emb.variant == "embedllm"].assign(method="EmbedLLM")
    irt = pd.read_csv(RESULTS / "baselines" / "irtnet_runs.csv").assign(method="IRT-Net")
    nn = pd.read_csv(RESULTS / "baselines" / "knn_kmeans_runs.csv")
    nn["method"] = nn.method.map({"kmeans": "K-Means", "knn": "kNN"})
    cols = KEYS + ["method", "routing", "correctness"]
    return pd.concat([d[cols] for d in (locus, emb, plus, irt, nn)], ignore_index=True)


def overall(df: pd.DataFrame) -> pd.DataFrame:
    order = ["LOCUS", "EmbedLLM", "EmbedLLM+", "IRT-Net", "K-Means", "kNN"]
    out = []
    for metric in ("routing", "correctness"):
        g = df.pivot_table(index="anchors", columns="method", values=metric, aggfunc="mean")[order]
        out.append(g.reset_index().assign(metric=metric))
    return pd.concat(out, ignore_index=True)[["metric", "anchors"] + order]


def paired(df: pd.DataFrame) -> pd.DataFrame:
    locus = df[df.method == "LOCUS"].set_index(KEYS)
    rows = []
    for m in ("EmbedLLM", "EmbedLLM+", "IRT-Net", "K-Means", "kNN"):
        other = df[df.method == m].set_index(KEYS)
        for metric in ("routing", "correctness"):
            d = (locus[metric] - other[metric]).dropna()
            rows.append({"vs": m, "metric": metric, "pairs": len(d), "locus_wins": int((d > 0).sum()),
                         "mean_diff": d.mean(), "min_diff": d.min(), "max_diff": d.max()})
    return pd.DataFrame(rows)


def fig1_claim(df: pd.DataFrame) -> list[str]:
    """Paper: LOCUS matches EmbedLLM@2048 / IRT-Net@2048 with 4.8x / 2.3x fewer evaluations (Fig. 1)."""
    r = df.groupby(["method", "anchors"]).routing.mean()
    locus = r["LOCUS"]
    lines = []
    for m, paper in PAPER_BASELINE.items():
        target = r[m, 2048]
        hit = locus[locus >= target]
        paper_hit = [a for a, v in sorted(PAPER_ROUTING.items()) if v >= paper[2048]]
        lines += [f"{m} @ 2048 anchors (ours): {target:.2f}  (paper Fig. 1: ≈{paper[2048]}). "
                  f"Smallest LOCUS budget reaching it: ours {hit.index.min() if len(hit) else 'none of 128-2048'}, "
                  f"paper's own numbers {paper_hit[0] if paper_hit else 'none'}."]
    return lines


def single_best() -> tuple[str, float]:
    """Trivial router: always pick the model with the highest train accuracy (no anchors needed)."""
    from repro.common import load_data
    data = load_data()
    best = int(data.train.Y.mean(0).argmax())
    return data.model_names[best], float(data.test.Y[:, best].mean() * 100)


def plot(df: pd.DataFrame, trivial: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"LOCUS": ("#2253C4", "o"), "EmbedLLM": ("#B0501A", "s"), "EmbedLLM+": ("#C9893A", "D"), "IRT-Net": ("#8E44AD", "P"),
             "K-Means": ("#2E8B57", "^"), "kNN": ("#7A7A7A", "v")}
    g = df.groupby(["method", "anchors"]).routing.agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=150)
    for m, (c, mk) in style.items():
        s = g[g.method == m]
        ax.errorbar(s.anchors, s["mean"], yerr=s["std"], color=c, marker=mk, capsize=3, label=f"{m} (ours)")
    xs = sorted(PAPER_ROUTING)
    ax.plot(xs, [PAPER_ROUTING[x] for x in xs], color="#2253C4", ls="--", alpha=0.6, label="LOCUS (paper)")
    xe = sorted(PAPER_EMBEDLLM)
    ax.plot(xe, [PAPER_EMBEDLLM[x] for x in xe], color="#B0501A", ls="--", alpha=0.6, label="EmbedLLM (paper)")
    xi = sorted(PAPER_IRTNET)
    ax.plot(xi, [PAPER_IRTNET[x] for x in xi], color="#8E44AD", ls="--", alpha=0.6, label="IRT-Net (paper)")
    ax.axhline(trivial, color="black", ls=":", lw=1, label="always the single best model")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.set_xlabel("query evaluations per model (anchors)")
    ax.set_ylabel("test routing accuracy, hit@1 (%)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_title("Fig. 1 reproduction: all methods, mean ± std over 3 seeds")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig1_methods.png")


def main() -> None:
    df = load()
    ov, pr = overall(df), paired(df)
    ov.to_csv(RESULTS / "stage4_comparison.csv", index=False)
    name, trivial = single_best()
    plot(df, trivial)
    md = ["# Stage 4: LOCUS vs. baselines on the same anchor sets", "",
          "Test metrics in %, mean over 3 seeds; every method selects its checkpoint / hyperparameter by val hit@1, except IRT-Net (val correctness accuracy, as in its official code).", "",
          f"Trivial router (always `{name}`, the best model by train accuracy): routing {trivial:.2f}.", "",
          "## Mean over seeds", "", to_md(ov), "",
          "## Paired by (anchors, seed): LOCUS minus baseline, 18 pairs", "", to_md(pr), "",
          "## Fig. 1 claim", "", *[f"- {s}" for s in fig1_claim(df)], ""]
    (RESULTS / "stage4_comparison.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
