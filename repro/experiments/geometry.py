#!/usr/bin/env python3
"""
Stage 9: embedding geometry (paper Sec. 3.4-3.5: Table 3 / Fig. 6, Fig. 7, Fig. 8).

Table 3 / Fig. 6  correlation over all 112*111/2 model pairs between embedding distance (cosine,
                  Euclidean) and correctness disagreement on the test split (normalized Hamming).
Fig. 7            hierarchical clustering (Euclidean) of 3 math + 4 code models; the paper's merge
                  heights equal the max pairwise distances, i.e. complete linkage; average and ward
                  are reported too.
Fig. 8            agreement of each model with its k-th nearest neighbour (cosine) on val+test pooled.

Embeddings: the Stage 6 bank (reference encoder, 4096 anchors).  For sensitivity, Table 3 and Fig. 8
are repeated with the Stage 3 encoders (1024 / 2048 anchors, 3 seeds) and with a no-encoder control:
the raw label vectors on the same 4096 anchors (distance = their normalized Hamming distance).

Outputs (results/): table3.csv, geometry_fig7.csv, fig8.csv, fig6.png, fig7.png, fig8.png, stage9_summary.md

Usage (project root):  python repro/experiments/geometry.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau, pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro.collect_results import to_md  # noqa: E402
from repro.common import (RESULTS, RUNS, distances, embed, load_data, load_locus,  # noqa: E402
                          meta_anchors, model_index)

BANK = RESULTS / "bank"
PAPER_T3 = {("pearson", "cosine"): 0.845, ("pearson", "euclidean"): 0.887,
            ("spearman", "cosine"): 0.886, ("spearman", "euclidean"): 0.876,
            ("kendall", "cosine"): 0.714, ("kendall", "euclidean"): 0.702}
FAMILY = {  # Fig. 7, in the paper's leaf order
    "MetaMath-Mistral-7B": "math", "llemma_34b": "math", "deepseek-math-7b-instruct": "math",
    "deepseek-coder-1.3b-base": "code", "octocoder": "code", "deepseek-coder-6.7b-instruct": "code",
    "CodeLlama-7b-hf": "code",
}
PAPER_FIG7 = {  # Fig. 7 heatmap, upper triangle in FAMILY order
    ("llemma_34b", "deepseek-math-7b-instruct"): 1.87, ("deepseek-coder-1.3b-base", "octocoder"): 1.66,
    ("deepseek-coder-6.7b-instruct", "CodeLlama-7b-hf"): 2.97, ("MetaMath-Mistral-7B", "deepseek-coder-1.3b-base"): 18.47,
    ("MetaMath-Mistral-7B", "llemma_34b"): 6.59, ("MetaMath-Mistral-7B", "deepseek-math-7b-instruct"): 7.22,
}
PAPER_FIG8 = {1: 0.79, 20: 0.75, 50: 0.72, 99: 0.62}  # read from the plot (approx.)


def hamming(Y: np.ndarray) -> np.ndarray:
    """[M, M] fraction of rows on which two label columns differ."""
    Y = Y.astype(np.float64)
    agree = Y.T @ Y + (1 - Y).T @ (1 - Y)
    return 1 - agree / len(Y)


def correlations(D: np.ndarray, target: np.ndarray) -> dict:
    iu = np.triu_indices_from(D, k=1)
    x, y = D[iu], target[iu]
    return {"pearson": pearsonr(x, y)[0], "spearman": spearmanr(x, y)[0], "kendall": kendalltau(x, y)[0]}


def kth_neighbour_agreement(D: np.ndarray, agree: np.ndarray) -> np.ndarray:
    """[M, M-1]: agreement of model m with its k-th nearest neighbour (k = 1..M-1) under distance D."""
    M = len(D)
    D = D.copy()
    np.fill_diagonal(D, np.inf)
    order = np.argsort(D, axis=1, kind="stable")[:, :M - 1]
    return np.take_along_axis(agree, order, axis=1)


def encoders(data) -> dict:
    """name -> (cosine D, Euclidean D) for the bank and the Stage 3 encoders; plus the raw-label control."""
    out = {}
    Z = np.load(BANK / "Z_ref.npy")
    out["ref_a4096"] = distances(Z)
    for A in (1024, 2048):
        for seed in (42, 43, 44):
            run = RUNS / "main" / f"a{A}_s{seed}"
            model, meta = load_locus(run, device="cpu")
            out[f"a{A}_s{seed}"] = distances(embed(model, data.train.X, data.train.Y, meta_anchors(meta), device="cpu"))
    anchors = np.load(BANK / "anchors_idx.npy")
    H = hamming(data.train.Y[anchors])
    out["raw_labels_a4096"] = (H, H)
    return out


def table3(dists: dict, dis_test: np.ndarray) -> pd.DataFrame:
    rows = []
    for name, (Dc, De) in dists.items():
        for kind, D in (("cosine", Dc), ("euclidean", De)):
            if name.startswith("raw") and kind == "euclidean":
                continue  # one distance only for the control
            c = correlations(D, dis_test)
            rows.append({"embedding": name, "distance": "hamming" if name.startswith("raw") else kind, **c,
                         **({f"paper_{k}": PAPER_T3[(k, kind)] for k in c} if name == "ref_a4096" else {})})
    return pd.DataFrame(rows)


def fig6(Dc: np.ndarray, De: np.ndarray, dis: np.ndarray, t3: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    iu = np.triu_indices_from(Dc, k=1)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), dpi=150, sharey=True)
    for ax, D, kind in ((axes[0], Dc, "cosine"), (axes[1], De, "euclidean")):
        r = t3[(t3.embedding == "ref_a4096") & (t3.distance == kind)].iloc[0]
        ax.scatter(D[iu], dis[iu], s=6, alpha=0.12, color="#2253C4", linewidths=0)
        ax.set_xlabel(f"model embedding {kind} distance")
        ax.set_title(f"Pearson {r.pearson:.3f} (paper {PAPER_T3[('pearson', kind)]})", fontsize=9)
        ax.grid(alpha=0.3, ls="--")
    axes[0].set_ylabel("correctness disagreement (test)")
    fig.suptitle("Fig. 6 reproduction: reference encoder (4096 anchors)", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig6.png")


def fig7(De: np.ndarray) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    short = list(FAMILY)
    idx = [model_index(s) for s in short]
    D = De[np.ix_(idx, idx)]
    pairs = []
    for i in range(len(short)):
        for j in range(i + 1, len(short)):
            pairs.append({"model_a": short[i], "model_b": short[j], "same_family": FAMILY[short[i]] == FAMILY[short[j]],
                          "euclidean": D[i, j], "paper": PAPER_FIG7.get((short[i], short[j]), np.nan)})
    colors = {"math": "#D9822B", "code": "#2E8B57"}
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), dpi=150, gridspec_kw={"width_ratios": [1.5, 1, 1, 1]})
    im = axes[0].imshow(D, cmap="coolwarm")
    for i in range(len(short)):
        for j in range(len(short)):
            if i != j:
                axes[0].text(j, i, f"{D[i, j]:.2f}", ha="center", va="center", fontsize=7)
    axes[0].set_xticks(range(len(short)), short, rotation=90, fontsize=7)
    axes[0].set_yticks(range(len(short)), short, fontsize=7)
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    heights = {}
    for ax, method in zip(axes[1:], ("complete", "average", "ward")):
        L = linkage(squareform(D, checks=False), method=method)
        heights[method] = L[:, 2]
        dendrogram(L, labels=short, ax=ax, leaf_rotation=90, color_threshold=0, above_threshold_color="#555")
        for t in ax.get_xticklabels():
            t.set_color(colors[FAMILY[t.get_text()]])
            t.set_fontsize(7)
        ax.set_title(f"{method} linkage", fontsize=9)
    fig.suptitle("Fig. 7 reproduction: Euclidean distances, reference encoder (math = orange, code = green)", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig7.png")
    df = pd.DataFrame(pairs)
    df.attrs["heights"] = heights
    df.attrs["separates"] = {m: families_separate(D, short, m) for m in ("complete", "average", "ward")}
    return df


def families_separate(D: np.ndarray, names: list, method: str) -> bool:
    """True if cutting the tree into 2 clusters gives exactly the math / code families."""
    from scipy.cluster.hierarchy import fcluster
    lab = fcluster(linkage(squareform(D, checks=False), method=method), 2, criterion="maxclust")
    fam = np.array([FAMILY[n] == "math" for n in names])
    return len(set(lab[fam])) == 1 and len(set(lab[~fam])) == 1 and set(lab[fam]) != set(lab[~fam])


def fig8(dists: dict, agree: np.ndarray) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    rows = []
    curves = {}
    for name, (Dc, _) in dists.items():
        A = kth_neighbour_agreement(Dc, agree)
        curves[name] = A
        for k in range(A.shape[1]):
            rows.append({"embedding": name, "k": k + 1, "mean": A[:, k].mean(), "std": A[:, k].std()})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=150)
    A = curves["ref_a4096"]
    k = np.arange(1, A.shape[1] + 1)
    ax.plot(k, A.mean(0), marker="o", ms=3, color="#2253C4", label="LOCUS, reference encoder (mean ± std over models)")
    ax.fill_between(k, A.mean(0) - A.std(0), A.mean(0) + A.std(0), color="#2253C4", alpha=0.18, lw=0)
    R = curves["raw_labels_a4096"]
    ax.plot(k, R.mean(0), color="#7A7A7A", ls="--", label="raw labels on the same 4096 anchors (Hamming)")
    ax.plot(list(PAPER_FIG8), list(PAPER_FIG8.values()), "s", color="#B0501A", label="paper (read from plot)")
    ax.axvline(99, color="black", lw=0.6, ls=":")
    ax.set_xlabel("k-th nearest neighbour in model embedding space (cosine)")
    ax.set_ylabel("correctness agreement (val+test)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig8.png")
    return df


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    torch.set_num_threads(4)
    data = load_data()
    names = json.loads((BANK / "model_names.json").read_text())
    assert names == data.model_names, "bank was built with a different model order"

    dis_test = hamming(data.test.Y)
    agree_vt = 1 - hamming(np.concatenate([data.val.Y, data.test.Y]))
    dists = encoders(data)

    t3 = table3(dists, dis_test)
    t3.to_csv(RESULTS / "table3.csv", index=False)
    Dc, De = dists["ref_a4096"]
    fig6(Dc, De, dis_test, t3)
    f7 = fig7(De)
    f7.to_csv(RESULTS / "geometry_fig7.csv", index=False)
    f8 = fig8(dists, agree_vt)
    f8.to_csv(RESULTS / "fig8.csv", index=False)

    ks = [1, 5, 20, 50, 99, 111]
    f8_tab = f8[f8.k.isin(ks)].pivot(index="embedding", columns="k", values="mean").reset_index()
    f8_tab.columns = ["embedding"] + [f"k={k}" for k in ks]
    paper_row = {"embedding": "paper (plot ≈)", **{f"k={k}": PAPER_FIG8.get(k, np.nan) for k in ks}}
    f8_tab = pd.concat([pd.DataFrame([paper_row]), f8_tab], ignore_index=True)
    iu = np.triu_indices_from(De, k=1)
    h = f7.attrs["heights"]
    md = ["# Stage 9: embedding geometry", "",
          "Reference encoder = Stage 6 bank (runs/ref/a4096_s42). `raw_labels_a4096` is a no-encoder control: "
          "the normalized Hamming distance between the models' labels on the same 4096 anchors.", "",
          "## Table 3: distance vs. test disagreement over 6216 model pairs", "", to_md(t3, "{:.3f}"), "",
          f"Distance ranges (reference encoder): cosine {Dc[iu].min():.3f}–{Dc[iu].max():.3f}, "
          f"Euclidean {De[iu].min():.2f}–{De[iu].max():.2f} (paper Fig. 6: cosine 0–0.9, Euclidean 0–22).", "",
          "## Fig. 7: Euclidean distances among math / code models", "", to_md(f7, "{:.2f}"), "",
          *[f"- {m} linkage: merge heights {', '.join(f'{x:.2f}' for x in h[m])}; "
            f"families separate at 2 clusters: {f7.attrs['separates'][m]}" for m in h],
          "- paper (complete linkage, from the dendrogram): 1.66, 1.87, 2.97, 7.22, 8.17, 18.47", "",
          "## Fig. 8: agreement with the k-th nearest neighbour (cosine), val+test pooled, mean over models", "",
          to_md(f8_tab, "{:.3f}"), ""]
    (RESULTS / "stage9_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
