#!/usr/bin/env python3
"""
Stage 8: robustness of embeddings to the query evaluations (paper Sec. 3.3, App. E.3: Figs. 4, 5, 13, 14).

Reference encoder and z_ref from the Stage 6 bank; R = the 4096 reference queries (shared by all models).

  size     for each model independently, sample n queries from R and re-embed (n up to 4096).
  overlap  keep n = 4096 (paper: "fixed at n_ref"), take alpha*n queries from R and the rest from the
           train queries outside R, independently per model.
For every setting and trial: test routing hit@1 and correctness accuracy for all 112 models (Fig. 4),
distance of each regenerated z to its z_ref (Figs. 13-14; the paper shows CodeLlama-13b-hf, which is not
among the 112 models, so CodeLlama-13b-Instruct-hf is used, plus the mean over all models), and
identification: the fraction of models whose own z_ref is the nearest reference to the regenerated z
(not in the paper; the fingerprinting claim of Sec. 3.3 in numbers).
Fig. 5: openTSNE of the 112 references plus regenerated embeddings of 4 models.

Outputs (results/): robustness_runs.csv, fig4.png, fig13_14.png, fig5.png, stage8_summary.md

Usage (project root):  python repro/experiments/robustness.py [--device cuda] [--trials 10]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro.collect_results import to_md  # noqa: E402
from repro.common import RESULTS, distances, embed, load_data, load_locus, metrics, model_index, score  # noqa: E402
from repro.experiments.make_bank import REF  # noqa: E402

BANK = RESULTS / "bank"
SIZES = [8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
ALPHAS = [round(a, 1) for a in np.arange(0, 1.01, 0.1)]
N_OVERLAP = 4096
FOCUS = "CodeLlama-13b-Instruct-hf"
TSNE_MODELS = ["Meta-Llama-3-70B-Instruct", "meta-llama__Meta-Llama-3-8B", "phi-1_5", "gemma-7b-it"]
PAPER = {
    "size": {"correctness_max": 0.711, "routing": {8: 0.47, 128: 0.625, 384: 0.64}, "routing_max": 0.649},
    "overlap": {"correctness": 0.712, "routing": 0.654},
    "fig13_euclid": {0.0: 1.27, 1.0: 0.0},
    "fig14_euclid": {256: 2.53, 1024: 1.08, 4096: 0.0},
    "rank1_euclid": 0.38,
}


def regen_sets(kind: str, value, R: np.ndarray, outside: np.ndarray, M: int, rng) -> list:
    """One query-index array per model."""
    if kind == "size":
        return [np.sort(rng.choice(R, value, replace=False)) for _ in range(M)]
    k = int(round(value * N_OVERLAP))
    return [np.sort(np.concatenate([rng.choice(R, k, replace=False), rng.choice(outside, N_OVERLAP - k, replace=False)]))
            for _ in range(M)]


def evaluate(Z: torch.Tensor, Zref: np.ndarray, model, data, device: str) -> dict:
    out = metrics(score(model, Z, data.test.X, device=device), data.test.Y, data.test.category)
    Zn = Z.float().cpu().numpy()
    euc = np.linalg.norm(Zn - Zref, axis=1)
    a = Zn / np.linalg.norm(Zn, axis=1, keepdims=True)
    b = Zref / np.linalg.norm(Zref, axis=1, keepdims=True)
    sim = a @ b.T                                                   # [regenerated, reference]
    return {"routing": out["rt_overall"]["hit@1"], "correctness": out["bp_overall"]["correctness_pred_acc"],
            "cos": 1 - np.diag(sim), "euc": euc, "identified": (sim.argmax(1) == np.arange(len(sim)))}


def run(model, data, Zref, device: str, trials: int, seed: int = 0):
    R = np.load(BANK / "anchors_idx.npy")
    outside = np.setdiff1d(np.arange(len(data.train.Y)), R)
    M = Zref.shape[0]
    focus = model_index(FOCUS)
    tsne_idx = [model_index(m) for m in TSNE_MODELS]
    rng = np.random.default_rng(seed)
    rows, tsne = [], []
    for kind, values in (("size", SIZES), ("overlap", ALPHAS)):
        for v in values:
            for t in range(trials):
                sets = regen_sets(kind, v, R, outside, M, rng)
                Z = embed(model, data.train.X, data.train.Y, sets, device=device, batch_models=16)
                e = evaluate(Z, Zref, model, data, device)
                rows.append({"kind": kind, "value": v, "trial": t, "routing": e["routing"], "correctness": e["correctness"],
                             "focus_cos": e["cos"][focus], "focus_euc": e["euc"][focus],
                             "mean_cos": e["cos"].mean(), "mean_euc": e["euc"].mean(),
                             "identified": e["identified"].mean()})
                if t == 0:
                    tsne += [{"kind": kind, "value": v, "model": m, "z": Z[m].float().cpu().numpy()} for m in tsne_idx]
            print(f"{kind} {v}: routing {np.mean([x['routing'] for x in rows[-trials:]]):.4f} "
                  f"identified {np.mean([x['identified'] for x in rows[-trials:]]):.3f}", flush=True)
    return pd.DataFrame(rows), tsne


def neighbour_lines(Zref: np.ndarray, m: int) -> dict:
    Dc, De = distances(Zref)
    out = {}
    for name, D in (("cos", Dc), ("euc", De)):
        d = np.sort(np.delete(D[m], m))
        out[name] = {r: d[r - 1] for r in (1, 2, 3, 5, 10)}
    return out


def plot_fig4(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    g = df.groupby(["kind", "value"])[["routing", "correctness"]].agg(["mean", "std"])
    fig, axes = plt.subplots(2, 2, figsize=(8, 4.8), dpi=150, sharey="row", sharex="col")
    for col, kind in enumerate(("size", "overlap")):
        s = g.loc[kind]
        if kind == "size":
            s = s[s.index <= 1024]                                  # the paper's x range
        for row, (metric, color) in enumerate((("correctness", "#2253C4"), ("routing", "#E07B1F"))):
            ax = axes[row, col]
            m, sd = s[(metric, "mean")], s[(metric, "std")]
            ax.plot(s.index, m, "o-", color=color, ms=4, label=f"{metric} (max={m.max():.3f})")
            ax.fill_between(s.index, m - sd, m + sd, color=color, alpha=0.2, lw=0)
            ax.grid(alpha=0.3, ls="--")
            ax.legend(frameon=False, fontsize=8, loc="lower right")
    axes[0, 0].set_ylim(0.6, 0.8)
    axes[1, 0].set_ylim(0.4, 0.8)
    pr = PAPER["size"]["routing"]
    axes[1, 0].plot(list(pr), list(pr.values()), "s", color="#B0501A", ms=5, label="paper (plot)")
    axes[1, 0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0, 0].set_ylabel("accuracy")
    axes[1, 0].set_ylabel("accuracy")
    axes[1, 0].set_xlabel("number of queries (subsampled from R)")
    axes[1, 1].set_xlabel(f"overlap fraction with R (n = {N_OVERLAP})")
    fig.suptitle("Fig. 4 reproduction: mean ± std over trials", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig4.png")


def plot_fig13_14(df: pd.DataFrame, lines: dict) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), dpi=150)
    colors = {1: "#2253C4", 2: "#E07B1F", 3: "#2E8B57", 5: "#C0392B", 10: "#8E44AD"}
    for row, (kind, xlabel) in enumerate((("overlap", "overlap fraction"), ("size", "number of queries"))):
        s = df[(df.kind == kind) & ((df.value >= 256) if kind == "size" else True)]
        g = s.groupby("value")[["focus_cos", "focus_euc", "mean_cos", "mean_euc"]].agg(["mean", "std"])
        for col, d in enumerate(("cos", "euc")):
            ax = axes[row, col]
            m, sd = g[(f"focus_{d}", "mean")], g[(f"focus_{d}", "std")]
            ax.plot(g.index, m, "o-", color="#1F3B73", label=f"{FOCUS}: new vs. reference")
            ax.fill_between(g.index, m - sd, m + sd, color="#1F3B73", alpha=0.18, lw=0)
            ax.plot(g.index, g[(f"mean_{d}", "mean")], color="#7A7A7A", ls="-.", label="mean over all 112 models")
            for r, c in colors.items():
                ax.axhline(lines[d][r], color=c, ls="--", lw=1, label=f"neighbour rank {r}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(("cosine" if d == "cos" else "Euclidean") + " distance")
            ax.grid(alpha=0.3, ls="--")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Figs. 13 (top) and 14 (bottom) reproduction", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig13_14.png")


def plot_fig5(Zref: np.ndarray, tsne: list) -> None:
    import matplotlib.pyplot as plt
    from openTSNE import TSNE
    names = {model_index(m): m for m in TSNE_MODELS}
    panels = (("overlap", [0.0, 0.2, 0.4, 0.6, 0.8], "overlap"), ("size", [512, 768, 1024], "# queries"))
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7.5), dpi=150)
    palette = ["#8E44AD", "#C0392B", "#2E8B57", "#E07B1F", "#2253C4"]
    for ax, (kind, values, label) in zip(axes, panels):
        pts = [p for p in tsne if p["kind"] == kind and p["value"] in values]
        X = np.vstack([Zref] + [p["z"] for p in pts])
        Y = np.asarray(TSNE(perplexity=30, metric="cosine", random_state=0, n_jobs=4).fit(X))
        ref, new = Y[:len(Zref)], Y[len(Zref):]
        sel = list(names)
        ax.scatter(*np.delete(ref, sel, axis=0).T, s=12, color="#BBBBBB", label="other models")
        for v, c in zip(values, palette):
            k = [i for i, p in enumerate(pts) if p["value"] == v]
            ax.scatter(*new[k].T, s=40, color=c, label=f"{label}={v}")
        ax.scatter(*ref[sel].T, s=70, marker="X", color="white", edgecolor="black", label="reference (selected)")
        for m in sel:
            ax.annotate(names[m].split("__")[-1], ref[m], xytext=(6, 6), textcoords="offset points", fontsize=8)
        ax.legend(frameon=False, fontsize=7)
        ax.grid(alpha=0.3, ls="--")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
    fig.suptitle("Fig. 5 reproduction (openTSNE, cosine)", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig5.png")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()
    torch.set_num_threads(4)

    data = load_data()
    Zref = np.load(BANK / "Z_ref.npy")
    model, _ = load_locus(REF, device=args.device)
    df, tsne = run(model, data, Zref, args.device, args.trials)
    df.to_csv(RESULTS / "robustness_runs.csv", index=False)

    lines = neighbour_lines(Zref, model_index(FOCUS))
    plot_fig4(df)
    plot_fig13_14(df, lines)
    plot_fig5(Zref, tsne)

    cols = ["routing", "correctness", "identified", "focus_euc", "mean_euc", "focus_cos", "mean_cos"]
    tab = df.groupby(["kind", "value"])[cols].mean().reset_index()
    p = PAPER
    md = ["# Stage 8: robustness of embeddings", "",
          "Reference encoder runs/ref/a4096_s42; R = its 4096 anchors. Test metrics; mean over "
          f"{args.trials} trials. `identified` = fraction of models whose own z_ref is the nearest reference "
          f"(cosine) to the regenerated z. Focus model for Figs. 13-14: {FOCUS}.", "",
          "Paper (plot): size: correctness max 0.711, routing ≈0.47 at 8, ≈0.625 at 128, ≈0.64 at 384, max 0.649; "
          "overlap: flat, correctness ≈0.712, routing ≈0.654.",
          "Paper Figs. 13-14 (CodeLlama-13b-hf, Euclidean): ≈1.27 at overlap 0, ≈2.53 at n=256, ≈1.08 at n=1024; "
          "rank-1 neighbour ≈0.38.", "",
          "Neighbour distances of the focus model's z_ref: "
          + "; ".join(f"{d}: " + ", ".join(f"r{r} {v:.4f}" for r, v in lines[d].items()) for d in ("cos", "euc")), "",
          to_md(tab, "{:.4f}"), ""]
    (RESULTS / "stage8_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
