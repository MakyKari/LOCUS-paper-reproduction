#!/usr/bin/env python3
"""
Stage 10: downstream applications of the embeddings (paper App. E.1, E.2, E.4: Figs. 11, 12, 9).

Fig. 12  fallback routing: the routed model m*(x) = argmax p(x) is unavailable; answer with its
         k-th nearest neighbour (cosine) instead, k = 1..50.
Fig. 9   portfolio selection from embedding geometry only (RBF kernel on cosine distance, sigma = median
         pairwise distance): k-center, k-medoids (PAM from the k-center solution), random; and a
         parameter budget with coverage-greedy (gain per parameter) vs. random feasible subsets.
         Accuracy = test routing hit@1 with argmax p restricted to the selected models.
Fig. 11  hypothetical embeddings: n random train queries get Bernoulli labels with model m's
         per-dataset train accuracy (the caption's "dataset-wise"; the text says "task-wise", so the
         80 raw categories are run too); recall@k of m among the 112 real embeddings (cosine), 5 trials.

Controls not in the paper (they use correctness data, so they are references, not competitors):
  Fig. 12  the router's own next choice (argmax p over the remaining models) and a random other model.
  Fig. 9   the k models / the models within budget with the highest train accuracy.
  Fig. 11  the real labels of the same n queries instead of synthetic ones.

Inputs: results/bank (Stage 6), results/model_params.csv.
Outputs (results/): fig9.png, fig11.png, fig12.png, fig9_count.csv, fig9_budget.csv, fig11.csv, fig12.csv,
                    stage10_summary.md

Usage (project root):  python repro/experiments/applications.py [--device cuda] [--trials 5]
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
from repro.common import RESULTS, _merge_dataset_names, distances, embed_sets, load_data, load_locus  # noqa: E402
from repro.experiments.make_bank import REF  # noqa: E402

BANK = RESULTS / "bank"
KS_COUNT = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25]
BUDGETS = [7, 15, 20, 30, 50, 75, 110, 150, 200, 300, 400, 500]
NS = [256, 512, 1024, 2048, 4096, 8192]
RECALL_KS = [1, 2, 3, 5, 10]
PAPER = {
    "fig12": {"router": 0.6573, 1: 0.558, 15: 0.46, 49: 0.42},
    "fig9_count": {("k-center", 5): 0.635, ("k-center", 15): 0.66, ("k-medoids", 15): 0.637, ("random", 15): 0.55},
    "fig9_budget": {150: 0.625, 500: 0.64},
    "fig11": {(10, 256): 0.53, (10, 1024): 0.715, (10, 4096): 0.94, (10, 8192): 0.97, (1, 8192): 0.25},
}


def route(S: np.ndarray, models=None) -> np.ndarray:
    """Routed label column per query; ties as in evaluation.py (first index of argsort(-s))."""
    if models is None:
        return np.argsort(-S, axis=1)[:, 0]
    models = np.asarray(models)
    return models[np.argsort(-S[:, models], axis=1)[:, 0]]


def acc(S: np.ndarray, Y: np.ndarray, models=None) -> float:
    m = route(S, models)
    return float(Y[np.arange(len(Y)), m].mean())


# ================================
# Fig. 12: fallback routing
# ================================
def fallback(S: np.ndarray, Y: np.ndarray, Dc: np.ndarray, kmax: int = 50) -> pd.DataFrame:
    n = np.arange(len(Y))
    m_star = route(S)
    D = Dc.copy()
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, axis=1, kind="stable")                       # nn[m, k-1] = k-th neighbour of m
    order = np.argsort(-S, axis=1)                                  # router's own ranking
    M = Y.shape[1]
    random_other = (Y.sum(1) - Y[n, m_star]) / (M - 1)              # expected accuracy of a random other model
    rows = []
    for k in range(1, kmax + 1):
        rows.append({"k": k, "neighbour": Y[n, nn[m_star, k - 1]].mean(),
                     "router_next": Y[n, order[:, k]].mean(),       # the router's (k+1)-th choice
                     "random_other": random_other.mean()})
    df = pd.DataFrame(rows)
    df.attrs["router"] = Y[n, m_star].mean()
    return df


# ================================
# Fig. 9: portfolio selection
# ================================
def kernel(Dc: np.ndarray) -> np.ndarray:
    sigma = np.median(Dc[np.triu_indices_from(Dc, k=1)])
    return np.exp(-(Dc ** 2) / sigma ** 2)


def k_center(K: np.ndarray, k: int) -> list:
    sel = [int(np.argmax(K.mean(0)))]                               # global kernel medoid, eq. 10
    while len(sel) < k:
        cover = K[:, sel].max(1)
        cover[sel] = np.inf
        sel.append(int(np.argmin(cover)))                           # least-covered model, eq. 11
    return sel


def k_medoids(K: np.ndarray, k: int, iters: int = 100) -> list:
    sel = k_center(K, k)
    J = lambda s: K[:, s].max(1).sum()                              # eq. 12
    best = J(sel)
    for _ in range(iters):
        cand = None
        for i in range(k):                                          # best-improvement PAM swap
            for h in range(len(K)):
                if h in sel:
                    continue
                s = sel[:i] + [h] + sel[i + 1:]
                v = J(s)
                if v > best + 1e-12:
                    best, cand = v, s
        if cand is None:
            break
        sel = cand
    return sel


def coverage_greedy(K: np.ndarray, cost: np.ndarray, budget: float) -> list:
    sel, b, left = [], np.zeros(len(K)), budget
    while True:
        feas = [m for m in range(len(K)) if m not in sel and cost[m] <= left + 1e-9]
        if not feas:
            return sel
        gain = np.array([np.maximum(0, K[:, m] - b).sum() / cost[m] for m in feas])   # eq. 15-16
        m = feas[int(np.argmax(gain))]
        sel.append(m)
        b = np.maximum(b, K[:, m])
        left -= cost[m]


def random_feasible(rng, cost: np.ndarray, budget: float) -> list:
    """A random maximal subset within the budget: add models in random order while they fit."""
    sel, left = [], budget
    for m in rng.permutation(len(cost)):
        if cost[m] <= left + 1e-9:
            sel.append(int(m))
            left -= cost[m]
    return sel


def portfolio(S, Y, Dc, train_acc, cost, trials: int, seed: int = 0):
    K = kernel(Dc)
    rng = np.random.default_rng(seed)
    by_acc = np.argsort(-train_acc, kind="stable")
    count = []
    for k in KS_COUNT:
        rnd = [acc(S, Y, rng.choice(len(K), k, replace=False)) for _ in range(trials)]
        count.append({"k": k, "k-center": acc(S, Y, k_center(K, k)), "k-medoids": acc(S, Y, k_medoids(K, k)),
                      "random": np.mean(rnd), "random_std": np.std(rnd), "top_train_acc": acc(S, Y, by_acc[:k])})
    budget = []
    for C in BUDGETS:
        g = coverage_greedy(K, cost, C)
        rnd = [acc(S, Y, random_feasible(rng, cost, C)) for _ in range(trials)]
        top, left = [], C
        for m in by_acc:                                            # most accurate models that still fit
            if cost[m] <= left + 1e-9:
                top.append(m)
                left -= cost[m]
        budget.append({"budget_b": C, "coverage-greedy": acc(S, Y, g), "greedy_models": len(g),
                       "greedy_params_b": cost[g].sum(), "random": np.mean(rnd), "random_std": np.std(rnd),
                       "top_train_acc": acc(S, Y, top)})
    return pd.DataFrame(count), pd.DataFrame(budget)


# ================================
# Fig. 11: hypothetical embeddings
# ================================
def recall_at(Zq: torch.Tensor, Zref: np.ndarray) -> dict:
    Zq = torch.nn.functional.normalize(Zq.float().cpu(), dim=1).numpy()
    Zr = Zref / np.linalg.norm(Zref, axis=1, keepdims=True)
    sim = Zq @ Zr.T                                                 # [M query embeddings, M real]
    rank = (sim > sim[np.arange(len(sim)), np.arange(len(sim))][:, None]).sum(1)   # 0 = true model is nearest
    return {k: float((rank < k).mean()) for k in RECALL_KS}


def hypothetical(data, Zref: np.ndarray, device: str, trials: int, seed: int = 0) -> pd.DataFrame:
    """labels: synthetic_dataset (10 benchmarks), synthetic_task (80 raw categories) or real."""
    model, _ = load_locus(REF, device=device)
    profiles = {}
    for level, g in (("dataset", _merge_dataset_names(data.train.category)), ("task", np.asarray(data.train.category))):
        groups = np.unique(g)
        gidx = np.searchsorted(groups, g)
        profiles[level] = (gidx, np.stack([data.train.Y[gidx == i].mean(0) for i in range(len(groups))]))  # [G, M]
    rng = np.random.default_rng(seed)
    N, M = data.train.Y.shape
    rows = []
    for t in range(trials):
        for n in NS:
            q = rng.choice(N, n, replace=False)
            variants = [("real", data.train.Y[q].astype(np.float32))]
            for level, (gidx, prof) in profiles.items():
                P = prof[gidx[q]]                                   # [n, M] target probabilities
                variants.append((f"synthetic_{level}", (rng.random(P.shape) < P).astype(np.float32)))
            for labels, Yq in variants:
                sets = [(data.train.X[q], Yq[:, m]) for m in range(M)]
                Z = embed_sets(model, sets, device=device, batch_models=8)
                rows.append({"trial": t, "n": n, "labels": labels, **{f"recall@{k}": v for k, v in recall_at(Z, Zref).items()}})
        print(f"trial {t}: done", flush=True)
    return pd.DataFrame(rows)


# ================================
# Plots
# ================================
def plot_fig12(fb: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=150)
    ax.plot(fb.k, fb.neighbour, "o-", ms=3, color="#2253C4", label="k-th nearest neighbour of the routed model")
    ax.plot(fb.k, fb.router_next, color="#2E8B57", ls="--", label="router's (k+1)-th choice (uses p̂)")
    ax.axhline(fb.random_other.iloc[0], color="#7A7A7A", ls=":", label="random other model")
    ax.axhline(fb.attrs["router"], color="#C0392B", ls="--", label=f"routed model, acc={fb.attrs['router']:.4f}")
    p = PAPER["fig12"]
    ax.plot([1, 15, 49], [p[1], p[15], p[49]], "s", color="#B0501A", label="paper (read from plot)")
    ax.set_xlabel("routing to the k-th neighbouring model embedding")
    ax.set_ylabel("test routing accuracy")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig12.png")


def plot_fig9(count: pd.DataFrame, budget: pd.DataFrame, full: float) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150, sharey=True)
    ax = axes[0]
    ax.plot(count.k, count.random, "o-", color="#2253C4", label="random (mean ± std)")
    ax.fill_between(count.k, count.random - count.random_std, count.random + count.random_std, color="#2253C4", alpha=0.18, lw=0)
    ax.plot(count.k, count["k-medoids"], "o-", color="#E07B1F", label="k-medoids")
    ax.plot(count.k, count["k-center"], "o-", color="#2E8B57", label="k-center")
    ax.plot(count.k, count.top_train_acc, "--", color="#7A7A7A", label="top-k by train accuracy")
    ax.set_xlabel("k (number of chosen models)")
    ax.set_ylabel("test routing accuracy")
    ax = axes[1]
    ax.plot(budget.budget_b, budget.random, "o-", color="#2253C4", label="random feasible (mean ± std)")
    ax.fill_between(budget.budget_b, budget.random - budget.random_std, budget.random + budget.random_std, color="#2253C4", alpha=0.18, lw=0)
    ax.plot(budget.budget_b, budget["coverage-greedy"], "o-", color="#E07B1F", label="coverage-greedy")
    ax.plot(budget.budget_b, budget.top_train_acc, "--", color="#7A7A7A", label="most accurate that fit")
    ax.set_xlabel("parameter budget (B params)")
    for ax in axes:
        ax.axhline(full, color="#C0392B", ls=":", lw=1, label=f"full pool ({full:.3f})")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig9.png")


def plot_fig11(h: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    g = h.groupby(["labels", "n"]).agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
    colors = ["#2253C4", "#E07B1F", "#2E8B57", "#C0392B", "#8E44AD"]
    for k, c in zip(RECALL_KS, colors):
        s = g.loc["synthetic_dataset"][f"recall@{k}"]
        ax.plot(s.index, s["mean"], "o-", color=c, label=f"k={k}")
        ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color=c, alpha=0.18, lw=0)
    for k, c in ((1, colors[0]), (10, colors[4])):
        r = g.loc["real"][f"recall@{k}"]
        ax.plot(r.index, r["mean"], ls=":", color=c, label=f"k={k}, real labels")
        r = g.loc["synthetic_task"][f"recall@{k}"]
        ax.plot(r.index, r["mean"], ls="--", color=c, label=f"k={k}, per-task (80) profile")
    ax.plot([256, 1024, 4096, 8192], [PAPER["fig11"][(10, n)] for n in (256, 1024, 4096, 8192)], "s",
            color="#8E44AD", mfc="none", label="paper k=10 (plot)")
    ax.set_xlabel("number of queries")
    ax.set_title("solid: per-dataset (10) profile", fontsize=9)
    ax.set_ylabel("fraction of original models in top-k neighbours")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig11.png")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trials", type=int, default=5, help="Fig. 11 trials")
    ap.add_argument("--random-trials", type=int, default=50, help="Fig. 9 random baselines")
    args = ap.parse_args()
    torch.set_num_threads(4)

    data = load_data()
    Zref = np.load(BANK / "Z_ref.npy")
    S, Y = np.load(BANK / "logits_test.npy"), data.test.Y
    Dc, _ = distances(Zref)
    params = pd.read_csv(RESULTS / "model_params.csv")
    assert list(params.model_id) == data.model_idx
    cost = params.params_b.to_numpy(float)

    fb = fallback(S, Y, Dc)
    fb.to_csv(RESULTS / "fig12.csv", index=False)
    plot_fig12(fb)

    count, budget = portfolio(S, Y, Dc, data.train.Y.mean(0), cost, args.random_trials)
    count.to_csv(RESULTS / "fig9_count.csv", index=False)
    budget.to_csv(RESULTS / "fig9_budget.csv", index=False)
    full = acc(S, Y)
    plot_fig9(count, budget, full)

    h = hypothetical(data, Zref, args.device, args.trials)
    h.to_csv(RESULTS / "fig11.csv", index=False)
    plot_fig11(h)
    h_tab = h.groupby(["labels", "n"])[[f"recall@{k}" for k in RECALL_KS]].mean().reset_index()

    p = PAPER
    md = ["# Stage 10: downstream applications", "",
          "Reference encoder (Stage 6 bank), test split. Accuracies are fractions.", "",
          "## Fig. 12: fallback to the k-th nearest neighbour of the routed model", "",
          f"Routed model accuracy {fb.attrs['router']:.4f} (paper {p['fig12']['router']}). "
          f"Paper (plot): k=1 {p['fig12'][1]}, k=15 {p['fig12'][15]}, k=49 {p['fig12'][49]}.", "",
          to_md(fb[fb.k.isin([1, 2, 3, 5, 10, 15, 20, 30, 49, 50])], "{:.4f}"), "",
          f"k=1 keeps {fb.neighbour.iloc[0] / fb.attrs['router'] * 100:.1f}% of the routed accuracy (paper ≈85%).", "",
          "## Fig. 9 (left): count-constrained portfolio", "",
          f"Full pool: {full:.4f}. Paper (plot): k-center ≈0.635 at k=5, ≈0.66 at 15-25; k-medoids ≈0.637 at 15; random ≈0.55 at 15.", "",
          to_md(count, "{:.4f}"), "",
          "## Fig. 9 (right): parameter budget", "",
          f"Total parameters {cost.sum():.1f}B (paper 1930B). Paper (plot): coverage-greedy ≈0.625 at 150B, ≈0.64 at 500B.", "",
          to_md(budget, "{:.4f}"), "",
          "## Fig. 11: hypothetical embeddings, recall@k (mean over trials)", "",
          "Paper (plot): recall@10 ≈0.53 / 0.715 / 0.94 / 0.97 at n = 256 / 1024 / 4096 / 8192; recall@1 ≈0.25 at 8192.", "",
          to_md(h_tab, "{:.3f}"), ""]
    (RESULTS / "stage10_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
