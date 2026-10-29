#!/usr/bin/env python3
"""
Stage 11: instability of learned (SGD) model embeddings, EmbedLLM (paper App. F: Figs. 15-17).

Data: the 4096 reference anchors of the Stage 6 bank, all 112 models (the same evaluations LOCUS's
reference encoder used).  EmbedLLM: e_m (dim D) * W x -> MLP with n_hidden layers of width 64 -> 2 logits,
input noise alpha 0.05, Adam, CE loss, batch 2048 (as in repro/baselines/embedllm.py), early stopping on
val loss (all 112 models x val queries).

Fig. 16/17 grid: n_hidden in {0..4} x D in {64, 128, 232, 512}.  For each cell: train (e, G); freeze G,
re-initialise e, retrain e only on the same data (eq. 18). Report the mean cosine distance between original
and regenerated e (Fig. 16), the % of test (query, model) pairs whose predicted correctness flips (Fig. 17),
and, not in the paper, the fraction of models whose own original e is the nearest original to the
regenerated one (cosine), i.e. the `identified` metric of Stage 8.

Fig. 15 (n_hidden 0, D 232 = official EmbedLLM): regenerate e from other evaluation sets (overlap
0-0.8 with the 4096 anchors, n = 4096; or 512/768/1024 anchors subsampled), t-SNE as Fig. 5.

LOCUS comparison (not in the paper; the paper only notes that LOCUS re-embeds the same set exactly):
the reference encoder re-embeds every model from 4096 other train queries (overlap 0) and from 1024
subsampled anchors; same flip / identification metrics.

Outputs (results/): fig16_17.csv, fig16_17.png, fig15.csv, fig15.png, stage11_summary.md

Usage (project root):  python repro/experiments/embedllm_instability.py [--threads 3]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro.baselines.embedllm import ALPHA, BATCH, WD  # noqa: E402
from repro.collect_results import to_md  # noqa: E402
from repro.common import RESULTS, embed, load_data, load_locus, model_index, score  # noqa: E402
from repro.experiments.make_bank import REF  # noqa: E402
from repro.experiments.robustness import TSNE_MODELS  # noqa: E402

BANK = RESULTS / "bank"
HIDDEN = [0, 1, 2, 3, 4]
DIMS = [64, 128, 232, 512]
D_HIDDEN, LR, MAX_EPOCHS, PATIENCE = 64, 1e-3, 100, 5
PAPER_FIG16 = {(0, 64): 0.446, (0, 128): 0.511, (0, 232): 0.572, (0, 512): 0.648, (3, 64): 0.270}
PAPER_FIG17_RANGE = (1.2, 8.1)  # %, max at 3 hidden layers and D = 512


class EmbedMF(nn.Module):
    def __init__(self, num_models: int, dim: int, hidden: int, text_dim: int = 768):
        super().__init__()
        self.P = nn.Embedding(num_models, dim)
        self.text_proj = nn.Linear(text_dim, dim)
        layers, width = [], dim
        for _ in range(hidden):
            layers += [nn.Linear(width, D_HIDDEN), nn.ReLU()]
            width = D_HIDDEN
        self.classifier = nn.Sequential(*layers, nn.Linear(width, 2))

    def forward(self, m: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        if self.training:
            q = q + torch.randn_like(q) * ALPHA
        return self.classifier(self.P(m) * self.text_proj(q))

    @torch.no_grad()
    def logits(self, X: torch.Tensor, chunk: int = 512) -> np.ndarray:
        """[N, M] logit(correct) - logit(wrong)."""
        self.eval()
        out = []
        for i in range(0, len(X), chunk):
            u = self.text_proj(X[i:i + chunk])
            z = self.classifier(u[:, None, :] * self.P.weight[None])
            out.append((z[..., 1] - z[..., 0]).numpy())
        return np.concatenate(out)


def pairs(Y: np.ndarray, query_sets: list) -> tuple:
    """(model, query, label) tensors for per-model query index arrays."""
    m = np.concatenate([np.full(len(q), j) for j, q in enumerate(query_sets)])
    q = np.concatenate(query_sets)
    y = np.concatenate([Y[qs, j] for j, qs in enumerate(query_sets)])
    return torch.from_numpy(m), torch.from_numpy(q), torch.from_numpy(y.astype(np.int64))


def fit(net: EmbedMF, params, X: torch.Tensor, train: tuple, val: tuple, seed: int) -> int:
    """Adam on `params`; early stopping on val CE; restores the best state. Returns the best epoch."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(params, lr=LR, weight_decay=WD)
    loss_fn = nn.CrossEntropyLoss()
    Xv, (vm, vq, vy) = val[0], val[1]
    best, best_state, best_epoch = np.inf, None, -1
    for epoch in range(MAX_EPOCHS):
        net.train()
        m, q, y = train
        for idx in torch.randperm(len(y), generator=gen).split(BATCH):
            opt.zero_grad(set_to_none=True)
            loss_fn(net(m[idx], X[q[idx]]), y[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = sum(loss_fn(net(vm[i], Xv[vq[i]]), vy[i]).item() * len(i)
                    for i in torch.arange(len(vy)).split(65536)) / len(vy)
        if v < best - 1e-5:
            best, best_epoch = v, epoch
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        elif epoch - best_epoch >= PATIENCE:
            break
    net.load_state_dict(best_state)
    return best_epoch


def regenerate(net: EmbedMF, X, train, val, seed: int) -> tuple:
    """Freeze G (text_proj + classifier), re-initialise e and retrain it only (eq. 18)."""
    torch.manual_seed(seed)
    re = EmbedMF(net.P.num_embeddings, net.P.embedding_dim, 0)
    re.text_proj, re.classifier = net.text_proj, net.classifier
    for p in list(re.text_proj.parameters()) + list(re.classifier.parameters()):
        p.requires_grad_(False)
    ep = fit(re, re.P.parameters(), X, train, val, seed)
    for p in list(re.text_proj.parameters()) + list(re.classifier.parameters()):
        p.requires_grad_(True)
    return re, ep


def compare(E0: np.ndarray, E1: np.ndarray, S0: np.ndarray, S1: np.ndarray, Y: np.ndarray) -> dict:
    a = E0 / np.linalg.norm(E0, axis=1, keepdims=True)
    b = E1 / np.linalg.norm(E1, axis=1, keepdims=True)
    sim = b @ a.T                                                   # [regenerated, original]
    return {"cos_dist": float(np.mean(1 - np.diag(sim))),
            "flip_pct": float(((S0 >= 0) != (S1 >= 0)).mean() * 100),
            "identified": float((sim.argmax(1) == np.arange(len(sim))).mean()),
            "acc_orig": float(((S0 >= 0) == (Y > 0.5)).mean() * 100),
            "acc_regen": float(((S1 >= 0) == (Y > 0.5)).mean() * 100)}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    data = load_data()
    M = data.train.Y.shape[1]
    R = np.load(BANK / "anchors_idx.npy")
    X = torch.from_numpy(data.train.X)
    Xt = torch.from_numpy(data.test.X)
    val = (torch.from_numpy(data.val.X), pairs(data.val.Y, [np.arange(len(data.val.Y))] * M))
    train = pairs(data.train.Y, [R] * M)

    # ---- Figs. 16-17 grid
    rows = []
    nets = {}
    for h in HIDDEN:
        for D in DIMS:
            t0 = time.time()
            torch.manual_seed(args.seed)
            net = EmbedMF(M, D, h)
            ep0 = fit(net, net.parameters(), X, train, val, args.seed)
            re, ep1 = regenerate(net, X, train, val, args.seed + 1)
            r = compare(net.P.weight.detach().numpy(), re.P.weight.detach().numpy(),
                        net.logits(Xt), re.logits(Xt), data.test.Y)
            rows.append({"hidden": h, "D": D, "epoch_orig": ep0, "epoch_regen": ep1, **r,
                         "paper_cos": PAPER_FIG16.get((h, D), np.nan)})
            nets[h, D] = net
            print(f"h={h} D={D}: cos {r['cos_dist']:.3f} flip {r['flip_pct']:.2f}% identified {r['identified']:.2f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
            pd.DataFrame(rows).to_csv(RESULTS / "fig16_17.csv", index=False)
    grid = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    for ax, col, title in ((axes[0], "cos_dist", "Fig. 16: mean cosine distance, original vs. regenerated"),
                           (axes[1], "flip_pct", "Fig. 17: test correctness disagreement, %")):
        T = grid.pivot(index="hidden", columns="D", values=col)
        im = ax.imshow(T.to_numpy(), cmap="coolwarm", aspect="auto")
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                ax.text(j, i, f"{T.iat[i, j]:.3f}" if col == "cos_dist" else f"{T.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(DIMS)), DIMS)
        ax.set_yticks(range(len(HIDDEN)), HIDDEN)
        ax.set_xlabel("embedding dimension D")
        ax.set_ylabel("hidden layers (width 64)")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig16_17.png")

    # ---- Fig. 15 (official EmbedLLM: 0 hidden layers, D 232) and the LOCUS comparison
    net = nets[0, 232]
    E0, S0 = net.P.weight.detach().numpy(), net.logits(Xt)
    outside = np.setdiff1d(np.arange(len(data.train.Y)), R)
    rng = np.random.default_rng(args.seed)
    settings = [("overlap", a) for a in (0.0, 0.2, 0.4, 0.6, 0.8)] + [("size", n) for n in (512, 768, 1024)]
    sets = {}
    for kind, v in settings:
        if kind == "overlap":
            k = int(round(v * len(R)))
            sets[kind, v] = [np.sort(np.concatenate([rng.choice(R, k, replace=False), rng.choice(outside, len(R) - k, replace=False)]))
                             for _ in range(M)]
        else:
            sets[kind, v] = [np.sort(rng.choice(R, v, replace=False)) for _ in range(M)]
    locus, _ = load_locus(REF, device="cpu")
    Zref = np.load(BANK / "Z_ref.npy")
    L0 = np.load(BANK / "logits_test.npy")
    f15, tsne = [], []
    for (kind, v), qs in sets.items():
        re, _ = regenerate(net, X, pairs(data.train.Y, qs), val, args.seed + 2)
        E1 = re.P.weight.detach().numpy()
        f15.append({"method": "EmbedLLM", "kind": kind, "value": v, **compare(E0, E1, S0, re.logits(Xt), data.test.Y)})
        tsne.append((kind, v, E1))
        if (kind, v) in (("overlap", 0.0), ("size", 1024)):
            Z = embed(locus, data.train.X, data.train.Y, qs, device="cpu", batch_models=8).float().numpy()
            f15.append({"method": "LOCUS", "kind": kind, "value": v,
                        **compare(Zref, Z, L0, score(locus, Z, data.test.X, device="cpu"), data.test.Y)})
        print(f"fig15 {kind} {v} done", flush=True)
    Z = embed(locus, data.train.X, data.train.Y, R, device="cpu", batch_models=8).float().numpy()
    f15.append({"method": "LOCUS", "kind": "same set", "value": 1.0,
                **compare(Zref, Z, L0, score(locus, Z, data.test.X, device="cpu"), data.test.Y)})
    f15 = pd.DataFrame(f15)
    f15.to_csv(RESULTS / "fig15.csv", index=False)

    from openTSNE import TSNE
    sel = [model_index(m) for m in TSNE_MODELS]
    palette = ["#8E44AD", "#C0392B", "#2E8B57", "#E07B1F", "#2253C4"]
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7.5), dpi=150)
    for ax, kind, label in ((axes[0], "overlap", "overlap"), (axes[1], "size", "# queries")):
        pts = [(v, E) for k, v, E in tsne if k == kind]
        Xall = np.vstack([E0] + [E[sel] for _, E in pts])
        Y2 = np.asarray(TSNE(perplexity=30, metric="cosine", random_state=0, n_jobs=args.threads).fit(Xall))
        ref, new = Y2[:M], Y2[M:]
        ax.scatter(*np.delete(ref, sel, axis=0).T, s=12, color="#BBBBBB", label="other models")
        for i, ((v, _), c) in enumerate(zip(pts, palette)):
            ax.scatter(*new[i * len(sel):(i + 1) * len(sel)].T, s=40, color=c, label=f"{label}={v}")
        ax.scatter(*ref[sel].T, s=70, marker="X", color="white", edgecolor="black", label="original (selected)")
        for m, name in zip(sel, TSNE_MODELS):
            ax.annotate(name.split("__")[-1], ref[m], xytext=(6, 6), textcoords="offset points", fontsize=8)
        ax.legend(frameon=False, fontsize=7)
        ax.grid(alpha=0.3, ls="--")
    fig.suptitle("Fig. 15 reproduction: EmbedLLM (0 hidden, D=232), original vs. regenerated (openTSNE, cosine)", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig15.png")

    md = ["# Stage 11: instability of EmbedLLM embeddings", "",
          "Training data: the 4096 reference anchors x 112 models. Test split for flips / accuracy. "
          f"Early stopping on val CE (patience {PATIENCE}), lr {LR:g}.", "",
          f"Paper: Fig. 16 cosine 0.27-0.65 (e.g. 0 hidden: {PAPER_FIG16[(0, 64)]} at D=64, {PAPER_FIG16[(0, 512)]} at D=512); "
          f"Fig. 17 disagreement {PAPER_FIG17_RANGE[0]}-{PAPER_FIG17_RANGE[1]}%.", "",
          "## Figs. 16-17: retrain e with G frozen, same data", "", to_md(grid, "{:.3f}"), "",
          "## Fig. 15 settings and the LOCUS comparison", "",
          "`identified` = fraction of models whose own original embedding is the nearest original to the regenerated one. "
          "LOCUS `same set` re-embeds the exact anchors (deterministic, distance 0 by construction).", "",
          to_md(f15, "{:.3f}"), ""]
    (RESULTS / "stage11_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
