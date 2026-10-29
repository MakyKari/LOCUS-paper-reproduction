#!/usr/bin/env python3
"""
Stage 7: new-model onboarding (paper Sec. 3.2, Fig. 3).

Encoders trained on nested pools of 8..96 models (runs/onboard/m<k>_s42, models 0-95 only) and the
full-pool encoders (runs/main/a1024_s{42,43,44}, held-out models seen in training) embed the 16
held-out models 96-111 from n random train queries each (single forward pass, no training), then
predict correctness on all test queries. 5 query draws per n; the draws are shared by all encoders.

Metrics on the 16 held-out models: correctness accuracy pooled over models x queries (Fig. 3), and
routing hit@1 among the 16 (optional in the plan).
Control (not in the paper): no encoder; the score of (query, model) is the model's accuracy on the query's
dataset, estimated from the same n evaluations (overall accuracy if the dataset is unsampled): "correct"
iff >= 0.5, and routing to the highest score (ties: first model, as in evaluation.py).

Outputs (results/): fig3.csv, fig3.png, stage7_summary.md

Usage (project root):  python repro/experiments/onboarding.py [--device cuda]
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
from repro.common import HELD_OUT, RUNS, RESULTS, _merge_dataset_names, embed, load_data, load_locus, metrics, score  # noqa: E402

NS = [16, 32, 64, 128, 256, 512, 1024]
POOLS = [8, 16, 32, 64, 96]
DRAWS = 5
PAPER_FULL = {16: 0.685, 128: 0.70, 256: 0.706, 1024: 0.707}  # read from Fig. 3 (approx.)


def encoders() -> dict:
    out = {f"{k}": RUNS / "onboard" / f"m{k}_s42" for k in POOLS}
    out.update({f"112 (full, s{s})": RUNS / "main" / f"a1024_s{s}" for s in (42, 43, 44)})
    return {name: d for name, d in out.items() if (d / "training_summary.json").exists()}


def draws(n_train: int) -> dict:
    """(n, draw) -> one index array per held-out model; fixed seeds so every encoder sees the same sets."""
    out = {}
    for n in NS:
        for d in range(DRAWS):
            rng = np.random.default_rng([n, d])
            out[n, d] = [np.sort(rng.choice(n_train, n, replace=False)) for _ in HELD_OUT]
    return out


def profile_baseline(data, sets) -> np.ndarray:
    """[N_test, 16] 0/1 predictions from per-dataset accuracy on each model's own n evaluations."""
    ds_tr = _merge_dataset_names(data.train.category)
    ds_te = _merge_dataset_names(data.test.category)
    pred = np.zeros((len(ds_te), len(HELD_OUT)), dtype=np.float64)
    for j, (m, idx) in enumerate(zip(HELD_OUT, sets)):
        y, g = data.train.Y[idx, m], ds_tr[idx]
        overall = y.mean()
        for ds in np.unique(ds_te):
            sel = g == ds
            acc = y[sel].mean() if sel.any() else overall
            pred[ds_te == ds, j] = acc
    return pred


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.set_num_threads(4)

    data = load_data()
    assert data.model_idx == list(range(112))
    Yt = data.test.Y[:, HELD_OUT]
    sets = draws(len(data.train.Y))
    rows = []
    for name, run in encoders().items():
        model, meta = load_locus(run, device=args.device)
        seen = set(meta.get("model_indices", {}).get("train") or range(112))
        assert name.startswith("112") or not seen & set(HELD_OUT), f"{name}: held-out models were in training"
        for (n, d), qs in sets.items():
            Z = embed(model, data.train.X, data.train.Y, qs, models=HELD_OUT, device=args.device)
            out = metrics(score(model, Z, data.test.X, device=args.device), Yt, data.test.category)
            rows.append({"encoder": name, "n": n, "draw": d, "correctness": out["bp_overall"]["correctness_pred_acc"],
                         "routing16": out["rt_overall"]["hit@1"]})
        print(name, "done", flush=True)
    for (n, d), qs in sets.items():
        out = metrics(profile_baseline(data, qs), Yt, data.test.category, kind="probs")
        rows.append({"encoder": "no encoder: per-dataset accuracy", "n": n, "draw": d,
                     "correctness": out["bp_overall"]["correctness_pred_acc"], "routing16": out["rt_overall"]["hit@1"]})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "fig3.csv", index=False)

    order = ["112 (full, s42)", "112 (full, s43)", "112 (full, s44)"] + [str(k) for k in reversed(POOLS)] + \
            ["no encoder: per-dataset accuracy"]
    order = [o for o in order if o in set(df.encoder)]
    acc = df.groupby(["encoder", "n"]).correctness.mean().unstack()[NS].loc[order]
    rt = df.groupby(["encoder", "n"]).routing16.mean().unstack()[NS].loc[order]

    heat = acc.loc[[o for o in order if o == "112 (full, s42)" or o.isdigit()]]
    fig, ax = plt.subplots(figsize=(6, 3.8), dpi=150)
    im = ax.imshow(heat.to_numpy(), cmap="coolwarm", aspect="auto")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat.iat[i, j]:.3f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(NS)), NS)
    ax.set_yticks(range(len(heat)), ["112 (full)" if i.startswith("112") else i for i in heat.index])
    ax.set_xlabel("# queries in held-out model evaluation set")
    ax.set_ylabel("# models in training data")
    fig.colorbar(im, ax=ax)
    ax.set_title("Fig. 3 reproduction: correctness accuracy, 16 held-out models", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig3.png")

    full = acc.loc[[o for o in order if o.startswith("112")]].mean()
    gap = acc.loc[[o for o in order if o.isdigit()]] - full
    md = ["# Stage 7: new-model onboarding (Fig. 3)", "",
          f"16 held-out models (96-111), correctness accuracy on the test split pooled over models x queries, "
          f"mean over {DRAWS} draws of n train queries per model.", "",
          "Paper (plot, full row): " + ", ".join(f"n={k}: ≈{v}" for k, v in PAPER_FULL.items()) +
          "; claims: ≈128 evaluations suffice, partial vs. full pool gap < 1%.", "",
          "## Correctness accuracy", "", to_md(acc.reset_index(), "{:.4f}"), "",
          "## Pool minus full pool (mean of the 3 full-pool seeds), percentage points", "",
          to_md((gap * 100).reset_index(), "{:+.2f}"), "",
          "## Routing hit@1 among the 16 held-out models", "", to_md(rt.reset_index(), "{:.4f}"), ""]
    (RESULTS / "stage7_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
