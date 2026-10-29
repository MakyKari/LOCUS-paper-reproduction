#!/usr/bin/env python3
"""
Stage 12: encoder and decoder wall-clock time (R13, Fig. 18), reference encoder runs/ref/a4096_s42.

Encoder: model.embed_models for |M| in {1, 16, 32, 64, 96, 112} models x n in {256, ..., 4096} queries
per model, inputs already on the GPU. Decoder: logits for 112 models x 4096 queries, the same
computation as evaluation.score_queries_against_models (U(q), [z, u, z*u], mlp_head) without the
copy back to the host; the authors' function itself (with the copy) is timed as well.

Each point: 10 warm-up runs, then the median of 50 runs, torch.cuda.synchronize() around each.
If a configuration does not fit in GPU memory (peak allocation above 85% of the memory free at start, checked on a
trial run because Windows spills to system memory instead of raising OOM), the models (decoder: the
queries) are processed in chunks (column `model_chunk`; the time is the total over chunks). The paper's numbers are on a V100, ours on an
RTX 3050 Laptop (4 GB), so only the scaling is comparable.

Outputs (results/): timing.csv, fig18.png, stage12_summary.md

Usage (project root):  python repro/experiments/timing.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repro.collect_results import to_md  # noqa: E402
from repro.common import DTYPE, RESULTS, RUNS, load_data, load_locus, score  # noqa: E402

REF = RUNS / "ref" / "a4096_s42"
MODELS = [1, 16, 32, 64, 96, 112]
QUERIES = [256, 512, 1024, 2048, 4096]
WARMUP, REPEATS = 10, 50
MEM_FRACTION = 0.85  # peak allocation allowed, as a fraction of the memory free at start
USABLE = [0]         # bytes free on the card at start (other processes, e.g. the desktop, use some)
PAPER = {"encoder_112x4096_ms": 104, "decoder_112x4096_ms": 18}  # Fig. 18 / App., V100


def fits(fn) -> bool:
    """One trial run; on Windows (WDDM) the driver spills to shared system memory instead of raising
    OOM, which made runs up to ~65x slower, so the peak allocation is checked against the card."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        fn()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        return False
    return torch.cuda.max_memory_allocated() <= MEM_FRACTION * USABLE[0]


def bench(fn) -> float:
    """Median wall-clock time of fn() in ms."""
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


@torch.no_grad()
def time_encoder(model, q, y, mask) -> tuple[float, int]:
    """Try all models in one batch, halve the chunk while it does not fit in GPU memory."""
    M = q.shape[0]
    chunk = M
    while chunk >= 1:
        def run():
            for s in range(0, M, chunk):
                model.embed_models(q[s:s + chunk], y[s:s + chunk], mask[s:s + chunk])
        if fits(lambda: model.embed_models(q[:chunk], y[:chunk], mask[:chunk])):
            return bench(run), chunk
        chunk //= 2
    raise RuntimeError("encoder does not fit even with one model per chunk")


@torch.no_grad()
def time_decoder(model, Z, Q) -> tuple[float, int]:
    """GPU-only decoder for all (query, model) pairs; chunks over queries while it does not fit."""
    dec = model.decoder
    N, (M, Dz) = Q.shape[0], Z.shape
    chunk = N
    while chunk >= 1:
        def run(n_end=N):
            for s in range(0, n_end, chunk):
                qb = Q[s:s + chunk]
                u = dec.U(qb).unsqueeze(1).expand(-1, M, Dz)
                z = Z.unsqueeze(0).expand(qb.shape[0], M, Dz)
                dec.mlp_head(torch.cat([z, u, z * u], dim=-1).reshape(-1, 3 * Dz))
        if fits(lambda: run(chunk)):
            return bench(run), chunk
        chunk //= 2
    raise RuntimeError("decoder does not fit")


def plot(enc: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # |M| is a magnitude: one hue, light -> dark (blue ramp steps 250-650 of the default palette)
    blues = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#184f95", "#104281"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    for m, c in zip(MODELS, blues):
        s = enc[enc.models == m]
        ax.plot(s.queries, s.ms, color=c, lw=2, marker="o", ms=5, label=f"|M| = {m}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(QUERIES, [str(x) for x in QUERIES])
    ax.set_xlabel("queries per model (n)")
    ax.set_ylabel("encoder time, ms (median of 50)")
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, title="models", title_fontsize=8)
    ax.set_title(f"Fig. 18 reproduction: encoder time, {torch.cuda.get_device_name(0)}", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig18.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(REF))
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    dev = "cuda"
    torch.backends.cudnn.benchmark = False
    USABLE[0] = torch.cuda.mem_get_info()[0]
    print(f"GPU memory free at start: {USABLE[0] / 2**30:.2f} GiB", flush=True)

    data = load_data(splits=("train", "test"))
    model, meta = load_locus(args.run, device=dev)
    rows = []
    for n in QUERIES:
        q_all = torch.from_numpy(data.train.X[:n]).to(dev, DTYPE)
        y_all = torch.from_numpy(data.train.Y[:n].T.astype(np.float32)).to(dev, DTYPE)   # [112, n]
        for m in MODELS:
            q = q_all.unsqueeze(0).expand(m, n, q_all.shape[1]).contiguous()
            y = y_all[:m].unsqueeze(-1).contiguous()
            mask = torch.ones(m, n, dtype=torch.bool, device=dev)
            ms, chunk = time_encoder(model, q, y, mask)
            rows.append({"part": "encoder", "models": m, "queries": n, "ms": ms, "model_chunk": chunk})
            print(f"encoder |M|={m:<3} n={n:<4}: {ms:8.2f} ms (chunk {chunk})", flush=True)
            del q, y, mask
            torch.cuda.empty_cache()

    # Decoder: the 112 reference embeddings x the first 4096 train queries
    n = 4096
    with torch.no_grad():
        anchors = torch.as_tensor(meta["anchors_idx"])
        q = torch.from_numpy(data.train.X[anchors.numpy()]).to(dev, DTYPE).unsqueeze(0).expand(len(data.model_names), -1, -1)
        y = torch.from_numpy(data.train.Y[anchors.numpy()].T.astype(np.float32)).to(dev, DTYPE).unsqueeze(-1)
        Z = torch.cat([model.embed_models(q[s:s + 16], y[s:s + 16], torch.ones(y[s:s + 16].shape[:2], dtype=torch.bool, device=dev))
                       for s in range(0, len(y), 16)])
        del q, y
        torch.cuda.empty_cache()
    Q = torch.from_numpy(data.train.X[:n]).to(dev, DTYPE)
    ms, chunk = time_decoder(model, Z, Q)
    rows.append({"part": "decoder", "models": len(Z), "queries": n, "ms": ms, "model_chunk": chunk})
    print(f"decoder 112 x {n}: {ms:.2f} ms (query chunk {chunk})", flush=True)
    Qnp = data.train.X[:n]
    ms_auth = bench(lambda: score(model, Z, Qnp, device=dev))
    rows.append({"part": "decoder_authors_fn", "models": len(Z), "queries": n, "ms": ms_auth, "model_chunk": 1024})
    print(f"decoder via score_queries_against_models: {ms_auth:.2f} ms", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "timing.csv", index=False)
    enc = df[df.part == "encoder"]
    plot(enc)

    piv = enc.pivot_table(index="models", columns="queries", values="ms").round(2)
    piv.columns = [f"n={c}" for c in piv.columns]
    e112 = float(enc[(enc.models == 112) & (enc.queries == 4096)].ms.iloc[0])
    lin = enc[enc.models == 112].set_index("queries").ms
    flat = enc[enc.models == 1].set_index("queries").ms
    md = ["# Stage 12: timing", "",
          f"GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}, dtype {DTYPE}; "
          f"median of {REPEATS} runs after {WARMUP} warm-ups, ms. Paper: V100.", "",
          "## Encoder (ms)", "", to_md(piv.reset_index()), "",
          f"- 112 x 4096: {e112:.1f} ms (paper ≈{PAPER['encoder_112x4096_ms']} ms), "
          f"model chunk {int(enc[(enc.models == 112) & (enc.queries == 4096)].model_chunk.iloc[0])}.",
          f"- |M| = 112: time ratio n=4096 / n=256 = {lin[4096] / lin[256]:.1f} (linear would be 16).",
          f"- |M| = 1: time ratio n=4096 / n=256 = {flat[4096] / flat[256]:.1f}.",
          "- Chunked over models (did not fit at once): "
          + (", ".join(f"{r.models}x{r.queries} in chunks of {r.model_chunk}"
                       for r in enc[enc.model_chunk < enc.models].itertuples()) or "none") + ".", "",
          "## Decoder, 112 models x 4096 queries", "",
          f"- GPU only: {ms:.1f} ms (query chunk {chunk}); paper ≈{PAPER['decoder_112x4096_ms']} ms.",
          f"- Through the authors' score_queries_against_models (chunks 1024 x 256, copies to host): {ms_auth:.1f} ms.", ""]
    (RESULTS / "stage12_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
