"""
Shared helpers for the LOCUS reproduction (Stages 4-12).

Wraps the authors' code in locus_code_release/scripts so that every experiment
loads checkpoints, builds embeddings and computes metrics the same way the
training script does.

    from repro.common import load_data, load_locus, embed, score, metrics

    data = load_data()
    model, meta = load_locus("runs/main/a1024_s42")
    Z = embed(model, data.train.X, data.train.Y, meta_anchors(meta))    # all 112 models, shared anchors
    out = metrics(score(model, Z, data.test.X), data.test.Y, data.test.category)
    out["rt_overall"]["hit@1"], out["bp_overall"]["correctness_pred_acc"]
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "locus_code_release" / "scripts"
RUNS = PROJECT_ROOT / "runs"
RESULTS = PROJECT_ROOT / "results"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from config import DEVICE, DTYPE, EMB_COL, MODEL_ORDER_CSV, USE_AMP  # noqa: E402
from data_embedllm import load_split  # noqa: E402
from evaluation import (  # noqa: E402
    _merge_dataset_names,
    _pair_metrics_from_probs,
    _routing_metrics_from_scores,
    score_queries_against_models,
)
from router_model import Config, ModelEmbedRouter  # noqa: E402

# Fig. 3 held-out models (verified by name in results/data_stats.md)
HELD_OUT: List[int] = list(range(96, 112))


# ================================
# Data
# ================================
@dataclass
class Split:
    X: np.ndarray          # [N, Dq] float32 query encodings
    Y: np.ndarray          # [N, M] uint8 correctness labels (columns = models)
    category: np.ndarray   # [N] raw category strings
    prompt_id: np.ndarray  # [N]


@dataclass
class Data:
    train: Split
    val: Split
    test: Split
    model_idx: List[int]     # model id of each label column
    model_names: List[str]   # aligned with model_idx
    emb_col: str


def load_data(emb_col: str = EMB_COL, model_indices: Optional[Sequence[int]] = None,
              splits: Sequence[str] = ("train", "val", "test")) -> Data:
    """Load concise splits through the authors' load_split (same parsing and model filtering)."""
    loaded: Dict[str, Split] = {}
    label_cols = None
    for s in splits:
        d = load_split(s, emb_col=emb_col, model_indices=list(model_indices) if model_indices else None)
        loaded[s] = Split(X=d["X"], Y=d["Y"], category=np.asarray(d["category"]), prompt_id=d["prompt_id"])
        label_cols = d["label_cols"]
    model_idx = [int(c.split("_")[1]) for c in label_cols]
    names = model_order()
    return Data(train=loaded.get("train"), val=loaded.get("val"), test=loaded.get("test"),
                model_idx=model_idx, model_names=[names[i] for i in model_idx], emb_col=emb_col)


def model_order() -> List[str]:
    """Model names indexed by model_id ('Org__Name' as in model_order.csv)."""
    df = pd.read_csv(MODEL_ORDER_CSV).sort_values("model_id")
    return df["model_name"].astype(str).tolist()


def model_index(name: str) -> int:
    """Accepts 'Org/Name', 'Org__Name' or a unique case-insensitive substring of the name."""
    names = model_order()
    key = name.replace("/", "__")
    if key in names:
        return names.index(key)
    hits = [i for i, n in enumerate(names) if name.lower() in n.lower()]
    if len(hits) != 1:
        raise KeyError(f"{name!r} matches {len(hits)} models: {[names[i] for i in hits][:10]}")
    return hits[0]


def model_name(idx: int) -> str:
    return model_order()[idx]


# ================================
# Checkpoints
# ================================
def load_locus(run_dir: Union[str, Path], device: str = DEVICE) -> Tuple[ModelEmbedRouter, dict]:
    """Rebuild a trained encoder-decoder from a training_script.py output directory.

    The Config is taken from meta.json: the dataclass defaults (n_layers=3, decoder_hidden=256,
    dropout 0) differ from the CLI defaults, so Config(q_dim=...) would not match the checkpoint.
    """
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    model = ModelEmbedRouter(Config(**meta["cfg"]))
    model.encoder.load_state_dict(torch.load(run_dir / "encoder.pt", map_location="cpu", weights_only=True))
    model.decoder.load_state_dict(torch.load(run_dir / "decoder.pt", map_location="cpu", weights_only=True))
    model.to(device=device, dtype=DTYPE).eval()
    return model, meta


def meta_anchors(meta: dict) -> np.ndarray:
    """Train-split query indices the run used as anchors (shared by all models)."""
    return np.asarray(meta["anchors_idx"], dtype=np.int64)


def training_summary(run_dir: Union[str, Path]) -> dict:
    return json.loads((Path(run_dir) / "training_summary.json").read_text())


# ================================
# Embedding
# ================================
@torch.no_grad()
def embed_sets(model: ModelEmbedRouter, sets: Sequence[Tuple[np.ndarray, np.ndarray]],
               device: str = DEVICE, batch_models: int = 64) -> torch.Tensor:
    """Embed models from arbitrary evaluation sets.

    sets[m] = (Q_m [n_m, Dq], y_m [n_m]) with 0/1 (or [0,1]) scores; n_m may differ per model.
    Shorter sets are zero-padded and masked out (masked keys get -inf attention, so padding
    does not change the result). Labels need not be real: hypothetical embeddings (Fig. 11)
    pass synthetic y_m. Returns Z [M, Dz] on `device`.
    """
    model.eval()
    Z = []
    for start in range(0, len(sets), batch_models):
        chunk = sets[start:start + batch_models]
        sizes = [len(q) for q, _ in chunk]
        if min(sizes) == 0:
            raise ValueError("every model needs at least one evaluated query")
        A, Dq = max(sizes), chunk[0][0].shape[1]
        q = np.zeros((len(chunk), A, Dq), dtype=np.float32)
        y = np.zeros((len(chunk), A, 1), dtype=np.float32)
        mask = np.zeros((len(chunk), A), dtype=bool)
        for b, (qm, ym) in enumerate(chunk):
            n = len(qm)
            q[b, :n], y[b, :n, 0], mask[b, :n] = qm, ym, True
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=USE_AMP):
            z = model.embed_models(
                torch.from_numpy(q).to(device=device, dtype=DTYPE),
                torch.from_numpy(y).to(device=device, dtype=DTYPE),
                torch.from_numpy(mask).to(device=device),
            )
        Z.append(z)
    return torch.cat(Z, dim=0)


def embed(model: ModelEmbedRouter, Q: np.ndarray, Y: np.ndarray,
          query_sets: Union[np.ndarray, Sequence[np.ndarray]],
          models: Optional[Sequence[int]] = None, device: str = DEVICE,
          batch_models: int = 64) -> torch.Tensor:
    """Embed models (label columns of Y) from their real evaluations.

    Q: [N, Dq] query encodings (normally data.train.X); Y: [N, M] labels (data.train.Y).
    query_sets: one index array shared by all models, or one index array per model.
    models: label-column indices to embed (default: all columns). Returns Z [len(models), Dz].

    With a shared index array this equals evaluation.embed_all_models (same batching).
    """
    models = list(range(Y.shape[1])) if models is None else list(models)
    if isinstance(query_sets, np.ndarray) and query_sets.ndim == 1:
        query_sets = [query_sets] * len(models)
    if len(query_sets) != len(models):
        raise ValueError(f"{len(query_sets)} query sets for {len(models)} models")
    sets = [(Q[idx], Y[idx, m].astype(np.float32)) for m, idx in zip(models, query_sets)]
    return embed_sets(model, sets, device=device, batch_models=batch_models)


# ================================
# Scoring and metrics
# ================================
def score(model: ModelEmbedRouter, Z: Union[torch.Tensor, np.ndarray], Q: np.ndarray,
          device: str = DEVICE) -> np.ndarray:
    """Correctness logits S [N_queries, M]: S[n, m] = decoder(z_m, q_n)."""
    if isinstance(Z, np.ndarray):
        Z = torch.from_numpy(Z)
    Z = Z.to(device=device, dtype=DTYPE)
    return score_queries_against_models(model, Z, Q, device, q_chunk=1024, m_chunk=256)


def probs_to_logits(p: np.ndarray) -> np.ndarray:
    """For baselines that output probabilities. Monotonic, so routing order and ties are kept."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def metrics(scores: np.ndarray, Y: np.ndarray, category: np.ndarray,
            models: Optional[Sequence[int]] = None, kind: str = "logits",
            threshold: float = 0.5, topk: Tuple[int, ...] = (1, 3, 5)) -> Dict[str, object]:
    """Same outputs as evaluation.compute_val_metrics, computed from a precomputed score matrix.

    scores: [N, M] logits (kind="logits") or probabilities (kind="probs").
    models: restrict correctness and routing to these columns (portfolio / held-out models).
    Keys: bp_overall, rt_overall, bp_by_task, rt_by_task, bp_by_dataset, rt_by_dataset.
    """
    if kind == "probs":
        scores = probs_to_logits(scores)
    elif kind != "logits":
        raise ValueError("kind must be 'logits' or 'probs'")
    scores = np.asarray(scores)
    labels = np.asarray(Y).astype(np.uint8)
    if models is not None:
        models = list(models)
        scores, labels = scores[:, models], labels[:, models]
    if scores.shape != labels.shape:
        raise ValueError(f"scores {scores.shape} vs labels {labels.shape}")

    out: Dict[str, object] = {
        "bp_overall": _pair_metrics_from_probs(scores, labels, thr=threshold),
        "rt_overall": _routing_metrics_from_scores(scores, labels > 0.5, topk=topk),
    }
    for key, groups in (("task", np.asarray(category)), ("dataset", _merge_dataset_names(category))):
        bp, rt = [], []
        for g in np.unique(groups):
            idx = groups == g
            bp.append({key: str(g), **_pair_metrics_from_probs(scores[idx], labels[idx], thr=threshold)})
            rt.append({key: str(g), **_routing_metrics_from_scores(scores[idx], labels[idx] > 0.5, topk=topk)})
        out[f"bp_by_{key}"] = pd.DataFrame(bp).sort_values("num_pairs", ascending=False)
        out[f"rt_by_{key}"] = pd.DataFrame(rt).sort_values("num_queries", ascending=False)
    return out


def summarize(out: Dict[str, object]) -> dict:
    """Same JSON structure as training_summary.json['best_eval'][split]."""
    from training_script import _summarize_eval
    return _summarize_eval(out)


# ================================
# Geometry
# ================================
def distances(Z: Union[torch.Tensor, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Pairwise (cosine distance, Euclidean distance) matrices [M, M] between model embeddings."""
    Z = torch.as_tensor(Z).detach().double().cpu()
    Zn = torch.nn.functional.normalize(Z, dim=1, eps=1e-12)
    cos = (1.0 - Zn @ Zn.T).clamp_min(0.0)
    cos.fill_diagonal_(0.0)
    return cos.numpy(), torch.cdist(Z, Z).numpy()
