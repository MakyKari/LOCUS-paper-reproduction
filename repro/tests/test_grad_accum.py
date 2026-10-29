"""
Tests for the --grad-accum-steps patch in locus_code_release/scripts/train_utils.py.

Claim under test: splitting a batch of models into K micro-batches, weighting each
micro-batch loss by its share of the batch, and taking one optimizer step gives the
same update as the upstream single-batch step.

The comparison uses plain SGD (lr=1), so the parameter change equals the clipped
gradient. AdamW would hide small differences: its first step is roughly lr*sign(g),
so float rounding on near-zero gradients could flip whole elements.

Dropout and input noise are turned off, because they draw different random numbers
when the batch is split.

Exactness holds while the encoder sees every anchor, i.e. anchors_per_model <=
encoder_min_anchors (1024 by default). For 2048/4096 anchors the upstream code
samples the anchor count K ~ U[1024, A] once per forward pass, so each micro-batch
gets its own random subset. That is equivalent in expectation, not bit-for-bit,
and is not covered here.

Run from the project root:
    python -m pytest repro/tests/test_grad_accum.py -v
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

SCRIPTS = Path(__file__).resolve().parents[2] / "locus_code_release" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from config import DEVICE, DTYPE  # noqa: E402
from dataset_model_batches import ModelBatchDataset, collate_batch  # noqa: E402
from router_model import Config, ModelEmbedRouter  # noqa: E402
from train_utils import train_epoch  # noqa: E402
from utils import to_device_with_dtype  # noqa: E402

N_MODELS = 112
N_QUERIES = 1000
Q_DIM = 768
ANCHORS = 256
REL_TOL = 1e-4  # max |delta_K - delta_1| relative to max |delta_1|


# ---------------------------------------------------------------- fixtures / helpers
def make_data(seed: int = 0):
    """Synthetic EmbedLLM-like data: unit-norm query encodings, model-dependent labels."""
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((N_QUERIES, Q_DIM)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    W = rng.standard_normal((N_MODELS, Q_DIM)).astype(np.float32) * 3.0
    bias = rng.normal(-0.4, 0.5, size=(N_MODELS, 1)).astype(np.float32)
    P = 1.0 / (1.0 + np.exp(-(W @ Q.T + bias)))  # [M, Q]
    Y = (rng.random(P.shape) < P).astype(np.float32)
    anchors_idx = np.sort(rng.choice(N_QUERIES, size=ANCHORS, replace=False))
    return Q, Y, anchors_idx


def make_loader(Q, Y, anchors_idx, batch_size: int, rng_seed: int = 42):
    """Same construction as make_loaders() for the train split."""
    ds = ModelBatchDataset(
        Y_anchor=Y, Qenc_anchor=Q, anchors_idx=anchors_idx, Y_target=Y, Qenc_target=Q,
        num_target_per_model=ANCHORS, rng=np.random.default_rng(rng_seed), targets_from_anchors=True,
    )
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0,
                                       collate_fn=collate_batch, drop_last=False)


def make_cfg(**overrides) -> Config:
    kw = dict(
        q_dim=Q_DIM, z_dim=128, n_heads=4, n_layers=2, encoder_dropout=0.0, decoder_hidden=64,
        decoder_dropout=0.0, encoder_kind="isab", m_induce=64,
        encoder_min_anchors=1024, encoder_max_anchors=ANCHORS,  # training_script defaults
        add_noise_encoder_q=False, add_noise_decoder_q=False, add_noise_decoder_z=False,
    )
    kw.update(overrides)
    return Config(**kw)


@pytest.fixture(scope="module")
def data():
    return make_data()


@pytest.fixture(scope="module")
def init_state():
    torch.manual_seed(0)
    return copy.deepcopy(ModelEmbedRouter(make_cfg()).state_dict())


def run_one_epoch(data, init_state, k: int, batch_size: int = 128, lr: float = 1.0,
                  cfg: Config | None = None, epoch_fn=None):
    """Load the same init, train one epoch, return (param deltas, metrics)."""
    Q, Y, anchors_idx = data
    cfg = cfg or make_cfg()
    model = ModelEmbedRouter(cfg).to(DEVICE)
    model.load_state_dict(init_state)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loader = make_loader(Q, Y, anchors_idx, batch_size)
    torch.manual_seed(1)  # same DataLoader shuffle order for every k
    if epoch_fn is None:
        metrics = train_epoch(model, loader, opt, DEVICE, cfg, pos_weight=None, grad_accum_steps=k)
    else:
        metrics = epoch_fn(model, loader, opt, DEVICE, cfg)
    deltas = {n: (p.detach() - before[n]) for n, p in model.named_parameters()}
    return deltas, metrics


def max_rel_diff(d_ref: dict, d_new: dict) -> float:
    scale = max(float(v.abs().max()) for v in d_ref.values())
    diff = max(float((d_ref[n] - d_new[n]).abs().max()) for n in d_ref)
    return diff / scale


def upstream_train_epoch(model, loader, optimizer, device, cfg):
    """Verbatim logic of train_epoch at upstream commit e4ebf69 (no pos_weight, no AMP)."""
    model.train()
    metrics = {"loss": 0.0, "bce": 0.0}
    n_batches = 0
    for batch in loader:
        batch = to_device_with_dtype(batch, device, DTYPE)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        loss_bce = nn.BCEWithLogitsLoss()(out["logits"], batch["y_target"])
        loss_bce.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        metrics["loss"] += loss_bce.detach().item()
        metrics["bce"] += loss_bce.detach().item()
        n_batches += 1
    for k in metrics:
        metrics[k] /= max(1, n_batches)
    return metrics


# ---------------------------------------------------------------- tests
def test_cli_flag_exists():
    from training_script import build_argparser

    args = build_argparser().parse_args(["--grad-accum-steps", "2"])
    assert args.grad_accum_steps == 2
    assert build_argparser().parse_args([]).grad_accum_steps == 1  # default keeps upstream behaviour


def test_k1_matches_upstream(data, init_state):
    """K=1 must reproduce the original single-batch train_epoch."""
    d_up, m_up = run_one_epoch(data, init_state, k=1, epoch_fn=upstream_train_epoch)
    d_k1, m_k1 = run_one_epoch(data, init_state, k=1)
    assert max_rel_diff(d_up, d_k1) < 1e-6
    assert m_k1["bce"] == pytest.approx(m_up["bce"], rel=1e-6)


@pytest.mark.parametrize("k", [2, 3, 4, 7])
def test_accumulated_step_equals_full_batch(data, init_state, k):
    """One batch of 112 models: K micro-batches (even and uneven splits) give the same update."""
    d_1, m_1 = run_one_epoch(data, init_state, k=1)
    d_k, m_k = run_one_epoch(data, init_state, k=k)
    rel = max_rel_diff(d_1, d_k)
    assert rel < REL_TOL, f"K={k}: relative parameter-update difference {rel:.2e}"
    assert m_k["bce"] == pytest.approx(m_1["bce"], rel=1e-5)


def test_equivalence_with_multiple_batches_per_epoch(data, init_state):
    """batch_size 64 -> 2 optimizer steps per epoch; accumulation must not change either step."""
    d_1, m_1 = run_one_epoch(data, init_state, k=1, batch_size=64, lr=0.05)
    d_2, m_2 = run_one_epoch(data, init_state, k=2, batch_size=64, lr=0.05)
    assert max_rel_diff(d_1, d_2) < REL_TOL
    assert m_2["bce"] == pytest.approx(m_1["bce"], rel=1e-5)


def test_runs_with_default_noise_and_dropout(data):
    """Real training config (dropout 0.1, query noise): K=2 runs and gives a finite loss."""
    cfg = make_cfg(encoder_dropout=0.1, decoder_dropout=0.1, add_noise_encoder_q=True,
                   add_noise_decoder_q=True, noise_encoder_q_std=0.1, noise_decoder_q_std=0.1)
    torch.manual_seed(0)  # dropout adds nn.Dropout layers, so it needs its own init state
    state = copy.deepcopy(ModelEmbedRouter(cfg).state_dict())
    _, m = run_one_epoch(data, state, k=2, lr=2e-4, cfg=cfg)
    assert np.isfinite(m["bce"]) and 0.0 < m["bce"] < 5.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
