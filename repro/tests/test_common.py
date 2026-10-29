"""
Tests for repro/common.py against the authors' pipeline.

The key test reloads a finished run and must reproduce its training_summary.json
(test metrics at the best-val epoch), overall and per dataset.

Run from the project root (uses runs/smoke by default; any finished run works):
    python -m pytest repro/tests/test_common.py -v
    $env:LOCUS_RUN = "runs/main/a1024_s42"; python -m pytest repro/tests/test_common.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common  # noqa: E402
from evaluation import compute_val_metrics, embed_all_models  # noqa: E402

RUN_DIR = common.PROJECT_ROOT / os.environ.get("LOCUS_RUN", "runs/smoke")

pytestmark = pytest.mark.skipif(not (RUN_DIR / "training_summary.json").exists(),
                                reason=f"no finished run at {RUN_DIR}")


@pytest.fixture(scope="module")
def run():
    model, meta = common.load_locus(RUN_DIR)
    idx = meta.get("model_indices", {}).get("train")
    data = common.load_data(emb_col=meta["emb_col"], model_indices=idx)
    anchors = common.meta_anchors(meta)
    Z = common.embed(model, data.train.X, data.train.Y, anchors)
    return model, meta, data, anchors, Z


def test_config_comes_from_meta(run):
    model, meta, *_ = run
    assert model.cfg.n_layers == meta["cfg"]["n_layers"]
    assert model.cfg.decoder_hidden == meta["cfg"]["decoder_hidden"]
    assert not model.training


def test_embed_matches_embed_all_models(run):
    model, _, data, anchors, Z = run
    Y_tr = data.train.Y.T.astype(np.float32)  # embed_all_models expects [M, Q]
    Z_ref = embed_all_models(model, data.train.X, Y_tr, anchors, common.DEVICE, batch_models=64)
    assert torch.equal(Z, Z_ref), f"max diff {float((Z - Z_ref).abs().max()):.2e}"


def test_metrics_match_compute_val_metrics(run):
    model, _, data, _, Z = run
    ref = compute_val_metrics(model, Z, data.test.X, data.test.Y, data.test.category, common.DEVICE)
    out = common.metrics(common.score(model, Z, data.test.X), data.test.Y, data.test.category)
    assert out["rt_overall"] == ref["rt_overall"]
    assert out["bp_overall"] == ref["bp_overall"]


@pytest.mark.parametrize("split", ["val", "test"])
def test_reproduces_training_summary(run, split):
    """Reloaded checkpoint must give the numbers logged at the best epoch."""
    model, _, data, _, Z = run
    s = getattr(data, split)
    got = common.summarize(common.metrics(common.score(model, Z, s.X), s.Y, s.category))
    want = common.training_summary(RUN_DIR)["best_eval"][split]

    for k in ("hit@1", "hit@3", "hit@5"):
        assert got["overall"]["routing"][k] == pytest.approx(want["overall"]["routing"][k], abs=1e-9), k
    for k in ("correctness_pred_acc", "bce_per_pair", "roc_auc"):
        assert got["overall"]["correctness_prediction"][k] == pytest.approx(
            want["overall"]["correctness_prediction"][k], abs=1e-6), k

    w_rt = {r["dataset"]: r for r in want["by_dataset"]["routing"]}
    w_bp = {r["dataset"]: r for r in want["by_dataset"]["correctness_prediction"]}
    for r in got["by_dataset"]["routing"]:
        assert r["hit@1"] == pytest.approx(w_rt[r["dataset"]]["hit@1"], abs=1e-9), r["dataset"]
    for r in got["by_dataset"]["correctness_prediction"]:
        assert r["correctness_pred_acc"] == pytest.approx(
            w_bp[r["dataset"]]["correctness_pred_acc"], abs=1e-6), r["dataset"]


def test_padding_does_not_change_embedding(run):
    """A model embedded alone equals the same model embedded next to a larger (padded) set."""
    model, _, data, anchors, _ = run
    rng = np.random.default_rng(0)
    small = rng.choice(len(data.train.X), size=50, replace=False)
    large = rng.choice(len(data.train.X), size=300, replace=False)
    alone = common.embed(model, data.train.X, data.train.Y, [small], models=[0])
    padded = common.embed(model, data.train.X, data.train.Y, [small, large], models=[0, 1])
    assert torch.allclose(alone[0], padded[0], atol=1e-5), float((alone[0] - padded[0]).abs().max())


def test_embedding_is_order_invariant(run):
    model, _, data, anchors, _ = run
    perm = np.random.default_rng(1).permutation(anchors)
    a = common.embed(model, data.train.X, data.train.Y, anchors, models=[3])
    b = common.embed(model, data.train.X, data.train.Y, perm, models=[3])
    assert torch.allclose(a, b, atol=1e-5)


def test_model_subset_and_probs(run):
    model, _, data, _, Z = run
    S = common.score(model, Z, data.test.X)
    subset = [0, 5, 10]
    a = common.metrics(S, data.test.Y, data.test.category, models=subset)
    b = common.metrics(S[:, subset], data.test.Y[:, subset], data.test.category)
    assert a["rt_overall"] == b["rt_overall"]
    p = 1.0 / (1.0 + np.exp(-np.clip(S.astype(np.float64), -40, 40)))
    c = common.metrics(p, data.test.Y, data.test.category, kind="probs")
    d = common.metrics(S, data.test.Y, data.test.category)
    assert c["rt_overall"]["hit@1"] == pytest.approx(d["rt_overall"]["hit@1"])
    assert c["bp_overall"]["correctness_pred_acc"] == pytest.approx(d["bp_overall"]["correctness_pred_acc"])


def test_model_names_and_distances(run):
    _, _, data, _, Z = run
    assert common.model_index("Q-bert/Optimus-7B") == 111
    assert common.model_name(0) == "Qwen__Qwen1.5-7B-Chat"
    assert data.model_names[0] == common.model_name(data.model_idx[0])
    cos, euc = common.distances(Z)
    M = Z.shape[0]
    assert cos.shape == euc.shape == (M, M)
    assert np.allclose(np.diag(cos), 0) and np.allclose(np.diag(euc), 0, atol=1e-6)
    assert np.allclose(cos, cos.T) and (cos >= 0).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
