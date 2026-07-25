from __future__ import annotations

import numpy as np
import pytest
import torch

from cadence.model import HierarchicalControlledSSM, SequenceBatch
from cadence.training import FitConfig, fit_stage, save_frozen_prediction_bundle


def model_and_normal_batch(animal: str = "a") -> tuple[HierarchicalControlledSSM, SequenceBatch]:
    torch.manual_seed(1)
    model = HierarchicalControlledSSM(
        latent_dim=3,
        input_dim=1,
        behavior_dim=1,
        num_interventions=1,
        hidden_dim=8,
    )
    model.register_animal(animal, 4, donor=animal != "target")
    batch = SequenceBatch(
        animal,
        torch.randn(2, 6, 4),
        torch.randn(2, 6, 1),
        torch.randn(2, 6, 1),
        torch.zeros(2, 6, 1),
        onset=3,
    )
    return model, batch


def test_target_adaptation_rejects_donor_batch() -> None:
    model, target = model_and_normal_batch("target")
    _, donor = model_and_normal_batch("donor")
    with pytest.raises(ValueError, match="only to the target"):
        fit_stage(
            model,
            [target, donor],
            [target],
            stage="target_adaptation",
            target_animal="target",
            config=FitConfig(max_epochs=1, patience=1, device="cpu"),
        )


def test_small_normal_fit_runs() -> None:
    model, batch = model_and_normal_batch()
    result = fit_stage(
        model,
        [batch],
        [batch],
        stage="normal",
        config=FitConfig(max_epochs=2, patience=2, device="cpu", mixed_precision=False),
    )
    assert result.best_epoch >= 0
    assert np.isfinite(result.best_validation_loss)


def test_prediction_bundle_is_hashed(tmp_path) -> None:
    path = tmp_path / "sealed.npz"
    digest = save_frozen_prediction_bundle(
        path,
        predictions={"neural": np.zeros((2, 3))},
        config={"latent_dim": 4},
        split_manifest={"test_mice": ["m1"]},
        git_commit="abc",
    )
    assert len(digest) == 64
    assert path.exists()
    assert path.with_suffix(".npz.sha256").read_text().startswith(digest)
