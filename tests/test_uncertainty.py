from __future__ import annotations

import numpy as np
import torch

from cadence.model import HierarchicalControlledSSM
from cadence.uncertainty import (
    SimultaneousConformalizer,
    finite_sample_quantile,
    sample_target_intervention_residual,
)


def test_finite_sample_quantile_is_conservative() -> None:
    values = np.arange(10)
    assert finite_sample_quantile(values, 0.9) == 9


def test_conformal_bands_cover_calibration_cases_simultaneously() -> None:
    target = np.arange(60, dtype=float).reshape(5, 4, 3)
    prediction = target + 0.5
    conformal = SimultaneousConformalizer.fit(
        prediction,
        target,
        np.ones(3),
        coverage=0.8,
    )
    lower, upper = conformal.interval(prediction)
    inside = (target >= lower) & (target <= upper)
    assert np.all(inside)


def test_target_residual_sampling_shapes() -> None:
    torch.manual_seed(0)
    model = HierarchicalControlledSSM(
        latent_dim=4,
        input_dim=1,
        behavior_dim=2,
        num_interventions=2,
        hidden_dim=8,
    )
    model.register_animal("d1", 5, donor=True)
    model.register_animal("d2", 6, donor=True)
    model.register_animal("target", 7, donor=False)
    with torch.no_grad():
        list(model.donor_intervention_delta.values())[0].fill_(0.2)
        list(model.donor_intervention_delta.values())[1].fill_(-0.2)
    samples = sample_target_intervention_residual(
        model,
        "target",
        torch.zeros(3, 4),
        torch.zeros(3, 5, 1),
        torch.ones(3, 5, 2),
        draws=11,
        seed=2,
    )
    assert samples.latent.shape == (11, 3, 5, 4)
    assert samples.neural.shape == (11, 3, 5, 7)
    assert samples.behavior.shape == (11, 3, 5, 2)
