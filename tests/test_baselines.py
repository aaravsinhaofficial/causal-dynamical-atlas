from __future__ import annotations

import numpy as np
import pytest
import torch

from cadence.baselines import (
    AdditiveInterventionSSM,
    BlackBoxMetaGRU,
    ConditionTimeTemplate,
    LinearHierarchicalSSM,
)


@pytest.mark.parametrize(
    "model_type",
    [LinearHierarchicalSSM, AdditiveInterventionSSM, BlackBoxMetaGRU],
)
def test_baseline_rollout_shapes(model_type) -> None:
    model = model_type(
        latent_dim=4,
        input_dim=2,
        behavior_dim=1,
        num_interventions=3,
        hidden_dim=12,
    )
    model.register_animal("a", 9, donor=True)
    latent, neural, behavior = model.rollout(
        "a",
        torch.zeros(2, 4),
        torch.zeros(2, 7, 2),
        torch.zeros(2, 7, 3),
    )
    assert latent.shape == (2, 7, 4)
    assert neural.shape == (2, 7, 9)
    assert behavior.shape == (2, 7, 1)


def test_black_box_transition_preserves_nested_batch_axes() -> None:
    model = BlackBoxMetaGRU(
        latent_dim=4,
        input_dim=2,
        behavior_dim=1,
        num_interventions=3,
        hidden_dim=12,
    )
    model.register_animal("a", 9, donor=True)
    z = torch.randn(2, 5, 4, requires_grad=True)
    inputs = torch.randn(2, 5, 2)
    intervention = torch.randn(2, 5, 3)

    nested = model.transition(
        "a",
        z,
        inputs,
        intervention,
        include_animal_residual=False,
        include_donor_delta=False,
    )
    flattened = model.transition(
        "a",
        z.reshape(-1, 4),
        inputs.reshape(-1, 2),
        intervention.reshape(-1, 3),
        include_animal_residual=False,
        include_donor_delta=False,
    ).reshape_as(z)

    assert nested.shape == z.shape
    torch.testing.assert_close(nested, flattened)
    nested.sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_template_averages_animals_equally() -> None:
    effects = np.array([[[1.0]], [[1.0]], [[1.0]], [[9.0]]])
    descriptors = np.ones((4, 1))
    animals = np.array(["many", "many", "many", "few"])
    template = ConditionTimeTemplate.fit(effects, descriptors, animals)
    assert template.predict([[1]])[0, 0, 0] == pytest.approx(5)
