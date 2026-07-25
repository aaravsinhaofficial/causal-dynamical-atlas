from __future__ import annotations

import copy

import pytest
import torch

from cadence.model import HierarchicalControlledSSM, SequenceBatch


def make_model() -> HierarchicalControlledSSM:
    torch.manual_seed(4)
    model = HierarchicalControlledSSM(
        latent_dim=5,
        input_dim=2,
        behavior_dim=3,
        num_interventions=2,
        hidden_dim=16,
        residual_rank=2,
        dt=0.1,
    )
    model.register_animal("donor", neural_dim=7, donor=True)
    model.register_animal("target", neural_dim=11, donor=False)
    return model


def make_batch(animal: str = "donor", neural_dim: int = 7) -> SequenceBatch:
    generator = torch.Generator().manual_seed(8)
    neural = torch.randn(4, 12, neural_dim, generator=generator)
    behavior = torch.randn(4, 12, 3, generator=generator)
    inputs = torch.randn(4, 12, 2, generator=generator)
    intervention = torch.zeros(4, 12, 2)
    intervention[:, 5:9, 0] = 1
    return SequenceBatch(animal, neural, behavior, inputs, intervention, onset=5)


def test_variable_neural_dimensions() -> None:
    model = make_model()
    z = torch.randn(3, 5)
    donor_neural, donor_behavior = model.decode("donor", z)
    target_neural, target_behavior = model.decode("target", z)
    assert donor_neural.shape == (3, 7)
    assert target_neural.shape == (3, 11)
    assert donor_behavior.shape == target_behavior.shape == (3, 3)


def test_sessions_can_share_animal_intervention_residual() -> None:
    model = make_model()
    model.register_animal("donor-session-2", 5, donor=True, intervention_group="donor")
    assert len(model.donor_intervention_delta) == 1


def test_donor_delta_projection_is_exact_and_scoped() -> None:
    model = make_model()
    model.register_animal("donor-b", 6, donor=True)
    model.register_animal("validation", 5, donor=True)
    keys = tuple(model.donor_intervention_delta)
    training_keys = keys[:2]
    validation_key = keys[2]
    with torch.no_grad():
        model.donor_intervention_delta[training_keys[0]].fill_(1.0)
        model.donor_intervention_delta[training_keys[1]].fill_(3.0)
        model.donor_intervention_delta[validation_key].fill_(7.0)

    residual = model.project_donor_deltas_zero_mean(training_keys)

    assert residual < 1e-7
    projected = torch.stack([model.donor_intervention_delta[key] for key in training_keys])
    torch.testing.assert_close(projected.mean(dim=0), torch.zeros_like(projected[0]))
    torch.testing.assert_close(
        model.donor_intervention_delta[validation_key],
        torch.full_like(model.donor_intervention_delta[validation_key], 7.0),
    )


def test_intervention_loss_cannot_see_post_onset_inputs() -> None:
    model = make_model().eval()
    first = make_batch()
    second = copy.deepcopy(first)
    second.neural[:, second.onset :] = 1e6
    second.behavior[:, second.onset :] = -1e6

    captured: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        captured.append(args[0].detach().clone())

    handle = model.adapter("donor").encoder.register_forward_pre_hook(hook)
    model.intervention_loss(first, include_donor_delta=False)
    model.intervention_loss(second, include_donor_delta=False)
    handle.remove()
    assert len(captured) == 2
    torch.testing.assert_close(captured[0], captured[1])


def test_post_onset_targets_change_loss_but_not_predictions() -> None:
    model = make_model().eval()
    first = make_batch()
    second = copy.deepcopy(first)
    second.neural[:, second.onset :] += 10
    a = model.intervention_loss(first, include_donor_delta=False)
    b = model.intervention_loss(second, include_donor_delta=False)
    assert not torch.isclose(a.neural, b.neural)


def test_normal_loss_rejects_intervention_trials() -> None:
    model = make_model()
    with pytest.raises(ValueError, match="nonzero intervention"):
        model.normal_loss(make_batch())


def test_stage_freezing_matches_protocol() -> None:
    model = make_model()
    model.configure_stage("intervention")
    assert any(p.requires_grad for p in model.operator.parameters())
    assert not any(p.requires_grad for p in model.shared.parameters())
    assert not any(p.requires_grad for p in model.adapters.parameters())

    model.configure_stage("target_adaptation", target_animal="target")
    assert any(p.requires_grad for p in model.adapter("target").parameters())
    assert not any(p.requires_grad for p in model.adapter("donor").parameters())
    assert not any(p.requires_grad for p in model.operator.parameters())


def test_rollout_horizon_and_validation() -> None:
    model = make_model()
    batch = make_batch()
    z0 = torch.zeros(4, 5)
    latent, neural, behavior = model.rollout(
        "donor",
        z0,
        batch.inputs[:, 4:-1],
        batch.intervention[:, 4:-1],
    )
    assert latent.shape == (4, 7, 5)
    assert neural.shape == (4, 7, 7)
    assert behavior.shape == (4, 7, 3)
