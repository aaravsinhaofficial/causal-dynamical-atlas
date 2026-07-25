"""Hierarchical controlled latent dynamics.

The implementation deliberately separates two optimization stages:

1. animal observation maps and residual dynamics are learned from normal
   sequences only;
2. those animal-specific parameters are frozen before intervention outcomes
   are used to fit the shared intervention operator.

The same normal-only adaptation routine is used for a held-out animal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _module_key(animal_id: str | int) -> str:
    """Return a ModuleDict-safe, reversible-enough key."""
    return f"animal_{str(animal_id).replace('.', '_').replace('/', '_')}"


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        *,
        final_activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.final_activation = final_activation

    def forward(self, x: Tensor) -> Tensor:
        y = self.net(x)
        return self.final_activation(y) if self.final_activation is not None else y


class LowRankResidual(nn.Module):
    """Strongly capacity-limited animal-specific normal dynamics."""

    def __init__(self, latent_dim: int, rank: int) -> None:
        super().__init__()
        self.left = nn.Parameter(torch.zeros(latent_dim, rank))
        self.right = nn.Parameter(torch.zeros(rank, latent_dim))
        nn.init.normal_(self.left, std=0.01)
        nn.init.normal_(self.right, std=0.01)

    def forward(self, z: Tensor) -> Tensor:
        return torch.tanh(z) @ self.right.T @ self.left.T

    def squared_norm(self) -> Tensor:
        return self.left.square().sum() + self.right.square().sum()


class AnimalObservationMap(nn.Module):
    """Animal-specific encoder and neural decoder fitted on normal data only."""

    def __init__(
        self,
        neural_dim: int,
        behavior_dim: int,
        latent_dim: int,
        hidden_dim: int,
        residual_rank: int,
    ) -> None:
        super().__init__()
        self.neural_dim = neural_dim
        self.encoder = MLP(neural_dim + behavior_dim, 2 * latent_dim, hidden_dim)
        self.neural_decoder = MLP(latent_dim, neural_dim, hidden_dim)
        self.behavior_log_scale = nn.Parameter(torch.zeros(behavior_dim))
        self.behavior_bias = nn.Parameter(torch.zeros(behavior_dim))
        self.residual = LowRankResidual(latent_dim, residual_rank)

    def encode(self, neural: Tensor, behavior: Tensor, *, sample: bool) -> tuple[Tensor, Tensor]:
        stats = self.encoder(torch.cat((neural, behavior), dim=-1))
        mean, raw_scale = stats.chunk(2, dim=-1)
        log_var = raw_scale.clamp(-8.0, 4.0)
        latent = mean + torch.randn_like(mean) * torch.exp(0.5 * log_var) if sample else mean
        return latent, log_var

    def decode_neural(self, z: Tensor) -> Tensor:
        return self.neural_decoder(z)

    def calibrate_behavior(self, shared_behavior: Tensor) -> Tensor:
        return shared_behavior * torch.exp(self.behavior_log_scale) + self.behavior_bias


class SharedDynamics(nn.Module):
    """Residual discretization of a shared nonlinear normal-flow operator."""

    def __init__(self, latent_dim: int, input_dim: int, hidden_dim: int, dt: float) -> None:
        super().__init__()
        self.dt = dt
        self.flow = MLP(latent_dim + input_dim, latent_dim, hidden_dim)

    def forward(self, z: Tensor, inputs: Tensor) -> Tensor:
        drift = -z + self.flow(torch.cat((z, inputs), dim=-1))
        return z + self.dt * drift


class SharedInterventionOperator(nn.Module):
    """State-dependent low-rank intervention vector fields."""

    def __init__(
        self,
        latent_dim: int,
        num_interventions: int,
        rank: int,
        dt: float,
    ) -> None:
        super().__init__()
        self.dt = dt
        self.num_interventions = num_interventions
        self.bias = nn.Parameter(torch.zeros(num_interventions, latent_dim))
        self.left = nn.Parameter(torch.empty(num_interventions, latent_dim, rank))
        self.right = nn.Parameter(torch.empty(num_interventions, rank, latent_dim))
        nn.init.normal_(self.bias, std=0.01)
        nn.init.normal_(self.left, std=0.03)
        nn.init.normal_(self.right, std=0.03)

    def forward(self, z: Tensor, intervention: Tensor) -> Tensor:
        if intervention.shape[-1] != self.num_interventions:
            raise ValueError(
                f"expected {self.num_interventions} intervention channels, "
                f"got {intervention.shape[-1]}"
            )
        state = torch.tanh(z)
        state_field = torch.einsum("...d,crd,ckr->...ck", state, self.right, self.left)
        field = state_field + self.bias
        return self.dt * torch.einsum("...c,...cd->...d", intervention, field)


@dataclass(slots=True)
class SequenceBatch:
    """A same-animal sequence batch.

    Arrays have shape ``[batch, time, channels]``. ``onset`` is the first
    post-intervention time index. Ground-truth arrays at and after ``onset``
    are targets only; :meth:`HierarchicalControlledSSM.intervention_loss`
    never passes them to the encoder.
    """

    animal_id: str
    neural: Tensor
    behavior: Tensor
    inputs: Tensor
    intervention: Tensor
    onset: int
    neural_mask: Tensor | None = None
    behavior_mask: Tensor | None = None

    def validate(self) -> None:
        tensors = (self.neural, self.behavior, self.inputs, self.intervention)
        if any(x.ndim != 3 for x in tensors):
            raise ValueError("all sequence tensors must be [batch, time, channels]")
        shape = self.neural.shape[:2]
        if any(x.shape[:2] != shape for x in tensors[1:]):
            raise ValueError("batch and time dimensions must agree")
        if not 1 <= self.onset < shape[1]:
            raise ValueError("onset must leave both a pre-onset state and a query horizon")


@dataclass(slots=True)
class LossBreakdown:
    total: Tensor
    neural: Tensor
    behavior: Tensor
    dynamics: Tensor
    reconstruction: Tensor
    kl: Tensor
    residual_penalty: Tensor

    def detached(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().cpu())
            for name in (
                "total",
                "neural",
                "behavior",
                "dynamics",
                "reconstruction",
                "kl",
                "residual_penalty",
            )
        }


class HierarchicalControlledSSM(nn.Module):
    """Shared controlled dynamics with normal-only animal adaptation."""

    def __init__(
        self,
        *,
        latent_dim: int,
        input_dim: int,
        behavior_dim: int,
        num_interventions: int,
        hidden_dim: int = 96,
        residual_rank: int = 2,
        intervention_rank: int = 4,
        dt: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.behavior_dim = behavior_dim
        self.num_interventions = num_interventions
        self.hidden_dim = hidden_dim
        self.residual_rank = residual_rank
        self.shared = SharedDynamics(latent_dim, input_dim, hidden_dim, dt)
        self.operator = SharedInterventionOperator(
            latent_dim, num_interventions, intervention_rank, dt
        )
        self.behavior_decoder = MLP(latent_dim, behavior_dim, hidden_dim)
        self.adapters = nn.ModuleDict()
        self.donor_intervention_delta = nn.ParameterDict()
        self._intervention_groups: dict[str, str] = {}

    def register_animal(
        self,
        animal_id: str | int,
        neural_dim: int,
        *,
        donor: bool,
        intervention_group: str | int | None = None,
    ) -> AnimalObservationMap:
        key = _module_key(animal_id)
        group_key = _module_key(animal_id if intervention_group is None else intervention_group)
        if key in self.adapters:
            adapter = self.adapters[key]
            if not isinstance(adapter, AnimalObservationMap):
                raise TypeError(f"unexpected adapter type for {animal_id}")
            if adapter.neural_dim != neural_dim:
                raise ValueError(
                    f"animal {animal_id} already has {adapter.neural_dim} neural channels"
                )
            if self._intervention_groups.get(key) != group_key:
                raise ValueError(f"adapter {animal_id} was registered under a different group")
            return adapter
        adapter = AnimalObservationMap(
            neural_dim,
            self.behavior_dim,
            self.latent_dim,
            self.hidden_dim,
            self.residual_rank,
        )
        self.adapters[key] = adapter
        self._intervention_groups[key] = group_key
        if donor and group_key not in self.donor_intervention_delta:
            delta = nn.Parameter(torch.zeros(self.num_interventions, self.latent_dim))
            nn.init.normal_(delta, std=0.005)
            self.donor_intervention_delta[group_key] = delta
        return adapter

    def adapter(self, animal_id: str | int) -> AnimalObservationMap:
        key = _module_key(animal_id)
        if key not in self.adapters:
            raise KeyError(f"animal {animal_id!r} is not registered")
        adapter = self.adapters[key]
        if not isinstance(adapter, AnimalObservationMap):
            raise TypeError(f"unexpected adapter type for {animal_id}")
        return adapter

    def encode(
        self,
        animal_id: str | int,
        neural: Tensor,
        behavior: Tensor,
        *,
        sample: bool = False,
    ) -> tuple[Tensor, Tensor]:
        return self.adapter(animal_id).encode(neural, behavior, sample=sample)

    def decode(self, animal_id: str | int, z: Tensor) -> tuple[Tensor, Tensor]:
        adapter = self.adapter(animal_id)
        neural = adapter.decode_neural(z)
        behavior = adapter.calibrate_behavior(self.behavior_decoder(z))
        return neural, behavior

    def transition(
        self,
        animal_id: str | int,
        z: Tensor,
        inputs: Tensor,
        intervention: Tensor,
        *,
        include_animal_residual: bool,
        include_donor_delta: bool,
    ) -> Tensor:
        next_z = self.shared(z, inputs)
        if include_animal_residual:
            next_z = next_z + self.shared.dt * self.adapter(animal_id).residual(z)
        next_z = next_z + self.operator(z, intervention)
        key = _module_key(animal_id)
        group_key = self._intervention_groups[key]
        if include_donor_delta and group_key in self.donor_intervention_delta:
            delta = self.donor_intervention_delta[group_key]
            next_z = next_z + self.operator.dt * (intervention @ delta)
        return next_z

    def rollout(
        self,
        animal_id: str | int,
        z0: Tensor,
        inputs: Tensor,
        intervention: Tensor,
        *,
        include_animal_residual: bool = True,
        include_donor_delta: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Open-loop rollout using only scheduled inputs after ``z0``."""
        if inputs.shape[:-1] != intervention.shape[:-1]:
            raise ValueError("input and intervention batch/time dimensions differ")
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(f"expected {self.input_dim} scheduled input channels")
        states: list[Tensor] = []
        z = z0
        for step in range(inputs.shape[1]):
            z = self.transition(
                animal_id,
                z,
                inputs[:, step],
                intervention[:, step],
                include_animal_residual=include_animal_residual,
                include_donor_delta=include_donor_delta,
            )
            states.append(z)
        latent = torch.stack(states, dim=1)
        neural, behavior = self.decode(animal_id, latent)
        return latent, neural, behavior

    @staticmethod
    def _masked_mse(prediction: Tensor, target: Tensor, mask: Tensor | None) -> Tensor:
        error = (prediction - target).square()
        if mask is None:
            return error.mean()
        weight = mask.to(dtype=error.dtype)
        return (error * weight).sum() / weight.sum().clamp_min(1.0)

    def normal_loss(
        self,
        batch: SequenceBatch,
        *,
        beta_dynamics: float = 1.0,
        beta_kl: float = 1e-4,
        beta_residual: float = 1e-3,
    ) -> LossBreakdown:
        """Fit shared normal dynamics and an animal adapter on normal trials."""
        batch.validate()
        if torch.count_nonzero(batch.intervention).item() != 0:
            raise ValueError("normal_loss received nonzero intervention inputs")
        z, log_var = self.encode(batch.animal_id, batch.neural, batch.behavior, sample=True)
        neural_hat, behavior_hat = self.decode(batch.animal_id, z)
        neural_loss = self._masked_mse(neural_hat, batch.neural, batch.neural_mask)
        behavior_loss = self._masked_mse(behavior_hat, batch.behavior, batch.behavior_mask)
        reconstruction = neural_loss + behavior_loss

        predicted = self.transition(
            batch.animal_id,
            z[:, :-1],
            batch.inputs[:, :-1],
            batch.intervention[:, :-1],
            include_animal_residual=True,
            include_donor_delta=False,
        )
        dynamics = F.mse_loss(predicted, z[:, 1:].detach())
        kl = 0.5 * (z.square() + log_var.exp() - log_var - 1.0).mean()
        residual_penalty = self.adapter(batch.animal_id).residual.squared_norm()
        total = (
            reconstruction
            + beta_dynamics * dynamics
            + beta_kl * kl
            + beta_residual * residual_penalty
        )
        return LossBreakdown(
            total,
            neural_loss,
            behavior_loss,
            dynamics,
            reconstruction,
            kl,
            residual_penalty,
        )

    def intervention_loss(
        self,
        batch: SequenceBatch,
        *,
        include_donor_delta: bool,
        delta_shrinkage: float = 1e-2,
    ) -> LossBreakdown:
        """Score a post-onset rollout without encoding any post-onset target.

        Only the last observed pre-onset sample is encoded. The first rollout
        control corresponds to ``batch.onset - 1 -> batch.onset``.
        """
        batch.validate()
        pre_index = batch.onset - 1
        z0, _ = self.encode(
            batch.animal_id,
            batch.neural[:, pre_index],
            batch.behavior[:, pre_index],
            sample=False,
        )
        _, neural_hat, behavior_hat = self.rollout(
            batch.animal_id,
            z0,
            batch.inputs[:, pre_index:-1],
            batch.intervention[:, pre_index:-1],
            include_animal_residual=True,
            include_donor_delta=include_donor_delta,
        )
        neural_target = batch.neural[:, batch.onset :]
        behavior_target = batch.behavior[:, batch.onset :]
        if neural_hat.shape != neural_target.shape or behavior_hat.shape != behavior_target.shape:
            raise RuntimeError("rollout and target horizons do not match")
        neural_mask = None if batch.neural_mask is None else batch.neural_mask[:, batch.onset :]
        behavior_mask = (
            None if batch.behavior_mask is None else batch.behavior_mask[:, batch.onset :]
        )
        neural_loss = self._masked_mse(neural_hat, neural_target, neural_mask)
        behavior_loss = self._masked_mse(behavior_hat, behavior_target, behavior_mask)
        total = neural_loss + behavior_loss
        residual_penalty = torch.zeros((), device=total.device)
        key = _module_key(batch.animal_id)
        group_key = self._intervention_groups[key]
        if include_donor_delta and group_key in self.donor_intervention_delta:
            residual_penalty = self.donor_intervention_delta[group_key].square().mean()
            total = total + delta_shrinkage * residual_penalty
        zero = torch.zeros((), device=total.device)
        return LossBreakdown(
            total,
            neural_loss,
            behavior_loss,
            zero,
            neural_loss + behavior_loss,
            zero,
            residual_penalty,
        )

    def configure_stage(
        self,
        stage: Literal["normal", "intervention", "target_adaptation", "evaluation"],
        *,
        target_animal: str | int | None = None,
    ) -> None:
        """Enforce the parameter boundary for each experimental stage."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        if stage == "normal":
            for parameter in self.shared.parameters():
                parameter.requires_grad_(True)
            for parameter in self.behavior_decoder.parameters():
                parameter.requires_grad_(True)
            for adapter in self.adapters.values():
                for parameter in adapter.parameters():
                    parameter.requires_grad_(True)
        elif stage == "intervention":
            for parameter in self.operator.parameters():
                parameter.requires_grad_(True)
            for parameter in self.donor_intervention_delta.parameters():
                parameter.requires_grad_(True)
        elif stage == "target_adaptation":
            if target_animal is None:
                raise ValueError("target_adaptation requires target_animal")
            for parameter in self.adapter(target_animal).parameters():
                parameter.requires_grad_(True)
        elif stage != "evaluation":
            raise ValueError(f"unknown stage {stage!r}")

    def target_intervention_scale(self) -> Tensor:
        """Donor-estimated SD of unobserved target intervention residuals."""
        if not self.donor_intervention_delta:
            return torch.zeros(
                self.num_interventions,
                self.latent_dim,
                device=self.operator.bias.device,
            )
        stacked = torch.stack(list(self.donor_intervention_delta.values()), dim=0)
        return stacked.std(dim=0, unbiased=stacked.shape[0] > 1)

    def project_donor_deltas_zero_mean(
        self,
        group_keys: Sequence[str] | None = None,
    ) -> float:
        """Project a declared donor set onto the exact zero-mean constraint.

        The shared intervention field and the mean donor residual are otherwise
        non-identifiable. Experiment-specific fitting loops call this after
        every intervention-stage optimizer step, using only the donor groups
        that belong to that fit (never a held-out validation group). The
        returned norm is useful for fail-closed audit assertions.
        """

        keys = tuple(self.donor_intervention_delta) if group_keys is None else tuple(group_keys)
        if not keys:
            return 0.0
        if len(set(keys)) != len(keys):
            raise ValueError("donor-delta projection groups must be unique")
        missing = set(keys) - set(self.donor_intervention_delta)
        if missing:
            raise KeyError(f"unknown donor-delta groups: {sorted(missing)}")
        with torch.no_grad():
            selected = [self.donor_intervention_delta[key] for key in keys]
            mean = torch.stack(selected, dim=0).mean(dim=0)
            for delta in selected:
                delta.sub_(mean)
            residual_mean = torch.stack(selected, dim=0).mean(dim=0)
            return float(residual_mean.square().sum().sqrt().detach().cpu())
