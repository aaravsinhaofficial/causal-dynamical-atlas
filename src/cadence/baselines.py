"""Locked comparison models for cross-animal trajectory prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn

from cadence.model import HierarchicalControlledSSM

FloatArray = NDArray[np.floating]


class LinearHierarchicalSSM(HierarchicalControlledSSM):
    """Capacity-controlled linear shared dynamics and intervention transfer."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.linear_normal = nn.Linear(self.latent_dim + self.input_dim, self.latent_dim)
        self.linear_intervention = nn.Linear(
            self.num_interventions,
            self.latent_dim,
            bias=False,
        )

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
        drift = self.linear_normal(torch.cat((z, inputs), dim=-1))
        next_z = z + self.shared.dt * (-z + drift)
        next_z = next_z + self.shared.dt * self.linear_intervention(intervention)
        if include_animal_residual:
            next_z = next_z + self.shared.dt * self.adapter(animal_id).residual(z)
        key = f"animal_{str(animal_id).replace('.', '_').replace('/', '_')}"
        group_key = self._intervention_groups[key]
        if include_donor_delta and group_key in self.donor_intervention_delta:
            next_z = next_z + self.shared.dt * (
                intervention @ self.donor_intervention_delta[group_key]
            )
        return next_z

    def configure_stage(self, stage: str, *, target_animal: str | int | None = None) -> None:
        super().configure_stage(stage, target_animal=target_animal)  # type: ignore[arg-type]
        if stage == "normal":
            for parameter in self.shared.parameters():
                parameter.requires_grad_(False)
            for parameter in self.linear_normal.parameters():
                parameter.requires_grad_(True)
        elif stage == "intervention":
            for parameter in self.operator.parameters():
                parameter.requires_grad_(False)
            for parameter in self.linear_intervention.parameters():
                parameter.requires_grad_(True)


class AdditiveInterventionSSM(HierarchicalControlledSSM):
    """A state-independent intervention-vector baseline."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.additive_intervention = nn.Linear(
            self.num_interventions,
            self.latent_dim,
            bias=False,
        )

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
        next_z = next_z + self.shared.dt * self.additive_intervention(intervention)
        key = f"animal_{str(animal_id).replace('.', '_').replace('/', '_')}"
        group_key = self._intervention_groups[key]
        if include_donor_delta and group_key in self.donor_intervention_delta:
            next_z = next_z + self.shared.dt * (
                intervention @ self.donor_intervention_delta[group_key]
            )
        return next_z

    def configure_stage(self, stage: str, *, target_animal: str | int | None = None) -> None:
        super().configure_stage(stage, target_animal=target_animal)  # type: ignore[arg-type]
        if stage == "intervention":
            for parameter in self.operator.parameters():
                parameter.requires_grad_(False)
            for parameter in self.additive_intervention.parameters():
                parameter.requires_grad_(True)


class BlackBoxMetaGRU(HierarchicalControlledSSM):
    """Capacity-matched recurrent comparator without an operator decomposition."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.gru = nn.GRUCell(
            self.input_dim + self.num_interventions,
            self.latent_dim,
        )
        self.post_gru = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )

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
        controls = torch.cat((inputs, intervention), dim=-1)
        # ``GRUCell`` accepts only one- or two-dimensional inputs, whereas the
        # experiment losses also evaluate nested trial/time batches.  Collapse
        # every leading axis for the recurrent update and restore the original
        # latent shape afterwards.
        flat_controls = controls.reshape(-1, controls.shape[-1])
        flat_z = z.reshape(-1, z.shape[-1])
        recurrent = self.gru(flat_controls, flat_z).reshape_as(z)
        next_z = recurrent + self.shared.dt * self.post_gru(z)
        if include_animal_residual:
            next_z = next_z + self.shared.dt * self.adapter(animal_id).residual(z)
        return next_z

    def configure_stage(self, stage: str, *, target_animal: str | int | None = None) -> None:
        super().configure_stage(stage, target_animal=target_animal)  # type: ignore[arg-type]
        if stage in {"normal", "intervention"}:
            for parameter in self.shared.parameters():
                parameter.requires_grad_(False)
            for parameter in self.operator.parameters():
                parameter.requires_grad_(False)
            for parameter in self.gru.parameters():
                parameter.requires_grad_(True)
            for parameter in self.post_gru.parameters():
                parameter.requires_grad_(True)


@dataclass(slots=True)
class ConditionTimeTemplate:
    """Donor-average effect indexed by a discretized intervention descriptor."""

    templates: dict[tuple[float, ...], FloatArray]
    global_template: FloatArray

    @classmethod
    def fit(
        cls,
        effects: ArrayLike,
        descriptors: ArrayLike,
        animal_ids: ArrayLike,
        *,
        decimals: int = 4,
    ) -> ConditionTimeTemplate:
        effect_array = np.asarray(effects, dtype=np.float64)
        descriptor_array = np.asarray(descriptors, dtype=np.float64)
        animals = np.asarray(animal_ids)
        if effect_array.shape[0] != descriptor_array.shape[0] or animals.shape[0] != len(
            effect_array
        ):
            raise ValueError("effects, descriptors, and animal IDs must align")
        keys = [tuple(row) for row in np.round(descriptor_array, decimals)]
        templates: dict[tuple[float, ...], FloatArray] = {}
        for key in sorted(set(keys)):
            # Average trials within animal first, then animals equally.
            per_animal = []
            for animal in np.unique(animals):
                indices = [
                    index
                    for index, (trial_key, trial_animal) in enumerate(
                        zip(keys, animals, strict=True)
                    )
                    if trial_key == key and trial_animal == animal
                ]
                if indices:
                    per_animal.append(np.nanmean(effect_array[indices], axis=0))
            templates[key] = np.nanmean(per_animal, axis=0)
        global_per_animal = [
            np.nanmean(effect_array[animals == animal], axis=0) for animal in np.unique(animals)
        ]
        return cls(templates, np.nanmean(global_per_animal, axis=0))

    def predict(self, descriptors: ArrayLike, *, decimals: int = 4) -> FloatArray:
        descriptor_array = np.asarray(descriptors, dtype=np.float64)
        output = [
            self.templates.get(tuple(row), self.global_template)
            for row in np.round(descriptor_array, decimals)
        ]
        return np.stack(output)


def zero_effect(control_rollout: ArrayLike) -> FloatArray:
    """The eligible no-causal-effect baseline returns the normal rollout."""
    return np.asarray(control_rollout, dtype=np.float64).copy()
