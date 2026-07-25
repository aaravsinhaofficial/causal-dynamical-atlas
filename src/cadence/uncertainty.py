"""Uncertainty decomposition and simultaneous conformal trajectory bands."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor

from cadence.model import HierarchicalControlledSSM

FloatArray = NDArray[np.floating]


def finite_sample_quantile(values: ArrayLike, coverage: float) -> float:
    """Split-conformal quantile with the finite-sample rank correction."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("no finite calibration scores")
    if not 0 < coverage < 1:
        raise ValueError("coverage must lie strictly between zero and one")
    rank = min(array.size, int(np.ceil((array.size + 1) * coverage)))
    return float(np.partition(array, rank - 1)[rank - 1])


@dataclass(frozen=True, slots=True)
class SimultaneousConformalizer:
    """Maximum-standardized-residual conformalizer for whole trajectories."""

    quantile: float
    channel_scale: FloatArray
    coverage: float

    @classmethod
    def fit(
        cls,
        predictions: ArrayLike,
        targets: ArrayLike,
        channel_scale: ArrayLike,
        *,
        coverage: float = 0.9,
    ) -> SimultaneousConformalizer:
        prediction = np.asarray(predictions, dtype=np.float64)
        target = np.asarray(targets, dtype=np.float64)
        scale = np.asarray(channel_scale, dtype=np.float64)
        if prediction.shape != target.shape or prediction.ndim < 3:
            raise ValueError("expected matching [case, time, ..., channel] arrays")
        if scale.shape != (prediction.shape[-1],):
            raise ValueError("channel scale does not match trajectories")
        standardized = np.abs(prediction - target) / scale
        scores = np.nanmax(standardized.reshape(standardized.shape[0], -1), axis=1)
        return cls(finite_sample_quantile(scores, coverage), scale, coverage)

    def interval(self, prediction: ArrayLike) -> tuple[FloatArray, FloatArray]:
        center = np.asarray(prediction, dtype=np.float64)
        radius = self.quantile * self.channel_scale
        return center - radius, center + radius


@dataclass(slots=True)
class PosteriorTrajectories:
    latent: Tensor
    neural: Tensor
    behavior: Tensor
    sources: dict[str, str]

    def numpy(self) -> dict[str, FloatArray]:
        return {
            "latent": self.latent.detach().cpu().numpy(),
            "neural": self.neural.detach().cpu().numpy(),
            "behavior": self.behavior.detach().cpu().numpy(),
        }


@torch.no_grad()
def sample_target_intervention_residual(
    model: HierarchicalControlledSSM,
    animal_id: str,
    z0: Tensor,
    inputs: Tensor,
    intervention: Tensor,
    *,
    draws: int,
    seed: int,
    process_scale: float = 0.0,
) -> PosteriorTrajectories:
    """Integrate over donor-estimated target intervention heterogeneity.

    The held-out animal's intervention residual is never fitted. Each draw
    samples it from the donor distribution and optionally adds latent process
    noise estimated from normal validation data.
    """
    if draws < 1:
        raise ValueError("draws must be positive")
    generator = torch.Generator(device=z0.device).manual_seed(seed)
    residual_sd = model.target_intervention_scale().to(z0)
    latent_draws = []
    neural_draws = []
    behavior_draws = []
    for _ in range(draws):
        delta = (
            torch.randn(
                residual_sd.shape,
                generator=generator,
                device=z0.device,
                dtype=z0.dtype,
            )
            * residual_sd
        )
        states = []
        z = z0
        for step in range(inputs.shape[1]):
            z = model.transition(
                animal_id,
                z,
                inputs[:, step],
                intervention[:, step],
                include_animal_residual=True,
                include_donor_delta=False,
            )
            z = z + model.operator.dt * (intervention[:, step] @ delta)
            if process_scale > 0:
                noise = torch.randn(
                    z.shape,
                    generator=generator,
                    device=z.device,
                    dtype=z.dtype,
                )
                z = z + process_scale * noise
            states.append(z)
        latent = torch.stack(states, dim=1)
        neural, behavior = model.decode(animal_id, latent)
        latent_draws.append(latent)
        neural_draws.append(neural)
        behavior_draws.append(behavior)
    return PosteriorTrajectories(
        torch.stack(latent_draws),
        torch.stack(neural_draws),
        torch.stack(behavior_draws),
        {
            "adapter": "point estimate; bootstrap externally",
            "operator": "donor random-effect distribution",
            "process": f"Gaussian latent scale={process_scale}",
        },
    )


def combine_ensemble_draws(
    members: list[PosteriorTrajectories],
) -> PosteriorTrajectories:
    if not members:
        raise ValueError("at least one ensemble member is required")
    return PosteriorTrajectories(
        torch.cat([member.latent for member in members], dim=0),
        torch.cat([member.neural for member in members], dim=0),
        torch.cat([member.behavior for member in members], dim=0),
        {
            "operator": "donor-bootstrap ensemble plus donor random effects",
            "adapter": "normal-support bootstrap across ensemble members",
            "process": "member-specific normal-validation estimate",
        },
    )
