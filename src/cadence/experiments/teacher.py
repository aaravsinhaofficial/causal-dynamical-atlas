"""End-to-end teacher-RNN benchmark.

This module is intentionally separate from the procedural teacher and model
implementations.  It is the executable experiment layer which enforces the
scientific protocol:

1. fit shared normal dynamics and donor adapters using normal trials only;
2. freeze normal dynamics/adapters and fit the intervention operator on
   training-donor perturbations, selecting on held-out validation donors;
3. register each target and fit its adapter on ``normal_fit`` while selecting
   on ``normal_val``;
4. freeze the complete model before opening target counterfactual pairs, then
   roll out both treated and control schedules from the last pre-onset sample.

Teacher latent states are never model inputs.  They are used only after frozen
prediction for diagnostics.  A normal-fit-only affine gauge maps learned
coordinates to teacher coordinates for latent, vector-field, and operator
recovery scores.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import shutil
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
from torch.nn import functional as F

from cadence.baselines import (
    AdditiveInterventionSSM,
    BlackBoxMetaGRU,
    LinearHierarchicalSSM,
    zero_effect,
)
from cadence.metrics import (
    causal_skill,
    support_scale,
    time_resolved_r2,
    trajectory_nrmse,
)
from cadence.model import HierarchicalControlledSSM, SequenceBatch
from cadence.protocol import (
    FreezeAttestation,
    ProtocolViolation,
    attest_preoutcome_freeze,
)
from cadence.teacher import (
    AnimalDataset,
    CohortConfig,
    CounterfactualPair,
    InterventionConfig,
    ObservationConfig,
    TeacherConfig,
    TeacherDataset,
    TeacherGroundTruth,
    TrialConfig,
    generate_teacher_world,
    teacher_config_sha256,
    validate_locked_teacher_config,
)
from cadence.training import (
    EpochRecord,
    FitConfig,
    FitResult,
    move_batch,
    seed_everything,
)

FloatArray = npt.NDArray[np.float64]
MethodName = Literal["proposed", "linear", "additive", "black_box"]

LEARNED_METHODS: tuple[MethodName, ...] = (
    "proposed",
    "linear",
    "additive",
    "black_box",
)
DEFAULT_REPORT_METHODS = (
    *LEARNED_METHODS,
    "zero_effect",
    "proposed_native_decoder",
    "proposed_no_target_residual",
    "proposed_no_target_adaptation",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _safe_key(value: str) -> str:
    return value.replace("-", "_").replace("/", "_").replace(".", "_")


def _locked_teacher_output_identity(
    output: Path,
    truth: TeacherGroundTruth,
) -> str:
    """Require the single canonical output tree for one locked public world."""

    repository = Path(__file__).resolve().parents[3]
    relative = Path("results") / "teacher-locked" / "full" / f"locked-seed-{truth.seed_index:02d}"
    expected = (repository / relative).absolute()
    if output.absolute() != expected:
        raise ProtocolViolation(
            "locked teacher output must use the canonical one-shot path "
            f"{expected}; observed {output.absolute()}"
        )
    cursor = repository
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ProtocolViolation("locked teacher output path may not traverse a symlink")
    return relative.as_posix()


def _write_text_artifact(
    path: Path,
    text: str,
    *,
    exclusive: bool,
) -> None:
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as stream:
        stream.write(text)


@dataclass(frozen=True, slots=True)
class TeacherExperimentConfig:
    """Optimization and reporting settings for one teacher world."""

    profile: Literal["smoke", "full"] = "smoke"
    hidden_dim: int = 24
    intervention_rank: int = 2
    batch_size: int = 8
    validation_fraction: float = 0.25
    neural_transform: Literal["identity", "log1p"] = "log1p"
    normal_fit: FitConfig = field(
        default_factory=lambda: FitConfig(
            learning_rate=3e-3,
            weight_decay=1e-4,
            max_epochs=150,
            patience=12,
            gradient_clip=5.0,
            seed=0,
            device="cpu",
            mixed_precision=False,
        )
    )
    intervention_fit: FitConfig = field(
        default_factory=lambda: FitConfig(
            learning_rate=4e-3,
            weight_decay=1e-4,
            max_epochs=150,
            patience=12,
            gradient_clip=5.0,
            seed=1,
            device="cpu",
            mixed_precision=False,
        )
    )
    target_fit: FitConfig = field(
        default_factory=lambda: FitConfig(
            learning_rate=3e-3,
            weight_decay=1e-4,
            max_epochs=150,
            patience=12,
            gradient_clip=5.0,
            seed=2,
            device="cpu",
            mixed_precision=False,
        )
    )
    learned_methods: tuple[MethodName, ...] = LEARNED_METHODS
    include_ablations: bool = True
    max_vector_field_states: int = 256
    readout_learning_rate: float = 1e-2
    readout_max_epochs: int = 300
    readout_patience: int = 30
    readout_weight_decay: float = 2e-1
    readout_ridge_grid: tuple[float, ...] = (
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
        100.0,
    )
    linear_behavior_decoder: bool = False
    linear_observation_family: bool = False
    calibrate_quasilikelihood_readout: bool = True

    def validate(self) -> None:
        if self.profile not in {"smoke", "full"}:
            raise ValueError("profile must be smoke or full")
        if self.hidden_dim < 4 or self.intervention_rank < 1:
            raise ValueError("model dimensions must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.neural_transform not in {"identity", "log1p"}:
            raise ValueError("unknown neural transform")
        if not self.learned_methods:
            raise ValueError("at least one learned method is required")
        unknown = set(self.learned_methods) - set(LEARNED_METHODS)
        if unknown:
            raise ValueError(f"unknown learned methods: {sorted(unknown)}")
        if self.max_vector_field_states < 1:
            raise ValueError("max_vector_field_states must be positive")
        if (
            self.readout_learning_rate <= 0
            or self.readout_max_epochs < 1
            or self.readout_patience < 1
            or self.readout_weight_decay < 0
        ):
            raise ValueError("invalid neural readout optimization settings")
        if not self.readout_ridge_grid or any(value < 0 for value in self.readout_ridge_grid):
            raise ValueError("readout_ridge_grid must contain nonnegative values")

    def to_mapping(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def make_experiment_config(
    profile: Literal["smoke", "full"],
    *,
    seed: int = 0,
    device: str | None = None,
    learned_methods: Sequence[MethodName] | None = None,
) -> TeacherExperimentConfig:
    """Return a documented smoke or paper-scale optimization configuration."""

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = selected_device.startswith("cuda")
    if profile == "smoke":
        config = TeacherExperimentConfig(
            profile="smoke",
            hidden_dim=24,
            intervention_rank=2,
            batch_size=8,
            normal_fit=FitConfig(
                learning_rate=3e-3,
                max_epochs=150,
                patience=12,
                seed=seed * 100 + 11,
                device=selected_device,
                mixed_precision=use_amp,
            ),
            intervention_fit=FitConfig(
                learning_rate=4e-3,
                max_epochs=150,
                patience=12,
                seed=seed * 100 + 23,
                device=selected_device,
                mixed_precision=use_amp,
            ),
            target_fit=FitConfig(
                learning_rate=3e-3,
                max_epochs=150,
                patience=12,
                seed=seed * 100 + 37,
                device=selected_device,
                mixed_precision=use_amp,
            ),
        )
    else:
        config = TeacherExperimentConfig(
            profile="full",
            hidden_dim=96,
            intervention_rank=2,
            batch_size=16,
            normal_fit=FitConfig(
                learning_rate=1e-3,
                max_epochs=500,
                patience=40,
                seed=seed * 100 + 11,
                device=selected_device,
                mixed_precision=use_amp,
            ),
            intervention_fit=FitConfig(
                learning_rate=1e-3,
                max_epochs=500,
                patience=40,
                seed=seed * 100 + 23,
                device=selected_device,
                mixed_precision=use_amp,
            ),
            target_fit=FitConfig(
                learning_rate=1e-3,
                max_epochs=400,
                patience=40,
                seed=seed * 100 + 37,
                device=selected_device,
                mixed_precision=use_amp,
            ),
            max_vector_field_states=1024,
            readout_max_epochs=600,
            readout_patience=50,
        )
    if learned_methods is not None:
        config = replace(config, learned_methods=tuple(learned_methods))
    config.validate()
    return config


def _scientific_experiment_mapping(
    config: TeacherExperimentConfig,
) -> dict[str, Any]:
    """Remove only machine-specific execution fields from a config mapping."""

    mapping = asdict(config)
    for stage in ("normal_fit", "intervention_fit", "target_fit"):
        mapping[stage].pop("device")
        mapping[stage].pop("mixed_precision")
    return _jsonable(mapping)


def teacher_experiment_scientific_sha256(
    config: TeacherExperimentConfig,
) -> str:
    """Fingerprint every frozen hyperparameter except device execution mode."""

    payload = json.dumps(
        _scientific_experiment_mapping(config),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_locked_teacher_experiment_config(
    config: TeacherExperimentConfig,
) -> str:
    """Require the exact full, complete-method preregistered configuration."""

    expected = make_experiment_config(
        "full",
        seed=0,
        device="cpu",
        learned_methods=LEARNED_METHODS,
    )
    expected_mapping = _scientific_experiment_mapping(expected)
    observed_mapping = _scientific_experiment_mapping(config)
    expected_sha = teacher_experiment_scientific_sha256(expected)
    observed_sha = teacher_experiment_scientific_sha256(config)
    if observed_sha != expected_sha or observed_mapping != expected_mapping:
        raise ProtocolViolation(
            "post-freeze procedural teacher evaluation requires the exact full frozen "
            "TeacherExperimentConfig with complete methods and ablations; "
            f"expected sha256={expected_sha}, observed={observed_sha}"
        )
    return observed_sha


def make_profile_teacher_config(
    base: TeacherConfig,
    profile: Literal["smoke", "full"],
) -> TeacherConfig:
    """Shrink only the procedural workload for the smoke profile."""

    if profile == "full":
        return base
    return replace(
        base,
        release_name=f"{base.release_name}-smoke",
        cohort=CohortConfig(train_donors=2, validation_donors=1, targets=1),
        dynamics=replace(
            base.dynamics,
            latent_dim=3,
            task_input_dim=2,
            dt=0.10,
            process_noise_std=0.015,
        ),
        intervention=InterventionConfig(
            n_interventions=2,
            onset_step=7,
            offset_step=13,
            doses=(1.0,),
            shared_bias_norm=0.60,
            shared_state_norm=0.20,
            animal_residual_ratio=0.05,
        ),
        observations=ObservationConfig(
            neurons_min=10,
            neurons_max=18,
            behavior_dim=3,
            neural_bias_mean=base.observations.neural_bias_mean,
            neural_bias_std=base.observations.neural_bias_std,
            neural_map_scale=base.observations.neural_map_scale,
            neural_noise_model=base.observations.neural_noise_model,
            nb_dispersion=base.observations.nb_dispersion,
            behavior_noise_std=base.observations.behavior_noise_std,
            behavior_residual_ratio=base.observations.behavior_residual_ratio,
        ),
        trials=TrialConfig(
            steps=22,
            donor_normal_trials=12,
            donor_pairs_per_intervention=6,
            target_normal_fit_trials=10,
            target_normal_val_trials=4,
            target_normal_audit_trials=4,
            target_pairs_per_intervention=6,
        ),
    )


def _transform_neural(values: npt.ArrayLike, transform: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if transform == "log1p":
        return np.log1p(array)
    return array


def _inverse_neural(values: npt.ArrayLike, transform: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if transform == "log1p":
        # The decoder is unconstrained. Clipping avoids meaningless overflow
        # while retaining a valid nonnegative count prediction.
        return np.maximum(np.expm1(np.clip(array, 0.0, 12.0)), 0.0)
    return array


def _trajectory_batch(
    animal: AnimalDataset,
    trajectories: Sequence[Any],
    *,
    onset: int,
    neural_transform: str,
) -> SequenceBatch:
    if not trajectories:
        raise ValueError("cannot construct a batch from no trajectories")
    neural = np.stack(
        [_transform_neural(item.neural_counts, neural_transform) for item in trajectories]
    )
    behavior = np.stack([np.asarray(item.behavior) for item in trajectories])
    inputs = np.stack([np.asarray(item.task_input) for item in trajectories])
    intervention = np.stack([np.asarray(item.intervention) for item in trajectories])
    batch = SequenceBatch(
        animal_id=animal.animal_id,
        neural=torch.as_tensor(neural, dtype=torch.float32),
        behavior=torch.as_tensor(behavior, dtype=torch.float32),
        inputs=torch.as_tensor(inputs, dtype=torch.float32),
        intervention=torch.as_tensor(intervention, dtype=torch.float32),
        onset=onset,
    )
    batch.validate()
    return batch


def _chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def normal_sequence_batches(
    animal: AnimalDataset,
    *,
    batch_size: int,
    neural_transform: str,
    splits: set[str] | None = None,
    trials: Sequence[Any] | None = None,
) -> list[SequenceBatch]:
    """Construct same-animal batches without padding variable neuron counts."""

    selected = list(animal.normal_trials if trials is None else trials)
    if splits is not None:
        selected = [trial for trial in selected if trial.split in splits]
    trajectories = [trial.trajectory for trial in selected]
    return [
        _trajectory_batch(
            animal,
            chunk,
            onset=max(1, min(2, chunk[0].task_input.shape[0] - 1)),
            neural_transform=neural_transform,
        )
        for chunk in _chunks(trajectories, batch_size)
    ]


def intervention_sequence_batches(
    animal: AnimalDataset,
    pairs: Sequence[CounterfactualPair],
    *,
    batch_size: int,
    neural_transform: str,
    arm: Literal["treated", "control"] = "treated",
) -> list[SequenceBatch]:
    """Construct donor intervention batches from an explicit arm."""

    trajectories = [getattr(pair, arm) for pair in pairs]
    return [
        _trajectory_batch(
            animal,
            chunk,
            onset=pairs[0].onset_step,
            neural_transform=neural_transform,
        )
        for chunk in _chunks(trajectories, batch_size)
    ]


def _split_normal_trials(
    animal: AnimalDataset, validation_fraction: float
) -> tuple[list[Any], list[Any]]:
    trials = list(animal.normal_trials)
    validation_count = max(1, int(round(len(trials) * validation_fraction)))
    validation_count = min(validation_count, len(trials) - 1)
    return trials[:-validation_count], trials[-validation_count:]


def _split_pairs_stratified(
    pairs: Sequence[CounterfactualPair], validation_fraction: float
) -> tuple[list[CounterfactualPair], list[CounterfactualPair]]:
    groups: dict[tuple[int, float], list[CounterfactualPair]] = {}
    for pair in pairs:
        groups.setdefault((pair.intervention_index, pair.dose), []).append(pair)
    train: list[CounterfactualPair] = []
    validation: list[CounterfactualPair] = []
    for key in sorted(groups):
        group = groups[key]
        validation_count = max(1, int(round(len(group) * validation_fraction)))
        validation_count = min(validation_count, len(group) - 1)
        train.extend(group[:-validation_count])
        validation.extend(group[-validation_count:])
    return train, validation


def _make_model(
    method: MethodName,
    truth: TeacherGroundTruth,
    config: TeacherExperimentConfig,
    *,
    initialization_seed: int,
) -> HierarchicalControlledSSM:
    seed_everything(initialization_seed)
    kwargs: dict[str, Any] = {
        "latent_dim": truth.config.dynamics.latent_dim,
        "input_dim": truth.config.dynamics.task_input_dim,
        "behavior_dim": truth.config.observations.behavior_dim,
        "num_interventions": truth.config.intervention.n_interventions,
        "hidden_dim": config.hidden_dim,
        "residual_rank": truth.config.dynamics.residual_rank,
        "intervention_rank": config.intervention_rank,
        "dt": truth.config.dynamics.dt,
    }
    classes = {
        "proposed": HierarchicalControlledSSM,
        "linear": LinearHierarchicalSSM,
        "additive": AdditiveInterventionSSM,
        "black_box": BlackBoxMetaGRU,
    }
    model = classes[method](**kwargs)
    if config.linear_behavior_decoder:
        # The teacher observation family is affine before its count link. A
        # linear shared behavior map removes needless nonlinear gauge freedom.
        model.behavior_decoder = torch.nn.Linear(model.latent_dim, model.behavior_dim)
    return model


def _configure_animal_observation_family(
    model: HierarchicalControlledSSM,
    animal: AnimalDataset,
    config: TeacherExperimentConfig,
    *,
    intervention_donor: bool | None = None,
) -> None:
    model.register_animal(
        animal.animal_id,
        animal.neuron_count,
        donor=(animal.role == "train_donor" if intervention_donor is None else intervention_donor),
    )
    if not config.linear_observation_family:
        return
    adapter = model.adapter(animal.animal_id)
    adapter.encoder = torch.nn.Linear(
        animal.neuron_count + model.behavior_dim,
        2 * model.latent_dim,
    )
    adapter.neural_decoder = torch.nn.Linear(model.latent_dim, animal.neuron_count)


def _fit_result_summary(result: FitResult) -> dict[str, Any]:
    return {
        "stage": result.stage,
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "epochs_run": len(result.history),
        "final_train_loss": result.history[-1].train_loss,
        "final_validation_loss": result.history[-1].validation_loss,
    }


def _donor_delta_mean_norm(model: HierarchicalControlledSSM) -> float:
    if not model.donor_intervention_delta:
        return 0.0
    with torch.no_grad():
        deltas = torch.stack(list(model.donor_intervention_delta.values()), dim=0)
        return float(deltas.mean(dim=0).square().sum().sqrt().cpu())


@dataclass(slots=True)
class GaugeMap:
    """Normal-support affine alignment; invertibility is measured, not assumed."""

    linear: FloatArray
    offset: FloatArray
    rank: int
    condition_number: float

    def transform(self, values: npt.ArrayLike) -> FloatArray:
        array = np.asarray(values, dtype=np.float64)
        return array @ self.linear + self.offset

    def transform_vectors(self, vectors: npt.ArrayLike) -> FloatArray:
        return np.asarray(vectors, dtype=np.float64) @ self.linear


def fit_affine_gauge(
    learned: npt.ArrayLike,
    teacher: npt.ArrayLike,
    *,
    ridge: float = 1e-2,
) -> GaugeMap:
    """Fit a ridge-affine diagnostic using normal support only."""

    x = np.asarray(learned, dtype=np.float64).reshape(-1, np.shape(learned)[-1])
    y = np.asarray(teacher, dtype=np.float64).reshape(-1, np.shape(teacher)[-1])
    if x.shape[0] != y.shape[0] or x.shape[0] < x.shape[1] + 1:
        raise ValueError("gauge arrays must align and contain enough support")
    design = np.column_stack((x, np.ones(x.shape[0])))
    scale = float(np.trace(design[:, :-1].T @ design[:, :-1]) / x.shape[1])
    penalty = ridge * max(scale, np.finfo(np.float64).eps) * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    linear = coefficients[:-1]
    rank = int(np.linalg.matrix_rank(linear))
    condition = float(np.linalg.cond(linear)) if rank == min(linear.shape) else float("inf")
    return GaugeMap(
        linear=linear,
        offset=coefficients[-1],
        rank=rank,
        condition_number=condition,
    )


def _encode_normal(
    model: HierarchicalControlledSSM,
    animal: AnimalDataset,
    trials: Sequence[Any],
    neural_transform: str,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    neural = _transform_neural(
        np.stack([trial.trajectory.neural_counts for trial in trials]), neural_transform
    )
    behavior = np.stack([trial.trajectory.behavior for trial in trials])
    inputs = np.stack([trial.trajectory.task_input for trial in trials])
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        learned, _ = model.encode(
            animal.animal_id,
            torch.as_tensor(neural, dtype=torch.float32, device=device),
            torch.as_tensor(behavior, dtype=torch.float32, device=device),
            sample=False,
        )
    teacher = np.stack([trial.trajectory.latent for trial in trials])
    return (
        learned.detach().cpu().numpy().astype(np.float64),
        teacher.astype(np.float64),
        inputs.astype(np.float64),
    )


@dataclass(slots=True)
class TargetPrediction:
    """Frozen paired target prediction in observed and latent coordinates."""

    neural_treated: FloatArray
    neural_control: FloatArray
    behavior_treated: FloatArray
    behavior_control: FloatArray
    latent_treated: FloatArray
    latent_control: FloatArray


@dataclass(slots=True)
class PendingTeacherScore:
    """A score request held until the complete prediction bundle is hashed."""

    method: str
    animal: AnimalDataset
    model: HierarchicalControlledSSM
    prediction: TargetPrediction
    truth: TeacherGroundTruth
    gauge: GaugeMap
    learned_normal: FloatArray
    teacher_normal: FloatArray
    normal_inputs: FloatArray
    max_vector_field_states: int
    include_operator: bool = True
    extra_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SoftplusNeuralReadout:
    """Teacher-consistent animal map fitted exclusively on normal rollouts."""

    weight: FloatArray
    bias: FloatArray
    latent_mean: FloatArray
    latent_scale: FloatArray
    best_epoch: int
    validation_loss: float
    ridge: float
    design_rank: int
    design_condition_number: float
    rollout_anchor: int
    support_min: FloatArray
    support_max: FloatArray

    def predict(self, latent: npt.ArrayLike) -> FloatArray:
        standardized = (np.asarray(latent, dtype=np.float64) - self.latent_mean) / self.latent_scale
        linear = standardized @ self.weight + self.bias
        return np.maximum(linear, 0.0) + np.log1p(np.exp(-np.abs(linear)))

    def summary(self) -> dict[str, float | int]:
        return {
            "best_epoch": self.best_epoch,
            "validation_poisson_nll": self.validation_loss,
            "selected_ridge": self.ridge,
            "normal_rollout_design_rank": self.design_rank,
            "normal_rollout_design_condition_number": self.design_condition_number,
            "normal_rollout_anchor": self.rollout_anchor,
            "normal_rollout_support_max_abs_standardized": float(
                np.max(np.abs(np.concatenate((self.support_min, self.support_max))))
            ),
        }

    def extrapolation_summary(self, latent: npt.ArrayLike) -> dict[str, float]:
        standardized = (np.asarray(latent, dtype=np.float64) - self.latent_mean) / self.latent_scale
        outside = (standardized < self.support_min) | (standardized > self.support_max)
        return {
            "query_max_abs_standardized": float(np.max(np.abs(standardized))),
            "query_coordinate_fraction_outside_normal_rollout_range": float(np.mean(outside)),
        }


def _normal_rollout_readout_data(
    model: HierarchicalControlledSSM,
    animal: AnimalDataset,
    trials: Sequence[Any],
    *,
    neural_transform: str,
    anchor: int,
) -> tuple[FloatArray, FloatArray]:
    """Return frozen normal-rollout states and their future count targets.

    It is tempting to encode every target-normal observation and regress that
    same observation on its encoding.  That is not a valid decoder-selection
    criterion: the encoder has already seen the count noise being predicted,
    so even held-out trials reward an autoencoding shortcut that is absent in
    an open-loop intervention forecast.  Instead, each trajectory is encoded
    once at the same relative pre-query anchor used for target prediction.
    The frozen dynamics then generates every state paired with a future count.
    """

    if not trials:
        raise ValueError("normal rollout readout requires at least one trial")
    steps = trials[0].trajectory.neural_counts.shape[0]
    if not 0 <= anchor < steps - 1:
        raise ValueError("normal rollout anchor must leave a future horizon")
    neural = _transform_neural(
        np.stack([trial.trajectory.neural_counts[anchor] for trial in trials]),
        neural_transform,
    )
    behavior = np.stack([trial.trajectory.behavior[anchor] for trial in trials])
    inputs = np.stack([trial.trajectory.task_input[anchor:-1] for trial in trials])
    intervention = np.zeros(
        (
            len(trials),
            steps - anchor - 1,
            model.num_interventions,
        ),
        dtype=np.float64,
    )
    device = next(model.parameters()).device
    model.configure_stage("evaluation")
    model.eval()
    with torch.no_grad():
        z0, _ = model.encode(
            animal.animal_id,
            torch.as_tensor(neural, dtype=torch.float32, device=device),
            torch.as_tensor(behavior, dtype=torch.float32, device=device),
            sample=False,
        )
        latent, _, _ = model.rollout(
            animal.animal_id,
            z0,
            torch.as_tensor(inputs, dtype=torch.float32, device=device),
            torch.as_tensor(intervention, dtype=torch.float32, device=device),
            include_animal_residual=True,
            include_donor_delta=False,
        )
    target = np.stack([trial.trajectory.neural_counts[anchor + 1 :] for trial in trials])
    return (
        latent.detach().cpu().numpy().astype(np.float64),
        target.astype(np.float64),
    )


def fit_normal_only_neural_readout(
    model: HierarchicalControlledSSM,
    animal: AnimalDataset,
    experiment: TeacherExperimentConfig,
    *,
    seed: int,
) -> SoftplusNeuralReadout:
    """Estimate target H_i with a softplus-Poisson observation family.

    The teacher emits NB or Poisson counts with a softplus conditional mean.
    Poisson quasi-likelihood consistently estimates that mean under either
    observation noise model and uses no intervention trial.  Decoder fitting
    and selection use frozen open-loop normal rollouts, so the encoded count at
    a time point is never also the response for that point.
    """

    fit_trials = [trial for trial in animal.normal_trials if trial.split == "normal_fit"]
    validation_trials = [trial for trial in animal.normal_trials if trial.split == "normal_val"]
    if not validation_trials:
        raise ValueError("normal-only neural readout requires normal_val trials")
    if not animal.counterfactual_pairs:
        raise ValueError("normal-only neural readout requires a query anchor")
    anchor = animal.counterfactual_pairs[0].onset_step - 1
    learned_fit, fit_counts = _normal_rollout_readout_data(
        model,
        animal,
        fit_trials,
        neural_transform=experiment.neural_transform,
        anchor=anchor,
    )
    learned_validation, validation_counts = _normal_rollout_readout_data(
        model,
        animal,
        validation_trials,
        neural_transform=experiment.neural_transform,
        anchor=anchor,
    )
    x_train = learned_fit.reshape(-1, learned_fit.shape[-1])
    x_validation = learned_validation.reshape(-1, learned_validation.shape[-1])
    latent_mean = x_train.mean(axis=0)
    latent_scale = x_train.std(axis=0, ddof=1)
    latent_scale = np.maximum(latent_scale, 0.05)
    x_train = (x_train - latent_mean) / latent_scale
    x_validation = (x_validation - latent_mean) / latent_scale
    y_train = fit_counts.reshape(-1, animal.neuron_count)
    y_validation = validation_counts.reshape(-1, animal.neuron_count)
    design_rank = int(np.linalg.matrix_rank(x_train))
    design_condition = (
        float(np.linalg.cond(x_train)) if design_rank == x_train.shape[1] else float("inf")
    )

    seed_everything(seed)
    device = next(model.parameters()).device
    train_x = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    train_y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    validation_x = torch.as_tensor(x_validation, dtype=torch.float32, device=device)
    validation_y = torch.as_tensor(y_validation, dtype=torch.float32, device=device)
    empirical_mean = train_y.mean(dim=0).clamp_min(1e-3)
    # inverse softplus, stable for both small and large empirical rates
    initial_bias = empirical_mean + torch.log(-torch.expm1(-empirical_mean))
    best_loss = float("inf")
    best_epoch = -1
    best_ridge = float("nan")
    best_weight: torch.Tensor | None = None
    best_bias: torch.Tensor | None = None

    def poisson_nll(
        x: torch.Tensor,
        y: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        rate = F.softplus(x @ weight + bias).clamp_min(1e-6)
        return (rate - y * torch.log(rate)).mean()

    for ridge_index, ridge in enumerate(experiment.readout_ridge_grid):
        seed_everything(seed + ridge_index)
        weight = torch.nn.Parameter(
            torch.zeros(model.latent_dim, animal.neuron_count, device=device)
        )
        bias = torch.nn.Parameter(initial_bias.clone())
        optimizer = torch.optim.Adam(
            (weight, bias),
            lr=experiment.readout_learning_rate,
            weight_decay=experiment.readout_weight_decay,
        )
        stale = 0
        ridge_best = float("inf")

        for epoch in range(experiment.readout_max_epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = poisson_nll(train_x, train_y, weight, bias) + (ridge * weight.square().mean())
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                validation_loss = float(
                    poisson_nll(validation_x, validation_y, weight, bias).detach().cpu()
                )
            if np.isfinite(validation_loss) and validation_loss < ridge_best - 1e-8:
                ridge_best = validation_loss
                stale = 0
                if validation_loss < best_loss - 1e-8:
                    best_loss = validation_loss
                    best_epoch = epoch
                    best_ridge = ridge
                    best_weight = weight.detach().cpu().clone()
                    best_bias = bias.detach().cpu().clone()
            else:
                stale += 1
                if stale >= experiment.readout_patience:
                    break
    if best_weight is None or best_bias is None:
        raise RuntimeError("normal-only neural readout produced no checkpoint")
    return SoftplusNeuralReadout(
        weight=best_weight.numpy().astype(np.float64),
        bias=best_bias.numpy().astype(np.float64),
        latent_mean=latent_mean.astype(np.float64),
        latent_scale=latent_scale.astype(np.float64),
        best_epoch=best_epoch,
        validation_loss=best_loss,
        ridge=best_ridge,
        design_rank=design_rank,
        design_condition_number=design_condition,
        rollout_anchor=anchor,
        support_min=x_train.min(axis=0).astype(np.float64),
        support_max=x_train.max(axis=0).astype(np.float64),
    )


@dataclass(slots=True)
class PairedSequenceBatch:
    """Aligned treated/control batches with shared exogenous variables."""

    treated: SequenceBatch
    control: SequenceBatch

    def validate(self) -> None:
        self.treated.validate()
        self.control.validate()
        if self.treated.animal_id != self.control.animal_id:
            raise ValueError("paired arms must belong to the same animal")
        if self.treated.onset != self.control.onset:
            raise ValueError("paired arms have different onsets")
        if self.treated.neural.shape != self.control.neural.shape:
            raise ValueError("paired neural arrays differ in shape")


def paired_intervention_sequence_batches(
    animal: AnimalDataset,
    pairs: Sequence[CounterfactualPair],
    *,
    batch_size: int,
    neural_transform: str,
) -> list[PairedSequenceBatch]:
    """Build causally coupled batches for the donor effect objective."""

    output: list[PairedSequenceBatch] = []
    for chunk in _chunks(pairs, batch_size):
        treated = _trajectory_batch(
            animal,
            [pair.treated for pair in chunk],
            onset=chunk[0].onset_step,
            neural_transform=neural_transform,
        )
        control = _trajectory_batch(
            animal,
            [pair.control for pair in chunk],
            onset=chunk[0].onset_step,
            neural_transform=neural_transform,
        )
        batch = PairedSequenceBatch(treated=treated, control=control)
        batch.validate()
        output.append(batch)
    return output


def _paired_effect_loss(
    model: HierarchicalControlledSSM,
    batch: PairedSequenceBatch,
    *,
    absolute_weight: float = 0.1,
    delta_shrinkage: float = 1e-2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit G to counterfactual differences while retaining trajectory anchoring."""

    treated = batch.treated
    control = batch.control
    pre_index = treated.onset - 1
    z0, _ = model.encode(
        treated.animal_id,
        control.neural[:, pre_index],
        control.behavior[:, pre_index],
        sample=False,
    )
    treated_hat = model.rollout(
        treated.animal_id,
        z0,
        treated.inputs[:, pre_index:-1],
        treated.intervention[:, pre_index:-1],
        include_animal_residual=True,
        include_donor_delta=True,
    )
    control_hat = model.rollout(
        control.animal_id,
        z0,
        control.inputs[:, pre_index:-1],
        control.intervention[:, pre_index:-1],
        include_animal_residual=True,
        include_donor_delta=True,
    )
    treated_neural = treated.neural[:, treated.onset :]
    control_neural = control.neural[:, control.onset :]
    treated_behavior = treated.behavior[:, treated.onset :]
    control_behavior = control.behavior[:, control.onset :]
    neural_effect_loss = F.mse_loss(
        treated_hat[1] - control_hat[1], treated_neural - control_neural
    )
    behavior_effect_loss = F.mse_loss(
        treated_hat[2] - control_hat[2], treated_behavior - control_behavior
    )
    absolute = F.mse_loss(treated_hat[1], treated_neural) + F.mse_loss(
        treated_hat[2], treated_behavior
    )
    penalty = torch.zeros((), device=z0.device)
    centering = torch.zeros((), device=z0.device)
    if model.donor_intervention_delta:
        deltas = torch.stack(list(model.donor_intervention_delta.values()), dim=0)
        penalty = deltas.square().mean()
        centering = deltas.mean(dim=0).square().mean()
    total = (
        neural_effect_loss
        + behavior_effect_loss
        + absolute_weight * absolute
        + delta_shrinkage * penalty
        + delta_shrinkage * centering
    )
    return total, neural_effect_loss, behavior_effect_loss


def fit_paired_intervention_stage(
    model: HierarchicalControlledSSM,
    train_batches: Sequence[PairedSequenceBatch],
    validation_batches: Sequence[PairedSequenceBatch],
    *,
    config: FitConfig,
    fixed_epochs: int | None = None,
) -> FitResult:
    """Stage-locked G fitting on paired donor counterfactual effects.

    ``fixed_epochs`` is used only for the post-selection all-donor refit.  In
    that mode the final epoch is retained unconditionally; the supplied
    validation batches are monitored but cannot select or stop the fit.
    """

    if not train_batches or not validation_batches:
        raise ValueError("paired train and validation batches must be nonempty")
    if fixed_epochs is not None and fixed_epochs < 1:
        raise ValueError("fixed intervention epochs must be positive")
    seed_everything(config.seed)
    model.configure_stage("intervention")
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    projection_groups = tuple(model.donor_intervention_delta)
    initial_projection_residual = model.project_donor_deltas_zero_mean(projection_groups)
    if initial_projection_residual > 1e-7:
        raise ProtocolViolation("initial donor-delta zero-mean projection failed")
    train = [
        PairedSequenceBatch(move_batch(batch.treated, device), move_batch(batch.control, device))
        for batch in train_batches
    ]
    validation = [
        PairedSequenceBatch(move_batch(batch.treated, device), move_batch(batch.control, device))
        for batch in validation_batches
    ]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("intervention stage has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    use_amp = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    generator = random.Random(config.seed)
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[EpochRecord] = []

    epochs = config.max_epochs if fixed_epochs is None else fixed_epochs
    for epoch in range(epochs):
        model.train()
        order = list(range(len(train)))
        generator.shuffle(order)
        train_values: list[float] = []
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                total, _, _ = _paired_effect_loss(model, train[index])
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            projection_residual = model.project_donor_deltas_zero_mean(projection_groups)
            if projection_residual > 1e-7:
                raise ProtocolViolation(
                    "donor-delta zero-mean projection failed after optimizer step"
                )
            train_values.append(float(total.detach().cpu()))

        model.eval()
        validation_values: list[float] = []
        neural_values: list[float] = []
        behavior_values: list[float] = []
        with torch.no_grad():
            for batch in validation:
                total, neural, behavior = _paired_effect_loss(model, batch)
                validation_values.append(float(total.detach().cpu()))
                neural_values.append(float(neural.detach().cpu()))
                behavior_values.append(float(behavior.detach().cpu()))
        validation_loss = float(np.mean(validation_values))
        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=float(np.mean(train_values)),
                validation_loss=validation_loss,
                neural_loss=float(np.mean(neural_values)),
                behavior_loss=float(np.mean(behavior_values)),
            )
        )
        if fixed_epochs is not None:
            if not np.isfinite(validation_loss):
                raise RuntimeError("fixed intervention refit produced nonfinite loss")
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        elif np.isfinite(validation_loss) and validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("paired intervention optimization produced no checkpoint")
    model.load_state_dict(best_state)
    final_projection_residual = model.project_donor_deltas_zero_mean(projection_groups)
    if final_projection_residual > 1e-7:
        raise ProtocolViolation("final donor-delta zero-mean projection failed")
    return FitResult(
        stage="intervention",
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        history=history,
        config=config,
    )


def _deterministic_normal_loss(
    model: HierarchicalControlledSSM,
    batch: SequenceBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normal-only objective with deterministic latents and short open-loop fit."""

    if torch.count_nonzero(batch.intervention).item() != 0:
        raise ValueError("normal objective received an intervention")
    z, _ = model.encode(batch.animal_id, batch.neural, batch.behavior, sample=False)
    neural_hat, behavior_hat = model.decode(batch.animal_id, z)
    neural_loss = F.mse_loss(neural_hat, batch.neural)
    behavior_loss = F.mse_loss(behavior_hat, batch.behavior)
    predicted_next = model.transition(
        batch.animal_id,
        z[:, :-1],
        batch.inputs[:, :-1],
        batch.intervention[:, :-1],
        include_animal_residual=True,
        include_donor_delta=False,
    )
    dynamics_loss = F.mse_loss(predicted_next, z[:, 1:])

    horizon = min(10, batch.neural.shape[1] - 1)
    _, open_neural, open_behavior = model.rollout(
        batch.animal_id,
        z[:, 0],
        batch.inputs[:, :horizon],
        batch.intervention[:, :horizon],
        include_animal_residual=True,
        include_donor_delta=False,
    )
    open_loop_loss = F.mse_loss(open_neural, batch.neural[:, 1 : horizon + 1]) + F.mse_loss(
        open_behavior, batch.behavior[:, 1 : horizon + 1]
    )

    flattened = z.reshape(-1, z.shape[-1])
    centered = flattened - flattened.mean(dim=0, keepdim=True)
    variance = centered.square().mean(dim=0)
    # A soft variance floor prevents the dynamically trivial posterior-collapse
    # solution without fixing a privileged latent orientation.
    variance_floor = F.relu(0.15 - variance).square().mean()
    residual_penalty = model.adapter(batch.animal_id).residual.squared_norm()
    total = (
        neural_loss
        + behavior_loss
        + 2.0 * dynamics_loss
        + 0.25 * open_loop_loss
        + 0.05 * variance_floor
        + 1e-3 * residual_penalty
    )
    return total, neural_loss, behavior_loss


def fit_deterministic_normal_stage(
    model: HierarchicalControlledSSM,
    train_batches: Sequence[SequenceBatch],
    validation_batches: Sequence[SequenceBatch],
    *,
    stage: Literal["normal", "target_adaptation"],
    config: FitConfig,
    target_animal: str | None = None,
    fixed_epochs: int | None = None,
) -> FitResult:
    """Fit F/adapters without stochastic posterior noise or intervention data.

    ``fixed_epochs`` is reserved for the all-donor post-selection refit.  The
    final epoch is retained in that mode, so no refit observation is reused to
    make an early-stopping decision.
    """

    if not train_batches or not validation_batches:
        raise ValueError("normal train and validation batches must be nonempty")
    if fixed_epochs is not None and fixed_epochs < 1:
        raise ValueError("fixed normal epochs must be positive")
    if stage == "target_adaptation":
        if target_animal is None:
            raise ValueError("target adaptation requires an animal")
        used = {batch.animal_id for batch in (*train_batches, *validation_batches)}
        if used != {target_animal}:
            raise ValueError("target adaptation may only use its target animal")
    seed_everything(config.seed)
    model.configure_stage(stage, target_animal=target_animal)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    train = [move_batch(batch, device) for batch in train_batches]
    validation = [move_batch(batch, device) for batch in validation_batches]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("normal stage has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    use_amp = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    generator = random.Random(config.seed)
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[EpochRecord] = []

    epochs = config.max_epochs if fixed_epochs is None else fixed_epochs
    for epoch in range(epochs):
        model.train()
        order = list(range(len(train)))
        generator.shuffle(order)
        train_values: list[float] = []
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                total, _, _ = _deterministic_normal_loss(model, train[index])
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            train_values.append(float(total.detach().cpu()))
        model.eval()
        validation_values: list[float] = []
        neural_values: list[float] = []
        behavior_values: list[float] = []
        with torch.no_grad():
            for batch in validation:
                total, neural, behavior = _deterministic_normal_loss(model, batch)
                validation_values.append(float(total.detach().cpu()))
                neural_values.append(float(neural.detach().cpu()))
                behavior_values.append(float(behavior.detach().cpu()))
        validation_loss = float(np.mean(validation_values))
        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=float(np.mean(train_values)),
                validation_loss=validation_loss,
                neural_loss=float(np.mean(neural_values)),
                behavior_loss=float(np.mean(behavior_values)),
            )
        )
        if fixed_epochs is not None:
            if not np.isfinite(validation_loss):
                raise RuntimeError("fixed normal refit produced nonfinite loss")
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        elif np.isfinite(validation_loss) and validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("normal optimization produced no checkpoint")
    model.load_state_dict(best_state)
    return FitResult(
        stage=stage,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        history=history,
        config=config,
    )


def predict_target_pairs(
    model: HierarchicalControlledSSM,
    animal: AnimalDataset,
    *,
    neural_transform: str,
    include_animal_residual: bool = True,
    neural_readout: SoftplusNeuralReadout | None = None,
) -> TargetPrediction:
    """Predict target pairs open loop without mounting post-onset outcomes."""

    pairs = animal.counterfactual_pairs
    if not pairs:
        raise ValueError("target has no counterfactual query pairs")
    onset = pairs[0].onset_step
    pre_index = onset - 1
    # Every twin is identical through onset. Only this pre-onset observation is
    # passed to the encoder; scheduled task/action arrays are exogenous inputs.
    pre_neural = _transform_neural(
        np.stack([pair.control.neural_counts[pre_index] for pair in pairs]),
        neural_transform,
    )
    pre_behavior = np.stack([pair.control.behavior[pre_index] for pair in pairs])
    inputs = np.stack([pair.control.task_input[pre_index:-1] for pair in pairs])
    treated_action = np.stack([pair.treated.intervention[pre_index:-1] for pair in pairs])
    control_action = np.stack([pair.control.intervention[pre_index:-1] for pair in pairs])
    device = next(model.parameters()).device
    model.configure_stage("evaluation")
    model.eval()
    with torch.no_grad():
        z0, _ = model.encode(
            animal.animal_id,
            torch.as_tensor(pre_neural, dtype=torch.float32, device=device),
            torch.as_tensor(pre_behavior, dtype=torch.float32, device=device),
            sample=False,
        )
        scheduled_inputs = torch.as_tensor(inputs, dtype=torch.float32, device=device)
        treated = model.rollout(
            animal.animal_id,
            z0,
            scheduled_inputs,
            torch.as_tensor(treated_action, dtype=torch.float32, device=device),
            include_animal_residual=include_animal_residual,
            include_donor_delta=False,
        )
        control = model.rollout(
            animal.animal_id,
            z0,
            scheduled_inputs,
            torch.as_tensor(control_action, dtype=torch.float32, device=device),
            include_animal_residual=include_animal_residual,
            include_donor_delta=False,
        )
    latent_treated = treated[0].cpu().numpy().astype(np.float64)
    latent_control = control[0].cpu().numpy().astype(np.float64)
    if neural_readout is None:
        neural_treated = _inverse_neural(treated[1].cpu().numpy(), neural_transform)
        neural_control = _inverse_neural(control[1].cpu().numpy(), neural_transform)
    else:
        neural_treated = neural_readout.predict(latent_treated)
        neural_control = neural_readout.predict(latent_control)
    return TargetPrediction(
        neural_treated=neural_treated,
        neural_control=neural_control,
        behavior_treated=treated[2].cpu().numpy().astype(np.float64),
        behavior_control=control[2].cpu().numpy().astype(np.float64),
        latent_treated=latent_treated,
        latent_control=latent_control,
    )


def _variance_explained(prediction: npt.ArrayLike, target: npt.ArrayLike) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    obs = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(obs)
    centered = obs[valid] - np.mean(obs[valid])
    denominator = float(np.dot(centered, centered))
    if denominator <= np.finfo(np.float64).eps:
        return float("nan")
    return float(1.0 - np.square(pred[valid] - obs[valid]).sum() / denominator)


def _cosine_similarity(prediction: npt.ArrayLike, target: npt.ArrayLike) -> float:
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1, np.shape(prediction)[-1])
    obs = np.asarray(target, dtype=np.float64).reshape(-1, np.shape(target)[-1])
    denominator = np.linalg.norm(pred, axis=1) * np.linalg.norm(obs, axis=1)
    valid = denominator > np.finfo(np.float64).eps
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.sum(pred[valid] * obs[valid], axis=1) / denominator[valid]))


def _linear_cka(left: npt.ArrayLike, right: npt.ArrayLike) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1, np.shape(left)[-1])
    y = np.asarray(right, dtype=np.float64).reshape(-1, np.shape(right)[-1])
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denominator = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(cross / denominator) if denominator > 0 else float("nan")


def _true_shared_fields(truth: TeacherGroundTruth, states: FloatArray) -> FloatArray:
    fields = []
    for intervention_index in range(truth.config.intervention.n_interventions):
        state_part = np.tanh(states) @ truth.intervention_state[intervention_index].T
        fields.append(
            truth.stress.eta * (truth.intervention_bias[intervention_index][None, :] + state_part)
        )
    return np.stack(fields, axis=1)


def _learned_fields(
    model: HierarchicalControlledSSM,
    animal_id: str,
    states: FloatArray,
    inputs: FloatArray,
) -> FloatArray:
    """Evaluate a model-agnostic local causal field by action differencing."""

    device = next(model.parameters()).device
    z = torch.as_tensor(states, dtype=torch.float32, device=device)
    u = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    zero = torch.zeros(len(states), model.num_interventions, dtype=torch.float32, device=device)
    output: list[FloatArray] = []
    model.eval()
    with torch.no_grad():
        normal = model.transition(
            animal_id,
            z,
            u,
            zero,
            include_animal_residual=False,
            include_donor_delta=False,
        )
        for intervention_index in range(model.num_interventions):
            action = zero.clone()
            action[:, intervention_index] = 1.0
            acted = model.transition(
                animal_id,
                z,
                u,
                action,
                include_animal_residual=False,
                include_donor_delta=False,
            )
            output.append(
                ((acted - normal) / model.shared.dt).detach().cpu().numpy().astype(np.float64)
            )
    return np.stack(output, axis=1)


def _operator_diagnostics(
    model: HierarchicalControlledSSM,
    animal: AnimalDataset,
    truth: TeacherGroundTruth,
    learned_normal: FloatArray,
    teacher_normal: FloatArray,
    normal_inputs: FloatArray,
    gauge: GaugeMap,
    *,
    max_states: int,
) -> dict[str, float]:
    learned_states = learned_normal.reshape(-1, learned_normal.shape[-1])
    teacher_states = teacher_normal.reshape(-1, teacher_normal.shape[-1])
    inputs = normal_inputs.reshape(-1, normal_inputs.shape[-1])
    if len(learned_states) > max_states:
        indices = np.linspace(0, len(learned_states) - 1, max_states, dtype=int)
        learned_states = learned_states[indices]
        teacher_states = teacher_states[indices]
        inputs = inputs[indices]
    predicted = gauge.transform_vectors(
        _learned_fields(model, animal.animal_id, learned_states, inputs)
    )
    observed = _true_shared_fields(truth, teacher_states)
    return {
        "shared_vector_field_r2_affine_gauge": _variance_explained(predicted, observed),
        "shared_vector_field_cosine_affine_gauge": _cosine_similarity(predicted, observed),
        "shared_operator_linear_cka_affine_gauge": _linear_cka(predicted, observed),
    }


def _target_truth(animal: AnimalDataset) -> dict[str, FloatArray]:
    onset = animal.counterfactual_pairs[0].onset_step
    return {
        "neural_treated": np.stack(
            [pair.treated.neural_counts[onset:] for pair in animal.counterfactual_pairs]
        ).astype(np.float64),
        "neural_control": np.stack(
            [pair.control.neural_counts[onset:] for pair in animal.counterfactual_pairs]
        ).astype(np.float64),
        "neural_mean_treated": np.stack(
            [pair.treated.neural_mean[onset:] for pair in animal.counterfactual_pairs]
        ).astype(np.float64),
        "neural_mean_control": np.stack(
            [pair.control.neural_mean[onset:] for pair in animal.counterfactual_pairs]
        ).astype(np.float64),
        "behavior_treated": np.stack(
            [pair.treated.behavior[onset:] for pair in animal.counterfactual_pairs]
        ).astype(np.float64),
        "behavior_control": np.stack(
            [pair.control.behavior[onset:] for pair in animal.counterfactual_pairs]
        ).astype(np.float64),
        "behavior_mean_treated": np.stack(
            [pair.treated.behavior_mean[onset:] for pair in animal.counterfactual_pairs]
        ).astype(np.float64),
        "behavior_mean_control": np.stack(
            [pair.control.behavior_mean[onset:] for pair in animal.counterfactual_pairs]
        ).astype(np.float64),
        "latent_treated": np.stack(
            [pair.treated.latent[onset:] for pair in animal.counterfactual_pairs]
        ).astype(np.float64),
        "latent_control": np.stack(
            [pair.control.latent[onset:] for pair in animal.counterfactual_pairs]
        ).astype(np.float64),
    }


def _condition_average(values: npt.ArrayLike, animal: AnimalDataset) -> FloatArray:
    """Average repetitions within each intervention/dose condition."""

    array = np.asarray(values, dtype=np.float64)
    if array.shape[0] != len(animal.counterfactual_pairs):
        raise ValueError("condition averaging requires one row per target pair")
    groups: dict[tuple[int, float], list[int]] = {}
    for index, pair in enumerate(animal.counterfactual_pairs):
        groups.setdefault((pair.intervention_index, pair.dose), []).append(index)
    return np.stack([np.mean(array[groups[key]], axis=0) for key in sorted(groups)])


def _score_prediction_with_model(
    model: HierarchicalControlledSSM,
    prediction: TargetPrediction,
    animal: AnimalDataset,
    truth: TeacherGroundTruth,
    gauge: GaugeMap,
    *,
    learned_normal: FloatArray,
    teacher_normal: FloatArray,
    normal_inputs: FloatArray,
    max_vector_field_states: int,
    include_operator: bool = True,
) -> dict[str, Any]:
    # Keep the public scorer's numerical code centralized while passing the
    # frozen model only to the operator diagnostic.
    observed = _target_truth(animal)
    normal_trials = [trial for trial in animal.normal_trials if trial.split == "normal_fit"]
    normal_neural = np.stack([trial.trajectory.neural_counts for trial in normal_trials]).astype(
        np.float64
    )
    normal_behavior = np.stack([trial.trajectory.behavior for trial in normal_trials]).astype(
        np.float64
    )
    normal_neural_mean = np.stack([trial.trajectory.neural_mean for trial in normal_trials]).astype(
        np.float64
    )
    normal_behavior_mean = np.stack(
        [trial.trajectory.behavior_mean for trial in normal_trials]
    ).astype(np.float64)
    neural_scale = support_scale(normal_neural)
    behavior_scale = support_scale(normal_behavior)
    neural_mean_scale = support_scale(normal_neural_mean)
    behavior_mean_scale = support_scale(normal_behavior_mean)
    latent_scale = support_scale(teacher_normal)

    pred_neural_effect = prediction.neural_treated - prediction.neural_control
    obs_neural_effect = observed["neural_treated"] - observed["neural_control"]
    pred_behavior_effect = prediction.behavior_treated - prediction.behavior_control
    obs_behavior_effect = observed["behavior_treated"] - observed["behavior_control"]
    aligned_treated = gauge.transform(prediction.latent_treated)
    aligned_control = gauge.transform(prediction.latent_control)
    pred_latent_effect = aligned_treated - aligned_control
    obs_latent_effect = observed["latent_treated"] - observed["latent_control"]
    target_map = truth.neural_maps[animal.animal_index]
    target_bias = truth.neural_biases[animal.animal_index]
    oracle_treated_logits = aligned_treated @ target_map.T + target_bias
    oracle_control_logits = aligned_control @ target_map.T + target_bias
    oracle_treated_neural = np.maximum(oracle_treated_logits, 0.0) + np.log1p(
        np.exp(-np.abs(oracle_treated_logits))
    )
    oracle_control_neural = np.maximum(oracle_control_logits, 0.0) + np.log1p(
        np.exp(-np.abs(oracle_control_logits))
    )
    gauge_true_h_effect = oracle_treated_neural - oracle_control_neural
    true_neural_mean_effect = observed["neural_mean_treated"] - observed["neural_mean_control"]
    true_behavior_mean_effect = (
        observed["behavior_mean_treated"] - observed["behavior_mean_control"]
    )
    condition_pred_neural = _condition_average(pred_neural_effect, animal)
    condition_obs_neural = _condition_average(obs_neural_effect, animal)
    condition_pred_behavior = _condition_average(pred_behavior_effect, animal)
    condition_obs_behavior = _condition_average(obs_behavior_effect, animal)
    condition_true_neural_mean = _condition_average(true_neural_mean_effect, animal)
    condition_true_behavior_mean = _condition_average(true_behavior_mean_effect, animal)

    metrics: dict[str, Any] = {
        "affine_alignment_rank": gauge.rank,
        "affine_alignment_condition_number": (
            gauge.condition_number if np.isfinite(gauge.condition_number) else None
        ),
        "affine_alignment_full_rank": gauge.rank == gauge.linear.shape[0],
        # Primary ATE-style observed-space endpoints average repetitions within
        # intervention/dose before scoring the complete time series.
        "neural_condition_averaged_causal_skill": causal_skill(
            condition_pred_neural,
            condition_obs_neural,
            channel_scale=neural_scale,
        ),
        "behavior_condition_averaged_causal_skill": causal_skill(
            condition_pred_behavior,
            condition_obs_behavior,
            channel_scale=behavior_scale,
        ),
        # Synthetic-only recoverability endpoints compare against the exact
        # conditional observation mean, before realized count/behavior noise.
        "neural_pathwise_mean_causal_skill": causal_skill(
            pred_neural_effect,
            true_neural_mean_effect,
            channel_scale=neural_mean_scale,
        ),
        "behavior_pathwise_mean_causal_skill": causal_skill(
            pred_behavior_effect,
            true_behavior_mean_effect,
            channel_scale=behavior_mean_scale,
        ),
        # Evaluation-only localization oracle: normal-fit affine gauge followed
        # by the true target observation map. It is never an eligible method.
        "gauge_true_h_neural_pathwise_mean_causal_skill": causal_skill(
            gauge_true_h_effect,
            true_neural_mean_effect,
            channel_scale=neural_mean_scale,
        ),
        "gauge_true_h_neural_condition_averaged_causal_skill": causal_skill(
            _condition_average(gauge_true_h_effect, animal),
            condition_obs_neural,
            channel_scale=neural_scale,
        ),
        "neural_causal_skill": causal_skill(
            pred_neural_effect, obs_neural_effect, channel_scale=neural_scale
        ),
        "behavior_causal_skill": causal_skill(
            pred_behavior_effect, obs_behavior_effect, channel_scale=behavior_scale
        ),
        "neural_effect_nrmse": trajectory_nrmse(
            pred_neural_effect, obs_neural_effect, channel_scale=neural_scale
        ),
        "behavior_effect_nrmse": trajectory_nrmse(
            pred_behavior_effect, obs_behavior_effect, channel_scale=behavior_scale
        ),
        "neural_treated_nrmse": trajectory_nrmse(
            prediction.neural_treated,
            observed["neural_treated"],
            channel_scale=neural_scale,
        ),
        "behavior_treated_nrmse": trajectory_nrmse(
            prediction.behavior_treated,
            observed["behavior_treated"],
            channel_scale=behavior_scale,
        ),
        "neural_effect_time_r2": time_resolved_r2(pred_neural_effect, obs_neural_effect),
        "behavior_effect_time_r2": time_resolved_r2(pred_behavior_effect, obs_behavior_effect),
        "latent_effect_skill_affine_gauge": causal_skill(
            pred_latent_effect, obs_latent_effect, channel_scale=latent_scale
        ),
        "latent_treated_r2_affine_gauge": _variance_explained(
            aligned_treated, observed["latent_treated"]
        ),
        "neural_observation_oracle_causal_skill": causal_skill(
            true_neural_mean_effect,
            obs_neural_effect,
            channel_scale=neural_scale,
        ),
        "behavior_observation_oracle_causal_skill": causal_skill(
            true_behavior_mean_effect,
            obs_behavior_effect,
            channel_scale=behavior_scale,
        ),
        "neural_condition_averaged_oracle_causal_skill": causal_skill(
            condition_true_neural_mean,
            condition_obs_neural,
            channel_scale=neural_scale,
        ),
        "behavior_condition_averaged_oracle_causal_skill": causal_skill(
            condition_true_behavior_mean,
            condition_obs_behavior,
            channel_scale=behavior_scale,
        ),
    }
    if include_operator:
        metrics.update(
            _operator_diagnostics(
                model,
                animal,
                truth,
                learned_normal,
                teacher_normal,
                normal_inputs,
                gauge,
                max_states=max_vector_field_states,
            )
        )
    return _jsonable(metrics)


def _prediction_arrays(
    prefix: str,
    prediction: TargetPrediction,
    gauge: GaugeMap,
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}__neural_treated": prediction.neural_treated.astype(np.float32),
        f"{prefix}__neural_control": prediction.neural_control.astype(np.float32),
        f"{prefix}__behavior_treated": prediction.behavior_treated.astype(np.float32),
        f"{prefix}__behavior_control": prediction.behavior_control.astype(np.float32),
        f"{prefix}__latent_treated_model": prediction.latent_treated.astype(np.float32),
        f"{prefix}__latent_control_model": prediction.latent_control.astype(np.float32),
        f"{prefix}__latent_treated_aligned": gauge.transform(prediction.latent_treated).astype(
            np.float32
        ),
        f"{prefix}__latent_control_aligned": gauge.transform(prediction.latent_control).astype(
            np.float32
        ),
    }


def _zero_prediction(reference: TargetPrediction) -> TargetPrediction:
    neural = zero_effect(reference.neural_control)
    behavior = zero_effect(reference.behavior_control)
    latent = zero_effect(reference.latent_control)
    return TargetPrediction(
        neural_treated=neural,
        neural_control=neural.copy(),
        behavior_treated=behavior,
        behavior_control=behavior.copy(),
        latent_treated=latent,
        latent_control=latent.copy(),
    )


def _aggregate_metrics(
    by_method: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    aggregate: dict[str, dict[str, Any]] = {}
    scalar_names = (
        "neural_condition_averaged_causal_skill",
        "behavior_condition_averaged_causal_skill",
        "neural_pathwise_mean_causal_skill",
        "behavior_pathwise_mean_causal_skill",
        "gauge_true_h_neural_pathwise_mean_causal_skill",
        "gauge_true_h_neural_condition_averaged_causal_skill",
        "neural_causal_skill",
        "behavior_causal_skill",
        "neural_effect_nrmse",
        "behavior_effect_nrmse",
        "neural_treated_nrmse",
        "behavior_treated_nrmse",
        "latent_effect_skill_affine_gauge",
        "latent_treated_r2_affine_gauge",
        "shared_vector_field_r2_affine_gauge",
        "shared_vector_field_cosine_affine_gauge",
        "shared_operator_linear_cka_affine_gauge",
        "neural_observation_oracle_causal_skill",
        "behavior_observation_oracle_causal_skill",
        "neural_condition_averaged_oracle_causal_skill",
        "behavior_condition_averaged_oracle_causal_skill",
    )
    for method, animals in by_method.items():
        summary: dict[str, Any] = {"n_targets": len(animals)}
        for name in scalar_names:
            values = []
            for metrics in animals.values():
                value = metrics.get(name)
                if isinstance(value, int | float) and np.isfinite(value):
                    values.append(float(value))
            summary[f"{name}_mean"] = float(np.mean(values)) if values else float("nan")
            summary[f"{name}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            )
        aggregate[method] = summary
    return aggregate


def _normal_data_for_donors(
    dataset: TeacherDataset,
    config: TeacherExperimentConfig,
) -> tuple[list[SequenceBatch], list[SequenceBatch]]:
    """Selection-stage normals from training donors only."""

    train_batches: list[SequenceBatch] = []
    validation_batches: list[SequenceBatch] = []
    donors = [animal for animal in dataset.animals if animal.role == "train_donor"]
    for animal in donors:
        fit_trials, val_trials = _split_normal_trials(animal, config.validation_fraction)
        train_batches.extend(
            normal_sequence_batches(
                animal,
                batch_size=config.batch_size,
                neural_transform=config.neural_transform,
                trials=fit_trials,
            )
        )
        validation_batches.extend(
            normal_sequence_batches(
                animal,
                batch_size=config.batch_size,
                neural_transform=config.neural_transform,
                trials=val_trials,
            )
        )
    return train_batches, validation_batches


def _intervention_data_for_donors(
    dataset: TeacherDataset,
    config: TeacherExperimentConfig,
) -> tuple[list[PairedSequenceBatch], list[PairedSequenceBatch]]:
    training_animals = [animal for animal in dataset.animals if animal.role == "train_donor"]
    validation_animals = [animal for animal in dataset.animals if animal.role == "validation_donor"]
    train_batches: list[PairedSequenceBatch] = []
    validation_batches: list[PairedSequenceBatch] = []
    for animal in training_animals:
        train_pairs, heldout_pairs = _split_pairs_stratified(
            animal.counterfactual_pairs, config.validation_fraction
        )
        train_batches.extend(
            paired_intervention_sequence_batches(
                animal,
                train_pairs,
                batch_size=config.batch_size,
                neural_transform=config.neural_transform,
            )
        )
        # When no dedicated validation donors exist, use within-donor pairs.
        if not validation_animals:
            validation_batches.extend(
                paired_intervention_sequence_batches(
                    animal,
                    heldout_pairs,
                    batch_size=config.batch_size,
                    neural_transform=config.neural_transform,
                )
            )
    for animal in validation_animals:
        validation_batches.extend(
            paired_intervention_sequence_batches(
                animal,
                animal.counterfactual_pairs,
                batch_size=config.batch_size,
                neural_transform=config.neural_transform,
            )
        )
    return train_batches, validation_batches


def _register_donors(
    model: HierarchicalControlledSSM,
    dataset: TeacherDataset,
    config: TeacherExperimentConfig,
    *,
    include_validation_deltas: bool = False,
) -> None:
    for animal in dataset.animals:
        if animal.role == "target":
            continue
        # During selection, validation donors test shared G with no fitted
        # intervention delta. After epoch selection, the fresh all-donor refit
        # may treat them as donors without changing the selected epoch count.
        _configure_animal_observation_family(
            model,
            animal,
            config,
            intervention_donor=(animal.role == "train_donor" or include_validation_deltas),
        )


def _all_donor_normal_batches(
    dataset: TeacherDataset,
    config: TeacherExperimentConfig,
) -> list[SequenceBatch]:
    batches: list[SequenceBatch] = []
    for animal in dataset.animals:
        if animal.role == "target":
            continue
        batches.extend(
            normal_sequence_batches(
                animal,
                batch_size=config.batch_size,
                neural_transform=config.neural_transform,
            )
        )
    return batches


def _all_donor_intervention_batches(
    dataset: TeacherDataset,
    config: TeacherExperimentConfig,
) -> list[PairedSequenceBatch]:
    batches: list[PairedSequenceBatch] = []
    for animal in dataset.animals:
        if animal.role == "target":
            continue
        batches.extend(
            paired_intervention_sequence_batches(
                animal,
                animal.counterfactual_pairs,
                batch_size=config.batch_size,
                neural_transform=config.neural_transform,
            )
        )
    return batches


def _normal_audit_r2(
    model: HierarchicalControlledSSM,
    animal: AnimalDataset,
    gauge: GaugeMap,
    neural_transform: str,
) -> float:
    audit = [trial for trial in animal.normal_trials if trial.split == "normal_audit"]
    if not audit:
        return float("nan")
    learned, teacher, _ = _encode_normal(model, animal, audit, neural_transform)
    return _variance_explained(gauge.transform(learned), teacher)


def run_teacher_experiment(
    dataset: TeacherDataset,
    config: TeacherExperimentConfig,
    output_dir: str | Path,
    *,
    run_seed: int = 0,
    overwrite: bool = False,
    freeze_attestation: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run all requested methods for one already-materialized teacher world.

    Returns the same JSON-serializable payload written to ``metrics.json``.
    """

    config.validate()
    locked = dataset.ground_truth.seed_partition == "locked"
    if locked:
        validate_locked_teacher_config(dataset.ground_truth.config)
        validate_locked_teacher_experiment_config(config)
        if run_seed != 0:
            raise ProtocolViolation("post-freeze procedural teacher evaluation requires run_seed=0")
        if overwrite:
            raise ProtocolViolation(
                "post-freeze procedural teacher evaluation never overwrites completed artifacts"
            )
    dataset.validate()
    if locked and freeze_attestation is None:
        raise ProtocolViolation(
            "post-freeze procedural teacher outcomes require a validated "
            "pre-outcome freeze attestation"
        )
    if locked:
        validated_freeze = attest_preoutcome_freeze(repository=Path(__file__).resolve().parents[3])
        if dict(freeze_attestation or {}) != asdict(validated_freeze):
            raise ProtocolViolation("supplied teacher freeze attestation is not current")
    output = Path(output_dir)
    canonical_relative_output = (
        _locked_teacher_output_identity(output, dataset.ground_truth) if locked else None
    )
    metrics_path = output / "metrics.json"
    metrics_sha256_path = output / "metrics.json.sha256"
    predictions_path = output / "predictions.npz"
    predictions_sha256_path = output / "predictions.npz.sha256"
    completion_path = output / "completion.json"
    temporary_predictions = output / "predictions.npz.tmp"
    material_artifacts = (
        metrics_path,
        metrics_sha256_path,
        predictions_path,
        predictions_sha256_path,
        completion_path,
        temporary_predictions,
    )
    if not overwrite and any(path.exists() for path in material_artifacts):
        raise FileExistsError(f"refusing to overwrite experiment artifacts in {output}")
    output.mkdir(parents=True, exist_ok=True)

    truth = dataset.ground_truth
    donors = [animal for animal in dataset.animals if animal.role != "target"]
    targets = [animal for animal in dataset.animals if animal.role == "target"]
    if not donors or not targets:
        raise ValueError("teacher experiment requires donors and targets")

    normal_train, normal_validation = _normal_data_for_donors(dataset, config)
    intervention_train, intervention_validation = _intervention_data_for_donors(dataset, config)
    all_donor_normal = _all_donor_normal_batches(dataset, config)
    all_donor_intervention = _all_donor_intervention_batches(dataset, config)
    validation_donors = [animal for animal in dataset.animals if animal.role == "validation_donor"]
    stage_records: dict[str, Any] = {}
    metrics_by_method: dict[str, dict[str, dict[str, Any]]] = {}
    prediction_arrays: dict[str, np.ndarray] = {}
    pending_scores: list[PendingTeacherScore] = []
    protocol_audit: dict[str, Any] = {
        "seed_material_public": True,
        "prospective_seed_secrecy": False,
        "evaluation_role": (
            "post_freeze_deterministic_procedural_audit" if locked else "method_development"
        ),
        "eligible_for_biological_headline_conjunction": False,
        "canonical_relative_output": canonical_relative_output,
        "teacher_config_sha256": teacher_config_sha256(dataset.ground_truth.config),
        "teacher_experiment_scientific_sha256": (teacher_experiment_scientific_sha256(config)),
        "execution_devices": sorted(
            {
                config.normal_fit.device,
                config.intervention_fit.device,
                config.target_fit.device,
            }
        ),
        "execution_numerics_disclosure": (
            "device and mixed-precision mode may vary as execution details; "
            "all scientific hyperparameters and random seeds are fingerprinted, "
            "and small cross-device floating-point differences may remain"
        ),
        "target_intervention_batches_used_for_optimization": 0,
        "target_adaptation_splits": ["normal_fit", "normal_val"],
        "target_neural_readout": (
            "softplus-Poisson quasi-likelihood fit on frozen open-loop "
            "normal_fit rollouts, selected on frozen open-loop normal_val "
            "rollouts"
            if config.calibrate_quasilikelihood_readout
            else (
                "linear log-count decoder fit during normal-only target adaptation"
                if config.linear_observation_family
                else "MLP log-count decoder fit during normal-only target adaptation"
            )
        ),
        "target_normal_audit_used_for_optimization": False,
        "target_readout_contemporaneous_count_encoded_as_its_own_predictor": False,
        "donor_delta_identification": {},
        "nested_selection_topology": {},
        "prediction_initialization_sample": "onset_minus_1",
        "prediction_mode": "paired_open_loop",
        "post_onset_outcomes_mounted_as_inputs": False,
        "preoutcome_freeze": (None if freeze_attestation is None else dict(freeze_attestation)),
    }
    start_time = time.monotonic()

    for method_index, method in enumerate(config.learned_methods):
        method_start = time.monotonic()
        initialization_seed = run_seed * 10_000 + method_index * 101 + 7
        selection_model = _make_model(
            method,
            truth,
            config,
            initialization_seed=initialization_seed,
        )
        _register_donors(selection_model, dataset, config)
        normal_selection = fit_deterministic_normal_stage(
            selection_model,
            normal_train,
            normal_validation,
            stage="normal",
            config=replace(config.normal_fit, seed=config.normal_fit.seed + method_index),
        )
        shared_after_training_normals = {
            name: value.detach().cpu().clone()
            for name, value in selection_model.shared.state_dict().items()
        }
        validation_adapter_records: dict[str, Any] = {}
        for validation_index, validation_animal in enumerate(validation_donors):
            adapter_fit_trials, adapter_validation_trials = _split_normal_trials(
                validation_animal, config.validation_fraction
            )
            adapter_fit = normal_sequence_batches(
                validation_animal,
                batch_size=config.batch_size,
                neural_transform=config.neural_transform,
                trials=adapter_fit_trials,
            )
            adapter_validation = normal_sequence_batches(
                validation_animal,
                batch_size=config.batch_size,
                neural_transform=config.neural_transform,
                trials=adapter_validation_trials,
            )
            adapter_result = fit_deterministic_normal_stage(
                selection_model,
                adapter_fit,
                adapter_validation,
                stage="target_adaptation",
                target_animal=validation_animal.animal_id,
                config=replace(
                    config.target_fit,
                    seed=(config.target_fit.seed + 5000 + method_index * 101 + validation_index),
                ),
            )
            validation_adapter_records[validation_animal.animal_id] = _fit_result_summary(
                adapter_result
            )
        shared_change = 0.0
        for name, value in selection_model.shared.state_dict().items():
            difference = value.detach().cpu() - shared_after_training_normals[name]
            shared_change = max(shared_change, float(difference.abs().max()))
        if shared_change != 0.0:
            raise ProtocolViolation("validation-donor normal adaptation changed shared F")
        expected_selection_delta_groups = sum(
            animal.role == "train_donor" for animal in dataset.animals
        )
        if len(selection_model.donor_intervention_delta) != expected_selection_delta_groups:
            raise ProtocolViolation(
                "validation donor received an intervention delta before selection"
            )
        # fit_stage("intervention") freezes shared F, behavior decoder, and all
        # adapters before the first perturbation outcome is consumed.
        intervention_selection = fit_paired_intervention_stage(
            selection_model,
            intervention_train,
            intervention_validation,
            config=replace(
                config.intervention_fit,
                seed=config.intervention_fit.seed + method_index,
            ),
        )
        selection_delta_mean_norm = _donor_delta_mean_norm(selection_model)
        if selection_delta_mean_norm > 1e-7:
            raise ProtocolViolation(
                f"{method} selection donor deltas violate zero-mean identification"
            )

        selected_normal_epochs = normal_selection.best_epoch + 1
        selected_intervention_epochs = intervention_selection.best_epoch + 1
        # Fresh all-donor refit. Validation-donor outcomes enter only after both
        # epoch counts have been fixed by the nested selection fit above.
        model = _make_model(
            method,
            truth,
            config,
            initialization_seed=initialization_seed,
        )
        _register_donors(
            model,
            dataset,
            config,
            include_validation_deltas=True,
        )
        normal_refit = fit_deterministic_normal_stage(
            model,
            all_donor_normal,
            all_donor_normal,
            stage="normal",
            config=replace(
                config.normal_fit,
                seed=config.normal_fit.seed + method_index,
            ),
            fixed_epochs=selected_normal_epochs,
        )
        intervention_refit = fit_paired_intervention_stage(
            model,
            all_donor_intervention,
            all_donor_intervention,
            config=replace(
                config.intervention_fit,
                seed=config.intervention_fit.seed + method_index,
            ),
            fixed_epochs=selected_intervention_epochs,
        )
        donor_delta_mean_norm = _donor_delta_mean_norm(model)
        if donor_delta_mean_norm > 1e-7:
            raise ProtocolViolation(f"{method} donor deltas violate exact zero-mean identification")
        protocol_audit["donor_delta_identification"][method] = {
            "constraint": "exact_zero_mean_projection_after_every_optimizer_step",
            "training_group_count": len(model.donor_intervention_delta),
            "final_mean_l2_norm": donor_delta_mean_norm,
            "tolerance": 1e-7,
        }
        protocol_audit["nested_selection_topology"][method] = {
            "shared_normal_training_roles": ["train_donor"],
            "validation_donor_adapter_roles": ["validation_donor"],
            "validation_adapter_shared_parameter_max_abs_change": shared_change,
            "validation_interventions_used_for_gradient_steps_before_selection": False,
            "validation_intervention_delta_present_before_selection": False,
            "selection_training_delta_group_count": len(selection_model.donor_intervention_delta),
            "selected_normal_epochs": selected_normal_epochs,
            "selected_intervention_epochs": selected_intervention_epochs,
            "final_refit_roles": ["train_donor", "validation_donor"],
            "final_refit_epoch_selection_from_refit_data": False,
        }
        method_stage: dict[str, Any] = {
            "normal": _fit_result_summary(normal_refit),
            "intervention": _fit_result_summary(intervention_refit),
            "selection": {
                "normal_train_donors_only": _fit_result_summary(normal_selection),
                "validation_donor_normal_adaptation": (validation_adapter_records),
                "intervention_train_donors_validate_on_validation_donors": (
                    _fit_result_summary(intervention_selection)
                ),
                "topology_audit": protocol_audit["nested_selection_topology"][method],
            },
            "targets": {},
        }
        method_stage["intervention"]["donor_delta_identification"] = protocol_audit[
            "donor_delta_identification"
        ][method]
        metrics_by_method[method] = {}

        for target_index, animal in enumerate(targets):
            model_device = next(model.parameters()).device
            _configure_animal_observation_family(model, animal, config)
            # ModuleDict additions are initialized on CPU even when donor fitting
            # has already moved the parent model to CUDA.
            model.to(model_device)
            unadapted_model = (
                copy.deepcopy(model) if method == "proposed" and config.include_ablations else None
            )
            target_train = normal_sequence_batches(
                animal,
                batch_size=config.batch_size,
                neural_transform=config.neural_transform,
                splits={"normal_fit"},
            )
            target_validation = normal_sequence_batches(
                animal,
                batch_size=config.batch_size,
                neural_transform=config.neural_transform,
                splits={"normal_val"},
            )
            target_result = fit_deterministic_normal_stage(
                model,
                target_train,
                target_validation,
                stage="target_adaptation",
                target_animal=animal.animal_id,
                config=replace(
                    config.target_fit,
                    seed=config.target_fit.seed + method_index * 17 + target_index,
                ),
            )
            method_stage["targets"][animal.animal_id] = _fit_result_summary(target_result)
            neural_readout: SoftplusNeuralReadout | None = None
            if config.calibrate_quasilikelihood_readout:
                neural_readout = fit_normal_only_neural_readout(
                    model,
                    animal,
                    config,
                    seed=(config.target_fit.seed + method_index * 31 + target_index + 1000),
                )
                method_stage["targets"][animal.animal_id]["neural_readout"] = (
                    neural_readout.summary()
                )

            fit_trials = [trial for trial in animal.normal_trials if trial.split == "normal_fit"]
            learned_normal, teacher_normal, normal_inputs = _encode_normal(
                model, animal, fit_trials, config.neural_transform
            )
            gauge = fit_affine_gauge(learned_normal, teacher_normal)
            native_prediction = predict_target_pairs(
                model,
                animal,
                neural_transform=config.neural_transform,
                include_animal_residual=True,
            )
            prediction = predict_target_pairs(
                model,
                animal,
                neural_transform=config.neural_transform,
                include_animal_residual=True,
                neural_readout=neural_readout,
            )
            if neural_readout is not None:
                method_stage["targets"][animal.animal_id]["neural_readout"].update(
                    neural_readout.extrapolation_summary(
                        np.concatenate(
                            (
                                prediction.latent_treated,
                                prediction.latent_control,
                            ),
                            axis=0,
                        )
                    )
                )
            pending_scores.append(
                PendingTeacherScore(
                    method=method,
                    animal=animal,
                    model=model,
                    prediction=prediction,
                    truth=truth,
                    gauge=gauge,
                    learned_normal=learned_normal,
                    teacher_normal=teacher_normal,
                    normal_inputs=normal_inputs,
                    max_vector_field_states=config.max_vector_field_states,
                    extra_metrics={
                        "normal_audit_latent_r2_affine_gauge": _normal_audit_r2(
                            model, animal, gauge, config.neural_transform
                        )
                    },
                )
            )
            prefix = f"{_safe_key(method)}__{_safe_key(animal.animal_id)}"
            prediction_arrays.update(_prediction_arrays(prefix, prediction, gauge))

            if method == "proposed":
                zero = _zero_prediction(prediction)
                metrics_by_method.setdefault("zero_effect", {})
                pending_scores.append(
                    PendingTeacherScore(
                        method="zero_effect",
                        animal=animal,
                        model=model,
                        prediction=zero,
                        truth=truth,
                        gauge=gauge,
                        learned_normal=learned_normal,
                        teacher_normal=teacher_normal,
                        normal_inputs=normal_inputs,
                        max_vector_field_states=config.max_vector_field_states,
                        include_operator=False,
                    )
                )
                zero_prefix = f"zero_effect__{_safe_key(animal.animal_id)}"
                prediction_arrays.update(_prediction_arrays(zero_prefix, zero, gauge))

                if config.include_ablations:
                    if neural_readout is not None:
                        native_name = "proposed_native_decoder"
                        metrics_by_method.setdefault(native_name, {})
                        pending_scores.append(
                            PendingTeacherScore(
                                method=native_name,
                                animal=animal,
                                model=model,
                                prediction=native_prediction,
                                truth=truth,
                                gauge=gauge,
                                learned_normal=learned_normal,
                                teacher_normal=teacher_normal,
                                normal_inputs=normal_inputs,
                                max_vector_field_states=config.max_vector_field_states,
                            )
                        )
                        prediction_arrays.update(
                            _prediction_arrays(
                                f"{native_name}__{_safe_key(animal.animal_id)}",
                                native_prediction,
                                gauge,
                            )
                        )
                    no_residual = predict_target_pairs(
                        model,
                        animal,
                        neural_transform=config.neural_transform,
                        include_animal_residual=False,
                        neural_readout=neural_readout,
                    )
                    ablation_name = "proposed_no_target_residual"
                    metrics_by_method.setdefault(ablation_name, {})
                    pending_scores.append(
                        PendingTeacherScore(
                            method=ablation_name,
                            animal=animal,
                            model=model,
                            prediction=no_residual,
                            truth=truth,
                            gauge=gauge,
                            learned_normal=learned_normal,
                            teacher_normal=teacher_normal,
                            normal_inputs=normal_inputs,
                            max_vector_field_states=config.max_vector_field_states,
                        )
                    )
                    prediction_arrays.update(
                        _prediction_arrays(
                            f"{ablation_name}__{_safe_key(animal.animal_id)}",
                            no_residual,
                            gauge,
                        )
                    )

                    if unadapted_model is None:
                        raise RuntimeError("unadapted ablation snapshot is missing")
                    unadapted_learned, _, _ = _encode_normal(
                        unadapted_model,
                        animal,
                        fit_trials,
                        config.neural_transform,
                    )
                    unadapted_gauge = fit_affine_gauge(unadapted_learned, teacher_normal)
                    no_adaptation = predict_target_pairs(
                        unadapted_model,
                        animal,
                        neural_transform=config.neural_transform,
                    )
                    no_adaptation_name = "proposed_no_target_adaptation"
                    metrics_by_method.setdefault(no_adaptation_name, {})
                    pending_scores.append(
                        PendingTeacherScore(
                            method=no_adaptation_name,
                            animal=animal,
                            model=unadapted_model,
                            prediction=no_adaptation,
                            truth=truth,
                            gauge=unadapted_gauge,
                            learned_normal=unadapted_learned,
                            teacher_normal=teacher_normal,
                            normal_inputs=normal_inputs,
                            max_vector_field_states=config.max_vector_field_states,
                        )
                    )
                    prediction_arrays.update(
                        _prediction_arrays(
                            f"{no_adaptation_name}__{_safe_key(animal.animal_id)}",
                            no_adaptation,
                            unadapted_gauge,
                        )
                    )

            diagnostic_prefix = f"normal_diagnostic__{_safe_key(animal.animal_id)}"
            prediction_arrays[f"{diagnostic_prefix}__gauge_linear__{_safe_key(method)}"] = (
                gauge.linear.astype(np.float32)
            )
            prediction_arrays[f"{diagnostic_prefix}__gauge_offset__{_safe_key(method)}"] = (
                gauge.offset.astype(np.float32)
            )
            if neural_readout is not None:
                prediction_arrays[
                    f"{diagnostic_prefix}__normal_readout_weight__{_safe_key(method)}"
                ] = neural_readout.weight.astype(np.float32)
                prediction_arrays[
                    f"{diagnostic_prefix}__normal_readout_bias__{_safe_key(method)}"
                ] = neural_readout.bias.astype(np.float32)
                prediction_arrays[
                    f"{diagnostic_prefix}__normal_readout_latent_mean__{_safe_key(method)}"
                ] = neural_readout.latent_mean.astype(np.float32)
                prediction_arrays[
                    f"{diagnostic_prefix}__normal_readout_latent_scale__{_safe_key(method)}"
                ] = neural_readout.latent_scale.astype(np.float32)
                prediction_arrays[
                    f"{diagnostic_prefix}__normal_readout_support_min__{_safe_key(method)}"
                ] = neural_readout.support_min.astype(np.float32)
                prediction_arrays[
                    f"{diagnostic_prefix}__normal_readout_support_max__{_safe_key(method)}"
                ] = neural_readout.support_max.astype(np.float32)

        method_stage["wall_seconds"] = time.monotonic() - method_start
        stage_records[method] = method_stage

    # Persist and hash every eligible prediction before the first target
    # intervention truth is accessed by `_score_prediction_with_model`.
    prediction_metadata = {
        "schema_version": "cadence.teacher_prediction.v1",
        "world_id": truth.world_id,
        "run_seed": run_seed,
        "seed_material_public": True,
        "prospective_seed_secrecy": False,
        "eligible_for_biological_headline_conjunction": False,
        "canonical_relative_output": canonical_relative_output,
        "methods": sorted(metrics_by_method),
        "learned_methods": list(config.learned_methods),
        "canonical_learned_method_set_complete": (tuple(config.learned_methods) == LEARNED_METHODS),
        "targets": [animal.animal_id for animal in targets],
        "teacher_config_sha256": teacher_config_sha256(truth.config),
        "teacher_experiment_scientific_sha256": (teacher_experiment_scientific_sha256(config)),
        "preoutcome_freeze": (None if freeze_attestation is None else dict(freeze_attestation)),
        "contains_target_intervention_truth": False,
    }
    with temporary_predictions.open("wb") as stream:
        np.savez_compressed(
            stream,
            metadata_json=np.asarray(json.dumps(prediction_metadata, sort_keys=True)),
            **prediction_arrays,
        )
    if locked:
        with (
            temporary_predictions.open("rb") as source,
            predictions_path.open("xb") as destination,
        ):
            shutil.copyfileobj(source, destination)
        temporary_predictions.unlink()
    else:
        temporary_predictions.replace(predictions_path)
    prediction_sha256 = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    _write_text_artifact(
        predictions_sha256_path,
        f"{prediction_sha256}  {predictions_path.name}\n",
        exclusive=locked,
    )
    protocol_audit.update(
        {
            "prediction_sha256_before_score": prediction_sha256,
            "prediction_hashed_before_target_truth_access": True,
            "prediction_bundle_contains_target_intervention_truth": False,
            "teacher_outcome_boundary": (
                "deterministic simulator held in memory; score routine first invoked "
                "after complete prediction-bundle hash"
            ),
        }
    )

    verified_prediction_sha256 = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    if verified_prediction_sha256 != prediction_sha256:
        raise ProtocolViolation("prediction bundle changed after pre-score hash")
    if predictions_sha256_path.read_text(encoding="utf-8").split()[0] != prediction_sha256:
        raise ProtocolViolation("prediction hash sidecar is inconsistent")

    for pending in pending_scores:
        scores = _score_prediction_with_model(
            pending.model,
            pending.prediction,
            pending.animal,
            pending.truth,
            pending.gauge,
            learned_normal=pending.learned_normal,
            teacher_normal=pending.teacher_normal,
            normal_inputs=pending.normal_inputs,
            max_vector_field_states=pending.max_vector_field_states,
            include_operator=pending.include_operator,
        )
        scores.update(pending.extra_metrics)
        metrics_by_method[pending.method][pending.animal.animal_id] = scores

    aggregate = _aggregate_metrics(metrics_by_method)
    payload: dict[str, Any] = {
        "schema_version": "cadence.teacher_experiment.v1",
        "world": dataset.summary(),
        "canonical_relative_output": canonical_relative_output,
        "experiment_config": config.to_mapping(),
        "learned_methods": list(config.learned_methods),
        "canonical_learned_method_set_complete": (tuple(config.learned_methods) == LEARNED_METHODS),
        "reported_methods": sorted(metrics_by_method),
        "protocol_audit": protocol_audit,
        "metric_definitions": {
            "primary_observed_endpoints": (
                "condition-averaged paired effects, grouped by intervention and dose"
            ),
            "causal_skill_scaling": (
                "1 - weighted prediction SSE / weighted zero-effect SSE; "
                "per-channel weights are inverse target normal_fit standard "
                "deviations with frozen quantile clipping. Zero effect is 0, "
                "perfect prediction is 1, and negative values are intentionally "
                "unbounded rather than clipped"
            ),
            "single_trial_endpoints": (
                "realized paired effects before repetition averaging; retained as "
                "noise-sensitive secondary endpoints"
            ),
            "pathwise_mean_endpoints": (
                "synthetic-only predictions versus teacher observation means "
                "conditional on the realized future latent/process path; these are "
                "localization diagnostics, not forecastable E[outcome|pre,schedule]"
            ),
            "observation_oracle": (
                "exact teacher conditional-mean effect scored against the realized "
                "paired observations"
            ),
            "gauge_diagnostics": (
                "affine map fit only on target normal_fit states; no target "
                "intervention state is used to choose the map"
            ),
            "gauge_true_h_oracle": (
                "evaluation-only localization diagnostic: affine-gauge the frozen "
                "predicted latent trajectory using normal_fit, then apply the exact "
                "teacher target observation map; never treated as an eligible method"
            ),
        },
        "stage_fits": stage_records,
        "metrics_by_method_and_target": metrics_by_method,
        "aggregate": aggregate,
        "wall_seconds": time.monotonic() - start_time,
        "artifacts": {
            "metrics": metrics_path.name,
            "metrics_sha256": metrics_sha256_path.name,
            "predictions": predictions_path.name,
            "predictions_sha256": predictions_sha256_path.name,
            "completion": completion_path.name,
        },
    }
    payload = _jsonable(payload)
    metrics_text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _write_text_artifact(
        metrics_path,
        metrics_text,
        exclusive=locked,
    )
    metrics_sha256 = hashlib.sha256(metrics_text.encode()).hexdigest()
    _write_text_artifact(
        metrics_sha256_path,
        f"{metrics_sha256}  {metrics_path.name}\n",
        exclusive=locked,
    )
    completion = {
        "schema_version": "cadence.teacher_completion.v1",
        "world_id": truth.world_id,
        "seed_partition": truth.seed_partition,
        "seed_material_public": True,
        "evaluation_role": (
            "post_freeze_deterministic_procedural_audit" if locked else "method_development"
        ),
        "eligible_for_biological_headline_conjunction": False,
        "canonical_relative_output": canonical_relative_output,
        "learned_methods": list(config.learned_methods),
        "canonical_learned_method_set_complete": (tuple(config.learned_methods) == LEARNED_METHODS),
        "reported_methods": sorted(metrics_by_method),
        "teacher_config_sha256": teacher_config_sha256(truth.config),
        "teacher_experiment_scientific_sha256": (teacher_experiment_scientific_sha256(config)),
        "preoutcome_freeze": (None if freeze_attestation is None else dict(freeze_attestation)),
        "artifacts": {
            metrics_path.name: metrics_sha256,
            predictions_path.name: prediction_sha256,
        },
    }
    _write_text_artifact(
        completion_path,
        json.dumps(completion, indent=2, sort_keys=True, allow_nan=False) + "\n",
        exclusive=locked,
    )
    return payload


def generate_and_run_teacher_experiment(
    teacher_config: TeacherConfig,
    experiment_config: TeacherExperimentConfig,
    output_dir: str | Path,
    *,
    partition: Literal["development", "locked"] = "development",
    seed_index: int = 0,
    run_seed: int = 0,
    overwrite: bool = False,
    freeze_attestation: FreezeAttestation | None = None,
) -> dict[str, Any]:
    """Convenience wrapper used by the command-line runner."""

    profile_config = make_profile_teacher_config(teacher_config, experiment_config.profile)
    world = generate_teacher_world(
        profile_config,
        partition=partition,
        seed_index=seed_index,
        freeze_attestation=freeze_attestation,
    )
    dataset = world.generate_dataset()
    return run_teacher_experiment(
        dataset,
        experiment_config,
        output_dir,
        run_seed=run_seed,
        overwrite=overwrite,
        freeze_attestation=(None if freeze_attestation is None else asdict(freeze_attestation)),
    )


def compact_result_table(payload: dict[str, Any]) -> str:
    """Render primary ATE and synthetic recoverability endpoints."""

    header = "method                           neural_ATE  behavior_ATE  neural_mean  latent_skill"
    divider = "-" * len(header)
    rows = [header, divider]
    aggregate = payload["aggregate"]
    ordered = [name for name in DEFAULT_REPORT_METHODS if name in aggregate]
    ordered.extend(name for name in aggregate if name not in ordered)
    for method in ordered:
        values = aggregate[method]
        rows.append(
            f"{method:32s} "
            f"{values['neural_condition_averaged_causal_skill_mean']:10.4f} "
            f"{values['behavior_condition_averaged_causal_skill_mean']:13.4f} "
            f"{values['neural_pathwise_mean_causal_skill_mean']:12.4f} "
            f"{values['latent_effect_skill_affine_gauge_mean']:12.4f}"
        )
    return "\n".join(rows)


__all__ = [
    "DEFAULT_REPORT_METHODS",
    "GaugeMap",
    "LEARNED_METHODS",
    "TargetPrediction",
    "TeacherExperimentConfig",
    "compact_result_table",
    "fit_affine_gauge",
    "generate_and_run_teacher_experiment",
    "intervention_sequence_batches",
    "make_experiment_config",
    "make_profile_teacher_config",
    "normal_sequence_batches",
    "predict_target_pairs",
    "run_teacher_experiment",
    "teacher_experiment_scientific_sha256",
    "validate_locked_teacher_experiment_config",
]
