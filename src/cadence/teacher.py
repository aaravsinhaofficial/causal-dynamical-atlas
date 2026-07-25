"""Procedural teacher-RNN benchmark for cross-animal causal transfer.

The generator in this module is deliberately *procedural*: a world seed fixes the
causal operator, animal-specific residuals, observation maps, and every trial's
exogenous noise.  Interventional trials are emitted with a counterfactual control
twin which uses exactly the same initial state, task input, process innovations,
and observation-noise variables.

Two properties are especially useful for rigorous experiments:

* random streams are addressed by semantic labels rather than consumed from one
  global generator, so adding trials or changing a stress axis does not silently
  change unrelated examples;
* the ``independent_target_direction`` impossibility control admits a pair of
  worlds (``impossibility_variant=+1`` and ``-1``) that are identical under all
  normal activity but have opposite target-animal intervention directions.

The default configuration in ``configs/teacher.yaml`` contains public,
predeclared post-freeze procedural-audit seeds.  The historical internal
partition name ``locked`` is retained for schema compatibility, but it does not
imply prospective seed secrecy or blinding.  Small configurations can be
constructed with :func:`dataclasses.replace` for unit tests and smoke
experiments.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import yaml
from scipy import stats

from cadence.protocol import (
    FreezeAttestation,
    ProtocolViolation,
    attest_preoutcome_freeze,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int32]
BoolArray = npt.NDArray[np.bool_]
AnimalRole = Literal["train_donor", "validation_donor", "target"]
SeedPartition = Literal["development", "locked"]
NoiseModel = Literal["negative_binomial", "poisson"]
Coverage = Literal["full", "narrow"]
ImpossibilityMode = Literal["none", "independent_target_direction"]

SCHEMA_VERSION = "cadence.teacher.v1"
CANONICAL_TEACHER_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "teacher.yaml"
NORMAL_SPLIT_CODES = {"normal": 0, "normal_fit": 1, "normal_val": 2, "normal_audit": 3}
ROLE_CODES = {"train_donor": 0, "validation_donor": 1, "target": 2}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _jsonable(value: Any) -> Any:
    """Convert numpy/path/dataclass-adjacent values to canonical JSON values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _addressed_seed(base_seed: int, *address: object) -> int:
    """Return an ordering-independent uint64 seed for a semantic address."""

    payload = _canonical_json([SCHEMA_VERSION, int(base_seed), *address]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def _rng(base_seed: int, *address: object) -> np.random.Generator:
    return np.random.default_rng(_addressed_seed(base_seed, *address))


def _spectral_normalize(matrix: FloatArray, target_norm: float) -> FloatArray:
    norm = float(np.linalg.svd(matrix, compute_uv=False)[0])
    _require(norm > 0.0, "cannot normalize a zero matrix")
    return matrix * (target_norm / norm)


def _softplus(x: FloatArray) -> FloatArray:
    # Numerically stable log(1 + exp(x)).
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


@dataclass(frozen=True)
class CohortConfig:
    """Animal counts for the donor/target transfer protocol."""

    train_donors: int = 10
    validation_donors: int = 2
    targets: int = 4

    @property
    def n_animals(self) -> int:
        return self.train_donors + self.validation_donors + self.targets

    @property
    def roles(self) -> tuple[AnimalRole, ...]:
        return (
            *("train_donor" for _ in range(self.train_donors)),
            *("validation_donor" for _ in range(self.validation_donors)),
            *("target" for _ in range(self.targets)),
        )

    def validate(self) -> None:
        _require(self.train_donors >= 1, "at least one training donor is required")
        _require(self.validation_donors >= 0, "validation_donors must be nonnegative")
        _require(self.targets >= 1, "at least one target animal is required")


@dataclass(frozen=True)
class DynamicsConfig:
    """Shared and animal-residual latent dynamics."""

    latent_dim: int = 8
    task_input_dim: int = 4
    dt: float = 0.05
    shared_operator_norm: float = 0.55
    residual_rank: int = 2
    residual_ratio: float = 0.10
    process_noise_std: float = 0.025
    task_input_scale: float = 0.70

    def validate(self) -> None:
        _require(self.latent_dim >= 3, "latent_dim must be at least three")
        _require(self.task_input_dim >= 1, "task_input_dim must be positive")
        _require(0.0 < self.dt <= 1.0, "dt must be in (0, 1]")
        _require(0.0 < self.shared_operator_norm < 1.0, "shared operator must contract")
        _require(self.residual_rank == 2, "the locked teacher benchmark uses rank-2 residuals")
        _require(self.residual_rank < self.latent_dim, "residual rank must be below latent_dim")
        _require(0.0 <= self.residual_ratio <= 0.5, "residual_ratio must be in [0, .5]")
        _require(self.process_noise_std >= 0.0, "process noise must be nonnegative")
        _require(self.task_input_scale > 0.0, "task input scale must be positive")


@dataclass(frozen=True)
class InterventionConfig:
    """Controlled state-dependent intervention fields."""

    n_interventions: int = 6
    onset_step: int = 40
    offset_step: int = 60
    doses: tuple[float, ...] = (1.0,)
    shared_bias_norm: float = 0.60
    shared_state_norm: float = 0.20
    state_rank: int = 2
    animal_residual_ratio: float = 0.05

    def validate(self, steps: int, latent_dim: int) -> None:
        _require(self.n_interventions >= 1, "n_interventions must be positive")
        _require(0 <= self.onset_step < self.offset_step <= steps, "invalid pulse interval")
        _require(bool(self.doses), "at least one intervention dose is required")
        _require(all(dose > 0.0 for dose in self.doses), "all doses must be positive")
        _require(self.shared_bias_norm > 0.0, "shared intervention bias must be nonzero")
        _require(self.shared_state_norm >= 0.0, "shared intervention state norm is invalid")
        _require(self.state_rank == 2, "the main teacher intervention field is rank two")
        _require(self.state_rank < latent_dim, "intervention rank must be below latent_dim")
        _require(
            0.0 <= self.animal_residual_ratio <= 1.0,
            "animal intervention residual ratio must be in [0, 1]",
        )


@dataclass(frozen=True)
class ObservationConfig:
    """Variable neural observations and shared-unit behavioral readouts."""

    neurons_min: int = 64
    neurons_max: int = 128
    behavior_dim: int = 3
    neural_bias_mean: float = 1.0
    neural_bias_std: float = 0.25
    neural_map_scale: float = 0.75
    neural_noise_model: NoiseModel = "negative_binomial"
    nb_dispersion: float = 20.0
    behavior_noise_std: float = 0.04
    behavior_residual_ratio: float = 0.10

    def validate(self, latent_dim: int) -> None:
        _require(self.neurons_min >= 1, "neurons_min must be positive")
        _require(self.neurons_max >= self.neurons_min, "invalid neuron-count interval")
        _require(
            self.behavior_dim == 3,
            "the frozen main regime has three behavior channels",
        )
        _require(self.neural_map_scale > 0.0, "neural_map_scale must be positive")
        _require(
            self.neural_noise_model in {"negative_binomial", "poisson"},
            "neural_noise_model must be negative_binomial or poisson",
        )
        if self.neural_noise_model == "negative_binomial":
            _require(self.nb_dispersion > 0.0, "NB dispersion (size) must be positive")
        _require(self.behavior_noise_std >= 0.0, "behavior noise must be nonnegative")
        _require(
            0.0 <= self.behavior_residual_ratio <= 1.0,
            "behavior_residual_ratio must be in [0, 1]",
        )
        _require(latent_dim >= 3, "three behavioral outputs require latent_dim >= 3")


@dataclass(frozen=True)
class TrialConfig:
    """Time horizon and exact per-animal protocol allocations."""

    steps: int = 100
    donor_normal_trials: int = 96
    donor_pairs_per_intervention: int = 32
    target_normal_fit_trials: int = 64
    target_normal_val_trials: int = 16
    target_normal_audit_trials: int = 32
    target_pairs_per_intervention: int = 24

    @property
    def target_normal_trials(self) -> int:
        return (
            self.target_normal_fit_trials
            + self.target_normal_val_trials
            + self.target_normal_audit_trials
        )

    def validate(self) -> None:
        _require(self.steps >= 3, "steps must be at least three")
        values = (
            self.donor_normal_trials,
            self.donor_pairs_per_intervention,
            self.target_normal_fit_trials,
            self.target_normal_val_trials,
            self.target_normal_audit_trials,
            self.target_pairs_per_intervention,
        )
        _require(all(value >= 0 for value in values), "trial allocations must be nonnegative")
        _require(self.donor_normal_trials >= 1, "donors require normal calibration trials")
        _require(self.target_normal_fit_trials >= 1, "targets require normal-fit trials")


@dataclass(frozen=True)
class SeedConfig:
    """Separate development and public post-freeze procedural world seeds."""

    development: tuple[int, ...]
    locked: tuple[int, ...]

    def validate(self) -> None:
        _require(len(self.development) == 10, "benchmark requires 10 development seeds")
        _require(
            len(self.locked) == 20,
            "benchmark requires 20 post-freeze procedural seeds",
        )
        all_seeds = (*self.development, *self.locked)
        _require(len(set(all_seeds)) == len(all_seeds), "world seeds must be unique")
        _require(all(seed >= 0 for seed in all_seeds), "world seeds must be nonnegative")


@dataclass(frozen=True)
class TeacherConfig:
    """Complete procedural benchmark configuration."""

    release_name: str
    cohort: CohortConfig
    dynamics: DynamicsConfig
    intervention: InterventionConfig
    observations: ObservationConfig
    trials: TrialConfig
    seeds: SeedConfig

    def validate(self, *, strict_seed_counts: bool = True) -> None:
        _require(bool(self.release_name), "release_name cannot be empty")
        self.cohort.validate()
        self.dynamics.validate()
        self.trials.validate()
        self.intervention.validate(self.trials.steps, self.dynamics.latent_dim)
        self.observations.validate(self.dynamics.latent_dim)
        if strict_seed_counts:
            self.seeds.validate()
        else:
            _require(bool(self.seeds.development), "at least one development seed is required")
            _require(bool(self.seeds.locked), "at least one locked seed is required")

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], *, strict_seed_counts: bool = True
    ) -> TeacherConfig:
        """Parse the documented nested YAML/mapping schema."""

        config = cls(
            release_name=str(raw.get("release_name", "teacher-rnn-v1")),
            cohort=CohortConfig(**dict(raw["cohort"])),
            dynamics=DynamicsConfig(**dict(raw["dynamics"])),
            intervention=InterventionConfig(
                **{
                    **dict(raw["intervention"]),
                    "doses": tuple(raw["intervention"].get("doses", (1.0,))),
                }
            ),
            observations=ObservationConfig(**dict(raw["observations"])),
            trials=TrialConfig(**dict(raw["trials"])),
            seeds=SeedConfig(
                development=tuple(int(seed) for seed in raw["seeds"]["development"]),
                locked=tuple(int(seed) for seed in raw["seeds"]["locked"]),
            ),
        )
        config.validate(strict_seed_counts=strict_seed_counts)
        return config

    @classmethod
    def from_yaml(cls, path: str | Path, *, strict_seed_counts: bool = True) -> TeacherConfig:
        with Path(path).open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        _require(isinstance(raw, Mapping), "teacher config must contain a mapping")
        return cls.from_mapping(raw, strict_seed_counts=strict_seed_counts)

    def to_mapping(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class StressCondition:
    """One point on the preregistered benchmark stress grid.

    ``eta`` scales the conserved intervention component. ``rho`` is the
    animal-residual/shared-dynamics operator-norm ratio. ``support`` overrides
    donor pairs per intervention. ``target_neurons`` fixes target population
    size, and narrow state coverage restricts normal target task inputs.
    """

    eta: float = 1.0
    rho: float = 0.10
    target_neurons: int | None = None
    support: int | None = None
    state_coverage: Coverage = "full"
    impossibility: ImpossibilityMode = "none"
    impossibility_variant: int = 1

    def validate(self) -> None:
        _require(self.eta in {0.0, 0.5, 0.8, 1.0}, "eta must be one of {0, .5, .8, 1}")
        _require(self.rho in {0.0, 0.1, 0.25, 0.5}, "rho must be one of {0, .1, .25, .5}")
        _require(
            self.target_neurons in {None, 32, 64, 128},
            "target_neurons must be one of {32, 64, 128}",
        )
        _require(self.support in {None, 8, 16, 32, 64}, "support must be in {8,16,32,64}")
        _require(self.state_coverage in {"full", "narrow"}, "invalid state coverage")
        _require(
            self.impossibility in {"none", "independent_target_direction"},
            "invalid impossibility control",
        )
        _require(self.impossibility_variant in {-1, 1}, "impossibility_variant must be ±1")

    @property
    def tag(self) -> str:
        neurons = "native" if self.target_neurons is None else str(self.target_neurons)
        support = "native" if self.support is None else str(self.support)
        impossible = (
            "none" if self.impossibility == "none" else f"private{self.impossibility_variant:+d}"
        )
        return (
            f"eta{self.eta:g}-rho{self.rho:g}-n{neurons}-s{support}-"
            f"coverage{self.state_coverage}-impossible{impossible}"
        )


@dataclass(frozen=True)
class TeacherGroundTruth:
    """All matrices defining one teacher world."""

    schema_version: str
    release_name: str
    world_id: str
    world_seed: int
    seed_partition: SeedPartition
    seed_index: int
    config: TeacherConfig
    stress: StressCondition
    animal_roles: tuple[AnimalRole, ...]
    shared_recurrent: FloatArray
    shared_bias: FloatArray
    task_input_map: FloatArray
    residual_left: FloatArray
    residual_right: FloatArray
    intervention_bias: FloatArray
    intervention_state: FloatArray
    animal_intervention_residual: FloatArray
    animal_shared_intervention_gain: FloatArray
    behavior_shared: FloatArray
    behavior_residual: FloatArray
    neural_maps: tuple[FloatArray, ...]
    neural_biases: tuple[FloatArray, ...]
    neuron_counts: IntArray
    stability_bound: FloatArray

    @property
    def n_animals(self) -> int:
        return len(self.animal_roles)

    @property
    def max_neurons(self) -> int:
        return int(self.neuron_counts.max())

    def effective_intervention_field(
        self, animal_index: int, intervention_index: int, state: FloatArray
    ) -> FloatArray:
        """Evaluate the true action vector field for one animal and intervention."""

        shared = self.intervention_bias[intervention_index] + (
            self.intervention_state[intervention_index] @ np.tanh(state)
        )
        gain = self.animal_shared_intervention_gain[animal_index, intervention_index]
        return (
            self.stress.eta * gain * shared
            + self.animal_intervention_residual[animal_index, intervention_index]
        )

    def normal_operator_matrix(self, animal_index: int) -> FloatArray:
        """Linearized residual matrix (rank at most two) for diagnostics."""

        return self.residual_left[animal_index] @ self.residual_right[animal_index]

    def _padded_observation_arrays(self) -> tuple[FloatArray, FloatArray, BoolArray]:
        n_animals = self.n_animals
        latent_dim = self.config.dynamics.latent_dim
        maps = np.zeros((n_animals, self.max_neurons, latent_dim), dtype=np.float64)
        biases = np.zeros((n_animals, self.max_neurons), dtype=np.float64)
        mask = np.zeros((n_animals, self.max_neurons), dtype=np.bool_)
        for animal_index, count in enumerate(self.neuron_counts):
            n = int(count)
            maps[animal_index, :n] = self.neural_maps[animal_index]
            biases[animal_index, :n] = self.neural_biases[animal_index]
            mask[animal_index, :n] = True
        return maps, biases, mask

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "release_name": self.release_name,
            "world_id": self.world_id,
            "world_seed": self.world_seed,
            "seed_partition": self.seed_partition,
            "seed_index": self.seed_index,
            "seed_material_public": True,
            "prospective_seed_secrecy": False,
            "evaluation_role": (
                "post_freeze_deterministic_procedural_audit"
                if self.seed_partition == "locked"
                else "method_development"
            ),
            "eligible_for_biological_headline_conjunction": False,
            "config": self.config.to_mapping(),
            "teacher_config_sha256": teacher_config_sha256(self.config),
            "stress": _jsonable(asdict(self.stress)),
            "animal_roles": list(self.animal_roles),
            "neuron_counts": self.neuron_counts.tolist(),
            "stability_bound": self.stability_bound.tolist(),
            "seed_commitment_sha256": hashlib.sha256(str(self.world_seed).encode()).hexdigest(),
        }

    def save(self, path: str | Path) -> Path:
        """Serialize every true matrix without pickle."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        maps, biases, mask = self._padded_observation_arrays()
        np.savez_compressed(
            destination,
            metadata_json=np.asarray(_canonical_json(self.metadata())),
            shared_recurrent=self.shared_recurrent,
            shared_bias=self.shared_bias,
            task_input_map=self.task_input_map,
            residual_left=self.residual_left,
            residual_right=self.residual_right,
            intervention_bias=self.intervention_bias,
            intervention_state=self.intervention_state,
            animal_intervention_residual=self.animal_intervention_residual,
            animal_shared_intervention_gain=self.animal_shared_intervention_gain,
            behavior_shared=self.behavior_shared,
            behavior_residual=self.behavior_residual,
            neural_maps=maps,
            neural_biases=biases,
            neural_mask=mask,
            neuron_counts=self.neuron_counts,
            stability_bound=self.stability_bound,
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> TeacherGroundTruth:
        """Load a :meth:`save` artifact with object loading disabled."""

        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"]))
            counts = np.asarray(archive["neuron_counts"], dtype=np.int32)
            padded_maps = np.asarray(archive["neural_maps"], dtype=np.float64)
            padded_biases = np.asarray(archive["neural_biases"], dtype=np.float64)
            neural_maps = tuple(
                padded_maps[index, : int(count)].copy() for index, count in enumerate(counts)
            )
            neural_biases = tuple(
                padded_biases[index, : int(count)].copy() for index, count in enumerate(counts)
            )
            config = TeacherConfig.from_mapping(metadata["config"], strict_seed_counts=False)
            stress = StressCondition(**metadata["stress"])
            return cls(
                schema_version=str(metadata["schema_version"]),
                release_name=str(metadata["release_name"]),
                world_id=str(metadata["world_id"]),
                world_seed=int(metadata["world_seed"]),
                seed_partition=metadata["seed_partition"],
                seed_index=int(metadata["seed_index"]),
                config=config,
                stress=stress,
                animal_roles=tuple(metadata["animal_roles"]),
                shared_recurrent=np.asarray(archive["shared_recurrent"], dtype=np.float64),
                shared_bias=np.asarray(archive["shared_bias"], dtype=np.float64),
                task_input_map=np.asarray(archive["task_input_map"], dtype=np.float64),
                residual_left=np.asarray(archive["residual_left"], dtype=np.float64),
                residual_right=np.asarray(archive["residual_right"], dtype=np.float64),
                intervention_bias=np.asarray(archive["intervention_bias"], dtype=np.float64),
                intervention_state=np.asarray(archive["intervention_state"], dtype=np.float64),
                animal_intervention_residual=np.asarray(
                    archive["animal_intervention_residual"], dtype=np.float64
                ),
                animal_shared_intervention_gain=np.asarray(
                    archive["animal_shared_intervention_gain"], dtype=np.float64
                ),
                behavior_shared=np.asarray(archive["behavior_shared"], dtype=np.float64),
                behavior_residual=np.asarray(archive["behavior_residual"], dtype=np.float64),
                neural_maps=neural_maps,
                neural_biases=neural_biases,
                neuron_counts=counts,
                stability_bound=np.asarray(archive["stability_bound"], dtype=np.float64),
            )


@dataclass(frozen=True)
class Trajectory:
    """One observed trajectory and its explicit exogenous variables."""

    latent: FloatArray
    task_input: FloatArray
    intervention: FloatArray
    neural_mean: FloatArray
    neural_counts: IntArray
    behavior_mean: FloatArray
    behavior: FloatArray
    initial_state: FloatArray
    process_innovations: FloatArray
    neural_noise_uniforms: FloatArray
    behavior_innovations: FloatArray


@dataclass(frozen=True)
class NormalTrial:
    trial_id: str
    split: Literal["normal", "normal_fit", "normal_val", "normal_audit"]
    trajectory: Trajectory


@dataclass(frozen=True)
class CounterfactualPair:
    """Intervention/control twins coupled by identical exogenous noise."""

    pair_id: str
    intervention_index: int
    dose: float
    onset_step: int
    offset_step: int
    control: Trajectory
    treated: Trajectory

    def validate_pairing(self) -> None:
        """Raise if the two arms are not a properly isolated counterfactual pair."""

        for name in (
            "initial_state",
            "task_input",
            "process_innovations",
            "neural_noise_uniforms",
            "behavior_innovations",
        ):
            _require(
                np.array_equal(getattr(self.control, name), getattr(self.treated, name)),
                f"counterfactual arms do not share {name}",
            )
        _require(not np.any(self.control.intervention), "control arm contains an intervention")
        treated_action = self.treated.intervention
        active = np.flatnonzero(np.any(treated_action != 0.0, axis=1))
        expected = np.arange(self.onset_step, self.offset_step)
        _require(np.array_equal(active, expected), "treated pulse has incorrect temporal support")
        inactive_channels = np.delete(treated_action, self.intervention_index, axis=1)
        _require(not np.any(inactive_channels), "treated pulse leaks into another intervention")
        # z_t is observed before action a_t affects z_{t+1}; equality therefore
        # includes the onset sample.
        prefix = slice(None, self.onset_step + 1)
        _require(
            np.array_equal(self.control.latent[prefix], self.treated.latent[prefix]),
            "intervention affects latent state before its causal onset",
        )
        _require(
            np.array_equal(self.control.neural_counts[prefix], self.treated.neural_counts[prefix]),
            "intervention affects neural observations before its causal onset",
        )
        _require(
            np.array_equal(self.control.behavior[prefix], self.treated.behavior[prefix]),
            "intervention affects behavior before its causal onset",
        )


@dataclass(frozen=True)
class AnimalDataset:
    animal_index: int
    animal_id: str
    role: AnimalRole
    neuron_count: int
    normal_trials: tuple[NormalTrial, ...]
    counterfactual_pairs: tuple[CounterfactualPair, ...]


@dataclass(frozen=True)
class TeacherDataset:
    """Materialized trials for one procedural world."""

    ground_truth: TeacherGroundTruth
    dataset_seed: int
    animals: tuple[AnimalDataset, ...]

    def validate(self) -> None:
        cfg = self.ground_truth.config
        _require(len(self.animals) == cfg.cohort.n_animals, "wrong number of animals")
        for animal, expected_role, expected_neurons in zip(
            self.animals,
            self.ground_truth.animal_roles,
            self.ground_truth.neuron_counts,
            strict=True,
        ):
            _require(animal.role == expected_role, "animal role does not match ground truth")
            _require(animal.neuron_count == int(expected_neurons), "neuron count mismatch")
            expected_normal = (
                cfg.trials.target_normal_trials
                if animal.role == "target"
                else cfg.trials.donor_normal_trials
            )
            _require(len(animal.normal_trials) == expected_normal, "normal trial count mismatch")
            pairs_per_intervention = (
                cfg.trials.target_pairs_per_intervention
                if animal.role == "target"
                else (self.ground_truth.stress.support or cfg.trials.donor_pairs_per_intervention)
            )
            expected_pairs = (
                cfg.intervention.n_interventions
                * len(cfg.intervention.doses)
                * pairs_per_intervention
            )
            _require(
                len(animal.counterfactual_pairs) == expected_pairs,
                "counterfactual pair count mismatch",
            )
            for normal in animal.normal_trials:
                _validate_trajectory_shapes(normal.trajectory, cfg, animal.neuron_count)
                _require(
                    not np.any(normal.trajectory.intervention),
                    "normal trial contains an intervention",
                )
            for pair in animal.counterfactual_pairs:
                _validate_trajectory_shapes(pair.control, cfg, animal.neuron_count)
                _validate_trajectory_shapes(pair.treated, cfg, animal.neuron_count)
                pair.validate_pairing()

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "world_id": self.ground_truth.world_id,
            "world_seed": self.ground_truth.world_seed,
            "seed_partition": self.ground_truth.seed_partition,
            "dataset_seed": self.dataset_seed,
            "seed_material_public": True,
            "prospective_seed_secrecy": False,
            "evaluation_role": (
                "post_freeze_deterministic_procedural_audit"
                if self.ground_truth.seed_partition == "locked"
                else "method_development"
            ),
            "eligible_for_biological_headline_conjunction": False,
            "teacher_config_sha256": teacher_config_sha256(self.ground_truth.config),
            "stress": _jsonable(asdict(self.ground_truth.stress)),
            "n_animals": len(self.animals),
            "n_train_donors": sum(a.role == "train_donor" for a in self.animals),
            "n_validation_donors": sum(a.role == "validation_donor" for a in self.animals),
            "n_targets": sum(a.role == "target" for a in self.animals),
            "neuron_counts": [animal.neuron_count for animal in self.animals],
            "normal_trials": sum(len(a.normal_trials) for a in self.animals),
            "counterfactual_pairs": sum(len(a.counterfactual_pairs) for a in self.animals),
            "neural_noise_model": self.ground_truth.config.observations.neural_noise_model,
            "nb_dispersion": self.ground_truth.config.observations.nb_dispersion,
        }

    def to_padded_arrays(
        self, *, include_latents: bool = True, include_noise: bool = True
    ) -> dict[str, npt.NDArray[Any]]:
        """Convert variable-population objects to deterministic padded arrays.

        Trial and neuron masks make every padding location explicit.  This format
        is convenient for PyTorch/JAX loaders and contains no object arrays.
        """

        cfg = self.ground_truth.config
        n_animals = len(self.animals)
        steps = cfg.trials.steps
        latent_dim = cfg.dynamics.latent_dim
        input_dim = cfg.dynamics.task_input_dim
        behavior_dim = cfg.observations.behavior_dim
        n_interventions = cfg.intervention.n_interventions
        max_neurons = self.ground_truth.max_neurons
        max_normal = max(len(animal.normal_trials) for animal in self.animals)
        max_pairs = max(len(animal.counterfactual_pairs) for animal in self.animals)

        arrays: dict[str, npt.NDArray[Any]] = {
            "animal_role": np.asarray(
                [ROLE_CODES[animal.role] for animal in self.animals], dtype=np.int8
            ),
            "neuron_counts": self.ground_truth.neuron_counts.astype(np.int32, copy=True),
            "neuron_mask": np.arange(max_neurons)[None, :]
            < self.ground_truth.neuron_counts[:, None],
            "normal_trial_mask": np.zeros((n_animals, max_normal), dtype=np.bool_),
            "normal_split": np.full((n_animals, max_normal), -1, dtype=np.int8),
            "normal_task_input": np.zeros(
                (n_animals, max_normal, steps, input_dim), dtype=np.float32
            ),
            "normal_neural_counts": np.full(
                (n_animals, max_normal, steps, max_neurons), -1, dtype=np.int32
            ),
            "normal_behavior": np.full(
                (n_animals, max_normal, steps, behavior_dim), np.nan, dtype=np.float32
            ),
            "pair_mask": np.zeros((n_animals, max_pairs), dtype=np.bool_),
            "pair_intervention_index": np.full((n_animals, max_pairs), -1, dtype=np.int16),
            "pair_dose": np.full((n_animals, max_pairs), np.nan, dtype=np.float32),
            "pair_onset": np.full((n_animals, max_pairs), -1, dtype=np.int16),
            "pair_offset": np.full((n_animals, max_pairs), -1, dtype=np.int16),
            "pair_task_input": np.zeros((n_animals, max_pairs, steps, input_dim), dtype=np.float32),
            "pair_intervention": np.zeros(
                (n_animals, max_pairs, steps, n_interventions), dtype=np.float32
            ),
            "pair_control_neural_counts": np.full(
                (n_animals, max_pairs, steps, max_neurons), -1, dtype=np.int32
            ),
            "pair_treated_neural_counts": np.full(
                (n_animals, max_pairs, steps, max_neurons), -1, dtype=np.int32
            ),
            "pair_control_behavior": np.full(
                (n_animals, max_pairs, steps, behavior_dim), np.nan, dtype=np.float32
            ),
            "pair_treated_behavior": np.full(
                (n_animals, max_pairs, steps, behavior_dim), np.nan, dtype=np.float32
            ),
        }
        if include_latents:
            arrays.update(
                {
                    "normal_latent": np.full(
                        (n_animals, max_normal, steps, latent_dim),
                        np.nan,
                        dtype=np.float32,
                    ),
                    "pair_control_latent": np.full(
                        (n_animals, max_pairs, steps, latent_dim),
                        np.nan,
                        dtype=np.float32,
                    ),
                    "pair_treated_latent": np.full(
                        (n_animals, max_pairs, steps, latent_dim),
                        np.nan,
                        dtype=np.float32,
                    ),
                }
            )
        if include_noise:
            arrays.update(
                {
                    "pair_initial_state": np.full(
                        (n_animals, max_pairs, latent_dim), np.nan, dtype=np.float32
                    ),
                    "pair_process_innovations": np.full(
                        (n_animals, max_pairs, steps - 1, latent_dim),
                        np.nan,
                        dtype=np.float32,
                    ),
                    "pair_neural_noise_uniforms": np.full(
                        (n_animals, max_pairs, steps, max_neurons),
                        np.nan,
                        dtype=np.float32,
                    ),
                    "pair_behavior_innovations": np.full(
                        (n_animals, max_pairs, steps, behavior_dim),
                        np.nan,
                        dtype=np.float32,
                    ),
                }
            )

        for animal_index, animal in enumerate(self.animals):
            n = animal.neuron_count
            for trial_index, normal in enumerate(animal.normal_trials):
                tr = normal.trajectory
                arrays["normal_trial_mask"][animal_index, trial_index] = True
                arrays["normal_split"][animal_index, trial_index] = NORMAL_SPLIT_CODES[normal.split]
                arrays["normal_task_input"][animal_index, trial_index] = tr.task_input
                arrays["normal_neural_counts"][animal_index, trial_index, :, :n] = tr.neural_counts
                arrays["normal_behavior"][animal_index, trial_index] = tr.behavior
                if include_latents:
                    arrays["normal_latent"][animal_index, trial_index] = tr.latent
            for pair_index, pair in enumerate(animal.counterfactual_pairs):
                control = pair.control
                treated = pair.treated
                arrays["pair_mask"][animal_index, pair_index] = True
                arrays["pair_intervention_index"][animal_index, pair_index] = (
                    pair.intervention_index
                )
                arrays["pair_dose"][animal_index, pair_index] = pair.dose
                arrays["pair_onset"][animal_index, pair_index] = pair.onset_step
                arrays["pair_offset"][animal_index, pair_index] = pair.offset_step
                arrays["pair_task_input"][animal_index, pair_index] = treated.task_input
                arrays["pair_intervention"][animal_index, pair_index] = treated.intervention
                arrays["pair_control_neural_counts"][animal_index, pair_index, :, :n] = (
                    control.neural_counts
                )
                arrays["pair_treated_neural_counts"][animal_index, pair_index, :, :n] = (
                    treated.neural_counts
                )
                arrays["pair_control_behavior"][animal_index, pair_index] = control.behavior
                arrays["pair_treated_behavior"][animal_index, pair_index] = treated.behavior
                if include_latents:
                    arrays["pair_control_latent"][animal_index, pair_index] = control.latent
                    arrays["pair_treated_latent"][animal_index, pair_index] = treated.latent
                if include_noise:
                    arrays["pair_initial_state"][animal_index, pair_index] = treated.initial_state
                    arrays["pair_process_innovations"][animal_index, pair_index] = (
                        treated.process_innovations
                    )
                    arrays["pair_neural_noise_uniforms"][animal_index, pair_index, :, :n] = (
                        treated.neural_noise_uniforms
                    )
                    arrays["pair_behavior_innovations"][animal_index, pair_index] = (
                        treated.behavior_innovations
                    )
        return arrays

    def save(
        self,
        path: str | Path,
        *,
        include_latents: bool = True,
        include_noise: bool = True,
    ) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays = self.to_padded_arrays(include_latents=include_latents, include_noise=include_noise)
        arrays["metadata_json"] = np.asarray(_canonical_json(self.summary()))
        np.savez_compressed(destination, **arrays)
        return destination


def _validate_trajectory_shapes(
    trajectory: Trajectory, cfg: TeacherConfig, neuron_count: int
) -> None:
    steps = cfg.trials.steps
    latent_dim = cfg.dynamics.latent_dim
    input_dim = cfg.dynamics.task_input_dim
    n_interventions = cfg.intervention.n_interventions
    behavior_dim = cfg.observations.behavior_dim
    expected = {
        "latent": (steps, latent_dim),
        "task_input": (steps, input_dim),
        "intervention": (steps, n_interventions),
        "neural_mean": (steps, neuron_count),
        "neural_counts": (steps, neuron_count),
        "behavior_mean": (steps, behavior_dim),
        "behavior": (steps, behavior_dim),
        "initial_state": (latent_dim,),
        "process_innovations": (steps - 1, latent_dim),
        "neural_noise_uniforms": (steps, neuron_count),
        "behavior_innovations": (steps, behavior_dim),
    }
    for name, shape in expected.items():
        _require(getattr(trajectory, name).shape == shape, f"{name} has the wrong shape")
    _require(
        np.issubdtype(trajectory.neural_counts.dtype, np.integer),
        "neural counts must be integer-valued",
    )
    _require(np.all(trajectory.neural_counts >= 0), "neural counts cannot be negative")


@dataclass(frozen=True)
class TeacherWorld:
    """A fixed teacher operator capable of drawing deterministic trials."""

    ground_truth: TeacherGroundTruth

    @property
    def config(self) -> TeacherConfig:
        return self.ground_truth.config

    def _normal_split_schedule(self, role: AnimalRole) -> tuple[str, ...]:
        trials = self.config.trials
        if role != "target":
            return ("normal",) * trials.donor_normal_trials
        return (
            *("normal_fit" for _ in range(trials.target_normal_fit_trials)),
            *("normal_val" for _ in range(trials.target_normal_val_trials)),
            *("normal_audit" for _ in range(trials.target_normal_audit_trials)),
        )

    def _draw_exogenous(
        self,
        dataset_seed: int,
        animal_index: int,
        trial_address: Sequence[object],
        *,
        narrow_coverage: bool,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        cfg = self.config
        steps = cfg.trials.steps
        d = cfg.dynamics.latent_dim
        input_dim = cfg.dynamics.task_input_dim
        behavior_dim = cfg.observations.behavior_dim
        n = int(self.ground_truth.neuron_counts[animal_index])
        generator = _rng(dataset_seed, "trial", animal_index, *trial_address)

        initial_scale = 0.18 if narrow_coverage else 0.45
        initial_state = generator.normal(0.0, initial_scale, size=d)
        process = generator.normal(size=(steps - 1, d))
        task_noise = generator.normal(size=(steps, input_dim))
        task_input = np.zeros((steps, input_dim), dtype=np.float64)
        autoregressive = 0.92
        innovation_scale = math.sqrt(1.0 - autoregressive**2)
        task_input[0] = task_noise[0]
        for step in range(1, steps):
            task_input[step] = (
                autoregressive * task_input[step - 1] + innovation_scale * task_noise[step]
            )
        task_input *= cfg.dynamics.task_input_scale
        if narrow_coverage:
            task_input *= 0.20
            if input_dim > 1:
                task_input[:, (input_dim + 1) // 2 :] = 0.0

        # Open-interval uniforms support exact inverse-CDF coupling.
        eps = np.finfo(np.float64).eps
        neural_uniforms = np.clip(generator.random((steps, n)), eps, 1.0 - eps)
        behavior_innovations = generator.normal(size=(steps, behavior_dim))
        return (
            initial_state,
            process,
            task_input,
            neural_uniforms,
            behavior_innovations,
        )

    def _simulate_latent(
        self,
        animal_index: int,
        initial_state: FloatArray,
        process_innovations: FloatArray,
        task_input: FloatArray,
        intervention: FloatArray,
    ) -> FloatArray:
        cfg = self.config
        truth = self.ground_truth
        steps = cfg.trials.steps
        dt = cfg.dynamics.dt
        latent = np.empty((steps, cfg.dynamics.latent_dim), dtype=np.float64)
        latent[0] = initial_state
        for step in range(steps - 1):
            state = latent[step]
            shared_drive = (
                truth.shared_recurrent @ state
                + truth.task_input_map @ task_input[step]
                + truth.shared_bias
            )
            residual = truth.residual_left[animal_index] @ np.tanh(
                truth.residual_right[animal_index] @ state
            )
            intervention_drive = np.zeros(cfg.dynamics.latent_dim, dtype=np.float64)
            for intervention_index, amplitude in enumerate(intervention[step]):
                if amplitude:
                    intervention_drive += amplitude * truth.effective_intervention_field(
                        animal_index, intervention_index, state
                    )
            drift = -state + np.tanh(shared_drive) + residual + intervention_drive
            latent[step + 1] = (
                state
                + dt * drift
                + math.sqrt(dt) * cfg.dynamics.process_noise_std * process_innovations[step]
            )
        return latent

    def _observe(
        self,
        animal_index: int,
        latent: FloatArray,
        neural_uniforms: FloatArray,
        behavior_innovations: FloatArray,
    ) -> tuple[FloatArray, IntArray, FloatArray, FloatArray]:
        cfg = self.config
        truth = self.ground_truth
        logits = (
            latent @ truth.neural_maps[animal_index].T + truth.neural_biases[animal_index][None, :]
        )
        neural_mean = _softplus(logits)
        if cfg.observations.neural_noise_model == "negative_binomial":
            size = cfg.observations.nb_dispersion
            probability = size / (size + neural_mean)
            counts = stats.nbinom.ppf(neural_uniforms, size, probability)
        else:
            counts = stats.poisson.ppf(neural_uniforms, neural_mean)
        neural_counts = np.asarray(counts, dtype=np.int32)

        behavior_map = truth.behavior_shared + truth.behavior_residual[animal_index]
        behavior_mean = latent @ behavior_map.T
        behavior = behavior_mean + cfg.observations.behavior_noise_std * behavior_innovations
        return neural_mean, neural_counts, behavior_mean, behavior

    def _trajectory(
        self,
        animal_index: int,
        intervention: FloatArray,
        exogenous: tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray],
    ) -> Trajectory:
        (
            initial_state,
            process_innovations,
            task_input,
            neural_uniforms,
            behavior_innovations,
        ) = exogenous
        latent = self._simulate_latent(
            animal_index,
            initial_state,
            process_innovations,
            task_input,
            intervention,
        )
        neural_mean, neural_counts, behavior_mean, behavior = self._observe(
            animal_index, latent, neural_uniforms, behavior_innovations
        )
        # Copies prevent a caller from mutating one twin through a shared array.
        return Trajectory(
            latent=latent,
            task_input=task_input.copy(),
            intervention=intervention.copy(),
            neural_mean=neural_mean,
            neural_counts=neural_counts,
            behavior_mean=behavior_mean,
            behavior=behavior,
            initial_state=initial_state.copy(),
            process_innovations=process_innovations.copy(),
            neural_noise_uniforms=neural_uniforms.copy(),
            behavior_innovations=behavior_innovations.copy(),
        )

    def generate_dataset(self, dataset_seed: int | None = None) -> TeacherDataset:
        """Materialize the complete donor/target protocol for this world."""

        cfg = self.config
        truth = self.ground_truth
        if dataset_seed is None:
            dataset_seed = _addressed_seed(truth.world_seed, "dataset")
        animals: list[AnimalDataset] = []
        zero_action = np.zeros(
            (cfg.trials.steps, cfg.intervention.n_interventions), dtype=np.float64
        )
        for animal_index, role in enumerate(truth.animal_roles):
            normal_trials: list[NormalTrial] = []
            for trial_index, split in enumerate(self._normal_split_schedule(role)):
                narrow = (
                    role == "target"
                    and truth.stress.state_coverage == "narrow"
                    and split in {"normal_fit", "normal_val"}
                )
                address = ("normal", split, trial_index)
                exogenous = self._draw_exogenous(
                    dataset_seed,
                    animal_index,
                    address,
                    narrow_coverage=narrow,
                )
                normal_trials.append(
                    NormalTrial(
                        trial_id=f"a{animal_index:02d}-normal-{split}-{trial_index:03d}",
                        split=split,  # type: ignore[arg-type]
                        trajectory=self._trajectory(animal_index, zero_action, exogenous),
                    )
                )

            pairs: list[CounterfactualPair] = []
            pair_support = (
                cfg.trials.target_pairs_per_intervention
                if role == "target"
                else truth.stress.support or cfg.trials.donor_pairs_per_intervention
            )
            for intervention_index in range(cfg.intervention.n_interventions):
                for dose_index, dose in enumerate(cfg.intervention.doses):
                    for replicate in range(pair_support):
                        address = (
                            "pair",
                            intervention_index,
                            dose_index,
                            replicate,
                        )
                        exogenous = self._draw_exogenous(
                            dataset_seed,
                            animal_index,
                            address,
                            narrow_coverage=False,
                        )
                        treated_action = zero_action.copy()
                        treated_action[
                            cfg.intervention.onset_step : cfg.intervention.offset_step,
                            intervention_index,
                        ] = dose
                        control = self._trajectory(animal_index, zero_action, exogenous)
                        treated = self._trajectory(animal_index, treated_action, exogenous)
                        pair = CounterfactualPair(
                            pair_id=(
                                f"a{animal_index:02d}-i{intervention_index:02d}-"
                                f"d{dose_index:02d}-r{replicate:03d}"
                            ),
                            intervention_index=intervention_index,
                            dose=float(dose),
                            onset_step=cfg.intervention.onset_step,
                            offset_step=cfg.intervention.offset_step,
                            control=control,
                            treated=treated,
                        )
                        pair.validate_pairing()
                        pairs.append(pair)
            animals.append(
                AnimalDataset(
                    animal_index=animal_index,
                    animal_id=f"animal-{animal_index:02d}",
                    role=role,
                    neuron_count=int(truth.neuron_counts[animal_index]),
                    normal_trials=tuple(normal_trials),
                    counterfactual_pairs=tuple(pairs),
                )
            )
        dataset = TeacherDataset(
            ground_truth=truth, dataset_seed=int(dataset_seed), animals=tuple(animals)
        )
        dataset.validate()
        return dataset


def _draw_rank_two_residual(
    generator: np.random.Generator, latent_dim: int, operator_norm: float
) -> tuple[FloatArray, FloatArray]:
    rank = 2
    left = generator.normal(size=(latent_dim, rank))
    right = generator.normal(size=(rank, latent_dim))
    current_norm = float(np.linalg.svd(left @ right, compute_uv=False)[0])
    if operator_norm == 0.0:
        left = np.zeros_like(left)
    else:
        left *= operator_norm / current_norm
    return left, right


def _draw_neuron_counts(
    config: TeacherConfig, stress: StressCondition, world_seed: int
) -> IntArray:
    counts = np.empty(config.cohort.n_animals, dtype=np.int32)
    for animal_index, role in enumerate(config.cohort.roles):
        if role == "target" and stress.target_neurons is not None:
            count = stress.target_neurons
        else:
            generator = _rng(world_seed, "animal", animal_index, "neuron_count")
            count = int(
                generator.integers(
                    config.observations.neurons_min,
                    config.observations.neurons_max + 1,
                )
            )
        counts[animal_index] = count
    # "Variable maps/counts" is an invariant when the interval is non-degenerate.
    if (
        stress.target_neurons is None
        and config.observations.neurons_max > config.observations.neurons_min
        and np.all(counts == counts[0])
    ):
        counts[-1] = (
            config.observations.neurons_min
            if counts[0] != config.observations.neurons_min
            else config.observations.neurons_max
        )
    return counts


def teacher_config_sha256(config: TeacherConfig) -> str:
    """Return the canonical scientific-configuration fingerprint."""

    return hashlib.sha256(_canonical_json(config.to_mapping()).encode()).hexdigest()


def validate_locked_teacher_config(
    config: TeacherConfig,
    *,
    canonical_path: str | Path = CANONICAL_TEACHER_CONFIG_PATH,
) -> str:
    """Require exact equality with the tracked post-freeze configuration.

    The returned digest is suitable for manifests.  Content, rather than a
    caller-controlled path string, is compared so a byte-for-byte equivalent
    copy is harmless while an untracked modified YAML cannot label a world as
    locked.
    """

    canonical = TeacherConfig.from_yaml(canonical_path)
    expected = teacher_config_sha256(canonical)
    observed = teacher_config_sha256(config)
    if observed != expected or config.to_mapping() != canonical.to_mapping():
        raise ProtocolViolation(
            "post-freeze procedural teacher worlds require the exact tracked canonical "
            f"TeacherConfig; expected sha256={expected}, observed={observed}"
        )
    return observed


def generate_teacher_world(
    config: TeacherConfig,
    *,
    partition: SeedPartition = "development",
    seed_index: int = 0,
    world_seed: int | None = None,
    stress: StressCondition | None = None,
    strict_seed_counts: bool = True,
    freeze_attestation: FreezeAttestation | None = None,
) -> TeacherWorld:
    """Generate a deterministic causal world and all of its true matrices.

    The historically named ``locked`` worlds are a closed, public-seed
    procedural cohort: their seeds must come from the configured partition by
    index and their stress condition must equal the frozen default.  The name
    denotes a post-freeze code boundary, not prospective seed secrecy.
    Explicit seeds and stress sweeps remain development-only.
    """

    config.validate(strict_seed_counts=strict_seed_counts)
    _require(partition in {"development", "locked"}, "invalid seed partition")
    frozen_stress = StressCondition(rho=config.dynamics.residual_ratio)
    if partition == "locked" and world_seed is not None:
        raise ProtocolViolation(
            "post-freeze procedural teacher worlds require a configured seed selected by "
            "seed_index; explicit world_seed is development-only"
        )
    if partition == "locked" and stress is not None and stress != frozen_stress:
        raise ProtocolViolation(
            "post-freeze procedural teacher worlds require the frozen default stress; "
            "stress overrides are development-only"
        )
    if partition == "locked":
        validate_locked_teacher_config(config)
    stress = frozen_stress if stress is None else stress
    stress.validate()
    if partition == "locked" and freeze_attestation is None:
        raise ProtocolViolation(
            "post-freeze procedural teacher worlds require a validated "
            "pre-outcome freeze attestation"
        )
    if partition == "locked":
        validated_freeze = attest_preoutcome_freeze(repository=Path(__file__).resolve().parents[2])
        if freeze_attestation != validated_freeze:
            raise ProtocolViolation("supplied teacher freeze attestation is not current")
    seeds = config.seeds.development if partition == "development" else config.seeds.locked
    if world_seed is None:
        _require(0 <= seed_index < len(seeds), "seed_index is out of range")
        world_seed = int(seeds[seed_index])
    else:
        world_seed = int(world_seed)

    d = config.dynamics.latent_dim
    rank = config.dynamics.residual_rank
    n_animals = config.cohort.n_animals
    n_interventions = config.intervention.n_interventions
    behavior_dim = config.observations.behavior_dim

    shared_recurrent = _spectral_normalize(
        _rng(world_seed, "shared", "recurrent").normal(size=(d, d)),
        config.dynamics.shared_operator_norm,
    )
    shared_bias = _rng(world_seed, "shared", "bias").normal(0.0, 0.08, size=d)
    task_input_map = _rng(world_seed, "shared", "task_map").normal(
        0.0,
        0.30 / math.sqrt(config.dynamics.task_input_dim),
        size=(d, config.dynamics.task_input_dim),
    )

    residual_left = np.empty((n_animals, d, rank), dtype=np.float64)
    residual_right = np.empty((n_animals, rank, d), dtype=np.float64)
    residual_norm = stress.rho * config.dynamics.shared_operator_norm
    for animal_index in range(n_animals):
        left, right = _draw_rank_two_residual(
            _rng(world_seed, "animal", animal_index, "dynamics_residual"),
            d,
            residual_norm,
        )
        residual_left[animal_index] = left
        residual_right[animal_index] = right

    intervention_bias = np.empty((n_interventions, d), dtype=np.float64)
    intervention_state = np.empty((n_interventions, d, d), dtype=np.float64)
    for intervention_index in range(n_interventions):
        generator = _rng(world_seed, "intervention", intervention_index, "shared")
        bias = generator.normal(size=d)
        intervention_bias[intervention_index] = (
            bias / np.linalg.norm(bias) * config.intervention.shared_bias_norm
        )
        state_left, state_right = _draw_rank_two_residual(
            generator,
            d,
            config.intervention.shared_state_norm,
        )
        intervention_state[intervention_index] = state_left @ state_right

    animal_intervention_residual = np.empty((n_animals, n_interventions, d), dtype=np.float64)
    animal_shared_gain = np.ones((n_animals, n_interventions), dtype=np.float64)
    private_norm = config.intervention.animal_residual_ratio * config.intervention.shared_bias_norm
    for animal_index in range(n_animals):
        for intervention_index in range(n_interventions):
            private = _rng(
                world_seed,
                "animal",
                animal_index,
                "intervention_residual",
                intervention_index,
            ).normal(size=d)
            private /= np.linalg.norm(private)
            # As eta falls, a conserved field is progressively replaced by a
            # private component while retaining a small residual at eta=1.
            magnitude = private_norm + (1.0 - stress.eta) * config.intervention.shared_bias_norm
            animal_intervention_residual[animal_index, intervention_index] = magnitude * private

    if stress.impossibility == "independent_target_direction":
        for animal_index, role in enumerate(config.cohort.roles):
            if role != "target":
                continue
            for intervention_index in range(n_interventions):
                private = _rng(
                    world_seed,
                    "impossibility",
                    "target_private_direction",
                    animal_index,
                    intervention_index,
                ).normal(size=d)
                private /= np.linalg.norm(private)
                animal_shared_gain[animal_index, intervention_index] = 0.0
                animal_intervention_residual[animal_index, intervention_index] = (
                    stress.impossibility_variant * config.intervention.shared_bias_norm * private
                )

    behavior_shared = _rng(world_seed, "shared", "behavior").normal(size=(behavior_dim, d))
    behavior_shared = _spectral_normalize(behavior_shared, 1.0)
    behavior_residual = np.empty((n_animals, behavior_dim, d), dtype=np.float64)
    for animal_index in range(n_animals):
        raw = _rng(world_seed, "animal", animal_index, "behavior_residual").normal(
            size=(behavior_dim, d)
        )
        behavior_residual[animal_index] = _spectral_normalize(
            raw, config.observations.behavior_residual_ratio
        )

    neuron_counts = _draw_neuron_counts(config, stress, world_seed)
    neural_maps: list[FloatArray] = []
    neural_biases: list[FloatArray] = []
    for animal_index, count in enumerate(neuron_counts):
        generator = _rng(world_seed, "animal", animal_index, "neural_observation")
        observation_map = generator.normal(size=(int(count), d))
        row_norms = np.linalg.norm(observation_map, axis=1, keepdims=True)
        observation_map = config.observations.neural_map_scale * observation_map / row_norms
        neural_maps.append(observation_map)
        neural_biases.append(
            generator.normal(
                config.observations.neural_bias_mean,
                config.observations.neural_bias_std,
                size=int(count),
            )
        )

    # Upper bound on the normal discrete-time Jacobian norm:
    # ||(1-dt)I + dt D_t W + dt U D'_t V||.
    stability_bound = (
        1.0
        - config.dynamics.dt
        + config.dynamics.dt * (config.dynamics.shared_operator_norm + residual_norm)
    ) * np.ones(n_animals, dtype=np.float64)
    _require(
        bool(np.all(stability_bound < 1.0)),
        "normal dynamics are not contractive under the certified norm bound",
    )

    identity_payload = {
        "partition": partition,
        "seed_index": seed_index,
        "seed": world_seed,
        "config": config.to_mapping(),
        "stress": asdict(stress),
    }
    digest = hashlib.sha256(_canonical_json(identity_payload).encode()).hexdigest()[:12]
    world_id = f"{config.release_name}-{partition}-{seed_index:02d}-{digest}"
    truth = TeacherGroundTruth(
        schema_version=SCHEMA_VERSION,
        release_name=config.release_name,
        world_id=world_id,
        world_seed=world_seed,
        seed_partition=partition,
        seed_index=seed_index,
        config=config,
        stress=stress,
        animal_roles=config.cohort.roles,
        shared_recurrent=shared_recurrent,
        shared_bias=shared_bias,
        task_input_map=task_input_map,
        residual_left=residual_left,
        residual_right=residual_right,
        intervention_bias=intervention_bias,
        intervention_state=intervention_state,
        animal_intervention_residual=animal_intervention_residual,
        animal_shared_intervention_gain=animal_shared_gain,
        behavior_shared=behavior_shared,
        behavior_residual=behavior_residual,
        neural_maps=tuple(neural_maps),
        neural_biases=tuple(neural_biases),
        neuron_counts=neuron_counts,
        stability_bound=stability_bound,
    )
    return TeacherWorld(ground_truth=truth)


def load_teacher_config(path: str | Path) -> TeacherConfig:
    """Convenience alias used by command-line scripts."""

    return TeacherConfig.from_yaml(path)


def save_teacher_release(
    world: TeacherWorld,
    output_dir: str | Path,
    *,
    dataset_seed: int | None = None,
    include_latents: bool = True,
    include_noise: bool = True,
) -> dict[str, Path]:
    """Generate and serialize a dataset, exact ground truth, and JSON manifest."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dataset = world.generate_dataset(dataset_seed=dataset_seed)
    dataset_path = dataset.save(
        destination / "dataset.npz",
        include_latents=include_latents,
        include_noise=include_noise,
    )
    ground_truth_path = world.ground_truth.save(destination / "ground_truth.npz")
    manifest = {
        **dataset.summary(),
        "ground_truth": ground_truth_path.name,
        "dataset": dataset_path.name,
        "contains_latents": include_latents,
        "contains_exogenous_noise": include_noise,
        "ground_truth_metadata": world.ground_truth.metadata(),
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "dataset": dataset_path,
        "ground_truth": ground_truth_path,
        "manifest": manifest_path,
    }


__all__ = [
    "AnimalDataset",
    "CANONICAL_TEACHER_CONFIG_PATH",
    "CohortConfig",
    "CounterfactualPair",
    "DynamicsConfig",
    "InterventionConfig",
    "NormalTrial",
    "ObservationConfig",
    "SCHEMA_VERSION",
    "SeedConfig",
    "StressCondition",
    "TeacherConfig",
    "TeacherDataset",
    "TeacherGroundTruth",
    "TeacherWorld",
    "Trajectory",
    "TrialConfig",
    "generate_teacher_world",
    "load_teacher_config",
    "save_teacher_release",
    "teacher_config_sha256",
    "validate_locked_teacher_config",
]
