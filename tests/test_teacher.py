from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import cadence.teacher as teacher_module
from cadence.protocol import ProtocolViolation
from cadence.teacher import (
    CohortConfig,
    InterventionConfig,
    ObservationConfig,
    StressCondition,
    TeacherGroundTruth,
    TrialConfig,
    generate_teacher_world,
    load_teacher_config,
    save_teacher_release,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_teacher_release_cli() -> ModuleType:
    path = ROOT / "scripts" / "generate_teacher_release.py"
    spec = importlib.util.spec_from_file_location("cadence_test_generate_teacher_release", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def small_config():
    locked = load_teacher_config(ROOT / "configs" / "teacher.yaml")
    return replace(
        locked,
        release_name="teacher-test",
        cohort=CohortConfig(train_donors=2, validation_donors=1, targets=1),
        dynamics=replace(
            locked.dynamics,
            latent_dim=4,
            task_input_dim=2,
            process_noise_std=0.01,
        ),
        intervention=InterventionConfig(
            n_interventions=2,
            onset_step=5,
            offset_step=9,
            doses=(1.0,),
            shared_bias_norm=0.6,
            shared_state_norm=0.2,
            animal_residual_ratio=0.05,
        ),
        observations=ObservationConfig(
            neurons_min=6,
            neurons_max=10,
            behavior_dim=3,
            neural_bias_mean=1.0,
            neural_bias_std=0.25,
            neural_map_scale=0.75,
            neural_noise_model="negative_binomial",
            nb_dispersion=20.0,
            behavior_noise_std=0.04,
            behavior_residual_ratio=0.1,
        ),
        trials=TrialConfig(
            steps=14,
            donor_normal_trials=2,
            donor_pairs_per_intervention=2,
            target_normal_fit_trials=2,
            target_normal_val_trials=1,
            target_normal_audit_trials=1,
            target_pairs_per_intervention=2,
        ),
    )


def _assert_truth_equal(left, right) -> None:
    names = (
        "shared_recurrent",
        "shared_bias",
        "task_input_map",
        "residual_left",
        "residual_right",
        "intervention_bias",
        "intervention_state",
        "animal_intervention_residual",
        "animal_shared_intervention_gain",
        "behavior_shared",
        "behavior_residual",
        "neuron_counts",
        "stability_bound",
    )
    for name in names:
        assert np.array_equal(getattr(left, name), getattr(right, name)), name
    for left_map, right_map in zip(left.neural_maps, right.neural_maps, strict=True):
        assert np.array_equal(left_map, right_map)
    for left_bias, right_bias in zip(left.neural_biases, right.neural_biases, strict=True):
        assert np.array_equal(left_bias, right_bias)


def _assert_trajectory_equal(left, right) -> None:
    for name in (
        "latent",
        "task_input",
        "intervention",
        "neural_mean",
        "neural_counts",
        "behavior_mean",
        "behavior",
        "initial_state",
        "process_innovations",
        "neural_noise_uniforms",
        "behavior_innovations",
    ):
        assert np.array_equal(getattr(left, name), getattr(right, name)), name


def test_locked_yaml_matches_preregistered_main_regime() -> None:
    config = load_teacher_config(ROOT / "configs" / "teacher.yaml")

    assert config.dynamics.latent_dim == 8
    assert config.dynamics.task_input_dim == 4
    assert config.dynamics.dt == 0.05
    assert config.dynamics.residual_rank == 2
    assert config.dynamics.residual_ratio == 0.1
    assert config.trials.steps == 100
    assert (config.intervention.onset_step, config.intervention.offset_step) == (40, 60)
    assert config.intervention.n_interventions == 6
    assert config.intervention.state_rank == 2
    assert (config.observations.neurons_min, config.observations.neurons_max) == (64, 128)
    assert config.observations.neural_noise_model == "negative_binomial"
    assert config.observations.nb_dispersion == 20
    assert config.observations.behavior_dim == 3
    assert config.cohort.roles.count("train_donor") == 10
    assert config.cohort.roles.count("validation_donor") == 2
    assert config.cohort.roles.count("target") == 4
    assert config.trials.donor_normal_trials == 96
    assert config.trials.donor_pairs_per_intervention == 32
    assert config.trials.target_normal_fit_trials == 64
    assert config.trials.target_normal_val_trials == 16
    assert config.trials.target_normal_audit_trials == 32
    assert config.trials.target_pairs_per_intervention == 24
    assert len(config.seeds.development) == 10
    assert len(config.seeds.locked) == 20
    assert set(config.seeds.development).isdisjoint(config.seeds.locked)


def test_locked_world_cannot_be_generated_without_freeze_attestation() -> None:
    canonical = load_teacher_config(ROOT / "configs" / "teacher.yaml")
    with pytest.raises(ProtocolViolation, match="freeze attestation"):
        generate_teacher_world(canonical, partition="locked", seed_index=0)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"world_seed": 123456}, "configured seed"),
        (
            {"stress": StressCondition(eta=0.5, rho=0.1)},
            "frozen default stress",
        ),
    ],
)
def test_locked_world_rejects_cohort_overrides_before_attestation(
    small_config,
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, object],
    message: str,
) -> None:
    def unexpected_attestation(**_kwargs):
        pytest.fail("invalid locked scope reached freeze attestation")

    monkeypatch.setattr(teacher_module, "attest_preoutcome_freeze", unexpected_attestation)
    with pytest.raises(ProtocolViolation, match=message):
        generate_teacher_world(
            small_config,
            partition="locked",
            seed_index=0,
            **override,
        )


def test_locked_world_rejects_noncanonical_config_before_attestation(
    small_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_attestation(**_kwargs):
        pytest.fail("noncanonical locked config reached freeze attestation")

    monkeypatch.setattr(teacher_module, "attest_preoutcome_freeze", unexpected_attestation)
    with pytest.raises(ProtocolViolation, match="exact tracked canonical"):
        generate_teacher_world(
            small_config,
            partition="locked",
            seed_index=0,
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--world-seed", "123456"], "development-only"),
        (["--eta", "0.5"], "stress overrides are development-only"),
        (["--rho", "0.25"], "stress overrides are development-only"),
        (["--target-neurons", "32"], "stress overrides are development-only"),
        (["--support", "8"], "stress overrides are development-only"),
        (["--state-coverage", "narrow"], "stress overrides are development-only"),
        (
            ["--impossibility", "independent_target_direction"],
            "stress overrides are development-only",
        ),
        (
            ["--impossibility-variant", "-1"],
            "stress overrides are development-only",
        ),
        (
            ["--overwrite"],
            "overwrite is forbidden",
        ),
    ],
)
def test_locked_release_cli_rejects_overrides_before_attestation_or_generation(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    message: str,
) -> None:
    def unexpected_call(*_args, **_kwargs):
        pytest.fail("invalid locked CLI scope reached attestation or generation")

    teacher_release_cli = _load_teacher_release_cli()
    monkeypatch.setattr(teacher_release_cli, "attest_preoutcome_freeze", unexpected_call)
    monkeypatch.setattr(teacher_release_cli, "generate_teacher_world", unexpected_call)
    with pytest.raises(SystemExit, match=message):
        teacher_release_cli.main(["--partition", "locked", "--acknowledge-locked", *arguments])


def test_locked_release_cli_rejects_modified_external_config_before_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (ROOT / "configs" / "teacher.yaml").read_text(encoding="utf-8")
    modified = tmp_path / "modified-teacher.yaml"
    modified.write_text(
        source.replace(
            "release_name: cadence-teacher-rnn-v1",
            "release_name: modified-teacher",
        ),
        encoding="utf-8",
    )
    teacher_release_cli = _load_teacher_release_cli()

    def unexpected_call(*_args, **_kwargs):
        pytest.fail("modified locked config reached attestation or generation")

    monkeypatch.setattr(teacher_release_cli, "attest_preoutcome_freeze", unexpected_call)
    monkeypatch.setattr(teacher_release_cli, "generate_teacher_world", unexpected_call)
    with pytest.raises(SystemExit, match="exact tracked canonical"):
        teacher_release_cli.main(
            [
                "--partition",
                "locked",
                "--acknowledge-locked",
                "--config",
                str(modified),
            ]
        )


def test_world_and_dataset_are_bit_reproducible_and_serializable(
    small_config, tmp_path: Path
) -> None:
    left_world = generate_teacher_world(small_config, seed_index=2)
    right_world = generate_teacher_world(small_config, seed_index=2)
    assert all(
        np.linalg.matrix_rank(matrix, tol=1e-10) <= small_config.intervention.state_rank
        for matrix in left_world.ground_truth.intervention_state
    )
    _assert_truth_equal(left_world.ground_truth, right_world.ground_truth)
    assert left_world.ground_truth.world_id == right_world.ground_truth.world_id

    left_data = left_world.generate_dataset()
    right_data = right_world.generate_dataset()
    summary = left_data.summary()
    assert summary["seed_material_public"]
    assert not summary["prospective_seed_secrecy"]
    assert not summary["eligible_for_biological_headline_conjunction"]
    assert summary["evaluation_role"] == "method_development"
    assert len(summary["teacher_config_sha256"]) == 64
    assert left_data.dataset_seed == right_data.dataset_seed
    for left_animal, right_animal in zip(left_data.animals, right_data.animals, strict=True):
        for left_trial, right_trial in zip(
            left_animal.normal_trials, right_animal.normal_trials, strict=True
        ):
            _assert_trajectory_equal(left_trial.trajectory, right_trial.trajectory)
        for left_pair, right_pair in zip(
            left_animal.counterfactual_pairs,
            right_animal.counterfactual_pairs,
            strict=True,
        ):
            _assert_trajectory_equal(left_pair.control, right_pair.control)
            _assert_trajectory_equal(left_pair.treated, right_pair.treated)

    truth_path = left_world.ground_truth.save(tmp_path / "truth.npz")
    restored = TeacherGroundTruth.load(truth_path)
    _assert_truth_equal(left_world.ground_truth, restored)
    assert restored.metadata() == left_world.ground_truth.metadata()


def test_release_serialization_uses_numeric_padded_arrays(small_config, tmp_path: Path) -> None:
    world = generate_teacher_world(small_config)
    paths = save_teacher_release(world, tmp_path / "release")

    assert set(paths) == {"dataset", "ground_truth", "manifest"}
    assert all(path.is_file() for path in paths.values())
    with np.load(paths["dataset"], allow_pickle=False) as archive:
        assert archive["normal_neural_counts"].dtype == np.int32
        assert archive["pair_mask"].dtype == np.bool_
        assert archive["pair_process_innovations"].dtype == np.float32
        assert archive["pair_treated_behavior"].shape[-1] == 3
        assert archive["metadata_json"].ndim == 0


def test_shared_dynamics_are_stable_and_animal_residuals_are_rank_two(
    small_config,
) -> None:
    truth = generate_teacher_world(small_config).ground_truth

    assert np.linalg.svd(truth.shared_recurrent, compute_uv=False)[0] == pytest.approx(
        small_config.dynamics.shared_operator_norm
    )
    assert np.all(truth.stability_bound < 1.0)
    shared_norm = np.linalg.svd(truth.shared_recurrent, compute_uv=False)[0]
    for animal_index in range(truth.n_animals):
        residual = truth.normal_operator_matrix(animal_index)
        assert np.linalg.matrix_rank(residual, tol=1e-10) == 2
        residual_norm = np.linalg.svd(residual, compute_uv=False)[0]
        assert residual_norm / shared_norm == pytest.approx(0.1)


def test_counterfactual_twins_share_noise_and_have_isolated_pulses(
    small_config,
) -> None:
    dataset = generate_teacher_world(small_config).generate_dataset()
    pair = dataset.animals[0].counterfactual_pairs[0]
    pair.validate_pairing()

    control, treated = pair.control, pair.treated
    assert np.array_equal(control.initial_state, treated.initial_state)
    assert np.array_equal(control.process_innovations, treated.process_innovations)
    assert np.array_equal(control.neural_noise_uniforms, treated.neural_noise_uniforms)
    assert np.array_equal(control.behavior_innovations, treated.behavior_innovations)
    assert np.array_equal(control.task_input, treated.task_input)
    assert not control.intervention.any()

    action = treated.intervention
    onset, offset = small_config.intervention.onset_step, small_config.intervention.offset_step
    assert not action[:onset].any()
    assert np.all(action[onset:offset, pair.intervention_index] == pair.dose)
    assert not action[offset:].any()
    other = 1 - pair.intervention_index
    assert not action[:, other].any()

    # a_t enters z_{t+1}, so state/observations are identical through sample t=onset.
    np.testing.assert_array_equal(control.latent[: onset + 1], treated.latent[: onset + 1])
    np.testing.assert_array_equal(
        control.neural_counts[: onset + 1], treated.neural_counts[: onset + 1]
    )
    np.testing.assert_array_equal(control.behavior[: onset + 1], treated.behavior[: onset + 1])
    assert not np.array_equal(control.latent[onset + 1 :], treated.latent[onset + 1 :])


def test_shapes_variable_neurons_nb_counts_and_continuous_behavior(
    small_config,
) -> None:
    dataset = generate_teacher_world(small_config).generate_dataset()
    truth = dataset.ground_truth
    counts = truth.neuron_counts

    assert len(set(counts.tolist())) > 1
    for animal in dataset.animals:
        assert truth.neural_maps[animal.animal_index].shape == (
            animal.neuron_count,
            small_config.dynamics.latent_dim,
        )
        normal = animal.normal_trials[0].trajectory
        assert normal.neural_counts.shape == (
            small_config.trials.steps,
            animal.neuron_count,
        )
        assert np.issubdtype(normal.neural_counts.dtype, np.integer)
        assert np.all(normal.neural_counts >= 0)
        assert np.all(normal.neural_mean > 0.0)
        assert normal.behavior.shape == (
            small_config.trials.steps,
            small_config.observations.behavior_dim,
        )
        assert np.any(normal.behavior != np.round(normal.behavior))

    arrays = dataset.to_padded_arrays()
    max_neurons = int(counts.max())
    assert arrays["normal_neural_counts"].shape[-1] == max_neurons
    assert arrays["pair_treated_neural_counts"].shape[-1] == max_neurons
    assert arrays["normal_latent"].shape[-1] == small_config.dynamics.latent_dim
    for animal_index, neuron_count in enumerate(counts):
        assert arrays["neuron_mask"][animal_index, :neuron_count].all()
        assert not arrays["neuron_mask"][animal_index, neuron_count:].any()


def test_poisson_is_an_explicit_reproducible_observation_option(small_config) -> None:
    poisson_config = replace(
        small_config,
        observations=replace(
            small_config.observations,
            neural_noise_model="poisson",
        ),
    )
    left = generate_teacher_world(poisson_config).generate_dataset()
    right = generate_teacher_world(poisson_config).generate_dataset()
    left_counts = left.animals[0].normal_trials[0].trajectory.neural_counts
    right_counts = right.animals[0].normal_trials[0].trajectory.neural_counts

    assert np.array_equal(left_counts, right_counts)
    assert np.issubdtype(left_counts.dtype, np.integer)
    assert np.all(left_counts >= 0)


def test_impossibility_pair_has_identical_normal_world_and_opposite_hidden_response(
    small_config,
) -> None:
    positive = StressCondition(
        eta=1.0,
        rho=0.1,
        impossibility="independent_target_direction",
        impossibility_variant=1,
    )
    negative = replace(positive, impossibility_variant=-1)
    positive_world = generate_teacher_world(small_config, stress=positive)
    negative_world = generate_teacher_world(small_config, stress=negative)
    positive_truth = positive_world.ground_truth
    negative_truth = negative_world.ground_truth

    # Every normal-data-generating parameter is identical.
    for name in (
        "shared_recurrent",
        "shared_bias",
        "task_input_map",
        "residual_left",
        "residual_right",
        "behavior_shared",
        "behavior_residual",
        "neuron_counts",
    ):
        assert np.array_equal(getattr(positive_truth, name), getattr(negative_truth, name))
    for left, right in zip(positive_truth.neural_maps, negative_truth.neural_maps, strict=True):
        assert np.array_equal(left, right)

    target_index = small_config.cohort.n_animals - 1
    assert not positive_truth.animal_shared_intervention_gain[target_index].any()
    np.testing.assert_allclose(
        positive_truth.animal_intervention_residual[target_index],
        -negative_truth.animal_intervention_residual[target_index],
        atol=0.0,
        rtol=0.0,
    )

    positive_data = positive_world.generate_dataset()
    negative_data = negative_world.generate_dataset()
    for positive_animal, negative_animal in zip(
        positive_data.animals, negative_data.animals, strict=True
    ):
        for left, right in zip(
            positive_animal.normal_trials, negative_animal.normal_trials, strict=True
        ):
            _assert_trajectory_equal(left.trajectory, right.trajectory)
        for left, right in zip(
            positive_animal.counterfactual_pairs,
            negative_animal.counterfactual_pairs,
            strict=True,
        ):
            _assert_trajectory_equal(left.control, right.control)
            if positive_animal.role != "target":
                _assert_trajectory_equal(left.treated, right.treated)

    target_positive = positive_data.animals[target_index].counterfactual_pairs[0]
    target_negative = negative_data.animals[target_index].counterfactual_pairs[0]
    onset = small_config.intervention.onset_step
    np.testing.assert_array_equal(
        target_positive.treated.latent[: onset + 1],
        target_negative.treated.latent[: onset + 1],
    )
    assert not np.array_equal(
        target_positive.treated.latent[onset + 1 :],
        target_negative.treated.latent[onset + 1 :],
    )


def test_stress_axes_override_support_target_neurons_and_state_coverage(
    small_config,
) -> None:
    stress = StressCondition(
        eta=0.5,
        rho=0.25,
        target_neurons=32,
        support=8,
        state_coverage="narrow",
    )
    world = generate_teacher_world(small_config, stress=stress)
    dataset = world.generate_dataset()
    target = dataset.animals[-1]

    assert target.neuron_count == 32
    assert len(dataset.animals[0].counterfactual_pairs) == (
        small_config.intervention.n_interventions * len(small_config.intervention.doses) * 8
    )
    assert len(target.counterfactual_pairs) == (
        small_config.intervention.n_interventions
        * len(small_config.intervention.doses)
        * small_config.trials.target_pairs_per_intervention
    )
    fit_input = target.normal_trials[0].trajectory.task_input
    assert np.all(fit_input[:, 1:] == 0.0)
