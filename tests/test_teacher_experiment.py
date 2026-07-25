from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

import cadence.experiments.teacher as teacher_experiment_module
from cadence.experiments.teacher import (
    TeacherExperimentConfig,
    fit_affine_gauge,
    intervention_sequence_batches,
    make_experiment_config,
    make_profile_teacher_config,
    normal_sequence_batches,
    predict_target_pairs,
    run_teacher_experiment,
    teacher_experiment_scientific_sha256,
    validate_locked_teacher_experiment_config,
)
from cadence.model import HierarchicalControlledSSM
from cadence.protocol import FreezeAttestation
from cadence.teacher import (
    CohortConfig,
    InterventionConfig,
    ObservationConfig,
    TrialConfig,
    generate_teacher_world,
    load_teacher_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_teacher_experiment_cli() -> ModuleType:
    path = ROOT / "scripts" / "run_teacher_experiment.py"
    spec = importlib.util.spec_from_file_location("cadence_test_run_teacher_experiment", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiny_dataset():
    base = load_teacher_config(ROOT / "configs" / "teacher.yaml")
    smoke = make_profile_teacher_config(base, "smoke")
    config = replace(
        smoke,
        release_name="teacher-experiment-test",
        cohort=CohortConfig(train_donors=1, validation_donors=1, targets=1),
        dynamics=replace(
            smoke.dynamics,
            latent_dim=3,
            task_input_dim=1,
            process_noise_std=0.0,
        ),
        intervention=InterventionConfig(
            n_interventions=1,
            onset_step=3,
            offset_step=6,
            doses=(1.0,),
            shared_bias_norm=0.6,
            shared_state_norm=0.2,
            animal_residual_ratio=0.05,
        ),
        observations=ObservationConfig(
            neurons_min=5,
            neurons_max=8,
            behavior_dim=3,
            neural_bias_mean=1.0,
            neural_bias_std=0.25,
            neural_map_scale=0.75,
            neural_noise_model="negative_binomial",
            nb_dispersion=20.0,
            behavior_noise_std=0.02,
            behavior_residual_ratio=0.1,
        ),
        trials=TrialConfig(
            steps=9,
            donor_normal_trials=4,
            donor_pairs_per_intervention=2,
            target_normal_fit_trials=3,
            target_normal_val_trials=2,
            target_normal_audit_trials=1,
            target_pairs_per_intervention=2,
        ),
    )
    return generate_teacher_world(config).generate_dataset()


def _one_epoch_config() -> TeacherExperimentConfig:
    config = make_experiment_config("smoke", device="cpu", learned_methods=("proposed",))
    return replace(
        config,
        hidden_dim=8,
        intervention_rank=2,
        batch_size=4,
        normal_fit=replace(config.normal_fit, max_epochs=1, patience=1),
        intervention_fit=replace(config.intervention_fit, max_epochs=1, patience=1),
        target_fit=replace(config.target_fit, max_epochs=1, patience=1),
    )


def test_sequence_batches_preserve_variable_animal_channels(tiny_dataset) -> None:
    donors = [animal for animal in tiny_dataset.animals if animal.role != "target"]
    assert donors[0].neuron_count != tiny_dataset.animals[-1].neuron_count

    normal = normal_sequence_batches(
        donors[0],
        batch_size=2,
        neural_transform="log1p",
    )
    intervention = intervention_sequence_batches(
        donors[0],
        donors[0].counterfactual_pairs,
        batch_size=2,
        neural_transform="log1p",
    )

    assert normal[0].neural.shape[-1] == donors[0].neuron_count
    assert intervention[0].neural.shape[-1] == donors[0].neuron_count
    assert torch.count_nonzero(normal[0].intervention).item() == 0
    assert torch.count_nonzero(intervention[0].intervention).item() > 0


def test_selection_topology_keeps_validation_donors_out_of_shared_f_fit(
    tiny_dataset,
) -> None:
    config = _one_epoch_config()
    normal_train, normal_validation = teacher_experiment_module._normal_data_for_donors(
        tiny_dataset, config
    )
    intervention_train, intervention_validation = (
        teacher_experiment_module._intervention_data_for_donors(tiny_dataset, config)
    )
    roles = {animal.animal_id: animal.role for animal in tiny_dataset.animals}

    assert {roles[batch.animal_id] for batch in (*normal_train, *normal_validation)} == {
        "train_donor"
    }
    assert {roles[batch.treated.animal_id] for batch in intervention_train} == {"train_donor"}
    assert {roles[batch.treated.animal_id] for batch in intervention_validation} == {
        "validation_donor"
    }


def test_affine_gauge_recovery_is_coordinate_invariant() -> None:
    generator = np.random.default_rng(4)
    teacher = generator.normal(size=(200, 3))
    change = np.asarray([[1.3, 0.2, -0.1], [0.1, 0.8, 0.3], [-0.2, 0.1, 1.1]])
    offset = np.asarray([0.4, -0.2, 0.7])
    learned = teacher @ change + offset

    gauge = fit_affine_gauge(learned, teacher, ridge=1e-10)

    assert gauge.rank == 3
    assert np.isfinite(gauge.condition_number)
    np.testing.assert_allclose(gauge.transform(learned), teacher, atol=1e-8)
    vectors = generator.normal(size=(20, 3))
    np.testing.assert_allclose(gauge.transform_vectors(vectors @ change), vectors, atol=1e-8)


def test_open_loop_prediction_ignores_post_onset_outcomes(tiny_dataset) -> None:
    target = tiny_dataset.animals[-1]
    truth = tiny_dataset.ground_truth
    torch.manual_seed(9)
    model = HierarchicalControlledSSM(
        latent_dim=truth.config.dynamics.latent_dim,
        input_dim=truth.config.dynamics.task_input_dim,
        behavior_dim=truth.config.observations.behavior_dim,
        num_interventions=truth.config.intervention.n_interventions,
        hidden_dim=8,
        residual_rank=2,
        dt=truth.config.dynamics.dt,
    )
    model.register_animal(target.animal_id, target.neuron_count, donor=False)
    first = predict_target_pairs(model, target, neural_transform="log1p")

    changed_pairs = []
    for pair in target.counterfactual_pairs:
        onset = pair.onset_step
        neural = pair.control.neural_counts.copy()
        behavior = pair.control.behavior.copy()
        neural[onset:] += 10_000
        behavior[onset:] -= 10_000
        changed_control = replace(pair.control, neural_counts=neural, behavior=behavior)
        changed_pairs.append(replace(pair, control=changed_control))
    changed_target = replace(target, counterfactual_pairs=tuple(changed_pairs))
    second = predict_target_pairs(model, changed_target, neural_transform="log1p")

    np.testing.assert_array_equal(first.neural_treated, second.neural_treated)
    np.testing.assert_array_equal(first.behavior_treated, second.behavior_treated)
    np.testing.assert_array_equal(first.latent_treated, second.latent_treated)


def test_normal_readout_encodes_only_the_prequery_anchor(
    tiny_dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tiny_dataset.animals[-1]
    truth = tiny_dataset.ground_truth
    model = HierarchicalControlledSSM(
        latent_dim=truth.config.dynamics.latent_dim,
        input_dim=truth.config.dynamics.task_input_dim,
        behavior_dim=truth.config.observations.behavior_dim,
        num_interventions=truth.config.intervention.n_interventions,
        hidden_dim=8,
        residual_rank=2,
        intervention_rank=2,
        dt=truth.config.dynamics.dt,
    )
    model.register_animal(target.animal_id, target.neuron_count, donor=False)
    original_encode = model.encode
    encoded_neural: list[np.ndarray] = []

    def guarded_encode(animal_id, neural, behavior, *, sample=False):
        encoded_neural.append(neural.detach().cpu().numpy().copy())
        return original_encode(animal_id, neural, behavior, sample=sample)

    monkeypatch.setattr(model, "encode", guarded_encode)
    experiment = replace(
        _one_epoch_config(),
        readout_max_epochs=1,
        readout_patience=1,
        readout_ridge_grid=(0.1,),
    )
    readout = teacher_experiment_module.fit_normal_only_neural_readout(
        model,
        target,
        experiment,
        seed=19,
    )

    anchor = target.counterfactual_pairs[0].onset_step - 1
    fit = [trial for trial in target.normal_trials if trial.split == "normal_fit"]
    validation = [trial for trial in target.normal_trials if trial.split == "normal_val"]
    assert len(encoded_neural) == 2
    np.testing.assert_allclose(
        encoded_neural[0],
        np.log1p(np.stack([trial.trajectory.neural_counts[anchor] for trial in fit])),
    )
    np.testing.assert_allclose(
        encoded_neural[1],
        np.log1p(np.stack([trial.trajectory.neural_counts[anchor] for trial in validation])),
    )
    assert readout.rollout_anchor == anchor
    assert readout.design_rank <= truth.config.dynamics.latent_dim
    assert np.isfinite(readout.validation_loss)


def test_normal_readout_applies_fingerprinted_weight_decay(
    tiny_dataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tiny_dataset.animals[-1]
    truth = tiny_dataset.ground_truth
    model = HierarchicalControlledSSM(
        latent_dim=truth.config.dynamics.latent_dim,
        input_dim=truth.config.dynamics.task_input_dim,
        behavior_dim=truth.config.observations.behavior_dim,
        num_interventions=truth.config.intervention.n_interventions,
        hidden_dim=8,
        residual_rank=2,
        intervention_rank=2,
        dt=truth.config.dynamics.dt,
    )
    model.register_animal(target.animal_id, target.neuron_count, donor=False)
    experiment = replace(
        _one_epoch_config(),
        readout_max_epochs=1,
        readout_patience=1,
        readout_weight_decay=0.123,
        readout_ridge_grid=(0.1,),
    )
    observed_weight_decay: list[float] = []
    adam = torch.optim.Adam

    def audited_adam(parameters, **kwargs):
        observed_weight_decay.append(float(kwargs["weight_decay"]))
        return adam(parameters, **kwargs)

    monkeypatch.setattr(teacher_experiment_module.torch.optim, "Adam", audited_adam)
    teacher_experiment_module.fit_normal_only_neural_readout(
        model,
        target,
        experiment,
        seed=29,
    )

    assert observed_weight_decay == [experiment.readout_weight_decay]


def test_normal_readout_ignores_teacher_latents_and_normal_audit(
    tiny_dataset,
) -> None:
    target = tiny_dataset.animals[-1]
    truth = tiny_dataset.ground_truth
    model = HierarchicalControlledSSM(
        latent_dim=truth.config.dynamics.latent_dim,
        input_dim=truth.config.dynamics.task_input_dim,
        behavior_dim=truth.config.observations.behavior_dim,
        num_interventions=truth.config.intervention.n_interventions,
        hidden_dim=8,
        residual_rank=2,
        intervention_rank=2,
        dt=truth.config.dynamics.dt,
    )
    model.register_animal(target.animal_id, target.neuron_count, donor=False)
    experiment = replace(
        _one_epoch_config(),
        readout_max_epochs=1,
        readout_patience=1,
        readout_ridge_grid=(0.1,),
    )
    original = teacher_experiment_module.fit_normal_only_neural_readout(
        model,
        target,
        experiment,
        seed=23,
    )

    altered_trials = []
    for trial in target.normal_trials:
        trajectory = replace(
            trial.trajectory,
            latent=trial.trajectory.latent + 10_000.0,
        )
        if trial.split == "normal_audit":
            trajectory = replace(
                trajectory,
                neural_counts=trajectory.neural_counts + 10_000,
            )
        altered_trials.append(replace(trial, trajectory=trajectory))
    altered_target = replace(target, normal_trials=tuple(altered_trials))
    altered = teacher_experiment_module.fit_normal_only_neural_readout(
        model,
        altered_target,
        experiment,
        seed=23,
    )

    np.testing.assert_array_equal(original.weight, altered.weight)
    np.testing.assert_array_equal(original.bias, altered.bias)
    np.testing.assert_array_equal(original.latent_mean, altered.latent_mean)
    np.testing.assert_array_equal(original.latent_scale, altered.latent_scale)
    assert original.validation_loss == altered.validation_loss


def test_locked_experiment_fingerprint_allows_only_execution_device_changes() -> None:
    cpu = make_experiment_config(
        "full",
        seed=0,
        device="cpu",
        learned_methods=("proposed", "linear", "additive", "black_box"),
    )
    cuda = replace(
        cpu,
        normal_fit=replace(cpu.normal_fit, device="cuda", mixed_precision=True),
        intervention_fit=replace(cpu.intervention_fit, device="cuda", mixed_precision=True),
        target_fit=replace(cpu.target_fit, device="cuda", mixed_precision=True),
    )

    assert teacher_experiment_scientific_sha256(cpu) == teacher_experiment_scientific_sha256(cuda)
    validate_locked_teacher_experiment_config(cuda)
    with pytest.raises(teacher_experiment_module.ProtocolViolation):
        validate_locked_teacher_experiment_config(replace(cpu, include_ablations=False))


def test_locked_experiment_api_rejects_overrides_before_attestation(
    tiny_dataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_teacher = load_teacher_config(ROOT / "configs" / "teacher.yaml")
    locked_truth = replace(
        tiny_dataset.ground_truth,
        seed_partition="locked",
        config=canonical_teacher,
    )
    locked_metadata_only = replace(
        tiny_dataset,
        ground_truth=locked_truth,
    )

    def unexpected_attestation(**_kwargs):
        pytest.fail("invalid locked experiment reached freeze attestation")

    monkeypatch.setattr(
        teacher_experiment_module,
        "attest_preoutcome_freeze",
        unexpected_attestation,
    )
    with pytest.raises(
        teacher_experiment_module.ProtocolViolation,
        match="exact full frozen",
    ):
        run_teacher_experiment(
            locked_metadata_only,
            _one_epoch_config(),
            tmp_path / "bad-config",
        )

    canonical_experiment = make_experiment_config(
        "full",
        seed=0,
        device="cpu",
    )
    with pytest.raises(
        teacher_experiment_module.ProtocolViolation,
        match="run_seed=0",
    ):
        run_teacher_experiment(
            locked_metadata_only,
            canonical_experiment,
            tmp_path / "bad-seed",
            run_seed=1,
        )
    with pytest.raises(
        teacher_experiment_module.ProtocolViolation,
        match="never overwrites",
    ):
        run_teacher_experiment(
            locked_metadata_only,
            canonical_experiment,
            tmp_path / "bad-overwrite",
            overwrite=True,
        )


def test_locked_experiment_api_rejects_noncanonical_output_after_attestation(
    tiny_dataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_teacher = load_teacher_config(ROOT / "configs" / "teacher.yaml")
    locked_metadata_only = replace(
        tiny_dataset,
        ground_truth=replace(
            tiny_dataset.ground_truth,
            seed_partition="locked",
            config=canonical_teacher,
        ),
    )
    attestation = FreezeAttestation(
        commit="a" * 40,
        tag="pre-outcome-v1.0.0",
        tag_object="b" * 40,
    )
    monkeypatch.setattr(
        teacher_experiment_module,
        "attest_preoutcome_freeze",
        lambda **_kwargs: attestation,
    )
    monkeypatch.setattr(type(locked_metadata_only), "validate", lambda _self: None)

    with pytest.raises(
        teacher_experiment_module.ProtocolViolation,
        match="canonical one-shot path",
    ):
        run_teacher_experiment(
            locked_metadata_only,
            make_experiment_config("full", seed=0, device="cpu"),
            tmp_path / "copied-or-selected-output",
            freeze_attestation={
                "commit": attestation.commit,
                "tag": attestation.tag,
                "tag_object": attestation.tag_object,
            },
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ([], "--profile must be full"),
        (["--profile", "full", "--run-seed", "1"], "--run-seed must be 0"),
        (
            ["--profile", "full", "--methods", "proposed"],
            "--methods must be the complete frozen",
        ),
        (
            ["--profile", "full", "--normal-epochs", "1"],
            "epoch overrides are development-only",
        ),
        (
            ["--profile", "full", "--intervention-epochs", "1"],
            "epoch overrides are development-only",
        ),
        (
            ["--profile", "full", "--target-epochs", "1"],
            "epoch overrides are development-only",
        ),
        (
            ["--profile", "full", "--no-ablations"],
            "--no-ablations is development-only",
        ),
        (
            ["--profile", "full", "--overwrite"],
            "--overwrite is forbidden",
        ),
    ],
)
def test_locked_experiment_cli_rejects_overrides_before_attestation_or_world(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    message: str,
) -> None:
    cli = _load_teacher_experiment_cli()

    def unexpected_call(*_args, **_kwargs):
        pytest.fail("invalid locked CLI scope reached attestation or world generation")

    monkeypatch.setattr(cli, "attest_preoutcome_freeze", unexpected_call)
    monkeypatch.setattr(cli, "generate_teacher_world", unexpected_call)
    with pytest.raises(SystemExit, match=message):
        cli.main(
            [
                "--partition",
                "locked",
                "--acknowledge-locked",
                *arguments,
            ]
        )


def test_locked_experiment_cli_rejects_modified_external_config(
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
    cli = _load_teacher_experiment_cli()

    def unexpected_call(*_args, **_kwargs):
        pytest.fail("modified locked config reached attestation or generation")

    monkeypatch.setattr(cli, "attest_preoutcome_freeze", unexpected_call)
    monkeypatch.setattr(cli, "generate_teacher_world", unexpected_call)
    with pytest.raises(SystemExit, match="exact tracked canonical"):
        cli.main(
            [
                "--partition",
                "locked",
                "--acknowledge-locked",
                "--profile",
                "full",
                "--config",
                str(modified),
            ]
        )


@pytest.mark.parametrize(
    "artifact_name",
    [
        "metrics.json",
        "metrics.json.sha256",
        "predictions.npz",
        "predictions.npz.sha256",
        "completion.json",
        "predictions.npz.tmp",
    ],
)
def test_experiment_refuses_any_partial_artifact_without_overwrite(
    tiny_dataset,
    tmp_path: Path,
    artifact_name: str,
) -> None:
    (tmp_path / artifact_name).write_bytes(b"partial")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_teacher_experiment(
            tiny_dataset,
            _one_epoch_config(),
            tmp_path,
        )


@pytest.mark.parametrize("index_name", ["index.json", "index.json.sha256"])
def test_locked_experiment_cli_rejects_existing_index_before_attestation_or_world(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    index_name: str,
) -> None:
    index_dir = tmp_path / "full"
    index_dir.mkdir()
    (index_dir / index_name).write_text("existing", encoding="utf-8")
    cli = _load_teacher_experiment_cli()

    def unexpected_call(*_args, **_kwargs):
        pytest.fail("existing locked index reached attestation or world generation")

    monkeypatch.setattr(cli, "attest_preoutcome_freeze", unexpected_call)
    monkeypatch.setattr(cli, "generate_teacher_world", unexpected_call)
    with pytest.raises(SystemExit, match="index is append-only"):
        cli.main(
            [
                "--partition",
                "locked",
                "--acknowledge-locked",
                "--profile",
                "full",
                "--output",
                str(tmp_path),
            ]
        )


def test_one_epoch_experiment_writes_auditable_json_and_npz(
    tiny_dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_score = teacher_experiment_module._score_prediction_with_model
    score_calls = 0

    def guarded_score(*args, **kwargs):
        nonlocal score_calls
        prediction_path = tmp_path / "predictions.npz"
        digest_path = tmp_path / "predictions.npz.sha256"
        assert prediction_path.is_file()
        assert digest_path.is_file()
        expected = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
        assert digest_path.read_text(encoding="utf-8").split()[0] == expected
        score_calls += 1
        return original_score(*args, **kwargs)

    monkeypatch.setattr(
        teacher_experiment_module,
        "_score_prediction_with_model",
        guarded_score,
    )
    payload = run_teacher_experiment(
        tiny_dataset,
        _one_epoch_config(),
        tmp_path,
    )
    assert score_calls > 0

    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "metrics.json.sha256").is_file()
    assert (tmp_path / "predictions.npz").is_file()
    assert (tmp_path / "predictions.npz.sha256").is_file()
    assert (tmp_path / "completion.json").is_file()
    audit = payload["protocol_audit"]
    assert audit["target_intervention_batches_used_for_optimization"] == 0
    assert audit["target_adaptation_splits"] == ["normal_fit", "normal_val"]
    assert not audit["post_onset_outcomes_mounted_as_inputs"]
    assert audit["prediction_hashed_before_target_truth_access"]
    assert not audit["prediction_bundle_contains_target_intervention_truth"]
    identification = audit["donor_delta_identification"]["proposed"]
    assert identification["constraint"].startswith("exact_zero_mean_projection")
    assert identification["final_mean_l2_norm"] <= identification["tolerance"]
    topology = audit["nested_selection_topology"]["proposed"]
    assert topology["shared_normal_training_roles"] == ["train_donor"]
    assert topology["validation_adapter_shared_parameter_max_abs_change"] == 0.0
    assert not topology["validation_interventions_used_for_gradient_steps_before_selection"]
    assert not topology["validation_intervention_delta_present_before_selection"]
    assert not topology["final_refit_epoch_selection_from_refit_data"]
    assert (
        audit["prediction_sha256_before_score"]
        == hashlib.sha256((tmp_path / "predictions.npz").read_bytes()).hexdigest()
    )
    assert set(payload["aggregate"]) >= {"proposed", "zero_effect"}
    assert payload["aggregate"]["zero_effect"]["neural_causal_skill_mean"] == 0.0
    assert payload["aggregate"]["zero_effect"]["behavior_causal_skill_mean"] == 0.0
    readout_audit = next(iter(payload["stage_fits"]["proposed"]["targets"].values()))[
        "neural_readout"
    ]
    assert readout_audit["normal_rollout_design_rank"] > 0
    assert 0.0 <= (readout_audit["query_coordinate_fraction_outside_normal_rollout_range"]) <= 1.0
    metrics_digest = hashlib.sha256((tmp_path / "metrics.json").read_bytes()).hexdigest()
    assert (tmp_path / "metrics.json.sha256").read_text(encoding="utf-8").split()[
        0
    ] == metrics_digest
    completion = json.loads((tmp_path / "completion.json").read_text(encoding="utf-8"))
    assert completion["artifacts"]["metrics.json"] == metrics_digest
    assert completion["learned_methods"] == ["proposed"]
    assert not completion["canonical_learned_method_set_complete"]
    assert not completion["eligible_for_biological_headline_conjunction"]

    with (tmp_path / "metrics.json").open() as handle:
        restored = json.load(
            handle,
            parse_constant=lambda value: pytest.fail(f"non-standard JSON constant {value}"),
        )
    assert restored["schema_version"] == "cadence.teacher_experiment.v1"
    with np.load(tmp_path / "predictions.npz", allow_pickle=False) as archive:
        target_key = "proposed__animal_02__neural_treated"
        assert target_key in archive
        assert archive[target_key].shape[-1] == tiny_dataset.animals[-1].neuron_count
        assert archive["metadata_json"].ndim == 0
        prediction_metadata = json.loads(str(archive["metadata_json"]))
        assert prediction_metadata["learned_methods"] == ["proposed"]
        assert not prediction_metadata["canonical_learned_method_set_complete"]
        assert len(prediction_metadata["teacher_config_sha256"]) == 64
        assert len(prediction_metadata["teacher_experiment_scientific_sha256"]) == 64
        assert not any(name.startswith("truth__") for name in archive.files)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_teacher_experiment(
            tiny_dataset,
            _one_epoch_config(),
            tmp_path,
        )
