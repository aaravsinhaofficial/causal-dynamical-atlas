from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_builder() -> ModuleType:
    path = ROOT / "scripts" / "build_development_record.py"
    spec = importlib.util.spec_from_file_location(
        "cadence_test_build_development_record",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder() -> ModuleType:
    return _load_builder()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _digest(path)


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return _digest(path)


def _write_sidecar(path: Path) -> str:
    digest = _digest(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )
    return digest


def _write_completion(
    directory: Path,
    stage: str,
    artifacts: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": "cadence-allen-vbo-stage-completion-v1",
        "stage": stage,
        "artifacts": {relative: _digest(directory / relative) for relative in artifacts},
        "metadata": metadata,
    }
    path = directory / f"{stage}.complete.json"
    _write_json(path, payload)
    _write_sidecar(path)
    return payload


def _teacher_fit(
    builder: ModuleType,
    *,
    target: str,
    off_range: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    topology = {
        "selected_normal_epochs": 2,
        "selected_intervention_epochs": 3,
        "shared_normal_training_roles": ["train_donor"],
        "validation_donor_adapter_roles": ["validation_donor"],
        "validation_adapter_shared_parameter_max_abs_change": 0.0,
        "validation_interventions_used_for_gradient_steps_before_selection": False,
        "validation_intervention_delta_present_before_selection": False,
        "selection_training_delta_group_count": 2,
        "final_refit_roles": ["train_donor", "validation_donor"],
        "final_refit_epoch_selection_from_refit_data": False,
    }
    delta = {
        "constraint": "exact_zero_mean_projection_after_every_optimizer_step",
        "training_group_count": 3,
        "tolerance": 1e-7,
        "final_mean_l2_norm": 0.0,
    }
    fit = {
        "selection": {
            "topology_audit": topology,
            "normal_train_donors_only": {"best_epoch": 1},
            "intervention_train_donors_validate_on_validation_donors": {"best_epoch": 2},
            "validation_donor_normal_adaptation": [{}],
        },
        "normal": {"epochs_run": 2},
        "intervention": {
            "epochs_run": 3,
            "donor_delta_identification": delta,
        },
        "targets": {
            target: {
                "neural_readout": {
                    "best_epoch": 1,
                    "validation_poisson_nll": 0.5,
                    "selected_ridge": 0.001,
                    "normal_rollout_design_rank": 3,
                    "normal_rollout_design_condition_number": 2.0,
                    "normal_rollout_anchor": 6,
                    "normal_rollout_support_max_abs_standardized": 2.0,
                    "query_max_abs_standardized": 3.0,
                    builder.TEACHER_OFF_RANGE_ENDPOINT: off_range,
                }
            }
        },
    }
    return fit, topology, delta


def _teacher_tree(
    builder: ModuleType,
    root: Path,
) -> Path:
    base = builder.load_teacher_config(ROOT / "configs/teacher.yaml")
    teacher_config = builder.make_profile_teacher_config(base, "smoke")
    teacher_sha = builder.teacher_config_sha256(teacher_config)
    experiment = builder.make_experiment_config(
        "smoke",
        seed=0,
        device="cpu",
        learned_methods=builder.TEACHER_LEARNED_METHODS,
    )
    experiment_sha = builder.teacher_experiment_scientific_sha256(experiment)
    target = "animal-03"
    index_rows: list[dict[str, Any]] = []
    for seed_index in range(10):
        identity = builder._teacher_expected_identity(
            teacher_config,
            seed_index,
        )
        directory = root / f"development-seed-{seed_index:02d}"
        directory.mkdir(parents=True)
        fit, topology, delta = _teacher_fit(
            builder,
            target=target,
            off_range=0.05 + seed_index / 100.0,
        )
        stage_fits = {method: copy.deepcopy(fit) for method in builder.TEACHER_LEARNED_METHODS}
        metrics_by_method: dict[str, Any] = {}
        aggregate_by_method: dict[str, Any] = {}
        for method_index, method in enumerate(builder.TEACHER_REPORT_METHODS):
            target_metrics = {
                endpoint.removesuffix("_mean"): (seed_index / 100.0 + method_index / 1000.0)
                for endpoint in builder.TEACHER_AGGREGATE_ENDPOINTS
            }
            metrics_by_method[method] = {target: target_metrics}
            aggregate_by_method[method] = {
                "n_targets": 1,
                **{
                    endpoint: target_metrics[endpoint.removesuffix("_mean")]
                    for endpoint in builder.TEACHER_AGGREGATE_ENDPOINTS
                },
                **{
                    endpoint.removesuffix("_mean") + "_std": None
                    for endpoint in builder.TEACHER_AGGREGATE_ENDPOINTS
                },
            }
        protocol = {
            "teacher_config_sha256": teacher_sha,
            "teacher_experiment_scientific_sha256": experiment_sha,
            "target_intervention_batches_used_for_optimization": 0,
            "target_adaptation_splits": ["normal_fit", "normal_val"],
            "target_normal_audit_used_for_optimization": False,
            "post_onset_outcomes_mounted_as_inputs": False,
            "prediction_mode": "paired_open_loop",
            "target_neural_readout": builder.TEACHER_TARGET_NEURAL_READOUT,
            "target_readout_contemporaneous_count_encoded_as_its_own_predictor": False,
            "nested_selection_topology": {
                method: copy.deepcopy(topology) for method in builder.TEACHER_LEARNED_METHODS
            },
            "donor_delta_identification": {
                method: copy.deepcopy(delta) for method in builder.TEACHER_LEARNED_METHODS
            },
        }
        prediction_metadata = {
            "world_id": identity["world_id"],
            "run_seed": 0,
            "learned_methods": list(builder.TEACHER_LEARNED_METHODS),
            "canonical_learned_method_set_complete": True,
            "teacher_config_sha256": teacher_sha,
            "teacher_experiment_scientific_sha256": experiment_sha,
            "targets": [target],
            "contains_target_intervention_truth": False,
        }
        predictions_path = directory / "predictions.npz"
        np.savez(
            predictions_path,
            metadata_json=np.asarray(json.dumps(prediction_metadata, sort_keys=True)),
        )
        predictions_sha = _write_sidecar(predictions_path)
        protocol["prediction_sha256_before_score"] = predictions_sha
        payload = {
            "schema_version": "cadence.teacher_experiment.v1",
            "world": {
                "seed_partition": "development",
                "world_id": identity["world_id"],
                "world_seed": identity["world_seed"],
                "dataset_seed": identity["dataset_seed"],
                "stress": identity["stress"],
                "teacher_config_sha256": teacher_sha,
            },
            "experiment_config": experiment.to_mapping(),
            "learned_methods": list(builder.TEACHER_LEARNED_METHODS),
            "canonical_learned_method_set_complete": True,
            "reported_methods": list(builder.TEACHER_REPORT_METHODS),
            "protocol_audit": protocol,
            "stage_fits": stage_fits,
            "metrics_by_method_and_target": metrics_by_method,
            "aggregate": aggregate_by_method,
        }
        metrics_path = directory / "metrics.json"
        metrics_sha = _write_json(metrics_path, payload)
        _write_sidecar(metrics_path)
        completion = {
            "schema_version": "cadence.teacher_completion.v1",
            "world_id": identity["world_id"],
            "seed_partition": "development",
            "evaluation_role": "method_development",
            "learned_methods": list(builder.TEACHER_LEARNED_METHODS),
            "reported_methods": list(builder.TEACHER_REPORT_METHODS),
            "teacher_config_sha256": teacher_sha,
            "teacher_experiment_scientific_sha256": experiment_sha,
            "artifacts": {
                "metrics.json": metrics_sha,
                "predictions.npz": predictions_sha,
            },
        }
        completion_path = directory / "completion.json"
        _write_json(completion_path, completion)
        index_rows.append(
            {
                "seed_index": seed_index,
                "world_id": identity["world_id"],
                "output": str(directory),
                "aggregate": aggregate_by_method,
                "completion": str(completion_path),
                "artifact_sha256": completion["artifacts"],
            }
        )
    index = {
        "schema_version": "cadence.teacher_index.v2",
        "partition": "development",
        "evaluation_role": "method_development",
        "teacher_config_sha256": teacher_sha,
        "teacher_experiment_scientific_sha256": experiment_sha,
        "learned_methods": list(builder.TEACHER_LEARNED_METHODS),
        "canonical_learned_method_set_complete": True,
        "worlds": index_rows,
    }
    index_path = root / "index.json"
    _write_json(index_path, index)
    _write_sidecar(index_path)
    return root


def _resign_teacher_world(root: Path, seed_index: int) -> None:
    directory = root / f"development-seed-{seed_index:02d}"
    metrics_path = directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics_sha = _write_json(metrics_path, metrics)
    _write_sidecar(metrics_path)
    predictions_path = directory / "predictions.npz"
    completion_path = directory / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["artifacts"] = {
        "metrics.json": metrics_sha,
        "predictions.npz": _digest(predictions_path),
    }
    _write_json(completion_path, completion)
    index_path = root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    row = next(row for row in index["worlds"] if row["seed_index"] == seed_index)
    row["aggregate"] = metrics["aggregate"]
    row["artifact_sha256"] = completion["artifacts"]
    _write_json(index_path, index)
    _write_sidecar(index_path)


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("world", "dataset_seed"), 1, "world identity"),
        (("world", "stress", "eta"), 0.5, "world identity"),
        (
            ("protocol_audit", "prediction_sha256_before_score"),
            "0" * 64,
            "world identity",
        ),
        (
            (
                "aggregate",
                "proposed",
                "neural_condition_averaged_causal_skill_mean",
            ),
            0.9,
            "aggregate drift",
        ),
        (
            (
                "aggregate",
                "linear",
                "behavior_causal_skill_mean",
            ),
            0.9,
            "aggregate drift",
        ),
        (
            ("experiment_config", "readout_weight_decay"),
            0.3,
            "scientific experiment mapping mismatch",
        ),
        (
            ("protocol_audit", "target_neural_readout"),
            "self-reported quasi-likelihood",
            "target neural readout protocol",
        ),
        (
            (
                "protocol_audit",
                "target_readout_contemporaneous_count_encoded_as_its_own_predictor",
            ),
            True,
            "target neural readout protocol",
        ),
        (
            (
                "stage_fits",
                "proposed",
                "targets",
                "animal-03",
                "neural_readout",
                "normal_rollout_anchor",
            ),
            5,
            "neural readout audit is noncanonical",
        ),
        (
            (
                "stage_fits",
                "linear",
                "targets",
                "animal-03",
                "neural_readout",
                "selected_ridge",
            ),
            0.123,
            "neural readout audit is noncanonical",
        ),
        (
            (
                "stage_fits",
                "additive",
                "targets",
                "animal-03",
                "neural_readout",
                "normal_rollout_design_condition_number",
            ),
            "nan",
            "must be a finite number",
        ),
        (
            (
                "stage_fits",
                "black_box",
                "targets",
                "animal-03",
                "neural_readout",
                "best_epoch",
            ),
            300,
            "neural readout audit is noncanonical",
        ),
    ],
)
def test_teacher_signed_mutations_fail(
    builder: ModuleType,
    tmp_path: Path,
    path: tuple[str, ...],
    replacement: Any,
    message: str,
) -> None:
    root = _teacher_tree(builder, tmp_path / "teacher")
    metrics_path = root / "development-seed-00/metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    cursor: dict[str, Any] = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    _write_json(metrics_path, payload)
    _resign_teacher_world(root, 0)
    with pytest.raises(ValueError, match=message):
        builder._teacher_record(
            root,
            teacher_config_path=ROOT / "configs/teacher.yaml",
        )


def test_teacher_execution_only_config_fields_may_vary(
    builder: ModuleType,
    tmp_path: Path,
) -> None:
    root = _teacher_tree(builder, tmp_path / "teacher")
    metrics_path = root / "development-seed-00/metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    for stage in ("normal_fit", "intervention_fit", "target_fit"):
        payload["experiment_config"][stage]["device"] = "cuda:9"
        payload["experiment_config"][stage]["mixed_precision"] = True
    _write_json(metrics_path, payload)
    _resign_teacher_world(root, 0)
    rows, _ = builder._teacher_record(
        root,
        teacher_config_path=ROOT / "configs/teacher.yaml",
    )
    assert len(rows) == 10


def test_teacher_record_commits_off_range_and_csv(
    builder: ModuleType,
    tmp_path: Path,
) -> None:
    root = _teacher_tree(builder, tmp_path / "teacher")
    rows, summary = builder._teacher_record(
        root,
        teacher_config_path=ROOT / "configs/teacher.yaml",
    )
    assert rows[0]["endpoints"][builder.TEACHER_OFF_RANGE_ENDPOINT] == 0.05
    assert summary["proposed"][builder.TEACHER_OFF_RANGE_ENDPOINT]["n"] == 10
    csv_path = tmp_path / "teacher.csv"
    builder._write_long_csv(csv_path, rows, [])
    with csv_path.open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert any(
        row["endpoint"] == builder.TEACHER_OFF_RANGE_ENDPOINT
        and row["value"] == "0.050000000000000003"
        for row in csv_rows
    )
    assert {row["method"] for row in csv_rows} == set(builder.TEACHER_REPORT_METHODS)


def test_teacher_signed_off_range_mutation_fails(
    builder: ModuleType,
    tmp_path: Path,
) -> None:
    root = _teacher_tree(builder, tmp_path / "teacher")
    metrics_path = root / "development-seed-00/metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["stage_fits"]["proposed"]["targets"]["animal-03"]["neural_readout"][
        builder.TEACHER_OFF_RANGE_ENDPOINT
    ] = 1.1
    _write_json(metrics_path, payload)
    _resign_teacher_world(root, 0)
    with pytest.raises(ValueError, match="neural readout audit is noncanonical"):
        builder._teacher_record(
            root,
            teacher_config_path=ROOT / "configs/teacher.yaml",
        )


def test_teacher_signed_readout_structure_mutation_fails(
    builder: ModuleType,
    tmp_path: Path,
) -> None:
    root = _teacher_tree(builder, tmp_path / "teacher")
    metrics_path = root / "development-seed-00/metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["stage_fits"]["proposed"]["targets"]["animal-03"]["neural_readout"][
        "uncommitted_diagnostic"
    ] = 1.0
    _write_json(metrics_path, payload)
    _resign_teacher_world(root, 0)
    with pytest.raises(ValueError, match="neural readout audit is malformed"):
        builder._teacher_record(
            root,
            teacher_config_path=ROOT / "configs/teacher.yaml",
        )


def test_teacher_projection_norm_must_be_finite(
    builder: ModuleType,
) -> None:
    fit, topology, delta = _teacher_fit(
        builder,
        target="animal-03",
        off_range=0.1,
    )
    stage_fits = {method: copy.deepcopy(fit) for method in builder.TEACHER_LEARNED_METHODS}
    protocol = {
        "target_intervention_batches_used_for_optimization": 0,
        "target_adaptation_splits": ["normal_fit", "normal_val"],
        "target_normal_audit_used_for_optimization": False,
        "post_onset_outcomes_mounted_as_inputs": False,
        "prediction_mode": "paired_open_loop",
        "nested_selection_topology": {
            method: copy.deepcopy(topology) for method in builder.TEACHER_LEARNED_METHODS
        },
        "donor_delta_identification": {
            method: copy.deepcopy(delta) for method in builder.TEACHER_LEARNED_METHODS
        },
    }
    protocol["donor_delta_identification"]["proposed"]["final_mean_l2_norm"] = float("nan")
    stage_fits["proposed"]["intervention"]["donor_delta_identification"] = protocol[
        "donor_delta_identification"
    ]["proposed"]
    with pytest.raises(ValueError, match="finite number"):
        builder._validate_teacher_smoke_fits({"stage_fits": stage_fits, "protocol_audit": protocol})


def _allen_stage_records(
    builder: ModuleType,
    *,
    target: str,
    donors: list[str],
) -> tuple[dict[str, Any], str]:
    validation_mouse = donors[0]
    selection_training = [mouse for mouse in donors if mouse != validation_mouse]
    fit = {
        "inner_validation_mouse": validation_mouse,
        "normal_selection": {"stage": "normal", "best_epoch": 1},
        "intervention_selection": {
            "stage": "intervention",
            "best_epoch": 2,
        },
        "selection_boundary": {
            "selected_normal_epochs": 2,
            "selected_intervention_epochs": 3,
            "intervention_training_mice": selection_training,
            "intervention_validation_mice": [validation_mouse],
            "shared_f_fit_mice": selection_training,
            "shared_f_excluded_mice": [validation_mouse, target],
            "inner_validation_mimics_outer_target": True,
            "inner_validation_adapter": {
                "mouse_id": validation_mouse,
                "shared_f_frozen": True,
                "shared_state_sha256_before": "same",
                "shared_state_sha256_after": "same",
                "behavior_decoder_sha256_before": "same",
                "behavior_decoder_sha256_after": "same",
            },
        },
        "selection_validation_delta_present": False,
        "selection_delta_groups": selection_training,
        "selection_final_delta_mean_norm": 0.0,
        "delta_projection_tolerance": 1e-7,
        "refit_boundary": {
            "fresh_model": True,
            "normal_fixed_epochs": 2,
            "intervention_fixed_epochs": 3,
            "normal_refit_mice": donors,
            "intervention_refit_mice": donors,
            "normal_refit_partitions": ["fit", "val"],
        },
        "normal_refit": {"stage": "normal", "epochs_run": 2},
        "intervention_refit": {
            "stage": "intervention",
            "epochs_run": 3,
        },
        "refit_delta_groups": donors,
        "refit_final_delta_mean_norm": 0.0,
        "targets": {target: {"stage": "target_adaptation"}},
    }
    return (
        {method: copy.deepcopy(fit) for method in builder.ALLEN_LEARNED_METHODS},
        validation_mouse,
    )


def _allen_tree(
    builder: ModuleType,
    root: Path,
) -> tuple[Path, dict[str, Any]]:
    config = builder.make_allen_config(
        "full",
        seed=0,
        device="cpu",
        methods=builder.ALLEN_LEARNED_METHODS,
    )
    canonical_sha = builder._canonical_optimization_sha256(config)
    runtime_sha = builder._runtime_optimization_sha256(config)
    role_hashes = {
        mouse: {
            name: hashlib.sha256(f"{mouse}:{name}".encode()).hexdigest()
            for name in builder.ALLEN_ROLE_ARTIFACTS
        }
        for mouse in builder.ALLEN_MICE
    }
    source_audit = {
        "development_mice": {
            mouse: {"role_artifacts_sha256": role_hashes[mouse]} for mouse in builder.ALLEN_MICE
        }
    }
    for mouse in builder.ALLEN_MICE:
        directory = root / f"mouse_{mouse}"
        query_relative = f"queries/mouse_{mouse}/query_inputs.npz"
        sealed_relative = f"queries/mouse_{mouse}/sealed_outcomes.npz"
        query_sha = _write_bytes(
            directory / query_relative,
            f"query:{mouse}".encode(),
        )
        sealed_sha = _write_bytes(
            directory / sealed_relative,
            f"sealed:{mouse}".encode(),
        )
        donors = [item for item in builder.ALLEN_MICE if item != mouse]
        configuration_sha = builder._run_configuration_sha256(
            run_profile="development",
            fold=None,
            donors=donors,
            targets=[mouse],
            optimization=config,
            seed=0,
        )
        preparation = {
            "schema": "cadence-allen-vbo-preparation-v1",
            "run_profile": "development",
            "fold": None,
            "donors": donors,
            "targets": [mouse],
            "seed": 0,
            "configuration_sha256": configuration_sha,
            "canonical_optimization_sha256": canonical_sha,
            "preparation_runtime_optimization_sha256": runtime_sha,
            "role_artifacts": role_hashes,
            "experiment_artifacts": {
                mouse: {
                    "query_inputs.npz": query_sha,
                    "sealed_outcomes.npz": sealed_sha,
                }
            },
        }
        preparation_path = directory / "preparation.json"
        preparation_sha = _write_json(preparation_path, preparation)
        prepare_completion = _write_completion(
            directory,
            "prepare",
            ["preparation.json", query_relative],
            {
                "configuration_sha256": configuration_sha,
                "preparation_sha256": preparation_sha,
            },
        )
        predictions_path = directory / "predictions.npz"
        prediction_sha = _write_bytes(
            predictions_path,
            f"predictions:{mouse}".encode(),
        )
        _write_sidecar(predictions_path)
        stage_records, validation_mouse = _allen_stage_records(
            builder,
            target=mouse,
            donors=donors,
        )
        prediction = {
            "schema": "cadence-allen-vbo-prediction-v1",
            "run_profile": "development",
            "fold": None,
            "donors": donors,
            "targets": [mouse],
            "inner_validation_mouse": validation_mouse,
            "report_methods": list(builder.ALLEN_REPORT_METHODS),
            "configuration_sha256": configuration_sha,
            "canonical_optimization_sha256": canonical_sha,
            "prediction_runtime_optimization_sha256": runtime_sha,
            "preparation_sha256": preparation_sha,
            "prepare_completion_sha256": _digest(directory / "prepare.complete.json"),
            "optimization": asdict(config),
            "prediction_sha256": prediction_sha,
            "stage_records": stage_records,
        }
        _write_json(directory / "prediction_run.json", prediction)
        predict_completion = _write_completion(
            directory,
            "predict",
            [
                "prediction_run.json",
                "predictions.npz",
                "predictions.npz.sha256",
            ],
            {
                "configuration_sha256": configuration_sha,
                "preparation_sha256": preparation_sha,
                "prediction_sha256": prediction_sha,
            },
        )
        animals: dict[str, Any] = {}
        aggregate: dict[str, Any] = {}
        for method_index, method in enumerate(builder.ALLEN_REPORT_METHODS):
            metrics = {endpoint: method_index / 100.0 for endpoint in builder.ALLEN_ENDPOINTS}
            animals[method] = {mouse: metrics}
            aggregate[method] = dict(metrics)
        metrics_payload = {
            "schema": "cadence-allen-vbo-experiment-v2",
            "run_profile": "development",
            "optimization_profile": "full",
            "fold": None,
            "seed": 0,
            "donors": donors,
            "targets": [mouse],
            "stage_records": stage_records,
            "protocol_audit": {
                "target_intervention_outcomes_used_for_optimization": 0,
                "target_normal_audit_used_for_optimization": False,
                "inner_validation_unit": "mouse_id",
                "canonical_optimization_sha256": canonical_sha,
                "scoring_runtime_optimization_sha256": runtime_sha,
                "prediction_sha256_before_score": prediction_sha,
                "predict_completion_sha256": _digest(directory / "predict.complete.json"),
                "sealed_outcome_sha256": {mouse: sealed_sha},
            },
            "animals": animals,
            "aggregate": aggregate,
        }
        _write_json(directory / "metrics.json", metrics_payload)
        _write_bytes(directory / "metrics_long.csv", b"fixture\n")
        _write_completion(
            directory,
            "score",
            ["metrics.json", "metrics_long.csv"],
            {
                "configuration_sha256": configuration_sha,
                "prediction_sha256": prediction_sha,
                "sealed_outcome_sha256": {mouse: sealed_sha},
            },
        )
        assert prepare_completion["metadata"]["preparation_sha256"] == preparation_sha
        assert predict_completion["metadata"]["prediction_sha256"] == prediction_sha
    return root, source_audit


def _resign_allen_metrics(directory: Path) -> None:
    metrics_path = directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    _write_json(metrics_path, metrics)
    completion_path = directory / "score.complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["artifacts"]["metrics.json"] = _digest(metrics_path)
    _write_json(completion_path, completion)
    _write_sidecar(completion_path)


def test_allen_record_validates_full_chain_and_recomputes_targets(
    builder: ModuleType,
    tmp_path: Path,
) -> None:
    root, source_audit = _allen_tree(builder, tmp_path / "allen")
    rows, summary, observed_source = builder._allen_record(
        root,
        source_audit=source_audit,
    )
    assert len(rows) == 4
    assert observed_source == source_audit
    assert (
        rows[0]["methods"]["proposed"]["neural_causal_skill"]
        == summary["proposed"]["neural_causal_skill"]["mean"]
    )


def test_allen_signed_aggregate_drift_fails(
    builder: ModuleType,
    tmp_path: Path,
) -> None:
    root, source_audit = _allen_tree(builder, tmp_path / "allen")
    directory = root / f"mouse_{builder.ALLEN_MICE[0]}"
    metrics_path = directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["aggregate"]["proposed"]["neural_causal_skill"] = 0.8
    _write_json(metrics_path, metrics)
    _resign_allen_metrics(directory)
    with pytest.raises(ValueError, match="aggregate drift"):
        builder._allen_record(root, source_audit=source_audit)


def test_allen_mixed_stage_link_fails_after_resigning(
    builder: ModuleType,
    tmp_path: Path,
) -> None:
    root, source_audit = _allen_tree(builder, tmp_path / "allen")
    directory = root / f"mouse_{builder.ALLEN_MICE[0]}"
    prediction_path = directory / "prediction_run.json"
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    prediction["preparation_sha256"] = "0" * 64
    _write_json(prediction_path, prediction)
    predict_completion_path = directory / "predict.complete.json"
    predict_completion = json.loads(predict_completion_path.read_text(encoding="utf-8"))
    predict_completion["artifacts"]["prediction_run.json"] = _digest(prediction_path)
    _write_json(predict_completion_path, predict_completion)
    _write_sidecar(predict_completion_path)
    metrics_path = directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["protocol_audit"]["predict_completion_sha256"] = _digest(predict_completion_path)
    _write_json(metrics_path, metrics)
    _resign_allen_metrics(directory)
    with pytest.raises(ValueError, match="identity mismatch"):
        builder._allen_record(root, source_audit=source_audit)


def test_allen_sealed_outcome_link_fails_after_resigning(
    builder: ModuleType,
    tmp_path: Path,
) -> None:
    root, source_audit = _allen_tree(builder, tmp_path / "allen")
    directory = root / f"mouse_{builder.ALLEN_MICE[0]}"
    completion_path = directory / "score.complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["metadata"]["sealed_outcome_sha256"] = {builder.ALLEN_MICE[0]: "0" * 64}
    _write_json(completion_path, completion)
    _write_sidecar(completion_path)
    with pytest.raises(ValueError, match="identity mismatch"):
        builder._allen_record(root, source_audit=source_audit)


@pytest.mark.parametrize(
    ("artifact", "required", "message"),
    [
        ("extra.bin", {"payload.bin"}, "invalid Allen"),
        (
            "../outside.bin",
            {"../outside.bin"},
            "invalid completion artifact path",
        ),
    ],
)
def test_allen_completion_rejects_supersets_and_escape(
    builder: ModuleType,
    tmp_path: Path,
    artifact: str,
    required: set[str],
    message: str,
) -> None:
    directory = tmp_path / "run"
    directory.mkdir()
    _write_bytes(directory / "payload.bin", b"payload")
    outside = tmp_path / "outside.bin"
    _write_bytes(outside, b"outside")
    artifacts = {"payload.bin": _digest(directory / "payload.bin")}
    if artifact == "extra.bin":
        _write_bytes(directory / artifact, b"extra")
        artifacts[artifact] = _digest(directory / artifact)
    else:
        artifacts = {artifact: _digest(outside)}
    completion = {
        "schema": "cadence-allen-vbo-stage-completion-v1",
        "stage": "prepare",
        "artifacts": artifacts,
        "metadata": {},
    }
    path = directory / "prepare.complete.json"
    _write_json(path, completion)
    _write_sidecar(path)
    with pytest.raises(ValueError, match=message):
        builder._verify_allen_completion(
            directory,
            "prepare",
            required,
        )


def test_allen_projection_norm_must_be_finite(
    builder: ModuleType,
) -> None:
    target = builder.ALLEN_MICE[0]
    donors = [mouse for mouse in builder.ALLEN_MICE if mouse != target]
    stage_records, validation_mouse = _allen_stage_records(
        builder,
        target=target,
        donors=donors,
    )
    stage_records["proposed"]["selection_final_delta_mean_norm"] = float("nan")
    payload = {
        "stage_records": stage_records,
        "protocol_audit": {
            "target_intervention_outcomes_used_for_optimization": 0,
            "target_normal_audit_used_for_optimization": False,
            "inner_validation_unit": "mouse_id",
        },
    }
    with pytest.raises(ValueError, match="finite number"):
        builder._validate_allen_development_fits(
            payload,
            {"inner_validation_mouse": validation_mouse},
            target=target,
            donors=donors,
        )


def _canonical_source_tree(
    builder: ModuleType,
    repository: Path,
) -> tuple[str, str]:
    processed = repository / builder.CANONICAL_PROCESSED_ROOT_RELATIVE
    mice = [*builder.ALLEN_MICE, *[f"9{index:05d}" for index in range(28)]]
    rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    commitment_rows: list[dict[str, Any]] = []
    for index, mouse in enumerate(sorted(mice)):
        animal = processed / f"mouse_{mouse}"
        outputs = {
            name: _write_bytes(
                animal / name,
                f"{mouse}:{name}".encode(),
            )
            for name in builder.ALLEN_SOURCE_FILES
        }
        if mouse in builder.ALLEN_MICE:
            for name in builder.ALLEN_ROLE_ARTIFACTS:
                _write_bytes(animal / name, f"role:{mouse}:{name}".encode())
        provenance = {
            "mouse_id": mouse,
            "ophys_experiment_id": 1000 + index,
            "extractor": {
                "minimum_omissions": 80,
                "normal_calibration_trials_requested": None,
                "selection_seed": 20260725,
                "window_policy": {
                    "normal_contamination_guard_s": 3.0,
                    "rate_hz": 10.0,
                    "window_end_s": 2.0,
                    "window_start_s": -1.0,
                },
            },
            "outputs": {name: {"sha256": digest} for name, digest in outputs.items()},
        }
        _write_json(animal / "provenance.json", provenance)
        rows.append(
            {
                "mouse_id": mouse,
                "ophys_experiment_id": 1000 + index,
                "arrays": (f"data/processed/allen_vbo/mouse_{mouse}/windows.npz"),
                "arrays_sha256": outputs["windows.npz"],
                "provenance": (f"data/processed/allen_vbo/mouse_{mouse}/provenance.json"),
            }
        )
        manifest_rows.append({"mouse_id": mouse, "ophys_experiment_id": 1000 + index})
        commitment_rows.append(
            {
                "mouse_id": mouse,
                "ophys_experiment_id": 1000 + index,
                "outputs": outputs,
            }
        )
    commitment = builder._canonical_json_sha256(commitment_rows)
    index = {
        "schema": builder.EXPECTED_INDEX_SCHEMA,
        "release": builder.EXPECTED_ALLEN_RELEASE,
        "cohort_manifest": builder.CANONICAL_MANIFEST_RELATIVE.as_posix(),
        "animal_count": 32,
        "source_content_commitment": {
            "algorithm": "sha256-canonical-json-v1",
            "files_per_mouse": list(builder.ALLEN_SOURCE_FILES),
            "sha256": commitment,
        },
        "animals": rows,
    }
    index_sha = _write_json(
        repository / builder.CANONICAL_INDEX_RELATIVE,
        index,
    )
    _write_json(
        repository / builder.CANONICAL_MANIFEST_RELATIVE,
        {"nwb_files": manifest_rows},
    )
    return index_sha, commitment


def test_canonical_allen_source_commitment_and_mutation(
    builder: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    index_sha, commitment = _canonical_source_tree(builder, repository)
    monkeypatch.setattr(
        builder,
        "CANONICAL_ALLEN_PROCESSED_INDEX_SHA256",
        index_sha,
    )
    monkeypatch.setattr(
        builder,
        "CANONICAL_ALLEN_SOURCE_CONTENT_SHA256",
        commitment,
    )
    audit = builder._canonical_allen_source_audit(repository)
    assert audit["source_content_commitment"]["globally_verified_mouse_count"] == 32
    mouse = builder.ALLEN_MICE[0]
    source = (
        repository / builder.CANONICAL_PROCESSED_ROOT_RELATIVE / f"mouse_{mouse}" / "windows.npz"
    )
    source.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="source digest mismatch"):
        builder._canonical_allen_source_audit(repository)
    source.write_bytes(
        f"{mouse}:windows.npz".encode(),
    )
    index_path = repository / builder.CANONICAL_INDEX_RELATIVE
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["source_content_commitment"]["sha256"] = "0" * 64
    mutated_index_sha = _write_json(index_path, index)
    monkeypatch.setattr(
        builder,
        "CANONICAL_ALLEN_PROCESSED_INDEX_SHA256",
        mutated_index_sha,
    )
    with pytest.raises(ValueError, match="index commitment is invalid"):
        builder._canonical_allen_source_audit(repository)


def test_release_publish_is_atomic_append_only_and_manifested(
    builder: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "release"
    figure_root = tmp_path / "figures"

    def fake_plot(
        png_path: Path,
        pdf_path: Path,
        teacher_rows: list[dict[str, Any]],
        allen_rows: list[dict[str, Any]],
    ) -> None:
        assert teacher_rows == []
        assert allen_rows == []
        png_path.write_bytes(b"png")
        pdf_path.write_bytes(b"pdf")

    monkeypatch.setattr(builder, "_plot", fake_plot)
    json_path = builder._publish_release(
        record={"schema": "fixture"},
        teacher_rows=[],
        allen_rows=[],
        output=output,
        figure_root=figure_root,
    )
    assert json_path == output / "development_record.json"
    completion = json.loads((output / "development.complete.json").read_text(encoding="utf-8"))
    assert completion["artifacts"] == {
        name: _digest(output / name)
        for name in (
            "development_record.json",
            "development_metrics_long.csv",
            "development_diagnostics.png",
            "development_diagnostics.pdf",
        )
    }
    assert (output / "development.complete.json.sha256").read_text(encoding="utf-8").split()[
        0
    ] == _digest(output / "development.complete.json")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        builder._publish_release(
            record={"schema": "changed"},
            teacher_rows=[],
            allen_rows=[],
            output=output,
            figure_root=figure_root,
        )
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"schema": "fixture"}


def test_release_alias_failure_publishes_no_completion_and_reruns_cleanly(
    builder: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "release"
    figure_root = tmp_path / "figures"

    def fake_plot(
        png_path: Path,
        pdf_path: Path,
        teacher_rows: list[dict[str, Any]],
        allen_rows: list[dict[str, Any]],
    ) -> None:
        png_path.write_bytes(b"png")
        pdf_path.write_bytes(b"pdf")

    real_publish = builder._publish_exclusive_copy
    calls = 0

    def fail_second_alias(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second-alias failure")
        real_publish(source, destination)

    monkeypatch.setattr(builder, "_plot", fake_plot)
    monkeypatch.setattr(
        builder,
        "_publish_exclusive_copy",
        fail_second_alias,
    )
    with pytest.raises(OSError, match="second-alias failure"):
        builder._publish_release(
            record={"schema": "fixture"},
            teacher_rows=[],
            allen_rows=[],
            output=output,
            figure_root=figure_root,
        )
    assert not output.exists()
    assert not (output / "development.complete.json").exists()
    assert not (figure_root / "development_diagnostics.png").exists()
    assert not (figure_root / "development_diagnostics.pdf").exists()

    monkeypatch.setattr(builder, "_publish_exclusive_copy", real_publish)
    json_path = builder._publish_release(
        record={"schema": "fixture"},
        teacher_rows=[],
        allen_rows=[],
        output=output,
        figure_root=figure_root,
    )
    assert json_path.exists()
    assert (output / "development.complete.json").exists()
    assert (figure_root / "development_diagnostics.png").exists()
    assert (figure_root / "development_diagnostics.pdf").exists()


def test_release_plot_failure_publishes_nothing(
    builder: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "release"
    figure_root = tmp_path / "figures"

    def failing_plot(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("plot failed")

    monkeypatch.setattr(builder, "_plot", failing_plot)
    with pytest.raises(RuntimeError, match="plot failed"):
        builder._publish_release(
            record={"schema": "fixture"},
            teacher_rows=[],
            allen_rows=[],
            output=output,
            figure_root=figure_root,
        )
    assert not output.exists()
    assert not figure_root.exists()
