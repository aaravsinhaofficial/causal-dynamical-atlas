from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_teacher_attempts.py"
HEX_A = "a" * 64
HEX_B = "b" * 64


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "cadence_test_compare_teacher_attempts",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def comparator() -> ModuleType:
    return _load_script()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> str:
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


def _fixture_payloads(
    *,
    world_id: str = "teacher-locked-world-00",
    metric_wall_seconds: float = 100.0,
    method_wall_seconds: float = 80.0,
    nested_wall_seconds: float = 3.0,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    freeze = {
        "commit": "c" * 40,
        "tag": "pre-outcome-v1.0.0",
        "tag_object": "d" * 40,
    }
    canonical_output = "results/teacher-locked/full/locked-seed-00"
    learned_methods = ["proposed", "black_box", "neural_ode"]
    reported_methods = [*learned_methods, "zero_effect"]
    world = {
        "schema_version": "cadence.teacher.v1",
        "world_id": world_id,
        "world_seed": 12345,
        "seed_partition": "locked",
        "dataset_seed": 54321,
        "seed_material_public": True,
        "evaluation_role": "post_freeze_deterministic_procedural_audit",
        "eligible_for_biological_headline_conjunction": False,
        "teacher_config_sha256": HEX_A,
        "stress": {"eta": 1.0},
    }
    metadata = {
        "schema_version": "cadence.teacher_prediction.v1",
        "world_id": world_id,
        "run_seed": 0,
        "seed_material_public": True,
        "eligible_for_biological_headline_conjunction": False,
        "canonical_relative_output": canonical_output,
        "methods": reported_methods,
        "learned_methods": learned_methods,
        "canonical_learned_method_set_complete": True,
        "targets": ["target-0"],
        "teacher_config_sha256": HEX_A,
        "teacher_experiment_scientific_sha256": HEX_B,
        "preoutcome_freeze": freeze,
        "contains_target_intervention_truth": False,
    }
    metrics = {
        "schema_version": "cadence.teacher_experiment.v1",
        "world": world,
        "canonical_relative_output": canonical_output,
        "experiment_config": {
            "profile": "full",
            "hidden_dim": 96,
            "normal_fit": {"seed": 0, "device": "cuda:0"},
        },
        "learned_methods": learned_methods,
        "canonical_learned_method_set_complete": True,
        "reported_methods": reported_methods,
        "protocol_audit": {
            "canonical_relative_output": canonical_output,
            "preoutcome_freeze": freeze,
            "teacher_config_sha256": HEX_A,
            "teacher_experiment_scientific_sha256": HEX_B,
            "prediction_sha256_before_score": None,
        },
        "stage_fits": {
            method: {
                "wall_seconds": method_wall_seconds + index,
                "targets": {
                    "target-0": {
                        "wall_seconds": nested_wall_seconds,
                        "selected_epoch": 7,
                    }
                },
            }
            for index, method in enumerate(learned_methods)
        },
        "metrics_by_method_and_target": {
            "proposed": {
                "target-0": {
                    "neural_condition_averaged_causal_skill": 0.125,
                }
            }
        },
        "aggregate": {
            "proposed": {
                "neural_condition_averaged_causal_skill_mean": 0.125,
            }
        },
        "wall_seconds": metric_wall_seconds,
        "artifacts": {
            "metrics": "metrics.json",
            "metrics_sha256": "metrics.json.sha256",
            "predictions": "predictions.npz",
            "predictions_sha256": "predictions.npz.sha256",
            "completion": "completion.json",
        },
    }
    completion = {
        "schema_version": "cadence.teacher_completion.v1",
        "world_id": world_id,
        "seed_partition": "locked",
        "seed_material_public": True,
        "evaluation_role": "post_freeze_deterministic_procedural_audit",
        "eligible_for_biological_headline_conjunction": False,
        "canonical_relative_output": canonical_output,
        "learned_methods": learned_methods,
        "canonical_learned_method_set_complete": True,
        "reported_methods": reported_methods,
        "teacher_config_sha256": HEX_A,
        "teacher_experiment_scientific_sha256": HEX_B,
        "preoutcome_freeze": freeze,
        "artifacts": {},
    }
    return metrics, metadata, completion


def _write_attempt(
    directory: Path,
    *,
    world_id: str = "teacher-locked-world-00",
    metric_wall_seconds: float = 100.0,
    method_wall_seconds: float = 80.0,
    nested_wall_seconds: float = 3.0,
    prediction_value: float = 1.5,
    prediction_dtype: str = "float32",
) -> None:
    directory.mkdir()
    metrics, metadata, completion = _fixture_payloads(
        world_id=world_id,
        metric_wall_seconds=metric_wall_seconds,
        method_wall_seconds=method_wall_seconds,
        nested_wall_seconds=nested_wall_seconds,
    )
    predictions = directory / "predictions.npz"
    np.savez_compressed(
        predictions,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        proposed__target_0=np.asarray([prediction_value, 2.5], dtype=prediction_dtype),
    )
    prediction_digest = _write_sidecar(predictions)
    metrics["protocol_audit"]["prediction_sha256_before_score"] = prediction_digest
    metrics_path = directory / "metrics.json"
    metrics_digest = _write_json(metrics_path, metrics)
    _write_sidecar(metrics_path)
    completion["artifacts"] = {
        "metrics.json": metrics_digest,
        "predictions.npz": prediction_digest,
    }
    _write_json(directory / "completion.json", completion)


def _reauthenticate_metrics(directory: Path, metrics: dict[str, Any]) -> None:
    metrics_path = directory / "metrics.json"
    metrics_digest = _write_json(metrics_path, metrics)
    _write_sidecar(metrics_path)
    completion_path = directory / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["artifacts"]["metrics.json"] = metrics_digest
    _write_json(completion_path, completion)


def _copy_prediction_bundle(source: Path, destination: Path) -> None:
    for name in ("predictions.npz", "predictions.npz.sha256"):
        (destination / name).write_bytes((source / name).read_bytes())
    metrics_path = destination / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    prediction_digest = _digest(destination / "predictions.npz")
    metrics["protocol_audit"]["prediction_sha256_before_score"] = prediction_digest
    _reauthenticate_metrics(destination, metrics)
    completion_path = destination / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["artifacts"]["predictions.npz"] = prediction_digest
    _write_json(completion_path, completion)


def test_runtime_only_differences_match_and_completion_is_observed_only(
    comparator: ModuleType,
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_attempt(first, metric_wall_seconds=100.0, method_wall_seconds=80.0)
    _write_attempt(second, metric_wall_seconds=900.0, method_wall_seconds=700.0)
    _copy_prediction_bundle(first, second)

    report = comparator.compare_attempts(first, second)

    assert report["status"] == "MATCH"
    assert report["selection"] == "NONE"
    assert (
        report["validation_scope"]["canonical_scientific_eligibility"]
        == "DELEGATED_TO_FROZEN_REPORTER"
    )
    assert report["comparisons"]["metrics_semantics"]["status"] == "MATCH"
    assert report["comparisons"]["completion_semantics"]["status"] == "MATCH"
    assert report["comparisons"]["prediction_npz_bytes"]["status"] == "MATCH"
    assert {
        row["artifacts"]["completion.json"]["authentication_status"] for row in report["attempts"]
    } == {"OBSERVED_DIGEST_ONLY"}
    assert not any(
        row["artifacts"]["completion.json"]["sidecar_authenticated"] for row in report["attempts"]
    )
    assert (
        report["attempts"][0]["artifacts"]["metrics.json"]["sha256"]
        != report["attempts"][1]["artifacts"]["metrics.json"]["sha256"]
    )


def test_nested_runtime_difference_is_not_excluded(
    comparator: ModuleType,
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_attempt(first, nested_wall_seconds=3.0)
    _write_attempt(second, nested_wall_seconds=4.0)
    _copy_prediction_bundle(first, second)

    report = comparator.compare_attempts(first, second)

    assert report["status"] == "MISMATCH"
    metrics = report["comparisons"]["metrics_semantics"]
    assert metrics["status"] == "MISMATCH"
    assert "/stage_fits/black_box/targets/target-0/wall_seconds" in metrics["difference_paths"]


@pytest.mark.parametrize(
    "missing_field",
    ("top_level", "method"),
)
def test_required_runtime_field_cannot_disappear_under_normalization(
    comparator: ModuleType,
    tmp_path: Path,
    missing_field: str,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_attempt(first)
    _write_attempt(second)
    _copy_prediction_bundle(first, second)
    metrics_path = second / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if missing_field == "top_level":
        del metrics["wall_seconds"]
    else:
        del metrics["stage_fits"]["proposed"]["wall_seconds"]
    _reauthenticate_metrics(second, metrics)

    with pytest.raises(comparator.AuditError) as error:
        comparator.compare_attempts(first, second)

    assert error.value.code == "invalid_runtime_field"


def test_prediction_bytes_and_shape_dtype_are_compared(
    comparator: ModuleType,
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_attempt(first, prediction_dtype="float32")
    _write_attempt(second, prediction_dtype="float64")

    report = comparator.compare_attempts(first, second)

    assert report["status"] == "MISMATCH"
    assert report["comparisons"]["prediction_npz_bytes"]["status"] == "MISMATCH"
    assert report["comparisons"]["prediction_shapes_and_dtypes"]["status"] == "MISMATCH"
    assert (
        "/proposed__target_0/dtype"
        in report["comparisons"]["prediction_shapes_and_dtypes"]["difference_paths"]
    )


def test_coherent_world_identity_change_is_rejected_as_discrepancy(
    comparator: ModuleType,
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_attempt(first, world_id="teacher-locked-world-00")
    _write_attempt(second, world_id="teacher-locked-world-other")

    report = comparator.compare_attempts(first, second)

    assert report["status"] == "MISMATCH"
    assert report["comparisons"]["identity"]["status"] == "MISMATCH"
    assert "/world_id" in report["comparisons"]["identity"]["difference_paths"]


def test_completion_difference_other_than_metrics_hash_is_not_excluded(
    comparator: ModuleType,
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_attempt(first)
    _write_attempt(second)
    _copy_prediction_bundle(first, second)
    completion_path = second / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["unexpected_retry_note"] = "changed"
    _write_json(completion_path, completion)

    report = comparator.compare_attempts(first, second)

    assert report["status"] == "MISMATCH"
    assert report["comparisons"]["completion_semantics"]["difference_paths"] == [
        "/unexpected_retry_note"
    ]


def test_sidecar_and_completion_bindings_are_both_required(
    comparator: ModuleType,
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    _write_attempt(first)
    sidecar = first / "metrics.json.sha256"
    sidecar.write_text(f"{'0' * 64}  metrics.json\n", encoding="utf-8")
    with pytest.raises(comparator.AuditError, match="does not exactly bind") as sidecar_error:
        comparator.authenticate_attempt(first, "attempt_1")
    assert sidecar_error.value.code == "sidecar_mismatch"

    second = tmp_path / "second"
    _write_attempt(second)
    completion_path = second / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["artifacts"]["metrics.json"] = "0" * 64
    _write_json(completion_path, completion)
    with pytest.raises(comparator.AuditError, match="does not bind") as completion_error:
        comparator.authenticate_attempt(second, "attempt_2")
    assert completion_error.value.code == "completion_binding_mismatch"


def test_cli_output_is_deterministic_and_discrepancy_is_nonzero(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_attempt(first, nested_wall_seconds=3.0)
    _write_attempt(second, nested_wall_seconds=4.0)
    _copy_prediction_bundle(first, second)

    command = [sys.executable, str(SCRIPT), str(first), str(second)]
    run_1 = subprocess.run(command, check=False, capture_output=True, text=True)
    run_2 = subprocess.run(command, check=False, capture_output=True, text=True)

    assert run_1.returncode == 1
    assert run_2.returncode == 1
    assert run_1.stdout == run_2.stdout
    payload = json.loads(run_1.stdout)
    assert payload["status"] == "MISMATCH"
    assert payload["selection"] == "NONE"
    assert run_1.stderr == ""
    assert run_2.stderr == ""


def test_same_directory_or_alias_cannot_certify_a_retry(
    comparator: ModuleType,
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt"
    alias = tmp_path / "alias"
    _write_attempt(attempt)
    alias.symlink_to(attempt, target_is_directory=True)

    for second in (attempt, alias):
        with pytest.raises(comparator.AuditError) as error:
            comparator.compare_attempts(attempt, second)
        assert error.value.code == "identical_attempt_directories"


def test_corrupt_authenticated_npz_emits_deterministic_invalid_json(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_attempt(first)
    _write_attempt(second)
    predictions = second / "predictions.npz"
    predictions.write_bytes(predictions.read_bytes()[:80])
    prediction_digest = _write_sidecar(predictions)
    metrics_path = second / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["protocol_audit"]["prediction_sha256_before_score"] = prediction_digest
    _reauthenticate_metrics(second, metrics)
    completion_path = second / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["artifacts"]["predictions.npz"] = prediction_digest
    _write_json(completion_path, completion)

    command = [sys.executable, str(SCRIPT), str(first), str(second)]
    run_1 = subprocess.run(command, check=False, capture_output=True, text=True)
    run_2 = subprocess.run(command, check=False, capture_output=True, text=True)

    assert run_1.returncode == 2
    assert run_2.returncode == 2
    assert run_1.stdout == run_2.stdout
    payload = json.loads(run_1.stdout)
    assert payload["status"] == "INVALID"
    assert payload["error"]["code"] == "invalid_prediction_archive"
    assert run_1.stderr == ""
    assert run_2.stderr == ""


def test_prediction_metadata_mismatch_is_detected_even_with_matching_key_schema(
    comparator: ModuleType,
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_attempt(first)
    _write_attempt(second)

    with np.load(second / "predictions.npz", allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        prediction = archive["proposed__target_0"].copy()
    metadata["targets"] = ["target-1"]
    np.savez_compressed(
        second / "predictions.npz",
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        proposed__target_0=prediction,
    )
    prediction_digest = _write_sidecar(second / "predictions.npz")
    metrics_path = second / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["protocol_audit"]["prediction_sha256_before_score"] = prediction_digest
    _reauthenticate_metrics(second, metrics)
    completion_path = second / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["artifacts"]["predictions.npz"] = prediction_digest
    _write_json(completion_path, completion)

    report = comparator.compare_attempts(first, second)

    assert report["status"] == "MISMATCH"
    assert report["comparisons"]["prediction_metadata"]["difference_paths"] == ["/targets/0"]
    assert report["comparisons"]["prediction_keys"]["status"] == "MATCH"
    assert report["comparisons"]["prediction_shapes_and_dtypes"]["status"] == "MATCH"
