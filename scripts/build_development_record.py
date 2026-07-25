#!/usr/bin/env python3
"""Freeze the outcome-eligible development record before locked evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import statistics
import tempfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from cadence.experiments.allen import (
    CANONICAL_INDEX_RELATIVE,
    CANONICAL_MANIFEST_RELATIVE,
    CANONICAL_PROCESSED_ROOT_RELATIVE,
    EXPECTED_ALLEN_RELEASE,
    EXPECTED_INDEX_SCHEMA,
    _canonical_optimization_sha256,
    _run_configuration_sha256,
    _runtime_optimization_sha256,
    make_allen_config,
)
from cadence.experiments.allen import LEARNED_METHODS as ALLEN_LEARNED_METHODS
from cadence.experiments.teacher import (
    LEARNED_METHODS as TEACHER_LEARNED_METHODS,
)
from cadence.experiments.teacher import (
    make_experiment_config,
    make_profile_teacher_config,
    teacher_experiment_scientific_sha256,
)
from cadence.teacher import (
    StressCondition,
    _addressed_seed,
    load_teacher_config,
    teacher_config_sha256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ALLEN_PROCESSED_INDEX_SHA256 = (
    "b3a4a202a13a21449d952f5018e70feb619772311e5e1fda3cc4fead3fd1b5ce"
)
CANONICAL_ALLEN_SOURCE_CONTENT_SHA256 = (
    "9da3e04515a11b413e8784a7077d625933a6f12dd38273ac290a3db90eab7a2f"
)
ALLEN_SOURCE_FILES = (
    "stimulus_presentations.parquet",
    "window_index.parquet",
    "windows.npz",
)
ALLEN_ROLE_ARTIFACTS = (
    "normal_support.npz",
    "omission_query.npz",
    "sealed_omission_outcomes.npz",
)
TEACHER_AGGREGATE_METRICS = (
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
TEACHER_AGGREGATE_ENDPOINTS = tuple(f"{metric}_mean" for metric in TEACHER_AGGREGATE_METRICS)
TEACHER_OFF_RANGE_ENDPOINT = "query_coordinate_fraction_outside_normal_rollout_range"
TEACHER_ENDPOINTS = (*TEACHER_AGGREGATE_ENDPOINTS, TEACHER_OFF_RANGE_ENDPOINT)
TEACHER_TARGET_NEURAL_READOUT = (
    "softplus-Poisson quasi-likelihood fit on frozen open-loop normal_fit "
    "rollouts, selected on frozen open-loop normal_val rollouts"
)
TEACHER_READOUT_AUDIT_FIELDS = {
    "best_epoch",
    "validation_poisson_nll",
    "selected_ridge",
    "normal_rollout_design_rank",
    "normal_rollout_design_condition_number",
    "normal_rollout_anchor",
    "normal_rollout_support_max_abs_standardized",
    "query_max_abs_standardized",
    TEACHER_OFF_RANGE_ENDPOINT,
}
ALLEN_ENDPOINTS = (
    "neural_causal_skill",
    "running_causal_skill",
    "pupil_causal_skill",
    "lick_causal_skill",
)
ALLEN_MICE = ("539517", "448900", "484631", "423606")
ALLEN_REPORT_METHODS = (
    "proposed",
    "linear",
    "additive",
    "black_box",
    "proposed_no_residual",
    "proposed_no_target_adaptation",
    "functional_atlas",
    "no_effect",
    "condition_time",
    "nearest_donor",
)
TEACHER_REPORT_METHODS = (
    "additive",
    "black_box",
    "linear",
    "proposed",
    "proposed_native_decoder",
    "proposed_no_target_adaptation",
    "proposed_no_target_residual",
    "zero_effect",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_bytes(path: Path) -> tuple[bytes, str]:
    payload = path.read_bytes()
    return payload, hashlib.sha256(payload).hexdigest()


def _parse_json_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    raw, digest = _read_bytes(path)
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload, digest


def _validate_sidecar(path: Path, digest: str) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if fields != [digest, path.name]:
        raise ValueError(f"invalid SHA256 sidecar for {path}")


def _verify_sidecar(path: Path) -> str:
    digest = _sha256(path)
    _validate_sidecar(path, digest)
    return digest


def _read_verified_json(path: Path) -> tuple[dict[str, Any], str]:
    payload, digest = _parse_json_snapshot(path)
    _validate_sidecar(path, digest)
    return payload, digest


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _contained_artifact(directory: Path, relative: str) -> Path:
    candidate = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or candidate.is_absolute()
        or candidate.as_posix() != relative
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError(f"invalid completion artifact path {relative!r}")
    root = directory.resolve()
    artifact = directory.joinpath(*candidate.parts)
    try:
        artifact.resolve(strict=True).relative_to(root)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(
            f"completion artifact escapes or is missing from {directory}: {relative}"
        ) from error
    return artifact


def _verify_allen_completion(
    directory: Path,
    stage: str,
    required_artifacts: set[str],
) -> dict[str, Any]:
    path = directory / f"{stage}.complete.json"
    payload, digest = _read_verified_json(path)
    artifacts = payload.get("artifacts")
    if (
        payload.get("schema") != "cadence-allen-vbo-stage-completion-v1"
        or payload.get("stage") != stage
        or not isinstance(artifacts, dict)
        or set(artifacts) != required_artifacts
    ):
        raise ValueError(f"invalid Allen {stage} completion in {directory}")
    for relative, expected in artifacts.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"invalid Allen {stage} completion digest for {relative}")
        artifact = _contained_artifact(directory, relative)
        if _sha256(artifact) != expected:
            raise ValueError(f"Allen {stage} completion hash mismatch for {artifact}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Allen {stage} completion metadata is missing")
    payload["completion_sha256"] = digest
    return payload


def _canonical_allen_source_audit(
    repository: Path | None = None,
) -> dict[str, Any]:
    """Authenticate the canonical processed index and every committed source byte."""

    if repository is None:
        repository = REPOSITORY_ROOT
    repository = repository.resolve()
    processed_root = repository / CANONICAL_PROCESSED_ROOT_RELATIVE
    index_path = repository / CANONICAL_INDEX_RELATIVE
    manifest_path = repository / CANONICAL_MANIFEST_RELATIVE
    index, index_sha = _parse_json_snapshot(index_path)
    if index_sha != CANONICAL_ALLEN_PROCESSED_INDEX_SHA256:
        raise ValueError("canonical Allen processed index SHA256 changed")
    commitment = index.get("source_content_commitment")
    if (
        index.get("schema") != EXPECTED_INDEX_SCHEMA
        or index.get("release") != EXPECTED_ALLEN_RELEASE
        or index.get("cohort_manifest") != CANONICAL_MANIFEST_RELATIVE.as_posix()
        or index.get("animal_count") != 32
        or not isinstance(commitment, dict)
        or commitment
        != {
            "algorithm": "sha256-canonical-json-v1",
            "files_per_mouse": list(ALLEN_SOURCE_FILES),
            "sha256": CANONICAL_ALLEN_SOURCE_CONTENT_SHA256,
        }
    ):
        raise ValueError("canonical Allen processed index commitment is invalid")
    rows_raw = index.get("animals")
    if not isinstance(rows_raw, list) or len(rows_raw) != 32:
        raise ValueError("canonical Allen processed index must contain 32 animals")
    rows: dict[str, dict[str, Any]] = {}
    for row in rows_raw:
        if not isinstance(row, dict) or "mouse_id" not in row:
            raise ValueError("canonical Allen processed index row is malformed")
        mouse = str(row["mouse_id"])
        if mouse in rows:
            raise ValueError(f"duplicate canonical Allen mouse {mouse}")
        rows[mouse] = row
    manifest, manifest_sha = _parse_json_snapshot(manifest_path)
    manifest_rows = manifest.get("nwb_files")
    if not isinstance(manifest_rows, list):
        raise ValueError("canonical Allen manifest rows are missing")
    manifest_ids = {str(row["mouse_id"]): int(row["ophys_experiment_id"]) for row in manifest_rows}
    index_ids = {mouse: int(row["ophys_experiment_id"]) for mouse, row in rows.items()}
    if (
        len(manifest_ids) != 32
        or index_ids != manifest_ids
        or not set(ALLEN_MICE) <= set(index_ids)
    ):
        raise ValueError("canonical Allen index/manifest identities differ")

    commitment_rows: list[dict[str, Any]] = []
    development_sources: dict[str, Any] = {}
    for mouse in sorted(rows):
        row = rows[mouse]
        animal_root = processed_root / f"mouse_{mouse}"
        expected_arrays = CANONICAL_PROCESSED_ROOT_RELATIVE / f"mouse_{mouse}" / "windows.npz"
        expected_provenance = (
            CANONICAL_PROCESSED_ROOT_RELATIVE / f"mouse_{mouse}" / "provenance.json"
        )
        if (
            row.get("arrays") != expected_arrays.as_posix()
            or row.get("provenance") != expected_provenance.as_posix()
        ):
            raise ValueError(f"noncanonical Allen processed path for mouse {mouse}")
        provenance, provenance_sha = _parse_json_snapshot(repository / expected_provenance)
        if (
            str(provenance.get("mouse_id")) != mouse
            or int(provenance.get("ophys_experiment_id", -1)) != index_ids[mouse]
        ):
            raise ValueError(f"canonical Allen provenance identity mismatch for mouse {mouse}")
        extractor = provenance.get("extractor")
        expected_extractor = {
            "minimum_omissions": 80,
            "normal_calibration_trials_requested": None,
            "selection_seed": 20260725,
            "window_policy": {
                "normal_contamination_guard_s": 3.0,
                "rate_hz": 10.0,
                "window_end_s": 2.0,
                "window_start_s": -1.0,
            },
        }
        if (
            not isinstance(extractor, dict)
            or {key: extractor.get(key) for key in expected_extractor} != expected_extractor
        ):
            raise ValueError(
                f"canonical Allen preprocessing configuration mismatch for mouse {mouse}"
            )
        outputs = provenance.get("outputs")
        if not isinstance(outputs, dict):
            raise ValueError(f"canonical Allen provenance outputs missing for {mouse}")
        observed_outputs: dict[str, str] = {}
        for name in ALLEN_SOURCE_FILES:
            output = outputs.get(name)
            expected_sha = output.get("sha256") if isinstance(output, dict) else None
            if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
                raise ValueError(f"canonical Allen provenance omits {name} for mouse {mouse}")
            observed_sha = _sha256(animal_root / name)
            if observed_sha != expected_sha:
                raise ValueError(
                    f"canonical Allen source digest mismatch for mouse {mouse}: {name}"
                )
            observed_outputs[name] = observed_sha
        if observed_outputs["windows.npz"] != row.get("arrays_sha256"):
            raise ValueError(f"canonical Allen windows digest differs from index for mouse {mouse}")
        commitment_rows.append(
            {
                "mouse_id": mouse,
                "ophys_experiment_id": index_ids[mouse],
                "outputs": observed_outputs,
            }
        )
        if mouse in ALLEN_MICE:
            role_hashes = {name: _sha256(animal_root / name) for name in ALLEN_ROLE_ARTIFACTS}
            development_sources[mouse] = {
                "ophys_experiment_id": index_ids[mouse],
                "provenance_sha256": provenance_sha,
                "source_files_sha256": observed_outputs,
                "role_artifacts_sha256": role_hashes,
            }
    observed_commitment = _canonical_json_sha256(commitment_rows)
    if observed_commitment != CANONICAL_ALLEN_SOURCE_CONTENT_SHA256:
        raise ValueError("canonical Allen source-content commitment changed")
    return {
        "processed_root": CANONICAL_PROCESSED_ROOT_RELATIVE.as_posix(),
        "processed_index": {
            "path": CANONICAL_INDEX_RELATIVE.as_posix(),
            "sha256": index_sha,
            "schema": EXPECTED_INDEX_SCHEMA,
            "release": EXPECTED_ALLEN_RELEASE,
        },
        "cohort_manifest": {
            "path": CANONICAL_MANIFEST_RELATIVE.as_posix(),
            "sha256": manifest_sha,
        },
        "source_content_commitment": {
            "algorithm": "sha256-canonical-json-v1",
            "sha256": observed_commitment,
            "globally_verified_mouse_count": len(rows),
        },
        "development_mice": development_sources,
    }


def _teacher_expected_identity(
    teacher_config: Any,
    seed_index: int,
) -> dict[str, Any]:
    world_seed = int(teacher_config.seeds.development[seed_index])
    stress = asdict(StressCondition(rho=teacher_config.dynamics.residual_ratio))
    identity = {
        "partition": "development",
        "seed_index": seed_index,
        "seed": world_seed,
        "config": teacher_config.to_mapping(),
        "stress": stress,
    }
    digest = _canonical_json_sha256(identity)[:12]
    return {
        "world_id": (f"{teacher_config.release_name}-development-{seed_index:02d}-{digest}"),
        "world_seed": world_seed,
        "dataset_seed": _addressed_seed(world_seed, "dataset"),
        "stress": stress,
    }


def _teacher_scientific_experiment_mapping(
    mapping: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        raise ValueError(f"{label} teacher experiment configuration is missing")
    normalized = json.loads(json.dumps(mapping, allow_nan=False))
    for stage in ("normal_fit", "intervention_fit", "target_fit"):
        fit = normalized.get(stage)
        if (
            not isinstance(fit, dict)
            or not isinstance(fit.get("device"), str)
            or not isinstance(fit.get("mixed_precision"), bool)
        ):
            raise ValueError(f"{label} teacher {stage} execution fields are malformed")
        fit.pop("device")
        fit.pop("mixed_precision")
    return normalized


def _validate_teacher_readout_audits(
    payload: dict[str, Any],
    *,
    expected_teacher_config: Any,
    expected_experiment: Any,
) -> None:
    protocol = payload.get("protocol_audit")
    if (
        not isinstance(protocol, dict)
        or protocol.get("target_neural_readout") != TEACHER_TARGET_NEURAL_READOUT
        or protocol.get("target_readout_contemporaneous_count_encoded_as_its_own_predictor")
        is not False
    ):
        raise ValueError("teacher target neural readout protocol is noncanonical")
    expected_targets = {
        f"animal-{animal_index:02d}"
        for animal_index, role in enumerate(expected_teacher_config.cohort.roles)
        if role == "target"
    }
    expected_anchor = expected_teacher_config.intervention.onset_step - 1
    expected_rank = expected_teacher_config.dynamics.latent_dim
    allowed_ridges = set(expected_experiment.readout_ridge_grid)
    stage_fits = payload.get("stage_fits")
    if not isinstance(stage_fits, dict):
        raise ValueError("teacher readout stage records are missing")
    for method in TEACHER_LEARNED_METHODS:
        fit = stage_fits.get(method)
        targets = fit.get("targets") if isinstance(fit, dict) else None
        if not isinstance(targets, dict) or set(targets) != expected_targets:
            raise ValueError(f"teacher {method} readout target scope is incomplete")
        for target, target_fit in targets.items():
            readout = target_fit.get("neural_readout") if isinstance(target_fit, dict) else None
            if not isinstance(readout, dict) or set(readout) != TEACHER_READOUT_AUDIT_FIELDS:
                raise ValueError(f"teacher {method} {target} neural readout audit is malformed")
            best_epoch = readout.get("best_epoch")
            design_rank = readout.get("normal_rollout_design_rank")
            anchor = readout.get("normal_rollout_anchor")
            selected_ridge = _finite_float(
                readout.get("selected_ridge"),
                label=f"teacher {method} {target} selected ridge",
            )
            validation_loss = _finite_float(
                readout.get("validation_poisson_nll"),
                label=f"teacher {method} {target} validation loss",
            )
            condition = _finite_float(
                readout.get("normal_rollout_design_condition_number"),
                label=f"teacher {method} {target} design condition",
            )
            support_max = _finite_float(
                readout.get("normal_rollout_support_max_abs_standardized"),
                label=f"teacher {method} {target} support maximum",
            )
            query_max = _finite_float(
                readout.get("query_max_abs_standardized"),
                label=f"teacher {method} {target} query maximum",
            )
            off_range = _finite_float(
                readout.get(TEACHER_OFF_RANGE_ENDPOINT),
                label=f"teacher {method} {target} off-range fraction",
            )
            if (
                isinstance(best_epoch, bool)
                or not isinstance(best_epoch, int)
                or not 0 <= best_epoch < expected_experiment.readout_max_epochs
                or isinstance(design_rank, bool)
                or not isinstance(design_rank, int)
                or design_rank != expected_rank
                or isinstance(anchor, bool)
                or not isinstance(anchor, int)
                or anchor != expected_anchor
                or selected_ridge not in allowed_ridges
                or not math.isfinite(validation_loss)
                or condition < 1.0
                or support_max < 0.0
                or query_max < 0.0
                or not 0.0 <= off_range <= 1.0
            ):
                raise ValueError(f"teacher {method} {target} neural readout audit is noncanonical")


def _validate_teacher_smoke_fits(
    payload: dict[str, Any],
    *,
    expected_teacher_config: Any | None = None,
    expected_experiment: Any | None = None,
) -> None:
    if expected_teacher_config is None:
        expected_teacher_config = make_profile_teacher_config(
            load_teacher_config(REPOSITORY_ROOT / "configs/teacher.yaml"),
            "smoke",
        )
    if expected_experiment is None:
        expected_experiment = make_experiment_config(
            "smoke",
            seed=0,
            device="cpu",
            learned_methods=TEACHER_LEARNED_METHODS,
        )
    stage_fits = payload.get("stage_fits")
    protocol = payload.get("protocol_audit")
    if (
        not isinstance(stage_fits, dict)
        or set(stage_fits) != set(TEACHER_LEARNED_METHODS)
        or not isinstance(protocol, dict)
        or protocol.get("target_intervention_batches_used_for_optimization") != 0
        or protocol.get("target_adaptation_splits") != ["normal_fit", "normal_val"]
        or protocol.get("target_normal_audit_used_for_optimization") is not False
        or protocol.get("post_onset_outcomes_mounted_as_inputs") is not False
        or protocol.get("prediction_mode") != "paired_open_loop"
    ):
        raise ValueError("teacher development nested-fit audit is incomplete")
    topology_by_method = protocol.get("nested_selection_topology")
    delta_by_method = protocol.get("donor_delta_identification")
    if (
        not isinstance(topology_by_method, dict)
        or set(topology_by_method) != set(TEACHER_LEARNED_METHODS)
        or not isinstance(delta_by_method, dict)
        or set(delta_by_method) != set(TEACHER_LEARNED_METHODS)
    ):
        raise ValueError("teacher development method audit is incomplete")
    for method in TEACHER_LEARNED_METHODS:
        fit = stage_fits[method]
        selection = fit["selection"]
        topology = selection["topology_audit"]
        normal_epochs = topology["selected_normal_epochs"]
        intervention_epochs = topology["selected_intervention_epochs"]
        delta = delta_by_method[method]
        final_mean_l2_norm = _finite_float(
            delta.get("final_mean_l2_norm"),
            label=f"teacher {method} final projection norm",
        )
        tolerance = _finite_float(
            delta.get("tolerance"),
            label=f"teacher {method} projection tolerance",
        )
        if (
            topology != topology_by_method[method]
            or topology["shared_normal_training_roles"] != ["train_donor"]
            or topology["validation_donor_adapter_roles"] != ["validation_donor"]
            or topology["validation_adapter_shared_parameter_max_abs_change"] != 0.0
            or topology["validation_interventions_used_for_gradient_steps_before_selection"]
            is not False
            or topology["validation_intervention_delta_present_before_selection"] is not False
            or topology["selection_training_delta_group_count"] != 2
            or topology["final_refit_roles"] != ["train_donor", "validation_donor"]
            or topology["final_refit_epoch_selection_from_refit_data"] is not False
            or selection["normal_train_donors_only"]["best_epoch"] != normal_epochs - 1
            or selection["intervention_train_donors_validate_on_validation_donors"]["best_epoch"]
            != intervention_epochs - 1
            or len(selection["validation_donor_normal_adaptation"]) != 1
            or fit["normal"]["epochs_run"] != normal_epochs
            or fit["intervention"]["epochs_run"] != intervention_epochs
            or fit["intervention"]["donor_delta_identification"] != delta
            or delta["constraint"] != "exact_zero_mean_projection_after_every_optimizer_step"
            or delta["training_group_count"] != 3
            or tolerance != 1e-7
            or abs(final_mean_l2_norm) > tolerance
            or len(fit["targets"]) != 1
        ):
            raise ValueError(f"teacher development topology mismatch for {method}")
    _validate_teacher_readout_audits(
        payload,
        expected_teacher_config=expected_teacher_config,
        expected_experiment=expected_experiment,
    )


def _validate_allen_development_fits(
    payload: dict[str, Any],
    prediction: dict[str, Any],
    *,
    target: str,
    donors: list[str],
) -> None:
    stage_records = payload.get("stage_records")
    if (
        not isinstance(stage_records, dict)
        or set(stage_records) != set(ALLEN_LEARNED_METHODS)
        or payload["protocol_audit"]["target_intervention_outcomes_used_for_optimization"] != 0
        or payload["protocol_audit"]["target_normal_audit_used_for_optimization"] is not False
        or payload["protocol_audit"]["inner_validation_unit"] != "mouse_id"
    ):
        raise ValueError("Allen development fit audit is incomplete")
    validation_mouse = prediction.get("inner_validation_mouse")
    if validation_mouse not in donors:
        raise ValueError("Allen inner validation mouse is outside the donor set")
    selection_training = [mouse for mouse in donors if mouse != validation_mouse]
    for method in ALLEN_LEARNED_METHODS:
        fit = stage_records[method]
        boundary = fit["selection_boundary"]
        adapter = boundary["inner_validation_adapter"]
        tolerance = _finite_float(
            fit.get("delta_projection_tolerance"),
            label=f"Allen {method} projection tolerance",
        )
        selection_norm = _finite_float(
            fit.get("selection_final_delta_mean_norm"),
            label=f"Allen {method} selection projection norm",
        )
        refit_norm = _finite_float(
            fit.get("refit_final_delta_mean_norm"),
            label=f"Allen {method} refit projection norm",
        )
        normal_epochs = boundary["selected_normal_epochs"]
        intervention_epochs = boundary["selected_intervention_epochs"]
        if (
            fit["inner_validation_mouse"] != validation_mouse
            or fit["normal_selection"]["stage"] != "normal"
            or fit["normal_selection"]["best_epoch"] != normal_epochs - 1
            or fit["intervention_selection"]["stage"] != "intervention"
            or fit["intervention_selection"]["best_epoch"] != intervention_epochs - 1
            or boundary["intervention_training_mice"] != selection_training
            or boundary["intervention_validation_mice"] != [validation_mouse]
            or boundary["shared_f_fit_mice"] != selection_training
            or boundary["shared_f_excluded_mice"] != [validation_mouse, target]
            or boundary["inner_validation_mimics_outer_target"] is not True
            or adapter["mouse_id"] != validation_mouse
            or adapter["shared_f_frozen"] is not True
            or adapter["shared_state_sha256_before"] != adapter["shared_state_sha256_after"]
            or adapter["behavior_decoder_sha256_before"] != adapter["behavior_decoder_sha256_after"]
            or fit["selection_validation_delta_present"] is not False
            or set(fit["selection_delta_groups"]) != set(selection_training)
            or tolerance != 1e-7
            or abs(selection_norm) > tolerance
            or fit["refit_boundary"]["fresh_model"] is not True
            or fit["refit_boundary"]["normal_fixed_epochs"] != normal_epochs
            or fit["refit_boundary"]["intervention_fixed_epochs"] != intervention_epochs
            or fit["refit_boundary"]["normal_refit_mice"] != donors
            or fit["refit_boundary"]["intervention_refit_mice"] != donors
            or fit["refit_boundary"]["normal_refit_partitions"] != ["fit", "val"]
            or fit["normal_refit"]["stage"] != "normal"
            or fit["normal_refit"]["epochs_run"] != normal_epochs
            or fit["intervention_refit"]["stage"] != "intervention"
            or fit["intervention_refit"]["epochs_run"] != intervention_epochs
            or set(fit["refit_delta_groups"]) != set(donors)
            or abs(refit_norm) > tolerance
            or set(fit["targets"]) != {target}
            or fit["targets"][target]["stage"] != "target_adaptation"
        ):
            raise ValueError(f"Allen development topology mismatch for {method}")


def _describe(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "positive_n": sum(value > 0 for value in values),
    }


def _require_exact_mean(
    aggregate: dict[str, Any],
    aggregate_name: str,
    values: list[float],
    *,
    label: str,
) -> float:
    observed = _finite_float(
        aggregate.get(aggregate_name),
        label=f"{label} aggregate {aggregate_name}",
    )
    expected = float(np.mean(np.asarray(values, dtype=np.float64)))
    if observed != expected:
        raise ValueError(
            f"{label} aggregate drift for {aggregate_name}: "
            f"observed {observed}, recomputed {expected}"
        )
    return expected


def _teacher_method_endpoints(
    payload: dict[str, Any],
    *,
    expected_targets: list[str],
) -> dict[str, dict[str, float]]:
    metrics_by_method = payload.get("metrics_by_method_and_target")
    aggregate_by_method = payload.get("aggregate")
    if (
        not isinstance(metrics_by_method, dict)
        or set(metrics_by_method) != set(TEACHER_REPORT_METHODS)
        or not isinstance(aggregate_by_method, dict)
        or set(aggregate_by_method) != set(TEACHER_REPORT_METHODS)
    ):
        raise ValueError("teacher per-target method set is incomplete")
    expected_target_set = set(expected_targets)
    methods: dict[str, dict[str, float]] = {}
    expected_aggregate_keys = {
        "n_targets",
        *(
            f"{metric}_{suffix}"
            for metric in TEACHER_AGGREGATE_METRICS
            for suffix in ("mean", "std")
        ),
    }
    for method in TEACHER_REPORT_METHODS:
        targets = metrics_by_method.get(method)
        aggregate = aggregate_by_method.get(method)
        if (
            not isinstance(targets, dict)
            or set(targets) != expected_target_set
            or not isinstance(aggregate, dict)
            or aggregate.get("n_targets") != len(expected_targets)
            or set(aggregate) != expected_aggregate_keys
        ):
            raise ValueError(f"teacher exact per-target aggregate scope mismatch for {method}")
        methods[method] = {}
        for metric in TEACHER_AGGREGATE_METRICS:
            values = [
                float(value)
                for target in expected_targets
                if isinstance(
                    value := targets[target].get(metric),
                    int | float,
                )
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ]
            expected_mean = float(np.mean(np.asarray(values, dtype=np.float64))) if values else None
            expected_std = (
                float(np.std(np.asarray(values, dtype=np.float64), ddof=1))
                if len(values) > 1
                else None
            )
            for suffix, expected in (
                ("mean", expected_mean),
                ("std", expected_std),
            ):
                name = f"{metric}_{suffix}"
                observed = aggregate.get(name)
                if expected is None:
                    if observed is not None:
                        raise ValueError(f"teacher {method} aggregate drift for {name}")
                elif (
                    _finite_float(
                        observed,
                        label=f"teacher {method} aggregate {name}",
                    )
                    != expected
                ):
                    raise ValueError(f"teacher {method} aggregate drift for {name}")
            if expected_mean is not None:
                methods[method][f"{metric}_mean"] = expected_mean

    stage_fits = payload.get("stage_fits")
    if not isinstance(stage_fits, dict):
        raise ValueError("teacher readout stage records are missing")
    for method in TEACHER_LEARNED_METHODS:
        stage = stage_fits.get(method)
        stage_targets = stage.get("targets") if isinstance(stage, dict) else None
        if not isinstance(stage_targets, dict) or set(stage_targets) != expected_target_set:
            raise ValueError(f"teacher {method} readout target scope is incomplete")
        off_range = [
            _finite_float(
                stage_targets[target].get("neural_readout", {}).get(TEACHER_OFF_RANGE_ENDPOINT),
                label=f"teacher {method} {target} off-range fraction",
            )
            for target in expected_targets
        ]
        if any(value < 0.0 or value > 1.0 for value in off_range):
            raise ValueError("teacher off-range fraction must lie in [0, 1]")
        methods[method][TEACHER_OFF_RANGE_ENDPOINT] = float(
            np.mean(np.asarray(off_range, dtype=np.float64))
        )
    return methods


def _allen_method_endpoints(
    payload: dict[str, Any],
    *,
    target: str,
) -> dict[str, dict[str, float]]:
    animals = payload.get("animals")
    aggregate = payload.get("aggregate")
    expected_methods = set(ALLEN_REPORT_METHODS)
    if (
        not isinstance(animals, dict)
        or set(animals) != expected_methods
        or not isinstance(aggregate, dict)
        or set(aggregate) != expected_methods
    ):
        raise ValueError(f"Allen exact method scope mismatch for mouse {target}")
    result: dict[str, dict[str, float]] = {}
    for method in ALLEN_REPORT_METHODS:
        targets = animals.get(method)
        method_aggregate = aggregate.get(method)
        if (
            not isinstance(targets, dict)
            or set(targets) != {target}
            or not isinstance(method_aggregate, dict)
        ):
            raise ValueError(f"Allen exact per-target scope mismatch for {method} mouse {target}")
        target_metrics = targets[target]
        if not isinstance(target_metrics, dict):
            raise ValueError(f"Allen per-target metrics missing for {method} mouse {target}")
        result[method] = {}
        for endpoint in ALLEN_ENDPOINTS:
            value = _finite_float(
                target_metrics.get(endpoint),
                label=f"Allen {method} mouse {target} {endpoint}",
            )
            result[method][endpoint] = _require_exact_mean(
                method_aggregate,
                endpoint,
                [value],
                label=f"Allen {method} mouse {target}",
            )
    return result


def _teacher_record(
    root: Path,
    *,
    teacher_config_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if teacher_config_path is None:
        teacher_config_path = REPOSITORY_ROOT / "configs/teacher.yaml"
    base_teacher_config = load_teacher_config(teacher_config_path)
    expected_teacher_config = make_profile_teacher_config(
        base_teacher_config,
        "smoke",
    )
    expected_teacher_sha = teacher_config_sha256(expected_teacher_config)
    expected_experiment = make_experiment_config(
        "smoke",
        seed=0,
        device="cpu",
        learned_methods=TEACHER_LEARNED_METHODS,
    )
    expected_experiment_sha = teacher_experiment_scientific_sha256(expected_experiment)
    expected_experiment_mapping = _teacher_scientific_experiment_mapping(
        expected_experiment.to_mapping(),
        label="canonical",
    )
    expected_targets = [
        f"animal-{animal_index:02d}"
        for animal_index, role in enumerate(expected_teacher_config.cohort.roles)
        if role == "target"
    ]
    index_path = root / "index.json"
    index, index_sha = _read_verified_json(index_path)
    indexed_rows = index.get("worlds")
    if (
        index.get("schema_version") != "cadence.teacher_index.v2"
        or index.get("partition") != "development"
        or index.get("evaluation_role") != "method_development"
        or index.get("teacher_config_sha256") != expected_teacher_sha
        or index.get("teacher_experiment_scientific_sha256") != expected_experiment_sha
        or index.get("learned_methods") != list(TEACHER_LEARNED_METHODS)
        or index.get("canonical_learned_method_set_complete") is not True
        or not isinstance(indexed_rows, list)
        or len(indexed_rows) != 10
    ):
        raise ValueError("teacher development index is noncanonical")
    indexed_worlds = {
        int(row["seed_index"]): row
        for row in indexed_rows
        if isinstance(row, dict) and "seed_index" in row
    }
    if len(indexed_worlds) != 10 or set(indexed_worlds) != set(range(10)):
        raise ValueError("teacher development index seed set is incomplete")
    rows: list[dict[str, Any]] = []
    for seed_index in range(10):
        expected_identity = _teacher_expected_identity(
            expected_teacher_config,
            seed_index,
        )
        directory = root / f"development-seed-{seed_index:02d}"
        metrics_path = directory / "metrics.json"
        predictions_path = directory / "predictions.npz"
        completion_path = directory / "completion.json"
        payload, metrics_sha = _read_verified_json(metrics_path)
        prediction_bytes, predictions_sha = _read_bytes(predictions_path)
        _validate_sidecar(predictions_path, predictions_sha)
        completion, completion_sha = _parse_json_snapshot(completion_path)
        with np.load(io.BytesIO(prediction_bytes), allow_pickle=False) as archive:
            prediction_metadata = json.loads(str(archive["metadata_json"].item()))
        if not isinstance(prediction_metadata, dict):
            raise ValueError(f"teacher prediction metadata is invalid for seed {seed_index}")
        world = payload.get("world")
        protocol = payload.get("protocol_audit")
        observed_experiment_mapping = _teacher_scientific_experiment_mapping(
            payload.get("experiment_config"),
            label=f"seed {seed_index}",
        )
        if observed_experiment_mapping != expected_experiment_mapping:
            raise ValueError(
                f"teacher scientific experiment mapping mismatch for seed {seed_index}"
            )
        if (
            not isinstance(world, dict)
            or not isinstance(protocol, dict)
            or world.get("seed_partition") != "development"
            or world.get("world_id") != expected_identity["world_id"]
            or world.get("world_seed") != expected_identity["world_seed"]
            or world.get("dataset_seed") != expected_identity["dataset_seed"]
            or world.get("stress") != expected_identity["stress"]
            or world["teacher_config_sha256"] != expected_teacher_sha
            or payload.get("schema_version") != "cadence.teacher_experiment.v1"
            or payload.get("experiment_config", {}).get("profile") != "smoke"
            or payload.get("learned_methods") != list(TEACHER_LEARNED_METHODS)
            or payload.get("canonical_learned_method_set_complete") is not True
            or payload.get("reported_methods") != list(TEACHER_REPORT_METHODS)
            or protocol.get("teacher_config_sha256") != expected_teacher_sha
            or protocol.get("teacher_experiment_scientific_sha256") != expected_experiment_sha
            or protocol.get("prediction_sha256_before_score") != predictions_sha
            or completion.get("schema_version") != "cadence.teacher_completion.v1"
            or completion.get("world_id") != world["world_id"]
            or completion.get("seed_partition") != "development"
            or completion.get("evaluation_role") != "method_development"
            or completion.get("learned_methods") != list(TEACHER_LEARNED_METHODS)
            or completion.get("reported_methods") != list(TEACHER_REPORT_METHODS)
            or completion.get("teacher_config_sha256") != expected_teacher_sha
            or completion.get("teacher_experiment_scientific_sha256") != expected_experiment_sha
            or completion.get("artifacts")
            != {
                metrics_path.name: metrics_sha,
                predictions_path.name: predictions_sha,
            }
            or prediction_metadata.get("world_id") != world["world_id"]
            or prediction_metadata.get("run_seed") != 0
            or prediction_metadata.get("learned_methods") != list(TEACHER_LEARNED_METHODS)
            or prediction_metadata.get("canonical_learned_method_set_complete") is not True
            or prediction_metadata.get("teacher_config_sha256") != expected_teacher_sha
            or prediction_metadata.get("teacher_experiment_scientific_sha256")
            != expected_experiment_sha
            or prediction_metadata.get("targets") != expected_targets
            or prediction_metadata.get("contains_target_intervention_truth") is not False
        ):
            raise ValueError(f"teacher world identity mismatch for seed {seed_index}")
        indexed = indexed_worlds[seed_index]
        indexed_completion = Path(str(indexed.get("completion", "")))
        indexed_output = Path(str(indexed.get("output", "")))
        if not indexed_completion.is_absolute():
            indexed_completion = REPOSITORY_ROOT / indexed_completion
        if not indexed_output.is_absolute():
            indexed_output = REPOSITORY_ROOT / indexed_output
        if (
            indexed.get("world_id") != world["world_id"]
            or indexed.get("artifact_sha256") != completion["artifacts"]
            or indexed.get("aggregate") != payload.get("aggregate")
            or indexed_completion.resolve() != completion_path.resolve()
            or indexed_output.resolve() != directory.resolve()
        ):
            raise ValueError(f"teacher index/completion mismatch for seed {seed_index}")
        _validate_teacher_smoke_fits(
            payload,
            expected_teacher_config=expected_teacher_config,
            expected_experiment=expected_experiment,
        )
        methods = _teacher_method_endpoints(
            payload,
            expected_targets=expected_targets,
        )
        row = {
            "seed_index": seed_index,
            "world_id": world["world_id"],
            "world_seed": world["world_seed"],
            "dataset_seed": world["dataset_seed"],
            "stress": world["stress"],
            "metrics_path": str(metrics_path),
            "metrics_sha256": metrics_sha,
            "predictions_path": str(predictions_path),
            "predictions_sha256": predictions_sha,
            "completion_path": str(completion_path),
            "completion_sha256": completion_sha,
            "methods": methods,
            "endpoints": methods["proposed"],
        }
        rows.append(row)
    summary = {
        method: {
            endpoint: _describe([float(row["methods"][method][endpoint]) for row in rows])
            for endpoint in rows[0]["methods"][method]
        }
        for method in TEACHER_REPORT_METHODS
    }
    summary["_index"] = {"path": str(index_path), "sha256": index_sha}
    return rows, summary


def _allen_record(
    root: Path,
    *,
    source_audit: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    canonical_config = make_allen_config(
        "full",
        seed=0,
        device="cpu",
        methods=ALLEN_LEARNED_METHODS,
    )
    canonical_optimization_sha = _canonical_optimization_sha256(canonical_config)
    if source_audit is None:
        source_audit = _canonical_allen_source_audit()
    development_sources = source_audit.get("development_mice")
    if not isinstance(development_sources, dict) or set(development_sources) != set(ALLEN_MICE):
        raise ValueError("canonical Allen development source audit is incomplete")
    expected_role_artifacts = {
        mouse: development_sources[mouse]["role_artifacts_sha256"] for mouse in ALLEN_MICE
    }
    rows: list[dict[str, Any]] = []
    for mouse_id in ALLEN_MICE:
        directory = root / f"mouse_{mouse_id}"
        metrics_path = directory / "metrics.json"
        predictions_path = directory / "predictions.npz"
        preparation_path = directory / "preparation.json"
        prediction_run_path = directory / "prediction_run.json"
        prediction_sidecar_path = directory / "predictions.npz.sha256"
        payload, metrics_sha = _parse_json_snapshot(metrics_path)
        preparation, preparation_sha = _parse_json_snapshot(preparation_path)
        prediction, prediction_run_sha = _parse_json_snapshot(prediction_run_path)
        prepare_completion = _verify_allen_completion(
            directory,
            "prepare",
            {"preparation.json", f"queries/mouse_{mouse_id}/query_inputs.npz"},
        )
        predict_completion = _verify_allen_completion(
            directory,
            "predict",
            {"prediction_run.json", "predictions.npz", "predictions.npz.sha256"},
        )
        score_completion = _verify_allen_completion(
            directory,
            "score",
            {"metrics.json", "metrics_long.csv"},
        )
        donors = [mouse for mouse in ALLEN_MICE if mouse != mouse_id]
        prediction_sha = _verify_sidecar(predictions_path)
        prediction_sidecar_sha = _sha256(prediction_sidecar_path)
        configuration_sha = _run_configuration_sha256(
            run_profile="development",
            fold=None,
            donors=donors,
            targets=[mouse_id],
            optimization=canonical_config,
            seed=0,
        )
        runtime_optimization = prediction.get("optimization")
        runtime_devices = (
            {
                runtime_optimization[fit]["device"]
                for fit in ("normal_fit", "intervention_fit", "target_fit")
            }
            if isinstance(runtime_optimization, dict)
            and all(
                isinstance(runtime_optimization.get(fit), dict)
                and isinstance(runtime_optimization[fit].get("device"), str)
                for fit in ("normal_fit", "intervention_fit", "target_fit")
            )
            else set()
        )
        if len(runtime_devices) != 1:
            raise ValueError(f"Allen runtime devices differ across stages for mouse {mouse_id}")
        runtime_device = next(iter(runtime_devices))
        expected_runtime_config = make_allen_config(
            "full",
            seed=0,
            device=runtime_device,
            methods=ALLEN_LEARNED_METHODS,
        )
        runtime_sha = _runtime_optimization_sha256(expected_runtime_config)
        if _canonical_json_sha256(runtime_optimization) != runtime_sha:
            raise ValueError(
                f"Allen runtime optimization differs from full profile for mouse {mouse_id}"
            )
        experiment_artifacts = preparation.get("experiment_artifacts")
        expected_experiment_artifacts = (
            experiment_artifacts.get(mouse_id) if isinstance(experiment_artifacts, dict) else None
        )
        if not isinstance(expected_experiment_artifacts, dict):
            raise ValueError(f"Allen preparation artifacts missing for mouse {mouse_id}")
        query_relative = f"queries/mouse_{mouse_id}/query_inputs.npz"
        sealed_relative = f"queries/mouse_{mouse_id}/sealed_outcomes.npz"
        sealed_sha = _sha256(_contained_artifact(directory, sealed_relative))
        expected_sealed = {mouse_id: sealed_sha}
        protocol = payload.get("protocol_audit")
        if not isinstance(protocol, dict):
            raise ValueError(f"Allen protocol audit missing for mouse {mouse_id}")
        if (
            payload.get("schema") != "cadence-allen-vbo-experiment-v2"
            or payload.get("run_profile") != "development"
            or payload.get("optimization_profile") != "full"
            or payload.get("seed") != 0
            or payload.get("fold") is not None
            or payload.get("targets") != [mouse_id]
            or payload.get("donors") != donors
            or set(payload.get("animals", {})) != set(ALLEN_REPORT_METHODS)
            or set(payload.get("aggregate", {})) != set(ALLEN_REPORT_METHODS)
            or set(payload.get("stage_records", {})) != set(ALLEN_LEARNED_METHODS)
            or preparation.get("schema") != "cadence-allen-vbo-preparation-v1"
            or preparation.get("run_profile") != "development"
            or preparation.get("fold") is not None
            or preparation.get("targets") != [mouse_id]
            or preparation.get("donors") != donors
            or preparation.get("seed") != 0
            or preparation.get("configuration_sha256") != configuration_sha
            or preparation.get("canonical_optimization_sha256") != canonical_optimization_sha
            or preparation.get("preparation_runtime_optimization_sha256") != runtime_sha
            or preparation.get("role_artifacts") != expected_role_artifacts
            or set(experiment_artifacts) != {mouse_id}
            or expected_experiment_artifacts.get("query_inputs.npz")
            != prepare_completion["artifacts"][query_relative]
            or expected_experiment_artifacts.get("sealed_outcomes.npz") != sealed_sha
            or prediction.get("schema") != "cadence-allen-vbo-prediction-v1"
            or prediction.get("run_profile") != "development"
            or prediction.get("fold") is not None
            or prediction.get("targets") != [mouse_id]
            or prediction.get("donors") != donors
            or prediction.get("configuration_sha256") != configuration_sha
            or prediction.get("canonical_optimization_sha256") != canonical_optimization_sha
            or prediction.get("prediction_runtime_optimization_sha256") != runtime_sha
            or prediction.get("preparation_sha256") != preparation_sha
            or prediction.get("prepare_completion_sha256")
            != prepare_completion["completion_sha256"]
            or prediction.get("report_methods") != list(ALLEN_REPORT_METHODS)
            or prediction.get("prediction_sha256") != prediction_sha
            or prediction.get("stage_records") != payload.get("stage_records")
            or protocol.get("canonical_optimization_sha256") != canonical_optimization_sha
            or protocol.get("scoring_runtime_optimization_sha256") != runtime_sha
            or protocol.get("prediction_sha256_before_score") != prediction_sha
            or protocol.get("predict_completion_sha256") != predict_completion["completion_sha256"]
            or protocol.get("sealed_outcome_sha256") != expected_sealed
            or prepare_completion["metadata"]["configuration_sha256"] != configuration_sha
            or prepare_completion["metadata"]["preparation_sha256"] != preparation_sha
            or prepare_completion["artifacts"]["preparation.json"] != preparation_sha
            or predict_completion["metadata"]["configuration_sha256"] != configuration_sha
            or predict_completion["metadata"]["preparation_sha256"] != preparation_sha
            or score_completion["metadata"]["configuration_sha256"] != configuration_sha
            or predict_completion["metadata"]["prediction_sha256"] != prediction_sha
            or score_completion["metadata"]["prediction_sha256"] != prediction_sha
            or score_completion["metadata"]["sealed_outcome_sha256"] != expected_sealed
            or predict_completion["artifacts"]["prediction_run.json"] != prediction_run_sha
            or predict_completion["artifacts"]["predictions.npz"] != prediction_sha
            or predict_completion["artifacts"]["predictions.npz.sha256"] != prediction_sidecar_sha
            or score_completion["artifacts"]["metrics.json"] != metrics_sha
        ):
            raise ValueError(f"Allen development identity mismatch for mouse {mouse_id}")
        _validate_allen_development_fits(
            payload,
            prediction,
            target=mouse_id,
            donors=donors,
        )
        methods = _allen_method_endpoints(payload, target=mouse_id)
        rows.append(
            {
                "mouse_id": mouse_id,
                "metrics_path": str(metrics_path),
                "metrics_sha256": score_completion["artifacts"]["metrics.json"],
                "predictions_path": str(predictions_path),
                "predictions_sha256": prediction_sha,
                "completion_sha256": {
                    "prepare": prepare_completion["completion_sha256"],
                    "predict": predict_completion["completion_sha256"],
                    "score": score_completion["completion_sha256"],
                },
                "preparation_sha256": preparation_sha,
                "prediction_run_sha256": prediction_run_sha,
                "sealed_outcome_sha256": sealed_sha,
                "configuration_sha256": configuration_sha,
                "canonical_optimization_sha256": canonical_optimization_sha,
                "runtime_optimization_sha256": runtime_sha,
                "methods": methods,
            }
        )
    summary = {
        method: {
            endpoint: _describe([float(row["methods"][method][endpoint]) for row in rows])
            for endpoint in ALLEN_ENDPOINTS
        }
        for method in rows[0]["methods"]
    }
    return rows, summary, source_audit


def _write_long_csv(
    path: Path,
    teacher_rows: list[dict[str, Any]],
    allen_rows: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("dataset", "unit", "method", "endpoint", "value"),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in teacher_rows:
            for method, endpoints in row["methods"].items():
                for endpoint, value in endpoints.items():
                    writer.writerow(
                        {
                            "dataset": "teacher_development_smoke",
                            "unit": row["world_id"],
                            "method": method,
                            "endpoint": endpoint,
                            "value": f"{value:.17g}",
                        }
                    )
        for row in allen_rows:
            for method, endpoints in row["methods"].items():
                for endpoint, value in endpoints.items():
                    writer.writerow(
                        {
                            "dataset": "allen_vbo_development_full",
                            "unit": row["mouse_id"],
                            "method": method,
                            "endpoint": endpoint,
                            "value": f"{value:.17g}",
                        }
                    )


def _plot(
    png_path: Path,
    pdf_path: Path,
    teacher_rows: list[dict[str, Any]],
    allen_rows: list[dict[str, Any]],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
    teacher_x = list(range(10))
    for endpoint, label, color in (
        (
            "neural_condition_averaged_causal_skill_mean",
            "Neural",
            "#2B6CB0",
        ),
        (
            "behavior_condition_averaged_causal_skill_mean",
            "Behavior",
            "#D97706",
        ),
    ):
        axes[0].plot(
            teacher_x,
            [row["endpoints"][endpoint] for row in teacher_rows],
            marker="o",
            linewidth=1.2,
            markersize=4,
            label=label,
            color=color,
        )
    axes[0].axhline(0, color="#374151", linewidth=0.8)
    axes[0].set(
        title="Teacher development (smoke)",
        xlabel="Development world",
        ylabel="Causal skill",
        xticks=teacher_x,
    )
    axes[0].legend(frameon=False, loc="lower right")

    mouse_x = list(range(len(ALLEN_MICE)))
    width = 0.18
    for offset, endpoint, label, color in (
        (-width / 2, "neural_causal_skill", "Neural", "#2B6CB0"),
        (width / 2, "running_causal_skill", "Running", "#059669"),
    ):
        axes[1].scatter(
            [value + offset for value in mouse_x],
            [row["methods"]["proposed"][endpoint] for row in allen_rows],
            s=30,
            label=label,
            color=color,
            zorder=3,
        )
    axes[1].axhline(0, color="#374151", linewidth=0.8)
    axes[1].set(
        title="Allen development (full)",
        xlabel="Held-out development mouse",
        ylabel="Proposed causal skill",
        xticks=mouse_x,
        xticklabels=ALLEN_MICE,
    )
    axes[1].tick_params(axis="x", rotation=28)
    axes[1].legend(frameon=False, loc="best")
    figure.suptitle(
        "Pre-outcome development diagnostics — not confirmatory",
        fontweight="bold",
        fontsize=11,
    )
    figure.savefig(
        png_path,
        dpi=220,
        metadata={"Software": "CADENCE frozen development record"},
    )
    figure.savefig(
        pdf_path,
        metadata={
            "Title": "CADENCE pre-outcome development diagnostics",
            "Author": "Aarav Sinha",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def _write_fsynced(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _publish_exclusive_copy(source: Path, destination: Path) -> None:
    """Atomically publish one alias without ever replacing existing bytes."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(2**20), b""):
                temporary.write(block)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.link(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _publish_release(
    *,
    record: dict[str, Any],
    teacher_rows: list[dict[str, Any]],
    allen_rows: list[dict[str, Any]],
    output: Path,
    figure_root: Path,
) -> Path:
    """Build a self-contained release off-path, then publish it append-only."""

    figure_names = (
        "development_diagnostics.png",
        "development_diagnostics.pdf",
    )
    figure_targets = tuple(figure_root / name for name in figure_names)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite development release {output}")
    existing_figures = [path for path in figure_targets if path.exists()]
    if existing_figures:
        raise FileExistsError(
            "refusing to overwrite development figures: " + ", ".join(map(str, existing_figures))
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent,
        prefix=f".{output.name}.",
    ) as temporary:
        staged = Path(temporary) / "release"
        staged.mkdir()
        json_path = staged / "development_record.json"
        csv_path = staged / "development_metrics_long.csv"
        png_path = staged / figure_names[0]
        pdf_path = staged / figure_names[1]
        _write_fsynced(
            json_path,
            (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
        )
        _write_long_csv(csv_path, teacher_rows, allen_rows)
        _plot(png_path, pdf_path, teacher_rows, allen_rows)
        artifact_paths = (json_path, csv_path, png_path, pdf_path)
        for path in artifact_paths:
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        completion = {
            "schema": "cadence-development-release-completion-v1",
            "artifacts": {path.name: _sha256(path) for path in artifact_paths},
        }
        completion_path = staged / "development.complete.json"
        _write_fsynced(
            completion_path,
            (json.dumps(completion, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
        )
        completion_sha = _sha256(completion_path)
        _write_fsynced(
            staged / "development.complete.json.sha256",
            f"{completion_sha}  {completion_path.name}\n".encode(),
        )
        directory_fd = os.open(staged, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        published_figures: list[Path] = []
        try:
            for name, destination in zip(
                figure_names,
                figure_targets,
                strict=True,
            ):
                _publish_exclusive_copy(staged / name, destination)
                published_figures.append(destination)
            if output.exists():
                raise FileExistsError(f"refusing to overwrite development release {output}")
            os.rename(staged, output)
        except Exception:
            for path in published_figures:
                path.unlink(missing_ok=True)
            raise
    return output / "development_record.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher-root",
        type=Path,
        default=Path("results/teacher-development-freeze/smoke"),
    )
    parser.add_argument(
        "--allen-root",
        type=Path,
        default=Path("results/allen-vbo/development-final-full"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/releases/development"),
    )
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=Path("paper/figures"),
    )
    args = parser.parse_args()

    teacher_rows, teacher_summary = _teacher_record(args.teacher_root)
    allen_rows, allen_summary, allen_source_audit = _allen_record(args.allen_root)
    record = {
        "schema": "cadence-development-record-v1",
        "date": "2026-07-25",
        "role": "preoutcome_method_development_only",
        "eligible_for_biological_headline": False,
        "teacher": {
            "profile": "smoke",
            "seed_indices": list(range(10)),
            "worlds": teacher_rows,
            "summary": teacher_summary,
        },
        "allen_vbo": {
            "profile": "full",
            "held_out_development_mice": list(ALLEN_MICE),
            "canonical_source_audit": allen_source_audit,
            "mice": allen_rows,
            "summary": allen_summary,
        },
    }
    json_path = _publish_release(
        record=record,
        teacher_rows=teacher_rows,
        allen_rows=allen_rows,
        output=args.output,
        figure_root=args.figure_root,
    )
    print(json_path)


if __name__ == "__main__":
    main()
