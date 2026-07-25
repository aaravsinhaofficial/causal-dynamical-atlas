from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import cadence.reporting as reporting

_ATTESTATION = {
    "commit": "c" * 40,
    "tag": reporting.PREOUTCOME_TAG,
    "tag_object": "d" * 40,
}
_TARGET = "ICMS92"
_CANONICAL_RELATIVE_OUTPUT = f"results/icms/loao-{_TARGET}"
_SESSION_KEY = "ICMS92::synthetic-session"
_ADAPTER_ID = "ICMS92::synthetic-adapter"
_ARRAY_KEY = "synthetic_session"
_CONDITION_COLUMNS = [
    "animal_id",
    "session_id",
    "session_key",
    "method",
    "condition",
    "current_uA",
    "electrode_rel_y_um",
    "trials",
    "randomized_causal_eligible",
    "condition_primary_randomized_status",
    "session_primary_randomized_status",
    "block_validated",
    "block_rule",
    "missing_same_block_catches",
    "valid_primary_neural_cells",
    "valid_primary_behavior_cells",
    "observed_neural_effect_rms",
    "predicted_neural_effect_rms",
]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _authenticate_existing(path: Path) -> str:
    digest = _sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )
    return digest


def _write_bytes(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return _authenticate_existing(path)


def _write_json(path: Path, value: Any) -> str:
    return _write_bytes(
        path,
        (json.dumps(value, sort_keys=True, indent=2) + "\n").encode(),
    )


def _write_completion(
    directory: Path,
    stage: str,
    artifact_name: str,
    *,
    seal_transaction_sha256: str,
) -> str:
    return _write_json(
        directory / f"{stage}_complete.json",
        {
            "schema": f"cadence-icms-{stage}-complete-v1",
            "stage": stage,
            "artifact": artifact_name,
            "artifact_sha256": _sha256_file(directory / artifact_name),
            "append_only": True,
            "freeze_attestation": _ATTESTATION,
            "canonical_relative_output": _CANONICAL_RELATIVE_OUTPUT,
            "seal_transaction_sha256": seal_transaction_sha256,
        },
    )


def _full_config() -> dict[str, Any]:
    def fit(seed: int, epochs: int, patience: int) -> dict[str, Any]:
        return {
            "learning_rate": 0.001,
            "max_epochs": epochs,
            "patience": patience,
            "weight_decay": 0.0001,
            "gradient_clip": 5.0,
            "seed": seed,
            "device": "cpu",
            "mixed_precision": False,
        }

    return {
        "profile": "full",
        "latent_dim": 12,
        "hidden_dim": 96,
        "residual_rank": 2,
        "intervention_rank": 4,
        "batch_size": 32,
        "max_normal_trials_per_session": None,
        "max_stimulation_trials_per_session": None,
        "query_contexts_per_session": 16,
        "uncertainty_draws": 64,
        "pulse_frequency_hz": 100.0,
        "pulse_count": 70.0,
        "pulse_width_us": 167.0,
        "train_stop_s": 0.7,
        "artifact_exclusion_stop_s": 0.705,
        "seed": reporting.LOCK_SEED,
        "current_grid_uA": [float(value) for value in range(1, 14)],
        "normal_fit": fit(20_260_736, 400, 35),
        "intervention_fit": fit(20_260_748, 400, 35),
        "target_fit": fit(20_260_762, 300, 30),
    }


def _fit_audit(target: str, donors: list[str]) -> dict[str, Any]:
    ordered_animals = sorted(reporting.ICMS_TASK_MICE)
    validation_animal = ordered_animals[(ordered_animals.index(target) + 1) % len(ordered_animals)]
    selection_animals = sorted(set(donors) - {validation_animal})
    sorted_donors = sorted(donors)
    shared_sha = "e" * 64
    target_sha = "f" * 64
    return {
        "normal_selection": {"stage": "normal", "best_epoch": 0},
        "intervention_selection": {
            "stage": "intervention",
            "best_epoch": 0,
        },
        "normal_selection_training_animals": selection_animals,
        "intervention_inner_validation_animal": validation_animal,
        "validation_normal_gradient_to_shared_f": False,
        "shared_f_before_validation_normal_sha256": shared_sha,
        "shared_f_after_validation_normal_sha256": shared_sha,
        "shared_normal_stage_state_excluding_validation_adapters_sha256": (shared_sha),
        "validation_normal_adaptation": {
            f"{validation_animal}::adapter": {"stage": "target_adaptation"}
        },
        "intervention_selection_delta_audit": {
            "optimizer_steps": 1,
            "validation_animal": validation_animal,
            "validation_delta_requires_grad": False,
            "validation_delta_in_shrinkage": False,
            "validation_delta_centering_applied": False,
            "validation_delta_frozen_zero_during_selection": True,
            "identification_constraint": "exact_zero_mean_projection",
            "centering_group_animals": selection_animals,
            "centering_excluded_animals": [validation_animal],
            "projection_calls": 2,
            "validation_delta_l2_norm": 0.0,
            "maximum_validation_delta_shrinkage_term": 0.0,
            "prefit_projection_residual_norm": 0.0,
            "maximum_post_step_projection_residual_norm": 0.0,
        },
        "final_model_is_fresh": True,
        "final_normal_refit_selected_epochs": 1,
        "final_normal_refit": {
            "epochs": 1,
            "normal_refit_animals": sorted_donors,
            "normal_partitions": ["fit", "val"],
            "fresh_model": True,
        },
        "intervention_refit_all_donors_epochs": 1,
        "intervention_refit_delta_audit": {
            "optimizer_steps": 1,
            "identification_constraint": "exact_zero_mean_projection",
            "centering_group_animals": sorted_donors,
            "centering_group_count": len(sorted_donors),
            "projection_calls": 2,
            "refit_centering_covers_every_batch_donor": True,
            "final_donor_mean_delta_l2_norm": 0.0,
            "prefit_projection_residual_norm": 0.0,
            "maximum_post_step_projection_residual_norm": 0.0,
        },
        "target_normal_only_adaptation": {f"{target}::adapter": {"stage": "target_adaptation"}},
        "target_adaptation_nonadapter_state_before_sha256": target_sha,
        "target_adaptation_nonadapter_state_after_sha256": target_sha,
    }


def _write_prediction_archive(path: Path) -> str:
    arrays: dict[str, np.ndarray[Any, Any]] = {
        f"{_ARRAY_KEY}__condition_descriptors": np.zeros(
            (416, 1),
            dtype=np.float32,
        )
    }
    for method in reporting.ICMS_REPORT_METHOD_ORDER:
        for trajectory in (
            "neural_treated",
            "neural_control",
            "neural_effect",
            "behavior_treated",
            "behavior_control",
            "behavior_effect",
        ):
            arrays[f"{method}__{_ARRAY_KEY}__{trajectory}"] = np.zeros(
                (1,),
                dtype=np.float32,
            )
    arrays[f"proposed__{_ARRAY_KEY}__neural_effect_draws_condition_time"] = np.zeros(
        (64, 416, 1), dtype=np.float32
    )
    arrays[f"proposed__{_ARRAY_KEY}__behavior_effect_draws_condition_time"] = np.zeros(
        (64, 416, 1, 2), dtype=np.float32
    )
    np.savez_compressed(path, **arrays)
    return _authenticate_existing(path)


def _write_condition_metrics(path: Path) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=_CONDITION_COLUMNS)
    writer.writeheader()
    for method in reporting.ICMS_REPORT_METHOD_ORDER:
        writer.writerow(
            {
                "animal_id": _TARGET,
                "session_id": "synthetic-session",
                "session_key": _SESSION_KEY,
                "method": method,
                "condition": "1",
                "current_uA": "1.0",
                "electrode_rel_y_um": "0.0",
                "trials": "1",
                "randomized_causal_eligible": "True",
                "condition_primary_randomized_status": "EVALUATED",
                "session_primary_randomized_status": "EVALUATED",
                "block_validated": "True",
                "block_rule": "same-block",
                "missing_same_block_catches": "0",
                "valid_primary_neural_cells": "1",
                "valid_primary_behavior_cells": "1",
                "observed_neural_effect_rms": "0.1",
                "predicted_neural_effect_rms": "0.1",
            }
        )
    return _write_bytes(path, stream.getvalue().encode())


def _score_rows(
    *,
    eligible: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    status = "EVALUATED" if eligible else "NOT_EVALUATED"
    session: dict[str, dict[str, Any]] = {}
    aggregate: dict[str, dict[str, Any]] = {}
    for method_index, method in enumerate(reporting.ICMS_REPORT_METHOD_ORDER):
        offset = float(method_index) / 100.0
        session[method] = {
            "randomized_causal_eligible": eligible,
            "primary_randomized_status": status,
            "absolute_neural_nrmse": 1.0 + offset,
            "absolute_behavior_nrmse": 2.0 + offset,
            "neural_causal_skill": 0.3 + offset,
            "behavior_causal_skill": 0.2 + offset,
            "nonrandomized_iti_neural_skill": -0.1 + offset,
            "nonprimary_session_fallback_neural_skill": 0.05 + offset,
        }
        aggregate[method] = {
            "absolute_neural_nrmse_equal_session": 1.0 + offset,
            "absolute_behavior_nrmse_equal_session": 2.0 + offset,
            "neural_causal_skill_equal_session": (0.3 + offset if eligible else None),
            "behavior_causal_skill_equal_session": (0.2 + offset if eligible else None),
            "nonrandomized_iti_neural_skill_equal_session": -0.1 + offset,
            "nonprimary_session_fallback_neural_skill_equal_session": (0.05 + offset),
            "eligible_sessions": int(eligible),
            "primary_fold_status": status,
        }
    return session, aggregate


def _build_icms_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    target = _TARGET
    donors = [animal for animal in reporting.ICMS_TASK_MICE if animal != target]
    outer_mapping = {
        animal: [other for other in reporting.ICMS_TASK_MICE if other != animal]
        for animal in reporting.ICMS_TASK_MICE
    }
    h5_digests = {
        animal: hashlib.sha256(f"unopened:{animal}".encode()).hexdigest()
        for animal in reporting.ICMS_TASK_MICE
    }
    index = {
        "schema": "cadence-dandi-001868-index-v1",
        "dandiset_id": "001868",
        "dandiset_version": reporting.ICMS_DANDISET_VERSION,
        "split_unit": "animal_id",
        "totals": reporting.ICMS_INDEX_TOTALS,
        "animals": [
            {
                "animal_id": animal,
                "output_sha256": h5_digests[animal],
            }
            for animal in reporting.ICMS_TASK_MICE
        ],
    }
    raw_manifest = {
        "dandiset_id": "001868",
        "version": reporting.ICMS_DANDISET_VERSION,
        "task_mice": list(reporting.ICMS_TASK_MICE),
    }
    tracked_blobs = {
        "data/processed/dandi_001868/index.json": (
            json.dumps(index, sort_keys=True) + "\n"
        ).encode(),
        "configs/dandi_001868_assets.json": (
            json.dumps(raw_manifest, sort_keys=True) + "\n"
        ).encode(),
    }

    def fake_git_blob(relative_path: str, commit: str) -> bytes:
        assert commit == _ATTESTATION["commit"]
        return tracked_blobs[relative_path]

    monkeypatch.setattr(
        reporting,
        "_git_verify_annotated_attestation",
        lambda _attestation: None,
    )
    monkeypatch.setattr(
        reporting,
        "_require_canonical_source_path",
        lambda _path, _relative_path: None,
    )
    monkeypatch.setattr(reporting, "_git_blob_at_commit", fake_git_blob)

    index_sha = _sha256_bytes(tracked_blobs["data/processed/dandi_001868/index.json"])
    raw_manifest_sha = _sha256_bytes(tracked_blobs["configs/dandi_001868_assets.json"])
    source_root = str((tmp_path / "frozen-source").resolve())
    source_paths = {
        animal: f"/synthetic-unmaterialized/sub-{animal}.h5" for animal in reporting.ICMS_TASK_MICE
    }
    config = _full_config()
    config_sha = _sha256_bytes(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())
    canonical_scope = {
        "full_profile": True,
        "seed": reporting.LOCK_SEED,
        "ordered_report_methods": list(reporting.ICMS_REPORT_METHOD_ORDER),
        "canonical_target_order": list(reporting.ICMS_TASK_MICE),
        "outer_mapping": outer_mapping,
        "processed_index_sha256": index_sha,
        "raw_asset_manifest_sha256": raw_manifest_sha,
    }

    support_rows: list[dict[str, Any]] = []
    for animal in reporting.ICMS_TASK_MICE:
        adapter = _ADAPTER_ID if animal == target else f"{animal}::adapter"
        session_key = _SESSION_KEY if animal == target else f"{animal}::session"
        relative_path = f"support/{animal}/normal_support.npz"
        digest = _write_bytes(
            tmp_path / relative_path,
            f"synthetic normal support for {animal}".encode(),
        )
        support_rows.append(
            {
                "animal_id": animal,
                "adapter_id": adapter,
                "session_key": session_key,
                "path": relative_path,
                "sha256": digest,
            }
        )
    query_relative = f"queries/{target}/target_query.npz"
    query_sha = _write_bytes(
        tmp_path / query_relative,
        b"synthetic query descriptors only",
    )
    query_rows = [
        {
            "animal_id": target,
            "adapter_id": _ADAPTER_ID,
            "session_key": _SESSION_KEY,
            "path": query_relative,
            "sha256": query_sha,
        }
    ]

    seal = {
        "schema": "cadence-icms-physical-target-seal-v1",
        "target_animal": target,
        "target_path": source_paths[target],
        "processed_root": str((tmp_path / "synthetic-processed").resolve()),
        "fold_directory": str(tmp_path.resolve()),
        "canonical_relative_output": _CANONICAL_RELATIVE_OUTPUT,
        "expected_sha256": h5_digests[target],
        "sealed_mode": 0,
        "active": True,
        "original_mode": 0o640,
        "device_id": 7,
        "inode": 11,
    }
    seal_sha = _write_json(tmp_path / "target_seal.json", seal)
    prepare = {
        "schema": "cadence-icms-prepare-v1",
        "dataset": "DANDI:001868",
        "dataset_version": reporting.ICMS_DANDISET_VERSION,
        "run_mode": "biological",
        "canonical_relative_output": _CANONICAL_RELATIVE_OUTPUT,
        "target_animal": target,
        "donor_animals": donors,
        "canonical_target_order": list(reporting.ICMS_TASK_MICE),
        "canonical_outer_mapping": outer_mapping,
        "outer_scheme": "leave-one-animal-out",
        "intended_protocol_commit": _ATTESTATION["commit"],
        "required_preoutcome_tag": reporting.PREOUTCOME_TAG,
        "config": config,
        "config_sha256": config_sha,
        "freeze_attestation": {
            **_ATTESTATION,
            "source_root": source_root,
        },
        "canonical_provenance": {
            "source_root": source_root,
            "git_commit": _ATTESTATION["commit"],
            "preoutcome_tag": reporting.PREOUTCOME_TAG,
            "dandiset_id": "001868",
            "dandiset_version": reporting.ICMS_DANDISET_VERSION,
            "index_totals": reporting.ICMS_INDEX_TOTALS,
            "canonical_target_order": list(reporting.ICMS_TASK_MICE),
            "outer_mapping": outer_mapping,
            "processed_index": {
                "relative_path": ("data/processed/dandi_001868/index.json"),
                "sha256": index_sha,
                "git_blob_sha256": index_sha,
            },
            "raw_asset_manifest": {
                "relative_path": "configs/dandi_001868_assets.json",
                "sha256": raw_manifest_sha,
                "git_blob_sha256": raw_manifest_sha,
            },
            "provided_index_sha256": index_sha,
            "verified_h5_sha256": h5_digests,
        },
        "processed_source_sha256": h5_digests,
        "processed_source_paths": source_paths,
        "normal_supports": support_rows,
        "target_queries": query_rows,
        "target_seal_transaction_sha256": seal_sha,
        "physical_target_seal": {
            **seal,
            "path": "target_seal.json",
            "sha256": seal_sha,
        },
        "access_audit": {
            "prepare_target_stimulation_metadata_read": False,
            "prepare_target_stimulation_signals_read": False,
            "prepare_donor_stimulation_signals_read": False,
            "target_h5_read_permission_after_prepare": False,
        },
    }
    prepare_sha = _write_json(tmp_path / "prepare_manifest.json", prepare)

    prediction_sha = _write_prediction_archive(tmp_path / "predictions.npz")
    model_sha = _write_bytes(
        tmp_path / "frozen_models.pt",
        b"synthetic frozen model parameters",
    )
    prediction = {
        "schema": "cadence-icms-prediction-v1",
        "dataset": "DANDI:001868",
        "dataset_version": reporting.ICMS_DANDISET_VERSION,
        "run_mode": "biological",
        "canonical_relative_output": _CANONICAL_RELATIVE_OUTPUT,
        "target_animal": target,
        "donor_animals": donors,
        "freeze_attestation": _ATTESTATION,
        "config": config,
        "config_sha256": config_sha,
        "prepare_manifest_sha256": prepare_sha,
        "target_seal_transaction_sha256": seal_sha,
        "canonical_scope": canonical_scope,
        "methods": list(reporting.ICMS_REPORT_METHOD_ORDER),
        "verified_donor_source_sha256": {animal: h5_digests[animal] for animal in donors},
        "target_source_sha256_expected_but_not_opened": h5_digests[target],
        "sessions": [
            {
                "adapter_id": _ADAPTER_ID,
                "session_key": _SESSION_KEY,
                "array_key": _ARRAY_KEY,
                "query_sha256": query_sha,
                "condition_count": 416,
                "current_lattice_uA": [float(value) for value in range(1, 14)],
                "nearest_donor": donors[0],
            }
        ],
        "prediction_path": "predictions.npz",
        "prediction_sha256_before_target_open": prediction_sha,
        "model_path": "frozen_models.pt",
        "model_sha256": model_sha,
        "fit_audits": {
            method: _fit_audit(target, donors) for method in reporting.ICMS_REPORT_METHOD_ORDER[:4]
        },
        "access_audit": {
            "target_stimulation_metadata_read": False,
            "target_stimulation_outcomes_read": False,
            "target_stimulation_trials_in_fit_or_validation": 0,
            "prediction_hashed_before_target_container_open": True,
            "physical_target_seal_asserted_before_donor_open": True,
            "physical_target_h5_mode_during_predict": 0,
            "physical_target_seal_sha256": seal_sha,
            "session_specific_observation_maps": True,
            "encoder_receives_explicit_missingness_channels": True,
            "zero_filled_missing_bins_without_mask_channel": False,
            "donor_delta_grouping": "animal_id",
            "inner_validation_unit": "whole donor animal",
        },
        "uncertainty": {"split_conformal": "ABSENT_NOT_FIT"},
    }
    _write_json(
        tmp_path / "prediction_manifest.json",
        prediction,
    )

    sealed_outcomes_sha = _write_bytes(
        tmp_path / "sealed_target_outcomes.npz",
        b"synthetic sealed outcomes",
    )
    scored_trajectories_sha = _write_bytes(
        tmp_path / "scored_condition_trajectories.npz",
        b"synthetic scored trajectories",
    )
    condition_metrics_sha = _write_condition_metrics(tmp_path / "condition_metrics.csv")
    restore = {
        "schema": "cadence-icms-target-restore-v1",
        "target_animal": target,
        "target_path": source_paths[target],
        "sealed_mode": 0,
        "restored_mode": 0o640,
        "original_mode": 0o640,
        "original_mode_restored_exactly": True,
        "registry_retained_until_score_commit": True,
        "canonical_relative_output": _CANONICAL_RELATIVE_OUTPUT,
        "restoration_status": "PENDING_SCORE_COMMIT_FINALIZATION",
        "immutable_seal_sha256": seal_sha,
        "seal_transaction_sha256": seal_sha,
        "device_id": seal["device_id"],
        "inode": seal["inode"],
    }
    restore_sha = _write_json(tmp_path / "target_restore.json", restore)
    session_scores, animal_aggregate = _score_rows()
    metrics = {
        "schema": "cadence-icms-score-v1",
        "dataset": "DANDI:001868",
        "dataset_version": reporting.ICMS_DANDISET_VERSION,
        "run_mode": "biological",
        "canonical_relative_output": _CANONICAL_RELATIVE_OUTPUT,
        "target_animal": target,
        "freeze_attestation": _ATTESTATION,
        "canonical_scope": canonical_scope,
        "sealed_outcomes_sha256": sealed_outcomes_sha,
        "scored_condition_trajectories_sha256": scored_trajectories_sha,
        "condition_metrics_sha256": condition_metrics_sha,
        "physical_target_restore_sha256": restore_sha,
        "physical_target_restore": restore,
        "target_seal_transaction_sha256": seal_sha,
        "prediction_sha256_verified_before_target_open": prediction_sha,
        "target_source_sha256_verified_after_acknowledgement": (h5_digests[target]),
        "access_audit": {
            "query_and_support_hashes_verified_before_target_open": True,
            "prediction_hash_verified_before_target_open": True,
            "model_hash_verified_before_target_open": True,
            "target_stimulation_metadata_read_in_fit_or_predict": False,
            "target_stimulation_outcomes_read_in_fit_or_predict": False,
            "target_outcomes_opened_only_in_acknowledged_score": True,
            "target_h5_original_mode_restored_exactly": True,
            "immutable_target_seal_sha256": seal_sha,
            "sequential_loao_next_fold_ready": True,
        },
        "outcome_audit": {
            "post_onset_only": True,
            "target_outcomes_physically_separate_from_queries": True,
            "sessions": [
                {
                    "session_key": _SESSION_KEY,
                    "block_validated": True,
                    "randomized_catch_trials": 1,
                }
            ],
        },
        "causal_effect_eligibility": {
            "animal_eligible": True,
            "primary_fold_status": "EVALUATED",
            "design_maximum_primary_eligible_n": 5,
            "this_fold_primary_eligible_n": 1,
            "session_catch_fallback_in_primary": False,
            "absolute_trajectory_n": 6,
            "iti_is_randomized_counterfactual": False,
            "session_status": {_SESSION_KEY: "EVALUATED"},
        },
        "session_scores": {_SESSION_KEY: session_scores},
        "animal_aggregate": animal_aggregate,
        "uncertainty_audit": {
            "split_conformal": "ABSENT_NOT_FIT",
            "donor_draw_interval": ("uncalibrated_marginal_5_95_quantiles"),
            "simultaneous_coverage_exported": False,
            "conformal_coverage_exported": False,
        },
    }
    _write_json(tmp_path / "metrics.json", metrics)
    _write_completion(
        tmp_path,
        "prepare",
        "prepare_manifest.json",
        seal_transaction_sha256=seal_sha,
    )
    _write_completion(
        tmp_path,
        "predict",
        "prediction_manifest.json",
        seal_transaction_sha256=seal_sha,
    )
    score_completion_sha = _write_completion(
        tmp_path,
        "score",
        "metrics.json",
        seal_transaction_sha256=seal_sha,
    )
    restore_completion = {
        "schema": "cadence-icms-target-restore-completion-v1",
        "restored_after_score_commit": True,
        "canonical_relative_output": _CANONICAL_RELATIVE_OUTPUT,
        "target_animal": target,
        "target_path": source_paths[target],
        "restored_mode": seal["original_mode"],
        "target_sha256": h5_digests[target],
        "immutable_seal_sha256": seal_sha,
        "seal_transaction_sha256": seal_sha,
        "restore_audit_sha256": restore_sha,
        "score_completion_artifact": "metrics.json",
        "score_completion_sha256": score_completion_sha,
        "registry_retained_until_score_commit": True,
        "registry_removed_after_finalization": True,
    }
    _write_json(
        tmp_path / "target_restore_complete.json",
        restore_completion,
    )
    return {
        "directory": tmp_path,
        "metrics_path": tmp_path / "metrics.json",
        "metrics": metrics,
        "prepare": prepare,
        "prediction": prediction,
        "restore": restore,
        "restore_completion": restore_completion,
        "seal": seal,
        "seal_transaction_sha256": seal_sha,
    }


@pytest.fixture
def icms_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    return _build_icms_tree(tmp_path, monkeypatch)


def _validation(tree: dict[str, Any]) -> dict[str, Any]:
    return dict(
        reporting._validate_icms_locked_artifacts(
            tree["metrics"],
            tree["metrics_path"],
        )
    )


def _rewrite_stage(
    tree: dict[str, Any],
    stage: str,
    artifact_name: str,
    payload: dict[str, Any],
) -> None:
    _write_json(tree["directory"] / artifact_name, payload)
    completion_sha = _write_completion(
        tree["directory"],
        stage,
        artifact_name,
        seal_transaction_sha256=tree["seal_transaction_sha256"],
    )
    if stage == "score":
        tree["restore_completion"]["score_completion_sha256"] = completion_sha
        _write_json(
            tree["directory"] / "target_restore_complete.json",
            tree["restore_completion"],
        )


def test_canonical_randomized_fold_authenticates_without_h5(
    icms_tree: dict[str, Any],
) -> None:
    batch = reporting.adapt_icms_payload(
        icms_tree["metrics"],
        source_file=icms_tree["metrics_path"],
    )

    assert batch.artifact_validation["valid"] is True
    assert batch.artifact_validation["target_animal"] == _TARGET
    assert batch.artifact_validation["primary_estimand_evaluable"] is True
    assert len(batch.records) == len(reporting.ICMS_REPORT_METHOD_ORDER)
    assert all(record.randomized_estimand for record in batch.records)
    assert not list(icms_tree["directory"].rglob("*.h5"))


@pytest.mark.parametrize("mutation", ["missing", "tampered"])
def test_underscore_completion_fails_closed(
    icms_tree: dict[str, Any],
    mutation: str,
) -> None:
    completion = icms_tree["directory"] / "predict_complete.json"
    if mutation == "missing":
        completion.unlink()
    else:
        completion.write_bytes(completion.read_bytes() + b" ")

    validation = _validation(icms_tree)

    assert validation["valid"] is False
    assert "predict_complete.json" in validation["reason"]


@pytest.mark.parametrize("mutation", ["missing", "tampered", "rebound"])
def test_post_score_restoration_completion_fails_closed(
    icms_tree: dict[str, Any],
    mutation: str,
) -> None:
    completion = icms_tree["directory"] / "target_restore_complete.json"
    if mutation == "missing":
        completion.unlink()
    elif mutation == "tampered":
        completion.write_bytes(completion.read_bytes() + b" ")
    else:
        icms_tree["restore_completion"]["canonical_relative_output"] = "results/icms/loao-ICMS93"
        _write_json(completion, icms_tree["restore_completion"])

    validation = _validation(icms_tree)

    assert validation["valid"] is False
    assert (
        "target_restore_complete.json" in validation["reason"]
        or "restoration completion" in validation["reason"]
    )


@pytest.mark.parametrize("location", ["metrics", "completion"])
def test_icms_canonical_output_chain_rejects_rebinding(
    icms_tree: dict[str, Any],
    location: str,
) -> None:
    if location == "metrics":
        icms_tree["metrics"]["canonical_relative_output"] = "results/icms/loao-ICMS93"
        _rewrite_stage(
            icms_tree,
            "score",
            "metrics.json",
            icms_tree["metrics"],
        )
    else:
        completion_path = icms_tree["directory"] / "predict_complete.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["canonical_relative_output"] = "results/icms/loao-ICMS93"
        _write_json(completion_path, completion)

    validation = _validation(icms_tree)

    assert validation["valid"] is False
    assert (
        "canonical output" in validation["reason"] or "completion manifest" in validation["reason"]
    )


def test_caller_payload_must_equal_authenticated_source(
    icms_tree: dict[str, Any],
) -> None:
    caller_payload = copy.deepcopy(icms_tree["metrics"])
    caller_payload["target_animal"] = "ICMS93"

    validation = reporting._validate_icms_locked_artifacts(
        caller_payload,
        icms_tree["metrics_path"],
    )

    assert validation["valid"] is False
    assert "caller payload differs" in validation["reason"]


@pytest.mark.parametrize("mutation", ["run_mode", "method_order"])
def test_noncanonical_execution_scope_fails_closed(
    icms_tree: dict[str, Any],
    mutation: str,
) -> None:
    if mutation == "run_mode":
        icms_tree["metrics"]["run_mode"] = "development"
        _rewrite_stage(
            icms_tree,
            "score",
            "metrics.json",
            icms_tree["metrics"],
        )
    else:
        icms_tree["prediction"]["methods"] = list(reversed(reporting.ICMS_REPORT_METHOD_ORDER))
        _rewrite_stage(
            icms_tree,
            "predict",
            "prediction_manifest.json",
            icms_tree["prediction"],
        )

    validation = _validation(icms_tree)

    assert validation["valid"] is False
    assert (
        "biological mode" in validation["reason"]
        or "canonical locked scope" in validation["reason"]
    )


def _rebind_seal_chain(tree: dict[str, Any]) -> None:
    directory = tree["directory"]
    seal_sha = _write_json(directory / "target_seal.json", tree["seal"])
    tree["seal_transaction_sha256"] = seal_sha
    tree["prepare"]["target_seal_transaction_sha256"] = seal_sha
    tree["prepare"]["physical_target_seal"] = {
        **tree["seal"],
        "path": "target_seal.json",
        "sha256": seal_sha,
    }
    prepare_sha = _write_json(
        directory / "prepare_manifest.json",
        tree["prepare"],
    )
    tree["prediction"]["prepare_manifest_sha256"] = prepare_sha
    tree["prediction"]["target_seal_transaction_sha256"] = seal_sha
    tree["prediction"]["access_audit"]["physical_target_seal_sha256"] = seal_sha
    _write_json(
        directory / "prediction_manifest.json",
        tree["prediction"],
    )
    tree["restore"]["immutable_seal_sha256"] = seal_sha
    tree["restore"]["seal_transaction_sha256"] = seal_sha
    restore_sha = _write_json(
        directory / "target_restore.json",
        tree["restore"],
    )
    tree["metrics"]["physical_target_restore_sha256"] = restore_sha
    tree["metrics"]["physical_target_restore"] = tree["restore"]
    tree["metrics"]["target_seal_transaction_sha256"] = seal_sha
    tree["metrics"]["access_audit"]["immutable_target_seal_sha256"] = seal_sha
    _write_json(directory / "metrics.json", tree["metrics"])
    _write_completion(
        directory,
        "prepare",
        "prepare_manifest.json",
        seal_transaction_sha256=seal_sha,
    )
    _write_completion(
        directory,
        "predict",
        "prediction_manifest.json",
        seal_transaction_sha256=seal_sha,
    )
    score_completion_sha = _write_completion(
        directory,
        "score",
        "metrics.json",
        seal_transaction_sha256=seal_sha,
    )
    tree["restore_completion"].update(
        {
            "immutable_seal_sha256": seal_sha,
            "seal_transaction_sha256": seal_sha,
            "restore_audit_sha256": restore_sha,
            "score_completion_sha256": score_completion_sha,
        }
    )
    _write_json(
        directory / "target_restore_complete.json",
        tree["restore_completion"],
    )


@pytest.mark.parametrize("mutation", ["seal", "restore"])
def test_broken_physical_seal_or_restore_fails_closed(
    icms_tree: dict[str, Any],
    mutation: str,
) -> None:
    if mutation == "seal":
        icms_tree["seal"]["active"] = False
        _rebind_seal_chain(icms_tree)
    else:
        icms_tree["restore"]["original_mode_restored_exactly"] = False
        restore_sha = _write_json(
            icms_tree["directory"] / "target_restore.json",
            icms_tree["restore"],
        )
        icms_tree["metrics"]["physical_target_restore_sha256"] = restore_sha
        icms_tree["metrics"]["physical_target_restore"] = icms_tree["restore"]
        _rewrite_stage(
            icms_tree,
            "score",
            "metrics.json",
            icms_tree["metrics"],
        )

    validation = _validation(icms_tree)

    assert validation["valid"] is False
    if mutation == "seal":
        assert "target-seal transaction is invalid" in validation["reason"]
    else:
        assert "restore audit" in validation["reason"]


def test_randomized_fold_cannot_claim_false_primary_eligibility(
    icms_tree: dict[str, Any],
) -> None:
    session_scores, animal_aggregate = _score_rows(eligible=False)
    icms_tree["metrics"]["session_scores"] = {_SESSION_KEY: session_scores}
    icms_tree["metrics"]["animal_aggregate"] = animal_aggregate
    icms_tree["metrics"]["causal_effect_eligibility"].update(
        {
            "animal_eligible": False,
            "primary_fold_status": "NOT_EVALUATED",
            "this_fold_primary_eligible_n": 0,
            "session_status": {_SESSION_KEY: "NOT_EVALUATED"},
        }
    )
    _rewrite_stage(
        icms_tree,
        "score",
        "metrics.json",
        icms_tree["metrics"],
    )

    validation = _validation(icms_tree)

    assert validation["valid"] is False
    assert "eligibility audit" in validation["reason"]


def test_nonproducer_aggregate_field_is_rejected_before_reporting(
    icms_tree: dict[str, Any],
) -> None:
    icms_tree["metrics"]["animal_aggregate"]["proposed"]["post_hoc_headline_override"] = True
    _rewrite_stage(
        icms_tree,
        "score",
        "metrics.json",
        icms_tree["metrics"],
    )

    with pytest.raises(
        ValueError,
        match="non-producer aggregate fields",
    ):
        reporting.adapt_icms_payload(
            icms_tree["metrics"],
            source_file=icms_tree["metrics_path"],
        )
