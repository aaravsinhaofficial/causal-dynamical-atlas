from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import cadence.reporting as reporting_module
from cadence.experiments.teacher import (
    LEARNED_METHODS as TEACHER_EXPERIMENT_LEARNED_METHODS,
)
from cadence.experiments.teacher import (
    make_experiment_config,
    make_profile_teacher_config,
    teacher_experiment_scientific_sha256,
)
from cadence.reporting import (
    ALLEN_EXPECTED_METHODS,
    ALLEN_LOCKED_ANIMALS,
    ICMS_EXPECTED_METHODS,
    ICMS_RANDOMIZED_ANIMALS,
    LOCK_SEED,
    NOT_EVALUATED,
    PASS,
    PREOUTCOME_TAG,
    TEACHER_EXPECTED_METHODS,
    TEACHER_LEARNED_METHODS,
    TEACHER_LOCKED_WORLDS,
    TEACHER_TARGETS_PER_WORLD,
    adapt_allen_payload,
    adapt_icms_payload,
    adapt_teacher_payload,
    aggregate_batches,
    exact_binomial_lower_confidence_bound,
    exact_paired_sign_flip_test,
    strongest_baseline_envelope,
    write_report,
)
from cadence.teacher import load_teacher_config, teacher_config_sha256

_SYNTHETIC_ATTESTATION = {
    "commit": "a" * 40,
    "tag": PREOUTCOME_TAG,
    "tag_object": "b" * 40,
}


def _authenticated(batch, source: str, **validation_fields):
    return replace(
        batch,
        artifact_validation={
            "valid": True,
            "source_file": source,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "freeze_attestation": _SYNTHETIC_ATTESTATION,
            "checks": ["synthetic-unit-test-fixture"],
            **validation_fields,
        },
    )


def _write_json_sidecar(path: Path, payload: dict[str, object]) -> str:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )
    return digest


def _allen_payload(
    targets: list[str],
    *,
    fold: int | None,
    complete_evidence: bool = False,
    canonical_scope: bool = False,
    profile: str = "locked",
    schema: str = "cadence-allen-vbo-experiment-v1",
) -> dict[str, object]:
    methods: dict[str, dict[str, dict[str, float | bool]]] = {
        "proposed": {},
        "black_box": {},
        "condition_time": {},
        "target_oracle": {},
    }
    for target in targets:
        proposed: dict[str, float | bool] = {
            "neural_causal_skill": 0.60,
            "running_causal_skill": 0.50,
            # Secondary endpoints cannot affect the primary behavior gate.
            "pupil_causal_skill": -10.0,
            "lick_causal_skill": -10.0,
        }
        condition: dict[str, float | bool] = {
            "neural_causal_skill": 0.30,
            "running_causal_skill": 0.20,
        }
        if complete_evidence:
            proposed.update(
                {
                    "observed_neural_effect": -0.30,
                    "observed_running_effect": -0.20,
                    "neural_energy_score": 1.0,
                    "running_energy_score": 1.0,
                    "proper_score_draws_complete": True,
                    "proper_score_protocol": "frozen_full_predictive_v1",
                    "proper_score_draw_count": 64,
                    "proper_score_draws_sha256": "e" * 64,
                    "proper_score_metric": "multivariate_energy_score",
                    "neural_calibrated_simultaneous_coverage": True,
                    "running_calibrated_simultaneous_coverage": True,
                    "coverage_calibration_method": ("split_conformal_max_standardized_residual"),
                    "coverage_calibration_scope": (
                        "whole_animal_disjoint_from_fit_early_stopping_and_model_selection"
                    ),
                    "neural_predictive_draw_count": 64,
                    "neural_predictive_draw_protocol": (
                        "donor_bootstrap+target_normal_support_bootstrap+"
                        "donor_random_effects+process_variability"
                    ),
                    "neural_predictive_draws_sha256": "a" * 64,
                    "neural_simultaneous_band_nominal_level": 0.90,
                    "neural_simultaneous_band_mean_width": 0.5,
                    "neural_pointwise_coverage": 0.90,
                    "neural_simultaneous_band_sha256": "b" * 64,
                    "running_predictive_draw_count": 64,
                    "running_predictive_draw_protocol": (
                        "donor_bootstrap+target_normal_support_bootstrap+"
                        "donor_random_effects+process_variability"
                    ),
                    "running_predictive_draws_sha256": "c" * 64,
                    "running_simultaneous_band_nominal_level": 0.90,
                    "running_simultaneous_band_mean_width": 0.5,
                    "running_pointwise_coverage": 0.90,
                    "running_simultaneous_band_sha256": "d" * 64,
                    "pre_onset_null_passed": True,
                    "pre_onset_null_protocol": "frozen_equivalence_null_v1",
                    "pre_onset_null_artifact_sha256": "f" * 64,
                    "pre_onset_null_equivalence_margin": 0.10,
                    "pre_onset_null_ci_lower": -0.02,
                    "pre_onset_null_ci_upper": 0.02,
                    "pseudo_onset_null_passed": True,
                    "pseudo_onset_null_protocol": "frozen_equivalence_null_v1",
                    "pseudo_onset_null_artifact_sha256": "1" * 64,
                    "pseudo_onset_null_equivalence_margin": 0.10,
                    "pseudo_onset_null_ci_lower": -0.02,
                    "pseudo_onset_null_ci_upper": 0.02,
                }
            )
            condition.update(
                {
                    "neural_energy_score": 2.0,
                    "running_energy_score": 2.0,
                }
            )
        methods["proposed"][target] = proposed
        methods["black_box"][target] = {
            "neural_causal_skill": 0.20,
            "running_causal_skill": 0.10,
        }
        methods["condition_time"][target] = condition
        # This would dominate if target-outcome oracles were not excluded.
        methods["target_oracle"][target] = {
            "neural_causal_skill": 0.99,
            "running_causal_skill": 0.99,
        }
    if canonical_scope:
        methods.pop("target_oracle")
        for method in ALLEN_EXPECTED_METHODS - methods.keys():
            methods[method] = {
                target: {
                    "neural_causal_skill": 0.10,
                    "running_causal_skill": 0.05,
                }
                for target in targets
            }
    payload: dict[str, object] = {
        "schema": schema,
        "run_profile": profile,
        "fold": fold,
        "targets": targets,
        "animals": methods,
    }
    if complete_evidence:
        payload["headline_evidence"] = {
            "randomization_controls": {
                name: {
                    "status": "PASS",
                    "protocol": "frozen_randomization_control_v1",
                    "exact_or_preregistered": True,
                    "artifact_sha256": hashlib.sha256(name.encode()).hexdigest(),
                    "p_value": 0.01,
                }
                for name in (
                    "target_label_permutation",
                    "donor_semantic_shuffle",
                    "animal_adapter_shuffle",
                )
            }
        }
    return payload


def _teacher_payload(
    world_id: str,
    *,
    targets: int = 1,
    neural_skill: float = 0.4,
    cohort: str = "locked",
    canonical_scope: bool = False,
) -> dict[str, object]:
    proposed = {
        f"target-{index}": {
            "neural_condition_averaged_causal_skill": neural_skill,
            "behavior_condition_averaged_causal_skill": neural_skill - 0.1,
            "neural_pathwise_mean_causal_skill": neural_skill + 0.1,
        }
        for index in range(targets)
    }
    zero = {
        f"target-{index}": {
            "neural_condition_averaged_causal_skill": 0.0,
            "behavior_condition_averaged_causal_skill": 0.0,
            "neural_pathwise_mean_causal_skill": 0.0,
        }
        for index in range(targets)
    }
    methods = {
        "proposed": proposed,
        "zero_effect": zero,
    }
    if canonical_scope:
        for method in TEACHER_EXPECTED_METHODS - methods.keys():
            methods[method] = zero
    return {
        "schema_version": "cadence.teacher_experiment.v1",
        "world": {
            "world_id": world_id,
            "seed_partition": cohort,
        },
        "metrics_by_method_and_target": methods,
    }


def _teacher_nested_fit_fixture() -> tuple[dict[str, object], dict[str, object]]:
    topology_by_method: dict[str, object] = {}
    delta_by_method: dict[str, object] = {}
    stage_fits: dict[str, object] = {}
    for method in TEACHER_LEARNED_METHODS:
        topology = {
            "shared_normal_training_roles": ["train_donor"],
            "validation_donor_adapter_roles": ["validation_donor"],
            "validation_adapter_shared_parameter_max_abs_change": 0.0,
            "validation_interventions_used_for_gradient_steps_before_selection": False,
            "validation_intervention_delta_present_before_selection": False,
            "selection_training_delta_group_count": 10,
            "selected_normal_epochs": 3,
            "selected_intervention_epochs": 2,
            "final_refit_roles": ["train_donor", "validation_donor"],
            "final_refit_epoch_selection_from_refit_data": False,
        }
        delta = {
            "constraint": "exact_zero_mean_projection_after_every_optimizer_step",
            "training_group_count": 12,
            "final_mean_l2_norm": 0.0,
            "tolerance": 1e-7,
        }
        topology_by_method[method] = topology
        delta_by_method[method] = delta
        stage_fits[method] = {
            "normal": {"stage": "normal", "epochs_run": 3},
            "intervention": {
                "stage": "intervention",
                "epochs_run": 2,
                "donor_delta_identification": delta,
            },
            "selection": {
                "normal_train_donors_only": {
                    "stage": "normal",
                    "best_epoch": 2,
                },
                "validation_donor_normal_adaptation": {
                    "validation-0": {"stage": "target_adaptation"},
                    "validation-1": {"stage": "target_adaptation"},
                },
                "intervention_train_donors_validate_on_validation_donors": {
                    "stage": "intervention",
                    "best_epoch": 1,
                },
                "topology_audit": topology,
            },
            "targets": {
                f"target-{index}": {
                    "stage": "target_adaptation",
                    "neural_readout": {
                        "best_epoch": 2,
                        "validation_poisson_nll": 0.9,
                        "selected_ridge": 0.001,
                        "normal_rollout_design_rank": 8,
                        "normal_rollout_design_condition_number": 10.0,
                        "normal_rollout_anchor": 39,
                        "normal_rollout_support_max_abs_standardized": 4.0,
                        "query_max_abs_standardized": 5.0,
                        "query_coordinate_fraction_outside_normal_rollout_range": 0.1,
                    },
                }
                for index in range(TEACHER_TARGETS_PER_WORLD)
            },
        }
    protocol = {
        "target_intervention_batches_used_for_optimization": 0,
        "target_adaptation_splits": ["normal_fit", "normal_val"],
        "target_normal_audit_used_for_optimization": False,
        "post_onset_outcomes_mounted_as_inputs": False,
        "prediction_mode": "paired_open_loop",
        "prediction_initialization_sample": "onset_minus_1",
        "target_neural_readout": (
            "softplus-Poisson quasi-likelihood fit on frozen open-loop "
            "normal_fit rollouts, selected on frozen open-loop normal_val rollouts"
        ),
        "target_readout_contemporaneous_count_encoded_as_its_own_predictor": False,
        "nested_selection_topology": topology_by_method,
        "donor_delta_identification": delta_by_method,
    }
    return stage_fits, protocol


def test_exact_sign_flip_is_exhaustive_not_plus_one() -> None:
    assert exact_paired_sign_flip_test([1, 1, 1], alternative="greater") == pytest.approx(1 / 8)
    assert exact_paired_sign_flip_test([1, 1, 1], alternative="two-sided") == pytest.approx(2 / 8)
    # The 28-unit path uses meet-in-the-middle rather than a Monte Carlo fallback.
    assert exact_paired_sign_flip_test([1] * 28, alternative="greater") == pytest.approx(1 / 2**28)


@pytest.mark.parametrize(
    "entry",
    (
        {"ci_lower": 1.0, "ci_upper": -1.0},
        {"p_value": -1.0},
        {"p_value": 1.1},
    ),
)
def test_manipulation_gate_rejects_malformed_explicit_evidence(
    entry: dict[str, float],
) -> None:
    records = [
        reporting_module.AnimalResult(
            dataset="allen_vbo",
            cohort="locked",
            unit_id="mouse-0",
            animal_id="mouse-0",
            method="proposed",
            metrics={},
        )
    ]
    status, details = reporting_module._manipulation_component(
        records,
        ({"manipulation": {"neural": entry}},),
        "neural",
        bootstrap_repeats=100,
        seed=LOCK_SEED,
    )
    assert status == NOT_EVALUATED
    assert details["entries"][0]["reason"] == "effect interval or p-value missing"


def test_exact_coverage_bound_cannot_certify_five_of_five_at_point_eight() -> None:
    five_of_five = exact_binomial_lower_confidence_bound(5, 5)
    assert five_of_five == pytest.approx(0.05 ** (1 / 5))
    assert five_of_five < 0.80
    assert exact_binomial_lower_confidence_bound(28, 28) > 0.80

    records = [
        reporting_module.AnimalResult(
            dataset="icms",
            cohort="randomized_n5",
            unit_id=f"mouse-{index}",
            animal_id=f"mouse-{index}",
            method="proposed",
            metrics={
                "neural_calibrated_simultaneous_coverage": True,
                "behavior_calibrated_simultaneous_coverage": True,
                "coverage_calibration_method": "split_conformal_max_standardized_residual",
                "coverage_calibration_scope": (
                    "whole_animal_disjoint_from_fit_early_stopping_and_model_selection"
                ),
                "neural_predictive_draw_count": 64,
                "neural_predictive_draw_protocol": (
                    "donor_bootstrap+target_normal_support_bootstrap+"
                    "donor_random_effects+process_variability"
                ),
                "neural_predictive_draws_sha256": "a" * 64,
                "neural_simultaneous_band_nominal_level": 0.90,
                "neural_simultaneous_band_mean_width": 0.5,
                "neural_pointwise_coverage": 0.90,
                "neural_simultaneous_band_sha256": "b" * 64,
                "behavior_predictive_draw_count": 64,
                "behavior_predictive_draw_protocol": (
                    "donor_bootstrap+target_normal_support_bootstrap+"
                    "donor_random_effects+process_variability"
                ),
                "behavior_predictive_draws_sha256": "c" * 64,
                "behavior_simultaneous_band_nominal_level": 0.90,
                "behavior_simultaneous_band_mean_width": 0.5,
                "behavior_pointwise_coverage": 0.90,
                "behavior_simultaneous_band_sha256": "d" * 64,
            },
        )
        for index in range(5)
    ]
    status, details = reporting_module._coverage_component(records, ())

    assert status == NOT_EVALUATED
    for domain in ("neural", "behavior"):
        assert details[domain]["successes"] == 5
        assert details[domain]["n"] == 5
        assert details[domain]["ci_lower"] is None
        assert details[domain]["interval_method"] == "clopper_pearson_exact_one_sided"
        assert not details[domain]["predictive_draw_and_band_artifacts_complete"]
        assert all(
            audit["reason"] == reporting_module.SUPPLEMENTARY_GATE_ARTIFACT_REASON
            for audit in details[domain]["predictive_draw_and_band_artifacts"]
        )

    for record in records:
        record.metrics["coverage_calibration_scope"] = "validation_animals"
    stale_status, stale_details = reporting_module._coverage_component(records, ())
    assert stale_status == NOT_EVALUATED
    assert not stale_details["neural"]["calibration_provenance_complete"]

    for record in records:
        record.metrics["coverage_calibration_scope"] = (
            "whole_animal_disjoint_from_fit_early_stopping_and_model_selection"
        )
    records[0].metrics.pop("neural_predictive_draws_sha256")
    missing_status, missing_details = reporting_module._coverage_component(records, ())
    assert missing_status == NOT_EVALUATED
    assert missing_details["neural"]["ci_lower"] is None
    assert not missing_details["neural"]["predictive_draw_and_band_artifacts_complete"]


def test_teacher_nested_fit_audit_fails_closed_on_validation_leakage() -> None:
    stage_fits, protocol = _teacher_nested_fit_fixture()
    payload = {"stage_fits": stage_fits, "protocol_audit": protocol}
    reporting_module._validate_teacher_fit_audits(payload)
    topology = protocol["nested_selection_topology"]["proposed"]
    topology["validation_adapter_shared_parameter_max_abs_change"] = 1e-4
    with pytest.raises(ValueError, match="topology"):
        reporting_module._validate_teacher_fit_audits(payload)


def test_teacher_nested_fit_audit_requires_open_loop_readout_provenance() -> None:
    stage_fits, protocol = _teacher_nested_fit_fixture()
    payload = {"stage_fits": stage_fits, "protocol_audit": protocol}
    del stage_fits["proposed"]["targets"]["target-0"]["neural_readout"]
    with pytest.raises(ValueError, match="readout"):
        reporting_module._validate_teacher_fit_audits(payload)

    stage_fits, protocol = _teacher_nested_fit_fixture()
    payload = {"stage_fits": stage_fits, "protocol_audit": protocol}
    readout = stage_fits["black_box"]["targets"]["target-3"]["neural_readout"]
    readout["query_coordinate_fraction_outside_normal_rollout_range"] = float("nan")
    with pytest.raises(ValueError, match="readout"):
        reporting_module._validate_teacher_fit_audits(payload)

    stage_fits, protocol = _teacher_nested_fit_fixture()
    protocol["target_readout_contemporaneous_count_encoded_as_its_own_predictor"] = True
    with pytest.raises(ValueError, match="protocol"):
        reporting_module._validate_teacher_fit_audits(
            {"stage_fits": stage_fits, "protocol_audit": protocol}
        )


def test_allen_post_score_restoration_completion_is_hash_bound(
    tmp_path: Path,
) -> None:
    canonical = "results/allen-vbo/locked-fold-0"
    processed_root = (tmp_path / "processed").resolve()
    target_seals = {
        "legacy_combined": {
            "path": str(processed_root / "mouse_100" / "windows.npz"),
            "original_mode": "0600",
            "sealed_mode": "0000",
            "device_id": 7,
            "inode": 11,
            "sha256": "a" * 64,
        },
        "role_sealed": {
            "path": str(processed_root / "mouse_100" / "sealed_omission_outcomes.npz"),
            "original_mode": "0600",
            "sealed_mode": "0000",
            "device_id": 7,
            "inode": 12,
            "sha256": "b" * 64,
        },
        "experiment_sealed": {
            "path": str(tmp_path / "queries" / "mouse_100" / "sealed_outcomes.npz"),
            "original_mode": "0600",
            "sealed_mode": "0000",
            "device_id": 7,
            "inode": 13,
            "sha256": "c" * 64,
        },
    }
    transaction = {
        "schema": "cadence-allen-target-seal-transaction-v1",
        "fold": 0,
        "canonical_relative_output": canonical,
        "output_path": str(tmp_path.resolve()),
        "processed_root": str(processed_root),
        "targets": ["100"],
        "entries": [
            {"mouse": "100", "name": name, **target_seals[name]}
            for name in ("legacy_combined", "role_sealed", "experiment_sealed")
        ],
        "active": True,
        "restore_after_score_commit": True,
        "prepare_guard_sha256": "e" * 64,
    }
    transaction_text = json.dumps(transaction, indent=2, sort_keys=True) + "\n"
    transaction_sha = hashlib.sha256(transaction_text.encode()).hexdigest()
    preparation = {
        "fold": 0,
        "targets": ["100"],
        "target_seals": {"100": target_seals},
        "target_seal_transaction": {**transaction, "sha256": transaction_sha},
    }
    score_completion = {
        "stage": "score",
        "artifacts": {"metrics.json": "d" * 64},
        "metadata": {
            "canonical_processed_target_modes_restored": False,
            "target_mode_restoration_pending": True,
            "canonical_relative_output": canonical,
            "target_seal_transaction_sha256": transaction_sha,
        },
    }
    score_path = tmp_path / "score.complete.json"
    score_sha = _write_json_sidecar(score_path, score_completion)
    restored = {
        name: {
            "path": target_seals[name]["path"],
            "restored_mode": target_seals[name]["original_mode"],
            "sha256": target_seals[name]["sha256"],
        }
        for name in target_seals
    }
    restoration = {
        "schema": "cadence-allen-target-restore-completion-v1",
        "restored_after_score_commit": True,
        "eligible_for_later_donor_reuse": True,
        "canonical_relative_output": canonical,
        "score_completion_sha256": score_sha,
        "seal_transaction_sha256": transaction_sha,
        "mice": {"100": restored},
    }
    restoration_path = tmp_path / "restore.complete.json"
    _write_json_sidecar(restoration_path, restoration)

    authenticated = reporting_module._authenticate_allen_restoration_completion(
        tmp_path,
        preparation=preparation,
        score_completion=score_completion,
        canonical_relative_output=canonical,
    )
    assert authenticated == restoration

    restoration["seal_transaction_sha256"] = "f" * 64
    _write_json_sidecar(restoration_path, restoration)
    with pytest.raises(ValueError, match="restoration completion"):
        reporting_module._authenticate_allen_restoration_completion(
            tmp_path,
            preparation=preparation,
            score_completion=score_completion,
            canonical_relative_output=canonical,
        )


@pytest.mark.parametrize(
    "rebound",
    [
        "metrics",
        "prediction_run",
        "prediction_npz",
        "prepare_completion",
        "predict_completion",
        "score_completion",
        "prepare_artifact",
        "predict_artifact",
        "score_artifact",
        "restoration_plan",
        "restoration_completion",
    ],
)
def test_allen_transaction_references_reject_arbitrary_hex(rebound: str) -> None:
    transaction_sha = "a" * 64
    other_sha = "f" * 64
    payload = {
        "target_seal_transaction_sha256": transaction_sha,
        "protocol_audit": {
            "target_outcome_mode_restoration": {
                "seal_transaction_sha256": transaction_sha,
            }
        },
    }
    prediction = {"target_seal_transaction_sha256": transaction_sha}
    prediction_metadata = {"target_seal_transaction_sha256": transaction_sha}
    completion_artifacts = {
        "prepare": {"preparation.json": "1" * 64},
        "predict": {
            "predictions.npz": "2" * 64,
            "predictions.npz.sha256": "3" * 64,
            "prediction_run.json": "4" * 64,
        },
        "score": {"metrics.json": "5" * 64},
    }
    completions = {
        stage: {
            "artifacts": completion_artifacts[stage],
            "metadata": {
                "target_seal_transaction_sha256": transaction_sha,
            },
        }
        for stage in ("prepare", "predict", "score")
    }
    restoration_completion = {"seal_transaction_sha256": transaction_sha}

    if rebound == "metrics":
        payload["target_seal_transaction_sha256"] = other_sha
    elif rebound == "prediction_run":
        prediction["target_seal_transaction_sha256"] = other_sha
    elif rebound == "prediction_npz":
        prediction_metadata["target_seal_transaction_sha256"] = other_sha
    elif rebound.endswith("_completion") and rebound != "restoration_completion":
        stage = rebound.removesuffix("_completion")
        completions[stage]["metadata"]["target_seal_transaction_sha256"] = other_sha
    elif rebound.endswith("_artifact"):
        stage = rebound.removesuffix("_artifact")
        completions[stage]["artifacts"].clear()
    elif rebound == "restoration_plan":
        payload["protocol_audit"]["target_outcome_mode_restoration"]["seal_transaction_sha256"] = (
            other_sha
        )
    else:
        restoration_completion["seal_transaction_sha256"] = other_sha

    with pytest.raises(ValueError, match="transaction digest chain"):
        reporting_module._authenticate_allen_transaction_references(
            transaction_sha,
            payload=payload,
            prediction=prediction,
            prediction_metadata=prediction_metadata,
            completions=completions,
            restoration_completion=restoration_completion,
        )


def test_icms_transaction_chain_is_bound_to_authenticated_seal_bytes(
    tmp_path: Path,
) -> None:
    canonical = "results/icms/loao-ICMS92"
    seal = {
        "schema": "cadence-icms-physical-target-seal-v1",
        "target_animal": "ICMS92",
        "target_path": str((tmp_path / "processed" / "ICMS92.h5").resolve()),
        "processed_root": str((tmp_path / "processed").resolve()),
        "fold_directory": str(tmp_path.resolve()),
        "canonical_relative_output": canonical,
        "expected_sha256": "a" * 64,
        "device_id": 7,
        "inode": 11,
        "original_mode": 0o600,
        "sealed_mode": 0,
        "active": True,
    }
    seal_path = tmp_path / "target_seal.json"
    transaction_sha = _write_json_sidecar(seal_path, seal)
    prepare = {
        "target_animal": "ICMS92",
        "canonical_relative_output": canonical,
        "physical_target_seal": {
            **seal,
            "path": seal_path.name,
            "sha256": transaction_sha,
        },
        "target_seal_transaction_sha256": transaction_sha,
    }
    assert (
        reporting_module._authenticate_icms_seal_transaction(
            tmp_path,
            prepare,
            canonical,
        )
        == transaction_sha
    )

    prediction = {"target_seal_transaction_sha256": transaction_sha}
    metrics = {"target_seal_transaction_sha256": transaction_sha}
    completions = {
        stage: {"seal_transaction_sha256": transaction_sha}
        for stage in ("prepare", "predict", "score")
    }
    reporting_module._authenticate_icms_transaction_references(
        transaction_sha,
        prepare=prepare,
        prediction=prediction,
        payload=metrics,
        completions=completions,
    )

    rebound_prepare = {
        **prepare,
        "target_seal_transaction_sha256": "f" * 64,
    }
    with pytest.raises(ValueError, match="target-seal transaction"):
        reporting_module._authenticate_icms_seal_transaction(
            tmp_path,
            rebound_prepare,
            canonical,
        )

    metrics["target_seal_transaction_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="transaction digest chain"):
        reporting_module._authenticate_icms_transaction_references(
            transaction_sha,
            prepare=prepare,
            prediction=prediction,
            payload=metrics,
            completions=completions,
        )


def test_icms_restoration_completion_rejects_rebound_transaction(
    tmp_path: Path,
) -> None:
    canonical = "results/icms/loao-ICMS92"
    transaction_sha = "a" * 64
    target_path = str((tmp_path / "processed" / "ICMS92.h5").resolve())
    prepare = {
        "target_seal_transaction_sha256": transaction_sha,
        "physical_target_seal": {
            "target_animal": "ICMS92",
            "target_path": target_path,
            "original_mode": 0o600,
            "expected_sha256": "b" * 64,
            "sha256": transaction_sha,
        },
    }
    restore_sha = "c" * 64
    restore = {
        "seal_transaction_sha256": transaction_sha,
        "restoration_status": "PENDING_SCORE_COMMIT_FINALIZATION",
        "registry_retained_until_score_commit": True,
    }
    score_completion = {
        "artifact": "metrics.json",
        "seal_transaction_sha256": transaction_sha,
    }
    score_completion_sha = _write_json_sidecar(
        tmp_path / "score_complete.json",
        score_completion,
    )
    restoration = {
        "schema": "cadence-icms-target-restore-completion-v1",
        "restored_after_score_commit": True,
        "canonical_relative_output": canonical,
        "target_animal": "ICMS92",
        "target_path": target_path,
        "restored_mode": 0o600,
        "target_sha256": "b" * 64,
        "immutable_seal_sha256": transaction_sha,
        "seal_transaction_sha256": transaction_sha,
        "restore_audit_sha256": restore_sha,
        "score_completion_artifact": "metrics.json",
        "score_completion_sha256": score_completion_sha,
        "registry_retained_until_score_commit": True,
        "registry_removed_after_finalization": True,
    }
    restoration_path = tmp_path / "target_restore_complete.json"
    _write_json_sidecar(restoration_path, restoration)
    assert (
        reporting_module._authenticate_icms_restoration_completion(
            tmp_path,
            prepare=prepare,
            restore=restore,
            restore_sha256=restore_sha,
            score_completion=score_completion,
            canonical_relative_output=canonical,
        )
        == restoration
    )

    restoration["seal_transaction_sha256"] = "f" * 64
    _write_json_sidecar(restoration_path, restoration)
    with pytest.raises(ValueError, match="restoration completion"):
        reporting_module._authenticate_icms_restoration_completion(
            tmp_path,
            prepare=prepare,
            restore=restore,
            restore_sha256=restore_sha,
            score_completion=score_completion,
            canonical_relative_output=canonical,
        )


def test_bootstrap_gate_inputs_are_stably_ordered_by_unit() -> None:
    values = [1.0] * 25 + [-3.01] * 3
    records = [
        reporting_module.AnimalResult(
            dataset="allen_vbo",
            cohort="locked",
            unit_id=f"mouse-{index:02d}",
            animal_id=f"mouse-{index:02d}",
            method="proposed",
            metrics={"neural_causal_skill": value},
        )
        for index, value in enumerate(values)
    ]
    first = reporting_module._skill_component(
        records,
        "neural",
        bootstrap_repeats=20_000,
        seed=LOCK_SEED,
    )
    reversed_result = reporting_module._skill_component(
        list(reversed(records)),
        "neural",
        bootstrap_repeats=20_000,
        seed=LOCK_SEED,
    )
    assert first == reversed_result


def test_gain_gate_requires_every_frozen_comparator_endpoint() -> None:
    records: list[reporting_module.AnimalResult] = []
    for unit_index in range(2):
        unit_id = f"mouse-{unit_index}"
        for method in ALLEN_EXPECTED_METHODS:
            metrics = {
                "neural_causal_skill": 0.5 if method == "proposed" else 0.1,
            }
            if unit_index == 0 and method == "black_box":
                metrics = {}
            records.append(
                reporting_module.AnimalResult(
                    dataset="allen_vbo",
                    cohort="locked",
                    unit_id=unit_id,
                    animal_id=unit_id,
                    method=method,
                    metrics=metrics,
                )
            )
    status, details, _ = reporting_module._gain_component(
        records,
        "neural",
        bootstrap_repeats=100,
        seed=LOCK_SEED,
    )
    assert status == NOT_EVALUATED
    assert details["missing_or_nonfinite_baselines_by_unit"] == {"mouse-0": ["black_box"]}


def test_randomization_gate_rejects_positive_partial_unit_vectors() -> None:
    records = [
        reporting_module.AnimalResult(
            dataset="allen_vbo",
            cohort="locked",
            unit_id=f"mouse-{index:02d}",
            animal_id=f"mouse-{index:02d}",
            method="proposed",
            metrics={},
        )
        for index in range(28)
    ]
    partial_units = [record.unit_id for record in records[:5]]
    skills = {
        domain: {unit_id: 1.0 for unit_id in partial_units} for domain in ("neural", "behavior")
    }
    envelope = {
        domain: [{"unit_id": unit_id, "gain": 1.0} for unit_id in partial_units]
        for domain in ("neural", "behavior")
    }
    evidence = {
        "randomization_controls": {
            name: {
                "status": "PASS",
                "protocol": "frozen_randomization_control_v1",
                "exact_or_preregistered": True,
                "artifact_sha256": hashlib.sha256(name.encode()).hexdigest(),
                "p_value": 0.01,
            }
            for name in (
                "target_label_permutation",
                "donor_semantic_shuffle",
                "animal_adapter_shuffle",
            )
        }
    }
    status, details = reporting_module._randomization_component(
        records,
        (evidence,),
        skills,
        envelope,
    )
    assert status == NOT_EVALUATED
    assert details["neural_skill_greater_than_zero"] == NOT_EVALUATED
    assert details["behavior_gain_greater_than_zero"] == NOT_EVALUATED

    full_skills = {
        domain: {record.unit_id: 1.0 for record in records} for domain in ("neural", "behavior")
    }
    full_envelope = {
        domain: [{"unit_id": record.unit_id, "gain": 1.0} for record in records]
        for domain in ("neural", "behavior")
    }
    contradictory = json.loads(json.dumps(evidence))
    contradictory["randomization_controls"]["target_label_permutation"]["p_value"] = 0.99
    mismatch_status, mismatch_details = reporting_module._randomization_component(
        records,
        (contradictory,),
        full_skills,
        full_envelope,
    )
    assert mismatch_status == NOT_EVALUATED
    assert mismatch_details["target_label_permutation"] == NOT_EVALUATED


def test_incomplete_allen_never_evaluates_a_headline_gate() -> None:
    first = adapt_allen_payload(_allen_payload(["m1", "m2", "m3"], fold=0))
    second = adapt_allen_payload(_allen_payload(["m4", "m5"], fold=1))
    report = aggregate_batches([first, second], bootstrap_repeats=200, seed=7)
    analysis = report["analyses"]["allen_vbo:locked"]

    assert analysis["n_independent_units"] == 5
    assert analysis["n_proposed_units"] == 5
    assert len(analysis["target_rows"]) == 5
    assert len(analysis["animal_rows"]) == 5 * 4
    assert analysis["method_summaries"]["proposed"]["neural_causal_skill"] == {
        "n": 5,
        "estimate": pytest.approx(0.6),
        "ci_lower": pytest.approx(0.6),
        "ci_upper": pytest.approx(0.6),
        "confidence": 0.95,
        "bootstrap_repeats": 200,
        "equal_animal_weight": True,
        "endpoint": "neural",
        "endpoint_role": "primary",
    }
    assert analysis["cohort_completeness"]["complete"] is False
    assert analysis["cohort_completeness"]["expected_units"] == 28
    assert {gate["status"] for gate in analysis["conjunction"]["gates"]} == {NOT_EVALUATED}

    neural_envelope = strongest_baseline_envelope(first.records, domain="neural")
    assert [row["baseline_method"] for row in neural_envelope] == [
        "condition_time",
        "condition_time",
        "condition_time",
    ]
    assert [row["gain"] for row in neural_envelope] == pytest.approx([0.3, 0.3, 0.3])


def test_invented_supplementary_hashes_cannot_pass_frozen_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reporting_module,
        "_git_verify_clean_reporter_state",
        lambda attestation: {
            "schema": "cadence.reporter_attestation.v1",
            **dict(attestation),
            "clean_worktree": True,
        },
    )
    by_fold: dict[int, list[str]] = {fold: [] for fold in range(5)}
    ordered = sorted(
        ALLEN_LOCKED_ANIMALS,
        key=lambda mouse: (
            hashlib.sha256(f"{mouse}{LOCK_SEED}".encode()).hexdigest(),
            mouse,
        ),
    )
    for index, mouse in enumerate(ordered):
        by_fold[index % 5].append(mouse)
    batches = [
        _authenticated(
            adapt_allen_payload(
                _allen_payload(
                    targets,
                    fold=fold,
                    complete_evidence=True,
                    canonical_scope=True,
                )
            ),
            f"/synthetic/allen-fold-{fold}/metrics.json",
        )
        for fold, targets in by_fold.items()
    ]
    with pytest.raises(ValueError, match="preregistered"):
        aggregate_batches(batches, bootstrap_repeats=1, seed=1)
    report = aggregate_batches(batches)
    analysis = report["analyses"]["allen_vbo:locked"]
    assert analysis["cohort_completeness"]["complete"] is True
    conjunction = analysis["conjunction"]
    assert conjunction["overall_status"] == NOT_EVALUATED
    assert [gate["status"] for gate in conjunction["gates"]] == [
        PASS,
        PASS,
        PASS,
        PASS,
        NOT_EVALUATED,
        NOT_EVALUATED,
        NOT_EVALUATED,
        NOT_EVALUATED,
    ]
    assert (
        conjunction["gates"][4]["details"]["reason"]
        == reporting_module.SUPPLEMENTARY_GATE_ARTIFACT_REASON
    )
    randomization = conjunction["gates"][5]["details"]
    assert randomization["neural_skill_greater_than_zero"] == pytest.approx(1 / 2**28)
    assert randomization["behavior_gain_greater_than_zero"] == pytest.approx(1 / 2**28)
    assert randomization["target_label_permutation"] == NOT_EVALUATED


def test_complete_ids_without_authenticated_artifacts_fail_closed() -> None:
    by_fold: dict[int, list[str]] = {fold: [] for fold in range(5)}
    ordered = sorted(
        ALLEN_LOCKED_ANIMALS,
        key=lambda mouse: (
            hashlib.sha256(f"{mouse}{LOCK_SEED}".encode()).hexdigest(),
            mouse,
        ),
    )
    for index, mouse in enumerate(ordered):
        by_fold[index % 5].append(mouse)
    batches = [
        adapt_allen_payload(
            _allen_payload(
                targets,
                fold=fold,
                canonical_scope=True,
            )
        )
        for fold, targets in by_fold.items()
    ]
    analysis = aggregate_batches(batches, bootstrap_repeats=20)["analyses"]["allen_vbo:locked"]
    scope = analysis["cohort_completeness"]["locked_scope_validation"]
    assert analysis["cohort_completeness"]["complete"] is False
    assert scope["valid"] is False
    assert len(scope["artifact_failures"]) == 5
    assert {gate["status"] for gate in analysis["conjunction"]["gates"]} == {NOT_EVALUATED}


def test_authenticated_ids_with_incomplete_method_matrix_fail_closed() -> None:
    by_fold: dict[int, list[str]] = {fold: [] for fold in range(5)}
    ordered = sorted(
        ALLEN_LOCKED_ANIMALS,
        key=lambda mouse: (
            hashlib.sha256(f"{mouse}{LOCK_SEED}".encode()).hexdigest(),
            mouse,
        ),
    )
    for index, mouse in enumerate(ordered):
        by_fold[index % 5].append(mouse)
    batches = [
        _authenticated(
            adapt_allen_payload(_allen_payload(targets, fold=fold)),
            f"/synthetic/incomplete-methods-fold-{fold}/metrics.json",
        )
        for fold, targets in by_fold.items()
    ]
    analysis = aggregate_batches(batches, bootstrap_repeats=20)["analyses"]["allen_vbo:locked"]
    scope = analysis["cohort_completeness"]["locked_scope_validation"]
    assert analysis["cohort_completeness"]["complete"] is False
    assert scope["method_mismatches"]


def test_sealed_v2_adapter_ignores_injected_headline_statuses() -> None:
    payload = _allen_payload(
        ["423606"],
        fold=None,
        complete_evidence=True,
        profile="development",
        schema="cadence-allen-vbo-experiment-v2",
    )
    batch = adapt_allen_payload(payload)
    assert payload["headline_evidence"]
    assert batch.headline_evidence == {}


def test_allen_v2_development_is_explicitly_non_headline() -> None:
    batch = adapt_allen_payload(
        _allen_payload(
            ["423606"],
            fold=None,
            profile="development",
            schema="cadence-allen-vbo-experiment-v2",
        )
    )
    analysis = aggregate_batches([batch], bootstrap_repeats=20)["analyses"]["allen_vbo:development"]
    assert analysis["cohort_completeness"]["headline_profile"] is False
    assert analysis["conjunction"]["overall_status"] == NOT_EVALUATED
    assert {gate["status"] for gate in analysis["conjunction"]["gates"]} == {NOT_EVALUATED}


def test_teacher_world_qualifies_the_repeated_target_id() -> None:
    batches = [
        adapt_teacher_payload(_teacher_payload("world-a")),
        adapt_teacher_payload(_teacher_payload("world-b")),
    ]
    report = aggregate_batches(batches, bootstrap_repeats=50)
    analysis = report["analyses"]["teacher:locked"]
    assert analysis["n_independent_units"] == 2
    assert analysis["n_nested_target_units"] == 2
    assert analysis["replication_unit"] == "teacher_world"
    assert (
        analysis["method_summaries"]["proposed"]["neural_condition_averaged_causal_skill"]["n"] == 2
    )
    assert {row["unit_id"] for row in analysis["animal_rows"] if row["method"] == "proposed"} == {
        "world-a/target-0",
        "world-b/target-0",
    }
    assert "conditional_mean_endpoints" not in analysis["endpoint_hierarchy"]
    assert "realized_path_diagnostics" in analysis["endpoint_hierarchy"]


def test_teacher_locked_inference_averages_targets_within_world(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reporting_module,
        "_git_verify_clean_reporter_state",
        lambda attestation: {
            "schema": "cadence.reporter_attestation.v1",
            **dict(attestation),
            "clean_worktree": True,
        },
    )
    base = [
        _authenticated(
            adapt_teacher_payload(
                _teacher_payload(
                    reporting_module.TEACHER_LOCKED_WORLD_IDS[index],
                    targets=TEACHER_TARGETS_PER_WORLD,
                    neural_skill=0.2 + index / 100,
                    canonical_scope=True,
                )
            ),
            f"/synthetic/teacher-world-{index:02d}/metrics.json",
            seed_index=index,
            world_id=reporting_module.TEACHER_LOCKED_WORLD_IDS[index],
        )
        for index in range(TEACHER_LOCKED_WORLDS)
    ]
    duplicated = [
        _authenticated(
            adapt_teacher_payload(
                _teacher_payload(
                    reporting_module.TEACHER_LOCKED_WORLD_IDS[index],
                    targets=2 * TEACHER_TARGETS_PER_WORLD,
                    neural_skill=0.2 + index / 100,
                    canonical_scope=True,
                )
            ),
            f"/synthetic/teacher-world-{index:02d}/metrics.json",
            seed_index=index,
            world_id=reporting_module.TEACHER_LOCKED_WORLD_IDS[index],
        )
        for index in range(TEACHER_LOCKED_WORLDS)
    ]
    base_analysis = aggregate_batches(base)["analyses"]["teacher:locked"]
    duplicate_analysis = aggregate_batches(duplicated)["analyses"]["teacher:locked"]
    metric = "neural_condition_averaged_causal_skill"

    assert base_analysis["n_independent_units"] == TEACHER_LOCKED_WORLDS
    assert base_analysis["n_nested_target_units"] == 80
    assert base_analysis["method_summaries"]["proposed"][metric]["n"] == 20
    assert base_analysis["cohort_completeness"]["complete"] is True
    assert base_analysis["cohort_completeness"]["headline_profile"] is False
    assert base_analysis["cohort_completeness"]["seed_material_public"] is True
    assert base_analysis["conjunction"]["overall_status"] == NOT_EVALUATED
    assert {gate["status"] for gate in base_analysis["conjunction"]["gates"]} == {NOT_EVALUATED}
    assert duplicate_analysis["n_independent_units"] == TEACHER_LOCKED_WORLDS
    assert duplicate_analysis["n_nested_target_units"] == 160
    assert duplicate_analysis["method_summaries"]["proposed"][metric]["n"] == 20
    assert duplicate_analysis["cohort_completeness"]["complete"] is False
    for key in ("estimate", "ci_lower", "ci_upper"):
        assert duplicate_analysis["method_summaries"]["proposed"][metric][key] == (
            pytest.approx(base_analysis["method_summaries"]["proposed"][metric][key])
        )


def test_teacher_artifact_authentication_requires_complete_canonical_methods(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze = {
        "commit": "a" * 40,
        "tag": "pre-outcome-v1.0.0",
        "tag_object": "b" * 40,
    }
    source_root = Path(__file__).resolve().parents[1]
    teacher_config = make_profile_teacher_config(
        load_teacher_config(source_root / "configs" / "teacher.yaml"),
        "full",
    )
    experiment_config = make_experiment_config(
        "full",
        seed=0,
        device="cpu",
        learned_methods=TEACHER_EXPERIMENT_LEARNED_METHODS,
    )
    teacher_sha = teacher_config_sha256(teacher_config)
    experiment_sha = teacher_experiment_scientific_sha256(experiment_config)
    canonical_relative_output = "results/teacher-locked/full/locked-seed-00"
    metrics_by_method = {
        method: {
            f"target-{target_index}": {
                "neural_condition_averaged_causal_skill": 0.1,
                "behavior_condition_averaged_causal_skill": 0.1,
            }
            for target_index in range(TEACHER_TARGETS_PER_WORLD)
        }
        for method in TEACHER_EXPECTED_METHODS
    }
    prediction_metadata = {
        "preoutcome_freeze": freeze,
        "teacher_config_sha256": teacher_sha,
        "teacher_experiment_scientific_sha256": experiment_sha,
        "learned_methods": list(TEACHER_LEARNED_METHODS),
        "canonical_learned_method_set_complete": True,
        "canonical_relative_output": canonical_relative_output,
    }
    predictions = tmp_path / "predictions.npz"
    np.savez_compressed(
        predictions,
        metadata_json=np.asarray(json.dumps(prediction_metadata, sort_keys=True)),
    )
    prediction_sha = hashlib.sha256(predictions.read_bytes()).hexdigest()
    predictions.with_suffix(".npz.sha256").write_text(
        f"{prediction_sha}  {predictions.name}\n",
        encoding="utf-8",
    )
    stage_fits, nested_protocol = _teacher_nested_fit_fixture()
    payload = {
        "schema_version": "cadence.teacher_experiment.v1",
        "world": {
            "world_id": reporting_module.TEACHER_LOCKED_WORLD_IDS[0],
            "seed_partition": "locked",
            "seed_index": 0,
            "world_seed": reporting_module.TEACHER_LOCKED_SEEDS[0],
            "seed_material_public": True,
            "eligible_for_biological_headline_conjunction": False,
            "teacher_config_sha256": teacher_sha,
        },
        "learned_methods": list(TEACHER_LEARNED_METHODS),
        "canonical_learned_method_set_complete": True,
        "reported_methods": sorted(TEACHER_EXPECTED_METHODS),
        "canonical_relative_output": canonical_relative_output,
        "experiment_config": experiment_config.to_mapping(),
        "protocol_audit": {
            **nested_protocol,
            "canonical_relative_output": canonical_relative_output,
            "preoutcome_freeze": freeze,
            "prediction_sha256_before_score": prediction_sha,
            "teacher_config_sha256": teacher_sha,
            "teacher_experiment_scientific_sha256": experiment_sha,
        },
        "stage_fits": stage_fits,
        "metrics_by_method_and_target": metrics_by_method,
    }
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics_sha = hashlib.sha256(metrics.read_bytes()).hexdigest()
    metrics.with_suffix(".json.sha256").write_text(
        f"{metrics_sha}  {metrics.name}\n",
        encoding="utf-8",
    )
    (tmp_path / "completion.json").write_text(
        json.dumps(
            {
                "schema_version": "cadence.teacher_completion.v1",
                "preoutcome_freeze": freeze,
                "teacher_config_sha256": teacher_sha,
                "teacher_experiment_scientific_sha256": experiment_sha,
                "learned_methods": list(TEACHER_LEARNED_METHODS),
                "canonical_learned_method_set_complete": True,
                "canonical_relative_output": canonical_relative_output,
                "reported_methods": sorted(TEACHER_EXPECTED_METHODS),
                "artifacts": {
                    metrics.name: metrics_sha,
                    predictions.name: prediction_sha,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        reporting_module,
        "_git_verify_annotated_attestation",
        lambda _attestation: None,
    )
    monkeypatch.setattr(
        reporting_module,
        "_require_canonical_source_path",
        lambda _path, _relative_path: None,
    )

    valid = adapt_teacher_payload(payload, source_file=metrics)
    assert valid.artifact_validation["valid"] is True
    assert "canonical_method_scope" in valid.artifact_validation["checks"]

    def rewrite_authenticated_payload() -> None:
        metrics.write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        changed_sha = hashlib.sha256(metrics.read_bytes()).hexdigest()
        metrics.with_suffix(".json.sha256").write_text(
            f"{changed_sha}  {metrics.name}\n",
            encoding="utf-8",
        )
        completion = json.loads((tmp_path / "completion.json").read_text(encoding="utf-8"))
        completion["artifacts"][metrics.name] = changed_sha
        (tmp_path / "completion.json").write_text(
            json.dumps(completion, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    payload["experiment_config"]["hidden_dim"] = 95
    rewrite_authenticated_payload()
    wrong_config = adapt_teacher_payload(payload, source_file=metrics)
    assert wrong_config.artifact_validation["valid"] is False
    assert "exact frozen full configuration" in wrong_config.artifact_validation["reason"]

    payload["experiment_config"] = experiment_config.to_mapping()
    payload["learned_methods"] = ["proposed"]
    rewrite_authenticated_payload()
    invalid = adapt_teacher_payload(payload, source_file=metrics)
    assert invalid.artifact_validation["valid"] is False
    assert "learned-method scope" in invalid.artifact_validation["reason"]


def test_teacher_development_is_explicitly_non_headline() -> None:
    analysis = aggregate_batches(
        [adapt_teacher_payload(_teacher_payload("development-world", cohort="development"))],
        bootstrap_repeats=20,
    )["analyses"]["teacher:development"]
    assert analysis["cohort_completeness"]["headline_profile"] is False
    assert analysis["cohort_completeness"]["procedural_evaluation"] is True
    assert analysis["conjunction"]["overall_status"] == NOT_EVALUATED


def test_incomplete_teacher_locked_cohort_never_evaluates_gates() -> None:
    batches = [
        adapt_teacher_payload(
            _teacher_payload(
                f"world-{index:02d}",
                targets=TEACHER_TARGETS_PER_WORLD,
            )
        )
        for index in range(TEACHER_LOCKED_WORLDS - 1)
    ]
    analysis = aggregate_batches(batches, bootstrap_repeats=20)["analyses"]["teacher:locked"]
    assert analysis["n_independent_units"] == 19
    assert analysis["cohort_completeness"]["complete"] is False
    assert {gate["status"] for gate in analysis["conjunction"]["gates"]} == {NOT_EVALUATED}


def test_icms_randomized_n5_and_icms83_absolute_only_are_separate() -> None:
    animals = ["ICMS83", "ICMS92", "ICMS93", "ICMS98", "ICMS100", "ICMS101"]
    payload = {
        "schema": "cadence-icms-experiment-v1",
        "animals": {
            method: {
                animal: {
                    "neural_causal_skill": value,
                    "behavior_causal_skill": value,
                }
                for animal in animals
            }
            for method, value in (("proposed", 0.5), ("black_box", 0.1))
        },
    }
    report = aggregate_batches(
        [adapt_icms_payload(payload)],
        bootstrap_repeats=100,
    )
    randomized = report["analyses"]["icms:randomized_n5"]
    absolute = report["analyses"]["icms:absolute_only"]
    assert randomized["n_independent_units"] == 5
    assert randomized["expected_n"] == 5
    assert absolute["n_independent_units"] == 1
    assert absolute["conjunction"]["overall_status"] == NOT_EVALUATED
    assert {gate["status"] for gate in absolute["conjunction"]["gates"]} == {NOT_EVALUATED}
    assert all(not row["randomized_estimand"] for row in absolute["animal_rows"])


def test_icms_fold_native_animal_aggregate_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reporting_module,
        "_git_verify_clean_reporter_state",
        lambda attestation: {
            "schema": "cadence.reporter_attestation.v1",
            **dict(attestation),
            "clean_worktree": True,
        },
    )
    batches = [
        _authenticated(
            adapt_icms_payload(
                {
                    "schema": "cadence-icms-score-v1",
                    "target_animal": animal,
                    "animal_aggregate": {
                        method: {
                            "neural_causal_skill_equal_session": (
                                0.4
                                if method == "proposed"
                                else (0.1 if method == "condition_time" else 0.05)
                            ),
                            "behavior_causal_skill_equal_session": (
                                0.3
                                if method == "proposed"
                                else (0.1 if method == "condition_time" else 0.05)
                            ),
                        }
                        for method in ICMS_EXPECTED_METHODS
                    },
                }
            ),
            f"/synthetic/icms-{animal}/metrics.json",
            target_animal=animal,
            primary_estimand_evaluable=True,
        )
        for animal in sorted(ICMS_RANDOMIZED_ANIMALS)
    ]
    report = aggregate_batches(batches)
    analysis = report["analyses"]["icms:randomized_n5"]
    assert analysis["n_independent_units"] == 5
    assert analysis["cohort_completeness"]["complete"] is True
    neural = analysis["strongest_baseline_envelope"]["neural"][0]
    assert neural["baseline_method"] == "condition_time"
    assert neural["gain"] == pytest.approx(0.3)


def test_incomplete_icms_randomized_cohort_never_evaluates_gates() -> None:
    animals = sorted(ICMS_RANDOMIZED_ANIMALS)[:-1]
    payload = {
        "schema": "cadence-icms-experiment-v1",
        "animals": {
            method: {
                animal: {
                    "neural_causal_skill": value,
                    "behavior_causal_skill": value,
                }
                for animal in animals
            }
            for method, value in (("proposed", 0.5), ("black_box", 0.1))
        },
    }
    analysis = aggregate_batches([adapt_icms_payload(payload)], bootstrap_repeats=20)["analyses"][
        "icms:randomized_n5"
    ]
    assert analysis["cohort_completeness"]["complete"] is False
    assert {gate["status"] for gate in analysis["conjunction"]["gates"]} == {NOT_EVALUATED}


def test_writer_emits_json_long_csv_and_both_latex_tables(tmp_path: Path) -> None:
    report = aggregate_batches(
        [adapt_allen_payload(_allen_payload(["m1", "m2"], fold=0))],
        bootstrap_repeats=20,
    )
    output = tmp_path / "report"
    paths = write_report(report, output)
    assert set(paths) == {
        "summary",
        "long_csv",
        "latex_summary",
        "conjunction_csv",
        "latex_conjunction",
        "completion",
    }
    assert json.loads(paths["summary"].read_text())["schema_version"] == ("cadence.reporting.v1")
    with paths["long_csv"].open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    running = [
        row
        for row in rows
        if row["method"] == "proposed" and row["metric"] == "running_causal_skill"
    ]
    pupil = [
        row for row in rows if row["method"] == "proposed" and row["metric"] == "pupil_causal_skill"
    ]
    assert {row["endpoint_role"] for row in running} == {"primary"}
    assert {row["endpoint_role"] for row in pupil} == {"secondary"}
    assert "\\begin{tabular}" in paths["latex_summary"].read_text()
    assert "NOT\\_EVALUATED" in paths["latex_conjunction"].read_text()
    completion = json.loads(paths["completion"].read_text())
    assert completion["append_only"] is True
    for name, digest in completion["artifacts"].items():
        artifact = output / name
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == digest
        assert artifact.with_suffix(artifact.suffix + ".sha256").is_file()
    assert paths["completion"].with_suffix(".json.sha256").is_file()
    with pytest.raises(FileExistsError):
        write_report(report, output)
    with pytest.raises(ValueError, match="overwrite is disabled"):
        write_report(report, tmp_path / "other", overwrite=True)
    assert len(list(csv.DictReader(paths["conjunction_csv"].open()))) == 8


def test_duplicate_mouse_across_allen_fold_files_is_rejected() -> None:
    batches = [
        adapt_allen_payload(_allen_payload(["m1", "m2"], fold=0)),
        adapt_allen_payload(_allen_payload(["m2", "m3"], fold=1)),
    ]
    with pytest.raises(ValueError, match="overweight a target"):
        aggregate_batches(batches, bootstrap_repeats=20)
