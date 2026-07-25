from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cadence.baselines import BlackBoxMetaGRU
from cadence.data.splits import LeakageError
from cadence.experiments import allen as allen_module
from cadence.experiments.allen import (
    DEVELOPMENT_MICE,
    LEARNED_METHODS,
    MatchedInterventionBatch,
    MouseScaler,
    _add_zero_donor_delta,
    _assert_target_outcomes_sealed,
    _augment_neural_with_masks,
    _canonical_optimization_sha256,
    _expected_stratum_skill,
    _fit_stage,
    _intervention_loss,
    _load_query,
    _locked_append_only_gate,
    _normal_loss,
    _open_experiment_sealed_for_score,
    _read_stage_completion,
    _recover_allen_locked_stage,
    _require_canonical_locked_output,
    _restore_target_outcomes_after_score,
    _run_configuration_sha256,
    _score_allen_stage,
    _seal_target_outcomes,
    _selected_report_methods,
    _stage_completion_paths,
    _tracked_file_audit,
    _validate_locked_configuration,
    _verify_completed_artifact,
    _verify_locked_processed_inputs,
    _verify_locked_protocol_scope,
    _write_stage_completion,
    build_flash_inputs,
    locked_fold_table,
    make_allen_config,
    match_control_indices,
    resolve_run_mice,
    run_allen_experiment,
    split_normal_presentations,
    transition_index,
)
from cadence.model import HierarchicalControlledSSM, SequenceBatch
from cadence.protocol import FreezeAttestation, ProtocolViolation
from cadence.training import FitConfig

FROZEN_MANIFEST = Path("data/manifests/allen_vbo_slc17a7_visp175_familiar_active_v1.1.0.json")


def test_development_and_locked_mice_are_frozen_disjoint_partitions() -> None:
    table = locked_fold_table(FROZEN_MANIFEST)
    manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    all_mice = {str(entry["mouse_id"]) for entry in manifest["nwb_files"]}

    assert len(table) == 28
    assert set(table["mouse_id"]).isdisjoint(DEVELOPMENT_MICE)
    assert set(table["mouse_id"]) | set(DEVELOPMENT_MICE) == all_mice
    assert sorted(table.groupby("outer_fold").size().tolist()) == [5, 5, 6, 6, 6]


def test_locked_outcomes_require_explicit_post_freeze_acknowledgement() -> None:
    with pytest.raises(LeakageError, match="locked outcomes remain sealed"):
        resolve_run_mice(FROZEN_MANIFEST, profile="locked", fold=0)

    donors, targets = resolve_run_mice(
        FROZEN_MANIFEST,
        profile="locked",
        fold=0,
        acknowledge_locked=True,
    )
    assert set(donors).isdisjoint(targets)
    assert len(donors) + len(targets) == 28


def test_locked_run_rejects_single_process_all_stage() -> None:
    with pytest.raises(LeakageError, match="separate --stage"):
        run_allen_experiment(
            processed_root="unused",
            manifest_path=FROZEN_MANIFEST,
            output_directory="unused",
            run_profile="locked",
            optimization=make_allen_config("smoke", methods=("proposed",)),
            fold=0,
            acknowledge_locked=True,
            stage="all",
        )


@pytest.mark.parametrize(
    ("optimization", "seed", "match"),
    [
        (make_allen_config("smoke"), 0, "optimization='full'"),
        (make_allen_config("full", methods=("proposed",)), 0, "all frozen learned"),
        (make_allen_config("full"), 1, "seed=0"),
        (
            replace(make_allen_config("full"), intervention_rank=1),
            0,
            "intervention_rank=2",
        ),
    ],
)
def test_locked_api_rejects_noncanonical_optimization_scope(
    optimization: object,
    seed: int,
    match: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(ProtocolViolation, match=match):
        run_allen_experiment(
            processed_root=Path("data/processed/allen_vbo"),
            manifest_path=FROZEN_MANIFEST,
            output_directory=tmp_path / "output",
            run_profile="locked",
            optimization=optimization,  # type: ignore[arg-type]
            fold=0,
            acknowledge_locked=True,
            seed=seed,
            stage="prepare",
        )


def test_locked_full_configuration_and_device_independent_digest_are_frozen() -> None:
    cpu = make_allen_config("full", seed=0, device="cpu")
    cuda = make_allen_config("full", seed=0, device="cuda")
    cpu_audit = _validate_locked_configuration(cpu, seed=0)
    cuda_audit = _validate_locked_configuration(cuda, seed=0)

    assert cpu.intervention_rank == cuda.intervention_rank == 2
    assert cpu.learned_methods == cuda.learned_methods == LEARNED_METHODS
    assert cpu_audit["stage_seeds"] == {
        "normal": 11,
        "intervention": 23,
        "target_adaptation": 37,
    }
    assert (
        _canonical_optimization_sha256(cpu)
        == _canonical_optimization_sha256(cuda)
        == cpu_audit["canonical_optimization_sha256"]
        == cuda_audit["canonical_optimization_sha256"]
    )
    assert _run_configuration_sha256(
        run_profile="locked",
        fold=0,
        donors=["donor"],
        targets=["target"],
        optimization=cpu,
        seed=0,
    ) == _run_configuration_sha256(
        run_profile="locked",
        fold=0,
        donors=["donor"],
        targets=["target"],
        optimization=cuda,
        seed=0,
    )


def test_locked_api_rejects_overwrite_and_noncanonical_source_paths(
    tmp_path: Path,
) -> None:
    full = make_allen_config("full", seed=0)
    with pytest.raises(ProtocolViolation, match="overwrite is forbidden"):
        run_allen_experiment(
            processed_root=Path("data/processed/allen_vbo"),
            manifest_path=FROZEN_MANIFEST,
            output_directory=tmp_path / "output",
            run_profile="locked",
            optimization=full,
            fold=0,
            acknowledge_locked=True,
            overwrite=True,
            stage="prepare",
        )
    with pytest.raises(ProtocolViolation, match="canonical source manifest"):
        run_allen_experiment(
            processed_root=Path("data/processed/allen_vbo"),
            manifest_path=tmp_path / "manifest.json",
            output_directory=tmp_path / "output",
            run_profile="locked",
            optimization=full,
            fold=0,
            acknowledge_locked=True,
            stage="prepare",
        )
    with pytest.raises(ProtocolViolation, match="canonical processed root"):
        run_allen_experiment(
            processed_root=tmp_path / "processed",
            manifest_path=FROZEN_MANIFEST,
            output_directory=tmp_path / "output",
            run_profile="locked",
            optimization=full,
            fold=0,
            acknowledge_locked=True,
            stage="prepare",
        )


def test_locked_cli_rejects_overwrite_and_nonfull_scope(tmp_path: Path) -> None:
    script = Path("scripts/run_allen_experiment.py")
    common = [
        sys.executable,
        str(script),
        "--run-profile",
        "locked",
        "--stage",
        "prepare",
        "--fold",
        "0",
        "--acknowledge-locked",
        "--output",
        str(tmp_path / "output"),
    ]
    overwrite = subprocess.run(
        [*common, "--optimization", "full", "--overwrite"],
        capture_output=True,
        text=True,
        check=False,
    )
    nonfull = subprocess.run(
        [*common, "--optimization", "fast"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert overwrite.returncode == nonfull.returncode == 2
    assert "--overwrite is forbidden" in overwrite.stderr
    assert "--optimization full" in nonfull.stderr


def test_locked_append_only_stage_refusals(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "sentinel").write_text("occupied")
    with pytest.raises(FileExistsError, match="empty output"):
        _locked_append_only_gate(output, "prepare")

    (output / "sentinel").unlink()
    (output / "predictions.npz").write_bytes(b"already predicted")
    with pytest.raises(FileExistsError, match="predict is append-only"):
        _locked_append_only_gate(output, "predict")
    (output / "predictions.npz").unlink()

    (output / "score.complete.json").write_text("{}")
    with pytest.raises(FileExistsError, match="score is append-only"):
        _locked_append_only_gate(output, "score")


def test_locked_output_is_exact_canonical_path_and_rejects_copy_or_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(allen_module, "_repository_root", lambda: repository)
    canonical = repository / "results/allen-vbo/locked-fold-0"
    canonical.mkdir(parents=True)
    assert _require_canonical_locked_output(canonical, 0) == "results/allen-vbo/locked-fold-0"
    copied = tmp_path / "copy/results/allen-vbo/locked-fold-0"
    copied.mkdir(parents=True)
    with pytest.raises(ProtocolViolation, match="canonical one-shot"):
        _require_canonical_locked_output(copied, 0)

    canonical.rmdir()
    (repository / "results/allen-vbo").rmdir()
    (repository / "results").rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (repository / "results").symlink_to(outside, target_is_directory=True)
    symlinked = repository / "results/allen-vbo/locked-fold-0"
    symlinked.mkdir(parents=True)
    with pytest.raises(ProtocolViolation, match="symlink"):
        _require_canonical_locked_output(symlinked, 0)


def test_interrupted_allen_prepare_journal_restores_modes_and_quarantines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed = tmp_path / "processed"
    output = tmp_path / "locked-fold-0"
    animal = processed / "mouse_100"
    query_directory = output / "queries/mouse_100"
    animal.mkdir(parents=True)
    query_directory.mkdir(parents=True)
    monkeypatch.setattr(
        allen_module,
        "_locked_allen_prepare_scope",
        lambda _fold: (["100"], ["100"]),
    )
    legacy = animal / "windows.npz"
    role_sealed = animal / "sealed_omission_outcomes.npz"
    experiment_sealed = query_directory / "sealed_outcomes.npz"
    np.savez(legacy, response=np.asarray([1.0]))
    legacy.chmod(0o600)
    prepare_guard = allen_module._begin_allen_prepare_guard(
        processed_root=processed,
        output=output,
        fold=0,
        canonical_relative_output="results/allen-vbo/locked-fold-0",
        mice=["100"],
        targets=["100"],
    )
    for path, value in (
        (role_sealed, 2.0),
        (experiment_sealed, 3.0),
    ):
        np.savez(path, response=np.asarray([value]))
        path.chmod(0o600)
    experiment_sha256 = hashlib.sha256(experiment_sealed.read_bytes()).hexdigest()
    experiment_sealed.chmod(0)
    _seal_target_outcomes(
        processed,
        output,
        ["100"],
        fold=0,
        canonical_relative_output="results/allen-vbo/locked-fold-0",
        prepare_guard_sha256=prepare_guard["sha256"],
        experiment_sha256={"100": experiment_sha256},
    )
    assert (processed / allen_module.ALLEN_ACTIVE_SEAL_NAME).is_file()
    assert not os.access(animal / "windows.npz", os.R_OK)
    journal_path = processed / allen_module.ALLEN_ACTIVE_SEAL_NAME
    original_journal = journal_path.read_text(encoding="utf-8")
    rebound_journal = json.loads(original_journal)
    victim = tmp_path / "journal-must-not-touch"
    victim.write_text("preserve me", encoding="utf-8")
    rebound_journal["entries"][0]["path"] = str(victim)
    journal_path.write_text(
        json.dumps(rebound_journal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolViolation, match="entry binding"):
        _recover_allen_locked_stage(
            processed_root=processed,
            output=output,
            fold=0,
            stage="prepare",
            canonical_relative_output="results/allen-vbo/locked-fold-0",
        )
    assert victim.read_text(encoding="utf-8") == "preserve me"
    journal_path.write_text(original_journal, encoding="utf-8")

    assert (
        _recover_allen_locked_stage(
            processed_root=processed,
            output=output,
            fold=0,
            stage="prepare",
            canonical_relative_output="results/allen-vbo/locked-fold-0",
        )
        is None
    )
    assert not (processed / allen_module.ALLEN_ACTIVE_SEAL_NAME).exists()
    assert not (processed / allen_module.ALLEN_PREPARE_GUARD_NAME).exists()
    assert not output.exists()
    assert os.access(animal / "windows.npz", os.R_OK)
    assert not role_sealed.exists()
    quarantines = list(tmp_path.glob(".locked-fold-0.interrupted-*"))
    assert len(quarantines) == 1
    quarantined_outcome = quarantines[0] / "queries/mouse_100/sealed_outcomes.npz"
    assert not stat.S_IMODE(quarantined_outcome.stat().st_mode) & 0o444
    assert not os.access(quarantined_outcome, os.R_OK)


def test_allen_prepare_recovery_seals_outcome_copy_created_before_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "locked-fold-0"
    processed = tmp_path / "processed"
    animal = processed / "mouse_100"
    animal.mkdir(parents=True)
    monkeypatch.setattr(
        allen_module,
        "_locked_allen_prepare_scope",
        lambda _fold: (["100"], ["100"]),
    )
    outcome = output / "queries/mouse_100/sealed_outcomes.npz"
    outcome.parent.mkdir(parents=True)
    allen_module._begin_allen_prepare_guard(
        processed_root=processed,
        output=output,
        fold=0,
        canonical_relative_output="results/allen-vbo/locked-fold-0",
        mice=["100"],
        targets=["100"],
    )
    guard_path = processed / allen_module.ALLEN_PREPARE_GUARD_NAME
    original_guard = guard_path.read_text(encoding="utf-8")
    rebound_guard = json.loads(original_guard)
    victim = tmp_path / "must-not-be-unlinked"
    victim.write_text("preserve me", encoding="utf-8")
    rebound_guard["artifacts"][0]["path"] = str(victim)
    guard_path.write_text(
        json.dumps(rebound_guard, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolViolation, match="guard artifact binding"):
        _recover_allen_locked_stage(
            processed_root=processed,
            output=output,
            fold=0,
            stage="prepare",
            canonical_relative_output="results/allen-vbo/locked-fold-0",
        )
    assert victim.read_text(encoding="utf-8") == "preserve me"
    guard_path.write_text(original_guard, encoding="utf-8")
    role_sealed = animal / "sealed_omission_outcomes.npz"
    np.savez(role_sealed, response=np.asarray([2.0]))
    role_sealed.chmod(0o600)
    np.savez(outcome, response=np.asarray([1.0]))
    outcome.chmod(0o600)

    assert (
        _recover_allen_locked_stage(
            processed_root=processed,
            output=output,
            fold=0,
            stage="prepare",
            canonical_relative_output="results/allen-vbo/locked-fold-0",
        )
        is None
    )
    assert not role_sealed.exists()
    assert not (processed / allen_module.ALLEN_PREPARE_GUARD_NAME).exists()
    quarantines = list(tmp_path.glob(".locked-fold-0.interrupted-*"))
    assert len(quarantines) == 1
    quarantined_outcome = quarantines[0] / "queries/mouse_100/sealed_outcomes.npz"
    assert not stat.S_IMODE(quarantined_outcome.stat().st_mode) & 0o444
    assert not os.access(quarantined_outcome, os.R_OK)


def test_allen_restore_completion_recovers_json_only_and_rejects_rebinding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed = tmp_path / "processed"
    output = tmp_path / "locked-fold-0"
    animal = processed / "mouse_100"
    query_directory = output / "queries/mouse_100"
    animal.mkdir(parents=True)
    query_directory.mkdir(parents=True)
    monkeypatch.setattr(
        allen_module,
        "_locked_allen_prepare_scope",
        lambda _fold: (["100"], ["100"]),
    )
    target_paths = (
        animal / "windows.npz",
        animal / "sealed_omission_outcomes.npz",
        query_directory / "sealed_outcomes.npz",
    )
    for index, path in enumerate(target_paths):
        np.savez(path, response=np.asarray([float(index)]))
        path.chmod(0o600)
    experiment_sha256 = hashlib.sha256(target_paths[2].read_bytes()).hexdigest()
    target_paths[2].chmod(0)
    canonical = "results/allen-vbo/locked-fold-0"
    seals = _seal_target_outcomes(
        processed,
        output,
        ["100"],
        fold=0,
        canonical_relative_output=canonical,
        prepare_guard_sha256="a" * 64,
        experiment_sha256={"100": experiment_sha256},
    )
    preparation = {
        "targets": ["100"],
        "target_seals": seals,
        "canonical_relative_output": canonical,
        "fold": 0,
        "target_seal_transaction": allen_module._allen_seal_transaction_record(processed),
    }
    allen_module._atomic_write_json(
        output / "preparation.json",
        preparation,
        overwrite=False,
    )
    round_tripped_preparation = json.loads(
        (output / "preparation.json").read_text(encoding="utf-8")
    )
    assert (
        allen_module._validate_allen_seal_transaction_binding(
            round_tripped_preparation,
            processed_root=processed,
            output=output,
            canonical_relative_output=canonical,
            require_active_journal=True,
        )
        == preparation["target_seal_transaction"]["sha256"]
    )
    metrics = output / "metrics.json"
    metrics.write_text("{}")
    metrics_long = output / "metrics_long.csv"
    metrics_long.write_text("metric,value\nsentinel,1\n")
    _write_stage_completion(
        output,
        stage="score",
        artifacts=[metrics, metrics_long],
        metadata={
            "canonical_relative_output": canonical,
            "target_seal_transaction_sha256": preparation["target_seal_transaction"]["sha256"],
        },
        overwrite=False,
    )
    score_completion_path = output / "score.complete.json"
    score_completion_sidecar = output / "score.complete.json.sha256"
    score_completion_text = score_completion_path.read_text(encoding="utf-8")
    rebound_score_completion = json.loads(score_completion_text)
    rebound_score_completion["artifacts"].pop("metrics_long.csv")
    score_completion_path.write_text(
        json.dumps(rebound_score_completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rebound_score_sha = hashlib.sha256(score_completion_path.read_bytes()).hexdigest()
    score_completion_sidecar.write_text(
        f"{rebound_score_sha}  {score_completion_path.name}\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolViolation, match="score completion artifact set"):
        _restore_target_outcomes_after_score(
            seals,
            processed_root=processed,
            output=output,
            targets=["100"],
            preparation=preparation,
            canonical_relative_output=canonical,
        )
    score_completion_path.write_text(score_completion_text, encoding="utf-8")
    score_completion_sha = hashlib.sha256(score_completion_path.read_bytes()).hexdigest()
    score_completion_sidecar.write_text(
        f"{score_completion_sha}  {score_completion_path.name}\n",
        encoding="utf-8",
    )
    metrics_long_text = metrics_long.read_text(encoding="utf-8")
    metrics_long.write_text(f"{metrics_long_text}tampered,2\n", encoding="utf-8")
    with pytest.raises(LeakageError, match="completed artifact digest changed"):
        _restore_target_outcomes_after_score(
            seals,
            processed_root=processed,
            output=output,
            targets=["100"],
            preparation=preparation,
            canonical_relative_output=canonical,
        )
    metrics_long.write_text(metrics_long_text, encoding="utf-8")

    later_target = target_paths[1]
    later_target.chmod(0o600)
    later_bytes = later_target.read_bytes()
    later_target.write_bytes(later_bytes + b"digest drift")
    later_target.chmod(int(seals["100"]["role_sealed"]["sealed_mode"], 8))
    with pytest.raises(LeakageError, match="restored outcome digest changed"):
        _restore_target_outcomes_after_score(
            seals,
            processed_root=processed,
            output=output,
            targets=["100"],
            preparation=preparation,
            canonical_relative_output=canonical,
        )
    assert all(
        stat.S_IMODE(path.stat().st_mode) == int(seals["100"][name]["sealed_mode"], 8)
        for path, name in zip(
            target_paths,
            ("legacy_combined", "role_sealed", "experiment_sealed"),
            strict=True,
        )
    )
    later_target.chmod(0o600)
    later_target.write_bytes(later_bytes)
    later_target.chmod(int(seals["100"]["role_sealed"]["sealed_mode"], 8))

    atomic_write_bytes = allen_module._atomic_write_bytes

    def interrupt_sidecar(
        path: Path,
        payload: bytes,
        *,
        overwrite: bool,
    ) -> None:
        if path.name == "restore.complete.json.sha256":
            raise RuntimeError("injected crash between restore JSON and sidecar")
        atomic_write_bytes(path, payload, overwrite=overwrite)

    monkeypatch.setattr(allen_module, "_atomic_write_bytes", interrupt_sidecar)
    with pytest.raises(RuntimeError, match="injected crash"):
        _restore_target_outcomes_after_score(
            seals,
            processed_root=processed,
            output=output,
            targets=["100"],
            preparation=preparation,
            canonical_relative_output=canonical,
        )
    assert (output / "restore.complete.json").is_file()
    assert not (output / "restore.complete.json.sha256").exists()
    assert (processed / allen_module.ALLEN_ACTIVE_SEAL_NAME).is_file()
    assert all(
        stat.S_IMODE(path.stat().st_mode) == int(seals["100"][name]["sealed_mode"], 8)
        for path, name in zip(
            target_paths,
            ("legacy_combined", "role_sealed", "experiment_sealed"),
            strict=True,
        )
    )

    monkeypatch.setattr(allen_module, "_atomic_write_bytes", atomic_write_bytes)
    assert (
        _recover_allen_locked_stage(
            processed_root=processed,
            output=output,
            fold=0,
            stage="score",
            canonical_relative_output=canonical,
        )
        == "score_complete"
    )
    assert (output / "restore.complete.json.sha256").is_file()
    assert not (processed / allen_module.ALLEN_ACTIVE_SEAL_NAME).exists()
    valid_payload = json.loads((output / "restore.complete.json").read_text(encoding="utf-8"))
    completion_path = output / "restore.complete.json"
    sidecar = output / "restore.complete.json.sha256"

    restored_target = target_paths[0]
    original_bytes = restored_target.read_bytes()
    restored_target.chmod(0o400)
    with pytest.raises(ProtocolViolation, match="restored target mode changed"):
        _recover_allen_locked_stage(
            processed_root=processed,
            output=output,
            fold=0,
            stage="score",
            canonical_relative_output=canonical,
        )
    restored_target.chmod(0o600)

    restored_target.write_bytes(original_bytes + b"digest drift")
    with pytest.raises(ProtocolViolation, match="restored target digest changed"):
        _recover_allen_locked_stage(
            processed_root=processed,
            output=output,
            fold=0,
            stage="score",
            canonical_relative_output=canonical,
        )
    restored_target.write_bytes(original_bytes)

    original_target = restored_target.with_suffix(".original")
    restored_target.rename(original_target)
    try:
        restored_target.write_bytes(original_bytes)
        restored_target.chmod(0o600)
        with pytest.raises(ProtocolViolation, match="restored target identity changed"):
            _recover_allen_locked_stage(
                processed_root=processed,
                output=output,
                fold=0,
                stage="score",
                canonical_relative_output=canonical,
            )
    finally:
        restored_target.unlink(missing_ok=True)
        original_target.rename(restored_target)

    sidecar.write_text(f"{'0' * 64}  {completion_path.name}\n")
    with pytest.raises(ProtocolViolation, match="malformed"):
        _recover_allen_locked_stage(
            processed_root=processed,
            output=output,
            fold=0,
            stage="score",
            canonical_relative_output=canonical,
        )

    for key, changed in (
        ("schema", "cadence-allen-target-restore-completion-rebound"),
        ("canonical_relative_output", "results/allen-vbo/locked-fold-4"),
        ("score_completion_sha256", "f" * 64),
    ):
        rebound = {**valid_payload, key: changed}
        completion_path.write_text(
            json.dumps(rebound, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        digest = hashlib.sha256(completion_path.read_bytes()).hexdigest()
        sidecar.write_text(f"{digest}  {completion_path.name}\n", encoding="utf-8")
        with pytest.raises(ProtocolViolation, match="binding"):
            _recover_allen_locked_stage(
                processed_root=processed,
                output=output,
                fold=0,
                stage="score",
                canonical_relative_output=canonical,
            )


def test_locked_run_attests_tagged_freeze_before_loading_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[Path] = []

    def reject_before_data(*, repository: str | Path = ".") -> None:
        calls.append(Path(repository))
        raise ProtocolViolation("sentinel freeze rejection")

    monkeypatch.setattr(
        "cadence.experiments.allen.attest_preoutcome_freeze",
        reject_before_data,
    )
    monkeypatch.setattr(
        "cadence.experiments.allen.load_animal_support",
        lambda *_args, **_kwargs: pytest.fail("data opened before freeze attestation"),
    )
    with pytest.raises(ProtocolViolation, match="sentinel freeze rejection"):
        run_allen_experiment(
            processed_root=Path("data/processed/allen_vbo"),
            manifest_path=FROZEN_MANIFEST,
            output_directory=tmp_path / "output",
            run_profile="locked",
            optimization=make_allen_config("full", seed=0),
            fold=0,
            acknowledge_locked=True,
            stage="prepare",
        )
    assert len(calls) == 1
    assert calls[0].resolve() == Path(__file__).resolve().parents[1]


def test_development_profile_rejects_a_locked_mouse() -> None:
    locked_mouse = locked_fold_table(FROZEN_MANIFEST).iloc[0]["mouse_id"]
    with pytest.raises(LeakageError, match="four development mice"):
        resolve_run_mice(
            FROZEN_MANIFEST,
            profile="development",
            development_target=str(locked_mouse),
        )


def test_signal_blind_normal_selection_is_exact_and_order_invariant() -> None:
    ids = np.arange(10_000, 10_240, dtype=np.int64)
    first = split_normal_presentations("mouse-x", ids)
    permutation = np.random.default_rng(4).permutation(len(ids))
    shuffled = ids[permutation]
    second = split_normal_presentations("mouse-x", shuffled)

    assert {name: len(values) for name, values in first.items()} == {
        "fit": 112,
        "val": 24,
        "audit": 24,
        "match": 80,
    }
    selected_first = {name: set(ids[indices].tolist()) for name, indices in first.items()}
    selected_second = {name: set(shuffled[indices].tolist()) for name, indices in second.items()}
    assert selected_first == selected_second
    all_indices = np.concatenate(list(first.values()))
    assert len(np.unique(all_indices)) == len(ids)


def test_onset_pulse_indexes_transition_into_zero_sample() -> None:
    relative = np.linspace(-1.0, 2.0, 31)
    assert transition_index(relative, 0.0) == 9
    assert transition_index(relative, 0.75) == 17

    presentations = pd.DataFrame(
        {
            "start_time": [9.25, 10.0, 10.75],
            "image_index": [2, 2, 2],
            "omitted": [False, True, False],
            "active": [True, True, True],
        }
    )
    inputs = build_flash_inputs(presentations, [10.0], relative)
    assert inputs[0, 2, 2] == 1
    assert inputs[0, 9].sum() == 0
    assert inputs[0, 17, 2] == 1


def test_control_matching_prioritizes_image_then_risk_set() -> None:
    query = np.asarray([[2, 7], [4, 20]])
    controls = np.asarray([[1, 7], [2, 3], [2, 7], [4, 18], [4, 30]])
    matches, audit = match_control_indices(query, controls, controls_per_query=2)

    assert matches[0].tolist() == [2, -1]
    assert set(matches[1]) == {3, 4}
    assert audit["exact_risk_set_fraction"] == 0.5
    assert audit["same_preceding_image_fraction"] == 1.0
    assert audit["fallback_levels"] == [
        "exact_image_and_flashes",
        "image_and_risk_bin",
    ]
    assert audit["effective_controls"] == [1, 2]


def test_control_matching_uses_declared_fallback_levels_without_mixing() -> None:
    controls = np.asarray(
        [
            [2, 7],
            [3, 5],
            [4, 20],
            [8, 7],
        ]
    )
    queries = np.asarray(
        [
            [2, 7],  # exact
            [3, 6],  # same image and [5, 8] bin
            [4, 2],  # same image only
            [9, 7],  # same risk bin only
            [9, -1],  # complete pool
        ]
    )
    matches, audit = match_control_indices(queries, controls, controls_per_query=3)

    assert audit["fallback_levels"] == [
        "exact_image_and_flashes",
        "image_and_risk_bin",
        "image_only",
        "risk_bin_only",
        "complete_pool",
    ]
    assert audit["effective_controls"] == [1, 1, 1, 3, 3]
    assert matches[0].tolist() == [0, -1, -1]
    assert matches[2].tolist() == [2, -1, -1]


def test_scaler_ignores_every_non_fit_value() -> None:
    generator = np.random.default_rng(3)
    neural = generator.uniform(0, 3, size=(12, 5, 4))
    behavior = generator.normal(size=(12, 5, 3))
    valid_neural = np.ones_like(neural, dtype=bool)
    valid_behavior = np.ones_like(behavior, dtype=bool)
    fit = np.arange(7)
    ids = np.arange(100, 112)
    first = MouseScaler.fit(neural, valid_neural, behavior, valid_behavior, fit, ids[fit])
    neural[7:] = 1e9
    behavior[7:] = -1e12
    second = MouseScaler.fit(neural, valid_neural, behavior, valid_behavior, fit, ids[fit])

    np.testing.assert_allclose(first.neural_center, second.neural_center)
    np.testing.assert_allclose(first.neural_scale, second.neural_scale)
    np.testing.assert_allclose(first.behavior_center, second.behavior_center)
    np.testing.assert_allclose(first.behavior_scale, second.behavior_scale)
    assert first.fit_presentation_sha256 == second.fit_presentation_sha256


def test_explicit_missing_mask_channels_are_value_and_sentinel_invariant() -> None:
    values = np.asarray([[[1.0, 999.0], [2.0, -999.0]]], dtype=np.float32)
    neural_valid = np.asarray([[[True, False], [True, False]]])
    behavior_valid = np.asarray([[[True, False, True], [False, True, True]]])
    first, first_loss_mask = _augment_neural_with_masks(
        np.where(neural_valid, values, 0.0),
        neural_valid,
        behavior_valid,
    )
    changed = values.copy()
    changed[..., 1] = np.asarray([[1e20, -1e20]])
    second, second_loss_mask = _augment_neural_with_masks(
        np.where(neural_valid, changed, 0.0),
        neural_valid,
        behavior_valid,
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_loss_mask, second_loss_mask)
    # The encoder sees explicit masks, while auxiliary channels have no loss.
    assert first.shape[-1] == 2 * values.shape[-1] + 3
    assert not first_loss_mask[..., values.shape[-1] :].any()


def test_expected_effect_primary_score_weights_strata_equally() -> None:
    # Three trials in stratum A are perfect; one trial in stratum B is wrong.
    # Equal-stratum skill is therefore the mean of 1 and a negative score,
    # rather than the trial-count-weighted pooled result.
    observed = np.asarray([[[1.0]], [[1.0]], [[1.0]], [[1.0]]])
    predicted = np.asarray([[[1.0]], [[1.0]], [[1.0]], [[-1.0]]])
    descriptors = np.asarray([[0, 1], [0, 1], [0, 1], [1, 2]])
    fallback = np.asarray(
        [
            "exact_image_and_flashes",
            "exact_image_and_flashes",
            "exact_image_and_flashes",
            "image_only",
        ]
    )
    score, strata = _expected_stratum_skill(
        predicted,
        observed,
        np.ones_like(observed, dtype=bool),
        np.ones(1),
        descriptors,
        fallback,
    )

    assert len(strata) == 2
    assert score == pytest.approx(-1.0)


def test_query_loader_rejects_post_onset_outcomes(tmp_path: Path) -> None:
    legal = tmp_path / "legal.npz"
    np.savez(legal, pre_neural=np.zeros((2, 3)), treated_inputs=np.zeros((2, 4, 8)))
    query = _load_query(legal)
    assert set(query) == {"pre_neural", "treated_inputs"}

    leaked = tmp_path / "leaked.npz"
    np.savez(
        leaked,
        pre_neural=np.zeros((2, 3)),
        omission_neural=np.zeros((2, 10, 3)),
    )
    with pytest.raises(LeakageError, match="post-onset outcomes leaked"):
        _load_query(leaked)


def test_query_loading_is_invariant_when_sealed_file_has_no_permissions(
    tmp_path: Path,
) -> None:
    query_path = tmp_path / "query_inputs.npz"
    sealed_path = tmp_path / "sealed_outcomes.npz"
    np.savez(
        query_path,
        pre_neural=np.ones((2, 3)),
        pre_neural_mask=np.ones((2, 3), dtype=bool),
    )
    np.savez(sealed_path, omission_neural=np.full((2, 4, 3), 999.0))
    before = _load_query(query_path)
    sealed_path.chmod(0)
    try:
        after = _load_query(query_path)
    finally:
        sealed_path.chmod(0o600)
    np.testing.assert_array_equal(before["pre_neural"], after["pre_neural"])


def test_stage_completion_manifest_is_atomic_authenticated_and_append_only(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"immutable stage output")
    written = _write_stage_completion(
        tmp_path,
        stage="prepare",
        artifacts=[artifact],
        metadata={"configuration_sha256": "abc"},
        overwrite=False,
    )
    loaded = _read_stage_completion(tmp_path, "prepare")
    assert loaded["completion_sha256"] == written["completion_sha256"]
    assert loaded["artifacts"] == {
        "artifact.bin": hashlib.sha256(artifact.read_bytes()).hexdigest()
    }
    with pytest.raises(FileExistsError):
        _write_stage_completion(
            tmp_path,
            stage="prepare",
            artifacts=[artifact],
            metadata={},
            overwrite=False,
        )
    artifact.write_bytes(b"tampered after completion")
    with pytest.raises(LeakageError, match="artifact digest changed"):
        _verify_completed_artifact(tmp_path, loaded, "artifact.bin")
    completion, _ = _stage_completion_paths(tmp_path, "prepare")
    completion.write_text("{}")
    with pytest.raises(LeakageError, match="digest mismatch"):
        _read_stage_completion(tmp_path, "prepare")


def test_tracked_protocol_file_identity_is_bound_to_attested_commit(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    tracked = repository / "data/processed/allen_vbo/index.json"
    tracked.parent.mkdir(parents=True)
    tracked.write_text('{"release":"1.1.0"}\n')
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "frozen",
        ],
        cwd=repository,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    attestation = FreezeAttestation(commit=commit, tag="test", tag_object="test")
    audit = _tracked_file_audit(
        repository,
        Path("data/processed/allen_vbo/index.json"),
        attestation,
    )
    assert audit["commit"] == commit
    tracked.write_text('{"release":"tampered"}\n')
    with pytest.raises(ProtocolViolation, match="differs from attested commit"):
        _tracked_file_audit(
            repository,
            Path("data/processed/allen_vbo/index.json"),
            attestation,
        )


def test_locked_protocol_scope_verifies_release_config_and_cohort_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    sources = {
        Path(
            "data/manifests/allen_vbo_slc17a7_visp175_familiar_active_v1.1.0.json"
        ): FROZEN_MANIFEST,
        Path("data/processed/allen_vbo/index.json"): Path("data/processed/allen_vbo/index.json"),
        Path("configs/allen_experiment.yaml"): Path("configs/allen_experiment.yaml"),
    }
    for relative, source in sources.items():
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    monkeypatch.setattr(
        "cadence.experiments.allen._tracked_file_audit",
        lambda _repository, relative, _attestation: {
            "path": relative.as_posix(),
            "sha256": hashlib.sha256((_repository / relative).read_bytes()).hexdigest(),
            "git_blob": "synthetic",
            "commit": "frozen",
        },
    )
    attestation = FreezeAttestation(commit="frozen", tag="pre-outcome-v1.0.0", tag_object="tag")
    manifest = repository / next(iter(sources))
    processed = repository / "data/processed/allen_vbo"
    config = make_allen_config("full", seed=0, device="cpu")
    audit = _verify_locked_protocol_scope(
        processed_root=processed,
        manifest_path=manifest,
        optimization=config,
        seed=0,
        attestation=attestation,
        repository=repository,
    )
    assert audit["release"] == "1.1.0"
    assert audit["cohort_mouse_count"] == 32

    index_path = processed / "index.json"
    original_index = index_path.read_text()
    changed_index = json.loads(original_index)
    changed_index["release"] = "tampered"
    index_path.write_text(json.dumps(changed_index))
    with pytest.raises(ProtocolViolation, match="index release"):
        _verify_locked_protocol_scope(
            processed_root=processed,
            manifest_path=manifest,
            optimization=config,
            seed=0,
            attestation=attestation,
            repository=repository,
        )
    index_path.write_text(original_index)

    config_path = repository / "configs/allen_experiment.yaml"
    original_config = config_path.read_text()
    config_path.write_text(original_config.replace("intervention_rank: 2", "intervention_rank: 1"))
    with pytest.raises(ProtocolViolation, match="method/ablation"):
        _verify_locked_protocol_scope(
            processed_root=processed,
            manifest_path=manifest,
            optimization=config,
            seed=0,
            attestation=attestation,
            repository=repository,
        )
    config_path.write_text(original_config)

    changed_index = json.loads(original_index)
    changed_index["animals"][0]["ophys_experiment_id"] = -1
    index_path.write_text(json.dumps(changed_index))
    with pytest.raises(ProtocolViolation, match="identities differ"):
        _verify_locked_protocol_scope(
            processed_root=processed,
            manifest_path=manifest,
            optimization=config,
            seed=0,
            attestation=attestation,
            repository=repository,
        )


def test_processed_index_hash_identity_and_preprocessing_config_precede_split(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    processed = repository / "data/processed/allen_vbo"
    animal = processed / "mouse_100"
    animal.mkdir(parents=True)
    legacy = animal / "windows.npz"
    np.savez(legacy, sentinel=np.asarray([1, 2, 3]))
    stimulus_presentations = animal / "stimulus_presentations.parquet"
    window_index = animal / "window_index.parquet"
    stimulus_presentations.write_bytes(b"authenticated stimulus table fixture")
    window_index.write_bytes(b"authenticated window-index fixture")
    output_sha256 = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (stimulus_presentations, window_index, legacy)
    }
    legacy_sha256 = output_sha256["windows.npz"]
    provenance = {
        "mouse_id": "100",
        "ophys_experiment_id": 9001,
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
        "outputs": {name: {"sha256": digest} for name, digest in output_sha256.items()},
    }
    (animal / "provenance.json").write_text(json.dumps(provenance))
    commitment_rows = [
        {
            "mouse_id": "100",
            "ophys_experiment_id": 9001,
            "outputs": output_sha256,
        }
    ]
    commitment_sha256 = hashlib.sha256(
        json.dumps(
            commitment_rows,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    index = {
        "source_content_commitment": {
            "algorithm": "sha256-canonical-json-v1",
            "files_per_mouse": [
                "stimulus_presentations.parquet",
                "window_index.parquet",
                "windows.npz",
            ],
            "sha256": commitment_sha256,
        },
        "animals": [
            {
                "mouse_id": "100",
                "ophys_experiment_id": 9001,
                "arrays": "data/processed/allen_vbo/mouse_100/windows.npz",
                "arrays_sha256": legacy_sha256,
                "provenance": "data/processed/allen_vbo/mouse_100/provenance.json",
            }
        ],
    }
    (processed / "index.json").write_text(json.dumps(index))
    manifest = repository / "manifest.json"
    manifest.write_text(
        json.dumps({"nwb_files": [{"mouse_id": "100", "ophys_experiment_id": 9001}]})
    )
    audit = _verify_locked_processed_inputs(
        repository=repository,
        processed_root=processed,
        manifest_path=manifest,
        mouse_ids=["100"],
    )
    assert audit["verified_before_split"] is True
    assert audit["mice"]["100"]["legacy_sha256"] == legacy_sha256

    legacy.write_bytes(b"tampered before split")
    with pytest.raises(ProtocolViolation, match="digest mismatch"):
        _verify_locked_processed_inputs(
            repository=repository,
            processed_root=processed,
            manifest_path=manifest,
            mouse_ids=["100"],
        )


def test_physical_target_seal_score_unseal_and_later_donor_reuse(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    output = tmp_path / "run"
    animal = processed / "mouse_100"
    query_directory = output / "queries/mouse_100"
    animal.mkdir(parents=True)
    query_directory.mkdir(parents=True)
    legacy = animal / "windows.npz"
    role_sealed = animal / "sealed_omission_outcomes.npz"
    experiment_sealed = query_directory / "sealed_outcomes.npz"
    query = query_directory / "query_inputs.npz"
    np.savez(legacy, response=np.asarray([1.0]))
    np.savez(role_sealed, response=np.asarray([2.0]))
    np.savez(experiment_sealed, response=np.asarray([3.0]))
    np.savez(query, safe=np.asarray([4.0]))
    for path in (legacy, role_sealed, experiment_sealed, query):
        path.chmod(0o600)
    expected_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (legacy, role_sealed, experiment_sealed)
    }
    experiment_sealed.chmod(0)
    canonical_relative_output = "results/allen-vbo/locked-fold-0"
    seals = _seal_target_outcomes(
        processed,
        output,
        ["100"],
        fold=0,
        canonical_relative_output=canonical_relative_output,
        prepare_guard_sha256="a" * 64,
        experiment_sha256={"100": expected_hashes["sealed_outcomes.npz"]},
    )
    _assert_target_outcomes_sealed(
        seals,
        processed_root=processed,
        output=output,
        targets=["100"],
    )
    original_experiment = experiment_sealed.with_suffix(".original")
    experiment_sealed.rename(original_experiment)
    try:
        np.savez(experiment_sealed, response=np.asarray([3.0]))
        experiment_sealed.chmod(int(seals["100"]["experiment_sealed"]["sealed_mode"], 8))
        with pytest.raises(LeakageError, match="identity changed during prediction"):
            _assert_target_outcomes_sealed(
                seals,
                processed_root=processed,
                output=output,
                targets=["100"],
            )
        with pytest.raises(LeakageError, match="identity changed before scoring"):
            _open_experiment_sealed_for_score(
                experiment_sealed,
                seals["100"]["experiment_sealed"],
            )
    finally:
        experiment_sealed.unlink(missing_ok=True)
        original_experiment.rename(experiment_sealed)
    for path in (legacy, role_sealed, experiment_sealed):
        assert not stat.S_IMODE(path.stat().st_mode) & 0o444
        assert not os.access(path, os.R_OK)
        with pytest.raises(PermissionError):
            path.open("rb")
    with np.load(query, allow_pickle=False) as safe:
        np.testing.assert_array_equal(safe["safe"], [4.0])

    sealed, observed_hash = _open_experiment_sealed_for_score(
        experiment_sealed,
        seals["100"]["experiment_sealed"],
    )
    np.testing.assert_array_equal(sealed["response"], [3.0])
    assert observed_hash == expected_hashes["sealed_outcomes.npz"]
    for path in (legacy, role_sealed, experiment_sealed):
        assert not os.access(path, os.R_OK)

    preparation = {
        "targets": ["100"],
        "target_seals": seals,
        "canonical_relative_output": canonical_relative_output,
        "fold": 0,
        "target_seal_transaction": allen_module._allen_seal_transaction_record(processed),
        "processed_input_audit": {
            "mice": {"100": {"legacy_sha256": expected_hashes["windows.npz"]}}
        },
        "role_artifacts": {
            "100": {"sealed_omission_outcomes.npz": expected_hashes["sealed_omission_outcomes.npz"]}
        },
        "experiment_artifacts": {
            "100": {"sealed_outcomes.npz": expected_hashes["sealed_outcomes.npz"]}
        },
    }
    metrics = output / "metrics.json"
    metrics.write_text("{}")
    metrics_long = output / "metrics_long.csv"
    metrics_long.write_text("metric,value\nsentinel,1\n")
    _write_stage_completion(
        output,
        stage="score",
        artifacts=[metrics, metrics_long],
        metadata={
            "canonical_relative_output": canonical_relative_output,
            "target_seal_transaction_sha256": preparation["target_seal_transaction"]["sha256"],
        },
        overwrite=False,
    )
    restoration = _restore_target_outcomes_after_score(
        seals,
        processed_root=processed,
        output=output,
        targets=["100"],
        preparation=preparation,
        canonical_relative_output=canonical_relative_output,
    )
    assert restoration["eligible_for_later_donor_reuse"] is True
    assert restoration["restored_after_score_commit"] is True
    assert not (processed / allen_module.ALLEN_ACTIVE_SEAL_NAME).exists()
    assert (output / "restore.complete.json").is_file()
    for path in (legacy, role_sealed, experiment_sealed):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert os.access(path, os.R_OK)
    # The same mouse can now be consumed as a donor in a later outer fold.
    with np.load(role_sealed, allow_pickle=False) as donor_outcomes:
        np.testing.assert_array_equal(donor_outcomes["response"], [2.0])


def test_black_box_normal_objective_flattens_transition_time_axis() -> None:
    model = BlackBoxMetaGRU(
        latent_dim=3,
        input_dim=8,
        behavior_dim=3,
        num_interventions=1,
        hidden_dim=8,
        residual_rank=1,
        intervention_rank=1,
        dt=0.1,
    )
    model.register_animal("mouse-x", neural_dim=5, donor=False)
    batch = SequenceBatch(
        animal_id="mouse-x",
        neural=torch.randn(4, 7, 5),
        behavior=torch.randn(4, 7, 3),
        inputs=torch.randn(4, 7, 8),
        intervention=torch.zeros(4, 7, 1),
        onset=3,
        neural_mask=torch.ones(4, 7, 5, dtype=torch.bool),
        behavior_mask=torch.ones(4, 7, 3, dtype=torch.bool),
    )
    total, neural, behavior = _normal_loss(model, batch)
    assert torch.isfinite(total)
    assert torch.isfinite(neural)
    assert torch.isfinite(behavior)


def test_donor_intervention_delta_is_registered_shrunk_and_target_omits_it() -> None:
    model = HierarchicalControlledSSM(
        latent_dim=3,
        input_dim=8,
        behavior_dim=3,
        num_interventions=1,
        hidden_dim=8,
        residual_rank=1,
        intervention_rank=1,
        dt=0.1,
    )
    model.register_animal("donor-a", neural_dim=5, donor=True)
    model.register_animal("donor-b", neural_dim=5, donor=True)
    model.register_animal("target", neural_dim=5, donor=False)
    assert len(model.donor_intervention_delta) == 2
    with torch.no_grad():
        for delta in model.donor_intervention_delta.values():
            delta.zero_()
    treated = SequenceBatch(
        animal_id="donor-a",
        neural=torch.randn(4, 7, 5),
        behavior=torch.randn(4, 7, 3),
        inputs=torch.randn(4, 7, 8),
        intervention=torch.zeros(4, 7, 1),
        onset=3,
        neural_mask=torch.ones(4, 7, 5, dtype=torch.bool),
        behavior_mask=torch.ones(4, 7, 3, dtype=torch.bool),
    )
    treated.intervention[:, 2, 0] = 1.0
    batch = MatchedInterventionBatch(
        treated=treated,
        control_neural=torch.randn(4, 7, 5),
        control_behavior=torch.randn(4, 7, 3),
        control_neural_mask=torch.ones(4, 7, 5, dtype=torch.bool),
        control_behavior_mask=torch.ones(4, 7, 3, dtype=torch.bool),
        control_inputs=torch.randn(4, 7, 8),
    )
    model.configure_stage("intervention")
    total, _, _ = _intervention_loss(model, batch)
    total.backward()
    donor_gradients = [parameter.grad for parameter in model.donor_intervention_delta.values()]
    assert donor_gradients[0] is not None
    assert torch.count_nonzero(donor_gradients[0]) > 0


def test_selection_delta_topology_excludes_validation_then_refit_adds_all() -> None:
    base = HierarchicalControlledSSM(
        latent_dim=3,
        input_dim=8,
        behavior_dim=3,
        num_interventions=1,
        hidden_dim=8,
        residual_rank=1,
        intervention_rank=1,
        dt=0.1,
    )
    for mouse in ("train-a", "train-b", "validation"):
        base.register_animal(mouse, neural_dim=5, donor=False)
    selection = copy.deepcopy(base)
    _add_zero_donor_delta(selection, "train-a")
    _add_zero_donor_delta(selection, "train-b")
    validation_group = selection._intervention_groups["animal_validation"]
    assert validation_group not in selection.donor_intervention_delta
    assert all(
        torch.count_nonzero(delta).item() == 0
        for delta in selection.donor_intervention_delta.values()
    )

    refit = copy.deepcopy(base)
    for mouse in ("train-a", "train-b", "validation"):
        _add_zero_donor_delta(refit, mouse)
    assert len(refit.donor_intervention_delta) == 3
    assert all(
        torch.count_nonzero(delta).item() == 0 for delta in refit.donor_intervention_delta.values()
    )


def test_intervention_fit_projects_declared_donors_after_every_optimizer_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = HierarchicalControlledSSM(
        latent_dim=3,
        input_dim=8,
        behavior_dim=3,
        num_interventions=1,
        hidden_dim=8,
        residual_rank=1,
        intervention_rank=2,
        dt=0.1,
    )
    groups = []
    batches = []
    for index, mouse in enumerate(("donor-a", "donor-b")):
        model.register_animal(mouse, neural_dim=5, donor=False)
        groups.append(_add_zero_donor_delta(model, mouse))
        treated = SequenceBatch(
            animal_id=mouse,
            neural=torch.randn(2, 7, 5),
            behavior=torch.randn(2, 7, 3),
            inputs=torch.randn(2, 7, 8),
            intervention=torch.zeros(2, 7, 1),
            onset=3,
            neural_mask=torch.ones(2, 7, 5, dtype=torch.bool),
            behavior_mask=torch.ones(2, 7, 3, dtype=torch.bool),
        )
        treated.intervention[:, 2, 0] = 1.0
        batches.append(
            MatchedInterventionBatch(
                treated=treated,
                control_neural=torch.randn(2, 7, 5),
                control_behavior=torch.randn(2, 7, 3),
                control_neural_mask=torch.ones(2, 7, 5, dtype=torch.bool),
                control_behavior_mask=torch.ones(2, 7, 3, dtype=torch.bool),
                control_inputs=torch.randn(2, 7, 8),
            )
        )
        with torch.no_grad():
            model.donor_intervention_delta[groups[-1]].fill_(float(index + 1))

    calls = 0
    original = HierarchicalControlledSSM.project_donor_deltas_zero_mean

    def counted_projection(
        self: HierarchicalControlledSSM,
        group_keys: object = None,
    ) -> float:
        nonlocal calls
        calls += 1
        return original(self, group_keys)  # type: ignore[arg-type]

    monkeypatch.setattr(
        HierarchicalControlledSSM,
        "project_donor_deltas_zero_mean",
        counted_projection,
    )
    _fit_stage(
        model,
        batches,
        [batches[0]],
        stage="intervention",
        config=FitConfig(
            learning_rate=1e-3,
            max_epochs=1,
            patience=1,
            seed=23,
            device="cpu",
            mixed_precision=False,
        ),
        donor_projection_groups=groups,
    )
    # Initial projection + one projection per two batches + final projection.
    assert calls == 4
    stacked = torch.stack([model.donor_intervention_delta[group] for group in groups])
    assert torch.linalg.vector_norm(stacked.mean(dim=0)).item() <= 1e-7


def test_inner_validation_normal_gradients_cannot_reach_shared_f() -> None:
    model = HierarchicalControlledSSM(
        latent_dim=3,
        input_dim=8,
        behavior_dim=3,
        num_interventions=1,
        hidden_dim=8,
        residual_rank=1,
        intervention_rank=2,
        dt=0.1,
    )
    model.register_animal("intervention-train", neural_dim=5, donor=False)
    model.register_animal("intervention-validation", neural_dim=5, donor=False)
    model.configure_stage(
        "target_adaptation",
        target_animal="intervention-validation",
    )
    batch = SequenceBatch(
        animal_id="intervention-validation",
        neural=torch.randn(2, 7, 5),
        behavior=torch.randn(2, 7, 3),
        inputs=torch.randn(2, 7, 8),
        intervention=torch.zeros(2, 7, 1),
        onset=3,
        neural_mask=torch.ones(2, 7, 5, dtype=torch.bool),
        behavior_mask=torch.ones(2, 7, 3, dtype=torch.bool),
    )
    total, _, _ = _normal_loss(model, batch)
    total.backward()

    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in model.shared.parameters()
    )
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in model.behavior_decoder.parameters()
    )
    target_gradients = [
        parameter.grad
        for parameter in model.adapter("intervention-validation").parameters()
        if parameter.requires_grad
    ]
    assert target_gradients
    assert any(gradient is not None for gradient in target_gradients)


def test_report_methods_include_cheap_ablations_and_mark_pooled_separately() -> None:
    methods = _selected_report_methods(make_allen_config("smoke", methods=("proposed",)))
    assert "proposed_no_residual" in methods
    assert "proposed_no_target_adaptation" in methods
    assert "pooled" not in methods


def test_score_rejects_bad_prediction_hash_before_loading_support_or_sealed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    optimization = make_allen_config("smoke", methods=("proposed",))
    donors = ["539517", "448900"]
    targets = ["423606"]
    configuration = _run_configuration_sha256(
        run_profile="development",
        fold=None,
        donors=donors,
        targets=targets,
        optimization=optimization,
        seed=0,
    )
    preparation = {
        "configuration_sha256": configuration,
        "experiment_artifacts": {},
        "query_audits": {},
    }
    (tmp_path / "preparation.json").write_text(json.dumps(preparation))
    (tmp_path / "predictions.npz").write_bytes(b"corrupted")
    (tmp_path / "predictions.npz.sha256").write_text(f"{'0' * 64}  predictions.npz\n")
    monkeypatch.setattr(
        "cadence.experiments.allen.load_animal_support",
        lambda *_args, **_kwargs: pytest.fail(
            "normal support opened before prediction digest verification"
        ),
    )
    with pytest.raises(LeakageError, match="sealed outcomes remain unopened"):
        _score_allen_stage(
            processed_root=tmp_path,
            output=tmp_path,
            run_profile="development",
            fold=None,
            donors=donors,
            targets=targets,
            attestation=None,
            optimization=optimization,
            seed=0,
            overwrite=True,
        )
