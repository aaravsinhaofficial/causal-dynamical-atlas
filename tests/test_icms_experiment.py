from __future__ import annotations

import hashlib
import json
import stat
import subprocess
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

import cadence.experiments.icms as icms
from cadence.data.dandi_icms import (
    DANDISET_ID,
    DANDISET_VERSION,
    INTERVENTION_DESCRIPTOR_COLUMNS,
    TASK_MICE,
)
from cadence.protocol import ProtocolViolation
from cadence.training import FitConfig


def _strings(group: h5py.Group, name: str, values: list[str]) -> None:
    group.create_dataset(
        name,
        data=np.asarray(values, dtype=object),
        dtype=h5py.string_dtype("utf-8"),
    )


def _write_processed_animal(
    path: Path,
    animal: str,
    *,
    sessions: int = 1,
    no_catches: bool = False,
) -> None:
    time = np.arange(-0.5, 1.0, 0.1)
    normal_iti = 8
    task_trials = 10
    count = task_trials + normal_iti
    with h5py.File(path, "w") as file:
        file.attrs["schema"] = "cadence-dandi-001868-v1"
        file.attrs["dandiset_id"] = DANDISET_ID
        file.attrs["dandiset_version"] = DANDISET_VERSION
        file.attrs["animal_id"] = animal
        file.attrs["raw_stim_channel_in_descriptor"] = False
        _strings(file, "descriptor_columns", list(INTERVENTION_DESCRIPTOR_COLUMNS))
        file.create_dataset("time_s", data=time)
        root = file.create_group("sessions")
        for session_number in range(sessions):
            session_key = f"day-{session_number:02d}_synthetic-{session_number}"
            group = root.create_group(session_key)
            group.attrs["animal_id"] = animal
            group.attrs["session_id"] = f"synthetic-{session_number}"
            trials = group.create_group("trials")
            trial_index = np.concatenate(
                (np.arange(1, task_trials + 1), -np.arange(1, normal_iti + 1))
            )
            stimulated = np.zeros(count, dtype=bool)
            stimulated[1:task_trials:2] = True
            catch = ~stimulated
            catch[task_trials:] = False
            if no_catches:
                catch[:task_trials] = False
                stimulated[:task_trials] = True
            iti = np.zeros(count, dtype=bool)
            iti[task_trials:] = True
            normal = catch | iti
            trials.create_dataset("trial_index", data=trial_index)
            trials.create_dataset("is_normal_calibration", data=normal)
            trials.create_dataset("is_catch", data=catch)
            trials.create_dataset("is_iti_calibration", data=iti)
            event_start = np.full(count, np.nan)
            event_stop = np.full(count, np.nan)
            event_start[stimulated] = 0.0
            event_stop[stimulated] = 0.7
            _strings(
                trials,
                "window_kind",
                ["normal" if flag else "intervention" for flag in normal],
            )
            _strings(
                trials,
                "normal_source",
                [
                    "catch" if catch[index] else ("iti" if iti[index] else "none")
                    for index in range(count)
                ],
            )
            descriptors = np.zeros((count, len(INTERVENTION_DESCRIPTOR_COLUMNS)), dtype=np.float32)
            stim_rows = np.flatnonzero(stimulated)
            for order, row_index in enumerate(stim_rows):
                current = 1.5 + (order % 4) * 2.5
                depth = float((order % 3) * 60 + 480)
                descriptors[row_index] = np.asarray(
                    [
                        1.0,
                        current,
                        100.0,
                        70.0,
                        167.0,
                        0.0,
                        depth,
                        0.0,
                        depth - 930.0,
                        (depth - 930.0) / 930.0,
                    ]
                )
            # One long train per session is explicitly outside the primary
            # 70-pulse family and must never be silently projected to it.
            descriptors[stim_rows[-1], 3] = 800.0
            event_stop[stim_rows[-1]] = 8.0
            trials.create_dataset("event_start_time", data=event_start)
            trials.create_dataset("event_stop_time", data=event_stop)
            group.create_dataset("intervention_descriptors", data=descriptors)
            units = 2 + session_number
            signals = group.create_group("signals")
            base = (
                np.arange(count)[:, None, None] * 0.1
                + np.arange(len(time))[None, :, None] * 0.01
                + np.arange(units)[None, None, :] * 0.02
            )
            spikes = np.broadcast_to(base, (count, len(time), units)).copy()
            # If a target stimulation row leaked into prepare, this sentinel
            # would be unmistakable in its query/support artifact.
            spikes[stimulated] += 9999.0
            spike_mask = np.ones_like(spikes, dtype=bool)
            artifact = (time >= 0) & (time < 0.705)
            spike_mask &= ~(stimulated[:, None, None] & artifact[None, :, None])
            signals.create_dataset("spike_rate_hz", data=spikes)
            signals.create_dataset("spike_valid_mask", data=spike_mask)
            wheel = np.arange(count)[:, None] * 0.01 + time[None, :]
            wheel[stimulated] += 3.0
            signals.create_dataset("wheel_displacement", data=wheel)
            signals.create_dataset("wheel_velocity", data=2.0 * wheel)
            signals.create_dataset("wheel_valid_mask", data=np.ones_like(wheel, dtype=bool))
            unit_group = group.create_group("units")
            unit_group.create_dataset("unit_id", data=np.arange(units))


@pytest.fixture
def processed_root(tmp_path: Path) -> Path:
    root = tmp_path / "processed"
    root.mkdir()
    rows = []
    for animal in TASK_MICE:
        path = root / f"sub-{animal}.h5"
        _write_processed_animal(
            path,
            animal,
            sessions=2 if animal == "ICMS92" else 1,
            no_catches=animal == "ICMS83",
        )
        rows.append(
            {
                "animal_id": animal,
                "output": str(path),
                "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (root / "index.json").write_text(
        json.dumps(
            {
                "schema": "cadence-dandi-001868-index-v1",
                "dandiset_id": DANDISET_ID,
                "dandiset_version": DANDISET_VERSION,
                "animals": rows,
            }
        ),
        encoding="utf-8",
    )
    return root


@pytest.fixture
def tagged_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "tagged"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.org"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test Runner"],
        cwd=repository,
        check=True,
    )
    (repository / "frozen.txt").write_text("protocol\n", encoding="utf-8")
    subprocess.run(["git", "add", "frozen.txt"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "freeze"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "tag",
            "-a",
            icms.PREOUTCOME_TAG,
            "-m",
            "pre-outcome freeze",
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
    return repository, commit


def _small_config() -> icms.ICMSExperimentConfig:
    base = icms.make_icms_config("smoke", device="cpu")
    return replace(
        base,
        max_normal_trials_per_session=12,
        max_stimulation_trials_per_session=8,
        query_contexts_per_session=2,
        uncertainty_draws=2,
    )


def test_prepare_uses_session_adapters_and_never_mounts_target_stimulation(
    processed_root: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "fold"
    result = icms.prepare_fold(
        processed_root=processed_root,
        output_directory=output,
        target_animal="ICMS92",
        config=_small_config(),
        protocol_commit="a" * 40,
        run_mode="synthetic",
    )
    target_rows = [row for row in result["normal_supports"] if row["animal_id"] == "ICMS92"]
    assert len(target_rows) == 2
    assert {row["neural_channels"] for row in target_rows} == {2, 3}
    assert len({row["adapter_id"] for row in target_rows}) == 2
    assert all(row["adapter_id"].startswith("ICMS92::") for row in target_rows)
    assert len(result["target_queries"]) == 2
    for row in result["target_queries"]:
        with np.load(output / row["path"], allow_pickle=False) as query:
            assert not any(
                token in name for name in query.files for token in icms._FORBIDDEN_QUERY_TOKENS
            )
            assert np.max(query["pre_neural"]) < 100.0
            assert query["condition_descriptors"].shape == (32 * 13, 10)
    assert result["access_audit"]["prepare_target_stimulation_metadata_read"] is False
    assert result["access_audit"]["prepare_target_stimulation_signals_read"] is False
    assert ((processed_root / "sub-ICMS92.h5").stat().st_mode & 0o444) == 0
    assert (output / "prepare_complete.json").exists()
    assert result["canonical_outer_mapping"]["ICMS92"] == [
        animal for animal in TASK_MICE if animal != "ICMS92"
    ]


def test_continuous_current_interpolation_and_block_rules() -> None:
    config = _small_config()
    lattice = icms.physical_query_lattice(config)
    values = lattice[:, 1, None]
    descriptor = lattice[np.flatnonzero(lattice[:, 6] == 480.0)[0]].copy()
    descriptor[1] = 2.5
    interpolated = icms._lattice_interpolation(values, lattice, descriptor)
    assert interpolated.item() == pytest.approx(2.5)
    noncanonical = descriptor.copy()
    noncanonical[3] = 800
    assert not icms.canonical_icms_mask(noncanonical[None, :])[0]
    with pytest.raises(ProtocolViolation, match="noncanonical"):
        icms._lattice_interpolation(values, lattice, noncanonical)

    blocks, rule, valid = icms.derive_task_blocks(np.arange(1, 251))
    assert valid and "floor" in rule
    assert blocks[[0, 99, 100, 249]].tolist() == [0, 0, 1, 2]
    reset, _, valid = icms.derive_task_blocks(np.concatenate((np.arange(1, 101), np.arange(1, 31))))
    assert valid and reset[-1] == 1
    fallback, rule, valid = icms.derive_task_blocks([1, 200, 2])
    assert not valid and np.all(fallback == 0) and "fallback" in rule


def test_primary_block_estimand_never_uses_session_catch_fallback() -> None:
    shape = (2, 3, 1)
    ones = np.ones(shape, dtype=bool)
    outcome = icms.OutcomeSession(
        animal_id="ICMS92",
        session_key="session",
        session_id="session",
        time_s=np.arange(3),
        descriptors=np.zeros((2, 10)),
        trial_index=np.asarray([1, 101]),
        blocks=np.asarray([0, 1]),
        neural=np.asarray([[[1.0], [2.0], [3.0]], [[2.0], [3.0], [4.0]]]),
        neural_mask=ones,
        behavior=np.ones((2, 3, 2)),
        behavior_mask=np.ones((2, 3, 2), dtype=bool),
        catch_blocks=np.asarray([0]),
        catch_neural=np.zeros((1, 3, 1)),
        catch_neural_mask=np.ones((1, 3, 1), dtype=bool),
        catch_behavior=np.zeros((1, 3, 2)),
        catch_behavior_mask=np.ones((1, 3, 2), dtype=bool),
        iti_neural=np.empty((0, 3, 1)),
        iti_neural_mask=np.empty((0, 3, 1), dtype=bool),
        iti_behavior=np.empty((0, 3, 2)),
        iti_behavior_mask=np.empty((0, 3, 2), dtype=bool),
        block_rule="validated fixture",
        block_validated=True,
        positive_trial_count=2,
        excluded_noncanonical_count=0,
    )
    result = icms._observed_condition(outcome, np.asarray([0, 1]))
    assert result["primary_randomized_status"] == "NOT_EVALUATED"
    assert not result["same_block_catch_supported"]
    assert result["missing_same_block_catches"] == 1
    assert not np.any(result["neural_effect_mask"])
    assert np.any(result["session_fallback_neural_effect_mask"])

    outcome.blocks[:] = 0
    outcome.neural_mask[:] = False
    outcome.behavior_mask[:] = False
    outcome.catch_neural_mask[:] = False
    outcome.catch_behavior_mask[:] = False
    invalid = icms._observed_condition(outcome, np.asarray([0, 1]))
    assert invalid["primary_randomized_status"] == "NOT_EVALUATED"
    assert invalid["valid_primary_neural_cells"] == 0
    assert invalid["valid_primary_behavior_cells"] == 0


def test_encoder_observation_appends_missingness_and_never_hides_zero_fill() -> None:
    values = np.asarray([[[2.0, 999.0]]])
    mask = np.asarray([[[True, False]]])
    augmented, augmented_mask = icms.augment_neural_with_mask(values, mask)
    assert augmented[0, 0].tolist() == [2.0, 0.0, 1.0, 0.0]
    assert augmented_mask[0, 0].tolist() == [True, False, True, True]
    assert set(icms.REPORT_METHODS) == {
        "proposed",
        "linear",
        "additive",
        "black_box",
        "zero_effect",
        "condition_time",
        "nearest_donor",
    }
    assert {
        method: type(icms._model_for_method(method, _small_config())).__name__
        for method in icms.LEARNED_METHODS
    } == {
        "proposed": "HierarchicalControlledSSM",
        "linear": "LinearHierarchicalSSM",
        "additive": "AdditiveInterventionSSM",
        "black_box": "BlackBoxMetaGRU",
    }


def test_locked_stages_refuse_without_acknowledgement(tmp_path: Path) -> None:
    with pytest.raises(ProtocolViolation, match="donor stimulation outcomes remain closed"):
        icms.predict_fold(
            fold_directory=tmp_path,
            config=_small_config(),
            acknowledge_donor_outcomes=False,
        )
    with pytest.raises(ProtocolViolation, match="target stimulation outcomes remain sealed"):
        icms.score_fold(
            fold_directory=tmp_path,
            acknowledge_target_outcomes=False,
        )


def test_biological_scope_rejects_smoke_subsets_and_overwrite(tmp_path: Path) -> None:
    with pytest.raises(ProtocolViolation, match="full optimization"):
        icms._require_biological_config(_small_config())
    with pytest.raises(ProtocolViolation, match="complete ordered"):
        icms._require_biological_methods(("proposed", "zero_effect"))
    with pytest.raises(ProtocolViolation, match="append-only"):
        icms.prepare_fold(
            processed_root=tmp_path,
            output_directory=tmp_path / "out",
            target_animal="ICMS92",
            config=icms.make_icms_config("full", seed=20260725, device="cpu"),
            run_mode="biological",
            overwrite=True,
        )


def test_canonical_index_requires_byte_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "canonical-copy"
    root.mkdir()
    canonical = icms.SOURCE_ROOT / icms.CANONICAL_INDEX_RELATIVE
    provided = root / "index.json"
    provided.write_bytes(canonical.read_bytes())

    def fake_identity(relative: Path, commit: str) -> dict[str, str]:
        path = icms.SOURCE_ROOT / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "relative_path": relative.as_posix(),
            "sha256": digest,
            "git_blob_sha256": digest,
        }

    monkeypatch.setattr(icms, "_tracked_file_identity", fake_identity)
    provenance = icms._canonical_provenance(root, commit="a" * 40, verify_h5=False)
    assert provenance["index_totals"] == icms.CANONICAL_INDEX_TOTALS
    assert provenance["dandiset_version"] == DANDISET_VERSION
    with pytest.raises(ProtocolViolation, match="synthetic mode cannot open"):
        icms.prepare_fold(
            processed_root=root,
            output_directory=tmp_path / "forbidden-synthetic",
            target_animal="ICMS92",
            config=_small_config(),
            protocol_commit="a" * 40,
            run_mode="synthetic",
        )
    provided.write_bytes(provided.read_bytes() + b"\n")
    with pytest.raises(ProtocolViolation, match="byte-identical"):
        icms._canonical_provenance(root, commit="a" * 40, verify_h5=False)


def test_nonlearned_synthetic_fold_hashes_before_score_and_marks_icms83_ineligible(
    processed_root: Path,
    tagged_repository: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repository, commit = tagged_repository
    output = tmp_path / "fold-ICMS83"
    config = _small_config()
    target_path = processed_root / "sub-ICMS83.h5"
    original_target_mode = target_path.stat().st_mode & 0o777
    original_target_bytes = target_path.read_bytes()
    icms.prepare_fold(
        processed_root=processed_root,
        output_directory=output,
        target_animal="ICMS83",
        config=config,
        protocol_commit=commit,
        run_mode="synthetic",
    )
    opened_animals: list[str] = []
    original_read = icms.read_stimulation_sessions

    def audited_read(
        path: str | Path,
        supports: dict[str, icms.NormalSession],
        *,
        config: icms.ICMSExperimentConfig,
    ) -> list[icms.StimSession]:
        with h5py.File(path, "r") as file:
            opened_animals.append(str(file.attrs["animal_id"]))
        return original_read(path, supports, config=config)

    monkeypatch.setattr(icms, "read_stimulation_sessions", audited_read)
    prediction = icms.predict_fold(
        fold_directory=output,
        config=config,
        methods=("zero_effect", "condition_time", "nearest_donor"),
        acknowledge_donor_outcomes=True,
        run_mode="synthetic",
    )
    assert set(opened_animals) == set(TASK_MICE) - {"ICMS83"}
    assert prediction["access_audit"]["target_stimulation_outcomes_read"] is False
    assert prediction["access_audit"]["target_stimulation_metadata_read"] is False
    assert prediction["access_audit"]["donor_noncanonical_trains_excluded"] > 0
    assert prediction["access_audit"]["physical_target_seal_asserted_before_donor_open"]
    with pytest.raises(FileExistsError, match="prediction artifacts already exist"):
        icms.predict_fold(
            fold_directory=output,
            config=config,
            methods=("zero_effect", "condition_time", "nearest_donor"),
            acknowledge_donor_outcomes=True,
            run_mode="synthetic",
        )

    prediction_path = output / "predictions.npz"
    original_prediction = prediction_path.read_bytes()
    prediction_path.write_bytes(original_prediction + b"tampered")
    target_opened = False
    original_materialize = icms.materialize_target_outcomes

    def audited_materialize(**kwargs: object) -> object:
        nonlocal target_opened
        target_opened = True
        return original_materialize(**kwargs)

    monkeypatch.setattr(icms, "materialize_target_outcomes", audited_materialize)
    with pytest.raises(ProtocolViolation, match="immutable artifact changed"):
        icms.score_fold(
            fold_directory=output,
            acknowledge_target_outcomes=True,
            run_mode="synthetic",
        )
    assert not target_opened
    assert target_path.stat().st_mode & 0o444 == 0
    prediction_path.write_bytes(original_prediction)

    target_path.chmod(original_target_mode)
    target_path.write_bytes(original_target_bytes + b"tampered")
    target_path.chmod(0)
    with pytest.raises(ProtocolViolation, match="changed while sealed"):
        icms.score_fold(
            fold_directory=output,
            acknowledge_target_outcomes=True,
            run_mode="synthetic",
        )
    assert (target_path.stat().st_mode & 0o777) == 0
    assert (processed_root / icms.ACTIVE_SEAL_NAME).exists()
    assert not (output / "target_restore.json").exists()
    target_path.chmod(original_target_mode)
    target_path.write_bytes(original_target_bytes)
    target_path.chmod(0)

    score = icms.score_fold(
        fold_directory=output,
        acknowledge_target_outcomes=True,
        run_mode="synthetic",
    )
    assert target_opened
    assert score["causal_effect_eligibility"]["animal_eligible"] is False
    assert score["causal_effect_eligibility"]["design_maximum_primary_eligible_n"] == 5
    assert "primary_six_fold_eligible_n" not in score["causal_effect_eligibility"]
    assert score["causal_effect_eligibility"]["iti_is_randomized_counterfactual"] is False
    assert (
        sum(row["noncanonical_trains_excluded"] for row in score["outcome_audit"]["sessions"]) > 0
    )
    assert score["access_audit"]["prediction_hash_verified_before_target_open"]
    assert score["access_audit"]["target_h5_original_mode_restored_exactly"]
    assert score["access_audit"]["scoring_channel_scales_fit_partition"] == "normal_fit"
    assert score["access_audit"]["distinct_unmasked_full_train_spike_tensor_available"] is False
    assert score["access_audit"]["full_train_spike_sensitivity_status"] == ("NOT_EVALUATED")
    assert score["access_audit"]["calcium_read_or_scored_by_v1_experiment"] is False
    assert score["access_audit"]["sparse_calcium_secondary_status"] == "NOT_EVALUATED"
    for session in score["session_scores"].values():
        for method_score in session.values():
            assert "authors_full_train_neural_skill_sensitivity" not in method_score
    assert target_path.stat().st_mode & 0o777 == original_target_mode
    assert not (processed_root / icms.ACTIVE_SEAL_NAME).exists()
    assert (output / "sealed_target_outcomes.npz").exists()
    assert (output / "scored_condition_trajectories.npz").exists()
    with np.load(output / "scored_condition_trajectories.npz", allow_pickle=False) as data:
        assert not any("full_train" in name for name in data.files)
    assert (output / "score_complete.json").exists()
    with pytest.raises(FileExistsError, match="score artifacts already exist"):
        icms.score_fold(
            fold_directory=output,
            acknowledge_target_outcomes=True,
            run_mode="synthetic",
        )


def test_predict_refuses_readable_target_before_any_donor_open(
    processed_root: Path,
    tagged_repository: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repository, commit = tagged_repository
    output = tmp_path / "broken-seal"
    config = _small_config()
    target_path = processed_root / "sub-ICMS92.h5"
    original_mode = target_path.stat().st_mode & 0o777
    icms.prepare_fold(
        processed_root=processed_root,
        output_directory=output,
        target_animal="ICMS92",
        config=config,
        protocol_commit=commit,
        run_mode="synthetic",
    )
    target_path.chmod(original_mode)
    donor_opened = False

    def forbidden_read(*args: object, **kwargs: object) -> object:
        nonlocal donor_opened
        donor_opened = True
        raise AssertionError("donor loader must not run after seal failure")

    monkeypatch.setattr(icms, "read_stimulation_sessions", forbidden_read)
    with pytest.raises(ProtocolViolation, match="readable during predict"):
        icms.predict_fold(
            fold_directory=output,
            config=config,
            methods=("zero_effect",),
            acknowledge_donor_outcomes=True,
            run_mode="synthetic",
        )
    assert not donor_opened


def test_predict_refuses_substituted_target_inode_before_any_donor_open(
    processed_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "substituted-seal"
    config = _small_config()
    target_path = processed_root / "sub-ICMS92.h5"
    icms.prepare_fold(
        processed_root=processed_root,
        output_directory=output,
        target_animal="ICMS92",
        config=config,
        protocol_commit="c" * 40,
        run_mode="synthetic",
    )
    sealed_original = processed_root / "sub-ICMS92.sealed-original.h5"
    target_path.rename(sealed_original)
    target_path.touch(mode=0)
    donor_opened = False

    def forbidden_read(*args: object, **kwargs: object) -> object:
        nonlocal donor_opened
        donor_opened = True
        raise AssertionError("donor loader must not run after inode substitution")

    monkeypatch.setattr(icms, "read_stimulation_sessions", forbidden_read)
    with pytest.raises(ProtocolViolation, match="file identity"):
        icms.predict_fold(
            fold_directory=output,
            config=config,
            methods=("zero_effect",),
            acknowledge_donor_outcomes=True,
            run_mode="synthetic",
        )
    assert not donor_opened


def test_prepare_refuses_a_second_fold_while_a_target_seal_is_active(
    processed_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_config()
    icms.prepare_fold(
        processed_root=processed_root,
        output_directory=tmp_path / "first-fold",
        target_animal="ICMS92",
        config=config,
        protocol_commit="b" * 40,
        run_mode="synthetic",
    )
    source_opened = False

    def forbidden_read(*args: object, **kwargs: object) -> object:
        nonlocal source_opened
        source_opened = True
        raise AssertionError("no source should open while a physical seal is active")

    monkeypatch.setattr(icms, "read_normal_sessions", forbidden_read)
    with pytest.raises(ProtocolViolation, match="active physical target seal"):
        icms.prepare_fold(
            processed_root=processed_root,
            output_directory=tmp_path / "second-fold",
            target_animal="ICMS93",
            config=config,
            protocol_commit="b" * 40,
            run_mode="synthetic",
        )
    assert not source_opened


def test_biological_output_is_exact_canonical_path_and_rejects_copy_or_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(icms, "SOURCE_ROOT", repository)
    canonical = repository / "results/icms/loao-ICMS92"
    canonical.mkdir(parents=True)
    assert (
        icms._require_canonical_biological_output(canonical, "ICMS92") == "results/icms/loao-ICMS92"
    )
    copied = tmp_path / "copy/results/icms/loao-ICMS92"
    copied.mkdir(parents=True)
    with pytest.raises(ProtocolViolation, match="canonical one-shot"):
        icms._require_canonical_biological_output(copied, "ICMS92")

    canonical.rmdir()
    (repository / "results/icms").rmdir()
    (repository / "results").rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (repository / "results").symlink_to(outside, target_is_directory=True)
    symlinked = repository / "results/icms/loao-ICMS92"
    symlinked.mkdir(parents=True)
    with pytest.raises(ProtocolViolation, match="symlink"):
        icms._require_canonical_biological_output(symlinked, "ICMS92")


def test_icms_score_crash_reseals_then_finalizes_only_after_score_commit(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    directory = tmp_path / "loao-ICMS92"
    processed.mkdir()
    directory.mkdir()
    target = processed / "sub-ICMS92.h5"
    target.write_bytes(b"synthetic target identity")
    target.chmod(0o600)
    expected = hashlib.sha256(target.read_bytes()).hexdigest()
    canonical = "results/icms/loao-ICMS92"
    seal, seal_sha = icms._seal_target_source(
        target_path=target,
        processed_root=processed,
        fold_directory=directory,
        target_animal="ICMS92",
        expected_sha256=expected,
        canonical_relative_output=canonical,
    )
    freeze = {"commit": "a" * 40, "tag": "pre-outcome-v1.0.0"}
    prepare_manifest = {
        "physical_target_seal": {**seal, "sha256": seal_sha},
        "canonical_relative_output": canonical,
        "freeze_attestation": freeze,
        "target_seal_transaction_sha256": seal_sha,
    }
    prepare_path = directory / "prepare_manifest.json"
    prepare_sha = icms._atomic_json(prepare_path, prepare_manifest)
    icms._write_stage_completion(
        directory,
        stage="prepare",
        artifact_path=prepare_path,
        artifact_sha256=prepare_sha,
        freeze=freeze,
        canonical_relative_output=canonical,
        seal_transaction_sha256=seal_sha,
    )
    prediction_path = directory / "prediction_manifest.json"
    prediction_sha = icms._atomic_json(
        prediction_path,
        {
            "schema": "test",
            "freeze_attestation": freeze,
            "canonical_relative_output": canonical,
            "target_seal_transaction_sha256": seal_sha,
        },
    )
    icms._write_stage_completion(
        directory,
        stage="predict",
        artifact_path=prediction_path,
        artifact_sha256=prediction_sha,
        freeze=freeze,
        canonical_relative_output=canonical,
        seal_transaction_sha256=seal_sha,
    )

    # Simulate a hard stop after target restoration and a partial score write.
    target.chmod(0o600)
    icms._atomic_json(
        directory / "target_restore.json",
        {
            "schema": "cadence-icms-target-restore-v1",
            "canonical_relative_output": canonical,
        },
    )
    (directory / "metrics.json").write_text("partial")
    (directory / "sealed_target_outcomes.npz").write_bytes(b"readable interrupted target outcomes")
    (directory / "sealed_target_outcomes.npz").chmod(0o600)
    outcome_temp = directory / ".sealed_target_outcomes.npz.crash.tmp"
    outcome_temp.write_bytes(b"readable interrupted target outcome temp")
    outcome_temp.chmod(0o600)
    assert (
        icms._recover_icms_stage(
            directory=directory,
            prepare_manifest=prepare_manifest,
            stage="score",
            canonical_relative_output=canonical,
        )
        is None
    )
    assert stat.S_IMODE(target.stat().st_mode) == 0
    assert (processed / icms.ACTIVE_SEAL_NAME).is_file()
    assert not (directory / "metrics.json").exists()
    quarantines = list(tmp_path.glob(".loao-ICMS92.interrupted-stage-*"))
    assert len(quarantines) == 1
    for name in (
        "sealed_target_outcomes.npz",
        ".sealed_target_outcomes.npz.crash.tmp",
    ):
        quarantined = quarantines[0] / name
        assert stat.S_IMODE(quarantined.stat().st_mode) == 0
        assert not quarantined.stat().st_mode & stat.S_IRUSR

    # A committed score is the boundary after which restoration may finalize.
    target.chmod(0o600)
    restore_sha = icms._atomic_json(
        directory / "target_restore.json",
        {
            "schema": "cadence-icms-target-restore-v1",
            "canonical_relative_output": canonical,
        },
    )
    metrics_path = directory / "metrics.json"
    metrics_sha = icms._atomic_json(
        metrics_path,
        {
            "schema": "test-score",
            "freeze_attestation": freeze,
            "canonical_relative_output": canonical,
            "target_seal_transaction_sha256": seal_sha,
        },
    )
    icms._write_stage_completion(
        directory,
        stage="score",
        artifact_path=metrics_path,
        artifact_sha256=metrics_sha,
        freeze=freeze,
        canonical_relative_output=canonical,
        seal_transaction_sha256=seal_sha,
    )
    assert restore_sha == icms._verify_sidecar(directory / "target_restore.json")
    metrics_text = metrics_path.read_text(encoding="utf-8")
    score_completion_path = directory / "score_complete.json"
    score_completion_text = score_completion_path.read_text(encoding="utf-8")
    rebound_metrics = json.loads(metrics_text)
    rebound_metrics["target_seal_transaction_sha256"] = "b" * 64
    metrics_path.write_text(
        json.dumps(rebound_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rebound_metrics_sha = icms.hash_file(metrics_path)
    (directory / "metrics.json.sha256").write_text(
        f"{rebound_metrics_sha}  metrics.json\n",
        encoding="utf-8",
    )
    rebound_score_completion = json.loads(score_completion_text)
    rebound_score_completion["artifact_sha256"] = rebound_metrics_sha
    rebound_score_completion["seal_transaction_sha256"] = "b" * 64
    score_completion_path.write_text(
        json.dumps(rebound_score_completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rebound_score_sha = icms.hash_file(score_completion_path)
    (directory / "score_complete.json.sha256").write_text(
        f"{rebound_score_sha}  score_complete.json\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolViolation, match="score completion transaction"):
        icms._finalize_icms_target_restore(
            directory,
            prepare_manifest,
            canonical_relative_output=canonical,
        )
    assert (processed / icms.ACTIVE_SEAL_NAME).is_file()
    assert stat.S_IMODE(target.stat().st_mode) == 0
    metrics_path.write_text(metrics_text, encoding="utf-8")
    metrics_sha = icms.hash_file(metrics_path)
    (directory / "metrics.json.sha256").write_text(
        f"{metrics_sha}  metrics.json\n",
        encoding="utf-8",
    )
    score_completion_path.write_text(score_completion_text, encoding="utf-8")
    score_completion_sha = icms.hash_file(score_completion_path)
    (directory / "score_complete.json.sha256").write_text(
        f"{score_completion_sha}  score_complete.json\n",
        encoding="utf-8",
    )
    target.chmod(0o600)
    finalization = icms._finalize_icms_target_restore(
        directory,
        prepare_manifest,
        canonical_relative_output=canonical,
    )
    assert finalization["restored_after_score_commit"] is True
    assert not (processed / icms.ACTIVE_SEAL_NAME).exists()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert (directory / "target_restore_complete.json.sha256").is_file()

    # A pair is not trusted from existence alone: authenticate its sidecar and bindings.
    completion_path = directory / "target_restore_complete.json"
    sidecar = directory / "target_restore_complete.json.sha256"
    valid_payload = json.loads(completion_path.read_text(encoding="utf-8"))

    target.chmod(0o400)
    with pytest.raises(ProtocolViolation, match="restored ICMS target mode changed"):
        icms._recover_icms_stage(
            directory=directory,
            prepare_manifest=prepare_manifest,
            stage="score",
            canonical_relative_output=canonical,
        )
    target.chmod(0o600)

    original_target_bytes = target.read_bytes()
    target.write_bytes(original_target_bytes + b"digest drift")
    with pytest.raises(ProtocolViolation, match="restored ICMS target digest changed"):
        icms._recover_icms_stage(
            directory=directory,
            prepare_manifest=prepare_manifest,
            stage="score",
            canonical_relative_output=canonical,
        )
    target.write_bytes(original_target_bytes)

    original_target = target.with_suffix(".original")
    target.rename(original_target)
    try:
        target.write_bytes(original_target_bytes)
        target.chmod(0o600)
        with pytest.raises(ProtocolViolation, match="restored ICMS target identity changed"):
            icms._recover_icms_stage(
                directory=directory,
                prepare_manifest=prepare_manifest,
                stage="score",
                canonical_relative_output=canonical,
            )
    finally:
        target.unlink(missing_ok=True)
        original_target.rename(target)

    sidecar.write_text(f"{'0' * 64}  {completion_path.name}\n", encoding="utf-8")
    with pytest.raises(ProtocolViolation):
        icms._recover_icms_stage(
            directory=directory,
            prepare_manifest=prepare_manifest,
            stage="score",
            canonical_relative_output=canonical,
        )
    for key, changed in (
        ("schema", "cadence-icms-target-restore-completion-rebound"),
        ("canonical_relative_output", "results/icms/loao-ICMS93"),
        ("score_completion_sha256", "f" * 64),
        ("seal_transaction_sha256", "e" * 64),
    ):
        rebound = {**valid_payload, key: changed}
        completion_path.write_text(
            json.dumps(rebound, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        digest = icms.hash_file(completion_path)
        sidecar.write_text(f"{digest}  {completion_path.name}\n", encoding="utf-8")
        with pytest.raises(ProtocolViolation, match="binding"):
            icms._recover_icms_stage(
                directory=directory,
                prepare_manifest=prepare_manifest,
                stage="score",
                canonical_relative_output=canonical,
            )


def test_icms_stage_completion_rejects_rebound_control_fields(tmp_path: Path) -> None:
    directory = tmp_path / "fold"
    directory.mkdir()
    freeze = {"commit": "a" * 40, "tag": "pre-outcome-v1.0.0"}
    transaction_sha256 = "b" * 64
    canonical = "results/icms/loao-ICMS92"
    artifact = directory / "prepare_manifest.json"
    artifact_sha256 = icms._atomic_json(
        artifact,
        {
            "schema": "test-prepare",
            "freeze_attestation": freeze,
            "canonical_relative_output": canonical,
            "target_seal_transaction_sha256": transaction_sha256,
        },
    )
    completion, _ = icms._write_stage_completion(
        directory,
        stage="prepare",
        artifact_path=artifact,
        artifact_sha256=artifact_sha256,
        freeze=freeze,
        canonical_relative_output=canonical,
        seal_transaction_sha256=transaction_sha256,
    )
    valid_completion = json.loads(completion.read_text(encoding="utf-8"))

    def rewrite(path: Path, payload: dict[str, object]) -> None:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        digest = icms.hash_file(path)
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{digest}  {path.name}\n",
            encoding="utf-8",
        )

    mutations = (
        {"schema": "rebound"},
        {"stage": "score"},
        {"artifact": "/tmp/metrics.json"},
        {"artifact": "../prepare_manifest.json"},
        {"append_only": False},
        {"freeze_attestation": {"commit": "c" * 40, "tag": "other"}},
        {"canonical_relative_output": "results/icms/loao-ICMS93"},
        {"seal_transaction_sha256": "d" * 64},
    )
    for mutation in mutations:
        rewrite(completion, {**valid_completion, **mutation})
        with pytest.raises(ProtocolViolation, match="invalid ICMS prepare completion"):
            icms._verify_stage_completion(directory, "prepare")

    valid_artifact = json.loads(artifact.read_text(encoding="utf-8"))
    rebound_artifact = {
        **valid_artifact,
        "target_seal_transaction_sha256": "e" * 64,
    }
    rewrite(artifact, rebound_artifact)
    rebound_artifact_sha = icms.hash_file(artifact)
    rewrite(
        completion,
        {
            **valid_completion,
            "artifact_sha256": rebound_artifact_sha,
        },
    )
    with pytest.raises(ProtocolViolation, match="invalid ICMS prepare completion"):
        icms._verify_stage_completion(directory, "prepare")


def test_icms_prepare_recovery_rejects_registry_path_rebinding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed = tmp_path / "processed"
    directory = tmp_path / "loao-ICMS92"
    processed.mkdir()
    directory.mkdir()
    target = processed / "sub-ICMS92.h5"
    target.write_bytes(b"target")
    target.chmod(0o600)
    expected = icms.hash_file(target)
    rows = [
        {
            "animal_id": animal,
            "output": str(target if animal == "ICMS92" else processed / f"sub-{animal}.h5"),
            "output_sha256": expected if animal == "ICMS92" else "0" * 64,
        }
        for animal in TASK_MICE
    ]
    (processed / "index.json").write_text(
        json.dumps({"dandiset_id": DANDISET_ID, "animals": rows}),
        encoding="utf-8",
    )
    canonical = "results/icms/loao-ICMS92"
    monkeypatch.setattr(
        icms,
        "_canonical_icms_registry_scope",
        lambda _target: (directory.resolve(), canonical),
    )
    icms._seal_target_source(
        target_path=target,
        processed_root=processed,
        fold_directory=directory,
        target_animal="ICMS92",
        expected_sha256=expected,
        canonical_relative_output=canonical,
    )
    registry = processed / icms.ACTIVE_SEAL_NAME
    original_registry = registry.read_text(encoding="utf-8")
    rebound = json.loads(original_registry)
    victim = tmp_path / "must-not-be-chmodded"
    victim.write_bytes(b"preserve")
    victim.chmod(0o600)
    rebound["target_path"] = str(victim)
    registry.write_text(
        json.dumps(rebound, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProtocolViolation, match="registry binding changed"):
        icms._recover_icms_prepare(
            processed_root=processed,
            directory=directory,
            target_animal="ICMS92",
            canonical_relative_output=canonical,
        )
    assert victim.read_bytes() == b"preserve"
    assert stat.S_IMODE(victim.stat().st_mode) == 0o600
    registry.write_text(original_registry, encoding="utf-8")
    assert (
        icms._recover_icms_prepare(
            processed_root=processed,
            directory=directory,
            target_animal="ICMS92",
            canonical_relative_output=canonical,
        )
        is None
    )
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_proposed_model_smoke_uses_session_maps_and_animal_grouped_deltas(
    processed_root: Path,
    tagged_repository: tuple[Path, str],
    tmp_path: Path,
) -> None:
    _repository, commit = tagged_repository
    output = tmp_path / "learned-fold"
    base = _small_config()
    one_epoch = FitConfig(
        learning_rate=1e-3,
        max_epochs=1,
        patience=1,
        seed=7,
        device="cpu",
        mixed_precision=False,
    )
    config = replace(
        base,
        latent_dim=4,
        hidden_dim=8,
        normal_fit=one_epoch,
        intervention_fit=replace(one_epoch, seed=8),
        target_fit=replace(one_epoch, seed=9),
    )
    icms.prepare_fold(
        processed_root=processed_root,
        output_directory=output,
        target_animal="ICMS92",
        config=config,
        protocol_commit=commit,
        run_mode="synthetic",
    )
    result = icms.predict_fold(
        fold_directory=output,
        config=config,
        methods=("proposed",),
        acknowledge_donor_outcomes=True,
        run_mode="synthetic",
    )
    assert result["access_audit"]["session_specific_observation_maps"]
    assert result["access_audit"]["encoder_receives_explicit_missingness_channels"]
    assert not result["access_audit"]["zero_filled_missing_bins_without_mask_channel"]
    assert result["access_audit"]["donor_delta_grouping"] == "animal_id"
    assert result["access_audit"]["pooled_raw_unit_baseline_available"] is False
    assert set(result["fit_audits"]) == {"proposed"}
    topology = result["fit_audits"]["proposed"]
    validation_animal = topology["intervention_inner_validation_animal"]
    expected_donors = set(TASK_MICE) - {"ICMS92"}
    assert set(topology["normal_selection_training_animals"]) == (
        expected_donors - {validation_animal}
    )
    assert topology["validation_normal_gradient_to_shared_f"] is False
    assert (
        topology["shared_f_before_validation_normal_sha256"]
        == topology["shared_f_after_validation_normal_sha256"]
    )
    assert (
        topology["target_adaptation_nonadapter_state_before_sha256"]
        == topology["target_adaptation_nonadapter_state_after_sha256"]
    )
    assert topology["final_model_is_fresh"]
    assert set(topology["final_normal_refit"]["normal_refit_animals"]) == expected_donors
    delta_audit = result["fit_audits"]["proposed"]["intervention_selection_delta_audit"]
    assert delta_audit["validation_delta_frozen_zero_during_selection"]
    assert delta_audit["validation_delta_requires_grad"] is False
    assert delta_audit["validation_delta_l2_norm"] == 0.0
    assert delta_audit["maximum_validation_delta_shrinkage_term"] == 0.0
    assert delta_audit["validation_delta_in_shrinkage"] is False
    assert delta_audit["validation_delta_centering_applied"] is False
    assert set(delta_audit["centering_group_animals"]) == (expected_donors - {validation_animal})
    assert delta_audit["centering_excluded_animals"] == [validation_animal]
    assert delta_audit["validation_delta_group"] not in set(delta_audit["centering_group_keys"])
    assert delta_audit["identification_constraint"] == "exact_zero_mean_projection"
    assert delta_audit["projection_calls"] == delta_audit["optimizer_steps"] + 1
    assert delta_audit["maximum_post_step_projection_residual_norm"] <= 1e-7
    refit_audit = result["fit_audits"]["proposed"]["intervention_refit_delta_audit"]
    assert set(refit_audit["centering_group_animals"]) == expected_donors
    assert refit_audit["centering_group_count"] == 5
    assert refit_audit["refit_centering_covers_every_batch_donor"]
    assert refit_audit["identification_constraint"] == "exact_zero_mean_projection"
    assert refit_audit["projection_calls"] == refit_audit["optimizer_steps"] + 1
    assert refit_audit["final_donor_mean_delta_l2_norm"] <= 1e-7
    with np.load(output / "predictions.npz", allow_pickle=False) as predictions:
        treated = [
            predictions[name]
            for name in predictions.files
            if name.startswith("proposed__") and name.endswith("__neural_treated")
        ]
    assert len(treated) == 2
    assert {array.shape[-1] for array in treated} == {2, 3}
    score = icms.score_fold(
        fold_directory=output,
        acknowledge_target_outcomes=True,
        run_mode="synthetic",
    )
    persisted_score = json.loads((output / "metrics.json").read_text())
    serialized_sessions = json.dumps(persisted_score["session_scores"], sort_keys=True)
    assert "uncalibrated_marginal_neural_90_interval" in serialized_sessions
    assert "simultaneous_coverage" not in serialized_sessions
    assert "conformal" not in serialized_sessions
    assert score["uncertainty_audit"]["split_conformal"] == "ABSENT_NOT_FIT"
    assert not score["uncertainty_audit"]["simultaneous_coverage_exported"]
