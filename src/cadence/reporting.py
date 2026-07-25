"""Frozen, outcome-agnostic aggregation for CADENCE experiments.

The experiment runners deliberately write animal-level metrics rather than a
headline claim.  This module is the only layer that turns those metrics into
cross-animal summaries.  Its important invariants are:

* one biological target animal is one replication unit; teacher targets are
  nested and averaged equally within each independently generated world;
* the public-seed teacher partition is a deterministic procedural evaluation:
  its world-level intervals are descriptive and its headline conjunction is
  always ``NOT_EVALUATED``;
* all confidence intervals resample replication units with equal weight;
* baseline gains use the per-unit maximum over every emitted eligible non-oracle
  method, which is more conservative than selecting a single favorable method;
* sign-flip tests are exhaustive, including for the 28-mouse Allen cohort;
* every frozen headline criterion is tri-state.  Missing uncertainty,
  coverage, or falsification evidence is ``NOT_EVALUATED``, never a pass.

Future runners may expose optional gate evidence, but a supplementary artifact
cannot support a gate until its canonical relative filename and observed digest
are bound into the runner's authenticated completion chain.  The current
producer schemas do not export those completion-bound artifacts for Gates 5--8,
so those positive-evidence paths are deliberately disabled.  A hexadecimal
string inside ``metrics.json`` is never treated as proof that a retained draw,
randomization, band, or equivalence artifact exists.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import beta as beta_distribution

Status = Literal["PASS", "FAIL", "NOT_EVALUATED"]
FloatArray = NDArray[np.float64]

PASS: Status = "PASS"
FAIL: Status = "FAIL"
NOT_EVALUATED: Status = "NOT_EVALUATED"

SCHEMA_VERSION = "cadence.reporting.v1"
DEFAULT_BOOTSTRAP_REPEATS = 20_000
DEFAULT_SEED = 20_260_725
DEFAULT_CONFIDENCE = 0.95
MINIMUM_BASELINE_GAIN = 0.10
SIMULTANEOUS_COVERAGE_LOWER_BOUND = 0.80
DISJOINT_CALIBRATION_SCOPE = "whole_animal_disjoint_from_fit_early_stopping_and_model_selection"
ALPHA = 0.05
PROPOSED_METHOD = "proposed"
SUPPLEMENTARY_GATE_ARTIFACT_REASON = (
    "current canonical producer schemas do not bind this gate's supplementary "
    "artifact filename and observed digest into an authenticated completion chain"
)

ALLEN_LOCKED_ANIMALS = frozenset(
    {
        "450471",
        "425496",
        "547266",
        "484627",
        "456915",
        "491060",
        "476067",
        "479426",
        "457766",
        "548950",
        "456564",
        "447663",
        "547486",
        "459773",
        "477052",
        "442709",
        "456916",
        "459777",
        "453911",
        "453913",
        "513630",
        "449441",
        "512458",
        "472271",
        "533162",
        "533161",
        "461946",
        "431023",
    }
)
ICMS_RANDOMIZED_ANIMALS = frozenset({"ICMS92", "ICMS93", "ICMS98", "ICMS100", "ICMS101"})
ICMS_ABSOLUTE_ONLY_ANIMAL = "ICMS83"
ICMS_TASK_MICE = ("ICMS83", "ICMS92", "ICMS93", "ICMS98", "ICMS100", "ICMS101")
ICMS_DANDISET_VERSION = "0.260715.2016"
ICMS_INDEX_TOTALS = {
    "sessions": 45,
    "normal_calibration_trials": 2332,
    "catch_normal_calibration_trials": 1400,
    "iti_calibration_windows": 932,
    "stimulation_trials": 16640,
}
TEACHER_LOCKED_WORLDS = 20
TEACHER_TARGETS_PER_WORLD = 4
TEACHER_LOCKED_TARGET_UNITS = TEACHER_LOCKED_WORLDS * TEACHER_TARGETS_PER_WORLD
PREOUTCOME_TAG = "pre-outcome-v1.0.0"
LOCK_SEED = 20_260_725

ALLEN_EXPECTED_METHODS = frozenset(
    {
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
    }
)
ICMS_EXPECTED_METHODS = frozenset(
    {
        "proposed",
        "linear",
        "additive",
        "black_box",
        "zero_effect",
        "condition_time",
        "nearest_donor",
    }
)
ICMS_REPORT_METHOD_ORDER = (
    "proposed",
    "linear",
    "additive",
    "black_box",
    "zero_effect",
    "condition_time",
    "nearest_donor",
)
TEACHER_EXPECTED_METHODS = frozenset(
    {
        "proposed",
        "linear",
        "additive",
        "black_box",
        "zero_effect",
        "proposed_native_decoder",
        "proposed_no_target_residual",
        "proposed_no_target_adaptation",
    }
)
TEACHER_LEARNED_METHODS = (
    "proposed",
    "linear",
    "additive",
    "black_box",
)
TEACHER_LOCKED_SEEDS = (
    731_942_001,
    731_942_019,
    731_942_043,
    731_942_087,
    731_942_123,
    731_942_161,
    731_942_207,
    731_942_249,
    731_942_291,
    731_942_339,
    731_942_381,
    731_942_423,
    731_942_477,
    731_942_519,
    731_942_563,
    731_942_611,
    731_942_653,
    731_942_699,
    731_942_741,
    731_942_789,
)
TEACHER_LOCKED_WORLD_IDS = (
    "cadence-teacher-rnn-v1-locked-00-543128d5cd53",
    "cadence-teacher-rnn-v1-locked-01-a46edafc48fe",
    "cadence-teacher-rnn-v1-locked-02-1ff0763ef221",
    "cadence-teacher-rnn-v1-locked-03-c70547ac6e0f",
    "cadence-teacher-rnn-v1-locked-04-8047cf1b0d62",
    "cadence-teacher-rnn-v1-locked-05-7b3a60b4df4d",
    "cadence-teacher-rnn-v1-locked-06-9a70f96c1111",
    "cadence-teacher-rnn-v1-locked-07-6c0b10a30373",
    "cadence-teacher-rnn-v1-locked-08-999d8cc38919",
    "cadence-teacher-rnn-v1-locked-09-988784506563",
    "cadence-teacher-rnn-v1-locked-10-05aa6ed9ae34",
    "cadence-teacher-rnn-v1-locked-11-a7ec97dcee0d",
    "cadence-teacher-rnn-v1-locked-12-abd5c296e104",
    "cadence-teacher-rnn-v1-locked-13-d597d17c5ecb",
    "cadence-teacher-rnn-v1-locked-14-cfa645d3e9fe",
    "cadence-teacher-rnn-v1-locked-15-fa5bb07e1bee",
    "cadence-teacher-rnn-v1-locked-16-565b7564fb94",
    "cadence-teacher-rnn-v1-locked-17-26882ac1d117",
    "cadence-teacher-rnn-v1-locked-18-ce2d5a4ff652",
    "cadence-teacher-rnn-v1-locked-19-ae4dcd591047",
)
_HEX40 = re.compile(r"[0-9a-f]{40}")

_ORACLE_TOKENS = (
    "oracle",
    "supervised",
    "upper_bound",
    "upper-bound",
    "ceiling",
    "ground_truth",
    "ground-truth",
    "true_h",
)


@dataclass(frozen=True, slots=True)
class AnimalResult:
    """One method evaluated on one target row before hierarchical aggregation."""

    dataset: str
    cohort: str
    unit_id: str
    animal_id: str
    method: str
    metrics: Mapping[str, Any]
    source_file: str = ""
    run_id: str = ""
    fold: int | None = None
    world_id: str | None = None
    randomized_estimand: bool = True


@dataclass(frozen=True, slots=True)
class AdaptedBatch:
    """Canonical records and optional, explicitly declared gate evidence."""

    records: tuple[AnimalResult, ...]
    headline_evidence: Mapping[str, Any] = field(default_factory=dict)
    artifact_validation: Mapping[str, Any] = field(default_factory=dict)


def _scalar(value: Any) -> float | bool | None:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | float | np.integer | np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _source_label(source_file: str | Path | None) -> str:
    return "" if source_file is None else str(source_file)


def _require_canonical_source_path(path: Path, relative_path: str | Path) -> None:
    """Reject alternate locked-output trees that permit realization cherry-picking."""

    repository = Path(__file__).resolve().parents[2]
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("canonical locked-output path is unsafe")
    expected = (repository / relative).absolute()
    observed = path.absolute()
    if observed != expected:
        raise ValueError(
            f"locked artifact must use canonical one-shot path {expected}; observed {observed}"
        )
    cursor = repository
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("canonical locked-output path may not traverse a symlink")


def _require_method_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a nonempty method mapping")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_authenticated_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"missing SHA-256 sidecar for {path.name}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != path.name or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
        raise ValueError(f"malformed SHA-256 sidecar for {path.name}")
    observed = _sha256_file(path)
    if observed != fields[0]:
        raise ValueError(f"SHA-256 mismatch for {path.name}")
    return observed


def _freeze_attestation(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("pre-outcome freeze attestation is missing")
    attestation = {
        "commit": str(value.get("commit", "")),
        "tag": str(value.get("tag", "")),
        "tag_object": str(value.get("tag_object", "")),
    }
    if (
        _HEX40.fullmatch(attestation["commit"]) is None
        or attestation["tag"] != PREOUTCOME_TAG
        or _HEX40.fullmatch(attestation["tag_object"]) is None
        or attestation["commit"] == attestation["tag_object"]
    ):
        raise ValueError("pre-outcome freeze attestation is malformed")
    return attestation


def _git_verify_annotated_attestation(attestation: Mapping[str, str]) -> None:
    """Verify the embedded annotated-tag chain against the published repository."""

    repository = Path(__file__).resolve().parents[2]

    def git(*arguments: str) -> str:
        try:
            return subprocess.run(
                ["git", *arguments],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise ValueError(
                "cannot verify the embedded pre-outcome tag in this repository"
            ) from error

    if git("cat-file", "-t", attestation["tag_object"]) != "tag":
        raise ValueError("embedded tag object is not an annotated tag")
    if git("rev-parse", PREOUTCOME_TAG) != attestation["tag_object"]:
        raise ValueError("embedded tag object is not the published pre-outcome tag")
    if git("rev-parse", f"{attestation['tag_object']}^{{commit}}") != attestation["commit"]:
        raise ValueError("embedded annotated tag does not peel to the frozen commit")


def _git_verify_clean_reporter_state(attestation: Mapping[str, str]) -> dict[str, Any]:
    """Bind a headline decision to clean reporter code at the frozen tag."""

    _git_verify_annotated_attestation(attestation)
    repository = Path(__file__).resolve().parents[2]

    def git(*arguments: str) -> str:
        try:
            return subprocess.run(
                ["git", *arguments],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise ValueError("cannot authenticate the running reporter checkout") from error

    if git("rev-parse", "HEAD") != attestation["commit"]:
        raise ValueError("headline aggregation must run at the frozen tagged commit")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("headline aggregation requires a clean frozen checkout")
    code_sha256: dict[str, str] = {}
    for relative_path in (
        "src/cadence/reporting.py",
        "scripts/aggregate_results.py",
    ):
        current = repository / relative_path
        current_bytes = current.read_bytes()
        frozen_bytes = _git_blob_at_commit(relative_path, attestation["commit"])
        if current_bytes != frozen_bytes:
            raise ValueError(f"running reporter code differs from frozen {relative_path}")
        code_sha256[relative_path] = hashlib.sha256(current_bytes).hexdigest()
    return {
        "schema": "cadence.reporter_attestation.v1",
        **dict(attestation),
        "head": attestation["commit"],
        "clean_worktree": True,
        "code_sha256": code_sha256,
    }


def _invalid_artifact_validation(reason: str) -> dict[str, Any]:
    return {"valid": False, "reason": reason}


def _valid_artifact_validation(
    *,
    source_file: Path,
    attestation: Mapping[str, str],
    checks: Sequence[str],
) -> dict[str, Any]:
    return {
        "valid": True,
        "source_file": str(source_file.resolve()),
        "source_sha256": _sha256_file(source_file),
        "freeze_attestation": dict(attestation),
        "checks": list(checks),
    }


def _safe_relative_artifact(directory: Path, value: Any) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact manifest contains an unsafe path")
    path = (directory / relative).resolve()
    try:
        path.relative_to(directory.resolve())
    except ValueError as error:
        raise ValueError("artifact escapes its result directory") from error
    return path


def _authenticate_completion_manifest(
    directory: Path,
    stage: str,
) -> Mapping[str, Any]:
    path = directory / f"{stage}.complete.json"
    _read_authenticated_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("stage") != stage
        or not isinstance(payload.get("artifacts"), Mapping)
        or not payload["artifacts"]
    ):
        raise ValueError(f"invalid {stage} completion manifest")
    for name, expected in payload["artifacts"].items():
        artifact = _safe_relative_artifact(directory, name)
        if not artifact.is_file() or _sha256_file(artifact) != expected:
            raise ValueError(f"{stage} completion artifact failed authentication: {name}")
    return payload


def _authenticate_allen_seal_transaction(
    directory: Path,
    preparation: Mapping[str, Any],
    canonical_relative_output: str,
) -> str:
    """Reconstruct the retired Allen journal and authenticate its exact bytes."""

    record_value = preparation.get("target_seal_transaction")
    if not isinstance(record_value, Mapping):
        raise ValueError("Allen preparation omits its target-seal transaction")
    record = dict(record_value)
    digest = record.pop("sha256", None)
    targets = [str(value) for value in preparation.get("targets", ())]
    seals = _mapping_or_empty(preparation.get("target_seals"))
    expected_entries: list[dict[str, Any]] = []
    ordered_names = (
        "legacy_combined",
        "role_sealed",
        "experiment_sealed",
    )
    for mouse in targets:
        records = _mapping_or_empty(seals.get(mouse))
        if set(records) != set(ordered_names):
            raise ValueError(f"Allen target seal is incomplete for mouse {mouse}")
        for name in ordered_names:
            value = records.get(name)
            if not isinstance(value, Mapping):
                raise ValueError("Allen target-seal journal entry is malformed")
            if (
                set(value)
                != {
                    "path",
                    "original_mode",
                    "sealed_mode",
                    "device_id",
                    "inode",
                    "sha256",
                }
                or not isinstance(value.get("device_id"), int)
                or isinstance(value.get("device_id"), bool)
                or not isinstance(value.get("inode"), int)
                or isinstance(value.get("inode"), bool)
                or not _is_sha256(value.get("sha256"))
            ):
                raise ValueError("Allen target-seal journal entry is malformed")
            expected_entries.append({"mouse": mouse, "name": name, **dict(value)})

    encoded = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    processed_root = Path(str(record.get("processed_root", "")))
    expected_paths = {
        (mouse, name): (
            processed_root / f"mouse_{mouse}" / "windows.npz"
            if name == "legacy_combined"
            else (
                processed_root / f"mouse_{mouse}" / "sealed_omission_outcomes.npz"
                if name == "role_sealed"
                else directory / "queries" / f"mouse_{mouse}" / "sealed_outcomes.npz"
            )
        )
        for mouse in targets
        for name in ordered_names
    }
    if (
        not _is_sha256(digest)
        or hashlib.sha256(encoded).hexdigest() != digest
        or set(record)
        != {
            "schema",
            "fold",
            "canonical_relative_output",
            "output_path",
            "processed_root",
            "targets",
            "entries",
            "active",
            "restore_after_score_commit",
            "prepare_guard_sha256",
        }
        or record.get("schema") != "cadence-allen-target-seal-transaction-v1"
        or record.get("fold") != preparation.get("fold")
        or record.get("canonical_relative_output") != canonical_relative_output
        or Path(str(record.get("output_path", ""))) != directory.resolve()
        or not processed_root.is_absolute()
        or record.get("targets") != targets
        or record.get("entries") != expected_entries
        or record.get("active") is not True
        or record.get("restore_after_score_commit") is not True
        or not _is_sha256(record.get("prepare_guard_sha256"))
    ):
        raise ValueError("Allen target-seal transaction is invalid")
    for entry in expected_entries:
        key = (str(entry["mouse"]), str(entry["name"]))
        if Path(str(entry.get("path", ""))) != expected_paths[key].resolve():
            raise ValueError("Allen target-seal transaction path binding is invalid")
    return str(digest)


def _authenticate_icms_seal_transaction(
    directory: Path,
    prepare: Mapping[str, Any],
    canonical_relative_output: str,
) -> str:
    """Bind the ICMS transaction digest to the immutable registry byte image."""

    seal_row = _mapping_or_empty(prepare.get("physical_target_seal"))
    seal_path = _safe_relative_artifact(directory, seal_row.get("path"))
    seal_sha = _read_authenticated_sidecar(seal_path)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not isinstance(seal, Mapping):
        raise ValueError("ICMS target-seal transaction is invalid")
    expected_seal = {key: value for key, value in seal_row.items() if key not in {"path", "sha256"}}
    transaction_sha = prepare.get("target_seal_transaction_sha256")
    canonical_seal = (json.dumps(dict(seal), indent=2, sort_keys=True) + "\n").encode()
    if (
        seal_row.get("path") != "target_seal.json"
        or seal_path != (directory / "target_seal.json").resolve()
        or dict(seal) != expected_seal
        or not _is_sha256(transaction_sha)
        or seal_row.get("sha256") != seal_sha
        or transaction_sha != seal_sha
        or hashlib.sha256(canonical_seal).hexdigest() != seal_sha
        or set(seal)
        != {
            "schema",
            "target_animal",
            "target_path",
            "processed_root",
            "fold_directory",
            "canonical_relative_output",
            "expected_sha256",
            "device_id",
            "inode",
            "original_mode",
            "sealed_mode",
            "active",
        }
        or seal.get("schema") != "cadence-icms-physical-target-seal-v1"
        or seal.get("target_animal") != prepare.get("target_animal")
        or seal.get("canonical_relative_output") != canonical_relative_output
        or Path(str(seal.get("fold_directory", ""))) != directory.resolve()
        or not Path(str(seal.get("processed_root", ""))).is_absolute()
        or not Path(str(seal.get("target_path", ""))).is_absolute()
        or not _is_sha256(seal.get("expected_sha256"))
        or seal.get("sealed_mode") != 0
        or not isinstance(seal.get("original_mode"), int)
        or int(seal.get("original_mode", 0)) & 0o444 == 0
        or seal.get("active") is not True
    ):
        raise ValueError("ICMS target-seal transaction is invalid")
    return str(transaction_sha)


def _authenticate_allen_transaction_references(
    transaction_sha256: str,
    *,
    payload: Mapping[str, Any],
    prediction: Mapping[str, Any],
    prediction_metadata: Mapping[str, Any],
    completions: Mapping[str, Mapping[str, Any]],
    restoration_completion: Mapping[str, Any],
) -> None:
    """Require every Allen stage and restoration record to name one journal."""

    restoration_plan = _mapping_or_empty(
        _mapping_or_empty(payload.get("protocol_audit")).get("target_outcome_mode_restoration")
    )
    required_artifacts = {
        "prepare": {"preparation.json"},
        "predict": {
            "predictions.npz",
            "predictions.npz.sha256",
            "prediction_run.json",
        },
        "score": {"metrics.json"},
    }
    if (
        payload.get("target_seal_transaction_sha256") != transaction_sha256
        or prediction.get("target_seal_transaction_sha256") != transaction_sha256
        or prediction_metadata.get("target_seal_transaction_sha256") != transaction_sha256
        or restoration_plan.get("seal_transaction_sha256") != transaction_sha256
        or restoration_completion.get("seal_transaction_sha256") != transaction_sha256
        or set(completions) != {"prepare", "predict", "score"}
        or any(
            not names.issubset(set(_mapping_or_empty(completions[stage].get("artifacts"))))
            for stage, names in required_artifacts.items()
        )
        or any(
            _mapping_or_empty(completion.get("metadata")).get("target_seal_transaction_sha256")
            != transaction_sha256
            for completion in completions.values()
        )
    ):
        raise ValueError("Allen target-seal transaction digest chain is inconsistent")


def _authenticate_icms_transaction_references(
    transaction_sha256: str,
    *,
    prepare: Mapping[str, Any],
    prediction: Mapping[str, Any],
    payload: Mapping[str, Any],
    completions: Mapping[str, Mapping[str, Any]],
) -> None:
    """Require every ICMS stage and completion record to name one seal image."""

    if (
        any(
            stage.get("target_seal_transaction_sha256") != transaction_sha256
            for stage in (prepare, prediction, payload)
        )
        or set(completions) != {"prepare", "predict", "score"}
        or any(
            completion.get("seal_transaction_sha256") != transaction_sha256
            for completion in completions.values()
        )
    ):
        raise ValueError("ICMS target-seal transaction digest chain is inconsistent")


def _authenticate_icms_completion(
    directory: Path,
    stage: str,
    expected_artifact: str,
    attestation: Mapping[str, str],
    canonical_relative_output: str,
    seal_transaction_sha256: str,
) -> Mapping[str, Any]:
    """Authenticate the ICMS runner's singular-artifact completion record."""

    path = directory / f"{stage}_complete.json"
    _read_authenticated_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != f"cadence-icms-{stage}-complete-v1"
        or payload.get("stage") != stage
        or payload.get("artifact") != expected_artifact
        or payload.get("append_only") is not True
        or _freeze_attestation(payload.get("freeze_attestation")) != attestation
        or payload.get("canonical_relative_output") != canonical_relative_output
        or payload.get("seal_transaction_sha256") != seal_transaction_sha256
    ):
        raise ValueError(f"invalid ICMS {stage} completion manifest")
    artifact = directory / expected_artifact
    observed = _read_authenticated_sidecar(artifact)
    if payload.get("artifact_sha256") != observed:
        raise ValueError(f"ICMS {stage} completion does not bind {expected_artifact}")
    return payload


def _authenticate_allen_restoration_completion(
    directory: Path,
    *,
    preparation: Mapping[str, Any],
    score_completion: Mapping[str, Any],
    canonical_relative_output: str,
) -> Mapping[str, Any]:
    """Authenticate the durable restoration commit that follows Allen scoring."""

    path = directory / "restore.complete.json"
    _read_authenticated_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    score_completion_sha = _read_authenticated_sidecar(directory / "score.complete.json")
    targets = tuple(map(str, preparation.get("targets", ())))
    seals = _mapping_or_empty(preparation.get("target_seals"))
    transaction_sha = _authenticate_allen_seal_transaction(
        directory,
        preparation,
        canonical_relative_output,
    )
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "cadence-allen-target-restore-completion-v1"
        or payload.get("restored_after_score_commit") is not True
        or payload.get("eligible_for_later_donor_reuse") is not True
        or payload.get("canonical_relative_output") != canonical_relative_output
        or payload.get("score_completion_sha256") != score_completion_sha
        or payload.get("seal_transaction_sha256") != transaction_sha
        or set(_mapping_or_empty(payload.get("mice"))) != set(targets)
        or _mapping_or_empty(score_completion.get("metadata")).get(
            "canonical_processed_target_modes_restored"
        )
        is not False
        or _mapping_or_empty(score_completion.get("metadata")).get(
            "target_mode_restoration_pending"
        )
        is not True
        or _mapping_or_empty(score_completion.get("metadata")).get("target_seal_transaction_sha256")
        != transaction_sha
    ):
        raise ValueError("Allen restoration completion is invalid")
    restored_mice = _mapping_or_empty(payload.get("mice"))
    for mouse in targets:
        expected = _mapping_or_empty(seals.get(mouse))
        observed = _mapping_or_empty(restored_mice.get(mouse))
        if set(expected) != {
            "legacy_combined",
            "role_sealed",
            "experiment_sealed",
        } or set(observed) != set(expected):
            raise ValueError(f"Allen restoration scope is incomplete for mouse {mouse}")
        for name, seal_value in expected.items():
            seal = _mapping_or_empty(seal_value)
            row = _mapping_or_empty(observed.get(name))
            if (
                not _is_sha256(seal.get("sha256"))
                or row.get("path") != seal.get("path")
                or row.get("restored_mode") != seal.get("original_mode")
                or row.get("sha256") != seal.get("sha256")
            ):
                raise ValueError("Allen restoration artifact binding is invalid")
    return payload


def _authenticate_icms_restoration_completion(
    directory: Path,
    *,
    prepare: Mapping[str, Any],
    restore: Mapping[str, Any],
    restore_sha256: str,
    score_completion: Mapping[str, Any],
    canonical_relative_output: str,
) -> Mapping[str, Any]:
    """Authenticate the durable restoration commit that follows ICMS scoring."""

    path = directory / "target_restore_complete.json"
    _read_authenticated_sidecar(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    seal = _mapping_or_empty(prepare.get("physical_target_seal"))
    score_completion_sha = _read_authenticated_sidecar(directory / "score_complete.json")
    transaction_sha = prepare.get("target_seal_transaction_sha256")
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "cadence-icms-target-restore-completion-v1"
        or payload.get("restored_after_score_commit") is not True
        or payload.get("canonical_relative_output") != canonical_relative_output
        or payload.get("target_animal") != seal.get("target_animal")
        or payload.get("target_path") != seal.get("target_path")
        or payload.get("restored_mode") != seal.get("original_mode")
        or payload.get("target_sha256") != seal.get("expected_sha256")
        or payload.get("immutable_seal_sha256") != seal.get("sha256")
        or not _is_sha256(transaction_sha)
        or transaction_sha != seal.get("sha256")
        or restore.get("seal_transaction_sha256") != transaction_sha
        or payload.get("seal_transaction_sha256") != transaction_sha
        or score_completion.get("seal_transaction_sha256") != transaction_sha
        or payload.get("restore_audit_sha256") != restore_sha256
        or payload.get("score_completion_artifact") != score_completion.get("artifact")
        or payload.get("score_completion_artifact") != "metrics.json"
        or payload.get("score_completion_sha256") != score_completion_sha
        or payload.get("registry_retained_until_score_commit") is not True
        or payload.get("registry_removed_after_finalization") is not True
        or restore.get("restoration_status") != "PENDING_SCORE_COMMIT_FINALIZATION"
        or restore.get("registry_retained_until_score_commit") is not True
    ):
        raise ValueError("ICMS restoration completion is invalid")
    return payload


def _git_blob_at_commit(relative_path: str, commit: str) -> bytes:
    repository = Path(__file__).resolve().parents[2]
    if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
        raise ValueError("tracked provenance path is unsafe")
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise ValueError(f"cannot authenticate tracked provenance file {relative_path}") from error


def _validate_icms_tracked_identity(
    identity: Any,
    *,
    relative_path: str,
    commit: str,
) -> tuple[str, bytes]:
    row = _mapping_or_empty(identity)
    blob = _git_blob_at_commit(relative_path, commit)
    digest = hashlib.sha256(blob).hexdigest()
    if (
        row.get("relative_path") != relative_path
        or row.get("sha256") != digest
        or row.get("git_blob_sha256") != digest
    ):
        raise ValueError(f"ICMS tracked provenance identity failed for {relative_path}")
    return digest, blob


def _validate_icms_full_config(value: Any) -> Mapping[str, Any]:
    config = _mapping_or_empty(value)
    expected_scalars = {
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
        "seed": LOCK_SEED,
    }
    if any(config.get(key) != expected for key, expected in expected_scalars.items()):
        raise ValueError("ICMS full locked configuration differs from the canonical scope")
    if config.get("current_grid_uA") != [float(value) for value in range(1, 14)]:
        raise ValueError("ICMS current lattice differs from the canonical 1--13 uA grid")
    expected_fits = {
        "normal_fit": (20260736, 400, 35),
        "intervention_fit": (20260748, 400, 35),
        "target_fit": (20260762, 300, 30),
    }
    devices: set[str] = set()
    for name, (seed, epochs, patience) in expected_fits.items():
        fit = _mapping_or_empty(config.get(name))
        device = str(fit.get("device", ""))
        devices.add(device)
        expected = {
            "learning_rate": 0.001,
            "max_epochs": epochs,
            "patience": patience,
            "weight_decay": 0.0001,
            "gradient_clip": 5.0,
            "seed": seed,
            "device": device,
            "mixed_precision": device.startswith("cuda"),
        }
        if not device or fit != expected:
            raise ValueError(f"ICMS {name} differs from the canonical full fit")
    if len(devices) != 1:
        raise ValueError("ICMS fit stages were not run on one consistent device")
    return config


def _as_nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _as_small_projection_norm(value: Any, name: str) -> float:
    scalar = _scalar(value)
    if isinstance(scalar, bool) or scalar is None or abs(float(scalar)) > 1e-7:
        raise ValueError(f"{name} violates the exact donor-delta projection tolerance")
    return float(scalar)


def _canonical_teacher_configuration() -> tuple[
    str, str, dict[str, Any], int, int, tuple[float, ...]
]:
    """Return independently reconstructed frozen teacher configuration facts."""

    # Local imports avoid making the general reporting module depend on the
    # comparatively heavy experiment stack at import time.
    from cadence.experiments.teacher import (
        LEARNED_METHODS,
        make_experiment_config,
        make_profile_teacher_config,
        teacher_experiment_scientific_sha256,
    )
    from cadence.teacher import load_teacher_config, teacher_config_sha256

    source_root = Path(__file__).resolve().parents[2]
    teacher = make_profile_teacher_config(
        load_teacher_config(source_root / "configs" / "teacher.yaml"),
        "full",
    )
    experiment = make_experiment_config(
        "full",
        seed=0,
        device="cpu",
        learned_methods=LEARNED_METHODS,
    )
    scientific_mapping = experiment.to_mapping()
    for stage in ("normal_fit", "intervention_fit", "target_fit"):
        stage_mapping = dict(_mapping_or_empty(scientific_mapping.get(stage)))
        stage_mapping.pop("device", None)
        stage_mapping.pop("mixed_precision", None)
        scientific_mapping[stage] = stage_mapping
    return (
        teacher_config_sha256(teacher),
        teacher_experiment_scientific_sha256(experiment),
        scientific_mapping,
        teacher.intervention.onset_step - 1,
        teacher.dynamics.latent_dim,
        experiment.readout_ridge_grid,
    )


def _teacher_scientific_experiment_mapping(value: Any) -> dict[str, Any]:
    """Strip only the two documented machine-specific execution fields."""

    if not isinstance(value, Mapping):
        raise ValueError("teacher experiment configuration mapping is missing")
    mapping = {str(key): item for key, item in value.items()}
    for stage in ("normal_fit", "intervention_fit", "target_fit"):
        stage_value = mapping.get(stage)
        if not isinstance(stage_value, Mapping):
            raise ValueError(f"teacher {stage} configuration is missing")
        stage_mapping = {str(key): item for key, item in stage_value.items()}
        if "device" not in stage_mapping or "mixed_precision" not in stage_mapping:
            raise ValueError(f"teacher {stage} execution disclosure is incomplete")
        stage_mapping.pop("device")
        stage_mapping.pop("mixed_precision")
        mapping[stage] = stage_mapping
    return mapping


def _validate_icms_fit_audits(
    prediction: Mapping[str, Any],
    *,
    target: str,
    donors: Sequence[str],
) -> None:
    audits = _mapping_or_empty(prediction.get("fit_audits"))
    learned_methods = ICMS_REPORT_METHOD_ORDER[:4]
    if set(audits) != set(learned_methods):
        raise ValueError("ICMS learned-method fit audits are incomplete")
    ordered_animals = sorted(ICMS_TASK_MICE)
    validation_animal = ordered_animals[(ordered_animals.index(target) + 1) % len(ordered_animals)]
    selection_animals = sorted(set(donors) - {validation_animal})
    sorted_donors = sorted(donors)
    for method in learned_methods:
        audit = _mapping_or_empty(audits.get(method))
        normal_selection = _mapping_or_empty(audit.get("normal_selection"))
        intervention_selection = _mapping_or_empty(audit.get("intervention_selection"))
        normal_best = _as_nonnegative_int(
            normal_selection.get("best_epoch"),
            f"{method} normal best epoch",
        )
        intervention_best = _as_nonnegative_int(
            intervention_selection.get("best_epoch"),
            f"{method} intervention best epoch",
        )
        if (
            normal_selection.get("stage") != "normal"
            or intervention_selection.get("stage") != "intervention"
            or audit.get("normal_selection_training_animals") != selection_animals
            or audit.get("intervention_inner_validation_animal") != validation_animal
            or audit.get("validation_normal_gradient_to_shared_f") is not False
        ):
            raise ValueError(f"{method} nested selection topology is invalid")
        shared_before = audit.get("shared_f_before_validation_normal_sha256")
        shared_after = audit.get("shared_f_after_validation_normal_sha256")
        if (
            not _is_sha256(shared_before)
            or shared_before != shared_after
            or audit.get("shared_normal_stage_state_excluding_validation_adapters_sha256")
            != shared_after
        ):
            raise ValueError(f"{method} validation-normal adaptation changed shared F")
        validation_adaptation = _mapping_or_empty(audit.get("validation_normal_adaptation"))
        if not validation_adaptation or any(
            _mapping_or_empty(result).get("stage") != "target_adaptation"
            for result in validation_adaptation.values()
        ):
            raise ValueError(f"{method} validation-normal adapter audit is incomplete")

        selection_delta = _mapping_or_empty(audit.get("intervention_selection_delta_audit"))
        selection_steps = _as_nonnegative_int(
            selection_delta.get("optimizer_steps"),
            f"{method} selection optimizer steps",
        )
        if (
            selection_steps < 1
            or selection_delta.get("validation_animal") != validation_animal
            or selection_delta.get("validation_delta_requires_grad") is not False
            or selection_delta.get("validation_delta_in_shrinkage") is not False
            or selection_delta.get("validation_delta_centering_applied") is not False
            or selection_delta.get("validation_delta_frozen_zero_during_selection") is not True
            or selection_delta.get("identification_constraint") != "exact_zero_mean_projection"
            or selection_delta.get("centering_group_animals") != selection_animals
            or selection_delta.get("centering_excluded_animals") != [validation_animal]
            or selection_delta.get("projection_calls") != selection_steps + 1
        ):
            raise ValueError(f"{method} intervention selection delta audit is invalid")
        for key in (
            "validation_delta_l2_norm",
            "maximum_validation_delta_shrinkage_term",
            "prefit_projection_residual_norm",
            "maximum_post_step_projection_residual_norm",
        ):
            _as_small_projection_norm(selection_delta.get(key), f"{method} {key}")

        final_normal = _mapping_or_empty(audit.get("final_normal_refit"))
        if (
            audit.get("final_model_is_fresh") is not True
            or audit.get("final_normal_refit_selected_epochs") != normal_best + 1
            or final_normal.get("epochs") != normal_best + 1
            or final_normal.get("normal_refit_animals") != sorted_donors
            or final_normal.get("normal_partitions") != ["fit", "val"]
            or final_normal.get("fresh_model") is not True
        ):
            raise ValueError(f"{method} fresh all-donor normal refit is invalid")
        refit = _mapping_or_empty(audit.get("intervention_refit_delta_audit"))
        refit_steps = _as_nonnegative_int(
            refit.get("optimizer_steps"),
            f"{method} refit optimizer steps",
        )
        if (
            refit_steps < 1
            or audit.get("intervention_refit_all_donors_epochs") != intervention_best + 1
            or refit.get("identification_constraint") != "exact_zero_mean_projection"
            or refit.get("centering_group_animals") != sorted_donors
            or refit.get("centering_group_count") != len(sorted_donors)
            or refit.get("projection_calls") != refit_steps + 1
            or refit.get("refit_centering_covers_every_batch_donor") is not True
        ):
            raise ValueError(f"{method} all-donor intervention refit is invalid")
        for key in (
            "final_donor_mean_delta_l2_norm",
            "prefit_projection_residual_norm",
            "maximum_post_step_projection_residual_norm",
        ):
            _as_small_projection_norm(refit.get(key), f"{method} refit {key}")
        target_adaptation = _mapping_or_empty(audit.get("target_normal_only_adaptation"))
        before = audit.get("target_adaptation_nonadapter_state_before_sha256")
        after = audit.get("target_adaptation_nonadapter_state_after_sha256")
        if (
            not target_adaptation
            or any(
                _mapping_or_empty(result).get("stage") != "target_adaptation"
                for result in target_adaptation.values()
            )
            or not _is_sha256(before)
            or before != after
        ):
            raise ValueError(f"{method} target normal-only adaptation audit is invalid")


def _equal_optional_number(observed: Any, expected: float | None) -> bool:
    observed_scalar = _scalar(observed)
    if isinstance(observed_scalar, bool):
        return False
    if expected is None:
        return observed_scalar is None
    return observed_scalar is not None and math.isclose(
        float(observed_scalar),
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def _mean_session_metric(
    sessions: Sequence[Mapping[str, Any]],
    method: str,
    key: str,
    *,
    nested_key: str | None = None,
) -> float | None:
    values: list[float] = []
    for session in sessions:
        score = _mapping_or_empty(session.get(method))
        value: Any = score.get(key)
        if nested_key is not None:
            value = _mapping_or_empty(value).get(nested_key)
        scalar = _scalar(value)
        if scalar is not None and not isinstance(scalar, bool):
            values.append(float(scalar))
    return float(np.mean(values)) if values else None


def _validate_icms_score_aggregation(
    payload: Mapping[str, Any],
    *,
    primary_evaluable: bool,
) -> None:
    session_scores = _mapping_or_empty(payload.get("session_scores"))
    aggregates = _mapping_or_empty(payload.get("animal_aggregate"))
    if not session_scores or set(aggregates) != set(ICMS_REPORT_METHOD_ORDER):
        raise ValueError("ICMS session/animal score matrix is incomplete")
    sessions: list[Mapping[str, Any]] = []
    for value in session_scores.values():
        session = _mapping_or_empty(value)
        if set(session) != set(ICMS_REPORT_METHOD_ORDER):
            raise ValueError("ICMS session score method matrix is incomplete")
        sessions.append(session)
    direct_metrics = {
        "absolute_neural_nrmse_equal_session": "absolute_neural_nrmse",
        "absolute_behavior_nrmse_equal_session": "absolute_behavior_nrmse",
        "neural_causal_skill_equal_session": "neural_causal_skill",
        "behavior_causal_skill_equal_session": "behavior_causal_skill",
        "nonrandomized_iti_neural_skill_equal_session": ("nonrandomized_iti_neural_skill"),
        "nonprimary_session_fallback_neural_skill_equal_session": (
            "nonprimary_session_fallback_neural_skill"
        ),
    }
    optional_metrics = {
        "neural_energy_score": ("neural_population_energy_score", None),
        "behavior_energy_score": ("behavior_energy_score", None),
        "uncalibrated_marginal_neural_pointwise_coverage": (
            "uncalibrated_marginal_neural_90_interval",
            "pointwise_coverage",
        ),
        "uncalibrated_marginal_behavior_pointwise_coverage": (
            "uncalibrated_marginal_behavior_90_interval",
            "pointwise_coverage",
        ),
    }
    for method in ICMS_REPORT_METHOD_ORDER:
        aggregate = _mapping_or_empty(aggregates.get(method))
        eligible_count = 0
        for session in sessions:
            score = _mapping_or_empty(session.get(method))
            eligible = score.get("randomized_causal_eligible")
            if not isinstance(eligible, bool) or score.get("primary_randomized_status") != (
                "EVALUATED" if eligible else "NOT_EVALUATED"
            ):
                raise ValueError("ICMS per-session randomized status is inconsistent")
            eligible_count += int(eligible)
        if aggregate.get("eligible_sessions") != eligible_count or aggregate.get(
            "primary_fold_status"
        ) != ("EVALUATED" if primary_evaluable else "NOT_EVALUATED"):
            raise ValueError("ICMS animal aggregate eligibility is inconsistent")
        for aggregate_key, session_key in direct_metrics.items():
            expected = _mean_session_metric(sessions, method, session_key)
            if (
                aggregate_key
                in {
                    "neural_causal_skill_equal_session",
                    "behavior_causal_skill_equal_session",
                }
                and not primary_evaluable
            ):
                expected = None
            if not _equal_optional_number(aggregate.get(aggregate_key), expected):
                raise ValueError(
                    f"ICMS equal-session aggregate mismatch for {method}/{aggregate_key}"
                )
        for aggregate_key, (session_key, nested_key) in optional_metrics.items():
            expected = _mean_session_metric(
                sessions,
                method,
                session_key,
                nested_key=nested_key,
            )
            if expected is None:
                if aggregate_key in aggregate:
                    raise ValueError(
                        f"ICMS unexpected optional aggregate for {method}/{aggregate_key}"
                    )
            elif not _equal_optional_number(aggregate.get(aggregate_key), expected):
                raise ValueError(f"ICMS optional aggregate mismatch for {method}/{aggregate_key}")


def _validate_teacher_fit_audits(payload: Mapping[str, Any]) -> None:
    """Authenticate the nested teacher selection/refit topology emitted by the runner."""

    stage_fits = _mapping_or_empty(payload.get("stage_fits"))
    protocol = _mapping_or_empty(payload.get("protocol_audit"))
    topology_by_method = _mapping_or_empty(protocol.get("nested_selection_topology"))
    delta_by_method = _mapping_or_empty(protocol.get("donor_delta_identification"))
    if (
        set(stage_fits) != set(TEACHER_LEARNED_METHODS)
        or set(topology_by_method) != set(TEACHER_LEARNED_METHODS)
        or set(delta_by_method) != set(TEACHER_LEARNED_METHODS)
        or protocol.get("target_intervention_batches_used_for_optimization") != 0
        or protocol.get("target_adaptation_splits") != ["normal_fit", "normal_val"]
        or protocol.get("target_normal_audit_used_for_optimization") is not False
        or protocol.get("post_onset_outcomes_mounted_as_inputs") is not False
        or protocol.get("prediction_mode") != "paired_open_loop"
        or protocol.get("prediction_initialization_sample") != "onset_minus_1"
        or protocol.get("target_neural_readout")
        != (
            "softplus-Poisson quasi-likelihood fit on frozen open-loop "
            "normal_fit rollouts, selected on frozen open-loop normal_val rollouts"
        )
        or protocol.get("target_readout_contemporaneous_count_encoded_as_its_own_predictor")
        is not False
    ):
        raise ValueError("teacher nested-fit protocol audit is incomplete")
    (
        _,
        _,
        _,
        expected_readout_anchor,
        expected_latent_dim,
        expected_ridges,
    ) = _canonical_teacher_configuration()
    for method in TEACHER_LEARNED_METHODS:
        fit = _mapping_or_empty(stage_fits.get(method))
        selection = _mapping_or_empty(fit.get("selection"))
        normal_selection = _mapping_or_empty(selection.get("normal_train_donors_only"))
        intervention_selection = _mapping_or_empty(
            selection.get("intervention_train_donors_validate_on_validation_donors")
        )
        validation_adapters = _mapping_or_empty(selection.get("validation_donor_normal_adaptation"))
        topology = _mapping_or_empty(selection.get("topology_audit"))
        if topology != _mapping_or_empty(topology_by_method.get(method)):
            raise ValueError(f"teacher {method} topology audit copies are inconsistent")
        selected_normal_epochs = _as_nonnegative_int(
            topology.get("selected_normal_epochs"),
            f"teacher {method} selected normal epochs",
        )
        selected_intervention_epochs = _as_nonnegative_int(
            topology.get("selected_intervention_epochs"),
            f"teacher {method} selected intervention epochs",
        )
        if (
            selected_normal_epochs < 1
            or selected_intervention_epochs < 1
            or normal_selection.get("stage") != "normal"
            or normal_selection.get("best_epoch") != selected_normal_epochs - 1
            or intervention_selection.get("stage") != "intervention"
            or intervention_selection.get("best_epoch") != selected_intervention_epochs - 1
            or len(validation_adapters) != 2
            or any(
                _mapping_or_empty(result).get("stage") != "target_adaptation"
                for result in validation_adapters.values()
            )
            or topology.get("shared_normal_training_roles") != ["train_donor"]
            or topology.get("validation_donor_adapter_roles") != ["validation_donor"]
            or topology.get("validation_adapter_shared_parameter_max_abs_change") != 0.0
            or topology.get("validation_interventions_used_for_gradient_steps_before_selection")
            is not False
            or topology.get("validation_intervention_delta_present_before_selection") is not False
            or topology.get("selection_training_delta_group_count") != 10
            or topology.get("final_refit_roles") != ["train_donor", "validation_donor"]
            or topology.get("final_refit_epoch_selection_from_refit_data") is not False
        ):
            raise ValueError(f"teacher {method} nested whole-donor topology is invalid")

        final_normal = _mapping_or_empty(fit.get("normal"))
        final_intervention = _mapping_or_empty(fit.get("intervention"))
        if (
            final_normal.get("stage") != "normal"
            or final_normal.get("epochs_run") != selected_normal_epochs
            or final_intervention.get("stage") != "intervention"
            or final_intervention.get("epochs_run") != selected_intervention_epochs
        ):
            raise ValueError(f"teacher {method} fresh fixed-epoch refit is invalid")
        delta = _mapping_or_empty(delta_by_method.get(method))
        if (
            delta != _mapping_or_empty(final_intervention.get("donor_delta_identification"))
            or delta.get("constraint") != "exact_zero_mean_projection_after_every_optimizer_step"
            or delta.get("training_group_count") != 12
            or delta.get("tolerance") != 1e-7
        ):
            raise ValueError(f"teacher {method} donor-delta identification is invalid")
        _as_small_projection_norm(
            delta.get("final_mean_l2_norm"),
            f"teacher {method} final donor-delta mean",
        )
        targets = _mapping_or_empty(fit.get("targets"))
        if len(targets) != TEACHER_TARGETS_PER_WORLD or any(
            _mapping_or_empty(result).get("stage") != "target_adaptation"
            for result in targets.values()
        ):
            raise ValueError(f"teacher {method} target normal-only adaptation is incomplete")
        for target, result in targets.items():
            readout = _mapping_or_empty(_mapping_or_empty(result).get("neural_readout"))
            expected_keys = {
                "best_epoch",
                "validation_poisson_nll",
                "selected_ridge",
                "normal_rollout_design_rank",
                "normal_rollout_design_condition_number",
                "normal_rollout_anchor",
                "normal_rollout_support_max_abs_standardized",
                "query_max_abs_standardized",
                "query_coordinate_fraction_outside_normal_rollout_range",
            }
            if set(readout) != expected_keys:
                raise ValueError(f"teacher {method}/{target} neural readout audit is incomplete")
            best_epoch = _as_nonnegative_int(
                readout.get("best_epoch"),
                f"teacher {method}/{target} readout best epoch",
            )
            design_rank = _as_nonnegative_int(
                readout.get("normal_rollout_design_rank"),
                f"teacher {method}/{target} readout design rank",
            )
            finite_fields = (
                "validation_poisson_nll",
                "selected_ridge",
                "normal_rollout_design_condition_number",
                "normal_rollout_support_max_abs_standardized",
                "query_max_abs_standardized",
                "query_coordinate_fraction_outside_normal_rollout_range",
            )
            finite_values = {key: _scalar(readout.get(key)) for key in finite_fields}
            if (
                best_epoch < 0
                or readout.get("normal_rollout_anchor") != expected_readout_anchor
                or design_rank < 1
                or design_rank > expected_latent_dim
                or any(
                    value is None or isinstance(value, bool) or not math.isfinite(float(value))
                    for value in finite_values.values()
                )
                or float(finite_values["selected_ridge"]) not in expected_ridges
                or float(finite_values["normal_rollout_design_condition_number"]) <= 0.0
                or float(finite_values["normal_rollout_support_max_abs_standardized"]) < 0.0
                or float(finite_values["query_max_abs_standardized"]) < 0.0
                or not 0.0
                <= float(finite_values["query_coordinate_fraction_outside_normal_rollout_range"])
                <= 1.0
            ):
                raise ValueError(f"teacher {method}/{target} neural readout audit is invalid")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_allen_protocol_details(
    payload: Mapping[str, Any],
    preparation: Mapping[str, Any],
    prediction: Mapping[str, Any],
    completions: Mapping[str, Mapping[str, Any]],
    restoration_completion: Mapping[str, Any],
    attestation: Mapping[str, str],
) -> None:
    """Authenticate the scientific scope encoded inside one Allen fold."""

    donors = tuple(map(str, payload.get("donors", ())))
    targets = tuple(map(str, payload.get("targets", ())))
    if (
        not donors
        or not targets
        or set(donors) & set(targets)
        or set(donors) | set(targets) != set(ALLEN_LOCKED_ANIMALS)
        or payload.get("seed") != 0
        or payload.get("optimization_profile") != "full"
    ):
        raise ValueError("Allen fold donor/target/full-seed scope is not canonical")
    for artifact in (preparation, prediction):
        if (
            tuple(map(str, artifact.get("donors", ()))) != donors
            or tuple(map(str, artifact.get("targets", ()))) != targets
            or artifact.get("fold") != payload.get("fold")
            or artifact.get("run_profile") != "locked"
        ):
            raise ValueError("Allen prepare/predict/score fold identities differ")

    audit = _mapping_or_empty(payload.get("protocol_audit"))
    locked_scope = _mapping_or_empty(preparation.get("locked_scope_audit"))
    if (
        not locked_scope
        or prediction.get("locked_scope_audit") != locked_scope
        or audit.get("locked_scope_audit") != locked_scope
        or locked_scope.get("attested_commit") != attestation["commit"]
        or locked_scope.get("release") != "1.1.0"
        or locked_scope.get("cohort_mouse_count") != 32
        or not _is_sha256(locked_scope.get("cohort_identity_sha256"))
    ):
        raise ValueError("Allen canonical locked-scope audit chain is inconsistent")
    tracked = _mapping_or_empty(locked_scope.get("tracked_files"))
    if set(tracked) != {"manifest", "processed_index", "experiment_config"}:
        raise ValueError("Allen tracked protocol-file scope is incomplete")
    for record in tracked.values():
        row = _mapping_or_empty(record)
        if (
            row.get("commit") != attestation["commit"]
            or not _is_sha256(row.get("sha256"))
            or row.get("sha256") != row.get("git_blob_sha256")
        ):
            raise ValueError("Allen tracked protocol file differs from the frozen commit")
    frozen_configuration = _mapping_or_empty(locked_scope.get("configuration"))
    if (
        frozen_configuration.get("profile") != "full"
        or frozen_configuration.get("run_seed") != 0
        or frozen_configuration.get("learned_methods")
        != ["proposed", "linear", "additive", "black_box"]
        or frozen_configuration.get("ablations")
        != ["proposed_no_residual", "proposed_no_target_adaptation"]
        or frozen_configuration.get("intervention_rank") != 2
        or not _is_sha256(frozen_configuration.get("canonical_optimization_sha256"))
    ):
        raise ValueError("Allen locked optimization audit is not canonical")
    canonical_optimization_sha = frozen_configuration["canonical_optimization_sha256"]
    if any(
        value != canonical_optimization_sha
        for value in (
            preparation.get("canonical_optimization_sha256"),
            prediction.get("canonical_optimization_sha256"),
            audit.get("canonical_optimization_sha256"),
        )
    ):
        raise ValueError("Allen canonical optimization hash chain is inconsistent")

    processed = _mapping_or_empty(preparation.get("processed_input_audit"))
    mice = _mapping_or_empty(processed.get("mice"))
    if (
        processed.get("verified_before_split") is not True
        or processed.get("source_content_commitment_verified") is not True
        or processed.get("globally_verified_mouse_count") != 32
        or not _is_sha256(processed.get("source_content_commitment_sha256"))
        or processed.get("mouse_count") != len(ALLEN_LOCKED_ANIMALS)
        or set(mice) != set(ALLEN_LOCKED_ANIMALS)
    ):
        raise ValueError("Allen processed-input cohort was not fully authenticated")
    preprocessing_hashes: set[str] = set()
    for mouse, record in mice.items():
        row = _mapping_or_empty(record)
        if (
            not _is_sha256(row.get("legacy_sha256"))
            or not _is_sha256(row.get("index_row_sha256"))
            or not _is_sha256(row.get("preprocessing_configuration_sha256"))
            or set(_mapping_or_empty(row.get("source_files_sha256")))
            != {
                "stimulus_presentations.parquet",
                "window_index.parquet",
                "windows.npz",
            }
            or any(
                not _is_sha256(value)
                for value in _mapping_or_empty(row.get("source_files_sha256")).values()
            )
            or not isinstance(row.get("ophys_experiment_id"), int)
            or str(mouse) not in ALLEN_LOCKED_ANIMALS
        ):
            raise ValueError("Allen processed-input provenance row is malformed")
        preprocessing_hashes.add(str(row["preprocessing_configuration_sha256"]))
    if len(preprocessing_hashes) != 1:
        raise ValueError("Allen preprocessing configuration differs across mice")

    seals = _mapping_or_empty(preparation.get("target_seals"))
    if set(seals) != set(targets):
        raise ValueError("Allen target physical-seal scope is incomplete")
    for mouse, records in seals.items():
        by_name = _mapping_or_empty(records)
        if set(by_name) != {
            "legacy_combined",
            "role_sealed",
            "experiment_sealed",
        }:
            raise ValueError(f"Allen target seal is incomplete for mouse {mouse}")
        for record in by_name.values():
            row = _mapping_or_empty(record)
            try:
                sealed_mode = int(str(row.get("sealed_mode", "")), 8)
                original_mode = int(str(row.get("original_mode", "")), 8)
            except ValueError as error:
                raise ValueError("Allen target seal mode is malformed") from error
            if sealed_mode & 0o444 or not original_mode & 0o444:
                raise ValueError("Allen target outcome was not made unreadable")

    if set(prediction.get("report_methods", ())) != ALLEN_EXPECTED_METHODS:
        raise ValueError("Allen prediction report-method scope is not canonical")
    stage_records = _mapping_or_empty(prediction.get("stage_records"))
    if set(stage_records) != {"proposed", "linear", "additive", "black_box"}:
        raise ValueError("Allen learned-method stage records are incomplete")
    for method, record in stage_records.items():
        row = _mapping_or_empty(record)
        validation_mouse = str(row.get("inner_validation_mouse", ""))
        train_mice = set(donors) - {validation_mouse}
        selection = _mapping_or_empty(row.get("selection_boundary"))
        adapter = _mapping_or_empty(selection.get("inner_validation_adapter"))
        refit = _mapping_or_empty(row.get("refit_boundary"))
        if (
            validation_mouse not in donors
            or set(selection.get("shared_f_fit_mice", ())) != train_mice
            or set(selection.get("intervention_training_mice", ())) != train_mice
            or selection.get("intervention_validation_mice") != [validation_mouse]
            or selection.get("inner_validation_mimics_outer_target") is not True
            or adapter.get("mouse_id") != validation_mouse
            or adapter.get("shared_f_frozen") is not True
            or adapter.get("shared_state_sha256_before") != adapter.get("shared_state_sha256_after")
            or adapter.get("behavior_decoder_sha256_before")
            != adapter.get("behavior_decoder_sha256_after")
            or row.get("selection_validation_delta_present") is not False
            or refit.get("fresh_model") is not True
            or set(refit.get("normal_refit_mice", ())) != set(donors)
            or set(refit.get("intervention_refit_mice", ())) != set(donors)
            or refit.get("normal_refit_partitions") != ["fit", "val"]
        ):
            raise ValueError(f"Allen nested selection/refit audit failed for {method}")
        tolerance = _scalar(row.get("delta_projection_tolerance"))
        selection_norm = _scalar(row.get("selection_final_delta_mean_norm"))
        refit_norm = _scalar(row.get("refit_final_delta_mean_norm"))
        if (
            tolerance is None
            or isinstance(tolerance, bool)
            or float(tolerance) > 1e-7
            or selection_norm is None
            or isinstance(selection_norm, bool)
            or refit_norm is None
            or isinstance(refit_norm, bool)
            or float(selection_norm) > float(tolerance)
            or float(refit_norm) > float(tolerance)
        ):
            raise ValueError(f"Allen exact donor-delta projection failed for {method}")

    restoration = _mapping_or_empty(audit.get("target_outcome_mode_restoration"))
    restoration_mice = _mapping_or_empty(restoration.get("mice"))
    if (
        restoration.get("schema") != "cadence-allen-target-restoration-plan-v1"
        or restoration.get("restoration_status") != "PENDING_POST_SCORE_COMMIT"
        or restoration.get("journal_retained_until_score_commit") is not True
        or restoration.get("finalization_manifest") != "restore.complete.json"
        or restoration.get("canonical_relative_output") != payload.get("canonical_relative_output")
        or restoration.get("eligible_for_later_donor_reuse_after_finalization") is not True
        or set(restoration_mice) != set(targets)
        or _mapping_or_empty(completions["score"].get("metadata")).get(
            "canonical_processed_target_modes_restored"
        )
        is not False
        or _mapping_or_empty(completions["score"].get("metadata")).get(
            "target_mode_restoration_pending"
        )
        is not True
    ):
        raise ValueError("Allen target restoration plan is invalid")
    restored_mice = _mapping_or_empty(restoration_completion.get("mice"))
    for mouse, records in restoration_mice.items():
        plan_by_name = _mapping_or_empty(records)
        restored_by_name = _mapping_or_empty(restored_mice.get(mouse))
        if set(plan_by_name) != {
            "legacy_combined",
            "role_sealed",
            "experiment_sealed",
        } or set(restored_by_name) != set(plan_by_name):
            raise ValueError(f"Allen restoration plan is incomplete for mouse {mouse}")
        for name, record in plan_by_name.items():
            plan = _mapping_or_empty(record)
            restored = _mapping_or_empty(restored_by_name.get(name))
            seal = _mapping_or_empty(_mapping_or_empty(seals.get(mouse)).get(name))
            if (
                plan.get("path") != seal.get("path")
                or plan.get("expected_restored_mode") != seal.get("original_mode")
                or plan.get("sha256") != seal.get("sha256")
                or restored.get("path") != plan.get("path")
                or restored.get("restored_mode") != plan.get("expected_restored_mode")
                or restored.get("sha256") != plan.get("sha256")
            ):
                raise ValueError("Allen restoration plan/completion chain is malformed")

    random_effect_audit = _mapping_or_empty(audit.get("donor_intervention_random_effects"))
    required_true = (
        "selection_train_donors_only",
        "selection_centering_excludes_validation",
        "selection_f_fit_excludes_validation_mouse",
        "selection_validation_normal_adapter_with_f_frozen",
        "refit_fresh_model_on_all_donor_normals",
        "refit_all_donors_from_fresh_normal_refit",
        "exact_zero_mean_projection_after_each_step",
    )
    if (
        any(random_effect_audit.get(name) is not True for name in required_true)
        or random_effect_audit.get("selection_validation_delta_present") is not False
        or random_effect_audit.get("target_prediction_delta") != "integrated_at_zero_mean"
    ):
        raise ValueError("Allen donor-random-effect protocol audit is incomplete")


def _validate_allen_locked_artifacts(
    payload: Mapping[str, Any],
    source_file: str | Path | None,
) -> Mapping[str, Any]:
    if payload.get("run_profile") != "locked":
        return _invalid_artifact_validation("not an Allen locked-profile artifact")
    if payload.get("schema") != "cadence-allen-vbo-experiment-v2":
        return _invalid_artifact_validation("Allen headline requires the sealed v2 score schema")
    if source_file is None:
        return _invalid_artifact_validation(
            "Allen locked metrics require an authenticated source file"
        )
    path = Path(source_file)
    try:
        if path.name != "metrics.json" or not path.is_file():
            raise ValueError("Allen source must be a metrics.json file")
        directory = path.parent
        fold = payload.get("fold")
        if not isinstance(fold, int) or isinstance(fold, bool) or fold not in range(5):
            raise ValueError("Allen locked fold identity is invalid")
        canonical_relative_output = f"results/allen-vbo/locked-fold-{fold}"
        _require_canonical_source_path(
            path,
            Path(canonical_relative_output) / "metrics.json",
        )
        completions = {
            stage: _authenticate_completion_manifest(directory, stage)
            for stage in ("prepare", "predict", "score")
        }
        expected_metrics_sha = _mapping_or_empty(completions["score"].get("artifacts")).get(
            "metrics.json"
        )
        if not _is_sha256(expected_metrics_sha) or _sha256_file(path) != expected_metrics_sha:
            raise ValueError("score completion does not bind metrics.json")
        on_disk_metrics = json.loads(path.read_text(encoding="utf-8"))
        if on_disk_metrics != dict(payload):
            raise ValueError("Allen caller payload differs from authenticated metrics.json")
        preparation = json.loads((directory / "preparation.json").read_text(encoding="utf-8"))
        prediction = json.loads((directory / "prediction_run.json").read_text(encoding="utf-8"))
        transaction_sha = _authenticate_allen_seal_transaction(
            directory,
            preparation,
            canonical_relative_output,
        )
        if (
            payload.get("canonical_relative_output") != canonical_relative_output
            or preparation.get("canonical_relative_output") != canonical_relative_output
            or prediction.get("canonical_relative_output") != canonical_relative_output
            or any(
                _mapping_or_empty(completion.get("metadata")).get("canonical_relative_output")
                != canonical_relative_output
                for completion in completions.values()
            )
        ):
            raise ValueError("Allen canonical output binding chain is inconsistent")
        restoration_completion = _authenticate_allen_restoration_completion(
            directory,
            preparation=preparation,
            score_completion=completions["score"],
            canonical_relative_output=canonical_relative_output,
        )
        metric_attestation = _freeze_attestation(
            _mapping_or_empty(payload.get("protocol_audit")).get("preoutcome_freeze_attestation")
        )
        if _freeze_attestation(preparation.get("freeze_attestation")) != metric_attestation:
            raise ValueError("Allen prepare/score freeze attestations differ")
        if _freeze_attestation(prediction.get("freeze_attestation")) != metric_attestation:
            raise ValueError("Allen predict/score freeze attestations differ")
        _git_verify_annotated_attestation(metric_attestation)
        configuration = str(preparation.get("configuration_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", configuration):
            raise ValueError("Allen canonical run-configuration digest is missing")
        if prediction.get("configuration_sha256") != configuration:
            raise ValueError("Allen prepare/predict configuration digests differ")
        prediction_path = directory / "predictions.npz"
        prediction_sha = _read_authenticated_sidecar(prediction_path)
        protocol_audit = _mapping_or_empty(payload.get("protocol_audit"))
        with np.load(prediction_path, allow_pickle=False) as archive:
            if "metadata" not in archive.files or archive["metadata"].ndim != 0:
                raise ValueError("Allen prediction metadata is missing")
            prediction_metadata = json.loads(str(archive["metadata"].item()))
        if (
            prediction.get("prediction_sha256") != prediction_sha
            or protocol_audit.get("prediction_sha256_before_score") != prediction_sha
            or prediction_metadata.get("canonical_relative_output") != canonical_relative_output
        ):
            raise ValueError("Allen prediction hash chain is inconsistent")
        _authenticate_allen_transaction_references(
            transaction_sha,
            payload=payload,
            prediction=prediction,
            prediction_metadata=prediction_metadata,
            completions=completions,
            restoration_completion=restoration_completion,
        )
        _validate_allen_protocol_details(
            payload,
            preparation,
            prediction,
            completions,
            restoration_completion,
            metric_attestation,
        )
        return _valid_artifact_validation(
            source_file=path,
            attestation=metric_attestation,
            checks=(
                "annotated_tag_chain",
                "authenticated_payload_identity",
                "prepare_completion",
                "predict_completion",
                "score_completion",
                "post_score_restoration_completion",
                "target_seal_transaction_digest_chain",
                "canonical_output_binding_chain",
                "configuration_digest_chain",
                "prediction_before_score_hash_chain",
                "canonical_source_and_processed_provenance",
                "nested_selection_and_exact_delta_projection",
                "physical_seal_and_mode_restoration",
                "canonical_method_scope",
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as error:
        return _invalid_artifact_validation(str(error))


def _validate_icms_locked_artifacts(
    payload: Mapping[str, Any],
    source_file: str | Path | None,
) -> Mapping[str, Any]:
    if payload.get("schema") != "cadence-icms-score-v1":
        return _invalid_artifact_validation("ICMS headline requires the sealed score schema")
    if source_file is None:
        return _invalid_artifact_validation(
            "ICMS score metrics require an authenticated source file"
        )
    path = Path(source_file)
    try:
        if path.name != "metrics.json" or not path.is_file():
            raise ValueError("ICMS source must be a metrics.json file")
        directory = path.parent
        target = str(payload.get("target_animal", ""))
        canonical_relative_output = f"results/icms/loao-{target}"
        _require_canonical_source_path(
            path,
            Path(canonical_relative_output) / "metrics.json",
        )
        metrics_sha = _read_authenticated_sidecar(path)
        on_disk_metrics = json.loads(path.read_text(encoding="utf-8"))
        if on_disk_metrics != dict(payload):
            raise ValueError("ICMS caller payload differs from authenticated metrics.json")
        prepare_path = directory / "prepare_manifest.json"
        prediction_manifest_path = directory / "prediction_manifest.json"
        prepare_sha = _read_authenticated_sidecar(prepare_path)
        prediction_manifest_sha = _read_authenticated_sidecar(prediction_manifest_path)
        prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
        prediction = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
        transaction_sha = _authenticate_icms_seal_transaction(
            directory,
            prepare,
            canonical_relative_output,
        )
        attestation = _freeze_attestation(payload.get("freeze_attestation"))
        if any(
            _freeze_attestation(stage.get("freeze_attestation")) != attestation
            for stage in (prepare, prediction)
        ):
            raise ValueError("ICMS prepare/predict/score freeze attestations differ")
        _git_verify_annotated_attestation(attestation)
        completions = {
            "prepare": _authenticate_icms_completion(
                directory,
                "prepare",
                "prepare_manifest.json",
                attestation,
                canonical_relative_output,
                transaction_sha,
            ),
            "predict": _authenticate_icms_completion(
                directory,
                "predict",
                "prediction_manifest.json",
                attestation,
                canonical_relative_output,
                transaction_sha,
            ),
            "score": _authenticate_icms_completion(
                directory,
                "score",
                "metrics.json",
                attestation,
                canonical_relative_output,
                transaction_sha,
            ),
        }
        if (
            completions["prepare"].get("artifact_sha256") != prepare_sha
            or completions["predict"].get("artifact_sha256") != prediction_manifest_sha
            or completions["score"].get("artifact_sha256") != metrics_sha
        ):
            raise ValueError("ICMS completion digest chain is inconsistent")
        _authenticate_icms_transaction_references(
            transaction_sha,
            prepare=prepare,
            prediction=prediction,
            payload=payload,
            completions=completions,
        )
        if prediction.get("prepare_manifest_sha256") != prepare_sha:
            raise ValueError("ICMS prediction does not bind the prepare manifest")
        if any(
            stage.get("canonical_relative_output") != canonical_relative_output
            for stage in (prepare, prediction, payload)
        ):
            raise ValueError("ICMS canonical output binding chain is inconsistent")

        if (
            prepare.get("schema") != "cadence-icms-prepare-v1"
            or prediction.get("schema") != "cadence-icms-prediction-v1"
            or any(
                stage.get("dataset") != "DANDI:001868"
                or stage.get("dataset_version") != ICMS_DANDISET_VERSION
                or stage.get("run_mode") != "biological"
                for stage in (prepare, prediction, payload)
            )
        ):
            raise ValueError("ICMS stage schema, release, or biological mode is invalid")
        donors = [animal for animal in ICMS_TASK_MICE if animal != target]
        outer_mapping = {
            animal: [other for other in ICMS_TASK_MICE if other != animal]
            for animal in ICMS_TASK_MICE
        }
        if (
            target not in ICMS_TASK_MICE
            or prepare.get("target_animal") != target
            or prediction.get("target_animal") != target
            or prepare.get("donor_animals") != donors
            or prediction.get("donor_animals") != donors
            or prepare.get("canonical_target_order") != list(ICMS_TASK_MICE)
            or prepare.get("canonical_outer_mapping") != outer_mapping
            or prepare.get("outer_scheme") != "leave-one-animal-out"
            or prepare.get("intended_protocol_commit") != attestation["commit"]
            or prepare.get("required_preoutcome_tag") != PREOUTCOME_TAG
        ):
            raise ValueError("ICMS leave-one-animal-out target/donor scope is invalid")

        config = _validate_icms_full_config(prepare.get("config"))
        if prediction.get("config") != config:
            raise ValueError("ICMS prepare/predict configurations differ")
        config_sha = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if (
            prepare.get("config_sha256") != config_sha
            or prediction.get("config_sha256") != config_sha
        ):
            raise ValueError("ICMS configuration digest chain is invalid")
        canonical_scope = _mapping_or_empty(prediction.get("canonical_scope"))
        if (
            payload.get("canonical_scope") != canonical_scope
            or canonical_scope.get("full_profile") is not True
            or canonical_scope.get("seed") != LOCK_SEED
            or canonical_scope.get("ordered_report_methods") != list(ICMS_REPORT_METHOD_ORDER)
            or canonical_scope.get("canonical_target_order") != list(ICMS_TASK_MICE)
            or canonical_scope.get("outer_mapping") != outer_mapping
            or prediction.get("methods") != list(ICMS_REPORT_METHOD_ORDER)
        ):
            raise ValueError("ICMS canonical locked scope is incomplete")

        provenance = _mapping_or_empty(prepare.get("canonical_provenance"))
        source_root = str(provenance.get("source_root", ""))
        prepare_attestation = _mapping_or_empty(prepare.get("freeze_attestation"))
        if (
            not source_root
            or not Path(source_root).is_absolute()
            or prepare_attestation.get("source_root") != source_root
            or provenance.get("git_commit") != attestation["commit"]
            or provenance.get("preoutcome_tag") != PREOUTCOME_TAG
            or provenance.get("dandiset_id") != "001868"
            or provenance.get("dandiset_version") != ICMS_DANDISET_VERSION
            or provenance.get("index_totals") != ICMS_INDEX_TOTALS
            or provenance.get("canonical_target_order") != list(ICMS_TASK_MICE)
            or provenance.get("outer_mapping") != outer_mapping
        ):
            raise ValueError("ICMS canonical provenance scope is invalid")
        index_sha, index_blob = _validate_icms_tracked_identity(
            provenance.get("processed_index"),
            relative_path="data/processed/dandi_001868/index.json",
            commit=attestation["commit"],
        )
        raw_manifest_sha, raw_manifest_blob = _validate_icms_tracked_identity(
            provenance.get("raw_asset_manifest"),
            relative_path="configs/dandi_001868_assets.json",
            commit=attestation["commit"],
        )
        index = json.loads(index_blob)
        raw_manifest = json.loads(raw_manifest_blob)
        if (
            index.get("schema") != "cadence-dandi-001868-index-v1"
            or index.get("dandiset_id") != "001868"
            or index.get("dandiset_version") != ICMS_DANDISET_VERSION
            or index.get("split_unit") != "animal_id"
            or index.get("totals") != ICMS_INDEX_TOTALS
            or raw_manifest.get("dandiset_id") != "001868"
            or raw_manifest.get("version") != ICMS_DANDISET_VERSION
            or set(raw_manifest.get("task_mice", ())) != set(ICMS_TASK_MICE)
        ):
            raise ValueError("ICMS tracked index or raw-asset manifest is noncanonical")
        index_rows = index.get("animals", ())
        indexed_h5 = {str(row["animal_id"]): str(row["output_sha256"]) for row in index_rows}
        if set(indexed_h5) != set(ICMS_TASK_MICE) or len(index_rows) != len(ICMS_TASK_MICE):
            raise ValueError("ICMS tracked index does not contain exactly six task mice")
        if (
            provenance.get("provided_index_sha256") != index_sha
            or provenance.get("verified_h5_sha256") != indexed_h5
            or prepare.get("processed_source_sha256") != indexed_h5
            or canonical_scope.get("processed_index_sha256") != index_sha
            or canonical_scope.get("raw_asset_manifest_sha256") != raw_manifest_sha
            or prediction.get("verified_donor_source_sha256")
            != {animal: indexed_h5[animal] for animal in donors}
            or prediction.get("target_source_sha256_expected_but_not_opened") != indexed_h5[target]
        ):
            raise ValueError("ICMS processed/raw source digest chain is invalid")
        source_paths = _mapping_or_empty(prepare.get("processed_source_paths"))
        if set(source_paths) != set(ICMS_TASK_MICE) or any(
            Path(str(source_paths[animal])).name != f"sub-{animal}.h5" for animal in ICMS_TASK_MICE
        ):
            raise ValueError("ICMS processed source paths are incomplete")

        support_rows = prepare.get("normal_supports")
        query_rows = prepare.get("target_queries")
        if not isinstance(support_rows, list) or not isinstance(query_rows, list):
            raise ValueError("ICMS support/query manifests are malformed")
        support_animals: set[str] = set()
        target_support_by_adapter: dict[str, Mapping[str, Any]] = {}
        for row_value in support_rows:
            row = _mapping_or_empty(row_value)
            animal = str(row.get("animal_id", ""))
            adapter = str(row.get("adapter_id", ""))
            artifact = _safe_relative_artifact(directory, row.get("path"))
            observed = _read_authenticated_sidecar(artifact)
            if (
                animal not in ICMS_TASK_MICE
                or not adapter
                or row.get("sha256") != observed
                or artifact.parts[-3:-1] != ("support", animal)
            ):
                raise ValueError("ICMS normal-support artifact failed authentication")
            support_animals.add(animal)
            if animal == target:
                if adapter in target_support_by_adapter:
                    raise ValueError("duplicate ICMS target normal-support adapter")
                target_support_by_adapter[adapter] = row
        if support_animals != set(ICMS_TASK_MICE):
            raise ValueError("ICMS normal supports do not cover all six animals")
        query_by_adapter: dict[str, Mapping[str, Any]] = {}
        for row_value in query_rows:
            row = _mapping_or_empty(row_value)
            adapter = str(row.get("adapter_id", ""))
            artifact = _safe_relative_artifact(directory, row.get("path"))
            observed = _read_authenticated_sidecar(artifact)
            if (
                row.get("animal_id") != target
                or not adapter
                or adapter in query_by_adapter
                or row.get("sha256") != observed
                or artifact.parts[-3:-1] != ("queries", target)
                or adapter not in target_support_by_adapter
                or row.get("session_key") != target_support_by_adapter[adapter].get("session_key")
            ):
                raise ValueError("ICMS target-query artifact failed authentication")
            query_by_adapter[adapter] = row
        if not query_by_adapter:
            raise ValueError("ICMS target query manifest is empty")
        prediction_sessions = prediction.get("sessions")
        if not isinstance(prediction_sessions, list):
            raise ValueError("ICMS prediction session manifest is malformed")
        session_by_adapter = {
            str(_mapping_or_empty(row).get("adapter_id", "")): _mapping_or_empty(row)
            for row in prediction_sessions
        }
        array_keys = {
            str(_mapping_or_empty(row).get("array_key", "")) for row in prediction_sessions
        }
        if (
            len(session_by_adapter) != len(prediction_sessions)
            or set(session_by_adapter) != set(query_by_adapter)
            or len(array_keys) != len(prediction_sessions)
            or "" in array_keys
            or any(
                session_by_adapter[adapter].get("query_sha256")
                != query_by_adapter[adapter].get("sha256")
                or session_by_adapter[adapter].get("session_key")
                != query_by_adapter[adapter].get("session_key")
                or session_by_adapter[adapter].get("condition_count") != 416
                or session_by_adapter[adapter].get("current_lattice_uA")
                != [float(value) for value in range(1, 14)]
                or session_by_adapter[adapter].get("nearest_donor") not in donors
                for adapter in query_by_adapter
            )
        ):
            raise ValueError("ICMS query/prediction session hash chain is invalid")

        for filename, expected_key in (
            (
                str(prediction.get("prediction_path", "")),
                "prediction_sha256_before_target_open",
            ),
            (str(prediction.get("model_path", "")), "model_sha256"),
        ):
            artifact = _safe_relative_artifact(directory, filename)
            observed = _read_authenticated_sidecar(artifact)
            if observed != prediction.get(expected_key):
                raise ValueError(f"ICMS manifest hash mismatch for {filename}")
        if (
            prediction.get("prediction_path") != "predictions.npz"
            or prediction.get("model_path") != "frozen_models.pt"
        ):
            raise ValueError("ICMS canonical prediction/model filenames changed")
        expected_prediction_keys: set[str] = set()
        trajectory_names = (
            "neural_treated",
            "neural_control",
            "neural_effect",
            "behavior_treated",
            "behavior_control",
            "behavior_effect",
        )
        for session in session_by_adapter.values():
            array_key = str(session["array_key"])
            expected_prediction_keys.add(f"{array_key}__condition_descriptors")
            for method in ICMS_REPORT_METHOD_ORDER:
                expected_prediction_keys.update(
                    f"{method}__{array_key}__{name}" for name in trajectory_names
                )
            expected_prediction_keys.update(
                {
                    f"proposed__{array_key}__neural_effect_draws_condition_time",
                    f"proposed__{array_key}__behavior_effect_draws_condition_time",
                }
            )
        with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
            if set(archive.files) != expected_prediction_keys:
                raise ValueError("ICMS prediction array-key matrix is incomplete")
            for session in session_by_adapter.values():
                array_key = str(session["array_key"])
                descriptors = archive[f"{array_key}__condition_descriptors"]
                neural_draws = archive[f"proposed__{array_key}__neural_effect_draws_condition_time"]
                behavior_draws = archive[
                    f"proposed__{array_key}__behavior_effect_draws_condition_time"
                ]
                if (
                    descriptors.ndim != 2
                    or descriptors.shape[0] != 416
                    or neural_draws.ndim != 3
                    or neural_draws.shape[:2] != (64, 416)
                    or behavior_draws.ndim != 4
                    or behavior_draws.shape[:2] != (64, 416)
                    or behavior_draws.shape[-1] != 2
                ):
                    raise ValueError("ICMS prediction array shapes are noncanonical")
        for filename, expected_key in (
            ("sealed_target_outcomes.npz", "sealed_outcomes_sha256"),
            (
                "scored_condition_trajectories.npz",
                "scored_condition_trajectories_sha256",
            ),
            ("condition_metrics.csv", "condition_metrics_sha256"),
        ):
            artifact = directory / filename
            observed = _read_authenticated_sidecar(artifact)
            if observed != payload.get(expected_key):
                raise ValueError(f"ICMS score hash mismatch for {filename}")
        expected_condition_columns = [
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
        with (directory / "condition_metrics.csv").open(
            newline="",
            encoding="utf-8",
        ) as stream:
            condition_reader = csv.DictReader(stream)
            condition_rows = list(condition_reader)
        if (
            condition_reader.fieldnames != expected_condition_columns
            or not condition_rows
            or {row["animal_id"] for row in condition_rows} != {target}
            or {row["method"] for row in condition_rows} != set(ICMS_REPORT_METHOD_ORDER)
            or {row["session_key"] for row in condition_rows}
            != {str(session.get("session_key", "")) for session in session_by_adapter.values()}
        ):
            raise ValueError("ICMS condition-level CSV scope is invalid")
        restore_path = directory / "target_restore.json"
        restore_sha = _read_authenticated_sidecar(restore_path)
        restore = json.loads(restore_path.read_text(encoding="utf-8"))
        if (
            payload.get("physical_target_restore_sha256") != restore_sha
            or payload.get("physical_target_restore") != restore
            or restore.get("schema") != "cadence-icms-target-restore-v1"
            or restore.get("target_animal") != target
            or restore.get("target_path") != source_paths[target]
            or restore.get("sealed_mode") != 0
            or restore.get("restored_mode") != restore.get("original_mode")
            or restore.get("original_mode_restored_exactly") is not True
            or restore.get("registry_retained_until_score_commit") is not True
            or restore.get("canonical_relative_output") != canonical_relative_output
            or restore.get("restoration_status") != "PENDING_SCORE_COMMIT_FINALIZATION"
        ):
            raise ValueError("ICMS physical target restore audit is invalid")
        seal_row = _mapping_or_empty(prepare.get("physical_target_seal"))
        seal_path = _safe_relative_artifact(directory, seal_row.get("path"))
        seal_sha = _read_authenticated_sidecar(seal_path)
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        expected_seal = {
            key: value for key, value in seal_row.items() if key not in {"path", "sha256"}
        }
        if (
            seal_path.name != "target_seal.json"
            or seal_row.get("sha256") != seal_sha
            or seal != expected_seal
            or seal.get("schema") != "cadence-icms-physical-target-seal-v1"
            or seal.get("target_animal") != target
            or seal.get("target_path") != source_paths[target]
            or seal.get("canonical_relative_output") != canonical_relative_output
            or seal.get("expected_sha256") != indexed_h5[target]
            or seal.get("sealed_mode") != 0
            or seal.get("active") is not True
            or not isinstance(seal.get("processed_root"), str)
            or not Path(str(seal.get("processed_root"))).is_absolute()
            or not isinstance(seal.get("fold_directory"), str)
            or Path(str(seal.get("fold_directory"))).resolve() != directory.resolve()
            or not isinstance(seal.get("original_mode"), int)
            or int(seal.get("original_mode", 0)) & 0o444 == 0
            or restore.get("immutable_seal_sha256") != seal_sha
            or restore.get("device_id") != seal.get("device_id")
            or restore.get("inode") != seal.get("inode")
            or restore.get("original_mode") != seal.get("original_mode")
        ):
            raise ValueError("ICMS immutable physical target seal chain is invalid")
        _authenticate_icms_restoration_completion(
            directory,
            prepare=prepare,
            restore=restore,
            restore_sha256=restore_sha,
            score_completion=completions["score"],
            canonical_relative_output=canonical_relative_output,
        )
        if payload.get("prediction_sha256_verified_before_target_open") != prediction.get(
            "prediction_sha256_before_target_open"
        ):
            raise ValueError("ICMS prediction-before-unseal hash chain is inconsistent")
        if payload.get("target_source_sha256_verified_after_acknowledgement") != indexed_h5[target]:
            raise ValueError("ICMS restored target source hash is inconsistent")

        prepare_access = _mapping_or_empty(prepare.get("access_audit"))
        prediction_access = _mapping_or_empty(prediction.get("access_audit"))
        score_access = _mapping_or_empty(payload.get("access_audit"))
        if (
            prepare_access.get("prepare_target_stimulation_metadata_read") is not False
            or prepare_access.get("prepare_target_stimulation_signals_read") is not False
            or prepare_access.get("prepare_donor_stimulation_signals_read") is not False
            or prepare_access.get("target_h5_read_permission_after_prepare") is not False
            or prediction_access.get("target_stimulation_metadata_read") is not False
            or prediction_access.get("target_stimulation_outcomes_read") is not False
            or prediction_access.get("target_stimulation_trials_in_fit_or_validation") != 0
            or prediction_access.get("prediction_hashed_before_target_container_open") is not True
            or prediction_access.get("physical_target_seal_asserted_before_donor_open") is not True
            or prediction_access.get("physical_target_h5_mode_during_predict") != 0
            or prediction_access.get("physical_target_seal_sha256") != seal_sha
            or prediction_access.get("session_specific_observation_maps") is not True
            or prediction_access.get("encoder_receives_explicit_missingness_channels") is not True
            or prediction_access.get("zero_filled_missing_bins_without_mask_channel") is not False
            or prediction_access.get("donor_delta_grouping") != "animal_id"
            or prediction_access.get("inner_validation_unit") != "whole donor animal"
            or score_access.get("query_and_support_hashes_verified_before_target_open") is not True
            or score_access.get("prediction_hash_verified_before_target_open") is not True
            or score_access.get("model_hash_verified_before_target_open") is not True
            or score_access.get("target_stimulation_metadata_read_in_fit_or_predict") is not False
            or score_access.get("target_stimulation_outcomes_read_in_fit_or_predict") is not False
            or score_access.get("target_outcomes_opened_only_in_acknowledged_score") is not True
            or score_access.get("target_h5_original_mode_restored_exactly") is not True
            or score_access.get("immutable_target_seal_sha256") != seal_sha
            or score_access.get("sequential_loao_next_fold_ready") is not True
        ):
            raise ValueError("ICMS leakage-boundary access audit is invalid")

        _validate_icms_fit_audits(prediction, target=target, donors=donors)
        outcome_audit = _mapping_or_empty(payload.get("outcome_audit"))
        if (
            outcome_audit.get("post_onset_only") is not True
            or outcome_audit.get("target_outcomes_physically_separate_from_queries") is not True
            or not isinstance(outcome_audit.get("sessions"), list)
            or not outcome_audit["sessions"]
        ):
            raise ValueError("ICMS outcome materialization audit is invalid")
        eligibility = _mapping_or_empty(payload.get("causal_effect_eligibility"))
        primary_evaluable = eligibility.get("animal_eligible")
        expected_primary = target != ICMS_ABSOLUTE_ONLY_ANIMAL
        if (
            not isinstance(primary_evaluable, bool)
            or eligibility.get("primary_fold_status")
            != ("EVALUATED" if primary_evaluable else "NOT_EVALUATED")
            or eligibility.get("design_maximum_primary_eligible_n") != 5
            or eligibility.get("this_fold_primary_eligible_n") != int(primary_evaluable)
            or eligibility.get("session_catch_fallback_in_primary") is not False
            or eligibility.get("absolute_trajectory_n") != 6
            or eligibility.get("iti_is_randomized_counterfactual") is not False
            or primary_evaluable is not expected_primary
        ):
            raise ValueError("ICMS primary-estimand eligibility audit is invalid")
        session_status = _mapping_or_empty(eligibility.get("session_status"))
        if not session_status or any(
            status not in {"EVALUATED", "NOT_EVALUATED"} for status in session_status.values()
        ):
            raise ValueError("ICMS session eligibility statuses are malformed")
        if primary_evaluable and (
            any(status != "EVALUATED" for status in session_status.values())
            or any(
                _mapping_or_empty(row).get("block_validated") is not True
                or int(_mapping_or_empty(row).get("randomized_catch_trials", 0)) < 1
                for row in outcome_audit["sessions"]
            )
        ):
            raise ValueError("ICMS evaluated primary fold lacks same-block catch support")
        _validate_icms_score_aggregation(
            payload,
            primary_evaluable=primary_evaluable,
        )
        uncertainty = _mapping_or_empty(payload.get("uncertainty_audit"))
        prediction_uncertainty = _mapping_or_empty(prediction.get("uncertainty"))
        if (
            uncertainty.get("split_conformal") != "ABSENT_NOT_FIT"
            or uncertainty.get("donor_draw_interval") != "uncalibrated_marginal_5_95_quantiles"
            or uncertainty.get("simultaneous_coverage_exported") is not False
            or uncertainty.get("conformal_coverage_exported") is not False
            or prediction_uncertainty.get("split_conformal") != "ABSENT_NOT_FIT"
        ):
            raise ValueError("ICMS uncertainty scope is mislabeled or noncanonical")
        return {
            **_valid_artifact_validation(
                source_file=path,
                attestation=attestation,
                checks=(
                    "annotated_tag_chain",
                    "prepare_predict_score_completion_chain",
                    "post_score_restoration_completion",
                    "target_seal_transaction_digest_chain",
                    "canonical_output_binding_chain",
                    "canonical_target_donor_config_scope",
                    "tracked_raw_and_processed_provenance",
                    "normal_support_and_query_sidecars",
                    "prediction_before_target_open",
                    "scored_artifact_sidecars",
                    "physical_seal_and_exact_restore",
                    "nested_selection_fresh_refit_exact_delta_projection",
                    "leakage_boundary_access_audits",
                    "primary_estimand_eligibility",
                    "uncalibrated_uncertainty_scope",
                    "canonical_method_scope",
                ),
            ),
            "metrics_sha256": metrics_sha,
            "target_animal": target,
            "primary_estimand_evaluable": primary_evaluable,
        }
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as error:
        return _invalid_artifact_validation(str(error))


def _validate_teacher_locked_artifacts(
    payload: Mapping[str, Any],
    source_file: str | Path | None,
) -> Mapping[str, Any]:
    world = _mapping_or_empty(payload.get("world"))
    if world.get("seed_partition") != "locked":
        return _invalid_artifact_validation("not a teacher post-freeze procedural artifact")
    if source_file is None:
        return _invalid_artifact_validation(
            "teacher procedural metrics require an authenticated source file"
        )
    path = Path(source_file)
    try:
        if path.name != "metrics.json" or not path.is_file():
            raise ValueError("teacher source must be a metrics.json file")
        directory = path.parent
        metrics_sha = _read_authenticated_sidecar(path)
        on_disk_metrics = json.loads(path.read_text(encoding="utf-8"))
        if on_disk_metrics != dict(payload):
            raise ValueError("teacher caller payload differs from authenticated metrics.json")
        predictions = directory / "predictions.npz"
        prediction_sha = _read_authenticated_sidecar(predictions)
        with np.load(predictions, allow_pickle=False) as archive:
            if "metadata_json" not in archive:
                raise ValueError("teacher prediction metadata is missing")
            prediction_metadata = json.loads(str(archive["metadata_json"].item()))
        completion_path = directory / "completion.json"
        if not completion_path.is_file():
            raise ValueError("teacher completion manifest is missing")
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("schema_version") != "cadence.teacher_completion.v1":
            raise ValueError("teacher completion manifest schema is invalid")
        completion_artifacts = _mapping_or_empty(completion.get("artifacts"))
        if (
            completion_artifacts.get(path.name) != metrics_sha
            or completion_artifacts.get(predictions.name) != prediction_sha
        ):
            raise ValueError("teacher completion hash chain is inconsistent")
        audit = _mapping_or_empty(payload.get("protocol_audit"))
        attestation = _freeze_attestation(audit.get("preoutcome_freeze"))
        _git_verify_annotated_attestation(attestation)
        if (
            prediction_metadata.get("preoutcome_freeze") != attestation
            or completion.get("preoutcome_freeze") != attestation
        ):
            raise ValueError("teacher annotated freeze attestation is inconsistent")
        if audit.get("prediction_sha256_before_score") != prediction_sha:
            raise ValueError("teacher prediction-before-score hash chain is inconsistent")
        learned_methods = tuple(payload.get("learned_methods", ()))
        if (
            learned_methods != TEACHER_LEARNED_METHODS
            or payload.get("canonical_learned_method_set_complete") is not True
            or tuple(prediction_metadata.get("learned_methods", ())) != TEACHER_LEARNED_METHODS
            or prediction_metadata.get("canonical_learned_method_set_complete") is not True
            or tuple(completion.get("learned_methods", ())) != TEACHER_LEARNED_METHODS
            or completion.get("canonical_learned_method_set_complete") is not True
        ):
            raise ValueError("teacher learned-method scope is not complete and canonical")
        actual_reported_methods = set(
            _mapping_or_empty(payload.get("metrics_by_method_and_target"))
        )
        if (
            actual_reported_methods != TEACHER_EXPECTED_METHODS
            or set(payload.get("reported_methods", ())) != TEACHER_EXPECTED_METHODS
            or set(completion.get("reported_methods", ())) != TEACHER_EXPECTED_METHODS
        ):
            raise ValueError("teacher reported-method scope is not canonical")
        teacher_config_sha = audit.get("teacher_config_sha256")
        experiment_config_sha = audit.get("teacher_experiment_scientific_sha256")
        (
            expected_teacher_config_sha,
            expected_experiment_config_sha,
            expected_experiment_mapping,
            _,
            _,
            _,
        ) = _canonical_teacher_configuration()
        if (
            teacher_config_sha != expected_teacher_config_sha
            or experiment_config_sha != expected_experiment_config_sha
            or _teacher_scientific_experiment_mapping(payload.get("experiment_config"))
            != expected_experiment_mapping
        ):
            raise ValueError("teacher configuration is not the exact frozen full configuration")
        if any(
            value != teacher_config_sha
            for value in (
                world.get("teacher_config_sha256"),
                prediction_metadata.get("teacher_config_sha256"),
                completion.get("teacher_config_sha256"),
            )
        ) or any(
            value != experiment_config_sha
            for value in (
                prediction_metadata.get("teacher_experiment_scientific_sha256"),
                completion.get("teacher_experiment_scientific_sha256"),
            )
        ):
            raise ValueError("teacher configuration fingerprint chain is inconsistent")
        if world.get("seed_material_public") is not True:
            raise ValueError("teacher public-seed status is missing")
        if world.get("eligible_for_biological_headline_conjunction") is not False:
            raise ValueError("teacher artifact incorrectly claims biological eligibility")
        seed_index = world.get("seed_index")
        if (
            not isinstance(seed_index, int)
            or seed_index not in range(TEACHER_LOCKED_WORLDS)
            or world.get("world_seed") != TEACHER_LOCKED_SEEDS[seed_index]
        ):
            raise ValueError("teacher world seed/index is outside the canonical cohort")
        _require_canonical_source_path(
            path,
            Path("results")
            / "teacher-locked"
            / "full"
            / f"locked-seed-{seed_index:02d}"
            / "metrics.json",
        )
        canonical_relative_output = f"results/teacher-locked/full/locked-seed-{seed_index:02d}"
        if any(
            value != canonical_relative_output
            for value in (
                payload.get("canonical_relative_output"),
                audit.get("canonical_relative_output"),
                prediction_metadata.get("canonical_relative_output"),
                completion.get("canonical_relative_output"),
            )
        ):
            raise ValueError("teacher canonical one-shot output identity is inconsistent")
        world_id = str(world.get("world_id", ""))
        if world_id != TEACHER_LOCKED_WORLD_IDS[seed_index]:
            raise ValueError("teacher deterministic world identity is not canonical")
        metrics_by_method = _mapping_or_empty(payload.get("metrics_by_method_and_target"))
        target_ids = set(
            _mapping_or_empty(_mapping_or_empty(payload.get("stage_fits")).get("proposed")).get(
                "targets"
            )
        )
        if len(target_ids) != TEACHER_TARGETS_PER_WORLD or any(
            set(_mapping_or_empty(rows)) != target_ids for rows in metrics_by_method.values()
        ):
            raise ValueError("teacher reported target matrix is not complete and canonical")
        _validate_teacher_fit_audits(payload)
        return {
            **_valid_artifact_validation(
                source_file=path,
                attestation=attestation,
                checks=(
                    "annotated_tag_chain",
                    "metrics_sidecar",
                    "authenticated_payload_identity",
                    "completion_manifest_hash_chain",
                    "prediction_before_score_hash_chain",
                    "canonical_method_scope",
                    "exact_frozen_configuration_and_fingerprint_chain",
                    "canonical_public_seed_and_world_identity",
                    "canonical_one_shot_output_identity",
                    "nested_selection_fresh_refit_exact_delta_projection",
                    "biological_ineligibility",
                ),
            ),
            "seed_index": seed_index,
            "world_id": world_id,
        }
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as error:
        return _invalid_artifact_validation(str(error))


def adapt_allen_payload(
    payload: Mapping[str, Any],
    *,
    source_file: str | Path | None = None,
) -> AdaptedBatch:
    """Adapt one Allen fold JSON without pooling trials or folds."""

    schema = str(payload.get("schema", ""))
    supported_schemas = {
        "cadence-allen-vbo-experiment-v1",
        "cadence-allen-vbo-experiment-v2",
    }
    if schema and schema not in supported_schemas:
        raise ValueError(f"unsupported Allen schema: {schema}")
    by_method = _require_method_mapping(payload.get("animals"), "Allen animals")
    profile = str(payload.get("run_profile", "unknown"))
    fold_value = payload.get("fold")
    fold = int(fold_value) if fold_value is not None else None
    targets = tuple(str(value) for value in payload.get("targets", ()))
    if not targets:
        targets = tuple(
            sorted(
                {
                    str(animal)
                    for animals in by_method.values()
                    if isinstance(animals, Mapping)
                    for animal in animals
                }
            )
        )
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("Allen payload must identify unique target mice")
    target_set = set(targets)
    records: list[AnimalResult] = []
    source = _source_label(source_file)
    for method, animals in by_method.items():
        if not isinstance(animals, Mapping):
            raise ValueError(f"Allen method {method!r} does not map mice to metrics")
        unexpected = {str(animal) for animal in animals} - target_set
        if unexpected:
            raise ValueError(
                f"Allen method {method!r} contains non-target mice: {sorted(unexpected)}"
            )
        for animal, metrics in animals.items():
            if not isinstance(metrics, Mapping):
                raise ValueError(f"Allen metrics for {method}/{animal} are not a mapping")
            animal_id = str(animal)
            records.append(
                AnimalResult(
                    dataset="allen_vbo",
                    cohort=profile,
                    unit_id=animal_id,
                    animal_id=animal_id,
                    method=str(method),
                    metrics=dict(metrics),
                    source_file=source,
                    run_id=f"{profile}-fold-{fold}" if fold is not None else profile,
                    fold=fold,
                    randomized_estimand=True,
                )
            )
    # The v2 runner does not implement the full positive-claim falsification
    # suite.  Do not let a post-hoc field injected into metrics manufacture
    # evidence that the frozen scorer never emitted.
    headline_evidence = (
        {}
        if schema == "cadence-allen-vbo-experiment-v2"
        else _mapping_or_empty(payload.get("headline_evidence"))
    )
    return AdaptedBatch(
        records=tuple(records),
        headline_evidence=headline_evidence,
        artifact_validation=_validate_allen_locked_artifacts(payload, source_file),
    )


def adapt_teacher_payload(
    payload: Mapping[str, Any],
    *,
    source_file: str | Path | None = None,
) -> AdaptedBatch:
    """Adapt target rows from one procedural world; inference later averages them."""

    schema = str(payload.get("schema_version", ""))
    if schema and schema != "cadence.teacher_experiment.v1":
        raise ValueError(f"unsupported teacher schema: {schema}")
    by_method = _require_method_mapping(
        payload.get("metrics_by_method_and_target"),
        "teacher metrics_by_method_and_target",
    )
    world = _mapping_or_empty(payload.get("world"))
    world_id = str(world.get("world_id", "unknown-world"))
    cohort = str(world.get("seed_partition", "unknown"))
    records: list[AnimalResult] = []
    source = _source_label(source_file)
    for method, animals in by_method.items():
        if not isinstance(animals, Mapping):
            raise ValueError(f"teacher method {method!r} does not map targets to metrics")
        for animal, metrics in animals.items():
            if not isinstance(metrics, Mapping):
                raise ValueError(f"teacher metrics for {method}/{animal} are not a mapping")
            animal_id = str(animal)
            records.append(
                AnimalResult(
                    dataset="teacher",
                    cohort=cohort,
                    unit_id=f"{world_id}/{animal_id}",
                    animal_id=animal_id,
                    method=str(method),
                    metrics=dict(metrics),
                    source_file=source,
                    run_id=world_id,
                    world_id=world_id,
                    randomized_estimand=True,
                )
            )
    return AdaptedBatch(
        records=tuple(records),
        # Teacher outputs are procedural diagnostics, never biological gate
        # evidence, regardless of any user-authored top-level status field.
        headline_evidence={},
        artifact_validation=_validate_teacher_locked_artifacts(payload, source_file),
    )


def adapt_icms_payload(
    payload: Mapping[str, Any],
    *,
    source_file: str | Path | None = None,
) -> AdaptedBatch:
    """Adapt one sealed ICMS per-animal result JSON.

    The adapter accepts ``animals``, ``metrics_by_method_and_animal``, or
    ``metrics_by_method_and_target``.  It also accepts the fold-native
    ``target_animal`` plus ``animal_aggregate`` shape.  Regardless of input
    metadata, ICMS83 is placed in the fixed ``absolute_only`` cohort and never
    contributes to a randomized causal summary.  The other five frozen IDs
    are placed in ``randomized_n5``.
    """

    schema = str(payload.get("schema", ""))
    by_method: Any = None
    generic_keys = (
        "animals",
        "metrics_by_method_and_animal",
        "metrics_by_method_and_target",
    )
    if schema == "cadence-icms-score-v1":
        if any(payload.get(key) is not None for key in generic_keys):
            raise ValueError(
                "sealed ICMS score artifacts must use only target_animal + animal_aggregate"
            )
        if payload.get("animal_aggregate") is None:
            raise ValueError("sealed ICMS score artifact is missing animal_aggregate")
    else:
        for key in generic_keys:
            if payload.get(key) is not None:
                by_method = payload[key]
                break
    if by_method is None and payload.get("animal_aggregate") is not None:
        animal = payload.get("target_animal")
        if animal is None:
            raise ValueError("ICMS animal_aggregate requires target_animal")
        aggregate = _require_method_mapping(payload["animal_aggregate"], "ICMS animal_aggregate")
        by_method = {method: {str(animal): metrics} for method, metrics in aggregate.items()}
    by_method = _require_method_mapping(by_method, "ICMS per-animal metrics")
    records: list[AnimalResult] = []
    source = _source_label(source_file)
    run_id = str(payload.get("run_id", payload.get("fold", "icms-loao")))
    for method, animals in by_method.items():
        if not isinstance(animals, Mapping):
            raise ValueError(f"ICMS method {method!r} does not map animals to metrics")
        for animal, metrics in animals.items():
            if not isinstance(metrics, Mapping):
                raise ValueError(f"ICMS metrics for {method}/{animal} are not a mapping")
            if schema == "cadence-icms-score-v1":
                allowed_metrics = {
                    "absolute_neural_nrmse_equal_session",
                    "absolute_behavior_nrmse_equal_session",
                    "neural_causal_skill_equal_session",
                    "behavior_causal_skill_equal_session",
                    "nonrandomized_iti_neural_skill_equal_session",
                    "nonprimary_session_fallback_neural_skill_equal_session",
                    "eligible_sessions",
                    "primary_fold_status",
                    "neural_energy_score",
                    "behavior_energy_score",
                    "uncalibrated_marginal_neural_pointwise_coverage",
                    "uncalibrated_marginal_behavior_pointwise_coverage",
                }
                unexpected_metrics = set(metrics) - allowed_metrics
                if unexpected_metrics:
                    raise ValueError(
                        "sealed ICMS score contains non-producer aggregate fields: "
                        f"{sorted(unexpected_metrics)}"
                    )
            animal_id = str(animal)
            if animal_id == ICMS_ABSOLUTE_ONLY_ANIMAL:
                cohort = "absolute_only"
                randomized = False
            elif animal_id in ICMS_RANDOMIZED_ANIMALS:
                cohort = "randomized_n5"
                randomized = True
            else:
                raise ValueError(f"ICMS payload contains an unfrozen animal: {animal_id}")
            records.append(
                AnimalResult(
                    dataset="icms",
                    cohort=cohort,
                    unit_id=animal_id,
                    animal_id=animal_id,
                    method=str(method),
                    metrics=dict(metrics),
                    source_file=source,
                    run_id=run_id,
                    randomized_estimand=randomized,
                )
            )
    return AdaptedBatch(
        records=tuple(records),
        headline_evidence=(
            {}
            if schema == "cadence-icms-score-v1"
            else _mapping_or_empty(payload.get("headline_evidence"))
        ),
        artifact_validation=_validate_icms_locked_artifacts(payload, source_file),
    )


def adapt_payload(
    payload: Mapping[str, Any],
    *,
    kind: Literal["auto", "allen", "teacher", "icms"] = "auto",
    source_file: str | Path | None = None,
) -> AdaptedBatch:
    """Dispatch a metrics payload to its frozen adapter."""

    selected = kind
    if selected == "auto":
        schema = str(payload.get("schema", payload.get("schema_version", ""))).lower()
        if "allen" in schema:
            selected = "allen"
        elif "teacher" in schema:
            selected = "teacher"
        elif "icms" in schema:
            selected = "icms"
        elif "metrics_by_method_and_target" in payload and "world" in payload:
            selected = "teacher"
        else:
            raise ValueError("cannot infer result kind; pass an explicit adapter kind")
    if selected == "allen":
        return adapt_allen_payload(payload, source_file=source_file)
    if selected == "teacher":
        return adapt_teacher_payload(payload, source_file=source_file)
    if selected == "icms":
        return adapt_icms_payload(payload, source_file=source_file)
    raise ValueError(f"unknown adapter kind: {selected}")


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def eligible_non_oracle_baseline(method: str, *, proposed: str = PROPOSED_METHOD) -> bool:
    """Return the fixed baseline eligibility decision.

    All available methods other than the proposed model and explicitly named
    target-outcome oracles are included.  In particular, zero-effect,
    templates, mechanistic baselines, black boxes, and non-oracle ablations all
    enter the conservative envelope.
    """

    lowered = method.lower()
    return method != proposed and not any(token in lowered for token in _ORACLE_TOKENS)


def method_role(method: str, *, proposed: str = PROPOSED_METHOD) -> str:
    if method == proposed:
        return "proposed"
    if not eligible_non_oracle_baseline(method, proposed=proposed):
        return "oracle_ineligible"
    if method.startswith(f"{proposed}_"):
        return "non_oracle_ablation"
    return "eligible_baseline"


def equal_animal_bootstrap_ci(
    values: ArrayLike,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Percentile CI with animals (not trials, cells, or sessions) resampled."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if repeats < 1:
        raise ValueError("bootstrap repeats must be positive")
    result: dict[str, Any] = {
        "n": int(array.size),
        "estimate": float(np.mean(array)) if array.size else None,
        "ci_lower": None,
        "ci_upper": None,
        "confidence": confidence,
        "bootstrap_repeats": repeats,
        "equal_animal_weight": True,
    }
    if array.size < 2:
        return result
    generator = np.random.default_rng(seed)
    # Generate in bounded chunks so 20,000 x a future large cohort stays cheap.
    estimates = np.empty(repeats, dtype=np.float64)
    cursor = 0
    chunk_size = max(1, min(repeats, 4096))
    while cursor < repeats:
        stop = min(repeats, cursor + chunk_size)
        indices = generator.integers(0, array.size, size=(stop - cursor, array.size))
        estimates[cursor:stop] = np.mean(array[indices], axis=1)
        cursor = stop
    alpha = 1.0 - confidence
    lower, upper = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    result["ci_lower"] = float(lower)
    result["ci_upper"] = float(upper)
    return result


def exact_binomial_lower_confidence_bound(
    successes: int,
    trials: int,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
) -> float:
    """One-sided exact Clopper--Pearson lower bound for a binomial proportion."""

    if isinstance(successes, bool) or isinstance(trials, bool):
        raise TypeError("successes and trials must be integers")
    if not isinstance(successes, int | np.integer) or not isinstance(trials, int | np.integer):
        raise TypeError("successes and trials must be integers")
    successes = int(successes)
    trials = int(trials)
    if trials < 1:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie between zero and trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if successes == 0:
        return 0.0
    alpha = 1.0 - confidence
    return float(beta_distribution.ppf(alpha, successes, trials - successes + 1))


def _signed_sums(values: FloatArray) -> FloatArray:
    """All signed sums for one half of a meet-in-the-middle exact test."""

    sums = np.asarray([0.0], dtype=np.float64)
    for value in values:
        sums = np.concatenate((sums + value, sums - value))
    return sums


def exact_paired_sign_flip_test(
    differences: ArrayLike,
    *,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    null_value: float = 0.0,
    max_units: int = 40,
) -> float:
    """Exhaustive sign-flip p-value via meet-in-the-middle enumeration.

    Unlike a Monte Carlo permutation test, this returns ``extreme / 2**n``
    with no plus-one approximation.  Splitting the cohort makes the frozen
    28-mouse Allen test require only two arrays of length 16,384.
    """

    values = np.asarray(differences, dtype=np.float64).reshape(-1) - null_value
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("at least one finite paired difference is required")
    if values.size > max_units:
        raise ValueError(
            f"exact sign-flip test supports at most {max_units} units, got {values.size}"
        )
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(f"unknown alternative: {alternative}")
    split = values.size // 2
    left = _signed_sums(values[:split])
    right = np.sort(_signed_sums(values[split:]))
    observed = float(np.sum(values))
    tolerance = 16.0 * np.finfo(np.float64).eps * max(1.0, float(np.sum(np.abs(values))))
    total = int(left.size * right.size)

    if alternative == "greater":
        thresholds = observed - left - tolerance
        extreme = np.sum(right.size - np.searchsorted(right, thresholds, side="left"))
    elif alternative == "less":
        thresholds = observed - left + tolerance
        extreme = np.sum(np.searchsorted(right, thresholds, side="right"))
    else:
        bound = abs(observed)
        upper_threshold = bound - left - tolerance
        lower_threshold = -bound - left + tolerance
        upper_count = right.size - np.searchsorted(right, upper_threshold, side="left")
        lower_count = np.searchsorted(right, lower_threshold, side="right")
        extreme = np.sum(upper_count + lower_count)
    return float(min(int(extreme), total) / total)


def _stable_seed(base: int, *parts: object) -> int:
    digest = hashlib.sha256(
        "\0".join([str(base), *(str(part) for part in parts)]).encode()
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _endpoint_metadata(dataset: str, metric: str) -> tuple[str, str]:
    lowered = metric.lower()
    if lowered.startswith("neural") or "neural_" in lowered:
        return "neural", "primary"
    if lowered.startswith("running") or "running_" in lowered:
        return "running", "primary" if dataset == "allen_vbo" else "secondary"
    if lowered.startswith("pupil") or "pupil_" in lowered:
        return "pupil", "secondary"
    if lowered.startswith("lick") or "lick_" in lowered:
        return "lick", "secondary"
    if (
        lowered.startswith("behavior")
        or "behavior_" in lowered
        or lowered.startswith("wheel")
        or "wheel_" in lowered
    ):
        return "behavior", "primary"
    if lowered.startswith("latent") or "operator" in lowered or "vector_field" in lowered:
        return "latent_diagnostic", "diagnostic"
    return "joint_or_other", "other"


def _primary_metric_aliases(dataset: str, domain: str) -> tuple[str, ...]:
    if dataset == "allen_vbo":
        if domain == "neural":
            return ("neural_causal_skill",)
        return ("running_causal_skill",)
    if dataset == "teacher":
        if domain == "neural":
            return ("neural_condition_averaged_causal_skill",)
        return ("behavior_condition_averaged_causal_skill",)
    if domain == "neural":
        return (
            "neural_causal_skill",
            "neural_causal_skill_equal_session",
            "spike_causal_skill",
        )
    return (
        "behavior_causal_skill",
        "behavior_causal_skill_equal_session",
        "wheel_causal_skill",
    )


def _first_metric(metrics: Mapping[str, Any], aliases: Sequence[str]) -> float | bool | None:
    for alias in aliases:
        value = _scalar(metrics.get(alias))
        if value is not None:
            return value
    return None


def _records_by_unit_method(
    records: Sequence[AnimalResult],
) -> dict[tuple[str, str], AnimalResult]:
    indexed: dict[tuple[str, str], AnimalResult] = {}
    for record in records:
        key = (record.unit_id, record.method)
        if key in indexed:
            previous = indexed[key]
            raise ValueError(
                "duplicate target row/method would overweight a target: "
                f"{key} in {previous.source_file!r} and {record.source_file!r}"
            )
        indexed[key] = record
    return indexed


def _expected_allen_fold(animal_id: str) -> int:
    ordered = sorted(
        ALLEN_LOCKED_ANIMALS,
        key=lambda mouse: (
            hashlib.sha256(f"{mouse}{LOCK_SEED}".encode()).hexdigest(),
            mouse,
        ),
    )
    return ordered.index(animal_id) % 5


def _expected_methods(dataset: str) -> frozenset[str] | None:
    if dataset == "allen_vbo":
        return ALLEN_EXPECTED_METHODS
    if dataset == "icms":
        return ICMS_EXPECTED_METHODS
    if dataset == "teacher":
        return TEACHER_EXPECTED_METHODS
    return None


def _locked_scope_validation(
    records: Sequence[AnimalResult],
    validations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the complete run matrix, not merely proposed-method IDs."""

    dataset = records[0].dataset
    cohort = records[0].cohort
    expected = _expected_methods(dataset)
    methods_by_unit: dict[str, set[str]] = defaultdict(set)
    for record in records:
        methods_by_unit[record.unit_id].add(record.method)
    method_mismatches = {
        unit: {
            "missing": sorted((expected or frozenset()) - methods),
            "unexpected": sorted(methods - (expected or frozenset())),
        }
        for unit, methods in sorted(methods_by_unit.items())
        if expected is not None and methods != expected
    }

    if dataset == "allen_vbo" and cohort == "locked":
        expected_batches = 5
        observed_scopes = {record.fold for record in records}
        fold_mismatches = sorted(
            {
                record.animal_id
                for record in records
                if record.animal_id in ALLEN_LOCKED_ANIMALS
                and record.fold != _expected_allen_fold(record.animal_id)
            }
        )
        topology_valid = observed_scopes == set(range(5)) and not fold_mismatches
        topology = {
            "expected_folds": list(range(5)),
            "observed_folds": sorted(value for value in observed_scopes if value is not None),
            "fold_mismatch_animals": fold_mismatches,
        }
    elif dataset == "icms":
        expected_batches = 1 if cohort == "absolute_only" else 5
        labels_valid = all(
            record.randomized_estimand == (cohort == "randomized_n5") for record in records
        )
        expected_targets = (
            {ICMS_ABSOLUTE_ONLY_ANIMAL}
            if cohort == "absolute_only"
            else set(ICMS_RANDOMIZED_ANIMALS)
        )
        record_targets = {record.animal_id for record in records}
        authenticated_targets = {
            validation.get("target_animal")
            for validation in validations
            if validation.get("valid") is True
        }
        primary_flags = [
            validation.get("primary_estimand_evaluable")
            for validation in validations
            if validation.get("valid") is True
        ]
        eligibility_valid = (
            all(flag is True for flag in primary_flags)
            if cohort == "randomized_n5"
            else all(flag is False for flag in primary_flags)
        )
        topology_valid = (
            labels_valid
            and record_targets == expected_targets
            and authenticated_targets == expected_targets
            and len(primary_flags) == expected_batches
            and eligibility_valid
        )
        topology = {
            "expected_leave_one_animal_out_folds": expected_batches,
            "randomized_estimand_labels_consistent": labels_valid,
            "exact_target_set": record_targets == expected_targets,
            "authenticated_target_set": authenticated_targets == expected_targets,
            "primary_estimand_evaluable": eligibility_valid,
        }
    elif dataset == "teacher" and cohort == "locked":
        expected_batches = TEACHER_LOCKED_WORLDS
        world_ids = {record.world_id for record in records if record.world_id is not None}
        authenticated_seed_indices = {
            validation.get("seed_index")
            for validation in validations
            if validation.get("valid") is True
        }
        authenticated_world_ids = {
            validation.get("world_id")
            for validation in validations
            if validation.get("valid") is True
        }
        topology_valid = (
            world_ids == set(TEACHER_LOCKED_WORLD_IDS)
            and authenticated_seed_indices == set(range(TEACHER_LOCKED_WORLDS))
            and authenticated_world_ids == set(TEACHER_LOCKED_WORLD_IDS)
        )
        topology = {
            "expected_worlds": TEACHER_LOCKED_WORLDS,
            "observed_worlds": len(world_ids),
            "exact_seed_index_set": (
                authenticated_seed_indices == set(range(TEACHER_LOCKED_WORLDS))
            ),
            "deterministic_world_id_set": (
                authenticated_world_ids == set(TEACHER_LOCKED_WORLD_IDS)
            ),
        }
    else:
        expected_batches = None
        topology_valid = True
        topology = {}

    artifact_failures = [
        str(validation.get("reason", "artifact validation did not pass"))
        for validation in validations
        if validation.get("valid") is not True
    ]
    source_files = [
        str(validation.get("source_file", ""))
        for validation in validations
        if validation.get("valid") is True
    ]
    authenticated_inputs = [
        {
            "source_file": str(validation.get("source_file", "")),
            "sha256": str(validation.get("source_sha256", "")),
        }
        for validation in validations
        if validation.get("valid") is True
    ]
    unique_sources = {value for value in source_files if value}
    source_count_valid = expected_batches is None or (
        len(validations) == expected_batches and len(unique_sources) == expected_batches
    )
    input_hashes_valid = expected_batches is None or (
        len(authenticated_inputs) == expected_batches
        and all(item["source_file"] and _is_sha256(item["sha256"]) for item in authenticated_inputs)
    )
    attestations = [
        json.dumps(validation.get("freeze_attestation"), sort_keys=True)
        for validation in validations
        if validation.get("valid") is True
    ]
    attestation_valid = expected_batches is None or (
        len(attestations) == expected_batches
        and len(set(attestations)) == 1
        and attestations[0] != "null"
    )
    valid = (
        not method_mismatches
        and topology_valid
        and not artifact_failures
        and source_count_valid
        and input_hashes_valid
        and attestation_valid
    )
    freeze_attestation = json.loads(attestations[0]) if attestation_valid and attestations else None
    return {
        "valid": valid,
        "expected_methods": sorted(expected or ()),
        "method_mismatches": method_mismatches,
        "expected_artifact_batches": expected_batches,
        "observed_artifact_batches": len(validations),
        "unique_authenticated_sources": len(unique_sources),
        "authenticated_inputs": sorted(
            authenticated_inputs,
            key=lambda item: (item["source_file"], item["sha256"]),
        ),
        "authenticated_input_hashes_complete": input_hashes_valid,
        "consistent_freeze_attestation": attestation_valid,
        "freeze_attestation": freeze_attestation,
        "artifact_failures": artifact_failures,
        "topology": topology,
        "reason": (
            "exact canonical run matrix and artifact chain authenticated"
            if valid
            else "canonical run matrix or artifact authentication is incomplete"
        ),
    }


def _cohort_completeness(
    records: Sequence[AnimalResult],
    validations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail closed unless the entire frozen headline cohort is present."""

    if not records:
        raise ValueError("cohort completeness requires records")
    dataset = records[0].dataset
    cohort = records[0].cohort
    scope = _locked_scope_validation(records, validations)
    proposed = [record for record in records if record.method == PROPOSED_METHOD]
    proposed_units = {record.unit_id for record in proposed}
    base: dict[str, Any] = {
        "headline_profile": False,
        "complete": False,
        "gates_evaluable": False,
        "observed_proposed_units": len(proposed_units),
        "locked_scope_validation": scope,
    }

    if dataset == "allen_vbo":
        base["expected_units"] = len(ALLEN_LOCKED_ANIMALS)
        if cohort != "locked":
            return {
                **base,
                "reason": "Allen development/non-locked profiles are not headline analyses",
            }
        observed = {record.animal_id for record in proposed}
        missing = sorted(ALLEN_LOCKED_ANIMALS - observed)
        unexpected = sorted(observed - ALLEN_LOCKED_ANIMALS)
        cohort_complete = (
            not missing and not unexpected and len(proposed_units) == len(ALLEN_LOCKED_ANIMALS)
        )
        complete = cohort_complete and bool(scope["valid"])
        return {
            **base,
            "headline_profile": True,
            "complete": complete,
            "gates_evaluable": complete,
            "missing_animals": missing,
            "unexpected_animals": unexpected,
            "reason": (
                "complete authenticated frozen 28-mouse Allen cohort"
                if complete
                else (
                    "Allen headline requires all 28 frozen evaluation mice, all "
                    "five exact folds/method matrices, and authenticated artifacts"
                )
            ),
        }

    if dataset == "icms":
        if cohort == "absolute_only":
            return {
                **base,
                "expected_units": 1,
                "reason": ("ICMS83 has no randomized catch trials and is absolute-trajectory only"),
            }
        base["expected_units"] = len(ICMS_RANDOMIZED_ANIMALS)
        if cohort != "randomized_n5":
            return {
                **base,
                "reason": "unknown ICMS cohort is not a frozen headline analysis",
            }
        observed = {record.animal_id for record in proposed}
        missing = sorted(ICMS_RANDOMIZED_ANIMALS - observed)
        unexpected = sorted(observed - ICMS_RANDOMIZED_ANIMALS)
        cohort_complete = (
            not missing and not unexpected and len(proposed_units) == len(ICMS_RANDOMIZED_ANIMALS)
        )
        complete = cohort_complete and bool(scope["valid"])
        return {
            **base,
            "headline_profile": True,
            "complete": complete,
            "gates_evaluable": complete,
            "missing_animals": missing,
            "unexpected_animals": unexpected,
            "reason": (
                "complete authenticated five-mouse randomized ICMS cohort"
                if complete
                else (
                    "ICMS causal headline requires all five catch-supported mice, "
                    "the exact method matrix, and authenticated fold artifacts"
                )
            ),
        }

    if dataset == "teacher":
        world_targets: dict[str, set[str]] = defaultdict(set)
        for record in proposed:
            if record.world_id is not None:
                world_targets[record.world_id].add(record.animal_id)
        observed_worlds = len(world_targets)
        target_counts = {world: len(targets) for world, targets in sorted(world_targets.items())}
        base.update(
            {
                "expected_units": TEACHER_LOCKED_WORLDS,
                "expected_worlds": TEACHER_LOCKED_WORLDS,
                "expected_targets_per_world": TEACHER_TARGETS_PER_WORLD,
                "expected_target_units": TEACHER_LOCKED_TARGET_UNITS,
                "observed_proposed_units": observed_worlds,
                "observed_worlds": observed_worlds,
                "observed_target_units": len(proposed_units),
                "targets_per_world": target_counts,
                "procedural_evaluation": True,
                "seed_material_public": True,
                "biological_headline_eligible": False,
            }
        )
        if cohort != "locked":
            return {
                **base,
                "reason": (
                    "teacher development/non-locked profiles are procedural "
                    "and not headline analyses"
                ),
            }
        cohort_complete = (
            len(proposed_units) == TEACHER_LOCKED_TARGET_UNITS
            and observed_worlds == TEACHER_LOCKED_WORLDS
            and all(count == TEACHER_TARGETS_PER_WORLD for count in target_counts.values())
        )
        complete = cohort_complete and bool(scope["valid"])
        return {
            **base,
            "complete": complete,
            "headline_profile": False,
            "gates_evaluable": False,
            "schema_partition_label": "locked",
            "inference_scope": "procedural_world_level_intervals_only",
            "reason": (
                "teacher uses 20 public-seed deterministic post-freeze worlds "
                "and is ineligible for a headline conjunction"
                if complete
                else (
                    "teacher procedural evaluation is incomplete and, even "
                    "when complete, is ineligible for a headline conjunction"
                )
            ),
        }

    return {
        **base,
        "expected_units": None,
        "reason": "unknown dataset is not a frozen headline analysis",
    }


def _teacher_world_average_records(
    records: Sequence[AnimalResult],
) -> list[AnimalResult]:
    """Average nested teacher targets before any inferential operation."""

    proposed_targets: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.world_id is None:
            raise ValueError("teacher inference requires an explicit world_id")
        if record.method == PROPOSED_METHOD:
            proposed_targets[record.world_id].add(record.animal_id)
    grouped: dict[tuple[str, str], list[AnimalResult]] = defaultdict(list)
    for record in records:
        if record.world_id is None:
            raise ValueError("teacher inference requires an explicit world_id")
        grouped[(record.world_id, record.method)].append(record)

    output: list[AnimalResult] = []
    for (world_id, method), targets in sorted(grouped.items()):
        if {target.animal_id for target in targets} != proposed_targets[world_id]:
            # A comparator missing any nested target cannot contribute a
            # world-level paired comparison.
            continue
        metrics: dict[str, float | bool] = {}
        names = sorted({str(name) for target in targets for name in target.metrics})
        for name in names:
            values = [_scalar(target.metrics.get(name)) for target in targets]
            # A world-level endpoint exists only if every nested target
            # contributes that endpoint. Partial target metrics cannot create
            # a deceptively complete world.
            if any(value is None for value in values):
                continue
            if all(isinstance(value, bool) for value in values):
                metrics[name] = all(bool(value) for value in values)
            elif all(not isinstance(value, bool) for value in values):
                metrics[name] = float(np.mean([float(value) for value in values]))
        reference = targets[0]
        output.append(
            AnimalResult(
                dataset=reference.dataset,
                cohort=reference.cohort,
                unit_id=world_id,
                animal_id="WORLD_MEAN",
                method=method,
                metrics=metrics,
                source_file=";".join(
                    sorted({target.source_file for target in targets if target.source_file})
                ),
                run_id=world_id,
                world_id=world_id,
                randomized_estimand=all(target.randomized_estimand for target in targets),
            )
        )
    return output


def _primary_values(
    records: Sequence[AnimalResult],
    *,
    domain: str,
    method: str,
) -> dict[str, float]:
    if not records:
        return {}
    aliases = _primary_metric_aliases(records[0].dataset, domain)
    output: dict[str, float] = {}
    for record in sorted(records, key=lambda item: (item.unit_id, item.method)):
        if record.method != method:
            continue
        value = _first_metric(record.metrics, aliases)
        if isinstance(value, bool) or value is None:
            continue
        output[record.unit_id] = float(value)
    return output


def strongest_baseline_envelope(
    records: Sequence[AnimalResult],
    *,
    domain: Literal["neural", "behavior"],
    proposed: str = PROPOSED_METHOD,
) -> list[dict[str, Any]]:
    """Post-outcome per-unit maximum over every emitted eligible non-oracle method."""

    if not records:
        return []
    aliases = _primary_metric_aliases(records[0].dataset, domain)
    by_unit: dict[str, dict[str, AnimalResult]] = defaultdict(dict)
    for record in records:
        by_unit[record.unit_id][record.method] = record
    rows: list[dict[str, Any]] = []
    for unit_id in sorted(by_unit):
        methods = by_unit[unit_id]
        proposed_record = methods.get(proposed)
        if proposed_record is None:
            continue
        proposed_value = _first_metric(proposed_record.metrics, aliases)
        if proposed_value is None or isinstance(proposed_value, bool):
            continue
        candidates: list[tuple[float, str]] = []
        for name, record in methods.items():
            if not eligible_non_oracle_baseline(name, proposed=proposed):
                continue
            value = _first_metric(record.metrics, aliases)
            if value is not None and not isinstance(value, bool):
                candidates.append((float(value), name))
        if not candidates:
            rows.append(
                {
                    "unit_id": unit_id,
                    "animal_id": proposed_record.animal_id,
                    "proposed": float(proposed_value),
                    "baseline": None,
                    "baseline_method": None,
                    "gain": None,
                }
            )
            continue
        baseline, name = max(candidates, key=lambda item: (item[0], item[1]))
        rows.append(
            {
                "unit_id": unit_id,
                "animal_id": proposed_record.animal_id,
                "proposed": float(proposed_value),
                "baseline": baseline,
                "baseline_method": name,
                "gain": float(proposed_value) - baseline,
            }
        )
    return rows


def _summarize_methods(
    records: Sequence[AnimalResult],
    *,
    bootstrap_repeats: int,
    seed: int,
) -> dict[str, Any]:
    values: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    for record in records:
        for metric, raw in record.metrics.items():
            scalar = _scalar(raw)
            if scalar is None or isinstance(scalar, bool):
                continue
            values[(record.method, str(metric))].append((record.unit_id, float(scalar)))
    output: dict[str, dict[str, Any]] = defaultdict(dict)
    for (method, metric), identified_values in sorted(values.items()):
        metric_values = [value for _, value in sorted(identified_values, key=lambda item: item[0])]
        summary = equal_animal_bootstrap_ci(
            metric_values,
            repeats=bootstrap_repeats,
            seed=_stable_seed(seed, "method", method, metric),
        )
        endpoint, endpoint_role = _endpoint_metadata(records[0].dataset, metric)
        summary.update({"endpoint": endpoint, "endpoint_role": endpoint_role})
        output[method][metric] = summary
    return dict(output)


def _combine_statuses(statuses: Iterable[Status]) -> Status:
    values = tuple(statuses)
    if not values:
        return NOT_EVALUATED
    if FAIL in values:
        return FAIL
    if NOT_EVALUATED in values:
        return NOT_EVALUATED
    return PASS


def _status_from_explicit(value: Any) -> Status | None:
    if isinstance(value, Mapping):
        value = value.get("status")
    if isinstance(value, bool):
        return PASS if value else FAIL
    if isinstance(value, str):
        normalized = value.upper()
        if normalized in {PASS, FAIL, NOT_EVALUATED}:
            return normalized  # type: ignore[return-value]
    return None


def _nested_evidence(evidence: Sequence[Mapping[str, Any]], *keys: str) -> list[Any]:
    found: list[Any] = []
    for mapping in evidence:
        current: Any = mapping
        for key in keys:
            if not isinstance(current, Mapping) or key not in current:
                break
            current = current[key]
        else:
            found.append(current)
    return found


def _explicit_component_status(
    evidence: Sequence[Mapping[str, Any]], *paths: tuple[str, ...]
) -> Status | None:
    if not evidence:
        return None
    statuses: list[Status] = []
    matched_any = False
    for mapping in evidence:
        selected: Status | None = None
        for path in paths:
            values = _nested_evidence((mapping,), *path)
            if not values:
                continue
            selected = _status_from_explicit(values[0])
            if selected is not None:
                matched_any = True
                break
        statuses.append(selected or NOT_EVALUATED)
    return _combine_statuses(statuses) if matched_any else None


def _quantitative_evidence(
    evidence: Sequence[Mapping[str, Any]], *paths: tuple[str, ...]
) -> list[Mapping[str, Any]]:
    """Return one quantitative mapping per source, or an empty list if incomplete."""

    if not evidence:
        return []
    output: list[Mapping[str, Any]] = []
    for mapping in evidence:
        selected = next(
            (
                value
                for path in paths
                for value in _nested_evidence((mapping,), *path)
                if isinstance(value, Mapping)
            ),
            None,
        )
        if selected is None:
            return []
        output.append(selected)
    return output


def _metric_vector(
    records: Sequence[AnimalResult],
    aliases: Sequence[str],
    *,
    method: str = PROPOSED_METHOD,
    allow_bool: bool = False,
) -> list[float]:
    values: list[float] = []
    for record in sorted(records, key=lambda item: (item.unit_id, item.method)):
        if record.method != method:
            continue
        value = _first_metric(record.metrics, aliases)
        if value is None or (isinstance(value, bool) and not allow_bool):
            continue
        values.append(float(value))
    return values


def _manipulation_component(
    records: Sequence[AnimalResult],
    evidence: Sequence[Mapping[str, Any]],
    domain: str,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[Status, dict[str, Any]]:
    quantitative = _quantitative_evidence(
        evidence,
        ("manipulation", domain),
        (f"{domain}_manipulation_nonzero",),
    )
    if quantitative:
        statuses: list[Status] = []
        details: list[dict[str, Any]] = []
        for entry in quantitative:
            lower = _scalar(entry.get("ci_lower"))
            upper = _scalar(entry.get("ci_upper"))
            p_value = _scalar(entry.get("p_value"))
            if (
                lower is not None
                and upper is not None
                and not isinstance(lower, bool)
                and not isinstance(upper, bool)
                and float(lower) <= float(upper)
            ):
                passed = float(lower) > 0 or float(upper) < 0
                statuses.append(PASS if passed else FAIL)
                details.append({"ci_lower": float(lower), "ci_upper": float(upper)})
            elif (
                p_value is not None
                and not isinstance(p_value, bool)
                and 0.0 <= float(p_value) <= 1.0
            ):
                statuses.append(PASS if float(p_value) < ALPHA else FAIL)
                details.append({"p_value": float(p_value)})
            else:
                statuses.append(NOT_EVALUATED)
                details.append({"reason": "effect interval or p-value missing"})
        return _combine_statuses(statuses), {
            "source": "explicit_headline_evidence",
            "entries": details,
        }
    dataset = records[0].dataset
    if domain == "neural":
        aliases = (
            "observed_neural_effect",
            "neural_manipulation_effect",
            "neural_effect_size",
        )
    elif dataset == "allen_vbo":
        aliases = (
            "observed_running_effect",
            "running_manipulation_effect",
            "running_effect_size",
        )
    else:
        aliases = (
            "observed_behavior_effect",
            "behavior_manipulation_effect",
            "behavior_effect_size",
            "observed_wheel_effect",
        )
    values = _metric_vector(records, aliases)
    summary = equal_animal_bootstrap_ci(
        values,
        repeats=bootstrap_repeats,
        seed=_stable_seed(seed, "manipulation", domain),
    )
    expected_units = sum(record.method == PROPOSED_METHOD for record in records)
    summary["complete_units"] = len(values)
    summary["expected_units"] = expected_units
    if len(values) != expected_units or summary["ci_lower"] is None:
        return NOT_EVALUATED, summary
    nonzero = summary["ci_lower"] > 0 or summary["ci_upper"] < 0
    return (PASS if nonzero else FAIL), summary


def _skill_component(
    records: Sequence[AnimalResult],
    domain: str,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[Status, dict[str, Any], dict[str, float]]:
    values = _primary_values(records, domain=domain, method=PROPOSED_METHOD)
    summary = equal_animal_bootstrap_ci(
        list(values.values()),
        repeats=bootstrap_repeats,
        seed=_stable_seed(seed, "skill", domain),
    )
    expected_units = {record.unit_id for record in records if record.method == PROPOSED_METHOD}
    summary["complete_units"] = len(values)
    summary["expected_units"] = len(expected_units)
    if len(values) != len(expected_units) or summary["ci_lower"] is None:
        return NOT_EVALUATED, summary, values
    return (PASS if summary["ci_lower"] > 0 else FAIL), summary, values


def _gain_component(
    records: Sequence[AnimalResult],
    domain: str,
    *,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[Status, dict[str, Any], list[dict[str, Any]]]:
    envelope = strongest_baseline_envelope(records, domain=domain)
    complete = [row for row in envelope if row["gain"] is not None]
    proposed_units = {record.unit_id for record in records if record.method == PROPOSED_METHOD}
    dataset = records[0].dataset
    if dataset == "allen_vbo":
        frozen_methods = ALLEN_EXPECTED_METHODS
    elif dataset == "icms":
        frozen_methods = ICMS_EXPECTED_METHODS
    else:
        frozen_methods = frozenset(record.method for record in records)
    expected_baselines = sorted(
        method for method in frozen_methods if eligible_non_oracle_baseline(method)
    )
    aliases = _primary_metric_aliases(dataset, domain)
    by_unit_method = _records_by_unit_method(records)
    incomplete_baselines: dict[str, list[str]] = {}
    for unit_id in sorted(proposed_units):
        missing = [
            method
            for method in expected_baselines
            if (unit_id, method) not in by_unit_method
            or _first_metric(by_unit_method[(unit_id, method)].metrics, aliases) is None
            or isinstance(
                _first_metric(by_unit_method[(unit_id, method)].metrics, aliases),
                bool,
            )
        ]
        if missing:
            incomplete_baselines[unit_id] = missing
    gains = [float(row["gain"]) for row in complete]
    summary = equal_animal_bootstrap_ci(
        gains,
        repeats=bootstrap_repeats,
        seed=_stable_seed(seed, "gain", domain),
    )
    summary["complete_units"] = len(complete)
    summary["expected_units"] = len(proposed_units)
    summary["minimum_mean_gain"] = MINIMUM_BASELINE_GAIN
    summary["required_baseline_methods"] = expected_baselines
    summary["missing_or_nonfinite_baselines_by_unit"] = incomplete_baselines
    if len(complete) != len(proposed_units) or incomplete_baselines or summary["ci_lower"] is None:
        return NOT_EVALUATED, summary, envelope
    passed = summary["estimate"] >= MINIMUM_BASELINE_GAIN and summary["ci_lower"] > 0
    return (PASS if passed else FAIL), summary, envelope


def _proper_score_component(
    records: Sequence[AnimalResult],
) -> tuple[Status, dict[str, Any]]:
    # The current score schemas retain scalar energy scores but no canonical
    # predictive-draw artifact whose bytes are named and bound by the score
    # completion manifest.  Do not let a self-declared 64-hex string stand in
    # for that missing artifact.
    return NOT_EVALUATED, {
        "reason": SUPPLEMENTARY_GATE_ARTIFACT_REASON,
        "required_artifacts": ["full_predictive_draws"],
        "diagnostic_scalar_scores_only": bool(records),
    }


def _randomization_component(
    records: Sequence[AnimalResult],
    evidence: Sequence[Mapping[str, Any]],
    skill_values: Mapping[str, Mapping[str, float]],
    envelope: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[Status, dict[str, Any]]:
    tests: dict[str, Any] = {}
    computed_statuses: list[Status] = []
    expected_units = {record.unit_id for record in records if record.method == PROPOSED_METHOD}
    for domain in ("neural", "behavior"):
        domain_skills = skill_values[domain]
        skills_complete = set(domain_skills) == expected_units and bool(expected_units)
        skills = [
            float(domain_skills[unit_id])
            for unit_id in sorted(expected_units)
            if unit_id in domain_skills
        ]
        gain_by_unit = {
            str(row["unit_id"]): float(row["gain"])
            for row in envelope[domain]
            if row.get("gain") is not None
        }
        gains_complete = set(gain_by_unit) == expected_units and bool(expected_units)
        gains = [
            gain_by_unit[unit_id] for unit_id in sorted(expected_units) if unit_id in gain_by_unit
        ]
        if skills_complete:
            p_skill = exact_paired_sign_flip_test(skills, alternative="greater")
            tests[f"{domain}_skill_greater_than_zero"] = p_skill
            computed_statuses.append(PASS if p_skill < ALPHA else FAIL)
        else:
            tests[f"{domain}_skill_greater_than_zero"] = NOT_EVALUATED
            computed_statuses.append(NOT_EVALUATED)
        if gains_complete:
            p_gain = exact_paired_sign_flip_test(gains, alternative="greater")
            tests[f"{domain}_gain_greater_than_zero"] = p_gain
            computed_statuses.append(PASS if p_gain < ALPHA else FAIL)
        else:
            tests[f"{domain}_gain_greater_than_zero"] = NOT_EVALUATED
            computed_statuses.append(NOT_EVALUATED)

    required_controls = {
        "target_label_permutation": (
            ("randomization_controls", "target_label_permutation"),
            ("target_label_permutation",),
        ),
        "donor_semantic_shuffle": (
            ("randomization_controls", "donor_semantic_shuffle"),
            ("donor_semantic_shuffle",),
        ),
        "animal_adapter_shuffle": (
            ("randomization_controls", "animal_adapter_shuffle"),
            ("animal_adapter_shuffle",),
        ),
    }
    proposed_records = [record for record in records if record.method == PROPOSED_METHOD]
    control_statuses: list[Status] = []
    for name, paths in required_controls.items():
        entries = _quantitative_evidence(evidence, *paths)
        metric_claims = sum(
            isinstance(
                _first_metric(
                    record.metrics,
                    (f"{name}_passed", f"{name}_null_rejected"),
                ),
                bool,
            )
            for record in proposed_records
        )
        tests[name] = NOT_EVALUATED
        tests[f"{name}_artifact_audit"] = {
            "reason": SUPPLEMENTARY_GATE_ARTIFACT_REASON,
            "headline_claims_present": len(entries),
            "unit_metric_claims_present": metric_claims,
        }
        control_statuses.append(NOT_EVALUATED)
    return _combine_statuses([*computed_statuses, *control_statuses]), tests


def _coverage_artifact_audit(
    value: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Validate the retained draws and band summaries promised by Gate 7."""

    def field(name: str) -> Any:
        return value.get(f"{prefix}{name}")

    draw_count = field("predictive_draw_count")
    nominal = _scalar(field("simultaneous_band_nominal_level"))
    width = _scalar(field("simultaneous_band_mean_width"))
    pointwise = _scalar(field("pointwise_coverage"))
    complete = False
    return {
        "complete": complete,
        "reason": SUPPLEMENTARY_GATE_ARTIFACT_REASON,
        "predictive_draw_count": draw_count,
        "predictive_draw_protocol": field("predictive_draw_protocol"),
        "predictive_draws_sha256": field("predictive_draws_sha256"),
        "simultaneous_band_nominal_level": nominal,
        "simultaneous_band_mean_width": width,
        "pointwise_coverage": pointwise,
        "simultaneous_band_sha256": field("simultaneous_band_sha256"),
    }


def _coverage_component(
    records: Sequence[AnimalResult],
    evidence: Sequence[Mapping[str, Any]],
) -> tuple[Status, dict[str, Any]]:
    summaries: dict[str, Any] = {}
    statuses: list[Status] = []
    dataset = records[0].dataset
    proposed_records = [record for record in records if record.method == PROPOSED_METHOD]
    expected_units = len(proposed_records)
    aliases = {
        "neural": (
            "neural_calibrated_simultaneous_coverage",
            "neural_split_conformal_simultaneous_coverage",
        ),
        "behavior": (
            (
                "running_calibrated_simultaneous_coverage",
                "running_split_conformal_simultaneous_coverage",
            )
            if dataset == "allen_vbo"
            else (
                "behavior_calibrated_simultaneous_coverage",
                "wheel_split_conformal_simultaneous_coverage",
            )
        ),
    }
    for domain, names in aliases.items():
        quantitative = _quantitative_evidence(
            evidence,
            ("coverage", domain),
            (f"{domain}_simultaneous_coverage",),
        )
        if quantitative:
            successes = 0
            trials = 0
            provenance_complete = True
            counts_complete = True
            artifacts_complete = True
            artifact_audits: list[dict[str, Any]] = []
            for entry in quantitative:
                calibrated = bool(
                    entry.get("calibration_method") == "split_conformal_max_standardized_residual"
                    and entry.get("calibration_scope") == DISJOINT_CALIBRATION_SCOPE
                )
                provenance_complete &= calibrated
                entry_successes = _scalar(entry.get("successes"))
                entry_trials = _scalar(entry.get("n", entry.get("trials")))
                valid_counts = (
                    entry_successes is not None
                    and entry_trials is not None
                    and not isinstance(entry_successes, bool)
                    and not isinstance(entry_trials, bool)
                    and float(entry_successes).is_integer()
                    and float(entry_trials).is_integer()
                    and 0 <= int(entry_successes) <= int(entry_trials)
                    and int(entry_trials) > 0
                )
                counts_complete &= valid_counts
                artifact_audit = _coverage_artifact_audit(entry)
                artifact_audits.append(artifact_audit)
                artifacts_complete &= bool(artifact_audit["complete"])
                if valid_counts:
                    successes += int(entry_successes)
                    trials += int(entry_trials)
            complete = bool(
                provenance_complete
                and counts_complete
                and artifacts_complete
                and trials == expected_units
                and expected_units > 0
            )
            lower = exact_binomial_lower_confidence_bound(successes, trials) if complete else None
            summary = {
                "source": "explicit_headline_evidence",
                "n": trials,
                "successes": successes,
                "estimate": successes / trials if trials else None,
                "ci_lower": lower,
                "ci_upper": 1.0 if lower is not None else None,
                "confidence": DEFAULT_CONFIDENCE,
                "interval_method": "clopper_pearson_exact_one_sided",
                "replication_unit": "target_animal",
                "complete_units": trials,
                "expected_units": expected_units,
                "binary_counts_complete": counts_complete,
                "calibration_provenance_complete": provenance_complete,
                "predictive_draw_and_band_artifacts_complete": artifacts_complete,
                "predictive_draw_and_band_artifacts": artifact_audits,
                "required_lower_bound": SIMULTANEOUS_COVERAGE_LOWER_BOUND,
            }
            summaries[domain] = summary
            if lower is None:
                statuses.append(NOT_EVALUATED)
            else:
                statuses.append(PASS if lower >= SIMULTANEOUS_COVERAGE_LOWER_BOUND else FAIL)
            continue
        values = _metric_vector(records, names, allow_bool=True)
        provenance_complete = all(
            record.metrics.get("coverage_calibration_method")
            == "split_conformal_max_standardized_residual"
            and record.metrics.get("coverage_calibration_scope") == DISJOINT_CALIBRATION_SCOPE
            for record in records
            if record.method == PROPOSED_METHOD
        )
        behavior_prefix = "running" if dataset == "allen_vbo" else "behavior"
        metric_prefix = f"{'neural' if domain == 'neural' else behavior_prefix}_"
        artifact_audits = [
            {
                "unit_id": record.unit_id,
                **_coverage_artifact_audit(
                    record.metrics,
                    prefix=metric_prefix,
                ),
            }
            for record in sorted(proposed_records, key=lambda item: item.unit_id)
        ]
        artifacts_complete = bool(
            len(artifact_audits) == expected_units
            and all(audit["complete"] for audit in artifact_audits)
        )
        coverage = np.asarray(values, dtype=np.float64)
        binary_complete = bool(
            len(values) == expected_units
            and expected_units > 0
            and np.all((coverage == 0.0) | (coverage == 1.0))
        )
        successes = int(coverage.sum()) if binary_complete else None
        lower = (
            exact_binomial_lower_confidence_bound(successes, expected_units)
            if successes is not None and provenance_complete and artifacts_complete
            else None
        )
        summary = {
            "n": expected_units if binary_complete else len(values),
            "successes": successes,
            "estimate": successes / expected_units if successes is not None else None,
            "ci_lower": lower,
            "ci_upper": 1.0 if lower is not None else None,
            "confidence": DEFAULT_CONFIDENCE,
            "interval_method": "clopper_pearson_exact_one_sided",
            "replication_unit": "target_animal",
            "required_lower_bound": SIMULTANEOUS_COVERAGE_LOWER_BOUND,
            "complete_units": len(values),
            "expected_units": expected_units,
            "binary_values_complete": binary_complete,
            "calibration_provenance_complete": provenance_complete,
            "predictive_draw_and_band_artifacts_complete": artifacts_complete,
            "predictive_draw_and_band_artifacts": artifact_audits,
        }
        summaries[domain] = summary
        if lower is None:
            statuses.append(NOT_EVALUATED)
        else:
            statuses.append(PASS if lower >= SIMULTANEOUS_COVERAGE_LOWER_BOUND else FAIL)
    return _combine_statuses(statuses), summaries


def _negative_control_component(
    records: Sequence[AnimalResult],
    evidence: Sequence[Mapping[str, Any]],
) -> tuple[Status, dict[str, Any]]:
    def authenticated_null_status(value: Mapping[str, Any]) -> Status:
        del value
        return NOT_EVALUATED

    statuses: dict[str, Status] = {}
    proposed_records = [record for record in records if record.method == PROPOSED_METHOD]
    for name in ("pre_onset", "pseudo_onset"):
        explicit_entries = _quantitative_evidence(
            evidence,
            ("negative_controls", name),
            (name,),
        )
        if explicit_entries:
            statuses[name] = _combine_statuses(
                authenticated_null_status(entry) for entry in explicit_entries
            )
            continue
        unit_statuses: list[Status] = []
        for record in proposed_records:
            global_value = _first_metric(record.metrics, (f"{name}_null_passed",))
            global_entry = {
                "status": (
                    PASS if global_value is True else FAIL if global_value is False else None
                ),
                "protocol": record.metrics.get(f"{name}_null_protocol"),
                "artifact_sha256": record.metrics.get(f"{name}_null_artifact_sha256"),
                "equivalence_margin": record.metrics.get(f"{name}_null_equivalence_margin"),
                "ci_lower": record.metrics.get(f"{name}_null_ci_lower"),
                "ci_upper": record.metrics.get(f"{name}_null_ci_upper"),
            }
            global_status = authenticated_null_status(global_entry)
            if global_status != NOT_EVALUATED:
                unit_statuses.append(global_status)
                continue
            unit_statuses.append(NOT_EVALUATED)
        statuses[name] = _combine_statuses(unit_statuses)
    return _combine_statuses(statuses.values()), {
        **statuses,
        "artifact_audit": {
            "reason": SUPPLEMENTARY_GATE_ARTIFACT_REASON,
            "required_artifacts": ["pre_onset_equivalence", "pseudo_onset_equivalence"],
        },
    }


def _conjunction(
    records: Sequence[AnimalResult],
    evidence: Sequence[Mapping[str, Any]],
    *,
    bootstrap_repeats: int,
    seed: int,
    ineligible_reason: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if ineligible_reason is not None:
        rows = [
            {
                "gate_id": index,
                "criterion": criterion,
                "status": NOT_EVALUATED,
                "details": {"reason": ineligible_reason},
            }
            for index, criterion in _HEADLINE_CRITERIA
        ]
        return (
            {"overall_status": NOT_EVALUATED, "gates": rows},
            {"neural": [], "behavior": []},
            {},
        )

    neural_effect, neural_effect_detail = _manipulation_component(
        records,
        evidence,
        "neural",
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )
    behavior_effect, behavior_effect_detail = _manipulation_component(
        records,
        evidence,
        "behavior",
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
    )
    neural_skill, neural_skill_summary, neural_values = _skill_component(
        records, "neural", bootstrap_repeats=bootstrap_repeats, seed=seed
    )
    behavior_skill, behavior_skill_summary, behavior_values = _skill_component(
        records, "behavior", bootstrap_repeats=bootstrap_repeats, seed=seed
    )
    neural_gain, neural_gain_summary, neural_envelope = _gain_component(
        records, "neural", bootstrap_repeats=bootstrap_repeats, seed=seed
    )
    behavior_gain, behavior_gain_summary, behavior_envelope = _gain_component(
        records, "behavior", bootstrap_repeats=bootstrap_repeats, seed=seed
    )
    proper_status, proper_detail = _proper_score_component(records)
    randomization_status, randomization_detail = _randomization_component(
        records,
        evidence,
        {"neural": neural_values, "behavior": behavior_values},
        {"neural": neural_envelope, "behavior": behavior_envelope},
    )
    coverage_status, coverage_detail = _coverage_component(records, evidence)
    negative_status, negative_detail = _negative_control_component(records, evidence)
    rows = [
        {
            "gate_id": 1,
            "criterion": _HEADLINE_CRITERIA[0][1],
            "status": _combine_statuses((neural_effect, behavior_effect)),
            "details": {
                "neural": neural_effect_detail,
                "behavior": behavior_effect_detail,
            },
        },
        {
            "gate_id": 2,
            "criterion": _HEADLINE_CRITERIA[1][1],
            "status": neural_skill,
            "details": neural_skill_summary,
        },
        {
            "gate_id": 3,
            "criterion": _HEADLINE_CRITERIA[2][1],
            "status": behavior_skill,
            "details": behavior_skill_summary,
        },
        {
            "gate_id": 4,
            "criterion": _HEADLINE_CRITERIA[3][1],
            "status": _combine_statuses((neural_gain, behavior_gain)),
            "details": {
                "neural": neural_gain_summary,
                "behavior": behavior_gain_summary,
            },
        },
        {
            "gate_id": 5,
            "criterion": _HEADLINE_CRITERIA[4][1],
            "status": proper_status,
            "details": proper_detail,
        },
        {
            "gate_id": 6,
            "criterion": _HEADLINE_CRITERIA[5][1],
            "status": randomization_status,
            "details": randomization_detail,
        },
        {
            "gate_id": 7,
            "criterion": _HEADLINE_CRITERIA[6][1],
            "status": coverage_status,
            "details": coverage_detail,
        },
        {
            "gate_id": 8,
            "criterion": _HEADLINE_CRITERIA[7][1],
            "status": negative_status,
            "details": negative_detail,
        },
    ]
    return (
        {
            "overall_status": _combine_statuses(
                row["status"]
                for row in rows  # type: ignore[arg-type]
            ),
            "gates": rows,
        },
        {"neural": neural_envelope, "behavior": behavior_envelope},
        {
            "neural_skill": neural_skill_summary,
            "behavior_skill": behavior_skill_summary,
            "neural_baseline_gain": neural_gain_summary,
            "behavior_baseline_gain": behavior_gain_summary,
        },
    )


_HEADLINE_CRITERIA: tuple[tuple[int, str], ...] = (
    (1, "randomized manipulation has nonzero neural and primary behavioral effects"),
    (2, "equal-animal 95% CI for neural causal skill is above zero"),
    (3, "equal-animal 95% CI for primary behavioral causal skill is above zero"),
    (
        4,
        "mean gain over the per-target post-outcome envelope of all eligible "
        "non-oracle comparators and ablations is at least 0.10 and its 95% CI "
        "is above zero for neural and primary behavior endpoints",
    ),
    (
        5,
        "state-conditioned prediction improves proper score over the donor condition-time template",
    ),
    (6, "animal-level exact randomization tests reject every relevant null"),
    (
        7,
        "90% simultaneous-band coverage has a 95% lower bound of at least 0.80",
    ),
    (8, "pre-onset and pseudo-onset negative controls pass explicit null checks"),
)


def aggregate_batches(
    batches: Sequence[AdaptedBatch],
    *,
    bootstrap_repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Aggregate adapted payloads into the complete machine-readable report."""

    records = [record for batch in batches for record in batch.records]
    if not records:
        raise ValueError("no animal-level records were supplied")
    if bootstrap_repeats < 1:
        raise ValueError("bootstrap_repeats must be positive")
    evidence_by_analysis: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    validation_by_analysis: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for batch in batches:
        analyses = {f"{record.dataset}:{record.cohort}" for record in batch.records}
        for analysis in analyses:
            evidence_by_analysis[analysis].append(batch.headline_evidence)
            validation_by_analysis[analysis].append(batch.artifact_validation)

    grouped: dict[str, list[AnimalResult]] = defaultdict(list)
    for record in records:
        grouped[f"{record.dataset}:{record.cohort}"].append(record)

    analyses: dict[str, Any] = {}
    all_animal_rows: list[dict[str, Any]] = []
    reporter_attestations: dict[str, Mapping[str, Any]] = {}
    for analysis_id, group in sorted(grouped.items()):
        _records_by_unit_method(group)
        dataset = group[0].dataset
        cohort = group[0].cohort
        target_unit_ids = sorted({record.unit_id for record in group})
        inference_group = (
            _teacher_world_average_records(group) if dataset == "teacher" else list(group)
        )
        _records_by_unit_method(inference_group)
        inference_unit_ids = sorted({record.unit_id for record in inference_group})
        proposed_records = [
            record for record in inference_group if record.method == PROPOSED_METHOD
        ]
        completeness = _cohort_completeness(group, validation_by_analysis.get(analysis_id, ()))
        locked_scope = _mapping_or_empty(completeness.get("locked_scope_validation"))
        canonical_locked_complete = (
            completeness.get("complete") is True
            and locked_scope.get("valid") is True
            and (cohort == "locked" or (dataset == "icms" and cohort == "randomized_n5"))
        )
        if canonical_locked_complete and (
            bootstrap_repeats != DEFAULT_BOOTSTRAP_REPEATS or seed != DEFAULT_SEED
        ):
            raise ValueError(
                "complete frozen aggregation requires the preregistered "
                f"bootstrap_repeats={DEFAULT_BOOTSTRAP_REPEATS} and seed={DEFAULT_SEED}"
            )
        reporter_attestation: Mapping[str, Any] | None = None
        if canonical_locked_complete:
            freeze_attestation = _freeze_attestation(locked_scope.get("freeze_attestation"))
            reporter_attestation = _git_verify_clean_reporter_state(freeze_attestation)
            reporter_attestations[analysis_id] = reporter_attestation
        ineligible_reason = None if completeness["gates_evaluable"] else str(completeness["reason"])
        conjunction, envelope, primary = _conjunction(
            inference_group,
            evidence_by_analysis.get(analysis_id, ()),
            bootstrap_repeats=bootstrap_repeats,
            seed=_stable_seed(seed, analysis_id),
            ineligible_reason=ineligible_reason,
        )
        method_summaries = _summarize_methods(
            inference_group,
            bootstrap_repeats=bootstrap_repeats,
            seed=_stable_seed(seed, analysis_id, "methods"),
        )
        animal_rows: list[dict[str, Any]] = []
        for record in sorted(group, key=lambda item: (item.unit_id, item.method)):
            row = {
                "dataset": record.dataset,
                "cohort": record.cohort,
                "unit_id": record.unit_id,
                "animal_id": record.animal_id,
                "method": record.method,
                "method_role": method_role(record.method),
                "baseline_eligible": eligible_non_oracle_baseline(record.method),
                "randomized_estimand": record.randomized_estimand,
                "fold": record.fold,
                "world_id": record.world_id,
                "run_id": record.run_id,
                "source_file": record.source_file,
                "metrics": _jsonable(dict(record.metrics)),
            }
            animal_rows.append(row)
            all_animal_rows.append(row)
        target_rows: list[dict[str, Any]] = []
        for unit_id in target_unit_ids:
            unit_records = [record for record in group if record.unit_id == unit_id]
            reference = unit_records[0]
            target_rows.append(
                {
                    "dataset": reference.dataset,
                    "cohort": reference.cohort,
                    "unit_id": reference.unit_id,
                    "animal_id": reference.animal_id,
                    "fold": reference.fold,
                    "world_id": reference.world_id,
                    "randomized_estimand": reference.randomized_estimand,
                    "methods": {
                        record.method: _jsonable(dict(record.metrics))
                        for record in sorted(unit_records, key=lambda item: item.method)
                    },
                }
            )
        inference_rows = [
            {
                "dataset": record.dataset,
                "cohort": record.cohort,
                "unit_id": record.unit_id,
                "animal_id": record.animal_id,
                "method": record.method,
                "world_id": record.world_id,
                "metrics": _jsonable(dict(record.metrics)),
            }
            for record in sorted(inference_group, key=lambda item: (item.unit_id, item.method))
        ]
        analyses[analysis_id] = {
            "dataset": dataset,
            "cohort": cohort,
            "n_independent_units": len(inference_unit_ids),
            "n_nested_target_units": len(target_unit_ids),
            "expected_n": completeness["expected_units"],
            "cohort_completeness": completeness,
            "reporter_attestation": reporter_attestation,
            "replication_unit": ("teacher_world" if dataset == "teacher" else "target_animal"),
            "endpoint_hierarchy": _endpoint_hierarchy(dataset, cohort),
            "method_summaries": method_summaries,
            "primary_summaries": primary,
            "strongest_baseline_envelope": envelope,
            "conjunction": conjunction,
            # This is the transparent target matrix: one row per target mouse,
            # or per nested world/target in the procedural benchmark.
            "target_rows": target_rows,
            "inference_rows": inference_rows,
            "animal_rows": animal_rows,
            "n_proposed_units": len({record.unit_id for record in proposed_records}),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "parameters": {
            "bootstrap_repeats": bootstrap_repeats,
            "bootstrap_confidence": DEFAULT_CONFIDENCE,
            "seed": seed,
            "minimum_baseline_gain": MINIMUM_BASELINE_GAIN,
            "simultaneous_coverage_lower_bound": SIMULTANEOUS_COVERAGE_LOWER_BOUND,
            "sign_flip": "exact exhaustive meet-in-the-middle",
            "baseline_envelope": "per-unit maximum over every non-oracle method",
        },
        "reporter_attestations": reporter_attestations,
        "analyses": analyses,
        "animal_rows": all_animal_rows,
    }


def _endpoint_hierarchy(dataset: str, cohort: str) -> dict[str, Any]:
    if dataset == "allen_vbo":
        return {
            "neural_primary": "event_rate",
            "behavior_primary": "running_speed",
            "behavior_secondary": ["pupil_area", "lick_rate"],
            "secondary_cannot_rescue_primary": True,
        }
    if dataset == "icms":
        return {
            "neural_primary": "accepted_sorted_spike_rate",
            "behavior_primary": ["wheel_displacement", "wheel_velocity"],
            "randomized_causal_estimand": cohort == "randomized_n5",
            "ICMS83": "absolute_only",
        }
    return {
        "observed_primary": [
            "neural_condition_averaged_causal_skill",
            "behavior_condition_averaged_causal_skill",
        ],
        "realized_path_diagnostics": {
            "metrics": [
                "neural_pathwise_mean_causal_skill",
                "behavior_pathwise_mean_causal_skill",
            ],
            "role": (
                "synthetic-only localization conditional on the realized future "
                "latent/process path; not a forecastable endpoint"
            ),
        },
    }


def load_and_adapt(
    paths: Sequence[str | Path],
    *,
    kind: Literal["auto", "allen", "teacher", "icms"] = "auto",
) -> list[AdaptedBatch]:
    batches: list[AdaptedBatch] = []
    for path_value in paths:
        path = Path(path_value)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path} does not contain a JSON object")
        batches.append(adapt_payload(payload, kind=kind, source_file=path))
    return batches


def _long_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for animal in report["animal_rows"]:
        for metric, raw_value in animal["metrics"].items():
            endpoint, endpoint_role = _endpoint_metadata(animal["dataset"], metric)
            scalar = _scalar(raw_value)
            if scalar is not None:
                indexed_values: list[tuple[int | None, float | bool]] = [(None, scalar)]
            elif isinstance(raw_value, list):
                try:
                    flattened = np.asarray(raw_value, dtype=np.float64).reshape(-1)
                except (TypeError, ValueError):
                    continue
                indexed_values = [
                    (index, float(value))
                    for index, value in enumerate(flattened)
                    if math.isfinite(float(value))
                ]
            else:
                continue
            for metric_index, value in indexed_values:
                rows.append(
                    {
                        "dataset": animal["dataset"],
                        "cohort": animal["cohort"],
                        "unit_id": animal["unit_id"],
                        "animal_id": animal["animal_id"],
                        "fold": animal["fold"],
                        "world_id": animal["world_id"],
                        "randomized_estimand": animal["randomized_estimand"],
                        "method": animal["method"],
                        "method_role": animal["method_role"],
                        "baseline_eligible": animal["baseline_eligible"],
                        "endpoint": endpoint,
                        "endpoint_role": endpoint_role,
                        "metric": metric,
                        "metric_index": metric_index,
                        "value": value,
                        "source_file": animal["source_file"],
                    }
                )
    return rows


def _format_ci(summary: Mapping[str, Any] | None) -> str:
    if not summary or summary.get("estimate") is None:
        return "--"
    estimate = float(summary["estimate"])
    lower = summary.get("ci_lower")
    upper = summary.get("ci_upper")
    if lower is None or upper is None:
        return f"{estimate:.3f}"
    return f"{estimate:.3f} [{float(lower):.3f}, {float(upper):.3f}]"


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def render_latex_summary(report: Mapping[str, Any]) -> str:
    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Analysis & Method & Neural causal skill & Primary behavior causal skill \\",
        r"\midrule",
    ]
    for analysis_id, analysis in report["analyses"].items():
        neural_aliases = _primary_metric_aliases(analysis["dataset"], "neural")
        behavior_aliases = _primary_metric_aliases(analysis["dataset"], "behavior")
        methods = analysis["method_summaries"]
        ordered = [PROPOSED_METHOD] if PROPOSED_METHOD in methods else []
        ordered.extend(sorted(method for method in methods if method != PROPOSED_METHOD))
        for method in ordered:
            neural = next(
                (
                    methods[method].get(alias)
                    for alias in neural_aliases
                    if alias in methods[method]
                ),
                None,
            )
            behavior = next(
                (
                    methods[method].get(alias)
                    for alias in behavior_aliases
                    if alias in methods[method]
                ),
                None,
            )
            lines.append(
                f"{_latex_escape(analysis_id)} & {_latex_escape(method)} & "
                f"{_latex_escape(_format_ci(neural))} & "
                f"{_latex_escape(_format_ci(behavior))} \\\\"
            )
        lines.append(r"\addlinespace")
    if lines[-1] == r"\addlinespace":
        lines.pop()
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _conjunction_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for analysis_id, analysis in report["analyses"].items():
        for gate in analysis["conjunction"]["gates"]:
            rows.append(
                {
                    "analysis": analysis_id,
                    "gate_id": gate["gate_id"],
                    "status": gate["status"],
                    "criterion": gate["criterion"],
                    "details_json": json.dumps(
                        _jsonable(gate["details"]), sort_keys=True, separators=(",", ":")
                    ),
                }
            )
    return rows


def render_latex_conjunction(report: Mapping[str, Any]) -> str:
    lines = [
        r"\begin{tabular}{lllp{8.5cm}}",
        r"\toprule",
        r"Analysis & Gate & Status & Frozen criterion \\",
        r"\midrule",
    ]
    for row in _conjunction_rows(report):
        lines.append(
            f"{_latex_escape(row['analysis'])} & {row['gate_id']} & "
            f"{_latex_escape(row['status'])} & "
            f"{_latex_escape(row['criterion'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _csv_text(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def write_report(
    report: Mapping[str, Any],
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Atomically publish an append-only, self-authenticating report directory."""

    if overwrite:
        raise ValueError("report overwrite is disabled; choose a new output directory")
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to replace existing report directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    staging_paths = {
        "summary": staging / "summary.json",
        "long_csv": staging / "metrics_long.csv",
        "latex_summary": staging / "summary_table.tex",
        "conjunction_csv": staging / "conjunction_table.csv",
        "latex_conjunction": staging / "conjunction_table.tex",
    }
    long_rows = _long_rows(report)
    conjunction_rows = _conjunction_rows(report)
    long_fields = (
        "dataset",
        "cohort",
        "unit_id",
        "animal_id",
        "fold",
        "world_id",
        "randomized_estimand",
        "method",
        "method_role",
        "baseline_eligible",
        "endpoint",
        "endpoint_role",
        "metric",
        "metric_index",
        "value",
        "source_file",
    )
    conjunction_fields = (
        "analysis",
        "gate_id",
        "status",
        "criterion",
        "details_json",
    )
    try:
        staging_paths["summary"].write_text(
            json.dumps(_jsonable(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staging_paths["long_csv"].write_text(
            _csv_text(long_rows, long_fields),
            encoding="utf-8",
        )
        staging_paths["latex_summary"].write_text(
            render_latex_summary(report),
            encoding="utf-8",
        )
        staging_paths["conjunction_csv"].write_text(
            _csv_text(conjunction_rows, conjunction_fields),
            encoding="utf-8",
        )
        staging_paths["latex_conjunction"].write_text(
            render_latex_conjunction(report),
            encoding="utf-8",
        )
        artifact_hashes: dict[str, str] = {}
        for path in staging_paths.values():
            digest = _sha256_file(path)
            artifact_hashes[path.name] = digest
            path.with_suffix(path.suffix + ".sha256").write_text(
                f"{digest}  {path.name}\n",
                encoding="utf-8",
            )
        authenticated_inputs: list[Mapping[str, Any]] = []
        for analysis in _mapping_or_empty(report.get("analyses")).values():
            completeness = _mapping_or_empty(_mapping_or_empty(analysis).get("cohort_completeness"))
            locked_scope = _mapping_or_empty(completeness.get("locked_scope_validation"))
            authenticated_inputs.extend(
                item
                for item in locked_scope.get("authenticated_inputs", ())
                if isinstance(item, Mapping)
            )
        completion = {
            "schema": "cadence.reporting_completion.v1",
            "append_only": True,
            "artifacts": artifact_hashes,
            "authenticated_inputs": sorted(
                (
                    {
                        "source_file": str(item.get("source_file", "")),
                        "sha256": str(item.get("sha256", "")),
                    }
                    for item in authenticated_inputs
                ),
                key=lambda item: (item["source_file"], item["sha256"]),
            ),
            "reporter_attestations": _mapping_or_empty(report.get("reporter_attestations")),
        }
        completion_path = staging / "report.complete.json"
        completion_path.write_text(
            json.dumps(completion, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        completion_sha = _sha256_file(completion_path)
        completion_path.with_suffix(".json.sha256").write_text(
            f"{completion_sha}  {completion_path.name}\n",
            encoding="utf-8",
        )
        for path in staging.iterdir():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        directory_descriptor = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        staging.rename(output)
        parent_descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    paths = {name: output / path.name for name, path in staging_paths.items()}
    paths["completion"] = output / "report.complete.json"
    return paths


__all__ = [
    "ALPHA",
    "ALLEN_EXPECTED_METHODS",
    "ALLEN_LOCKED_ANIMALS",
    "AdaptedBatch",
    "AnimalResult",
    "DEFAULT_BOOTSTRAP_REPEATS",
    "DEFAULT_SEED",
    "FAIL",
    "ICMS_EXPECTED_METHODS",
    "ICMS_ABSOLUTE_ONLY_ANIMAL",
    "ICMS_RANDOMIZED_ANIMALS",
    "MINIMUM_BASELINE_GAIN",
    "LOCK_SEED",
    "NOT_EVALUATED",
    "PASS",
    "PREOUTCOME_TAG",
    "SCHEMA_VERSION",
    "SIMULTANEOUS_COVERAGE_LOWER_BOUND",
    "TEACHER_LOCKED_TARGET_UNITS",
    "TEACHER_LOCKED_WORLDS",
    "TEACHER_EXPECTED_METHODS",
    "TEACHER_LEARNED_METHODS",
    "TEACHER_TARGETS_PER_WORLD",
    "adapt_allen_payload",
    "adapt_icms_payload",
    "adapt_payload",
    "adapt_teacher_payload",
    "aggregate_batches",
    "eligible_non_oracle_baseline",
    "equal_animal_bootstrap_ci",
    "exact_binomial_lower_confidence_bound",
    "exact_paired_sign_flip_test",
    "load_and_adapt",
    "method_role",
    "render_latex_conjunction",
    "render_latex_summary",
    "strongest_baseline_envelope",
    "write_report",
]
