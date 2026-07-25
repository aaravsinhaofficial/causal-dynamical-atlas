"""Leakage-sealed leave-one-animal-out experiment for DANDI:001868.

The processed DANDI files deliberately keep all trials for one animal in a
single container.  This module therefore exposes three separate commands:

``prepare``
    Read only zero-current catch and signal-blind guarded-ITI rows.  Build
    session-specific normal supports and a prespecified physical query lattice.
``predict``
    After the tagged pre-outcome freeze, open stimulation responses from the
    five donor animals, fit the operator, adapt each target-session observation
    map using its normal support only, and hash predictions.
``score``
    After an explicit second acknowledgement, verify every earlier hash and
    only then open target stimulation rows.

Sorted-unit identities are session-specific.  Every session consequently has
its own observation adapter while donor intervention random effects are grouped
at the animal level.  Target stimulation metadata are not used to choose the
query: predictions cover a frozen NET32 depth/current lattice and are
interpolated to observed target currents only after the prediction bundle has
been hashed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from cadence.baselines import (
    AdditiveInterventionSSM,
    BlackBoxMetaGRU,
    LinearHierarchicalSSM,
)
from cadence.data.dandi_icms import (
    DANDISET_ID,
    DANDISET_VERSION,
    DEFAULT_PREPROCESS_CONFIG,
    INTERVENTION_DESCRIPTOR_COLUMNS,
    NET32_DEPTH_CENTER_UM,
    NET32_DEPTH_HALF_RANGE_UM,
    TASK_MICE,
    load_frozen_manifest,
)
from cadence.metrics import (
    causal_skill,
    energy_score,
    interval_coverage,
    support_scale,
    trajectory_nrmse,
)
from cadence.model import HierarchicalControlledSSM, SequenceBatch
from cadence.protocol import ProtocolViolation, attest_preoutcome_freeze
from cadence.training import (
    EpochRecord,
    FitConfig,
    FitResult,
    fit_stage,
    move_batch,
    seed_everything,
)
from cadence.uncertainty import sample_target_intervention_residual

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
MethodName = Literal[
    "proposed",
    "linear",
    "additive",
    "black_box",
    "zero_effect",
    "condition_time",
    "nearest_donor",
]
RunMode = Literal["biological", "synthetic"]

LEARNED_METHODS: tuple[MethodName, ...] = (
    "proposed",
    "linear",
    "additive",
    "black_box",
)
REPORT_METHODS: tuple[MethodName, ...] = (
    *LEARNED_METHODS,
    "zero_effect",
    "condition_time",
    "nearest_donor",
)
BEHAVIOR_DIM = 2
INPUT_DIM = 1
DESCRIPTOR_DIM = len(INTERVENTION_DESCRIPTOR_COLUMNS)
PREOUTCOME_TAG = "pre-outcome-v1.0.0"
LOCK_SEED = 20260725
SOURCE_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_INDEX_RELATIVE = Path("data/processed/dandi_001868/index.json")
CANONICAL_RAW_MANIFEST_RELATIVE = Path("configs/dandi_001868_assets.json")
CANONICAL_INDEX_TOTALS = {
    "sessions": 45,
    "normal_calibration_trials": 2332,
    "catch_normal_calibration_trials": 1400,
    "iti_calibration_windows": 932,
    "stimulation_trials": 16640,
}
ACTIVE_SEAL_NAME = ".cadence-icms-active-target-seal.json"
ICMS_RESTORE_COMPLETION_SCHEMA = "cadence-icms-target-restore-completion-v1"
CANONICAL_PULSE_COUNT = 70.0
CANONICAL_FREQUENCY_HZ = 100.0
CANONICAL_PULSE_WIDTH_US = 167.0
CANONICAL_TRAIN_DURATION_S = 0.7
CANONICAL_DURATION_TOLERANCE_S = 0.01
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_FORBIDDEN_QUERY_TOKENS = (
    "target_stimulation",
    "stim_neural",
    "stim_behavior",
    "is_hit",
    "response_time",
    "observed_outcome",
)


def _canonical_icms_relative_output(target_animal: str) -> Path:
    if target_animal not in TASK_MICE:
        raise ValueError(f"target must be one of {TASK_MICE}")
    return Path("results") / "icms" / f"loao-{target_animal}"


def _require_canonical_biological_output(path: Path, target_animal: str) -> str:
    """Bind biological stages to one nonsymlink target-specific output."""

    relative = _canonical_icms_relative_output(target_animal)
    expected = (SOURCE_ROOT / relative).absolute()
    observed = path.absolute()
    if observed != expected:
        raise ProtocolViolation(
            f"biological ICMS output must be the canonical one-shot path {expected}; "
            f"observed {observed}"
        )
    cursor = SOURCE_ROOT
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ProtocolViolation("biological ICMS output may not traverse a symlink")
    return relative.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _safe_key(value: str) -> str:
    return value.replace("-", "_").replace("/", "_").replace(".", "_").replace(":", "_")


def _stable_digest(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def _artifact_paths_with_sidecars(paths: Iterable[Path]) -> tuple[Path, ...]:
    return tuple(candidate for path in paths for candidate in (path, _hash_sidecar(path)))


def _temporary_output_path(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    return Path(name)


def _publish_without_replace(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
        _fsync_directory(destination.parent)
    except FileExistsError as error:
        raise FileExistsError(f"append-only artifact already exists: {destination}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _create_exclusive_text(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
    except BaseException:
        path.unlink(missing_ok=True)
        _fsync_directory(path.parent)
        raise


def _write_hash_sidecar(path: Path, digest: str) -> None:
    _create_exclusive_text(
        _hash_sidecar(path),
        f"{digest}  {path.name}\n",
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or _hash_sidecar(path).exists():
        raise FileExistsError(f"append-only JSON artifact already exists: {path}")
    temporary = _temporary_output_path(path)
    text = json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        _publish_without_replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    digest = hash_file(path)
    _write_hash_sidecar(path, digest)
    return digest


def _atomic_npz(path: Path, **arrays: npt.ArrayLike) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or _hash_sidecar(path).exists():
        raise FileExistsError(f"append-only NPZ artifact already exists: {path}")
    temporary = _temporary_output_path(path)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        _publish_without_replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    digest = hash_file(path)
    _write_hash_sidecar(path, digest)
    return digest


def _atomic_csv(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or _hash_sidecar(path).exists():
        raise FileExistsError(f"append-only CSV artifact already exists: {path}")
    temporary = _temporary_output_path(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        _publish_without_replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    digest = hash_file(path)
    _write_hash_sidecar(path, digest)
    return digest


def _verify_artifact(path: Path, expected: str) -> None:
    observed = hash_file(path)
    if observed != expected:
        raise ProtocolViolation(
            f"immutable artifact changed: {path} expected {expected}, observed {observed}"
        )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _attest_source_freeze() -> Any:
    """Attest only the source tree executing this module."""

    return attest_preoutcome_freeze(
        required_tag=PREOUTCOME_TAG,
        repository=SOURCE_ROOT,
    )


def _freeze_mapping(attestation: Any) -> dict[str, str]:
    """Serialize the annotated-tag identity; retain synthetic compatibility."""

    mapping = {
        "commit": str(attestation.commit),
        "tag": str(attestation.tag),
    }
    tag_object = getattr(attestation, "tag_object", None)
    if tag_object is not None:
        mapping["tag_object"] = str(tag_object)
    return mapping


def _git_blob(relative_path: Path, commit: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{relative_path.as_posix()}"],
            cwd=SOURCE_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise ProtocolViolation(
            f"canonical tracked file is absent at {commit}: {relative_path}"
        ) from error


def _tracked_file_identity(relative_path: Path, commit: str) -> dict[str, str]:
    path = SOURCE_ROOT / relative_path
    working = path.read_bytes()
    committed = _git_blob(relative_path, commit)
    if working != committed:
        raise ProtocolViolation(
            f"working canonical file differs byte-for-byte from {commit}: {relative_path}"
        )
    return {
        "relative_path": relative_path.as_posix(),
        "sha256": hashlib.sha256(working).hexdigest(),
        "git_blob_sha256": hashlib.sha256(committed).hexdigest(),
    }


def _require_biological_config(config: ICMSExperimentConfig) -> None:
    if config.profile != "full" or config.seed != LOCK_SEED:
        raise ProtocolViolation("biological ICMS is frozen to full optimization and seed 20260725")
    expected = make_icms_config(
        "full",
        seed=LOCK_SEED,
        device=config.normal_fit.device,
    )
    if config.to_mapping() != expected.to_mapping():
        raise ProtocolViolation(
            "biological ICMS configuration differs from the frozen full profile"
        )


def _require_biological_methods(methods: Sequence[MethodName]) -> None:
    if tuple(methods) != REPORT_METHODS:
        raise ProtocolViolation("biological ICMS requires the complete ordered REPORT_METHODS")


def _canonical_provenance(
    processed_root: Path,
    *,
    commit: str,
    verify_h5: bool,
) -> dict[str, Any]:
    """Verify tracked manifests and the exact frozen processed release."""

    index_identity = _tracked_file_identity(CANONICAL_INDEX_RELATIVE, commit)
    raw_manifest_identity = _tracked_file_identity(CANONICAL_RAW_MANIFEST_RELATIVE, commit)
    canonical_index = SOURCE_ROOT / CANONICAL_INDEX_RELATIVE
    provided_index = processed_root / "index.json"
    if provided_index.read_bytes() != canonical_index.read_bytes():
        raise ProtocolViolation(
            "provided processed index is not byte-identical to the tracked canonical index"
        )
    index = _load_json(canonical_index)
    expected_config = asdict(DEFAULT_PREPROCESS_CONFIG)
    if (
        index.get("schema") != "cadence-dandi-001868-index-v1"
        or index.get("dandiset_id") != DANDISET_ID
        or index.get("dandiset_version") != DANDISET_VERSION
        or index.get("split_unit") != "animal_id"
        or index.get("config") != expected_config
        or index.get("totals") != CANONICAL_INDEX_TOTALS
    ):
        raise ProtocolViolation("tracked DANDI processed index metadata changed")
    rows = index.get("animals", [])
    if {str(row["animal_id"]) for row in rows} != set(TASK_MICE) or len(rows) != 6:
        raise ProtocolViolation("canonical processed index does not contain six task mice")
    raw_manifest = load_frozen_manifest(SOURCE_ROOT / CANONICAL_RAW_MANIFEST_RELATIVE)
    if (
        raw_manifest["dandiset_id"] != DANDISET_ID
        or raw_manifest["version"] != DANDISET_VERSION
        or set(raw_manifest["task_mice"]) != set(TASK_MICE)
    ):
        raise ProtocolViolation("canonical raw-asset manifest identity changed")
    h5_digests: dict[str, str] = {}
    if verify_h5:
        for row in rows:
            animal = str(row["animal_id"])
            path = processed_root / Path(str(row["output"])).name
            observed = hash_file(path)
            if observed != str(row["output_sha256"]):
                raise ProtocolViolation(f"processed H5 differs from committed index for {animal}")
            h5_digests[animal] = observed
    return {
        "source_root": str(SOURCE_ROOT),
        "git_commit": commit,
        "preoutcome_tag": PREOUTCOME_TAG,
        "processed_index": index_identity,
        "provided_index_sha256": hash_file(provided_index),
        "raw_asset_manifest": raw_manifest_identity,
        "dandiset_id": DANDISET_ID,
        "dandiset_version": DANDISET_VERSION,
        "index_totals": CANONICAL_INDEX_TOTALS,
        "verified_h5_sha256": h5_digests,
        "canonical_target_order": list(TASK_MICE),
        "outer_mapping": {
            target: [animal for animal in TASK_MICE if animal != target] for target in TASK_MICE
        },
    }


def _reject_canonical_release_in_synthetic_mode(processed_root: Path) -> None:
    """Prevent development settings from being pointed at the biological release."""

    provided_path = processed_root / "index.json"
    if not provided_path.exists():
        return
    provided = _load_json(provided_path)
    canonical = _load_json(SOURCE_ROOT / CANONICAL_INDEX_RELATIVE)

    def identities(payload: Mapping[str, Any]) -> dict[str, str]:
        return {
            str(row["animal_id"]): str(row["output_sha256"]) for row in payload.get("animals", [])
        }

    if (
        provided.get("dandiset_id") == canonical.get("dandiset_id")
        and provided.get("dandiset_version") == canonical.get("dandiset_version")
        and identities(provided) == identities(canonical)
    ):
        raise ProtocolViolation("synthetic mode cannot open the canonical biological ICMS release")


def _write_stage_completion(
    directory: Path,
    *,
    stage: Literal["prepare", "predict", "score"],
    artifact_path: Path,
    artifact_sha256: str,
    freeze: Mapping[str, Any],
    canonical_relative_output: str | None,
    seal_transaction_sha256: str,
) -> tuple[Path, str]:
    path = directory / f"{stage}_complete.json"
    if path.exists() or _hash_sidecar(path).exists():
        raise ProtocolViolation(f"{stage} completion already exists")
    digest = _atomic_json(
        path,
        {
            "schema": f"cadence-icms-{stage}-complete-v1",
            "stage": stage,
            "artifact": artifact_path.name,
            "artifact_sha256": artifact_sha256,
            "freeze_attestation": dict(freeze),
            "canonical_relative_output": canonical_relative_output,
            "seal_transaction_sha256": seal_transaction_sha256,
            "append_only": True,
        },
    )
    return path, digest


def _verify_stage_completion(
    directory: Path,
    stage: Literal["prepare", "predict", "score"],
) -> dict[str, Any]:
    expected_artifact = {
        "prepare": "prepare_manifest.json",
        "predict": "prediction_manifest.json",
        "score": "metrics.json",
    }[stage]
    path = directory / f"{stage}_complete.json"
    _verify_sidecar(path)
    payload = _load_json(path)
    artifact = directory / expected_artifact
    artifact_sha256 = _verify_sidecar(artifact)
    artifact_payload = _load_json(artifact)

    def freeze_fields(value: Any) -> dict[str, str]:
        if not isinstance(value, Mapping):
            raise ProtocolViolation(f"{stage} completion freeze attestation is missing")
        result = {
            "commit": str(value.get("commit", "")),
            "tag": str(value.get("tag", "")),
        }
        if value.get("tag_object") is not None:
            result["tag_object"] = str(value["tag_object"])
        if _COMMIT_PATTERN.fullmatch(result["commit"]) is None or not result["tag"]:
            raise ProtocolViolation(f"{stage} completion freeze attestation is malformed")
        if "tag_object" in result and _COMMIT_PATTERN.fullmatch(result["tag_object"]) is None:
            raise ProtocolViolation(f"{stage} completion tag object is malformed")
        return result

    transaction_sha256 = artifact_payload.get("target_seal_transaction_sha256")
    if (
        payload.get("schema") != f"cadence-icms-{stage}-complete-v1"
        or payload.get("stage") != stage
        or payload.get("artifact") != expected_artifact
        or payload.get("artifact_sha256") != artifact_sha256
        or payload.get("append_only") is not True
        or freeze_fields(payload.get("freeze_attestation"))
        != freeze_fields(artifact_payload.get("freeze_attestation"))
        or payload.get("canonical_relative_output")
        != artifact_payload.get("canonical_relative_output")
        or not isinstance(transaction_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", transaction_sha256) is None
        or payload.get("seal_transaction_sha256") != transaction_sha256
    ):
        raise ProtocolViolation(f"invalid ICMS {stage} completion manifest")
    _verify_artifact(artifact, payload["artifact_sha256"])
    return payload


def _active_seal_path(processed_root: Path) -> Path:
    return processed_root / ACTIVE_SEAL_NAME


def _validate_icms_seal_transaction(
    directory: Path,
    prepare_manifest: Mapping[str, Any],
    *,
    require_active_registry: bool,
) -> str:
    seal = dict(prepare_manifest.get("physical_target_seal", {}))
    transaction_sha256 = prepare_manifest.get("target_seal_transaction_sha256")
    if (
        not isinstance(transaction_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", transaction_sha256) is None
        or seal.get("sha256") != transaction_sha256
        or seal.get("path") != "target_seal.json"
    ):
        raise ProtocolViolation("ICMS target-seal transaction binding is malformed")
    seal_path = directory / "target_seal.json"
    if _verify_sidecar(seal_path) != transaction_sha256:
        raise ProtocolViolation("ICMS immutable seal differs from its transaction digest")
    immutable_seal = _load_json(seal_path)
    expected_seal = {key: value for key, value in seal.items() if key not in {"path", "sha256"}}
    if immutable_seal != expected_seal:
        raise ProtocolViolation("ICMS prepare manifest differs from its immutable seal")
    registry = _active_seal_path(Path(str(seal["processed_root"])))
    if require_active_registry:
        if not registry.exists() or hash_file(registry) != transaction_sha256:
            raise ProtocolViolation("ICMS active seal registry differs from its transaction digest")
        if _load_json(registry) != immutable_seal:
            raise ProtocolViolation("ICMS active seal registry differs from its immutable seal")
    return transaction_sha256


def _create_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(_jsonable(dict(payload)), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
    except BaseException:
        path.unlink(missing_ok=True)
        _fsync_directory(path.parent)
        raise


def _seal_target_source(
    *,
    target_path: Path,
    processed_root: Path,
    fold_directory: Path,
    target_animal: str,
    expected_sha256: str,
    canonical_relative_output: str | None,
) -> tuple[dict[str, Any], str]:
    registry = _active_seal_path(processed_root)
    source_stat = target_path.stat()
    original_mode = stat.S_IMODE(source_stat.st_mode)
    payload = {
        "schema": "cadence-icms-physical-target-seal-v1",
        "target_animal": target_animal,
        "target_path": str(target_path.resolve()),
        "processed_root": str(processed_root.resolve()),
        "fold_directory": str(fold_directory.resolve()),
        "canonical_relative_output": canonical_relative_output,
        "expected_sha256": expected_sha256,
        "device_id": int(source_stat.st_dev),
        "inode": int(source_stat.st_ino),
        "original_mode": original_mode,
        "sealed_mode": 0,
        "active": True,
    }
    _create_exclusive_json(registry, payload)
    try:
        os.chmod(target_path, 0)
        observed_mode = stat.S_IMODE(target_path.stat().st_mode)
        if observed_mode != payload["sealed_mode"]:
            raise ProtocolViolation("target H5 did not enter the exact physical sealed mode")
        seal_path = fold_directory / "target_seal.json"
        seal_sha = _atomic_json(seal_path, payload)
        if hash_file(registry) != seal_sha:
            raise ProtocolViolation("active target-seal transaction digest changed")
    except Exception:
        os.chmod(target_path, original_mode)
        registry.unlink(missing_ok=True)
        _fsync_directory(registry.parent)
        raise
    return {**payload, "path": seal_path.name}, seal_sha


def _assert_target_source_sealed(
    directory: Path,
    prepare_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_icms_seal_transaction(
        directory,
        prepare_manifest,
        require_active_registry=True,
    )
    seal_row = prepare_manifest["physical_target_seal"]
    seal_path = directory / seal_row["path"]
    _verify_artifact(seal_path, seal_row["sha256"])
    seal = _load_json(seal_path)
    target_path = Path(seal["target_path"])
    source_stat = target_path.stat()
    if int(source_stat.st_dev) != int(seal["device_id"]) or int(source_stat.st_ino) != int(
        seal["inode"]
    ):
        raise ProtocolViolation("target H5 file identity differs from the immutable seal")
    observed_mode = stat.S_IMODE(source_stat.st_mode)
    if observed_mode & 0o444:
        raise ProtocolViolation("target outcome-bearing H5 is readable during predict")
    if observed_mode != int(seal["sealed_mode"]):
        raise ProtocolViolation("target H5 permissions differ from the immutable seal")
    registry = _active_seal_path(Path(seal["processed_root"]))
    if not registry.exists() or _load_json(registry) != seal:
        raise ProtocolViolation("active target-seal registry is absent or inconsistent")
    return seal


def _restore_target_source(
    directory: Path,
    prepare_manifest: Mapping[str, Any],
    *,
    canonical_relative_output: str | None,
) -> tuple[Path, dict[str, Any], str, str]:
    seal = _assert_target_source_sealed(directory, prepare_manifest)
    target_path = Path(seal["target_path"])
    os.chmod(target_path, int(seal["original_mode"]))
    restored_mode = stat.S_IMODE(target_path.stat().st_mode)
    if restored_mode != int(seal["original_mode"]):
        raise ProtocolViolation("target H5 original mode was not restored exactly")
    try:
        observed_digest = hash_file(target_path)
        if observed_digest != str(seal["expected_sha256"]):
            raise ProtocolViolation(
                "processed target file changed while sealed; physical seal remains active"
            )
    except Exception as error:
        os.chmod(target_path, int(seal["sealed_mode"]))
        resealed_mode = stat.S_IMODE(target_path.stat().st_mode)
        if resealed_mode != int(seal["sealed_mode"]):
            raise ProtocolViolation(
                "target integrity failed and the H5 could not be re-sealed"
            ) from error
        raise
    audit = {
        "schema": "cadence-icms-target-restore-v1",
        "target_animal": seal["target_animal"],
        "target_path": str(target_path),
        "sealed_mode": seal["sealed_mode"],
        "device_id": seal["device_id"],
        "inode": seal["inode"],
        "original_mode": seal["original_mode"],
        "restored_mode": restored_mode,
        "original_mode_restored_exactly": True,
        "registry_retained_until_score_commit": True,
        "canonical_relative_output": canonical_relative_output,
        "restoration_status": "PENDING_SCORE_COMMIT_FINALIZATION",
        "immutable_seal_sha256": prepare_manifest["physical_target_seal"]["sha256"],
        "seal_transaction_sha256": prepare_manifest["target_seal_transaction_sha256"],
    }
    audit_path = directory / "target_restore.json"
    audit_sha = _atomic_json(audit_path, audit)
    return target_path, audit, audit_sha, observed_digest


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _quarantine_icms_path(path: Path) -> Path:
    if not path.exists():
        return path
    placeholder = Path(
        tempfile.mkdtemp(
            prefix=f".{path.name}.interrupted-",
            dir=path.parent,
        )
    )
    placeholder.rmdir()
    path.rename(placeholder)
    _fsync_directory(path.parent)
    return placeholder


def _quarantine_icms_artifacts(directory: Path, names: Sequence[str]) -> Path | None:
    existing = [directory / name for name in names if (directory / name).exists()]
    existing.extend(
        path for path in directory.glob(".*.tmp") if path.is_file() and path not in existing
    )
    if not existing:
        return None
    recovery = Path(
        tempfile.mkdtemp(
            prefix=f".{directory.name}.interrupted-stage-",
            dir=directory.parent,
        )
    )
    for path in existing:
        if path.name == "sealed_target_outcomes.npz" or (
            path.name.startswith(".sealed_target_outcomes.npz.") and path.name.endswith(".tmp")
        ):
            if path.is_symlink():
                raise ProtocolViolation("interrupted ICMS outcome copy is a symlink")
            path.chmod(0)
            if stat.S_IMODE(path.stat().st_mode) != 0 or os.access(path, os.R_OK):
                raise ProtocolViolation("interrupted ICMS outcome copy could not be sealed")
        path.rename(recovery / path.name)
    _fsync_directory(directory)
    _fsync_directory(recovery)
    _fsync_directory(directory.parent)
    return recovery


def _icms_completion_exists(directory: Path, stage: Literal["prepare", "predict", "score"]) -> bool:
    path = directory / f"{stage}_complete.json"
    sidecar = _hash_sidecar(path)
    if path.exists() != sidecar.exists():
        return False
    if not path.exists():
        return False
    _verify_stage_completion(directory, stage)
    return True


def _finalize_icms_target_restore_unprotected(
    directory: Path,
    prepare_manifest: Mapping[str, Any],
    *,
    canonical_relative_output: str | None,
) -> dict[str, Any]:
    """Finalize and attest source restoration only after score commit."""

    score_completion = _verify_stage_completion(directory, "score")
    score_completion_path = directory / "score_complete.json"
    score_completion_sha = _verify_sidecar(score_completion_path)
    seal = dict(prepare_manifest["physical_target_seal"])
    registry = _active_seal_path(Path(str(seal["processed_root"])))
    transaction_sha256 = _validate_icms_seal_transaction(
        directory,
        prepare_manifest,
        require_active_registry=registry.exists(),
    )
    if score_completion.get("seal_transaction_sha256") != transaction_sha256:
        raise ProtocolViolation("ICMS score completion transaction binding changed")
    final_path = directory / "target_restore_complete.json"
    final_sidecar = _hash_sidecar(final_path)

    def validate_completion(payload: Mapping[str, Any]) -> dict[str, Any]:
        target_path = Path(str(seal["target_path"]))
        try:
            source_stat = target_path.stat()
        except OSError as error:
            raise ProtocolViolation("restored ICMS target is unavailable") from error
        if int(source_stat.st_dev) != int(seal["device_id"]) or int(source_stat.st_ino) != int(
            seal["inode"]
        ):
            raise ProtocolViolation("restored ICMS target identity changed")
        original_mode = int(seal["original_mode"])
        if stat.S_IMODE(source_stat.st_mode) != original_mode or not os.access(
            target_path, os.R_OK
        ):
            raise ProtocolViolation("restored ICMS target mode changed")
        observed_digest = hash_file(target_path)
        if observed_digest != seal["expected_sha256"]:
            raise ProtocolViolation("restored ICMS target digest changed")
        restore_sha = _verify_sidecar(directory / "target_restore.json")
        expected = {
            "schema": ICMS_RESTORE_COMPLETION_SCHEMA,
            "restored_after_score_commit": True,
            "canonical_relative_output": canonical_relative_output,
            "target_animal": seal["target_animal"],
            "target_path": str(target_path),
            "restored_mode": original_mode,
            "target_sha256": observed_digest,
            "immutable_seal_sha256": seal["sha256"],
            "seal_transaction_sha256": transaction_sha256,
            "restore_audit_sha256": restore_sha,
            "score_completion_artifact": score_completion["artifact"],
            "score_completion_sha256": score_completion_sha,
            "registry_retained_until_score_commit": True,
            "registry_removed_after_finalization": True,
        }
        if dict(payload) != expected:
            raise ProtocolViolation("ICMS restore completion binding changed")
        return dict(payload)

    if final_path.exists() and final_sidecar.exists():
        _verify_sidecar(final_path)
        payload = validate_completion(_load_json(final_path))
        registry.unlink(missing_ok=True)
        _fsync_directory(registry.parent)
        return payload
    if final_path.exists() != final_sidecar.exists():
        if not registry.exists():
            raise ProtocolViolation(
                "ICMS restore completion publication is incomplete and its "
                "active registry is missing"
            )
        if final_path.exists():
            try:
                payload = validate_completion(_load_json(final_path))
            except (OSError, ValueError, TypeError, KeyError, ProtocolViolation):
                _quarantine_icms_artifacts(directory, (final_path.name,))
            else:
                digest = hash_file(final_path)
                _write_hash_sidecar(final_path, digest)
                registry.unlink()
                _fsync_directory(registry.parent)
                return payload
        else:
            _quarantine_icms_artifacts(directory, (final_sidecar.name,))
    if not registry.exists():
        raise ProtocolViolation("ICMS active target-seal registry disappeared before finalization")
    immutable_seal = _load_json(directory / "target_seal.json")
    if _load_json(registry) != immutable_seal:
        raise ProtocolViolation("ICMS active target-seal registry changed before finalization")
    target_path = Path(str(immutable_seal["target_path"]))
    source_stat = target_path.stat()
    if int(source_stat.st_dev) != int(immutable_seal["device_id"]) or int(
        source_stat.st_ino
    ) != int(immutable_seal["inode"]):
        raise ProtocolViolation("ICMS target identity changed before restoration finalization")
    original_mode = int(immutable_seal["original_mode"])
    if stat.S_IMODE(source_stat.st_mode) != original_mode:
        os.chmod(target_path, original_mode)
    observed_digest = hash_file(target_path)
    if observed_digest != immutable_seal["expected_sha256"]:
        os.chmod(target_path, int(immutable_seal["sealed_mode"]))
        raise ProtocolViolation("ICMS target changed before restoration finalization")
    restore_path = directory / "target_restore.json"
    restore_sha = _verify_sidecar(restore_path)
    payload = {
        "schema": ICMS_RESTORE_COMPLETION_SCHEMA,
        "restored_after_score_commit": True,
        "canonical_relative_output": canonical_relative_output,
        "target_animal": immutable_seal["target_animal"],
        "target_path": str(target_path),
        "restored_mode": original_mode,
        "target_sha256": observed_digest,
        "immutable_seal_sha256": prepare_manifest["physical_target_seal"]["sha256"],
        "seal_transaction_sha256": transaction_sha256,
        "restore_audit_sha256": restore_sha,
        "score_completion_artifact": score_completion["artifact"],
        "score_completion_sha256": score_completion_sha,
        "registry_retained_until_score_commit": True,
        "registry_removed_after_finalization": True,
    }
    _atomic_json(final_path, payload)
    registry.unlink()
    _fsync_directory(registry.parent)
    return payload


def _finalize_icms_target_restore(
    directory: Path,
    prepare_manifest: Mapping[str, Any],
    *,
    canonical_relative_output: str | None,
) -> dict[str, Any]:
    """Finalize restoration, re-sealing on every pre-commit failure."""

    seal = dict(prepare_manifest["physical_target_seal"])
    registry = _active_seal_path(Path(str(seal["processed_root"])))
    _validate_icms_seal_transaction(
        directory,
        prepare_manifest,
        require_active_registry=registry.exists(),
    )
    try:
        return _finalize_icms_target_restore_unprotected(
            directory,
            prepare_manifest,
            canonical_relative_output=canonical_relative_output,
        )
    except BaseException:
        if registry.exists():
            _reseal_icms_transaction(seal)
        raise


def _reseal_icms_transaction(seal: Mapping[str, Any]) -> None:
    target_path = Path(str(seal["target_path"]))
    source_stat = target_path.stat()
    if int(source_stat.st_dev) != int(seal["device_id"]) or int(source_stat.st_ino) != int(
        seal["inode"]
    ):
        raise ProtocolViolation("ICMS recovery found a substituted target H5")
    original_mode = int(seal["original_mode"])
    sealed_mode = int(seal["sealed_mode"])
    observed_mode = stat.S_IMODE(source_stat.st_mode)
    if observed_mode == original_mode:
        observed_sha256 = hash_file(target_path)
        os.chmod(target_path, sealed_mode)
        if observed_sha256 != seal["expected_sha256"]:
            raise ProtocolViolation("ICMS recovery found changed readable target H5")
    elif observed_mode != sealed_mode:
        raise ProtocolViolation("ICMS recovery found an unexpected target H5 mode")
    if stat.S_IMODE(target_path.stat().st_mode) != sealed_mode:
        raise ProtocolViolation("ICMS recovery could not re-seal the target H5")


def _rollback_incomplete_icms_prepare(
    seal: Mapping[str, Any],
    *,
    registry: Path,
    directory: Path,
) -> None:
    target_path = Path(str(seal["target_path"]))
    source_stat = target_path.stat()
    if int(source_stat.st_dev) != int(seal["device_id"]) or int(source_stat.st_ino) != int(
        seal["inode"]
    ):
        raise ProtocolViolation("ICMS prepare recovery found a substituted target H5")
    original_mode = int(seal["original_mode"])
    sealed_mode = int(seal["sealed_mode"])
    observed_mode = stat.S_IMODE(source_stat.st_mode)
    if observed_mode not in {original_mode, sealed_mode}:
        raise ProtocolViolation("ICMS prepare recovery found an unexpected target H5 mode")
    if observed_mode != original_mode:
        os.chmod(target_path, original_mode)
    if hash_file(target_path) != seal["expected_sha256"]:
        os.chmod(target_path, sealed_mode)
        raise ProtocolViolation("ICMS prepare recovery found changed target H5")
    registry.unlink()
    _fsync_directory(registry.parent)
    if directory.exists():
        _quarantine_icms_path(directory)


def _validate_icms_active_registry_scope(
    seal: Mapping[str, Any],
    *,
    processed_root: Path,
    expected_directory: Path,
    expected_target_animal: str,
    expected_canonical_relative_output: str,
) -> None:
    """Validate an uncommitted registry before trusting any path it contains."""

    source_digests, source_paths = _source_index(processed_root)
    expected_target = source_paths[expected_target_animal].resolve()
    expected_keys = {
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
    if (
        set(seal) != expected_keys
        or seal.get("schema") != "cadence-icms-physical-target-seal-v1"
        or seal.get("target_animal") != expected_target_animal
        or Path(str(seal.get("target_path", ""))) != expected_target
        or Path(str(seal.get("processed_root", ""))) != processed_root.resolve()
        or Path(str(seal.get("fold_directory", ""))) != expected_directory.resolve()
        or seal.get("canonical_relative_output") != expected_canonical_relative_output
        or seal.get("expected_sha256") != source_digests[expected_target_animal]
        or re.fullmatch(r"[0-9a-f]{64}", str(seal.get("expected_sha256", ""))) is None
        or not isinstance(seal.get("device_id"), int)
        or int(seal["device_id"]) < 0
        or not isinstance(seal.get("inode"), int)
        or int(seal["inode"]) < 0
        or not isinstance(seal.get("original_mode"), int)
        or int(seal["original_mode"]) < 0
        or int(seal["original_mode"]) > 0o7777
        or not int(seal["original_mode"]) & 0o444
        or seal.get("sealed_mode") != 0
        or seal.get("active") is not True
    ):
        raise ProtocolViolation("ICMS active prepare registry binding changed")
    source_stat = expected_target.stat()
    if (
        int(source_stat.st_dev) != int(seal["device_id"])
        or int(source_stat.st_ino) != int(seal["inode"])
        or stat.S_IMODE(source_stat.st_mode)
        not in {int(seal["original_mode"]), int(seal["sealed_mode"])}
    ):
        raise ProtocolViolation("ICMS active prepare registry target identity changed")


def _canonical_icms_registry_scope(target_animal: str) -> tuple[Path, str]:
    relative = _canonical_icms_relative_output(target_animal)
    return (SOURCE_ROOT / relative).resolve(), relative.as_posix()


def _recover_icms_prepare(
    *,
    processed_root: Path,
    directory: Path,
    target_animal: str,
    canonical_relative_output: str,
) -> str | None:
    registry = _active_seal_path(processed_root)
    if not registry.exists():
        if (
            directory.exists()
            and any(directory.iterdir())
            and not _icms_completion_exists(directory, "prepare")
        ):
            _quarantine_icms_path(directory)
        return None
    seal = _load_json(registry)
    registry_target = seal.get("target_animal")
    if not isinstance(registry_target, str) or registry_target not in TASK_MICE:
        raise ProtocolViolation("ICMS active prepare registry target scope changed")
    prior_directory, prior_canonical = _canonical_icms_registry_scope(registry_target)
    _validate_icms_active_registry_scope(
        seal,
        processed_root=processed_root,
        expected_directory=prior_directory,
        expected_target_animal=registry_target,
        expected_canonical_relative_output=prior_canonical,
    )
    try:
        prior_score_complete = _icms_completion_exists(prior_directory, "score")
    except BaseException:
        _reseal_icms_transaction(seal)
        raise
    if prior_score_complete:
        prior_manifest = _load_json(prior_directory / "prepare_manifest.json")
        _finalize_icms_target_restore(
            prior_directory,
            prior_manifest,
            canonical_relative_output=prior_canonical,
        )
        if prior_directory != directory.resolve():
            return None
        return "score_complete"
    if (
        prior_directory != directory.resolve()
        or registry_target != target_animal
        or prior_canonical != canonical_relative_output
    ):
        raise ProtocolViolation(
            "another ICMS fold has an active target-seal transaction; "
            "resume that canonical fold first"
        )
    if _icms_completion_exists(directory, "prepare"):
        prepare_manifest = _load_json(directory / "prepare_manifest.json")
        transaction_sha256 = _validate_icms_seal_transaction(
            directory,
            prepare_manifest,
            require_active_registry=True,
        )
        if (
            _verify_stage_completion(directory, "prepare").get("seal_transaction_sha256")
            != transaction_sha256
        ):
            raise ProtocolViolation("ICMS prepare completion transaction binding changed")
        _reseal_icms_transaction(seal)
        return "prepare_complete"
    _rollback_incomplete_icms_prepare(
        seal,
        registry=registry,
        directory=directory,
    )
    return None


def _recover_icms_stage(
    *,
    directory: Path,
    prepare_manifest: Mapping[str, Any],
    stage: Literal["predict", "score"],
    canonical_relative_output: str,
) -> str | None:
    seal = dict(prepare_manifest["physical_target_seal"])
    registry = _active_seal_path(Path(str(seal["processed_root"])))
    _validate_icms_seal_transaction(
        directory,
        prepare_manifest,
        require_active_registry=registry.exists(),
    )
    try:
        score_complete = _icms_completion_exists(directory, "score")
    except BaseException:
        if registry.exists():
            _reseal_icms_transaction(seal)
        raise
    if score_complete:
        _finalize_icms_target_restore(
            directory,
            prepare_manifest,
            canonical_relative_output=canonical_relative_output,
        )
        return "score_complete"
    transaction_sha256 = _validate_icms_seal_transaction(
        directory,
        prepare_manifest,
        require_active_registry=True,
    )
    if not registry.exists() or _load_json(registry) != _load_json(directory / "target_seal.json"):
        raise ProtocolViolation("ICMS active target-seal registry is absent or inconsistent")
    if (
        seal.get("canonical_relative_output") != canonical_relative_output
        or Path(str(seal.get("fold_directory", ""))) != directory.resolve()
    ):
        raise ProtocolViolation("ICMS active target-seal output binding changed")
    _reseal_icms_transaction(seal)
    predict_complete = _icms_completion_exists(directory, "predict")
    if stage == "predict":
        if predict_complete:
            if (
                _verify_stage_completion(directory, "predict").get("seal_transaction_sha256")
                != transaction_sha256
            ):
                raise ProtocolViolation("ICMS predict completion transaction binding changed")
            return "predict_complete"
        _quarantine_icms_artifacts(
            directory,
            (
                "predictions.npz",
                "predictions.npz.sha256",
                "prediction_manifest.json",
                "prediction_manifest.json.sha256",
                "frozen_models.pt",
                "frozen_models.pt.sha256",
                "predict_complete.json",
                "predict_complete.json.sha256",
            ),
        )
        return None
    if not predict_complete:
        raise ProtocolViolation("ICMS predict stage must be resumed before score")
    if (
        _verify_stage_completion(directory, "predict").get("seal_transaction_sha256")
        != transaction_sha256
    ):
        raise ProtocolViolation("ICMS predict completion transaction binding changed")
    _quarantine_icms_artifacts(
        directory,
        (
            "metrics.json",
            "metrics.json.sha256",
            "sealed_target_outcomes.npz",
            "sealed_target_outcomes.npz.sha256",
            "scored_condition_trajectories.npz",
            "scored_condition_trajectories.npz.sha256",
            "condition_metrics.csv",
            "condition_metrics.csv.sha256",
            "target_restore.json",
            "target_restore.json.sha256",
            "score_complete.json",
            "score_complete.json.sha256",
            "target_restore_complete.json",
            "target_restore_complete.json.sha256",
        ),
    )
    return None


def _decode_strings(values: npt.ArrayLike) -> npt.NDArray[np.str_]:
    array = np.asarray(values)
    if array.dtype.kind == "S":
        return np.char.decode(array, "utf-8")
    return array.astype(str)


@dataclass(frozen=True, slots=True)
class ICMSExperimentConfig:
    """Prespecified optimization, query, and uncertainty policy."""

    profile: Literal["smoke", "full"] = "smoke"
    latent_dim: int = 8
    hidden_dim: int = 24
    residual_rank: int = 2
    intervention_rank: int = 3
    batch_size: int = 16
    max_normal_trials_per_session: int | None = 40
    max_stimulation_trials_per_session: int | None = 48
    query_contexts_per_session: int = 4
    uncertainty_draws: int = 8
    current_grid_uA: tuple[float, ...] = tuple(float(value) for value in range(1, 14))
    pulse_frequency_hz: float = CANONICAL_FREQUENCY_HZ
    pulse_count: float = CANONICAL_PULSE_COUNT
    pulse_width_us: float = CANONICAL_PULSE_WIDTH_US
    train_stop_s: float = 0.7
    artifact_exclusion_stop_s: float = 0.705
    normal_fit: FitConfig = field(
        default_factory=lambda: FitConfig(
            learning_rate=3e-3,
            max_epochs=8,
            patience=3,
            seed=11,
            device="cpu",
            mixed_precision=False,
        )
    )
    intervention_fit: FitConfig = field(
        default_factory=lambda: FitConfig(
            learning_rate=3e-3,
            max_epochs=8,
            patience=3,
            seed=23,
            device="cpu",
            mixed_precision=False,
        )
    )
    target_fit: FitConfig = field(
        default_factory=lambda: FitConfig(
            learning_rate=3e-3,
            max_epochs=8,
            patience=3,
            seed=37,
            device="cpu",
            mixed_precision=False,
        )
    )
    seed: int = 20260725

    def validate(self) -> None:
        if self.profile not in {"smoke", "full"}:
            raise ValueError("profile must be smoke or full")
        if self.latent_dim < 2 or self.hidden_dim < 4:
            raise ValueError("model dimensions are too small")
        if self.residual_rank < 1 or self.intervention_rank < 1:
            raise ValueError("ranks must be positive")
        if self.batch_size < 1 or self.query_contexts_per_session < 1:
            raise ValueError("batch and context counts must be positive")
        if self.uncertainty_draws < 2:
            raise ValueError("at least two uncertainty draws are required")
        current = np.asarray(self.current_grid_uA, dtype=np.float64)
        if (
            current.ndim != 1
            or len(current) < 2
            or np.any(~np.isfinite(current))
            or np.any(current <= 0)
            or np.any(np.diff(current) <= 0)
        ):
            raise ValueError("current grid must be finite, positive, and strictly increasing")
        if not np.isclose(current[0], 1.0) or not np.isclose(current[-1], 13.0):
            raise ValueError("the frozen query lattice must span 1--13 uA")
        if not 0 < self.train_stop_s <= self.artifact_exclusion_stop_s:
            raise ValueError("invalid pulse-train or artifact interval")
        if (
            not np.isclose(self.pulse_count, CANONICAL_PULSE_COUNT)
            or not np.isclose(self.pulse_frequency_hz, CANONICAL_FREQUENCY_HZ)
            or not np.isclose(self.pulse_width_us, CANONICAL_PULSE_WIDTH_US)
            or not np.isclose(self.train_stop_s, CANONICAL_TRAIN_DURATION_S)
        ):
            raise ValueError("primary ICMS query must remain the canonical 70-pulse train")

    def to_mapping(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def make_icms_config(
    profile: Literal["smoke", "full"],
    *,
    seed: int = 20260725,
    device: str | None = None,
) -> ICMSExperimentConfig:
    """Return the fixed smoke or paper-scale settings."""

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = selected_device.startswith("cuda")
    if profile == "smoke":
        config = ICMSExperimentConfig(
            profile="smoke",
            seed=seed,
            normal_fit=replace(
                ICMSExperimentConfig().normal_fit,
                seed=seed + 11,
                device=selected_device,
                mixed_precision=amp,
            ),
            intervention_fit=replace(
                ICMSExperimentConfig().intervention_fit,
                seed=seed + 23,
                device=selected_device,
                mixed_precision=amp,
            ),
            target_fit=replace(
                ICMSExperimentConfig().target_fit,
                seed=seed + 37,
                device=selected_device,
                mixed_precision=amp,
            ),
        )
    else:
        base = ICMSExperimentConfig()
        config = ICMSExperimentConfig(
            profile="full",
            latent_dim=12,
            hidden_dim=96,
            residual_rank=2,
            intervention_rank=4,
            batch_size=32,
            max_normal_trials_per_session=None,
            max_stimulation_trials_per_session=None,
            query_contexts_per_session=16,
            uncertainty_draws=64,
            normal_fit=replace(
                base.normal_fit,
                learning_rate=1e-3,
                max_epochs=400,
                patience=35,
                seed=seed + 11,
                device=selected_device,
                mixed_precision=amp,
            ),
            intervention_fit=replace(
                base.intervention_fit,
                learning_rate=1e-3,
                max_epochs=400,
                patience=35,
                seed=seed + 23,
                device=selected_device,
                mixed_precision=amp,
            ),
            target_fit=replace(
                base.target_fit,
                learning_rate=1e-3,
                max_epochs=300,
                patience=30,
                seed=seed + 37,
                device=selected_device,
                mixed_precision=amp,
            ),
            seed=seed,
        )
    config.validate()
    return config


@dataclass(slots=True)
class NormalSession:
    """One session's normal-only support with a session-specific unit map."""

    animal_id: str
    session_key: str
    session_id: str
    time_s: FloatArray
    trial_keys: npt.NDArray[np.str_]
    normal_source: npt.NDArray[np.str_]
    neural_raw: FloatArray
    neural: FloatArray
    neural_mask: BoolArray
    behavior_raw: FloatArray
    behavior: FloatArray
    behavior_mask: BoolArray
    neural_center: FloatArray
    neural_scale: FloatArray
    behavior_center: FloatArray
    behavior_scale: FloatArray
    partitions: dict[str, npt.NDArray[np.int64]]
    unit_ids: npt.NDArray[np.int64]

    @property
    def adapter_id(self) -> str:
        return f"{self.animal_id}::{self.session_key}"

    @property
    def onset(self) -> int:
        return int(np.searchsorted(self.time_s, 0.0, side="left"))


@dataclass(slots=True)
class StimSession:
    """Donor or newly unsealed target stimulation rows."""

    animal_id: str
    session_key: str
    session_id: str
    time_s: FloatArray
    trial_index: npt.NDArray[np.int64]
    descriptors: FloatArray
    neural_raw: FloatArray
    neural: FloatArray
    neural_mask: BoolArray
    behavior_raw: FloatArray
    behavior: FloatArray
    behavior_mask: BoolArray
    blocks: npt.NDArray[np.int64]
    block_rule: str
    block_validated: bool
    positive_trial_count: int
    excluded_noncanonical_count: int

    @property
    def adapter_id(self) -> str:
        return f"{self.animal_id}::{self.session_key}"

    @property
    def onset(self) -> int:
        return int(np.searchsorted(self.time_s, 0.0, side="left"))


def canonical_icms_mask(
    descriptors: npt.ArrayLike,
    *,
    event_duration_s: npt.ArrayLike | None = None,
) -> BoolArray:
    """Identify the frozen 70-pulse, 100 Hz, 167 us, ~0.7 s train family."""

    values = np.asarray(descriptors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != DESCRIPTOR_DIM:
        raise ValueError("expected a matrix of physical intervention descriptors")
    column = {name: index for index, name in enumerate(INTERVENTION_DESCRIPTOR_COLUMNS)}
    mask = (
        (values[:, column["stim_present"]] > 0)
        & np.isclose(
            values[:, column["pulse_count"]],
            CANONICAL_PULSE_COUNT,
            rtol=0.0,
            atol=1e-6,
        )
        & np.isclose(
            values[:, column["frequency_hz"]],
            CANONICAL_FREQUENCY_HZ,
            rtol=0.0,
            atol=1e-6,
        )
        & np.isclose(
            values[:, column["pulse_width_us"]],
            CANONICAL_PULSE_WIDTH_US,
            rtol=0.0,
            atol=1e-6,
        )
    )
    if event_duration_s is not None:
        duration = np.asarray(event_duration_s, dtype=np.float64)
        if duration.shape != (len(values),):
            raise ValueError("event durations must align with descriptors")
        mask &= np.isfinite(duration) & np.isclose(
            duration,
            CANONICAL_TRAIN_DURATION_S,
            rtol=0.0,
            atol=CANONICAL_DURATION_TOLERANCE_S,
        )
    return mask.astype(bool)


def _event_duration(trials: h5py.Group) -> FloatArray | None:
    if "event_start_time" not in trials or "event_stop_time" not in trials:
        return None
    return np.asarray(trials["event_stop_time"], dtype=np.float64) - np.asarray(
        trials["event_start_time"], dtype=np.float64
    )


def _signal_blind_order(keys: Sequence[str], *salt: object) -> npt.NDArray[np.int64]:
    return np.asarray(
        sorted(
            range(len(keys)),
            key=lambda index: (_stable_digest(*salt, keys[index]), keys[index]),
        ),
        dtype=np.int64,
    )


def split_normal_rows(
    trial_keys: Sequence[str],
    *,
    animal_id: str,
    session_key: str,
    seed: int,
    maximum: int | None,
) -> dict[str, npt.NDArray[np.int64]]:
    """Deterministic 60/20/20 normal split using identifiers only."""

    keys = [str(value) for value in trial_keys]
    if len(keys) < 3 or len(set(keys)) != len(keys):
        raise ValueError("a session needs at least three unique normal windows")
    order = _signal_blind_order(keys, "icms-normal-v1", seed, animal_id, session_key)
    if maximum is not None:
        order = order[:maximum]
    count = len(order)
    fit_stop = max(1, int(math.floor(0.6 * count)))
    val_stop = max(fit_stop + 1, int(math.floor(0.8 * count)))
    val_stop = min(val_stop, count - 1)
    partitions = {
        "fit": np.sort(order[:fit_stop]),
        "val": np.sort(order[fit_stop:val_stop]),
        "audit": np.sort(order[val_stop:]),
    }
    joined = np.concatenate(list(partitions.values()))
    if len(joined) != count or len(np.unique(joined)) != count:
        raise AssertionError("normal split is not a disjoint partition")
    return partitions


def _masked_channel_location_scale(
    values: FloatArray,
    mask: BoolArray,
    fit_indices: npt.NDArray[np.int64],
) -> tuple[FloatArray, FloatArray]:
    fit = np.where(mask[fit_indices], values[fit_indices], np.nan)
    center = np.nanmean(fit, axis=(0, 1))
    scale = np.nanstd(fit, axis=(0, 1), ddof=1)
    center = np.nan_to_num(center, nan=0.0)
    valid = scale[np.isfinite(scale) & (scale > 1e-6)]
    floor = max(float(np.quantile(valid, 0.1)), 1e-3) if len(valid) else 1.0
    scale = np.where(np.isfinite(scale) & (scale > floor), scale, floor)
    return center.astype(np.float64), scale.astype(np.float64)


def _scale_session(
    neural_raw: FloatArray,
    neural_mask: BoolArray,
    behavior_raw: FloatArray,
    behavior_mask: BoolArray,
    fit: npt.NDArray[np.int64],
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    transformed = np.log1p(np.maximum(neural_raw, 0.0))
    ncenter, nscale = _masked_channel_location_scale(transformed, neural_mask, fit)
    bcenter, bscale = _masked_channel_location_scale(behavior_raw, behavior_mask, fit)
    neural = np.where(
        neural_mask,
        (transformed - ncenter[None, None, :]) / nscale[None, None, :],
        0.0,
    )
    behavior = np.where(
        behavior_mask,
        (behavior_raw - bcenter[None, None, :]) / bscale[None, None, :],
        0.0,
    )
    return neural, behavior, ncenter, nscale, bcenter, bscale


def _indexed(dataset: h5py.Dataset, indices: npt.NDArray[np.int64]) -> np.ndarray:
    """Read only selected rows; indices must be increasing for h5py."""

    indices = np.asarray(indices, dtype=np.int64)
    if np.any(np.diff(indices) <= 0):
        raise ValueError("HDF5 row selection must be strictly increasing")
    return np.asarray(dataset[indices])


def _expand_neural_mask(mask: npt.ArrayLike, neural_shape: tuple[int, ...]) -> BoolArray:
    values = np.asarray(mask, dtype=bool)
    if values.shape == neural_shape:
        return values
    if values.shape == neural_shape[:2]:
        return np.broadcast_to(values[:, :, None], neural_shape).copy()
    raise ProtocolViolation(
        f"spike validity mask {values.shape} cannot cover neural array {neural_shape}"
    )


def read_normal_sessions(
    animal_file: str | Path,
    *,
    config: ICMSExperimentConfig,
) -> list[NormalSession]:
    """Read only sealed normal rows from one processed animal file.

    The function never accesses positive-current descriptor rows or their
    signal slices.  Selection columns are limited to the loader's
    ``is_normal_calibration`` flag and identifiers needed for an audit.
    """

    sessions: list[NormalSession] = []
    with h5py.File(animal_file, "r") as file:
        animal_id = str(file.attrs["animal_id"])
        if animal_id not in TASK_MICE:
            raise ProtocolViolation(f"unexpected ICMS task animal {animal_id}")
        if bool(file.attrs.get("raw_stim_channel_in_descriptor", True)):
            raise ProtocolViolation("raw apparatus channel leaked into intervention descriptor")
        time_s = np.asarray(file["time_s"], dtype=np.float64)
        for session_key in sorted(file["sessions"]):
            group = file["sessions"][session_key]
            trials = group["trials"]
            normal_flag = np.asarray(trials["is_normal_calibration"], dtype=bool)
            indices = np.flatnonzero(normal_flag).astype(np.int64)
            if len(indices) < 3:
                continue
            descriptors = _indexed(group["intervention_descriptors"], indices)
            if np.any(descriptors != 0):
                raise ProtocolViolation("normal support contains a nonzero intervention descriptor")
            window_kind = _decode_strings(_indexed(trials["window_kind"], indices))
            if not np.all(window_kind == "normal"):
                raise ProtocolViolation("normal support contains a non-normal window")
            normal_source = _decode_strings(_indexed(trials["normal_source"], indices))
            if not set(normal_source).issubset({"catch", "iti"}):
                raise ProtocolViolation("unknown normal-calibration source")
            trial_index = _indexed(trials["trial_index"], indices).astype(np.int64)
            trial_keys = np.asarray(
                [
                    f"{animal_id}/{session_key}/{source}/{index}"
                    for source, index in zip(normal_source, trial_index, strict=True)
                ],
                dtype=str,
            )
            partitions = split_normal_rows(
                trial_keys,
                animal_id=animal_id,
                session_key=session_key,
                seed=config.seed,
                maximum=config.max_normal_trials_per_session,
            )
            signals = group["signals"]
            neural_raw = _indexed(signals["spike_rate_hz"], indices).astype(np.float64)
            neural_mask = _expand_neural_mask(
                _indexed(signals["spike_valid_mask"], indices), neural_raw.shape
            )
            displacement = _indexed(signals["wheel_displacement"], indices).astype(np.float64)
            velocity = _indexed(signals["wheel_velocity"], indices).astype(np.float64)
            wheel_mask = _indexed(signals["wheel_valid_mask"], indices).astype(bool)
            behavior_raw = np.stack((displacement, velocity), axis=-1)
            behavior_mask = np.repeat(wheel_mask[:, :, None], BEHAVIOR_DIM, axis=-1)
            (
                neural,
                behavior,
                neural_center,
                neural_scale,
                behavior_center,
                behavior_scale,
            ) = _scale_session(
                neural_raw,
                neural_mask,
                behavior_raw,
                behavior_mask,
                partitions["fit"],
            )
            unit_ids = np.asarray(group["units"]["unit_id"], dtype=np.int64)
            if neural_raw.shape[2] != len(unit_ids):
                raise ProtocolViolation("session unit table and spike-rate channels disagree")
            session = NormalSession(
                animal_id=animal_id,
                session_key=session_key,
                session_id=str(group.attrs["session_id"]),
                time_s=time_s,
                trial_keys=trial_keys,
                normal_source=normal_source,
                neural_raw=neural_raw,
                neural=neural,
                neural_mask=neural_mask,
                behavior_raw=behavior_raw,
                behavior=behavior,
                behavior_mask=behavior_mask,
                neural_center=neural_center,
                neural_scale=neural_scale,
                behavior_center=behavior_center,
                behavior_scale=behavior_scale,
                partitions=partitions,
                unit_ids=unit_ids,
            )
            if not 1 <= session.onset < len(time_s):
                raise ProtocolViolation("session time grid has no valid zero onset")
            sessions.append(session)
    if not sessions:
        raise ProtocolViolation(f"{animal_file} has no usable normal-support session")
    return sessions


def _normal_session_arrays(session: NormalSession) -> dict[str, npt.ArrayLike]:
    return {
        "animal_id": np.asarray(session.animal_id),
        "session_key": np.asarray(session.session_key),
        "session_id": np.asarray(session.session_id),
        "time_s": session.time_s,
        "trial_keys": session.trial_keys,
        "normal_source": session.normal_source,
        "neural_raw": session.neural_raw,
        "neural": session.neural,
        "neural_mask": session.neural_mask,
        "behavior_raw": session.behavior_raw,
        "behavior": session.behavior,
        "behavior_mask": session.behavior_mask,
        "neural_center": session.neural_center,
        "neural_scale": session.neural_scale,
        "behavior_center": session.behavior_center,
        "behavior_scale": session.behavior_scale,
        "partition_fit": session.partitions["fit"],
        "partition_val": session.partitions["val"],
        "partition_audit": session.partitions["audit"],
        "unit_ids": session.unit_ids,
    }


def load_normal_session(path: str | Path) -> NormalSession:
    with np.load(path, allow_pickle=False) as values:
        arrays = {name: values[name] for name in values.files}
    return NormalSession(
        animal_id=str(arrays["animal_id"]),
        session_key=str(arrays["session_key"]),
        session_id=str(arrays["session_id"]),
        time_s=arrays["time_s"].astype(np.float64),
        trial_keys=arrays["trial_keys"].astype(str),
        normal_source=arrays["normal_source"].astype(str),
        neural_raw=arrays["neural_raw"].astype(np.float64),
        neural=arrays["neural"].astype(np.float64),
        neural_mask=arrays["neural_mask"].astype(bool),
        behavior_raw=arrays["behavior_raw"].astype(np.float64),
        behavior=arrays["behavior"].astype(np.float64),
        behavior_mask=arrays["behavior_mask"].astype(bool),
        neural_center=arrays["neural_center"].astype(np.float64),
        neural_scale=arrays["neural_scale"].astype(np.float64),
        behavior_center=arrays["behavior_center"].astype(np.float64),
        behavior_scale=arrays["behavior_scale"].astype(np.float64),
        partitions={
            "fit": arrays["partition_fit"].astype(np.int64),
            "val": arrays["partition_val"].astype(np.int64),
            "audit": arrays["partition_audit"].astype(np.int64),
        },
        unit_ids=arrays["unit_ids"].astype(np.int64),
    )


def physical_query_lattice(config: ICMSExperimentConfig) -> FloatArray:
    """Frozen target-independent descriptor lattice.

    The 32 NET32 depths and every integer current from 1 through 13 uA are
    emitted before target stimulation metadata are opened.  The learned models
    consume current continuously; scoring linearly interpolates the already
    frozen predictions if an adaptive current lies between grid points.
    """

    depths = np.arange(0.0, 1860.0 + 60.0, 60.0)
    rows = np.zeros((len(depths) * len(config.current_grid_uA), DESCRIPTOR_DIM))
    column = {name: index for index, name in enumerate(INTERVENTION_DESCRIPTOR_COLUMNS)}
    cursor = 0
    for depth in depths:
        for current in config.current_grid_uA:
            row = rows[cursor]
            row[column["stim_present"]] = 1.0
            row[column["current_uA"]] = current
            row[column["frequency_hz"]] = config.pulse_frequency_hz
            row[column["pulse_count"]] = config.pulse_count
            row[column["pulse_width_us"]] = config.pulse_width_us
            row[column["electrode_rel_x_um"]] = 0.0
            row[column["electrode_rel_y_um"]] = depth
            row[column["electrode_rel_z_um"]] = 0.0
            row[column["electrode_depth_centered_um"]] = depth - NET32_DEPTH_CENTER_UM
            row[column["electrode_depth_fraction"]] = (
                depth - NET32_DEPTH_CENTER_UM
            ) / NET32_DEPTH_HALF_RANGE_UM
            cursor += 1
    return rows.astype(np.float64)


def normalize_descriptors(descriptors: npt.ArrayLike) -> FloatArray:
    """Prespecified dimensionless physical descriptor transform."""

    values = np.asarray(descriptors, dtype=np.float64).copy()
    if values.shape[-1] != DESCRIPTOR_DIM:
        raise ValueError("intervention descriptor width changed")
    scale = np.asarray(
        [1.0, 13.0, 100.0, 70.0, 167.0, 1860.0, 1860.0, 1860.0, 930.0, 1.0],
        dtype=np.float64,
    )
    return values / scale


def intervention_schedule(
    descriptors: npt.ArrayLike,
    time_s: npt.ArrayLike,
    *,
    train_stop_s: float,
) -> FloatArray:
    descriptor = normalize_descriptors(descriptors)
    time = np.asarray(time_s, dtype=np.float64)
    schedule = np.zeros((len(descriptor), len(time), DESCRIPTOR_DIM), dtype=np.float64)
    active_samples = np.flatnonzero((time >= 0.0) & (time < train_stop_s))
    if not len(active_samples):
        raise ValueError("time grid does not represent the stimulation train")
    # Control at sample j-1 drives the transition into sample j.
    transition_indices = np.clip(active_samples - 1, 0, len(time) - 2)
    schedule[:, transition_indices, :] = descriptor[:, None, :]
    return schedule


def _query_arrays(session: NormalSession, config: ICMSExperimentConfig) -> dict[str, Any]:
    context_pool = session.partitions["audit"]
    context_order = _signal_blind_order(
        [session.trial_keys[index] for index in context_pool],
        "icms-query-context-v1",
        config.seed,
        session.adapter_id,
    )
    selected = context_pool[context_order[: config.query_contexts_per_session]]
    if not len(selected):
        raise ProtocolViolation("target session has no untouched normal audit context")
    pre = session.onset - 1
    descriptors = physical_query_lattice(config)
    arrays: dict[str, Any] = {
        "animal_id": np.asarray(session.animal_id),
        "session_key": np.asarray(session.session_key),
        "session_id": np.asarray(session.session_id),
        "adapter_id": np.asarray(session.adapter_id),
        "time_s": session.time_s,
        "onset": np.asarray(session.onset, dtype=np.int64),
        "context_trial_keys": session.trial_keys[selected],
        "pre_neural": session.neural[selected, pre],
        "pre_neural_mask": session.neural_mask[selected, pre],
        "pre_behavior": session.behavior[selected, pre],
        "pre_behavior_mask": session.behavior_mask[selected, pre],
        "condition_descriptors": descriptors,
        "treated_intervention": intervention_schedule(
            descriptors, session.time_s, train_stop_s=config.train_stop_s
        ),
        "control_inputs": np.zeros(
            (len(descriptors), len(session.time_s), INPUT_DIM), dtype=np.float32
        ),
        "normal_control_neural": np.nanmean(
            np.where(
                session.neural_mask[selected, session.onset :],
                session.neural_raw[selected, session.onset :],
                np.nan,
            ),
            axis=0,
        ),
        "normal_control_behavior": np.nanmean(
            np.where(
                session.behavior_mask[selected, session.onset :],
                session.behavior_raw[selected, session.onset :],
                np.nan,
            ),
            axis=0,
        ),
        "neural_center": session.neural_center,
        "neural_scale": session.neural_scale,
        "behavior_center": session.behavior_center,
        "behavior_scale": session.behavior_scale,
        "unit_ids": session.unit_ids,
    }
    forbidden = [name for name in arrays if any(token in name for token in _FORBIDDEN_QUERY_TOKENS)]
    if forbidden:
        raise ProtocolViolation(f"target outcomes leaked into query bundle: {forbidden}")
    return arrays


def _source_index(processed_root: Path) -> tuple[dict[str, str], dict[str, Path]]:
    index_path = processed_root / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"processed index is missing: {index_path}")
    payload = _load_json(index_path)
    if payload.get("dandiset_id") != DANDISET_ID:
        raise ProtocolViolation("processed index is not DANDI:001868")
    digests: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for row in payload["animals"]:
        animal = str(row["animal_id"])
        source = Path(row["output"])
        if not source.is_absolute():
            candidates = (processed_root / source.name, source)
            source = next((candidate for candidate in candidates if candidate.exists()), source)
        digests[animal] = str(row["output_sha256"])
        paths[animal] = source
    if set(digests) != set(TASK_MICE):
        raise ProtocolViolation("processed index must contain exactly the six task mice")
    return digests, paths


def prepare_fold(
    *,
    processed_root: str | Path,
    output_directory: str | Path,
    target_animal: str,
    config: ICMSExperimentConfig,
    protocol_commit: str | None = None,
    run_mode: RunMode = "biological",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Prepare normal-only support and target-independent query files."""

    config.validate()
    if target_animal not in TASK_MICE:
        raise ValueError(f"target must be one of {TASK_MICE}")
    processed = Path(processed_root)
    destination = Path(output_directory)
    canonical_relative_output: str | None = None
    if run_mode == "biological":
        if overwrite:
            raise ProtocolViolation("biological ICMS stages are append-only")
        _require_biological_config(config)
        attestation = _attest_source_freeze()
        canonical_relative_output = _require_canonical_biological_output(
            destination,
            target_animal,
        )
        if protocol_commit is not None and protocol_commit != attestation.commit:
            raise ProtocolViolation("caller protocol commit differs from source freeze")
        recovery = _recover_icms_prepare(
            processed_root=processed,
            directory=destination,
            target_animal=target_animal,
            canonical_relative_output=canonical_relative_output,
        )
        if recovery is not None and (destination / "prepare_manifest.json").is_file():
            manifest = _load_json(destination / "prepare_manifest.json")
            return {
                **manifest,
                "manifest_sha256": _verify_sidecar(destination / "prepare_manifest.json"),
                "completion_path": str(destination / "prepare_complete.json"),
                "completion_sha256": _verify_sidecar(destination / "prepare_complete.json"),
                "output": str(destination),
            }
        protocol_commit = attestation.commit
        canonical_provenance = _canonical_provenance(
            processed,
            commit=attestation.commit,
            verify_h5=True,
        )
    elif run_mode == "synthetic":
        if _active_seal_path(processed).exists():
            raise ProtocolViolation(
                "another ICMS fold has an active physical target seal; score or "
                "recover that fold before preparing the next LOAO target"
            )
        _reject_canonical_release_in_synthetic_mode(processed)
        protocol_commit = protocol_commit or ("0" * 40)
        if not _COMMIT_PATTERN.fullmatch(protocol_commit):
            raise ProtocolViolation("synthetic prepare requires a 40-hex commit identity")
        attestation = type(
            "SyntheticAttestation",
            (),
            {"commit": protocol_commit, "tag": "synthetic-development"},
        )()
        canonical_provenance = {
            "mode": "synthetic-development",
            "canonical_biological_scope": False,
        }
    else:
        raise ValueError("run_mode must be biological or synthetic")
    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    source_digests, source_paths = _source_index(processed)
    donors = [animal for animal in TASK_MICE if animal != target_animal]
    support_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    for animal in TASK_MICE:
        sessions = read_normal_sessions(source_paths[animal], config=config)
        for session in sessions:
            support_name = f"support/{animal}/{_safe_key(session.session_key)}.normal-only.npz"
            support_path = destination / support_name
            support_sha = _atomic_npz(support_path, **_normal_session_arrays(session))
            row = {
                "animal_id": animal,
                "session_key": session.session_key,
                "session_id": session.session_id,
                "adapter_id": session.adapter_id,
                "neural_channels": int(session.neural.shape[2]),
                "normal_windows": int(len(session.neural)),
                "normal_sources": sorted(set(session.normal_source)),
                "path": support_name,
                "sha256": support_sha,
            }
            support_rows.append(row)
            if animal == target_animal:
                query_name = f"queries/{animal}/{_safe_key(session.session_key)}.query-inputs.npz"
                query_path = destination / query_name
                query_sha = _atomic_npz(query_path, **_query_arrays(session, config))
                query_rows.append(
                    {
                        "animal_id": animal,
                        "session_key": session.session_key,
                        "adapter_id": session.adapter_id,
                        "path": query_name,
                        "sha256": query_sha,
                    }
                )
    target_path = source_paths[target_animal].resolve()
    seal, seal_sha = _seal_target_source(
        target_path=target_path,
        processed_root=processed,
        fold_directory=destination,
        target_animal=target_animal,
        expected_sha256=source_digests[target_animal],
        canonical_relative_output=canonical_relative_output,
    )
    manifest = {
        "schema": "cadence-icms-prepare-v1",
        "dataset": f"DANDI:{DANDISET_ID}",
        "dataset_version": DANDISET_VERSION,
        "target_animal": target_animal,
        "donor_animals": donors,
        "canonical_target_order": list(TASK_MICE),
        "canonical_outer_mapping": {
            target: [animal for animal in TASK_MICE if animal != target] for target in TASK_MICE
        },
        "outer_scheme": "leave-one-animal-out",
        "run_mode": run_mode,
        "canonical_relative_output": canonical_relative_output,
        "session_unit_policy": "session-specific adapters; donor delta grouped by animal",
        "config": config.to_mapping(),
        "config_sha256": hashlib.sha256(
            json.dumps(config.to_mapping(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "intended_protocol_commit": protocol_commit,
        "required_preoutcome_tag": PREOUTCOME_TAG,
        "freeze_attestation": {
            **_freeze_mapping(attestation),
            "source_root": str(SOURCE_ROOT) if run_mode == "biological" else None,
        },
        "canonical_provenance": canonical_provenance,
        "processed_source_sha256": source_digests,
        "processed_source_paths": {
            key: str(value.resolve()) for key, value in source_paths.items()
        },
        "normal_supports": support_rows,
        "target_queries": query_rows,
        "physical_target_seal": {
            **seal,
            "sha256": seal_sha,
        },
        "target_seal_transaction_sha256": seal_sha,
        "access_audit": {
            "prepare_target_stimulation_metadata_read": False,
            "prepare_target_stimulation_signals_read": False,
            "prepare_donor_stimulation_signals_read": False,
            "target_signal_rows_read": "is_normal_calibration == true only",
            "normal_sources": ["catch", "iti"],
            "iti_guard_s": 2.0,
            "query_condition_source": (
                "prespecified 32-depth NET32 x 1--13 uA lattice; no target stimulation row accessed"
            ),
            "query_forbidden_fields": list(_FORBIDDEN_QUERY_TOKENS),
            "target_h5_read_permission_after_prepare": False,
        },
    }
    manifest_path = destination / "prepare_manifest.json"
    digest = _atomic_json(manifest_path, manifest)
    completion_path, completion_sha = _write_stage_completion(
        destination,
        stage="prepare",
        artifact_path=manifest_path,
        artifact_sha256=digest,
        freeze=_freeze_mapping(attestation),
        canonical_relative_output=canonical_relative_output,
        seal_transaction_sha256=seal_sha,
    )
    return {
        **manifest,
        "manifest_sha256": digest,
        "completion_path": str(completion_path),
        "completion_sha256": completion_sha,
        "output": str(destination),
    }


def derive_task_blocks(
    trial_indices: npt.ArrayLike,
) -> tuple[npt.NDArray[np.int64], str, bool]:
    """Derive 100-trial blocks from signal-blind trial ordering.

    Two release encodings are accepted: a monotonically increasing session
    index, or an index that resets at block boundaries.  Any ambiguous pattern
    falls back to one session-wide stratum and is disclosed in the score.
    """

    indices = np.asarray(trial_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("trial indices must be a nonempty vector")
    differences = np.diff(indices)
    if np.all(differences > 0):
        anchor = int(indices.min())
        blocks = ((indices - anchor) // 100).astype(np.int64)
        valid = all(
            len(indices[blocks == block]) <= 100 and np.ptp(indices[blocks == block]) <= 99
            for block in np.unique(blocks)
        )
        if valid:
            return blocks, "floor((trial_index - session_min) / 100)", True
    starts = np.concatenate(([0], np.flatnonzero(differences <= 0) + 1))
    stops = np.concatenate((starts[1:], [len(indices)]))
    reset_valid = len(starts) > 1
    blocks = np.empty(len(indices), dtype=np.int64)
    for block, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        segment = indices[start:stop]
        reset_valid &= (
            len(segment) <= 100 and np.all(np.diff(segment) > 0) and np.ptp(segment) <= 99
        )
        blocks[start:stop] = block
    if reset_valid:
        return blocks, "trial_index reset boundaries; <=100 ordered trials/block", True
    return (
        np.zeros(len(indices), dtype=np.int64),
        "session-level fallback: trial_index did not validate as 100-trial blocks",
        False,
    )


def _read_task_block_vector(trials: h5py.Group) -> tuple[np.ndarray, str, bool]:
    trial_index = np.asarray(trials["trial_index"], dtype=np.int64)
    if "is_iti_calibration" in trials:
        task = ~np.asarray(trials["is_iti_calibration"], dtype=bool)
    else:
        task = np.ones(len(trial_index), dtype=bool)
    task_rows = np.flatnonzero(task)
    task_blocks, rule, validated = derive_task_blocks(trial_index[task])
    blocks = np.full(len(trial_index), -1, dtype=np.int64)
    blocks[task_rows] = task_blocks
    return blocks, rule, validated


def _apply_support_scaler(
    support: NormalSession,
    neural_raw: FloatArray,
    neural_mask: BoolArray,
    behavior_raw: FloatArray,
    behavior_mask: BoolArray,
) -> tuple[FloatArray, FloatArray]:
    neural = (
        np.log1p(np.maximum(neural_raw, 0.0)) - support.neural_center[None, None, :]
    ) / support.neural_scale[None, None, :]
    behavior = (behavior_raw - support.behavior_center[None, None, :]) / support.behavior_scale[
        None, None, :
    ]
    return (
        np.where(neural_mask, neural, 0.0),
        np.where(behavior_mask, behavior, 0.0),
    )


def read_stimulation_sessions(
    animal_file: str | Path,
    supports: Mapping[str, NormalSession],
    *,
    config: ICMSExperimentConfig,
) -> list[StimSession]:
    """Open positive-current responses.

    This function is intentionally not reachable from :func:`prepare_fold`.
    Callers must first attest the tagged freeze.
    """

    sessions: list[StimSession] = []
    with h5py.File(animal_file, "r") as file:
        animal_id = str(file.attrs["animal_id"])
        time_s = np.asarray(file["time_s"], dtype=np.float64)
        for session_key in sorted(file["sessions"]):
            if session_key not in supports:
                continue
            support = supports[session_key]
            group = file["sessions"][session_key]
            trials = group["trials"]
            descriptors_all = np.asarray(group["intervention_descriptors"], dtype=np.float64)
            positive = descriptors_all[:, 0] > 0
            event_duration = _event_duration(trials)
            if event_duration is None:
                raise ProtocolViolation(
                    "canonical ICMS classification requires event start/stop times"
                )
            canonical = canonical_icms_mask(
                descriptors_all,
                event_duration_s=event_duration,
            )
            positive_count = int(positive.sum())
            excluded_noncanonical = int(np.count_nonzero(positive & ~canonical))
            indices = np.flatnonzero(canonical).astype(np.int64)
            if not len(indices):
                continue
            if config.max_stimulation_trials_per_session is not None:
                trial_index_all = np.asarray(trials["trial_index"], dtype=np.int64)
                keys = [
                    f"{animal_id}/{session_key}/stim/{trial_index_all[index]}" for index in indices
                ]
                order = _signal_blind_order(
                    keys, "icms-stimulation-cap-v1", config.seed, animal_id, session_key
                )
                indices = np.sort(indices[order[: config.max_stimulation_trials_per_session]])
            descriptors = descriptors_all[indices]
            if np.any(descriptors[:, 0] <= 0) or np.any(descriptors[:, 1] <= 0):
                raise ProtocolViolation("stimulation selection contains a zero-current row")
            signals = group["signals"]
            neural_raw = _indexed(signals["spike_rate_hz"], indices).astype(np.float64)
            neural_mask = _expand_neural_mask(
                _indexed(signals["spike_valid_mask"], indices), neural_raw.shape
            )
            displacement = _indexed(signals["wheel_displacement"], indices).astype(np.float64)
            velocity = _indexed(signals["wheel_velocity"], indices).astype(np.float64)
            wheel_mask = _indexed(signals["wheel_valid_mask"], indices).astype(bool)
            behavior_raw = np.stack((displacement, velocity), axis=-1)
            behavior_mask = np.repeat(wheel_mask[:, :, None], BEHAVIOR_DIM, axis=-1)
            neural, behavior = _apply_support_scaler(
                support, neural_raw, neural_mask, behavior_raw, behavior_mask
            )
            all_blocks, block_rule, block_validated = _read_task_block_vector(trials)
            sessions.append(
                StimSession(
                    animal_id=animal_id,
                    session_key=session_key,
                    session_id=str(group.attrs["session_id"]),
                    time_s=time_s,
                    trial_index=_indexed(trials["trial_index"], indices).astype(np.int64),
                    descriptors=descriptors,
                    neural_raw=neural_raw,
                    neural=neural,
                    neural_mask=neural_mask,
                    behavior_raw=behavior_raw,
                    behavior=behavior,
                    behavior_mask=behavior_mask,
                    blocks=all_blocks[indices],
                    block_rule=block_rule,
                    block_validated=block_validated,
                    positive_trial_count=positive_count,
                    excluded_noncanonical_count=excluded_noncanonical,
                )
            )
    return sessions


def _batch_chunks(
    indices: npt.NDArray[np.int64],
    *,
    batch_size: int,
    keys: Sequence[str],
    salt: tuple[object, ...],
) -> list[npt.NDArray[np.int64]]:
    if not len(indices):
        return []
    ordered_local = _signal_blind_order([keys[index] for index in indices], *salt)
    ordered = indices[ordered_local]
    return [
        np.sort(ordered[start : start + batch_size]) for start in range(0, len(ordered), batch_size)
    ]


def augment_neural_with_mask(
    neural: npt.ArrayLike,
    mask: npt.ArrayLike,
) -> tuple[FloatArray, BoolArray]:
    """Append explicit validity channels to every session observation.

    Masked signal values are zeroed, but the appended mask disambiguates
    "normal-support mean" from "not observed."  The mask-channel targets are
    always observed; signal targets retain their original loss mask.
    """

    values = np.asarray(neural, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if values.shape != valid.shape or values.ndim < 2:
        raise ValueError("neural values and masks must have identical shapes")
    augmented = np.concatenate(
        (np.where(valid, values, 0.0), valid.astype(np.float64)),
        axis=-1,
    )
    augmented_mask = np.concatenate(
        (valid, np.ones_like(valid, dtype=bool)),
        axis=-1,
    )
    return augmented, augmented_mask


def _normal_batches(
    sessions: Sequence[NormalSession],
    partition: Literal["fit", "val"],
    config: ICMSExperimentConfig,
) -> list[SequenceBatch]:
    result: list[SequenceBatch] = []
    for session in sessions:
        augmented_neural, augmented_mask = augment_neural_with_mask(
            session.neural, session.neural_mask
        )
        for indices in _batch_chunks(
            session.partitions[partition],
            batch_size=config.batch_size,
            keys=session.trial_keys,
            salt=("icms-normal-batch-v1", config.seed, session.adapter_id, partition),
        ):
            shape = (len(indices), len(session.time_s))
            result.append(
                SequenceBatch(
                    animal_id=session.adapter_id,
                    neural=torch.as_tensor(augmented_neural[indices], dtype=torch.float32),
                    behavior=torch.as_tensor(session.behavior[indices], dtype=torch.float32),
                    inputs=torch.zeros((*shape, INPUT_DIM), dtype=torch.float32),
                    intervention=torch.zeros((*shape, DESCRIPTOR_DIM), dtype=torch.float32),
                    onset=session.onset,
                    neural_mask=torch.as_tensor(augmented_mask[indices], dtype=torch.bool),
                    behavior_mask=torch.as_tensor(session.behavior_mask[indices], dtype=torch.bool),
                )
            )
    return result


def _stim_batches(
    sessions: Sequence[StimSession],
    *,
    config: ICMSExperimentConfig,
) -> list[SequenceBatch]:
    result: list[SequenceBatch] = []
    for session in sessions:
        augmented_neural, augmented_mask = augment_neural_with_mask(
            session.neural, session.neural_mask
        )
        keys = [
            f"{session.animal_id}/{session.session_key}/{index}" for index in session.trial_index
        ]
        all_indices = np.arange(len(keys), dtype=np.int64)
        for indices in _batch_chunks(
            all_indices,
            batch_size=config.batch_size,
            keys=keys,
            salt=("icms-stim-batch-v1", config.seed, session.adapter_id),
        ):
            inputs = torch.zeros(
                (len(indices), len(session.time_s), INPUT_DIM), dtype=torch.float32
            )
            schedule = intervention_schedule(
                session.descriptors[indices],
                session.time_s,
                train_stop_s=config.train_stop_s,
            )
            result.append(
                SequenceBatch(
                    animal_id=session.adapter_id,
                    neural=torch.as_tensor(augmented_neural[indices], dtype=torch.float32),
                    behavior=torch.as_tensor(session.behavior[indices], dtype=torch.float32),
                    inputs=inputs,
                    intervention=torch.as_tensor(schedule, dtype=torch.float32),
                    onset=session.onset,
                    neural_mask=torch.as_tensor(augmented_mask[indices], dtype=torch.bool),
                    behavior_mask=torch.as_tensor(session.behavior_mask[indices], dtype=torch.bool),
                )
            )
    return result


def _model_for_method(
    method: MethodName,
    config: ICMSExperimentConfig,
) -> HierarchicalControlledSSM:
    kwargs = {
        "latent_dim": config.latent_dim,
        "input_dim": INPUT_DIM,
        "behavior_dim": BEHAVIOR_DIM,
        "num_interventions": DESCRIPTOR_DIM,
        "hidden_dim": config.hidden_dim,
        "residual_rank": config.residual_rank,
        "intervention_rank": config.intervention_rank,
        "dt": 1.0 / 30.0,
    }
    classes: dict[str, type[HierarchicalControlledSSM]] = {
        "proposed": HierarchicalControlledSSM,
        "linear": LinearHierarchicalSSM,
        "additive": AdditiveInterventionSSM,
        "black_box": BlackBoxMetaGRU,
    }
    if method not in classes:
        raise ValueError(f"{method} is not a learned method")
    return classes[method](**kwargs)


def _inner_validation_animal(target: str) -> str:
    ordered = tuple(sorted(TASK_MICE))
    index = ordered.index(target)
    return ordered[(index + 1) % len(ordered)]


def _parameter_group_key(animal_id: str) -> str:
    return f"animal_{animal_id.replace('.', '_').replace('/', '_')}"


def _batch_animal_id(batch: SequenceBatch) -> str:
    return str(batch.animal_id).split("::", maxsplit=1)[0]


def _fit_intervention_selection(
    model: HierarchicalControlledSSM,
    train_batches: Sequence[SequenceBatch],
    validation_batches: Sequence[SequenceBatch],
    *,
    validation_animal: str,
    config: FitConfig,
) -> tuple[FitResult, dict[str, Any]]:
    """Select intervention epochs with a zero, frozen validation-animal delta."""

    if not train_batches or not validation_batches:
        raise ValueError("intervention selection batches must be nonempty")
    seed_everything(config.seed)
    model.configure_stage("intervention")
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    group_key = _parameter_group_key(validation_animal)
    if group_key not in model.donor_intervention_delta:
        raise ProtocolViolation("validation animal has no registered donor delta")
    validation_delta = model.donor_intervention_delta[group_key]
    with torch.no_grad():
        validation_delta.zero_()
    validation_delta.requires_grad_(False)
    training_animals = sorted({_batch_animal_id(batch) for batch in train_batches})
    validation_animals = sorted({_batch_animal_id(batch) for batch in validation_batches})
    if validation_animals != [validation_animal]:
        raise ProtocolViolation(
            "candidate-selection validation batches are not one held-out animal"
        )
    if validation_animal in training_animals:
        raise ProtocolViolation("validation animal appears in intervention training")
    centering_groups = tuple(sorted(_parameter_group_key(animal) for animal in training_animals))
    if group_key in centering_groups:
        raise ProtocolViolation("validation delta entered the centering group set")
    prefit_projection_norm = model.project_donor_deltas_zero_mean(centering_groups)
    if prefit_projection_norm > 1e-7:
        raise ProtocolViolation("training donor deltas failed pre-fit projection")
    train = [move_batch(batch, device) for batch in train_batches]
    validation = [move_batch(batch, device) for batch in validation_batches]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    use_amp = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    generator = random.Random(config.seed)
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[EpochRecord] = []
    stale = 0
    maximum_validation_shrinkage = 0.0
    maximum_post_step_projection_norm = 0.0
    optimizer_steps = 0
    for epoch in range(config.max_epochs):
        model.train()
        order = list(range(len(train)))
        generator.shuffle(order)
        train_losses = []
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                loss = model.intervention_loss(train[index], include_donor_delta=True)
                optimized_total = loss.total
            scaler.scale(optimized_total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            projection_norm = model.project_donor_deltas_zero_mean(centering_groups)
            maximum_post_step_projection_norm = max(
                maximum_post_step_projection_norm, projection_norm
            )
            optimizer_steps += 1
            if projection_norm > 1e-7:
                raise ProtocolViolation("training donor deltas failed exact post-step projection")
            train_losses.append(float(optimized_total.detach().cpu()))
            if torch.count_nonzero(validation_delta.detach()).item() != 0:
                raise ProtocolViolation("validation-animal delta changed during selection")
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in validation:
                # The held-out delta is absent from both the trajectory and
                # shrinkage objective during candidate selection.
                loss = model.intervention_loss(batch, include_donor_delta=False)
                validation_losses.append(loss)
                maximum_validation_shrinkage = max(
                    maximum_validation_shrinkage,
                    float(loss.residual_penalty.detach().cpu()),
                )
        train_total = float(np.mean(train_losses))
        validation_total = float(
            np.mean([float(loss.total.detach().cpu()) for loss in validation_losses])
        )
        validation_neural = float(
            np.mean([float(loss.neural.detach().cpu()) for loss in validation_losses])
        )
        validation_behavior = float(
            np.mean([float(loss.behavior.detach().cpu()) for loss in validation_losses])
        )
        history.append(
            EpochRecord(
                epoch,
                train_total,
                validation_total,
                validation_neural,
                validation_behavior,
            )
        )
        if validation_total < best_loss - 1e-8:
            best_loss = validation_total
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("intervention selection produced no finite checkpoint")
    model.load_state_dict(best_state)
    final_norm = float(validation_delta.detach().square().sum().sqrt().cpu())
    if validation_delta.requires_grad or final_norm != 0.0 or maximum_validation_shrinkage != 0.0:
        raise ProtocolViolation("held-out validation delta was not exactly frozen at zero")
    result = FitResult(
        "intervention",
        best_epoch,
        best_loss,
        history,
        config,
    )
    return result, {
        "validation_animal": validation_animal,
        "validation_delta_group": group_key,
        "validation_delta_requires_grad": False,
        "validation_delta_l2_norm": final_norm,
        "maximum_validation_delta_shrinkage_term": maximum_validation_shrinkage,
        "validation_delta_in_shrinkage": False,
        "validation_delta_centering_applied": False,
        "validation_delta_frozen_zero_during_selection": True,
        "identification_constraint": "exact_zero_mean_projection",
        "centering_group_keys": list(centering_groups),
        "centering_group_animals": training_animals,
        "centering_excluded_group_keys": [group_key],
        "centering_excluded_animals": [validation_animal],
        "prefit_projection_residual_norm": prefit_projection_norm,
        "maximum_post_step_projection_residual_norm": (maximum_post_step_projection_norm),
        "optimizer_steps": optimizer_steps,
        "projection_calls": optimizer_steps + 1,
    }


def _fixed_intervention_refit(
    model: HierarchicalControlledSSM,
    batches: Sequence[SequenceBatch],
    *,
    epochs: int,
    config: FitConfig,
) -> dict[str, Any]:
    """Refit the selected number of epochs on all five donor animals."""

    if epochs < 1 or not batches:
        raise ValueError("fixed intervention refit needs epochs and batches")
    donor_animals = sorted({_batch_animal_id(batch) for batch in batches})
    centering_groups = tuple(sorted(_parameter_group_key(animal) for animal in donor_animals))
    seed_everything(config.seed)
    model.configure_stage("intervention")
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    moved = [
        SequenceBatch(
            animal_id=batch.animal_id,
            neural=batch.neural.to(device),
            behavior=batch.behavior.to(device),
            inputs=batch.inputs.to(device),
            intervention=batch.intervention.to(device),
            onset=batch.onset,
            neural_mask=(None if batch.neural_mask is None else batch.neural_mask.to(device)),
            behavior_mask=(None if batch.behavior_mask is None else batch.behavior_mask.to(device)),
        )
        for batch in batches
    ]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    use_amp = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    generator = random.Random(config.seed)
    prefit_projection_norm = model.project_donor_deltas_zero_mean(centering_groups)
    if prefit_projection_norm > 1e-7:
        raise ProtocolViolation("all-donor deltas failed pre-fit projection")
    maximum_post_step_projection_norm = 0.0
    optimizer_steps = 0
    for _ in range(epochs):
        order = list(range(len(moved)))
        generator.shuffle(order)
        model.train()
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                loss = model.intervention_loss(moved[index], include_donor_delta=True).total
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            projection_norm = model.project_donor_deltas_zero_mean(centering_groups)
            maximum_post_step_projection_norm = max(
                maximum_post_step_projection_norm, projection_norm
            )
            optimizer_steps += 1
            if projection_norm > 1e-7:
                raise ProtocolViolation("all-donor deltas failed exact post-step projection")
    final_center = torch.stack(
        [model.donor_intervention_delta[key] for key in centering_groups]
    ).mean(dim=0)
    final_center_norm = float(final_center.detach().square().sum().sqrt().cpu())
    if final_center_norm > 1e-7:
        raise ProtocolViolation("final all-donor delta mean is not zero")
    return {
        "identification_constraint": "exact_zero_mean_projection",
        "centering_group_keys": list(centering_groups),
        "centering_group_animals": donor_animals,
        "centering_group_count": len(centering_groups),
        "final_donor_mean_delta_l2_norm": final_center_norm,
        "prefit_projection_residual_norm": prefit_projection_norm,
        "maximum_post_step_projection_residual_norm": (maximum_post_step_projection_norm),
        "optimizer_steps": optimizer_steps,
        "projection_calls": optimizer_steps + 1,
        "refit_centering_covers_every_batch_donor": True,
        "mixed_precision": use_amp,
    }


def _model_state_digest_excluding_adapters(
    model: torch.nn.Module,
    adapter_ids: Sequence[str],
) -> str:
    excluded_prefixes = tuple(
        f"adapters.{_parameter_group_key(adapter_id)}." for adapter_id in adapter_ids
    )
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(excluded_prefixes):
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _assert_only_adapter_trainable(
    model: HierarchicalControlledSSM,
    adapter_id: str,
) -> None:
    prefix = f"adapters.{_parameter_group_key(adapter_id)}."
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not trainable or any(not name.startswith(prefix) for name in trainable):
        raise ProtocolViolation(
            "normal-only adaptation exposed parameters outside the selected adapter"
        )


def _fixed_normal_refit(
    model: HierarchicalControlledSSM,
    batches: Sequence[SequenceBatch],
    *,
    epochs: int,
    config: FitConfig,
) -> dict[str, Any]:
    """Fresh fixed-epoch normal refit over all outer-fold donor animals."""

    if epochs < 1 or not batches:
        raise ValueError("fixed normal refit needs epochs and batches")
    donor_animals = sorted({_batch_animal_id(batch) for batch in batches})
    seed_everything(config.seed)
    model.configure_stage("normal")
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    moved = [move_batch(batch, device) for batch in batches]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    use_amp = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    generator = random.Random(config.seed)
    optimizer_steps = 0
    for _ in range(epochs):
        order = list(range(len(moved)))
        generator.shuffle(order)
        model.train()
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                loss = model.normal_loss(moved[index]).total
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1
    return {
        "epochs": epochs,
        "optimizer_steps": optimizer_steps,
        "normal_refit_animals": donor_animals,
        "normal_partitions": ["fit", "val"],
        "fresh_model": True,
        "mixed_precision": use_amp,
    }


def _fit_one_model(
    method: MethodName,
    donor_supports: Sequence[NormalSession],
    target_supports: Sequence[NormalSession],
    donor_stim: Sequence[StimSession],
    *,
    target_animal: str,
    config: ICMSExperimentConfig,
) -> tuple[HierarchicalControlledSSM, dict[str, Any]]:
    validation_animal = _inner_validation_animal(target_animal)
    selection_supports = [
        support for support in donor_supports if support.animal_id != validation_animal
    ]
    validation_supports = [
        support for support in donor_supports if support.animal_id == validation_animal
    ]
    selection_train = [session for session in donor_stim if session.animal_id != validation_animal]
    selection_validation = [
        session for session in donor_stim if session.animal_id == validation_animal
    ]
    if (
        not selection_supports
        or not validation_supports
        or not selection_train
        or not selection_validation
    ):
        raise ProtocolViolation("nested whole-animal selection split is empty")

    # Candidate-selection topology: F sees normal activity from only the four
    # intervention-training donors. The validation animal then receives
    # normal-only session adapters with F frozen.
    seed_everything(config.seed)
    selection_model = _model_for_method(method, config)
    for support in selection_supports:
        selection_model.register_animal(
            support.adapter_id,
            2 * support.neural.shape[2],
            donor=True,
            intervention_group=support.animal_id,
        )
    normal_selection = fit_stage(
        selection_model,
        _normal_batches(selection_supports, "fit", config),
        _normal_batches(selection_supports, "val", config),
        stage="normal",
        config=config.normal_fit,
    )
    for support in validation_supports:
        selection_model.register_animal(
            support.adapter_id,
            2 * support.neural.shape[2],
            donor=True,
            intervention_group=support.animal_id,
        )
    validation_adapter_ids = [support.adapter_id for support in validation_supports]
    shared_before_validation_adaptation = _model_state_digest_excluding_adapters(
        selection_model,
        validation_adapter_ids,
    )
    validation_normal_adaptation: dict[str, Any] = {}
    for support in validation_supports:
        selection_model.configure_stage(
            "target_adaptation",
            target_animal=support.adapter_id,
        )
        _assert_only_adapter_trainable(selection_model, support.adapter_id)
        result = fit_stage(
            selection_model,
            _normal_batches([support], "fit", config),
            _normal_batches([support], "val", config),
            stage="target_adaptation",
            target_animal=support.adapter_id,
            config=config.target_fit,
        )
        validation_normal_adaptation[support.adapter_id] = result.to_dict()
    shared_after_validation_adaptation = _model_state_digest_excluding_adapters(
        selection_model,
        validation_adapter_ids,
    )
    if shared_before_validation_adaptation != shared_after_validation_adaptation:
        raise ProtocolViolation("validation-normal adaptation changed non-adapter model state")
    intervention_selection, selection_delta_audit = _fit_intervention_selection(
        selection_model,
        _stim_batches(selection_train, config=config),
        _stim_batches(selection_validation, config=config),
        validation_animal=validation_animal,
        config=config.intervention_fit,
    )
    selected_normal_epochs = normal_selection.best_epoch + 1
    selected_epochs = intervention_selection.best_epoch + 1

    # Final topology is a fresh model. F is refit on all five donor-normal
    # supports for the selected normal epoch count, then G on all five donor
    # intervention sets for the selected intervention epoch count.
    seed_everything(config.seed + 1)
    model = _model_for_method(method, config)
    for support in donor_supports:
        model.register_animal(
            support.adapter_id,
            2 * support.neural.shape[2],
            donor=True,
            intervention_group=support.animal_id,
        )
    final_normal_batches = [
        *_normal_batches(donor_supports, "fit", config),
        *_normal_batches(donor_supports, "val", config),
    ]
    final_normal_audit = _fixed_normal_refit(
        model,
        final_normal_batches,
        epochs=selected_normal_epochs,
        config=config.normal_fit,
    )
    refit_delta_audit = _fixed_intervention_refit(
        model,
        _stim_batches(donor_stim, config=config),
        epochs=selected_epochs,
        config=config.intervention_fit,
    )
    expected_refit_animals = sorted({session.animal_id for session in donor_stim})
    if refit_delta_audit["centering_group_animals"] != expected_refit_animals:
        raise ProtocolViolation("all-donor refit centering group set is incomplete")
    for support in target_supports:
        model.register_animal(
            support.adapter_id,
            2 * support.neural.shape[2],
            donor=False,
            intervention_group=support.animal_id,
        )
    target_adapter_ids = [support.adapter_id for support in target_supports]
    state_before_target_adaptation = _model_state_digest_excluding_adapters(
        model,
        target_adapter_ids,
    )
    target_histories: dict[str, Any] = {}
    for support in target_supports:
        model.configure_stage(
            "target_adaptation",
            target_animal=support.adapter_id,
        )
        _assert_only_adapter_trainable(model, support.adapter_id)
        target_result = fit_stage(
            model,
            _normal_batches([support], "fit", config),
            _normal_batches([support], "val", config),
            stage="target_adaptation",
            target_animal=support.adapter_id,
            config=config.target_fit,
        )
        target_histories[support.adapter_id] = target_result.to_dict()
    state_after_target_adaptation = _model_state_digest_excluding_adapters(
        model,
        target_adapter_ids,
    )
    if state_before_target_adaptation != state_after_target_adaptation:
        raise ProtocolViolation("target normal-only adaptation changed non-adapter model state")
    model.configure_stage("evaluation")
    return model, {
        "normal_selection": normal_selection.to_dict(),
        "normal_selection_training_animals": sorted(
            {support.animal_id for support in selection_supports}
        ),
        "validation_normal_adaptation": validation_normal_adaptation,
        "validation_normal_gradient_to_shared_f": False,
        "shared_f_before_validation_normal_sha256": (shared_before_validation_adaptation),
        "shared_f_after_validation_normal_sha256": (shared_after_validation_adaptation),
        "shared_normal_stage_state_excluding_validation_adapters_sha256": (
            shared_after_validation_adaptation
        ),
        "intervention_selection": intervention_selection.to_dict(),
        "intervention_inner_validation_animal": validation_animal,
        "intervention_selection_delta_audit": selection_delta_audit,
        "final_model_is_fresh": True,
        "final_normal_refit_selected_epochs": selected_normal_epochs,
        "final_normal_refit": final_normal_audit,
        "intervention_refit_all_donors_epochs": selected_epochs,
        "intervention_refit_delta_audit": refit_delta_audit,
        "target_normal_only_adaptation": target_histories,
        "target_adaptation_nonadapter_state_before_sha256": (state_before_target_adaptation),
        "target_adaptation_nonadapter_state_after_sha256": (state_after_target_adaptation),
        "encoder_missingness_policy": (
            "per-unit [scaled spike, validity] channels; masked values zero-filled "
            "only alongside explicit validity"
        ),
    }


def _normal_signature(sessions: Sequence[NormalSession]) -> FloatArray:
    per_session = []
    for session in sessions:
        indices = np.concatenate((session.partitions["fit"], session.partitions["val"]))
        neural_curve = np.nanmean(
            np.where(
                session.neural_mask[indices],
                session.neural[indices],
                np.nan,
            ),
            axis=(0, 2),
        )
        behavior_curve = np.nanmean(
            np.where(
                session.behavior_mask[indices],
                session.behavior[indices],
                np.nan,
            ),
            axis=0,
        ).reshape(-1)
        per_session.append(np.nan_to_num(np.concatenate((neural_curve, behavior_curve))))
    return np.mean(np.stack(per_session), axis=0)


def _donor_template(
    donor_stim: Sequence[StimSession],
    donor_supports: Mapping[str, NormalSession],
    lattice: FloatArray,
    *,
    selected_animal: str | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Equal-session, then equal-animal condition/time effect template."""

    by_animal: dict[str, list[tuple[FloatArray, FloatArray, FloatArray]]] = {}
    for stim in donor_stim:
        if selected_animal is not None and stim.animal_id != selected_animal:
            continue
        support = donor_supports[stim.adapter_id]
        control_indices = np.concatenate((support.partitions["fit"], support.partitions["val"]))
        neural_control = np.nanmean(
            np.where(
                support.neural_mask[control_indices],
                support.neural[control_indices],
                np.nan,
            ),
            axis=0,
        )
        behavior_control = np.nanmean(
            np.where(
                support.behavior_mask[control_indices],
                support.behavior[control_indices],
                np.nan,
            ),
            axis=0,
        )
        neural_difference = stim.neural - neural_control[None, :, :]
        neural_count = stim.neural_mask.sum(axis=2)
        neural_effect = np.divide(
            np.where(stim.neural_mask, neural_difference, 0.0).sum(axis=2),
            neural_count,
            out=np.zeros(neural_count.shape, dtype=np.float64),
            where=neural_count > 0,
        )[:, stim.onset :]
        behavior_effect = np.where(
            stim.behavior_mask,
            stim.behavior - behavior_control[None, :, :],
            np.nan,
        )[:, stim.onset :]
        by_animal.setdefault(stim.animal_id, []).append(
            (stim.descriptors, neural_effect, behavior_effect)
        )
    if not by_animal:
        raise ProtocolViolation("no donor response supports a template")
    query = normalize_descriptors(lattice)
    # Current and physical depth carry the transferable condition geometry.
    columns = {name: index for index, name in enumerate(INTERVENTION_DESCRIPTOR_COLUMNS)}
    distance_columns = [
        columns["current_uA"],
        columns["electrode_depth_fraction"],
    ]
    animal_neural = []
    animal_behavior = []
    for sessions in by_animal.values():
        session_neural = []
        session_behavior = []
        for descriptors, neural_effect, behavior_effect in sessions:
            donor = normalize_descriptors(descriptors)
            distance = np.square(
                query[:, None, distance_columns] - donor[None, :, distance_columns]
            ).sum(axis=2)
            nearest = np.argmin(distance, axis=1)
            session_neural.append(neural_effect[nearest])
            session_behavior.append(behavior_effect[nearest])
        animal_neural.append(np.nanmean(np.stack(session_neural), axis=0))
        animal_behavior.append(np.nanmean(np.stack(session_behavior), axis=0))
    return (
        np.nanmean(np.stack(animal_neural), axis=0),
        np.nanmean(np.stack(animal_behavior), axis=0),
    )


def _load_query(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        query = {name: values[name] for name in values.files}
    forbidden = [name for name in query if any(token in name for token in _FORBIDDEN_QUERY_TOKENS)]
    if forbidden:
        raise ProtocolViolation(f"sealed target fields in query: {forbidden}")
    if np.any(query["condition_descriptors"][:, 0] <= 0):
        raise ProtocolViolation("query lattice contains a non-intervention row")
    return query


def _inverse_neural(values: FloatArray, query: Mapping[str, np.ndarray]) -> FloatArray:
    transformed = (
        values * query["neural_scale"][None, None, :] + query["neural_center"][None, None, :]
    )
    return np.maximum(np.expm1(np.clip(transformed, -5.0, 12.0)), 0.0)


def _inverse_behavior(values: FloatArray, query: Mapping[str, np.ndarray]) -> FloatArray:
    return values * query["behavior_scale"][None, None, :] + query["behavior_center"][None, None, :]


def _predict_learned(
    model: HierarchicalControlledSSM,
    query: Mapping[str, np.ndarray],
    *,
    uncertainty_draws: int,
    seed: int,
    include_uncertainty: bool,
) -> dict[str, FloatArray]:
    device = next(model.parameters()).device
    conditions = len(query["condition_descriptors"])
    contexts = len(query["pre_neural"])
    onset = int(query["onset"])
    pre = onset - 1
    augmented_pre_neural, _ = augment_neural_with_mask(
        query["pre_neural"], query["pre_neural_mask"]
    )
    neural = np.tile(augmented_pre_neural, (conditions, 1))
    behavior = np.tile(query["pre_behavior"], (conditions, 1))
    treated_intervention = np.repeat(query["treated_intervention"], contexts, axis=0)
    inputs = np.repeat(query["control_inputs"], contexts, axis=0)
    with torch.no_grad():
        model.eval()
        z0, _ = model.encode(
            str(query["adapter_id"]),
            torch.as_tensor(neural, dtype=torch.float32, device=device),
            torch.as_tensor(behavior, dtype=torch.float32, device=device),
            sample=False,
        )
        treated = model.rollout(
            str(query["adapter_id"]),
            z0,
            torch.as_tensor(inputs[:, pre:-1], dtype=torch.float32, device=device),
            torch.as_tensor(
                treated_intervention[:, pre:-1],
                dtype=torch.float32,
                device=device,
            ),
            include_animal_residual=True,
            include_donor_delta=False,
        )
        control = model.rollout(
            str(query["adapter_id"]),
            z0,
            torch.as_tensor(inputs[:, pre:-1], dtype=torch.float32, device=device),
            torch.zeros(
                (conditions * contexts, len(query["time_s"]) - onset, DESCRIPTOR_DIM),
                dtype=torch.float32,
                device=device,
            ),
            include_animal_residual=True,
            include_donor_delta=False,
        )
    horizon = len(query["time_s"]) - onset
    units = query["pre_neural"].shape[1]
    treated_neural_scaled = (
        treated[1]
        .detach()
        .cpu()
        .numpy()
        .reshape(conditions, contexts, horizon, 2 * units)[..., :units]
    )
    control_neural_scaled = (
        control[1]
        .detach()
        .cpu()
        .numpy()
        .reshape(conditions, contexts, horizon, 2 * units)[..., :units]
    )
    treated_behavior_scaled = (
        treated[2].detach().cpu().numpy().reshape(conditions, contexts, horizon, BEHAVIOR_DIM)
    )
    control_behavior_scaled = (
        control[2].detach().cpu().numpy().reshape(conditions, contexts, horizon, BEHAVIOR_DIM)
    )
    neural_treated = np.mean(
        _inverse_neural(treated_neural_scaled.reshape(-1, horizon, units), query).reshape(
            conditions, contexts, horizon, units
        ),
        axis=1,
    )
    neural_control = np.mean(
        _inverse_neural(control_neural_scaled.reshape(-1, horizon, units), query).reshape(
            conditions, contexts, horizon, units
        ),
        axis=1,
    )
    behavior_treated = np.mean(
        _inverse_behavior(
            treated_behavior_scaled.reshape(-1, horizon, BEHAVIOR_DIM), query
        ).reshape(conditions, contexts, horizon, BEHAVIOR_DIM),
        axis=1,
    )
    behavior_control = np.mean(
        _inverse_behavior(
            control_behavior_scaled.reshape(-1, horizon, BEHAVIOR_DIM), query
        ).reshape(conditions, contexts, horizon, BEHAVIOR_DIM),
        axis=1,
    )
    output = {
        "neural_treated": neural_treated,
        "neural_control": neural_control,
        "neural_effect": neural_treated - neural_control,
        "behavior_treated": behavior_treated,
        "behavior_control": behavior_control,
        "behavior_effect": behavior_treated - behavior_control,
    }
    if include_uncertainty:
        samples = sample_target_intervention_residual(
            model,
            str(query["adapter_id"]),
            z0,
            torch.as_tensor(inputs[:, pre:-1], dtype=torch.float32, device=device),
            torch.as_tensor(
                treated_intervention[:, pre:-1],
                dtype=torch.float32,
                device=device,
            ),
            draws=uncertainty_draws,
            seed=seed,
        )
        neural_draws_scaled = (
            samples.neural.detach()
            .cpu()
            .numpy()
            .reshape(
                uncertainty_draws,
                conditions,
                contexts,
                horizon,
                2 * units,
            )[..., :units]
        )
        behavior_draws_scaled = (
            samples.behavior.detach()
            .cpu()
            .numpy()
            .reshape(
                uncertainty_draws,
                conditions,
                contexts,
                horizon,
                BEHAVIOR_DIM,
            )
        )
        neural_draws = []
        behavior_draws = []
        for draw in range(uncertainty_draws):
            decoded_neural = _inverse_neural(
                neural_draws_scaled[draw].reshape(-1, horizon, units), query
            ).reshape(conditions, contexts, horizon, units)
            decoded_behavior = _inverse_behavior(
                behavior_draws_scaled[draw].reshape(-1, horizon, BEHAVIOR_DIM),
                query,
            ).reshape(conditions, contexts, horizon, BEHAVIOR_DIM)
            neural_draws.append(
                np.mean(decoded_neural, axis=(1, 3)) - np.mean(neural_control, axis=2)
            )
            behavior_draws.append(np.mean(decoded_behavior, axis=1) - behavior_control)
        output["neural_effect_draws_condition_time"] = np.stack(neural_draws)
        output["behavior_effect_draws_condition_time"] = np.stack(behavior_draws)
    return output


def _template_prediction(
    query: Mapping[str, np.ndarray],
    neural_effect: FloatArray,
    behavior_effect: FloatArray,
) -> dict[str, FloatArray]:
    control_neural = np.broadcast_to(
        query["normal_control_neural"][None, :, :],
        (len(neural_effect), *query["normal_control_neural"].shape),
    ).copy()
    control_behavior = np.broadcast_to(
        query["normal_control_behavior"][None, :, :],
        (len(behavior_effect), *query["normal_control_behavior"].shape),
    ).copy()
    transformed = np.log1p(np.maximum(control_neural, 0.0))
    standardized = (transformed - query["neural_center"][None, None, :]) / query["neural_scale"][
        None, None, :
    ]
    treated_neural = np.maximum(
        np.expm1(
            np.clip(
                (standardized + neural_effect[:, :, None]) * query["neural_scale"][None, None, :]
                + query["neural_center"][None, None, :],
                -5.0,
                12.0,
            )
        ),
        0.0,
    )
    treated_behavior = (
        (control_behavior - query["behavior_center"][None, None, :])
        / query["behavior_scale"][None, None, :]
        + behavior_effect
    ) * query["behavior_scale"][None, None, :] + query["behavior_center"][None, None, :]
    return {
        "neural_treated": treated_neural,
        "neural_control": control_neural,
        "neural_effect": treated_neural - control_neural,
        "behavior_treated": treated_behavior,
        "behavior_control": control_behavior,
        "behavior_effect": treated_behavior - control_behavior,
    }


def _zero_prediction(query: Mapping[str, np.ndarray]) -> dict[str, FloatArray]:
    conditions = len(query["condition_descriptors"])
    neural = np.broadcast_to(
        query["normal_control_neural"][None, :, :],
        (conditions, *query["normal_control_neural"].shape),
    ).copy()
    behavior = np.broadcast_to(
        query["normal_control_behavior"][None, :, :],
        (conditions, *query["normal_control_behavior"].shape),
    ).copy()
    return {
        "neural_treated": neural,
        "neural_control": neural.copy(),
        "neural_effect": np.zeros_like(neural),
        "behavior_treated": behavior,
        "behavior_control": behavior.copy(),
        "behavior_effect": np.zeros_like(behavior),
    }


def _verify_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists():
        raise ProtocolViolation(f"hash sidecar is missing: {sidecar}")
    tokens = sidecar.read_text(encoding="utf-8").strip().split()
    if len(tokens) != 2 or tokens[1] != path.name:
        raise ProtocolViolation(f"malformed hash sidecar: {sidecar}")
    _verify_artifact(path, tokens[0])
    return tokens[0]


def _load_prepared_fold(
    directory: Path,
) -> tuple[dict[str, Any], dict[str, NormalSession], dict[str, Path]]:
    manifest_path = directory / "prepare_manifest.json"
    _verify_sidecar(manifest_path)
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != "cadence-icms-prepare-v1":
        raise ProtocolViolation("unknown ICMS prepare manifest schema")
    supports: dict[str, NormalSession] = {}
    for row in manifest["normal_supports"]:
        path = directory / row["path"]
        _verify_artifact(path, row["sha256"])
        session = load_normal_session(path)
        if session.adapter_id != row["adapter_id"]:
            raise ProtocolViolation("support identity differs from immutable manifest")
        supports[session.adapter_id] = session
    query_paths: dict[str, Path] = {}
    for row in manifest["target_queries"]:
        path = directory / row["path"]
        _verify_artifact(path, row["sha256"])
        query_paths[row["adapter_id"]] = path
    return manifest, supports, query_paths


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or _hash_sidecar(path).exists():
        raise FileExistsError(f"append-only model artifact already exists: {path}")
    temporary = _temporary_output_path(path)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        _publish_without_replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    digest = hash_file(path)
    _write_hash_sidecar(path, digest)
    return digest


def predict_fold(
    *,
    fold_directory: str | Path,
    config: ICMSExperimentConfig,
    methods: Sequence[MethodName] = REPORT_METHODS,
    acknowledge_donor_outcomes: bool = False,
    run_mode: RunMode = "biological",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fit on donors and hash target predictions without opening target ICMS.

    A clean worktree at the exact ``pre-outcome-v1.0.0`` tag is a hard
    prerequisite.  The acknowledgement is checked before any donor response
    dataset is opened.
    """

    if not acknowledge_donor_outcomes:
        raise ProtocolViolation(
            "donor stimulation outcomes remain closed; pass the explicit "
            "acknowledgement only after the pre-outcome freeze"
        )
    if run_mode == "biological":
        if overwrite:
            raise ProtocolViolation("biological ICMS stages are append-only")
        _require_biological_config(config)
        _require_biological_methods(methods)
        attestation = _attest_source_freeze()
    elif run_mode == "synthetic":
        attestation = None
    else:
        raise ValueError("run_mode must be biological or synthetic")
    directory = Path(fold_directory)
    prepare_completion = _verify_stage_completion(directory, "prepare")
    prepare_manifest_path = directory / "prepare_manifest.json"
    _verify_sidecar(prepare_manifest_path)
    manifest = _load_json(prepare_manifest_path)
    if manifest.get("run_mode") != run_mode:
        raise ProtocolViolation("predict run mode differs from prepared fold")
    canonical_relative_output = manifest.get("canonical_relative_output")
    if run_mode == "biological":
        expected_output = _require_canonical_biological_output(
            directory,
            str(manifest.get("target_animal", "")),
        )
        if canonical_relative_output != expected_output:
            raise ProtocolViolation("ICMS prepare canonical output binding changed")
        if prepare_completion.get("canonical_relative_output") != expected_output:
            raise ProtocolViolation("ICMS prepare completion output binding changed")
        if prepare_completion.get("seal_transaction_sha256") != manifest.get(
            "target_seal_transaction_sha256"
        ):
            raise ProtocolViolation("ICMS prepare seal-transaction binding changed")
        recovery = _recover_icms_stage(
            directory=directory,
            prepare_manifest=manifest,
            stage="predict",
            canonical_relative_output=expected_output,
        )
        if recovery is not None and (directory / "prediction_manifest.json").is_file():
            prediction_manifest = _load_json(directory / "prediction_manifest.json")
            return {
                **prediction_manifest,
                "prediction_manifest_sha256": _verify_sidecar(
                    directory / "prediction_manifest.json"
                ),
                "completion_path": str(directory / "predict_complete.json"),
                "completion_sha256": _verify_sidecar(directory / "predict_complete.json"),
                "output": str(directory),
            }
    elif canonical_relative_output is not None:
        raise ProtocolViolation("synthetic ICMS prepare must not claim a canonical output")
    manifest, supports, query_paths = _load_prepared_fold(directory)
    if run_mode == "synthetic":
        attestation = type(
            "SyntheticAttestation",
            (),
            {
                "commit": manifest["intended_protocol_commit"],
                "tag": "synthetic-development",
            },
        )()
    if manifest["intended_protocol_commit"] != attestation.commit:
        raise ProtocolViolation("prepared fold targets a different commit than the tagged freeze")
    freeze_mapping = _freeze_mapping(attestation)
    if prepare_completion["freeze_attestation"] != freeze_mapping:
        raise ProtocolViolation("prepare completion freeze identity changed")
    expected_prepare_freeze = {
        **freeze_mapping,
        "source_root": str(SOURCE_ROOT) if run_mode == "biological" else None,
    }
    if manifest["freeze_attestation"] != expected_prepare_freeze:
        raise ProtocolViolation("prepare manifest freeze identity changed")
    if run_mode == "biological":
        _canonical_provenance(
            Path(next(iter(manifest["processed_source_paths"].values()))).parent,
            commit=attestation.commit,
            verify_h5=False,
        )
    if (
        manifest["config_sha256"]
        != hashlib.sha256(
            json.dumps(config.to_mapping(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    ):
        raise ProtocolViolation("prediction configuration differs from prepared query")
    selected = tuple(dict.fromkeys(methods))
    if not selected or not set(selected).issubset(REPORT_METHODS):
        raise ValueError(f"methods must be drawn from {REPORT_METHODS}")
    prediction_path = directory / "predictions.npz"
    prediction_manifest_path = directory / "prediction_manifest.json"
    model_path = directory / "frozen_models.pt"
    writable_prediction_artifacts = (
        prediction_path,
        prediction_manifest_path,
        model_path,
    )
    protected_prediction_artifacts = (
        directory / "predict_complete.json",
        directory / "metrics.json",
        directory / "score_complete.json",
        directory / "sealed_target_outcomes.npz",
        directory / "scored_condition_trajectories.npz",
        directory / "condition_metrics.csv",
        directory / "target_restore.json",
    )
    post_prepare_artifacts = _artifact_paths_with_sidecars(
        (*writable_prediction_artifacts, *protected_prediction_artifacts)
    )
    existing = [path for path in post_prepare_artifacts if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"prediction artifacts already exist: {existing[0]}")
    if overwrite:
        protected = [
            path
            for path in _artifact_paths_with_sidecars(protected_prediction_artifacts)
            if path.exists()
        ]
        if protected:
            raise FileExistsError(
                f"completed or downstream prediction artifact exists: {protected[0]}"
            )
        for path in _artifact_paths_with_sidecars(writable_prediction_artifacts):
            path.unlink(missing_ok=True)

    target_animal = str(manifest["target_animal"])
    donor_animals = tuple(str(value) for value in manifest["donor_animals"])
    donor_supports = [
        support for support in supports.values() if support.animal_id in donor_animals
    ]
    target_supports = [
        support for support in supports.values() if support.animal_id == target_animal
    ]
    source_paths = {
        animal: Path(path) for animal, path in manifest["processed_source_paths"].items()
    }
    physical_seal = _assert_target_source_sealed(directory, manifest)
    # Only donor files are content-verified here.  Hashing/opening the target
    # container is deferred to the acknowledged score stage.
    verified_donor_sources: dict[str, str] = {}
    donor_stim: list[StimSession] = []
    support_by_animal_session = {support.animal_id: {} for support in donor_supports}
    for support in donor_supports:
        support_by_animal_session[support.animal_id][support.session_key] = support
    for animal in donor_animals:
        observed_digest = hash_file(source_paths[animal])
        expected_digest = manifest["processed_source_sha256"][animal]
        if observed_digest != expected_digest:
            raise ProtocolViolation(f"processed donor file changed for {animal}")
        verified_donor_sources[animal] = observed_digest
        donor_stim.extend(
            read_stimulation_sessions(
                source_paths[animal],
                support_by_animal_session[animal],
                config=config,
            )
        )
    if {session.animal_id for session in donor_stim} != set(donor_animals):
        raise ProtocolViolation("every donor animal must contribute stimulation trials")

    donor_support_map = {support.adapter_id: support for support in donor_supports}
    lattice = physical_query_lattice(config)
    template_neural, template_behavior = _donor_template(donor_stim, donor_support_map, lattice)
    donor_signatures = {
        animal: _normal_signature(
            [support for support in donor_supports if support.animal_id == animal]
        )
        for animal in donor_animals
    }

    models: dict[str, HierarchicalControlledSSM] = {}
    fit_audits: dict[str, Any] = {}
    for method in selected:
        if method not in LEARNED_METHODS:
            continue
        model, audit = _fit_one_model(
            method,
            donor_supports,
            target_supports,
            donor_stim,
            target_animal=target_animal,
            config=config,
        )
        models[method] = model
        fit_audits[method] = audit

    prediction_arrays: dict[str, np.ndarray] = {}
    session_records: list[dict[str, Any]] = []
    for session_number, support in enumerate(target_supports):
        query_path = query_paths[support.adapter_id]
        query = _load_query(query_path)
        key = _safe_key(support.adapter_id)
        prediction_arrays[f"{key}__condition_descriptors"] = query["condition_descriptors"]
        target_signature = _normal_signature([support])
        nearest_animal = min(
            donor_animals,
            key=lambda animal: float(
                np.mean(np.square(donor_signatures[animal] - target_signature))
            ),
        )
        nearest_neural, nearest_behavior = _donor_template(
            donor_stim,
            donor_support_map,
            lattice,
            selected_animal=nearest_animal,
        )
        method_outputs: dict[str, dict[str, FloatArray]] = {}
        for method in selected:
            if method in LEARNED_METHODS:
                method_outputs[method] = _predict_learned(
                    models[method],
                    query,
                    uncertainty_draws=config.uncertainty_draws,
                    seed=config.seed + session_number * 101,
                    include_uncertainty=method == "proposed",
                )
            elif method == "zero_effect":
                method_outputs[method] = _zero_prediction(query)
            elif method == "condition_time":
                method_outputs[method] = _template_prediction(
                    query, template_neural, template_behavior
                )
            elif method == "nearest_donor":
                method_outputs[method] = _template_prediction(
                    query, nearest_neural, nearest_behavior
                )
            else:
                raise AssertionError(f"unhandled method {method}")
            for name, values in method_outputs[method].items():
                prediction_arrays[f"{method}__{key}__{name}"] = np.asarray(values)
        session_records.append(
            {
                "adapter_id": support.adapter_id,
                "session_key": support.session_key,
                "session_id": support.session_id,
                "unit_ids": support.unit_ids.tolist(),
                "condition_count": int(len(lattice)),
                "current_lattice_uA": list(config.current_grid_uA),
                "nearest_donor": nearest_animal,
                "array_key": key,
                "query_sha256": hash_file(query_path),
            }
        )

    model_payload = {
        "schema": "cadence-icms-models-v1",
        "target_animal": target_animal,
        "attestation": freeze_mapping,
        "config": config.to_mapping(),
        "methods": {
            method: {
                "state_dict": model.state_dict(),
                "fit_audit": fit_audits[method],
            }
            for method, model in models.items()
        },
    }
    model_sha = _atomic_torch_save(model_path, model_payload)
    prediction_sha = _atomic_npz(prediction_path, **prediction_arrays)
    prediction_manifest = {
        "schema": "cadence-icms-prediction-v1",
        "dataset": f"DANDI:{DANDISET_ID}",
        "dataset_version": DANDISET_VERSION,
        "target_animal": target_animal,
        "donor_animals": list(donor_animals),
        "methods": list(selected),
        "sessions": session_records,
        "config": config.to_mapping(),
        "config_sha256": manifest["config_sha256"],
        "prepare_manifest_sha256": hash_file(directory / "prepare_manifest.json"),
        "prediction_path": prediction_path.name,
        "prediction_sha256_before_target_open": prediction_sha,
        "model_path": model_path.name,
        "model_sha256": model_sha,
        "freeze_attestation": freeze_mapping,
        "run_mode": run_mode,
        "canonical_relative_output": canonical_relative_output,
        "target_seal_transaction_sha256": manifest["target_seal_transaction_sha256"],
        "canonical_scope": {
            "full_profile": config.profile == "full",
            "seed": config.seed,
            "ordered_report_methods": list(selected),
            "canonical_target_order": manifest["canonical_target_order"],
            "outer_mapping": manifest["canonical_outer_mapping"],
            "processed_index_sha256": manifest["canonical_provenance"].get("provided_index_sha256"),
            "raw_asset_manifest_sha256": manifest["canonical_provenance"]
            .get("raw_asset_manifest", {})
            .get("sha256"),
        },
        "verified_donor_source_sha256": verified_donor_sources,
        "target_source_sha256_expected_but_not_opened": manifest["processed_source_sha256"][
            target_animal
        ],
        "fit_audits": fit_audits,
        "access_audit": {
            "donor_stimulation_metadata_read": True,
            "donor_stimulation_outcomes_read": True,
            "donor_positive_trials_seen": int(
                sum(session.positive_trial_count for session in donor_stim)
            ),
            "donor_canonical_trials_used": int(
                sum(len(session.descriptors) for session in donor_stim)
            ),
            "donor_noncanonical_trains_excluded": int(
                sum(session.excluded_noncanonical_count for session in donor_stim)
            ),
            "primary_train_family": ("70 pulses, 100 Hz, 167 us, 0.700 s +/- 0.010 s"),
            "noncanonical_long_train_sensitivity": "OUT_OF_SCOPE_NOT_SCORED",
            "target_stimulation_metadata_read": False,
            "target_stimulation_outcomes_read": False,
            "target_adapter_sources": ["zero-current catch", "guarded ITI"],
            "target_stimulation_trials_in_fit_or_validation": 0,
            "target_condition_grid_source": (
                "prespecified NET32/current lattice from prepare manifest"
            ),
            "prediction_hashed_before_target_container_open": True,
            "physical_target_seal_asserted_before_donor_open": True,
            "physical_target_seal_sha256": manifest["physical_target_seal"]["sha256"],
            "physical_target_h5_mode_during_predict": physical_seal["sealed_mode"],
            "session_specific_observation_maps": True,
            "encoder_receives_explicit_missingness_channels": True,
            "zero_filled_missing_bins_without_mask_channel": False,
            "donor_delta_grouping": "animal_id",
            "inner_validation_unit": "whole donor animal",
            "inner_validation_delta_policy": (
                "present, exactly zero, requires_grad=false; excluded from "
                "rollout, shrinkage, and training-donor centering"
            ),
            "donor_delta_centering_policy": (
                "exact zero-mean projection across four training groups in inner "
                "selection and all five donor groups in fixed refit"
            ),
            "pooled_raw_unit_baseline_available": False,
            "pooled_raw_unit_baseline_reason": (
                "unit identity and channel count vary by session; no stable raw "
                "population can be concatenated"
            ),
        },
        "uncertainty": {
            "proposed": (
                "donor random-effect draws via sample_target_intervention_residual; "
                "full condition-time population and behavior summaries retained"
            ),
            "draws": config.uncertainty_draws,
            "split_conformal": "ABSENT_NOT_FIT",
            "donor_draw_quantiles": ("uncalibrated marginal 5--95%; not simultaneous or conformal"),
        },
    }
    manifest_sha = _atomic_json(prediction_manifest_path, prediction_manifest)
    completion_path, completion_sha = _write_stage_completion(
        directory,
        stage="predict",
        artifact_path=prediction_manifest_path,
        artifact_sha256=manifest_sha,
        freeze=freeze_mapping,
        canonical_relative_output=canonical_relative_output,
        seal_transaction_sha256=manifest["target_seal_transaction_sha256"],
    )
    return {
        **prediction_manifest,
        "prediction_manifest_sha256": manifest_sha,
        "completion_path": str(completion_path),
        "completion_sha256": completion_sha,
        "output": str(directory),
    }


@dataclass(slots=True)
class OutcomeSession:
    """Target outcomes opened only after frozen-prediction verification."""

    animal_id: str
    session_key: str
    session_id: str
    time_s: FloatArray
    descriptors: FloatArray
    trial_index: npt.NDArray[np.int64]
    blocks: npt.NDArray[np.int64]
    neural: FloatArray
    neural_mask: BoolArray
    behavior: FloatArray
    behavior_mask: BoolArray
    catch_blocks: npt.NDArray[np.int64]
    catch_neural: FloatArray
    catch_neural_mask: BoolArray
    catch_behavior: FloatArray
    catch_behavior_mask: BoolArray
    iti_neural: FloatArray
    iti_neural_mask: BoolArray
    iti_behavior: FloatArray
    iti_behavior_mask: BoolArray
    block_rule: str
    block_validated: bool
    positive_trial_count: int
    excluded_noncanonical_count: int

    @property
    def adapter_id(self) -> str:
        return f"{self.animal_id}::{self.session_key}"


def _read_outcome_rows(
    signals: h5py.Group,
    indices: npt.NDArray[np.int64],
    onset: int,
) -> tuple[FloatArray, BoolArray, FloatArray, BoolArray]:
    if not len(indices):
        time = signals["spike_rate_hz"].shape[1] - onset
        units = signals["spike_rate_hz"].shape[2]
        return (
            np.empty((0, time, units), dtype=np.float64),
            np.empty((0, time, units), dtype=bool),
            np.empty((0, time, BEHAVIOR_DIM), dtype=np.float64),
            np.empty((0, time, BEHAVIOR_DIM), dtype=bool),
        )
    neural = _indexed(signals["spike_rate_hz"], indices)[:, onset:].astype(np.float64)
    raw_neural_mask = _indexed(signals["spike_valid_mask"], indices)[:, onset:]
    neural_mask = _expand_neural_mask(raw_neural_mask, neural.shape)
    displacement = _indexed(signals["wheel_displacement"], indices)[:, onset:].astype(np.float64)
    velocity = _indexed(signals["wheel_velocity"], indices)[:, onset:].astype(np.float64)
    wheel_mask = _indexed(signals["wheel_valid_mask"], indices)[:, onset:].astype(bool)
    behavior = np.stack((displacement, velocity), axis=-1)
    behavior_mask = np.repeat(wheel_mask[:, :, None], BEHAVIOR_DIM, axis=-1)
    return neural, neural_mask, behavior, behavior_mask


def materialize_target_outcomes(
    *,
    animal_file: str | Path,
    target_supports: Mapping[str, NormalSession],
    destination: str | Path,
) -> tuple[list[OutcomeSession], str, dict[str, Any]]:
    """Write a physically separate, post-onset-only target outcome bundle."""

    arrays: dict[str, np.ndarray] = {}
    sessions: list[OutcomeSession] = []
    audit_sessions: list[dict[str, Any]] = []
    with h5py.File(animal_file, "r") as file:
        animal_id = str(file.attrs["animal_id"])
        time_s = np.asarray(file["time_s"], dtype=np.float64)
        for session_key in sorted(file["sessions"]):
            if session_key not in target_supports:
                continue
            support = target_supports[session_key]
            onset = support.onset
            group = file["sessions"][session_key]
            trials = group["trials"]
            descriptors_all = np.asarray(group["intervention_descriptors"], dtype=np.float64)
            positive = descriptors_all[:, 0] > 0
            event_duration = _event_duration(trials)
            if event_duration is None:
                raise ProtocolViolation(
                    "canonical ICMS classification requires event start/stop times"
                )
            canonical = canonical_icms_mask(
                descriptors_all,
                event_duration_s=event_duration,
            )
            stimulated = np.flatnonzero(canonical).astype(np.int64)
            positive_count = int(positive.sum())
            excluded_noncanonical = int(np.count_nonzero(positive & ~canonical))
            normal = np.asarray(trials["is_normal_calibration"], dtype=bool)
            is_catch = np.asarray(trials["is_catch"], dtype=bool)
            is_iti = np.asarray(trials["is_iti_calibration"], dtype=bool)
            catch = np.flatnonzero(normal & is_catch).astype(np.int64)
            iti = np.flatnonzero(normal & is_iti).astype(np.int64)
            all_blocks, rule, validated = _read_task_block_vector(trials)
            signals = group["signals"]
            neural, neural_mask, behavior, behavior_mask = _read_outcome_rows(
                signals, stimulated, onset
            )
            (
                catch_neural,
                catch_neural_mask,
                catch_behavior,
                catch_behavior_mask,
            ) = _read_outcome_rows(signals, catch, onset)
            iti_neural, iti_neural_mask, iti_behavior, iti_behavior_mask = _read_outcome_rows(
                signals, iti, onset
            )
            outcome = OutcomeSession(
                animal_id=animal_id,
                session_key=session_key,
                session_id=str(group.attrs["session_id"]),
                time_s=time_s[onset:],
                descriptors=descriptors_all[stimulated],
                trial_index=_indexed(trials["trial_index"], stimulated).astype(np.int64),
                blocks=all_blocks[stimulated],
                neural=neural,
                neural_mask=neural_mask,
                behavior=behavior,
                behavior_mask=behavior_mask,
                catch_blocks=all_blocks[catch],
                catch_neural=catch_neural,
                catch_neural_mask=catch_neural_mask,
                catch_behavior=catch_behavior,
                catch_behavior_mask=catch_behavior_mask,
                iti_neural=iti_neural,
                iti_neural_mask=iti_neural_mask,
                iti_behavior=iti_behavior,
                iti_behavior_mask=iti_behavior_mask,
                block_rule=rule,
                block_validated=validated,
                positive_trial_count=positive_count,
                excluded_noncanonical_count=excluded_noncanonical,
            )
            sessions.append(outcome)
            key = _safe_key(outcome.adapter_id)
            for name in (
                "descriptors",
                "trial_index",
                "blocks",
                "neural",
                "neural_mask",
                "behavior",
                "behavior_mask",
                "catch_blocks",
                "catch_neural",
                "catch_neural_mask",
                "catch_behavior",
                "catch_behavior_mask",
                "iti_neural",
                "iti_neural_mask",
                "iti_behavior",
                "iti_behavior_mask",
                "time_s",
            ):
                arrays[f"{key}__{name}"] = np.asarray(getattr(outcome, name))
            audit_sessions.append(
                {
                    "adapter_id": outcome.adapter_id,
                    "positive_stimulation_trials": positive_count,
                    "canonical_stimulation_trials_scored": int(len(stimulated)),
                    "noncanonical_trains_excluded": excluded_noncanonical,
                    "noncanonical_long_train_sensitivity": "OUT_OF_SCOPE_NOT_SCORED",
                    "randomized_catch_trials": int(len(catch)),
                    "guarded_iti_windows": int(len(iti)),
                    "block_rule": rule,
                    "block_validated": validated,
                }
            )
    if not sessions:
        raise ProtocolViolation("target animal has no stimulation outcomes")
    digest = _atomic_npz(Path(destination), **arrays)
    return (
        sessions,
        digest,
        {
            "sessions": audit_sessions,
            "post_onset_only": True,
            "target_outcomes_physically_separate_from_queries": True,
        },
    )


def _condition_groups(descriptors: FloatArray) -> list[npt.NDArray[np.int64]]:
    rounded = np.round(descriptors, 6)
    keys = [tuple(row) for row in rounded]
    return [
        np.asarray([index for index, observed in enumerate(keys) if observed == key])
        for key in sorted(set(keys))
    ]


def _masked_mean(
    values: FloatArray,
    mask: BoolArray,
    *,
    axis: int = 0,
) -> tuple[FloatArray, BoolArray]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(values)
    numerator = np.where(valid, values, 0.0).sum(axis=axis)
    count = valid.sum(axis=axis)
    return np.divide(
        numerator,
        count,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=count > 0,
    ), count > 0


def _lattice_interpolation(
    values: FloatArray,
    lattice: FloatArray,
    descriptor: FloatArray,
    *,
    condition_axis: int = 0,
) -> FloatArray:
    """Interpolate a frozen lattice prediction to one continuous current."""

    if not canonical_icms_mask(np.asarray(descriptor)[None, :])[0]:
        raise ProtocolViolation("noncanonical ICMS train cannot be mapped onto the primary lattice")
    columns = {name: index for index, name in enumerate(INTERVENTION_DESCRIPTOR_COLUMNS)}
    depth_column = columns["electrode_rel_y_um"]
    current_column = columns["current_uA"]
    depth = float(descriptor[depth_column])
    depths = np.unique(lattice[:, depth_column])
    nearest_depth = float(depths[np.argmin(np.abs(depths - depth))])
    if abs(nearest_depth - depth) > 1e-5:
        raise ProtocolViolation(
            f"target depth {depth} is absent from the prespecified NET32 lattice"
        )
    rows = np.flatnonzero(np.isclose(lattice[:, depth_column], nearest_depth))
    currents = lattice[rows, current_column]
    order = np.argsort(currents)
    rows = rows[order]
    currents = currents[order]
    current = float(descriptor[current_column])
    if current < currents[0] - 1e-8 or current > currents[-1] + 1e-8:
        raise ProtocolViolation(f"target current {current} lies outside frozen 1--13 uA support")
    right = int(np.searchsorted(currents, current, side="left"))
    if right < len(currents) and np.isclose(currents[right], current, atol=1e-8):
        return np.take(values, rows[right], axis=condition_axis)
    left = max(0, right - 1)
    right = min(right, len(currents) - 1)
    weight = (current - currents[left]) / (currents[right] - currents[left])
    low = np.take(values, rows[left], axis=condition_axis)
    high = np.take(values, rows[right], axis=condition_axis)
    return (1.0 - weight) * low + weight * high


def _observed_condition(
    outcome: OutcomeSession,
    indices: npt.NDArray[np.int64],
) -> dict[str, Any]:
    neural, neural_mask = _masked_mean(outcome.neural[indices], outcome.neural_mask[indices])
    behavior, behavior_mask = _masked_mean(
        outcome.behavior[indices], outcome.behavior_mask[indices]
    )
    catch_supported = len(outcome.catch_neural) > 0
    neural_block_effects = []
    neural_block_masks = []
    behavior_block_effects = []
    behavior_block_masks = []
    missing_same_block_catches = 0
    primary_block_supported = bool(catch_supported and outcome.block_validated)
    if primary_block_supported:
        for block in np.unique(outcome.blocks[indices]):
            stimulated = indices[outcome.blocks[indices] == block]
            catches = np.flatnonzero(outcome.catch_blocks == block)
            if not len(catches):
                missing_same_block_catches += 1
                primary_block_supported = False
                continue
            stim_neural, stim_neural_mask = _masked_mean(
                outcome.neural[stimulated], outcome.neural_mask[stimulated]
            )
            control_neural, control_neural_mask = _masked_mean(
                outcome.catch_neural[catches], outcome.catch_neural_mask[catches]
            )
            stim_behavior, stim_behavior_mask = _masked_mean(
                outcome.behavior[stimulated], outcome.behavior_mask[stimulated]
            )
            control_behavior, control_behavior_mask = _masked_mean(
                outcome.catch_behavior[catches], outcome.catch_behavior_mask[catches]
            )
            neural_block_effects.append(stim_neural - control_neural)
            neural_block_masks.append(stim_neural_mask & control_neural_mask)
            behavior_block_effects.append(stim_behavior - control_behavior)
            behavior_block_masks.append(stim_behavior_mask & control_behavior_mask)
    valid_primary_neural_cells = int(sum(np.count_nonzero(mask) for mask in neural_block_masks))
    valid_primary_behavior_cells = int(sum(np.count_nonzero(mask) for mask in behavior_block_masks))
    primary_block_supported = bool(
        primary_block_supported
        and neural_block_effects
        and valid_primary_neural_cells > 0
        and valid_primary_behavior_cells > 0
    )
    if primary_block_supported and neural_block_effects:
        neural_effect, neural_effect_mask = _masked_mean(
            np.stack(neural_block_effects), np.stack(neural_block_masks)
        )
        behavior_effect, behavior_effect_mask = _masked_mean(
            np.stack(behavior_block_effects), np.stack(behavior_block_masks)
        )
    else:
        neural_effect = np.full_like(neural, np.nan)
        neural_effect_mask = np.zeros_like(neural_mask)
        behavior_effect = np.full_like(behavior, np.nan)
        behavior_effect_mask = np.zeros_like(behavior_mask)
    if catch_supported:
        session_neural, session_neural_mask = _masked_mean(
            outcome.catch_neural, outcome.catch_neural_mask
        )
        session_behavior, session_behavior_mask = _masked_mean(
            outcome.catch_behavior, outcome.catch_behavior_mask
        )
        session_fallback_neural_effect = neural - session_neural
        session_fallback_neural_effect_mask = neural_mask & session_neural_mask
        session_fallback_behavior_effect = behavior - session_behavior
        session_fallback_behavior_effect_mask = behavior_mask & session_behavior_mask
    else:
        session_fallback_neural_effect = np.full_like(neural, np.nan)
        session_fallback_neural_effect_mask = np.zeros_like(neural_mask)
        session_fallback_behavior_effect = np.full_like(behavior, np.nan)
        session_fallback_behavior_effect_mask = np.zeros_like(behavior_mask)
    if len(outcome.iti_neural):
        iti_neural, iti_neural_mask = _masked_mean(outcome.iti_neural, outcome.iti_neural_mask)
        iti_behavior, iti_behavior_mask = _masked_mean(
            outcome.iti_behavior, outcome.iti_behavior_mask
        )
        iti_neural_effect = neural - iti_neural
        iti_neural_effect_mask = neural_mask & iti_neural_mask
        iti_behavior_effect = behavior - iti_behavior
        iti_behavior_effect_mask = behavior_mask & iti_behavior_mask
    else:
        iti_neural_effect = np.full_like(neural, np.nan)
        iti_neural_effect_mask = np.zeros_like(neural_mask)
        iti_behavior_effect = np.full_like(behavior, np.nan)
        iti_behavior_effect_mask = np.zeros_like(behavior_mask)
    return {
        "neural": neural,
        "neural_mask": neural_mask,
        "behavior": behavior,
        "behavior_mask": behavior_mask,
        "neural_effect": neural_effect,
        "neural_effect_mask": neural_effect_mask,
        "behavior_effect": behavior_effect,
        "behavior_effect_mask": behavior_effect_mask,
        "session_fallback_neural_effect": session_fallback_neural_effect,
        "session_fallback_neural_effect_mask": (session_fallback_neural_effect_mask),
        "session_fallback_behavior_effect": session_fallback_behavior_effect,
        "session_fallback_behavior_effect_mask": (session_fallback_behavior_effect_mask),
        "iti_neural_effect": iti_neural_effect,
        "iti_neural_effect_mask": iti_neural_effect_mask,
        "iti_behavior_effect": iti_behavior_effect,
        "iti_behavior_effect_mask": iti_behavior_effect_mask,
        "catch_supported": catch_supported,
        "block_validated": outcome.block_validated,
        "same_block_catch_supported": primary_block_supported,
        "primary_randomized_status": ("EVALUATED" if primary_block_supported else "NOT_EVALUATED"),
        "missing_same_block_catches": missing_same_block_catches,
        "valid_primary_neural_cells": valid_primary_neural_cells,
        "valid_primary_behavior_cells": valid_primary_behavior_cells,
    }


def _safe_nrmse(
    prediction: FloatArray,
    target: FloatArray,
    mask: BoolArray,
    scale: FloatArray,
) -> float:
    return trajectory_nrmse(
        np.where(mask, prediction, np.nan),
        np.where(mask, target, np.nan),
        channel_scale=scale,
    )


def _safe_time_r2(
    prediction: FloatArray,
    target: FloatArray,
    mask: BoolArray,
) -> FloatArray:
    predicted = np.where(mask, prediction, np.nan)
    observed = np.where(mask, target, np.nan)
    output = np.full(predicted.shape[1], np.nan, dtype=np.float64)
    for time in range(output.size):
        pred = np.take(predicted, time, axis=1).reshape(-1)
        obs = np.take(observed, time, axis=1).reshape(-1)
        valid = np.isfinite(pred) & np.isfinite(obs)
        if not np.any(valid):
            continue
        centered = obs[valid] - np.mean(obs[valid])
        denominator = np.dot(centered, centered)
        if denominator > np.finfo(float).eps:
            output[time] = 1 - np.square(pred[valid] - obs[valid]).sum() / denominator
    return output


def _mean_finite(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")


def score_fold(
    *,
    fold_directory: str | Path,
    acknowledge_target_outcomes: bool = False,
    run_mode: RunMode = "biological",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Verify frozen predictions, then open and score target responses once."""

    if not acknowledge_target_outcomes:
        raise ProtocolViolation(
            "target stimulation outcomes remain sealed; pass the explicit score "
            "acknowledgement only after predictions are frozen"
        )
    if run_mode == "biological":
        if overwrite:
            raise ProtocolViolation("biological ICMS stages are append-only")
        attestation = _attest_source_freeze()
    elif run_mode == "synthetic":
        attestation = None
    else:
        raise ValueError("run_mode must be biological or synthetic")
    directory = Path(fold_directory)
    prepare_completion = _verify_stage_completion(directory, "prepare")
    predict_completion = _verify_stage_completion(directory, "predict")
    prepare_manifest_path = directory / "prepare_manifest.json"
    _verify_sidecar(prepare_manifest_path)
    prepare_manifest = _load_json(prepare_manifest_path)
    prediction_manifest_path = directory / "prediction_manifest.json"
    _verify_sidecar(prediction_manifest_path)
    prediction_manifest = _load_json(prediction_manifest_path)
    if prediction_manifest.get("schema") != "cadence-icms-prediction-v1":
        raise ProtocolViolation("unknown ICMS prediction manifest")
    if (
        prepare_manifest.get("run_mode") != run_mode
        or prediction_manifest.get("run_mode") != run_mode
    ):
        raise ProtocolViolation("score run mode differs from earlier stages")
    canonical_relative_output = prepare_manifest.get("canonical_relative_output")
    if run_mode == "biological":
        expected_output = _require_canonical_biological_output(
            directory,
            str(prepare_manifest.get("target_animal", "")),
        )
        if (
            canonical_relative_output != expected_output
            or prediction_manifest.get("canonical_relative_output") != expected_output
            or prepare_completion.get("canonical_relative_output") != expected_output
            or predict_completion.get("canonical_relative_output") != expected_output
            or prepare_manifest.get("target_seal_transaction_sha256")
            != prediction_manifest.get("target_seal_transaction_sha256")
            or prepare_completion.get("seal_transaction_sha256")
            != prepare_manifest.get("target_seal_transaction_sha256")
            or predict_completion.get("seal_transaction_sha256")
            != prepare_manifest.get("target_seal_transaction_sha256")
        ):
            raise ProtocolViolation("ICMS stage canonical output binding changed")
        recovery = _recover_icms_stage(
            directory=directory,
            prepare_manifest=prepare_manifest,
            stage="score",
            canonical_relative_output=expected_output,
        )
        if recovery == "score_complete":
            metrics = _load_json(directory / "metrics.json")
            return {
                **metrics,
                "metrics_sha256": _verify_sidecar(directory / "metrics.json"),
                "completion_path": str(directory / "score_complete.json"),
                "completion_sha256": _verify_sidecar(directory / "score_complete.json"),
                "restoration_completion": _load_json(directory / "target_restore_complete.json"),
                "output": str(directory),
            }
    elif (
        canonical_relative_output is not None
        or prediction_manifest.get("canonical_relative_output") is not None
    ):
        raise ProtocolViolation("synthetic ICMS stages must not claim a canonical output")
    prepare_manifest, supports, _ = _load_prepared_fold(directory)
    if run_mode == "synthetic":
        attestation = type(
            "SyntheticAttestation",
            (),
            {
                "commit": prepare_manifest["intended_protocol_commit"],
                "tag": "synthetic-development",
            },
        )()
    freeze_mapping = _freeze_mapping(attestation)
    if (
        prepare_completion["freeze_attestation"] != freeze_mapping
        or predict_completion["freeze_attestation"] != freeze_mapping
    ):
        raise ProtocolViolation("stage completion freeze identity changed")
    if prediction_manifest["freeze_attestation"] != freeze_mapping:
        raise ProtocolViolation("score checkout differs from prediction freeze")
    if run_mode == "biological":
        _require_biological_methods(prediction_manifest["methods"])
        _canonical_provenance(
            Path(next(iter(prepare_manifest["processed_source_paths"].values()))).parent,
            commit=attestation.commit,
            verify_h5=False,
        )
    if prediction_manifest["prepare_manifest_sha256"] != hash_file(
        directory / "prepare_manifest.json"
    ):
        raise ProtocolViolation("prepare manifest changed after prediction")
    prediction_path = directory / prediction_manifest["prediction_path"]
    _verify_artifact(
        prediction_path,
        prediction_manifest["prediction_sha256_before_target_open"],
    )
    model_path = directory / prediction_manifest["model_path"]
    _verify_artifact(model_path, prediction_manifest["model_sha256"])
    # All immutable prediction checks above precede restoring or hashing target H5.
    target = str(prediction_manifest["target_animal"])
    metrics_path = directory / "metrics.json"
    sealed_path = directory / "sealed_target_outcomes.npz"
    scored_path = directory / "scored_condition_trajectories.npz"
    condition_csv = directory / "condition_metrics.csv"
    writable_score_artifacts = (
        metrics_path,
        sealed_path,
        scored_path,
        condition_csv,
        directory / "target_restore.json",
    )
    protected_score_artifacts = (directory / "score_complete.json",)
    existing_score_artifacts = [
        path
        for path in _artifact_paths_with_sidecars(
            (*writable_score_artifacts, *protected_score_artifacts)
        )
        if path.exists()
    ]
    if existing_score_artifacts:
        if not overwrite:
            raise FileExistsError("score artifacts already exist")
        protected = [
            path
            for path in _artifact_paths_with_sidecars(protected_score_artifacts)
            if path.exists()
        ]
        if protected:
            raise FileExistsError("completed score artifacts already exist")
        for path in _artifact_paths_with_sidecars(writable_score_artifacts):
            path.unlink(missing_ok=True)
    target_path, restore_audit, restore_audit_sha, target_digest = _restore_target_source(
        directory,
        prepare_manifest,
        canonical_relative_output=canonical_relative_output,
    )

    target_supports = {
        support.session_key: support for support in supports.values() if support.animal_id == target
    }
    outcomes, sealed_sha, outcome_audit = materialize_target_outcomes(
        animal_file=target_path,
        target_supports=target_supports,
        destination=sealed_path,
    )
    with np.load(prediction_path, allow_pickle=False) as values:
        predictions = {name: values[name] for name in values.files}
    methods = [str(value) for value in prediction_manifest["methods"]]
    session_manifest = {row["adapter_id"]: row for row in prediction_manifest["sessions"]}
    support_by_adapter = {support.adapter_id: support for support in target_supports.values()}
    session_scores: dict[str, dict[str, Any]] = {}
    condition_rows: list[dict[str, Any]] = []
    scored_arrays: dict[str, np.ndarray] = {}
    session_primary_eligibility: dict[str, bool] = {}
    for outcome in outcomes:
        adapter_id = outcome.adapter_id
        if adapter_id not in session_manifest:
            raise ProtocolViolation("target outcome session lacks a frozen prediction")
        record = session_manifest[adapter_id]
        key = record["array_key"]
        lattice = predictions[f"{key}__condition_descriptors"]
        support = support_by_adapter[adapter_id]
        post = support.onset
        scale_rows = support.partitions["fit"]
        neural_scale = support_scale(
            np.where(
                support.neural_mask[scale_rows, post:],
                support.neural_raw[scale_rows, post:],
                np.nan,
            )
        )
        behavior_scale = support_scale(
            np.where(
                support.behavior_mask[scale_rows, post:],
                support.behavior_raw[scale_rows, post:],
                np.nan,
            )
        )
        groups = _condition_groups(outcome.descriptors)
        observed = [_observed_condition(outcome, indices) for indices in groups]
        observed_neural = np.stack([entry["neural"] for entry in observed])
        observed_neural_mask = np.stack([entry["neural_mask"] for entry in observed])
        observed_behavior = np.stack([entry["behavior"] for entry in observed])
        observed_behavior_mask = np.stack([entry["behavior_mask"] for entry in observed])
        observed_neural_effect = np.stack([entry["neural_effect"] for entry in observed])
        observed_neural_effect_mask = np.stack([entry["neural_effect_mask"] for entry in observed])
        observed_behavior_effect = np.stack([entry["behavior_effect"] for entry in observed])
        observed_behavior_effect_mask = np.stack(
            [entry["behavior_effect_mask"] for entry in observed]
        )
        observed_session_fallback_neural_effect = np.stack(
            [entry["session_fallback_neural_effect"] for entry in observed]
        )
        observed_session_fallback_neural_mask = np.stack(
            [entry["session_fallback_neural_effect_mask"] for entry in observed]
        )
        observed_session_fallback_behavior_effect = np.stack(
            [entry["session_fallback_behavior_effect"] for entry in observed]
        )
        observed_session_fallback_behavior_mask = np.stack(
            [entry["session_fallback_behavior_effect_mask"] for entry in observed]
        )
        observed_iti_neural_effect = np.stack([entry["iti_neural_effect"] for entry in observed])
        observed_iti_neural_mask = np.stack([entry["iti_neural_effect_mask"] for entry in observed])
        observed_iti_behavior_effect = np.stack(
            [entry["iti_behavior_effect"] for entry in observed]
        )
        observed_iti_behavior_mask = np.stack(
            [entry["iti_behavior_effect_mask"] for entry in observed]
        )
        descriptors = np.stack([outcome.descriptors[indices[0]] for indices in groups])
        session_primary_eligible = bool(
            target != "ICMS83"
            and outcome.block_validated
            and observed
            and all(entry["same_block_catch_supported"] for entry in observed)
        )
        session_primary_eligibility[adapter_id] = session_primary_eligible
        session_scores[adapter_id] = {}
        scored_arrays[f"{key}__observed_neural"] = observed_neural
        scored_arrays[f"{key}__observed_neural_mask"] = observed_neural_mask
        scored_arrays[f"{key}__observed_behavior"] = observed_behavior
        scored_arrays[f"{key}__observed_behavior_mask"] = observed_behavior_mask
        scored_arrays[f"{key}__observed_neural_effect"] = observed_neural_effect
        scored_arrays[f"{key}__observed_behavior_effect"] = observed_behavior_effect
        scored_arrays[f"{key}__condition_descriptors"] = descriptors
        for method in methods:
            prefix = f"{method}__{key}"
            aligned = {
                name: np.stack(
                    [
                        _lattice_interpolation(
                            predictions[f"{prefix}__{name}"],
                            lattice,
                            descriptor,
                        )
                        for descriptor in descriptors
                    ]
                )
                for name in (
                    "neural_treated",
                    "neural_control",
                    "neural_effect",
                    "behavior_treated",
                    "behavior_control",
                    "behavior_effect",
                )
            }
            for name, values in aligned.items():
                scored_arrays[f"{method}__{key}__{name}"] = values
            absolute_neural_nrmse = _safe_nrmse(
                aligned["neural_treated"],
                observed_neural,
                observed_neural_mask,
                neural_scale,
            )
            absolute_behavior_nrmse = _safe_nrmse(
                aligned["behavior_treated"],
                observed_behavior,
                observed_behavior_mask,
                behavior_scale,
            )
            randomized_eligible = session_primary_eligible
            if randomized_eligible:
                neural_skill = causal_skill(
                    aligned["neural_effect"],
                    observed_neural_effect,
                    channel_scale=neural_scale,
                    mask=observed_neural_effect_mask,
                )
                behavior_skill = causal_skill(
                    aligned["behavior_effect"],
                    observed_behavior_effect,
                    channel_scale=behavior_scale,
                    mask=observed_behavior_effect_mask,
                )
                neural_r2 = _safe_time_r2(
                    aligned["neural_effect"],
                    observed_neural_effect,
                    observed_neural_effect_mask,
                )
                behavior_r2 = _safe_time_r2(
                    aligned["behavior_effect"],
                    observed_behavior_effect,
                    observed_behavior_effect_mask,
                )
            else:
                neural_skill = float("nan")
                behavior_skill = float("nan")
                neural_r2 = np.full(observed_neural.shape[1], np.nan)
                behavior_r2 = np.full(observed_behavior.shape[1], np.nan)
            iti_eligible = bool(np.any(observed_iti_neural_mask))
            iti_neural_skill = (
                causal_skill(
                    aligned["neural_effect"],
                    observed_iti_neural_effect,
                    channel_scale=neural_scale,
                    mask=observed_iti_neural_mask,
                )
                if iti_eligible
                else float("nan")
            )
            iti_behavior_skill = (
                causal_skill(
                    aligned["behavior_effect"],
                    observed_iti_behavior_effect,
                    channel_scale=behavior_scale,
                    mask=observed_iti_behavior_mask,
                )
                if bool(np.any(observed_iti_behavior_mask))
                else float("nan")
            )
            session_fallback_eligible = bool(np.any(observed_session_fallback_neural_mask))
            session_fallback_neural_skill = (
                causal_skill(
                    aligned["neural_effect"],
                    observed_session_fallback_neural_effect,
                    channel_scale=neural_scale,
                    mask=observed_session_fallback_neural_mask,
                )
                if session_fallback_eligible
                else float("nan")
            )
            session_fallback_behavior_skill = (
                causal_skill(
                    aligned["behavior_effect"],
                    observed_session_fallback_behavior_effect,
                    channel_scale=behavior_scale,
                    mask=observed_session_fallback_behavior_mask,
                )
                if bool(np.any(observed_session_fallback_behavior_mask))
                else float("nan")
            )
            method_score: dict[str, Any] = {
                "absolute_neural_nrmse": absolute_neural_nrmse,
                "absolute_behavior_nrmse": absolute_behavior_nrmse,
                "randomized_causal_eligible": randomized_eligible,
                "primary_randomized_status": (
                    "EVALUATED" if randomized_eligible else "NOT_EVALUATED"
                ),
                "neural_causal_skill": neural_skill,
                "behavior_causal_skill": behavior_skill,
                "neural_time_r2": neural_r2,
                "behavior_time_r2": behavior_r2,
                "nonrandomized_iti_sensitivity_eligible": iti_eligible,
                "nonrandomized_iti_neural_skill": iti_neural_skill,
                "nonrandomized_iti_behavior_skill": iti_behavior_skill,
                "nonprimary_session_fallback_eligible": (session_fallback_eligible),
                "nonprimary_session_fallback_neural_skill": (session_fallback_neural_skill),
                "nonprimary_session_fallback_behavior_skill": (session_fallback_behavior_skill),
            }
            draw_neural_key = f"{prefix}__neural_effect_draws_condition_time"
            draw_behavior_key = f"{prefix}__behavior_effect_draws_condition_time"
            if draw_neural_key in predictions and randomized_eligible:
                neural_draws = np.stack(
                    [
                        np.stack(
                            [
                                _lattice_interpolation(
                                    predictions[draw_neural_key],
                                    lattice,
                                    descriptor,
                                    condition_axis=1,
                                )
                                for descriptor in descriptors
                            ]
                        )
                        for _ in [0]
                    ]
                )[0]
                # Interpolation above yields [condition, draw, time].
                neural_draws = np.moveaxis(neural_draws, 1, 0)
                behavior_draws = np.stack(
                    [
                        _lattice_interpolation(
                            predictions[draw_behavior_key],
                            lattice,
                            descriptor,
                            condition_axis=1,
                        )
                        for descriptor in descriptors
                    ]
                )
                behavior_draws = np.moveaxis(behavior_draws, 1, 0)
                observed_population, observed_population_mask = _masked_mean(
                    observed_neural_effect,
                    observed_neural_effect_mask,
                    axis=2,
                )
                low, high = np.quantile(neural_draws, [0.05, 0.95], axis=0)
                coverage = interval_coverage(low, high, observed_population)
                method_score["neural_population_energy_score"] = (
                    energy_score(
                        neural_draws[:, observed_population_mask],
                        observed_population[observed_population_mask],
                    )
                    if np.any(observed_population_mask)
                    else float("nan")
                )
                method_score["uncalibrated_marginal_neural_90_interval"] = {
                    "pointwise_coverage": coverage[0],
                    "whole_trajectory_containment_diagnostic": coverage[1],
                    "mean_width": coverage[2],
                    "calibrated": False,
                }
                behavior_energy_mask = (
                    observed_behavior_effect_mask
                    & np.isfinite(observed_behavior_effect)
                    & np.all(np.isfinite(behavior_draws), axis=0)
                )
                method_score["behavior_energy_score"] = (
                    energy_score(
                        behavior_draws[:, behavior_energy_mask],
                        observed_behavior_effect[behavior_energy_mask],
                    )
                    if np.any(behavior_energy_mask)
                    else float("nan")
                )
                behavior_low, behavior_high = np.quantile(behavior_draws, [0.05, 0.95], axis=0)
                behavior_coverage = interval_coverage(
                    behavior_low,
                    behavior_high,
                    observed_behavior_effect,
                )
                method_score["uncalibrated_marginal_behavior_90_interval"] = {
                    "pointwise_coverage": behavior_coverage[0],
                    "whole_trajectory_containment_diagnostic": (behavior_coverage[1]),
                    "mean_width": behavior_coverage[2],
                    "calibrated": False,
                }
            session_scores[adapter_id][method] = method_score
            for condition_number, (indices, descriptor, entry) in enumerate(
                zip(groups, descriptors, observed, strict=True)
            ):
                condition_rows.append(
                    {
                        "animal_id": target,
                        "session_id": outcome.session_id,
                        "session_key": outcome.session_key,
                        "method": method,
                        "condition": condition_number,
                        "current_uA": float(descriptor[1]),
                        "electrode_rel_y_um": float(descriptor[6]),
                        "trials": int(len(indices)),
                        "randomized_causal_eligible": randomized_eligible,
                        "condition_primary_randomized_status": entry["primary_randomized_status"],
                        "session_primary_randomized_status": (
                            "EVALUATED" if session_primary_eligible else "NOT_EVALUATED"
                        ),
                        "block_validated": outcome.block_validated,
                        "block_rule": outcome.block_rule,
                        "missing_same_block_catches": int(entry["missing_same_block_catches"]),
                        "valid_primary_neural_cells": int(entry["valid_primary_neural_cells"]),
                        "valid_primary_behavior_cells": int(entry["valid_primary_behavior_cells"]),
                        "observed_neural_effect_rms": float(
                            np.sqrt(np.nanmean(np.square(entry["neural_effect"])))
                        )
                        if randomized_eligible
                        else np.nan,
                        "predicted_neural_effect_rms": float(
                            np.sqrt(
                                np.nanmean(np.square(aligned["neural_effect"][condition_number]))
                            )
                        ),
                    }
                )

    scored_sha = _atomic_npz(scored_path, **scored_arrays)
    condition_frame = pd.DataFrame(condition_rows)
    condition_csv_sha = _atomic_csv(condition_csv, condition_frame)
    fold_primary_eligible = bool(
        target != "ICMS83"
        and len(session_primary_eligibility) == len(outcomes)
        and all(session_primary_eligibility.values())
    )
    aggregate: dict[str, Any] = {}
    for method in methods:
        per_session = [scores[method] for scores in session_scores.values() if method in scores]
        aggregate[method] = {
            "absolute_neural_nrmse_equal_session": _mean_finite(
                score["absolute_neural_nrmse"] for score in per_session
            ),
            "absolute_behavior_nrmse_equal_session": _mean_finite(
                score["absolute_behavior_nrmse"] for score in per_session
            ),
            "neural_causal_skill_equal_session": _mean_finite(
                score["neural_causal_skill"] for score in per_session
            ),
            "behavior_causal_skill_equal_session": _mean_finite(
                score["behavior_causal_skill"] for score in per_session
            ),
            "nonrandomized_iti_neural_skill_equal_session": _mean_finite(
                score["nonrandomized_iti_neural_skill"] for score in per_session
            ),
            "nonprimary_session_fallback_neural_skill_equal_session": _mean_finite(
                score["nonprimary_session_fallback_neural_skill"] for score in per_session
            ),
            "eligible_sessions": int(
                sum(score["randomized_causal_eligible"] for score in per_session)
            ),
            "primary_fold_status": ("EVALUATED" if fold_primary_eligible else "NOT_EVALUATED"),
        }
        if not fold_primary_eligible:
            aggregate[method]["neural_causal_skill_equal_session"] = float("nan")
            aggregate[method]["behavior_causal_skill_equal_session"] = float("nan")
        optional_metrics = {
            "neural_energy_score": [
                score["neural_population_energy_score"]
                for score in per_session
                if "neural_population_energy_score" in score
            ],
            "behavior_energy_score": [
                score["behavior_energy_score"]
                for score in per_session
                if "behavior_energy_score" in score
            ],
            "uncalibrated_marginal_neural_pointwise_coverage": [
                float(score["uncalibrated_marginal_neural_90_interval"]["pointwise_coverage"])
                for score in per_session
                if "uncalibrated_marginal_neural_90_interval" in score
            ],
            "uncalibrated_marginal_behavior_pointwise_coverage": [
                float(score["uncalibrated_marginal_behavior_90_interval"]["pointwise_coverage"])
                for score in per_session
                if "uncalibrated_marginal_behavior_90_interval" in score
            ],
        }
        for name, values in optional_metrics.items():
            if values:
                aggregate[method][name] = _mean_finite(values)
    metrics = {
        "schema": "cadence-icms-score-v1",
        "dataset": f"DANDI:{DANDISET_ID}",
        "dataset_version": DANDISET_VERSION,
        "target_animal": target,
        "run_mode": run_mode,
        "canonical_relative_output": canonical_relative_output,
        "target_seal_transaction_sha256": prepare_manifest["target_seal_transaction_sha256"],
        "freeze_attestation": freeze_mapping,
        "canonical_scope": prediction_manifest["canonical_scope"],
        "prediction_sha256_verified_before_target_open": prediction_manifest[
            "prediction_sha256_before_target_open"
        ],
        "target_source_sha256_verified_after_acknowledgement": target_digest,
        "sealed_outcomes_sha256": sealed_sha,
        "scored_condition_trajectories_sha256": scored_sha,
        "condition_metrics_sha256": condition_csv_sha,
        "physical_target_restore": restore_audit,
        "physical_target_restore_sha256": restore_audit_sha,
        "session_scores": session_scores,
        "animal_aggregate": aggregate,
        "causal_effect_eligibility": {
            "animal_eligible": fold_primary_eligible,
            "primary_fold_status": ("EVALUATED" if fold_primary_eligible else "NOT_EVALUATED"),
            "reason": (
                "validated 100-trial blocks and same-block catches for every condition/session"
                if fold_primary_eligible
                else (
                    "primary block estimand failed: ICMS83 lacks catches, a task "
                    "block was unvalidated, a condition lacked same-block catches, "
                    "or a primary outcome domain had no valid cells"
                )
            ),
            "design_maximum_primary_eligible_n": 5,
            "this_fold_primary_eligible_n": int(fold_primary_eligible),
            "session_status": {
                key: ("EVALUATED" if value else "NOT_EVALUATED")
                for key, value in session_primary_eligibility.items()
            },
            "session_catch_fallback_in_primary": False,
            "absolute_trajectory_n": 6,
            "iti_is_randomized_counterfactual": False,
        },
        "aggregation": (
            "conditions within session; sessions equally within animal; "
            "downstream fold combination must weight animals equally"
        ),
        "outcome_audit": outcome_audit,
        "uncertainty_audit": {
            "split_conformal": "ABSENT_NOT_FIT",
            "donor_draw_interval": "uncalibrated_marginal_5_95_quantiles",
            "simultaneous_coverage_exported": False,
            "conformal_coverage_exported": False,
        },
        "access_audit": {
            "query_and_support_hashes_verified_before_target_open": True,
            "prediction_hash_verified_before_target_open": True,
            "model_hash_verified_before_target_open": True,
            "target_stimulation_metadata_read_in_fit_or_predict": False,
            "target_stimulation_outcomes_read_in_fit_or_predict": False,
            "target_outcomes_opened_only_in_acknowledged_score": True,
            "target_outcome_bundle_post_onset_only": True,
            "physical_query_outcome_separation": True,
            "target_h5_original_mode_restored_exactly": restore_audit[
                "original_mode_restored_exactly"
            ],
            "target_h5_restored_mode": restore_audit["restored_mode"],
            "immutable_target_seal_sha256": restore_audit["immutable_seal_sha256"],
            "sequential_loao_next_fold_ready": True,
            "primary_artifact_exclusion_s": [0.0, 0.705],
            "scoring_channel_scales_fit_partition": "normal_fit",
            "distinct_unmasked_full_train_spike_tensor_available": False,
            "full_train_spike_sensitivity_status": "NOT_EVALUATED",
            "calcium_read_or_scored_by_v1_experiment": False,
            "sparse_calcium_secondary_status": "NOT_EVALUATED",
        },
    }
    metrics_sha = _atomic_json(metrics_path, metrics)
    completion_path, completion_sha = _write_stage_completion(
        directory,
        stage="score",
        artifact_path=metrics_path,
        artifact_sha256=metrics_sha,
        freeze=freeze_mapping,
        canonical_relative_output=canonical_relative_output,
        seal_transaction_sha256=prepare_manifest["target_seal_transaction_sha256"],
    )
    restoration_completion = _finalize_icms_target_restore(
        directory,
        prepare_manifest,
        canonical_relative_output=canonical_relative_output,
    )
    return {
        **metrics,
        "metrics_sha256": metrics_sha,
        "completion_path": str(completion_path),
        "completion_sha256": completion_sha,
        "restoration_completion": restoration_completion,
        "output": str(directory),
    }
