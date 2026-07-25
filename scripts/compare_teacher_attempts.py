#!/usr/bin/env python3
"""Authenticate and compare two completed locked teacher-world attempts.

The frozen teacher producer writes SHA-256 sidecars for ``metrics.json`` and
``predictions.npz`` and binds both digests in ``completion.json``. It does not
write a sidecar for ``completion.json`` itself. Accordingly, this audit reports
the completion digest as observed but not independently sidecar-authenticated.

No attempt is selected as preferred. A successful comparison means that:

* both attempts independently satisfy the frozen artifact/hash contract;
* their schema, world, configuration, and canonical identities match;
* their prediction archives are byte-identical and have matching metadata,
  keys, shapes, and dtypes;
* their metrics match after removing only the top-level ``wall_seconds`` and
  each exact ``stage_fits.<method>.wall_seconds`` field; and
* their completions match after removing only
  ``artifacts["metrics.json"]``.

The command emits deterministic JSON to stdout. It exits 0 for a match, 1 for
an authenticated discrepancy, and 2 when either attempt is invalid. This is a
retry-equivalence audit, not a second implementation of the frozen scientific
eligibility validator: exact cohort seeds, world IDs, method sets, and
scientific configuration fingerprints remain the responsibility of the frozen
reporter used for final aggregation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import zipfile
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple, NoReturn

import numpy as np

AUDIT_SCHEMA = "cadence.teacher_attempt_comparison.v1"
METRICS_SCHEMA = "cadence.teacher_experiment.v1"
PREDICTION_SCHEMA = "cadence.teacher_prediction.v1"
COMPLETION_SCHEMA = "cadence.teacher_completion.v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

METRICS_NAME = "metrics.json"
PREDICTIONS_NAME = "predictions.npz"
COMPLETION_NAME = "completion.json"
ARTIFACT_NAMES = (METRICS_NAME, PREDICTIONS_NAME)


class AuditError(ValueError):
    """An attempt failed authentication or canonical identity validation."""

    def __init__(self, code: str, artifact: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.artifact = artifact
        self.detail = detail


class AttemptData(NamedTuple):
    """Authenticated private payloads plus their public digest-only summary."""

    metrics: dict[str, Any]
    prediction_metadata: dict[str, Any]
    prediction_keys: tuple[str, ...]
    prediction_array_schema: dict[str, dict[str, Any]]
    completion: dict[str, Any]
    identity: dict[str, Any]
    summary: dict[str, Any]
    prediction_bytes_sha256: str
    prediction_path: Path


def _fail(code: str, artifact: str, detail: str) -> NoReturn:
    raise AuditError(code, artifact, detail)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path, artifact: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(2**20), b""):
                digest.update(block)
    except OSError:
        _fail("artifact_read_failed", artifact, f"cannot read {artifact}")
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        _fail("non_canonical_json_value", "payload", str(error))
    return text.encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _parse_json(raw: bytes | str, artifact: str) -> Any:
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        _fail("invalid_json", artifact, f"{artifact} is not strict UTF-8 JSON: {error}")


def _read_artifact(directory: Path, name: str) -> tuple[bytes, str]:
    path = directory / name
    try:
        if not path.is_file():
            _fail("missing_artifact", name, f"required artifact {name} is missing")
        raw = path.read_bytes()
    except OSError:
        _fail("artifact_read_failed", name, f"cannot read {name}")
    return raw, _sha256_bytes(raw)


def _digest_artifact(directory: Path, name: str) -> str:
    path = directory / name
    try:
        if not path.is_file():
            _fail("missing_artifact", name, f"required artifact {name} is missing")
    except OSError:
        _fail("artifact_read_failed", name, f"cannot inspect {name}")
    return _sha256_file(path, name)


def _verify_sidecar(directory: Path, name: str, digest: str) -> None:
    sidecar_name = f"{name}.sha256"
    try:
        text = (directory / sidecar_name).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        _fail("sidecar_read_failed", name, f"cannot read {sidecar_name}")
    expected = f"{digest}  {name}\n"
    if text != expected:
        _fail(
            "sidecar_mismatch",
            name,
            f"{sidecar_name} does not exactly bind the observed {name} digest",
        )


def _require_mapping(value: Any, artifact: str, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("invalid_structure", artifact, f"{field} must be a JSON object")
    return value


def _require_object(value: Any, artifact: str, field: str) -> dict[str, Any]:
    return dict(_require_mapping(value, artifact, field))


def _require_string(value: Any, artifact: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("invalid_identity", artifact, f"{field} must be a non-empty string")
    return value


def _require_sha256(value: Any, artifact: str, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        _fail("invalid_digest", artifact, f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_finite_number(value: Any, artifact: str, field: str) -> int | float:
    try:
        valid = (
            not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)
        )
    except OverflowError:
        valid = False
    if not valid:
        _fail("invalid_runtime_field", artifact, f"{field} must be a finite number")
    return value


def _require_equal(
    values: Mapping[str, Any],
    artifact: str,
    field: str,
) -> Any:
    entries = list(values.items())
    if not entries:
        _fail("invalid_identity", artifact, f"{field} has no identity sources")
    reference = _canonical_bytes(entries[0][1])
    if any(_canonical_bytes(value) != reference for _, value in entries[1:]):
        source_names = ", ".join(name for name, _ in entries)
        _fail(
            "identity_chain_mismatch",
            artifact,
            f"{field} disagrees across authenticated sources: {source_names}",
        )
    return entries[0][1]


def _load_prediction_archive(
    path: Path,
) -> tuple[dict[str, Any], tuple[str, ...], dict[str, dict[str, Any]]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            archive_keys = tuple(archive.files)
            if len(set(archive_keys)) != len(archive_keys):
                _fail(
                    "invalid_prediction_archive",
                    PREDICTIONS_NAME,
                    "prediction archive contains duplicate keys",
                )
            if "metadata_json" not in archive_keys:
                _fail(
                    "invalid_prediction_archive",
                    PREDICTIONS_NAME,
                    "prediction archive is missing metadata_json",
                )
            keys = tuple(sorted(archive_keys))
            array_schema: dict[str, dict[str, Any]] = {}
            metadata_text: str | None = None
            for key in keys:
                array = archive[key]
                array_schema[key] = {
                    "shape": list(array.shape),
                    "dtype": array.dtype.str,
                }
                if key != "metadata_json":
                    continue
                if array.shape != () or array.dtype.kind not in {"U", "S"}:
                    _fail(
                        "invalid_prediction_metadata",
                        PREDICTIONS_NAME,
                        "metadata_json must be a scalar Unicode or byte string",
                    )
                metadata_value = array.item()
                if isinstance(metadata_value, bytes):
                    try:
                        metadata_text = metadata_value.decode("utf-8")
                    except UnicodeDecodeError as error:
                        _fail(
                            "invalid_prediction_metadata",
                            PREDICTIONS_NAME,
                            f"metadata_json is not UTF-8: {error}",
                        )
                else:
                    metadata_text = str(metadata_value)
    except AuditError:
        raise
    except (OSError, ValueError, EOFError, zipfile.BadZipFile, zlib.error):
        _fail(
            "invalid_prediction_archive",
            PREDICTIONS_NAME,
            "cannot read prediction archive without pickle",
        )

    if metadata_text is None:
        _fail(
            "invalid_prediction_metadata",
            PREDICTIONS_NAME,
            "metadata_json could not be decoded",
        )
    metadata = _parse_json(metadata_text, "predictions.npz:metadata_json")
    if not isinstance(metadata, dict):
        _fail(
            "invalid_prediction_metadata",
            PREDICTIONS_NAME,
            "metadata_json must contain a JSON object",
        )
    if len(keys) < 2:
        _fail(
            "invalid_prediction_archive",
            PREDICTIONS_NAME,
            "prediction archive contains metadata but no prediction arrays",
        )

    return metadata, keys, array_schema


def _validate_schema(payload: Mapping[str, Any], expected: str, artifact: str) -> None:
    if payload.get("schema_version") != expected:
        _fail(
            "schema_mismatch",
            artifact,
            f"schema_version must be {expected!r}",
        )


def _identity_for_attempt(
    metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
    completion: Mapping[str, Any],
    prediction_digest: str,
) -> dict[str, Any]:
    world = _require_object(metrics.get("world"), METRICS_NAME, "world")
    protocol = _require_object(
        metrics.get("protocol_audit"),
        METRICS_NAME,
        "protocol_audit",
    )

    world_id = _require_equal(
        {
            "metrics.world.world_id": world.get("world_id"),
            "predictions.metadata_json.world_id": metadata.get("world_id"),
            "completion.world_id": completion.get("world_id"),
        },
        "identity",
        "world_id",
    )
    _require_string(world_id, "identity", "world_id")

    seed_partition = _require_equal(
        {
            "metrics.world.seed_partition": world.get("seed_partition"),
            "completion.seed_partition": completion.get("seed_partition"),
        },
        "identity",
        "seed_partition",
    )
    if seed_partition != "locked":
        _fail(
            "invalid_identity",
            "identity",
            "seed_partition must be 'locked' for a post-freeze comparison",
        )

    evaluation_role = _require_equal(
        {
            "metrics.world.evaluation_role": world.get("evaluation_role"),
            "completion.evaluation_role": completion.get("evaluation_role"),
        },
        "identity",
        "evaluation_role",
    )
    if evaluation_role != "post_freeze_deterministic_procedural_audit":
        _fail(
            "invalid_identity",
            "identity",
            "evaluation_role is not the post-freeze procedural audit role",
        )

    canonical_relative_output = _require_equal(
        {
            "metrics.canonical_relative_output": metrics.get("canonical_relative_output"),
            "metrics.protocol_audit.canonical_relative_output": protocol.get(
                "canonical_relative_output"
            ),
            "predictions.metadata_json.canonical_relative_output": metadata.get(
                "canonical_relative_output"
            ),
            "completion.canonical_relative_output": completion.get("canonical_relative_output"),
        },
        "identity",
        "canonical_relative_output",
    )
    _require_string(
        canonical_relative_output,
        "identity",
        "canonical_relative_output",
    )
    if (
        re.fullmatch(
            r"results/teacher-locked/full/locked-seed-[0-9]{2}",
            canonical_relative_output,
        )
        is None
    ):
        _fail(
            "invalid_identity",
            "identity",
            "canonical_relative_output is not a canonical locked teacher-world path",
        )

    teacher_config_sha256 = _require_equal(
        {
            "metrics.world.teacher_config_sha256": world.get("teacher_config_sha256"),
            "metrics.protocol_audit.teacher_config_sha256": protocol.get("teacher_config_sha256"),
            "predictions.metadata_json.teacher_config_sha256": metadata.get(
                "teacher_config_sha256"
            ),
            "completion.teacher_config_sha256": completion.get("teacher_config_sha256"),
        },
        "identity",
        "teacher_config_sha256",
    )
    _require_sha256(
        teacher_config_sha256,
        "identity",
        "teacher_config_sha256",
    )

    experiment_config_sha256 = _require_equal(
        {
            "metrics.protocol_audit.teacher_experiment_scientific_sha256": protocol.get(
                "teacher_experiment_scientific_sha256"
            ),
            "predictions.metadata_json.teacher_experiment_scientific_sha256": metadata.get(
                "teacher_experiment_scientific_sha256"
            ),
            "completion.teacher_experiment_scientific_sha256": completion.get(
                "teacher_experiment_scientific_sha256"
            ),
        },
        "identity",
        "teacher_experiment_scientific_sha256",
    )
    _require_sha256(
        experiment_config_sha256,
        "identity",
        "teacher_experiment_scientific_sha256",
    )

    freeze_attestation = _require_equal(
        {
            "metrics.protocol_audit.preoutcome_freeze": protocol.get("preoutcome_freeze"),
            "predictions.metadata_json.preoutcome_freeze": metadata.get("preoutcome_freeze"),
            "completion.preoutcome_freeze": completion.get("preoutcome_freeze"),
        },
        "identity",
        "preoutcome_freeze",
    )
    _require_mapping(freeze_attestation, "identity", "preoutcome_freeze")

    learned_methods = _require_equal(
        {
            "metrics.learned_methods": metrics.get("learned_methods"),
            "predictions.metadata_json.learned_methods": metadata.get("learned_methods"),
            "completion.learned_methods": completion.get("learned_methods"),
        },
        "identity",
        "learned_methods",
    )
    if (
        not isinstance(learned_methods, list)
        or not learned_methods
        or any(not isinstance(method, str) or not method for method in learned_methods)
        or len(set(learned_methods)) != len(learned_methods)
    ):
        _fail("invalid_identity", "identity", "learned_methods must be a non-empty list")

    canonical_method_set = _require_equal(
        {
            "metrics.canonical_learned_method_set_complete": metrics.get(
                "canonical_learned_method_set_complete"
            ),
            "predictions.metadata_json.canonical_learned_method_set_complete": metadata.get(
                "canonical_learned_method_set_complete"
            ),
            "completion.canonical_learned_method_set_complete": completion.get(
                "canonical_learned_method_set_complete"
            ),
        },
        "identity",
        "canonical_learned_method_set_complete",
    )
    if canonical_method_set is not True:
        _fail(
            "invalid_identity",
            "identity",
            "canonical_learned_method_set_complete must be true",
        )

    reported_methods = _require_equal(
        {
            "metrics.reported_methods": metrics.get("reported_methods"),
            "predictions.metadata_json.methods": metadata.get("methods"),
            "completion.reported_methods": completion.get("reported_methods"),
        },
        "identity",
        "reported_methods",
    )
    if (
        not isinstance(reported_methods, list)
        or not reported_methods
        or any(not isinstance(method, str) or not method for method in reported_methods)
        or len(set(reported_methods)) != len(reported_methods)
    ):
        _fail("invalid_identity", "identity", "reported_methods must be a non-empty list")

    run_seed = metadata.get("run_seed")
    if isinstance(run_seed, bool) or not isinstance(run_seed, int) or run_seed != 0:
        _fail(
            "invalid_identity",
            PREDICTIONS_NAME,
            "metadata_json.run_seed must be the canonical locked value 0",
        )

    prediction_audit_digest = _require_sha256(
        protocol.get("prediction_sha256_before_score"),
        METRICS_NAME,
        "protocol_audit.prediction_sha256_before_score",
    )
    if prediction_audit_digest != prediction_digest:
        _fail(
            "prediction_audit_mismatch",
            METRICS_NAME,
            "protocol_audit.prediction_sha256_before_score does not match predictions.npz",
        )

    public_seed = _require_equal(
        {
            "metrics.world.seed_material_public": world.get("seed_material_public"),
            "predictions.metadata_json.seed_material_public": metadata.get("seed_material_public"),
            "completion.seed_material_public": completion.get("seed_material_public"),
        },
        "identity",
        "seed_material_public",
    )
    if public_seed is not True:
        _fail("invalid_identity", "identity", "seed_material_public must be true")

    biological_eligibility = _require_equal(
        {
            "metrics.world.eligible_for_biological_headline_conjunction": world.get(
                "eligible_for_biological_headline_conjunction"
            ),
            "predictions.metadata_json.eligible_for_biological_headline_conjunction": (
                metadata.get("eligible_for_biological_headline_conjunction")
            ),
            "completion.eligible_for_biological_headline_conjunction": completion.get(
                "eligible_for_biological_headline_conjunction"
            ),
        },
        "identity",
        "eligible_for_biological_headline_conjunction",
    )
    if biological_eligibility is not False:
        _fail(
            "invalid_identity",
            "identity",
            "eligible_for_biological_headline_conjunction must be false",
        )

    experiment_config = _require_object(
        metrics.get("experiment_config"),
        METRICS_NAME,
        "experiment_config",
    )
    _require_finite_number(
        metrics.get("wall_seconds"),
        METRICS_NAME,
        "wall_seconds",
    )
    stage_fits = _require_mapping(
        metrics.get("stage_fits"),
        METRICS_NAME,
        "stage_fits",
    )
    if set(stage_fits) != set(learned_methods):
        _fail(
            "invalid_runtime_field",
            METRICS_NAME,
            "stage_fits keys must exactly match learned_methods",
        )
    for method in learned_methods:
        method_fit = _require_mapping(
            stage_fits.get(method),
            METRICS_NAME,
            f"stage_fits.{method}",
        )
        _require_finite_number(
            method_fit.get("wall_seconds"),
            METRICS_NAME,
            f"stage_fits.{method}.wall_seconds",
        )
    _require_string(world.get("schema_version"), METRICS_NAME, "world.schema_version")
    return {
        "schemas": {
            "metrics": metrics["schema_version"],
            "predictions": metadata["schema_version"],
            "completion": completion["schema_version"],
            "world": world.get("schema_version"),
        },
        "world": world,
        "world_id": world_id,
        "seed_partition": seed_partition,
        "evaluation_role": evaluation_role,
        "canonical_relative_output": canonical_relative_output,
        "teacher_config_sha256": teacher_config_sha256,
        "teacher_experiment_scientific_sha256": experiment_config_sha256,
        "experiment_config": experiment_config,
        "preoutcome_freeze": freeze_attestation,
        "learned_methods": learned_methods,
        "reported_methods": reported_methods,
        "run_seed": run_seed,
        "canonical_learned_method_set_complete": canonical_method_set,
        "seed_material_public": public_seed,
        "eligible_for_biological_headline_conjunction": biological_eligibility,
    }


def authenticate_attempt(directory: str | Path, label: str) -> AttemptData:
    """Authenticate one attempt without comparing or preferring it."""

    root = Path(directory)
    try:
        if not root.is_dir():
            _fail("missing_attempt_directory", label, f"{label} is not a directory")
    except OSError:
        _fail("attempt_directory_read_failed", label, f"cannot inspect {label}")

    metrics_raw, metrics_digest = _read_artifact(root, METRICS_NAME)
    prediction_digest = _digest_artifact(root, PREDICTIONS_NAME)
    completion_raw, completion_digest = _read_artifact(root, COMPLETION_NAME)

    _verify_sidecar(root, METRICS_NAME, metrics_digest)
    _verify_sidecar(root, PREDICTIONS_NAME, prediction_digest)

    metrics = _parse_json(metrics_raw, METRICS_NAME)
    completion = _parse_json(completion_raw, COMPLETION_NAME)
    if not isinstance(metrics, dict):
        _fail("invalid_structure", METRICS_NAME, "metrics.json must contain a JSON object")
    if not isinstance(completion, dict):
        _fail(
            "invalid_structure",
            COMPLETION_NAME,
            "completion.json must contain a JSON object",
        )

    metadata, prediction_keys, prediction_array_schema = _load_prediction_archive(
        root / PREDICTIONS_NAME
    )
    _validate_schema(metrics, METRICS_SCHEMA, METRICS_NAME)
    _validate_schema(metadata, PREDICTION_SCHEMA, PREDICTIONS_NAME)
    _validate_schema(completion, COMPLETION_SCHEMA, COMPLETION_NAME)

    completion_artifacts = _require_mapping(
        completion.get("artifacts"),
        COMPLETION_NAME,
        "artifacts",
    )
    if set(completion_artifacts) != set(ARTIFACT_NAMES):
        _fail(
            "invalid_completion_artifacts",
            COMPLETION_NAME,
            "completion artifacts must be exactly metrics.json and predictions.npz",
        )
    declared_metrics = _require_sha256(
        completion_artifacts.get(METRICS_NAME),
        COMPLETION_NAME,
        f"artifacts.{METRICS_NAME}",
    )
    declared_predictions = _require_sha256(
        completion_artifacts.get(PREDICTIONS_NAME),
        COMPLETION_NAME,
        f"artifacts.{PREDICTIONS_NAME}",
    )
    if declared_metrics != metrics_digest:
        _fail(
            "completion_binding_mismatch",
            METRICS_NAME,
            "completion.json does not bind the observed metrics.json digest",
        )
    if declared_predictions != prediction_digest:
        _fail(
            "completion_binding_mismatch",
            PREDICTIONS_NAME,
            "completion.json does not bind the observed predictions.npz digest",
        )

    identity = _identity_for_attempt(
        metrics,
        metadata,
        completion,
        prediction_digest,
    )
    summary = {
        "label": label,
        "artifacts": {
            METRICS_NAME: {
                "sha256": metrics_digest,
                "sidecar_authenticated": True,
                "completion_bound": True,
                "authentication_status": "AUTHENTICATED",
            },
            PREDICTIONS_NAME: {
                "sha256": prediction_digest,
                "sidecar_authenticated": True,
                "completion_bound": True,
                "authentication_status": "AUTHENTICATED",
            },
            COMPLETION_NAME: {
                "sha256": completion_digest,
                "sidecar_authenticated": False,
                "completion_bound": False,
                "authentication_status": "OBSERVED_DIGEST_ONLY",
                "reason": (
                    "cadence.teacher_completion.v1 does not emit a completion.json "
                    "SHA-256 sidecar or self-binding"
                ),
            },
        },
        "identity_sha256": _canonical_sha256(identity),
        "prediction_metadata_sha256": _canonical_sha256(metadata),
        "prediction_keys_sha256": _canonical_sha256(list(prediction_keys)),
        "prediction_array_schema_sha256": _canonical_sha256(prediction_array_schema),
        "prediction_key_count": len(prediction_keys),
    }
    return AttemptData(
        metrics=metrics,
        prediction_metadata=metadata,
        prediction_keys=prediction_keys,
        prediction_array_schema=prediction_array_schema,
        completion=completion,
        identity=identity,
        summary=summary,
        prediction_bytes_sha256=prediction_digest,
        prediction_path=root / PREDICTIONS_NAME,
    )


def _normalized_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(payload))
    normalized.pop("wall_seconds", None)
    stage_fits = normalized.get("stage_fits")
    if isinstance(stage_fits, dict):
        for method_payload in stage_fits.values():
            if isinstance(method_payload, dict):
                method_payload.pop("wall_seconds", None)
    return normalized


def _normalized_completion(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(payload))
    artifacts = normalized.get("artifacts")
    if isinstance(artifacts, dict):
        artifacts.pop(METRICS_NAME, None)
    return normalized


def _json_pointer_component(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _difference_paths(left: Any, right: Any, path: str = "") -> list[str]:
    if type(left) is not type(right):
        return [path or "/"]
    if isinstance(left, dict):
        differences: list[str] = []
        all_keys = sorted(set(left) | set(right))
        for key in all_keys:
            child = f"{path}/{_json_pointer_component(str(key))}"
            if key not in left or key not in right:
                differences.append(child)
            else:
                differences.extend(_difference_paths(left[key], right[key], child))
        return differences
    if isinstance(left, list):
        differences = []
        for index in range(max(len(left), len(right))):
            child = f"{path}/{index}"
            if index >= len(left) or index >= len(right):
                differences.append(child)
            else:
                differences.extend(_difference_paths(left[index], right[index], child))
        return differences
    return [] if left == right else [path or "/"]


def _comparison_row(left: Any, right: Any) -> dict[str, Any]:
    left_digest = _canonical_sha256(left)
    right_digest = _canonical_sha256(right)
    matches = left_digest == right_digest
    return {
        "status": "MATCH" if matches else "MISMATCH",
        "attempt_1_sha256": left_digest,
        "attempt_2_sha256": right_digest,
        "difference_paths": [] if matches else _difference_paths(left, right),
    }


def _files_equal(left: Path, right: Path) -> bool:
    try:
        if left.stat().st_size != right.stat().st_size:
            return False
        with left.open("rb") as left_stream, right.open("rb") as right_stream:
            while True:
                left_block = left_stream.read(2**20)
                right_block = right_stream.read(2**20)
                if left_block != right_block:
                    return False
                if not left_block:
                    return True
    except OSError:
        _fail(
            "artifact_read_failed",
            PREDICTIONS_NAME,
            "cannot re-read predictions.npz for exact byte comparison",
        )


def compare_attempts(first: str | Path, second: str | Path) -> dict[str, Any]:
    """Return a deterministic, value-free comparison report."""

    try:
        same_directory = Path(first).samefile(Path(second))
    except OSError:
        same_directory = Path(first).resolve(strict=False) == Path(second).resolve(strict=False)
    if same_directory:
        _fail(
            "identical_attempt_directories",
            "attempts",
            "attempt_1 and attempt_2 must resolve to distinct directories",
        )

    attempt_1 = authenticate_attempt(first, "attempt_1")
    attempt_2 = authenticate_attempt(second, "attempt_2")

    identity = _comparison_row(attempt_1.identity, attempt_2.identity)
    prediction_metadata = _comparison_row(
        attempt_1.prediction_metadata,
        attempt_2.prediction_metadata,
    )
    prediction_keys = _comparison_row(
        list(attempt_1.prediction_keys),
        list(attempt_2.prediction_keys),
    )
    prediction_array_schema = _comparison_row(
        attempt_1.prediction_array_schema,
        attempt_2.prediction_array_schema,
    )
    metrics = _comparison_row(
        _normalized_metrics(attempt_1.metrics),
        _normalized_metrics(attempt_2.metrics),
    )
    completions = _comparison_row(
        _normalized_completion(attempt_1.completion),
        _normalized_completion(attempt_2.completion),
    )
    prediction_digest_match = attempt_1.prediction_bytes_sha256 == attempt_2.prediction_bytes_sha256
    exact_prediction_bytes_match = _files_equal(
        attempt_1.prediction_path,
        attempt_2.prediction_path,
    )
    prediction_bytes_match = prediction_digest_match and exact_prediction_bytes_match
    prediction_bytes = {
        "status": "MATCH" if prediction_bytes_match else "MISMATCH",
        "attempt_1_sha256": attempt_1.prediction_bytes_sha256,
        "attempt_2_sha256": attempt_2.prediction_bytes_sha256,
        "exact_byte_comparison_status": ("MATCH" if exact_prediction_bytes_match else "MISMATCH"),
    }
    comparisons = {
        "identity": identity,
        "prediction_npz_bytes": prediction_bytes,
        "prediction_metadata": prediction_metadata,
        "prediction_keys": prediction_keys,
        "prediction_shapes_and_dtypes": prediction_array_schema,
        "metrics_semantics": {
            **metrics,
            "excluded_fields": [
                "/wall_seconds",
                "/stage_fits/<method>/wall_seconds",
            ],
        },
        "completion_semantics": {
            **completions,
            "excluded_field": "/artifacts/metrics.json",
        },
    }
    status = (
        "MATCH" if all(row["status"] == "MATCH" for row in comparisons.values()) else "MISMATCH"
    )
    return {
        "schema_version": AUDIT_SCHEMA,
        "status": status,
        "selection": "NONE",
        "validation_scope": {
            "retry_equivalence": "VALIDATED_HERE",
            "canonical_scientific_eligibility": "DELEGATED_TO_FROZEN_REPORTER",
        },
        "attempts": [attempt_1.summary, attempt_2.summary],
        "comparisons": comparisons,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt_1", type=Path, help="first completed teacher world directory")
    parser.add_argument("attempt_2", type=Path, help="second completed teacher world directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = compare_attempts(args.attempt_1, args.attempt_2)
    except AuditError as error:
        report = {
            "schema_version": AUDIT_SCHEMA,
            "status": "INVALID",
            "selection": "NONE",
            "validation_scope": {
                "retry_equivalence": "VALIDATED_HERE",
                "canonical_scientific_eligibility": "DELEGATED_TO_FROZEN_REPORTER",
            },
            "error": {
                "code": error.code,
                "artifact": error.artifact,
                "detail": error.detail,
            },
        }
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["status"] == "MATCH" else 1


if __name__ == "__main__":
    raise SystemExit(main())
