"""Leakage-sealed Allen Visual Behavior omission experiment.

The experimental unit is the mouse.  Four hash-selected mice are an explicit
development sandbox; the other 28 mice are locked until the caller opts into a
post-freeze run.  A target mouse contributes only ordinary image presentations
to preprocessing and adapter fitting.  Omission queries are materialized as
two files: an input-only bundle and a sealed post-onset outcome bundle.

The intervention is represented as the combination of (i) the absence of the
scheduled image-flash input and (ii) an omission pulse.  A pulse at relative
time zero is placed at ``onset - 1`` because CADENCE transition controls index
the transition *into* the corresponding observed sample.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import stat
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import yaml
from sklearn.linear_model import Ridge
from torch.nn import functional as F

from cadence.baselines import (
    AdditiveInterventionSSM,
    BlackBoxMetaGRU,
    LinearHierarchicalSSM,
)
from cadence.data.allen_vbo import (
    split_combined_animal_artifact,
    split_window_arrays,
)
from cadence.data.splits import LeakageError, assert_calibration_is_normal
from cadence.metrics import causal_skill, support_scale, time_resolved_r2, trajectory_nrmse
from cadence.model import HierarchicalControlledSSM, SequenceBatch
from cadence.protocol import FreezeAttestation, ProtocolViolation, attest_preoutcome_freeze
from cadence.training import EpochRecord, FitConfig, FitResult, move_batch, seed_everything

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
MethodName = Literal["proposed", "linear", "additive", "black_box"]

DEVELOPMENT_MICE: tuple[str, ...] = ("539517", "448900", "484631", "423606")
DEVELOPMENT_HASH_RULE = "sha256(mouse_id + '20260725'), first four"
LOCK_SEED = 20260725
LEARNED_METHODS: tuple[MethodName, ...] = (
    "proposed",
    "linear",
    "additive",
    "black_box",
)
REPORT_METHODS = (
    *LEARNED_METHODS,
    "functional_atlas",
    "no_effect",
    "condition_time",
    "nearest_donor",
)
INPUT_DIM = 8
BEHAVIOR_DIM = 3
DELTA_PROJECTION_TOLERANCE = 1e-7
CANONICAL_MANIFEST_RELATIVE = Path(
    "data/manifests/allen_vbo_slc17a7_visp175_familiar_active_v1.1.0.json"
)
CANONICAL_PROCESSED_ROOT_RELATIVE = Path("data/processed/allen_vbo")
CANONICAL_INDEX_RELATIVE = CANONICAL_PROCESSED_ROOT_RELATIVE / "index.json"
CANONICAL_CONFIG_RELATIVE = Path("configs/allen_experiment.yaml")
EXPECTED_ALLEN_RELEASE = "1.1.0"
EXPECTED_INDEX_SCHEMA = "cadence-allen-vbo-processed-index-v1"
EXPECTED_MANIFEST_SCHEMA = "1.0"
STAGE_COMPLETION_SCHEMA = "cadence-allen-vbo-stage-completion-v1"
ALLEN_ACTIVE_SEAL_NAME = ".cadence-allen-active-target-seal.json"
ALLEN_SEAL_TRANSACTION_SCHEMA = "cadence-allen-target-seal-transaction-v1"
ALLEN_PREPARE_GUARD_NAME = ".cadence-allen-active-prepare.json"
ALLEN_PREPARE_GUARD_SCHEMA = "cadence-allen-prepare-guard-v1"
ALLEN_RESTORE_COMPLETION_SCHEMA = "cadence-allen-target-restore-completion-v1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


def _stable_digest(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def _repository_root() -> Path:
    """Return the source-tree root; locked callers cannot select another checkout."""

    return Path(__file__).resolve().parents[3]


def _canonical_locked_relative_output(fold: int) -> Path:
    if fold not in range(5):
        raise ValueError("locked profile requires --fold in {0,1,2,3,4}")
    return Path("results") / "allen-vbo" / f"locked-fold-{fold}"


def _require_canonical_locked_output(path: Path, fold: int) -> str:
    """Bind every locked stage to one nonsymlink output path."""

    repository = _repository_root()
    relative = _canonical_locked_relative_output(fold)
    expected = (repository / relative).absolute()
    observed = path.absolute()
    if observed != expected:
        raise ProtocolViolation(
            f"locked Allen output must be the canonical one-shot path {expected}; "
            f"observed {observed}"
        )
    cursor = repository
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ProtocolViolation("locked Allen output may not traverse a symlink")
    return relative.as_posix()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes, *, overwrite: bool) -> None:
    """Write one small control artifact atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        if overwrite:
            os.replace(temporary_path, path)
        else:
            try:
                os.link(temporary_path, path)
            except FileExistsError as error:
                raise FileExistsError(f"refusing to overwrite {path}") from error
        _fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    encoded = (json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n").encode()
    _atomic_write_bytes(path, encoded, overwrite=overwrite)


def _run_git(
    repository: Path,
    *arguments: str,
    text: bool = True,
) -> str | bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=text,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise ProtocolViolation(
            f"cannot verify frozen tracked artifact: git {' '.join(arguments)}"
        ) from error
    return result.stdout if not text else result.stdout.strip()


def _tracked_file_audit(
    repository: Path,
    relative_path: Path,
    attestation: FreezeAttestation,
) -> dict[str, str]:
    """Verify that working bytes equal the exact blob in the attested commit."""

    relative = relative_path.as_posix()
    path = repository / relative_path
    _run_git(repository, "ls-files", "--error-unmatch", "--", relative)
    committed = _run_git(
        repository,
        "show",
        f"{attestation.commit}:{relative}",
        text=False,
    )
    if not isinstance(committed, bytes):
        raise AssertionError("binary git output unexpectedly decoded")
    working_sha256 = _sha256_path(path)
    committed_sha256 = hashlib.sha256(committed).hexdigest()
    if working_sha256 != committed_sha256:
        raise ProtocolViolation(
            f"working {relative} differs from attested commit {attestation.commit}"
        )
    blob = _run_git(
        repository,
        "rev-parse",
        f"{attestation.commit}:{relative}",
    )
    if not isinstance(blob, str):
        raise AssertionError("git blob identity unexpectedly binary")
    return {
        "path": relative,
        "sha256": working_sha256,
        "git_blob": blob,
        "git_blob_sha256": committed_sha256,
        "commit": attestation.commit,
    }


def manifest_mouse_ids(manifest_path: str | Path) -> list[str]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    ids = [str(entry["mouse_id"]) for entry in manifest["nwb_files"]]
    if len(ids) != 32 or len(set(ids)) != 32:
        raise ValueError("the frozen Allen manifest must contain exactly 32 unique mice")
    missing = set(DEVELOPMENT_MICE) - set(ids)
    if missing:
        raise ValueError(f"development mice absent from manifest: {sorted(missing)}")
    observed = tuple(
        sorted(ids, key=lambda mouse: hashlib.sha256(f"{mouse}{LOCK_SEED}".encode()).hexdigest())[
            :4
        ]
    )
    if set(observed) != set(DEVELOPMENT_MICE):
        raise ValueError("development hash rule no longer matches the frozen constant")
    return ids


def locked_fold_table(
    manifest_path: str | Path,
    *,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Round-robin hash folds for the 28 animals outside development."""

    if n_folds != 5:
        raise ValueError("the locked protocol freezes exactly five outer folds")
    mice = [mouse for mouse in manifest_mouse_ids(manifest_path) if mouse not in DEVELOPMENT_MICE]
    ordered = sorted(
        mice,
        key=lambda mouse: (hashlib.sha256(f"{mouse}{LOCK_SEED}".encode()).hexdigest(), mouse),
    )
    table = pd.DataFrame(
        {
            "mouse_id": ordered,
            "outer_fold": np.arange(len(ordered), dtype=np.int64) % n_folds,
            "profile": "locked",
            "split_seed": LOCK_SEED,
        }
    )
    return table.sort_values("mouse_id", kind="stable").reset_index(drop=True)


def resolve_run_mice(
    manifest_path: str | Path,
    *,
    profile: Literal["development", "locked"],
    fold: int | None = None,
    development_target: str = "423606",
    development_donors: Sequence[str] = ("539517", "448900"),
    acknowledge_locked: bool = False,
) -> tuple[list[str], list[str]]:
    """Resolve donors and targets while enforcing the locked-outcome gate."""

    all_mice = set(manifest_mouse_ids(manifest_path))
    if profile == "development":
        if fold is not None:
            raise ValueError("development profile does not accept an outer fold")
        target = str(development_target)
        donors = [str(mouse) for mouse in development_donors]
        used = set(donors) | {target}
        if not used <= set(DEVELOPMENT_MICE):
            raise LeakageError("development runs may only open the four development mice")
        if target in donors or len(donors) < 2 or len(set(donors)) != len(donors):
            raise ValueError("development smoke requires at least two distinct non-target donors")
        return donors, [target]
    if profile != "locked":
        raise ValueError(f"unknown profile: {profile}")
    if not acknowledge_locked:
        raise LeakageError(
            "locked outcomes remain sealed; pass --acknowledge-locked only after the "
            "protocol/code commit has been frozen"
        )
    if fold is None or fold not in range(5):
        raise ValueError("locked profile requires --fold in {0,1,2,3,4}")
    table = locked_fold_table(manifest_path)
    targets = table.loc[table["outer_fold"].eq(fold), "mouse_id"].tolist()
    donors = table.loc[~table["outer_fold"].eq(fold), "mouse_id"].tolist()
    if (set(donors) | set(targets)) != all_mice - set(DEVELOPMENT_MICE):
        raise AssertionError("locked donor/target partition does not cover the locked cohort")
    if set(donors) & set(targets):
        raise AssertionError("locked donor and target mice overlap")
    return donors, targets


def split_normal_presentations(
    mouse_id: str,
    presentation_ids: npt.ArrayLike,
    *,
    seed: int = LOCK_SEED,
    adapter_count: int = 160,
) -> dict[str, npt.NDArray[np.int64]]:
    """Select 160 signal-blind adapter windows, then split them 70/15/15.

    All remaining clean normal windows form a separate, untouched matching
    pool.  Selection uses presentation identifiers only and therefore cannot
    depend on neural or behavioral responses.
    """

    ids = np.asarray(presentation_ids, dtype=np.int64)
    if (
        ids.ndim != 1
        or len(ids) < adapter_count
        or len(np.unique(ids)) != len(ids)
        or adapter_count < 10
    ):
        raise ValueError(
            f"normal presentation IDs must be unique and contain at least {adapter_count} entries"
        )
    order = np.asarray(
        sorted(
            range(len(ids)),
            key=lambda index: (
                _stable_digest("allen-normal-v1", seed, mouse_id, int(ids[index])),
                int(ids[index]),
            ),
        ),
        dtype=np.int64,
    )
    adapter = order[:adapter_count]
    fit_stop = int(math.floor(0.70 * adapter_count))
    val_stop = int(math.floor(0.85 * adapter_count))
    result = {
        "fit": np.sort(adapter[:fit_stop]),
        "val": np.sort(adapter[fit_stop:val_stop]),
        "audit": np.sort(adapter[val_stop:]),
        "match": np.sort(order[adapter_count:]),
    }
    joined = np.concatenate(list(result.values()))
    if len(joined) != len(ids) or len(np.unique(joined)) != len(ids):
        raise AssertionError("normal split is not a partition")
    return result


def transition_index(relative_time_s: npt.ArrayLike, event_relative_s: float) -> int:
    """Control index for the transition into the first sample at/after an event."""

    relative = np.asarray(relative_time_s, dtype=np.float64)
    if relative.ndim != 1 or len(relative) < 3 or np.any(np.diff(relative) <= 0):
        raise ValueError("relative times must be a strictly increasing vector")
    sample = int(np.searchsorted(relative, event_relative_s, side="left"))
    return int(np.clip(sample - 1, 0, len(relative) - 2))


def build_flash_inputs(
    presentations: pd.DataFrame,
    event_times: npt.ArrayLike,
    relative_time_s: npt.ArrayLike,
) -> npt.NDArray[np.float32]:
    """Build eight physical image-flash channels from the presentation schedule."""

    relative = np.asarray(relative_time_s, dtype=np.float64)
    centers = np.asarray(event_times, dtype=np.float64)
    output = np.zeros((len(centers), len(relative), INPUT_DIM), dtype=np.float32)
    required = {"start_time", "image_index", "omitted", "active"}
    missing = required - set(presentations)
    if missing:
        raise KeyError(f"stimulus table lacks schedule columns: {sorted(missing)}")
    active = presentations.loc[
        presentations["active"].astype(bool) & ~presentations["omitted"].astype(bool)
    ].copy()
    active["image_index"] = pd.to_numeric(active["image_index"], errors="coerce")
    active = active.loc[active["image_index"].between(0, INPUT_DIM - 1)]
    starts = active["start_time"].to_numpy(dtype=np.float64)
    images = active["image_index"].to_numpy(dtype=np.int64)
    for trial, center in enumerate(centers):
        left = int(np.searchsorted(starts, center + relative[0] - 1e-8, side="left"))
        right = int(np.searchsorted(starts, center + relative[-1] + 1e-8, side="right"))
        for start, image in zip(starts[left:right], images[left:right], strict=True):
            index = transition_index(relative, float(start - center))
            output[trial, index, image] = 1.0
    return output


def omission_descriptors(
    omission_rows: pd.DataFrame,
    presentations: pd.DataFrame,
) -> npt.NDArray[np.int64]:
    """Return preceding-image and flashes-since-change descriptors."""

    ordered = presentations.sort_values("start_time", kind="stable").reset_index(drop=True)
    starts = ordered["start_time"].to_numpy(dtype=np.float64)
    image = pd.to_numeric(ordered["image_index"], errors="coerce").to_numpy(dtype=np.float64)
    omitted = ordered["omitted"].astype(bool).to_numpy()
    active = ordered["active"].astype(bool).to_numpy()
    output = np.empty((len(omission_rows), 2), dtype=np.int64)
    for index, row in enumerate(omission_rows.itertuples(index=False)):
        position = int(np.searchsorted(starts, float(row.event_time), side="left"))
        preceding = position - 1
        while preceding >= 0 and (
            not active[preceding]
            or omitted[preceding]
            or not np.isfinite(image[preceding])
            or not 0 <= image[preceding] < INPUT_DIM
        ):
            preceding -= 1
        if preceding < 0:
            raise ValueError("omission has no preceding ordinary image")
        flashes = getattr(row, "flashes_since_change", -1)
        output[index] = (int(image[preceding]), int(flashes) if pd.notna(flashes) else -1)
    return output


def normal_descriptors(normal_rows: pd.DataFrame) -> npt.NDArray[np.int64]:
    image = pd.to_numeric(normal_rows["image_index"], errors="raise").to_numpy(dtype=np.int64)
    flashes = (
        pd.to_numeric(normal_rows["flashes_since_change"], errors="coerce")
        .fillna(-1)
        .to_numpy(dtype=np.int64)
    )
    return np.column_stack((image, flashes))


def match_control_indices(
    query_descriptors: npt.ArrayLike,
    control_descriptors: npt.ArrayLike,
    *,
    controls_per_query: int = 3,
) -> tuple[npt.NDArray[np.int64], dict[str, Any]]:
    """Apply the frozen, categorical normal-control fallback hierarchy.

    Each row contains local control indices and is padded with ``-1`` when the
    first nonempty eligible pool contains fewer than ``controls_per_query``
    unique controls. No lower fallback level is mixed into a higher one.
    """

    query = np.asarray(query_descriptors, dtype=np.int64)
    control = np.asarray(control_descriptors, dtype=np.int64)
    if query.ndim != 2 or control.ndim != 2 or query.shape[1] != 2 or control.shape[1] != 2:
        raise ValueError("descriptors must be [trial, (image, flashes_since_change)]")
    if len(control) == 0 or controls_per_query < 1:
        raise ValueError("at least one eligible normal control is required")
    count = min(controls_per_query, len(control))
    matches = np.full((len(query), count), -1, dtype=np.int64)
    exact = 0
    same_image = 0
    fallback_levels: list[str] = []
    effective_controls: list[int] = []
    level_names = (
        "exact_image_and_flashes",
        "image_and_risk_bin",
        "image_only",
        "risk_bin_only",
        "complete_pool",
    )
    for index, (image, flashes) in enumerate(query):
        query_bin = _flash_risk_bin(int(flashes))
        control_bins = np.asarray(
            [_flash_risk_bin(int(value)) for value in control[:, 1]],
            dtype=np.int64,
        )
        candidate_masks = (
            (control[:, 0] == image) & (control[:, 1] == flashes),
            (control[:, 0] == image) & (control_bins == query_bin),
            control[:, 0] == image,
            control_bins == query_bin,
            np.ones(len(control), dtype=bool),
        )
        level = next(
            level_index
            for level_index, candidate_mask in enumerate(candidate_masks)
            if np.any(candidate_mask)
        )
        candidates = np.flatnonzero(candidate_masks[level])
        # Within a declared category, nearest raw risk count is a deterministic
        # tie-breaker rather than an additional fallback rule.
        distance = np.abs(
            np.log1p(np.maximum(control[candidates, 1], 0)) - np.log1p(max(int(flashes), 0))
        )
        order = np.lexsort((candidates, distance))
        selected = candidates[order[:count]]
        matches[index, : len(selected)] = selected
        fallback_levels.append(level_names[level])
        effective_controls.append(len(selected))
        exact += int(level == 0)
        same_image += int(level <= 2)
    level_counts = {name: int(fallback_levels.count(name)) for name in level_names}
    audit = {
        "queries": int(len(query)),
        "controls_per_query_requested": int(controls_per_query),
        "fallback_levels": fallback_levels,
        "fallback_level_counts": level_counts,
        "effective_controls": effective_controls,
        "effective_controls_min": int(min(effective_controls)),
        "effective_controls_mean": float(np.mean(effective_controls)),
        "effective_controls_max": int(max(effective_controls)),
        "exact_risk_set_fraction": float(exact / max(len(query), 1)),
        "same_preceding_image_fraction": float(same_image / max(len(query), 1)),
    }
    return matches, audit


def _flash_risk_bin(flashes_since_change: int) -> int:
    if flashes_since_change <= 0:
        return 0
    if flashes_since_change <= 2:
        return flashes_since_change
    if flashes_since_change <= 4:
        return 3
    if flashes_since_change <= 8:
        return 4
    return 5


def _map_match_indices(
    pool_indices: npt.ArrayLike,
    local_matches: npt.ArrayLike,
) -> npt.NDArray[np.int64]:
    pool = np.asarray(pool_indices, dtype=np.int64)
    local = np.asarray(local_matches, dtype=np.int64)
    if np.any(local >= len(pool)):
        raise IndexError("matched control index is outside its eligible pool")
    return np.where(local >= 0, pool[np.maximum(local, 0)], -1)


@dataclass(frozen=True, slots=True)
class MouseScaler:
    """Per-mouse transform fit only on the normal-fit support partition."""

    neural_center: FloatArray
    neural_scale: FloatArray
    behavior_center: FloatArray
    behavior_scale: FloatArray
    fit_presentation_sha256: str

    @classmethod
    def fit(
        cls,
        neural: npt.ArrayLike,
        neural_valid: npt.ArrayLike,
        behavior: npt.ArrayLike,
        behavior_valid: npt.ArrayLike,
        fit_indices: npt.ArrayLike,
        fit_presentation_ids: npt.ArrayLike,
    ) -> MouseScaler:
        indices = np.asarray(fit_indices, dtype=np.int64)
        transformed = np.log1p(np.maximum(np.asarray(neural, dtype=np.float64)[indices], 0.0))
        nvalid = np.asarray(neural_valid, dtype=bool)[indices] & np.isfinite(transformed)
        bvalues = np.asarray(behavior, dtype=np.float64)[indices]
        bvalid = np.asarray(behavior_valid, dtype=bool)[indices] & np.isfinite(bvalues)

        def moments(values: FloatArray, valid: BoolArray) -> tuple[FloatArray, FloatArray]:
            masked = np.where(valid, values, np.nan)
            center = np.nanmean(masked, axis=(0, 1))
            scale = np.nanstd(masked, axis=(0, 1), ddof=1)
            center = np.nan_to_num(center, nan=0.0)
            finite = scale[np.isfinite(scale) & (scale > 1e-8)]
            fallback = float(np.median(finite)) if len(finite) else 1.0
            scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, fallback)
            return center, scale

        neural_center, neural_scale = moments(transformed, nvalid)
        behavior_center, behavior_scale = moments(bvalues, bvalid)
        ids = np.sort(np.asarray(fit_presentation_ids, dtype=np.int64))
        digest = hashlib.sha256(ids.tobytes()).hexdigest()
        return cls(neural_center, neural_scale, behavior_center, behavior_scale, digest)

    def transform_neural(
        self, values: npt.ArrayLike, valid: npt.ArrayLike
    ) -> tuple[npt.NDArray[np.float32], BoolArray]:
        raw = np.log1p(np.maximum(np.asarray(values, dtype=np.float64), 0.0))
        mask = np.asarray(valid, dtype=bool) & np.isfinite(raw)
        result = (raw - self.neural_center) / self.neural_scale
        return np.where(mask, result, 0.0).astype(np.float32), mask

    def inverse_neural(self, values: npt.ArrayLike) -> FloatArray:
        transformed = np.asarray(values, dtype=np.float64) * self.neural_scale + self.neural_center
        return np.maximum(np.expm1(np.clip(transformed, 0.0, 12.0)), 0.0)

    def transform_behavior(
        self, values: npt.ArrayLike, valid: npt.ArrayLike
    ) -> tuple[npt.NDArray[np.float32], BoolArray]:
        raw = np.asarray(values, dtype=np.float64)
        mask = np.asarray(valid, dtype=bool) & np.isfinite(raw)
        result = (raw - self.behavior_center) / self.behavior_scale
        return np.where(mask, result, 0.0).astype(np.float32), mask

    def inverse_behavior(self, values: npt.ArrayLike) -> FloatArray:
        return np.asarray(values, dtype=np.float64) * self.behavior_scale + self.behavior_center


@dataclass(slots=True)
class AnimalSupport:
    mouse_id: str
    directory: Path
    relative_time_s: FloatArray
    cell_roi_ids: npt.NDArray[np.int64]
    normal_neural_raw: npt.NDArray[np.float32]
    normal_neural_valid: BoolArray
    normal_behavior_raw: npt.NDArray[np.float32]
    normal_behavior_valid: BoolArray
    normal_inputs: npt.NDArray[np.float32]
    normal_descriptors: npt.NDArray[np.int64]
    normal_rows: pd.DataFrame
    partitions: dict[str, npt.NDArray[np.int64]]
    scaler: MouseScaler

    @property
    def onset(self) -> int:
        indices = np.flatnonzero(self.relative_time_s >= 0)
        if not len(indices) or not np.isclose(self.relative_time_s[indices[0]], 0.0):
            raise ValueError("relative-time grid does not contain intervention onset")
        return int(indices[0])

    @property
    def model_neural_dim(self) -> int:
        return 2 * len(self.cell_roi_ids) + BEHAVIOR_DIM

    def normal_batch(self, indices: npt.ArrayLike) -> SequenceBatch:
        selected = np.asarray(indices, dtype=np.int64)
        neural, nmask = self.scaler.transform_neural(
            self.normal_neural_raw[selected], self.normal_neural_valid[selected]
        )
        behavior, bmask = self.scaler.transform_behavior(
            self.normal_behavior_raw[selected], self.normal_behavior_valid[selected]
        )
        augmented, augmented_mask = _augment_neural_with_masks(neural, nmask, bmask)
        return SequenceBatch(
            animal_id=self.mouse_id,
            neural=torch.as_tensor(augmented),
            behavior=torch.as_tensor(behavior),
            inputs=torch.as_tensor(self.normal_inputs[selected]),
            intervention=torch.zeros((len(selected), len(self.relative_time_s), 1)),
            onset=self.onset,
            neural_mask=torch.as_tensor(augmented_mask),
            behavior_mask=torch.as_tensor(bmask),
        )


def _augment_neural_with_masks(
    neural: npt.ArrayLike,
    neural_valid: npt.ArrayLike,
    behavior_valid: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float32], BoolArray]:
    """Expose missingness to the encoder without changing shared model code.

    Auxiliary mask channels are encoder inputs but receive zero reconstruction
    weight, so observed-neural scoring and the decoder likelihood remain on the
    original cell channels.
    """

    values = np.asarray(neural, dtype=np.float32)
    nmask = np.asarray(neural_valid, dtype=bool)
    bmask = np.asarray(behavior_valid, dtype=bool)
    if values.shape != nmask.shape or values.shape[:-1] != bmask.shape[:-1]:
        raise ValueError("neural/behavior masks do not align for augmentation")
    augmented = np.concatenate(
        (values, nmask.astype(np.float32), bmask.astype(np.float32)),
        axis=-1,
    )
    auxiliary = np.zeros(augmented.shape[:-1] + (nmask.shape[-1] + bmask.shape[-1],), bool)
    augmented_mask = np.concatenate((nmask, auxiliary), axis=-1)
    return augmented, augmented_mask


def load_animal_support(
    processed_root: str | Path,
    mouse_id: str,
    *,
    split_seed: int = LOCK_SEED,
    require_physical_split: bool = False,
) -> AnimalSupport:
    """Load normal support only; omission arrays are deliberately not accessed."""

    directory = Path(processed_root) / f"mouse_{mouse_id}"
    normal_path = directory / "normal_support.npz"
    if not normal_path.exists():
        if require_physical_split:
            raise LeakageError(f"mouse {mouse_id} lacks role-separated normal_support.npz")
        normal_path = directory / "windows.npz"
    rows = pd.read_parquet(directory / "window_index.parquet")
    normal_rows = (
        rows.loc[rows["window_kind"].eq("normal")]
        .sort_values("window_index", kind="stable")
        .reset_index(drop=True)
    )
    assert_calibration_is_normal(normal_rows)
    presentations = pd.read_parquet(directory / "stimulus_presentations.parquet")
    with np.load(normal_path, allow_pickle=False) as arrays:
        relative = arrays["relative_time_s"].astype(np.float64)
        roi_ids = arrays["cell_roi_ids"].astype(np.int64)
        neural = arrays["normal_neural"].astype(np.float32)
        neural_valid = arrays["normal_neural_valid"].astype(bool)
        behavior = arrays["normal_behavior"].astype(np.float32)
        behavior_valid = arrays["normal_behavior_valid"].astype(bool)
        event_times = arrays["normal_event_times"].astype(np.float64)
        presentation_ids = arrays["normal_presentation_ids"].astype(np.int64)
    if not np.array_equal(
        normal_rows["stimulus_presentation_id"].to_numpy(np.int64), presentation_ids
    ):
        raise ValueError(f"normal table/array alignment failed for mouse {mouse_id}")
    partitions = split_normal_presentations(mouse_id, presentation_ids, seed=split_seed)
    scaler = MouseScaler.fit(
        neural,
        neural_valid,
        behavior,
        behavior_valid,
        partitions["fit"],
        presentation_ids[partitions["fit"]],
    )
    return AnimalSupport(
        mouse_id=str(mouse_id),
        directory=directory,
        relative_time_s=relative,
        cell_roi_ids=roi_ids,
        normal_neural_raw=neural,
        normal_neural_valid=neural_valid,
        normal_behavior_raw=behavior,
        normal_behavior_valid=behavior_valid,
        normal_inputs=build_flash_inputs(presentations, event_times, relative),
        normal_descriptors=normal_descriptors(normal_rows),
        normal_rows=normal_rows,
        partitions=partitions,
        scaler=scaler,
    )


@dataclass(slots=True)
class OmissionData:
    neural: npt.NDArray[np.float32]
    neural_valid: BoolArray
    behavior: npt.NDArray[np.float32]
    behavior_valid: BoolArray
    inputs: npt.NDArray[np.float32]
    descriptors: npt.NDArray[np.int64]
    presentation_ids: npt.NDArray[np.int64]


def prepare_role_separated_artifacts(
    processed_root: str | Path,
    mouse_ids: Sequence[str],
    *,
    overwrite: bool = False,
    verify_against_combined: bool = False,
) -> dict[str, dict[str, str]]:
    """Create and fingerprint role-separated artifacts for a staged run."""

    result: dict[str, dict[str, str]] = {}
    for mouse in mouse_ids:
        directory = Path(processed_root) / f"mouse_{mouse}"
        normal, query, sealed = split_combined_animal_artifact(
            directory,
            overwrite=overwrite,
        )
        if verify_against_combined:
            with np.load(directory / "windows.npz", allow_pickle=False) as archive:
                combined = {name: archive[name] for name in archive.files}
            expected_payloads = split_window_arrays(combined)
            for path, expected in zip((normal, query, sealed), expected_payloads, strict=True):
                with np.load(path, allow_pickle=False) as archive:
                    if set(archive.files) != set(expected):
                        raise ProtocolViolation(
                            f"role artifact keys differ from windows.npz: {path}"
                        )
                    for name, values in expected.items():
                        observed = archive[name]
                        equal = (
                            np.array_equal(observed, values, equal_nan=True)
                            if observed.dtype.kind in {"f", "c"}
                            else np.array_equal(observed, values)
                        )
                        if not equal:
                            raise ProtocolViolation(
                                f"role artifact differs from windows.npz: {path.name}/{name}"
                            )
        result[mouse] = {path.name: _sha256_path(path) for path in (normal, query, sealed)}
    return result


def load_omission_data(
    support: AnimalSupport,
    *,
    require_physical_split: bool = False,
) -> OmissionData:
    """Open omission outcomes. Callers must apply the development/locked gate."""

    rows = pd.read_parquet(support.directory / "window_index.parquet")
    omission_rows = (
        rows.loc[rows["window_kind"].eq("omission")]
        .sort_values("window_index", kind="stable")
        .reset_index(drop=True)
    )
    presentations = pd.read_parquet(support.directory / "stimulus_presentations.parquet")
    query_path = support.directory / "omission_query.npz"
    sealed_path = support.directory / "sealed_omission_outcomes.npz"
    if query_path.exists() and sealed_path.exists():
        with np.load(query_path, allow_pickle=False) as query:
            pre_neural = query["omission_pre_neural"].astype(np.float32)
            pre_neural_valid = query["omission_pre_neural_valid"].astype(bool)
            pre_behavior = query["omission_pre_behavior"].astype(np.float32)
            pre_behavior_valid = query["omission_pre_behavior_valid"].astype(bool)
            event_times = query["omission_event_times"].astype(np.float64)
            presentation_ids = query["omission_presentation_ids"].astype(np.int64)
        with np.load(sealed_path, allow_pickle=False) as sealed:
            if not np.array_equal(
                presentation_ids,
                sealed["omission_presentation_ids"].astype(np.int64),
            ):
                raise ValueError("query/sealed omission presentation IDs differ")
            neural = np.concatenate(
                (pre_neural, sealed["omission_post_neural"].astype(np.float32)),
                axis=1,
            )
            neural_valid = np.concatenate(
                (pre_neural_valid, sealed["omission_post_neural_valid"].astype(bool)),
                axis=1,
            )
            behavior = np.concatenate(
                (pre_behavior, sealed["omission_post_behavior"].astype(np.float32)),
                axis=1,
            )
            behavior_valid = np.concatenate(
                (
                    pre_behavior_valid,
                    sealed["omission_post_behavior_valid"].astype(bool),
                ),
                axis=1,
            )
    else:
        if require_physical_split:
            raise LeakageError(f"mouse {support.mouse_id} lacks role-separated omission artifacts")
        with np.load(support.directory / "windows.npz", allow_pickle=False) as arrays:
            neural = arrays["omission_neural"].astype(np.float32)
            neural_valid = arrays["omission_neural_valid"].astype(bool)
            behavior = arrays["omission_behavior"].astype(np.float32)
            behavior_valid = arrays["omission_behavior_valid"].astype(bool)
            event_times = arrays["omission_event_times"].astype(np.float64)
            presentation_ids = arrays["omission_presentation_ids"].astype(np.int64)
    if not np.array_equal(
        omission_rows["stimulus_presentation_id"].to_numpy(np.int64), presentation_ids
    ):
        raise ValueError(f"omission table/array alignment failed for mouse {support.mouse_id}")
    return OmissionData(
        neural=neural,
        neural_valid=neural_valid,
        behavior=behavior,
        behavior_valid=behavior_valid,
        inputs=build_flash_inputs(presentations, event_times, support.relative_time_s),
        descriptors=omission_descriptors(omission_rows, presentations),
        presentation_ids=presentation_ids,
    )


def _chunks(indices: npt.ArrayLike, size: int) -> Iterable[npt.NDArray[np.int64]]:
    values = np.asarray(indices, dtype=np.int64)
    for start in range(0, len(values), size):
        yield values[start : start + size]


def normal_batches(
    support: AnimalSupport,
    partition: Literal["fit", "val", "audit"],
    *,
    batch_size: int,
    limit: int | None = None,
) -> list[SequenceBatch]:
    indices = support.partitions[partition]
    if limit is not None:
        indices = indices[:limit]
    return [support.normal_batch(chunk) for chunk in _chunks(indices, batch_size)]


def _counterfactual_control_inputs(
    treated_inputs: npt.NDArray[np.float32],
    descriptors: npt.NDArray[np.int64],
    onset: int,
) -> npt.NDArray[np.float32]:
    control = treated_inputs.copy()
    transition = onset - 1
    for trial, image in enumerate(descriptors[:, 0]):
        control[trial, transition, :] = 0.0
        control[trial, transition, int(image)] = 1.0
    return control


def _average_matched(
    values: npt.ArrayLike,
    valid: npt.ArrayLike,
    matches: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.float32], BoolArray]:
    array = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(array)
    match_array = np.asarray(matches, dtype=np.int64)
    if np.any(match_array >= len(array)):
        raise IndexError("matched control index is outside the source array")
    present = match_array >= 0
    safe = np.maximum(match_array, 0)
    selected = array[safe]
    selected_mask = mask[safe]
    present_shape = present.shape + (1,) * (selected_mask.ndim - present.ndim)
    selected_mask &= present.reshape(present_shape)
    numerator = np.where(selected_mask, selected, 0.0).sum(axis=1)
    denominator = selected_mask.sum(axis=1)
    averaged = numerator / np.maximum(denominator, 1)
    return averaged.astype(np.float32), denominator > 0


@dataclass(slots=True)
class MatchedInterventionBatch:
    treated: SequenceBatch
    control_neural: torch.Tensor
    control_behavior: torch.Tensor
    control_neural_mask: torch.Tensor
    control_behavior_mask: torch.Tensor
    control_inputs: torch.Tensor

    def validate(self) -> None:
        self.treated.validate()
        if self.control_neural.shape != self.treated.neural.shape:
            raise ValueError("matched neural control shape differs from omission batch")
        if self.control_behavior.shape != self.treated.behavior.shape:
            raise ValueError("matched behavior control shape differs from omission batch")
        if self.control_inputs.shape != self.treated.inputs.shape:
            raise ValueError("matched control schedule shape differs from omission batch")


def intervention_batches(
    support: AnimalSupport,
    omission: OmissionData,
    *,
    batch_size: int,
    max_trials: int | None = None,
    controls_per_query: int = 3,
) -> tuple[list[MatchedInterventionBatch], dict[str, float]]:
    """Build donor omission/control pairs using the untouched normal match pool."""

    audit_indices = support.partitions["match"]
    if not len(audit_indices):
        # Compatibility for a legacy 160-window smoke artifact. Paper runs
        # require preprocessing with ``--all-normal`` and never take this path.
        audit_indices = support.partitions["audit"]
    matches_local, audit = match_control_indices(
        omission.descriptors,
        support.normal_descriptors[audit_indices],
        controls_per_query=controls_per_query,
    )
    matches = _map_match_indices(audit_indices, matches_local)
    control_neural, control_nvalid = _average_matched(
        support.normal_neural_raw, support.normal_neural_valid, matches
    )
    control_behavior, control_bvalid = _average_matched(
        support.normal_behavior_raw, support.normal_behavior_valid, matches
    )
    treated_neural, treated_nvalid = support.scaler.transform_neural(
        omission.neural, omission.neural_valid
    )
    treated_behavior, treated_bvalid = support.scaler.transform_behavior(
        omission.behavior, omission.behavior_valid
    )
    matched_neural, matched_nvalid = support.scaler.transform_neural(control_neural, control_nvalid)
    matched_behavior, matched_bvalid = support.scaler.transform_behavior(
        control_behavior, control_bvalid
    )
    treated_neural, treated_nvalid = _augment_neural_with_masks(
        treated_neural, treated_nvalid, treated_bvalid
    )
    matched_neural, matched_nvalid = _augment_neural_with_masks(
        matched_neural, matched_nvalid, matched_bvalid
    )
    control_inputs = _counterfactual_control_inputs(
        omission.inputs, omission.descriptors, support.onset
    )
    indices = np.arange(len(omission.neural), dtype=np.int64)
    indices = np.asarray(
        sorted(
            indices,
            key=lambda index: _stable_digest(
                "allen-omission-order-v1",
                LOCK_SEED,
                support.mouse_id,
                int(omission.presentation_ids[index]),
            ),
        ),
        dtype=np.int64,
    )
    if max_trials is not None:
        indices = indices[:max_trials]
    output = []
    for selected in _chunks(indices, batch_size):
        intervention = np.zeros((len(selected), len(support.relative_time_s), 1), np.float32)
        intervention[:, support.onset - 1, 0] = 1.0
        treated = SequenceBatch(
            animal_id=support.mouse_id,
            neural=torch.as_tensor(treated_neural[selected]),
            behavior=torch.as_tensor(treated_behavior[selected]),
            inputs=torch.as_tensor(omission.inputs[selected]),
            intervention=torch.as_tensor(intervention),
            onset=support.onset,
            neural_mask=torch.as_tensor(treated_nvalid[selected]),
            behavior_mask=torch.as_tensor(treated_bvalid[selected]),
        )
        batch = MatchedInterventionBatch(
            treated=treated,
            control_neural=torch.as_tensor(matched_neural[selected]),
            control_behavior=torch.as_tensor(matched_behavior[selected]),
            control_neural_mask=torch.as_tensor(matched_nvalid[selected]),
            control_behavior_mask=torch.as_tensor(matched_bvalid[selected]),
            control_inputs=torch.as_tensor(control_inputs[selected]),
        )
        batch.validate()
        output.append(batch)
    return output, audit


def _masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = mask.to(dtype=prediction.dtype)
    return ((prediction - target).square() * valid).sum() / valid.sum().clamp_min(1.0)


def _normal_loss(
    model: HierarchicalControlledSSM,
    batch: SequenceBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z, _ = model.encode(batch.animal_id, batch.neural, batch.behavior, sample=False)
    neural_hat, behavior_hat = model.decode(batch.animal_id, z)
    neural = _masked_mse(neural_hat, batch.neural, batch.neural_mask)
    behavior = _masked_mse(behavior_hat, batch.behavior, batch.behavior_mask)
    batch_count, transition_count, latent_dim = z[:, :-1].shape
    predicted = model.transition(
        batch.animal_id,
        z[:, :-1].reshape(-1, latent_dim),
        batch.inputs[:, :-1].reshape(-1, batch.inputs.shape[-1]),
        batch.intervention[:, :-1].reshape(-1, batch.intervention.shape[-1]),
        include_animal_residual=True,
        include_donor_delta=False,
    ).reshape(batch_count, transition_count, latent_dim)
    dynamics = F.mse_loss(predicted, z[:, 1:].detach())
    variance = z.reshape(-1, z.shape[-1]).var(dim=0, unbiased=False)
    variance_floor = F.relu(0.10 - variance).square().mean()
    residual = model.adapter(batch.animal_id).residual.squared_norm()
    total = neural + behavior + dynamics + 0.02 * variance_floor + 1e-3 * residual
    return total, neural, behavior


def _move_matched(
    batch: MatchedInterventionBatch, device: torch.device
) -> MatchedInterventionBatch:
    return MatchedInterventionBatch(
        treated=move_batch(batch.treated, device),
        control_neural=batch.control_neural.to(device),
        control_behavior=batch.control_behavior.to(device),
        control_neural_mask=batch.control_neural_mask.to(device),
        control_behavior_mask=batch.control_behavior_mask.to(device),
        control_inputs=batch.control_inputs.to(device),
    )


def _intervention_loss(
    model: HierarchicalControlledSSM,
    batch: MatchedInterventionBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    treated = batch.treated
    pre = treated.onset - 1
    z0, _ = model.encode(
        treated.animal_id,
        treated.neural[:, pre],
        treated.behavior[:, pre],
        sample=False,
    )
    horizon_inputs = treated.inputs[:, pre:-1]
    treated_rollout = model.rollout(
        treated.animal_id,
        z0,
        horizon_inputs,
        treated.intervention[:, pre:-1],
        include_animal_residual=True,
        include_donor_delta=True,
    )
    control_action = torch.zeros_like(treated.intervention[:, pre:-1])
    control_rollout = model.rollout(
        treated.animal_id,
        z0,
        batch.control_inputs[:, pre:-1],
        control_action,
        include_animal_residual=True,
        include_donor_delta=True,
    )
    treated_neural = treated.neural[:, treated.onset :]
    treated_behavior = treated.behavior[:, treated.onset :]
    control_neural = batch.control_neural[:, treated.onset :]
    control_behavior = batch.control_behavior[:, treated.onset :]
    neural_mask = (
        treated.neural_mask[:, treated.onset :] & batch.control_neural_mask[:, treated.onset :]
    )
    behavior_mask = (
        treated.behavior_mask[:, treated.onset :] & batch.control_behavior_mask[:, treated.onset :]
    )
    neural = _masked_mse(
        treated_rollout[1] - control_rollout[1],
        treated_neural - control_neural,
        neural_mask,
    )
    behavior = _masked_mse(
        treated_rollout[2] - control_rollout[2],
        treated_behavior - control_behavior,
        behavior_mask,
    )
    absolute = _masked_mse(
        treated_rollout[1],
        treated_neural,
        treated.neural_mask[:, treated.onset :],
    ) + _masked_mse(
        treated_rollout[2],
        treated_behavior,
        treated.behavior_mask[:, treated.onset :],
    )
    donor_penalty = torch.zeros((), device=z0.device)
    donor_centering = torch.zeros((), device=z0.device)
    if model.donor_intervention_delta:
        deltas = torch.stack(list(model.donor_intervention_delta.values()), dim=0)
        donor_penalty = deltas.square().mean()
        donor_centering = deltas.mean(dim=0).square().mean()
    total = neural + behavior + 0.10 * absolute + 0.01 * donor_penalty + 0.01 * donor_centering
    return total, neural, behavior


def _trainable(model: torch.nn.Module) -> list[torch.Tensor]:
    result = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not result:
        raise ValueError("optimization stage has no trainable parameters")
    return result


def _fit_stage(
    model: HierarchicalControlledSSM,
    train_batches: Sequence[SequenceBatch] | Sequence[MatchedInterventionBatch],
    validation_batches: Sequence[SequenceBatch] | Sequence[MatchedInterventionBatch],
    *,
    stage: Literal["normal", "intervention", "target_adaptation"],
    config: FitConfig,
    target_animal: str | None = None,
    fixed_epochs: int | None = None,
    donor_projection_groups: Sequence[str] | None = None,
) -> FitResult:
    if not train_batches:
        raise ValueError("training collection is empty")
    if fixed_epochs is None and not validation_batches:
        raise ValueError("early stopping requires validation batches")
    seed_everything(config.seed)
    model.configure_stage(stage, target_animal=target_animal)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    if stage == "intervention":
        if donor_projection_groups is None:
            raise ValueError("intervention fit requires declared donor projection groups")
        projection_groups = tuple(donor_projection_groups)
        if set(projection_groups) != set(model.donor_intervention_delta):
            raise ProtocolViolation(
                "intervention projection groups must exactly equal fitted donor deltas"
            )
        initial_projection_norm = model.project_donor_deltas_zero_mean(projection_groups)
        if initial_projection_norm > DELTA_PROJECTION_TOLERANCE:
            raise ProtocolViolation("initial donor-delta projection failed")
        train = [_move_matched(batch, device) for batch in train_batches]  # type: ignore[arg-type]
        validation = [
            _move_matched(batch, device)
            for batch in validation_batches  # type: ignore[arg-type]
        ]
        objective = _intervention_loss
    else:
        if donor_projection_groups is not None:
            raise ValueError("donor projection groups are intervention-stage only")
        projection_groups = ()
        train = [move_batch(batch, device) for batch in train_batches]  # type: ignore[arg-type]
        validation = [
            move_batch(batch, device)
            for batch in validation_batches  # type: ignore[arg-type]
        ]
        objective = _normal_loss
    parameters = _trainable(model)
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    use_amp = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    generator = random.Random(config.seed)
    epochs = fixed_epochs if fixed_epochs is not None else config.max_epochs
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[EpochRecord] = []
    for epoch in range(epochs):
        model.train()
        order = list(range(len(train)))
        generator.shuffle(order)
        train_values = []
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                total, _, _ = objective(model, train[index])  # type: ignore[arg-type]
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            if stage == "intervention":
                projection_norm = model.project_donor_deltas_zero_mean(projection_groups)
                if projection_norm > DELTA_PROJECTION_TOLERANCE:
                    raise ProtocolViolation("donor-delta projection failed after optimizer step")
            train_values.append(float(total.detach().cpu()))
        model.eval()
        validation_values: list[float] = []
        validation_neural: list[float] = []
        validation_behavior: list[float] = []
        with torch.no_grad():
            for batch in validation:
                total, neural, behavior = objective(model, batch)  # type: ignore[arg-type]
                validation_values.append(float(total.detach().cpu()))
                validation_neural.append(float(neural.detach().cpu()))
                validation_behavior.append(float(behavior.detach().cpu()))
        validation_loss = (
            float(np.mean(validation_values)) if validation_values else float(np.mean(train_values))
        )
        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=float(np.mean(train_values)),
                validation_loss=validation_loss,
                neural_loss=float(np.mean(validation_neural))
                if validation_neural
                else float("nan"),
                behavior_loss=float(np.mean(validation_behavior))
                if validation_behavior
                else float("nan"),
            )
        )
        if not np.isfinite(validation_loss):
            raise RuntimeError(f"{stage} optimization produced non-finite loss")
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if fixed_epochs is None and stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError(f"{stage} optimization produced no checkpoint")
    if fixed_epochs is None:
        model.load_state_dict(best_state)
    if stage == "intervention":
        final_projection_norm = model.project_donor_deltas_zero_mean(projection_groups)
        if final_projection_norm > DELTA_PROJECTION_TOLERANCE:
            raise ProtocolViolation("final donor-delta projection failed")
    return FitResult(stage, best_epoch, best_loss, history, config)


@dataclass(frozen=True, slots=True)
class AllenExperimentConfig:
    profile: Literal["smoke", "fast", "full"] = "smoke"
    latent_dim: int = 4
    hidden_dim: int = 24
    residual_rank: int = 2
    intervention_rank: int = 2
    batch_size: int = 16
    max_normal_trials: int | None = 48
    max_omission_trials: int | None = 48
    controls_per_query: int = 3
    learned_methods: tuple[MethodName, ...] = LEARNED_METHODS
    normal_fit: FitConfig = field(
        default_factory=lambda: FitConfig(
            learning_rate=3e-3,
            max_epochs=10,
            patience=3,
            seed=11,
            device="cuda",
            mixed_precision=True,
        )
    )
    intervention_fit: FitConfig = field(
        default_factory=lambda: FitConfig(
            learning_rate=4e-3,
            max_epochs=12,
            patience=4,
            seed=23,
            device="cuda",
            mixed_precision=True,
        )
    )
    target_fit: FitConfig = field(
        default_factory=lambda: FitConfig(
            learning_rate=3e-3,
            max_epochs=10,
            patience=3,
            seed=37,
            device="cuda",
            mixed_precision=True,
        )
    )

    def validate(self) -> None:
        if self.profile not in {"smoke", "fast", "full"}:
            raise ValueError("unknown optimization profile")
        if self.latent_dim < 2 or self.hidden_dim < 4 or self.batch_size < 1:
            raise ValueError("invalid model/batch dimensions")
        if not set(self.learned_methods) <= set(LEARNED_METHODS):
            raise ValueError("unknown learned comparator")


def make_allen_config(
    profile: Literal["smoke", "fast", "full"],
    *,
    seed: int = 0,
    device: str | None = None,
    methods: Sequence[MethodName] | None = None,
) -> AllenExperimentConfig:
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = selected_device.startswith("cuda")
    if profile == "smoke":
        config = AllenExperimentConfig(
            profile=profile,
            normal_fit=replace(
                AllenExperimentConfig().normal_fit,
                seed=seed * 100 + 11,
                device=selected_device,
                mixed_precision=amp,
            ),
            intervention_fit=replace(
                AllenExperimentConfig().intervention_fit,
                seed=seed * 100 + 23,
                device=selected_device,
                mixed_precision=amp,
            ),
            target_fit=replace(
                AllenExperimentConfig().target_fit,
                seed=seed * 100 + 37,
                device=selected_device,
                mixed_precision=amp,
            ),
        )
    elif profile == "fast":
        config = AllenExperimentConfig(
            profile=profile,
            latent_dim=8,
            hidden_dim=48,
            batch_size=32,
            max_normal_trials=96,
            max_omission_trials=96,
            normal_fit=FitConfig(
                learning_rate=2e-3,
                max_epochs=24,
                patience=6,
                seed=seed * 100 + 11,
                device=selected_device,
                mixed_precision=amp,
            ),
            intervention_fit=FitConfig(
                learning_rate=2e-3,
                max_epochs=30,
                patience=7,
                seed=seed * 100 + 23,
                device=selected_device,
                mixed_precision=amp,
            ),
            target_fit=FitConfig(
                learning_rate=2e-3,
                max_epochs=24,
                patience=6,
                seed=seed * 100 + 37,
                device=selected_device,
                mixed_precision=amp,
            ),
        )
    else:
        config = AllenExperimentConfig(
            profile=profile,
            latent_dim=12,
            hidden_dim=96,
            batch_size=64,
            max_normal_trials=None,
            max_omission_trials=None,
            normal_fit=FitConfig(
                learning_rate=1e-3,
                max_epochs=500,
                patience=40,
                seed=seed * 100 + 11,
                device=selected_device,
                mixed_precision=amp,
            ),
            intervention_fit=FitConfig(
                learning_rate=1e-3,
                max_epochs=500,
                patience=40,
                seed=seed * 100 + 23,
                device=selected_device,
                mixed_precision=amp,
            ),
            target_fit=FitConfig(
                learning_rate=1e-3,
                max_epochs=400,
                patience=40,
                seed=seed * 100 + 37,
                device=selected_device,
                mixed_precision=amp,
            ),
        )
    if methods is not None:
        config = replace(config, learned_methods=tuple(methods))
    config.validate()
    return config


def _optimization_protocol_payload(config: AllenExperimentConfig) -> dict[str, Any]:
    """Device-independent optimization identity frozen for locked evaluation."""

    payload = asdict(config)
    for fit_name in ("normal_fit", "intervention_fit", "target_fit"):
        payload[fit_name].pop("device")
        payload[fit_name].pop("mixed_precision")
    return _jsonable(payload)


def _canonical_optimization_sha256(config: AllenExperimentConfig) -> str:
    return _canonical_json_sha256(_optimization_protocol_payload(config))


def _runtime_optimization_sha256(config: AllenExperimentConfig) -> str:
    """Exact stage-local configuration, including permitted device choices."""

    return _canonical_json_sha256(asdict(config))


def _validate_locked_configuration(
    optimization: AllenExperimentConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    """Reject any locked optimization scope not frozen before outcomes."""

    if seed != 0:
        raise ProtocolViolation("locked Allen runs require the canonical run seed=0")
    if optimization.profile != "full":
        raise ProtocolViolation("locked Allen runs require optimization='full'")
    if tuple(optimization.learned_methods) != LEARNED_METHODS:
        raise ProtocolViolation(
            "locked Allen runs require all frozen learned methods in canonical order"
        )
    if optimization.intervention_rank != 2:
        raise ProtocolViolation("locked Allen runs require intervention_rank=2")
    expected = make_allen_config(
        "full",
        seed=0,
        device=optimization.normal_fit.device,
        methods=LEARNED_METHODS,
    )
    observed_payload = _optimization_protocol_payload(optimization)
    expected_payload = _optimization_protocol_payload(expected)
    if observed_payload != expected_payload:
        raise ProtocolViolation(
            "locked Allen optimization differs from the canonical full configuration"
        )
    devices = {
        optimization.normal_fit.device,
        optimization.intervention_fit.device,
        optimization.target_fit.device,
    }
    if len(devices) != 1:
        raise ProtocolViolation("locked Allen fit stages must use one declared device")
    return {
        "profile": "full",
        "run_seed": 0,
        "stage_seeds": {
            "normal": optimization.normal_fit.seed,
            "intervention": optimization.intervention_fit.seed,
            "target_adaptation": optimization.target_fit.seed,
        },
        "learned_methods": list(optimization.learned_methods),
        "ablations": [
            "proposed_no_residual",
            "proposed_no_target_adaptation",
        ],
        "intervention_rank": optimization.intervention_rank,
        "canonical_optimization_sha256": _canonical_optimization_sha256(optimization),
    }


def _make_model(
    method: MethodName,
    config: AllenExperimentConfig,
    *,
    seed: int,
) -> HierarchicalControlledSSM:
    seed_everything(seed)
    arguments: dict[str, Any] = {
        "latent_dim": config.latent_dim,
        "input_dim": INPUT_DIM,
        "behavior_dim": BEHAVIOR_DIM,
        "num_interventions": 1,
        "hidden_dim": config.hidden_dim,
        "residual_rank": config.residual_rank,
        "intervention_rank": config.intervention_rank,
        "dt": 0.1,
    }
    classes = {
        "proposed": HierarchicalControlledSSM,
        "linear": LinearHierarchicalSSM,
        "additive": AdditiveInterventionSSM,
        "black_box": BlackBoxMetaGRU,
    }
    return classes[method](**arguments)


def _adapter_key(animal_id: str) -> str:
    return f"animal_{animal_id.replace('.', '_').replace('/', '_')}"


def _add_zero_donor_delta(
    model: HierarchicalControlledSSM,
    animal_id: str,
) -> str:
    """Add an exact-zero donor effect after normal fitting."""

    adapter_key = _adapter_key(animal_id)
    if adapter_key not in model._intervention_groups:
        raise KeyError(f"animal {animal_id} must be registered before adding a delta")
    group_key = model._intervention_groups[adapter_key]
    if group_key in model.donor_intervention_delta:
        raise ValueError(f"donor delta already exists for {animal_id}")
    reference = model.operator.bias
    model.donor_intervention_delta[group_key] = torch.nn.Parameter(
        torch.zeros(
            model.num_interventions,
            model.latent_dim,
            dtype=reference.dtype,
            device=reference.device,
        )
    )
    return group_key


def _fit_summary(result: FitResult) -> dict[str, Any]:
    return {
        "stage": result.stage,
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "epochs_run": len(result.history),
        "last_train_loss": result.history[-1].train_loss,
    }


def _module_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _fit_inner_validation_adapter(
    model: HierarchicalControlledSSM,
    support: AnimalSupport,
    *,
    config: FitConfig,
    batch_size: int,
    normal_limit: int | None,
) -> tuple[FitResult, dict[str, Any]]:
    """Adapt an inner target on normal support while proving shared F is frozen."""

    shared_before = _module_state_sha256(model.shared)
    behavior_before = _module_state_sha256(model.behavior_decoder)
    result = _fit_stage(
        model,
        normal_batches(
            support,
            "fit",
            batch_size=batch_size,
            limit=normal_limit,
        ),
        normal_batches(
            support,
            "val",
            batch_size=batch_size,
        ),
        stage="target_adaptation",
        target_animal=support.mouse_id,
        config=config,
    )
    shared_after = _module_state_sha256(model.shared)
    behavior_after = _module_state_sha256(model.behavior_decoder)
    if shared_after != shared_before or behavior_after != behavior_before:
        raise ProtocolViolation(
            "inner-validation normal adaptation modified shared normal dynamics F"
        )
    return result, {
        "mouse_id": support.mouse_id,
        "shared_f_frozen": True,
        "normal_fit_partition": "fit",
        "normal_validation_partition": "val",
        "shared_state_sha256_before": shared_before,
        "shared_state_sha256_after": shared_after,
        "behavior_decoder_sha256_before": behavior_before,
        "behavior_decoder_sha256_after": behavior_after,
    }


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted"


def _atomic_npz(
    path: Path,
    *,
    overwrite: bool = False,
    sealed: bool = False,
    **arrays: npt.ArrayLike,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        if sealed:
            os.fchmod(stream.fileno(), 0)
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
        stream.seek(0)
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
        temporary = Path(stream.name)
    try:
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise FileExistsError(f"refusing to overwrite {path}") from error
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()


def prepare_target_query_files(
    support: AnimalSupport,
    output_directory: str | Path,
    *,
    controls_per_query: int,
    max_queries: int | None,
    overwrite: bool = False,
    seal_outcomes_on_publish: bool = False,
) -> tuple[Path, Path, dict[str, float], str]:
    """Materialize input-only and sealed-outcome files for one target."""

    omission = load_omission_data(support)
    indices = np.arange(len(omission.neural), dtype=np.int64)
    indices = np.asarray(
        sorted(
            indices,
            key=lambda index: _stable_digest(
                "allen-target-query-v1",
                LOCK_SEED,
                support.mouse_id,
                int(omission.presentation_ids[index]),
            ),
        ),
        dtype=np.int64,
    )
    if max_queries is not None:
        indices = indices[:max_queries]
    omission = OmissionData(
        neural=omission.neural[indices],
        neural_valid=omission.neural_valid[indices],
        behavior=omission.behavior[indices],
        behavior_valid=omission.behavior_valid[indices],
        inputs=omission.inputs[indices],
        descriptors=omission.descriptors[indices],
        presentation_ids=omission.presentation_ids[indices],
    )
    pre = support.onset - 1
    neural_scaled, neural_mask = support.scaler.transform_neural(
        omission.neural[:, pre], omission.neural_valid[:, pre]
    )
    behavior_scaled, behavior_mask = support.scaler.transform_behavior(
        omission.behavior[:, pre], omission.behavior_valid[:, pre]
    )
    intervention = np.zeros((len(indices), len(support.relative_time_s), 1), np.float32)
    intervention[:, pre, 0] = 1.0
    control_inputs = _counterfactual_control_inputs(
        omission.inputs, omission.descriptors, support.onset
    )

    legal_control_indices = np.concatenate((support.partitions["fit"], support.partitions["val"]))
    legal_matches_local, support_match_audit = match_control_indices(
        omission.descriptors,
        support.normal_descriptors[legal_control_indices],
        controls_per_query=controls_per_query,
    )
    legal_matches = _map_match_indices(legal_control_indices, legal_matches_local)
    baseline_neural, baseline_nvalid = _average_matched(
        support.normal_neural_raw, support.normal_neural_valid, legal_matches
    )
    baseline_behavior, baseline_bvalid = _average_matched(
        support.normal_behavior_raw, support.normal_behavior_valid, legal_matches
    )

    audit_indices = support.partitions["match"]
    if not len(audit_indices):
        audit_indices = support.partitions["audit"]
    outcome_matches_local, outcome_match_audit = match_control_indices(
        omission.descriptors,
        support.normal_descriptors[audit_indices],
        controls_per_query=controls_per_query,
    )
    outcome_matches = _map_match_indices(audit_indices, outcome_matches_local)
    observed_control_neural, observed_control_nvalid = _average_matched(
        support.normal_neural_raw, support.normal_neural_valid, outcome_matches
    )
    observed_control_behavior, observed_control_bvalid = _average_matched(
        support.normal_behavior_raw, support.normal_behavior_valid, outcome_matches
    )

    destination = Path(output_directory) / "queries" / f"mouse_{support.mouse_id}"
    query_path = destination / "query_inputs.npz"
    sealed_path = destination / "sealed_outcomes.npz"
    _atomic_npz(
        query_path,
        overwrite=overwrite,
        mouse_id=np.asarray(support.mouse_id),
        relative_time_s=support.relative_time_s,
        onset=np.asarray(support.onset, dtype=np.int64),
        presentation_ids=omission.presentation_ids,
        descriptors=omission.descriptors,
        pre_neural=neural_scaled,
        pre_neural_mask=neural_mask,
        pre_behavior=behavior_scaled,
        pre_behavior_mask=behavior_mask,
        treated_inputs=omission.inputs,
        control_inputs=control_inputs,
        treated_intervention=intervention,
        baseline_neural=baseline_neural,
        baseline_neural_valid=baseline_nvalid,
        baseline_behavior=baseline_behavior,
        baseline_behavior_valid=baseline_bvalid,
        neural_center=support.scaler.neural_center,
        neural_scale=support.scaler.neural_scale,
        behavior_center=support.scaler.behavior_center,
        behavior_scale=support.scaler.behavior_scale,
        fit_presentation_sha256=np.asarray(support.scaler.fit_presentation_sha256),
    )
    sealed_sha256 = _atomic_npz(
        sealed_path,
        overwrite=overwrite,
        sealed=seal_outcomes_on_publish,
        mouse_id=np.asarray(support.mouse_id),
        presentation_ids=omission.presentation_ids,
        descriptors=omission.descriptors,
        control_fallback_levels=np.asarray(outcome_match_audit["fallback_levels"], dtype="U32"),
        omission_neural=omission.neural[:, support.onset :],
        omission_neural_valid=omission.neural_valid[:, support.onset :],
        omission_behavior=omission.behavior[:, support.onset :],
        omission_behavior_valid=omission.behavior_valid[:, support.onset :],
        matched_control_neural=observed_control_neural[:, support.onset :],
        matched_control_neural_valid=observed_control_nvalid[:, support.onset :],
        matched_control_behavior=observed_control_behavior[:, support.onset :],
        matched_control_behavior_valid=observed_control_bvalid[:, support.onset :],
        normal_matching_presentation_ids=support.normal_rows.iloc[audit_indices][
            "stimulus_presentation_id"
        ].to_numpy(np.int64),
    )
    audit = {
        **{f"support_{key}": value for key, value in support_match_audit.items()},
        **{f"outcome_{key}": value for key, value in outcome_match_audit.items()},
        "target_normal_audit_used_for_optimization": 0.0,
        "post_onset_outcomes_in_query_bundle": 0.0,
    }
    return query_path, sealed_path, audit, sealed_sha256


def _load_query(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        query = {name: arrays[name] for name in arrays.files}
    forbidden = {
        name
        for name in query
        if name.startswith("omission_") or name.startswith("matched_control_")
    }
    if forbidden:
        raise LeakageError(f"post-onset outcomes leaked into query bundle: {sorted(forbidden)}")
    return query


def _predict_model(
    model: HierarchicalControlledSSM,
    mouse_id: str,
    query: Mapping[str, np.ndarray],
    *,
    include_animal_residual: bool = True,
) -> dict[str, FloatArray]:
    device = next(model.parameters()).device
    onset = int(query["onset"])
    pre = onset - 1
    augmented_pre, _ = _augment_neural_with_masks(
        query["pre_neural"],
        query["pre_neural_mask"],
        query["pre_behavior_mask"],
    )
    model.configure_stage("evaluation")
    model.eval()
    with torch.no_grad():
        z0, _ = model.encode(
            mouse_id,
            torch.as_tensor(augmented_pre, dtype=torch.float32, device=device),
            torch.as_tensor(query["pre_behavior"], dtype=torch.float32, device=device),
            sample=False,
        )
        treated = model.rollout(
            mouse_id,
            z0,
            torch.as_tensor(query["treated_inputs"][:, pre:-1], dtype=torch.float32, device=device),
            torch.as_tensor(
                query["treated_intervention"][:, pre:-1],
                dtype=torch.float32,
                device=device,
            ),
            include_animal_residual=include_animal_residual,
            include_donor_delta=False,
        )
        control = model.rollout(
            mouse_id,
            z0,
            torch.as_tensor(query["control_inputs"][:, pre:-1], dtype=torch.float32, device=device),
            torch.zeros(
                (
                    len(query["pre_neural"]),
                    query["control_inputs"][:, pre:-1].shape[1],
                    1,
                ),
                dtype=torch.float32,
                device=device,
            ),
            include_animal_residual=include_animal_residual,
            include_donor_delta=False,
        )
    neural_center = query["neural_center"]
    neural_scale = query["neural_scale"]
    behavior_center = query["behavior_center"]
    behavior_scale = query["behavior_scale"]

    def inverse_neural(values: torch.Tensor) -> FloatArray:
        observed = values[..., : len(neural_center)]
        transformed = observed.cpu().numpy().astype(np.float64) * neural_scale + neural_center
        return np.maximum(np.expm1(np.clip(transformed, 0.0, 12.0)), 0.0)

    def inverse_behavior(values: torch.Tensor) -> FloatArray:
        return values.cpu().numpy().astype(np.float64) * behavior_scale + behavior_center

    return {
        "neural_treated": inverse_neural(treated[1]),
        "neural_control": inverse_neural(control[1]),
        "behavior_treated": inverse_behavior(treated[2]),
        "behavior_control": inverse_behavior(control[2]),
    }


@dataclass(slots=True)
class SharedEffectTemplate:
    """Equal-animal population-effect template with optional nearest donor."""

    animal_neural: dict[str, FloatArray]
    animal_behavior: dict[str, FloatArray]
    global_neural: FloatArray
    global_behavior: FloatArray
    normal_signatures: dict[str, FloatArray]

    def nearest(self, target_signature: FloatArray) -> str:
        return min(
            self.normal_signatures,
            key=lambda mouse: float(
                np.nanmean(np.square(self.normal_signatures[mouse] - target_signature))
            ),
        )


def _normal_signature(support: AnimalSupport) -> FloatArray:
    indices = np.concatenate((support.partitions["fit"], support.partitions["val"]))
    neural, _ = support.scaler.transform_neural(
        support.normal_neural_raw[indices], support.normal_neural_valid[indices]
    )
    behavior, _ = support.scaler.transform_behavior(
        support.normal_behavior_raw[indices], support.normal_behavior_valid[indices]
    )
    return np.concatenate(
        (np.nanmean(neural, axis=(0, 2)), np.nanmean(behavior, axis=0).reshape(-1))
    ).astype(np.float64)


def fit_shared_effect_template(
    supports: Mapping[str, AnimalSupport],
    omissions: Mapping[str, OmissionData],
    *,
    controls_per_query: int,
) -> SharedEffectTemplate:
    animal_neural: dict[str, FloatArray] = {}
    animal_behavior: dict[str, FloatArray] = {}
    signatures: dict[str, FloatArray] = {}
    for mouse, support in supports.items():
        omission = omissions[mouse]
        audit = support.partitions["match"]
        if not len(audit):
            audit = support.partitions["audit"]
        matches_local, _ = match_control_indices(
            omission.descriptors,
            support.normal_descriptors[audit],
            controls_per_query=controls_per_query,
        )
        matches = _map_match_indices(audit, matches_local)
        control_neural, control_nvalid = _average_matched(
            support.normal_neural_raw, support.normal_neural_valid, matches
        )
        control_behavior, control_bvalid = _average_matched(
            support.normal_behavior_raw, support.normal_behavior_valid, matches
        )
        omission_neural, omission_nvalid = support.scaler.transform_neural(
            omission.neural, omission.neural_valid
        )
        matched_neural, matched_nvalid = support.scaler.transform_neural(
            control_neural, control_nvalid
        )
        omission_behavior, omission_bvalid = support.scaler.transform_behavior(
            omission.behavior, omission.behavior_valid
        )
        matched_behavior, matched_bvalid = support.scaler.transform_behavior(
            control_behavior, control_bvalid
        )
        horizon = slice(support.onset, None)
        neural_valid = omission_nvalid[:, horizon] & matched_nvalid[:, horizon]
        behavior_valid = omission_bvalid[:, horizon] & matched_bvalid[:, horizon]
        neural_effect = np.where(
            neural_valid,
            omission_neural[:, horizon] - matched_neural[:, horizon],
            np.nan,
        )
        behavior_effect = np.where(
            behavior_valid,
            omission_behavior[:, horizon] - matched_behavior[:, horizon],
            np.nan,
        )
        animal_neural[mouse] = np.nanmean(neural_effect, axis=(0, 2))
        animal_behavior[mouse] = np.nanmean(behavior_effect, axis=0)
        signatures[mouse] = _normal_signature(support)
    return SharedEffectTemplate(
        animal_neural=animal_neural,
        animal_behavior=animal_behavior,
        global_neural=np.nanmean(np.stack(list(animal_neural.values())), axis=0),
        global_behavior=np.nanmean(np.stack(list(animal_behavior.values())), axis=0),
        normal_signatures=signatures,
    )


@dataclass(slots=True)
class FunctionalAtlas:
    """Cell-functional shared operator from normal tuning to omission effects.

    Every training row is one donor cell in one omission trial.  Features are
    the cell's normal flash-response curve, its last pre-onset activity,
    preceding image identity, and log risk-set position.  Multi-output ridge
    predicts the complete post-onset effect curve.  Sample weights make each
    donor animal contribute exactly equal total weight regardless of its cell
    or trial count.
    """

    ridge: Ridge
    horizon: int

    def predict_neural_effect(
        self,
        support: AnimalSupport,
        query: Mapping[str, np.ndarray],
    ) -> FloatArray:
        fit_val = np.concatenate((support.partitions["fit"], support.partitions["val"]))
        normal, normal_valid = support.scaler.transform_neural(
            support.normal_neural_raw[fit_val],
            support.normal_neural_valid[fit_val],
        )
        normal_curve = np.nan_to_num(np.nanmean(np.where(normal_valid, normal, np.nan), axis=0).T)
        descriptors = np.asarray(query["descriptors"], dtype=np.int64)
        pre = np.asarray(query["pre_neural"], dtype=np.float64)
        rows = []
        for trial, (image, flashes) in enumerate(descriptors):
            image_one_hot = np.zeros((normal_curve.shape[0], INPUT_DIM), dtype=np.float64)
            image_one_hot[:, int(image)] = 1.0
            risk = np.full((normal_curve.shape[0], 1), np.log1p(max(int(flashes), 0)))
            rows.append(np.column_stack((normal_curve, pre[trial, :, None], image_one_hot, risk)))
        predicted = self.ridge.predict(np.concatenate(rows, axis=0))
        return predicted.reshape(len(descriptors), normal_curve.shape[0], self.horizon).transpose(
            0, 2, 1
        )


def fit_functional_atlas(
    supports: Mapping[str, AnimalSupport],
    omissions: Mapping[str, OmissionData],
    *,
    controls_per_query: int,
    max_trials: int | None,
    alpha: float = 10.0,
) -> FunctionalAtlas:
    features: list[FloatArray] = []
    targets: list[FloatArray] = []
    weights: list[FloatArray] = []
    horizon: int | None = None
    for mouse, support in supports.items():
        omission = omissions[mouse]
        indices = np.arange(len(omission.neural), dtype=np.int64)
        indices = np.asarray(
            sorted(
                indices,
                key=lambda index: _stable_digest(
                    "allen-functional-atlas-v1",
                    LOCK_SEED,
                    mouse,
                    int(omission.presentation_ids[index]),
                ),
            ),
            dtype=np.int64,
        )
        if max_trials is not None:
            indices = indices[:max_trials]
        match_pool = support.partitions["match"]
        if not len(match_pool):
            match_pool = support.partitions["audit"]
        matches_local, _ = match_control_indices(
            omission.descriptors[indices],
            support.normal_descriptors[match_pool],
            controls_per_query=controls_per_query,
        )
        matches = _map_match_indices(match_pool, matches_local)
        control, control_valid = _average_matched(
            support.normal_neural_raw, support.normal_neural_valid, matches
        )
        treated, treated_valid = support.scaler.transform_neural(
            omission.neural[indices], omission.neural_valid[indices]
        )
        matched, matched_valid = support.scaler.transform_neural(control, control_valid)
        fit_val = np.concatenate((support.partitions["fit"], support.partitions["val"]))
        normal, normal_valid = support.scaler.transform_neural(
            support.normal_neural_raw[fit_val],
            support.normal_neural_valid[fit_val],
        )
        normal_curve = np.nan_to_num(np.nanmean(np.where(normal_valid, normal, np.nan), axis=0).T)
        pre = support.onset - 1
        trial_features: list[FloatArray] = []
        trial_targets: list[FloatArray] = []
        for local_index, source_index in enumerate(indices):
            image, flashes = omission.descriptors[source_index]
            image_one_hot = np.zeros((normal_curve.shape[0], INPUT_DIM), dtype=np.float64)
            image_one_hot[:, int(image)] = 1.0
            risk = np.full((normal_curve.shape[0], 1), np.log1p(max(int(flashes), 0)))
            trial_features.append(
                np.column_stack(
                    (
                        normal_curve,
                        treated[local_index, pre, :, None],
                        image_one_hot,
                        risk,
                    )
                )
            )
            effect = (
                treated[local_index, support.onset :] - matched[local_index, support.onset :]
            ).T
            valid = (
                treated_valid[local_index, support.onset :]
                & matched_valid[local_index, support.onset :]
            ).T
            # Ridge has no elementwise output mask. Missing samples are rare;
            # zero is the normal-support mean in standardized coordinates.
            trial_targets.append(np.where(valid, effect, 0.0))
        animal_features = np.concatenate(trial_features, axis=0)
        animal_targets = np.concatenate(trial_targets, axis=0)
        horizon = animal_targets.shape[1]
        features.append(animal_features)
        targets.append(animal_targets)
        weights.append(np.full(len(animal_features), 1.0 / len(animal_features), dtype=np.float64))
    if horizon is None:
        raise ValueError("functional atlas requires at least one donor")
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    sample_weight = np.concatenate(weights, axis=0)
    sample_weight *= len(sample_weight) / sample_weight.sum()
    ridge.fit(
        np.concatenate(features, axis=0),
        np.concatenate(targets, axis=0),
        sample_weight=sample_weight,
    )
    return FunctionalAtlas(ridge=ridge, horizon=horizon)


def _functional_atlas_prediction(
    query: Mapping[str, np.ndarray],
    neural_effect: FloatArray,
    behavior_effect: FloatArray,
) -> dict[str, FloatArray]:
    onset = int(query["onset"])
    neural_control = np.asarray(query["baseline_neural"], dtype=np.float64)[:, onset:]
    behavior_control = np.asarray(query["baseline_behavior"], dtype=np.float64)[:, onset:]
    standardized_control = (
        np.log1p(np.maximum(neural_control, 0.0)) - query["neural_center"]
    ) / query["neural_scale"]
    treated_standardized = standardized_control + neural_effect
    transformed = treated_standardized * query["neural_scale"] + query["neural_center"]
    return {
        "neural_treated": np.maximum(np.expm1(np.clip(transformed, 0.0, 12.0)), 0.0),
        "neural_control": neural_control,
        "behavior_treated": (
            behavior_control + behavior_effect[None, :, :] * query["behavior_scale"]
        ),
        "behavior_control": behavior_control,
    }


def _template_prediction(
    query: Mapping[str, np.ndarray],
    neural_effect: FloatArray,
    behavior_effect: FloatArray,
) -> dict[str, FloatArray]:
    neural_control = np.asarray(query["baseline_neural"], dtype=np.float64)[
        :, int(query["onset"]) :
    ]
    behavior_control = np.asarray(query["baseline_behavior"], dtype=np.float64)[
        :, int(query["onset"]) :
    ]
    log_control = np.log1p(np.maximum(neural_control, 0.0))
    neural_treated = np.maximum(
        np.expm1(
            np.clip(
                log_control + neural_effect[None, :, None] * query["neural_scale"],
                0.0,
                12.0,
            )
        ),
        0.0,
    )
    behavior_treated = behavior_control + behavior_effect[None, :, :] * query["behavior_scale"]
    return {
        "neural_treated": neural_treated,
        "neural_control": neural_control,
        "behavior_treated": behavior_treated,
        "behavior_control": behavior_control,
    }


def _expected_stratum_skill(
    predicted_effect: npt.ArrayLike,
    observed_effect: npt.ArrayLike,
    valid_mask: npt.ArrayLike,
    channel_scale: npt.ArrayLike,
    descriptors: npt.ArrayLike,
    fallback_levels: npt.ArrayLike,
) -> tuple[float, dict[str, float]]:
    """Average expected-effect skill equally over condition/fallback strata."""

    predicted = np.asarray(predicted_effect, dtype=np.float64)
    observed = np.asarray(observed_effect, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    descriptor_array = np.asarray(descriptors, dtype=np.int64)
    fallback = np.asarray(fallback_levels).astype(str)
    if len(predicted) != len(descriptor_array) or len(predicted) != len(fallback):
        raise ValueError("stratum metadata does not align with query effects")
    keys = [
        f"image={image}|risk_bin={_flash_risk_bin(int(flashes))}|fallback={level}"
        for (image, flashes), level in zip(descriptor_array, fallback, strict=True)
    ]
    scores: dict[str, float] = {}
    for key in sorted(set(keys)):
        indices = np.asarray([value == key for value in keys], dtype=bool)
        stratum_valid = valid[indices]
        denominator = stratum_valid.sum(axis=0)
        predicted_mean = np.where(stratum_valid, predicted[indices], 0.0).sum(axis=0) / np.maximum(
            denominator, 1
        )
        observed_mean = np.where(stratum_valid, observed[indices], 0.0).sum(axis=0) / np.maximum(
            denominator, 1
        )
        scores[key] = causal_skill(
            predicted_mean,
            observed_mean,
            channel_scale=channel_scale,
            mask=denominator > 0,
        )
    finite = [value for value in scores.values() if np.isfinite(value)]
    return (float(np.mean(finite)) if finite else float("nan")), scores


def _method_score(
    prediction: Mapping[str, np.ndarray],
    sealed: Mapping[str, np.ndarray],
    support: AnimalSupport,
) -> dict[str, Any]:
    observed_neural = sealed["omission_neural"].astype(np.float64)
    observed_behavior = sealed["omission_behavior"].astype(np.float64)
    control_neural = sealed["matched_control_neural"].astype(np.float64)
    control_behavior = sealed["matched_control_behavior"].astype(np.float64)
    neural_mask = sealed["omission_neural_valid"].astype(bool)
    behavior_mask = sealed["omission_behavior_valid"].astype(bool)
    effect_neural_mask = neural_mask & sealed["matched_control_neural_valid"].astype(bool)
    effect_behavior_mask = behavior_mask & sealed["matched_control_behavior_valid"].astype(bool)
    neural_scale = support_scale(support.normal_neural_raw[support.partitions["fit"]])
    behavior_scale = support_scale(support.normal_behavior_raw[support.partitions["fit"]])
    predicted_neural = np.asarray(prediction["neural_treated"], dtype=np.float64)
    predicted_behavior = np.asarray(prediction["behavior_treated"], dtype=np.float64)
    predicted_neural_effect = predicted_neural - np.asarray(
        prediction["neural_control"], dtype=np.float64
    )
    predicted_behavior_effect = predicted_behavior - np.asarray(
        prediction["behavior_control"], dtype=np.float64
    )
    observed_neural_effect = observed_neural - control_neural
    observed_behavior_effect = observed_behavior - control_behavior
    descriptors = sealed["descriptors"].astype(np.int64)
    fallback_levels = sealed["control_fallback_levels"].astype(str)
    neural_expected, neural_strata = _expected_stratum_skill(
        predicted_neural_effect,
        observed_neural_effect,
        effect_neural_mask,
        neural_scale,
        descriptors,
        fallback_levels,
    )
    running_expected, running_strata = _expected_stratum_skill(
        predicted_behavior_effect[..., 0:1],
        observed_behavior_effect[..., 0:1],
        effect_behavior_mask[..., 0:1],
        behavior_scale[0:1],
        descriptors,
        fallback_levels,
    )
    pupil_expected, pupil_strata = _expected_stratum_skill(
        predicted_behavior_effect[..., 1:2],
        observed_behavior_effect[..., 1:2],
        effect_behavior_mask[..., 1:2],
        behavior_scale[1:2],
        descriptors,
        fallback_levels,
    )
    lick_expected, lick_strata = _expected_stratum_skill(
        predicted_behavior_effect[..., 2:3],
        observed_behavior_effect[..., 2:3],
        effect_behavior_mask[..., 2:3],
        behavior_scale[2:3],
        descriptors,
        fallback_levels,
    )
    return {
        "neural_absolute_nrmse": trajectory_nrmse(
            np.where(neural_mask, predicted_neural, np.nan),
            np.where(neural_mask, observed_neural, np.nan),
            channel_scale=neural_scale,
        ),
        "running_absolute_nrmse": trajectory_nrmse(
            np.where(
                behavior_mask[..., 0:1],
                predicted_behavior[..., 0:1],
                np.nan,
            ),
            np.where(
                behavior_mask[..., 0:1],
                observed_behavior[..., 0:1],
                np.nan,
            ),
            channel_scale=behavior_scale[0:1],
        ),
        "pupil_absolute_nrmse": trajectory_nrmse(
            np.where(behavior_mask[..., 1:2], predicted_behavior[..., 1:2], np.nan),
            np.where(behavior_mask[..., 1:2], observed_behavior[..., 1:2], np.nan),
            channel_scale=behavior_scale[1:2],
        ),
        "lick_absolute_nrmse": trajectory_nrmse(
            np.where(behavior_mask[..., 2:3], predicted_behavior[..., 2:3], np.nan),
            np.where(behavior_mask[..., 2:3], observed_behavior[..., 2:3], np.nan),
            channel_scale=behavior_scale[2:3],
        ),
        "neural_causal_skill": neural_expected,
        "running_causal_skill": running_expected,
        "pupil_causal_skill": pupil_expected,
        "lick_causal_skill": lick_expected,
        "neural_trial_causal_skill": causal_skill(
            predicted_neural_effect,
            observed_neural_effect,
            channel_scale=neural_scale,
            mask=effect_neural_mask,
        ),
        "running_trial_causal_skill": causal_skill(
            predicted_behavior_effect[..., 0:1],
            observed_behavior_effect[..., 0:1],
            channel_scale=behavior_scale[0:1],
            mask=effect_behavior_mask[..., 0:1],
        ),
        "pupil_trial_causal_skill": causal_skill(
            predicted_behavior_effect[..., 1:2],
            observed_behavior_effect[..., 1:2],
            channel_scale=behavior_scale[1:2],
            mask=effect_behavior_mask[..., 1:2],
        ),
        "lick_trial_causal_skill": causal_skill(
            predicted_behavior_effect[..., 2:3],
            observed_behavior_effect[..., 2:3],
            channel_scale=behavior_scale[2:3],
            mask=effect_behavior_mask[..., 2:3],
        ),
        "neural_stratum_count": float(len(neural_strata)),
        "running_stratum_count": float(len(running_strata)),
        "pupil_stratum_count": float(len(pupil_strata)),
        "lick_stratum_count": float(len(lick_strata)),
        "neural_stratum_skills": neural_strata,
        "running_stratum_skills": running_strata,
        "pupil_stratum_skills": pupil_strata,
        "lick_stratum_skills": lick_strata,
        "neural_time_r2": time_resolved_r2(
            np.where(neural_mask, predicted_neural, np.nan),
            np.where(neural_mask, observed_neural, np.nan),
        ),
        "running_time_r2": time_resolved_r2(
            np.where(
                behavior_mask[..., 0:1],
                predicted_behavior[..., 0:1],
                np.nan,
            ),
            np.where(
                behavior_mask[..., 0:1],
                observed_behavior[..., 0:1],
                np.nan,
            ),
        ),
        "pupil_time_r2": time_resolved_r2(
            np.where(behavior_mask[..., 1:2], predicted_behavior[..., 1:2], np.nan),
            np.where(behavior_mask[..., 1:2], observed_behavior[..., 1:2], np.nan),
        ),
        "lick_time_r2": time_resolved_r2(
            np.where(behavior_mask[..., 2:3], predicted_behavior[..., 2:3], np.nan),
            np.where(behavior_mask[..., 2:3], observed_behavior[..., 2:3], np.nan),
        ),
    }


def _write_long_metrics(
    metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    relative_time: FloatArray,
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    rows: list[dict[str, Any]] = []
    for method, animals in metrics.items():
        for mouse, values in animals.items():
            for name, value in values.items():
                if isinstance(value, dict):
                    endpoint = name.removesuffix("_stratum_skills")
                    for stratum, item in value.items():
                        rows.append(
                            {
                                "method": method,
                                "mouse_id": mouse,
                                "endpoint": endpoint,
                                "metric": "expected_effect_causal_skill",
                                "stratum": stratum,
                                "time_s": np.nan,
                                "value": float(item),
                            }
                        )
                elif isinstance(value, np.ndarray):
                    endpoint = name.removesuffix("_time_r2")
                    for time_s, item in zip(relative_time, value, strict=True):
                        rows.append(
                            {
                                "method": method,
                                "mouse_id": mouse,
                                "endpoint": endpoint,
                                "metric": "time_resolved_r2",
                                "time_s": float(time_s),
                                "value": float(item),
                            }
                        )
                else:
                    endpoint = name.split("_", maxsplit=1)[0]
                    rows.append(
                        {
                            "method": method,
                            "mouse_id": mouse,
                            "endpoint": endpoint,
                            "metric": name.removeprefix(f"{endpoint}_"),
                            "time_s": np.nan,
                            "value": float(value),
                        }
                    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        pd.DataFrame(rows).to_csv(stream, index=False)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    try:
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise FileExistsError(f"refusing to overwrite {path}") from error
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably create a transaction journal before changing any source mode."""

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
        raise


def _allen_active_seal_path(processed_root: str | Path) -> Path:
    return Path(processed_root) / ALLEN_ACTIVE_SEAL_NAME


def _allen_prepare_guard_path(processed_root: str | Path) -> Path:
    return Path(processed_root) / ALLEN_PREPARE_GUARD_NAME


def _begin_allen_prepare_guard(
    *,
    processed_root: str | Path,
    output: Path,
    fold: int,
    canonical_relative_output: str,
    mice: Sequence[str],
    targets: Sequence[str],
) -> dict[str, Any]:
    """Record split-artifact preexistence before locked prepare can create copies."""

    root = Path(processed_root)
    target_set = set(targets)
    artifacts: list[dict[str, Any]] = []
    for mouse in mice:
        directory = root / f"mouse_{mouse}"
        for name in (
            "normal_support.npz",
            "omission_query.npz",
            "sealed_omission_outcomes.npz",
        ):
            path = directory / name
            record: dict[str, Any] = {
                "mouse": mouse,
                "name": name,
                "path": str(path.resolve()),
                "target_outcome": mouse in target_set and name == "sealed_omission_outcomes.npz",
                "preexisting": path.exists(),
            }
            if path.exists():
                source_stat = path.stat()
                mode = stat.S_IMODE(source_stat.st_mode)
                if record["target_outcome"] and (not mode & 0o444 or not os.access(path, os.R_OK)):
                    raise ProtocolViolation(
                        f"preexisting Allen target role outcome is unreadable: mouse {mouse}"
                    )
                record.update(
                    {
                        "device_id": int(source_stat.st_dev),
                        "inode": int(source_stat.st_ino),
                        "mode": f"{mode:04o}",
                        "sha256": _sha256_path(path),
                    }
                )
            artifacts.append(record)
    payload = {
        "schema": ALLEN_PREPARE_GUARD_SCHEMA,
        "fold": fold,
        "canonical_relative_output": canonical_relative_output,
        "output_path": str(output.resolve()),
        "processed_root": str(root.resolve()),
        "mice": list(mice),
        "targets": list(targets),
        "artifacts": artifacts,
    }
    path = _allen_prepare_guard_path(root)
    _create_exclusive_json(path, payload)
    return {**payload, "sha256": _sha256_path(path)}


def _load_allen_prepare_guard(processed_root: str | Path) -> dict[str, Any] | None:
    path = _allen_prepare_guard_path(processed_root)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != ALLEN_PREPARE_GUARD_SCHEMA or not isinstance(
        payload.get("artifacts"), list
    ):
        raise ProtocolViolation("unknown Allen prepare guard")
    return {**payload, "sha256": _sha256_path(path)}


def _validate_allen_prepare_guard(
    guard: Mapping[str, Any],
    *,
    processed_root: str | Path,
    output: Path,
    fold: int,
    canonical_relative_output: str,
    expected_mice: Sequence[str],
    expected_targets: Sequence[str],
) -> None:
    root = Path(processed_root).resolve()
    mice_value = guard.get("mice")
    targets_value = guard.get("targets")
    artifacts_value = guard.get("artifacts")
    if (
        set(guard)
        != {
            "schema",
            "fold",
            "canonical_relative_output",
            "output_path",
            "processed_root",
            "mice",
            "targets",
            "artifacts",
            "sha256",
        }
        or guard.get("schema") != ALLEN_PREPARE_GUARD_SCHEMA
        or guard.get("fold") != fold
        or guard.get("canonical_relative_output") != canonical_relative_output
        or Path(str(guard.get("output_path", ""))) != output.resolve()
        or Path(str(guard.get("processed_root", ""))) != root
        or re.fullmatch(r"[0-9a-f]{64}", str(guard.get("sha256", ""))) is None
        or not isinstance(mice_value, list)
        or mice_value != list(expected_mice)
        or any(
            not isinstance(mouse, str) or re.fullmatch(r"\d+", mouse) is None
            for mouse in mice_value
        )
        or len(set(mice_value)) != len(mice_value)
        or not isinstance(targets_value, list)
        or targets_value != list(expected_targets)
        or len(set(targets_value)) != len(targets_value)
        or not isinstance(artifacts_value, list)
    ):
        raise ProtocolViolation("Allen prepare guard binding changed")
    encoded_guard = {key: value for key, value in guard.items() if key != "sha256"}
    encoded = (json.dumps(_jsonable(encoded_guard), indent=2, sort_keys=True) + "\n").encode()
    if hashlib.sha256(encoded).hexdigest() != guard["sha256"]:
        raise ProtocolViolation("Allen prepare guard digest changed")
    expected_pairs = {
        (mouse, name)
        for mouse in mice_value
        for name in (
            "normal_support.npz",
            "omission_query.npz",
            "sealed_omission_outcomes.npz",
        )
    }
    observed_pairs: set[tuple[str, str]] = set()
    target_set = set(targets_value)
    for value in artifacts_value:
        if not isinstance(value, Mapping):
            raise ProtocolViolation("Allen prepare guard artifact record is malformed")
        record = dict(value)
        mouse = record.get("mouse")
        name = record.get("name")
        pair = (mouse, name)
        if pair not in expected_pairs or pair in observed_pairs:
            raise ProtocolViolation("Allen prepare guard artifact scope changed")
        observed_pairs.add(pair)
        preexisting = record.get("preexisting")
        target_outcome = mouse in target_set and name == "sealed_omission_outcomes.npz"
        base_keys = {
            "mouse",
            "name",
            "path",
            "target_outcome",
            "preexisting",
        }
        expected_keys = (
            base_keys | {"device_id", "inode", "mode", "sha256"}
            if preexisting is True
            else base_keys
        )
        expected_path = (root / f"mouse_{mouse}" / str(name)).resolve()
        if (
            not isinstance(preexisting, bool)
            or set(record) != expected_keys
            or Path(str(record.get("path", ""))) != expected_path
            or record.get("target_outcome") is not target_outcome
        ):
            raise ProtocolViolation("Allen prepare guard artifact binding changed")
        if preexisting and (
            not isinstance(record.get("device_id"), int)
            or int(record["device_id"]) < 0
            or not isinstance(record.get("inode"), int)
            or int(record["inode"]) < 0
            or re.fullmatch(r"[0-7]{4}", str(record.get("mode", ""))) is None
            or re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", ""))) is None
        ):
            raise ProtocolViolation("Allen prepare guard preexisting record is malformed")
    if observed_pairs != expected_pairs:
        raise ProtocolViolation("Allen prepare guard artifact scope is incomplete")


def _locked_allen_prepare_scope(fold: int) -> tuple[list[str], list[str]]:
    donors, targets = resolve_run_mice(
        _repository_root() / CANONICAL_MANIFEST_RELATIVE,
        profile="locked",
        fold=fold,
        acknowledge_locked=True,
    )
    return [*donors, *targets], targets


def _validate_active_allen_transaction_scope(
    transaction: Mapping[str, Any],
    *,
    processed_root: str | Path,
    output: Path,
    fold: int,
    canonical_relative_output: str,
    expected_targets: Sequence[str],
) -> None:
    """Validate a pre-manifest journal before trusting any recorded path."""

    if (
        set(transaction)
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
        or transaction.get("schema") != ALLEN_SEAL_TRANSACTION_SCHEMA
        or transaction.get("fold") != fold
        or transaction.get("canonical_relative_output") != canonical_relative_output
        or Path(str(transaction.get("output_path", ""))) != output.resolve()
        or Path(str(transaction.get("processed_root", ""))) != Path(processed_root).resolve()
        or transaction.get("targets") != list(expected_targets)
        or transaction.get("active") is not True
        or transaction.get("restore_after_score_commit") is not True
        or re.fullmatch(r"[0-9a-f]{64}", str(transaction.get("prepare_guard_sha256", ""))) is None
        or not isinstance(transaction.get("entries"), list)
    ):
        raise ProtocolViolation("Allen active seal transaction scope changed")
    expected_pairs = [
        (mouse, name)
        for mouse in expected_targets
        for name in ("legacy_combined", "role_sealed", "experiment_sealed")
    ]
    entries = transaction["entries"]
    if len(entries) != len(expected_pairs):
        raise ProtocolViolation("Allen active seal transaction entries are incomplete")
    for value, (mouse, name) in zip(entries, expected_pairs, strict=True):
        if not isinstance(value, Mapping):
            raise ProtocolViolation("Allen active seal transaction entry is malformed")
        entry = dict(value)
        expected_path = _target_outcome_paths(processed_root, output, mouse)[name].resolve()
        original_mode_value = entry.get("original_mode")
        sealed_mode_value = entry.get("sealed_mode")
        modes_well_formed = (
            re.fullmatch(r"[0-7]{4}", str(original_mode_value)) is not None
            and re.fullmatch(r"[0-7]{4}", str(sealed_mode_value)) is not None
        )
        original_mode = int(str(original_mode_value), 8) if modes_well_formed else 0
        sealed_mode = int(str(sealed_mode_value), 8) if modes_well_formed else 0o7777
        expected_modes = (
            original_mode == 0o600 and sealed_mode == 0
            if name == "experiment_sealed"
            else bool(original_mode & 0o444) and sealed_mode == original_mode & ~0o444
        )
        if (
            set(entry)
            != {
                "mouse",
                "name",
                "path",
                "original_mode",
                "sealed_mode",
                "device_id",
                "inode",
                "sha256",
            }
            or entry.get("mouse") != mouse
            or entry.get("name") != name
            or Path(str(entry.get("path", ""))) != expected_path
            or not modes_well_formed
            or not expected_modes
            or not isinstance(entry.get("device_id"), int)
            or int(entry["device_id"]) < 0
            or not isinstance(entry.get("inode"), int)
            or int(entry["inode"]) < 0
            or re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))) is None
        ):
            raise ProtocolViolation("Allen active seal transaction entry binding changed")


def _cleanup_uncommitted_allen_role_artifacts(guard: Mapping[str, Any]) -> None:
    """Remove only role artifacts absent before prepare, sealing outcomes first."""

    for value in guard["artifacts"]:
        record = dict(value)
        path = Path(str(record["path"]))
        if record.get("preexisting") is True:
            source_stat = path.stat()
            if (
                int(source_stat.st_dev) != int(record["device_id"])
                or int(source_stat.st_ino) != int(record["inode"])
                or stat.S_IMODE(source_stat.st_mode) != int(str(record["mode"]), 8)
                or _sha256_path(path) != record["sha256"]
            ):
                raise ProtocolViolation("preexisting Allen role artifact changed during prepare")
            continue
        candidates = [
            path,
            *path.parent.glob(f".{path.name}.*.tmp"),
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            if candidate.is_symlink():
                raise ProtocolViolation("uncommitted Allen role artifact is a symlink")
            if record.get("target_outcome") is True:
                candidate.chmod(0)
                if stat.S_IMODE(candidate.stat().st_mode) != 0 or os.access(candidate, os.R_OK):
                    raise ProtocolViolation(
                        "uncommitted Allen target role outcome could not be sealed"
                    )
            candidate.unlink()
            _fsync_directory(candidate.parent)


def _clear_allen_prepare_guard(processed_root: str | Path) -> None:
    path = _allen_prepare_guard_path(processed_root)
    path.unlink(missing_ok=True)
    _fsync_directory(path.parent)


def _quarantine_path(path: Path) -> Path:
    """Move an interrupted artifact tree aside without deleting evidence."""

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


def _seal_uncommitted_allen_output_copies(output: Path) -> None:
    """Clear every read bit on interrupted experiment-owned outcome copies."""

    if not output.exists():
        return
    candidates = [
        *output.rglob("sealed_outcomes.npz"),
        *output.rglob(".sealed_outcomes.npz.*.tmp"),
    ]
    for path in candidates:
        if path.is_symlink():
            raise ProtocolViolation("interrupted Allen outcome copy is a symlink")
        if path.is_file():
            observed_mode = stat.S_IMODE(path.stat().st_mode)
            sealed_mode = observed_mode & ~0o444
            if observed_mode != sealed_mode:
                path.chmod(sealed_mode)


def _load_allen_seal_transaction(processed_root: str | Path) -> dict[str, Any] | None:
    path = _allen_active_seal_path(processed_root)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != ALLEN_SEAL_TRANSACTION_SCHEMA:
        raise ProtocolViolation("unknown Allen target-seal transaction journal")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ProtocolViolation("Allen target-seal transaction has no entries")
    return payload


def _allen_seal_transaction_record(processed_root: str | Path) -> dict[str, Any]:
    transaction = _load_allen_seal_transaction(processed_root)
    if transaction is None:
        raise ProtocolViolation("Allen active target-seal transaction is missing")
    return {
        **transaction,
        "sha256": _sha256_path(_allen_active_seal_path(processed_root)),
    }


def _validate_allen_seal_transaction_binding(
    preparation: Mapping[str, Any],
    *,
    processed_root: str | Path,
    output: Path,
    canonical_relative_output: str,
    require_active_journal: bool,
) -> str:
    record_value = preparation.get("target_seal_transaction")
    if not isinstance(record_value, Mapping):
        raise ProtocolViolation("Allen preparation omits its target-seal transaction")
    record = dict(record_value)
    digest = record.pop("sha256", None)
    encoded = (json.dumps(_jsonable(record), indent=2, sort_keys=True) + "\n").encode()
    targets = [str(value) for value in preparation.get("targets", ())]
    seals = preparation.get("target_seals")
    expected_entries: list[dict[str, Any]] = []
    if not isinstance(seals, Mapping) or set(seals) != set(targets):
        raise ProtocolViolation("Allen preparation target seals are malformed")
    for mouse in targets:
        records = seals.get(mouse)
        if not isinstance(records, Mapping):
            raise ProtocolViolation("Allen preparation target seal scope is incomplete")
        if set(records) != {
            "legacy_combined",
            "role_sealed",
            "experiment_sealed",
        }:
            raise ProtocolViolation("Allen preparation target seal artifacts are incomplete")
        for name in ("legacy_combined", "role_sealed", "experiment_sealed"):
            value = records[name]
            if not isinstance(value, Mapping) or set(value) != {
                "path",
                "original_mode",
                "sealed_mode",
                "device_id",
                "inode",
                "sha256",
            }:
                raise ProtocolViolation("Allen preparation target seal record is malformed")
            expected_entries.append({"mouse": mouse, "name": name, **dict(value)})
    if (
        not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
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
        or re.fullmatch(r"[0-9a-f]{64}", str(record.get("prepare_guard_sha256", ""))) is None
        or record.get("schema") != ALLEN_SEAL_TRANSACTION_SCHEMA
        or record.get("fold") != preparation.get("fold")
        or record.get("canonical_relative_output") != canonical_relative_output
        or Path(str(record.get("output_path", ""))) != output.resolve()
        or Path(str(record.get("processed_root", ""))) != Path(processed_root).resolve()
        or record.get("targets") != targets
        or record.get("entries") != expected_entries
        or record.get("active") is not True
        or record.get("restore_after_score_commit") is not True
    ):
        raise ProtocolViolation("Allen target-seal transaction binding changed")
    if require_active_journal:
        live = _load_allen_seal_transaction(processed_root)
        journal_path = _allen_active_seal_path(processed_root)
        if live != record or not journal_path.exists() or _sha256_path(journal_path) != digest:
            raise ProtocolViolation("Allen active target-seal journal binding changed")
    return digest


def _stage_completion_paths(output: Path, stage: str) -> tuple[Path, Path]:
    manifest = output / f"{stage}.complete.json"
    return manifest, manifest.with_suffix(manifest.suffix + ".sha256")


def _write_stage_completion(
    output: Path,
    *,
    stage: Literal["prepare", "predict", "score"],
    artifacts: Sequence[Path],
    metadata: Mapping[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    """Atomically commit a stage only after all declared artifacts exist."""

    completion_path, digest_path = _stage_completion_paths(output, stage)
    artifact_digests: dict[str, str] = {}
    for path in artifacts:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(output.resolve()).as_posix()
        except ValueError as error:
            raise ValueError("stage completion artifacts must live under output") from error
        artifact_digests[relative] = _sha256_path(path)
    payload = {
        "schema": STAGE_COMPLETION_SCHEMA,
        "stage": stage,
        "artifacts": artifact_digests,
        "metadata": _jsonable(dict(metadata)),
    }
    _atomic_write_json(completion_path, payload, overwrite=overwrite)
    digest = _sha256_path(completion_path)
    _atomic_write_bytes(
        digest_path,
        f"{digest}  {completion_path.name}\n".encode(),
        overwrite=overwrite,
    )
    return {
        **payload,
        "completion_manifest": completion_path.name,
        "completion_sha256": digest,
    }


def _read_stage_completion(
    output: Path,
    stage: Literal["prepare", "predict", "score"],
) -> dict[str, Any]:
    """Authenticate a completion manifest without opening its listed artifacts."""

    completion_path, digest_path = _stage_completion_paths(output, stage)
    fields = digest_path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != completion_path.name:
        raise LeakageError(f"malformed {stage} completion digest")
    observed = _sha256_path(completion_path)
    if observed != fields[0]:
        raise LeakageError(f"{stage} completion manifest digest mismatch")
    payload = json.loads(completion_path.read_text(encoding="utf-8"))
    if payload.get("schema") != STAGE_COMPLETION_SCHEMA or payload.get("stage") != stage:
        raise LeakageError(f"invalid {stage} completion manifest")
    payload["completion_sha256"] = observed
    return payload


def _verify_completed_artifact(
    output: Path,
    completion: Mapping[str, Any],
    relative_path: str,
) -> str:
    expected = completion.get("artifacts", {}).get(relative_path)
    if not isinstance(expected, str):
        raise LeakageError(f"completion omits required artifact {relative_path}")
    observed = _sha256_path(output / relative_path)
    if observed != expected:
        raise LeakageError(f"completed artifact digest changed: {relative_path}")
    return observed


def _validate_allen_score_completion_artifacts(
    output: Path,
    completion: Mapping[str, Any],
) -> None:
    expected = {"metrics.json", "metrics_long.csv"}
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected:
        raise ProtocolViolation("Allen score completion artifact set changed")
    for relative_path in sorted(expected):
        _verify_completed_artifact(output, completion, relative_path)


def _validate_allen_completion_transaction(
    output: Path,
    stage: Literal["prepare", "predict", "score"],
    expected_sha256: str,
) -> None:
    completion = _read_stage_completion(output, stage)
    if (
        dict(completion.get("metadata", {})).get("target_seal_transaction_sha256")
        != expected_sha256
    ):
        raise ProtocolViolation(f"Allen {stage} completion seal-transaction binding changed")


def _locked_append_only_gate(
    output: Path,
    stage: Literal["prepare", "predict", "score"],
) -> None:
    """Fail closed before any locked stage mutates or opens outcome artifacts."""

    if stage == "prepare":
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("locked prepare requires an empty output directory")
        return
    prediction_or_score = (
        "predictions.npz",
        "predictions.npz.sha256",
        "prediction_run.json",
        "predict.complete.json",
        "predict.complete.json.sha256",
        "metrics.json",
        "metrics_long.csv",
        "score.complete.json",
        "score.complete.json.sha256",
    )
    if stage == "predict":
        existing = [name for name in prediction_or_score if (output / name).exists()]
        if existing:
            raise FileExistsError(
                f"locked predict is append-only; artifacts already exist: {existing}"
            )
        _read_stage_completion(output, "prepare")
        return
    score_artifacts = (
        "metrics.json",
        "metrics_long.csv",
        "score.complete.json",
        "score.complete.json.sha256",
    )
    existing = [name for name in score_artifacts if (output / name).exists()]
    if existing:
        raise FileExistsError(
            f"locked score is append-only; completion/artifacts exist: {existing}"
        )
    _read_stage_completion(output, "predict")


def _target_outcome_paths(
    processed_root: str | Path,
    output: Path,
    mouse: str,
) -> dict[str, Path]:
    directory = Path(processed_root) / f"mouse_{mouse}"
    return {
        "legacy_combined": directory / "windows.npz",
        "role_sealed": directory / "sealed_omission_outcomes.npz",
        "experiment_sealed": (output / "queries" / f"mouse_{mouse}" / "sealed_outcomes.npz"),
    }


def _seal_one_outcome(path: Path) -> dict[str, Any]:
    original_mode = stat.S_IMODE(path.stat().st_mode)
    if not original_mode & 0o444:
        raise ProtocolViolation(f"outcome artifact was already unreadable: {path}")
    sealed_mode = original_mode & ~0o444
    path.chmod(sealed_mode)
    observed_mode = stat.S_IMODE(path.stat().st_mode)
    if observed_mode & 0o444 or os.access(path, os.R_OK):
        path.chmod(original_mode)
        raise ProtocolViolation(f"failed to seal outcome artifact: {path}")
    return {
        "path": str(path.resolve()),
        "original_mode": f"{original_mode:04o}",
        "sealed_mode": f"{sealed_mode:04o}",
    }


def _seal_target_outcomes(
    processed_root: str | Path,
    output: Path,
    targets: Sequence[str],
    *,
    fold: int,
    canonical_relative_output: str,
    experiment_sha256: Mapping[str, str] | None = None,
    prepare_guard_sha256: str | None = None,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    for mouse in targets:
        result[mouse] = {}
        for name, path in _target_outcome_paths(processed_root, output, mouse).items():
            source_stat = path.stat()
            observed_mode = stat.S_IMODE(source_stat.st_mode)
            if name == "experiment_sealed" and not observed_mode & 0o444:
                if observed_mode != 0:
                    raise ProtocolViolation(
                        "pre-sealed Allen experiment outcome mode is not exactly 0000"
                    )
                original_mode = 0o600
                sealed_mode = observed_mode
                if experiment_sha256 is None or mouse not in experiment_sha256:
                    raise ProtocolViolation(
                        "pre-sealed Allen experiment outcome lacks its publication digest"
                    )
                outcome_sha256 = str(experiment_sha256[mouse])
            else:
                original_mode = observed_mode
                if not original_mode & 0o444:
                    raise ProtocolViolation(f"outcome artifact was already unreadable: {path}")
                sealed_mode = original_mode & ~0o444
                outcome_sha256 = _sha256_path(path)
            record = {
                "path": str(path.resolve()),
                "original_mode": f"{original_mode:04o}",
                "sealed_mode": f"{sealed_mode:04o}",
                "device_id": int(source_stat.st_dev),
                "inode": int(source_stat.st_ino),
                "sha256": outcome_sha256,
            }
            result[mouse][name] = record
            entries.append({"mouse": mouse, "name": name, **record})

    journal_path = _allen_active_seal_path(processed_root)
    journal = {
        "schema": ALLEN_SEAL_TRANSACTION_SCHEMA,
        "fold": fold,
        "canonical_relative_output": canonical_relative_output,
        "output_path": str(output.resolve()),
        "processed_root": str(Path(processed_root).resolve()),
        "targets": list(targets),
        "entries": entries,
        "active": True,
        "restore_after_score_commit": True,
    }
    if prepare_guard_sha256 is not None:
        journal["prepare_guard_sha256"] = prepare_guard_sha256
    _create_exclusive_json(journal_path, journal)
    sealed_so_far: list[tuple[Path, int, int, str]] = []
    try:
        for entry in entries:
            path = Path(entry["path"])
            source_stat = path.stat()
            if int(source_stat.st_dev) != int(entry["device_id"]) or int(source_stat.st_ino) != int(
                entry["inode"]
            ):
                raise ProtocolViolation("Allen outcome identity changed during sealing")
            sealed_mode = int(entry["sealed_mode"], 8)
            original_mode = int(entry["original_mode"], 8)
            path.chmod(sealed_mode)
            sealed_so_far.append((path, original_mode, sealed_mode, str(entry["name"])))
            observed_mode = stat.S_IMODE(path.stat().st_mode)
            if observed_mode != sealed_mode or observed_mode & 0o444 or os.access(path, os.R_OK):
                raise ProtocolViolation(f"failed to seal outcome artifact: {path}")
    except BaseException:
        for path, original_mode, sealed_mode, name in reversed(sealed_so_far):
            path.chmod(sealed_mode if name == "experiment_sealed" else original_mode)
        _seal_uncommitted_allen_output_copies(output)
        journal_path.unlink(missing_ok=True)
        _fsync_directory(journal_path.parent)
        raise
    return result


def _assert_target_outcomes_sealed(
    seals: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    processed_root: str | Path,
    output: Path,
    targets: Sequence[str],
    canonical_relative_output: str | None = None,
) -> None:
    if canonical_relative_output is not None:
        transaction = _load_allen_seal_transaction(processed_root)
        if (
            transaction is None
            or transaction.get("canonical_relative_output") != canonical_relative_output
            or Path(str(transaction.get("output_path", ""))) != output.resolve()
        ):
            raise LeakageError("Allen active seal journal is absent or bound to another output")
    for mouse in targets:
        expected_paths = _target_outcome_paths(processed_root, output, mouse)
        records = seals.get(mouse, {})
        if set(records) != set(expected_paths):
            raise LeakageError(f"incomplete target seal record for mouse {mouse}")
        for name, path in expected_paths.items():
            record = records[name]
            if Path(record["path"]) != path.resolve():
                raise LeakageError(f"target seal path changed for mouse {mouse} {name}")
            source_stat = path.stat()
            if int(source_stat.st_dev) != int(record["device_id"]) or int(
                source_stat.st_ino
            ) != int(record["inode"]):
                raise LeakageError(
                    f"target outcome identity changed during prediction: mouse {mouse} {name}"
                )
            observed_mode = stat.S_IMODE(source_stat.st_mode)
            if (
                observed_mode != int(record["sealed_mode"], 8)
                or observed_mode & 0o444
                or os.access(path, os.R_OK)
            ):
                raise LeakageError(
                    f"target outcome is readable during prediction: mouse {mouse} {name}"
                )


def _open_experiment_sealed_for_score(
    path: Path,
    record: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], str]:
    """Explicitly unseal one experiment payload, load it, then reseal it."""

    if Path(record["path"]) != path.resolve():
        raise LeakageError("experiment sealed-outcome path differs from preparation")
    sealed_mode = int(record["sealed_mode"], 8)
    original_mode = int(record["original_mode"], 8)
    source_stat = path.stat()
    if int(source_stat.st_dev) != int(record["device_id"]) or int(source_stat.st_ino) != int(
        record["inode"]
    ):
        raise LeakageError("experiment sealed-outcome identity changed before scoring")
    if stat.S_IMODE(source_stat.st_mode) != sealed_mode:
        raise LeakageError("experiment sealed-outcome mode changed before scoring")
    path.chmod(original_mode)
    try:
        if not os.access(path, os.R_OK):
            raise LeakageError("scorer could not explicitly unseal experiment outcomes")
        observed_sha256 = _sha256_path(path)
        with np.load(path, allow_pickle=False) as arrays:
            payload = {name: arrays[name] for name in arrays.files}
        return payload, observed_sha256
    finally:
        path.chmod(sealed_mode)


def _target_restoration_plan(
    seals: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    targets: Sequence[str],
    canonical_relative_output: str,
    seal_transaction_sha256: str,
) -> dict[str, Any]:
    mice: dict[str, Any] = {}
    for mouse in targets:
        mice[mouse] = {}
        for name, record in seals[mouse].items():
            mice[mouse][name] = {
                "path": str(record["path"]),
                "expected_restored_mode": str(record["original_mode"]),
                "sha256": str(record["sha256"]),
            }
    return {
        "schema": "cadence-allen-target-restoration-plan-v1",
        "restoration_status": "PENDING_POST_SCORE_COMMIT",
        "journal_retained_until_score_commit": True,
        "finalization_manifest": "restore.complete.json",
        "canonical_relative_output": canonical_relative_output,
        "seal_transaction_sha256": seal_transaction_sha256,
        "eligible_for_later_donor_reuse_after_finalization": True,
        "mice": mice,
    }


def _validate_allen_restoration_completion(
    payload: Mapping[str, Any],
    *,
    seals: Mapping[str, Mapping[str, Mapping[str, Any]]],
    targets: Sequence[str],
    canonical_relative_output: str,
    score_completion_sha256: str,
    transaction_sha256: str | None,
) -> None:
    """Validate the durable post-score restoration attestation and its bindings."""

    seal_transaction_sha256 = payload.get("seal_transaction_sha256")
    if (
        payload.get("schema") != ALLEN_RESTORE_COMPLETION_SCHEMA
        or payload.get("restored_after_score_commit") is not True
        or payload.get("eligible_for_later_donor_reuse") is not True
        or payload.get("canonical_relative_output") != canonical_relative_output
        or payload.get("score_completion_sha256") != score_completion_sha256
        or not isinstance(seal_transaction_sha256, str)
        or len(seal_transaction_sha256) != 64
    ):
        raise ProtocolViolation("Allen restoration completion binding changed")
    if transaction_sha256 is not None and seal_transaction_sha256 != transaction_sha256:
        raise ProtocolViolation("Allen restoration completion transaction binding changed")
    mice = payload.get("mice")
    if not isinstance(mice, Mapping) or set(mice) != set(targets):
        raise ProtocolViolation("Allen restoration completion target set changed")
    for mouse in targets:
        records = mice.get(mouse)
        expected = seals.get(mouse)
        if (
            not isinstance(records, Mapping)
            or not isinstance(expected, Mapping)
            or set(records) != set(expected)
        ):
            raise ProtocolViolation(
                f"Allen restoration completion artifacts changed for mouse {mouse}"
            )
        for name, seal_value in expected.items():
            seal = dict(seal_value)
            record_value = records[name]
            if not isinstance(record_value, Mapping):
                raise ProtocolViolation(
                    f"Allen restoration completion is malformed for mouse {mouse} {name}"
                )
            record = dict(record_value)
            if (
                Path(str(record.get("path", ""))) != Path(str(seal["path"])).resolve()
                or record.get("restored_mode") != str(seal["original_mode"])
                or record.get("sha256") != seal["sha256"]
            ):
                raise ProtocolViolation(
                    f"Allen restoration completion artifact binding changed: {mouse} {name}"
                )
            path = Path(str(seal["path"]))
            try:
                source_stat = path.stat()
            except OSError as error:
                raise ProtocolViolation(
                    f"Allen restored target is unavailable: {mouse} {name}"
                ) from error
            if int(source_stat.st_dev) != int(seal["device_id"]) or int(source_stat.st_ino) != int(
                seal["inode"]
            ):
                raise ProtocolViolation(f"Allen restored target identity changed: {mouse} {name}")
            original_mode = int(str(seal["original_mode"]), 8)
            if stat.S_IMODE(source_stat.st_mode) != original_mode or not os.access(path, os.R_OK):
                raise ProtocolViolation(f"Allen restored target mode changed: {mouse} {name}")
            if _sha256_path(path) != seal["sha256"]:
                raise ProtocolViolation(f"Allen restored target digest changed: {mouse} {name}")


def _read_allen_restoration_completion(
    completion_path: Path,
    completion_sidecar: Path,
    *,
    seals: Mapping[str, Mapping[str, Mapping[str, Any]]],
    targets: Sequence[str],
    canonical_relative_output: str,
    score_completion_sha256: str,
    transaction_sha256: str | None,
) -> dict[str, Any]:
    if not completion_path.exists() or not completion_sidecar.exists():
        raise ProtocolViolation("Allen restoration completion publication is incomplete")
    fields = completion_sidecar.read_text(encoding="utf-8").strip().split()
    observed = _sha256_path(completion_path)
    if len(fields) != 2 or fields[1] != completion_path.name or observed != fields[0]:
        raise ProtocolViolation("Allen restoration completion is malformed")
    try:
        payload = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolViolation("Allen restoration completion is malformed") from error
    if not isinstance(payload, Mapping):
        raise ProtocolViolation("Allen restoration completion is malformed")
    _validate_allen_restoration_completion(
        payload,
        seals=seals,
        targets=targets,
        canonical_relative_output=canonical_relative_output,
        score_completion_sha256=score_completion_sha256,
        transaction_sha256=transaction_sha256,
    )
    return {
        **payload,
        "completion_sha256": observed,
    }


def _restore_target_outcomes_after_score(
    seals: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    processed_root: str | Path,
    output: Path,
    targets: Sequence[str],
    preparation: Mapping[str, Any],
    canonical_relative_output: str,
) -> dict[str, Any]:
    """Idempotently finalize restoration only after score completion exists."""

    score_completion = _read_stage_completion(output, "score")
    _validate_allen_score_completion_artifacts(output, score_completion)
    completion_path = output / "restore.complete.json"
    completion_sidecar = completion_path.with_suffix(".json.sha256")
    journal_path = _allen_active_seal_path(processed_root)
    transaction = _load_allen_seal_transaction(processed_root)
    if preparation.get("canonical_relative_output") != canonical_relative_output:
        raise ProtocolViolation("Allen preparation restoration binding changed")
    transaction_sha256 = _validate_allen_seal_transaction_binding(
        preparation,
        processed_root=processed_root,
        output=output,
        canonical_relative_output=canonical_relative_output,
        require_active_journal=transaction is not None,
    )
    for stage in ("prepare", "predict", "score"):
        stage_completion_path, stage_sidecar_path = _stage_completion_paths(output, stage)
        if stage_completion_path.exists() and stage_sidecar_path.exists():
            _validate_allen_completion_transaction(
                output,
                stage,
                transaction_sha256,
            )
    if completion_path.exists() and completion_sidecar.exists():
        payload = _read_allen_restoration_completion(
            completion_path,
            completion_sidecar,
            seals=seals,
            targets=targets,
            canonical_relative_output=canonical_relative_output,
            score_completion_sha256=score_completion["completion_sha256"],
            transaction_sha256=transaction_sha256,
        )
        journal_path.unlink(missing_ok=True)
        _fsync_directory(journal_path.parent)
        return payload
    if completion_path.exists() != completion_sidecar.exists():
        if transaction is None:
            raise ProtocolViolation(
                "Allen restoration completion publication is incomplete and its "
                "active transaction is missing"
            )
        if completion_path.exists():
            try:
                partial_payload = json.loads(completion_path.read_text(encoding="utf-8"))
                if not isinstance(partial_payload, Mapping):
                    raise ProtocolViolation("Allen restoration completion publication is malformed")
                _validate_allen_restoration_completion(
                    partial_payload,
                    seals=seals,
                    targets=targets,
                    canonical_relative_output=canonical_relative_output,
                    score_completion_sha256=score_completion["completion_sha256"],
                    transaction_sha256=transaction_sha256,
                )
            except (OSError, ValueError, TypeError, KeyError, ProtocolViolation):
                _quarantine_known_artifacts(output, (completion_path.name,))
            else:
                digest = _sha256_path(completion_path)
                try:
                    _atomic_write_bytes(
                        completion_sidecar,
                        f"{digest}  {completion_path.name}\n".encode(),
                        overwrite=False,
                    )
                except BaseException:
                    _reseal_allen_transaction(transaction)
                    raise
                journal_path.unlink()
                _fsync_directory(journal_path.parent)
                return {
                    **partial_payload,
                    "completion_sha256": digest,
                }
        else:
            _quarantine_known_artifacts(output, (completion_sidecar.name,))
    if transaction is None:
        raise ProtocolViolation("Allen active target-seal journal is missing before restoration")
    if (
        transaction.get("canonical_relative_output") != canonical_relative_output
        or Path(str(transaction.get("output_path", ""))) != output.resolve()
    ):
        raise ProtocolViolation("Allen target-seal journal output binding changed")

    audit: dict[str, Any] = {}
    restored: list[tuple[Path, int]] = []
    try:
        for mouse in targets:
            audit[mouse] = {}
            for name, record in seals[mouse].items():
                path = Path(record["path"])
                source_stat = path.stat()
                if int(source_stat.st_dev) != int(record["device_id"]) or int(
                    source_stat.st_ino
                ) != int(record["inode"]):
                    raise ProtocolViolation(
                        f"Allen target identity changed before restoration: {mouse}"
                    )
                original_mode = int(record["original_mode"], 8)
                sealed_mode = int(record["sealed_mode"], 8)
                observed_mode = stat.S_IMODE(source_stat.st_mode)
                if observed_mode not in {sealed_mode, original_mode}:
                    raise LeakageError(
                        f"target outcome mode changed before restoration: {mouse} {name}"
                    )
                if observed_mode != original_mode:
                    path.chmod(original_mode)
                restored.append((path, sealed_mode))
                restored_mode = stat.S_IMODE(path.stat().st_mode)
                if restored_mode != original_mode or not os.access(path, os.R_OK):
                    raise ProtocolViolation(
                        f"failed to restore target outcome mode: {mouse} {name}"
                    )
                observed_sha256 = _sha256_path(path)
                if observed_sha256 != record["sha256"]:
                    path.chmod(sealed_mode)
                    raise LeakageError(f"restored outcome digest changed: mouse {mouse} {name}")
                audit[mouse][name] = {
                    "path": str(path.resolve()),
                    "restored_mode": f"{restored_mode:04o}",
                    "sha256": observed_sha256,
                }
    except BaseException:
        for path, sealed_mode in reversed(restored):
            path.chmod(sealed_mode)
        raise
    payload = {
        "schema": ALLEN_RESTORE_COMPLETION_SCHEMA,
        "restored_after_score_commit": True,
        "eligible_for_later_donor_reuse": True,
        "canonical_relative_output": canonical_relative_output,
        "score_completion_sha256": score_completion["completion_sha256"],
        "seal_transaction_sha256": transaction_sha256,
        "mice": audit,
    }
    try:
        _atomic_write_json(completion_path, payload, overwrite=False)
        digest = _sha256_path(completion_path)
        _atomic_write_bytes(
            completion_sidecar,
            f"{digest}  {completion_path.name}\n".encode(),
            overwrite=False,
        )
    except BaseException:
        _reseal_allen_transaction(transaction)
        raise
    journal_path.unlink()
    _fsync_directory(journal_path.parent)
    return {
        **payload,
        "completion_sha256": digest,
    }


def _completion_exists(output: Path, stage: Literal["prepare", "predict", "score"]) -> bool:
    manifest, sidecar = _stage_completion_paths(output, stage)
    if manifest.exists() != sidecar.exists():
        return False
    if not manifest.exists():
        return False
    _read_stage_completion(output, stage)
    return True


def _quarantine_known_artifacts(output: Path, names: Sequence[str]) -> Path | None:
    existing = [output / name for name in names if (output / name).exists()]
    existing.extend(
        path for path in output.glob(".*.tmp") if path.is_file() and path not in existing
    )
    if not existing:
        return None
    recovery = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.interrupted-stage-",
            dir=output.parent,
        )
    )
    for path in existing:
        destination = recovery / path.name
        path.rename(destination)
    _fsync_directory(output)
    _fsync_directory(recovery)
    _fsync_directory(output.parent)
    return recovery


def _reseal_allen_transaction(transaction: Mapping[str, Any]) -> None:
    """Return every transaction entry to its immutable sealed state."""

    for entry_value in transaction["entries"]:
        entry = dict(entry_value)
        path = Path(str(entry["path"]))
        source_stat = path.stat()
        if int(source_stat.st_dev) != int(entry["device_id"]) or int(source_stat.st_ino) != int(
            entry["inode"]
        ):
            raise ProtocolViolation("Allen recovery found a substituted outcome artifact")
        original_mode = int(str(entry["original_mode"]), 8)
        sealed_mode = int(str(entry["sealed_mode"]), 8)
        observed_mode = stat.S_IMODE(source_stat.st_mode)
        if observed_mode == original_mode:
            observed_sha256 = _sha256_path(path)
            path.chmod(sealed_mode)
            if observed_sha256 != entry["sha256"]:
                raise ProtocolViolation("Allen recovery found changed readable target outcomes")
        elif observed_mode != sealed_mode:
            raise ProtocolViolation("Allen recovery found an unexpected target outcome mode")
        if stat.S_IMODE(path.stat().st_mode) != sealed_mode:
            raise ProtocolViolation("Allen recovery could not re-seal target outcomes")


def _rollback_incomplete_allen_prepare(
    transaction: Mapping[str, Any],
    *,
    journal_path: Path,
    output: Path,
    prepare_guard: Mapping[str, Any] | None,
) -> None:
    """Restore modes and preserve an interrupted, uncommitted prepare tree."""

    _seal_uncommitted_allen_output_copies(output)
    newly_created_target_roles = {
        str(value["path"])
        for value in (() if prepare_guard is None else prepare_guard["artifacts"])
        if value.get("target_outcome") is True and value.get("preexisting") is False
    }
    restored: list[tuple[Path, int]] = []
    try:
        for entry_value in transaction["entries"]:
            entry = dict(entry_value)
            path = Path(str(entry["path"]))
            source_stat = path.stat()
            if int(source_stat.st_dev) != int(entry["device_id"]) or int(source_stat.st_ino) != int(
                entry["inode"]
            ):
                raise ProtocolViolation(
                    "Allen prepare recovery found a substituted outcome artifact"
                )
            original_mode = int(str(entry["original_mode"]), 8)
            sealed_mode = int(str(entry["sealed_mode"]), 8)
            observed_mode = stat.S_IMODE(source_stat.st_mode)
            if observed_mode not in {original_mode, sealed_mode}:
                raise ProtocolViolation("Allen prepare recovery found an unexpected outcome mode")
            if entry.get("name") == "experiment_sealed":
                if observed_mode != sealed_mode:
                    path.chmod(sealed_mode)
                continue
            if (
                entry.get("name") == "role_sealed"
                and str(path.resolve()) in newly_created_target_roles
            ):
                if observed_mode == original_mode:
                    observed_sha256 = _sha256_path(path)
                    path.chmod(sealed_mode)
                    if observed_sha256 != entry["sha256"]:
                        raise ProtocolViolation(
                            "Allen prepare recovery found changed target outcomes"
                        )
                continue
            if observed_mode != original_mode:
                path.chmod(original_mode)
            if _sha256_path(path) != entry["sha256"]:
                path.chmod(sealed_mode)
                raise ProtocolViolation("Allen prepare recovery found changed target outcomes")
            restored.append((path, sealed_mode))
    except BaseException:
        for path, sealed_mode in reversed(restored):
            path.chmod(sealed_mode)
        raise
    if prepare_guard is not None:
        _cleanup_uncommitted_allen_role_artifacts(prepare_guard)
    journal_path.unlink()
    _fsync_directory(journal_path.parent)
    _clear_allen_prepare_guard(transaction["processed_root"])
    if output.exists():
        _seal_uncommitted_allen_output_copies(output)
        _quarantine_path(output)


def _recover_allen_locked_stage(
    *,
    processed_root: str | Path,
    output: Path,
    fold: int,
    stage: Literal["prepare", "predict", "score"],
    canonical_relative_output: str,
) -> str | None:
    """Recover interrupted mode transitions and make locked stages resumable."""

    journal_path = _allen_active_seal_path(processed_root)
    transaction = _load_allen_seal_transaction(processed_root)
    prepare_guard = _load_allen_prepare_guard(processed_root)
    expected_guard_mice, expected_guard_targets = _locked_allen_prepare_scope(fold)
    prepare_complete = _completion_exists(output, "prepare")
    predict_complete = _completion_exists(output, "predict")
    score_complete = _completion_exists(output, "score")

    if transaction is None:
        if prepare_guard is not None:
            _validate_allen_prepare_guard(
                prepare_guard,
                processed_root=processed_root,
                output=output,
                fold=fold,
                canonical_relative_output=canonical_relative_output,
                expected_mice=expected_guard_mice,
                expected_targets=expected_guard_targets,
            )
            _cleanup_uncommitted_allen_role_artifacts(prepare_guard)
            _clear_allen_prepare_guard(processed_root)
        if score_complete:
            preparation = json.loads((output / "preparation.json").read_text(encoding="utf-8"))
            _restore_target_outcomes_after_score(
                preparation["target_seals"],
                processed_root=processed_root,
                output=output,
                targets=preparation["targets"],
                preparation=preparation,
                canonical_relative_output=canonical_relative_output,
            )
            return "score_complete"
        if (
            stage == "prepare"
            and output.exists()
            and any(output.iterdir())
            and not prepare_complete
        ):
            _seal_uncommitted_allen_output_copies(output)
            _quarantine_path(output)
        return None
    if (
        transaction.get("fold") != fold
        or transaction.get("canonical_relative_output") != canonical_relative_output
        or Path(str(transaction.get("output_path", ""))) != output.resolve()
    ):
        raise ProtocolViolation(
            "another Allen fold has an active seal transaction; resume that canonical fold first"
        )
    _validate_active_allen_transaction_scope(
        transaction,
        processed_root=processed_root,
        output=output,
        fold=fold,
        canonical_relative_output=canonical_relative_output,
        expected_targets=expected_guard_targets,
    )
    if score_complete:
        preparation = json.loads((output / "preparation.json").read_text(encoding="utf-8"))
        _restore_target_outcomes_after_score(
            preparation["target_seals"],
            processed_root=processed_root,
            output=output,
            targets=preparation["targets"],
            preparation=preparation,
            canonical_relative_output=canonical_relative_output,
        )
        return "score_complete"
    expected_guard_sha256 = transaction.get("prepare_guard_sha256")
    if not prepare_complete and expected_guard_sha256 is not None:
        if prepare_guard is None or prepare_guard.get("sha256") != expected_guard_sha256:
            raise ProtocolViolation("Allen active prepare guard binding changed")
        _validate_allen_prepare_guard(
            prepare_guard,
            processed_root=processed_root,
            output=output,
            fold=fold,
            canonical_relative_output=canonical_relative_output,
            expected_mice=expected_guard_mice,
            expected_targets=expected_guard_targets,
        )
    if not prepare_complete:
        _rollback_incomplete_allen_prepare(
            transaction,
            journal_path=journal_path,
            output=output,
            prepare_guard=prepare_guard,
        )
        return None
    if prepare_guard is not None:
        if (
            expected_guard_sha256 is not None
            and prepare_guard.get("sha256") != expected_guard_sha256
        ):
            raise ProtocolViolation("Allen active prepare guard binding changed")
        _validate_allen_prepare_guard(
            prepare_guard,
            processed_root=processed_root,
            output=output,
            fold=fold,
            canonical_relative_output=canonical_relative_output,
            expected_mice=expected_guard_mice,
            expected_targets=expected_guard_targets,
        )
        _clear_allen_prepare_guard(processed_root)
    preparation = json.loads((output / "preparation.json").read_text(encoding="utf-8"))
    transaction_sha256 = _validate_allen_seal_transaction_binding(
        preparation,
        processed_root=processed_root,
        output=output,
        canonical_relative_output=canonical_relative_output,
        require_active_journal=True,
    )
    _validate_allen_completion_transaction(output, "prepare", transaction_sha256)
    if predict_complete:
        _validate_allen_completion_transaction(output, "predict", transaction_sha256)
    _reseal_allen_transaction(transaction)
    if stage == "prepare":
        return "prepare_complete"
    if not predict_complete:
        if stage != "predict":
            raise ProtocolViolation("Allen predict stage must be resumed before score")
        _quarantine_known_artifacts(
            output,
            (
                "predictions.npz",
                "predictions.npz.sha256",
                "prediction_run.json",
                "predict.complete.json",
                "predict.complete.json.sha256",
            ),
        )
        return None
    if stage == "predict":
        return "predict_complete"
    _quarantine_known_artifacts(
        output,
        (
            "metrics.json",
            "metrics_long.csv",
            "score.complete.json",
            "score.complete.json.sha256",
            "restore.complete.json",
            "restore.complete.json.sha256",
        ),
    )
    return None


def _verify_locked_protocol_scope(
    *,
    processed_root: str | Path,
    manifest_path: str | Path,
    optimization: AllenExperimentConfig,
    seed: int,
    attestation: FreezeAttestation,
    repository: Path,
) -> dict[str, Any]:
    """Verify canonical tracked protocol inputs without opening response arrays."""

    canonical_manifest = (repository / CANONICAL_MANIFEST_RELATIVE).resolve()
    canonical_processed = (repository / CANONICAL_PROCESSED_ROOT_RELATIVE).resolve()
    if Path(manifest_path).resolve() != canonical_manifest:
        raise ProtocolViolation(
            f"locked Allen manifest must be {CANONICAL_MANIFEST_RELATIVE.as_posix()}"
        )
    if Path(processed_root).resolve() != canonical_processed:
        raise ProtocolViolation(
            f"locked Allen processed root must be {CANONICAL_PROCESSED_ROOT_RELATIVE.as_posix()}"
        )
    configuration_audit = _validate_locked_configuration(optimization, seed=seed)
    tracked = {
        name: _tracked_file_audit(repository, relative, attestation)
        for name, relative in {
            "manifest": CANONICAL_MANIFEST_RELATIVE,
            "processed_index": CANONICAL_INDEX_RELATIVE,
            "experiment_config": CANONICAL_CONFIG_RELATIVE,
        }.items()
    }
    manifest = json.loads(canonical_manifest.read_text(encoding="utf-8"))
    index_path = repository / CANONICAL_INDEX_RELATIVE
    index = json.loads(index_path.read_text(encoding="utf-8"))
    config = yaml.safe_load((repository / CANONICAL_CONFIG_RELATIVE).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != EXPECTED_MANIFEST_SCHEMA:
        raise ProtocolViolation("canonical Allen cohort manifest schema changed")
    if manifest.get("dataset", {}).get("release") != EXPECTED_ALLEN_RELEASE:
        raise ProtocolViolation("canonical Allen cohort release changed")
    if manifest.get("selection") != {
        "num_animals": 32,
        "one_experiment_per_mouse": True,
        "tie_breaker": ("preferred_session_then_earliest_acquisition_then_experiment_id"),
        "unit": "mouse_id",
    }:
        raise ProtocolViolation("canonical Allen cohort selection contract changed")
    if index.get("schema") != EXPECTED_INDEX_SCHEMA:
        raise ProtocolViolation("canonical processed index schema changed")
    if index.get("release") != EXPECTED_ALLEN_RELEASE:
        raise ProtocolViolation("canonical processed index release changed")
    if index.get("cohort_manifest") != CANONICAL_MANIFEST_RELATIVE.as_posix():
        raise ProtocolViolation("processed index points to a noncanonical cohort")
    if index.get("animal_count") != 32 or len(index.get("animals", [])) != 32:
        raise ProtocolViolation("processed index must contain exactly 32 animals")
    manifest_ids = {
        str(row["mouse_id"]): int(row["ophys_experiment_id"]) for row in manifest["nwb_files"]
    }
    index_ids = {str(row["mouse_id"]): int(row["ophys_experiment_id"]) for row in index["animals"]}
    if len(manifest_ids) != 32 or len(index_ids) != 32 or index_ids != manifest_ids:
        raise ProtocolViolation(
            "processed index mouse/experiment identities differ from the cohort manifest"
        )
    if config.get("schema") != "cadence-allen-vbo-experiment-v1":
        raise ProtocolViolation("canonical Allen experiment config schema changed")
    sequestration = config.get("sequestration", {})
    if (
        sequestration.get("seed") != LOCK_SEED
        or sequestration.get("development_mice") != list(DEVELOPMENT_MICE)
        or sequestration.get("locked_run_seeds") != [0]
    ):
        raise ProtocolViolation("canonical Allen seed/cohort configuration changed")
    model = config.get("model", {})
    if (
        model.get("intervention_rank") != 2
        or model.get("learned_methods") != list(LEARNED_METHODS)
        or model.get("ablations")
        != {
            "proposed_no_residual": "enabled",
            "proposed_no_target_adaptation": "enabled",
            "pooled": "unavailable_heterogeneous_observation_dimensions",
        }
    ):
        raise ProtocolViolation("canonical Allen method/ablation configuration changed")
    full = config.get("profiles", {}).get("full", {})
    expected_full = {
        "latent_dim": 12,
        "hidden_dim": 96,
        "normal_epochs": 500,
        "intervention_epochs": 500,
        "target_epochs": 400,
        "maximum_normal_fit_windows": None,
        "maximum_omission_queries": None,
    }
    if full != expected_full:
        raise ProtocolViolation("canonical Allen full optimization profile changed")
    return {
        "repository": str(repository.resolve()),
        "attested_commit": attestation.commit,
        "release": EXPECTED_ALLEN_RELEASE,
        "cohort_mouse_count": len(manifest_ids),
        "cohort_identity_sha256": _canonical_json_sha256(manifest_ids),
        "tracked_files": tracked,
        "configuration": configuration_audit,
    }


def _verify_locked_processed_inputs(
    *,
    repository: Path,
    processed_root: str | Path,
    manifest_path: str | Path,
    mouse_ids: Sequence[str],
) -> dict[str, Any]:
    """Hash every canonical legacy array before a steward opens or splits it."""

    index = json.loads((repository / CANONICAL_INDEX_RELATIVE).read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = {str(row["mouse_id"]): row for row in index["animals"]}
    raw_ids = {
        str(row["mouse_id"]): int(row["ophys_experiment_id"]) for row in manifest["nwb_files"]
    }
    requested = tuple(dict.fromkeys(map(str, mouse_ids)))
    if set(requested) - set(rows):
        raise ProtocolViolation("locked fold references mice absent from processed index")
    commitment = index.get("source_content_commitment", {})
    source_names = (
        "stimulus_presentations.parquet",
        "window_index.parquet",
        "windows.npz",
    )
    if commitment != {
        "algorithm": "sha256-canonical-json-v1",
        "files_per_mouse": list(source_names),
        "sha256": commitment.get("sha256"),
    } or not re.fullmatch(r"[0-9a-f]{64}", str(commitment.get("sha256", ""))):
        raise ProtocolViolation("processed index source-content commitment is malformed")
    expected_extractor = {
        "minimum_omissions": 80,
        "normal_calibration_trials_requested": None,
        "selection_seed": LOCK_SEED,
        "window_policy": {
            "normal_contamination_guard_s": 3.0,
            "rate_hz": 10.0,
            "window_end_s": 2.0,
            "window_start_s": -1.0,
        },
    }
    canonical_root = Path(processed_root).resolve()
    verified: dict[str, Any] = {}
    commitment_rows: list[dict[str, Any]] = []
    for mouse in sorted(rows):
        row = rows[mouse]
        expected_arrays = CANONICAL_PROCESSED_ROOT_RELATIVE / f"mouse_{mouse}" / "windows.npz"
        expected_provenance = (
            CANONICAL_PROCESSED_ROOT_RELATIVE / f"mouse_{mouse}" / "provenance.json"
        )
        if row.get("arrays") != expected_arrays.as_posix():
            raise ProtocolViolation(f"noncanonical legacy array path for mouse {mouse}")
        if row.get("provenance") != expected_provenance.as_posix():
            raise ProtocolViolation(f"noncanonical provenance path for mouse {mouse}")
        experiment_id = int(row["ophys_experiment_id"])
        if experiment_id != raw_ids.get(mouse):
            raise ProtocolViolation(f"raw experiment identity mismatch for mouse {mouse}")
        provenance_path = repository / expected_provenance
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        observed_extractor = {
            key: provenance.get("extractor", {}).get(key)
            for key in (
                "minimum_omissions",
                "normal_calibration_trials_requested",
                "selection_seed",
                "window_policy",
            )
        }
        if observed_extractor != expected_extractor:
            raise ProtocolViolation(f"preprocessing configuration mismatch for mouse {mouse}")
        if (
            str(provenance.get("mouse_id")) != mouse
            or int(provenance.get("ophys_experiment_id", -1)) != experiment_id
        ):
            raise ProtocolViolation(f"preprocessing provenance identity mismatch for mouse {mouse}")
        observed_outputs: dict[str, str] = {}
        for name in source_names:
            expected_sha256 = provenance.get("outputs", {}).get(name, {}).get("sha256")
            if not isinstance(expected_sha256, str) or not re.fullmatch(
                r"[0-9a-f]{64}", expected_sha256
            ):
                raise ProtocolViolation(f"preprocessing provenance omits {name} for mouse {mouse}")
            observed_sha256 = _sha256_path(canonical_root / f"mouse_{mouse}" / name)
            if observed_sha256 != expected_sha256:
                raise ProtocolViolation(f"processed {name} digest mismatch for mouse {mouse}")
            observed_outputs[name] = observed_sha256
        if observed_outputs["windows.npz"] != row.get("arrays_sha256"):
            raise ProtocolViolation(
                f"legacy windows.npz digest differs from index for mouse {mouse}"
            )
        commitment_rows.append(
            {
                "mouse_id": mouse,
                "ophys_experiment_id": experiment_id,
                "outputs": observed_outputs,
            }
        )
        verified[mouse] = {
            "legacy_path": expected_arrays.as_posix(),
            "legacy_sha256": observed_outputs["windows.npz"],
            "source_files_sha256": observed_outputs,
            "index_row_sha256": _canonical_json_sha256(row),
            "ophys_experiment_id": experiment_id,
            "preprocessing_configuration_sha256": _canonical_json_sha256(observed_extractor),
        }
    observed_commitment = _canonical_json_sha256(commitment_rows)
    if observed_commitment != commitment["sha256"]:
        raise ProtocolViolation("processed source-content commitment differs from tracked index")
    result = {mouse: verified[mouse] for mouse in requested}
    return {
        "verified_before_split": True,
        "mouse_count": len(requested),
        "source_content_commitment_verified": True,
        "source_content_commitment_sha256": observed_commitment,
        "globally_verified_mouse_count": len(verified),
        "mice": result,
    }


def _run_configuration_sha256(
    *,
    run_profile: str,
    fold: int | None,
    donors: Sequence[str],
    targets: Sequence[str],
    optimization: AllenExperimentConfig,
    seed: int,
) -> str:
    payload = {
        "run_profile": run_profile,
        "fold": fold,
        "donors": list(donors),
        "targets": list(targets),
        "optimization": _optimization_protocol_payload(optimization),
        "seed": seed,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _resolve_staged_identity(
    *,
    manifest_path: str | Path,
    run_profile: Literal["development", "locked"],
    fold: int | None,
    acknowledge_locked: bool,
    development_target: str,
    development_donors: Sequence[str],
) -> tuple[list[str], list[str], FreezeAttestation | None]:
    attestation = (
        attest_preoutcome_freeze(repository=_repository_root()) if run_profile == "locked" else None
    )
    donors, targets = resolve_run_mice(
        manifest_path,
        profile=run_profile,
        fold=fold,
        development_target=development_target,
        development_donors=development_donors,
        acknowledge_locked=acknowledge_locked,
    )
    return donors, targets, attestation


def _prepare_allen_stage(
    *,
    processed_root: str | Path,
    output: Path,
    run_profile: Literal["development", "locked"],
    fold: int | None,
    donors: list[str],
    targets: list[str],
    attestation: FreezeAttestation | None,
    optimization: AllenExperimentConfig,
    seed: int,
    overwrite: bool,
    locked_scope_audit: Mapping[str, Any] | None = None,
    canonical_relative_output: str | None = None,
) -> dict[str, Any]:
    preparation_path = output / "preparation.json"
    if run_profile == "locked":
        _locked_append_only_gate(output, "prepare")
    if preparation_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {preparation_path}")
    processed_input_audit = (
        _verify_locked_processed_inputs(
            repository=_repository_root(),
            processed_root=processed_root,
            manifest_path=(_repository_root() / CANONICAL_MANIFEST_RELATIVE),
            mouse_ids=(*donors, *targets),
        )
        if run_profile == "locked"
        else None
    )
    output.mkdir(parents=True, exist_ok=True)
    prepare_guard = (
        _begin_allen_prepare_guard(
            processed_root=processed_root,
            output=output,
            fold=int(fold),
            canonical_relative_output=str(canonical_relative_output),
            mice=(*donors, *targets),
            targets=targets,
        )
        if run_profile == "locked" and fold is not None and canonical_relative_output is not None
        else None
    )
    role_artifacts = prepare_role_separated_artifacts(
        processed_root,
        (*donors, *targets),
        overwrite=False,
        verify_against_combined=run_profile == "locked",
    )
    supports = {
        mouse: load_animal_support(processed_root, mouse, require_physical_split=True)
        for mouse in targets
    }
    query_audits: dict[str, Any] = {}
    experiment_artifacts: dict[str, dict[str, str]] = {}
    for mouse, support in supports.items():
        query_path, sealed_path, audit, sealed_sha256 = prepare_target_query_files(
            support,
            output,
            controls_per_query=optimization.controls_per_query,
            max_queries=optimization.max_omission_trials,
            overwrite=overwrite and run_profile == "development",
            seal_outcomes_on_publish=run_profile == "locked",
        )
        query_audits[mouse] = audit
        experiment_artifacts[mouse] = {
            "query_inputs.npz": _sha256_path(query_path),
            "sealed_outcomes.npz": sealed_sha256,
        }
    if run_profile == "locked":
        if fold is None or canonical_relative_output is None:
            raise AssertionError("locked Allen prepare lacks canonical output binding")
        target_seals = _seal_target_outcomes(
            processed_root,
            output,
            targets,
            fold=fold,
            canonical_relative_output=canonical_relative_output,
            experiment_sha256={
                mouse: experiment_artifacts[mouse]["sealed_outcomes.npz"] for mouse in targets
            },
            prepare_guard_sha256=(None if prepare_guard is None else str(prepare_guard["sha256"])),
        )
        target_seal_transaction = _allen_seal_transaction_record(processed_root)
    else:
        target_seals = {}
        target_seal_transaction = None
    payload = {
        "schema": "cadence-allen-vbo-preparation-v1",
        "run_profile": run_profile,
        "fold": fold,
        "donors": donors,
        "targets": targets,
        "seed": seed,
        "canonical_relative_output": canonical_relative_output,
        "configuration_sha256": _run_configuration_sha256(
            run_profile=run_profile,
            fold=fold,
            donors=donors,
            targets=targets,
            optimization=optimization,
            seed=seed,
        ),
        "canonical_optimization_sha256": _canonical_optimization_sha256(optimization),
        "preparation_runtime_optimization_sha256": (_runtime_optimization_sha256(optimization)),
        "freeze_attestation": (None if attestation is None else asdict(attestation)),
        "role_artifacts": role_artifacts,
        "experiment_artifacts": experiment_artifacts,
        "query_audits": query_audits,
        "locked_scope_audit": (None if locked_scope_audit is None else dict(locked_scope_audit)),
        "processed_input_audit": processed_input_audit,
        "target_seals": target_seals,
        "target_seal_transaction": target_seal_transaction,
    }
    transaction_sha256 = (
        None if target_seal_transaction is None else target_seal_transaction["sha256"]
    )
    if run_profile == "locked":
        _validate_allen_seal_transaction_binding(
            payload,
            processed_root=processed_root,
            output=output,
            canonical_relative_output=str(canonical_relative_output),
            require_active_journal=True,
        )
    _atomic_write_json(
        preparation_path,
        payload,
        overwrite=overwrite and run_profile == "development",
    )
    preparation_sha256 = _sha256_path(preparation_path)
    completion = _write_stage_completion(
        output,
        stage="prepare",
        artifacts=[
            preparation_path,
            *[output / "queries" / f"mouse_{mouse}" / "query_inputs.npz" for mouse in targets],
        ],
        metadata={
            "configuration_sha256": payload["configuration_sha256"],
            "preparation_sha256": preparation_sha256,
            "target_outcomes_physically_sealed": run_profile == "locked",
            "canonical_relative_output": canonical_relative_output,
            "target_seal_transaction_sha256": transaction_sha256,
        },
        overwrite=overwrite and run_profile == "development",
    )
    if run_profile == "locked":
        _clear_allen_prepare_guard(processed_root)
    payload["preparation_sha256"] = preparation_sha256
    payload["stage_completion"] = completion
    return payload


def _read_preparation(
    output: Path,
    *,
    expected_configuration_sha256: str,
) -> tuple[dict[str, Any], str]:
    path = output / "preparation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("configuration_sha256") != expected_configuration_sha256:
        raise LeakageError("prepared query configuration differs from this staged run")
    return payload, _sha256_path(path)


def _verify_safe_preparation_inputs(
    preparation: Mapping[str, Any],
    processed_root: str | Path,
    *,
    donors: Sequence[str],
    targets: Sequence[str],
    output: Path,
) -> None:
    """Verify predict-stage inputs without opening any target sealed file."""

    root = Path(processed_root)
    for mouse in (*donors, *targets):
        expected = preparation["role_artifacts"][mouse]
        names = ["normal_support.npz", "omission_query.npz"]
        if mouse in donors:
            names.append("sealed_omission_outcomes.npz")
        for name in names:
            observed = _sha256_path(root / f"mouse_{mouse}" / name)
            if observed != expected[name]:
                raise LeakageError(f"prepared artifact digest changed: mouse {mouse} {name}")
    for mouse in targets:
        path = output / "queries" / f"mouse_{mouse}" / "query_inputs.npz"
        expected = preparation["experiment_artifacts"][mouse]["query_inputs.npz"]
        if _sha256_path(path) != expected:
            raise LeakageError(f"target query-input digest changed for mouse {mouse}")


def _selected_report_methods(
    optimization: AllenExperimentConfig,
) -> tuple[str, ...]:
    ablations = (
        ("proposed_no_residual", "proposed_no_target_adaptation")
        if "proposed" in optimization.learned_methods
        else ()
    )
    return (
        *optimization.learned_methods,
        *ablations,
        "functional_atlas",
        "no_effect",
        "condition_time",
        "nearest_donor",
    )


def _predict_allen_stage(
    *,
    processed_root: str | Path,
    output: Path,
    run_profile: Literal["development", "locked"],
    fold: int | None,
    donors: list[str],
    targets: list[str],
    attestation: FreezeAttestation | None,
    optimization: AllenExperimentConfig,
    seed: int,
    overwrite: bool,
    locked_scope_audit: Mapping[str, Any] | None = None,
    canonical_relative_output: str | None = None,
) -> dict[str, Any]:
    prediction_path = output / "predictions.npz"
    prepare_completion_path, _ = _stage_completion_paths(output, "prepare")
    prepare_completion = (
        _read_stage_completion(output, "prepare") if prepare_completion_path.exists() else None
    )
    if run_profile == "locked":
        _locked_append_only_gate(output, "predict")
        prepare_completion = _read_stage_completion(output, "prepare")
    if prediction_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {prediction_path}")
    configuration_sha256 = _run_configuration_sha256(
        run_profile=run_profile,
        fold=fold,
        donors=donors,
        targets=targets,
        optimization=optimization,
        seed=seed,
    )
    if prepare_completion is not None:
        _verify_completed_artifact(
            output,
            prepare_completion,
            "preparation.json",
        )
    preparation, preparation_sha256 = _read_preparation(
        output, expected_configuration_sha256=configuration_sha256
    )
    if preparation.get("canonical_relative_output") != canonical_relative_output:
        raise LeakageError("preparation canonical output binding differs from predictor")
    if (
        prepare_completion is not None
        and dict(prepare_completion.get("metadata", {})).get("canonical_relative_output")
        != canonical_relative_output
    ):
        raise LeakageError("prepare completion canonical output binding differs from predictor")
    current_attestation = None if attestation is None else asdict(attestation)
    if preparation.get("freeze_attestation") != current_attestation:
        raise LeakageError("preparation freeze attestation differs from predictor")
    if preparation.get("locked_scope_audit") != (
        None if locked_scope_audit is None else dict(locked_scope_audit)
    ):
        raise LeakageError("preparation locked-scope audit differs from predictor")
    transaction_sha256 = None
    if run_profile == "locked":
        transaction_sha256 = _validate_allen_seal_transaction_binding(
            preparation,
            processed_root=processed_root,
            output=output,
            canonical_relative_output=str(canonical_relative_output),
            require_active_journal=True,
        )
        if (
            dict(prepare_completion.get("metadata", {})).get("target_seal_transaction_sha256")
            != transaction_sha256
        ):
            raise LeakageError("prepare completion seal-transaction binding differs")
        _assert_target_outcomes_sealed(
            preparation["target_seals"],
            processed_root=processed_root,
            output=output,
            targets=targets,
            canonical_relative_output=canonical_relative_output,
        )
    _verify_safe_preparation_inputs(
        preparation,
        processed_root,
        donors=donors,
        targets=targets,
        output=output,
    )
    supports = {
        mouse: load_animal_support(processed_root, mouse, require_physical_split=True)
        for mouse in (*donors, *targets)
    }
    reference_time = supports[donors[0]].relative_time_s
    if any(
        not np.array_equal(support.relative_time_s, reference_time) for support in supports.values()
    ):
        raise ValueError("mice do not share the frozen relative-time grid")
    inner_validation = sorted(
        donors,
        key=lambda mouse: _stable_digest("allen-inner-donor-v1", LOCK_SEED, fold, mouse),
    )[-1]
    train_donors = [mouse for mouse in donors if mouse != inner_validation]
    if not train_donors:
        raise ValueError("inner validation must leave at least one training donor")
    donor_omissions = {
        mouse: load_omission_data(supports[mouse], require_physical_split=True) for mouse in donors
    }
    donor_intervention: dict[str, list[MatchedInterventionBatch]] = {}
    matching_audits: dict[str, Any] = {}
    for mouse in donors:
        batches, audit = intervention_batches(
            supports[mouse],
            donor_omissions[mouse],
            batch_size=optimization.batch_size,
            max_trials=optimization.max_omission_trials,
            controls_per_query=optimization.controls_per_query,
        )
        donor_intervention[mouse] = batches
        matching_audits[mouse] = audit
    template = fit_shared_effect_template(
        {mouse: supports[mouse] for mouse in donors},
        donor_omissions,
        controls_per_query=optimization.controls_per_query,
    )
    functional_atlas = fit_functional_atlas(
        {mouse: supports[mouse] for mouse in donors},
        donor_omissions,
        controls_per_query=optimization.controls_per_query,
        max_trials=optimization.max_omission_trials,
    )
    selection_normal_train = [
        batch
        for mouse in train_donors
        for batch in normal_batches(
            supports[mouse],
            "fit",
            batch_size=optimization.batch_size,
            limit=optimization.max_normal_trials,
        )
    ]
    selection_normal_validation = [
        batch
        for mouse in train_donors
        for batch in normal_batches(
            supports[mouse],
            "val",
            batch_size=optimization.batch_size,
            limit=None
            if optimization.max_normal_trials is None
            else max(8, optimization.max_normal_trials // 3),
        )
    ]
    refit_normal_train = [
        batch
        for mouse in donors
        for partition in ("fit", "val")
        for batch in normal_batches(
            supports[mouse],
            partition,
            batch_size=optimization.batch_size,
            limit=optimization.max_normal_trials,
        )
    ]
    report_methods = _selected_report_methods(optimization)
    predictions: dict[str, dict[str, dict[str, FloatArray]]] = {
        method: {} for method in report_methods
    }
    arrays: dict[str, np.ndarray] = {}
    stage_records: dict[str, Any] = {}
    for method_index, method in enumerate(optimization.learned_methods):
        model_seed = seed * 10_000 + method_index * 101 + 7
        selection_model = _make_model(
            method,
            optimization,
            seed=model_seed,
        )
        for mouse in train_donors:
            selection_model.register_animal(mouse, supports[mouse].model_neural_dim, donor=False)
        selection_normal_result = _fit_stage(
            selection_model,
            selection_normal_train,
            selection_normal_validation,
            stage="normal",
            config=replace(
                optimization.normal_fit,
                seed=optimization.normal_fit.seed + method_index,
            ),
        )
        selected_normal_epochs = selection_normal_result.best_epoch + 1
        if selection_model.donor_intervention_delta:
            raise AssertionError("post-normal checkpoint unexpectedly has donor deltas")
        selection_device = next(selection_model.parameters()).device
        selection_model.register_animal(
            inner_validation,
            supports[inner_validation].model_neural_dim,
            donor=False,
        )
        selection_model.to(selection_device)
        validation_adapter_result, validation_boundary = _fit_inner_validation_adapter(
            selection_model,
            supports[inner_validation],
            config=replace(
                optimization.target_fit,
                seed=optimization.target_fit.seed + method_index,
            ),
            batch_size=optimization.batch_size,
            normal_limit=optimization.max_normal_trials,
        )
        selection_groups = {
            mouse: _add_zero_donor_delta(selection_model, mouse) for mouse in train_donors
        }
        validation_group = selection_model._intervention_groups[_adapter_key(inner_validation)]
        if validation_group in selection_model.donor_intervention_delta:
            raise AssertionError("inner-validation donor delta leaked into selection")
        selection_result = _fit_stage(
            selection_model,
            [batch for mouse in train_donors for batch in donor_intervention[mouse]],
            donor_intervention[inner_validation],
            stage="intervention",
            config=replace(
                optimization.intervention_fit,
                seed=optimization.intervention_fit.seed + method_index,
            ),
            donor_projection_groups=tuple(selection_groups.values()),
        )
        if validation_group in selection_model.donor_intervention_delta:
            raise AssertionError("selection created a validation-donor delta")
        selected_intervention_epochs = selection_result.best_epoch + 1
        selection_final_mean_norm = selection_model.project_donor_deltas_zero_mean(
            tuple(selection_groups.values())
        )
        if selection_final_mean_norm > DELTA_PROJECTION_TOLERANCE:
            raise ProtocolViolation("selection donor deltas are not zero mean")

        model = _make_model(method, optimization, seed=model_seed)
        for mouse in donors:
            model.register_animal(mouse, supports[mouse].model_neural_dim, donor=False)
        refit_normal_result = _fit_stage(
            model,
            refit_normal_train,
            [],
            stage="normal",
            config=replace(
                optimization.normal_fit,
                seed=optimization.normal_fit.seed + method_index,
            ),
            fixed_epochs=selected_normal_epochs,
        )
        if model.donor_intervention_delta:
            raise AssertionError("fresh normal refit unexpectedly has donor deltas")
        refit_groups = {mouse: _add_zero_donor_delta(model, mouse) for mouse in donors}
        if set(refit_groups) != set(donors):
            raise AssertionError("refit donor-delta topology is incomplete")
        refit_result = _fit_stage(
            model,
            [batch for mouse in donors for batch in donor_intervention[mouse]],
            [],
            stage="intervention",
            config=replace(
                optimization.intervention_fit,
                seed=optimization.intervention_fit.seed + method_index,
            ),
            fixed_epochs=selected_intervention_epochs,
            donor_projection_groups=tuple(refit_groups.values()),
        )
        refit_final_mean_norm = model.project_donor_deltas_zero_mean(tuple(refit_groups.values()))
        if refit_final_mean_norm > DELTA_PROJECTION_TOLERANCE:
            raise ProtocolViolation("refit donor deltas are not zero mean")
        method_stages: dict[str, Any] = {
            "normal": _fit_summary(refit_normal_result),
            "normal_selection": _fit_summary(selection_normal_result),
            "inner_validation_normal_adaptation": _fit_summary(validation_adapter_result),
            "normal_refit": _fit_summary(refit_normal_result),
            "intervention_selection": _fit_summary(selection_result),
            "intervention_refit": _fit_summary(refit_result),
            "inner_validation_mouse": inner_validation,
            "selection_boundary": {
                "shared_f_fit_mice": train_donors,
                "shared_f_excluded_mice": [inner_validation, *targets],
                "inner_validation_mimics_outer_target": True,
                "inner_validation_adapter": validation_boundary,
                "selected_normal_epochs": selected_normal_epochs,
                "selected_intervention_epochs": selected_intervention_epochs,
                "intervention_training_mice": train_donors,
                "intervention_validation_mice": [inner_validation],
            },
            "refit_boundary": {
                "fresh_model": True,
                "normal_refit_mice": donors,
                "normal_refit_partitions": ["fit", "val"],
                "intervention_refit_mice": donors,
                "normal_fixed_epochs": selected_normal_epochs,
                "intervention_fixed_epochs": selected_intervention_epochs,
            },
            "selection_delta_groups": selection_groups,
            "selection_validation_delta_present": False,
            "selection_final_delta_mean_norm": selection_final_mean_norm,
            "refit_delta_groups": refit_groups,
            "refit_final_delta_mean_norm": refit_final_mean_norm,
            "delta_projection_tolerance": DELTA_PROJECTION_TOLERANCE,
            "targets": {},
        }
        for target_index, mouse in enumerate(targets):
            support = supports[mouse]
            model_device = next(model.parameters()).device
            model.register_animal(mouse, support.model_neural_dim, donor=False)
            model.to(model_device)
            unadapted = copy.deepcopy(model) if method == "proposed" else None
            target_result = _fit_stage(
                model,
                normal_batches(
                    support,
                    "fit",
                    batch_size=optimization.batch_size,
                    limit=optimization.max_normal_trials,
                ),
                normal_batches(
                    support,
                    "val",
                    batch_size=optimization.batch_size,
                ),
                stage="target_adaptation",
                target_animal=mouse,
                config=replace(
                    optimization.target_fit,
                    seed=(optimization.target_fit.seed + method_index * 17 + target_index),
                ),
            )
            query_path = output / "queries" / f"mouse_{mouse}" / "query_inputs.npz"
            query = _load_query(query_path)
            prediction = _predict_model(model, mouse, query)
            predictions[method][mouse] = prediction
            method_stages["targets"][mouse] = _fit_summary(target_result)
            for name, values in prediction.items():
                arrays[f"{method}__{mouse}__{name}"] = values
            if method == "proposed":
                no_residual = _predict_model(model, mouse, query, include_animal_residual=False)
                if unadapted is None:
                    raise AssertionError("missing unadapted proposed model")
                no_adaptation = _predict_model(unadapted, mouse, query)
                for ablation_name, ablation_prediction in (
                    ("proposed_no_residual", no_residual),
                    ("proposed_no_target_adaptation", no_adaptation),
                ):
                    predictions[ablation_name][mouse] = ablation_prediction
                    for name, values in ablation_prediction.items():
                        arrays[f"{ablation_name}__{mouse}__{name}"] = values
        stage_records[method] = method_stages

    for mouse in targets:
        query_path = output / "queries" / f"mouse_{mouse}" / "query_inputs.npz"
        query = _load_query(query_path)
        baseline_predictions = {
            "functional_atlas": _functional_atlas_prediction(
                query,
                functional_atlas.predict_neural_effect(supports[mouse], query),
                template.global_behavior,
            ),
            "no_effect": _template_prediction(
                query,
                np.zeros_like(template.global_neural),
                np.zeros_like(template.global_behavior),
            ),
            "condition_time": _template_prediction(
                query, template.global_neural, template.global_behavior
            ),
        }
        nearest_mouse = template.nearest(_normal_signature(supports[mouse]))
        baseline_predictions["nearest_donor"] = _template_prediction(
            query,
            template.animal_neural[nearest_mouse],
            template.animal_behavior[nearest_mouse],
        )
        for method, prediction in baseline_predictions.items():
            predictions[method][mouse] = prediction
            for name, values in prediction.items():
                arrays[f"{method}__{mouse}__{name}"] = values
        arrays[f"nearest_donor__{mouse}__source_mouse"] = np.asarray(nearest_mouse)

    metadata = {
        "run_profile": run_profile,
        "canonical_relative_output": canonical_relative_output,
        "target_seal_transaction_sha256": transaction_sha256,
        "fold": fold,
        "donors": donors,
        "targets": targets,
        "inner_validation_mouse": inner_validation,
        "report_methods": list(report_methods),
        "configuration_sha256": configuration_sha256,
        "canonical_optimization_sha256": _canonical_optimization_sha256(optimization),
        "prediction_runtime_optimization_sha256": (_runtime_optimization_sha256(optimization)),
        "preparation_sha256": preparation_sha256,
        "prepare_completion_sha256": (
            None if prepare_completion is None else prepare_completion["completion_sha256"]
        ),
        "optimization": _jsonable(asdict(optimization)),
        "git_commit": _git_commit(),
        "freeze_attestation": (None if attestation is None else asdict(attestation)),
        "locked_scope_audit": (None if locked_scope_audit is None else dict(locked_scope_audit)),
    }
    _atomic_npz(
        prediction_path,
        overwrite=overwrite and run_profile == "development",
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        **arrays,
    )
    prediction_sha256 = _sha256_path(prediction_path)
    prediction_sidecar = output / "predictions.npz.sha256"
    _atomic_write_bytes(
        prediction_sidecar,
        f"{prediction_sha256}  predictions.npz\n".encode(),
        overwrite=overwrite and run_profile == "development",
    )
    run_record = {
        "schema": "cadence-allen-vbo-prediction-v1",
        **metadata,
        "prediction_sha256": prediction_sha256,
        "stage_records": stage_records,
        "matching_audits": matching_audits,
        "ablations": {
            "proposed_no_residual": "available"
            if "proposed" in optimization.learned_methods
            else "not_requested",
            "proposed_no_target_adaptation": "available"
            if "proposed" in optimization.learned_methods
            else "not_requested",
            "pooled": (
                "unavailable: heterogeneous observed neuron dimensions require "
                "animal-specific observation maps; condition_time is the pooled "
                "nonparametric comparator"
            ),
        },
    }
    prediction_run_path = output / "prediction_run.json"
    _atomic_write_json(
        prediction_run_path,
        run_record,
        overwrite=overwrite and run_profile == "development",
    )
    if run_profile == "locked":
        _assert_target_outcomes_sealed(
            preparation["target_seals"],
            processed_root=processed_root,
            output=output,
            targets=targets,
            canonical_relative_output=canonical_relative_output,
        )
    completion = _write_stage_completion(
        output,
        stage="predict",
        artifacts=[prediction_path, prediction_sidecar, prediction_run_path],
        metadata={
            "configuration_sha256": configuration_sha256,
            "preparation_sha256": preparation_sha256,
            "prediction_sha256": prediction_sha256,
            "target_outcomes_remained_physically_sealed": run_profile == "locked",
            "canonical_relative_output": canonical_relative_output,
            "target_seal_transaction_sha256": transaction_sha256,
        },
        overwrite=overwrite and run_profile == "development",
    )
    run_record["stage_completion"] = completion
    return run_record


def _verify_prediction_bundle(output: Path) -> tuple[Path, str]:
    """Verify prediction bytes before a scorer is allowed to open outcomes."""

    prediction = output / "predictions.npz"
    sidecar = output / "predictions.npz.sha256"
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != prediction.name:
        raise LeakageError("malformed prediction SHA256 sidecar")
    expected = fields[0]
    observed = _sha256_path(prediction)
    if observed != expected:
        raise LeakageError("prediction digest mismatch; sealed outcomes remain unopened")
    return prediction, observed


def _load_prediction_arrays(
    prediction_path: Path,
) -> tuple[dict[str, dict[str, dict[str, FloatArray]]], dict[str, Any]]:
    predictions: dict[str, dict[str, dict[str, FloatArray]]] = {}
    with np.load(prediction_path, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata"]))
        for name in arrays.files:
            if name == "metadata" or name.endswith("__source_mouse"):
                continue
            method, mouse, field = name.split("__", maxsplit=2)
            predictions.setdefault(method, {}).setdefault(mouse, {})[field] = arrays[name].astype(
                np.float64
            )
    return predictions, metadata


def _score_allen_stage(
    *,
    processed_root: str | Path,
    output: Path,
    run_profile: Literal["development", "locked"],
    fold: int | None,
    donors: list[str],
    targets: list[str],
    attestation: FreezeAttestation | None,
    optimization: AllenExperimentConfig,
    seed: int,
    overwrite: bool,
    locked_scope_audit: Mapping[str, Any] | None = None,
    canonical_relative_output: str | None = None,
) -> dict[str, Any]:
    metrics_path = output / "metrics.json"
    prepare_completion_path, _ = _stage_completion_paths(output, "prepare")
    prepare_completion = (
        _read_stage_completion(output, "prepare") if prepare_completion_path.exists() else None
    )
    predict_completion_path, _ = _stage_completion_paths(output, "predict")
    predict_completion = (
        _read_stage_completion(output, "predict") if predict_completion_path.exists() else None
    )
    if run_profile == "locked":
        _locked_append_only_gate(output, "score")
        predict_completion = _read_stage_completion(output, "predict")
    if metrics_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {metrics_path}")
    configuration_sha256 = _run_configuration_sha256(
        run_profile=run_profile,
        fold=fold,
        donors=donors,
        targets=targets,
        optimization=optimization,
        seed=seed,
    )
    if prepare_completion is not None:
        _verify_completed_artifact(
            output,
            prepare_completion,
            "preparation.json",
        )
    preparation, preparation_sha256 = _read_preparation(
        output, expected_configuration_sha256=configuration_sha256
    )
    if preparation.get("canonical_relative_output") != canonical_relative_output:
        raise LeakageError("preparation canonical output binding differs from scorer")
    if (
        prepare_completion is not None
        and dict(prepare_completion.get("metadata", {})).get("canonical_relative_output")
        != canonical_relative_output
    ):
        raise LeakageError("prepare completion canonical output binding differs from scorer")
    current_attestation = None if attestation is None else asdict(attestation)
    if preparation.get("freeze_attestation") != current_attestation:
        raise LeakageError("preparation freeze attestation differs from scorer")
    if preparation.get("locked_scope_audit") != (
        None if locked_scope_audit is None else dict(locked_scope_audit)
    ):
        raise LeakageError("preparation locked-scope audit differs from scorer")
    transaction_sha256 = None
    if run_profile == "locked":
        transaction_sha256 = _validate_allen_seal_transaction_binding(
            preparation,
            processed_root=processed_root,
            output=output,
            canonical_relative_output=str(canonical_relative_output),
            require_active_journal=True,
        )
        if (
            dict(prepare_completion.get("metadata", {})).get("target_seal_transaction_sha256")
            != transaction_sha256
        ):
            raise LeakageError("prepare completion seal-transaction binding differs")
        _assert_target_outcomes_sealed(
            preparation["target_seals"],
            processed_root=processed_root,
            output=output,
            targets=targets,
            canonical_relative_output=canonical_relative_output,
        )
    if predict_completion is not None:
        for relative_path in (
            "predictions.npz",
            "predictions.npz.sha256",
            "prediction_run.json",
        ):
            _verify_completed_artifact(output, predict_completion, relative_path)
    # Completion digests and the prediction sidecar are verified before any
    # target support or sealed payload is opened.
    prediction_path, prediction_sha256 = _verify_prediction_bundle(output)
    predictions, metadata = _load_prediction_arrays(prediction_path)
    if metadata.get("canonical_relative_output") != canonical_relative_output:
        raise LeakageError("prediction canonical output binding differs from scorer")
    if metadata.get("target_seal_transaction_sha256") != transaction_sha256:
        raise LeakageError("prediction seal-transaction binding differs from scorer")
    if (
        predict_completion is not None
        and dict(predict_completion.get("metadata", {})).get("canonical_relative_output")
        != canonical_relative_output
    ):
        raise LeakageError("predict completion canonical output binding differs from scorer")
    if (
        predict_completion is not None
        and dict(predict_completion.get("metadata", {})).get("target_seal_transaction_sha256")
        != transaction_sha256
    ):
        raise LeakageError("predict completion seal-transaction binding differs from scorer")
    if metadata.get("configuration_sha256") != configuration_sha256:
        raise LeakageError("prediction bundle configuration differs from scorer")
    if metadata.get("preparation_sha256") != preparation_sha256:
        raise LeakageError("prediction bundle references a different preparation")
    run_record = json.loads((output / "prediction_run.json").read_text(encoding="utf-8"))
    supports = {
        mouse: load_animal_support(processed_root, mouse, require_physical_split=True)
        for mouse in targets
    }
    scores: dict[str, dict[str, dict[str, Any]]] = {
        method: {} for method in metadata["report_methods"]
    }
    sealed_hashes: dict[str, str] = {}
    for mouse in targets:
        sealed_path = output / "queries" / f"mouse_{mouse}" / "sealed_outcomes.npz"
        if run_profile == "locked":
            sealed, observed_hash = _open_experiment_sealed_for_score(
                sealed_path,
                preparation["target_seals"][mouse]["experiment_sealed"],
            )
        else:
            observed_hash = _sha256_path(sealed_path)
            with np.load(sealed_path, allow_pickle=False) as arrays:
                sealed = {name: arrays[name] for name in arrays.files}
        expected_hash = preparation["experiment_artifacts"][mouse]["sealed_outcomes.npz"]
        if observed_hash != expected_hash:
            raise LeakageError(f"sealed outcome digest changed for mouse {mouse}")
        sealed_hashes[mouse] = observed_hash
        for method, animal_predictions in predictions.items():
            if mouse in animal_predictions:
                scores[method][mouse] = _method_score(
                    animal_predictions[mouse], sealed, supports[mouse]
                )
    restoration_audit = (
        _target_restoration_plan(
            preparation["target_seals"],
            targets=targets,
            canonical_relative_output=str(canonical_relative_output),
            seal_transaction_sha256=preparation["target_seal_transaction"]["sha256"],
        )
        if run_profile == "locked"
        else None
    )
    scalar_names = (
        "neural_absolute_nrmse",
        "running_absolute_nrmse",
        "pupil_absolute_nrmse",
        "lick_absolute_nrmse",
        "neural_causal_skill",
        "running_causal_skill",
        "pupil_causal_skill",
        "lick_causal_skill",
        "neural_trial_causal_skill",
        "running_trial_causal_skill",
        "pupil_trial_causal_skill",
        "lick_trial_causal_skill",
    )
    aggregate = {
        method: {
            name: float(np.nanmean([values[name] for values in animals.values()]))
            for name in scalar_names
        }
        for method, animals in scores.items()
    }
    protocol_audit = {
        "development_mice": list(DEVELOPMENT_MICE),
        "development_hash_rule": DEVELOPMENT_HASH_RULE,
        "locked_acknowledged": run_profile == "locked",
        "preoutcome_freeze_attestation": (None if attestation is None else asdict(attestation)),
        "outer_split_unit": "mouse_id",
        "inner_validation_unit": "mouse_id",
        "target_intervention_outcomes_used_for_optimization": 0,
        "target_normal_audit_used_for_optimization": False,
        "target_adapter_partitions": ["normal_fit", "normal_val"],
        "query_initialization_sample": "onset_minus_1",
        "post_onset_outcomes_in_prediction_input": False,
        "prediction_sha256_before_score": prediction_sha256,
        "predict_completion_sha256": (
            None if predict_completion is None else predict_completion["completion_sha256"]
        ),
        "canonical_optimization_sha256": _canonical_optimization_sha256(optimization),
        "scoring_runtime_optimization_sha256": _runtime_optimization_sha256(optimization),
        "locked_scope_audit": (None if locked_scope_audit is None else dict(locked_scope_audit)),
        "prediction_hashed_before_sealed_outcomes_opened_for_scoring": True,
        "sealed_outcome_open_order": (
            "separate predict process -> SHA256 sidecar -> scorer verifies bytes "
            "-> scorer hashes/opens sealed outcomes"
        ),
        "sealed_outcome_sha256": sealed_hashes,
        "target_outcome_mode_restoration": restoration_audit,
        "control_match_fallback_hierarchy": [
            "exact_image_and_flashes",
            "image_and_risk_bin",
            "image_only",
            "risk_bin_only",
            "complete_pool",
        ],
        "primary_effect_score": (
            "condition-and-fallback-stratum expected effects, equal stratum weight"
        ),
        "secondary_effect_score": "trial-level pooled causal skill",
        "missingness_encoding": (
            "explicit neural-valid and behavior-valid encoder channels; "
            "auxiliary reconstruction weight zero"
        ),
        "donor_intervention_random_effects": {
            "selection_train_donors_only": True,
            "selection_validation_delta_present": False,
            "selection_centering_excludes_validation": True,
            "selection_f_fit_excludes_validation_mouse": True,
            "selection_validation_normal_adapter_with_f_frozen": True,
            "refit_fresh_model_on_all_donor_normals": True,
            "refit_all_donors_from_fresh_normal_refit": True,
            "exact_zero_mean_projection_after_each_step": True,
            "zero_mean_tolerance": DELTA_PROJECTION_TOLERANCE,
            "target_prediction_delta": "integrated_at_zero_mean",
            "l2_shrinkage": 0.01,
            "mean_centering_penalty": 0.01,
        },
        "matching_audits": run_record["matching_audits"],
        "query_audits": preparation["query_audits"],
        "ablations": run_record["ablations"],
    }
    payload = {
        "schema": "cadence-allen-vbo-experiment-v2",
        "run_profile": run_profile,
        "canonical_relative_output": canonical_relative_output,
        "target_seal_transaction_sha256": transaction_sha256,
        "optimization_profile": optimization.profile,
        "fold": fold,
        "seed": seed,
        "donors": donors,
        "targets": targets,
        "stage_records": run_record["stage_records"],
        "protocol_audit": protocol_audit,
        "aggregate": aggregate,
        "animals": {
            method: {mouse: _jsonable(values) for mouse, values in animals.items()}
            for method, animals in scores.items()
        },
    }
    _atomic_write_json(
        metrics_path,
        payload,
        overwrite=overwrite and run_profile == "development",
    )
    reference = supports[targets[0]]
    _write_long_metrics(
        scores,
        reference.relative_time_s[reference.onset :],
        output / "metrics_long.csv",
        overwrite=overwrite and run_profile == "development",
    )
    completion = _write_stage_completion(
        output,
        stage="score",
        artifacts=[metrics_path, output / "metrics_long.csv"],
        metadata={
            "configuration_sha256": configuration_sha256,
            "prediction_sha256": prediction_sha256,
            "sealed_outcome_sha256": sealed_hashes,
            "canonical_processed_target_modes_restored": False,
            "target_mode_restoration_pending": run_profile == "locked",
            "canonical_relative_output": canonical_relative_output,
            "target_seal_transaction_sha256": transaction_sha256,
        },
        overwrite=overwrite and run_profile == "development",
    )
    restoration_completion = (
        _restore_target_outcomes_after_score(
            preparation["target_seals"],
            processed_root=processed_root,
            output=output,
            targets=targets,
            preparation=preparation,
            canonical_relative_output=str(canonical_relative_output),
        )
        if run_profile == "locked"
        else None
    )
    payload["stage_completion"] = completion
    payload["restoration_completion"] = restoration_completion
    return payload


def run_allen_experiment(
    *,
    processed_root: str | Path,
    manifest_path: str | Path,
    output_directory: str | Path,
    run_profile: Literal["development", "locked"],
    optimization: AllenExperimentConfig,
    fold: int | None = None,
    acknowledge_locked: bool = False,
    development_target: str = "423606",
    development_donors: Sequence[str] = ("539517", "448900"),
    seed: int = 0,
    overwrite: bool = False,
    stage: Literal["prepare", "predict", "score", "all"] = "all",
) -> dict[str, Any]:
    """Execute one role-separated experiment stage.

    Locked runs categorically reject ``stage='all'`` so preparation, prediction,
    and scoring occur in distinct processes. Development retains the convenience
    path for tests and method-development records.
    """

    optimization.validate()
    output = Path(output_directory)
    locked_scope_audit: dict[str, Any] | None = None
    canonical_relative_output: str | None = None
    if run_profile == "locked":
        if stage == "all":
            raise LeakageError(
                "locked runs require separate --stage prepare, predict, and score processes"
            )
        if overwrite:
            raise ProtocolViolation("locked Allen stages are append-only; overwrite is forbidden")
        if not acknowledge_locked:
            raise LeakageError(
                "locked outcomes remain sealed; pass --acknowledge-locked only after the "
                "protocol/code commit has been frozen"
            )
        repository = _repository_root()
        if Path(manifest_path).resolve() != (repository / CANONICAL_MANIFEST_RELATIVE).resolve():
            raise ProtocolViolation("locked Allen runs require the canonical source manifest")
        if (
            Path(processed_root).resolve()
            != (repository / CANONICAL_PROCESSED_ROOT_RELATIVE).resolve()
        ):
            raise ProtocolViolation("locked Allen runs require the canonical processed root")
        _validate_locked_configuration(optimization, seed=seed)
        if stage not in {"prepare", "predict", "score"}:
            raise ValueError(f"unknown Allen experiment stage: {stage}")
    donors, targets, attestation = _resolve_staged_identity(
        manifest_path=manifest_path,
        run_profile=run_profile,
        fold=fold,
        acknowledge_locked=acknowledge_locked,
        development_target=development_target,
        development_donors=development_donors,
    )
    if run_profile == "locked":
        if attestation is None:
            raise AssertionError("locked run is missing freeze attestation")
        if fold is None:
            raise AssertionError("locked run is missing fold")
        canonical_relative_output = _require_canonical_locked_output(output, fold)
        locked_scope_audit = _verify_locked_protocol_scope(
            processed_root=processed_root,
            manifest_path=manifest_path,
            optimization=optimization,
            seed=seed,
            attestation=attestation,
            repository=_repository_root(),
        )
        recovery = _recover_allen_locked_stage(
            processed_root=processed_root,
            output=output,
            fold=fold,
            stage=stage,
            canonical_relative_output=canonical_relative_output,
        )
        if recovery is not None:
            artifact_name = {
                "prepare": "preparation.json",
                "predict": "prediction_run.json",
                "score": "metrics.json",
            }[stage]
            artifact = output / artifact_name
            if artifact.is_file():
                payload = json.loads(artifact.read_text(encoding="utf-8"))
                payload["stage_completion"] = _read_stage_completion(output, stage)
                if stage == "score":
                    payload["restoration_completion"] = json.loads(
                        (output / "restore.complete.json").read_text(encoding="utf-8")
                    )
                return payload
        _locked_append_only_gate(output, stage)
    common = {
        "processed_root": processed_root,
        "output": output,
        "run_profile": run_profile,
        "fold": fold,
        "donors": donors,
        "targets": targets,
        "attestation": attestation,
        "optimization": optimization,
        "seed": seed,
        "overwrite": overwrite,
        "locked_scope_audit": locked_scope_audit,
        "canonical_relative_output": canonical_relative_output,
    }
    if stage == "prepare":
        return _prepare_allen_stage(**common)
    if stage == "predict":
        return _predict_allen_stage(**common)
    if stage == "score":
        return _score_allen_stage(**common)
    if stage != "all":
        raise ValueError(f"unknown Allen experiment stage: {stage}")
    _prepare_allen_stage(**common)
    _predict_allen_stage(**common)
    return _score_allen_stage(**common)
