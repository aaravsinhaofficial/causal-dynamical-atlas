"""Leakage-safe Allen Visual Behavior Ophys cohort and NWB preparation.

The module uses only public, anonymous HTTPS endpoints.  Release manifests pin
every NWB to an S3 version and Allen-provided BLAKE2b-512 digest.  Data are
selected one experiment per mouse before any split is constructed.

Run ``python -m cadence.data.allen_vbo --help`` for the standalone interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from cadence.data.splits import assert_calibration_is_normal

ALLEN_BUCKET = "visual-behavior-ophys-data"
ALLEN_PREFIX = "visual-behavior-ophys"
ALLEN_RELEASE = "1.1.0"
ALLEN_PROJECT_MANIFEST_URL = (
    "https://visual-behavior-ophys-data.s3.us-west-2.amazonaws.com/"
    "visual-behavior-ophys/manifests/"
    "visual-behavior-ophys_project_manifest_v1.1.0.json"
)
MANIFEST_SCHEMA_VERSION = "1.0"

_STIMULUS_COLUMNS = (
    "start_time",
    "stop_time",
    "stimulus_block",
    "stimulus_block_name",
    "image_index",
    "image_name",
    "duration",
    "is_change",
    "omitted",
    "is_sham_change",
    "flashes_since_change",
    "trials_id",
    "active",
)
_FLAG_COLUMNS = ("active", "omitted", "is_change", "is_sham_change")


@dataclass(frozen=True)
class CohortSpec:
    """Biological and quality filters applied before mouse-level selection."""

    targeted_structure: str = "VISp"
    cre_line: str = "Slc17a7-IRES2-Cre"
    imaging_depth_um: int = 175
    experience_level: str = "Familiar"
    passive: bool = False
    preferred_session_number: int = 1
    minimum_cells: int = 40


@dataclass(frozen=True)
class WindowPolicy:
    """Definition of intervention and strictly normal calibration windows."""

    rate_hz: float = 10.0
    window_start_s: float = -1.0
    window_end_s: float = 2.0
    normal_contamination_guard_s: float = 3.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.rate_hz) or self.rate_hz <= 0:
            raise ValueError("rate_hz must be positive and finite")
        if not self.window_start_s < self.window_end_s:
            raise ValueError("window_start_s must be less than window_end_s")
        if self.normal_contamination_guard_s < 0:
            raise ValueError("normal_contamination_guard_s may not be negative")
        duration_samples = (self.window_end_s - self.window_start_s) * self.rate_hz
        if not np.isclose(duration_samples, round(duration_samples), atol=1e-8):
            raise ValueError("window duration must contain an integer number of samples")

    @property
    def relative_time(self) -> np.ndarray:
        sample_count = int(round((self.window_end_s - self.window_start_s) * self.rate_hz))
        return self.window_start_s + np.arange(sample_count + 1) / self.rate_hz


DEFAULT_COHORT_SPEC = CohortSpec()
DEFAULT_WINDOW_POLICY = WindowPolicy()


@dataclass(frozen=True)
class WindowSelection:
    """Accepted omission and calibration presentations plus rejection audit."""

    omissions: pd.DataFrame
    normal: pd.DataFrame
    audit: dict[str, int]

    @property
    def table(self) -> pd.DataFrame:
        return pd.concat([self.omissions, self.normal], ignore_index=True)


@dataclass(frozen=True)
class AnimalOutputPaths:
    """Files emitted for one mouse."""

    directory: Path
    arrays: Path
    normal_support: Path
    omission_query: Path
    sealed_omission_outcomes: Path
    stimulus_presentations: Path
    windows: Path
    provenance: Path


@dataclass
class _NWBSignals:
    experiment_id: str
    mouse_id: str
    neural_data: Any
    neural_timestamps: np.ndarray
    cell_roi_ids: np.ndarray
    cell_specimen_ids: np.ndarray
    running_speed: np.ndarray
    running_timestamps: np.ndarray
    pupil_area: np.ndarray
    pupil_timestamps: np.ndarray
    likely_blink: np.ndarray
    lick_timestamps: np.ndarray
    presentations: pd.DataFrame


def _canonical_identifier(value: object) -> str:
    if pd.isna(value):
        raise ValueError("identifier may not be missing")
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating) and float(value).is_integer():
        return str(int(value))
    result = str(value).strip()
    if not result:
        raise ValueError("identifier may not be empty")
    return result


def _as_bool(value: object, *, missing: bool = False) -> bool:
    if value is None or pd.isna(value):
        return missing
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | float | np.integer | np.floating):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", ""}:
        return False
    raise ValueError(f"cannot interpret {value!r} as a boolean")


def _boolean_series(values: pd.Series, *, missing: bool = False) -> pd.Series:
    return values.map(lambda value: _as_bool(value, missing=missing)).astype(bool)


def _read_frame(frame_or_path: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(frame_or_path, pd.DataFrame):
        return frame_or_path.copy()
    return pd.read_csv(frame_or_path)


def select_one_experiment_per_mouse(
    experiments: pd.DataFrame | str | Path,
    cells: pd.DataFrame | str | Path,
    *,
    spec: CohortSpec = DEFAULT_COHORT_SPEC,
) -> pd.DataFrame:
    """Apply the frozen cohort filter and deterministically select one NWB per mouse.

    Preferred session number wins, then the earliest acquisition timestamp, then
    the smallest ophys experiment ID.  The final tie-breaker makes selection
    invariant to metadata input order.
    """

    experiment_frame = _read_frame(experiments)
    cell_frame = _read_frame(cells)
    required_experiment_columns = {
        "ophys_experiment_id",
        "ophys_session_id",
        "mouse_id",
        "cre_line",
        "targeted_structure",
        "imaging_depth",
        "experience_level",
        "passive",
        "session_number",
        "date_of_acquisition",
        "file_id",
    }
    missing = sorted(required_experiment_columns - set(experiment_frame))
    if missing:
        raise KeyError(f"experiment metadata is missing columns: {missing}")
    if "ophys_experiment_id" not in cell_frame:
        raise KeyError("cell metadata is missing ophys_experiment_id")

    experiment_frame["mouse_id"] = experiment_frame["mouse_id"].map(_canonical_identifier)
    experiment_frame["ophys_experiment_id"] = pd.to_numeric(
        experiment_frame["ophys_experiment_id"], errors="raise"
    ).astype("int64")
    experiment_frame["ophys_session_id"] = pd.to_numeric(
        experiment_frame["ophys_session_id"], errors="raise"
    ).astype("int64")
    experiment_frame["file_id"] = pd.to_numeric(experiment_frame["file_id"], errors="raise").astype(
        "int64"
    )

    cell_frame["ophys_experiment_id"] = pd.to_numeric(
        cell_frame["ophys_experiment_id"], errors="coerce"
    )
    cell_frame = cell_frame.dropna(subset=["ophys_experiment_id"]).copy()
    cell_frame["ophys_experiment_id"] = cell_frame["ophys_experiment_id"].astype("int64")
    if "cell_roi_id" in cell_frame:
        cell_counts = cell_frame.groupby("ophys_experiment_id")["cell_roi_id"].nunique()
    else:
        cell_counts = cell_frame.groupby("ophys_experiment_id").size()
    experiment_frame["num_cells"] = (
        experiment_frame["ophys_experiment_id"].map(cell_counts).fillna(0).astype("int64")
    )

    depth = pd.to_numeric(experiment_frame["imaging_depth"], errors="coerce")
    passive = _boolean_series(experiment_frame["passive"], missing=True)
    keep = (
        experiment_frame["targeted_structure"].astype(str).eq(spec.targeted_structure)
        & experiment_frame["cre_line"].astype(str).eq(spec.cre_line)
        & depth.eq(spec.imaging_depth_um)
        & experiment_frame["experience_level"].astype(str).eq(spec.experience_level)
        & passive.eq(spec.passive)
        & experiment_frame["num_cells"].ge(spec.minimum_cells)
    )
    candidates = experiment_frame.loc[keep].copy()
    if candidates.empty:
        raise ValueError("the cohort filter selected no experiments")

    candidates["_session_number"] = pd.to_numeric(candidates["session_number"], errors="coerce")
    candidates["_preferred_session"] = ~candidates["_session_number"].eq(
        spec.preferred_session_number
    )
    candidates["_acquisition_time"] = pd.to_datetime(
        candidates["date_of_acquisition"], errors="coerce", utc=True
    )
    # Missing acquisition dates sort last; experiment ID remains a complete tie-breaker.
    candidates = candidates.sort_values(
        [
            "mouse_id",
            "_preferred_session",
            "_acquisition_time",
            "ophys_experiment_id",
        ],
        kind="stable",
        na_position="last",
    )
    cohort = candidates.drop_duplicates("mouse_id", keep="first").copy()
    cohort["selection_preferred_session"] = cohort["_session_number"].eq(
        spec.preferred_session_number
    )
    cohort["cohort_rank_rule"] = "preferred_session_then_earliest_acquisition_then_experiment_id"
    cohort = cohort.drop(columns=["_session_number", "_preferred_session", "_acquisition_time"])
    cohort = cohort.sort_values("mouse_id", kind="stable").reset_index(drop=True)

    if cohort["mouse_id"].duplicated().any():
        raise AssertionError("one-experiment-per-mouse invariant failed")
    if cohort["ophys_experiment_id"].duplicated().any():
        raise AssertionError("an ophys experiment was assigned to multiple mice")
    return cohort


def _read_url_bytes(source: str | Path, *, timeout_s: float = 60.0) -> bytes:
    path = Path(source)
    if path.exists():
        return path.read_bytes()
    request = urllib.request.Request(str(source), headers={"User-Agent": "cadence-neuro/0.1"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read()


def load_project_manifest(
    source: str | Path = ALLEN_PROJECT_MANIFEST_URL,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Read a local or public release manifest and fingerprint the exact bytes."""

    payload = _read_url_bytes(source)
    manifest = json.loads(payload)
    if manifest.get("project_name") != "visual-behavior-ophys":
        raise ValueError("unexpected Allen project manifest")
    source_string = str(source)
    provenance = {
        "source": source_string,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    return manifest, provenance


def _head_public_object(url: str, *, timeout_s: float = 30.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "cadence-neuro/0.1"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        headers = response.headers
        etag = headers.get("ETag", "").strip('"') or None
        size_header = headers.get("Content-Length")
        return {
            "size_bytes": int(size_header) if size_header is not None else None,
            "etag": etag,
            "etag_is_content_md5": bool(etag and re.fullmatch(r"[0-9a-fA-F]{32}", etag)),
            "last_modified": headers.get("Last-Modified"),
        }


def _public_url_to_s3_uri(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    key = parsed.path.lstrip("/")
    return f"s3://{ALLEN_BUCKET}/{key}"


def _enrich_remote_entries(
    entries: list[dict[str, Any]],
    *,
    workers: int = 8,
) -> list[dict[str, Any]]:
    # Query the exact immutable S3 version, not whichever object is current
    # when the cohort manifest happens to be regenerated.
    urls = [_versioned_url(entry) for entry in entries]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        heads = list(executor.map(_head_public_object, urls))
    return [{**entry, **head} for entry, head in zip(entries, heads, strict=True)]


def build_public_download_manifest(
    cohort: pd.DataFrame,
    project_manifest: Mapping[str, Any],
    *,
    cohort_spec: CohortSpec = DEFAULT_COHORT_SPEC,
    release_manifest_provenance: Mapping[str, str] | None = None,
    fetch_object_metadata: bool = True,
    head_workers: int = 8,
) -> dict[str, Any]:
    """Build a version-pinned, checksummed public S3 manifest for the cohort."""

    required = {"mouse_id", "ophys_experiment_id", "ophys_session_id", "file_id", "num_cells"}
    missing = sorted(required - set(cohort))
    if missing:
        raise KeyError(f"cohort is missing columns: {missing}")
    if cohort["mouse_id"].map(_canonical_identifier).duplicated().any():
        raise ValueError("download cohort contains repeated mice")

    data_files = project_manifest.get("data_files")
    if not isinstance(data_files, Mapping):
        raise ValueError("project manifest has no data_files mapping")

    entries: list[dict[str, Any]] = []
    for row in cohort.sort_values("mouse_id", kind="stable").to_dict(orient="records"):
        file_id = _canonical_identifier(row["file_id"])
        if file_id not in data_files:
            raise KeyError(f"file_id {file_id} is absent from the release manifest")
        released = data_files[file_id]
        url = str(released["url"])
        experiment_id = int(row["ophys_experiment_id"])
        expected_name = f"behavior_ophys_experiment_{experiment_id}.nwb"
        if not urllib.parse.urlsplit(url).path.endswith(expected_name):
            raise ValueError(
                f"file_id {file_id} URL does not match experiment {experiment_id}: {url}"
            )
        entries.append(
            {
                "mouse_id": _canonical_identifier(row["mouse_id"]),
                "ophys_experiment_id": experiment_id,
                "ophys_session_id": int(row["ophys_session_id"]),
                "file_id": int(row["file_id"]),
                "num_cells": int(row["num_cells"]),
                "session_number": _json_scalar(row.get("session_number")),
                "date_of_acquisition": str(row.get("date_of_acquisition", "")),
                "url": url,
                "s3_uri": _public_url_to_s3_uri(url),
                "s3_version_id": str(released["version_id"]),
                "blake2b_512": str(released["file_hash"]).lower(),
                "local_filename": expected_name,
            }
        )
    if fetch_object_metadata:
        entries = _enrich_remote_entries(entries, workers=head_workers)

    metadata_entries: list[dict[str, Any]] = []
    for name, released in sorted(project_manifest.get("metadata_files", {}).items()):
        url = str(released["url"])
        metadata_entries.append(
            {
                "name": str(name),
                "url": url,
                "s3_uri": _public_url_to_s3_uri(url),
                "s3_version_id": str(released["version_id"]),
                "blake2b_512": str(released["file_hash"]).lower(),
                "local_filename": Path(urllib.parse.urlsplit(url).path).name,
            }
        )
    if fetch_object_metadata and metadata_entries:
        metadata_entries = _enrich_remote_entries(metadata_entries, workers=head_workers)

    result = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": {
            "name": "Allen Visual Behavior Ophys",
            "project_name": project_manifest.get("project_name"),
            "release": project_manifest.get("manifest_version"),
            "public_bucket": f"s3://{ALLEN_BUCKET}/{ALLEN_PREFIX}",
            "authentication": "anonymous_public_read",
            "license": "CC BY 4.0",
        },
        "cohort_spec": asdict(cohort_spec),
        "selection": {
            "unit": "mouse_id",
            "one_experiment_per_mouse": True,
            "tie_breaker": ("preferred_session_then_earliest_acquisition_then_experiment_id"),
            "num_animals": len(entries),
        },
        "checksum": {
            "content_hash": "blake2b-512",
            "source": "Allen release manifest",
            "etag_note": (
                "S3 multipart ETags are recorded for provenance but are not content MD5s"
            ),
        },
        "release_manifest": dict(release_manifest_provenance or {}),
        "metadata_files": metadata_entries,
        "nwb_files": entries,
    }
    return result


def write_json(payload: Mapping[str, Any], destination: str | Path) -> Path:
    """Atomically write deterministic, human-readable JSON."""

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        json.dump(_jsonify(payload), temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, output)
    return output


def hash_file(path: str | Path, algorithm: str = "blake2b") -> str:
    """Stream a file through a cryptographic digest without loading it into RAM."""

    hasher = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _versioned_url(entry: Mapping[str, Any]) -> str:
    url = str(entry["url"])
    version_id = entry.get("s3_version_id")
    if not version_id:
        return url
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query.append(("versionId", str(version_id)))
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(query), parsed.fragment)
    )


def verify_download(path: str | Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Validate byte length and Allen's BLAKE2b-512 content hash."""

    local = Path(path)
    actual_size = local.stat().st_size
    expected_size = entry.get("size_bytes")
    if expected_size is not None and actual_size != int(expected_size):
        raise ValueError(
            f"size mismatch for {local}: expected {expected_size}, observed {actual_size}"
        )
    actual_hash = hash_file(local, "blake2b")
    expected_hash = str(entry["blake2b_512"]).lower()
    if actual_hash != expected_hash:
        raise ValueError(
            f"BLAKE2b-512 mismatch for {local}: expected {expected_hash}, observed {actual_hash}"
        )
    return {
        "path": str(local),
        "size_bytes": actual_size,
        "blake2b_512": actual_hash,
        "verified": True,
    }


def download_manifest_nwbs(
    manifest: Mapping[str, Any] | str | Path,
    destination: str | Path,
    *,
    experiment_ids: Sequence[int] | None = None,
    overwrite: bool = False,
    verify: bool = True,
) -> pd.DataFrame:
    """Atomically download selected public NWBs; never downloads implicitly."""

    if not isinstance(manifest, Mapping):
        manifest = json.loads(Path(manifest).read_text())
    requested = None if experiment_ids is None else {int(value) for value in experiment_ids}
    entries = [
        entry
        for entry in manifest["nwb_files"]
        if requested is None or int(entry["ophys_experiment_id"]) in requested
    ]
    if requested is not None:
        found = {int(entry["ophys_experiment_id"]) for entry in entries}
        missing = sorted(requested - found)
        if missing:
            raise KeyError(f"requested experiments are not in the cohort manifest: {missing}")

    output_directory = Path(destination)
    output_directory.mkdir(parents=True, exist_ok=True)
    results = []
    for entry in entries:
        output = output_directory / str(entry["local_filename"])
        if output.exists() and not overwrite:
            result = (
                verify_download(output, entry)
                if verify
                else {
                    "path": str(output),
                    "size_bytes": output.stat().st_size,
                    "verified": False,
                }
            )
            results.append({**result, "status": "existing"})
            continue

        partial = output_directory / f".{output.name}.partial"
        if partial.exists():
            partial.unlink()
        request = urllib.request.Request(
            _versioned_url(entry),
            headers={"User-Agent": "cadence-neuro/0.1"},
        )
        try:
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                partial.open("wb") as stream,
            ):
                while chunk := response.read(8 * 1024 * 1024):
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            result = (
                verify_download(partial, entry)
                if verify
                else {
                    "path": str(partial),
                    "size_bytes": partial.stat().st_size,
                    "verified": False,
                }
            )
            os.replace(partial, output)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        result["path"] = str(output)
        results.append({**result, "status": "downloaded"})
    return pd.DataFrame(results)


def _decode_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind == "S":
        return np.char.decode(array, "utf-8")
    if array.dtype.kind == "O":
        return np.asarray(
            [
                value.decode("utf-8") if isinstance(value, bytes | np.bytes_) else value
                for value in array
            ]
        )
    return array


def _h5_scalar(dataset: h5py.Dataset) -> object:
    value = dataset[()]
    if isinstance(value, bytes | np.bytes_):
        return value.decode("utf-8")
    return value


def _standardize_presentations(frame: pd.DataFrame) -> pd.DataFrame:
    presentations = frame.copy()
    if "stimulus_presentation_id" not in presentations:
        presentations.insert(0, "stimulus_presentation_id", np.arange(len(presentations)))
    presentations["stimulus_presentation_id"] = pd.to_numeric(
        presentations["stimulus_presentation_id"], errors="raise"
    ).astype("int64")
    presentations["start_time"] = pd.to_numeric(presentations["start_time"], errors="coerce")
    if "stop_time" in presentations:
        presentations["stop_time"] = pd.to_numeric(presentations["stop_time"], errors="coerce")
    for column in _FLAG_COLUMNS:
        if column not in presentations:
            presentations[column] = column == "active"
        presentations[column] = _boolean_series(presentations[column], missing=(column == "active"))
    for column in presentations.select_dtypes(include=["object"]).columns:
        presentations[column] = presentations[column].map(
            lambda value: value.decode("utf-8") if isinstance(value, bytes | np.bytes_) else value
        )
    return presentations.sort_values(
        ["start_time", "stimulus_presentation_id"], kind="stable"
    ).reset_index(drop=True)


def _presentations_from_h5(file: h5py.File) -> pd.DataFrame:
    tables = [
        group
        for group in file["intervals"].values()
        if isinstance(group, h5py.Group)
        and "start_time" in group
        and "omitted" in group
        and "is_change" in group
    ]
    if not tables:
        raise KeyError("could not find an image-presentation interval table")
    table = max(tables, key=lambda group: len(group["start_time"]))
    columns: dict[str, np.ndarray] = {
        "stimulus_presentation_id": np.asarray(table["id"][:], dtype=np.int64)
        if "id" in table
        else np.arange(len(table["start_time"]), dtype=np.int64)
    }
    for column in _STIMULUS_COLUMNS:
        if column in table:
            columns[column] = _decode_array(table[column][:])
    return _standardize_presentations(pd.DataFrame(columns))


def _presentations_from_pynwb(nwbfile: Any) -> pd.DataFrame:
    tables = [
        table
        for table in nwbfile.intervals.values()
        if "start_time" in table.colnames
        and "omitted" in table.colnames
        and "is_change" in table.colnames
    ]
    if not tables:
        raise KeyError("could not find an image-presentation interval table")
    table = max(tables, key=len)
    columns: dict[str, np.ndarray] = {
        "stimulus_presentation_id": np.asarray(table.id[:], dtype=np.int64)
    }
    for column in _STIMULUS_COLUMNS:
        if column in table.colnames:
            columns[column] = _decode_array(np.asarray(table[column].data[:]))
    return _standardize_presentations(pd.DataFrame(columns))


def _cell_ids_h5(file: h5py.File, event_group: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    roi_indices = np.asarray(event_group["rois"][:], dtype=np.int64)
    table = file["processing/ophys/image_segmentation/cell_specimen_table"]
    roi_ids = np.asarray(table["id"][:], dtype=np.int64)[roi_indices]
    if "cell_specimen_id" in table:
        specimen_ids = np.asarray(table["cell_specimen_id"][:], dtype=np.int64)[roi_indices]
    else:
        specimen_ids = np.full(len(roi_ids), -1, dtype=np.int64)
    return roi_ids, specimen_ids


@contextmanager
def _open_h5_source(path: Path) -> Iterator[_NWBSignals]:
    with h5py.File(path, "r") as file:
        event = file["processing/ophys/event_detection"]
        running = file["processing/running/speed"]
        pupil = file["acquisition/EyeTracking/pupil_tracking"]
        pupil_timestamps = np.asarray(pupil["timestamps"][:], dtype=np.float64)
        blink_path = "acquisition/EyeTracking/likely_blink/data"
        blink = (
            np.asarray(file[blink_path][:], dtype=bool)
            if blink_path in file
            else np.zeros(len(pupil_timestamps), dtype=bool)
        )
        lick_path = "processing/licking/licks/timestamps"
        licks = (
            np.asarray(file[lick_path][:], dtype=np.float64)
            if lick_path in file
            else np.empty(0, dtype=np.float64)
        )
        roi_ids, specimen_ids = _cell_ids_h5(file, event)
        source = _NWBSignals(
            experiment_id=_canonical_identifier(_h5_scalar(file["identifier"])),
            mouse_id=_canonical_identifier(_h5_scalar(file["general/subject/subject_id"])),
            neural_data=event["data"],
            neural_timestamps=np.asarray(event["timestamps"][:], dtype=np.float64),
            cell_roi_ids=roi_ids,
            cell_specimen_ids=specimen_ids,
            running_speed=np.asarray(running["data"][:], dtype=np.float64),
            running_timestamps=np.asarray(running["timestamps"][:], dtype=np.float64),
            pupil_area=np.asarray(pupil["area"][:], dtype=np.float64),
            pupil_timestamps=pupil_timestamps,
            likely_blink=blink,
            lick_timestamps=licks,
            presentations=_presentations_from_h5(file),
        )
        yield source


@contextmanager
def _open_pynwb_source(path: Path) -> Iterator[_NWBSignals]:
    with NWBHDF5IO(path, mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        event = nwbfile.processing["ophys"].data_interfaces["event_detection"]
        running = nwbfile.processing["running"].data_interfaces["speed"]
        eye = nwbfile.acquisition["EyeTracking"]
        pupil = eye.pupil_tracking
        blink = np.asarray(eye.likely_blink.data[:], dtype=bool)
        licking = nwbfile.processing["licking"].data_interfaces.get("licks")
        licks = (
            np.asarray(licking.timestamps[:], dtype=np.float64)
            if licking is not None
            else np.empty(0, dtype=np.float64)
        )
        roi_indices = np.asarray(event.rois.data[:], dtype=np.int64)
        roi_table = event.rois.table
        roi_ids = np.asarray(roi_table.id[:], dtype=np.int64)[roi_indices]
        specimen_ids = (
            np.asarray(roi_table["cell_specimen_id"].data[:], dtype=np.int64)[roi_indices]
            if "cell_specimen_id" in roi_table.colnames
            else np.full(len(roi_ids), -1, dtype=np.int64)
        )
        source = _NWBSignals(
            experiment_id=_canonical_identifier(nwbfile.identifier),
            mouse_id=_canonical_identifier(nwbfile.subject.subject_id),
            neural_data=event.data,
            neural_timestamps=np.asarray(event.timestamps[:], dtype=np.float64),
            cell_roi_ids=roi_ids,
            cell_specimen_ids=specimen_ids,
            running_speed=np.asarray(running.data[:], dtype=np.float64),
            running_timestamps=np.asarray(running.timestamps[:], dtype=np.float64),
            pupil_area=np.asarray(pupil.area[:], dtype=np.float64),
            pupil_timestamps=np.asarray(pupil.timestamps[:], dtype=np.float64),
            likely_blink=blink,
            lick_timestamps=licks,
            presentations=_presentations_from_pynwb(nwbfile),
        )
        yield source


@contextmanager
def open_nwb_signals(
    path: str | Path,
    *,
    backend: Literal["pynwb", "h5py"] = "pynwb",
) -> Iterator[_NWBSignals]:
    """Open only the series needed for CADENCE, keeping neural traces lazy."""

    nwb_path = Path(path)
    if backend == "pynwb":
        with _open_pynwb_source(nwb_path) as source:
            yield _validate_source(source)
    elif backend == "h5py":
        with _open_h5_source(nwb_path) as source:
            yield _validate_source(source)
    else:
        raise ValueError(f"unknown NWB backend: {backend}")


def _validate_timestamps(name: str, timestamps: np.ndarray, sample_count: int) -> None:
    if timestamps.ndim != 1 or len(timestamps) != sample_count:
        raise ValueError(f"{name} timestamps do not match data")
    if len(timestamps) < 2 or not np.isfinite(timestamps).all():
        raise ValueError(f"{name} timestamps are missing or non-finite")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{name} timestamps are not strictly increasing")


def _validate_source(source: _NWBSignals) -> _NWBSignals:
    if len(source.neural_data.shape) != 2:
        raise ValueError("neural event data must be time by cell")
    _validate_timestamps("neural", source.neural_timestamps, source.neural_data.shape[0])
    _validate_timestamps("running", source.running_timestamps, len(source.running_speed))
    _validate_timestamps("pupil", source.pupil_timestamps, len(source.pupil_area))
    if len(source.likely_blink) != len(source.pupil_area):
        raise ValueError("blink mask does not match pupil samples")
    if source.neural_data.shape[1] != len(source.cell_roi_ids):
        raise ValueError("neural columns do not match ROI identifiers")
    if len(source.cell_specimen_ids) != len(source.cell_roi_ids):
        raise ValueError("cell specimen and ROI identifier lengths differ")
    return source


def _has_event_in_interval(
    event_times: np.ndarray,
    start: float,
    stop: float,
    *,
    exclude_time: float | None = None,
) -> bool:
    left = int(np.searchsorted(event_times, start, side="left"))
    right = int(np.searchsorted(event_times, stop, side="right"))
    if right <= left:
        return False
    events = event_times[left:right]
    if exclude_time is not None:
        matching = np.flatnonzero(np.isclose(events, exclude_time, atol=1e-9, rtol=0))
        if len(matching):
            # Remove the target itself, but retain a duplicate at the same time
            # so malformed presentation tables fail closed.
            events = np.delete(events, matching[0])
    return bool(len(events))


def construct_windows(
    presentations: pd.DataFrame,
    *,
    policy: WindowPolicy = DEFAULT_WINDOW_POLICY,
    support_start_s: float = -math.inf,
    support_stop_s: float = math.inf,
) -> WindowSelection:
    """Construct clean omission and strictly ordinary calibration windows.

    Omission windows reject any *other* omission and any image/change or sham
    change inside the analysis interval.  Normal centers must themselves be
    active, non-omitted, non-change, and non-sham.  They additionally reject
    every omission/change/sham event within the symmetric contamination guard
    (or the analysis interval, whichever extends farther).
    """

    frame = _standardize_presentations(presentations)
    frame = frame.loc[frame["active"] & frame["start_time"].notna()].copy()
    omitted_times = np.sort(frame.loc[frame["omitted"], "start_time"].to_numpy(dtype=np.float64))
    change_mask = frame["is_change"] | frame["is_sham_change"]
    change_times = np.sort(frame.loc[change_mask, "start_time"].to_numpy(dtype=np.float64))

    omission_candidates = frame.loc[frame["omitted"]].copy()
    normal_candidates = frame.loc[
        ~frame["omitted"] & ~frame["is_change"] & ~frame["is_sham_change"]
    ].copy()
    audit = {
        "omission_candidates": len(omission_candidates),
        "normal_candidates": len(normal_candidates),
        "omission_rejected_boundary": 0,
        "omission_rejected_contamination": 0,
        "normal_rejected_boundary": 0,
        "normal_rejected_contamination": 0,
    }

    def in_support(center: float) -> bool:
        return (
            center + policy.window_start_s >= support_start_s
            and center + policy.window_end_s <= support_stop_s
        )

    accepted_omissions = []
    for index, row in omission_candidates.iterrows():
        center = float(row["start_time"])
        if not in_support(center):
            audit["omission_rejected_boundary"] += 1
            continue
        window_start = center + policy.window_start_s
        window_stop = center + policy.window_end_s
        contaminated = _has_event_in_interval(change_times, window_start, window_stop)
        contaminated |= _has_event_in_interval(
            omitted_times,
            window_start,
            window_stop,
            exclude_time=center,
        )
        if contaminated:
            audit["omission_rejected_contamination"] += 1
            continue
        accepted_omissions.append(index)

    guard_start = min(
        policy.window_start_s,
        -policy.normal_contamination_guard_s,
    )
    guard_stop = max(
        policy.window_end_s,
        policy.normal_contamination_guard_s,
    )
    accepted_normal = []
    for index, row in normal_candidates.iterrows():
        center = float(row["start_time"])
        if not in_support(center):
            audit["normal_rejected_boundary"] += 1
            continue
        contamination_start = center + guard_start
        contamination_stop = center + guard_stop
        contaminated = _has_event_in_interval(change_times, contamination_start, contamination_stop)
        contaminated |= _has_event_in_interval(
            omitted_times, contamination_start, contamination_stop
        )
        if contaminated:
            audit["normal_rejected_contamination"] += 1
            continue
        accepted_normal.append(index)

    omissions = frame.loc[accepted_omissions].copy()
    omissions["window_kind"] = "omission"
    normal = frame.loc[accepted_normal].copy()
    normal["window_kind"] = "normal"
    for selected in (omissions, normal):
        selected["event_time"] = selected["start_time"].astype(float)
        selected["window_start_time"] = selected["event_time"] + policy.window_start_s
        selected["window_stop_time"] = selected["event_time"] + policy.window_end_s
    omissions = omissions.reset_index(drop=True)
    normal = normal.reset_index(drop=True)
    audit["omission_accepted"] = len(omissions)
    audit["normal_accepted"] = len(normal)
    assert_calibration_is_normal(normal)
    return WindowSelection(omissions=omissions, normal=normal, audit=audit)


def deterministic_calibration_subset(
    windows: pd.DataFrame,
    count: int | None,
    *,
    seed: int = 20260725,
) -> pd.DataFrame:
    """Select a reproducible session-wide subset without relying on row order."""

    assert_calibration_is_normal(windows)
    if count is None or count >= len(windows):
        return windows.sort_values("event_time", kind="stable").reset_index(drop=True)
    if count <= 0:
        raise ValueError("calibration window count must be positive")

    def digest(row: pd.Series) -> str:
        identifier = _canonical_identifier(row["stimulus_presentation_id"])
        value = f"cadence-normal-v1\0{seed}\0{identifier}\0{row['event_time']:.9f}"
        return hashlib.sha256(value.encode()).hexdigest()

    ranked = windows.copy()
    ranked["_selection_digest"] = ranked.apply(digest, axis=1)
    ranked = ranked.sort_values(
        ["_selection_digest", "stimulus_presentation_id"],
        kind="stable",
    ).head(count)
    return (
        ranked.drop(columns="_selection_digest")
        .sort_values("event_time", kind="stable")
        .reset_index(drop=True)
    )


def _infer_max_gap(timestamps: np.ndarray, multiplier: float = 5.0) -> float:
    return float(np.median(np.diff(timestamps)) * multiplier)


def _linear_matrix_window(
    data: Any,
    timestamps: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate one trial while reading one contiguous HDF5 slice."""

    right = np.searchsorted(timestamps, targets, side="left")
    right = np.clip(right, 1, len(timestamps) - 1)
    left = right - 1
    first = int(left.min())
    last = int(right.max()) + 1
    block = np.asarray(data[first:last, :], dtype=np.float32)
    local_left = left - first
    local_right = right - first
    left_time = timestamps[left]
    right_time = timestamps[right]
    delta = right_time - left_time
    weight = (targets - left_time) / delta
    left_values = block[local_left]
    right_values = block[local_right]
    values = left_values + weight[:, None].astype(np.float32) * (right_values - left_values)
    exact = np.isclose(targets, right_time, atol=1e-10, rtol=0)
    values[exact] = right_values[exact]
    bracket_valid = (
        (targets >= timestamps[0])
        & (targets <= timestamps[-1])
        & (delta <= _infer_max_gap(timestamps))
    )[:, None]
    bracket_valid = bracket_valid & np.isfinite(left_values) & np.isfinite(right_values)
    exact_valid = exact[:, None] & np.isfinite(right_values)
    valid = bracket_valid | exact_valid
    values[~valid] = np.nan
    return values.astype(np.float32, copy=False), valid


def _linear_vector(
    values: np.ndarray,
    timestamps: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(timestamps, targets, side="left")
    right = np.clip(right, 1, len(timestamps) - 1)
    left = right - 1
    left_time = timestamps[left]
    right_time = timestamps[right]
    delta = right_time - left_time
    left_values = values[left]
    right_values = values[right]
    weight = (targets - left_time) / delta
    result = left_values + weight * (right_values - left_values)
    exact = np.isclose(targets, right_time, atol=1e-10, rtol=0)
    result[exact] = right_values[exact]
    bracket_valid = (
        (targets >= timestamps[0])
        & (targets <= timestamps[-1])
        & (delta <= _infer_max_gap(timestamps))
        & np.isfinite(left_values)
        & np.isfinite(right_values)
    )
    valid = bracket_valid | (exact & np.isfinite(right_values))
    result = result.astype(np.float32, copy=False)
    result[~valid] = np.nan
    return result, valid


def _lick_rate(
    lick_timestamps: np.ndarray,
    targets: np.ndarray,
    *,
    rate_hz: float,
) -> np.ndarray:
    half_width = 0.5 / rate_hz
    edges = np.concatenate(
        ([targets[0] - half_width], (targets[:-1] + targets[1:]) / 2, [targets[-1] + half_width])
    )
    counts, _ = np.histogram(lick_timestamps, bins=edges)
    return counts.astype(np.float32) * rate_hz


def _extract_windows(
    source: _NWBSignals,
    windows: pd.DataFrame,
    relative_time: np.ndarray,
    *,
    rate_hz: float,
) -> dict[str, np.ndarray]:
    trial_count = len(windows)
    time_count = len(relative_time)
    cell_count = len(source.cell_roi_ids)
    neural = np.empty((trial_count, time_count, cell_count), dtype=np.float32)
    neural_valid = np.empty_like(neural, dtype=bool)
    behavior = np.empty((trial_count, time_count, 3), dtype=np.float32)
    behavior_valid = np.empty_like(behavior, dtype=bool)
    pupil = source.pupil_area.copy()
    pupil[source.likely_blink | ~np.isfinite(pupil) | (pupil <= 0)] = np.nan

    for trial_index, event_time in enumerate(windows["event_time"].to_numpy(float)):
        targets = event_time + relative_time
        neural[trial_index], neural_valid[trial_index] = _linear_matrix_window(
            source.neural_data,
            source.neural_timestamps,
            targets,
        )
        running, running_valid = _linear_vector(
            source.running_speed,
            source.running_timestamps,
            targets,
        )
        pupil_area, pupil_valid = _linear_vector(
            pupil,
            source.pupil_timestamps,
            targets,
        )
        behavior[trial_index, :, 0] = running
        behavior[trial_index, :, 1] = pupil_area
        behavior[trial_index, :, 2] = _lick_rate(
            source.lick_timestamps,
            targets,
            rate_hz=rate_hz,
        )
        behavior_valid[trial_index, :, 0] = running_valid
        behavior_valid[trial_index, :, 1] = pupil_valid
        behavior_valid[trial_index, :, 2] = True

    return {
        "neural": neural,
        "neural_valid": neural_valid,
        "behavior": behavior,
        "behavior_valid": behavior_valid,
        "event_times": windows["event_time"].to_numpy(np.float64),
        "presentation_ids": windows["stimulus_presentation_id"].to_numpy(np.int64),
    }


def _signal_support(source: _NWBSignals) -> tuple[float, float]:
    start = max(
        source.neural_timestamps[0],
        source.running_timestamps[0],
        source.pupil_timestamps[0],
    )
    stop = min(
        source.neural_timestamps[-1],
        source.running_timestamps[-1],
        source.pupil_timestamps[-1],
    )
    if not start < stop:
        raise ValueError("neural and behavioral signals have no common time support")
    return float(start), float(stop)


def _sha256_file(path: Path) -> str:
    return hash_file(path, "sha256")


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        np.savez_compressed(temporary, **arrays)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def split_window_arrays(
    arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Physically separate normal support, safe query data, and sealed outcomes."""

    relative = np.asarray(arrays["relative_time_s"], dtype=np.float64)
    onset_candidates = np.flatnonzero(relative >= 0)
    if not len(onset_candidates) or not np.isclose(relative[onset_candidates[0]], 0.0):
        raise ValueError("window grid must contain an exact zero-time onset sample")
    onset = int(onset_candidates[0])
    shared_names = (
        "relative_time_s",
        "behavior_channel_names",
        "cell_roi_ids",
        "cell_specimen_ids",
    )
    shared = {name: np.asarray(arrays[name]) for name in shared_names}
    normal = {
        **shared,
        **{
            name: np.asarray(arrays[name])
            for name in (
                "normal_neural",
                "normal_neural_valid",
                "normal_behavior",
                "normal_behavior_valid",
                "normal_event_times",
                "normal_presentation_ids",
            )
        },
    }
    query = {
        **shared,
        "onset": np.asarray(onset, dtype=np.int64),
        "omission_pre_neural": np.asarray(arrays["omission_neural"])[:, :onset],
        "omission_pre_neural_valid": np.asarray(arrays["omission_neural_valid"])[:, :onset],
        "omission_pre_behavior": np.asarray(arrays["omission_behavior"])[:, :onset],
        "omission_pre_behavior_valid": np.asarray(arrays["omission_behavior_valid"])[:, :onset],
        "omission_event_times": np.asarray(arrays["omission_event_times"]),
        "omission_presentation_ids": np.asarray(arrays["omission_presentation_ids"]),
    }
    sealed = {
        "onset": np.asarray(onset, dtype=np.int64),
        "omission_presentation_ids": np.asarray(arrays["omission_presentation_ids"]),
        "omission_post_neural": np.asarray(arrays["omission_neural"])[:, onset:],
        "omission_post_neural_valid": np.asarray(arrays["omission_neural_valid"])[:, onset:],
        "omission_post_behavior": np.asarray(arrays["omission_behavior"])[:, onset:],
        "omission_post_behavior_valid": np.asarray(arrays["omission_behavior_valid"])[:, onset:],
    }
    return normal, query, sealed


def split_combined_animal_artifact(
    animal_directory: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path, Path]:
    """Migrate one legacy ``windows.npz`` into three role-separated artifacts."""

    directory = Path(animal_directory)
    source = directory / "windows.npz"
    destinations = (
        directory / "normal_support.npz",
        directory / "omission_query.npz",
        directory / "sealed_omission_outcomes.npz",
    )
    if all(path.exists() for path in destinations) and not overwrite:
        return destinations
    with np.load(source, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    normal, query, sealed = split_window_arrays(arrays)
    for path, payload in zip(destinations, (normal, query, sealed), strict=True):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite split artifact {path}")
        _atomic_npz(path, payload)
    return destinations


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        frame.to_parquet(temporary_path, index=False, compression="zstd")
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def extract_animal_nwb(
    nwb_path: str | Path,
    output_directory: str | Path,
    *,
    backend: Literal["pynwb", "h5py"] = "pynwb",
    policy: WindowPolicy = DEFAULT_WINDOW_POLICY,
    normal_calibration_trials: int | None = 160,
    minimum_omissions: int = 0,
    selection_seed: int = 20260725,
    manifest_entry: Mapping[str, Any] | None = None,
    verify_input_hash: bool = False,
) -> AnimalOutputPaths:
    """Extract one animal into compressed arrays, tables, and full provenance."""

    source_path = Path(nwb_path)
    if manifest_entry is not None:
        expected_experiment = _canonical_identifier(manifest_entry["ophys_experiment_id"])
        expected_mouse = _canonical_identifier(manifest_entry["mouse_id"])
        expected_size = manifest_entry.get("size_bytes")
        if expected_size is not None and source_path.stat().st_size != int(expected_size):
            raise ValueError("input NWB size does not match the release manifest")
        if verify_input_hash:
            verify_download(source_path, manifest_entry)
    else:
        expected_experiment = None
        expected_mouse = None

    with open_nwb_signals(source_path, backend=backend) as source:
        if expected_experiment is not None and source.experiment_id != expected_experiment:
            raise ValueError("NWB experiment identifier does not match the manifest")
        if expected_mouse is not None and source.mouse_id != expected_mouse:
            raise ValueError("NWB mouse identifier does not match the manifest")
        support_start, support_stop = _signal_support(source)
        selection = construct_windows(
            source.presentations,
            policy=policy,
            support_start_s=support_start,
            support_stop_s=support_stop,
        )
        if len(selection.omissions) < minimum_omissions:
            raise ValueError(
                f"only {len(selection.omissions)} clean omissions; "
                f"minimum requested is {minimum_omissions}"
            )
        normal = deterministic_calibration_subset(
            selection.normal,
            normal_calibration_trials,
            seed=selection_seed,
        )
        omissions = selection.omissions.sort_values("event_time", kind="stable").reset_index(
            drop=True
        )
        relative_time = policy.relative_time.astype(np.float64)
        omission_arrays = _extract_windows(
            source,
            omissions,
            relative_time,
            rate_hz=policy.rate_hz,
        )
        normal_arrays = _extract_windows(
            source,
            normal,
            relative_time,
            rate_hz=policy.rate_hz,
        )

        arrays: dict[str, np.ndarray] = {
            "relative_time_s": relative_time,
            "behavior_channel_names": np.asarray(
                ["running_speed_cm_per_s", "pupil_area", "lick_rate_hz"],
                dtype="U24",
            ),
            "cell_roi_ids": source.cell_roi_ids.astype(np.int64),
            "cell_specimen_ids": source.cell_specimen_ids.astype(np.int64),
            "neural_source_timestamps_s": source.neural_timestamps.astype(np.float64),
            "running_source_timestamps_s": source.running_timestamps.astype(np.float64),
            "pupil_source_timestamps_s": source.pupil_timestamps.astype(np.float64),
            "lick_event_timestamps_s": source.lick_timestamps.astype(np.float64),
        }
        for kind, extracted in (("omission", omission_arrays), ("normal", normal_arrays)):
            arrays.update({f"{kind}_{name}": value for name, value in extracted.items()})

        windows = pd.concat(
            [
                omissions.assign(window_index=np.arange(len(omissions), dtype=np.int64)),
                normal.assign(window_index=np.arange(len(normal), dtype=np.int64)),
            ],
            ignore_index=True,
        )
        windows.insert(0, "mouse_id", source.mouse_id)
        windows.insert(1, "ophys_experiment_id", int(source.experiment_id))
        assert_calibration_is_normal(windows.loc[windows["window_kind"].eq("normal")])

        animal_directory = Path(output_directory) / f"mouse_{source.mouse_id}"
        paths = AnimalOutputPaths(
            directory=animal_directory,
            arrays=animal_directory / "windows.npz",
            normal_support=animal_directory / "normal_support.npz",
            omission_query=animal_directory / "omission_query.npz",
            sealed_omission_outcomes=(animal_directory / "sealed_omission_outcomes.npz"),
            stimulus_presentations=animal_directory / "stimulus_presentations.parquet",
            windows=animal_directory / "window_index.parquet",
            provenance=animal_directory / "provenance.json",
        )
        _atomic_npz(paths.arrays, arrays)
        normal_support, omission_query, sealed_omission = split_window_arrays(arrays)
        _atomic_npz(paths.normal_support, normal_support)
        _atomic_npz(paths.omission_query, omission_query)
        _atomic_npz(paths.sealed_omission_outcomes, sealed_omission)
        _atomic_parquet(source.presentations, paths.stimulus_presentations)
        _atomic_parquet(windows, paths.windows)

        output_hashes = {
            path.name: {
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in (
                paths.arrays,
                paths.normal_support,
                paths.omission_query,
                paths.sealed_omission_outcomes,
                paths.stimulus_presentations,
                paths.windows,
            )
        }
        input_fingerprint: dict[str, Any] = {
            "path": str(source_path.resolve()),
            "size_bytes": source_path.stat().st_size,
        }
        if manifest_entry is not None:
            input_fingerprint.update(
                {
                    key: manifest_entry.get(key)
                    for key in (
                        "url",
                        "s3_uri",
                        "s3_version_id",
                        "blake2b_512",
                        "etag",
                    )
                }
            )
            input_fingerprint["content_hash_verified_this_run"] = verify_input_hash
        else:
            input_fingerprint["sha256"] = _sha256_file(source_path)

        provenance = {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "mouse_id": source.mouse_id,
            "ophys_experiment_id": int(source.experiment_id),
            "input": input_fingerprint,
            "extractor": {
                "backend": backend,
                "window_policy": asdict(policy),
                "normal_calibration_trials_requested": normal_calibration_trials,
                "minimum_omissions": minimum_omissions,
                "selection_seed": selection_seed,
                "versions": {
                    "cadence-neuro": _package_version("cadence-neuro"),
                    "h5py": _package_version("h5py"),
                    "numpy": _package_version("numpy"),
                    "pandas": _package_version("pandas"),
                    "pyarrow": _package_version("pyarrow"),
                    "pynwb": _package_version("pynwb"),
                },
            },
            "signals": {
                "neural_event_samples": len(source.neural_timestamps),
                "cells": len(source.cell_roi_ids),
                "running_samples": len(source.running_timestamps),
                "pupil_samples": len(source.pupil_timestamps),
                "lick_events": len(source.lick_timestamps),
                "common_support_s": [support_start, support_stop],
                "behavior_channels": arrays["behavior_channel_names"].tolist(),
            },
            "window_audit": {
                **selection.audit,
                "normal_selected": len(normal),
                "omission_selected": len(omissions),
                "time_samples_per_window": len(relative_time),
            },
            "outputs": output_hashes,
        }
        write_json(provenance, paths.provenance)
        return paths


def inspect_nwb(
    nwb_path: str | Path,
    *,
    backend: Literal["pynwb", "h5py"] = "pynwb",
    policy: WindowPolicy = DEFAULT_WINDOW_POLICY,
) -> dict[str, Any]:
    """Return a lightweight schema/cohort inspection without extracting arrays."""

    path = Path(nwb_path)
    with open_nwb_signals(path, backend=backend) as source:
        support_start, support_stop = _signal_support(source)
        windows = construct_windows(
            source.presentations,
            policy=policy,
            support_start_s=support_start,
            support_stop_s=support_stop,
        )
        return {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "backend": backend,
            "ophys_experiment_id": source.experiment_id,
            "mouse_id": source.mouse_id,
            "neural_event_shape": list(source.neural_data.shape),
            "neural_timestamp_count": len(source.neural_timestamps),
            "running_sample_count": len(source.running_timestamps),
            "pupil_sample_count": len(source.pupil_timestamps),
            "lick_event_count": len(source.lick_timestamps),
            "stimulus_presentation_count": len(source.presentations),
            "common_support_s": [support_start, support_stop],
            "window_audit": windows.audit,
        }


def _json_scalar(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _jsonify(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def _manifest_entry_for_experiment(
    manifest_path: str | Path | None,
    experiment_id: int,
) -> Mapping[str, Any] | None:
    if manifest_path is None:
        return None
    manifest = json.loads(Path(manifest_path).read_text())
    matches = [
        entry
        for entry in manifest["nwb_files"]
        if int(entry["ophys_experiment_id"]) == experiment_id
    ]
    if len(matches) != 1:
        raise KeyError(f"expected one manifest entry for experiment {experiment_id}")
    return matches[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cadence.data.allen_vbo",
        description="Prepare the frozen Allen Visual Behavior Ophys mouse cohort.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cohort = subparsers.add_parser("cohort", help="select mice and write a public manifest")
    cohort.add_argument("--experiments", required=True, type=Path)
    cohort.add_argument("--cells", required=True, type=Path)
    cohort.add_argument(
        "--project-manifest",
        default=ALLEN_PROJECT_MANIFEST_URL,
        help="v1.1.0 manifest path or public URL",
    )
    cohort.add_argument("--output", required=True, type=Path)
    cohort.add_argument("--cohort-parquet", type=Path)
    cohort.add_argument(
        "--skip-head",
        action="store_true",
        help="do not query public S3 sizes/ETags",
    )

    download = subparsers.add_parser("download", help="download only explicitly selected NWBs")
    download.add_argument("--manifest", required=True, type=Path)
    download.add_argument("--destination", required=True, type=Path)
    download.add_argument("--experiment-id", action="append", type=int)
    download.add_argument("--overwrite", action="store_true")
    download.add_argument("--no-verify", action="store_true")

    inspect = subparsers.add_parser("inspect", help="inspect signal and window counts")
    inspect.add_argument("--nwb", required=True, type=Path)
    inspect.add_argument("--backend", choices=("pynwb", "h5py"), default="pynwb")
    inspect.add_argument("--output", type=Path)

    extract = subparsers.add_parser("extract", help="write compressed per-animal outputs")
    extract.add_argument("--nwb", required=True, type=Path)
    extract.add_argument("--output-directory", required=True, type=Path)
    extract.add_argument("--manifest", type=Path)
    extract.add_argument("--backend", choices=("pynwb", "h5py"), default="pynwb")
    extract.add_argument("--normal-trials", type=int, default=160)
    extract.add_argument("--minimum-omissions", type=int, default=80)
    extract.add_argument("--verify-input-hash", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "cohort":
        spec = CohortSpec()
        selected = select_one_experiment_per_mouse(args.experiments, args.cells, spec=spec)
        project, project_provenance = load_project_manifest(args.project_manifest)
        manifest = build_public_download_manifest(
            selected,
            project,
            cohort_spec=spec,
            release_manifest_provenance=project_provenance,
            fetch_object_metadata=not args.skip_head,
        )
        write_json(manifest, args.output)
        if args.cohort_parquet is not None:
            _atomic_parquet(selected, args.cohort_parquet)
        print(json.dumps({"animals": len(selected), "manifest": str(args.output)}))
        return 0
    if args.command == "download":
        result = download_manifest_nwbs(
            args.manifest,
            args.destination,
            experiment_ids=args.experiment_id,
            overwrite=args.overwrite,
            verify=not args.no_verify,
        )
        print(result.to_json(orient="records", indent=2))
        return 0
    if args.command == "inspect":
        result = inspect_nwb(args.nwb, backend=args.backend)
        if args.output is not None:
            write_json(result, args.output)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "extract":
        with h5py.File(args.nwb, "r") as file:
            experiment_id = int(_h5_scalar(file["identifier"]))
        entry = _manifest_entry_for_experiment(args.manifest, experiment_id)
        paths = extract_animal_nwb(
            args.nwb,
            args.output_directory,
            backend=args.backend,
            normal_calibration_trials=args.normal_trials,
            minimum_omissions=args.minimum_omissions,
            manifest_entry=entry,
            verify_input_hash=args.verify_input_hash,
        )
        print(json.dumps({key: str(value) for key, value in asdict(paths).items()}, indent=2))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
