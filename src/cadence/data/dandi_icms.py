"""Frozen DANDI:001868 ICMS download and leakage-safe preprocessing.

The biological unit is the animal.  The six task animals are fixed by the
published release and all preprocessing keeps their sessions in separate
files.  Catch-trial calibration is deliberately conservative: a trial is
normal only when ``current_uA == 0`` *and* no electrical-stimulation interval
overlaps the complete extracted window.  Because task mouse ICMS83 has no
catches, the same normal partition can include deterministically sampled
continuous ephys/wheel ITI windows whose full extent is at least two seconds
from every trial and stimulation interval.

The raw Ripple ``stim_channel`` is retained only as audit metadata.  It is
never part of :data:`INTERVENTION_DESCRIPTOR_COLUMNS`; stimulation is encoded
using physical NET32 coordinates and pulse-train physics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import pandas as pd

DANDISET_ID = "001868"
DANDISET_VERSION = "0.260715.2016"
DANDI_API = "https://api.dandiarchive.org/api"
DANDI_VERSION_URL = f"{DANDI_API}/dandisets/{DANDISET_ID}/versions/{DANDISET_VERSION}/"
DANDI_LANDING_URL = f"https://dandiarchive.org/dandiset/{DANDISET_ID}/{DANDISET_VERSION}"

TASK_MICE = ("ICMS83", "ICMS92", "ICMS93", "ICMS98", "ICMS100", "ICMS101")
EXPECTED_ASSET_COUNT = 85
EXPECTED_TOTAL_BYTES = 7_504_049_197
EXPECTED_TASK_ASSET_COUNT = 55
EXPECTED_TASK_ASSET_BYTES = 7_414_701_188
EXPECTED_TRIMODAL_ASSET_COUNT = 45
EXPECTED_TRIMODAL_ASSET_BYTES = 6_812_254_225

DEFAULT_MANIFEST_PATH = Path("configs/dandi_001868_assets.json")
DEFAULT_RAW_ROOT = Path("data/raw/dandi_001868")
DEFAULT_PROCESSED_ROOT = Path("data/processed/dandi_001868")

# The published NWBs identify channel_name as the original Ripple channel and
# rel_y as the physical coordinate in the NET32 electrode group.  Wiring is not
# identical across implants (notably ICMS83), which is exactly why raw channel
# identity cannot be transferred across animals.  The common map below applies
# to five implants.  ICMS83's independently observed wiring is frozen
# separately.  Coordinates present in each NWB always take precedence and are
# validated against the physical 60 um grid.
_COMMON_CHANNEL_REL_Y_UM: dict[int, float] = {
    1: 480.0,
    2: 960.0,
    3: 1440.0,
    4: 0.0,
    5: 600.0,
    6: 1080.0,
    7: 1560.0,
    8: 120.0,
    9: 720.0,
    10: 1200.0,
    11: 1680.0,
    12: 240.0,
    13: 840.0,
    14: 360.0,
    15: 1800.0,
    16: 1320.0,
    17: 1020.0,
    18: 1500.0,
    19: 60.0,
    20: 540.0,
    21: 1140.0,
    22: 660.0,
    23: 180.0,
    24: 1620.0,
    25: 1260.0,
    26: 780.0,
    27: 300.0,
    28: 1740.0,
    29: 1380.0,
    30: 900.0,
    31: 420.0,
    32: 1860.0,
}
_ICMS83_CHANNEL_REL_Y_UM: dict[int, float] = {
    1: 1380.0,
    2: 900.0,
    3: 420.0,
    4: 1860.0,
    5: 1260.0,
    6: 780.0,
    7: 300.0,
    9: 1140.0,
    11: 180.0,
    13: 1020.0,
    14: 1500.0,
    20: 1320.0,
    22: 1200.0,
    24: 240.0,
    25: 600.0,
    26: 1080.0,
    28: 120.0,
    29: 480.0,
    30: 960.0,
    32: 0.0,
}
TASK_ANIMAL_CHANNEL_REL_Y_UM: dict[str, dict[int, float]] = {
    animal: dict(_COMMON_CHANNEL_REL_Y_UM)
    for animal in ("ICMS92", "ICMS93", "ICMS98", "ICMS100", "ICMS101")
}
TASK_ANIMAL_CHANNEL_REL_Y_UM["ICMS83"] = _ICMS83_CHANNEL_REL_Y_UM
NET32_DEPTH_CENTER_UM = 930.0
NET32_DEPTH_HALF_RANGE_UM = 930.0
_NET32_ALLOWED_DEPTHS_UM = np.arange(0.0, 1860.0 + 60.0, 60.0)

INTERVENTION_DESCRIPTOR_COLUMNS = (
    "stim_present",
    "current_uA",
    "frequency_hz",
    "pulse_count",
    "pulse_width_us",
    "electrode_rel_x_um",
    "electrode_rel_y_um",
    "electrode_rel_z_um",
    "electrode_depth_centered_um",
    "electrode_depth_fraction",
)

_TRIAL_REQUIRED_COLUMNS = {
    "start_time",
    "stop_time",
    "trial_index",
    "current_uA",
    "stim_channel",
    "is_hit",
    "response_time",
    "is_good_trial",
}
_EVENT_REQUIRED_COLUMNS = {
    "start_time",
    "stop_time",
    "trial_index",
    "current_uA",
    "stim_channel",
    "pulse_count",
    "frequency_hz",
    "pulse_width_us",
}


class DANDIICMSError(ValueError):
    """Raised when a release, NWB, or split invariant is violated."""


@dataclass(frozen=True)
class ICMSPreprocessConfig:
    """Prespecified time grid and data-quality policy."""

    sample_rate_hz: float = 30.0
    window_start_s: float = -1.0
    window_stop_s: float = 3.0
    calcium_max_gap_s: float = 0.75
    wheel_max_gap_s: float = 0.050
    ephys_artifact_start_s: float = -0.002
    ephys_artifact_stop_s: float = 0.705
    event_onset_tolerance_s: float = 0.300
    include_iti_calibration: bool = True
    iti_guard_s: float = 2.0
    iti_windows_per_session: int = 40
    good_trials_only: bool = True
    hdf5_compression_level: int = 4

    def __post_init__(self) -> None:
        finite_positive = {
            "sample_rate_hz": self.sample_rate_hz,
            "calcium_max_gap_s": self.calcium_max_gap_s,
            "wheel_max_gap_s": self.wheel_max_gap_s,
            "event_onset_tolerance_s": self.event_onset_tolerance_s,
        }
        for name, value in finite_positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not self.window_start_s < self.window_stop_s:
            raise ValueError("window_start_s must be less than window_stop_s")
        if not self.ephys_artifact_start_s < self.ephys_artifact_stop_s:
            raise ValueError("ephys_artifact_start_s must be less than ephys_artifact_stop_s")
        if not math.isfinite(self.iti_guard_s) or self.iti_guard_s < 0:
            raise ValueError("iti_guard_s must be finite and non-negative")
        if self.iti_windows_per_session < 0:
            raise ValueError("iti_windows_per_session may not be negative")
        samples = (self.window_stop_s - self.window_start_s) * self.sample_rate_hz
        if not np.isclose(samples, round(samples), atol=1e-9):
            raise ValueError("the analysis window must contain an integer number of bins")
        if not 0 <= self.hdf5_compression_level <= 9:
            raise ValueError("hdf5_compression_level must be between 0 and 9")

    @property
    def n_time(self) -> int:
        return int(round((self.window_stop_s - self.window_start_s) * self.sample_rate_hz))

    @property
    def relative_edges_s(self) -> np.ndarray:
        return self.window_start_s + np.arange(self.n_time + 1) / self.sample_rate_hz

    @property
    def relative_time_s(self) -> np.ndarray:
        edges = self.relative_edges_s
        return 0.5 * (edges[:-1] + edges[1:])


DEFAULT_PREPROCESS_CONFIG = ICMSPreprocessConfig()


@dataclass(frozen=True)
class DANDIAsset:
    """One checksummed asset from the frozen published release."""

    asset_id: str
    blob_id: str
    path: str
    size: int
    sha256: str
    dandi_etag: str
    download_url: str
    task_animal: bool
    trimodal: bool

    @property
    def animal_id(self) -> str:
        return Path(self.path).parts[0].removeprefix("sub-")


@dataclass
class ICMSSession:
    """Fully aligned arrays and audit metadata for one trimodal session."""

    animal_id: str
    session_id: str
    session_start_time: str
    source_path: str
    source_sha256: str | None
    time_s: np.ndarray
    trial_metadata: pd.DataFrame
    intervention_descriptors: np.ndarray
    calcium_dff: np.ndarray
    calcium_valid_mask: np.ndarray
    calcium_observed_mask: np.ndarray
    spike_rate_hz: np.ndarray
    spike_valid_mask: np.ndarray
    wheel_position: np.ndarray
    wheel_displacement: np.ndarray
    wheel_velocity: np.ndarray
    wheel_valid_mask: np.ndarray
    roi_metadata: pd.DataFrame
    unit_metadata: pd.DataFrame
    audit: dict[str, Any]


def _json_request(url: str, *, timeout_s: float = 120.0) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "cadence-neuro-dandi-001868/1",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.load(response)


def _asset_subject(path: str) -> str:
    parts = Path(path).parts
    if len(parts) < 2 or not parts[0].startswith("sub-"):
        raise DANDIICMSError(f"unexpected DANDI asset path: {path!r}")
    return parts[0].removeprefix("sub-")


def _is_trimodal_task_path(path: str) -> bool:
    return _asset_subject(path) in TASK_MICE and Path(path).name.endswith(
        "_behavior+ecephys+ophys.nwb"
    )


def build_published_manifest(*, workers: int = 12) -> dict[str, Any]:
    """Query the immutable published API version and freeze all SHA-256 digests."""

    version = _json_request(DANDI_VERSION_URL)
    if str(version.get("version")) != DANDISET_VERSION:
        raise DANDIICMSError("the API did not return the requested published version")
    if not version.get("datePublished"):
        raise DANDIICMSError("DANDI version is not published")
    license_values = list(version.get("license", []))
    if "spdx:CC-BY-4.0" not in license_values:
        raise DANDIICMSError(f"unexpected DANDI license: {license_values}")

    page_url: str | None = f"{DANDI_VERSION_URL}assets/?page_size=100"
    listed: list[dict[str, Any]] = []
    while page_url:
        page = _json_request(page_url)
        listed.extend(page["results"])
        page_url = page.get("next")

    def enrich(asset: Mapping[str, Any]) -> dict[str, Any]:
        info = _json_request(f"{DANDI_API}/assets/{asset['asset_id']}/info/")
        metadata = info["metadata"]
        digest = metadata.get("digest", {})
        sha256 = str(digest.get("dandi:sha2-256", "")).lower()
        dandi_etag = str(digest.get("dandi:dandi-etag", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise DANDIICMSError(f"asset {asset['asset_id']} has no SHA-256 digest")
        path = str(asset["path"])
        size = int(asset["size"])
        if str(metadata.get("path")) != path or int(metadata.get("contentSize", -1)) != size:
            raise DANDIICMSError(f"listing/metadata mismatch for {path}")
        return {
            "asset_id": str(asset["asset_id"]),
            "blob_id": str(asset["blob"]),
            "path": path,
            "size": size,
            "sha256": sha256,
            "dandi_etag": dandi_etag,
            "download_url": f"{DANDI_API}/assets/{asset['asset_id']}/download/",
            "task_animal": _asset_subject(path) in TASK_MICE,
            "trimodal": _is_trimodal_task_path(path),
        }

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        assets = list(executor.map(enrich, listed))
    assets.sort(key=lambda item: item["path"])
    result = {
        "schema_version": 1,
        "dandiset_id": DANDISET_ID,
        "version": DANDISET_VERSION,
        "published_at": str(version["datePublished"]),
        "license": license_values,
        "api_version_url": DANDI_VERSION_URL,
        "landing_url": DANDI_LANDING_URL,
        "asset_count": len(assets),
        "total_bytes": sum(int(item["size"]) for item in assets),
        "task_mice": sorted(TASK_MICE),
        "task_asset_count": sum(bool(item["task_animal"]) for item in assets),
        "task_asset_bytes": sum(int(item["size"]) for item in assets if item["task_animal"]),
        "trimodal_asset_count": sum(bool(item["trimodal"]) for item in assets),
        "trimodal_asset_bytes": sum(int(item["size"]) for item in assets if item["trimodal"]),
        "assets": assets,
    }
    validate_frozen_manifest(result)
    return result


def write_json_atomic(payload: Mapping[str, Any], destination: str | Path) -> Path:
    """Write deterministic JSON through an fsync'd temporary file."""

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, output)
    return output


def validate_frozen_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed if the frozen release no longer has its audited shape."""

    expected_scalars = {
        "schema_version": 1,
        "dandiset_id": DANDISET_ID,
        "version": DANDISET_VERSION,
        "asset_count": EXPECTED_ASSET_COUNT,
        "total_bytes": EXPECTED_TOTAL_BYTES,
        "task_asset_count": EXPECTED_TASK_ASSET_COUNT,
        "task_asset_bytes": EXPECTED_TASK_ASSET_BYTES,
        "trimodal_asset_count": EXPECTED_TRIMODAL_ASSET_COUNT,
        "trimodal_asset_bytes": EXPECTED_TRIMODAL_ASSET_BYTES,
    }
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected:
            raise DANDIICMSError(
                f"frozen manifest {key} mismatch: expected {expected!r}, "
                f"observed {manifest.get(key)!r}"
            )
    if tuple(sorted(manifest.get("task_mice", []))) != tuple(sorted(TASK_MICE)):
        raise DANDIICMSError("frozen manifest task-mouse cohort changed")
    if "spdx:CC-BY-4.0" not in manifest.get("license", []):
        raise DANDIICMSError("frozen manifest does not retain CC BY 4.0")

    raw_assets = manifest.get("assets")
    if not isinstance(raw_assets, list) or len(raw_assets) != EXPECTED_ASSET_COUNT:
        raise DANDIICMSError("frozen manifest asset list has the wrong length")
    paths: set[str] = set()
    ids: set[str] = set()
    task_counts = {mouse: 0 for mouse in TASK_MICE}
    trimodal_counts = {mouse: 0 for mouse in TASK_MICE}
    total = task_total = trimodal_total = 0
    for item in raw_assets:
        path = str(item.get("path", ""))
        asset_id = str(item.get("asset_id", ""))
        sha256 = str(item.get("sha256", "")).lower()
        size = int(item.get("size", -1))
        if path in paths or asset_id in ids:
            raise DANDIICMSError("frozen manifest repeats an asset path or ID")
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise DANDIICMSError(f"invalid SHA-256 for {path}")
        if size <= 0:
            raise DANDIICMSError(f"invalid byte length for {path}")
        expected_task = _asset_subject(path) in TASK_MICE
        expected_trimodal = _is_trimodal_task_path(path)
        if bool(item.get("task_animal")) != expected_task:
            raise DANDIICMSError(f"incorrect task-animal flag for {path}")
        if bool(item.get("trimodal")) != expected_trimodal:
            raise DANDIICMSError(f"incorrect trimodal flag for {path}")
        paths.add(path)
        ids.add(asset_id)
        total += size
        if expected_task:
            task_counts[_asset_subject(path)] += 1
            task_total += size
        if expected_trimodal:
            trimodal_counts[_asset_subject(path)] += 1
            trimodal_total += size

    if total != EXPECTED_TOTAL_BYTES or task_total != EXPECTED_TASK_ASSET_BYTES:
        raise DANDIICMSError("manifest byte totals do not match asset rows")
    if trimodal_total != EXPECTED_TRIMODAL_ASSET_BYTES:
        raise DANDIICMSError("manifest trimodal byte total does not match asset rows")
    if task_counts != {
        "ICMS83": 9,
        "ICMS92": 10,
        "ICMS93": 10,
        "ICMS98": 10,
        "ICMS100": 7,
        "ICMS101": 9,
    }:
        raise DANDIICMSError(f"task-session counts changed: {task_counts}")
    if trimodal_counts != {
        "ICMS83": 9,
        "ICMS92": 9,
        "ICMS93": 10,
        "ICMS98": 7,
        "ICMS100": 4,
        "ICMS101": 6,
    }:
        raise DANDIICMSError(f"trimodal-session counts changed: {trimodal_counts}")


def load_frozen_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    """Load and validate the committed immutable release manifest."""

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_frozen_manifest(manifest)
    return manifest


def manifest_assets(
    manifest: Mapping[str, Any],
    *,
    scope: Literal["all", "task", "trimodal"] = "task",
) -> list[DANDIAsset]:
    """Return deterministic typed asset rows for a requested download scope."""

    validate_frozen_manifest(manifest)
    result = []
    for item in manifest["assets"]:
        if scope == "task" and not item["task_animal"]:
            continue
        if scope == "trimodal" and not item["trimodal"]:
            continue
        if scope not in {"all", "task", "trimodal"}:
            raise ValueError(f"unknown asset scope: {scope}")
        result.append(DANDIAsset(**item))
    return result


def hash_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_asset(path: str | Path, asset: DANDIAsset) -> dict[str, Any]:
    """Verify both byte length and the DANDI-published SHA-256 digest."""

    local = Path(path)
    observed_size = local.stat().st_size
    if observed_size != asset.size:
        raise DANDIICMSError(
            f"size mismatch for {local}: expected {asset.size}, observed {observed_size}"
        )
    observed_sha256 = hash_file(local)
    if observed_sha256 != asset.sha256:
        raise DANDIICMSError(
            f"SHA-256 mismatch for {local}: expected {asset.sha256}, observed {observed_sha256}"
        )
    return {
        "path": str(local),
        "size": observed_size,
        "sha256": observed_sha256,
        "verified": True,
    }


def _download_one_asset(
    asset: DANDIAsset,
    destination_root: Path,
    *,
    overwrite: bool,
    retries: int,
) -> dict[str, Any]:
    output = destination_root / asset.path
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        return {**verify_asset(output, asset), "status": "existing"}

    partial = output.with_name(f".{output.name}.partial")
    if overwrite:
        partial.unlink(missing_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "cadence-neuro-dandi-001868/1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(asset.download_url, headers=headers)
            with urllib.request.urlopen(request, timeout=180) as response:
                append = offset > 0 and getattr(response, "status", None) == 206
                mode = "ab" if append else "wb"
                with partial.open(mode) as stream:
                    while chunk := response.read(8 * 1024 * 1024):
                        stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
            result = verify_asset(partial, asset)
            os.replace(partial, output)
            result["path"] = str(output)
            result["status"] = "downloaded"
            return result
        except (OSError, urllib.error.URLError, DANDIICMSError) as error:
            last_error = error
            if partial.exists() and partial.stat().st_size > asset.size:
                partial.unlink()
            if attempt == retries:
                break
    raise DANDIICMSError(f"failed to download {asset.path}: {last_error}") from last_error


def download_release(
    manifest: Mapping[str, Any] | str | Path,
    destination_root: str | Path = DEFAULT_RAW_ROOT,
    *,
    scope: Literal["all", "task", "trimodal"] = "task",
    workers: int = 4,
    overwrite: bool = False,
    retries: int = 2,
) -> pd.DataFrame:
    """Download and verify a frozen release scope, atomically per asset."""

    if not isinstance(manifest, Mapping):
        manifest = load_frozen_manifest(manifest)
    assets = manifest_assets(manifest, scope=scope)
    output_root = Path(destination_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _download_one_asset,
                asset,
                output_root,
                overwrite=overwrite,
                retries=retries,
            ): asset
            for asset in assets
        }
        for future in as_completed(futures):
            asset = futures[future]
            rows.append(
                {
                    "asset_id": asset.asset_id,
                    "asset_path": asset.path,
                    "animal_id": asset.animal_id,
                    **future.result(),
                }
            )

    frame = pd.DataFrame(rows).sort_values("asset_path", kind="stable").reset_index(drop=True)
    selected_bytes = int(sum(asset.size for asset in assets))
    if int(frame["size"].sum()) != selected_bytes or not frame["verified"].all():
        raise DANDIICMSError("download completion audit failed")
    provenance = {
        "dandiset_id": DANDISET_ID,
        "version": DANDISET_VERSION,
        "scope": scope,
        "asset_count": len(assets),
        "total_bytes": selected_bytes,
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "assets": frame.to_dict(orient="records"),
    }
    write_json_atomic(provenance, output_root / "_download_provenance.json")
    return frame


def audit_frozen_manifest_against_api(
    frozen: Mapping[str, Any],
    *,
    workers: int = 12,
) -> None:
    """Re-query the published API and require byte-for-byte manifest equality."""

    current = build_published_manifest(workers=workers)
    if current != dict(frozen):
        raise DANDIICMSError(
            "the current published DANDI API response differs from the frozen manifest"
        )


def _decode(value: Any) -> Any:
    if isinstance(value, bytes | np.bytes_):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind in {"S", "O"}:
        return np.asarray([_decode(item) for item in value])
    return value


def _read_scalar(file: h5py.File, path: str) -> Any:
    if path not in file:
        raise DANDIICMSError(f"NWB is missing {path}")
    return _decode(file[path][()])


def _read_dynamic_table(group: h5py.Group) -> pd.DataFrame:
    columns: dict[str, Any] = {}
    expected_length: int | None = None
    for name, value in group.items():
        if not isinstance(value, h5py.Dataset) or value.ndim != 1:
            continue
        array = _decode(value[()])
        if expected_length is None:
            expected_length = len(array)
        if len(array) != expected_length:
            raise DANDIICMSError(f"ragged unsupported column {group.name}/{name}")
        columns[name] = array
    frame = pd.DataFrame(columns)
    if "id" in frame:
        frame = frame.rename(columns={"id": "row_id"})
    return frame


def _require_columns(frame: pd.DataFrame, required: set[str], table_name: str) -> None:
    missing = sorted(required - set(frame))
    if missing:
        raise DANDIICMSError(f"{table_name} is missing columns: {missing}")


def _series_times(series: h5py.Group, length: int) -> tuple[np.ndarray, float | None]:
    if "timestamps" in series:
        timestamps = np.asarray(series["timestamps"], dtype=np.float64)
        source_rate = None
    elif "starting_time" in series:
        starting = series["starting_time"]
        if "rate" not in starting.attrs:
            raise DANDIICMSError(f"{series.name} has no timestamp rate")
        source_rate = float(starting.attrs["rate"])
        timestamps = float(starting[()]) + np.arange(length, dtype=np.float64) / source_rate
    else:
        raise DANDIICMSError(f"{series.name} has neither timestamps nor starting_time")
    if timestamps.shape != (length,):
        raise DANDIICMSError(f"{series.name} timestamp length does not match data")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise DANDIICMSError(f"{series.name} timestamps are not strictly increasing")
    return timestamps, source_rate


def _event_overlaps_window(
    event_start: np.ndarray,
    event_stop: np.ndarray,
    window_start: float,
    window_stop: float,
) -> bool:
    return bool(np.any((event_start < window_stop) & (event_stop > window_start)))


def classify_trials_and_events(
    trials: pd.DataFrame,
    events: pd.DataFrame,
    *,
    config: ICMSPreprocessConfig = DEFAULT_PREPROCESS_CONFIG,
) -> pd.DataFrame:
    """Join ICMS trains to trials and seal uncontaminated catch calibration."""

    _require_columns(trials, _TRIAL_REQUIRED_COLUMNS, "trials")
    _require_columns(events, _EVENT_REQUIRED_COLUMNS, "electrical_stimulation")
    result = trials.copy().reset_index(drop=True)
    for name in ("start_time", "stop_time", "current_uA", "response_time"):
        result[name] = pd.to_numeric(result[name], errors="coerce")
    result["trial_index"] = pd.to_numeric(result["trial_index"], errors="raise").astype("int64")
    result["stim_channel"] = pd.to_numeric(result["stim_channel"], errors="raise").astype("int64")
    if result["trial_index"].duplicated().any():
        raise DANDIICMSError("trial_index is not unique")
    if result["current_uA"].isna().any():
        raise DANDIICMSError("trial current_uA contains missing values")

    event_frame = events.copy().reset_index(drop=True)
    for name in (
        "start_time",
        "stop_time",
        "current_uA",
        "frequency_hz",
        "pulse_count",
        "pulse_width_us",
    ):
        event_frame[name] = pd.to_numeric(event_frame[name], errors="raise")
    event_frame["trial_index"] = pd.to_numeric(event_frame["trial_index"], errors="raise").astype(
        "int64"
    )
    event_frame["stim_channel"] = pd.to_numeric(event_frame["stim_channel"], errors="raise").astype(
        "int64"
    )
    if event_frame["trial_index"].duplicated().any():
        repeated = sorted(
            event_frame.loc[
                event_frame["trial_index"].duplicated(keep=False), "trial_index"
            ].unique()
        )
        raise DANDIICMSError(f"multiple pulse-train events for trials {repeated}")
    if (event_frame["current_uA"] <= 0).any():
        raise DANDIICMSError("electrical_stimulation table contains non-positive current")
    if (event_frame["stop_time"] <= event_frame["start_time"]).any():
        raise DANDIICMSError("electrical_stimulation contains a non-positive interval")

    indexed_events = event_frame.set_index("trial_index", drop=False)
    stimulated_delays: list[float] = []
    event_columns = {
        "event_start_time": np.full(len(result), np.nan),
        "event_stop_time": np.full(len(result), np.nan),
        "frequency_hz": np.zeros(len(result)),
        "pulse_count": np.zeros(len(result)),
        "pulse_width_us": np.zeros(len(result)),
    }
    for row_position, trial in result.iterrows():
        trial_index = int(trial["trial_index"])
        current = float(trial["current_uA"])
        has_event = trial_index in indexed_events.index
        if current == 0.0:
            if has_event:
                raise DANDIICMSError(
                    f"catch trial {trial_index} has a stimulation event with the same index"
                )
            continue
        if current < 0:
            raise DANDIICMSError(f"trial {trial_index} has negative current")
        if not has_event:
            raise DANDIICMSError(f"stimulated trial {trial_index} has no event")
        event = indexed_events.loc[trial_index]
        if not np.isclose(float(event["current_uA"]), current, atol=1e-6, rtol=0):
            raise DANDIICMSError(f"current mismatch for trial {trial_index}")
        if int(event["stim_channel"]) != int(trial["stim_channel"]):
            raise DANDIICMSError(f"stimulation-channel mismatch for trial {trial_index}")
        delay = float(event["start_time"]) - float(trial["start_time"])
        if not 0 <= delay <= config.event_onset_tolerance_s:
            raise DANDIICMSError(
                f"event onset for trial {trial_index} is {delay:.6f}s after trial start"
            )
        stimulated_delays.append(delay)
        event_columns["event_start_time"][row_position] = float(event["start_time"])
        event_columns["event_stop_time"][row_position] = float(event["stop_time"])
        event_columns["frequency_hz"][row_position] = float(event["frequency_hz"])
        event_columns["pulse_count"][row_position] = float(event["pulse_count"])
        event_columns["pulse_width_us"][row_position] = float(event["pulse_width_us"])

    orphan_events = sorted(set(event_frame["trial_index"]) - set(result["trial_index"]))
    if orphan_events:
        raise DANDIICMSError(f"events refer to absent trials: {orphan_events[:10]}")
    cue_delay = float(np.median(stimulated_delays)) if stimulated_delays else 0.0
    for name, values in event_columns.items():
        result[name] = values
    result["anchor_time"] = np.where(
        result["current_uA"].eq(0.0),
        result["start_time"] + cue_delay,
        result["event_start_time"],
    )
    result["anchor_source"] = np.where(
        result["current_uA"].eq(0.0),
        "catch_pseudo_onset",
        "electrical_stimulation",
    )
    result["is_catch"] = result["current_uA"].eq(0.0)

    event_start = event_frame["start_time"].to_numpy(dtype=np.float64)
    event_stop = event_frame["stop_time"].to_numpy(dtype=np.float64)
    clean_normal = np.zeros(len(result), dtype=bool)
    overlap_count = np.zeros(len(result), dtype=np.int64)
    for index, trial in result.iterrows():
        window_start = float(trial["anchor_time"]) + config.window_start_s
        window_stop = float(trial["anchor_time"]) + config.window_stop_s
        overlap = (event_start < window_stop) & (event_stop > window_start)
        overlap_count[index] = int(overlap.sum())
        clean_normal[index] = bool(trial["is_catch"]) and not overlap.any()
    result["overlapping_stimulation_events"] = overlap_count
    result["is_normal_calibration"] = clean_normal
    result["is_iti_calibration"] = False
    result["normal_source"] = np.where(clean_normal, "catch", "none")
    result["trial_record_type"] = "task_trial"
    result["window_kind"] = np.select(
        [
            result["is_normal_calibration"],
            result["current_uA"].gt(0.0),
        ],
        ["normal", "intervention"],
        default="excluded_catch_overlap",
    )
    return result


def _merged_blocked_intervals(
    trials: pd.DataFrame,
    events: pd.DataFrame,
    *,
    support_start: float,
    support_stop: float,
    guard_s: float,
) -> list[tuple[float, float]]:
    intervals = []
    for frame in (trials, events):
        if not {"start_time", "stop_time"}.issubset(frame):
            raise DANDIICMSError("interval table lacks start_time or stop_time")
        for start, stop in frame[["start_time", "stop_time"]].itertuples(index=False, name=None):
            start = float(start)
            stop = float(stop)
            if not np.isfinite([start, stop]).all() or stop <= start:
                raise DANDIICMSError("trial/event table contains an invalid interval")
            expanded_start = max(support_start, start - guard_s)
            expanded_stop = min(support_stop, stop + guard_s)
            if expanded_stop > expanded_start:
                intervals.append((expanded_start, expanded_stop))
    intervals.sort()
    merged: list[tuple[float, float]] = []
    for start, stop in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, stop))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
    return merged


def select_iti_calibration_windows(
    trials: pd.DataFrame,
    events: pd.DataFrame,
    *,
    support_start: float,
    support_stop: float,
    config: ICMSPreprocessConfig = DEFAULT_PREPROCESS_CONFIG,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select deterministic, full-length normal windows wholly inside ITIs.

    Trial and ICMS-event intervals are unioned, expanded by the prespecified
    guard, and complemented inside the continuous wheel support.  Only gaps
    *between* blocked intervals count as ITIs; session-leading/trailing margins
    are not used.  One centered candidate is made per eligible ITI, then at
    most ``iti_windows_per_session`` candidates are selected at evenly spaced
    ordinal positions.  No signal value or intervention response affects the
    choice.
    """

    if not np.isfinite([support_start, support_stop]).all() or support_stop <= support_start:
        raise DANDIICMSError("continuous-signal support is invalid")
    duration = config.window_stop_s - config.window_start_s
    blocked = _merged_blocked_intervals(
        trials,
        events,
        support_start=support_start,
        support_stop=support_stop,
        guard_s=config.iti_guard_s,
    )
    candidates: list[dict[str, Any]] = []
    for gap_index, (left, right) in enumerate(zip(blocked[:-1], blocked[1:], strict=True)):
        gap_start = left[1]
        gap_stop = right[0]
        if gap_stop - gap_start + 1e-12 < duration:
            continue
        window_start = 0.5 * (gap_start + gap_stop - duration)
        window_stop = window_start + duration
        candidates.append(
            {
                "iti_gap_index": gap_index,
                "iti_gap_start_time": gap_start,
                "iti_gap_stop_time": gap_stop,
                "start_time": window_start,
                "stop_time": window_stop,
                "anchor_time": window_start - config.window_start_s,
            }
        )

    maximum = config.iti_windows_per_session
    if maximum == 0 or not candidates:
        selected_candidates: list[dict[str, Any]] = []
    elif len(candidates) <= maximum:
        selected_candidates = candidates
    else:
        selected_indices = np.linspace(0, len(candidates) - 1, maximum, dtype=np.int64)
        if len(np.unique(selected_indices)) != maximum:
            raise AssertionError("deterministic ITI subsampling produced duplicate indices")
        selected_candidates = [candidates[int(index)] for index in selected_indices]

    rows = []
    for selected_index, candidate in enumerate(selected_candidates):
        rows.append(
            {
                "row_id": np.nan,
                "start_time": candidate["start_time"],
                "stop_time": candidate["stop_time"],
                "trial_index": -(selected_index + 1),
                "current_uA": 0.0,
                "stim_channel": 0,
                "is_hit": False,
                "response_time": np.nan,
                "is_good_trial": True,
                "event_start_time": np.nan,
                "event_stop_time": np.nan,
                "frequency_hz": 0.0,
                "pulse_count": 0.0,
                "pulse_width_us": 0.0,
                "anchor_time": candidate["anchor_time"],
                "anchor_source": "iti_center",
                "is_catch": False,
                "overlapping_stimulation_events": 0,
                "is_normal_calibration": True,
                "is_iti_calibration": True,
                "normal_source": "iti",
                "trial_record_type": "iti_window",
                "window_kind": "normal",
                "iti_gap_index": candidate["iti_gap_index"],
                "iti_gap_start_time": candidate["iti_gap_start_time"],
                "iti_gap_stop_time": candidate["iti_gap_stop_time"],
            }
        )
    frame = pd.DataFrame(rows)
    audit = {
        "support_start_s": float(support_start),
        "support_stop_s": float(support_stop),
        "blocked_interval_count": len(blocked),
        "candidate_iti_count": len(candidates),
        "selected_iti_count": len(rows),
        "iti_guard_s": config.iti_guard_s,
        "window_duration_s": duration,
        "maximum_windows": maximum,
        "selection": "one_centered_per_eligible_iti_then_evenly_spaced_ordinals",
        "uses_signal_values": False,
        "uses_intervention_pre_onset_segments": False,
    }
    return frame, audit


def assert_iti_windows_are_isolated(
    iti_windows: pd.DataFrame,
    trials: pd.DataFrame,
    events: pd.DataFrame,
    *,
    guard_s: float,
) -> None:
    """Prove that selected ITI windows do not touch any guarded trial/event."""

    if iti_windows.empty:
        return
    if not iti_windows["is_iti_calibration"].astype(bool).all():
        raise DANDIICMSError("ITI audit received a non-ITI row")
    blocked_start = np.concatenate(
        (
            trials["start_time"].to_numpy(dtype=np.float64),
            events["start_time"].to_numpy(dtype=np.float64),
        )
    )
    blocked_stop = np.concatenate(
        (
            trials["stop_time"].to_numpy(dtype=np.float64),
            events["stop_time"].to_numpy(dtype=np.float64),
        )
    )
    for window in iti_windows.itertuples(index=False):
        overlap = (blocked_start - guard_s < float(window.stop_time)) & (
            blocked_stop + guard_s > float(window.start_time)
        )
        if overlap.any():
            raise DANDIICMSError("ITI calibration window overlaps a guarded trial/event")


def assert_normal_calibration_sealed(trials: pd.DataFrame) -> None:
    """Audit a selected calibration frame immediately before adaptation."""

    required = {
        "current_uA",
        "is_catch",
        "is_iti_calibration",
        "normal_source",
        "is_normal_calibration",
        "overlapping_stimulation_events",
        "window_kind",
    }
    _require_columns(trials, required, "calibration trials")
    if trials.empty:
        raise DANDIICMSError("calibration trials are empty")
    if not trials["current_uA"].eq(0.0).all():
        raise DANDIICMSError("calibration contains non-zero current")
    if not trials["is_normal_calibration"].astype(bool).all():
        raise DANDIICMSError("calibration contains a rejected catch/ITI window")
    sources = set(trials["normal_source"].astype(str))
    if not sources.issubset({"catch", "iti"}):
        raise DANDIICMSError(f"calibration contains an invalid normal source: {sources}")
    catch = trials["normal_source"].astype(str).eq("catch")
    iti = trials["normal_source"].astype(str).eq("iti")
    if not trials.loc[catch, "is_catch"].astype(bool).all():
        raise DANDIICMSError("catch calibration contains a non-catch row")
    if not trials.loc[iti, "is_iti_calibration"].astype(bool).all():
        raise DANDIICMSError("ITI calibration contains a non-ITI row")
    if not trials["overlapping_stimulation_events"].eq(0).all():
        raise DANDIICMSError("calibration overlaps stimulation")
    if not trials["window_kind"].astype(str).eq("normal").all():
        raise DANDIICMSError("calibration window_kind is not normal")


def _electrode_geometry(
    file: h5py.File,
    animal_id: str,
) -> dict[int, tuple[float, float, float]]:
    path = "/general/extracellular_ephys/electrodes"
    if path not in file:
        raise DANDIICMSError("NWB has no electrodes table")
    table = _read_dynamic_table(file[path])
    required = {"channel_name", "rel_x", "rel_y", "rel_z"}
    _require_columns(table, required, "electrodes")
    result: dict[int, tuple[float, float, float]] = {}
    for row in table.itertuples(index=False):
        channel = int(str(row.channel_name))
        coordinate = (float(row.rel_x), float(row.rel_y), float(row.rel_z))
        if channel in result:
            raise DANDIICMSError(f"electrodes table repeats Ripple channel {channel}")
        if not 1 <= channel <= 32:
            raise DANDIICMSError(f"unexpected NET32 channel {channel}")
        expected_y = TASK_ANIMAL_CHANNEL_REL_Y_UM.get(animal_id, {}).get(channel)
        on_physical_grid = bool(
            np.any(np.isclose(coordinate[1], _NET32_ALLOWED_DEPTHS_UM, atol=1e-6))
        )
        if (
            not np.isclose(coordinate[0], 0.0, atol=1e-6)
            or not np.isclose(coordinate[2], 0.0, atol=1e-6)
            or not on_physical_grid
            or (expected_y is not None and not np.isclose(coordinate[1], expected_y, atol=1e-6))
        ):
            raise DANDIICMSError(
                f"NWB coordinate {coordinate} disagrees with frozen NET32 "
                f"geometry for {animal_id} channel {channel}"
            )
        result[channel] = coordinate
    return result


def intervention_descriptors(
    trial_metadata: pd.DataFrame,
    electrode_geometry: Mapping[int, tuple[float, float, float]],
    *,
    fallback_channel_depth_um: Mapping[int, float] | None = None,
) -> np.ndarray:
    """Encode pulse physics and coordinates, never raw channel identity."""

    rows = np.zeros(
        (len(trial_metadata), len(INTERVENTION_DESCRIPTOR_COLUMNS)),
        dtype=np.float32,
    )
    column = {name: index for index, name in enumerate(INTERVENTION_DESCRIPTOR_COLUMNS)}
    for index, trial in trial_metadata.reset_index(drop=True).iterrows():
        current = float(trial["current_uA"])
        if current == 0.0:
            continue
        channel = int(trial["stim_channel"])
        coordinate = electrode_geometry.get(channel)
        if coordinate is None and fallback_channel_depth_um is not None:
            depth = fallback_channel_depth_um.get(channel)
            coordinate = None if depth is None else (0.0, float(depth), 0.0)
        if coordinate is None:
            coordinate = (np.nan, np.nan, np.nan)
        if not np.isfinite(coordinate).all():
            raise DANDIICMSError(
                f"no physical coordinate is available for stimulation channel {channel}"
            )
        x_um, y_um, z_um = coordinate
        rows[index, column["stim_present"]] = 1.0
        rows[index, column["current_uA"]] = current
        rows[index, column["frequency_hz"]] = float(trial["frequency_hz"])
        rows[index, column["pulse_count"]] = float(trial["pulse_count"])
        rows[index, column["pulse_width_us"]] = float(trial["pulse_width_us"])
        rows[index, column["electrode_rel_x_um"]] = x_um
        rows[index, column["electrode_rel_y_um"]] = y_um
        rows[index, column["electrode_rel_z_um"]] = z_um
        rows[index, column["electrode_depth_centered_um"]] = y_um - NET32_DEPTH_CENTER_UM
        rows[index, column["electrode_depth_fraction"]] = (
            y_um - NET32_DEPTH_CENTER_UM
        ) / NET32_DEPTH_HALF_RANGE_UM
    return rows


def _select_dff_series(file: h5py.File) -> h5py.Group:
    path = "/processing/ophys/DfOverF"
    if path not in file:
        raise DANDIICMSError("NWB has no DfOverF processing interface")
    candidates = [
        value for value in file[path].values() if isinstance(value, h5py.Group) and "data" in value
    ]
    if not candidates:
        raise DANDIICMSError("DfOverF processing interface contains no ROI response series")
    candidates.sort(
        key=lambda value: (
            "Volumetric" not in value.name,
            -int(value["data"].shape[-1]),
            value.name,
        )
    )
    return candidates[0]


def _interpolate_columns(
    source_time: np.ndarray,
    source_values: np.ndarray,
    target_time: np.ndarray,
    *,
    max_gap_s: float,
    observed_tolerance_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear interpolation with explicit support and direct-observation masks."""

    source_time = np.asarray(source_time, dtype=np.float64)
    values = np.asarray(source_values)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != len(source_time):
        raise DANDIICMSError("source values are not time by feature")
    targets = np.asarray(target_time, dtype=np.float64)
    flat_targets = targets.reshape(-1)
    output = np.full((len(flat_targets), values.shape[1]), np.nan, dtype=np.float32)
    valid = np.zeros(output.shape, dtype=bool)
    observed = np.zeros(output.shape, dtype=bool)
    finite_time = np.isfinite(source_time)
    for feature in range(values.shape[1]):
        finite = finite_time & np.isfinite(values[:, feature])
        source_x = source_time[finite]
        source_y = np.asarray(values[finite, feature], dtype=np.float64)
        if len(source_x) == 0:
            continue
        right = np.searchsorted(source_x, flat_targets, side="left")
        right_clipped = np.clip(right, 0, len(source_x) - 1)
        left_clipped = np.clip(right - 1, 0, len(source_x) - 1)
        nearest_distance = np.minimum(
            np.abs(flat_targets - source_x[left_clipped]),
            np.abs(flat_targets - source_x[right_clipped]),
        )
        directly_observed = nearest_distance <= observed_tolerance_s
        if len(source_x) == 1:
            supported = directly_observed
            interpolated = np.full(len(flat_targets), source_y[0], dtype=np.float64)
        else:
            interior = (right > 0) & (right < len(source_x))
            gap = source_x[right_clipped] - source_x[left_clipped]
            supported = directly_observed | (interior & (gap <= max_gap_s))
            interpolated = np.interp(flat_targets, source_x, source_y)
        interpolated[~supported] = np.nan
        output[:, feature] = interpolated.astype(np.float32)
        valid[:, feature] = supported
        observed[:, feature] = directly_observed & supported
    shape = (*targets.shape, values.shape[1])
    return output.reshape(shape), valid.reshape(shape), observed.reshape(shape)


def _roi_metadata(file: h5py.File, series: h5py.Group) -> pd.DataFrame:
    roi_ids = np.asarray(series["rois"], dtype=np.int64)
    result = pd.DataFrame({"roi_id": roi_ids})
    if "table" not in series["rois"].attrs:
        return result
    table = file[series["rois"].attrs["table"]]
    if not isinstance(table, h5py.Group) or "id" not in table:
        return result
    table_ids = np.asarray(table["id"], dtype=np.int64)
    id_to_row = {int(identifier): row for row, identifier in enumerate(table_ids)}
    if not set(map(int, roi_ids)).issubset(id_to_row):
        raise DANDIICMSError("DfOverF ROI region refers to absent segmentation rows")
    if "voxel_mask" not in table or "voxel_mask_index" not in table:
        return result
    voxel_mask = np.asarray(table["voxel_mask"])
    stops = np.asarray(table["voxel_mask_index"], dtype=np.int64)
    starts = np.concatenate(([0], stops[:-1]))
    centroids: dict[int, tuple[float, float, float]] = {}
    for row, identifier in enumerate(table_ids):
        voxels = voxel_mask[starts[row] : stops[row]]
        if len(voxels) == 0:
            centroids[int(identifier)] = (np.nan, np.nan, np.nan)
            continue
        weights = np.asarray(voxels["weight"], dtype=np.float64)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = np.ones(len(voxels), dtype=np.float64)
        centroids[int(identifier)] = tuple(
            float(np.average(np.asarray(voxels[axis], dtype=np.float64), weights=weights))
            for axis in ("x", "y", "z")
        )
    xyz = np.asarray([centroids[int(identifier)] for identifier in roi_ids])
    result[["centroid_x_px", "centroid_y_px", "centroid_z_index"]] = xyz
    return result


def _unit_spike_times(file: h5py.File) -> tuple[list[np.ndarray], pd.DataFrame]:
    if "/units" not in file:
        raise DANDIICMSError("NWB has no sorted units table")
    units = file["/units"]
    required = {"id", "spike_times", "spike_times_index"}
    missing = sorted(required - set(units))
    if missing:
        raise DANDIICMSError(f"units table is missing datasets: {missing}")
    ids = np.asarray(units["id"], dtype=np.int64)
    accepted = (
        np.asarray(units["accepted"], dtype=bool)
        if "accepted" in units
        else np.ones(len(ids), dtype=bool)
    )
    stops = np.asarray(units["spike_times_index"], dtype=np.int64)
    starts = np.concatenate(([0], stops[:-1]))
    all_spikes = np.asarray(units["spike_times"], dtype=np.float64)
    spikes = [
        all_spikes[starts[index] : stops[index]] for index in range(len(ids)) if accepted[index]
    ]
    metadata = pd.DataFrame(
        {
            "unit_id": ids[accepted],
            "accepted": np.ones(int(accepted.sum()), dtype=bool),
        }
    )
    optional = {
        "cell_type": object,
        "peak_channel_index": np.int64,
        "unit_x_um": np.float64,
        "unit_y_um": np.float64,
    }
    for name, dtype in optional.items():
        if name in units:
            values = _decode(units[name][()])[accepted]
            metadata[name] = np.asarray(values, dtype=dtype)
    return spikes, metadata


def _bin_spikes(
    spike_times: Sequence[np.ndarray],
    anchors: np.ndarray,
    relative_edges: np.ndarray,
    stimulated: np.ndarray,
    *,
    config: ICMSPreprocessConfig,
) -> tuple[np.ndarray, np.ndarray]:
    absolute_edges = anchors[:, None] + relative_edges[None, :]
    rates = np.zeros(
        (len(anchors), len(relative_edges) - 1, len(spike_times)),
        dtype=np.float32,
    )
    for unit_index, times in enumerate(spike_times):
        indices = np.searchsorted(times, absolute_edges, side="left")
        rates[:, :, unit_index] = np.diff(indices, axis=1) * config.sample_rate_hz
    bin_start = relative_edges[:-1]
    bin_stop = relative_edges[1:]
    artifact_bins = (bin_start < config.ephys_artifact_stop_s) & (
        bin_stop > config.ephys_artifact_start_s
    )
    valid = np.ones(rates.shape[:2], dtype=bool)
    valid[np.asarray(stimulated, dtype=bool)] &= ~artifact_bins
    rates[~valid, :] = np.nan
    return rates, valid


def load_icms_session(
    path: str | Path,
    *,
    config: ICMSPreprocessConfig = DEFAULT_PREPROCESS_CONFIG,
    expected_animal: str | None = None,
    source_asset_path: str | None = None,
    source_sha256: str | None = None,
) -> ICMSSession:
    """Load one trimodal NWB into a common event-aligned representation."""

    source = Path(path)
    with h5py.File(source, "r") as file:
        animal_id = str(_read_scalar(file, "/general/subject/subject_id"))
        if animal_id not in TASK_MICE:
            raise DANDIICMSError(f"{animal_id!r} is not a task animal")
        if expected_animal is not None and animal_id != expected_animal:
            raise DANDIICMSError(
                f"animal isolation failure: expected {expected_animal}, observed {animal_id}"
            )
        session_start = str(_read_scalar(file, "/session_start_time"))
        match = re.search(r"_ses-(\d{4}-\d{2}-\d{2})_", source.name)
        session_id = match.group(1) if match else session_start[:10]

        if "/intervals/trials" not in file or "/intervals/electrical_stimulation" not in file:
            raise DANDIICMSError("NWB lacks trial or electrical-stimulation intervals")
        trials = _read_dynamic_table(file["/intervals/trials"])
        events = _read_dynamic_table(file["/intervals/electrical_stimulation"])
        task_trials = classify_trials_and_events(trials, events, config=config)
        if config.good_trials_only:
            task_trials = task_trials.loc[task_trials["is_good_trial"].astype(bool)].copy()
        task_trials = task_trials.reset_index(drop=True)
        if task_trials.empty:
            raise DANDIICMSError("session has no trials after quality filtering")

        dff_series = _select_dff_series(file)
        dff_dataset = dff_series["data"]
        dff_data = np.asarray(dff_dataset, dtype=np.float32)
        dff_time, dff_rate = _series_times(dff_series, dff_data.shape[0])
        if dff_data.ndim != 2:
            raise DANDIICMSError("DfOverF data is not time by ROI")
        roi_metadata = _roi_metadata(file, dff_series)

        wheel_path = "/processing/behavior/wheel/wheel_position_processed"
        if wheel_path not in file:
            raise DANDIICMSError("NWB has no processed wheel-position series")
        wheel_series = file[wheel_path]
        wheel_data = np.asarray(wheel_series["data"], dtype=np.float32)
        if wheel_data.ndim != 1:
            raise DANDIICMSError("processed wheel position is not one-dimensional")
        wheel_time, wheel_rate = _series_times(wheel_series, len(wheel_data))

        if config.include_iti_calibration:
            iti_trials, iti_audit = select_iti_calibration_windows(
                trials,
                events,
                support_start=float(wheel_time[0]),
                support_stop=float(wheel_time[-1]),
                config=config,
            )
            assert_iti_windows_are_isolated(
                iti_trials,
                trials,
                events,
                guard_s=config.iti_guard_s,
            )
        else:
            iti_trials = pd.DataFrame()
            iti_audit = {
                "selected_iti_count": 0,
                "selection": "disabled",
                "uses_signal_values": False,
                "uses_intervention_pre_onset_segments": False,
            }
        classified = pd.concat([task_trials, iti_trials], ignore_index=True, sort=False)
        classified = classified.sort_values(
            ["anchor_time", "trial_record_type", "trial_index"],
            kind="stable",
        ).reset_index(drop=True)

        anchors = classified["anchor_time"].to_numpy(dtype=np.float64)
        relative_time = config.relative_time_s
        absolute_time = anchors[:, None] + relative_time[None, :]
        geometry = _electrode_geometry(file, animal_id)
        descriptors = intervention_descriptors(
            classified,
            geometry,
            fallback_channel_depth_um=TASK_ANIMAL_CHANNEL_REL_Y_UM.get(animal_id),
        )

        observation_tolerance = (
            0.55 / dff_rate if dff_rate is not None else 0.55 * float(np.median(np.diff(dff_time)))
        )
        calcium, calcium_valid, calcium_observed = _interpolate_columns(
            dff_time,
            dff_data,
            absolute_time,
            max_gap_s=config.calcium_max_gap_s,
            observed_tolerance_s=observation_tolerance,
        )
        wheel_tolerance = (
            0.55 / wheel_rate
            if wheel_rate is not None
            else 0.55 * float(np.median(np.diff(wheel_time)))
        )
        wheel_3d, wheel_valid_3d, _ = _interpolate_columns(
            wheel_time,
            wheel_data,
            absolute_time,
            max_gap_s=config.wheel_max_gap_s,
            observed_tolerance_s=wheel_tolerance,
        )
        wheel_position = wheel_3d[:, :, 0]
        wheel_valid = wheel_valid_3d[:, :, 0]
        zero_index = int(np.argmin(np.abs(relative_time)))
        wheel_displacement = wheel_position - wheel_position[:, [zero_index]]
        wheel_velocity = np.gradient(
            wheel_position.astype(np.float64),
            1.0 / config.sample_rate_hz,
            axis=1,
        ).astype(np.float32)
        wheel_position[~wheel_valid] = np.nan
        wheel_displacement[~wheel_valid] = np.nan
        wheel_velocity[~wheel_valid] = np.nan

        spikes, unit_metadata = _unit_spike_times(file)
        spike_rate, spike_valid = _bin_spikes(
            spikes,
            anchors,
            config.relative_edges_s,
            classified["current_uA"].gt(0.0).to_numpy(),
            config=config,
        )

    normal = classified.loc[classified["is_normal_calibration"]].copy()
    if not normal.empty:
        assert_normal_calibration_sealed(normal)
    iti_mask = classified["is_iti_calibration"].astype(bool).to_numpy()
    catch_normal = classified["is_catch"].astype(bool) & classified["is_normal_calibration"].astype(
        bool
    )
    iti_total_spikes = (
        float(np.nansum(spike_rate[iti_mask]) / config.sample_rate_hz) if iti_mask.any() else 0.0
    )
    audit = {
        "n_trials_raw": int(len(trials)),
        "n_task_trials_output": int(len(task_trials)),
        "n_trials_output": int(len(classified)),
        "n_stimulation_trials": int(classified["current_uA"].gt(0.0).sum()),
        "n_catch_trials": int(classified["is_catch"].sum()),
        "n_catch_normal_calibration_trials": int(catch_normal.sum()),
        "n_iti_calibration_windows": int(iti_mask.sum()),
        "n_normal_calibration_trials": int(classified["is_normal_calibration"].sum()),
        "n_excluded_catch_overlap": int(
            classified["window_kind"].eq("excluded_catch_overlap").sum()
        ),
        "n_rois": int(calcium.shape[2]),
        "n_units": int(spike_rate.shape[2]),
        "calcium_valid_fraction": float(calcium_valid.mean()),
        "calcium_directly_observed_fraction": float(calcium_observed.mean()),
        "wheel_valid_fraction": float(wheel_valid.mean()),
        "spike_valid_fraction": float(spike_valid.mean()),
        "iti_calcium_valid_fraction": (
            float(calcium_valid[iti_mask].mean()) if iti_mask.any() else None
        ),
        "iti_wheel_valid_fraction": (
            float(wheel_valid[iti_mask].mean()) if iti_mask.any() else None
        ),
        "iti_spike_valid_fraction": (
            float(spike_valid[iti_mask].mean()) if iti_mask.any() else None
        ),
        "iti_total_spikes": iti_total_spikes,
        "iti_nonzero_spike_bin_fraction": (
            float(np.any(spike_rate[iti_mask] > 0, axis=2).mean())
            if iti_mask.any() and spike_rate.shape[2] > 0
            else None
        ),
        "iti_selection": iti_audit,
        "ephys_artifact_mask_s": [
            config.ephys_artifact_start_s,
            config.ephys_artifact_stop_s,
        ],
        "descriptor_columns": list(INTERVENTION_DESCRIPTOR_COLUMNS),
        "raw_channel_is_descriptor": False,
    }
    return ICMSSession(
        animal_id=animal_id,
        session_id=session_id,
        session_start_time=session_start,
        source_path=source_asset_path or str(source),
        source_sha256=source_sha256,
        time_s=relative_time.astype(np.float32),
        trial_metadata=classified,
        intervention_descriptors=descriptors,
        calcium_dff=calcium,
        calcium_valid_mask=calcium_valid,
        calcium_observed_mask=calcium_observed,
        spike_rate_hz=spike_rate,
        spike_valid_mask=spike_valid,
        wheel_position=wheel_position.astype(np.float32),
        wheel_displacement=wheel_displacement.astype(np.float32),
        wheel_velocity=wheel_velocity,
        wheel_valid_mask=wheel_valid,
        roi_metadata=roi_metadata,
        unit_metadata=unit_metadata,
        audit=audit,
    )


def _h5_write_array(
    group: h5py.Group,
    name: str,
    values: np.ndarray,
    *,
    compression_level: int,
) -> h5py.Dataset:
    array = np.asarray(values)
    kwargs: dict[str, Any] = {}
    if array.size and compression_level > 0:
        kwargs = {
            "compression": "gzip",
            "compression_opts": compression_level,
            "shuffle": array.dtype.kind not in {"O", "S", "U"},
        }
        if array.ndim == 3:
            kwargs["chunks"] = (1, array.shape[1], min(array.shape[2], 256))
        elif array.ndim == 2:
            kwargs["chunks"] = (min(array.shape[0], 256), array.shape[1])
    return group.create_dataset(name, data=array, **kwargs)


def _h5_write_frame(
    group: h5py.Group,
    frame: pd.DataFrame,
    *,
    compression_level: int,
) -> None:
    for column in frame:
        values = frame[column].to_numpy()
        if values.dtype.kind in {"O", "U"}:
            string_dtype = h5py.string_dtype(encoding="utf-8")
            group.create_dataset(
                column,
                data=np.asarray(
                    ["" if pd.isna(value) else str(value) for value in values],
                    dtype=object,
                ),
                dtype=string_dtype,
            )
        else:
            _h5_write_array(
                group,
                column,
                values,
                compression_level=compression_level,
            )


def write_animal_file(
    sessions: Sequence[ICMSSession],
    output: str | Path,
    *,
    config: ICMSPreprocessConfig = DEFAULT_PREPROCESS_CONFIG,
) -> tuple[Path, dict[str, Any]]:
    """Atomically write one multi-session HDF5 file for exactly one animal."""

    if not sessions:
        raise DANDIICMSError("cannot write an animal file without sessions")
    animals = {session.animal_id for session in sessions}
    if len(animals) != 1:
        raise DANDIICMSError(f"animal isolation failure: mixed sessions from {sorted(animals)}")
    animal_id = next(iter(animals))
    ordered = sorted(sessions, key=lambda item: (item.session_start_time, item.session_id))
    if len({session.session_id for session in ordered}) != len(ordered):
        raise DANDIICMSError("animal file contains duplicate session IDs")
    reference_time = ordered[0].time_s
    if any(not np.array_equal(reference_time, session.time_s) for session in ordered[1:]):
        raise DANDIICMSError("sessions do not share an identical time grid")

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with h5py.File(temporary, "w") as file:
            file.attrs["schema"] = "cadence-dandi-001868-v1"
            file.attrs["dandiset_id"] = DANDISET_ID
            file.attrs["dandiset_version"] = DANDISET_VERSION
            file.attrs["animal_id"] = animal_id
            file.attrs["split_unit"] = "animal_id"
            file.attrs["calibration_policy"] = (
                "current_uA == 0 and no overlapping electrical_stimulation"
            )
            file.attrs["raw_stim_channel_in_descriptor"] = False
            file.create_dataset(
                "descriptor_columns",
                data=np.asarray(INTERVENTION_DESCRIPTOR_COLUMNS, dtype=object),
                dtype=h5py.string_dtype("utf-8"),
            )
            file.create_dataset("time_s", data=reference_time)
            session_group = file.create_group("sessions")
            first_date = pd.Timestamp(ordered[0].session_start_time)
            for day_index, session in enumerate(ordered):
                key = f"day-{day_index:02d}_{session.session_id}"
                group = session_group.create_group(key)
                group.attrs["animal_id"] = animal_id
                group.attrs["session_id"] = session.session_id
                group.attrs["session_start_time"] = session.session_start_time
                group.attrs["session_day_index"] = day_index
                group.attrs["days_since_first_session"] = int(
                    (pd.Timestamp(session.session_start_time).date() - first_date.date()).days
                )
                group.attrs["source_asset_path"] = session.source_path
                if session.source_sha256:
                    group.attrs["source_sha256"] = session.source_sha256
                trials_group = group.create_group("trials")
                _h5_write_frame(
                    trials_group,
                    session.trial_metadata,
                    compression_level=config.hdf5_compression_level,
                )
                _h5_write_array(
                    group,
                    "intervention_descriptors",
                    session.intervention_descriptors,
                    compression_level=config.hdf5_compression_level,
                )
                signals = group.create_group("signals")
                for name, values in (
                    ("calcium_dff", session.calcium_dff),
                    ("calcium_valid_mask", session.calcium_valid_mask),
                    ("calcium_observed_mask", session.calcium_observed_mask),
                    ("spike_rate_hz", session.spike_rate_hz),
                    ("spike_valid_mask", session.spike_valid_mask),
                    ("wheel_position", session.wheel_position),
                    ("wheel_displacement", session.wheel_displacement),
                    ("wheel_velocity", session.wheel_velocity),
                    ("wheel_valid_mask", session.wheel_valid_mask),
                ):
                    _h5_write_array(
                        signals,
                        name,
                        values,
                        compression_level=config.hdf5_compression_level,
                    )
                _h5_write_frame(
                    group.create_group("rois"),
                    session.roi_metadata,
                    compression_level=config.hdf5_compression_level,
                )
                _h5_write_frame(
                    group.create_group("units"),
                    session.unit_metadata,
                    compression_level=config.hdf5_compression_level,
                )
                group.attrs["audit_json"] = json.dumps(session.audit, sort_keys=True)
            file.flush()
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    provenance = {
        "schema": "cadence-dandi-001868-provenance-v1",
        "dandiset_id": DANDISET_ID,
        "dandiset_version": DANDISET_VERSION,
        "animal_id": animal_id,
        "split_unit": "animal_id",
        "config": asdict(config),
        "descriptor_columns": list(INTERVENTION_DESCRIPTOR_COLUMNS),
        "raw_stim_channel_in_descriptor": False,
        "session_count": len(ordered),
        "sessions": [
            {
                "session_id": session.session_id,
                "session_day_index": index,
                "session_start_time": session.session_start_time,
                "source_asset_path": session.source_path,
                "source_sha256": session.source_sha256,
                "audit": session.audit,
            }
            for index, session in enumerate(ordered)
        ],
        "output": str(destination),
        "output_size": destination.stat().st_size,
        "output_sha256": hash_file(destination),
        "written_at_utc": datetime.now(UTC).isoformat(),
    }
    return destination, provenance


def discover_trimodal_assets(
    manifest: Mapping[str, Any],
    raw_root: str | Path,
    *,
    allow_partial: bool = False,
) -> dict[str, list[tuple[DANDIAsset, Path]]]:
    """Resolve frozen trimodal paths and keep every animal in its own bucket."""

    result = {animal: [] for animal in TASK_MICE}
    root = Path(raw_root)
    missing = []
    for asset in manifest_assets(manifest, scope="trimodal"):
        local = root / asset.path
        if not local.exists():
            missing.append(asset.path)
            continue
        result[asset.animal_id].append((asset, local))
    if missing and not allow_partial:
        raise FileNotFoundError(
            f"{len(missing)} frozen trimodal assets are missing under {root}; "
            f"first missing asset: {missing[0]}"
        )
    if not allow_partial:
        empty = [animal for animal, rows in result.items() if not rows]
        if empty:
            raise DANDIICMSError(f"task animals have no trimodal files: {empty}")
    for rows in result.values():
        rows.sort(key=lambda item: item[0].path)
    return result


def preprocess_release(
    manifest: Mapping[str, Any] | str | Path,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
    output_root: str | Path = DEFAULT_PROCESSED_ROOT,
    *,
    config: ICMSPreprocessConfig = DEFAULT_PREPROCESS_CONFIG,
    animals: Iterable[str] | None = None,
    allow_partial: bool = False,
    verify_raw: bool = True,
) -> pd.DataFrame:
    """Preprocess trimodal sessions to one checksummed HDF5 per task animal."""

    if not isinstance(manifest, Mapping):
        manifest = load_frozen_manifest(manifest)
    selected_animals = set(TASK_MICE if animals is None else animals)
    unknown = sorted(selected_animals - set(TASK_MICE))
    if unknown:
        raise DANDIICMSError(f"unknown task animals: {unknown}")
    discovered = discover_trimodal_assets(
        manifest,
        raw_root,
        allow_partial=allow_partial,
    )
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for animal_id in TASK_MICE:
        if animal_id not in selected_animals:
            continue
        asset_paths = discovered[animal_id]
        if not asset_paths:
            if allow_partial:
                continue
            raise DANDIICMSError(f"no trimodal sessions found for {animal_id}")
        sessions = []
        for asset, local in asset_paths:
            if verify_raw:
                verify_asset(local, asset)
            sessions.append(
                load_icms_session(
                    local,
                    config=config,
                    expected_animal=animal_id,
                    source_asset_path=asset.path,
                    source_sha256=asset.sha256,
                )
            )
        output, provenance = write_animal_file(
            sessions,
            destination / f"sub-{animal_id}.h5",
            config=config,
        )
        provenance_path = destination / f"sub-{animal_id}.provenance.json"
        write_json_atomic(provenance, provenance_path)
        rows.append(
            {
                "animal_id": animal_id,
                "sessions": len(sessions),
                "normal_calibration_trials": sum(
                    int(session.audit["n_normal_calibration_trials"]) for session in sessions
                ),
                "catch_normal_calibration_trials": sum(
                    int(session.audit["n_catch_normal_calibration_trials"]) for session in sessions
                ),
                "iti_calibration_windows": sum(
                    int(session.audit["n_iti_calibration_windows"]) for session in sessions
                ),
                "stimulation_trials": sum(
                    int(session.audit["n_stimulation_trials"]) for session in sessions
                ),
                "output": str(output),
                "output_sha256": provenance["output_sha256"],
                "provenance": str(provenance_path),
            }
        )
    index = pd.DataFrame(rows)
    if not index.empty:
        index = index.sort_values("animal_id", kind="stable").reset_index(drop=True)
    index_rows = index.to_dict(orient="records")
    count_columns = (
        "sessions",
        "normal_calibration_trials",
        "catch_normal_calibration_trials",
        "iti_calibration_windows",
        "stimulation_trials",
    )
    write_json_atomic(
        {
            "schema": "cadence-dandi-001868-index-v1",
            "dandiset_id": DANDISET_ID,
            "dandiset_version": DANDISET_VERSION,
            "split_unit": "animal_id",
            "config": asdict(config),
            "totals": {
                column: int(index[column].sum()) if column in index else 0
                for column in count_columns
            },
            "animals": index_rows,
        },
        destination / "index.json",
    )
    return index


def _parse_config(path: str | Path | None) -> ICMSPreprocessConfig:
    if path is None:
        return DEFAULT_PREPROCESS_CONFIG
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    settings = payload.get("preprocessing", payload)
    return ICMSPreprocessConfig(**settings)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze-manifest")
    freeze.add_argument("--output", type=Path, default=DEFAULT_MANIFEST_PATH)
    freeze.add_argument("--workers", type=int, default=12)
    audit = subparsers.add_parser("audit-manifest")
    audit.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    audit.add_argument("--workers", type=int, default=12)
    download = subparsers.add_parser("download")
    download.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    download.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    download.add_argument("--scope", choices=("all", "task", "trimodal"), default="task")
    download.add_argument("--workers", type=int, default=4)
    download.add_argument("--overwrite", action="store_true")
    preprocess = subparsers.add_parser("preprocess")
    preprocess.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    preprocess.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    preprocess.add_argument("--output-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    preprocess.add_argument("--config", type=Path)
    preprocess.add_argument("--animal", action="append", dest="animals")
    preprocess.add_argument("--allow-partial", action="store_true")
    preprocess.add_argument("--skip-raw-verification", action="store_true")
    args = parser.parse_args()

    if args.command == "freeze-manifest":
        output = write_json_atomic(
            build_published_manifest(workers=args.workers),
            args.output,
        )
        print(output)
    elif args.command == "audit-manifest":
        frozen = load_frozen_manifest(args.manifest)
        audit_frozen_manifest_against_api(frozen, workers=args.workers)
        print("frozen manifest exactly matches the published DANDI API")
    elif args.command == "download":
        result = download_release(
            args.manifest,
            args.raw_root,
            scope=args.scope,
            workers=args.workers,
            overwrite=args.overwrite,
        )
        print(result.to_string(index=False))
    elif args.command == "preprocess":
        result = preprocess_release(
            args.manifest,
            args.raw_root,
            args.output_root,
            config=_parse_config(args.config),
            animals=args.animals,
            allow_partial=args.allow_partial,
            verify_raw=not args.skip_raw_verification,
        )
        print(result.to_string(index=False))


if __name__ == "__main__":
    _main()
