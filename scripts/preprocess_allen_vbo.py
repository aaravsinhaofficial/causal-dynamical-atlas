#!/usr/bin/env python
"""Preprocess every mouse in the frozen Allen omission cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cadence.data.allen_vbo import extract_animal_nwb, write_json


def _extract_one(
    entry: dict[str, Any],
    raw_root: str,
    output_root: str,
    backend: str,
    normal_trials: int | None,
    minimum_omissions: int,
) -> dict[str, Any]:
    raw_path = Path(raw_root) / entry["local_filename"]
    paths = extract_animal_nwb(
        raw_path,
        output_root,
        backend=backend,  # type: ignore[arg-type]
        normal_calibration_trials=normal_trials,
        minimum_omissions=minimum_omissions,
        manifest_entry=entry,
        verify_input_hash=True,
    )
    provenance = json.loads(paths.provenance.read_text(encoding="utf-8"))
    return {
        "mouse_id": str(entry["mouse_id"]),
        "ophys_experiment_id": int(entry["ophys_experiment_id"]),
        "normal_windows": int(provenance["window_audit"]["normal_selected"]),
        "omission_windows": int(provenance["window_audit"]["omission_selected"]),
        "cells": int(provenance["signals"]["cells"]),
        "arrays": str(paths.arrays),
        "arrays_sha256": provenance["outputs"]["windows.npz"]["sha256"],
        "provenance": str(paths.provenance),
    }


def _source_content_commitment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Bind every file consumed by the experiment to the tracked index."""

    names = (
        "stimulus_presentations.parquet",
        "window_index.parquet",
        "windows.npz",
    )
    records = []
    for row in rows:
        provenance = json.loads(Path(row["provenance"]).read_text(encoding="utf-8"))
        records.append(
            {
                "mouse_id": str(row["mouse_id"]),
                "ophys_experiment_id": int(row["ophys_experiment_id"]),
                "outputs": {name: str(provenance["outputs"][name]["sha256"]) for name in names},
            }
        )
    records.sort(key=lambda record: record["mouse_id"])
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return {
        "algorithm": "sha256-canonical-json-v1",
        "files_per_mouse": list(names),
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/allen_vbo_slc17a7_visp175_familiar_active_v1.1.0.json"),
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/allen_vbo/nwb"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/allen_vbo"),
    )
    parser.add_argument("--backend", choices=("h5py", "pynwb"), default="h5py")
    parser.add_argument("--normal-trials", type=int, default=160)
    parser.add_argument(
        "--all-normal",
        action="store_true",
        help=(
            "retain every clean eligible normal window; the experiment runner "
            "still selects its fixed normal-only adapter subset"
        ),
    )
    parser.add_argument("--minimum-omissions", type=int, default=80)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = list(manifest["nwb_files"])
    missing = [
        str(args.raw_root / entry["local_filename"])
        for entry in entries
        if not (args.raw_root / entry["local_filename"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} frozen NWBs are missing; first missing file: {missing[0]}"
        )

    rows: list[dict[str, Any]] = []
    normal_trials = None if args.all_normal else args.normal_trials
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                _extract_one,
                entry,
                str(args.raw_root),
                str(args.output_root),
                args.backend,
                normal_trials,
                args.minimum_omissions,
            ): entry
            for entry in entries
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    rows.sort(key=lambda row: row["mouse_id"])
    if len(rows) != int(manifest["selection"]["num_animals"]):
        raise RuntimeError("processed cohort count does not match the frozen manifest")
    index = {
        "schema": "cadence-allen-vbo-processed-index-v1",
        "release": manifest["dataset"]["release"],
        "cohort_manifest": str(args.manifest),
        "animal_count": len(rows),
        "total_cells": sum(int(row["cells"]) for row in rows),
        "total_normal_windows": sum(int(row["normal_windows"]) for row in rows),
        "total_omission_windows": sum(int(row["omission_windows"]) for row in rows),
        "source_content_commitment": _source_content_commitment(rows),
        "animals": rows,
    }
    index_path = write_json(index, args.output_root / "index.json")
    print(json.dumps({"index": str(index_path), "animal_count": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
