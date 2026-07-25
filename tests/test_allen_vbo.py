from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from cadence.data.allen_vbo import (
    CohortSpec,
    WindowPolicy,
    build_public_download_manifest,
    construct_windows,
    extract_animal_nwb,
    select_one_experiment_per_mouse,
)


def _cell_rows(experiment_id: int, count: int) -> list[dict[str, int]]:
    return [
        {
            "ophys_experiment_id": experiment_id,
            "cell_roi_id": experiment_id * 1_000 + index,
            "cell_specimen_id": experiment_id * 10_000 + index,
        }
        for index in range(count)
    ]


def test_cohort_is_one_deterministic_experiment_per_mouse() -> None:
    common = {
        "cre_line": "Slc17a7-IRES2-Cre",
        "targeted_structure": "VISp",
        "imaging_depth": 175,
        "experience_level": "Familiar",
        "passive": False,
    }
    experiments = pd.DataFrame(
        [
            {
                **common,
                "mouse_id": "m1",
                "ophys_experiment_id": 11,
                "ophys_session_id": 101,
                "session_number": 3,
                "date_of_acquisition": "2020-01-01",
                "file_id": 111,
            },
            {
                **common,
                "mouse_id": "m1",
                "ophys_experiment_id": 13,
                "ophys_session_id": 103,
                "session_number": 1,
                "date_of_acquisition": "2020-01-03",
                "file_id": 113,
            },
            {
                **common,
                "mouse_id": "m1",
                "ophys_experiment_id": 12,
                "ophys_session_id": 102,
                "session_number": 1,
                "date_of_acquisition": "2020-01-02",
                "file_id": 112,
            },
            {
                **common,
                "mouse_id": "m2",
                "ophys_experiment_id": 21,
                "ophys_session_id": 201,
                "session_number": 1,
                "date_of_acquisition": "2020-02-01",
                "file_id": 121,
            },
            {
                **common,
                "mouse_id": "m2",
                "ophys_experiment_id": 22,
                "ophys_session_id": 202,
                "session_number": 3,
                "date_of_acquisition": "2020-02-02",
                "file_id": 122,
            },
            {
                **common,
                "mouse_id": "m3",
                "ophys_experiment_id": 31,
                "ophys_session_id": 301,
                "session_number": 1,
                "date_of_acquisition": "2020-03-01",
                "file_id": 131,
                "passive": True,
            },
            {
                **common,
                "mouse_id": "m5",
                "ophys_experiment_id": 51,
                "ophys_session_id": 501,
                "session_number": 1,
                "date_of_acquisition": "2020-05-01",
                "file_id": 151,
            },
            {
                **common,
                "mouse_id": "m5",
                "ophys_experiment_id": 50,
                "ophys_session_id": 500,
                "session_number": 1,
                "date_of_acquisition": "2020-05-01",
                "file_id": 150,
            },
        ]
    )
    cells = pd.DataFrame(
        _cell_rows(11, 45)
        + _cell_rows(12, 50)
        + _cell_rows(13, 55)
        + _cell_rows(21, 39)
        + _cell_rows(22, 41)
        + _cell_rows(31, 100)
        + _cell_rows(50, 40)
        + _cell_rows(51, 40)
    )

    selected = select_one_experiment_per_mouse(
        experiments.sample(frac=1, random_state=7),
        cells.sample(frac=1, random_state=8),
        spec=CohortSpec(minimum_cells=40),
    )

    assert selected["mouse_id"].tolist() == ["m1", "m2", "m5"]
    assert selected["ophys_experiment_id"].tolist() == [12, 22, 50]
    assert selected["mouse_id"].is_unique


def test_public_manifest_carries_release_hash_and_mouse_ids() -> None:
    cohort = pd.DataFrame(
        {
            "mouse_id": ["423606"],
            "ophys_experiment_id": [822024770],
            "ophys_session_id": [821471625],
            "file_id": [894],
            "num_cells": [436],
            "session_number": [1],
            "date_of_acquisition": ["2019-02-12"],
        }
    )
    digest = "a" * 128
    project = {
        "project_name": "visual-behavior-ophys",
        "manifest_version": "1.1.0",
        "data_files": {
            "894": {
                "url": (
                    "https://visual-behavior-ophys-data.s3.us-west-2.amazonaws.com/"
                    "visual-behavior-ophys/behavior_ophys_experiments/"
                    "behavior_ophys_experiment_822024770.nwb"
                ),
                "version_id": "fixed-version",
                "file_hash": digest,
            }
        },
        "metadata_files": {},
    }

    result = build_public_download_manifest(
        cohort,
        project,
        fetch_object_metadata=False,
    )

    assert result["selection"]["unit"] == "mouse_id"
    assert result["selection"]["num_animals"] == 1
    entry = result["nwb_files"][0]
    assert entry["mouse_id"] == "423606"
    assert entry["blake2b_512"] == digest
    assert entry["s3_version_id"] == "fixed-version"
    assert entry["s3_uri"].startswith("s3://visual-behavior-ophys-data/")


def _presentation_frame() -> pd.DataFrame:
    rows = [
        # Boundary rejection: the requested window starts before signal support.
        (0, 1.0, False, False, False, True),
        # A change at +1 s contaminates this omission analysis window.
        (1, 10.0, True, False, False, True),
        (2, 11.0, False, True, False, True),
        # Clean omission.
        (3, 20.0, True, False, False, True),
        # Normal at exactly -3 s from an omission is guard-contaminated.
        (4, 17.0, False, False, False, True),
        # Clean normals.
        (5, 16.0, False, False, False, True),
        (6, 24.0, False, False, False, True),
        # Sham events and a normal at the exact guard boundary are excluded.
        (7, 30.0, False, False, True, True),
        (8, 27.0, False, False, False, True),
        # Other omission at the analysis endpoint contaminates the first one.
        (9, 40.0, True, False, False, True),
        (10, 42.0, True, False, False, True),
        # Inactive presentations never enter either candidate pool.
        (11, 50.0, True, False, False, False),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "stimulus_presentation_id",
            "start_time",
            "omitted",
            "is_change",
            "is_sham_change",
            "active",
        ],
    ).assign(stop_time=lambda frame: frame["start_time"] + 0.25)


def test_window_rules_enforce_boundaries_and_contamination_guards() -> None:
    selection = construct_windows(
        _presentation_frame(),
        policy=WindowPolicy(
            rate_hz=2.0,
            window_start_s=-1.0,
            window_end_s=2.0,
            normal_contamination_guard_s=3.0,
        ),
        support_start_s=0.5,
        support_stop_s=44.0,
    )

    assert selection.omissions["stimulus_presentation_id"].tolist() == [3, 10]
    assert selection.normal["stimulus_presentation_id"].tolist() == [5, 6]
    assert not selection.normal[["omitted", "is_change", "is_sham_change"]].any().any()
    assert selection.audit["omission_rejected_contamination"] == 2
    assert selection.audit["normal_rejected_boundary"] == 1
    assert selection.audit["normal_rejected_contamination"] == 2


def _write_synthetic_nwb_layout(path: Path) -> None:
    timestamps = np.arange(0.0, 20.0 + 1e-9, 0.1)
    text_dtype = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as file:
        file.create_dataset("identifier", data=np.bytes_("123"))
        subject = file.create_group("general/subject")
        subject.create_dataset("subject_id", data=np.bytes_("mouse-x"))

        event = file.create_group("processing/ophys/event_detection")
        event.create_dataset(
            "data",
            data=np.column_stack([timestamps, 2 * timestamps]),
        )
        event.create_dataset("timestamps", data=timestamps)
        event.create_dataset("rois", data=np.asarray([0, 1], dtype=np.int64))
        cells = file.create_group("processing/ophys/image_segmentation/cell_specimen_table")
        cells.create_dataset("id", data=np.asarray([101, 102], dtype=np.int64))
        cells.create_dataset(
            "cell_specimen_id",
            data=np.asarray([1001, 1002], dtype=np.int64),
        )

        running = file.create_group("processing/running/speed")
        running.create_dataset("data", data=3 * timestamps)
        running.create_dataset("timestamps", data=timestamps)
        licks = file.create_group("processing/licking/licks")
        licks.create_dataset("timestamps", data=np.asarray([4.95, 5.05, 16.0]))

        pupil = file.create_group("acquisition/EyeTracking/pupil_tracking")
        pupil.create_dataset("area", data=100 + timestamps)
        pupil.create_dataset("timestamps", data=timestamps)
        blink = file.create_group("acquisition/EyeTracking/likely_blink")
        blink.create_dataset("data", data=np.zeros(len(timestamps), dtype=bool))

        stimulus = file.create_group("intervals/image_presentations")
        presentation_times = np.asarray([1.0, 5.0, 8.0, 12.0, 16.0])
        stimulus.create_dataset("id", data=np.arange(len(presentation_times)))
        stimulus.create_dataset("start_time", data=presentation_times)
        stimulus.create_dataset("stop_time", data=presentation_times + 0.25)
        stimulus.create_dataset(
            "omitted",
            data=np.asarray([False, True, False, False, False]),
        )
        stimulus.create_dataset(
            "is_change",
            data=np.asarray([False, False, False, True, False]),
        )
        stimulus.create_dataset(
            "is_sham_change",
            data=np.zeros(len(presentation_times), dtype=bool),
        )
        stimulus.create_dataset(
            "active",
            data=np.ones(len(presentation_times), dtype=bool),
        )
        stimulus.create_dataset(
            "image_name",
            data=np.asarray(["a", "omitted", "a", "b", "b"], dtype=object),
            dtype=text_dtype,
        )


def test_h5py_extraction_writes_aligned_per_animal_artifacts(tmp_path: Path) -> None:
    nwb = tmp_path / "synthetic.nwb"
    _write_synthetic_nwb_layout(nwb)
    output = tmp_path / "processed"

    paths = extract_animal_nwb(
        nwb,
        output,
        backend="h5py",
        policy=WindowPolicy(
            rate_hz=2.0,
            window_start_s=-0.5,
            window_end_s=0.5,
            normal_contamination_guard_s=1.5,
        ),
        normal_calibration_trials=2,
        minimum_omissions=1,
        selection_seed=3,
    )

    assert paths.directory.name == "mouse_mouse-x"
    assert all(
        path.exists()
        for path in (
            paths.arrays,
            paths.normal_support,
            paths.omission_query,
            paths.sealed_omission_outcomes,
            paths.stimulus_presentations,
            paths.windows,
            paths.provenance,
        )
    )
    with np.load(paths.arrays, allow_pickle=False) as arrays:
        np.testing.assert_allclose(arrays["relative_time_s"], [-0.5, 0.0, 0.5])
        assert arrays["omission_neural"].shape == (1, 3, 2)
        np.testing.assert_allclose(
            arrays["omission_neural"][0, :, 0],
            [4.5, 5.0, 5.5],
            atol=1e-5,
        )
        np.testing.assert_allclose(
            arrays["omission_behavior"][0, :, 0],
            [13.5, 15.0, 16.5],
            atol=1e-5,
        )
        np.testing.assert_allclose(
            arrays["omission_behavior"][0, :, 2],
            [0.0, 4.0, 0.0],
            atol=1e-5,
        )
        assert arrays["normal_neural"].shape == (2, 3, 2)
        assert arrays["cell_specimen_ids"].tolist() == [1001, 1002]
    with np.load(paths.normal_support, allow_pickle=False) as normal:
        assert not any(name.startswith("omission") for name in normal.files)
        assert normal["normal_neural"].shape == (2, 3, 2)
    with np.load(paths.omission_query, allow_pickle=False) as query:
        assert not any("post" in name for name in query.files)
        assert query["omission_pre_neural"].shape == (1, 1, 2)
        assert int(query["onset"]) == 1
    with np.load(paths.sealed_omission_outcomes, allow_pickle=False) as sealed:
        assert not any(name.startswith("omission_pre_") for name in sealed.files)
        assert sealed["omission_post_neural"].shape == (1, 2, 2)

    windows = pd.read_parquet(paths.windows)
    assert windows.loc[windows["window_kind"].eq("omission"), "omitted"].all()
    assert (
        not windows.loc[
            windows["window_kind"].eq("normal"),
            ["omitted", "is_change", "is_sham_change"],
        ]
        .any()
        .any()
    )
    provenance = json.loads(paths.provenance.read_text())
    assert provenance["mouse_id"] == "mouse-x"
    assert provenance["signals"]["cells"] == 2
    assert provenance["window_audit"]["omission_selected"] == 1
    assert set(provenance["outputs"]) == {
        "normal_support.npz",
        "omission_query.npz",
        "sealed_omission_outcomes.npz",
        "stimulus_presentations.parquet",
        "window_index.parquet",
        "windows.npz",
    }
