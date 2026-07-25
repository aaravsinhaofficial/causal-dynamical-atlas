from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from cadence.data.dandi_icms import (
    EXPECTED_ASSET_COUNT,
    EXPECTED_TASK_ASSET_COUNT,
    EXPECTED_TRIMODAL_ASSET_COUNT,
    INTERVENTION_DESCRIPTOR_COLUMNS,
    TASK_MICE,
    DANDIAsset,
    DANDIICMSError,
    ICMSPreprocessConfig,
    assert_iti_windows_are_isolated,
    assert_normal_calibration_sealed,
    classify_trials_and_events,
    intervention_descriptors,
    load_frozen_manifest,
    load_icms_session,
    manifest_assets,
    select_iti_calibration_windows,
    verify_asset,
    write_animal_file,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_MANIFEST = REPOSITORY_ROOT / "configs/dandi_001868_assets.json"


def _configured_real_icms83_path() -> tuple[Path | None, str]:
    if os.environ.get("CADENCE_RUN_REAL_ICMS_AUDIT") != "1":
        return (
            None,
            "real-container schema audit skipped: set CADENCE_RUN_REAL_ICMS_AUDIT=1 to opt in",
        )

    configured_root = os.environ.get("CADENCE_REAL_ICMS_EXAMPLE_DIR")
    if not configured_root:
        return (
            None,
            "real-container schema audit skipped: CADENCE_REAL_ICMS_EXAMPLE_DIR is not configured",
        )

    example_root = Path(configured_root).expanduser()
    if not example_root.is_dir():
        return (
            None,
            "real-container schema audit skipped: "
            "CADENCE_REAL_ICMS_EXAMPLE_DIR must be an existing directory",
        )

    example_path = example_root / "sub-ICMS83.nwb"
    if not example_path.is_file():
        return (
            None,
            "real-container schema audit skipped: the configured directory "
            "does not contain sub-ICMS83.nwb",
        )
    return example_path, ""


REAL_ICMS83_PATH, REAL_ICMS_AUDIT_SKIP_REASON = _configured_real_icms83_path()


def _dataset(group: h5py.Group, name: str, values: object) -> h5py.Dataset:
    array = np.asarray(values)
    if array.dtype.kind in {"O", "U"}:
        return group.create_dataset(
            name,
            data=np.asarray(values, dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
    return group.create_dataset(name, data=array)


def _write_synthetic_nwb(path: Path, *, animal_id: str = "ICMS100") -> Path:
    """Write the minimal real DANDI schema, with analytically known signals."""

    with h5py.File(path, "w") as file:
        subject = file.create_group("general").create_group("subject")
        _dataset(subject, "subject_id", animal_id)
        _dataset(file, "session_start_time", "2023-11-02T00:00:00+00:00")

        intervals = file.create_group("intervals")
        trials = intervals.create_group("trials")
        trial_values = {
            "id": [0, 1],
            "start_time": [2.0, 6.0],
            "stop_time": [2.7, 6.7],
            "trial_index": [1, 2],
            "current_uA": [5.0, 0.0],
            "stim_channel": [3, 0],
            "is_hit": [True, False],
            "response_time": [0.25, np.nan],
            "is_good_trial": [True, True],
        }
        for name, values in trial_values.items():
            _dataset(trials, name, values)

        events = intervals.create_group("electrical_stimulation")
        event_values = {
            "id": [0],
            "start_time": [2.01],
            "stop_time": [2.71],
            "trial_index": [1],
            "current_uA": [5.0],
            "stim_channel": [3],
            "pulse_count": [70],
            "frequency_hz": [100.0],
            "pulse_width_us": [167.0],
        }
        for name, values in event_values.items():
            _dataset(events, name, values)

        ecephys = file["general"].create_group("extracellular_ephys")
        electrodes = ecephys.create_group("electrodes")
        for name, values in {
            "id": [0, 1],
            "channel_name": ["1", "5"],
            "rel_x": [0.0, 0.0],
            "rel_y": [480.0, 600.0],
            "rel_z": [0.0, 0.0],
        }.items():
            _dataset(electrodes, name, values)

        processing = file.create_group("processing")
        ophys = processing.create_group("ophys")
        dff = ophys.create_group("DfOverF").create_group("DfOverF_Volumetric")
        source_time = np.arange(0.0, 10.0, 0.1)
        _dataset(dff, "data", np.column_stack((source_time, 2.0 * source_time)))
        _dataset(dff, "rois", [10, 11])
        starting = _dataset(dff, "starting_time", 0.0)
        starting.attrs["rate"] = 10.0

        behavior = processing.create_group("behavior")
        wheel = behavior.create_group("wheel").create_group("wheel_position_processed")
        _dataset(wheel, "data", source_time)
        starting = _dataset(wheel, "starting_time", 0.0)
        starting.attrs["rate"] = 10.0

        units = file.create_group("units")
        for name, values in {
            "id": [10, 11],
            "accepted": [True, False],
            "spike_times": [1.99, 2.02, 2.40, 2.80, 6.02, 6.80, 7.00],
            "spike_times_index": [5, 7],
            "cell_type": ["pyramidal", "unknown"],
            "peak_channel_index": [0, 1],
            "unit_x_um": [0.0, 0.0],
            "unit_y_um": [480.0, 600.0],
        }.items():
            _dataset(units, name, values)
    return path


@pytest.fixture
def synthetic_nwb(tmp_path: Path) -> Path:
    return _write_synthetic_nwb(tmp_path / "sub-ICMS100_ses-2023-11-02_behavior+ecephys+ophys.nwb")


@pytest.fixture
def small_config() -> ICMSPreprocessConfig:
    return ICMSPreprocessConfig(
        sample_rate_hz=10.0,
        window_start_s=-0.5,
        window_stop_s=1.0,
        calcium_max_gap_s=0.25,
        wheel_max_gap_s=0.25,
        ephys_artifact_start_s=-0.002,
        ephys_artifact_stop_s=0.705,
        include_iti_calibration=False,
    )


def test_frozen_manifest_identifies_release_and_six_task_mice() -> None:
    manifest = load_frozen_manifest(FROZEN_MANIFEST)
    assert manifest["asset_count"] == EXPECTED_ASSET_COUNT == 85
    assert manifest["task_asset_count"] == EXPECTED_TASK_ASSET_COUNT == 55
    assert manifest["trimodal_asset_count"] == EXPECTED_TRIMODAL_ASSET_COUNT == 45
    assert manifest["total_bytes"] == 7_504_049_197
    assert manifest["task_asset_bytes"] == 7_414_701_188
    assert manifest["trimodal_asset_bytes"] == 6_812_254_225
    assert set(manifest["task_mice"]) == set(TASK_MICE)

    all_assets = manifest_assets(manifest, scope="all")
    task_assets = manifest_assets(manifest, scope="task")
    trimodal = manifest_assets(manifest, scope="trimodal")
    assert (len(all_assets), len(task_assets), len(trimodal)) == (85, 55, 45)
    assert sum(asset.size for asset in all_assets) == manifest["total_bytes"]
    assert len({asset.path for asset in all_assets}) == 85
    assert len({asset.asset_id for asset in all_assets}) == 85
    assert all(re.fullmatch(r"[0-9a-f]{64}", asset.sha256) for asset in all_assets)
    assert all(asset.animal_id in TASK_MICE and asset.trimodal for asset in trimodal)

    counts = pd.Series([asset.animal_id for asset in trimodal]).value_counts().to_dict()
    assert counts == {
        "ICMS83": 9,
        "ICMS92": 9,
        "ICMS93": 10,
        "ICMS98": 7,
        "ICMS100": 4,
        "ICMS101": 6,
    }


def test_prespecified_continuous_calibration_policy_is_frozen() -> None:
    config = ICMSPreprocessConfig()
    assert config.include_iti_calibration
    assert config.iti_guard_s == 2.0
    assert config.iti_windows_per_session == 40
    assert config.event_onset_tolerance_s == 0.3
    assert config.window_stop_s - config.window_start_s == 4.0


def test_manifest_corruption_fails_closed() -> None:
    manifest = load_frozen_manifest(FROZEN_MANIFEST)
    corrupted = json.loads(json.dumps(manifest))
    corrupted["assets"][0]["sha256"] = "0" * 64
    # A syntactically valid but incorrect checksum is caught by a re-query or
    # download verification; a malformed one is caught at manifest load.
    corrupted["assets"][0]["sha256"] = "not-a-sha"
    with pytest.raises(DANDIICMSError, match="invalid SHA-256"):
        from cadence.data.dandi_icms import validate_frozen_manifest

        validate_frozen_manifest(corrupted)


def test_asset_verification_checks_length_and_sha256(tmp_path: Path) -> None:
    import hashlib

    payload = b"frozen DANDI asset fixture"
    local = tmp_path / "fixture.nwb"
    local.write_bytes(payload)
    asset = DANDIAsset(
        asset_id="fixture",
        blob_id="fixture",
        path="sub-ICMS100/fixture.nwb",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        dandi_etag="fixture",
        download_url="https://example.invalid/fixture",
        task_animal=True,
        trimodal=True,
    )
    assert verify_asset(local, asset)["verified"]
    local.write_bytes(payload + b"corruption")
    with pytest.raises(DANDIICMSError, match="size mismatch"):
        verify_asset(local, asset)


def test_catch_requires_zero_current_and_no_window_overlap(
    small_config: ICMSPreprocessConfig,
) -> None:
    trials = pd.DataFrame(
        {
            "start_time": [2.0, 6.0, 6.90],
            "stop_time": [2.7, 6.7, 7.60],
            "trial_index": [1, 2, 3],
            "current_uA": [5.0, 0.0, 4.0],
            "stim_channel": [3, 0, 5],
            "is_hit": [True, False, True],
            "response_time": [0.2, np.nan, 0.3],
            "is_good_trial": [True, True, True],
        }
    )
    events = pd.DataFrame(
        {
            "start_time": [2.01, 6.91],
            "stop_time": [2.71, 7.61],
            "trial_index": [1, 3],
            "current_uA": [5.0, 4.0],
            "stim_channel": [3, 5],
            "pulse_count": [70, 70],
            "frequency_hz": [100.0, 100.0],
            "pulse_width_us": [167.0, 167.0],
        }
    )
    classified = classify_trials_and_events(trials, events, config=small_config)
    catch = classified.loc[classified["trial_index"].eq(2)].iloc[0]
    assert catch["current_uA"] == 0.0
    assert catch["is_catch"]
    assert catch["overlapping_stimulation_events"] == 1
    assert not catch["is_normal_calibration"]
    assert catch["window_kind"] == "excluded_catch_overlap"

    with pytest.raises(DANDIICMSError, match="rejected catch"):
        assert_normal_calibration_sealed(classified.loc[classified["trial_index"].eq(2)])


def test_catch_with_same_index_event_is_hard_error(
    small_config: ICMSPreprocessConfig,
) -> None:
    trials = pd.DataFrame(
        {
            "start_time": [6.0],
            "stop_time": [6.7],
            "trial_index": [2],
            "current_uA": [0.0],
            "stim_channel": [0],
            "is_hit": [False],
            "response_time": [np.nan],
            "is_good_trial": [True],
        }
    )
    events = pd.DataFrame(
        {
            "start_time": [6.01],
            "stop_time": [6.71],
            "trial_index": [2],
            "current_uA": [5.0],
            "stim_channel": [3],
            "pulse_count": [70],
            "frequency_hz": [100.0],
            "pulse_width_us": [167.0],
        }
    )
    with pytest.raises(DANDIICMSError, match="catch trial 2 has a stimulation event"):
        classify_trials_and_events(trials, events, config=small_config)


def test_iti_windows_are_deterministic_and_wholly_outside_trials_and_events() -> None:
    trials = pd.DataFrame(
        {
            "start_time": [2.0, 8.0, 14.0, 20.0, 26.0],
            "stop_time": [2.7, 8.7, 14.7, 20.7, 26.7],
        }
    )
    events = pd.DataFrame(
        {
            "start_time": [2.01, 8.01, 14.01, 20.01, 26.01],
            "stop_time": [2.71, 8.71, 14.71, 20.71, 26.71],
        }
    )
    config = ICMSPreprocessConfig(
        sample_rate_hz=10.0,
        window_start_s=-0.5,
        window_stop_s=1.0,
        iti_guard_s=0.25,
        iti_windows_per_session=3,
    )
    selected, audit = select_iti_calibration_windows(
        trials,
        events,
        support_start=0.0,
        support_stop=30.0,
        config=config,
    )
    shuffled, shuffled_audit = select_iti_calibration_windows(
        trials.sample(frac=1.0, random_state=7),
        events.sample(frac=1.0, random_state=9),
        support_start=0.0,
        support_stop=30.0,
        config=config,
    )
    assert len(selected) == 3
    pd.testing.assert_frame_equal(selected, shuffled)
    assert audit == shuffled_audit
    assert audit["candidate_iti_count"] == 4
    assert audit["selection"] == ("one_centered_per_eligible_iti_then_evenly_spaced_ordinals")
    assert not audit["uses_signal_values"]
    assert not audit["uses_intervention_pre_onset_segments"]
    assert selected["current_uA"].eq(0.0).all()
    assert selected["is_normal_calibration"].all()
    assert selected["is_iti_calibration"].all()
    assert selected["normal_source"].eq("iti").all()
    assert_iti_windows_are_isolated(
        selected,
        trials,
        events,
        guard_s=config.iti_guard_s,
    )
    assert_normal_calibration_sealed(selected)


def test_loader_adds_continuous_iti_ephys_and_wheel_windows(
    synthetic_nwb: Path,
) -> None:
    config = ICMSPreprocessConfig(
        sample_rate_hz=10.0,
        window_start_s=-0.5,
        window_stop_s=1.0,
        calcium_max_gap_s=0.25,
        wheel_max_gap_s=0.25,
        iti_guard_s=0.25,
        iti_windows_per_session=4,
    )
    session = load_icms_session(synthetic_nwb, config=config)
    iti = session.trial_metadata["is_iti_calibration"].astype(bool).to_numpy()
    assert iti.sum() == 1
    assert session.audit["n_iti_calibration_windows"] == 1
    assert session.audit["iti_wheel_valid_fraction"] == 1.0
    assert session.audit["iti_spike_valid_fraction"] == 1.0
    assert session.spike_valid_mask[iti].all()
    assert session.wheel_valid_mask[iti].all()
    assert (session.intervention_descriptors[iti] == 0.0).all()
    assert_normal_calibration_sealed(session.trial_metadata.loc[iti])


def test_loader_aligns_all_modalities_and_preserves_masks(
    synthetic_nwb: Path,
    small_config: ICMSPreprocessConfig,
) -> None:
    session = load_icms_session(
        synthetic_nwb,
        config=small_config,
        expected_animal="ICMS100",
    )
    assert session.animal_id == "ICMS100"
    assert session.session_id == "2023-11-02"
    assert session.calcium_dff.shape == (2, 15, 2)
    assert session.calcium_valid_mask.shape == session.calcium_dff.shape
    assert session.calcium_observed_mask.shape == session.calcium_dff.shape
    assert session.spike_rate_hz.shape == (2, 15, 1)
    assert session.spike_valid_mask.shape == (2, 15)
    assert session.wheel_position.shape == (2, 15)
    assert session.wheel_velocity.shape == (2, 15)
    assert session.wheel_valid_mask.shape == (2, 15)
    np.testing.assert_allclose(
        session.time_s,
        -0.45 + np.arange(15) * 0.1,
        atol=1e-7,
    )

    # Calcium column 0 and wheel position both equal absolute source time.
    expected_stim_time = 2.01 + session.time_s
    expected_catch_time = 6.01 + session.time_s
    np.testing.assert_allclose(session.calcium_dff[0, :, 0], expected_stim_time, atol=5e-7)
    np.testing.assert_allclose(session.calcium_dff[1, :, 0], expected_catch_time, atol=5e-7)
    np.testing.assert_allclose(session.wheel_position[0], expected_stim_time, atol=5e-7)
    np.testing.assert_allclose(session.wheel_position[1], expected_catch_time, atol=5e-7)
    np.testing.assert_allclose(session.wheel_velocity, 1.0, atol=2e-5)
    assert session.calcium_valid_mask.all()
    assert session.wheel_valid_mask.all()

    trials = session.trial_metadata.set_index("trial_index")
    assert trials.loc[1, "anchor_source"] == "electrical_stimulation"
    assert trials.loc[2, "anchor_source"] == "catch_pseudo_onset"
    assert trials.loc[2, "anchor_time"] == pytest.approx(6.01)
    normal = session.trial_metadata.loc[session.trial_metadata["is_normal_calibration"]]
    assert list(normal["trial_index"]) == [2]
    assert_normal_calibration_sealed(normal)


def test_descriptor_uses_coordinates_and_physics_never_raw_channel(
    synthetic_nwb: Path,
    small_config: ICMSPreprocessConfig,
) -> None:
    session = load_icms_session(synthetic_nwb, config=small_config)
    assert "stim_channel" not in INTERVENTION_DESCRIPTOR_COLUMNS
    assert "stim_channel" in session.trial_metadata
    assert not session.audit["raw_channel_is_descriptor"]
    descriptor = dict(
        zip(
            INTERVENTION_DESCRIPTOR_COLUMNS,
            session.intervention_descriptors[0],
            strict=True,
        )
    )
    assert descriptor == pytest.approx(
        {
            "stim_present": 1.0,
            "current_uA": 5.0,
            "frequency_hz": 100.0,
            "pulse_count": 70.0,
            "pulse_width_us": 167.0,
            "electrode_rel_x_um": 0.0,
            "electrode_rel_y_um": 1440.0,
            "electrode_rel_z_um": 0.0,
            "electrode_depth_centered_um": 510.0,
            "electrode_depth_fraction": 510.0 / 930.0,
        }
    )
    assert (session.intervention_descriptors[1] == 0.0).all()

    # Equal channel integers may intentionally map to different coordinates in
    # different animals; only the coordinate can cross the split boundary.
    trial = session.trial_metadata.iloc[[0]]
    common = intervention_descriptors(
        trial,
        {3: (0.0, 1440.0, 0.0)},
    )
    icms83 = intervention_descriptors(
        trial,
        {3: (0.0, 420.0, 0.0)},
    )
    y_column = INTERVENTION_DESCRIPTOR_COLUMNS.index("electrode_rel_y_um")
    assert common[0, y_column] == 1440.0
    assert icms83[0, y_column] == 420.0


def test_ephys_artifact_is_masked_only_on_stimulation(
    synthetic_nwb: Path,
    small_config: ICMSPreprocessConfig,
) -> None:
    session = load_icms_session(synthetic_nwb, config=small_config)
    edges = small_config.relative_edges_s
    expected_artifact = (edges[:-1] < small_config.ephys_artifact_stop_s) & (
        edges[1:] > small_config.ephys_artifact_start_s
    )
    assert np.array_equal(~session.spike_valid_mask[0], expected_artifact)
    assert session.spike_valid_mask[1].all()
    assert np.isnan(session.spike_rate_hz[0, expected_artifact]).all()
    assert np.isfinite(session.spike_rate_hz[1]).all()


def test_animal_isolation_is_enforced_on_load_and_write(
    synthetic_nwb: Path,
    small_config: ICMSPreprocessConfig,
    tmp_path: Path,
) -> None:
    with pytest.raises(DANDIICMSError, match="animal isolation failure"):
        load_icms_session(
            synthetic_nwb,
            config=small_config,
            expected_animal="ICMS92",
        )
    session = load_icms_session(synthetic_nwb, config=small_config)
    other = dataclasses.replace(session, animal_id="ICMS92")
    with pytest.raises(DANDIICMSError, match="mixed sessions"):
        write_animal_file(
            [session, other],
            tmp_path / "mixed.h5",
            config=small_config,
        )


def test_per_animal_hdf5_schema_and_provenance(
    synthetic_nwb: Path,
    small_config: ICMSPreprocessConfig,
    tmp_path: Path,
) -> None:
    session = load_icms_session(synthetic_nwb, config=small_config)
    output, provenance = write_animal_file(
        [session],
        tmp_path / "sub-ICMS100.h5",
        config=small_config,
    )
    assert provenance["animal_id"] == "ICMS100"
    assert provenance["split_unit"] == "animal_id"
    assert re.fullmatch(r"[0-9a-f]{64}", provenance["output_sha256"])
    with h5py.File(output, "r") as file:
        assert file.attrs["animal_id"] == "ICMS100"
        assert not file.attrs["raw_stim_channel_in_descriptor"]
        descriptors = [value.decode() for value in file["descriptor_columns"][()]]
        assert tuple(descriptors) == INTERVENTION_DESCRIPTOR_COLUMNS
        group = file["sessions/day-00_2023-11-02"]
        assert group.attrs["session_day_index"] == 0
        assert group.attrs["days_since_first_session"] == 0
        assert group["signals/calcium_dff"].shape == (2, 15, 2)
        assert group["signals/spike_rate_hz"].shape == (2, 15, 1)
        assert group["signals/wheel_velocity"].shape == (2, 15)
        assert group["trials/trial_index"].shape == (2,)
        assert group["rois/roi_id"].shape == (2,)
        assert group["units/unit_id"].shape == (1,)


@pytest.mark.skipif(
    REAL_ICMS83_PATH is None,
    reason=REAL_ICMS_AUDIT_SKIP_REASON,
)
def test_real_icms83_wiring_and_shapes() -> None:
    """Regression-test the implant whose channel wiring differs from the others."""

    assert REAL_ICMS83_PATH is not None
    session = load_icms_session(REAL_ICMS83_PATH)
    assert session.animal_id == "ICMS83"
    assert session.calcium_dff.shape == (220, 120, 1761)
    assert session.spike_rate_hz.shape == (220, 120, 6)
    assert session.wheel_position.shape == (220, 120)
    assert session.audit["n_catch_trials"] == 0
    assert session.audit["n_stimulation_trials"] == 180
    assert session.audit["n_iti_calibration_windows"] == 40
    assert session.audit["n_normal_calibration_trials"] == 40
    assert session.audit["iti_wheel_valid_fraction"] == 1.0
    assert session.audit["iti_spike_valid_fraction"] == 1.0
    assert session.audit["iti_total_spikes"] > 0
    descriptor_y = INTERVENTION_DESCRIPTOR_COLUMNS.index("electrode_rel_y_um")
    channels = session.trial_metadata["stim_channel"].to_numpy()
    assert np.unique(session.intervention_descriptors[channels == 3, descriptor_y]) == 420.0
    assert np.unique(session.intervention_descriptors[channels == 4, descriptor_y]) == 1860.0
