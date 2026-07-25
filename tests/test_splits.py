from __future__ import annotations

import pandas as pd
import pytest

from cadence.data.splits import (
    LeakageError,
    assert_calibration_is_normal,
    assert_disjoint_animals,
    make_mouse_folds,
)


def test_mouse_folds_are_deterministic_and_row_order_invariant() -> None:
    experiments = pd.DataFrame(
        {
            "mouse_id": ["m3", "m1", "m2", "m1", "m6", "m5", "m4"],
            "ophys_experiment_id": [31, 11, 21, 12, 61, 51, 41],
        }
    )
    first = make_mouse_folds(experiments, n_splits=3, seed=17)
    second = make_mouse_folds(
        experiments.sample(frac=1, random_state=99),
        n_splits=3,
        seed=17,
    )

    pd.testing.assert_frame_equal(first, second)
    assert first["mouse_id"].is_unique
    assert set(first["outer_fold"]) == {0, 1, 2}

    joined = experiments.assign(mouse_id=experiments["mouse_id"].astype(str)).merge(
        first[["mouse_id", "outer_fold"]],
        on="mouse_id",
        validate="many_to_one",
    )
    assert joined.groupby("mouse_id")["outer_fold"].nunique().eq(1).all()


def test_animal_overlap_is_a_hard_error() -> None:
    train = pd.DataFrame({"mouse_id": ["m1", "m2"]})
    test = pd.DataFrame({"mouse_id": ["m2", "m3"]})

    with pytest.raises(LeakageError, match="m2"):
        assert_disjoint_animals(train, test)


def test_calibration_partition_rejects_intervention_and_change_rows() -> None:
    clean = pd.DataFrame(
        {
            "window_kind": ["normal", "normal"],
            "omitted": [False, False],
            "is_change": [False, False],
            "is_sham_change": [False, False],
        }
    )
    assert_calibration_is_normal(clean)

    intervention = clean.copy()
    intervention.loc[1, "window_kind"] = "omission"
    intervention.loc[1, "omitted"] = True
    with pytest.raises(LeakageError, match="non-normal"):
        assert_calibration_is_normal(intervention)

    changed = clean.copy()
    changed.loc[0, "is_change"] = True
    with pytest.raises(LeakageError, match="is_change"):
        assert_calibration_is_normal(changed)
