"""Deterministic mouse-level splitting and leakage audits."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

import pandas as pd


class LeakageError(ValueError):
    """Raised when an animal or intervention leaks across a sealed boundary."""


def _canonical_mouse_ids(values: Iterable[object]) -> list[str]:
    ids = []
    for value in values:
        if pd.isna(value):
            raise ValueError("mouse identifiers may not be missing")
        identifier = str(value).strip()
        if not identifier:
            raise ValueError("mouse identifiers may not be empty")
        ids.append(identifier)
    return ids


def make_mouse_folds(
    records: pd.DataFrame | Sequence[object],
    *,
    n_splits: int = 5,
    seed: int = 20260725,
    mouse_col: str = "mouse_id",
) -> pd.DataFrame:
    """Assign each mouse to exactly one deterministic outer fold.

    The hash-based ordering is stable across pandas, NumPy, Python hash seeds,
    input row order, and machines.  Repeated experiment rows for a mouse receive
    the same fold after joining this table on ``mouse_id``.
    """

    if isinstance(records, pd.DataFrame):
        if mouse_col not in records:
            raise KeyError(f"missing required mouse column: {mouse_col}")
        values = records[mouse_col].tolist()
    else:
        values = list(records)

    mouse_ids = sorted(set(_canonical_mouse_ids(values)))
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > len(mouse_ids):
        raise ValueError(
            f"n_splits={n_splits} exceeds the number of unique mice ({len(mouse_ids)})"
        )

    def digest(mouse_id: str) -> bytes:
        return hashlib.sha256(f"cadence-mouse-fold-v1\0{seed}\0{mouse_id}".encode()).digest()

    ordered = sorted(mouse_ids, key=lambda mouse_id: (digest(mouse_id), mouse_id))
    assignments = pd.DataFrame(
        {
            mouse_col: ordered,
            "outer_fold": [index % n_splits for index in range(len(ordered))],
        }
    )
    assignments["split_seed"] = int(seed)
    assignments["split_unit"] = mouse_col
    return assignments.sort_values(mouse_col, kind="stable").reset_index(drop=True)


def assert_disjoint_animals(
    *partitions: pd.DataFrame | Iterable[object],
    mouse_col: str = "mouse_id",
) -> None:
    """Prove that no mouse appears in more than one supplied partition."""

    seen: dict[str, int] = {}
    for partition_index, partition in enumerate(partitions):
        if isinstance(partition, pd.DataFrame):
            if mouse_col not in partition:
                raise KeyError(f"missing required mouse column: {mouse_col}")
            values = partition[mouse_col]
        else:
            values = partition
        for mouse_id in set(_canonical_mouse_ids(values)):
            if mouse_id in seen:
                raise LeakageError(
                    f"mouse {mouse_id!r} occurs in partitions "
                    f"{seen[mouse_id]} and {partition_index}"
                )
            seen[mouse_id] = partition_index


def assert_calibration_is_normal(
    calibration: pd.DataFrame,
    *,
    kind_col: str = "window_kind",
) -> None:
    """Reject intervention, change, or sham-change rows in animal calibration.

    This is an intentionally redundant gate: selection code already creates
    normal windows, but serialized partitions are audited again immediately
    before they can be used to adapt a held-out animal.
    """

    if calibration.empty:
        raise LeakageError("calibration partition is empty")
    if kind_col not in calibration:
        raise KeyError(f"missing required calibration kind column: {kind_col}")

    kinds = calibration[kind_col].astype(str).str.lower()
    if not kinds.eq("normal").all():
        leaked = sorted(calibration.loc[~kinds.eq("normal"), kind_col].astype(str).unique())
        raise LeakageError(f"sealed calibration includes non-normal windows: {leaked}")

    true_values = {"1", "true", "t", "yes", "y"}
    false_values = {"0", "false", "f", "no", "n", "", "nan", "none"}
    for column in ("omitted", "is_change", "is_sham_change"):
        if column not in calibration:
            continue

        def normalize(value: object, column_name: str = column) -> bool:
            if pd.isna(value):
                return False
            if isinstance(value, bool):
                return value
            if isinstance(value, int | float):
                return bool(value)
            text = str(value).strip().lower()
            if text in true_values:
                return True
            if text in false_values:
                return False
            raise LeakageError(f"cannot audit non-boolean {column_name} value: {value!r}")

        if calibration[column].map(normalize).any():
            raise LeakageError(f"sealed calibration includes rows with {column}=True")
