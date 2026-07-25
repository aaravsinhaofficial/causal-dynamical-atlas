"""Dataset preparation utilities.

The public functions in this package deliberately keep animal identity explicit.
Splits and calibration/evaluation partitions must never be inferred from row
order or session identifiers.
"""

from cadence.data.splits import (
    LeakageError,
    assert_calibration_is_normal,
    assert_disjoint_animals,
    make_mouse_folds,
)

__all__ = [
    "LeakageError",
    "assert_calibration_is_normal",
    "assert_disjoint_animals",
    "make_mouse_folds",
]
