from __future__ import annotations

import numpy as np
import pytest

from cadence.metrics import (
    AnimalScore,
    animal_bootstrap_ci,
    causal_skill,
    energy_score,
    intersection_union_gate,
    interval_coverage,
    paired_sign_flip_test,
    split_half_ceiling,
    support_scale,
)


def test_causal_skill_anchors() -> None:
    observed = np.arange(24, dtype=float).reshape(2, 3, 4) - 4
    assert causal_skill(observed, observed) == pytest.approx(1)
    assert causal_skill(np.zeros_like(observed), observed) == pytest.approx(0)
    assert causal_skill(-observed, observed) < 0


def test_support_scale_is_channelwise_and_clipped() -> None:
    values = np.stack(
        [np.arange(30), np.arange(30) * 100, np.ones(30)],
        axis=1,
    )
    scale = support_scale(values)
    assert scale.shape == (3,)
    assert np.all(np.isfinite(scale))
    assert np.all(scale > 0)


def test_energy_score_prefers_centered_samples() -> None:
    rng = np.random.default_rng(0)
    observation = np.zeros((3, 2))
    good = rng.normal(0, 0.1, size=(50, 3, 2))
    bad = rng.normal(5, 0.1, size=(50, 3, 2))
    assert energy_score(good, observation) < energy_score(bad, observation)


def test_interval_coverage_reports_simultaneous() -> None:
    target = np.array([0.0, 1.0, 2.0])
    pointwise, simultaneous, width = interval_coverage([-1, 0, 3], [1, 2, 4], target)
    assert pointwise == pytest.approx(2 / 3)
    assert simultaneous is False
    assert width == pytest.approx(5 / 3)


def test_split_half_ceiling_reliable_trials() -> None:
    rng = np.random.default_rng(1)
    template = rng.normal(size=(10, 4))
    trials = template[None] + rng.normal(scale=0.05, size=(30, 10, 4))
    ceiling, distribution = split_half_ceiling(trials, repeats=50)
    assert ceiling > 0.95
    assert distribution.shape == (50,)


def test_animal_inference_uses_animal_count() -> None:
    estimate, lower, upper = animal_bootstrap_ci([1, 2, 3], repeats=1_000)
    assert estimate == pytest.approx(2)
    assert lower <= estimate <= upper
    assert paired_sign_flip_test(np.ones(6)) < 0.05


def test_legacy_gate_uses_canonical_mean_and_positive_lower_bound() -> None:
    scores = [
        AnimalScore(
            animal_id=f"mouse-{index}",
            neural_skill=0.5,
            behavior_skill=0.5,
            neural_nrmse=0.0,
            behavior_nrmse=0.0,
            neural_ceiling=1.0,
            behavior_ceiling=1.0,
        )
        for index in range(2)
    ]
    passing = intersection_union_gate(
        scores,
        baseline_neural_improvement=[0.051, 0.151],
        baseline_behavior_improvement=[0.051, 0.151],
    )
    assert passing["neural_baseline_gain_mean_ci"][0] == pytest.approx(0.101)
    assert 0 < passing["neural_baseline_gain_mean_ci"][1] < 0.10
    assert passing["gates"]["neural_baseline_margin"] is True
    assert passing["gates"]["behavior_baseline_margin"] is True
    assert passing["passed"] is True

    failing = intersection_union_gate(
        scores,
        baseline_neural_improvement=[0.04, 0.14],
        baseline_behavior_improvement=[0.04, 0.14],
    )
    assert failing["neural_baseline_gain_mean_ci"][0] == pytest.approx(0.09)
    assert failing["neural_baseline_gain_mean_ci"][1] > 0
    assert failing["gates"]["neural_baseline_margin"] is False
    assert failing["gates"]["behavior_baseline_margin"] is False
    assert failing["passed"] is False
