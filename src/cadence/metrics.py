"""Animal-level metrics for complete perturbation-response trajectories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]


def _as_float(array: ArrayLike) -> FloatArray:
    return np.asarray(array, dtype=np.float64)


def support_scale(
    normal_support: ArrayLike,
    *,
    floor_quantile: float = 0.1,
    cap_quantile: float = 0.9,
) -> FloatArray:
    """Channel scales fit exclusively on normal support.

    The input may have any leading dimensions; the final dimension is treated
    as channels. Quantile clipping prevents near-constant cells from receiving
    extreme weight.
    """
    values = _as_float(normal_support)
    if values.ndim < 2:
        raise ValueError("normal support must have samples and channels")
    flattened = values.reshape(-1, values.shape[-1])
    scale = np.nanstd(flattened, axis=0, ddof=1)
    valid = scale[np.isfinite(scale) & (scale > 0)]
    if valid.size == 0:
        return np.ones(values.shape[-1], dtype=np.float64)
    low, high = np.quantile(valid, [floor_quantile, cap_quantile])
    return np.clip(np.nan_to_num(scale, nan=high, posinf=high, neginf=low), low, high)


def causal_skill(
    predicted_effect: ArrayLike,
    observed_effect: ArrayLike,
    *,
    channel_scale: ArrayLike | None = None,
    mask: ArrayLike | None = None,
) -> float:
    """Skill relative to predicting no perturbation effect.

    Arrays end in a channel dimension. Every preceding element is weighted
    equally. A perfect prediction scores 1, the no-effect predictor scores 0,
    and values below 0 are worse than no effect.
    """
    prediction = _as_float(predicted_effect)
    observed = _as_float(observed_effect)
    if prediction.shape != observed.shape:
        raise ValueError("predicted and observed effect shapes differ")
    if channel_scale is None:
        scale = np.ones(prediction.shape[-1])
    else:
        scale = _as_float(channel_scale)
        if scale.shape != (prediction.shape[-1],):
            raise ValueError("channel_scale must match the final dimension")
    error = np.square((prediction - observed) / scale)
    reference = np.square(observed / scale)
    valid = np.isfinite(error) & np.isfinite(reference)
    if mask is not None:
        valid &= np.broadcast_to(np.asarray(mask, dtype=bool), prediction.shape)
    denominator = reference[valid].sum()
    if denominator <= np.finfo(np.float64).eps:
        return float("nan")
    return float(1.0 - error[valid].sum() / denominator)


def trajectory_nrmse(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    channel_scale: ArrayLike,
) -> float:
    prediction_array = _as_float(prediction)
    target_array = _as_float(target)
    scale = _as_float(channel_scale)
    if prediction_array.shape != target_array.shape:
        raise ValueError("prediction and target shapes differ")
    return float(np.sqrt(np.nanmean(((prediction_array - target_array) / scale) ** 2)))


def time_resolved_r2(prediction: ArrayLike, target: ArrayLike) -> FloatArray:
    """Observed-space R2 at each time, pooling trials/conditions and channels."""
    prediction_array = _as_float(prediction)
    target_array = _as_float(target)
    if prediction_array.shape != target_array.shape or prediction_array.ndim < 3:
        raise ValueError("expected matching [sample, time, ..., channel] arrays")
    time_axis = 1
    output = np.empty(prediction_array.shape[time_axis], dtype=np.float64)
    for time in range(output.size):
        pred = np.take(prediction_array, time, axis=time_axis).reshape(-1)
        obs = np.take(target_array, time, axis=time_axis).reshape(-1)
        valid = np.isfinite(pred) & np.isfinite(obs)
        centered = obs[valid] - np.mean(obs[valid])
        denominator = np.dot(centered, centered)
        output[time] = (
            np.nan
            if denominator <= np.finfo(float).eps
            else 1 - np.square(pred[valid] - obs[valid]).sum() / denominator
        )
    return output


def energy_score(samples: ArrayLike, observation: ArrayLike) -> float:
    """Multivariate proper score for posterior trajectory samples.

    ``samples`` is ``[draw, ...]`` and ``observation`` matches all remaining
    dimensions. Lower is better.
    """
    draws = _as_float(samples)
    observed = _as_float(observation)
    if draws.shape[1:] != observed.shape:
        raise ValueError("posterior sample and observation shapes differ")
    flat = draws.reshape(draws.shape[0], -1)
    obs = observed.reshape(-1)
    first = np.linalg.norm(flat - obs[None, :], axis=1).mean()
    pairwise = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
    return float(first - 0.5 * pairwise.mean())


def interval_coverage(
    lower: ArrayLike,
    upper: ArrayLike,
    target: ArrayLike,
) -> tuple[float, bool, float]:
    """Return pointwise coverage, simultaneous coverage, and mean width."""
    low = _as_float(lower)
    high = _as_float(upper)
    observed = _as_float(target)
    if low.shape != high.shape or low.shape != observed.shape:
        raise ValueError("interval and target shapes differ")
    valid = np.isfinite(low) & np.isfinite(high) & np.isfinite(observed)
    inside = (observed >= low) & (observed <= high) & valid
    pointwise = float(inside.sum() / valid.sum())
    simultaneous = bool(np.all(inside[valid]))
    width = float(np.nanmean(np.where(valid, high - low, np.nan)))
    return pointwise, simultaneous, width


def split_half_ceiling(
    trials: ArrayLike,
    *,
    repeats: int = 200,
    seed: int = 0,
) -> tuple[float, FloatArray]:
    """Reliability ceiling from random trial split halves."""
    values = _as_float(trials)
    if values.ndim < 3 or values.shape[0] < 4:
        raise ValueError("need [trial, time, channel] with at least four trials")
    generator = np.random.default_rng(seed)
    scores = []
    count = values.shape[0] // 2
    for _ in range(repeats):
        order = generator.permutation(values.shape[0])
        first = np.nanmean(values[order[:count]], axis=0).reshape(-1)
        second = np.nanmean(values[order[count : 2 * count]], axis=0).reshape(-1)
        valid = np.isfinite(first) & np.isfinite(second)
        if valid.sum() > 2 and np.std(first[valid]) > 0 and np.std(second[valid]) > 0:
            correlation = np.corrcoef(first[valid], second[valid])[0, 1]
            # Spearman-Brown correction estimates full-trial reliability.
            scores.append(2 * correlation / (1 + correlation) if correlation > -1 else -1)
    distribution = np.asarray(scores, dtype=np.float64)
    return float(np.nanmedian(distribution)), distribution


def animal_bootstrap_ci(
    animal_values: ArrayLike,
    *,
    statistic: Callable[[FloatArray], float] = np.mean,
    confidence: float = 0.95,
    repeats: int = 20_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Equal-animal nonparametric bootstrap confidence interval."""
    values = _as_float(animal_values).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("at least two finite animal values are required")
    generator = np.random.default_rng(seed)
    estimates = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sample = generator.choice(values, size=values.size, replace=True)
        estimates[index] = statistic(sample)
    alpha = 1 - confidence
    lower, upper = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return float(statistic(values)), float(lower), float(upper)


def paired_sign_flip_test(
    animal_differences: ArrayLike,
    *,
    exact_limit: int = 20,
    permutations: int = 100_000,
    seed: int = 0,
) -> float:
    """Two-sided paired randomization test with animals as replication units."""
    differences = _as_float(animal_differences).reshape(-1)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        raise ValueError("no finite animal differences")
    observed = abs(float(np.mean(differences)))
    if differences.size <= exact_limit:
        assignments = np.arange(1 << differences.size, dtype=np.uint64)[:, None]
        bits = (assignments >> np.arange(differences.size, dtype=np.uint64)) & 1
        signs = 2 * bits.astype(np.float64) - 1
    else:
        generator = np.random.default_rng(seed)
        signs = generator.choice((-1.0, 1.0), size=(permutations, differences.size))
    null = np.abs(np.mean(signs * differences[None, :], axis=1))
    return float((np.count_nonzero(null >= observed) + 1) / (null.size + 1))


@dataclass(frozen=True, slots=True)
class AnimalScore:
    animal_id: str
    neural_skill: float
    behavior_skill: float
    neural_nrmse: float
    behavior_nrmse: float
    neural_ceiling: float
    behavior_ceiling: float


def intersection_union_gate(
    scores: list[AnimalScore],
    *,
    baseline_neural_improvement: ArrayLike,
    baseline_behavior_improvement: ArrayLike,
    minimum_improvement: float = 0.1,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, object]:
    """Legacy convenience gate aligned with the canonical reporter.

    Gate 4 requires a mean baseline gain of at least ``minimum_improvement``
    and a confidence-interval lower bound above zero for each endpoint.
    """
    if len(scores) < 2:
        raise ValueError("at least two animals are required")
    neural = np.asarray([score.neural_skill for score in scores])
    behavior = np.asarray([score.behavior_skill for score in scores])
    neural_summary = animal_bootstrap_ci(neural, confidence=confidence, seed=seed)
    behavior_summary = animal_bootstrap_ci(behavior, confidence=confidence, seed=seed + 1)
    neural_gain = _as_float(baseline_neural_improvement).reshape(-1)
    behavior_gain = _as_float(baseline_behavior_improvement).reshape(-1)
    if neural_gain.size != len(scores) or behavior_gain.size != len(scores):
        raise ValueError("baseline improvement arrays must have one value per animal")
    neural_gain_ci = animal_bootstrap_ci(neural_gain, confidence=confidence, seed=seed + 2)
    behavior_gain_ci = animal_bootstrap_ci(behavior_gain, confidence=confidence, seed=seed + 3)
    gates = {
        "neural_skill_positive": neural_summary[1] > 0,
        "behavior_skill_positive": behavior_summary[1] > 0,
        "neural_baseline_margin": (
            neural_gain_ci[0] >= minimum_improvement and neural_gain_ci[1] > 0
        ),
        "behavior_baseline_margin": (
            behavior_gain_ci[0] >= minimum_improvement and behavior_gain_ci[1] > 0
        ),
    }
    return {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "neural_skill_mean_ci": neural_summary,
        "behavior_skill_mean_ci": behavior_summary,
        "neural_baseline_gain_mean_ci": neural_gain_ci,
        "behavior_baseline_gain_mean_ci": behavior_gain_ci,
    }
