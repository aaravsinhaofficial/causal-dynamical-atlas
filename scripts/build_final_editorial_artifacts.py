#!/usr/bin/env python3
"""Build source-bound editorial artifacts from the frozen final report.

This script is deliberately downstream of ``scripts/aggregate_results.py``. It
reads only ``summary.json``, ``report.complete.json``, and their SHA-256
sidecars; it never searches result trees or reconstructs scientific evidence.

Example
-------
Run after the append-only final report has been published::

    uv run python scripts/build_final_editorial_artifacts.py \
      --generated-at 2026-08-01

The date is explicit rather than taken from the wall clock so identical inputs
and arguments produce identical bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPORT_SCHEMA = "cadence.reporting.v1"
COMPLETION_SCHEMA = "cadence.reporting_completion.v1"
SITE_SCHEMA = "cadence-site-final-outcome-v1"
STATUSES = ("PASS", "FAIL", "NOT_EVALUATED")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GATE_IDS = tuple(range(1, 9))

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = REPOSITORY_ROOT / "results" / "releases" / "final" / "summary.json"
DEFAULT_COMPLETION = REPOSITORY_ROOT / "results" / "releases" / "final" / "report.complete.json"
DEFAULT_SITE_OUTPUT = REPOSITORY_ROOT / "site" / "data" / "final-outcome.json"
DEFAULT_FIGURE_DIRECTORY = REPOSITORY_ROOT / "paper" / "figures"
DEFAULT_PAPER_INCLUDE = REPOSITORY_ROOT / "paper" / "includes" / "final_outcome.tex"

UNIT_FIGURE_STEM = "final_unit_skill"
GATE_FIGURE_STEM = "final_gate_matrix"


class AnalysisSpec(NamedTuple):
    analysis_id: str
    dataset: str
    cohort: str
    replication_unit: str
    site_id: str
    label: str
    role: str


ANALYSIS_SPECS = (
    AnalysisSpec(
        analysis_id="allen_vbo:locked",
        dataset="allen_vbo",
        cohort="locked",
        replication_unit="target_animal",
        site_id="allen",
        label="Allen image omissions",
        role="Primary biological evaluation (headline)",
    ),
    AnalysisSpec(
        analysis_id="icms:randomized_n5",
        dataset="icms",
        cohort="randomized_n5",
        replication_unit="target_animal",
        site_id="icms-randomized",
        label="ICMS randomized five-mouse cohort",
        role="Exploratory randomized biological analysis (non-headline)",
    ),
    AnalysisSpec(
        analysis_id="teacher:locked",
        dataset="teacher",
        cohort="locked",
        replication_unit="teacher_world",
        site_id="teacher",
        label="Synthetic teacher benchmark",
        role="Procedural world-level evaluation (non-headline)",
    ),
    AnalysisSpec(
        analysis_id="icms:absolute_only",
        dataset="icms",
        cohort="absolute_only",
        replication_unit="target_animal",
        site_id="icms-absolute",
        label="ICMS83 absolute trajectories",
        role="Descriptive absolute-trajectory analysis (non-headline)",
    ),
)
SPECS_BY_ID = {spec.analysis_id: spec for spec in ANALYSIS_SPECS}

SKILL_METRICS = {
    "allen_vbo:locked": {
        "neural": ("neural_causal_skill",),
        "behavior": ("running_causal_skill",),
    },
    "icms:randomized_n5": {
        "neural": (
            "neural_causal_skill",
            "neural_causal_skill_equal_session",
            "spike_causal_skill",
        ),
        "behavior": (
            "behavior_causal_skill",
            "behavior_causal_skill_equal_session",
            "wheel_causal_skill",
        ),
    },
    "teacher:locked": {
        "neural": ("neural_condition_averaged_causal_skill",),
        "behavior": ("behavior_condition_averaged_causal_skill",),
    },
}

STATUS_COLORS = {
    "PASS": "#147D64",
    "FAIL": "#B63A3A",
    "NOT_EVALUATED": "#98A0AA",
}
STATUS_SHORT = {
    "PASS": "PASS",
    "FAIL": "FAIL",
    "NOT_EVALUATED": "NE",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read required artifact {path}: {error}") from error
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload, digest


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def _verify_sidecar(path: Path, digest: str) -> None:
    sidecar = _sidecar_path(path)
    try:
        fields = sidecar.read_text(encoding="utf-8").strip().split()
    except OSError as error:
        raise ValueError(f"cannot read SHA-256 sidecar {sidecar}: {error}") from error
    if fields != [digest, path.name]:
        raise ValueError(f"invalid SHA-256 sidecar for {path}: expected {digest}  {path.name}")


def _verified_report_inputs(
    summary_path: Path,
    completion_path: Path,
) -> tuple[dict[str, Any], str]:
    if summary_path.name != "summary.json":
        raise ValueError("the report summary must be named summary.json")
    if completion_path.name != "report.complete.json":
        raise ValueError("the completion manifest must be named report.complete.json")

    summary, summary_digest = _load_json_object(summary_path)
    _verify_sidecar(summary_path, summary_digest)
    completion, completion_digest = _load_json_object(completion_path)
    _verify_sidecar(completion_path, completion_digest)

    if completion.get("schema") != COMPLETION_SCHEMA:
        raise ValueError(f"report completion schema must be {COMPLETION_SCHEMA!r}")
    if completion.get("append_only") is not True:
        raise ValueError("report completion manifest must declare append_only=true")
    artifacts = _require_mapping(completion.get("artifacts"), "completion.artifacts")
    declared_digest = artifacts.get("summary.json")
    if not isinstance(declared_digest, str) or not SHA256_PATTERN.fullmatch(declared_digest):
        raise ValueError("completion.artifacts['summary.json'] must be a lowercase SHA-256 digest")
    if declared_digest != summary_digest:
        raise ValueError("summary.json digest does not match report.complete.json")

    _validate_report(summary)
    return summary, summary_digest


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _require_status(value: Any, label: str) -> str:
    if value not in STATUSES:
        raise ValueError(f"{label} must be one of {STATUSES}")
    return str(value)


def _nullable_finite(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a finite number or null")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number or null")
    return result


def _validate_summary_shape(value: Any, label: str) -> None:
    summary = _require_mapping(value, label)
    _nullable_finite(summary.get("estimate"), f"{label}.estimate")
    lower = _nullable_finite(summary.get("ci_lower"), f"{label}.ci_lower")
    upper = _nullable_finite(summary.get("ci_upper"), f"{label}.ci_upper")
    if (lower is None) != (upper is None):
        raise ValueError(f"{label} CI endpoints must both be numeric or both be null")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError(f"{label} CI endpoints are reversed")
    if "n" in summary:
        _require_nonnegative_int(summary["n"], f"{label}.n")
    confidence = summary.get("confidence")
    if lower is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, int | float)
        or not math.isclose(float(confidence), 0.95)
    ):
        raise ValueError(f"{label} must identify its exported interval as a 95% CI")


def _validate_method_summaries(value: Any, label: str) -> None:
    methods = _require_mapping(value, label)
    for method, metrics_value in methods.items():
        _require_string(method, f"{label} method name")
        metrics = _require_mapping(metrics_value, f"{label}.{method}")
        for metric, summary in metrics.items():
            _require_string(metric, f"{label}.{method} metric name")
            _validate_summary_shape(summary, f"{label}.{method}.{metric}")


def _validate_primary_summaries(value: Any, label: str) -> None:
    summaries = _require_mapping(value, label)
    allowed = {
        "neural_skill",
        "behavior_skill",
        "neural_baseline_gain",
        "behavior_baseline_gain",
    }
    unexpected = set(summaries) - allowed
    if unexpected:
        raise ValueError(f"{label} has unexpected summaries: {sorted(unexpected)}")
    for name, summary in summaries.items():
        _validate_summary_shape(summary, f"{label}.{name}")


def _validate_target_rows(
    rows_value: Any,
    *,
    analysis_id: str,
    expected_count: int,
) -> None:
    rows = _require_list(rows_value, f"{analysis_id}.target_rows")
    if len(rows) != expected_count:
        raise ValueError(
            f"{analysis_id}.target_rows has {len(rows)} rows; expected {expected_count}"
        )
    unit_ids: set[str] = set()
    for index, row_value in enumerate(rows):
        label = f"{analysis_id}.target_rows[{index}]"
        row = _require_mapping(row_value, label)
        unit_id = _require_string(row.get("unit_id"), f"{label}.unit_id")
        if unit_id in unit_ids:
            raise ValueError(f"{analysis_id}.target_rows repeats unit_id {unit_id!r}")
        unit_ids.add(unit_id)
        methods = _require_mapping(row.get("methods"), f"{label}.methods")
        proposed = _require_mapping(methods.get("proposed"), f"{label}.methods.proposed")
        for metric, metric_value in proposed.items():
            _require_string(metric, f"{label}.methods.proposed metric name")
            if isinstance(metric_value, bool) or metric_value is None:
                continue
            if isinstance(metric_value, int | float) and not math.isfinite(float(metric_value)):
                raise ValueError(f"{label}.methods.proposed.{metric} must be finite when numeric")


def _validate_inference_rows(
    rows_value: Any,
    *,
    analysis_id: str,
    expected_count: int,
) -> None:
    rows = _require_list(rows_value, f"{analysis_id}.inference_rows")
    proposed_units: set[str] = set()
    seen: set[tuple[str, str]] = set()
    for index, row_value in enumerate(rows):
        label = f"{analysis_id}.inference_rows[{index}]"
        row = _require_mapping(row_value, label)
        unit_id = _require_string(row.get("unit_id"), f"{label}.unit_id")
        method = _require_string(row.get("method"), f"{label}.method")
        key = (unit_id, method)
        if key in seen:
            raise ValueError(f"{analysis_id}.inference_rows repeats {key!r}")
        seen.add(key)
        metrics = _require_mapping(row.get("metrics"), f"{label}.metrics")
        if method == "proposed":
            proposed_units.add(unit_id)
            for metric, metric_value in metrics.items():
                _require_string(metric, f"{label}.metrics metric name")
                if isinstance(metric_value, bool) or metric_value is None:
                    continue
                if isinstance(metric_value, int | float) and not math.isfinite(float(metric_value)):
                    raise ValueError(f"{label}.metrics.{metric} must be finite")
    if len(proposed_units) != expected_count:
        raise ValueError(
            f"{analysis_id}.inference_rows has {len(proposed_units)} proposed "
            f"independent units; expected {expected_count}"
        )


def _validate_analysis(analysis_value: Any, spec: AnalysisSpec) -> None:
    analysis = _require_mapping(analysis_value, spec.analysis_id)
    if analysis.get("dataset") != spec.dataset:
        raise ValueError(f"{spec.analysis_id}.dataset must be {spec.dataset!r}")
    if analysis.get("cohort") != spec.cohort:
        raise ValueError(f"{spec.analysis_id}.cohort must be {spec.cohort!r}")
    if analysis.get("replication_unit") != spec.replication_unit:
        raise ValueError(f"{spec.analysis_id}.replication_unit must be {spec.replication_unit!r}")

    independent = _require_nonnegative_int(
        analysis.get("n_independent_units"),
        f"{spec.analysis_id}.n_independent_units",
    )
    nested = _require_nonnegative_int(
        analysis.get("n_nested_target_units"),
        f"{spec.analysis_id}.n_nested_target_units",
    )
    completeness = _require_mapping(
        analysis.get("cohort_completeness"),
        f"{spec.analysis_id}.cohort_completeness",
    )
    if not isinstance(completeness.get("complete"), bool):
        raise ValueError(f"{spec.analysis_id}.cohort_completeness.complete must be boolean")
    if not isinstance(completeness.get("gates_evaluable"), bool):
        raise ValueError(f"{spec.analysis_id}.cohort_completeness.gates_evaluable must be boolean")
    observed, expected = _cohort_unit_counts(analysis, spec.analysis_id)
    if completeness["complete"] and observed != expected:
        raise ValueError(
            f"{spec.analysis_id} declares a complete cohort with "
            f"{observed} observed units but {expected} expected"
        )
    if not completeness["complete"] and completeness["gates_evaluable"]:
        raise ValueError(f"{spec.analysis_id} is incomplete but declares gates_evaluable=true")

    _validate_method_summaries(
        analysis.get("method_summaries"),
        f"{spec.analysis_id}.method_summaries",
    )
    _validate_primary_summaries(
        analysis.get("primary_summaries"),
        f"{spec.analysis_id}.primary_summaries",
    )
    _validate_target_rows(
        analysis.get("target_rows"),
        analysis_id=spec.analysis_id,
        expected_count=nested,
    )
    _validate_inference_rows(
        analysis.get("inference_rows"),
        analysis_id=spec.analysis_id,
        expected_count=independent,
    )

    conjunction = _require_mapping(analysis.get("conjunction"), f"{spec.analysis_id}.conjunction")
    _require_status(
        conjunction.get("overall_status"),
        f"{spec.analysis_id}.conjunction.overall_status",
    )
    gates = _require_list(conjunction.get("gates"), f"{spec.analysis_id}.conjunction.gates")
    observed_ids: list[int] = []
    for index, gate_value in enumerate(gates):
        label = f"{spec.analysis_id}.conjunction.gates[{index}]"
        gate = _require_mapping(gate_value, label)
        gate_id = _require_nonnegative_int(gate.get("gate_id"), f"{label}.gate_id")
        observed_ids.append(gate_id)
        _require_string(gate.get("criterion"), f"{label}.criterion")
        _require_status(gate.get("status"), f"{label}.status")
        _require_mapping(gate.get("details"), f"{label}.details")
    if tuple(observed_ids) != GATE_IDS:
        raise ValueError(f"{spec.analysis_id}.conjunction.gates must contain ordered gate IDs 1-8")
    if not completeness["complete"]:
        # This authenticates a frozen ``cadence.reporting.v1`` producer
        # invariant: _conjunction short-circuits every ineligible cohort to NE.
        # Rendering still copies each accepted status verbatim below.
        gate_statuses = [str(gate["status"]) for gate in gates]
        if conjunction["overall_status"] != "NOT_EVALUATED" or set(gate_statuses) != {
            "NOT_EVALUATED"
        }:
            raise ValueError(
                f"{spec.analysis_id} is incomplete but its conjunction gates are "
                "not uniformly NOT_EVALUATED"
            )

    primary = _require_mapping(
        analysis.get("primary_summaries"), f"{spec.analysis_id}.primary_summaries"
    )
    if completeness["gates_evaluable"]:
        for name in ("neural_skill", "behavior_skill"):
            if name not in primary:
                raise ValueError(
                    f"{spec.analysis_id} is gate-evaluable but lacks primary_summaries.{name}"
                )


def _cohort_unit_counts(
    analysis: Mapping[str, Any],
    analysis_id: str,
) -> tuple[int, int]:
    """Return validated observed/expected independent-unit counts.

    ``cohort_completeness.expected_units`` is the reporter-native field.
    ``expected_n`` is retained as a validated fallback for older complete
    reports and must agree whenever both fields are populated.
    """

    observed = _require_nonnegative_int(
        analysis.get("n_independent_units"),
        f"{analysis_id}.n_independent_units",
    )
    completeness = _require_mapping(
        analysis.get("cohort_completeness"),
        f"{analysis_id}.cohort_completeness",
    )

    observed_proposed_value = completeness.get("observed_proposed_units")
    if observed_proposed_value is not None:
        observed_proposed = _require_nonnegative_int(
            observed_proposed_value,
            f"{analysis_id}.cohort_completeness.observed_proposed_units",
        )
        if observed_proposed != observed:
            raise ValueError(
                f"{analysis_id}.cohort_completeness.observed_proposed_units "
                "must match n_independent_units"
            )

    completeness_expected: int | None = None
    if completeness.get("expected_units") is not None:
        completeness_expected = _require_nonnegative_int(
            completeness["expected_units"],
            f"{analysis_id}.cohort_completeness.expected_units",
        )

    expected_n: int | None = None
    if analysis.get("expected_n") is not None:
        expected_n = _require_nonnegative_int(
            analysis["expected_n"],
            f"{analysis_id}.expected_n",
        )

    if completeness_expected is None and expected_n is None:
        raise ValueError(
            f"{analysis_id} must provide cohort_completeness.expected_units or expected_n"
        )
    if (
        completeness_expected is not None
        and expected_n is not None
        and completeness_expected != expected_n
    ):
        raise ValueError(
            f"{analysis_id}.cohort_completeness.expected_units does not match expected_n"
        )
    expected = completeness_expected if completeness_expected is not None else int(expected_n)
    if expected < observed:
        raise ValueError(
            f"{analysis_id} has {observed} observed independent units but only {expected} expected"
        )
    return observed, expected


def _cohort_complete(analysis: Mapping[str, Any], analysis_id: str) -> bool:
    completeness = _require_mapping(
        analysis["cohort_completeness"],
        f"{analysis_id}.cohort_completeness",
    )
    complete = completeness["complete"]
    if not isinstance(complete, bool):
        raise ValueError(f"{analysis_id}.cohort_completeness.complete must be boolean")
    return complete


def _editorial_incomplete(analysis: Mapping[str, Any], analysis_id: str) -> bool:
    """Identify cohorts that should carry an editorial incomplete warning.

    ICMS83's absolute-only cohort is intentionally non-gate-evaluable and the
    reporter therefore never calls it ``complete``. It is not a partial cohort
    when its sole expected unit is present, so preserve its descriptive label.
    """

    if _cohort_complete(analysis, analysis_id):
        return False
    observed, expected = _cohort_unit_counts(analysis, analysis_id)
    return analysis_id != "icms:absolute_only" or observed < expected


def _validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError(f"summary schema_version must be {REPORT_SCHEMA!r}")
    analyses = _require_mapping(report.get("analyses"), "summary.analyses")
    missing = [spec.analysis_id for spec in ANALYSIS_SPECS if spec.analysis_id not in analyses]
    if missing:
        raise ValueError(f"summary is missing required analyses: {missing}")
    for spec in ANALYSIS_SPECS:
        _validate_analysis(analyses[spec.analysis_id], spec)


def _analysis(report: Mapping[str, Any], analysis_id: str) -> Mapping[str, Any]:
    analyses = _require_mapping(report["analyses"], "summary.analyses")
    return _require_mapping(analyses[analysis_id], analysis_id)


def _summary_or_none(
    analysis: Mapping[str, Any],
    *path: str,
) -> Mapping[str, Any] | None:
    value: Any = analysis
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value if isinstance(value, Mapping) else None


def _summary_triplet(
    summary: Mapping[str, Any] | None,
) -> tuple[float | None, float | None, float | None]:
    if summary is None:
        return None, None, None
    return (
        _nullable_finite(summary.get("estimate"), "summary.estimate"),
        _nullable_finite(summary.get("ci_lower"), "summary.ci_lower"),
        _nullable_finite(summary.get("ci_upper"), "summary.ci_upper"),
    )


def _gate(analysis: Mapping[str, Any], gate_id: int) -> Mapping[str, Any]:
    conjunction = _require_mapping(analysis["conjunction"], "analysis.conjunction")
    gates = _require_list(conjunction["gates"], "analysis.conjunction.gates")
    return _require_mapping(gates[gate_id - 1], f"gate {gate_id}")


def _replication_text(analysis: Mapping[str, Any], spec: AnalysisSpec) -> str:
    independent, expected = _cohort_unit_counts(analysis, spec.analysis_id)
    nested = int(analysis["n_nested_target_units"])
    if _editorial_incomplete(analysis, spec.analysis_id):
        prefix = f"{independent} observed of {expected}; incomplete; gates NOT_EVALUATED"
        if spec.replication_unit == "teacher_world":
            world_word = "world" if independent == 1 else "worlds"
            target_word = "target unit" if nested == 1 else "target units"
            return f"{prefix}; independent teacher {world_word}; {nested} nested {target_word}"
        animal_word = "animal" if independent == 1 else "animals"
        return f"{prefix}; independent target {animal_word}"
    if spec.replication_unit == "teacher_world":
        world_word = "world" if independent == 1 else "worlds"
        target_word = "target unit" if nested == 1 else "target units"
        return f"{independent} independent teacher {world_word}; {nested} nested {target_word}"
    animal_word = "animal" if independent == 1 else "animals"
    return f"{independent} independent target {animal_word}"


def _endpoint(
    label: str,
    summary: Mapping[str, Any] | None,
    *,
    status: str | None = None,
    note: str,
) -> dict[str, Any]:
    estimate, lower, upper = _summary_triplet(summary)
    result: dict[str, Any] = {
        "label": label,
        "value": estimate,
        "ciLow": lower,
        "ciHigh": upper,
        "note": note,
    }
    if status is not None:
        result["status"] = status
    return result


def _canonical_gate_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gate_id in GATE_IDS:
        gate = _gate(analysis, gate_id)
        details = json.dumps(
            gate["details"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        rows.append(
            {
                "number": gate_id,
                "title": str(gate["criterion"]),
                "status": str(gate["status"]),
                "evidence": f"Canonical reporter details: {details}",
            }
        )
    return rows


def _dataset_endpoints(
    analysis: Mapping[str, Any],
    spec: AnalysisSpec,
) -> list[dict[str, Any]]:
    if spec.analysis_id in {"allen_vbo:locked", "icms:randomized_n5"}:
        if not _cohort_complete(analysis, spec.analysis_id):
            behavior_label = (
                "Running-speed causal skill"
                if spec.analysis_id == "allen_vbo:locked"
                else "Primary behavior causal skill"
            )
            note = (
                "Incomplete cohort: primary and gate-linked summaries are withheld; "
                "partial method summaries are descriptive only."
            )
            return [
                _endpoint(
                    "Neural causal skill",
                    None,
                    status=str(_gate(analysis, 2)["status"]),
                    note=note,
                ),
                _endpoint(
                    behavior_label,
                    None,
                    status=str(_gate(analysis, 3)["status"]),
                    note=note,
                ),
            ]
        primary = _require_mapping(
            analysis["primary_summaries"], f"{spec.analysis_id}.primary_summaries"
        )
        return [
            _endpoint(
                "Neural causal skill",
                _summary_or_none(primary, "neural_skill"),
                status=str(_gate(analysis, 2)["status"]),
                note="Gate 2 input: equal-unit 95% bootstrap interval.",
            ),
            _endpoint(
                (
                    "Running-speed causal skill"
                    if spec.analysis_id == "allen_vbo:locked"
                    else "Primary behavior causal skill"
                ),
                _summary_or_none(primary, "behavior_skill"),
                status=str(_gate(analysis, 3)["status"]),
                note="Gate 3 input: equal-unit 95% bootstrap interval.",
            ),
        ]

    methods = _require_mapping(analysis["method_summaries"], f"{spec.analysis_id}.method_summaries")
    proposed = _summary_or_none(methods, "proposed")
    if spec.analysis_id == "teacher:locked":
        return [
            _endpoint(
                "Neural condition-averaged causal skill",
                _summary_or_none(proposed or {}, "neural_condition_averaged_causal_skill"),
                note=(
                    "Descriptive world-level interval; non-headline and not "
                    "biological gate evidence."
                ),
            ),
            _endpoint(
                "Behavior condition-averaged causal skill",
                _summary_or_none(proposed or {}, "behavior_condition_averaged_causal_skill"),
                note=(
                    "Descriptive world-level interval; non-headline and not "
                    "biological gate evidence."
                ),
            ),
        ]
    return [
        _endpoint(
            "Absolute neural NRMSE (equal session)",
            _summary_or_none(proposed or {}, "absolute_neural_nrmse_equal_session"),
            note="Descriptive absolute error for ICMS83; not a randomized causal skill.",
        ),
        _endpoint(
            "Absolute behavior NRMSE (equal session)",
            _summary_or_none(proposed or {}, "absolute_behavior_nrmse_equal_session"),
            note="Descriptive absolute error for ICMS83; not a randomized causal skill.",
        ),
    ]


def build_site_payload(
    report: Mapping[str, Any],
    *,
    summary_digest: str,
    release_label: str,
    generated_at: str,
    protocol_version: str,
) -> dict[str, Any]:
    allen = _analysis(report, "allen_vbo:locked")
    allen_status = str(
        _require_mapping(allen["conjunction"], "allen conjunction")["overall_status"]
    )
    datasets: list[dict[str, Any]] = []
    for spec in ANALYSIS_SPECS:
        analysis = _analysis(report, spec.analysis_id)
        conjunction = _require_mapping(analysis["conjunction"], f"{spec.analysis_id}.conjunction")
        status = str(conjunction["overall_status"])
        datasets.append(
            {
                "id": spec.site_id,
                "label": spec.label,
                "evidentialRole": spec.role,
                "status": status,
                "replication": _replication_text(analysis, spec),
                "summary": (
                    f"Machine-derived reporter status: {status}; "
                    f"{_replication_text(analysis, spec)}."
                ),
                "endpoints": _dataset_endpoints(analysis, spec),
                "gates": _canonical_gate_rows(analysis),
            }
        )
    return {
        "schema": SITE_SCHEMA,
        "release": {
            "label": release_label,
            "generatedAt": generated_at,
            "protocolVersion": protocol_version,
            "sourceSummarySha256": summary_digest,
        },
        "headline": {
            "status": allen_status,
            "summary": (
                "Allen primary biological evaluation reporter status: "
                f"{allen_status}; "
                f"{_replication_text(allen, SPECS_BY_ID['allen_vbo:locked'])}."
            ),
        },
        "datasets": datasets,
    }


def _first_finite_metric(
    metrics: Mapping[str, Any],
    aliases: Sequence[str],
) -> float | None:
    for alias in aliases:
        if alias not in metrics:
            continue
        value = metrics[alias]
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        result = float(value)
        if math.isfinite(result):
            return result
    return None


def _independent_skill_values(
    analysis: Mapping[str, Any],
    analysis_id: str,
    domain: str,
) -> list[tuple[str, float]]:
    aliases = SKILL_METRICS[analysis_id][domain]
    values: list[tuple[str, float]] = []
    if analysis_id == "teacher:locked":
        rows = _require_list(analysis["inference_rows"], f"{analysis_id}.inference_rows")
        for row_value in rows:
            row = _require_mapping(row_value, f"{analysis_id}.inference row")
            if row["method"] != "proposed":
                continue
            metrics = _require_mapping(row["metrics"], f"{analysis_id}.inference metrics")
            value = _first_finite_metric(metrics, aliases)
            if value is not None:
                values.append((str(row["unit_id"]), value))
    else:
        rows = _require_list(analysis["target_rows"], f"{analysis_id}.target_rows")
        for row_value in rows:
            row = _require_mapping(row_value, f"{analysis_id}.target row")
            methods = _require_mapping(row["methods"], f"{analysis_id}.target methods")
            metrics = _require_mapping(
                methods["proposed"], f"{analysis_id}.target proposed metrics"
            )
            value = _first_finite_metric(metrics, aliases)
            if value is not None:
                values.append((str(row["unit_id"]), value))
    return sorted(values)


def _deterministic_jitter(analysis_id: str, domain: str, unit_id: str) -> float:
    digest = hashlib.sha256(f"{analysis_id}\0{domain}\0{unit_id}".encode()).digest()
    fraction = int.from_bytes(digest[:2], "big") / 65535.0
    return (fraction - 0.5) * 0.18


def _save_figure(fig: plt.Figure, png_path: Path, pdf_path: Path) -> None:
    title = fig._suptitle.get_text() if fig._suptitle is not None else "CADENCE final outcome"
    fig.savefig(
        png_path,
        format="png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "CADENCE final editorial artifact builder"},
    )
    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Title": title,
            "Author": "CADENCE final editorial artifact builder",
            "Subject": "Machine-derived final report visualization",
            "Creator": "CADENCE final editorial artifact builder",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )


def render_unit_skill_figure(report: Mapping[str, Any], directory: Path) -> None:
    rows = (
        ("allen_vbo:locked", "Allen\nprimary headline", "#355C9A"),
        ("icms:randomized_n5", "ICMS randomized\nexploratory", "#C67A20"),
        ("teacher:locked", "Teacher worlds\nprocedural", "#76528B"),
    )
    domains = (("neural", "Neural causal skill"), ("behavior", "Primary behavior skill"))
    values_by_panel: dict[tuple[str, str], list[tuple[str, float]]] = {}
    limits: list[float] = [0.0]
    for analysis_id, _, _ in rows:
        analysis = _analysis(report, analysis_id)
        for domain, _ in domains:
            values = _independent_skill_values(analysis, analysis_id, domain)
            values_by_panel[(analysis_id, domain)] = values
            limits.extend(value for _, value in values)
            if analysis_id != "teacher:locked" and _cohort_complete(analysis, analysis_id):
                summary = _summary_or_none(analysis, "primary_summaries", f"{domain}_skill")
                limits.extend(value for value in _summary_triplet(summary) if value is not None)

    low = min(limits)
    high = max(limits)
    span = max(high - low, 0.2)
    y_min = low - 0.12 * span
    y_max = high + 0.16 * span

    rc = {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig, grid = plt.subplots(
            3,
            3,
            figsize=(7.6, 7.4),
            sharey=False,
            gridspec_kw={"width_ratios": (0.27, 1.0, 1.0)},
        )
        label_axes = grid[:, 0]
        axes = grid[:, 1:]
        fig.suptitle(
            "Independent-unit causal skill",
            fontsize=12,
            fontweight="bold",
            y=0.995,
        )
        for row_index, (analysis_id, row_label, color) in enumerate(rows):
            analysis = _analysis(report, analysis_id)
            observed, expected = _cohort_unit_counts(analysis, analysis_id)
            complete = _cohort_complete(analysis, analysis_id)
            if _editorial_incomplete(analysis, analysis_id):
                row_label = (
                    f"{row_label}\n{observed} observed of {expected}\n"
                    "incomplete; gates NOT_EVALUATED"
                )
            label_ax = label_axes[row_index]
            label_ax.axis("off")
            label_ax.text(
                0.95,
                0.5,
                row_label,
                transform=label_ax.transAxes,
                ha="right",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="#2D343C",
                linespacing=1.3,
            )
            for column_index, (domain, column_label) in enumerate(domains):
                ax = axes[row_index][column_index]
                values = values_by_panel[(analysis_id, domain)]
                ax.axhline(0.0, color="#5D6570", linewidth=0.8, linestyle="--", zorder=0)
                ax.grid(axis="y", color="#D8DCE2", linewidth=0.55, alpha=0.8)
                ax.set_axisbelow(True)
                for unit_id, value in values:
                    ax.scatter(
                        _deterministic_jitter(analysis_id, domain, unit_id),
                        value,
                        s=28,
                        color=color,
                        edgecolor="white",
                        linewidth=0.45,
                        alpha=0.9,
                        zorder=3,
                    )
                if analysis_id != "teacher:locked" and complete:
                    summary = _summary_or_none(analysis, "primary_summaries", f"{domain}_skill")
                    estimate, lower, upper = _summary_triplet(summary)
                    if estimate is not None:
                        if lower is not None and upper is not None:
                            ax.vlines(
                                0.43,
                                lower,
                                upper,
                                color="#111820",
                                linewidth=1.4,
                                zorder=4,
                            )
                            ax.hlines(
                                [lower, upper],
                                0.415,
                                0.445,
                                color="#111820",
                                linewidth=1.2,
                                zorder=4,
                            )
                            ax.scatter(
                                [0.43],
                                [estimate],
                                marker="D",
                                s=34,
                                color="#111820",
                                zorder=5,
                            )
                        else:
                            ax.scatter(
                                [0.43],
                                [estimate],
                                marker="D",
                                s=34,
                                color="#111820",
                                zorder=4,
                            )
                    elif not values:
                        ax.text(
                            0.5,
                            0.5,
                            "primary skill unavailable",
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                            color="#59616B",
                        )
                elif analysis_id == "teacher:locked":
                    ax.text(
                        0.97,
                        0.05,
                        "descriptive only;\nno gate-linked CI",
                        transform=ax.transAxes,
                        ha="right",
                        va="bottom",
                        color="#59616B",
                        fontsize=7,
                    )
                else:
                    ax.text(
                        0.97,
                        0.05,
                        "incomplete cohort;\nprimary CI withheld",
                        transform=ax.transAxes,
                        ha="right",
                        va="bottom",
                        color="#59616B",
                        fontsize=7,
                    )
                ax.text(
                    0.03,
                    0.94,
                    f"exported n={len(values)}/{expected}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7,
                    color="#59616B",
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.78,
                    },
                )
                ax.set_xlim(-0.18, 0.62)
                ax.set_ylim(y_min, y_max)
                if column_index == 1:
                    ax.tick_params(axis="y", labelleft=False)
                ax.set_xticks([0.0, 0.43])
                ax.set_xticklabels(
                    [
                        "worlds" if analysis_id == "teacher:locked" else "target animals",
                        ("mean + 95% CI" if analysis_id != "teacher:locked" and complete else ""),
                    ]
                )
                if row_index == 0:
                    ax.set_title(column_label, fontweight="bold", pad=8)
                if column_index == 0:
                    ax.set_ylabel("Causal skill")
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)
                ax.spines["left"].set_color("#9EA5AE")
                ax.spines["bottom"].set_color("#9EA5AE")
        legend = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#5978A8",
                markeredgecolor="white",
                markersize=6,
                label="exported independent unit",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="#111820",
                markersize=5,
                label="reporter primary mean and 95% CI",
            ),
        ]
        fig.legend(
            handles=legend,
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.005),
        )
        fig.text(
            0.5,
            0.045,
            (
                "Allen alone supplies the headline. ICMS is exploratory; teacher "
                "world values are procedural and non-headline. ICMS83 has no "
                "randomized causal-skill panel."
            ),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#4C545E",
        )
        fig.tight_layout(rect=(0.01, 0.085, 1.0, 0.925), h_pad=1.6, w_pad=0.65)
        _save_figure(
            fig,
            directory / f"{UNIT_FIGURE_STEM}.png",
            directory / f"{UNIT_FIGURE_STEM}.pdf",
        )
        plt.close(fig)


def render_gate_matrix_figure(report: Mapping[str, Any], directory: Path) -> None:
    rows = (
        ("allen_vbo:locked", "Allen · primary headline"),
        ("icms:randomized_n5", "ICMS randomized · exploratory"),
        ("teacher:locked", "Teacher · procedural, non-headline"),
        ("icms:absolute_only", "ICMS83 · descriptive, non-headline"),
    )
    columns = [*GATE_IDS, "Overall"]
    status_rows: list[list[str]] = []
    row_labels: list[str] = []
    for analysis_id, label in rows:
        analysis = _analysis(report, analysis_id)
        observed, expected = _cohort_unit_counts(analysis, analysis_id)
        if _editorial_incomplete(analysis, analysis_id):
            label = f"{label} · {observed} observed of {expected}; incomplete; gates NOT_EVALUATED"
        row_labels.append(label)
        conjunction = _require_mapping(analysis["conjunction"], f"{analysis_id}.conjunction")
        statuses = [str(_gate(analysis, gate_id)["status"]) for gate_id in GATE_IDS]
        statuses.append(str(conjunction["overall_status"]))
        status_rows.append(statuses)
    color_index = {status: index for index, status in enumerate(STATUSES)}
    matrix = [[color_index[status] for status in statuses] for statuses in status_rows]
    cmap = matplotlib.colors.ListedColormap([STATUS_COLORS[status] for status in STATUSES])
    norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    rc = {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(8.0, 3.5))
        fig.suptitle(
            "Frozen conjunction gate matrix",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )
        ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels(
            [f"Gate {value}" if isinstance(value, int) else value for value in columns]
        )
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(row_labels)
        ax.tick_params(axis="both", length=0)
        for row_index, statuses in enumerate(status_rows):
            for column_index, status in enumerate(statuses):
                text_color = "white" if status in {"PASS", "FAIL"} else "#20262D"
                ax.text(
                    column_index,
                    row_index,
                    STATUS_SHORT[status],
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=7.5,
                    fontweight="bold",
                )
        ax.axvline(7.5, color="white", linewidth=3.0)
        ax.set_xticks([value - 0.5 for value in range(1, len(columns))], minor=True)
        ax.set_yticks([value - 0.5 for value in range(1, len(rows))], minor=True)
        ax.grid(which="minor", color="white", linewidth=1.25)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        legend = [
            Patch(facecolor=STATUS_COLORS[status], edgecolor="none", label=status)
            for status in STATUSES
        ]
        ax.legend(
            handles=legend,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.19),
            ncol=3,
            frameon=False,
        )
        fig.text(
            0.5,
            0.01,
            (
                "Every cell and overall state is copied verbatim from the reporter; "
                "overall status is not recomputed downstream."
            ),
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#4C545E",
        )
        fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.92))
        _save_figure(
            fig,
            directory / f"{GATE_FIGURE_STEM}.png",
            directory / f"{GATE_FIGURE_STEM}.pdf",
        )
        plt.close(fig)


def _latex_escape(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _latex_number(value: float | None) -> str:
    if value is None:
        return r"\textemdash{}"
    encoded = json.dumps(value, allow_nan=False)
    if "e" not in encoded and "E" not in encoded:
        return encoded
    mantissa, exponent = re.split("[eE]", encoded)
    return rf"{mantissa}\mathbin{{\times}}10^{{{int(exponent)}}}"


def _latex_ci(summary: Mapping[str, Any] | None, *, label: str) -> str:
    estimate, lower, upper = _summary_triplet(summary)
    if estimate is None:
        return rf"\textemdash{{}} ({_latex_escape(label)})"
    estimate_text = _latex_number(estimate)
    if lower is None or upper is None:
        return (
            rf"\ensuremath{{{estimate_text}}} "
            rf"(95\% CI \textemdash{{}}; {_latex_escape(label)})"
        )
    return (
        rf"\ensuremath{{{estimate_text}\;[{_latex_number(lower)},\,"
        rf"{_latex_number(upper)}]}} ({_latex_escape(label)})"
    )


def _paper_endpoint_summaries(
    analysis: Mapping[str, Any],
    analysis_id: str,
) -> tuple[tuple[Mapping[str, Any] | None, str], tuple[Mapping[str, Any] | None, str]]:
    if analysis_id in {"allen_vbo:locked", "icms:randomized_n5"}:
        if not _cohort_complete(analysis, analysis_id):
            return (
                (None, "withheld: incomplete cohort; not gate evidence"),
                (None, "withheld: incomplete cohort; not gate evidence"),
            )
        return (
            (_summary_or_none(analysis, "primary_summaries", "neural_skill"), "causal skill"),
            (
                _summary_or_none(analysis, "primary_summaries", "behavior_skill"),
                "causal skill",
            ),
        )
    if analysis_id == "teacher:locked":
        return (
            (
                _summary_or_none(
                    analysis,
                    "method_summaries",
                    "proposed",
                    "neural_condition_averaged_causal_skill",
                ),
                "descriptive causal skill",
            ),
            (
                _summary_or_none(
                    analysis,
                    "method_summaries",
                    "proposed",
                    "behavior_condition_averaged_causal_skill",
                ),
                "descriptive causal skill",
            ),
        )
    return (
        (
            _summary_or_none(
                analysis,
                "method_summaries",
                "proposed",
                "absolute_neural_nrmse_equal_session",
            ),
            "descriptive absolute NRMSE",
        ),
        (
            _summary_or_none(
                analysis,
                "method_summaries",
                "proposed",
                "absolute_behavior_nrmse_equal_session",
            ),
            "descriptive absolute NRMSE",
        ),
    )


def render_paper_include(report: Mapping[str, Any], summary_digest: str) -> str:
    lines = [
        "% AUTO-GENERATED by scripts/build_final_editorial_artifacts.py.",
        "% Do not edit by hand or substitute development values.",
        f"% Authenticated source summary SHA-256: {summary_digest}",
        r"\begin{table}[t]",
        r"  \centering",
        (
            r"  \caption{\textbf{Machine-derived final outcome ledger.} "
            r"Statuses are copied verbatim from the frozen reporter. Allen alone "
            r"is the headline biological evaluation; randomized ICMS is exploratory, "
            r"and teacher/ICMS83 results are non-headline. Brackets are exported "
            r"95\% intervals; missing intervals remain unavailable.}"
        ),
        r"  \label{tab:final-outcome-machine}",
        r"  \scriptsize",
        r"  \begin{tabularx}{\linewidth}{@{}p{0.12\linewidth}p{0.18\linewidth}"
        r"p{0.14\linewidth}p{0.105\linewidth}XX@{}}",
        r"    \toprule",
        (
            r"    Analysis & Evidential role & Independent units & Reporter status "
            r"& Neural result & Primary behavior result \\"
        ),
        r"    \midrule",
    ]
    role_labels = {
        "allen_vbo:locked": "Primary headline",
        "icms:randomized_n5": "Exploratory, non-headline",
        "teacher:locked": "Procedural, non-headline",
        "icms:absolute_only": "Descriptive, non-headline",
    }
    paper_labels = {
        "allen_vbo:locked": "Allen",
        "icms:randomized_n5": "ICMS randomized",
        "teacher:locked": "Teacher",
        "icms:absolute_only": "ICMS83 absolute",
    }
    for spec in ANALYSIS_SPECS:
        analysis = _analysis(report, spec.analysis_id)
        conjunction = _require_mapping(analysis["conjunction"], f"{spec.analysis_id}.conjunction")
        neural, behavior = _paper_endpoint_summaries(analysis, spec.analysis_id)
        lines.append(
            "    "
            f"{_latex_escape(paper_labels[spec.analysis_id])} & "
            f"{_latex_escape(role_labels[spec.analysis_id])} & "
            f"{_latex_escape(_replication_text(analysis, spec))} & "
            rf"\texttt{{{_latex_escape(conjunction['overall_status'])}}} & "
            f"{_latex_ci(neural[0], label=neural[1])} & "
            f"{_latex_ci(behavior[0], label=behavior[1])} \\\\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabularx}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _artifact_targets(
    site_output: Path,
    figure_directory: Path,
    paper_include: Path,
) -> dict[str, Path]:
    return {
        "site": site_output,
        "unit_skill_png": figure_directory / f"{UNIT_FIGURE_STEM}.png",
        "unit_skill_pdf": figure_directory / f"{UNIT_FIGURE_STEM}.pdf",
        "gate_matrix_png": figure_directory / f"{GATE_FIGURE_STEM}.png",
        "gate_matrix_pdf": figure_directory / f"{GATE_FIGURE_STEM}.pdf",
        "paper_include": paper_include,
    }


def _prepare_target_transaction(targets: Mapping[str, Path]) -> Path:
    canonical: dict[Path, str] = {}
    for name, target in targets.items():
        resolved = target.resolve()
        previous = canonical.get(resolved)
        if previous is not None:
            raise ValueError(
                f"editorial artifact targets overlap: {previous!r} and {name!r} "
                f"both resolve to {resolved}"
            )
        canonical[resolved] = name
        if target.is_symlink():
            raise ValueError(f"refusing to replace symlinked artifact target: {target}")
        if target.exists() and not target.is_file():
            raise ValueError(f"artifact target is not a regular file: {target}")

    for target in targets.values():
        target.parent.mkdir(parents=True, exist_ok=True)
    common_parent = Path(
        os.path.commonpath([str(path.resolve().parent) for path in targets.values()])
    )
    common_parent.mkdir(parents=True, exist_ok=True)
    devices = {target.parent.stat().st_dev for target in targets.values()}
    devices.add(common_parent.stat().st_dev)
    if len(devices) != 1:
        raise ValueError(
            "all editorial targets must share one filesystem for rollback-safe publication"
        )
    return common_parent


def _publish_staged(
    staged: Mapping[str, Path],
    targets: Mapping[str, Path],
    staging: Path,
) -> None:
    backups: dict[str, Path] = {}
    installed: list[str] = []
    try:
        for index, (name, target) in enumerate(targets.items()):
            if not target.exists():
                continue
            backup = staging / f"backup-{index:02d}-{target.name}"
            os.replace(target, backup)
            backups[name] = backup
        for name, target in targets.items():
            os.replace(staged[name], target)
            installed.append(name)
    except BaseException as error:
        rollback_errors: list[str] = []
        for name in reversed(installed):
            target = targets[name]
            try:
                target.unlink(missing_ok=True)
            except OSError as rollback_error:
                rollback_errors.append(f"remove {target}: {rollback_error}")
        for name in reversed(tuple(backups)):
            backup = backups[name]
            target = targets[name]
            try:
                os.replace(backup, target)
            except OSError as rollback_error:
                rollback_errors.append(f"restore {target}: {rollback_error}")
        if rollback_errors:
            raise RuntimeError(
                "editorial artifact publication failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from error
        raise


def build_editorial_artifacts(
    *,
    summary_path: Path,
    completion_path: Path,
    site_output: Path,
    figure_directory: Path,
    paper_include: Path,
    release_label: str,
    generated_at: str,
    protocol_version: str,
) -> dict[str, Path]:
    if not release_label:
        raise ValueError("release_label must be non-empty")
    if not protocol_version:
        raise ValueError("protocol_version must be non-empty")
    try:
        parsed_date = date.fromisoformat(generated_at)
    except ValueError as error:
        raise ValueError("generated_at must be an ISO date (YYYY-MM-DD)") from error
    if parsed_date.isoformat() != generated_at:
        raise ValueError("generated_at must use canonical ISO date form YYYY-MM-DD")

    report, summary_digest = _verified_report_inputs(summary_path, completion_path)
    site_payload = build_site_payload(
        report,
        summary_digest=summary_digest,
        release_label=release_label,
        generated_at=generated_at,
        protocol_version=protocol_version,
    )
    site_text = (
        json.dumps(
            site_payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    paper_text = render_paper_include(report, summary_digest)
    targets = _artifact_targets(site_output, figure_directory, paper_include)

    common_parent = _prepare_target_transaction(targets)
    with tempfile.TemporaryDirectory(
        prefix=".final-editorial-staging-",
        dir=common_parent,
    ) as temporary:
        staging = Path(temporary)
        staged = {
            name: staging / f"{index:02d}-{path.name}"
            for index, (name, path) in enumerate(targets.items())
        }
        _write_text(staged["site"], site_text)
        _write_text(staged["paper_include"], paper_text)
        render_unit_skill_figure(report, staging)
        (staging / f"{UNIT_FIGURE_STEM}.png").replace(staged["unit_skill_png"])
        (staging / f"{UNIT_FIGURE_STEM}.pdf").replace(staged["unit_skill_pdf"])
        render_gate_matrix_figure(report, staging)
        (staging / f"{GATE_FIGURE_STEM}.png").replace(staged["gate_matrix_png"])
        (staging / f"{GATE_FIGURE_STEM}.pdf").replace(staged["gate_matrix_pdf"])
        _publish_staged(staged, targets, staging)
    return targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--completion", type=Path, default=DEFAULT_COMPLETION)
    parser.add_argument("--site-output", type=Path, default=DEFAULT_SITE_OUTPUT)
    parser.add_argument(
        "--figure-directory",
        type=Path,
        default=DEFAULT_FIGURE_DIRECTORY,
    )
    parser.add_argument("--paper-include", type=Path, default=DEFAULT_PAPER_INCLUDE)
    parser.add_argument(
        "--release-label",
        default="Frozen outcome release",
        help="editorial label stored in the site derivative",
    )
    parser.add_argument(
        "--generated-at",
        required=True,
        help="explicit ISO release date; never inferred from the wall clock",
    )
    parser.add_argument(
        "--protocol-version",
        default="1.0.0",
        help="editorial protocol-version label stored in the site derivative",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        paths = build_editorial_artifacts(
            summary_path=args.summary,
            completion_path=args.completion,
            site_output=args.site_output,
            figure_directory=args.figure_directory,
            paper_include=args.paper_include,
            release_label=args.release_label,
            generated_at=args.generated_at,
            protocol_version=args.protocol_version,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"final editorial artifact build failed: {error}") from error
    print(
        json.dumps(
            {
                name: {
                    "path": str(path),
                    "sha256": _sha256(path),
                }
                for name, path in paths.items()
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
