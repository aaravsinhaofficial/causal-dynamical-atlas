from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib.image as mpimg
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_builder() -> ModuleType:
    path = ROOT / "scripts" / "build_final_editorial_artifacts.py"
    spec = importlib.util.spec_from_file_location(
        "cadence_test_build_final_editorial_artifacts",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder() -> ModuleType:
    return _load_builder()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return _digest(path)


def _write_sidecar(path: Path) -> str:
    digest = _digest(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )
    return digest


def _metric_summary(
    estimate: float | None,
    lower: float | None,
    upper: float | None,
    *,
    n: int,
) -> dict[str, Any]:
    return {
        "n": n,
        "estimate": estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": 0.95,
        "bootstrap_repeats": 20000,
        "equal_animal_weight": True,
    }


def _gates(statuses: list[str], overall: str) -> dict[str, Any]:
    return {
        "overall_status": overall,
        "gates": [
            {
                "gate_id": index,
                "criterion": f"synthetic frozen criterion {index}",
                "status": status,
                "details": {
                    "synthetic": True,
                    "gate": index,
                    "estimate": 900 + index,
                },
            }
            for index, status in enumerate(statuses, start=1)
        ],
    }


def _target_row(
    *,
    dataset: str,
    cohort: str,
    unit_id: str,
    metrics: dict[str, float | None],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "cohort": cohort,
        "unit_id": unit_id,
        "animal_id": unit_id,
        "fold": None,
        "world_id": None,
        "randomized_estimand": dataset != "teacher",
        "methods": {"proposed": metrics},
    }


def _inference_row(
    *,
    dataset: str,
    cohort: str,
    unit_id: str,
    metrics: dict[str, float | None],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "cohort": cohort,
        "unit_id": unit_id,
        "animal_id": unit_id,
        "method": "proposed",
        "world_id": unit_id if dataset == "teacher" else None,
        "metrics": metrics,
    }


def _analysis(
    *,
    dataset: str,
    cohort: str,
    replication_unit: str,
    targets: list[dict[str, Any]],
    inference: list[dict[str, Any]],
    conjunction: dict[str, Any],
    method_summaries: dict[str, Any],
    primary_summaries: dict[str, Any],
    gates_evaluable: bool,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "cohort": cohort,
        "n_independent_units": len(inference),
        "n_nested_target_units": len(targets),
        "expected_n": len(inference),
        "cohort_completeness": {
            "complete": True,
            "gates_evaluable": gates_evaluable,
        },
        "replication_unit": replication_unit,
        "method_summaries": method_summaries,
        "primary_summaries": primary_summaries,
        "strongest_baseline_envelope": {"neural": [], "behavior": []},
        "conjunction": conjunction,
        "target_rows": targets,
        "inference_rows": inference,
    }


def _synthetic_summary() -> dict[str, Any]:
    allen_targets = [
        _target_row(
            dataset="allen_vbo",
            cohort="locked",
            unit_id=f"allen-{index}",
            metrics={
                "neural_causal_skill": neural,
                "running_causal_skill": behavior,
            },
        )
        for index, (neural, behavior) in enumerate(((0.4, 0.2), (0.6, 0.4)), start=1)
    ]
    allen_inference = [
        _inference_row(
            dataset="allen_vbo",
            cohort="locked",
            unit_id=row["unit_id"],
            metrics=copy.deepcopy(row["methods"]["proposed"]),
        )
        for row in allen_targets
    ]
    allen = _analysis(
        dataset="allen_vbo",
        cohort="locked",
        replication_unit="target_animal",
        targets=allen_targets,
        inference=allen_inference,
        conjunction=_gates(["PASS"] * 8, "PASS"),
        method_summaries={
            "proposed": {
                # Deliberately differs from the gate-linked summary.
                "neural_causal_skill": _metric_summary(0.91, 0.90, 0.92, n=2),
                "running_causal_skill": _metric_summary(0.81, 0.80, 0.82, n=2),
            }
        },
        primary_summaries={
            "neural_skill": _metric_summary(0.5, 0.4, 0.6, n=2),
            "behavior_skill": _metric_summary(0.3, 0.2, 0.4, n=2),
        },
        gates_evaluable=True,
    )

    icms_targets = [
        _target_row(
            dataset="icms",
            cohort="randomized_n5",
            unit_id=f"icms-{index}",
            metrics={
                "neural_causal_skill_equal_session": neural,
                "behavior_causal_skill_equal_session": behavior,
            },
        )
        for index, (neural, behavior) in enumerate(((0.15, -0.1), (0.35, 0.1)), start=1)
    ]
    icms_inference = [
        _inference_row(
            dataset="icms",
            cohort="randomized_n5",
            unit_id=row["unit_id"],
            metrics=copy.deepcopy(row["methods"]["proposed"]),
        )
        for row in icms_targets
    ]
    icms = _analysis(
        dataset="icms",
        cohort="randomized_n5",
        replication_unit="target_animal",
        targets=icms_targets,
        inference=icms_inference,
        conjunction=_gates(
            [
                "PASS",
                "PASS",
                "FAIL",
                "NOT_EVALUATED",
                "NOT_EVALUATED",
                "FAIL",
                "NOT_EVALUATED",
                "NOT_EVALUATED",
            ],
            "FAIL",
        ),
        method_summaries={
            "proposed": {
                "neural_causal_skill_equal_session": _metric_summary(0.25, 0.15, 0.35, n=2),
                "behavior_causal_skill_equal_session": _metric_summary(0.0, -0.1, 0.1, n=2),
            }
        },
        primary_summaries={
            "neural_skill": _metric_summary(0.25, 0.15, 0.35, n=2),
            "behavior_skill": _metric_summary(0.0, -0.1, 0.1, n=2),
        },
        gates_evaluable=True,
    )

    teacher_target_values = (
        ("teacher-world-1-target-1", 0.1, 0.2),
        ("teacher-world-1-target-2", 0.3, 0.4),
        ("teacher-world-2-target-1", 0.5, 0.6),
        ("teacher-world-2-target-2", 0.7, 0.8),
    )
    teacher_targets = [
        _target_row(
            dataset="teacher",
            cohort="locked",
            unit_id=unit_id,
            metrics={
                "neural_condition_averaged_causal_skill": neural,
                "behavior_condition_averaged_causal_skill": behavior,
            },
        )
        for unit_id, neural, behavior in teacher_target_values
    ]
    teacher_inference = [
        _inference_row(
            dataset="teacher",
            cohort="locked",
            unit_id=f"teacher-world-{index}",
            metrics={
                "neural_condition_averaged_causal_skill": neural,
                "behavior_condition_averaged_causal_skill": behavior,
            },
        )
        for index, (neural, behavior) in enumerate(((0.2, 0.3), (0.6, 0.7)), start=1)
    ]
    teacher = _analysis(
        dataset="teacher",
        cohort="locked",
        replication_unit="teacher_world",
        targets=teacher_targets,
        inference=teacher_inference,
        conjunction=_gates(["NOT_EVALUATED"] * 8, "NOT_EVALUATED"),
        method_summaries={
            "proposed": {
                "neural_condition_averaged_causal_skill": _metric_summary(0.4, 0.2, 0.6, n=2),
                "behavior_condition_averaged_causal_skill": _metric_summary(0.5, 0.3, 0.7, n=2),
            }
        },
        primary_summaries={},
        gates_evaluable=False,
    )

    absolute_targets = [
        _target_row(
            dataset="icms",
            cohort="absolute_only",
            unit_id="ICMS83",
            metrics={
                "absolute_neural_nrmse_equal_session": 0.8,
                "absolute_behavior_nrmse_equal_session": 0.9,
            },
        )
    ]
    absolute = _analysis(
        dataset="icms",
        cohort="absolute_only",
        replication_unit="target_animal",
        targets=absolute_targets,
        inference=[
            _inference_row(
                dataset="icms",
                cohort="absolute_only",
                unit_id="ICMS83",
                metrics=copy.deepcopy(absolute_targets[0]["methods"]["proposed"]),
            )
        ],
        conjunction=_gates(["NOT_EVALUATED"] * 8, "NOT_EVALUATED"),
        method_summaries={
            "proposed": {
                "absolute_neural_nrmse_equal_session": _metric_summary(0.8, None, None, n=1),
                "absolute_behavior_nrmse_equal_session": _metric_summary(0.9, None, None, n=1),
            }
        },
        primary_summaries={},
        gates_evaluable=False,
    )
    return {
        "schema_version": "cadence.reporting.v1",
        "parameters": {"bootstrap_repeats": 20000},
        "analyses": {
            "allen_vbo:locked": allen,
            "icms:randomized_n5": icms,
            "teacher:locked": teacher,
            "icms:absolute_only": absolute,
        },
        "animal_rows": [],
    }


def _write_authenticated_report(
    root: Path,
    summary: dict[str, Any],
) -> tuple[Path, Path, str]:
    summary_path = root / "summary.json"
    digest = _write_json(summary_path, summary)
    _write_sidecar(summary_path)
    completion_path = root / "report.complete.json"
    _write_json(
        completion_path,
        {
            "schema": "cadence.reporting_completion.v1",
            "append_only": True,
            "artifacts": {"summary.json": digest},
            "authenticated_inputs": [],
            "reporter_attestations": {},
        },
    )
    _write_sidecar(completion_path)
    return summary_path, completion_path, digest


def _build(
    builder: ModuleType,
    root: Path,
    summary_path: Path,
    completion_path: Path,
) -> dict[str, Path]:
    return builder.build_editorial_artifacts(
        summary_path=summary_path,
        completion_path=completion_path,
        site_output=root / "site" / "data" / "final-outcome.json",
        figure_directory=root / "paper" / "figures",
        paper_include=root / "paper" / "includes" / "final_outcome.tex",
        release_label="Synthetic frozen outcome",
        generated_at="2026-08-01",
        protocol_version="1.0.0",
    )


def test_builder_maps_authenticated_synthetic_report_without_recomputing_status(
    builder: ModuleType,
    tmp_path: Path,
) -> None:
    summary_path, completion_path, digest = _write_authenticated_report(
        tmp_path / "report",
        _synthetic_summary(),
    )
    paths = _build(builder, tmp_path / "out", summary_path, completion_path)

    assert set(paths) == {
        "site",
        "unit_skill_png",
        "unit_skill_pdf",
        "gate_matrix_png",
        "gate_matrix_pdf",
        "paper_include",
    }
    site = json.loads(paths["site"].read_text(encoding="utf-8"))
    assert site["schema"] == "cadence-site-final-outcome-v1"
    assert site["release"]["sourceSummarySha256"] == digest
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert site["headline"]["status"] == "PASS"

    by_id = {dataset["id"]: dataset for dataset in site["datasets"]}
    assert by_id["allen"]["status"] == "PASS"
    assert by_id["icms-randomized"]["status"] == "FAIL"
    assert by_id["teacher"]["status"] == "NOT_EVALUATED"
    assert by_id["icms-absolute"]["status"] == "NOT_EVALUATED"
    # Gate-linked site intervals come from primary_summaries, not the deliberately
    # different method summary or gate-detail sentinel.
    assert by_id["allen"]["endpoints"][0] == {
        "label": "Neural causal skill",
        "value": 0.5,
        "ciLow": 0.4,
        "ciHigh": 0.6,
        "status": "PASS",
        "note": "Gate 2 input: equal-unit 95% bootstrap interval.",
    }
    assert by_id["icms-randomized"]["endpoints"][1]["status"] == "FAIL"
    assert by_id["teacher"]["endpoints"][0]["value"] == 0.4
    assert by_id["icms-absolute"]["endpoints"][0]["value"] == 0.8
    assert (
        by_id["icms-randomized"]["gates"][3]["evidence"]
        == 'Canonical reporter details: {"estimate":904,"gate":4,"synthetic":true}'
    )

    paper = paths["paper_include"].read_text(encoding="utf-8")
    assert f"Authenticated source summary SHA-256: {digest}" in paper
    assert r"\texttt{PASS}" in paper
    assert r"\texttt{FAIL}" in paper
    assert "2 independent target animals" in paper
    assert "2 independent teacher worlds; 4 nested target units" in paper
    assert r"\ensuremath{0.5\;[0.4,\,0.6]} (causal skill)" in paper
    assert "ICMS83 absolute" in paper

    for key in ("unit_skill_pdf", "gate_matrix_pdf"):
        assert paths[key].read_bytes().startswith(b"%PDF")
        assert paths[key].stat().st_size > 5_000
    for key in ("unit_skill_png", "gate_matrix_png"):
        image = mpimg.imread(paths[key])
        assert image.shape[0] >= 800
        assert image.shape[1] >= 1200


def test_builder_is_byte_reproducible_for_identical_inputs(
    builder: ModuleType,
    tmp_path: Path,
) -> None:
    summary_path, completion_path, _ = _write_authenticated_report(
        tmp_path / "report",
        _synthetic_summary(),
    )
    first = _build(builder, tmp_path / "first", summary_path, completion_path)
    second = _build(builder, tmp_path / "second", summary_path, completion_path)
    assert {name: _digest(path) for name, path in first.items()} == {
        name: _digest(path) for name, path in second.items()
    }


def test_builder_is_byte_reproducible_across_processes(
    tmp_path: Path,
) -> None:
    summary_path, completion_path, _ = _write_authenticated_report(
        tmp_path / "report",
        _synthetic_summary(),
    )

    def run(output: Path, hash_seed: str) -> dict[str, str]:
        mpl_config = output / "matplotlib-config"
        mpl_config.mkdir(parents=True)
        environment = os.environ.copy()
        environment["MPLCONFIGDIR"] = str(mpl_config)
        environment["PYTHONHASHSEED"] = hash_seed
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_final_editorial_artifacts.py"),
                "--summary",
                str(summary_path),
                "--completion",
                str(completion_path),
                "--site-output",
                str(output / "site" / "data" / "final-outcome.json"),
                "--figure-directory",
                str(output / "paper" / "figures"),
                "--paper-include",
                str(output / "paper" / "includes" / "final_outcome.tex"),
                "--release-label",
                "Synthetic frozen outcome",
                "--generated-at",
                "2026-08-01",
                "--protocol-version",
                "1.0.0",
            ],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        payload = json.loads(completed.stdout)
        return {name: artifact["sha256"] for name, artifact in payload.items()}

    assert run(tmp_path / "process-one", "1") == run(tmp_path / "process-two", "8675309")


def test_icms_unit_dots_use_the_reporter_primary_alias_precedence(
    builder: ModuleType,
) -> None:
    report = _synthetic_summary()
    analysis = report["analyses"]["icms:randomized_n5"]
    for index, row in enumerate(analysis["target_rows"], start=1):
        metrics = row["methods"]["proposed"]
        metrics["neural_causal_skill"] = 0.01 * index
        metrics["behavior_causal_skill"] = -0.01 * index
        # These equal-session variants deliberately conflict. The reporter's
        # primary alias order chooses the generic causal-skill fields first.
        metrics["neural_causal_skill_equal_session"] = 90.0 + index
        metrics["behavior_causal_skill_equal_session"] = 80.0 + index

    assert builder._independent_skill_values(analysis, "icms:randomized_n5", "neural") == [
        ("icms-1", 0.01),
        ("icms-2", 0.02),
    ]
    assert builder._independent_skill_values(analysis, "icms:randomized_n5", "behavior") == [
        ("icms-1", -0.01),
        ("icms-2", -0.02),
    ]


def test_publication_rolls_back_existing_outputs_after_replace_failure(
    builder: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    targets = {
        "first": tmp_path / "first.txt",
        "second": tmp_path / "second.txt",
    }
    targets["first"].write_text("old first\n", encoding="utf-8")
    staged = {
        "first": staging / "new-first.txt",
        "second": staging / "new-second.txt",
    }
    staged["first"].write_text("new first\n", encoding="utf-8")
    staged["second"].write_text("new second\n", encoding="utf-8")

    real_replace = os.replace
    calls = 0

    def fail_during_second_install(source: Any, destination: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("synthetic publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(builder.os, "replace", fail_during_second_install)
    with pytest.raises(OSError, match="synthetic publication failure"):
        builder._publish_staged(staged, targets, staging)

    assert targets["first"].read_text(encoding="utf-8") == "old first\n"
    assert not targets["second"].exists()


def test_builder_rejects_overlapping_output_targets(
    builder: ModuleType,
    tmp_path: Path,
) -> None:
    summary_path, completion_path, _ = _write_authenticated_report(
        tmp_path / "report",
        _synthetic_summary(),
    )
    collision = tmp_path / "out" / "collision.json"
    with pytest.raises(ValueError, match="targets overlap"):
        builder.build_editorial_artifacts(
            summary_path=summary_path,
            completion_path=completion_path,
            site_output=collision,
            figure_directory=tmp_path / "out" / "figures",
            paper_include=collision,
            release_label="Synthetic frozen outcome",
            generated_at="2026-08-01",
            protocol_version="1.0.0",
        )


def test_builder_fails_closed_on_sidecar_and_completion_digest_mismatches(
    builder: ModuleType,
    tmp_path: Path,
) -> None:
    summary_path, completion_path, _ = _write_authenticated_report(
        tmp_path / "report",
        _synthetic_summary(),
    )
    summary_path.write_text(
        summary_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid SHA-256 sidecar"):
        _build(builder, tmp_path / "sidecar-out", summary_path, completion_path)

    _write_sidecar(summary_path)
    with pytest.raises(ValueError, match="does not match report.complete.json"):
        _build(builder, tmp_path / "digest-out", summary_path, completion_path)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda summary: summary["analyses"].pop("teacher:locked"),
            "missing required analyses",
        ),
        (
            lambda summary: summary["analyses"]["allen_vbo:locked"]["conjunction"].update(
                {"overall_status": "UNKNOWN"}
            ),
            "must be one of",
        ),
        (
            lambda summary: summary["analyses"]["icms:randomized_n5"].update(
                {"primary_summaries": {}}
            ),
            "gate-evaluable but lacks",
        ),
    ],
)
def test_builder_rejects_malformed_required_analysis_schema(
    builder: ModuleType,
    tmp_path: Path,
    mutate: Any,
    match: str,
) -> None:
    summary = _synthetic_summary()
    mutate(summary)
    summary_path, completion_path, _ = _write_authenticated_report(
        tmp_path / "report",
        summary,
    )
    with pytest.raises(ValueError, match=match):
        _build(builder, tmp_path / "out", summary_path, completion_path)
