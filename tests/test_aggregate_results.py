from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_aggregate_script() -> ModuleType:
    path = ROOT / "scripts" / "aggregate_results.py"
    spec = importlib.util.spec_from_file_location(
        "cadence_test_aggregate_results",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def aggregate_script() -> ModuleType:
    return _load_aggregate_script()


def _write_metric(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    return path


def test_metric_discovery_skips_hidden_and_quarantine_components(
    aggregate_script: ModuleType,
    tmp_path: Path,
) -> None:
    root = tmp_path / "results"
    visible = {
        _write_metric(root / "allen-vbo/locked-fold-0/metrics.json"),
        _write_metric(root / "icms/loao-ICMS92/metrics.json"),
        _write_metric(root / "teacher-locked/full/locked-seed-00/metrics.json"),
    }
    _write_metric(root / "allen-vbo/.locked-fold-0.interrupted-abc/metrics.json")
    _write_metric(root / "icms/loao-ICMS92-interrupted-stage-abc/metrics.json")
    _write_metric(root / "teacher-locked/full/quarantine/locked-seed-00/metrics.json")
    _write_metric(root / ".scratch/allen-vbo/locked-fold-1/metrics.json")

    assert set(aggregate_script._metric_files([root])) == visible


def test_explicit_hidden_or_quarantine_metric_is_skipped(
    aggregate_script: ModuleType,
    tmp_path: Path,
) -> None:
    hidden = _write_metric(tmp_path / ".locked-fold-0.interrupted-abc/metrics.json")
    quarantine = _write_metric(tmp_path / "quarantine/metrics.json")

    assert aggregate_script._metric_files([hidden, quarantine]) == []


def test_visible_alias_to_hidden_metric_is_skipped(
    aggregate_script: ModuleType,
    tmp_path: Path,
) -> None:
    hidden = _write_metric(tmp_path / ".interrupted-stage-abc/metrics.json")
    visible_alias = tmp_path / "results/allen-vbo/locked-fold-0/metrics.json"
    visible_alias.parent.mkdir(parents=True)
    visible_alias.symlink_to(hidden)

    assert aggregate_script._metric_files([tmp_path / "results"]) == []
